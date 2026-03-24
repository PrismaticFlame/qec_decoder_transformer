# model.py - AlphaQubit model with proper per-distance dilations (trans7)
# Identical to trans6 except ScatteringResidualConvBlock dilations are wired in
# from the hyperparameter table (d=3: [1,1,1], d=5: [1,1,2], d>=7: [1,2,4]).
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# ---------------------------
# Helper: LayerNorm wrapper
# ---------------------------
class LN(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


# ============================================================
# Embedding residual block
# ============================================================
class _EmbedResBlock(nn.Module):
    """Single residual block: LayerNorm -> Linear -> GELU -> Linear."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        return x + h


# ============================================================
# Stabilizer Embedding (2-input: measurement, event)
# ============================================================
class StabilizerEmbedding(nn.Module):
    """
    AlphaQubit-style 2-input embedding (scaling/DEM experiment).

    Per Fig 1C: h = proj_meas(m) + proj_event(e) + E_stab(stab_id)
    h = ResBlock_1(h)
    h = ResBlock_2(h)

    No cycle embedding here — per the paper the cycle index embedding belongs
    in the Readout (Fig 1F), not the StabilizerEmbedder.
    No leakage inputs — those are only used in the Sycamore/Pauli+ experiments.
    """

    def __init__(self, num_stab: int, d_model: int = 256):
        super().__init__()
        self.proj_meas = nn.Linear(1, d_model)
        self.proj_event = nn.Linear(1, d_model)
        self.stab_emb = nn.Embedding(num_stab, d_model)
        self.res1 = _EmbedResBlock(d_model)
        self.res2 = _EmbedResBlock(d_model)

    def forward(
        self,
        meas: torch.Tensor,  # (B, S)
        event: torch.Tensor,  # (B, S)
        stab_ids: torch.Tensor,  # (B, S) or (S,)
    ) -> torch.Tensor:
        B, S = meas.shape
        if stab_ids.dim() == 1:
            stab_ids = stab_ids.unsqueeze(0).expand(B, S)

        h = (
            self.proj_meas(meas.unsqueeze(-1))
            + self.proj_event(event.unsqueeze(-1))
            + self.stab_emb(stab_ids)
        )
        h = self.res1(h)
        h = self.res2(h)
        return h


# ============================================================
# Final-round on-basis embedding (separate linear projection)
# ============================================================
class FinalRoundEmbedding(nn.Module):
    """
    Separate embedding for on-basis stabilizers in the final syndrome round.

    Per the paper: a dedicated linear projection (not a full ResNet embedding)
    for on-basis computed stabilizers. Off-basis stabilizers in the final round
    use a single shared learned vector (offbasis_final_emb in AlphaQubitModel).

    h = proj_meas(m) + proj_event(e) + E_stab(stab_id)
    (no ResBlocks, no cycle embedding — matches Fig 1C)
    """

    def __init__(self, num_stab: int, d_model: int = 256):
        super().__init__()
        self.proj_meas = nn.Linear(1, d_model)
        self.proj_event = nn.Linear(1, d_model)
        self.stab_emb = nn.Embedding(num_stab, d_model)

    def forward(
        self,
        meas: torch.Tensor,  # (B, S)
        event: torch.Tensor,  # (B, S)
        stab_ids: torch.Tensor,  # (B, S) or (S,)
    ) -> torch.Tensor:
        B, S = meas.shape
        if stab_ids.dim() == 1:
            stab_ids = stab_ids.unsqueeze(0).expand(B, S)

        return (
            self.proj_meas(meas.unsqueeze(-1))
            + self.proj_event(event.unsqueeze(-1))
            + self.stab_emb(stab_ids)
        )


# ============================================================
# Algorithm 1: AttentionWithBias (single head)
# ============================================================
class AttentionWithBiasHead(nn.Module):
    def __init__(self, d_d: int, d_attn: int, d_mid: int):
        super().__init__()
        self.Wq = nn.Linear(d_d, d_attn, bias=True)
        self.Wk = nn.Linear(d_d, d_attn, bias=True)
        self.Wv = nn.Linear(d_d, d_mid, bias=True)
        self.d_attn = d_attn

    def forward(self, X: torch.Tensor, Bp: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)
        S = torch.matmul(Q, K.transpose(-2, -1)) + Bp
        A = F.softmax(S / math.sqrt(self.d_attn), dim=-1)
        return torch.matmul(A, V), A


# ============================================================
# Algorithm 2: Multi-head AttentionWithBias
# ============================================================
class MHAttentionWithBias(nn.Module):
    def __init__(self, d_d: int, d_attn: int, d_mid: int, db: int, H: int):
        super().__init__()
        self.H = H
        self.d_attn = d_attn
        self.Wb = nn.Linear(db, H, bias=False)
        self.Wq = nn.Linear(d_d, H * d_attn, bias=True)
        self.Wk = nn.Linear(d_d, H * d_attn, bias=True)
        self.Wv = nn.Linear(d_d, H * d_mid, bias=True)
        self.Wo = nn.Linear(H * d_mid, d_d, bias=True)

    def forward(self, X, bias):
        B, L, _ = X.shape
        Bp = self.Wb(bias).permute(0, 3, 1, 2)  # (B, H, S, S)
        Q = self.Wq(X).view(B, L, self.H, self.d_attn).transpose(1, 2)
        K = self.Wk(X).view(B, L, self.H, self.d_attn).transpose(1, 2)
        V = self.Wv(X).view(B, L, self.H, -1).transpose(1, 2)
        Y = F.scaled_dot_product_attention(Q, K, V, attn_mask=Bp)
        return self.Wo(Y.transpose(1, 2).contiguous().view(B, L, -1))


# ============================================================
# Algorithm 3: GatedDenseBlock
# ============================================================
class GatedDenseBlock(nn.Module):
    def __init__(self, d_d: int, w: int = 4):
        super().__init__()
        hidden = w * d_d
        assert hidden % 2 == 0
        self.fc1 = nn.Linear(d_d, hidden, bias=True)
        self.fc2 = nn.Linear(hidden // 2, d_d, bias=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.fc1(X)
        a, g = torch.chunk(Y, 2, dim=-1)
        return self.fc2(F.gelu(a) * g)


# ============================================================
# Algorithm 4: ScatteringResidualConvBlock (with dilations)
# ============================================================
class ScatteringResidualConvBlock(nn.Module):
    """
    Scatter tokens -> 2D grid, apply L dilated 3x3 convs with residual, gather back.

    dilation_list: per-layer dilation values from the hyperparameter table:
        d=3 -> [1,1,1], d=5 -> [1,1,2], d>=7 -> [1,2,4]
    """

    def __init__(
        self,
        d: int,
        d_d: int,
        L_layers: int,
        channels_list: List[int],
        coord_to_index: Dict,
        index_to_coord: List,
        dilation_list: Optional[List[int]] = None,
    ):
        super().__init__()
        self.d = d
        self.d_d = d_d
        # coord_to_index: stab_id -> token_position  (legacy, kept for compat)
        # index_to_coord: token_position -> (i, j) grid position
        self.coord_to_index = coord_to_index
        self.index_to_coord = index_to_coord
        # Precompute stab_id -> (grid_i, grid_j) lookup tensors.
        # num_stab = len(index_to_coord) since coord_to_index maps all stab positions.
        num_stab = len(index_to_coord)
        gi = torch.zeros(num_stab, dtype=torch.long)
        gj = torch.zeros(num_stab, dtype=torch.long)
        for sid, (ci, cj) in enumerate(index_to_coord):
            gi[sid] = ci
            gj[sid] = cj
        self.register_buffer("_grid_i", gi)  # (num_stab,)
        self.register_buffer("_grid_j", gj)  # (num_stab,)
        self.grid_h = int(gi.max().item()) + 1 if num_stab > 0 else d + 1
        self.grid_w = int(gj.max().item()) + 1 if num_stab > 0 else d + 1

        self.P = nn.Parameter(torch.zeros(d_d))
        self.lns = nn.ModuleList([nn.LayerNorm(d_d) for _ in range(L_layers)])
        self.convs3 = nn.ModuleList()
        self.proj1 = nn.ModuleList()

        in_ch = d_d
        for l in range(L_layers):
            out_ch = channels_list[l]
            dil = 1 if dilation_list is None else dilation_list[l]
            # padding = dilation to maintain spatial size
            self.convs3.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dil, dilation=dil)
            )
            self.proj1.append(
                nn.Conv2d(out_ch, d_d, kernel_size=1)
                if out_ch != d_d
                else nn.Identity()
            )
            in_ch = d_d

    def forward(
        self, X: torch.Tensor, stab_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        X        : (B, S, Dd)  — S active stabilizer tokens this cycle
        stab_ids : (S,)        — stabilizer ID for each token position (0..num_stab-1)
                                 If None, assumes token index == stab_id (legacy).
        """
        B, S, Dd = X.shape
        H = self.grid_h
        W = self.grid_w

        grid = X.new_empty((B, H, W, Dd))
        grid[:] = self.P.view(1, 1, 1, Dd)

        if stab_ids is not None:
            gi = self._grid_i[stab_ids]  # (S,)
            gj = self._grid_j[stab_ids]  # (S,)
        else:
            gi = self._grid_i[:S]
            gj = self._grid_j[:S]

        # Scatter: place all S tokens onto the 2D grid in one operation
        grid[:, gi, gj, :] = X

        Y = grid.permute(0, 3, 1, 2).contiguous()  # (B, Dd, H, W)

        for l in range(len(self.convs3)):
            Yn = self.lns[l](Y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            Yn = F.gelu(self.convs3[l](Yn))
            Yn = self.proj1[l](Yn)
            Y = Y + Yn

        Y_hw = Y.permute(0, 2, 3, 1).contiguous()

        # Gather: read all S token positions from the grid in one operation
        Z = Y_hw[:, gi, gj, :]
        return Z


# ============================================================
# Algorithm 5: RNNCore
# ============================================================
class RNNCore(nn.Module):
    def __init__(
        self,
        d_d: int,
        d_attn: int,
        d_mid: int,
        db: int,
        H: int,
        n_layers: int,
        conv_blocks: nn.ModuleList,
        widen: int = 4,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.ln1 = nn.ModuleList([LN(d_d) for _ in range(n_layers)])
        self.attn = nn.ModuleList(
            [MHAttentionWithBias(d_d, d_attn, d_mid, db, H) for _ in range(n_layers)]
        )
        self.ln2 = nn.ModuleList([LN(d_d) for _ in range(n_layers)])
        self.ffn = nn.ModuleList(
            [GatedDenseBlock(d_d, w=widen) for _ in range(n_layers)]
        )
        self.conv_blocks = conv_blocks

    def forward(
        self,
        X: torch.Tensor,
        S: torch.Tensor,
        Bbias: torch.Tensor,
        stab_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        X = (X + S) / math.sqrt(2.0)
        for l in range(self.n_layers):
            Xn = self.ln1[l](X)
            X = X + self.attn[l](Xn, Bbias)
            Xn = self.ln2[l](X)
            X = X + self.ffn[l](Xn)
            X = self.conv_blocks[l](X, stab_ids=stab_ids)
        return X


# ============================================================
# Next-stabilizer predictor head (auxiliary task)
# ============================================================
class NextStabPredictor(nn.Module):
    def __init__(self, d_d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_d, d_d // 2),
            nn.GELU(),
            nn.Linear(d_d // 2, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X).squeeze(-1)


# ============================================================
# Readout Residual Block
# ============================================================
class ReadoutResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(x)))


# ============================================================
# ReadoutResNet
# ============================================================
class ReadoutResNet(nn.Module):
    """
    Scatter (B,S,D) -> 2D grid -> Conv2d(stride=2) -> dim_reduce ->
    directional mean-pool -> per-position ResNet -> averaged logit.
    """

    def __init__(
        self,
        d_model: int,
        readout_dim: int = 48,
        num_layers: int = 16,
        num_cycles: int = 26,
        distance: int = 3,
        coord_to_index: Optional[dict] = None,
        index_to_coord: Optional[List] = None,
        basis: str = "z",
    ):
        super().__init__()
        self.d_model = d_model
        self.readout_dim = readout_dim
        self.distance = distance
        self.coord_to_index = coord_to_index or {}
        self.basis = basis
        # Per Fig 1F: "Cycle k n → Embed" fed into the line mean pool stage.
        self.cycle_emb = nn.Embedding(num_cycles, readout_dim)

        # Precompute stab_id -> (grid_i, grid_j) lookup tensors from index_to_coord.
        # index_to_coord[stab_id] = (i, j) — one entry per stabilizer.
        itc = index_to_coord or []
        num_stab = len(itc)
        gi = torch.zeros(num_stab, dtype=torch.long)
        gj = torch.zeros(num_stab, dtype=torch.long)
        for sid, (ci, cj) in enumerate(itc):
            gi[sid] = ci
            gj[sid] = cj
        self.register_buffer("_grid_i", gi)
        self.register_buffer("_grid_j", gj)
        self.grid_h = int(gi.max().item()) + 1 if num_stab > 0 else distance + 1
        self.grid_w = int(gj.max().item()) + 1 if num_stab > 0 else distance + 1

        self.P = nn.Parameter(torch.zeros(d_model))
        self.spatial_conv = nn.Conv2d(
            d_model, d_model, kernel_size=2, stride=2, padding=0
        )
        self.dim_reduce = nn.Conv2d(d_model, readout_dim, kernel_size=1)
        self.res_blocks = nn.ModuleList(
            [ReadoutResidualBlock(readout_dim) for _ in range(num_layers)]
        )
        self.out_linear = nn.Linear(readout_dim, 1)

    def forward(
        self,
        X: torch.Tensor,
        basis_idx: Optional[torch.Tensor] = None,
        stab_ids: Optional[torch.Tensor] = None,
        cycle_n: Optional[int] = None,
    ) -> torch.Tensor:
        """
        X         : (B, S, D)
        basis_idx : (B,) int tensor with 0=X, 1=Z. If None, falls back to self.basis.
        stab_ids  : (S,) stabilizer ID for each token. Uses precomputed grid lookup.
                    If None, assumes token index == stab_id (legacy).
        cycle_n   : int, the final cycle index (T-1). Used for the cycle embedding
                    added at the line mean pool stage per Fig 1F.
        """
        B, S, D = X.shape
        H = self.grid_h
        W = self.grid_w

        grid = X.new_empty((B, H, W, D))
        grid[:] = self.P.view(1, 1, 1, D)
        if stab_ids is not None:
            gi = self._grid_i[stab_ids]
            gj = self._grid_j[stab_ids]
        else:
            gi = self._grid_i[:S]
            gj = self._grid_j[:S]

        # Scatter: place all S tokens onto the 2D grid in one operation
        grid[:, gi, gj, :] = X

        grid = grid.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
        x = F.gelu(self.spatial_conv(grid))
        x = F.gelu(self.dim_reduce(x))  # (B, readout_dim, H', W')

        # Directional pool perpendicular to logical observable.
        # basis_idx=0 → X basis → pool over W columns (H' positions remain)
        # basis_idx=1 → Z basis → pool over H rows (W' positions remain)
        if basis_idx is None:
            # Fall back to fixed basis set at construction time
            if self.basis.lower() == "x":
                x = x.mean(dim=3)
            else:
                x = x.mean(dim=2)
        else:
            # Per-sample pooling to handle mixed-basis batches
            x_pool_x = x.mean(dim=3)  # (B, readout_dim, H')
            x_pool_z = x.mean(dim=2)  # (B, readout_dim, W')
            basis_b = basis_idx.view(B).to(x.device)
            mask = (basis_b == 0).view(B, 1, 1)  # True → X-basis, False → Z-basis
            x = torch.where(mask, x_pool_x, x_pool_z)

        K = x.shape[2]
        # Fig 1F: add cycle embedding ("Cycle k n → Embed") before the ResNet.
        if cycle_n is not None:
            cn = torch.tensor(cycle_n, dtype=torch.long, device=x.device)
            cyc_e = self.cycle_emb(cn)  # (readout_dim,)
            x = x + cyc_e.view(1, self.readout_dim, 1)  # broadcast over B and K

        x = x.permute(0, 2, 1).contiguous().reshape(B * K, self.readout_dim)

        for block in self.res_blocks:
            x = block(x)

        logits = self.out_linear(x).squeeze(-1).view(B, K).mean(dim=1)
        return logits


# ============================================================
# Full AlphaQubit model (trans7)
# ============================================================
class AlphaQubitModel(nn.Module):
    """
    AlphaQubit model following the paper pseudocode.
    Matches trans6 structurally but with dilations properly wired in.

    Forward input (batch dict):
        syndrome        (B, L)  detection events
        measurements    (B, L)  stabilizer measurements (optional, falls back to cumXOR)
        stab_id         (L,)    stabilizer index per detector
        cycle_id        (L,)    cycle index per detector
        stab_type       (num_stab,) 1=on-basis, 0=off-basis
        stab_xy         (S, 2)  stabilizer xy coordinates (for bias provider)
        true_stabs      (B,T-1,S)  next-stab targets (optional, for aux loss)
        token_mask      (B,T-1,S)  mask for next-stab loss (optional)
        cycle_index     (T, S)  precomputed per-cycle index
        cycle_pad_mask  (T, S)  True = real token
        (no leakage inputs — scaling/DEM experiment uses 2-input embedding only)

    Forward output dict:
        logical_logits  (B,)
        pred_stabs      (B, T-1, S)  if use_next_stab
    """

    def __init__(
        self,
        *,
        num_stab: int,
        num_cycles: int,
        d_model: int,
        d_attn: int,
        d_mid: int,
        db: int,
        H: int,
        n_layers: int,
        widen: int,
        conv_blocks: nn.ModuleList,
        bias_provider: nn.Module,
        use_next_stab: bool = True,
        readout_dim: int = 48,
        readout_resnet_layers: int = 16,
        distance: int = 3,
        coord_to_index: Optional[dict] = None,
        index_to_coord: Optional[List] = None,
        basis: str = "x",
        use_grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.embed = StabilizerEmbedding(num_stab=num_stab, d_model=d_model)
        self.final_proj = FinalRoundEmbedding(num_stab=num_stab, d_model=d_model)
        self.offbasis_final_emb = nn.Parameter(torch.zeros(d_model))

        self.core = RNNCore(
            d_model, d_attn, d_mid, db, H, n_layers, conv_blocks, widen=widen
        )

        self.readout_ln = nn.LayerNorm(d_model)
        self.readout = ReadoutResNet(
            d_model=d_model,
            readout_dim=readout_dim,
            num_layers=readout_resnet_layers,
            num_cycles=num_cycles,
            distance=distance,
            coord_to_index=coord_to_index or {},
            index_to_coord=index_to_coord or [],
            basis=basis,
        )

        self.use_next_stab = use_next_stab
        self.next_head = NextStabPredictor(d_model) if use_next_stab else None
        self.bias_provider = bias_provider
        self.d_model = d_model
        self.use_grad_checkpoint = use_grad_checkpoint

        self._cycle_index: Optional[torch.Tensor] = None
        self._cycle_pad_mask: Optional[torch.Tensor] = None
        self._T: Optional[int] = None
        self._S: Optional[int] = None
        self._L: Optional[int] = None

    @staticmethod
    def _build_cycle_index(
        stab_id: torch.Tensor, cycle_id: torch.Tensor
    ) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
        L = cycle_id.numel()
        T = int(cycle_id.max().item()) + 1
        indices_per_t: List[torch.Tensor] = []
        S_max = 0
        for t in range(T):
            idx_t = torch.nonzero(cycle_id == t, as_tuple=False).view(-1)
            if idx_t.numel() > 0:
                idx_t = idx_t[torch.argsort(stab_id[idx_t])]
            indices_per_t.append(idx_t)
            S_max = max(S_max, idx_t.numel())

        S = int(S_max)
        if S <= 0:
            z = torch.zeros((T, 0), dtype=torch.long, device=cycle_id.device)
            return T, S, z, z.bool()

        padded, masks = [], []
        for idx_t in indices_per_t:
            n = idx_t.numel()
            if n < S:
                pad = torch.zeros(S - n, dtype=torch.long, device=cycle_id.device)
                padded.append(torch.cat([idx_t, pad]))
                masks.append(
                    torch.cat(
                        [
                            torch.ones(n, dtype=torch.bool, device=cycle_id.device),
                            torch.zeros(
                                S - n, dtype=torch.bool, device=cycle_id.device
                            ),
                        ]
                    )
                )
            else:
                padded.append(idx_t[:S])
                masks.append(torch.ones(S, dtype=torch.bool, device=cycle_id.device))

        return T, S, torch.stack(padded).long(), torch.stack(masks)

    def _ensure_cycle_index(
        self, *, stab_id: torch.Tensor, cycle_id: torch.Tensor, syndrome: torch.Tensor
    ) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
        L = int(cycle_id.numel())
        T_now = int(cycle_id.max().item()) + 1
        need_rebuild = (
            self._cycle_index is None
            or self._L != L
            or self._T != T_now
            or (
                self._cycle_index is not None
                and self._cycle_index.device != syndrome.device
            )
        )
        if need_rebuild:
            T, S, idx, pmask = self._build_cycle_index(stab_id, cycle_id)
            self._cycle_index = idx.to(syndrome.device)
            self._cycle_pad_mask = pmask.to(syndrome.device)
            self._T, self._S, self._L = T, S, L
        return self._T, self._S, self._cycle_index, self._cycle_pad_mask

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        syndrome = batch["syndrome"]
        stab_id = batch["stab_id"]
        cycle_id = batch["cycle_id"]

        if syndrome.dim() != 2:
            raise ValueError(f"syndrome must be (B,L), got {syndrome.shape}")
        B, L = syndrome.shape

        if stab_id.dim() == 2:
            stab_id = stab_id[0]
        if cycle_id.dim() == 2:
            cycle_id = cycle_id[0]

        stab_id = stab_id.to(syndrome.device).long().view(-1)
        cycle_id = cycle_id.to(syndrome.device).long().view(-1)

        # Use precomputed cycle_index/cycle_pad_mask from the batch (built by SyndromeDataset).
        # All samples in a batch share the same layout (GroupedBatchSampler guarantees same D),
        # so taking [0] from the batch dimension is correct.
        if "cycle_index" in batch and "cycle_pad_mask" in batch:
            idx = batch["cycle_index"]
            pad_mask = batch["cycle_pad_mask"]
            if idx.dim() == 3:
                idx = idx[0]
                pad_mask = pad_mask[0]
            idx = idx.to(syndrome.device).long()
            pad_mask = pad_mask.to(syndrome.device)
            T, S = idx.shape[0], idx.shape[1]
        else:
            T, S, idx, pad_mask = self._ensure_cycle_index(
                stab_id=stab_id, cycle_id=cycle_id, syndrome=syndrome
            )
            idx = idx.long()

        # Gather (B, T, S) arrays for the 4 embedding inputs
        idx_flat = idx.view(-1)
        idx_flat_expand = idx_flat.unsqueeze(0).expand(B, -1)

        event_ts = torch.gather(syndrome.float(), 1, idx_flat_expand).view(B, T, S)

        measurements = batch.get("measurements")
        if (
            measurements is not None
            and measurements.dim() == 2
            and measurements.shape[1] == L
        ):
            meas_ts = torch.gather(measurements.float(), 1, idx_flat_expand).view(
                B, T, S
            )
        elif measurements is not None:
            meas_ts = measurements.float()
        else:
            meas_ts = torch.cumsum(event_ts, dim=1) % 2

        stab_ts = stab_id[idx].unsqueeze(0).expand(B, T, S)

        # 2-input embedding for all cycles (meas + event + stab_id; no cycle or leakage)
        # Per Fig 1C: Index i (stab_id) is embedded; cycle index goes to Readout (Fig 1F).
        S_seq = self.embed(
            meas_ts.reshape(B * T, S),
            event_ts.reshape(B * T, S),
            stab_ts.reshape(B * T, S),
        ).reshape(B, T, S, -1)
        D = S_seq.size(-1)

        # Final-round treatment: separate linear projection for on-basis stabs,
        # single learned vector for off-basis stabs.
        stab_type = batch.get("stab_type")
        if stab_type is not None and T > 0:
            if stab_type.dim() == 2:
                stab_type = stab_type[0]
            stab_type = stab_type.to(syndrome.device)

            final_emb = self.final_proj(
                meas_ts[:, -1],
                event_ts[:, -1],
                stab_ts[:, -1],
            )  # (B, S, D)

            stab_ids_final = stab_id[idx[-1]]
            on_basis_mask = stab_type[stab_ids_final].bool()  # (S,)
            off_basis_emb = self.offbasis_final_emb.unsqueeze(0).expand(B, S, D)
            on_mask = on_basis_mask.unsqueeze(0).unsqueeze(-1).expand(B, S, D)
            S_seq[:, -1] = torch.where(on_mask, final_emb, off_basis_emb)

        # Zero out padded positions
        pad_mask_btsd = pad_mask.unsqueeze(0).unsqueeze(-1).expand(B, T, S, D)
        S_seq = S_seq * pad_mask_btsd.float()

        # Recurrent core
        X = S_seq.new_zeros((B, S, D))
        pred_list: List[torch.Tensor] = []

        for t in range(T):
            token_t = S_seq[:, t]

            # stab_ids for this cycle: the stabilizer ID of each of the S token slots
            stab_ids_t = stab_id[idx[t]]  # (S,) — stab ID for each token in cycle t

            # Call bias provider (supports cycle-aware providers)
            fwd = self.bias_provider.forward
            if "cycle" in fwd.__code__.co_varnames:
                Bbias = self.bias_provider(batch, S=S, cycle=t)
            else:
                Bbias = self.bias_provider(batch, S=S)

            if self.use_grad_checkpoint and self.training:
                # Checkpoint recomputes activations on backward to save memory.
                # stab_ids_t is not a tensor with grad, pass via closure.
                _sids = stab_ids_t
                X = grad_checkpoint(
                    lambda _X, _tok, _B: self.core(_X, _tok, _B, stab_ids=_sids),
                    X,
                    token_t,
                    Bbias,
                    use_reentrant=False,
                )
            else:
                X = self.core(X, token_t, Bbias, stab_ids=stab_ids_t)

            if self.use_next_stab and t < T - 1:
                pred_list.append(self.next_head(X))

        # Readout — pass basis_idx, last-cycle stab_ids, and cycle index (T-1) per Fig 1F
        Xp = self.readout_ln(X)
        basis_idx = batch.get("basis_idx")
        # stab_ids_t is still set to the last cycle's stab IDs from the loop above
        logical_logits = self.readout(
            Xp, basis_idx=basis_idx, stab_ids=stab_ids_t, cycle_n=T - 1
        )

        out: Dict[str, torch.Tensor] = {"logical_logits": logical_logits}

        if self.use_next_stab:
            if T <= 1:
                out["pred_stabs"] = syndrome.new_zeros((B, 0, S))
            else:
                out["pred_stabs"] = torch.stack(pred_list, dim=1)

        return out
