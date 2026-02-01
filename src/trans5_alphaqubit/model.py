# model.py - Single-basis AlphaQubit-like model with ReadoutResNet
# Modified from trans4_alphaqubit/model.py:
#   - ReadoutResNet replaces simple Linear classifier
#   - Scatter -> 2x2 Conv -> dim reduction -> mean pool -> residual MLP -> logit
from __future__ import annotations

import sys
from pathlib import Path

# Import shared components from trans3_alphaqubit
sys.path.insert(0, str(Path(__file__).parent.parent / "trans3_alphaqubit"))
from utils import ManhattanDistanceBias

import math
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Helper: (B, L, D) LayerNorm
# ---------------------------
class LN(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


# ============================================================
# Stabilizer Embedding (hard/soft)
# ============================================================
class StabilizerEmbedding(nn.Module):
    """
    S_ni = E_stab(i) + E_cycle(n) + E_value(s_ni)
    - hard: syndrome is int/long -> val_emb
    - soft: syndrome is float -> analog_proj
    """

    def __init__(self, num_stab: int, num_cycles: int, d_model: int = 256):
        super().__init__()
        self.stab_emb = nn.Embedding(num_stab, d_model)
        self.cycle_emb = nn.Embedding(num_cycles, d_model)
        self.val_emb = nn.Embedding(2, d_model)
        self.analog_proj = nn.Linear(1, d_model)

    def forward(self, syndrome: torch.Tensor, stab_id: torch.Tensor, cycle_id: torch.Tensor) -> torch.Tensor:
        B, L = syndrome.shape

        if stab_id.dim() == 1:
            stab_id = stab_id.unsqueeze(0).expand(B, L)
        elif stab_id.dim() == 2 and stab_id.size(0) == 1 and B > 1:
            stab_id = stab_id.expand(B, L)

        if cycle_id.dim() == 1:
            cycle_id = cycle_id.unsqueeze(0).expand(B, L)
        elif cycle_id.dim() == 2 and cycle_id.size(0) == 1 and B > 1:
            cycle_id = cycle_id.expand(B, L)

        e_stab = self.stab_emb(stab_id)
        e_cycle = self.cycle_emb(cycle_id)

        if syndrome.is_floating_point():
            e_val = self.analog_proj(syndrome.unsqueeze(-1))
        else:
            e_val = self.val_emb(syndrome.long())

        return e_stab + e_cycle + e_val


# ============================================================
# Algorithm 1: AttentionWithBias (single head)
# ============================================================
class AttentionWithBiasHead(nn.Module):
    def __init__(self, d_d: int, d_attn: int, d_mid: int):
        super().__init__()
        self.Wq = nn.Linear(d_d, d_attn, bias=True)
        self.Wk = nn.Linear(d_d, d_attn, bias=True)
        self.Wv = nn.Linear(d_d, d_mid,  bias=True)
        self.d_attn = d_attn

    def forward(self, X: torch.Tensor, Bp: torch.Tensor) -> torch.Tensor:
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        S = torch.matmul(Q, K.transpose(-2, -1))
        S = S + Bp

        A = F.softmax(S / math.sqrt(self.d_attn), dim=-1)
        out = torch.matmul(A, V)
        return out, A


# ============================================================
# Algorithm 2: Multi-head AttentionWithBias
# ============================================================
class MHAttentionWithBias(nn.Module):
    def __init__(self, d_d: int, d_attn: int, d_mid: int, db: int, H: int):
        super().__init__()
        self.H = H
        self.Wb = nn.Linear(db, H, bias=False)
        self.heads = nn.ModuleList([AttentionWithBiasHead(d_d, d_attn, d_mid) for _ in range(H)])
        self.Wo = nn.Linear(H * d_mid, d_d, bias=True)

    def forward(self, X: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        Bsz, L, _, db = bias.shape

        Bprime = self.Wb(bias)
        Bprime = Bprime.permute(0, 3, 1, 2)

        Ys = []
        for h in range(self.H):
            out_h, _ = self.heads[h](X, Bprime[:, h, :, :])
            Ys.append(out_h)
        Y = torch.cat(Ys, dim=-1)
        return self.Wo(Y)


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
        Y = F.gelu(a) * g
        return self.fc2(Y)


# ============================================================
# Algorithm 4: ScatteringResidualConvBlock
# ============================================================
class ScatteringResidualConvBlock(nn.Module):
    def __init__(
        self,
        d: int,
        d_d: int,
        L_layers: int,
        channels_list,
        coord_to_index,
        index_to_coord,
        dilation_list=None
    ):
        super().__init__()
        self.d = d
        self.d_d = d_d
        self.coord_to_index = coord_to_index
        self.index_to_coord = index_to_coord

        self.P = nn.Parameter(torch.zeros(d_d))

        self.lns = nn.ModuleList([nn.LayerNorm(d_d) for _ in range(L_layers)])
        self.convs3 = nn.ModuleList()
        self.proj1 = nn.ModuleList()

        in_ch = d_d
        for l in range(L_layers):
            out_ch = channels_list[l]
            dil = 1 if dilation_list is None else dilation_list[l]
            self.convs3.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dil, dilation=dil))
            self.proj1.append(nn.Conv2d(out_ch, d_d, kernel_size=1) if out_ch != d_d else nn.Identity())
            in_ch = d_d

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, L, Dd = X.shape
        H = W = self.d + 1

        grid = X.new_empty((B, H, W, Dd))
        grid[:] = self.P.view(1, 1, 1, Dd)
        for (i, j), idx in self.coord_to_index.items():
            grid[:, i, j, :] = X[:, idx, :]

        Y = grid.permute(0, 3, 1, 2).contiguous()

        for l in range(len(self.convs3)):
            Yn = Y.permute(0, 2, 3, 1)
            Yn = self.lns[l](Yn)
            Yn = Yn.permute(0, 3, 1, 2)

            Yn = self.convs3[l](Yn)
            Yn = F.gelu(Yn)
            Yn = self.proj1[l](Yn)
            Y = Y + Yn

        Y_hw = Y.permute(0, 2, 3, 1).contiguous()
        Z = X.new_empty((B, L, Dd))
        for idx, (i, j) in enumerate(self.index_to_coord):
            Z[:, idx, :] = Y_hw[:, i, j, :]
        return Z


# ============================================================
# Algorithm 5: RNNCore
# ============================================================
class RNNCore(nn.Module):
    def __init__(
        self,
        d_d: int, d_attn: int, d_mid: int, db: int, H: int, n_layers: int,
        conv_block: ScatteringResidualConvBlock,
        widen: int = 4
    ):
        super().__init__()
        self.n_layers = n_layers
        self.ln1 = nn.ModuleList([LN(d_d) for _ in range(n_layers)])
        self.attn = nn.ModuleList([MHAttentionWithBias(d_d, d_attn, d_mid, db, H) for _ in range(n_layers)])
        self.ln2 = nn.ModuleList([LN(d_d) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([GatedDenseBlock(d_d, w=widen) for _ in range(n_layers)])
        self.conv_block = conv_block

    def forward(self, X: torch.Tensor, S: torch.Tensor, Bbias: torch.Tensor) -> torch.Tensor:
        X = (X + S) / math.sqrt(2.0)
        for l in range(self.n_layers):
            Xn = self.ln1[l](X)
            X = X + self.attn[l](Xn, Bbias)
            Xn = self.ln2[l](X)
            X = X + self.ffn[l](Xn)
            X = self.conv_block(X)
        return X


# ============================================================
# Next-stabilizer predictor head (optional auxiliary task)
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
    """Single residual block: Linear -> GELU -> Linear + skip connection."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x + residual


# ============================================================
# ReadoutResNet: AlphaQubit-paper readout network
# ============================================================
class ReadoutResNet(nn.Module):
    """
    AlphaQubit-paper readout network.

    Pipeline:
      1. Scatter (B, S, D) -> 2D grid (B, D, H, W) using coord_to_index
      2. Conv2d(D, D, kernel_size=2, stride=2) for spatial reduction
      3. Project D -> readout_dim via 1x1 conv
      4. Global mean pool over spatial dims -> (B, readout_dim)
      5. num_layers residual blocks (Linear->GELU->Linear + skip)
      6. Final Linear(readout_dim, 1) -> logit
    """

    def __init__(
        self,
        d_model: int,
        readout_dim: int = 48,
        num_layers: int = 16,
        distance: int = 3,
        coord_to_index: Optional[dict] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.readout_dim = readout_dim
        self.distance = distance
        self.coord_to_index = coord_to_index or {}

        # Learnable padding value for empty grid cells
        self.P = nn.Parameter(torch.zeros(d_model))

        # 2x2 conv with stride 2 for spatial reduction
        self.spatial_conv = nn.Conv2d(d_model, d_model, kernel_size=2, stride=2, padding=0)

        # Dimensionality reduction: d_model -> readout_dim
        self.dim_reduce = nn.Conv2d(d_model, readout_dim, kernel_size=1)

        # Residual MLP blocks
        self.res_blocks = nn.ModuleList([
            ReadoutResidualBlock(readout_dim)
            for _ in range(num_layers)
        ])

        # Final classifier
        self.out_linear = nn.Linear(readout_dim, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (B, S, D) final hidden state from transformer core (after LayerNorm)

        Returns:
            logits: (B,) single logit per sample
        """
        B, S, D = X.shape
        H = W = self.distance + 1

        # Step 1: Scatter to 2D grid
        grid = X.new_empty((B, H, W, D))
        grid[:] = self.P.view(1, 1, 1, D)
        for (i, j), idx in self.coord_to_index.items():
            grid[:, i, j, :] = X[:, idx, :]

        # (B, H, W, D) -> (B, D, H, W) for conv
        grid = grid.permute(0, 3, 1, 2).contiguous()

        # Step 2: 2x2 conv spatial reduction
        x = self.spatial_conv(grid)  # (B, d_model, H//2, W//2)
        x = F.gelu(x)

        # Step 3: Dimensionality reduction
        x = self.dim_reduce(x)  # (B, readout_dim, H//2, W//2)
        x = F.gelu(x)

        # Step 4: Global mean pool
        x = x.mean(dim=[2, 3])  # (B, readout_dim)

        # Step 5: Residual MLP blocks
        for block in self.res_blocks:
            x = block(x)

        # Step 6: Final linear
        logits = self.out_linear(x).squeeze(-1)  # (B,)
        return logits


# ============================================================
# Single-basis AlphaQubit-like model with ReadoutResNet
# ============================================================
class AlphaQubitLikeModel(nn.Module):
    """
    Single-basis model following the AlphaQubit paper approach.
    Trains on one basis (X or Z) and predicts a single logical error.

    Changes from trans4:
      - ReadoutResNet replaces Linear classifier
      - Accepts readout_dim, readout_resnet_layers, distance, coord_to_index

    batch from dataset:
      syndrome:        (B,L)
      stab_id:         (L,)
      cycle_id:        (L,)
      logical_labels:  (B,) or (B,1)
      true_stabs:      (B,T-1,S)       # optional (for next-stab loss)
      token_mask:      (B,T-1,S)       # optional (mask leakage/pad/terminal)
    outputs:
      logical_logits:  (B,)            # single logical error prediction
      pred_stabs:      (B,T-1,S)       # optional if use_next_stab
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
        conv_block,  # ScatteringResidualConvBlock
        bias_provider: nn.Module,  # returns bias (B,S,S,db)
        use_next_stab: bool = True,
        # ReadoutResNet parameters
        readout_dim: int = 48,
        readout_resnet_layers: int = 16,
        distance: int = 3,
        coord_to_index: Optional[dict] = None,
    ):
        super().__init__()
        self.embed = StabilizerEmbedding(num_stab=num_stab, num_cycles=num_cycles, d_model=d_model)
        self.core = RNNCore(d_model, d_attn, d_mid, db, H, n_layers, conv_block, widen=widen)

        self.readout_ln = nn.LayerNorm(d_model)
        # ReadoutResNet: scatter -> conv -> reduce -> pool -> ResNet -> logit
        self.readout = ReadoutResNet(
            d_model=d_model,
            readout_dim=readout_dim,
            num_layers=readout_resnet_layers,
            distance=distance,
            coord_to_index=coord_to_index or {},
        )

        self.use_next_stab = use_next_stab
        self.next_head = NextStabPredictor(d_model) if use_next_stab else None

        self.bias_provider = bias_provider

        # cache indices (rebuilt when layout changes)
        self._cycle_index: Optional[torch.Tensor] = None
        self._cycle_pad_mask: Optional[torch.Tensor] = None
        self._T: Optional[int] = None
        self._S: Optional[int] = None
        self._L: Optional[int] = None

    @staticmethod
    def _build_cycle_index(stab_id: torch.Tensor, cycle_id: torch.Tensor) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
        """Pad to S_max instead of truncating to S_min. Returns (T, S, indices, pad_mask)."""
        if stab_id.dim() != 1 or cycle_id.dim() != 1:
            raise ValueError(f"stab_id and cycle_id must be 1D, got {stab_id.shape}, {cycle_id.shape}")
        if stab_id.numel() != cycle_id.numel():
            raise ValueError(f"stab_id and cycle_id must have same length, got {stab_id.numel()} vs {cycle_id.numel()}")

        L = cycle_id.numel()
        T = int(cycle_id.max().item()) + 1
        if T <= 0:
            raise ValueError(f"Invalid T computed from cycle_id.max(): T={T}")

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
            indices = torch.zeros((T, 0), dtype=torch.long, device=cycle_id.device)
            pad_mask = torch.zeros((T, 0), dtype=torch.bool, device=cycle_id.device)
            return T, S, indices, pad_mask

        padded = []
        masks = []
        for idx_t in indices_per_t:
            n = idx_t.numel()
            if n < S:
                pad = torch.zeros(S - n, dtype=torch.long, device=cycle_id.device)
                padded.append(torch.cat([idx_t, pad], dim=0))
                masks.append(torch.cat([
                    torch.ones(n, dtype=torch.bool, device=cycle_id.device),
                    torch.zeros(S - n, dtype=torch.bool, device=cycle_id.device),
                ]))
            else:
                padded.append(idx_t[:S])
                masks.append(torch.ones(S, dtype=torch.bool, device=cycle_id.device))

        indices = torch.stack(padded, dim=0).long()
        pad_mask = torch.stack(masks, dim=0)
        return T, S, indices, pad_mask

    def _ensure_cycle_index(
        self,
        *,
        stab_id: torch.Tensor,
        cycle_id: torch.Tensor,
        syndrome: torch.Tensor,
    ) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
        L = int(cycle_id.numel())
        T_now = int(cycle_id.max().item()) + 1

        need_rebuild = (
            self._cycle_index is None
            or self._T is None
            or self._S is None
            or self._L is None
            or self._L != L
            or self._T != T_now
        )

        if self._cycle_index is not None and self._cycle_index.device != syndrome.device:
            need_rebuild = True

        if need_rebuild:
            T, S, idx, pmask = self._build_cycle_index(stab_id, cycle_id)
            self._cycle_index = idx.to(device=syndrome.device)
            self._cycle_pad_mask = pmask.to(device=syndrome.device)
            self._T, self._S, self._L = T, S, L

        assert self._cycle_index is not None
        assert self._cycle_pad_mask is not None
        assert self._T is not None and self._S is not None

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

        stab_id = stab_id.to(device=syndrome.device).long().view(-1)
        cycle_id = cycle_id.to(device=syndrome.device).long().view(-1)

        if stab_id.numel() != L or cycle_id.numel() != L:
            raise ValueError(
                f"stab_id/cycle_id must have length L={L}, got {stab_id.numel()} and {cycle_id.numel()}"
            )

        T, S, idx, pad_mask = self._ensure_cycle_index(stab_id=stab_id, cycle_id=cycle_id, syndrome=syndrome)

        if S == 0 or T == 0:
            raise ValueError(
                f"Invalid per-cycle layout: T={T}, S={S}. "
                f"(cycle_id.max={int(cycle_id.max().item()) if cycle_id.numel() > 0 else 'NA'}, "
                f"L={L})"
            )

        idx = idx.long()
        if idx.numel() > 0:
            if torch.any(idx < 0) or torch.any(idx >= L):
                raise RuntimeError(f"cycle index out of bounds: idx.min={idx.min().item()}, idx.max={idx.max().item()}, L={L}")

        idx_flat = idx.view(-1)
        idx_flat_expand = idx_flat.unsqueeze(0).expand(B, -1)
        synd_flat = torch.gather(syndrome, dim=1, index=idx_flat_expand)
        synd_ts = synd_flat.view(B, T, S)

        stab_ts = stab_id[idx].unsqueeze(0).expand(B, T, S)
        cyc_ts = cycle_id[idx].unsqueeze(0).expand(B, T, S)

        S_seq = self.embed(
            synd_ts.reshape(B * T, S),
            stab_ts.reshape(B * T, S),
            cyc_ts.reshape(B * T, S),
        ).reshape(B, T, S, -1)
        D = S_seq.size(-1)

        # Zero out embeddings for padded positions so they don't
        # contribute to attention or convolution.
        # pad_mask: (T, S), expand to (B, T, S, D)
        pad_mask_btsd = pad_mask.unsqueeze(0).unsqueeze(-1).expand(B, T, S, D)
        S_seq = S_seq * pad_mask_btsd.float()

        # ---- Recurrent core + optional next-stab head ----
        X = S_seq.new_zeros((B, S, D))
        pred_list: List[torch.Tensor] = []
        if self.use_next_stab:
            if self.next_head is None:
                raise RuntimeError("use_next_stab=True but next_head is None")

        for t in range(T):
            token_t = S_seq[:, t]

            if hasattr(self.bias_provider, 'forward') and 'cycle' in self.bias_provider.forward.__code__.co_varnames:
                Bbias = self.bias_provider(batch, S=S, cycle=t)
            else:
                Bbias = self.bias_provider(batch, S=S)

            X = self.core(X, token_t, Bbias)

            # Zero out padded positions in hidden state after each step
            # Use the pad_mask from the cycle with most stabilizers (all True)
            # Since pad_mask varies per cycle, use a static mask for the
            # hidden state: positions that are NEVER real in ANY cycle
            # should stay zero. But since middle cycles have all 8, those
            # positions are used — only boundary cycles have padding.
            # The hidden state accumulates across all cycles, so we do NOT
            # zero it here; the zero-embedding for padded tokens at boundary
            # cycles is sufficient.

            if self.use_next_stab and (t < T - 1):
                pred_t = self.next_head(X)
                pred_list.append(pred_t)

        # ---- Readout: ReadoutResNet ----
        Xp = self.readout_ln(X)
        logical_logits = self.readout(Xp)  # (B,)

        out: Dict[str, torch.Tensor] = {
            "logical_logits": logical_logits,
        }

        if self.use_next_stab:
            if T <= 1:
                out["pred_stabs"] = syndrome.new_zeros((B, 0, S))
            else:
                out["pred_stabs"] = torch.stack(pred_list, dim=1)

        return out
