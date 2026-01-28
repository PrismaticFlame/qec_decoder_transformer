# model.py ok done
from __future__ import annotations
try:
    from .utils import ManhattanDistanceBias
except ImportError:
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
        """
        syndrome: (B, L) int(0/1) or float
        stab_id:  (L,) or (B,L)
        cycle_id: (L,) or (B,L)
        """
        B, L = syndrome.shape

        if stab_id.dim() == 1:
            stab_id = stab_id.unsqueeze(0).expand(B, L)
        elif stab_id.dim() == 2 and stab_id.size(0) == 1 and B > 1:
            stab_id = stab_id.expand(B, L)

        if cycle_id.dim() == 1:
            cycle_id = cycle_id.unsqueeze(0).expand(B, L)
        elif cycle_id.dim() == 2 and cycle_id.size(0) == 1 and B > 1:
            cycle_id = cycle_id.expand(B, L)

        e_stab = self.stab_emb(stab_id)       # (B,L,D)
        e_cycle = self.cycle_emb(cycle_id)    # (B,L,D)

        if syndrome.is_floating_point():
            e_val = self.analog_proj(syndrome.unsqueeze(-1))  # (B,L,1)->(B,L,D) linear只改最後一維 需要先unsqueeze
        else:
            e_val = self.val_emb(syndrome.long())             # (B,L)->(B,L,D) embedding可以自動新增最後一維

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
        # X: (B,L,Dd), Bp: (B,L,L)
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        S = torch.matmul(Q, K.transpose(-2, -1))   # matmul只看最後兩維(B,L,D)(B,D,L)=(B,L,L)
        S = S + Bp

        A = F.softmax(S / math.sqrt(self.d_attn), dim=-1) #.sqrt讓分布較平滑
        out = torch.matmul(A, V)                   # (B,L,Dmid)
        return out, A


# ============================================================
# Algorithm 2: Multi-head AttentionWithBias
# ============================================================
class MHAttentionWithBias(nn.Module):
    def __init__(self, d_d: int, d_attn: int, d_mid: int, db: int, H: int):
        super().__init__()
        self.H = H #head
        self.Wb = nn.Linear(db, H, bias=False) #weight bias
        self.heads = nn.ModuleList([AttentionWithBiasHead(d_d, d_attn, d_mid) for _ in range(H)])
        self.Wo = nn.Linear(H * d_mid, d_d, bias=True)

    def forward(self, X: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """
        X:    (B,L,Dd)
        bias: (B,L,L,db)
        """
        Bsz, L, _, db = bias.shape

        Bprime = self.Wb(bias)                # (B,L,L,H)
        Bprime = Bprime.permute(0, 3, 1, 2)   # (B,H,L,L)

        Ys = []
        for h in range(self.H):
            out_h, _ = self.heads[h](X, Bprime[:, h, :, :])  # (B,L,Dmid), attention weights ignored
            Ys.append(out_h)
        Y = torch.cat(Ys, dim=-1)             # (B,L,H*Dmid)
        return self.Wo(Y)                     # (B,L,Dd)


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

        self.P = nn.Parameter(torch.zeros(d_d)) #存在parameter才會被optimizer更新

        self.lns = nn.ModuleList([nn.LayerNorm(d_d) for _ in range(L_layers)])
        self.convs3 = nn.ModuleList()
        self.proj1 = nn.ModuleList() #ModuleList：.to(device) 會一起移動它們

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
        grid[:] = self.P.view(1, 1, 1, Dd) #[:]inplace，省記憶體 .view()類似reshape 要求連續數值
        for (i, j), idx in self.coord_to_index.items(): #.items()：index 非連續數值
            grid[:, i, j, :] = X[:, idx, :]

        Y = grid.permute(0, 3, 1, 2).contiguous()  # (B,Dd,H,W)

        for l in range(len(self.convs3)):
            Yn = Y.permute(0, 2, 3, 1) #(B,H,W,Dd)
            Yn = self.lns[l](Yn) #最後一維做layernorm
            Yn = Yn.permute(0, 3, 1, 2)  # (B,Dd,H,W)

            Yn = self.convs3[l](Yn)
            Yn = F.gelu(Yn)
            Yn = self.proj1[l](Yn)
            Y = Y + Yn

        Y_hw = Y.permute(0, 2, 3, 1).contiguous() #contiguous把資料在記憶體裡排成連續的一塊 把它當正常 tensor 用
        Z = X.new_empty((B, L, Dd))
        for idx, (i, j) in enumerate(self.index_to_coord): #enumerate：index 連續數值對應到座標
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
        X = (X + S) / math.sqrt(2.0) #把殘差相加後除以 √2，是為了讓「特徵的尺度（variance）」保持穩定，不隨層數爆掉
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
class NextStabPredictor(nn.Module): #模型早點學會 stabilizer 間的關聯 加速收斂但會影響最終效能->可試試warm up(非論文)
    def __init__(self, d_d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_d, d_d // 2),
            nn.GELU(),
            nn.Linear(d_d // 2, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, S, Dd) -> (B,S)
        return self.net(X).squeeze(-1)


# ============================================================
# Top-level model that takes your batch dict (B,L)
# and internally builds (B,T,S,D)
# ============================================================


class AlphaQubitLikeModel(nn.Module):
    """
    batch from dataset:
      syndrome:        (B,L)
      stab_id:         (L,)
      cycle_id:        (L,)
      logical_labels:  (B,) or (B,1)   # 由 dataset 提供（model 不用它算 loss）
      label_x:         (B,)            # X 型邏輯錯誤標籤
      label_z:         (B,)            # Z 型邏輯錯誤標籤
      true_stabs:      (B,T-1,S)       # optional（若要 next-stab loss）
      token_mask:      (B,T-1,S)       # optional（mask leakage/pad/terminal）
    outputs:
      logical_logits:  (B,)            # 向後兼容（等於 logical_logits_x）
      logical_logits_x: (B,)           # X 型邏輯錯誤 logits
      logical_logits_z: (B,)           # Z 型邏輯錯誤 logits
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
    ):
        super().__init__()
        self.embed = StabilizerEmbedding(num_stab=num_stab, num_cycles=num_cycles, d_model=d_model)
        self.core = RNNCore(d_model, d_attn, d_mid, db, H, n_layers, conv_block, widen=widen)

        self.readout_ln = nn.LayerNorm(d_model)
        # 分別的 X 和 Z 分類器
        self.classifier_x = nn.Linear(d_model, 1)
        self.classifier_z = nn.Linear(d_model, 1)
        # 向後兼容：保留單個 classifier
        self.classifier = self.classifier_x

        self.use_next_stab = use_next_stab
        self.next_head = NextStabPredictor(d_model) if use_next_stab else None

        self.bias_provider = bias_provider

        # cache indices (rebuilt when layout changes)
        self._cycle_index: Optional[torch.Tensor] = None  # (T,S) indices into L
        self._T: Optional[int] = None
        self._S: Optional[int] = None
        self._L: Optional[int] = None #_:不要從外部直接修改這些數值 避免你換了不同 layout 還沿用舊 index

    @staticmethod
    def _build_cycle_index(stab_id: torch.Tensor, cycle_id: torch.Tensor) -> Tuple[int, int, torch.Tensor]:
        """
        Build indices mapping from flat L tokens -> per-cycle (T,S) tokens.
        For each cycle t, select positions where cycle_id == t, then sort by stab_id
        so the per-cycle ordering is consistent across time.
        If some cycles have fewer tokens, truncate all cycles to the minimum S.
        """
        if stab_id.dim() != 1 or cycle_id.dim() != 1:
            raise ValueError(f"stab_id and cycle_id must be 1D, got {stab_id.shape}, {cycle_id.shape}")
        if stab_id.numel() != cycle_id.numel():
            raise ValueError(f"stab_id and cycle_id must have same length, got {stab_id.numel()} vs {cycle_id.numel()}")

        L = cycle_id.numel()
        # cycles assumed 0..T-1
        T = int(cycle_id.max().item()) + 1
        if T <= 0:
            raise ValueError(f"Invalid T computed from cycle_id.max(): T={T}")

        indices_per_t: List[torch.Tensor] = [] #存per cycle的detector event
        S_min: Optional[int] = None #所有回合中，穩定子數量最少是多少(有時候leak會導致缺少)

        for t in range(T):
            idx_t = torch.nonzero(cycle_id == t, as_tuple=False).view(-1)  # positions in [0,L), as_tuple=False等於tensor
            if idx_t.numel() == 0:
                # allow empty cycle but then S becomes 0
                indices_per_t.append(idx_t) #即使沒資料也要放空的站位 避免cycle亂掉 RNN不接受後面遞補
                S_min = 0 if S_min is None else min(S_min, 0) #只要有一輪空掉，整組的cycle都不要了
                continue

            # sort within cycle by stab_id to make stable order
            idx_t = idx_t[torch.argsort(stab_id[idx_t])] #argsort接收tensor, 由小到大
            indices_per_t.append(idx_t)

            if S_min is None: 
                S_min = idx_t.numel() #第一回合的初始化用上一輪的數據個數
            else:
                S_min = min(S_min, idx_t.numel()) #每一cycle更新

        S = int(S_min or 0)
        if S < 0:
            raise ValueError(f"Invalid S computed: S={S}")

        # stack (T,S) by truncating each cycle to S
        if S == 0:
            indices = torch.zeros((T, 0), dtype=torch.long, device=cycle_id.device)
        else:
            indices = torch.stack([idx_t[:S] for idx_t in indices_per_t], dim=0).long()  # (T,S)。[:S]剪枝

        return T, S, indices #T:幾個cycle（<=總cycle） S:每一輪我們能用的穩定子數量 

    def _ensure_cycle_index(
        self,
        *,
        stab_id: torch.Tensor,
        cycle_id: torch.Tensor,
        syndrome: torch.Tensor,
    ) -> Tuple[int, int, torch.Tensor]:
        """
        Ensure cached (T,S,idx) matches current layout. Rebuild if needed.
        """
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

        # also rebuild if device mismatch
        if self._cycle_index is not None and self._cycle_index.device != syndrome.device:
            need_rebuild = True

        if need_rebuild:
            T, S, idx = self._build_cycle_index(stab_id, cycle_id)
            self._cycle_index = idx.to(device=syndrome.device)
            self._T, self._S, self._L = T, S, L

        # mypy-friendly
        assert self._cycle_index is not None
        assert self._T is not None and self._S is not None

        return self._T, self._S, self._cycle_index

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        syndrome = batch["syndrome"]  # (B,L)
        stab_id = batch["stab_id"]    # (L,) or (B,L) if batched
        cycle_id = batch["cycle_id"]  # (L,) or (B,L) if batched

        if syndrome.dim() != 2:
            raise ValueError(f"syndrome must be (B,L), got {syndrome.shape}")
        B, L = syndrome.shape

        # ---- Handle batched layout tensors (DataLoader stacks them) ----
        # stab_id and cycle_id are the same for all samples, so take first row if batched
        if stab_id.dim() == 2:
            stab_id = stab_id[0]  # (B,L) -> (L,)
        if cycle_id.dim() == 2:
            cycle_id = cycle_id[0]  # (B,L) -> (L,)

        # ---- Make sure ids are on same device + correct dtype for Embedding/indexing ----
        # Keep syndrome dtype as-is (could be float or int). ids must be long.
        stab_id = stab_id.to(device=syndrome.device).long().view(-1)     # (L,)
        cycle_id = cycle_id.to(device=syndrome.device).long().view(-1)   # (L,)

        if stab_id.numel() != L or cycle_id.numel() != L:
            raise ValueError(
                f"stab_id/cycle_id must have length L={L}, got {stab_id.numel()} and {cycle_id.numel()}"
            )

        # ---- Build / reuse (T,S) index safely ----
        T, S, idx = self._ensure_cycle_index(stab_id=stab_id, cycle_id=cycle_id, syndrome=syndrome)

        # If S==0, short-circuit to avoid downstream shape surprises
        if S == 0 or T == 0:
            raise ValueError(
                f"Invalid per-cycle layout: T={T}, S={S}. "
                f"(cycle_id.max={int(cycle_id.max().item()) if cycle_id.numel() > 0 else 'NA'}, "
                f"L={L})"
            )

        # ---- Gather per-cycle syndrome: (B,T,S) ----
        idx = idx.long()
        if idx.numel() > 0:
            # sanity bounds (debug-safe; cheap)
            if torch.any(idx < 0) or torch.any(idx >= L):
                raise RuntimeError(f"cycle index out of bounds: idx.min={idx.min().item()}, idx.max={idx.max().item()}, L={L}")

        # idx is (T, S), need to gather from syndrome (B, L) -> (B, T, S)
        # Flatten idx to (T*S,), gather, then reshape
        idx_flat = idx.view(-1)  # (T*S,)
        idx_flat_expand = idx_flat.unsqueeze(0).expand(B, -1)  # (B, T*S)
        synd_flat = torch.gather(syndrome, dim=1, index=idx_flat_expand)  # (B, T*S)
        synd_ts = synd_flat.view(B, T, S)  # (B, T, S)

        # ---- Build per-cycle ids via direct indexing: (T,S) -> (B,T,S) ----
        stab_ts = stab_id[idx].unsqueeze(0).expand(B, T, S)   # (1,T,S)->(B,T,S)
        cyc_ts = cycle_id[idx].unsqueeze(0).expand(B, T, S)   # (B,T,S)

        # ---- Embed to (B,T,S,D) ----
        S_seq = self.embed(
        synd_ts.reshape(B * T, S),
        stab_ts.reshape(B * T, S),
        cyc_ts.reshape(B * T, S),
        ).reshape(B, T, S, -1)
        #為什麼要reshape不直接用(B,T,S)embed?對GPU來說處理二維比較有效率 
        D = S_seq.size(-1)

        # ---- Bias: expected (B,S,S,db) ----
        # Note: For AttentionBiasProvider, bias may depend on cycle, so we compute it per cycle
        # For backward compatibility with ManhattanDistanceBias, we compute once if cycle=None
        
        # ---- Recurrent core + optional next-stab head ----
        X = S_seq.new_zeros((B, S, D))  # hidden：模型目前對這一輪（到目前為止）整個 stabilizer lattice 的內部記憶
        pred_list: List[torch.Tensor] = []
        if self.use_next_stab:
            if self.next_head is None:
                raise RuntimeError("use_next_stab=True but next_head is None")

        for t in range(T):
            token_t = S_seq[:, t]               # (B,T,S,D)->(B,S,D)
            
            # Compute bias (may be cycle-dependent for AttentionBiasProvider)
            # Check if bias_provider accepts cycle parameter
            if hasattr(self.bias_provider, 'forward') and 'cycle' in self.bias_provider.forward.__code__.co_varnames:
                # New AttentionBiasProvider that supports cycle-dependent bias
                Bbias = self.bias_provider(batch, S=S, cycle=t)
            else:
                # Backward compatibility: ManhattanDistanceBias (cycle-independent)
                Bbias = self.bias_provider(batch, S=S)
            
            X = self.core(X, token_t, Bbias)    # (B,S,D)

            # Predict NEXT cycle syndrome after seeing current cycle t
            # => total predictions: (T-1)
            if self.use_next_stab and (t < T - 1):
                pred_t = self.next_head(X)      # (B,S)
                pred_list.append(pred_t)

        # ---- Readout logical ----
        Xp = self.readout_ln(X)                 # (B,S,D)
        pooled = Xp.mean(dim=1)                 # (B,D)
        
        # 分別計算 X 和 Z 的 logits
        logical_logits_x = self.classifier_x(pooled).squeeze(-1)  # (B,)
        logical_logits_z = self.classifier_z(pooled).squeeze(-1)  # (B,)
        
        # 向後兼容：logical_logits 等於 logical_logits_x
        logical_logits = logical_logits_x

        out: Dict[str, torch.Tensor] = {
            "logical_logits": logical_logits,      # 向後兼容
            "logical_logits_x": logical_logits_x,  # X 型邏輯錯誤 logits
            "logical_logits_z": logical_logits_z,  # Z 型邏輯錯誤 logits
        }

        if self.use_next_stab:
            # Always return shape (B, T-1, S)
            if T <= 1:
                out["pred_stabs"] = syndrome.new_zeros((B, 0, S))
            else:
                out["pred_stabs"] = torch.stack(pred_list, dim=1)  # (B,T-1,S)

        return out