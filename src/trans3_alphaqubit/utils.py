# utils.py ok
from __future__ import annotations
import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, Optional

import numpy as np
import torch


# -----------------------------
# 1) Lion optimizer (minimal, stable, efficient)
# -----------------------------
class Lion(torch.optim.Optimizer):
    """
    Minimal Lion optimizer (sign-based) with decoupled weight decay (AdamW style).

    Update (common Lion):
      u = sign(beta1 * m + (1 - beta1) * g)
      p = p - lr * u
      m = beta2 * m + (1 - beta2) * g

    Notes:
      - Call after loss.backward()
      - Does NOT support sparse grads
    """

    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.95), weight_decay: float = 1e-7):
        beta1, beta2 = betas
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"betas must be in [0,1), got {betas}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            wd: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients.")

                g = p.grad.detach()

                state = self.state[p] #第一次會得到空dict
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format) #動量從0開始
                    #memory_format=torch.preserve_format 請讓這個新的張量（exp_avg）在記憶體裡的排列方式，跟原本的參數 p 完全一模一樣 節省電腦運算時間
                m = state["exp_avg"] #之後動量都更新到m

                # decoupled weight decay (AdamW-style)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # u = sign(beta1*m + (1-beta1)*g)  (no clone; use a temp buffer)
             
                u = torch.add(m, g, alpha=(1.0 - beta1) / max(beta1, 1e-12)) if beta1 != 0 else g
                # Explanation:
                # beta1*m + (1-beta1)*g = beta1*(m + ((1-beta1)/beta1)*g)
                # (avoids an extra mul on m, but still allocs u)因為只要看正負號 改成這樣提升效能（計算都在beta純量上）
                # torch.add(input, other, alpha=...)=out=input+α×other
                u = u.sign()

                p.add_(u, alpha=-lr)

                # m = beta2*m + (1-beta2)*g
                m.mul_(beta2).add_(g, alpha=(1.0 - beta2))


# -------------------------
# 3) Fit gate utilities (R^2, intercept, sigma) 要再研究
# -------------------------
def fit_line_with_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]: #p.52 早期訓練（樣本數前 2.5% - 5%）時，模型還不穩定 所以需要門檻過濾掉Validation時的假象model 
    """
    Fit y = a*x + b with OLS.
    Returns slope a, intercept b, R^2, and std error of intercept (sigma_b). 如果 R**2太低，代表 LER 隨 Cycle 增加的規律很亂，這個數據點不可信
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = int(len(x)) 
    if n < 3: #至少要有三個數據(cycle)才能算出「殘差的標準差」
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "sigma_intercept": float("inf")}

    x_mean = float(x.mean())
    y_mean = float(y.mean())

    dx = x - x_mean
    dy = y - y_mean
    Sxx = float((dx ** 2).sum())
    if Sxx <= 0.0: #防止除以零
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "sigma_intercept": float("inf")}

    slope = float((dx * dy).sum() / Sxx)
    intercept = float(y_mean - slope * x_mean)

    y_hat = slope * x + intercept
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y_mean) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else float("nan")

    # Standard error of intercept: 在物理上，當 Cycle = 0 時，邏輯錯誤率應該接近 0。如果擬合出來的直線截距「負得太多」（低於 -0.02 或低於標準差 σ），代表這條線在數學上是扭曲的，不符合物理邏輯
    # sigma^2 = SSE/(n-2); Var(b) = sigma^2 * (1/n + xbar^2/Sxx)
    dof = n - 2
    if dof <= 0:
        sigma_intercept = float("inf")
    else:
        sigma2 = ss_res / dof
        var_b = sigma2 * (1.0 / n + (x_mean ** 2) / Sxx)
        sigma_intercept = float(math.sqrt(max(var_b, 0.0)))

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "sigma_intercept": sigma_intercept,
    }


def gate_fit_ok(
    stats: Dict[str, float],
    *,
    min_r2: float = 0.9,
    min_intercept: float = -0.02,
) -> Tuple[bool, float]:
    """
    Paper gate (as you described):
      - R^2 > min_r2
      - intercept > max(min_intercept, -sigma_intercept)

    Returns:
      (ok, intercept_threshold)
    """
    r2 = float(stats.get("r2", float("nan")))
    b = float(stats.get("intercept", float("nan")))
    sigma_b = float(stats.get("sigma_intercept", float("inf")))

    intercept_thresh = max(float(min_intercept), -sigma_b)
    ok = (not math.isnan(r2)) and (r2 > float(min_r2)) and (not math.isnan(b)) and (b > intercept_thresh)
    return ok, intercept_thresh


class ManhattanDistanceBias(nn.Module):
    """
    Provide pairwise bias based on Manhattan distance between stabilizers.

    Output shape:
      (B, S, S, db)

    Assumptions:
    - batch["stab_xy"] exists with shape (S, 2) or (B, S, 2)
      where stab_xy[..., 0] = x, stab_xy[..., 1] = y
    - Same stabilizer layout for all batches (common in surface code)
    """

    def __init__(self, *, db: int, max_dist: int = 8):
        super().__init__()
        self.db = db
        self.max_dist = max_dist

        # distance embedding: d = 0..max_dist
        self.dist_emb = nn.Embedding(max_dist + 1, db)

        # cache (distance matrix is geometry-only)
        self._cached_dist: Optional[torch.Tensor] = None  # (S,S)

    def _build_dist_matrix(self, stab_xy: torch.Tensor) -> torch.Tensor:
        """
        stab_xy: (S,2)
        return: (S,S) Manhattan distance, clipped to max_dist
        """
        xy = stab_xy.float()  # (S,2)
        dx = xy[:, None, 0] - xy[None, :, 0]
        dy = xy[:, None, 1] - xy[None, :, 1]
        dist = dx.abs() + dy.abs()
        dist = dist.long().clamp_(0, self.max_dist)
        return dist

    def forward(self, batch: Dict[str, torch.Tensor], *, S: int) -> torch.Tensor:
        """
        Returns:
          bias: (B, S, S, db)
        """
        # ---- get stabilizer coordinates ----
        stab_xy = batch.get("stab_xy", None)
        if stab_xy is None:
            raise KeyError(
                "bias_provider requires batch['stab_xy'] with shape (S,2) or (B,S,2)"
            )

        # accept (B,S,2) or (S,2)
        if stab_xy.dim() == 3:
            stab_xy = stab_xy[0]  # assume same layout for all batches

        if stab_xy.dim() != 2 or stab_xy.size(0) != S or stab_xy.size(1) != 2:
            raise ValueError(f"stab_xy must be (S,2), got {stab_xy.shape}")

        stab_xy = stab_xy.to(next(self.parameters()).device)

        # ---- build / reuse distance matrix ----
        if self._cached_dist is None or self._cached_dist.size(0) != S:
            self._cached_dist = self._build_dist_matrix(stab_xy)

        dist = self._cached_dist  # (S,S)

        # ---- embed distance -> bias channels ----
        bias_ssdb = self.dist_emb(dist)  # (S,S,db)

        # ---- expand batch ----
        B = batch["syndrome"].size(0)
        bias = bias_ssdb.unsqueeze(0).expand(B, S, S, self.db)

        return bias


# ============================================================
# Complete Attention Bias Provider (AlphaQubit paper style)
# ============================================================
class AttentionBiasProvider(nn.Module):
    """
    Complete attention bias provider following AlphaQubit paper.
    
    Features included:
    1. Manhattan distance
    2. Coordinates (x, y) for each stabilizer
    3. Offset (dx, dy) between stabilizer pairs
    4. Stabilizer type (X or Z)
    5. Event indicators (detection events from syndrome)
    
    Output shape: (B, S, S, db)
    
    The features are processed through a ResNet to produce the final bias.
    """

    def __init__(
        self,
        *,
        db: int,
        max_dist: int = 8,
        num_residual_layers: int = 8,
        indicator_features: int = 7,
        coord_scale: float = 0.5,  # Normalize coordinates
    ):
        super().__init__()
        self.db = db
        self.max_dist = max_dist
        self.coord_scale = coord_scale

        # Feature dimensions:
        # - Manhattan distance: 1 (embedded to db_dim)
        # - Coordinates: 2 (x, y) per stabilizer -> 4 (x_i, y_i, x_j, y_j) per pair
        # - Offset: 2 (dx, dy)
        # - Type: 2 (type_i, type_j) -> could be binary or embedded
        # - Event indicators: indicator_features (per pair)
        
        # Distance embedding
        self.dist_emb = nn.Embedding(max_dist + 1, db // 4)  # Use 1/4 of db for distance
        
        # Type embedding (X=0, Z=1, or could be more types)
        self.type_emb = nn.Embedding(2, db // 8)  # X/Z type
        
        # Feature dimensions breakdown:
        # - distance: db//4
        # - coords (4): 4 * (db//16) = db//4
        # - offset (2): 2 * (db//16) = db//8
        # - type (2): 2 * (db//8) = db//4
        # - events: indicator_features
        # Total input: db//4 + db//4 + db//8 + db//4 + indicator_features
        
        # Coordinate and offset projections
        coord_dim = db // 4
        self.coord_proj = nn.Linear(4, coord_dim)  # (x_i, y_i, x_j, y_j)
        self.offset_proj = nn.Linear(2, db // 8)   # (dx, dy)
        
        # Event indicator projection
        self.indicator_features = indicator_features
        event_dim = db // 4 if indicator_features > 0 else 0
        if indicator_features > 0:
            self.event_proj = nn.Linear(indicator_features, event_dim)
        else:
            self.event_proj = None
        
        # Total input feature dimension
        input_dim = (db // 4) + coord_dim + (db // 8) + (db // 4) + event_dim
        self.input_dim = input_dim
        
        # ResNet to process features
        self.resnet_layers = nn.ModuleList()
        for i in range(num_residual_layers):
            self.resnet_layers.append(
                nn.Sequential(
                    nn.Linear(input_dim, db),
                    nn.LayerNorm(db),
                    nn.GELU(),
                    nn.Linear(db, input_dim),
                )
            )
        
        # Final projection to db
        self.final_proj = nn.Linear(input_dim, db)
        
        # Cache for geometry-only features
        self._cached_geom: Optional[Dict[str, torch.Tensor]] = None

    def _build_geometric_features(
        self,
        stab_xy: torch.Tensor,
        stab_type: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build geometric features that don't depend on batch/syndrome.
        
        Returns:
            dict with:
                - dist: (S, S) Manhattan distance
                - coords: (S, S, 4) [x_i, y_i, x_j, y_j]
                - offset: (S, S, 2) [dx, dy]
                - type_pair: (S, S, 2) [type_i, type_j]
        """
        S = stab_xy.size(0)
        xy = stab_xy.float()  # (S, 2)
        
        # Manhattan distance
        dx = xy[:, None, 0] - xy[None, :, 0]  # (S, S)
        dy = xy[:, None, 1] - xy[None, :, 1]  # (S, S)
        dist = (dx.abs() + dy.abs()).long().clamp_(0, self.max_dist)  # (S, S)
        
        # Coordinates: [x_i, y_i, x_j, y_j] for each pair
        x_i = xy[:, 0].unsqueeze(1).expand(S, S)  # (S, S)
        y_i = xy[:, 1].unsqueeze(1).expand(S, S)  # (S, S)
        x_j = xy[:, 0].unsqueeze(0).expand(S, S)  # (S, S)
        y_j = xy[:, 1].unsqueeze(0).expand(S, S)  # (S, S)
        coords = torch.stack([x_i, y_i, x_j, y_j], dim=-1)  # (S, S, 4)
        coords = coords * self.coord_scale  # Normalize
        
        # Offset: [dx, dy]
        offset = torch.stack([dx, dy], dim=-1)  # (S, S, 2)
        offset = offset * self.coord_scale  # Normalize
        
        # Type pairs
        if stab_type is None:
            # Default: assume all are Z-type (0) for surface code
            stab_type = torch.zeros(S, dtype=torch.long, device=stab_xy.device)
        
        type_i = stab_type.unsqueeze(1).expand(S, S)  # (S, S)
        type_j = stab_type.unsqueeze(0).expand(S, S)  # (S, S)
        type_pair = torch.stack([type_i, type_j], dim=-1)  # (S, S, 2)
        
        return {
            "dist": dist,
            "coords": coords,
            "offset": offset,
            "type_pair": type_pair,
        }

    def _build_event_indicators(
        self,
        syndrome: torch.Tensor,
        cycle_index: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        Build event indicators from syndrome values.
        
        Args:
            syndrome: (B, L) detection events
            cycle_index: (T, S) indices mapping
            t: current cycle index
            
        Returns:
            event_features: (B, S, S, indicator_features)
        """
        B = syndrome.size(0)
        S = cycle_index.size(1)
        
        if t >= cycle_index.size(0):
            # Return zeros if cycle doesn't exist
            return torch.zeros(B, S, S, self.event_proj.in_features if self.event_proj else 0,
                             device=syndrome.device, dtype=syndrome.dtype)
        
        # Get syndrome values for current cycle
        idx_t = cycle_index[t]  # (S,)
        synd_t = syndrome[:, idx_t]  # (B, S)
        
        # Build pairwise event indicators
        # Features could include:
        # - event_i, event_j (detection events at i and j)
        # - event_i * event_j (both triggered)
        # - |event_i - event_j| (difference)
        # - max(event_i, event_j), min(event_i, event_j)
        # - etc.
        
        event_i = synd_t.unsqueeze(2).expand(B, S, S)  # (B, S, S)
        event_j = synd_t.unsqueeze(1).expand(B, S, S)  # (B, S, S)
        
        # Build indicator features
        features = []
        features.append(event_i)  # event at i
        features.append(event_j)  # event at j
        features.append(event_i * event_j)  # both triggered
        features.append((event_i - event_j).abs())  # difference
        features.append(torch.maximum(event_i, event_j))  # max
        features.append(torch.minimum(event_i, event_j))  # min
        features.append((event_i + event_j) / 2.0)  # average
        
        # Stack to (B, S, S, indicator_features)
        event_features = torch.stack(features, dim=-1)  # (B, S, S, 7)
        
        return event_features

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        S: int,
        cycle: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Build complete attention bias.
        
        Args:
            batch: dict containing:
                - "syndrome": (B, L) detection events
                - "stab_xy": (S, 2) or (B, S, 2) stabilizer coordinates
                - "stab_type": (S,) optional, stabilizer types (0=X, 1=Z)
                - "cycle_index": (T, S) optional, for event indicators
            S: number of stabilizers per cycle
            cycle: current cycle index (for event indicators)
        
        Returns:
            bias: (B, S, S, db)
        """
        # ---- Get stabilizer coordinates ----
        stab_xy = batch.get("stab_xy", None)
        if stab_xy is None:
            raise KeyError("bias_provider requires batch['stab_xy']")
        
        # Accept (B, S, 2) or (S, 2)
        if stab_xy.dim() == 3:
            stab_xy = stab_xy[0]  # Assume same layout for all batches
        
        if stab_xy.dim() != 2 or stab_xy.size(0) != S or stab_xy.size(1) != 2:
            raise ValueError(f"stab_xy must be (S,2), got {stab_xy.shape}")
        
        stab_xy = stab_xy.to(next(self.parameters()).device)
        
        # ---- Get stabilizer type (optional) ----
        stab_type = batch.get("stab_type", None)
        if stab_type is not None:
            if stab_type.dim() == 2:
                stab_type = stab_type[0]  # (S,)
            stab_type = stab_type.to(stab_xy.device).long()
        
        # ---- Build/cache geometric features ----
        cache_key = (S, stab_xy.device)
        if self._cached_geom is None or self._cached_geom.get("S") != S:
            geom = self._build_geometric_features(stab_xy, stab_type)
            self._cached_geom = {"S": S, **geom}
        else:
            geom = {k: v for k, v in self._cached_geom.items() if k != "S"}
        
        # Move to device
        for k, v in geom.items():
            geom[k] = v.to(stab_xy.device)
        
        B = batch["syndrome"].size(0)
        
        # ---- Build event indicators (if available) ----
        if self.event_proj is not None and cycle is not None:
            cycle_index = batch.get("cycle_index", None)
            if cycle_index is not None:
                event_features = self._build_event_indicators(
                    batch["syndrome"], cycle_index, cycle
                )  # (B, S, S, indicator_features)
            else:
                # Fallback: use zeros if cycle_index not available
                event_features = torch.zeros(
                    B, S, S, self.indicator_features,
                    device=stab_xy.device, dtype=batch["syndrome"].dtype
                )
        else:
            # No event features
            event_features = torch.zeros(
                B, S, S, self.indicator_features if self.event_proj else 0,
                device=stab_xy.device, dtype=batch["syndrome"].dtype
            )
        
        # ---- Embed features ----
        # Distance
        dist_emb = self.dist_emb(geom["dist"])  # (S, S, db//4)
        dist_emb = dist_emb.unsqueeze(0).expand(B, S, S, -1)  # (B, S, S, db//4)
        
        # Coordinates
        coords_emb = self.coord_proj(geom["coords"])  # (S, S, coord_dim)
        coords_emb = coords_emb.unsqueeze(0).expand(B, S, S, -1)  # (B, S, S, coord_dim)
        
        # Offset
        offset_emb = self.offset_proj(geom["offset"])  # (S, S, db//8)
        offset_emb = offset_emb.unsqueeze(0).expand(B, S, S, -1)  # (B, S, S, db//8)
        
        # Type
        type_i_emb = self.type_emb(geom["type_pair"][:, :, 0])  # (S, S, db//8)
        type_j_emb = self.type_emb(geom["type_pair"][:, :, 1])  # (S, S, db//8)
        type_emb = torch.cat([type_i_emb, type_j_emb], dim=-1)  # (S, S, db//4)
        type_emb = type_emb.unsqueeze(0).expand(B, S, S, -1)  # (B, S, S, db//4)
        
        # Events
        if self.event_proj is not None:
            event_emb = self.event_proj(event_features)  # (B, S, S, event_dim)
        else:
            event_emb = torch.zeros(B, S, S, 0, device=stab_xy.device)
        
        # ---- Concatenate all features ----
        features = torch.cat([
            dist_emb,      # (B, S, S, db//4)
            coords_emb,    # (B, S, S, coord_dim)
            offset_emb,    # (B, S, S, db//8)
            type_emb,      # (B, S, S, db//4)
            event_emb,     # (B, S, S, event_dim)
        ], dim=-1)  # (B, S, S, input_dim)
        
        # ---- Process through ResNet ----
        x = features
        for layer in self.resnet_layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection
        
        # ---- Final projection ----
        bias = self.final_proj(x)  # (B, S, S, db)
        
        return bias