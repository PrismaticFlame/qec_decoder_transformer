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
# Shared helpers used by both bias provider variants
# ============================================================

def _build_geometric_features(
    stab_xy: torch.Tensor,
    stab_type: Optional[torch.Tensor],
    coord_scale: float,
    max_dist: int,
) -> Dict[str, torch.Tensor]:
    """
    Build pairwise geometry tensors from stabilizer coordinates and types.
    Returns dict of (S, S, *) tensors — no batch dimension.
    """
    S = stab_xy.size(0)
    xy = stab_xy.float()
    dx = xy[:, None, 0] - xy[None, :, 0]  # (S, S)
    dy = xy[:, None, 1] - xy[None, :, 1]  # (S, S)
    dist = (dx.abs() + dy.abs()).long().clamp_(0, max_dist)
    coords = torch.stack([
        xy[:, 0].unsqueeze(1).expand(S, S),
        xy[:, 1].unsqueeze(1).expand(S, S),
        xy[:, 0].unsqueeze(0).expand(S, S),
        xy[:, 1].unsqueeze(0).expand(S, S),
    ], dim=-1) * coord_scale  # (S, S, 4)
    offset = torch.stack([dx, dy], dim=-1) * coord_scale  # (S, S, 2)
    if stab_type is None:
        stab_type = torch.zeros(S, dtype=torch.long, device=stab_xy.device)
    type_pair = torch.stack([
        stab_type.unsqueeze(1).expand(S, S),
        stab_type.unsqueeze(0).expand(S, S),
    ], dim=-1)  # (S, S, 2)
    return {"dist": dist, "coords": coords, "offset": offset, "type_pair": type_pair}


def _build_event_indicators(
    syndrome: torch.Tensor,
    cycle_index: torch.Tensor,
    t: int,
    indicator_features: int,
) -> torch.Tensor:
    """
    Build pairwise event indicator features for cycle t.
    Returns (B, S, S, indicator_features).
    """
    if cycle_index.dim() == 3:
        cycle_index = cycle_index[0]  # (B, T, S) -> (T, S)
    B = syndrome.size(0)
    S = cycle_index.size(1)
    if t >= cycle_index.size(0):
        return torch.zeros(
            B, S, S, indicator_features,
            device=syndrome.device, dtype=torch.float32,
        )
    synd_t = syndrome[:, cycle_index[t]].float()  # (B, S)
    ei = synd_t.unsqueeze(2)  # (B, S, 1)
    ej = synd_t.unsqueeze(1)  # (B, 1, S)
    return torch.stack([
        ei.expand(B, S, S),
        ej.expand(B, S, S),
        ei * ej,
        (ei - ej).abs(),
        torch.maximum(ei, ej).expand(B, S, S),
        torch.minimum(ei, ej).expand(B, S, S),
        ((ei + ej) / 2.0).expand(B, S, S),
    ], dim=-1)  # (B, S, S, 7)


def _extract_batch_geometry(
    batch: Dict[str, torch.Tensor],
    S: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[str]]:
    """
    Extract and validate stab_xy, stab_type, and patch_id from the batch dict.
    Returns (stab_xy, stab_type, patch_id).
    """
    stab_xy = batch.get("stab_xy")
    if stab_xy is None:
        raise KeyError("bias_provider requires batch['stab_xy']")
    if stab_xy.dim() == 3:
        stab_xy = stab_xy[0]
    if stab_xy.dim() != 2 or stab_xy.size(0) != S:
        raise ValueError(f"stab_xy must be (S,2), got {stab_xy.shape}")
    stab_xy = stab_xy.to(device)

    stab_type = batch.get("stab_type")
    if stab_type is not None:
        if stab_type.dim() == 2:
            stab_type = stab_type[0]
        stab_type = stab_type.to(device).long()

    # DataLoader collates strings into a list; all samples in a batch share the
    # same patch, so the first element is the correct identifier.
    patch_id = batch.get("patch_id")
    if isinstance(patch_id, (list, tuple)):
        patch_id = patch_id[0]

    return stab_xy, stab_type, patch_id


# ============================================================
# AttentionBiasProviderUnified — AlphaQubit-faithful baseline
# ============================================================

class AttentionBiasProviderUnified(nn.Module):
    """
    Original AlphaQubit-faithful attention bias provider.

    Geometry and event features are concatenated and processed jointly through
    a single ResNet every cycle step. This allows the network to learn arbitrary
    non-linear interactions between spatial layout and detection events (e.g.
    "close together AND both fired").

    Cost per cycle step: (B, S, S, input_dim) through num_residual_layers.
    No caching — everything recomputed each step.

    Controlled by ModelConfig.bias_residual_layers.
    """

    def __init__(
        self,
        *,
        db: int,
        max_dist: int = 8,
        num_residual_layers: int = 8,
        indicator_features: int = 7,
        coord_scale: float = 0.5,
    ):
        super().__init__()
        self.db = db
        self.max_dist = max_dist
        self.coord_scale = coord_scale
        self.indicator_features = indicator_features

        # Geometry embeddings
        self.dist_emb = nn.Embedding(max_dist + 1, db // 4)
        self.type_emb = nn.Embedding(2, db // 8)
        self.coord_proj = nn.Linear(4, db // 4)   # (x_i, y_i, x_j, y_j)
        self.offset_proj = nn.Linear(2, db // 8)  # (dx, dy)

        # Event projection: indicator_features -> event_dim
        event_dim = db // 4 if indicator_features > 0 else 0
        self.event_proj = (
            nn.Linear(indicator_features, event_dim)
            if indicator_features > 0 else None
        )

        # Combined input dim: geom_dim + event_dim
        # geom_dim  = db//4 + db//4 + db//8 + db//4 = 7*db//8
        # event_dim = db//4
        # input_dim = db//4 + db//4 + db//8 + db//4 + db//4 = db + db//8
        geom_dim = (db // 4) + (db // 4) + (db // 8) + (db // 4)
        input_dim = geom_dim + event_dim
        self.input_dim = input_dim

        # Unified ResNet: runs on (B, S, S, input_dim) every cycle step.
        # Geometry and events are processed together so the network can learn
        # their interactions across all layers.
        self.resnet = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, db),
                nn.LayerNorm(db),
                nn.GELU(),
                nn.Linear(db, input_dim),
            )
            for _ in range(num_residual_layers)
        ])
        self.final_proj = nn.Linear(input_dim, db)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        S: int,
        cycle: Optional[int] = None,
    ) -> torch.Tensor:
        """Returns bias (B, S, S, db). patch_id is accepted but ignored."""
        device = next(self.parameters()).device
        stab_xy, stab_type, _ = _extract_batch_geometry(batch, S, device)

        geom = _build_geometric_features(stab_xy, stab_type, self.coord_scale, self.max_dist)
        B = batch["syndrome"].size(0)

        # Embed geometry — expand to batch dimension
        dist_emb    = self.dist_emb(geom["dist"]).unsqueeze(0).expand(B, S, S, -1)
        coords_emb  = self.coord_proj(geom["coords"]).unsqueeze(0).expand(B, S, S, -1)
        offset_emb  = self.offset_proj(geom["offset"]).unsqueeze(0).expand(B, S, S, -1)
        type_emb    = torch.cat([
            self.type_emb(geom["type_pair"][:, :, 0]),
            self.type_emb(geom["type_pair"][:, :, 1]),
        ], dim=-1).unsqueeze(0).expand(B, S, S, -1)

        # Embed events
        if self.event_proj is not None and cycle is not None:
            cycle_index = batch.get("cycle_index")
            if cycle_index is not None:
                event_features = _build_event_indicators(
                    batch["syndrome"], cycle_index, cycle, self.indicator_features
                )
            else:
                event_features = torch.zeros(
                    B, S, S, self.indicator_features, device=device,
                )
            event_emb = self.event_proj(event_features.float())  # (B, S, S, event_dim)
            x = torch.cat(
                [dist_emb, coords_emb, offset_emb, type_emb, event_emb], dim=-1
            )
        else:
            x = torch.cat([dist_emb, coords_emb, offset_emb, type_emb], dim=-1)

        # Unified ResNet — geometry and events interact across all layers
        for layer in self.resnet:
            x = x + layer(x)

        return self.final_proj(x)  # (B, S, S, db)


# ============================================================
# AttentionBiasProviderSplit — our efficiency-optimised variant
# ============================================================

class AttentionBiasProviderSplit(nn.Module):
    """
    Split attention bias provider: separate geometry and interaction paths.

    Geometry path (geom_resnet_layers, default 2):
        Encodes the fixed spatial layout (distances, coordinates, types) into
        (S, S, db). No batch dimension. During eval this result is cached per
        physical patch (patch_id) so the geometry ResNet never reruns. During
        training it is recomputed each step so geometry weights receive gradients.

    Interaction path (interaction_resnet_layers, default 8):
        Runs on (B, S, S, db) every cycle step. Takes the sum of the geometry
        bias and the projected event features, then processes them jointly so the
        network can learn non-linear geometry-event interactions (e.g. "close
        together AND both fired").

    This preserves the expressiveness of the unified approach at lower cost:
    the expensive deep ResNet operates on (S, S, *) without a batch dimension
    for geometry, and on (B, S, S, db) — already at the final dimensionality —
    for interactions.

    Controlled by ModelConfig.geom_resnet_layers and interaction_resnet_layers.
    """

    def __init__(
        self,
        *,
        db: int,
        max_dist: int = 8,
        geom_resnet_layers: int = 2,
        interaction_resnet_layers: int = 8,
        indicator_features: int = 7,
        coord_scale: float = 0.5,
    ):
        super().__init__()
        self.db = db
        self.max_dist = max_dist
        self.coord_scale = coord_scale
        self.indicator_features = indicator_features

        # Geometry embeddings
        self.dist_emb = nn.Embedding(max_dist + 1, db // 4)
        self.type_emb = nn.Embedding(2, db // 8)
        self.coord_proj = nn.Linear(4, db // 4)
        self.offset_proj = nn.Linear(2, db // 8)

        # geom_dim = db//4 + db//4 + db//8 + db//4 = 7*db//8
        geom_dim = (db // 4) + (db // 4) + (db // 8) + (db // 4)
        self.geom_dim = geom_dim

        # Geometry ResNet: small, runs on (S, S, geom_dim) — no batch, no events.
        # 1-2 layers is sufficient; geometry is fixed and shallow to encode.
        self.geom_resnet = nn.ModuleList([
            nn.Sequential(
                nn.Linear(geom_dim, db),
                nn.LayerNorm(db),
                nn.GELU(),
                nn.Linear(db, geom_dim),
            )
            for _ in range(geom_resnet_layers)
        ])
        self.geom_proj = nn.Linear(geom_dim, db)

        # Event projection: indicator_features -> db (matches geometry output space)
        self.event_proj = (
            nn.Linear(indicator_features, db)
            if indicator_features > 0 else None
        )

        # Interaction ResNet: runs on (B, S, S, db) every cycle step.
        # Sees geometry + events in the same db-dimensional space and learns
        # their non-linear interactions. Deeper than geometry ResNet because
        # this is where the meaningful learning happens.
        self.interaction_resnet = nn.ModuleList([
            nn.Sequential(
                nn.Linear(db, db),
                nn.LayerNorm(db),
                nn.GELU(),
                nn.Linear(db, db),
            )
            for _ in range(interaction_resnet_layers)
        ])

        # Per-patch eval cache: patch_id -> (S, S, db) geometry bias tensor.
        # Only populated during eval. During training always recomputed for gradients.
        self._geom_bias_cache: Dict[str, torch.Tensor] = {}

    def _compute_geom_bias(self, geom: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed geometry features and run through the geometry ResNet. Returns (S, S, db)."""
        dist_emb   = self.dist_emb(geom["dist"])            # (S, S, db//4)
        coords_emb = self.coord_proj(geom["coords"])         # (S, S, db//4)
        offset_emb = self.offset_proj(geom["offset"])        # (S, S, db//8)
        type_emb   = torch.cat([
            self.type_emb(geom["type_pair"][:, :, 0]),
            self.type_emb(geom["type_pair"][:, :, 1]),
        ], dim=-1)                                           # (S, S, db//4)

        x = torch.cat([dist_emb, coords_emb, offset_emb, type_emb], dim=-1)  # (S, S, geom_dim)
        for layer in self.geom_resnet:
            x = x + layer(x)
        return self.geom_proj(x)  # (S, S, db)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        S: int,
        cycle: Optional[int] = None,
    ) -> torch.Tensor:
        """Returns bias (B, S, S, db)."""
        device = next(self.parameters()).device
        stab_xy, stab_type, patch_id = _extract_batch_geometry(batch, S, device)

        # --- Geometry bias (cached during eval, recomputed during training) ---
        cached = (
            not self.training
            and patch_id is not None
            and patch_id in self._geom_bias_cache
        )
        if cached:
            geom_bias = self._geom_bias_cache[patch_id]
        else:
            geom = _build_geometric_features(
                stab_xy, stab_type, self.coord_scale, self.max_dist
            )
            geom_bias = self._compute_geom_bias(geom)  # (S, S, db)
            if not self.training and patch_id is not None:
                self._geom_bias_cache[patch_id] = geom_bias

        B = batch["syndrome"].size(0)

        # --- Event embedding (per cycle step) ---
        if self.event_proj is not None and cycle is not None:
            cycle_index = batch.get("cycle_index")
            if cycle_index is not None:
                event_features = _build_event_indicators(
                    batch["syndrome"], cycle_index, cycle, self.indicator_features
                )
            else:
                event_features = torch.zeros(
                    B, S, S, self.indicator_features, device=device,
                )
            event_emb = self.event_proj(event_features.float())  # (B, S, S, db)
        else:
            event_emb = geom_bias.new_zeros(B, S, S, self.db)

        # --- Interaction ResNet ---
        # Geometry and events are combined in db space then processed jointly.
        # The network can learn non-linear interactions across all layers.
        x = geom_bias.unsqueeze(0) + event_emb  # (B, S, S, db)
        for layer in self.interaction_resnet:
            x = x + layer(x)

        return x  # (B, S, S, db)