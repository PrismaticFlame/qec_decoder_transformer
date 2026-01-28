# parameter.py ok
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch


# -------------------------
# Model (Table S3: network hyperparameters)
# -------------------------
@dataclass
class ModelConfigScaling:
    # Feature embedding
    feature_embed_resnet_layers: int = 2

    # Syndrome transformer
    syndrome_layers: int = 3
    d_model: int = 256
    num_heads: int = 4
    key_size: int = 32
    conv_layers: int = 3
    conv_dim: int = 128
    dense_widen: int = 5

    # Attention bias
    bias_dim: int = 48
    bias_residual_layers: int = 8
    indicator_features: int = 7

    # Readout ResNet
    readout_resnet_layers: int = 16
    readout_dim: int = 48


# -------------------------
# Train (Table S3: optimizer/schedule hyperparameters)
# -------------------------
@dataclass
class ScalingConfig:
    # Optimizer / schedule
    optimizer: str = "lion"
    lr: float = 1.3e-4
    weight_decay: float = 1e-7
    beta1: float = 0.9
    beta2: float = 0.95

    batch_init: int = 256
    batch_final: int = 1024
    batch_change_step: int = 800_000

    lr_decay_factor: float = 0.7
    lr_decay_steps: List[int] = field(default_factory=lambda: [400_000, 800_000, 1_600_000])
    grad_clip_norm: float = 1.0

    # Loss weights
    next_stab_pred_weight: float = 0.02

    # EMA (paper-style)
    use_ema: bool = True
    ema_alpha: float = 1e-4  # paper constant: new weight fraction (alpha)

    # Logging / runtime
    num_steps: int = 2_000_000
    log_every: int = 50
    eval_every: int = 5_000
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 4
    pin_memory: bool = True

    # Reproducibility / fitting gates
    seed: int = 42
    min_r2: float = 0.9
    min_intercept: float = -0.02

    # Evaluation cycles
    eval_cycles: Tuple[int, ...] = (3, 5, 7, 9, 11, 13, 15, 25)
    eval_fit_mode: str = "simple"  # "sycamore" for multi-cycle fit, "simple" for direct LER

    # ---- Backward-compat alias ----
    # 你原本叫 ema_decay，但其實語意是 alpha=1e-4
    @property
    def ema_decay(self) -> float:
        # keep old name usable
        return self.ema_alpha

    @ema_decay.setter
    def ema_decay(self, v: float) -> None:
        self.ema_alpha = float(v)


def build_scaling_config() -> ScalingConfig:
    # 目前 default_factory 已經把 lr_decay_steps 填好了
    return ScalingConfig()


# -------------------------
# Another model config (your internal variant)
# -------------------------
@dataclass
class ModelConfig:
    d: int = 3
    d_d: int = 256
    d_attn: int = 64
    d_mid: int = 64
    H: int = 8
    db: int = 32
    n_layers: int = 4
    widen: int = 4

    conv_layers: int = 2
    conv_channels: List[int] = field(default_factory=lambda: [128, 256])
    conv_dilations: Optional[List[int]] = None

    use_next_stab: bool = True
    next_stab_weight: float = 0.02


def build_model_config(d: int = 3) -> ModelConfig:
    cfg = ModelConfig(d=d)
    # 如果你想依 distance 自動調 conv_dilations，也可以在這裡寫
    if cfg.conv_dilations is None:
        cfg.conv_dilations = [1] * cfg.conv_layers
    return cfg
