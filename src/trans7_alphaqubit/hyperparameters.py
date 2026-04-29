# hyperparameters.py - Model and training hyperparameters for trans7 (AlphaQubit)
#
# Values from AlphaQubit Nature paper Table S3 (Scaling experiment column).
# Dilation table from Table S4: d=3 -> [1,1,1], d=5 -> [1,1,2], d>=7 -> [1,2,4]
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

# -----------------------------------------------------------------------
# Dilation table (per code distance)
# -----------------------------------------------------------------------

DILATION_TABLE = {
    3: [1, 1, 1],
    5: [1, 1, 2],
    7: [1, 2, 4],
    9: [1, 2, 4],
    11: [1, 2, 4],
}


def get_dilations(distance: int) -> List[int]:
    """Return the 3-layer dilation list for a given code distance."""
    if distance in DILATION_TABLE:
        return list(DILATION_TABLE[distance])
    # Distances >= 7 all use [1, 2, 4]
    if distance >= 7:
        return [1, 2, 4]
    # distance <= 3 fallback
    return [1, 1, 1]


# Learning rate table (Scaling column, Table S4)
LR_TABLE = {
    3: 1.3e-4,
    5: 1.15e-4,
    7: 1.0e-4,
    9: 7.0e-5,
    11: 5.0e-5,
}


def get_lr(distance: int) -> float:
    """Return the base learning rate for a given code distance."""
    return LR_TABLE.get(distance, 1.3e-4)


# -----------------------------------------------------------------------
# Model hyperparameters
# -----------------------------------------------------------------------


@dataclass
class ModelConfig:
    # Feature embedding
    resnet_layers: int = 2

    # Syndrome transformer
    syndrome_layers: int = 3
    d_model: int = 256
    num_heads: int = 4
    key_size: int = 32
    conv_layers: int = 3
    conv_dim: int = 128  # convolution channel dimension
    dense_widen: int = 5  # GatedDenseBlock widening factor

    # Attention bias
    bias_dim: int = 48
    indicator_features: int = 7

    # Which bias provider to use:
    #   "unified" — AlphaQubit-faithful: geometry + events processed jointly every
    #               cycle step. Uses bias_residual_layers. Slower, more expressive.
    #   "split"   — Our variant: geometry ResNet (cached) + interaction ResNet
    #               (per step). Uses geom_resnet_layers + interaction_resnet_layers.
    bias_mode: str = "unified"
    # bias_mode: str = "split"

    # Unified provider layer count
    bias_residual_layers: int = 8

    # Split provider layer counts
    geom_resnet_layers: int = 2         # geometry only, no batch dim, cached
    interaction_resnet_layers: int = 8  # geometry + events jointly, per step

    # Readout ResNet
    readout_resnet_layers: int = 16
    readout_dim: int = 48


# -----------------------------------------------------------------------
# Training / schedule hyperparameters
# -----------------------------------------------------------------------


@dataclass
class TrainConfig:
    # Optimizer
    optimizer: str = "lion"
    lr: float = 1.3e-4  # overridden per distance in run scripts
    weight_decay: float = 1e-7
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0

    # Batch schedule
    batch_init: int = 256
    batch_final: int = 1024
    batch_change_step: int = 800_000  # Scaling: 8e5

    # Gradient accumulation: number of micro-steps per optimizer step.
    # Effective batch size = batch_size * grad_accum_steps * world_size.
    # 1 = disabled (default, matches current behaviour).
    grad_accum_steps: int = 1

    # LR schedule: linear warmup then piecewise constant decay (AlphaQubit paper)
    lr_warmup_steps: int = 10_000
    lr_decay_factor: float = 0.7
    lr_decay_steps: List[int] = field(
        default_factory=lambda: [400_000, 800_000, 1_600_000]
    )

    # Loss weights
    next_stab_pred_weight: float = 0.02
    next_stab_schedule: str = "alphaqubit"  # cosine anneal after warmup
    next_stab_weight_min: float = 0.0
    next_stab_warmup_ratio: float = 0.3

    # EMA
    use_ema: bool = True
    ema_alpha: float = 1e-4  # paper: "parameter EMA constant = 0.0001"

    # Training length
    num_steps: int = 2_000_000

    # Evaluation
    eval_every: int = 5_000
    log_every: int = 50
    eval_fit_mode: str = "simple"  # "simple" | "sycamore"
    min_r2: float = 0.9
    min_intercept: float = -0.02
    eval_cycles: tuple = (3, 5, 7, 9, 11, 13, 15, 25)

    # Reproducibility
    seed: int = 42
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    num_workers: int = 4
    pin_memory: bool = True

    # Optional fine-tuning weight decay (stronger, relative to pretrained params)
    fine_tuning_weight_decay: Optional[float] = None  # 0.08 for Sycamore fine-tuning

    # Positive class weight for BCE (auto-computed from data if None)
    logical_pos_weight: Optional[float] = None


def pretrain_config(distance: int = 3) -> TrainConfig:
    """Config for the pretraining phase."""
    cfg = TrainConfig()
    cfg.lr = get_lr(distance)
    return cfg


def finetune_config(distance: int = 3) -> TrainConfig:
    """
    Config for the fine-tuning phase.
    Uses stronger weight decay relative to pretrained parameters.
    """
    cfg = TrainConfig()
    cfg.lr = get_lr(distance)
    cfg.fine_tuning_weight_decay = 0.08  # AlphaQubit Sycamore fine-tuning value
    cfg.num_steps = 200_000  # fine-tuning is shorter
    cfg.lr_decay_steps = []  # simpler schedule for fine-tuning
    return cfg
