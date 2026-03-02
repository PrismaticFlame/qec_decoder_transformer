from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch


@dataclass
class ModelConfigScaling:
    # feature embedding
    resnet_layers: int = 2

    # syndrome transformer
    syndrome_layers: int = 3

    d_model_sycamore: int = 320
    d_model_scaling: int = 256

    num_heads: int = 4
    key_size: int = 32
    convolution_layers: int = 3

    convolution_dimensions_sycamore: int = 160
    convolution_dimensions_scaling: int = 128

    dense_block_widening: int = 5

    # attention bias
    bias_dimensions: int = 48
    bias_residual_layers: int = 8
    indicator_features: int = 7

    # readout resnet
    readout_resnet_layers: int = 16
    readout_dimensions: int = 48


@dataclass
class ScalingConfig:
    # optimizer and schedule
    optimizer: str = "lion"
    lr: float = 1.3e-4

    weight_decay_sycamore: float = 1e-5
    weight_decay_scaling: float = 1e-7

    fine_tuning_weight_decay: float = 0.08

    beta2: float = 0.95

    batch_init: int = 256
    batch_final: int = 1024

    batch_change_step_sycamore: int = 4_000_000
    batch_change_step_scaling: int = 800_000

    lr_decay_factor: float = 0.7

    lr_decay_steps_sycamore: List[int] = \
        field(default_factory=lambda: [80_000, 200_000, 400_000, 1_000_000, 2_000_000])
    lr_decay_steps_scaling: List[int] = \
        field(default_factory=lambda: [400_000, 800_000, 1_600_000])
    
    grad_clip_norm: float = 1.0

    # loss weights
    next_stab_pred_weight: float = 0.02
    parameter_exp_moving_avg_const: float = 0.0001

    


