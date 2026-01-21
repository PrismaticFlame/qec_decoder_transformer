# 從各個子模組匯入核心類別，建立捷徑
from .model import AlphaQubitLikeModel, RNNCore
from .parameter import ModelConfig, ScalingConfig, build_model_config, build_scaling_config
from .ema import EMA
from .dataset import SyndromeDataset, make_loader
from .utils import ManhattanDistanceBias, AttentionBiasProvider, Lion
from .eval import (
    compute_ler_from_logits,
    compute_cycle_ler,
    evaluate_ler_with_fit,
    evaluate_model,
    evaluate_test_set,
)

# Aliases for backward compatibility
SyndromeRNNDecoder = AlphaQubitLikeModel
DecoderConfig = ModelConfig
config = build_model_config()
build_qec_dataloader = make_loader

__all__ = [
    "AlphaQubitLikeModel",
    "SyndromeRNNDecoder",  # alias
    "RNNCore",
    "ModelConfig",
    "DecoderConfig",  # alias
    "ScalingConfig",
    "build_model_config",
    "build_scaling_config",
    "config",  # alias
    "EMA",
    "SyndromeDataset",
    "make_loader",
    "build_qec_dataloader",  # alias
    "ManhattanDistanceBias",
    "AttentionBiasProvider",
    "Lion",
    # Evaluation functions
    "compute_ler_from_logits",
    "compute_cycle_ler",
    "evaluate_ler_with_fit",
    "evaluate_model",
    "evaluate_test_set",
]