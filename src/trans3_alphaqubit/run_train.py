#!/usr/bin/env python3
"""
run_train.py - Training entry point for AlphaQubit-like QEC Transformer

Usage:
    python run_train.py --train_data data/train.npz --val_data data/val.npz --layout data/layout.json
    python run_train.py --config config.json
    python run_train.py --resume checkpoint.pth
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn

# Import from local modules
from model import AlphaQubitLikeModel, ScatteringResidualConvBlock
from dataset import SyndromeDataset, make_loader
from collections import Counter
from parameter import ScalingConfig, ModelConfigScaling, build_scaling_config
from utils import AttentionBiasProvider, ManhattanDistanceBias
from train import train


def compute_effective_S(layout: Dict[str, Any]) -> int:
    """
    Compute the effective S (min stabilizers per cycle) from layout.

    Due to boundary effects in surface code, the first and last cycles
    may have fewer stabilizers than the middle cycles. The model's
    cycle_index builder truncates all cycles to the minimum.
    """
    cycle_ids = layout["cycle_id"]
    cycle_counts = Counter(cycle_ids)
    # Minimum number of stabilizers across all cycles
    return min(cycle_counts.values())


def get_stabilizers_for_effective_S(layout: Dict[str, Any], effective_S: int) -> list:
    """
    Get the stabilizer IDs that are present in the cycles with minimum count.
    This returns the stabilizer IDs that will actually be used at runtime.
    """
    cycle_ids = layout["cycle_id"]
    stab_ids = layout["stab_id"]

    # Find which cycle has the minimum count
    cycle_counts = Counter(cycle_ids)
    min_cycle = min(cycle_counts, key=cycle_counts.get)

    # Get stabilizer IDs for that cycle (sorted)
    stabs_in_min_cycle = sorted([
        stab_ids[i] for i in range(len(stab_ids)) if cycle_ids[i] == min_cycle
    ])

    return stabs_in_min_cycle[:effective_S]


def build_conv_block(
    distance: int,
    d_model: int,
    layout: Dict[str, Any],
    model_cfg: ModelConfigScaling,
) -> ScatteringResidualConvBlock:
    """
    Build ScatteringResidualConvBlock from layout.
    
    Args:
        distance: Code distance
        d_model: Model dimension
        layout: Layout dictionary with x, y coordinates
        model_cfg: Model configuration
    
    Returns:
        ScatteringResidualConvBlock
    """
    # Extract coordinates from layout
    x_coords = np.array(layout["x"], dtype=np.float32)
    y_coords = np.array(layout["y"], dtype=np.float32)
    
    # Build coord_to_index and index_to_coord mappings
    # For surface code, we need to map from (i, j) grid coordinates to stabilizer indices
    # The grid size is (d+1) x (d+1)
    # IMPORTANT: The ConvBlock operates on S stabilizers per cycle, not L total detectors
    # Due to boundary effects, first/last cycles may have fewer stabilizers, so we use
    # the effective S (minimum across all cycles) to match runtime behavior.
    H = W = distance + 1

    # Compute effective S (minimum stabilizers per cycle)
    effective_S = compute_effective_S(layout)
    active_stab_ids = get_stabilizers_for_effective_S(layout, effective_S)

    # Get unique stabilizer coordinates: for each active stab_id, find its (x, y) position
    stab_ids_arr = np.array(layout["stab_id"])
    num_stab = layout["num_stab"]  # Total unique stabilizers

    # Find representative coordinates for each unique stabilizer
    stab_x_all = np.zeros(num_stab, dtype=np.float32)
    stab_y_all = np.zeros(num_stab, dtype=np.float32)
    stab_found = np.zeros(num_stab, dtype=bool)

    for idx in range(len(stab_ids_arr)):
        sid = stab_ids_arr[idx]
        if not stab_found[sid]:
            stab_x_all[sid] = x_coords[idx]
            stab_y_all[sid] = y_coords[idx]
            stab_found[sid] = True

    # Only use coordinates for active stabilizers (those in all cycles)
    stab_x = np.array([stab_x_all[sid] for sid in active_stab_ids], dtype=np.float32)
    stab_y = np.array([stab_y_all[sid] for sid in active_stab_ids], dtype=np.float32)

    # Quantize coordinates to grid positions
    coord_quant = layout.get("coord_quant", 0.5)
    x_quantized = np.round(stab_x / coord_quant) * coord_quant
    y_quantized = np.round(stab_y / coord_quant) * coord_quant

    # Map to grid indices (i, j) where i, j in [0, H-1]
    # Normalize coordinates to [0, H-1] range
    x_min, x_max = x_quantized.min(), x_quantized.max()
    y_min, y_max = y_quantized.min(), y_quantized.max()

    if x_max > x_min:
        x_norm = (x_quantized - x_min) / (x_max - x_min) * (H - 1)
    else:
        x_norm = np.zeros_like(x_quantized)

    if y_max > y_min:
        y_norm = (y_quantized - y_min) / (y_max - y_min) * (W - 1)
    else:
        y_norm = np.zeros_like(y_quantized)

    # Round to nearest integer grid position
    i_coords = np.round(x_norm).astype(np.int32).clip(0, H - 1)
    j_coords = np.round(y_norm).astype(np.int32).clip(0, W - 1)

    # Build mappings for effective_S stabilizers (those present in all cycles)
    # The index here is 0..effective_S-1, matching the runtime tensor shape (B, S, D)
    coord_to_index = {}
    index_to_coord = []

    for idx in range(effective_S):
        i, j = int(i_coords[idx]), int(j_coords[idx])
        coord_to_index[(i, j)] = idx
        index_to_coord.append((i, j))
    
    # Build channels list
    channels_list = [model_cfg.conv_dim] * model_cfg.conv_layers
    
    conv_block = ScatteringResidualConvBlock(
        d=distance,
        d_d=d_model,
        L_layers=model_cfg.conv_layers,
        channels_list=channels_list,
        coord_to_index=coord_to_index,
        index_to_coord=index_to_coord,
        dilation_list=None,  # Can be customized
    )
    
    return conv_block


def build_model(
    layout: Dict[str, Any],
    model_cfg: ModelConfigScaling,
    train_cfg: ScalingConfig,
    use_full_bias: bool = True,
) -> AlphaQubitLikeModel:
    """
    Build AlphaQubitLikeModel from configuration.
    
    Args:
        layout: Layout dictionary
        model_cfg: Model configuration
        train_cfg: Training configuration
        use_full_bias: Whether to use AttentionBiasProvider (True) or ManhattanDistanceBias (False)
    
    Returns:
        AlphaQubitLikeModel
    """
    num_stab = layout["num_stab"]
    num_cycles = layout["num_cycles"]
    distance = layout["distance"]
    
    # Build conv block
    conv_block = build_conv_block(distance, model_cfg.d_model, layout, model_cfg)
    
    # Build bias provider
    if use_full_bias:
        bias_provider = AttentionBiasProvider(
            db=model_cfg.bias_dim,
            max_dist=8,
            num_residual_layers=model_cfg.bias_residual_layers,
            indicator_features=model_cfg.indicator_features,
            coord_scale=0.5,
        )
    else:
        bias_provider = ManhattanDistanceBias(
            db=model_cfg.bias_dim,
            max_dist=8,
        )
    
    # Build model
    # Note: d_mid is typically same as d_attn (key_size)
    model = AlphaQubitLikeModel(
        num_stab=num_stab,
        num_cycles=num_cycles,
        d_model=model_cfg.d_model,
        d_attn=model_cfg.key_size,
        d_mid=model_cfg.key_size,  # Typically same as key_size
        db=model_cfg.bias_dim,
        H=model_cfg.num_heads,
        n_layers=model_cfg.syndrome_layers,
        widen=model_cfg.dense_widen,
        conv_block=conv_block,
        bias_provider=bias_provider,
        use_next_stab=True,
    )
    
    return model


def load_data(
    train_path: str,
    val_path: str,
    layout_path: str,
    input_mode: str = "soft",
    mask_last_cycle: bool = False,
) -> tuple:
    """
    Load training and validation datasets.
    
    Args:
        train_path: Path to training data .npz file
        val_path: Path to validation data .npz file
        layout_path: Path to layout.json
        input_mode: "hard" or "soft"
        mask_last_cycle: Whether to mask last cycle
    
    Returns:
        (train_dataset, val_dataset)
    """
    # Load data
    train_data = np.load(train_path)
    val_data = np.load(val_path)
    
    # Extract arrays (support different key names)
    train_samples = train_data.get("det_soft", train_data.get("samples", train_data.get("det_hard")))
    train_labels = train_data.get("obs", train_data.get("labels", train_data.get("logical_labels")))
    train_leakage = train_data.get("leakage_mask", None)  # Optional leakage mask
    
    val_samples = val_data.get("det_soft", val_data.get("samples", val_data.get("det_hard")))
    val_labels = val_data.get("obs", val_data.get("labels", val_data.get("logical_labels")))
    val_leakage = val_data.get("leakage_mask", None)  # Optional leakage mask
    
    # Create datasets
    train_dataset = SyndromeDataset(
        samples=train_samples,
        labels=train_labels,
        layout_json_path=layout_path,
        input_mode=input_mode,
        mask_last_cycle=mask_last_cycle,
        leakage_mask=train_leakage,
    )
    
    val_dataset = SyndromeDataset(
        samples=val_samples,
        labels=val_labels,
        layout_json_path=layout_path,
        input_mode=input_mode,
        mask_last_cycle=mask_last_cycle,
        leakage_mask=val_leakage,
    )
    
    return train_dataset, val_dataset


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """
    Load checkpoint.
    
    Returns:
        Dictionary with checkpoint info (step, best_ler, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint


def save_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    best_ler: float = float("inf"),
    **kwargs,
):
    """Save checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "best_ler": best_ler,
        **kwargs,
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AlphaQubit-like QEC Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data paths
    parser.add_argument("--train_data", type=str, help="Path to training data .npz file")
    parser.add_argument("--val_data", type=str, help="Path to validation data .npz file")
    parser.add_argument("--layout", type=str, help="Path to layout.json file")
    
    # Configuration
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    # Model options
    parser.add_argument("--use_full_bias", action="store_true", default=True,
                       help="Use AttentionBiasProvider (default) or ManhattanDistanceBias")
    parser.add_argument("--input_mode", type=str, default="soft", choices=["hard", "soft"],
                       help="Input mode: hard (0/1) or soft (continuous)")
    parser.add_argument("--mask_last_cycle", action="store_true",
                       help="Mask last cycle to avoid label leakage")
    
    # Training options
    parser.add_argument("--run_name", type=str, default="scaling_run",
                       help="Run name for wandb/logging")
    parser.add_argument("--use_wandb", action="store_true", default=True,
                       help="Use wandb for logging")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    
    # Override config options
    parser.add_argument("--num_steps", type=int, help="Number of training steps")
    parser.add_argument("--batch_init", type=int, help="Initial batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--eval_every", type=int, help="Evaluate every N steps")
    parser.add_argument("--eval_fit_mode", type=str, choices=["sycamore", "simple"],
                       default="simple", help="Evaluation mode: sycamore (fit across cycles) or simple (direct LER)")

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        train_cfg = ScalingConfig(**config_dict.get("training", {}))
        model_cfg = ModelConfigScaling(**config_dict.get("model", {}))
    else:
        train_cfg = build_scaling_config()
        model_cfg = ModelConfigScaling()
    
    # Override with command line arguments
    if args.num_steps:
        train_cfg.num_steps = args.num_steps
    if args.batch_init:
        train_cfg.batch_init = args.batch_init
    if args.lr:
        train_cfg.lr = args.lr
    if args.seed:
        train_cfg.seed = args.seed
    if args.eval_every:
        train_cfg.eval_every = args.eval_every
    if args.eval_fit_mode:
        train_cfg.eval_fit_mode = args.eval_fit_mode

    # Load layout
    if args.layout:
        with open(args.layout, "r") as f:
            layout = json.load(f)
        # Ensure distance is in layout (required for conv_block)
        if "distance" not in layout:
            # Try to infer from num_stab or other fields
            # For surface code: num_stab ≈ distance^2
            if "num_stab" in layout:
                import math
                layout["distance"] = int(math.sqrt(layout["num_stab"]))
                print(f"Warning: distance not in layout, inferred as {layout['distance']}")
            else:
                raise ValueError("layout.json must contain 'distance' field")
    else:
        raise ValueError("--layout is required")
    
    # Load datasets
    if args.train_data and args.val_data:
        train_dataset, val_dataset = load_data(
            args.train_data,
            args.val_data,
            args.layout,
            input_mode=args.input_mode,
            mask_last_cycle=args.mask_last_cycle,
        )
    else:
        raise ValueError("--train_data and --val_data are required")
    
    # Build model
    model = build_model(
        layout=layout,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        use_full_bias=args.use_full_bias,
    )
    
    # Load checkpoint if resuming
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        load_checkpoint(args.resume, model)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training
    print(f"Starting training with {train_cfg.num_steps} steps...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=train_cfg,
        run_name=args.run_name,
        use_wandb=args.use_wandb,
    )
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.output_dir, f"{args.run_name}_final.pth")
    save_checkpoint(
        final_checkpoint_path,
        model,
        step=train_cfg.num_steps,
        best_ler=best["ler"],
        fit_r2=best.get("fit_r2"),
        fit_intercept=best.get("fit_intercept"),
    )
    
    print(f"\nTraining completed!")
    print(f"Best LER: {best['ler']:.6f} at step {best['step']}")
    print(f"Final checkpoint saved to {final_checkpoint_path}")


if __name__ == "__main__":
    main()
