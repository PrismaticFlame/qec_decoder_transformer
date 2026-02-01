#!/usr/bin/env python3
"""
run_train.py - Train single-basis AlphaQubit-like QEC Transformer (trans5)

Changes from trans4:
  - ReadoutResNet replaces simple Linear classifier
  - Default d_model = 128 (up from trans4's 64 / parameter.py's 256)
  - AlphaQubit next-stab cosine annealing schedule enabled by default

Usage:
    python run_train.py --basis z --distance 3
    python run_train.py --bases z x --distance 3 --num_steps 50000
    python run_train.py --bases z x --distances 3 5 7

Data is read from:
    {data_dir}/{basis}_basis/d{distance}_r{rounds}/train.npz
    {data_dir}/{basis}_basis/d{distance}_r{rounds}/val.npz
    {data_dir}/{basis}_basis/d{distance}_r{rounds}/layout.json

Checkpoints saved to:
    checkpoints/{basis}_d{distance}_r{rounds}.pth
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn

# IMPORTANT: Import local (trans5) model BEFORE adding trans3 to sys.path.
# This ensures we get the trans5 model with ReadoutResNet.
# model.py adds trans3 to sys.path internally for its own dependencies.
from model import AlphaQubitLikeModel, ScatteringResidualConvBlock

# Add trans3_alphaqubit to path for remaining shared modules
_trans3_dir = str(Path(__file__).parent.parent / "trans3_alphaqubit")
if _trans3_dir not in sys.path:
    sys.path.insert(0, _trans3_dir)

# Import shared modules from trans3
from dataset import SyndromeDataset, make_loader
from parameter import ScalingConfig, ModelConfigScaling, build_scaling_config
from utils import AttentionBiasProvider, ManhattanDistanceBias
from train import train


def compute_effective_S(layout: Dict[str, Any]) -> int:
    """S_max: the maximum number of stabilizers in any cycle."""
    cycle_ids = layout["cycle_id"]
    cycle_counts = Counter(cycle_ids)
    return max(cycle_counts.values())


def get_stabilizers_for_effective_S(layout: Dict[str, Any], effective_S: int) -> list:
    """Get the stabilizer IDs from the cycle with the most detectors."""
    cycle_ids = layout["cycle_id"]
    stab_ids = layout["stab_id"]
    cycle_counts = Counter(cycle_ids)
    max_cycle = max(cycle_counts, key=cycle_counts.get)
    stabs_in_max_cycle = sorted([
        stab_ids[i] for i in range(len(stab_ids)) if cycle_ids[i] == max_cycle
    ])
    return stabs_in_max_cycle[:effective_S]


def build_conv_block(
    distance: int,
    d_model: int,
    layout: Dict[str, Any],
    model_cfg: ModelConfigScaling,
) -> ScatteringResidualConvBlock:
    x_coords = np.array(layout["x"], dtype=np.float32)
    y_coords = np.array(layout["y"], dtype=np.float32)

    H = W = distance + 1

    effective_S = compute_effective_S(layout)
    active_stab_ids = get_stabilizers_for_effective_S(layout, effective_S)

    stab_ids_arr = np.array(layout["stab_id"])
    num_stab = layout["num_stab"]

    stab_x_all = np.zeros(num_stab, dtype=np.float32)
    stab_y_all = np.zeros(num_stab, dtype=np.float32)
    stab_found = np.zeros(num_stab, dtype=bool)

    for idx in range(len(stab_ids_arr)):
        sid = stab_ids_arr[idx]
        if not stab_found[sid]:
            stab_x_all[sid] = x_coords[idx]
            stab_y_all[sid] = y_coords[idx]
            stab_found[sid] = True

    stab_x = np.array([stab_x_all[sid] for sid in active_stab_ids], dtype=np.float32)
    stab_y = np.array([stab_y_all[sid] for sid in active_stab_ids], dtype=np.float32)

    coord_quant = layout.get("coord_quant", 0.5)
    x_quantized = np.round(stab_x / coord_quant) * coord_quant
    y_quantized = np.round(stab_y / coord_quant) * coord_quant

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

    i_coords = np.round(x_norm).astype(np.int32).clip(0, H - 1)
    j_coords = np.round(y_norm).astype(np.int32).clip(0, W - 1)

    coord_to_index = {}
    index_to_coord = []

    for idx in range(effective_S):
        i, j = int(i_coords[idx]), int(j_coords[idx])
        coord_to_index[(i, j)] = idx
        index_to_coord.append((i, j))

    channels_list = [model_cfg.conv_dim] * model_cfg.conv_layers

    conv_block = ScatteringResidualConvBlock(
        d=distance,
        d_d=d_model,
        L_layers=model_cfg.conv_layers,
        channels_list=channels_list,
        coord_to_index=coord_to_index,
        index_to_coord=index_to_coord,
        dilation_list=None,
    )

    return conv_block


def build_model(
    layout: Dict[str, Any],
    model_cfg: ModelConfigScaling,
    train_cfg: ScalingConfig,
    use_full_bias: bool = True,
) -> AlphaQubitLikeModel:
    num_stab = layout["num_stab"]
    num_cycles = layout["num_cycles"]
    distance = layout["distance"]

    conv_block = build_conv_block(distance, model_cfg.d_model, layout, model_cfg)

    # Extract coord_to_index from conv_block for the ReadoutResNet
    coord_to_index = conv_block.coord_to_index

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

    model = AlphaQubitLikeModel(
        num_stab=num_stab,
        num_cycles=num_cycles,
        d_model=model_cfg.d_model,
        d_attn=model_cfg.key_size,
        d_mid=model_cfg.key_size,
        db=model_cfg.bias_dim,
        H=model_cfg.num_heads,
        n_layers=model_cfg.syndrome_layers,
        widen=model_cfg.dense_widen,
        conv_block=conv_block,
        bias_provider=bias_provider,
        use_next_stab=True,
        # ReadoutResNet parameters
        readout_dim=model_cfg.readout_dim,
        readout_resnet_layers=model_cfg.readout_resnet_layers,
        distance=distance,
        coord_to_index=coord_to_index,
    )

    return model


def load_data(
    train_path: str,
    val_path: str,
    layout_path: str,
    input_mode: str = "hard",
) -> tuple:
    train_data = np.load(train_path)
    val_data = np.load(val_path)

    train_samples = train_data.get("det_soft", train_data.get("det_hard"))
    train_labels = train_data.get("obs", train_data.get("labels"))

    val_samples = val_data.get("det_soft", val_data.get("det_hard"))
    val_labels = val_data.get("obs", val_data.get("labels"))

    train_dataset = SyndromeDataset(
        samples=train_samples,
        labels=train_labels,
        layout_json_path=layout_path,
        input_mode=input_mode,
    )

    val_dataset = SyndromeDataset(
        samples=val_samples,
        labels=val_labels,
        layout_json_path=layout_path,
        input_mode=input_mode,
    )

    return train_dataset, val_dataset


def save_training_history(history: Dict, checkpoint_dir: Path, run_name: str):
    """Save training history to CSV files and generate a training plot."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save training loss CSV
    if history.get("train"):
        loss_csv = checkpoint_dir / f"{run_name}_loss.csv"
        with open(loss_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "loss", "loss_main", "lr"])
            writer.writeheader()
            writer.writerows(history["train"])
        print(f"  Loss history saved to {loss_csv}")

    # Save eval LER CSV
    if history.get("eval"):
        eval_csv = checkpoint_dir / f"{run_name}_eval.csv"
        with open(eval_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "dev_ler", "fit_ok", "best_ler"])
            writer.writeheader()
            writer.writerows(history["eval"])
        print(f"  Eval history saved to {eval_csv}")

    # Generate plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(14, 6))

        # Training loss (left y-axis)
        if history.get("train"):
            steps = [r["step"] for r in history["train"]]
            losses = [r["loss"] for r in history["train"]]

            ax1.plot(steps, losses, alpha=0.2, color="tab:blue", linewidth=0.5)

            # Smoothed loss
            if len(losses) > 20:
                window = min(50, len(losses) // 5)
                smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
                smooth_steps = steps[window - 1:]
                ax1.plot(smooth_steps, smoothed, color="tab:blue", linewidth=2,
                         label=f"Train loss (smoothed, w={window})")
            else:
                ax1.plot(steps, losses, color="tab:blue", linewidth=2, label="Train loss")

        ax1.set_xlabel("Step")
        ax1.set_ylabel("Training Loss", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # Validation LER (right y-axis)
        if history.get("eval"):
            ax2 = ax1.twinx()
            eval_steps = [r["step"] for r in history["eval"]]
            eval_lers = [r["dev_ler"] for r in history["eval"]]
            best_lers = [r["best_ler"] for r in history["eval"]]

            ax2.plot(eval_steps, eval_lers, "o-", color="tab:red", markersize=3,
                     linewidth=1, alpha=0.7, label="Val LER")
            ax2.plot(eval_steps, best_lers, "--", color="tab:orange", linewidth=1.5,
                     label="Best LER so far")
            ax2.set_ylabel("Validation LER", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")

            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        else:
            ax1.legend(loc="upper right")

        ax1.set_title(f"Training Progress: {run_name}")
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = checkpoint_dir / f"{run_name}_training.png"
        fig.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        print(f"  Training plot saved to {plot_path}")
    except ImportError:
        print("  matplotlib not available, skipping plot generation")


def resolve_data_path(data_dir: Path, basis: str, distance: int, rounds_multiplier: int) -> Path:
    """Resolve the data path for a given basis and distance."""
    rounds = distance * rounds_multiplier
    return data_dir / f"{basis}_basis" / f"d{distance}_r{rounds}"


def train_single(
    basis: str,
    distance: int,
    data_dir: Path,
    checkpoint_dir: Path,
    rounds_multiplier: int = 2,
    train_cfg: Optional[ScalingConfig] = None,
    model_cfg: Optional[ModelConfigScaling] = None,
    use_full_bias: bool = True,
    input_mode: str = "hard",
    use_wandb: bool = False,
) -> Dict[str, Any]:
    """Train a single model for one (basis, distance) pair."""
    if train_cfg is None:
        train_cfg = build_scaling_config()
    if model_cfg is None:
        model_cfg = ModelConfigScaling()

    dataset_path = resolve_data_path(data_dir, basis, distance, rounds_multiplier)
    rounds = distance * rounds_multiplier

    train_path = str(dataset_path / "train.npz")
    val_path = str(dataset_path / "val.npz")
    layout_path = str(dataset_path / "layout.json")

    # Verify files exist
    for p in [train_path, val_path, layout_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    print(f"\n{'='*60}")
    print(f"Training: {basis.upper()}-basis  d={distance}  rounds={rounds}")
    print(f"  Data: {dataset_path}")
    print(f"{'='*60}")

    # Load layout
    with open(layout_path, "r") as f:
        layout = json.load(f)
    if "distance" not in layout:
        layout["distance"] = distance

    # Auto-compute pos_weight from data if not already set
    if not hasattr(train_cfg, "logical_pos_weight") or getattr(train_cfg, "logical_pos_weight", None) is None:
        info_path = dataset_path / "info.json"
        if info_path.exists():
            with open(info_path, "r") as f:
                info = json.load(f)
            obs_rate = info.get("obs_rate")
            if obs_rate is not None and 0 < obs_rate < 1:
                auto_pw = (1.0 - obs_rate) / obs_rate
                train_cfg.logical_pos_weight = auto_pw
                print(f"  Auto pos_weight: {auto_pw:.2f} (obs_rate={obs_rate:.4f})")

    # Load datasets
    train_dataset, val_dataset = load_data(
        train_path, val_path, layout_path,
        input_mode=input_mode,
    )

    # Build model
    model = build_model(layout, model_cfg, train_cfg, use_full_bias=use_full_bias)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    print(f"  Device: {device}")

    # Train
    run_name = f"{basis}_d{distance}_r{rounds}"
    start_time = time.time()

    best = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=train_cfg,
        run_name=run_name,
        use_wandb=use_wandb,
    )

    train_time = time.time() - start_time

    # Save checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_name}.pth"

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "best_ler": best["ler"],
        "best_step": best["step"],
        "fit_r2": best.get("fit_r2"),
        "fit_intercept": best.get("fit_intercept"),
        "basis": basis,
        "distance": distance,
        "rounds": rounds,
        "train_time": train_time,
    }, checkpoint_path)

    print(f"\n  Best LER: {best['ler']:.6f} at step {best['step']}")
    print(f"  Checkpoint: {checkpoint_path}")

    # Save training history and generate plot
    history = best.get("history")
    if history:
        save_training_history(history, checkpoint_dir, run_name)

    return {
        "basis": basis,
        "distance": distance,
        "rounds": rounds,
        "best_ler": best["ler"],
        "best_step": best["step"],
        "train_time": train_time,
        "checkpoint_path": str(checkpoint_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train single-basis AlphaQubit-like QEC Transformer (trans5: ReadoutResNet + next-stab schedule)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # What to train
    parser.add_argument("--basis", type=str, choices=["z", "x"],
                        help="Single basis to train")
    parser.add_argument("--bases", type=str, nargs="+", choices=["z", "x"],
                        help="Multiple bases to train sequentially")
    parser.add_argument("--distance", type=int,
                        help="Single code distance")
    parser.add_argument("--distances", type=int, nargs="+",
                        help="Multiple code distances")
    parser.add_argument("--rounds_multiplier", type=int, default=2,
                        help="rounds = distance * multiplier")

    # Data
    parser.add_argument("--data_dir", type=str, default="../prop_data_gen/data",
                        help="Root data directory (relative to this script)")
    parser.add_argument("--input_mode", type=str, default="hard", choices=["hard", "soft"],
                        help="Input mode: hard (binary) or soft (continuous)")

    # Training overrides
    parser.add_argument("--num_steps", type=int, default=500,
                        help="Training steps per model")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate override")
    parser.add_argument("--weight_decay", type=float,
                        help="Weight decay (default: 1e-7)")
    parser.add_argument("--lr_decay_steps", type=int, nargs="+",
                        help="Steps at which to decay LR (default: 400000 800000 1600000)")
    parser.add_argument("--eval_every", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--eval_fit_mode", type=str, default="simple",
                        choices=["sycamore", "simple"],
                        help="Evaluation mode")
    parser.add_argument("--pos_weight", type=float,
                        help="Positive class weight for BCE loss (auto-computed from data if not set)")

    # Model architecture overrides
    parser.add_argument("--d_model", type=int,
                        help="Model hidden dimension (default: 128 for trans5)")
    parser.add_argument("--syndrome_layers", type=int,
                        help="Number of transformer layers (default: 3)")
    parser.add_argument("--num_heads", type=int,
                        help="Number of attention heads (default: 4)")
    parser.add_argument("--dense_widen", type=int,
                        help="FFN width multiplier (default: 5)")
    parser.add_argument("--conv_dim", type=int,
                        help="Convolution channel dimension (default: 128)")
    parser.add_argument("--conv_layers", type=int,
                        help="Number of convolution layers (default: 3)")
    parser.add_argument("--key_size", type=int,
                        help="Attention key dimension (default: 32)")

    # ReadoutResNet overrides
    parser.add_argument("--readout_dim", type=int,
                        help="ReadoutResNet hidden dimension (default: 48)")
    parser.add_argument("--readout_resnet_layers", type=int,
                        help="Number of ReadoutResNet residual blocks (default: 16)")

    # Next-stab auxiliary loss schedule
    parser.add_argument("--next_stab_schedule", type=str, default="alphaqubit",
                        choices=["none", "alphaqubit"],
                        help="Schedule for next-stab weight annealing")
    parser.add_argument("--next_stab_weight_min", type=float, default=0.0,
                        help="Minimum next-stab weight after annealing")
    parser.add_argument("--next_stab_warmup_ratio", type=float, default=0.3,
                        help="Fraction of training to keep full next-stab weight")

    # Output
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory for saving checkpoints (relative to this script)")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--use_full_bias", action="store_true", default=True,
                        help="Use full AttentionBiasProvider")

    args = parser.parse_args()

    # Resolve bases
    bases = args.bases or ([args.basis] if args.basis else None)
    if bases is None:
        parser.error("Must specify --basis or --bases")

    # Resolve distances
    distances = args.distances or ([args.distance] if args.distance else None)
    if distances is None:
        parser.error("Must specify --distance or --distances")

    # Resolve paths relative to script
    script_dir = Path(__file__).parent
    data_dir = (script_dir / args.data_dir).resolve()
    checkpoint_dir = (script_dir / args.checkpoint_dir).resolve()

    if not data_dir.exists():
        parser.error(f"Data directory not found: {data_dir}")

    # Build training config
    train_cfg = build_scaling_config()
    train_cfg.num_steps = args.num_steps
    train_cfg.batch_init = args.batch_size
    train_cfg.eval_every = args.eval_every
    train_cfg.eval_fit_mode = args.eval_fit_mode
    if args.lr:
        train_cfg.lr = args.lr
    if args.weight_decay is not None:
        train_cfg.weight_decay = args.weight_decay
    if args.lr_decay_steps is not None:
        train_cfg.lr_decay_steps = args.lr_decay_steps

    # Set next-stab schedule on train_cfg so trans3's train.py picks it up
    # (train.py reads these via getattr(cfg, "next_stab_schedule", "none"))
    train_cfg.next_stab_schedule = args.next_stab_schedule
    train_cfg.next_stab_weight_min = args.next_stab_weight_min
    train_cfg.next_stab_warmup_ratio = args.next_stab_warmup_ratio

    # Set pos_weight if explicitly provided (otherwise auto-computed per-model
    # from info.json in train_single)
    if args.pos_weight is not None:
        train_cfg.logical_pos_weight = args.pos_weight

    # Build model config with trans5 defaults
    model_cfg = ModelConfigScaling()
    # Trans5 default: d_model=128 (compromise between paper's 256 and trans4's 64)
    if args.d_model is None:
        model_cfg.d_model = 128
    else:
        model_cfg.d_model = args.d_model
    if args.syndrome_layers is not None:
        model_cfg.syndrome_layers = args.syndrome_layers
    if args.num_heads is not None:
        model_cfg.num_heads = args.num_heads
    if args.dense_widen is not None:
        model_cfg.dense_widen = args.dense_widen
    if args.conv_dim is not None:
        model_cfg.conv_dim = args.conv_dim
    if args.conv_layers is not None:
        model_cfg.conv_layers = args.conv_layers
    if args.key_size is not None:
        model_cfg.key_size = args.key_size
    if args.readout_dim is not None:
        model_cfg.readout_dim = args.readout_dim
    if args.readout_resnet_layers is not None:
        model_cfg.readout_resnet_layers = args.readout_resnet_layers

    # Train all combinations
    results = []
    for basis in bases:
        for distance in distances:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            result = train_single(
                basis=basis,
                distance=distance,
                data_dir=data_dir,
                checkpoint_dir=checkpoint_dir,
                rounds_multiplier=args.rounds_multiplier,
                train_cfg=train_cfg,
                model_cfg=model_cfg,
                use_full_bias=args.use_full_bias,
                input_mode=args.input_mode,
                use_wandb=args.use_wandb,
            )
            results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Basis':<8} {'Distance':>8} {'Rounds':>8} {'LER':>12} {'Step':>8}")
    print(f"{'-'*52}")
    for r in results:
        print(f"{r['basis'].upper():<8} {r['distance']:>8} {r['rounds']:>8} "
              f"{r['best_ler']:>12.6f} {r['best_step']:>8}")
    print(f"{'='*60}")

    # Save results index
    results_path = checkpoint_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
