#!/usr/bin/env python3
"""
eval_combine.py - Evaluate separate X/Z models and compute combined LER

Loads the Z-basis and X-basis model checkpoints, evaluates each on its
respective validation data, and averages the LERs (following AlphaQubit).

Usage:
    python eval_combine.py --distance 3
    python eval_combine.py --distance 3 --d_model 64
    python eval_combine.py --distance 3 --data_dir ../prop_data_gen/data
    python eval_combine.py --distances 3 5 7
    python eval_combine.py --distance 3 \
        --z_checkpoint checkpoints/z_d3_r6.pth \
        --x_checkpoint checkpoints/x_d3_r6.pth
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn

# IMPORTANT: Import local (trans4) model and run_train BEFORE adding trans3 to
# sys.path. This ensures we get the single-classifier model from trans4, not
# trans3's dual-head version (which has classifier_x / classifier_z).
from model import AlphaQubitLikeModel
from run_train import build_model, load_data

# Add trans3_alphaqubit to path for remaining shared modules
_trans3_dir = str(Path(__file__).parent.parent / "trans3_alphaqubit")
if _trans3_dir not in sys.path:
    sys.path.insert(0, _trans3_dir)

from dataset import SyndromeDataset, make_loader
from parameter import ScalingConfig, ModelConfigScaling, build_scaling_config
from eval import compute_ler_from_logits


@torch.no_grad()
def evaluate_basis(
    model: nn.Module,
    val_loader,
    device: torch.device,
) -> float:
    """Evaluate a single-basis model and return its LER."""
    model.eval()

    all_logits = []
    all_labels = []

    for batch in val_loader:
        if isinstance(batch, dict):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            labels = batch.get("logical_labels", batch.get("label"))
        else:
            raise TypeError("Expected dict batch")

        out = model(batch)
        logits = out["logical_logits"]

        all_logits.append(logits.detach())
        all_labels.append(labels.view(-1).detach())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    ler = compute_ler_from_logits(logits_cat, labels_cat)

    return ler


def evaluate_distance(
    distance: int,
    data_dir: Path,
    checkpoint_dir: Path,
    rounds_multiplier: int = 2,
    batch_size: int = 128,
    input_mode: str = "hard",
    model_cfg: Optional[ModelConfigScaling] = None,
) -> Dict[str, Any]:
    """Evaluate both X and Z models for a given distance and combine."""
    rounds = distance * rounds_multiplier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for basis in ["z", "x"]:
        # Resolve paths
        dataset_path = data_dir / f"{basis}_basis" / f"d{distance}_r{rounds}"
        checkpoint_path = checkpoint_dir / f"{basis}_d{distance}_r{rounds}.pth"

        val_path = str(dataset_path / "val.npz")
        layout_path = str(dataset_path / "layout.json")

        if not checkpoint_path.exists():
            print(f"  Warning: checkpoint not found: {checkpoint_path}")
            results[f"{basis}_ler"] = float("nan")
            continue

        if not dataset_path.exists():
            print(f"  Warning: data not found: {dataset_path}")
            results[f"{basis}_ler"] = float("nan")
            continue

        # Load layout
        with open(layout_path, "r") as f:
            layout = json.load(f)
        if "distance" not in layout:
            layout["distance"] = distance

        # Load checkpoint first to check for saved model config
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Determine model config priority: checkpoint saved > CLI args > defaults
        saved_cfg = checkpoint.get("model_cfg")
        if saved_cfg is not None:
            cfg = ModelConfigScaling(**saved_cfg)
            print(f"  Using model config from checkpoint (d_model={cfg.d_model})")
        elif model_cfg is not None:
            cfg = model_cfg
            print(f"  Using CLI model config (d_model={cfg.d_model})")
        else:
            cfg = ModelConfigScaling()
            print(f"  Using default model config (d_model={cfg.d_model})")

        # Build model
        train_cfg = build_scaling_config()
        model = build_model(layout, cfg, train_cfg, use_full_bias=True)
        model = model.to(device)

        # Load weights (strict=False to handle old trans3 checkpoints that
        # have extra classifier_x / classifier_z keys)
        load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if load_result.unexpected_keys:
            print(f"  Note: Ignored extra checkpoint keys: {load_result.unexpected_keys}")
        if load_result.missing_keys:
            print(f"  WARNING: Missing keys in checkpoint: {load_result.missing_keys}")

        # Load validation data
        val_data = np.load(val_path)
        val_samples = val_data.get("det_soft", val_data.get("det_hard"))
        val_labels = val_data.get("obs", val_data.get("labels"))

        val_dataset = SyndromeDataset(
            samples=val_samples,
            labels=val_labels,
            layout_json_path=layout_path,
            input_mode=input_mode,
        )

        val_loader = make_loader(val_dataset, batch_size, train_cfg, shuffle=False)

        # Evaluate
        ler = evaluate_basis(model, val_loader, device)
        results[f"{basis}_ler"] = ler

        ckpt_info = checkpoint.get("best_step", "?")
        print(f"  {basis.upper()}-basis LER: {ler:.6f}  (checkpoint step: {ckpt_info})")

    # Combine
    z_ler = results.get("z_ler", float("nan"))
    x_ler = results.get("x_ler", float("nan"))

    if not (np.isnan(z_ler) or np.isnan(x_ler)):
        combined_ler = (z_ler + x_ler) / 2.0
    else:
        combined_ler = float("nan")

    results["combined_ler"] = combined_ler
    results["distance"] = distance
    results["rounds"] = rounds

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate separate X/Z models and compute combined LER",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--distance", type=int,
                        help="Single code distance to evaluate")
    parser.add_argument("--distances", type=int, nargs="+",
                        help="Multiple code distances to evaluate")
    parser.add_argument("--rounds_multiplier", type=int, default=2,
                        help="rounds = distance * multiplier")

    parser.add_argument("--data_dir", type=str, default="../prop_data_gen/data",
                        help="Root data directory (relative to this script)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Checkpoint directory (relative to this script)")

    parser.add_argument("--z_checkpoint", type=str,
                        help="Override: explicit path to Z-basis checkpoint")
    parser.add_argument("--x_checkpoint", type=str,
                        help="Override: explicit path to X-basis checkpoint")

    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for evaluation")
    parser.add_argument("--input_mode", type=str, default="hard",
                        choices=["hard", "soft"],
                        help="Input mode")

    # Model architecture overrides (fallback when checkpoint doesn't contain config)
    parser.add_argument("--d_model", type=int,
                        help="Model hidden dimension (default: 256)")
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

    args = parser.parse_args()

    # Resolve distances
    distances = args.distances or ([args.distance] if args.distance else None)
    if distances is None:
        parser.error("Must specify --distance or --distances")

    # Resolve paths relative to script
    script_dir = Path(__file__).parent
    data_dir = (script_dir / args.data_dir).resolve()
    checkpoint_dir = (script_dir / args.checkpoint_dir).resolve()

    # Build model_cfg from CLI args (used as fallback if checkpoint
    # doesn't contain a saved model config)
    has_model_overrides = any(
        getattr(args, attr) is not None
        for attr in ["d_model", "syndrome_layers", "num_heads", "dense_widen",
                      "conv_dim", "conv_layers", "key_size"]
    )

    model_cfg = None
    if has_model_overrides:
        model_cfg = ModelConfigScaling()
        if args.d_model is not None:
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

    all_results = []

    for distance in distances:
        print(f"\n{'='*60}")
        print(f"Evaluating: d={distance}")
        print(f"{'='*60}")

        results = evaluate_distance(
            distance=distance,
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            rounds_multiplier=args.rounds_multiplier,
            batch_size=args.batch_size,
            input_mode=args.input_mode,
            model_cfg=model_cfg,
        )
        all_results.append(results)

    # Print summary table
    print(f"\n{'='*60}")
    print("COMBINED EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"{'Distance':>8} {'Rounds':>8} {'Z LER':>12} {'X LER':>12} {'Combined':>12}")
    print(f"{'-'*56}")

    for r in all_results:
        z_str = f"{r['z_ler']:.6f}" if not np.isnan(r['z_ler']) else "N/A"
        x_str = f"{r['x_ler']:.6f}" if not np.isnan(r['x_ler']) else "N/A"
        c_str = f"{r['combined_ler']:.6f}" if not np.isnan(r['combined_ler']) else "N/A"
        print(f"{r['distance']:>8} {r['rounds']:>8} {z_str:>12} {x_str:>12} {c_str:>12}")

    print(f"{'='*60}")

    # Check if LER decreases with distance (the key metric)
    if len(all_results) >= 2:
        valid = [r for r in all_results if not np.isnan(r['combined_ler'])]
        if len(valid) >= 2:
            valid.sort(key=lambda r: r['distance'])
            improving = all(
                valid[i]['combined_ler'] >= valid[i+1]['combined_ler']
                for i in range(len(valid) - 1)
            )
            if improving:
                print("LER decreases with distance (expected behavior)")
            else:
                print("WARNING: LER does NOT decrease with distance")

    # Save results
    results_path = checkpoint_dir / "eval_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
