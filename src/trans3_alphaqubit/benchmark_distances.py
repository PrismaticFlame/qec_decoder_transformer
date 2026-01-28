#!/usr/bin/env python3
"""
benchmark_distances.py - Generate datasets, train models, and benchmark performance
across different code distances and dataset sizes.

Usage:
    python benchmark_distances.py --generate   # Generate all datasets
    python benchmark_distances.py --train      # Train on all datasets
    python benchmark_distances.py --plot       # Plot results
    python benchmark_distances.py --all        # Do everything
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import subprocess
import time

import numpy as np
import matplotlib.pyplot as plt


# Configuration
DISTANCES = [3, 5, 7]
SHOTS_LIST = [20000, 50000, 100000]
ROUNDS_MULTIPLIER = 2  # rounds = distance * multiplier
ERROR_RATE = 0.005  # 0.5% physical error rate - below threshold to see LER decrease with distance
TRAIN_SPLIT = 0.8

# Training config
TRAIN_STEPS = 500
BATCH_SIZE = 128
EVAL_EVERY = 100

# Batch size per distance (larger distances need smaller batches to fit in memory)
BATCH_SIZE_BY_DISTANCE = {
    3: 128,
    5: 64,
    7: 32,
    9: 16,
    11: 8,
}


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent.parent.parent / "data" / "benchmark"


def get_dataset_name(distance: int, shots: int) -> str:
    """Generate dataset name."""
    return f"d{distance}_shots{shots//1000}k"


def generate_dataset(distance: int, shots: int, output_dir: Path) -> Dict[str, Any]:
    """Generate a single dataset with given parameters."""
    from gen_soft_surrogate import gen_soft_surrogate_dataset
    from layout import build_layout_from_circuit, save_layout_json

    rounds = distance * ROUNDS_MULTIPLIER
    name = get_dataset_name(distance, shots)

    print(f"\n{'='*60}")
    print(f"Generating dataset: {name}")
    print(f"  Distance: {distance}, Rounds: {rounds}, Shots: {shots}")
    print(f"{'='*60}")

    # Generate soft data
    circ, det_hard, det_soft, obs, leakage_mask = gen_soft_surrogate_dataset(
        distance=distance,
        rounds=rounds,
        p=ERROR_RATE,
        shots=shots,
        mu=1.2,
        sigma=1.0,
        seed_meas=42,
        seed_analog=43,
        do_sanity_check=True,
        apply_leakage=False,
    )

    # Create output directory
    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Split into train/val
    n_samples = det_soft.shape[0]
    n_train = int(n_samples * TRAIN_SPLIT)

    indices = np.random.RandomState(42).permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Save train data
    train_path = dataset_dir / "train.npz"
    np.savez_compressed(
        train_path,
        det_soft=det_soft[train_idx],
        det_hard=det_hard[train_idx],
        obs=obs[train_idx],
    )

    # Save val data
    val_path = dataset_dir / "val.npz"
    np.savez_compressed(
        val_path,
        det_soft=det_soft[val_idx],
        det_hard=det_hard[val_idx],
        obs=obs[val_idx],
    )

    # Save layout
    layout = build_layout_from_circuit(circ)
    layout["distance"] = distance
    layout_path = dataset_dir / "layout.json"
    save_layout_json(layout, str(layout_path))

    info = {
        "name": name,
        "distance": distance,
        "rounds": rounds,
        "shots": shots,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "num_detectors": int(circ.num_detectors),
        "det_rate": float(det_hard.mean()),
        "obs_rate": float(obs.mean()),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "layout_path": str(layout_path),
    }

    # Save info
    info_path = dataset_dir / "info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"  Saved to {dataset_dir}")
    print(f"  Train samples: {info['n_train']}, Val samples: {info['n_val']}")
    print(f"  Detection rate: {info['det_rate']:.4%}")
    print(f"  Logical error rate: {info['obs_rate']:.4%}")

    return info


def generate_all_datasets():
    """Generate all datasets."""
    output_dir = get_data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_info = []

    for distance in DISTANCES:
        for shots in SHOTS_LIST:
            info = generate_dataset(distance, shots, output_dir)
            all_info.append(info)

    # Save master index
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(all_info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generated {len(all_info)} datasets")
    print(f"Index saved to {index_path}")
    print(f"{'='*60}")

    return all_info


def train_on_dataset(dataset_info: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Train model on a single dataset."""
    from model import AlphaQubitLikeModel, ScatteringResidualConvBlock
    from dataset import SyndromeDataset, make_loader
    from parameter import ScalingConfig, ModelConfigScaling, build_scaling_config
    from utils import AttentionBiasProvider
    from train import train
    from run_train import build_model, build_conv_block, compute_effective_S

    import torch

    name = dataset_info["name"]
    print(f"\n{'='*60}")
    print(f"Training on: {name}")
    print(f"{'='*60}")

    # Load layout
    with open(dataset_info["layout_path"], "r") as f:
        layout = json.load(f)

    # Load data
    train_data = np.load(dataset_info["train_path"])
    val_data = np.load(dataset_info["val_path"])

    # Create datasets - SyndromeDataset expects (samples, labels, layout_json_path)
    train_dataset = SyndromeDataset(
        samples=train_data["det_soft"],
        labels=train_data["obs"],
        layout_json_path=dataset_info["layout_path"],
        input_mode="soft",
    )

    val_dataset = SyndromeDataset(
        samples=val_data["det_soft"],
        labels=val_data["obs"],
        layout_json_path=dataset_info["layout_path"],
        input_mode="soft",
    )

    # Build config - use distance-appropriate batch size
    distance = dataset_info["distance"]
    batch_size = BATCH_SIZE_BY_DISTANCE.get(distance, 32)

    train_cfg = build_scaling_config()
    train_cfg.num_steps = TRAIN_STEPS
    train_cfg.batch_init = batch_size
    train_cfg.eval_every = EVAL_EVERY
    train_cfg.eval_fit_mode = "simple"

    print(f"  Using batch_size={batch_size} for distance={distance}")

    model_cfg = ModelConfigScaling()

    # Build model
    model = build_model(layout, model_cfg, train_cfg, use_full_bias=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Train - train() creates loaders internally
    start_time = time.time()
    best = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=train_cfg,
        run_name=name,
        use_wandb=False,
    )
    train_time = time.time() - start_time

    # Save checkpoint
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{name}_best.pth"

    torch.save({
        "model_state_dict": model.state_dict(),
        "best": best,
        "dataset_info": dataset_info,
    }, checkpoint_path)

    result = {
        "name": name,
        "distance": dataset_info["distance"],
        "shots": dataset_info["shots"],
        "best_ler": best["ler"],
        "best_step": best["step"],
        "train_time": train_time,
        "checkpoint_path": str(checkpoint_path),
    }

    print(f"  Best LER: {best['ler']:.6f} at step {best['step']}")
    print(f"  Training time: {train_time:.1f}s")

    return result


def train_all_datasets():
    """Train on all datasets."""
    data_dir = get_data_dir()
    output_dir = data_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load index
    index_path = data_dir / "index.json"
    if not index_path.exists():
        print("No datasets found. Run with --generate first.")
        return []

    with open(index_path, "r") as f:
        all_info = json.load(f)

    results = []
    results_path = output_dir / "training_results.json"

    for info in all_info:
        try:
            # Clear CUDA cache before each training run
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            result = train_on_dataset(info, output_dir)
            results.append(result)

            # Save partial results after each successful training
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"\n  ERROR training {info['name']}: {e}")
            result = {
                "name": info["name"],
                "distance": info["distance"],
                "shots": info["shots"],
                "best_ler": float("nan"),
                "best_step": -1,
                "train_time": 0,
                "error": str(e),
            }
            results.append(result)

            # Save partial results even on error
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Trained {len(results)} models")
    print(f"Results saved to {results_path}")
    print(f"{'='*60}")

    return results


def plot_results():
    """Plot benchmark results."""
    data_dir = get_data_dir()
    results_path = data_dir / "results" / "training_results.json"

    if not results_path.exists():
        print("No results found. Run with --train first.")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    # Organize results by distance and shots
    by_distance = {}
    by_shots = {}

    for r in results:
        d = r["distance"]
        s = r["shots"]
        ler = r["best_ler"]

        if d not in by_distance:
            by_distance[d] = []
        by_distance[d].append((s, ler))

        if s not in by_shots:
            by_shots[s] = []
        by_shots[s].append((d, ler))

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: LER vs Shots for each distance
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(DISTANCES)))

    for i, d in enumerate(sorted(by_distance.keys())):
        data = sorted(by_distance[d])
        shots = [x[0] for x in data]
        lers = [x[1] for x in data]
        ax1.plot(shots, lers, 'o-', color=colors[i], label=f'd={d}', linewidth=2, markersize=8)

    ax1.set_xlabel('Number of Training Samples', fontsize=12)
    ax1.set_ylabel('Logical Error Rate (LER)', fontsize=12)
    ax1.set_title('LER vs Dataset Size', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    if all(r["best_ler"] > 0 for r in results):
        ax1.set_yscale('log')

    # Plot 2: LER vs Distance for each shot count
    ax2 = axes[1]
    colors2 = plt.cm.plasma(np.linspace(0, 0.8, len(SHOTS_LIST)))

    for i, s in enumerate(sorted(by_shots.keys())):
        data = sorted(by_shots[s])
        distances = [x[0] for x in data]
        lers = [x[1] for x in data]
        ax2.plot(distances, lers, 's-', color=colors2[i], label=f'{s//1000}k shots', linewidth=2, markersize=8)

    ax2.set_xlabel('Code Distance', fontsize=12)
    ax2.set_ylabel('Logical Error Rate (LER)', fontsize=12)
    ax2.set_title('LER vs Code Distance', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(DISTANCES)
    if all(r["best_ler"] > 0 for r in results):
        ax2.set_yscale('log')

    # Plot 3: Heatmap
    ax3 = axes[2]

    # Create matrix
    ler_matrix = np.zeros((len(DISTANCES), len(SHOTS_LIST)))
    for r in results:
        i = DISTANCES.index(r["distance"])
        j = SHOTS_LIST.index(r["shots"])
        ler_matrix[i, j] = r["best_ler"]

    # Handle zeros for log scale
    ler_matrix_plot = np.where(ler_matrix > 0, ler_matrix, 1e-10)

    im = ax3.imshow(np.log10(ler_matrix_plot + 1e-10), cmap='RdYlGn_r', aspect='auto')

    ax3.set_xticks(range(len(SHOTS_LIST)))
    ax3.set_xticklabels([f'{s//1000}k' for s in SHOTS_LIST])
    ax3.set_yticks(range(len(DISTANCES)))
    ax3.set_yticklabels([f'd={d}' for d in DISTANCES])
    ax3.set_xlabel('Training Samples', fontsize=12)
    ax3.set_ylabel('Code Distance', fontsize=12)
    ax3.set_title('LER Heatmap (log10)', fontsize=14)

    # Add text annotations
    for i in range(len(DISTANCES)):
        for j in range(len(SHOTS_LIST)):
            val = ler_matrix[i, j]
            if val == 0:
                text = "0"
            elif val < 0.01:
                text = f'{val:.2e}'
            else:
                text = f'{val:.3f}'
            ax3.text(j, i, text, ha='center', va='center', fontsize=9,
                    color='white' if ler_matrix[i, j] > 0.1 else 'black')

    plt.colorbar(im, ax=ax3, label='log10(LER)')

    plt.tight_layout()

    # Save plot
    plot_path = data_dir / "results" / "benchmark_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")

    # Also save as PDF
    pdf_path = data_dir / "results" / "benchmark_results.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to {pdf_path}")

    plt.show()

    # Print summary table
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    print(f"{'Dataset':<20} {'Distance':>10} {'Shots':>10} {'LER':>15} {'Time (s)':>10}")
    print("-"*70)
    for r in sorted(results, key=lambda x: (x["distance"], x["shots"])):
        ler_str = f"{r['best_ler']:.6f}" if r['best_ler'] > 0 else "0.000000"
        print(f"{r['name']:<20} {r['distance']:>10} {r['shots']:>10} {ler_str:>15} {r['train_time']:>10.1f}")
    print("="*70)


def main():
    global DISTANCES, SHOTS_LIST, TRAIN_STEPS

    parser = argparse.ArgumentParser(
        description="Benchmark QEC transformer across distances and dataset sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--generate", action="store_true",
                       help="Generate all datasets")
    parser.add_argument("--train", action="store_true",
                       help="Train on all datasets")
    parser.add_argument("--plot", action="store_true",
                       help="Plot results")
    parser.add_argument("--all", action="store_true",
                       help="Generate, train, and plot")

    # Override defaults
    parser.add_argument("--distances", type=int, nargs="+", default=DISTANCES,
                       help=f"Code distances to test (default: {DISTANCES})")
    parser.add_argument("--shots", type=int, nargs="+", default=SHOTS_LIST,
                       help=f"Shot counts to test (default: {SHOTS_LIST})")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS,
                       help=f"Training steps per model (default: {TRAIN_STEPS})")

    args = parser.parse_args()

    # Update globals with command line args
    DISTANCES = args.distances
    SHOTS_LIST = args.shots
    TRAIN_STEPS = args.steps

    if args.all:
        args.generate = True
        args.train = True
        args.plot = True

    if not any([args.generate, args.train, args.plot]):
        parser.print_help()
        return

    if args.generate:
        generate_all_datasets()

    if args.train:
        train_all_datasets()

    if args.plot:
        plot_results()


if __name__ == "__main__":
    main()
