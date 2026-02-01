#!/usr/bin/env python3
"""
sweep_physical_error_rates.py - Train trans5 at each physical error rate and compare vs MWPM

For each physical error rate p:
  1. Generate fresh basis-separated data using Stim
  2. Train a trans5 model from scratch
  3. Run MWPM on the same validation data
  4. Record both LERs

Produces a plot comparing trans5 vs MWPM across error rates.

Usage (in Docker):
    python src/trans5_alphaqubit/sweep_physical_error_rates.py
    python src/trans5_alphaqubit/sweep_physical_error_rates.py --basis z --shots 100000
    python src/trans5_alphaqubit/sweep_physical_error_rates.py --num_steps 50000

Output:
    src/trans5_alphaqubit/checkpoints/sweep_p/sweep_results.png
    src/trans5_alphaqubit/checkpoints/sweep_p/sweep_results.json
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stim
import torch

# ---------------------------------------------------------------------------
# Resolve imports
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent

# Trans5 model (local import)
sys.path.insert(0, str(_script_dir))
from model import AlphaQubitLikeModel, ScatteringResidualConvBlock

# Trans3 shared modules
_trans3_dir = str(_script_dir.parent / "trans3_alphaqubit")
if _trans3_dir not in sys.path:
    sys.path.insert(0, _trans3_dir)

from dataset import SyndromeDataset, make_loader
from parameter import ScalingConfig, ModelConfigScaling, build_scaling_config
from utils import AttentionBiasProvider, ManhattanDistanceBias
from train import train
from eval import compute_ler_from_logits

# Prop data gen for data generation and layout
_prop_dir = str(_script_dir.parent / "prop_data_gen")
if _prop_dir not in sys.path:
    sys.path.insert(0, _prop_dir)

from gen_basis_data import generate_basis_dataset
_layout_dir = str(_script_dir.parent / "trans3_alphaqubit")
from layout import build_layout_from_circuit, save_layout_json

# MWPM
import pymatching


# ---------------------------------------------------------------------------
# Model building (reused from run_train.py)
# ---------------------------------------------------------------------------
from run_train import build_model, build_conv_block, compute_effective_S


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def generate_data_for_p(
    basis: str, distance: int, rounds: int, p: float,
    shots: int, seed: int, train_split: float, output_dir: Path,
):
    """Generate data for a single (basis, distance, p) and save to disk."""
    dataset_dir = output_dir / f"p{p:.6f}" / f"{basis}_basis" / f"d{distance}_r{rounds}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    circ, det_hard, obs = generate_basis_dataset(
        basis=basis, distance=distance, rounds=rounds,
        p=p, shots=shots, seed=seed,
    )

    # Train/val split
    n_samples = det_hard.shape[0]
    n_train = int(n_samples * train_split)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    np.savez_compressed(dataset_dir / "train.npz",
                        det_hard=det_hard[train_idx], obs=obs[train_idx])
    np.savez_compressed(dataset_dir / "val.npz",
                        det_hard=det_hard[val_idx], obs=obs[val_idx])

    layout = build_layout_from_circuit(circ)
    layout["distance"] = distance
    layout_path = dataset_dir / "layout.json"
    save_layout_json(layout, str(layout_path))

    obs_rate = float(obs.mean())
    info = {
        "basis": basis, "distance": distance, "rounds": rounds,
        "p": p, "shots": shots, "n_train": len(train_idx),
        "n_val": len(val_idx), "num_detectors": int(circ.num_detectors),
        "obs_rate": obs_rate,
    }
    with open(dataset_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    return dataset_dir, circ, det_hard[val_idx], obs[val_idx], obs_rate


def run_mwpm(circuit: stim.Circuit, det_hard_val: np.ndarray, obs_val: np.ndarray) -> float:
    """Run MWPM on validation data and return LER."""
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    predictions = matching.decode_batch(det_hard_val.astype(np.uint8))
    pred_obs = predictions[:, 0]
    obs_flat = obs_val.reshape(-1)
    return float(np.mean(pred_obs != obs_flat))


def train_and_eval_trans5(
    dataset_dir: Path, basis: str, distance: int,
    num_steps: int, eval_every: int, batch_size: int,
    obs_rate: float,
) -> float:
    """Train trans5 from scratch on the generated data and return best LER."""
    rounds = distance * 2

    train_path = str(dataset_dir / "train.npz")
    val_path = str(dataset_dir / "val.npz")
    layout_path = str(dataset_dir / "layout.json")

    with open(layout_path, "r") as f:
        layout = json.load(f)
    if "distance" not in layout:
        layout["distance"] = distance

    # Build model config (trans5 defaults)
    model_cfg = ModelConfigScaling()
    model_cfg.d_model = 128

    # Build training config
    train_cfg = build_scaling_config()
    train_cfg.num_steps = num_steps
    train_cfg.batch_init = batch_size
    train_cfg.eval_every = eval_every
    train_cfg.eval_fit_mode = "simple"
    train_cfg.weight_decay = 1e-4

    # LR decay at 60%, 80%, 90% of training
    train_cfg.lr_decay_steps = [
        int(num_steps * 0.6),
        int(num_steps * 0.8),
        int(num_steps * 0.9),
    ]

    # Next-stab schedule
    train_cfg.next_stab_schedule = "alphaqubit"
    train_cfg.next_stab_weight_min = 0.0
    train_cfg.next_stab_warmup_ratio = 0.3

    # Auto pos_weight from obs_rate
    if 0 < obs_rate < 1:
        train_cfg.logical_pos_weight = (1.0 - obs_rate) / obs_rate

    # Load data
    train_data = np.load(train_path)
    val_data = np.load(val_path)

    train_dataset = SyndromeDataset(
        samples=train_data.get("det_hard"),
        labels=train_data.get("obs"),
        layout_json_path=layout_path,
        input_mode="hard",
    )
    val_dataset = SyndromeDataset(
        samples=val_data.get("det_hard"),
        labels=val_data.get("obs"),
        layout_json_path=layout_path,
        input_mode="hard",
    )

    # Build model
    model = build_model(layout, model_cfg, train_cfg, use_full_bias=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    run_name = f"sweep_{basis}_d{distance}_p{obs_rate:.4f}"

    best = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=train_cfg,
        run_name=run_name,
        use_wandb=False,
    )

    # Clean up GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best["ler"]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(ps, mwpm_lers, trans5_lers, shots, basis, distance, output_path):
    """Create comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 7))

    ps = np.array(ps)
    mwpm_lers = np.array(mwpm_lers)
    trans5_lers = np.array(trans5_lers)

    # Error bars (binomial standard error)
    std_mwpm = (mwpm_lers * (1 - mwpm_lers) / (shots * 0.2)) ** 0.5  # val is 20% of shots
    std_t5 = (trans5_lers * (1 - trans5_lers) / (shots * 0.2)) ** 0.5

    ax.errorbar(ps, mwpm_lers, yerr=std_mwpm, fmt="o-", color="#2ecc71",
                linewidth=2.5, markersize=7, capsize=4, label="MWPM (pymatching)")

    ax.errorbar(ps, trans5_lers, yerr=std_t5, fmt="s-", color="#2e75b6",
                linewidth=2.5, markersize=7, capsize=4, label="Trans5 (trained per p)")

    # Shade region where trans5 beats MWPM
    below = trans5_lers < mwpm_lers
    if below.any():
        ax.fill_between(ps, mwpm_lers, trans5_lers, where=below,
                        alpha=0.15, color="#2e75b6", label="Trans5 < MWPM")

    # Shade region where MWPM beats trans5
    above = trans5_lers > mwpm_lers
    if above.any():
        ax.fill_between(ps, mwpm_lers, trans5_lers, where=above,
                        alpha=0.15, color="#e74c3c", label="MWPM < Trans5")

    ax.set_yscale("log")
    ax.set_xlabel("Physical Error Rate (p)", fontsize=13)
    ax.set_ylabel("Logical Error Rate (LER)", fontsize=13)
    ax.set_title(
        f"Trans5 vs MWPM across Physical Error Rates\n"
        f"{basis.upper()}-basis  d={distance}  ({shots:,} shots/point, trained per p)",
        fontsize=14,
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Sweep physical error rates: train trans5 at each p, compare vs MWPM",
    )
    parser.add_argument("--basis", type=str, default="x", choices=["x", "z"])
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--shots", type=int, default=100000,
                        help="Samples per error rate (80%% train, 20%% val)")
    parser.add_argument("--num_steps", type=int, default=25000,
                        help="Training steps per model")
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_points", type=int, default=10)
    parser.add_argument("--p_min", type=float, default=0.001)
    parser.add_argument("--p_max", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")

    args = parser.parse_args()

    rounds = args.distance * 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"Physical Error Rate Sweep")
    print(f"  Basis: {args.basis}, Distance: {args.distance}, Rounds: {rounds}")
    print(f"  Shots: {args.shots:,}, Steps: {args.num_steps:,}")
    print(f"  Error rates: {args.num_points} points [{args.p_min}, {args.p_max}]")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    ps = np.linspace(args.p_min, args.p_max, args.num_points)

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = _script_dir / "checkpoints" / "sweep_p"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Temp data directory
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    mwpm_lers = []
    trans5_lers = []
    results = []

    total_start = time.time()

    for i, p in enumerate(ps):
        p = float(p)
        iter_start = time.time()

        print(f"\n{'#'*60}")
        print(f"  [{i+1}/{len(ps)}] Physical error rate p = {p:.4f}")
        print(f"{'#'*60}")

        seed_i = args.seed + i * 100

        # 1. Generate data
        print(f"\n  Generating {args.shots:,} shots...")
        dataset_dir, circ, det_hard_val, obs_val, obs_rate = generate_data_for_p(
            basis=args.basis, distance=args.distance, rounds=rounds,
            p=p, shots=args.shots, seed=seed_i,
            train_split=0.8, output_dir=data_dir,
        )
        print(f"  obs_rate = {obs_rate:.4f}")

        # 2. Run MWPM on validation data
        print(f"\n  Running MWPM...")
        mwpm_ler = run_mwpm(circ, det_hard_val, obs_val)
        mwpm_lers.append(mwpm_ler)
        print(f"  MWPM LER: {mwpm_ler:.6f}")

        # 3. Train trans5 and evaluate
        print(f"\n  Training trans5 ({args.num_steps} steps)...")
        t5_ler = train_and_eval_trans5(
            dataset_dir=dataset_dir,
            basis=args.basis,
            distance=args.distance,
            num_steps=args.num_steps,
            eval_every=args.eval_every,
            batch_size=args.batch_size,
            obs_rate=obs_rate,
        )
        trans5_lers.append(t5_ler)

        iter_time = time.time() - iter_start
        ratio = t5_ler / mwpm_ler if mwpm_ler > 0 else float("nan")

        print(f"\n  Trans5 LER: {t5_ler:.6f}")
        print(f"  Ratio (T5/MWPM): {ratio:.2f}x")
        print(f"  Iteration time: {iter_time:.0f}s")

        result = {
            "p": p, "obs_rate": obs_rate,
            "mwpm_ler": mwpm_ler, "trans5_ler": t5_ler,
            "ratio": ratio,
        }
        results.append(result)

        # Save incremental results after each p
        incremental = {
            "basis": args.basis, "distance": args.distance, "rounds": rounds,
            "shots": args.shots, "num_steps": args.num_steps,
            "results": results,
        }
        with open(out_dir / "sweep_results.json", "w") as f:
            json.dump(incremental, f, indent=2)

        # Update plot after each p
        if len(mwpm_lers) >= 2:
            plot_results(
                ps[:len(mwpm_lers)], mwpm_lers, trans5_lers,
                args.shots, args.basis, args.distance,
                out_dir / "sweep_results.png",
            )

    total_time = time.time() - total_start

    # Final plot
    plot_results(
        ps, mwpm_lers, trans5_lers,
        args.shots, args.basis, args.distance,
        out_dir / "sweep_results.png",
    )

    # Print summary table
    print(f"\n{'='*72}")
    print(f"RESULTS SUMMARY  ({total_time:.0f}s total)")
    print(f"{'='*72}")
    print(f"{'p':>8}  {'obs_rate':>8}  {'MWPM':>10}  {'Trans5':>10}  {'T5/MWPM':>8}")
    print(f"{'-'*72}")
    for r in results:
        print(f"{r['p']:>8.4f}  {r['obs_rate']:>8.4f}  "
              f"{r['mwpm_ler']:>10.6f}  {r['trans5_ler']:>10.6f}  {r['ratio']:>8.2f}x")
    print(f"{'='*72}")
    print(f"\nPlot saved to: {out_dir / 'sweep_results.png'}")
    print(f"Data saved to: {out_dir / 'sweep_results.json'}")


if __name__ == "__main__":
    main()