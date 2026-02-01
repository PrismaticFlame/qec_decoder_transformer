#!/usr/bin/env python3
"""
plot_vs_mwpm.py - Plot trans5 training LER curves for Z and X basis against MWPM.

Computes MWPM baselines live from validation data using pymatching, then
plots each basis with a full training curve and a zoomed convergence panel.

Usage:
    python plot_vs_mwpm.py
    python plot_vs_mwpm.py --distance 3
    python plot_vs_mwpm.py --bases z          # single basis only
    python plot_vs_mwpm.py --output my_plot.png
    python plot_vs_mwpm.py --mwpm_z 0.032655  # skip MWPM computation for Z
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stim
import pymatching


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_eval_csv(path: Path):
    """Load evaluation CSV and return steps, dev_ler, best_ler arrays."""
    steps, dev_lers, best_lers = [], [], []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            dev_lers.append(float(row["dev_ler"]))
            best_lers.append(float(row["best_ler"]))
    return np.array(steps), np.array(dev_lers), np.array(best_lers)


def compute_mwpm_ler(basis: str, distance: int, rounds: int, p: float,
                     val_data_path: Path) -> float:
    """Compute MWPM LER on validation data by rebuilding the Stim circuit."""
    circuit_name = f"surface_code:rotated_memory_{basis}"
    circ = stim.Circuit.generated(
        circuit_name,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )

    dem = circ.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    val_data = np.load(str(val_data_path))
    det_hard = val_data["det_hard"]
    obs = val_data["obs"].reshape(-1)

    predictions = matching.decode_batch(det_hard.astype(np.uint8))
    pred_obs = predictions[:, 0]
    return float(np.mean(pred_obs != obs))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
COLORS = {
    "z": {"train": "#5b9bd5", "best": "#2e75b6", "mwpm": "#c0392b", "gap": "#e74c3c"},
    "x": {"train": "#7fc97f", "best": "#2ca02c", "mwpm": "#d35400", "gap": "#e67e22"},
}


def plot_basis_row(axes, basis: str, steps, dev_lers, best_lers,
                   mwpm_ler: float, distance: int):
    """Plot full curve + zoomed convergence for one basis on two axes."""
    ax_full, ax_zoom = axes
    c = COLORS[basis]
    basis_upper = basis.upper()
    final_best = best_lers[-1]
    final_step = steps[-1]

    # --- Full training curve ---
    ax_full.plot(steps, dev_lers, color=c["train"], alpha=0.4, linewidth=0.8,
                 label="Val LER (per eval)")
    ax_full.plot(steps, best_lers, color=c["best"], linewidth=2.0,
                 label="Best LER (monotonic)")
    ax_full.axhline(y=mwpm_ler, color=c["mwpm"], linewidth=2.0, linestyle="--",
                     label=f"MWPM = {mwpm_ler:.4f}")

    ax_full.set_xlabel("Training Step", fontsize=11)
    ax_full.set_ylabel("Logical Error Rate", fontsize=11)
    ax_full.set_title(
        f"{basis_upper}-basis  d={distance}\nFull Training Curve",
        fontsize=12,
    )
    ax_full.legend(fontsize=9, loc="upper right")
    ax_full.set_ylim(bottom=0)
    ax_full.grid(True, alpha=0.3)

    # Annotate final best
    ax_full.annotate(
        f"Best: {final_best:.4f}",
        xy=(final_step, final_best),
        xytext=(final_step * 0.65, final_best + 0.02),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=c["best"]),
        color=c["best"],
        fontweight="bold",
    )

    # --- Zoomed convergence region ---
    zoom_mask = best_lers < 0.08
    if zoom_mask.sum() > 5:
        zs = steps[zoom_mask]
        zd = dev_lers[zoom_mask]
        zb = best_lers[zoom_mask]

        ax_zoom.plot(zs, zd, color=c["train"], alpha=0.5, linewidth=1.0,
                     marker="o", markersize=2.5, label="Val LER")
        ax_zoom.plot(zs, zb, color=c["best"], linewidth=2.0,
                     label="Best LER")
        ax_zoom.axhline(y=mwpm_ler, color=c["mwpm"], linewidth=2.0,
                         linestyle="--", label=f"MWPM = {mwpm_ler:.4f}")

        gap_pp = (final_best - mwpm_ler) * 100
        ratio = final_best / mwpm_ler if mwpm_ler > 0 else float("nan")
        ax_zoom.fill_between(
            zs, mwpm_ler, zb,
            where=zb > mwpm_ler,
            alpha=0.10, color=c["gap"],
            label=f"Gap: {gap_pp:.2f} pp  ({ratio:.2f}x)",
        )

        ax_zoom.set_xlabel("Training Step", fontsize=11)
        ax_zoom.set_ylabel("Logical Error Rate", fontsize=11)
        ax_zoom.set_title(
            f"{basis_upper}-basis  d={distance}\nConvergence Region (LER < 8%)",
            fontsize=12,
        )
        ax_zoom.legend(fontsize=8, loc="upper right")
        ax_zoom.grid(True, alpha=0.3)

        y_min = min(mwpm_ler, zb.min()) * 0.92
        y_max = zb.max() * 1.05
        ax_zoom.set_ylim(y_min, y_max)
    else:
        ax_zoom.text(0.5, 0.5, "Not enough data\nbelow 8% LER",
                     ha="center", va="center", fontsize=11,
                     transform=ax_zoom.transAxes)
        ax_zoom.set_title(f"{basis_upper}-basis  d={distance}\nConvergence Region",
                          fontsize=12)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot trans5 training curves vs MWPM for both bases",
    )
    parser.add_argument("--bases", type=str, nargs="+", default=["z", "x"],
                        choices=["z", "x"],
                        help="Which bases to plot (default: both)")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds_multiplier", type=int, default=2)
    parser.add_argument("--p", type=float, default=0.005,
                        help="Physical error rate (for MWPM circuit)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory (default: ../prop_data_gen/data)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path")
    parser.add_argument("--mwpm_z", type=float, default=None,
                        help="Override Z-basis MWPM LER (skip computation)")
    parser.add_argument("--mwpm_x", type=float, default=None,
                        help="Override X-basis MWPM LER (skip computation)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    ckpt_dir = script_dir / "checkpoints"
    rounds = args.distance * args.rounds_multiplier

    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        data_dir = (script_dir.parent / "prop_data_gen" / "data").resolve()

    bases = args.bases
    mwpm_overrides = {"z": args.mwpm_z, "x": args.mwpm_x}

    # Collect per-basis data
    basis_data = {}

    for basis in bases:
        # Find eval CSV
        candidates = sorted(ckpt_dir.glob(f"{basis}_d{args.distance}_r*_eval.csv"))
        if not candidates:
            print(f"WARNING: No eval CSV for {basis.upper()}-basis d={args.distance}, skipping")
            continue
        eval_path = candidates[-1]
        print(f"Loading {basis.upper()}-basis eval CSV: {eval_path}")
        steps, dev_lers, best_lers = load_eval_csv(eval_path)

        # MWPM LER
        if mwpm_overrides.get(basis) is not None:
            mwpm_ler = mwpm_overrides[basis]
            print(f"  MWPM LER (override): {mwpm_ler:.6f}")
        else:
            val_path = data_dir / f"{basis}_basis" / f"d{args.distance}_r{rounds}" / "val.npz"
            if not val_path.exists():
                print(f"  WARNING: val data not found at {val_path}, skipping MWPM")
                mwpm_ler = None
            else:
                print(f"  Computing MWPM on {val_path}...")
                mwpm_ler = compute_mwpm_ler(
                    basis=basis, distance=args.distance, rounds=rounds,
                    p=args.p, val_data_path=val_path,
                )
                print(f"  MWPM LER: {mwpm_ler:.6f}")

        if mwpm_ler is None:
            continue

        basis_data[basis] = {
            "steps": steps,
            "dev_lers": dev_lers,
            "best_lers": best_lers,
            "mwpm_ler": mwpm_ler,
        }

    if not basis_data:
        print("ERROR: No basis data found to plot.")
        sys.exit(1)

    # Create figure: one row per basis, two columns (full + zoom)
    n_bases = len(basis_data)
    fig, axes = plt.subplots(n_bases, 2, figsize=(14, 5.5 * n_bases))

    # Handle single-basis case (axes shape is (2,) not (1, 2))
    if n_bases == 1:
        axes = axes.reshape(1, 2)

    for row, basis in enumerate(basis_data):
        d = basis_data[basis]
        plot_basis_row(
            axes[row], basis,
            d["steps"], d["dev_lers"], d["best_lers"],
            d["mwpm_ler"], args.distance,
        )

    fig.suptitle(
        f"Trans5 vs MWPM  |  d={args.distance}  p={args.p}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout(pad=2.0)

    out_path = Path(args.output) if args.output else ckpt_dir / "trans5_vs_mwpm.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to: {out_path}")

    # Print summary table
    print(f"\n{'='*62}")
    print(f"  {'Basis':<8} {'Trans5 LER':>12} {'MWPM LER':>12} {'Gap (pp)':>10} {'Ratio':>8}")
    print(f"  {'-'*54}")
    for basis in basis_data:
        d = basis_data[basis]
        t5 = d["best_lers"][-1]
        mw = d["mwpm_ler"]
        gap = (t5 - mw) * 100
        ratio = t5 / mw if mw > 0 else float("nan")
        print(f"  {basis.upper():<8} {t5:>12.6f} {mw:>12.6f} {gap:>+10.2f} {ratio:>8.3f}x")

    if len(basis_data) == 2 and "z" in basis_data and "x" in basis_data:
        t5_z = basis_data["z"]["best_lers"][-1]
        t5_x = basis_data["x"]["best_lers"][-1]
        mw_z = basis_data["z"]["mwpm_ler"]
        mw_x = basis_data["x"]["mwpm_ler"]
        combined_t5 = (t5_z + t5_x) / 2
        combined_mw = (mw_z + mw_x) / 2
        gap = (combined_t5 - combined_mw) * 100
        ratio = combined_t5 / combined_mw if combined_mw > 0 else float("nan")
        print(f"  {'-'*54}")
        print(f"  {'Avg':<8} {combined_t5:>12.6f} {combined_mw:>12.6f} {gap:>+10.2f} {ratio:>8.3f}x")

    print(f"{'='*62}")


if __name__ == "__main__":
    main()