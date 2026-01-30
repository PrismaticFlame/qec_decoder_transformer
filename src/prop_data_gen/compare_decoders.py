#!/usr/bin/env python3
"""
compare_decoders.py - Compare transformer decoder against MWPM baseline

Loads MWPM results (from mwpm_decoder.py) and transformer results
(from trans4_alphaqubit/eval_combine.py) and prints a comparison table.

Usage:
    python compare_decoders.py --distance 3
    python compare_decoders.py --distances 3 5 7
    python compare_decoders.py --distance 3 --run_mwpm   # Run MWPM first if needed
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_mwpm_result(data_dir: Path, basis: str, distance: int, rounds: int, split: str = "val"):
    """Load MWPM results from saved JSON."""
    result_path = data_dir / f"{basis}_basis" / f"d{distance}_r{rounds}" / f"mwpm_results_{split}.json"
    if not result_path.exists():
        return None
    with open(result_path, "r") as f:
        return json.load(f)


def load_transformer_result(checkpoint_dir: Path):
    """Load transformer evaluation results."""
    result_path = checkpoint_dir / "eval_results.json"
    if not result_path.exists():
        return None
    with open(result_path, "r") as f:
        return json.load(f)


def compare_for_distance(
    distance: int,
    data_dir: Path,
    transformer_dir: Path,
    rounds_multiplier: int = 2,
    split: str = "val",
) -> dict:
    """Compare MWPM and transformer for a given distance."""
    rounds = distance * rounds_multiplier

    # Load MWPM results
    z_mwpm = load_mwpm_result(data_dir, "z", distance, rounds, split)
    x_mwpm = load_mwpm_result(data_dir, "x", distance, rounds, split)

    mwpm_z_ler = z_mwpm["ler"] if z_mwpm else float("nan")
    mwpm_x_ler = x_mwpm["ler"] if x_mwpm else float("nan")

    if not (np.isnan(mwpm_z_ler) or np.isnan(mwpm_x_ler)):
        mwpm_combined = (mwpm_z_ler + mwpm_x_ler) / 2.0
    else:
        mwpm_combined = float("nan")

    # Load transformer results
    transformer_results = load_transformer_result(transformer_dir)
    trans_z_ler = float("nan")
    trans_x_ler = float("nan")
    trans_combined = float("nan")

    if transformer_results:
        for r in transformer_results:
            if r.get("distance") == distance:
                trans_z_ler = r.get("z_ler", float("nan"))
                trans_x_ler = r.get("x_ler", float("nan"))
                trans_combined = r.get("combined_ler", float("nan"))
                break

    return {
        "distance": distance,
        "rounds": rounds,
        "mwpm_z_ler": mwpm_z_ler,
        "mwpm_x_ler": mwpm_x_ler,
        "mwpm_combined": mwpm_combined,
        "trans_z_ler": trans_z_ler,
        "trans_x_ler": trans_x_ler,
        "trans_combined": trans_combined,
    }


def fmt(val):
    """Format a value for display."""
    if np.isnan(val):
        return "N/A"
    return f"{val:.4%}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare transformer decoder against MWPM baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--distance", type=int,
                        help="Single code distance")
    parser.add_argument("--distances", type=int, nargs="+",
                        help="Multiple code distances")
    parser.add_argument("--rounds_multiplier", type=int, default=2,
                        help="rounds = distance * multiplier")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory (relative to this script)")
    parser.add_argument("--transformer_dir", type=str,
                        default="../trans4_alphaqubit/checkpoints",
                        help="Transformer checkpoint directory (relative to this script)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"])
    parser.add_argument("--run_mwpm", action="store_true",
                        help="Run MWPM decoder first if results don't exist")

    args = parser.parse_args()

    distances = args.distances or ([args.distance] if args.distance else None)
    if distances is None:
        parser.error("Must specify --distance or --distances")

    script_dir = Path(__file__).parent
    data_dir = (script_dir / args.data_dir).resolve()
    transformer_dir = (script_dir / args.transformer_dir).resolve()

    # Optionally run MWPM first
    if args.run_mwpm:
        from mwpm_decoder import run_mwpm_on_dataset
        for distance in distances:
            for basis in ["z", "x"]:
                rounds = distance * args.rounds_multiplier
                result_path = (data_dir / f"{basis}_basis"
                               / f"d{distance}_r{rounds}"
                               / f"mwpm_results_{args.split}.json")
                if not result_path.exists():
                    print(f"Running MWPM for {basis.upper()}-basis d={distance}...")
                    run_mwpm_on_dataset(
                        basis=basis,
                        distance=distance,
                        data_dir=data_dir,
                        rounds_multiplier=args.rounds_multiplier,
                        split=args.split,
                    )

    # Compare
    all_results = []
    for distance in distances:
        result = compare_for_distance(
            distance=distance,
            data_dir=data_dir,
            transformer_dir=transformer_dir,
            rounds_multiplier=args.rounds_multiplier,
            split=args.split,
        )
        all_results.append(result)

    # Print comparison table
    print(f"\n{'='*76}")
    print("DECODER COMPARISON: Transformer vs MWPM")
    print(f"{'='*76}")
    print(f"{'':>10} {'--- MWPM ---':>28} {'--- Transformer ---':>28}")
    print(f"{'d':>4} {'r':>4}   {'Z':>8} {'X':>8} {'Combined':>8}   "
          f"{'Z':>8} {'X':>8} {'Combined':>8}")
    print(f"{'-'*76}")

    for r in all_results:
        print(f"{r['distance']:>4} {r['rounds']:>4}   "
              f"{fmt(r['mwpm_z_ler']):>8} {fmt(r['mwpm_x_ler']):>8} "
              f"{fmt(r['mwpm_combined']):>8}   "
              f"{fmt(r['trans_z_ler']):>8} {fmt(r['trans_x_ler']):>8} "
              f"{fmt(r['trans_combined']):>8}")

    print(f"{'='*76}")

    # Print ratio (how much better/worse transformer is vs MWPM)
    print(f"\nTransformer / MWPM ratio (< 1.0 means transformer is better):")
    for r in all_results:
        if not (np.isnan(r['trans_combined']) or np.isnan(r['mwpm_combined'])
                or r['mwpm_combined'] == 0):
            ratio = r['trans_combined'] / r['mwpm_combined']
            print(f"  d={r['distance']}: {ratio:.2f}x")


if __name__ == "__main__":
    main()
