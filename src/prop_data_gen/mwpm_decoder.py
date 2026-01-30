#!/usr/bin/env python3
"""
mwpm_decoder.py - MWPM baseline decoder for basis-separated QEC data

Decodes the same validation data used by the transformer, using Minimum Weight
Perfect Matching (via pymatching), and saves the results.

Usage:
    python mwpm_decoder.py --basis z --distance 3
    python mwpm_decoder.py --bases z x --distances 3 5 7
    python mwpm_decoder.py --basis z --distance 3 --data_dir data

Requires: pip install pymatching

Results saved to: data/{basis}_basis/d{d}_r{rounds}/mwpm_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import stim

try:
    import pymatching
except ImportError:
    print("Error: pymatching is required. Install with: pip install pymatching")
    raise


def decode_with_mwpm(
    basis: str,
    distance: int,
    rounds: int,
    p: float,
    det_hard: np.ndarray,
    obs: np.ndarray,
) -> dict:
    """
    Decode detector events with MWPM and compute LER.

    Args:
        basis: "x" or "z"
        distance: Surface code distance
        rounds: Number of QEC rounds
        p: Physical error probability (must match data generation)
        det_hard: (N, D) binary detector events
        obs: (N, 1) or (N,) actual logical error labels

    Returns:
        Dictionary with LER and details
    """
    # Recreate the same circuit used for data generation
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

    # Build matching from detector error model
    dem = circ.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    # Decode each sample
    obs_flat = obs.reshape(-1)
    n_samples = det_hard.shape[0]
    n_correct = 0

    predictions = matching.decode_batch(det_hard.astype(np.uint8))
    # predictions shape: (N, num_observables)
    # Take only the primary observable (index 0)
    pred_obs = predictions[:, 0]

    n_correct = np.sum(pred_obs == obs_flat)
    n_wrong = n_samples - n_correct
    ler = n_wrong / n_samples

    return {
        "ler": float(ler),
        "n_samples": int(n_samples),
        "n_correct": int(n_correct),
        "n_wrong": int(n_wrong),
        "basis": basis,
        "distance": distance,
        "rounds": rounds,
        "p": p,
    }


def run_mwpm_on_dataset(
    basis: str,
    distance: int,
    data_dir: Path,
    rounds_multiplier: int = 2,
    split: str = "val",
) -> dict:
    """Run MWPM decoder on a dataset and save results."""
    rounds = distance * rounds_multiplier
    dataset_path = data_dir / f"{basis}_basis" / f"d{distance}_r{rounds}"

    data_path = dataset_path / f"{split}.npz"
    info_path = dataset_path / "info.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    # Load info for p value
    with open(info_path, "r") as f:
        info = json.load(f)
    p = info["p"]

    # Load data
    data = np.load(data_path)
    det_hard = data.get("det_hard")
    obs = data.get("obs", data.get("labels"))

    print(f"  {basis.upper()}-basis d={distance} r={rounds}: "
          f"{det_hard.shape[0]} samples, {det_hard.shape[1]} detectors")

    # Decode
    result = decode_with_mwpm(
        basis=basis,
        distance=distance,
        rounds=rounds,
        p=p,
        det_hard=det_hard,
        obs=obs,
    )

    # Save results
    result_path = dataset_path / f"mwpm_results_{split}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  MWPM LER: {result['ler']:.6f}  "
          f"({result['n_correct']}/{result['n_samples']} correct)")
    print(f"  Saved to {result_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run MWPM decoder on basis-separated QEC data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--basis", type=str, choices=["z", "x"],
                        help="Single basis to decode")
    parser.add_argument("--bases", type=str, nargs="+", choices=["z", "x"],
                        help="Multiple bases to decode")
    parser.add_argument("--distance", type=int,
                        help="Single code distance")
    parser.add_argument("--distances", type=int, nargs="+",
                        help="Multiple code distances")
    parser.add_argument("--rounds_multiplier", type=int, default=2,
                        help="rounds = distance * multiplier")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory (relative to this script)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"],
                        help="Which data split to decode")

    args = parser.parse_args()

    bases = args.bases or ([args.basis] if args.basis else None)
    if bases is None:
        parser.error("Must specify --basis or --bases")

    distances = args.distances or ([args.distance] if args.distance else None)
    if distances is None:
        parser.error("Must specify --distance or --distances")

    script_dir = Path(__file__).parent
    data_dir = (script_dir / args.data_dir).resolve()

    results = []

    for basis in bases:
        for distance in distances:
            print(f"\n{'='*60}")
            print(f"MWPM Decoding: {basis.upper()}-basis  d={distance}")
            print(f"{'='*60}")

            result = run_mwpm_on_dataset(
                basis=basis,
                distance=distance,
                data_dir=data_dir,
                rounds_multiplier=args.rounds_multiplier,
                split=args.split,
            )
            results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print("MWPM RESULTS")
    print(f"{'='*60}")
    print(f"{'Basis':<8} {'Distance':>8} {'Rounds':>8} {'LER':>12} {'Correct':>12}")
    print(f"{'-'*52}")
    for r in results:
        print(f"{r['basis'].upper():<8} {r['distance']:>8} {r['rounds']:>8} "
              f"{r['ler']:>12.6f} {r['n_correct']:>5}/{r['n_samples']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
