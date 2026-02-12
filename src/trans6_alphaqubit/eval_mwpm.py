#!/usr/bin/env python3
"""
eval_mwpm.py - Evaluate MWPM baseline for comparison with trans6.

Generates Stim circuits, decodes with PyMatching, and computes:
  - LER (Logical Error Rate) at each (distance, rounds)
  - Per-round error rate epsilon
  - Lambda (error suppression factor) across distances

Usage:
    python eval_mwpm.py --distances 3 5 --basis z --p 0.005 --shots 100000
    python eval_mwpm.py --distances 3 5 7 --rounds_multipliers 2 3 4 5 --shots 1000000
    python eval_mwpm.py --distances 3 5 --rounds_list 6 10 15 20 25 --shots 100000
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import stim
import pymatching


def ler_to_epsilon(ler: float, rounds: int) -> float:
    """Convert total LER E(n) to per-round error rate epsilon.

    From AlphaQubit paper Eq. 4:
        epsilon = (1/2) * (1 - (1 - 2*E(n))^(1/n))
    """
    if ler <= 0.0:
        return 0.0
    if ler >= 0.5:
        return 0.5
    return 0.5 * (1.0 - (1.0 - 2.0 * ler) ** (1.0 / rounds))


def mwpm_decode(basis: str, distance: int, rounds: int, p: float,
                shots: int, seed: int = 42) -> dict:
    """Generate data and decode with MWPM, return LER and stats."""
    circ = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )

    sampler = circ.compile_detector_sampler(seed=seed)
    det_events, obs_flips = sampler.sample(shots, separate_observables=True)

    dem = circ.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    predictions = matching.decode_batch(det_events.astype(np.uint8))
    if predictions.ndim == 2:
        pred_obs = predictions[:, 0]
    else:
        pred_obs = predictions
    obs_flat = obs_flips[:, 0]

    num_errors = int(np.sum(pred_obs != obs_flat))
    ler = num_errors / shots
    epsilon = ler_to_epsilon(ler, rounds)

    return {
        "basis": basis,
        "distance": distance,
        "rounds": rounds,
        "p": p,
        "shots": shots,
        "num_errors": num_errors,
        "ler": ler,
        "epsilon": epsilon,
    }


def compute_lambda(eps_small_d: float, eps_large_d: float) -> float:
    """Compute Lambda = epsilon(d_small) / epsilon(d_large)."""
    if eps_large_d <= 0:
        return float("inf")
    return eps_small_d / eps_large_d


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MWPM baseline decoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5],
                        help="Code distances to evaluate")
    parser.add_argument("--rounds_multipliers", type=int, nargs="+", default=None,
                        help="Evaluate at rounds = distance * each multiplier")
    parser.add_argument("--rounds_list", type=int, nargs="+", default=None,
                        help="Explicit round counts to evaluate (same for all distances)")
    parser.add_argument("--basis", type=str, default="z", choices=["z", "x"],
                        help="Measurement basis")
    parser.add_argument("--p", type=float, default=0.005,
                        help="Physical error rate")
    parser.add_argument("--shots", type=int, default=100000,
                        help="Number of shots per evaluation")
    parser.add_argument("--seed", type=int, default=99999,
                        help="Random seed for sampling")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")

    args = parser.parse_args()

    # Determine round counts per distance
    if args.rounds_list is not None:
        rounds_per_distance = {d: args.rounds_list for d in args.distances}
    elif args.rounds_multipliers is not None:
        rounds_per_distance = {
            d: [d * m for m in args.rounds_multipliers]
            for d in args.distances
        }
    else:
        rounds_per_distance = {d: [d * 2] for d in args.distances}

    print(f"MWPM Baseline Evaluation")
    print(f"  Basis: {args.basis.upper()}")
    print(f"  Physical error rate: {args.p}")
    print(f"  Shots: {args.shots:,}")
    print(f"  Distances: {args.distances}")
    print()

    # Run evaluations
    all_results = []
    epsilon_by_distance = {}

    for d in sorted(args.distances):
        rounds_list = sorted(rounds_per_distance[d])
        epsilons = []

        for r in rounds_list:
            print(f"  Evaluating d={d}, r={r}...", end=" ", flush=True)
            result = mwpm_decode(
                basis=args.basis, distance=d, rounds=r,
                p=args.p, shots=args.shots, seed=args.seed + d * 100 + r,
            )
            all_results.append(result)
            epsilons.append(result["epsilon"])
            print(f"LER={result['ler']:.6f}  epsilon={result['epsilon']:.6f}")

        # Use the average epsilon across round counts for this distance
        epsilon_by_distance[d] = np.mean(epsilons)

    # Print results table
    print(f"\n{'='*80}")
    print(f"{'Dist':>6} {'Rounds':>8} {'LER':>12} {'epsilon':>12} {'Errors':>10} {'Shots':>10}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['distance']:>6} {r['rounds']:>8} {r['ler']:>12.6f} "
              f"{r['epsilon']:>12.6f} {r['num_errors']:>10,} {r['shots']:>10,}")

    # Compute Lambda between distance pairs
    sorted_distances = sorted(args.distances)
    if len(sorted_distances) >= 2:
        print(f"\n{'='*80}")
        print("Lambda (error suppression factors):")
        print(f"{'='*80}")

        d_min = sorted_distances[0]
        for d_large in sorted_distances[1:]:
            lam = compute_lambda(epsilon_by_distance[d_min], epsilon_by_distance[d_large])
            print(f"  Lambda_{d_min}/{d_large} = {lam:.4f}  "
                  f"(eps_{d_min}={epsilon_by_distance[d_min]:.6f}, "
                  f"eps_{d_large}={epsilon_by_distance[d_large]:.6f})")

    # Save results
    output_path = args.output
    if output_path is None:
        script_dir = Path(__file__).parent
        output_path = str(script_dir / "mwpm_results.json")

    output = {
        "decoder": "MWPM (PyMatching)",
        "basis": args.basis,
        "p": args.p,
        "shots": args.shots,
        "results": all_results,
        "epsilon_by_distance": {str(k): v for k, v in epsilon_by_distance.items()},
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
