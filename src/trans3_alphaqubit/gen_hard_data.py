#!/usr/bin/env python3
"""
gen_hard_data.py - Generate hard (binary) detector data for QEC transformer pretraining

This generates hard syndrome data directly from Stim's detector sampler.
Hard data is binary (0/1) and is typically used for pretraining.

Usage:
    python gen_hard_data.py
    python gen_hard_data.py --distance 5 --rounds 10 --shots 50000 --output data/train_hard.npz

Output .npz contains:
    - det_hard: (N, D) int8 - Binary detector events
    - obs: (N, K) int8 - Logical error labels (K=2 for surface code: [X, Z])
"""

import argparse
import json
import os

import numpy as np
import stim

from layout import build_layout_from_circuit, save_layout_json


def gen_hard_data(
    distance: int = 3,
    rounds: int = 5,
    p: float = 1e-3,
    shots: int = 20000,
    seed: int = 42,
) -> tuple:
    """
    Generate hard (binary) detector data using Stim's sampler.

    This uses the same approach as gen_soft_surrogate.py: sample raw measurements
    first, then compute detector events via XOR parity.

    Args:
        distance: Surface code distance
        rounds: Number of QEC rounds
        p: Physical error probability
        shots: Number of samples
        seed: Random seed (used to seed numpy RNG before sampling)

    Returns:
        (circuit, det_hard, obs)
        - circuit: stim.Circuit
        - det_hard: (N, D) int8 binary detector events
        - obs: (N, K) int8 logical error labels
    """
    # Import the dependency extraction functions from gen_soft_surrogate
    from gen_soft_surrogate import (
        extract_detector_rec_dependencies,
        extract_observable_rec_dependencies,
        hard_detectors_from_meas_bits_by_deps,
        hard_observables_from_meas_bits_by_deps,
    )

    # Generate circuit
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )

    # Sample raw measurement bits
    # Note: older Stim versions don't support seed parameter, so we sample without it
    # For reproducibility across runs, use different seed values for train/val
    meas_sampler = circ.compile_sampler()

    # Try with seed first, fall back to no seed for older Stim versions
    try:
        meas_bits = meas_sampler.sample(shots=shots, seed=seed).astype(np.uint8)
    except TypeError:
        # Older Stim version - seed not supported
        print(f"  Note: Stim version doesn't support seed parameter, using unseeded sampling")
        meas_bits = meas_sampler.sample(shots=shots).astype(np.uint8)

    # Extract dependencies
    det_deps = extract_detector_rec_dependencies(circ)
    obs_deps = extract_observable_rec_dependencies(circ)

    # Compute hard detector events and observables via XOR parity
    det_hard = hard_detectors_from_meas_bits_by_deps(meas_bits, det_deps)
    obs = hard_observables_from_meas_bits_by_deps(meas_bits, obs_deps)

    # Convert to int8
    det_hard = det_hard.astype(np.int8)
    obs = obs.astype(np.int8)

    # Reorder observables: Stim returns [Z, X], we want [X, Z]
    # to match dataset.py expectation
    if obs.shape[1] == 2:
        obs = obs[:, [1, 0]]  # Swap columns

    return circ, det_hard, obs


def main():
    parser = argparse.ArgumentParser(
        description="Generate hard (binary) detector data for QEC transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--distance", type=int, default=3,
                        help="Surface code distance")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of QEC rounds")
    parser.add_argument("--p", type=float, default=1e-3,
                        help="Physical error probability")
    parser.add_argument("--shots", type=int, default=20000,
                        help="Number of samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="../../data/train_hard.npz",
                        help="Output .npz file path")
    parser.add_argument("--save_layout", action="store_true",
                        help="Also save layout.json")
    parser.add_argument("--layout_output", type=str, default="../../data/layout.json",
                        help="Output path for layout.json")

    args = parser.parse_args()

    print(f"Generating HARD data for surface code d={args.distance}, rounds={args.rounds}")
    print(f"  Error rate: p={args.p}")
    print(f"  Shots: {args.shots}")
    print(f"  Seed: {args.seed}")
    print()

    # Generate data
    circ, det_hard, obs = gen_hard_data(
        distance=args.distance,
        rounds=args.rounds,
        p=args.p,
        shots=args.shots,
        seed=args.seed,
    )

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Save data
    np.savez_compressed(
        args.output,
        det_hard=det_hard,
        obs=obs,
    )

    print(f"Saved: {args.output}")
    print(f"  det_hard: {det_hard.shape} {det_hard.dtype}")
    print(f"  obs:      {obs.shape} {obs.dtype}")
    if obs.shape[1] == 2:
        print(f"    obs[:, 0] = X-type logical error")
        print(f"    obs[:, 1] = Z-type logical error")

    # Optionally save layout
    if args.save_layout:
        layout = build_layout_from_circuit(circ)
        layout["distance"] = args.distance
        os.makedirs(os.path.dirname(args.layout_output) or ".", exist_ok=True)
        save_layout_json(layout, args.layout_output)
        print(f"\nSaved layout: {args.layout_output}")
        print(f"  num_detectors: {layout['num_detectors']}")
        print(f"  num_stab: {layout['num_stab']}")
        print(f"  num_cycles: {layout['num_cycles']}")

    # Print some statistics
    print()
    print("Statistics:")
    det_rate = det_hard.mean()
    obs_rate = obs.mean(axis=0)
    print(f"  Detection event rate: {det_rate:.4%}")
    if obs.shape[1] == 2:
        print(f"  Logical X error rate: {obs_rate[0]:.4%}")
        print(f"  Logical Z error rate: {obs_rate[1]:.4%}")
    else:
        print(f"  Logical error rate: {obs_rate[0]:.4%}")


if __name__ == "__main__":
    main()
