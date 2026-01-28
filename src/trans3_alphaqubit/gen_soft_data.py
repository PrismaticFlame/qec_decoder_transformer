#!/usr/bin/env python3
"""
gen_soft_data.py - Generate soft (continuous) detector data for QEC transformer finetuning

This generates soft syndrome data by simulating Gaussian readout noise.
Soft data is continuous (log-likelihood ratios) and is used for finetuning.

Usage:
    python gen_soft_data.py
    python gen_soft_data.py --distance 5 --rounds 10 --shots 50000 --output data/train_soft.npz

Output .npz contains:
    - det_soft: (N, D) float32 - Soft detector LLRs
    - det_hard: (N, D) int8 - Binary detector events (for reference)
    - obs: (N, K) int8 - Logical error labels (K=2 for surface code: [X, Z])
    - leakage_mask: (N, D) uint8 - Optional leakage mask (if --apply_leakage)
"""

import argparse
import json
import os

import numpy as np
import stim

from gen_soft_surrogate import gen_soft_surrogate_dataset
from layout import build_layout_from_circuit, save_layout_json


def main():
    parser = argparse.ArgumentParser(
        description="Generate soft (continuous) detector data for QEC transformer",
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
    parser.add_argument("--mu", type=float, default=1.2,
                        help="Mean for Gaussian soft readout simulation")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Std dev for Gaussian soft readout simulation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for measurement sampling")
    parser.add_argument("--seed_analog", type=int, default=None,
                        help="Random seed for analog noise (defaults to seed+1)")
    parser.add_argument("--output", type=str, default="../../data/train_soft.npz",
                        help="Output .npz file path")
    parser.add_argument("--apply_leakage", action="store_true",
                        help="Apply leakage masking to data")
    parser.add_argument("--p_leak_stab", type=float, default=1e-3,
                        help="Leakage probability for stabilizer detectors")
    parser.add_argument("--p_leak_data", type=float, default=3e-3,
                        help="Leakage probability for data qubit detectors")
    parser.add_argument("--save_layout", action="store_true",
                        help="Also save layout.json")
    parser.add_argument("--layout_output", type=str, default="../../data/layout.json",
                        help="Output path for layout.json")

    args = parser.parse_args()

    seed_analog = args.seed_analog if args.seed_analog is not None else args.seed + 1

    print(f"Generating SOFT data for surface code d={args.distance}, rounds={args.rounds}")
    print(f"  Error rate: p={args.p}")
    print(f"  Shots: {args.shots}")
    print(f"  Gaussian params: mu={args.mu}, sigma={args.sigma}")
    print(f"  Seeds: meas={args.seed}, analog={seed_analog}")
    if args.apply_leakage:
        print(f"  Leakage: enabled (stab={args.p_leak_stab}, data={args.p_leak_data})")
    print()

    # Generate data
    circ, det_hard, det_soft, obs, leakage_mask = gen_soft_surrogate_dataset(
        distance=args.distance,
        rounds=args.rounds,
        p=args.p,
        shots=args.shots,
        mu=args.mu,
        sigma=args.sigma,
        seed_meas=args.seed,
        seed_analog=seed_analog,
        do_sanity_check=True,
        apply_leakage=args.apply_leakage,
        p_leak_stab=args.p_leak_stab,
        p_leak_data=args.p_leak_data,
        seed_leakage=args.seed + 2,
    )

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Build save dictionary
    save_dict = {
        "det_soft": det_soft,
        "det_hard": det_hard,
        "obs": obs,
    }

    if leakage_mask is not None:
        save_dict["leakage_mask"] = leakage_mask.astype(np.uint8)

    # Save data
    np.savez_compressed(args.output, **save_dict)

    print(f"Saved: {args.output}")
    print(f"  det_soft: {det_soft.shape} {det_soft.dtype}")
    print(f"  det_hard: {det_hard.shape} {det_hard.dtype}")
    print(f"  obs:      {obs.shape} {obs.dtype}")
    if obs.shape[1] == 2:
        print(f"    obs[:, 0] = X-type logical error")
        print(f"    obs[:, 1] = Z-type logical error")
    if leakage_mask is not None:
        leak_rate = 1.0 - leakage_mask.mean()
        print(f"  leakage_mask: {leakage_mask.shape}, leak rate: {leak_rate:.4%}")

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
    det_rate = (det_hard != 0).mean()
    obs_rate = obs.mean(axis=0)
    print(f"  Detection event rate: {det_rate:.4%}")
    if obs.shape[1] == 2:
        print(f"  Logical X error rate: {obs_rate[0]:.4%}")
        print(f"  Logical Z error rate: {obs_rate[1]:.4%}")
    else:
        print(f"  Logical error rate: {obs_rate[0]:.4%}")

    soft_stats = f"mean={det_soft.mean():.3f}, std={det_soft.std():.3f}, min={det_soft.min():.3f}, max={det_soft.max():.3f}"
    print(f"  Soft detector stats: {soft_stats}")


if __name__ == "__main__":
    main()
