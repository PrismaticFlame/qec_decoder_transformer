#!/usr/bin/env python3
"""
gen_basis_data.py - Generate basis-separated hard detector data for QEC decoding

Generates separate X-basis and Z-basis datasets following the AlphaQubit approach:
each basis gets its own dataset with a single logical observable label.

Data is fully reproducible: Stim sampling and train/val splits are both seeded.
Re-running with the same arguments produces identical output.

Usage:
    python gen_basis_data.py
    python gen_basis_data.py --distances 3 5 7 --shots 50000
    python gen_basis_data.py --bases z --p 0.005
    python gen_basis_data.py --ps 0.001 0.003 0.005 0.007 0.01
    python gen_basis_data.py --output_dir ../../data

Output structure:
    data/{basis}_basis/d{d}_r{rounds}_p{p}_s{seed}/train.npz
    data/{basis}_basis/d{d}_r{rounds}_p{p}_s{seed}/val.npz
    data/{basis}_basis/d{d}_r{rounds}_p{p}_s{seed}/layout.json
    data/{basis}_basis/d{d}_r{rounds}_p{p}_s{seed}/info.json

Each .npz contains:
    - det_hard: (N, D) int8 - Binary detector events
    - meas_hard: (N, D) int8 - Reconstructed measurements (cumulative XOR of events)
    - obs: (N, 1) int8 - Single logical error label matching the basis

Seed derivation (deterministic):
    effective_seed = base_seed + basis_offset + distance_offset + p_offset
    where basis_offset  = {"z": 0, "x": 10000}
          distance_offset = distance * 100
          p_offset        = round(p * 1e6)   (microsecond-scale int from p)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import stim

# Local import (trans6 is self-contained)
from layout import build_layout_from_circuit, save_layout_json


def derive_seed(base_seed: int, basis: str, distance: int, p: float) -> int:
    """Deterministic seed from (base_seed, basis, distance, p).

    Same inputs always produce the same seed. Different (basis, distance, p)
    combos produce different seeds to ensure independent samples.
    """
    basis_offset = {"z": 0, "x": 10000}
    distance_offset = distance * 100
    p_offset = round(p * 1e6)  # e.g. p=0.005 -> 5000
    return base_seed + basis_offset[basis] + distance_offset + p_offset


def format_p(p: float) -> str:
    """Format p for directory names: 0.005 -> '0.005', 0.0005 -> '5e-04'."""
    if p >= 0.001:
        return f"{p:.4f}".rstrip("0").rstrip(".")
    else:
        return f"{p:.2e}"


def dataset_dir_name(distance: int, rounds: int, p: float, seed: int) -> str:
    """Build directory name encoding all parameters."""
    return f"d{distance}_r{rounds}_p{format_p(p)}_s{seed}"


def _compute_meas_from_det(det_hard: np.ndarray, layout: dict) -> np.ndarray:
    """Compute per-stabilizer measurements from detection events via cumulative XOR.

    For each stabilizer s, detection events across cycles satisfy:
        event(t, s) = meas(t, s) XOR meas(t-1, s)
    so cumulative XOR recovers the original measurement values.

    Args:
        det_hard: (N, D) int8 detection events
        layout: dict with "stab_id" and "cycle_id" lists of length D

    Returns:
        meas_hard: (N, D) int8 reconstructed measurements, same shape/indexing as det_hard
    """
    stab_id = layout["stab_id"]
    cycle_id = layout["cycle_id"]
    num_stab = max(stab_id) + 1

    meas_hard = np.zeros_like(det_hard)
    for s in range(num_stab):
        # Find detector indices for this stabilizer, sorted by cycle
        det_indices = [i for i, sid in enumerate(stab_id) if sid == s]
        det_indices.sort(key=lambda i: cycle_id[i])
        if len(det_indices) == 0:
            continue
        idx = np.array(det_indices)
        # Cumulative XOR along time axis: meas(t) = XOR(event(0), ..., event(t))
        meas_hard[:, idx] = np.cumsum(det_hard[:, idx], axis=1) % 2

    return meas_hard.astype(np.int8)


def generate_basis_dataset(
    basis: str,
    distance: int,
    rounds: int,
    p: float,
    shots: int,
    seed: int,
):
    """
    Generate hard detector data and reconstructed measurements for a single basis.

    Fully reproducible: Stim's detector sampler is seeded.

    Args:
        basis: "x" or "z"
        distance: Surface code distance
        rounds: Number of QEC rounds
        p: Physical error probability
        shots: Number of samples
        seed: Random seed for Stim sampling

    Returns:
        (circuit, det_hard, meas_hard, obs)
        - circuit: stim.Circuit
        - det_hard: (N, D) int8 binary detector events
        - meas_hard: (N, D) int8 reconstructed measurements
        - obs: (N, 1) int8 single logical error label
    """
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

    sampler = circ.compile_detector_sampler(seed=seed)
    det_events, obs_flips = sampler.sample(shots, separate_observables=True)

    det_hard = det_events.astype(np.int8)
    # Take only the primary observable (index 0), which matches the basis
    obs = obs_flips[:, 0:1].astype(np.int8)

    # Compute measurements from detection events using layout
    layout = build_layout_from_circuit(circ)
    meas_hard = _compute_meas_from_det(det_hard, layout)

    return circ, det_hard, meas_hard, obs


def generate_and_save(
    basis: str,
    distance: int,
    rounds: int,
    p: float,
    shots: int,
    seed: int,
    train_split: float,
    output_dir: Path,
):
    """Generate data for one (basis, distance, rounds, p) config and save to disk."""
    dir_name = dataset_dir_name(distance, rounds, p, seed)
    dataset_dir = output_dir / f"{basis}_basis" / dir_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating: {basis.upper()}-basis  d={distance}  rounds={rounds}  p={p}")
    print(f"  seed={seed}  shots={shots}")
    print(f"  Output: {dataset_dir}")
    print(f"{'='*60}")

    circ, det_hard, meas_hard, obs = generate_basis_dataset(
        basis=basis,
        distance=distance,
        rounds=rounds,
        p=p,
        shots=shots,
        seed=seed,
    )

    # Train/val split (seeded for reproducibility)
    n_samples = det_hard.shape[0]
    n_train = int(n_samples * train_split)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Save train data
    train_path = dataset_dir / "train.npz"
    np.savez_compressed(
        train_path,
        det_hard=det_hard[train_idx],
        meas_hard=meas_hard[train_idx],
        obs=obs[train_idx],
    )

    # Save val data
    val_path = dataset_dir / "val.npz"
    np.savez_compressed(
        val_path,
        det_hard=det_hard[val_idx],
        meas_hard=meas_hard[val_idx],
        obs=obs[val_idx],
    )

    # Save layout
    layout = build_layout_from_circuit(circ)
    layout["distance"] = distance
    layout_path = dataset_dir / "layout.json"
    save_layout_json(layout, str(layout_path))

    # Compute stats
    det_rate = det_hard.mean()
    obs_rate = obs.mean()

    info = {
        "basis": basis,
        "distance": distance,
        "rounds": rounds,
        "p": p,
        "shots": shots,
        "seed": seed,
        "train_split": train_split,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "num_detectors": int(circ.num_detectors),
        "det_rate": float(det_rate),
        "obs_rate": float(obs_rate),
        "dir_name": dir_name,
    }

    # Save info
    info_path = dataset_dir / "info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"  Detectors: {circ.num_detectors}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
    print(f"  det_hard:  {det_hard.shape} {det_hard.dtype}")
    print(f"  meas_hard: {meas_hard.shape} {meas_hard.dtype}")
    print(f"  obs:       {obs.shape} {obs.dtype}")
    print(f"  Detection rate:      {det_rate:.4%}")
    print(f"  Logical error rate:  {obs_rate:.4%}")

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Generate basis-separated hard detector data for QEC decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7],
                        help="Code distances to generate")
    parser.add_argument("--rounds_multiplier", type=int, default=2,
                        help="rounds = distance * multiplier")
    parser.add_argument("--shots", type=int, default=20000,
                        help="Number of samples per dataset")
    parser.add_argument("--p", type=float, default=None,
                        help="Single physical error probability")
    parser.add_argument("--ps", type=float, nargs="+", default=None,
                        help="Multiple physical error probabilities")
    parser.add_argument("--bases", type=str, nargs="+", default=["z", "x"],
                        choices=["z", "x"],
                        help="Which bases to generate")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output root directory (relative to this script)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (combined with basis/distance/p for uniqueness)")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Fraction of data for training")

    args = parser.parse_args()

    # Resolve error rates
    if args.ps is not None:
        ps = args.ps
    elif args.p is not None:
        ps = [args.p]
    else:
        ps = [0.005]  # default

    # Resolve output dir relative to script location
    script_dir = Path(__file__).parent
    output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Bases: {args.bases}")
    print(f"Distances: {args.distances}")
    print(f"Physical error rates: {ps}")
    print(f"Base seed: {args.seed}")

    all_info = []

    for basis in args.bases:
        for distance in args.distances:
            for p in ps:
                rounds = distance * args.rounds_multiplier
                seed = derive_seed(args.seed, basis, distance, p)

                info = generate_and_save(
                    basis=basis,
                    distance=distance,
                    rounds=rounds,
                    p=p,
                    shots=args.shots,
                    seed=seed,
                    train_split=args.train_split,
                    output_dir=output_dir,
                )
                all_info.append(info)

    # Save master index
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(all_info, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Basis':<6} {'d':>3} {'r':>3} {'p':>8} {'Seed':>8} {'Shots':>8} {'LER':>10} {'Directory'}")
    print(f"{'-'*80}")
    for info in all_info:
        print(f"{info['basis'].upper():<6} {info['distance']:>3} {info['rounds']:>3} "
              f"{info['p']:>8.4f} {info['seed']:>8} {info['shots']:>8} "
              f"{info['obs_rate']:>10.4%} {info['dir_name']}")
    print(f"{'='*60}")
    print(f"Index saved to {index_path}")
    print(f"\nTo reproduce any dataset, re-run with the same --seed ({args.seed}).")


if __name__ == "__main__":
    main()
