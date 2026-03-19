"""
Build a monolithic HDF5 pretraining file from all surface_code_b* directories.

Each directory becomes one HDF5 group keyed by folder name. The file contains:

    /<folder_name>/
        events        (N, D) int8
        labels        (N,)   int8
        measurements  (N, D) int8   [if available]
        attrs: distance, basis, num_stab, num_cycles, num_detectors,
               tokens_per_cycle, stab_id (JSON), cycle_id (JSON),
               x (JSON), y (JSON), stab_type (JSON)

    /sample_index_train  (M_train, 2) int32  — globally shuffled train split
    /sample_index_val    (M_val, 2)   int32  — globally shuffled val split
    /group_keys          variable-length string dataset — maps group_idx -> folder name

The train/val split is performed per group: for each experiment the first
`train_shots` shots (after a seeded per-group shuffle) go to train, the
remaining `val_shots` go to val. Both index arrays are then globally shuffled.

Usage:
    python data/data_random_sample.py
    python data/data_random_sample.py --data_dir data/trans7_data --output data/trans7_data/pretrain.h5 --seed 42
    python data/data_random_sample.py --train_shots 16000 --val_shots 4000
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Add src/trans7_alphaqubit to path for layout and dataset imports
REPO_ROOT = Path(__file__).resolve().parents[1]
_candidates = [
    REPO_ROOT / "src" / "trans7_alphaqubit",  # local: repo root is QEC_Decoder_Transformer/
    REPO_ROOT / "trans7_alphaqubit",           # cluster: repo root is already src/
]
_module_dir = next((p for p in _candidates if (p / "dataset.py").exists()), None)
if _module_dir is None:
    raise RuntimeError(f"Could not find trans7_alphaqubit module. Tried: {_candidates}")
sys.path.insert(0, str(_module_dir))

from dataset import load_folder  # noqa: E402
from layout import get_or_build_layout  # noqa: E402


def parse_folder_name(name: str):
    """Extract (basis, distance, rounds) from surface_code_bX_d3_r01_center_* name."""
    m = re.match(r"surface_code_b([XZ])_d(\d+)_r(\d+)_center_\d+_\d+", name)
    if m is None:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def main():
    parser = argparse.ArgumentParser(
        description="Build monolithic HDF5 pretraining file from surface_code_b* directories"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/trans7_data",
        help="Directory containing surface_code_b* subdirectories (relative to repo root)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HDF5 path (default: <data_dir>/pretrain.h5)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train_shots",
        type=int,
        default=16_000,
        help="Number of training shots per experiment group (default: 16000)",
    )
    parser.add_argument(
        "--val_shots",
        type=int,
        default=4_000,
        help="Number of validation shots per experiment group (default: 4000)",
    )
    args = parser.parse_args()

    try:
        import h5py
    except ImportError:
        print("ERROR: h5py is required. Install with: pip install h5py")
        sys.exit(1)

    data_dir = (REPO_ROOT / args.data_dir).resolve()
    if not data_dir.exists():
        print(f"ERROR: data_dir not found: {data_dir}")
        sys.exit(1)

    output_path = (
        Path(args.output).resolve() if args.output else data_dir / "pretrain.h5"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover all surface_code_b* folders
    folders = sorted(
        p
        for p in data_dir.iterdir()
        if p.is_dir() and p.name.startswith("surface_code_b")
    )
    if not folders:
        print(f"ERROR: No surface_code_b* directories found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(folders)} folders in {data_dir}")
    print(f"Output: {output_path}\n")

    rng = np.random.RandomState(args.seed)
    group_keys = []
    group_sample_counts = []

    with h5py.File(output_path, "w") as hf:
        for folder in folders:
            parsed = parse_folder_name(folder.name)
            if parsed is None:
                print(f"  Skipping (unrecognised name): {folder.name}")
                continue

            basis, distance, rounds = parsed
            print(f"  {folder.name} ...", end=" ", flush=True)

            # Build / load layout
            try:
                layout = get_or_build_layout(folder, distance=distance)
            except FileNotFoundError as e:
                print(f"SKIP — no layout: {e}")
                continue

            layout = {**layout, "basis": basis}

            # Load DEM-simulated data only (prefer_hardware=False for pretraining)
            try:
                events, labels, measurements = load_folder(
                    folder, layout, prefer_hardware=False
                )
            except FileNotFoundError as e:
                print(f"SKIP — no data: {e}")
                continue

            N, D = events.shape
            if N == 0:
                print("SKIP — empty dataset")
                continue

            # Write HDF5 group
            grp = hf.create_group(folder.name)
            chunk_rows = min(1000, N)

            grp.create_dataset(
                "events",
                data=events,
                dtype="int8",
                chunks=(chunk_rows, D),
                compression="gzip",
                compression_opts=1,
            )
            grp.create_dataset(
                "labels",
                data=labels,
                dtype="int8",
                chunks=(chunk_rows,),
                compression="gzip",
                compression_opts=1,
            )
            if measurements is not None:
                grp.create_dataset(
                    "measurements",
                    data=measurements,
                    dtype="int8",
                    chunks=(chunk_rows, D),
                    compression="gzip",
                    compression_opts=1,
                )

            # Store scalar layout fields as group attributes
            grp.attrs["distance"] = layout["distance"]
            grp.attrs["basis"] = basis
            grp.attrs["rounds"] = rounds
            grp.attrs["num_stab"] = layout["num_stab"]
            grp.attrs["num_cycles"] = layout["num_cycles"]
            grp.attrs["num_detectors"] = layout["num_detectors"]
            grp.attrs["tokens_per_cycle"] = layout.get("tokens_per_cycle", 0)

            # Store list fields as JSON strings (HDF5 attrs don't natively support lists)
            for key in ("stab_id", "cycle_id", "x", "y", "stab_type"):
                if key in layout:
                    grp.attrs[key] = json.dumps(layout[key])

            group_keys.append(folder.name)
            group_sample_counts.append(N)
            print(f"{N:,} shots  D={D}")

        if not group_keys:
            print(
                "\nERROR: No groups were written. Check that data folders contain events.01 and obs_flips.01."
            )
            sys.exit(1)

        # Build per-group train/val split, then globally shuffle each.
        train_shots = args.train_shots
        val_shots = args.val_shots
        needed = train_shots + val_shots

        train_rows = []
        val_rows = []
        total_train = 0
        total_val = 0

        print(
            f"\nBuilding train/val sample indices "
            f"({train_shots}/{val_shots} per group) ...",
            flush=True,
        )

        for g_idx, N in enumerate(group_sample_counts):
            if N < needed:
                print(
                    f"  WARNING: group {group_keys[g_idx]} has {N} shots "
                    f"< {needed} requested. Using {N} shots "
                    f"(train={N - N * val_shots // needed}, "
                    f"val={N * val_shots // needed})."
                )
                n_val = N * val_shots // needed
                n_train = N - n_val
            else:
                n_train = train_shots
                n_val = val_shots

            # Per-group shuffle so the split is random within each experiment
            local_perm = rng.permutation(N).astype(np.int32)
            tr_local = local_perm[:n_train]
            va_local = local_perm[n_train : n_train + n_val]

            g_col_tr = np.full(n_train, g_idx, dtype=np.int32)
            train_rows.append(np.stack([g_col_tr, tr_local], axis=1))

            g_col_va = np.full(n_val, g_idx, dtype=np.int32)
            val_rows.append(np.stack([g_col_va, va_local], axis=1))

            total_train += n_train
            total_val += n_val

        sample_index_train = np.concatenate(train_rows, axis=0)
        sample_index_val = np.concatenate(val_rows, axis=0)
        rng.shuffle(sample_index_train)
        rng.shuffle(sample_index_val)

        hf.create_dataset(
            "sample_index_train",
            data=sample_index_train,
            dtype="int32",
            chunks=(min(10_000, len(sample_index_train)), 2),
        )
        hf.create_dataset(
            "sample_index_val",
            data=sample_index_val,
            dtype="int32",
            chunks=(min(10_000, len(sample_index_val)), 2),
        )

        # Store group_keys as a variable-length string dataset
        dt = h5py.special_dtype(vlen=str)
        hf.create_dataset(
            "group_keys", data=np.array(group_keys, dtype=object), dtype=dt
        )

    size_mb = output_path.stat().st_size / 1024**2
    print(f"\nDone.")
    print(f"  Groups  : {len(group_keys)}")
    print(f"  Train   : {total_train:,}")
    print(f"  Val     : {total_val:,}")
    print(f"  Total   : {total_train + total_val:,}")
    print(f"  File    : {output_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
