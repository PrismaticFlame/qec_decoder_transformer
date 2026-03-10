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

    /sample_index   (M, 2) int32  — globally shuffled [group_idx, local_idx]
    /group_keys     variable-length string dataset — maps group_idx -> folder name

Usage:
    python data/data_random_sample.py
    python data/data_random_sample.py --data_dir data/trans7_data --output data/trans7_data/pretrain.h5 --seed 42
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Add src/trans7_alphaqubit to path for layout and dataset imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "trans7_alphaqubit"))

from layout import get_or_build_layout  # noqa: E402
from dataset import load_folder         # noqa: E402


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
        "--data_dir", type=str, default="data/trans7_data",
        help="Directory containing surface_code_b* subdirectories (relative to repo root)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HDF5 path (default: <data_dir>/pretrain.h5)"
    )
    parser.add_argument("--seed", type=int, default=42)
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
        Path(args.output).resolve() if args.output
        else data_dir / "pretrain.h5"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover all surface_code_b* folders
    folders = sorted(
        p for p in data_dir.iterdir()
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
                "events", data=events, dtype="int8",
                chunks=(chunk_rows, D), compression="gzip", compression_opts=1
            )
            grp.create_dataset(
                "labels", data=labels, dtype="int8",
                chunks=(chunk_rows,), compression="gzip", compression_opts=1
            )
            if measurements is not None:
                grp.create_dataset(
                    "measurements", data=measurements, dtype="int8",
                    chunks=(chunk_rows, D), compression="gzip", compression_opts=1
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
            print("\nERROR: No groups were written. Check that data folders contain events.01 and obs_flips.01.")
            sys.exit(1)

        # Build globally shuffled sample_index: (M, 2) int32 — [group_idx, local_idx]
        print(f"\nBuilding shuffled sample index ({sum(group_sample_counts):,} total samples) ...", flush=True)
        rows = []
        for g_idx, N in enumerate(group_sample_counts):
            g_col = np.full(N, g_idx, dtype=np.int32)
            l_col = np.arange(N, dtype=np.int32)
            rows.append(np.stack([g_col, l_col], axis=1))

        sample_index = np.concatenate(rows, axis=0)  # (M, 2)
        rng.shuffle(sample_index)

        hf.create_dataset(
            "sample_index", data=sample_index, dtype="int32",
            chunks=(min(10_000, len(sample_index)), 2)
        )

        # Store group_keys as a variable-length string dataset
        dt = h5py.special_dtype(vlen=str)
        hf.create_dataset(
            "group_keys",
            data=np.array(group_keys, dtype=object),
            dtype=dt
        )

    total = sum(group_sample_counts)
    size_mb = output_path.stat().st_size / 1024 ** 2
    print(f"\nDone.")
    print(f"  Groups  : {len(group_keys)}")
    print(f"  Samples : {total:,}")
    print(f"  File    : {output_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
