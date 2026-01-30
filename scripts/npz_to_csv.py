#!/usr/bin/env python3
"""
npz_to_csv.py - Convert .npz files to .csv files

Usage:
    python npz_to_csv.py data/benchmark/d3_shots20k/train.npz
    python npz_to_csv.py data/benchmark/d3_shots20k/train.npz --output_dir output/
    python npz_to_csv.py data/benchmark/ --recursive

Each array in the .npz file is saved as a separate .csv file:
    train.npz -> train_det_soft.csv, train_det_hard.csv, train_obs.csv, etc.
"""

import argparse
import os
from pathlib import Path

import numpy as np


def npz_to_csv(npz_path: str, output_dir: str = None, verbose: bool = True):
    """
    Convert a single .npz file to CSV files.

    Args:
        npz_path: Path to the .npz file
        output_dir: Directory to save CSV files (defaults to same dir as npz)
        verbose: Print progress information
    """
    npz_path = Path(npz_path)

    if not npz_path.exists():
        print(f"Error: File not found: {npz_path}")
        return

    if output_dir is None:
        output_dir = npz_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    base_name = npz_path.stem  # e.g., "train" from "train.npz"

    if verbose:
        print(f"\nProcessing: {npz_path}")

    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return

    keys = list(data.keys())
    if verbose:
        print(f"  Found {len(keys)} arrays: {keys}")

    for key in keys:
        arr = data[key]
        csv_name = f"{base_name}_{key}.csv"
        csv_path = output_dir / csv_name

        if verbose:
            print(f"  Saving {key}: shape={arr.shape}, dtype={arr.dtype} -> {csv_path}")

        # Handle different array dimensions
        if arr.ndim == 1:
            # 1D array: save as single column
            np.savetxt(csv_path, arr, delimiter=",", fmt=get_format(arr.dtype))
        elif arr.ndim == 2:
            # 2D array: save as rows x columns
            np.savetxt(csv_path, arr, delimiter=",", fmt=get_format(arr.dtype))
        else:
            # 3D+ array: flatten to 2D (samples x flattened_features)
            flat_shape = (arr.shape[0], -1)
            arr_flat = arr.reshape(flat_shape)
            if verbose:
                print(f"    Note: Flattened {arr.shape} -> {arr_flat.shape}")
            np.savetxt(csv_path, arr_flat, delimiter=",", fmt=get_format(arr.dtype))

    data.close()

    if verbose:
        print(f"  Done! Saved {len(keys)} CSV files to {output_dir}")


def get_format(dtype):
    """Get appropriate format string for numpy savetxt based on dtype."""
    if np.issubdtype(dtype, np.integer):
        return "%d"
    elif np.issubdtype(dtype, np.floating):
        return "%.6g"
    else:
        return "%s"


def find_npz_files(path: str, recursive: bool = False):
    """Find all .npz files in a path."""
    path = Path(path)

    if path.is_file():
        if path.suffix == ".npz":
            return [path]
        else:
            return []

    if recursive:
        return list(path.rglob("*.npz"))
    else:
        return list(path.glob("*.npz"))


def main():
    parser = argparse.ArgumentParser(
        description="Convert .npz files to CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("input", type=str,
                        help="Path to .npz file or directory containing .npz files")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Output directory for CSV files (default: same as input)")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Search for .npz files recursively in directories")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress verbose output")

    args = parser.parse_args()

    npz_files = find_npz_files(args.input, args.recursive)

    if not npz_files:
        print(f"No .npz files found in: {args.input}")
        return

    print(f"Found {len(npz_files)} .npz file(s)")

    for npz_file in npz_files:
        # If output_dir is specified, mirror the directory structure
        if args.output_dir:
            input_path = Path(args.input)
            if input_path.is_dir():
                rel_path = npz_file.parent.relative_to(input_path)
                out_dir = Path(args.output_dir) / rel_path
            else:
                out_dir = Path(args.output_dir)
        else:
            out_dir = None

        npz_to_csv(npz_file, out_dir, verbose=not args.quiet)

    print(f"\nCompleted! Processed {len(npz_files)} file(s)")


if __name__ == "__main__":
    main()
