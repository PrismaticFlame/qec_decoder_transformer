#!/usr/bin/env python3
"""
surface_code_dirs_finder.py

Finds all surface_code_b* directories under a root path, consolidates them
into the single parent directory that already contains the most of them,
and validates that the total count matches the expected number.

Usage:
    python surface_code_dirs_finder.py --root /path/to/repo
    python surface_code_dirs_finder.py --root /path/to/repo --dry-run
    python surface_code_dirs_finder.py --root /path/to/repo --expected 130
"""

import argparse
import fnmatch
import os
import shutil
from collections import defaultdict
from pathlib import Path


PATTERN = "surface_code_b*"
DEFAULT_EXPECTED = 130


# ------------------------------------------------------------------------------
# Step 1: Discover
# ------------------------------------------------------------------------------

def find_all(root: Path) -> list[Path]:
    """
    Return all directories matching surface_code_b* under root.
    Does not recurse into surface_code_b* directories themselves.
    """
    results = []
    for dirpath, dirnames, _ in os.walk(root):
        matches = [d for d in dirnames if fnmatch.fnmatch(d, PATTERN)]
        for m in matches:
            results.append(Path(dirpath) / m)
        # Prune in-place: os.walk will not descend into these
        dirnames[:] = [d for d in dirnames if not fnmatch.fnmatch(d, PATTERN)]
    return results


# ------------------------------------------------------------------------------
# Step 2: Group by parent
# ------------------------------------------------------------------------------

def group_by_parent(dirs: list[Path]) -> dict[Path, list[Path]]:
    """Map each parent directory to the list of surface_code_b* dirs inside it."""
    groups: dict[Path, list[Path]] = defaultdict(list)
    for d in dirs:
        groups[d.parent].append(d)
    return dict(groups)


# ------------------------------------------------------------------------------
# Step 3: Pick target
# ------------------------------------------------------------------------------

def pick_target(groups: dict[Path, list[Path]]) -> Path:
    """
    Choose the parent directory with the most surface_code_b* subdirectories.
    Tiebreak: shallowest path (fewest parts = closest to repo root).
    """
    return max(
        groups,
        key=lambda p: (len(groups[p]), -len(p.parts))
    )


# ------------------------------------------------------------------------------
# Step 4: Consolidate
# ------------------------------------------------------------------------------

def consolidate(groups: dict[Path, list[Path]], target: Path, dry_run: bool = False) -> int:
    """
    Move all surface_code_b* dirs from non-target parents into target.
    Skips duplicates (same directory name already exists in target).
    Returns the number of directories moved.
    """
    existing_names = {d.name for d in groups.get(target, [])}
    moved = 0

    for parent, dirs in groups.items():
        if parent == target:
            continue
        for src in dirs:
            if src.name in existing_names:
                print(f"  SKIP (duplicate): {src}")
                continue
            dest = target / src.name
            if dry_run:
                print(f"  [dry-run] MOVE: {src}  →  {dest}")
            else:
                print(f"  MOVE: {src}  →  {dest}")
                shutil.move(str(src), str(dest))
            existing_names.add(src.name)
            moved += 1

    return moved


# ------------------------------------------------------------------------------
# Step 5: Validate
# ------------------------------------------------------------------------------

def validate(target: Path, expected: int = DEFAULT_EXPECTED) -> bool:
    """Count surface_code_b* dirs in target and check against expected."""
    actual = sum(1 for p in target.glob(PATTERN) if p.is_dir())
    if actual == expected:
        print(f"  PASS: {actual}/{expected} surface_code_b* dirs in {target}")
        return True
    else:
        print(f"  FAIL: found {actual}, expected {expected}  (in {target})")
        return False


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Consolidate surface_code_b* directories.")
    parser.add_argument("--root",     type=Path, required=True, help="Root path to search under")
    parser.add_argument("--dry-run",  action="store_true",      help="Print actions without moving anything")
    parser.add_argument("--expected", type=int, default=DEFAULT_EXPECTED,
                        help=f"Expected total count after consolidation (default: {DEFAULT_EXPECTED})")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"ERROR: root path does not exist: {root}")
        return

    # 1. Discover
    print(f"\n[1] Searching for {PATTERN} directories under {root} ...")
    all_dirs = find_all(root)
    print(f"    Found {len(all_dirs)} total.")
    if not all_dirs:
        print("ERROR: no surface_code_b* directories found.")
        return

    # 2. Group
    groups = group_by_parent(all_dirs)
    print(f"\n[2] Found {len(groups)} parent director{'y' if len(groups)==1 else 'ies'}:")
    for parent, dirs in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"    {len(dirs):4d}  {parent}")

    # 3. Pick target
    target = pick_target(groups)
    print(f"\n[3] Target directory: {target}  ({len(groups[target])} dirs)")

    # 4. Consolidate
    if len(groups) == 1:
        print("\n[4] Only one parent — no moves needed.")
    else:
        print(f"\n[4] Consolidating into target {'[dry-run]' if args.dry_run else ''}...")
        moved = consolidate(groups, target, dry_run=args.dry_run)
        print(f"    {'Would move' if args.dry_run else 'Moved'}: {moved} directories.")

    # 5. Validate
    print(f"\n[5] Validating ...")
    if args.dry_run:
        print(f"    [dry-run] Skipping filesystem validation. Total found: {len(all_dirs)}")
    else:
        ok = validate(target, expected=args.expected)
        if not ok:
            raise SystemExit(1)

    print(f"\nData directory: {target}\n")


if __name__ == "__main__":
    main()
