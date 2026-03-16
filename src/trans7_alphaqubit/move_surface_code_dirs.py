"""
Script to find all subdirectories named 'surface_code_b*' under data/
and move them into data/trans7_data/.

Can also be imported and called programmatically via ensure_surface_code_data().
"""

import shutil
from pathlib import Path


def ensure_surface_code_data(data_dir: Path, dest_dir: Path) -> bool:
    """
    Ensure surface_code_b* directories are present in dest_dir.

    If dest_dir already contains surface code directories, returns True immediately.
    Otherwise, searches data_dir recursively, moves any found directories (and the
    companion files dataset.docx / generate_all.py) into dest_dir, then removes
    the now-empty source parent directories.

    Returns True if dest_dir contains surface code data after the call, False if
    no surface code directories could be found anywhere.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Already in place?
    if any(dest_dir.glob("surface_code_b*")):
        return True

    dirs = [p for p in data_dir.rglob("surface_code_b*") if p.is_dir()]
    files = [
        p for name in ("dataset.docx", "generate_all.py")
        for p in data_dir.rglob(name) if p.is_file()
    ]

    if not dirs:
        return False

    print(f"  Data check: moving {len(dirs)} surface_code_b* dir(s) to {dest_dir}")
    for src in dirs:
        dst = dest_dir / src.name
        print(f"    {src.name} -> {dest_dir.name}/")
        shutil.move(str(src), str(dst))

    for src in files:
        dst = dest_dir / src.name
        print(f"    {src.name} -> {dest_dir.name}/")
        shutil.move(str(src), str(dst))

    source_parents = {src.parent for src in dirs + files}
    if dest_dir not in source_parents:
        for parent in source_parents:
            if parent.exists():
                print(f"    Removing original directory: {parent}")
                shutil.rmtree(str(parent))

    return True


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    data_dir  = repo_root / "data"
    dest_dir  = data_dir  / "trans7_data"

    if ensure_surface_code_data(data_dir, dest_dir):
        print(f"Done. Surface code data is in {dest_dir}")
    else:
        print("No surface code directories found. "
              "Please add surface_code_b* directories to data/ and try again.")
