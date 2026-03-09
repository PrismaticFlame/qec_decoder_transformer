"""
Script to find all subdirectories named 'surface_code_b*' under data/
and move them into data/trans7_data/.
"""

import shutil
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
data_dir = repo_root / "data"
dest_dir = data_dir / "trans7_data"

dest_dir.mkdir(parents=True, exist_ok=True)

dirs = [p for p in data_dir.rglob("surface_code_b*") if p.is_dir()]
files = [
    p for name in ("dataset.docx", "generate_all.py")
    for p in data_dir.rglob(name) if p.is_file()
]

if not dirs:
    print("No surface code directories found. Please add " \
    "surface code directories to data directory and try again.")
else:
    for src in dirs:
        dst = dest_dir / src.name
        print(f"Moving {src} -> {dst}")
        shutil.move(str(src), str(dst))
    print(f"\nDone. Moved {len(dirs)} director(ies) to {dest_dir}")

source_parents = {src.parent for src in dirs + files}

for src in files:
    dst = dest_dir / src.name
    print(f"Moving {src} -> {dst}")
    shutil.move(str(src), str(dst))

if dest_dir in source_parents:
    print("WARNING: One or more source files came from the destination directory. Skipping deletion.")
else:
    for parent in source_parents:
        if parent.exists():
            print(f"Removing original directory: {parent}")
            shutil.rmtree(str(parent))
