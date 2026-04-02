#!/usr/bin/env python3
import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


# =========================
# Metadata (folder name parsing)
# =========================

@dataclass
class DatasetMeta:
    folder: str
    basis: Optional[str] = None
    d: Optional[int] = None
    r: Optional[int] = None
    center_x: Optional[int] = None
    center_y: Optional[int] = None


FOLDER_RE = re.compile(
    r"surface_code_(?P<basis>b[ZX])_d(?P<d>\d+)_r(?P<r>\d+)_center_(?P<x>\d+)_(?P<y>\d+)$"
)


def parse_meta(folder_name: str) -> DatasetMeta:
    m = FOLDER_RE.match(folder_name)
    if not m:
        return DatasetMeta(folder=folder_name)
    return DatasetMeta(
        folder=folder_name,
        basis=m.group("basis"),
        d=int(m.group("d")),
        r=int(m.group("r")),
        center_x=int(m.group("x")),
        center_y=int(m.group("y")),
    )


# =========================
# Utilities
# =========================

def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def file_ok(path: Path, min_bytes: int = 10) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size >= min_bytes


def load_properties_yml(folder: Path) -> Optional[Dict[str, Any]]:
    """
    Minimal YAML reader for the simple `key: value` format used by properties.yml.
    No external dependencies (no PyYAML).
    """
    p = folder / "properties.yml"
    if not p.exists():
        return None

    props: Dict[str, Any] = {}
    for raw in p.read_text(encoding="utf-8", errors="strict").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        # Try parse int / float / bool / string
        if v.lower() in ("true", "false"):
            props[k] = (v.lower() == "true")
        else:
            # int?
            try:
                props[k] = int(v)
                continue
            except ValueError:
                pass
            # float?
            try:
                props[k] = float(v)
                continue
            except ValueError:
                pass
            # strip quotes if any
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            props[k] = v
    return props


def infer_num_detectors_from_dem(circuit: Path) -> int:
    """
    Fallback: infer num_detectors from DEM (stim analyze_errors).
    (This part is reliable; the unreliable part was using DEM to infer num_measurements.)
    """
    p = subprocess.run(
        ["stim", "analyze_errors", "--in", str(circuit)],
        capture_output=True,
        text=True,
        check=True,
    )
    dem = p.stdout
    return sum(1 for l in dem.splitlines() if l.startswith("detector"))


# =========================
# .01 bit-length checks (ignore shots)
# =========================

def bitlen_set_01(path: Path) -> set[int]:
    lens = set()
    with path.open("r", encoding="utf-8", errors="strict") as f:
        for line in f:
            lens.add(len(line.rstrip("\n")))
    return lens


def check_internal_bits(path: Path, label: str) -> Optional[int]:
    lens = bitlen_set_01(path)
    if 0 in lens:
        print(f"  ⚠️  {label}: {path.name} contains empty lines")
        return None
    if len(lens) != 1:
        print(f"  ⚠️  {label}: {path.name} inconsistent bit lengths: {sorted(lens)}")
        return None
    bits = next(iter(lens))
    print(f"  ✔ {label}: {path.name} bits_per_line={bits}")
    return bits


def compare_bit_length_only(p1: Path, p2: Path, label: str) -> None:
    """
    Compare ONLY bits per line between two .01 files (ignore line counts).
    """
    b1 = check_internal_bits(p1, f"{label}/A")
    b2 = check_internal_bits(p2, f"{label}/B")
    if b1 is None or b2 is None:
        return
    if b1 != b2:
        print(f"  ⚠️  {label}: bit-length mismatch {p1.name}={b1}, {p2.name}={b2}")
    else:
        print(f"  ✔ {label}: bit-length match ({b1})")


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("data"))
    ap.add_argument("--shots", type=int, default=20000, help="shots for stim sample (circuit-generated data)")
    ap.add_argument("--force", action="store_true", help="regenerate meas/events/obs even if exist")
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        raise SystemExit(f"data_dir not found: {data_dir}")

    folders = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("surface_code_"))
    if not folders:
        raise SystemExit("No surface_code_* folders found")

    print(f"Found {len(folders)} datasets")

    for folder in folders:
        print(f"\n▶ Processing {folder.name}")
        _ = parse_meta(folder.name)

        circuit = folder / "circuit_noisy.stim"
        if not circuit.exists():
            print("  ⚠️  missing circuit_noisy.stim, skip")
            continue

        props = load_properties_yml(folder)
        if props is None:
            print("  ⚠️  properties.yml not found (will use fallback for detectors; measurements.b8 may be skipped)")

        # outputs from circuit-generated pipeline
        meas = folder / "meas.01"
        events = folder / "events.01"
        obs = folder / "obs_flips.01"

        # existing Google-style b8 files
        meas_b8 = folder / "measurements.b8"
        dets_b8 = folder / "detection_events.b8"

        # ---- 1) Convert b8 → 01 using properties.yml (preferred) ----
        if meas_b8.exists() or dets_b8.exists():
            print("  🔄 converting b8 → 01 (using properties.yml when available)")

            num_meas: Optional[int] = None
            num_dets: Optional[int] = None

            if props is not None:
                num_meas = props.get("circuit_measurements")
                num_dets = props.get("circuit_detectors")

            # Fallback for detectors is okay
            if num_dets is None:
                num_dets = infer_num_detectors_from_dem(circuit)
                print(f"  ℹ️  fallback num_detectors from DEM: {num_dets}")

            # For measurements: if properties missing, do NOT guess from DEM.
            # We can fallback to existing meas.01 bit length if it already exists.
            if num_meas is None and meas.exists():
                b = check_internal_bits(meas, "MEAS/FALLBACK")
                num_meas = b
                if num_meas is not None:
                    print(f"  ℹ️  fallback num_measurements from meas.01 bits: {num_meas}")

            if meas_b8.exists():
                if num_meas is None or num_meas <= 0:
                    print("  ⚠️  cannot determine num_measurements for measurements.b8 (need properties.yml or existing meas.01). Skipping meas b8 conversion.")
                else:
                    run_cmd(
                        [
                            "stim", "convert",
                            "--in", meas_b8.name,
                            "--in_format", "b8",
                            "--out", "meas_from_b8.01",
                            "--out_format", "01",
                            "--num_measurements", str(num_meas),
                        ],
                        folder,
                    )

            if dets_b8.exists():
                if num_dets is None or num_dets <= 0:
                    print("  ⚠️  cannot determine num_detectors for detection_events.b8. Skipping dets b8 conversion.")
                else:
                    run_cmd(
                        [
                            "stim", "convert",
                            "--in", dets_b8.name,
                            "--in_format", "b8",
                            "--out", "events_from_b8.01",
                            "--out_format", "01",
                            "--num_detectors", str(num_dets),
                        ],
                        folder,
                    )

        # ---- 2) Generate from circuit (stim sample + stim m2d) ----
        if not args.force and file_ok(meas) and file_ok(events) and file_ok(obs):
            print("  ✔ circuit outputs already exist, skip")
        else:
            run_cmd(
                ["stim", "sample", "--in", circuit.name, "--shots", str(args.shots),
                 "--out", meas.name, "--out_format", "01"],
                folder,
            )

            # Defensive cleanup: remove empty lines (you hit this before)
            lines = meas.read_text(encoding="utf-8", errors="strict").splitlines()
            lines = [ln for ln in lines if ln.strip() != ""]
            meas.write_text("\n".join(lines) + "\n", encoding="utf-8")

            run_cmd(
                ["stim", "m2d",
                 "--circuit", circuit.name,
                 "--in", meas.name,
                 "--in_format", "01",
                 "--out", events.name,
                 "--out_format", "01",
                 "--obs_out", obs.name,
                 "--obs_out_format", "01"],
                folder,
            )

        # ---- 3) Bit-length comparisons (ignore shots) ----
        meas_from_b8 = folder / "meas_from_b8.01"
        events_from_b8 = folder / "events_from_b8.01"

        if meas.exists() and meas_from_b8.exists():
            compare_bit_length_only(meas, meas_from_b8, "MEAS")

        if events.exists() and events_from_b8.exists():
            compare_bit_length_only(events, events_from_b8, "EVENTS")

    print("\n✅ All done")


if __name__ == "__main__":
    main()