# layout.py - Build and cache layout JSON from a data folder (trans7)
#
# Each data folder in goog_dem_data/data_0301 may contain:
#   circuit_noisy.stim   - noisy circuit (preferred for layout)
#   circuit_ideal.stim   - ideal circuit (fallback)
#
# The layout JSON encodes:
#   num_detectors, num_stab, num_cycles
#   stab_id[d], cycle_id[d], x[d], y[d]  -- per-detector metadata
#   stab_type[s]                           -- 1=on-basis, 0=off-basis
#   distance, coord_quant
#
# The layout is cached as layout.json in the folder so it only needs to be
# computed once.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _quantize(v: np.ndarray, q: float = 1.0) -> np.ndarray:
    return np.round(v / q) * q


def build_layout_from_stim_file(
    stim_path: str,
    *,
    distance: Optional[int] = None,
    coord_quant: float = 0.5,
    t_eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    Build a layout dict from a .stim circuit file.

    Parameters
    ----------
    stim_path : path to the .stim circuit file
    distance  : code distance (read from properties.yml if not given, or inferred)
    coord_quant : coordinate quantisation step
    """
    import stim

    circ = stim.Circuit.from_file(stim_path)
    return build_layout_from_circuit(circ, distance=distance, coord_quant=coord_quant, t_eps=t_eps)


def build_layout_from_circuit(
    circ,
    *,
    distance: Optional[int] = None,
    coord_quant: float = 0.5,
    t_eps: float = 1e-6,
) -> Dict[str, Any]:
    """Build layout dict from a stim.Circuit object."""
    num_detectors = circ.num_detectors
    coords_map = circ.get_detector_coordinates()

    x_raw = np.zeros(num_detectors, dtype=np.float64)
    y_raw = np.zeros(num_detectors, dtype=np.float64)
    t_raw = np.zeros(num_detectors, dtype=np.float64)

    for i in range(num_detectors):
        c = coords_map.get(i)
        if c is None:
            continue
        if len(c) > 0:
            x_raw[i] = float(c[0])
        if len(c) > 1:
            y_raw[i] = float(c[1])
        if len(c) > 2:
            t_raw[i] = float(c[2])

    xq = _quantize(x_raw, coord_quant)
    yq = _quantize(y_raw, coord_quant)

    cycle_id = np.floor(t_raw + t_eps).astype(np.int64)
    cycle_id -= cycle_id.min()
    num_cycles = int(cycle_id.max()) + 1

    xy = list(zip(xq.tolist(), yq.tolist()))
    unique_xy = sorted(set(xy))
    xy_to_stab = {p: k for k, p in enumerate(unique_xy)}
    stab_id = np.array([xy_to_stab[p] for p in xy], dtype=np.int64)
    num_stab = int(stab_id.max()) + 1

    # Determine cycle sizes
    sizes: List[int] = []
    indices_by_cycle: List[List[int]] = []
    for t in range(num_cycles):
        idx = np.where(cycle_id == t)[0]
        idx_sorted = idx[np.argsort(stab_id[idx], kind="stable")]
        indices_by_cycle.append(idx_sorted.tolist())
        sizes.append(len(idx_sorted))

    tokens_per_cycle = int(min(sizes)) if sizes else 0

    # On-basis stabilisers: those present in cycle 0
    cycle_0_stab_ids = set(stab_id[cycle_id == 0].tolist())
    stab_type = np.zeros(num_stab, dtype=np.int64)
    for s in cycle_0_stab_ids:
        stab_type[s] = 1

    layout: Dict[str, Any] = {
        "num_detectors": int(num_detectors),
        "num_stab": int(num_stab),
        "num_cycles": int(num_cycles),
        "stab_id": stab_id.tolist(),
        "cycle_id": cycle_id.tolist(),
        "x": xq.tolist(),
        "y": yq.tolist(),
        "t_raw": t_raw.tolist(),
        "tokens_per_cycle": int(tokens_per_cycle),
        "cycle_sizes": sizes,
        "stab_type": stab_type.tolist(),
        "coord_quant": float(coord_quant),
    }

    if distance is not None:
        layout["distance"] = int(distance)
    else:
        # Infer distance from number of stabilisers: d^2 - 1 = num_stab
        # => d = sqrt(num_stab + 1)  (valid for rotated surface code)
        d_inferred = int(round((num_stab + 1) ** 0.5))
        layout["distance"] = d_inferred

    return layout


def get_or_build_layout(
    folder: Path,
    *,
    distance: Optional[int] = None,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Return layout for a data folder, building and caching it if necessary.

    Looks for a pre-existing layout.json first (fastest). If not found (or
    force_rebuild=True), builds from circuit_noisy.stim or circuit_ideal.stim.

    The parsed distance from properties.yml is used when available.
    """
    cache_path = folder / "layout.json"

    if cache_path.exists() and not force_rebuild:
        with open(cache_path) as f:
            layout = json.load(f)
        # Ensure distance is present
        if "distance" not in layout and distance is not None:
            layout["distance"] = distance
        return layout

    # Read distance from properties.yml if not provided
    if distance is None:
        props_path = folder / "properties.yml"
        if props_path.exists():
            import yaml  # only needed here
            with open(props_path) as f:
                props = yaml.safe_load(f)
            distance = int(props.get("distance", 3))

    # Try stim circuit files
    for stim_name in ("circuit_noisy.stim", "circuit_ideal.stim"):
        stim_path = folder / stim_name
        if stim_path.exists():
            layout = build_layout_from_stim_file(
                str(stim_path), distance=distance
            )
            # Cache for future use
            with open(cache_path, "w") as f:
                json.dump(layout, f, indent=2)
            return layout

    raise FileNotFoundError(
        f"Cannot build layout for {folder}: no layout.json, "
        "circuit_noisy.stim, or circuit_ideal.stim found."
    )


def get_layout_for_round(
    data_root: Path,
    basis: str,
    distance: int,
    rounds: int,
    *,
    center: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function: find the right subfolder in data_root and return its layout.

    Folder naming convention: surface_code_b{X/Z}_d{d}_r{r:02d}_center_{row}_{col}
    When center is None, the first matching folder is used.
    """
    prefix = f"surface_code_b{basis.upper()}_d{distance}_r{rounds:02d}"
    candidates = sorted(data_root.glob(f"{prefix}_center_*"))
    if not candidates:
        raise FileNotFoundError(
            f"No folder matching '{prefix}_center_*' found in {data_root}"
        )
    if center is not None:
        row, col = center.split("_")
        candidates = [c for c in candidates if c.name.endswith(f"center_{row}_{col}")]
        if not candidates:
            raise FileNotFoundError(
                f"No folder for center={center} in {data_root}/{prefix}*"
            )
    folder = candidates[0]
    return get_or_build_layout(folder, distance=distance)
