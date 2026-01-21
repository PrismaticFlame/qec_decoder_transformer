import json
import numpy as np
import stim
from typing import Tuple, List, Dict, Any


def _quantize(v: np.ndarray, q: float = 1.0) -> np.ndarray:
    """
    Quantize coordinates to nearest multiple of q.
    e.g. q=0.5 allows half-grid coordinates.
    """
    return np.round(v / q) * q


def build_layout_from_circuit(
    circ: stim.Circuit,
    *,
    coord_quant: float = 0.5,   # Stim 常見 0.5 grid，保守用 0.5
    t_eps: float = 1e-6
) -> Dict[str, Any]:
    num_detectors = circ.num_detectors
    coords_map = circ.get_detector_coordinates()

    # ---- collect raw coords ----
    x_raw = np.zeros(num_detectors, dtype=np.float64)
    y_raw = np.zeros(num_detectors, dtype=np.float64)
    t_raw = np.zeros(num_detectors, dtype=np.float64)

    for i in range(num_detectors):
        c = coords_map.get(i, None)
        if c is None:
            continue
        if len(c) > 0: x_raw[i] = float(c[0])
        if len(c) > 1: y_raw[i] = float(c[1])
        if len(c) > 2: t_raw[i] = float(c[2])

    # ---- quantize x,y and normalize cycles ----
    xq = _quantize(x_raw, coord_quant)
    yq = _quantize(y_raw, coord_quant)

    cycle_id = np.floor(t_raw + t_eps).astype(np.int64)
    cycle_id -= cycle_id.min()

    num_cycles = int(cycle_id.max()) + 1

    # ---- build stab_id by quantized (x,y) ----
    # Using (xq,yq) as stabilizer identity across cycles
    xy = list(zip(xq.tolist(), yq.tolist()))

    # sanity: within same cycle, (x,y) should be unique (otherwise ambiguous)
    pairs = list(zip(cycle_id.tolist(), xy))
    if len(pairs) != len(set(pairs)):
        raise ValueError("Duplicated (cycle, x,y) found after quantization. Increase coord_quant precision or adjust keying.")

    unique_xy = sorted(set(xy))
    xy_to_stab = {p: k for k, p in enumerate(unique_xy)}
    stab_id = np.array([xy_to_stab[p] for p in xy], dtype=np.int64)
    num_stab = int(stab_id.max()) + 1

    # ---- indices by cycle (sorted by stab_id for stable per-cycle token order) ----
    indices_by_cycle: List[List[int]] = []
    order_in_cycle = np.full(num_detectors, -1, dtype=np.int64)

    sizes = []
    for t in range(num_cycles):
        idx = np.where(cycle_id == t)[0]
        # stable order: by stab_id
        idx_sorted = idx[np.argsort(stab_id[idx], kind="stable")]
        indices_by_cycle.append(idx_sorted.tolist())
        sizes.append(len(idx_sorted))
        for k, det_i in enumerate(idx_sorted.tolist()):
            order_in_cycle[det_i] = k

    # check if each cycle has same number of detectors (helps reshape B,L -> B,T,S)
    same_size = (len(set(sizes)) == 1)
    tokens_per_cycle = int(min(sizes)) if len(sizes) > 0 else 0

    layout = {
        "num_detectors": int(num_detectors),
        "num_stab": int(num_stab),
        "num_cycles": int(num_cycles),

        # per-detector arrays length L
        "stab_id": stab_id.tolist(),
        "cycle_id": cycle_id.tolist(),
        "x": xq.tolist(),
        "y": yq.tolist(),
        "t_raw": t_raw.tolist(),
        "order_in_cycle": order_in_cycle.tolist(),

        # cycle structure
        "indices_by_cycle": indices_by_cycle,
        "tokens_per_cycle": int(tokens_per_cycle),
        "equal_tokens_each_cycle": bool(same_size),
        "cycle_sizes": sizes,

        # debug / provenance
        "coord_quant": float(coord_quant),
        "cycle_min_raw": float(t_raw.min()),
        "cycle_max_raw": float(t_raw.max()),
    }
    return layout


def save_layout_json(layout: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(layout, f, indent=2)
