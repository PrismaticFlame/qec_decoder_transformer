# src/layout.py
from __future__ import annotations

import json
from typing import List, Dict, Any

import numpy as np
import stim


def _quantize(v: np.ndarray, q: float = 1.0) -> np.ndarray:
    return np.round(v / q) * q
#防止模型過度依賴「非常精準但不穩定」的 soft feature 量化是一種正則化



def build_layout_from_circuit(
    circ: stim.Circuit,
    *,
    coord_quant: float = 0.5,
    t_eps: float = 1e-6,
) -> Dict[str, Any]:
    num_detectors = circ.num_detectors #獲取電路中偵測器的總數 D
    coords_map = circ.get_detector_coordinates() #從電路中提取每個偵測器的座標

    x_raw = np.zeros(num_detectors, dtype=np.float64)
    y_raw = np.zeros(num_detectors, dtype=np.float64)
    t_raw = np.zeros(num_detectors, dtype=np.float64) #半開區間邊界

    for i in range(num_detectors):
        c = coords_map.get(i, None)
        if c is None:
            continue
        if len(c) > 0: x_raw[i] = float(c[0])
        if len(c) > 1: y_raw[i] = float(c[1])
        if len(c) > 2: t_raw[i] = float(c[2])

    xq = _quantize(x_raw, coord_quant)
    yq = _quantize(y_raw, coord_quant)

    cycle_id = np.floor(t_raw + t_eps).astype(np.int64) #t_raw從 Stim來的時間座標, 把「連續時間」變成「離散 cycle index」, t_eps：極小正數把「連續時間」變成「離散 cycle index」
    cycle_id -= cycle_id.min() #從0開始
    num_cycles = int(cycle_id.max()) + 1

    xy = list(zip(xq.tolist(), yq.tolist()))
    pairs = list(zip(cycle_id.tolist(), xy))
    if len(pairs) != len(set(pairs)): #檢查兩個偵測器被擠到了同一個時空座標
        raise ValueError("Duplicated (cycle,x,y) found after quantization. Adjust coord_quant or keying.")

    unique_xy = sorted(set(xy))
    xy_to_stab = {p: k for k, p in enumerate(unique_xy)}
    stab_id = np.array([xy_to_stab[p] for p in xy], dtype=np.int64)
    num_stab = int(stab_id.max()) + 1

    indices_by_cycle: List[List[int]] = [] #每一輪detector indice會新增
    order_in_cycle = np.full(num_detectors, -1, dtype=np.int64) #-1初始值代表未定義

    sizes = []
    for t in range(num_cycles):
        idx = np.where(cycle_id == t)[0]
        idx_sorted = idx[np.argsort(stab_id[idx], kind="stable")]#kind="stable"保證按照原始索引順序
        indices_by_cycle.append(idx_sorted.tolist()) #NumPy 做計算,.tolist() 做邊界
        sizes.append(len(idx_sorted)) #最後一輪是偵測data qubit, detector 數不同
        for k, det_i in enumerate(idx_sorted.tolist()):
            order_in_cycle[det_i] = k #det_i絕對index, k相對index（當輪）

    same_size = (len(set(sizes)) == 1)
    tokens_per_cycle = int(min(sizes)) if sizes else 0 #truncate（不用 padding）

    # ---- Determine on-basis vs off-basis stabilizer type ----
    # In a surface code memory experiment, the first cycle (cycle 0) only has
    # detectors for on-basis stabilizers (whose initial eigenvalue is known).
    # For Z-memory: Z-stabilizers are on-basis. For X-memory: X-stabilizers.
    # stab_type[s] = 1 (on-basis), 0 (off-basis)
    cycle_0_stab_ids = set(stab_id[cycle_id == 0].tolist())
    stab_type = np.zeros(num_stab, dtype=np.int64)
    for s in cycle_0_stab_ids:
        stab_type[s] = 1

    return {
        "num_detectors": int(num_detectors),
        "num_stab": int(num_stab),
        "num_cycles": int(num_cycles),

        "stab_id": stab_id.tolist(),
        "cycle_id": cycle_id.tolist(),
        "x": xq.tolist(),
        "y": yq.tolist(),
        "t_raw": t_raw.tolist(),
        "order_in_cycle": order_in_cycle.tolist(),

        "indices_by_cycle": indices_by_cycle,
        "tokens_per_cycle": int(tokens_per_cycle),
        "equal_tokens_each_cycle": bool(same_size),
        "cycle_sizes": sizes,

        "stab_type": stab_type.tolist(),  # 1=on-basis, 0=off-basis

        "coord_quant": float(coord_quant),
        "cycle_min_raw": float(t_raw.min()),
        "cycle_max_raw": float(t_raw.max()),
    }


def save_layout_json(layout: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(layout, f, indent=2)
