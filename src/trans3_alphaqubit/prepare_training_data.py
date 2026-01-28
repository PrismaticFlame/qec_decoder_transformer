#!/usr/bin/env python3
"""
Prepare training data from gen_soft_surrogate.py output.
Creates train.npz, val.npz, and layout.json.
"""

import numpy as np
import stim
import json
import os

from layout import build_layout_from_circuit, save_layout_json

def main():
    # Parameters (should match gen_soft_surrogate.py)
    distance = 3
    rounds = 5
    p = 1e-3
    val_ratio = 0.2  # 20% for validation
    
    # Load generated data
    data_path = "../../data/stim_soft_surrogate.npz"
    print(f"Loading data from {data_path}")
    data = np.load(data_path)
    
    det_hard = data["det_hard"]
    det_soft = data["det_soft"]
    obs = data["obs"]
    leakage_mask = data.get("leakage_mask", None)
    
    N = det_hard.shape[0]
    print(f"Total samples: {N}")
    
    # Split into train/val
    n_val = int(N * val_ratio)
    n_train = N - n_val
    
    # Shuffle indices
    np.random.seed(42)
    indices = np.random.permutation(N)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    print(f"Train samples: {n_train}")
    print(f"Val samples: {n_val}")
    
    # Create train data
    train_dict = {
        "det_hard": det_hard[train_idx],
        "det_soft": det_soft[train_idx],
        "obs": obs[train_idx],
    }
    if leakage_mask is not None:
        train_dict["leakage_mask"] = leakage_mask[train_idx]
    
    # Create val data
    val_dict = {
        "det_hard": det_hard[val_idx],
        "det_soft": det_soft[val_idx],
        "obs": obs[val_idx],
    }
    if leakage_mask is not None:
        val_dict["leakage_mask"] = leakage_mask[val_idx]
    
    # Save train/val data
    output_dir = "../../data"
    
    train_path = os.path.join(output_dir, "train.npz")
    np.savez_compressed(train_path, **train_dict)
    print(f"Saved: {train_path}")
    
    val_path = os.path.join(output_dir, "val.npz")
    np.savez_compressed(val_path, **val_dict)
    print(f"Saved: {val_path}")
    
    # Generate layout from circuit
    print("\nGenerating layout.json...")
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    
    layout = build_layout_from_circuit(circ)
    layout["distance"] = distance  # Required by run_train.py
    
    layout_path = os.path.join(output_dir, "layout.json")
    save_layout_json(layout, layout_path)
    print(f"Saved: {layout_path}")
    
    print(f"\nLayout info:")
    print(f"  num_detectors: {layout['num_detectors']}")
    print(f"  num_stab: {layout['num_stab']}")
    print(f"  num_cycles: {layout['num_cycles']}")
    print(f"  distance: {layout['distance']}")
    
    print("\n" + "="*50)
    print("READY FOR TRAINING")
    print("="*50)
    print(f"\nTo train with SOFT data:")
    print(f"  python run_train.py --train_data {train_path} --val_data {val_path} --layout {layout_path} --input_mode soft")
    print(f"\nTo train with HARD data:")
    print(f"  python run_train.py --train_data {train_path} --val_data {val_path} --layout {layout_path} --input_mode hard")

if __name__ == "__main__":
    main()
