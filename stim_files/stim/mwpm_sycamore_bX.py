"""
Compare multiple decoders' LER on Google Sycamore surface code data.

Each dataset is a d=3 surface code memory experiment at different round counts
(r01, r03, r05, r07, ...), each with 50k shots of real hardware data.

For this file to work correctly, have all experiments named 
"surface_code_bX_d3_rXX_center_3_5" folders within the data directory, 
going from rounds 01 to 25. This will compare the LER of all given decoders 
and MWPM across all rounds in the X basis only.

The plot shows LER vs number of rounds for five decoders:
  - Standard MWPM (pymatching, run fresh on the detection events + DEM)
  - Pymatching predictions provided in the dataset (sanity check)
  - Correlated matching predictions provided in the dataset
  - Belief matching predictions provided in the dataset
  - Tensor network contraction predictions provided in the dataset
"""

import os
import math
import numpy as np
import pymatching as pm
import stim
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

DATASETS = [
    ("r01", 1,  8),
    ("r03", 3, 24),
    ("r05", 5, 40),
    ("r07", 7, 56),
    ("r09", 9, 72),
    ("r11", 11, 88),
    ("r13", 13, 104),
    ("r15", 15, 120),
    ("r17", 17, 136),
    ("r19", 19, 152),
    ("r21", 21, 168),
    ("r23", 23, 184),
    ("r25", 25, 200)
]


def load_b8(path: str, num_detectors: int, num_shots: int) -> np.ndarray:
    """Load a .b8 file into a bool array of shape (num_shots, num_detectors).

    .b8 packs detector bits LSB-first, 8 bits per byte, zero-padded to the
    nearest byte per shot.
    """
    bytes_per_shot = math.ceil(num_detectors / 8)
    raw = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
    assert raw.size == num_shots * bytes_per_shot, (
        f"Expected {num_shots * bytes_per_shot} bytes, got {raw.size}"
    )
    raw = raw.reshape(num_shots, bytes_per_shot)
    # unpack bits: np.unpackbits is MSB-first per byte, so reverse within each byte
    bits = np.unpackbits(raw, axis=1, bitorder="little")
    # trim to exact number of detectors
    return bits[:, :num_detectors].astype(bool)


def load_01(path: str) -> np.ndarray:
    """Load a .01 text file (one integer per line) as a bool array."""
    return np.loadtxt(path, dtype=np.int8).astype(bool)


results = []

for tag, rounds, num_detectors in DATASETS:
    folder = os.path.join(BASE_DIR, f"surface_code_bX_d3_{tag}_center_3_5")
    num_shots = 50_000

    if not os.path.isdir(folder):
        print(f"\n--- {tag}: skipping (folder not found) ---")
        continue

    print(f"\n--- {tag}: rounds={rounds}, detectors={num_detectors} ---")

    # Load detection events
    det_events = load_b8(
        os.path.join(folder, "detection_events.b8"),
        num_detectors=num_detectors,
        num_shots=num_shots,
    )

    # Load ground truth
    obs_actual = load_01(os.path.join(folder, "obs_flips_actual.01"))

    # Load pre-computed predictions for comparison
    obs_correlated = load_01(
        os.path.join(folder, "obs_flips_predicted_by_correlated_matching.01")
    )
    obs_pymatching_provided = load_01(
        os.path.join(folder, "obs_flips_predicted_by_pymatching.01")
    )
    obs_belief = load_01(
        os.path.join(folder, "obs_flips_predicted_by_belief_matching.01")
    )
    obs_tensor = load_01(
        os.path.join(folder, "obs_flips_predicted_by_tensor_network_contraction.01")
    )

    # Run pymatching fresh using the hardware-fitted DEM
    dem = stim.DetectorErrorModel.from_file(
        os.path.join(folder, "circuit_detector_error_model.dem")
    )
    matcher = pm.Matching.from_detector_error_model(dem)
    obs_pymatching_fresh = matcher.decode_batch(det_events).astype(bool).ravel()

    # Compute LER for each decoder
    ler_mwpm_fresh     = np.mean(obs_pymatching_fresh    != obs_actual)
    ler_mwpm_provided  = np.mean(obs_pymatching_provided != obs_actual)
    ler_correlated     = np.mean(obs_correlated          != obs_actual)
    ler_belief         = np.mean(obs_belief              != obs_actual)
    ler_tensor         = np.mean(obs_tensor              != obs_actual)

    print(f"  LER (MWPM, fresh):          {ler_mwpm_fresh:.4f}")
    print(f"  LER (MWPM, provided):       {ler_mwpm_provided:.4f}")
    print(f"  LER (correlated matching):  {ler_correlated:.4f}")
    print(f"  LER (belief matching):      {ler_belief:.4f}")
    print(f"  LER (tensor network):       {ler_tensor:.4f}")

    results.append((rounds, ler_mwpm_fresh, ler_mwpm_provided, ler_correlated, ler_belief, ler_tensor))

# --- Plot ---
rounds_list, ler_fresh, ler_prov, ler_corr, ler_belief, ler_tensor = zip(*results)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rounds_list, ler_fresh,  "o-",  label="MWPM (pymatching, re-run)")
ax.plot(rounds_list, ler_prov,   "s--", label="MWPM (provided)", alpha=0.7)
ax.plot(rounds_list, ler_corr,   "^-",  label="Correlated matching (provided)")
ax.plot(rounds_list, ler_belief, "D-",  label="Belief matching (provided)")
ax.plot(rounds_list, ler_tensor, "P-",  label="Tensor network (provided)")

ax.set_xlabel("Number of rounds")
ax.set_ylabel("Logical Error Rate")
ax.set_yscale("log")
ax.set_title("LER vs Rounds — d=3 Surface Code (Google Sycamore, basis X)")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)
ax.set_xticks(rounds_list)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "mwpm_sycamore_ler.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")
plt.show()