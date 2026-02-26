"""
google_circuit.py

Loads the Google Sycamore surface-code circuit and hardware-fitted DEM from
  stim_files/data/surface_code_bX_d3_r05_center_3_5/

Then generates synthetic shots from the noisy circuit and decodes with
PyMatching using the hardware-fitted DEM, so results can be compared to
the real hardware shots in detection_events.b8 / obs_flips_actual.01.

Key files used:
  circuit_noisy.stim                  — noisy circuit with DEPOLARIZE, etc.
  circuit_detector_error_model.dem    — hardware-fitted DEM for PyMatching
  detection_events.b8                 — real hardware detection events
  obs_flips_actual.01                 — real hardware observable outcomes
"""

import math
import pathlib
import numpy as np
import stim
import pymatching as pm

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = (
    pathlib.Path(__file__).parent.parent          # stim_files/
    / "data"
    / "surface_code_bX_d3_r05_center_3_5"
)

NUM_SYNTHETIC_SHOTS = 100_000
SEED = 42


# ── Helpers ────────────────────────────────────────────────────────────────

def load_b8(path: pathlib.Path, num_bits: int, num_shots: int) -> np.ndarray:
    """Load a .b8 packed-bit file → bool array of shape (num_shots, num_bits)."""
    bytes_per_shot = math.ceil(num_bits / 8)
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    assert raw.size == num_shots * bytes_per_shot, (
        f"{path.name}: expected {num_shots * bytes_per_shot} bytes, got {raw.size}"
    )
    raw  = raw.reshape(num_shots, bytes_per_shot)
    bits = np.unpackbits(raw, axis=1, bitorder="little")
    return bits[:, :num_bits].astype(bool)


def load_01(path: pathlib.Path) -> np.ndarray:
    """Load a .01 text file (one int per line) → bool array."""
    return np.loadtxt(str(path), dtype=np.int8).astype(bool)


def ler_to_epsilon(ler: float, rounds: int) -> float:
    """Per-round error rate from total LER (AlphaQubit paper Eq. 4)."""
    if ler <= 0:
        return 0.0
    if ler >= 0.5:
        return 0.5
    return 0.5 * (1.0 - (1.0 - 2.0 * ler) ** (1.0 / rounds))


# ── Load circuit and DEM ───────────────────────────────────────────────────

print("=" * 60)
print("Loading Google Sycamore surface-code dataset")
print(f"  {DATA_DIR}")
print("=" * 60)

circuit_path = DATA_DIR / "circuit_noisy.stim"
dem_path     = DATA_DIR / "circuit_detector_error_model.dem"

circuit = stim.Circuit.from_file(str(circuit_path))
dem     = stim.DetectorErrorModel.from_file(str(dem_path))

num_detectors  = circuit.num_detectors
num_observables = circuit.num_observables
num_qubits     = circuit.num_qubits

# Sweep bits are used for initial-state randomisation in hardware.
# circuit.num_sweep_bits gives the count; for synthetic shots we
# pass all-zero sweep data → data qubits all start in |0> / |+>.
num_sweep_bits = circuit.num_sweep_bits

print(f"\nCircuit properties:")
print(f"  Qubits:       {num_qubits}")
print(f"  Detectors:    {num_detectors}  (= {num_detectors // 8} rounds × 8 stabilisers + boundary)")
print(f"  Observables:  {num_observables}")
print(f"  Sweep bits:   {num_sweep_bits}  (one per data qubit — set to 0 for synthetic shots)")


# ── Build PyMatching decoder from hardware-fitted DEM ─────────────────────

print(f"\nBuilding PyMatching from hardware-fitted DEM …")
matcher = pm.Matching.from_detector_error_model(dem)
print("  Done.")


# ── Generate synthetic shots ───────────────────────────────────────────────

print(f"\nGenerating {NUM_SYNTHETIC_SHOTS:,} synthetic shots from circuit_noisy.stim …")

sampler = circuit.compile_detector_sampler(seed=SEED)

# Sweep bits default to all-False (zero) when not supplied,
# meaning every data qubit is initialised in the |0> / |+> eigenstate.
det_synthetic, obs_synthetic = sampler.sample(
    NUM_SYNTHETIC_SHOTS,
    separate_observables=True,
)

det_synthetic = det_synthetic.astype(np.uint8)
obs_synthetic = obs_synthetic.astype(bool).ravel()

print(f"  det_synthetic shape : {det_synthetic.shape}   dtype: {det_synthetic.dtype}")
print(f"  obs_synthetic shape : {obs_synthetic.shape}   dtype: {obs_synthetic.dtype}")
print(f"  Detection rate      : {det_synthetic.mean():.4%}")
print(f"  Logical error rate  : {obs_synthetic.mean():.4%}")


# ── Decode synthetic shots ─────────────────────────────────────────────────

print(f"\nDecoding synthetic shots with PyMatching …")
pred_synthetic = matcher.decode_batch(det_synthetic).astype(bool).ravel()
ler_synthetic  = float(np.mean(pred_synthetic != obs_synthetic))
eps_synthetic  = ler_to_epsilon(ler_synthetic, rounds=5)

print(f"  Synthetic LER     : {ler_synthetic:.6f}")
print(f"  Synthetic epsilon : {eps_synthetic:.6f}  (per-round)")


# ── Load and decode real hardware shots ───────────────────────────────────

print(f"\nLoading real hardware shots …")
NUM_HW_SHOTS = 50_000

det_hw  = load_b8(DATA_DIR / "detection_events.b8",
                  num_bits=num_detectors, num_shots=NUM_HW_SHOTS)
obs_hw  = load_01(DATA_DIR / "obs_flips_actual.01").ravel()

print(f"  det_hw shape : {det_hw.shape}   dtype: {det_hw.dtype}")
print(f"  obs_hw shape : {obs_hw.shape}   dtype: {obs_hw.dtype}")
print(f"  Detection rate     : {det_hw.mean():.4%}")
print(f"  Logical error rate : {obs_hw.mean():.4%}")

print(f"\nDecoding hardware shots with PyMatching …")
pred_hw  = matcher.decode_batch(det_hw.astype(np.uint8)).astype(bool).ravel()
ler_hw   = float(np.mean(pred_hw != obs_hw))
eps_hw   = ler_to_epsilon(ler_hw, rounds=5)

print(f"  Hardware LER     : {ler_hw:.6f}")
print(f"  Hardware epsilon : {eps_hw:.6f}  (per-round)")


# ── Summary ────────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"{'':30s}  {'LER':>10}  {'epsilon':>10}")
print(f"{'-' * 60}")
print(f"{'Synthetic (noisy circuit)':30s}  {ler_synthetic:>10.6f}  {eps_synthetic:>10.6f}")
print(f"{'Hardware (Google Sycamore)':30s}  {ler_hw:>10.6f}  {eps_hw:>10.6f}")
print(f"{'=' * 60}")
