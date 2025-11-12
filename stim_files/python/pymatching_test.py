import numpy as np
import stim
import pymatching


# =====================================
# DECODING STIM CIRCUITS
# =====================================

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    distance=5,
    rounds=5,
    after_clifford_depolarization=0.005
)
model = circuit.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(model)
sampler = circuit.compile_detector_sampler()
syndrome, actual_observables = sampler.sample(shots=1000, separate_observables=True)
predicted_observables = matching.decode_batch(syndrome)
num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))

print(f"There were {num_errors} errors.")