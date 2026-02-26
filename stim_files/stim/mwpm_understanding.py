import stim
import pymatching as pm
import numpy as np
import matplotlib.pyplot as plt

NUM_SHOTS = 100_000

distances = [3, 5, 7]
physical_error_rates = np.linspace(0.001, 0.01, 10)

fig, ax = plt.subplots(figsize=(8, 6))

for d in distances:
    logical_error_rates = []
    for p_phys in physical_error_rates:
        # Generate circuit
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=d,
            rounds=d * 2,
            after_clifford_depolarization=p_phys
        )
        # Get DEM
        dem = circuit.detector_error_model(decompose_errors=True)
        # Decode with pymatching
        matcher = pm.Matching.from_detector_error_model(dem)
        # Sample detection events and observable flips
        sampler = circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(
            NUM_SHOTS, separate_observables=True
        )
        # Decode: predict observable flips from detection events
        predictions = matcher.decode_batch(detection_events)
        # Logical error = prediction disagrees with actual observable flip
        num_errors = np.sum(predictions != observable_flips)
        ler = num_errors / NUM_SHOTS
        logical_error_rates.append(ler)
        print(f"d={d}, p={p_phys:.4f}, LER={ler:.6f}")

    ax.plot(physical_error_rates, logical_error_rates, 'o-', label=f'd={d}')

ax.set_xlabel("Physical Error Rate")
ax.set_ylabel("Logical Error Rate")
ax.set_yscale("log")
ax.set_title("MWPM Logical Error Rate vs Physical Error Rate (Basis X)")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("mwpm_ler_vs_physical.png", dpi=150)
plt.show()
