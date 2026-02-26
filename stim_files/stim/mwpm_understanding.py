import stim
import pymatching as pm
import numpy as np
import matplotlib.pyplot as plt
import pathlib

OUT_DIR = pathlib.Path(__file__).parent / "decoder_inputs"
OUT_DIR.mkdir(exist_ok=True)

NUM_SHOTS = 100_000

distances = [3, 5, 7, 9, 11]
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
        # Inspect detection_events before decoding
        # print(f"\n--- detection_events (d={d}, p={p_phys:.4f}) ---")
        # print(f"  shape : {detection_events.shape}  "
        #       f"(num_shots={detection_events.shape[0]}, num_detectors={detection_events.shape[1]})")
        # print(f"  dtype : {detection_events.dtype}")
        # fired = detection_events.sum(axis=1)   # detectors fired per shot
        # print(f"  detectors fired per shot — min:{fired.min()}  max:{fired.max()}  "
        #       f"mean:{fired.mean():.2f}")
        # print(f"  first 5 shots (each row = one shot, columns = detectors):")
        # print(detection_events[:5])

        # ------------------------------------------------------------------

        # Write decoder input to files for inspection
        # tag = f"d{d}_p{p_phys:.4f}"

        # # Full arrays as .npy (compact, lossless — reload with np.load)
        # np.save(OUT_DIR / f"detection_events_{tag}.npy", detection_events)
        # np.save(OUT_DIR / f"observable_flips_{tag}.npy", observable_flips)

        # # First 200 shots as CSV for easy human inspection
        # # Header: det_0, det_1, ..., det_N, obs_0
        # num_det = detection_events.shape[1]
        # num_obs = observable_flips.shape[1]
        # header = (
        #     ",".join(f"det_{i}" for i in range(num_det))
        #     + ","
        #     + ",".join(f"obs_{i}" for i in range(num_obs))
        # )
        # sample = np.hstack([detection_events[:200], observable_flips[:200]]).astype(np.uint8)
        # np.savetxt(
        #     OUT_DIR / f"sample_200shots_{tag}.csv",
        #     sample,
        #     delimiter=",",
        #     header=header,
        #     comments="",
        #     fmt="%d",
        # )
        # print(f"  Saved → decoder_inputs/detection_events_{tag}.npy  "
        #       f"(full {detection_events.shape[0]} shots)")
        # print(f"  Saved → decoder_inputs/sample_200shots_{tag}.csv  "
        #       f"(first 200 shots, human-readable)")
        
        # ------------------------------------------------------------------------

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
