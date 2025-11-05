import stim
import numpy as np

def gen_dem(distance=3, rounds=5, p=1e-3, shots=20000):
    circ = stim.Circuit.generated("surface_code:rotated_memory_z",
                                  distance=distance, 
                                  rounds=rounds, 
                                  after_clifford_depolarization=p, # Apply depolarizing noise after Clifford gates (idk what those are atm)
                                  before_round_data_depolarization=p, # Apply depolarizing noise before each round
                                  before_measure_flip_probability=p, # Measurement errors
                                  after_reset_flip_probability=p # Reset errors
                                  )
    
    # get detector and observable samples
    sampler = circ.compile_detector_sampler()
    det_samples, obs_samples = sampler.sample(shots=shots, separate_observables=True)
    
    return det_samples, obs_samples

det_samples, logical_labels = gen_dem(distance=3, rounds=5, p=1e-3, shots=20000)

print(f"Detector samples shape: {det_samples.shape}")
print(f"Logical observale sample shape: {logical_labels.shape}")

np.savez_compressed('stim_files/data/surface_code_data.npz',
                    detectors=det_samples,
                    observables=logical_labels)

np.savetxt('stim_files/data/detector_samples.csv', det_samples, delimiter=',', fmt='%d')
np.savetxt('stim_files/data/logical_labels.csv', logical_labels, delimiter=',', fmt='%d')

