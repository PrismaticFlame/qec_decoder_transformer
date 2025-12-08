import stim
import numpy as np
import pymatching as pm
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# # Locate project root from notebook file location
# project_root = Path().resolve().parents[2]    # adjust if needed
# src_path = project_root / "src"

# if str(src_path) not in sys.path:
#     sys.path.append(str(src_path))

# print("Project root:", project_root)
# print("src added to path:", src_path)

# Now import your train() function
from qec.train_transformer import train
from qec.train_transformer import predict

def attach_analog(det_bits, token_round_idx, rounds=5, m0=1.2, m1=1.2, sigma=0.8): 
    # det_bits: [shots, N*T] in {0,1} 
    shots, D = det_bits.shape 
    rounds = rounds  # know T; compute per-token round index 0..T-1 
    r = np.empty_like(det_bits, dtype=np.float32) 
    for t in range(rounds): 
        mask_t = token_round_idx == t 
        mu0, mu1 = (+m0, -m1) 
        if (t % 2) == 0:   # even-round drift 
            mu0, mu1 = -mu0, -mu1 
        mu = np.where(det_bits[:, mask_t]==0, mu0, mu1) 
        r[:, mask_t] = np.random.normal(mu, sigma) 
    return r 

def gen_dem(distance=3, rounds=5, ps=[1e-3], shots=20000):
    # fig, ax = plt.subplots(len(ps))
    transformer_ler_errors = []
    log_errors = []
    data = {}
    for i, p in enumerate(ps):
        pname = f"physical_error_rate_{str(p)}"
        data[pname] = []
        circ = stim.Circuit.generated("surface_code:rotated_memory_x",
                                      distance=distance, 
                                      rounds=rounds, 
                                      after_clifford_depolarization=p, # Apply depolarizing noise after Clifford gates (idk what those are atm)
                                      before_round_data_depolarization=p, # Apply depolarizing noise before each round
                                      before_measure_flip_probability=p, # Measurement errors
                                      after_reset_flip_probability=p # Reset errors
                                      )
        
        # get error model, building matching, and get detector and observable samples
        # the error model
        model = circ.detector_error_model(decompose_errors=True)
    
        # pymatching model built on error model
        matching = pm.Matching.from_detector_error_model(model)
        
        # generate circuit graph
        circ.diagram('timeline-svg')

        # get analog readout wrapper
        # get measurements
        measurement_sampler = circ.compile_sampler()
        measurements = measurement_sampler.sample(shots)
    
        measurement_round_idx = np.zeros(measurements.shape[1], dtype=np.int64)
        start = 2 * (distance**2) - 1
        for j in range(rounds):
            if j == 0:
                continue;
            measurement_round_idx[start:start + 8] = j
            start = start + 8

        # get soft measurements
        soft_measurements = attach_analog(measurements, measurement_round_idx, rounds=5)
        data[pname].append(soft_measurements)

        # generating syndromes and actual observations
        # syndromes: qubit error detector results * number of rounds for each shot
        # actual_observations: the logical value for each shot
        
        # convert measurements to detector
        m2d = circ.compile_m2d_converter()
        measurements = measurement_sampler.sample(shots)
        syndrome, actual_observations = m2d.convert(measurements=measurements, separate_observables=True)
        data[pname].append(syndrome)
        data[pname].append(actual_observations)
    
        # predicted (corrected) logical values from running pymatching on syndrome
        predicted_observables = matching.decode_batch(syndrome)
        data[pname].append(predicted_observables)
        
        # count of errors (mismatch between actual and predicted)
        num_errors = np.sum(np.any(predicted_observables != actual_observations, axis=1))
        log_errors.append(num_errors / shots)

        # train once, predict from saved model otehr times:
        if i == 0:
            best_ler, best_epoch = train(soft_measurements, actual_observations, measurement=True)
            print(f"Best LER: {best_ler}, Best Epoch: {best_epoch}")
        pred_loss, pred_ler = predict(soft_measurements, actual_observations, measurement=True)
        transformer_ler_errors.append(pred_ler)
    
    return data, np.array(log_errors), np.array(transformer_ler_errors)

def main():
    ps = np.linspace(0.001, 0.01, 10)
    num_shots = 20000
    # data, errors, transformer_errors = gen_dem(distance=3, rounds=5, ps=ps, shots=num_shots)
    errors = np.array([0.0017,  0.0053,  0.01075, 0.02035, 0.0314,  0.04165, 0.05835, 0.0686,  0.08455, 0.105  ])
    transformer_errors = np.array([0.0331,     0.0636,     0.09635,    0.1217,     0.1497,     0.16945,
 0.19760001, 0.2198,     0.2359,     0.24855   ])
    print(errors)
    print(transformer_errors)
    # print(f"Syndromes shape (p=0.001): {data['physical_error_rate_.001'][0].shape}")
    # print(f"Actual Observations sample shape (p=0.001): {data['physical_error_rate_.001'][1].shape}")
    # print(f"Predicted Observations sample shape (p=0.001): {data['physical_error_rate_.001'][2].shape}")
    
    fig, ax = plt.subplots()
    # for logical errors in zip(Ls, log_errors_all_L):
    std_err = (errors*(1-errors)/num_shots)**0.5
    ax.errorbar(ps, errors, yerr=std_err, label="MWPM")
    std_err_transformer = (transformer_errors*(1-transformer_errors)/num_shots)**0.5
    ax.errorbar(ps, transformer_errors, yerr=std_err_transformer, label="Transformer")
    # for i, j in zip(ps, errors):
    #     ax.annotate(str(j), xy=(i, j), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')
    yinterval = np.round(np.logspace(-3, 0, num=10), 3)
    print(yinterval)
    ax.set_yscale("log")
    ax.set_yticks(yinterval)
    ax.set_yticklabels(yinterval)
    ax.set_xlabel("Physical error rate") 
    ax.set_ylabel("Logical error rate") 
    ax.set_title("Logical error rate at various physical error rates for MWPM baseline algorithm")
    ax.legend(loc='upper right')
    # plt. legend (loc=0);
    fig.savefig("ler_mwpm_and_transformer", bbox_inches='tight')

if __name__ == "__main__":
    main()