import pandas as pd

# Overview
overviewProperties = [
    "Framework", 
    "Basis handling",
    "Readout network",
    "Embedding inputs",
    "Embedding ResNet",
    "Final-round handling",
    "Data source",
    "Training strategy",
    "Tested distances",
    "Max rounds tested",
]
overviewT3 = [
    "PyTorch", 
    "Dual-heaed (X + Z)", 
    "Linear classifier",
    "1 (syndrome value)",
    "None",
    "No special treatment",
    "Stim cirtuit depolarizing noise",
    "Single-stage",
    "d=3, d=5",
    25,
]
overviewT5 = [
    "Pytorch",
    "Single-basis per model",
    "ReadoutResNet",
    "1 (syndrome value)",
    "None",
    "No special treatment",
    "Stim ciruit depolarizing noise",
    "Single-stage",
    "d=3, d=5, d=7",
    "Varies"
]
overviewT6 = [
    "Pytorch",
    "Single-basis per model",
    "ReadoutResNet",
    "4 (meas, event, leak, event_leak)",
    "2-layer ResNet after summation",
    "Seperate final_embed + learned off-basis vector",
    "Stim ciruit depolarizing noise",
    "Single-stage",
    "d3",
    6
]
overviewAlpha = [
    "JAX / Haiku / JAXline",
    "Single-basis per model",
    "ReadoutResNet (scatter -> conv -> pool -> ResNet)",
    "4 (post1, post2 + events + leakage + events)",
    "2-layer ResNet after summation",
    "Seperate final-round embedding + off-basis embedding",
    "SI1000 + Pauli + (cross-talk, leakage, I/Q) + experimental",
    "Two-stage (pretrain on simulated + finetune on experimental)",
    "d=3, 5, 7, 9, 11",
    "100,000"
]
overviewDF = pd.DataFrame({
    "Property": overviewProperties,
    "v3": overviewT3,
    "v5": overviewT5,   
    "v6": overviewT6,
    "AlphaQubit (Paper)": overviewAlpha
})
overviewDF = overviewDF.set_index("Property")


# Data input and structure
#Consider seperating the sections or keeping it together
# 1.1 Data Generation
generationApsect = [
    "Noise model",
    "Noise parameters",
    "Typical p values",
    "Shots",
    "Leakage",
    "Soft readouts (I/Q)",
    "Cross-talk",
]
generationV3 = [
    "Stim surface_code:rotated_memory_{basis} with uniform depolarization",
    "after_clifford_depolarization=p, before_round_data_depolarization=p, before_measure_flip_probability=p, after_reset_flip_probability=p",
    "0.005",
    "20,000-50,000",
    "No",
    "No",
    "No"
]
generationV5 = [
    "Stim surface_code:rotated_memory_{basis} with uniform depolarization",
    "after_clifford_depolarization=p, before_round_data_depolarization=p, before_measure_flip_probability=p, after_reset_flip_probability=p",
    "0.005",
    "20,000-50,000",
    "No",
    "No",
    "No"
]
generationV6 = [
    "Stim surface_code:rotated_memory_{basis} with uniform depolarization",
    "after_clifford_depolarization=p, before_round_data_depolarization=p, before_measure_flip_probability=p, after_reset_flip_probability=p",
    "0.005",
    "20,000",
    "No (placeholder zeros in embedding)",
    "No",
    "No"
]
generationAlpha = [
    "SI1000 (Pauli noise, non-uniform strengths), Pauli+ (cross-talk, leakage, amplitude damping), DEMs (fitted to device)",
    "SI1000: measurement noise = 5p, single-qubit/idle = p/10, 2-qubit = p. Pauli+ adds leakage channels, cross-talk unitaries",
    "~0.001 (Pauli + tuned for Lambda ~ 4)",
    "Up to 2.5 billion (pretrain), 10^5-10^8 (finetune)",
    "Yes, modeleted in Pauli+ simulator (states)",
    "Yes, 1D analogue readout with SNR and amplitude damping",
    "Yes, Pauli-twirled correlated channels on groups of up to 4 qubits"
]
generationDF = pd.DataFrame({
    "Aspect": generationApsect,
    "v3": generationV3,
    "v5": generationV5,
    "v6": generationV6,
    "AlphaQubit": generationAlpha
})
generationDF = generationDF.set_index("Aspect")

# 1.2 Data Format
formatField = [
    "Primary input",
    "Labels",
    "Supplementary data",
    "Layout metadata",
]
formatV3 = [
    "det_hard or det_soft (N, D)",
    "obs (N, 2) for X and Z",
    "None",
    "layout.json with stab_id, cycle_id, x, y coordinates, num_detectors, distance"
]
formatV5 = [
    "det_hard or det_soft (N, D)",
    "obs (N, 1) single basis",
    "None",
    "Same"
]
formatV6 = [
    "det_hard (N, D) detection events + meas_hard (N, D) measurements",
    "obs (N, 1) single basis",
    "meas_hard reconstructed via cumulative XOR of events per stabilizer",
    "Same + stab_type (on-basis vs off-basis per stabilizer)"
]
formatAlpha = [
    "Detection events + measurements (soft probabilities) + leakage + leakage events",
    "Logical error label per basis",
    "Intermediate labels at every round (for simulated data only, not used at inference)",
    "Per-stabilizer spatial layout, stabilizer types, circuit connectivity"
]
formatDF = pd.DataFrame({
    "Field": formatField,
    "v3": formatV3,
    "v5": formatV5,
    "v6": formatV6,
    "AlphaQubit": formatAlpha
})
formatDF = formatDF.set_index("Field")

# 1.3 Input Representation to the Model
inputChannel = [
    "Detection events",
    "Measurements",
    "Leakage probability",
    "Leakage event",
    "Stabilizer index",
    "Cycle index"
]
inputV3 = [
    "Yes (sole input)",
    "No (events only)",
    "No",
    "No",
    "Learned embedding",
    "Learned embedding"
]
inputV5 = [
    "Yes (sole input)",
    "No (events only)",
    "No",
    "No",
    "Learned embedding",
    "Learned embedding"
]
inputV6 = [
    "Yes (via proj_event)",
    "Yes (via proj_meas, cumulative XOR of events)",
    "Placeholder zeros (via proj_leak)",
    "Placeholder zeros (via proj_event_leak)",
    "Learned embedding",
    "Learned embedding",
]
inputAlpha = [
    "Yes (both hard and soft)",
    "Yes (soft posterior probabilities, found to improve over events alone)",
    "Yes (post2 = posterior P)",
    "Yes (temporal difference of leakage)",
    "Learned embedding (+ relative positional for multi-distance)",
    "Implicit via recurrent processing"
]
inputDF = pd.DataFrame({
    "Input channel": inputChannel,
    "v3": inputV3,
    "v5": inputV5,
    "v6": inputV6,
    "AlphaQubit": inputAlpha
})
inputDF = inputDF.set_index("Input channel")

# Model architecture


# Parameters and Hyperparameters

#...

# Sort by version
def getAll(desDf: pd.DataFrame):
    return desDf

def getV3(desDf: pd.DataFrame):
    returnDf = desDf.copy()
    dropColumns = ["v5", "v6"]
    returnDf = returnDf.drop(dropColumns, axis=1)
    return returnDf

def getV5(desDf: pd.DataFrame):
    returnDf = desDf.copy()
    dropColumns = ["v3", "v6"]
    returnDf = returnDf.drop(dropColumns, axis=1)
    return returnDf

def getV6(desDf: pd.DataFrame):
    returnDf = desDf.copy()
    dropColumns = ["v3", "v5"]
    returnDf = returnDf.drop(dropColumns, axis=1)
    return returnDf

# Overview
def getOverview(version = "All"):
    match (version):
        case "v3":
            return getV3(overviewDF)
        case "v5":
            return getV5(overviewDF)
        case "v6":
            return getV6(overviewDF)
        case _:
            return getAll(overviewDF)

# Structure   
def getGeneration(version = "All"):
    match (version):
        case "v3":
            return getV3(generationDF)
        case "v5":
            return getV5(generationDF)
        case "v6":
            return getV6(generationDF)
        case _:
            return getAll(generationDF)

def getFormat(version = "All"):
    match (version):
        case "v3":
            return getV3(formatDF)
        case "v5":
            return getV5(formatDF)
        case "v6":
            return getV6(formatDF)
        case _:
            return getAll(formatDF)

def getInput(version = "All"):
    match (version):
        case "v3":
            return getV3(inputDF)
        case "v5":
            return getV5(inputDF)
        case "v6":
            return getV6(inputDF)
        case _:
            return getAll(inputDF)
        
def getStructure(version = "All"):
    return getGeneration(version), getFormat(version), getInput(version)