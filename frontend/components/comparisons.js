// Short summary
const summaryAll = "<p>Transformer versions 1, 2, and 4 are not included in this data. V1 was the first attempt at training, and produced no output. V2 never became a functional transformer model. V4 was functional, but was too similar to v3 to warrant its own results.</p>"
const summaryV3 = "<p>This is the summary for V3</p>"
const summaryV5 = "<p>This is the summary for V5</p>"
const summaryV6 = "<p>This is the summary for V6</p>"
const summary = {"All": summaryAll, "V3": summaryV3, "V5": summaryV5, "V6": summaryV6}

// Overview
const overviewProperties = [
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
const overviewT3 = [
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
const overviewT5 = [
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
const overviewT6 = [
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
const overviewAlpha = [
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
const overviewDict = {
    label: "Property",
    labels: overviewProperties,
    datasets: [
        {
            label: "V3",
            data: overviewT3
        },
        {
            label: "V5",
            data: overviewT5
        },
        {
            label: "V6",
            data: overviewT6
        },
        {
            label: "AlphaQubit (Paper)",
            data: overviewAlpha
        }
    ]
}

// Data input and structure
// 1.1 Data Generation
const generationApsect = [
    "Noise model",
    "Noise parameters",
    "Typical p values",
    "Shots",
    "Leakage",
    "Soft readouts (I/Q)",
    "Cross-talk",
]
const generationV3 = [
    "Stim surface_code: rotated_memory_{basis} with uniform depolarization",
    "after_clifford_depolarization =p, before_round_data_depolarization =p, before_measure_flip_probability =p, after_reset_flip_probability =p",
    "0.005",
    "20,000-50,000",
    "No",
    "No",
    "No"
]
const generationV5 = [
    "Stim surface_code: rotated_memory_{basis} with uniform depolarization",
    "after_clifford_depolarization =p, before_round_data_depolarization =p, before_measure_flip_probability =p, after_reset_flip_probability =p",
    "0.005",
    "20,000-50,000",
    "No",
    "No",
    "No"
]
const generationV6 = [
    "Stim surface_code: rotated_memory_{basis} with uniform depolarization",
    "after_clifford_depolarization =p, before_round_data_depolarization =p, before_measure_flip_probability =p, after_reset_flip_probability =p",
    "0.005",
    "20,000",
    "No (placeholder zeros in embedding)",
    "No",
    "No"
]
const generationAlpha = [
    "SI1000 (Pauli noise, non-uniform strengths), Pauli+ (cross-talk, leakage, amplitude damping), DEMs (fitted to device)",
    "SI1000: measurement noise = 5p, single-qubit/idle = p/10, 2-qubit = p. Pauli+ adds leakage channels, cross-talk unitaries",
    "~0.001 (Pauli + tuned for Lambda ~ 4)",
    "Up to 2.5 billion (pretrain), 10^5-10^8 (finetune)",
    "Yes, modeleted in Pauli+ simulator (states)",
    "Yes, 1D analogue readout with SNR and amplitude damping",
    "Yes, Pauli-twirled correlated channels on groups of up to 4 qubits"
]
const generationDict = {
    label: "Aspect",
    labels: generationApsect,
    datasets: [
        {
            label: "V3",
            data: generationV3
        },
        {
            label: "V5",
            data: generationV5
        },
        {
            label: "V6",
            data: generationV6
        },
        {
            label: "AlphaQubit (Paper)",
            data: generationAlpha
        }
    ]
}

// 1.2 Data Format
const formatField = [
    "Primary input",
    "Labels",
    "Supplementary data",
    "Layout metadata",
]
const formatV3 = [
    "det_hard or det_soft (N, D)",
    "obs (N, 2) for X and Z",
    "None",
    "layout.json with stab_id, cycle_id, x, y coordinates, num_detectors, distance"
]
const formatV5 = [
    "det_hard or det_soft (N, D)",
    "obs (N, 1) single basis",
    "None",
    "Same"
]
const formatV6 = [
    "det_hard (N, D) detection events + meas_hard (N, D) measurements",
    "obs (N, 1) single basis",
    "meas_hard reconstructed via cumulative XOR of events per stabilizer",
    "Same + stab_type (on-basis vs off-basis per stabilizer)"
]
const formatAlpha = [
    "Detection events + measurements (soft probabilities) + leakage + leakage events",
    "Logical error label per basis",
    "Intermediate labels at every round (for simulated data only, not used at inference)",
    "Per-stabilizer spatial layout, stabilizer types, circuit connectivity"
]
const formatDict = {
    label: "Field",
    labels: formatField,
    datasets: [
        {
            label: "V3",
            data: formatV3
        },
        {
            label: "V5",
            data: formatV5
        },
        {
            label: "V6",
            data: formatV6
        },
        {
            label: "AlphaQubit (Paper)",
            data: formatAlpha
        }
    ]
}

// 1.3 Input Representation to the Model
const inputChannel = [
    "Detection events",
    "Measurements",
    "Leakage probability",
    "Leakage event",
    "Stabilizer index",
    "Cycle index"
]
const inputV3 = [
    "Yes (sole input)",
    "No (events only)",
    "No",
    "No",
    "Learned embedding",
    "Learned embedding"
]
const inputV5 = [
    "Yes (sole input)",
    "No (events only)",
    "No",
    "No",
    "Learned embedding",
    "Learned embedding"
]
const inputV6 = [
    "Yes (via proj_event)",
    "Yes (via proj_meas, cumulative XOR of events)",
    "Placeholder zeros (via proj_leak)",
    "Placeholder zeros (via proj_event_leak)",
    "Learned embedding",
    "Learned embedding",
]
const inputAlpha = [
    "Yes (both hard and soft)",
    "Yes (soft posterior probabilities, found to improve over events alone)",
    "Yes (post2 = posterior P)",
    "Yes (temporal difference of leakage)",
    "Learned embedding (+ relative positional for multi-distance)",
    "Implicit via recurrent processing"
]
const inputDict = {
    label: "Channel",
    labels: inputChannel,
    datasets: [
        {
            label: "V3",
            data: inputV3
        },
        {
            label: "V5",
            data: inputV5
        },
        {
            label: "V6",
            data: inputV6
        },
        {
            label: "AlphaQubit (Paper)",
            data: inputAlpha
        }
    ]
}

// Sort by version
function getV3(dict) {
    var copyDict = {}
    Object.assign(copyDict, dict)
    copyDict.datasets = [dict.datasets[0], dict.datasets[3]]
    return copyDict
}

function getV5(dict) {
    var copyDict = {}
    Object.assign(copyDict, dict)
    copyDict.datasets = [dict.datasets[1], dict.datasets[3]]
    return copyDict
}

function getV6(dict) {
    var copyDict = {}
    Object.assign(copyDict, dict)
    copyDict.datasets = [dict.datasets[2], dict.datasets[3]]
    return copyDict
}

// Short summary
export function getSummary(version = "All") {
    return summary[version]
}

// Overview
export function getOverview(version = "All") {
    switch (version) {
        case "V3":
            return getV3(overviewDict)
        case "V5":
            return getV5(overviewDict)
        case "V6":
            return getV6(overviewDict)
        default:
            return overviewDict
    }
}

// Data input and structure
function getGeneration(version = "All") {
    switch (version) {
        case "V3":
            return getV3(generationDict)
        case "V5":
            return getV5(generationDict)
        case "V6":
            return getV6(generationDict)
        default:
            return generationDict
    }
}

function getFormat(version = "All") {
    switch (version) {
        case "V3":
            return getV3(formatDict)
        case "V5":
            return getV5(formatDict)
        case "V6":
            return getV6(formatDict)
        default:
            return formatDict
    }
}

function getInput(version = "All") {
    switch (version) {
        case "V3":
            return getV3(inputDict)
        case "V5":
            return getV5(inputDict)
        case "V6":
            return getV6(inputDict)
        default:
            return inputDict
    }
}
        
export function getStructure(version = "All") {
    return [getGeneration(version), getFormat(version), getInput(version)]
}