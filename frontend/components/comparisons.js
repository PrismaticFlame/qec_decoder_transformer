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
            label: "AlphaQubit",
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
            label: "AlphaQubit",
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
            label: "AlphaQubit",
            data: inputAlpha
        }
    ]
}

// Model architecture
// 2.1 Stabilizer
const stabilizerComponent = [
    "Input projections",
    "Positional info",
    "Post-summation processing",
    "Final-round treatment"
]
const stabilizerV3 = [
    "1 (analog_proj for soft / val_emb for hard)",
    "stab_emb(i) + cycle_emb(n)",
    "None (direct sum: e_stab + e_cycle + e_val)",
    "None"
]
const stabilizerV5 = [
    "1 (analog_proj for soft / val_emb for hard)",
    "stab_emb(i) + cycle_emb(n)",
    "None",
    "None"
]
const stabilizerV6 = [
    "4 (proj_meas, proj_event, proj_leak, proj_event_leak)",
    "stab_emb(i) + cycle_emb(n)",
    "2-layer residual blocks (_EmbedResBlock) after summation",
    "Separate final_embed module; off-basis stabilizers get learned vector offbasis_final_emb"
]
const stabilizerAlpha = [
    "4 linear projections (Extended Data Fig. 4c)",
    "Stabilizer index embedding + (optional) relative positional embedding",
    "2-layer ResNet (paper: Extended Data Fig. 4c)",
    "Separate final-round linear projection for on-basis; single learned embedding for off-basis"
]
const stabilizerDict = {
    label: "Component",
    labels: stabilizerComponent,
    datasets: [
        {
            label: "V3",
            data: stabilizerV3
        },
        {
            label: "V5",
            data: stabilizerV5
        },
        {
            label: "V6",
            data: stabilizerV6
        },
        {
            label: "AlphaQubit",
            data: stabilizerAlpha
        }
    ]
}

// 2.2 Syndrome
const syndromeComponent = [
    "Layers per round",
    "Multi-head attention",
    "Gated dense block",
    "Convolution block",
    "Conv weight sharing",
    "Learned padding",
    "State combination"
]
const syndromeV3 = [
    "3 (scaling) / 4 (internal)",
    "Alg 2: B'=W_b B, per-head Q,K,V projections, S=QK^T + B', softmax. V uses d_mid (not d_attn). Output projection: W_o(H*d_mid -> d_d)",
    "Alg 3: W_1 expands to w*d_d, split in half, GELU(a) * g, W_2 projects back to d_d",
    "Alg 4: scatter to (d+1)x(d+1) grid -> per-layer: LN -> 3x3 Conv -> GELU -> [1x1 proj if c≠d_d] -> residual -> gather",
    "Shared across all L layers (single instance)",
    "Yes (nn.Parameter for empty grid cells)",
    "(X + S) / sqrt(2)"
]
const syndromeV5 = [
    "3",
    "Alg 2: B'=W_b B, per-head Q,K,V projections, S=QK^T + B', softmax. V uses d_mid (not d_attn). Output projection: W_o(H*d_mid -> d_d)",
    "Alg 3: W_1 expands to w*d_d, split in half, GELU(a) * g, W_2 projects back to d_d",
    "Alg 4: scatter to (d+1)x(d+1) grid -> per-layer: LN -> 3x3 Conv -> GELU -> [1x1 proj if c≠d_d] -> residual -> gather",
    "Shared",
    "Yes (nn.Parameter for empty grid cells)",
    "(X + S) / sqrt(2)"
]
const syndromeV6 = [
    "3",
    "Alg 2: B'=W_b B, per-head Q,K,V projections, S=QK^T + B', softmax. V uses d_mid (not d_attn). Output projection: W_o(H*d_mid -> d_d)",
    "Alg 3: W_1 expands to w*d_d, split in half, GELU(a) * g, W_2 projects back to d_d",
    "Alg 4: scatter to (d+1)x(d+1) grid -> per-layer: LN -> 3x3 Conv -> GELU -> [1x1 proj if c≠d_d] -> residual -> gather",
    "Separate per layer (nn.ModuleList of L independent blocks)",
    "Yes (nn.Parameter for empty grid cells)",
    "(X + S) / sqrt(2)"
]
const syndromeAlpha = [
    "3",
    "Pseudocode algorithms 1, 2", // Return and fix
    "Alg 3: W_1 expands to w*d_d, split in half, GELU(a) * g, W_2 projects back to d_d",
    "Alg 4: scatter to (d+1)x(d+1) grid -> per-layer: LN -> 3x3 Conv -> GELU -> [1x1 proj if c≠d_d] -> residual -> gather",
    "Pseudocode ambiguous (separate weights per layer is standard)",
    "Yes (learned padding vector P)",
    "Same (scale factor 1/sqrt(2) to control magnitude)"
]
const syndromeDict = {
    label: "Component",
    labels: syndromeComponent,
    datasets: [
        {
            label: "V3",
            data: syndromeV3
        },
        {
            label: "V5",
            data: syndromeV5
        },
        {
            label: "V6",
            data: syndromeV6
        },
        {
            label: "AlphaQubit",
            data: syndromeAlpha
        }
    ]
}
// 2.3 Attention
const attentionAspect = [
    "Bias type",
    "Features encoded",
    "Event indicator features",
    "Precomputable"
]
const attentionV3 = [
    "ManhattanDistanceBias (simple) or AttentionBiasProvider (full)",
    "Manhattan distance between stabilizers; or spatial coords + offsets + Manhattan + same-type bit + event correlations",
    "7 features (spatial and time-space event correlations) in full mode",
    "Yes (static part)"
]
const attentionV5 = [
    "AttentionBiasProvider (full)",
    "Manhattan distance between stabilizers; or spatial coords + offsets + Manhattan + same-type bit + event correlations",
    "7 features (spatial and time-space event correlations) in full mode",
    "Yes (static part)"
]
const attentionV6 = [
    "AttentionBiasProvider (full, default)",
    "Spatial coords + offsets + Manhattan + same-type bit + event correlations (full 7-feature)",
    "7 features (same as Trans3 full mode)",
    "Yes (static part; dynamic event features per round)"
]
const attentionAlpha = [
    "Learned attention bias embedding (d^2-1 x d^2-1 x 48)",
    "Spatial coords of i,j; signed offset; Manhattan distance; same/different basis bit",
    "7 features (spatial and time-space event products)",
    "Yes (static embedding; dynamic event features require per-round update)"
]
const attentionDict = {
    label: "Aspect",
    labels: attentionAspect,
    datasets: [
        {
            label: "V3",
            data: attentionV3
        },
        {
            label: "V5",
            data: attentionV5
        },
        {
            label: "V6",
            data: attentionV6
        },
        {
            label: "AlphaQubit",
            data: attentionAlpha
        }
    ]
}

// 2.4 Readout
const readoutComponent = [
    "Architecture",
    "Spatial awareness",
    "Residual blocks",
    "Hidden dim",
    "Output heads"
]
const readoutV3 = [
    "LayerNorm -> mean pool over stabilizers -> Linear(d_model, 1)",
    "No (mean pool destroys spatial info)",
    "0",
    "d_model (256)",
    "2 (classifier_x + classifier_z)"
]
const readoutV5 = [
    "LayerNorm -> ReadoutResNet (global mean pool)",
    "Yes (scatter to 2D grid -> 2x2 conv)",
    "16 (default)",
    "readout_dim (48)",
    "1 (single basis)"
]
const readoutV6 = [
    "LayerNorm -> ReadoutResNet (directional pool perpendicular to logical observable)",
    "Yes (scatter + conv, directional pooling matching paper)",
    "16 (default)",
    "readout_dim (48)",
    "1 per basis (K position logits averaged, matching paper)"
]
const readoutAlpha = [
    "Scatter to 2D -> 2x2 Conv -> dim reduction -> mean pool perpendicular to logical observables -> ResNet -> Linear",
    "Yes (scatter + conv, pooling perpendicular to logical observables)",
    "16 (Sycamore) / 4 (scaling)",
    "64 (Sycamore) / 32 (scaling)",
    "1 per basis (d equivalent logical observable predictions averaged during training)"
]
const readoutDict = {
    label: "Component",
    labels: readoutComponent,
    datasets: [
        {
            label: "V3",
            data: readoutV3
        },
        {
            label: "V5",
            data: readoutV5
        },
        {
            label: "V6",
            data: readoutV6
        },
        {
            label: "AlphaQubit",
            data: readoutAlpha
        }
    ]
}

// 2.5 Auxiliary
const auxiliaryAspect = [
    "Next-stabilizer prediction",
    "Loss weight",
    "Intermediate labels",
    "Weight schedule"
]
const auxiliaryV3 = [
    "Yes (linear -> GELU -> linear per stabilizer)",
    "0.02",
    "Not used",
    "Constant"
]
const auxiliaryV5 = [
    "Yes (linear -> GELU -> linear per stabilizer)",
    "0.02 (with cosine annealing schedule)",
    "Not used",
    "Cosine annealing (warmup 30% -> anneal to 0)",
]
const auxiliaryV6 = [
    "Yes (linear -> GELU -> linear per stabilizer)",
    "0.02",
    "Not used",
    "Constant"
]
const auxiliaryAlpha = [
    "Yes (linear projection + logistic output)",
    "0.02 (slightly detracts from final performance but faster training)",
    'Used during pretraining (simulated data provides alternative "last rounds" at every step)',
    "Not specified (constant implied)"
]
const auxiliaryDict = {
    label: "Aspect",
    labels: auxiliaryAspect,
    datasets: [
        {
            label: "V3",
            data: auxiliaryV3
        },
        {
            label: "V5",
            data: auxiliaryV5
        },
        {
            label: "V6",
            data: auxiliaryV6
        },
        {
            label: "AlphaQubit",
            data: auxiliaryAlpha
        }
    ]
}

// Parameters and Hyperparameters

// ...

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

// Model Architecture

function getStabilizer(version = "All") {
    switch (version) {
        case "V3":
            return getV3(stabilizerDict)
        case "V5":
            return getV5(stabilizerDict)
        case "V6":
            return getV6(stabilizerDict)
        default:
            return stabilizerDict
    }
}
function getSyndrome(version = "All") {
    switch (version) {
        case "V3":
            return getV3(syndromeDict)
        case "V5":
            return getV5(syndromeDict)
        case "V6":
            return getV6(syndromeDict)
        default:
            return syndromeDict
    }
}

function getAttention(version = "All") {
    switch (version) {
        case "V3":
            return getV3(attentionDict)
        case "V5":
            return getV5(attentionDict)
        case "V6":
            return getV6(attentionDict)
        default:
            return attentionDict
    }
}

function getReadout(version = "All") {
    switch (version) {
        case "V3":
            return getV3(readoutDict)
        case "V5":
            return getV5(readoutDict)
        case "V6":
            return getV6(readoutDict)
        default:
            return readoutDict
    }
}

function getAuxiliary(version = "All") {
    switch (version) {
        case "V3":
            return getV3(auxiliaryDict)
        case "V5":
            return getV5(auxiliaryDict)
        case "V6":
            return getV6(auxiliaryDict)
        default:
            return auxiliaryDict
    }
}

export function getArchitecture(version = "All") {
    return [getStabilizer(version), getSyndrome(version), getAttention(version), getReadout(version), getAuxiliary(version)]
}