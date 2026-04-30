// Short summary
const summaryAll = "<p>Transformer versions 1, 2, and 4 are not included in this data. V1 was the first attempt at training, and produced no output. V2 never became a functional transformer model. V3 generated results, but we were unable to obtain evaulation data for this model. V4 was functional, but was too similar to v3 to warrant its own results.</p>"
const summaryV3 = "<p>Transformer version 3 does not have data available at this time.</p>"
const summaryV5 = "<p>This is the summary for V5.</p>"
const summaryV6 = "<p>This is the summary for V6.</p>"
const summaryV7 = "<p>Transformer version 7 does not have data available at this time."
const summary = {"All": summaryAll, "V3": summaryV3, "V5": summaryV5, "V6": summaryV6, "V7": summaryV7}

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
    "Multi-GPU support",
    "LR per distance",
    "Dilations per distance",
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
    "Stim cirtuit depolarizing noise (fresh)",
    "Single-stage",
    "No",
    "No (fixed 1.3e-4)",
    "No (fixed)",
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
    "Stim cirtuit depolarizing noise (fresh)",
    "Single-stage",
    "No",
    "No (fixed 1.3e-4)",
    "No (fixed)",    
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
    "Stim cirtuit depolarizing noise (fresh)",
    "Single-stage",
    "No",
    "No (fixed 1.3e-4)",
    "No (fixed)",    
    "d3",
    6
]
const overviewT7 = [
    "PyTorch",
    "Mixed-basis (X+Z trained jointly)",
    "ReadoutResNet",
    "4 (meas, event, leak, event_leak)",
    "2-layer ResNet after summation",
    "Seperate final_embed + learned off-basis vector",
    "Tzu-Chen fixed dataset (Stim, fixed)",
    "Single-stage (pretrain)",
    "Single-stage (pretrain)",
    "Yes (DDP, torchrun)",
    "Yes (Table S3 of paper)",
    "Yes (Table S4 of paper)",
    "d=3",
    25
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
    "Yes (TPU pods)",
    "Yes (Table S3 of paper)",
    "Yes (Table S4 of paper)",
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
            label: "V7",
            data: overviewT7
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
    "Data mutability"
]
const generationV3 = [
    "Stim surface_code: rotated_memory_{basis} with uniform depolarization",
    "after_clifford_depolarization =p, before_round_data_depolarization =p, before_measure_flip_probability =p, after_reset_flip_probability =p",
    "0.005",
    "20,000-50,000",
    "No",
    "No",
    "No",
    "Fresh each run"
]
const generationV5 = [
    "Stim surface_code: rotated_memory_{basis} with uniform depolarization",
    "after_clifford_depolarization =p, before_round_data_depolarization =p, before_measure_flip_probability =p, after_reset_flip_probability =p",
    "0.005",
    "20,000-50,000",
    "No",
    "No",
    "No",
    "Fresh each run"
]
const generationV6 = [
    "Stim surface_code: rotated_memory_{basis} with uniform depolarization",
    "after_clifford_depolarization =p, before_round_data_depolarization =p, before_measure_flip_probability =p, after_reset_flip_probability =p",
    "0.005",
    "20,000",
    "No (placeholder zeros in embedding)",
    "No",
    "No",
    "Fresh each run"
]
const generationV7 = [
    "Same noise model, but using the fixed Tzu-Chen dataset rather than freshly generated circuits",
    "Same as Trans3 (fixed in dataset)",
    "0.005 (fixed in dataset)",
    "~1M per distance (streaming from pretrain.h5)",
    "No (placeholder zeros)",
    "No",
    "No",
    "Fixed (Tzu-Chen dataset, reproducible)"
]
const generationAlpha = [
    "SI1000 (Pauli noise, non-uniform strengths), Pauli+ (cross-talk, leakage, amplitude damping), DEMs (fitted to device)",
    "SI1000: measurement noise = 5p, single-qubit/idle = p/10, 2-qubit = p. Pauli+ adds leakage channels, cross-talk unitaries",
    "~0.001 (Pauli + tuned for Lambda ~ 4)",
    "Up to 2.5 billion (pretrain), 10^5-10^8 (finetune)",
    "Yes, modeleted in Pauli+ simulator (states)",
    "Yes, 1D analogue readout with SNR and amplitude damping",
    "Yes, Pauli-twirled correlated channels on groups of up to 4 qubits",
    "Fixed (experimental + simulator)"
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
            label: "V7",
            data: generationV7
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
    "Storage format",
    "Layout metadata",
]
const formatV3 = [
    "det_hard or det_soft (N, D)",
    "obs (N, 2) for X and Z",
    "None",
    "Individual folder per (basis, distance, rounds, seed)",
    "layout.json with stab_id, cycle_id, x, y coordinates, num_detectors, distance"
]
const formatV5 = [
    "det_hard or det_soft (N, D)",
    "obs (N, 1) single basis",
    "None",
    "Individual folder per (basis, distance, rounds, seed)",
    "Same"
]
const formatV6 = [
    "det_hard (N, D) detection events + meas_hard (N, D) measurements",
    "obs (N, 1) single basis",
    "meas_hard reconstructed via cumulative XOR of events per stabilizer",
    "Individual folder per (basis, distance, rounds, seed)",
    "Same + stab_type (on/off basis)"
]
const formatV7 = [
    "det_hard (N, D) + meas_hard (N, D), stored in pretrain.h5",
    "obs (N, 1) single basis",
    "meas_hard reconstructed via cumulative XOR of events per stabilizer",
    "Single pretrain.h5 HDF5 file (all 130 subdirectories compressed into 157MB)",
    "Same as Trans6 (embedded in HDF5)"
]
const formatAlpha = [
    "Detection events + measurements (soft probabilities) + leakage + leakage events",
    "Logical error label per basis",
    "Intermediate labels at every round (for simulated data only, not used at inference)",
    "Not specified",
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
            label: "V7",
            data: formatV7
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
const inputV7 = [
    "Yes (via proj_event)",
    "Yes (via proj_meas)",
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
            label: "V7",
            data: inputV7
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
    "Separate final_embed module; off-basis stabilizers get offbasis_final_emb"
]
const stabilizerV7 = [
    "4 (proj_meas, proj_event, proj_leak, proj_event_leak)",
    "stab_emb(i) + cycle_emb(n)",
    "2-layer residual blocks (_EmbedResBlock) after summation",
    "Separate final_embed module; off-basis stabilizers get offbasis_final_emb"
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
            label: "V7",
            data: stabilizerV7
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
    "Dilation per distance",
    "Learned padding",
    "State combination"
]
const syndromeV3 = [
    "3 (scaling) / 4 (internal)",
    "Alg 2: B'=W_b B, per-head Q,K,V projections, S=QK^T + B', softmax. V uses d_mid (not d_attn). Output projection: W_o(H*d_mid -> d_d)",
    "Alg 3: W_1 expands to w*d_d, split in half, GELU(a) * g, W_2 projects back to d_d",
    "Alg 4: scatter to (d+1)x(d+1) grid -> per-layer: LN -> 3x3 Conv -> GELU -> [1x1 proj if c≠d_d] -> residual -> gather",
    "Shared across all L layers",
    "Fixed (not from paper)",
    "Yes (nn.Parameter for empty grid cells)",
    "(X + S) / sqrt(2)"
]
const syndromeV5 = [
    "3",
    "Alg 2: B'=W_b B, per-head Q,K,V projections, S=QK^T + B', softmax. V uses d_mid (not d_attn). Output projection: W_o(H*d_mid -> d_d)",
    "Alg 3: W_1 expands to w*d_d, split in half, GELU(a) * g, W_2 projects back to d_d",
    "Alg 4: scatter to (d+1)x(d+1) grid -> per-layer: LN -> 3x3 Conv -> GELU -> [1x1 proj if c≠d_d] -> residual -> gather",
    "Shared across all L layers",
    "Fixed (not from paper)",
    "Yes (nn.Parameter for empty grid cells)",
    "(X + S) / sqrt(2)"
]
const syndromeV6 = [
    "3",
    "Alg 2: B'=W_b B, per-head Q,K,V projections, S=QK^T + B', softmax. V uses d_mid (not d_attn). Output projection: W_o(H*d_mid -> d_d)",
    "Alg 3: W_1 expands to w*d_d, split in half, GELU(a) * g, W_2 projects back to d_d",
    "Alg 4: scatter to (d+1)x(d+1) grid -> per-layer: LN -> 3x3 Conv -> GELU -> [1x1 proj if c≠d_d] -> residual -> gather",
    "Separate per layer (nn.ModuleList)",
    "Fixed (not from paper)",
    "Yes (nn.Parameter for empty grid cells)",
    "(X + S) / sqrt(2)"
]
const syndromeV7 = [
    "3",
    "Alg 2: B'=W_b B, per-head Q,K,V projections, S=QK^T + B', softmax. V uses d_mid (not d_attn). Output projection: W_o(H*d_mid -> d_d)",
    "Alg 3: W_1 expands to w*d_d, split in half, GELU(a) * g, W_2 projects back to d_d",
    "Alg 4: scatter to (d+1)x(d+1) grid -> per-layer: LN -> 3x3 Conv -> GELU -> [1x1 proj if c≠d_d] -> residual -> gather",
    "Separate per layer (nn.ModuleList)",
    "Per-distance from Table S4",
    "Yes (nn.Parameter for empty grid cells)",
    "(X + S) / sqrt(2)"
]
const syndromeAlpha = [
    "3",
    "Pseudocode algorithms 1, 2", // Return and fix
    "Alg 3: W_1 expands to w*d_d, split in half, GELU(a) * g, W_2 projects back to d_d",
    "Alg 4: scatter to (d+1)x(d+1) grid -> per-layer: LN -> 3x3 Conv -> GELU -> [1x1 proj if c≠d_d] -> residual -> gather",
    "Seperate per layer (standard)",
    "Per-distance (Table S4)",
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
            label: "V7",
            data: syndromeV7
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
    "7 features in full mode",
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
    "7 features",
    "Yes (static part; dynamic event features per round)"
]
const attentionV7 = [
    "AttentionBiasProvider (full, default)",
    "Spatial coords + offsets + Manhattan + same-type bit + event correlations (full 7-feature)",
    "7 features",
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
            label: "V7",
            data: attentionV7
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
    "Mixed-basis pooling"
]
const readoutV3 = [
    "LayerNorm -> mean pool over stabilizers -> Linear(d_model, 1)",
    "No (mean pool destroys spatial info)",
    "0",
    "d_model (256)",
    "N/A (dual head)"
]
const readoutV5 = [
    "LayerNorm -> ReadoutResNet (global mean pool)",
    "Yes (scatter to 2D grid -> 2x2 conv)",
    "16 (default)",
    "readout_dim (48)",
    "N/A (single basis)"
]
const readoutV6 = [
    "LayerNorm -> ReadoutResNet (directional pool)",
    "Yes (scatter + conv, directional pooling matching paper)",
    "16 (default)",
    "readout_dim (48)",
    "torch.where selects X or Z pool direction per sample"
]
const readoutV7 = [
    "LayerNorm -> ReadoutResNet (directional pool)",
    "Yes (scatter + conv, directional pooling matching paper)",
    "16 (default)",
    "readout_dim (48)",
    "torch.where selects X or Z pool direction per sample"
]
const readoutAlpha = [
    "Scatter to 2D -> 2x2 Conv -> dim reduction -> mean pool perpendicular to logical observables -> ResNet -> Linear",
    "Yes (scatter + conv, pooling perpendicular to logical observables)",
    "16 (Sycamore) / 4 (scaling)",
    "64 (Sycamore) / 32 (scaling)",
    "Seperate decoders per basis"
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
            label: "V7",
            data: readoutV7
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
const auxiliaryV7 = [
    "Yes (linear -> GELU -> linear per stabilizer)",
    "0.02 (with cosine annealing schedule)",
    "Not used",
    "Cosine annealing (warmup 30% -> anneal to 0)",
]
const auxiliaryAlpha = [
    "Yes (linear projection + logistic output)",
    "0.02 (slightly detracts from final performance but faster training)",
    'Used during pretraining (simulated data provides alternative "last rounds" at every step)',
    "Not specified"
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
            label: "V7",
            data: auxiliaryV7
        },
        {
            label: "AlphaQubit",
            data: auxiliaryAlpha
        }
    ]
}

// Parameters and Hyperparameters
// 3.1 Model Parameters
const modelParameters = [
    "d_model",
    "num_head (H)",
    "key_size (d_attn)",
    "d_min",
    "syndrome_layers",
    "dense_widen",
    "conv_layers",
    "conv_dim",
    "bias_dm (db)",
    "bias_residual_layers",
    "readout_resnet_layers",
    "readout_dim",
    "featured_embed_resnet_layers",
    "Total params (approx)"
]
const modelV3Scale = [
    "256",
    "4",
    "32",
    "32",
    "3",
    "5",
    "3",
    "128",
    "48",
    "8",
    "16",
    "48",
    "2",
    "~5M (d=3)"
]
const modelV3Internal = [
    "256",
    "8",
    "64",
    "64",
    "4",
    "4",
    "2",
    "128/256",
    "32",
    "-",
    "N/A (linear)",
    "N/A",
    "0",
    "Varies"
]
const modelV5 = [
    "128",
    "4",
    "32",
    "32",
    "3",
    "5",
    "3",
    "128",
    "48",
    "8",
    "16",
    "48",
    "2 (from config, but not used in embedding class)",
    "~1-2M (d=3, d_model=128)"
]
const modelV6 = [
    "256",
    "4",
    "32",
    "32",
    "3",
    "5",
    "3",
    "128",
    "48",
    "8",
    "16",
    "48",
    "2",
    "~5M (d=3)"
]
const modelV7 = [
    "256",
    "4",
    "32",
    "32",
    "3",
    "5",
    "3",
    "128",
    "48",
    "8",
    "16",
    "48",
    "2",
    "~5M (d=3)"
]
const modelSycamore = [
    "320",
    "4",
    "32",
    "-",
    "-",
    "3",
    "5",
    "3",
    "160",
    "48",
    "8",
    "16",
    "64",
    "2",
    "~5.4M across all distances"
]
const modelScaling = [
    "256",
    "4",
    "32",
    "-",
    "3",
    "5",
    "3",
    "128",
    "- (no attention bias in scaling)",
    "-",
    "4",
    "32",
    "2",
    "~5.4M across all distances"
]
const modelDict = {
    label: "Parameter",
    labels: modelParameters,
    datasets: [
        {
            label: "Trans3 (Scaling)",
            data: modelV3Scale,
        },
        {
            label: "Trans3 (Internal)",
            data: modelV3Internal,
        },
        {
            label: "Trans5",
            data: modelV5
        },
        {
            label: "Trans6",
            data: modelV6
        },
        {
            label: "Trans7",
            data: modelV7
        },
        {
            label: "AlphaQubit (Sycamore)",
            data: modelSycamore,
        },
        {
            label: "AlphaQubit (Scaling)",
            data: modelScaling
        }
    ]
    
}

// 3.2 Training Hyperparameters
const trainingHyper = [
    "Optimizer",
    "Learning rate",
    "Weight decay",
    "Beta1, Beta2",
    "Batch size",
    "LR schedule",
    "LR decay steps",
    "Gradient clipping",
    "EMA",
    "Total training steps",
    "Noise curriculum",
    "Rounds curriculem",
    "Next-stab weight",
    "Loss",
    "Resume support",
    "Multi-GPU"
]
const trainingV3 = [
    "Lion",
    "1.3e-4",
    "1e-7",
    "0.9, 0.95",
    "256 -> 1024",
    "Piecewise constant (0.7x at milestones)",
    "400K, 800K, 1.6M",
    "1.0",
    "Yes (alpha=1e-4)",
    "2M",
    "Not implemented",
    "Not implemented",
    "0.02 (constant)",
    "BCE with logits",
    "No",
    "No"
]
const trainingV5 = [
    "Lion",
    "1.3e-4",
    "1e-7",
    "0.9, 0.95",
    "128 (fixed)",
    "Piecewise constant (0.7x at milestones)",
    "400K, 800K, 1.6M",
    "1.0",
    "Yes (alpha=1e-4)",
    "500 (default)",
    "No",
    "No",
    "0.02 (cosine annealing)",
    "BCE with logits",
    "No",
    "No"
]
const trainingV6 = [
    "Lion",
    "1.3e-4",
    "1e-7",
    "0.9, 0.95",
    "256 -> 1024",
    "Piecewise constant (0.7x at milestones)",
    "400K, 800K, 1.6M",
    "1.0",
    "Yes (alpha=1e-4)",
    "2M",
    "Not implemented",
    "Not implemented",
    "0.02 (constant)",
    "BCE with logits",
    "No",
    "No"
]
const trainingV7 = [
    "Lion",
    "Per-distance (Table S3): 1.3e-4 (d=3), 1.15e-4 (d=5), 1.0e-4 (d=7), 7e-5 (d=9), 5e-5 (d=11)",
    "1e-7",
    "0.9, 0.95",
    "256 (effective, 128/GPU × 2 GPUs) -> 1024 at step 800k",
    "Piecewise constant (0.7x at milestones)",
    "400K, 800K, 1.6M",
    "1.0",
    "Yes (alpha=1e-4)",
    "1M",
    "No",
    "No",
    "0.02 (cosine annealing, warmup 30%)",
    "BCE with logits",
    "Yes (_resume.pth with optimizer state)",
    "Yes (DDP, torchrun, NCCL)"
]
const trainingSycamore = [
    "Lamb",
    "3.46e-4 (d=3), 2.45e-4 (d=5)",
    "1e-5 (pretrain), 0.08 (finetune, relative to pretrained weights)",
    "0.9, 0.95 (b2 for Lamb)",
    "256 -> 1024 (increase at 4M steps)",
    "Piecewise constant (0.7x at milestones)",
    "{0.8, 2, 4, 10, 20} x 10^5",
    "Not specified",
    "Yes (alpha=1e-4)",
    "Up to 2B samples",
    "Yes",
    "Not mentioned",
    "0.02",
    "Cross-entropy",
    "N/A",
    "Yes (TPU)"
]
const trainingScaling = [
    "Lion",
    "7e-4 (d=3) to 3e-4 (d=11)",
    "1e-7",
    "0.9, 0.95",
    "256-> 1024",
    "Cosine",
    "Cosine schedule",
    "Not specified",
    "Yes",
    "Up to 2.5B samples",
    "Yes (scale noise from 0.5 to 1.0 during training)",
    "Yes (3 -> 6 -> 12 -> 25)",
    "0.01",
    "Cross-entropy",
    "N/A",
    "Yes (TPU)"
]
const trainingDict = {
    label: "Hyperparameter",
    labels: trainingHyper,
    datasets: [
        {
            label: "Trans3",
            data: trainingV3,
        },
        {
            label: "Trans5",
            data: trainingV5
        },
        {
            label: "Trans6",
            data: trainingV6
        },
        {
            label: "Trans3",
            data: trainingV7
        },
        {
            label: "AlphaQubit (Sycamore)",
            data: trainingSycamore,
        },
        {
            label: "AlphaQubit (Scaling)",
            data: trainingScaling
        }
    ]
    
}

// 3.3 Convolution Dilations
const convolutionDistance = [
    "d=3",
    "d=5",
    "d=7",
    "d=9",
    "d=11"
]
const convolutionV3 = [
    "[1, 1] or [1, 1, 1]",
    "[1, 1, 1]",
    "-",
    "-",
    "-"
]
const convolutionV5 = [
    "[1, 1, 1]",
    "[1, 1, 1]",
    "-",
    "-",
    "-"
]
const convolutionV6 = [
    "[1, 1, 1]",
    "-",
    "-",
    "-",
    "-",
]
const convolutionV7 = [
    "[1, 1, 1] (Table S4)",
    "[1, 1, 2] (Table S4)",
    "[1, 2, 4] (Table S4)",
    "[1, 2, 4] (Table S4)",
    "[1, 2, 4] (Table S4)"   
]
const convolutionSycamore = [
    "[1, 1, 1]",
    "[1, 1, 2]",
    "-",
    "-",
    "-",
]
const convolutionScaling = [
    "[1, 2, 4]",
    "[1, 2, 4]",
    "[1, 2, 4]",
    "[1, 2, 4]",
    "[1, 2, 4]",
]
const convolutionDict = {
    label: "Distance",
    labels: convolutionDistance,
    datasets: [
        {
            label: "Trans3",
            data: convolutionV3,
        },
        {
            label: "Trans5",
            data: convolutionV5
        },
        {
            label: "Trans6",
            data: convolutionV6
        },
        {
            label: "Trans7",
            data: convolutionV7
        },
        {
            label: "AlphaQubit (Sycamore)",
            data: convolutionSycamore,
        },
        {
            label: "AlphaQubit (Scaling)",
            data: convolutionScaling
        }
    ]
    
}

// Key archictectural differences
// 4.1 Trans3 -> Trans5 Changes
const V3V5 = [
    "<b>Single-basis training</b>: Removed dual X/Z output heads. Each model trains on one basis only, matching the AlphaQubit approach where decoders are basis-specific.",
    "<b>ReadoutResNet</b>: Replaced Linear(d_model, 1) with a 6-stage pipeline: scatter to 2D -> 2x2 stride-2 conv -> 1x1 dim reduction -> global mean pool -> 16 residual blocks -> Linear(readout_dim, 1).",
    "<b>Reduced d_model</b>: Default 128 (down from 256) as a compromise between capacity and compute.",
    "<b>Next-stab cosine annealing</b>: Auxiliary loss weight anneals from 0.02 -> 0 over training (warmup 30% of steps at full weight, then cosine decay).",
    "<b>Padding strategy</b>: Changed from truncate-to-S_min to pad-to-S_max with boolean pad_mask, preserving all detector information."
]

// 4.2 Trans5 -> Trans6 Changes
const V5V6 = [
    "<b>4-input embedding</b>: Four separate linear projections for measurement, event, leakage, and leakage-event inputs, summed with stabilizer/cycle embeddings, then two residual blocks.",
    "<b>Measurement input</b>: Pre-computes measurements from detection events via cumulative XOR per stabilizer.",
    "<b>Final-round handling</b>: Separate final_embed module for on-basis stabilizers in the last cycle; off-basis gets offbasis_final_emb.",
    "<b>stab_type metadata</b>: Layout tracks on-basis vs off-basis per stabilizer.",
    "<b>Leakage-ready architecture</b>: Accepts zero-valued leakage inputs for future finetuning without architecture changes.", 
    "<b>Separate conv weights per layer</b>: nn.ModuleList of independent conv blocks (vs. shared weights in Trans3/5)."
]

// 4.2 Trans6 -> Trans7 Changes
const V6V7 = [
    "<b>Per-distance learning rates</b> (from paper Table S3): d=3 → 1.3e-4, d=5 → 1.15e-4, d=7 → 1.0e-4, d=9 → 7e-5, d=11 → 5e-5. Prior versions used a fixed 1.3e-4 regardless of distance.",
    "<b>Per-distance dilation schedules</b> (from paper Table S4): d=3 → [1,1,1], d=5 → [1,1,2], d≥7 → [1,2,4]. Prior versions used fixed dilations that didn't follow the paper.",
    "<b>Multi-GPU DDP training</b>: torchrun launches DistributedDataParallel across N GPUs. Gradients are synchronized via NCCL all-reduce. Batch size is set per-GPU to keep the total effective batch at 256 regardless of GPU count.",
    "<b>Fixed Tzu-Chen dataset</b>: Training uses a reproducible fixed dataset rather than freshly generated Stim circuits. Eliminates run-to-run data variation; ensures comparability across experiments.",
    "<b>HDF5 streaming data</b> (pretrain.h5): All 130 surface code subdirectories compressed into a single 157MB file. The ChunkedHDF5Dataset loads data in 50k-sample chunks with background prefetching, and the file is copied to node-local /tmp/ before training to eliminate network I/O overhead.",
    "<b>Next-stab cosine annealing</b>: Restored from Trans5 (was constant in Trans6).",
    "<b>Resume/checkpoint support</b>: _resume.pth saves optimizer momentum states, EMA state, current step, layout, and training history so jobs can be interrupted and continued exactly where they left off.",
    "<b>Auto-requeue SLURM</b>: SLURM signal trap (--signal=B:USR1@120) automatically submits the next job 120s before the time limit, enabling seamless multi-job training runs.",
    "<b>Mixed-basis training</b>: X and Z basis samples are trained together in the same run using torch.where for basis-conditional pooling (avoids aten.nonzero graph breaks in torch.compile).",
]

// 4.3 Trans7 -> vs AlphaQubit (Remaining Gaps)
const gapsFeature = [
    "Per-distance LR",
    "Per-distance dilations",
    "Soft I/Q readouts",
    "Leakage simulation",
    "Cross-talk",
    "Two-stage training",
    "Noise curriculum",
    "Rounds curriculum",
    "Multi-distance training",
    "Ensembling",
    "Intermediate labels",
    "Data scale",
    "Post-selection"
]
const gapsV7 = [
    "✓ Implemented (Table S3)",
    "✓ Implemented (Table S4)",
    "Not implemented (hard binary only)",
    "Placeholder zeros",
    "Not modeled",
    "Not implemented (pretrain only)",
    "Not implemented",
    "Not implemented",
    "Not implemented (one distance per run)",
    "Not implemented",
    "Not used",
    "~1M samples (streaming)",
    "Not implemented"
]
const gapsAlpha = [
    "Yes",
    "Yes",
    "Analogue I/Q signals with posterior probabilities",
    "Pauli+ model with realistic leakage channels",
    "Pauli-twirled correlated channels from CZ interactions",
    "Pretrain on SI1000/DEM -> finetune on Pauli+/experimental",
    "Gradually scale noise from 0.5x to 1.0x during pretrain",
    "Train on 3 -> 6 -> 12 -> 25 rounds progressively",
    "Single model on mixture of d=3 to d=11",
    "15-20 independently trained models, averaged logits",
    "Used during pretraining at every round",
    "Up to 2.5 billion pretrain + 100M finetune per distance",
    "Confidence-based post-selection on probabilistic output"
]
const gapsDict = {
    label: "Feature",
    labels: gapsFeature,
    datasets: [
        {
            label: "Trans 7 Status",
            data: gapsV7
        },
        {
            label: "AlphaQubit Paper",
            data: gapsAlpha
        }
    ]
}

// ...

// Sort by version
function getV3(dict) {
    var copyDict = {}
    Object.assign(copyDict, dict)
    copyDict.datasets = [dict.datasets[0], dict.datasets[4]]
    return copyDict
}

function getV5(dict) {
    var copyDict = {}
    Object.assign(copyDict, dict)
    copyDict.datasets = [dict.datasets[1], dict.datasets[4]]
    return copyDict
}

function getV6(dict) {
    var copyDict = {}
    Object.assign(copyDict, dict)
    copyDict.datasets = [dict.datasets[2], dict.datasets[4]]
    return copyDict
}

function getV7(dict) {
    var copyDict = {}
    Object.assign(copyDict, dict)
    copyDict.datasets = [dict.datasets[3], dict.datasets[4]]
    return copyDict
}

function getCustom(dict, numDataCols = 2, col1, col2, col3 = 0, col4 = 0) {
    var copyDict = {}
    Object.assign(copyDict, dict)
    switch (numDataCols) {
        case 2:
            copyDict.datasets = [dict.datasets[col1], dict.datasets[col2]]
            break
        case 3:
            copyDict.datasets = [dict.datasets[col1], dict.datasets[col2], dict.datasets[col3]]
            break
        case 4:
            copyDict.datasets = [dict.datasets[col1], dict.datasets[col2], dict.datasets[col3], dict.datasets[col4]]
    }
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
        case "V7":
            return getV7(overviewDict)
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
        case "V7":
            return getV7(generationDict)
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
        case "V7":
            return getV7(formatDict)
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
        case "V7":
            return getV7(inputDict)
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
        case "V7":
            return getV7(stabilizerDict)
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
        case "V7":
            return getV7(syndromeDict)
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
        case "V7":
            return getV7(attentionDict)
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
        case "V7":
            return getV7(readoutDict)
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
        case "V7":
            return getV7(auxiliaryDict)
        default:
            return auxiliaryDict
    }
}

export function getArchitecture(version = "All") {
    return [getStabilizer(version), getSyndrome(version), getAttention(version), getReadout(version), getAuxiliary(version)]
}

// Parameters and Hyperparameters
function getModel(version = "All") {
    switch (version) {
        case "V3":
            return getCustom(modelDict, 4, 0, 1, 5, 6)
        case "V5":
            return getCustom(modelDict, 3, 2, 5, 6)
        case "V6":
            return getCustom(modelDict, 3, 3, 5, 6)
        case "V7":
            return getCustom(modelDict, 3, 4, 5, 6)
        default:
            return modelDict
    }
}

function getTraining(version = "All") {
    switch (version) {
        case "V3":
            return getCustom(trainingDict, 3, 0, 4, 5)
        case "V5":
            return getCustom(trainingDict, 3, 1, 4, 5)
        case "V6":
            return getCustom(trainingDict, 3, 2, 4, 5)
        case "V7":
            return getCustom(trainingDict, 3, 3, 4, 5)
        default:
            return trainingDict
    }
}

function getConvolution(version = "All") {
    switch (version) {
        case "V3":
            return getCustom(convolutionDict, 3, 0, 4, 5)
        case "V5":
            return getCustom(convolutionDict, 3, 1, 4, 5)
        case "V6":
            return getCustom(convolutionDict, 3, 2, 4, 5)
        case "V7":
            return getCustom(convolutionDict, 3, 3, 4, 5)
        default:
            return convolutionDict
    }
}

export function getParameters(version = "All") {
    return [getModel(version), getTraining(version), getConvolution(version)]
}

// Key archictectural differences
export function getDifferences(version = "All") {
    var differences = []
    switch (version) {
        case "V3":
            differences.push(V3V5)
            break
        case "V5":
            differences.push(V3V5)
            differences.push(V5V6)
            break
        case "V6":
            differences.push(V5V6)
            differences.push(V6V7)
            break
        case "V7":
            differences.push(V6V7)
            differences.push(gapsDict)
            break
        default:
            differences.push(V3V5)
            differences.push(V5V6)
            differences.push(V6V7)
            differences.push(gapsDict)
    }
    return differences
}

// Data Flow Diagrams