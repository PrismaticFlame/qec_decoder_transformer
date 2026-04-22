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

// Sort by version
function getV3(dict) {
    dict.datasets = [dict.datasets[0], dict.datasets[3]]
    return dict
}

function getV5(dict) {
    dict.datasets = [dict.datasets[1], dict.datasets[3]]
    return dict
}

function getV6(dict) {
    dict.datasets = [dict.datasets[2], dict.datasets[3]]
    return dict
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