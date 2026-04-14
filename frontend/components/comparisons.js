// Short summary
const summaryAll = "<p>Transformer versions 1, 2, and 4 are not included in this data. V1 was the first attempt at training, and produced no output. V2 never became a functional transformer model. V4 was functional, but was too similar to v3 to warrant its own results.</p>"
const summaryV3 = "<p>This is the summary for V3</p>"
const summaryV5 = "<p>This is the summary for V5</p>"
const summaryV6 = "<p>This is the summary for V6</p>"
const summary = {"All": summaryAll, "V3": summaryV3, "V5": summaryV5, "V6": summaryV6}



// Short summary
export default function getSummary(version = "All") {
    return summary[version]
}