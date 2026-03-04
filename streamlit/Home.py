import streamlit as st

# TODO decide what goes in the homepage

st.title("Quantum Error Correction: A Neural Network Decoder for Surface Codes")

# Maybe change this section's contents
st.header("What is quantum error correction?")
st.text("Quantum error correction - or QEC for short - is an algorithm known to identify and fix errors in quantum computers. This error-correcting algorithm is able to draw from validated mathematical approaches used to engineer special “radiation-hardened” classical microprocessors deployed in space or other extreme environments where errors are much more likely to occur.")
st.text("In QEC quantum information stored in a single qubit is distributed across other supporting qubits; we say that this information is \"encoded\" in a logical quantum bit. This procedure protects the integrity of the original quantum information even while the quantum processor runs - but at a cost in terms of how many qubits are required. Overall, the worse your noise is, the more qubits you need.")

st.header("What is a transformer?")
st.text("A Transformer is a neural network architecture introduced in the 2017 paper \"Attention is All You Need\" by Vaswani et al. Unlike previous models (RNNs, LSTMs) that process sequences step-by-step, transformers process entire sequences at once using attention mechanisms.")
st.subheader("Key Innovation: Self-Attention")
st.text("The transformer can look at all words in a sentence simultaneously and figure out which words are most relevant to understanding each other word. For example, in \"The animal didn't cross the street because it was too tired\", the model learns that \"it\" refers to \"animal\", not \"street\".")

st.header("How does a neural network decode quantum errors?")
st.subheader("The Syndrome")
st.text("A neural network decoder analyzes syndrome data produced by a quantum error-correcting code to determine whether errors have occurred on the underlying qubits. In quantum computers, directly measuring a qubit would collapse its quantum state, destroying the superposition that gives quantum computing its power. Instead, quantum error correction measures ancilla (support) qubits that perform parity checks on groups of data qubits. These measurements produce a sequence of values called a syndrome, which indicates whether a change in parity has occurred. Such changes reveal detection events that suggest a possible error, such as a bit flip or phase flip, without directly measuring the quantum information itself.")
st.subheader("The Decoder")
st.text("A transformer-based neural network decoder takes these syndrome sequences and learns patterns of how errors propagate across both space (different qubits) and time (repeated measurement cycles). By processing the full syndrome history with attention mechanisms, the model can identify correlations that indicate when and where an error likely occurred. The network then outputs a prediction—often a binary decision indicating whether the logical qubit state has changed. By continuously tracking these predictions across time steps, the system can determine the correct logical state and apply the appropriate correction. Recent work such as the AlphaQubit decoder demonstrates that transformer-based models can learn these patterns directly from data and outperform traditional decoding algorithms in some experimental settings.")


st.header("Key definitions")
st.text("Lorem — ipsum dolor sit amet consectetur adipiscing elit")
st.text("Quisque — faucibus ex sapien vitae pellentesque sem placerat")
st.text("In — id cursus mi pretium tellus duis convallis")
st.text("Basis — consectetur adipiscing elit")
st.text("LER —")
st.text("Shots —")
st.text("Steps —")
st.text("Distance —")
st.text("Rounds —")