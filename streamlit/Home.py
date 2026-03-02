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
st.text("Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.")
st.text("Lorem ipsum dolor sit amet consectetur adipiscing elit. Quisque faucibus ex sapien vitae pellentesque sem placerat. In id cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas. Iaculis massa nisl malesuada lacinia integer nunc posuere. Ut hendrerit semper vel class aptent taciti sociosqu. Ad litora torquent per conubia nostra inceptos himenaeos.")

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