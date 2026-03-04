import streamlit as st

st.title("Links")

# TODO eventually use proper citations
url = "https://arxiv.org/abs/1706.03762"
st.write("[Attention is All You Need](%s) by Vaswani et al" % url)
url = "https://q-ctrl.com/topics/what-is-quantum-error-correction"
st.write("[Basics of QEC](%s) by Q_CTRL" % url)
url = "https://www.nature.com/articles/s41586-024-08148-8"
st.write("[AlphaQubit](%s)" % url)
url="https://doi.org/10.1103/RevModPhys.87.307"
st.write("[Quantum Error Correction Review](%s)" % url)