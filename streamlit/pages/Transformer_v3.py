import streamlit as st
from pages.Sample_options import createDataAlair
st.title("Transformer v3")

x_csv = 'x_d3_r6_eval.csv'
z_csv = 'z_d3_r6_eval.csv'
x_base = 0.05
z_base = 0.03
csv = x_csv
base = x_base

st.header("What is different about this version?")
st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
st.space("small")

basis_container = st.container()
left, right = st.columns(2, vertical_alignment="bottom")

if left.button("X data", use_container_width=True):
    st.session_state['basis'] = 'X'
    csv = x_csv
    base = x_base
if right.button("Z data", use_container_width=True):
    st.session_state['basis'] = 'Z'
    csv = z_csv
    base = z_base

basis_container.write(f"Current basis: {st.session_state['basis']}")


st.write("Altair")
graph = st.container()
with st.expander("Show raw metrics"):
    data = st.container()
createDataAlair(csv, base, graph, data)