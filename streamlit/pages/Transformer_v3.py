import streamlit as st
from components.graphs import createDataPlotly

x_csv = 'x_d3_r6_eval.csv'
z_csv = 'z_d3_r6_eval.csv'
x_base = 0.05
z_base = 0.03
csv = x_csv
base = x_base

if 'basis' not in st.session_state:
    st.session_state['basis'] = 'X'

def btnClick():
    if st.session_state['basis'] == 'X':
        st.session_state['basis'] = 'Z'
    else:
        st.session_state['basis'] = 'X'

def isX():
    return st.session_state['basis'] == 'X'

st.title("Transformer v3")

st.header("What is different about this version?")
st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
st.space("small")

st.header("Resuslts")

basis_container = st.container()
left, right = st.columns(2, vertical_alignment="bottom")

if left.button("X data", use_container_width=True, on_click = btnClick, disabled=isX()):
    csv = x_csv
    base = x_base
if right.button("Z data", use_container_width=True, on_click = btnClick, disabled= not isX()):
    csv = z_csv
    base = z_base

basis_container.write(f"Current basis: {st.session_state['basis']}")


graph = st.container()
with st.expander("Show raw metrics"):
    data = st.container()
createDataPlotly(csv, base, graph, data, "Test title", "This is a subtitle")

st.header("Summary of results")
st.write("Lorem ipsum dolor sit amet consectetur adipiscing elit. Consectetur adipiscing elit quisque faucibus ex sapien vitae. Ex sapien vitae pellentesque sem placerat in id. Placerat in id cursus mi pretium tellus duis. Pretium tellus duis convallis tempus leo eu aenean.")