import streamlit as st
from components.graphs import createDataPlotly
from components.Data import Data

x_csv = 'x_d3_r6_eval.csv'
z_csv = 'z_d3_r6_eval.csv'
x_base = 0.05
z_base = 0.03
csv = x_csv
base = x_base

x_data = Data('x_d3_r6_eval.csv', 0.05, "X", 3, 6, 500)
z_data = Data('z_d3_r6_eval.csv', 0.03, "Z", 3, 6, 50000)

if 'basis' not in st.session_state:
    st.session_state['basis'] = 'X'

dataDict = {
    'X': x_data,
    'Z': z_data,
}

data = dataDict[st.session_state['basis']]

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

left.button("X data", width='stretch', on_click = btnClick, disabled=isX())
right.button("Z data", width='stretch', on_click = btnClick, disabled= not isX())

basis_container.write(f"Current basis: {st.session_state['basis']}")


graph = st.container()
with st.expander("Show raw metrics"):
    dataContainer = st.container()
createDataPlotly(data, graph, dataContainer, "Test title", "This is a subtitle")

st.header("Summary of results")
st.write("Lorem ipsum dolor sit amet consectetur adipiscing elit. Consectetur adipiscing elit quisque faucibus ex sapien vitae. Ex sapien vitae pellentesque sem placerat in id. Placerat in id cursus mi pretium tellus duis. Pretium tellus duis convallis tempus leo eu aenean.")