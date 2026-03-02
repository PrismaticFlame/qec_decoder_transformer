import streamlit as st
from components.graphs import createDataPlotly

# Temp files
x_csv = 'x_d3_r6_eval.csv'
z_csv = 'z_d3_r6_eval.csv'
x_base = 0.05
z_base = 0.03

# Set initial files
if 'basis' not in st.session_state:
    st.session_state['basis'] = 'X'

if 'version' not in st.session_state:
    st.session_state['version'] = 'All'

# CSV files
csvs = {
    'All': {'X': x_csv, 'Z': z_csv}, 
    'v3': {'X': x_csv, 'Z': z_csv},
    'v5': {'X': x_csv, 'Z': z_csv},
    'v6': {'X': x_csv, 'Z': z_csv},
}
bases = {
    'All': {'X': x_base, 'Z': z_base}, 
    'v3': {'X': x_base, 'Z': z_base},
    'v5': {'X': x_base, 'Z': z_base},
    'v6': {'X': x_base, 'Z': z_base},
}
csv = csvs[st.session_state['version']][st.session_state['basis']]
base = bases[st.session_state['version']][st.session_state['basis']]

# Functions for buttons
def btnClickAll():
    st.session_state['version'] = 'All'

def btnClickV3():
    st.session_state['version'] = 'v3' # TODO v3 vs V3

def btnClickV5():
    st.session_state['version'] = 'v5'

def btnClickV6():
    st.session_state['version'] = 'v6'

def btnClickXZ():
    if st.session_state['basis'] == 'X':
        st.session_state['basis'] = 'Z'
    else:
        st.session_state['basis'] = 'X'

def isX():
    return st.session_state['basis'] == 'X'

def getVersion():
    return st.session_state['version']


st.title("Transformers")

# Buttons to select version
allBtn, v3Btn, v5Btn, v6Btn = st.columns(4, vertical_alignment="bottom")

allBtn.button("All", use_container_width=True, on_click = btnClickAll, disabled= getVersion() == "All")
v3Btn.button("v3", use_container_width=True, on_click = btnClickV3, disabled= getVersion() == "v3")
v5Btn.button("v5", use_container_width=True, on_click = btnClickV5, disabled= getVersion() == "v5")
v6Btn.button("v6", use_container_width=True, on_click = btnClickV6, disabled= getVersion() == "v6")

if (st.session_state['version'] != 'All'):
    st.header(f"{st.session_state['version']}: Short Summary")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
    st.space("small")

basis_container = st.container()

# X and Z buttons
xBtn, zBtn = st.columns(2, vertical_alignment="bottom")

xBtn.button("X data", use_container_width=True, on_click = btnClickXZ, disabled=isX())
zBtn.button("Z data", use_container_width=True, on_click = btnClickXZ, disabled= not isX())

basis_container.write(f"Current basis: {st.session_state['basis']}")


st.header("Results")
graph = st.container()
with st.expander("Show raw metrics"):
    data = st.container()
createDataPlotly(csv, base, graph, data, title=f"{st.session_state['basis']} basis, #### shots", subtitle="Distance 3, Rounds 6")

st.header("Model details")
st.write("Lorem ipsum dolor sit amet consectetur adipiscing elit. Consectetur adipiscing elit quisque faucibus ex sapien vitae. Ex sapien vitae pellentesque sem placerat in id. Placerat in id cursus mi pretium tellus duis. Pretium tellus duis convallis tempus leo eu aenean.")