import streamlit as st
from components.graphs import createDataPlotly
from components.Data import Data

# Data file initialization
x_data = Data('x_d3_r6_eval.csv', 0.05, "X", 3, 6, 500)
z_data = Data('z_d3_r6_eval.csv', 0.03, "Z", 3, 6, 50000)

dataDict = {
    'All': {'X': x_data, 'Z': z_data}, 
    'v3': {'X': x_data, 'Z': z_data},
    'v5': {'X': x_data, 'Z': z_data},
    'v6': {'X': x_data, 'Z': z_data},
}

# Set initial files
if 'basis' not in st.session_state:
    st.session_state['basis'] = 'X'

if 'version' not in st.session_state:
    st.session_state['version'] = 'All'

# Sets data based on current version and basis
data = dataDict[st.session_state['version']][st.session_state['basis']]

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

allBtn.button("All", width='stretch', on_click = btnClickAll, disabled= getVersion() == "All")
v3Btn.button("v3", width='stretch', on_click = btnClickV3, disabled= getVersion() == "v3")
v5Btn.button("v5", width='stretch', on_click = btnClickV5, disabled= getVersion() == "v5")
v6Btn.button("v6", width='stretch', on_click = btnClickV6, disabled= getVersion() == "v6")

if (st.session_state['version'] != 'All'):
    st.header(f"{st.session_state['version']}: Short Summary")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
    st.space("small")

basis_container = st.container()

# X and Z buttons
xBtn, zBtn = st.columns(2, vertical_alignment="bottom")

xBtn.button("X data", width='stretch', on_click = btnClickXZ, disabled=isX())
zBtn.button("Z data", width='stretch', on_click = btnClickXZ, disabled= not isX())

basis_container.write(f"Current basis: {st.session_state['basis']}")


st.header("Results")
graph = st.container()
with st.expander("Show raw metrics"):
    dataContainer = st.container()
createDataPlotly(data, graph, dataContainer, title=f"{data.basis} basis, Distance {data.d}", subtitle=f"{data.shots} shots, Rounds {data.r}")

st.header("Model details")
st.write("Lorem ipsum dolor sit amet consectetur adipiscing elit. Consectetur adipiscing elit quisque faucibus ex sapien vitae. Ex sapien vitae pellentesque sem placerat in id. Placerat in id cursus mi pretium tellus duis. Pretium tellus duis convallis tempus leo eu aenean.")