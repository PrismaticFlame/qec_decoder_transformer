import streamlit as st
from components.graphs import createDataPlotly
from components.Data import Data
from components.comparisons import getOverview, getStructure, getSummary

# Data file initialization
x_data = Data('x_d3_r6_eval.csv', "X", 3, 6, 500)
z_data = Data('z_d3_r6_eval.csv', "Z", 3, 6, 50000)

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

# if (st.session_state['version'] != 'All'):
st.header(f"{st.session_state['version']}: Short Summary")
st.write(getSummary(st.session_state['version']))

basis_container = st.container()


st.header("Results")
# X and Z buttons
xBtn, zBtn = st.columns(2, vertical_alignment="bottom")

xBtn.button("X data", width='stretch', on_click = btnClickXZ, disabled=isX())
zBtn.button("Z data", width='stretch', on_click = btnClickXZ, disabled= not isX())


graph = st.container()
with st.expander("Show raw metrics"):
    dataContainer = st.container()
createDataPlotly(data, graph, dataContainer, title=f"{data.basis} basis, Distance {data.d}", subtitle=f"{data.shots} shots, Rounds {data.r}")

st.header("Model details")
st.info("Double click a cell to expand its contents")
overviewTab, structureTab = st.tabs(["Overview", "Data Input and Structure"])

#Overview
overviewTab.dataframe(getOverview(getVersion()))

#Generation
generationDF, formatDF, inputDF = (getStructure(getVersion()))
structureTab.subheader("Data Generation")
structureTab.dataframe(generationDF)
structureTab.subheader("Data Format")
structureTab.dataframe(formatDF)
structureTab.subheader("Input Representation to the Model")
structureTab.dataframe(inputDF)