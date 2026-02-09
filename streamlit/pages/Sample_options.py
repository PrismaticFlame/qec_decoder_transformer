import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

# TODO move reusable functions to components folder after version has been decided

color_map = {
    "dev_ler": "#1f77b4",
    "best_ler": "#ff7f0e",
    "base": "#2ca02c",
}

color_scale = alt.Scale(
    domain=["dev_ler", "best_ler", "base"],
    range=["#1f77b4", "#ff7f0e",  "#2ca02c"]  # blue, orange, green
)

def createDataAlair(csv, base, container_graph, container_data):
    try:
        # Set up data
        init_df = pd.read_csv(f"data/{csv}")
        base = ([base] * init_df.shape[0])
        df = pd.DataFrame({
            "step": init_df["step"],
            "base": base,
            "dev_ler": init_df["dev_ler"],
            "best_ler": init_df["best_ler"],
        })
        df = df.set_index("step")

        long_df = df.reset_index().melt(
            id_vars="step",
            var_name="metric",
            value_name="LER",
        )

        # Create the graph
        chart = alt.Chart(long_df).mark_line().encode(
            x=alt.X("step:Q", title="Training Step"),
            y=alt.Y("LER:Q", title="LER", scale=alt.Scale(type="log")),
            color=alt.Color("metric:N", title="Metric", scale=color_scale),
            tooltip=["step", "metric", "LER"]
        )

        container_graph.altair_chart(chart, use_container_width=True)

        container_data.dataframe(df)
    except Exception as e:
        if len(e.args) > 1 and e.args[1] == 'No such file or directory':
            st.error(f"{csv} file not found")
        else:
            st.error(e)

def createDataPlotly(csv, base, container_graph, container_data):
    try:
        # # Set up data
        init_df = pd.read_csv(f"data/{csv}")
        base = ([base] * init_df.shape[0])
        df = pd.DataFrame({
            "step": init_df["step"],
            "base": base,
            "dev_ler": init_df["dev_ler"],
            "best_ler": init_df["best_ler"],
        })
        df = df.set_index("step")

        # Create the graph
        fig = px.line(
            df.reset_index(),
            x="step",
            y=["dev_ler", "best_ler", "base"],
            labels={"value": "LER", "variable": "Metric", "step": "Steps"},
            color_discrete_map=color_map
        )

        fig.update_yaxes(type="log")

        container_graph.plotly_chart(fig, use_container_width=True)

        container_data.dataframe(df)
    except Exception as e:
        if len(e.args) > 1 and e.args[1] == 'No such file or directory':
            st.error(f"{csv} file not found")
        else:
            st.error(e)

st.title("Training Sample Options")

if 'name' not in st.session_state:
    st.session_state['basis'] = 'X'

x_csv = 'x_d3_r6_eval.csv'
z_csv = 'z_d3_r6_eval.csv'
x_base = 0.05
z_base = 0.03
csv = x_csv
base = x_base

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
altair_graph = st.container()
with st.expander("Show raw metrics"):
    altair_data = st.container()
st.write("Plotly")
plotly_graph = st.container()
with st.expander("Show raw metrics"):
    plotly_data = st.container()
createDataAlair(csv, base, altair_graph, altair_data)
createDataPlotly(csv, base, plotly_graph, plotly_data)