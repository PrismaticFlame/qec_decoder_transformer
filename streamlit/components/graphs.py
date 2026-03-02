import streamlit as st
import pandas as pd
import plotly.express as px

color_map = {
    "dev_ler": "#1f77b4",
    "best_ler": "#ff7f0e",
    "base": "#2ca02c",
}

def createDataPlotly(csv, base, container_graph, container_data, title = "", subtitle = ""):
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
            color_discrete_map=color_map,
            title = title,
            subtitle= subtitle,
        )

        fig.update_yaxes(type="log")

        container_graph.plotly_chart(fig, use_container_width=True)

        container_data.dataframe(df)
    except Exception as e:
        if len(e.args) > 1 and e.args[1] == 'No such file or directory':
            st.error(f"{csv} file not found")
        else:
            st.error(e)