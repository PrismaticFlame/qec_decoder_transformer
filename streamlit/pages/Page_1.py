import pandas as pd
import streamlit as st
from numpy.random import default_rng as rng

st.title("Example graph")

df = pd.DataFrame(
    {
        "Index": list(range(20)) * 3,
        "Value": rng(0).standard_normal(60),
        "Section": ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
    }
)

st.line_chart(df, x="Index", y="Value", color="Section")