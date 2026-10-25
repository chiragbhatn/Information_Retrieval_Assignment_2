"""Plotly helpers so every page uses the same colours and layout."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CATEGORY_COLORS = {
    "business": "#2F6DB5", "entertainment": "#D9822B", "politics": "#C23B3B",
    "sport": "#2E9E6B", "tech": "#7B5CC4",
}
PALETTE = px.colors.qualitative.Safe + px.colors.qualitative.Vivid


def color_map(values) -> dict:
    out, i = {}, 0
    for v in sorted({str(x) for x in values}):
        if v in CATEGORY_COLORS:
            out[v] = CATEGORY_COLORS[v]
        else:
            out[v] = PALETTE[i % len(PALETTE)]
            i += 1
    return out


def style(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      title=title, legend_title_text="", template="plotly_white")
    return fig


def show(fig: go.Figure, height: int = 380, title: str | None = None) -> None:
    st.plotly_chart(style(fig, height, title), width="stretch")


def heatmap(df: pd.DataFrame, title: str | None = None, height: int = 420, fmt: str = "d",
            scale: str = "Blues") -> None:
    text_auto = True if fmt == "d" else (fmt or False)
    fig = px.imshow(df.values, x=[str(c) for c in df.columns], y=[str(i) for i in df.index],
                    text_auto=text_auto, color_continuous_scale=scale, aspect="auto")
    fig.update_xaxes(side="top")
    style(fig, height, title)
    fig.update_layout(margin=dict(t=95 if title else 40))
    st.plotly_chart(fig, width="stretch")


def table(df: pd.DataFrame, height: int | None = None, **kwargs) -> None:
    if height:
        kwargs["height"] = height
    st.dataframe(df, width="stretch", hide_index=True, **kwargs)
