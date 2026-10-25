"""NewsLens - BBC News Text Mining & Document Intelligence System.

Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="NewsLens | Text Intelligence", page_icon=":material/newspaper:",
                   layout="wide")

from ui import state  # noqa: E402

state.init_state()

pages = {
    "Overview": [
        st.Page("views/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
    ],
    "Workflow": [
        st.Page("views/ingestion.py", title="Data Ingestion", icon=":material/upload_file:"),
        st.Page("views/preprocessing.py", title="Preprocessing & TF-IDF", icon=":material/tune:"),
        st.Page("views/mining_hub.py", title="Text Mining Hub", icon=":material/hub:"),
        st.Page("views/recommendations.py", title="Recommendations", icon=":material/recommend:"),
        st.Page("views/performance.py", title="Performance Analytics", icon=":material/monitoring:"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    c = state.corpus()
    st.caption("ACTIVE CORPUS")
    st.markdown(f"**{c.name}** · {len(c):,} documents")
    prep, km = state.cfg("prep"), state.cfg("km")
    st.caption(f"Preprocessing: {prep.short_label()}")
    st.caption(f"K-Means: k={km.k}, {km.init}")
    st.divider()
    st.caption("Information Retrieval · Assignment 2 · Group G07")

nav.run()
