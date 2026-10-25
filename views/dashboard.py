import pandas as pd
import plotly.express as px
import streamlit as st

from core.data_loader import corpus_stats
from core.duplicates import exact_duplicate_groups
from ui import charts, state

st.title("NewsLens: BBC News Intelligence System")
st.caption("Text mining, classification, clustering and recommendation for a news-monitoring desk")

with st.container(border=True):
    st.markdown(
        "**Business context.** An investment-research firm's news desk receives hundreds of articles a day. "
        "Analysts need each story **routed to the right desk** (business, politics, tech, sport, entertainment), "
        "**emerging themes grouped** without manual tagging, **duplicate wire copies removed**, and "
        "**related coverage surfaced** when they read a story. NewsLens does this on the BBC News corpus "
        "(2,224 articles, 2004-2005) using hand-built TF-IDF, Naive Bayes, K-Means and cosine-similarity "
        "recommendation."
    )

c = state.corpus()
stats = corpus_stats(c)
dup_groups = exact_duplicate_groups(c.texts.tolist())

cols = st.columns(6)
cols[0].metric("Documents", f"{stats['documents']:,}", border=True)
cols[1].metric("Categories", stats["categories"], border=True)
cols[2].metric("Raw text", f"{stats['raw_text_mb']} MB", border=True)
cols[3].metric("Avg words / doc", stats["avg_words"], border=True)
cols[4].metric("Total words", f"{stats['total_words']:,}", border=True)
cols[5].metric("Duplicate groups", len(dup_groups), help="Groups of articles with identical text", border=True)

left, right = st.columns(2)
with left:
    counts = c.meta[c.label_col].value_counts().rename_axis("category").reset_index(name="documents")
    fig = px.bar(counts, x="category", y="documents", color="category", text="documents",
                 color_discrete_map=charts.color_map(counts["category"]))
    fig.update_layout(showlegend=False)
    charts.show(fig, 340, "Documents per category")
with right:
    lengths = pd.DataFrame({"category": c.labels, "words": c.texts.str.split().str.len().values})
    fig = px.box(lengths, x="category", y="words", color="category", points=False,
                 color_discrete_map=charts.color_map(lengths["category"]))
    fig.update_layout(showlegend=False)
    charts.show(fig, 340, "Document length by category (words)")

st.subheader("Results snapshot")
if not st.session_state.get("pipeline_ran"):
    st.info("Run the pipeline to fill in the headline results (takes a few seconds).")
    if st.button("Run full pipeline", type="primary", icon=":material/play_arrow:"):
        with st.status("Running pipeline...", expanded=False):
            state.classification()
            state.clustering()
        st.rerun()
else:
    # Cached on the current corpus and settings, so these always match what the other pages show.
    nb, kr = state.classification().scores, state.clustering()
    rec = state.recommendation_quality(5)
    m = st.columns(5)
    m[0].metric("NB accuracy", f"{nb['accuracy']:.3f}", border=True)
    m[1].metric("NB macro F1", f"{nb['macro_f1']:.3f}", border=True)
    m[2].metric("K-Means purity", f"{kr.purity:.3f}", border=True)
    m[3].metric("K-Means Rand Index", f"{kr.rand['rand_index']:.3f}", border=True)
    m[4].metric("Recommender P@5", f"{rec['precision_at_k']:.3f}", border=True,
                help="Share of top-5 recommendations from the same category as the query")

st.subheader("Workflow")
steps = [
    ("views/ingestion.py", "1. Data Ingestion", "Load the bundled corpus or upload CSV / JSON / ZIP; corpus statistics."),
    ("views/preprocessing.py", "2. Preprocessing & TF-IDF", "Configure cleaning, stop-words, stemming / lemmatization, n-grams; inverted index and TF-IDF."),
    ("views/mining_hub.py", "3. Text Mining Hub", "Corpus analytics, Naive Bayes classification, K-Means clustering, Purity and Rand Index."),
    ("views/recommendations.py", "4. Recommendations", "Top-K related documents with similarity scores and metadata."),
    ("views/performance.py", "5. Performance Analytics", "Metrics comparison and experiments (preprocessing, duplicates, k, init)."),
]
grid = st.columns(len(steps))
for col, (page, label, desc) in zip(grid, steps):
    with col.container(border=True, height=170):
        st.page_link(page, label=label)
        st.caption(desc)
