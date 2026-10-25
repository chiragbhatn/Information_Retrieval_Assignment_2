import pandas as pd
import plotly.express as px
import streamlit as st

from core import data_loader as dl
from core.duplicates import exact_duplicate_groups
from ui import charts, state

st.title("Data Ingestion")
st.caption("Load a document collection. Document text and metadata are stored separately and joined on doc_id.")

source = st.radio("Data source", ["Bundled BBC News dataset", "Upload CSV / JSON", "Upload ZIP of .txt files"],
                  horizontal=True)

if source == "Bundled BBC News dataset":
    base = state.bundled_corpus()
    with st.form("bundled"):
        st.markdown("**BBC News** - 2,224 articles in 5 categories, from Hugging Face "
                    "[`gopalkalpande/bbc-news-summary`](https://huggingface.co/datasets/gopalkalpande/bbc-news-summary) "
                    "(original corpus: Greene & Cunningham, 2006). Files: `data/bbc_documents.csv` (doc_id, text) and "
                    "`data/bbc_metadata.csv` (doc_id, title, category, source, summary).")
        c1, c2 = st.columns(2)
        size = c1.slider("Corpus size (stratified sample)", 100, len(base), len(state.corpus()) if
                         state.corpus().name.startswith("BBC") else len(base), step=1,
                         help="Use a smaller sample for a quick run; the full corpus for final results.")
        seed = c2.number_input("Sampling seed", 0, 9999, 42)
        if st.form_submit_button("Load corpus", type="primary"):
            state.set_corpus(base.stratified_sample(size, int(seed)), "Bundled BBC News dataset")
            st.rerun()

elif source == "Upload CSV / JSON":
    up = st.file_uploader("CSV, JSON or JSON-lines file", type=["csv", "json", "jsonl"])
    if up is not None:
        try:
            df = dl.read_table(up, up.name)
        except Exception as exc:  # show a readable error instead of a stack trace
            st.error(f"Could not read the file: {exc}")
            st.stop()
        st.dataframe(df.head(), width="stretch")
        with st.form("mapping"):
            cols = list(df.columns)
            guess = lambda names, default=0: next((cols.index(c) for c in cols if c.lower() in names), default)  # noqa: E731
            c1, c2, c3, c4 = st.columns(4)
            id_col = c1.selectbox("Document ID column", ["<auto>"] + cols,
                                  index=guess({"doc_id", "id", "docid"}, -1) + 1)
            text_col = c2.selectbox("Text column", cols, index=guess({"text", "content", "body", "article"}))
            label_col = c3.selectbox("Category label column", cols, index=guess({"category", "label", "topic", "class"}))
            title_col = c4.selectbox("Title column (optional)", ["<none>"] + cols,
                                     index=guess({"title", "headline", "subject"}, -1) + 1)
            meta_cols = st.multiselect("Other metadata columns", [c for c in cols if c not in (text_col,)],
                                       default=[c for c in cols if c.lower() in {"date", "source", "author", "url"}])
            if st.form_submit_button("Load corpus", type="primary"):
                corpus = dl.from_table(df, id_col, text_col, label_col,
                                       None if title_col == "<none>" else title_col, meta_cols, up.name)
                state.set_corpus(corpus, f"Uploaded file: {up.name}")
                st.rerun()

else:
    st.caption("Each sub-folder name is used as the category, e.g. `business/001.txt`. "
               "The first line of each file is used as its title.")
    up = st.file_uploader("ZIP file", type=["zip"])
    if up is not None and st.button("Load corpus", type="primary"):
        try:
            state.set_corpus(dl.from_zip(up, up.name), f"Uploaded folder: {up.name}")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not read the zip: {exc}")

# ---------------------------------------------------------------- current corpus
c = state.corpus()
st.divider()
st.subheader(f"Current corpus: {c.name}")
st.caption(st.session_state.get("corpus_source", ""))

stats = dl.corpus_stats(c)
m = st.columns(7)
for col, (label, key) in zip(m, [("Documents", "documents"), ("Categories", "categories"),
                                 ("Raw text (MB)", "raw_text_mb"), ("Avg words", "avg_words"),
                                 ("Median words", "median_words"), ("Shortest", "min_words"),
                                 ("Longest", "max_words")]):
    col.metric(label, f"{stats[key]:,}" if isinstance(stats[key], int) else stats[key], border=True)

tab_docs, tab_meta, tab_quality, tab_dups = st.tabs(
    ["Document text store", "Metadata table", "Data quality / noise", "Duplicates"])

with tab_docs:
    docs_view = pd.DataFrame({"doc_id": c.texts.index, "words": c.texts.str.split().str.len().values,
                              "text": c.texts.str.slice(0, 300).values})
    charts.table(docs_view, height=380)

with tab_meta:
    meta_view = c.meta.reset_index()
    if "summary" in meta_view.columns:
        meta_view["summary"] = meta_view["summary"].astype(str).str.slice(0, 160)
    charts.table(meta_view, height=380)
    fig = px.histogram(pd.DataFrame({"words": c.texts.str.split().str.len().values, "category": c.labels}),
                       x="words", color="category", nbins=60, barmode="overlay",
                       color_discrete_map=charts.color_map(c.labels))
    charts.show(fig, 320, "Document length distribution")

with tab_quality:
    texts = c.texts
    checks = pd.DataFrame([
        ("Encoding artefacts (mojibake / replacement chars)",
         int(texts.str.contains("�|Â|â€", regex=True).sum()), int(texts.str.count("�|Â|â€").sum())),
        ("Non-ASCII symbols (e.g. £)", int(texts.str.contains(r"[^\x00-\x7f]", regex=True).sum()),
         int(texts.str.count(r"[^\x00-\x7f]").sum())),
        ("Paragraph breaks written as '..'", int(texts.str.contains(r"\.\.", regex=True).sum()),
         int(texts.str.count(r"\.{2,}").sum())),
        ("HTML / markup tags", int(texts.str.contains(r"<[^>]+>", regex=True).sum()),
         int(texts.str.count(r"<[^>]+>").sum())),
        ("URLs", int(texts.str.contains(r"https?://|www\.", regex=True).sum()),
         int(texts.str.count(r"https?://\S+|www\.\S+").sum())),
        ("ALL-CAPS words (inconsistent casing)", int(texts.str.contains(r"\b[A-Z]{4,}\b", regex=True).sum()),
         int(texts.str.count(r"\b[A-Z]{4,}\b").sum())),
        ("Numbers / currency amounts", int(texts.str.contains(r"\d", regex=True).sum()),
         int(texts.str.count(r"\d+(?:[.,]\d+)*").sum())),
        ("Empty / missing titles", int((c.meta.get(c.title_col, pd.Series("", index=c.meta.index))
                                        .fillna("").astype(str).str.strip() == "").sum()), None),
    ], columns=["check", "documents affected", "occurrences"])
    charts.table(checks)
    sample = texts[texts.str.contains(r"\.\.", regex=True)].head(1)
    if len(sample):
        snippet = sample.iloc[0]
        pos = snippet.find("..")
        st.markdown("**Example of raw noise** (`" + sample.index[0] + "`):")
        st.code(snippet[max(0, pos - 120): pos + 120], language=None, wrap_lines=True)

with tab_dups:
    groups = exact_duplicate_groups(c.texts.tolist())
    st.markdown(f"**{len(groups)} groups** of exact duplicates "
                f"({sum(len(g) - 1 for g in groups)} redundant copies). "
                "Near-duplicates (high cosine similarity) are detected after TF-IDF on the Preprocessing page.")
    if groups:
        ids, titles, labels = c.doc_ids, c.titles, c.labels
        rows = [{"group": n + 1, "copies": len(g), "doc_ids": ", ".join(ids[i] for i in g),
                 "title": titles[g[0]], "categories": ", ".join(sorted({labels[i] for i in g}))}
                for n, g in enumerate(groups)]
        charts.table(pd.DataFrame(rows), height=360)
