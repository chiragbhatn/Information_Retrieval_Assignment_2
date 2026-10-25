import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core import preprocessing as pp
from core.recommender import recommend, shared_terms
from ui import charts, state

st.title("Document Recommendation")
st.caption("Find the Top-K articles most related to a chosen article or to free text.")

fs = state.features()
model = fs.model
meta = state.corpus().meta
cmap = charts.color_map(fs.labels)


def summary_of(doc_id: str) -> str:
    """The article's summary metadata, or '' when the corpus has none (or it is blank)."""
    if "summary" not in meta.columns:
        return ""
    value = meta.at[doc_id, "summary"]
    return "" if pd.isna(value) else str(value).strip()


c1, c2, c3 = st.columns([2, 1, 1])
mode = c1.radio("Query", ["Pick an article", "Type or paste text"], horizontal=True)
k = c2.slider("Top-K", 1, 20, 5)
method = c3.selectbox("Method", ["Content-based (cosine)", "Hybrid (content + cluster)"])
weight = 0.0
if method.startswith("Hybrid"):
    weight = st.slider("Cluster weight w", 0.05, 0.9, 0.2, 0.05,
                       help="score = (1 - w) * cosine + w * [same K-Means cluster as the query]")
hide_dups = st.checkbox("Hide duplicates of the query (cosine >= 0.98)", True)

km = state.clustering().km
clusters = km.labels
query_idx = None

if mode == "Pick an article":
    f1, f2 = st.columns([1, 3])
    cat = f1.selectbox("Category filter", ["All"] + sorted(set(fs.labels)))
    options = [i for i in range(len(fs.doc_ids)) if cat == "All" or fs.labels[i] == cat]
    query_idx = f2.selectbox("Article", options, format_func=lambda i: f"{fs.doc_ids[i]} · {fs.titles[i]}")
    qvec = fs.X[query_idx]
    q_cluster = int(clusters[query_idx])
    with st.container(border=True):
        st.markdown(f"**{fs.titles[query_idx]}**  \n`{fs.doc_ids[query_idx]}` · category **{fs.labels[query_idx]}** "
                    f"· cluster **{q_cluster}**")
        preview = summary_of(fs.doc_ids[query_idx]) or fs.bodies[query_idx]
        st.caption(preview[:500].lstrip("- ").replace(" - ", " · "))
        st.caption("Top terms: " + ", ".join(t for t, _ in model.top_terms_for_doc(query_idx, 10)))
else:
    text = st.text_area("Query text", "Oil prices rise as OPEC cuts production and markets worry about economic growth",
                        height=100)
    qvec = model.transform([pp.preprocess(text, fs.prep)])[0]
    if not qvec.any():
        st.warning("None of the query words are in the vocabulary. Try different words.")
        st.stop()
    q_cluster = int(((km.centroids - qvec) ** 2).sum(axis=1).argmin())
    st.caption(f"Query terms after preprocessing: {', '.join(pp.preprocess(text, fs.prep))} · nearest cluster: {q_cluster}")

exclude = set()
if query_idx is not None:
    exclude.add(query_idx)
    if hide_dups:
        exclude |= {int(i) for i in np.flatnonzero(fs.X @ qvec >= 0.98)}

recs = recommend(fs.X, qvec, k, exclude, clusters if weight > 0 else None, q_cluster, weight)

rows = []
for rank, (i, score, cos) in enumerate(recs, 1):
    m = meta.loc[fs.doc_ids[i]]
    rows.append({"rank": rank, "doc_id": fs.doc_ids[i], "title": fs.titles[i], "category": fs.labels[i],
                 "cluster": int(clusters[i]), "score": round(score, 4), "cosine": round(cos, 4),
                 "shared terms": ", ".join(shared_terms(qvec, fs.X[i], model.vocab, 5)),
                 "source": m.get("source", ""),
                 "same category": (fs.labels[i] == fs.labels[query_idx]) if query_idx is not None else None})
rec_df = pd.DataFrame(rows)

st.subheader(f"Top-{k} recommendations")
if query_idx is not None and len(rec_df):
    hit = rec_df["same category"].mean()
    st.caption(f"{hit:.0%} of these recommendations share the query's category (precision@{k} for this query).")
charts.table(rec_df.drop(columns=[c for c in ["same category"] if query_idx is None]),
             column_config={"score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0, format="%.4f")})

if len(rec_df):
    fig = px.bar(rec_df, x="score", y=rec_df["rank"].astype(str) + ". " + rec_df["title"], orientation="h",
                 color="category", color_discrete_map=cmap, hover_data=["doc_id", "cosine", "shared terms"])
    fig.update_yaxes(autorange="reversed", title=None)
    charts.show(fig, 120 + 36 * len(rec_df), "Recommendation scores")

    for r in rows:
        with st.expander(f"{r['rank']}. {r['title']}  ·  {r['category']}  ·  score {r['score']:.3f}"):
            m = meta.loc[r["doc_id"]]
            st.markdown(f"`{r['doc_id']}` · cluster {r['cluster']} · shared terms: *{r['shared terms']}*")
            st.write(summary_of(r["doc_id"]) or fs.bodies[fs.doc_ids.index(r["doc_id"])][:600])

st.divider()
q = state.recommendation_quality(k, weight)
st.metric(f"Corpus-wide precision@{k} ({'hybrid' if weight else 'content-based'})", f"{q['precision_at_k']:.3f}",
          help="Every article is used once as the query; a recommendation counts as relevant if it has the same category.")

with st.expander("How the recommender works"):
    st.markdown(
        "1. Every article is a **unit-length TF-IDF vector** (built by hand on the Preprocessing page).\n"
        "2. The query (an article, or free text run through the *same* preprocessing pipeline and the corpus IDF) "
        "is compared with every article using **cosine similarity**, which for unit vectors is just the dot product.\n"
        "3. The **Top-K** highest scores are returned, skipping the query itself and, optionally, its duplicates.\n"
        "4. **Hybrid mode** adds a bonus for articles in the query's K-Means cluster: "
        "`score = (1 - w) * cosine + w * same_cluster`, which pulls in thematically related articles that share fewer exact words.\n"
        "5. **Shared terms** are the terms with the largest product of query and document weights, i.e. the words "
        "that contributed most to the score.")
with st.expander("Limitations on this dataset"):
    st.markdown(
        "- **Vocabulary mismatch:** cosine only matches identical terms, so synonyms ('film' vs 'movie') and "
        "paraphrases are missed. Lemmatization helps a little; embeddings would help more.\n"
        "- **Duplicates crowd the list:** without de-duplication the top result is often a copy of the query.\n"
        "- **No user signal:** there is no reading history, so recommendations are not personalised and cannot "
        "learn what analysts found useful (collaborative filtering would need user-article interaction data).\n"
        "- **Topic drift by shared names:** articles that mention the same person or company can score high "
        "even when the stories are about different things.\n"
        "- **Static corpus:** IDF is fixed at fit time; new articles with new terms need the index rebuilt.\n"
        "- **Evaluation proxy:** 'same category' is a coarse stand-in for relevance; two business stories may be unrelated.")
