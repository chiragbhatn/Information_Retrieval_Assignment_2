from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import preprocessing as pp
from core.pipeline import DedupConfig
from core.preprocessing import PreprocessConfig
from core.tfidf import TfidfConfig, cosine_similarity_matrix
from ui import charts, state

st.title("Preprocessing & Feature Engineering")
st.caption("Configure the text pipeline, then inspect the vocabulary, the inverted index and the TF-IDF vectors.")

prep, tfidf_cfg, dedup = state.cfg("prep"), state.cfg("tfidf"), state.cfg("dedup")

# ---------------------------------------------------------------- configuration form
with st.form("prep_config"):
    st.markdown("##### Pipeline configuration")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Cleaning**")
        include_title = st.checkbox("Include headline in text", prep.include_title)
        fix_encoding = st.checkbox("Repair encoding artefacts (mojibake, curly quotes)", prep.fix_encoding)
        strip_markup = st.checkbox("Strip markup, URLs, e-mails", prep.strip_markup)
        lowercase = st.checkbox("Lowercase", prep.lowercase)
        remove_numbers = st.checkbox("Remove numbers", prep.remove_numbers)
        remove_punct = st.checkbox("Remove punctuation / symbols", prep.remove_punctuation)
    with c2:
        st.markdown("**Tokens**")
        remove_stop = st.checkbox("Remove English stop-words", prep.remove_stopwords)
        news_stop = st.checkbox("Also remove news stop-words (said, mr, year...)", prep.news_stopwords)
        custom = st.text_input("Extra stop-words (comma separated)", ", ".join(prep.custom_stopwords))
        normalizer = st.radio("Word normalisation", ["none", "stem", "lemmatize"],
                              index=["none", "stem", "lemmatize"].index(prep.normalizer), horizontal=True,
                              format_func={"none": "None", "stem": "Porter stemming",
                                           "lemmatize": "WordNet lemmatization"}.get)
        min_len = st.slider("Minimum word length", 1, 5, prep.min_token_len)
        ngram = st.select_slider("N-grams", [1, 2, 3], prep.ngram_max,
                                 format_func={1: "unigrams", 2: "uni + bigrams", 3: "up to trigrams"}.get)
    with c3:
        st.markdown("**TF-IDF vocabulary**")
        min_df = st.number_input("Min document frequency", 1, 50, tfidf_cfg.min_df)
        max_df = st.slider("Max document frequency (share of docs)", 0.1, 1.0, tfidf_cfg.max_df_ratio, 0.05)
        max_feat = st.number_input("Max features (0 = no cap)", 0, 30000, tfidf_cfg.max_features, step=500,
                                   help="Keeps the terms with the highest document frequency. "
                                        "Large values use more memory.")
        sublinear = st.checkbox("Sublinear tf (1 + log10 tf)", tfidf_cfg.sublinear_tf)
    with c4:
        st.markdown("**Duplicates**")
        rm_exact = st.checkbox("Remove exact duplicates", dedup.remove_exact)
        rm_near = st.checkbox("Remove near-duplicates", dedup.remove_near)
        near_t = st.slider("Near-duplicate cosine threshold", 0.70, 0.99, dedup.near_threshold, 0.01)
    if st.form_submit_button("Apply configuration", type="primary"):
        state.set_cfg("prep", PreprocessConfig(
            include_title=include_title, fix_encoding=fix_encoding, strip_markup=strip_markup, lowercase=lowercase,
            remove_numbers=remove_numbers, remove_punctuation=remove_punct, remove_stopwords=remove_stop,
            news_stopwords=news_stop, custom_stopwords=tuple(w.strip() for w in custom.split(",") if w.strip()),
            normalizer=normalizer, min_token_len=min_len, ngram_max=ngram))
        state.set_cfg("tfidf", TfidfConfig(int(min_df), float(max_df), int(max_feat), sublinear))
        state.set_cfg("dedup", DedupConfig(rm_exact, rm_near, float(near_t)))
        st.rerun()

c = state.corpus()
fs = state.features()
prep = fs.prep
model = fs.model

if prep.normalizer == "lemmatize":
    st.caption(f"Lemmatizer backend: {pp.LEMMATIZER_BACKEND}")

tab_trace, tab_vocab, tab_tfidf, tab_index, tab_sim, tab_dup = st.tabs(
    ["Step-by-step example", "Vocabulary before / after", "TF-IDF features", "Inverted index explorer",
     "Cosine similarity", "Duplicates"])

# ---------------------------------------------------------------- 1. trace
with tab_trace:
    options = list(range(len(fs.doc_ids)))
    pick = st.selectbox("Document", options, format_func=lambda i: f"{fs.doc_ids[i]} · {fs.titles[i]}")
    raw = pp.document_text(fs.titles[pick], fs.bodies[pick], prep)
    for name, output in pp.preprocess_with_trace(raw, prep):
        with st.container(border=True):
            if isinstance(output, list):
                st.markdown(f"**{name}** - {len(output)} tokens")
                st.code(" | ".join(output[:80]) + (" ..." if len(output) > 80 else ""), language=None,
                        wrap_lines=True)
            else:
                st.markdown(f"**{name}** - {len(output.split())} words")
                st.code(output[:700] + (" ..." if len(output) > 700 else ""), language=None, wrap_lines=True)

# ---------------------------------------------------------------- 2. vocabulary before / after
@st.cache_data(show_spinner=False)
def raw_counts(key: str, _texts: tuple) -> Counter:
    counts = Counter()
    for t in _texts:
        counts.update(pp.raw_tokens(t))
    return counts


with tab_vocab:
    raw_texts = tuple(pp.document_text(t, b, prep) for t, b in zip(fs.titles, fs.bodies))
    before = raw_counts(str((c.fingerprint, prep.include_title, fs.dedup)), raw_texts)
    after = Counter()
    for toks in fs.tokens:
        after.update(toks)
    raw_lengths = np.array([len(pp.raw_tokens(t)) for t in raw_texts])
    proc_lengths = np.array([len(t) for t in fs.tokens])

    def summary(counter: Counter, lengths: np.ndarray) -> dict:
        total = sum(counter.values())
        top20 = sum(v for _, v in counter.most_common(20))
        return {"Total tokens": total, "Vocabulary size (unique terms)": len(counter),
                "Type / token ratio": round(len(counter) / total, 4) if total else 0,
                "Terms used once (hapax)": sum(1 for v in counter.values() if v == 1),
                "Avg tokens per document": round(float(lengths.mean()), 1),
                "Share of tokens in top-20 terms": round(top20 / total, 4) if total else 0}

    sb, sa = summary(before, raw_lengths), summary(after, proc_lengths)
    stats = pd.DataFrame({"statistic": list(sb), "before (raw split)": list(sb.values()),
                          "after preprocessing": list(sa.values())})
    stats["change"] = [f"{(a - b) / b * 100:+.1f}%" if b else "" for b, a in zip(sb.values(), sa.values())]
    charts.table(stats.astype({"before (raw split)": str, "after preprocessing": str}))

    l, r = st.columns(2)
    with l:
        top_b = pd.DataFrame(before.most_common(20), columns=["term", "frequency"])
        fig = px.bar(top_b, x="frequency", y="term", orientation="h", color_discrete_sequence=["#8A8F98"])
        fig.update_yaxes(autorange="reversed")
        charts.show(fig, 520, "Top 20 terms BEFORE preprocessing")
    with r:
        top_a = pd.DataFrame(after.most_common(20), columns=["term", "frequency"])
        fig = px.bar(top_a, x="frequency", y="term", orientation="h", color_discrete_sequence=["#2F6DB5"])
        fig.update_yaxes(autorange="reversed")
        charts.show(fig, 520, "Top 20 terms AFTER preprocessing")

    l, r = st.columns(2)
    with l:
        fig = go.Figure()
        for name, counter, color in (("before", before, "#8A8F98"), ("after", after, "#2F6DB5")):
            freqs = np.array(sorted(counter.values(), reverse=True))
            fig.add_trace(go.Scatter(x=np.arange(1, len(freqs) + 1), y=freqs, mode="lines", name=name,
                                     line=dict(color=color)))
        fig.update_xaxes(type="log", title="rank")
        fig.update_yaxes(type="log", title="frequency")
        charts.show(fig, 380, "Zipf plot: term frequency vs rank (log-log)")
    with r:
        lens = pd.DataFrame({"tokens": np.concatenate([raw_lengths, proc_lengths]),
                             "stage": ["before"] * len(raw_lengths) + ["after"] * len(proc_lengths)})
        fig = px.histogram(lens, x="tokens", color="stage", barmode="overlay", nbins=60,
                           color_discrete_map={"before": "#8A8F98", "after": "#2F6DB5"})
        charts.show(fig, 380, "Tokens per document")

    funnel = pd.DataFrame({"stage": ["Raw vocabulary (whitespace split)", "After preprocessing",
                                     f"After df filter (min_df={fs.tfidf_cfg.min_df}, max_df={fs.tfidf_cfg.max_df_ratio})",
                                     "Feature vocabulary used"],
                           "terms": [len(before), model.full_vocab_size, model.vocab_after_df_filter, len(model.vocab)]})
    fig = px.funnel(funnel, x="terms", y="stage", color_discrete_sequence=["#2F6DB5"])
    charts.show(fig, 300, "Vocabulary reduction")

# ---------------------------------------------------------------- 3. TF-IDF
with tab_tfidf:
    s = model.summary()
    m = st.columns(5)
    m[0].metric("Matrix shape", f"{s['documents']} x {s['feature_vocabulary']}", border=True)
    m[1].metric("Non-zero weights", f"{s['non_zero_weights']:,}", border=True)
    m[2].metric("Sparsity", f"{s['sparsity_pct']}%", border=True)
    m[3].metric("Docs removed as duplicates", len(fs.removed["exact"]) + len(fs.removed["near"]), border=True)
    m[4].metric("Empty documents", s["empty_documents"], border=True)
    st.latex(r"w_{t,d} = \frac{\left(1+\log_{10} tf_{t,d}\right)\cdot \log_{10}\frac{N}{df_t}}"
             r"{\sqrt{\sum_{t'} \left[\left(1+\log_{10} tf_{t',d}\right)\log_{10}\frac{N}{df_{t'}}\right]^2}}"
             if fs.tfidf_cfg.sublinear_tf else
             r"w_{t,d} = \frac{tf_{t,d}\cdot \log_{10}\frac{N}{df_t}}{\sqrt{\sum_{t'} (tf_{t',d}\log_{10}\frac{N}{df_{t'}})^2}}")

    idf_df = pd.DataFrame({"term": model.vocab, "df": model.df, "idf": model.idf})
    l, r = st.columns([3, 2])
    with l:
        fig = px.histogram(idf_df, x="idf", nbins=50, color_discrete_sequence=["#2F6DB5"])
        charts.show(fig, 330, "IDF distribution across the vocabulary")
    with r:
        st.markdown("**Most common terms (lowest IDF)**")
        charts.table(idf_df.sort_values("df", ascending=False).head(8).round(3))
        st.markdown("**Rarest kept terms (highest IDF)**")
        charts.table(idf_df.sort_values(["df", "term"]).head(8).round(3))

    pick = st.selectbox("Inspect a document vector", list(range(len(fs.doc_ids))), key="vec_doc",
                        format_func=lambda i: f"{fs.doc_ids[i]} · {fs.titles[i]}")
    top = pd.DataFrame(model.top_terms_for_doc(pick, 15), columns=["term", "normalized tf-idf"])
    fig = px.bar(top, x="normalized tf-idf", y="term", orientation="h",
                 color_discrete_sequence=[charts.color_map([fs.labels[pick]])[fs.labels[pick]]])
    fig.update_yaxes(autorange="reversed")
    charts.show(fig, 420, f"Top weighted terms in {fs.doc_ids[pick]} ({fs.labels[pick]})")
    st.caption(f"Vector length before normalisation: {model.doc_norms[pick]:.4f} -> after: "
               f"{float(np.linalg.norm(model.matrix[pick])):.4f}")

# ---------------------------------------------------------------- 4. inverted index
with tab_index:
    st.markdown("The inverted index maps each term to its **postings list**: the documents containing it and "
                "the term frequency in each. It is built with plain Python dictionaries.")
    default_term = max(model.vocab, key=lambda t: model.df[model.term_to_col[t]] if model.idf[model.term_to_col[t]] > 0.5 else 0)
    term = st.text_input("Look up a term (after preprocessing, e.g. a stem or lemma)", default_term).strip()
    if prep.lowercase:
        term = term.lower()
    if term not in model.term_to_col:
        processed = pp.preprocess(term, prep)
        if processed and processed[0] in model.term_to_col:
            st.caption(f"'{term}' becomes '{processed[0]}' after preprocessing.")
            term = processed[0]
    if term in model.term_to_col:
        col = model.term_to_col[term]
        m = st.columns(4)
        m[0].metric("Document frequency (df)", int(model.df[col]), border=True)
        m[1].metric("IDF = log10(N/df)", f"{model.idf[col]:.4f}", border=True)
        m[2].metric("Collection frequency", sum(model.index[term].values()), border=True)
        m[3].metric("N (documents)", model.n_docs, border=True)
        charts.table(model.postings_table(term, fs.doc_ids, fs.titles), height=360)
    else:
        st.warning(f"'{term}' is not in the feature vocabulary.")
    st.markdown("**Sample of the index** (first 15 terms alphabetically)")
    sample = [{"term": t, "df": len(model.index[t]),
               "postings (doc_id: tf)": ", ".join(f"{fs.doc_ids[d]}:{tf}" for d, tf in list(model.index[t].items())[:6])
               + (" ..." if len(model.index[t]) > 6 else "")} for t in model.vocab[:15]]
    charts.table(pd.DataFrame(sample))

# ---------------------------------------------------------------- 5. cosine similarity
with tab_sim:
    st.markdown("Cosine similarity of unit-length TF-IDF vectors: "
                r"$\cos(d_1,d_2)=\frac{\vec d_1\cdot\vec d_2}{|\vec d_1||\vec d_2|}$. "
                "Documents from the same category should form bright blocks.")
    per_cat = st.slider("Documents per category", 2, 8, 4)
    rng = np.random.default_rng(0)
    chosen = []
    labels = np.array(fs.labels)
    for cat in sorted(set(fs.labels)):
        idx = np.flatnonzero(labels == cat)
        chosen.extend(rng.choice(idx, size=min(per_cat, len(idx)), replace=False).tolist())
    sims = cosine_similarity_matrix(fs.X[chosen])
    names = [f"{fs.doc_ids[i]}" for i in chosen]
    charts.heatmap(pd.DataFrame(sims, index=names, columns=names), height=640, fmt=".2f", scale="Viridis",
                   title="Pairwise cosine similarity")
    within = [sims[i, j] for i in range(len(chosen)) for j in range(len(chosen))
              if i != j and labels[chosen[i]] == labels[chosen[j]]]
    across = [sims[i, j] for i in range(len(chosen)) for j in range(len(chosen)) if labels[chosen[i]] != labels[chosen[j]]]
    st.caption(f"Mean similarity within the same category: {np.mean(within):.3f} | across categories: {np.mean(across):.3f}")

# ---------------------------------------------------------------- 6. duplicates
with tab_dup:
    m = st.columns(4)
    m[0].metric("Exact duplicate groups (input)", len(fs.exact_groups), border=True)
    m[1].metric("Exact copies removed", len(fs.removed["exact"]), border=True)
    m[2].metric(f"Near-duplicate pairs (cos >= {fs.dedup.near_threshold})", len(fs.near_pairs), border=True)
    m[3].metric("Near-duplicates removed", len(fs.removed["near"]), border=True)
    st.markdown("**Near-duplicate pairs still in the corpus** (detected on TF-IDF cosine similarity)")
    if len(fs.near_pairs):
        charts.table(fs.near_pairs, height=320)
        cross = fs.near_pairs[~fs.near_pairs["same_label"]]
        if len(cross):
            st.warning(f"{len(cross)} near-duplicate pairs carry different category labels (label noise).")
    else:
        st.success("No near-duplicate pairs above the threshold.")
    if fs.removed["exact"] or fs.removed["near"]:
        with st.expander("Removed document IDs"):
            st.write({"exact": fs.removed["exact"], "near": fs.removed["near"]})
