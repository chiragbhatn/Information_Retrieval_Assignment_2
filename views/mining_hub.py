from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.metrics import purity, rand_index
from core.pipeline import KMConfig, NBConfig
from core.tfidf import cosine_similarity_matrix
from ui import charts, state

st.title("Text Mining & Corpus Analytics Hub")
st.caption("Patterns in the collection, supervised classification (Naive Bayes), unsupervised clustering "
           "(K-Means) and their evaluation. All four algorithms are implemented from scratch.")

fs = state.features()
labels = np.array(fs.labels)
classes = sorted(set(fs.labels))
cmap = charts.color_map(classes)

tab_corpus, tab_nb, tab_km, tab_eval = st.tabs(
    ["Corpus analytics", "Classification: Naive Bayes", "Clustering: K-Means", "Evaluation & comparison"])

# ================================================================ corpus analytics
with tab_corpus:
    st.markdown("##### Most characteristic terms per category (mean TF-IDF weight)")
    cat_means = {c: fs.X[labels == c].mean(axis=0) for c in classes}
    grid = st.columns(len(classes))
    for col, c in zip(grid, classes):
        idx = np.argsort(-cat_means[c])[:10]
        df = pd.DataFrame({"term": [fs.model.vocab[i] for i in idx], "weight": cat_means[c][idx]})
        fig = px.bar(df, x="weight", y="term", orientation="h", color_discrete_sequence=[cmap[c]])
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_xaxes(title=None, showticklabels=False)
        with col:
            charts.show(fig, 330, c)

    l, r = st.columns(2)
    with l:
        centroids = np.vstack([cat_means[c] for c in classes])
        sims = cosine_similarity_matrix(centroids)
        charts.heatmap(pd.DataFrame(sims, index=classes, columns=classes), fmt=".2f", scale="Blues",
                       title="Similarity between category centroids", height=400)
        st.caption("Off-diagonal values show which desks share vocabulary (e.g. business and politics).")
    with r:
        st.markdown("**Term usage by category**")
        term = st.selectbox("Term", fs.model.vocab,
                            index=fs.model.vocab.index("economy") if "economy" in fs.model.vocab else 0)
        col = fs.model.term_to_col[term]
        df_by_cat = Counter(fs.labels[d] for d in fs.model.index[term])
        size_by_cat = Counter(fs.labels)
        usage = pd.DataFrame({"category": classes,
                              "% of documents containing term": [100 * df_by_cat.get(c, 0) / size_by_cat[c] for c in classes]})
        fig = px.bar(usage, x="category", y="% of documents containing term", color="category",
                     color_discrete_map=cmap)
        fig.update_layout(showlegend=False)
        charts.show(fig, 330)

    st.markdown("##### Frequent bigrams per category")
    rows = {}
    for c in classes:
        cnt = Counter()
        for toks, lab in zip(fs.tokens, fs.labels):
            if lab != c:
                continue
            uni = [t for t in toks if "_" not in t]
            cnt.update(f"{a} {b}" for a, b in zip(uni, uni[1:]))
        rows[c] = [f"{bg} ({n})" for bg, n in cnt.most_common(10)]
    charts.table(pd.DataFrame(rows))

# ================================================================ Naive Bayes
with tab_nb:
    nb_cfg = state.cfg("nb")
    with st.form("nb_form"):
        c1, c2, c3, c4, c5 = st.columns(5)
        test_size = c1.slider("Test set share", 0.1, 0.5, nb_cfg.test_size, 0.05)
        seed = c2.number_input("Split seed", 0, 9999, nb_cfg.seed)
        alpha = c3.number_input("Laplace smoothing alpha", 0.001, 10.0, nb_cfg.alpha, 0.1, format="%.3f")
        feats = c4.radio("Features", ["counts", "tfidf"], ["counts", "tfidf"].index(nb_cfg.features),
                         format_func={"counts": "Term counts", "tfidf": "TF-IDF weights"}.get)
        grouped = c5.checkbox("Keep duplicate copies on one side of the split", nb_cfg.group_duplicates)
        if st.form_submit_button("Train & evaluate", type="primary"):
            state.set_cfg("nb", NBConfig(float(test_size), int(seed), float(alpha), feats, grouped))
            st.rerun()

    cr = state.classification()
    s = cr.scores
    m = st.columns(6)
    m[0].metric("Accuracy", f"{s['accuracy']:.3f}", border=True)
    m[1].metric("Macro precision", f"{s['macro_precision']:.3f}", border=True)
    m[2].metric("Macro recall", f"{s['macro_recall']:.3f}", border=True)
    m[3].metric("Macro F1", f"{s['macro_f1']:.3f}", border=True)
    m[4].metric("Train / test docs", f"{len(cr.train_idx)} / {len(cr.test_idx)}", border=True)
    m[5].metric("Test docs with a copy in train", cr.leakage, border=True,
                help="Exact duplicates split across train and test inflate accuracy.")

    with st.expander("How the classifier works / model parameters"):
        st.latex(r"\hat c = \arg\max_c \Big[\log P(c) + \sum_{t \in d} tf_{t,d}\,\log P(t\mid c)\Big],\quad "
                 r"P(t\mid c)=\frac{count(t,c)+\alpha}{\sum_{t'}count(t',c)+\alpha|V|}")
        d = cr.model.describe()
        priors = pd.DataFrame({"class": d["classes"],
                               "training documents": cr.model.class_count_.astype(int),
                               "prior P(c)": [d["priors"][c] for c in d["classes"]],
                               "total term count": cr.model.feature_count_.sum(axis=1).round(1)})
        charts.table(priors)
        st.caption(f"Vocabulary |V| = {d['vocabulary_size']:,} · alpha = {d['alpha']} · features = {cr.config.features}")

    l, r = st.columns(2)
    with l:
        dist = pd.DataFrame({"category": classes * 2,
                             "documents": [cr.y_true.count(c) for c in classes] + [cr.y_pred.count(c) for c in classes],
                             "set": ["actual"] * len(classes) + ["predicted"] * len(classes)})
        fig = px.bar(dist, x="category", y="documents", color="set", barmode="group", text="documents",
                     color_discrete_map={"actual": "#8A8F98", "predicted": "#2F6DB5"})
        charts.show(fig, 420, "Actual vs predicted class distribution (test set)")
    with r:
        cm = cr.confusion.copy()
        cm.index = [i.replace("actual: ", "") for i in cm.index]
        cm.columns = [c.replace("pred: ", "") for c in cm.columns]
        fig = px.imshow(cm.values, x=list(cm.columns), y=list(cm.index), text_auto=True,
                        color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="docs"))
        charts.show(fig, 420, "Confusion matrix")

    st.markdown("##### Per-class metrics")
    rep = cr.report.copy()
    for col in ("precision", "recall", "f1"):
        rep[col] = rep[col].round(4)
    charts.table(rep)

    st.markdown("##### Test-set predictions")
    pos = {c: i for i, c in enumerate(cr.classes)}
    pred_df = pd.DataFrame({
        "doc_id": [fs.doc_ids[i] for i in cr.test_idx],
        "title": [fs.titles[i] for i in cr.test_idx],
        "actual": cr.y_true, "predicted": cr.y_pred,
        "confidence": [float(cr.proba[n, pos[p]]) for n, p in enumerate(cr.y_pred)],
    })
    pred_df["correct"] = pred_df["actual"] == pred_df["predicted"]
    only_err = st.toggle(f"Show only misclassified ({int((~pred_df['correct']).sum())})")
    view = pred_df[~pred_df["correct"]] if only_err else pred_df
    charts.table(view, height=360, column_config={
        "confidence": st.column_config.ProgressColumn("confidence", min_value=0.0, max_value=1.0, format="%.3f"),
        "correct": st.column_config.CheckboxColumn("correct")})

    st.markdown("##### Most indicative terms per class")
    st.caption("Score = log P(term | class) minus the average log P(term | other classes).")
    grid = st.columns(len(cr.top_terms))
    for col, (c, terms) in zip(grid, cr.top_terms.items()):
        col.markdown(f"**{c}**")
        col.dataframe(pd.DataFrame(terms, columns=["term", "score"]).round(2), hide_index=True, height=300)

# ================================================================ K-Means
with tab_km:
    km_cfg = state.cfg("km")
    with st.form("km_form"):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        k = c1.slider("k (clusters)", 2, 12, km_cfg.k)
        init = c2.selectbox("Initialisation", ["k-means++", "random"], ["k-means++", "random"].index(km_cfg.init))
        max_iter = c3.number_input("Max iterations", 1, 500, km_cfg.max_iter)
        tol = c4.select_slider("Tolerance", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2], km_cfg.tol, format_func=lambda v: f"{v:g}")
        seed = c5.number_input("Seed", 0, 9999, km_cfg.seed)
        n_init = c6.number_input("Restarts (keep lowest SSE)", 1, 30, km_cfg.n_init)
        if st.form_submit_button("Run K-Means", type="primary"):
            state.set_cfg("km", KMConfig(int(k), init, int(max_iter), float(tol), int(seed), int(n_init)))
            st.rerun()

    kr = state.clustering()
    km = kr.km
    m = st.columns(6)
    m[0].metric("Clusters (k)", kr.config.k, border=True)
    m[1].metric("Iterations", km.n_iter, border=True)
    m[2].metric("SSE (within-cluster)", f"{km.sse:.2f}", border=True)
    m[3].metric("Purity", f"{kr.purity:.3f}", border=True)
    m[4].metric("Rand Index", f"{kr.rand['rand_index']:.3f}", border=True)
    m[5].metric("Runtime", f"{kr.seconds:.2f} s", border=True)

    st.markdown("##### Algorithm steps")
    s1, s2 = st.columns([2, 3])
    with s1:
        with st.container(border=True):
            st.markdown(f"**Step 1 - Initialisation ({kr.config.init}, seed {km.seed})**: "
                        f"{kr.config.k} documents picked as starting centroids.")
            init_df = pd.DataFrame({"cluster": range(len(km.init_indices)),
                                    "seed doc": [fs.doc_ids[i] for i in km.init_indices],
                                    "title": [fs.titles[i] for i in km.init_indices],
                                    "category": [fs.labels[i] for i in km.init_indices]})
            charts.table(init_df)
            if len(km.all_runs) > 1:
                st.caption("Restarts (the run with the lowest SSE is kept):")
                charts.table(pd.DataFrame(km.all_runs))
        with st.container(border=True):
            st.markdown("**Step 2 - Assignment**: each document joins the nearest centroid "
                        r"($\arg\min_j \lVert x - \mu_j \rVert^2$).  "
                        "\n\n**Step 3 - Update**: each centroid moves to the mean of its documents "
                        r"($\mu_j = \frac{1}{|C_j|}\sum_{x\in C_j} x$).  "
                        f"\n\n**Step 4 - Stopping criterion**: stop when no document changes cluster, the largest "
                        f"centroid shift is below tol = {kr.config.tol:g}, or after {kr.config.max_iter} iterations.")
            st.success(f"Stopped: {km.stop_reason}")
    with s2:
        hist = pd.DataFrame(km.history)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=hist["iteration"], y=hist["reassigned"], name="documents reassigned",
                             marker_color="#D9822B", opacity=0.6), secondary_y=True)
        fig.add_trace(go.Scatter(x=hist["iteration"], y=hist["sse"], name="SSE", mode="lines+markers",
                                 line=dict(color="#2F6DB5")), secondary_y=False)
        fig.update_yaxes(title_text="SSE", secondary_y=False)
        fig.update_yaxes(title_text="reassigned", secondary_y=True)
        fig.update_xaxes(title_text="iteration", dtick=1)
        charts.show(fig, 330, "Convergence")
        hist_view = hist.copy()
        hist_view["cluster_sizes"] = hist_view["cluster_sizes"].astype(str)
        charts.table(hist_view, height=220)

    st.markdown("##### Cluster visualisation")
    coords, explained = state.projection()
    color_by = st.radio("Colour points by", ["K-Means cluster", "True category"], horizontal=True)
    scatter = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "doc_id": fs.doc_ids, "title": fs.titles,
                            "category": fs.labels, "cluster": [f"cluster {c}" for c in km.labels]})
    key = "cluster" if color_by == "K-Means cluster" else "category"
    cluster_names = sorted(scatter["cluster"].unique(), key=lambda s: int(s.split()[-1]))
    color_seq = px.colors.qualitative.Bold + px.colors.qualitative.Pastel
    colors = ({n: color_seq[i % len(color_seq)] for i, n in enumerate(cluster_names)} if key == "cluster" else cmap)
    fig = px.scatter(scatter, x="PC1", y="PC2", color=key, hover_data=["doc_id", "title", "category", "cluster"],
                     color_discrete_map=colors, category_orders={"cluster": cluster_names}, opacity=0.75)
    fig.update_traces(marker=dict(size=6))
    cent = scatter.groupby("cluster")[["PC1", "PC2"]].mean().reset_index()
    if key == "cluster":
        fig.add_trace(go.Scatter(x=cent["PC1"], y=cent["PC2"], mode="markers+text", text=cent["cluster"],
                                 textposition="top center", name="centroids",
                                 marker=dict(symbol="x", size=14, color="black", line=dict(width=2))))
    charts.show(fig, 560, "Documents in 2-D (PCA, for display only)")
    st.caption(f"PCA is used only to draw this plot; it explains {100 * sum(explained):.1f}% of the variance. "
               "Clustering runs on the full TF-IDF vectors.")

    l, r = st.columns(2)
    sizes = pd.DataFrame({"cluster": [f"cluster {c}" for c in range(kr.config.k)],
                          "documents": np.bincount(km.labels, minlength=kr.config.k)})
    sizes["majority category"] = [kr.mapping.get(c, "-") for c in range(kr.config.k)]
    with l:
        fig = px.bar(sizes, x="cluster", y="documents", color="majority category", text="documents",
                     color_discrete_map=cmap)
        charts.show(fig, 380, "Documents per cluster")
    with r:
        charts.heatmap(kr.contingency, title="Cluster vs true category (contingency table)", height=380)

    st.markdown("##### Cluster membership")
    member = pd.DataFrame({"doc_id": fs.doc_ids, "title": fs.titles, "true category": fs.labels,
                           "cluster": km.labels, "cluster's majority category": [kr.mapping[c] for c in km.labels]})
    member["matches majority"] = member["true category"] == member["cluster's majority category"]
    pick = st.multiselect("Filter clusters", list(range(kr.config.k)), placeholder="All clusters")
    charts.table(member[member["cluster"].isin(pick)] if pick else member, height=360)

    st.markdown("##### Top terms per cluster (highest centroid weights)")
    grid = st.columns(min(kr.config.k, 6))
    for c, terms in kr.top_terms.items():
        with grid[c % len(grid)]:
            st.markdown(f"**cluster {c}** · {kr.mapping.get(c, '')}")
            st.caption(", ".join(t for t, _ in terms[:10]))

# ================================================================ evaluation & comparison
with tab_eval:
    cr, kr = state.classification(), state.clustering()
    ri = kr.rand
    l, r = st.columns(2)
    with l:
        with st.container(border=True):
            st.markdown("**Purity**")
            st.latex(r"\text{Purity}(\Omega, C) = \frac{1}{N}\sum_k \max_j |\omega_k \cap c_j|")
            correct = int(kr.contingency.max(axis=1).sum())
            st.markdown(f"Sum of the largest class in each cluster = **{correct}** of **{len(fs)}** documents "
                        f"-> Purity = **{kr.purity:.4f}**")
    with r:
        with st.container(border=True):
            st.markdown("**Rand Index**")
            st.latex(r"RI = \frac{TP + TN}{TP + FP + FN + TN}")
            pairs = pd.DataFrame({"": ["same cluster", "different cluster"],
                                  "same class": [f"TP = {ri['TP']:,}", f"FN = {ri['FN']:,}"],
                                  "different class": [f"FP = {ri['FP']:,}", f"TN = {ri['TN']:,}"]})
            charts.table(pairs)
            st.markdown(f"({ri['TP']:,} + {ri['TN']:,}) / {ri['total_pairs']:,} pairs = **{ri['rand_index']:.4f}**")

    # Same-footing comparison on the test documents
    test = cr.test_idx
    km_test = [int(kr.km.labels[i]) for i in test]
    rows = [
        {"method": "Naive Bayes (supervised)", "evaluated on": f"test set ({len(test)} docs)",
         "accuracy": cr.scores["accuracy"], "macro F1": cr.scores["macro_f1"],
         "purity": purity(cr.y_true, cr.y_pred), "rand index": rand_index(cr.y_true, cr.y_pred)["rand_index"]},
        {"method": "K-Means (unsupervised)", "evaluated on": f"all documents ({len(fs)})",
         "accuracy": kr.mapped_accuracy, "macro F1": None,
         "purity": kr.purity, "rand index": ri["rand_index"]},
        {"method": "K-Means (unsupervised)", "evaluated on": f"test set ({len(test)} docs)",
         "accuracy": None, "macro F1": None,
         "purity": purity(cr.y_true, km_test), "rand index": rand_index(cr.y_true, km_test)["rand_index"]},
    ]
    comp = pd.DataFrame(rows)
    st.markdown("##### Side-by-side comparison")
    st.caption("K-Means 'accuracy' = each cluster labelled with its majority category (only meaningful when k "
               "equals the number of classes). Purity and Rand Index for Naive Bayes treat predicted classes as clusters.")
    charts.table(comp.round(4))
    long = comp.melt(id_vars=["method", "evaluated on"], var_name="metric", value_name="score").dropna()
    long["model"] = long["method"] + " · " + long["evaluated on"]
    fig = px.bar(long, x="metric", y="score", color="model", barmode="group", text_auto=".3f",
                 color_discrete_sequence=["#2F6DB5", "#D9822B", "#F2B880"])
    fig.update_yaxes(range=[0, 1.05])
    charts.show(fig, 420, "Naive Bayes vs K-Means")

    tn_share = ri["TN"] / ri["total_pairs"]
    gap = cr.scores["accuracy"] - kr.purity
    st.markdown("##### Interpretation")
    st.markdown(
        f"- **Naive Bayes** learns from {len(cr.train_idx)} labelled articles and reaches **{cr.scores['accuracy']:.1%} accuracy** "
        f"(macro F1 {cr.scores['macro_f1']:.3f}) on unseen articles. It is told which categories exist and which words signal them.\n"
        f"- **K-Means** never sees a label, yet its {kr.config.k} clusters reach **purity {kr.purity:.3f}** and "
        f"**Rand Index {ri['rand_index']:.3f}**: the vocabulary alone separates the desks well. "
        f"The gap to Naive Bayes ({gap:+.3f}) is the price of having no supervision.\n"
        f"- The Rand Index is high partly because **{tn_share:.0%} of all pairs are true negatives** "
        "(different class, different cluster). With 5 classes most pairs are 'different', so RI should be read "
        "together with purity and the contingency table.\n"
        "- Purity rewards many small clusters (k = N gives purity 1), so it is only comparable at a fixed k. "
        "The k sweep on the Performance page shows this trade-off.\n"
        "- **When to use which:** classification when labelled history exists and categories are fixed (routing "
        "stories to desks); clustering to discover new or cross-cutting themes when there are no labels.")
