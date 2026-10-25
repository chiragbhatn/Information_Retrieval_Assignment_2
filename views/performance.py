from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from core import experiments as ex
from ui import charts, state

RESULTS = Path(__file__).resolve().parents[1] / "results"

st.title("Performance Analytics")
st.caption("Headline metrics for the current configuration, runtime per stage, and experiments that show how "
           "preprocessing, duplicates, k and initialisation change the results.")

fs, cr, kr = state.features(), state.classification(), state.clustering()
rec5 = state.recommendation_quality(5)

m = st.columns(6)
m[0].metric("NB accuracy", f"{cr.scores['accuracy']:.3f}", border=True)
m[1].metric("NB macro F1", f"{cr.scores['macro_f1']:.3f}", border=True)
m[2].metric("K-Means purity", f"{kr.purity:.3f}", border=True)
m[3].metric("K-Means Rand Index", f"{kr.rand['rand_index']:.3f}", border=True)
m[4].metric("Recommender P@5", f"{rec5['precision_at_k']:.3f}", border=True)
m[5].metric("Documents used", f"{len(fs):,} / {fs.input_size:,}", border=True)

l, r = st.columns([3, 2])
with l:
    timing = {**{f"features: {k}": v for k, v in fs.timings.items()},
              "naive bayes (train + test)": cr.seconds, "k-means": kr.seconds}
    tdf = pd.DataFrame({"stage": list(timing), "seconds": list(timing.values())})
    fig = px.bar(tdf, x="seconds", y="stage", orientation="h", text_auto=".2f", color_discrete_sequence=["#2F6DB5"])
    fig.update_yaxes(autorange="reversed", title=None)
    charts.show(fig, 330, "Runtime per stage (first run; later runs are cached)")
with r:
    comp = pd.DataFrame([
        {"metric": "Accuracy", "Naive Bayes": cr.scores["accuracy"], "K-Means": kr.mapped_accuracy},
        {"metric": "Macro F1", "Naive Bayes": cr.scores["macro_f1"], "K-Means": None},
        {"metric": "Purity", "Naive Bayes": None, "K-Means": kr.purity},
        {"metric": "Rand Index", "Naive Bayes": None, "K-Means": kr.rand["rand_index"]},
    ])
    st.markdown("**Classification vs clustering**")
    charts.table(comp.round(4))
    st.caption("K-Means accuracy = clusters labelled by their majority category. "
               "See the Text Mining Hub > Evaluation tab for the full comparison.")

st.divider()
st.subheader("Experiments")
src = "saved results (scripts/run_experiments.py)" if RESULTS.exists() else "none saved yet"
st.caption(f"Each experiment can be run live on the active corpus ({len(state.corpus()):,} documents). "
           f"Saved results folder: {src}. Live results replace saved ones for this session.")


def experiment(key: str, title: str, runner, help_text: str):
    """Show saved or live results for one experiment; returns a DataFrame or None."""
    st.markdown(f"#### {title}")
    st.caption(help_text)
    c1, c2 = st.columns([1, 4])
    if c1.button("Run live", key=f"run_{key}", icon=":material/play_arrow:"):
        bar = st.progress(0.0, text="Starting...")
        st.session_state[f"exp_{key}"] = runner(lambda frac, name: bar.progress(frac, text=name))
        bar.empty()
    df = st.session_state.get(f"exp_{key}")
    origin = "live run"
    if df is None and (RESULTS / f"{key}.csv").exists():
        df, origin = pd.read_csv(RESULTS / f"{key}.csv"), "saved file"
    if df is None:
        c2.info("Not run yet.")
        return None
    c2.caption(f"Showing: {origin}")
    charts.table(df.round(4))
    st.download_button("Download CSV", df.to_csv(index=False), f"{key}.csv", "text/csv", key=f"dl_{key}")
    return df


corpus = state.corpus()
cfgs = dict(tfidf_cfg=state.cfg("tfidf"), nb_cfg=state.cfg("nb"), km_cfg=state.cfg("km"))
t1, t2, t3, t4, t5, t6 = st.tabs(["1. Preprocessing", "2. Duplicates", "3. Choice of k", "4. Initialisation",
                                  "5. Recommender", "6. Naive Bayes variants"])

with t1:
    df = experiment("exp1_preprocessing_ablation", "Effect of preprocessing choices",
                    lambda p: ex.preprocessing_ablation(corpus, dedup=state.cfg("dedup"), progress=p, **cfgs),
                    "Same data, eight pipelines from minimal to full. Answers report question 1.")
    if df is not None:
        df = df.assign(code=df["config"].str.split(".").str[0])
        l, r = st.columns(2)
        with l:
            fig = px.bar(df, x="code", y=["full_vocabulary", "vocab_after_df_filter"], barmode="group",
                         hover_data=["config"], log_y=True)
            fig.update_xaxes(title="configuration (see table)")
            charts.show(fig, 400, "Vocabulary size (log scale)")
        with r:
            metric_cols = [c for c in df.columns if c in ("nb_accuracy", "nb_macro_f1", "km_purity", "km_rand_index")
                           or c.startswith("rec_precision")]
            fig = px.line(df, x="code", y=metric_cols, markers=True, hover_data=["config"])
            fig.update_xaxes(title="configuration (see table)")
            charts.show(fig, 400, "Quality metrics")

with t2:
    df = experiment("exp2_duplicate_impact", "Effect of duplicate and near-duplicate documents",
                    lambda p: ex.duplicate_impact(corpus, state.cfg("prep"), progress=p, **cfgs),
                    "Keeps or removes duplicates and compares a random split with a duplicate-aware split. "
                    "Answers report question 2.")
    if df is not None:
        metric_cols = [c for c in ("nb_accuracy", "km_purity", "km_rand_index") if c in df.columns] + \
                      [c for c in df.columns if c.startswith("rec_precision")]
        long = df.melt(id_vars="scenario", value_vars=metric_cols, var_name="metric", value_name="score")
        fig = px.bar(long, x="metric", y="score", color="scenario", barmode="group", text_auto=".3f")
        fig.update_yaxes(range=[0.5, 1.02])
        charts.show(fig, 420, "Metrics by duplicate-handling scenario")

with t3:
    df = experiment("exp3_k_sweep", "Choosing k", lambda p: ex.k_sweep(fs, progress=p),
                    "K-Means for k = 2..10 (k-means++, 3 restarts). SSE always falls as k grows; "
                    "purity and Rand Index show where clusters match the real categories.")
    if df is not None:
        l, r = st.columns(2)
        with l:
            charts.show(px.line(df, x="k", y="sse", markers=True), 360, "Elbow plot (SSE)")
        with r:
            charts.show(px.line(df, x="k", y=["purity", "rand_index"], markers=True), 360, "Purity and Rand Index vs k")

with t4:
    df = experiment("exp4_init_comparison", "Random vs k-means++ initialisation",
                    lambda p: ex.init_comparison(fs, k=state.cfg("km").k),
                    "Five seeds each, one run per seed. Shows how sensitive K-Means is to the starting centroids.")
    if df is not None:
        l, r = st.columns(2)
        with l:
            charts.show(px.strip(df, x="init", y="purity", color="init", hover_data=["seed", "sse"]), 360, "Purity by seed")
        with r:
            charts.show(px.scatter(df, x="sse", y="purity", color="init", hover_data=["seed"]), 360,
                        "Lower SSE vs purity")

with t5:
    df = experiment("exp5_recommender_eval", "Recommendation quality",
                    lambda p: ex.recommender_eval(fs, kr.km.labels),
                    "Precision@K with every article as a query (relevant = same category), content-based vs hybrid.")
    if df is not None:
        charts.show(px.line(df, x="k", y="precision_at_k", color="method", markers=True), 380, "Precision@K")

with t6:
    df = experiment("exp6_naive_bayes_variants", "Naive Bayes: smoothing and feature type",
                    lambda p: ex.nb_variants(fs, state.cfg("nb")),
                    "Laplace alpha from 0.01 to 2, with term counts vs TF-IDF weights as features.")
    if df is not None:
        fig = px.line(df, x="alpha", y="accuracy", color="features", markers=True, log_x=True)
        charts.show(fig, 380, "Accuracy vs smoothing alpha")
