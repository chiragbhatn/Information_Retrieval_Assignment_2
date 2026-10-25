"""Run every experiment headless and save the tables to results/.

    python scripts/run_experiments.py              # full corpus (use this in the lab)
    python scripts/run_experiments.py --sample 500 # quick local check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from core import experiments as ex  # noqa: E402
from core.data_loader import corpus_stats, load_bundled  # noqa: E402
from core.pipeline import (DedupConfig, KMConfig, NBConfig, build_features, config_snapshot,  # noqa: E402
                           run_classification, run_clustering)
from core.preprocessing import PreprocessConfig  # noqa: E402
from core.tfidf import TfidfConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="stratified sample size (0 = full corpus)")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = load_bundled()
    if args.sample:
        corpus = corpus.stratified_sample(args.sample)
    print(f"Corpus: {corpus.name} | {corpus_stats(corpus)}")

    prep, tfidf_cfg, dedup = PreprocessConfig(), TfidfConfig(), DedupConfig()
    nb_cfg, km_cfg = NBConfig(), KMConfig()
    t_all = time.perf_counter()

    def log(frac, name):
        print(f"   - {name}")

    def save(name: str, df: pd.DataFrame) -> None:
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"\n== {name} ==\n{df.round(4).to_string(index=False)}")

    # Baseline run with the default configuration
    fs = build_features(corpus, prep, tfidf_cfg, dedup)
    cr = run_classification(fs, nb_cfg)
    kr = run_clustering(fs, km_cfg)
    baseline = {
        "documents_in": fs.input_size, "documents_used": len(fs), "removed": {k: len(v) for k, v in fs.removed.items()},
        "tfidf": fs.model.summary(), "naive_bayes": {k: float(v) for k, v in cr.scores.items()},
        "kmeans": {"purity": kr.purity, "rand_index": kr.rand["rand_index"], "sse": kr.km.sse,
                   "iterations": kr.km.n_iter, "stop_reason": kr.km.stop_reason, "cluster_to_label": kr.mapping},
        "configs": config_snapshot(preprocess=prep, tfidf=tfidf_cfg, dedup=dedup, naive_bayes=nb_cfg, kmeans=km_cfg),
    }
    (out / "baseline.json").write_text(json.dumps(baseline, indent=2, default=str))
    save("baseline_classification_report", cr.report)
    save("baseline_confusion_matrix", cr.confusion.reset_index(names="actual"))
    save("baseline_contingency", kr.contingency.reset_index(names="cluster"))

    print("\nRunning preprocessing ablation...")
    save("exp1_preprocessing_ablation", ex.preprocessing_ablation(corpus, tfidf_cfg, dedup, nb_cfg, km_cfg, progress=log))
    print("\nRunning duplicate impact...")
    save("exp2_duplicate_impact", ex.duplicate_impact(corpus, prep, tfidf_cfg, nb_cfg, km_cfg, progress=log))
    save("exp3_k_sweep", ex.k_sweep(fs))
    save("exp4_init_comparison", ex.init_comparison(fs))
    save("exp5_recommender_eval", ex.recommender_eval(fs, kr.km.labels))
    save("exp6_naive_bayes_variants", ex.nb_variants(fs, nb_cfg))
    print(f"\nAll experiments finished in {time.perf_counter() - t_all:.1f}s -> {out}")


if __name__ == "__main__":
    main()
