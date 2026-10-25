"""Sanity checks for the hand-written algorithms. Run: python -m tests.test_core"""
import math
import random

import numpy as np

from core.kmeans import kmeans
from core.metrics import (accuracy, classification_report, purity, rand_index,
                          rand_index_bruteforce)
from core.naive_bayes import MultinomialNaiveBayes, stratified_split
from core.preprocessing import PreprocessConfig, add_ngrams, preprocess
from core.tfidf import TfidfConfig, TfidfModel, build_inverted_index, cosine_similarity


def test_inverted_index():
    idx = build_inverted_index([["a", "b", "a"], ["b", "c"]])
    assert idx == {"a": {0: 2}, "b": {0: 1, 1: 1}, "c": {1: 1}}


def test_tfidf_matches_hand_calculation():
    docs = [["car", "car", "insurance"], ["car", "auto"], ["insurance", "best"]]
    m = TfidfModel(TfidfConfig(min_df=1, max_df_ratio=1.0, max_features=0, sublinear_tf=True)).fit(docs)
    # doc 0: car tf=2, df=2 ; insurance tf=1, df=2 ; N=3
    w_car = (1 + math.log10(2)) * math.log10(3 / 2)
    w_ins = 1 * math.log10(3 / 2)
    norm = math.sqrt(w_car ** 2 + w_ins ** 2)
    assert abs(m.matrix[0, m.term_to_col["car"]] - w_car / norm) < 1e-6
    assert abs(m.matrix[0, m.term_to_col["insurance"]] - w_ins / norm) < 1e-6
    assert np.allclose(np.linalg.norm(m.matrix, axis=1), 1.0, atol=1e-6)


def test_cosine():
    assert abs(cosine_similarity(np.array([1.0, 0]), np.array([1.0, 1])) - 1 / math.sqrt(2)) < 1e-9
    assert cosine_similarity(np.zeros(2), np.ones(2)) == 0.0


def test_purity_and_rand_index_textbook_example():
    # Manning, Raghavan & Schuetze, Introduction to IR, Fig. 16.4: purity 0.71, RI 0.68
    clusters = [1] * 6 + [2] * 6 + [3] * 5
    truth = list("xxxxxo") + list("xooooD") + list("xxDDD")
    assert abs(purity(truth, clusters) - 12 / 17) < 1e-9
    ri = rand_index(truth, clusters)
    assert (ri["TP"], ri["FP"], ri["FN"], ri["TN"]) == (20, 20, 24, 72)
    assert abs(ri["rand_index"] - 92 / 136) < 1e-9


def test_rand_index_equals_bruteforce():
    rng = random.Random(0)
    for _ in range(20):
        t = [rng.choice("abc") for _ in range(40)]
        c = [rng.randint(0, 3) for _ in range(40)]
        assert abs(rand_index(t, c)["rand_index"] - rand_index_bruteforce(t, c)) < 1e-12


def test_classification_metrics():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    assert accuracy(y_true, y_pred) == 0.75
    rep = classification_report(y_true, y_pred, ["a", "b"]).set_index("class")
    assert rep.loc["a", "precision"] == 1.0 and rep.loc["a", "recall"] == 0.5
    assert abs(rep.loc["b", "precision"] - 2 / 3) < 1e-9


def test_naive_bayes_textbook_example():
    # Manning et al. Example 13.1: test doc "Chinese Chinese Chinese Tokyo Japan" -> class c
    vocab = ["Chinese", "Beijing", "Shanghai", "Macao", "Tokyo", "Japan"]
    col = {t: i for i, t in enumerate(vocab)}

    def vec(words):
        d = {}
        for w in words.split():
            d[col[w]] = d.get(col[w], 0) + 1
        return d

    train = [vec("Chinese Beijing Chinese"), vec("Chinese Chinese Shanghai"), vec("Chinese Macao"),
             vec("Tokyo Japan Chinese")]
    nb = MultinomialNaiveBayes(1.0).fit(train, ["c", "c", "c", "j"], len(vocab))
    assert abs(math.exp(nb.feature_log_prob_[0, col["Chinese"]]) - 3 / 7) < 1e-9
    assert abs(math.exp(nb.feature_log_prob_[1, col["Tokyo"]]) - 2 / 9) < 1e-9
    assert nb.predict([vec("Chinese Chinese Chinese Tokyo Japan")]) == ["c"]


def test_stratified_split_keeps_groups_together():
    labels = ["a"] * 10 + ["b"] * 10
    groups = list(range(20))
    groups[1] = groups[0]  # docs 0 and 1 are duplicates
    train, test = stratified_split(labels, 0.3, seed=1, groups=groups)
    assert sorted(train + test) == list(range(20))
    assert (0 in train) == (1 in train)


def test_stratified_split_keeps_single_document_class_in_training():
    labels = ["a"] * 10 + ["b"]
    train, test = stratified_split(labels, 0.3, seed=0)
    assert 10 in train and 10 not in test
    assert any(labels[i] == "a" for i in test) and any(labels[i] == "a" for i in train)


def test_read_table_formats():
    import io
    import json
    from core.data_loader import read_table
    rows = [{"id": 1, "text": "a b"}, {"id": 2, "text": "c d"}]
    pretty = json.dumps({"documents": rows}, indent=2).encode()
    assert len(read_table(io.BytesIO(pretty), "x.json")) == 2
    lines = "\n".join(json.dumps(r) for r in rows).encode()
    assert len(read_table(io.BytesIO(lines), "x.json")) == 2
    assert len(read_table(io.BytesIO(lines), "x.jsonl")) == 2


def test_empty_vocabulary_is_reported():
    import pandas as pd
    from core.data_loader import Corpus
    from core.pipeline import DedupConfig, build_features
    c = Corpus(pd.Series(["alpha", "beta"], index=["d1", "d2"]),
               pd.DataFrame({"category": ["x", "y"]}, index=["d1", "d2"]))
    try:
        build_features(c, PreprocessConfig(), TfidfConfig(min_df=2), DedupConfig())
    except ValueError as exc:
        assert "No terms are left" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_kmeans_separates_obvious_clusters():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (30, 2)), rng.normal(5, 0.1, (30, 2))]).astype(np.float32)
    res = kmeans(X, 2, seed=3)
    assert len(set(res.labels[:30])) == 1 and len(set(res.labels[30:])) == 1
    assert res.labels[0] != res.labels[-1]
    assert "no document changed" in res.stop_reason or "shift" in res.stop_reason


def test_preprocessing():
    cfg = PreprocessConfig(normalizer="stem", ngram_max=1)
    toks = preprocess("The <b>Markets</b> were RALLYING..Shares rose 5% to �200", cfg)
    assert toks == ["market", "ralli", "share", "rose"]
    assert add_ngrams(["a", "b", "c"], 2) == ["a", "b", "c", "a_b", "b_c"]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} tests passed")
