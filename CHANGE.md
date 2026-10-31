# Changes in this PR

Three fixes from the pre-submission review, plus the regenerated results. Items 4, 5 and 6 from the review (repo visibility, lab execution, report PDF) are not in this PR.

## 1. Train-only TF-IDF fit for Naive Bayes

Problem. `build_features` fits the TF-IDF vocabulary (top 5000 terms by document frequency), the min_df/max_df filter and the IDF values on all documents. The 80/20 train/test split happens afterwards in `run_classification`. So the vocabulary selection and the IDF see the test documents. The internal `leakage` counter only measures exact-text duplicate leakage between train and test, which was already zero after exact dedup; it does not catch the TF-IDF leakage.

Fix. `run_classification` now does the stratified split first, fits a fresh `TfidfModel` on the training tokens using the same `tfidf_cfg`, and transforms the test tokens through that train vocabulary. Both "counts" and "tfidf" feature paths now use the train-fit vocabulary. Clustering and the recommender are unsupervised, so they keep using the full-corpus fit as before.

Files: `core/pipeline.py`.

## 2. Default normalizer is now Porter stemming

Problem. The default was WordNet lemmatization. If the lab machine cannot download the WordNet data, the code falls back to a small rule-based lemmatizer and does not surface the switch, so lab numbers may not match laptop numbers.

Fix. `PreprocessConfig.normalizer` now defaults to `"stem"`. Porter stemming is bundled with NLTK and needs no download. WordNet lemmatization is still available in the UI by picking "lemmatize", and `baseline.json` records the active backend when lemmatize is used (new `lemmatizer_backend` field).

Files: `core/preprocessing.py`, `scripts/run_experiments.py`.

## 3. Near-duplicate threshold sweep

Problem. The cosine threshold of 0.90 used by near-duplicate detection was not justified.

Fix. New experiment `near_duplicate_sweep` runs the full pipeline at thresholds 0.80, 0.85, 0.90 and 0.95 and writes `results/exp7_near_dup_threshold.csv`. The default `DedupConfig.remove_near` stays `False` so baseline numbers are unchanged in logic; the sweep is a report-side artifact for inference question 2.

Files: `core/experiments.py`, `scripts/run_experiments.py`.

## Regenerated results

The combined effect of fixes 1 and 2 shifts the headline numbers slightly (K-Means scores improve because stemming gives a tighter vocabulary; NB scores are unchanged within noise):

| Task | Before | After |
|---|---|---|
| NB accuracy | 0.9712 | 0.9736 |
| NB macro F1 | 0.9702 | 0.9732 |
| K-Means Purity (k=5) | 0.9482 | 0.9660 |
| K-Means Rand Index (k=5) | 0.9603 | 0.9738 |
| Recommender P@5 (content) | 0.9300 | 0.9327 |
| Recommender P@5 (hybrid, w=0.2) | 0.9610 | 0.9631 |

All CSVs under `results/` and `results/baseline.json` are regenerated from the full corpus. These will be overwritten again by the lab run before submission.

## Tests

`python -m tests.test_core` still passes all 13 tests.
