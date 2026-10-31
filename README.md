# NewsLens: BBC News Text Mining & Document Intelligence System

**Information Retrieval (AIMLCZG537 / DSECLZG537) · BITS Pilani WILP · Assignment 2 · Group G07**

NewsLens is a Streamlit app that runs a full text-mining pipeline on the BBC News corpus. The setting is the news-monitoring desk of an investment research firm. The app routes each story to the right desk (Naive Bayes classification), groups stories into themes (K-Means clustering), removes duplicate wire copies (duplicate detection) and recommends related coverage (cosine similarity on TF-IDF). The core algorithms are written by hand. scikit-learn is not used.

| Name | Registration number |
|---|---|
| Devansh Sadhotra | 2025AG05360 |
| Gunda Ravindra | 2025AG05539 |
| Vanshika Sangal | 2025AG05381 |
| Chirag Bhatnagar | 2025AG05066 |
| Bhatraju Sreevarsha | 2025AG05363 |

**Dataset.** Hugging Face [`gopalkalpande/bbc-news-summary`](https://huggingface.co/datasets/gopalkalpande/bbc-news-summary) (CC0). The original corpus is Greene & Cunningham (2006). It has 2,224 articles in 5 categories: business 510, sport 510, politics 417, tech 401, entertainment 386. That is 4.96 MB of text, with 138 exact duplicates.

---

## 1. Requirements and installation

- Python 3.10 or newer. Tested locally with Python 3.10 on Windows. The BITS Virtual Lab runs Linux (see section 4).
- Libraries (`requirements.txt`): Streamlit, pandas, numpy, Plotly, NLTK, openpyxl.

Create a virtual environment and install:

```bash
python -m venv .venv
```

Activate it.

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Windows (Command Prompt):

```bat
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Then install the packages:

```bash
pip install -r requirements.txt
```

**NLTK data.** Porter stemming is the default normalizer and needs no download. WordNet lemmatization is available in the UI; its data downloads the first time it is used, which needs internet. If the download fails, the app falls back to a small rule-based lemmatizer and `results/baseline.json` records the active backend.

---

## 2. Run the app

```bash
streamlit run app.py
```

Streamlit opens the app at `http://localhost:8501`. In the BITS lab (or any remote Linux machine), use:

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
```

The app has six pages: **Dashboard**, **Data Ingestion**, **Preprocessing & TF-IDF**, **Text Mining Hub**, **Recommendations** and **Performance Analytics**. Data Ingestion can also load your own CSV / JSON file or a ZIP of `.txt` files.

**Quick run on a sample.** Set `IR_SAMPLE` to load a stratified sample instead of the full 2,224 articles. Leave it unset (or `0`) for the full corpus.

bash:

```bash
IR_SAMPLE=600 streamlit run app.py
```

PowerShell:

```powershell
$env:IR_SAMPLE = "600"; streamlit run app.py
# afterwards, back to the full corpus:
Remove-Item Env:IR_SAMPLE
```

---

## 3. Reproduce the results

Run every experiment on the full corpus. This takes about 1 minute on a laptop and writes the tables to `results/`:

```bash
python scripts/run_experiments.py
```

Quick check on a stratified sample of 500 articles:

```bash
python scripts/run_experiments.py --sample 500
```

Note: `--sample` also writes to `results/` and overwrites the full-corpus tables. Add `--out <folder>` to write somewhere else, for example `--out results_sample`.

**Files in `results/`**

| File | Contents |
|---|---|
| `baseline.json` | Default run: documents in/used, duplicates removed, TF-IDF summary (vocabulary, sparsity), Naive Bayes scores, K-Means Purity / Rand Index / SSE / iterations, and every configuration used |
| `baseline_classification_report.csv` | Naive Bayes precision, recall, F1 and support per class, plus macro and weighted averages |
| `baseline_confusion_matrix.csv` | Naive Bayes confusion matrix (actual vs predicted) |
| `baseline_contingency.csv` | K-Means clusters vs true categories |
| `exp1_preprocessing_ablation.csv` | 8 preprocessing set-ups (A to H: minimal, case kept, stop-words, stemming, lemmatization, bigrams, news stop-words) vs vocabulary size, NB accuracy/F1, K-Means Purity/RI and Precision@5 |
| `exp2_duplicate_impact.csv` | Keep vs remove duplicates, and random vs duplicate-aware train/test split |
| `exp3_k_sweep.csv` | K-Means for k = 2 to 10: SSE, Purity, Rand Index, iterations |
| `exp4_init_comparison.csv` | Random vs k-means++ initialisation over 10 seeds (k = 5) |
| `exp5_recommender_eval.csv` | Content-based vs hybrid (cluster weight 0.2 and 0.5) recommender: Precision@1/3/5/10 and how often a duplicate appears in the Top-K |
| `exp6_naive_bayes_variants.csv` | Naive Bayes on counts vs TF-IDF features, smoothing alpha 0.01 to 2.0 |
| `exp7_near_dup_threshold.csv` | Near-duplicate removal at cosine thresholds 0.80, 0.85, 0.90, 0.95 (documents kept, near pairs left, NB and K-Means scores, Precision@5) |
| `experiments_log.txt`, `tests_output.txt` | Console output saved by `scripts/lab_run.sh` |
| `lab_environment.txt` | Lab machine details and run time (created by `scripts/lab_run.sh`) |

Headline numbers (full corpus, default settings) are produced by `python scripts/run_experiments.py` and written to `results/baseline.json`.

**Run the tests** (13 tests, including the textbook examples from Manning et al. for Naive Bayes, Purity and Rand Index):

```bash
python -m tests.test_core
```

**Rebuild the dataset files** from the raw Hugging Face CSV (`data/raw/bbc-news-summary.csv`):

```bash
python scripts/prepare_dataset.py
```

This writes `data/bbc_documents.csv` (`doc_id, text`) and `data/bbc_metadata.csv` (`doc_id, title, category, source, summary`). Text and metadata are kept in separate files and joined on `doc_id`.

---

## 4. BITS Virtual Lab (RunPod JupyterLab)

1. Start the lab and open JupyterLab. Open a terminal (**File → New → Terminal**).
2. Get the project into the pod. Either `git clone` the repository, or upload the project ZIP and unzip it. Then `cd` into the project folder.
3. Run the one-shot script:

   ```bash
   bash scripts/lab_run.sh
   ```

   It installs the requirements, runs the tests, runs all experiments on the full corpus, and packs `results/` into `lab_results.zip`.

   If `python3` is not the right command:

   ```bash
   PYTHON=python bash scripts/lab_run.sh
   ```

   Quick run on a 300-article sample:

   ```bash
   SAMPLE=300 bash scripts/lab_run.sh
   ```

4. Start the app:

   ```bash
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
   ```

5. Open the app in your browser. The pod exposes only port 8888.
   - **Option A: Jupyter server proxy** (if it is installed). Open `https://<pod-id>-8888.proxy.runpod.net/proxy/8501/`. Use the same pod ID as your JupyterLab URL.
   - **Option B: temporary Cloudflare tunnel.** Open a second terminal. Download `cloudflared` from the official GitHub releases (https://github.com/cloudflare/cloudflared/releases) and start a tunnel:

     ```bash
     wget -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
     chmod +x cloudflared
     ./cloudflared tunnel --url http://localhost:8501
     ```

     It prints a `https://....trycloudflare.com` link. Open it. The link stops working when you stop the tunnel.

6. **Before you stop the pod, download `lab_results.zip`** (right-click it in the JupyterLab file browser → Download). Also take your screenshots or recording.
   - The pod is deleted when it stops. Anything not downloaded is lost.
   - Sessions end after 15 idle minutes.
   - Usage is billed per minute, so stop the pod when you are done.

---

## 5. Project structure

```
ir assignment 2/
├── app.py                    # Streamlit entry point: page navigation and sidebar
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── core/                     # Text-mining library (no Streamlit code)
│   ├── data_loader.py        # Corpus class; loads bundled data and CSV / JSON / ZIP uploads
│   ├── preprocessing.py      # Cleaning, tokenizing, stop-words, stemming / lemmatization, n-grams
│   ├── resources.py          # Bundled English and news stop-word lists
│   ├── tfidf.py              # Inverted index, TF-IDF with length normalisation, cosine similarity
│   ├── duplicates.py         # Exact (hash) and near-duplicate (cosine) detection
│   ├── naive_bayes.py        # Multinomial Naive Bayes and stratified, duplicate-aware split
│   ├── kmeans.py             # K-Means with k-means++ / random init and restarts
│   ├── metrics.py            # Accuracy, precision, recall, F1, confusion matrix, Purity, Rand Index
│   ├── recommender.py        # Top-K content-based and hybrid recommendations, Precision@K
│   ├── reduction.py          # PCA to 2-D (used only for the cluster plot)
│   ├── pipeline.py           # Ties the steps together; config classes
│   └── experiments.py        # The six experiments (used by the app and the script)
├── views/                    # The six app pages
│   ├── dashboard.py          # Dashboard
│   ├── ingestion.py          # Data Ingestion
│   ├── preprocessing.py      # Preprocessing & TF-IDF
│   ├── mining_hub.py         # Text Mining Hub (classification, clustering)
│   ├── recommendations.py    # Recommendations
│   └── performance.py        # Performance Analytics
├── ui/
│   ├── state.py              # Session state and cached pipeline steps (reads IR_SAMPLE)
│   └── charts.py             # Shared Plotly colours and chart helpers
├── scripts/
│   ├── run_experiments.py    # Runs all experiments headless, writes results/
│   ├── lab_run.sh            # One-shot BITS lab run: install, test, experiments, zip
│   └── prepare_dataset.py    # Builds data/*.csv from the raw Hugging Face CSV
├── tests/
│   └── test_core.py          # 13 sanity tests for the hand-written algorithms
├── data/
│   ├── raw/bbc-news-summary.csv  # Raw dataset from Hugging Face
│   ├── bbc_documents.csv     # doc_id, text
│   └── bbc_metadata.csv      # doc_id, title, category, source, summary
└── results/                  # Experiment outputs (see section 3)
```

---

## 6. How it works

| Step | What happens | File |
|---|---|---|
| 1. Load | Text and metadata are loaded separately and joined on `doc_id` | `core/data_loader.py` |
| 2. Preprocess | Fix encoding, strip markup, lowercase, drop numbers and punctuation, tokenize, remove stop-words, then Porter stemming or WordNet lemmatization (NLTK), optional n-grams. Each step can be switched on or off in the app | `core/preprocessing.py` |
| 3. Remove duplicates | Exact duplicates: hash of the normalised text. Near duplicates: cosine similarity ≥ 0.9 (off by default) | `core/duplicates.py` |
| 4. Index and weight | Inverted index; TF-IDF with `1 + log10(tf)` and `log10(N / df)`; vectors normalised to unit length; cosine similarity. Defaults: min_df 2, max_df 90 %, top 5,000 terms | `core/tfidf.py` |
| 5. Classify | Multinomial Naive Bayes with Laplace smoothing, in log space. 80/20 stratified split that keeps duplicates on the same side | `core/naive_bayes.py` |
| 6. Cluster | K-Means (k-means++ or random init), 10 restarts, keeps the run with the lowest SSE | `core/kmeans.py` |
| 7. Evaluate | Accuracy, precision, recall, F1, confusion matrix; Purity and Rand Index (pair counting) | `core/metrics.py` |
| 8. Recommend | Top-K by cosine similarity; hybrid mode adds a bonus for the same K-Means cluster | `core/recommender.py` |
| 9. Plot | PCA to 2-D, only to draw the cluster scatter plot | `core/reduction.py` |

`core/pipeline.py` connects steps 2 to 7 (`build_features`, `run_classification`, `run_clustering`). `ui/state.py` caches each step, so a change on one page updates the others. numpy is used only for array arithmetic.

---

## 7. Submission checklist (`G07_Assignment2.zip`)

- [ ] Source code: all `.py` files (`app.py`, `core/`, `views/`, `ui/`, `scripts/`, `tests/`) plus `requirements.txt` and `scripts/lab_run.sh`
- [ ] Dataset: `data/` (raw CSV and the two prepared CSV files)
- [ ] Report (PDF)
- [ ] BITS Virtual Lab screenshots / screen recording (and `lab_results.zip` contents)
- [ ] `README.md`
- [ ] `G07_Contribution.xlsx`
