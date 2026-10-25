#!/usr/bin/env bash
# One-shot run for the BITS Virtual Lab:
#   install -> sanity tests -> all experiments on the full corpus -> zip results
#
#   bash scripts/lab_run.sh              # full corpus (2,224 articles)
#   SAMPLE=300 bash scripts/lab_run.sh   # quick check on a small sample
#   PYTHON=python bash scripts/lab_run.sh  # if python3 is not the right command
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
SAMPLE=${SAMPLE:-0}
mkdir -p results

echo "== 1/5 Environment"
{
  echo "date: $(date)"
  echo "host: $(hostname)"
  echo "python: $("$PY" --version 2>&1)"
  uname -a 2>/dev/null || true
  echo "cpus: $(nproc 2>/dev/null || echo n/a)"
  free -h 2>/dev/null || true
  nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "gpu: n/a"
} | tee results/lab_environment.txt

echo "== 2/5 Installing requirements"
"$PY" -m pip install -q -r requirements.txt
"$PY" -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

echo "== 3/5 Sanity tests"
"$PY" -m tests.test_core | tee results/tests_output.txt

echo "== 4/5 Experiments (sample=$SAMPLE, 0 = full corpus)"
start=$(date +%s)
"$PY" scripts/run_experiments.py --sample "$SAMPLE" | tee results/experiments_log.txt
echo "experiments took $(( $(date +%s) - start )) s" | tee -a results/lab_environment.txt

echo "== 5/5 Packaging"
"$PY" -m zipfile -c lab_results.zip results/
echo
echo "Done. Download lab_results.zip BEFORE stopping the session (the pod is deleted on stop)."
echo "Start the app with:"
echo "  streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"
