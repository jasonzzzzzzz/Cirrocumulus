#!/usr/bin/env bash
# quick_test.sh -- LOGIN NODE smoke test. Target: < 10 minutes, small model.
# Proves the whole path works before you burn a SLURM allocation.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

MODEL="${1:-qwen3-1.7b}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"   # legacy name, same precedence trap
export TOKENIZERS_PARALLELISM=false
# The debug tier is exempt from the corpus gate so this script still runs on a
# fresh checkout, but if a corpus IS staged we exercise the real path here rather
# than discovering a corpus bug on an H100.
export H0_CORPUS="${H0_CORPUS:-$PWD/.h0_corpus/pg19}"

echo "=== 0. environment ==============================================="
python -c "import torch,transformers;print('torch',torch.__version__,
'| transformers',transformers.__version__,
'| cuda',torch.cuda.is_available(),
'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python - <<'PY'
import transformers
from packaging.version import parse
assert parse(transformers.__version__) >= parse("4.48"), \
    "need transformers>=4.48 for ALL_ATTENTION_FUNCTIONS"
print("transformers version OK")
PY

echo; echo "=== 1. stage weights + haystack (login node has internet) ========"
python h0_measurement/prefetch.py -m "$MODEL"
if [[ -d "$H0_CORPUS" ]]; then
  python h0_measurement/prefetch_corpus.py --verify --out "$H0_CORPUS"
else
  echo "no corpus at $H0_CORPUS -- debug tier will use synthetic filler."
  echo "before any main/large run:  python h0_measurement/prefetch_corpus.py"
fi

echo; echo "=== 2. unit tests (no GPU needed) ================================"
python tests/test_units.py

echo; echo "=== 3. all three validation levels =============================="
python h0_measurement/run_h0.py --model "$MODEL" --validate-only

echo; echo "=== 4. tiny end-to-end run + PDF ================================="
python h0_measurement/run_h0.py --model "$MODEL" \
       --out-dir results_smoke --override n_prompts=1 n_decode=1 ctx=4096
python h0_measurement/report.py "results_smoke/*.parquet" -o reports/smoke.pdf

echo; echo "=== PASS. reports/smoke.pdf written. Safe to submit SLURM. ======="