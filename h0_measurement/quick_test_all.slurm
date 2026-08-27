#!/bin/bash
#SBATCH --job-name=sieve-h0-quicktest
#SBATCH --output=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/logs/quicktest_%j.out
#SBATCH --error=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/logs/quicktest_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jzzzzzzz23@foxmail.com





# no --gres: this is CPU-only, on purpose -- do not queue it on the GPU partition.

#
# quick_test_all.slurm -- CPU cluster. Runs prefetch.py + quick_test.sh for every
# model you intend to hand to submit_h0.sh, one at a time. Catches a broken
# tokenizer cache, a missing/gated validate_with proxy, or a probe/transformers
# incompatibility per model, in minutes on CPU -- instead of at hour 3 inside a
# paid H100 allocation.
#
#   sbatch h0_measurement/quick_test_all.slurm
#   sbatch h0_measurement/quick_test_all.slurm qwen3-8b mistral-7b   # a subset
#
# Keep the default list below in sync with submit_h0.sh's default -- this should
# be the same set of models you plan to submit next.

set -uo pipefail   # deliberately NOT -e: one model's failure must not skip the rest

# Hardcoded rather than derived from SLURM_SUBMIT_DIR/$PWD: those reflect wherever
# `sbatch` was invoked from, which breaks silently (wrong h0_measurement/models.yaml
# lookup) if you submit from inside h0_measurement/ instead of the repo root.
export PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
cd "$PROJECT_ROOT"

MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(qwen3-8b llama31-8b mistral-7b)
fi

# Logs live beside this script, in h0_measurement/logs -- NOT the repo-root logs/,
# which is where a bare relative "logs/..." would land after the cd above. Must
# match the absolute #SBATCH --output/--error paths at the top of this file.
LOG_DIR="$PROJECT_ROOT/h0_measurement/logs"
mkdir -p "$LOG_DIR" results_smoke reports
# HF_TOKEN is read from an untracked file, never hard-coded here.
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$PROJECT_ROOT/.hf_token}"
if [[ ! -r "$HF_TOKEN_FILE" ]]; then
  echo "ERROR: HF token file not found or unreadable: $HF_TOKEN_FILE" >&2
  exit 1
fi
export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
# ONE knob for the cache: HF_HOME. Override it to relocate everything at once.
# HF_HUB_CACHE is always $HF_HOME/hub -- the exact path huggingface_hub derives
# on its own -- so a script that sets only HF_HOME still lands in this same
# cache instead of silently re-downloading. Precedence is
# HF_HUB_CACHE > HUGGINGFACE_HUB_CACHE > $HF_HOME/hub; all three are pinned to
# one directory here so no caller can half-redirect the weights.
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"   # legacy name, same precedence trap
export TOKENIZERS_PARALLELISM=false
source .venv/bin/activate

# Fail fast on a typo'd tag rather than discover it after the first model's
# prefetch has already started. Checked explicitly (not via `set -e`, which is
# off below on purpose) so this exit code is never swallowed.
python - "${MODELS[@]}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open("h0_measurement/models.yaml"))
known = {m["tag"] for m in cfg["models"]}
bad = [t for t in sys.argv[1:] if t not in known]
if bad:
    sys.exit(f"ERROR: unknown model tag(s): {' '.join(bad)}\n"
             f"available: {' '.join(sorted(known))}")
PY
if [[ $? -ne 0 ]]; then
  exit 1
fi

echo "quick-testing: ${MODELS[*]}"
RESULTS=""   # "model:PASS" / "model:FAIL" lines -- avoids bash 4-only assoc arrays
STAMP="${SLURM_JOB_ID:-local}"

for MODEL in "${MODELS[@]}"; do
  echo
  echo "================================================================"
  echo "=== $MODEL"
  echo "================================================================"

  LOG="$LOG_DIR/quicktest_${STAMP}_${MODEL}.log"
  {
    echo "--- prefetch (weights + validate_with proxy) ---"
    python h0_measurement/prefetch.py -m "$MODEL" \
      && echo "--- quick_test.sh ---" \
      && ./h0_measurement/quick_test.sh "$MODEL"
  } > "$LOG" 2>&1
  rc=$?

  if [[ $rc -eq 0 ]]; then
    RESULTS="${RESULTS}${MODEL}:PASS"$'\n'
    echo "PASS  $MODEL   (log: $LOG)"
  else
    RESULTS="${RESULTS}${MODEL}:FAIL"$'\n'
    echo "FAIL  $MODEL   (log: $LOG) -- last 20 lines:"
    tail -20 "$LOG"
  fi
done

# reports/smoke.pdf and results_smoke/ are shared scratch across every model in
# this loop (that's quick_test.sh's own design, unchanged here) -- smoke.pdf ends
# up covering whichever models most recently reached its report step. It is a
# throwaway sanity artifact, not the real measurement; the real, run-scoped
# results are what submit_h0.sh/report.slurm produce under results/<RUN_ID>/.

echo
echo "================ summary ================"
fail=0
while IFS=: read -r model status; do
  [[ -z "$model" ]] && continue
  printf "  %-20s %s\n" "$model" "$status"
  [[ "$status" == "FAIL" ]] && fail=1
done <<< "$RESULTS"

if [[ $fail -eq 1 ]]; then
  echo
  echo "one or more models FAILED the smoke test -- do not run submit_h0.sh yet."
  exit 1
fi
echo
echo "all models passed. safe to run: ./h0_measurement/submit_h0.sh ${MODELS[*]}"
