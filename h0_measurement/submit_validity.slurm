#!/bin/bash
#SBATCH --job-name=sieve-validity
#SBATCH --output=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/logs/validity_%A_%a.out
#SBATCH --error=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/logs/validity_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jzzzzzzz23@foxmail.com

#
# ============================================================================
# submit_validity.slurm -- the INPUT-validity probe.
#
# Asks one question: did the haystack induce retrieval? It prefills the same
# prompts a full run would, decodes greedily, and records (a) whether the model
# emits the needle code and (b) how much attention mass each head puts on the
# needle span. No quantization sweep, no value tensors, niah only -- minutes per
# model instead of hours.
#
# It is legitimate to validate an EXISTING measurement run with this, because
# the haystack is seeded on prompt_idx alone: at the same ctx this reads
# byte-identical text to the run it is validating.
#
# NO EVICTION-CORNER KNOBS HERE, deliberately. SIEVE_EVICTORS /
# SIEVE_CORNER_POLICIES exist in the three measurement scripts
# (bugs/2_towards_real_evictor) and are meaningless in this one: --validity-only
# forms no corners at all -- it never materialises V, never runs the bit sweep,
# and never keeps lagged attention state. run_h0.py therefore prints
# "eviction corners: (none)" here and writes an empty `evictors` column, so a
# validity parquet cannot be mistaken for a run that measured a corner. It also
# means this script needs none of the extra --mem or walltime the measurement
# scripts took on: the ~28 B per layer-head-token of lagged state is not
# allocated, so the header below is unchanged.
#
# SUBMIT FROM trig-login01. This cluster rejects any GPU request made from a CPU
# login node ("GPU resources requested from a CPU login node"), so sbatch fails
# here with an error that does not mention the script. Same for submit_h0*.slurm.
#
#   sbatch h0_measurement/submit_validity.slurm                       # 5 models
#   sbatch --array=0-0 h0_measurement/submit_validity.slurm qwen3-8b
#   # large models do not fit one GPU -- the CLI beats the header:
#   sbatch --array=0-1 --gpus-per-node=4 --time=01:30:00 \
#          h0_measurement/submit_validity.slurm llama33-70b qwen3-30b-a3b-2507
#
# Output: results/<RUN_ID>/validity_<tag>_<ctx>.parquet. Feed it to report.py
# ALONGSIDE the measurement parquet -- report.py keeps the two frames separate
# and uses the probe only for the gate:
#
#   python h0_measurement/report.py \
#          "h0_measurement/results/job852849/*.parquet" \
#          "h0_measurement/results/job852851/*.parquet" \
#          "h0_measurement/results/<VALIDITY_RUN_ID>/*.parquet" \
#          -o h0_measurement/reports/h0_report_gated.pdf
# ============================================================================

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant}"
cd "$PROJECT_ROOT"

if [[ -n "${SIEVE_MODELS:-}" ]]; then
  IFS=':' read -r -a MODELS <<< "$SIEVE_MODELS"
elif [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(qwen3-8b llama31-8b mistral-7b qwen15-moe-a2.7b qwen3-30b-a3b-2507)
fi

VENV="${SIEVE_VENV:-$PROJECT_ROOT/.venv}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$PROJECT_ROOT/.hf_token}"
if [[ ! -r "$HF_TOKEN_FILE" ]]; then
  echo "ERROR: HF token file not found or unreadable: $HF_TOKEN_FILE" >&2
  exit 1
fi
export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export H0_CORPUS="${H0_CORPUS:-$PROJECT_ROOT/.h0_corpus/pg19}"

source "$VENV/bin/activate"

LOG_DIR="$PROJECT_ROOT/h0_measurement/logs"
RUN_ID="${SIEVE_RUN_ID:-validity${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"
OUT_DIR="$PROJECT_ROOT/h0_measurement/results/${RUN_ID}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "host=$(hostname) role=validity model=$MODEL task=$SLURM_ARRAY_TASK_ID run_id=$RUN_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The probe measures the INPUT, so an unusable corpus makes it meaningless --
# fatal here, exactly as in the measurement scripts.
if ! python h0_measurement/prefetch_corpus.py --verify --out "$H0_CORPUS"; then
  echo "ERROR: no usable PG-19 haystack at H0_CORPUS=$H0_CORPUS" >&2
  echo "  stage it on the LOGIN node:" >&2
  echo "    python h0_measurement/prefetch_corpus.py --check-ctx 131072" >&2
  exit 1
fi

python h0_measurement/prefetch.py -m "$MODEL"

# --skip-external-check: L3 reloads the proxy to compare against HF eager, which
# is about the probe's quantization math. This run does no quantization.
srun python h0_measurement/run_h0.py \
     --model "$MODEL" \
     --validity-only \
     --skip-external-check \
     --out-dir "$OUT_DIR"

echo
echo "Read the TASK-LEVEL / HEAD-LEVEL lines above. MIN_MASS in"
echo "sievelib/validity.py is uncalibrated and advisory: set it from the"
echo "reported per-head distribution, then flip ENFORCE_HEAD_LEVEL."
