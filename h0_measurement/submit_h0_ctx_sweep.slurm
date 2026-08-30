#!/bin/bash
#SBATCH --job-name=sieve-h0-ctxsweep
#SBATCH --output=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/logs/h0ctx_%A_%a.out
#SBATCH --error=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/logs/h0ctx_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-2                  # one task per CTX VALUE; match your ctx list
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jzzzzzzz23@foxmail.com

#
# ============================================================================
# submit_h0_ctx_sweep.slurm -- ONE model swept across CONTEXT LENGTHS
# (bugs/3_context_sweep_and_reports). Same two-stage structure as
# submit_h0.slurm, but the array axis is ctx, not model: task i runs
# `run_h0.py --model $MODEL --override ctx=${CTXS[i]}`.
#
# This measures the band-fraction-vs-ctx slope directly: eviction keeps a fixed
# FRACTION of tokens (B*L/maxb), so a head whose support is ~constant in absolute
# tokens gets relatively cheaper to evict as L grows -- report.py's ctx-slope
# page turns this run into that figure.
#
#   sbatch h0_measurement/submit_h0_ctx_sweep.slurm llama31-8b
#       # default sweep 32768 65536 131072 -- matches the header's --array=0-2
#   sbatch --array=0-5 h0_measurement/submit_h0_ctx_sweep.slurm \
#          llama31-8b 4096 8192 16384 32768 65536 131072
#   sbatch --gpus-per-node=4 --cpus-per-task=32 --time=02:30:00 --array=0-2 \
#          h0_measurement/submit_h0_ctx_sweep.slurm llama33-70b
#       # a tier-large model needs the large-tier resources on the command line;
#       # the header is sized for a 1-H100 model.
#
# Arg 1 is the model tag (REQUIRED); remaining args are ctx values. The --array
# range must match the number of ctx values (0-2 for 3 values); a mismatch is
# caught below before any GPU work. Every ctx must be <= the model's registry
# ctx -- those are native RoPE limits (README ctx audit), so among main-tier
# models only llama31-8b can sweep past 40960.
#
# HOW THE TWO STAGES LIVE IN ONE FILE
# Identical to submit_h0.slurm: the file re-submits itself once for the report
# and dispatches on SIEVE_ROLE. Array task 0 chains the report with
# --dependency=afterok on the whole array, because tasks finish in no guaranteed
# order (the 128k task is ~16x the 32k task) and afterok cannot fire early.
# Output parquet is h0_<tag>_<ctx>.parquet -- ctx is in the name, so the sweep's
# tasks share one OUT_DIR without collisions, and report.py groups by
# (model, ctx) so each ctx is its own experiment in the PDF.
# ============================================================================

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant}"
cd "$PROJECT_ROOT"

# MUST be this file: the report stage is a re-submission of SELF (see the note in
# submit_h0_large_models.slurm for why pointing at a sibling script goes wrong).
SELF="$PROJECT_ROOT/h0_measurement/submit_h0_ctx_sweep.slurm"
SIEVE_ROLE="${SIEVE_ROLE:-measure}"

# SIEVE_MODEL / SIEVE_CTXS travel through sbatch --export for the report
# resubmission; SIEVE_CTXS is COLON-separated for the same portability reason as
# SIEVE_MODELS in the sibling scripts.
if [[ -n "${SIEVE_MODEL:-}" ]]; then
  MODEL="$SIEVE_MODEL"
elif [[ $# -gt 0 ]]; then
  MODEL="$1"; shift
else
  echo "ERROR: no model tag. usage:" >&2
  echo "  sbatch [--array=0-N] $0 <model-tag> [ctx ctx ...]" >&2
  exit 1
fi
if [[ -n "${SIEVE_CTXS:-}" ]]; then
  IFS=':' read -r -a CTXS <<< "$SIEVE_CTXS"
elif [[ $# -gt 0 ]]; then
  CTXS=("$@")
else
  CTXS=(32768 65536 131072)   # the plan's llama31-8b sweep; matches --array=0-2
fi
CTXS_COLON=$(IFS=':'; echo "${CTXS[*]}")

# Fail on malformed or duplicate ctx values before touching the queue: a
# duplicate would make two tasks write the same h0_<tag>_<ctx>.parquet and the
# report's count check would then fail an otherwise-green array.
declare -A _seen=()
for c in "${CTXS[@]}"; do
  if [[ ! "$c" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ctx values must be positive integers, got '$c'" >&2
    exit 1
  fi
  if [[ -n "${_seen[$c]:-}" ]]; then
    echo "ERROR: duplicate ctx $c -- two tasks would write the same parquet" >&2
    exit 1
  fi
  _seen[$c]=1
done

# The header's --array is static but CTXS is not (same trap as the large script's
# MODELS): catch the mismatch before bash -u aborts with a GPU already allocated.
if [[ "$SIEVE_ROLE" == "measure" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if (( SLURM_ARRAY_TASK_ID >= ${#CTXS[@]} )); then
    echo "ERROR: array task $SLURM_ARRAY_TASK_ID but only ${#CTXS[@]} ctx value(s):" \
         "${CTXS[*]}" >&2
    echo "       re-submit with --array=0-$(( ${#CTXS[@]} - 1 ))" >&2
    exit 1
  fi
fi

VENV="${SIEVE_VENV:-$PROJECT_ROOT/.venv}"
# HF_TOKEN is read from an untracked file, never hard-coded here.
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$PROJECT_ROOT/.hf_token}"
if [[ ! -r "$HF_TOKEN_FILE" ]]; then
  echo "ERROR: HF token file not found or unreadable: $HF_TOKEN_FILE" >&2
  exit 1
fi
export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
# ONE knob for the cache: HF_HOME (see quick_test_all.slurm for the full note).
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"   # legacy name, same precedence trap
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Forced OFFLINE -- compute nodes have no internet; stage weights on a login node
# first (see the '[Errno 101]' note in submit_h0.slurm).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# The PG-19 haystack, staged on the login node. The sweep is ABOUT how attention
# over a real document changes with ctx, so filler would be doubly meaningless
# here: the filler artifact itself scales with ctx (README "synthetic-haystack
# confound"), i.e. it manufactures exactly the slope this script measures.
export H0_CORPUS="${H0_CORPUS:-$PROJECT_ROOT/.h0_corpus/pg19}"
export H0_ALLOW_SYNTHETIC="${H0_ALLOW_SYNTHETIC:-0}"   # =1 to run on filler anyway

source "$VENV/bin/activate"

LOG_DIR="$PROJECT_ROOT/h0_measurement/logs"
REPORT_DIR="$PROJECT_ROOT/h0_measurement/reports"
RUN_ID="${SIEVE_RUN_ID:-job${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"
OUT_DIR="$PROJECT_ROOT/h0_measurement/results/${RUN_ID}"
mkdir -p "$LOG_DIR" "$REPORT_DIR" "$OUT_DIR"

# ---------------------------------------------------------------- report stage
if [[ "$SIEVE_ROLE" == "report" ]]; then
  echo "host=$(hostname) role=report run_id=$RUN_ID model=$MODEL ctxs=${CTXS[*]}"

  # afterok already proves every task exited 0; this catches a task that exited 0
  # having written nothing. One parquet per CTX VALUE here, not per model.
  n_found=$(find "$OUT_DIR" -name '*.parquet' | wc -l | tr -d ' ')
  if [[ "$n_found" -ne "${#CTXS[@]}" ]]; then
    echo "ERROR: ${#CTXS[@]} ctx values submitted but $n_found parquet in $OUT_DIR" >&2
    find "$OUT_DIR" -name '*.parquet' >&2
    exit 1
  fi

  PDF="${REPORT_DIR}/h0_ctxsweep_${MODEL}_${RUN_ID}_$(date +%Y%m%d).pdf"
  python h0_measurement/report.py "${OUT_DIR}/*.parquet" -o "$PDF"
  echo
  echo "report: $PDF"
  echo "per-head csv: ${PDF%.pdf}_per_head.csv"
  echo
  echo "The 'Band fraction vs context length' page is the point of this run: its"
  echo "slope is the phi-window ctx-coupling from bugs/3_context_sweep_and_reports,"
  echo "measured instead of assumed."
  exit 0
fi

# --------------------------------------------------------------- measure stage
CTX="${CTXS[$SLURM_ARRAY_TASK_ID]}"
echo "host=$(hostname) role=measure model=$MODEL ctx=$CTX task=$SLURM_ARRAY_TASK_ID run_id=$RUN_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Validate tag and EVERY ctx against the registry before any staging: the
# registry ctx is a native RoPE limit (README ctx audit), so e.g. qwen3-8b at
# 65536 would run 1.6x past trained positions and the whole sweep's slope would
# include an out-of-distribution point. Checking the full list (not just this
# task's ctx) fails the array fast instead of leaving a partial sweep.
python - "$MODEL" "${CTXS[@]}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open("h0_measurement/models.yaml"))
m = next((x for x in cfg["models"] if x["tag"] == sys.argv[1]), None)
if m is None:
    sys.exit(f"ERROR: unknown model tag '{sys.argv[1]}'\n"
             f"available: {' '.join(sorted(x['tag'] for x in cfg['models']))}")
cap = int(m.get("ctx", cfg.get("defaults", {}).get("ctx", 0)))
bad = [c for c in sys.argv[2:] if int(c) > cap]
if bad:
    sys.exit(f"ERROR: ctx {' '.join(bad)} exceed {m['tag']}'s registry ctx {cap} "
             f"(a native limit, not advisory -- see README ctx audit). "
             f"Only llama31-8b, qwen3-30b-a3b-2507 and llama33-70b reach 131072.")
PY

# Corpus precondition, FATAL -- same reasoning as submit_h0.slurm, and --verify
# is offline (sha256 against MANIFEST.json).
if [[ "$H0_ALLOW_SYNTHETIC" == "1" ]]; then
  echo "WARNING: H0_ALLOW_SYNTHETIC=1 -- measuring on synthetic filler."
  echo "         report.py will stamp every verdict UNKNOWN, and the filler"
  echo "         artifact scales with ctx, i.e. it fakes the very slope this"
  echo "         sweep exists to measure."
elif ! python h0_measurement/prefetch_corpus.py --verify --out "$H0_CORPUS"; then
  {
    echo "ERROR: no usable PG-19 haystack at H0_CORPUS=$H0_CORPUS"
    echo "  stage it on the LOGIN node (this node has no internet):"
    echo "    python h0_measurement/prefetch_corpus.py --check-ctx 131072 --n-prompts 6"
  } >&2
  exit 1
fi

# Task 0 chains the report BEFORE the long run. Resource overrides undo whatever
# the sbatch line requested (this matters when the sweep was submitted with
# --gpus-per-node=4 for a large model -- the report is pure CPU).
if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 && "${SIEVE_NO_REPORT:-0}" != "1" ]]; then
  { date; echo "model: ${MODEL}"; echo "ctx sweep: ${CTXS[*]}";
    echo "array job: ${SLURM_ARRAY_JOB_ID}"; } \
    > "${OUT_DIR}/RUN_INFO.txt"
  REPORT_JOB=$(sbatch --parsable \
    --dependency="afterok:${SLURM_ARRAY_JOB_ID}" \
    --job-name=sieve-h0-ctxsweep-report \
    --output="${LOG_DIR}/h0ctx_report_%j.out" \
    --error="${LOG_DIR}/h0ctx_report_%j.err" \
    --array=0-0 \
    --gpus-per-node=0 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=00:30:00 \
    --export="ALL,SIEVE_ROLE=report,SIEVE_RUN_ID=${RUN_ID},SIEVE_MODEL=${MODEL},SIEVE_CTXS=${CTXS_COLON}" \
    "$SELF") \
    && echo "chained report job ${REPORT_JOB} (afterok:${SLURM_ARRAY_JOB_ID})" \
    || echo "WARNING: could not chain the report job; run report.py by hand" >&2
fi

# Local completeness CHECK of the weights + validate_with proxy (HF_HUB_OFFLINE=1,
# so this cannot download). Stage on a login node: prefetch.py -m "$MODEL"
python h0_measurement/prefetch.py -m "$MODEL"

# VRAM preflight at THIS TASK'S ctx -- generic (works at 1 or 4 GPUs), and cheap.
# device_map=auto would otherwise shard onto too little aggregate VRAM and OOM
# 20 minutes into the load.
python - "$MODEL" "$CTX" <<'PY'
import sys, yaml, torch
from transformers import AutoConfig
cfg = yaml.safe_load(open("h0_measurement/models.yaml"))
m = next(x for x in cfg["models"] if x["tag"] == sys.argv[1])
ctx = int(sys.argv[2])
c = AutoConfig.from_pretrained(m["id"], trust_remote_code=True)
kv = getattr(c, "num_key_value_heads", c.num_attention_heads)
hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
kv_gb = 2 * kv * hd * 2 * c.num_hidden_layers * ctx / 1e9
n = torch.cuda.device_count()
have = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 1e9
print(f"  gpus={n}  vram={have:.0f} GB  kv@{ctx}={kv_gb:.0f} GB")
if have < kv_gb * 1.15:
    sys.exit(f"ERROR: {have:.0f} GB VRAM across {n} GPU(s) cannot even hold the "
             f"{kv_gb:.0f} GB KV cache for {sys.argv[1]} at ctx {ctx}, before "
             f"weights. Re-submit with --gpus-per-node=4 (see the large-model "
             f"example in the header).")
PY

srun python h0_measurement/run_h0.py \
     --model "$MODEL" \
     --out-dir "$OUT_DIR" \
     --override ctx="$CTX"
