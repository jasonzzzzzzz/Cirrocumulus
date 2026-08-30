#!/bin/bash
#SBATCH --job-name=sieve-h0
#SBATCH --output=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/logs/h0_%A_%a.out
#SBATCH --error=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/logs/h0_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-3                  # one task per model; override for a subset
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jzzzzzzz23@foxmail.com

#
# ============================================================================
# submit_h0.slurm -- the ENTIRE H0 campaign in one file: measurement array plus
# the report that aggregates it. One `sbatch`, nothing else to run.
#
# --chdir above anchors the job to PROJECT_ROOT, so the relative --output/--error
# paths and everything in the body land there no matter where you run sbatch from.
# Edit --chdir and PROJECT_ROOT together if the repo ever moves.
#
#   mkdir -p h0_measurement/{logs,results,reports}   # ONCE -- sbatch cannot create it
#   python h0_measurement/prefetch_corpus.py         # ONCE -- the PG-19 haystack,
#                                                    # LOGIN node (no internet here)
#   sbatch h0_measurement/submit_h0.slurm
#   sbatch h0_measurement/submit_h0.slurm qwen3-8b mistral-7b    # a subset
#   SIEVE_CTX=32768 sbatch h0_measurement/submit_h0.slurm        # matched-ctx campaign
#
# The --array size above must match the number of models. If you pass a subset,
# override it:  sbatch --array=0-1 h0_measurement/submit_h0.slurm a b
#
# HOW THE TWO STAGES LIVE IN ONE FILE
# The file re-submits itself once, for the report, and dispatches on SIEVE_ROLE:
#   measure (default)  one array task per model -> h0_measurement/results/<RUN_ID>/*.parquet
#   report             CPU-only, runs ONLY after every task succeeds
# Array task 0 submits the report stage with --dependency=afterok on the whole
# array. That dependency is the point: array tasks are scheduled independently
# and in NO guaranteed order -- mistral-7b (ctx 32768) usually finishes before
# the two 131072-ctx tasks -- so an in-array "am I the last index?" test would
# fire while sibling parquet files do not exist yet and silently emit a partial
# report. afterok cannot fire early.
#
# Everything is scoped by RUN_ID (= the array job id, identical across sibling
# tasks by construction), so a rerun can never mix into an earlier run's glob.
# ============================================================================

set -euo pipefail

# Anchor everything to one absolute root. --chdir in the header points SLURM at
# the same path so the log files above land inside it too; this cd covers the
# case where the script is run directly, without sbatch.
export PROJECT_ROOT="${PROJECT_ROOT:-/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant}"
cd "$PROJECT_ROOT"

SELF="$PROJECT_ROOT/h0_measurement/submit_h0.slurm"
SIEVE_ROLE="${SIEVE_ROLE:-measure}"

# SIEVE_MODELS is COLON-separated on purpose. It travels through sbatch --export,
# whose value list is comma-separated and whose handling of embedded spaces is
# not portable -- a space-separated list there can arrive truncated.
if [[ -n "${SIEVE_MODELS:-}" ]]; then
  IFS=':' read -r -a MODELS <<< "$SIEVE_MODELS"
elif [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(qwen3-8b llama31-8b mistral-7b qwen15-moe-a2.7b)
fi
MODELS_COLON=$(IFS=':'; echo "${MODELS[*]}")

# SIEVE_CTX: optional MATCHED-CONTEXT override (bugs/3_context_sweep_and_reports).
# When set, every model runs at this ctx via `--override ctx=` instead of its
# per-model registry value, so the cross-model table compares architectures at one
# context length instead of confounding model size with ctx. 32768 is the largest
# value every main-tier model supports natively (qwen3-8b caps at 40960,
# mistral-7b and qwen15-moe at 32768 -- see README "context length methodology").
# Values above a model's registry ctx are rejected per task below: the registry
# caps are native limits, not advisory. Unset = today's behavior, per-model ctx.
SIEVE_CTX="${SIEVE_CTX:-}"
if [[ -n "$SIEVE_CTX" && ! "$SIEVE_CTX" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SIEVE_CTX must be a positive integer, got '$SIEVE_CTX'" >&2
  exit 1
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
# HF_HUB_CACHE must stay $HF_HOME/hub so the weights submit_h0.sh staged are the
# ones this job reads -- a mismatch here re-downloads 44G inside the allocation.
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"   # legacy name, same precedence trap
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Forced OFFLINE. These compute nodes have NO outbound internet -- with this at 0
# every cache HIT still paid a HEAD request that died as
#   '[Errno 101] Network is unreachable' ... Retrying in 1s [Retry 1/5]
# and burned GPU time through five backoff rounds per file (see the Qwen3-0.6B
# custom_generate/generate.py stall in logs/h0_845890_0.err). The cache cannot
# self-heal here regardless, so fail fast instead: stage on the LOGIN node with
#   python h0_measurement/prefetch.py -m <tags>
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# The haystack: PG-19 books, staged on the login node like the weights and for
# the same reason (no internet here). This is the INPUT the campaign is about --
# unset means synthetic filler, i.e. 8 sentences tiled ~970x at ctx 131072, which
# measures how a model degenerates on repetition rather than how it attends over
# a long document. run_h0.py refuses tier main/large without it.
#   python h0_measurement/prefetch_corpus.py --check-ctx 131072 --n-prompts 6
export H0_CORPUS="${H0_CORPUS:-$PROJECT_ROOT/.h0_corpus/pg19}"
export H0_ALLOW_SYNTHETIC="${H0_ALLOW_SYNTHETIC:-0}"   # =1 to run on filler anyway

source "$VENV/bin/activate"

# Absolute, to match the header's --output/--error. The chained report job below
# reuses this, so its logs land beside the array's instead of wherever sbatch ran.
LOG_DIR="$PROJECT_ROOT/h0_measurement/logs"
REPORT_DIR="$PROJECT_ROOT/h0_measurement/reports"
RUN_ID="${SIEVE_RUN_ID:-job${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"
OUT_DIR="$PROJECT_ROOT/h0_measurement/results/${RUN_ID}"
mkdir -p "$LOG_DIR" "$REPORT_DIR" "$OUT_DIR"

# ---------------------------------------------------------------- report stage
if [[ "$SIEVE_ROLE" == "report" ]]; then
  echo "host=$(hostname) role=report run_id=$RUN_ID models=${MODELS[*]}" \
       "ctx=${SIEVE_CTX:-per-model}"

  # afterok already proves every task exited 0, but a task could exit 0 having
  # written nothing if run_h0.py changes later. Fail loudly rather than publish a
  # report that is quietly missing a model.
  n_found=$(find "$OUT_DIR" -name '*.parquet' | wc -l | tr -d ' ')
  if [[ "$n_found" -ne "${#MODELS[@]}" ]]; then
    echo "ERROR: ${#MODELS[@]} models submitted but $n_found parquet in $OUT_DIR" >&2
    find "$OUT_DIR" -name '*.parquet' >&2
    exit 1
  fi

  PDF="${REPORT_DIR}/h0_report_${RUN_ID}_$(date +%Y%m%d).pdf"
  python h0_measurement/report.py "${OUT_DIR}/*.parquet" -o "$PDF"
  echo
  echo "report: $PDF"
  echo "per-head csv: ${PDF%.pdf}_per_head.csv"
  echo
  echo "A verdict of UNKNOWN means the input-validity gate failed: the haystack"
  echo "was synthetic, or niah's sensitivity ladder did not beat cont's, so no"
  echo "retrieval behaviour was induced and the band fraction describes the"
  echo "prompt rather than the model."
  exit 0
fi

# --------------------------------------------------------------- measure stage
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "host=$(hostname) role=measure model=$MODEL task=$SLURM_ARRAY_TASK_ID run_id=$RUN_ID ctx=${SIEVE_CTX:-per-model}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# prefetch.py skips tags it does not recognise and still exits 0, so a typo would
# stage nothing and surface only as a confusing failure later. Reject it here.
# Also reject a SIEVE_CTX above this model's registry ctx: the registry values
# are native RoPE limits (README ctx audit), so running past them measures
# untrained positions and the parquet would poison the matched-ctx table.
python - "$MODEL" <<'PY'
import os, sys, yaml
cfg = yaml.safe_load(open("h0_measurement/models.yaml"))
m = next((x for x in cfg["models"] if x["tag"] == sys.argv[1]), None)
if m is None:
    sys.exit(f"ERROR: unknown model tag '{sys.argv[1]}'\n"
             f"available: {' '.join(sorted(x['tag'] for x in cfg['models']))}")
ctx = os.environ.get("SIEVE_CTX")
if ctx:
    cap = int(m.get("ctx", cfg.get("defaults", {}).get("ctx", 0)))
    if int(ctx) > cap:
        sys.exit(f"ERROR: SIEVE_CTX={ctx} exceeds {m['tag']}'s registry ctx {cap} "
                 f"(a native limit, not advisory -- see README ctx audit). "
                 f"Drop this model from the list or lower SIEVE_CTX.")
PY

# Corpus precondition, FATAL. Every model in the default list is tier main, which
# run_h0.py refuses to run on filler, so a missing corpus is a certain failure --
# stop here rather than chain a report job to an array that cannot succeed.
# --verify is offline (sha256 against MANIFEST.json) and also catches a truncated
# or half-copied corpus, which run_h0.py's preflight would happily accept.
if [[ "$H0_ALLOW_SYNTHETIC" == "1" ]]; then
  echo "WARNING: H0_ALLOW_SYNTHETIC=1 -- measuring on synthetic filler."
  echo "         report.py will stamp every verdict UNKNOWN."
elif ! python h0_measurement/prefetch_corpus.py --verify --out "$H0_CORPUS"; then
  {
    echo "ERROR: no usable PG-19 haystack at H0_CORPUS=$H0_CORPUS"
    echo "  stage it on the LOGIN node (this node has no internet):"
    echo "    python h0_measurement/prefetch_corpus.py --check-ctx 131072 --n-prompts 6"
    echo "  or resubmit with H0_ALLOW_SYNTHETIC=1 to measure filler anyway"
    echo "  (results are then stamped UNKNOWN -- see"
    echo "   h0_measurement/bugs/1_from_synthetic_to_real_corpus/)."
  } >&2
  exit 1
fi

# Task 0 chains the report BEFORE the long run, so it is queued even if task 0 is
# the first to finish. Resource overrides on the command line beat the #SBATCH
# header above: no GPU, single task, short walltime.
if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 && "${SIEVE_NO_REPORT:-0}" != "1" ]]; then
  { date; echo "models: ${MODELS[*]}"; echo "array job: ${SLURM_ARRAY_JOB_ID}";
    echo "ctx override: ${SIEVE_CTX:-none (per-model registry ctx)}"; } \
    > "${OUT_DIR}/RUN_INFO.txt"
  REPORT_JOB=$(sbatch --parsable \
    --dependency="afterok:${SLURM_ARRAY_JOB_ID}" \
    --job-name=sieve-h0-report \
    --output="${LOG_DIR}/h0_report_%j.out" \
    --error="${LOG_DIR}/h0_report_%j.err" \
    --array=0-0 \
    --time=00:30:00 \
    --export="ALL,SIEVE_ROLE=report,SIEVE_RUN_ID=${RUN_ID},SIEVE_MODELS=${MODELS_COLON},SIEVE_CTX=${SIEVE_CTX}" \
    "$SELF") \
    && echo "chained report job ${REPORT_JOB} (afterok:${SLURM_ARRAY_JOB_ID})" \
    || echo "WARNING: could not chain the report job; run report.py by hand" >&2
fi

# Stage this model's weights AND its validate_with proxy. run_h0.py loads the
# proxy for L1/L3 probe validation before it touches the real model, so a missing
# proxy would kill the job at validation with the GPU already allocated.
python h0_measurement/prefetch.py -m "$MODEL"

# ${SIEVE_CTX:+...} appends the override only when SIEVE_CTX is set, so the
# unset case is byte-identical to the pre-bug-3 invocation.
srun python h0_measurement/run_h0.py \
     --model "$MODEL" \
     --out-dir "$OUT_DIR" \
     ${SIEVE_CTX:+--override ctx=$SIEVE_CTX}

