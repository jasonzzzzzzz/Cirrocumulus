#!/bin/bash
#SBATCH --job-name=sieve-h0-large
#SBATCH --output=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/logs/h0large_%A_%a.out
#SBATCH --error=/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/logs/h0large_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-1                  # one task per model; override for a subset
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jzzzzzzz23@foxmail.com

#
# ============================================================================
# submit_h0_large_models.slurm -- the tier-"large" H0 campaign. Same two-stage
# structure as submit_h0.slurm (measure array + chained report); the ONLY thing
# that differs is resources, because these two models do not fit one H100.
#
#   python h0_measurement/prefetch_corpus.py         # ONCE -- the PG-19 haystack,
#                                                    # LOGIN node (no internet here)
#   sbatch h0_measurement/submit_h0_large_models.slurm
#   sbatch --array=0-0 h0_measurement/submit_h0_large_models.slurm llama33-70b
#
# WHY 4 GPUs (measured from each model's live config, at ctx 131072, bf16):
#
#   llama33-70b          80L, 64Q/8KV heads, head_dim 128
#     weights  70.6B x 2B                        = 141 GB
#     KV cache 2*8*128*2B * 80L * 131072 tok     =  43 GB
#     total                                      = 184 GB  -> 3 cards minimum,
#     4 for activations + the quantization sweep's working copies.
#
#   qwen3-30b-a3b-2507   48L, 32Q/4KV heads, head_dim 128 (MoE: all 30.5B
#                        params are resident even though only 3B activate)
#     weights  30.5B x 2B                        =  61 GB
#     KV cache 2*4*128*2B * 48L * 131072 tok     =  13 GB
#     total                                      =  74 GB  -> will NOT hold on
#     one 80 GB card once activations land; 2 cards is the real floor.
#
# One #SBATCH header covers every array task, so both tasks get 4 GPUs and the
# MoE task leaves ~2 idle. That is deliberate -- one queue entry beats two, and
# device_map=auto shards to whatever it is given. If you would rather not park
# 2 idle H100s, submit them separately instead:
#   sbatch --array=0-0 --gpus-per-node=4 submit_h0_large_models.slurm llama33-70b
#   sbatch --array=0-0 --gpus-per-node=2 submit_h0_large_models.slurm qwen3-30b-a3b-2507
#
# WEIGHTS ARE NOT STAGED BY THIS JOB. HF_HUB_OFFLINE=1 below (compute nodes have
# no internet), so the in-job prefetch is only a local completeness CHECK that
# fails in seconds. Stage ~202 GB on a LOGIN node first, or every task dies:
#   python h0_measurement/prefetch.py -m llama33-70b qwen3-30b-a3b-2507
#
# HOW THE TWO STAGES LIVE IN ONE FILE
# The file re-submits itself once, for the report, and dispatches on SIEVE_ROLE:
#   measure (default)  one array task per model -> h0_measurement/results/<RUN_ID>/*.parquet
#   report             CPU-only, runs ONLY after every task succeeds
# Array task 0 submits the report stage with --dependency=afterok on the whole
# array. That dependency is the point: array tasks are scheduled independently
# and in NO guaranteed order -- the MoE task usually finishes well before the
# 70B one -- so an in-array "am I the last index?" test would
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

# MUST be this file, not submit_h0.slurm. The report stage is a re-submission of
# SELF, so pointing at the other script would queue the small-model file to write
# this run's report -- it would inherit SIEVE_MODELS and mostly work, then silently
# diverge the moment the two files' report stages differ.
SELF="$PROJECT_ROOT/h0_measurement/submit_h0_large_models.slurm"
SIEVE_ROLE="${SIEVE_ROLE:-measure}"

# SIEVE_MODELS is COLON-separated on purpose. It travels through sbatch --export,
# whose value list is comma-separated and whose handling of embedded spaces is
# not portable -- a space-separated list there can arrive truncated.
if [[ -n "${SIEVE_MODELS:-}" ]]; then
  IFS=':' read -r -a MODELS <<< "$SIEVE_MODELS"
elif [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  # tier: large in models.yaml. Keep in sync with --array above (0-1 = 2 models).
  MODELS=(qwen3-30b-a3b-2507 llama33-70b)
fi

# The header's --array is static but MODELS is not, so a subset invocation that
# forgets to override --array would index past the end. Bash -u turns that into
# an unbound-variable abort with no explanation, 4 H100s already allocated.
if [[ "${SIEVE_ROLE:-measure}" == "measure" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if (( SLURM_ARRAY_TASK_ID >= ${#MODELS[@]} )); then
    echo "ERROR: array task $SLURM_ARRAY_TASK_ID but only ${#MODELS[@]} model(s):" \
         "${MODELS[*]}" >&2
    echo "       re-submit with --array=0-$(( ${#MODELS[@]} - 1 ))" >&2
    exit 1
  fi
fi
MODELS_COLON=$(IFS=':'; echo "${MODELS[*]}")

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
# a long document. Both models here are tier large, so run_h0.py refuses without
# it. This tier is also where filler hurt most: llama3.3-70B smeared (63% of
# heads near-uniform) while qwen3-30B collapsed onto sinks, in opposite
# directions, which is exactly the "bigger models benefit less" trend the old
# campaign reported.
#   python h0_measurement/prefetch_corpus.py --check-ctx 131072 --n-prompts 4
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
  echo "host=$(hostname) role=report run_id=$RUN_ID models=${MODELS[*]}"

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
echo "host=$(hostname) role=measure model=$MODEL task=$SLURM_ARRAY_TASK_ID run_id=$RUN_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# prefetch.py skips tags it does not recognise and still exits 0, so a typo would
# stage nothing and surface only as a confusing failure later. Reject it here.
python - "$MODEL" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open("h0_measurement/models.yaml"))
known = {m["tag"] for m in cfg["models"]}
if sys.argv[1] not in known:
    sys.exit(f"ERROR: unknown model tag '{sys.argv[1]}'\n"
             f"available: {' '.join(sorted(known))}")
PY

# Corpus precondition, FATAL. Both models here are tier large, which run_h0.py
# refuses to run on filler, so a missing corpus is a certain failure -- and the
# thing it would waste is a 4xH100 allocation. Stop before that is requested, and
# before a report job is chained to an array that cannot succeed. --verify is
# offline (sha256 against MANIFEST.json) and also catches a truncated or
# half-copied corpus, which run_h0.py's preflight would happily accept.
if [[ "$H0_ALLOW_SYNTHETIC" == "1" ]]; then
  echo "WARNING: H0_ALLOW_SYNTHETIC=1 -- measuring on synthetic filler."
  echo "         report.py will stamp every verdict UNKNOWN."
elif ! python h0_measurement/prefetch_corpus.py --verify --out "$H0_CORPUS"; then
  {
    echo "ERROR: no usable PG-19 haystack at H0_CORPUS=$H0_CORPUS"
    echo "  stage it on the LOGIN node (this node has no internet):"
    echo "    python h0_measurement/prefetch_corpus.py --check-ctx 131072 --n-prompts 4"
    echo "  or resubmit with H0_ALLOW_SYNTHETIC=1 to measure filler anyway"
    echo "  (results are then stamped UNKNOWN -- see"
    echo "   h0_measurement/bugs/1_from_synthetic_to_real_corpus/)."
  } >&2
  exit 1
fi

# Task 0 chains the report BEFORE the long run, so it is queued even if task 0 is
# the first to finish. Resource overrides on the command line beat the #SBATCH
# header above -- and here they MUST, because that header now asks for 4 H100s.
# The report is pure CPU (pandas + matplotlib); inheriting --gpus-per-node=4
# would park four idle H100s for 30 minutes and put the report behind the whole
# 4-GPU queue. --gpus-per-node=0 and the small cpu/mem values undo the header.
if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 && "${SIEVE_NO_REPORT:-0}" != "1" ]]; then
  { date; echo "models: ${MODELS[*]}"; echo "array job: ${SLURM_ARRAY_JOB_ID}"; } \
    > "${OUT_DIR}/RUN_INFO.txt"
  REPORT_JOB=$(sbatch --parsable \
    --dependency="afterok:${SLURM_ARRAY_JOB_ID}" \
    --job-name=sieve-h0-report \
    --output="${LOG_DIR}/h0_report_%j.out" \
    --error="${LOG_DIR}/h0_report_%j.err" \
    --array=0-0 \
    --gpus-per-node=0 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=00:30:00 \
    --export="ALL,SIEVE_ROLE=report,SIEVE_RUN_ID=${RUN_ID},SIEVE_MODELS=${MODELS_COLON}" \
    "$SELF") \
    && echo "chained report job ${REPORT_JOB} (afterok:${SLURM_ARRAY_JOB_ID})" \
    || echo "WARNING: could not chain the report job; run report.py by hand" >&2
fi

# NOT a download -- HF_HUB_OFFLINE=1 above. This is a local completeness CHECK of
# this model's weights AND its validate_with proxy (run_h0.py loads the proxy for
# L1/L3 probe validation before it touches the real model). It costs seconds and
# fails the task immediately if staging was missed, instead of after a 141 GB
# load. Stage on a login node: python h0_measurement/prefetch.py -m "$MODEL"
python h0_measurement/prefetch.py -m "$MODEL"

# Fail before the load, not 20 minutes into it. device_map=auto will happily start
# sharding onto too little aggregate VRAM and OOM partway through.
python - "$MODEL" <<'PY'
import sys, yaml, torch
from transformers import AutoConfig
cfg = yaml.safe_load(open("h0_measurement/models.yaml"))
d = cfg["defaults"]; m = next(x for x in cfg["models"] if x["tag"] == sys.argv[1])
ctx = int(m.get("ctx", d["ctx"]))
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
             f"weights. Raise --gpus-per-node.")
PY

srun python h0_measurement/run_h0.py \
     --model "$MODEL" \
     --out-dir "$OUT_DIR"

