#!/bin/bash
#SBATCH --job-name=sieve-r8
#SBATCH --output=h0_measurement/logs/r8_%j.out
#SBATCH --error=h0_measurement/logs/r8_%j.err
#SBATCH --nodes=1
##SBATCH --mem=96G
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#
# ============================================================================
# submit_r8.slurm -- R8: end-task accuracy under a compressed KV cache.
# Runs h0_measurement/run_r8.py for ONE (model, ctx) cell.
#
#   sbatch h0_measurement/submit_r8.slurm R8_MODEL=llama31-8b R8_CTX=32768 \
#          R8_ARMS=fp,uniform,evict R8_BUDGETS=1,2,3,4 R8_N_PROMPTS=20
#
# A NEW file, not a mode of submit_h0.slurm: co-design wave 4 runs through that
# script, and R8 needs none of its measurement machinery. The environment setup
# below is submit_h0.slurm's, repeated deliberately.
#
# OVERRIDES ARE ARGUMENTS. This cluster's `sbatch` is a shell function that adds
# --export=NONE, so `R8_CTX=... sbatch` never reaches the job. Every R8_*=value
# argument after the script path is exported here instead.
#
# PATHS. The #SBATCH log paths are RELATIVE, i.e. to the directory sbatch was
# called from, and PROJECT_ROOT defaults to that same directory -- so this file
# works wherever the checkout lives, as long as it is submitted from the project
# root (bugs/8_router_endtask/script.sh cds there first). mkdir the log dir once.
# ============================================================================
set -euo pipefail

for _a in "$@"; do
  if [[ "$_a" =~ ^R8_[A-Z0-9_]+= ]]; then export "$_a"
  else echo "ERROR: unexpected argument '$_a' (use R8_NAME=value)" >&2; exit 1; fi
done

export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
[[ -f h0_measurement/run_r8.py ]] || {
  echo "ERROR: $PROJECT_ROOT is not the project root (no h0_measurement/run_r8.py)." >&2
  echo "       Submit from the project root." >&2; exit 1; }

: "${R8_MODEL:?R8_MODEL=<tag> is required}"
: "${R8_CTX:?R8_CTX=<tokens> is required}"
R8_TASKS="${R8_TASKS:-niah_single,niah_multikey,niah_multivalue,vt}"
R8_ARMS="${R8_ARMS:-fp,uniform,evict}"
R8_BUDGETS="${R8_BUDGETS:-1,2,3,4}"
R8_N_PROMPTS="${R8_N_PROMPTS:-20}"
R8_PROMPT_OFFSET="${R8_PROMPT_OFFSET:-0}"
R8_N_KEYS="${R8_N_KEYS:-4}"
R8_N_VALUES="${R8_N_VALUES:-4}"
R8_N_HOPS="${R8_N_HOPS:-4}"
R8_WINDOW="${R8_WINDOW:-32}"
# P2 (unset = the P0 path, byte for byte)
R8_HEAD_ERROR="${R8_HEAD_ERROR:-0}"         # 1 = per-head output errors (P-4's data)
R8_N_Q="${R8_N_Q:-8}"                       # answer-span queries the errors average over
R8_THETA="${R8_THETA:-1.0}"                 # router margin
R8_CASCADE_BITS="${R8_CASCADE_BITS:-4}"
R8_ROUTES="${R8_ROUTES:-}"                  # router_calib: a calibration run's routes JSON
R8_WRITE_ROUTES="${R8_WRITE_ROUTES:-}"      # calibration run: where to write the routes
R8_QA="${R8_QA:-0}"                         # 1 = question-agnostic: compress before the question (plan.md 12)
# Step 3 whole-policy diagnostic (unset/0 leaves the historical invocation unchanged)
R8_POLICY_DIAGNOSTIC="${R8_POLICY_DIAGNOSTIC:-0}"
R8_POLICY_CANDIDATES="${R8_POLICY_CANDIDATES:-uniform,evict,interior}"
R8_POLICY_TRACE_STEPS="${R8_POLICY_TRACE_STEPS:-8}"

# pattern checks: fail before the GPU is held, not after
for _v in R8_CTX R8_N_PROMPTS R8_PROMPT_OFFSET R8_N_KEYS R8_N_VALUES R8_N_HOPS R8_WINDOW R8_N_Q R8_CASCADE_BITS R8_HEAD_ERROR R8_QA R8_POLICY_DIAGNOSTIC R8_POLICY_TRACE_STEPS; do
  [[ "${!_v}" =~ ^[0-9]+$ ]] || { echo "ERROR: $_v must be an integer, got '${!_v}'" >&2; exit 1; }
done
for _v in R8_N_KEYS R8_N_VALUES R8_N_HOPS; do
  (( ${!_v} >= 1 )) || { echo "ERROR: $_v must be positive, got '${!_v}'" >&2; exit 1; }
done
[[ "$R8_BUDGETS" =~ ^[0-9]+(\.[0-9]+)?(,[0-9]+(\.[0-9]+)?)*$ ]] || { echo "ERROR: R8_BUDGETS='$R8_BUDGETS'" >&2; exit 1; }
[[ "$R8_THETA" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "ERROR: R8_THETA='$R8_THETA'" >&2; exit 1; }
[[ "$R8_POLICY_DIAGNOSTIC" == 0 || "$R8_POLICY_DIAGNOSTIC" == 1 ]] || {
  echo "ERROR: R8_POLICY_DIAGNOSTIC must be 0 or 1, got '$R8_POLICY_DIAGNOSTIC'" >&2; exit 1; }
(( R8_POLICY_TRACE_STEPS >= 1 )) || {
  echo "ERROR: R8_POLICY_TRACE_STEPS must be positive, got '$R8_POLICY_TRACE_STEPS'" >&2; exit 1; }
[[ "$R8_POLICY_CANDIDATES" =~ ^[A-Za-z0-9_]+(,[A-Za-z0-9_]+)*$ ]] || {
  echo "ERROR: R8_POLICY_CANDIDATES must be a nonempty comma-separated arm-label list, got '$R8_POLICY_CANDIDATES'" >&2; exit 1; }
if [[ "$R8_POLICY_DIAGNOSTIC" == 1 ]]; then
  [[ ",$R8_ARMS," == *,fp,* ]] || {
    echo "ERROR: policy diagnostics require fp in R8_ARMS='$R8_ARMS'" >&2; exit 1; }
  IFS=, read -r -a _policy_candidates <<< "$R8_POLICY_CANDIDATES"
  _seen_policy_candidates=,
  for _candidate in "${_policy_candidates[@]}"; do
    [[ "$_candidate" != fp && "$_candidate" != router* ]] || {
      echo "ERROR: policy candidate '$_candidate' must be a fixed compressed-cache arm" >&2; exit 1; }
    [[ "$_seen_policy_candidates" != *,"$_candidate",* ]] || {
      echo "ERROR: duplicate policy candidate '$_candidate' in R8_POLICY_CANDIDATES" >&2; exit 1; }
    _seen_policy_candidates+="$_candidate,"
  done
fi
if [[ -n "$R8_ROUTES" && ! -f "$R8_ROUTES" ]]; then
  echo "ERROR: R8_ROUTES=$R8_ROUTES does not exist -- run the calibration first" >&2; exit 1; fi

# ---- environment: submit_h0.slurm's, verbatim in effect -------------------
VENV="${SIEVE_VENV:-$PROJECT_ROOT/.venv}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$PROJECT_ROOT/.hf_token}"
[[ -f "$HF_TOKEN_FILE" ]] && export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export H0_CORPUS="${H0_CORPUS:-$PROJECT_ROOT/.h0_corpus/pg19}"
source "$VENV/bin/activate"

# Resolve configurable SOTA arm specs through the same canonical parser as
# run_r8.py, then require exactly fp followed by the declared policy candidates.
# This rejects aliases, missing/extra arms, and order drift before model work.
if [[ "$R8_POLICY_DIAGNOSTIC" == 1 ]]; then
  python - "$R8_ARMS" "$R8_POLICY_CANDIDATES" <<'POLICY_PY'
import sys
from sievelib.baselines import resolve_arms

raw_arms = [x for x in sys.argv[1].split(",") if x]
candidates = [x for x in sys.argv[2].split(",") if x]
try:
    arms, _ = resolve_arms(raw_arms)
except (KeyError, ValueError) as exc:
    raise SystemExit(f"ERROR: invalid R8_ARMS for policy diagnostic: {exc}")
expected = ["fp", *candidates]
if arms != expected:
    raise SystemExit(
        "ERROR: policy diagnostic requires resolved R8_ARMS to equal "
        f"fp plus R8_POLICY_CANDIDATES in order; resolved={arms}, expected={expected}"
    )
POLICY_PY
fi

RUN_ID="${R8_RUN_ID:-${R8_RUN_PREFIX:-r8job}${SLURM_JOB_ID:-manual}}"
OUT_DIR="$PROJECT_ROOT/h0_measurement/results/${RUN_ID}"
mkdir -p "$OUT_DIR" h0_measurement/logs
POLICY_DIAGNOSTIC_OUT="$OUT_DIR/r8policy_${R8_MODEL}_${R8_CTX}_k${R8_N_KEYS}_v${R8_N_VALUES}_h${R8_N_HOPS}.parquet"

if [[ "$R8_POLICY_DIAGNOSTIC" == 1 ]]; then
  _policy_out_log="$POLICY_DIAGNOSTIC_OUT"
else
  _policy_out_log=none
fi
echo "host=$(hostname) role=r8 run_id=$RUN_ID model=$R8_MODEL ctx=$R8_CTX arms=$R8_ARMS budgets=$R8_BUDGETS tasks=$R8_TASKS difficulty=k${R8_N_KEYS}_v${R8_N_VALUES}_h${R8_N_HOPS} prompts=$R8_N_PROMPTS@$R8_PROMPT_OFFSET window=$R8_WINDOW qa=$R8_QA head_error=$R8_HEAD_ERROR n_q=$R8_N_Q theta=$R8_THETA routes=${R8_ROUTES:-none} write_routes=${R8_WRITE_ROUTES:-none} policy_diagnostic=$R8_POLICY_DIAGNOSTIC policy_candidates=$R8_POLICY_CANDIDATES policy_trace_steps=$R8_POLICY_TRACE_STEPS policy_out=$_policy_out_log"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__)"

# the haystack: run_r8.py refuses tier main/large without a real one
if ! python h0_measurement/prefetch_corpus.py --verify --out "$H0_CORPUS"; then
  echo "ERROR: no usable PG-19 haystack at H0_CORPUS=$H0_CORPUS -- stage it on the login node:"
  echo "    python h0_measurement/prefetch_corpus.py --check-ctx $R8_CTX --n-prompts $R8_N_PROMPTS"
  exit 1
fi

# gate: the R8 and paper-baseline tensor anchors (seconds). Their model halves
# need a tiny model on CPU and stay in the development test suite.
python tests/test_r8.py --fast || { echo "ERROR: R8 unit tests failed -- not running"; exit 1; }
python tests/test_baselines.py --fast || { echo "ERROR: baseline unit tests failed -- not running"; exit 1; }
if [[ "$R8_POLICY_DIAGNOSTIC" == 1 ]]; then
  python tests/test_policy_diagnostic.py || { echo "ERROR: policy diagnostic unit tests failed -- not running"; exit 1; }
  python tests/test_policy_reader.py || { echo "ERROR: policy reader unit tests failed -- not running"; exit 1; }
fi

python h0_measurement/prefetch.py -m "$R8_MODEL"

P2_ARGS=(--n-q "$R8_N_Q" --theta "$R8_THETA" --cascade-bits "$R8_CASCADE_BITS")
[[ "$R8_HEAD_ERROR" == 1 ]] && P2_ARGS+=(--head-error)
[[ -n "$R8_ROUTES" ]] && P2_ARGS+=(--routes "$R8_ROUTES")
[[ -n "$R8_WRITE_ROUTES" ]] && P2_ARGS+=(--write-routes "$R8_WRITE_ROUTES")
[[ "$R8_QA" == 1 ]] && P2_ARGS+=(--question-agnostic)
POLICY_ARGS=()
if [[ "$R8_POLICY_DIAGNOSTIC" == 1 ]]; then
  POLICY_ARGS+=(--policy-diagnostic-out "$POLICY_DIAGNOSTIC_OUT")
  POLICY_ARGS+=(--policy-candidates "$R8_POLICY_CANDIDATES")
  POLICY_ARGS+=(--policy-trace-steps "$R8_POLICY_TRACE_STEPS")
fi

srun python h0_measurement/run_r8.py \
  --model "$R8_MODEL" --ctx "$R8_CTX" --tasks "$R8_TASKS" --arms "$R8_ARMS" \
  --budgets "$R8_BUDGETS" --n-prompts "$R8_N_PROMPTS" \
  --prompt-offset "$R8_PROMPT_OFFSET" --n-keys "$R8_N_KEYS" \
  --n-values "$R8_N_VALUES" --n-hops "$R8_N_HOPS" \
  --window "$R8_WINDOW" --out-dir "$OUT_DIR" "${P2_ARGS[@]}" "${POLICY_ARGS[@]}"
