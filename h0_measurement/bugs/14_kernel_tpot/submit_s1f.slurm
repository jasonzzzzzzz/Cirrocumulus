#!/bin/bash
#SBATCH --job-name=r14s1f
#SBATCH --output=h0_measurement/logs/r14s1f_%j.out
#SBATCH --error=h0_measurement/logs/r14s1f_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#
# R14 Stage 1f worker: one run_s1f.py process (a calibration, one evaluation or
# reuse block with every arm in this one process). S1F_MODE=pilot runs a pilot
# with its own calibration back to back: a mechanics calibration (one prompt per
# S1F_CAL_TASKS task, search forced), then every pilot arm with its routes.
# Submit through script_stage1f.sh, from the project root (the root is the
# submit directory, so the same file runs on Trillium and Rorqual). Settings are
# ARGUMENTS (S1F_NAME=value): Trillium's sbatch adds --export=NONE.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
[[ -f h0_measurement/bugs/14_kernel_tpot/run_s1f.py ]] || { echo "ERROR: submit from the project root" >&2; exit 1; }
for _a in "$@"; do
  if [[ "$_a" =~ ^S1F_[A-Z0-9_]+= ]]; then export "$_a"
  else echo "ERROR: unexpected argument '$_a' (use S1F_NAME=value)" >&2; exit 1; fi
done
: "${S1F_MODE:?}" "${S1F_TAG:?}" "${S1F_PRESET:?}" "${S1F_CTX:?}"
S1F_TASKS="${S1F_TASKS:-niah_single,niah_multikey,niah_multivalue,vt}"
S1F_N_PROMPTS="${S1F_N_PROMPTS:-0}"
S1F_PROMPT_OFFSET="${S1F_PROMPT_OFFSET:-0}"
S1F_PROMPT_LIST="${S1F_PROMPT_LIST:-}"
S1F_TASK_COUNTS="${S1F_TASK_COUNTS:-}"
S1F_ROUTES_1B="${S1F_ROUTES_1B:-}"
S1F_ROUTES_1D="${S1F_ROUTES_1D:-}"
S1F_ROUTES_1E="${S1F_ROUTES_1E:-}"
S1F_WRITE_ROUTES="${S1F_WRITE_ROUTES:-}"
S1F_CAL_OFFSET="${S1F_CAL_OFFSET:-0}"          # pilot: its mechanics calibration's prompt
S1F_CAL_TASKS="${S1F_CAL_TASKS:-niah_single}"  # pilot: one prompt of each
[[ "$S1F_MODE" =~ ^(calibrate|evaluate|reuse|pilot)$ ]] || { echo "ERROR: S1F_MODE" >&2; exit 1; }
for _v in S1F_CTX S1F_N_PROMPTS S1F_PROMPT_OFFSET S1F_CAL_OFFSET; do
  [[ "${!_v}" =~ ^[0-9]+$ ]] || { echo "ERROR: $_v must be an integer" >&2; exit 1; }
done

VENV="$PROJECT_ROOT/.venv"
HF_TOKEN_FILE="$PROJECT_ROOT/.hf_token"
[[ -f "$HF_TOKEN_FILE" ]] && export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export H0_CORPUS="$PROJECT_ROOT/.h0_corpus/pg19"
export OMP_NUM_THREADS=8
source "$VENV/bin/activate"

OUT_DIR="$PROJECT_ROOT/h0_measurement/results/r14s1f_${S1F_TAG}_${SLURM_JOB_ID:-manual}"
echo "host=$(hostname) role=r14s1f mode=$S1F_MODE tag=$S1F_TAG preset=$S1F_PRESET ctx=$S1F_CTX prompts=$S1F_N_PROMPTS@$S1F_PROMPT_OFFSET list=${S1F_PROMPT_LIST:-none} counts=${S1F_TASK_COUNTS:-none} tasks=$S1F_TASKS routes_1b=${S1F_ROUTES_1B:-none} routes_1d=${S1F_ROUTES_1D:-none} routes_1e=${S1F_ROUTES_1E:-none} write_routes=${S1F_WRITE_ROUTES:-none} out=$OUT_DIR"
for f in "$S1F_ROUTES_1B" "$S1F_ROUTES_1D" "$S1F_ROUTES_1E"; do
  [[ -z "$f" ]] || echo "routes $(sha256sum "$f" | cut -c1-10) $f"
done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python h0_measurement/prefetch_corpus.py --verify --out "$H0_CORPUS" || { echo "ERROR: corpus"; exit 1; }
for t in test_r14_stage1f test_r14_stage1e; do
  python -u h0_measurement/bugs/14_kernel_tpot/$t.py --fast || { echo "ERROR: $t failed -- not running"; exit 1; }
done
python -u tests/test_r8.py --fast || { echo "ERROR: R8 tests failed -- not running"; exit 1; }

RUN=(python -u h0_measurement/bugs/14_kernel_tpot/run_s1f.py --preset "$S1F_PRESET" --ctx "$S1F_CTX"
     --out-dir "$OUT_DIR")
ROUTES=()
[[ -n "$S1F_ROUTES_1B" ]] && ROUTES+=(--routes-1b "$S1F_ROUTES_1B")
[[ -n "$S1F_ROUTES_1D" ]] && ROUTES+=(--routes-1d "$S1F_ROUTES_1D")
SEL=(--tasks "$S1F_TASKS")
if [[ -n "$S1F_PROMPT_LIST" ]]; then SEL+=(--prompt-list "$S1F_PROMPT_LIST")
elif [[ -n "$S1F_TASK_COUNTS" ]]; then SEL+=(--task-counts "$S1F_TASK_COUNTS" --prompt-offset "$S1F_PROMPT_OFFSET")
else SEL+=(--n-prompts "$S1F_N_PROMPTS" --prompt-offset "$S1F_PROMPT_OFFSET"); fi
if [[ "$S1F_MODE" == pilot ]]; then
  PILOT_RT="$OUT_DIR/pilot_routes_1f.json"
  CC=$(echo "$S1F_CAL_TASKS" | tr ',' '\n' | sed 's/$/=1/' | paste -sd,)
  "${RUN[@]}" --mode calibrate --tasks "$S1F_CAL_TASKS" --task-counts "$CC" --prompt-offset "$S1F_CAL_OFFSET" \
      --force-search --routes-1b "$S1F_ROUTES_1B" --write-routes "$PILOT_RT"
  "${RUN[@]}" --mode evaluate "${SEL[@]}" "${ROUTES[@]}" --routes-1e "$PILOT_RT"
else
  ARGS=(--mode "$S1F_MODE" "${SEL[@]}" "${ROUTES[@]}")
  [[ -n "$S1F_ROUTES_1E" ]] && ARGS+=(--routes-1e "$S1F_ROUTES_1E")
  [[ -n "$S1F_WRITE_ROUTES" ]] && ARGS+=(--write-routes "$S1F_WRITE_ROUTES")
  "${RUN[@]}" "${ARGS[@]}"
fi
