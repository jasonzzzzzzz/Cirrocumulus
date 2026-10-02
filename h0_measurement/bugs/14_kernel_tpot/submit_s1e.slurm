#!/bin/bash
#SBATCH --job-name=r14s1e
#SBATCH --output=h0_measurement/logs/r14s1e_%j.out
#SBATCH --error=h0_measurement/logs/r14s1e_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#
# R14 Stage 1e worker: one run_s1e.py process (a calibration, one evaluation or
# reuse block with every arm in this one process). S1E_MODE=pilot runs the tail
# pilot's two steps back to back: a mechanics calibration (one prompt-task,
# search forced), then every pilot arm with its routes. Submit through
# script_stage1e.sh. Settings are ARGUMENTS (S1E_NAME=value): Trillium's sbatch
# adds --export=NONE. GPUs: pass --gpus-per-node=2 to sbatch for Qwen at 128K.
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
cd "$PROJECT_ROOT"
for _a in "$@"; do
  if [[ "$_a" =~ ^S1E_[A-Z0-9_]+= ]]; then export "$_a"
  else echo "ERROR: unexpected argument '$_a' (use S1E_NAME=value)" >&2; exit 1; fi
done
: "${S1E_MODE:?}" "${S1E_TAG:?}" "${S1E_PRESET:?}" "${S1E_CTX:?}"
S1E_MODEL="${S1E_MODEL:-}"
S1E_TASKS="${S1E_TASKS:-niah_single,niah_multikey,niah_multivalue,vt}"
S1E_N_PROMPTS="${S1E_N_PROMPTS:-0}"
S1E_PROMPT_OFFSET="${S1E_PROMPT_OFFSET:-0}"
S1E_PROMPT_LIST="${S1E_PROMPT_LIST:-}"
S1E_TASK_COUNTS="${S1E_TASK_COUNTS:-}"
S1E_ROUTES_STD="${S1E_ROUTES_STD:-}"
S1E_ROUTES_1B="${S1E_ROUTES_1B:-}"
S1E_ROUTES_1D="${S1E_ROUTES_1D:-}"
S1E_ROUTES_1E="${S1E_ROUTES_1E:-}"
S1E_WRITE_ROUTES="${S1E_WRITE_ROUTES:-}"
S1E_CAL_OFFSET="${S1E_CAL_OFFSET:-0}"          # pilot: its mechanics calibration's prompt
[[ "$S1E_MODE" =~ ^(calibrate|evaluate|reuse|pilot)$ ]] || { echo "ERROR: S1E_MODE" >&2; exit 1; }
for _v in S1E_CTX S1E_N_PROMPTS S1E_PROMPT_OFFSET S1E_CAL_OFFSET; do
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

OUT_DIR="$PROJECT_ROOT/h0_measurement/results/r14s1e_${S1E_TAG}_${SLURM_JOB_ID:-manual}"
echo "host=$(hostname) role=r14s1e mode=$S1E_MODE tag=$S1E_TAG preset=$S1E_PRESET model=${S1E_MODEL:-preset} ctx=$S1E_CTX prompts=$S1E_N_PROMPTS@$S1E_PROMPT_OFFSET list=${S1E_PROMPT_LIST:-none} counts=${S1E_TASK_COUNTS:-none} tasks=$S1E_TASKS routes_std=${S1E_ROUTES_STD:-none} routes_1b=${S1E_ROUTES_1B:-none} routes_1d=${S1E_ROUTES_1D:-none} routes_1e=${S1E_ROUTES_1E:-none} write_routes=${S1E_WRITE_ROUTES:-none} out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python h0_measurement/prefetch_corpus.py --verify --out "$H0_CORPUS" || { echo "ERROR: corpus"; exit 1; }
for t in test_r14_stage1e test_r14_stage1d; do
  python -u h0_measurement/bugs/14_kernel_tpot/$t.py --fast || { echo "ERROR: $t failed -- not running"; exit 1; }
done
python -u tests/test_r8.py --fast || { echo "ERROR: R8 tests failed -- not running"; exit 1; }

RUN=(python -u h0_measurement/bugs/14_kernel_tpot/run_s1e.py --preset "$S1E_PRESET" --ctx "$S1E_CTX"
     --out-dir "$OUT_DIR")
[[ -n "$S1E_MODEL" ]] && RUN+=(--model "$S1E_MODEL")
ROUTES=()
[[ -n "$S1E_ROUTES_STD" ]] && ROUTES+=(--routes-std "$S1E_ROUTES_STD")
[[ -n "$S1E_ROUTES_1B" ]] && ROUTES+=(--routes-1b "$S1E_ROUTES_1B")
[[ -n "$S1E_ROUTES_1D" ]] && ROUTES+=(--routes-1d "$S1E_ROUTES_1D")
SEL=(--tasks "$S1E_TASKS")
if [[ -n "$S1E_PROMPT_LIST" ]]; then SEL+=(--prompt-list "$S1E_PROMPT_LIST")
elif [[ -n "$S1E_TASK_COUNTS" ]]; then SEL+=(--task-counts "$S1E_TASK_COUNTS" --prompt-offset "$S1E_PROMPT_OFFSET")
else SEL+=(--n-prompts "$S1E_N_PROMPTS" --prompt-offset "$S1E_PROMPT_OFFSET"); fi
if [[ "$S1E_MODE" == pilot ]]; then
  PILOT_RT="$OUT_DIR/pilot_routes_1e.json"
  "${RUN[@]}" --mode calibrate --tasks niah_single --task-counts niah_single=1 --prompt-offset "$S1E_CAL_OFFSET" \
      --force-search --routes-1b "$S1E_ROUTES_1B" --write-routes "$PILOT_RT"
  "${RUN[@]}" --mode evaluate "${SEL[@]}" "${ROUTES[@]}" --routes-1e "$PILOT_RT"
else
  ARGS=(--mode "$S1E_MODE" "${SEL[@]}" "${ROUTES[@]}")
  [[ -n "$S1E_ROUTES_1E" ]] && ARGS+=(--routes-1e "$S1E_ROUTES_1E")
  [[ -n "$S1E_WRITE_ROUTES" ]] && ARGS+=(--write-routes "$S1E_WRITE_ROUTES")
  "${RUN[@]}" "${ARGS[@]}"
fi
