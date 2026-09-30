#!/bin/bash
#SBATCH --job-name=r14s1c
#SBATCH --output=h0_measurement/logs/r14s1c_%j.out
#SBATCH --error=h0_measurement/logs/r14s1c_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#
# R14 Stage 1c worker: one run_s1c.py process (a calibration, or one evaluation
# block with every arm in this one process). S1C_MODE=pilot runs the pilot's two
# steps back to back: a mechanics calibration (one prompt-task, search forced),
# then every pilot arm with its routes. Submit through script_stage1c.sh.
# Settings are ARGUMENTS (S1C_NAME=value): Trillium's sbatch adds --export=NONE.
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
cd "$PROJECT_ROOT"
for _a in "$@"; do
  if [[ "$_a" =~ ^S1C_[A-Z0-9_]+= ]]; then export "$_a"
  else echo "ERROR: unexpected argument '$_a' (use S1C_NAME=value)" >&2; exit 1; fi
done
: "${S1C_MODE:?}" "${S1C_TAG:?}" "${S1C_PRESET:?}" "${S1C_CTX:?}" "${S1C_N_PROMPTS:?}" "${S1C_PROMPT_OFFSET:?}"
S1C_MODEL="${S1C_MODEL:-llama31-8b}"
S1C_TASKS="${S1C_TASKS:-niah_single,niah_multikey,niah_multivalue,vt}"
S1C_ROUTES_STD="${S1C_ROUTES_STD:-}"
S1C_ROUTES_1B="${S1C_ROUTES_1B:-}"
S1C_ROUTES_1C="${S1C_ROUTES_1C:-}"
S1C_WRITE_ROUTES="${S1C_WRITE_ROUTES:-}"
S1C_CAL_OFFSET="${S1C_CAL_OFFSET:-0}"          # pilot: its mechanics calibration's prompt
[[ "$S1C_MODE" =~ ^(calibrate|evaluate|pilot)$ ]] || { echo "ERROR: S1C_MODE" >&2; exit 1; }
for _v in S1C_CTX S1C_N_PROMPTS S1C_PROMPT_OFFSET S1C_CAL_OFFSET; do
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

OUT_DIR="$PROJECT_ROOT/h0_measurement/results/r14s1c_${S1C_TAG}_${SLURM_JOB_ID:-manual}"
echo "host=$(hostname) role=r14s1c mode=$S1C_MODE tag=$S1C_TAG preset=$S1C_PRESET model=$S1C_MODEL ctx=$S1C_CTX prompts=$S1C_N_PROMPTS@$S1C_PROMPT_OFFSET tasks=$S1C_TASKS routes_std=${S1C_ROUTES_STD:-none} routes_1b=${S1C_ROUTES_1B:-none} routes_1c=${S1C_ROUTES_1C:-none} write_routes=${S1C_WRITE_ROUTES:-none} out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python h0_measurement/prefetch_corpus.py --verify --out "$H0_CORPUS" || { echo "ERROR: corpus"; exit 1; }
python -u h0_measurement/bugs/14_kernel_tpot/test_r14_stage1c.py --fast || { echo "ERROR: Stage 1c tests failed -- not running"; exit 1; }
python -u h0_measurement/bugs/14_kernel_tpot/test_r14_stage1b.py --fast || { echo "ERROR: Stage 1b tests failed -- not running"; exit 1; }
python -u tests/test_r8.py --fast || { echo "ERROR: R8 tests failed -- not running"; exit 1; }

RUN=(python -u h0_measurement/bugs/14_kernel_tpot/run_s1c.py --preset "$S1C_PRESET"
     --model "$S1C_MODEL" --ctx "$S1C_CTX" --out-dir "$OUT_DIR")
if [[ "$S1C_MODE" == pilot ]]; then
  PILOT_RT="$OUT_DIR/pilot_routes_1c.json"
  "${RUN[@]}" --mode calibrate --tasks niah_single --n-prompts 1 --prompt-offset "$S1C_CAL_OFFSET" \
      --force-search --routes-1b "$S1C_ROUTES_1B" --write-routes "$PILOT_RT"
  "${RUN[@]}" --mode evaluate --tasks "$S1C_TASKS" --n-prompts "$S1C_N_PROMPTS" \
      --prompt-offset "$S1C_PROMPT_OFFSET" --routes-std "$S1C_ROUTES_STD" \
      --routes-1b "$S1C_ROUTES_1B" --routes-1c "$PILOT_RT"
else
  ARGS=(--mode "$S1C_MODE" --tasks "$S1C_TASKS" --n-prompts "$S1C_N_PROMPTS"
        --prompt-offset "$S1C_PROMPT_OFFSET")
  [[ -n "$S1C_ROUTES_STD" ]] && ARGS+=(--routes-std "$S1C_ROUTES_STD")
  [[ -n "$S1C_ROUTES_1B" ]] && ARGS+=(--routes-1b "$S1C_ROUTES_1B")
  [[ -n "$S1C_ROUTES_1C" ]] && ARGS+=(--routes-1c "$S1C_ROUTES_1C")
  [[ -n "$S1C_WRITE_ROUTES" ]] && ARGS+=(--write-routes "$S1C_WRITE_ROUTES")
  "${RUN[@]}" "${ARGS[@]}"
fi
