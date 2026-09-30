#!/bin/bash
#SBATCH --job-name=r14s1d
#SBATCH --output=h0_measurement/logs/r14s1d_%j.out
#SBATCH --error=h0_measurement/logs/r14s1d_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#
# R14 Stage 1d worker: one run_s1d.py process (a calibration, or one evaluation
# block with every arm in this one process). S1D_MODE=pilot runs the pilot's two
# steps back to back: a mechanics calibration (one prompt-task, search forced),
# then every pilot arm with its routes. Submit through script_stage1d.sh.
# Settings are ARGUMENTS (S1D_NAME=value): Trillium's sbatch adds --export=NONE.
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
cd "$PROJECT_ROOT"
for _a in "$@"; do
  if [[ "$_a" =~ ^S1D_[A-Z0-9_]+= ]]; then export "$_a"
  else echo "ERROR: unexpected argument '$_a' (use S1D_NAME=value)" >&2; exit 1; fi
done
: "${S1D_MODE:?}" "${S1D_TAG:?}" "${S1D_PRESET:?}" "${S1D_CTX:?}" "${S1D_N_PROMPTS:?}" "${S1D_PROMPT_OFFSET:?}"
S1D_MODEL="${S1D_MODEL:-}"
S1D_TASKS="${S1D_TASKS:-niah_single,niah_multikey,niah_multivalue,vt}"
S1D_ROUTES_STD="${S1D_ROUTES_STD:-}"
S1D_ROUTES_1B="${S1D_ROUTES_1B:-}"
S1D_ROUTES_1C="${S1D_ROUTES_1C:-}"
S1D_ROUTES_1D="${S1D_ROUTES_1D:-}"
S1D_WRITE_ROUTES="${S1D_WRITE_ROUTES:-}"
S1D_CAL_OFFSET="${S1D_CAL_OFFSET:-0}"          # pilot: its mechanics calibration's prompt
[[ "$S1D_MODE" =~ ^(calibrate|evaluate|pilot)$ ]] || { echo "ERROR: S1D_MODE" >&2; exit 1; }
for _v in S1D_CTX S1D_N_PROMPTS S1D_PROMPT_OFFSET S1D_CAL_OFFSET; do
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

OUT_DIR="$PROJECT_ROOT/h0_measurement/results/r14s1d_${S1D_TAG}_${SLURM_JOB_ID:-manual}"
echo "host=$(hostname) role=r14s1d mode=$S1D_MODE tag=$S1D_TAG preset=$S1D_PRESET model=${S1D_MODEL:-preset} ctx=$S1D_CTX prompts=$S1D_N_PROMPTS@$S1D_PROMPT_OFFSET tasks=$S1D_TASKS routes_std=${S1D_ROUTES_STD:-none} routes_1b=${S1D_ROUTES_1B:-none} routes_1c=${S1D_ROUTES_1C:-none} routes_1d=${S1D_ROUTES_1D:-none} write_routes=${S1D_WRITE_ROUTES:-none} out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python h0_measurement/prefetch_corpus.py --verify --out "$H0_CORPUS" || { echo "ERROR: corpus"; exit 1; }
for t in test_r14_stage1d test_r14_stage1c test_r14_stage1b; do
  python -u h0_measurement/bugs/14_kernel_tpot/$t.py --fast || { echo "ERROR: $t failed -- not running"; exit 1; }
done
python -u tests/test_r8.py --fast || { echo "ERROR: R8 tests failed -- not running"; exit 1; }

RUN=(python -u h0_measurement/bugs/14_kernel_tpot/run_s1d.py --preset "$S1D_PRESET" --ctx "$S1D_CTX"
     --out-dir "$OUT_DIR")
[[ -n "$S1D_MODEL" ]] && RUN+=(--model "$S1D_MODEL")
ROUTES=()
[[ -n "$S1D_ROUTES_STD" ]] && ROUTES+=(--routes-std "$S1D_ROUTES_STD")
[[ -n "$S1D_ROUTES_1B" ]] && ROUTES+=(--routes-1b "$S1D_ROUTES_1B")
[[ -n "$S1D_ROUTES_1C" ]] && ROUTES+=(--routes-1c "$S1D_ROUTES_1C")
if [[ "$S1D_MODE" == pilot ]]; then
  PILOT_RT="$OUT_DIR/pilot_routes_1d.json"
  "${RUN[@]}" --mode calibrate --tasks niah_single --n-prompts 1 --prompt-offset "$S1D_CAL_OFFSET" \
      --force-search --routes-1b "$S1D_ROUTES_1B" --write-routes "$PILOT_RT"
  "${RUN[@]}" --mode evaluate --tasks "$S1D_TASKS" --n-prompts "$S1D_N_PROMPTS" \
      --prompt-offset "$S1D_PROMPT_OFFSET" "${ROUTES[@]}" --routes-1d "$PILOT_RT"
else
  ARGS=(--mode "$S1D_MODE" --tasks "$S1D_TASKS" --n-prompts "$S1D_N_PROMPTS"
        --prompt-offset "$S1D_PROMPT_OFFSET" "${ROUTES[@]}")
  [[ -n "$S1D_ROUTES_1D" ]] && ARGS+=(--routes-1d "$S1D_ROUTES_1D")
  [[ -n "$S1D_WRITE_ROUTES" ]] && ARGS+=(--write-routes "$S1D_WRITE_ROUTES")
  "${RUN[@]}" "${ARGS[@]}"
fi
