#!/bin/bash
#SBATCH --job-name=r14s1b
#SBATCH --output=h0_measurement/logs/r14s1b_%j.out
#SBATCH --error=h0_measurement/logs/r14s1b_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#
# R14 Stage 1b worker: one run_s1b.py process (a calibration or one evaluation
# block, every arm in this one process). Submit through script_stage1b.sh.
# Settings are ARGUMENTS (S1B_NAME=value): Trillium's sbatch adds --export=NONE.
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
cd "$PROJECT_ROOT"
for _a in "$@"; do
  if [[ "$_a" =~ ^S1B_[A-Z0-9_]+= ]]; then export "$_a"
  else echo "ERROR: unexpected argument '$_a' (use S1B_NAME=value)" >&2; exit 1; fi
done
: "${S1B_MODE:?}" "${S1B_TAG:?}" "${S1B_PRESET:?}" "${S1B_CTX:?}" "${S1B_N_PROMPTS:?}" "${S1B_PROMPT_OFFSET:?}"
S1B_MODEL="${S1B_MODEL:-llama31-8b}"
S1B_TASKS="${S1B_TASKS:-niah_single,niah_multikey,niah_multivalue,vt}"
S1B_ROUTES_STD="${S1B_ROUTES_STD:-}"
S1B_ROUTES_1B="${S1B_ROUTES_1B:-}"
S1B_WRITE_ROUTES="${S1B_WRITE_ROUTES:-}"
[[ "$S1B_MODE" == calibrate || "$S1B_MODE" == evaluate ]] || { echo "ERROR: S1B_MODE" >&2; exit 1; }
for _v in S1B_CTX S1B_N_PROMPTS S1B_PROMPT_OFFSET; do
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

OUT_DIR="$PROJECT_ROOT/h0_measurement/results/r14s1b_${S1B_TAG}_${SLURM_JOB_ID:-manual}"
echo "host=$(hostname) role=r14s1b mode=$S1B_MODE tag=$S1B_TAG preset=$S1B_PRESET model=$S1B_MODEL ctx=$S1B_CTX prompts=$S1B_N_PROMPTS@$S1B_PROMPT_OFFSET tasks=$S1B_TASKS routes_std=${S1B_ROUTES_STD:-none} routes_1b=${S1B_ROUTES_1B:-none} write_routes=${S1B_WRITE_ROUTES:-none} out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python h0_measurement/prefetch_corpus.py --verify --out "$H0_CORPUS" || { echo "ERROR: corpus"; exit 1; }
python -u h0_measurement/bugs/14_kernel_tpot/test_r14_stage1b.py --fast || { echo "ERROR: Stage 1b tests failed -- not running"; exit 1; }
python -u tests/test_r8.py --fast || { echo "ERROR: R8 tests failed -- not running"; exit 1; }

ARGS=(--mode "$S1B_MODE" --preset "$S1B_PRESET" --model "$S1B_MODEL" --ctx "$S1B_CTX"
      --tasks "$S1B_TASKS" --n-prompts "$S1B_N_PROMPTS" --prompt-offset "$S1B_PROMPT_OFFSET"
      --out-dir "$OUT_DIR")
[[ -n "$S1B_ROUTES_STD" ]] && ARGS+=(--routes-std "$S1B_ROUTES_STD")
[[ -n "$S1B_ROUTES_1B" ]] && ARGS+=(--routes-1b "$S1B_ROUTES_1B")
[[ -n "$S1B_WRITE_ROUTES" ]] && ARGS+=(--write-routes "$S1B_WRITE_ROUTES")
python -u h0_measurement/bugs/14_kernel_tpot/run_s1b.py "${ARGS[@]}"
