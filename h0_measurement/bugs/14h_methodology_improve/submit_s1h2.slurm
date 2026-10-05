#!/bin/bash
#SBATCH --job-name=r14s1h2
#SBATCH --output=h0_measurement/logs/r14s1h2_%j.out
#SBATCH --error=h0_measurement/logs/r14s1h2_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jzzzzzzz23@foxmail.com

#
# R14 Stage 1h R2 worker: one run_s1h2.py evaluation block (every arm of the block in
# this one process). Submit through script_stage1h.sh, from the project root (the
# root is the submit directory, so the same file runs on Trillium and Rorqual).
# Settings are ARGUMENTS (S1H_NAME=value): Trillium's sbatch adds --export=NONE.
#   S1H_TAG S1H_PRESET S1H_CTX            required
#   S1H_N_PROMPTS S1H_PROMPT_OFFSET       a prompt range, or
#   S1H_PROMPT_LIST                       p:task,...
#   S1H_TASKS                             default: the four RULER tasks
#   S1H_SEED                              rotation seed (--override rot_seed=N); default: models.yaml's
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14h_methodology_improve
[[ -f $DIR/run_s1h2.py ]] || { echo "ERROR: submit from the project root" >&2; exit 1; }
for _a in "$@"; do
  if [[ "$_a" =~ ^S1H_[A-Z0-9_]+= ]]; then export "$_a"
  else echo "ERROR: unexpected argument '$_a' (use S1H_NAME=value)" >&2; exit 1; fi
done
: "${S1H_TAG:?}" "${S1H_PRESET:?}" "${S1H_CTX:?}"
S1H_TASKS="${S1H_TASKS:-niah_single,niah_multikey,niah_multivalue,vt}"
S1H_N_PROMPTS="${S1H_N_PROMPTS:-0}"
S1H_PROMPT_OFFSET="${S1H_PROMPT_OFFSET:-0}"
S1H_PROMPT_LIST="${S1H_PROMPT_LIST:-}"
S1H_SEED="${S1H_SEED:-}"
for _v in S1H_CTX S1H_N_PROMPTS S1H_PROMPT_OFFSET; do
  [[ "${!_v}" =~ ^[0-9]+$ ]] || { echo "ERROR: $_v must be an integer" >&2; exit 1; }
done
[[ -z "$S1H_SEED" || "$S1H_SEED" =~ ^[0-9]+$ ]] || { echo "ERROR: S1H_SEED must be an integer" >&2; exit 1; }

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

OUT_DIR="$PROJECT_ROOT/h0_measurement/results/r14s1h_${S1H_TAG}_${SLURM_JOB_ID:-manual}"
echo "host=$(hostname) role=r14s1h2 tag=$S1H_TAG preset=$S1H_PRESET ctx=$S1H_CTX prompts=$S1H_N_PROMPTS@$S1H_PROMPT_OFFSET list=${S1H_PROMPT_LIST:-none} tasks=$S1H_TASKS seed=${S1H_SEED:-default} out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python h0_measurement/prefetch_corpus.py --verify --out "$H0_CORPUS" || { echo "ERROR: corpus"; exit 1; }
python -u $DIR/test_r14_stage1h_r2.py --fast || { echo "ERROR: Stage 1h R2 tests failed -- not running"; exit 1; }
python -u $DIR/test_r14_stage1h.py --fast || { echo "ERROR: Stage 1h tests failed -- not running"; exit 1; }
python -u h0_measurement/bugs/14_kernel_tpot/test_r14_stage1g.py --fast || { echo "ERROR: Stage 1g tests failed -- not running"; exit 1; }
python -u tests/test_r8.py --fast || { echo "ERROR: R8 tests failed -- not running"; exit 1; }

SEL=(--tasks "$S1H_TASKS")
if [[ -n "$S1H_PROMPT_LIST" ]]; then SEL+=(--prompt-list "$S1H_PROMPT_LIST")
else SEL+=(--n-prompts "$S1H_N_PROMPTS" --prompt-offset "$S1H_PROMPT_OFFSET"); fi
OVR=()
[[ -n "$S1H_SEED" ]] && OVR=(--override "rot_seed=$S1H_SEED")
python -u $DIR/run_s1h2.py --mode evaluate --preset "$S1H_PRESET" --ctx "$S1H_CTX" "${SEL[@]}" "${OVR[@]}" \
    --out-dir "$OUT_DIR"
