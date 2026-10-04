#!/bin/bash
#SBATCH --job-name=r14s1g
#SBATCH --output=h0_measurement/logs/r14s1g_%j.out
#SBATCH --error=h0_measurement/logs/r14s1g_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#
# R14 Stage 1g worker: one run_s1g.py evaluation block (every arm of the block in
# this one process). Submit through script_stage1g.sh, from the project root (the
# root is the submit directory, so the same file runs on Trillium and Rorqual).
# Settings are ARGUMENTS (S1G_NAME=value): Trillium's sbatch adds --export=NONE.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
[[ -f h0_measurement/bugs/14_kernel_tpot/run_s1g.py ]] || { echo "ERROR: submit from the project root" >&2; exit 1; }
for _a in "$@"; do
  if [[ "$_a" =~ ^S1G_[A-Z0-9_]+= ]]; then export "$_a"
  else echo "ERROR: unexpected argument '$_a' (use S1G_NAME=value)" >&2; exit 1; fi
done
: "${S1G_TAG:?}" "${S1G_PRESET:?}" "${S1G_CTX:?}"
S1G_TASKS="${S1G_TASKS:-niah_single,niah_multikey,niah_multivalue,vt}"
S1G_N_PROMPTS="${S1G_N_PROMPTS:-0}"
S1G_PROMPT_OFFSET="${S1G_PROMPT_OFFSET:-0}"
S1G_PROMPT_LIST="${S1G_PROMPT_LIST:-}"
for _v in S1G_CTX S1G_N_PROMPTS S1G_PROMPT_OFFSET; do
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

OUT_DIR="$PROJECT_ROOT/h0_measurement/results/r14s1g_${S1G_TAG}_${SLURM_JOB_ID:-manual}"
echo "host=$(hostname) role=r14s1g tag=$S1G_TAG preset=$S1G_PRESET ctx=$S1G_CTX prompts=$S1G_N_PROMPTS@$S1G_PROMPT_OFFSET list=${S1G_PROMPT_LIST:-none} tasks=$S1G_TASKS out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python h0_measurement/prefetch_corpus.py --verify --out "$H0_CORPUS" || { echo "ERROR: corpus"; exit 1; }
for t in test_r14_stage1g test_r14_stage1f; do
  python -u h0_measurement/bugs/14_kernel_tpot/$t.py --fast || { echo "ERROR: $t failed -- not running"; exit 1; }
done
python -u tests/test_r8.py --fast || { echo "ERROR: R8 tests failed -- not running"; exit 1; }

SEL=(--tasks "$S1G_TASKS")
if [[ -n "$S1G_PROMPT_LIST" ]]; then SEL+=(--prompt-list "$S1G_PROMPT_LIST")
else SEL+=(--n-prompts "$S1G_N_PROMPTS" --prompt-offset "$S1G_PROMPT_OFFSET"); fi
python -u h0_measurement/bugs/14_kernel_tpot/run_s1g.py --mode evaluate --preset "$S1G_PRESET" --ctx "$S1G_CTX" \
    "${SEL[@]}" --out-dir "$OUT_DIR"
