#!/bin/bash
#SBATCH --job-name=r14s1h4
#SBATCH --output=h0_measurement/logs/r14s1h4_%j.out
#SBATCH --error=h0_measurement/logs/r14s1h4_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jzzzzzzz23@foxmail.com

#
# R14 Stage 1h R4 worker: one run_s1h4.py evaluation block (every arm of the block in
# one process), or S1H_SPLIT blocks at once on separate GPUs. Submit through script_stage1h.sh, from the project root (the
# root is the submit directory, so the same file runs on Trillium and Rorqual).
# Settings are ARGUMENTS (S1H_NAME=value): Trillium's sbatch adds --export=NONE.
#   S1H_TAG S1H_PRESET S1H_CTX S1H_TASKS  required (R4's: lbv2, or HELMET tasks; one suite)
#   S1H_N_PROMPTS S1H_PROMPT_OFFSET       the prompt range, or
#   S1H_PROMPT_LIST                       p:task,... (the pilots)
#   S1H_SEED                              rotation seed (--override rot_seed=N); default: models.yaml's
#   S1H_SPLIT                             blocks run at once in this job (default 1): block i takes prompts
#                                         OFFSET + i x N_PROMPTS on its own group of the job's GPUs, and writes
#                                         r14s1h_<tag>_<job>_<i>/ (log logs/r14s1h4_<job>_<i>.log)
#   S1H_GPUS                              GPUs per process (default: the job's GPUs / S1H_SPLIT)
# GPUs: the submit line's --gpus-per-node. Trillium gives a GPU job 1 GPU or whole nodes (4): Qwen at
# 128K needs 2, so its blocks take a node and run S1H_SPLIT=2 blocks on 2 GPUs each, and its pilot
# takes a node and runs on S1H_GPUS=2 (the gate's per-GPU peak is then a block's).
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14h_methodology_improve
[[ -f $DIR/run_s1h4.py ]] || { echo "ERROR: submit from the project root" >&2; exit 1; }
for _a in "$@"; do
  if [[ "$_a" =~ ^S1H_[A-Z0-9_]+= ]]; then export "$_a"
  else echo "ERROR: unexpected argument '$_a' (use S1H_NAME=value)" >&2; exit 1; fi
done
: "${S1H_TAG:?}" "${S1H_PRESET:?}" "${S1H_CTX:?}" "${S1H_TASKS:?}"
S1H_N_PROMPTS="${S1H_N_PROMPTS:-0}"
S1H_PROMPT_OFFSET="${S1H_PROMPT_OFFSET:-0}"
S1H_PROMPT_LIST="${S1H_PROMPT_LIST:-}"
S1H_SEED="${S1H_SEED:-}"
S1H_SPLIT="${S1H_SPLIT:-1}"
S1H_GPUS="${S1H_GPUS:-0}"
for _v in S1H_CTX S1H_N_PROMPTS S1H_PROMPT_OFFSET S1H_SPLIT S1H_GPUS; do
  [[ "${!_v}" =~ ^[0-9]+$ ]] || { echo "ERROR: $_v must be an integer" >&2; exit 1; }
done
[[ -z "$S1H_SEED" || "$S1H_SEED" =~ ^[0-9]+$ ]] || { echo "ERROR: S1H_SEED must be an integer" >&2; exit 1; }
(( S1H_SPLIT >= 1 )) || { echo "ERROR: S1H_SPLIT must be at least 1" >&2; exit 1; }

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
echo "host=$(hostname) role=r14s1h4 tag=$S1H_TAG preset=$S1H_PRESET ctx=$S1H_CTX prompts=$S1H_N_PROMPTS@$S1H_PROMPT_OFFSET tasks=$S1H_TASKS list=${S1H_PROMPT_LIST:-none} seed=${S1H_SEED:-default} split=$S1H_SPLIT gpus=${S1H_GPUS/#0/all} out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
GPUS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true); fi
NG=${#GPUS[@]}
PER=$S1H_GPUS
(( PER > 0 )) || PER=$((NG / S1H_SPLIT))
if (( S1H_SPLIT > 1 || S1H_GPUS > 0 )); then
  (( PER >= 1 && PER * S1H_SPLIT <= NG )) || { echo "ERROR: GPUs [${GPUS[*]}] cannot run $S1H_SPLIT process(es) x $PER" >&2; exit 1; }
fi
python h0_measurement/prefetch_corpus.py --verify --out "$H0_CORPUS" || { echo "ERROR: corpus"; exit 1; }
python -u $DIR/test_r14_stage1h_r4.py --fast || { echo "ERROR: Stage 1h R4 tests failed -- not running"; exit 1; }
python -u $DIR/test_r14_stage1h_r3b.py --fast || { echo "ERROR: Stage 1h R3b tests failed -- not running"; exit 1; }
python -u $DIR/test_r14_stage1h_r3.py --fast || { echo "ERROR: Stage 1h R3a tests failed -- not running"; exit 1; }
python -u $DIR/test_r14_stage1h_r2.py --fast || { echo "ERROR: Stage 1h R2 tests failed -- not running"; exit 1; }
python -u $DIR/test_r14_stage1h.py --fast || { echo "ERROR: Stage 1h tests failed -- not running"; exit 1; }
python -u h0_measurement/bugs/14_kernel_tpot/test_r14_stage1g.py --fast || { echo "ERROR: Stage 1g tests failed -- not running"; exit 1; }
python -u tests/test_r8.py --fast || { echo "ERROR: R8 tests failed -- not running"; exit 1; }

SEL=(--tasks "$S1H_TASKS")
if [[ -n "$S1H_PROMPT_LIST" ]]; then SEL+=(--prompt-list "$S1H_PROMPT_LIST")
else SEL+=(--n-prompts "$S1H_N_PROMPTS" --prompt-offset "$S1H_PROMPT_OFFSET"); fi
OVR=()
[[ -n "$S1H_SEED" ]] && OVR=(--override "rot_seed=$S1H_SEED")
if (( S1H_SPLIT == 1 )); then
  (( S1H_GPUS > 0 )) && export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]:0:$PER}")
  python -u $DIR/run_s1h4.py --mode evaluate --preset "$S1H_PRESET" --ctx "$S1H_CTX" "${SEL[@]}" "${OVR[@]}" \
      --out-dir "$OUT_DIR"
  exit 0
fi
[[ -z "$S1H_PROMPT_LIST" ]] || { echo "ERROR: S1H_SPLIT needs a prompt range, not a list" >&2; exit 1; }
PIDS=()
for ((i = 0; i < S1H_SPLIT; i++)); do
  DEVS=$(IFS=,; echo "${GPUS[*]:$((i * PER)):$PER}")
  OFF=$((S1H_PROMPT_OFFSET + i * S1H_N_PROMPTS))
  echo "block $i: GPUs $DEVS, prompts $S1H_N_PROMPTS@$OFF, out ${OUT_DIR}_$i"
  CUDA_VISIBLE_DEVICES=$DEVS python -u $DIR/run_s1h4.py --mode evaluate --preset "$S1H_PRESET" --ctx "$S1H_CTX" \
      --tasks "$S1H_TASKS" --n-prompts "$S1H_N_PROMPTS" --prompt-offset "$OFF" "${OVR[@]}" --out-dir "${OUT_DIR}_$i" \
      > "h0_measurement/logs/r14s1h4_${SLURM_JOB_ID:-manual}_$i.log" 2>&1 &
  PIDS+=($!)
done
RC=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then echo "block $i done"; else echo "ERROR: block $i failed" >&2; RC=1; fi
done
exit $RC
