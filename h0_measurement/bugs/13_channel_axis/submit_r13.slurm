#!/bin/bash
#SBATCH --job-name=sieve-r13-chan
#SBATCH --output=h0_measurement/logs/r13_%A_%a.out
#SBATCH --error=h0_measurement/logs/r13_%A_%a.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-5
#SBATCH --time=05:00:00
#
# R13 channel-axis worker (bugs/13_channel_axis/plan.md). Submit through
# bugs/13_channel_axis/script.sh. One array task = one cell = ONE process that
# measures every arm (raw channel, rotated channel, token axis) from the same
# captured q/K/V.
#   pilot (array 0-0): qwen3-1.7b @2048, cont, 1 prompt @2000 -- excluded check
#   main  (array 0-5): the six cells of plan.md section 3, prompts 2000-2003
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
cd "$PROJECT_ROOT"

PHASE="${1:-}"
[[ "$PHASE" == pilot || "$PHASE" == main ]] || { echo "ERROR: phase must be pilot or main" >&2; exit 2; }
[[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: run as a Slurm array task" >&2; exit 1; }
CELL="$SLURM_ARRAY_TASK_ID"

FAMILIES="niah,qa,cont"; N_PROMPTS=4; PROMPT_OFFSET=2000
if [[ "$PHASE" == pilot ]]; then
  [[ "$CELL" == 0 ]] || { echo "ERROR: pilot cell must be 0" >&2; exit 1; }
  MODEL="qwen3-1.7b"; CTX=2048; N_PROMPTS=1; FAMILIES="cont"
else
  case "$CELL" in
    0) MODEL="llama31-8b";       CTX=8192 ;;
    1) MODEL="llama31-8b";       CTX=32768 ;;
    2) MODEL="qwen3-8b";         CTX=8192 ;;
    3) MODEL="qwen3-8b";         CTX=32768 ;;
    4) MODEL="mistral-7b";       CTX=8192 ;;
    5) MODEL="qwen15-moe-a2.7b"; CTX=8192 ;;
    *) echo "ERROR: main cell must be 0-5" >&2; exit 1 ;;
  esac
fi

PY="$PROJECT_ROOT/.venv/bin/python"
DIR="$PROJECT_ROOT/h0_measurement/bugs/13_channel_axis"

export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export H0_CORPUS="$PROJECT_ROOT/.h0_corpus/pg19"
export H0_ALLOW_SYNTHETIC=0

case "$MODEL" in
  qwen3-1.7b)       MODEL_CACHE="models--Qwen--Qwen3-1.7B"; MODEL_REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e" ;;
  llama31-8b)       MODEL_CACHE="models--meta-llama--Llama-3.1-8B-Instruct"; MODEL_REVISION="0e9e39f249a16976918f6564b8830bc894c89659" ;;
  qwen3-8b)         MODEL_CACHE="models--Qwen--Qwen3-8B"; MODEL_REVISION="b968826d9c46dd6066d109eabc6255188de91218" ;;
  mistral-7b)       MODEL_CACHE="models--mistralai--Mistral-7B-Instruct-v0.3"; MODEL_REVISION="c170c708c41dac9275d15a8fff4eca08d52bab71" ;;
  qwen15-moe-a2.7b) MODEL_CACHE="models--Qwen--Qwen1.5-MoE-A2.7B-Chat"; MODEL_REVISION="ec052fda178e241c7c443468d2fa1db6618996be" ;;
esac
REF="$HF_HUB_CACHE/$MODEL_CACHE/refs/main"
[[ -f "$REF" && "$(tr -d '[:space:]' < "$REF")" == "$MODEL_REVISION" &&
   -d "$HF_HUB_CACHE/$MODEL_CACHE/snapshots/$MODEL_REVISION" ]] || {
  echo "ERROR: $MODEL_CACHE is not pinned at $MODEL_REVISION" >&2; exit 1; }

RUN_ID="r13_${PHASE}_${SLURM_ARRAY_JOB_ID}_${CELL}"
OUT_DIR="$PROJECT_ROOT/h0_measurement/results/$RUN_ID"
[[ ! -e "$OUT_DIR" ]] || { echo "ERROR: refusing to reuse $OUT_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"
sha256sum "$DIR"/r13lib.py "$DIR"/run_r13.py "$DIR"/read_r13.py "$DIR"/plan.md \
  h0_measurement/run_h0.py sievelib/quant.py sievelib/alloc.py sievelib/probe.py \
  sievelib/prompts.py > "$OUT_DIR/SOURCES.sha256"

echo "host=$(hostname) protocol=r13_channel_axis_v1 phase=$PHASE job=$SLURM_ARRAY_JOB_ID cell=$CELL model=$MODEL ctx=$CTX prompts=$N_PROMPTS@$PROMPT_OFFSET families=$FAMILIES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# No srun: one task, and job 995473 died before Python started on
# "srun: Unable to confirm allocation ... Socket timed out" (controller stall).
"$PY" -u "$DIR/run_r13.py" --model "$MODEL" --ctx "$CTX" \
  --n-prompts "$N_PROMPTS" --prompt-offset "$PROMPT_OFFSET" --families "$FAMILIES" \
  --out-dir "$OUT_DIR"

touch "$OUT_DIR/COMPLETE.pending"
"$PY" "$DIR/read_r13.py" --validate "$OUT_DIR"
mv "$OUT_DIR/COMPLETE.pending" "$OUT_DIR/COMPLETE"
echo "COMPLETE $RUN_ID"
