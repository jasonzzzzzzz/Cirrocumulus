#!/bin/bash
#SBATCH --job-name=sieve-r11x
#SBATCH --output=h0_measurement/logs/r11x_%A_%a.out
#SBATCH --error=h0_measurement/logs/r11x_%A_%a.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-9
#SBATCH --time=04:00:00
#
# Frozen R11-ext worker (bugs/11_nested_code_overhead/plan_ext.md).  Submit
# through bugs/11_nested_code_overhead/script_ext.sh.  Same in-process A/B
# layout as R11 A2: one array task = ONE run_h0 process measuring both codebooks
# from the same captured q/K/V.  A cell may span several tasks (disjoint prompt
# blocks).
#   pilot  (array 0-1): mistral-7b / qwen15-moe @2048, cont, 1 prompt @0
#   main   (array 0-9): X1 qwen3-30b @8K x6 tasks, X2 qwen15-moe @8K x2,
#                       X3 mistral-7b @8K, X4 mistral-7b @32K
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
cd "$PROJECT_ROOT"

PHASE="${1:-}"
PILOT_JOB="${2:-}"
case "$PHASE" in
  pilot) [[ $# -eq 1 ]] || { echo "ERROR: usage: $0 pilot" >&2; exit 2; } ;;
  # main no longer waits on a pilot lock (2026-09-26): pilot 991618 passed
  # (pilot_ext_advance_lock_991618.json), and the lock name the worker looked
  # for never matched the one the reader writes. An optional pilot ID is kept
  # only as provenance in RUN_INFO.
  main)  [[ $# -le 2 && ( -z "$PILOT_JOB" || "$PILOT_JOB" =~ ^[0-9]+$ ) ]] || {
           echo "ERROR: usage: $0 main [PILOT_JOB_ID]" >&2; exit 2; } ;;
  *) echo "ERROR: phase must be pilot or main" >&2; exit 2 ;;
esac
[[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: run as a Slurm array task" >&2; exit 1; }
TASK="$SLURM_ARRAY_TASK_ID"

if [[ "$PHASE" == pilot ]]; then
  case "$TASK" in
    0) MODEL="mistral-7b";       CTX=2048; N_PROMPTS=1; PROMPT_OFFSET=0 ;;
    1) MODEL="qwen15-moe-a2.7b"; CTX=2048; N_PROMPTS=1; PROMPT_OFFSET=0 ;;
    *) echo "ERROR: pilot task must be 0-1" >&2; exit 1 ;;
  esac
  FAMILIES="cont"
else
  case "$TASK" in
    0|1|2|3|4|5) MODEL="qwen3-30b-a3b-2507"; CTX=8192; N_PROMPTS=4; PROMPT_OFFSET=$((28 + 4 * TASK)) ;;
    6|7) MODEL="qwen15-moe-a2.7b"; CTX=8192;  N_PROMPTS=6; PROMPT_OFFSET=$((24 + 6 * (TASK - 6))) ;;
    8)   MODEL="mistral-7b";       CTX=8192;  N_PROMPTS=6; PROMPT_OFFSET=24 ;;
    9)   MODEL="mistral-7b";       CTX=32768; N_PROMPTS=6; PROMPT_OFFSET=24 ;;
    *) echo "ERROR: main task must be 0-9" >&2; exit 1 ;;
  esac
  FAMILIES="niah,qa,cont"
fi

PY="$PROJECT_ROOT/.venv/bin/python"
READER="$PROJECT_ROOT/h0_measurement/bugs/11_nested_code_overhead/read_nested_ext.py"
LEDGER="$PROJECT_ROOT/h0_measurement/bugs/11_nested_code_overhead/source_ledger_ext.json"

export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export H0_CORPUS="$PROJECT_ROOT/.h0_corpus/pg19"
export H0_ALLOW_SYNTHETIC=0

case "$MODEL" in
  mistral-7b)         MODEL_CACHE="models--mistralai--Mistral-7B-Instruct-v0.3"; MODEL_REVISION="c170c708c41dac9275d15a8fff4eca08d52bab71" ;;
  qwen15-moe-a2.7b)   MODEL_CACHE="models--Qwen--Qwen1.5-MoE-A2.7B-Chat"; MODEL_REVISION="ec052fda178e241c7c443468d2fa1db6618996be" ;;
  qwen3-30b-a3b-2507) MODEL_CACHE="models--Qwen--Qwen3-30B-A3B-Instruct-2507"; MODEL_REVISION="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe" ;;
esac
REF="$HF_HUB_CACHE/$MODEL_CACHE/refs/main"
[[ -f "$REF" && "$(tr -d '[:space:]' < "$REF")" == "$MODEL_REVISION" &&
   -d "$HF_HUB_CACHE/$MODEL_CACHE/snapshots/$MODEL_REVISION" ]] || {
  echo "ERROR: $MODEL_CACHE is not pinned at $MODEL_REVISION" >&2; exit 1; }

"$PY" "$READER" --preflight
LEDGER_SHA="$(sha256sum "$LEDGER")"; LEDGER_SHA="${LEDGER_SHA%% *}"


RUN_ID="r11x_${PHASE}_${SLURM_ARRAY_JOB_ID}_${TASK}"
OUT_DIR="$PROJECT_ROOT/h0_measurement/results/$RUN_ID"
[[ ! -e "$OUT_DIR" ]] || { echo "ERROR: refusing to reuse $OUT_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"

OVERRIDES=(
  "ctx=$CTX" "n_prompts=$N_PROMPTS" "prompt_offset=$PROMPT_OFFSET"
  "families=$FAMILIES" "n_decode=8" "quant_every=4" "rot_seed=2"
  "bit_list=1,2,3,4,5,6,8" "budgets=1,2,3,4" "extra_budgets=1,2,3,4"
  "evictors=oracle,accum" "corner_policies=frac" "interior_scores=accum"
  "group_alloc=true" "coarse_bits=4" "tier_panel=nested3"
  "codebook=lloyd" "codebook_ab=nested3"
)

"$PY" - "$OUT_DIR/RUN_INFO.json" "$PHASE" "$TASK" "$SLURM_ARRAY_JOB_ID" "$MODEL" \
  "$CTX" "$LEDGER_SHA" "$PILOT_JOB" "${OVERRIDES[@]}" <<'PY'
import json, sys
out, phase, task, job, model, ctx, lsha, pilot, *ov = sys.argv[1:]
json.dump({"protocol": "r11_nested_code_ext_v1", "layout": "in_process_ab",
           "phase": phase, "task": int(task), "job": job, "model": model,
           "ctx": int(ctx), "codebook": "lloyd", "codebook_ab": "nested3",
           "source_ledger_sha256": lsha, "pilot_job": pilot or None,
           "overrides": ov}, open(out, "w"), indent=2, sort_keys=True)
PY

echo "host=$(hostname) protocol=r11_nested_code_ext_v1 layout=in_process_ab phase=$PHASE job=$SLURM_ARRAY_JOB_ID task=$TASK model=$MODEL ctx=$CTX prompts=$N_PROMPTS@$PROMPT_OFFSET codebook=lloyd codebook_ab=nested3 ledger=$LEDGER_SHA pilot=${PILOT_JOB:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" h0_measurement/run_h0.py \
  --config h0_measurement/models.yaml --model "$MODEL" \
  --out-dir "$OUT_DIR" --override "${OVERRIDES[@]}"

# COMPLETE only after the reader re-opens the artifact, splits the two
# codebooks and checks V1-V3 on this cell.
touch "$OUT_DIR/COMPLETE.pending"
"$PY" "$READER" --validate-task-dir "$OUT_DIR" --phase "$PHASE" --task "$TASK" \
  --job "$SLURM_ARRAY_JOB_ID"
mv "$OUT_DIR/COMPLETE.pending" "$OUT_DIR/COMPLETE"
echo "COMPLETE $RUN_ID layout=in_process_ab"
