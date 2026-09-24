#!/bin/bash
#SBATCH --job-name=sieve-r11-nested
#SBATCH --output=h0_measurement/logs/r11nested_%A_%a.out
#SBATCH --error=h0_measurement/logs/r11nested_%A_%a.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-7
#SBATCH --time=05:00:00
#
# Frozen R11 nested-code worker (bugs/11_nested_code_overhead/plan.md).  Submit
# through bugs/11_nested_code_overhead/script.sh.  Even task = monolithic arm,
# odd task = nested arm of the same cell; the two arms differ ONLY in codebook.
#   pilot  (array 0-1): qwen3-1.7b @2048, cont, 1 prompt @0 -- excluded check
#   main   (array 0-7): four R10 cells x two arms, prompt block @24
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
cd "$PROJECT_ROOT"

PHASE="${1:-}"
PILOT_JOB="${2:-}"
case "$PHASE" in
  pilot) [[ $# -eq 1 ]] || { echo "ERROR: usage: $0 pilot" >&2; exit 2; } ;;
  main)  [[ $# -eq 2 && "$PILOT_JOB" =~ ^[0-9]+$ ]] || {
           echo "ERROR: usage: $0 main PILOT_JOB_ID" >&2; exit 2; } ;;
  *) echo "ERROR: phase must be pilot or main" >&2; exit 2 ;;
esac
[[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: run as a Slurm array task" >&2; exit 1; }
TASK="$SLURM_ARRAY_TASK_ID"

if [[ "$PHASE" == pilot ]]; then
  case "$TASK" in
    0|1) MODEL="qwen3-1.7b"; CTX=2048; N_PROMPTS=1; PROMPT_OFFSET=0 ;;
    *) echo "ERROR: pilot task must be 0 or 1" >&2; exit 1 ;;
  esac
  FAMILIES="cont"
else
  case "$TASK" in
    0|1) MODEL="llama31-8b";         CTX=32768;  N_PROMPTS=6; PROMPT_OFFSET=24 ;;
    2|3) MODEL="llama31-8b";         CTX=131072; N_PROMPTS=6; PROMPT_OFFSET=24 ;;
    4|5) MODEL="qwen3-8b";           CTX=8192;   N_PROMPTS=6; PROMPT_OFFSET=24 ;;
    6|7) MODEL="qwen3-30b-a3b-2507"; CTX=8192;   N_PROMPTS=4; PROMPT_OFFSET=24 ;;
    *) echo "ERROR: main task must be 0-7" >&2; exit 1 ;;
  esac
  FAMILIES="niah,qa,cont"
fi
if (( TASK % 2 == 0 )); then CODEBOOK="lloyd"; else CODEBOOK="nested3"; fi

PY="$PROJECT_ROOT/.venv/bin/python"
READER="$PROJECT_ROOT/h0_measurement/bugs/11_nested_code_overhead/read_nested_code.py"
LEDGER="$PROJECT_ROOT/h0_measurement/bugs/11_nested_code_overhead/source_ledger.json"

export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export H0_CORPUS="$PROJECT_ROOT/.h0_corpus/pg19"
export H0_ALLOW_SYNTHETIC=0

case "$MODEL" in
  qwen3-1.7b)         MODEL_CACHE="models--Qwen--Qwen3-1.7B"; MODEL_REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e" ;;
  llama31-8b)         MODEL_CACHE="models--meta-llama--Llama-3.1-8B-Instruct"; MODEL_REVISION="0e9e39f249a16976918f6564b8830bc894c89659" ;;
  qwen3-8b)           MODEL_CACHE="models--Qwen--Qwen3-8B"; MODEL_REVISION="b968826d9c46dd6066d109eabc6255188de91218" ;;
  qwen3-30b-a3b-2507) MODEL_CACHE="models--Qwen--Qwen3-30B-A3B-Instruct-2507"; MODEL_REVISION="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe" ;;
esac
REF="$HF_HUB_CACHE/$MODEL_CACHE/refs/main"
[[ -f "$REF" && "$(tr -d '[:space:]' < "$REF")" == "$MODEL_REVISION" &&
   -d "$HF_HUB_CACHE/$MODEL_CACHE/snapshots/$MODEL_REVISION" ]] || {
  echo "ERROR: $MODEL_CACHE is not pinned at $MODEL_REVISION" >&2; exit 1; }

"$PY" "$READER" --preflight
LEDGER_SHA="$(sha256sum "$LEDGER")"; LEDGER_SHA="${LEDGER_SHA%% *}"

if [[ "$PHASE" == main ]]; then
  LOCK="$PROJECT_ROOT/h0_measurement/bugs/11_nested_code_overhead/pilot_advance_lock_${PILOT_JOB}.json"
  "$PY" - "$LOCK" "$PILOT_JOB" "$LEDGER_SHA" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
need = {"protocol": "r11_nested_code_v1", "decision": "advance_main",
        "pilot_job": sys.argv[2], "source_ledger_sha256": sys.argv[3]}
bad = {k: (x.get(k), v) for k, v in need.items() if x.get(k) != v}
if bad:
    raise SystemExit(f"ERROR: pilot lock authentication failed: {bad}")
PY
fi

RUN_ID="r11_nested_${PHASE}_${SLURM_ARRAY_JOB_ID}_${TASK}"
OUT_DIR="$PROJECT_ROOT/h0_measurement/results/$RUN_ID"
[[ ! -e "$OUT_DIR" ]] || { echo "ERROR: refusing to reuse $OUT_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"

OVERRIDES=(
  "ctx=$CTX" "n_prompts=$N_PROMPTS" "prompt_offset=$PROMPT_OFFSET"
  "families=$FAMILIES" "n_decode=8" "quant_every=4" "rot_seed=2"
  "bit_list=1,2,3,4,5,6,8" "budgets=1,2,3,4" "extra_budgets=1,2,3,4"
  "evictors=oracle,accum" "corner_policies=frac" "interior_scores=accum"
  "group_alloc=true" "coarse_bits=4" "tier_panel=nested3"
  "codebook=$CODEBOOK"
)

"$PY" - "$OUT_DIR/RUN_INFO.json" "$PHASE" "$TASK" "$SLURM_ARRAY_JOB_ID" "$MODEL" \
  "$CTX" "$CODEBOOK" "$LEDGER_SHA" "$PILOT_JOB" "${OVERRIDES[@]}" <<'PY'
import json, sys
out, phase, task, job, model, ctx, cb, lsha, pilot, *ov = sys.argv[1:]
json.dump({"protocol": "r11_nested_code_v1", "phase": phase, "task": int(task),
           "job": job, "model": model, "ctx": int(ctx), "codebook": cb,
           "source_ledger_sha256": lsha, "pilot_job": pilot or None,
           "overrides": ov}, open(out, "w"), indent=2, sort_keys=True)
PY

echo "host=$(hostname) protocol=r11_nested_code_v1 phase=$PHASE job=$SLURM_ARRAY_JOB_ID task=$TASK model=$MODEL ctx=$CTX prompts=$N_PROMPTS@$PROMPT_OFFSET codebook=$CODEBOOK ledger=$LEDGER_SHA pilot=${PILOT_JOB:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun "$PY" h0_measurement/run_h0.py \
  --config h0_measurement/models.yaml --model "$MODEL" \
  --out-dir "$OUT_DIR" --override "${OVERRIDES[@]}"

# COMPLETE only after the reader re-opens and authenticates the artifact.
touch "$OUT_DIR/COMPLETE.pending"
"$PY" "$READER" --validate-task-dir "$OUT_DIR" --phase "$PHASE" --task "$TASK" \
  --job "$SLURM_ARRAY_JOB_ID"
mv "$OUT_DIR/COMPLETE.pending" "$OUT_DIR/COMPLETE"
echo "COMPLETE $RUN_ID codebook=$CODEBOOK"
