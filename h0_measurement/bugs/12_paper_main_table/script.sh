#!/usr/bin/env bash
# R12-paper: matched H2O + key-quantization baselines, and the non-ceiling cell.
# See plan.md. Submit from trig-login01. Nothing runs without a mode.
#
#   bash script.sh --dry        print every sbatch line
#   bash script.sh --submit     submit A (5 cells) and B (calibration -> eval)
#   bash script.sh --submit --only=A|B|C|D   (C, D only when named)
set -euo pipefail

MODE="" ONLY=""
for arg in "$@"; do
  case "$arg" in
    --dry) MODE=dry ;;
    --submit) MODE=submit ;;
    --only=*) ONLY="${arg#--only=}" ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done
[[ -n "$MODE" ]] || { sed -n 2,8p "$0"; exit 0; }

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant}"
cd "$PROJECT_ROOT"
WORKER=h0_measurement/submit_r8.slurm
ROUTES=h0_measurement/results/r8_routes
mkdir -p h0_measurement/logs

# interior_pool and router_oracle are DIAGNOSTICS for the failure case (plan.md
# "Failure-case diagnostics"), not methods in the main table
ARMS="fp,uniform,evict,evict_h2o,interior_cascade,router_calib,adakv,dropkv,obcache_k:alloc=ada@obck_ada,laprox,kivi,kivi_g128,kvquant,interior_pool,router_oracle"
COMMON=(R8_RUN_PREFIX=r12job R8_BUDGETS=2,3 R8_WINDOW=32 R8_QA=1 R8_THETA=1.0 R8_HEAD_ERROR=1)

sub() {   # sub <sbatch args...> ; prints job id
  if [[ "$MODE" == dry ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2; echo DRY; return
  fi
  local out; out=$(sbatch --parsable "$@"); echo "$out" >&2; echo "${out%%;*}"
}

if [[ -z "$ONLY" || "$ONLY" == A ]]; then
  # model:ctx:wall. About 2x the runtime projected from R9's per-phase timings
  # (H100): decode x1.7 for 29 vs 17 decodes per unit, allocation x1.6, plus
  # H2O's quadratic prefill capture (~+3.3 h at 128K, the least certain term).
  # Projected: 0.9 / 1.7 / 7.5 / 0.9 / 1.4 h. run_r8.py writes results only at
  # the end, so a timeout loses the whole cell; scale up on slower GPUs.
  for cell in llama31-8b:8192:02:00:00 llama31-8b:32768:03:30:00 \
              llama31-8b:131072:14:00:00 qwen3-8b:8192:02:00:00 qwen3-8b:32768:03:00:00; do
    IFS=: read -r m c h mi s <<< "$cell"
    r="$ROUTES/${m}_${c}_qa.json"
    [[ -f "$r" ]] || { echo "missing routes $r" >&2; exit 1; }
    echo "A  $m @$c prompts 20@100" >&2
    sub --job-name="r12A-$m-$c" --time="$h:$mi:$s" "$WORKER" R8_MODEL="$m" R8_CTX="$c" \
      R8_ARMS="$ARMS" R8_N_PROMPTS=20 R8_PROMPT_OFFSET=100 R8_ROUTES="$r" \
      R8_TASKS=niah_single,niah_multikey,niah_multivalue,vt "${COMMON[@]}" >/dev/null
  done
fi

if [[ -z "$ONLY" || "$ONLY" == B ]]; then
  r="$ROUTES/llama31-8b_32768_k32_v4_h4_tasks_niah_multikey_qa.json"
  BK=(R8_N_KEYS=32 R8_N_VALUES=4 R8_N_HOPS=4 R8_TASKS=niah_multikey)
  echo "B  calibration llama31-8b @32768 k32 prompts 10@1100 -> $r" >&2
  cal=$(sub --job-name=r12B-cal --time=00:30:00 "$WORKER" R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS=fp,uniform,evict,interior R8_N_PROMPTS=10 R8_PROMPT_OFFSET=1100 \
    R8_WRITE_ROUTES="$r" "${BK[@]}" "${COMMON[@]}")
  dep=()
  [[ "$cal" =~ ^[0-9]+$ ]] && dep=(--dependency=afterok:"$cal")
  echo "B  evaluation llama31-8b @32768 k32 prompts 60@1000 (after $cal)" >&2
  sub "${dep[@]}" --job-name=r12B-eval --time=03:00:00 "$WORKER" R8_MODEL=llama31-8b \
    R8_CTX=32768 R8_ARMS="$ARMS" R8_N_PROMPTS=60 R8_PROMPT_OFFSET=1000 R8_ROUTES="$r" \
    "${BK[@]}" "${COMMON[@]}" >/dev/null
fi
# C and D run only when named explicitly (plan.md addendum, 2026-09-26)
if [[ "$ONLY" == C ]]; then
  # question-visible control on the A prompts; P0 path, no head errors
  for cell in llama31-8b:8192:01:00:00 llama31-8b:32768:01:30:00 \
              llama31-8b:131072:05:00:00 qwen3-8b:8192:01:00:00 qwen3-8b:32768:01:30:00; do
    IFS=: read -r m c h mi s <<< "$cell"
    echo "C  $m @$c question-visible prompts 20@100" >&2
    sub --job-name="r12C-$m-$c" --time="$h:$mi:$s" "$WORKER" R8_MODEL="$m" R8_CTX="$c" \
      R8_ARMS=fp,uniform,evict,evict_h2o R8_BUDGETS=2,3 R8_N_PROMPTS=20 R8_PROMPT_OFFSET=100 \
      R8_WINDOW=32 R8_QA=0 R8_RUN_PREFIX=r12qvjob \
      R8_TASKS=niah_single,niah_multikey,niah_multivalue,vt >/dev/null
  done
fi

if [[ "$ONLY" == D ]]; then
  # Mistral-7B: calibrate routes (prompts 0-9), then evaluate A's arms (100-119)
  for cell in 8192:00:45:00:02:00:00 32768:01:00:00:03:30:00; do
    IFS=: read -r c ch cm cs eh em es <<< "$cell"
    r="$ROUTES/mistral-7b_${c}_qa.json"
    echo "D  calibration mistral-7b @$c prompts 10@0 -> $r" >&2
    cal=$(sub --job-name="r12D-cal-$c" --time="$ch:$cm:$cs" "$WORKER" R8_MODEL=mistral-7b R8_CTX="$c" \
      R8_ARMS=fp,uniform,evict,interior R8_N_PROMPTS=10 R8_PROMPT_OFFSET=0 R8_WRITE_ROUTES="$r" \
      R8_TASKS=niah_single,niah_multikey,niah_multivalue,vt "${COMMON[@]}")
    dep=()
    [[ "$cal" =~ ^[0-9]+$ ]] && dep=(--dependency=afterok:"$cal")
    echo "D  evaluation mistral-7b @$c prompts 20@100 (after $cal)" >&2
    sub "${dep[@]}" --job-name="r12D-eval-$c" --time="$eh:$em:$es" "$WORKER" R8_MODEL=mistral-7b \
      R8_CTX="$c" R8_ARMS="$ARMS" R8_N_PROMPTS=20 R8_PROMPT_OFFSET=100 R8_ROUTES="$r" \
      R8_TASKS=niah_single,niah_multikey,niah_multivalue,vt "${COMMON[@]}" >/dev/null
  done
fi
echo "results: h0_measurement/results/r12job<JOBID>/   logs: h0_measurement/logs/r8_<JOBID>.out" >&2
