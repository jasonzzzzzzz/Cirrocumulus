#!/usr/bin/env bash
# R12-paper: matched H2O + key-quantization baselines, and the non-ceiling cell.
# See plan.md. Submit from trig-login01. Nothing runs without a mode.
#
#   bash script.sh --dry        print every sbatch line
#   bash script.sh --submit     submit A (5 cells) and B (calibration -> eval)
#   bash script.sh --submit --only=A|B
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
  # model:ctx:wall  (R9 walls, which covered 9 arms, with headroom for H2O's
  # extra prefill pass and four more arms)
  for cell in llama31-8b:8192:05:00:00 llama31-8b:32768:09:00:00 \
              llama31-8b:131072:23:00:00 qwen3-8b:8192:05:00:00 qwen3-8b:32768:09:00:00; do
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
  cal=$(sub --job-name=r12B-cal --time=02:00:00 "$WORKER" R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS=fp,uniform,evict,interior R8_N_PROMPTS=10 R8_PROMPT_OFFSET=1100 \
    R8_WRITE_ROUTES="$r" "${BK[@]}" "${COMMON[@]}")
  dep=()
  [[ "$cal" =~ ^[0-9]+$ ]] && dep=(--dependency=afterok:"$cal")
  echo "B  evaluation llama31-8b @32768 k32 prompts 60@1000 (after $cal)" >&2
  sub "${dep[@]}" --job-name=r12B-eval --time=08:00:00 "$WORKER" R8_MODEL=llama31-8b \
    R8_CTX=32768 R8_ARMS="$ARMS" R8_N_PROMPTS=60 R8_PROMPT_OFFSET=1000 R8_ROUTES="$r" \
    "${BK[@]}" "${COMMON[@]}" >/dev/null
fi
echo "results: h0_measurement/results/r12job<JOBID>/   logs: h0_measurement/logs/r8_<JOBID>.out" >&2
