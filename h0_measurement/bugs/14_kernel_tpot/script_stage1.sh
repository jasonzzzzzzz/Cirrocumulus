#!/usr/bin/env bash
# R14 Stage 1: Llama-3.1-8B @128K at B = 2, 3, 4, FP8 KV, FP8-value pairs.
# Protocol and frozen decisions: plan.md §5 and amendment A1 (written before any
# Stage 1 output). Worker: h0_measurement/submit_r8.slurm (R14 options additive).
#
# Run ON trig-login01 (GPU submits are refused elsewhere), from anywhere:
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1.sh --preflight   # CPU tests, corpus, test-only
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1.sh --run-dry     # print the sbatch lines
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1.sh --run         # submit the chain
#
#   pilot (excluded)  1 prompt @3100, niah_single, fp/uniform/kivi_g128/router_oracle,
#                     B=3,4, every R14 arm                               ~15 min
#   gate              read_stage1.py --pilot: mechanics, peak memory, wall projection
#   calibration       routes at B=2,3,4 on prompts 0-9 (this cluster's corpus)
#                     -> results/r8_routes/r14_llama31-8b_131072_qa_b234.json   ~3 h
#   main A, main B    prompts 100-109 and 110-119, all arms in one process    ~4.5 h each
#                     (after BOTH the gate and the calibration succeed)
#   reader            read_stage1.py --main A B -> bugs/14_kernel_tpot/stage1.{json,md}
#
# The calibration does not use the R14 options (it is the unchanged R12 path),
# so it starts at once, in parallel with the pilot.
#   --status P G CAL A B RD   /   --cancel P G CAL A B RD   /   --read A B
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14_kernel_tpot
W=h0_measurement/submit_r8.slurm
PY=.venv/bin/python
ROUTES=h0_measurement/results/r8_routes/r14_llama31-8b_131072_qa_b234.json
TASKS4=R8_TASKS=niah_single,niah_multikey,niah_multivalue,vt
COMMON=(R8_MODEL=llama31-8b R8_CTX=131072 R8_WINDOW=32 R8_QA=1 R8_THETA=1.0 R8_HEAD_ERROR=1)
R14=(R8_FP8KV=1)
READER_SB="--partition=compute --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:30:00"
MODE="${1:-}"

need_id() { [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo "ERROR: job IDs must be decimal" >&2; exit 2; }; }

preflight() {
  [[ "$(hostname -s)" == trig-login01* ]] || echo "WARN: not on trig-login01; GPU sbatch will be refused here"
  export OMP_NUM_THREADS=8
  bash -n "$W"
  $PY -u $DIR/test_r14_stage1.py | tail -1
  $PY -u $DIR/test_r14.py | tail -1
  $PY -u tests/test_r8.py --fast | tail -1
  $PY -u tests/test_kv_quant_baselines.py --fast | tail -1
  $PY h0_measurement/prefetch_corpus.py --verify --out .h0_corpus/pg19
  [[ ! -e "$ROUTES" ]] || { echo "ERROR: $ROUTES exists; refusing to overwrite a calibration" >&2; exit 1; }
  mkdir -p h0_measurement/logs
  echo "R14 Stage 1 preflight passed"
}

SUBMITTED=()
sub() {   # sbatch, print only the job ID (or DRY)
  if [[ "$MODE" == --run-dry ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2; echo DRY; return
  fi
  local out id
  out=$(sbatch --parsable "$@") || { echo "ERROR: sbatch failed after ${SUBMITTED[*]:-nothing}" >&2; exit 1; }
  id="${out%%;*}"; id="${id##* }"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "ERROR: no job id from: $out" >&2; exit 1; }
  SUBMITTED+=("$id"); echo "$id"
}

chain() {
  local P G CAL A B RD
  P=$(sub --job-name=r14s1-pilot --time=00:50:00 $W "${COMMON[@]}" "${R14[@]}" \
      R8_RUN_PREFIX=r14s1pilot R8_TASKS=niah_single R8_N_PROMPTS=1 R8_PROMPT_OFFSET=3100 \
      R8_ARMS=fp,uniform,kivi_g128,router_oracle R8_BUDGETS=3,4 R8_V_FP8_BUDGETS=3,4 \
      R8_V_FP8_ARMS=fp,uniform,kivi_g128,router_oracle)
  echo "PILOT=$P" >&2
  G=$(sub --dependency=afterok:$P --job-name=r14s1-gate $READER_SB \
      --output=h0_measurement/logs/r14s1gate_%j.out --error=h0_measurement/logs/r14s1gate_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1.py --pilot $P")
  echo "GATE=$G" >&2
  CAL=$(sub --job-name=r14s1-cal --time=06:00:00 $W "${COMMON[@]}" $TASKS4 \
      R8_RUN_PREFIX=r14s1cal R8_N_PROMPTS=10 R8_PROMPT_OFFSET=0 \
      R8_ARMS=fp,uniform,evict,interior R8_BUDGETS=2,3,4 R8_WRITE_ROUTES=$ROUTES)
  echo "CAL=$CAL" >&2
  local ids=()
  for off in 100 110; do
    ids+=("$(sub --dependency=afterok:$G:$CAL --job-name=r14s1-main-$off --time=10:00:00 $W \
      "${COMMON[@]}" "${R14[@]}" $TASKS4 R8_RUN_PREFIX=r14s1job R8_N_PROMPTS=10 \
      R8_PROMPT_OFFSET=$off R8_BUDGETS=2,3,4 R8_ROUTES=$ROUTES \
      R8_ARMS=fp,uniform,kivi_g128,kivi,kvquant,router_calib,router_oracle \
      R8_V_FP8_ARMS=fp,uniform,kivi_g128,router_calib R8_V_FP8_BUDGETS=2,3,4)")
  done
  A=${ids[0]}; B=${ids[1]}
  echo "MAIN_A=$A MAIN_B=$B" >&2
  RD=$(sub --dependency=afterany:$A:$B --job-name=r14s1-read $READER_SB \
      --output=h0_measurement/logs/r14s1read_%j.out --error=h0_measurement/logs/r14s1read_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1.py --main $A $B")
  echo "READ=$RD" >&2
  echo "next: bash $DIR/script_stage1.sh --status $P $G $CAL $A $B $RD"
}

case "$MODE" in
  --preflight) preflight ;;
  --run-dry)   preflight; chain ;;
  --run)       preflight; chain ;;
  --status|--cancel)
    (($# == 7)) || { echo "ERROR: $MODE needs PILOT GATE CAL MAIN_A MAIN_B READ" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    ids=$(IFS=,; echo "${*:2}")
    if [[ "$MODE" == --status ]]; then
      sacct -j "$ids" -X --format=JobID%12,JobName%18,State%24,Elapsed,ExitCode
    else
      scancel "${@:2}"
    fi ;;
  --read)
    (($# == 3)) || { echo "ERROR: --read needs MAIN_A MAIN_B" >&2; exit 2; }
    need_id "$2"; need_id "$3"
    OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1.py --main "$2" "$3" ;;
  *) sed -n 2,26p "$0" >&2; exit 2 ;;
esac
