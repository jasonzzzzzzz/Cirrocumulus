#!/usr/bin/env bash
# R14 Stage 1c: report.md Part D's three designs before any kernel -- a value-aware
# hybrid, a sequence-calibrated pooled router, dense storage with question-time
# reads -- each against TurboQuant-3 in its value lens, Llama-3.1-8B at 128K and
# 32K, 40 fresh prompts per cell. Design: s1c_lib.py. Frozen rules:
# read_stage1c.py (docstring). Worker: submit_s1c.slurm. No shared code edited.
#
# Run ON trig-login01 (GPU submits are refused elsewhere), from anywhere:
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1c.sh --preflight
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1c.sh --run-dry
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1c.sh --run
#
#   pilot (excluded)  128K: forced search on prompt 0 niah_single, then preset
#                     pilot128 (20 arms) on prompt 3102 niah_single            ~15-30 min
#   gate              read_stage1c.py --pilot (mechanics, memory, wall projection)
#   cal128, cal32     sequence-router search on prompts 0-9 (after the gate)   ~1-2 h / ~0.5 h
#   main128 x 4       prompts 5000-5039 in blocks of 10, 47 arms per prompt-task ~2.5-3 h each
#   main32  x 4       prompts 5100-5139 in blocks of 10, 47 arms               ~1 h each
#                     (main128 after gate + cal128; main32 after gate + cal32)
#   reader            read_stage1c.py -> bugs/14_kernel_tpot/stage1c.{json,md}
#
#   --status P G C128 C32 A1 A2 A3 A4 B1 B2 B3 B4 RD   /   --cancel (same)
#   --read A1 A2 A3 A4 B1 B2 B3 B4 C128 C32
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14_kernel_tpot
W=$DIR/submit_s1c.slurm
PY=.venv/bin/python
RT_S1=h0_measurement/results/r8_routes/r14_llama31-8b_131072_qa_b234.json
RT128_1B=h0_measurement/results/r8_routes/r14s1b_llama31-8b_131072_routes.json
RT32_1B=h0_measurement/results/r8_routes/r14s1b_llama31-8b_32768_routes.json
RT128_1C=h0_measurement/results/r8_routes/r14s1c_llama31-8b_131072_routes.json
RT32_1C=h0_measurement/results/r8_routes/r14s1c_llama31-8b_32768_routes.json
READER_SB="--partition=compute --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:30:00"
MODE="${1:-}"

need_id() { [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo "ERROR: job IDs must be decimal" >&2; exit 2; }; }

preflight() {
  [[ "$(hostname -s)" == trig-login01* ]] || echo "WARN: not on trig-login01; GPU sbatch will be refused here"
  export OMP_NUM_THREADS=8
  bash -n "$W"
  $PY -u $DIR/test_r14_stage1c.py --fast | tail -1
  $PY -u $DIR/test_r14_stage1b.py --fast | tail -1
  $PY -u tests/test_r8.py --fast | tail -1
  $PY h0_measurement/prefetch_corpus.py --verify --out .h0_corpus/pg19
  for f in "$RT_S1" "$RT128_1B" "$RT32_1B"; do
    [[ -f "$f" ]] || { echo "ERROR: routes $f missing" >&2; exit 1; }
  done
  for f in "$RT128_1C" "$RT32_1C"; do
    [[ ! -e "$f" ]] || { echo "ERROR: $f exists; refusing to overwrite a calibration" >&2; exit 1; }
  done
  mkdir -p h0_measurement/logs
  echo "R14 Stage 1c preflight passed"
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
  local P G C128 C32 RD o dep
  local -a A=() B=()
  P=$(sub --job-name=r14s1c-pilot --time=01:30:00 $W S1C_MODE=pilot S1C_TAG=pilot128 \
      S1C_PRESET=pilot128 S1C_CTX=131072 S1C_N_PROMPTS=1 S1C_PROMPT_OFFSET=3102 \
      S1C_TASKS=niah_single S1C_CAL_OFFSET=0 S1C_ROUTES_STD=$RT_S1 S1C_ROUTES_1B=$RT128_1B)
  echo "PILOT=$P" >&2
  G=$(sub --dependency=afterok:$P --job-name=r14s1c-gate $READER_SB \
      --output=h0_measurement/logs/r14s1cgate_%j.out --error=h0_measurement/logs/r14s1cgate_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1c.py --pilot $P")
  echo "GATE=$G" >&2
  C128=$(sub --dependency=afterok:$G --job-name=r14s1c-cal128 --time=06:00:00 $W \
      S1C_MODE=calibrate S1C_TAG=cal128 S1C_PRESET=main128 S1C_CTX=131072 S1C_N_PROMPTS=10 \
      S1C_PROMPT_OFFSET=0 S1C_ROUTES_1B=$RT128_1B S1C_WRITE_ROUTES=$RT128_1C)
  echo "CAL128=$C128" >&2
  C32=$(sub --dependency=afterok:$G --job-name=r14s1c-cal32 --time=03:00:00 $W \
      S1C_MODE=calibrate S1C_TAG=cal32 S1C_PRESET=main32 S1C_CTX=32768 S1C_N_PROMPTS=10 \
      S1C_PROMPT_OFFSET=0 S1C_ROUTES_1B=$RT32_1B S1C_WRITE_ROUTES=$RT32_1C)
  echo "CAL32=$C32" >&2
  for o in 5000 5010 5020 5030; do
    A+=("$(sub --dependency=afterok:$G:$C128 --job-name=r14s1c-m128-$o --time=10:00:00 $W \
        S1C_MODE=evaluate S1C_TAG=main128 S1C_PRESET=main128 S1C_CTX=131072 S1C_N_PROMPTS=10 \
        S1C_PROMPT_OFFSET=$o S1C_ROUTES_STD=$RT_S1 S1C_ROUTES_1B=$RT128_1B S1C_ROUTES_1C=$RT128_1C)")
  done
  echo "MAIN128=${A[*]}" >&2
  for o in 5100 5110 5120 5130; do
    B+=("$(sub --dependency=afterok:$G:$C32 --job-name=r14s1c-m32-$o --time=04:00:00 $W \
        S1C_MODE=evaluate S1C_TAG=main32 S1C_PRESET=main32 S1C_CTX=32768 S1C_N_PROMPTS=10 \
        S1C_PROMPT_OFFSET=$o S1C_ROUTES_1B=$RT32_1B S1C_ROUTES_1C=$RT32_1C)")
  done
  echo "MAIN32=${B[*]}" >&2
  dep=$(IFS=:; echo "${A[*]}:${B[*]}:$C128:$C32")
  RD=$(sub --dependency=afterany:$dep --job-name=r14s1c-read $READER_SB \
      --output=h0_measurement/logs/r14s1cread_%j.out --error=h0_measurement/logs/r14s1cread_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1c.py --main128 ${A[*]} --main32 ${B[*]} --cal128 $C128 --cal32 $C32")
  echo "READ=$RD" >&2
  echo "next: bash $DIR/script_stage1c.sh --status $P $G $C128 $C32 ${A[*]} ${B[*]} $RD"
}

case "$MODE" in
  --preflight) preflight ;;
  --run-dry)   preflight; chain ;;
  --run)       preflight; chain ;;
  --status|--cancel)
    (($# == 14)) || { echo "ERROR: $MODE needs P G C128 C32 A1 A2 A3 A4 B1 B2 B3 B4 RD" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    ids=$(IFS=,; echo "${*:2}")
    if [[ "$MODE" == --status ]]; then
      sacct -j "$ids" -X --format=JobID%12,JobName%18,State%24,Elapsed,ExitCode
    else
      scancel "${@:2}"
    fi ;;
  --read)
    (($# == 11)) || { echo "ERROR: --read needs A1 A2 A3 A4 B1 B2 B3 B4 C128 C32" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1c.py --main128 "$2" "$3" "$4" "$5" \
        --main32 "$6" "$7" "$8" "$9" --cal128 "${10}" --cal32 "${11}" ;;
  *) sed -n 2,25p "$0" >&2; exit 2 ;;
esac
