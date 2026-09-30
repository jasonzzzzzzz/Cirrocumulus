#!/usr/bin/env bash
# R14 Stage 1b: hybrids, value twins, pooled / half-bit routers, teacher-forced
# replay -- Llama-3.1-8B at 128K and 32K. Design: s1b_lib.py. Frozen rules:
# read_stage1b.py (docstring). Worker: submit_s1b.slurm. No shared code edited.
#
# Run ON trig-login01 (GPU submits are refused elsewhere), from anywhere:
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1b.sh --preflight
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1b.sh --run-dry
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1b.sh --run
#
#   pilot (excluded)  128K, 1 prompt @3101, niah_single, preset pilot128   ~20 min
#   gate              read_stage1b.py --pilot (mechanics, memory, wall projection)
#   cal128, cal32     both routers' routes on prompts 0-9 (start at once)  ~1 h / ~15 min
#   main128 A, B      prompts 4000-4009 / 4010-4019, 33 arms per prompt-task  ~3-4 h each
#   main32  A, B      prompts 4100-4109 / 4110-4119, 32 arms                   ~1 h each
#                     (main128 after gate + cal128; main32 after gate + cal32)
#   reader            read_stage1b.py -> bugs/14_kernel_tpot/stage1b.{json,md}
#
#   --status P G C128 C32 A B C D RD   /   --cancel (same)   /   --read A B C D
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14_kernel_tpot
W=$DIR/submit_s1b.slurm
PY=.venv/bin/python
RT_S1=h0_measurement/results/r8_routes/r14_llama31-8b_131072_qa_b234.json
RT128=h0_measurement/results/r8_routes/r14s1b_llama31-8b_131072_routes.json
RT32=h0_measurement/results/r8_routes/r14s1b_llama31-8b_32768_routes.json
READER_SB="--partition=compute --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:30:00"
MODE="${1:-}"

need_id() { [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo "ERROR: job IDs must be decimal" >&2; exit 2; }; }

preflight() {
  [[ "$(hostname -s)" == trig-login01* ]] || echo "WARN: not on trig-login01; GPU sbatch will be refused here"
  export OMP_NUM_THREADS=8
  bash -n "$W"
  $PY -u $DIR/test_r14_stage1b.py --fast | tail -1
  $PY -u $DIR/test_r14_stage1.py --fast | tail -1
  $PY -u tests/test_r8.py --fast | tail -1
  $PY h0_measurement/prefetch_corpus.py --verify --out .h0_corpus/pg19
  [[ -f "$RT_S1" ]] || { echo "ERROR: Stage 1 routes $RT_S1 missing" >&2; exit 1; }
  for f in "$RT128" "$RT32"; do
    [[ ! -e "$f" ]] || { echo "ERROR: $f exists; refusing to overwrite a calibration" >&2; exit 1; }
  done
  mkdir -p h0_measurement/logs
  echo "R14 Stage 1b preflight passed"
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
  local P G C128 C32 A B C D RD
  P=$(sub --job-name=r14s1b-pilot --time=01:00:00 $W S1B_MODE=evaluate S1B_TAG=pilot128 \
      S1B_PRESET=pilot128 S1B_CTX=131072 S1B_N_PROMPTS=1 S1B_PROMPT_OFFSET=3101 \
      S1B_TASKS=niah_single S1B_ROUTES_STD=$RT_S1)
  echo "PILOT=$P" >&2
  G=$(sub --dependency=afterok:$P --job-name=r14s1b-gate $READER_SB \
      --output=h0_measurement/logs/r14s1bgate_%j.out --error=h0_measurement/logs/r14s1bgate_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1b.py --pilot $P")
  echo "GATE=$G" >&2
  C128=$(sub --job-name=r14s1b-cal128 --time=03:00:00 $W S1B_MODE=calibrate S1B_TAG=cal128 \
      S1B_PRESET=main128 S1B_CTX=131072 S1B_N_PROMPTS=10 S1B_PROMPT_OFFSET=0 \
      S1B_WRITE_ROUTES=$RT128)
  echo "CAL128=$C128" >&2
  C32=$(sub --job-name=r14s1b-cal32 --time=01:00:00 $W S1B_MODE=calibrate S1B_TAG=cal32 \
      S1B_PRESET=main32 S1B_CTX=32768 S1B_N_PROMPTS=10 S1B_PROMPT_OFFSET=0 \
      S1B_WRITE_ROUTES=$RT32)
  echo "CAL32=$C32" >&2
  A=$(sub --dependency=afterok:$G:$C128 --job-name=r14s1b-m128-4000 --time=10:00:00 $W \
      S1B_MODE=evaluate S1B_TAG=main128 S1B_PRESET=main128 S1B_CTX=131072 S1B_N_PROMPTS=10 \
      S1B_PROMPT_OFFSET=4000 S1B_ROUTES_STD=$RT_S1 S1B_ROUTES_1B=$RT128)
  B=$(sub --dependency=afterok:$G:$C128 --job-name=r14s1b-m128-4010 --time=10:00:00 $W \
      S1B_MODE=evaluate S1B_TAG=main128 S1B_PRESET=main128 S1B_CTX=131072 S1B_N_PROMPTS=10 \
      S1B_PROMPT_OFFSET=4010 S1B_ROUTES_STD=$RT_S1 S1B_ROUTES_1B=$RT128)
  echo "MAIN128=$A,$B" >&2
  C=$(sub --dependency=afterok:$G:$C32 --job-name=r14s1b-m32-4100 --time=04:00:00 $W \
      S1B_MODE=evaluate S1B_TAG=main32 S1B_PRESET=main32 S1B_CTX=32768 S1B_N_PROMPTS=10 \
      S1B_PROMPT_OFFSET=4100 S1B_ROUTES_1B=$RT32)
  D=$(sub --dependency=afterok:$G:$C32 --job-name=r14s1b-m32-4110 --time=04:00:00 $W \
      S1B_MODE=evaluate S1B_TAG=main32 S1B_PRESET=main32 S1B_CTX=32768 S1B_N_PROMPTS=10 \
      S1B_PROMPT_OFFSET=4110 S1B_ROUTES_1B=$RT32)
  echo "MAIN32=$C,$D" >&2
  RD=$(sub --dependency=afterany:$A:$B:$C:$D --job-name=r14s1b-read $READER_SB \
      --output=h0_measurement/logs/r14s1bread_%j.out --error=h0_measurement/logs/r14s1bread_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1b.py --main128 $A $B --main32 $C $D")
  echo "READ=$RD" >&2
  echo "next: bash $DIR/script_stage1b.sh --status $P $G $C128 $C32 $A $B $C $D $RD"
}

case "$MODE" in
  --preflight) preflight ;;
  --run-dry)   preflight; chain ;;
  --run)       preflight; chain ;;
  --status|--cancel)
    (($# == 10)) || { echo "ERROR: $MODE needs P G C128 C32 A B C D RD" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    ids=$(IFS=,; echo "${*:2}")
    if [[ "$MODE" == --status ]]; then
      sacct -j "$ids" -X --format=JobID%12,JobName%18,State%24,Elapsed,ExitCode
    else
      scancel "${@:2}"
    fi ;;
  --read)
    (($# == 5)) || { echo "ERROR: --read needs MAIN128_A MAIN128_B MAIN32_A MAIN32_B" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1b.py --main128 "$2" "$3" --main32 "$4" "$5" ;;
  *) sed -n 2,24p "$0" >&2; exit 2 ;;
esac
