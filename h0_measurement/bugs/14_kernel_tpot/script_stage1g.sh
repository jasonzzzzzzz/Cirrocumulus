#!/usr/bin/env bash
# R14 Stage 1g: G1 (4-bit first tier), G2 (keys-only second tier), G3 (the question
# again over the fetched rows), G5 (read floor), the system that combines them, and
# G4 (the pipelined fetch and the time per token). Design: s1g_lib.py, s1g_kernel.py.
# Frozen rules: read_stage1g.py (docstring) and s1g_kernel.py (time).
# Workers: submit_s1g.slurm, submit_s1g_kernel.slurm. No shared code edited.
#
# Run where GPU sbatch works (Trillium: trig-login01 only). The project root is this
# file's ../../.. .
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1g.sh --preflight
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1g.sh --run-g-dry      | --run-g
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1g.sh --run-kernel-dry | --run-kernel
#
# --run-g (Llama-3.1-8B; no calibration, no routes):
#   gpilot (excl.)   128K prompt 3111, niah_multivalue + niah_single -> gate (A2 self-check per arm name)
#   g128 x 4         prompts 9100-9139 (17 arms)                             ~1.5-2 h each
#   gregress128      g128's plan on 8109, 8901, 8937 (niah_multikey)         ~15 min
#   g32  x 4         prompts 9200-9239 (20 arms; the floor r = 1/2 here)     ~40 min each
#   reader           read_stage1g.py -> bugs/14_kernel_tpot/stage1g.{json,md}
# --run-kernel:      one GPU: the 4-bit vote's correctness, then bench_s1g_kernel.py
#                    -> results/r14s1g_kernel_<job>/kernel_g_bench.{json,md}   (QUICK=1: mechanics only)
#
#   --status ID...   /   --cancel ID...   /   --read G1 G2 G3 G4 H1 H2 H3 H4 REG
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14_kernel_tpot
W=$DIR/submit_s1g.slurm
WK=$DIR/submit_s1g_kernel.slurm
PY=.venv/bin/python
GREG="8109:niah_multikey,8901:niah_multikey,8937:niah_multikey"     # s1g_lib.REGRESS_G128
READER_PART=""
[[ "$(hostname -s)" == trig* ]] && READER_PART="--partition=compute"
READER_SB="$READER_PART --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:30:00"
MODE="${1:-}"

need_id() { [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo "ERROR: job IDs must be decimal" >&2; exit 2; }; }

preflight() {
  [[ "$(hostname -s)" == trig-login01* || "$(hostname -s)" != trig* ]] \
    || echo "WARN: on Trillium, GPU sbatch only works from trig-login01"
  export OMP_NUM_THREADS=8
  bash -n "$W"
  bash -n "$WK"
  for t in test_r14_stage1g test_r14_stage1f; do
    $PY -u $DIR/$t.py --fast | tail -1
  done
  $PY -u tests/test_r8.py --fast | tail -1
  $PY h0_measurement/prefetch_corpus.py --verify --out .h0_corpus/pg19
  echo "R14 Stage 1g preflight passed"
}

SUBMITTED=()
sub() {   # sbatch, print only the job ID (or DRY)
  if [[ "$MODE" == *-dry ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2; echo DRY; return
  fi
  local out id
  out=$(sbatch --parsable "$@") || { echo "ERROR: sbatch failed after ${SUBMITTED[*]:-nothing}" >&2; exit 1; }
  id="${out%%;*}"; id="${id##* }"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "ERROR: no job id from: $out" >&2; exit 1; }
  SUBMITTED+=("$id"); echo "$id"
}

chain_g() {
  local P G REG RD o dep
  local -a A=() B=()
  P=$(sub --job-name=r14s1g-pilot --time=02:00:00 $W S1G_TAG=gpilot S1G_PRESET=gpilot S1G_CTX=131072 \
      S1G_N_PROMPTS=1 S1G_PROMPT_OFFSET=3111 S1G_TASKS=niah_multivalue,niah_single)
  G=$(sub --dependency=afterok:$P --job-name=r14s1g-gate $READER_SB \
      --output=h0_measurement/logs/r14s1ggate_%j.out --error=h0_measurement/logs/r14s1ggate_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1g.py --pilot $P --pilot-tag gpilot")
  echo "PILOT=$P GATE=$G" >&2
  for o in 9100 9110 9120 9130; do
    A+=("$(sub --dependency=afterok:$G --job-name=r14s1g-g128-$o --time=06:00:00 $W S1G_TAG=g128 S1G_PRESET=g128 \
        S1G_CTX=131072 S1G_N_PROMPTS=10 S1G_PROMPT_OFFSET=$o)")
  done
  REG=$(sub --dependency=afterok:$G --job-name=r14s1g-greg --time=01:30:00 $W S1G_TAG=gregress128 \
      S1G_PRESET=g128 S1G_CTX=131072 S1G_PROMPT_LIST=$GREG)
  for o in 9200 9210 9220 9230; do
    B+=("$(sub --dependency=afterok:$G --job-name=r14s1g-g32-$o --time=03:00:00 $W S1G_TAG=g32 S1G_PRESET=g32 \
        S1G_CTX=32768 S1G_N_PROMPTS=10 S1G_PROMPT_OFFSET=$o)")
  done
  echo "G128=${A[*]} GREGRESS=$REG G32=${B[*]}" >&2
  dep=$(IFS=:; echo "${A[*]}:$REG:${B[*]}")
  RD=$(sub --dependency=afterany:$dep --job-name=r14s1g-read $READER_SB \
      --output=h0_measurement/logs/r14s1gread_%j.out --error=h0_measurement/logs/r14s1gread_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1g.py --g128 ${A[*]} --g32 ${B[*]} \
--gregress $REG --out-stem $PROJECT_ROOT/$DIR/stage1g")
  echo "READ=$RD" >&2
  echo "next: bash $DIR/script_stage1g.sh --status $P $G ${A[*]} $REG ${B[*]} $RD"
}

chain_kernel() {
  local K
  K=$(sub --job-name=r14s1g-kernel $WK S1G_QUICK=${QUICK:-0})
  echo "KERNEL=$K" >&2
  echo "next: bash $DIR/script_stage1g.sh --status $K   (results: h0_measurement/results/r14s1g_kernel_$K/)"
}

case "$MODE" in
  --preflight) preflight ;;
  --run-g-dry|--run-g) preflight; chain_g ;;
  --run-kernel-dry|--run-kernel) bash -n "$WK"; chain_kernel ;;
  --status|--cancel)
    (($# >= 2)) || { echo "ERROR: $MODE needs job IDs" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    ids=$(IFS=,; echo "${*:2}")
    if [[ "$MODE" == --status ]]; then
      sacct -j "$ids" -X --format=JobID%12,JobName%20,State%24,Elapsed,ExitCode
    else
      scancel "${@:2}"
    fi ;;
  --read)
    (($# == 10)) || { echo "ERROR: --read needs G1 G2 G3 G4 H1 H2 H3 H4 REG" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1g.py --g128 "$2" "$3" "$4" "$5" --g32 "$6" "$7" "$8" "$9" \
        --gregress "${10}" --out-stem $DIR/stage1g ;;
  *) sed -n 2,30p "$0" >&2; exit 2 ;;
esac
