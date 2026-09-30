#!/usr/bin/env bash
# R14 Stage 1d: the fixed metric (answer-value NLL), the second-round designs
# (head-aware value-aware hybrid, head-budget routers, question-time reads with
# protected heads and re-selection), the setup-head mechanism arms, and the
# Qwen3-30B-A3B-2507 replication. Design: s1d_lib.py. Frozen rules:
# read_stage1d.py (docstring). Worker: submit_s1d.slurm. No shared code edited.
#
# Run ON trig-login01 (GPU submits are refused elsewhere), from anywhere:
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1d.sh --preflight
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1d.sh --run-dry | --run            (Llama)
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1d.sh --run-qwen-dry | --run-qwen  (Qwen)
#
# Llama-3.1-8B chain (--run):
#   pilot (excluded)  128K: forced search on prompt 0, then preset pilot128 on prompt 3103  ~20-30 min
#   gate              read_stage1d.py --pilot
#   cal128, cal32     seq2 calibration (answer-value statistic) on prompts 0-9           ~0.5-2 h each
#   main128 x 4       prompts 7000-7039, 36 arms per prompt-task                          ~3 h each
#   main32  x 4       prompts 7100-7139, 36 arms                                          ~1 h each
#   reader            read_stage1d.py -> bugs/14_kernel_tpot/stage1d.{json,md}
# Qwen3-30B-A3B-2507 chain (--run-qwen; 32K on one H100):
#   qr0-32, qr0-8     Qwen's own pooled routes (Stage 1b driver, unchanged), prompts 0-9 at 32K / 8K
#   qpilot (excl.)    32K: forced search on prompt 0, then preset qwenpilot on prompt 3103
#   qgate             read_stage1d.py --pilot --pilot-tag qwenpilot
#   qcal32, qcal8     seq2 calibration at 32K and 8K (8K: calibration only, for the length test)
#   qwen32 x 4        prompts 7100-7139 (Llama's 32K indices), 15 arms
#   qreader           read_stage1d.py -> bugs/14_kernel_tpot/stage1d_qwen.{json,md}
#
#   --status ID...   /   --cancel ID...
#   --read A1 A2 A3 A4 B1 B2 B3 B4 C128 C32   /   --read-qwen Q1 Q2 Q3 Q4 QC32 QC8
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14_kernel_tpot
W=$DIR/submit_s1d.slurm
W1B=$DIR/submit_s1b.slurm
PY=.venv/bin/python
RTD=h0_measurement/results/r8_routes
RT_S1=$RTD/r14_llama31-8b_131072_qa_b234.json
RT128_1B=$RTD/r14s1b_llama31-8b_131072_routes.json
RT32_1B=$RTD/r14s1b_llama31-8b_32768_routes.json
RT128_1C=$RTD/r14s1c_llama31-8b_131072_routes.json
RT32_1C=$RTD/r14s1c_llama31-8b_32768_routes.json
RT128_1D=$RTD/r14s1d_llama31-8b_131072_routes.json
RT32_1D=$RTD/r14s1d_llama31-8b_32768_routes.json
QM=qwen3-30b-a3b-2507
RTQ32_1B=$RTD/r14s1b_${QM}_32768_routes.json
RTQ8_1B=$RTD/r14s1b_${QM}_8192_routes.json
RTQ32_1D=$RTD/r14s1d_${QM}_32768_routes.json
RTQ8_1D=$RTD/r14s1d_${QM}_8192_routes.json
READER_SB="--partition=compute --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:30:00"
MODE="${1:-}"

need_id() { [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo "ERROR: job IDs must be decimal" >&2; exit 2; }; }

preflight() {
  [[ "$(hostname -s)" == trig-login01* ]] || echo "WARN: not on trig-login01; GPU sbatch will be refused here"
  export OMP_NUM_THREADS=8
  bash -n "$W"
  bash -n "$W1B"
  for t in test_r14_stage1d test_r14_stage1c test_r14_stage1b; do
    $PY -u $DIR/$t.py --fast | tail -1
  done
  $PY -u tests/test_r8.py --fast | tail -1
  $PY h0_measurement/prefetch_corpus.py --verify --out .h0_corpus/pg19
  echo "R14 Stage 1d preflight passed"
}

need_files() { for f in "$@"; do [[ -f "$f" ]] || { echo "ERROR: routes $f missing" >&2; exit 1; }; done; }
refuse_files() {
  for f in "$@"; do [[ ! -e "$f" ]] || { echo "ERROR: $f exists; refusing to overwrite a calibration" >&2; exit 1; }; done
}

SUBMITTED=()
sub() {   # sbatch, print only the job ID (or DRY)
  if [[ "$MODE" == --run-dry || "$MODE" == --run-qwen-dry ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2; echo DRY; return
  fi
  local out id
  out=$(sbatch --parsable "$@") || { echo "ERROR: sbatch failed after ${SUBMITTED[*]:-nothing}" >&2; exit 1; }
  id="${out%%;*}"; id="${id##* }"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "ERROR: no job id from: $out" >&2; exit 1; }
  SUBMITTED+=("$id"); echo "$id"
}

reader() {   # dependency list, log stem, reader args
  local dep=$1 stem=$2; shift 2
  sub --dependency=afterany:$dep --job-name=r14s1d-$stem $READER_SB \
      --output=h0_measurement/logs/r14s1d${stem}_%j.out --error=h0_measurement/logs/r14s1d${stem}_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1d.py $*"
}

chain_llama() {
  need_files "$RT_S1" "$RT128_1B" "$RT32_1B" "$RT128_1C" "$RT32_1C"
  refuse_files "$RT128_1D" "$RT32_1D"
  local P G C128 C32 RD o dep
  local -a A=() B=()
  P=$(sub --job-name=r14s1d-pilot --time=01:30:00 $W S1D_MODE=pilot S1D_TAG=pilot128 S1D_PRESET=pilot128 \
      S1D_CTX=131072 S1D_N_PROMPTS=1 S1D_PROMPT_OFFSET=3103 S1D_TASKS=niah_single S1D_CAL_OFFSET=0 \
      S1D_ROUTES_STD=$RT_S1 S1D_ROUTES_1B=$RT128_1B S1D_ROUTES_1C=$RT128_1C)
  echo "PILOT=$P" >&2
  G=$(sub --dependency=afterok:$P --job-name=r14s1d-gate $READER_SB \
      --output=h0_measurement/logs/r14s1dgate_%j.out --error=h0_measurement/logs/r14s1dgate_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1d.py --pilot $P")
  echo "GATE=$G" >&2
  C128=$(sub --dependency=afterok:$G --job-name=r14s1d-cal128 --time=06:00:00 $W S1D_MODE=calibrate \
      S1D_TAG=cal128 S1D_PRESET=main128 S1D_CTX=131072 S1D_N_PROMPTS=10 S1D_PROMPT_OFFSET=0 \
      S1D_ROUTES_1B=$RT128_1B S1D_WRITE_ROUTES=$RT128_1D)
  C32=$(sub --dependency=afterok:$G --job-name=r14s1d-cal32 --time=03:00:00 $W S1D_MODE=calibrate \
      S1D_TAG=cal32 S1D_PRESET=main32 S1D_CTX=32768 S1D_N_PROMPTS=10 S1D_PROMPT_OFFSET=0 \
      S1D_ROUTES_1B=$RT32_1B S1D_WRITE_ROUTES=$RT32_1D)
  echo "CAL128=$C128 CAL32=$C32" >&2
  for o in 7000 7010 7020 7030; do
    A+=("$(sub --dependency=afterok:$G:$C128 --job-name=r14s1d-m128-$o --time=10:00:00 $W \
        S1D_MODE=evaluate S1D_TAG=main128 S1D_PRESET=main128 S1D_CTX=131072 S1D_N_PROMPTS=10 \
        S1D_PROMPT_OFFSET=$o S1D_ROUTES_STD=$RT_S1 S1D_ROUTES_1B=$RT128_1B S1D_ROUTES_1C=$RT128_1C \
        S1D_ROUTES_1D=$RT128_1D)")
  done
  for o in 7100 7110 7120 7130; do
    B+=("$(sub --dependency=afterok:$G:$C32 --job-name=r14s1d-m32-$o --time=04:00:00 $W \
        S1D_MODE=evaluate S1D_TAG=main32 S1D_PRESET=main32 S1D_CTX=32768 S1D_N_PROMPTS=10 \
        S1D_PROMPT_OFFSET=$o S1D_ROUTES_1B=$RT32_1B S1D_ROUTES_1C=$RT32_1C S1D_ROUTES_1D=$RT32_1D)")
  done
  echo "MAIN128=${A[*]} MAIN32=${B[*]}" >&2
  dep=$(IFS=:; echo "${A[*]}:${B[*]}:$C128:$C32")
  RD=$(reader "$dep" read --main128 ${A[*]} --main32 ${B[*]} --cal128 $C128 --cal32 $C32)
  echo "READ=$RD" >&2
  echo "next: bash $DIR/script_stage1d.sh --status $P $G $C128 $C32 ${A[*]} ${B[*]} $RD"
}

chain_qwen() {
  refuse_files "$RTQ32_1B" "$RTQ8_1B" "$RTQ32_1D" "$RTQ8_1D"
  local R32 R8 P G C32 C8 RD o dep
  local -a Q=()
  R32=$(sub --job-name=r14s1d-qr0-32 --time=04:00:00 $W1B S1B_MODE=calibrate S1B_TAG=qcal32 \
      S1B_PRESET=main32 S1B_MODEL=$QM S1B_CTX=32768 S1B_N_PROMPTS=10 S1B_PROMPT_OFFSET=0 \
      S1B_WRITE_ROUTES=$RTQ32_1B)
  R8=$(sub --job-name=r14s1d-qr0-8 --time=02:00:00 $W1B S1B_MODE=calibrate S1B_TAG=qcal8 \
      S1B_PRESET=main32 S1B_MODEL=$QM S1B_CTX=8192 S1B_N_PROMPTS=10 S1B_PROMPT_OFFSET=0 \
      S1B_WRITE_ROUTES=$RTQ8_1B)
  echo "QWEN_R0_32=$R32 QWEN_R0_8=$R8" >&2
  P=$(sub --dependency=afterok:$R32 --job-name=r14s1d-qpilot --time=02:00:00 $W S1D_MODE=pilot \
      S1D_TAG=qwenpilot S1D_PRESET=qwenpilot S1D_CTX=32768 S1D_N_PROMPTS=1 S1D_PROMPT_OFFSET=3103 \
      S1D_TASKS=niah_single S1D_CAL_OFFSET=0 S1D_ROUTES_1B=$RTQ32_1B)
  G=$(sub --dependency=afterok:$P --job-name=r14s1d-qgate $READER_SB \
      --output=h0_measurement/logs/r14s1dqgate_%j.out --error=h0_measurement/logs/r14s1dqgate_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1d.py --pilot $P --pilot-tag qwenpilot")
  echo "QPILOT=$P QGATE=$G" >&2
  C32=$(sub --dependency=afterok:$G:$R32 --job-name=r14s1d-qcal32 --time=06:00:00 $W S1D_MODE=calibrate \
      S1D_TAG=calq32 S1D_PRESET=qwen32 S1D_CTX=32768 S1D_N_PROMPTS=10 S1D_PROMPT_OFFSET=0 \
      S1D_ROUTES_1B=$RTQ32_1B S1D_WRITE_ROUTES=$RTQ32_1D)
  C8=$(sub --dependency=afterok:$G:$R8 --job-name=r14s1d-qcal8 --time=03:00:00 $W S1D_MODE=calibrate \
      S1D_TAG=calq8 S1D_PRESET=qwen8cal S1D_CTX=8192 S1D_N_PROMPTS=10 S1D_PROMPT_OFFSET=0 \
      S1D_ROUTES_1B=$RTQ8_1B S1D_WRITE_ROUTES=$RTQ8_1D)
  echo "QCAL32=$C32 QCAL8=$C8" >&2
  for o in 7100 7110 7120 7130; do
    Q+=("$(sub --dependency=afterok:$G:$C32 --job-name=r14s1d-q32-$o --time=08:00:00 $W \
        S1D_MODE=evaluate S1D_TAG=qwen32 S1D_PRESET=qwen32 S1D_CTX=32768 S1D_N_PROMPTS=10 \
        S1D_PROMPT_OFFSET=$o S1D_ROUTES_1B=$RTQ32_1B S1D_ROUTES_1D=$RTQ32_1D)")
  done
  echo "QWEN32=${Q[*]}" >&2
  dep=$(IFS=:; echo "${Q[*]}:$C32:$C8")
  RD=$(reader "$dep" qread --qwen32 ${Q[*]} --calq32 $C32 --calq8 $C8 \
       --out-stem $PROJECT_ROOT/$DIR/stage1d_qwen)
  echo "QREAD=$RD" >&2
  echo "next: bash $DIR/script_stage1d.sh --status $R32 $R8 $P $G $C32 $C8 ${Q[*]} $RD"
}

case "$MODE" in
  --preflight) preflight ;;
  --run-dry|--run) preflight; chain_llama ;;
  --run-qwen-dry|--run-qwen) preflight; chain_qwen ;;
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
    (($# == 11)) || { echo "ERROR: --read needs A1 A2 A3 A4 B1 B2 B3 B4 C128 C32" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1d.py --main128 "$2" "$3" "$4" "$5" \
        --main32 "$6" "$7" "$8" "$9" --cal128 "${10}" --cal32 "${11}" ;;
  --read-qwen)
    (($# == 7)) || { echo "ERROR: --read-qwen needs Q1 Q2 Q3 Q4 QC32 QC8" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1d.py --qwen32 "$2" "$3" "$4" "$5" --calq32 "$6" \
        --calq8 "$7" --out-stem $DIR/stage1d_qwen ;;
  *) sed -n 2,27p "$0" >&2; exit 2 ;;
esac
