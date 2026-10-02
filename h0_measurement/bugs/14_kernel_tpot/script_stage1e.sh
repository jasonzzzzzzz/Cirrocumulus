#!/usr/bin/env bash
# R14 Stage 1e: E1 (reads over an exact store: noise or dilution), E2 (router tail:
# nested dense sets + a multikey-weighted calibration), E3 (two questions per
# stored context: reads vs SnapKV-with-question), E4 (Qwen: stop rule, V4 arms,
# 128K reads), E5 (the compaction kernel). Design: s1e_lib.py, s1e_kernel.py.
# Frozen rules: read_stage1e.py (docstring) and s1e_kernel.py (kernel rule).
# Workers: submit_s1e.slurm, submit_s1e_kernel.slurm. No shared code edited.
#
# Run where GPU sbatch works (Trillium: trig-login01 only; Rorqual: a login node):
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1e.sh --preflight
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1e.sh --run-tail-dry   | --run-tail     (E1 + E2)
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1e.sh --run-reuse-dry  | --run-reuse    (E3)
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1e.sh --run-qwen-dry   | --run-qwen     (E4)
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1e.sh --run-kernel-dry | --run-kernel   (E5)
#
# --run-tail (Llama-3.1-8B):
#   pilot (excluded)   128K: forced calibration on prompt 0, then preset pilot128e on prompt 3104  ~30 min
#   gate               read_stage1e.py --pilot
#   cal128e, cal32e    E2's calibration on prompts 8000-8039 (multikey x 40, the others x 10)   ~1-3 h each
#   tail128 x 4        prompts 8100-8139, 20 arms; regress128: Stage 1d's 7020, 7036 (multikey)  ~2 h each
#   tail32  x 4        prompts 8200-8239, 21 arms; regress32: 7112, 7113, 7117 (multikey)        ~0.7 h each
#   reader             read_stage1e.py -> bugs/14_kernel_tpot/stage1e.{json,md}
# --run-reuse (Llama-3.1-8B, mixed two-question prompts):
#   reusepilot (excl.) 128K prompt 3105 -> gate -> reuse128 x 2 (8300-8339), reuse32 x 2 (8400-8439)
#   reader             -> stage1e_reuse.{json,md}
# --run-qwen (Qwen3-30B-A3B-2507; stop rule eos_only):
#   qpilot32 (excl.)   32K prompt 3106, multivalue + single -> qgate32 -> qwen32e x 4 (8200-8239)
#   qpilot128 (excl.)  128K prompt 3107 on 2 GPUs -> qgate128 -> qwen128q x 4 (8100-8139, 2 GPUs)
#   reader             -> stage1e_qwen.{json,md}
# --run-kernel:        one GPU: kernel vs reference, then the micro-benchmark
#                      -> results/r14s1e_kernel_<job>/kernel_bench.{json,md}   (add QUICK=1 for the short sweep)
#
#   --status ID...   /   --cancel ID...
#   --read-tail A1 A2 A3 A4 R128 C128 B1 B2 B3 B4 R32 C32   /   --read-reuse U1 U2 V1 V2
#   --read-qwen Q1 Q2 Q3 Q4 P1 P2 P3 P4
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14_kernel_tpot
W=$DIR/submit_s1e.slurm
WK=$DIR/submit_s1e_kernel.slurm
PY=.venv/bin/python
RTD=h0_measurement/results/r8_routes
RT128_1B=$RTD/r14s1b_llama31-8b_131072_routes.json
RT32_1B=$RTD/r14s1b_llama31-8b_32768_routes.json
RT128_1D=$RTD/r14s1d_llama31-8b_131072_routes.json
RT32_1D=$RTD/r14s1d_llama31-8b_32768_routes.json
RT128_1E=$RTD/r14s1e_llama31-8b_131072_routes.json
RT32_1E=$RTD/r14s1e_llama31-8b_32768_routes.json
QM=qwen3-30b-a3b-2507
RTQ32_1B=$RTD/r14s1b_${QM}_32768_routes.json
RTQ32_1D=$RTD/r14s1d_${QM}_32768_routes.json
CAL_COUNTS="niah_single=10,niah_multikey=40,niah_multivalue=10,vt=10"
REG128="7020:niah_multikey,7036:niah_multikey"
REG32="7112:niah_multikey,7113:niah_multikey,7117:niah_multikey"
# `compute` exists only on Trillium (trig); elsewhere (e.g. rorqual) let the scheduler pick.
READER_PART=""
[[ "$(hostname -s)" == trig* ]] && READER_PART="--partition=compute"
READER_SB="$READER_PART --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:30:00"
GPU2="--gpus-per-node=2 --cpus-per-task=16"
MODE="${1:-}"

need_id() { [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo "ERROR: job IDs must be decimal" >&2; exit 2; }; }

preflight() {
  [[ "$(hostname -s)" == trig-login01* || "$(hostname -s)" != trig* ]] \
    || echo "WARN: on Trillium, GPU sbatch only works from trig-login01"
  export OMP_NUM_THREADS=8
  bash -n "$W"
  bash -n "$WK"
  for t in test_r14_stage1e test_r14_stage1d; do
    $PY -u $DIR/$t.py --fast | tail -1
  done
  $PY -u tests/test_r8.py --fast | tail -1
  $PY h0_measurement/prefetch_corpus.py --verify --out .h0_corpus/pg19
  echo "R14 Stage 1e preflight passed"
}

need_files() { for f in "$@"; do [[ -f "$f" ]] || { echo "ERROR: routes $f missing" >&2; exit 1; }; done; }
refuse_files() {
  for f in "$@"; do [[ ! -e "$f" ]] || { echo "ERROR: $f exists; refusing to overwrite a calibration" >&2; exit 1; }; done
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

gate() {   # dependency, log stem, pilot tag
  sub --dependency=afterok:$1 --job-name=r14s1e-$2 $READER_SB \
      --output=h0_measurement/logs/r14s1e$2_%j.out --error=h0_measurement/logs/r14s1e$2_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1e.py --pilot $1 --pilot-tag $3"
}

reader() {   # dependency list, log stem, reader args
  local dep=$1 stem=$2; shift 2
  sub --dependency=afterany:$dep --job-name=r14s1e-$stem $READER_SB \
      --output=h0_measurement/logs/r14s1e${stem}_%j.out --error=h0_measurement/logs/r14s1e${stem}_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1e.py $*"
}

chain_tail() {
  need_files "$RT128_1B" "$RT32_1B" "$RT128_1D" "$RT32_1D"
  refuse_files "$RT128_1E" "$RT32_1E"
  local P G C128 C32 R128 R32 RD o dep
  local -a A=() B=()
  P=$(sub --job-name=r14s1e-pilot --time=02:00:00 $W S1E_MODE=pilot S1E_TAG=pilot128e S1E_PRESET=pilot128e \
      S1E_CTX=131072 S1E_N_PROMPTS=1 S1E_PROMPT_OFFSET=3104 S1E_TASKS=niah_single S1E_CAL_OFFSET=0 \
      S1E_ROUTES_1B=$RT128_1B S1E_ROUTES_1D=$RT128_1D)
  G=$(gate "$P" gate pilot128e)
  echo "PILOT=$P GATE=$G" >&2
  C128=$(sub --dependency=afterok:$G --job-name=r14s1e-cal128 --time=10:00:00 $W S1E_MODE=calibrate \
      S1E_TAG=cal128e S1E_PRESET=tail128 S1E_CTX=131072 S1E_PROMPT_OFFSET=8000 S1E_TASK_COUNTS=$CAL_COUNTS \
      S1E_ROUTES_1B=$RT128_1B S1E_WRITE_ROUTES=$RT128_1E)
  C32=$(sub --dependency=afterok:$G --job-name=r14s1e-cal32 --time=05:00:00 $W S1E_MODE=calibrate \
      S1E_TAG=cal32e S1E_PRESET=tail32 S1E_CTX=32768 S1E_PROMPT_OFFSET=8000 S1E_TASK_COUNTS=$CAL_COUNTS \
      S1E_ROUTES_1B=$RT32_1B S1E_WRITE_ROUTES=$RT32_1E)
  echo "CAL128=$C128 CAL32=$C32" >&2
  for o in 8100 8110 8120 8130; do
    A+=("$(sub --dependency=afterok:$G:$C128 --job-name=r14s1e-t128-$o --time=06:00:00 $W \
        S1E_MODE=evaluate S1E_TAG=tail128 S1E_PRESET=tail128 S1E_CTX=131072 S1E_N_PROMPTS=10 \
        S1E_PROMPT_OFFSET=$o S1E_ROUTES_1B=$RT128_1B S1E_ROUTES_1D=$RT128_1D S1E_ROUTES_1E=$RT128_1E)")
  done
  R128=$(sub --dependency=afterok:$G:$C128 --job-name=r14s1e-reg128 --time=02:00:00 $W S1E_MODE=evaluate \
      S1E_TAG=regress128 S1E_PRESET=tail128 S1E_CTX=131072 S1E_PROMPT_LIST=$REG128 \
      S1E_ROUTES_1B=$RT128_1B S1E_ROUTES_1D=$RT128_1D S1E_ROUTES_1E=$RT128_1E)
  for o in 8200 8210 8220 8230; do
    B+=("$(sub --dependency=afterok:$G:$C32 --job-name=r14s1e-t32-$o --time=03:00:00 $W \
        S1E_MODE=evaluate S1E_TAG=tail32 S1E_PRESET=tail32 S1E_CTX=32768 S1E_N_PROMPTS=10 \
        S1E_PROMPT_OFFSET=$o S1E_ROUTES_1B=$RT32_1B S1E_ROUTES_1D=$RT32_1D S1E_ROUTES_1E=$RT32_1E)")
  done
  R32=$(sub --dependency=afterok:$G:$C32 --job-name=r14s1e-reg32 --time=01:00:00 $W S1E_MODE=evaluate \
      S1E_TAG=regress32 S1E_PRESET=tail32 S1E_CTX=32768 S1E_PROMPT_LIST=$REG32 \
      S1E_ROUTES_1B=$RT32_1B S1E_ROUTES_1D=$RT32_1D S1E_ROUTES_1E=$RT32_1E)
  echo "TAIL128=${A[*]} REGRESS128=$R128 TAIL32=${B[*]} REGRESS32=$R32" >&2
  dep=$(IFS=:; echo "${A[*]}:${B[*]}:$R128:$R32:$C128:$C32")
  RD=$(reader "$dep" read --tail128 ${A[*]} --regress128 $R128 --cal128 $C128 \
       --tail32 ${B[*]} --regress32 $R32 --cal32 $C32)
  echo "READ=$RD" >&2
  echo "next: bash $DIR/script_stage1e.sh --status $P $G $C128 $C32 ${A[*]} $R128 ${B[*]} $R32 $RD"
}

chain_reuse() {
  need_files "$RT128_1B" "$RT32_1B" "$RT128_1D" "$RT32_1D"
  local P G RD dep
  local -a U=() V=()
  P=$(sub --job-name=r14s1e-upilot --time=02:00:00 $W S1E_MODE=reuse S1E_TAG=reusepilot S1E_PRESET=reusepilot \
      S1E_CTX=131072 S1E_N_PROMPTS=1 S1E_PROMPT_OFFSET=3105 S1E_ROUTES_1B=$RT128_1B S1E_ROUTES_1D=$RT128_1D)
  G=$(gate "$P" ugate reusepilot)
  echo "REUSE_PILOT=$P GATE=$G" >&2
  for o in 8300 8320; do
    U+=("$(sub --dependency=afterok:$G --job-name=r14s1e-u128-$o --time=06:00:00 $W S1E_MODE=reuse \
        S1E_TAG=reuse128 S1E_PRESET=reuse128 S1E_CTX=131072 S1E_N_PROMPTS=20 S1E_PROMPT_OFFSET=$o \
        S1E_ROUTES_1B=$RT128_1B S1E_ROUTES_1D=$RT128_1D)")
  done
  for o in 8400 8420; do
    V+=("$(sub --dependency=afterok:$G --job-name=r14s1e-u32-$o --time=02:30:00 $W S1E_MODE=reuse \
        S1E_TAG=reuse32 S1E_PRESET=reuse32 S1E_CTX=32768 S1E_N_PROMPTS=20 S1E_PROMPT_OFFSET=$o \
        S1E_ROUTES_1B=$RT32_1B S1E_ROUTES_1D=$RT32_1D)")
  done
  echo "REUSE128=${U[*]} REUSE32=${V[*]}" >&2
  dep=$(IFS=:; echo "${U[*]}:${V[*]}")
  RD=$(reader "$dep" uread --reuse128 ${U[*]} --reuse32 ${V[*]} --out-stem $PROJECT_ROOT/$DIR/stage1e_reuse)
  echo "READ=$RD" >&2
  echo "next: bash $DIR/script_stage1e.sh --status $P $G ${U[*]} ${V[*]} $RD"
}

chain_qwen() {
  need_files "$RTQ32_1B" "$RTQ32_1D"
  local P32 G32 P128 G128 RD o dep
  local -a Q=() X=()
  P32=$(sub --job-name=r14s1e-qpilot32 --time=02:00:00 $W S1E_MODE=evaluate S1E_TAG=qwenpilot32e \
      S1E_PRESET=qwenpilot32e S1E_CTX=32768 S1E_N_PROMPTS=1 S1E_PROMPT_OFFSET=3106 \
      S1E_TASKS=niah_multivalue,niah_single S1E_ROUTES_1B=$RTQ32_1B S1E_ROUTES_1D=$RTQ32_1D)
  G32=$(gate "$P32" qgate32 qwenpilot32e)
  P128=$(sub $GPU2 --job-name=r14s1e-qpilot128 --time=03:00:00 $W S1E_MODE=evaluate S1E_TAG=qwenpilot128 \
      S1E_PRESET=qwenpilot128 S1E_CTX=131072 S1E_N_PROMPTS=1 S1E_PROMPT_OFFSET=3107 S1E_TASKS=niah_single)
  G128=$(gate "$P128" qgate128 qwenpilot128)
  echo "QPILOT32=$P32 QGATE32=$G32 QPILOT128=$P128 QGATE128=$G128" >&2
  for o in 8200 8210 8220 8230; do
    Q+=("$(sub --dependency=afterok:$G32 --job-name=r14s1e-q32-$o --time=04:00:00 $W S1E_MODE=evaluate \
        S1E_TAG=qwen32e S1E_PRESET=qwen32e S1E_CTX=32768 S1E_N_PROMPTS=10 S1E_PROMPT_OFFSET=$o \
        S1E_ROUTES_1B=$RTQ32_1B S1E_ROUTES_1D=$RTQ32_1D)")
  done
  for o in 8100 8110 8120 8130; do
    X+=("$(sub --dependency=afterok:$G128 $GPU2 --job-name=r14s1e-q128-$o --time=10:00:00 $W S1E_MODE=evaluate \
        S1E_TAG=qwen128q S1E_PRESET=qwen128q S1E_CTX=131072 S1E_N_PROMPTS=10 S1E_PROMPT_OFFSET=$o)")
  done
  echo "QWEN32=${Q[*]} QWEN128=${X[*]}" >&2
  dep=$(IFS=:; echo "${Q[*]}:${X[*]}")
  RD=$(reader "$dep" qread --qwen32 ${Q[*]} --qwen128 ${X[*]} --out-stem $PROJECT_ROOT/$DIR/stage1e_qwen)
  echo "READ=$RD" >&2
  echo "next: bash $DIR/script_stage1e.sh --status $P32 $G32 $P128 $G128 ${Q[*]} ${X[*]} $RD"
}

chain_kernel() {
  local K
  K=$(sub --job-name=r14s1e-kernel $WK S1E_QUICK=${QUICK:-0})
  echo "KERNEL=$K" >&2
  echo "next: bash $DIR/script_stage1e.sh --status $K   (results: h0_measurement/results/r14s1e_kernel_$K/)"
}

case "$MODE" in
  --preflight) preflight ;;
  --run-tail-dry|--run-tail) preflight; chain_tail ;;
  --run-reuse-dry|--run-reuse) preflight; chain_reuse ;;
  --run-qwen-dry|--run-qwen) preflight; chain_qwen ;;
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
  --read-tail)
    (($# == 13)) || { echo "ERROR: --read-tail needs A1 A2 A3 A4 R128 C128 B1 B2 B3 B4 R32 C32" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1e.py --tail128 "$2" "$3" "$4" "$5" --regress128 "$6" --cal128 "$7" \
        --tail32 "$8" "$9" "${10}" "${11}" --regress32 "${12}" --cal32 "${13}" ;;
  --read-reuse)
    (($# == 5)) || { echo "ERROR: --read-reuse needs U1 U2 V1 V2" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1e.py --reuse128 "$2" "$3" --reuse32 "$4" "$5" \
        --out-stem $DIR/stage1e_reuse ;;
  --read-qwen)
    (($# == 9)) || { echo "ERROR: --read-qwen needs Q1 Q2 Q3 Q4 P1 P2 P3 P4" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1e.py --qwen32 "$2" "$3" "$4" "$5" --qwen128 "$6" "$7" "$8" "$9" \
        --out-stem $DIR/stage1e_qwen ;;
  *) sed -n 2,37p "$0" >&2; exit 2 ;;
esac
