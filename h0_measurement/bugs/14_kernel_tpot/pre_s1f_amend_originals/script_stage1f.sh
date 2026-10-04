#!/usr/bin/env bash
# R14 Stage 1f: F1 (kernel v2 vs FlashAttention), F2 (two-tier exact-row reads),
# F3 (reuse again: >= 80 units per role, second-question label), F4 (Qwen
# recalibrated with multivalue). Design: s1f_lib.py, s1f_kernel.py. Frozen rules:
# read_stage1f.py (docstring) and s1f_kernel.py (kernel and two-tier timing).
# Workers: submit_s1f.slurm, submit_s1f_kernel.slurm. No shared code edited.
#
# Run where GPU sbatch works (Trillium: trig-login01 only; Rorqual: a login node).
# The project root is this file's ../../..; routes files are picked by sha256
# (plain name or <stem>@<cluster>.json), so the same script runs on both clusters.
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1f.sh --preflight
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1f.sh --run-reuse-dry    | --run-reuse     (F3)
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1f.sh --run-qwen-dry     | --run-qwen      (F4 + F2 on Qwen)
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1f.sh --run-twotier-dry  | --run-twotier   (F2, Llama)
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1f.sh --run-kernel-dry   | --run-kernel    (F1 + F2 timing)
#
# --run-reuse (Llama-3.1-8B, mixed two-question prompts; nested router from Stage 1e's calibration):
#   reusepilotf (excl.) 128K prompt 3109 -> gate
#   reuse128f x 4       prompts 8500-8595 (24 per block, 13 arms x 2 questions)          ~5-6 h each
#   reuse32f  x 4       prompts 8600-8687 (22 per block)                                 ~2 h each
#   reader              read_stage1f.py -> bugs/14_kernel_tpot/stage1f_reuse.{json,md}
# --run-qwen (Qwen3-30B-A3B-2507 @32K; stop rule eos_only):
#   qwenpilotf (excl.)  forced calibration on prompt 0 (multivalue + single), then prompt 3108 -> gate
#   qcal32f             calibration, prompts 8700-8729: multikey, multivalue x 30; single, vt x 10
#                       -> results/r8_routes/r14s1f_qwen3-30b-a3b-2507_32768_routes.json         ~4-8 h
#   qwen32f x 4         prompts 8800-8839; qregress32f: 8215, 8218, 8235 (multivalue), 8234 (multikey)
#   reader              -> stage1f_qwen.{json,md}
# --run-twotier (Llama-3.1-8B):
#   ttpilot (excl.)     128K prompt 3110 -> gate -> tt128 x 4 (8900-8939), tt32 x 4 (9000-9039)
#   reader              -> stage1f_tt.{json,md}
# --run-kernel:         one GPU: v2 vs reference, then the benchmark
#                       -> results/r14s1f_kernel_<job>/kernel_v2_bench.{json,md}   (QUICK=1: mechanics only)
#
#   --status ID...   /   --cancel ID...
#   --read-reuse U1 U2 U3 U4 V1 V2 V3 V4   /   --read-qwen Q1 Q2 Q3 Q4 REG CAL   /   --read-tt A1 A2 A3 A4 B1 B2 B3 B4
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14_kernel_tpot
W=$DIR/submit_s1f.slurm
WK=$DIR/submit_s1f_kernel.slurm
PY=.venv/bin/python
RTD=h0_measurement/results/r8_routes
QM=qwen3-30b-a3b-2507
RTQ32_1F=$RTD/r14s1f_${QM}_32768_routes.json
QCAL_COUNTS="niah_single=10,niah_multikey=30,niah_multivalue=30,vt=10"
QREG="8215:niah_multivalue,8218:niah_multivalue,8235:niah_multivalue,8234:niah_multikey"
# `compute` exists only on Trillium (trig); elsewhere (e.g. rorqual) let the scheduler pick.
READER_PART=""
[[ "$(hostname -s)" == trig* ]] && READER_PART="--partition=compute"
READER_SB="$READER_PART --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:30:00"
MODE="${1:-}"

need_id() { [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo "ERROR: job IDs must be decimal" >&2; exit 2; }; }

# pick STEM SHA10: the routes file with this content, under its plain name or a
# <stem>@<cluster>.json copy. These are the files Stages 1b-1e used:
#   Llama 128K: 1b e4afe5279e, 1d 0d4622defc, 1e d5c84e5e6d (Trillium calibrations; Stage 1e tail128)
#   Llama 32K:  1b ef94b3e5b1, 1d 01c8cafa52, 1e 3b6c7d559f (Rorqual calibrations; Stage 1e tail32)
#   Qwen 32K:   1b adf2b42ea2, 1d 9beb8b2c5b                (Stage 1e qwen32e)
pick() {
  local f
  for f in "$RTD/$1.json" "$RTD/$1"@*.json; do
    if [[ -f "$f" && "$(sha256sum "$f" | cut -c1-10)" == "$2" ]]; then echo "$f"; return 0; fi
  done
  echo "ERROR: no $RTD/$1.json or $RTD/$1@<cluster>.json with sha256 $2..." >&2
  return 1
}

preflight() {
  [[ "$(hostname -s)" == trig-login01* || "$(hostname -s)" != trig* ]] \
    || echo "WARN: on Trillium, GPU sbatch only works from trig-login01"
  export OMP_NUM_THREADS=8
  bash -n "$W"
  bash -n "$WK"
  for t in test_r14_stage1f test_r14_stage1e; do
    $PY -u $DIR/$t.py --fast | tail -1
  done
  $PY -u tests/test_r8.py --fast | tail -1
  $PY h0_measurement/prefetch_corpus.py --verify --out .h0_corpus/pg19
  echo "R14 Stage 1f preflight passed"
}

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
  sub --dependency=afterok:$1 --job-name=r14s1f-$2 $READER_SB \
      --output=h0_measurement/logs/r14s1f$2_%j.out --error=h0_measurement/logs/r14s1f$2_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1f.py --pilot $1 --pilot-tag $3"
}

reader() {   # dependency list, log stem, reader args
  local dep=$1 stem=$2; shift 2
  sub --dependency=afterany:$dep --job-name=r14s1f-$stem $READER_SB \
      --output=h0_measurement/logs/r14s1f${stem}_%j.out --error=h0_measurement/logs/r14s1f${stem}_%j.err \
      --wrap "cd $PROJECT_ROOT && OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1f.py $*"
}

chain_reuse() {
  local B128 D128 E128 B32 D32 E32 P G RD o dep
  local -a U=() V=()
  B128=$(pick r14s1b_llama31-8b_131072_routes e4afe5279e)
  E128=$(pick r14s1e_llama31-8b_131072_routes d5c84e5e6d)
  B32=$(pick r14s1b_llama31-8b_32768_routes ef94b3e5b1)
  D32=$(pick r14s1d_llama31-8b_32768_routes 01c8cafa52)
  E32=$(pick r14s1e_llama31-8b_32768_routes 3b6c7d559f)
  D128=$(pick r14s1d_llama31-8b_131072_routes 0d4622defc)
  P=$(sub --job-name=r14s1f-upilot --time=02:00:00 $W S1F_MODE=reuse S1F_TAG=reusepilotf S1F_PRESET=reusepilotf \
      S1F_CTX=131072 S1F_N_PROMPTS=1 S1F_PROMPT_OFFSET=3109 S1F_ROUTES_1B=$B128 S1F_ROUTES_1D=$D128 \
      S1F_ROUTES_1E=$E128)
  G=$(gate "$P" ugate reusepilotf)
  echo "REUSE_PILOT=$P GATE=$G" >&2
  for o in 8500 8524 8548 8572; do
    U+=("$(sub --dependency=afterok:$G --job-name=r14s1f-u128-$o --time=07:00:00 $W S1F_MODE=reuse \
        S1F_TAG=reuse128f S1F_PRESET=reuse128f S1F_CTX=131072 S1F_N_PROMPTS=24 S1F_PROMPT_OFFSET=$o \
        S1F_ROUTES_1B=$B128 S1F_ROUTES_1D=$D128 S1F_ROUTES_1E=$E128)")
  done
  for o in 8600 8622 8644 8666; do
    V+=("$(sub --dependency=afterok:$G --job-name=r14s1f-u32-$o --time=03:00:00 $W S1F_MODE=reuse \
        S1F_TAG=reuse32f S1F_PRESET=reuse32f S1F_CTX=32768 S1F_N_PROMPTS=22 S1F_PROMPT_OFFSET=$o \
        S1F_ROUTES_1B=$B32 S1F_ROUTES_1D=$D32 S1F_ROUTES_1E=$E32)")
  done
  echo "REUSE128=${U[*]} REUSE32=${V[*]}" >&2
  dep=$(IFS=:; echo "${U[*]}:${V[*]}")
  RD=$(reader "$dep" uread --reuse128 ${U[*]} --reuse32 ${V[*]} --out-stem $PROJECT_ROOT/$DIR/stage1f_reuse)
  echo "READ=$RD" >&2
  echo "next: bash $DIR/script_stage1f.sh --status $P $G ${U[*]} ${V[*]} $RD"
}

chain_qwen() {
  local Q1B Q1D P G C REG RD o dep
  local -a Q=()
  Q1B=$(pick r14s1b_${QM}_32768_routes adf2b42ea2)
  Q1D=$(pick r14s1d_${QM}_32768_routes 9beb8b2c5b)
  refuse_files "$RTQ32_1F"
  P=$(sub --job-name=r14s1f-qpilot --time=03:00:00 $W S1F_MODE=pilot S1F_TAG=qwenpilotf S1F_PRESET=qwenpilotf \
      S1F_CTX=32768 S1F_N_PROMPTS=1 S1F_PROMPT_OFFSET=3108 S1F_TASKS=niah_multivalue,niah_single \
      S1F_CAL_OFFSET=0 S1F_CAL_TASKS=niah_multivalue,niah_single S1F_ROUTES_1B=$Q1B S1F_ROUTES_1D=$Q1D)
  G=$(gate "$P" qgate qwenpilotf)
  echo "QPILOT=$P QGATE=$G" >&2
  C=$(sub --dependency=afterok:$G --job-name=r14s1f-qcal --time=12:00:00 $W S1F_MODE=calibrate S1F_TAG=qcal32f \
      S1F_PRESET=qwen32f S1F_CTX=32768 S1F_PROMPT_OFFSET=8700 S1F_TASK_COUNTS=$QCAL_COUNTS \
      S1F_ROUTES_1B=$Q1B S1F_WRITE_ROUTES=$RTQ32_1F)
  echo "QCAL=$C" >&2
  for o in 8800 8810 8820 8830; do
    Q+=("$(sub --dependency=afterok:$G:$C --job-name=r14s1f-q32-$o --time=05:00:00 $W S1F_MODE=evaluate \
        S1F_TAG=qwen32f S1F_PRESET=qwen32f S1F_CTX=32768 S1F_N_PROMPTS=10 S1F_PROMPT_OFFSET=$o \
        S1F_ROUTES_1B=$Q1B S1F_ROUTES_1D=$Q1D S1F_ROUTES_1E=$RTQ32_1F)")
  done
  REG=$(sub --dependency=afterok:$G:$C --job-name=r14s1f-qreg --time=02:00:00 $W S1F_MODE=evaluate \
      S1F_TAG=qregress32f S1F_PRESET=qwen32f S1F_CTX=32768 S1F_PROMPT_LIST=$QREG \
      S1F_ROUTES_1B=$Q1B S1F_ROUTES_1D=$Q1D S1F_ROUTES_1E=$RTQ32_1F)
  echo "QWEN32=${Q[*]} QREGRESS=$REG" >&2
  dep=$(IFS=:; echo "${Q[*]}:$REG:$C")
  RD=$(reader "$dep" qread --qwen32 ${Q[*]} --qregress $REG --qcal $C --out-stem $PROJECT_ROOT/$DIR/stage1f_qwen)
  echo "READ=$RD" >&2
  echo "next: bash $DIR/script_stage1f.sh --status $P $G $C ${Q[*]} $REG $RD"
}

chain_twotier() {
  local B32 D32 P G RD o dep
  local -a A=() B=()
  B32=$(pick r14s1b_llama31-8b_32768_routes ef94b3e5b1)
  D32=$(pick r14s1d_llama31-8b_32768_routes 01c8cafa52)    # tt32's protected read (Stage 1d's critical heads)
  P=$(sub --job-name=r14s1f-ttpilot --time=02:00:00 $W S1F_MODE=evaluate S1F_TAG=ttpilot S1F_PRESET=ttpilot \
      S1F_CTX=131072 S1F_N_PROMPTS=1 S1F_PROMPT_OFFSET=3110 S1F_TASKS=niah_single)
  G=$(gate "$P" tgate ttpilot)
  echo "TT_PILOT=$P GATE=$G" >&2
  for o in 8900 8910 8920 8930; do
    A+=("$(sub --dependency=afterok:$G --job-name=r14s1f-tt128-$o --time=06:00:00 $W S1F_MODE=evaluate \
        S1F_TAG=tt128 S1F_PRESET=tt128 S1F_CTX=131072 S1F_N_PROMPTS=10 S1F_PROMPT_OFFSET=$o)")
  done
  for o in 9000 9010 9020 9030; do
    B+=("$(sub --dependency=afterok:$G --job-name=r14s1f-tt32-$o --time=02:30:00 $W S1F_MODE=evaluate \
        S1F_TAG=tt32 S1F_PRESET=tt32 S1F_CTX=32768 S1F_N_PROMPTS=10 S1F_PROMPT_OFFSET=$o \
        S1F_ROUTES_1B=$B32 S1F_ROUTES_1D=$D32)")
  done
  echo "TT128=${A[*]} TT32=${B[*]}" >&2
  dep=$(IFS=:; echo "${A[*]}:${B[*]}")
  RD=$(reader "$dep" tread --tt128 ${A[*]} --tt32 ${B[*]} --out-stem $PROJECT_ROOT/$DIR/stage1f_tt)
  echo "READ=$RD" >&2
  echo "next: bash $DIR/script_stage1f.sh --status $P $G ${A[*]} ${B[*]} $RD"
}

chain_kernel() {
  local K
  K=$(sub --job-name=r14s1f-kernel $WK S1F_QUICK=${QUICK:-0})
  echo "KERNEL=$K" >&2
  echo "next: bash $DIR/script_stage1f.sh --status $K   (results: h0_measurement/results/r14s1f_kernel_$K/)"
}

case "$MODE" in
  --preflight) preflight ;;
  --run-reuse-dry|--run-reuse) preflight; chain_reuse ;;
  --run-qwen-dry|--run-qwen) preflight; chain_qwen ;;
  --run-twotier-dry|--run-twotier) preflight; chain_twotier ;;
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
  --read-reuse)
    (($# == 9)) || { echo "ERROR: --read-reuse needs U1 U2 U3 U4 V1 V2 V3 V4" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1f.py --reuse128 "$2" "$3" "$4" "$5" --reuse32 "$6" "$7" "$8" "$9" \
        --out-stem $DIR/stage1f_reuse ;;
  --read-qwen)
    (($# == 7)) || { echo "ERROR: --read-qwen needs Q1 Q2 Q3 Q4 REG CAL" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1f.py --qwen32 "$2" "$3" "$4" "$5" --qregress "$6" --qcal "$7" \
        --out-stem $DIR/stage1f_qwen ;;
  --read-tt)
    (($# == 9)) || { echo "ERROR: --read-tt needs A1 A2 A3 A4 B1 B2 B3 B4" >&2; exit 2; }
    for x in "${@:2}"; do need_id "$x"; done
    OMP_NUM_THREADS=8 $PY -u $DIR/read_stage1f.py --tt128 "$2" "$3" "$4" "$5" --tt32 "$6" "$7" "$8" "$9" \
        --out-stem $DIR/stage1f_tt ;;
  *) sed -n 2,40p "$0" >&2; exit 2 ;;
esac
