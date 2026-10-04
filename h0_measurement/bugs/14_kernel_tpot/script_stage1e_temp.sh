#!/usr/bin/env bash
# R14 Stage 1e, one-off: ONLY the three missing Llama-3.1-8B 128K tail blocks
# (preset tail128, prompts 8110-8139). Block 8100 (22224545), the regression block
# (22224549) and the 128K calibration (22224543) already ran on Rorqual; these blocks
# read the same routes files (identical sha256: 1b e4afe527.., 1d 0d4622de.., 1e
# d5c84e5e..). No pilot, gate, calibration or reader: the full reader runs after
# all four blocks are in one results folder. Worker: submit_s1e.slurm (unchanged).
#
# Trillium: GPU sbatch only works from trig-login01.
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1e_temp.sh --run-dry | --run
#   bash h0_measurement/bugs/14_kernel_tpot/script_stage1e_temp.sh --status ID...
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
cd "$PROJECT_ROOT"
DIR=h0_measurement/bugs/14_kernel_tpot
W=$DIR/submit_s1e.slurm
PY=.venv/bin/python
RTD=h0_measurement/results/r8_routes
RT128_1B=$RTD/r14s1b_llama31-8b_131072_routes.json
RT128_1D=$RTD/r14s1d_llama31-8b_131072_routes.json
RT128_1E=$RTD/r14s1e_llama31-8b_131072_routes.json
# the files Rorqual's block 8100 read (sha256 prefixes from its sidecar)
declare -A SHA=([$RT128_1B]=e4afe5279e [$RT128_1D]=0d4622defc [$RT128_1E]=d5c84e5e6d)
MODE="${1:-}"

check_routes() {
  for f in "${!SHA[@]}"; do
    [[ -f "$f" ]] || { echo "ERROR: routes $f missing" >&2; exit 1; }
    got=$(sha256sum "$f" | cut -c1-10)
    [[ "$got" == "${SHA[$f]}" ]] || { echo "ERROR: $f sha $got, block 8100 read ${SHA[$f]}" >&2; exit 1; }
  done
  echo "routes match block 8100's" >&2
}

preflight() {
  [[ "$(hostname -s)" == trig-login01* ]] || echo "WARN: not on trig-login01; GPU sbatch will be refused here" >&2
  export OMP_NUM_THREADS=8
  bash -n "$W"
  $PY -u $DIR/test_r14_stage1e.py --fast | tail -1
  $PY h0_measurement/prefetch_corpus.py --verify --out .h0_corpus/pg19
}

sub() {
  if [[ "$MODE" == --run-dry ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2; echo DRY; return
  fi
  local out id
  out=$(sbatch --parsable "$@") || { echo "ERROR: sbatch failed" >&2; exit 1; }
  id="${out%%;*}"; id="${id##* }"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "ERROR: no job id from: $out" >&2; exit 1; }
  echo "$id"
}

case "$MODE" in
  --run-dry|--run)
    check_routes
    preflight
    A=()
    for o in 8110 8120 8130; do
      A+=("$(sub --job-name=r14s1e-t128-$o --time=06:00:00 $W S1E_MODE=evaluate S1E_TAG=tail128 \
          S1E_PRESET=tail128 S1E_CTX=131072 S1E_N_PROMPTS=10 S1E_PROMPT_OFFSET=$o \
          S1E_ROUTES_1B=$RT128_1B S1E_ROUTES_1D=$RT128_1D S1E_ROUTES_1E=$RT128_1E)")
    done
    echo "TAIL128 (8110 8120 8130) = ${A[*]}"
    echo "next: bash $DIR/script_stage1e_temp.sh --status ${A[*]}" ;;
  --status)
    (($# >= 2)) || { echo "ERROR: --status needs job IDs" >&2; exit 2; }
    sacct -j "$(IFS=,; echo "${*:2}")" -X --format=JobID%12,JobName%20,State%24,Elapsed,ExitCode ;;
  *) sed -n 2,13p "$0" >&2; exit 2 ;;
esac
