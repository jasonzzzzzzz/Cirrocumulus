#!/usr/bin/env bash
# Resubmit a lost R6 block from a shadow project root (plan.md, 2026-10-18 amendment "R6: resubmitting a lost block").
#
# Why: the pinned R6 test (test_r14_stage1h_r6.py, test_fresh) fails if any results/ folder outside R6 holds an R6 unit,
# and R7a / R7b (frozen after R6) rerun R6's units by design. The worker (submit_s1h5.slurm) runs that test at job start,
# and both files are pinned (R6_code.sha256; the test also by R7a and R7b), so neither may change. A shadow root links
# every file of this project, and every results/ folder except the later runs'. The pinned code runs unchanged (the
# worker's root is the submit directory), its fresh-prompt check sees the runs that came before R6, and the new block is
# moved into the real results/ afterwards. Logs go to the real h0_measurement/logs/ (a link).
#
#   bash r6_shadow.sh build SHADOW     # SHADOW: a new or empty directory outside the project, on this cluster
#   cd SHADOW && R6_ONLY=h6qwen128_hm bash h0_measurement/bugs/14h_methodology_improve/script_stage1h_r6.sh --run-r6
#   bash r6_shadow.sh collect SHADOW   # after the job ends: move its new R6 block(s) into the real results/
set -euo pipefail

REAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
LATER=(r14r7a_ r14r7b_)   # runs frozen after R6 that rerun R6's units by design
CMD="${1:-}"; S="${2:-}"
[[ ( "$CMD" == build || "$CMD" == collect ) && -n "$S" ]] || { sed -n 2,14p "$0"; exit 2; }

case "$CMD" in
build)
  [[ ! -e "$S" || -z "$(ls -A "$S")" ]] || { echo "ERROR: $S is not empty" >&2; exit 1; }
  mkdir -p "$S"; S="$(cd "$S" && pwd -P)"
  [[ "$S/" != "$REAL/"* ]] || { echo "ERROR: the shadow must be outside $REAL" >&2; exit 1; }
  mkdir -p "$S/h0_measurement/results"
  shopt -s dotglob nullglob
  for e in "$REAL"/*; do [[ "$(basename "$e")" == h0_measurement ]] || ln -s "$e" "$S/"; done
  for e in "$REAL"/h0_measurement/*; do [[ "$(basename "$e")" == results ]] || ln -s "$e" "$S/h0_measurement/"; done
  n=0; k=0
  for e in "$REAL"/h0_measurement/results/*; do
    b="$(basename "$e")"; skip=0
    for p in "${LATER[@]}"; do [[ "$b" == "$p"* ]] && skip=1; done
    if (( skip )); then k=$((k + 1)); else ln -s "$e" "$S/h0_measurement/results/"; n=$((n + 1)); fi
  done
  echo "shadow root $S -> $REAL"
  echo "results/: $n folders linked, $k of later runs (${LATER[*]}) left out"
  ;;
collect)
  S="$(cd "$S" && pwd -P)"
  shopt -s nullglob
  m=0
  for d in "$S"/h0_measurement/results/r14s1h_h6*; do
    [[ -L "$d" ]] && continue
    t="$REAL/h0_measurement/results/$(basename "$d")"
    [[ ! -e "$t" ]] || { echo "ERROR: $t exists; not moved" >&2; exit 1; }
    mv "$d" "$t"; echo "moved $(basename "$d") into $REAL/h0_measurement/results/"; m=$((m + 1))
  done
  (( m )) || echo "no new R6 block in $S/h0_measurement/results/"
  ;;
esac
