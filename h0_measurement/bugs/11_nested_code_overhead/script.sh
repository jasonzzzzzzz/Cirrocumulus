#!/usr/bin/env bash
# R11 nested-code rate overhead: excluded pilot, then the locked 4-cell primary
# array.  Protocol: plan.md (frozen before any R11 output) + amendments A1, A2.
# Layout (A2): each cell is ONE run_h0 process measuring both codebooks from the
# same captured tensors; results land in h0_measurement/results/r11ab_*.
#
# All cluster work runs on trig-login01 (GPU submits are refused elsewhere).
#
#   1. Seal the source ledger after the LAST edit to any ledgered source
#      (quant/alloc/evict/prompts/probe, run_h0, models.yaml, the worker,
#      plan.md, the reader, the corpus manifest).  This file is deliberately
#      NOT ledgered, so fixing the control sheet never invalidates a run.
#        bash h0_measurement/bugs/11_nested_code_overhead/script.sh --seal
#   2. Preflight (tests + ledger + corpus + sbatch --test-only), no submission:
#        bash h0_measurement/bugs/11_nested_code_overhead/script.sh --run-dry
#   3. Submit the whole chain; each step waits on the previous with afterok:
#        bash h0_measurement/bugs/11_nested_code_overhead/script.sh --run
#          pilot array 0-0 -> gate (reader --pilot-ab, writes lock)
#          -> main array 0-3 (one cell per task, both codebooks in one pass)
#          -> analysis (reader --main-ab)
#   4. Watch / cancel with the four printed IDs:
#        bash .../script.sh --status PILOT GATE MAIN READ
#        bash .../script.sh --cancel PILOT GATE MAIN READ
#   Manual re-reads (idempotent): --pilot-read PILOT, --main-read MAIN
#   (in-process layout; the two-run reader modes remain for jobs 987079/987153)
#
# Gate and analysis jobs use `compute`, not `debug`: the trig debug QOS holds
# ~1 queued job per user (R10 lost its analysis submission to that limit).
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
REMOTE_HOST="trig-login01"
SSH=(ssh -o BatchMode=yes "$REMOTE_HOST")
DIR="h0_measurement/bugs/11_nested_code_overhead"
READER="$DIR/read_nested_code.py"
WORKER="h0_measurement/submit_r11_nested_code.slurm"
PY=".venv/bin/python"
PILOT_SBATCH="--array=0-0 --time=01:00:00"
MAIN_SBATCH="--array=0-3 --time=06:00:00"
READER_SBATCH="--partition=compute --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:45:00"

remote_run() { "${SSH[@]}" "cd '$PROJECT_ROOT' && $1"; }

need_id() { [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo "ERROR: job IDs must be decimal" >&2; exit 2; }; }

preflight() {
  remote_run "
    export OMP_NUM_THREADS=8 &&
    test -x '$PY' && bash -n '$WORKER' &&
    grep -Fqx '#SBATCH --array=0-3' '$WORKER' &&
    '$PY' -u tests/test_r11_nested_codebook.py &&
    '$PY' -u tests/test_r10_tier_panel.py &&
    '$PY' h0_measurement/prefetch_corpus.py --verify --out .h0_corpus/pg19 &&
    '$PY' '$READER' --preflight
  "
  echo "R11 preflight passed on $REMOTE_HOST"
}

submit() {   # run one sbatch on trig-login01, print only its job ID
  local out id
  out=$(remote_run "sbatch --parsable $1")
  id="${out%%;*}"; id="${id##* }"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "ERROR: could not parse a job ID from: $out" >&2; exit 1; }
  printf '%s' "$id"
}

gate_cmd() { echo "--dependency=afterok:$1 --job-name=sieve-r11-gate $READER_SBATCH --output=h0_measurement/logs/r11gate_%j.out --error=h0_measurement/logs/r11gate_%j.err --wrap \"cd $PROJECT_ROOT && export OMP_NUM_THREADS=8 && $PY -u $READER --pilot-ab $1\""; }
main_cmd() { echo "--dependency=afterok:$1 $MAIN_SBATCH $WORKER main $2"; }
read_cmd() { echo "--dependency=afterok:$1 --job-name=sieve-r11-read $READER_SBATCH --output=h0_measurement/logs/r11read_%j.out --error=h0_measurement/logs/r11read_%j.err --wrap \"cd $PROJECT_ROOT && export OMP_NUM_THREADS=8 && $PY -u $READER --main-ab $1\""; }

case "${1:-}" in
  --seal)
    remote_run "OMP_NUM_THREADS=8 '$PY' '$READER' --seal"
    ;;
  --preflight)
    preflight
    ;;
  --run-dry)
    preflight
    echo "DRY RUN; the chain --run submits, in order:"
    echo "  1. sbatch --parsable $PILOT_SBATCH $WORKER pilot"
    echo "  2. sbatch --parsable $(gate_cmd PILOT)"
    echo "  3. sbatch --parsable $(main_cmd GATE PILOT)"
    echo "  4. sbatch --parsable $(read_cmd MAIN)"
    remote_run "sbatch --test-only $PILOT_SBATCH '$WORKER' pilot" 2>&1 | tail -n 1
    remote_run "sbatch --test-only $MAIN_SBATCH '$WORKER' main 1" 2>&1 | tail -n 1
    remote_run "sbatch --test-only $READER_SBATCH --wrap true" 2>&1 | tail -n 1
    ;;
  --run)
    preflight
    pilot=$(submit "$PILOT_SBATCH '$WORKER' pilot");  echo "R11_PILOT_JOB_ID=$pilot"
    gate=$(submit "$(gate_cmd "$pilot")");             echo "R11_GATE_JOB_ID=$gate"
    main=$(submit "$(main_cmd "$gate" "$pilot")");     echo "R11_MAIN_JOB_ID=$main"
    rd=$(submit "$(read_cmd "$main")");                echo "R11_READ_JOB_ID=$rd"
    echo "next: bash $DIR/script.sh --status $pilot $gate $main $rd"
    ;;
  --status|--cancel)
    (($# == 5)) || { echo "ERROR: $1 needs PILOT GATE MAIN READ" >&2; exit 2; }
    for x in "$2" "$3" "$4" "$5"; do need_id "$x"; done
    if [[ "$1" == --status ]]; then
      remote_run "sacct -j '$2,$3,$4,$5' -X --format=JobID%16,JobName%18,State%26,Elapsed,ExitCode; squeue -j '$2,$3,$4,$5' -o '%.16i %.18j %.10T %.30R' 2>/dev/null || true"
    else
      remote_run "scancel '$2' '$3' '$4' '$5'"
    fi
    ;;
  --pilot-read|--main-read)
    (($# == 2)) || { echo "ERROR: $1 needs one JOB_ID" >&2; exit 2; }
    need_id "$2"
    remote_run "OMP_NUM_THREADS=8 '$PY' -u '$READER' ${1%-read}-ab '$2'"
    ;;
  *)
    sed -n '2,25p' "$0" >&2
    exit 2
    ;;
esac
