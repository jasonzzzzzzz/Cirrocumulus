#!/usr/bin/env bash
# R13 channel axis: excluded pilot, then the 6-cell main array, then the reader.
# Protocol and frozen decision table: plan.md (written before any R13 output).
#
# All cluster work runs on trig-login01 (GPU submits are refused elsewhere).
#
#   1. Preflight (CPU tests + corpus + sbatch --test-only), no submission:
#        bash h0_measurement/bugs/13_channel_axis/script.sh --run-dry
#   2. Submit the whole chain; each step waits on the previous with afterok:
#        bash h0_measurement/bugs/13_channel_axis/script.sh --run
#          pilot array 0-0 (qwen3-1.7b @2048)  -> gate (reader --pilot)
#          -> main array 0-5 (one cell per task, all arms in one process)
#          -> analysis (reader --main, writes main_<MAIN>.{json,txt})
#   3. Watch / cancel with the four printed IDs:
#        bash .../script.sh --status PILOT GATE MAIN READ
#        bash .../script.sh --cancel PILOT GATE MAIN READ
#   Manual re-read (idempotent): --main-read MAIN
#
# Gate and analysis jobs use `compute`, not `debug`: the trig debug QOS holds
# ~1 queued job per user.
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
REMOTE_HOST="trig-login01"
SSH=(ssh -o BatchMode=yes "$REMOTE_HOST")
DIR="h0_measurement/bugs/13_channel_axis"
READER="$DIR/read_r13.py"
WORKER="$DIR/submit_r13.slurm"
PY=".venv/bin/python"
PILOT_SBATCH="--array=0-0 --time=00:45:00"
MAIN_SBATCH="--array=0-5 --time=05:00:00"
READER_SBATCH="--partition=compute --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:30:00"

remote_run() { "${SSH[@]}" "cd '$PROJECT_ROOT' && $1"; }

need_id() { [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo "ERROR: job IDs must be decimal" >&2; exit 2; }; }

preflight() {
  remote_run "
    export OMP_NUM_THREADS=8 &&
    test -x '$PY' && bash -n '$WORKER' &&
    grep -Fqx '#SBATCH --array=0-5' '$WORKER' &&
    '$PY' -u '$DIR/test_r13.py' &&
    '$PY' h0_measurement/prefetch_corpus.py --verify --out .h0_corpus/pg19 &&
    mkdir -p h0_measurement/logs
  "
  echo "R13 preflight passed on $REMOTE_HOST"
}

submit() {   # run one sbatch on trig-login01, print only its job ID
  local out id
  out=$(remote_run "sbatch --parsable $1")
  id="${out%%;*}"; id="${id##* }"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "ERROR: could not parse a job ID from: $out" >&2; exit 1; }
  printf '%s' "$id"
}

pilot_cmd() { echo "$PILOT_SBATCH $WORKER pilot"; }
gate_cmd() { echo "--dependency=afterok:$1 --job-name=sieve-r13-gate $READER_SBATCH --output=h0_measurement/logs/r13gate_%j.out --error=h0_measurement/logs/r13gate_%j.err --wrap \"cd $PROJECT_ROOT && export OMP_NUM_THREADS=8 && $PY -u $READER --pilot $1\""; }
main_cmd() { echo "--dependency=afterok:$1 $MAIN_SBATCH $WORKER main"; }
read_cmd() { echo "--dependency=afterany:$1 --job-name=sieve-r13-read $READER_SBATCH --output=h0_measurement/logs/r13read_%j.out --error=h0_measurement/logs/r13read_%j.err --wrap \"cd $PROJECT_ROOT && export OMP_NUM_THREADS=8 && $PY -u $READER --main $1\""; }

case "${1:-}" in
  --preflight)
    preflight
    ;;
  --run-dry)
    preflight
    echo "DRY RUN; the chain --run submits, in order:"
    echo "  1. sbatch --parsable $(pilot_cmd)"
    echo "  2. sbatch --parsable $(gate_cmd PILOT)"
    echo "  3. sbatch --parsable $(main_cmd GATE)"
    echo "  4. sbatch --parsable $(read_cmd MAIN)"
    remote_run "sbatch --test-only $PILOT_SBATCH '$WORKER' pilot" 2>&1 | tail -n 1
    remote_run "sbatch --test-only $MAIN_SBATCH '$WORKER' main" 2>&1 | tail -n 1
    remote_run "sbatch --test-only $READER_SBATCH --wrap true" 2>&1 | tail -n 1
    ;;
  --run)
    preflight
    pilot=$(submit "$(pilot_cmd)");            echo "R13_PILOT_JOB_ID=$pilot"
    gate=$(submit "$(gate_cmd "$pilot")");     echo "R13_GATE_JOB_ID=$gate"
    main=$(submit "$(main_cmd "$gate")");      echo "R13_MAIN_JOB_ID=$main"
    rd=$(submit "$(read_cmd "$main")");        echo "R13_READ_JOB_ID=$rd"
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
  --main-read)
    (($# == 2)) || { echo "ERROR: $1 needs one JOB_ID" >&2; exit 2; }
    need_id "$2"
    remote_run "OMP_NUM_THREADS=8 '$PY' -u '$READER' --main '$2'"
    ;;
  *)
    sed -n '2,22p' "$0" >&2
    exit 2
    ;;
esac
