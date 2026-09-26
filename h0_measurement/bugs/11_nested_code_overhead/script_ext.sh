#!/usr/bin/env bash
# R11-ext: MoE budget-2 tail (fresh Qwen3-30B prompts) + two new architectures
# (Mistral-7B, Qwen1.5-MoE).  Protocol: plan_ext.md (frozen before any output).
# Same in-process A/B layout as R11 A2.  All cluster work runs on trig-login01.
#
#   1. --seal       write source_ledger_ext.json (after the LAST ledgered edit)
#   2. --run-dry    tests + ledger + corpus + sbatch --test-only; submits nothing
#   3. --run        main array 0-9 -> analysis (reader --main); no pilot gate
#      --run-pilot  optional pilot array 0-1 -> gate (reader --pilot), not chained
#   4. --status / --cancel MAIN READ;  --pilot-read / --main-read JOB
#
# This file is NOT ledgered.  Gate/analysis jobs use `compute` (debug QOS ~1 job).
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
REMOTE_HOST="trig-login01"
SSH=(ssh -o BatchMode=yes "$REMOTE_HOST")
DIR="h0_measurement/bugs/11_nested_code_overhead"
READER="$DIR/read_nested_ext.py"
WORKER="h0_measurement/submit_r11_ext.slurm"
PY=".venv/bin/python"
PILOT_SBATCH="--array=0-1 --time=01:00:00"
MAIN_SBATCH="--array=0-9 --time=04:00:00"
READER_SBATCH="--partition=compute --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=16 --time=00:45:00"

remote_run() { "${SSH[@]}" "cd '$PROJECT_ROOT' && $1"; }

need_id() { [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo "ERROR: job IDs must be decimal" >&2; exit 2; }; }

preflight() {
  remote_run "
    export OMP_NUM_THREADS=8 &&
    test -x '$PY' && bash -n '$WORKER' &&
    grep -Fqx '#SBATCH --array=0-9' '$WORKER' &&
    '$PY' -u tests/test_r11_nested_codebook.py &&
    '$PY' -u tests/test_r10_tier_panel.py &&
    '$PY' h0_measurement/prefetch_corpus.py --verify --out .h0_corpus/pg19 &&
    '$PY' '$READER' --preflight
  "
  echo "R11-ext preflight passed on $REMOTE_HOST"
}

submit() {   # run one sbatch on trig-login01, print only its job ID
  local out id
  out=$(remote_run "sbatch --parsable $1")
  id="${out%%;*}"; id="${id##* }"
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "ERROR: could not parse a job ID from: $out" >&2; exit 1; }
  printf '%s' "$id"
}

gate_cmd() { echo "--dependency=afterok:$1 --job-name=sieve-r11x-gate $READER_SBATCH --output=h0_measurement/logs/r11xgate_%j.out --error=h0_measurement/logs/r11xgate_%j.err --wrap \"cd $PROJECT_ROOT && export OMP_NUM_THREADS=8 && $PY -u $READER --pilot $1\""; }
main_cmd() { echo "$MAIN_SBATCH $WORKER main"; }
read_cmd() { echo "--dependency=afterok:$1 --job-name=sieve-r11x-read $READER_SBATCH --output=h0_measurement/logs/r11xread_%j.out --error=h0_measurement/logs/r11xread_%j.err --wrap \"cd $PROJECT_ROOT && export OMP_NUM_THREADS=8 && $PY -u $READER --main $1\""; }

case "${1:-}" in
  --seal)
    remote_run "OMP_NUM_THREADS=8 '$PY' '$READER' --seal"
    ;;
  --preflight)
    preflight
    ;;
  --run-dry)
    preflight
    echo "DRY RUN; --run submits, in order:"
    echo "  1. sbatch --parsable $(main_cmd)"
    echo "  2. sbatch --parsable $(read_cmd MAIN)"
    remote_run "sbatch --test-only $MAIN_SBATCH '$WORKER' main" 2>&1 | tail -n 1
    remote_run "sbatch --test-only $READER_SBATCH --wrap true" 2>&1 | tail -n 1
    ;;
  --run)
    preflight
    main=$(submit "$(main_cmd)");                      echo "R11X_MAIN_JOB_ID=$main"
    rd=$(submit "$(read_cmd "$main")");                echo "R11X_READ_JOB_ID=$rd"
    echo "next: bash $DIR/script_ext.sh --status $main $rd"
    ;;
  --run-pilot)
    preflight
    pilot=$(submit "$PILOT_SBATCH '$WORKER' pilot");  echo "R11X_PILOT_JOB_ID=$pilot"
    gate=$(submit "$(gate_cmd "$pilot")");             echo "R11X_GATE_JOB_ID=$gate"
    ;;
  --status|--cancel)
    (($# >= 2)) || { echo "ERROR: $1 needs job IDs, e.g. MAIN READ" >&2; exit 2; }
    ids=("${@:2}")
    for x in "${ids[@]}"; do need_id "$x"; done
    csv=$(IFS=,; echo "${ids[*]}")
    if [[ "$1" == --status ]]; then
      remote_run "sacct -j '$csv' -X --format=JobID%16,JobName%18,State%26,Elapsed,ExitCode; squeue -j '$csv' -o '%.16i %.18j %.10T %.30R' 2>/dev/null || true"
    else
      remote_run "scancel ${ids[*]}"
    fi
    ;;
  --pilot-read|--main-read)
    (($# == 2)) || { echo "ERROR: $1 needs one JOB_ID" >&2; exit 2; }
    need_id "$2"
    remote_run "OMP_NUM_THREADS=8 '$PY' -u '$READER' ${1%-read} '$2'"
    ;;
  *)
    sed -n '2,13p' "$0" >&2
    exit 2
    ;;
esac
