#!/usr/bin/env bash
# R8/R9 follow-up after the completed five-cell campaign (jobs 978479--978489).
#
# WHY THIS FILE EXISTS
# --------------------
# The completed campaign already passed its implementation/provenance checks.
# Do not rerun it merely to "reproduce" the table. Its main unresolved mechanism
# question is P-5: did router_calib fail because its offline routes do not
# transfer, or does the output-error routing objective itself fail on Llama?
#
# This script runs ONE inexpensive diagnostic at llama31-8b @32K, B=2:
#
#   router_calib  fixed routes learned from disjoint prompts 0..9
#   router_oracle chooses, on each evaluation prompt and KV head, the candidate
#                 with the smallest measured attention-output error
#
# router_oracle is an OUTPUT-ERROR oracle. It sees the FP answer queries from the
# evaluation prompt and minimizes ||o_compressed-o_FP||/||o_FP|| per head. It is
# not an end-task oracle and does not directly choose the policy that maximizes
# exact answer accuracy. It is an unavailable upper bound on this routing proxy.
#
# RUN THESE STEPS IN ORDER FROM THE PROJECT ROOT OR FROM ANY DIRECTORY:
#
#   1. Print the exact sbatch command; submit nothing:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --oracle-dry
#
#   2. Submit the one diagnostic job:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --oracle-submit
#      The script prints ORACLE_JOB_ID=<id>. Save that integer.
#
#   3. Check it without blocking:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --oracle-status <id>
#      To follow the complete live log separately:
#        tail -f h0_measurement/logs/r8_<id>.out
#
#   4. After Slurm reports COMPLETED, read only this diagnostic:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --oracle-read <id>
#      The reader output is also saved as:
#        h0_measurement/bugs/9_sota_eviction_baselines/oracle_<id>.txt
#
# DECISION AFTER STEP 4
# ---------------------
#   oracle ~= uniform, oracle >> router_calib
#       Offline calibration is the main failure. Inspect prompt/task route
#       variation and only then consider a theta sweep or learned router.
#
#   oracle is also far below uniform
#       Better calibration cannot rescue the current routing formulation. The
#       local output-error proxy, independent per-head composition, or both do
#       not preserve task accuracy. Stop threshold sweeps and revise the design.
#
#   oracle offers only a few points over router_calib
#       Routing headroom is small; do not spend on a new five-cell grid.
#
# WHY 32K, B=2, AND PROMPTS 300..309?
# -----------------------------------
# Llama 32K/B=2 has the same large router failure as the expensive 128K cell;
# 32K is cheaper. B=2 is where the failure is identifiable. Prompts 300..309
# are disjoint from calibration 0..9 and prior evaluation 100..119. The task set
# must exactly match the route file's four-task calibration metadata.
#
# CONDITIONAL LATER WORK -- NOT IMPLEMENTED, DO NOT RUN YET
# ---------------------------------------------------------
# A harder-task rerun is needed only if the goal is a new non-ceiling benchmark
# or if the oracle shows that routing has useful headroom. It is not needed to
# report the current negative result.
#
# The current CLI hardcodes n_keys=n_values=n_hops=4. Before a hard-task run,
# add --n-keys/--n-values/--n-hops through run_r8.py, submit_r8.slurm, and the
# campaign script. Stamp the task configuration into result rows, sidecars,
# route metadata, route filenames, completion guards, and reader provenance.
# Otherwise an old four-key route could be silently reused for an eight-key run.
#
# Prefer hardening the tasks at the existing B=2 quantizer width. Do not simply
# choose B=1.5: uniform has no 1.5-bit quantizer and is currently skipped there.
# First gate a single 32K cell with fp/uniform/evict; require FP >= .95 and
# uniform accuracy in roughly [.50,.80] before any new router/SOTA campaign.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant}"
cd "$PROJECT_ROOT"

PY="${SIEVE_VENV:-$PROJECT_ROOT/.venv}/bin/python"
WORKER="h0_measurement/submit_r8.slurm"
READER="h0_measurement/bugs/8_router_endtask/read_r8.py"
ROUTES="h0_measurement/results/r8_routes/llama31-8b_32768_qa.json"
REPORT_DIR="h0_measurement/bugs/9_sota_eviction_baselines"

ARMS="fp,uniform,evict,interior,interior_cascade,router_calib,router_oracle"
TASKS="niah_single,niah_multikey,niah_multivalue,vt"

usage() {
  cat <<'USAGE'
usage:
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --oracle-dry
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --oracle-submit
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --oracle-status JOB_ID
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --oracle-read JOB_ID

This script intentionally does not rerun the completed five-cell campaign.
Read the ordered instructions and decision table at the top of the file.
USAGE
}

need_file() {
  [[ -f "$1" ]] || { echo "ERROR: missing $1" >&2; exit 1; }
}

need_job_id() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: a numeric Slurm JOB_ID is required" >&2
    usage >&2
    exit 2
  }
}

oracle_command() {
  printf '%q ' sbatch --parsable --time=02:00:00 "$WORKER" \
    R8_RUN_PREFIX=r8diag \
    R8_MODEL=llama31-8b \
    R8_CTX=32768 \
    R8_ARMS="$ARMS" \
    R8_BUDGETS=2 \
    R8_TASKS="$TASKS" \
    R8_N_PROMPTS=10 \
    R8_PROMPT_OFFSET=300 \
    R8_HEAD_ERROR=1 \
    R8_QA=1 \
    R8_THETA=1.0 \
    R8_ROUTES="$ROUTES"
  printf '\n'
}

submit_oracle() {
  local output job_id
  output=$(sbatch --parsable --time=02:00:00 "$WORKER" \
    R8_RUN_PREFIX=r8diag \
    R8_MODEL=llama31-8b \
    R8_CTX=32768 \
    R8_ARMS="$ARMS" \
    R8_BUDGETS=2 \
    R8_TASKS="$TASKS" \
    R8_N_PROMPTS=10 \
    R8_PROMPT_OFFSET=300 \
    R8_HEAD_ERROR=1 \
    R8_QA=1 \
    R8_THETA=1.0 \
    R8_ROUTES="$ROUTES")
  job_id="${output%%;*}"
  job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse a Slurm job id from: $output" >&2
    exit 1
  }
  echo "submitted output-error oracle diagnostic"
  echo "ORACLE_JOB_ID=$job_id"
  echo "log:    h0_measurement/logs/r8_${job_id}.out"
  echo "result: h0_measurement/results/r8diag${job_id}/"
  echo "next:   bash $REPORT_DIR/steps.sh --oracle-status $job_id"
}

status_oracle() {
  local job_id=$1 log="h0_measurement/logs/r8_${1}.out"
  if command -v squeue >/dev/null 2>&1; then
    squeue -j "$job_id" || true
  fi
  if command -v sacct >/dev/null 2>&1; then
    sacct -j "$job_id" --format=JobID,State,Elapsed,ExitCode || true
  fi
  if [[ -f "$log" ]]; then
    echo
    echo "last 25 log lines:"
    tail -n 25 "$log"
  else
    echo "log not created yet: $log"
  fi
}

read_oracle() {
  local job_id=$1
  local parquet="h0_measurement/results/r8diag${job_id}/r8_llama31-8b_32768.parquet"
  local out="$REPORT_DIR/oracle_${job_id}.txt"
  need_file "$parquet"
  env OMP_NUM_THREADS=8 "$PY" "$READER" "$parquet" --p2 > "$out"
  cat "$out"
  echo
  echo "saved: $out"
}

MODE="${1:-}"
case "$MODE" in
  --oracle-dry)
    need_file "$WORKER"
    need_file "$ROUTES"
    need_file "$READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    echo "DRY RUN; submit nothing:"
    oracle_command
    ;;
  --oracle-submit)
    need_file "$WORKER"
    need_file "$ROUTES"
    need_file "$READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
    submit_oracle
    ;;
  --oracle-status)
    need_job_id "${2:-}"
    status_oracle "$2"
    ;;
  --oracle-read)
    need_job_id "${2:-}"
    need_file "$READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    read_oracle "$2"
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    echo "ERROR: unknown mode '$MODE'" >&2
    usage >&2
    exit 2
    ;;
esac
