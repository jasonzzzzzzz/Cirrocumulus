#!/usr/bin/env bash
# R9 physical-KV-group K* qualification.
#
# Run these commands from any machine that can SSH to trig-login01.  The script
# always performs cluster work through that login node and uses the shared
# project path below.
#
#   1. Reproduce the authenticated CPU audit of the legacy 24-cell evidence:
#        bash h0_measurement/bugs/10_kstar_budget/steps.sh --audit-existing
#
#   2. Check the new source/worker interface and print the exact sbatch command.
#      This submits nothing:
#        bash h0_measurement/bugs/10_kstar_budget/steps.sh --qual-dry
#
#   3. Submit exactly one two-task Slurm array (task 0=Llama GQA,
#      task 1=Qwen MHA) and save the printed R9_QUAL_JOB_ID:
#        bash h0_measurement/bugs/10_kstar_budget/steps.sh --qual-submit
#
#   4. Check both array tasks without blocking:
#        bash h0_measurement/bugs/10_kstar_budget/steps.sh --qual-status JOB_ID
#
#   5. Only after both tasks are COMPLETED, authenticate both exact result
#      directories, apply Q1--Q4, and save the report/CSV/conditional lock:
#        bash h0_measurement/bugs/10_kstar_budget/steps.sh --qual-read JOB_ID
#
# A valid scientific gate failure exits zero with decision
# `stop_r9_qualification` and writes no advance lock.  Authentication, schema,
# source, or provenance failures exit nonzero and must be repaired before the
# same frozen split is rerun.  Do not submit development without an
# authenticated `advance_development` lock.

set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
REMOTE_HOST="trig-login01"
SSH=(ssh -o BatchMode=yes "$REMOTE_HOST")
REPORT_DIR="h0_measurement/bugs/10_kstar_budget"
AUDIT="$REPORT_DIR/audit_existing.py"
READER="$REPORT_DIR/read_qualification.py"
INPUT_MANIFEST="$REPORT_DIR/qualification_input_manifest.json"
SOURCE_LEDGER="$REPORT_DIR/source_ledger.json"
RUNNER="h0_measurement/run_kstar_budget_qualification.py"
WORKER="h0_measurement/submit_kstar_budget_qualification.slurm"
PY=".venv/bin/python"

usage() {
  cat <<'USAGE'
usage:
  bash h0_measurement/bugs/10_kstar_budget/steps.sh --audit-existing
  bash h0_measurement/bugs/10_kstar_budget/steps.sh --qual-dry
  bash h0_measurement/bugs/10_kstar_budget/steps.sh --qual-submit
  bash h0_measurement/bugs/10_kstar_budget/steps.sh --qual-status JOB_ID
  bash h0_measurement/bugs/10_kstar_budget/steps.sh --qual-read JOB_ID
USAGE
}

need_job_id() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: JOB_ID must be a positive decimal Slurm job ID" >&2
    exit 2
  }
}

# Arguments to this helper are fixed by this file, except for job IDs already
# restricted to decimal digits.  Keeping the remote command in one argument
# makes `cd` and the requested operation run in the same remote shell.
remote_run() {
  local command=$1
  "${SSH[@]}" "cd '$PROJECT_ROOT' && $command"
}

preflight_command="
  test -x '$PY' &&
  test -f '$AUDIT' && test ! -L '$AUDIT' &&
  test -f '$READER' && test ! -L '$READER' &&
  test -f '$INPUT_MANIFEST' && test ! -L '$INPUT_MANIFEST' &&
  test -f '$SOURCE_LEDGER' && test ! -L '$SOURCE_LEDGER' &&
  test -f '$RUNNER' && test ! -L '$RUNNER' &&
  test -f '$WORKER' && test ! -L '$WORKER' &&
  grep -Fqx '#SBATCH --array=0-1' '$WORKER' &&
  '$PY' -m py_compile '$READER' '$RUNNER' &&
  bash -n '$WORKER'"

gpu_preflight_command="$preflight_command &&
  '$PY' '$READER' --preflight \\
    --input-manifest '$INPUT_MANIFEST' --source-ledger '$SOURCE_LEDGER'
"

qual_command() {
  printf "ssh -o BatchMode=yes %q %q\n" "$REMOTE_HOST" \
    "cd '$PROJECT_ROOT' && sbatch --parsable '$WORKER'"
}

audit_existing() {
  # The canonical audit uses Arrow's schema reader.  It intentionally runs in
  # the cluster module environment rather than the experiment virtualenv,
  # whose Parquet engine is fastparquet.
  remote_run "module --force purge >/dev/null 2>&1 && module load StdEnv/2026 python/3.14 arrow/25.0.1 >/dev/null 2>&1 && python '$AUDIT'"
}

qual_dry() {
  remote_run "$gpu_preflight_command"
  echo "DRY RUN; submit nothing. Exact one-array command:"
  qual_command
  echo "array task 0: llama31-8b (GQA, n_rep=4)"
  echo "array task 1: qwen15-moe-a2.7b (MHA, n_rep=1)"
  echo "results: h0_measurement/results/r9_kstar_qualification_JOB_ID_{0,1}/"
}

qual_submit() {
  local output job_id
  remote_run "$gpu_preflight_command"
  output=$(remote_run "sbatch --parsable '$WORKER'")
  job_id="${output%%;*}"
  job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse one Slurm array job ID from: $output" >&2
    exit 1
  }
  echo "submitted R9 K* qualification as one two-task array"
  echo "R9_QUAL_JOB_ID=$job_id"
  echo "task 0: llama31-8b; task 1: qwen15-moe-a2.7b"
  echo "logs: h0_measurement/logs/r9kstarq_${job_id}_{0,1}.{out,err}"
  echo "next: bash $REPORT_DIR/steps.sh --qual-status $job_id"
}

qual_status() {
  local job_id=$1
  remote_run "
    if command -v squeue >/dev/null 2>&1; then squeue -j '$job_id' || true; fi
    if command -v sacct >/dev/null 2>&1; then
      sacct -j '$job_id' --format=JobID,State,Elapsed,MaxRSS,ExitCode || true
    fi
    for task in 0 1; do
      out='h0_measurement/logs/r9kstarq_${job_id}_'\"\$task\"'.out'
      err='h0_measurement/logs/r9kstarq_${job_id}_'\"\$task\"'.err'
      echo
      echo \"task \$task stdout (last 30 lines):\"
      if test -f \"\$out\"; then tail -n 30 \"\$out\"; else echo \"not created: \$out\"; fi
      if test -s \"\$err\"; then
        echo \"task \$task stderr (last 30 lines):\"
        tail -n 30 \"\$err\"
      fi
    done
  "
}

qual_read() {
  local job_id=$1
  local text_path="$REPORT_DIR/qualification_${job_id}.txt"
  local csv_path="$REPORT_DIR/qualification_${job_id}_summary.csv"
  local lock_path="$REPORT_DIR/qualification_advance_lock_${job_id}.json"
  remote_run "$preflight_command"
  remote_run "
    tmp='$text_path.tmp'
    rm -f \"\$tmp\"
    if '$PY' '$READER' --job-id '$job_id' \\
         --results-root h0_measurement/results \\
         --input-manifest '$INPUT_MANIFEST' \\
         --source-ledger '$SOURCE_LEDGER' \\
         --csv '$csv_path' --lock '$lock_path' > \"\$tmp\"; then
      mv \"\$tmp\" '$text_path'
    else
      status=\$?
      cat \"\$tmp\" >&2 || true
      rm -f \"\$tmp\"
      exit \"\$status\"
    fi
    cat '$text_path'
    echo
    echo 'saved: $text_path'
    echo 'saved: $csv_path'
    if test -f '$lock_path'; then
      echo 'saved: $lock_path'
      echo 'next: development is authorized only through this lock'
    else
      echo 'lock: none (the authenticated qualification did not advance)'
    fi
  "
}

mode="${1:-}"
case "$mode" in
  --audit-existing)
    (($# == 1)) || { echo "ERROR: --audit-existing accepts no arguments" >&2; exit 2; }
    audit_existing
    ;;
  --qual-dry)
    (($# == 1)) || { echo "ERROR: --qual-dry accepts no arguments" >&2; exit 2; }
    qual_dry
    ;;
  --qual-submit)
    (($# == 1)) || { echo "ERROR: --qual-submit accepts no arguments" >&2; exit 2; }
    qual_submit
    ;;
  --qual-status)
    (($# == 2)) || { echo "ERROR: --qual-status requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    qual_status "$2"
    ;;
  --qual-read)
    (($# == 2)) || { echo "ERROR: --qual-read requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    qual_read "$2"
    ;;
  -h|--help)
    (($# == 1)) || { echo "ERROR: --help accepts no arguments" >&2; exit 2; }
    usage
    ;;
  "")
    echo "ERROR: choose one explicit step; bare execution is disabled" >&2
    usage >&2
    exit 2
    ;;
  *)
    echo "ERROR: unknown mode '$mode'" >&2
    usage >&2
    exit 2
    ;;
esac
