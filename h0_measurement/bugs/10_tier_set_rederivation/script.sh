#!/usr/bin/env bash
# R10 tier-set re-derivation: excluded implementation pilot, then one locked
# four-cell primary confirmation array.
#
# Run each command from the project root or any directory on a machine that can
# SSH to trig-login01.  All cluster operations use the shared absolute project
# path; bare execution is disabled.
#
#   1. Authenticate frozen sources, reader, corpus, model snapshots, and shell
#      interfaces without submitting a job:
#        bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --preflight
#
#   2. Print the exact excluded-pilot command, then submit it:
#        bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --pilot-dry
#        bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --pilot-submit
#      Save the printed R10_PILOT_JOB_ID.
#
#   3. Check the two pilot tasks without blocking.  After both are COMPLETED,
#      authenticate them and apply all five pilot acceptance checks:
#        bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --pilot-status PILOT_JOB
#        bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --pilot-read PILOT_JOB
#      Only a successful read emits pilot_advance_lock_PILOT_JOB.json.
#
#   4. Use that exact pilot job to inspect and submit the locked primary array:
#        bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --main-dry PILOT_JOB
#        bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --main-submit PILOT_JOB
#      Save the printed R10_MAIN_JOB_ID.  The wrapper reauthenticates the pilot;
#      every main worker also verifies and copies its pilot lock.
#
#   5. Check all four tasks.  After all are COMPLETED, authenticate and analyze
#      the B=3 primary decision, B=2 scope result, and sequential attribution:
#        bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --main-status MAIN_JOB
#        bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --main-read MAIN_JOB
#
# One-shot chain (the same four sbatch steps, linked by Slurm dependencies so
# nothing is submitted by hand after the pilot):
#        bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --seal
#        bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --run-dry
#        bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --run
#      --seal writes source_ledger.json from the current source bytes; rerun it
#      after ANY edit to a ledgered source (this file included).  --run submits
#        pilot array (0-1)  -> afterok -> pilot gate (reader --pilot, writes lock)
#        -> afterok -> main array (0-3) -> afterok -> main analysis (reader --main)
#      A failed pilot gate leaves the main array and analysis pending with
#      DependencyNeverSatisfied; cancel them with --cancel-chain.
#
# Pilot failure is an implementation/input failure and writes no lock.  A valid
# main design failure is a scientific result; do not retune a ladder on these
# prompts.  Follow the prespecified branch in plan.md.
set -euo pipefail

PROJECT_ROOT="/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant"
REMOTE_HOST="trig-login01"
SSH=(ssh -o BatchMode=yes "$REMOTE_HOST")
REPORT_DIR="h0_measurement/bugs/10_tier_set_rederivation"
CONTROL="$REPORT_DIR/script.sh"
PLAN="$REPORT_DIR/plan.md"
READER="$REPORT_DIR/read_tier_set.py"
SOURCE_LEDGER="$REPORT_DIR/source_ledger.json"
WORKER="h0_measurement/submit_r10_tier_set.slurm"
RUNNER="h0_measurement/run_h0.py"
ALLOC="sievelib/alloc.py"
TIER_TEST="tests/test_r10_tier_panel.py"
READER_TEST="tests/test_r10_tier_set_reader.py"
CONFIG="h0_measurement/models.yaml"
CORPUS_TOOL="h0_measurement/prefetch_corpus.py"
CORPUS=".h0_corpus/pg19"
PY=".venv/bin/python"

usage() {
  cat <<'USAGE'
usage:
  bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --preflight
  bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --pilot-dry
  bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --pilot-submit
  bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --pilot-status PILOT_JOB_ID
  bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --pilot-read PILOT_JOB_ID
  bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --main-dry PILOT_JOB_ID
  bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --main-submit PILOT_JOB_ID
  bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --main-status MAIN_JOB_ID
  bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --main-read MAIN_JOB_ID
  bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --seal
  bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --run-dry
  bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --run
  bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --chain-status PILOT GATE MAIN READ
  bash h0_measurement/bugs/10_tier_set_rederivation/script.sh --cancel-chain PILOT GATE MAIN READ
USAGE
}

need_job_id() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: JOB_ID must be a positive decimal Slurm job ID" >&2
    exit 2
  }
}

# Arguments passed into this helper are fixed by this file, except job IDs that
# have already been restricted to decimal digits.
remote_run() {
  local command=$1
  "${SSH[@]}" "cd '$PROJECT_ROOT' && $command"
}

preflight_command="
  test -x '$PY' &&
  test -f '$PLAN' && test ! -L '$PLAN' &&
  test -f '$READER' && test ! -L '$READER' &&
  test -f '$SOURCE_LEDGER' && test ! -L '$SOURCE_LEDGER' &&
  test -f '$WORKER' && test ! -L '$WORKER' &&
  test -f '$RUNNER' && test ! -L '$RUNNER' &&
  test -f '$ALLOC' && test ! -L '$ALLOC' &&
  test -f '$TIER_TEST' && test ! -L '$TIER_TEST' &&
  test -f '$READER_TEST' && test ! -L '$READER_TEST' &&
  test -f '$CONFIG' && test ! -L '$CONFIG' &&
  test -f '$CORPUS_TOOL' && test ! -L '$CORPUS_TOOL' &&
  test -f '$CONTROL' && test ! -L '$CONTROL' &&
  grep -Fqx '#SBATCH --array=0-3' '$WORKER' &&
  grep -Fqx '#SBATCH --time=05:00:00' '$WORKER' &&
  '$PY' -m py_compile '$RUNNER' '$ALLOC' '$READER' &&
  '$PY' '$TIER_TEST' &&
  '$PY' '$READER_TEST' &&
  bash -n '$WORKER' && bash -n '$CONTROL' &&
  grep -Fqx '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e' '.hf_cache/hub/models--Qwen--Qwen3-1.7B/refs/main' &&
  test -d '.hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e' &&
  grep -Fqx 'ec052fda178e241c7c443468d2fa1db6618996be' '.hf_cache/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/refs/main' &&
  test -d '.hf_cache/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be' &&
  grep -Fqx '0e9e39f249a16976918f6564b8830bc894c89659' '.hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/refs/main' &&
  test -d '.hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659' &&
  grep -Fqx 'b968826d9c46dd6066d109eabc6255188de91218' '.hf_cache/hub/models--Qwen--Qwen3-8B/refs/main' &&
  test -d '.hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218' &&
  grep -Fqx '0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe' '.hf_cache/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/refs/main' &&
  test -d '.hf_cache/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe' &&
  grep -Fqx 'c1899de289a04d12100db370d81485cdf75e47ca' '.hf_cache/hub/models--Qwen--Qwen3-0.6B/refs/main' &&
  test -d '.hf_cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca' &&
  grep -Fqx '9213176726f574b556790deb65791e0c5aa438b6' '.hf_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/refs/main' &&
  test -d '.hf_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6' &&
  grep -Fqx '4d14e384a4b037942bb3f3016665157c8bcb70ea' '.hf_cache/hub/models--Qwen--Qwen1.5-0.5B-Chat/refs/main' &&
  test -d '.hf_cache/hub/models--Qwen--Qwen1.5-0.5B-Chat/snapshots/4d14e384a4b037942bb3f3016665157c8bcb70ea' &&
  '$PY' '$CORPUS_TOOL' --verify --out '$CORPUS' &&
  '$PY' '$READER' --preflight --source-ledger '$SOURCE_LEDGER'
"

preflight() {
  remote_run "$preflight_command"
  echo "R10 preflight passed on $REMOTE_HOST"
}

pilot_command() {
  printf "ssh -o BatchMode=yes %q %q\n" "$REMOTE_HOST" \
    "cd '$PROJECT_ROOT' && sbatch --parsable --array=0-1 --time=02:00:00 '$WORKER' pilot"
}

main_command() {
  local pilot_job=$1
  printf "ssh -o BatchMode=yes %q %q\n" "$REMOTE_HOST" \
    "cd '$PROJECT_ROOT' && sbatch --parsable --array=0-3 --time=05:00:00 '$WORKER' main '$pilot_job'"
}

pilot_dry() {
  preflight
  echo "DRY RUN; submit nothing. Exact excluded-pilot command:"
  pilot_command
  echo "task 0: qwen3-1.7b @ 2048, cont, 1 prompt @ 0"
  echo "task 1: qwen15-moe-a2.7b @ 8192, cont, 1 prompt @ 0 (n_rep=1 control)"
  echo "results: h0_measurement/results/r10_tiers_pilot_JOB_ID_{0,1}/"
}

pilot_submit() {
  local output job_id
  preflight
  output=$(remote_run "sbatch --parsable --array=0-1 --time=02:00:00 '$WORKER' pilot")
  job_id="${output%%;*}"
  job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse one pilot array job ID from: $output" >&2
    exit 1
  }
  echo "submitted excluded R10 pilot as one two-task array"
  echo "R10_PILOT_JOB_ID=$job_id"
  echo "logs: h0_measurement/logs/r10tiers_${job_id}_{0,1}.{out,err}"
  echo "next: bash $CONTROL --pilot-status $job_id"
}

status() {
  local phase=$1 job_id=$2 last_task
  [[ "$phase" == pilot ]] && last_task=1 || last_task=3
  remote_run "
    if command -v squeue >/dev/null 2>&1; then squeue -j '$job_id' || true; fi
    if command -v sacct >/dev/null 2>&1; then
      sacct -j '$job_id' --format=JobID,State,Elapsed,MaxRSS,ExitCode || true
    fi
    for task in \$(seq 0 '$last_task'); do
      dir='h0_measurement/results/r10_tiers_${phase}_${job_id}_'\"\$task\"
      out='h0_measurement/logs/r10tiers_${job_id}_'\"\$task\"'.out'
      err='h0_measurement/logs/r10tiers_${job_id}_'\"\$task\"'.err'
      echo
      echo \"$phase task \$task: COMPLETE=\$(test -f \"\$dir/COMPLETE\" && echo yes || echo no)\"
      echo \"stdout (last 25 lines):\"
      if test -f \"\$out\"; then tail -n 25 \"\$out\"; else echo \"not created: \$out\"; fi
      if test -s \"\$err\"; then
        echo \"stderr (last 25 lines):\"
        tail -n 25 \"\$err\"
      fi
    done
  "
}

pilot_paths() {
  local job_id=$1
  PILOT_TEXT="$REPORT_DIR/pilot_${job_id}.txt"
  PILOT_CSV="$REPORT_DIR/pilot_${job_id}_summary.csv"
  PILOT_LOCK="$REPORT_DIR/pilot_advance_lock_${job_id}.json"
}

pilot_read() {
  local job_id=$1
  pilot_paths "$job_id"
  remote_run "$preflight_command"
  remote_run "
    '$PY' '$READER' --pilot '$job_id' \\
      --results-root h0_measurement/results \\
      --source-ledger '$SOURCE_LEDGER' \\
      --lock '$PILOT_LOCK' --txt '$PILOT_TEXT' --csv '$PILOT_CSV'
    test -f '$PILOT_LOCK' && test ! -L '$PILOT_LOCK'
    echo
    echo 'saved: $PILOT_TEXT'
    echo 'saved: $PILOT_CSV'
    echo 'saved: $PILOT_LOCK'
  "
}

# Rerun the strict pilot reader immediately before either printing or executing
# a main command.  It is idempotent and binds the lock to both artifact hashes
# and the current source-ledger hash, so a stale hand-written file cannot open
# the main array.
authenticate_pilot_for_main() {
  local job_id=$1
  pilot_paths "$job_id"
  remote_run "$preflight_command"
  remote_run "
    test -f '$PILOT_LOCK' && test ! -L '$PILOT_LOCK'
    '$PY' '$READER' --pilot '$job_id' \\
      --results-root h0_measurement/results \\
      --source-ledger '$SOURCE_LEDGER' \\
      --lock '$PILOT_LOCK' --txt '$PILOT_TEXT' --csv '$PILOT_CSV'
    test -f '$PILOT_LOCK' && test ! -L '$PILOT_LOCK'
  "
}

main_dry() {
  local pilot_job=$1
  authenticate_pilot_for_main "$pilot_job"
  echo "DRY RUN; submit nothing. Exact locked primary command:"
  main_command "$pilot_job"
  echo "task 0: llama31-8b @ 32768, 6 prompts @ 18"
  echo "task 1: llama31-8b @ 131072, 6 prompts @ 18"
  echo "task 2: qwen3-8b @ 8192, 6 prompts @ 18"
  echo "task 3: qwen3-30b-a3b-2507 @ 8192, 4 prompts @ 12"
  echo "results: h0_measurement/results/r10_tiers_main_JOB_ID_{0,1,2,3}/"
}

main_submit() {
  local pilot_job=$1 output job_id
  authenticate_pilot_for_main "$pilot_job"
  output=$(remote_run "sbatch --parsable --array=0-3 --time=05:00:00 '$WORKER' main '$pilot_job'")
  job_id="${output%%;*}"
  job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse one main array job ID from: $output" >&2
    exit 1
  }
  echo "submitted locked R10 primary panel as one four-task array"
  echo "R10_MAIN_JOB_ID=$job_id"
  echo "R10_PILOT_JOB_ID=$pilot_job"
  echo "logs: h0_measurement/logs/r10tiers_${job_id}_{0,1,2,3}.{out,err}"
  echo "next: bash $CONTROL --main-status $job_id"
}

main_read() {
  local job_id=$1
  local text_path="$REPORT_DIR/main_${job_id}.txt"
  local csv_path="$REPORT_DIR/main_${job_id}_summary.csv"
  remote_run "$preflight_command"
  remote_run "
    '$PY' '$READER' --main '$job_id' \\
      --results-root h0_measurement/results \\
      --source-ledger '$SOURCE_LEDGER' \\
      --txt '$text_path' --csv '$csv_path'
    echo
    echo 'saved: $text_path'
    echo 'saved: $csv_path'
  "
}

# ---------------------------------------------------------------- one-shot chain
# The reader-only steps (pilot gate, main analysis) run as short debug jobs so
# the whole chain is held by Slurm dependencies instead of by a human.  They
# request one GPU only because the trig cluster schedules by GPU; the reader
# itself is CPU-only.
REPORTER_SLURM=(--partition=debug --nodes=1 --gpus-per-node=1
                --ntasks-per-node=1 --cpus-per-task=16 --time=00:45:00)

seal() {
  # Hash every required source exactly as read_tier_set.py will verify it.
  remote_run "
    '$PY' - '$SOURCE_LEDGER' <<'PY'
import importlib.util, json, pathlib, sys
spec = importlib.util.spec_from_file_location('r10r', '$READER')
r = importlib.util.module_from_spec(spec); sys.modules['r10r'] = r
spec.loader.exec_module(r)
src = {rel: r.sha256_file(r.PROJECT / rel) for rel in sorted(r.REQUIRED_LEDGER_SOURCES)}
led = {'ledger_version': r.LEDGER_VERSION,
       'content_sha256': r.ledger_content_sha256(r.LEDGER_VERSION, src),
       'source_sha256': src}
out = pathlib.Path(sys.argv[1]); tmp = out.with_name(out.name + '.tmp')
tmp.write_text(json.dumps(led, indent=2, sort_keys=True) + '\n'); tmp.replace(out)
print(f'sealed {len(src)} sources -> {out}')
PY
    '$PY' '$READER' --preflight --source-ledger '$SOURCE_LEDGER'
  "
}

gate_wrap() {
  local pilot_job=$1
  pilot_paths "$pilot_job"
  printf '%s' "cd $PROJECT_ROOT && export OMP_NUM_THREADS=8 && $PY -u $READER --pilot $pilot_job --results-root h0_measurement/results --source-ledger $SOURCE_LEDGER --lock $PILOT_LOCK --txt $PILOT_TEXT --csv $PILOT_CSV && test -f $PILOT_LOCK"
}

read_wrap() {
  local main_job=$1
  printf '%s' "cd $PROJECT_ROOT && export OMP_NUM_THREADS=8 && $PY -u $READER --main $main_job --results-root h0_measurement/results --source-ledger $SOURCE_LEDGER --txt $REPORT_DIR/main_${main_job}.txt --csv $REPORT_DIR/main_${main_job}_summary.csv"
}

submit_remote() {
  # Submit one command on trig-login01 and return only the numeric job ID.
  local output job_id
  output=$(remote_run "$1")
  job_id="${output%%;*}"
  job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse a job ID from: $output" >&2
    exit 1
  }
  printf '%s' "$job_id"
}

run_dry() {
  preflight
  echo "DRY RUN; submit nothing. The chain --run would submit, in order:"
  echo "  1. sbatch --parsable --array=0-1 --time=02:00:00 $WORKER pilot"
  echo "  2. sbatch --parsable --dependency=afterok:<PILOT> --job-name=sieve-r10-gate ${REPORTER_SLURM[*]} --output=h0_measurement/logs/r10gate_%j.out --wrap \"$(gate_wrap PILOT)\""
  echo "  3. sbatch --parsable --dependency=afterok:<GATE> --array=0-3 --time=05:00:00 $WORKER main <PILOT>"
  echo "  4. sbatch --parsable --dependency=afterok:<MAIN> --job-name=sieve-r10-read ${REPORTER_SLURM[*]} --output=h0_measurement/logs/r10read_%j.out --wrap \"$(read_wrap MAIN)\""
  remote_run "sbatch --test-only --array=0-1 --time=02:00:00 '$WORKER' pilot" 2>&1 | tail -n 2
  remote_run "sbatch --test-only ${REPORTER_SLURM[*]} --wrap true" 2>&1 | tail -n 2
}

run_chain() {
  local pilot gate main reader
  preflight
  pilot=$(submit_remote "sbatch --parsable --array=0-1 --time=02:00:00 '$WORKER' pilot")
  echo "R10_PILOT_JOB_ID=$pilot"
  gate=$(submit_remote "sbatch --parsable --dependency=afterok:$pilot --job-name=sieve-r10-gate ${REPORTER_SLURM[*]} --output=h0_measurement/logs/r10gate_%j.out --error=h0_measurement/logs/r10gate_%j.err --wrap \"$(gate_wrap "$pilot")\"")
  echo "R10_GATE_JOB_ID=$gate"
  main=$(submit_remote "sbatch --parsable --dependency=afterok:$gate --array=0-3 --time=05:00:00 '$WORKER' main '$pilot'")
  echo "R10_MAIN_JOB_ID=$main"
  reader=$(submit_remote "sbatch --parsable --dependency=afterok:$main --job-name=sieve-r10-read ${REPORTER_SLURM[*]} --output=h0_measurement/logs/r10read_%j.out --error=h0_measurement/logs/r10read_%j.err --wrap \"$(read_wrap "$main")\"")
  echo "R10_READ_JOB_ID=$reader"
  echo "next: bash $CONTROL --chain-status $pilot $gate $main $reader"
}

chain_status() {
  remote_run "sacct -j '$1,$2,$3,$4' -X --format=JobID%18,JobName%18,State%24,Elapsed,ExitCode || true"
  remote_run "squeue -j '$1,$2,$3,$4' -o '%.18i %.18j %.10T %.30R' 2>/dev/null || true"
}

mode="${1:-}"
case "$mode" in
  --seal)
    (($# == 1)) || { echo "ERROR: --seal accepts no arguments" >&2; exit 2; }
    seal
    ;;
  --run-dry)
    (($# == 1)) || { echo "ERROR: --run-dry accepts no arguments" >&2; exit 2; }
    run_dry
    ;;
  --run)
    (($# == 1)) || { echo "ERROR: --run accepts no arguments" >&2; exit 2; }
    run_chain
    ;;
  --chain-status|--cancel-chain)
    (($# == 5)) || { echo "ERROR: $mode requires PILOT GATE MAIN READ job IDs" >&2; exit 2; }
    for id in "$2" "$3" "$4" "$5"; do need_job_id "$id"; done
    if [[ "$mode" == --chain-status ]]; then
      chain_status "$2" "$3" "$4" "$5"
    else
      remote_run "scancel '$2' '$3' '$4' '$5'"
    fi
    ;;
  --preflight)
    (($# == 1)) || { echo "ERROR: --preflight accepts no arguments" >&2; exit 2; }
    preflight
    ;;
  --pilot-dry)
    (($# == 1)) || { echo "ERROR: --pilot-dry accepts no arguments" >&2; exit 2; }
    pilot_dry
    ;;
  --pilot-submit)
    (($# == 1)) || { echo "ERROR: --pilot-submit accepts no arguments" >&2; exit 2; }
    pilot_submit
    ;;
  --pilot-status)
    (($# == 2)) || { echo "ERROR: --pilot-status requires one PILOT_JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    status pilot "$2"
    ;;
  --pilot-read)
    (($# == 2)) || { echo "ERROR: --pilot-read requires one PILOT_JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    pilot_read "$2"
    ;;
  --main-dry)
    (($# == 2)) || { echo "ERROR: --main-dry requires one PILOT_JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    main_dry "$2"
    ;;
  --main-submit)
    (($# == 2)) || { echo "ERROR: --main-submit requires one PILOT_JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    main_submit "$2"
    ;;
  --main-status)
    (($# == 2)) || { echo "ERROR: --main-status requires one MAIN_JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    status main "$2"
    ;;
  --main-read)
    (($# == 2)) || { echo "ERROR: --main-read requires one MAIN_JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    main_read "$2"
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
