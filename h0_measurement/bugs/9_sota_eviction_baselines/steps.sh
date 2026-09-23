#!/usr/bin/env bash
# R8/R9 follow-up after the completed five-cell campaign (jobs 978479--978489).
#
# WHY THIS FILE EXISTS
# --------------------
# The completed campaign already passed its implementation/provenance checks.
# Do not rerun it merely to "reproduce" the table. Its main unresolved mechanism
# question was P-5: did router_calib fail because its offline routes do not
# transfer, or does the output-error routing objective itself fail on Llama?
#
# RESULT (job 979308, completed 2026-09-22)
# ---------------------------------------------------------
# Both contribute. The oracle improves mean accuracy by +0.298 over calibrated
# routing [paired prompt-block 95% CI +0.139,+0.459], but remains -0.201 below
# uniform [-0.324,-0.085]. It minimizes the stated proxy (mean error 0.135 vs
# uniform 0.652) while losing task accuracy (0.761 vs 0.963). Do not resubmit
# this diagnostic. Read it with:
#   bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --oracle-read 979308
#
# This script defines ONE inexpensive diagnostic at llama31-8b @32K, B=2:
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
# NEXT ITERATION: NON-CEILING TASK SCREEN
# ---------------------------------------
# The task interface is now explicit and provenance-safe. Step 2 screens two
# independent configurations on prompts 400..409, Llama 32K, B=2, QA mode:
#
#   screen: k8/v6/h6 and k16/v8/h8, fp + uniform, three discriminating tasks
#   mid:    k16/v7/h7, multivalue + VT only, after level 6 was too easy and
#           level 8 failed the FP gate for those tasks
#
# Run:
#   bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --difficulty-dry
#   bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --difficulty-submit
#   bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --difficulty-status JOB [JOB...]
#   bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --difficulty-read JOB [JOB...]
#
# Each initial result must contain 60 rows; the midpoint must contain 40 rows.
# Every result must use one real-corpus SHA and pass the B=2 audit. FP >= .95
# is the eligibility gate, not a claim that every screened task passes.
# A useful task has uniform mean in [.50,.80], or a 90% interval overlapping
# that band at this screening size.
#
# DEVELOPMENT RESULT (jobs 980284, 980285, 980356, 980355)
# ---------------------------------------------------------------------------
# Only multikey at n_keys=16 passes. Multivalue v7 is too easy; v8 fails FP.
# VT h7 is censored; cap-fixed h8 is too easy. Do not add harder value/hop
# points after these failures. Confirm the fixed multikey point once:
#
#   1. Print the exact held-out command:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --confirm-dry
#   2. Submit it and save CONFIRM_JOB_ID:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --confirm-submit
#   3. Check completion:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --confirm-status JOB
#   4. After COMPLETED, audit and save the result:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --confirm-read JOB
#
# This confirmation is k16/v4/h4, multikey only, fp+uniform, B=2, prompts
# 420..439 (40 rows). It passes only with FP >= .95, no incomplete capped FP,
# and uniform in [.50,.80]. Do not tune n_keys on the confirmation prompts.
#
# CONFIRMATION RESULT (job 980414, completed 2026-09-23)
# ---------------------------------------------------------------------------
# Exact provenance and bits checks pass. FP=1.00, uniform=0.80; FP-uniform
# is +0.20 with paired 90% interval [+0.05,+0.35]. No incomplete FP row is
# capped. Do not resubmit; read with --confirm-read 980414. Step 3 is the
# whole-policy diagnostic specified in plan.md.
#
# STEP 3: WHOLE-POLICY DIAGNOSTIC
# --------------------------------
# The development split is fixed to prompts 440..459. It compares independently
# decoded fp/uniform/evict/interior accuracy and replays one shared, at-most
# eight-token FP teacher-forced trace for uniform/evict/interior. The replay
# writes summaries to a separate r8policy parquet; it never makes an oracle an
# arm or writes raw logits.
#
# Run the development split in this order:
#
#   1. Inspect the exact command (submits nothing):
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --policy-dry dev
#   2. Submit and save POLICY_JOB_ID:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --policy-submit dev
#   3. Check it without blocking:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --policy-status JOB
#   4. After COMPLETED, authenticate, analyze, and save all three reports:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --policy-read dev JOB
#
# The reader applies the frozen H/G gates in plan.md. Only if its decision is
# `advance_mean_kl` may the locked prompts 460..499 be submitted:
#
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --policy-dry confirm
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --policy-submit confirm
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --policy-status JOB
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --policy-read confirm JOB
#
# Do not submit `confirm` after `revise_candidates`, `reject_mean_kl`, or
# `more_development_prompts`. Those outcomes return to the matching conditional
# branch in plan.md using development data only.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant}"
cd "$PROJECT_ROOT"

PY="${SIEVE_VENV:-$PROJECT_ROOT/.venv}/bin/python"
WORKER="h0_measurement/submit_r8.slurm"
READER="h0_measurement/bugs/8_router_endtask/read_r8.py"
POLICY_READER="h0_measurement/bugs/9_sota_eviction_baselines/read_policy.py"
ROUTES="h0_measurement/results/r8_routes/llama31-8b_32768_qa.json"
REPORT_DIR="h0_measurement/bugs/9_sota_eviction_baselines"

ARMS="fp,uniform,evict,interior,interior_cascade,router_calib,router_oracle"
TASKS="niah_single,niah_multikey,niah_multivalue,vt"
DIFFICULTY_TASKS="niah_multikey,niah_multivalue,vt"
CONFIRM_TASKS="niah_multikey"
POLICY_ARMS="fp,uniform,evict,interior"
POLICY_CANDIDATES="uniform,evict,interior"

usage() {
  cat <<'USAGE'
usage:
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --oracle-dry
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --oracle-submit
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --oracle-status JOB_ID
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --oracle-read JOB_ID

  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --difficulty-dry [screen|easy|hard|mid|vt-hard]
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --difficulty-submit [screen|easy|hard|mid|vt-hard]
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --difficulty-status JOB_ID [JOB_ID...]
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --difficulty-read JOB_ID [JOB_ID...]

  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --confirm-dry
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --confirm-submit
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --confirm-status JOB_ID
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --confirm-read JOB_ID

  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --policy-dry [dev|confirm]
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --policy-submit [dev|confirm]
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --policy-status JOB_ID [JOB_ID...]
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --policy-read [dev|confirm] JOB_ID

The default selector "screen" addresses all three tasks at (8,6,6) and
(16,8,8). The post-screen "mid" selector reruns multivalue and VT at (16,7,7)
with difficulty-aware answer limits; "vt-hard" does the same for VT at h8.
The confirmation interface is fixed to the selected k16/v4/h4 multikey cell on
held-out prompts 420--439 and cannot accept a difficulty selector.
The policy development split is prompts 440--459. The policy confirmation split
is prompts 460--499 and must be run only after the development reader prints
`decision       advance_mean_kl`.
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
  local output job_id completed="h0_measurement/results/r8diag979308/r8_llama31-8b_32768.parquet"
  if [[ -f "$completed" ]]; then
    echo "oracle diagnostic already completed: job 979308"
    echo "read: bash $REPORT_DIR/steps.sh --oracle-read 979308"
    return 0
  fi
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

difficulty_specs() {
  case "${1:-screen}" in
    screen) printf '%s\n' "8 6 6 $DIFFICULTY_TASKS any" "16 8 8 $DIFFICULTY_TASKS any" ;;
    easy) printf '%s\n' "8 6 6 $DIFFICULTY_TASKS any" ;;
    hard) printf '%s\n' "16 8 8 $DIFFICULTY_TASKS any" ;;
    mid) printf '%s\n' "16 7 7 niah_multivalue,vt difficulty_v1" ;;
    vt-hard) printf '%s\n' "16 7 8 vt difficulty_v1" ;;
    *) echo "ERROR: difficulty selector must be screen, easy, hard, mid, or vt-hard" >&2; exit 2 ;;
  esac
}

grid_complete() {
  local nk=$1 nv=$2 nh=$3 tasks=$4 contract=$5
  local n_prompts=$6 prompt_offset=$7 run_prefix=$8
  "$PY" - "$nk" "$nv" "$nh" "$tasks" "$contract" \
    "$n_prompts" "$prompt_offset" "$run_prefix" <<'PY'
from collections import Counter
import glob
import json
import os
import sys

import pandas as pd

nk, nv, nh = map(int, sys.argv[1:4])
want_tasks = set(sys.argv[4].split(","))
contract = sys.argv[5]
n_prompts, prompt_offset = map(int, sys.argv[6:8])
run_prefix = sys.argv[8]
expected_rows = n_prompts * len(want_tasks) * 2
expected_prompts = set(range(prompt_offset, prompt_offset + n_prompts))
base = {"niah_single": 24, "niah_multikey": 24, "niah_multivalue": 64, "vt": 64}
expected_limits = {
    t: (base[t] + 8 * max(nv - 4, 0) if t == "niah_multivalue"
        else base[t] + 16 * max(nh - 4, 0) if t == "vt"
        else base[t])
    for t in want_tasks
}
tag = f"k{nk}_v{nv}_h{nh}"
want_cfg = {"n_keys": nk, "n_values": nv, "n_hops": nh}
pattern = (
    f"h0_measurement/results/{run_prefix}*/"
    f"r8_llama31-8b_32768_{tag}.json"
)

for js in sorted(glob.glob(pattern)):
    try:
        meta = json.load(open(js))
        parquet = js[:-5] + ".parquet"
        with open(parquet, "rb") as fh:
            fh.seek(-4, os.SEEK_END)
            complete = fh.read() == b"PAR1"

        columns = [
            "prompt_idx", "task", "arm", "B", "n_keys", "n_values", "n_hops",
            "corpus_sha", "synthetic", "bits_per_token", "question_agnostic",
        ]
        if contract != "any":
            columns += ["max_new_tokens", "gen_len", "reached_max_new"]
        frame = pd.read_parquet(parquet, columns=columns)
        rows = {column: frame[column].tolist() for column in columns}

        combos = Counter(zip(rows["prompt_idx"], rows["task"], rows["arm"]))
        expected_combos = {
            (prompt, task, arm)
            for prompt in expected_prompts
            for task in want_tasks
            for arm in {"fp", "uniform"}
        }
        row_contract = (
            len(frame) == expected_rows
            and set(rows["prompt_idx"]) == expected_prompts
            and set(rows["task"]) == want_tasks
            and set(rows["arm"]) == {"fp", "uniform"}
            and all(
                (arm == "fp" and float(budget) == 0.0)
                or (arm == "uniform" and float(budget) == 2.0)
                for arm, budget in zip(rows["arm"], rows["B"])
            )
            and all(bool(value) for value in rows["question_agnostic"])
            and set(combos) == expected_combos
            and all(count == 1 for count in combos.values())
            and set(rows["n_keys"]) == {nk}
            and set(rows["n_values"]) == {nv}
            and set(rows["n_hops"]) == {nh}
            and len(set(rows["corpus_sha"])) == 1
            and next(iter(set(rows["corpus_sha"]))) not in {"", "synthetic"}
            and not any(bool(x) for x in rows["synthetic"])
            and all(
                bits is None or float(bits) <= 2.0 + 1e-6
                for bits, arm in zip(rows["bits_per_token"], rows["arm"])
                if arm == "uniform"
            )
        )
        if contract != "any":
            row_contract = row_contract and all(
                int(limit) == expected_limits[task]
                and bool(reached) == (int(gen_len) >= int(limit))
                for limit, gen_len, reached, task in zip(
                    rows["max_new_tokens"], rows["gen_len"],
                    rows["reached_max_new"], rows["task"]
                )
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        continue

    if (
        meta.get("task_config") == want_cfg
        and meta.get("model") == "llama31-8b"
        and meta.get("ctx") == 32768
        and set(meta.get("tasks", [])) == want_tasks
        and set(meta.get("arms", [])) == {"fp", "uniform"}
        and {float(x) for x in meta.get("budgets", [])} == {2.0}
        and meta.get("n_prompts") == n_prompts
        and meta.get("prompt_offset") == prompt_offset
        and bool(meta.get("question_agnostic", False))
        and meta.get("rows") == expected_rows
        and meta.get("corpus_sha") == next(iter(set(rows["corpus_sha"])))
        and (
            contract == "any"
            or (
                meta.get("generation_limit_version") == contract
                and meta.get("generation_limits") == expected_limits
            )
        )
        and complete
        and row_contract
    ):
        print(os.path.dirname(js))
        raise SystemExit(0)
raise SystemExit(1)
PY
}

difficulty_complete() {
  local tag="k${1}_v${2}_h${3}"
  grid_complete "$1" "$2" "$3" "$4" "$5" 10 400 "r8diff_${tag}_"
}

confirmation_complete() {
  grid_complete 16 4 4 "$CONFIRM_TASKS" difficulty_v1 20 420 \
    "r8confirm_k16_v4_h4_"
}

difficulty_command() {
  local nk=$1 nv=$2 nh=$3 tasks=$4 tag="k${1}_v${2}_h${3}"
  printf '%q ' sbatch --parsable --time=00:30:00 "$WORKER" \
    R8_RUN_PREFIX="r8diff_${tag}_" R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS=fp,uniform R8_BUDGETS=2 R8_TASKS="$tasks" \
    R8_N_PROMPTS=10 R8_PROMPT_OFFSET=400 R8_QA=1 \
    R8_N_KEYS="$nk" R8_N_VALUES="$nv" R8_N_HOPS="$nh"
  printf '\n'
}

confirmation_command() {
  printf '%q ' sbatch --parsable --time=00:30:00 "$WORKER" \
    R8_RUN_PREFIX=r8confirm_k16_v4_h4_ \
    R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS=fp,uniform R8_BUDGETS=2 R8_TASKS="$CONFIRM_TASKS" \
    R8_N_PROMPTS=20 R8_PROMPT_OFFSET=420 R8_QA=1 \
    R8_N_KEYS=16 R8_N_VALUES=4 R8_N_HOPS=4
  printf '\n'
}

submit_difficulty_one() {
  local nk=$1 nv=$2 nh=$3 tasks=$4 contract=$5 tag="k${1}_v${2}_h${3}" output job_id where
  if where=$(difficulty_complete "$nk" "$nv" "$nh" "$tasks" "$contract"); then
    echo "difficulty $tag tasks=$tasks already complete: $where"
    return 0
  fi
  output=$(sbatch --parsable --time=00:30:00 "$WORKER" \
    R8_RUN_PREFIX="r8diff_${tag}_" R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS=fp,uniform R8_BUDGETS=2 R8_TASKS="$tasks" \
    R8_N_PROMPTS=10 R8_PROMPT_OFFSET=400 R8_QA=1 \
    R8_N_KEYS="$nk" R8_N_VALUES="$nv" R8_N_HOPS="$nh")
  job_id="${output%%;*}"; job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse a Slurm job id from: $output" >&2; exit 1; }
  echo "DIFFICULTY_JOB_ID=$job_id config=$tag tasks=$tasks contract=$contract"
}

submit_confirmation() {
  local output job_id where
  if where=$(confirmation_complete); then
    echo "held-out confirmation already complete: $where"
    return 0
  fi
  output=$(sbatch --parsable --time=00:30:00 "$WORKER" \
    R8_RUN_PREFIX=r8confirm_k16_v4_h4_ \
    R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS=fp,uniform R8_BUDGETS=2 R8_TASKS="$CONFIRM_TASKS" \
    R8_N_PROMPTS=20 R8_PROMPT_OFFSET=420 R8_QA=1 \
    R8_N_KEYS=16 R8_N_VALUES=4 R8_N_HOPS=4)
  job_id="${output%%;*}"
  job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse a Slurm job id from: $output" >&2
    exit 1
  }
  echo "submitted held-out multikey confirmation"
  echo "CONFIRM_JOB_ID=$job_id"
  echo "config=k16_v4_h4 tasks=$CONFIRM_TASKS prompts=420..439 contract=difficulty_v1"
  echo "next: bash $REPORT_DIR/steps.sh --confirm-status $job_id"
}

status_difficulty() {
  shift
  local job_id
  for job_id in "$@"; do
    need_job_id "$job_id"
    echo "===== job $job_id ====="
    status_oracle "$job_id"
  done
}

read_grid_jobs() {
  local label=$1 run_stem=$2
  shift 2
  (($#)) || { echo "ERROR: at least one $label JOB_ID is required" >&2; exit 2; }
  local job_id f out ids=""
  local -a parquets=()
  for job_id in "$@"; do
    need_job_id "$job_id"
    ids="${ids}${ids:+_}${job_id}"
    f=$(find h0_measurement/results -maxdepth 2 -type f \
      -path "*${run_stem}_*_${job_id}/r8_llama31-8b_32768*.parquet" \
      -print -quit)
    [[ -n "$f" ]] || {
      echo "ERROR: no completed $label parquet for job $job_id" >&2
      exit 1
    }
    parquets+=("$f")
  done
  out="$REPORT_DIR/${label}_${ids}.txt"
  env OMP_NUM_THREADS=8 "$PY" "$READER" "${parquets[@]}" > "$out"
  cat "$out"
  echo
  echo "saved: $out"
}

read_difficulty() {
  shift
  read_grid_jobs difficulty r8diff "$@"
}

read_confirmation() {
  shift
  read_grid_jobs confirmation r8confirm "$@"
}

policy_spec() {
  case "${1:-dev}" in
    dev)     printf '%s\n' "20 440 r8policy_dev_ 00:20:00 80 60" ;;
    confirm) printf '%s\n' "40 460 r8policy_confirm_ 01:30:00 160 120" ;;
    *) echo "ERROR: policy split must be dev or confirm" >&2; exit 2 ;;
  esac
}

policy_command() {
  local split=$1 n_prompts prompt_offset prefix walltime expected_accuracy expected_policy
  read -r n_prompts prompt_offset prefix walltime expected_accuracy expected_policy \
    < <(policy_spec "$split")
  printf '%q ' sbatch --parsable --time="$walltime" "$WORKER" \
    R8_RUN_PREFIX="$prefix" R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS="$POLICY_ARMS" R8_BUDGETS=2 R8_TASKS=niah_multikey \
    R8_N_PROMPTS="$n_prompts" R8_PROMPT_OFFSET="$prompt_offset" R8_QA=1 \
    R8_N_KEYS=16 R8_N_VALUES=4 R8_N_HOPS=4 R8_HEAD_ERROR=0 \
    R8_POLICY_DIAGNOSTIC=1 R8_POLICY_CANDIDATES="$POLICY_CANDIDATES" \
    R8_POLICY_TRACE_STEPS=8
  printf '\n'
}

policy_artifact_complete() {
  local dir=$1 n_prompts=$2 prompt_offset=$3 expected_accuracy=$4 expected_policy=$5
  local accuracy="$dir/r8_llama31-8b_32768_k16_v4_h4.parquet"
  local diagnostic="$dir/r8policy_llama31-8b_32768_k16_v4_h4.parquet"
  [[ -f "$accuracy" && -f "${accuracy%.parquet}.json" \
     && -f "$diagnostic" && -f "${diagnostic%.parquet}.json" ]] || return 1

  # read_policy authenticates the separate sidecars, the linked accuracy SHA,
  # candidate provenance, shared trace, exact one-to-one join, and raw-logit ban.
  "$PY" "$POLICY_READER" "$accuracy" "$diagnostic" \
    --bootstrap 100 >/dev/null 2>&1 || return 1

  "$PY" - "$accuracy" "$diagnostic" "$n_prompts" "$prompt_offset" \
    "$expected_accuracy" "$expected_policy" <<'PY_AUDIT'
from collections import Counter
import json
import os
import sys

import pandas as pd

accuracy_path, diagnostic_path = sys.argv[1:3]
n_prompts, prompt_offset, expected_accuracy, expected_policy = map(int, sys.argv[3:])
want_prompts = set(range(prompt_offset, prompt_offset + n_prompts))
want_arms = {"fp", "uniform", "evict", "interior"}
want_candidates = ["uniform", "evict", "interior"]
want_cfg = {"n_keys": 16, "n_values": 4, "n_hops": 4}

for path in (accuracy_path, diagnostic_path):
    with open(path, "rb") as handle:
        handle.seek(-4, os.SEEK_END)
        assert handle.read() == b"PAR1", f"incomplete parquet footer: {path}"

accuracy = pd.read_parquet(accuracy_path)
diagnostic = pd.read_parquet(diagnostic_path)
assert len(accuracy) == expected_accuracy
assert len(diagnostic) == expected_policy
assert set(accuracy.prompt_idx) == want_prompts
assert set(diagnostic.prompt_idx) == want_prompts
assert set(accuracy.task) == {"niah_multikey"}
assert set(diagnostic.task) == {"niah_multikey"}
assert set(accuracy.arm) == want_arms
assert set(diagnostic.candidate) == set(want_candidates)
assert set(zip(diagnostic.candidate, diagnostic.candidate_order)) == \
       {(name, i) for i, name in enumerate(want_candidates)}

accuracy_keys = Counter(zip(accuracy.prompt_idx, accuracy.task, accuracy.arm))
assert set(accuracy_keys) == {(p, "niah_multikey", arm) for p in want_prompts for arm in want_arms}
assert set(accuracy_keys.values()) == {1}
diagnostic_keys = Counter(zip(diagnostic.prompt_idx, diagnostic.task, diagnostic.B,
                              diagnostic.candidate))
assert set(diagnostic_keys) == {(p, "niah_multikey", 2.0, candidate)
                               for p in want_prompts for candidate in want_candidates}
assert set(diagnostic_keys.values()) == {1}
assert all((arm == "fp" and float(B) == 0.0) or
           (arm != "fp" and float(B) == 2.0)
           for arm, B in zip(accuracy.arm, accuracy.B))
assert all(bool(x) for x in accuracy.question_agnostic)
assert all(bool(x) for x in diagnostic.question_agnostic)
assert not any(bool(x) for x in accuracy.synthetic)
assert not any(bool(x) for x in diagnostic.synthetic)
assert len(set(accuracy.corpus_sha)) == 1 and accuracy.corpus_sha.iloc[0] not in ("", "synthetic")
assert len(set(diagnostic.corpus_sha)) == 1 and diagnostic.corpus_sha.iloc[0] == accuracy.corpus_sha.iloc[0]
for field, value in want_cfg.items():
    assert set(accuracy[field]) == {value}
    assert set(diagnostic[field]) == {value}
assert "selected_policy" in diagnostic and "fp_argmax_verified" in diagnostic
assert diagnostic.fp_argmax_verified.astype(bool).all()
assert set(diagnostic.selected_policy).issubset(set(want_candidates))
assert "allocator_budget_rule" in diagnostic
assert set(diagnostic.allocator_budget_rule) == {"feasible"}
assert not any("logit" in column.lower() for column in diagnostic.columns)
for _, group in diagnostic.groupby(["prompt_idx", "task", "B"], sort=False):
    group = group.sort_values("candidate_order")
    selected = group.loc[group.mean_kl.idxmin(), "candidate"]
    assert set(group.selected_policy) == {selected}

accuracy_side = json.load(open(accuracy_path[:-8] + ".json"))
diagnostic_side = json.load(open(diagnostic_path[:-8] + ".json"))
assert accuracy_side["arms"] == ["fp", "uniform", "evict", "interior"]
assert [float(x) for x in accuracy_side["budgets"]] == [2.0]
assert accuracy_side["n_prompts"] == n_prompts
assert accuracy_side["prompt_offset"] == prompt_offset
assert accuracy_side["task_config"] == want_cfg
assert accuracy_side["generation_limit_version"] == "difficulty_v1"
assert diagnostic_side["expected_rows"] == expected_policy
assert diagnostic_side["candidates"] == want_candidates
assert diagnostic_side["trace_steps_requested"] == 8
assert diagnostic_side["no_raw_logits"] is True
PY_AUDIT
}

policy_find_complete() {
  local split=$1 n_prompts prompt_offset prefix walltime expected_accuracy expected_policy dir
  read -r n_prompts prompt_offset prefix walltime expected_accuracy expected_policy \
    < <(policy_spec "$split")
  for dir in h0_measurement/results/${prefix}*; do
    [[ -d "$dir" ]] || continue
    if policy_artifact_complete "$dir" "$n_prompts" "$prompt_offset" \
        "$expected_accuracy" "$expected_policy"; then
      printf '%s\n' "$dir"
      return 0
    fi
  done
  return 1
}

policy_dir_for_job() {
  local split=$1 job_id=$2 n_prompts prompt_offset prefix walltime expected_accuracy expected_policy
  read -r n_prompts prompt_offset prefix walltime expected_accuracy expected_policy \
    < <(policy_spec "$split")
  local dir="h0_measurement/results/${prefix}${job_id}"
  [[ -d "$dir" ]] || {
    echo "ERROR: no $split policy result directory for job $job_id: $dir" >&2
    return 1
  }
  printf '%s\n' "$dir"
}

policy_dev_advances() {
  local dir accuracy diagnostic
  dir=$(policy_find_complete dev) || return 1
  accuracy="$dir/r8_llama31-8b_32768_k16_v4_h4.parquet"
  diagnostic="$dir/r8policy_llama31-8b_32768_k16_v4_h4.parquet"
  "$PY" - "$accuracy" "$diagnostic" "$POLICY_READER" <<'PY_GATE'
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("read_policy_gate", sys.argv[3])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
summary, _ = mod.load_pair(sys.argv[1], sys.argv[2], n_boot=100, seed=0)
raise SystemExit(0 if len(summary) == 1 and summary.iloc[0].decision == "advance_mean_kl" else 1)
PY_GATE
}

submit_policy() {
  local split=$1 n_prompts prompt_offset prefix walltime expected_accuracy expected_policy
  local where output job_id
  read -r n_prompts prompt_offset prefix walltime expected_accuracy expected_policy \
    < <(policy_spec "$split")
  if where=$(policy_find_complete "$split"); then
    echo "$split policy diagnostic already complete: $where"
    echo "read it using the numeric suffix in: bash $REPORT_DIR/steps.sh --policy-read $split JOB"
    return 0
  fi
  if [[ "$split" == confirm ]] && ! policy_dev_advances; then
    echo "ERROR: locked confirmation requires a complete development artifact with decision=advance_mean_kl" >&2
    exit 1
  fi
  output=$(sbatch --parsable --time="$walltime" "$WORKER" \
    R8_RUN_PREFIX="$prefix" R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS="$POLICY_ARMS" R8_BUDGETS=2 R8_TASKS=niah_multikey \
    R8_N_PROMPTS="$n_prompts" R8_PROMPT_OFFSET="$prompt_offset" R8_QA=1 \
    R8_N_KEYS=16 R8_N_VALUES=4 R8_N_HOPS=4 R8_HEAD_ERROR=0 \
    R8_POLICY_DIAGNOSTIC=1 R8_POLICY_CANDIDATES="$POLICY_CANDIDATES" \
    R8_POLICY_TRACE_STEPS=8)
  job_id="${output%%;*}"; job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse a Slurm job id from: $output" >&2; exit 1; }
  echo "submitted $split whole-policy diagnostic"
  echo "POLICY_JOB_ID=$job_id"
  echo "config=k16_v4_h4 task=niah_multikey B=2 prompts=$prompt_offset..$((prompt_offset+n_prompts-1))"
  echo "rows=accuracy:$expected_accuracy diagnostic:$expected_policy trace=fp_teacher_forced_v1<=8"
  echo "log:    h0_measurement/logs/r8_${job_id}.out"
  echo "result: h0_measurement/results/${prefix}${job_id}/"
  echo "next:   bash $REPORT_DIR/steps.sh --policy-status $job_id"
}

read_policy_job() {
  local split=$1 job_id=$2 n_prompts prompt_offset prefix walltime expected_accuracy expected_policy
  local dir accuracy diagnostic out summary_csv prompt_csv
  read -r n_prompts prompt_offset prefix walltime expected_accuracy expected_policy \
    < <(policy_spec "$split")
  dir=$(policy_dir_for_job "$split" "$job_id")
  accuracy="$dir/r8_llama31-8b_32768_k16_v4_h4.parquet"
  diagnostic="$dir/r8policy_llama31-8b_32768_k16_v4_h4.parquet"
  if ! policy_artifact_complete "$dir" "$n_prompts" "$prompt_offset" \
      "$expected_accuracy" "$expected_policy"; then
    echo "ERROR: job $job_id does not satisfy the complete $split policy-artifact contract" >&2
    exit 1
  fi
  out="$REPORT_DIR/policy_${job_id}.txt"
  summary_csv="$REPORT_DIR/policy_${job_id}_summary.csv"
  prompt_csv="$REPORT_DIR/policy_${job_id}_prompts.csv"
  env OMP_NUM_THREADS=8 "$PY" "$POLICY_READER" "$accuracy" "$diagnostic" \
    --csv "$summary_csv" --prompt-csv "$prompt_csv" > "$out"
  cat "$out"
  echo
  echo "saved: $out"
  echo "saved: $summary_csv"
  echo "saved: $prompt_csv"
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
  --difficulty-dry)
    need_file "$WORKER"; need_file "$READER"
    echo "DRY RUN; submit nothing:"
    while read -r nk nv nh tasks contract; do difficulty_command "$nk" "$nv" "$nh" "$tasks"; done \
      < <(difficulty_specs "${2:-screen}")
    ;;
  --difficulty-submit)
    need_file "$WORKER"; need_file "$READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
    while read -r nk nv nh tasks contract; do
      submit_difficulty_one "$nk" "$nv" "$nh" "$tasks" "$contract"
    done < <(difficulty_specs "${2:-screen}")
    ;;
  --difficulty-status)
    (($# >= 2)) || { echo "ERROR: at least one JOB_ID is required" >&2; exit 2; }
    status_difficulty "$@"
    ;;
  --difficulty-read)
    need_file "$READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    read_difficulty "$@"
    ;;
  --confirm-dry)
    need_file "$WORKER"; need_file "$READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    echo "DRY RUN; submit nothing:"
    confirmation_command
    ;;
  --confirm-submit)
    need_file "$WORKER"; need_file "$READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
    submit_confirmation
    ;;
  --confirm-status)
    (($# >= 2)) || { echo "ERROR: at least one JOB_ID is required" >&2; exit 2; }
    status_difficulty "$@"
    ;;
  --confirm-read)
    need_file "$READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    read_confirmation "$@"
    ;;
  --policy-dry)
    need_file "$WORKER"; need_file "$POLICY_READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    echo "DRY RUN; submit nothing:"
    policy_command "${2:-dev}"
    ;;
  --policy-submit)
    need_file "$WORKER"; need_file "$POLICY_READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
    submit_policy "${2:-dev}"
    ;;
  --policy-status)
    (($# >= 2)) || { echo "ERROR: at least one JOB_ID is required" >&2; exit 2; }
    status_difficulty "$@"
    ;;
  --policy-read)
    need_file "$POLICY_READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    need_job_id "${3:-}"
    read_policy_job "${2:-dev}" "$3"
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
