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
#
# V2-A: OPERATING-POINT SCREEN (RUN BEFORE THE NEXT POLICY DIAGNOSTIC)
# -------------------------------------------------------------------
# The confirmed k16 point leaves only 20% accuracy headroom below FP. Screen
# two harder, prespecified multikey points on fresh prompts before spending on a
# second policy diagnostic. Both jobs use Llama 32K, B=2, QA mode, fp+uniform,
# v4/h4, and prompts 500..539:
#
#   k24: n_keys=24 (80 accuracy rows)
#   k32: n_keys=32 (80 accuracy rows)
#
# Run the screen in this order:
#
#   1. Inspect both exact commands; submit nothing:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-dry screen
#      Use `k24` or `k32` instead of `screen` to inspect one cell.
#
#   2. Submit both cells and save the two printed OP2_JOB_ID values:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-submit screen
#      The script authenticates and skips an already complete cell independently.
#
#   3. Check all newly submitted IDs without blocking:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-status JOB [JOB...]
#
#   4. After both cells complete, pass the k24 ID first and k32 ID second:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-read K24_JOB K32_JOB
#
# `--op2-read` accepts only this fixed pair. It authenticates each exact job
# directory, then read_op2.py applies the frozen operating-point selection rule.
# Do not launch policy V2-B until that report selects an operating point.
#
# V2-A2: K40 FOLLOW-UP (ONLY AFTER V2-A REPORTS BOTH CELLS TOO EASY)
# -------------------------------------------------------------------
# Jobs 982121/982122 completed the fixed k24/k32 screen, but neither cell met
# every preregistered gate: k24 was perfect and k32 had uniform=0.825, above
# the [.50,.75] primary band. The frozen branch in plan.md therefore calls
# for exactly one fresh k40 cell. It uses prompts 540..579; do not reuse these
# prompts for policy development after inspecting this screen.
#
# Run this branch in order:
#
#   1. Inspect the exact command; submit nothing:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-k40-dry
#
#   2. Submit the k40 cell to the short debug queue and save OP2_K40_JOB_ID:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-k40-submit
#
#   3. Check it without blocking:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-k40-status JOB
#
#   4. Only after Slurm reports COMPLETED, authenticate and analyze it:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-k40-read JOB
#
# The dependency is the completed and recorded paired V2-A decision, produced
# by `--op2-read 982121 982122`. The k40 reader accepts only the exact k40,
# prompts 540..579, fp+uniform, B=2 artifact and applies its frozen gates.
#
# V2-A3: FINAL K48 BRACKET (ONLY AFTER K40 IS DIRECTIONALLY TOO EASY)
# ------------------------------------------------------------------------
# Job 982613 completed the exact k40 screen with FP=1.000 and uniform=0.925;
# both halves are above the stability upper bound (0.950/0.900). The bounded
# plan permits one final fresh k48 cell on prompts 580..619. Run in order:
#
#   1. Reauthenticate k40 and inspect the exact command; submit nothing:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-k48-dry
#   2. Submit and save OP2_K48_JOB_ID:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-k48-submit
#   3. Check without blocking:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-k48-status JOB
#   4. After COMPLETED, authenticate and apply all original eligibility gates:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-k48-read JOB
#
# The guard recomputes and byte-compares the canonical job-982613 summary. It
# requires valid FP/provenance and both k40 halves >.85; no hand-entered outcome
# can open this branch. K48 is the last key-count screen. If it fails any gate,
# stop; do not interpolate, pool adaptive cells, or alter a threshold.
#
# RESULT (job 982702, completed 2026-09-23)
# ---------------------------------------------------------
# FP=1.000, uniform=.850, halves=.900/.800, FP-uniform=.150 with paired
# 90% CI [.075,.250], and no incomplete capped FP. The [.50,.75] mean gate and
# first-half upper bound fail, so there is no selection. Do not resubmit or run
# V2-B. Reproduce the authenticated report with --op2-k48-read 982702.
#
# V3: CONTRASTIVE MULTIKEY-PANEL QUALIFICATION
# ---------------------------------------------------------
# The terminal V2 result motivates a new task contract rather than another
# key-count bracket. Qualification uses one shared 48-needle context and four
# independently decoded questions per prompt. It is fixed to prompts 700..739,
# fp+uniform, B=2, and 320 accuracy rows. Run in order:
#
#   1. Inspect the exact command; submit nothing:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --panel-dry
#   2. Submit one qualification job and save PANEL_JOB_ID:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --panel-submit
#   3. Check without blocking:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --panel-status JOB
#   4. After COMPLETED, authenticate and apply every frozen gate:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --panel-read JOB
#
# Only `decision: advance_policy_development` opens prompts 740..779. A valid
# `stop_panel` result exits zero and ends V3 without changing this task.
#
# RESULT (job 983199, completed 2026-09-23)
# ---------------------------------------------------------
# FP=.981, uniform=.850, halves=.863/.838, slots=.950/.850/.800/.800,
# FP-uniform=.131 with paired 90% CI [.088,.181]. The uniform mean and first
# half fail the frozen gates, so the reader returns stop_panel. Do not resubmit,
# alter the panel score/configuration, or open prompts 740..819. Reproduce with:
#   bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --panel-read 983199
#
# V4: PINNED LONGBENCH-V2 NATURAL-TASK QUALIFICATION
# --------------------------------------------------
# V3 is closed. V4 first runs the CPU-only audit and then exactly one frozen
# qualification job. Do not run development unless the strict qualification
# reader prints `decision        advance_development`.
#
# Run these commands in order:
#
#   1. Rebuild/authenticate the gold-free manifest (idempotent; no GPU):
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-audit
#   2. Print the exact Slurm command; submit nothing:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-qual-dry
#   3. Submit qualification and save LBV2_JOB_ID:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-qual-submit
#   4. Check without blocking:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-qual-status JOB
#   5. After COMPLETED, authenticate, apply the frozen gates, and save the report:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-qual-read JOB
#
# Qualification is the immutable 20-item split, fp+uniform at B=2, greedy
# EOS-or-128 decode, and the canonical A/B/C/D choice-logit diagnostic. Expected
# artifacts are 40 accuracy rows and 20 choice rows. A valid `stop_v4` result is
# a scientific stop, not a pipeline error, and must not be rerun or tuned.
#
# RESULT (job 983715): stop_v4. The two capped invalid FP answers bind. Do not
# rerun V4 or invoke its unused development mode.
#
# V5: FORCED-CHOICE OPPORTUNITY + BRANCH-BLIND SCAFFOLD PROXY
# ----------------------------------------------------------------
# V5 is a separately frozen protocol. It performed no free generation and
# opened the then-untouched 52-row development split once. The commands below
# reproduce or audit the completed workflow in order:
#
#   1. Print the exact Slurm command; submit nothing:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v5-dev-dry
#   2. Submit the one development job and save LBV2_V5_JOB_ID:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v5-dev-submit
#   3. Check without blocking:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v5-dev-status JOB
#   4. After COMPLETED, authenticate and apply the competence, H, then G gates:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v5-dev-read JOB
#
# The exact job is FP plus eight B=2 complete policies at Llama 32K. It writes
# 468 label-free forced-choice prediction rows and 416 branch-blind proxy rows.
# The primary proxy uses only output positions 0..4 that predict the fixed
# response scaffold; position 5 predicts A/B/C/D and cannot enter selection.
# Only decision `advance_confirmation` creates a lock and permits a later
# confirmation interface. Every other valid decision closes V5 development.
#
# RESULT (job 983888, completed 2026-09-23)
# ---------------------------------------------------------
# Provenance passed. FP and every compressed arm score 19/52; the candidate
# oracle scores 22/52, so H=3/52=.058 with q05=.017 and fixed halves
# .071/.042. The point and second-half gates fail: `stop_no_opportunity`.
# G is suppressed, no lock exists, and V5 confirmation is closed. Do not
# resubmit or inspect confirmation. Reproduce the binding read with:
#   bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v5-dev-read 983888
#
# V6: QWEN3-30B FORCED-CHOICE QUALIFICATION
# -----------------------------------------
# V5 is closed. V6 changes the model once while retaining B=2. It begins with
# only FP and uniform on the 20 disclosed qualification items; no development
# command exists until this gate passes. Run these commands in order:
#
#   1. Build/authenticate the gold-free Qwen manifest (CPU only):
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-audit
#   2. Print the exact Slurm command; submit nothing:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-qual-dry
#   3. Submit the one qualification job and save LBV2_V6_JOB_ID:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-qual-submit
#   4. Check without blocking:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-qual-status JOB
#   5. After COMPLETED, authenticate, apply the frozen gates, and write a lock
#      only when the decision is `advance_v6_development`:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-qual-read JOB
#
# The exact job is Qwen3-30B-A3B-Instruct-2507 revision 0d7cf239 at ctx 40960,
# W=32 and B=2. It writes 40 label-free forced-choice rows and no proxy artifact.
#
# QUALIFICATION RESULT (job 984224, completed 2026-09-23)
# --------------------------------------------------------
# FP=12/20=.600 (component-bootstrap q05=.421) and uniform=12/20=.600
# (q05=.429). Every frozen gate and provenance check passes, yielding
# `advance_v6_development` and the authenticated lock fixed below.
#
# V6 CONDITIONAL DEVELOPMENT (AUTHORIZED BY JOB 984224 ONLY)
# ----------------------------------------------------------
# Run this sequence from the project root or any directory:
#
#   1. Reauthenticate the lock and print the exact command; submit nothing:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-dev-dry
#   2. Submit the one 52-row development job and save LBV2_V6_DEV_JOB_ID:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-dev-submit
#   3. Check without blocking:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-dev-status JOB
#   4. After COMPLETED, authenticate both artifacts and apply competence, H,
#      then G in that order:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-dev-read JOB
#
# The job is FP plus the frozen eight B=2 candidates on 52 development items in
# 44 components. It writes 468 prediction rows and 416 branch-blind proxy rows.
# G is suppressed unless competence and every H gate pass. Only
# `advance_v6_confirmation` writes a confirmation lock; all other valid decisions
# close this V6 policy branch without touching the 45 confirmation items.
#
# V7: FRESH 128K STRUCTURED-QUERY MECHANISM QUALIFICATION (TERMINAL STOP)
# -----------------------------------------------------------------------
# V6 is terminal. V7 proposed changing one upstream mechanism on a source
# partition disjoint from all V4--V6 exact-context/question components. Its
# qualification ran only FP and exact all-2 uniform; structured policies were
# inaccessible unless this gate wrote an authenticated advance lock. Historical
# reproduction/read sequence:
#
#   1. Reproduce the label-free 20/52/45 manifest and semantic token spans:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v7-audit
#   2. Print the exact Slurm command; submit nothing:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v7-qual-dry
#   3. Submit once and save LBV2_V7_JOB_ID:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v7-qual-submit
#   4. Check without blocking:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v7-qual-status JOB
#   5. After COMPLETED, authenticate and apply the frozen FP/uniform gates:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v7-qual-read JOB
#
# This was Qwen3-30B-A3B-Instruct-2507 at ctx=131072, W=32, B=2,
# 20 singleton components, 40 label-free rows, and no proxy. Job 984886
# completed with exit 0:0 and passed provenance. The frozen reader found
# FP=9/20=0.450 (q05=0.250) and uniform=7/20=0.350 (q05=0.200), yielding
# stop_v7_qualification. No advance lock exists. V7 is closed: do not resubmit,
# create a development ledger, or run development/confirmation on this split.
#
# V2-B: EXPANDED-POLICY DEVELOPMENT (ONLY AFTER V2-A SELECTS K)
# ----------------------------------------------------------------
# Every V2-B dry, submit, and read command takes the selected key count and the
# exact paired V2-A job IDs. The workflow re-authenticates that pair and requires
# its frozen selection to equal SELECTED_K before it prints, submits, or reads.
#
# Run in this order (K24_JOB must precede K32_JOB):
#
#   1. Authenticate and record the paired V2-A selection:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-read K24_JOB K32_JOB
#
#   2. If that report says `selection: k=K`, inspect the exact V2-B command:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --v2b-dry K24_JOB K32_JOB K
#
#   3. Submit the one selected-k job and save V2B_JOB_ID:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --v2b-submit K24_JOB K32_JOB K
#
#   4. Check it without blocking:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --v2b-status V2B_JOB_ID
#
#   5. After completion, authenticate, analyze nested prefixes, and write a lock
#      only if a preregistered selector advances:
#        bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --v2b-read K24_JOB K32_JOB K V2B_JOB_ID
#
# V2-B uses prompts 540..579, fp plus eight ordered complete policies, and an
# at-most-eight-token FP teacher-forced trace. Expected counts are 360 accuracy
# rows and 320 diagnostic rows. Its 01:00 walltime allows margin over the roughly
# 30-minute estimate from the measured four-arm development run.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant}"
cd "$PROJECT_ROOT"

PY="${SIEVE_VENV:-$PROJECT_ROOT/.venv}/bin/python"
WORKER="h0_measurement/submit_r8.slurm"
READER="h0_measurement/bugs/8_router_endtask/read_r8.py"
POLICY_READER="h0_measurement/bugs/9_sota_eviction_baselines/read_policy.py"
OP2_READER="h0_measurement/bugs/9_sota_eviction_baselines/read_op2.py"
OP2_K40_READER="h0_measurement/bugs/9_sota_eviction_baselines/read_op2_k40.py"
OP2_K48_READER="h0_measurement/bugs/9_sota_eviction_baselines/read_op2_k48.py"
PANEL_READER="h0_measurement/bugs/9_sota_eviction_baselines/read_panel_qual.py"
V2B_READER="h0_measurement/bugs/9_sota_eviction_baselines/read_policy_v2.py"
LBV2_WORKER="h0_measurement/submit_longbench_v2.slurm"
LBV2_READER="h0_measurement/bugs/9_sota_eviction_baselines/read_longbench_v2.py"
LBV2_V5_WORKER="h0_measurement/submit_longbench_v2_forced_choice.slurm"
LBV2_V5_READER="h0_measurement/bugs/9_sota_eviction_baselines/read_longbench_v2_forced_choice.py"
LBV2_V6_WORKER="h0_measurement/submit_longbench_v2_qwen_qualification.slurm"
LBV2_V6_READER="h0_measurement/bugs/9_sota_eviction_baselines/read_longbench_v2_qwen_qualification.py"
LBV2_V6_AUDIT="h0_measurement/audit_longbench_v2_qwen.py"
LBV2_V6_DEV_RUNNER="h0_measurement/run_longbench_v2_qwen_development.py"
LBV2_V6_DEV_WORKER="h0_measurement/submit_longbench_v2_qwen_development.slurm"
LBV2_V6_DEV_READER="h0_measurement/bugs/9_sota_eviction_baselines/read_longbench_v2_qwen_development.py"
LBV2_V6_QUAL_LOCK="h0_measurement/bugs/9_sota_eviction_baselines/longbench_v2_v6_qualification_lock_984224.json"
LBV2_AUDIT="h0_measurement/audit_longbench_v2.py"
LBV2_DATA=".h0_corpus/longbench_v2/data-2b48e494.json"
LBV2_MANIFEST="h0_measurement/bugs/9_sota_eviction_baselines/longbench_v2_manifest.json"
LBV2_V6_MANIFEST="h0_measurement/bugs/9_sota_eviction_baselines/longbench_v2_qwen30_manifest.json"
LBV2_V7_RUNNER="h0_measurement/run_longbench_v2_qwen_v7_qualification.py"
LBV2_V7_WORKER="h0_measurement/submit_longbench_v2_qwen_v7_qualification.slurm"
LBV2_V7_READER="h0_measurement/bugs/9_sota_eviction_baselines/read_longbench_v2_qwen_v7_qualification.py"
LBV2_V7_AUDIT="h0_measurement/audit_longbench_v2_qwen_v7.py"
LBV2_V7_MANIFEST="h0_measurement/bugs/9_sota_eviction_baselines/longbench_v2_qwen30_v7_manifest.json"
LBV2_V7_LEDGER="h0_measurement/bugs/9_sota_eviction_baselines/longbench_v2_qwen_v7_source_ledger.json"
ROUTES="h0_measurement/results/r8_routes/llama31-8b_32768_qa.json"
REPORT_DIR="h0_measurement/bugs/9_sota_eviction_baselines"

ARMS="fp,uniform,evict,interior,interior_cascade,router_calib,router_oracle"
TASKS="niah_single,niah_multikey,niah_multivalue,vt"
DIFFICULTY_TASKS="niah_multikey,niah_multivalue,vt"
CONFIRM_TASKS="niah_multikey"
POLICY_ARMS="fp,uniform,evict,interior"
POLICY_CANDIDATES="uniform,evict,interior"
V2B_ARMS="fp,uniform,evict,interior,interior_pool,interior_cascade,obcache_k,obcache_k:alloc=ada@obck_ada,laprox"
V2B_CANDIDATES="uniform,evict,interior,interior_pool,interior_cascade,obcache_k,obck_ada,laprox"

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

  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-dry [k24|k32|screen]
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-submit [k24|k32|screen]
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-status JOB_ID [JOB_ID...]
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-read K24_JOB_ID K32_JOB_ID

  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-k40-dry
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-k40-submit
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-k40-status JOB_ID
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-k40-read JOB_ID

  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-k48-dry
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-k48-submit
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-k48-status JOB_ID
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --op2-k48-read JOB_ID

  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --panel-dry
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --panel-submit
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --panel-status JOB_ID
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --panel-read JOB_ID

  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --v2b-dry K24_JOB_ID K32_JOB_ID SELECTED_K
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --v2b-submit K24_JOB_ID K32_JOB_ID SELECTED_K
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --v2b-status JOB_ID [JOB_ID...]
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --v2b-read K24_JOB_ID K32_JOB_ID SELECTED_K V2B_JOB_ID

  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-audit
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-qual-dry
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-qual-submit
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-qual-status JOB_ID
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-qual-read JOB_ID

  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v5-dev-dry
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v5-dev-submit
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v5-dev-status JOB_ID
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v5-dev-read JOB_ID

  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-audit
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-qual-dry
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-qual-submit
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-qual-status JOB_ID
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-qual-read JOB_ID

  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v7-audit
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v7-qual-dry
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v7-qual-submit
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v7-qual-status JOB_ID
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v7-qual-read JOB_ID

  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-dev-dry
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-dev-submit
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-dev-status JOB_ID
  bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --lbv2-v6-dev-read JOB_ID

The default selector "screen" addresses all three tasks at (8,6,6) and
(16,8,8). The post-screen "mid" selector reruns multivalue and VT at (16,7,7)
with difficulty-aware answer limits; "vt-hard" does the same for VT at h8.
The confirmation interface is fixed to the selected k16/v4/h4 multikey cell on
held-out prompts 420--439 and cannot accept a difficulty selector.
The policy development split is prompts 440--459. The policy confirmation split
is prompts 460--499 and must be run only after the development reader prints
`decision       advance_mean_kl`.
The V2-A operating-point screen is prompts 500--539 at k24 and k32. Its `screen`
selector means both cells. The read interface requires the k24 job first and
the k32 job second so one cell can never be silently mistaken for the other.
The V2-A2 k40 branch is prompts 540--579 and is eligible only after that paired
screen reports no selection because both cells are too easy. Its exact k40
reader is separate so the original two-cell report cannot accept this new split.
The final k48 bracket is prompts 580--619 and is reachable only from exact k40
job 982613 when FP/provenance pass and the mean plus both halves are too easy.
The V3 panel qualification is prompts 700--739 at k48/v4/h4, with four
questions over each shared context/allocation and exactly 320 rows. Its reader
uses first_ok, aggregates within 40 contexts, and is the sole gate for panel
policy development.
V2-B is prompts 540--579 at the explicitly selected V2-A key count. Its dry,
submit, and read modes re-authenticate the exact pair and refuse a mismatched key
count. The raw adaptive OBCache arm resolves to the stable label `obck_ada`.
V4 uses a separate LongBench-v2 runner, worker, and reader. Its qualification
split is fixed by the gold-free manifest and ended with `stop_v4`.
V5 is separately versioned and used forced choice plus a branch-blind scaffold
proxy on its development split. Job 983888 ended `stop_no_opportunity`; no
confirmation lock exists, its confirmation split is closed, and this script
intentionally exposes no V5 confirmation submission.
V6 uses separate Qwen qualification and development runners, readers, workers,
and result namespaces. Job 984224 passed qualification; the development modes
below require and reauthenticate its exact advancing lock before use.
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


# V2-A has two fixed operating-point cells. `screen` expands to both in the
# declared order so dry-run, submission, and the paired reader agree on k.
op2_specs() {
  case "${1:-screen}" in
    k24)    printf '%s\n' "24 r8op2_k24_" ;;
    k32)    printf '%s\n' "32 r8op2_k32_" ;;
    screen) printf '%s\n' "24 r8op2_k24_" "32 r8op2_k32_" ;;
    *) echo "ERROR: operating-point selector must be k24, k32, or screen" >&2; exit 2 ;;
  esac
}

op2_command() {
  local nk=$1 prefix=$2
  printf '%q ' sbatch --parsable --time=00:12:00 "$WORKER" \
    R8_RUN_PREFIX="$prefix" R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS=fp,uniform R8_BUDGETS=2 R8_TASKS=niah_multikey \
    R8_N_PROMPTS=40 R8_PROMPT_OFFSET=500 R8_QA=1 \
    R8_N_KEYS="$nk" R8_N_VALUES=4 R8_N_HOPS=4 R8_HEAD_ERROR=0
  printf '\n'
}

# Validate one exact job directory. The standalone audit makes the completion
# test independent of similarly named runs; read_op2.py then authenticates the
# parquet/sidecar pair with the same reader used for the scientific report.
op2_artifact_complete() {
  local dir=$1 nk=$2
  local parquet="$dir/r8_llama31-8b_32768_k${nk}_v4_h4.parquet"
  local sidecar="${parquet%.parquet}.json"
  [[ -f "$parquet" && -f "$sidecar" ]] || return 1

  "$PY" - "$parquet" "$sidecar" "$nk" <<'PY_OP2_AUDIT' || return 1
from collections import Counter
import json
import os
import sys

import pandas as pd

parquet, sidecar, raw_nk = sys.argv[1:]
nk = int(raw_nk)
with open(parquet, "rb") as handle:
    handle.seek(-4, os.SEEK_END)
    assert handle.read() == b"PAR1", f"incomplete parquet footer: {parquet}"

columns = [
    "model", "ctx", "prompt_idx", "task", "arm", "B", "n_keys",
    "n_values", "n_hops", "corpus_sha", "synthetic", "bits_per_token",
    "question_agnostic", "max_new_tokens", "gen_len", "reached_max_new",
]
frame = pd.read_parquet(parquet, columns=columns)
want_prompts = set(range(500, 540))
want_arms = {"fp", "uniform"}
assert len(frame) == 80
assert set(frame.model) == {"llama31-8b"}
assert set(frame.ctx.astype(int)) == {32768}
assert set(frame.prompt_idx.astype(int)) == want_prompts
assert set(frame.task) == {"niah_multikey"}
assert set(frame.arm) == want_arms
assert set(frame.n_keys.astype(int)) == {nk}
assert set(frame.n_values.astype(int)) == {4}
assert set(frame.n_hops.astype(int)) == {4}
assert frame.question_agnostic.astype(bool).all()
assert not frame.synthetic.astype(bool).any()
assert len(set(frame.corpus_sha)) == 1
assert frame.corpus_sha.iloc[0] not in ("", "synthetic")
keys = Counter(zip(frame.prompt_idx.astype(int), frame.task, frame.arm))
assert set(keys) == {(prompt, "niah_multikey", arm)
                     for prompt in want_prompts for arm in want_arms}
assert set(keys.values()) == {1}
assert all((arm == "fp" and float(budget) == 0.0) or
           (arm == "uniform" and float(budget) == 2.0)
           for arm, budget in zip(frame.arm, frame.B))
assert all(value is None or float(value) <= 2.0 + 1e-6
           for value, arm in zip(frame.bits_per_token, frame.arm)
           if arm == "uniform")
assert set(frame.max_new_tokens.astype(int)) == {24}
assert all(bool(reached) == (int(length) >= int(limit))
           for reached, length, limit in zip(
               frame.reached_max_new, frame.gen_len, frame.max_new_tokens))

with open(sidecar) as handle:
    meta = json.load(handle)
assert meta["parquet"] == os.path.basename(parquet)
assert meta["model"] == "llama31-8b"
assert int(meta["ctx"]) == 32768
assert meta["tasks"] == ["niah_multikey"]
assert meta["arms"] == ["fp", "uniform"]
assert [float(value) for value in meta["budgets"]] == [2.0]
assert meta["task_config"] == {"n_keys": nk, "n_values": 4, "n_hops": 4}
assert int(meta["n_prompts"]) == 40
assert int(meta["prompt_offset"]) == 500
assert bool(meta["question_agnostic"])
assert int(meta["rows"]) == 80
assert meta["generation_limit_version"] == "difficulty_v1"
assert meta["generation_limits"] == {"niah_multikey": 24}
assert meta["corpus_sha"] == frame.corpus_sha.iloc[0]
PY_OP2_AUDIT

  "$PY" "$OP2_READER" --validate-only "$parquet" >/dev/null 2>&1 || return 1
}

op2_find_complete() {
  local nk=$1 prefix=$2 dir base
  for dir in "h0_measurement/results/${prefix}"*; do
    [[ -d "$dir" ]] || continue
    base=${dir##*/}
    [[ "$base" =~ ^${prefix}[0-9]+$ ]] || continue
    if op2_artifact_complete "$dir" "$nk"; then
      printf '%s\n' "$dir"
      return 0
    fi
  done
  return 1
}

op2_job_id_from_dir() {
  local dir=$1 prefix=$2 base=${1##*/}
  [[ "$base" =~ ^${prefix}([0-9]+)$ ]] || return 1
  printf '%s\n' "${BASH_REMATCH[1]}"
}

submit_op2_one() {
  local nk=$1 prefix=$2 label="k${1}" where output job_id
  if where=$(op2_find_complete "$nk" "$prefix"); then
    job_id=$(op2_job_id_from_dir "$where" "$prefix") || {
      echo "ERROR: valid $label artifact has an unexpected directory name: $where" >&2
      exit 1
    }
    echo "operating-point $label already complete: $where"
    echo "OP2_${label^^}_JOB_ID=$job_id (existing)"
    echo "result: $where/"
    return 0
  fi
  output=$(sbatch --parsable --time=00:12:00 "$WORKER" \
    R8_RUN_PREFIX="$prefix" R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS=fp,uniform R8_BUDGETS=2 R8_TASKS=niah_multikey \
    R8_N_PROMPTS=40 R8_PROMPT_OFFSET=500 R8_QA=1 \
    R8_N_KEYS="$nk" R8_N_VALUES=4 R8_N_HOPS=4 R8_HEAD_ERROR=0)
  job_id="${output%%;*}"; job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse a Slurm job id from: $output" >&2; exit 1; }
  echo "submitted operating-point $label screen"
  echo "OP2_${label^^}_JOB_ID=$job_id"
  echo "config=${label}_v4_h4 task=niah_multikey B=2 prompts=500..539 rows=80"
  echo "log:    h0_measurement/logs/r8_${job_id}.out"
  echo "result: h0_measurement/results/${prefix}${job_id}/"
}

op2_dir_for_job() {
  local nk=$1 prefix=$2 job_id=$3
  local dir="h0_measurement/results/${prefix}${job_id}"
  [[ -d "$dir" ]] || {
    echo "ERROR: no exact k$nk operating-point directory for job $job_id: $dir" >&2
    return 1
  }
  printf '%s\n' "$dir"
}

read_op2_pair() {
  local k24_job=$1 k32_job=$2 k24_dir k32_dir k24_parquet k32_parquet out summary_csv
  k24_dir=$(op2_dir_for_job 24 r8op2_k24_ "$k24_job")
  k32_dir=$(op2_dir_for_job 32 r8op2_k32_ "$k32_job")
  if ! op2_artifact_complete "$k24_dir" 24; then
    echo "ERROR: job $k24_job does not satisfy the complete k24 operating-point contract" >&2
    exit 1
  fi
  if ! op2_artifact_complete "$k32_dir" 32; then
    echo "ERROR: job $k32_job does not satisfy the complete k32 operating-point contract" >&2
    exit 1
  fi
  k24_parquet="$k24_dir/r8_llama31-8b_32768_k24_v4_h4.parquet"
  k32_parquet="$k32_dir/r8_llama31-8b_32768_k32_v4_h4.parquet"
  out="$REPORT_DIR/op2_${k24_job}_${k32_job}.txt"
  summary_csv="$REPORT_DIR/op2_${k24_job}_${k32_job}_summary.csv"
  env OMP_NUM_THREADS=8 "$PY" "$OP2_READER" "$k24_parquet" "$k32_parquet" \
    --csv "$summary_csv" > "$out"
  cat "$out"
  echo
  echo "saved: $out"
  echo "saved: $summary_csv"
}


# V2-A2 is an adaptive branch of the exact completed V2-A pair. Reauthenticate
# that pair and its recorded result before dry-run, submission, or reporting;
# comments alone are not a scientific dependency gate.
op2_k40_require_v2a_branch() {
  local k24_job=982121 k32_job=982122 k24_dir k32_dir k24_parquet k32_parquet
  local recorded_csv gate_csv gate_status
  k24_dir=$(op2_dir_for_job 24 r8op2_k24_ "$k24_job") || return 1
  k32_dir=$(op2_dir_for_job 32 r8op2_k32_ "$k32_job") || return 1
  if ! op2_artifact_complete "$k24_dir" 24; then
    echo "ERROR: k40 prerequisite rejected unauthenticated V2-A k24 job $k24_job" >&2
    return 1
  fi
  if ! op2_artifact_complete "$k32_dir" 32; then
    echo "ERROR: k40 prerequisite rejected unauthenticated V2-A k32 job $k32_job" >&2
    return 1
  fi
  k24_parquet="$k24_dir/r8_llama31-8b_32768_k24_v4_h4.parquet"
  k32_parquet="$k32_dir/r8_llama31-8b_32768_k32_v4_h4.parquet"
  recorded_csv="$REPORT_DIR/op2_${k24_job}_${k32_job}_summary.csv"
  [[ -f "$recorded_csv" ]] || {
    echo "ERROR: missing canonical paired V2-A summary: $recorded_csv" >&2
    echo "run: bash $REPORT_DIR/steps.sh --op2-read $k24_job $k32_job" >&2
    return 1
  }
  gate_csv=$(mktemp "${TMPDIR:-/tmp}/sieve-op2-k40-gate.XXXXXX.csv") || {
    echo "ERROR: could not create a temporary k40 prerequisite summary" >&2
    return 1
  }
  if "$PY" - "$OP2_READER" "$k24_parquet" "$k32_parquet" \
      "$recorded_csv" "$gate_csv" <<'PY_K40_GATE'
import importlib.util
import sys
from pathlib import Path

import pandas as pd

reader_path, k24_path, k32_path, recorded_path, gate_path = sys.argv[1:]
spec = importlib.util.spec_from_file_location("read_op2_k40_gate", reader_path)
reader = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(reader)
summary, selected = reader.analyze_artifacts([k24_path, k32_path])
summary.to_csv(gate_path, index=False)
if Path(recorded_path).read_bytes() != Path(gate_path).read_bytes():
    raise SystemExit(
        "ERROR: canonical paired V2-A summary differs from a fresh authenticated recomputation"
    )
if selected is not None or summary.selected.astype(bool).any():
    raise SystemExit(
        f"ERROR: k40 branch requires no V2-A selection; recomputation selected {selected!r}"
    )
if len(summary) != 2 or set(summary.n_keys.astype(int)) != {24, 32}:
    raise SystemExit("ERROR: k40 branch requires the exact k24/k32 V2-A pair")
if summary.k24_fp_stop.astype(bool).any():
    raise SystemExit("ERROR: k40 branch is forbidden because the V2-A k24 FP-stop fired")
if not (pd.to_numeric(summary.uniform_mean, errors="raise") > 0.75).all():
    means = summary.sort_values("n_keys")[["n_keys", "uniform_mean"]].to_dict("records")
    raise SystemExit(
        "ERROR: k40 branch requires both V2-A cells to be too easy "
        f"(uniform_mean > .75); got {means!r}"
    )
PY_K40_GATE
  then
    gate_status=0
  else
    gate_status=$?
  fi
  rm -f "$gate_csv"
  return "$gate_status"
}

# Keep this path separate from op2_artifact_complete/read_op2.py, whose contract
# intentionally remains the paired k24/k32 screen on prompts 500..539.
op2_k40_command() {
  printf '%q ' sbatch --parsable --partition=debug --time=00:12:00 "$WORKER" \
    R8_RUN_PREFIX=r8op2_k40_ R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS=fp,uniform R8_BUDGETS=2 R8_TASKS=niah_multikey \
    R8_N_PROMPTS=40 R8_PROMPT_OFFSET=540 R8_QA=1 \
    R8_N_KEYS=40 R8_N_VALUES=4 R8_N_HOPS=4 R8_HEAD_ERROR=0
  printf '\n'
}

op2_k40_artifact_complete() {
  local dir=$1
  local parquet="$dir/r8_llama31-8b_32768_k40_v4_h4.parquet"
  local sidecar="${parquet%.parquet}.json"
  [[ -f "$parquet" && -f "$sidecar" ]] || return 1
  "$PY" "$OP2_K40_READER" "$parquet" --validate-only >/dev/null 2>&1
}

op2_k40_find_complete() {
  local prefix=r8op2_k40_ dir base
  for dir in "h0_measurement/results/${prefix}"*; do
    [[ -d "$dir" ]] || continue
    base=${dir##*/}
    [[ "$base" =~ ^${prefix}[0-9]+$ ]] || continue
    if op2_k40_artifact_complete "$dir"; then
      printf '%s\n' "$dir"
      return 0
    fi
  done
  return 1
}

submit_op2_k40() {
  local prefix=r8op2_k40_ where output job_id
  if where=$(op2_k40_find_complete); then
    job_id=$(op2_job_id_from_dir "$where" "$prefix") || {
      echo "ERROR: valid k40 artifact has an unexpected directory name: $where" >&2
      exit 1
    }
    echo "operating-point k40 follow-up already complete: $where"
    echo "OP2_K40_JOB_ID=$job_id (existing)"
    echo "result: $where/"
    return 0
  fi
  output=$(sbatch --parsable --partition=debug --time=00:12:00 "$WORKER" \
    R8_RUN_PREFIX=r8op2_k40_ R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS=fp,uniform R8_BUDGETS=2 R8_TASKS=niah_multikey \
    R8_N_PROMPTS=40 R8_PROMPT_OFFSET=540 R8_QA=1 \
    R8_N_KEYS=40 R8_N_VALUES=4 R8_N_HOPS=4 R8_HEAD_ERROR=0)
  job_id="${output%%;*}"; job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse a Slurm job id from: $output" >&2
    exit 1
  }
  echo "submitted operating-point k40 follow-up"
  echo "OP2_K40_JOB_ID=$job_id"
  echo "config=k40_v4_h4 task=niah_multikey B=2 prompts=540..579 rows=80"
  echo "partition=debug estimated_runtime=about_5m walltime=00:12:00"
  echo "log:    h0_measurement/logs/r8_${job_id}.out"
  echo "result: h0_measurement/results/r8op2_k40_${job_id}/"
  echo "next:   bash $REPORT_DIR/steps.sh --op2-k40-status $job_id"
}

op2_k40_dir_for_job() {
  local job_id=$1 dir="h0_measurement/results/r8op2_k40_${1}"
  [[ -d "$dir" ]] || {
    echo "ERROR: no exact k40 follow-up directory for job $job_id: $dir" >&2
    return 1
  }
  printf '%s\n' "$dir"
}

read_op2_k40_job() {
  local job_id=$1 dir parquet out summary_csv
  dir=$(op2_k40_dir_for_job "$job_id")
  if ! op2_k40_artifact_complete "$dir"; then
    echo "ERROR: job $job_id does not satisfy the complete k40 follow-up contract" >&2
    exit 1
  fi
  parquet="$dir/r8_llama31-8b_32768_k40_v4_h4.parquet"
  out="$REPORT_DIR/op2_k40_${job_id}.txt"
  summary_csv="$REPORT_DIR/op2_k40_${job_id}_summary.csv"
  env OMP_NUM_THREADS=8 "$PY" "$OP2_K40_READER" "$parquet" \
    --csv "$summary_csv" > "$out"
  cat "$out"
  echo
  echo "saved: $out"
  echo "saved: $summary_csv"
}


# V2-A3 is the single final bracket permitted after the authenticated k40 cell
# is directionally too easy in both halves. Recompute that exact outcome before
# dry-run, submission, or reporting so the branch cannot be entered by hand.
op2_k48_require_k40_branch() {
  local k40_job=982613 k40_dir k40_parquet recorded_csv gate_csv gate_status
  k40_dir=$(op2_k40_dir_for_job "$k40_job") || return 1
  if ! op2_k40_artifact_complete "$k40_dir"; then
    echo "ERROR: k48 prerequisite rejected unauthenticated k40 job $k40_job" >&2
    return 1
  fi
  k40_parquet="$k40_dir/r8_llama31-8b_32768_k40_v4_h4.parquet"
  recorded_csv="$REPORT_DIR/op2_k40_${k40_job}_summary.csv"
  [[ -f "$recorded_csv" ]] || {
    echo "ERROR: missing canonical k40 summary: $recorded_csv" >&2
    echo "run: bash $REPORT_DIR/steps.sh --op2-k40-read $k40_job" >&2
    return 1
  }
  gate_csv=$(mktemp "${TMPDIR:-/tmp}/sieve-op2-k48-gate.XXXXXX.csv") || {
    echo "ERROR: could not create a temporary k48 prerequisite summary" >&2
    return 1
  }
  if "$PY" - "$OP2_K40_READER" "$k40_parquet" \
      "$recorded_csv" "$gate_csv" <<'PY_K48_GATE'
import importlib.util
import sys
from pathlib import Path

reader_path, k40_path, recorded_path, gate_path = sys.argv[1:]
spec = importlib.util.spec_from_file_location("read_op2_k48_gate", reader_path)
reader = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(reader)
summary, eligible = reader.analyze_artifact(k40_path)
summary.to_csv(gate_path, index=False)
if Path(recorded_path).read_bytes() != Path(gate_path).read_bytes():
    raise SystemExit(
        "ERROR: canonical k40 summary differs from a fresh authenticated recomputation"
    )
if len(summary) != 1 or int(summary.iloc[0].n_keys) != 40:
    raise SystemExit("ERROR: k48 branch requires the exact k40 prerequisite cell")
row = summary.iloc[0]
if eligible or bool(row.selected):
    raise SystemExit("ERROR: k48 branch is forbidden because k40 was eligible")
if not bool(row.gate_fp_ge_095) or not bool(row.gate_no_incomplete_capped_fp):
    raise SystemExit("ERROR: k48 branch is forbidden because k40 failed FP/cap validity")
if not float(row.uniform_mean) > 0.75:
    raise SystemExit(
        f"ERROR: k48 branch requires k40 uniform_mean > .75; got {row.uniform_mean!r}"
    )
first = float(row.uniform_540_559)
second = float(row.uniform_560_579)
if not (first > 0.85 and second > 0.85):
    raise SystemExit(
        "ERROR: k48 branch requires both k40 halves to be directionally too easy "
        f"(> .85); got {first!r}, {second!r}"
    )
PY_K48_GATE
  then
    gate_status=0
  else
    gate_status=$?
  fi
  rm -f "$gate_csv"
  return "$gate_status"
}

op2_k48_command() {
  printf '%q ' sbatch --parsable --partition=debug --time=00:12:00 "$WORKER" \
    R8_RUN_PREFIX=r8op2_k48_ R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS=fp,uniform R8_BUDGETS=2 R8_TASKS=niah_multikey \
    R8_N_PROMPTS=40 R8_PROMPT_OFFSET=580 R8_QA=1 \
    R8_N_KEYS=48 R8_N_VALUES=4 R8_N_HOPS=4 R8_HEAD_ERROR=0
  printf '\n'
}

op2_k48_artifact_complete() {
  local dir=$1
  local parquet="$dir/r8_llama31-8b_32768_k48_v4_h4.parquet"
  local sidecar="${parquet%.parquet}.json"
  [[ -f "$parquet" && -f "$sidecar" ]] || return 1
  "$PY" "$OP2_K48_READER" "$parquet" --validate-only >/dev/null 2>&1
}

op2_k48_find_complete() {
  local prefix=r8op2_k48_ dir base
  for dir in "h0_measurement/results/${prefix}"*; do
    [[ -d "$dir" ]] || continue
    base=${dir##*/}
    [[ "$base" =~ ^${prefix}[0-9]+$ ]] || continue
    if op2_k48_artifact_complete "$dir"; then
      printf '%s\n' "$dir"
      return 0
    fi
  done
  return 1
}

submit_op2_k48() {
  local prefix=r8op2_k48_ where output job_id
  if where=$(op2_k48_find_complete); then
    job_id=$(op2_job_id_from_dir "$where" "$prefix") || {
      echo "ERROR: valid k48 artifact has an unexpected directory name: $where" >&2
      exit 1
    }
    echo "operating-point k48 final bracket already complete: $where"
    echo "OP2_K48_JOB_ID=$job_id (existing)"
    echo "result: $where/"
    return 0
  fi
  output=$(sbatch --parsable --partition=debug --time=00:12:00 "$WORKER" \
    R8_RUN_PREFIX=r8op2_k48_ R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS=fp,uniform R8_BUDGETS=2 R8_TASKS=niah_multikey \
    R8_N_PROMPTS=40 R8_PROMPT_OFFSET=580 R8_QA=1 \
    R8_N_KEYS=48 R8_N_VALUES=4 R8_N_HOPS=4 R8_HEAD_ERROR=0)
  job_id="${output%%;*}"; job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse a Slurm job id from: $output" >&2
    exit 1
  }
  echo "submitted operating-point k48 final bracket"
  echo "OP2_K48_JOB_ID=$job_id"
  echo "config=k48_v4_h4 task=niah_multikey B=2 prompts=580..619 rows=80"
  echo "partition=debug estimated_runtime=about_5m walltime=00:12:00"
  echo "log:    h0_measurement/logs/r8_${job_id}.out"
  echo "result: h0_measurement/results/r8op2_k48_${job_id}/"
  echo "next:   bash $REPORT_DIR/steps.sh --op2-k48-status $job_id"
}

op2_k48_dir_for_job() {
  local job_id=$1 dir="h0_measurement/results/r8op2_k48_${1}"
  [[ -d "$dir" ]] || {
    echo "ERROR: no exact k48 final-bracket directory for job $job_id: $dir" >&2
    return 1
  }
  printf '%s\n' "$dir"
}

read_op2_k48_job() {
  local job_id=$1 dir parquet out summary_csv
  dir=$(op2_k48_dir_for_job "$job_id")
  if ! op2_k48_artifact_complete "$dir"; then
    echo "ERROR: job $job_id does not satisfy the complete k48 final-bracket contract" >&2
    exit 1
  fi
  parquet="$dir/r8_llama31-8b_32768_k48_v4_h4.parquet"
  out="$REPORT_DIR/op2_k48_${job_id}.txt"
  summary_csv="$REPORT_DIR/op2_k48_${job_id}_summary.csv"
  env OMP_NUM_THREADS=8 "$PY" "$OP2_K48_READER" "$parquet" \
    --csv "$summary_csv" > "$out"
  cat "$out"
  echo
  echo "saved: $out"
  echo "saved: $summary_csv"
}



# V3 qualification is isolated from every legacy R8 task and V2 key-count cell.
# The strict reader authenticates the exact panel variant and 4-query structure.
panel_command() {
  printf '%q ' sbatch --parsable --partition=debug --time=00:20:00 "$WORKER" \
    R8_RUN_PREFIX=r8panel_qual_ R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS=fp,uniform R8_BUDGETS=2 R8_TASKS=niah_multikey \
    R8_TASK_VARIANT=multikey_panel_v1 R8_N_PROMPTS=40 R8_PROMPT_OFFSET=700 \
    R8_QA=1 R8_N_KEYS=48 R8_N_VALUES=4 R8_N_HOPS=4 R8_HEAD_ERROR=0
  printf '\n'
}

panel_artifact_complete() {
  local dir=$1
  local parquet="$dir/r8_llama31-8b_32768_k48_v4_h4_multikey_panel_v1.parquet"
  local sidecar="${parquet%.parquet}.json"
  [[ -f "$parquet" && -f "$sidecar" ]] || return 1
  "$PY" "$PANEL_READER" "$parquet" --validate-only >/dev/null 2>&1
}

panel_find_complete() {
  local prefix=r8panel_qual_ dir base
  for dir in "h0_measurement/results/${prefix}"*; do
    [[ -d "$dir" ]] || continue
    base=${dir##*/}
    [[ "$base" =~ ^${prefix}[0-9]+$ ]] || continue
    if panel_artifact_complete "$dir"; then
      printf '%s\n' "$dir"
      return 0
    fi
  done
  return 1
}

panel_dir_for_job() {
  local job_id=$1 dir="h0_measurement/results/r8panel_qual_${1}"
  [[ -d "$dir" ]] || {
    echo "ERROR: no exact panel qualification directory for job $job_id: $dir" >&2
    return 1
  }
  printf '%s\n' "$dir"
}

submit_panel() {
  local prefix=r8panel_qual_ where output job_id
  if where=$(panel_find_complete); then
    job_id=$(op2_job_id_from_dir "$where" "$prefix") || {
      echo "ERROR: valid panel artifact has an unexpected directory name: $where" >&2
      exit 1
    }
    echo "panel qualification already complete: $where"
    echo "PANEL_JOB_ID=$job_id (existing)"
    echo "result: $where/"
    return 0
  fi
  output=$(sbatch --parsable --partition=debug --time=00:20:00 "$WORKER" \
    R8_RUN_PREFIX=r8panel_qual_ R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS=fp,uniform R8_BUDGETS=2 R8_TASKS=niah_multikey \
    R8_TASK_VARIANT=multikey_panel_v1 R8_N_PROMPTS=40 R8_PROMPT_OFFSET=700 \
    R8_QA=1 R8_N_KEYS=48 R8_N_VALUES=4 R8_N_HOPS=4 R8_HEAD_ERROR=0)
  job_id="${output%%;*}"; job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse a Slurm job id from: $output" >&2
    exit 1
  }
  echo "submitted V3 contrastive multikey-panel qualification"
  echo "PANEL_JOB_ID=$job_id"
  echo "config=k48_v4_h4 variant=multikey_panel_v1 B=2 prompts=700..739 rows=320"
  echo "partition=debug conservative_walltime=00:20:00"
  echo "log:    h0_measurement/logs/r8_${job_id}.out"
  echo "result: h0_measurement/results/r8panel_qual_${job_id}/"
  echo "next:   bash $REPORT_DIR/steps.sh --panel-status $job_id"
}

read_panel_job() {
  local job_id=$1 dir parquet out summary_csv
  dir=$(panel_dir_for_job "$job_id")
  if ! panel_artifact_complete "$dir"; then
    echo "ERROR: job $job_id does not satisfy the complete panel qualification contract" >&2
    exit 1
  fi
  parquet="$dir/r8_llama31-8b_32768_k48_v4_h4_multikey_panel_v1.parquet"
  out="$REPORT_DIR/panel_qual_${job_id}.txt"
  summary_csv="$REPORT_DIR/panel_qual_${job_id}_summary.csv"
  env OMP_NUM_THREADS=8 "$PY" "$PANEL_READER" "$parquet" \
    --csv "$summary_csv" > "$out"
  cat "$out"
  echo
  echo "saved: $out"
  echo "saved: $summary_csv"
}

need_v2b_n_keys() {
  [[ "${1:-}" == 24 || "${1:-}" == 32 ]] || {
    echo "ERROR: SELECTED_K must be exactly 24 or 32, got '${1:-}'" >&2
    exit 2
  }
}

v2b_paths() {
  local nk=$1 dir=$2
  printf '%s\n' \
    "$dir/r8_llama31-8b_32768_k${nk}_v4_h4.parquet" \
    "$dir/r8policy_llama31-8b_32768_k${nk}_v4_h4.parquet"
}

# Recompute V2-A's frozen paired decision from the exact job directories. This
# never trusts a hand-written key count or an unauthenticated/stale CSV.
v2b_require_op2_selection() {
  local want_k=$1 k24_job=$2 k32_job=$3 k24_dir k32_dir k24_parquet k32_parquet
  local recorded_csv gate_csv gate_status
  k24_dir=$(op2_dir_for_job 24 r8op2_k24_ "$k24_job") || return 1
  k32_dir=$(op2_dir_for_job 32 r8op2_k32_ "$k32_job") || return 1
  if ! op2_artifact_complete "$k24_dir" 24; then
    echo "ERROR: V2-B gate rejected unauthenticated k24 job $k24_job" >&2
    return 1
  fi
  if ! op2_artifact_complete "$k32_dir" 32; then
    echo "ERROR: V2-B gate rejected unauthenticated k32 job $k32_job" >&2
    return 1
  fi
  k24_parquet="$k24_dir/r8_llama31-8b_32768_k24_v4_h4.parquet"
  k32_parquet="$k32_dir/r8_llama31-8b_32768_k32_v4_h4.parquet"
  recorded_csv="$REPORT_DIR/op2_${k24_job}_${k32_job}_summary.csv"
  [[ -f "$recorded_csv" ]] || {
    echo "ERROR: missing exact paired V2-A summary: $recorded_csv" >&2
    echo "run: bash $REPORT_DIR/steps.sh --op2-read $k24_job $k32_job" >&2
    return 1
  }
  gate_csv=$(mktemp "${TMPDIR:-/tmp}/sieve-op2-gate.XXXXXX.csv") || {
    echo "ERROR: could not create a temporary V2-A gate summary" >&2
    return 1
  }
  if ! "$PY" "$OP2_READER" "$k24_parquet" "$k32_parquet" \
      --csv "$gate_csv" >/dev/null; then
    rm -f "$gate_csv"
    echo "ERROR: V2-B gate could not authenticate the paired V2-A summary" >&2
    return 1
  fi
  if ! cmp -s "$recorded_csv" "$gate_csv"; then
    rm -f "$gate_csv"
    echo "ERROR: $recorded_csv does not match the freshly authenticated exact V2-A pair" >&2
    echo "rerun: bash $REPORT_DIR/steps.sh --op2-read $k24_job $k32_job" >&2
    return 1
  fi
  if "$PY" - "$recorded_csv" "$want_k" <<'PY_V2B_GATE'
import sys
import pandas as pd

path, raw_want = sys.argv[1:]
want = int(raw_want)
frame = pd.read_csv(path)
required = {"n_keys", "selected"}
if not required.issubset(frame.columns):
    raise SystemExit("ERROR: authenticated V2-A summary lacks n_keys/selected")
selected_mask = frame.selected.astype(str).str.lower().isin({"true", "1"})
selected = frame.loc[selected_mask, "n_keys"].astype(int).tolist()
if selected != [want]:
    actual = "no selection" if not selected else ",".join(map(str, selected))
    raise SystemExit(
        f"ERROR: V2-A selected {actual}; refusing V2-B requested SELECTED_K={want}"
    )
PY_V2B_GATE
  then
    gate_status=0
  else
    gate_status=$?
  fi
  rm -f "$gate_csv"
  return "$gate_status"
}

v2b_command() {
  local nk=$1 prefix="r8policy_v2b_k${1}_"
  printf '%q ' sbatch --parsable --time=01:00:00 "$WORKER" \
    R8_RUN_PREFIX="$prefix" R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS="$V2B_ARMS" R8_BUDGETS=2 R8_TASKS=niah_multikey \
    R8_N_PROMPTS=40 R8_PROMPT_OFFSET=540 R8_QA=1 \
    R8_N_KEYS="$nk" R8_N_VALUES=4 R8_N_HOPS=4 R8_HEAD_ERROR=0 \
    R8_POLICY_DIAGNOSTIC=1 R8_POLICY_CANDIDATES="$V2B_CANDIDATES" \
    R8_POLICY_TRACE_STEPS=8
  printf '\n'
}

v2b_artifact_complete() {
  local dir=$1 nk=$2 paths accuracy diagnostic
  mapfile -t paths < <(v2b_paths "$nk" "$dir")
  accuracy=${paths[0]}; diagnostic=${paths[1]}
  [[ -f "$accuracy" && -f "${accuracy%.parquet}.json" \
     && -f "$diagnostic" && -f "${diagnostic%.parquet}.json" ]] || return 1
  "$PY" "$V2B_READER" "$accuracy" "$diagnostic" \
    --n-keys "$nk" --validate-only >/dev/null 2>&1 || return 1
}

v2b_find_complete() {
  local nk=$1 prefix="r8policy_v2b_k${1}_" dir base
  for dir in "h0_measurement/results/${prefix}"*; do
    [[ -d "$dir" ]] || continue
    base=${dir##*/}
    [[ "$base" =~ ^${prefix}[0-9]+$ ]] || continue
    if v2b_artifact_complete "$dir" "$nk"; then
      printf '%s\n' "$dir"
      return 0
    fi
  done
  return 1
}

submit_v2b() {
  local nk=$1 prefix="r8policy_v2b_k${1}_" where output job_id
  if where=$(v2b_find_complete "$nk"); then
    job_id=$(op2_job_id_from_dir "$where" "$prefix") || {
      echo "ERROR: valid V2-B artifact has an unexpected directory name: $where" >&2
      exit 1
    }
    echo "V2-B k$nk already complete: $where"
    echo "V2B_JOB_ID=$job_id (existing)"
    echo "result: $where/"
    return 0
  fi
  output=$(sbatch --parsable --time=01:00:00 "$WORKER" \
    R8_RUN_PREFIX="$prefix" R8_MODEL=llama31-8b R8_CTX=32768 \
    R8_ARMS="$V2B_ARMS" R8_BUDGETS=2 R8_TASKS=niah_multikey \
    R8_N_PROMPTS=40 R8_PROMPT_OFFSET=540 R8_QA=1 \
    R8_N_KEYS="$nk" R8_N_VALUES=4 R8_N_HOPS=4 R8_HEAD_ERROR=0 \
    R8_POLICY_DIAGNOSTIC=1 R8_POLICY_CANDIDATES="$V2B_CANDIDATES" \
    R8_POLICY_TRACE_STEPS=8)
  job_id="${output%%;*}"; job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse a Slurm job id from: $output" >&2; exit 1; }
  echo "submitted expanded-policy V2-B development"
  echo "V2B_JOB_ID=$job_id"
  echo "config=k${nk}_v4_h4 task=niah_multikey B=2 prompts=540..579"
  echo "rows=accuracy:360 diagnostic:320 trace=fp_teacher_forced_v1<=8"
  echo "estimated_runtime=about_30m walltime=01:00:00"
  echo "log:    h0_measurement/logs/r8_${job_id}.out"
  echo "result: h0_measurement/results/${prefix}${job_id}/"
  echo "next:   bash $REPORT_DIR/steps.sh --v2b-status $job_id"
}

v2b_dir_for_job() {
  local nk=$1 job_id=$2
  local dir="h0_measurement/results/r8policy_v2b_k${nk}_${job_id}"
  [[ -d "$dir" ]] || {
    echo "ERROR: no exact V2-B k$nk directory for job $job_id: $dir" >&2
    return 1
  }
  printf '%s\n' "$dir"
}

read_v2b_job() {
  local nk=$1 job_id=$2 dir paths accuracy diagnostic out summary_csv prompt_csv lock_out
  dir=$(v2b_dir_for_job "$nk" "$job_id")
  if ! v2b_artifact_complete "$dir" "$nk"; then
    echo "ERROR: job $job_id does not satisfy the complete V2-B k$nk artifact contract" >&2
    exit 1
  fi
  mapfile -t paths < <(v2b_paths "$nk" "$dir")
  accuracy=${paths[0]}; diagnostic=${paths[1]}
  out="$REPORT_DIR/policy_v2_${job_id}.txt"
  summary_csv="$REPORT_DIR/policy_v2_${job_id}_summary.csv"
  prompt_csv="$REPORT_DIR/policy_v2_${job_id}_prompts.csv"
  lock_out="$REPORT_DIR/policy_v2_${job_id}_lock.json"
  env OMP_NUM_THREADS=8 "$PY" "$V2B_READER" "$accuracy" "$diagnostic" \
    --n-keys "$nk" --csv "$summary_csv" --prompt-csv "$prompt_csv" \
    --lock-out "$lock_out" > "$out"
  cat "$out"
  echo
  echo "saved: $out"
  echo "saved: $summary_csv"
  echo "saved: $prompt_csv"
  if [[ -f "$lock_out" ]]; then
    echo "saved: $lock_out"
  else
    echo "lock:  none (the authenticated V2-B decision did not advance a selector)"
  fi
}

# V4 qualification is isolated from RULER and every prior synthetic campaign.
# Its strict reader authenticates both cross-linked parquets before a directory
# can count as complete.
lbv2_snapshot() {
  printf '%s\n' "$PROJECT_ROOT/.hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
}

lbv2_audit() {
  local snapshot
  snapshot=$(lbv2_snapshot)
  need_file "$LBV2_DATA"
  need_file "$LBV2_MANIFEST"
  need_file "$LBV2_AUDIT"
  [[ -d "$snapshot" ]] || {
    echo "ERROR: missing pinned Llama snapshot $snapshot" >&2
    return 1
  }
  "$PY" "$LBV2_AUDIT" \
    --data "$LBV2_DATA" --model-id "$snapshot" --out "$LBV2_MANIFEST"
}

lbv2_command() {
  printf '%q ' sbatch --parsable --partition=debug --time=02:00:00 \
    "$LBV2_WORKER" LBV2_PHASE=qualification \
    LBV2_RUN_PREFIX=lbv2_qualification_
  printf '\n'
}

lbv2_artifact_complete() {
  local dir=$1
  local accuracy="$dir/longbench_v2_qualification_llama31-8b_32768.parquet"
  local choice="$dir/longbench_v2_choice_qualification_llama31-8b_32768.parquet"
  [[ -f "$accuracy" && -f "${accuracy%.parquet}.json" \
     && -f "$choice" && -f "${choice%.parquet}.json" ]] || return 1
  "$PY" "$LBV2_READER" "$accuracy" "$choice" \
    --manifest "$LBV2_MANIFEST" --dataset "$LBV2_DATA" \
    --validate-only >/dev/null 2>&1
}

lbv2_find_complete() {
  local prefix=lbv2_qualification_ dir base
  for dir in h0_measurement/results/${prefix}*; do
    [[ -d "$dir" ]] || continue
    base=${dir##*/}
    [[ "$base" =~ ^${prefix}[0-9]+$ ]] || continue
    if lbv2_artifact_complete "$dir"; then
      printf '%s\n' "$dir"
      return 0
    fi
  done
  return 1
}

lbv2_dir_for_job() {
  local job_id=$1 dir="h0_measurement/results/lbv2_qualification_${1}"
  [[ -d "$dir" ]] || {
    echo "ERROR: no exact V4 qualification directory for job $job_id: $dir" >&2
    return 1
  }
  printf '%s\n' "$dir"
}

submit_lbv2_qualification() {
  local prefix=lbv2_qualification_ where output job_id
  if where=$(lbv2_find_complete); then
    job_id=${where##*/}; job_id=${job_id#${prefix}}
    echo "V4 LongBench-v2 qualification already complete: $where"
    echo "LBV2_JOB_ID=$job_id (existing)"
    echo "read: bash $REPORT_DIR/steps.sh --lbv2-qual-read $job_id"
    return 0
  fi
  output=$(sbatch --parsable --partition=debug --time=02:00:00 \
    "$LBV2_WORKER" LBV2_PHASE=qualification \
    LBV2_RUN_PREFIX=lbv2_qualification_)
  job_id="${output%%;*}"; job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse a Slurm job id from: $output" >&2
    exit 1
  }
  echo "submitted V4 LongBench-v2 qualification"
  echo "LBV2_JOB_ID=$job_id"
  echo "split=qualification items=20 accuracy_rows=40 choice_rows=20"
  echo "model=llama31-8b ctx=32768 B=2 arms=fp,uniform max_new=128"
  echo "partition=debug walltime=02:00:00"
  echo "log:    h0_measurement/logs/lbv2_${job_id}.out"
  echo "result: h0_measurement/results/lbv2_qualification_${job_id}/"
  echo "next:   bash $REPORT_DIR/steps.sh --lbv2-qual-status $job_id"
}

status_lbv2_qualification() {
  local job_id=$1 log="h0_measurement/logs/lbv2_${1}.out"
  if command -v squeue >/dev/null 2>&1; then
    squeue -j "$job_id" || true
  fi
  if command -v sacct >/dev/null 2>&1; then
    sacct -j "$job_id" --format=JobID,State,Elapsed,ExitCode || true
  fi
  if [[ -f "$log" ]]; then
    echo
    echo "last 30 log lines:"
    tail -n 30 "$log"
  else
    echo "log not created yet: $log"
  fi
}

read_lbv2_qualification() {
  local job_id=$1 dir accuracy choice out summary_csv
  dir=$(lbv2_dir_for_job "$job_id")
  if ! lbv2_artifact_complete "$dir"; then
    echo "ERROR: job $job_id does not satisfy the complete V4 qualification contract" >&2
    exit 1
  fi
  accuracy="$dir/longbench_v2_qualification_llama31-8b_32768.parquet"
  choice="$dir/longbench_v2_choice_qualification_llama31-8b_32768.parquet"
  out="$REPORT_DIR/longbench_v2_qualification_${job_id}.txt"
  summary_csv="$REPORT_DIR/longbench_v2_qualification_${job_id}_summary.csv"
  env OMP_NUM_THREADS=8 "$PY" "$LBV2_READER" "$accuracy" "$choice" \
    --manifest "$LBV2_MANIFEST" --dataset "$LBV2_DATA" \
    --csv "$summary_csv" > "$out"
  cat "$out"
  echo
  echo "saved: $out"
  echo "saved: $summary_csv"
}

# V5 development is a new forced-choice protocol. It cannot reuse V4 output
# names, and its complete check runs the label-joining reader only after both
# reciprocal artifacts and sidecars exist.
lbv2_v5_command() {
  printf '%q ' sbatch --parsable --partition=debug --time=02:00:00 \
    "$LBV2_V5_WORKER" LBV2_V5_PHASE=development \
    LBV2_V5_RUN_PREFIX=lbv2_v5_development_
  printf '\n'
}

lbv2_v5_artifact_complete() {
  local dir=$1
  local predictions="$dir/longbench_v2_v5_forced_choice_development.parquet"
  local proxy="$dir/longbench_v2_v5_scaffold_proxy_development.parquet"
  [[ -f "$predictions" && -f "${predictions%.parquet}.json" \
     && -f "$proxy" && -f "${proxy%.parquet}.json" ]] || return 1
  "$PY" "$LBV2_V5_READER" "$predictions" "$proxy" \
    --manifest "$LBV2_MANIFEST" --dataset "$LBV2_DATA" \
    --validate-only >/dev/null 2>&1
}

lbv2_v5_find_complete() {
  local prefix=lbv2_v5_development_ dir base
  for dir in h0_measurement/results/${prefix}*; do
    [[ -d "$dir" ]] || continue
    base=${dir##*/}
    [[ "$base" =~ ^${prefix}[0-9]+$ ]] || continue
    if lbv2_v5_artifact_complete "$dir"; then
      printf '%s\n' "$dir"
      return 0
    fi
  done
  return 1
}

lbv2_v5_dir_for_job() {
  local job_id=$1 dir="h0_measurement/results/lbv2_v5_development_${1}"
  [[ -d "$dir" ]] || {
    echo "ERROR: no exact V5 development directory for job $job_id: $dir" >&2
    return 1
  }
  printf '%s\n' "$dir"
}

submit_lbv2_v5_development() {
  local prefix=lbv2_v5_development_ where output job_id
  if where=$(lbv2_v5_find_complete); then
    job_id=${where##*/}; job_id=${job_id#${prefix}}
    echo "V5 LongBench-v2 forced-choice development already complete: $where"
    echo "LBV2_V5_JOB_ID=$job_id (existing)"
    echo "read: bash $REPORT_DIR/steps.sh --lbv2-v5-dev-read $job_id"
    return 0
  fi
  output=$(sbatch --parsable --partition=debug --time=02:00:00 \
    "$LBV2_V5_WORKER" LBV2_V5_PHASE=development \
    LBV2_V5_RUN_PREFIX=lbv2_v5_development_)
  job_id="${output%%;*}"; job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse a Slurm job ID from: $output" >&2
    exit 1
  }
  echo "submitted V5 LongBench-v2 forced-choice development"
  echo "LBV2_V5_JOB_ID=$job_id"
  echo "split=development items=52 components=44 prediction_rows=468 proxy_rows=416"
  echo "model=llama31-8b ctx=32768 B=2 arms=fp+8 no_generation=1"
  echo "partition=debug walltime=02:00:00"
  echo "log:    h0_measurement/logs/lbv2v5_${job_id}.out"
  echo "result: h0_measurement/results/lbv2_v5_development_${job_id}/"
  echo "next:   bash $REPORT_DIR/steps.sh --lbv2-v5-dev-status $job_id"
}

status_lbv2_v5_development() {
  local job_id=$1 log="h0_measurement/logs/lbv2v5_${1}.out"
  if command -v squeue >/dev/null 2>&1; then
    squeue -j "$job_id" || true
  fi
  if command -v sacct >/dev/null 2>&1; then
    sacct -j "$job_id" --format=JobID,State,Elapsed,ExitCode || true
  fi
  if [[ -f "$log" ]]; then
    echo
    echo "last 30 log lines:"
    tail -n 30 "$log"
  else
    echo "log not created yet: $log"
  fi
}

read_lbv2_v5_development() {
  local job_id=$1 dir predictions proxy out summary_csv lock
  dir=$(lbv2_v5_dir_for_job "$job_id")
  if ! lbv2_v5_artifact_complete "$dir"; then
    echo "ERROR: job $job_id does not satisfy the complete V5 development contract" >&2
    exit 1
  fi
  predictions="$dir/longbench_v2_v5_forced_choice_development.parquet"
  proxy="$dir/longbench_v2_v5_scaffold_proxy_development.parquet"
  out="$REPORT_DIR/longbench_v2_v5_development_${job_id}.txt"
  summary_csv="$REPORT_DIR/longbench_v2_v5_development_${job_id}_summary.csv"
  lock="$REPORT_DIR/longbench_v2_v5_confirmation_lock_${job_id}.json"
  env OMP_NUM_THREADS=8 "$PY" "$LBV2_V5_READER" "$predictions" "$proxy" \
    --manifest "$LBV2_MANIFEST" --dataset "$LBV2_DATA" \
    --csv "$summary_csv" --lock "$lock" > "$out"
  cat "$out"
  echo
  echo "saved: $out"
  echo "saved: $summary_csv"
  if [[ -f "$lock" ]]; then
    echo "saved: $lock"
    echo "next: confirmation remains blocked until its lock-checking worker is implemented"
  else
    echo "lock:  none (the authenticated V5 decision did not advance)"
  fi
}


# V6 is an isolated Qwen qualification. These helpers expose only the frozen
# 20-item FP/uniform screen; there is intentionally no development submission
# until this reader emits an authenticated advance lock.
lbv2_v6_snapshot() {
  printf '%s\n' "$PROJECT_ROOT/.hf_cache/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
}

lbv2_v6_audit() {
  local snapshot
  snapshot=$(lbv2_v6_snapshot)
  need_file "$LBV2_DATA"
  need_file "$LBV2_MANIFEST"
  need_file "$LBV2_V6_MANIFEST"
  need_file "$LBV2_V6_AUDIT"
  [[ -d "$snapshot" && ! -L "$snapshot" ]] || {
    echo "ERROR: missing or symlinked pinned Qwen snapshot $snapshot" >&2
    return 1
  }
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" "$LBV2_V6_AUDIT" \
    --data "$LBV2_DATA" \
    --source-manifest "$LBV2_MANIFEST" \
    --tokenizer "$snapshot" \
    --manifest "$LBV2_V6_MANIFEST"
}

lbv2_v6_command() {
  printf '%q ' sbatch --parsable --partition=debug --time=02:00:00 \
    "$LBV2_V6_WORKER"
  printf '\n'
}

lbv2_v6_artifact_complete() {
  local dir=$1 snapshot predictions
  snapshot=$(lbv2_v6_snapshot)
  predictions="$dir/longbench_v2_v6_qwen_forced_choice_qualification.parquet"
  [[ -f "$predictions" && -f "${predictions%.parquet}.json" ]] || return 1
  [[ -f "$dir/COMPLETE" ]] || return 1
  [[ $(<"$dir/COMPLETE") == "longbench_v2_v6_qwen_qualification_v1" ]] || return 1
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" "$LBV2_V6_READER" \
    --predictions "$predictions" \
    --dataset "$LBV2_DATA" \
    --source-manifest "$LBV2_MANIFEST" \
    --manifest "$LBV2_V6_MANIFEST" \
    --tokenizer "$snapshot" \
    --config h0_measurement/models.yaml \
    --runner h0_measurement/run_longbench_v2_qwen_qualification.py \
    --validate-only >/dev/null 2>&1
}

lbv2_v6_find_complete() {
  local prefix=lbv2_v6_qwen_qualification_ dir base
  for dir in h0_measurement/results/${prefix}*; do
    [[ -d "$dir" ]] || continue
    base=${dir##*/}
    [[ "$base" =~ ^${prefix}[0-9]+$ ]] || continue
    if lbv2_v6_artifact_complete "$dir"; then
      printf '%s\n' "$dir"
      return 0
    fi
  done
  return 1
}

lbv2_v6_dir_for_job() {
  local job_id=$1 dir="h0_measurement/results/lbv2_v6_qwen_qualification_${1}"
  [[ -d "$dir" ]] || {
    echo "ERROR: no exact V6 qualification directory for job $job_id: $dir" >&2
    return 1
  }
  printf '%s\n' "$dir"
}

submit_lbv2_v6_qualification() {
  local prefix=lbv2_v6_qwen_qualification_ where output job_id
  lbv2_v6_audit
  if where=$(lbv2_v6_find_complete); then
    job_id=${where##*/}; job_id=${job_id#${prefix}}
    echo "V6 Qwen qualification already complete: $where"
    echo "LBV2_V6_JOB_ID=$job_id (existing)"
    echo "read: bash $REPORT_DIR/steps.sh --lbv2-v6-qual-read $job_id"
    return 0
  fi
  output=$(sbatch --parsable --partition=debug --time=02:00:00 \
    "$LBV2_V6_WORKER")
  job_id="${output%%;*}"; job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse a Slurm job ID from: $output" >&2
    exit 1
  }
  echo "submitted V6 Qwen LongBench-v2 qualification"
  echo "LBV2_V6_JOB_ID=$job_id"
  echo "split=qualification items=20 components=19 prediction_rows=40"
  echo "model=qwen3-30b-a3b-2507 ctx=40960 B=2 arms=fp,uniform no_generation=1"
  echo "partition=debug walltime=02:00:00"
  echo "log:    h0_measurement/logs/lbv2v6q_${job_id}.out"
  echo "result: h0_measurement/results/${prefix}${job_id}/"
  echo "next:   bash $REPORT_DIR/steps.sh --lbv2-v6-qual-status $job_id"
}

status_lbv2_v6_qualification() {
  local job_id=$1 log="h0_measurement/logs/lbv2v6q_${1}.out"
  local err="h0_measurement/logs/lbv2v6q_${1}.err"
  if command -v squeue >/dev/null 2>&1; then
    squeue -j "$job_id" || true
  fi
  if command -v sacct >/dev/null 2>&1; then
    sacct -j "$job_id" --format=JobID,State,Elapsed,ExitCode || true
  fi
  if [[ -f "$log" ]]; then
    echo
    echo "last 40 stdout lines:"
    tail -n 40 "$log"
  else
    echo "stdout log not created yet: $log"
  fi
  if [[ -s "$err" ]]; then
    echo
    echo "last 40 stderr lines:"
    tail -n 40 "$err"
  fi
}

read_lbv2_v6_qualification() {
  local job_id=$1 dir predictions snapshot out summary_csv lock
  dir=$(lbv2_v6_dir_for_job "$job_id")
  if ! lbv2_v6_artifact_complete "$dir"; then
    echo "ERROR: job $job_id does not satisfy the complete V6 qualification contract" >&2
    exit 1
  fi
  predictions="$dir/longbench_v2_v6_qwen_forced_choice_qualification.parquet"
  snapshot=$(lbv2_v6_snapshot)
  out="$REPORT_DIR/longbench_v2_v6_qualification_${job_id}.txt"
  summary_csv="$REPORT_DIR/longbench_v2_v6_qualification_${job_id}_summary.csv"
  lock="$REPORT_DIR/longbench_v2_v6_qualification_lock_${job_id}.json"
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 \
    "$PY" "$LBV2_V6_READER" \
    --predictions "$predictions" \
    --dataset "$LBV2_DATA" \
    --source-manifest "$LBV2_MANIFEST" \
    --manifest "$LBV2_V6_MANIFEST" \
    --tokenizer "$snapshot" \
    --config h0_measurement/models.yaml \
    --runner h0_measurement/run_longbench_v2_qwen_qualification.py \
    --csv "$summary_csv" --lock "$lock" > "$out"
  cat "$out"
  echo
  echo "saved: $out"
  echo "saved: $summary_csv"
  if [[ -f "$lock" ]]; then
    echo "saved: $lock"
    echo "next: implement the lock-authenticating V6 development worker"
  else
    echo "lock:  none (the authenticated V6 qualification did not advance)"
  fi
}

# V7 is an isolated fresh 128K Qwen qualification. These helpers expose only the frozen
# 20-item FP/uniform screen; there is intentionally no development submission
# until this reader emits an authenticated advance lock.
lbv2_v7_snapshot() {
  printf '%s\n' "$PROJECT_ROOT/.hf_cache/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
}

lbv2_v7_audit() {
  local snapshot
  snapshot=$(lbv2_v7_snapshot)
  need_file "$LBV2_DATA"
  need_file "$LBV2_V6_MANIFEST"
  need_file "$LBV2_V7_MANIFEST"
  need_file "$LBV2_V7_AUDIT"
  need_file "$LBV2_V7_LEDGER"
  [[ -d "$snapshot" && ! -L "$snapshot" ]] || {
    echo "ERROR: missing or symlinked pinned Qwen snapshot $snapshot" >&2
    return 1
  }
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" -c 'import sys; from h0_measurement import run_longbench_v2_qwen_v7_qualification as r; h=r.executed_source_hashes(sys.argv[1], sys.argv[2]); r.verify_source_ledger(sys.argv[2], h); print("PASS V7 canonical source ledger")' \
    h0_measurement/models.yaml "$LBV2_V7_LEDGER"
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" "$LBV2_V7_AUDIT" \
    --data "$LBV2_DATA" \
    --legacy-manifest "$LBV2_V6_MANIFEST" \
    --tokenizer "$snapshot" \
    --manifest "$LBV2_V7_MANIFEST"
}

lbv2_v7_command() {
  printf '%q ' sbatch --parsable --partition=debug --time=02:00:00 \
    "$LBV2_V7_WORKER"
  printf '\n'
}

lbv2_v7_artifact_complete() {
  local dir=$1 snapshot predictions
  snapshot=$(lbv2_v7_snapshot)
  predictions="$dir/longbench_v2_v7_qwen131072_forced_choice_qualification.parquet"
  [[ -f "$predictions" && -f "${predictions%.parquet}.json" ]] || return 1
  [[ -f "$dir/COMPLETE" ]] || return 1
  [[ $(<"$dir/COMPLETE") == "longbench_v2_sieve_v7_qwen131072_qualification_v1" ]] || return 1
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" "$LBV2_V7_READER" \
    --predictions "$predictions" \
    --dataset "$LBV2_DATA" \
    --legacy-manifest "$LBV2_V6_MANIFEST" \
    --manifest "$LBV2_V7_MANIFEST" \
    --tokenizer "$snapshot" \
    --config h0_measurement/models.yaml \
    --source-ledger "$LBV2_V7_LEDGER" \
    --runner "$LBV2_V7_RUNNER" \
    --validate-only >/dev/null 2>&1
}

lbv2_v7_find_complete() {
  local prefix=lbv2_v7_qwen_qualification_ dir base
  for dir in h0_measurement/results/${prefix}*; do
    [[ -d "$dir" ]] || continue
    base=${dir##*/}
    [[ "$base" =~ ^${prefix}[0-9]+$ ]] || continue
    if lbv2_v7_artifact_complete "$dir"; then
      printf '%s\n' "$dir"
      return 0
    fi
  done
  return 1
}

lbv2_v7_dir_for_job() {
  local job_id=$1 dir="h0_measurement/results/lbv2_v7_qwen_qualification_${1}"
  [[ -d "$dir" ]] || {
    echo "ERROR: no exact V7 qualification directory for job $job_id: $dir" >&2
    return 1
  }
  printf '%s\n' "$dir"
}

submit_lbv2_v7_qualification() {
  local prefix=lbv2_v7_qwen_qualification_ where output job_id
  lbv2_v7_audit
  if where=$(lbv2_v7_find_complete); then
    job_id=${where##*/}; job_id=${job_id#${prefix}}
    echo "V7 Qwen qualification already complete: $where"
    echo "LBV2_V7_JOB_ID=$job_id (existing)"
    echo "read: bash $REPORT_DIR/steps.sh --lbv2-v7-qual-read $job_id"
    return 0
  fi
  output=$(sbatch --parsable --partition=debug --time=02:00:00 \
    "$LBV2_V7_WORKER")
  job_id="${output%%;*}"; job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse a Slurm job ID from: $output" >&2
    exit 1
  }
  echo "submitted V7 Qwen LongBench-v2 qualification"
  echo "LBV2_V7_JOB_ID=$job_id"
  echo "split=qualification items=20 components=20 prediction_rows=40"
  echo "model=qwen3-30b-a3b-2507 ctx=131072 B=2 arms=fp,uniform no_generation=1"
  echo "partition=debug walltime=02:00:00"
  echo "log:    h0_measurement/logs/lbv2v7q_${job_id}.out"
  echo "result: h0_measurement/results/${prefix}${job_id}/"
  echo "next:   bash $REPORT_DIR/steps.sh --lbv2-v7-qual-status $job_id"
}

status_lbv2_v7_qualification() {
  local job_id=$1 log="h0_measurement/logs/lbv2v7q_${1}.out"
  local err="h0_measurement/logs/lbv2v7q_${1}.err"
  if command -v squeue >/dev/null 2>&1; then
    squeue -j "$job_id" || true
  fi
  if command -v sacct >/dev/null 2>&1; then
    sacct -j "$job_id" --format=JobID,State,Elapsed,ExitCode || true
  fi
  if [[ -f "$log" ]]; then
    echo
    echo "last 40 stdout lines:"
    tail -n 40 "$log"
  else
    echo "stdout log not created yet: $log"
  fi
  if [[ -s "$err" ]]; then
    echo
    echo "last 40 stderr lines:"
    tail -n 40 "$err"
  fi
}

read_lbv2_v7_qualification() {
  local job_id=$1 dir predictions snapshot out summary_csv lock
  dir=$(lbv2_v7_dir_for_job "$job_id")
  if ! lbv2_v7_artifact_complete "$dir"; then
    echo "ERROR: job $job_id does not satisfy the complete V7 qualification contract" >&2
    exit 1
  fi
  predictions="$dir/longbench_v2_v7_qwen131072_forced_choice_qualification.parquet"
  snapshot=$(lbv2_v7_snapshot)
  out="$REPORT_DIR/longbench_v2_v7_qualification_${job_id}.txt"
  summary_csv="$REPORT_DIR/longbench_v2_v7_qualification_${job_id}_summary.csv"
  lock="$REPORT_DIR/longbench_v2_v7_qualification_lock_${job_id}.json"
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 \
    "$PY" "$LBV2_V7_READER" \
    --predictions "$predictions" \
    --dataset "$LBV2_DATA" \
    --legacy-manifest "$LBV2_V6_MANIFEST" \
    --manifest "$LBV2_V7_MANIFEST" \
    --tokenizer "$snapshot" \
    --config h0_measurement/models.yaml \
    --source-ledger "$LBV2_V7_LEDGER" \
    --runner "$LBV2_V7_RUNNER" \
    --csv "$summary_csv" --lock "$lock" > "$out"
  cat "$out"
  echo
  echo "saved: $out"
  echo "saved: $summary_csv"
  if [[ -f "$lock" ]]; then
    echo "saved: $lock"
    echo "next: create and authenticate the deterministic V7 development ledger"
  else
    echo "lock:  none (the authenticated V7 qualification did not advance; V7 is terminal)"
  fi
}

# V6 development is authorized only by the exact advancing qualification lock
# from job 984224. Every dry run, submission, and read reauthenticates it.
lbv2_v6_dev_authorize() {
  local snapshot
  snapshot=$(lbv2_v6_snapshot)
  need_file "$LBV2_V6_DEV_RUNNER"
  need_file "$LBV2_V6_DEV_WORKER"
  need_file "$LBV2_V6_DEV_READER"
  need_file "$LBV2_V6_QUAL_LOCK"
  [[ -d "$snapshot" && ! -L "$snapshot" ]] || {
    echo "ERROR: missing or symlinked pinned Qwen snapshot $snapshot" >&2
    return 1
  }
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 \
    "$PY" -c "import sys; from h0_measurement import run_longbench_v2_qwen_development as r; r.authenticate_qualification_lock(sys.argv[1], dataset_path=sys.argv[2], source_manifest_path=sys.argv[3], manifest_path=sys.argv[4], config_path=sys.argv[5], model_source=sys.argv[6]); print('PASS V6 development qualification-lock authorization')" \
    "$LBV2_V6_QUAL_LOCK" "$LBV2_DATA" "$LBV2_MANIFEST" \
    "$LBV2_V6_MANIFEST" h0_measurement/models.yaml "$snapshot"
}

lbv2_v6_dev_command() {
  printf '%q ' sbatch --parsable --partition=debug --time=02:00:00 \
    "$LBV2_V6_DEV_WORKER"
  printf '\n'
}

lbv2_v6_dev_artifact_complete() {
  local dir=$1 snapshot predictions proxy
  snapshot=$(lbv2_v6_snapshot)
  predictions="$dir/longbench_v2_v6_qwen_forced_choice_development.parquet"
  proxy="$dir/longbench_v2_v6_qwen_scaffold_proxy_development.parquet"
  [[ -f "$predictions" && -f "${predictions%.parquet}.json" ]] || return 1
  [[ -f "$proxy" && -f "${proxy%.parquet}.json" ]] || return 1
  [[ -f "$dir/COMPLETE" ]] || return 1
  [[ $(<"$dir/COMPLETE") == "longbench_v2_v6_qwen_development_v1" ]] || return 1
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 \
    "$PY" "$LBV2_V6_DEV_READER" "$predictions" "$proxy" \
    --manifest "$LBV2_V6_MANIFEST" \
    --source-manifest "$LBV2_MANIFEST" \
    --dataset "$LBV2_DATA" \
    --model-source "$snapshot" \
    --qualification-lock "$LBV2_V6_QUAL_LOCK" \
    --config h0_measurement/models.yaml \
    --runner "$LBV2_V6_DEV_RUNNER" \
    --validate-only >/dev/null 2>&1
}

lbv2_v6_dev_find_complete() {
  local prefix=lbv2_v6_qwen_development_ dir base
  for dir in h0_measurement/results/${prefix}*; do
    [[ -d "$dir" ]] || continue
    base=${dir##*/}
    [[ "$base" =~ ^${prefix}[0-9]+$ ]] || continue
    if lbv2_v6_dev_artifact_complete "$dir"; then
      printf '%s\n' "$dir"
      return 0
    fi
  done
  return 1
}

lbv2_v6_dev_dir_for_job() {
  local job_id=$1 dir="h0_measurement/results/lbv2_v6_qwen_development_${1}"
  [[ -d "$dir" ]] || {
    echo "ERROR: no exact V6 development directory for job $job_id: $dir" >&2
    return 1
  }
  printf '%s\n' "$dir"
}

submit_lbv2_v6_development() {
  local prefix=lbv2_v6_qwen_development_ where output job_id
  lbv2_v6_dev_authorize
  lbv2_v6_audit
  if where=$(lbv2_v6_dev_find_complete); then
    job_id=${where##*/}; job_id=${job_id#${prefix}}
    echo "V6 Qwen development already complete: $where"
    echo "LBV2_V6_DEV_JOB_ID=$job_id (existing)"
    echo "read: bash $REPORT_DIR/steps.sh --lbv2-v6-dev-read $job_id"
    return 0
  fi
  output=$(sbatch --parsable --partition=debug --time=02:00:00 \
    "$LBV2_V6_DEV_WORKER")
  job_id="${output%%;*}"; job_id="${job_id##* }"
  [[ "$job_id" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not parse a Slurm job ID from: $output" >&2
    exit 1
  }
  echo "submitted V6 Qwen LongBench-v2 development"
  echo "LBV2_V6_DEV_JOB_ID=$job_id"
  echo "split=development items=52 components=44 prediction_rows=468 proxy_rows=416"
  echo "model=qwen3-30b-a3b-2507 ctx=40960 B=2 arms=fp+8_candidates"
  echo "partition=debug walltime=02:00:00"
  echo "log:    h0_measurement/logs/lbv2v6d_${job_id}.out"
  echo "result: h0_measurement/results/${prefix}${job_id}/"
  echo "next:   bash $REPORT_DIR/steps.sh --lbv2-v6-dev-status $job_id"
}

status_lbv2_v6_development() {
  local job_id=$1 log="h0_measurement/logs/lbv2v6d_${1}.out"
  local err="h0_measurement/logs/lbv2v6d_${1}.err"
  if command -v squeue >/dev/null 2>&1; then
    squeue -j "$job_id" || true
  fi
  if command -v sacct >/dev/null 2>&1; then
    sacct -j "$job_id" --format=JobID,State,Elapsed,MaxRSS,ExitCode || true
  fi
  if [[ -f "$log" ]]; then
    echo
    echo "last 40 stdout lines:"
    tail -n 40 "$log"
  else
    echo "stdout log not created yet: $log"
  fi
  if [[ -s "$err" ]]; then
    echo
    echo "last 40 stderr lines:"
    tail -n 40 "$err"
  fi
}

read_lbv2_v6_development() {
  local job_id=$1 dir predictions proxy snapshot out summary_csv lock
  dir=$(lbv2_v6_dev_dir_for_job "$job_id")
  if ! lbv2_v6_dev_artifact_complete "$dir"; then
    echo "ERROR: job $job_id does not satisfy the complete V6 development contract" >&2
    exit 1
  fi
  predictions="$dir/longbench_v2_v6_qwen_forced_choice_development.parquet"
  proxy="$dir/longbench_v2_v6_qwen_scaffold_proxy_development.parquet"
  snapshot=$(lbv2_v6_snapshot)
  out="$REPORT_DIR/longbench_v2_v6_development_${job_id}.txt"
  summary_csv="$REPORT_DIR/longbench_v2_v6_development_${job_id}_summary.csv"
  lock="$REPORT_DIR/longbench_v2_v6_confirmation_lock_${job_id}.json"
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 \
    "$PY" "$LBV2_V6_DEV_READER" "$predictions" "$proxy" \
    --manifest "$LBV2_V6_MANIFEST" \
    --source-manifest "$LBV2_MANIFEST" \
    --dataset "$LBV2_DATA" \
    --model-source "$snapshot" \
    --qualification-lock "$LBV2_V6_QUAL_LOCK" \
    --config h0_measurement/models.yaml \
    --runner "$LBV2_V6_DEV_RUNNER" \
    --csv "$summary_csv" --lock "$lock" > "$out"
  cat "$out"
  echo
  echo "saved: $out"
  echo "saved: $summary_csv"
  if [[ -f "$lock" ]]; then
    echo "saved: $lock"
    echo "next: implement the lock-authenticating V6 confirmation worker"
  else
    echo "lock:  none (the authenticated V6 development decision did not advance)"
  fi
}

MODE="${1:-}"
case "$MODE" in
  --lbv2-v6-dev-dry)
    (($# == 1)) || { echo "ERROR: --lbv2-v6-dev-dry accepts no arguments" >&2; exit 2; }
    need_file "$LBV2_V6_DEV_RUNNER"; need_file "$LBV2_V6_DEV_WORKER"; need_file "$LBV2_V6_DEV_READER"
    need_file "$LBV2_V6_QUAL_LOCK"; need_file "$LBV2_V6_AUDIT"
    need_file "$LBV2_DATA"; need_file "$LBV2_MANIFEST"; need_file "$LBV2_V6_MANIFEST"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    lbv2_v6_dev_authorize
    lbv2_v6_audit
    echo "DRY RUN; submit nothing:"
    lbv2_v6_dev_command
    ;;
  --lbv2-v6-dev-submit)
    (($# == 1)) || { echo "ERROR: --lbv2-v6-dev-submit accepts no arguments" >&2; exit 2; }
    need_file "$LBV2_V6_DEV_RUNNER"; need_file "$LBV2_V6_DEV_WORKER"; need_file "$LBV2_V6_DEV_READER"
    need_file "$LBV2_V6_QUAL_LOCK"; need_file "$LBV2_V6_AUDIT"
    need_file "$LBV2_DATA"; need_file "$LBV2_MANIFEST"; need_file "$LBV2_V6_MANIFEST"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
    submit_lbv2_v6_development
    ;;
  --lbv2-v6-dev-status)
    (($# == 2)) || { echo "ERROR: --lbv2-v6-dev-status requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    status_lbv2_v6_development "$2"
    ;;
  --lbv2-v6-dev-read)
    (($# == 2)) || { echo "ERROR: --lbv2-v6-dev-read requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    need_file "$LBV2_V6_DEV_RUNNER"; need_file "$LBV2_V6_DEV_READER"; need_file "$LBV2_V6_QUAL_LOCK"
    need_file "$LBV2_DATA"; need_file "$LBV2_MANIFEST"; need_file "$LBV2_V6_MANIFEST"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    read_lbv2_v6_development "$2"
    ;;
  --lbv2-v7-audit)
    (($# == 1)) || { echo "ERROR: --lbv2-v7-audit accepts no arguments" >&2; exit 2; }
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    lbv2_v7_audit
    ;;
  --lbv2-v7-qual-dry)
    (($# == 1)) || { echo "ERROR: --lbv2-v7-qual-dry accepts no arguments" >&2; exit 2; }
    need_file "$LBV2_V7_RUNNER"; need_file "$LBV2_V7_WORKER"; need_file "$LBV2_V7_RUNNER"; need_file "$LBV2_V7_READER"; need_file "$LBV2_V7_AUDIT"
    need_file "$LBV2_DATA"; need_file "$LBV2_V6_MANIFEST"; need_file "$LBV2_V7_MANIFEST"; need_file "$LBV2_V7_LEDGER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    lbv2_v7_audit
    echo "DRY RUN; submit nothing:"
    lbv2_v7_command
    ;;
  --lbv2-v7-qual-submit)
    (($# == 1)) || { echo "ERROR: --lbv2-v7-qual-submit accepts no arguments" >&2; exit 2; }
    need_file "$LBV2_V7_RUNNER"; need_file "$LBV2_V7_WORKER"; need_file "$LBV2_V7_RUNNER"; need_file "$LBV2_V7_READER"; need_file "$LBV2_V7_AUDIT"
    need_file "$LBV2_DATA"; need_file "$LBV2_V6_MANIFEST"; need_file "$LBV2_V7_MANIFEST"; need_file "$LBV2_V7_LEDGER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
    submit_lbv2_v7_qualification
    ;;
  --lbv2-v7-qual-status)
    (($# == 2)) || { echo "ERROR: --lbv2-v7-qual-status requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    status_lbv2_v7_qualification "$2"
    ;;
  --lbv2-v7-qual-read)
    (($# == 2)) || { echo "ERROR: --lbv2-v7-qual-read requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    need_file "$LBV2_V7_RUNNER"; need_file "$LBV2_V7_READER"; need_file "$LBV2_V7_AUDIT"
    need_file "$LBV2_DATA"; need_file "$LBV2_V6_MANIFEST"; need_file "$LBV2_V7_MANIFEST"; need_file "$LBV2_V7_LEDGER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    read_lbv2_v7_qualification "$2"
    ;;
  --lbv2-v6-audit)
    (($# == 1)) || { echo "ERROR: --lbv2-v6-audit accepts no arguments" >&2; exit 2; }
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    lbv2_v6_audit
    ;;
  --lbv2-v6-qual-dry)
    (($# == 1)) || { echo "ERROR: --lbv2-v6-qual-dry accepts no arguments" >&2; exit 2; }
    need_file "$LBV2_V6_WORKER"; need_file "$LBV2_V6_READER"; need_file "$LBV2_V6_AUDIT"
    need_file "$LBV2_DATA"; need_file "$LBV2_MANIFEST"; need_file "$LBV2_V6_MANIFEST"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    lbv2_v6_audit
    echo "DRY RUN; submit nothing:"
    lbv2_v6_command
    ;;
  --lbv2-v6-qual-submit)
    (($# == 1)) || { echo "ERROR: --lbv2-v6-qual-submit accepts no arguments" >&2; exit 2; }
    need_file "$LBV2_V6_WORKER"; need_file "$LBV2_V6_READER"; need_file "$LBV2_V6_AUDIT"
    need_file "$LBV2_DATA"; need_file "$LBV2_MANIFEST"; need_file "$LBV2_V6_MANIFEST"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
    submit_lbv2_v6_qualification
    ;;
  --lbv2-v6-qual-status)
    (($# == 2)) || { echo "ERROR: --lbv2-v6-qual-status requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    status_lbv2_v6_qualification "$2"
    ;;
  --lbv2-v6-qual-read)
    (($# == 2)) || { echo "ERROR: --lbv2-v6-qual-read requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    need_file "$LBV2_V6_READER"; need_file "$LBV2_V6_AUDIT"
    need_file "$LBV2_DATA"; need_file "$LBV2_MANIFEST"; need_file "$LBV2_V6_MANIFEST"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    read_lbv2_v6_qualification "$2"
    ;;
  --lbv2-v5-dev-dry)
    (($# == 1)) || { echo "ERROR: --lbv2-v5-dev-dry accepts no arguments" >&2; exit 2; }
    need_file "$LBV2_V5_WORKER"; need_file "$LBV2_V5_READER"
    need_file "$LBV2_DATA"; need_file "$LBV2_MANIFEST"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    echo "DRY RUN; submit nothing:"
    lbv2_v5_command
    ;;
  --lbv2-v5-dev-submit)
    (($# == 1)) || { echo "ERROR: --lbv2-v5-dev-submit accepts no arguments" >&2; exit 2; }
    need_file "$LBV2_V5_WORKER"; need_file "$LBV2_V5_READER"
    need_file "$LBV2_DATA"; need_file "$LBV2_MANIFEST"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
    submit_lbv2_v5_development
    ;;
  --lbv2-v5-dev-status)
    (($# == 2)) || { echo "ERROR: --lbv2-v5-dev-status requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    status_lbv2_v5_development "$2"
    ;;
  --lbv2-v5-dev-read)
    (($# == 2)) || { echo "ERROR: --lbv2-v5-dev-read requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    need_file "$LBV2_V5_READER"; need_file "$LBV2_DATA"; need_file "$LBV2_MANIFEST"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    read_lbv2_v5_development "$2"
    ;;
  --lbv2-audit)
    (($# == 1)) || { echo "ERROR: --lbv2-audit accepts no arguments" >&2; exit 2; }
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    lbv2_audit
    ;;
  --lbv2-qual-dry)
    (($# == 1)) || { echo "ERROR: --lbv2-qual-dry accepts no arguments" >&2; exit 2; }
    need_file "$LBV2_WORKER"; need_file "$LBV2_READER"
    need_file "$LBV2_DATA"; need_file "$LBV2_MANIFEST"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    echo "DRY RUN; submit nothing:"
    lbv2_command
    ;;
  --lbv2-qual-submit)
    (($# == 1)) || { echo "ERROR: --lbv2-qual-submit accepts no arguments" >&2; exit 2; }
    need_file "$LBV2_WORKER"; need_file "$LBV2_READER"
    need_file "$LBV2_DATA"; need_file "$LBV2_MANIFEST"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
    submit_lbv2_qualification
    ;;
  --lbv2-qual-status)
    (($# == 2)) || { echo "ERROR: --lbv2-qual-status requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    status_lbv2_qualification "$2"
    ;;
  --lbv2-qual-read)
    (($# == 2)) || { echo "ERROR: --lbv2-qual-read requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    need_file "$LBV2_READER"; need_file "$LBV2_DATA"; need_file "$LBV2_MANIFEST"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    read_lbv2_qualification "$2"
    ;;
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
  --op2-dry)
    (($# <= 2)) || { echo "ERROR: --op2-dry accepts at most one selector" >&2; exit 2; }
    need_file "$WORKER"
    op2_specs "${2:-screen}" >/dev/null
    echo "DRY RUN; submit nothing:"
    while read -r nk prefix; do op2_command "$nk" "$prefix"; done \
      < <(op2_specs "${2:-screen}")
    ;;
  --op2-submit)
    (($# <= 2)) || { echo "ERROR: --op2-submit accepts at most one selector" >&2; exit 2; }
    need_file "$WORKER"; need_file "$OP2_READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    op2_specs "${2:-screen}" >/dev/null
    command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
    while read -r nk prefix; do submit_op2_one "$nk" "$prefix"; done \
      < <(op2_specs "${2:-screen}")
    ;;
  --op2-status)
    (($# >= 2)) || { echo "ERROR: at least one JOB_ID is required" >&2; exit 2; }
    status_difficulty "$@"
    ;;
  --op2-read)
    (($# == 3)) || {
      echo "ERROR: --op2-read requires exactly K24_JOB_ID K32_JOB_ID (in that order)" >&2
      exit 2
    }
    need_job_id "$2"; need_job_id "$3"
    need_file "$OP2_READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    read_op2_pair "$2" "$3"
    ;;
  --op2-k40-dry)
    (($# == 1)) || { echo "ERROR: --op2-k40-dry accepts no arguments" >&2; exit 2; }
    need_file "$WORKER"; need_file "$OP2_READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    op2_k40_require_v2a_branch
    echo "DRY RUN; authenticated V2-A jobs 982121/982122 as the both-too-easy branch; submit nothing:"
    op2_k40_command
    ;;
  --op2-k40-submit)
    (($# == 1)) || { echo "ERROR: --op2-k40-submit accepts no arguments" >&2; exit 2; }
    need_file "$WORKER"; need_file "$OP2_READER"; need_file "$OP2_K40_READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
    op2_k40_require_v2a_branch
    submit_op2_k40
    ;;
  --op2-k40-status)
    (($# == 2)) || { echo "ERROR: --op2-k40-status requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    status_difficulty "$@"
    ;;
  --op2-k40-read)
    (($# == 2)) || { echo "ERROR: --op2-k40-read requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    need_file "$OP2_READER"; need_file "$OP2_K40_READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    op2_k40_require_v2a_branch
    read_op2_k40_job "$2"
    ;;
  --op2-k48-dry)
    (($# == 1)) || { echo "ERROR: --op2-k48-dry accepts no arguments" >&2; exit 2; }
    need_file "$WORKER"; need_file "$OP2_K40_READER"; need_file "$OP2_K48_READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    op2_k48_require_k40_branch
    echo "DRY RUN; authenticated k40 job 982613 as directionally too easy; submit nothing:"
    op2_k48_command
    ;;
  --op2-k48-submit)
    (($# == 1)) || { echo "ERROR: --op2-k48-submit accepts no arguments" >&2; exit 2; }
    need_file "$WORKER"; need_file "$OP2_K40_READER"; need_file "$OP2_K48_READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
    op2_k48_require_k40_branch
    submit_op2_k48
    ;;
  --op2-k48-status)
    (($# == 2)) || { echo "ERROR: --op2-k48-status requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    status_difficulty "$@"
    ;;
  --op2-k48-read)
    (($# == 2)) || { echo "ERROR: --op2-k48-read requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    need_file "$OP2_K40_READER"; need_file "$OP2_K48_READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    op2_k48_require_k40_branch
    read_op2_k48_job "$2"
    ;;
  --panel-dry)
    (($# == 1)) || { echo "ERROR: --panel-dry accepts no arguments" >&2; exit 2; }
    need_file "$WORKER"; need_file "$PANEL_READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    echo "DRY RUN; submit nothing:"
    panel_command
    ;;
  --panel-submit)
    (($# == 1)) || { echo "ERROR: --panel-submit accepts no arguments" >&2; exit 2; }
    need_file "$WORKER"; need_file "$PANEL_READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
    submit_panel
    ;;
  --panel-status)
    (($# == 2)) || { echo "ERROR: --panel-status requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    status_difficulty "$@"
    ;;
  --panel-read)
    (($# == 2)) || { echo "ERROR: --panel-read requires exactly one JOB_ID" >&2; exit 2; }
    need_job_id "$2"
    need_file "$PANEL_READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    read_panel_job "$2"
    ;;
  --v2b-dry)
    (($# == 4)) || {
      echo "ERROR: --v2b-dry requires K24_JOB_ID K32_JOB_ID SELECTED_K" >&2
      exit 2
    }
    need_job_id "$2"; need_job_id "$3"; need_v2b_n_keys "$4"
    need_file "$WORKER"; need_file "$OP2_READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    v2b_require_op2_selection "$4" "$2" "$3"
    echo "DRY RUN; authenticated V2-A selected k=$4; submit nothing:"
    v2b_command "$4"
    ;;
  --v2b-submit)
    (($# == 4)) || {
      echo "ERROR: --v2b-submit requires K24_JOB_ID K32_JOB_ID SELECTED_K" >&2
      exit 2
    }
    need_job_id "$2"; need_job_id "$3"; need_v2b_n_keys "$4"
    need_file "$WORKER"; need_file "$OP2_READER"; need_file "$V2B_READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch is unavailable" >&2; exit 1; }
    v2b_require_op2_selection "$4" "$2" "$3"
    submit_v2b "$4"
    ;;
  --v2b-status)
    (($# >= 2)) || { echo "ERROR: at least one JOB_ID is required" >&2; exit 2; }
    status_difficulty "$@"
    ;;
  --v2b-read)
    (($# == 5)) || {
      echo "ERROR: --v2b-read requires K24_JOB_ID K32_JOB_ID SELECTED_K V2B_JOB_ID" >&2
      exit 2
    }
    need_job_id "$2"; need_job_id "$3"; need_v2b_n_keys "$4"; need_job_id "$5"
    need_file "$OP2_READER"; need_file "$V2B_READER"
    [[ -x "$PY" ]] || { echo "ERROR: missing executable $PY" >&2; exit 1; }
    v2b_require_op2_selection "$4" "$2" "$3"
    read_v2b_job "$4" "$5"
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
