#!/usr/bin/env bash
# =============================================================================
# R3 RE-RUN SHEET (2026-09-19, after the fresh-token fix). Generated from
# script.sh; submits ONLY what still lacks a FIXED result:
#
#     bash h0_measurement/bugs/2_towards_real_evictor/script_temp.sh
#
# WHY EVERYTHING IN R3 RE-RUNS. job214003* / job214064* EVICTED the token each
# lagged score had never seen (R3-report.md section 2): every interior column
# (gain_pp*, interior_lag_cost*) is invalid, and so is the lag:k corner for
# k >= 2. Their oracle / E2-cell columns are valid, but those come back for free
# in the re-run, so no R3 command counts as done.
#
# GUARDED. Each line submits only the models with no COMPLETE, FIXED result
# under h0_measurement/results/job*/: PAR1 footer, a .json beside it, the
# corner tag, and "interior_unseen_policy": "floor_maxb" -- which job214* does
# not have, so it can never satisfy the guard. Re-running this sheet after a
# partial failure submits only what is still missing.
#
# SAVING vs script.sh: section B main drops mistral-7b and qwen15-moe, whose
# registry ctx (32768) is the SAME cell as section A's 32k line.
#
# Submit from trig-login01 with the fixed sievelib/{evict,alloc}.py and
# h0_measurement/run_h0.py on that machine -- the unit-test gate below runs
# test_unseen_floor and refuses to submit without the fix.
#
# ---- script.sh header follows ----
# bug 2 -- submission sheet.  NOTHING RUNS WITHOUT A SECTION FLAG.
#
#     bash script.sh --r3 --pilot   ONE cheap job that sizes section C. Do first.
#     bash script.sh --r3           re-run R3 (the current experiment)
#     bash script.sh --legacy       re-submit the ORIGINAL E1/E2 grid (history)
#
# Gated because the file is both a record and a runnable sheet, and the two
# halves must not fire together: a bare `bash script.sh` used to submit the
# entire historical E1/E2 grid.
#
# ENV-PREFIX OVERRIDES DO NOT WORK ON THIS CLUSTER. Its exported `sbatch` shell
# function adds --export=NONE, so `SIEVE_X=... sbatch` reaches no job; that is
# what emptied job92* and the first pilot job934606. The --r3 section passes
# overrides as script ARGUMENTS instead (submit_h0*.slurm export them). The
# --legacy lines keep the old env-prefix form as the record of what ran on the
# previous cluster (job2001* JSONs confirm they applied there); rewrite them as
# arguments before ever re-submitting them here.
# =============================================================================
case "${1:-}" in
  ""|--r3)  SECTION="--r3" ;;       # re-run sheet: no flag submits the re-runs
  --legacy) SECTION="$1" ;;
  *) echo "usage: bash $0      (submits the R3 re-runs only)"; exit 0 ;;
esac

if [[ "$SECTION" == "--legacy" ]]; then
# REFUSED ON THIS CLUSTER. Every line below would misfire on Trillium: the
# SIEVE_* env prefixes are dropped by the --export=NONE sbatch function (the
# jobs would silently run the per-model defaults), --mem=... is rejected
# outright, and submit_h0_ctx_sweep.slurm does not parse SIEVE_* arguments at
# all. Kept verbatim as the record of what ran on the previous cluster.
echo "--legacy is a record of the previous cluster's E1/E2 grid; it cannot be"
echo "re-submitted on Trillium as written (see the note in this block). Rewrite"
echo "the lines with SIEVE_* as ARGUMENTS and without --mem first."
exit 1
# ----- BEGIN legacy E1/E2 grid (already executed; kept for reproduction) -----
# =============================================================================
# bug 2 (honest eviction corner) -- submission sheet.
#
# One sbatch per (ctx, tier). The array gives one model per task.
#
# WHAT TO SPLIT ON -- MEASURED, not guessed. Do NOT split on the corner axis.
# Timing quant_metrics at L=8192 over 4 budgets (median of 5, warmed):
#     oracle                        46.8 ms   x1.00
#     oracle,accum                  52.2 ms   x1.11
#     oracle,accum,window           55.2 ms   x1.18
#     oracle,accum,window,recency   55.4 ms   x1.18
# One evict_error_curve pass is 0.6 ms against a ~47 ms base that is dominated by
# waterfill + exact_error(wf) + exact_error(uniform) + noise_model. Adding all
# three practical evictors costs ~14% of a task; the frac/abs policy axis rides
# the same pass and is free (alloc.py:332 loops over EVICTORS, not policies).
#
# So a per-evictor split saves ~6% per job and pays for it by re-running the
# whole prefill, the whole decode, the entire 7-width quantize_keys sweep, and
# the oracle corner once PER SUBMISSION -- about 3x the total GPU time for three
# corners, and worse at long ctx where prefill dominates. It also destroys the
# min-over-corners verdict, which needs every corner in one frame.
#
# One submission per (ctx, tier), all corners in it. The array already gives one
# model per task, which is the real unit of queue time. If a task is still too
# long, the levers are n_prompts / quant_every / bit_list / budgets (models.yaml),
# not the corner set.
#
# Splitting on POLICY additionally breaks K*: it is defined against
# pol0 = "frac" if present else the first policy (alloc.py:325), so an abs-only
# job silently measures K* against the abs budget instead of the fractional one.
#
# ALWAYS INCLUDE `oracle`. It holds no host state (0 B/slot) and costs one curve
# pass, and without it run_h0 emits no gain_e<B>, gain_best<B>,
# oracle_evict_advantage<B> or kstar<B> -- exactly what left the job1998*
# campaign unable to show its own before/after.
#
# --array MUST match the item count:
#   submit_h0.slurm              4 models -> --array=0-3
#   submit_h0_large_models.slurm 2 models -> --array=0-1
#   submit_h0_ctx_sweep.slurm    N ctx    -> --array=0-(N-1)
# =============================================================================


# --- ROUND 1 (already run as job1998*, --array fixed) ------------------------
# Kept for the record. These had NO oracle, so their parquets carry no
# gain_best<B> / oracle_evict_advantage<B> / kstar<B>; round 2 supersedes them.

SIEVE_CTX=8192 SIEVE_EVICTORS='accum' SIEVE_CORNER_POLICIES='frac' sbatch --array=0-3 h0_measurement/submit_h0.slurm
SIEVE_CTX=8192 SIEVE_EVICTORS='accum' SIEVE_CORNER_POLICIES='abs' sbatch --array=0-3 h0_measurement/submit_h0.slurm
SIEVE_CTX=8192 SIEVE_EVICTORS='window' SIEVE_CORNER_POLICIES='frac' sbatch --array=0-3 h0_measurement/submit_h0.slurm
SIEVE_CTX=8192 SIEVE_EVICTORS='window' SIEVE_CORNER_POLICIES='abs' sbatch --array=0-3 h0_measurement/submit_h0.slurm

SIEVE_CTX=32768 SIEVE_EVICTORS='accum' SIEVE_CORNER_POLICIES='frac' sbatch --array=0-3 h0_measurement/submit_h0.slurm
SIEVE_CTX=32768 SIEVE_EVICTORS='accum' SIEVE_CORNER_POLICIES='abs' sbatch --array=0-3 h0_measurement/submit_h0.slurm
SIEVE_CTX=32768 SIEVE_EVICTORS='window' SIEVE_CORNER_POLICIES='frac' sbatch --array=0-3 h0_measurement/submit_h0.slurm
SIEVE_CTX=32768 SIEVE_EVICTORS='window' SIEVE_CORNER_POLICIES='abs' sbatch --array=0-3 h0_measurement/submit_h0.slurm

SIEVE_CTX=8192 SIEVE_EVICTORS='accum' SIEVE_CORNER_POLICIES='frac' sbatch --array=0-1 h0_measurement/submit_h0_large_models.slurm
SIEVE_CTX=8192 SIEVE_EVICTORS='accum' SIEVE_CORNER_POLICIES='abs' sbatch --array=0-1 h0_measurement/submit_h0_large_models.slurm
SIEVE_CTX=8192 SIEVE_EVICTORS='window' SIEVE_CORNER_POLICIES='frac' sbatch --array=0-1 h0_measurement/submit_h0_large_models.slurm
SIEVE_CTX=8192 SIEVE_EVICTORS='window' SIEVE_CORNER_POLICIES='abs' sbatch --array=0-1 h0_measurement/submit_h0_large_models.slurm

SIEVE_CTX=32768 SIEVE_EVICTORS='accum' SIEVE_CORNER_POLICIES='frac' sbatch --array=0-1 h0_measurement/submit_h0_large_models.slurm
SIEVE_CTX=32768 SIEVE_EVICTORS='accum' SIEVE_CORNER_POLICIES='abs' sbatch --array=0-1 h0_measurement/submit_h0_large_models.slurm
SIEVE_CTX=32768 SIEVE_EVICTORS='window' SIEVE_CORNER_POLICIES='frac' sbatch --array=0-1 h0_measurement/submit_h0_large_models.slurm
SIEVE_CTX=32768 SIEVE_EVICTORS='window' SIEVE_CORNER_POLICIES='abs' sbatch --array=0-1 h0_measurement/submit_h0_large_models.slurm

SIEVE_EVICTORS='oracle,accum,window' SIEVE_CORNER_POLICIES='frac,abs' sbatch --array=0-2 --time=10:00:00 --mem=496G --gpus-per-node=4 h0_measurement/submit_h0_ctx_sweep.slurm qwen3-30b-a3b-2507 8192 32768 131072

# FIXED: was --array=0-5 with ONE ctx value; the script rejects tasks 1..5.
SIEVE_EVICTORS='oracle,accum,window' SIEVE_CORNER_POLICIES='frac,abs' sbatch --array=0-0 --mem=496G --gpus-per-node=4 h0_measurement/submit_h0_ctx_sweep.slurm qwen3-30b-a3b-2507 131072


# --- ROUND 2: OPP 1 + 2 + 3 -- oracle back in, plus recency ------------------
# Restores the in-run before/after (gain_best<B> vs gain_best_practical<B>),
# oracle_evict_advantage<B>, and kstar<B>. kappa needs no separate run: the
# kstar_over_n95<B> column IS the kappa that would make abs match frac (median
# 27.9 at 128K, against the configured 4.0).
# Host RAM per layer-head-token: accum 5 B, window 17 B, recency 1 B.

SIEVE_CTX=8192 SIEVE_EVICTORS='oracle,accum,window,recency' SIEVE_CORNER_POLICIES='frac,abs' sbatch --array=0-3 h0_measurement/submit_h0.slurm
SIEVE_CTX=16384 SIEVE_EVICTORS='oracle,accum,window,recency' SIEVE_CORNER_POLICIES='frac,abs' sbatch --array=0-3 h0_measurement/submit_h0.slurm
SIEVE_CTX=32768 SIEVE_EVICTORS='oracle,accum,window,recency' SIEVE_CORNER_POLICIES='frac,abs' sbatch --array=0-3 h0_measurement/submit_h0.slurm
SIEVE_CTX=65536 SIEVE_EVICTORS='oracle,accum,window,recency' SIEVE_CORNER_POLICIES='frac,abs' sbatch --array=0-3 h0_measurement/submit_h0.slurm

SIEVE_CTX=8192 SIEVE_EVICTORS='oracle,accum,window,recency' SIEVE_CORNER_POLICIES='frac,abs' sbatch --array=0-1 h0_measurement/submit_h0_large_models.slurm
SIEVE_CTX=16384 SIEVE_EVICTORS='oracle,accum,window,recency' SIEVE_CORNER_POLICIES='frac,abs' sbatch --array=0-1 h0_measurement/submit_h0_large_models.slurm
SIEVE_CTX=32768 SIEVE_EVICTORS='oracle,accum,window,recency' SIEVE_CORNER_POLICIES='frac,abs' sbatch --array=0-1 h0_measurement/submit_h0_large_models.slurm
SIEVE_CTX=65536 SIEVE_EVICTORS='oracle,accum,window,recency' SIEVE_CORNER_POLICIES='frac,abs' sbatch --array=0-1 h0_measurement/submit_h0_large_models.slurm

# --- OPP 4: does the oracle's edge stay on DIFFUSE heads at every ctx? -------
# At 128K, spearman(oracle_evict_advantage, n95) = +0.51 and vs tau = -0.70 --
# the opposite of why.md's stated mechanism. One ctx cannot show whether that
# flips with L. llama31-8b is the only main-tier model that reaches 128k natively
# (16 GB weights + 17 GB KV on one H100; 3.1 GB host state at 128k, 23 B/slot).

SIEVE_EVICTORS='oracle,accum,window,recency' SIEVE_CORNER_POLICIES='frac,abs' sbatch --array=0-4 --time=06:00:00 --mem=248G h0_measurement/submit_h0_ctx_sweep.slurm llama31-8b 8192 16384 32768 65536 131072


# --- OPP 5: the 128k row -- currently ONE model (llama31-8b, 28.8% NARROW) -----
# submit_h0_ctx_sweep.slurm takes ONE model tag, then its ctx list (MODEL="$1";
# shift). A single line naming two models makes the second one a ctx value, and
# the script rejects it: "ERROR: ctx values must be positive integers, got
# 'qwen3-30b-a3b-2507'". Two submissions, one model each, --array=0-0 for one ctx.
# Host state at 128k, 23 B/slot: llama33-70b 15.4 GB, qwen3-30b 4.6 GB.

SIEVE_EVICTORS='oracle,accum,window,recency' SIEVE_CORNER_POLICIES='frac,abs' sbatch --array=0-0 --gpus-per-node=4 --mem=496G --time=02:00:00 h0_measurement/submit_h0_ctx_sweep.slurm llama33-70b 131072

SIEVE_EVICTORS='oracle,accum,window,recency' SIEVE_CORNER_POLICIES='frac,abs' sbatch --array=0-0 --gpus-per-node=4 --mem=496G --time=02:00:00 h0_measurement/submit_h0_ctx_sweep.slurm qwen3-30b-a3b-2507 131072


# --- reporting ---------------------------------------------------------------
# Report ROUND 2 on its own. Pooling it with ROUND 1 mixes corner configs, which
# report.py now flags as mixed(...) and which makes the corner columns
# non-comparable. RUN_IDs are printed at submission and in each RUN_INFO.txt.
#
# python h0_measurement/report.py \
#        "h0_measurement/results/<R2_8K>/*.parquet" \
#        "h0_measurement/results/<R2_32K>/*.parquet" \
#        "h0_measurement/results/<R2_8K_LARGE>/*.parquet" \
#        "h0_measurement/results/<R2_32K_LARGE>/*.parquet" \
#        "h0_measurement/results/validity<VJOBID>/*.parquet" \
#        -o h0_measurement/reports/h0_corner_round2.pdf


# --- NOT REACHABLE by SIEVE_* today ------------------------------------------
# OPP 2, validation half: corner_kappa / corner_floor are models.yaml keys and
# the scripts only forward SIEVE_CTX / SIEVE_EVICTORS / SIEVE_CORNER_POLICIES
# into --override (submit_h0.slurm:116-118). Adding the knob is 2 lines per
# script, mirroring SIEVE_CORNER_POLICIES:
#     SIEVE_CORNER_KAPPA="${SIEVE_CORNER_KAPPA:-}"
#     [[ -n "$SIEVE_CORNER_KAPPA" ]] && OVERRIDES+=("corner_kappa=$SIEVE_CORNER_KAPPA")
# then, to test the K*-derived kappa=28 against the current 4:
# for K in 4 8 16 28 64; do
#   SIEVE_CTX=32768 SIEVE_EVICTORS='oracle,accum' SIEVE_CORNER_POLICIES='frac,abs' \
#     SIEVE_CORNER_KAPPA=$K sbatch --array=0-3 h0_measurement/submit_h0.slurm
# done
#
# A second INDEPENDENT 128K sample for qwen3-30b. Prompts are seeded on
# prompt_idx, so a resubmit reproduces the run exactly -- job19983220 and
# job19983483 are one sample, not two (19.1% vs 19.3% is numerical noise). A real
# second sample needs --override n_prompts=8 or a different rot_seed, neither of
# which the scripts forward.




# ----- END legacy E1/E2 grid -----------------------------------------------
fi

if [[ "$SECTION" != "--r3" ]]; then exit 0; fi
# =============================================================================
# R3 / E2b -- THE SYMMETRIC CELL.  v3, after the job934606 pilot.
# =============================================================================
#
#     bash h0_measurement/bugs/2_towards_real_evictor/script.sh --r3 --pilot
#     bash h0_measurement/bugs/2_towards_real_evictor/script.sh --r3
#
# OVERRIDES TRAVEL AS ARGUMENTS, NEVER AS ENV PREFIXES.  This cluster exports
# `sbatch` as a shell function -- `type sbatch` shows
#     sbatch () { /opt/slurm/bin/sbatch --export=NONE --get-user-env "$@"; }
# -- and every non-interactive bash inherits it. --export=NONE starts the job
# from the LOGIN environment, so `SIEVE_CTX=8192 sbatch ...` never arrives. That,
# not a split `VAR=x \` continuation (the earlier diagnosis, which was wrong), is
# why job92* logged `ctx=per-model evictors=per-config`, and why the first pilot
# (job934606) ran the per-model 128k default instead of the 8k lag sweep. The
# submit scripts now export any `SIEVE_NAME=value` ARGUMENT (verbatim, commas
# fine), so every line below passes them after the script path:
#     sbatch ... h0_measurement/submit_h0.slurm SIEVE_CTX=8192 llama31-8b
# The first log line must echo them back: `ctx=8192 evictors=oracle,...`.
# If it says `ctx=per-model`, cancel -- the job is not the experiment.
#
# WHY job214* MUST BE RE-RUN, not reanalysed (R3-report.md section 2). The
# first re-run (job214003*, job214064*) used Evictor.score(rank_bump=False) for
# the interior. That removed the ordinal "never evict at birth" bump -- and with
# it the only protection the token appended THIS step had: scoring 0, it got
# 0 bits, i.e. was EVICTED while the current query attends to it. Interior lag
# cost read 5-13x instead of ~1.1-1.5x; every gain_pp*/interior_lag_cost*
# column is invalid. The lag:k corner had the same defect for k >= 2 (the k-1
# tokens newer than the snapshot scored 0). FIXED: Evictor.unseen() marks every
# position a score knows nothing about; the corner bumps it, the interior holds
# it at maxb and water-fills the rest (alloc.waterfill_floor, budget-matched).
# Pinned by test_units.py::test_unseen_floor; validated on real attention by
# R3-fresh-token-test.py (int_fixed == int_floor, cor_lag2_fixed == _prot).
# Fixed runs carry "interior_unseen_policy": "floor_maxb" in their .json; runs
# without it are NOT R3 results. The old bump was harmless (bump/floor = 1.00),
# so job92* stays as a plausible, unconfirmed reference.

cd "${PROJECT_ROOT:-/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant}"

# =============================================================================
# WALLTIME BUDGET
# =============================================================================
# Unit = one (prompt, family) pair: 18 per main-tier job (n_prompts 6 x 3
# families), 12 per large-tier job (n_prompts 4). The log's "N rows  Ts" counter
# starts AFTER weight load + L1/L3, so overhead is added on top: 10 min main,
# 15 min large (measured bound: <= 440 s on job934606, 562 s on llama33-70b).
#
# Rule for every header below:  s/unit x units x 1.25  + overhead, rounded up
# to the next 15 min. The 1.25 covers node-to-node spread (measured <= 4%
# across three identical runs) plus model error on the rates that are not
# directly measured.
#
# THE ONE DIRECT MEASUREMENT OF THE R3 CONFIG. job934606 lost its overrides and
# so ran exactly section B's llama31-8b cell (128k, oracle+accum, frac, interior
# accum): 14 units, steady 222-228 s/unit (first unit 260 incl. L2 check).
#
# Everything else is decomposed from three config generations in the logs:
#   g0 = no corners, g1 = five corners, g2 = five corners + interior accum
#   per-corner cost c = (g1-g0)/5,  interior cost i = g2-g1
#   R3 (two corners + interior) = g0 + 2c + i
#
#   model              ctx    g0     g1     g2    R3 (predicted)    units
#   llama31-8b        128k   107    291    345    235  MEASURED 224    18
#   qwen3-8b           41k    82    256    312    208                  18
#   mistral-7b         32k    72    124    150    119                  18
#   qwen15-moe-a2.7b   32k    29     49     59     47                  18
#   llama33-70b       128k   557   1350   1490   1014                  12
#   qwen3-30b-a3b     128k   165     --    443    443 (no g1 run exists:
#                                                  all of g2-g0 charged to the
#                                                  interior, i.e. no credit)
# The one cell that can be checked, llama31-8b, predicts 235 and measures 224:
# the decomposition is 5% conservative.
#
# ctx scaling, from same-head-count pairs (mistral 32k vs llama31 128k):
# time ~ L^0.3 without corners, ~ L^0.6 with. Scaling DOWN uses 0.3 because it
# predicts the SMALLER saving:  128k->32k and 32k->8k are each x0.66,
# 41k->32k x0.93, 41k->8k x0.61, 128k->8k x0.44.

# --- 0. login node, before any GPU time ---------------------------------------
# The project venv, by path: submit_h0*.slurm activate it INSIDE the job, but
# this runs in the submitting shell, where a bare `python` has no pandas/torch.
# And STOP on failure -- without the `||` a failing test printed its count and
# the script went on to submit every job below anyway.
PY="${SIEVE_VENV:-$PWD/.venv}/bin/python"
[[ -x "$PY" ]] || { echo "no venv python at $PY (set SIEVE_VENV)"; exit 1; }
"$PY" -c "import sys;sys.path.insert(0,'tests');import test_units as T;\
T.test_practical_interior();T.test_corner_provenance();T.test_unseen_floor();\
T.test_rescore_is_idempotent();\
print('fails',T.fails);sys.exit(1 if T.fails else 0)" \
  || { echo "unit tests failed -- not submitting"; exit 1; }

# The staleness-sweep override set, used by C0 and C.
# SPELLING (evict.parse_specs): once any spec carries options, specs are joined
# with ';' -- a ',' string containing '=' is read as ONE spec, which is what
# killed pilots job934932/47/63 ("corner label 'oracle,lag'"). Each lag needs its
# own @alias, because four bare `lag:` specs would all be labelled `lag` while
# SIEVE_INTERIOR_SCORES names lag1..lag8.
LAGS=(SIEVE_N_DECODE=16 SIEVE_QUANT_EVERY=2 SIEVE_CORNER_POLICIES=frac
      'SIEVE_EVICTORS=oracle;lag:k=1@lag1;lag:k=2@lag2;lag:k=4@lag4;lag:k=8@lag8'
      'SIEVE_INTERIOR_SCORES=lag1,lag2,lag4,lag8')
R3=(SIEVE_EVICTORS=oracle,accum SIEVE_CORNER_POLICIES=frac SIEVE_INTERIOR_SCORES=accum)

# ---- re-run guard --------------------------------------------------------------
# fixed MODEL CTX CORNER_TAG -- a COMPLETE result WITH the fresh-token fix exists.
# The corner tag matters: the qwen3-30b 8k R3 run (or-ac_f) and its lag sweep
# (or-la-la-la-la_f) write the SAME filename.
fixed() {
  local p
  for p in h0_measurement/results/job*/h0_"$1"_"$2".parquet; do
    [[ -f "$p" && -f "${p%.parquet}.json" ]] || continue
    [[ "$(tail -c4 "$p" | od -An -c | tr -d ' \n')" == "PAR1" ]] || continue
    grep -q "\"tag\": \"$3\"" "${p%.parquet}.json" || continue
    grep -q '"interior_unseen_policy": "floor_maxb"' "${p%.parquet}.json" || continue
    return 0
  done
  return 1
}
# run_missing TAG "model:ctx ..." <sbatch options> <slurm script> <script args>
# Submits ONE array over the models still missing (appended as positional args,
# which is how submit_h0*.slurm take their model list), or says it skipped.
run_missing() {
  local tag=$1 cells=$2 mc m=(); shift 2
  for mc in $cells; do
    if fixed "${mc%%:*}" "${mc##*:}" "$tag"; then echo "done   ${mc} [$tag]"
    else m+=("${mc%%:*}"); fi
  done
  (( ${#m[@]} )) || return 0
  echo "submit ${m[*]} [$tag]"
  sbatch --array=0-$(( ${#m[@]} - 1 )) "$@" "${m[@]}"
}

# --- C0. THE PILOT.  Submit this ALONE and wait for it. -----------------------
# Section C changes the decode shape and no run has ever used it, so C's headers
# are an upper-bound model, not a measurement. One prompt (3 units) per C model
# gives the real s/unit. A pilot CANCELLED by its time limit is still useful --
# every completed "N rows  Ts" line is a measurement. Size C from it:
#     header = s/unit x 9 units x 1.25 + 10 min (main) / 15 min (large)
# The qwen3-30b pilot is the important one: its C estimate spans 3h-6.5h
# depending on how its g2-g0 splits between corners and interior, which no
# existing run separates.
# DONE -- pilots ran; section C is sized. Not re-submitted.
# if [[ "${2:-}" == "--pilot" ]]; then
#   sbatch --array=0-0 --time=01:00:00 h0_measurement/submit_h0.slurm "${LAGS[@]}" SIEVE_CTX=8192 SIEVE_N_PROMPTS=1 llama31-8b
#   sbatch --array=0-0 --gpus-per-node=4 --time=01:30:00 h0_measurement/submit_h0_large_models.slurm "${LAGS[@]}" SIEVE_CTX=8192 SIEVE_N_PROMPTS=1 qwen3-30b-a3b-2507
#   echo "pilots submitted -- check line 1 of each log shows ctx=8192 evictors=oracle;lag:k=1@lag1;..."
#   exit 0
# fi

# --- A. the matched-ctx grid (was specified, never ran) -----------------------
# Every model at ONE context, so the cross-model column is not confounded with
# ctx. 32k is the largest every main-tier model supports natively. Each array's
# header is set by its slowest model:
#    8k main   qwen3-8b     208 x0.61 = 127 s/u x18 = 38m -> 58m  -> 01:15:00
#   32k main   qwen3-8b     208 x0.93 = 193 s/u x18 = 58m -> 82m  -> 01:30:00
#    8k large  llama33-70b 1014 x0.44 = 441 s/u x12 = 88m -> 125m -> 02:15:00
#   32k large  llama33-70b 1014 x0.66 = 669 s/u x12 =134m -> 182m -> 03:15:00
# (qwen3-30b in the same large arrays needs 39m and 59m.)
# RE-RUN -- job21400319 / 21406448 / 21400321 / 21400322: interior invalid.
run_missing or-ac_f "qwen3-8b:8192 llama31-8b:8192 mistral-7b:8192 qwen15-moe-a2.7b:8192" \
  --time=01:15:00 h0_measurement/submit_h0.slurm "${R3[@]}" SIEVE_CTX=8192
run_missing or-ac_f "qwen3-30b-a3b-2507:8192 llama33-70b:8192" \
  --gpus-per-node=4 --time=02:15:00 h0_measurement/submit_h0_large_models.slurm "${R3[@]}" SIEVE_CTX=8192
run_missing or-ac_f "qwen3-8b:32768 llama31-8b:32768 mistral-7b:32768 qwen15-moe-a2.7b:32768" \
  --time=01:30:00 h0_measurement/submit_h0.slurm "${R3[@]}" SIEVE_CTX=32768
run_missing or-ac_f "qwen3-30b-a3b-2507:32768 llama33-70b:32768" \
  --gpus-per-node=4 --time=03:15:00 h0_measurement/submit_h0_large_models.slurm "${R3[@]}" SIEVE_CTX=32768

# --- B. registry-ctx cells, to replace the contaminated job92* numbers --------
# Same cells job92* produced, so the re-run is directly comparable, plus the two
# it lost (qwen3-8b timed out; llama33-70b never produced a parquet).
#   main   llama31-8b@128k  224 s/u MEASURED x18 = 67m -> 94m -> 01:45:00
#          (qwen3-8b@41k 208 x18 = 62m; mistral 36m; qwen15-moe 14m)
#   large  llama33-70b@128k 1014 x12 = 203m -> 269m -> 04:30:00 by the rule,
#          raised to 05:00:00: the lean rate is predicted, not measured, on
#          the 70B, and a cancellation there throws away 4 GPUs x 4.5h. 05:00
#          still sits under the five-corner g2 rate (298m), so it is not padding.
# RE-RUN -- job21400323 / 21406450: interior invalid. mistral-7b and
# qwen15-moe are dropped here: their registry ctx is 32768, the same cell the
# 32k line above runs. Only the models whose registry ctx differs remain.
run_missing or-ac_f "qwen3-8b:40960 llama31-8b:131072" \
  --time=01:45:00 h0_measurement/submit_h0.slurm "${R3[@]}"
run_missing or-ac_f "qwen3-30b-a3b-2507:131072 llama33-70b:131072" \
  --gpus-per-node=4 --time=05:00:00 h0_measurement/submit_h0_large_models.slurm "${R3[@]}"

# --- C. THE STALENESS SWEEP (never ran; the design-deciding experiment) -------
# `lag:k=N` scores from the attention EXACTLY N steps ago, so cost(k) prices the
# real architectural choice: allocate once at prefill (large k) vs re-budget
# during decode (k=1). The default decode shape (n_decode 8, quant_every 4 ->
# quant steps 0 and 4) can never score lag8, hence the override.
#
# quant_every=2, NOT 1. observe() runs on every decode step whether or not that
# step quantises (run_h0.py's ev.observe loop sits outside the do_quant guard), so the lag ring
# buffer is fed every step and quant_every does not affect lag CORRECTNESS -- it
# only sets how many steps produce a priced row. n_decode=16 quant_every=2 gives
# quant steps 0,2,...,14; lag:k=8 is ready() from step 8, so 8,10,12,14 are four
# valid lag8 measurements. quant_every=1 would have paid for 16 steps to get
# eight. Read interior_lag_cost3_lag{1,2,4,8} PER LAG -- the completeness guard
# blanks the AGGREGATE on early steps by design; per-lag columns stay valid.
#
# UPPER-BOUND headers until C0 reports. Model: quant steps 2 -> 8 (x4), five
# corners (oracle + four lags), four interiors, 9 units (n_prompts 3), and the
# WHOLE per-unit cost -- prefill included -- charged as if it scaled with steps:
#   llama31-8b  8k   (g0+5c+4i) at 8k = 220 x4 =  882 s/u x9 = 132m -> 03:00:00
#   llama31-8b 32k   882 x1.52             = 1337 s/u x9 = 200m -> 04:30:00
#   qwen3-30b   8k   (g0+4i) at 8k = 502 x4 = 2008 s/u x9 = 301m -> 06:30:00
# qwen3-30b is expensive only because its interior share is unknown and taken as
# 100%; if C0 shows ~800 s/u, its header drops to about 03:00:00.
# RE-RUN -- job21400325 / 21400326: lag:k>=2 evicted the k-1 newest tokens
# (corner and interior). qwen3-30b never produced a result (21400327, 21406451).
run_missing or-la-la-la-la_f "llama31-8b:8192" \
  --time=03:00:00 h0_measurement/submit_h0.slurm "${LAGS[@]}" SIEVE_CTX=8192 SIEVE_N_PROMPTS=3
run_missing or-la-la-la-la_f "llama31-8b:32768" \
  --time=04:30:00 h0_measurement/submit_h0.slurm "${LAGS[@]}" SIEVE_CTX=32768 SIEVE_N_PROMPTS=3
run_missing or-la-la-la-la_f "qwen3-30b-a3b-2507:8192" \
  --gpus-per-node=4 --time=06:30:00 h0_measurement/submit_h0_large_models.slurm "${LAGS[@]}" SIEVE_CTX=8192 SIEVE_N_PROMPTS=3

# --- D. verify the overrides ACTUALLY applied, before trusting anything -------
cat <<'CHECK'

  1. Within a minute of each job starting, line 1 of its log must echo the
     overrides:   head -1 h0_measurement/logs/h0_<JOBID>_0.out
       A/B:  ctx=8192|32768|per-model  evictors=oracle,accum
       C:    ctx=8192|32768            evictors=oracle;lag:k=1@lag1;...
     "evictors=per-config" on any of them -> scancel; the job is not R3.

  2. After the first parquet appears:

    .venv/bin/python - <<'PY'
    import pandas as pd, glob
    for f in sorted(glob.glob('h0_measurement/results/job*/*.parquet'))[-4:]:
        d = pd.read_parquet(f)
        lags = sorted({c.split('_lag')[-1] for c in d.columns
                       if 'interior_lag_cost3_lag' in c})
        print(f.split('/')[-2], d.model.iloc[0], 'ctx', d.ctx.iloc[0],
              '| rows at', sorted(d.step.unique()),
              '| quantized', sorted(d.loc[d.quantized, 'step'].unique()),
              '| pp:', [c for c in d.columns if c.startswith('gain_pp3_')],
              '| lags:', lags)
    PY

  Expected for A/B: gain_pp3_accum present, rows at 0..7, quantized [0, 4].
  Expected for C:   four lag columns, rows at 0..15, quantized [0, 2, ..., 14].

  3. THE FIX IS IN (job214* lacked it -- R3-report.md section 2):
       grep -L '"interior_unseen_policy": "floor_maxb"' \
            h0_measurement/results/job<NEW>*/*.json      # must print NOTHING
     and the parquets carry unseen_frac_pp3_<score> (~1/L for accum, ~k/L for
     lag:k). Median interior_lag_cost3_accum should be ~1.1-1.6x, not 5-13x.
  (A dense run emits a row at EVERY step; only the quantized ones carry the
  gain/lag columns.)

CHECK

# --- E. read it, AFTER every job above has finished -----------------------------
# Not sbatch: CPU only, on the login node. R3-figure.py reads the fixed runs by
# the job ids you pass; replace the default glob (it points at job214*, which is
# invalid for every interior column) with the new ones:
#   .venv/bin/python h0_measurement/bugs/2_towards_real_evictor/R3-figure.py \
#       --results "h0_measurement/results/job<NEW_A_B_C>*/*.parquet"
# and re-validate the fix on real attention (CPU, ~1 min):
#   .venv/bin/python h0_measurement/bugs/2_towards_real_evictor/R3-fresh-token-test.py
