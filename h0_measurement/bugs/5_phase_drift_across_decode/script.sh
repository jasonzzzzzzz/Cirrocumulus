#!/usr/bin/env bash
# =============================================================================
# R5 -- does a head's PHASE drift across a long decode?   (ROADMAP.md tier 1;
#       defends C4 "one L-free calibration pass", tests C2 inside a generation)
#
#     bash h0_measurement/bugs/5_phase_drift_across_decode/script.sh --pilot
#     bash h0_measurement/bugs/5_phase_drift_across_decode/script.sh --run
#
# NOTHING RUNS WITHOUT A FLAG. Submit --pilot ALONE, read it (section P), then
# --run. Layout follows bugs/2_towards_real_evictor/script.sh, which this
# replaces for R5; bug 2's sections are not repeated here.
# =============================================================================
#
# THE QUESTION. C4 says a router reads each head's phase from ONE offline
# calibration pass. Nothing has checked whether that phase survives a long
# generation. Two separate things can move it:
#
#   CONTENT   the query distribution changes as the text the model writes moves
#             away from the prompt. Independent of L.
#   LENGTH    tau rises with L in 6/6 models (C2). Generating G tokens grows the
#             cache from L to L+G, so tau should move by slope * log2(1+G/L).
#             That is a within-generation test of C2 -- the across-ctx slope
#             was only ever measured ACROSS prompts of different length.
#
# WHAT THE ROADMAP ENTRY GOT WRONG (fixed there too):
#   * "one config at quant_every=1, n_decode=32 settles it" -- it cannot. 32
#     tokens grow a 128k cache by 0.02% and an 8k cache by 0.4%, so the LENGTH
#     half is unmeasurable at any ctx, and 32 steps is short for CONTENT drift
#     too. It would also quantize all 32 steps, i.e. pay ~16x a campaign unit
#     for a 32-token window.
#   * "provably mis-specified by the end of a long generation" -- not proved.
#     The slope is an across-prompt fact; whether it holds inside one generation
#     is what this run measures. Section A is built to test it.
#   * the dense schedule is no longer the only one. run_h0.py now has a SPARSE
#     schedule (`measure_steps`): decode to the last listed step with the probe
#     off in between, quantize only the listed steps. That is what makes 4,096
#     tokens affordable.
#   * the R3 staleness sweep (bug 2 section C, lag:k=1..8) and R5 are different
#     questions. lag:k prices re-budgeting WITHIN a head over a few steps; R5
#     asks whether the ROUTE (interior vs baseline) and the head's phase hold
#     over thousands. R3-report.md already found lag cost 1.36-1.58x at k=1 on
#     in-band heads, so the allocation inside a head is known to go stale fast;
#     R5 decides whether the cheap per-head ROUTER can still be calibrated once.
#
# BUGS FOUND AND FIXED BEFORE ANY GPU TIME:
#   1. The R5 plumbing in run_h0.py was unreachable. `measure_steps`, `families`,
#      `decode_temperature`, `decode_top_p` and `decode_seed` existed in
#      run_h0.py, but submit_h0*.slurm forwarded none of them, so a sparse run
#      could not be submitted. Now: SIEVE_MEASURE_STEPS / SIEVE_FAMILIES /
#      SIEVE_DECODE_TEMPERATURE / SIEVE_DECODE_TOP_P / SIEVE_DECODE_SEED /
#      SIEVE_DECODE_BAN_EOS, pattern-checked, echoed on log line 1, recorded in
#      RUN_INFO.txt.
#   2. EOS ends the measurement. With no chat template an instruct model closes a
#      continuation within a few hundred tokens; every later row is a post-EOS
#      continuation no deployment runs. `past_eos` only FLAGS those rows. New
#      opt-in `decode_ban_eos` (the min_new_tokens mechanism) keeps the
#      generation going; stamped per row and in the .json sidecar.
#      tests/test_units.py::test_ban_eos pins it.
#   3. The default corner cannot run sparse. `accum` (H2O) sums EVERY step, so
#      decode_plan refuses it under measure_steps (correct: with gaps it would be
#      last_step under another name). R5 therefore runs `last_step` (TOVA) as the
#      practical corner and the interior score. That changes the verdict corner
#      from the campaign's, so section B measures the substitution.
#   4. The interior default silently vanishes. interior_scores defaults to
#      `accum`, and a DEFAULT that names an absent evictor is dropped without an
#      error (evict._parse_interior). Without SIEVE_INTERIOR_SCORES=last_step the
#      run would have had no practical interior at all.
#   5. report.py cannot read this. per_head() takes the median over every row of
#      a head, steps included, so drift is averaged away. drift.py (beside this
#      file) keeps step as an axis; the chained report is switched off below.
#   6. No practical gain exists at step 0: last_step has no history there, and
#      alloc.py withholds the practical columns rather than degrade silently.
#      Calibration is step 1 (drift.py --calib 1); step 0 still carries tau,
#      ladder, n95 and the dead tiers.
#
# SUBMIT FROM trig-login01 (the GPU login node). From a CPU login node Slurm
# rejects every line: "GPU resources requested from a CPU login node". And pass
# NO --mem: Trillium refuses the flag outright ("--mem=... request is not
# allowed"; every job gets the whole node, 745 GiB). The --mem=496G carried by
# bugs/2 and bugs/4 sheets fails submission here -- it is not used below.
#
# OVERRIDES ARE ARGUMENTS. This cluster's `sbatch` is a shell function that adds
# --export=NONE, so `SIEVE_X=... sbatch` never reaches the job. Every SIEVE_*
# below goes AFTER the script path; submit_h0*.slurm export them.
# =============================================================================
case "${1:-}" in
  --pilot|--run) MODE="$1" ;;
  *) echo "usage: bash $0 --pilot | --run"; exit 0 ;;
esac

cd "${PROJECT_ROOT:-/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant}"

# --- 0. login node, before any GPU time ---------------------------------------
python -c "import sys;sys.path.insert(0,'tests');import test_units as T;\
T.test_decode_plan();T.test_override_lists();T.test_rescore_is_idempotent();\
T.test_ban_eos();T.test_practical_interior();\
print('fails',T.fails);sys.exit(1 if T.fails else 0)" || { echo "unit tests failed -- not submitting"; exit 1; }

# --- the configuration --------------------------------------------------------
# Measured steps, log-spaced to 4,096: dense where the calibration is read,
# sparse where only the trend matters. last_step needs one warm step before
# each, so 14 measured + 11 warm = 25 probed steps out of 4,097.
MS=0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096
# Sampled, not greedy: greedy decoding falls into repetition loops over
# thousands of tokens, and a loop would read as phase drift. T/top_p are a
# typical deployment setting, the same for every model so rows are comparable.
# `cont` only: niah/qa answer in ~5 tokens and the rest is not the task.
# SIEVE_NO_REPORT=1: the chained report.py medians over steps (bug 5 above);
# the .json sidecar beside each parquet records the effective config instead.
R5=(SIEVE_EVICTORS=oracle,last_step SIEVE_CORNER_POLICIES=frac
    SIEVE_INTERIOR_SCORES=last_step
    SIEVE_MEASURE_STEPS=$MS SIEVE_FAMILIES=cont
    SIEVE_DECODE_TEMPERATURE=0.7 SIEVE_DECODE_TOP_P=0.9 SIEVE_DECODE_SEED=0
    SIEVE_DECODE_BAN_EOS=1 SIEVE_NO_REPORT=1)

# =============================================================================
# WALLTIME
# =============================================================================
# Unit = one (prompt, family) = one prompt here (cont only). Anchor: the lean
# oracle+accum+interior config MEASURED on llama31-8b at 128k, 224 s/unit
# (job934606), for a dense 8-step decode with 2 quantized steps. Scaled down in
# ctx by L^0.3 (the exponent that predicts the SMALLER saving): 32k x0.66, 8k
# x0.44. qwen3-30b: 443 s/unit at 128k, its five-corner rate (no lean run
# exists for it, so no credit for dropping corners).
#
# Per unit = (anchor x 7) + plain decode. x7 because 14 quantized steps replace
# 2, charged on the WHOLE unit, prefill included, so an upper bound. Plain
# decode is ~4,070 unprobed forwards: 30 ms/token at 8k, 50 ms at 32k, 100 ms
# at 128k on one H100, 150 ms for the MoE on 4 cards (pipeline, not parallel).
# Rule: s/unit x units x 1.25 + 10 min (main) / 15 min (large), rounded up.
#
#   llama31-8b    8k    99x7 = 690 + 120 =  810 s/u x3 = 41m -> 61m  -> 01:15:00
#   llama31-8b   32k   148x7 =1035 + 200 = 1235 s/u x3 = 62m -> 87m  -> 01:30:00
#   llama31-8b  128k   224x7 =1568 + 410 = 1978 s/u x3 = 99m -> 134m -> 02:30:00
#                      (02:15 by the rule, raised one step: the 128k plain-decode
#                      rate is assumed, and a cancellation there loses ~2 h)
#   qwen3-30b    32k   292x7 =2046 + 615 = 2661 s/u x3 =133m -> 181m -> 03:15:00
#   bridge (B)   8k    dense 33 steps, 5 quantized, 3 corners    -> 01:00:00
#   greedy (C)   8k    same as the 8k cell                       -> 01:15:00
# The pilot replaces every figure above with a measurement.

# =============================================================================
# P. THE PILOT.  Submit ALONE; read it before --run.
# =============================================================================
# One prompt per model at the full 4,096-step schedule. A pilot CANCELLED by
# its time limit is still useful -- the log's "N rows  Ts" line is a
# measurement. It answers three things --run depends on:
#   a. s/unit. Re-size each header:  s/unit x 3 x 1.25 + 10m / 15m.
#   b. does the text stay text? `gen_distinct4` at the last step. If the median
#      is < 0.5 the sampler is looping; raise SIEVE_DECODE_TEMPERATURE to 0.9
#      before --run, or the drift will be a loop.
#   c. does the probe hold at step 4,096? L2 capture fidelity runs on the first
#      probed step only; confirm rows exist for step 4096 in the parquet.
if [[ "$MODE" == "--pilot" ]]; then
  sbatch --array=0-0 --time=00:45:00 h0_measurement/submit_h0.slurm "${R5[@]}" SIEVE_CTX=8192 SIEVE_N_PROMPTS=1 llama31-8b
  sbatch --array=0-0 --gpus-per-node=4 --time=01:30:00 h0_measurement/submit_h0_large_models.slurm "${R5[@]}" SIEVE_CTX=32768 SIEVE_N_PROMPTS=1 qwen3-30b-a3b-2507
  cat <<'CHECK'
pilots submitted. Within a minute of each start:
  head -1 h0_measurement/logs/h0_<JOBID>_0.out        (h0large_ for the MoE)
    must read  ctx=8192|32768 evictors=oracle,last_step measure_steps=0,1,2,...
  grep "decode schedule" on the same log
    must read  decode schedule: sparse, 4097 steps, 25 probed, 14 measured,
               14 quantized   sampling T=0.7 top_p=0.9   EOS banned
  "measure_steps=dense" or "evictors=per-config" -> scancel; not the experiment.
Then:
  python h0_measurement/bugs/5_phase_drift_across_decode/drift.py \
         "h0_measurement/results/job<PILOT>/*.parquet"
CHECK
  exit 0
fi

# =============================================================================
# A. THE DRIFT GRID
# =============================================================================
# llama31-8b at three ctx. It is the one main-tier model that reaches 128k
# natively, and the only one with a MEASURED s/unit for the lean config. The
# three ctx separate the two mechanisms: 4,096 generated tokens grow the cache
#     8k -> 12k   (+0.585 octave)   LENGTH drift should be visible
#    32k -> 36k   (+0.170 octave)   LENGTH drift ~ 1/3.5 of the 8k one
#   128k -> 132k  (+0.045 octave)   LENGTH drift ~ 0; anything left is CONTENT
# and the step-1 rows at three ctx give the across-ctx slope that predicts the
# first two (drift.py prints predicted vs measured dtau).
# qwen3-30b-a3b-2507 at 32k is the second architecture (MoE, 48L, kv=4) and the
# highest dead-tier model in the set, i.e. where C1 says the router matters most.
sbatch --array=0-0 --time=01:15:00 h0_measurement/submit_h0.slurm "${R5[@]}" SIEVE_CTX=8192   SIEVE_N_PROMPTS=3 llama31-8b
sbatch --array=0-0 --time=01:30:00 h0_measurement/submit_h0.slurm "${R5[@]}" SIEVE_CTX=32768  SIEVE_N_PROMPTS=3 llama31-8b
sbatch --array=0-0 --time=02:30:00 h0_measurement/submit_h0.slurm "${R5[@]}" SIEVE_CTX=131072 SIEVE_N_PROMPTS=3 llama31-8b
sbatch --array=0-0 --gpus-per-node=4 --time=03:15:00 h0_measurement/submit_h0_large_models.slurm "${R5[@]}" SIEVE_CTX=32768 SIEVE_N_PROMPTS=3 qwen3-30b-a3b-2507

# =============================================================================
# B. THE BRIDGE -- last_step (what a sparse run can field) vs accum (the
#    campaign's corner), measured on the SAME rows
# =============================================================================
# DENSE, so accum is legal: n_decode 33, quant_every 8 -> quantized steps
# 0, 8, 16, 24, 32, of which 8/16/32 coincide with section A. Both corners and
# both interiors in one run; drift.py prints, per step, the median ratio of the
# two corners' gains and the band fraction under each. If the ratio is flat in
# t, section A's drift is a drift in the phase, not in the corner substitution.
sbatch --array=0-0 --time=01:00:00 h0_measurement/submit_h0.slurm \
  SIEVE_EVICTORS=oracle,accum,last_step SIEVE_CORNER_POLICIES=frac \
  SIEVE_INTERIOR_SCORES=accum,last_step SIEVE_N_DECODE=33 SIEVE_QUANT_EVERY=8 \
  SIEVE_FAMILIES=cont SIEVE_DECODE_TEMPERATURE=0.7 SIEVE_DECODE_TOP_P=0.9 \
  SIEVE_DECODE_SEED=0 SIEVE_DECODE_BAN_EOS=1 SIEVE_NO_REPORT=1 \
  SIEVE_CTX=8192 SIEVE_N_PROMPTS=3 llama31-8b

# =============================================================================
# C. THE DECODING CONTROL -- greedy, same cell as A's 8k
# =============================================================================
# Every campaign before R5 decoded greedily. If greedy and sampled drift the
# same way, sampling is not what R5 measures; if greedy drifts more and its
# gen_distinct4 collapses, the difference is loops, and the sampled run is the
# one that describes deployment.
sbatch --array=0-0 --time=01:15:00 h0_measurement/submit_h0.slurm "${R5[@]}" SIEVE_DECODE_TEMPERATURE=0 SIEVE_CTX=8192 SIEVE_N_PROMPTS=3 llama31-8b
# (the later SIEVE_DECODE_TEMPERATURE=0 wins: arguments are exported in order)

# =============================================================================
# D. VERIFY, then READ
# =============================================================================
cat <<'CHECK'

  1. Line 1 of every log (h0_<JOBID>_0.out / h0large_<JOBID>_0.out):
       A, C:  evictors=oracle,last_step measure_steps=0,1,2,4,...,4096 T=0.7|0
       B:     evictors=oracle,accum,last_step measure_steps=dense T=0.7
     and "decode schedule: sparse, 4097 steps, 25 probed, 14 measured" for A/C,
         "decode schedule: dense, 33 steps, 33 probed, 33 measured, 5 quantized"
         for B.

  2. Read all of it at once -- drift.py groups by (model, ctx, decode_temp):

     python h0_measurement/bugs/5_phase_drift_across_decode/drift.py \
            "h0_measurement/results/<A_8K>/*.parquet"  \
            "h0_measurement/results/<A_32K>/*.parquet" \
            "h0_measurement/results/<A_128K>/*.parquet" \
            "h0_measurement/results/<A_QWEN>/*.parquet" \
            "h0_measurement/results/<C_GREEDY>/*.parquet" \
            --csv h0_measurement/reports/r5_drift.csv
     python h0_measurement/bugs/5_phase_drift_across_decode/drift.py \
            "h0_measurement/results/<B_BRIDGE>/*.parquet" --calib 8

     The bridge reads with --calib 8: in a dense run step 1 is not quantized.

CHECK

# =============================================================================
# DECISION TABLE, written before the run
# =============================================================================
# Read flip_rtr (the route an offline router would assign) against its noise
# floor (calib vs the next measured step; per-head values carry up to 80% GPU
# nondeterminism, R3-report.md finding 5), and regret90 (what keeping the
# calibration route costs at step t).
#
#   flip_rtr within ~5 pts of the floor through 4,096 AND regret90 < 1.1
#       -> ONE-PASS CALIBRATION HOLDS. C4 gets a number instead of an
#          assumption. The router is offline; re-budgeting (R3 lag cost) is a
#          WITHIN-head concern and the router does not need to move with it.
#
#   flip_rtr grows steadily with t, at 128k as well as at 8k
#       -> CONTENT DRIFT. The phase moves with what is being written, not with
#          L. The router needs periodic re-calibration; the slope of flip_rtr
#          in log(t) says how often. This turns C4 from "one pass" into "one
#          pass per N tokens", which is still cheap and is a method, not a map.
#
#   flip_rtr grows at 8k, less at 32k, ~not at 128k, and drift.py's measured
#   dtau matches the predicted one
#       -> LENGTH DRIFT, i.e. C2 holds INSIDE a generation. The mis-specification
#          is real and predictable from the calibration pass itself: the router
#          can pre-compute the route for L+G from the slope. Strongest outcome
#          for the paper -- C2 becomes actionable.
#
#   measured dtau ~ 0 at 8k where predicted dtau is clearly not
#       -> the across-ctx slope is a property of PROMPTS (different documents
#          at different lengths), not of cache length. C2 must be restated as
#          a statement about the context a model is given, not about how long
#          it has been running.
#
#   bridge ratio drifts with t
#       -> section A's drift is partly the last_step/accum substitution. Report
#          A against the bridge-corrected band, and say so.
#
# THE ONE THING THAT WOULD INVALIDATE A: generated text degenerating. If the
# median gen_distinct4 at step 4,096 is below ~0.5 in the sampled runs, the late
# rows measure a loop and are excluded from the verdict, whatever they show.
