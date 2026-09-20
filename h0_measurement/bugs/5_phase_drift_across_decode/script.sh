#!/usr/bin/env bash
# =============================================================================
# R5 -- does a head's PHASE drift across a long decode?   (ROADMAP.md tier 1;
#       defends C4 "one L-free calibration pass", tests C2 inside a generation)
#
#     bash h0_measurement/bugs/5_phase_drift_across_decode/script.sh --run
#
# STATUS 2026-09-20 -- the first campaign ran; its results forced two changes,
# so every drift cell RE-RUNS. --run is guarded and submits only what is missing.
#   job21406669/70/71  llama31-8b 8k/32k/128k sampled   superseded (see below)
#   job21406673        the bridge, dense                KEPT -- corner-only, valid
#   job21406674        llama31-8b 8k greedy             superseded
#   job21424195        qwen3-30b @32k                   superseded
#
# WHAT THE FIRST CAMPAIGN SHOWED, AND WHAT IT COST
#   * THE GENERATIONS LOOPED. Banning EOS keeps decode going, but at 4,096 tokens
#     the text degenerates: distinct-4 fell to 0.05/0.27/0.10 on all three
#     qwen3-30b @32k prompts and to 0.04 on one llama31-8b @8k prompt, and the
#     greedy control looped from step ~512. A loop is a real attention regime but
#     not the one a deployed sampler lives in, and it reads as phase drift. Fixed
#     on BOTH sides: decoding now carries a repetition penalty and an n-gram
#     block (SIEVE_DECODE_REP_PENALTY / SIEVE_DECODE_NO_REPEAT), and drift.py
#     truncates a run at the first step any prompt degenerates.
#     llama31-8b @128k never looped (0.89-0.95) -- a long prompt sustains text.
#   * NOTHING MEASURED THE CLAIM. C4 says the phase is read from ONE offline
#     calibration pass, but the campaign only priced re-budgeting from one step
#     ago (last_step). `lag:k` cannot reach k = 4096 -- it needs k buffers. NEW
#     `first` evictor: the first probed step's attention, frozen, one buffer, so
#     interior_lag_cost3_first at step t IS the price of keeping the prefill-time
#     allocation t steps later, and froz/lag1 is what re-budgeting buys.
#   * qwen3-30b ran at ONE ctx, so the L-growth test (its "double duty") could
#     not run for it: it needs the same model at >= 2 ctx. Section A now adds
#     qwen3-30b @8k.
#
# Want the old cells kept instead? They are superseded by the corner change
# (tag or-la_f -> or-la-fi_f), so the guard re-runs them regardless.
#
# NOTHING RUNS WITHOUT --run (--pilot is retired, see STATUS). Layout follows
# bugs/2_towards_real_evictor/script.sh; bug 2's sections are not repeated here.
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
#      NOTE since `first` joined the evictor list (bug 10): an interior score
#      must BE an evictor, so `first` is also a corner, and
#      `gain_best_practical` is now a min over {last_step, first}. That would
#      quietly strengthen the competitor and move the band against the first
#      campaign and against the bridge, so drift.py recomputes the R5 verdict as
#      min(uniform, last_step corner) -- the same quantity in every campaign.
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
# FOUND AFTER THE FIRST RUN (fixed 2026-09-19):
#   7. The lagged INTERIOR evicted the token appended each step
#      (bugs/2_towards_real_evictor/R3-report.md section 2): with
#      rank_bump=False an unseen position scored 0 and got 0 bits. Fixed by
#      Evictor.unseen() + alloc.waterfill_floor (held at maxb, budget-matched);
#      fixed runs stamp "interior_unseen_policy": "floor_maxb" in the .json.
#      For R5 this touches ONLY the interior columns (interior_lag_cost*,
#      gain_pp*), i.e. drift.py's `lag` column, which drift.py now blanks for
#      pre-fix runs. The last_step CORNER was never affected -- its only unseen
#      token is the current one, which score() always bumped -- so the route,
#      band, tau, dead-tier, flip, regret and bridge columns of job2140666x/7x
#      are valid. The drift cells re-run anyway, for bugs 9 and 10; the bridge
#      does not, because it compares corners only.
#   8. submit_h0_large_models.slurm's host-RAM preflight rebuilt the corner
#      without SIEVE_INTERIOR_SCORES, so with evictors=oracle,last_step it hit the
#      `accum` interior default and died:
#        ValueError: interior_scores ['accum'] are not practical evictors in this
#        run (configured: ['last_step'])
#      That is why job21406672 (qwen3-30b @32k) is empty. Fixed in the preflight
#      (and models.yaml no longer spells the default out).
#
#   9. The decode LOOPED (above). `decode_ban_eos` bought steps and then spent
#      them on repetition. run_h0.next_token now takes `decode_rep_penalty`
#      (the CTRL rule) and `decode_no_repeat_ngram`; both are stamped per row and
#      in the .json. tests/test_units.py::test_anti_loop_decoding pins them.
#  10. No score survived the generation. Added `first` (sievelib/evict.py): the
#      frozen first probed step, persistent across the sparse schedule's block
#      resets (Evictor.persistent, honoured by run_h0), with the positions it
#      never saw marked unseen so the corner bumps them and the interior floors
#      them. tests/test_units.py::test_first_evictor pins it.
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
# The project venv, called by path. submit_h0*.slurm activate it INSIDE the job,
# but this check runs in the submitting shell, where a bare `python` is the
# system one (no pandas/torch). Same override as the slurm scripts.
PY="${SIEVE_VENV:-$PWD/.venv}/bin/python"
[[ -x "$PY" ]] || { echo "no venv python at $PY (set SIEVE_VENV)"; exit 1; }
"$PY" -c "import sys;sys.path.insert(0,'tests');import test_units as T;\
T.test_decode_plan();T.test_override_lists();T.test_rescore_is_idempotent();\
T.test_ban_eos();T.test_practical_interior();T.test_unseen_floor();\
print('fails',T.fails);sys.exit(1 if T.fails else 0)" || { echo "unit tests failed -- not submitting"; exit 1; }

# --- the configuration --------------------------------------------------------
# Measured steps, log-spaced to 4,096: dense where the calibration is read,
# sparse where only the trend matters. last_step needs one warm step before
# each, so 14 measured + 11 warm = 25 probed steps out of 4,097.
MS=0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096
# TWO HORIZONS LIVE ON THIS STEP LIST, and they point at different cells.
#   LENGTH drift needs t to be a large fraction of L -> the 8k cell (4,096
#     generated tokens grow the cache 0.585 octave; at 128k it is 0.045).
#   The FROZEN column needs the opposite: `first` has never seen the t generated
#     tokens, so the interior floors them at maxb and that floor costs maxb*t of
#     the head's B*L bits. At B = 3, maxb = 8 it passes 10% of the budget at
#     t ~ 0.037*L -- step ~300 at 8k, ~1,200 at 32k, ~4,900 at 128k. Past that
#     `froz` prices the floor, not staleness, and drift.py blanks it
#     (--max-floor-share). So the 128k cell is the one that answers C4 over the
#     WHOLE sweep, and the 8k cell answers C2 inside a generation. Both run.
# Sampled, not greedy: greedy decoding falls into repetition loops over
# thousands of tokens, and a loop would read as phase drift. T/top_p are a
# typical deployment setting, the same for every model so rows are comparable.
# `cont` only: niah/qa answer in ~5 tokens and the rest is not the task.
# SIEVE_NO_REPORT=1: the chained report.py medians over steps (bug 5 above);
# the .json sidecar beside each parquet records the effective config instead.
# `first` is the frozen prefill-time score (bug 10) and rides along at one extra
# waterfill + exact_error per (head, budget, measured step). The anti-loop pair
# (bug 9): a mild repetition penalty plus a hard block on repeating any 8-gram,
# which is what actually bounds a loop -- 8-gram repeats are rare in book prose,
# so it binds on loops and almost nowhere else.
R5=(SIEVE_EVICTORS=oracle,last_step,first SIEVE_CORNER_POLICIES=frac
    SIEVE_INTERIOR_SCORES=last_step,first
    SIEVE_MEASURE_STEPS=$MS SIEVE_FAMILIES=cont
    SIEVE_DECODE_TEMPERATURE=0.7 SIEVE_DECODE_TOP_P=0.9 SIEVE_DECODE_SEED=0
    SIEVE_DECODE_BAN_EOS=1 SIEVE_DECODE_REP_PENALTY=1.05 SIEVE_DECODE_NO_REPEAT=8
    SIEVE_NO_REPORT=1)

# ---- re-run guard --------------------------------------------------------------
# done_r5 MODEL CTX CORNER_TAG TEMPERATURE SCHEDULE [NO_REPEAT] -- a complete
# result for this cell exists: PAR1 footer, a .json beside it, and the same
# corner tag, decode_temperature, schedule and (when given) anti-loop setting.
# The temperature matters: A's 8k cell and C's greedy control write the SAME
# filename with the same corner. R5_NEED_INTERIOR=1 also requires the bug-7 fix
# marker, which the bridge (corner-only, still valid) does not need.
done_r5() {
  local p js
  for p in h0_measurement/results/job*/h0_"$1"_"$2".parquet; do
    js="${p%.parquet}.json"
    [[ -f "$p" && -f "$js" ]] || continue
    [[ "$(tail -c4 "$p" | od -An -c | tr -d ' \n')" == "PAR1" ]] || continue
    "$PY" - "$js" "$3" "$4" "$5" "${R5_NEED_INTERIOR:-0}" "${6:-}" <<'PYG' || continue
import json, sys
j = json.load(open(sys.argv[1])); tag, T, sched, need, norep = sys.argv[2:7]
ok = ((j.get("corner") or {}).get("tag") == tag
      and abs(float(j.get("decode_temperature", 0)) - float(T)) < 1e-9
      and j.get("schedule") == sched
      and (need != "1" or j.get("interior_unseen_policy") == "floor_maxb")
      and (not norep or int(j.get("decode_no_repeat_ngram", 0) or 0) == int(norep)))
sys.exit(0 if ok else 1)
PYG
    echo "done   $1 @$2 [$3 T=$4 $5]  ($p)"
    return 0
  done
  return 1
}

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
# The FIRST campaign measured these cells, so the headers below are no longer a
# model: the llama cells finished inside 01:15/01:30/02:30 and qwen3-30b @32k
# inside 03:15. What changed since is the SECOND interior score (`first`): one
# extra waterfill + exact_error per (head, budget) on each of the 14 measured
# steps, on top of a per-unit cost dominated by prefill and the 7-width
# quantize sweep. Measured share of the old rate: the interior was ~1 of 3
# priced passes, so +30% is the honest upper bound -- every header below is the
# old one plus that, rounded up to the next 15 min.
#
#   llama31-8b    8k    810 s/u x1.3 = 1053 x3 = 53m -> 76m  -> 01:30:00
#   llama31-8b   32k   1235 s/u x1.3 = 1606 x3 = 80m -> 110m -> 02:00:00
#   llama31-8b  128k   1978 s/u x1.3 = 2571 x3 =129m -> 171m -> 03:30:00
#   qwen3-30b    32k   2661 s/u x1.3 = 3459 x3 =173m -> 231m -> 04:30:00
#                      (both raised one step over the rule: x1.3 is an estimate,
#                      the first campaign's true elapsed time lives only in that
#                      cluster's logs, and these two ran closest to their old
#                      headers. Read the real rates before trusting the rest:
#                        sacct -j 21406669,21406670,21406671,21406674,21424195 \
#                              -o JobID%18,State,Elapsed,Timelimit
#                      and scale every header if a cell came within ~15 min of
#                      its limit.)
#   qwen3-30b     8k   x0.66 of its 32k rate = 2283 x3 =114m -> 158m -> 03:00:00
#   bridge (B)   8k    unchanged, not re-run                     -> 01:00:00
#   greedy (C)   8k    same as A's 8k cell                       -> 01:30:00
# The anti-loop knobs cost nothing measurable: a set lookup per step.

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
  echo "the pilot is retired: the llama cells ran and measured the schedule"; exit 0
fi
if false; then       # kept for the record
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
# qwen3-30b-a3b-2507 is the second architecture (MoE, 48L, kv=4) and the highest
# dead-tier model in the set, i.e. where C1 says the router matters most. It now
# runs at TWO ctx: the L-growth test needs a within-model slope, and with one ctx
# drift.py could only print "needs one model at >= 2 ctx values". At 8k its
# 4,096 generated tokens grow the cache by 0.585 octave -- the largest LENGTH
# signal available anywhere in this design.
#
# Every cell re-runs: the corner is now oracle,last_step,first (tag or-la-fi_f),
# so the guard does not match the first campaign's or-la_f results.
done_r5 llama31-8b 8192   or-la-fi_f 0.7 sparse 8 || \
sbatch --array=0-0 --time=01:30:00 h0_measurement/submit_h0.slurm "${R5[@]}" SIEVE_CTX=8192   SIEVE_N_PROMPTS=3 llama31-8b
done_r5 llama31-8b 32768  or-la-fi_f 0.7 sparse 8 || \
sbatch --array=0-0 --time=02:00:00 h0_measurement/submit_h0.slurm "${R5[@]}" SIEVE_CTX=32768  SIEVE_N_PROMPTS=3 llama31-8b
done_r5 llama31-8b 131072 or-la-fi_f 0.7 sparse 8 || \
sbatch --array=0-0 --time=03:30:00 h0_measurement/submit_h0.slurm "${R5[@]}" SIEVE_CTX=131072 SIEVE_N_PROMPTS=3 llama31-8b
done_r5 qwen3-30b-a3b-2507 32768 or-la-fi_f 0.7 sparse 8 || \
sbatch --array=0-0 --gpus-per-node=4 --time=04:30:00 h0_measurement/submit_h0_large_models.slurm "${R5[@]}" SIEVE_CTX=32768 SIEVE_N_PROMPTS=3 qwen3-30b-a3b-2507
done_r5 qwen3-30b-a3b-2507 8192  or-la-fi_f 0.7 sparse 8 || \
sbatch --array=0-0 --gpus-per-node=4 --time=03:00:00 h0_measurement/submit_h0_large_models.slurm "${R5[@]}" SIEVE_CTX=8192  SIEVE_N_PROMPTS=3 qwen3-30b-a3b-2507

# =============================================================================
# B. THE BRIDGE -- last_step (what a sparse run can field) vs accum (the
#    campaign's corner), measured on the SAME rows
# =============================================================================
# DENSE, so accum is legal: n_decode 33, quant_every 8 -> quantized steps
# 0, 8, 16, 24, 32, of which 8/16/32 coincide with section A. Both corners and
# both interiors in one run; drift.py prints, per step, the median ratio of the
# two corners' gains and the band fraction under each. If the ratio is flat in
# t, section A's drift is a drift in the phase, not in the corner substitution.
# DONE job21406673, and KEPT: the bridge compares two CORNERS, and neither the
# fresh-token defect nor the loop question touches those columns (its dense
# 33-step decode stops long before the text degenerates -- distinct-4 was 0.92
# at step 32). Its interior columns stay blanked by drift.py.
done_r5 llama31-8b 8192 or-ac-la_f 0.7 dense || \
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
# The first greedy run looped from step ~512 (distinct-4 0.50/0.31/0.43), which
# is exactly what this control exists to show -- but with the anti-loop knobs on
# BOTH arms the comparison is now sampling-vs-greedy rather than
# loop-vs-no-loop, which is the comparison worth having.
done_r5 llama31-8b 8192 or-la-fi_f 0 sparse 8 || \
sbatch --array=0-0 --time=01:30:00 h0_measurement/submit_h0.slurm "${R5[@]}" SIEVE_DECODE_TEMPERATURE=0 SIEVE_CTX=8192 SIEVE_N_PROMPTS=3 llama31-8b
# (the later SIEVE_DECODE_TEMPERATURE=0 wins: arguments are exported in order)

# =============================================================================
# D. VERIFY, then READ
# =============================================================================
cat <<'CHECK'

  1. Line 1 of every log (h0_<JOBID>_0.out / h0large_<JOBID>_0.out):
       A, C:  evictors=oracle,last_step,first measure_steps=0,1,2,4,...,4096
              T=0.7|0, and the schedule line must end
              "EOS banned   rep_penalty=1.05   no_repeat_ngram=8"
       B:     evictors=oracle,accum,last_step measure_steps=dense T=0.7
     and "decode schedule: sparse, 4097 steps, 25 probed, 14 measured" for A/C,
         "decode schedule: dense, 33 steps, 33 probed, 33 measured, 5 quantized"
         for B.

  2. The new columns are there:
       The `first` CORNER columns (gain_e3_first_frac and its w2p twin) are NOT
       a baseline anyone would field: `first` bumps every position its snapshot
       never saw, so past t = B*L/maxb its corner keeps only generated tokens.
       It is an INTERIOR score here -- read interior_lag_cost3_first.

       unseen_frac_pp3_first  ~ t/L at step t (the frozen snapshot never saw the
                                generated tokens); unseen_frac_pp3_last_step ~ 1/L
       interior_lag_cost3_first  the frozen prefill allocation = drift.py's `froz`
     and the .json carries "decode_no_repeat_ngram": 8, "decode_rep_penalty":
     1.05 and "interior_unseen_policy": "floor_maxb".

  3. Read all of it at once -- drift.py groups by (model, ctx, decode_temp,
     schedule, corner) and truncates each run at the first looping step:

     python h0_measurement/bugs/5_phase_drift_across_decode/drift.py \
            "h0_measurement/results/job21406669/*.parquet" \
            "h0_measurement/results/job21406670/*.parquet" \
            "h0_measurement/results/job21406671/*.parquet" \
            "h0_measurement/results/<A_QWEN, the new job>/*.parquet" \
            "h0_measurement/results/job21406674/*.parquet" \
            --csv h0_measurement/reports/r5_drift.csv
     python h0_measurement/bugs/5_phase_drift_across_decode/drift.py \
            "h0_measurement/results/job21406673/*.parquet" --calib 8

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
# AND THE NEW AXIS -- froz/lag1, the price of ONE calibration pass:
#
#   froz/lag1 ~ 1 at every t
#       -> RE-BUDGETING BUYS NOTHING. The prefill-time allocation is as good as
#          one computed from the previous step, so C4's single pass covers the
#          allocation as well as the route.
#
#   froz/lag1 grows with t while froz stays below the interior's edge
#       -> RE-BUDGET ON A SCHEDULE, and the slope says how often. R3's sweep
#          prices k = 1..8 inside a head; `first` extends that axis to k = t.
#          This is the expected outcome and it is the cascade argument.
#
#   froz exceeds the interior's gain over the corner at some t
#       -> A PREFILL-CALIBRATED INTERIOR STOPS PAYING there. Report that t as a
#          deployment horizon, not as a tuning constant.
#
# THE ONE THING THAT WOULD INVALIDATE A: generated text degenerating. drift.py
# now enforces this instead of leaving it to the reader -- it truncates each run
# at the first step ANY prompt falls to distinct-4 <= 0.5 and says where. If a
# cell still truncates early (qwen3-30b @32k stopped at 1,024 in the first
# campaign), the anti-loop knobs were not enough: raise SIEVE_DECODE_REP_PENALTY
# toward 1.15 or lower SIEVE_DECODE_NO_REPEAT toward 6 and re-run that cell --
# do not read the tail.
