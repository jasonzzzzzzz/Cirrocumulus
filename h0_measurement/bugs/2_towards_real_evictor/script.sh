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


# =============================================================================
# R3 / E2b -- THE SYMMETRIC CELL.  (ROADMAP.md tier 1, highest-value item left)
# =============================================================================
#
# WHAT IS BEING FIXED. E2 demoted the eviction CORNER from an oracle to lagged
# attention, and every STOP verdict disappeared. But the INTERIOR still
# water-fills on w2 = (a*||v-o||)^2, where `a` is the CURRENT step's attention --
# exactly the oracle information E2 took away from the corner. So the published
# comparison still gives our own method information the baseline is denied, which
# is the same criticism E2 levelled at the old corner.
#
# WHAT THE CODE NOW DOES (alloc.py, CornerSpec.interior_scores):
#   ap  = normalise(lagged score)          # H2O's running attention sum
#   op  = ap @ V                           # lagged output, from the KV cache only
#   w2p = (ap * ||v - op||)^2              # lagged sensitivity
#   bwp = waterfill(w2p, ...)              # the DECISION is lagged
#   err = exact_error(s, shat, V, bwp)     # the EVALUATION stays exact
# Nothing in w2p touches the current query. That is the asymmetry a deployed
# system actually faces: decide from history, get judged on the truth.
#
# NEW COLUMNS      err_wf_pp<B>_<score>          lagged-interior error
#                  interior_lag_cost<B>_<score>  err_wf_pp / err_wf  (>= 1)
#                  gain_pp<B>_<score>            SYMMETRIC gain, in_band_pp<B>_*
#                  gain_pp_sym<B>_<score>        + corner also ranked by w2p
#                  err_e<B>_<score>_w2p_<policy> the w2p-ranked corner
# report.py prints these as the "SYMMETRIC CELL @3b" block.
#
# COST. One extra waterfill + one exact_error per (head, budget, interior score).
# Measured base is ~47 ms/head/4-budgets (see the timing table at the top of this
# file) and exact_error is one of its three dominant terms, so ONE interior score
# is roughly +20-25% task time. Default is `accum` alone, not all four evictors:
# accum (H2O) wins 74-93% of corners, so it is the score a deployable system
# would carry, and each extra entry multiplies the added cost.
#
# WHY NOT REUSE the round-2 parquets: `gain_pp*` did not exist when they were
# written. This needs a rerun. Budget and walltime are unchanged from round 2
# apart from the +25%, which the existing --time already absorbs at 8k/32k.

# --- 0. login node: confirm the plumbing before burning an allocation --------
# python -c "import sys;sys.path.insert(0,'tests');import test_units as T;\
#            T.test_practical_interior();print('fails',T.fails)"
# python -c "import sys;sys.path.insert(0,'.');sys.path.insert(0,'h0_measurement');\
#            from run_h0 import load_cfg; from sievelib.evict import CornerSpec;\
#            print(CornerSpec.from_cfg(load_cfg('h0_measurement/models.yaml','llama31-8b',\
#            ['interior_scores=accum'])).interior_scores)"

# --- 1. the matched-ctx grid, 8k and 32k, main tier --------------------------
# `oracle,accum` only: the symmetric cell needs accum (the interior score) and
# oracle (the bound every legacy column is defined against). Dropping window and
# recency pays for the interior's extra exact_error, and bugs/2/report.md 6.4
# already showed they win 6-22% and 1-12% of heads for 18 of the 23 bytes of
# host state -- they are not worth carrying into this campaign.
SIEVE_CTX=8192  SIEVE_EVICTORS='oracle,accum' SIEVE_CORNER_POLICIES='frac' \
  SIEVE_INTERIOR_SCORES='accum' sbatch --array=0-3 h0_measurement/submit_h0.slurm
SIEVE_CTX=32768 SIEVE_EVICTORS='oracle,accum' SIEVE_CORNER_POLICIES='frac' \
  SIEVE_INTERIOR_SCORES='accum' sbatch --array=0-3 h0_measurement/submit_h0.slurm

# --- 2. same, large tier (4 GPUs; --mem stays 320G) --------------------------
SIEVE_CTX=8192  SIEVE_EVICTORS='oracle,accum' SIEVE_CORNER_POLICIES='frac' \
  SIEVE_INTERIOR_SCORES='accum' sbatch --array=0-1 h0_measurement/submit_h0_large_models.slurm
SIEVE_CTX=32768 SIEVE_EVICTORS='oracle,accum' SIEVE_CORNER_POLICIES='frac' \
  SIEVE_INTERIOR_SCORES='accum' sbatch --array=0-1 h0_measurement/submit_h0_large_models.slurm

# --- 3. the ctx axis, so the symmetric cell has a slope not just a level -----
# The central claim is that the band decays with context. If the symmetric cell
# is only measured at 8k/32k it cannot say whether demoting the interior changes
# that SLOPE -- which is the question, since the corner's oracle advantage was
# itself largest at long ctx. One sweep per swept model.
SIEVE_EVICTORS='oracle,accum' SIEVE_CORNER_POLICIES='frac' \
  SIEVE_INTERIOR_SCORES='accum' \
  sbatch --array=0-4 h0_measurement/submit_h0_ctx_sweep.slurm llama31-8b 8192 16384 32768 65536 131072
SIEVE_EVICTORS='oracle,accum' SIEVE_CORNER_POLICIES='frac' \
  SIEVE_INTERIOR_SCORES='accum' \
  sbatch --array=0-4 --mem=496G --gpus-per-node=4 --time=10:00:00 \
  h0_measurement/submit_h0_ctx_sweep.slurm qwen3-30b-a3b-2507 8192 16384 32768 65536 131072

# --- 4. read it --------------------------------------------------------------
# python h0_measurement/report.py \
#        "h0_measurement/results/<R3_8K>/*.parquet" \
#        "h0_measurement/results/<R3_32K>/*.parquet" \
#        "h0_measurement/results/<R3_8K_LARGE>/*.parquet" \
#        "h0_measurement/results/<R3_32K_LARGE>/*.parquet" \
#        "h0_measurement/results/<R3_SWEEP_8B>/*.parquet" \
#        "h0_measurement/results/<R3_SWEEP_30B>/*.parquet" \
#        -o h0_measurement/reports/h0_symmetric_cell.pdf
#
# WHAT TO LOOK FOR, and what each outcome means for the paper:
#
#   interior_lag_cost3_accum ~ 1.0-1.3x
#       The allocator barely needs the current query. Strongest possible result:
#       the interior's edge is about mixed-precision SHAPE, not privileged
#       information, and the GO verdicts stand with no asymmetry left to attack.
#
#   in_band_pp3_accum still >= 35% on the models that were GO
#       The symmetric cell keeps the verdict. This is the headline to write.
#
#   in_band_pp3 collapses (interior_lag_cost >> 1.5x)
#       Allocation genuinely needs current-query information. NOT a dead end --
#       it is the measured motivation for the cascade's cheap first pass, and it
#       converts the router claim from "score per head offline" into "re-budget
#       during decode". Report it as such; do not bury it.
#
#   gain_pp_sym3 (corner ALSO ranked by w2p) much lower than gain_pp3
#       The corner was being handicapped by ranking on raw attention rather than
#       on sensitivity. That is a finding about every H2O-style evictor, not
#       about us, and it belongs in the paper either way.
#
# EXPECTED DIRECTION, stated in advance so it cannot be rationalised after:
# gain_pp3 <= gain_best_practical3 per head, always -- the lagged allocator
# chooses from strictly less information and both are scored by the same exact
# recomputation. tests/test_units.py::test_practical_interior asserts it. Unlike
# the corner monotonicity claim (which this project already got wrong once, see
# plan.md "provably monotone"), this direction IS a per-head guarantee, because
# here it is the ALLOCATION that is being restricted, not the ranking.
