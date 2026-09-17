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
# R3 / E2b -- THE SYMMETRIC CELL, and the staleness curve that decides the design
# =============================================================================
#
# WHAT E2 LEFT ASYMMETRIC. E2 demoted the eviction CORNER from an oracle to
# lagged attention, and every STOP verdict disappeared. The INTERIOR still
# water-fills on w2 = (a*||v-o||)^2 with `a` the CURRENT step's attention --
# exactly the oracle information E2 took away from the corner.
#
# WHY THE FIRST VERSION OF THIS EXPERIMENT ASKED THE WRONG QUESTION.
# On a synthetic head the 1-step-lagged allocator costs 1.19x the error and the
# gain falls 2.11x -> 1.77x, i.e. across the 2x band line. Read naively that is
# "the interior needs clairvoyance". But eviction and allocation are NOT
# symmetric decisions, and pretending they are mis-states the design:
#
#   eviction    is IRREVERSIBLE and must be decided WITHOUT the query -- the
#               whole point is to not store the token. Lagged is intrinsic.
#   allocation  assigns bits to tokens already stored. It is decided ONCE when
#               the cache is written and then read by every later query.
#
# So the interior's real handicap is not "one step of lag". It is STALENESS: the
# allocation is computed at some step and then serves queries arriving 1, 10,
# 1000 steps later. A system that allocates at prefill and decodes 2,000 tokens
# is operating at lag ~2000, not lag 1. Measuring only k=1 prices the cheapest
# possible re-budgeter and says nothing about the architecture anyone would ship.
#
# WHAT THE DESIGNER ACTUALLY NEEDS, and what this campaign now measures:
#
#   cost(k) = err(allocate on step t-k) / err(allocate on step t),  k = 1,2,4,8
#
#   cost(k) FLAT in k          -> allocate ONCE at prefill, never re-budget.
#                                 Cheapest architecture, no per-step pass, and
#                                 the 1.19x is a fixed one-off toll, not a slope.
#   cost(k) RISING in k        -> re-budgeting during decode is load-bearing, and
#                                 the slope says how often. That is a real cost
#                                 to own -- and it is also the one thing a
#                                 prefill-only allocator structurally cannot do.
#
# Either answer is a design result. Only the second is bad news, and even then it
# converts the router claim from "score once offline" into "re-budget every N
# steps", with N measured rather than asserted.
#
# `lag:k=N` (sievelib/evict.py) is the probe: attention from EXACTLY N steps ago.
# k=1 reduces to last_step by construction (asserted in the unit test).
#
# SECOND DESIGNER-FACING NUMBER: ROUTER MISCALIBRATION. A router whose threshold
# was calibrated on the ORACLE gain routes a head to the interior whenever
# gain_best >= 2x. If that head's HONEST gain is < 2x it will not pay for itself
# once deployed. report.py now prints that rate. It is the practical cost of
# calibrating on the wrong column, and it is actionable today: recalibrate the
# threshold on gain_pp, not gain_best.
#
# NEW COLUMNS   err_wf_pp<B>_<score>          lagged-interior error
#               interior_lag_cost<B>_<score>  err_wf_pp / err_wf  (>= 1)
#               gain_pp<B>_<score>            SYMMETRIC gain, + in_band_pp
#               gain_pp_sym<B>_<score>        + corner also ranked by w2p
#               err_e<B>_<score>_w2p_<policy> the w2p-ranked corner
# report.py prints "SYMMETRIC CELL @3b", "ROUTER MISCALIBRATION" and
# "STALENESS OF THE ALLOCATION".

# --- 0. login node: confirm the plumbing before burning an allocation --------
# python -c "import sys;sys.path.insert(0,'tests');import test_units as T;\
#            T.test_practical_interior();print('fails',T.fails)"

# --- A. the symmetric cell on the main grid ----------------------------------
# Does the verdict survive when the interior is demoted too? `oracle,accum` only:
# accum is the interior score, oracle is the bound every legacy column is defined
# against. window/recency win 6-22% and 1-12% of heads for 18 of the 23 bytes of
# host state (report.md 6.4) -- they are not worth carrying here, and dropping
# them pays for the interior's extra exact_error.
SIEVE_CTX=8192  SIEVE_EVICTORS='oracle,accum' SIEVE_CORNER_POLICIES='frac' \
  SIEVE_INTERIOR_SCORES='accum' sbatch --array=0-3 h0_measurement/submit_h0.slurm
SIEVE_CTX=32768 SIEVE_EVICTORS='oracle,accum' SIEVE_CORNER_POLICIES='frac' \
  SIEVE_INTERIOR_SCORES='accum' sbatch --array=0-3 h0_measurement/submit_h0.slurm

SIEVE_CTX=8192  SIEVE_EVICTORS='oracle,accum' SIEVE_CORNER_POLICIES='frac' \
  SIEVE_INTERIOR_SCORES='accum' sbatch --array=0-1 h0_measurement/submit_h0_large_models.slurm
SIEVE_CTX=32768 SIEVE_EVICTORS='oracle,accum' SIEVE_CORNER_POLICIES='frac' \
  SIEVE_INTERIOR_SCORES='accum' sbatch --array=0-1 h0_measurement/submit_h0_large_models.slurm

# --- B. THE STALENESS SWEEP -- the design-deciding run -----------------------
# quant_every=1 and n_decode=16 because a lag of k only scores from step k
# onward: with the default (n_decode 8, quant_every 4) the only quant steps are
# 0 and 4, so lag8 never scores at all and lag4 scores once. 16 steps quantized
# every step gives lag8 eight usable measurements per prompt.
#
# n_prompts=3 pays for that: the cost is (quant steps) x (heads), and this moves
# quant steps 2 -> 16, so trimming prompts 6 -> 3 leaves it ~4x the default task
# rather than ~8x. The staleness curve is a within-head comparison across k, so
# prompt count buys much less here than step count does.
#
# Read the per-lag columns (interior_lag_cost3_lag*), NOT gain_best_practical3:
# each lag scores from step k, so the completeness guard blanks the AGGREGATE on
# early steps by design while the per-lag columns stay valid throughout.
#
# Two models, chosen to bracket the phase diagram: llama31-8b traverses the
# diagram on ctx (dead-2 19->54%), qwen3-30b is pinned in the evict corner at
# every length (60-73%). If staleness behaves the same in both, it is a property
# of attention; if it tracks dead-2, it is another face of the phase variable.
SIEVE_CTX=8192 SIEVE_N_DECODE=16 SIEVE_QUANT_EVERY=1 SIEVE_N_PROMPTS=3 \
  SIEVE_EVICTORS='oracle,lag:k=1,lag:k=2,lag:k=4,lag:k=8' \
  SIEVE_CORNER_POLICIES='frac' \
  SIEVE_INTERIOR_SCORES='lag1,lag2,lag4,lag8' \
  sbatch --array=0-0 --time=02:30:00 h0_measurement/submit_h0.slurm llama31-8b

SIEVE_CTX=32768 SIEVE_N_DECODE=16 SIEVE_QUANT_EVERY=1 SIEVE_N_PROMPTS=3 \
  SIEVE_EVICTORS='oracle,lag:k=1,lag:k=2,lag:k=4,lag:k=8' \
  SIEVE_CORNER_POLICIES='frac' \
  SIEVE_INTERIOR_SCORES='lag1,lag2,lag4,lag8' \
  sbatch --array=0-0 --time=03:30:00 h0_measurement/submit_h0.slurm llama31-8b

SIEVE_CTX=8192 SIEVE_N_DECODE=16 SIEVE_QUANT_EVERY=1 SIEVE_N_PROMPTS=3 \
  SIEVE_EVICTORS='oracle,lag:k=1,lag:k=2,lag:k=4,lag:k=8' \
  SIEVE_CORNER_POLICIES='frac' \
  SIEVE_INTERIOR_SCORES='lag1,lag2,lag4,lag8' \
  sbatch --array=0-0 --time=04:00:00 --mem=496G --gpus-per-node=4 \
  h0_measurement/submit_h0_large_models.slurm qwen3-30b-a3b-2507

# The decode shape comes from SIEVE_N_DECODE / SIEVE_QUANT_EVERY / SIEVE_N_PROMPTS,
# NOT from trailing --override args: the scripts do `MODELS=("$@")`, so anything
# after the model name is parsed as another MODEL NAME and the job dies on an
# unknown tag (or worse, silently runs the wrong set). Those four knobs were added
# for this campaign and they also close the gap report.md 7 flagged for R7 --
# SIEVE_ROT_SEED is what makes a second independent sample of a cell possible.

# --- C. read it --------------------------------------------------------------
# python h0_measurement/report.py \
#        "h0_measurement/results/<R3_A_*>/*.parquet" \
#        "h0_measurement/results/<R3_B_*>/*.parquet" \
#        -o h0_measurement/reports/h0_symmetric_cell.pdf
#
# DECISION TABLE -- written before the run so it cannot be rationalised after.
#
#   A: in_band_pp3 >= 35% where the oracle interior was GO
#      The verdict survives a fully symmetric comparison. Strongest result the
#      framework can produce; no reviewer can attribute it to asymmetry.
#
#   A: in_band_pp3 collapses but interior_lag_cost3 ~ 1.0-1.2x
#      The interior is fine; the BAND THRESHOLD was being carried by the oracle.
#      Report the error ratios (robust) rather than the thresholded count -- the
#      same move R2 already made for the phase panel -- and recalibrate the
#      router on gain_pp. Not a method failure, a statistic failure.
#
#   B: cost(k) flat  (cost(8)/cost(1) < 1.10)
#      ALLOCATE ONCE AT PREFILL. No per-step pass, no re-budgeting machinery,
#      and the 1.19x becomes a fixed toll rather than a growing one. This is the
#      cheap architecture and it is the outcome to hope for.
#
#   B: cost(k) rising
#      RE-BUDGET DURING DECODE, every ~N steps where N is read off the curve.
#      This is a real cost, and it is also the one thing a prefill-only allocator
#      cannot do -- so it becomes the architectural differentiator rather than
#      an embarrassment. It also makes R5 (phase drift) mandatory rather than
#      nice-to-have, since the two measure the same clock.
#
#   B: cost(k) tracks dead-2 across the two models
#      Staleness is another face of the phase variable, which would extend C1
#      rather than complicate it. Check this before writing either story.
#
# EXPECTED DIRECTION, stated in advance: gain_pp3 <= gain_best_practical3 per
# head, always, and cost(k) >= 1. The lagged allocator chooses from strictly less
# information and both are scored by the same exact recomputation.
# tests/test_units.py::test_practical_interior asserts both. Unlike the corner
# monotonicity claim -- which this project already got wrong once (plan.md,
# "provably monotone") -- this direction IS a per-head guarantee, because here it
# is the ALLOCATION being restricted, not the ranking.
