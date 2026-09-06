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
