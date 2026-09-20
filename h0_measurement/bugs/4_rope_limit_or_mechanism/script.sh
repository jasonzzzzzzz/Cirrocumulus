# =============================================================================
# R4 -- is the 64k->128k collapse a property of ATTENTION, or of running models
#       at their TRAINED LIMIT?   (ROADMAP.md tier 1; defends C1 and C2)
# =============================================================================
#
# THE OBSERVATION. Every model that reaches 128k drops sharply there:
#   llama33-70b  61.9 -> 25.0 band  (-36.9, the steepest move in the study)
#   llama31-8b   34.2 -> 28.8       (-5.4)
#   qwen3-30b    24.3 -> 18.6       (-5.7)
# with tau and n95 both jumping (70B: tau 2.06 -> 3.31, n95 117 -> 4,854).
#
# THE CONFOUND, read off the registry rather than guessed:
#
#   model                 default ctx   RoPE window   128k is...
#   llama31-8b              131,072       131,072     100% of the window
#   llama33-70b             131,072       131,072     100% of the window
#   qwen3-30b-a3b-2507      131,072       262,144      50% of the window
#
# The two models AT their cap show the drops. The one with 2x headroom shows the
# smallest. So "the band collapses past 64k" and "the band collapses as a model
# approaches its trained limit" fit the existing data EQUALLY WELL, and they are
# not the same claim:
#
#   ABSOLUTE-L   the decay continues past 128k. C2's slope is a property of
#                attention, the method's reach genuinely ends near 64k, and
#                "ways this dies #1" is real.
#   ROPE-FRAC    part of the steep slope is an artifact of measuring models at
#                their limit. C2's slope must be restated, the 128k row stops
#                being evidence about long context, and the honest claim becomes
#                narrower but cleaner.
#
# WHY THIS IS CHEAP TO SETTLE. qwen3-30b-a3b-2507 is the only model in the
# registry with real headroom, and it has exactly 2x. Pushing it to 192k/256k
# moves rope_frac 0.50 -> 0.75 -> 1.00 while absolute L moves 128k -> 256k. The
# two hypotheses then make OPPOSITE within-model predictions:
#
#   ABSOLUTE-L   qwen3-30b's sharpest drop already happened (64k->128k) and
#                192k/256k continue the same gentle slope.
#   ROPE-FRAC    qwen3-30b's sharpest drop is STILL AHEAD, at 256k, where it
#                finally reaches the limit llama was at when it fell.
#
# A 96k point on the llama pair brackets their step from below (rope_frac 0.73)
# so their curve has a shape rather than two endpoints.
#
# WHAT WAS FIXED FOR THIS (all landed, no GPU needed):
#   models.yaml          `native_ctx` per model -- the README ctx audit made
#                        machine-readable, verified against every live config.
#   submit scripts       the ctx guard now caps SIEVE_CTX at native_ctx, not at
#                        the default ctx. It used to refuse 192k for
#                        qwen3-30b-a3b-2507 on the stated grounds that "the
#                        registry values are native RoPE limits" -- true for six
#                        of eight models and FALSE for the only one this needs.
#                        Past the WINDOW is still refused (untrained positions);
#                        past the DEFAULT now prints a headroom note.
#   run_h0.py            stamps `native_ctx`, `rope_frac = ctx/native_ctx` and
#                        `rope_type` per row, read from the LIVE config and
#                        cross-checked against models.yaml with a warning on
#                        drift. rope_frac is the independent variable; without it
#                        in the parquet the question cannot be asked afterwards.
#   report.py            `page_rope` plots each model's curve against BOTH axes
#                        and prints where each model's own steepest per-octave
#                        drop falls.

# --- 0. login node: confirm the guard lets the headroom through --------------
# python - <<'PY'
# import yaml
# c=yaml.safe_load(open('h0_measurement/models.yaml'))
# for t in ('qwen3-30b-a3b-2507','llama31-8b'):
#     m=next(x for x in c['models'] if x['tag']==t)
#     print(t, 'default', m['ctx'], 'window', m['native_ctx'])
# PY

# --- A. the decisive run: qwen3-30b-a3b-2507 into its headroom ---------------
# 192k = 75% of the window, 256k = 100%. If the collapse tracks the window, the
# big drop lands at 256k; if it tracks absolute L, it already happened at 128k
# and these two continue the existing gentle slope.
#
# COST. KV cache scales linearly and attention work superlinearly: at 256k the
# cache is ~26 GB (vs 13 at 128k) against 61 GB of weights, so 4 GPUs still hold
# it, but prefill roughly doubles per octave and the per-head exact_error work
# grows with L. n_prompts=3 and the minimal corner set pay for that.
#
# WALLTIME. Rates from the table in bugs/2_towards_real_evictor/script.sh; the
# lean oracle+accum config was MEASURED on llama31-8b at 128k (job934606: 224
# s/unit). Rule: s/unit x units x 1.25 + 10m (main) / 15m (large), and an extra
# x1.5 past 131k, where no run has ever been. Scaling UP uses L^0.6 (the larger
# exponent measured), scaling DOWN L^0.3 -- the pessimistic one in each direction.
#   qwen3-30b 196k  443 x1.27 = 563 s/u x9 = 84m  -> x1.875 +15 = 173m -> 03:00:00
#   qwen3-30b 256k  443 x1.52 = 673 s/u x6 = 67m  -> x1.875 +15 = 141m -> 02:30:00
#   llama31-8b 96k  224 x0.92 = 205 s/u x9 = 31m  -> x1.25  +10 =  49m -> 01:00:00
#   llama33-70b 96k 1014 x0.92= 930 s/u x9 = 140m -> x1.25  +15 = 190m -> 03:15:00
#   qwen3-1.7b <=41k  (28x16 heads, below qwen15-moe's 47 s/u at 32k) -> 01:00:00
# (443 for qwen3-30b is its five-corner+interior rate: no lean-config or
# corner-only run exists for it, so it takes no credit for dropping corners.)
#
# OVERRIDES ARE ARGUMENTS. This cluster's `sbatch` is a shell function that adds
# --export=NONE, so an env prefix (`SIEVE_CTX=... sbatch`) never reaches the job;
# submit_h0*.slurm export any SIEVE_NAME=value argument instead. Line 1 of each
# log must read `ctx=<the value>` -- `ctx=per-model` means cancel.
cd "${PROJECT_ROOT:-/scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant}"
R3=(SIEVE_EVICTORS=oracle,accum SIEVE_CORNER_POLICIES=frac SIEVE_INTERIOR_SCORES=accum)

sbatch --array=0-0 --gpus-per-node=4 --time=03:00:00 h0_measurement/submit_h0_large_models.slurm "${R3[@]}" SIEVE_CTX=196608 SIEVE_N_PROMPTS=3 qwen3-30b-a3b-2507
sbatch --array=0-0 --gpus-per-node=4 --time=02:30:00 h0_measurement/submit_h0_large_models.slurm "${R3[@]}" SIEVE_CTX=262144 SIEVE_N_PROMPTS=2 qwen3-30b-a3b-2507

# --- B. bracket the llama step from below ------------------------------------
# 96k = 73% of a 131,072 window. Both llama models are AT their cap, so 96k is
# the last point below it and the only way to give their 64k->128k step a shape.
sbatch --array=0-0 --time=01:00:00 h0_measurement/submit_h0.slurm "${R3[@]}" SIEVE_CTX=98304 SIEVE_N_PROMPTS=3 llama31-8b
sbatch --array=0-0 --gpus-per-node=4 --time=03:15:00 h0_measurement/submit_h0_large_models.slurm "${R3[@]}" SIEVE_CTX=98304 SIEVE_N_PROMPTS=3 llama33-70b

# NOTE on the qwen3-1.7b control below: its tokenizer files were missing from
# .hf_cache the last time this was checked, and run_h0 fails fast on that with a
# clear message. Re-stage first if it trips:
#   python h0_measurement/prefetch.py -m qwen3-1.7b

# --- C. the debug-tier control, nearly free ----------------------------------
# qwen3-1.7b defaults to ctx 8192 against a 40,960 window -- 5x headroom, the
# most in the registry, on a model small enough to sweep the whole range in one
# short job. If rope_frac is the real axis, this model should look FLAT from 8k
# to 32k (rope_frac 0.20 -> 0.80) and only fall approaching 40k. It is a tier
# `debug` model, so treat it as a mechanism check, not as a headline point.
for CX in 8192 16384 32768 40960; do
  sbatch --array=0-0 --time=01:00:00 h0_measurement/submit_h0.slurm "${R3[@]}" SIEVE_CTX=$CX SIEVE_N_PROMPTS=3 qwen3-1.7b
done

# --- D. read it --------------------------------------------------------------
# Pool the NEW points with the existing sweep so each model has a full curve:
# python h0_measurement/report.py \
#        "h0_measurement/results/job20014005/*.parquet" \
#        "h0_measurement/results/<R4_QWEN_192K>/*.parquet" \
#        "h0_measurement/results/<R4_QWEN_256K>/*.parquet" \
#        "h0_measurement/results/<R4_LLAMA8B_96K>/*.parquet" \
#        "h0_measurement/results/<R4_LLAMA70B_96K>/*.parquet" \
#        "h0_measurement/results/<R4_QWEN17B_*>/*.parquet" \
#        -o h0_measurement/reports/h0_rope_vs_length.pdf
#
# `page_rope` prints the decisive line directly:
#     R4 -- steepest per-octave drop, and where it falls:
#       llama31-8b     steepest at ctx  98,304 (rope_frac 0.75, window 131,072)
#       qwen3-30b...   steepest at ctx 262,144 (rope_frac 1.00, window 262,144)
#
# DECISION TABLE, written before the run.
#
#   Every model's steepest drop at rope_frac ~ 1.0, at DIFFERENT absolute L
#       -> ROPE-FRAC. The collapse is about the trained limit. Restate C2: the
#          band decays with context AND falls off a cliff as the model runs out
#          of trained positions, which are two mechanisms, not one. The 128k row
#          stops being evidence about long-context attention and becomes evidence
#          about extrapolation. Honest, narrower, and a cleaner mechanism --
#          also directly actionable, since it says a deployment at 50% of window
#          behaves unlike one at 100%.
#
#   qwen3-30b keeps sliding gently through 192k and 256k with no cliff
#       -> ABSOLUTE-L. C2 stands as written, the decay is a property of
#          attention, and "ways this dies #1" is confirmed rather than explained
#          away. Then llama33-70b's -36.9 needs its own account (its n95 explodes
#          117 -> 4,854, i.e. it genuinely diffuses); report that separately.
#
#   MIXED: qwen3-30b drops at 256k but less steeply than llama did at 128k
#       -> BOTH axes matter. Report rope_frac as a second covariate of C2 rather
#          than a replacement, and say so. This is the most likely outcome and it
#          is not a failure -- it makes the phase diagram two-dimensional in the
#          deployment variables, which is a stronger claim than one axis.
#
#   qwen3-1.7b flat from 8k to 32k then falls at 40k
#       -> corroborates ROPE-FRAC on a fifth model for almost no GPU time.
#
# CORPUS CAPACITY -- ALREADY CHECKED, no staging needed:
#   ctx  98,304  needs   452,198 chars/prompt   37/40 books are that big
#   ctx 196,608  needs   904,396               27/40
#   ctx 262,144  needs 1,205,862               22/40
# Against n_prompts of 2-3, so every run gets distinct single-book windows with
# no splicing. Re-verify after any corpus change with:
#   python h0_measurement/prefetch_corpus.py --check-ctx 262144 --n-prompts 2
#
# THE ONE THING THAT WOULD INVALIDATE THE TEST: a 192k/256k run failing the input
# validity gate for a different reason -- if qwen3-30b stops retrieving the needle
# at 256k, its band is not comparable to its 128k band and the cliff would be
# confounded with the model simply breaking. That is itself informative (a model
# that cannot use its own window is a finding), but it must be reported as such
# rather than as a phase measurement. Read the TASK-LEVEL line in the log first.
