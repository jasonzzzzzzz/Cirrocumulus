# Theorem R at 4-bit keys: where the gap comes from, what closes it, what explains the design (2026-10-10)

CPU only, real weights. Code: `exp4bit_s1h5.py` (this folder; numbers in `findings_4bit.json`), `theory3_s1h5.py`
(Theorem R v3, Theorem M), tests `test_theory3_s1h5.py` (all pass, `--fast` and full). Theory: `theory.md` §6 (v3),
§10 (Theorem M), §11 (decode), §12 (this study, in short).

## The question

The guidance to check had three parts:
- **Claim:** the main open problem is Theorem R at 4 bits, and the 206× gap to the oracle comes mostly from guarding
  against one tail row being badly underestimated.
- **Remedies:** (1) give the truncation's failure budget mostly to the rows the vote's sketch ranks as heavy;
  (2) handle the few heaviest rows by a separate deterministic term.
- **Expectation:** either could plausibly close a large part of the gap.

**Verdict.**
- The diagnosis is right in substance, with one correction. What costs the most (about 10×) is the unknown weight of
  the heaviest unread rows. A row's weight is known only up to e^{zσ} ≈ 7.5, and this enters Bennett's range (twice)
  and the self-bounds' maxima (squared). But the tail at the median head-step is not one dominant row: the heaviest
  unread row carries 3.9% of the unread mass, and the unread rows are worth about 124 equal rows.
- The remedies help less than expected. Made rigorous, budgeting gives 1.4–1.5×, peeling 1.2–1.7×, and both together
  1.7–2.1×: 201 → 98× on L1, 217 → 126× on L3.
- They are not enough for FP8-level certificates at 4-bit keys. Even a bound at 2× the oracle would certify only 34%
  of head-steps at FP8's level, and no variant tried gets below about 100×. The obstacle is structural (below).
- What does carry over: a tighter, still rigorous Theorem R (v3); an exact second-moment theorem (M) that explains and
  predicts the 4-bit design's accuracy; and a proof that the guarantee holds through free-running decode until the
  first token that differs from the exact model's (DD).

## Setup

oracle_s1h5's: one layer, the first L − 48 tokens are the context, the next 32 queries vote (a Lloyd-Max 4-bit
sketch, coin-free) for C/8 rows per KV head, and the last 16 are measured. Tier 1 is dithered (keys at 4/6/8 bits,
values at 4), the tail unbiased, δ = 0.05 per head-step. The oracle width is the 95% quantile of the error over 200
draws (the best any certificate could do); bounds are the median of 10 draws. "Within FP8" means the bound is at most
FP8's error at that head-step.

## 1. The diagnosis

Llama-3.2-1B L1, 4-bit keys. These counterfactuals replace estimated quantities by the true ones (they are not
certificates):

| v2 with … | bound / oracle |
|---|---|
| nothing changed | 201 |
| the true weights inside every max (Bennett's ranges, the self-bounds' maxima) | 19.6 |
| the true self-bounded sums | 71.6 |
| both | 11.3 |
| both, and the true distances ‖v_i − c‖ | 9.8 |

The ceiling: the oracle is 0.74× FP8's error at the median. If a bound were k× the oracle, the share certified within
FP8 would be 62% (k = 1), 34% (2), 7% (5), 2.7% (10) or 0.4% (20).

## 2. The remedies, made rigorous

All rows are certificates with coverage 1 over 200 draws:

| certificate | L1 | L3 |
|---|---|---|
| v2 | 201 | 217 |
| budget: half / 90% of the truncation budget by the vote's sketch weights | 143 / 133 | 159 / 149 |
| peel: rows with true weight > κM in a deterministic term (on the truncation event); κ = 0.01 / 0.001 / a grid of 7 with a union | 153 / 124 / 121 | 188 / 154 / 154 |
| both (grid + 90% budget) | 98 | 126 |
| v3 (theory.md §6: one-row maxima, identity regrouped by noise source, χ² value term) | 114.5 | 124 |

Why they stop at about 2×:
- *Peeling.* The heavy rows' term pays e^{t} − 1 per row with no cancellation between rows. Alone it is 52× the
  oracle, so it trades one large term for another.
- *Budgeting.* The truncation level grows like √(2 ln(1/p_i)), so even 90% of the budget on a few rows lowers their
  z only from 4.8 to about 3.3 (e^{1.5σ} ≈ 1.9×). The sketch also does not always rank the heaviest unread rows first.
- *The two overlap with v3's one-row maxima*, which attack the same unknown-weight factor inside maxima: a side run
  of v3 with the sketch budget gave 105× on L1.

## 3. The exact law of the tier's error does not rescue Bennett

The dominant key sum, with true weights; radius / the true 95% quantile of the sum (`lawcheck`):

| radius | ratio |
|---|---|
| Bennett, true variance and range at the truncation | 9.2 |
| exact truncated law, per-row scales (rigorous tail integral) | 6.9 |
| exact truncated law, one common scale | 99 |

Without truncation the exact-law exponent explodes. Z = e^{−δ} is lognormal-like, and its moment generating function
exists only because |δ| ≤ b ≈ 11σ almost surely. So every Chernoff-type argument has to truncate every unread row,
and a union over |T| rows sets the level at z ≈ √(2 ln(4|T|/δ)) ≈ 4.8. An earlier Monte Carlo estimate of the
exact-law gain (2.2×) was optimistic because it never sampled the far tail.

## 4. Theorem R v3 across layers and models

Bound / oracle, median, v2 → v3; in brackets the share within FP8, v2 → v3:

| view (context rows) | 4-bit keys | 6-bit | 8-bit |
|---|---|---|---|
| Llama-1B L0 (1,552) | 80 → 52 (0 → 0) | 13.8 → 11.5 (0.002 → 0.004) | 10.8 → 8.5 (0.006 → 0.014) |
| Llama-1B L1 (1,552) | 201 → 114.5 (0 → 0) | 20.2 → 17.3 (0.09 → 0.10) | 14.3 → 11.1 (0.28 → 0.37) |
| Llama-1B L3 (1,552) | 217 → 124 (0 → 0) | 19.1 → 16.2 (0.12 → 0.16) | 13.3 → 10.5 (0.37 → 0.47) |
| Llama-1B L1 (8,144) | 337 → 159 (0 → 0) | 18.5 → 14.9 (0.11 → 0.16) | 11.2 → 7.7 (0.31 → 0.46) |
| GPT-OSS L1, sinks (1,552) | 26,057 → 7,485 (0 → 0) | 51.7 → 39.3 (0.014 → 0.014) | 18.3 → 14.6 (0.035 → 0.050) |
| Gemma 3 L5, global (1,552) | 14.2 → 11.6 (0 → 0) | 9.2 → 6.9 (0 → 0) | 8.5 → 6.3 (0 → 0) |

Brute force (random, spiky, 30× outlier and adversarial data; 2/4/6-bit keys; plain and unbiased tails; 19,200 head
cases): no failure at any δ from 0.3 to 10⁻³, and never more than 5% above v2.

## 5. Theorem M: why the 4-bit design works

The identity (M + Σ_T w̃_i)(o − õ) = Σ_T w_i(1 − Z_i)(v_i − o) + Σ_T w_i Z_i e_i gives the tail's MSE exactly
(theory.md §10). Medians over head-steps, 200 draws. The columns: the unread mass and the unread share of
Σ w²‖v − o‖²; the tail's MSE over a dense dithered tier's at 4-bit keys (predicted / measured); measured / predicted MSE
at 4 / 6 / 8-bit keys; the tail's RMS error over FP8's at 4 / 6 / 8 bits; a dense dithered 4-bit tier's and eviction's
error over FP8's.

| view | unread mass | unread share | tail / dense MSE | measured / predicted | tail RMS / FP8 | dense / FP8 | eviction / FP8 |
|---|---|---|---|---|---|---|---|
| Llama-1B L0 | 0.240 | 0.198 | 0.220 / 0.222 | 0.996 / 1.000 / 1.003 | 1.39 / 0.58 / 0.47 | 3.4 | 18.4 |
| Llama-1B L1 | 0.024 | 0.0018 | 0.0186 / 0.0187 | 0.991 / 0.994 / 0.999 | 0.52 / 0.16 / 0.11 | 3.4 | 3.35 |
| Llama-1B L3 | 0.029 | 0.0009 | 0.0095 / 0.0090 | 0.981 / 1.003 / 0.999 | 0.38 / 0.12 / 0.09 | 4.3 | 2.53 |
| Llama-1B L1, 8K | 0.049 | 0.0015 | 0.0223 / 0.0226 | 0.990 / 1.002 / 1.002 | 0.57 / 0.18 / 0.13 | 3.8 | 6.03 |
| GPT-OSS L1 | 0.121 | 0.199 | 0.227 / 0.229 | 0.922 / 0.995 / 1.001 | 4.14 / 0.91 / 0.43 | 9.2 | 8.58 |
| Gemma 3 L5 | 0.772 | 0.655 | 0.663 / 0.663 | 1.000 / 1.001 / 1.001 | 3.40 / 2.73 / 2.69 | 4.2 | 89.9 |

Readings:
- **The mechanism.** On Llama-1B L1 the vote leaves 2.4% of the mass unread but 0.18% of Σ w²‖v − o‖². The tail's
  MSE is 1.86% of a dense 4-bit tier's (predicted) and 1.87% (measured). Eviction's error is 3.35× FP8's; the 4-bit
  tail's is 0.52×.
- **Per-model bits.**
  - Llama: 4/4 is enough in L1/L3.
  - GPT-OSS: its σ ≈ 0.94 at 4-bit keys needs 6–8-bit keys.
  - Gemma 3: value-limited; its 4-bit values hold the tail at 2.7× FP8 at any key width.
  This matches the R5 adapters' earlier qualitative findings, but from a formula rather than a bound/oracle
  heuristic.
- **When the design helps at all.** The unread share of Σ w²‖v − o‖² is the predictor. The MSE gain over a dense
  tier (the measured ratio, value term included) follows it:
  - Llama L1/L3: shares 0.0018 / 0.0009, gains of 53× and 111×;
  - Llama L0 and GPT-OSS L1: shares about 0.2, gains of 4.4–4.5×;
  - Gemma 3 L5: share 0.66, a gain of 1.5×. That global layer's attention is flat: 77% of the mass is unread at 1/8,
    and evicting it costs 90× FP8's error. The tail is nearly a dense 4-bit store, and with QK-norm's small σ the value
    term dominates.
  So Gemma's "value limit" is first a flat-attention effect, which Theorem M exposes from one FP pass.

## Regenerate

```bash
cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
P=h0_measurement/bugs/14h_theory/exp4bit_s1h5.py
OMP_NUM_THREADS=8 .venv/bin/python -u $P capture llama1b 1          # also: llama1b 0, llama1b 3, llama1b 1 8192, gptoss 1, gemma3 5
OMP_NUM_THREADS=8 .venv/bin/python -u $P diagnose view_llama1b_L1_1600.pt
OMP_NUM_THREADS=8 .venv/bin/python -u $P remedies view_llama1b_L1_1600.pt
OMP_NUM_THREADS=8 .venv/bin/python -u $P lawcheck view_llama1b_L1_1600.pt
OMP_NUM_THREADS=8 .venv/bin/python -u $P evaluate view_llama1b_L1_1600.pt llama1b_L1
```
Views are cached in `.views/` (or `$TH_CACHE`); each command takes seconds to a few minutes on 8 threads.
