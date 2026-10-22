# Part 8: the tilted tail and a Hsu–Kakade–Zhang key sum (CPU, 2026-10-10)

> **Correction (2026-10-12): every Gemma-3-12B number in this file before that date came from a randomly initialized model.**
> `adapters_s1h5._load` built `Gemma3ForCausalLM` from the multimodal checkpoint. transformers 5.16 does not map that checkpoint's `language_model.*` keys onto it, so every text weight was freshly initialized, and that branch skipped the missing-weight check. Random weights give flat attention.
> - Gemma 3's "flat, value-leaning" class (family E), its 6/6 choice, L5 as the flat-attention example (66% of the second moment unread) and R8's Gemma 3 predictions all describe the random model.
> - With the checkpoint's weights, L5 at 1,600 tokens has 7.6% of the second moment unread (between, not flat; unread mass 0.20, against 0.77) and a key share of 0.86 at 6/6 (key-leaning, against 0.37). Uniform 6/4 meets FP8's error at the median head-step (0.37×), not at p90 (1.36×).
> - The theorems' checks on those views (coverage, Theorem M's accuracy) remain valid statements about those inputs, but they are not evidence about Gemma 3.
> - Fixed: the loader uses the multimodal class and refuses any missing weight; `g0map_s1h5.provenance` compares loaded weights with the checkpoint's. No other model was affected: the 18 GQA/MHA models and the two MLA models load with no missing weight. G0 (`findings/findings_g0.md`) recomputes Gemma 3 at 1,600 tokens and at 32K.

`STATUS.md` step 4: tighten Theorem R at the R6 design's width, 6-bit keys.
- Code: `../theory5_s1h5.py` (`tail_bound_rand_split_hkz`, `key_proxy`, `exact_key_proxy`, `prep`, `certified_read`;
  harnesses `compare` and `certread`).
- Tests: `../test_theory5_s1h5.py`.
- Theory: `../theory.md` §6f–§6g.
- Data: `findings_part8.json`, next to this file.

Setup: part 7's (oracle_s1h5's configuration on cached real layers). The vote reads 1/8 of the context exactly (a
Lloyd-Max 4-bit sketch, 32 voter queries); 16 later queries per head are measured; 4-bit dithered values; δ = 10⁻³ per
head-step; 3 draws (2 for the certified read). Views are in `../.views/` (moved to `../../14_logs/14h_theory/views/` on 2026-10-22); `exp4bit_s1h5.py capture` regenerates them.

## What changed

Two changes to part 7's split, each valid for every input:
1. **The tilted tail.** E[e^{−δ}; |δ| > t] is bounded by a tilted Chernoff bound (≈ p) instead of Cauchy–Schwarz
   (≈ √p). This removes a bias of about 6·10⁻⁴ per row and a variance inflation of about 1.2·10⁻³ per row; at 8-bit keys
   that inflation was larger than σ² itself.
2. **The key sum by Hsu–Kakade–Zhang** instead of Bennett. Given the value coins, K is a sum of independent scalars
   along observed vectors. HKZ needs a sub-Gaussian proxy for the clipped weight ratio Z̃; there are two:
   - **P-LSI** (closed form, from the cube's sharp log-Sobolev inequality): tight for small σ, loose at 6 bits (3.3σ²)
     and very loose at 4 bits (129σ²);
   - **P-exact** (from the dither's exact law, with every step an inequality): within 1.07× of the optimal proxy at 6
     bits (1.68σ² against 1.57σ²) and 1.05× at 4 bits (14.0σ² against 13.3σ²). It is computed for each head-step's 16
     largest contributors.

The colleague's remedies were tested on top: remedy 1 (truncation budget to the sketch's 16 heaviest unread rows, share
0.5) and remedy 2 (the heaviest rows in a deterministic term).

## Validity

- **Brute force** (random, spiky, 30×-outlier and adversarial data; 2–8-bit keys; δ from 0.5 to 10⁻³; 1,536 head
  cases per form): every form, including P-exact and remedy 1, has 0 failures at δ = 10⁻³, and fails less than δ × cases
  at every δ.
- **The proxies against exact laws.** Quadrature for one or two channels and convolution for 8 and 64, |λ|s from 10⁻³
  to 3,000, both signs: ψ(λ) ≤ λ²s²/2 everywhere. The largest ratio is 0.998 for P-exact and 0.80 for P-LSI.
- **The density bound behind P-exact.** Every tilted uniform sum checked has sup f²·Var ≤ 1; the largest value is
  0.999, from one strongly tilted channel, which approaches the exponential's 1.
- **The certified read** with the new bound meets its target on every synthetic head case (δ = 0.05).
- **On real weights:** coverage is 1.000 in every cell below.

## Every certificate on the same draws

Median bound / error · the share of head-steps certified within FP8's per-head error:

| view | key bits | split (part 7) | + tilted tail, HKZ (P-LSI) | + P-exact | + remedy 1 |
|---|---|---|---|---|---|
| Llama-1B L1 (1.6K) | 6 | 18.0 · 0.13 | 9.36 · 0.34 | 7.08 · 0.45 | **6.75 · 0.47** |
| Llama-1B L3 (1.6K) | 6 | 17.4 · 0.21 | 9.02 · 0.46 | 7.02 · 0.58 | **6.72 · 0.59** |
| Llama-1B L0 (1.6K) | 6 | 10.3 · 0.00 | 5.25 · 0.07 | 4.68 · 0.08 | **4.60 · 0.08** |
| Llama-1B L1 (8K) | 6 | 19.3 · 0.15 | 10.4 · 0.30 | 7.81 · 0.38 | **7.34 · 0.41** |
| Gemma 3 L5 global (1.6K) | 6 | 4.04 · 0 | 2.06 · 0 | 2.06 · 0 | **2.05 · 0** |
| GPT-OSS L1 (1.6K) | 6 | 77.1 · 0.02 | 63.7 · 0.03 | 35.5 · 0.04 | **32.4 · 0.05** |
| Llama-1B L1 (1.6K) | 8 | 5.77 · 0.66 | 2.62 · 0.88 | 2.61 · 0.88 | **2.57 · 0.89** |
| Llama-1B L3 (1.6K) | 8 | 5.64 · 0.75 | 2.56 · 0.95 | 2.55 · 0.95 | **2.51 · 0.95** |
| Llama-1B L0 (1.6K) | 8 | 4.33 · 0.14 | 2.07 · 0.48 | 2.07 · 0.48 | **2.05 · 0.49** |
| Llama-1B L1 (8K) | 8 | 5.30 · 0.70 | 2.66 · 0.89 | 2.64 · 0.89 | **2.59 · 0.90** |
| Gemma 3 L5 global (1.6K) | 8 | 2.79 · 0 | 1.37 · 0 | 1.37 · 0 | **1.37 · 0** |
| GPT-OSS L1 (1.6K) | 8 | 10.3 · 0.09 | 4.78 · 0.22 | 4.28 · 0.24 | **4.21 · 0.24** |
| Llama-1B L1 (1.6K) | 4 | 522 · 0 | 1,035 · 0 | 326 · 0 | **287 · 0** |
| Llama-1B L3 (1.6K) | 4 | 534 · 0 | 1,043 · 0 | 379 · 0 | **359 · 0** |
| Llama-1B L0 (1.6K) | 4 | 137 · 0 | 194 · 0 | **92.2 · 0** | 94.6 · 0 |
| Llama-1B L1 (8K) | 4 | 875 · 0 | 1,947 · 0 | 634 · 0 | **554 · 0** |
| Gemma 3 L5 global (1.6K) | 4 | 15.0 · 0 | 9.28 · 0 | **8.49 · 0** | 8.50 · 0 |
| GPT-OSS L1 (1.6K) | 4 | **147,291** · 0 | 878,145 · 0 | 858,969 · 0 | 836,052 · 0 |

Readings:
- **6-bit keys (the R6 design's width): 2.4–2.7× tighter than the split on Llama-1B.** The bound is 6.7–7.3× the error
  on L1/L3 (1.6K and 8K), and the share certified at FP8's level rises from 0.13–0.21 to 0.41–0.59.
  - Of the gain, the range-free HKZ form with P-LSI gives about 1.9× and P-exact about 1.3×.
  - Remedy 1 adds 5%.
- **8-bit keys: 2.0–2.3× tighter**, at 2.0–2.6× the error. The value term now dominates; 0.89–0.95 of L1/L3
  head-steps are certified at FP8's level.
- **Gemma 3's global layer is certified at 1.4–2.1× the error and still not at FP8's level.** Its 4-bit values keep
  the error itself above FP8's (flat attention; Theorem M).
- **GPT-OSS gains 2.4–2.5× but stays loose** at 6 bits (32×). Its score noise there is about a quarter of its 4-bit
  σ ≈ 0.9, so σ ≈ 0.2 and e^t ≈ 3. At 4 bits every form is
  ≥ 10⁵× the error. With HKZ alone it is worse there than the split, because P-exact is skipped for rows whose series
  would exceed 6,000 terms. Keeping Bennett as a second form (`forms=("bennett", "hkz")`) recovers the split's value
  at a small event cost. The key width is fixed by the design, so this choice is coin-free.
- **4-bit keys on Llama-1B: 1.5–1.8× tighter than the split** (92–554×). On L1/L3 at 1.6K it is also tighter than
  v3 + sure sums (287 and 359×, against 484 and 505×).
  It is still nowhere near FP8's level (§12).

## The colleague's remedies, with P-exact (Llama-1B L1, 6-bit keys, one draw)

| variant | bound / error | certified within FP8 |
|---|---|---|
| HKZ + P-exact | 6.86 | 0.45 |
| remedy 1: 4 / 16 / 64 sketch-heavy rows, share 0.5 | 6.55 / 6.49 / 6.53 | 0.46 |
| remedy 1: 4 / 16 / 64 rows, share 0.9 | 6.67 / 6.53 / 6.51 | 0.44–0.46 |
| remedy 2: the 1 / 2 / 4 heaviest rows deterministic | 8.42 / 9.55 / 11.46 | 0.36 / 0.33 / 0.25 |
| remedies 1 + 2 (1 / 2 / 4 rows, share 0.5) | 7.79 / 8.62 / 9.99 | 0.38 / 0.36 / 0.31 |

- The sketch's top-4 and top-16 unread rows overlap the true top rows by 64% and 73%.
- Remedy 1 moves z from 5.6 to about 4.7 on the favoured rows, and z grows like √(ln 1/p): a 5% gain.
- Remedy 2 adds (e^t − 1)·w̃·‖u‖ linearly per row, where HKZ adds the rows in quadrature with near-exact proxies, so
  it costs more than it saves.

## What is left at 6 bits

Non-rigorous substitutions on Llama-1B L1 (6-bit, draw 0), as bound / error:

| substitution | bound / error | of which K | V |
|---|---|---|---|
| none (HKZ + P-LSI) | 9.17 | 7.26 | 1.77 |
| the true weights w for w̄ | 4.86 | 3.83 | 1.00 |
| Var Z as the proxy | 5.53 | 3.81 | 1.76 |
| both | 2.97 | 2.00 | 1.00 |

The second substitution is optimistic: the optimal valid proxy is 1.57σ², not Var Z ≈ 1.02σ². P-exact reaches 1.68σ²,
so the proxy's slack is now about 7%.

The remaining factor is the sure weights: w̄ ≥ w costs e^t ≈ 1.75 at t ≈ 0.56, on both K and V. Two ways around it
were tried, and neither helps:
- a self-bound, which needs no single row to carry more than about 1% of the sum, whereas the median head-step's
  heaviest unread row carries 21%;
- peeling up to 128 sketch-ranked rows off the self-bound.

## The certified adaptive read

Setup: part 7's schedule. The vote is read first, then 0, 1/256, …, 1/2 more rows in the sketch's order, then
everything; each set is bound at δ/9 (δ = 10⁻³). The target is FP8's own per-head error, an oracle target. "Fine" is a
16-set coin-free schedule (`theory5_s1h5.FINE`: steps of 1/64 between 1/32 and 1/8, then 5/32, 3/16, 1/4, 3/8, 1/2),
each set at δ/16. Traffic = (1 − f)(k + 4)/32 + f, with f the share of rows read exactly. The oracle stop is the first
set whose true error meets the target, a lower limit for any certificate. Coverage is 1.0000 in every cell.

| view | key bits | split (part 7) | part 8 (HKZ + P-exact) | part 8, fine schedule | oracle stop |
|---|---|---|---|---|---|
| Llama-1B L1 | 6 | 0.432 (0.609) | 0.263 (0.493) | **0.244 (0.480)** | 0.125 (0.399) |
| Llama-1B L3 | 6 | 0.449 (0.621) | 0.287 (0.510) | **0.271 (0.499)** | 0.125 (0.398) |
| Llama-1B L1, 8K | 6 | 0.279 (0.505) | 0.181 (0.437) | **0.175 (0.433)** | 0.125 (0.399) |
| Llama-1B L0 | 6 | 0.846 (0.894) | 0.605 (0.728) | — | 0.144 (0.411) |
| Gemma 3 L5 (global) | 6 | 1.000 (1.000) | 1.000 (1.000) | — | 0.643 (0.754) |
| Llama-1B L1 | 8 | 0.167 (0.479) | **0.129 (0.456)** | — | 0.125 (0.453) |
| Llama-1B L3 | 8 | 0.182 (0.489) | **0.135 (0.460)** | — | 0.125 (0.453) |
| Llama-1B L1, 8K | 8 | 0.130 (0.457) | **0.126 (0.454)** | 0.126 (0.454) | 0.125 (0.453) |
| Llama-1B L0 | 8 | 0.449 (0.655) | **0.241 (0.526)** | — | 0.137 (0.461) |
| Gemma 3 L5 (global) | 8 | 1.000 (1.000) | 0.944 (0.965) | — | 0.638 (0.774) |

Each cell is rows read exactly (traffic), the mean of 2 draws. The fine schedule is 2 draws at 1.6K, which agree to
±0.001, and 1 draw at 8K, where a single draw of the coarse schedule matched the 2-draw mean to ±0.002. Remedy 1
changes the rule's reads by at most 0.001.

Readings:
- **At 6-bit keys, the R6 design's width, the certified read meets FP8's per-head error at traffic 0.480 (L1) and
  0.499 (L3) with the fine schedule, and at 0.433 on L1 at 8K.** That is at or below FP8's 0.50; part 7 needed
  0.50–0.62. The longer context leaves a flatter unread tail, which is cheaper to certify, so 128K may do better still
  (step 3 measures it). It is now certified FP8-level per-head error at about FP8's traffic. It is not clearly below it
  at 1.6K, and the target is an oracle (`STATUS.md` step 6).
- **Gemma 3's global layer cannot meet FP8's per-head error with 4-bit values at any read set short of nearly all.**
  The oracle itself needs 64% of rows; the rule reads 94–100%. Its remedy is more value bits, not a tighter
  certificate.
- **At 8-bit keys the rule reads what the oracle reads**: 12.6–13.5% of rows against 12.5%, at traffic 0.454–0.460.
  The certificate is no longer what costs traffic there.
- **L0 (the first layer, flat attention) stays far above 0.5** at both widths: 0.728 at 6 bits, 0.526 at 8. Its
  oracle stop needs 14% of rows.
- What remains between the rule and the oracle stop at 6 bits (0.175–0.27 against 0.125 rows) is mainly the sure weights
  (e^t ≈ 1.75; see "What is left at 6 bits").
