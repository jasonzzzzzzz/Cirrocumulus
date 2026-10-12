# The nested coin-free store (CPU, 2026-10-11)

RESEARCH_PLAN.md §3 N; theory.md §13. Code: `../nested_s1h5.py`, tests `../test_nested_s1h5.py`, end to end
`../e2e_s1h5.py`. Data: `findings_nested.json`, and the end-to-end runs in `klbridge_runs/`.

**The question.** Theorem R needs the rows read exactly to be chosen without tier 1's coins (R1).
- R6 met R1 with a separate Lloyd-Max 4/4 store for the vote. That store is 0.25 of FP16 K+V, and no reported memory
  number counts it (CLAIMS_LEDGER S2).
- Can one store serve both the vote and the tail?

**The design.**
- **Keys:** a deterministic 4-bit Lloyd-Max base, plus a 2-bit residual quantized with subtractive dither inside the base
  cell.
  - The base is exactly `quant.quantize_keys`, the sketch the vote reads, so the vote is coin-free.
  - The tail reads base + residual: 6-bit keys.
- **Values:** the 4-bit dithered values, as now.
- **Lemma N.** Given the base, the residual error is uniform on ±D_j/2, independent across coordinates and rows, and
  independent of the data. So Theorems R and M hold with per-coordinate steps.

## 1. Lemma N, by simulation
2,048 Gaussian rows (every 7th with a 12× outlier coordinate), d = 64, 400 dither draws. In units of each coordinate's
step D:

| residual | mean | variance × 12 | 4th moment × 80 | max |error| | corr(mean error, y) | base = `quantize_keys` |
|---|---|---|---|---|---|---|
| 2 bits | +4.6e-5 | 1.0001 | 1.0001 | 0.5000 | +1.3e-5 | to 6e-6 (fp32) |
| 3 bits | −2.4e-5 | 1.0000 | 1.0000 | 0.5000 | −3.4e-5 | to 6e-6 |

A uniform on ±1/2 has mean 0, variance × 12 = 1 and 4th moment × 80 = 1. The largest per-coordinate mean error is
4.7 s.e. over 131,072 coordinates, the size the maximum of that many normal draws should have.

## 2. Brute force (`test_nested_s1h5.py`)
- **Reduction.** With one step per row, every per-coordinate quantity matches rquant_s1h5's and theory3_s1h5's to
  6e-16: the coin moments, σ, the almost-sure bound, the exact moments and part 8's exact-law proxy.
- **Monte Carlo.** E Z, E Z² and Theorem M's E‖N‖² are within 4.5 s.e. of Monte Carlo (max 3.82 and 1.88).
- **Every Theorem R form** has no failures at any δ (0.5, 0.1, 0.01, 10⁻³):
  - forms: v1, v2, the split, and part 8 (HKZ; + the exact proxy; + remedy 1);
  - data: random, spiky, 30× outlier and adversarial instances;
  - widths: bases of 2–4 bits, residuals of 2–3 bits;
  - 768 head cases per form and δ.
- **Median bound / error at 6–7 bits:** v1 15.5, v2 14.7, the split 6.6, part 8 2.4.

## 3. Real layers
Thirteen layers spanning every family in `../MODEL_FAMILY.md`, including R6's two models. Each uses the 1,600-token
PG-19 view, with the same read set (the vote's 1/8) for both tiers.
- **Error columns:** the tail's RMS error over FP8's (median over head-steps), and Theorem M's measured / predicted MSE
  over 200 draws, across all four configurations.
- **Certificate columns:** coverage over all forms and both tiers, and part 8's (hkz+exact) median bound / error and
  share certified within FP8's error, uniform 6/4 / nested 4+2 (3 draws).
- **Vote columns:** the share of the measured queries' attention mass that each vote reads (R6's sketch / the nested
  base), and the share of rows the two votes have in common.

| layer | uniform 6/4 | nested 4+2/4 | uniform 7/4 | nested 4+3/4 | measured / predicted | coverage | part 8 bound / error | certified within FP8 | vote mass, R6 / nested | rows in common |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B L1 | 0.159 | 0.177 (1.11×) | 0.119 | 0.121 | 0.994–0.999 | 1.000 | 7.1 / 9.1 | 0.45 / 0.29 | 0.925 / 0.924 | 0.79 |
| Llama-3.2-1B L3 | 0.123 | 0.139 (1.14×) | 0.094 | 0.097 | 0.995–1.003 | 1.000 | 7.0 / 9.7 | 0.58 / 0.39 | 0.932 / 0.932 | 0.77 |
| Llama-3.1-8B (R6) L1 | 0.255 | 0.275 (1.08×) | 0.217 | 0.216 | 0.999–1.001 | 1.000 | 5.3 / 6.2 | 0.33 / 0.19 | 0.927 / 0.927 | 0.80 |
| Llama-3.1-8B (R6) L16 | 0.049 | 0.051 (1.06×) | 0.043 | 0.043 | 0.997–1.000 | 1.000 | 4.6 / 5.4 | 0.86 / 0.84 | 0.953 / 0.952 | 0.86 |
| Llama-3.3-70B L20 | 0.081 | 0.084 (1.05×) | 0.075 | 0.075 | 0.997–0.998 | 1.000 | 3.4 / 4.0 | 0.93 / 0.89 | 0.948 / 0.948 | 0.91 |
| Qwen3-30B-A3B (R6) L1 | 1.255 | 1.436 (1.14×) | 0.935 | 0.947 | 1.001–1.003 | 1.000 | 11.7 / 17.5 | 0.02 / 0.01 | 0.541 / 0.544 | 0.80 |
| Qwen3-30B-A3B (R6) L3 | 0.277 | 0.322 (1.16×) | 0.140 | 0.138 | 0.998–1.004 | 1.000 | 775 / 1,627 | 0.00 / 0.00 | 0.831 / 0.829 | 0.31 |
| Qwen3-30B-A3B (R6) L24 | 0.037 | 0.042 (1.14×) | 0.030 | 0.030 | 0.996–1.000 | 1.000 | 6.5 / 8.6 | 0.87 / 0.82 | 0.923 / 0.924 | 0.87 |
| Qwen3-1.7B L13 | 0.131 | 0.144 (1.09×) | 0.103 | 0.105 | 0.997–0.999 | 1.000 | 6.6 / 8.0 | 0.56 / 0.48 | 0.827 / 0.828 | 0.89 |
| GPT-OSS-20B L1 | 0.908 | 1.110 (1.22×) | 0.560 | 0.592 | 0.995–1.001 | 1.000 | 36.0 / 64.7 | 0.04 / 0.03 | 0.548 / 0.537 | 0.61 |
| Gemma-3-12B L5 | 2.736 | 2.747 (1.00×) | 2.703 | 2.702 | 1.001–1.002 | 1.000 | 2.1 / 2.2 | 0.00 / 0.00 | 0.213 / 0.214 | 0.89 |
| Granite-4.2-30B L4 | 2.473 | 2.473 (1.00×) | 2.473 | 2.473 | 0.996–0.996 | 1.000 | 1.3 / 1.3 | 0.00 / 0.00 | 0.125 / 0.125 | 0.98 |
| Granite-4.2-30B L16 | 0.762 | 0.794 (1.04×) | 0.662 | 0.663 | 0.999–1.001 | 1.000 | 4.4 / 5.4 | 0.07 / 0.04 | 0.539 / 0.532 | 0.87 |

- **Theorem M is exact for the nested tier.** Measured / predicted is 0.994–1.004 on every layer and configuration.
- **The cost is about 0.1–0.3 bit of key precision.**
  - Nested 4+2 raises the tail's RMS error by 0–22% over uniform 6-bit keys (median about 1.1×), at the same traffic.
  - Nested 4+3 matches uniform 7-bit keys within 2%.
  - Where the value term dominates (Gemma 3, Granite-4.2), the key width doesn't matter.
- **Every certificate stays valid** (coverage 1.000).
  - Part 8 is 1.05–1.5× looser (median about 1.2×), from the larger error and the outer cells' wider steps, and about
    2× on the noisiest layers (GPT-OSS L1, Qwen3-30B L3).
  - The certified share falls accordingly; on Llama-1B L1 it goes from 0.45 to 0.29.
- **The vote is unchanged.** On every layer the base's vote reads the same attention mass as R6's sketch, within 0.011.
  The rows the two pick differ only among near-ties, because the rotations differ (as few as 31% of rows in common on
  Qwen3-30B L3).

## 4. End to end (Llama-3.2-1B, CPU)
KL per answer token against FP, teacher-forced, from `../e2e_s1h5.py`. Units are 1,536 context rows, a 32-token question
and a 32-token answer. They come from 4 texts (PG-19, and three LongBench v2 contexts: a code repository, a table, an
agent log) × 2 windows; 2 dither seeds each.

| unit | uniform 6/4 (0.398) | nested 4+2/4 (0.398) | nested 4+3/4 (0.426) | dense 6/6 (0.375) | FP8 (0.50) |
|---|---|---|---|---|---|
| PG-19, window 0 | 3.5e-5 | 4.8e-5 | 1.8e-5 | 1.6e-4 | 3.4e-4 |
| PG-19, window 1 | 7.5e-5 | 9.1e-5 | 4.1e-5 | 4.2e-4 | 4.1e-4 |
| code, window 0 | 1.9e-5 | 3.3e-5 | 1.8e-5 | 3.5e-4 | 5.9e-4 |
| code, window 1 | 3.3e-5 | 6.1e-5 | 3.0e-5 | 2.9e-4 | 6.3e-4 |
| table, window 0 | 1.8e-4 | 1.9e-4 | 1.5e-4 | 5.0e-4 | 5.5e-4 |
| table, window 1 | 3.0e-4 | 3.1e-4 | 1.8e-4 | 2.3e-4 | 4.5e-4 |
| agent, window 0 | 3.2e-4 | 2.0e-4 | 1.5e-4 | 5.7e-4 | 6.6e-4 |
| agent, window 1 | 7.3e-5 | 7.5e-5 | 4.0e-5 | 2.5e-4 | 3.9e-4 |

Traffic is in parentheses. Each value is the KL per token, the mean over 2 seeds.
- **Nested 4+2 against uniform 6/4:** median 1.14× (0.62–1.84), at the same traffic. With 8 seeds (§6) it is 1.27× in
  geometric mean (0.90–1.57) and 1.15× on mean KL. §6's numbers supersede this one.
- **FP8 against nested 4+2:** FP8 is 4.9× higher (median), between 1.5× and 18× on every unit.
- **Nested 4+3** halves uniform 6/4's KL (0.57×) for +0.027 traffic.
- **Dense 6/6** costs about the same traffic as the two-tier design and has 3.9× its KL: two tiers beat one.

## 5. Memory and traffic
- **Traffic per decode step is unchanged:** the tail reads 6-bit keys and 4-bit values, 0.398 of FP16 at f = 1/8.
- **Memory:**
  - 10 bits per key-and-value element pair, plus per-row scalars: the key norm γ and maximum M, and the value step;
  - one more fp16 scalar per row than the uniform tier: 0.125 bit per coordinate at d = 128;
  - at rest, 0.31 of FP16 K+V;
  - during a question, about 0.44, counting the exact 1/8 fetched;
  - FP8 is 0.50.
- **The vote store disappears.** R6 allocated it at 4/4, 0.25 of FP16 K+V.
- **The vote's question pass** reads the base keys and the tier's 4-bit dithered values. R6's pass read Lloyd-Max 4-bit
  values.
  - R1 still holds layer by layer, because each layer has its own independent dither (Lemma C's argument).
  - Dithered rather than Lloyd-Max values in that pass have no measurable effect on the read set or the KL (§6,
    2026-10-12). §4's harness used the question's exact queries instead; §6 shows that is equivalent too.

## 6. The vote's question pass over the single store (CPU, 2026-10-12)
**The question.** R6's vote runs the question through a 4-bit store (`run_s1h5.vote_sets`): SnapKV's rule over
Lloyd-Max keys and values. In the single store, that pass reads the base keys, the same Lloyd-Max keys, but the tier's
dithered values, with this seed's coins. Does the vote still read as well, and is the KL unchanged? This was the one
untested piece of the memory claim (CLAIMS_LEDGER S2).

**Setup** (`../e2e_s1h5.py --plan vote`; `Unit.store_vote`).
- **The question pass:** at every layer, the question's rows attend to the context's store and to the question rows
  before them. The store's values are one of three:
  - R6's: Lloyd-Max 4-bit;
  - the single store's: the tier's dithered 4-bit values, the same coins the tail reads;
  - exact.
- **The vote:** each question row's softmax over the context, summed over the question's rows and the KV group. It keeps
  the top 1/8 per KV head.
- **The reference:** the exact-query vote, the question's exact queries over the base sketch, as in §4 and in
  `findings_klbridge.md`.
- **Units:** Llama-3.2-1B, 8 units (4 texts × 2 windows), and Qwen3-1.7B, 4 units (PG-19 and code × 2 windows). Each
  has 8 dither seeds.
- **Measured per layer:** the share of the answer queries' exact attention mass that each read set covers, and the
  read set's unread share of Σ w²‖v − o‖² (Theorem M's value-term weight).
- **Measured end to end:** KL per answer token for each design under each vote.

| | Llama-3.2-1B | Qwen3-1.7B |
|---|---|---|
| rows in common, dithered-value vote vs Lloyd-Max-value vote | 98.9% (98.5–99.3) | 98.7% (98.5–98.9) |
| rows in common with the exact-query vote, dithered / Lloyd-Max | 96.0% / 96.0% | 90.1% / 90.1% |
| attention mass read, dithered / Lloyd-Max / exact-query vote | 0.6356 / 0.6357 / 0.6354 | 0.5623 / 0.5623 / 0.5625 |
| largest per-layer mass difference, dithered − Lloyd-Max | 0.0018 | 0.0008 |
| unread share of Σ w²‖v − o‖², dithered / Lloyd-Max / exact-query | 0.0166 / 0.0166 / 0.0165 | 0.0358 / 0.0358 / 0.0355 |
| KL, single store: dithered-value vote / Lloyd-Max-value vote | 0.99 (0.97–1.02) | 1.01 (0.98–1.05) |
| KL, uniform 6/4: dithered-value vote / Lloyd-Max-value vote | 1.00 (0.96–1.05) | 0.98 (0.95–1.01) |
| KL, single store: dithered-value vote / exact-query vote | 1.00 (0.95–1.09) | 1.01 (0.96–1.07) |

The rows-in-common and mass rows are layer means, with ranges over units. The KL rows are geometric means over units,
with ranges; each unit is the mean of 8 seeds.

**The dithered values change nothing measurable.**
- The two votes share 98.7–98.9% of their rows and read the same attention mass, within 0.002 in every layer.
- The KL is unchanged to within ±5% per unit and 1% overall.
- R6's question pass and the exact-query vote are also equivalent: the same mass and the same KL, with 90–96% of rows in
  common.

**R1 still holds.**
- Layer l's vote reads only layer l's base keys and the question's queries at layer l.
- Those queries depend on the coins of lower layers only, through the pass's dithered values there.
- Each layer has its own independent coins (Lemma C), so layer l's read set is independent of layer l's coins. Theorem R
  conditions on everything else.

So the single store needs no vote store. Memory is 0.31 of FP16 K+V at rest and about 0.44 during a question (§5),
and every term of that accounting is now tested on CPU.

**The single store against R6's design, each with its own vote** (R7b's E4 pairing):

| unit | Llama-3.2-1B: single store / R6 design / FP8 | ratio | unit | Qwen3-1.7B: single store / R6 design / FP8 | ratio |
|---|---|---|---|---|---|
| PG-19 w0 | 4.7e-5 / 3.5e-5 / 3.4e-4 | 1.35 | PG-19 w0 | 6.8e-5 / 4.8e-5 / 2.7e-3 | 1.42 |
| PG-19 w1 | 1.1e-4 / 6.8e-5 / 4.1e-4 | 1.58 | PG-19 w1 | 1.6e-4 / 2.2e-4 / 2.6e-3 | 0.73 |
| code w0 | 2.8e-5 / 1.9e-5 / 5.9e-4 | 1.43 | code w0 | 2.2e-5 / 2.1e-5 / 1.8e-3 | 1.04 |
| code w1 | 4.9e-5 / 3.9e-5 / 6.3e-4 | 1.26 | code w1 | 3.7e-5 / 2.5e-5 / 1.1e-3 | 1.44 |
| table w0 | 3.3e-4 / 2.4e-4 / 5.5e-4 | 1.35 | | | |
| table w1 | 3.5e-4 / 3.2e-4 / 4.5e-4 | 1.10 | | | |
| agent w0 | 2.9e-4 / 3.5e-4 / 6.6e-4 | 0.85 | | | |
| agent w1 | 1.1e-4 / 7.5e-5 / 3.9e-4 | 1.44 | | | |
| **geometric mean** | | **1.27** | | | **1.12** |
| **ratio of mean KLs** | | **1.15** | | | **0.91** |

KL per token; 8 seeds for the two designs, FP8 is deterministic. The single store is nested 4+2/4 with the
dithered-value vote; R6's design is uniform 6/4 with the Lloyd-Max-value vote.
- **The single store costs 1.12–1.27× R6's design in geometric mean,** at the same traffic.
  - On mean KL, the quantity R7b's primary test uses, the cost is 1.15× on Llama-1B and 0.91× on Qwen3-1.7B.
  - The units with the most KL have the smallest ratios.
  - Under a common vote the ratios are the same (1.27 and 1.09).
- **§4's 1.14× was the median of 2 seeds; with 8 seeds it is 1.27× on Llama-1B.** R7b's prediction should read
  1.1–1.3× per unit, and about 0.9–1.15× on mean KL.
- **FP8 is 1.3–81× the single store's KL.** The smallest margins are Llama-1B's table units (1.3–1.7×), where the tail's KL
  is highest.

## 7. Part 8 at 32K on R7b's models (CPU, 2026-10-12)
**Data.** The R7b session captured every layer of Llama-3.1-8B and Qwen3-30B-A3B at 32K tokens (PG-19), as part of its
full-depth analysis (`../../14i_r7_theory_at_scale/r7b/findings/depth_r7b.md`). That analysis has no 8-bit keys, so
this section reruns part 8 on its views, read-only, with `../nested_s1h5.py depth` (2 draws, δ = 10⁻³; one Trillium CPU
node, 9 minutes). Per-layer results are in `depth32k/`, the summary in `findings_nested.json` → `depth32k`.

**What is measured.** On the vote's read set (`exp4bit_s1h5.load_setup`: 32 vote queries, 16 measured), for each of
four key widths (uniform 6/4 and 8/4, nested 4+2/4 and 4+4/4):
- coverage;
- part 8's median bound / error;
- the share of head-steps certified within FP8's error;
- σ_w, the attention-weighted score noise of the unread rows (median over head-steps).

| | Llama-3.1-8B (32 layers) | Qwen3-30B-A3B (48 layers) |
|---|---|---|
| part 8 bound / error, nested 4+2 | 3.6–7.6× in every layer | 4.0–1,960×; above 20× in 14 layers |
| certified within FP8, nested 4+2 | 0.68–1.00 | 0.00–1.00 |
| part 8 bound / error, nested 4+4 | 1.6–2.2× | 1.7–8.6× in every layer |
| certified within FP8, nested 4+4 | 0.96–1.00 | 0.69–1.00 |
| σ_w at 6-bit keys | 0.04–0.09 | 0.05–0.49 |
| coverage, every layer and width | 1.000 | 1.000 |

Qwen3-30B-A3B's loose layers at 4+2 (bound > 20× the error):

| layer | 3 | 16 | 28 | 8 | 40 | 5 | 20 | 9 | 7 | 32 | 12 | 1 | 0 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bound / error, 4+2 | 1,960 | 168 | 91 | 75 | 72 | 51 | 39 | 38 | 35 | 35 | 24 | 21 | 21 | 20 |
| bound / error, 4+4 | 8.6 | 4.6 | 4.0 | 3.9 | 3.9 | 3.6 | 3.4 | 3.4 | 3.1 | 3.3 | 2.8 | 2.8 | 2.7 | 2.7 |
| σ_w, 6-bit | 0.49 | 0.30 | 0.25 | 0.26 | 0.23 | 0.22 | 0.19 | 0.20 | 0.19 | 0.19 | 0.17 | 0.14 | 0.13 | 0.15 |

- **Llama-3.1-8B needs nothing extra:** part 8 is within 8× the error in every layer with the single store.
- **Qwen3-30B-A3B has 14 loose layers at 32K, not the 2 that the 1.6K maps flagged (L3, L8).** The tail's actual error
  in them is small: 0.01–0.39× FP8's, and at most 0.07× in 12 of the 14 (`depth_r7b.md`). The certificate is what
  fails.
- **The 4+4 store fixes every one of them.** The bound falls to at most 8.6× the error, and 69–100% of head-steps are
  certified within FP8 at the vote's read set alone.
- **The slack grows exponentially in the score noise.** Over all 160 (layer, width) cases, uniform 6/4 and 8/4 on both
  models:

  ln(bound / error) = 0.51 + 13.9·σ_w, with R² = 0.987 and a largest residual of 0.65.

  This is §12's diagnosis at scale (theory.md). Each unread row's weight is known only up to e^{zσ}, with the union's
  z = √(2 ln(C·m/δ)) ≈ 6.3 at C = 32K, m = 16 sets and δ = 10⁻³, and it is paid twice: e^{2zσ} = e^{12.7σ}.
  - Since z grows like √ln C, the slack should grow only slowly with context (z ≈ 6.6 at 128K).
  - Each extra key bit halves σ_w.
- **A rule for the certified read:** give 8-bit keys to the layers where part 8 at the store's width exceeds about 20×
  the error, that is σ_w ≳ 0.15 at 6-bit keys. Measure it on calibration prompts at the run's length; one forward pass
  and one dither draw suffice (`nested_s1h5.py depth`).
  - For Qwen3-30B-A3B at 32K that is the 14 layers above: +0.016 traffic at the vote, and about +0.02 of FP16 K+V in
    memory.
  - The confirmatory arm (KL non-inferiority) needs no certificate, so it keeps 4+2 everywhere.
