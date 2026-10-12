# The KL bridge, water-filling allocation and run-time targets (CPU, 2026-10-11)

RESEARCH_PLAN.md §3 KL, WF, τ; theory.md §14. Code: `../e2e_s1h5.py` (the end-to-end harness), `../klbridge_s1h5.py`
(the analysis). Raw runs: `klbridge_runs/`.

## Setup
- **Model.** Llama-3.2-1B on CPU, all 16 layers. Qwen3-1.7B is in §5. §6 (2026-10-12) has the single store with
  the certified read on both models, how to calibrate τ, and the FP8-quantile rule end to end.
- **Units.** 1,600 tokens each: 1,536 context rows, a 32-token question and a 32-token answer. There are 8 units:
  - 4 texts: PG-19, and three LongBench v2 contexts (a code repository, a table, an agent log);
  - × 2 non-overlapping windows. Window 0 is for calibration, window 1 is held out.
- **The pass.**
  - One exact prefill of [context | question].
  - Each arm then runs the answer span, teacher-forced, with its attention at the answer positions in the layers it
    configures.
  - KL(FP ‖ arm) per answer token, in the same process.
- **The vote.** The question's exact queries over the 4-bit Lloyd-Max sketch (the nested store's base). It reads the
  top 1/8 per KV head.
- **Draws.** 2 dither seeds per arm, and an independent stream per layer.

## 1. What makes the KL: three facts
- **The KL is quadratic in the error.** Isotropic Gaussian noise at 3× the size gives 8.3–11.8× the KL (9 if exactly
  quadratic). This holds in each of the 16 layers and every unit.
- **Layers nearly add.** At 6/4 the single-layer KLs sum to 0.82–1.64× the all-layer KL (median 1.17). The excess is
  mild sub-additivity, and it is the main reason the bridge errs high.
- **Key and value errors nearly add, and key errors cost more.**
  - Keys-only (6 bits) plus values-only (4 bits) gives 0.93–1.49× the full 6/4 design's KL (median 1.06).
  - Per unit of MSE, a value error costs a median of 0.46× the KL of a key error (c^V / c^K, p10–p90 0.14–0.96, over
    layers and units).
  - At 6/4 the keys carry 32% of Theorem M's MSE but 57% of the KL (medians).

## 2. The bridge: predicted / measured KL per token
Columns:
- **structured:** ½ Σ_l (c_l^K M_l^K + c_l^V M_l^V). M comes from Theorem M (one FP pass, no draws of the configuration).
  c^K and c^V come from dithering only that layer's keys (or values) at 4 bits.
- **isotropic:** ½ Σ_l c_l M_l, with c_l from isotropic noise.
- **tail-shaped, realized:** ½ Σ_l c_l^tail × the configuration's realized per-layer MSE, with c^tail from the 6/4 design
  in that layer alone.

Each cell is the median over the 8 units, with the 10th–90th percentile.

| configuration | structured | isotropic | tail-shaped, realized |
|---|---|---|---|
| tail 4/4 | 1.29 (1.01–1.66) | 0.66 (0.14–2.52) | 0.70 (0.26–0.99) |
| tail 5/4 | 1.31 (1.08–2.61) | 0.73 (0.39–3.69) | 0.97 (0.61–1.28) |
| tail 6/4 | 1.21 (0.83–1.49) | 0.76 (0.31–5.00) | 1.18 (0.76–1.62) |
| tail 6/6 | 1.06 (0.77–1.67) | 0.50 (0.15–1.73) | 0.63 (0.38–0.98) |
| tail 8/4 | 1.01 (0.57–1.64) | 1.09 (0.35–6.09) | 1.37 (0.91–2.35) |
| tail 8/8 | 1.03 (0.23–1.96) | 0.47 (0.06–2.64) | 0.67 (0.16–0.92) |
| nested 4+2/4 | 1.25 (0.92–2.44) | 0.82 (0.53–3.05) | 1.19 (0.74–1.80) |
| nested 4+3/4 | 1.00 (0.80–1.55) | 1.06 (0.43–4.65) | 1.22 (0.97–2.30) |
| keys only, 4 | 1.34 (1.04–1.86) | 0.64 (0.13–2.26) | 0.66 (0.25–0.97) |
| keys only, 6 | 1.09 (0.70–1.62) | 0.45 (0.14–1.67) | 0.58 (0.35–0.90) |
| values only, 4 | 1.08 (0.81–1.42) | 1.23 (0.52–6.11) | 1.44 (1.27–2.86) |
| dense 4/4 | — | — | 0.75 (0.22–1.28) |
| dense 6/6 | — | — | 0.67 (0.25–1.46) |
| FP8 | — | 0.61 (0.22–1.11) | 0.83 (0.21–1.63) |

- **The structured bridge works.** From one FP pass and 32 single-layer calibration runs, it predicts every tail
  configuration's KL with a median ratio of 1.00–1.34, erring on the safe side.
  - This covers 4/4 to 8/8, the nested store, keys only and values only.
  - The 10th–90th percentile is within 0.6–2.6. The widest is 8/8, whose KL is near 10⁻⁶ and noisy across two seeds.
- **The isotropic bridge does not transfer across inputs** (0.06–6.1). The error's shape matters, and the sensitivities
  must be measured with the design's own errors.
- **FP8 and dense stores** are only roughly predicted by tail-shaped sensitivities (0.2–1.6×). FP8's deterministic
  errors are 4–5× more harmful per unit MSE than the tail's on the code input.

## 3. Water-filling allocation (Corollary WF)
- **The surface:** each layer's predicted KL at every width in {4, 5, 6, 8}² (keys, values), at the 1/8 read. It is
  averaged over the calibration units (window 0).
- **The allocation:** the per-layer Lagrangian choice.
- **Tested** on the held-out units (window 1) against uniform widths and the FP8-target allocation. The FP8-target
  allocation is the cheapest per-layer width whose per-head error is ≤ FP8's for 90% of head-steps, taking the per-layer
  maximum over the calibration units (`findings_bits.md`).

Mean KL per token over the 4 units; in brackets, the geometric mean of KL over uniform 6/4's. The first two pairs of
columns use the raw average; the last pair weights every input equally.

| allocation | traffic | raw average, in sample | raw average, held out | equal weights, in sample | equal weights, held out |
|---|---|---|---|---|---|
| uniform 6/4 | 0.398 | 1.39e-4 (1.00) | 1.20e-4 (1.00) | 1.39e-4 (1.00) | 1.20e-4 (1.00) |
| water-filled at uniform 6/4's predicted KL | 0.383 (raw) / 0.398 (equal) | 1.44e-4 (1.36) | 1.90e-4 (1.67) | 9.2e-5 (0.76) | 1.01e-4 (0.84) |
| water-filled at 0.371 | 0.369 / 0.371 | 2.31e-4 (2.07) | 2.91e-4 (2.69) | 2.09e-4 (1.84) | 2.49e-4 (2.03) |
| water-filled at 0.36 | 0.359 | 5.05e-4 (4.22) | 3.98e-4 (3.68) | 4.10e-4 (3.71) | 4.71e-4 (4.39) |
| uniform 5/4 | 0.371 | 2.74e-4 (2.40) | 2.51e-4 (2.29) | 2.74e-4 (2.40) | 2.51e-4 (2.29) |
| FP8-target (per-head p90) | 0.386 | 2.00e-4 (1.69) | 2.10e-4 (1.80) | 2.00e-4 (1.69) | 2.10e-4 (1.80) |

- **The raw average does not transfer.** It is dominated by the hardest inputs (table and agent have about 10× the KL).
  In sample it matches uniform 6/4's mean KL at 0.383 traffic. Held out, it is 1.6× worse.
- **With every input weighted equally, it transfers.**
  - Each unit's surface is first divided by its own predicted uniform 6/4 KL.
  - At the design's own traffic (0.398) the held-out KL is 16% lower than uniform 6/4's, against 18% predicted.
  - At 0.371 it matches uniform 5/4 on mean KL and is 11% better in geometric mean.
- **The FP8-target allocation is worse than uniform** at equal traffic (1.7–1.8× uniform 6/4's KL at 0.386), both in and
  out of sample.
- **The gains are modest on this key-heavy model:** at most about 20% KL at equal traffic, or about 0.015 traffic at
  equal KL. The per-layer sensitivities move with the input. So an allocation must be calibrated on the deployment's
  input mix, weighting inputs equally.
- **For R7b's exploratory per-layer arm:**
  - expect gains of this size, not a step change;
  - calibrate on many prompts per family;
  - keep a uniform 5/4 control.

## 4. Run-time targets for the certified read (τ)

**The certified read (theory.md §6e).**
- It walks a nested coin-free schedule (theory5_s1h5.FINE: the vote's 1/8, then rows by the base sketch's priority).
- It stops each (KV head, position) at the first set whose certificate meets a target τ for every query head of the
  group. The certificate is part 8's bound, hkz+exact, at δ = 10⁻³ split over the sets.
- Until now τ was FP8's per-head error: an oracle, because it needs the exact rows.

**Rules compared** (all but the oracle are computable at run time):
- **oracle_m:** m × FP8's per-head error.
- **fp8q:** one number per layer, a quantile of FP8's per-head error over the calibration units.
- **klb:** a KL budget B split equally over (layer, head) components with the layer's sensitivity: τ_l² = 2B / (L·H·c_l).
  By Proposition KL, if every head's error is within τ_l the predicted KL is at most B.
- **rel:** τ = ρ·|õ|, relative to the design's own output.

**Material.** 6-bit keys and 4-bit values. Every set's bound and realized error, per layer, on every 4th answer position
(`e2e_s1h5.py --plan tau`).

**Each rule's parameter is swept.** The table gives the least held-out traffic (window 1) at which the predicted KL,
½ Σ_l c_l × the realized MSE (c from the calibration units), falls below each level.

| predicted KL ≤ | 10⁻⁶ | 3·10⁻⁶ | 10⁻⁵ | 3·10⁻⁵ | 10⁻⁴ |
|---|---|---|---|---|---|
| klb (KL budget) | **0.461** | **0.418** | **0.402** | **0.399** | 0.398 |
| fp8q (per-layer FP8 quantile) | 0.462 | 0.423 | 0.409 | 0.399 | 0.399 |
| oracle_m (m × FP8's per-head error) | — | 0.447 | 0.406 | 0.400 | 0.399 |
| rel (ρ |õ|) | 0.485 | 0.436 | 0.411 | 0.399 | 0.398 |

The vote alone (no certified read) has traffic 0.398 and predicted KL 2.1·10⁻⁴.

- **The KL-budget τ is the best rule at every level, and it is computable at run time.** It beats even the oracle
  target: FP8's per-head error is the wrong currency, because heads differ in how much their error moves the logits.
  §6.3 refines this. Scored with each held-out unit's own sensitivities, a flat τ ties the KL budget on Llama-1B, and
  the KL budget's edge appears on Qwen3-1.7B (1.35–1.4×, §6.2).
- **The certified read buys large KL reductions for almost no traffic.**
  - It reads extra rows only where the certificate flags a large error in a sensitive head.
  - Predicted KL falls 7× (to 3·10⁻⁵) at 0.399 and 20× (to 10⁻⁵) at 0.402.
- **The old oracle target (m = 1)** costs 0.475 for a predicted KL of 10⁻⁶. That is far more accuracy than needed: the
  certificate is about 8× the error, so demanding FP8's per-head error drives the actual error to about an eighth of it.

**End to end** (the certified read inside the forward pass, `cert` arm; held-out units, 2 seeds):

KL-budget targets for predicted KL levels of 3·10⁻⁵ and 10⁻⁵, with B taken from the frontier (calibration c_l). Cells
are KL per token (the mean of 2 seeds), with KL over uniform 6/4's in brackets; the last row gives traffic.

| unit | uniform 6/4 | certified, budget 3·10⁻⁵ | certified, budget 10⁻⁵ |
|---|---|---|---|
| PG-19 | 7.5e-5 | 3.9e-5 (0.53) | 2.0e-5 (0.27) |
| code | 3.3e-5 | 1.2e-5 (0.36) | 6.3e-6 (0.19) |
| table | 3.0e-4 | 6.9e-5 (0.23) | 9.2e-6 (0.03; 1 seed) |
| agent | 7.3e-5 | 9.0e-6 (0.12) | 1.3e-6 (0.02) |
| **mean** | **1.2e-4** | **3.2e-5 (geometric mean 0.27)** | **9.2e-6 (geometric mean 0.073)** |
| traffic | 0.3984 | 0.3988 | 0.4018 |

- **The certified read meets its budget end to end.** Measured 3.2·10⁻⁵ against 2.9·10⁻⁵ predicted, and 9.2·10⁻⁶
  against 7.9·10⁻⁶: within 15%.
- **It cuts the 6/4 design's KL 3.7× for +0.0004 traffic, and 14× for +0.003** (geometric means over held-out inputs).
  FP8's KL on these units is about 4·10⁻⁴.
- **It works as a certified, query-aware patch to the vote.**
  - The schedule's first steps add the base sketch's top unread rows for the current answer query.
  - The certificate fires only on the head-steps whose error is large in a sensitive head. These are mostly rows the
    question's vote missed.
  - So 0.1–0.9% more rows remove most of the KL.
- **Caveat.** One table seed at the tighter budget hit the CPU-time limit, so that cell has 1 seed.

## 5. A second architecture: Qwen3-1.7B
**Setup.** 28 layers; QK-norm; noisy early layers (σ ≈ 0.5 at 6-bit keys, ≈ 2 at 4). 4 units: PG-19 and code × 2
windows.

**End to end** (medians over units):
- FP8's KL is 39.5× the 6/4 design's (12–66×);
- dense 6/6 is 15× the design's;
- the nested store at 4+2 is 1.13× uniform 6/4's (0.96–1.59), and at 4+3 it is 0.39×;
- uniform 4-bit keys are 0.46× FP8's.

**The structured bridge** (predicted / measured, median over units):

| configuration | structured | tail-shaped, realized |
|---|---|---|
| tail 6/4 | 0.73 (0.54–0.85) | 0.67 |
| tail 6/6 | 0.82 (0.37–1.23) | 0.65 |
| tail 8/4 | 0.83 (0.78–0.96) | 1.23 |
| tail 8/8 | 0.79 (0.53–1.01) | 0.58 |
| nested 4+2/4 | 0.96 (0.63–1.34) | 0.74 |
| nested 4+3/4 | 0.95 (0.79–1.13) | 1.11 |
| values only, 4 | 0.83 (0.81–0.97) | 1.24 |
| keys only, 6 | 0.81 (0.41–0.97) | 0.62 |
| tail 5/4 | 1.87 (0.83–16.6) | 0.74 |
| tail 4/4 | **about 10⁶** | 0.76 |

- **With keys of ≥ 5 bits, the bridge holds** at a median ratio of 0.73–0.96, erring low here.
  - Qwen's layers compound: the single-layer KLs sum to only 0.55–0.90× the all-layer KL (median 0.63).
  - Llama's were mildly sub-additive (1.17).
- **At 4-bit keys the structured bridge fails.**
  - Theorem M's first-order form divides by the exact normalizer instead of the random one.
  - Where σ ≈ 2 a few rows' E Z² is astronomically large, while the design's own normalization keeps the actual error
    bounded.
  - This is theory.md §10's "first-order limit", now seen end to end.
  - The realized-error version still works (0.76), and E‖N‖²/M² remains a valid bound.
  - A second-order Theorem M, or the realized MSE, is needed wherever σ ≳ 1.
- **The asymmetry holds here too.** c^V / c^K has a median of 0.62 (0.24–1.36). At 6/4 the keys carry 28% of the MSE but
  70% of the KL.

## 6. The deployed combination, calibrating τ, and the FP8-quantile rule (CPU, 2026-10-12)
RESEARCH_PLAN.md §8, inputs 2–4. Runs are in `klbridge_runs/`; summaries are in `klbridge_runs/checks_summary.json`,
`calib_<model>.json` and `calibfront_<model>.json` (`../klbridge_s1h5.py calib | calibfront | cvar`). The heavier runs
went to one Trillium CPU node.

### 6.1 The single store with the certified read (input 2)
**The combination.** The design R7b deploys, plus E1's certified read:
- the single store, nested 4+2/4 (`findings_nested.md`);
- R6's question-pass vote over that store, with its dithered values (`findings_nested.md` §6);
- the certified read with the KL-budget τ, using part 8's bound on the nested tier (Theorem N). This is the `certn` arm
  of `../e2e_s1h5.py`.

**Calibration.** τ_l comes from the window-0 units' sensitivities, with §4's two budgets B = 0.178 and 0.0056. On
Llama-1B these are the frontier points for predicted KL 3·10⁻⁵ and 10⁻⁵ (the uniform tier). Held-out window-1 units,
2 seeds each.

Llama-3.2-1B (KL per token):

| unit | FP8 | R6 design | single store | certified, uniform tier, B = 0.178 | certified, single store, B = 0.178 | certified, uniform tier, B = 0.0056 | certified, single store, B = 0.0056 |
|---|---|---|---|---|---|---|---|
| PG-19 | 4.1e-4 | 6.8e-5 | 1.1e-4 | 3.9e-5 | 3.6e-5 | 2.0e-5 | 1.4e-5 |
| code | 6.3e-4 | 3.9e-5 | 4.9e-5 | 1.2e-5 | 1.3e-5 | 6.3e-6 | 6.4e-6 |
| table | 4.5e-4 | 3.2e-4 | 3.5e-4 | 6.9e-5 | 5.0e-5 | 9.5e-6 | 6.9e-6 |
| agent | 3.9e-4 | 7.5e-5 | 1.1e-4 | 9.0e-6 | 8.4e-6 | 1.3e-6 | 9.1e-7 |
| **mean** | **4.7e-4** | **1.3e-4** | **1.5e-4** | **3.2e-5** | **2.7e-5** | **9.3e-6** | **6.9e-6** |
| traffic | 0.50 | 0.398 | 0.398 | 0.3988 | 0.3991 | 0.4018 | 0.4049 |

The R6 design and single-store columns are 8-seed means (`findings_nested.md` §6). The uniform-tier certified read
uses the exact-query vote (§4).

- **The deployed combination works on Llama-1B.** Against the single store alone (geometric means over units):
  - at B = 0.178: 5.6× lower KL for +0.001 traffic, 4.2× below R6's design, and 22× below FP8;
  - at B = 0.0056: 25× lower KL for +0.007 traffic, 18× below R6's design, and 96× below FP8.
- **The single store does at least as well as the uniform tier here.**
  - Its certificate is about 1.2× looser (`findings_nested.md` §3), so at the same τ it reads a few more rows: +0.0003
    and +0.003 traffic.
  - It ends at the same or lower KL: 0.84× and 0.75× the uniform tier's.
  - The single store's higher vote-only KL (1.2×) disappears once the certificate patches the head-steps with the
    largest errors.

Qwen3-1.7B (c_l from its window-0 units, the same B):

| unit | FP8 | R6 design | single store | certified, single store, B = 0.178 | certified, single store, B = 0.0056 |
|---|---|---|---|---|---|
| PG-19 | 2.6e-3 | 2.2e-4 | 1.6e-4 | 3.3e-5 (0.448) | 1.4e-5 (0.488) |
| code | 1.1e-3 | 2.5e-5 | 3.7e-5 | 7.9e-6 (0.440) | 3.6e-6 (0.472) |
| **mean** | **1.9e-3** | **1.2e-4** | **9.8e-5** | **2.1e-5 (0.444)** | **8.9e-6 (0.480)** |

Traffic is in parentheses.
- **The KL falls 4.7× and 11× below the single store's.** That is 105× and 238× below FP8's.
- **The traffic rises by 0.046 and 0.082.** That is a large cost, and almost all of it comes from the first six layers
  and layer 10 (§6.2).

### 6.2 Qwen's noisy layers: a vacuous certificate, and 8-bit keys as the remedy
**Where the rows go.** Rows read per layer at B = 0.0056:
- layer 0: 0.89, against the vote's 0.125;
- layers 1–5: 0.25–0.83;
- layer 10: 0.32–0.49;
- every layer from 18 on: the vote's 0.125.

**The cause is the certificate, not the error.**
- In layer 0, part 8's median bound on the vote set is 20–75, while FP8's median per-head error is 0.002.
- From the certified-read material (`--plan tau`, uniform 6/4), 39–55% of layer 0's head-steps never certify at any
  useful target. Even at B = 100 they fall back to the exact read; so do 3–6% of layer 5's.
- That alone is a floor of about 2% extra rows (+0.014 traffic) under every target rule.
- These are the layers where Theorem M's map gives the score noise σ ≥ 0.25 at 6-bit keys (0.50 in layer 0, 0.46 in
  layer 5; `layermap_qwen3_1p7b.json`).
- The tail's actual error there is moderate: Theorem M's tail / FP8 RMS is 0.07–3.0, about 0.008 per head in layer 0.
  So the bound overstates it by more than 10³.
- The bound grows quickly with the score noise: its coin moments grow like e^{k²σ_i²/2} per row, so a few noisy rows can
  dominate it. Which rows do so in layer 0 is not yet checked.

**The remedy.** Give the layers with σ ≥ 0.25 (0–5 and 10) 8-bit keys, a 4+4 single store. σ falls 4×, to 0.12 or
less. The traffic cost is 2 key bits in 7 of 28 layers, +0.014 at the vote.

Qwen3-1.7B, held-out units, 2 seeds each. "8-bit noisy" is the single store with 4+4 keys in layers 0–5 and 10 and
4+2 elsewhere:

| arm | traffic | KL per token | PG-19 | code |
|---|---|---|---|---|
| single store, 6-bit keys, vote only | 0.398 | 9.8e-5 | 1.6e-4 | 3.7e-5 |
| 8-bit noisy, vote only | 0.412 | 6.6e-5 | 1.1e-4 | 2.2e-5 |
| 6-bit, certified, KL budget B = 0.178 | 0.444 | 2.1e-5 | 3.3e-5 | 7.9e-6 |
| **8-bit noisy, certified, B = 0.178** | **0.413** | **2.3e-5** | 3.8e-5 | 8.0e-6 |
| 6-bit, certified, B = 0.0056 | 0.480 | 8.9e-6 | 1.4e-5 | 3.6e-6 |
| **8-bit noisy, certified, B = 0.0056** | **0.421** | **1.3e-5** | 2.1e-5 | 3.8e-6 |
| 8-bit noisy, certified, flat τ = 4.0 | 0.413 | 3.1e-5 | 5.4e-5 | 7.6e-6 |
| 8-bit noisy, certified, flat τ = 1.41 | 0.416 | 2.3e-5 | 4.0e-5 | 5.2e-6 |
| 8-bit noisy, certified, flat τ = 0.38 | 0.433 | 1.0e-5 | 1.8e-5 | 1.9e-6 |
| FP8 | 0.50 | 1.9e-3 | 2.6e-3 | 1.1e-3 |

- **With 8-bit keys in the noisy layers the certificate works.**
  - Layer 0 reads 14–23% of its rows instead of 89%; layers 1–5 and 10 stay at the vote's 12.5%.
  - The certified read costs +0.001 and +0.009 traffic over the store alone, for 2.9× and 5.3× lower KL (ratios of
    means).
  - Against the 6-bit store's certified read: 1.1× the KL at 0.413 instead of 0.444, and 1.4× the KL at 0.421 instead
    of 0.480. At about 10⁻⁵ the saving is about 0.045 traffic.
  - The 8-bit keys alone cut the vote-only KL 1.5× for +0.014.
- **Here the KL-budget shape beats a flat τ.** At matched traffic the flat τ's KL is 1.35× (at 0.413) and about 1.4×
  (at 0.421, interpolated) the KL budget's.
  - The flat τ spends rows in late layers: 15, 17, 22 and 27 read 13–34%, where Qwen's sensitivities are 20–3,300×
    smaller than layer 0's.
  - The KL budget reads more in layer 0 and leaves layers 18–27 at the vote.
  - Unlike Llama-1B (§6.4), Qwen's sensitivities fall steeply with depth, so the shape matters.
- **For R7b's models**, at 32K on every layer (R7b's captures; `findings_nested.md` §7):
  - Llama-3.1-8B needs nothing: part 8 is 3.6–7.6× the error everywhere with the single store;
  - Qwen3-30B-A3B has 14 loose layers (20–1,960×), not the 2 that the 1.6K maps flagged. The 4+4 store brings every
    one within 8.6×, for +0.016 traffic at the vote.
  - Part 8's slack grows like e^{14σ_w} (R² 0.99), so the criterion is σ_w ≳ 0.15 at 6-bit keys, or directly a bound
    above about 20× the error.

### 6.3 Calibrating τ (input 3)
**The sensitivities move with the input.**
- Llama-1B, 8 units: per layer, c_l spans 9–90× over units. The sd of log c_l has a median of 1.09 over layers, and 81%
  of its variance lies between texts rather than between windows of the same text.
- Qwen3-1.7B, 4 units: the sd of log c_l is 0.47 and 25% of it lies between texts (2 texts only).

**Shape against level.** A target rule has a shape (τ_l across layers) and a level (one global knob).
- The shape decides how efficiently rows are spent.
- The level decides which KL a given input lands at.

The two are tested separately, each on held-out units. Each unit's KL is predicted with its own sensitivities (the
bridge, §2). End to end, the certified read's KL was within 5–30% of its prediction even with the calibration units'
sensitivities (§4, §6.4).

**Shape.** Each rule's knob is swept, and the table gives the mean held-out KL at matched mean extra rows read (beyond
the vote's 1/8). Calibration is on the other texts' units (leave the family out).

| rule (shape) | Llama: 0.05% | 0.2% | 1% | Qwen: 5% | 10% | 20% |
|---|---|---|---|---|---|---|
| each unit's own c_l (the best a sensitivity table could do) | 2.3e-5 | 1.1e-5 | 4.3e-6 | 1.5e-5 | 5.1e-6 | 7.4e-7 |
| KL budget, c_l from the other texts | 2.8e-5 | 1.4e-5 | 5.7e-6 | 1.7e-5 | 5.9e-6 | 8.5e-7 |
| KL budget, c_l from the same text's other window | 2.6e-5 | 1.3e-5 | 5.1e-6 | 1.6e-5 | 6.0e-6 | 8.8e-7 |
| **flat: one τ for every layer (no calibration)** | **2.7e-5** | **1.4e-5** | **5.3e-6** | **1.2e-5** | **4.7e-6** | **1.2e-6** |
| FP8 scale: m × the layer's median FP8 per-head error | 2.7e-5 | 1.5e-5 | 6.7e-6 | 2.9e-5 | 1.1e-5 | 1.2e-6 |
| FP8 quantile (fp8q) | — | 1.8e-5 | 7.9e-6 | — | 9.5e-6 | — |

On Qwen every rule reads at least 2% extra rows (§6.2), so its columns start at 5%. "—" means the rule cannot reach
that level: fp8q's loosest point is q = 1, the calibration maximum.

- **On Llama-1B a flat τ is as good as the calibrated KL-budget shape** (within 5%), and end to end too (§6.4).
- **On Qwen the offline comparison misleads.** On the uniform 6-bit material, flat looks better at 5–10% extra rows
  (0.74–0.79×) and worse at 20% (1.4×). That regime is dominated by the noisy layers' exact fallbacks (§6.2). End to end,
  on the store with 8-bit noisy layers, the KL budget beats flat by 1.35–1.4× (§6.2).
- **Even perfect sensitivities would buy little.**
  - The KL-budget shape built from each held-out unit's own c_l, the best a sensitivity table could give, is 10–25%
    better than the calibrated one.
  - That gain does not transfer: c_l from other inputs gives back all of it.
  - On Qwen at 5–10% extra rows, even that shape is worse than flat. The rule τ_l ∝ c_l^(−1/2) is not the optimal
    allocation, because the bound's slack and the rows each set costs vary by layer.
- **Shapes built on FP8's error profile are worse.**
  - On Llama, the FP8 scale equals the KL budget at 0.05% extra rows and is 1.17× worse at 1%. The FP8 quantile is
    1.25–1.38× worse.
  - On Qwen, both are about 2× worse than flat, and 1.6–1.8× worse than the KL budget.
  - FP8's per-head error says how far FP8 is from exact, not how much a head's error moves the logits.
- So §4's ranking was optimistic for the KL budget. Its frontier predicted the test units' KL with the calibration
  units' c_l, the same c_l the rule is built from.

**Level.** The KL budget's B is fitted so that the calibration units' mean predicted KL meets a level. The table gives
the held-out KL over the level: the median over held-out units (10–90%), and the mean.

| calibration set | Llama, 3·10⁻⁵ | Llama, 10⁻⁵ | Qwen, 3·10⁻⁵ | Qwen, 10⁻⁵ |
|---|---|---|---|---|
| the unit itself (the oracle) | 0.99 (0.83–1.00); 0.95 | 1.00 (0.96–1.00); 0.99 | 0.72 (0.42–1.00); 0.71 | 1.00; 1.00 |
| the same text's other window (1 unit) | 1.20 (0.52–1.65); 1.14 | 1.29 (0.68–2.15); 1.24 | 0.53 (0.42–2.27); 0.94 | 1.12 (0.77–1.58); 1.15 |
| the other texts (leave the family out) | 0.67 (0.12–5.34); 1.46 | 0.86 (0.16–4.06); 1.42 | 0.77 (0.31–4.97); 1.70 | 1.29 (0.35–4.35); 1.82 |
| k random other units: k = 1 / 3 / 7 (mean) | 1.95 / 1.45 / 1.19 | 2.62 / 1.40 / 1.14 | 1.46 / 1.65 / — | 1.69 / 1.29 / — |

Qwen's oracle at 3·10⁻⁵ is below 1 because of §6.2's floor: the least read already overshoots the target.
- **The level must be calibrated per family.**
  - The KL a given τ reaches differs up to about 10× between texts.
  - Calibrating on other families misses a held-out unit by 0.12–5.3×, and by 1.4–1.8× on average over units.
  - One unit of the same family gets within 0.4–2.3× per unit, and within 0.94–1.24× on average.
- **Individual inputs scatter about ±2× even with same-family calibration.** More calibration prompts shrink the error
  of the family's mean, not this scatter.
  - Pooling k units across families, the held-out mean falls from about 2–2.6× the level at k = 1 to 1.14–1.19× at
    k = 7.
  - The within-family scatter is about σ_log ≈ 0.45 per unit. So a family mean over N test units, calibrated on k
    prompts, should land within about 0.45·√(1/k + 1/N) in log KL: about ±20% at k = 8 and N = 30.
  - The two windows of one text are likely more alike than two prompts of one task family, so at scale the per-family
    scatter is probably larger. Plan on at least 8–10 calibration prompts per family.
- **The FP8 quantile needs no sensitivities, and its level is stable.**
  - At fixed q ≤ 0.99, the held-out KL varies by no more than about 1.2× across calibration protocols (own, same text,
    other texts, k = 1–7).
  - At q = 1 it varies 1.5×, because the calibration maximum grows with the calibration set.
  - It does not target a KL level, though. On Llama at q = 0.99 it lands at 7–9·10⁻⁶ with 0.4–0.7% extra rows; on Qwen
    at 8–9·10⁻⁶ with 9–10%.

### 6.4 The FP8-quantile rule and a flat τ, end to end (input 4)
All arms here use the uniform 6/4 tier and the exact-query vote, as §4's end-to-end runs. The rules are matched in
traffic:
- the FP8 quantile at q = 1 and q = 0.95 (the frontier's points for predicted 3·10⁻⁵ and 10⁻⁵);
- the KL budget at B = 0.0316 and 0.001, its frontier points at the same traffic;
- a flat τ of 0.1625 and 0.0294, set to the KL budget's predicted extra rows on these units.

Llama-3.2-1B, 4 held-out units × 2 seeds:

| rule | traffic | KL per token (mean) | predicted | PG-19 | code | table | agent |
|---|---|---|---|---|---|---|---|
| KL budget, B = 0.178 (§4) | 0.3988 | 3.2e-5 | 2.9e-5 | 3.9e-5 | 1.2e-5 | 6.9e-5 | 9.0e-6 |
| KL budget, B = 0.0316 | 0.3994 | 1.8e-5 | 1.6e-5 | 2.9e-5 | 1.0e-5 | 3.1e-5 | 3.1e-6 |
| FP8 quantile, q = 1 | 0.3994 | 1.8e-5 | 1.8e-5 | 2.9e-5 | 1.0e-5 | 3.1e-5 | 2.9e-6 |
| KL budget, B = 0.0056 (§4) | 0.4018 | 9.3e-6 | 7.9e-6 | 2.0e-5 | 6.3e-6 | 9.5e-6 | 1.3e-6 |
| FP8 quantile, q = 0.95 | 0.4090 | 6.4e-6 | 5.2e-6 | 1.4e-5 | 4.9e-6 | 5.8e-6 | 7.4e-7 |
| KL budget, B = 0.001 | 0.4105 | 4.2e-6 | 3.3e-6 | 9.2e-6 | 3.9e-6 | 3.4e-6 | 4.0e-7 |
| flat, τ = 0.1625 | 0.3995 | 1.7e-5 | 1.6e-5 | 2.8e-5 | 1.0e-5 | 2.5e-5 | 3.5e-6 |
| flat, τ = 0.0294 | 0.4104 | 4.5e-6 | 3.5e-6 | 1.0e-5 | 4.0e-6 | 3.4e-6 | 3.8e-7 |

"Predicted" is the frontier's prediction from the calibration units' c_l. Each flat τ is paired with the KL
budget at the same traffic: τ = 0.1625 with B = 0.0316, and τ = 0.0294 with B = 0.001.
- **A flat τ equals the KL budget end to end.**
  - At 0.3995 its KL is 0.91× the KL budget's on the mean (0.97× in geometric mean).
  - At 0.4104 it is 1.06× (1.02×).
  - Per unit the two read almost the same rows (traffic within 0.0004), so on Llama-1B the calibrated shape buys
    nothing.
- **The FP8 quantile holds up end to end.**
  - At 0.3994 it matches the KL budget exactly.
  - At 0.409 it has 1.33× the KL budget's KL (6.4·10⁻⁶ against 4.8·10⁻⁶, interpolated), as the frontier predicted.
  - Its predictions are as accurate as the KL budget's: measured / predicted 1.05–1.24, against 1.13–1.29.
- **Its limits:**
  - its loosest point is q = 1, the calibration maximum, so it cannot run in the cheapest regime (below about 0.1%
    extra rows);
  - q = 1 depends on the calibration set's size;
  - on Qwen its shape costs 1.6–1.8× against the KL budget (§6.3).

### 6.5 What R7b's E1 should use
**τ rule: the KL budget, τ_l² = 2B/(L·H·c_l).**
- **Calibration:** c_l is the design's own tail sensitivity: one single-layer arm per layer, on calibration prompts,
  pooled (the mean) across families.
  - Per-family c_l gains at most 10% (§6.3).
  - On Qwen, c_l from 2 prompts already gave the held-out gains. Use 5–10.
  - E3's water-filling needs c^K and c^V instead: 2L arms per prompt.
- **Why not a flat τ:** it equals the KL budget on Llama-1B, end to end, but is 1.35–1.4× worse on Qwen3-1.7B, whose
  sensitivities fall steeply with depth.
- **Why not the FP8 quantile:**
  - it needs no calibration, and on Llama-1B it matches the KL budget at the cheapest level;
  - but it is 1.33× worse at 1.5% extra rows, 1.6–1.8× worse in shape on Qwen (offline), and its loosest point
    (q = 1) depends on the calibration set's size;
  - it is the fallback if the calibration stage is cut.

**Level: two fixed budgets per model, no KL target.**
- Pick two B per model, before the test blocks, for about +0.002 and +0.01 traffic over the store. Use R7a's probe
  material or the calibration prompts.
- Report the two (traffic, KL) points.
- Do not claim a KL target. The KL a given B reaches varies about 10× across families, and individual prompts scatter
  about ±2× within one (§6.3). A target would need B fitted per family on at least 8–10 prompts each.

**Noisy layers: 8-bit keys.**
- Where part 8 at the store's width exceeds about 20× the error on calibration prompts (σ_w ≳ 0.15 at 6-bit keys),
  use a 4+4 store. Otherwise part 8 is vacuous there and the read falls back to exact reads: 0.444–0.480 traffic
  instead of 0.413–0.421 on Qwen3-1.7B (§6.2).
- For R7b, at 32K: Qwen3-30B-A3B's 14 loose layers (+0.016 traffic); Llama-3.1-8B has none (`findings_nested.md` §7).
  Recheck at 128K.
- This makes E1's store differ from the primary arm's in those layers. The alternative is the primary store, with the
  fallback reported.

**Stop rule:** per (KV head, step), every query head of the group must meet τ, because the group shares the rows read.
R7a's port should be checked for the same rule.

**Expected, from CPU:**
- Llama-1B: 5.6× and 25× below the single store's KL, at +0.001 and +0.007 traffic;
- Qwen3-1.7B, with 8-bit noisy layers: 2.9× and 5.3× below that store's KL, at +0.001 and +0.009.
