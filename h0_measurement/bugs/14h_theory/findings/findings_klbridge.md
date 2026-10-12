# The KL bridge, water-filling allocation and run-time targets (CPU, 2026-10-11)

RESEARCH_PLAN.md §3 KL, WF, τ; theory.md §14. Code: `../e2e_s1h5.py` (the end-to-end harness), `../klbridge_s1h5.py`
(the analysis). Raw runs: `klbridge_runs/`.

## Setup
- **Model.** Llama-3.2-1B on CPU, all 16 layers. Qwen3-1.7B is in §5.
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
