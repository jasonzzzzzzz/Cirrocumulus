# R13 · Scope the K-channel axis — plan (frozen before any R13 output)

Written 2026-09-27. ROADMAP §5 R13: "We study the token axis ... Per-channel K
treatment is a different axis and is orthogonal and composable with ours. State
that explicitly in §1. The optional upside is to ask whether the channel-side
ladder has its own dead tiers, which would extend C1 to a second axis."

This plan runs the optional measurement. Its job is to decide which §1 sentence
is true, not to add a contribution.

## 1. The theory's prediction (written before the run)

Token axis (C1): dropping token i costs `w2_i * 1`, and a b-bit key costs
`w2_i * sig2_b`, where `sig2_b` is the absolute logit-noise variance. That
variance grows with τ, but the eviction cost is capped at 1 because softmax
saturates. A tier is **dead** when `sig2_b > c0 = 1`. dead-2 is the order
parameter.

Channel axis: the logit error is `δ_i = scale · Σ_c q_c ε_ic`. Dropping channel c
(zeroing it, or replacing it by its mean, which is the same thing under softmax
shift-invariance) costs `q_c² Σ_i w_i (k_ic − μ_c)²`. A b-bit channel costs
`q_c² Σ_i w_i (ε_ic(b) − ε̄_c)²`: every cost is centered, because a
per-channel constant error is a softmax-invariant shift. Both carry the same `q_c²` and the same channel
scale, so **nothing caps the zero-rate cost**. Whether a rung is dead
(`cost(b) ≥ cost(0)`) then depends only on how well the quantizer fits the
channel's marginal (outliers, skew), not on τ or L.

Predictions:
- **P1:** raw per-channel min-max quantization has a dead 1-bit rung on
  outlier channels. Rungs ≥2 are essentially never dead.
- **P2:** the channel dead fraction does not move with L (8K→32K), while the
  token axis's dead-2 does.
- **P3:** TurboQuant's random rotation largely equalises channel variance, so
  channel allocation in the rotated basis buys much less than in the raw basis.

## 2. What is measured (all arms in ONE process per cell)

Per (prompt, family, decode step, layer, KV head) we capture post-RoPE `q` for
every query head and `K`, `V` from the cache, as `run_h0` does. Nothing is
compared across processes (ROADMAP methods rule).

Tier set on both axes: `{0,1,2,3,4,5,6,8}`. Budgets: B = 2, 3, 4 average bits.

| family | basis | quantizer | tier 0 means |
|---|---|---|---|
| `ch` | raw post-RoPE channels | per-channel asymmetric min-max uniform over token groups of 128 (KIVI-style) | replace the channel by its group mean |
| `rc` | TurboQuant rotated coordinates `k' = K Rᵀ` | per-token norm γ, Lloyd-Max per coordinate, **no** norm correction (so costs decompose per coordinate) | replace the coordinate by its token mean |
| `t` | production TurboQuant (`quant.quantize_keys`, norm-corrected) | per-token tiers | evict the token |

Arms per family and budget. Every error is exact recomputation: real quantized
keys, real softmax, `‖ô − o‖/‖o‖`.

- `ch_u`, `rc_u`, `t_u`: uniform B.
- `ch_wf`, `rc_wf`: an oracle channel water-fill per KV head. It uses the
  group-summed, `‖o_h‖²`-normalised weights `rw_hi` (as in co-design `_rel`) and
  the current query. The cost is
  `cost_c(b) = Σ_h q_hc² Σ_i rw_hi ε_ic(b)²`, solved by Lagrangian bisection plus
  a greedy top-up to spend exactly `B·d` where the hull allows it.
- `ch_cal`, `rc_cal`: same-prompt reference (near-oracle, see A2). The allocation is fitted on decode step 1 and
  frozen, then evaluated on steps 3, 5, 7. It is L-free, since the channel count
  does not grow.
- `ch_xcal`, `rc_xcal` (A2): deployable across prompts. One allocation per
  (layer, KV head, basis, B) is fitted on a disjoint calibration prompt (2100,
  `cont`; costs summed over steps 1, 3, 5, 7), then applied to every evaluation
  prompt and family.
- `ch_ks`, `rc_ks`: deployable, query-free. It water-fills on the key statistics
  `Σ_i ε_ic(b)²` alone.
- `t_wf`: the token-axis group water-fill (`alloc.waterfill_group` on `_rel`
  weights, the per-head measured `sig2`). This is the paper's axis and the
  positive control.

Dead-rung statistics:
- **Channel, C1 sense:** the fraction of (KV head, channel) units with
  `cost_c(b) ≥ cost_c(0)` under the oracle weights, for b = 1, 2, 3, in both
  bases.
- **Token, C1 sense:** the fraction of query heads with `sig2_b > 1` (dead-b).
- **Allocator sense (R10):** the share of units the water-fill assigns to each
  rung. A rung is "unused" at ≤0.5%.

Metadata is not counted in any family's budget. KIVI's per-group scale and zero
point cost about 0.25 bit/element; TurboQuant's per-token norm costs about 0.125.
All verdicts compare arms within a family, so this does not enter any gate.
Absolute errors across families are reported for context only.

## 3. Cells

Prompt block 2000–2003 is fresh: every earlier campaign used indices below
1,200. Families are `niah,qa,cont`, with 4 prompts. `n_decode=8`, calibration
at step 1, evaluation at steps 3, 5, 7. Greedy decoding, `rot_seed=2`.

| cell | model | ctx | n_rep | role |
|---|---|---|---|---|
| 0 | llama31-8b | 8192 | 4 | Q2 pair |
| 1 | llama31-8b | 32768 | 4 | Q2 pair |
| 2 | qwen3-8b | 8192 | 4 | Q2 pair (QK-norm) |
| 3 | qwen3-8b | 32768 | 4 | Q2 pair (QK-norm) |
| 4 | mistral-7b | 8192 | 4 | third dense GQA |
| 5 | qwen15-moe-a2.7b | 8192 | 1 | n_rep = 1 control |

The pilot (excluded) is qwen3-1.7b @2048, 1 prompt, `cont`. It checks only
that the implementation runs and passes V1–V3.

## 4. Validity gates (a failure means `invalid_r13`, not a verdict)

- **V1:** `rc_u` at every B reproduces `quantize_keys(norm_correct=False)`
  logits (max |Δs| ≤ 1e-3·max|s|). Checked in-run on the first measured layer.
- **V2:** `ch` at 8 bits has a relative key error below 1%.
- **V3:** every water-fill spends ≤ B·d, and the median spend is ≥ B − 0.05.
- **V4:** L2 capture fidelity passes, as in `run_h0`.
- **V5:** all 6 main cells are COMPLETE, each with ≥1,000 evaluation head rows.

## 5. Decision table (frozen)

Pooled statistics are medians over evaluation head rows. Dead fractions are
pooled over units in a cell.

| Q | question | verdicts |
|---|---|---|
| Q1 | does the raw channel ladder have C1-dead rungs? | **C1_DEAD_CH** if raw dead-2 ≥ 5% in ≥3/6 cells; **ONLY_1BIT_DEAD** if dead-1 ≥ 5% but dead-2 < 5% in ≥3/6 cells; **NO_DEAD_CH** otherwise |
| Q2 | is channel deadness L-driven (an order parameter)? | for llama31-8b and qwen3-8b, 8K→32K: **L_DRIVEN** if the channel dead-2 (amended A3) moves ≥3 pts in the same direction in both; **L_FREE** if \|Δ\| < 1.5 pts in both; **MIXED** otherwise. The positive control is token dead-2 moving ≥3 pts in at least one model; if it does not, report `control_flat` |
| Q3 | does channel allocation pay in the raw basis? | best deployable (`max(ch_xcal, ch_ks)`, amended A2) median gain over `ch_u` at B=2: **PAYS** ≥ 1.20 in ≥3/6 cells; **NEGLIGIBLE** < 1.05 in all; **MARGINAL** otherwise |
| Q4 | does the channel axis survive the rotation? | `rc_xcal` (amended A2) median gain over `rc_u` at B=3: **SURVIVES** ≥ 1.10 in ≥3/6 cells; **CONSUMED** < 1.03 in all; **MARGINAL** otherwise |
| Q5 | is the channel axis orthogonal to ours? | Spearman over (layer, KV head) of median log `ch_wf` gain vs median log `t_wf` gain, B=3: **ORTHOGONAL** if \|ρ\| < 0.3 in ≥4/6 cells; **COUPLED** otherwise |

The §1 sentence depends on the outcome:
- **Q4 = CONSUMED:** the rotated quantizer already spends the channel axis.
  Per-channel methods (KIVI/KVQuant) are an *alternative* treatment of the same
  outliers, not something stacked on top of TurboQuant. The sentence must say
  so.
- **Q4 = SURVIVES:** the axes compose as per-coordinate offsets on top of token
  tiers. "Orthogonal and composable" stands as written.
- **Q1/Q2** decide whether any C1-style extension to the channel axis may be
  claimed. It may be claimed only for C1_DEAD_CH + L_DRIVEN.

## 6. Amendments (2026-09-27, before any main or pilot output)

These were made during a code review, after the first submission (jobs
995473–995476) had been queued but before any of it started. Those jobs were
cancelled unrun. The only R13 output seen before the review was a CPU smoke
test: qwen3-1.7b at 512 tokens, 1 prompt, excluded by construction. It exists
to exercise the code, not to measure. Every amendment either fixes an
implementation error or makes a gate stricter.

- **A1: centered costs.** `channel_costs` summed raw ε². A per-channel constant
  error is a softmax-invariant logit shift and costs nothing. The token side
  already drops it (`noise_model` uses Var(δ)). Uncentered, every b>0 tier in
  the rotated basis is charged for the gap between the Lloyd levels (centered on
  0) and the coordinate's mean, which biases both `rc` allocation and `rc`
  deadness. Costs are now the `rw`-weighted variance of ε (query-free `ks`: the
  unweighted variance), in float64. Pinned by `test_channel_costs_forms`, which
  checks shift invariance and that tier 0 equals zeroing.
- **A2: a cross-prompt calibrated arm `xcal`; Q3 and Q4 now read it.** The `cal`
  arm is fitted on step 1 of the *same* sequence and evaluated 2–6 tokens later,
  so it is close to the oracle and is not the deployment question. `xcal` sums
  the costs of probe steps 1, 3, 5, 7 on one disjoint prompt (2100, `cont`, the
  same ctx) and allocates once per (layer, KV head, basis, B). That allocation
  is applied unchanged to every evaluation prompt and family. `cal` is kept and
  reported as the same-prompt reference. Q3 now reads `max(ch_xcal, ch_ks)` and
  Q4 reads `rc_xcal`. This is stricter than before.
- **A3: Q2 is read on dead-2, not dead-1.** 1-bit min-max quantization puts both
  levels at the group's extremes, so its dead-1 is near-saturated by
  construction and cannot move with L. dead-2 is also the rung C1 uses on the
  token side, which makes Q2 a symmetric comparison. The Q2 thresholds are
  unchanged. dead-1 and dead-3 are still reported.
- **A4: strict dead criterion.** A rung is dead iff `cost(b) > cost(0)`, as on
  the token side (`sig2_b > 1`). A zero-cost tie, e.g. a constant channel, is
  not a dead rung.
- **A5: masks.** Decode masks are read as HF writes them: hidden positions are
  `finfo.min`, which is finite. Positions below −1e4 are dropped, and the
  visible entries must be one constant, otherwise the task stops. Previously a
  masked position would have aborted the task rather than being dropped.
- **Wall time:** main tasks get 5 h, up from 3 h.
