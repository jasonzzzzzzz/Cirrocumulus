# Findings: the ctx sweep separates the confound — and the axes come apart cleanly

Data: 20 valid (model, ctx) points, all post probe-fix, all PG-19, all passing the
input-validity gate (head-level basis; these runs predate the task-decode
extension, so task-level reads are truncation-limited and ignored).

- `job19960531` llama31-8b at 4k/8k/16k/32k/64k/128k
- `job19960567` qwen3-30b-2507 at 8k/16k/32k (+128k from `job19960863`)
- `job19959945`+`job19960025` all six models at 8k
- `job19959944`+`job19960027` all six models at 32k
- `job19960861`+`job19960863` the registry-ctx campaign

Reproducibility: llama31-8b@128k measured in two independent jobs: 12.5% vs
12.8% in band; qwen3-30b@32k: 6.2% vs 6.4%. (Same prompts by construction —
haystack is seeded on prompt_idx — so this checks pipeline determinism, not
sampling error.)

Report: `reports/h0_report_ctx_pooled.pdf`.

## 1. ctx is a first-class axis: one model crosses all three verdicts

llama31-8b, nothing varied but context length:

| ctx | band % | gain vs eviction | n95 (abs tokens) | dead-2 % | verdict |
|---|---|---|---|---|---|
| 4k | 46.3 | 1.85x | ~105 | 14.4 | **GO** |
| 8k | 39.3 | 1.68x | ~148 | 18.9 | **GO** |
| 16k | 33.0 | 1.41x | ~249 | 23.4 | NARROW |
| 32k | 25.0 | 1.19x | ~279 | 26.5 | NARROW |
| 64k | 17.5 | 1.07x | ~380 | 30.3 | NARROW |
| 128k | 12.5 | 1.03x | ~1260 | 53.9 | **STOP** |

The mechanism is exactly the φ-window prediction from `analysis_from_fable.md`,
now confirmed causally: a head's absolute attention support grows sublinearly in
L (105 → 380 tokens while L grows 16x), so φ = n95/L falls, the eviction corner
— which keeps a fixed *fraction* — gets relatively cheaper, and gain-vs-eviction
slides from 1.85x to 1.03x. The interior doesn't stop working; the corner
catches up.

ctx is a first-class axis — one model crosses all three verdicts on its own. llama31-8b: GO at 4k (46.3%) → NARROW at 16–64k → STOP at 128k (12.5%). And the mechanism is exactly the φ-window prediction from analysis_from_fable.md, now confirmed causally: its absolute attention support grows sublinearly (~105 → ~380 tokens while L grows 16×), so the fraction-keeping eviction corner gets mechanically cheaper — gain-vs-eviction slides 1.85× → 1.03×. The interior doesn't stop working; the corner catches up.

## 2. At matched ctx, architecture ordering is perfectly stable

Same six models, same context, rank identical at 8k and at 32k:

| model | 8k | 32k | dead-2 @32k |
|---|---|---|---|
| llama33-70b | **79.1** (GO) | **58.5** (GO) | 9.2 |
| mistral-7b | 51.3 (GO) | 41.5 (GO) | 20.1 |
| llama31-8b | 39.3 (GO) | 25.0 (NARROW) | 26.5 |
| qwen15-moe | 29.4 (NARROW) | 20.1 (NARROW) | 37.1 |
| qwen3-30b | 10.1 (STOP) | 6.4 (STOP) | 65.8 |
| qwen3-8b | 9.1 (STOP) | 6.1 (STOP) | 57.5 |

So band ≈ architecture-offset minus a shared ctx-slope: ctx moves everyone down
the same ladder, architecture decides where you start.

At matched ctx, architecture ordering is perfectly stable. The six-model ranking is identical at 8k and 32k: llama33-70b > mistral > llama31-8b > qwen15-moe > qwen3-30b > qwen3-8b. So band ≈ architecture offset − shared ctx slope. The two axes separate.

## 3. "Bigger → less benefit" is not just dead — it is REVERSED

At matched ctx, llama33-70B sits far ABOVE llama31-8B (79 vs 39 at 8k, 59 vs 25
at 32k). The original trend was two confounds stacked: the registry compared the
70B at 128k against small models at 32-41k (ctx effect), on top of the
causal-mask bug (which compressed everything toward diffuse). Within the Qwen3
family, size does ~nothing (6.4 vs 6.1 at 32k). The honest claim: **scale helps
or is neutral at matched ctx; context length is what hurts.**

At matched ctx the 70B is the best model (79.1% at 8k, 58.5% at 32k — both GO), roughly double llama31-8B. The registry table that showed 70B@128k below mistral@32k had the ordering inverted by the ctx confound. Within Qwen3, size does nothing (6.4 vs 6.1). Honest claim: scale helps or is neutral; context is what hurts.

## 4. Qwen3's phase is architectural; Llama's is ctx-driven

The qwen3-30b sweep barely moves (10.3 → 4.1 over 16x ctx) and its dead-2
fraction is 60-73% at EVERY length — it is pinned in the evict corner by its
sink-heavy attention (median top-1 0.46-0.51), not pushed there by long context.
llama31-8b's steep slope (46 → 12.5) is the opposite: a model whose phase is set
by ctx. Two different reasons to be out of band, distinguishable on the same
axis.

Two distinct ways to be out of band, distinguishable on one axis. Qwen3-30b's sweep barely moves (10.3 → 4.1 over 16× ctx) with dead-2 at 60–73% everywhere — pinned in the evict corner architecturally (sink attention, top-1 ≈ 0.5). Llama's steep slope is ctx-driven. Same corner, different causes, and the data tells them apart.

## 5. The dead-2-tier axis unifies both effects

Pooled over all 20 points — sweeps and matched-ctx runs together —
**Spearman(dead-2 fraction, band) = −0.964**, against −0.728 for ladder width.
One derived, L-free quantity predicts the band fraction regardless of whether
you moved along the ctx axis or across architectures. The 39-75% region that the
first campaign left unconstrained is now filled (points at 48, 54, 57, 60, 61,
63, 66); the STOP boundary sits near dead-2 ≈ 45-50%, GO territory below ≈ 20%.
(`page_phase`'s hatch is now data-driven accordingly.)

The insight that survives everything: dead-2-tier fraction is the phase variable. Pooled over all 20 points — sweeps and matched runs together — Spearman(dead-2, band) = −0.964, vs −0.728 for ladder width. One L-free, derived quantity predicts the band whether you moved along ctx or across architectures. And the 39–75% region the first campaign left hatched as "unconstrained" is now filled with six points; the STOP boundary sits near dead-2 ≈ 45–50%. I made page_phase's hatch data-driven, since the hardcoded band is now factually wrong for any report that includes these runs.

## Overall H0 conclusion

The productive band is real: 2/6 architectures are GO at 32k, and everything
non-Qwen3 is GO at 8k. But it shrinks with context by a shared, mechanistic
slope (the fractional-budget eviction corner), and at 128k only llama33-70b
retains NARROW. Two consequences for SIEVE: the router must condition on
(architecture, ctx) — a per-model verdict is ill-posed; and the strongest
version of the contribution is the phase diagram itself, with dead-2-tier
fraction as the boundary variable — it is derived from τ²c_b vs c₀=1, L-free,
and now supported by 20 points at ρ = −0.96. The fractional eviction budget
(B·L/maxb tokens) is also worth revisiting: heads' absolute support grows
sublinearly, so a fraction-of-L budget gets mechanically more generous with L —
an absolute-support formulation is the natural v3 question (deferred item in
plan.md).
