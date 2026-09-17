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



# The reframing

φ is dead, and it failed worse than τ did. This is the headline correction. On the 20 valid points, φ = n₉₅/L correlates with the band at +0.26 — the wrong sign. It fit six clustered points in v6 and inverted the moment you populated the context axis, because φ divides by L and so conflates architecture with the context it was measured at. I've kept it in both docs as a labelled negative result, with an inset in the figure, because the pattern is the methodological argument: an index chosen for fit gets overturned; one derived from the cost comparison doesn't.

The replacement was already in the theory. The dead-tier fraction — heads where τ²c_b exceeds the derived c₀ = 1 — is L-free, derived rather than fitted, and predicts the band at ρ = −0.96 across both axes. It's the same comparison that generates the phase boundary, so the boundary variable and the boundary now come from one equation.

Context length is promoted to a first-class contribution. The causal chain is fully measured: τ rises monotonically with L in all six models (within-model rank correlation 1.00) → rungs die from the bottom → the ladder degenerates toward the two-tier solution that is eviction. Gain over uniform actually grows with context (23×→33×); the interior isn't getting worse, the corner is catching up. Both docs now say plainly that a per-model verdict is ill-posed — which invalidates how the literature reports results, including our own prior versions.

Your v3 design question is now a first-class contribution, not a caveat. I quantified it: eviction keeps ∝L tokens while support grows as L^0.63–0.92, so the corner is handed 15.8× a head's support at 4k and 70.8× at 64k. It's framed as contribution #6 and as E1, the highest-priority experiment, with both outcomes explicitly publishable. llama3.3-70B is noted as the exception that proves the mechanism — its n₉₅ grows superlinearly (L^1.81), it genuinely diffuses rather than being squeezed by the budget definition, and it's the only model still holding NARROW at 128k.

# Two things got better, not worse

Re-verifying the headline claims on valid data: C4 tightened from 3.4% to 1.4% worst-case (median 0.4%) over 3× the configurations, and the linearization gap closed from 0.93–4.89 to 0.80–1.35 — the 4.9× overstatement was a bug artifact, not a property of the theory. Scoring moved up accordingly (novelty 7.5→8.0, empirical 7.5→8.0, rigor recovered to 7.5).

# Next steps

E1 (absolute-support budget, 2 days) is the decision point — it determines whether acceptance sits at ~55% or ~70%, and nothing else is close in information-per-GPU-hour. Then E2 practical evictor (2 days, and its upside is largest exactly where you now lose), E5 completing the sweep grid for the remaining four models (2 days, closes the central claim's evidence base), then the 3-bit base layer, GQA union, and the kernel.

Two notes. I made the measurement-integrity work an explicit contribution rather than hiding it — a reviewer's first question about a 33,920-head attention study is "how do you know your probe is right," and "we broke it three times, here's each bisect and regression test" is a strong answer. And fig5_phase.png was a broken reference in both docs even before this; it's now generated by make_fig_phase.py so it regenerates with each campaign instead of drifting.



---

# Correction (2026-09-17): the τ/ln2 result is an identity, not a prediction

Every version of this project through proposal v8 reported "ladder width =
τ/ln2, confirmed to 1.4–1.5%" and called it **the strongest single piece of
evidence in the paper**. It is algebra.

`ladder_bits_a_only = std_i(log₂ aᵢ)`, and `log₂ aᵢ = sᵢ/ln2 − log₂Z` where `Z`
is constant in `i`. A constant offset does not change a standard deviation, so

```
std_i(log₂ aᵢ) = std_i(sᵢ)/ln2 = τ/ln2      exactly
```

Measured on the real campaign, per head, all 24 configurations:

```
max relative error    8.0e-16
median of medians     1.3e-16
```

That is float noise. The former Fig 5-right plotted x against x.

**Why the "1.5% error" was not zero.** The headline compared τ/ln2 against
`ladder_bits`, which includes the value term `‖vᵢ−o‖`. That term is the *only*
empirical content in the comparison — and the same documents measure it at
**≤0.035 bits** and call it "decorative". So the advertised agreement was
quantifying the size of a quantity we simultaneously described as negligible.
The two ladders agree to 0.40–1.17% across the campaign, which is exactly the
1.5% that was being reported as confirmation of a closed form.

**What replaced it.** `lin_ratio3` — the linearized cost model against the
exactly-recomputed gain — **0.79–1.34× on per-configuration medians** (per-head
p10–p90 envelope 0.39–2.58×). This is a genuine forward prediction: it was free
to be wrong by 5×, and was (0.93–4.89) before the probe fix. It is also
corner-independent, being a property of the allocator's cost function rather
than of any baseline.

**What did not change.** No measurement, no band fraction, no phase-variable
correlation. τ/ln2 remains the correct way to *compute* ladder width, and the
router that reads ladder width as its signal (ρ = −0.41…−0.77 in 24/24) is
unaffected — a signal does not need to be a prediction to be useful.

**Pinned so it cannot recur:** `sievelib/alloc.py` carries the derivation as a
comment at the ladder computation, and
`tests/test_units.py::test_ladder_identity` asserts both halves — that the
identity holds to float precision, and that the value term (its only empirical
content) is small. If someone re-promotes this to evidence, the test's own
message says why not.
