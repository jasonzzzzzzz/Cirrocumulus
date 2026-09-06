# Plan: an honest eviction corner — **EXECUTED**

> **Status: complete.** The work landed, the campaign ran (24 configurations, six
> models, 8k–128k), and the outcomes are in `report.md`. This file is kept as the
> record of what was planned and **how the plan differed from what the data
> said** — two of its predictions were wrong, and that is the useful part.
>
> - `fix.md` — the engineering record: what changed, the interface, how to reproduce.
> - `report.md` — the results, the mechanism findings, and the design rules.

---

## What was planned, and what happened

| item | plan | outcome |
|---|---|---|
| **P0** unbreak `practical_score` | prerequisite for everything | **done** — the guard `pa.numel() >= sh.numel()` was false on essentially every step, so the practical corner had never run in any campaign |
| **E2** pluggable evictors, oracle kept | widen the band where the oracle flattered itself | **done, and it is what rescued the paper** — every STOP verdict disappeared, band +6 to +29 points across all 24 cells |
| **E1** fractional vs absolute-support budget | "decides whether the 128k story is real" | **done, result NEGATIVE** — see below |
| **E2b** the practical *interior* | deferred at plan time | **still deferred**, and still the honest gap |

---

## The two places the plan was wrong

Both are worth reading before trusting a similar argument again.

### 1. "This is provably monotone" — it is not

The plan's acceptance criterion said:

> gain_best_practical = min(e_uniform, e_practical)/e_wf ≥ gain_best, since a
> lagged-attention evictor can only be worse than the oracle. […] Any cell that
> falls indicates a bug.

**That is false per head, and must not be used as a bug detector.** The `oracle`
corner is an oracle only with respect to the *first-order proxy*
`w² = (a·‖v−o‖)²`, while the reported error is exact recomputation — `alloc.py`
keeps those strictly separate by design. Ranking by the proxy is not the argmin of
the exact error, so a differently-ranked corner can land on a better kept set.
Measured: a practical corner beats the oracle on **15.9% of head-rows**, and the
oracle even loses to *uniform* on 0.1%.

The direction holds decisively in aggregate, which is the claim to make. Two test
assertions encoding the false per-head claim were removed; they had passed only
because one synthetic draw happened to align.

### 2. The budget was not the mis-specification — the *scoring rule* was

The plan argued that eviction's fixed-fraction budget (`B·L/maxb`, linear in L)
hands the corner steadily more slack than any head can use, and that this
"plausibly accounts for its apparent dominance at 128k".

**Tested and refuted.** K\* — the smallest keep-count landing within 10% of the
full-budget corner's error — is **100% of the budget in 24 of 24 runs**, 8k
through 128k. The corner needs every token it is given; there is no slack. The
proposed `abs` policy at κ=4 is not a cheaper corner but a broken one, costing
1.9–7.4× the error.

The artifact was E2's, not E1's: the corner's *scoring rule* was an oracle. Fixing
that removed every STOP verdict; fixing the budget would have removed none.

**E1 was still worth running.** Its diagnostic returned a better finding than its
hypothesis: **K\*/n₉₅ spans 37×** across the model set (5.8 for mistral-7b to 214.7
for llama33-70b) and is *not* ordered by support. That is the number that says a
per-head keep-count cannot be a global constant.

### A third thing the plan asserted, also backwards

`why.md` (and the plan's framing) expected the oracle's advantage to be largest on
**sharp** heads — "a lagged score that misses one heavy-hitter is catastrophic".
The data says the opposite in **24 of 24 runs**: `spearman(oracle_advantage, n₉₅)`
is positive everywhere (+0.19 to +0.68), and the diffuse quartile shows a larger
advantage than the sharp quartile in every run. On a sharp head the heavy hitter
is *stable across steps*, so lagged attention identifies it perfectly.

---

## What the plan did not anticipate: P1

Adding a second practical evictor made a latent defect reachable. The verdict
corner is `min` over the practical evictors, but they do not become available at
the same time — `recency` needs no history, `accum` and `window` do. On decode
step 0 the `min` collapsed onto the weakest corner, inflating the band by ~29
points on average and turning the band-vs-context curve **non-monotone**.

Found because an earlier campaign that happened to lack `recency` disagreed by 29
points and agreed with the filtered data to 0.5. Fixed with a completeness guard;
see `fix.md` §1 (P1).

---

## Still open

**E2b — the practical interior.** After E2 the comparison is still asymmetric: the
interior waterfills on oracle sensitivity while only the corner has been demoted.
The symmetric cell builds `w2p` from the lagged score (`ō = pa @ V`, computable
from cache), waterfills on `w2p`, and evaluates on true logits — decision lagged,
evaluation honest, the same asymmetry a deployed system faces. `practical_scores`
is the plumbing it needs. **If the interior's edge survives with both sides
practical, that is the strongest GO this framework can produce.**

Everything else the plan listed under "Rerun + reporting" has shipped: report.py
populates the practical columns, prints the per-evictor `oracle_evict_advantage`,
draws the corner grid and K\* panels, and names its corner on the verdict line.
`dead-2` remains the phase variable and did not move with the corner change
(ρ = −0.983 oracle → −0.953 practical), which was the check the plan asked for.
