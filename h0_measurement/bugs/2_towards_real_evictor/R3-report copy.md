# R3 — the symmetric cell: results

> ## ⚠ PROVISIONAL — every w2p-derived number below must be re-measured
>
> A post-hoc audit (2026-09-18) found a defect in **this analysis's own
> instrumentation**, not in the runs it analysed.
>
> `Evictor.score()` applies a *"never evict a token at birth"* rule that rewrites
> freshly-appended positions to `max + 1`. That is **ordinal** — correct for a
> corner, which consumes a ranking. The R3 interior consumes the same tensor as a
> **magnitude**: it normalises it into a distribution to build `w2p`. Measured on
> the real code path, the bump absorbs **25.7%** of the resulting distribution,
> and the share is larger on concentrated heads (**33% sharp vs 20% diffuse**)
> because `mx` is larger there.
>
> Concentrated heads are high-gain heads. **So the artifact is correlated with
> gain in exactly the direction of §3 finding 2**, which is this report's
> headline mechanism. I cannot separate the two with the job92* data.
>
> **Affected (do not cite):** every `gain_pp*` / `band_pp` / `interior_lag_cost`
> number — so §2's table, §3 findings 1, 2, 3, 4, and §4 entirely.
> **Unaffected:** §1 (the completeness audit), and the `band_or` / `band_pr`
> columns, which never touch `w2p`.
>
> Fixed by `Evictor.score(rank_bump=False)` + `quant_metrics(interior_raw=...)`,
> pinned by `test_units.py::test_practical_interior`. Re-run with
> `bash script.sh --r3`.


**One-line result: E2's headline was an artifact of asymmetry.** Demoting the
corner to a real evictor while leaving the interior on oracle sensitivity raised
every band fraction by 12–19 points. Demoting the interior as well takes all of
it back — and lands within ±5 points of where the *fully oracle* comparison
started. Two STOP verdicts return.

Figures: `R3-fig.png`, produced by `R3-figure.py`. Per-run numbers:
`R3-per-run.csv`.

---

## 1. What actually ran — read this first

**The campaigns in `script.sh` did not run as written.** None of the `SIEVE_*`
overrides reached the jobs; every task logged
`ctx=per-model evictors=per-config` and recorded the models.yaml defaults:

| specified in script.sh | what the parquets say |
|---|---|
| `SIEVE_CTX=8192` / `32768` (matched grid) | per-model registry ctx (32k / 128k) |
| `SIEVE_EVICTORS='oracle,accum'` | `oracle,last_step,accum,window,recency` (the default five) |
| campaign B: `lag:k=1,2,4,8`, `n_decode=16`, `quant_every=1` | **no lag columns at all** — never ran |

The most likely cause is that the multi-line `VAR=x \` continuations were split
on paste, so only the bare `sbatch ...` line executed. The symmetric cell was
measured anyway, but by luck: `interior_scores: [accum]` is the models.yaml
default that R3 added, so `gain_pp3_accum` is present in every run.

**Completeness.** 8 job directories, 10 parquet files, **4 distinct (model, ctx)
cells** — each replicated 2–3 times:

| model | ctx | runs | heads | corpus | needle retrieved |
|---|---|---|---|---|---|
| mistral-7b | 32k | 3 | 1,024 | PG-19 | 6/6 |
| qwen15-moe-a2.7b | 32k | 2 | 384 | PG-19 | 6/6 |
| llama31-8b | 128k | 2 | 1,024 | PG-19 | 6/6 |
| qwen3-30b-a3b-2507 | 128k | 3 | 1,536 | PG-19 | 4/4 |

Every run passes the input-validity gate: real text, full needle retrieval.

**Missing, and why:**
- **qwen3-8b** — `CANCELLED ... DUE TO TIME LIMIT` in `h0_929735_0`,
  `h0_929944_0`, `h0_929945_0`. The default five-corner set plus the new interior
  allocation overran `--time=01:15:00`. The script specified `oracle,accum`
  precisely to pay for the interior's extra `exact_error`; with that override
  lost, the job had both costs.
- **llama33-70b** — no parquet and no `h0_92*` log. The large-tier submissions
  appear not to have run.
- **the matched-ctx grid** — each model is at its own registry ctx, so the
  cross-model comparison below is confounded with context length exactly as
  `bugs/3` warned. It is still valid *within* each cell.
- **the staleness sweep (campaign B)** — entirely absent. The design question it
  was built to answer (allocate once vs re-budget) is **not** answered here.

**Is this sufficient?** For R3's primary question — does the verdict survive a
symmetric comparison — yes, on 4 of 6 models with replicates. For the ctx slope
of the symmetric cell, and for the staleness curve, no.

---

## 2. Results

All numbers are per-head medians over quantized, complete-corner rows, then
aggregated; replicates averaged.

| model | ctx | band, both oracle | band, real corner only (E2) | band, **both real** (R3) | verdict |
|---|---|---|---|---|---|
| mistral-7b | 32k | 41.1 | 60.2 | **45.2** | GO → **GO** |
| qwen15-moe-a2.7b | 32k | 22.4 | 37.2 | **20.8** | GO → **NARROW** |
| llama31-8b | 128k | 13.1 | 24.5 | **12.5** | NARROW → **STOP** |
| qwen3-30b-a3b-2507 | 128k | 4.2 | 14.4 | **4.7** | STOP → **STOP** |

Replicate spread on the symmetric band: **0.00–0.46 points** (sd over 2–3 runs).

Supporting numbers:

| model | lag cost (median) | lag cost (p90) | routed, E2 | routed, R3 | router miscalibration | dead-2 |
|---|---|---|---|---|---|---|
| mistral-7b | 1.18 | 1.88 | 2.46 | 1.84 | 15.0% | 18.6 |
| qwen15-moe | 1.04 | 1.93 | 1.85 | 1.44 | 16.4% | 37.8 |
| llama31-8b | 1.02 | 1.44 | 1.50 | 1.29 | 12.1% | 56.2 |
| qwen3-30b | 1.01 | 1.39 | 1.28 | 1.15 | 9.7% | 73.2 |

---

## 3. What I expected, and what happened

### Expected, and confirmed

**The symmetric band is below the half-demoted band, in aggregate.** It is, in
4/4 cells, by 10–16 points.

**The phase axis survives.** dead-2 orders the symmetric band perfectly
(18.6 → 45.2, 37.8 → 20.8, 56.2 → 12.5, 73.2 → 4.7; Spearman −1.00 on four
points). C1 has now survived two independent redefinitions of its own target.
Four points is weak evidence alone, but it is consistent with the 24-point
ρ = −0.945.

**The router-miscalibration number is real and sizeable** — 9.7–16.4% of all
heads are routed to the interior by an oracle-calibrated threshold and do not
pay for themselves once the interior is honest.

### Not expected

**1. Both *symmetric* comparisons agree; only the mixed one is inflated.**
This is the result I did not anticipate and it reframes E2.

```
both oracle      41.1   22.4   13.1    4.2
both lagged      45.2   20.8   12.5    4.7     <- differs by -1.6 to +4.9
corner-only      60.2   37.2   24.5   14.4     <- +12 to +19 above BOTH
```

The *level* of information barely matters as long as both sides have the same
amount. What moved the band was the **asymmetry**, and E2 was the asymmetric
cell. So E2's headline — "every STOP verdict disappears, band +6 to +29 points" —
does not survive: under a symmetric comparison two of these four cells are STOP,
and the band is within a few points of the original oracle-corner numbers.

E2 was still worth doing. It established that the corner's scoring rule was
*ours* to fix and produced the 1.02–1.41× price of decode-time-available
scoring. But the band movement it reported should be attributed to the
asymmetry it introduced, not to a gain.

**2. The lag cost lands almost entirely on the heads that matter.**
I predicted a roughly uniform toll and wrote "cost(k) flat → allocate once" into
the decision table. The median lag cost is 1.01–1.18×, which reads as *nearly
free* — and that reading is wrong:

| | lag cost |
|---|---|
| out-of-band heads (gain < 2×) | **1.00 – 1.03×** |
| in-band heads (gain ≥ 2×) | **1.36 – 1.58×** |
| ρ(lag cost, gain) | **+0.81 to +0.94**, all four models |

The distribution is heavily skewed: p50 = 1.04, p90 = 1.64, p99 = 3.65, max 19.9;
13.3% of heads pay more than 1.5×.

**Why this makes sense in hindsight.** A head is in-band precisely because its
sensitivity is concentrated somewhere a flat allocation cannot exploit — and that
concentration is what *moves between decode steps*. Heads where lagged scoring is
free are heads whose sensitivity is flat enough that allocation was pointless
anyway. **The interior's advantage and its dependence on current-query
information are the same property.** That is why the median is the wrong summary:
it averages over heads that were never going to use the interior.

This is the mirror image of E2's finding, and the two now fit together. E2 found
the *corner's* oracle advantage was largest on **diffuse** heads, because on a
sharp head the heavy hitter is stable across steps and lagged attention finds it
perfectly. R3 finds the *interior's* lag cost is largest on **high-gain** heads
for the same underlying reason: stability across steps is what lagged scoring
needs, and it is exactly what the interior's best heads lack.

**3. Ranking the corner by `w2p` changes almost nothing.** `gain_pp_sym`
(corner ranked by the same lagged sensitivity the interior uses) versus
`gain_pp` (corner ranked by raw lagged attention): 44.5 vs 45.2, 20.8 vs 20.8,
12.4 vs 12.5, 4.4 vs 4.7. This column was built to isolate "is the interior's
edge mixed-precision *shape*, or score information the corner lacks?" The answer
is that the corner's *ranking* is not the lever at all — consistent with
`report.md` §6.6, "scoring is solved; allocation is where the value is."

**4. My asserted per-head guarantee is false — and it is the second time.**
The script and the unit test both claimed `gain_pp ≤ gain_best_practical` holds
*per head*, "because here it is the ALLOCATION being restricted, not the
ranking". Measured: **13.8% of head-rows violate it** (4.1% mistral, 12.7%
llama31-8b, 16.7% qwen15-moe, 20.2% qwen3-30b).

The reason is the one `plan.md` already records for the corner version of this
claim: `waterfill` minimises the **first-order proxy** `w²`, while the reported
error is **exact recomputation**, and `alloc.py` keeps those separate by design.
An allocation chosen by a different proxy can land on a better exact-error
allocation. The project has now made this mistake twice, on the ranking and on
the allocation.

**The general rule, which should go in the paper's methods section: in this
framework "strictly less information" is never a per-head bound, only an
aggregate tendency.** Both false assertions passed their unit tests because a
single synthetic head happened to satisfy them — the same failure mode `plan.md`
documents for the first pair. The test now asserts the aggregate direction only.

**5. Free error bars.** Replicates were meant to be identical (prompts are seeded
on `prompt_idx`), but GPU nondeterminism makes per-head values differ by up to
80%. Aggregates nonetheless reproduce to **0.00–0.46 band points**. That is a
tighter reproducibility claim than the project has had, and it arrived by
accident. It also independently supports finding 2: per-head values are noisy,
aggregate statistics are not, and the band's fragility lives in the threshold.

---

## 4. What this means for the design

**Re-budgeting during decode is load-bearing, not optional.** The median lag cost
said "allocate once, it is nearly free"; the conditional cost says the opposite
for every head the router actually sends to the interior. A prefill-only
allocator gives up 1.4–1.6× on exactly its best heads.

**Calibrate the router threshold on the honest gain.** 9.7–16.4% of heads are
mis-routed otherwise. This is actionable today and costs nothing.

**Stop reporting the band fraction as the headline.** A 1.0–1.6× change in one
input moves it 12–19 points; both symmetric comparisons agree on the *error
ratios* while the thresholded count swings. R2 already moved the figure to routed
gain — the text should follow.

---

## 5. What to run next

1. **Campaign B, the staleness sweep** — unrun, and it is the experiment that
   decides the architecture. Finding 2 makes it more important, not less: if the
   lag cost is already 1.4–1.6× at k=1 on in-band heads, the curve in k is what
   says whether re-budgeting every step is required or every N.
   Use `SIEVE_*` from a **script file**, not pasted continuations.
2. **qwen3-8b and llama33-70b** — raise `--time` or pass the intended
   `SIEVE_EVICTORS='oracle,accum'`; the timeout was the five-corner default plus
   the new interior cost.
3. **The matched-ctx grid** — every cell here is at its own registry ctx, so the
   cross-model column confounds architecture with context.
4. **Re-examine E2's writeup in the proposal and pitch.** The claim "no
   configuration remains STOP" is currently in both documents and does not
   survive §3 finding 1.
