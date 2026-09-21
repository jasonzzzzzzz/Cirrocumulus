# Co-design waves 1–2: what 10 cells bought, across R4, R5 and R6

**Status 2026-09-20.** Waves 1 (`job214651*`, 8 cells) and 2 (`job214652*`,
2 cells) landed, together with the three decisive R4 cells (`job2144472*`).
Wave 3 is ready and its rationale has changed — §5. Wave 4 is still held: it
needs the co-design columns, which none of these cells carry.

This report covers only what these ten cells measured. The two new *designs*
(GQA-group allocation, cascade score) are in `plan.md`; nothing here tests them,
because waves 1–3 were deliberately submitted without those columns.

| what | where |
|---|---|
| R4's answer, rewritten | `../4_rope_limit_or_mechanism/report.md` §8–§9 |
| R5 campaign 2, 4 of 6 cells | §3 here; `../5_phase_drift_across_decode/report.md` still describes campaign 1 |
| R6 with 25 cells | §4 here |
| the co-design columns | `plan.md`, not measured yet |

---

## 0. The four things worth knowing

1. **R4 reverses.** On τ against a pre-registered curved null, the cap effect
   largely disappears; the best-sampled model sits *below* the null at its cap.
   The old evidence was dead-2 *rates*, which are a threshold artifact, and
   linear extrapolation on models with two points. (§2)
2. **R5's C4 question has a two-timescale answer.** The **route** survives 4,096
   tokens — p90 regret is **1.00 in all four cells**. The **allocation** does
   not: the prefill-time allocation costs **2.5–3.4×** what a one-step-old one
   does by the end of its valid horizon. Router offline, allocator re-budgeted.
   (§3)
3. **R5 campaign 1's headline was a loop.** The anti-loop knobs held (distinct-4
   0.71–0.98 at step 4,096, against 0.04–0.45 before), and with the loops gone
   the "τ drifts with content, not length" finding **disappears**: measured Δτ
   is now the same order as the across-ctx prediction. (§3.3)
4. **R6 cannot close the GO line by running cells.** Nine new cells turned its
   GO bracket from a *gap* into an **overlap** — different models on both sides
   at the same dead-2. That is architecture scatter, which more cells do not
   fix. The honest verdict is *pinned per model, not universal*. (§4)

---

## 1. What ran

All ten in R3's exact configuration (`or-ac_f`, `floor_maxb`, dense, 6 prompts
main / 4 large), so they pool with R3's 16 into one grid.

| id | cell | job | serves | rows |
|---|---|---|---|---|
| N1 | llama31-8b @131,072 DRIFT | 21465134 | R5 ★ | 43,008 |
| N2 | llama31-8b @8,192 DRIFT | 21465141 | R5 | 43,008 |
| N3 | qwen3-8b @8,192 DRIFT | 21465137 | R5 (2nd arch) | 48,384 |
| N4 | qwen3-8b @32,768 DRIFT | 21465138 | R5 (2nd arch) | 48,384 |
| N8 | qwen3-8b @16,384 | 21465135 | R6 ★, R4 | 55,296 |
| N9 | qwen3-8b @24,576 | 21465136 | R6 ★, R4 | 55,296 |
| N12 | llama31-8b @65,536 | 21465139 | R6, R4 | 49,152 |
| N13 | llama31-8b @98,304 (6 prompts) | 21465140 | R4 ★ control | 49,152 |
| N10 | llama31-8b @16,384 | 21465223 | R6, R4 | 49,152 |
| N15 | qwen3-30b @4,096 | 21465224 | R6 ★ | 49,152 |

**Validity — confirmed from `logs/h0_design_logs/`, not proxied.** Every niah
cell retrieved every needle: **6/6** on all five wave-1/2 main-tier cells, **4/4**
on qwen3-30b @4,096 (the cell flagged as the real gate risk, being below this
project's 8k floor), and **3/3, 2/2, 3/3** on the three R4 cells. The four DRIFT
cells are `cont`-only and have no needle; their gate is the text, which they
passed (§3.1).

---

## 2. R4 — the answer moved, and N13 is why we can trust it

Full treatment in `../4_rope_limit_or_mechanism/report.md` §8. The part that
belongs here is the control this sheet was built to provide.

**N13 = llama31-8b @98,304 at 6 prompts**, against the R4 cell's 3:

| | τ | dead-2 | n₉₅ | sym band |
|---|---|---|---|---|
| 3 prompts (job21444723) | 2.8112 | 40.66 | 750 | 29.90 |
| 6 prompts (N13) | 2.8084 | 42.19 | 655 | 27.52 |

**τ reproduces to 0.003** across a doubling of the sample; the band moves 2.4
points. So τ is the statistic R4 can safely pool across prompt counts, and the
band is not — exactly as R7 predicted (plan.md B3), now measured on the same
cell rather than inferred from sub-sampling.

That control is what licenses the R4 re-analysis: an excess of +0.20 in τ is
60× its reproducibility, so it is real; and llama31-8b's **−0.135** at its own
cap is real too, and points the other way.

N10/N12 gave llama31-8b six context points, which is what allowed a *quadratic*
null instead of a line — and the line is where the old "cap effect" came from.

---

## 3. R5 — the C4 answer, and a correction to campaign 1

Read with `drift.py` over N1–N4. Campaign 2 is 4 of its 6 cells; N17 (greedy)
and N18 (32k) are wave 3, and the two qwen3-30b cells were dropped for qwen3-8b
at a tenth of the GPU cost.

### 3.1 The anti-loop knobs held

distinct-4 of the last 256 generated tokens at step 4,096 (below ~0.5 is a loop):

| cell | campaign 1 | **campaign 2** |
|---|---|---|
| llama31-8b @8k | 0.04 (truncated) | **0.93** |
| llama31-8b @128k | 0.95 | **0.98** |
| qwen3-8b @8k | — | **0.79** |
| qwen3-8b @32k | — | **0.71** |

No cell truncated. `rep_penalty=1.05` + `no_repeat_ngram=8` are enough, and
wave 3's two cells need no re-tuning — which removes the only reason to hold
them back.

### 3.2 Two timescales, and they are different

| cell | flip floor | flip @4,096 | excess | **p90 regret** | ρ(gain, calib) | **froz / lag1** at the horizon |
|---|---|---|---|---|---|---|
| llama31-8b @8k | 23% | 25% | +3 | **1.00** | 0.63 | **3.03** (step 256) |
| llama31-8b @128k | 8% | 15% | +7 | **1.00** | 0.69 | **3.39** (step 4,096) |
| qwen3-8b @8k | 16% | 21% | +5 | **1.00** | 0.69 | **2.52** (step 256) |
| qwen3-8b @32k | 14% | 17% | +2 | **1.00** | 0.65 | **2.85** (step 1,024) |

**The route is a one-pass quantity.** Route flips run 2–7 points over a same-run
noise floor of 8–23%, and — the number that decides it — **p90 regret of keeping
the calibration route is 1.00 in every cell at step 4,096**. The flips land on
heads sitting near the 2× threshold, where being on the wrong side costs
essentially nothing. C4's "one offline calibration pass" now has a measurement
behind it instead of an assumption.

**The allocation is not.** `froz` — the prefill-time allocation applied t steps
later, which is the quantity C4 actually needs and which campaign 1 never
measured — costs **2.5–3.4× a one-step-old allocation** by the end of its valid
horizon, on both architectures and at every context. Rank correlation of gain
against calibration decays to 0.63–0.69.

**So the architecture is two-timescale:** the per-head *router* is calibrated
once, offline; the *allocator* inside a routed head must be re-budgeted on a
schedule. R3 §2.3 already priced the fast end (the band quarters within 8 steps);
`froz` prices the slow end. Both halves now have numbers, and they disagree by
design, not by accident.

### 3.3 Campaign 1's headline finding does not survive its own loops

Campaign 1 concluded: *"τ moves within a generation — but not by as much as
cache growth predicts, nor in proportion to it … the drift tracks the content
the model writes, not the length of its cache,"* on measured Δτ of 4–23× the
prediction. With loops suppressed:

| cell | cache growth | predicted Δτ | campaign 1 | **campaign 2** |
|---|---|---|---|---|
| llama31-8b @8k | +0.626 oct | +0.146 | +0.310 | **+0.208** |
| llama31-8b @128k | +0.048 | +0.011 | +0.248 | **−0.029** |
| qwen3-8b @8k | +0.626 | +0.170 | — | **+0.076** |
| qwen3-8b @32k | +0.184 | +0.050 | — | **−0.008** |

Measured is now **0.45–1.4×** the prediction, not 4–23×, and it is ~0 exactly
where the prediction is ~0. **The "content drift" result was the degeneration.**
C2's across-prompt slope transfers to within-generation growth about as well as
one could ask. `../5_phase_drift_across_decode/report.md` §0.3 and §2.3 must be
rewritten; this is the second time in this project a finding has turned out to
be an artifact of text quality, and the first time the fix was already in hand.

---

## 4. R6 — the boundary is pinned per model, not universally

25 cells, 6 models (was 16 and 6), symmetric target, `--boot 400`.

| line | monotone d\* | was | 90% meas | 90% arch | bracket |
|---|---|---|---|---|---|
| STOP (15%) | **54.1%** | 57.4 | [49.6, 58.8] | [48.0, 61.8] | **GAP [53.7, 54.3], 0.7 pts** |
| GO (35%) | **29.0%** | 26.8 | [25.7, 32.0] | [25.5, 34.6] | **OVERLAP [27.0, 28.8], 1 inversion** |

Spearman(dead-2, band) = **−0.978**, and **−0.915 partialled on model identity**.

**The STOP gap closed to 0.7 points and the disagreement got sharper, not
softer.** The two cells bracketing it are different models: llama31-8b @128k
reads 20.2% in band at dead-2 53.7, and qwen3-8b @16k reads 12.2% at dead-2
54.3. **An 8-point band difference across 0.6 points of dead-2** is not a
measurement gap that another cell can fill — it is architecture.

**The GO bracket became an OVERLAP.** `boundary.py`'s own definition: cells of
different models on both sides of the line at the same dead-2, with an inverted
pair. Per-model crossings are llama31-8b 26.7, llama33-70b 31.9, qwen15-moe 32.1
— still 5.4 points apart after nine new cells. The `arch` interval stayed ~1.5×
`meas` at both lines.

This selects the second branch of the decision table written before the run:
**PINNED PER MODEL, NOT UNIVERSAL.** dead-2 *orders* the band extremely well —
and a single log-linear law over all 25 cells has R² = 0.971 with a
leave-one-model-out error of **2.60 band points** — but the *critical value* is
architecture-specific. C1 should be stated as an ordering claim plus a
per-architecture threshold, and C4's calibration pass should fit the threshold
as well as read dead-2. It already has the data to do so.

**The band saturates at the bottom, the routed gain does not.** qwen3-8b's band
goes 16.5 → 12.2 → 10.9 → 11.5 → 12.2 across dead-2 50.5 → 62.4, i.e. it rises
twice. Both inversions have overlapping 90% intervals, so C1's monotonicity is
**not** broken — but the band has flattened into its own noise. Over the same
five cells the routed gain is monotone: 1.421 → 1.317 → 1.298 → 1.283 → 1.281.
That is R2's argument arriving from a third direction; the phase figure should
use routed gain.

---

## 5. Wave 3 is ready — and N16's justification has changed

| cell | status |
|---|---|
| **N16** qwen3-30b @65,536 | **Run it** — for a sharper reason than the original "only if Q1/Q2 are flat" |
| **N17** llama31-8b @8k greedy | **Run it** — the loop risk that argued for waiting is gone (§3.1) |
| **N18** llama31-8b @32k DRIFT | **Run it** — same |

**Why N16 is now clearly worth it.** qwen3-30b's τ at its cap sits +0.202 above
the null — the single ambiguous number left in R4. That null is fitted across a
**two-octave hole** (32k → 128k): drop the 128k point and the prediction at 262k
swings by **0.470**, more than twice the effect being tested. 64k (rope 0.25) is
the missing point, and it is the cheapest way to make R4 decisive rather than
"mostly ABSOLUTE-L with a residual".

```bash
bash h0_measurement/bugs/co-design/script.sh --run --wave=3        # --dry first
```

### When to run `--extra`

**Now, and before wave 3.** Both extras have become more valuable:

- **X1 — DONE** (`job21484596`). See §5.1: it fits, but do **not** set
  `Q30_ONE_GPU=1` yet.
- **X2** (llama31-8b @128k DRIFT, prompts 3–5, ~3.5 GPU-h) — N1 is the cell that
  answers C4, and its route statistic has a **noise floor of 8% against an
  excess of 7** (§3.2). The verdict currently rests on p90 regret, not on the
  flip rate. Doubling to 6 prompts is what makes the flip rate readable, and
  §3.1 shows the cell will not loop.

```bash
bash h0_measurement/bugs/co-design/script.sh --run --wave=2 --extra   # wave 2 is done; this submits X1 + X2 only
```

### 5.1 X1: one GPU is 4× cheaper at the SAME speed — switch

`job21484596` — qwen3-30b-a3b-2507 @8,192, 1 prompt, **one** H100. Its log
(`h0_21484596_0.out`) settles both halves of the question.

**It fits, and it measures the same thing:**

| | τ | dead-2 | n₉₅ | sym band | lag cost | needle |
|---|---|---|---|---|---|---|
| X1, **1 GPU**, 1 prompt | 2.1630 | 60.74 | 142 | **18.06** | 1.128 | 1/1 |
| ref, 4 GPU, 4 prompts | 2.1680 | 62.33 | 104 | **18.03** | 1.115 | 4/4 |

**And it is exactly as fast per unit:**

| | rows | in-loop | units | **s/unit** | **GPU-s/unit** |
|---|---|---|---|---|---|
| X1, **1 GPU** | 36,864 | 468 s | 3 | **156** | **156** |
| ref, 4 GPU (job21421770) | 147,456 | 1,874 s | 12 | **156** | **625** |

The per-unit rate is **identical to three significant figures**. `device_map=auto`
pipelines the layers across the four cards, it does not shard the work, so three
GPUs sat idle for the whole run. **One GPU is 4× cheaper in GPU-hours at the same
wall time.** `Q30_ONE_GPU` now defaults to **1**, which takes N16 from 8 GPU-h to
**2** and N14 (wave 4) from 8 to **2**.

Two things this also explains:

- **Wall-clock is nearly all weight staging.** X1 spent 468 s (7.8 min) computing
  inside a 142-minute wall window; N15 spent 30 min in-loop inside 367. Staging
  61 GB of MoE weights dominates both. Size walltime headers on the in-loop rate
  and budget the staging on top — do not infer either from the other.
- **N15 was overpaid.** It ran on 4 GPUs before this was known, so ~3.7 of its
  ~5 GPU-h bought nothing. Not recoverable, and not worth re-running: the cell
  is complete and correct.

**The one risk X1 did not test is the KV cache.** It ran at 8k, where the cache
is ~0.8 GB against ~19 GB of headroom after 61 GB of weights. N16 is at 64k,
where the cache is ~6.4 GB — it should fit with ~12 GB spare, but that is an
estimate, not a measurement. If N16 OOMs it fails fast and costs almost nothing;
re-run that one cell with `Q30_ONE_GPU=0`.

### 5.2 Measured rates, for future walltimes

From the wave-1 logs (18 units each, main tier, 1 GPU):

| cell | in-loop | s/unit |
|---|---|---|
| llama31-8b @16,384 | 1,972 s | 110 |
| qwen3-8b @16,384 | 2,158 s | 120 |
| qwen3-8b @24,576 | 2,405 s | 134 |
| llama31-8b @65,536 | 2,791 s | 155 |
| llama31-8b @98,304 | 3,439 s | 191 |
| qwen3-30b @8,192 (4 GPU, 12 u) | 1,874 s | 156 |
| qwen3-30b @4,096 (4 GPU, 12 u) | 1,780 s | 148 |

Every wave-1 header was generous — the sheet's `s/unit × units × 1.25 + 10 min`
rule held with room to spare. Note wall-clock is a poor proxy on the large tier:
N15 spent 367 min wall against 30 min in-loop, nearly all of it staging 61 GB of
weights. Size headers on the in-loop rate, and budget the staging separately.

---

## 6. What these cells do NOT say

- **Nothing about the two new designs.** No cell here carries `group_alloc` or
  `coarse_bits`; wave 4 is the first that will. The only GQA numbers so far are
  the CPU pilot's at n_rep = 2 (`plan.md` §7.0), and every headline cell is
  n_rep 4 or 8.
- **R5 is 4 of 6 cells**, one model family spans the length axis, and 3 prompts
  is thin for a statistic whose floor is 8–23%. X2 is the fix.
- **The band numbers here are 6-prompt (main) / 4-prompt (large)** and are not
  comparable with R4's 2–3-prompt cells to better than ~2 points (§2). R7's
  intervals are what turn that into a stated uncertainty, and R7's blocks are in
  wave 4.
- **R6's STOP line rests on qwen3-8b**, the only model that crosses it within its
  own curve, and that curve has saturated (§4). A third architecture in the
  50–60 dead-2 band would be worth more than another qwen3-8b cell.
