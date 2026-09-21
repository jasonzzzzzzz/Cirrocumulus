# Co-design waves 1–4: R4, R5, R6, R7 and the two new designs

**Status 2026-09-21.** ALL FOUR WAVES LANDED. Waves 1–2 are §1–§6; **waves 3 and 4
are §7** — they deliver the verdicts on both co-design designs, R7's error bars,
the last R4 cell and the rest of R5. Every wave-3/4 number is reproducible from
`wave3.csv` / `wave4.csv` beside this file (§7.1).

This report covers only what these ten cells measured. The two new *designs*
(GQA-group allocation, cascade score) are in `plan.md`; nothing here tests them,
because waves 1–3 were deliberately submitted without those columns.

| what | where |
|---|---|
| R4's answer, rewritten | `../4_rope_limit_or_mechanism/report.md` §8–§9 |
| R5 campaign 2, 4 of 6 cells | §3 here; `../5_phase_drift_across_decode/report.md` still describes campaign 1 |
| R6 with 25 cells | §4 here |
| **the two co-design designs — verdicts** | **§7.2–§7.4** |
| **R7 error bars** | **§7.5** |
| wave 3/4 data, per head | `wave3.csv`, `wave4.csv` (§7.1) |

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
3. **R5 campaign 1's loops are fixed, but the τ-drift question is open.** The
   anti-loop knobs held (distinct-4 0.71–0.98 at step 4,096, against 0.04–0.45
   before). ~~With the loops gone the "content, not length" finding disappears~~
   — **corrected in §7.7**: per-prompt Δτ is too noisy (sd ~0.22) for 3–6 prompts
   to tell content drift from length drift either way. (§3.3)
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

> **⚠ CORRECTED 2026-09-21 — this section overstates.** Its conclusion ("measured
> is 0.45–1.4× the prediction … the content-drift result was the degeneration")
> was drawn from four cells with a statistic whose noise exceeds the effect. Wave 3
> doubled the 128k cell to 6 prompts and its Δτ moved from −0.029 to **+0.092**;
> per-prompt Δτ there spans **−0.212 … +0.324**, so a 3-prompt block has an sd of
> ~0.19 and every prediction below sits inside one block-sd. **The L-growth test
> is underpowered: it separates neither content drift nor length drift.** The loop
> fix itself (§3.1) and the two-timescale result (§3.2) stand. See §7.7.

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

---

## 7. Waves 3 and 4 — the verdicts (2026-09-21)

### 7.0 The five things worth knowing

1. **GQA: per-head routing is realizable, at a discount that grows with group
   size.** The grouping's own cost on in-band heads is **1.000× / 1.136× /
   1.316×** at n_rep **1 / 4 / 8** — monotone, so head heterogeneity inside a
   KV group is the mechanism. ROADMAP R12's threat (">3×") does not materialise.
   (§7.2)
2. **The shared constraint hurts eviction MORE than the interior.** On the same
   corner and ranking rule, the group band is **+4.3 to +6.5 points above** the
   per-head band. The realizable (GQA) comparison *widens* SIEVE's edge. (§7.2)
3. **The cascade pays: build it at bc = 4.** A current-query score read off the
   4-bit base tier removes **82–91% (median 89%)** of the lag penalty on in-band
   heads; bc = 3 removes 59–79%. The deployable form (lagged `o`, no V read)
   matches its own bound everywhere. (§7.3)
4. **R7: prompts are a small error component; the corner set is the big one.**
   At the reference block size the prompt-block sd is **0.4–3.0 band points**
   (not the 9.6 the block-size-2 screen predicted), while a five-corner minimum
   moves the symmetric band by **−5 to −9 points** and changes verdicts. (§7.5)
5. **R5's two-timescale result holds on six cells, and one of my own claims does
   not.** The route survives 4,096 tokens (p90 regret **1.00** everywhere); the
   allocation goes stale (`froz/lag1` **2.5–4.3×**). But the within-generation
   τ test is underpowered, correcting §3.3. (§7.7)

### 7.1 What ran, and where the data is

| wave | cell | job | model @ ctx | config |
|---|---|---|---|---|
| extra | X1 | 955334 | qwen3-30b @8k | LEAN, 1 prompt, **1 GPU** (Trillium re-run) |
| extra | X2 | 955335 | llama31-8b @128k | DRIFT, prompts 3–5 (N1's second block) |
| 3 | N16 | 958168 | qwen3-30b @64k | LEAN, 3 prompts |
| 3 | N17 | 958169 | llama31-8b @8k | DRIFT, **greedy** |
| 3 | N18 | 958170 | llama31-8b @32k | DRIFT |
| 4 | N7a/b | 961033/34 | llama31-8b @128k | FIVE+, blocks 6–11 / 12–17 |
| 4 | N5a/b | 961035/36 | llama31-8b @32k | FIVE+, blocks 6–11 / 12–17 |
| 4 | N6a/b | 961037/38 | qwen3-8b @8k | FIVE+, blocks 6–11 / 12–17 |
| 4 | N11 | 961039 | qwen15-moe @16k | LEAN+, the **n_rep = 1 control** |
| 4 | N14a/b | 961040/41 | qwen3-30b @8k | FIVE+, blocks 4–7 / 8–11 |

All nine wave-4 sidecars carry `group_alloc: true`, `coarse_bits: [3, 4]` and
`floor_maxb`. X1 on Trillium reproduces the rorqual X1 across clusters
(τ 2.1634 vs 2.1630).

**The data, for later retrieval** — `build_wave_csv.py` writes both files:

| file | rows × cols | size | granularity |
|---|---|---|---|
| `wave3.csv` | 147,456 × 127 | 127 MB | LEAN cells: per head at the last quantized step · DRIFT cells: per head at **every** measured step |
| `wave4.csv` | 158,976 × 152 | 198 MB | per head at the last quantized step (step 4) |

That is the level every number below is computed from, so each is a `groupby`
away. Columns: provenance (`cell`, `job`, `serves`, `model`, `ctx`, `n_rep`,
`prompt`, `prompt_offset`, `family`, `step`, `layer`, `head`, `kv_head`,
`rot_seed`, `corpus_sha`, `corner_tag`, …), the budget-independent phase
statistics, and every column at the **headline budget B = 3** including all 44
co-design columns. The B = 1/2/4 copies and the per-width noise-fit diagnostics
are dropped; the parquets remain the full record. Floats are written to 7
significant digits. **The build round-trips:** row counts, band counts (i.e. no
gain rounded across the 2× line) and medians to 1e-5 all match the parquets, and
the n_rep = 1 identity holds inside the CSV (max diff 0.0).

### 7.2 G — GQA group allocation: realizable at a discount

Read on in-band heads (`gain_pp3_accum ≥ 2`), the only heads that use the
interior. `grp_pp_accum_over_head3` is the grouping's **own** cost — the shared
allocation against the per-head one it replaces — with R3's lag tail divided out.

| cell | n_rep | **grouping cost, in band** | pure constraint (oracle info), in band | band: per-head → **group** | + cascade bc 4 |
|---|---|---|---|---|---|
| qwen15-moe @16k | **1** | **1.000** | 1.000 | 37.1 → **37.1** | 46.4 |
| llama31-8b @32k | 4 | **1.115** | 1.176 | 39.9 → **46.4** | 51.5 |
| llama31-8b @128k | 4 | **1.136** | 1.183 | 23.8 → **28.9** | 36.5 |
| qwen3-8b @8k | 4 | **1.191** | 1.255 | 19.9 → **24.7** | 39.6 |
| qwen3-30b @8k | **8** | **1.316** | 1.400 | 17.3 → **21.6** | 35.1 |

- **The control is exact on real attention.** At n_rep = 1 every grouping
  marginal is 1.000 (p90 1.000) and the bands agree to the decimal.
- **Dose-response, 1 → 4 → 8: 1.000 → 1.136 → 1.316.** Monotone in group size.
  Heads that share a KV head are alike but not identical, and the more of them
  share, the more the shared allocation costs. That is the mechanism, and it turns
  a number into a claim.
- **The verdict (plan.md §8): REALIZABLE AT A DISCOUNT.** Worst in-band cell
  1.316×, inside 1.15–2.0×. Across all heads the cost is tiny (1.003–1.056×):
  like R3's lag cost, it concentrates on the heads that matter.
- **Grouping is cheaper under lagged information than under oracle information**
  (1.115–1.316 vs 1.176–1.400). A lagged score already blurs the differences
  between a group's heads, so forcing them to share costs less.
- **The group band sits 4.3–6.5 points ABOVE the per-head band**, on the same
  corner and ranking rule. Eviction is all-or-nothing, so one keep-set shared by
  n_rep heads is a harsher compromise than one set of bit-widths. **Under the
  realizable constraint, the interior's edge over eviction grows.**

**n_rep 2 was the mild case.** The CPU pilot (qwen3-1.7b) read 1.005×; the
headline ratios are 4 and 8, which is why wave 4 had to run.

### 7.3 C — the cascade score: build it at bc = 4

`closed` = the share of the per-head lag penalty the cascade removes, in-band.

| cell | lag penalty, in band | **closed, bc = 3** | **closed, bc = 4** |
|---|---|---|---|
| llama31-8b @32k | 1.465× | 0.59 | 0.82 |
| llama31-8b @128k | 1.514× | 0.71 | 0.86 |
| qwen15-moe @16k | 1.567× | 0.72 | 0.89 |
| qwen3-30b @8k | 1.545× | 0.72 | 0.89 |
| qwen3-8b @8k | 1.647× | 0.79 | 0.91 |
| **median** | | **0.72** | **0.89** |

- **Verdict: BUILD IT at bc = 4** (plan.md §8 threshold 0.5). Median 0.89, and no
  cell below 0.82.
- **The deployable variant is as good as its bound.** `csv` (lagged `o`, no V read
  at the current step) matches `cs` (exact `o`) to two decimals in every cell. The
  cascade needs only the base-tier **keys**.
- **The CPU pilot was pessimistic.** qwen3-1.7b at 2k read 0.31–0.35 in-band at
  bc = 3; the headline cells read 0.59–0.79. A 3-bit base tier is marginal on a
  small debug model and useful on the real ones.
- **It attacks exactly the gap it was designed for.** B3 found the E2-vs-symmetric
  gap largest on the qwen3 models (up to 24 points); the cascade adds the most
  there (qwen3-8b +14.9, qwen3-30b +13.5 band points in the group frame).

### 7.4 The storable design

Group allocation **and** cascade at bc = 4 — what a KV cache can actually store,
scored from what a deployed system actually has — against the group corner:

| cell | group, lagged | **group + cascade** | change |
|---|---|---|---|
| llama31-8b @32k | 46.4 | **51.5** | +5.1 |
| llama31-8b @128k | 28.9 | **36.5** | +7.6 |
| qwen15-moe @16k | 37.1 | **46.4** | +9.3 |
| qwen3-8b @8k | 24.7 | **39.6** | +14.9 |
| qwen3-30b @8k | 21.6 | **35.1** | +13.5 |

qwen3-30b @8k — a STOP cell in R3's symmetric table — reaches 35.1%, the GO line,
once the cascade is in. **These are group-corner bands**, a weaker competitor than
R3's per-head minimum over corners, so compare *changes* here, not levels with R3.

### 7.5 R7 — error bars: the corner set, not the prompts

Wave 4's five-corner blocks pooled with R3's reference block, at each cell's
reference size (6 main / 4 large), R3's lag-sweep runs excluded. Blocks cross
two corpora, so they are independent samples, not a rotation control.

| cell | sym (accum) [90%] | prompt-block sd | verdict | **sym (min over 5)** [90%] | verdict |
|---|---|---|---|---|---|
| llama31-8b @32k | 37.0 [31.4, 42.5] | 2.98 | straddles GO | **28.4** [23.6, 33.1] | NARROW, clear |
| llama31-8b @128k | 20.5 [15.9, 25.0] | **1.29** | NARROW, clear | **14.5** [10.5, 18.5] | straddles STOP |
| qwen3-8b @8k | 14.7 [11.5, 17.9] | 1.53 | straddles STOP | **7.9** [4.9, 11.0] | STOP, clear |
| qwen3-30b @8k | 13.7 [10.0, 17.3] | 0.41 | straddles STOP | **8.6** [5.7, 11.5] | STOP, clear |

- **Prompts: sd 0.4–3.0 band points.** The block-size-2 screen predicted ~9.6 for
  llama31-8b @128k; it measures **1.29**. R7's decision-table branch "*n_prompts
  ≥ 12 is the standard for 128k*" is **refuted** — six prompts suffice. The large
  screen value was the noise of 2-prompt medians, as B3 warned.
- **The corner set: −5.1 to −8.6 points on the symmetric band**, −9.2 to −10.9
  within single blocks of llama31-8b @32k. It is the dominant component, and it
  moves verdicts: with the accum corner alone three of four cells straddle a line;
  with the five-corner minimum three of four are **clear**.
- **So the paper must name its corner set.** "The band is X%" is not a
  measurement without it. The five-corner minimum is the stronger, more defensible
  competitor, and under it llama31-8b @128k is the one cell that becomes
  borderline (14.5, straddling STOP).

### 7.6 R4 — N16 settles the last residual

| qwen3-30b point | τ |
|---|---|
| 32k (rope 0.125) | 2.370 |
| **64k (rope 0.25) — N16** | **2.563** |
| 128k (rope 0.50) | 2.900 |

- **Out of sample:** the quadratic null fitted *without* 64k predicted τ(64k) =
  2.611; N16 measured **2.563** (−0.048). The two-octave hole never made the null
  fragile.
- **With N16 in the fit:** excess at 262k (rope 1.00) = **+0.209** (was +0.202), at
  197k (rope 0.75) = +0.028. Stable to 0.007.
- **R4's answer is now final:** mostly ABSOLUTE-L — τ is convex in log L — with a
  real, measured **+0.21 τ excess at exactly rope_frac = 1.00** in qwen3-30b (and
  qwen3-8b, +0.216), none in llama31-8b (−0.135) and none at 0.75.
  `../4_rope_limit_or_mechanism/report.md` §8.7 updated.

### 7.7 R5 — two timescales on six cells, and a correction

| cell | flip floor | flip @4,096 | excess | **p90 regret** | ρ(gain, calib) | **froz/lag1** | d4 |
|---|---|---|---|---|---|---|---|
| llama31-8b @8k, **greedy** (N17) | 12% | 18% | +6 | **1.00** | 0.80 | **3.18** | 0.95 |
| llama31-8b @8k | 23% | 25% | +3 | **1.00** | 0.63 | **3.03** | 0.93 |
| llama31-8b @32k (N18) | 15% | 14% | 0 | **1.00** | 0.78 | **4.30** | 0.99 |
| llama31-8b @128k, **6 prompts** (N1+X2) | **6%** | 9% | **+3** | **1.00** | **0.83** | **3.94** | 0.98 |
| qwen3-8b @8k | 16% | 21% | +5 | **1.00** | 0.69 | **2.52** | 0.79 |
| qwen3-8b @32k | 14% | 17% | +2 | **1.00** | 0.65 | **2.85** | 0.71 |

`froz/lag1` at each context's valid horizon (256 / 1,024 / 4,096 steps).

- **Route: one-pass, on every cell.** p90 regret 1.00 in all six, two
  architectures, three contexts, sampled and greedy. **Doubling the 128k cell to 6
  prompts made it look MORE stable** — floor 8% → 6%, excess +7 → +3, ρ 0.69 →
  0.83 — so the residual drift at 3 prompts was sampling noise.
- **Allocation: re-budget on a schedule.** `froz/lag1` = 2.5–4.3× everywhere.
- **Greedy no longer loops** (d4 0.95; campaign 1's greedy looped from step ~512),
  so the anti-loop knobs fix both decoders. Greedy halves the step-to-step noise
  floor (12% vs 23%) and changes no conclusion.
- **⚠ Correction — the L-growth test is underpowered.** §3.3 concluded that with
  loops gone the within-generation Δτ matches the cache-growth prediction. With
  wave 3:

  | llama31-8b | predicted Δτ | measured |
  |---|---|---|
  | 8k | +0.143 | +0.208 |
  | 32k (N18) | +0.042 | +0.126 |
  | 128k, 6 prompts | +0.011 | +0.092 |

  and the 128k cell's two 3-prompt blocks read **−0.029** and **+0.236**; its six
  prompts individually span **−0.212 … +0.324**. A per-prompt sd of ~0.22 means a
  3–6-prompt cell cannot resolve effects of 0.01–0.17. **Neither campaign 1's
  "content drift" nor this report's "length drift" is supported.** Resolving it
  needs ~20 prompts per cell (sd ~0.05). It does not touch C4: the two-timescale
  result rests on regret and `froz`, which are far less noisy.

### 7.8 Two reader defects found on the real data

| defect | symptom | fix |
|---|---|---|
| the "matched" per-head band ranked the corner by **raw attention**, while the group corner ranks by lagged **sensitivity** `w2p` | at **n_rep = 1**, where the two must be identical, they read **37.6 vs 37.1** | reference is now `err_e3_<score>_w2p_frac`; n_rep = 1 agrees exactly (37.1 = 37.1; corner errors identical to 0.0) |
| `errorbars.py` took its block size from the smallest run in a cell, which on the llama cells was R3's 3-prompt **lag sweep** | blocks of 3 instead of R7's reference 6, and a lag-sweep run pooled into the error bar | re-read with the lag sweeps excluded — a usage fix, recorded here so the glob is not repeated |

The first is the more important: it would have mis-stated every group-vs-per-head
band comparison by 0.5–0.7 points, and it surfaced only because the n_rep = 1
control made a wrong number *visibly* wrong.

### 7.9 What this changes

| claim | before waves 3–4 | now |
|---|---|---|
| **C4 per-head routing** | assumed storable; R12 unmeasured | **storable at 1.14× (n_rep 4) – 1.32× (n_rep 8)** in-band; the realizable comparison favours SIEVE more, not less |
| **the interior's score** | lagged `accum` | **cascade at bc = 4**, closing ~89% of the lag gap from the base tier's keys alone |
| **the design** | per-head, lagged | **per-KV-head allocation + cascade score + two-timescale (offline router, re-budgeted allocator)** |
| **R7** | "corner set matters" | measured: corner set −5 to −9 pts, prompts ≤ 3; name the corner set, six prompts is enough |
| **R4** | mostly ABSOLUTE-L, residual unresolved | **final:** convex τ plus a real +0.21 cap excess at rope 1.00 |
| **R5 τ drift** | "matches the length prediction" | **underpowered** — open, needs ~20 prompts/cell |
| **R8** | granularity and score undecided | **`--granularity kv_head --interior-score cascade --coarse-bits 4`** — both P2 flags are now decided |

The last row is the one to act on: wave 4 was the gate for R8's router arms, and
it has now selected both design parameters.
