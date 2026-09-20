# R3 — the symmetric cell: MEASURED (job214217\*)

**The interior no longer gets information the corner is denied.** Both sides now
allocate from lagged attention, with positions the score has never seen held at
the top tier instead of evicted. 16 cells — all 6 models at 8k and at 32k, plus the 4 registry-ctx cells whose
context differs from those (mistral-7b and qwen15-moe are at 32k natively) —
plus a 4-point staleness sweep on 3 cells. Every run carries
`"interior_unseen_policy": "floor_maxb"`; nothing here is provisional.

Figures: `R3-fig.png` (from `R3-figure.py`). Numbers: `R3-cells.csv` (one row per
model × ctx, with verdicts), `R3-sweep.csv` (the lag sweep), `R3-per-run.csv`
(every run of all three campaigns), `R3-fresh-token.csv`
(`R3-fresh-token-test.py`, the controlled test behind §5).

---

## 0. Key results

1. **The interior's edge survives a symmetric comparison — about half of it.**
   Demoting the corner to `accum` (H2O) adds 6.4–28.7 band points over the
   all-oracle cell; demoting the interior as well gives back 42% of that
   (median cell; 6–63% range). The symmetric cell still sits **above** the
   all-oracle cell in **16/16** cells, by +1.2 to +12.5 points.
2. **But "no configuration is STOP" does not survive.** Under the symmetric
   comparison **5 of 16 cells are STOP** and 5 more are NARROW; the E2 cell
   (corner demoted only) has none.
3. **The lag cost lands on the heads that matter** — the R3 mechanism,
   confirmed with the fix in: out-of-band heads pay **1.00–1.04×**, in-band
   heads **1.13–1.64×**, ρ(lag cost, gain) = **+0.50 to +0.97** in 16/16 cells.
   The median head (1.00–1.28×) is an average over heads that were never going
   to use the interior.
4. **Staleness is real and rises with k.** The sweep (`lag:k` = score from
   exactly k steps ago) gives corner error / oracle corner of 1.17–1.33× at
   k = 1 up to 1.56–2.29× at k = 8, and the symmetric band falls from 45.5% to
   10.4% (llama31-8b @8k). **cost(k) is not flat: re-budgeting during decode is
   load-bearing**, and this is now measured rather than asserted.
5. **The phase axis is the strongest it has ever been:** Spearman(dead-2,
   symmetric band) = **−0.985** over 16 cells, and **−1.00 within each of the
   three context lengths** separately.
6. **The price of deployable scoring falls with ctx:** accum corner / oracle
   corner = 1.29–1.47× at 8k, 1.12–1.42× at 32k, 1.03–1.14× at 128k.
7. **The router needs recalibrating on the honest gain:** 3.9–24.0% of all heads
   are sent to the interior by an oracle-calibrated threshold and do not pay for
   themselves once the interior is honest.
8. **The fix is validated at scale.** On the cells all three campaigns ran, the
   old ordinal bump and the new floor agree to **0.98–1.00×** on interior lag
   cost, while the zeroing campaign reads **4.9–12.9×**. The controlled test
   predicted exactly this (§5).

---

## 1. What ran

`script_temp.sh` (the guarded re-run sheet) on Trillium, every override passed as
an argument, all nine commands.

| § | cells | job | status |
|---|---|---|---|
| A | main 4 models @8k | 21421769 | ✅ |
| A | qwen3-30b, llama33-70b @8k | 21421770 | ✅ |
| A | main 4 models @32k | 21421771 | ✅ |
| A | qwen3-30b, **llama33-70b @32k** | 21421772 | ✅ (missing when the previous campaign was analysed; its pre-fix copy landed later in job21406449) |
| B | qwen3-8b @40k, llama31-8b @128k | 21421773 | ✅ |
| B | qwen3-30b, llama33-70b @128k | 21421774 | ✅ |
| C | lag sweep llama31-8b @8k | 21421775 | ✅ |
| C | lag sweep llama31-8b @32k | 21421776 | ✅ |
| C | lag sweep qwen3-30b @8k | 21421777 | ✅ (never ran before) |

Corner `oracle,accum`, policy `frac`, dense decode, quantized steps 0 and 4
(sweep: `oracle` + `lag:k=1,2,4,8`, 16 steps, quantized every 2). Every parquet's
row count matches its `.json`; all prompts, families, steps and heads present;
PG-19 haystack; **every niah needle retrieved** (6/6 main, 4/4 large, 3/3 sweep).

**Row selection.** Per-head medians over the rows where every configured corner
scored — step 4, or steps 8–14 in the sweep, where all four lags are ready — for
the oracle and the practical columns alike, so all three cells describe the same
rows.

**Three cells are compared throughout:**

| cell | corner ranks by | interior water-fills on |
|---|---|---|
| oracle | current-step `a·‖v−o‖` | current-step `w²` |
| E2 | lagged `accum` | current-step `w²` |
| **symmetric (R3)** | lagged `accum` | **lagged `w2p`** |

`gain_pp_sym` (corner ranked by the same `w2p` the interior uses) is in the CSVs;
it tracks the symmetric band within 0.3–2.4 points everywhere, so the corner's
*ranking* is still not the lever.

---

## 2. Results

### 2.1 The three cells, 16 configurations

% of heads in band (≥ 2× at 3 b/token). GO ≥ 35, STOP < 15.

| model | ctx | oracle | E2 | **symmetric** | verdicts (or → E2 → sym) | dead-2 | τ |
|---|---|---|---|---|---|---|---|
| llama33-70b | 8k | 83.6 | 92.8 | **87.4** | GO → GO → **GO** | 5.5 | 1.52 |
| mistral-7b | 8k | 53.7 | 76.7 | **66.2** | GO → GO → **GO** | 10.4 | 2.00 |
| llama31-8b | 8k | 43.4 | 67.5 | **54.5** | GO → GO → **GO** | 18.8 | 2.08 |
| qwen15-moe | 8k | 31.0 | 55.5 | **41.4** | NARROW → GO → **GO** | 25.8 | 2.42 |
| qwen3-8b | 8k | 11.7 | 40.5 | **16.5** | STOP → GO → **NARROW** | 50.5 | 2.60 |
| qwen3-30b | 8k | 10.4 | 34.8 | **14.1** | STOP → NARROW → **STOP** | 62.0 | 2.17 |
| llama33-70b | 32k | 60.9 | 75.4 | **70.0** | GO → GO → **GO** | 9.6 | 1.94 |
| mistral-7b | 32k | 45.9 | 66.0 | **55.5** | GO → GO → **GO** | 20.3 | 2.21 |
| llama31-8b | 32k | 25.5 | 45.1 | **34.0** | NARROW → GO → **NARROW** | 27.0 | 2.41 |
| qwen15-moe | 32k | 22.4 | 37.0 | **30.2** | NARROW → GO → **NARROW** | 36.8 | 3.10 |
| qwen3-8b | 32k | 6.5 | 26.7 | **11.5** | STOP → NARROW → **STOP** | 59.8 | 3.08 |
| qwen3-30b | 32k | 6.9 | 25.3 | **8.1** | STOP → NARROW → **STOP** | 67.7 | 2.39 |
| qwen3-8b | 40k | 5.9 | 26.6 | **12.2** | STOP → NARROW → **STOP** | 62.4 | 3.35 |
| llama33-70b | 128k | 18.8 | 25.2 | **21.3** | NARROW → NARROW → **NARROW** | 41.4 | 3.32 |
| llama31-8b | 128k | 13.5 | 29.2 | **20.2** | STOP → NARROW → **NARROW** | 53.7 | 2.95 |
| qwen3-30b | 128k | 4.7 | 19.3 | **7.5** | STOP → NARROW → **STOP** | 73.5 | 2.83 |

Median **routed gain** (the robust statistic R2 asked for) runs 1.05–3.21× in the
symmetric cell against 1.08–4.77× in the E2 cell.

**Caveat on the corner set.** These corners are `accum` alone, which is what the
campaign ran to pay for the interior's extra water-fill. A five-corner `min`
(job92\*) is a stronger competitor and lowers *both* practical cells by up to 6
points — enough to move qwen3-30b @128k from NARROW to STOP (§4).

### 2.2 Where the interior's cost lands

| | interior lag cost, `err_wf^lagged / err_wf` |
|---|---|
| median head | 1.00–1.28× |
| **out-of-band heads (gain < 2×)** | **1.00–1.04×** |
| **in-band heads (gain ≥ 2×)** | **1.13–1.64×** |
| ρ(lag cost, gain) | **+0.50 to +0.97**, 16/16 cells |

The mechanism `bugs/2` proposed in its provisional version is confirmed with the
defect removed: a head is in band precisely because its sensitivity is
concentrated somewhere a flat allocation cannot exploit, and that concentration is
what moves between steps. **The interior's advantage and its dependence on
current-query information are the same property.**

The lagged allocator evicts 39–54% of tokens on its own, against 40–54% for the
oracle allocator — tier 0 is inside the allocator either way (C3).

### 2.3 The staleness sweep — the architecture question

`lag:k` scores from the attention exactly k steps ago, so k prices "allocate once
at prefill" (large k) against "re-budget every step" (k = 1). Steps 8–14, where
every lag is ready, so each k describes the same rows.

| model | ctx | k | corner / oracle | interior lag, median | interior lag, in-band | symmetric band % |
|---|---|---|---|---|---|---|
| llama31-8b | 8k | 1 | 1.33 | 1.20 | 1.41 | 45.5 |
| | | 2 | 1.54 | 1.43 | 1.68 | 29.5 |
| | | 4 | 1.82 | 1.66 | 1.96 | 20.6 |
| | | 8 | 2.29 | 2.24 | 2.62 | 10.4 |
| llama31-8b | 32k | 1 | 1.25 | 1.06 | 1.37 | 27.5 |
| | | 8 | 1.86 | 1.40 | 2.18 | 9.0 |
| qwen3-30b | 8k | 1 | 1.17 | 1.05 | 1.56 | 5.6 |
| | | 8 | 1.56 | 1.26 | 2.39 | 1.1 |

**Reading.** Both sides decay, and the band roughly quarters over 8 steps. A
prefill-only allocation is therefore not a design option at these budgets: the
question is only *how often* to re-budget, and the slope says the answer is
single-digit steps, not thousands. (R5 asks the complementary question — whether
the per-head *route* survives a long generation.)

### 2.4 The phase axis

| cells | Spearman(dead-2, band) |
|---|---|
| all 16, symmetric cell | **−0.985** |
| all 16, E2 cell | −0.921 |
| all 16, oracle cell | −0.979 |
| within 8k / 32k / 128k (6 / 6 / 3 models) | **−1.00 / −1.00 / −1.00** |

C1 now holds for all three corner/interior definitions on a matched grid, and
perfectly within every fixed context length.

### 2.5 The price of deployable scoring

Median head, accum-corner error / oracle-corner error:

| model | 8k | 32k | 40k | 128k |
|---|---|---|---|---|
| llama31-8b | 1.47 | 1.31 | | 1.14 |
| mistral-7b | 1.47 | 1.41 | | |
| qwen15-moe | 1.39 | 1.18 | | |
| qwen3-8b | 1.40 | 1.12 | 1.12 | |
| llama33-70b | 1.31 | 1.25 | | 1.03 |
| qwen3-30b | 1.29 | 1.18 | | 1.06 |

Range **1.03–1.47×**, falling with ctx in 6/6 models: at long context, lagged
attention is nearly as good as current-step truth for deciding *what to keep*.
C5's "scoring is solved" is strongest exactly where the band is lowest.

### 2.6 Two numbers for the methods section

- **Router miscalibration: 3.9–24.0% of all heads.** An oracle-calibrated 2×
  threshold sends them to the interior; under the honest interior they do not
  pay. The reverse error (heads that should be routed and are not) is ≤ 0.3%.
- **"Less information" is not a per-head bound: 9.1–32.1% of head-rows violate
  it** (`gain_pp > gain_best_practical`), up from the 13.8% job92\* reported, and
  rising with ctx. `waterfill` minimises the first-order proxy `w²` while the
  reported error is exact recomputation, so a lagged allocation can land on a
  better exact-error allocation. Aggregate direction only — the unit test
  asserts nothing per head.

### 2.7 Reproducibility

- **Across campaigns, same cluster** (this campaign vs the zeroing campaign,
  which differ only in the interior fix): the oracle-cell band agrees to
  **≤ 1.4 points** on all 16 shared cells.
- **Across clusters** (job92\*, the previous cluster): ≤ 3.5 points, the maximum
  on mistral-7b @32k (42.4 vs 45.9/45.7).

---

## 3. What this means for the design

1. **The interior is worth building, but it is not free.** It beats the best
   deployable corner on 7.5–87% of heads depending on the cell, from information
   a deployed system actually has. The honest edge is roughly *half* of what the
   E2 campaign implied.
2. **Re-budget during decode.** §2.3 measures the decay directly: allocate once
   at prefill and the band quarters within 8 steps.
3. **Calibrate the router on the honest gain**, not on the oracle gain: 3.9–24%
   of heads are mis-routed otherwise, and the mis-routed ones are concentrated
   where the cost is highest.
4. **Report routed gain, not only the band fraction.** The band moves 6–29 points
   with the corner set and the interior's information; the error ratios behind it
   move far less.
5. **Where the method does not pay:** qwen3-30b at every ctx, and qwen3-8b at
   ≥ 32k, are STOP under the symmetric comparison. They are also the cells with
   the highest dead-tier fractions — which is what C1 predicts.

---

## 4. Corrections to earlier versions of this report

| earlier claim | status |
|---|---|
| "E2's headline was an artifact of asymmetry; the symmetric band lands within ±5 points of the all-oracle cell" | **Refuted.** The symmetric cell is above the oracle cell in 16/16 cells (+1.2 to +12.5). The job92\* impression came from its five-corner `min`, not from the interior: a stronger corner lowers the practical cells. |
| "Two STOP verdicts return" | **Superseded:** 5 of 16 cells are STOP under the symmetric cell, and llama31-8b @128k moves the other way (STOP in the oracle cell → NARROW). |
| "The lag cost lands on in-band heads (1.36–1.58× in band, ρ +0.81 to +0.94)" | **Confirmed** with the defect removed: 1.13–1.64× in band, 1.00–1.04× out, ρ +0.50 to +0.97 over 16 cells. |
| "Median lag cost 1.01–1.18× reads nearly free" | **Updated:** 1.00–1.28×, and still the wrong summary — see §2.2. |
| "Campaign B (the staleness sweep) is unrun / only k = 1 is usable" | **Superseded:** it ran on 3 cells and every k is now valid (§2.3). |
| "Re-budgeting during decode is load-bearing" (asserted from k = 1 alone) | **Measured** (§2.3). |
| "13.8% of head-rows violate the per-head bound" | **Updated:** 9.1–32.1%, rising with ctx. |
| Scoring price "1.02–1.41×" | **Updated:** 1.03–1.47×, falling with ctx (§2.5). |
| "Replicates agree to 0.00–0.46 band points" | **Updated:** ≤ 1.4 points across campaigns on the same cluster, ≤ 3.5 across clusters (§2.7). |
| The fresh-token defect banner ("every `w2p` number must be re-measured") | **Discharged.** Re-measured here; the defect and its evidence are kept in §5 for the record. |
| "llama33-70b @32k missing", "qwen3-30b lag sweep never ran" | **Superseded** — both ran (§1). |

---

## 5. The defect, for the record

Two campaigns preceded this one and disagree with it on the interior columns
only. Both are in `R3-per-run.csv` under `treatment`, with their interior columns
prefixed `invalid_`.

- **`zero` (job2140\*, job21406\*)** — `Evictor.score(rank_bump=False)` returned
  0 for positions the history had never seen, so `w2p = 0` and `waterfill`
  **evicted the token appended at that very step**, which the current query
  attends to. Interior lag cost read **4.9–12.9×**.
- **`bump` (job92\*)** — those positions scored `max + 1`, an ordinal convention
  that inflates their share of the normalised `w2p` mass but, as it turns out,
  costs almost nothing.
- **`floor` (this campaign)** — unseen positions are held at `maxb` and the rest
  water-filled on the remaining budget (`alloc.waterfill_floor`, budget-matched);
  the corner bumps the same set, which also repairs `lag:k` for k ≥ 2, where the
  k−1 tokens newer than the snapshot used to be evicted.

**The controlled test predicted this campaign's numbers.** On real attention
(`R3-fresh-token-test.py`, Llama-3.2-1B, 512 heads), zero/floor = 6.4–7.0× and
bump/floor = 1.00 per head. At scale, on the cells all three campaigns ran:

| cell | bump (job92\*) | zero (job2140\*) | **floor (this campaign)** |
|---|---|---|---|
| mistral-7b @32k | 1.18 | 6.18 | **1.21** |
| qwen15-moe @32k | 1.04 | 5.67 | **1.04** |
| llama31-8b @128k | 1.02 | 12.85 | **1.02** |
| qwen3-30b @128k | 1.01 | 5.72 | **1.01** |

bump/floor = 0.98–1.00× — the old campaign's interior numbers were never
materially biased, and the "fix" that zeroed the token was the larger error.
Fixed by `Evictor.unseen()` + `alloc.waterfill_floor`, pinned by
`test_units.py::test_unseen_floor`.

---

## 6. What to run next

1. **A five-corner cell at one or two configurations** to price §2.1's caveat:
   how much of the symmetric band survives a `min` over `accum, window,
   recency, last_step`? Only `accum` was affordable alongside the interior in
   this campaign, and the corner set moves the STOP line.
2. **Extend the staleness sweep** to a second large model and to 128k. The
   decay slope is the number the cascade design needs, and it is currently
   measured on 3 cells.
3. **R7 error bars** should vary the corner set as well as the seed (§2.1).
4. **Documents:** ROADMAP R3 and the proposal can now cite the symmetric cell as
   measured. Anything asserting "no configuration is STOP" must go.
