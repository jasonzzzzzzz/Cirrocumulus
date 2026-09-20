# R6 — pin the sharp boundary: findings, design, and what to run

> ## Status: the boundary is **located** but not yet **shown to be universal**
>
> Nothing new has been measured for R6. Everything in §1–§3 is a re-analysis of
> R3's 16 measured symmetric cells (`../2_towards_real_evictor/R3-report.md`
> §2.1, jobs `job214217*`) with `boundary.py`, which is new and landed here. The
> GPU cells in §4 are designed and **not yet submitted**.
>
> The re-analysis is enough to say where the boundary is. It is not enough to say
> the location is the same for every architecture — and that, not the location,
> is what C1 actually claims.

Reader: `boundary.py` (beside this file). Submission sheet: `script.sh`.
Source numbers: R3's `R3-cells.csv`, and `h0_measurement/reports/r6_boundary.csv`
once §4 has run.

---

## 0. Key results

1. **The boundary has a location, on the honest comparison.** On the symmetric
   cell (interior *and* corner on lagged `accum`), the band crosses
   **GO (35%) at 26.8% dead-2** and **STOP (15%) at 57.4% dead-2** — monotone
   fits, 16 cells, 6 models. In the robust statistic (R2), those are routed
   gains of **1.93×** and **1.35×**.
2. **A smooth fit is the wrong tool here, and it matters.** The logistic puts GO
   at **34.6%** — but no cell exists between 25.8% and 27.0%, where the band
   actually crosses 35%. The curve flattens above 30% dead-2 and the logistic
   pays for that flat tail by moving the crossing. `boundary.py` therefore leads
   with an isotonic (monotone) crossing, which cannot leave the bracket, and
   **warns** when a parametric form does. This is the single most important
   implementation decision in R6: the earlier plan of "fit a curve and quote
   d\*" would have published 34.6%.
3. **Architecture, not measurement noise, is the remaining uncertainty.** At GO
   the 90% measurement interval is 8.4 pts wide and the architecture interval
   10.3; at STOP, 8.5 against 11.2. Every per-model crossing is an
   *interpolation* across a wide gap in that model's own curve (§3), and where
   two models can be compared at matched dead-2 they **disagree**: at ~53.7%
   llama31-8b still reads 20.2% in band, while qwen3-8b has already crossed 15%
   by 53.2%.
4. **The boundary moves with the comparison**, by more than its own uncertainty
   — so "the boundary" is only meaningful once the cell is named (§2.2). The
   oracle cell crosses STOP at 46.9% and the symmetric cell at 57.4%: a
   **10.5-point** shift from information asymmetry alone. Under the E2 cell the
   STOP line **is never crossed**.
5. **The symmetric cell is the better-behaved phase variable**, which is a new
   argument for reporting it. Its brackets are *gaps* (missing data, closable by
   running cells); the E2 cell's GO bracket is an *overlap* — models on both
   sides of the line over a 9.1-pt range, with an inverted pair. Spearman
   (dead-2, band): **−0.985** symmetric, −0.979 oracle, −0.921 E2.
6. **dead-2 is corner-independent, and measurably so.** Across the E2 campaign
   and the R3 re-run — different corner sets, different interior code, different
   clusters — dead-2 agrees on all 15 shared cells to **≤ 0.30 pts** (median
   0.05). This is what makes §4's predicted cell positions trustworthy: they are
   read off an existing campaign, not modelled.
7. **The roadmap's R6 numbers were wrong in five ways**, listed in §5 of
   `script.sh`. The two that changed the design: the STOP guess of "45–50%" was
   **low** (that is roughly the *oracle* cell's crossing), and "12k, 24k, 48k"
   picks cells that mostly miss both gaps.

---

## 1. Method

`boundary.py` reads any set of result parquets and reports, per target:

| step | what it does |
|---|---|
| rows | quantized rows where **every** configured corner scored (step 4 in the dense decode) — the same population R3-report.md §1 uses, so oracle, E2 and symmetric describe the same heads. dead-2 is read on those rows too. |
| provenance | the symmetric column is read **only** from runs stamped `"interior_unseen_policy": "floor_maxb"`. `job92*` (ordinal bump) and `job2140*` (fresh token evicted) are blanked unless `--allow-unfixed`, which stamps every line PROVISIONAL. |
| cells | per-head medians → one cell per (model, ctx, corner set); replicate runs averaged, their spread reported. Cells failing the input-validity gate are named and dropped. |
| **monotone d\*** | isotonic (PAVA) fit of band vs dead-2, then linear interpolation to the line. Assumes only what C1 claims — that the band is non-increasing in dead-2 — and lies inside the bracket by construction. |
| logistic d\* | cross-check with a smooth form; **warned** when it falls outside a gap bracket. |
| 90% meas | layer-cluster bootstrap inside each cell (heads in a layer share a residual stream, so heads are not independent), refit per replicate. |
| 90% arch | the same, plus resampling **models** with replacement. Coarse with 6 models, but it is the interval that matches the C1 claim. |
| bracket | model-free: **GAP** = last cell above the line sits below the first cell below it, nothing measured inside. **OVERLAP** = cells of different models on both sides at the same dead-2 — architecture scatter, which running more cells does *not* fix. |
| per model | each model's own crossing, interpolated along its own curve, or a one-sided bound. |
| guards | one (model, ctx) contributes **one** cell: pooled campaigns with different corner sets would otherwise enter the same head population twice and double-weight it. With fewer than 3 cells it prints the table and skips the fit — that is the pilot's read (§7 step 2). |

Neither interval includes seed variance or corner-set variance (R7).

Reproduce §2 exactly:

```bash
.venv/bin/python h0_measurement/bugs/6_pin_sharp_boundary/boundary.py \
    "h0_measurement/results/job214217*/*.parquet" --boot 1000
```

Regression checks run while building it: on the E2 campaign the E2 bands
reproduce `../2_towards_real_evictor/report.md` exactly (llama33-70b 92.6 / 86.0
/ 75.4 / 61.9 / 25.0; qwen3-30b @128k 18.6), and on `job92*` with
`--allow-unfixed` the symmetric bands reproduce the old provisional R3 numbers
(45.2 / 20.8 / 12.5 / 4.7).

---

## 2. The measured boundary

### 2.1 The 16 cells

Symmetric cell; dead-2 and band in %, routed = geometric mean of max(gain, 1).
Oracle and E2 bands from the same rows, for §2.2.

| model | ctx | dead-2 | oracle | E2 | **symmetric** | routed (sym) |
|---|---|---|---|---|---|---|
| llama33-70b | 8k | 5.5 | 83.6 | 92.8 | **87.4** | 3.17× |
| llama33-70b | 32k | 9.6 | 60.9 | 75.4 | **70.0** | 2.56× |
| mistral-7b | 8k | 10.4 | 53.7 | 76.7 | **66.2** | 2.31× |
| llama31-8b | 8k | 18.8 | 43.4 | 67.5 | **54.5** | 2.04× |
| mistral-7b | 32k | 20.3 | 45.9 | 66.0 | **55.5** | 2.05× |
| qwen15-moe | 8k | 25.8 | 31.0 | 55.5 | **41.4** | 1.79× |
| llama31-8b | 32k | 27.0 | 25.5 | 45.1 | **34.0** | 1.68× |
| qwen15-moe | 32k | 36.8 | 22.4 | 37.0 | **30.2** | 1.57× |
| llama33-70b | 128k | 41.4 | 18.8 | 25.2 | **21.3** | 1.40× |
| qwen3-8b | 8k | 50.5 | 11.7 | 40.5 | **16.5** | 1.42× |
| llama31-8b | 128k | 53.7 | 13.5 | 29.2 | **20.2** | 1.41× |
| qwen3-8b | 32k | 59.8 | 6.5 | 26.7 | **11.5** | 1.28× |
| qwen3-30b | 8k | 62.0 | 10.4 | 34.8 | **14.1** | 1.38× |
| qwen3-8b | 40k | 62.4 | 5.9 | 26.6 | **12.2** | 1.28× |
| qwen3-30b | 32k | 67.7 | 6.9 | 25.3 | **8.1** | 1.28× |
| qwen3-30b | 128k | 73.5 | 4.7 | 19.3 | **7.5** | 1.21× |

### 2.2 The crossing, per target

| target | line | monotone d\* | 90% meas | 90% arch | logistic | bracket |
|---|---|---|---|---|---|---|
| **symmetric** | GO | **26.8** | [25.1, 33.5] | [24.9, 35.1] | 34.6 ⚠ outside | GAP [25.8, 27.0], 1.2 |
| **symmetric** | STOP | **57.4** | [52.3, 60.8] | [50.7, 61.9] | 54.5 | GAP [53.7, 59.8], 6.1 |
| oracle | GO | 24.2 | [21.2, 26.0] | [20.9, 28.6] | 27.4 ⚠ outside | GAP [20.3, 25.8], 5.5 |
| oracle | STOP | 46.9 | [42.0, 52.9] | [42.0, 54.2] | 46.5 | GAP [41.4, 50.5], 9.1 |
| E2 | GO | 39.0 | [33.3, 51.4] | [33.4, 56.2] | 50.0 | **OVERLAP** [41.4, 50.5], 1 inversion |
| E2 | STOP | — | — | — | 77.1 (extrapolated) | never crossed (min cell 19.3%) |

⚠ = `boundary.py` flags the parametric fit as misspecified: it lands outside a
bracket that contains no data. Every interval above is from the one `--boot
1000` run quoted in §1; the bootstrap is seeded (`--seed 0`), so it reproduces.

**Read this table twice.** Down a column it says the boundary is located to a
few points. Across the rows it says the location depends on which comparison is
being made, by 10.5 points at STOP (46.9 oracle → 57.4 symmetric) — more than
either interval. Any statement of the form "the boundary is at X% dead tiers"
must name the cell, or it is not a measurement.

---

## 3. Why the design is what it is

The pooled brackets are already narrow (1.2 pts at GO). What is wide is the
disagreement between models, and **every per-model crossing is interpolated
across a gap in that model's own curve**:

| model | line | crossing | interpolated across | cells |
|---|---|---|---|---|
| llama31-8b | GO | 26.6 | 18.8 → 27.0 (8k → 32k) | 3 |
| qwen15-moe | GO | 32.1 | 25.8 → 36.8 (8k → 32k) | 2 |
| llama33-70b | GO | 32.5 | **9.6 → 41.4** (32k → 128k) | 3 |
| qwen3-8b | STOP | 53.2 | 50.5 → 59.8 (8k → 32k) | 3 |
| qwen3-30b | STOP | < 62.0 | one-sided: its *lowest* cell is already STOP | 3 |
| llama31-8b | STOP | > 53.7 | one-sided, and **unimprovable** — 131,072 is its RoPE window | 3 |
| mistral-7b | — | — | crosses nothing (10–20% dead-2 over its whole range) | 2 |

So R6 needs cells that make a model **straddle** a line with two adjacent
measured points, in the two gaps the data leave open: **[53.7, 59.8]** at STOP
and **[25.8, 27.0]** at GO. Three consequences for the design:

- **qwen3-8b carries the STOP line.** It is the only model that crosses it
  within its own curve. 16k and 24k land at ~54 and ~57 — inside both its own
  crossing and the pooled gap.
- **qwen3-30b turns a bound into a crossing.** Its lowest cell (8k) is already
  STOP, so it only bounds the boundary from above. 4k lands at ~59 and makes
  STOP a three-architecture statement.
- **llama31-8b's STOP bound cannot be improved.** Its 128k cell sits at 53.7%
  and 131,072 is its RoPE window. Whatever the other models say, this model's
  "still NARROW at 53.7%" stands — and it is exactly the disagreement in §0.3.
  Report it; do not try to run it away.

Predicted dead-2 for each new cell is read off the E2 campaign, which measured
most of these (model, ctx) pairs already; §0.6 shows that transfer is good to
0.30 pts.

---

## 4. The runs

All cells use **R3's exact configuration**, so they pool with R3's 16 into one
fit: corner `oracle,accum`, policy `frac`, interior score `accum`, dense decode
(quantized steps 0 and 4), registry `n_prompts` (6 main / 4 large), PG-19
haystack, and the fresh-token floor (`floor_maxb`).

| # | cell | gap it fills | predicted dead-2 | tier | GPUs | units | walltime |
|---|---|---|---|---|---|---|---|
| A1 | qwen3-8b @ 16,384 | STOP [53.7, 59.8] | 54.1 (measured, E2 campaign) | main | 1 | 18 | 01:15 |
| A2 | qwen3-8b @ 24,576 | STOP [53.7, 59.8] | ~57 (interp. 54.1→59.8) | main | 1 | 18 | 01:30 |
| A3 | qwen3-30b-a3b-2507 @ 4,096 | STOP, from above | ~59 (extrap. below 62.0) | large | 4 | 12 | 01:15 |
| B1 | llama31-8b @ 16,384 | GO [25.8, 27.0] | 23.6 (measured, E2) | main | 1 | 18 | 01:00 |
| B2 | qwen15-moe-a2.7b @ 16,384 | GO [25.8, 27.0] | 34.1 (measured, E2) | main | 1 | 18 | 00:45 |
| B3 | llama31-8b @ 65,536 | curve shape 27.0→53.7 | 29.8 (measured, E2) | main | 1 | 18 | 01:30 |
| C1 | llama33-70b @ 98,304 | GO, the 32-pt jump | between 9.6 and 41.4 | large | 4 | 12 | 04:15 |
| C2 | llama33-70b @ 114,688 | GO, the 32-pt jump | between 9.6 and 41.4 | large | 4 | 12 | 05:00 |

A1–B3 ≈ **10 GPU-h**. C1–C2 add ≈ **70 GPU-h** and are opt-in (`--with-70b`);
they are worth it only if the GO line's architecture spread goes in the paper.
C1 is also `bugs/4` section B — whichever runs first, the guard skips the other,
provided R4 is submitted with the fresh-token fix in place.

Walltimes follow `script_temp.sh`'s rule (s/unit × units × 1.25 + 10 min main /
15 min large, rounded up to 15 min), anchored on llama31-8b's **measured** 224
s/unit at 128k and the decomposed rates for the rest, scaled by L^0.3. The R3
re-run logs now measure these on this cluster; if a shared cell ran slower than
its anchor, scale the headers before submitting.

**Before the batch.** Two cheap steps come first (§7): `--plan` re-derives this
table from the results on disk — if R3 is re-run and the gaps move, it says
which planned cells stopped being useful — and `--pilot` runs one prompt on the
two cells the design is least sure of. A pilot writes a real parquet for a cell
at `n_prompts=1`, so the "already measured" guard also checks the prompt count;
otherwise a pilot would make `--run` skip the very cell it was sizing.

**The commands.** `script.sh --run` issues exactly these, after the unit tests
and the fix gate, skipping any cell that already has a complete fixed result:

```bash
R3=(SIEVE_EVICTORS=oracle,accum SIEVE_CORNER_POLICIES=frac SIEVE_INTERIOR_SCORES=accum)

# A -- the STOP gap
sbatch --array=0-0 --time=01:15:00 h0_measurement/submit_h0.slurm "${R3[@]}" SIEVE_CTX=16384 qwen3-8b
sbatch --array=0-0 --time=01:30:00 h0_measurement/submit_h0.slurm "${R3[@]}" SIEVE_CTX=24576 qwen3-8b
sbatch --array=0-0 --gpus-per-node=4 --time=01:15:00 h0_measurement/submit_h0_large_models.slurm "${R3[@]}" SIEVE_CTX=4096 qwen3-30b-a3b-2507

# B -- the GO gap, and the curve-shape cell
sbatch --array=0-0 --time=01:00:00 h0_measurement/submit_h0.slurm "${R3[@]}" SIEVE_CTX=16384 llama31-8b
sbatch --array=0-0 --time=00:45:00 h0_measurement/submit_h0.slurm "${R3[@]}" SIEVE_CTX=16384 qwen15-moe-a2.7b
sbatch --array=0-0 --time=01:30:00 h0_measurement/submit_h0.slurm "${R3[@]}" SIEVE_CTX=65536 llama31-8b

# C -- only with --with-70b
sbatch --array=0-0 --gpus-per-node=4 --time=04:15:00 h0_measurement/submit_h0_large_models.slurm "${R3[@]}" SIEVE_CTX=98304 llama33-70b
sbatch --array=0-0 --gpus-per-node=4 --time=05:00:00 h0_measurement/submit_h0_large_models.slurm "${R3[@]}" SIEVE_CTX=114688 llama33-70b
```

Overrides travel as **arguments**: Trillium's `sbatch` is a shell function that
adds `--export=NONE`, so an env prefix never reaches the job. No `--mem`.

**Not submitted, and why:** llama31-8b past 128k (its RoPE window); mistral-7b
(crosses nothing); qwen3-8b @12k (lands ~52, below the gap, between two cells
that already agree); 48k on anything (past three models' windows, outside both
gaps for the llamas); qwen3-30b past 128k (R4 takes it to 192k/256k, far above
both lines).

---

## 5. Decision table, written before the run

| outcome | reading |
|---|---|
| gaps close, per-model crossings agree within the meas interval, `arch` narrows toward `meas` | **PINNED AND UNIVERSAL.** State C1 with a location: the band crosses GO at d\*\_GO [lo, hi] and STOP at d\*\_STOP [lo, hi] % dead tiers, across N architectures. C4's router threshold inherits a default needing no per-model calibration. |
| gaps close but per-model crossings stay apart (`arch` ≈ 2× `meas`; e.g. qwen3-8b ~53 vs llama31-8b still NARROW at 53.7) | **PINNED PER MODEL, NOT UNIVERSAL.** dead-2 still *orders* the band (ρ = −0.985, and −1.00 within every fixed ctx) but the critical value is architecture-specific. Restate C1 as an ordering claim plus a per-architecture threshold; C4's calibration pass fits the threshold as well as reading dead-2. |
| a new cell lands inside a gap on the **wrong side** of its line | **dead-2 is not monotone within a model** — the one outcome that breaks C1 as an order parameter rather than moving its threshold. Check τ and `rope_frac` on that cell first (R4: the two move together at the long end). |
| qwen3-30b @4k comes back **above** 15% | its STOP crossing is between 59 and 62; the gap closes from the high side and STOP becomes a three-architecture statement. |

**What would invalidate the run:** a cell failing the input-validity gate (needle
not retrieved). `boundary.py` drops and names such cells; a dropped cell inside a
gap means the gap is still open. Watch qwen3-30b @4k — 4,096 is the shortest
context any cell in this project has used, and it sits below the study's 8k
floor. It is a phase point, not a long-context point; say so when reporting it.

---

## 6. Limits of what R6 can conclude

- **The corner is `accum` alone.** A five-corner `min` is a stronger competitor
  and lowers both practical cells by up to 6 points, which moves the STOP line
  (R3-report.md §2.1, §4). Every d\* here is "against the accum corner". Pricing
  the five-corner version is R3-report.md §6 item 1, not this sheet.
- **No seed or corner-set variance** is in either interval — that is R7, which
  R3-report.md §6 item 3 says must vary the corner set too.
- **Six architectures** is a small sample for an "is it universal" question; the
  `arch` interval says so honestly rather than hiding it.
- **`report.py`'s phase page is untouched.** Its hatch marks the widest gap in
  dead-2 *coverage* (and only past 12 pts, so it never draws on these data), not
  the bracket of the crossing. Changing it while R3's chained report job could
  still fire was not worth the risk; `boundary.py` prints the bracket instead.
  If the phase figure goes in the paper, this is a small follow-up.

---

## 7. Next steps

All commands run from the project root, on the login node that has the fixed
`sievelib` (on Trillium, `trig-login01` — a CPU login node rejects GPU jobs).

1. **Re-derive the design — 0 GPU, ~2 minutes. Do this first.**

   ```bash
   bash h0_measurement/bugs/6_pin_sharp_boundary/script.sh --plan
   ```

   It recomputes both brackets from the symmetric cells on disk and, for every
   planned cell, prints where it is predicted to land, whether that is inside
   the pooled bracket, and whether it **splits that model's own crossing** —
   the two ways a cell can be worth running. As of now it reproduces §2's
   brackets exactly ([25.8, 27.0] and [53.7, 59.8]) and every planned cell
   scores "yes" in at least one column. If R3 is ever re-run, run this again
   before spending anything: it is the step that catches a design that has gone
   stale.

2. **Pilot the two least certain cells — ~35 min of GPU.**

   ```bash
   bash h0_measurement/bugs/6_pin_sharp_boundary/script.sh --pilot
   ```

   One prompt each on qwen15-moe @16k (the cheapest cell — the pipeline in
   miniature) and qwen3-30b @4k (the riskiest — below this project's 8k floor,
   so the input-validity gate is a real risk, and the only cell whose dead-2 is
   extrapolated). The script prints the four checks to make: overrides applied
   and `floor_maxb` stamped; **the needle retrieved at 4k**; dead-2 within ±3
   of the prediction; and s/unit against the walltime anchors. Pilot results do
   not satisfy the run guard, so nothing is lost.

3. **Submit the batch.**

   ```bash
   bash h0_measurement/bugs/6_pin_sharp_boundary/script.sh --run
   # add --with-70b for cells C1-C2 (~70 GPU-h)
   ```

   Both `--pilot` and `--run` first run `test_practical_interior`,
   `test_corner_provenance`, `test_unseen_floor`, `test_rescore_is_idempotent`
   and `test_rope_window`, then gate **G**: it finds a finished `floor_maxb`
   result and refuses to submit unless its median interior lag cost is < 2× (the
   fresh-token defect reads 4.9–12.9×). Verified against R3's re-run: **1.23×**,
   16 fixed results.

4. **Verify within a minute of each job starting** — line 1 of every log must
   echo the overrides:

   ```bash
   head -1 h0_measurement/logs/h0_<JOBID>_0.out     # h0large_ for the 4-GPU cells
   # ctx=16384|24576|4096|65536|98304|114688   evictors=oracle,accum
   # "ctx=per-model" or "evictors=per-config" -> scancel; the overrides were lost
   grep -L '"interior_unseen_policy": "floor_maxb"' h0_measurement/results/job<NEW>*/*.json
   # must print NOTHING
   ```

5. **Sanity-check each new cell before reading the boundary:** its dead-2 should
   land within ±3 pts of §4's prediction, and median
   `interior_lag_cost3_accum` should be ~1.0–1.3×. A cell that misses the gap it
   was chosen for does not close that gap — say so rather than re-fitting.

6. **Read the boundary** (CPU, login node):

   ```bash
   .venv/bin/python h0_measurement/bugs/6_pin_sharp_boundary/boundary.py \
       "h0_measurement/results/job214217*/*.parquet" \
       "h0_measurement/results/<R6_JOBS>/*.parquet" \
       "h0_measurement/results/<R4_JOBS>/*.parquet" \
       --csv h0_measurement/reports/r6_boundary.csv
   ```

   In this order: the per-model crossing lines (do they agree now?), the
   `bracket` line (did the gap close?), the `arch` interval (did it narrow
   toward `meas`?), and only then the monotone d\*. `bash script.sh --read`
   prints this command.

7. **Then update §0 and §2 of this file with the measured result**, take the
   §5 branch the data select, and propagate:
   - **ROADMAP R6** — replace "where it stands" with the closed numbers.
   - **C1's row in ROADMAP §1** — it currently cites "ρ = −0.95 over 24 configs"
     from the E2 campaign; the symmetric cell gives ρ = −0.985 over 16, plus a
     boundary location.
   - **R7** — if the branch is "pinned per model", R7's error bars must cover the
     corner set as well as the seed, since the corner set moves the STOP line by
     up to 6 points.
   - **R8** — the router-on/off design picks the highest and lowest dead-tier
     models; a pinned boundary says which side of the line each one is on, and
     the routed gain at d\* (1.35× at STOP, 1.93× at GO) is the effect size it
     should expect to recover.
