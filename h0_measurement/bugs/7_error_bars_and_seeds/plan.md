# R7 — error bars, and the cells whose verdict is not reportable

**Plan, 2026-09-20.** R5/R6 (and R4's ctx points, job214447\*) are in flight on
another machine, so every change to a **shared** file is additive and
default-off — §6 lists each one and what it does to a queued job: nothing.
Everything else is new files in this folder, or 0 GPU on parquets already on
disk.

**The R3 re-run has landed** (job21421769–74, all stamped
`"interior_unseen_policy": "floor_maxb"`, interior lag cost 1.23×). That is what
makes this plan different from the ROADMAP entry it replaces: the **symmetric
cell exists for the first time**, and it says the cell R7 was written about is
not in danger, while three others are.

ROADMAP.md §3 R7 (before today) read:

> qwen3-30b @128k sits at 18.6%, **3.6 points above the STOP line** […]
> `n_prompts` and `rot_seed` are not reachable through `SIEVE_*` […] Two extra
> seeds on the 128k-capable cells is enough.

Every clause is now wrong. §1 is why; §3 is the measurement; §4–§5 are what to
run instead.

---

## 1. Bugs and stale claims

### B1 · `SIEVE_ROT_SEED` does not produce a second sample

`rot_seed` reaches exactly one place: the quantizer's random rotation
(`run_h0.py` → `quant.random_rotation`). The **prompts do not depend on it**.
Each prompt is keyed by its index alone: `prompts.build(..., prompt_idx=p)`,
`hay_key = prompt_idx`, book `big[key % len(big)]`, offset
`Random(_seed("offset", name, key))`, needle `Random(_seed("needle", hay_key,
family))`. A re-submission with `SIEVE_ROT_SEED=1` reads **the same books at the
same offsets with the same needles**. That is a real but small variance
component (§3.4), not an independent sample.

### B2 · `SIEVE_N_PROMPTS` is nested, not disjoint

The loop was `for p in range(n_prompts)`: prompts always start at 0, so
`n_prompts=12` re-runs the reference's prompts and appends new ones, and a
disjoint replicate could not be submitted as its own job. Fixed by
`prompt_offset` (§6, S1 — applied, default 0).

### B3 · The band is not invariant to `n_prompts`, so blocks must match in size

A head's gain is a **median over its rows**, and the band counts heads over 2×.
Fewer prompts, noisier medians, more heads pushed over the line. Mean band over
all k-prompt subsets of the same run (step 4, B=3):

| cell | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|---|
| qwen3-30b @128k, oracle | 6.2 | 5.2 | 4.9 | **4.7** | | |
| qwen3-30b @128k, E2 | 21.0 | 20.7 | 19.1 | **18.9** | | |
| llama33-70b @128k, E2 | 26.0 | 25.9 | 25.3 | **25.1** | | |
| llama31-8b @128k, oracle | 14.5 | 13.8 | 13.6 | 13.6 | 13.5 | **13.7** |
| llama31-8b @128k, E2 | 30.9 | 31.0 | 29.8 | 29.7 | 29.3 | **29.0** |

Consequences: (1) a replicate is a **block of the reference's size**, never a
pooled 12-prompt run against a 4-prompt one; (2) the pooled estimate is a
better but *different* statistic, reported separately; (3) **bands from
different `n_prompts` are not comparable** — R4's ctx points run at
`n_prompts` 2–3 (`bugs/4/script.sh`) and R6 pools them with 4- and 6-prompt
cells, a bias of ~+0.5 to +2 points in the direction that inflates the band
(§6, S8).

### B4 · The cell R7 was written about is not the cell at risk

Measured with `errorbars.py` on the landed R3 re-run, symmetric target
(`gain_pp3_accum` — interior *and* corner lagged, the honest headline), at each
cell's own reference block size, **layer-cluster interval only**:

| cell | `sym_acc` | 90% (layers) | `e2_acc` | 90% (layers) | at risk? |
|---|---|---|---|---|---|
| llama31-8b @32k | 34.0 | [29.2, 38.7] | 45.1 | [40.1, 50.2] | **crosses GO** |
| qwen3-8b @8k | 16.5 | [13.5, 19.5] | 40.5 | [34.9, 46.1] | **crosses STOP *and* GO** |
| qwen3-30b @8k | 14.1 | [10.5, 17.7] | 34.8 | [30.1, 39.5] | **crosses STOP *and* GO** |
| llama31-8b @128k | 20.2 | [15.6, 24.9] | 29.2 | [24.6, 33.8] | widest prompt spread (§3.3) |
| llama33-70b @128k | 21.6 | [17.5, 25.7] | 26.1 | [21.7, 30.4] | clear |
| qwen3-30b @128k | **7.5** | [4.9, 10.2] | 20.8 | [16.5, 25.0] | **clear STOP** |

18.6% was an E2 number for a cell that reads **7.5%** on the target the paper
must state. Spending ~18 GPU-h there buys precision on a verdict nothing
threatens; llama33-70b @128k (40 GPU-h) is the same story. Meanwhile three
cheap cells — one at 32k and two at 8k — carry labels the data cannot support.

Note also what the symmetric grid does to a headline: **"no configuration is
STOP" is false on the symmetric cell** (qwen3-30b reads 7.5 @128k, 8.7 @32k,
14.1 @8k; qwen3-8b 11.6–16.5).

### B5 · The corner set moves a band further than any seed

Within one run, on identical rows (`errorbars.py`, five-corner runs
job929911/13/40):

| cell | run | `e2_acc` (accum alone) | `e2_min5` (min over 5) | shift |
|---|---|---|---|---|
| qwen3-30b @128k | job929913 | 17.8 | 14.5 | −3.3 |
| qwen3-30b @128k | job929940 | 17.5 | 14.5 | −3.1 |
| llama31-8b @128k, block 0–2 | job929911 | 24.6 | 21.4 | −3.2 |
| llama31-8b @128k, block 3–5 | job929911 | 35.9 | 29.1 | −6.8 |

R3-report §3.4 asked for exactly this: *"R7's error bars must include the corner
set, not only seeds."* R7 gets it free by running five corners and
reconstructing the accum-only columns (§2).

### B6 · A replicate in the R3 configuration would be adopted by the R3/R6 guards

`bugs/2/script_temp.sh` and `bugs/6/script.sh` accept any
`results/job*/h0_<model>_<ctx>.parquet` with corner tag `or-ac_f` and the
`floor_maxb` marker, and `boundary.py` averages replicates of a cell at equal
weight regardless of prompt count or overlap. R7 sidesteps both by running the
**five-corner set** (tag `or-la-ac-wi-re_f`), which no guard matches and which
`boundary.py`'s `--evictors oracle,accum` filter drops by itself (§6, S6).

### B7 · The sidecar did not record which sample a run measured

`n_prompts` was top level, `rot_seed` only inside `config`, and the prompt range
was implicit because it always started at 0. Fixed additively by S3:
`prompt_offset`, `prompt_block` and `rot_seed` are now stamped in the sidecar
and per row.

### B8 · The one true sentence, now outdated

`SIEVE_N_PROMPTS` / `SIEVE_ROT_SEED` have been forwarded since 2026-09-18. The
only knob missing for R7 was `prompt_offset` (S1/S2).
`submit_h0_ctx_sweep.slurm` reads `SIEVE_*` from the environment only and does
not parse them as arguments, so it stays unusable on Trillium — R7, like
R4/R5/R6, uses `submit_h0.slurm` / `submit_h0_large_models.slurm`, one cell per
job.

### B9 · The haystack is keyed on (corpus, prompt index) — nowhere recorded until now

`prompts.corpus_window` shuffles the book order with
`_seed("corpus-order", corpus_sha(corpus_dir))`. **Re-stage the corpus and
prompt *k* becomes a different novel.** Measured on two runs of the same cell:

| | prompt 0 | prompt 1 | prompt 2 |
|---|---|---|---|
| job21400323 (`corpus b524da5e`) | martin-chuzzlewit @183792 | crime-and-punishment @227979 | jane-eyre @65727 |
| job929911 (`corpus 0a26bc1e`) | anna-karenina @519712 | wealth-of-nations @226889 | brothers-karamazov @1146388 |

Two consequences. (1) Any "same prompts, different run" comparison **across
corpora is a prompt comparison, not a rerun or rotation one** — including
R3-report §3.5's "across campaigns … within 1.4 points", which is better
evidence than it claims to be, for a different claim. (2) R7's `--control`
block is a rotation control **only if its `corpus_sha` matches the R3 re-run's**.
`errorbars.py` groups samples by `(corpus_sha, prompt indices)` and warns when a
cell mixes corpora; `script.sh` gate C compares the staged corpus against the
R3 re-run's sidecars and says so before any GPU time. (As of today this
checkout's corpus is `0a26bc1e` while the R3 re-run used `b524da5e`.)

---

## 2. What R7 ships

> An interval, not a point, for every band the paper prints — covering the
> three things that actually move a verdict: **which prompts** were drawn,
> **which corners** were configured, **which heads** were measured. The
> quantizer seed is the noise floor, not the error bar.

Targets, all at B=3, band = gain ≥ 2× (`report.BAND_MIN`):

| key | column | meaning |
|---|---|---|
| `sym_acc` | `gain_pp3_accum` | symmetric cell, accum corner — **the headline** |
| `sym_min5` | `gain_pp3_accum` of a five-corner run | symmetric cell, min over five corners |
| `e2_acc` | reconstructed | practical corner (accum), oracle interior |
| `e2_min5` | `gain_best_practical3` of a five-corner run | the corner-set alternative |
| `oracle` | `gain_best3` | oracle corner and interior |
| `routed` | median routed gain | R2's robust statistic, same blocks |

**Reconstruction, verified exactly** (max \|diff\| = 0.0 against the real columns
on `job21406450/h0_qwen3-30b-a3b-2507_131072.parquet`), so one five-corner run
serves both corner sets:

```
e2_acc  = min(err_uniform3, err_e3_accum_frac) / err_wf3          == gain_best_practical3
sym_acc = min(err_uniform3, err_e3_accum_frac) / err_wf_pp3_accum == gain_pp3_accum
```

(`alloc.py`: `e_best` is the min over the *configured* practical corners, and the
per-corner `err_e3_<name>_frac` columns give any single-corner version back.)

---

## 3. What is already measured — 0 GPU, on parquets on disk

### 3.1 The symmetric grid, with layer-cluster intervals

§1 B4's table. Produced by:

```bash
.venv/bin/python h0_measurement/bugs/7_error_bars_and_seeds/errorbars.py \
   "h0_measurement/results/job214217*/h0_*.parquet" \
   --block-size 2 --targets sym_acc,e2_acc --csv h0_measurement/reports/r7_grid.csv
```

### 3.2 Prompt-block spread, from the runs on disk

Cutting each landed cell into 2-prompt blocks (a noisier statistic — for
*sizing* the effect and choosing cells, not for the table). sd across blocks, in
band points:

| cell | `sym_acc` sd | `e2_acc` sd |
|---|---|---|
| llama31-8b @128k | 4.0 | **9.6** |
| llama31-8b @32k | 4.9 | 4.0 |
| llama31-8b @8k | 4.0 | 0.5 |
| qwen3-8b @8k | 1.7 | 5.9 |
| qwen3-30b @8k | 1.3 | 1.8 |
| llama33-70b @128k | 0.6 | 0.6 |
| qwen3-30b @128k | 1.0 | 0.6 |

And the older per-prompt jackknife on the E2 cells (job21406450, job21400323):
per-prompt bands 17.7–24.9 (qwen3-30b @128k), 20.6–30.8 (llama33-70b @128k),
**22.6–42.9** (llama31-8b @128k); jackknife se 1.2–3.9 points, against
**0.18–0.55** for in-campaign replicates of an identical configuration
(R3-report §3.5).

### 3.3 The corner-set component

§1 B5's table: −3.1 to −6.8 points, within one run, on identical rows.

### 3.4 The four components, ordered

| component | size today | how R7 measures it | GPU |
|---|---|---|---|
| prompts | sd 0.6–9.6 pts, worst at 128k | disjoint blocks of the reference size | yes |
| heads / layers | ±3–5 pts (90%) | layer-cluster bootstrap (`errorbars.py`, `boundary.py`) | 0 |
| corner set | 3.1–6.8 pts | five-corner run, accum-only columns reconstructed | rides along |
| rotation + rerun | ≤0.6 pts | `--control` block against the R3 cell, same corpus | opt-in |

**The ordering is the finding:** the component the ROADMAP proposed to measure
is the smallest, by an order of magnitude, and the one it did not mention (the
corner set) is larger than the gap it called decisive.

---

## 4. Design

Each R7 job runs the **R3 interior and policy**
(`SIEVE_INTERIOR_SCORES=accum`, `SIEVE_CORNER_POLICIES=frac`) with the
**five-corner** evictor set and `SIEVE_ROT_SEED=1`, over one **disjoint prompt
block** of the cell's reference size (6 for main tier, 4 for large):

```
 block 0   prompts 0 … n-1     the R3 cell itself -- already measured, not re-run
                               (--control re-runs it at rot_seed 1 for the
                                rotation component, only worth it when the
                                corpus_sha matches)
 block 1   prompts n … 2n-1    independent sample
 block 2   prompts 2n … 3n-1   independent sample
```

**One job per block, not one job per cell.** `run_h0.py` writes its parquet once,
at the end, so a job cancelled on walltime loses everything it measured; per
block, a timeout costs one block, the blocks queue in parallel, and block 0 is
not re-paid. `--one-job` is the fallback for a checkout without the
`prompt_offset` knob (gate P refuses the default there rather than silently
measuring prompts 0…n−1 twice).

**Blocks are disjoint *windows*, and here also disjoint *books*:** the staged
corpus has 36 full-window books at 128k and 40 at 32k/8k, against 18 prompt
indices, so no block reuses a document. Gate C prints the counts, and
`errorbars.py` names any overlap it finds.

---

## 5. Configurations

Submit from **trig-login01**; every override is an **argument** (Trillium's
`sbatch` is a shell function carrying `--export=NONE`); no `--mem`.

### 5.1 Already on disk — the reference cells (0 GPU)

| id | cell | config | n_prompts | rot | targets | status | result folder |
|---|---|---|---|---|---|---|---|
| REF-1 | llama31-8b @32768 | `or-ac_f` + `floor_maxb` | 6 | 0 | sym, e2, oracle | done | `results/job21421771` |
| REF-2 | qwen3-8b @8192 | `or-ac_f` + `floor_maxb` | 6 | 0 | sym, e2, oracle | done | `results/job21421769` |
| REF-3 | qwen3-30b-a3b-2507 @8192 | `or-ac_f` + `floor_maxb` | 4 | 0 | sym, e2, oracle | done | `results/job21421770` |
| REF-4 | llama31-8b @131072 | `or-ac_f` + `floor_maxb` | 6 | 0 | sym, e2, oracle | done | `results/job21421773` |
| REF-5 | qwen3-30b @131072, llama33-70b @131072 | `or-ac_f` + `floor_maxb` | 4 | 0 | sym, e2, oracle | done | `results/job21421774` |
| REF-6 | qwen15-moe @32768 | `or-ac_f` + `floor_maxb` | 6 | 0 | sym, e2, oracle | done | `results/job21421771` |
| REF-7 | five-corner runs of the 128k cells | `or-la-ac-wi-re_fa` | 4 / 6 | 0 | e2, oracle (corner-set component) | done | `results/job929911`, `job929913`, `job929940` |

### 5.2 To submit — chosen from §1 B4, not from the ROADMAP

Walltime: `s/unit × units × 1.25 + 10 min (main) / 15 min (large)`, rounded up
to 15 min; units = prompts × 3 families; rates are the five-corner + interior
(g2) column of `bugs/2/script_temp.sh`, scaled by L^0.3.

| id | cell | block (offset+n) | GPUs | walltime | why | status | result folder |
|---|---|---|---|---|---|---|---|
| A1 | llama31-8b @32768 | 6+6, 12+6 | 1 | 01:45:00 | sym 34.0 crosses **GO** | not submitted | |
| A2 | qwen3-8b @8192 | 6+6, 12+6 | 1 | 01:30:00 | sym 16.5 crosses **STOP**, e2 40.5 crosses **GO** | not submitted | |
| A3 | qwen3-30b-a3b-2507 @8192 | 4+4, 8+4 | 4 | 01:15:00 | sym 14.1 crosses **STOP**, e2 34.8 crosses **GO** | not submitted | |
| B | llama31-8b @131072 | 6+6, 12+6 | 1 | 02:30:00 | widest prompt spread (sd 9.6); the ctx the phase story rests on | not submitted | |
| C | qwen15-moe-a2.7b @32768 | 6+6, 12+6 | 1 | 00:45:00 | `--near-line`: e2 38.1 crosses GO (sym clear) | not submitted | |
| D | llama33-70b @131072 | 4+4 | 4 | 06:30:00 | `--with-70b`: **not recommended**, clear of both lines | not submitted | |
| W | qwen3-30b-a3b-2507 @131072 | 4+4, 8+4 | 4 | 02:30:00 | `--wide`: the ROADMAP's cell; sym 7.5 is a clear STOP | not submitted | |
| ctrl | qwen3-8b @8192 | 0+6 at rot_seed 1 | 1 | 01:30:00 | `--control`: rotation component; needs the R3 corpus_sha | not submitted | |

Default set = A1+A2+A3+B, **~11 GPU-h** (A3 is the only 4-GPU job). The
ROADMAP's plan (qwen3-30b and llama33-70b at 128k) would have cost ~58 GPU-h on
two cells that are not in doubt.

### 5.3 Commands

```bash
bash h0_measurement/bugs/7_error_bars_and_seeds/script.sh --run --dry   # check first
bash h0_measurement/bugs/7_error_bars_and_seeds/script.sh --run
bash h0_measurement/bugs/7_error_bars_and_seeds/script.sh --run --near-line
bash h0_measurement/bugs/7_error_bars_and_seeds/script.sh --read
```

The sheet gates on: R7's unit tests (`test_prompt_offset` + the R3 ones), the
`prompt_offset` knob being present in the code this cluster runs (gate P), the
corpus (gate C: sha256, which corpus, how many full-window books), and the R3
fresh-token fix working on real attention (gate G — passing today at 1.23×). It
skips any block that already has a complete result with the same
`(corner tag, rot_seed, prompt block)`, so re-running it submits only what is
missing.

---

## 6. Shared code

**Applied** (additive, default-off; a queued or running job behaves exactly as
before, because every new key defaults to today's value):

| id | file | change |
|---|---|---|
| **S1** | `run_h0.py` | `prompt_offset`: prompts run `offset … offset+n_prompts-1`. Default 0 ⇒ byte-identical to every earlier run. |
| **S2** | `submit_h0.slurm`, `submit_h0_large_models.slurm` | forward `SIEVE_PROMPT_OFFSET`; first log line now echoes `prompts=<n>@<offset> rot_seed=<s>`; RUN_INFO.txt records the sample knobs. |
| **S3** | `run_h0.py` | stamp `prompt_offset`, `prompt_block`, `rot_seed` in the sidecar and per row (B7). |
| **S7** | `ROADMAP.md` §3 R7 | rewritten around the measured symmetric grid. |
| — | `tests/test_units.py` | `test_prompt_offset` (8 checks) pins S1/S3: offset 0 unchanged, blocks disjoint, prompt identity independent of every seed, band(1) > band(6). |

**Flagged, not changed** — each touches a file a running campaign reads, or a
reader whose output is being quoted now:

| id | file | change | when |
|---|---|---|---|
| **S4** | `bugs/6/boundary.py` | consume R7's per-cell CIs (its docstring already says "Neither includes seed variance (R7)"), and stop averaging replicates of unequal `n_prompts` / overlapping prompts at equal weight | after R6's own read has been done once |
| **S5** | `report.py` | print band ± CI; R2 also wants median routed gain as the primary y-axis | after the R3 array's chained report job has run |
| **S6** | `bugs/2/script_temp.sh`, `bugs/6/script.sh` | `fixed()` should also require `rot_seed == 0` and the expected `n_prompts`/`prompt_offset`, so an R7 replicate is never adopted as the R3/R6 cell (B6) | with the next edit to those sheets; R7 sidesteps it meanwhile via the five-corner tag |
| **S8** | `bugs/4/script.sh` | run the ctx points at the tier's default `n_prompts`, or apply the k-correction before pooling them into R6's fit (B3) | before R6's boundary fit is quoted |
| **S9** | `prefetch_corpus.py` | `--verify` exits before `--check-ctx`/`--n-prompts`, so those flags are silently ignored with it (R7 does the window arithmetic itself in gate C) | any time; cosmetic but it has already misled one preflight |

---

## 7. Files here

| file | state | what it does |
|---|---|---|
| `plan.md` | this file | |
| `script.sh` | written, dry-run clean | the guarded sheet for §5.2 (`--run`, `--near-line`, `--wide`, `--with-70b`, `--control`, `--one-job`, `--dry`, `--read`), with the decision table written before the run |
| `errorbars.py` | written, exercised on every landed cell | the reader: cuts runs into equal-size prompt blocks, six targets, the four components separately, a 90% interval, and a verdict line that says when an interval straddles STOP or GO. Warns on mixed corpora and shared documents. Emits `reports/r7_errorbars.csv` for S4. |
| `../../reports/r7_grid.csv` | written | the symmetric screen of §3.1 |

---

## 8. Dependencies

- **R3 — hard, satisfied.** The symmetric target needs `floor_maxb`; gate G
  confirms the fix on real attention (1.23× today). The R3 cells are R7's
  `rot_seed=0` reference blocks and are not re-measured.
- **R6 — soft, both ways.** R7 produces the intervals `boundary.py` says it
  lacks; R6 produces the fit they go into. R7's five-corner tag keeps its runs
  out of R6's pool deliberately.
- **R5 — none.** Different decode shape; shares only the queue.
- **R4 — soft.** Its ctx points run at `n_prompts` 2–3 and are pooled by R6;
  S8 is the correction. R7's intervals also say whether R4's single points are
  separable at all.
