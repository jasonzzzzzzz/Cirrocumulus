# R5 — phase drift across decode

**Question (C4).** A router is supposed to read each head's phase from *one*
offline calibration pass. Does that phase survive a long generation? And, as a
second duty, does τ's rise with L happen *inside* a generation, where the cache
grows by the tokens the model itself writes (C2)?

**Status 2026-09-20.** Campaign 1 ran and is readable, but two defects cap what
it can say, and both are now fixed in code. Campaign 2 (§4) re-runs every drift
cell and adds the measurement C4 actually needs. Nothing below is a final
verdict; §2 is what the *valid* columns of campaign 1 already show.

Files here: `script.sh` (the submission sheet), `drift.py` (the reader),
`results.txt` (campaign-1 reader output).

---

## 0. Findings so far

1. **The route does drift, but not far above the noise.** Heads whose route
   (interior vs baseline) differs from the calibration step run 13–30% at the
   last clean step, against a same-run noise floor of 8–23%. The excess is
   real but small, and it is largest at short context.
2. **Rank order decays steadily**: Spearman(gain at t, gain at calibration)
   falls from 1.00 to 0.61–0.77 by step 1–2k in every cell.
3. **τ moves within a generation — but not by as much as cache growth
   predicts, nor in proportion to it.** Measured Δτ is +0.17 to +0.31 at every
   context, while the across-ctx slope predicts +0.011 to +0.081. **The drift
   tracks the content the model writes, not the length of its cache.** That is
   a different claim from C2's, and it is the most interesting thing in the
   campaign.
4. **Banning EOS bought steps and spent them on repetition.** At step 4,096 the
   text had degenerated in 3/4 sampled cells. Any "drift" measured there is a
   loop (§2.1).
5. **The C4 number was never measured.** Campaign 1 priced re-budgeting from
   *one step ago* (`last_step`); nothing priced keeping the *prefill-time*
   allocation t steps later. The new `first` evictor does (§3.1).
6. **The sparse schedule's corner is not a handicap.** `last_step` (TOVA) is a
   slightly *stronger* corner than `accum` (H2O) on the same rows, so a sparse
   run's band reads ~5 points **low**, not high (§2.4).

---

## 1. What ran (campaign 1)

| cell | job | config | outcome |
|---|---|---|---|
| llama31-8b @8k, sampled | 21406669 | sparse, 14 measured steps to 4,096 | ✅ 43,008 rows |
| llama31-8b @32k | 21406670 | " | ✅ |
| llama31-8b @128k | 21406671 (+21417146 replicate) | " | ✅ |
| qwen3-30b @32k | 21424195 | ", fresh-token fix present | ✅ |
| qwen3-30b @32k, first try | 21406672 | " | ❌ empty — preflight `interior_scores` crash, since fixed |
| bridge, llama31-8b @8k | 21406673 | **dense** 33 steps, corners `oracle,accum,last_step` | ✅ still valid, not re-run |
| greedy control, llama31-8b @8k | 21406674 | sparse, T = 0 | ✅ |

All: `cont` family only, 3 prompts, EOS banned, corner `oracle,last_step`,
interior `last_step`, budget 3 b/token. Every niah gate is irrelevant here
(`cont` has no needle); the validity gate for these runs is the text itself
(§2.1).

---

## 2. Results

### 2.1 The generations looped — and that is what step 4,096 was measuring

Distinct-4-gram fraction of the last 256 generated tokens, per prompt, at step
4,096 (below ~0.5 is a loop):

| cell | per-prompt distinct-4 | reader verdict |
|---|---|---|
| qwen3-30b @32k | **0.05, 0.27, 0.10** | truncated at step 1,024 |
| llama31-8b @8k sampled | **0.04**, 0.61, 0.92 | truncated at 4,096 |
| llama31-8b @32k | **0.45**, 0.87, 0.68 | truncated at 4,096 |
| llama31-8b @8k greedy | 0.52, 0.77, **0.15** | loops from ~512 |
| llama31-8b @128k | 0.95, 0.89, 0.94 | clean to 4,096 |

A loop is a real attention regime, but not the one a deployed sampler lives in,
and it reads as phase drift. The 128k cell stays clean because a long prompt
sustains fresh text — which is also why 128k is the cell that can answer C4 over
the whole sweep (§3.3).

### 2.2 Route drift against its own noise floor

Per-head medians over the 3 prompts; `flip_rtr` = heads whose route differs from
the calibration step (step 1); floor = the same statistic between step 1 and the
next measured step, which is noise, not drift.

| cell | floor | last clean step | flip_rtr there | ρ(gain, gain@calib) | band: calib → last | p90 regret |
|---|---|---|---|---|---|---|
| llama31-8b @8k | 23% | 2,048 | **30%** | 0.61 | 66% → 42% | 1.00 |
| llama31-8b @32k | 11% | 2,048 | **16%** | 0.77 | 39% → 33% | 1.00 |
| llama31-8b @128k | 8% | 4,096 | **13%** | 0.64 | 22% → 12% | 1.00 |
| qwen3-30b @32k | 14% | 512 | **24%** | 0.66 | 29% → 45% | 2.96 |

**Reading.** The excess over the floor is 5–10 points everywhere: a real but
modest drift on a 3-prompt sample. p90 regret of keeping the calibration route
is 1.00 in the llama cells — the flips happen on heads near the 2× threshold,
where being on the wrong side costs almost nothing — but 2.96 on qwen3-30b,
whose band *rises* over the generation. The noise floor is high enough (8–23%)
that this is the campaign's weakest statistic; more prompts would sharpen it
(§7).

### 2.3 τ drifts with content, not with cache length

The across-ctx slope at calibration (+0.233 τ per octave of L) predicts how far
τ should move when a generation grows the cache. It does not:

| cell | tokens generated | cache growth | predicted Δτ | **measured Δτ** |
|---|---|---|---|---|
| llama31-8b @8k | 2,048 | +0.347 octave | +0.081 | **+0.310** |
| llama31-8b @32k | 2,048 | +0.095 | +0.022 | **+0.169** |
| llama31-8b @128k | 4,096 | +0.048 | +0.011 | **+0.248** |

Measured drift is 4–23× the prediction and roughly **constant in the cache
growth** — the opposite of what the LENGTH mechanism implies. If campaign 2
reproduces this with loops excluded by construction, R5's "double duty" resolves
against the length reading: what moves τ inside a generation is *what the model
is writing*, not how long its cache has become. C2's across-prompt slope is
untouched by this; it just does not transfer to within-generation growth.

### 2.4 The bridge: `last_step` vs `accum` on the same rows

Dense 33-step run, both corners measured together (job21406673):

| step | gain ratio last_step / accum | band under accum | band under last_step |
|---|---|---|---|
| 8 | 0.942 | 62.3% | 57.0% |
| 16 | 0.955 | 61.4% | 57.8% |
| 24 | 0.905 | 66.4% | 59.2% |
| 32 | 0.962 | 69.6% | 64.6% |

Ratio < 1 means `last_step` is the **stronger** corner, so the sparse runs'
band reads about 5 points **low** relative to the campaign's `accum` corner —
and the gap is flat in t, so nothing in §2.2 is a corner-substitution artifact.
This cell is not re-run.

### 2.5 What campaign 1 could not say

- **Interior columns are invalid** in every cell except qwen3-30b @32k: they
  predate the fresh-token fix, where an unseen position scored 0 and the
  allocator evicted the token appended that step (`bugs/2/R3-report.md` §2).
  `drift.py` blanks them on load.
- **The C4 quantity is absent.** `last_step` prices re-budgeting from one step
  ago. `lag:k` cannot reach k = 4,096 (it needs k buffers). See §3.1.

---

## 3. Implementation changes

| # | change | file | pinned by |
|---|---|---|---|
| 1 | `first` evictor: the first probed step's attention, frozen | `sievelib/evict.py` | `test_first_evictor` |
| 2 | persistent evictors survive sparse block resets | `sievelib/evict.py`, `run_h0.py` | `test_first_evictor` |
| 3 | persistent evictors survive an unprobed **gap** | `sievelib/evict.py` | `test_first_evictor` (regression) |
| 4 | repetition penalty + no-repeat-ngram, and an all-banned fallback | `run_h0.py` | `test_anti_loop_decoding` |
| 5 | `SIEVE_DECODE_REP_PENALTY`, `SIEVE_DECODE_NO_REPEAT` | `submit_h0*.slurm` | — |
| 6 | loop truncation, `froz` column and its horizon, verdict pinned to `last_step` | `drift.py` | — |

### 3.1 `first`: the frozen prefill-time score

C4's claim is about one calibration pass, so the quantity that prices it is the
error of the **prefill-time allocation applied at step t**. `lag:k` is the wrong
instrument (k buffers for k steps of lag). `first` records the first probed
step's attention and never updates: one snapshot plus one coverage mask, any t.
`interior_lag_cost3_first` (the reader's `froz`) is that price, and
`froz / lag1` is exactly what re-budgeting every step buys.

Positions the snapshot never saw are marked `unseen`, so the corner bumps them
and the interior floors them at the top tier rather than evicting them.

**Two bugs found while building it**, both caught before any GPU time:

- *Block resets re-calibrated it.* A sparse schedule restarts evictor history at
  each measured block, which is right for a window over recent steps and wrong
  for a frozen snapshot. Persistent evictors are now exempt.
- *An unprobed gap reset it anyway.* The cache grows by more than one position
  across a gap, and alignment treated that as "position identity lost". An
  end-to-end smoke run exposed it: `first` was byte-identical to `last_step`,
  and its unseen fraction read 1/L at step 4 instead of 4/L. Growth is the one
  shape where identity survives, so a persistent evictor now pads instead of
  resetting.

Verified end to end on CPU (Llama-3.2-1B, real PG-19 prompt): unseen fraction
0.00106 / 0.00423 / 0.01670 at steps 1 / 4 / 16 against t/L = 0.00106 / 0.00424
/ 0.01695, `froz` = `lag1` at step 1 by construction and 1.85× it by step 16.

### 3.2 Anti-loop decoding

`decode_rep_penalty` (the CTRL rule, applied to generated tokens) and
`decode_no_repeat_ngram` (blocks any token completing a repeated n-gram).
Campaign 2 uses 1.05 and 8: an 8-gram repeat is rare in book prose, so the block
binds on loops and almost nowhere else. If every candidate is banned, the step
falls back to the unbanned distribution rather than dead-ending.

### 3.3 The `froz` horizon — the constraint that shapes the design

The frozen score has never seen the t generated tokens, so the interior floors
them at `maxb`, costing `maxb·t` of the head's `B·L` bits:

| cell | floor's share of the budget at t = 4,096 | left for the rest |
|---|---|---|
| 8k | **89%** | 0.5 b/token |
| 32k | 30% | 2.38 b/token |
| 128k | 8% | 2.84 b/token |

Past roughly `t = 0.037·L` (B = 3, maxb = 8) `froz` prices the floor policy, not
staleness, so the reader blanks it there and prints the horizon: **~300 steps at
8k, ~1,200 at 32k, ~4,900 at 128k.**

**This splits the two questions across cells**: 128k answers C4 over the whole
sweep, 8k answers C2 inside a generation (where the cache grows most). Both run.

### 3.4 Two reader decisions worth knowing

- **Verdict pinned to `last_step`.** An interior score must also be an evictor,
  so `first` became a second corner and `gain_best_practical` silently turned
  into a min over `{last_step, first}` — a stronger competitor that would move
  the band against campaign 1 and the bridge. The reader recomputes
  `min(uniform, last_step)`, the same quantity in every campaign. The `first`
  *corner* columns are meaningless by construction (it bumps every unseen
  position) and must not be read.
- **Loops truncate the whole run**, not the offending prompt: dropping prompts
  individually leaves later steps resting on whichever prompts have not looped
  yet, which is itself a drift. Truncating keeps every step on the same prompts.

---

## 4. Campaign 2 — runs and configuration

Every cell shares this override set (`R5` in `script.sh`):

```
SIEVE_EVICTORS=oracle,last_step,first     SIEVE_CORNER_POLICIES=frac
SIEVE_INTERIOR_SCORES=last_step,first     SIEVE_FAMILIES=cont
SIEVE_MEASURE_STEPS=0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096
SIEVE_DECODE_TEMPERATURE=0.7  SIEVE_DECODE_TOP_P=0.9  SIEVE_DECODE_SEED=0
SIEVE_DECODE_BAN_EOS=1  SIEVE_DECODE_REP_PENALTY=1.05  SIEVE_DECODE_NO_REPEAT=8
SIEVE_NO_REPORT=1   (+ SIEVE_N_PROMPTS=3, corner tag `or-la-fi_f`)
```

| # | cell | slurm script | sbatch options | per-cell config | what it answers |
|---|---|---|---|---|---|
| A1 | llama31-8b @8k | `submit_h0.slurm` | `--array=0-0 --time=01:30:00` | `SIEVE_CTX=8192` | C2 inside a generation: +0.585 octave of cache growth, the largest LENGTH signal in the design |
| A2 | llama31-8b @32k | `submit_h0.slurm` | `--array=0-0 --time=02:00:00` | `SIEVE_CTX=32768` | the middle point of the length axis; `froz` valid to ~1,200 |
| A3 | llama31-8b @128k | `submit_h0.slurm` | `--array=0-0 --time=03:30:00` | `SIEVE_CTX=131072` | **C4 over the whole sweep** — `froz` valid to ~4,900, and the only cell that never looped |
| A4 | qwen3-30b @32k | `submit_h0_large_models.slurm` | `--array=0-0 --gpus-per-node=4 --time=04:30:00` | `SIEVE_CTX=32768` | second architecture (MoE), highest dead-tier model; its band *rose* in campaign 1 |
| A5 | qwen3-30b @8k | `submit_h0_large_models.slurm` | `--array=0-0 --gpus-per-node=4 --time=03:00:00` | `SIEVE_CTX=8192` | **new** — gives qwen a second ctx, without which the L-growth test cannot run for it |
| C | llama31-8b @8k, greedy | `submit_h0.slurm` | `--array=0-0 --time=01:30:00` | `SIEVE_DECODE_TEMPERATURE=0` (overrides the set) | sampling vs greedy, now that neither arm loops |
| B | bridge, llama31-8b @8k | — | — | dense `n_decode=33 quant_every=8`, corners `oracle,accum,last_step` | **skipped**: done in job21406673, corner-only, unaffected by both defects |

Cost: ~8.5 h on 1 GPU (A1–A3, C) + ~7.5 h on 4 GPUs (A4, A5) ≈ **38 GPU-h**.

The sheet is guarded: each line submits only if no complete result exists with
the same corner tag, temperature, schedule and anti-loop setting, so re-running
it after a partial failure submits only what is missing.

---

## 5. How to read campaign 2

**Route drift (C4, the router).** `flip_rtr` against its floor, and `rgr90`:

- within ~5 points of the floor through the sweep, regret ≈ 1 → one-pass
  calibration holds, and C4 gets a number instead of an assumption;
- growing steadily, at 128k as well as 8k → content drift; the router needs
  re-calibration every N tokens, and the slope says N;
- growing at 8k, flat at 128k, with Δτ matching the prediction → length drift,
  and the route can be pre-computed from the calibration pass itself.

**Allocation staleness (C4, the allocator).** `froz / lag1`, within the horizon:

- ≈ 1 → re-budgeting buys nothing; one pass covers the allocation too;
- growing with t, `froz` below the interior's edge → re-budget on a schedule
  (the cascade argument, and the expected outcome);
- `froz` above the interior's gain over the corner at some t → a
  prefill-calibrated interior stops paying there; report that t as a horizon.

**Length vs content (C2).** Compare measured Δτ with the printed prediction at
the last clean step of each ctx; §2.3 is the campaign-1 version of this table.

**Validity.** The reader truncates each run at the first step any prompt falls
to distinct-4 ≤ 0.5 and says where. If a cell still truncates early, the
anti-loop knobs were not enough: raise `SIEVE_DECODE_REP_PENALTY` toward 1.15
or lower `SIEVE_DECODE_NO_REPEAT` toward 6 and re-run **that cell** — do not
read the tail.

---

## 6. Next steps

1. **Sync the fixed code to the submitting machine** (nothing else uses these
   paths, and the gate below refuses to submit without them):
   `sievelib/evict.py`, `sievelib/alloc.py`, `h0_measurement/run_h0.py`,
   `h0_measurement/submit_h0.slurm`,
   `h0_measurement/submit_h0_large_models.slurm`, `tests/test_units.py`,
   and this folder's `script.sh` + `drift.py`.

2. **Check campaign 1's real runtimes** before trusting the headers (§4 raised
   two of them on an estimate):
   ```
   sacct -j 21406669,21406670,21406671,21406674,21424195 \
         -o JobID%18,State,Elapsed,Timelimit
   ```
   Scale every header by `Elapsed / old header` if any cell came within ~15 min
   of its limit.

3. **Submit** from `trig-login01` (a CPU login node rejects GPU jobs; pass no
   `--mem`). The sheet runs the unit-test gate first and refuses to submit if
   the fix is missing:
   ```
   bash h0_measurement/bugs/5_phase_drift_across_decode/script.sh --run
   ```
   Expect 6 submissions (A1–A5, C) and one `done` line for the bridge.

4. **Verify within a minute of each job starting** (section D of the sheet):
   log line 1 must echo `evictors=oracle,last_step,first` and the measure-step
   list, and the schedule line must end
   `EOS banned   rep_penalty=1.05   no_repeat_ngram=8`.

5. **Read it**:
   ```
   .venv/bin/python h0_measurement/bugs/5_phase_drift_across_decode/drift.py \
       "h0_measurement/results/<A1>/*.parquet" ... "h0_measurement/results/<C>/*.parquet" \
       --csv h0_measurement/reports/r5_drift.csv
   .venv/bin/python h0_measurement/bugs/5_phase_drift_across_decode/drift.py \
       "h0_measurement/results/job21406673/*.parquet" --calib 8     # the bridge
   ```

6. **Write §2 of this report from campaign 2**, then update ROADMAP R5 (it still
   describes campaign 1) and, if §2.3 holds up, C2's wording: the τ–L slope is
   an across-prompt fact that does not transfer to within-generation growth.

---

## 7. Open issues

- **Three prompts is thin** for a statistic whose noise floor is 8–23%. If the
  route-drift excess stays at 5–10 points, the next campaign should raise
  `SIEVE_N_PROMPTS` to 6 on the two cheapest cells rather than add cells.
- **One model family carries the length axis.** Only llama31-8b spans
  8k/32k/128k; qwen3-30b now has two points. A third architecture would make
  §2.3 a claim rather than an observation.
- **`froz` is horizon-limited by construction** (§3.3). Reporting it at 8k
  beyond ~300 steps requires a different policy for unseen tokens — e.g. a
  recency window at the top tier — which is a design choice, not a measurement
  fix.
- **The interior columns of campaign 1 stay invalid.** They are blanked on
  load; nothing in §2 depends on them.
