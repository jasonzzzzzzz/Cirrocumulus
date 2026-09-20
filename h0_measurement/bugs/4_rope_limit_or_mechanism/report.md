# R4 — is the long-context collapse a property of attention, or of running a model at its trained limit?

**The question.** Every model that reaches 128k drops sharply there. Two
readings fit the old data equally well:

- **ABSOLUTE-L** — the decay is a property of attention. The method's reach
  genuinely ends near 64k, and "ways this dies #1" is real.
- **ROPE-FRAC** — part of the slope is an artifact of measuring models *at their
  trained limit*. The two models that collapsed were at 100% of their RoPE
  window; the one with 2× headroom barely moved.

**Status 2026-09-20.** Five of eight cells landed (`job214447*`). They are
enough to state a preliminary answer, **on a statistic the band fraction cannot
provide**, and the answer leans ROPE-FRAC. The three decisive large-tier cells —
including both qwen3-30b headroom runs, the ones designed to settle it — are not
on disk (§1). Nothing here is final.

Numbers in this report pool the R4 cells with the R3 fixed campaign
(`job214217*`), which ran the identical corner config (`or-ac_f`, floor_maxb),
so each model's curve joins across campaigns.

---

## 0. Findings

1. **The band fraction cannot answer R4 for most models — it saturates.**
   qwen3-1.7b reads 3.3% in band at 8k and 3.3% at 32k: there is no room left
   to fall. Three of seven models never exceed 17% at any context they were
   measured at. The unsaturated statistics —
   dead-2 tier fraction and τ — carry the signal instead. (This is R2's point
   arriving from a second direction.)
2. **Every model that reaches 100% of its RoPE window accelerates in its final
   octave; the one model that never reaches it does not.** Dead-2 rate, earlier
   segments → final segment:

   | model | rope_frac reached | dead-2 rate before → at the cap |
   |---|---|---|
   | llama33-70b | 1.00 | +2.0 → **+15.9** pts/octave (×7.8) |
   | llama31-8b | 1.00 | +6.0 → **+33.8** (×5.6) |
   | qwen3-1.7b | 1.00 | +1.7 → **+6.2** (×3.7) |
   | qwen3-8b | 1.00 | +4.7 → **+8.1** (×1.7) |
   | **qwen3-30b-a3b-2507** | **0.50 (never)** | +2.8 → **+2.9** (flat) |

3. **The control does the work ABSOLUTE-L cannot explain.** qwen3-1.7b
   accelerates at **40,960 tokens** — nowhere near 64k–128k, where the absolute
   reading puts the cliff. Its only distinguishing feature there is that it has
   run out of trained positions.
4. **The new 96k bracket point splits llama31-8b's collapse.** Its steepest
   drop is not the 64k→128k octave as a whole: dead-2 rises +12.6 points over
   32k→96k (64k tokens) and **+14.1 points over 96k→128k (32k tokens)**. The
   collapse is concentrated in the last quarter of the window.
5. **rope_frac sets the slope, not the level.** At rope_frac = 1.00 the
   symmetric band ranges from 55.5% (mistral-7b @32k) to 2.5% (qwen3-1.7b
   @41k). Where a model sits is architecture (C1's dead-tier axis); how fast it
   falls is what the window fraction modifies.
6. **The honest caveat is in §2.6**: dead-2 is a threshold statistic, so its
   *rate* accelerates wherever a model's τ distribution crosses the threshold,
   cap or no cap. The missing qwen3-30b runs are what separate the two.

---

## 1. What ran, and what is missing

| § | cell | rope_frac | job | status |
|---|---|---|---|---|
| A | qwen3-30b @196,608 | 0.75 | — | ❌ **not on disk** |
| A | qwen3-30b @262,144 | 1.00 | — | ❌ **not on disk** |
| B | llama31-8b @98,304 | 0.75 | 21444723 | ✅ 73,728 rows, needle 3/3 |
| B | llama33-70b @98,304 | 0.75 | — | ❌ **not on disk** |
| C | qwen3-1.7b @8,192 | 0.20 | 21444725 | ✅ 8,064 rows, needle 3/3 |
| C | qwen3-1.7b @16,384 | 0.40 | 21444726 | ✅ needle **2/3** |
| C | qwen3-1.7b @32,768 | 0.80 | 21444727 | ✅ needle 3/3 |
| C | qwen3-1.7b @40,960 | 1.00 | 21444728 | ✅ needle 3/3 |

All landed cells: corner `oracle,accum`, policy `frac`, interior `accum`,
`floor_maxb` present, dense decode, 3 prompts, PG-19 haystack, all three
families.

**The three missing cells are exactly the three large-tier (4-GPU) jobs**, and
the job-id sequence (…721, 722, **723** = llama31-8b, **724**, 725–728 =
qwen3-1.7b) shows ids were allocated for them: they were submitted and then
either failed at run time or are still queued — 4-GPU jobs wait longer than the
1-GPU ones that finished. This machine cannot see that queue; §6 step 1 is the
check.

---

## 2. Results

### 2.1 The cells

Symmetric = both sides lagged (R3's honest cell); E2 = corner lagged only;
dead-2 = heads whose 2-bit tier is dead; τ, ladder and n₉₅ are per-head medians.

| model | ctx | rope_frac | symmetric band | E2 band | dead-2 | τ | n₉₅ |
|---|---|---|---|---|---|---|---|
| llama33-70b | 8k | 0.06 | 87.4 | 92.8 | 5.5 | 1.52 | 40 |
| llama33-70b | 32k | 0.25 | 70.0 | 75.4 | 9.6 | 1.94 | 76 |
| llama33-70b | 128k | **1.00** | 21.3 | 25.2 | 41.4 | 3.32 | **4,861** |
| llama31-8b | 8k | 0.06 | 54.5 | 67.5 | 18.8 | 2.08 | 169 |
| llama31-8b | 32k | 0.25 | 34.0 | 45.1 | 27.0 | 2.41 | 282 |
| **llama31-8b** | **96k** | **0.75** | **26.7** | **36.6** | **39.6** | **2.81** | **815** |
| llama31-8b | 128k | **1.00** | 20.2 | 29.2 | 53.7 | 2.95 | 1,239 |
| mistral-7b | 8k | 0.25 | 66.2 | 76.7 | 10.4 | 2.00 | 474 |
| mistral-7b | 32k | **1.00** | 55.5 | 66.0 | 20.3 | 2.21 | 1,600 |
| qwen15-moe | 8k | 0.25 | 41.4 | 55.5 | 25.8 | 2.42 | 251 |
| qwen15-moe | 32k | **1.00** | 30.2 | 37.0 | 36.8 | 3.10 | 649 |
| qwen3-8b | 8k | 0.20 | 16.5 | 40.5 | 50.5 | 2.60 | 157 |
| qwen3-8b | 32k | 0.80 | 11.5 | 26.7 | 59.8 | 3.08 | 323 |
| qwen3-8b | 41k | **1.00** | 12.2 | 26.6 | 62.4 | 3.35 | 528 |
| **qwen3-1.7b** | **8k** | 0.20 | 3.3 | 24.3 | 77.7 | 2.59 | 105 |
| **qwen3-1.7b** | **16k** | 0.40 | 3.3 | 22.1 | 79.2 | 2.74 | 159 |
| **qwen3-1.7b** | **32k** | 0.80 | 3.3 | 15.8 | 81.0 | 3.13 | 315 |
| **qwen3-1.7b** | **41k** | **1.00** | 2.5 | 15.8 | 83.0 | 3.35 | 372 |
| qwen3-30b | 8k | 0.03 | 14.1 | 34.8 | 62.0 | 2.17 | 96 |
| qwen3-30b | 32k | 0.12 | 8.1 | 25.3 | 67.7 | 2.39 | 266 |
| qwen3-30b | 128k | 0.50 | 7.5 | 19.3 | 73.5 | 2.83 | 719 |

### 2.2 The band axis is the wrong instrument here

qwen3-1.7b — the control the design leans on — reads 3.3, 3.3, 3.3, 2.5% in
band across a 5× context range. It is in STOP territory everywhere, so a cliff
has nothing to fall into. The same floor affects qwen3-8b (11–16%) and
qwen3-30b (7–14%).

Dead-2 and τ are monotone and unsaturated over the same range (77.7 → 83.0 and
2.59 → 3.35 for the control), so §2.3 reads the segments there. This is not a
workaround: dead-2 is C1's order parameter, and τ is the variable C2 says
context acts through.

### 2.3 The within-model test: rate before the cap vs rate at the cap

Per-octave rates, each model's own curve (full per-segment table in §2.1's
source, `/tmp` analysis reproduced by the command in §6):

| model | segment | rope_frac | dead-2 /oct | τ /oct | symmetric band /oct |
|---|---|---|---|---|---|
| llama31-8b | 8k→32k | 0.06→0.25 | +4.1 | +0.167 | −10.3 |
| llama31-8b | 32k→96k | 0.25→0.75 | +8.0 | +0.248 | −4.6 |
| llama31-8b | **96k→128k** | **0.75→1.00** | **+33.8** | **+0.350** | **−15.5** |
| llama33-70b | 8k→32k | 0.06→0.25 | +2.0 | +0.209 | −8.7 |
| llama33-70b | **32k→128k** | **0.25→1.00** | **+15.9** | **+0.688** | **−24.3** |
| qwen3-1.7b | 8k→16k | 0.20→0.40 | +1.6 | +0.147 | 0.0 |
| qwen3-1.7b | 16k→32k | 0.40→0.80 | +1.8 | +0.390 | 0.0 |
| qwen3-1.7b | **32k→41k** | **0.80→1.00** | **+6.2** | **+0.673** | −2.8 |
| qwen3-8b | 8k→32k | 0.20→0.80 | +4.7 | +0.239 | −2.5 |
| qwen3-8b | **32k→41k** | **0.80→1.00** | **+8.1** | **+0.833** | +2.4 |
| qwen3-30b | 8k→32k | 0.03→0.12 | +2.8 | +0.110 | −3.0 |
| qwen3-30b | 32k→128k | 0.12→0.50 | +2.9 | +0.223 | −0.3 |

Four models cross into their last 20–25% of window and all four accelerate
(×1.7 to ×7.8 on dead-2). The fifth never gets past half its window and its rate
is flat to two significant figures (+2.8 → +2.9).

### 2.4 The control is the part ABSOLUTE-L cannot absorb

qwen3-1.7b's acceleration happens between 32,768 and 40,960 tokens. Under
ABSOLUTE-L nothing should happen there — the collapse is supposed to live near
64k–128k. The only thing distinguishing that segment is that the model has run
out of trained positions. Its band is floored (§2.2) and its dead-2 is nearing
its own ceiling (83%), so τ is the cleanest reading: +0.269/oct before,
**+0.673/oct** in the final segment.

### 2.5 The bracket point, and what n₉₅ does at the cap

llama31-8b @96k is the first measurement inside the 64k→128k step. It shows the
step is not uniform: dead-2 adds 12.6 points over the 64k tokens from 32k→96k,
then 14.1 points over the 32k tokens from 96k→128k.

n₉₅ (tokens carrying 95% of attention) is the other tell. At the cap it does not
merely grow, it explodes: llama33-70b goes 76 → **4,861** between 32k and 128k,
llama31-8b 815 → 1,239 across its last quarter-window. Attention genuinely
diffuses as trained positions run out.

### 2.6 The alternative explanation, stated plainly

**Dead-2 is a threshold statistic** — the fraction of heads with τ²c₂ > 1 — so
its rate accelerates wherever a model's τ distribution sweeps through the
threshold, whether or not the cap is involved. Two facts keep that from
explaining everything: the control accelerates on **τ itself** (§2.4), and
qwen3-30b's rate stays flat while its τ rises through the same range that
accelerates other models. But the clean separation is exactly what the missing
cells provide: qwen3-30b at 0.75 and 1.00 of its window moves rope_frac while
absolute L moves the other way relative to every other model.

Three more caveats, in the order they would bite:

1. The final segments are short (0.32–0.42 octave for three of the four), so
   per-octave normalisation amplifies whatever curvature exists.
2. The R4 cells ran at 3 prompts; the R3 cells they join ran at 4–6. Aggregates
   reproduce to ~1.4 band points across campaigns (R3-report §2.7), which is
   smaller than every effect above, but it is not zero.
3. qwen3-1.7b @16k retrieved 2/3 needles. One weak prompt in a tier-`debug`
   model — it does not invalidate the cell, but the control deserves the same
   scrutiny as a headline cell if it is going to carry §2.4.

---

## 3. What this would mean for C2

If the missing cells confirm §2.3, C2 becomes **two axes rather than one**:

- context length acts through τ, as C2 already says, and
- **fraction of the trained window consumed** modifies the slope, so a
  deployment at 50% of window behaves unlike one at 100% *at the same absolute
  length*.

That is narrower than the current claim and more actionable: it says the 128k
row is evidence about extrapolation, not only about long-context attention, and
it predicts that a model with headroom keeps its band further out. It also
explains the old anomaly that motivated R4 — the one model with 2× headroom
dropped least — without appealing to architecture.

If instead qwen3-30b collapses at 192k with its window only 75% consumed, the
absolute reading stands, C2 needs no restatement, and "the method's reach ends
near 64k" is confirmed rather than explained away.

---

## 4. Runs to be accomplished

Shared config for every cell (identical to R3's fixed campaign, so the curves
pool):

```
SIEVE_EVICTORS=oracle,accum   SIEVE_CORNER_POLICIES=frac   SIEVE_INTERIOR_SCORES=accum
(+ SIEVE_CTX and SIEVE_N_PROMPTS per cell; families default niah,qa,cont)
```

| # | cell | rope_frac | slurm script | sbatch options | per-cell config | why |
|---|---|---|---|---|---|---|
| A1 | qwen3-30b @196,608 | 0.75 | `submit_h0_large_models.slurm` | `--array=0-0 --gpus-per-node=4 --time=03:00:00` | `SIEVE_CTX=196608 SIEVE_N_PROMPTS=3` | **the decisive cell**: moves rope_frac past 0.5 while absolute L passes every other model's cap |
| A2 | qwen3-30b @262,144 | 1.00 | `submit_h0_large_models.slurm` | `--array=0-0 --gpus-per-node=4 --time=02:30:00` | `SIEVE_CTX=262144 SIEVE_N_PROMPTS=2` | the cap itself; ROPE-FRAC predicts the steep segment lands here |
| B2 | llama33-70b @98,304 | 0.75 | `submit_h0_large_models.slurm` | `--array=0-0 --gpus-per-node=4 --time=03:15:00` | `SIEVE_CTX=98304 SIEVE_N_PROMPTS=3` | splits the 70B's 2-octave segment, the widest gap in §2.3, and shares the cell with bugs/6 group C |
| — | llama31-8b @96k | 0.75 | — | — | — | **done** (job21444723) |
| — | qwen3-1.7b ×4 | 0.20–1.00 | — | — | — | **done** (job21444725–728) |

Optional, only if §2.3 survives the three above:

| # | cell | rope_frac | script / options | why |
|---|---|---|---|---|
| D1 | qwen3-30b @65,536 | 0.25 | large, `--time=02:00:00`, `SIEVE_N_PROMPTS=3` | its curve has a 2-octave hole at 32k→128k; without it the "flat rate" claim rests on two segments |
| D2 | llama31-8b @96k at 6 prompts | 0.75 | main, `--time=01:15:00`, `SIEVE_N_PROMPTS=6` | removes the prompt-count difference from the bracket point (§2.6 caveat 2) |
| D3 | mistral-7b @16k | 0.50 | main, `--time=01:00:00`, `SIEVE_N_PROMPTS=6` | mistral has only two points, both endpoints; a midpoint gives a second "before the cap" rate on a model whose band is NOT floored |

Cost: A1+A2+B2 ≈ 35 GPU-h (4 GPUs each). D1–D3 ≈ 10 GPU-h more.

---

## 5. How to read the result

Read the **rate** of dead-2 and τ per octave inside each model, never the band
level across models (§2.2, §2.5).

| outcome | reading |
|---|---|
| qwen3-30b's rate stays flat through 0.75 and then jumps at 1.00 | **ROPE-FRAC.** Restate C2 as two axes (§3); the 128k row becomes evidence about extrapolation. The control (§2.4) already points here. |
| qwen3-30b's rate jumps at 192k, with the window only 75% consumed | **ABSOLUTE-L.** C2 stands as written; "the reach ends near 64k" is confirmed; §2.3's pattern was dead-2's threshold nonlinearity. |
| the rate rises smoothly at 0.75 and again at 1.00, no discontinuity | **BOTH.** rope_frac is a covariate of C2 rather than a replacement — report it as a second deployment variable and say so. |
| 192k/256k fails the input-validity gate (needle not retrieved) | the cell is not a phase measurement at all. A model that cannot use its own window is a finding, but report it as that, not as a band point. |

Whatever the outcome, §2.2 stands on its own: **the band fraction is the wrong
y-axis for R4**, and any future version of this figure should plot dead-2 or τ.

---

## 6. Next steps

1. **Find out what happened to the three large-tier jobs** (this machine cannot
   see that queue). On the submitting machine:
   ```
   squeue -u $USER -o "%.12i %.20j %.8T %.10M %.10l %R"
   sacct -S 2026-09-19 -u $USER -o JobID%18,JobName%22,State,Elapsed,Timelimit,ExitCode | grep -v '\.ba\|\.ex'
   ```
   If they are `PENDING`, wait — nothing else to do. If they `FAILED` or
   `TIMEOUT`, read `h0_measurement/logs/h0large_<jobid>_0.{out,err}`: line 1
   must echo `ctx=196608|262144|98304 evictors=oracle,accum`, and the last line
   before death says whether it was the ctx guard, the corpus preflight, VRAM,
   or the wall clock.

2. **Re-submit whatever is not running**, from `trig-login01`, with no `--mem`:
   ```
   cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
   bash h0_measurement/bugs/4_rope_limit_or_mechanism/script.sh
   ```
   The sheet is not guarded, so it re-submits **everything** including the five
   finished cells. To submit only the missing three, run the three lines from
   §4 directly (they are lines 103, 104 and 110 of that script).

3. **Verify within a minute of each start**: log line 1 echoes the ctx, and the
   run prints `RoPE window 262,144 … this run uses 75%|100% of it`. For the
   256k cell, check the corpus line: 22/40 books cover a 262,144-token window,
   against `n_prompts=2`.

4. **Re-read the whole set** once they land:
   ```
   .venv/bin/python h0_measurement/report.py \
       "h0_measurement/results/job214217*/*.parquet" \
       "h0_measurement/results/job214447*/*.parquet" \
       "h0_measurement/results/<NEW_A1>/*.parquet" \
       "h0_measurement/results/<NEW_A2>/*.parquet" \
       "h0_measurement/results/<NEW_B2>/*.parquet" \
       -o h0_measurement/reports/h0_rope_vs_length.pdf
   ```
   `page_rope` plots each model against both axes and prints where its steepest
   per-octave drop falls. **Read its output against §5, not against the band
   level**, and cross-check the rate table with the analysis that produced §2.3
   (it is a short pandas script over the same parquets; regenerate it rather
   than trusting this file's copy if the cells change).

5. **Then update**: this report's §2–§3 with the decisive cells, `ROADMAP.md`
   R4 (it still says "PLUMBED, ready to submit"), and — only if §2.3 survives —
   C2's wording in the proposal.

---

## 7. Open issues

- **`bugs/4/script.sh` is unguarded.** Every other sheet in `bugs/` now skips
  cells that already have a complete result; this one does not, so running it
  again pays for the five finished cells a second time. Worth porting the
  `fixed()` guard from `bugs/6_pin_sharp_boundary/script.sh` before the next
  submission.
- **The control has no dynamic range on the band axis.** qwen3-1.7b was chosen
  for headroom (5× window) and cheapness, but it is STOP everywhere, so it can
  only ever speak through dead-2 and τ. If a control with both headroom *and* a
  live band is available, it would carry §2.4 far better.
- **qwen3-30b is the only model in the registry with real headroom**, so the
  within-model test rests entirely on it. A second headroom model would turn
  §3's restatement from a two-model claim into a real one.
- **The prompt-count difference** between the R4 cells (3) and the R3 cells they
  pool with (4–6) is small relative to the effects but should be removed from
  the bracket point if it ends up in the paper (D2 in §4).
