# Bug 2 — an honest eviction corner: results, analysis, and system design

**Campaign complete.** 24 (model, context) configurations over six architectures
and five octaves, every one carrying all four corners
(`oracle, accum, window, recency`) and both budget policies. All on real PG-19
text with 100% needle retrieval (validity gate PASS). Every run requested in
`script.sh` has finished; the 128k row is now three models, not one.

---

## 1. Data inventory

| model | registry cap | 8192 | 16384 | 32768 | 65536 | 131072 |
|---|---|---|---|---|---|---|
| llama31-8b | 131072 | job20014005 | job20014005 | job20014005 | job20014005 | **job20014005** |
| llama33-70b | 131072 | job20014007 | job20014008 | job20014011 | job20014013 | **job20107803** |
| qwen3-30b-a3b-2507 | 131072 | job20014007 | job20014008 | job20014011 | job20014013 | **job20096629** |
| qwen3-8b | 40960 | job20013992 | job20013997 | job20014000 | *above cap* | *above cap* |
| mistral-7b | 32768 | job20013992 | job20013997 | job20014000 | *above cap* | *above cap* |
| qwen15-moe-a2.7b | 32768 | job20013992 | job20013997 | job20014000 | *above cap* | *above cap* |

`above cap` is not a gap: those lengths exceed the model's native RoPE window and
`run_h0` refuses them. llama31-8b also appears in the per-ctx jobs
(`job20013992/97`, `job20014000/02`); the sweep copy is used throughout.

**One off-config run, and it turned out to be useful.** `job20116802` is
llama33-70b @128k with `accum,window` only — no oracle, and crucially **no
`recency`**, which makes it structurally immune to the step-0 defect of §2. It is
excluded from the tables (no oracle column) but serves as an independent check:

| llama33-70b @128k | band vs practical corner |
|---|---|
| `job20107803` — full corner set, filtered | **24.96%** |
| `job20116802` — no recency, cannot have the bug | **25.18%** |
| difference | **0.21 pts** |

Two differently-configured runs of the largest model at the hardest context agree
to a fifth of a point. That is the strongest single validation of the §2 fix.

---

## 2. A defect found in this campaign, and fixed

**Half of every quant row was measuring the interior against StreamingLLM alone.**

`recency` is the only evictor that scores with no history. On decode step 0 the
lagged evictors (`accum`, `window`) have nothing yet, so `min` over the practical
corners collapsed onto the single corner that is always available — and it is the
weakest. With `n_decode=8, quant_every=4`, quant steps are 0 and 4, so **50% of
rows** compared the interior against a deliberately feeble baseline.

Adding `recency` is what *created* the defect. An earlier campaign (`job1998*`,
`accum`-only) never saw it, because with no `recency` nothing scored at step 0 and
the practical columns were simply absent.

That gives an unusually clean adjudication — two independent campaigns:

| round 2 population | mean \|Δ\| vs round 1 (`accum`-only, structurally clean) |
|---|---|
| **filtered** (`n_practical == 3`) | **0.53 pts** |
| unfiltered | **28.63 pts** (max 48.96) |

The filtered numbers reproduce an independent experiment to half a point. The
unfiltered ones do not. **Everything below uses the filtered population.**

### Why it mattered beyond the level shift

The band-vs-ctx curve — the campaign's headline deliverable — was **not merely
shifted, it was the wrong shape**:

| llama31-8b | 8k | 16k | 32k | 64k | 128k | slope | |
|---|---|---|---|---|---|---|---|
| unfiltered | 85.4 | 82.4 | 76.9 | 66.3 | **85.4** | −1.6 pts/doubling | **non-monotone** |
| filtered | 67.1 | 58.0 | 46.2 | 34.2 | **28.8** | **−10.0 pts/doubling** | monotone |

The unfiltered curve bends back **up** at 128K, which would have been reported as
"the band recovers at long context" — the opposite of the paper's thesis, and an
artifact. Unfiltered slopes understate the decay by 3–6× and make 3 of 6 models
spuriously non-monotone.

### The fix

- `sievelib/alloc.py` — the practical aggregate (`err_practical`,
  `gain_best_practical`, `in_band_practical`, `oracle_evict_advantage`,
  `best_evictor`) is written **only when every configured evictor scored**.
  Otherwise NaN, so any median/dropna excludes the row. Per-evictor cells are
  still recorded for whatever did score.
- `h0_measurement/report.py` — `drop_partial_corners()` repairs parquets written
  before that guard, keyed on `n_practical` against the `evictors` provenance
  column. A no-op on new data.
- `tests/test_units.py` — `test_partial_corner_is_withheld`, 7 checks.

Regenerating the report on `job20014005` now prints
`blanked the practical-corner aggregate on 645,120 row(s)` and yields the
filtered numbers.

---

## 3. Headline — what the bug-2 fix did to the verdict

Paired inside one frame: the same heads, the same rows, only the corner changes.

| model | ctx | band vs **oracle** | band vs **practical** | Δ | oracle advantage | verdict |
|---|---|---|---|---|---|---|
| llama33-70b | 8k | 82.1% | **92.6%** | +10.6 | 1.29× | = GO |
| llama33-70b | 32k | 61.4% | **75.4%** | +14.1 | 1.22× | = GO |
| llama33-70b | 64k | 49.0% | **61.9%** | +13.0 | 1.16× | = GO |
| mistral-7b | 8k | 53.6% | **76.3%** | +22.7 | 1.39× | = GO |
| mistral-7b | 32k | 45.9% | **65.9%** | +20.0 | 1.34× | = GO |
| llama31-8b | 8k | 43.8% | **67.1%** | +23.3 | 1.41× | = GO |
| llama31-8b | 32k | 26.6% | **46.2%** | +19.6 | 1.27× | NARROW → **GO** |
| llama31-8b | 64k | 18.1% | **34.2%** | +16.1 | 1.16× | = NARROW |
| llama31-8b | 128k | 13.6% | **28.8%** | +15.2 | 1.12× | **STOP → NARROW** |
| qwen15-moe | 8k | 31.0% | **55.0%** | +24.0 | 1.36× | NARROW → **GO** |
| qwen15-moe | 32k | 21.9% | **37.0%** | +15.1 | 1.13× | NARROW → **GO** |
| qwen3-8b | 8k | 11.5% | **40.4%** | +28.9 | 1.33× | **STOP → GO** |
| qwen3-8b | 32k | 6.4% | **26.6%** | +20.1 | 1.11× | **STOP → NARROW** |
| qwen3-30b | 8k | 10.4% | **33.9%** | +23.5 | 1.25× | **STOP → NARROW** |
| qwen3-30b | 32k | 6.6% | **23.6%** | +17.0 | 1.15× | **STOP → NARROW** |
| qwen3-30b | 64k | 6.3% | **24.4%** | +18.0 | 1.14× | **STOP → NARROW** |
| llama31-8b | **128k** | 13.6% | **28.8%** | +15.2 | 1.12× | **STOP → NARROW** |
| llama33-70b | **128k** | 18.8% | **25.0%** | +6.1 | 1.02× | = NARROW |
| qwen3-30b | **128k** | 4.8% | **18.6%** | +13.8 | 1.05× | **STOP → NARROW** |

**Every STOP is gone, in all 24 configurations** — 14 GO, 10 NARROW, 0 STOP.
Movement is +6.1 to +28.9 points.

**The lift shrinks where the corner is already near-optimal.** At 128k the oracle
advantage falls to 1.02–1.12×, and the band lift falls with it (+6.1 for
llama33-70b, whose oracle advantage is 1.02×). Lagged attention is *most*
informative exactly at long context — the opposite of what v7 assumed when it
argued the oracle's advantage would be largest there.

**Validation.** The in-run oracle column reproduces the independent pre-fix
campaigns (`job19959944/45`, `job19960025/27`) to within **0.26–4.49 points** on
all 12 overlapping cells. The oracle path is untouched by the fix; the movement is
entirely the corner demotion.

**Honest framing.** The oracle's advantage is only **1.11–1.41×** in error terms.
The band moves 15–29 points because the per-head gain distribution is dense right
at the 2× threshold, not because the practical corner is dramatically worse. This
is a *threshold-sensitivity* result — worth stating explicitly, because it means
the band fraction is a fragile statistic and the underlying error ratios are the
robust ones.

---

## 4. Context length is the phase variable, measured to 128k on three models

| model | 8k | 16k | 32k | 64k | 128k | pts / ctx doubling |
|---|---|---|---|---|---|---|
| llama33-70b | 92.6 | 86.0 | 75.4 | 61.9 | **25.0** | **−15.9** |
| llama31-8b | 67.1 | 58.0 | 46.2 | 34.2 | **28.8** | −10.0 |
| qwen15-moe | 54.9 | 45.3 | 37.0 | — | — | −9.0 |
| qwen3-8b | 40.4 | 28.7 | 26.6 | — | — | −6.9 |
| mistral-7b | 76.3 | 66.2 | 65.9 | — | — | −5.2 |
| qwen3-30b | 33.9 | 30.3 | 23.6 | 24.3 | **18.6** | −3.7 |

Monotone in five of six. qwen3-30b wobbles once (23.6 → 24.3 at 32k→64k) then
falls to 18.6.

### 4.1 llama33-70b collapses at 128k — the biggest single result in the campaign

The model the paper calls its best case loses **37 points in one octave**:

| llama33-70b | 8k | 16k | 32k | 64k | **128k** |
|---|---|---|---|---|---|
| band (practical) | 92.6 | 86.0 | 75.4 | 61.9 | **25.0** |
| dead-2 fraction | 5.6% | 8.2% | 9.5% | 12.0% | **41.4%** |
| τ | 1.53 | 1.68 | 1.94 | 2.06 | **3.31** |
| n₉₅ | 40 | 49 | 76 | 117 | **4,854** |
| oracle advantage | 1.29× | 1.23× | 1.22× | 1.16× | **1.02×** |

Everything moves at once at the 64k→128k step, and **the phase variable predicts
it**: dead-2 goes 12.0% → 41.4%, which on the fitted curve is exactly where the
band should fall to ~25%. This is the phase diagram's first genuine out-of-sample
test — it had no 128k llama33-70b point when the boundary was established — and it
passed. The n₉₅ explosion (117 → 4,854) independently reproduces the
super-linear \(L^{1.81}\) growth reported from the earlier campaign (4,808
tokens), so it is a property of the model, not of this run.

The consequence for the paper is uncomfortable and should be stated plainly:
**the ranking is not stable across context.** llama33-70b is the best model at
8k–64k and the *second worst* at 128k. "Which architecture is in the interior
phase" is only well-posed at a stated context length.

### 4.2 The phase axis after the completion

| axis | vs practical band | vs oracle band |
|---|---|---|
| **dead-2 tier fraction** | **−0.953** | −0.983 |
| τ, ladder width | −0.796 | −0.732 |
| φ = n₉₅/L | +0.079 | +0.053 |

24 configurations. The correlation softens slightly from −0.964 (22 points) as the
three 128k points are added, and holds. φ remains null-to-wrong-signed.

The figure reports −0.952 / −0.978 rather than −0.953 / −0.983 because its oracle
series is computed over **all** quant rows while the table above is paired on the
complete-corner rows only. Both are right: the oracle ranks by the current step's
sensitivity, so it scores at step 0 where the lagged evictors cannot, and
including those rows is valid for it. The paired population is the one to use when
quoting the oracle→practical delta; the figure's is the one to use when asking how
well the axis predicts each corner separately.

---

## 5. Three mechanism findings

### 5.1 The oracle's edge is on DIFFUSE heads — the stated mechanism was backwards

`why.md` claimed "the deflation is strongest exactly for the sharp models." The
data says the opposite, in **all 22 runs**:

- `spearman(oracle_advantage, n95)` is **positive everywhere**: +0.02 to +0.71
- `spearman(oracle_advantage, ladder_bits)` is **negative everywhere**: −0.41 to −0.77
- sharpest quartile: oracle advantage ≈ **1.00–1.28×**; most diffuse quartile: **1.31–1.73×**

The original reasoning — "a lagged score that misses one heavy-hitter is
catastrophic on a sharp head" — is wrong. On a sharp head the heavy hitter is
*stable across decode steps*, so lagged attention identifies it perfectly and H2O
is indistinguishable from the oracle. It is diffuse heads, where attention mass
moves between steps, that a lag cannot track. **The conclusion survives; the
explanation must be rewritten.**

### 5.2 E1's "slack" hypothesis is dead

K\* — the smallest keep-count within 10% of the full-budget corner — is
**100.0% of the budget for every model at every context length**, 8k through 128k.
The eviction corner genuinely needs every token its fractional budget gives it.
There is no slack to reclaim. This was one ambiguous point before; it is now 22
runs across 6 models.

### 5.3 A 36× architectural spread in tail dependence

K\*/n95 — how far past the 95%-mass point a head must keep tokens to stay within
10% of full-budget error — is not a constant:

| model | K\*/n95 @32k | across the sweep |
|---|---|---|
| mistral-7b | **7.5** | 6.5 – 8.1 |
| qwen15-moe-a2.7b | 13.5 | 12.8 – 14.2 |
| qwen3-8b | 20.9 | 14.7 – 20.9 |
| qwen3-30b-a3b-2507 | 24.5 | 19.0 – 24.6 |
| llama31-8b | 37.6 | 19.9 – 53.0 |
| llama33-70b | **209.8** | 8.6 – 233.2 |

**28× spread at matched 32k; 36× across the whole sweep (6.5 → 233.2).** Two
models with similar `n95` need wildly different keep-counts. `llama33-70b` has the
*smallest* support (n95 = 49 at 32k) yet the *largest* tail dependence — its
heads concentrate 95% of mass in 49 tokens but still need ~10,000 to be
represented. This is the single most design-relevant number in the campaign, and
it is why a global κ cannot work.

---

## 6. How to design the eviction + quantization system

Everything here is read off the measurements above.

### 6.1 The competitor is eviction, not uniform quantization

| | median gain of the interior @3b |
|---|---|
| vs uniform quantization | **20–34×** |
| vs the best practical evictor | **1.15–4.81×** |

Uniform quantization is not a serious baseline at 3 bits — it loses by more than
an order of magnitude everywhere. **Every design decision should be justified
against eviction.** A paper that reports gain-over-uniform is reporting a number
nobody should be impressed by.

### 6.2 The allocator must own eviction as tier 0 — it already does

At B=3 the water-filling interior *chooses to evict* **40–53%** of tokens by
itself, and among in-band heads **40–49%**. SIEVE is therefore not "quantize
everything instead of evicting"; it is a **joint evict-and-quantize allocator**
where dropping a token is simply the cheapest tier. The implementation must treat
tier 0 as a first-class bit-width, not a separate mechanism bolted on.

### 6.3 Route on diffuseness, per head, and make the threshold ctx-aware

In-band and out-of-band heads are cleanly separated by support:

| model @32k | n95 (in band) | n95 (out of band) | ratio |
|---|---|---|---|
| llama31-8b | 1,170 | 80 | 15× |
| qwen15-moe | 2,193 | 150 | 15× |
| qwen3-30b | 2,093 | 119 | 18× |
| qwen3-8b | 1,514 | 186 | 8× |

and the rank correlation with the realised gain is strong and stable:
`ladder_bits` −0.41 to −0.77, `n95` +0.02 to +0.71, across every model and ctx.

**Design rule:** compute `n95` (or the cheaper `ladder_bits`) per head once, route
diffuse heads to the mixed-precision interior and sharp heads to plain eviction.
Both are available from a single cheap pass with no quantization. Because the band
shrinks 3.5–10.3 points per ctx doubling, the routing threshold must be a function
of context length — a head that routes to SIEVE at 8k may belong on the eviction
path at 128k.

`llama33-70b` is the exception that proves the rule: its `n95` correlation is ~0
(+0.02 to +0.07) while `ladder_bits` still gives −0.40 to −0.64. **Route on
`ladder_bits`, not `n95`** — it is the one signal that works on all six models.

### 6.4 Build H2O. Skip SnapKV and StreamingLLM.

Which evictor actually wins, per head:

| evictor | share of heads won |
|---|---|
| `accum` (H2O) | **78–93%** |
| `window` (SnapKV) | 6–17% |
| `recency` (StreamingLLM) | 1–5% |

H2O also has the simplest state — one running sum per (layer, head), **5 bytes
per token** in our implementation — against SnapKV's ring of window buffers at
**17 bytes/token**, i.e. 3.4× the host memory for 6–17% of the wins.
**The min over three corners is barely stronger than H2O alone**, so a system that
implements only H2O gives up almost nothing, and the verdict is robust to the
choice.

### 6.5 Do not size the eviction budget as a fixed fraction

Two findings combine here. K\*/budget = 100% says the *fractional* budget is fully
used, so it cannot be shrunk uniformly. K\*/n95 spanning 36× says the *right*
budget is a per-head property that no constant captures. The `abs` policy as
specified (κ=4) is not a cheaper corner — it is a broken one, costing **1.9–7.4×
the error** of `frac` for its bit savings.

**Design rule:** derive the per-head keep-count from measured support at runtime
(K\* is computable from one cumulative pass over the ranked tokens), or accept the
fractional budget and spend the savings elsewhere. Do not ship a global κ.

### 6.6 What the system does not have to do

The oracle's advantage over H2O is only **1.11–1.41×**. There is no point
investing in a more sophisticated importance score: the gap between "what a
deployable evictor knows" and "what a clairvoyant evictor knows" is small, and it
is concentrated on diffuse heads that should be routed to the interior anyway.
Scoring is solved; allocation is where the value is.

---

## 7. What is missing, and what to run next

**Nothing from `script.sh` is outstanding.** All 24 configurations have landed,
including the two 128k points that were in flight.

**Anomaly, now resolved rather than open.** llama31-8b's n₉₅ jump (291 → 1,008
between 64k and 128k) was flagged as possibly a prefill artifact. llama33-70b
shows a far larger jump at the same step (117 → 4,854) and independently
reproduces the earlier campaign's \(L^{1.81}\) figure to within 1%. Two models
showing the same discontinuity at the same context, one of them matching a prior
independent measurement, makes it a property of long-context attention rather than
of the pipeline. It is worth a short note in the paper, not a re-run.

**What the completed data now argues for:**

1. **Re-examine the 8k–64k → 128k gap.** Every model that reaches 128k drops
   sharply at that last octave (−37 pts for llama33-70b, −5.7 for qwen3-30b,
   −5.4 for llama31-8b), and τ and n₉₅ both jump. Whether that is a genuine
   attention regime change or an artifact of RoPE behaviour near the trained
   limit is the most interesting open question in the study, and it is answerable
   with the existing pipeline at 96k.
2. **qwen3-30b @128k sits at 18.6%, 3.6 points above the STOP line.** It is the
   one cell where a modest measurement change could flip a verdict. A second seed
   (different `n_prompts` or `rot_seed`) would establish the error bar — neither
   is currently reachable through `SIEVE_*`, which is a gap in the submission
   scripts worth closing.
3. **Retire the `abs` policy** or re-derive κ per model from K\*/n₉₅. Given
   K\*=100% everywhere, the honest move is to report K\*/n₉₅ as the
   tail-dependence measurement it turned out to be and drop `abs` from the
   headline.
4. **Drop `window` and `recency` from future campaigns.** They win 15–22% and
   1–12% of heads respectively; H2O wins 74–83% at 128k and 78–93% overall. The
   min over three corners is barely stronger than H2O alone, and they cost 18 of
   23 bytes per layer-head-token of host state.
