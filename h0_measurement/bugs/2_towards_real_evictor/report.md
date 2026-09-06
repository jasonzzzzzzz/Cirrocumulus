# Bug 2 — an honest eviction corner: results and system design

**Campaign complete.** 24 (model, context) configurations, six architectures,
8k–128k, every cell carrying all four corners (`oracle, accum, window, recency`)
and both budget policies. Real PG-19 text, 100% needle retrieval, validity gate
PASS everywhere.

**What this bug was.** H0 asks whether mixed-precision allocation (the "interior")
beats the better of two corners: uniform quantization, and eviction. The eviction
corner ranked tokens by the true current-step sensitivity `a_i·‖v_i−o‖` — which
needs the very attention weights eviction exists to avoid computing. **Every band
fraction the project had reported was measured against a baseline no deployable
system can build.** This report is what changed when the corner was rebuilt from
real evictors (H2O / SnapKV / StreamingLLM on lagged attention), with the oracle
retained beside it as a reported bound.

> **Reading convention.** Unless stated otherwise every number is a **per-head
> median over the complete-corner rows** (`quantized & n_practical == 3`), then
> aggregated across heads. §2 explains why that population. Where the oracle is
> quoted alongside, it is computed on the *same* rows so the pair is comparable.

---

## 1. Data inventory

| model | RoPE cap | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|
| llama31-8b | 131072 | job20014005 | job20014005 | job20014005 | job20014005 | job20014005 |
| llama33-70b | 131072 | job20014007 | job20014008 | job20014011 | job20014013 | job20107803 |
| qwen3-30b-a3b-2507 | 131072 | job20014007 | job20014008 | job20014011 | job20014013 | job20096629 |
| qwen3-8b | 40960 | job20013992 | job20013997 | job20014000 | *above cap* | *above cap* |
| mistral-7b | 32768 | job20013992 | job20013997 | job20014000 | *above cap* | *above cap* |
| qwen15-moe-a2.7b | 32768 | job20013992 | job20013997 | job20014000 | *above cap* | *above cap* |

*above cap* is not a gap — those lengths exceed the model's native RoPE window and
`run_h0` refuses them. llama31-8b also appears in `job20013992/97` and
`job20014000/02`; the sweep copy (`job20014005`) is used throughout so each model
contributes exactly one run per context.

**One off-config extra, which became a control.** `job20116802` is llama33-70b
@128k with `accum,window` only — no oracle, and crucially **no `recency`**, which
makes it structurally immune to the defect in §2. It is excluded from the tables
(no oracle column) but it independently checks the fix:

| llama33-70b @128k | band vs the practical corner |
|---|---|
| `job20107803` — full corner set, filtered per §2 | **24.96%** |
| `job20116802` — no `recency`, cannot have the defect | **25.18%** |
| difference | **0.21 pts** |

Two differently-configured runs of the largest model at the hardest context, 0.21
points apart. This is the strongest single validation that §2's filter is right.

---

## 2. A second defect, found in this campaign

**Half of every quant row was comparing the interior against StreamingLLM alone.**

### Why it happens

The verdict corner is `min` over the configured practical evictors — deliberately,
so the baseline is as strong as any deployable system could make it. But `recency`
(StreamingLLM) scores from *position alone* and needs no history, while `accum`
(H2O) and `window` (SnapKV) need at least one prior decode step. On step 0 they
return nothing, so the `min` collapses onto the one corner that is always
available — and it is the weakest of the three.

With `n_decode: 8, quant_every: 4` the quantization steps are 0 and 4. **Step 0 is
always a quant step and never has history**, so exactly 50% of quant rows measured
the interior against a deliberately feeble baseline.

**Adding `recency` is what created the defect.** The earlier `job1998*` campaign
used `accum` alone; with no `recency`, *nothing* scored at step 0, the practical
columns were absent, and those rows dropped out of every median by themselves.
That campaign was accidentally immune.

### How we know which population is right

Two independently-configured campaigns, same models, same contexts:

| round-2 population | mean \|Δ\| vs round 1 (`accum`-only, structurally immune) |
|---|---|
| **filtered** (`n_practical == 3`) | **0.53 pts** |
| unfiltered | **28.63 pts** (max 48.96) |

The filtered numbers reproduce an independent experiment to half a point; the
unfiltered ones do not. That settles it without appealing to judgement.

### Why it mattered beyond a level shift

The band-vs-context curve — this campaign's headline deliverable — was not merely
shifted, it was **the wrong shape**:

| llama31-8b | 8k | 16k | 32k | 64k | 128k | slope | |
|---|---|---|---|---|---|---|---|
| unfiltered | 85.4 | 82.4 | 76.9 | 66.3 | **85.4** | −1.6 /doubling | **non-monotone** |
| filtered | 67.1 | 58.0 | 46.2 | 34.2 | **28.8** | **−10.0 /doubling** | monotone |

The unfiltered curve bends back **up** at 128k. Reported as-is it would have said
"the band recovers at long context" — the opposite of the thesis, from an
artifact. Unfiltered slopes understate the decay by 3–6× and make three of six
models spuriously non-monotone.

### The generalisable lesson

**A baseline defined as a minimum over several methods silently becomes the
weakest of them wherever the others are undefined.** Any `min`/`max` over a set
whose members have different availability conditions needs an explicit
completeness guard.

### The fix

| file | change |
|---|---|
| `sievelib/alloc.py` | the practical aggregate (`err_practical`, `gain_best_practical`, `in_band_practical`, `oracle_evict_advantage`, `best_evictor`) is written **only when every configured evictor scored**; otherwise NaN, so medians exclude the row. Per-evictor cells still record whatever did score. |
| `h0_measurement/report.py` | `drop_partial_corners()` repairs parquets written before that guard, keyed on `n_practical` against the `evictors` provenance column. A no-op on new data. |
| `tests/test_units.py` | `test_partial_corner_is_withheld` — 7 checks. |

Reproduce:

```bash
python h0_measurement/report.py "h0_measurement/results/job20014005/*.parquet" -o /tmp/x.pdf
# note: blanked the practical-corner aggregate on 645,120 row(s)
```

---

## 3. Headline — what an honest corner does to the verdict

Paired inside one frame: same heads, same rows, only the corner definition changes.

| model | ctx | vs **oracle** | vs **practical** | Δ | oracle adv | verdict |
|---|---|---|---|---|---|---|
| llama33-70b | 8k | 82.1% | **92.6%** | +10.6 | 1.29× | = GO |
| llama33-70b | 16k | 74.4% | **86.0%** | +11.6 | 1.23× | = GO |
| llama33-70b | 32k | 61.4% | **75.4%** | +14.1 | 1.22× | = GO |
| llama33-70b | 64k | 49.0% | **61.9%** | +13.0 | 1.16× | = GO |
| llama33-70b | 128k | 18.8% | **25.0%** | +6.1 | 1.02× | = NARROW |
| mistral-7b | 8k | 53.6% | **76.3%** | +22.7 | 1.39× | = GO |
| mistral-7b | 16k | 47.1% | **66.2%** | +19.1 | 1.31× | = GO |
| mistral-7b | 32k | 45.9% | **65.9%** | +20.0 | 1.34× | = GO |
| llama31-8b | 8k | 43.8% | **67.1%** | +23.3 | 1.41× | = GO |
| llama31-8b | 16k | 35.6% | **58.0%** | +22.4 | 1.35× | NARROW → **GO** |
| llama31-8b | 32k | 26.6% | **46.2%** | +19.6 | 1.27× | NARROW → **GO** |
| llama31-8b | 64k | 18.1% | **34.2%** | +16.1 | 1.16× | = NARROW |
| llama31-8b | 128k | 13.6% | **28.8%** | +15.2 | 1.12× | **STOP → NARROW** |
| qwen15-moe | 8k | 31.0% | **54.9%** | +24.0 | 1.36× | NARROW → **GO** |
| qwen15-moe | 16k | 25.5% | **45.3%** | +19.8 | 1.27× | NARROW → **GO** |
| qwen15-moe | 32k | 21.9% | **37.0%** | +15.1 | 1.13× | NARROW → **GO** |
| qwen3-8b | 8k | 11.5% | **40.4%** | +28.9 | 1.33× | **STOP → GO** |
| qwen3-8b | 16k | 7.5% | **28.7%** | +21.3 | 1.16× | **STOP → NARROW** |
| qwen3-8b | 32k | 6.4% | **26.6%** | +20.1 | 1.11× | **STOP → NARROW** |
| qwen3-30b | 8k | 10.4% | **33.9%** | +23.5 | 1.25× | **STOP → NARROW** |
| qwen3-30b | 16k | 8.6% | **30.3%** | +21.7 | 1.21× | **STOP → NARROW** |
| qwen3-30b | 32k | 6.6% | **23.6%** | +17.0 | 1.15× | **STOP → NARROW** |
| qwen3-30b | 64k | 6.3% | **24.3%** | +18.0 | 1.14× | **STOP → NARROW** |
| qwen3-30b | 128k | 4.8% | **18.6%** | +13.8 | 1.05× | **STOP → NARROW** |

**Every STOP is gone: 14 GO, 10 NARROW, 0 STOP.** Movement +6.1 to +28.9 points.

**Validation.** The in-run oracle column reproduces the independent pre-fix
campaigns (`job19959944/45`, `job19960025/27`) to within **0.26–4.49 points** on
all 12 overlapping cells. The oracle path is untouched by the fix, so the movement
is entirely the corner demotion.

### Why the band moves so much when the corner barely changes

The oracle is only **1.02–1.41×** stronger in error terms, yet the band moves
6–29 points. That is not a contradiction: **the per-head gain distribution is
dense right at the 2× band threshold**, so a small shift in the denominator
reclassifies many heads. Two consequences worth carrying into any writeup:

1. **The band fraction is a fragile statistic.** It is a thresholded count; the
   underlying error ratios are the robust quantity.
2. **The lift shrinks where the corner is already near-optimal.** At 128k the
   oracle advantage falls to 1.02–1.12× and the lift falls with it (+6.1 for
   llama33-70b). Lagged attention is *most* informative at long context — the
   opposite of what the original plan assumed.

---

## 4. Context length is the phase variable

| model | 8k | 16k | 32k | 64k | 128k | pts / ctx doubling |
|---|---|---|---|---|---|---|
| llama33-70b | 92.6 | 86.0 | 75.4 | 61.9 | **25.0** | **−15.9** |
| mistral-7b | 76.3 | 66.2 | 65.9 | — | — | −5.2 |
| llama31-8b | 67.1 | 58.0 | 46.2 | 34.2 | **28.8** | −10.0 |
| qwen15-moe | 54.9 | 45.3 | 37.0 | — | — | −9.0 |
| qwen3-8b | 40.4 | 28.7 | 26.6 | — | — | −6.9 |
| qwen3-30b | 33.9 | 30.3 | 23.6 | 24.3 | **18.6** | −3.7 |

Monotone in five of six; qwen3-30b wobbles once (23.6 → 24.3 at 32k→64k) then
falls to 18.6. **Context is the steepest axis in the study — steeper than
architecture.**

### 4.1 llama33-70b collapses at 128k

The best model at every other length loses **37 points in one octave**:

| llama33-70b | 8k | 16k | 32k | 64k | **128k** |
|---|---|---|---|---|---|
| band (practical) | 92.6 | 86.0 | 75.4 | 61.9 | **25.0** |
| dead-2 tier fraction | 5.6% | 8.2% | 9.5% | 12.0% | **41.4%** |
| τ | 1.53 | 1.68 | 1.94 | 2.06 | **3.31** |
| n₉₅ | 40 | 49 | 76 | 117 | **4,854** |
| oracle advantage | 1.29× | 1.23× | 1.22× | 1.16× | **1.02×** |

Everything moves at the same step, and **the phase variable predicts it**: dead-2
goes 12.0% → 41.4%, which on the established boundary is exactly where the band
should land near 25%. This is the phase diagram's first genuine **out-of-sample
test** — no 128k llama33-70b point existed when the boundary was drawn — and it
passed. The n₉₅ explosion independently reproduces the earlier campaign's
super-linear L^1.81 growth (4,808 tokens) to within 1%, so it is a property of the
model, not of this run.

**Consequence, stated plainly: the architecture ranking is not stable across
context.** llama33-70b is the best model at 8k–64k and the second worst at 128k.
"Which phase is this architecture in" is only well-posed at a stated context
length.

### 4.2 The phase axis survives the corner change

| axis | vs practical band | vs oracle band |
|---|---|---|
| **dead-2 tier fraction** | **−0.953** | −0.983 |
| τ, ladder width | −0.796 | −0.732 |
| φ = n₉₅/L | +0.079 | +0.053 |

24 configurations. Softens slightly from −0.964 (22 points) as the three 128k
points are added, and holds. φ remains null-to-wrong-signed.

*Figure note:* `docs/fig5_phase.png` reports −0.952 / −0.978 rather than
−0.953 / −0.983 because its oracle series is computed over **all** quant rows
while the table above is paired on complete-corner rows. Both are correct — the
oracle ranks by the current step's sensitivity, so it *does* score at step 0 where
the lagged evictors cannot, and including those rows is valid for it. Use the
paired population when quoting the oracle→practical delta; use the figure's when
asking how well the axis predicts each corner separately.

---

## 5. Mechanism findings

### 5.1 The oracle's edge is on DIFFUSE heads — the original reasoning was backwards

`why.md` claimed "the deflation is strongest exactly for the sharp models". The
data says the opposite, in **24 of 24 runs**:

- `spearman(oracle_advantage, n₉₅)` is **positive in all 24**: +0.19 to +0.68
- sharpest quartile: oracle advantage **1.00–1.33×**
- most diffuse quartile: **1.10–1.72×** — larger than the sharp quartile in **24/24**

**Why the original reasoning failed.** It argued that "a lagged score that misses
one heavy-hitter is catastrophic on a sharp head". But on a sharp head the heavy
hitter is *stable across decode steps*, so lagged attention identifies it
perfectly and H2O is indistinguishable from the oracle (1.00× on the sharpest
quartile). It is diffuse heads, where attention mass moves between steps, that a
one-step lag cannot track.

**The conclusion survives; the explanation had to be rewritten.**

### 5.2 The "corner wins on slack" hypothesis is refuted

K\* — the smallest keep-count landing within 10% of the full-budget corner's error
— is **100.0% of the budget in 24 of 24 runs**, 8k through 128k. The eviction
corner genuinely needs every token its fractional budget gives it. There is no
slack to reclaim.

The `abs` policy as specified (κ=4) is therefore not a cheaper corner but a broken
one, costing **1.9–7.4× the error** of `frac` for its bit savings.

### 5.3 Tail dependence is architectural and spans 37×

K\*/n₉₅ — how far past the 95%-mass point a head must keep tokens to stay within
10% of full-budget error:

| model | K\*/n₉₅ range | tail dependence |
|---|---|---|
| mistral-7b | **5.8 – 7.2** | shallow — broad support, short tail |
| qwen15-moe-a2.7b | 9.2 – 14.1 | shallow |
| qwen3-8b | 11.7 – 19.0 | moderate |
| qwen3-30b-a3b-2507 | 17.9 – 27.2 | moderate |
| llama31-8b | 15.4 – 40.4 | deep |
| llama33-70b | 8.6 – **214.7** | very deep at ≤64k; drops at 128k as n₉₅ explodes |

**37× spread** (5.8 → 214.7), and it is *not* ordered by support: llama33-70b has
the smallest n₉₅ in the study at ≤64k (40–117 tokens) and the largest tail
dependence, needing ~10,000 tokens to stay within 10%. Two models can concentrate
95% of attention mass identically and still need order-of-magnitude different
keep-counts.

**This is why no global κ can work**, and it is the most design-relevant number in
the campaign.

---

## 6. How to design the eviction + quantization system

Every rule below is read directly off the measurements above.

### 6.1 Beat eviction, not uniform quantization

| interior's median gain @3b | |
|---|---|
| vs uniform quantization | **20.3 – 37.2×** |
| vs the best practical evictor | **1.07 – 4.81×** |

Uniform quantization loses by more than an order of magnitude at 3 bits — it is
not a serious baseline. **Every design decision and headline number must be stated
against eviction**, or it is not informative.

### 6.2 Eviction is tier 0, inside the allocator

At B=3 the water-filling interior **chooses to evict 40–54%** of tokens on its own
(40–49% among in-band heads). SIEVE is therefore not "quantize instead of evict";
it is a **joint evict-and-quantize allocator** in which dropping a token is simply
the cheapest tier. Tier 0 must be a first-class bit-width in the envelope, not a
separate mechanism bolted alongside.

### 6.3 Route on ladder width — per head, and per deployment context

Only one candidate signal works on every model:

| signal | correlation with realised gain | works everywhere? |
|---|---|---|
| **ladder width** | **−0.41 to −0.77**, negative in **24/24** | **yes** |
| n₉₅ | −0.02 to +0.71 | **no** |

n₉₅ *inverts* on llama33-70b: at 16k/32k/64k its in-band heads have **lower** n₉₅
than its out-of-band heads (ratio 0.44–0.61) and the rank correlation is ~0. On
every other model, in-band heads carry **7.5–22.7×** the n₉₅ of out-of-band heads.
Ladder width stays negative throughout, llama33-70b included (−0.41 to −0.72).

**Design rule:** compute ladder width per head in one cheap pass (no quantization
needed), route wide-ladder heads to eviction and narrow-ladder heads to the
mixed-precision interior. Because the band moves −3.7 to −15.9 points per context
doubling, the threshold must be a function of deployment context — a head that
belongs in the interior at 8k may belong on the eviction path at 128k.

### 6.4 Ship H2O. Skip SnapKV and StreamingLLM.

| evictor | share of heads won | host state |
|---|---|---|
| `accum` (H2O) | **74–93%** | 5 B per layer-head-token |
| `window` (SnapKV) | 6–22% | 17 B |
| `recency` (StreamingLLM) | 1–12% | 1 B |

H2O wins the large majority and has the simplest state — one running sum per
(layer, head). SnapKV costs 3.4× the host memory for 6–22% of the wins. **The min
over three corners is barely stronger than H2O alone**, so a system implementing
only H2O gives up almost nothing, and the verdict is robust to the choice.

### 6.5 Do not size the eviction budget as a fixed fraction

Two findings combine. K\*/budget = 100% says the fractional budget cannot be
shrunk uniformly. K\*/n₉₅ spanning 37× says the *right* budget is a per-head
property that no constant captures.

**Design rule:** derive the per-head keep-count from measured support at runtime —
K\* is computable in one cumulative pass over the ranked tokens (see `fix.md` §1)
— or accept the fractional budget and spend the savings elsewhere. Do not ship a
global κ.

### 6.6 What the system does not need

The oracle's advantage over H2O is **1.02–1.41×**, and it is concentrated on
diffuse heads that the router sends to the interior anyway. There is no return on
a more sophisticated importance score. **Scoring is solved; allocation is where
the value is.**

---

## 7. Status and what to run next

**Nothing from `script.sh` is outstanding.** All 24 configurations have landed,
including both 128k points that were previously in flight.

**Resolved, not open.** llama31-8b's n₉₅ jump (291 → 1,008 between 64k and 128k)
was flagged as possibly a prefill artifact. llama33-70b shows a far larger jump at
the same step (117 → 4,854) and independently reproduces the earlier campaign's
L^1.81 figure to 1%. Two models showing the same discontinuity at the same
context, one matching a prior independent measurement, makes it a property of
long-context attention. Worth a note in the paper; not worth a re-run.

**Ranked next steps:**

1. **Probe the 64k→128k step.** Every model that reaches 128k drops sharply there
   (−36.9 llama33-70b, −5.7 qwen3-30b, −5.4 llama31-8b) with τ and n₉₅ both
   jumping. Whether that is a genuine attention regime change or RoPE behaviour
   near the trained limit is the most interesting open question in the study, and
   a 96k point answers it with the existing pipeline.
2. **Error-bar qwen3-30b @128k.** At 18.6% it sits 3.6 points above the STOP line
   — the one cell where a modest measurement change flips a verdict. A second seed
   needs `n_prompts` or `rot_seed`, neither of which is reachable through
   `SIEVE_*`; closing that gap in the submission scripts is small and worthwhile.
3. **Retire the `abs` policy**, or re-derive κ per model from K\*/n₉₅ if it is to
   be reported at all. Given K\*=100% everywhere, the honest move is to report
   K\*/n₉₅ as the tail-dependence measurement it turned out to be.
4. **Drop `window` and `recency` from future campaigns.** They cost 18 of the 23
   bytes per layer-head-token of host state and win 6–22% and 1–12% of heads
   respectively. Keep `oracle` (free, and the bound every legacy column is defined
   against) and `accum`.
