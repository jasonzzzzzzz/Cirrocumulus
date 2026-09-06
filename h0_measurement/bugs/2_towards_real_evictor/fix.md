# Fix: an honest eviction corner

Four defects, one code change set. **Results are in `report.md`; this file is the
engineering record — what broke, what changed, and how to reproduce it.**

| # | defect | effect if unfixed |
|---|---|---|
| **P0** | the lagged score was never computed (an off-by-one guard rejected every step) | the practical corner had **never run**, in any campaign |
| **E2** | the eviction corner ranked by an *oracle* score no deployable system has | every band fraction deflated by 6–29 points |
| **E1** | the corner's budget was a fixed fraction of L, untested against head support | the "corner wins on slack" hypothesis was unmeasurable |
| **P1** | `min` over the practical corners collapsed onto the weakest one on step 0 | band inflated by ~29 points; the ctx curve came out **non-monotone** |

P0/E2/E1 were planned (`plan.md`). **P1 was discovered by the fix itself** — it
only becomes reachable once more than one practical evictor exists.

**Out of scope: E2b, the practical-INTERIOR cell.** The interior still waterfills
on oracle sensitivity, so the comparison remains asymmetric in the interior's
favour. The plumbing it needs (`practical_scores`) is in place; see `plan.md`.

---

## 1. Methodology of the fix

### P0 — why the practical corner had never run

`run_h0.py` gated its lagged score on

```python
if pa is not None and pa.numel() >= sh.numel():   # prev_a is ALWAYS one shorter
```

`prev_a` is step *t*'s attention over *t*'s live positions; `sh` is step *t+1*'s
logits, which has one more entry for the token just generated. The guard is false
on essentially every step, so `practical_score` silently became `None`. **No
campaign has ever populated a single practical column.**

The fix is not a `+1` — it is admitting that the score is indexed by *KV-cache
position*, and cache geometry changes under us between steps. `sievelib/evict.py`
handles alignment once, for every evictor:

| cache length change | meaning | action |
|---|---|---|
| `L+1` | full attention, one new token | append a slot |
| `L` unchanged | sliding window; `DynamicCache` trims the front | roll left one |
| anything else | static/wrapping cache, >1 token/step, missed reset | **reset, report no history** |

That last row is deliberate. Position identity is not recoverable from a length
change we do not model, and silently mis-attributing history to the wrong tokens
is the same class of bug being removed. Failing loud beats scoring garbage.

A position never observed (this step's new token) has no history at all. Every
real evictor keeps the newest token by recency, so `score()` ranks such positions
strictly first — applied at scoring time only, never contaminating the
accumulator.

### E2 — WHO the corner is

The oracle ranks by the current step's `a_i·‖v_i−o‖`, which needs the very
attention weights eviction exists to avoid computing. It is an upper bound on
every real evictor, so a below-band cell measured against it proves nothing.

**The oracle is kept, and stays configurable.** It is listed in `evictors` by
default, keeps its legacy column names, and is what `gain_best<B>` /
`err_evict<B>` still mean — so the monotonicity check is computable row-by-row in
one frame. What changed is the corner the **verdict** keys off: the strongest
corner a deployable system could field. The oracle is reported beside it as the
bound, and `oracle_evict_advantage` is now a real per-head number.

The verdict aggregate takes the **strongest** (lowest-error) practical corner, not
the mean — it makes the baseline as hard to beat as any deployable evictor could
make it, so an in-band verdict cannot be dismissed as a weak competitor. It is
mildly optimistic about evictor *selection*, which is why the per-evictor columns
are kept: rerun any analysis against one fixed evictor from those.

### E1 — HOW MUCH the corner may keep

The corner kept `B·L/maxb` tokens — linear in `L`, while head support grows as
`L^0.63–0.92`. `corner_tokens` makes the budget a policy axis: `frac` (status quo)
and `abs` = `min(frac, max(κ·n95, floor))`, capped at a multiple of the head's own
measured support, with `corner_bits_used` recording what it declined to spend.

Plus the diagnostic that needs no policy choice: **K\***, the smallest kept-token
count within `kstar_tol` of the full-budget corner.

### P1 — a `min` over corners with different availability

The verdict corner is `min` over the configured practical evictors, so the
baseline is as strong as any deployable system could make it. But the evictors do
not all become available at the same time: `recency` scores from position alone
and needs no history, while `accum` and `window` need a prior decode step. **On
step 0 the `min` therefore collapses onto the one corner that is always available,
which is also the weakest.** With `quant_every: 4, n_decode: 8` the quant steps are
0 and 4, so half of every campaign's rows compared the interior against
StreamingLLM alone.

The guard is one condition — emit the aggregate only when every configured evictor
scored:

```python
_complete = bool(scores) and (set(corner.practical).issubset(scores)
                              or set(scores) == {"practical"})
if _complete:
    ...                       # err_practical, gain_best_practical, best_evictor, ...
```

Otherwise the columns are absent (NaN), so any `median`/`dropna` drops the row by
itself. Per-evictor cells still record whatever *did* score.

`report.py.drop_partial_corners()` applies the same rule to parquets written
before the guard, keyed on `n_practical` against the `evictors` provenance column.

**The generalisable lesson:** a baseline defined as a minimum over several methods
silently becomes the weakest of them wherever the others are undefined. Any
`min`/`max` over a set with differing availability needs a completeness guard.

### The trick that made E1 affordable

An eviction corner keeps the top-K by some ranking, so walking K walks a **nested**
family of kept sets: the softmax numerator and denominator are running sums along
the ranking. `evict_error_curve` computes the *entire* K-curve in one pass for
about the cost of a single `exact_error`, chunked so peak memory is `O(chunk·d)`.
Verified exact against `exact_error` to 1.2e-14 — it is a factoring, not an
approximation. Without it the (evictor × policy) grid plus a 12-point K* ladder
would have been ~17 `exact_error` calls per head per budget instead of ~5.
`exact_error` also now takes a precomputed `o`, which every call was recomputing.

---

## 2. Files changed

| file | change | defect |
|---|---|---|
| `sievelib/evict.py` | **new** — evictor registry, cache-position alignment, `corner_tokens`, `CornerSpec`, `state_bytes_per_slot` | P0, E2, E1 |
| `sievelib/alloc.py` | `evict_error_curve`, `_kstar_grid`, `exact_error(..., o)`, the (evictor × policy) grid and K\* in `quant_metrics`, `head_metrics` plumbing | E2, E1 |
| `sievelib/alloc.py` | completeness guard on the practical aggregate | **P1** |
| `h0_measurement/run_h0.py` | `prev_a` → evictor packs; `CornerSpec` resolved before the GPU is held; provenance columns; sidecar JSON | P0, E2 |
| `h0_measurement/models.yaml` | `evictors`, `corner_policies`, `corner_kappa`, `corner_floor`, `kstar`, `corner_in_filename` + docs | E2, E1 |
| `h0_measurement/report.py` | verdict keys off the practical corner and names it; corner grid / spend / K\* panels; `drop_partial_corners`; legacy fallback | E2, E1, **P1** |
| `h0_measurement/make_fig_phase.py` | plots both corners; applies the P1 filter | E2, P1 |
| `tests/test_units.py` | 7 tests, 60 checks | all |

---

## 3. Interface

```
   models.yaml / --override                                        fails HERE,
   ┌──────────────────────────────────────────────────────┐        before the
   │ evictors:        [oracle, last_step, accum,          │        tokenizer and
   │                   window, recency]      <- WHO       │        the GPU alloc
   │ corner_policies: [frac, abs]            <- HOW MUCH  │
   │ corner_kappa: 4.0   corner_floor: 256   kstar: true  │
   └───────────────────────────┬──────────────────────────┘
                    evict.CornerSpec.from_cfg(c)
                               │
   run_h0.py decode loop       ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ per prompt:  evs.clear()        history never crosses prompts       │
   │ per step, per (layer, head):    [skipped in --validity-only]        │
   │                                                                     │
   │   fin  = isfinite(logits)     live cache positions, bool [Lcache]   │
   │   pack = evs[(li,h)]  ──────► {label: Evictor}   STATEFUL, lagged   │
   │                                                                     │
   │   ┌ scores = {lab: ev.score(fin)}  ── PAST attention only ──┐       │
   │   │       None until history exists (recency: never None)    │      │
   │   ▼                                                          │      │
   │   head_metrics(sh, shat, Vh, practical_scores=scores,        │      │
   │                corner=corner)                                │      │
   │   │                                                          │      │
   │   └ ev.observe(softmax(sh), fin)  ── AFTER scoring ──────────┘      │
   │           this ordering IS the lag                                  │
   └───────────────────────────┬────────────────────────────────────────┘
                               ▼
   alloc.quant_metrics — the corner grid
   ┌─────────────────────────────────────────────────────────────────────┐
   │ interior : waterfill(w²)                        → err_wf            │
   │ corner   : uniform B bits                       → err_uniform       │
   │ corner   : for each evictor x each policy:                          │
   │              oracle    ← w² (added HERE; needs the current step)    │
   │              last_step │ accum │ window │ recency  ← lagged         │
   │              x  frac (B·L/maxb)  │  abs (κ·n95 cap)                 │
   │            one evict_error_curve pass per ranking covers every K    │
   │            + the K* ladder                                          │
   └───────────────────────────┬─────────────────────────────────────────┘
                               ▼
   columns ────────────────────────────────► report.py (verdict names its corner)
```

### Downward — adding an evictor

```python
@register("mine")
class Mine(Evictor):
    n_bufs = 1
    def __init__(self, decay: float = 0.9):     # becomes  mine:decay=0.95
        self.decay = float(decay); super().__init__()
    def _accum(self, a, fin):                   # this step's attention
        self._bufs[0] *= self.decay
        self._bufs[0][fin] += a
    def _raw(self):                             # score over all cache positions
        return self._bufs[0]
```

That is the whole contract. `run_h0` finds it through the registry, `alloc.py`
emits its columns, `report.py` picks them up from the column names. Alignment, the
fresh-token rule, reset, and the config surface are the base class's job.

Shipped: `oracle` (bound), `last_step` (TOVA), `accum` (H2O), `window` (SnapKV,
`window`/`pool`), `recency` (StreamingLLM, `sinks`). Paper names work as aliases.
Options as `name:k=v,k=v`, optional `@alias` when one evictor appears twice.

### Upward — config

```bash
--override evictors=oracle,accum                  # comma list
--override evictors=window:window=8,pool=13       # options after ':'
--override 'evictors=oracle;window:window=2@w2'   # ';' separates when options present
--override corner_policies=frac,abs
--override kstar=false                            # drop the K* ladder
```

### Upward — columns

Per budget `B`, per evictor `<n>`, per policy `<p>` ∈ {frac, abs}:

| column | meaning |
|---|---|
| `err_e<B>_<n>_<p>`, `gain_e<B>_<n>_<p>` | that corner cell |
| `oracle_evict_advantage<B>_<n>` | what evictor `<n>` loses to the oracle — the per-head-at-scale number |
| `corner_tokens<B>_<p>`, `corner_bits_used<B>_<p>` | what the policy kept / spent |
| `kstar<B>`, `kstar_frac<B>`, `kstar_over_n95<B>` | E1's slack diagnostic |

Verdict (practical only — never sees the oracle):
`err_practical<B>`, `gain_practical<B>`, **`gain_best_practical<B>`**,
`in_band_practical<B>`, `best_evictor<B>`, `oracle_evict_advantage<B>`,
`n_practical`, plus provenance `evictors` / `corner_policies`.

Legacy, unchanged in meaning (oracle @ frac): `err_evict<B>`, `gain_e<B>`,
`gain_best<B>`, `in_band<B>`. report.py falls back to these when a parquet has no
practical columns, and labels which corner it used.

---

## 4. Results, caveats, and cost

**See `report.md`.** It carries the completed 24-configuration campaign: the
before/after table, the context sweep, the mechanism findings, and the system
design rules that follow from them.

Two things belong here rather than there, because they are properties of the
*measurement* rather than of the models:

**The band fraction is a fragile statistic.** The oracle is only 1.02–1.41×
stronger in error terms, yet demoting the corner moves the band 6–29 points. The
per-head gain distribution is dense at the 2× threshold, so small changes in the
denominator reclassify many heads. Quote error ratios when you want a robust
number; quote the band when you want the decision.

**`gain_best_practical ≥ gain_best` is NOT a per-head theorem.** `plan.md`
asserted it was "provably monotone" and that any falling cell indicates a bug.
That is false and must not be used as a bug detector. `oracle` is an oracle only
with respect to the *first-order proxy* `w² = (a·‖v−o‖)²`, while the reported error
is exact recomputation — `alloc.py` keeps those strictly separate by design.
Ranking by the proxy is not the argmin of the exact error, so a differently-ranked
corner can land on a better kept set. Measured on a real run: a practical corner
beats the oracle on **15.9% of head-rows**. The direction holds decisively in
aggregate, which is the claim to make. **Compare distributions, not individual
heads.**

Two caveats that affect how a campaign should be configured:

1. **`accum` accumulates from decode, not prefill.** The probe only captures
   decode queries (`q_len == 1`), so our H2O is weaker than a deployed one, which
   biases gains **up**. At `n_decode ≤ 8` treat it as a floor on the practical
   corner's strength.
2. **`quant_every` interacts with P1.** The guard makes step-0 rows *safe* (they
   are dropped) but they are still wasted — half the quant rows of a
   `quant_every: 4, n_decode: 8` campaign produce no verdict. Either accept the
   waste, or change the parity so step 0 is not a quant step. We did **not** change
   `do_quant = step % quant_every == 0` (`run_h0.py:372`): it silently re-bases
   every existing metric, which is a decision for whoever owns the comparison to
   prior campaigns.

### Cost

**Compute — measured, not estimated.** `quant_metrics` at L=8192 over four
budgets, median of five warmed runs:

| corners | time | vs 1 |
|---|---|---|
| `oracle` | 46.8 ms | ×1.00 |
| `oracle,accum` | 52.2 ms | ×1.11 |
| `oracle,accum,window` | 55.2 ms | ×1.18 |
| `oracle,accum,window,recency` | 55.4 ms | ×1.18 |

One `evict_error_curve` pass is **0.6 ms** against a ~47 ms base dominated by
waterfill, the two baseline `exact_error` calls, and `noise_model`. **All three
practical evictors cost ~18% of a task**, and the frac/abs policy axis rides the
same pass and is free — `alloc.py` loops over evictors, not policies. That is why
splitting a campaign on the corner axis is a mistake: it re-runs the prefill, the
decode, and the whole 7-width `quantize_keys` sweep to save ~6% per job.

**Host RAM.** The lagged state is per (layer, head) and linear in ctx:
`n_layers × n_heads × ctx × slot`, where `slot = Σ (4·n_bufs + 1)` per stateful
evictor — `accum` and `last_step` 5 B, `window` 17 B, `recency` 1 B, `oracle` 0 B.

| corner set | slot | llama31-8b @128k | llama33-70b @128k |
|---|---|---|---|
| old (`prev_a`) | 4 B | 0.5 GB | 2.7 GB |
| `oracle,accum,recency` | 6 B | 0.8 GB | 4.0 GB |
| campaign default (`+window`) | 23 B | 3.1 GB | 15.4 GB |
| models.yaml default (`+last_step`) | 28 B | 3.8 GB | 18.8 GB |

Freed at each prompt boundary. Drop `window` first if the host runs short — it is
17 of those bytes. `evict.state_bytes_per_slot()` is the single source of truth,
used by `run_h0.py`'s banner and the SLURM preflights.

---

## 5. Reproducing this

**Use the project venv** (`.venv/bin/python`) — the bare login-node interpreter has
no torch/pandas, and the failure looks like a missing module rather than a missing
venv.

### Unit tests

```bash
.venv/bin/python tests/test_units.py
```

The login node enforces a CPU-time rlimit that kills a single full run (exit 152,
no summary). Split it — the corner tests alone are:

```bash
.venv/bin/python -c "
import sys; sys.argv=['x']
import tests.test_units as T
for t in (T.test_p0_alignment, T.test_e2_registry, T.test_e1_budget_policy,
          T.test_corner_columns, T.test_corner_provenance,
          T.test_report_survives_missing_corner_columns,
          T.test_partial_corner_is_withheld): t()
print('FAILS:', T.fails)"
```

**7 tests, 60 checks, 0 failures:**

| test | checks | pins |
|---|---|---|
| `test_p0_alignment` | 7 | scores on the step after the first (the P0 bug); sliding-window front-roll; unmodelled length jump resets rather than mis-aligns; a new token outranks all history |
| `test_e2_registry` | 14 | each evictor's scoring rule; paper-name aliases; oracle is a corner not a stateful evictor; oracle stays configurable and on by default; bad names/options/policies rejected |
| `test_e1_budget_policy` | 9 | frac/abs/floor arithmetic; `evict_error_curve == exact_error` at every K (1.2e-14); corner error is **not** monotone in K; K\* detects slack |
| `test_corner_columns` | 11 | every (evictor, policy) cell present; legacy oracle columns bit-for-bit; verdict takes the strongest practical corner; mis-aligned scores rejected |
| `test_corner_provenance` | 8 | the corner tag is stable, filesystem-safe, and records only parameters that applied; `config_record` is lossless and JSON-serialisable |
| `test_report_survives_missing_corner_columns` | 4 | report.py does not assume `gain_u` implies `gain_e` (the `KeyError: 'gain_e2'` crash) |
| `test_partial_corner_is_withheld` | 7 | **P1** — a partial corner withholds the verdict aggregate; report.py repairs older parquets |

### End-to-end, on a real model, in about 15 seconds

`qwen3-1.7b` is the debug tier and is corpus-exempt, so this runs on CPU with no
GPU allocation:

```bash
export HF_HOME=$PWD/.hf_cache H0_CORPUS=$PWD/.h0_corpus/pg19
.venv/bin/python h0_measurement/run_h0.py --model qwen3-1.7b --skip-external-check \
    --out-dir /tmp/smoke --override ctx=2048 n_prompts=1 n_decode=4 quant_every=1 \
    'families=["niah"]' 'bit_list=[3,8]' 'budgets=[3]' dtype=float32 chunk=1024 \
    device_map=cpu
```

Then confirm the practical corner actually populated — this is the check that
would have caught P0 immediately:

```bash
.venv/bin/python - <<'PY'
import glob, pandas as pd
q = pd.concat([pd.read_parquet(f) for f in glob.glob("/tmp/smoke/h0_*.parquet")])
q = q[q.quantized]
print("corners:", q.evictors.iloc[0])
print("practical scored per row:", q.n_practical.value_counts().to_dict())
print("gain_best_practical3 null frac: %.3f" % q.gain_best_practical3.isna().mean())
print("band  oracle %.1f%%  ->  practical %.1f%%"
      % (100*(q.gain_best3 >= 2).mean(), 100*(q.gain_best_practical3 >= 2).mean()))
PY
```

`null frac 1.000` means no practical corner ran — that is P0. A null frac equal to
the step-0 share means P1's guard is doing its job.

### Regenerating the figure

```bash
.venv/bin/python h0_measurement/make_fig_phase.py \
    "h0_measurement/results/job20014005/*.parquet" \
    "h0_measurement/results/job20013992/*.parquet" ... \
    -o docs/fig5_phase.png
```

Pass the ctx-sweep run **first**: `collect()` de-duplicates on (model, ctx) keeping
the first match, so leading with the sweep makes the swept model's points come
from one run.

---

## 6. What is not covered

- **E2b — the practical interior.** The interior still allocates on oracle
  sensitivity `w² = (a·‖v−o‖)²`, so only the corner has been demoted. The fully
  symmetric cell would waterfill on the lagged `(pa·‖v−ō‖)²`, with `ō = pa @ V`
  computable from cache. `practical_scores` is the plumbing it needs.
- **GQA head coupling.** Heads sharing a KV head cannot be given independent
  rates; the corner grid treats every query head independently.
- **End-task accuracy.** Everything here is attention-output error under exact
  recomputation. Nothing in this bug touches RULER/AIME.
