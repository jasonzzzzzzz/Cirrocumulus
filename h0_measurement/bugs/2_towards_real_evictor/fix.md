# Fix: an honest eviction corner (P0 + E2 + E1)

Reconciled against the LIVE tree (probe fix, validity gate, `--validity-only`,
needle columns). The drafts in this directory predate all of that and were used
as reference only; nothing was copied wholesale.

**Scope: P0, E2, E1. E2b (the practical-INTERIOR cell) is NOT implemented** — the
interior still waterfills on oracle sensitivity. The plumbing it needs
(`practical_scores`) is in place, so it is an additive change later.

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

| File | Change |
|---|---|
| `sievelib/evict.py` | **new** — registry, alignment, `corner_tokens`, `CornerSpec` |
| `sievelib/alloc.py` | `evict_error_curve`, `_kstar_grid`, `exact_error(...,o)`, corner grid + K* in `quant_metrics`, `head_metrics` plumbing |
| `h0_measurement/run_h0.py` | `prev_a` → evictor packs; `CornerSpec` resolved pre-GPU; provenance columns |
| `h0_measurement/models.yaml` | `evictors`, `corner_policies`, `corner_kappa`, `corner_floor`, `kstar` + docs |
| `h0_measurement/report.py` | verdict keys off the practical corner and names it; corner grid, spend and K\* panels; legacy fallback |
| `tests/test_units.py` | 4 new tests, 40 checks |

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

## 4. What to expect

**Correction to the plan's acceptance criterion.** The plan states
`gain_best_practical ≥ gain_best` is "provably monotone" and that "any cell that
falls indicates a bug". **That is not true per head, and the criterion must not be
used as a bug detector.** `oracle` is an oracle only with respect to the
first-order proxy `w² = (a·‖v−o‖)²`, while the reported error is exact
recomputation — alloc.py keeps those strictly separate by design. Ranking by the
proxy is not the argmin of the exact error, so a differently-ranked corner can
land on a better kept set.

Measured on a real qwen3-1.7b run (ctx 2048, 1,344 head-rows): a practical corner
beats the oracle on **15.9% of rows**, by up to 4×. Per evictor: `accum` 12.2%,
`last_step` 12.0%, `window` 9.3%, `recency` 7.0%. The oracle corner even loses to
*uniform* on 0.1% of rows, by the same mechanism.

**The direction holds decisively in aggregate**, which is the claim to make:

| | oracle corner | practical corner |
|---|---|---|
| heads in band @3b | 2.5% | **32.6%** |
| median `err_practical / err_evict` | — | 1.19 |

I removed the two test assertions that encoded the false per-head claim (they had
passed only because one synthetic draw happened to align) and replaced them with
formula checks. **Compare distributions, not individual heads.**

Largest movement expected on sharp models (`qwen3-*`) and 128k rows, where oracle
eviction was near-lossless. Watch `llama31-8b@128k` (12.5%) and `qwen3-30b`
(4–10%) against the 15% STOP line.

**One finding already, from the E1 implementation.** On a sharp synthetic head
(τ=4.0, L=16384, n95=180) the `abs` corner spends **8.6× fewer bits** than `frac`
(0.35 vs 3.00 b/tok) and is *also more accurate* (2.64e-2 vs 2.82e-2), with
**K\* = 9.3% of the corner's budget**. Corner error is **not monotone in K**: every
kept token is quantized at `maxb`, so extending the keep-set down the tail adds
low-weight tokens carrying quantization noise and the renormalised softmax comes
out worse.

That is stronger than the plan's framing. E1 anticipated "the corner has slack";
the measurement says the fractional budget is *actively counterproductive* on
sharp heads. It also means **the frac/abs comparison has no assumed sign** — I
removed a test assertion of mine that claimed `abs` is never stronger, which had
passed only by luck on one random draw. Both cells are reported; neither is
assumed to dominate.

Two honest caveats for the writeup:

1. **`accum` accumulates from decode, not prefill** — the probe only captures
   decode queries (`q_len == 1`). Deployed H2O sees the prefill too, so ours is
   *weaker* than the real thing, which biases gains **up**. At `n_decode ≤ 8`
   treat it as a floor on the practical corner's strength.
2. **`quant_every` thins the comparison.** Practical columns need `shat`, and
   `do_quant = step % quant_every == 0`, so step 0 is always a quant step and step
   0 never has lagged history — with the defaults (`n_decode: 8`, `quant_every: 4`)
   the quant steps are 0 and 4, and half the quant rows carry only `recency`.
   Filter on `n_practical`, or raise `quant_every`. I did **not** change the
   parity to `(step+1) % quant_every`, which would put both quant steps on real
   history, because it silently re-bases every existing metric — a one-line change
   at `run_h0.py:372` if you want it, but that is a re-baseline decision, not mine.

### Cost

Per (head, budget): ~5 cumulative passes with the default 5 corners, versus 4
`exact_error` calls before — the curve trick absorbs the policy axis and the K*
ladder almost entirely. CPU state per (layer, head) is `ctx × (4·n_bufs + 1)`
bytes per evictor (`last_step` 1, `accum` 1, `window` its `window`, `recency` 0):

| config | bytes / (layer, head, position) | llama33-70b @ 128k |
|---|---|---|
| old (`prev_a`) | 4 | 2.7 GB |
| `last_step,accum,recency` | 11 | 7.4 GB |
| default (adds `window`) | 28 | 18.8 GB |

Freed at each prompt boundary. Drop `window` first if RAM is tight.

---

## 5. Verification

Run with the project venv (`.venv/bin/python`), not the bare login-node python.

`tests/test_units.py` gains 4 tests / 40 checks; the full suite passes (run in two
halves — the login node's CPU rlimit kills a single full run):

- **`test_p0_alignment`** — [REGRESSION] scores on the step after the first (the
  exact bug); sliding-window front-roll; unmodelled length jump resets rather than
  mis-aligns; new token outranks all history.
- **`test_e2_registry`** — accum sums / last_step forgets / window pools / recency
  ranks sinks then newest; paper-name aliases; oracle is a corner not a stateful
  evictor; **oracle stays configurable and on by default**, and a run may drop it;
  bad names, options, policies and config strings all rejected loudly.
- **`test_e1_budget_policy`** — frac/abs/floor/fallback arithmetic; abs never
  exceeds frac; `evict_error_curve == exact_error` at every K (1.2e-14);
  [REGRESSION] abs is cheaper *and* more accurate on a sharp head; K* detects the
  slack.
- **`test_corner_columns`** — every (evictor, policy) cell present; legacy oracle
  columns bit-for-bit; verdict columns; verdict takes the strongest practical
  corner; both `gain_best*` formulas; mis-aligned and `oracle`-labelled scores
  rejected; oracle-only run still produces the legacy corner.

Beyond unit tests:

- Drove `run_h0.py`'s per-head loop against synthetic tensors in **both** cache
  modes (growing and fixed-length/sliding) with the shipped default config: all
  four practical evictors score from step 1, `recency` from step 0, monotonicity
  holds on every step.
- Ran **report.py end-to-end** on a synthetic parquet with the new columns and on
  a legacy one without them. Both produce a PDF; the new schema keys the verdict
  off the practical corner and the legacy schema falls back to the oracle, each
  labelled. Corner grid, spend, K\* and the corner-naming VERDICT line all render.

**A real model, end to end.** `qwen3-1.7b` at ctx 2048 on CPU, real PG-19
haystack, ~15 s — the full probe path (L1 drop-in PASS, L2 capture 2.9e-06 PASS,
needle 66224 RETRIEVED, GQA 16H/kv=8):

```
rows 1344 | corners: oracle,last_step,accum,window,recency
practical evictors scored per row: {4: 896, 1: 448}    # step 0 = recency only
gain_best_practical3 null frac: 0.000                  # <- the bug, gone
band   oracle 2.5%  ->  practical 32.6%
winning corner: {'last_step': 527, 'recency': 495, 'accum': 233, 'window': 89}
corner spend  frac 3.00 b/tok (718 tok)   abs 1.07 b/tok (256 tok)
oracle_evict_advantage3  accum 1.09x  last_step 1.09x  window 1.32x  recency 2.88x
```

No evictor dominates — the winner varies per head, which is exactly why the
verdict takes the min over the configured set and why the per-evictor columns are
kept. `K*` is 100% of budget here, as expected at ctx 2048: E1's slack only exists
at long context.

**Still not verified: a GPU run, a sliding-window model, and long context.** The
alignment path's roll-left branch is exercised only by synthetic stand-ins, and no
128k run was made. Start the campaign with a debug-tier run and confirm
`gain_best_practical3` is non-null before committing array hours.
