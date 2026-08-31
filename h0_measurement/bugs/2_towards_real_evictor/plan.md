# Plan: an honest eviction corner (E1 + E2, one rerun)

Two defects in the eviction corner, one campaign to fix both. They are the same
experiment because they are the same baseline: E2 fixes WHO the corner is
(an oracle no deployable system has), E1 fixes HOW MUCH it is given (a budget
that grows linearly in L while head support grows as L^0.63–0.92). Both biases
currently point the same way — they flatter the corner, hardest at long context
and on sharp heads — so every 128k STOP verdict rests on a baseline that is
doubly too strong. Fixing them can only raise band fractions (both changes are
provably monotone), which is why this runs before any systems work: it decides
whether the 128k story is real.

## Contexts

Live tree (authoritative — it has moved past the drafts below: needle/validity
columns, task_decode_extra, the probe mask fix):

- ../../run_h0.py                — decode loop, prev_a maintenance, the off-by-one
- ../../../sievelib/alloc.py     — quant_metrics: corner construction, budgets, waterfill
- ../../report.py                — gain_best_practical/oracle_evict_advantage columns (printed, never populated)
- ../../models.yaml              — per-model config; new keys land here
- ../../bugs/3_context_sweep_and_reports/findings.md — the n95 ~ L^p measurements E1 rests on

Drafts staged in THIS directory (`evict.py`, `run_h0.py`, `alloc.py`, `fix.md`,
`test_units.py`): reference implementations from the earlier analysis. Reconcile
against the live tree rather than copying — they predate the probe fix and the
validity gate, and any rerun must include both.

## P0 — unbreak `practical_score` (prerequisite for everything)

`prev_a` from step t has one fewer entry than step t+1's logits, so the guard
`pa.numel() >= sh.numel()` in run_h0.py almost always fails and silently falls
back to None: the practical corner has NEVER run, in any campaign. Fix: pad the
newest position (+inf — every real evictor keeps the newest token by recency)
instead of requiring equal length. Edge cases, handled deliberately:

- sliding-window models: `fin` holds length constant while the oldest position
  drops — pad at the end, truncate at the front;
- `quant_every` is already fine (prev_a updates every step, so quant steps see a
  1-step-lagged score).

~15 lines + a unit test asserting the score populates from step 1 onward.

## E2 — pluggable evictors, oracle kept as one of them

The oracle stays: gain-vs-oracle is a valid lower bound, and
`oracle_evict_advantage` (how much H2O/SnapKV-style scoring loses to the oracle,
per head, at scale) is a publishable number on its own — expect it largest on
sharp heads, where a lagged score that misses one heavy-hitter is catastrophic.
What changes is which corner the VERDICT keys off: GO/STOP must be measured
against the best corner a deployable system could field, with the oracle
reported alongside as the bound (why.md Q0).

Registry in a new `sievelib/evict.py` — one score-maintenance rule per evictor,
selected per run:

| name         | score                                             | anchors        |
|--------------|---------------------------------------------------|----------------|
| `oracle`     | current-step a·‖v−o‖ (status quo)                 | upper bound    |
| `last_step`  | prev_a (P0 makes this real)                       | TOVA           |
| `accum`      | running sum of attention received                 | H2O            |
| `window`     | max-pool of prev_a over last W steps              | SnapKV         |
| `recency`    | position only: sinks + newest                     | StreamingLLM (floor) |

Config: `evictors: [oracle, last_step, accum]` in models.yaml defaults,
overridable per run (`--override evictors='["oracle","accum"]'`). Adding an
evictor = one entry in the registry (a `(state, a_t) -> state` update and a
`state -> score` read), nothing else. Columns come out per evictor:
`gain_e{B}_{name}`, and the verdict uses
`gain_best{B} = min(e_uniform, min over practical evictors) / e_wf`.

## E1 — the budget axis: fractional vs absolute-support

findings.md measured n95 ∝ L^0.63 (llama31-8b) to L^0.92, against a corner
budget of B·L/maxb tokens — linear in L. At B=3 the corner holds 15.8× a head's
support at 4k and 70.8× at 64k. The corner may be winning at 128k on slack, not
on merit. Two deliverables, both cheap once the corner is pluggable:

1. **Budget policy as a config axis**, parallel to the evictor axis:
   - `fractional` — keep B·L/maxb tokens (status quo; the literature's definition);
   - `absolute` — keep min(B·L/maxb, max(κ·n95_head, floor)) tokens at maxb.
     κ=4, floor=256 to start; n95 comes from the same pass. This corner spends
     FEWER bits than the budget allows — record `corner_bits_used{B}` so the
     comparison is explicit, not smuggled.
2. **The diagnostic that needs no policy choice**: e_evict as a function of
   kept-token count K, per head — report K*(head) = smallest K within 10% of the
   full-budget corner. If K* ≪ B·L/maxb at 128k (predicted), the slack is
   measured directly and the mis-specification claim in the proposal (§2)
   has its number.

Note for the paper, not this fix: the real design implication is cross-head —
bits the absolute corner does NOT spend on sharp heads are exactly what the
router should reinvest in interior heads. That is a SIEVE feature, not a
baseline fix; keep it out of this rerun.

## E2b — the fully honest cell (promoted from "defer": same rerun, ~20 lines)

After E2 the comparison is still asymmetric: the interior waterfills on oracle
sensitivity while only the corner is practical. Add the practical-interior
column per why.md Q2: build w2p from the lagged score (ō = pa @ V is computable
from cache), waterfill on w2p, evaluate on true logits — decision lagged,
evaluation honest, same asymmetry a deployed system faces. Columns `gain_pp{B}`,
`in_band_pp{B}`. Rank the corner both by raw pa (literature-faithful) and by w2p
(isolates whether the edge is mixed-precision shape vs score information). If
the interior's edge survives with both sides practical, that is the strongest GO
this framework can produce.

## Rerun + reporting

- One campaign over the 20-point grid (matched 8k/32k sixes + both sweeps),
  post probe-fix, validity gate on. Extra cost per head/budget is one
  `exact_error` per (evictor, policy) cell — small next to quantize_keys.
- report.py: populate the existing practical columns; per-evictor
  `oracle_evict_advantage`; a corner-comparison panel (oracle vs each practical,
  fractional vs absolute) on the phase axis. Keep dead-2 as the phase variable —
  it is defined by sig2 vs the DERIVED c0=1 and does not move with the corner;
  what moves is the band, so the (dead2 → band) curve should shift up while the
  axis itself stays put. That is itself a check.
- Verdict line names its corner: `-> GO  (vs accum/absolute; oracle bound 1.31x)`.

## Expectations / acceptance

- Monotone: every band fraction rises or holds vs the v7 table (min over MORE
  and weaker corners can only shrink e_corner's advantage... i.e.
  gain_best_practical ≥ gain_best). Any cell that falls indicates a bug.
- Largest movement: sharp models (qwen3-*) and 128k rows. Watch specifically
  whether llama31-8b@128k (12.5%, STOP) and qwen3-30b (4–10%, STOP) cross 15%.
- The scientific outcome is symmetric and both branches are publishable:
  128k STOPs melt under an honest corner → the systems claim is intact and the
  field's baseline definition takes the blame; they survive → the band genuinely
  ends near 32–64k and the paper narrows honestly (proposal §9.1).

## Effort

P0 half a day (with test). E2 registry + plumbing 1–1.5 days. E1 policies +
K* diagnostic 1 day. E2b ~20 lines inside E2's edit. Rerun: same GPU budget as
the v7 campaign (the grid is the cost; new columns are marginal). Report: half
a day. Total ≈ 4 days wall, one campaign.
