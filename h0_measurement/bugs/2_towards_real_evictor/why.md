

Its bias is systematically against your hypothesis: every in-band fraction is deflated because the eviction corner is an oracle no deployable system has, and the deflation is strongest exactly for the sharp models the current table says STOP. The headline your own docs promise (alloc.py) simply does not exist in the data. It ranks below P1 only because a fixed practical corner measured on filler is still uninterpretable.

How results would change if fixed: This one is provably monotone: gain_best_practical = min(e_uniform, e_practical)/e_wf ≥ gain_best, since a lagged-attention evictor can only be worse than the oracle. So band fractions can only rise. Expected effects:

Largest movement for qwen3-30B (99% of its out-of-band heads were out because oracle eviction was near-lossless) — likely STOP → NARROW or better.
You also get oracle_evict_advantage for free, which is a publishable number in its own right: how much H2O/SnapKV-style scoring loses to the oracle, per head, at scale. Report.py already prints all these columns  — they've just never been populated.

## Q&A

**Q0: Why do I need a practical evictor instead of an oracle one? An oracle gives me the minimum gain of my design.**

That's correct as far as it goes — the oracle corner (ranks tokens by true sensitivity `w² = (a·‖v−o‖)²`, using the *current* query's attention `a`) is an upper bound on any real evictor, so gain measured against it is a lower bound on real gain. The problem is what the table does with that lower bound: a lower bound is only decisive in one direction. If gain-vs-oracle is *above* the band, that's a genuine GO, no caveats. If it's *below* the band, you've proven nothing — you've only shown the interior loses to a competitor that cannot be built, since the oracle needs the current step's attention weights to decide what to evict, which is exactly the computation eviction exists to avoid. Yet the current table converts those uninformative below-band cells into STOP verdicts. That's the systematic bias: it kills the design on a bound that can't support that conclusion in the losing direction. The oracle should stay as one selectable option (its gap to the practical evictor, `oracle_evict_advantage`, is a useful number on its own), but the GO/STOP headline needs to be measured against what a deployable system could actually field.

**Q0b: Which practical evictors should we compare against — is it H2O and SnapKV?**

Yes, those are the right anchors; they're what reviewers will name. Concretely, the family worth implementing behind the pluggable-evictor interface:
1. **Last-step attention** (TOVA-style) — what `prev_a` in run_h0.py already computes; cheapest, currently wired in but broken by the off-by-one.
2. **Accumulated attention** (H2O proper) — running sum of attention received over all past steps, not just the last one.
3. **Window-pooled attention** (SnapKV-style) — attention summed/max-pooled over the last few steps' queries.
4. **Recency + sinks** (StreamingLLM) — no attention statistics at all; the floor of the family.

`quant_metrics`'s `practical_score` parameter is already an opaque tensor, so adding an evictor is just a new score-maintenance rule in run_h0.py's decode loop plus a config flag.

**Q1: What does "the deflation is strongest exactly for the sharp models the current table says STOP" mean?**

"Sharp" = heads whose attention concentrates on a handful of tokens (spiky softmax). For a sharp head the oracle evictor is nearly lossless — it knows exactly which few tokens carry the mass, keeps them, and evicting the rest costs ~nothing — so `e_evict ≈ 0`, `gain_best ≈ 0`, head counted out-of-band, model → STOP (this is the qwen3-30B case: 99% of its out-of-band heads were out for this reason alone). But sharp heads are simultaneously where a *practical* evictor is most fragile: with all mass on a few tokens, a lagged score that misses one heavy-hitter takes a catastrophic error, whereas a flat head barely notices a wrong eviction. So the oracle-vs-practical gap (the "deflation") is largest precisely on the sharp heads — the STOP verdicts are concentrated exactly where the measurement bias is worst.

**Q2: If, even after the practical-corner fix, the comparison is still asymmetric because the interior waterfill allocates using oracle sensitivity while only the corner is practical — how do I build the fully honest cell (practical-interior vs practical-corner)?**

1. **Prerequisite:** fix the padding bug in run_h0.py so `practical_score` actually populates (currently the `pa.numel() >= sh.numel()` guard almost always fails and silently falls back to None).
2. **In `quant_metrics`, build the practical sensitivity once**, outside the budget loop:
   ```python
   if practical_score is not None:
       ap = practical_score.double().clamp_min(0)
       ap = ap / ap.sum().clamp_min(1e-300)
       op = ap @ Vd                      # ō: lagged output, computable from cache
       w2p = (ap * (Vd - op).norm(dim=-1)) ** 2
   ```
   `ō = pa @ V` needs only last step's attention and cached V — nothing from the current query.
3. **Inside the budget loop**, waterfill on `w2p` and evaluate with the true logits (decision is lagged, evaluation is honest — same asymmetry a deployed system actually faces):
   ```python
   bwp = waterfill(w2p, sig2, float(B), maxb)
   e_wf_pr = exact_error(s, shat, V, bwp)
   out[f"gain_pp{B}"] = min(e_un, e_pr) / max(e_wf_pr, 1e-12)
   out[f"in_band_pp{B}"] = float(out[f"gain_pp{B}"] >= BAND_MIN)
   ```
4. **Corner ranking choice:** report the practical corner ranked both by raw `pa` (literature-faithful H2O/SnapKV) and by `w2p` (apples-to-apples with the interior) — the latter isolates whether the edge comes from mixed-precision *shape* rather than from score information the corner lacks.
5. Plumb the new columns through report.py.

If the interior's edge survives with both sides practical, that's the strongest GO this framework can produce — no reviewer can attribute it to information asymmetry.


