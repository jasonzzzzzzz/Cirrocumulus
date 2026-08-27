## TL;DR

The measurement machinery (probe, exact recomputation, waterfill) looks sound, but the experiment as run cannot support the conclusion "larger models benefit less," for three specific reasons I verified in the data: **(1)** every run used the synthetic 8-sentence filler haystack, and the validity check your own code documents for that case fired — the prompt families are statistically indistinguishable, so no retrieval behavior was ever induced; **(2)** the practical-evictor baseline silently never ran due to an off-by-one bug, so every eviction corner in the table is the oracle; **(3)** the in-band verdict is mechanically a window in φ = n95/L centered near the kept fraction B/8, which makes context length a direct input to the verdict — the 32k and 128k rows are not comparable. Meanwhile, the τ/ln2 vs ladder match that gave you confidence is close to circular and validates bookkeeping, not the theory.

## 1. The workload is degenerate, and the built-in kill-switch fired

Every run logged `(synthetic haystack)` — `H0_CORPUS` was never set, so all contexts are the 8 `FILLER` sentences in [prompts.py:13-22](ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/sievelib/prompts.py#L13-L22) repeated ~2,700× at 32k and ~10,000× at 128k. The module docstring states the acceptance test: *"If niah does not show a wider ladder than cont, the haystack is not inducing retrieval behaviour — point H0_CORPUS at real text and rerun."* Measured median ladder width by family:

| model | cont | niah | qa |
|---|---|---|---|
| mistral-7b | 3.08 | 3.08 | 3.10 |
| llama3.1-8B | 3.32 | 3.31 | 3.33 |
| llama3.3-70B | 1.97 | 1.98 | 1.92 |
| qwen3-30B-2507 | 4.34 | 4.54 | 4.46 |

The families are identical for all six models. In-band fractions by family are also flat (llama70B: 29.3/29.4/26.0). So H0 measured how attention behaves on essentially zero-entropy repeated text, and the needle (~20 tokens out of 120k) is invisible in per-head medians.

This matters most for exactly the comparison you're worried about, because models react to degenerate text in *opposite* pathological ways. On 10,000 near-identical copies, keys differ only by position; a model that represents content well has no basis to prefer one copy, so attention smears — llama3.3-70B has 63% of heads with φ > 0.55 (near-uniform) and τ = 1.36. Qwen3 collapses the other way: qwen3-30B's *median* top-1 attention weight is **0.198 over 120k tokens** — that is sink/degenerate-collapse territory, and 77.7% of its heads put >10% of mass on one token. Neither behavior tells you what these models do on a real 128k document. The "big models are diffuse" trend may be partly real, but on this input it is confounded with "how does this model degenerate on repeated filler."

## 2. The practical evictor never ran (real bug)

In [run_h0.py:250-254](ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/run_h0.py#L250-L254), `prev_a` stored at step *t* has length L+t, but at step *t+1* the current logits have L+t+1 entries, so the guard `pa.numel() >= sh.numel()` is **always false** and `practical_score` is always `None`. I confirmed: zero `*_practical*` or `oracle_evict_advantage*` columns exist in any of the 1.15M rows. Yet [alloc.py:166-168](ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/sievelib/alloc.py#L166-L168) says the oracle corner "is therefore an upper bound on any real evictor and the honest headline is the gain over the PRACTICAL corner" — and report.py would have printed it, so its silent absence went unnoticed.

Consequence: every `gain_e` compares against an evictor that ranks by true current-step sensitivity a·‖v−o‖ — information no deployable evictor has. This biases *every* in-band fraction downward, and most strongly for the sharp models: 99.0% of qwen3-30B's out-of-band heads are out because *oracle* eviction is near-lossless (`gain_e < 2`). The STOP verdict literally reads "eviction already captures the gain" — what was measured is "an oracle evictor captures the gain." The fix is one line: pad `pa` for the newly generated token (any real evictor keeps the newest token by recency) instead of requiring `pa` to be at least as long.

## 3. The in-band metric is a φ-window, so ctx is baked into the verdict

I binned per-head in-band rate by φ = n95/L. For **every** model, in-band peaks at φ ∈ (0.125, 0.375] — right at/below the fraction the B=3 eviction corner keeps (3/8 = 37.5%) — and collapses at both ends. Example (llama3.3-70B, % in band by φ bin): φ<0.02 → 26%, 0.125–0.25 → 70%, 0.25–0.375 → **94%**, 0.55–0.75 → 20%, >0.75 → 1%. The models' headline numbers are just their φ histograms pushed through this fixed window:

- llama3.3-70B: 63% of heads at φ>0.55 → out on the diffuse side (90% of its out-of-band heads have `gain_u<2`, i.e. uniform-is-fine).
- qwen3-30B: 61% of heads at φ<0.05 → out on the sharp side (99% evict-fine).
- mistral: φ mass sits in the middle → 84.7%.

Since φ = n95/L, doubling the context with fixed absolute attention support halves φ and slides heads toward the evict corner, while near-uniform heads stay pinned at φ≈1. So a 128k run and a 32k run of the *same* model would land in different verdict regimes purely mechanically. Your observation about the 128k rows is partly this (though note llama3.1-8B at 128k is still 60.4% — the two failures are one 128k-diffuse and one sharp model, and qwen3-8B at 40k is also low; there's a strong Qwen3-family/sink effect on top of the ctx effect). Also, the band fraction is budget-dependent in the direction you'd predict: qwen3-30B goes 0.6% → 14.8% from B=1 to B=3 (larger kept fraction overshoots its tiny φ less), and "in band at any budget" is 18% vs llama70B's 43%.

## On the τ/ln2 "prediction" — it's near-circular

`ladder_bits` is std(log₂(aᵢ·wᵢ)) ([alloc.py:140-143](ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/sievelib/alloc.py#L140-L143)), and log₂ aᵢ = sᵢ/ln2 − log₂Z, so std(log₂ a) = τ/ln2 *by algebra*. The measured ladder differs from τ/ln2 only by the value-norm term w and the sens>0 filter. The 1–3% agreement across three architectures validates that your pipeline computes consistent quantities — it is not independent evidence the theory is right. The real external test of the theory was the Gaussian-logit simulation, and as your own analysis note correctly concludes, it failed because φ decouples from τ on heavy-tailed real logits. So "the prediction is correct, therefore the measurement must be trustworthy" doesn't hold.

## Direct answers to your three questions

**(1) Are we measuring what we want, including for large models?** The per-head machinery, yes (L2 capture passed at ~3e-3 on both large models; the exact-error paths are careful). But the workload invalidates the *content* of the measurement — you measured attention regimes on repeated filler, and the failed niah/cont check proves it. One caveat specific to large models: L1/L3 validated on proxies with n_rep=4 and 2, while both 128k large models run n_rep=8, so their attention shape was covered only by unit tests plus the single L2 check. Good news: the 128k qwen3-30B run *was* the `-2507` variant (native 262k), so there's no RoPE-extrapolation issue.

**(2) Is the methodology correct?** The in-band construction is a fair operationalization of "interior beats both corners," with two flaws: the eviction corner is accidentally oracle-only (bug above), and the corner's kept fraction B/maxb makes the band a fixed *fractional* window while real head support is closer to absolute — which is what couples the verdict to ctx.

**(3) Do parameters make comparisons non-comparable?** Yes, in this order of importance: **ctx** (via the φ window — this is structural, not incidental), **prompt degeneracy interacting with model identity** (amplifies both the 70B diffuseness and the Qwen3 sink sharpness), **architecture family** (all Qwen3 models are sink-heavy; qwen3-30B-2507 is also a different post-training lineage than qwen3-8B, so it's not a clean "same family, bigger" comparison — the only clean size comparison you have is llama 8B vs 70B at matched 128k), and the **oracle corner** (penalizes sharp models most). Head count itself is not a mechanical problem (per-head medians), though more heads in a bigger model plausibly means more no-op/redundant heads, which is a *real* effect that would survive better prompts.

## What I'd run before believing any cross-model conclusion

1. Point `H0_CORPUS` at real long text (e.g., PG-19 books, concatenated code), rerun, and gate on the niah > cont ladder check actually passing.
2. Fix the `practical_score` off-by-one and re-derive the verdicts against the practical corner — this is the headline your own docs promise, and it will most likely move qwen3-30B off STOP.
3. Run one model (llama3.1-8B is cheapest) at 32k/64k/128k on the same corpus to measure how much of the band fraction is pure φ-window mechanics, and run every model at a common 32k for the cross-model table.
4. Report a sink-excluded φ (drop the top-k or position-0 mass from n95) alongside the current one — for Qwen3 that single change will likely reshape the φ histogram dramatically.

The uncomfortable trend might survive all of this — "bigger model → more heads → more redundant/diffuse heads" is a coherent hypothesis. But right now the table is measuring (filler degeneracy × ctx window × oracle corner), and each of those three biases happens to push the large models down.
