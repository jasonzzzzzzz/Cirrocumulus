This report is more informative than it looks, and I think the gate is wrong — not the data. Let me check that claim numerically before asserting it.The gate looks unpassable by construction. Let me verify that arithmetically.Confirmed: **the gate cannot pass**. Let me write the replacement and the corrected phase figure.Panel 2 is misleading — it implies ladder width separates the sharp edge, but qwen3-8B (4.42, fine) and qwen3-30B (4.55, fails) sit on the same side. That's precisely why the second axis is needed. Fixing.Short answer: **fix the gate, keep the design.** The experiment is fine — one uncalibrated threshold is vetoing a good result. Here's the reasoning for each question.

## (2) Why every model fails the gate

The gate is **unpassable by construction**, and I wrote it, so this is my error.

It demands the median-over-heads ladder width on `niah` exceed `cont` by ≥ 0.1 bits. But ladder = τ/ln2, and τ is the standard deviation over **all L logits** — a bulk second moment. A needle is *one token out of 131,072*. I simulated it:

| needle tokens | needle logit | Δ ladder | vs 0.1 b gate |
|---|---|---|---|
| 1 | 10·τ | **+0.0016 b** | fail |
| 1 | 20·τ | +0.0063 b | fail |
| 10 | 10·τ | +0.0159 b | fail |
| 100 | 10·τ | +0.156 b | pass |

A *perfect* single-needle retrieval moves the statistic by ~0.0016 b against a 0.1 b threshold — **70× less than the phenomenon can physically produce.** You'd need ~100–500 simultaneous needles to trip it.

Second compounding error: retrieval is carried by a small minority of heads, and I took the **median** over heads. That's dominated by heads that never participate, so it cannot move no matter how well retrieval works.

This is the *same mistake* the project already made once — using τ, a bulk moment, to detect what is actually an extreme-value event. I diagnosed that for the regime variable and then rebuilt it into the validity check.

**Your observed deltas (−0.010 to +0.061 b) are exactly what successful retrieval looks like under this statistic.** `sievelib/validity.py` replaces it with two checks matched to the phenomenon: task-level (does the model emit the needle code — behavioural ground truth) and head-level (attention mass on the needle span in the *top* heads, a max not a median). Either suffices.

## (3) Why it looks bad when the numbers are good

Because the gate is stamping UNKNOWN over a result that got **better and more robust**:

| model | band (synthetic) | band (PG-19) | Δ |
|---|---|---|---|
| mistral-7b | 84.7% | 84.5% | −0.2 |
| llama3.1-8B | 60.4% | 69.1% | +8.7 |
| qwen1.5-MoE | 62.0% | 68.8% | +6.8 |
| qwen3-8B | 45.9% | 54.3% | +8.4 |
| qwen3-30B | 14.8% | 14.1% | −0.7 |

Max shift 8.7 points, **no phase assignment changed**, all four in-band models improved. Mean band ~58%, routed gains 1.94–4.06×. That's a robustness result: the conclusion doesn't depend on the haystack. Your instinct is right — this run is stronger than the last one.

## (1) The thresholds — you're right to doubt them

φ = n₉₅/L was an **empirical proxy that happened to correlate on six points**, and it's now not even monotone on the new data (φ 0.207 → 69.1% but 0.265 → 68.8%). Worse, φ is mechanically confounded with context length: the same head at different L gets different φ. That's disqualifying for a phase-diagram claim.

Two theory-derived quantities do the job better, and each owns one edge:

- **Ladder width** (diffuse edge). llama3.3-70B at 1.96 b fails; everything ≥ 2.68 b works. But at the wide end it separates *nothing* — qwen3-8B (4.42 b, fine) and qwen3-30B (4.55 b, fails) are indistinguishable.
- **Dead-2-bit-tier fraction** (sharp edge). 3% → 84.5%, 7% → 68.8%, 13% → 69.1%, 39% → 54.3%, **75% → 14.1%**. Essentially monotone, and it's the only axis that separates those two models.

Both come straight from comparing τ²c_b against the derived c₀ = 1. That makes the boundaries **derived rather than fitted** — a strictly better claim than φ.

Being honest about what's still weak: the diffuse boundary rests on one point (llama3.3-70B, which wasn't even in this re-run), and the sharp boundary is unconstrained anywhere between 39% and 75% dead tiers. The figure hatches that region rather than drawing a confident line.

## What I'd actually do

**Don't touch the design.** Four of five models sit at 54–85% in band with 1.9–4.1× routed gain, the one failure fails by the predicted mechanism (75% dead tiers, gain vs eviction exactly 1.0×), and everything survived a corpus change.

**Do three things, roughly a week:**

1. Swap in the new validity gate and re-run — no GPU cost beyond the re-run itself.


One reframing worth taking seriously: `niah ≈ cont` in bulk statistics is not a bug, it's a **finding**. It says a head's phase is a property of the model, not the prompt — which is exactly what makes offline per-head calibration transfer across workloads. That's the assumption the router needs, and this run supports it. Turn the failed gate into a contribution.