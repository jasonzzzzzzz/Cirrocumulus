


# Why synthetic is bad

it's a wrong input: the per-head numbers are internally correct measurements of attention on 8 sentences repeated 10,000×. The experiment's own acceptance test (niah ladder > cont ladder) fired, and the other problems and bugs only matter once the input is meaningful — a perfectly de-biased verdict on filler text is still a verdict about filler text.

- The two pathologies push in opposite directions and both should relax on real text: llama3.3-70B's 63% near-uniform heads (φ > 0.55) exist because no copy of the filler is preferable to another; real content gives heads a basis to concentrate, so its diffuse-side out-of-band mass should shrink. Qwen3's sink collapse (median top-1 = 0.198 over 120k tokens) should soften, raising φ off the floor and pulling heads out of the evict-corner. Both movements are toward the interior band, so both currently-STOP models plausibly move up, and the "bigger → less benefit" trend could attenuate substantially or invert.

- The family separation (niah wider than cont) should finally appear, which becomes your standing validity gate.

- Retrieval heads will exist for the first time — the ~20-token needle is currently invisible in per-head medians.

---

## Q&A: why input affects head heterogeneity, and how to choose a corpus

### Isn't what we profile agnostic to the input prompt?

No. Every metric H0 reports is a function of the attention distribution, which
depends on both the model *and* the prompt — not a structural property of the
weights alone. Only things like layer/head count or GQA ratio are input-agnostic,
and none of those are the verdict.

**Independent variables:** model, ctx `L`, prompt family (`niah`/`qa`/`cont`),
prompt seed, decode step, layer, head, budget `B` ∈ {1..4}, `maxb`=8.

**Measured metrics (all per head/prompt/step, all input-dependent):**

| metric | what it is |
|---|---|
| `tau`, `tau_nosink` | std of attention logits |
| `top1` | largest attention weight |
| `n95`, `eff_frac` (φ = n95/L) | tokens holding 95% of mass |
| `entropy` | attention entropy |
| `ladder_bits`, `ladder_bits_a_only` | std of log₂ sensitivity (≈ τ/ln2 algebraically) |
| `w_cv` | spread of ‖vᵢ−o‖ |
| gains vs corners, `in_band`, routed gain | the go/no-go statistic, a function of all of the above |

There is no input-agnostic metric in the pipeline, so the verdict inherits the
input-dependence fully.

### (1) "The acceptance test (niah ladder > cont ladder) fired"

"Fired" = the built-in validity alarm went off. `prompts.py`'s own docstring
says niah should induce retrieval behavior → wider `ladder_bits` than cont; if
not, "the haystack is not inducing retrieval behaviour — point H0_CORPUS at
real text and rerun." In the actual runs the families came out statistically
indistinguishable — the check tripped, and its own documented remedy is this bug.

### (2) "A perfectly de-biased verdict on filler text is still a verdict about filler text"

There are internal bugs (off-by-one in the practical evictor, φ-window/ctx
confound, oracle corner) and an external validity problem (wrong input). Even
with every internal bug fixed, you'd have a flawless measurement of the wrong
question — "how do heads behave on 8 sentences tiled 972×" — which says nothing
about deployment. The input fix gates whether the other fixes matter.

### (3) "Real content gives heads a basis to concentrate, so its diffuse-side out-of-band mass should shrink"

On synthetic filler, every copy of a sentence is identical content differing
only by position, so a content-based head has no reason to prefer one copy over
another — attention smears near-uniformly (llama3.3-70B: 63% of heads at
φ > 0.55). These heads are out-of-band on the diffuse side (uniform
quantization already fine, no allocation gain). Real text gives distinct
tokens, so a head *can* single out relevant ones → n95 shrinks → φ moves off
~1 into the interior → the diffuse-side pile should shrink.

### (4) "Raising φ off the floor and pulling heads out of the evict-corner"

Qwen3's opposite pathology: sink collapse (median top-1 = 0.198 over 120k
tokens) → n95 tiny → φ ≈ 0 (the floor). Such heads already sit where the
eviction corner (keep a fraction of tokens at full precision) is optimal, so
allocation can't beat it — out-of-band on the sharp side. If real text softens
the sink, mass spreads over more genuinely useful tokens, φ rises off the
floor, and heads move into the interior band. Both (3) and (4) converge on the
middle band — why both STOP models plausibly move up.

(Reminder: the "corners" are the two pure baselines allocation is compared
against — the quantization corner, keep every token at uniform low bits, and
the eviction corner, keep a fraction of tokens at max bits. "In band" = beats
the best of the two by ≥2×.)

### (5) What is a "family"? What is the invisible needle?

A family is a prompt template layered on the haystack (`prompts.build`):
- **niah** (needle-in-a-haystack): haystack + one hidden sentence with an
  access code + a retrieval question. Expected to induce sharp retrieval
  heads → wide ladder.
- **qa**: haystack + a summarization instruction. Mixed regime.
- **cont**: the haystack alone, pure continuation. Expected diffuse heads →
  narrow ladder.

The niah>cont ladder gap is the validity gate from (1). "The ~20-token needle
is currently invisible in per-head medians" means the needle is ~20 tokens out
of ~120k (0.02%), and on the degenerate haystack no head locks onto it —
attention is instead captured by repetition (induction) or the sink. Since
each head's stats are reported as medians across prompts/steps, that tiny
effect washes out entirely; no head currently behaves like a retrieval head.

### Which real corpus to choose, and does it matter?

Yes, corpus choice moves the numbers, but synthetic→real crosses a regime
boundary (degenerate → normal attention), while fiction-vs-tech-report is
variation *within* the normal regime — first-order vs. second-order effect.
Guidance:

1. **Match the claim.** The conclusion is about long real documents, so the
   corpus should represent that distribution. `plan.md` already proposes
   **PG-19 books + concatenated code** — narrative prose brackets the
   diffuse/low-repetition end, code is naturally repetitive/structured
   (legitimate induction-head activity, unlike the pathological filler
   repetition). PG-19 alone is the conservative default if only one is used.

2. **Treat corpus as an axis, not a nuisance.** Run the same models on both
   corpus types and check whether band fraction and phase assignment
   (diffuse-out / in-band / evict-out) are stable. A flipped verdict between
   books and code is a real finding about input-dependence, not a bug — same
   pattern as the ctx-as-axis recommendation in the README.

3. **The niah>cont gate protects you per-corpus.** Whatever text is chosen,
   the ladder check tells you whether it induces retrieval behavior, so a bad
   corpus choice degrades to "gate fails, rerun" rather than a silent wrong
   conclusion (once `plan.md`'s UNKNOWN-stamp fix lands).

Practical note from `plan.md`: at 128k ctx the builder needs ~600KB of text
per prompt; with only 6 prompts, add the per-prompt file/offset rotation
(`_corpus_text` currently only shuffles file order), or different prompts can
land on near-identical text — a milder version of the same repetition problem
across prompts instead of within one.


# Prompts

This is a debugger chat for bug ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/bugs/1_from_synthetic_to_real_corpus. 

I cannot understand texts in @ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/bugs/1_from_synthetic_to_real_corpus/why.md .  

My basic question is : why the input affect the head heterogenuity characterization? Isn't the things we profile is agnostic with the input prompt? What metrics we tested correlates (directly and indirectly) with the input -- list all the variables and metrics we test and list all the affected metrics in this case? 

Even if I know using 8 sentences repeated 10,000× is definitely not as good as using a real document long enought. My further questions: 
(1) What does "The experiment's own acceptance test (niah ladder > cont ladder) fired" mean? 
(2) What does "a perfectly de-biased verdict on filler text is still a verdict about filler text" mean?
(3) What does "real content gives heads a basis to concentrate, so its diffuse-side out-of-band mass should shrink." mean?
(4) What does "raising φ off the floor and pulling heads out of the evict-corner" mean?
(5) What is a "family"? What are each family, like niah, and cont? What is "the ~20-token needle is currently invisible in per-head medians"

My final question is: if non-synthetic input is important , which affects the metrics we measure, then which real prompt I should choose -- this still have large impact, right? Choosing a tech report, and a fiction book may make the metrics we will measure different. How to choose?




