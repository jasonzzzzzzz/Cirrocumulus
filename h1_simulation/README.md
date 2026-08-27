
# The actual problem being solved

When an LLM generates text, it keeps a KV cache — a Key vector and Value vector for every past token, in every layer. For long contexts this cache gets huge, and it lives in HBM (GPU memory), which is limited and also the memory-bandwidth bottleneck. You already know the fix: store it at lower precision (FP8, INT4, etc.) to shrink it. But there's a design question underneath "just pick a bit-width":

Given a fixed total memory budget for the KV cache, is it better to (a) give every token the same number of bits, (b) keep a few tokens at full precision and throw the rest away entirely, or (c) give different tokens different numbers of bits based on how important each one is?

That's H1. The code is a simulation that tests all three strategies at the same total budget and measures which produces the least error in the attention output

# What "bits" 0, 1, 2, ... mean here


Not literally FP8 vs INT4 formats — the code is deliberately format-agnostic. b is an abstract "how many bits does this token's K/V get" knob, and C_MSE[b] / C_PROD[b] are just TurboQuant's published numbers for how much noise a quantizer of that many bits injects, regardless of whether it's implemented as int, float, whatever. So b=4 means "4-bit quantization, with the distortion TurboQuant measured for 4-bit quantization" — you could point that at INT4 or FP4, the simulation doesn't care, it's only reasoning about the noise/bit tradeoff. b=0 is the special case of zero bits = don't store this token at all = evicted. b=99 is a sentinel meaning "don't quantize, keep it exact" (used only in the H2O/SnapKV-style scheme).

# The budget IS the capacity constraint — just expressed differently

You asked "did you consider HBM capacity, only evict when exceeding capacity?" — yes, implicitly, that's exactly what B (budgets=(1,2,3,4,6)) is. B = mean bits per token, averaged over the whole cache. Memory used = B × L × D × (num tokens/layers/heads) — so fixing B is mathematically the same thing as fixing "how many bytes of HBM this cache is allowed to occupy." The three strategies are three different ways to spend that same fixed memory budget:

- uniform: everyone gets exactly B bits (e.g. everyone gets 3 bits).
- evict+fp16: some fraction of tokens get full 16-bit precision, the rest get 0 bits (thrown away), sized so the average still comes out to B.
- waterfill: bits are handed out unevenly — important tokens get more, unimportant tokens get fewer or zero — but again the average must equal B.

So there's no separate "if we exceed capacity, evict" rule. Eviction happens because the memory budget is tight and some tokens simply aren't worth spending any of that budget on. If you set B=16 (full precision budget), nothing would need to be evicted at all — there'd be enough budget to give everyone full bits.

# Why the algorithm starts everyone at b=0 (evicted) and "buys" bits upward

Think of it like grocery shopping with a fixed amount of cash, where every item has diminishing returns — the first bite of quantization precision buys you a huge error reduction, the fifth bite barely helps. You start with an empty cart (every token at 0 bits — full eviction, cost = 1.0 = "worst case, I know nothing about this token"). Then you repeatedly ask: of every possible next purchase, across every token, which one gives me the most error-reduction per bit of budget spent? You buy that one. Repeat until your budget (B × L total bits) runs out.

- "Cost-reduction per bit spent" = (c[b] - c[b+1]) / (b+1 - b) — how much distortion you shave off by bumping this token from b bits to b+1 bits, divided by how many extra bits that costs.
- "weighted by w" — w = a_i * ||v_i - o|| is how much this particular token actually matters to the final attention output (its attention weight, times how far its value vector is from the output — i.e. how wrong the answer would be if you got this token totally wrong). A token with huge attention mass gets weighted much more heavily than a token nobody attends to.

So the purchase priority is: spend your limited bits where they buy the most error-reduction for the tokens that matter most. Tokens with tiny w (attention basically ignores them) never get a good enough "deal" to win a purchase before the money runs out — so they stay at b=0, i.e., evicted. That's not a separate eviction decision, it's just "we ran out of budget before this token's number came up." This is literally the classic fractional knapsack greedy algorithm, applied to KV cache bits — greedy-by-best-ratio is provably optimal here because the cost curve is convex (diminishing returns per bit), which is exactly why convex_envelope() matters: it filters bit-widths down to only the ones that can ever be an optimal "next purchase" (a bit-width that's dominated by a mix of its neighbors on the curve is never worth buying directly).

The term "reverse water-filling" comes from information theory's water-filling (pour more power into channels with less noise) — here it's "pour more bits into tokens with more attention mass," and unlike classic water-filling, the low-value tail gets driven all the way to zero (evicted) rather than just "a little less."

# Worked example: 3 tokens, a tiny budget

Use 3 tokens and a small budget so every greedy step can be checked by hand.

**Importance weights** (`w = a_i * ||v_i - o||` — attention mass times how wrong the output would be without this token):

| Token | w (importance) |
|---|---|
| A (a "needle" the query aligns with) | 9 |
| B (moderate) | 3 |
| C (background) | 1 |

**Cost table** (a simplified `C_MSE` — distortion at each bit-width, lower is better, decreasing/convex by construction so nothing gets dropped by `convex_envelope()`):

| bits b | 0 (evict) | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| c[b] | 1.0 | 0.4 | 0.2 | 0.1 | 0.05 |

**Budget**: `B = 1` bit/token average, `L = 3` tokens → total budget = **3 bits** to spend across all 3 tokens.

## Step 1 — list every possible purchase

For every token, for every step up the bit ladder, the gain-per-bit is `w_i × (c[b] − c[b+1])` (each step costs 1 bit here):

| purchase | gain (w × Δc) |
|---|---|
| A: 0→1 bit | 9 × (1.0−0.4) = **5.4** |
| A: 1→2 | 9 × (0.4−0.2) = **1.8** |
| B: 0→1 | 3 × (1.0−0.4) = **1.8** |
| A: 2→3 | 9 × (0.2−0.1) = 0.9 |
| B: 1→2 | 3 × (0.4−0.2) = 0.6 |
| C: 0→1 | 1 × (1.0−0.4) = 0.6 |
| ... | (smaller and smaller) |

This is exactly the `gains` array built at `h1_sim.py:109`.

## Step 2 — sort by gain, descending, spend the budget

This is `h1_sim.py:115` (`order = np.argsort(-gains)`) and the loop after it: a token's next step can only be bought after the previous one (`level[i] != k: continue` — no skipping bits), and buying stops once the budget is spent.

1. **Buy A: 0→1** (gain 5.4, cost 1 bit). Spent = 1/3. A is now at **1 bit**.
2. Next best is a tie at 1.8: **A: 1→2** and **B: 0→1**. Buy both.
   - Buy A: 1→2. Spent = 2/3. A is now at **2 bits**.
   - Buy B: 0→1. Spent = 3/3. B is now at **1 bit**.
3. Budget exhausted (spent = budget = 3). Loop stops. **C never wins a purchase** — it stays at its starting value, `b=0`.

## Result

| Token | bits allocated | weighted cost `w × c[b]` |
|---|---|---|
| A | 2 | 9 × 0.2 = 1.8 |
| B | 1 | 3 × 0.4 = 1.2 |
| C | 0 (evicted) | 1 × 1.0 = 1.0 |
| **total** | **3 bits spent** | **4.0** |

Compare to the **uniform** corner at the same 3-bit budget (everyone gets 1 bit, since 3 tokens × 1 bit = 3 bits):

| Token | bits | weighted cost |
|---|---|---|
| A | 1 | 9 × 0.4 = 3.6 |
| B | 1 | 3 × 0.4 = 1.2 |
| C | 1 | 1 × 0.4 = 0.4 |
| **total** | **3 bits spent** | **5.2** |

Same total memory, but water-filling's total weighted error (4.0) beats uniform's (5.2) — it took the bit uniform "wasted" on C (which barely matters) and gave it to A (which matters a lot and has a much steeper cost curve at low bit-widths). C wasn't evicted by a rule — it lost a bidding war: its gains were always smallest on the list, so the budget ran out before its turn.

# Where this toy example breaks — the flaws in `superseded_v1`

The mechanics above (start everyone at b=0, greedily buy the best gain-per-bit) are exactly what `h1_sim.py` and `h1_tau.py` do. But three things about the *numbers feeding that mechanism* are wrong in those files — see `superseded_v1/README.md` for the original audit note. Here's what each one would change in this exact example.

## Flaw 1 — gains reported against uniform only

Above, we only checked water-filling against the **uniform** corner (4.0 vs 5.2, a 1.3× gain). A fair test also has to check it against the **evict + fp16** corner (keep a few tokens at full precision, drop the rest entirely) and report the gain against whichever corner is *better*, not against uniform by default. Our 3-token/3-bit example is too small to show this cleanly — a full-precision token costs 16 bits in the real code, so a 3-bit total budget can't afford even one, and the evict+fp16 corner degenerates to "evict everyone" (weighted cost 9+3+1=13, much worse than uniform). But at larger budgets or higher τ, evict+fp16 becomes the *stronger* corner, and `h1_tau.py` never checks that — it only ever compares against uniform, which is why its reported gain climbs to 70× instead of the corrected code's finding that the gain peaks near τ≈1.25 and collapses back to 1.0× by τ≈2 once you compare against the better of both corners.

## Flaw 2 — mixed cost units

Our cost table `{0:1.0, 1:0.4, 2:0.2, 3:0.1, 4:0.05}` is only valid **at τ=1** (the std of the logits). That's exactly `h1_sim.py`'s bug: it never has a τ knob at all, so `C_MSE` gets reused unchanged no matter how spread out the real logits are.

The corrected code (`sievelib/alloc.py`) documents a concrete case of what goes wrong: at τ=2.5, the buggy version told the allocator a 1-bit key costs **0.42** — cheaper than eviction's 1.0, a "good deal," buy it. The true absolute cost at that τ is **2.6** — nearly 3× *worse* than eviction. A 6× error, and it errs in exactly one direction: it never lets the allocator evict.

Translated into our example: if Token C were being evaluated at τ=2.5 instead of τ=1, its 0→1 purchase option should use the true absolute cost (2.6) instead of the table's 0.4, giving a gain of `1 × (1.0 − 2.6) = **−1.6**` — negative. Buying C a bit would make total error *worse*, so it should never be purchased; C should either be evicted outright or (if the budget allows) jumped straight to a much higher bit-width where the absolute cost finally drops back below 1.0. The uncorrected table instead reports a *positive* gain of `1 × (1.0 − 0.4) = 0.6` for that same purchase and happily buys it.

`h1_tau.py` tries to fix this by scaling — `tau**2 * C_MSE[b]` — but then clamps: `min(0.999, tau**2 * c_b)`. At τ=2.5 that clamp reports C's 1-bit cost as `min(0.999, 2.5**2 * 0.4) = min(0.999, 2.5) = 0.999` — almost identical to eviction's 1.0, instead of the true 2.6. The clamp doesn't just fail to fix the units bug — it manufactures a fake "quantization is always ≈free" signal at precisely the τ where the honest story is "quantization is now actively harmful, evict instead."

## Flaw 3 — linearized cost, not exact recomputation

The "weighted cost" totals we computed above (4.0 for water-filling, 5.2 for uniform) are `Σ w_i × c(b_i)` — an additive, per-token, first-order proxy for error. That's fine to use for *choosing* bit-widths (it's what `alloc_waterfill`'s greedy purchases are ranked by), but `h1_sim.py` and `h1_tau.py` also use it — or a Monte-Carlo stand-in that injects synthetic Gaussian noise with that same variance — to *report* the gain. Real attention error doesn't decompose additively like that: softmax has one shared normalizer (`a /= a.sum()`), so noise on token A's logit changes the effective weight of every *other* token too, an interaction the sum `Σ w_i c(b_i)` can't see.

The corrected pipeline (`sievelib/alloc.py: exact_error()`) keeps the linear proxy only to *choose* the allocation, then measures the *reported* error by actually quantizing the real keys and recomputing the real softmax and output — no modeled noise standing in for a real quantizer. On synthetic heads, the linear proxy this toy example uses overstates the true, exactly-recomputed gain by 20–60% (`lin_ratio` ≈ 1.2–1.6). So the 1.3× headline from our worked example (4.0 vs 5.2) should be read as an optimistic upper bound, not the actual expected benefit — the real gain from running quantized keys through a real softmax would likely come out smaller.

# Where `C_MSE` actually comes from, how real keys get measured, and what belongs in a paper

## 1. Where "TurboQuant" comes from, and does the repo's table match it

There is no citation, URL, or DOI to an external "TurboQuant" reference anywhere in this repo — the docstrings just assert "TurboQuant's tabulated distortion constants" without sourcing them. Searching externally, there is a real match: **[TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874)**, a Google Research paper. The mechanism lines up almost exactly with `sievelib/quant.py`'s docstring: random (Haar) rotation, then per-coordinate scalar Lloyd-Max quantization, distortion provably within a small constant factor (~2.7×) of the information-theoretic optimum, and an MSE-only variant vs. a variant that layers a Quantized-JL (QJL) correction on top — exactly the `C_MSE` vs. `C_PROD` split in `superseded_v1/h1_sim.py:29-32`.

One nuance: the paper says each rotated coordinate follows a **Beta distribution that converges to Gaussian as dimension d grows**, not an exact Gaussian — close enough to "roughly Gaussian" for these dimensions, but worth being precise about.

**Do the exact numbers `{0.36, 0.117, 0.03, 0.009}` (`C_MSE`) match a specific table in that paper?** Unverified. The paper's abstract/HTML don't expose the numeric distortion table, and the PDF didn't extract cleanly through automated tools (no local PDF renderer available, and the fetch summarizer couldn't pull tabular data out of the compressed PDF stream). Treat "these numbers come from that paper's table X" as **plausible but unconfirmed** — before citing it, open the PDF directly and find the actual bits-vs-distortion table/figure to confirm the numbers, not just the method, match.

Adjacent/derivative papers also worth knowing about: [Statistical Inference and Quality Measures of KV Cache Quantisations Inspired by TurboQuant (2605.08114)](https://arxiv.org/pdf/2605.08114) and [TurboESM (2603.26110)](https://arxiv.org/pdf/2603.26110), which apply the same rotation+QJL idea specifically to KV caches.

## 2. How `sievelib/alloc.py` measures on real keys

Traced through `h0_measurement/models.yaml`, `sievelib/probe.py`, and `h0_measurement/run_h0.py`:

- **Which models**: a real, swappable roster of open-weight HF models — `Qwen3-1.7B`, `Qwen3-8B`, `Qwen3-30B-A3B`, `Llama-3.1-8B-Instruct`, `Llama-3.3-70B-Instruct`, `Mistral-7B-Instruct-v0.3` (`models.yaml:39-70`). Adding a model is just appending a YAML entry.
- **Which keys**: `probe.py` hooks into `transformers.ALL_ATTENTION_FUNCTIONS` to capture the model's actual post-RoPE queries during real decoding on prompt families (`niah`/`qa`/`cont`), and reads the real Keys/Values straight out of the model's own KV cache — architecture-agnostic across Llama, Qwen3 (QK-norm), Gemma-style models.
- **Which quantizer**: TurboQuant's own pipeline, implemented locally in `sievelib/quant.py` (rotation + Lloyd-Max + norm correction), run on those real captured keys. In `run_h0.py:157-163`: for every bit-width in `bit_list` (1,2,3,4,5,6,8), `quant.quantize_keys(K, b, R, norm_correct)` quantizes the *actual* keys `K`, then `quant.logits_gqa(qd, Kq, scl)` recomputes the *actual* attention logits with those quantized keys, giving `shat_all[b]`.
- **How the noise gets measured**: `alloc.noise_model(s, shat)` computes `sig2[b] = Var(shat[b] - s)` directly (`alloc.py:52-64`) — the real logit error from the real quantizer on real keys from a real model, not a lookup in a published table.

## 3. Can both idealized constants and measured results be reported, and which motivates the problem better?

Yes to both — this repo's own structure already argues for exactly that split.

- **The idealized, dataset/model-independent constants** (TurboQuant's theoretical curve) are the right thing to open a paper's motivation with. They establish the *fundamental, irreducible* shape of the noise-vs-bits tradeoff — diminishing returns per bit — independent of any specific model's quirks, which is exactly the intuition that makes reverse water-filling plausible: *if* the cost curve is convex, unequal allocation beats equal allocation, regardless of what model you're on. A reader can verify that argument without trusting your measurement pipeline.
- **The measured results** are what the paper's actual claims and numbers must rest on — real KV caches have outlier channels, per-head structure, and non-Gaussian statistics the idealized table can't see, which is exactly why the corrected pipeline replaced the table lookup with `noise_model()`'s direct measurement. Headline numbers must come from here.

This repo's structure already encodes that division: H1 (the idealized simulation) is motivation/oracle-only — it asks "does the theorem have *any* chance of paying off," not "what's the number" — while H0 (real-model measurement) is the actual go/no-go gate. The `superseded_v1` mistake was never "using the idealized model" — it's a legitimate first pass. The mistake was letting the idealized model's numbers *also* stand in as the reported evidence (flaw 3 above), and letting its cost units silently assume τ=1 without ever validating against measurement (flaw 2 above). The rule that keeps both citable without contradiction, straight from `sievelib/alloc.py`'s own header comment: **the theoretical/idealized model may be used to *choose* the allocation; only exact, measured recomputation may be used to *report* the result.**





