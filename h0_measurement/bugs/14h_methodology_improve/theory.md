**Short answers:**
- **Does the design work in theory?** As built now, only under a condition it never checks; R1–R4 are what theory predicts. With a certified controller on top of the existing two tiers, yes: provably correct at the attention level on any softmax-attention model.
- **The claim:** it holds as a theorem at the attention level (part 2). My earlier sentence was too strong end to end. The tolerance needed for a given output fidelity depends on the model, as one measured number.
- **SOTA models:** the theory applies to every softmax-attention layer in any architecture. Latent attention, local/global layouts and hybrids need small adapters, and the savings vary a lot by family.

Your message was cut off after "Kimi 3,", so part 4 covers the open-model families broadly. Tell me any specific model you meant.

## 1. Does the design work in theory?

**The current design** (fixed r = 1/8, one vote at question time, unread rows evicted). For one head at one step, attending only to rows S gives an exact identity:

```
o − o_S = ε · (ō_unread − ō_read)      ε = attention mass on unread rows,  ō = attention-weighted mean value
```

The design picks S before any of the answer's queries exist. Unless the read keys surround all the others, some query direction puts almost all its mass on unread rows. So no selection made in advance can carry a guarantee. The design is correct exactly when two conditions hold:
- **C1, concentration:** a small set of rows holds almost all of the answer-time attention.
- **C2, predictability:** the question's attention already points to that set.

| Workload | C1 | C2 | Theory predicts | Observed |
|---|---|---|---|---|
| Single needle (R1, R2) | yes | yes | near FP | near FP, 0 answers lost |
| Many values (R3a multivalue) | yes, per step | no: each step needs a different needle | vote fails, oracle fine | Qwen: oracle +0.6, vote +69 nats |
| Aggregation (cwe, fwe) | no: mass is flat | — | even the oracle fails | oracle +0.75 to +2.6 |
| Enumeration (re-ranking) | partly | no: drifts during the answer | re-selection wins | Quest, which re-selects every step, beats us by 14.7 nats |

Every R1–R4 result fits this table. The failures come from the design's structure, not from badly chosen parameters.

**The two-tier store is the right foundation.** A guarantee needs two things:
- a cheap view of every row, to bound what goes unread;
- an exact fallback, to fix what the bound flags.

Tier 1 (4-bit, rotated, all rows) and tier 2 (exact) provide exactly these. R1 already shows the effect: dense 4-bit costs +0.07 to +0.12 nats on needles, against +0.016 for the two-tier system. Dense 4-bit puts its error on the high-mass rows, and tier 2 makes those rows exact.

**With a certified controller** (proved in part 2): per head and step, read rows exactly from tier 2 until a computable upper bound on the error falls below a target τ. Where attention is flat, read the remaining rows from tier 1 instead of evicting them.
- **Accuracy:** provably within τ, at every head and step, on any model.
- **GPU memory:** the same as a 4-bit cache, plus a small exact working set.
- **Speed:** theory promises nothing here. Per-step memory traffic, as a fraction of FP16 dense attention:
  - a full 4-bit score scan for the bound costs about 1/8;
  - reading 1/8 of the rows exactly costs about 0.08;
  - that totals about 0.20, against 0.25 for a plain 4-bit cache, so the speed gain nearly vanishes;
  - cheaper page-level bounds (per-channel min/max per 16 rows, as Quest stores) bring the total to 0.11–0.14, but only if they're tight enough on real attention, which R5 can measure from FP logs;
  - on flat heads the cost is about that of a 4-bit cache: no gain, but no loss.

**Conclusions:**
1. The current design isn't wrong. It's the special case with the certificate switched off, and it is valid only where C1 and C2 hold.
2. Keep the two tiers; they are exactly what a guarantee needs.
3. A certified controller makes attention-level correctness independent of the model.
4. Savings are limited by how concentrated attention is and by how tight a cheap bound can be. Whether the design pays is an empirical question that theory can frame but not answer.
5. The question-time vote becomes a warm start. A good vote lowers cost, but correctness no longer depends on it.

## 2. The claim, made precise and proved

**Setting.**
- One softmax attention head with query q, scores s_i = ⟨q, k_i⟩/√d + β_i, where β_i is any exactly known bias (sinks, ALiBi, masks).
- Tier 1 stores k̂_i and v̂_i for every row. The quantizer guarantees ‖k_i − k̂_i‖ ≤ η_i and ‖v_i − v̂_i‖ ≤ ν, both computable from its stored step sizes (the rotation preserves norms).
- V = max‖v_i‖, computed once after prefill. T is the set of unread rows. Read rows use exact keys.
- If read rows use 4-bit values, as the current system does, every bound below gains an additive ν.

**Lemma 1 (evicting).** o − o_S = ε·(ō_T − ō_S), so ‖o − o_S‖ ≤ 2Vε.
*Proof.* o = (1 − ε)·ō_S + ε·ō_T, and the renormalized output over S is ō_S. Subtract. ∎

**Lemma 2 (the certificate).** Let ŝ_i be the tier-1 scores and b_i = ‖q‖·η_i/√d. By Cauchy–Schwarz, |s_i − ŝ_i| ≤ b_i. With M_in = Σ_S e^{s_i} (exact, from tier-2 keys) and U = Σ_T e^{ŝ_i + b_i}:

```
ε ≤ ε̄ = U / (M_in + U)
```

*Proof.* The true unread mass Σ_T e^{s_i} is at most U, and x/(M_in + x) increases with x. ∎

**Lemma 3 (reading the unread rows at 4 bits instead of evicting).** With b = max over T of b_i:

```
‖o − õ‖ ≤ ε · [ (e^{2b} − 1)(3V + ν) + ν ]   ≈   ε · (6bV + ν)
```

*Proof.*
- Write o − õ = (ε̃ − ε)(ō_S − õ_T) + ε(ō_T − õ_T).
- Scores on S are exact, so only two things change: the split of mass between S and T, and the weights inside T. Each weight moves by a factor within [e^{−2b}, e^{2b}].
- That gives |ε̃ − ε| ≤ ε(e^{2b} − 1) and ‖ō_T − õ_T‖ ≤ (e^{2b} − 1)V + ν. ∎

The error becomes a product of two small numbers: unread mass times 4-bit error.

**Theorem.** The controller works as follows:
- start from any S (for example, the vote's rows);
- compute the bound from Lemma 1 (evict) or Lemma 3 (read the rest at 4 bits), using ε̄ in place of ε;
- while the bound exceeds τ, fetch exactly the unread row with the largest e^{ŝ_i + b_i}.

For any model with softmax attention over cached rows, any input, and any layer, head and step, it returns an attention output within τ of exact, or it reports that its row cap was reached.

*Proof.*
- The bound uses only quantities the controller computes: exact scores on S, tier-1 scores, the quantizer's step sizes, ‖q‖ and V.
- Its derivation uses only softmax algebra, Cauchy–Schwarz and the quantizer's guarantee. Nothing about the model's weights, training or data enters.
- Each added row moves its term from U to M_in, so the bound only falls.
- At S = all rows, U = 0, leaving at most the value floor ν (zero if exact values are fetched). ∎

**Why model properties change only the cost.** Each one only lengthens the loop:
- **Flat attention:** more rows before ε̄ ≤ τ.
- **Large query norms or key outliers:** larger b_i, a looser bound, extra rows.
- **Shared rows:** with many query heads per KV head (GQA), or all heads sharing rows (MLA), the bound must hold for every head, which takes more rows.
- **Answers that drift:** the bound fails more often, so more re-fetches.

None of these can push the output past τ without the controller reporting it.

**What can't be model-independent: end-to-end fidelity.** For any τ > 0, some model and input exist where a perturbation of size τ in one head flips the greedy token, because the top logits are nearly tied. That is true of FP8 and of every KV format, not only ours. So the τ needed for a given output tolerance depends on the model. Two ways to set it without tuning:
- **Anchor to FP8:** set τ to a high quantile of the per-head attention error that FP8 KV causes on that model, measured from FP runs. The design then stays within FP8-sized attention errors at every head and step.
- **Weight by sensitivity:** set τ per layer from measured sensitivity.

The injected-error check (step 3 of the protocol I described last time) confirms the link.

**The corrected claim:** a model can change how much the design saves, and how strict τ must be for a given output tolerance (one measured number). It cannot make the design exceed its attention-level target without the design reporting it.

## 3. Fit to the SOTA models we discussed

I'm not certain of the newer models' architectures; check each `config.json`.

| Model | Attention, as I understand it | Where the theory applies | What drives cost | Payoff |
|---|---|---|---|---|
| Mistral-Small-3.2-24B | full GQA, 32 query / 8 KV heads, no QK-norm | every layer, no adapter | no QK-norm → key outliers → looser bounds (rotation helps) | high; closest to Llama |
| GLM-4.5-Air | full GQA, about 12 query heads per KV head, partial RoPE | every layer | bound must hold for 12 heads per KV head; thinking → long answers → re-fetches | medium-high |
| MiniMax-M2.7 | full softmax attention, if it follows M2 | every layer | always thinks → drift → re-fetches; around 230B, larger than one node in bf16 | high per token, hard to test |
| Granite-4.2-30B | Mamba-2 hybrid with few attention layers, if it follows 4.0 | attention layers only | each attention layer carries all retrieval → stricter τ | small at 128K, grows with context |
| Qwen3.5 / Qwen3.8 27B | Gated DeltaNet hybrid, softmax in 1 layer of 4 (Qwen3.5 as I understand it; Qwen3.8 unknown to me) | the softmax layers | as for other hybrids; the output gate (≤ 1) shrinks the error | medium, grows with context |
| Gemma 4 | if it keeps Gemma 3's 5 local : 1 global with QK-norm | global layers; keep the 1K-token local windows exact | QK-norm → tight, cheap bounds | medium: 1 layer in 6 holds the long cache |

## 4. Fitting most open-source SOTA models

I don't have reliable details on "Kimi 3". Kimi's released lines are latent attention (K2) and a delta-rule hybrid (Kimi Linear), so it most likely falls into one of these families:

| Family | Examples | How the theory maps | Payoff |
|---|---|---|---|
| Full GQA/MQA | Llama 3.x, Mistral, Qwen3 / 2507, GLM-4.5/4.6, MiniMax-M2, ERNIE 4.5 | directly: one row per KV head per token | highest |
| Local + global, or chunked | Gemma 3; GPT-OSS (alternating 128-token banded and full layers, learned sinks); Llama 4 (chunked RoPE layers, a NoPE global layer every 4th) | global layers only. Sinks, NoPE, attention temperature and logit soft-capping are exact or 1-Lipschitz changes to the scores, so the bounds still hold. | medium |
| Latent attention (MLA) | DeepSeek-V3/R1/V3.1, Kimi K2, and Kimi 3 if it keeps MLA | a row is the token's 576-dim latent, shared by all heads, scored with absorbed queries. It behaves like one KV head for all 64–128 heads, so the bound must hold for every head. Values also come from the latent. | lower: the cache is already ~69 KB per token (DeepSeek-V3), against ~516 KB for Llama-3.1-405B, and the union over all heads is large |
| Trained sparse selection | DeepSeek-V3.2 (its indexer picks the top 2,048 tokens per query), NSA, MoBA | the selector already exists and is trained. Our addition is the certificate and the two tiers, turning a learned heuristic into a checked one. | depends on how often the indexer misses |
| Hybrid linear + softmax | Qwen3-Next / Qwen3.5, Kimi Linear (3 delta-rule : 1 MLA), MiniMax-M1 (7 linear : 1 softmax), Granite 4, Nemotron-H, Jamba | softmax layers only. Linear and SSM layers keep a fixed-size state: nothing to select, and no error is added. | grows with context; the softmax layers' cache dominates at 256K–1M tokens |

**Adapting to a new model** takes five steps:
1. Find the layers that use softmax attention over a growing cache.
2. Define a row as the model stores it: per KV head (GQA/MQA) or per token latent (MLA).
3. Define the query transform used for scoring: absorbed queries for MLA, plus temperature or sink terms where present.
4. Make each stored row the selection unit, with the bound required for every head that reads it.
5. Anchor τ to FP8's per-head error on that model, and measure attention concentration to forecast the savings.

**Testing:** the theorem depends on the architecture's structure, not its size. Small members of each family can validate the adapters cheaply:
- GPT-OSS-20B (local/global with sinks, 1 GPU);
- DeepSeek-V2-Lite or Moonlight-16B-A3B (MLA, 1 GPU);
- Gemma-3-12B (local/global, 1 GPU);
- Granite-4.0-H-Tiny or Kimi-Linear-48B-A3B (hybrids).

After that, the large SOTA models only need their savings measured.

If you'd like, I can write parts 1–2 into `plan.md` as R5's theory section: the lemmas, the controller, and the hypotheses R5's logs would test.