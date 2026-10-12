# Model families: the bits each kind of attention needs, and where the theory holds

*14h_theory, 2026-10-11. A CPU study of 21 open-weight models. Data and code are listed in §9.*

This note answers two questions about the two-tier KV-cache design studied in this folder:
1. **Which key and value bit widths does each kind of model need?** In particular, is it right that 6-bit keys with
   4-bit values (6/4) suit concentrated-attention models, and that 6/6 suits every model?
2. **Which architectures do the design's theorems cover**, current frontier models included?

It is self-contained: §1 defines the design, the metrics and the results it uses, and §2 says how the numbers were
produced.

## Summary

- **The theorems do not depend on the architecture, within softmax attention.** The run-time certificate (Theorem R)
  and the exact error formula (Theorem M) hold for every input, under three conditions (§1.3). They hold for any softmax
  attention over a per-layer cache of rows. That covers MHA, MQA, GQA, MLA, local:global layers, sinks, biases and gates,
  and the softmax layers of hybrids. The certificate was checked on all 21 models and the formula on 19.
- **The bit widths do depend on the model: no single configuration reaches FP8's per-head error everywhere.**
  - **6/4**, the design point of the confirmatory run R6, suits concentrated layers: 211 of 218 meet the target. But a
    per-layer choice is cheaper there: traffic 0.35–0.39 against 0.398.
  - **6/6** fixes most flat layers, but fails 29 of 313 layers. Three kinds fail:
    - noisy scores: all 12 GPT-OSS full-attention layers among them, which need 8-bit keys;
    - near-constant scores, which need 8-bit values;
    - flat layers where both errors are large.
- **Two numbers per layer, read from one full-precision pass, decide the bits.** Flatness sets how many bits a layer
  needs; the ratio of key to value sensitivity sets how to split them (§4).
- **These numbers describe layers, not models.** All 18 models in the bit study have at least one flat layer, and in
  every mostly-concentrated model the first one to three layers are flat. The best choice also moves with the input (§5).
- **The defensible claim:** "the certificate and the error formula hold for any softmax attention, under stated
  conditions; the formula chooses the bits per layer from a few calibration inputs, and the certificate checks every
  call" (§7).

## 1. Background

### 1.1 The design
- **Tier 1** holds every context row (key and value) in low-bit, randomized form:
  - each row is rotated by a fixed random orthogonal matrix R;
  - it is then quantized with subtractive dither: a seeded uniform offset is added before rounding and subtracted
    after, with a per-row step D = 2‖Rx‖_∞/(2^b − 1) at b bits;
  - each coordinate's error is then exactly uniform on [−D/2, D/2] and independent of the data (Schuchman's theorem).
- **Tier 2** holds the exact rows (FP16), for instance in host memory.
- **At each decode step:**
  - the rows a question-time vote selects, a fraction f of the context (1/8 here), are read exactly from tier 2;
  - every other row is read from tier 1, with its weight divided by the dither's known mean effect, which makes the
    estimate unbiased (the "unbiased tail");
  - nothing is evicted, and the most recent rows and the question stay exact.
- **The vote** ranks rows on a separate 4-bit store that never sees tier 1's dither. Condition R1 in §1.3 needs this.
- **Notation.** k/v is tier 1's key width / value width, so 6/4 means 6-bit keys and 4-bit values. R6, the
  confirmatory run, tests 6/4 with f = 1/8 (the arm `dtail6ux_v4`).

### 1.2 Metrics
- **Traffic**: the bits read per decode step, as a fraction of reading every row's keys and values in FP16:
  (1 − f)(k + v)/32 + f. It counts element bits only and leaves out the question-time vote, as R6's traffic model does.
  With f = 1/8 (8/8 with f = 1/4 costs 0.625):

  | configuration | 4/4 | 6/4 or 4/6 | 6/6 or 8/4 | 8/6 | 8/8 | FP8 keys and values |
  |---|---|---|---|---|---|---|
  | traffic | 0.344 | 0.398 | 0.453 | 0.508 | 0.563 | 0.500 |

- **Head-step**: one query head at one decode position.
- **Error ratio**: a head-step's RMS attention-output error over the dither, divided by FP8 KV's error at the same
  head-step. FP8 here is E4M3 keys and values with one scale per KV head. Below 1, the design is more accurate than FP8
  there.
- **Targets.** A layer meets the *median target* if the ratio is ≤ 1 at its median head-step. It meets the stricter
  *p90 target* if the ratio is ≤ 1 at 90% of its head-steps. Tables use p90 unless they say otherwise.

### 1.3 The results this note uses (`theory.md`)
Setting: one head attends over cached rows i.
- Its weights are w_i = e^{s_i}, with s_i = scaling·⟨q, k_i⟩ plus any known bias; Z = Σ_i w_i and o = Σ_i w_i v_i / Z.
- T is the set of rows read from tier 1.
- The key dither multiplies row i's weight by a random factor Z_i, and the value dither adds an error e_i to its value.

**Theorem R, the certificate (§6).** At every head-step, the attention error of the two-tier read is at most a bound
computed at run time from observed quantities. This holds with probability ≥ 1 − δ over the dither, for every input,
under three conditions:
- **(R1)** the rows read exactly are chosen without looking at tier 1's dither;
- **(R2)** the queries do not depend on this layer's dither. A separate dither per layer meets this in prefill and
  teacher-forced passes. In free-running decode it holds up to the first token that differs from the exact model's
  (Lemma C and Theorem DD, §11);
- the seed is independent of the input.

**Theorem M, the exact error (§10).** For any cache, query and read set chosen as in (R1),

  E‖N‖² = Σ_{i∈T} w_i²·[Var(Z_i)·‖v_i − o‖² + E(Z_i²)·E‖e_i‖²] + ‖Σ_{i∈T} w_i(1 − E Z_i)(v_i − o)‖²,

with E‖o − õ‖² ≈ E‖N‖²/Z² to first order.
- The terms: the first is the **key term** (score noise) and the second the **value term**. The last (bias) vanishes
  for the unbiased tail.
- Every moment is a closed form of the layer's own queries, keys and values. So the error comes from one
  full-precision pass, with no dither draws.
- MLA, where one latent dither makes both errors, has a coupled form.

**Corollary M-B, choosing the bits (§10b).** For small score noise,

  MSE(k, v) ≈ [a·c(k) + b·c(v)] / Z²,  with c(x) = (2^x − 1)^(−2).
- The **key sensitivity** is a = Σ_T w_i²·(scaling²‖q‖²/12)·(2‖R k_i‖_∞)²·‖v_i − o‖².
- The **value sensitivity** is b = Σ_T w_i²·(d_v/12)·(2‖R_v v_i‖_∞)².
- **Fewest bits for a target ε:** k* = log₂(1 + √(2a/(εZ²))) and v* = log₂(1 + √(2b/(εZ²))). Each tier needs one more
  bit per 4× of its own sensitivity.
- **The split:** k* − v* ≈ ½·log₂(a/b). At equal widths, a/(a + b) is the key term's share of the error.

### 1.4 Two numbers that classify a layer
1. **Flatness: the unread second-moment share.** This is the fraction of Σ_i w_i²‖v_i − o‖² that falls on the rows left
   to tier 1 after the exact read.
   - Both a and b are sums over those rows, so this share sets the overall level of the error, and with it how many
     bits the layer needs.
   - It is not the unread *mass*. On Llama-3.2-1B's layer 1 the exact read leaves 2.4% of the mass to tier 1, but only
     0.18% of the second moment.
   - Classes: **concentrated** ≤ 1%, **flat** ≥ 10%, **between** 1–10%.
   - The score noise σ, the standard deviation of a tier-1 score error (∝ scaling·‖q‖·D_k), scales a. Noisy scores make
     the key term large whatever the flatness.
2. **The split** k* − v* = ½·log₂(a/b), computed exactly from the key term's share of the error at 6/6. It says which
   tier should get the bits:

   | class | k* − v* | key term's share at equal widths |
   |---|---|---|
   | key-heavy | ≥ 1.5 | ≥ 89% |
   | key-leaning | 0.5 to 1.5 | 67–89% |
   | balanced | −0.5 to 0.5 | 33–67% |
   | value-leaning | ≤ −0.5 | ≤ 33% |

## 2. How the numbers were produced
- **Models.** The bit study uses the 18 models in the table below, whose caches are GQA or MHA. Three MLA models entered
  the theorem checks but not the bit study: DeepSeek-V2-Lite, Moonlight-16B-A3B and Kimi-Linear-48B-A3B. Together these
  are every model in the local Hugging Face cache with weights and a softmax KV cache.
- **Input.** 1,600 tokens of a PG-19 book passage. Six models were also run on three LongBench v2 contexts: a code
  repository, a table, and an agent's history log (§5).
- **Per layer:**
  - the first 1,552 rows are the cache, and the last 48 positions stay exact;
  - of those 48, the first 32 cast the vote: the top 1/8 of rows by summed attention, on a separate 4-bit store;
  - the last 16 are the measured queries.
- **Prediction.** Theorem M's exact moments, per head-step and without dither draws (`layermap_s1h5.py --grid`). They
  are evaluated for every (k, v) in {4, 5, 6, 8}² and f in {1/8, 1/4}.
- **Allocation.** Per layer, the cheapest configuration that meets the target. A layer that no configuration meets is
  "unmet" and counted at 8/8 with a 1/4 read (0.625).
- **Checks (Monte Carlo, 200 dither draws).** Measured over predicted MSE is 0.98–1.005 at 5/5, 6/6, 8/6 and 6/8. The
  seven layers checked are of different kinds: Llama-3.2-1B L1, Gemma 3 L5, GPT-OSS L1, Granite-4.2 L1, Qwen3-1.7B L1,
  MiniMax-M2 L1 and Qwen3-30B L1.
  - At 4-bit keys on the two noisiest of them (GPT-OSS L1, Qwen3-1.7B L1) the ratio is 0.92: the formula overpredicts,
    which is the safe direction.
- **A correction made while writing this note.** Granite-4.2's layers 4 and 6 have near-zero scores (|s| ≈ 10⁻⁹).
  - There, `theory3_s1h5._log_sinhc` lost every digit and added a spurious key term, so the first grid called both
    layers unmet even at 8/8.
  - The fix (2026-10-11) uses a series below |x| = 0.1, checked against a 100-digit reference in `test_theory3_s1h5.py`.
  - With it, Theorem M matches 200 dither draws on both layers within 1%, and both meet the target at 6/6. No other
    model had scores small enough to be affected.

| model | attention (query : KV heads) | layers in the bit study |
|---|---|---|
| Llama-3.2-1B | GQA 32:8 | all 16 |
| Llama-3.1-8B (R6) | GQA 32:8 | all 32 |
| Llama-3.3-70B | GQA 64:8 | 0–11 of 80 |
| Mistral-7B-v0.3 | GQA 32:8 | all 32 |
| Mistral-Small-3.2-24B (the text model of a VLM) | GQA 32:8 | 0–19 of 40 |
| Qwen1.5-0.5B | MHA, QKV bias | all 24 |
| Qwen1.5-MoE-A2.7B | MHA, QKV bias, MoE | all 24 |
| Qwen3-1.7B | GQA 16:8, QK-norm | all 28 |
| Qwen3-30B-A3B (R6) | GQA 32:4, QK-norm, MoE | all 48 |
| GLM-4.5-Air | GQA 96:8, partial RoPE, QKV bias, MoE | 0–8 of 46 |
| MiniMax-M2 | GQA 48:8, QK-norm, MoE, FP8 weights | 0–5 of 62 |
| Granite-4.2-30B | GQA 32:8, attention multiplier 1/128 | 0–15 of 64 |
| GPT-OSS-20B | GQA 64:8, learned sinks; banded (128 rows) and full layers alternate | the 12 full layers |
| Gemma-3-12B | GQA 16:8, QK-norm; 5 local (1,024 rows) : 1 global | the 8 global layers |
| Gemma-4-31B | 5 local (1,024) : 1 global; global layers GQA 32:4, head dim 512, keys = values | 6 of the 10 global layers |
| Qwen3.5-27B | 3 Gated DeltaNet : 1 gated attention; GQA 24:4, head dim 256 | 8 of the 16 attention layers |
| Qwen3.8-27B | as Qwen3.5, with partial RoPE and a swish output gate | 8 of the 16 attention layers |
| Granite-4.0-H-Tiny | 9 Mamba-2 : 1 attention (no position embedding); GQA 12:4 | all 4 attention layers |

## 3. Correction: "6/4 for concentrated attention, 6/6 for every model"

A natural summary of the first architecture study (`findings/findings_architectures.md`) was:
- 6/4 is right for concentrated-attention models and wrong for flat-attention ones;
- 6/6 works for both.

The first half holds, roughly. The second does not.

**What holds.**
- 6/4 keeps concentrated layers at or below FP8's per-head error. Across the 18 models, 211 of the 218 concentrated
  layers (≤ 1% of the second moment unread) meet the p90 target at 6/4. Five of the seven exceptions are
  Qwen3-30B-A3B layers with noisy scores (σ = 0.13–0.54 at 6 bits).
- With 4-bit values, flat layers stay above FP8's error at any key width: 6/4 meets 4 of the 69 flat layers, and 8/4
  meets 13.
- 6-bit values fix most of them: 6/6 meets 46 of the 69, and every mapped global layer of Gemma 3, Gemma 4 and
  Qwen3.5.

**What does not hold.**
1. **6/6 is not universal.** It misses the p90 target on 29 of 313 layers, of three kinds:

   | kind | layers | why 6/6 fails | what Theorem M picks instead |
   |---|---|---|---|
   | noisy scores (σ = 0.14–0.73 at 6 bits) | all 12 GPT-OSS full layers; Qwen3-30B-A3B L0, L1, L3, L8, L16; Qwen3-1.7B L0–L2; Qwen1.5-0.5B L0 (21) | the key term is ≥ 95% of the error at 6/6, and value bits do not touch it | 8-bit keys: 8/4 (14), 8/5 (6, one with a 1/4 read); Qwen3-1.7B L0 is unmet even at 8/8 with a 1/4 read |
   | near-constant scores, very flat (59–85% of the second moment unread) | Granite-4.2 L2, L3, L8; Granite-4.0-H L5 (4) | the value term is ≈ 100% of the error, and 6-bit values are not enough | 8-bit values: 4/8 (4) |
   | flat, with both terms large (30–52% unread) | MiniMax-M2 L1, L2, L4; Qwen3.8 L7 (4) | the key term is 63–77% of the error at 6/6, and the level is high | one tier to 8 bits: 8/6, 6/8, 8/5, 6/8 |

   Noisy scores are independent of flatness: Qwen3-30B-A3B's L3, L8 and L16 are concentrated (≤ 0.5% unread) and still
   need 8-bit keys.
2. **6/4 is not the right choice for concentrated models either.**
   - Most of their layers need less: 4/4 or 5/4 meets the target in 191 of the 218 concentrated layers.
   - A few need more: the noisy-score layers, and the flat first layers.
   - A per-layer choice costs 0.35–0.39, against uniform 6/4's 0.398, and uniform 6/4 misses 0–8 layers per model.
3. **"Concentrated" and "flat" describe layers, not models** (§5).
4. **Flatness is one of two axes.** The other is the split between key and value sensitivity (§1.4), and the score
   noise σ enters through it. Both come from one full-precision pass.

## 4. Families by attention statistics: which bits

### 4.1 Families of layers
313 layers of 18 models, PG-19 input, p90 target; f = 1/8 unless noted.

| family | flatness | split k* − v* | σ at 6-bit keys | layers | Theorem M's choice (layers) | mean traffic | meet 6/4 | meet 6/6 |
|---|---|---|---|---|---|---|---|---|
| **A.** concentrated, key-leaning | ≤ 1% | 0.5 to 1.5 | 0.04–0.08 | 125 | 4/4 (81), 5/4 (33), 5/5 (7) | 0.356 | 124 | 125 |
| **B.** concentrated, key-heavy | ≤ 1% | ≥ 1.5 | 0.07–0.54 | 93 | 5/4 (40), 4/4 (32), 5/5 and 6/4 (6 each), 8/4 (3) | 0.370 | 87 | 90 |
| **C.** flat, key-heavy (noisy scores) | ≥ 10% | ≥ 1.5 | 0.09–0.73 | 17 | 8/4 (8), 8/5 (6), unmet (1) | 0.469 | 2 | 2 |
| **D.** flat, key-leaning or balanced | ≥ 10% | −0.5 to 1.5 | 0.02–0.07 | 32 | 6/6 (19), 6/5 (4), 5/6 (3), one tier at 8 bits (4) | 0.450 | 2 | 28 |
| **E.** flat, value-leaning | ≥ 10% | ≤ −0.5 | 0–0.03 | 20 | 4/6 (7), 6/6 (7), 4/8 (4) | 0.430 | 0 | 16 |
| between | 1–10% | mixed | 0.04–0.22 | 26 | 5/4 (8), 5/5 (4), 6/5 (4), 8/4 (3) | 0.405 | 14 | 23 |

No concentrated layer is balanced or value-leaning. Sharp attention comes with large scores, and the score noise per
key bit grows with the scores' scale (σ ∝ scaling·‖q‖·‖k‖_∞), so the key term weighs more there.

Where each family occurs:
- **A:** most layers of Llama-3.1-8B (25 of 32), Llama-3.3-70B (9 of 12), Mistral-7B (28 of 32), Mistral-Small-3.2
  (14 of 20) and Qwen1.5-MoE (16 of 24). Also the quieter layers of Qwen3 and GLM-4.5-Air, and the last global layers of
  Qwen3.5/3.8.
- **B:** most layers of Llama-3.2-1B (15 of 16), Qwen1.5-0.5B (16 of 24), Qwen3-1.7B (16 of 28) and Qwen3-30B-A3B
  (29 of 48).
- **C:** 9 of GPT-OSS's 12 full layers, and the first two or three layers of Qwen1.5-0.5B, Qwen3-1.7B and Qwen3-30B-A3B.
- **D:**
  - all six mapped MiniMax-M2 layers;
  - the early global layers of Gemma 4 and of Qwen3.5/3.8 (4 each);
  - Granite-4.2 L10–L15 and Gemma 3's L5;
  - the first layer of Llama-3.2-1B, Llama-3.1-8B and Mistral-7B, and the first two of Qwen1.5-MoE and GLM-4.5-Air.
- **E:**
  - Gemma 3's other seven global layers;
  - Granite-4.2 L0–L9 and Granite-4.0-H L5;
  - the first layer of Llama-3.3-70B and of Mistral-Small-3.2.

Why each family needs what it needs:
- **A and B:** the exact read takes ≥ 99% of the second moment, so tier 1's error is small at any width. Keys need about
  one to two bits more than values: a key's quantization error reaches the weights multiplied by scaling·‖q‖, which
  is large where attention is sharp. B's noisiest layers (Qwen3's QK-norm layers, σ up to 0.54) need 6/4 to 8/4.
- **C:** the attention is flat *and* the scores are noisy. GPT-OSS has the highest typical σ of all 18 models (0.15–0.23
  at 6 bits), and its key term is 97–99% of the error at 6/6. Only key bits help.
- **D:** 10–65% of the second moment is unread, so both terms are large. Two more bits on the values (6/6) is usually
  enough; the flattest layers need one tier at 8 bits.
- **E:** the scores barely vary, so the key term is negligible and key bits are wasted. Granite-4.2's attention
  multiplier is 1/128 rather than 1/√128, which shrinks its scores 11-fold, and in L4/L6 they are ≈ 0. Values take the
  bits: 4/6 or 4/8, or 6/6.

### 4.2 Families of models
A model is best described by its majority family and its exceptions (counts in §4.3).

| model family | models | flatness (median over layers) | per-layer allocation (p90) | uniform 6/4 misses | uniform 6/6 misses |
|---|---|---|---|---|---|
| concentrated, key-leaning (mostly A) | Llama-3.1-8B (R6), Llama-3.3-70B, Mistral-7B, Mistral-Small-3.2, Qwen1.5-MoE, GLM-4.5-Air | 0.05–0.72% | 0.352–0.387 | 0–4 layers | none |
| concentrated, key-heavy (mostly B) | Llama-3.2-1B, Qwen1.5-0.5B, Qwen3-1.7B, Qwen3-30B-A3B (R6) | 0.11–0.33% | 0.368–0.388 | 1–8 layers | 0–5 layers |
| flat, noisy scores (C) | GPT-OSS-20B | 15% | 0.458 (8/4 in 10 of 12) | 12 of 12 | 12 of 12 |
| flat, balanced or key-leaning (D) | Gemma-4-31B, Qwen3.5-27B, Qwen3.8-27B, MiniMax-M2 | 12–49% | 0.429–0.471 | 6 layers each | 0–3 layers |
| flat, value-leaning (E) | Gemma-3-12B, Granite-4.2-30B | 62–64% | 0.429–0.453 | every layer | 0–3 layers |
| hybrid with few attention layers | Granite-4.0-H-Tiny (one layer each of A, E and two between) | 2.6% | 0.398 | 1 of 4 | 1 of 4 |

R6's two models fall in different families: Llama-3.1-8B is concentrated and key-leaning, and Qwen3-30B-A3B is
concentrated and key-heavy, with five noisy-score layers that 6/6 misses.

### 4.3 Per model
**Allocation and uniform configurations** (p90 target; f = 1/8). The traffic columns give the allocation's mean
traffic under the median and the p90 target. The middle columns count the layers each uniform configuration misses.

| model | layers | traffic: median / p90 target | 4/4 | 6/4 | 4/6 | 6/6 | 8/4 | 8/6 | 8/8 | most chosen (p90) |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 16 | 0.345 / 0.368 | 12 | 1 | 11 | 0 | 1 | 0 | 0 | 5/4 ×10, 4/4 ×4 |
| Llama-3.1-8B (R6) | 32 | 0.345 / 0.354 | 9 | 1 | 8 | 0 | 0 | 0 | 0 | 4/4 ×23, 5/4 ×6 |
| Llama-3.3-70B | 12 | 0.344 / 0.387 | 12 | 1 | 11 | 0 | 1 | 0 | 0 | 5/5 ×6, 5/4 ×5 |
| Mistral-7B | 32 | 0.344 / 0.362 | 17 | 0 | 16 | 0 | 0 | 0 | 0 | 4/4 ×15, 5/4 ×13 |
| Mistral-Small-3.2 | 20 | 0.346 / 0.363 | 8 | 4 | 6 | 0 | 4 | 0 | 0 | 4/4 ×12 |
| Qwen1.5-0.5B | 24 | 0.347 / 0.377 | 21 | 1 | 21 | 1 | 0 | 0 | 0 | 5/4 ×15 |
| Qwen1.5-MoE | 24 | 0.346 / 0.352 | 3 | 1 | 3 | 0 | 1 | 0 | 0 | 4/4 ×21 |
| Qwen3-1.7B | 28 | 0.357 / 0.388 (1 unmet) | 21 | 3 | 20 | 3 | 3 | 2 | 1 | 5/4 ×13, 4/4 ×7 |
| Qwen3-30B-A3B (R6) | 48 | 0.350 / 0.374 | 23 | 8 | 23 | 5 | 2 | 0 | 0 | 4/4 ×25, 5/4 ×9, 6/4 ×4, 8/4 ×3 |
| GLM-4.5-Air | 9 | 0.359 / 0.380 | 5 | 3 | 5 | 0 | 2 | 0 | 0 | 4/4 ×4, 5/4 ×2 |
| Granite-4.0-H-Tiny | 4 | 0.357 / 0.398 | 4 | 1 | 4 | 1 | 1 | 1 | 0 | 5/4 ×2 |
| GPT-OSS-20B | 12 | 0.398 / 0.458 | 12 | 12 | 12 | 12 | 2 | 0 | 0 | 8/4 ×10, 8/5 ×2 |
| Gemma-4-31B | 6 | 0.376 / 0.435 | 6 | 6 | 6 | 0 | 6 | 0 | 0 | 5/6 ×4, 6/6 ×2 |
| Qwen3.5-27B | 8 | 0.381 / 0.429 | 8 | 6 | 8 | 0 | 6 | 0 | 0 | 6/6 ×5, 5/4 ×2 |
| Qwen3.8-27B | 8 | 0.381 / 0.436 | 8 | 6 | 8 | 1 | 6 | 0 | 0 | 6/6 ×4, 5/4 ×2 |
| MiniMax-M2 | 6 | 0.417 / 0.471 | 6 | 6 | 6 | 3 | 6 | 0 | 0 | 6/6 ×2 |
| Gemma-3-12B | 8 | 0.453 / 0.453 | 8 | 8 | 8 | 0 | 8 | 0 | 0 | 6/6 ×8 |
| Granite-4.2-30B | 16 | 0.400 / 0.429 | 16 | 16 | 10 | 3 | 16 | 3 | 0 | 4/6 ×6, 6/6 ×5, 4/8 ×3 |

**Statistics.** Flatness is the median over layers, with the range in brackets. The split column gives the median with
the 10th–90th percentile. The family column gives each model's layer counts.

| model | flatness: median (range) | σ at 6 bits | split k* − v* | layers per family |
|---|---|---|---|---|
| Llama-3.2-1B | 0.11% (0.06–20%) | 0.07–0.11 | +1.99 (+1.76, +2.11) | B 15, D 1 (L0) |
| Llama-3.1-8B (R6) | 0.05% (0.01–26%) | 0.04–0.08 | +1.42 (+1.31, +1.59) | A 25, B 6, D 1 (L0) |
| Llama-3.3-70B | 0.72% (0.20–66%) | 0.01–0.09 | +1.41 (+1.02, +1.70) | A 9, B 2, E 1 (L0) |
| Mistral-7B | 0.33% (0.07–36%) | 0.03–0.07 | +1.20 (+1.02, +1.39) | A 28, D 1 (L0), between 3 |
| Mistral-Small-3.2 | 0.28% (0.08–41%) | 0.02–0.05 | +0.79 (+0.68, +0.92) | A 14, E 1 (L0), between 5 |
| Qwen1.5-0.5B | 0.33% (0.05–43%) | 0.09–0.73 | +2.14 (+1.96, +2.33) | B 16, C 3 (L0–L2), between 5 |
| Qwen1.5-MoE | 0.07% (0.02–19%) | 0.04–0.08 | +1.37 (+1.09, +1.52) | A 16, B 6, D 2 (L0–L1) |
| Qwen3-1.7B | 0.24% (0.04–47%) | 0.06–0.50 | +1.92 (+1.36, +3.49) | A 8, B 16, C 3 (L0–L2), between 1 |
| Qwen3-30B-A3B (R6) | 0.11% (0.02–33%) | 0.04–0.54 | +1.77 (+1.14, +3.02) | A 15, B 29, C 2 (L0–L1), between 2 |
| GLM-4.5-Air | 0.23% (0.12–65%) | 0.02–0.08 | +1.46 (−0.31, +1.61) | A 4, B 3, D 2 (L0–L1) |
| Granite-4.0-H-Tiny | 2.6% (0.6–66%) | 0.00–0.08 | +1.31 (−4.11, +1.58) | A 1, E 1 (L5), between 2 |
| GPT-OSS-20B | 15% (2.8–33%) | 0.15–0.23 | +2.69 (+2.47, +3.00) | C 9, between 3 |
| Gemma-4-31B | 12% (4.8–34%) | 0.04 | +0.32 (+0.10, +0.42) | D 4, between 2 |
| Qwen3.5-27B | 14% (0.20–30%) | 0.04–0.06 | +0.76 (+0.27, +1.03) | A 3, D 4, between 1 |
| Qwen3.8-27B | 15% (0.30–30%) | 0.04–0.06 | +0.82 (+0.34, +1.13) | A 2, D 4, between 2 |
| MiniMax-M2 | 49% (40–60%) | 0.03–0.05 | +0.55 (−0.32, +0.88) | D 6 |
| Gemma-3-12B | 64% (62–67%) | 0.03 | −0.59 (−0.66, −0.43) | D 1 (L5), E 7 |
| Granite-4.2-30B | 62% (12–85%) | 0–0.06 | −2.32 (below −5, +0.81) | D 6 (L10–L15), E 10 (L0–L9) |

A split below −5 means the key term is ≈ 0 (near-constant scores).

## 5. Layers, not models; and inputs

**Every model mixes kinds of layers.**
- All 18 models have at least one flat layer.
- In each of the ten mostly-concentrated models, the first one to three layers are flat (14–66% unread). They account
  for 13 of the 23 places where uniform 6/4 misses on those models; Qwen3-30B-A3B's noisy-score layers account for
  5 more.
- Conversely, the last two mapped global layers of Qwen3.5 and Qwen3.8 are concentrated (≤ 0.3%) and take 5/4.
- The split also spans classes within a model: Qwen3-30B-A3B runs from +0.8 to +4.3, Granite-4.2 from a key term of ≈ 0
  to +1.15, and GLM-4.5-Air from −0.3 to +1.6.

**The best choice moves with the input** (p90 target):

| model | layers compared | traffic: PG-19 / code / table / agent log | per-layer maximum over the four | layers where PG-19's choice misses on code / table / agent |
|---|---|---|---|---|
| Mistral-7B | 16 | 0.369 / 0.390 / 0.344 / 0.349 | 0.393 | 9 / 0 / 0 |
| Llama-3.1-8B (R6) | 16 | 0.364 / 0.381 / 0.347 / 0.373 | 0.386 | 9 / 0 / 9 |
| Qwen3-30B-A3B (R6) | 12 | 0.410 / 0.412 / 0.389 / 0.380 | 0.421 | 3 / 2 / 0 |
| GPT-OSS-20B | 12 | 0.458 / 0.480 / 0.453 / 0.453 | 0.480 | 10 / 0 / 1 |
| Gemma-3-12B | 8 | 0.453 / 0.453 / 0.429 / 0.453 | 0.453 | 0 / 0 / 0 |
| Granite-4.2-30B | 16 | 0.429 / 0.431 / 0.381 / 0.386 | 0.434 | 2 / 0 / 0 |

These rows cover the layers mapped on all four inputs (Mistral-7B and Llama-3.1-8B: layers 0–15; Qwen3-30B-A3B: 0–11;
the others: every mapped layer), so their traffic differs from §4.3's.

Code is the hardest of the four inputs. A per-layer choice calibrated on all four (the per-layer maximum) costs
0.39–0.48, still below FP8's 0.50. What calibration misses on a new input is caught at run time by the certificate,
which checks every head-step.

**So the unit of choice is the layer, or the head, on several calibration inputs.** The family labels of §4.2 describe
where a model's layers mostly sit; they do not replace the per-layer map.

## 6. Families by attention mechanism: where the theorems hold

The theorems need softmax attention over a per-layer cache of rows, with the three conditions of §1.3. Positional
schemes, normalizations, biases and gates do not matter, because the scores stay bilinear in a stored row. Sinks enter
as exact fixed rows.

| mechanism | examples | status | why |
|---|---|---|---|
| full softmax attention over per-token keys and values: MHA, MQA, GQA, with RoPE, partial RoPE or none, QK-norm, QKV bias, attention multipliers, output gates, learned sinks | Llama 3.x, Mistral 7B and Small 3.2, Qwen1.5/2/2.5/3, GLM-4.5, MiniMax-M2, GPT-OSS, Granite-4.2 | **verified** on 12 such models and GPT-OSS's full layers (on CPU; Llama-3.1-8B's certificate on GPU at 128K, in R5.4) | the cache is rows; the scores are bilinear in the stored key |
| latent cache (MLA) | DeepSeek-V2/V3/R1, Kimi K2, Moonlight, the MLA layers of Kimi-Linear | **verified** on 3 (Theorem R; Theorem M in its coupled form, coupling term ≤ 0.1% of the error) | one dithered latent feeds both the scores and the values; the coupled form accounts for it |
| local : global (sliding or chunked windows plus full layers) | Gemma 2/3/4, GPT-OSS, Llama 4, Command A | **verified** on the global layers of Gemma 3, Gemma 4 and GPT-OSS | the windows (128 to a few thousand rows) do not grow with the context and stay exact; the global layers are ordinary softmax caches |
| hybrids: recurrent or linear layers plus some softmax layers | tested: Granite-4.0-H, Qwen3.5, Qwen3.8, Kimi-Linear. Untested: Qwen3-Next, Nemotron-H, Jamba, MiniMax-M1 | softmax layers **verified** on the 4 tested | the recurrent state has a fixed size and is not a cache of rows: out of scope, and there is nothing to compress |
| sparse attention with a learned selector | DeepSeek-V3.2 (DSA), NSA, MoBA | **covered if** the selector reads a store other than the dithered tier; untested (no weights here) | R1: the rows read must not depend on the dither. A selector that scores the dithered keys breaks it; one with its own index keys, kept exact or separately quantized, does not |
| cross-layer KV sharing | Gemma 3n, YOCO, CLA | **needs an extension** | R2: a later layer reads the same dithered rows with a query that depends on their dither, through the earlier layers' outputs |
| non-softmax or differential attention | sigmoid attention, Differential Transformer | **needs new formulas** | the identities use the softmax normalizer. Differential attention subtracts two softmax maps over the same dithered keys, so their errors are correlated |
| closed frontier models | GPT-5, Claude, Gemini | **unknown** | the architectures are not public. The theory applies layer by layer to whichever of the mechanisms above they use |

Mechanisms outside the 21 tested models are described from their papers and model cards, not checked here.

So the theory transfers to every mainstream open architecture whose attention is a softmax over a per-layer KV cache:
- What transfers is validity. The certificate holds and the error formula is exact on any such layer.
- What does not transfer is the design point: §3–§5 show that the bits must be read off each model's layers.
- Two growing families need work first: selector-based sparse attention, where the selector must not read the dithered
  tier, and cross-layer KV sharing.

## 7. Is "the certificate and the error formula hold for any softmax attention; the formula picks the bits" true?

As a slogan it is too broad. The version this folder supports (`CLAIMS_LEDGER.md` T8):

> For any softmax attention over a per-layer cache of rows, Theorem R (the certificate) and Theorem M (the exact error
> formula) hold for every input, given three conditions: a subtractive-dithered tier, exact-read sets chosen without
> looking at its dither, and a separate dither per layer. That covers MHA, MQA, GQA, MLA, local:global layers, sinks,
> biases, gates, and the softmax layers of hybrids. Theorem M predicts the error of every key width, value width and
> exact-read set from one full-precision pass, so the bits are chosen per layer (Corollary M-B) and checked per call by
> Theorem R.

Two qualifications go with it:
- **"Picks the bits" is a per-input prediction.** A deployment calibrates the choice on several inputs and relies on the
  run-time certificate for the rest (§5).
- **The target is attention-level.** It is FP8's per-head error, not end-to-end KL or accuracy. The bridge from one to
  the other is open (`STATUS.md` step 7), and R6 measures end-to-end quality directly. FP8's per-head error is computed
  from exact data, so a deployed allocator needs a target it can compute (`STATUS.md` step 6).

Not covered until extended: a selector that reads the dithered tier, cross-layer KV sharing, non-softmax or
differential attention, and recurrent states (§6).

## 8. Limits
- **CPU only, 1,600-token inputs.** R6 runs at 32K and 128K. Flatness and σ can change with length; the long-context
  check is `STATUS.md` step 3.
- **Partial depth for seven models** (§2's table). Their families are those of the mapped layers.
- **Inputs:** one for 12 models, and four for 6.
- **The layer is the unit studied.** A per-head allocation would be cheaper and is not evaluated here.
- **MLA:** the theorems were checked on MLA layers, but the bit grid was not run on them (`layermap_s1h5.py` takes GQA
  views).
- **The prediction's accuracy.** The grid uses Theorem M's exact moments with the first-order normalizer, which is
  accurate to 2% wherever keys have at least 5 bits. At 4-bit keys on noisy layers it overpredicts, which is
  conservative.
- **Other architectures:** statements about architectures beyond the 21 tested come from public descriptions.

## 9. Sources
- **Data (all in `findings/`):**
  - `findings_bits.json`: the grid per model, input and layer, the off-diagonal Monte Carlo checks, and the
    near-zero-score checks (`montecarlo_near_zero_scores`);
  - `findings_bits_alloc.json`: the allocations;
  - `findings_bits.md`: the bit study's write-up;
  - `findings_architectures.md` and `.json`: validity on the 21 models;
  - `layermap_<model>.json`: the per-layer maps;
  - `findings_mla_m.json`: Theorem M for MLA.
- **Theory:** `theory.md` §1 (setting), §6 (Theorem R), §10 (Theorem M), §10b (Corollary M-B), §11 (Lemma C, Theorem DD).
- **Code:**
  - `layermap_s1h5.py --grid` makes the grid; `H0_TEXT=<file>` switches the input text;
  - `theory3_s1h5.py` has `second_moment` and `exact_coin_moments`;
  - `adapters_s1h5.py` has `MODELS` and the loaders.

  The allocation rule is the one in §2; it reproduces `findings_bits_alloc.json` from `findings_bits.json`.
- **Claims:** `CLAIMS_LEDGER.md` T8 and §6.
