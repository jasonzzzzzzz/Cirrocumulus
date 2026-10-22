# The theorems on Qwen: Qwen3-1.7B and R6's Qwen3-30B-A3B (CPU, 2026-10-10)

> **Correction (2026-10-12):** every Gemma-3-12B result before that date came from a randomly initialized model (a loader bug; `MODEL_FAMILY.md`'s correction note). Gemma 3 is not flat at 1,600 tokens: L5 leaves 7.6% of the second moment unread, not 66%.

Every result so far was CPU-validated on Llama-3.2-1B (plus single layers of Gemma 3 and GPT-OSS). This checks them on
Qwen3, the family of R6's second model. Qwen3's attention differs from Llama's in having QK-norm (an RMSNorm with
learned scales on q and k).

Models:
- **Qwen3-1.7B**: 28 layers; 16 query heads on 8 KV heads; head dim 128.
- **Qwen3-30B-A3B-Instruct-2507**, R6's model: 48 layers; 32 query heads on 4 KV heads; head dim 128; MoE.

Code:
- `../theory5_s1h5.py` (`compare`, `certread`) and `../exp4bit_s1h5.py` (`evaluate`: Theorem M, 200 draws);
- `../adapters_s1h5.py --model`: the architecture checks;
- `../layermap_s1h5.py`, new: Theorem M's per-layer map from one full-precision pass.

Data: `findings_qwen.json`, `layermap_{llama1b,llama8b,qwen3_1p7b,qwen3_30b}.json`, and the Qwen entries of
`findings_4bit.json` (Theorem M).

Setup: part 8's. The vote reads 1/8 of the context exactly (a Lloyd-Max 4-bit sketch, 32 voter queries); 16 later
queries per head are measured; values are 4-bit dithered; δ = 10⁻³ per head-step; 1,600 tokens of a PG-19 passage
(8,192 for two Qwen3-1.7B views). "Part 8" is the tilted tail, the HKZ key sum with the exact-law proxy, and remedy 1.

## 1. Validity: every theorem holds on Qwen

- **The architecture checks** (`adapters_s1h5.py`, layers 0–3 of both models):
  - each view reconstructs the layer's output to ≤ 3.4·10⁻⁶;
  - Lemmas 2 and 4 hold;
  - Theorem R (v1) covers every draw.
- **Part 8 and the certified read:** coverage is 1.000 in every cell below: 12 views, 3 key widths, 3 draws, plus the
  certified reads.
- **Theorem M:** the measured MSE is within 1% of the prediction on every view at 6- and 8-bit keys (0.990–1.004).
  At 4-bit keys it is within 9% where σ ≤ 1.1, but **the first-order step fails in the two layers with σ ≈ 2**:
  - Qwen3-1.7B L0 (σ = 2.09): measured / predicted 0.20;
  - R6's L3 (σ = 2.27): 0.31.

  The theorem's identity and its bound E‖N‖²/M² still hold. Its "≈ E‖N‖²/Z²" reading replaces the random normaliser Z̃
  by Z, and at σ ≈ 2 Z̃ swings far enough that this overpredicts the error. The prediction is conservative there.

## 2. What is different about Qwen: the score noise varies by layer

Theorem M's per-layer map (`layermap_s1h5.py`; exact, no sampling) on every layer of four models. σ is the score
noise of a dithered key tier (median over unread rows). "Tail / FP8" is the tail design's predicted RMS attention
error over FP8's actual error, median over head-steps.

| model | layers | σ at 6-bit keys | layers with σ > 0.6 at 4 bits | tail / FP8 at 6 bits: median, max | layers where tail > FP8 at 4 / 6 / 8 bits | dense 4-bit tier / FP8 |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | 16 | 0.067–0.108 | none | 0.12, 0.58 (L0) | L0 / none / none | 3.1–4.8 |
| **Llama-3.1-8B (R6)** | 32 | 0.043–0.082 | none | 0.06, 0.61 (L0) | L0 / none / none | 1.8–5.8 |
| Qwen3-1.7B | 28 | 0.059–0.498 | 12 layers (L0–L7, L9, L10, L12, L15) | 0.10, 3.04 (L0) | L0–L2, L5 / **L0–L2** / L0 | 1.7–107 |
| **Qwen3-30B-A3B (R6)** | 48 | 0.044–0.541 | 13 layers (L3, L5, L7–L9, L12, L16, L19, L20, L28, L31, L32, L40) | 0.08, 1.25 (L1) | L0–L3, L8, L16, L28 / **L1** / none | 1.5–18 |

Readings:
- **Llama's score noise is nearly uniform across depth; Qwen's is not.** QK-norm's learned scales give some Qwen
  layers 2–7× Llama's σ at the same key width.
- **Qwen's first two or three layers also have flat attention.** The vote leaves 24–47% of the second moment
  Σ w²‖v − o‖² unread there, against a median of 0.1–0.2% elsewhere (the same as Llama, whose L0 alone is flat).
  Flat attention and large σ together put the tail's error above FP8's in exactly those layers.
- **Where σ is large but attention is concentrated** (Qwen3-30B L3, L8, L16, L28), the vote's exact reads absorb it:
  - at 6 bits the tail stays at 0.2–0.5× FP8's error;
  - at 4 bits it does not.
- **The mechanism holds on every model.** The dense 4-bit tier is 1.5–107× FP8's error in every layer. Reading the
  vote's 1/8 exactly cuts that error by a median factor of 10–22 per model (both at 4-bit keys). The cut is only
  1.4–2.6× in the flat first layers and in the high-σ L3 of R6's Qwen. The unread second-moment share sets the error
  (Theorem M, consequence 1).

**For R6's design (6-bit keys),** Theorem M predicts the per-head attention error at or below FP8's:
- in all 32 Llama-3.1-8B layers;
- in 47 of 48 Qwen3-30B-A3B layers. The exception is L1, at 1.25× FP8's median error; L0 is at 0.75×.

This is at 1.6K tokens, with the vote reading 1/8 as in R6. If R6's Qwen arm shows a gap against FP8, Qwen's L0–L2
are the first place to look. Two remedies:
- 8-bit keys in Qwen's first four layers would cost about +0.005 traffic (0.398 → 0.403);
- the map chooses per-layer widths from one full-precision pass (theory.md §8, step 2).

How much a per-head error costs in KL depends on each layer's downstream sensitivity (`STATUS.md` step 7).

## 3. Part 8 on Qwen

Median bound / error · share of head-steps certified within FP8's per-head error (3 draws; coverage 1.000):

| view | σ at 6 bits | split, 6-bit | part 8, 6-bit | split, 8-bit | part 8, 8-bit |
|---|---|---|---|---|---|
| Qwen3-1.7B L0 | 0.498 | 1,254 · 0 | 900 · 0 | 22.5 · 0 | 9.44 · 0.01 |
| Qwen3-1.7B L1 | 0.261 | 91.6 · 0.03 | 62.5 · 0.05 | 10.6 · 0.08 | 4.86 · 0.16 |
| Qwen3-1.7B L3 | 0.252 | 64.4 · 0.10 | 31.2 · 0.22 | 9.02 · 0.60 | 3.78 · 0.84 |
| Qwen3-1.7B L13 | 0.090 | 15.2 · 0.30 | **6.12 · 0.58** | 4.94 · 0.69 | **2.24 · 0.87** |
| Qwen3-1.7B L13, 8K | 0.090 | 16.4 · 0.50 | **6.52 · 0.72** | 4.48 · 0.81 | **2.24 · 0.90** |
| Qwen3-1.7B L26 | 0.059 | 10.0 · 0.61 | **4.01 · 0.86** | 4.12 · 0.89 | **1.85 · 0.99** |
| Qwen3-1.7B L1, 8K | 0.261 | 81.3 · 0 | 61.4 · 0.02 | 10.0 · 0.18 | 4.63 · 0.43 |
| R6 Qwen L0 | 0.135 | 25.4 · 0 | 13.4 · 0.02 | 5.94 · 0.16 | 2.89 · 0.35 |
| R6 Qwen L1 | 0.138 | 25.2 · 0 | 11.1 · 0.03 | 6.37 · 0.07 | 2.85 · 0.14 |
| R6 Qwen L3 | 0.541 | 877 · 0 | 771 · 0 | 22.6 · 0.26 | 9.71 · 0.58 |
| R6 Qwen L12 | 0.169 | 33.0 · 0.24 | 13.1 · 0.44 | 7.25 · 0.70 | 3.02 · 0.89 |
| R6 Qwen L24 | 0.098 | 15.3 · 0.66 | **6.15 · 0.88** | 4.96 · 0.91 | **2.24 · 0.96** |

- **In typical layers part 8 transfers unchanged.** At σ ≈ 0.06–0.10 (6 bits) it is 4.0–6.5× the error, against Llama's
  6.7–7.3×, with 0.58–0.88 certified at FP8's level. At 8 bits it is 1.9–2.2×, with 0.87–0.99 certified.
- **The bound grows with σ, as theory.md §12 says:** e^t with t ≈ 5.6σ enters the weights. Qwen's high-σ layers (L0–L3,
  R6's L12) are 11–900× the error at 6 bits. That is still 1.1–2.5× tighter than the split, but rarely within FP8's.
- **At 4-bit keys, part 8 is tighter than the split only where σ is small** (Qwen3-1.7B L13 and L26, R6's L24:
  189–298×, 44×, 214×). Elsewhere it is not. Keep Bennett as a second form at 4 bits (findings_part8.md).

## 4. The certified read on Qwen

Rows read exactly (traffic); FP8's per-head error as the oracle target; 2 draws; coverage 1.000.

| view | 6-bit: split | part 8 | part 8, fine schedule | oracle stop | 8-bit: split | part 8 | oracle stop |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B L26 | 0.181 (0.437) | **0.131 (0.403)** | 0.131 (0.403) | 0.125 (0.398) | 0.129 (0.456) | **0.125 (0.453)** | 0.125 (0.453) |
| Qwen3-1.7B L13 | 0.201 (0.451) | 0.161 (0.423) | **0.156 (0.420)** | 0.125 (0.399) | 0.139 (0.462) | **0.127 (0.454)** | 0.125 (0.453) |
| R6 Qwen L24 | 0.184 (0.439) | **0.138 (0.408)** | — | 0.126 (0.399) | 0.132 (0.458) | **0.126 (0.454)** | 0.125 (0.453) |
| Qwen3-1.7B L3 | 0.622 (0.740) | 0.481 (0.643) | 0.470 (0.636) | 0.126 (0.399) | 0.174 (0.484) | 0.131 (0.457) | 0.125 (0.453) |
| R6 Qwen L12 | 0.726 (0.812) | 0.556 (0.695) | — | 0.130 (0.402) | 0.241 (0.526) | 0.152 (0.470) | 0.125 (0.453) |
| Qwen3-1.7B L1 | 0.989 (0.993) | 0.958 (0.971) | 0.961 (0.973) | 0.281 (0.506) | 0.708 (0.818) | 0.464 (0.665) | 0.145 (0.466) |
| Qwen3-1.7B L0 | 0.969 (0.979) | 0.954 (0.968) | 0.950 (0.966) | 0.563 (0.699) | 0.894 (0.933) | 0.837 (0.898) | 0.228 (0.518) |
| R6 Qwen L0 | 1.000 (1.000) | 1.000 (1.000) | — | 0.290 (0.512) | 0.930 (0.956) | 0.600 (0.750) | 0.189 (0.493) |
| R6 Qwen L1 | 1.000 (1.000) | 0.994 (0.996) | — | 0.350 (0.553) | 0.809 (0.880) | 0.554 (0.721) | 0.232 (0.520) |
| R6 Qwen L3 | 1.000 (1.000) | 1.000 (1.000) | — | 0.294 (0.515) | 0.870 (0.919) | 0.652 (0.782) | 0.147 (0.467) |

- **In typical layers the certified read is at the oracle.** At 6 bits it reaches FP8's per-head error at traffic
  0.403–0.423, against the oracle stop's 0.398–0.399 and FP8's 0.50; at 8 bits, 0.453–0.454. That is better than on
  Llama-1B's early layers (0.480–0.499).
- **In Qwen's early layers FP8's per-head error is out of reach at FP8's traffic even for the oracle**: 0.51–0.70 at
  6 bits, 0.47–0.52 at 8. The causes:
  - FP8's own error there is tiny (2.5·10⁻⁴ on R6's L1);
  - attention is flat;
  - R6's model shares a KV head among 8 query heads, and the worst head decides for the group.

  The certificate cannot help where the oracle cannot.

## Caveats

- 1,600 tokens (two views at 8,192), one passage, per-head medians.
- R6 runs 32K and 128K: the unread share at those lengths is not measured here. The 8K Qwen3-1.7B views behave like
  1.6K, and longer contexts helped Llama (findings_part8.md).
- FP8's per-head error is an oracle target. Per-head error above FP8's does not imply worse end-to-end KL; R6 measures
  that.
