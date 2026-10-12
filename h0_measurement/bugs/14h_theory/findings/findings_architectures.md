# The theorems on every available architecture (CPU, 2026-10-10/11)

Every model in the local Hugging Face cache that has weights and a softmax KV cache was tested. That is 21 models,
counting the ones done before.

| family | models | tested before? |
|---|---|---|
| GQA | Llama-3.2-1B, Llama-3.1-8B (R6), **Llama-3.3-70B**, **Mistral-7B-v0.3**, **Mistral-Small-3.2-24B** (the text model of a VLM), **Granite-4.2-30B** (attention multiplier 1/128) | the first two |
| MHA, QKV bias | **Qwen1.5-0.5B**, **Qwen1.5-MoE-A2.7B** | no |
| QK-norm | Qwen3-1.7B, Qwen3-30B-A3B (R6), **MiniMax-M2** (FP8, MoE, 6 query heads per KV head) | the first two |
| partial RoPE, QKV bias | **GLM-4.5-Air** (12 query heads per KV head) | no |
| local : global | Gemma-3-12B, **Gemma-4-31B** (global head dim 512; keys and values from one projection) | Gemma 3 |
| sinks, banded : full | GPT-OSS-20B | yes |
| hybrid recurrent : attention | Granite-4.0-H (Mamba-2 : attention), **Qwen3.5-27B** and **Qwen3.8-27B** (Gated DeltaNet : gated attention, head dim 256) | Granite-4.0-H |
| MLA | DeepSeek-V2-Lite, Moonlight, Kimi-Linear | Theorem R only |

Models in bold are new here.

**Not tested.**
- **No weights downloaded** (config only): Nemotron-3-Nano-Omni, Qwen3-VL (multimodal RoPE), Qwen3.5-9B, Gemma-4-12B,
  MiniCPM-o-4.5, VideoChat3-4B. I did not download them.
- **Same architecture as a tested model:** Llama-3.3 (= 3.1), Qwen3-0.6B/8B, MiniMax-M2.7 (= M2), Qwen3-30B-A3B base.
- **Outside the theory's scope by design:** recurrent and linear-attention layers (Mamba-2, Gated DeltaNet, KDA) and
  sliding-window layers. They keep no growing per-row cache to compress.

Code:
- `../adapters_s1h5.py`, now with a multimodal text-model path, block-FP8 dequantization, a Mistral tekken tokenizer and a
  hard check that every attention weight loaded;
- `../exp4bit_s1h5.py capture | evaluate`;
- `../theory5_s1h5.py compare | certread`;
- `../layermap_s1h5.py [--vbits]`;
- `../mla_m_s1h5.py`, new: Theorem M for MLA.

Data: `findings_architectures.json`, `layermap_<model>.json` (and `_v6` with 6-bit values), `findings_mla_m.json`, and
the new entries of `findings_4bit.json`.

Setup as part 8: 1,600 tokens of a PG-19 passage; the vote's 1/8 read exactly; 16 measured queries per head; δ = 10⁻³;
4-bit dithered values unless stated. One early and one deeper global layer per new model.

## 1. Validity: every theorem holds on every architecture

- **Architecture checks** (`adapters_s1h5.py`; the first 4 layers, or the global layers in the first 8–12):
  - every view reconstructs its layer's output to ≤ 9·10⁻⁶;
  - Lemmas 2 and 4 hold;
  - Theorem R (v1) covers every draw.

  This holds on all 11 new models, for 38 layers in all.

  MiniMax-M2's FP8 weights were checked against a manual block dequantization: the difference is 0.
- **Theorem R, tight forms:** coverage is 1.000 in every cell. That is part 8's bound on 22 new views at 4, 6 and 8-bit
  keys (3 draws), and the certified read on 22 views at 6 and 8 bits (2 draws).
- **Theorem M, GQA:** measured / predicted MSE (200 draws) is 0.990–1.003 on all 22 new views at 4, 6 and 8-bit keys.
- **Theorem M, MLA** (new, `mla_m_s1h5.py`, theory.md §10): 0.98–1.01 at 4, 6 and 8 bits on Moonlight L1/L13,
  DeepSeek-V2-Lite L1/L13 and Kimi-Linear's MLA layer (200 draws). The coupling term that GQA lacks is ≤ 0.1% of the
  error.

| model | layers checked | query : KV heads per group | head dim | view error | Lemmas 2, 4 | Theorem R v1 coverage |
|---|---|---|---|---|---|---|
| Qwen1.5-0.5B | 0–3 | 16 × 1 (MHA) | 64 | 9e-06 | ✓ | 1.000 |
| Qwen1.5-MoE-A2.7B | 0–3 | 16 × 1 (MHA) | 128 | 2e-06 | ✓ | 1.000 |
| Mistral-7B-v0.3 | 0–3 | 8 × 4 | 128 | 2e-06 | ✓ | 1.000 |
| Mistral-Small-3.2-24B | 0–3 | 8 × 4 | 128 | 2e-06 | ✓ | 1.000 |
| Granite-4.2-30B | 0–3 | 8 × 4 | 128 | 2e-06 | ✓ | 1.000 |
| Llama-3.3-70B | 0–3 | 8 × 8 | 128 | 2e-06 | ✓ | 1.000 |
| GLM-4.5-Air | 0–3 | 8 × 12 | 128 | 3e-06 | ✓ | 1.000 |
| MiniMax-M2 | 0–3 | 8 × 6 | 128 | 1e-06 | ✓ | 1.000 |
| Qwen3.5-27B | 3, 7 | 4 × 6 | 256 | 5e-06 | ✓ | 1.000 |
| Qwen3.8-27B | 3, 7 | 4 × 6 | 256 | 5e-06 | ✓ | 1.000 |
| Gemma-4-31B | 5, 11 | 4 × 8 | 512 | 5e-06 | ✓ | 1.000 |

## 2. Part 8 and Theorem M per view

Median bound / error · share of head-steps certified within FP8's per-head error. "Tail / FP8" is the tail design's
measured RMS error over FP8's at 4 / 6 / 8-bit keys.

| view | split, 6-bit | part 8, 6-bit | part 8, 8-bit | part 8, 4-bit (split) | Theorem M, 4 / 6 / 8 | tail / FP8, 4 / 6 / 8 |
|---|---|---|---|---|---|---|
| Qwen1.5-0.5B L1 | 15.1 · 0.03 | 5.93 · 0.17 | 2.43 · 0.75 | 165 (278) | 0.996 / 1.003 / 1.000 | 1.06 / 0.38 / 0.28 |
| Qwen1.5-0.5B L12 | 21.6 · 0.23 | 7.82 · 0.53 | 2.77 · 0.87 | 396 (755) | 0.990 / 0.995 / 1.000 | 0.40 / 0.12 / 0.09 |
| Qwen1.5-MoE L1 | 5.95 · 0.04 | 2.92 · 0.08 | 1.65 · 0.22 | 15.4 (26.5) | 0.998 / 0.999 / 0.998 | 1.72 / 0.96 / 0.90 |
| Qwen1.5-MoE L12 | 12.3 · 0.60 | 4.62 · 0.85 | 2.01 · 0.94 | 69.2 (148) | 0.998 / 1.002 / 1.001 | 0.16 / 0.07 / 0.06 |
| Mistral-7B L1 | 9.24 · 0.41 | 4.06 · 0.73 | 1.86 · 0.92 | 57.1 (82.7) | 1.001 / 0.998 / 0.998 | 0.33 / 0.14 / 0.11 |
| Mistral-7B L16 | 9.95 · 0.45 | 3.98 · 0.79 | 1.86 · 0.98 | 43.4 (92.2) | 0.996 / 1.000 / 1.001 | 0.26 / 0.11 / 0.10 |
| Mistral-Small L1 | 6.82 · 0.01 | 3.16 · 0.11 | 1.67 · 0.40 | 20.5 (37.8) | 1.000 / 0.997 / 1.000 | 1.43 / 0.74 / 0.69 |
| Mistral-Small L20 | 8.58 · 0.82 | 3.44 · 0.95 | 1.77 · 0.99 | 24.7 (55.3) | 1.002 / 0.998 / 0.999 | 0.09 / 0.04 / 0.04 |
| Granite-4.2 L1 | 4.32 · 0.00 | 1.33 · 0.12 | 1.29 · 0.12 | 1.4 (4.37) | 1.003 / 1.003 / 1.003 | 1.77 / 1.77 / 1.77 |
| Granite-4.2 L16 | 9.58 · 0.02 | 4.34 · 0.07 | 1.91 · 0.39 | 78.2 (108) | 0.995 / 1.001 / 1.000 | 1.74 / 0.76 / 0.64 |
| Llama-3.3-70B L1 | 10.2 · 0.08 | 4.43 · 0.36 | 1.90 · 0.89 | 69.7 (103) | 0.998 / 0.999 / 1.001 | 0.74 / 0.29 / 0.24 |
| Llama-3.3-70B L20 | 7.26 · 0.72 | 3.35 · 0.93 | 1.72 · 1.00 | 29.4 (44.2) | 0.995 / 0.997 / 0.996 | 0.17 / 0.08 / 0.07 |
| GLM-4.5-Air L1 | 7.70 · 0.01 | 3.65 · 0.01 | 1.78 · 0.11 | 33.2 (53.0) | 0.997 / 0.998 / 0.999 | 2.37 / 1.19 / 1.07 |
| GLM-4.5-Air L8 | 11.5 · 0.34 | 4.35 · 0.84 | 1.94 · 0.99 | 60.2 (125) | 0.994 / 1.000 / 1.001 | 0.30 / 0.12 / 0.10 |
| MiniMax-M2 L1 | 7.15 · 0.01 | 3.35 · 0.01 | 1.75 · 0.06 | 22.3 (38.4) | 0.992 / 0.996 / 0.996 | 2.78 / 1.69 / 1.55 |
| MiniMax-M2 L5 | 7.93 · 0.09 | 3.37 · 0.18 | 1.71 · 0.26 | 22.5 (53.3) | 0.997 / 0.999 / 1.001 | 2.20 / 1.11 / 1.00 |
| Qwen3.5-27B L3 | 6.05 · 0.01 | 2.63 · 0.05 | 1.50 · 0.12 | 14.4 (33.9) | 0.998 / 0.999 / 1.000 | 2.03 / 1.34 / 1.29 |
| Qwen3.5-27B L15 | 8.69 · 0.03 | 3.58 · 0.10 | 1.71 · 0.24 | 24.4 (62.4) | 0.995 / 0.999 / 0.998 | 2.09 / 1.15 / 1.06 |
| Qwen3.8-27B L3 | 6.34 · 0.00 | 2.69 · 0.04 | 1.52 · 0.18 | 15.2 (38.4) | 0.999 / 0.999 / 0.998 | 1.90 / 1.22 / 1.17 |
| Qwen3.8-27B L15 | 9.12 · 0.04 | 3.65 · 0.11 | 1.72 · 0.26 | 27.1 (71.8) | 1.000 / 0.999 / 1.001 | 2.18 / 1.18 / 1.09 |
| Gemma-4-31B L5 | 7.56 · 0.04 | 3.11 · 0.10 | 1.55 · 0.22 | 20.8 (58.2) | 0.995 / 1.000 / 1.000 | 2.36 / 1.52 / 1.45 |
| Gemma-4-31B L29 | 6.25 · 0.03 | 2.80 · 0.15 | 1.48 · 0.39 | 18.4 (45.0) | 0.998 / 0.999 / 0.999 | 1.20 / 0.84 / 0.81 |

- **The certificate is tight everywhere:**
  - part 8 is 1.3–7.8× the error at 6-bit keys, median about 3.5;
  - it is 1.3–2.8× at 8-bit keys;
  - it is 2.0–3.3× tighter than the split on every view, at both 6 and 8 bits;
  - at 4-bit keys it beats the split on every new view (1.4–396× against 4.4–755×).
- **The share certified within FP8 varies because the design's own error varies, not because the bound does.** Where
  tail / FP8 exceeds 1, no valid certificate can reach FP8's level.

## 3. Which bits a model needs: Theorem M's per-layer maps

`layermap_s1h5.py` on every global layer of the smaller models and on the first 6–20 layers (or the first global
layers) of the larger ones. The table gives the layers whose median tail error exceeds FP8's, at 4 / 6 / 8-bit keys
with 4-bit values, and the median unread share of Σ w²‖v − o‖².

| model | layers mapped | σ at 6 bits | above FP8's error at 4 / 6 / 8-bit keys | unread second-moment share | key share of the tail MSE at 6 bits |
|---|---|---|---|---|---|
| Llama-3.2-1B | all 16 | 0.07–0.11 | L0 / — / — | 0.0011 | — |
| Llama-3.1-8B (R6) | all 32 | 0.04–0.08 | L0 / — / — | 0.0005 | — |
| Llama-3.3-70B | 0–11 | 0.01–0.09 | — / — / — | 0.0072 | 0.29 |
| Mistral-7B | all 32 | 0.03–0.07 | — / — / — | 0.0033 | — |
| Mistral-Small-3.2 | 0–19 | 0.02–0.05 | L1, L2 / — / — | 0.0028 | — |
| Qwen1.5-0.5B | all 24 | 0.09–0.73 | L0, L1 / — / — | 0.0033 | 0.53 |
| Qwen1.5-MoE | all 24 | 0.04–0.08 | L1 / — / — | 0.0007 | — |
| Qwen3-1.7B | all 28 | 0.06–0.50 | 4 layers / L0–L2 / L0 | 0.0024 | — |
| Qwen3-30B-A3B (R6) | all 48 | 0.04–0.54 | 7 layers / L1 / — | 0.0011 | — |
| GLM-4.5-Air | 0–8 | 0.02–0.08 | L0–L2 / L0, L1 / L0, L1 | 0.0023 | 0.30 |
| GPT-OSS-20B | all 12 full | 0.15–0.23 | all 12 / L3 / — | 0.154 | 0.74 |
| Gemma-4-31B | 6 globals (5–35) | 0.04 | 4 / L5 / L5 | 0.122 | 0.08 |
| Qwen3.5-27B | 8 globals (3–31) | 0.04–0.06 | 5 / L3–L15 / L3–L15 | 0.142 | 0.15 |
| Qwen3.8-27B | 8 globals (3–31) | 0.04–0.06 | 5 / L3–L15 / L3–L15 | 0.146 | — |
| MiniMax-M2 | 0–5 | 0.03–0.05 | all 6 / L1–L5 / L1, L2, L4, L5 | 0.488 | 0.12 |
| Granite-4.0-H | all 4 attention (5–35) | 0.00–0.08 | L5 / L5 / L5 | 0.026 | — |
| Gemma-3-12B | all 8 globals | 0.03 | all / all / all | 0.640 | 0.02 |
| Granite-4.2-30B | 0–15 | 0.00–0.06 | 15 / 14 / 13 layers | 0.618 | 0.04 |

Two regimes:
- **Concentrated attention**: Llama, Mistral, Qwen1.5 and Qwen3, and GLM-4.5 beyond its first two layers. The
  vote's 1/8 leaves ≤ 1% of the second moment unread, and R6's design point (6-bit keys, 4-bit values) keeps the tail
  below FP8's error in every layer except at most the first three.
- **Flat attention**: Gemma 3 and 4, Granite-4.2, MiniMax-M2's first layers, and Qwen3.5/3.8's global layers. The vote
  leaves 12–64% of the second moment unread, and 76–84% of the mass for Gemma 3 and Granite-4.2. Granite-4.2's
  attention multiplier (1/128 rather than 1/√128) flattens its scores 11-fold. The 4-bit values then dominate the error:
  the key term is 2–15% of it, so 8-bit keys barely help.

**Theorem M names the fix: value bits.** The same maps with 6-bit values (`layermap_<model>_v6.json`), at 6-bit keys:

| model | tail / FP8, 4-bit → 6-bit values (median) | layers above FP8 |
|---|---|---|
| Gemma-3-12B | 2.73 → 0.76 | 8 → 0 |
| Granite-4.2-30B | 1.89 → 0.51 | 14 → 0 |
| MiniMax-M2 | 1.18 → 0.53 | 5 → 0 |
| Qwen3.5-27B | 0.86 → 0.36 | 4 → 0 |
| Gemma-4-31B | 0.79 → 0.27 | 1 → 0 |
| GLM-4.5-Air | 0.17 → 0.10 | 2 (L0, L1) → 0 |

Corrected 2026-10-11: this table first showed Granite-4.2 L4 and L6 still above FP8 with 6-bit values. Their scores are
near zero (|s| ≈ 10⁻⁹), where `theory3_s1h5._log_sinhc` lost every digit and added a spurious key term. With the fix,
Theorem M matches 200 dither draws there within 1% (`findings_bits.json`, `montecarlo_near_zero_scores`).

Six-bit keys with 6-bit values cost traffic (1 − 1/8)·12/32 + 1/8 = 0.453, the same as 8-bit keys with 4-bit values and
below FP8's 0.50. So the map's recommendation per architecture, from one full-precision pass, is:
- 6/4 for concentrated attention;
- 6/6 for flat attention, or per layer;
- 8-bit keys where σ is large (GPT-OSS: the key term is 74% of its error).

Refined per layer in `findings_bits.md` and `../MODEL_FAMILY.md`: 6/6 is not universal either.

## 4. The certified read

Rows read exactly (traffic); FP8's per-head error as the oracle target; coverage 1.000 everywhere.

| view | 6-bit: split | part 8 | oracle stop | 8-bit: part 8 | oracle stop |
|---|---|---|---|---|---|
| Qwen1.5-MoE L12 | 0.137 (0.407) | **0.127 (0.400)** | 0.125 (0.399) | 0.125 (0.453) | 0.125 (0.453) |
| Mistral-Small L20 | 0.146 (0.413) | **0.129 (0.401)** | 0.125 (0.398) | 0.125 (0.453) | 0.125 (0.453) |
| Qwen1.5-0.5B L12 | 0.160 (0.423) | **0.135 (0.405)** | 0.125 (0.399) | 0.126 (0.454) | 0.125 (0.453) |
| Llama-3.3-70B L20 | 0.235 (0.474) | **0.145 (0.412)** | 0.125 (0.398) | 0.125 (0.453) | 0.125 (0.453) |
| Mistral-7B L16 | 0.221 (0.465) | **0.150 (0.416)** | 0.125 (0.399) | 0.127 (0.454) | 0.125 (0.453) |
| GLM-4.5-Air L8 | 0.423 (0.604) | **0.187 (0.441)** | 0.125 (0.398) | 0.127 (0.454) | 0.125 (0.453) |
| Qwen1.5-0.5B L1 | 0.398 (0.586) | 0.259 (0.491) | 0.125 (0.399) | 0.131 (0.457) | 0.125 (0.453) |
| Mistral-7B L1 | 0.406 (0.592) | 0.268 (0.497) | 0.133 (0.404) | 0.160 (0.475) | 0.128 (0.455) |
| Gemma-4-31B L29 | 0.465 (0.632) | 0.316 (0.530) | 0.150 (0.416) | 0.195 (0.497) | 0.149 (0.468) |
| Qwen3.5-27B L15 | 0.625 (0.742) | 0.426 (0.605) | 0.188 (0.442) | 0.260 (0.537) | 0.181 (0.488) |
| Qwen3.8-27B L15 | 0.658 (0.765) | 0.445 (0.619) | 0.207 (0.455) | 0.287 (0.554) | 0.200 (0.500) |
| Qwen3.8-27B L3 | 0.727 (0.812) | 0.459 (0.628) | 0.234 (0.474) | 0.334 (0.584) | 0.227 (0.517) |
| Llama-3.3-70B L1 | 0.758 (0.834) | 0.475 (0.639) | 0.131 (0.402) | 0.165 (0.478) | 0.129 (0.455) |
| Qwen1.5-MoE L1 | 0.627 (0.743) | 0.501 (0.657) | 0.266 (0.496) | 0.369 (0.605) | 0.262 (0.539) |
| Mistral-Small L1 | 0.759 (0.835) | 0.575 (0.708) | 0.222 (0.465) | 0.319 (0.574) | 0.214 (0.509) |
| Qwen3.5-27B L3 | 0.772 (0.844) | 0.600 (0.725) | 0.291 (0.513) | 0.379 (0.612) | 0.285 (0.553) |
| MiniMax-M2 L5 | 0.778 (0.848) | 0.610 (0.732) | 0.258 (0.490) | 0.433 (0.646) | 0.241 (0.526) |
| Gemma-4-31B L5 | 0.850 (0.897) | 0.635 (0.749) | 0.274 (0.501) | 0.388 (0.617) | 0.260 (0.537) |
| Granite-4.2 L16 | 0.902 (0.933) | 0.752 (0.829) | 0.233 (0.473) | 0.388 (0.618) | 0.209 (0.505) |
| Granite-4.2 L1 | 1.000 (1.000) | 0.891 (0.925) | 0.804 (0.865) | 0.885 (0.928) | 0.804 (0.878) |
| MiniMax-M2 L1 | 0.989 (0.993) | 0.953 (0.967) | 0.547 (0.689) | 0.776 (0.860) | 0.526 (0.704) |
| GLM-4.5-Air L1 | 1.000 (1.000) | 0.991 (0.994) | 0.554 (0.693) | 0.797 (0.873) | 0.481 (0.675) |

- **In concentrated layers the certified read is at the oracle:** 6-bit traffic 0.400–0.441, against the oracle stop's
  0.398, on six architectures. At 8 bits it is 0.453–0.454.
- **Where 4-bit values hold the error above FP8's, even the oracle needs 0.44–0.87.** The certified read then costs
  0.61–0.99. Theorem M's remedy is value bits (§3).

## 5. What this says about the theory's generality

- **Validity is architectural-detail-free.** Every check holds on GQA, MHA, QK-norm, QKV bias, partial RoPE, attention
  multipliers, FP8 weights, local:global, sinks, the gated full attention of recurrent hybrids, and MLA. Theorem R and
  Theorem M depend only on the cache being softmax-attended rows, and the dither makes them hold for every input.
- **Tightness depends on σ and on how concentrated attention is**, and Theorem M reads both from one pass.
- **The design point does not transfer unchanged.** R6's 6/4 is right for concentrated-attention models (R6's two
  models included) and wrong for flat-attention ones, where 6/6 (0.453, the traffic of 8/4) fixes most layers. The
  paper's generality claim should therefore be "the certificate and the error formula hold for any softmax attention;
  the formula picks the bits". It should not be "6/4 is FP8-level everywhere". `../MODEL_FAMILY.md` §3 and §7 refine
  this: 6/6 also fails some layers, and the claim holds only under its stated conditions.
