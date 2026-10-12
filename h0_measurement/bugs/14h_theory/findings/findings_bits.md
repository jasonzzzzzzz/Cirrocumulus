# Choosing key and value bits from Theorem M (CPU, 2026-10-11)

Question (user, 2026-10-11): can the theory describe and manage the errors of 6/4, 4/6, 4/4, …, and does one
configuration (6/6) serve every kind of model?

Method:
- `../layermap_s1h5.py --grid` evaluates Theorem M's exact tail error on every layer for key bits {4, 5, 6, 8} × value
  bits {4, 5, 6, 8} × exact read {1/8, 1/4}, from one FP pass with no draws (theory.md §10b, Corollary M-B);
- 18 models on PG-19 (1,600 tokens);
- 6 models also on three LongBench v2 contexts: a code repository, a table, and an agent's history log.

The target is FP8's per-head error, either for the median head-step or, stricter, for 90% of them (p90). Traffic is
(1 − f)(k + v)/32 + f.

Data:
- `findings_bits.json` (every grid and the Monte Carlo checks);
- `findings_bits_alloc.json` (the allocations).

Corrected 2026-10-11:
- **What was wrong.** Granite-4.2's L4 and L6 have near-zero scores (|s| ≈ 10⁻⁹). There, `theory3_s1h5._log_sinhc` lost
  every digit and added a spurious key term, so this note first reported both layers unmet even at 8/8.
- **The fix.** A series below |x| = 0.1, tested against a 100-digit reference. Granite-4.2's grids were re-run on all
  four inputs.
- **The check.** Theorem M now matches 200 dither draws on both layers (0.99–1.00; `montecarlo_near_zero_scores`).
- **Scope.** No other model's scores are small enough to be affected.

`../MODEL_FAMILY.md` turns these results into families of layers and models.

## 1. The error surface is predicted exactly

Monte Carlo (200 draws) over the prediction at off-diagonal configurations, median over head-steps:

| view | 5/5 | 4/6 | 6/6 | 8/6 | 4/8 | 6/8 |
|---|---|---|---|---|---|---|
| Llama-1B L1 (concentrated) | 1.001 | 0.989 | 0.991 | 0.997 | 0.989 | 0.991 |
| Gemma 3 L5 (flat) | 1.001 | 1.000 | 1.000 | 1.000 | 1.001 | 0.998 |
| Granite-4.2 L1 (flat) | 1.000 | 1.001 | 1.001 | 1.001 | 0.997 | 0.999 |
| MiniMax-M2 L1 | 0.999 | 0.989 | 1.001 | 1.000 | 0.989 | 0.998 |
| R6 Qwen L1 | 0.997 | 0.985 | 0.997 | 0.999 | 0.986 | 0.998 |
| GPT-OSS L1 (key-limited) | 0.981 | **0.921** | 0.996 | 1.000 | **0.922** | 0.995 |
| Qwen3-1.7B L1 (key-limited, flat) | 0.995 | **0.916** | 1.004 | 1.001 | **0.916** | 1.005 |

Within 2% wherever keys have ≥ 5 bits. At 4-bit keys on high-σ layers the first-order reading overpredicts by about
8% (conservative).

## 2. The cheapest bits per layer, against uniform configurations

Per model: the mean traffic of the per-layer allocation (the cheapest configuration meeting the target in each layer),
and the number of layers each uniform configuration fails, for the median / p90 target.

| model | layers | allocation, median target | allocation, p90 target | 4/4 | 6/4 | 4/6 | 6/6 | 8/4 | 8/6 | 8/8 | most-chosen (p90): k/v/(1/f) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 16 | 0.345 | 0.368 | 1/12 | 0/1 | 1/11 | 0/0 | 0/1 | 0/0 | 0/0 | 5/4 ×10, 4/4 ×4 |
| Llama-3.1-8B (R6) | 32 | 0.345 | 0.354 | 1/9 | 0/1 | 0/8 | 0/0 | 0/0 | 0/0 | 0/0 | 4/4 ×23, 5/4 ×6 |
| Llama-3.3-70B | 12 | 0.344 | 0.387 | 0/12 | 0/1 | 0/11 | 0/0 | 0/1 | 0/0 | 0/0 | 5/5 ×6, 5/4 ×5 |
| Mistral-7B | 32 | 0.344 | 0.362 | 0/17 | 0/0 | 0/16 | 0/0 | 0/0 | 0/0 | 0/0 | 4/4 ×15, 5/4 ×13 |
| Mistral-Small-3.2 | 20 | 0.346 | 0.363 | 2/8 | 0/4 | 2/6 | 0/0 | 0/4 | 0/0 | 0/0 | 4/4 ×12 |
| Qwen1.5-0.5B | 24 | 0.347 | 0.377 | 2/21 | 0/1 | 1/21 | 0/1 | 0/0 | 0/0 | 0/0 | 5/4 ×15 |
| Qwen1.5-MoE | 24 | 0.346 | 0.352 | 1/3 | 0/1 | 1/3 | 0/0 | 0/1 | 0/0 | 0/0 | 4/4 ×21 |
| Qwen3-1.7B | 28 | 0.357 | 0.388 (1 unmet) | 4/21 | 3/3 | 4/20 | 3/3 | 1/3 | 0/2 | 0/1 | 5/4 ×13, 4/4 ×7 |
| Qwen3-30B-A3B (R6) | 48 | 0.350 | 0.374 | 7/23 | 1/8 | 6/23 | 0/5 | 0/2 | 0/0 | 0/0 | 4/4 ×25, 5/4 ×9, 8/4 ×3 |
| GLM-4.5-Air | 9 | 0.359 | 0.380 | 3/5 | 2/3 | 2/5 | 0/0 | 2/2 | 0/0 | 0/0 | 4/4 ×4 |
| Granite-4.0-H | 4 | 0.357 | 0.398 | 1/4 | 1/1 | 0/4 | 0/1 | 1/1 | 0/1 | 0/0 | 5/4 ×2 |
| Gemma-4-31B | 6 | 0.376 | 0.435 | 4/6 | 1/6 | 2/6 | 0/0 | 1/6 | 0/0 | 0/0 | 5/6 ×4, 6/6 ×2 |
| Qwen3.5-27B | 8 | 0.381 | 0.429 | 5/8 | 4/6 | 5/8 | 0/0 | 4/6 | 0/0 | 0/0 | 6/6 ×5 |
| Qwen3.8-27B | 8 | 0.381 | 0.436 | 5/8 | 4/6 | 5/8 | 0/1 | 4/6 | 0/0 | 0/0 | 6/6 ×4 |
| Gemma-3-12B | 8 | 0.453 | 0.453 | 8/8 | 8/8 | 8/8 | 0/0 | 8/8 | 0/0 | 0/0 | 6/6 ×8 |
| GPT-OSS-20B | 12 | 0.398 | 0.458 | 12/12 | 1/12 | 12/12 | **0/12** | 0/2 | 0/0 | 0/0 | 8/4 ×10, 8/5 ×2 |
| Granite-4.2-30B | 16 | 0.400 | 0.429 | 15/16 | 14/16 | 6/10 | 0/3 | 13/16 | 0/3 | 0/0 | 4/6 ×6, 6/6 ×5, 4/8 ×3 |
| MiniMax-M2 | 6 | 0.417 | 0.471 | 6/6 | 5/6 | 5/6 | 0/3 | 4/6 | 0/0 | 0/0 | 6/6 ×2 |

Uniform traffic, for comparison: 4/4 0.344; 6/4 and 4/6 0.398; 6/6 and 8/4 0.453; 8/6 0.508; 8/8 0.563. FP8 is 0.50.
"Unmet" means no configuration up to 8/8 with a 1/4 read meets the target.

## 3. The optimal split, and the kinds of layers

Corollary M-B gives the split in closed form, k* − v* ≈ ½·log₂(a/b), with a and b the layer's key and value
sensitivities. At equal widths a/(a + b) is the key term's share, so the 6/6 grid gives it directly. Median over layers
(10th–90th percentile):

| model | k* − v* | class |
|---|---|---|
| GPT-OSS-20B | +2.69 (+2.47, +3.00) | key-heavy: sharp, noisy scores |
| Qwen1.5-0.5B | +2.14 (+1.96, +2.33) | key-heavy |
| Llama-3.2-1B | +1.99 (+1.76, +2.11) | key-heavy |
| Qwen3-1.7B | +1.92 (+1.36, +3.49) | key-heavy, with a wide spread (its early layers) |
| Qwen3-30B-A3B (R6) | +1.77 (+1.14, +3.02) | key-heavy |
| GLM-4.5-Air | +1.46 (−0.31, +1.61) | key-leaning |
| Llama-3.1-8B (R6), Llama-3.3-70B | +1.42, +1.41 | key-leaning |
| Qwen1.5-MoE | +1.37 | key-leaning |
| Mistral-7B, Mistral-Small-3.2 | +1.20, +0.79 | key-leaning |
| Qwen3.5-27B, Qwen3.8-27B | +0.76, +0.82 | key-leaning to balanced |
| MiniMax-M2 | +0.55 (−0.32, +0.88) | balanced |
| Gemma-4-31B | +0.32 (+0.10, +0.42) | balanced |
| Gemma-3-12B | −0.59 (−0.66, −0.43) | value-leaning: flat attention |
| Granite-4.2-30B | −2.32 (below −5, +0.81) | value-leaning; most of L0–L9 almost all value |

The split classes the layer by its key-to-value sensitivity: key-heavy, key-leaning, balanced, or value-leaning. How
many bits it needs in total is a separate question, set by the unread second moment. A model mixes layers of several
classes, and no single class describes a model.

## 4. Stability across inputs (p90 target)

| model | traffic: PG-19 / code / table / agent log | per-layer max over the four | layers where PG-19's choice misses on code / table / agent |
|---|---|---|---|
| Mistral-7B | 0.369 / 0.390 / 0.344 / 0.349 | 0.393 | 9 / 0 / 0 of 16 |
| Llama-3.1-8B (R6) | 0.364 / 0.381 / 0.347 / 0.373 | 0.386 | 9 / 0 / 9 of 16 |
| Qwen3-30B-A3B (R6) | 0.410 / 0.412 / 0.389 / 0.380 | 0.421 | 3 / 2 / 0 of 12 |
| GPT-OSS-20B | 0.458 / 0.480 / 0.453 / 0.453 | 0.480 | 10 / 0 / 1 of 12 |
| Gemma-3-12B | 0.453 / 0.453 / 0.429 / 0.453 | 0.453 | 0 / 0 / 0 of 8 |
| Granite-4.2-30B | 0.429 / 0.431 / 0.381 / 0.386 | 0.434 | 2 / 0 / 0 of 16 |

The best bits depend on the input; code is the hardest of the four. An allocation calibrated on several inputs (the
per-layer maximum) costs 0.39–0.48, still below FP8's 0.50. A run-time check must cover what calibration misses.

## Readings

- **6/4 suffices on concentrated models, but a per-layer mix is cheaper and safer.** On Llama, Mistral, Qwen1.5 and
  Qwen3 it is 4/4 or 5/4 in most layers and 6/4–8/5 in a few: traffic 0.35–0.39 at the p90 target, against uniform
  6/4's 0.398, which misses 0–8 layers.
- **6/6 is not universal:**
  - it fails every GPT-OSS full-attention layer at p90, which need 8-bit keys;
  - it fails 3–5 Qwen3 layers (key-heavy) and 3 MiniMax-M2 layers;
  - it fails 3 Granite-4.2 layers (L2, L3, L8: near-constant scores, which need 8-bit values).

  It suits most flat layers: every mapped global layer of Gemma 3, Gemma 4 and Qwen3.5.
- **One layer cannot reach FP8's per-head error even at 8/8 with a 1/4 read:** Qwen3-1.7B L0 (flat, with σ ≈ 0.5 at
  6-bit keys). It needs a larger exact read.
- **The unit of choice is the layer, and the input matters.** Concentrated and flat are properties of layers and
  inputs, not of models.

## The design this supports (theory.md §10b)

1. **Offline:** a per-layer (or per-head) allocation from Theorem M's surface on a few diverse calibration inputs,
   taking the per-layer maximum.
2. **At run time:** Theorem R's certificate (part 8) on every head-step.
3. **Escalation when the certificate misses:** the certified read's nested coin-free schedule. It can fetch more rows
   exactly, or more precision from dithered residual planes: base + residual is again a subtractive-dithered tier with
   an exact law, so Theorems R and M hold for any per-row mix chosen coin-free.

The residual planes are not implemented yet.
