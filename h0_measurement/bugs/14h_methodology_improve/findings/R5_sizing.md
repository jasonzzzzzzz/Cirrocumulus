# R5 → the confirmatory campaign's size (draft input to plan.md amendment "R6 (draft)")

`size_confirm_r5.py`; R5's valid blocks and R5.3's finished ones (h53llama128_r1_22773130). Claim per model × family: mean KL/token(design) < 1.25 × mean KL/token(FP8) + 0.0002 (one-sided α = 0.05, unit bootstrap upper bound, paired per unit); all families of a model must pass (intersection–union: no multiplicity correction, so each family is powered at 0.8^(1/K)).
Units are prompt × task units of the R5 cells (R5 used 5–20 per group). n_sim draws units from R5's own per-unit differences, so a group with few or tied units gives a coarse answer; n_cons uses the SD at its 80% upper confidence limit and the mean at its 80% upper bound. ∞ = R5's mean already misses the margin (non-inferiority not expected to hold there).

## qread2t4kqT_v4@0.125 (margin 1.25 × FP8 + 0.0002)

| model | family | R5 units | KL/tok design | KL/tok FP8 | mean y | SD y | power/family | n normal | n cons | n sim |
|---|---|---|---|---|---|---|---|---|---|---|
| llama31-8b | aggregation | 16 | 0.0009 | 0.0007 | -0.0000 | 0.0006 | 0.969 | 80 | 568 | 80 |
| llama31-8b | icl | 10 | 0.0007 | 0.0035 | -0.0037 | 0.0060 | 0.969 | 30 | 152 | 10 |
| llama31-8b | multivalue | 20 | 0.0019 | 0.0040 | -0.0031 | 0.0038 | 0.969 | 17 | 39 | 15 |
| llama31-8b | rag | 10 | 0.0015 | 0.0027 | -0.0019 | 0.0015 | 0.969 | 7 | 17 | 10 |
| llama31-8b | rerank | 5 | 0.0107 | 0.0078 | +0.0009 | 0.0056 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | retrieval | 40 | 0.0004 | 0.0006 | -0.0004 | 0.0022 | 0.969 | 196 | 1125 | 240 |
| llama31-8b | tracking | 20 | 0.0012 | 0.0026 | -0.0021 | 0.0014 | 0.969 | 5 | 8 | 10 |
| qwen3-30b-a3b-2507 | aggregation | 20 | 0.0029 | 0.0026 | -0.0004 | 0.0021 | 0.946 | 143 | 2273 | 160 |
| qwen3-30b-a3b-2507 | multivalue | 20 | 0.0009 | 0.0006 | +0.0002 | 0.0008 | 0.946 | 4544 | ∞ | ∞ |
| qwen3-30b-a3b-2507 | retrieval | 40 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.946 | 5 | 5 | 10 |
| qwen3-30b-a3b-2507 | tracking | 20 | 0.0031 | 0.0013 | +0.0014 | 0.0106 | 0.946 | ∞ | ∞ | ∞ |

- **llama31-8b**: planned units ∞ over 7 families (aggregation 80, icl 30, multivalue 20, rag 10, rerank ∞, retrieval 240, tracking 10); 6-arm GPU time ≈ 8.7 h + model loads
- **qwen3-30b-a3b-2507**: planned units ∞ over 4 families (aggregation 160, multivalue ∞, retrieval 10, tracking ∞); 6-arm GPU time ≈ 1.5 h + model loads

## tail4x_v4@0.125 (margin 1.25 × FP8 + 0.0002)

| model | family | R5 units | KL/tok design | KL/tok FP8 | mean y | SD y | power/family | n normal | n cons | n sim |
|---|---|---|---|---|---|---|---|---|---|---|
| llama31-8b | multivalue | 3 | 0.0005 | 0.0019 | -0.0019 | 0.0008 | 0.969 | 5 | 15 | — |
| llama31-8b | retrieval | 6 | 0.0001 | 0.0003 | -0.0003 | 0.0005 | 0.969 | 15 | 93 | 10 |
| llama31-8b | tracking | 3 | 0.0004 | 0.0024 | -0.0026 | 0.0015 | 0.969 | 5 | 32 | — |

- **llama31-8b**: planned units 40 over 3 families (multivalue 10, retrieval 20, tracking 10); 6-arm GPU time ≈ 1.1 h + model loads; no R5.3 data yet for aggregation, icl, rag, rerank

## tail3x_v3@0.125 (margin 1.25 × FP8 + 0.0002)

| model | family | R5 units | KL/tok design | KL/tok FP8 | mean y | SD y | power/family | n normal | n cons | n sim |
|---|---|---|---|---|---|---|---|---|---|---|
| llama31-8b | multivalue | 3 | 0.0015 | 0.0019 | -0.0009 | 0.0019 | 0.969 | 37 | ∞ | — |
| llama31-8b | retrieval | 6 | 0.0001 | 0.0003 | -0.0003 | 0.0006 | 0.969 | 20 | 147 | 10 |
| llama31-8b | tracking | 3 | 0.0006 | 0.0024 | -0.0025 | 0.0015 | 0.969 | 5 | 37 | — |

- **llama31-8b**: planned units 70 over 3 families (multivalue 40, retrieval 20, tracking 10); 6-arm GPU time ≈ 2.1 h + model loads; no R5.3 data yet for aggregation, icl, rag, rerank

## tail2x_v2@0.125 (margin 1.25 × FP8 + 0.0002)

| model | family | R5 units | KL/tok design | KL/tok FP8 | mean y | SD y | power/family | n normal | n cons | n sim |
|---|---|---|---|---|---|---|---|---|---|---|
| llama31-8b | multivalue | 3 | 0.0104 | 0.0019 | +0.0080 | 0.0048 | 0.969 | ∞ | ∞ | — |
| llama31-8b | retrieval | 6 | 0.0001 | 0.0003 | -0.0002 | 0.0005 | 0.969 | 19 | 135 | 10 |
| llama31-8b | tracking | 3 | 0.0028 | 0.0024 | -0.0002 | 0.0014 | 0.969 | 129 | ∞ | — |

- **llama31-8b**: planned units ∞ over 3 families (multivalue ∞, retrieval 20, tracking 130); 6-arm GPU time ≈ 5.2 h + model loads; no R5.3 data yet for aggregation, icl, rag, rerank

## tail2x_v2@0.25 (margin 1.25 × FP8 + 0.0002)

| model | family | R5 units | KL/tok design | KL/tok FP8 | mean y | SD y | power/family | n normal | n cons | n sim |
|---|---|---|---|---|---|---|---|---|---|---|
| llama31-8b | multivalue | 3 | 0.0059 | 0.0019 | +0.0035 | 0.0032 | 0.969 | ∞ | ∞ | — |
| llama31-8b | retrieval | 6 | 0.0001 | 0.0003 | -0.0002 | 0.0005 | 0.969 | 18 | 118 | 10 |
| llama31-8b | tracking | 3 | 0.0018 | 0.0024 | -0.0013 | 0.0015 | 0.969 | 14 | 380 | — |

- **llama31-8b**: planned units ∞ over 3 families (multivalue ∞, retrieval 20, tracking 20); 6-arm GPU time ≈ 1.1 h + model loads; no R5.3 data yet for aggregation, icl, rag, rerank

## uniform+v4@4 (margin 1.25 × FP8 + 0.0002)

| model | family | R5 units | KL/tok design | KL/tok FP8 | mean y | SD y | power/family | n normal | n cons | n sim |
|---|---|---|---|---|---|---|---|---|---|---|
| llama31-8b | aggregation | 16 | 0.0013 | 0.0007 | +0.0004 | 0.0008 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | icl | 10 | 0.0053 | 0.0035 | +0.0009 | 0.0046 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | multivalue | 20 | 0.0125 | 0.0040 | +0.0075 | 0.0120 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | rag | 10 | 0.0042 | 0.0027 | +0.0008 | 0.0036 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | rerank | 5 | 0.0151 | 0.0078 | +0.0053 | 0.0114 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | retrieval | 40 | 0.0019 | 0.0006 | +0.0012 | 0.0025 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | tracking | 20 | 0.0083 | 0.0026 | +0.0050 | 0.0035 | 0.969 | ∞ | ∞ | ∞ |
| qwen3-30b-a3b-2507 | aggregation | 20 | 0.0094 | 0.0026 | +0.0062 | 0.0077 | 0.946 | ∞ | ∞ | ∞ |
| qwen3-30b-a3b-2507 | multivalue | 20 | 0.0016 | 0.0006 | +0.0008 | 0.0026 | 0.946 | ∞ | ∞ | ∞ |
| qwen3-30b-a3b-2507 | retrieval | 40 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.946 | 5 | 5 | 10 |
| qwen3-30b-a3b-2507 | tracking | 20 | 0.0021 | 0.0013 | +0.0005 | 0.0020 | 0.946 | ∞ | ∞ | ∞ |

- **llama31-8b**: planned units ∞ over 7 families (aggregation ∞, icl ∞, multivalue ∞, rag ∞, rerank ∞, retrieval ∞, tracking ∞); 6-arm GPU time ≈ 0.0 h + model loads
- **qwen3-30b-a3b-2507**: planned units ∞ over 4 families (aggregation ∞, multivalue ∞, retrieval 10, tracking ∞); 6-arm GPU time ≈ 0.1 h + model loads

## kivi4_v4@4 (margin 1.25 × FP8 + 0.0002)

| model | family | R5 units | KL/tok design | KL/tok FP8 | mean y | SD y | power/family | n normal | n cons | n sim |
|---|---|---|---|---|---|---|---|---|---|---|
| llama31-8b | aggregation | 16 | 0.0016 | 0.0007 | +0.0007 | 0.0008 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | icl | 10 | 0.0022 | 0.0035 | -0.0022 | 0.0052 | 0.969 | 58 | 624 | 30 |
| llama31-8b | multivalue | 20 | 0.0098 | 0.0040 | +0.0048 | 0.0087 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | rag | 10 | 0.0042 | 0.0027 | +0.0008 | 0.0018 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | rerank | 5 | 0.0121 | 0.0078 | +0.0023 | 0.0078 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | retrieval | 40 | 0.0022 | 0.0006 | +0.0014 | 0.0044 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | tracking | 20 | 0.0070 | 0.0026 | +0.0037 | 0.0032 | 0.969 | ∞ | ∞ | ∞ |
| qwen3-30b-a3b-2507 | aggregation | 20 | 0.0030 | 0.0026 | -0.0002 | 0.0025 | 0.946 | 425 | ∞ | 480 |
| qwen3-30b-a3b-2507 | multivalue | 20 | 0.0018 | 0.0006 | +0.0010 | 0.0029 | 0.946 | ∞ | ∞ | ∞ |
| qwen3-30b-a3b-2507 | retrieval | 40 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.946 | 5 | 5 | 10 |
| qwen3-30b-a3b-2507 | tracking | 20 | 0.0019 | 0.0013 | +0.0003 | 0.0019 | 0.946 | ∞ | ∞ | ∞ |

- **llama31-8b**: planned units ∞ over 7 families (aggregation ∞, icl 60, multivalue ∞, rag ∞, rerank ∞, retrieval ∞, tracking ∞); 6-arm GPU time ≈ 0.8 h + model loads
- **qwen3-30b-a3b-2507**: planned units ∞ over 4 families (aggregation 480, multivalue ∞, retrieval 10, tracking ∞); 6-arm GPU time ≈ 4.5 h + model loads

## Sensitivity to the margin (normal approximation, point estimates; units per family)

| design | model | family | 1.25 × FP8 + 0.0002 | 1.0 × FP8 + 0.0002 | 1.5 × FP8 + 0.0005 |
|---|---|---|---|---|---|
| qread2t4kqT_v4@0.125 | llama31-8b | aggregation | 80 | 1291 | 11 |
| qread2t4kqT_v4@0.125 | llama31-8b | icl | 30 | 32 | 26 |
| qread2t4kqT_v4@0.125 | llama31-8b | multivalue | 17 | 24 | 13 |
| qread2t4kqT_v4@0.125 | llama31-8b | rag | 7 | 12 | 5 |
| qread2t4kqT_v4@0.125 | llama31-8b | rerank | ∞ | ∞ | 148 |
| qread2t4kqT_v4@0.125 | llama31-8b | retrieval | 196 | 286 | 77 |
| qread2t4kqT_v4@0.125 | llama31-8b | tracking | 5 | 6 | 5 |
| qread2t4kqT_v4@0.125 | qwen3-30b-a3b-2507 | aggregation | 143 | ∞ | 26 |
| qread2t4kqT_v4@0.125 | qwen3-30b-a3b-2507 | multivalue | 4544 | ∞ | 20 |
| qread2t4kqT_v4@0.125 | qwen3-30b-a3b-2507 | retrieval | 5 | 5 | 5 |
| qread2t4kqT_v4@0.125 | qwen3-30b-a3b-2507 | tracking | ∞ | ∞ | ∞ |
| tail4x_v4@0.125 | llama31-8b | multivalue | 5 | 5 | 5 |
| tail4x_v4@0.125 | llama31-8b | retrieval | 15 | 13 | 7 |
| tail4x_v4@0.125 | llama31-8b | tracking | 5 | 5 | 5 |
| tail3x_v3@0.125 | llama31-8b | multivalue | 37 | 104 | 14 |
| tail3x_v3@0.125 | llama31-8b | retrieval | 20 | 18 | 9 |
| tail3x_v3@0.125 | llama31-8b | tracking | 5 | 5 | 5 |
| tail2x_v2@0.125 | llama31-8b | multivalue | ∞ | ∞ | ∞ |
| tail2x_v2@0.125 | llama31-8b | retrieval | 19 | 17 | 8 |
| tail2x_v2@0.125 | llama31-8b | tracking | 129 | ∞ | 19 |
| tail2x_v2@0.25 | llama31-8b | multivalue | ∞ | ∞ | ∞ |
| tail2x_v2@0.25 | llama31-8b | retrieval | 18 | 16 | 8 |
| tail2x_v2@0.25 | llama31-8b | tracking | 14 | 27 | 7 |
| uniform+v4@4 | llama31-8b | aggregation | ∞ | ∞ | 100 |
| uniform+v4@4 | llama31-8b | icl | ∞ | ∞ | 1283 |
| uniform+v4@4 | llama31-8b | multivalue | ∞ | ∞ | ∞ |
| uniform+v4@4 | llama31-8b | rag | ∞ | ∞ | 1130 |
| uniform+v4@4 | llama31-8b | rerank | ∞ | ∞ | ∞ |
| uniform+v4@4 | llama31-8b | retrieval | ∞ | ∞ | ∞ |
| uniform+v4@4 | llama31-8b | tracking | ∞ | ∞ | ∞ |
| uniform+v4@4 | qwen3-30b-a3b-2507 | aggregation | ∞ | ∞ | ∞ |
| uniform+v4@4 | qwen3-30b-a3b-2507 | multivalue | ∞ | ∞ | ∞ |
| uniform+v4@4 | qwen3-30b-a3b-2507 | retrieval | 5 | 5 | 5 |
| uniform+v4@4 | qwen3-30b-a3b-2507 | tracking | ∞ | ∞ | 387 |
| kivi4_v4@4 | llama31-8b | aggregation | ∞ | ∞ | ∞ |
| kivi4_v4@4 | llama31-8b | icl | 58 | 90 | 39 |
| kivi4_v4@4 | llama31-8b | multivalue | ∞ | ∞ | ∞ |
| kivi4_v4@4 | llama31-8b | rag | ∞ | ∞ | 295 |
| kivi4_v4@4 | llama31-8b | rerank | ∞ | ∞ | 79293 |
| kivi4_v4@4 | llama31-8b | retrieval | ∞ | ∞ | ∞ |
| kivi4_v4@4 | llama31-8b | tracking | ∞ | ∞ | ∞ |
| kivi4_v4@4 | qwen3-30b-a3b-2507 | aggregation | 425 | ∞ | 43 |
| kivi4_v4@4 | qwen3-30b-a3b-2507 | multivalue | ∞ | ∞ | ∞ |
| kivi4_v4@4 | qwen3-30b-a3b-2507 | retrieval | 5 | 5 | 5 |
| kivi4_v4@4 | qwen3-30b-a3b-2507 | tracking | ∞ | ∞ | 153 |

## Accuracy floor (design − FP8 on FP-correct units, pooled per model; lower bound > −δ, 80% power)

| model | design | pool | δ | FP-correct units in R5 | units lower / higher | mean Δ | SD Δ | FP-correct units needed |
|---|---|---|---|---|---|---|---|---|
| llama31-8b | qread2t4kqT_v4 | all families | 0.02 | 82 | 2 / 3 | -0.0083 | 0.1531 | 1063 |
| llama31-8b | qread2t4kqT_v4 | all families | 0.05 | 82 | 2 / 3 | -0.0083 | 0.1531 | 84 |
| llama31-8b | qread2t4kqT_v4 | without aggregation | 0.02 | 77 | 0 / 3 | +0.0128 | 0.0768 | 150 |
| llama31-8b | qread2t4kqT_v4 | without aggregation | 0.05 | 77 | 0 / 3 | +0.0128 | 0.0768 | 60 |
| qwen3-30b-a3b-2507 | qread2t4kqT_v4 | all families | 0.02 | 86 | 2 / 0 | -0.0012 | 0.0077 | 150 |
| qwen3-30b-a3b-2507 | qread2t4kqT_v4 | all families | 0.05 | 86 | 2 / 0 | -0.0012 | 0.0077 | 60 |
| qwen3-30b-a3b-2507 | qread2t4kqT_v4 | without aggregation | 0.02 | 78 | 2 / 0 | -0.0013 | 0.0081 | 150 |
| qwen3-30b-a3b-2507 | qread2t4kqT_v4 | without aggregation | 0.05 | 78 | 2 / 0 | -0.0013 | 0.0081 | 60 |

## Seconds per unit for six arms (FP, FP8, the design, dense 4/4, KIVI-4, fp_noise; prefill included)

| model | family | s / unit |
|---|---|---|
| llama31-8b | aggregation | 132 |
| llama31-8b | icl | 49 |
| llama31-8b | multivalue | 125 |
| llama31-8b | rag | 70 |
| llama31-8b | rerank | 121 |
| llama31-8b | retrieval | 62 |
| llama31-8b | tracking | 135 |
| qwen3-30b-a3b-2507 | aggregation | 33 |
| qwen3-30b-a3b-2507 | multivalue | 93 |
| qwen3-30b-a3b-2507 | retrieval | 25 |
| qwen3-30b-a3b-2507 | tracking | 48 |
