# R5 → the confirmatory campaign's size (draft input to plan.md amendment "R6 (draft)")

`size_confirm_r5.py`; R5's valid blocks and R5.3's finished ones (h53llama128_r1_22773130, h53llama128_r3a_1062541, h53llama128_r3b_1062542, h53llama128_hm_1062543, h53qwen32_r2_1062544, h53qwen32_r3a_1062545, h53qwen32_r3b_1062546). Claim per model × family: mean KL/token(design) < 1.25 × mean KL/token(FP8) + 0.0002 (one-sided α = 0.05, unit bootstrap upper bound, paired per unit); all families of a model must pass (intersection–union: no multiplicity correction, so each family is powered at 0.8^(1/K)).
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
| llama31-8b | aggregation | 8 | 0.0006 | 0.0006 | -0.0002 | 0.0007 | 0.969 | 41 | 409 | 40 |
| llama31-8b | icl | 10 | 0.0003 | 0.0035 | -0.0041 | 0.0061 | 0.969 | 25 | 113 | 10 |
| llama31-8b | multivalue | 8 | 0.0005 | 0.0044 | -0.0049 | 0.0044 | 0.969 | 9 | 31 | 10 |
| llama31-8b | rag | 10 | 0.0012 | 0.0024 | -0.0018 | 0.0014 | 0.969 | 6 | 15 | 10 |
| llama31-8b | rerank | 5 | 0.0098 | 0.0078 | +0.0000 | 0.0070 | 0.969 | 15686 | ∞ | ∞ |
| llama31-8b | retrieval | 16 | 0.0001 | 0.0004 | -0.0005 | 0.0008 | 0.969 | 17 | 43 | 10 |
| llama31-8b | tracking | 8 | 0.0004 | 0.0026 | -0.0028 | 0.0013 | 0.969 | 5 | 6 | 10 |
| qwen3-30b-a3b-2507 | aggregation | 10 | 0.0021 | 0.0029 | -0.0015 | 0.0012 | 0.946 | 6 | 15 | 10 |
| qwen3-30b-a3b-2507 | multivalue | 8 | 0.0008 | 0.0003 | +0.0004 | 0.0008 | 0.946 | ∞ | ∞ | ∞ |
| qwen3-30b-a3b-2507 | retrieval | 16 | 0.0000 | 0.0000 | -0.0000 | 0.0000 | 0.946 | 5 | 5 | 10 |
| qwen3-30b-a3b-2507 | tracking | 8 | 0.0005 | 0.0016 | -0.0015 | 0.0020 | 0.946 | 16 | 73 | 10 |

- **llama31-8b**: planned units ∞ over 7 families (aggregation 50, icl 30, multivalue 10, rag 10, rerank ∞, retrieval 20, tracking 10); 6-arm GPU time ≈ 3.5 h + model loads
- **qwen3-30b-a3b-2507**: planned units ∞ over 4 families (aggregation 10, multivalue ∞, retrieval 10, tracking 20); 6-arm GPU time ≈ 0.4 h + model loads

## tail3x_v3@0.125 (margin 1.25 × FP8 + 0.0002)

| model | family | R5 units | KL/tok design | KL/tok FP8 | mean y | SD y | power/family | n normal | n cons | n sim |
|---|---|---|---|---|---|---|---|---|---|---|
| llama31-8b | aggregation | 8 | 0.0020 | 0.0006 | +0.0012 | 0.0008 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | icl | 10 | 0.0021 | 0.0035 | -0.0023 | 0.0048 | 0.969 | 47 | 377 | 30 |
| llama31-8b | multivalue | 8 | 0.0012 | 0.0044 | -0.0043 | 0.0045 | 0.969 | 13 | 49 | 10 |
| llama31-8b | rag | 10 | 0.0057 | 0.0024 | +0.0027 | 0.0102 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | rerank | 5 | 0.0346 | 0.0078 | +0.0248 | 0.0449 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | retrieval | 16 | 0.0001 | 0.0004 | -0.0005 | 0.0008 | 0.969 | 18 | 47 | 10 |
| llama31-8b | tracking | 8 | 0.0008 | 0.0026 | -0.0023 | 0.0013 | 0.969 | 5 | 8 | 10 |
| qwen3-30b-a3b-2507 | aggregation | 10 | 0.0177 | 0.0029 | +0.0141 | 0.0167 | 0.946 | ∞ | ∞ | ∞ |
| qwen3-30b-a3b-2507 | multivalue | 8 | 0.0009 | 0.0003 | +0.0005 | 0.0020 | 0.946 | ∞ | ∞ | ∞ |
| qwen3-30b-a3b-2507 | retrieval | 16 | 0.0000 | 0.0000 | +0.0000 | 0.0001 | 0.946 | 5 | 5 | 10 |
| qwen3-30b-a3b-2507 | tracking | 8 | 0.0029 | 0.0016 | +0.0009 | 0.0036 | 0.946 | ∞ | ∞ | ∞ |

- **llama31-8b**: planned units ∞ over 7 families (aggregation ∞, icl 50, multivalue 20, rag ∞, rerank ∞, retrieval 20, tracking 10); 6-arm GPU time ≈ 2.1 h + model loads
- **qwen3-30b-a3b-2507**: planned units ∞ over 4 families (aggregation ∞, multivalue ∞, retrieval 10, tracking ∞); 6-arm GPU time ≈ 0.1 h + model loads

## tail2x_v2@0.125 (margin 1.25 × FP8 + 0.0002)

| model | family | R5 units | KL/tok design | KL/tok FP8 | mean y | SD y | power/family | n normal | n cons | n sim |
|---|---|---|---|---|---|---|---|---|---|---|
| llama31-8b | aggregation | 8 | 0.0158 | 0.0006 | +0.0151 | 0.0088 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | icl | 10 | 0.0189 | 0.0035 | +0.0146 | 0.0146 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | multivalue | 8 | 0.0112 | 0.0044 | +0.0057 | 0.0068 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | rag | 10 | 0.0453 | 0.0024 | +0.0423 | 0.0942 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | rerank | 5 | 0.1203 | 0.0078 | +0.1106 | 0.1585 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | retrieval | 16 | 0.0003 | 0.0004 | -0.0002 | 0.0009 | 0.969 | 60 | 316 | 60 |
| llama31-8b | tracking | 8 | 0.0049 | 0.0026 | +0.0017 | 0.0021 | 0.969 | ∞ | ∞ | ∞ |
| qwen3-30b-a3b-2507 | aggregation | 10 | 0.2009 | 0.0029 | +0.1973 | 0.3040 | 0.946 | ∞ | ∞ | ∞ |
| qwen3-30b-a3b-2507 | multivalue | 8 | 0.0062 | 0.0003 | +0.0058 | 0.0095 | 0.946 | ∞ | ∞ | ∞ |
| qwen3-30b-a3b-2507 | retrieval | 16 | 0.0001 | 0.0000 | +0.0001 | 0.0001 | 0.946 | 17 | 44 | 20 |
| qwen3-30b-a3b-2507 | tracking | 8 | 0.0284 | 0.0016 | +0.0264 | 0.0132 | 0.946 | ∞ | ∞ | ∞ |

- **llama31-8b**: planned units ∞ over 7 families (aggregation ∞, icl ∞, multivalue ∞, rag ∞, rerank ∞, retrieval 60, tracking ∞); 6-arm GPU time ≈ 1.0 h + model loads
- **qwen3-30b-a3b-2507**: planned units ∞ over 4 families (aggregation ∞, multivalue ∞, retrieval 20, tracking ∞); 6-arm GPU time ≈ 0.1 h + model loads

## tail2x_v2@0.25 (margin 1.25 × FP8 + 0.0002)

| model | family | R5 units | KL/tok design | KL/tok FP8 | mean y | SD y | power/family | n normal | n cons | n sim |
|---|---|---|---|---|---|---|---|---|---|---|
| llama31-8b | aggregation | 8 | 0.0082 | 0.0006 | +0.0074 | 0.0040 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | icl | 10 | 0.0096 | 0.0035 | +0.0052 | 0.0068 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | multivalue | 8 | 0.0056 | 0.0044 | +0.0002 | 0.0051 | 0.969 | 170891 | ∞ | ∞ |
| llama31-8b | rag | 10 | 0.0351 | 0.0024 | +0.0321 | 0.0866 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | rerank | 5 | 0.1066 | 0.0078 | +0.0968 | 0.1616 | 0.969 | ∞ | ∞ | ∞ |
| llama31-8b | retrieval | 16 | 0.0001 | 0.0004 | -0.0004 | 0.0008 | 0.969 | 20 | 55 | 10 |
| llama31-8b | tracking | 8 | 0.0025 | 0.0026 | -0.0007 | 0.0013 | 0.969 | 30 | 213 | 30 |
| qwen3-30b-a3b-2507 | aggregation | 10 | 0.1093 | 0.0029 | +0.1057 | 0.1774 | 0.946 | ∞ | ∞ | ∞ |
| qwen3-30b-a3b-2507 | multivalue | 8 | 0.0081 | 0.0003 | +0.0077 | 0.0108 | 0.946 | ∞ | ∞ | ∞ |
| qwen3-30b-a3b-2507 | retrieval | 16 | 0.0001 | 0.0000 | +0.0001 | 0.0001 | 0.946 | 5 | 8 | 10 |
| qwen3-30b-a3b-2507 | tracking | 8 | 0.0107 | 0.0016 | +0.0087 | 0.0049 | 0.946 | ∞ | ∞ | ∞ |

- **llama31-8b**: planned units ∞ over 7 families (aggregation ∞, icl ∞, multivalue ∞, rag ∞, rerank ∞, retrieval 20, tracking 30); 6-arm GPU time ≈ 1.5 h + model loads
- **qwen3-30b-a3b-2507**: planned units ∞ over 4 families (aggregation ∞, multivalue ∞, retrieval 10, tracking ∞); 6-arm GPU time ≈ 0.1 h + model loads

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
- **qwen3-30b-a3b-2507**: planned units ∞ over 4 families (aggregation 480, multivalue ∞, retrieval 10, tracking ∞); 6-arm GPU time ≈ 4.3 h + model loads

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
| tail4x_v4@0.125 | llama31-8b | aggregation | 41 | 99 | 10 |
| tail4x_v4@0.125 | llama31-8b | icl | 25 | 25 | 22 |
| tail4x_v4@0.125 | llama31-8b | multivalue | 9 | 10 | 9 |
| tail4x_v4@0.125 | llama31-8b | rag | 6 | 9 | 5 |
| tail4x_v4@0.125 | llama31-8b | rerank | 15686 | ∞ | 96 |
| tail4x_v4@0.125 | llama31-8b | retrieval | 17 | 16 | 10 |
| tail4x_v4@0.125 | llama31-8b | tracking | 5 | 5 | 5 |
| tail4x_v4@0.125 | qwen3-30b-a3b-2507 | aggregation | 6 | 8 | 5 |
| tail4x_v4@0.125 | qwen3-30b-a3b-2507 | multivalue | ∞ | ∞ | 94 |
| tail4x_v4@0.125 | qwen3-30b-a3b-2507 | retrieval | 5 | 5 | 5 |
| tail4x_v4@0.125 | qwen3-30b-a3b-2507 | tracking | 16 | 17 | 12 |
| tail3x_v3@0.125 | llama31-8b | aggregation | ∞ | ∞ | ∞ |
| tail3x_v3@0.125 | llama31-8b | icl | 47 | 68 | 33 |
| tail3x_v3@0.125 | llama31-8b | multivalue | 13 | 15 | 11 |
| tail3x_v3@0.125 | llama31-8b | rag | ∞ | ∞ | ∞ |
| tail3x_v3@0.125 | llama31-8b | rerank | ∞ | ∞ | ∞ |
| tail3x_v3@0.125 | llama31-8b | retrieval | 18 | 17 | 10 |
| tail3x_v3@0.125 | llama31-8b | tracking | 5 | 5 | 5 |
| tail3x_v3@0.125 | qwen3-30b-a3b-2507 | aggregation | ∞ | ∞ | ∞ |
| tail3x_v3@0.125 | qwen3-30b-a3b-2507 | multivalue | ∞ | ∞ | 5937 |
| tail3x_v3@0.125 | qwen3-30b-a3b-2507 | retrieval | 5 | 5 | 5 |
| tail3x_v3@0.125 | qwen3-30b-a3b-2507 | tracking | ∞ | ∞ | ∞ |
| tail2x_v2@0.125 | llama31-8b | aggregation | ∞ | ∞ | ∞ |
| tail2x_v2@0.125 | llama31-8b | icl | ∞ | ∞ | ∞ |
| tail2x_v2@0.125 | llama31-8b | multivalue | ∞ | ∞ | ∞ |
| tail2x_v2@0.125 | llama31-8b | rag | ∞ | ∞ | ∞ |
| tail2x_v2@0.125 | llama31-8b | rerank | ∞ | ∞ | ∞ |
| tail2x_v2@0.125 | llama31-8b | retrieval | 60 | 83 | 20 |
| tail2x_v2@0.125 | llama31-8b | tracking | ∞ | ∞ | ∞ |
| tail2x_v2@0.125 | qwen3-30b-a3b-2507 | aggregation | ∞ | ∞ | ∞ |
| tail2x_v2@0.125 | qwen3-30b-a3b-2507 | multivalue | ∞ | ∞ | ∞ |
| tail2x_v2@0.125 | qwen3-30b-a3b-2507 | retrieval | 17 | 17 | 5 |
| tail2x_v2@0.125 | qwen3-30b-a3b-2507 | tracking | ∞ | ∞ | ∞ |
| tail2x_v2@0.25 | llama31-8b | aggregation | ∞ | ∞ | ∞ |
| tail2x_v2@0.25 | llama31-8b | icl | ∞ | ∞ | ∞ |
| tail2x_v2@0.25 | llama31-8b | multivalue | 170891 | ∞ | 205 |
| tail2x_v2@0.25 | llama31-8b | rag | ∞ | ∞ | ∞ |
| tail2x_v2@0.25 | llama31-8b | rerank | ∞ | ∞ | ∞ |
| tail2x_v2@0.25 | llama31-8b | retrieval | 20 | 19 | 11 |
| tail2x_v2@0.25 | llama31-8b | tracking | 30 | 333 | 9 |
| tail2x_v2@0.25 | qwen3-30b-a3b-2507 | aggregation | ∞ | ∞ | ∞ |
| tail2x_v2@0.25 | qwen3-30b-a3b-2507 | multivalue | ∞ | ∞ | ∞ |
| tail2x_v2@0.25 | qwen3-30b-a3b-2507 | retrieval | 5 | 5 | 5 |
| tail2x_v2@0.25 | qwen3-30b-a3b-2507 | tracking | ∞ | ∞ | ∞ |
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
| llama31-8b | aggregation | 128 |
| llama31-8b | icl | 48 |
| llama31-8b | multivalue | 129 |
| llama31-8b | rag | 69 |
| llama31-8b | rerank | 114 |
| llama31-8b | retrieval | 62 |
| llama31-8b | tracking | 141 |
| qwen3-30b-a3b-2507 | aggregation | 32 |
| qwen3-30b-a3b-2507 | multivalue | 91 |
| qwen3-30b-a3b-2507 | retrieval | 25 |
| qwen3-30b-a3b-2507 | tracking | 47 |
