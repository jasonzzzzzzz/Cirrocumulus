# R5.4 → R6's arm: units each candidate would need (`size_confirm_r5.py --r54`)

Blocks: h54llama128_r3a_1066612, h54llama128_hm_1066613, h54qwen32_r3a_1066614, h54qwen32_r3b_1066615. Every arm on the same R5.4 units (paired with FP8 within the block). Margin: mean KL/token(arm) < 1.5 × FP8 + 0.0005 (R6's, decided), one-sided α = 0.05, power 0.8 per model (each family at 0.8^(1/K), K = the model's families here). 3–10 units per group, so every count is rough; ∞ = the mean already misses the margin. `fp_noise` is the calibration: where its own count is large the family is noise-limited.

| model | family | units seen | tail4x_v4 | dtail4x_v4 | dtail4ux_v4 | dtail6ux_v4 | dtail8ux_v4 | uniform+v4 | fp_noise |
|---|---|---|---|---|---|---|---|---|---|
| llama31-8b | icl | 10 | 0.0003 / 22, 90, 10 | 0.0004 / 23, 99, 10 | 0.0003 / 22, 91, 10 | 0.0003 / 22, 93, 10 | 0.0004 / 22, 95, 10 | 0.0053 / 1233, ∞, ∞ | 0.0005 / 24, 104, 10 |
| llama31-8b | multivalue | 5 | 0.0006 / 6, 24, 10 | 0.0006 / 6, 26, 10 | 0.0005 / 6, 25, 10 | 0.0003 / 5, 23, 10 | 0.0003 / 6, 24, 10 | 0.0179 / ∞, ∞, ∞ | 0.0006 / 6, 25, 10 |
| llama31-8b | rag | 10 | 0.0012 / 5, 9, 10 | 0.0014 / 5, 12, 10 | 0.0018 / 26, 122, 30 | 0.0011 / 5, 12, 10 | 0.0007 / 5, 10, 10 | 0.0041 / 356005, ∞, ∞ | 0.0005 / 5, 11, 10 |
| llama31-8b | rerank | 5 | 0.0099 / 102, ∞, 80 | 0.0088 / 20, 224, 15 | 0.0074 / 15, 116, 10 | 0.0041 / 8, 40, 10 | 0.0011 / 10, 57, 10 | 0.0151 / ∞, ∞, ∞ | 0.0014 / 10, 57, 10 |
| llama31-8b | retrieval | 10 | 0.0001 / 10, 28, 10 | 0.0001 / 8, 22, 10 | 0.0000 / 9, 25, 10 | 0.0001 / 9, 26, 10 | 0.0000 / 9, 26, 10 | 0.0024 / ∞, ∞, ∞ | 0.0001 / 11, 33, 10 |
| llama31-8b | tracking | 5 | 0.0004 / 5, 7, 10 | 0.0004 / 5, 6, 10 | 0.0004 / 5, 7, 10 | 0.0003 / 5, 7, 10 | 0.0003 / 5, 7, 10 | 0.0106 / ∞, ∞, ∞ | 0.0005 / 5, 7, 10 |
| qwen3-30b-a3b-2507 | aggregation | 10 | 0.0021 / 5, 10, 10 | 0.0037 / 58, 801, 60 | 0.0016 / 7, 17, 10 | 0.0012 / 6, 16, 10 | 0.0008 / 5, 10, 10 | 0.0115 / ∞, ∞, ∞ | 0.0040 / 222, ∞, 240 |
| qwen3-30b-a3b-2507 | multivalue | 5 | 0.0012 / 6521, ∞, ∞ | 0.0001 / 16, 150, 10 | 0.0001 / 14, 123, 10 | 0.0001 / 15, 129, 10 | 0.0001 / 14, 117, 10 | 0.0013 / ∞, ∞, ∞ | 0.0001 / 15, 129, 10 |
| qwen3-30b-a3b-2507 | retrieval | 10 | 0.0000 / 5, 5, 10 | 0.0000 / 5, 5, 10 | 0.0000 / 5, 5, 10 | 0.0000 / 5, 5, 10 | 0.0000 / 5, 5, 10 | 0.0000 / 5, 5, 10 | 0.0000 / 5, 5, 10 |
| qwen3-30b-a3b-2507 | tracking | 5 | 0.0004 / 8, 44, 10 | 0.0013 / 20, 267, 10 | 0.0006 / 9, 51, 10 | 0.0003 / 9, 52, 10 | 0.0012 / 18, 210, 10 | 0.0039 / 43577, ∞, ∞ | 0.0024 / 62, ∞, 40 |

Cell = KL/token / units needed: normal, conservative, simulated. FP8's KL/token per group:

- llama31-8b icl: 0.0035
- llama31-8b multivalue: 0.0058
- llama31-8b rag: 0.0024
- llama31-8b rerank: 0.0078
- llama31-8b retrieval: 0.0005
- llama31-8b tracking: 0.0026
- qwen3-30b-a3b-2507 aggregation: 0.0029
- qwen3-30b-a3b-2507 multivalue: 0.0005
- qwen3-30b-a3b-2507 retrieval: 0.0000
- qwen3-30b-a3b-2507 tracking: 0.0023

## Planned units per model (max of normal and simulated, rounded up to 10)

| arm | model | units | per family | families with no finite size |
|---|---|---|---|---|
| tail4x_v4 | llama31-8b | 180 | icl 30, multivalue 10, rag 10, rerank 110, retrieval 10, tracking 10 | — |
| tail4x_v4 | qwen3-30b-a3b-2507 | ∞ | aggregation 10, multivalue ∞, retrieval 10, tracking 10 | multivalue |
| dtail4x_v4 | llama31-8b | 90 | icl 30, multivalue 10, rag 10, rerank 20, retrieval 10, tracking 10 | — |
| dtail4x_v4 | qwen3-30b-a3b-2507 | 110 | aggregation 60, multivalue 20, retrieval 10, tracking 20 | — |
| dtail4ux_v4 | llama31-8b | 110 | icl 30, multivalue 10, rag 30, rerank 20, retrieval 10, tracking 10 | — |
| dtail4ux_v4 | qwen3-30b-a3b-2507 | 50 | aggregation 10, multivalue 20, retrieval 10, tracking 10 | — |
| dtail6ux_v4 | llama31-8b | 80 | icl 30, multivalue 10, rag 10, rerank 10, retrieval 10, tracking 10 | — |
| dtail6ux_v4 | qwen3-30b-a3b-2507 | 50 | aggregation 10, multivalue 20, retrieval 10, tracking 10 | — |
| dtail8ux_v4 | llama31-8b | 80 | icl 30, multivalue 10, rag 10, rerank 10, retrieval 10, tracking 10 | — |
| dtail8ux_v4 | qwen3-30b-a3b-2507 | 60 | aggregation 10, multivalue 20, retrieval 10, tracking 20 | — |
| uniform+v4 | llama31-8b | ∞ | icl ∞, multivalue ∞, rag ∞, rerank ∞, retrieval ∞, tracking ∞ | icl, multivalue, rag, rerank, retrieval, tracking |
| uniform+v4 | qwen3-30b-a3b-2507 | ∞ | aggregation ∞, multivalue ∞, retrieval 10, tracking ∞ | aggregation, multivalue, tracking |
| fp_noise | llama31-8b | 90 | icl 30, multivalue 10, rag 10, rerank 10, retrieval 20, tracking 10 | — |
| fp_noise | qwen3-30b-a3b-2507 | 340 | aggregation 240, multivalue 20, retrieval 10, tracking 70 | — |

## Accuracy against FP8 on FP-correct units (outside aggregation)

| arm | model | FP-correct units | lower / higher | mean Δ |
|---|---|---|---|---|
| tail4x_v4 | llama31-8b | 27 | 0 / 1 | +0.0028 |
| tail4x_v4 | qwen3-30b-a3b-2507 | 20 | 1 / 0 | -0.0021 |
| dtail4x_v4 | llama31-8b | 27 | 0 / 1 | +0.0028 |
| dtail4x_v4 | qwen3-30b-a3b-2507 | 20 | 0 / 1 | +0.0029 |
| dtail4ux_v4 | llama31-8b | 27 | 0 / 1 | +0.0028 |
| dtail4ux_v4 | qwen3-30b-a3b-2507 | 20 | 0 / 1 | +0.0029 |
| dtail6ux_v4 | llama31-8b | 27 | 0 / 1 | +0.0028 |
| dtail6ux_v4 | qwen3-30b-a3b-2507 | 20 | 0 / 0 | +0.0000 |
| dtail8ux_v4 | llama31-8b | 27 | 0 / 1 | +0.0028 |
| dtail8ux_v4 | qwen3-30b-a3b-2507 | 20 | 0 / 0 | +0.0000 |
| uniform+v4 | llama31-8b | 27 | 0 / 0 | +0.0000 |
| uniform+v4 | qwen3-30b-a3b-2507 | 20 | 1 / 0 | -0.0021 |
| fp_noise | llama31-8b | 27 | 0 / 1 | +0.0028 |
| fp_noise | qwen3-30b-a3b-2507 | 20 | 0 / 1 | +0.0029 |
