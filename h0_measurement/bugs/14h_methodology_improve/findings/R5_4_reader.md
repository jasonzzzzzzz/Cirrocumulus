# R14 Stage 1h — R5.4 read: the randomized tier 1 against tail4x_v4 (rules: plan.md, amendment 'R5.4')

Unit-clustered bootstrap 90% intervals; paired per unit; each model × cell × family separately.

- every block valid

## Main read (75 units)

### Q1 randomized vs Lloyd-Max tier (4-bit, plain)

| model / cell / family | arm | against | Δ KL/tok | Δ dP/tok | Δ accuracy | traffic (arm / ref) |
|---|---|---|---|---|---|---|
| llama31-8b/h54llama128_hm/icl | dtail4x_v4@0.125 | tail4x_v4@0.125 | 0.0001 [-0.0002, 0.0003] (n=10) | 0.0057 [0.0009, 0.0104] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.344 |
| llama31-8b/h54llama128_hm/rag | dtail4x_v4@0.125 | tail4x_v4@0.125 | 0.0002 [-0.0000, 0.0004] (n=10) | -0.0048 [-0.0143, 0.0037] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.344 |
| llama31-8b/h54llama128_hm/rerank | dtail4x_v4@0.125 | tail4x_v4@0.125 | -0.0011 [-0.0049, 0.0022] (n=5) | 0.0049 [-0.0162, 0.0245] (n=5) | 0.029 [-0.001, 0.090] (n=5) | 0.344 / 0.344 |
| llama31-8b/h54llama128_r3a/multivalue | dtail4x_v4@0.125 | tail4x_v4@0.125 | 0.0001 [-0.0002, 0.0003] (n=5) | -0.0029 [-0.0067, 0.0006] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.344 / 0.344 |
| llama31-8b/h54llama128_r3a/retrieval | dtail4x_v4@0.125 | tail4x_v4@0.125 | 0.0000 [-0.0001, 0.0001] (n=10) | -0.0003 [-0.0018, 0.0008] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.344 |
| llama31-8b/h54llama128_r3a/tracking | dtail4x_v4@0.125 | tail4x_v4@0.125 | 0.0000 [-0.0001, 0.0001] (n=5) | -0.0006 [-0.0026, 0.0022] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.344 / 0.344 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | dtail4x_v4@0.125 | tail4x_v4@0.125 | -0.0011 [-0.0029, -0.0001] (n=5) | -0.0010 [-0.0028, 0.0000] (n=5) | 0.008 [0.000, 0.025] (n=5) | 0.344 / 0.344 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | dtail4x_v4@0.125 | tail4x_v4@0.125 | -0.0000 [-0.0000, 0.0000] (n=10) | -0.0000 [-0.0001, 0.0000] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.344 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | dtail4x_v4@0.125 | tail4x_v4@0.125 | 0.0009 [0.0000, 0.0019] (n=5) | 0.0010 [-0.0018, 0.0044] (n=5) | 0.012 [0.000, 0.024] (n=5) | 0.344 / 0.344 |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | dtail4x_v4@0.125 | tail4x_v4@0.125 | 0.0015 [0.0007, 0.0024] (n=10) | 0.0029 [-0.0017, 0.0074] (n=10) | 0.010 [0.000, 0.030] (n=10) | 0.344 / 0.344 |

### Q2 unbiased vs plain (randomized, 4-bit)

| model / cell / family | arm | against | Δ KL/tok | Δ dP/tok | Δ accuracy | traffic (arm / ref) |
|---|---|---|---|---|---|---|
| llama31-8b/h54llama128_hm/icl | dtail4ux_v4@0.125 | dtail4x_v4@0.125 | -0.0001 [-0.0003, 0.0001] (n=10) | -0.0026 [-0.0054, -0.0002] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.344 |
| llama31-8b/h54llama128_hm/rag | dtail4ux_v4@0.125 | dtail4x_v4@0.125 | 0.0005 [-0.0004, 0.0018] (n=10) | 0.0062 [0.0004, 0.0146] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.344 |
| llama31-8b/h54llama128_hm/rerank | dtail4ux_v4@0.125 | dtail4x_v4@0.125 | -0.0014 [-0.0023, -0.0005] (n=5) | -0.0040 [-0.0074, -0.0015] (n=5) | -0.013 [-0.040, 0.000] (n=5) | 0.344 / 0.344 |
| llama31-8b/h54llama128_r3a/multivalue | dtail4ux_v4@0.125 | dtail4x_v4@0.125 | -0.0002 [-0.0004, 0.0000] (n=5) | 0.0002 [-0.0027, 0.0039] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.344 / 0.344 |
| llama31-8b/h54llama128_r3a/retrieval | dtail4ux_v4@0.125 | dtail4x_v4@0.125 | -0.0000 [-0.0001, 0.0000] (n=10) | 0.0001 [-0.0011, 0.0014] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.344 |
| llama31-8b/h54llama128_r3a/tracking | dtail4ux_v4@0.125 | dtail4x_v4@0.125 | 0.0000 [-0.0001, 0.0001] (n=5) | -0.0001 [-0.0024, 0.0020] (n=5) | -0.046 [-0.138, 0.000] (n=5) | 0.344 / 0.344 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | dtail4ux_v4@0.125 | dtail4x_v4@0.125 | -0.0000 [-0.0001, -0.0000] (n=5) | -0.0000 [-0.0000, 0.0000] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.344 / 0.344 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | dtail4ux_v4@0.125 | dtail4x_v4@0.125 | -0.0000 [-0.0000, -0.0000] (n=10) | -0.0000 [-0.0000, -0.0000] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.344 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | dtail4ux_v4@0.125 | dtail4x_v4@0.125 | -0.0007 [-0.0018, 0.0001] (n=5) | -0.0026 [-0.0051, -0.0006] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.344 / 0.344 |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | dtail4ux_v4@0.125 | dtail4x_v4@0.125 | -0.0021 [-0.0029, -0.0013] (n=10) | 0.0004 [-0.0041, 0.0052] (n=10) | -0.010 [-0.030, 0.000] (n=10) | 0.344 / 0.344 |

### Q2b the unbiased randomized tail vs R6's candidate

| model / cell / family | arm | against | Δ KL/tok | Δ dP/tok | Δ accuracy | traffic (arm / ref) |
|---|---|---|---|---|---|---|
| llama31-8b/h54llama128_hm/icl | dtail4ux_v4@0.125 | tail4x_v4@0.125 | -0.0000 [-0.0003, 0.0001] (n=10) | 0.0031 [-0.0010, 0.0075] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.344 |
| llama31-8b/h54llama128_hm/rag | dtail4ux_v4@0.125 | tail4x_v4@0.125 | 0.0006 [-0.0003, 0.0022] (n=10) | 0.0014 [-0.0041, 0.0069] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.344 |
| llama31-8b/h54llama128_hm/rerank | dtail4ux_v4@0.125 | tail4x_v4@0.125 | -0.0025 [-0.0062, 0.0011] (n=5) | 0.0009 [-0.0189, 0.0177] (n=5) | 0.016 [-0.001, 0.050] (n=5) | 0.344 / 0.344 |
| llama31-8b/h54llama128_r3a/multivalue | dtail4ux_v4@0.125 | tail4x_v4@0.125 | -0.0001 [-0.0003, 0.0000] (n=5) | -0.0026 [-0.0034, -0.0019] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.344 / 0.344 |
| llama31-8b/h54llama128_r3a/retrieval | dtail4ux_v4@0.125 | tail4x_v4@0.125 | -0.0000 [-0.0001, 0.0000] (n=10) | -0.0003 [-0.0009, 0.0004] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.344 |
| llama31-8b/h54llama128_r3a/tracking | dtail4ux_v4@0.125 | tail4x_v4@0.125 | 0.0000 [-0.0001, 0.0002] (n=5) | -0.0007 [-0.0041, 0.0027] (n=5) | -0.046 [-0.138, 0.000] (n=5) | 0.344 / 0.344 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | dtail4ux_v4@0.125 | tail4x_v4@0.125 | -0.0011 [-0.0029, -0.0001] (n=5) | -0.0011 [-0.0028, 0.0000] (n=5) | 0.008 [0.000, 0.025] (n=5) | 0.344 / 0.344 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | dtail4ux_v4@0.125 | tail4x_v4@0.125 | -0.0000 [-0.0000, -0.0000] (n=10) | -0.0000 [-0.0001, -0.0000] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.344 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | dtail4ux_v4@0.125 | tail4x_v4@0.125 | 0.0002 [-0.0001, 0.0006] (n=5) | -0.0016 [-0.0032, -0.0001] (n=5) | 0.012 [0.000, 0.024] (n=5) | 0.344 / 0.344 |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | dtail4ux_v4@0.125 | tail4x_v4@0.125 | -0.0005 [-0.0011, 0.0001] (n=10) | 0.0033 [0.0002, 0.0064] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.344 |

### Q3 key bits (unbiased)

| model / cell / family | arm | against | Δ KL/tok | Δ dP/tok | Δ accuracy | traffic (arm / ref) |
|---|---|---|---|---|---|---|
| llama31-8b/h54llama128_hm/icl | dtail6ux_v4@0.125 | dtail4ux_v4@0.125 | 0.0001 [-0.0001, 0.0003] (n=10) | 0.0026 [-0.0003, 0.0056] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.398 / 0.344 |
| llama31-8b/h54llama128_hm/icl | dtail8ux_v4@0.125 | dtail6ux_v4@0.125 | 0.0001 [-0.0002, 0.0003] (n=10) | -0.0021 [-0.0062, 0.0018] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.453 / 0.398 |
| llama31-8b/h54llama128_hm/rag | dtail6ux_v4@0.125 | dtail4ux_v4@0.125 | -0.0007 [-0.0021, 0.0001] (n=10) | -0.0024 [-0.0116, 0.0075] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.398 / 0.344 |
| llama31-8b/h54llama128_hm/rag | dtail8ux_v4@0.125 | dtail6ux_v4@0.125 | -0.0005 [-0.0015, 0.0002] (n=10) | -0.0061 [-0.0154, 0.0011] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.453 / 0.398 |
| llama31-8b/h54llama128_hm/rerank | dtail6ux_v4@0.125 | dtail4ux_v4@0.125 | -0.0032 [-0.0062, -0.0006] (n=5) | -0.0039 [-0.0144, 0.0062] (n=5) | 0.018 [-0.006, 0.043] (n=5) | 0.398 / 0.344 |
| llama31-8b/h54llama128_hm/rerank | dtail8ux_v4@0.125 | dtail6ux_v4@0.125 | -0.0031 [-0.0062, -0.0004] (n=5) | -0.0045 [-0.0109, 0.0008] (n=5) | -0.024 [-0.048, 0.000] (n=5) | 0.453 / 0.398 |
| llama31-8b/h54llama128_r3a/multivalue | dtail6ux_v4@0.125 | dtail4ux_v4@0.125 | -0.0001 [-0.0003, 0.0000] (n=5) | 0.0040 [0.0008, 0.0085] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.398 / 0.344 |
| llama31-8b/h54llama128_r3a/multivalue | dtail8ux_v4@0.125 | dtail6ux_v4@0.125 | -0.0000 [-0.0001, 0.0000] (n=5) | 0.0010 [-0.0009, 0.0031] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.453 / 0.398 |
| llama31-8b/h54llama128_r3a/retrieval | dtail6ux_v4@0.125 | dtail4ux_v4@0.125 | 0.0000 [-0.0000, 0.0000] (n=10) | -0.0007 [-0.0015, -0.0002] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.398 / 0.344 |
| llama31-8b/h54llama128_r3a/retrieval | dtail8ux_v4@0.125 | dtail6ux_v4@0.125 | -0.0000 [-0.0000, 0.0000] (n=10) | 0.0006 [0.0000, 0.0013] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.453 / 0.398 |
| llama31-8b/h54llama128_r3a/tracking | dtail6ux_v4@0.125 | dtail4ux_v4@0.125 | -0.0001 [-0.0002, -0.0000] (n=5) | -0.0016 [-0.0039, 0.0008] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.398 / 0.344 |
| llama31-8b/h54llama128_r3a/tracking | dtail8ux_v4@0.125 | dtail6ux_v4@0.125 | -0.0000 [-0.0001, 0.0000] (n=5) | -0.0008 [-0.0030, 0.0014] (n=5) | -0.015 [-0.046, 0.000] (n=5) | 0.453 / 0.398 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | dtail6ux_v4@0.125 | dtail4ux_v4@0.125 | 0.0000 [0.0000, 0.0001] (n=5) | -0.0001 [-0.0002, 0.0001] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.398 / 0.344 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | dtail8ux_v4@0.125 | dtail6ux_v4@0.125 | -0.0000 [-0.0001, 0.0000] (n=5) | -0.0000 [-0.0002, 0.0001] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.453 / 0.398 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | dtail6ux_v4@0.125 | dtail4ux_v4@0.125 | 0.0000 [-0.0000, 0.0000] (n=10) | -0.0000 [-0.0001, 0.0000] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.398 / 0.344 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | dtail8ux_v4@0.125 | dtail6ux_v4@0.125 | -0.0000 [-0.0000, 0.0000] (n=10) | 0.0001 [-0.0000, 0.0002] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.453 / 0.398 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | dtail6ux_v4@0.125 | dtail4ux_v4@0.125 | -0.0003 [-0.0006, 0.0000] (n=5) | 0.0008 [0.0000, 0.0017] (n=5) | -0.012 [-0.024, 0.000] (n=5) | 0.398 / 0.344 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | dtail8ux_v4@0.125 | dtail6ux_v4@0.125 | 0.0009 [0.0004, 0.0015] (n=5) | 0.0010 [-0.0005, 0.0023] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.453 / 0.398 |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | dtail6ux_v4@0.125 | dtail4ux_v4@0.125 | -0.0004 [-0.0007, -0.0001] (n=10) | 0.0007 [-0.0022, 0.0041] (n=10) | 0.000 [-0.040, 0.040] (n=10) | 0.398 / 0.344 |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | dtail8ux_v4@0.125 | dtail6ux_v4@0.125 | -0.0004 [-0.0008, -0.0000] (n=10) | 0.0017 [-0.0024, 0.0050] (n=10) | 0.020 [0.000, 0.060] (n=10) | 0.453 / 0.398 |

### Q4 against FP8

| model / cell / family | arm | against | Δ KL/tok | Δ dP/tok | Δ accuracy | traffic (arm / ref) |
|---|---|---|---|---|---|---|
| llama31-8b/h54llama128_hm/icl | tail4x_v4@0.125 | fp8kv@8 | -0.0032 [-0.0057, -0.0011] (n=10) | 0.0154 [-0.0027, 0.0378] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.500 |
| llama31-8b/h54llama128_hm/icl | dtail4x_v4@0.125 | fp8kv@8 | -0.0031 [-0.0058, -0.0010] (n=10) | 0.0211 [0.0040, 0.0422] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.500 |
| llama31-8b/h54llama128_hm/icl | dtail4ux_v4@0.125 | fp8kv@8 | -0.0032 [-0.0058, -0.0011] (n=10) | 0.0186 [0.0007, 0.0397] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.500 |
| llama31-8b/h54llama128_hm/icl | dtail6ux_v4@0.125 | fp8kv@8 | -0.0032 [-0.0057, -0.0010] (n=10) | 0.0212 [0.0038, 0.0420] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.398 / 0.500 |
| llama31-8b/h54llama128_hm/icl | dtail8ux_v4@0.125 | fp8kv@8 | -0.0031 [-0.0057, -0.0010] (n=10) | 0.0191 [0.0006, 0.0407] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.453 / 0.500 |
| llama31-8b/h54llama128_hm/icl | fp_noise@0 | fp8kv@8 | -0.0030 [-0.0056, -0.0008] (n=10) | 0.0199 [0.0012, 0.0413] (n=10) | 0.000 [0.000, 0.000] (n=10) | 1.000 / 0.500 |
| llama31-8b/h54llama128_hm/rag | tail4x_v4@0.125 | fp8kv@8 | -0.0012 [-0.0018, -0.0006] (n=10) | 0.0012 [-0.0110, 0.0152] (n=10) | -0.100 [-0.300, 0.000] (n=10) | 0.344 / 0.500 |
| llama31-8b/h54llama128_hm/rag | dtail4x_v4@0.125 | fp8kv@8 | -0.0011 [-0.0017, -0.0003] (n=10) | -0.0036 [-0.0182, 0.0089] (n=10) | -0.100 [-0.300, 0.000] (n=10) | 0.344 / 0.500 |
| llama31-8b/h54llama128_hm/rag | dtail4ux_v4@0.125 | fp8kv@8 | -0.0006 [-0.0020, 0.0013] (n=10) | 0.0026 [-0.0063, 0.0123] (n=10) | -0.100 [-0.300, 0.000] (n=10) | 0.344 / 0.500 |
| llama31-8b/h54llama128_hm/rag | dtail6ux_v4@0.125 | fp8kv@8 | -0.0013 [-0.0020, -0.0005] (n=10) | 0.0003 [-0.0080, 0.0071] (n=10) | -0.100 [-0.300, 0.000] (n=10) | 0.398 / 0.500 |
| llama31-8b/h54llama128_hm/rag | dtail8ux_v4@0.125 | fp8kv@8 | -0.0018 [-0.0024, -0.0012] (n=10) | -0.0058 [-0.0159, 0.0038] (n=10) | -0.100 [-0.300, 0.000] (n=10) | 0.453 / 0.500 |
| llama31-8b/h54llama128_hm/rag | fp_noise@0 | fp8kv@8 | -0.0020 [-0.0027, -0.0013] (n=10) | -0.0017 [-0.0149, 0.0114] (n=10) | -0.100 [-0.300, 0.000] (n=10) | 1.000 / 0.500 |
| llama31-8b/h54llama128_hm/rerank | tail4x_v4@0.125 | fp8kv@8 | 0.0021 [-0.0019, 0.0062] (n=5) | -0.0048 [-0.0145, 0.0050] (n=5) | -0.023 [-0.047, 0.000] (n=5) | 0.344 / 0.500 |
| llama31-8b/h54llama128_hm/rerank | dtail4x_v4@0.125 | fp8kv@8 | 0.0010 [-0.0009, 0.0029] (n=5) | 0.0002 [-0.0282, 0.0254] (n=5) | 0.006 [-0.046, 0.065] (n=5) | 0.344 / 0.500 |
| llama31-8b/h54llama128_hm/rerank | dtail4ux_v4@0.125 | fp8kv@8 | -0.0004 [-0.0024, 0.0015] (n=5) | -0.0039 [-0.0308, 0.0209] (n=5) | -0.007 [-0.046, 0.032] (n=5) | 0.344 / 0.500 |
| llama31-8b/h54llama128_hm/rerank | dtail6ux_v4@0.125 | fp8kv@8 | -0.0037 [-0.0060, -0.0018] (n=5) | -0.0078 [-0.0257, 0.0078] (n=5) | 0.012 [-0.003, 0.038] (n=5) | 0.398 / 0.500 |
| llama31-8b/h54llama128_hm/rerank | dtail8ux_v4@0.125 | fp8kv@8 | -0.0067 [-0.0114, -0.0021] (n=5) | -0.0123 [-0.0337, 0.0067] (n=5) | -0.012 [-0.025, 0.000] (n=5) | 0.453 / 0.500 |
| llama31-8b/h54llama128_hm/rerank | fp_noise@0 | fp8kv@8 | -0.0064 [-0.0109, -0.0020] (n=5) | -0.0081 [-0.0294, 0.0106] (n=5) | -0.023 [-0.046, 0.000] (n=5) | 1.000 / 0.500 |
| llama31-8b/h54llama128_r3a/multivalue | tail4x_v4@0.125 | fp8kv@8 | -0.0053 [-0.0080, -0.0031] (n=5) | 0.0034 [-0.0060, 0.0180] (n=5) | -0.013 [-0.062, 0.025] (n=5) | 0.344 / 0.500 |
| llama31-8b/h54llama128_r3a/multivalue | dtail4x_v4@0.125 | fp8kv@8 | -0.0052 [-0.0080, -0.0029] (n=5) | 0.0005 [-0.0074, 0.0100] (n=5) | -0.013 [-0.062, 0.025] (n=5) | 0.344 / 0.500 |
| llama31-8b/h54llama128_r3a/multivalue | dtail4ux_v4@0.125 | fp8kv@8 | -0.0054 [-0.0083, -0.0031] (n=5) | 0.0007 [-0.0088, 0.0150] (n=5) | -0.013 [-0.062, 0.025] (n=5) | 0.344 / 0.500 |
| llama31-8b/h54llama128_r3a/multivalue | dtail6ux_v4@0.125 | fp8kv@8 | -0.0055 [-0.0083, -0.0032] (n=5) | 0.0048 [-0.0070, 0.0237] (n=5) | -0.013 [-0.062, 0.025] (n=5) | 0.398 / 0.500 |
| llama31-8b/h54llama128_r3a/multivalue | dtail8ux_v4@0.125 | fp8kv@8 | -0.0056 [-0.0084, -0.0031] (n=5) | 0.0058 [-0.0070, 0.0273] (n=5) | -0.013 [-0.062, 0.025] (n=5) | 0.453 / 0.500 |
| llama31-8b/h54llama128_r3a/multivalue | fp_noise@0 | fp8kv@8 | -0.0053 [-0.0082, -0.0029] (n=5) | 0.0040 [-0.0056, 0.0181] (n=5) | -0.013 [-0.062, 0.025] (n=5) | 1.000 / 0.500 |
| llama31-8b/h54llama128_r3a/retrieval | tail4x_v4@0.125 | fp8kv@8 | -0.0005 [-0.0009, -0.0001] (n=10) | -0.0008 [-0.0047, 0.0022] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.500 |
| llama31-8b/h54llama128_r3a/retrieval | dtail4x_v4@0.125 | fp8kv@8 | -0.0005 [-0.0008, -0.0002] (n=10) | -0.0011 [-0.0066, 0.0028] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.500 |
| llama31-8b/h54llama128_r3a/retrieval | dtail4ux_v4@0.125 | fp8kv@8 | -0.0005 [-0.0009, -0.0002] (n=10) | -0.0010 [-0.0052, 0.0017] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.500 |
| llama31-8b/h54llama128_r3a/retrieval | dtail6ux_v4@0.125 | fp8kv@8 | -0.0005 [-0.0009, -0.0002] (n=10) | -0.0018 [-0.0059, 0.0010] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.398 / 0.500 |
| llama31-8b/h54llama128_r3a/retrieval | dtail8ux_v4@0.125 | fp8kv@8 | -0.0005 [-0.0009, -0.0002] (n=10) | -0.0012 [-0.0053, 0.0018] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.453 / 0.500 |
| llama31-8b/h54llama128_r3a/retrieval | fp_noise@0 | fp8kv@8 | -0.0004 [-0.0008, -0.0001] (n=10) | -0.0007 [-0.0047, 0.0027] (n=10) | 0.000 [0.000, 0.000] (n=10) | 1.000 / 0.500 |
| llama31-8b/h54llama128_r3a/tracking | tail4x_v4@0.125 | fp8kv@8 | -0.0022 [-0.0028, -0.0014] (n=5) | -0.0001 [-0.0055, 0.0057] (n=5) | 0.108 [0.000, 0.277] (n=5) | 0.344 / 0.500 |
| llama31-8b/h54llama128_r3a/tracking | dtail4x_v4@0.125 | fp8kv@8 | -0.0022 [-0.0028, -0.0015] (n=5) | -0.0006 [-0.0056, 0.0046] (n=5) | 0.108 [0.000, 0.277] (n=5) | 0.344 / 0.500 |
| llama31-8b/h54llama128_r3a/tracking | dtail4ux_v4@0.125 | fp8kv@8 | -0.0022 [-0.0028, -0.0014] (n=5) | -0.0007 [-0.0054, 0.0065] (n=5) | 0.062 [0.000, 0.138] (n=5) | 0.344 / 0.500 |
| llama31-8b/h54llama128_r3a/tracking | dtail6ux_v4@0.125 | fp8kv@8 | -0.0023 [-0.0029, -0.0015] (n=5) | -0.0023 [-0.0072, 0.0030] (n=5) | 0.062 [0.000, 0.138] (n=5) | 0.398 / 0.500 |
| llama31-8b/h54llama128_r3a/tracking | dtail8ux_v4@0.125 | fp8kv@8 | -0.0023 [-0.0030, -0.0015] (n=5) | -0.0031 [-0.0088, 0.0034] (n=5) | 0.046 [0.000, 0.092] (n=5) | 0.453 / 0.500 |
| llama31-8b/h54llama128_r3a/tracking | fp_noise@0 | fp8kv@8 | -0.0021 [-0.0028, -0.0014] (n=5) | -0.0014 [-0.0062, 0.0056] (n=5) | 0.108 [0.000, 0.277] (n=5) | 1.000 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | tail4x_v4@0.125 | fp8kv@8 | 0.0007 [0.0001, 0.0017] (n=5) | 0.0006 [-0.0002, 0.0015] (n=5) | -0.008 [-0.025, 0.000] (n=5) | 0.344 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | dtail4x_v4@0.125 | fp8kv@8 | -0.0004 [-0.0011, 0.0001] (n=5) | -0.0005 [-0.0011, -0.0000] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.344 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | dtail4ux_v4@0.125 | fp8kv@8 | -0.0004 [-0.0012, 0.0000] (n=5) | -0.0005 [-0.0012, -0.0001] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.344 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | dtail6ux_v4@0.125 | fp8kv@8 | -0.0004 [-0.0011, 0.0001] (n=5) | -0.0005 [-0.0011, -0.0002] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.398 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | dtail8ux_v4@0.125 | fp8kv@8 | -0.0004 [-0.0012, 0.0000] (n=5) | -0.0005 [-0.0013, -0.0000] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.453 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | fp_noise@0 | fp8kv@8 | -0.0004 [-0.0012, 0.0000] (n=5) | -0.0004 [-0.0010, -0.0000] (n=5) | 0.000 [0.000, 0.000] (n=5) | 1.000 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | tail4x_v4@0.125 | fp8kv@8 | 0.0000 [-0.0000, 0.0000] (n=10) | 0.0000 [0.0000, 0.0000] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | dtail4x_v4@0.125 | fp8kv@8 | -0.0000 [-0.0000, 0.0000] (n=10) | -0.0000 [-0.0001, 0.0000] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | dtail4ux_v4@0.125 | fp8kv@8 | -0.0000 [-0.0000, -0.0000] (n=10) | -0.0000 [-0.0001, 0.0000] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | dtail6ux_v4@0.125 | fp8kv@8 | 0.0000 [-0.0000, 0.0000] (n=10) | -0.0000 [-0.0002, 0.0000] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.398 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | dtail8ux_v4@0.125 | fp8kv@8 | -0.0000 [-0.0000, 0.0000] (n=10) | 0.0000 [0.0000, 0.0000] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.453 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | fp_noise@0 | fp8kv@8 | -0.0000 [-0.0000, 0.0000] (n=10) | -0.0000 [-0.0001, 0.0000] (n=10) | 0.000 [0.000, 0.000] (n=10) | 1.000 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | tail4x_v4@0.125 | fp8kv@8 | -0.0020 [-0.0033, -0.0008] (n=5) | -0.0020 [-0.0059, 0.0012] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.344 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | dtail4x_v4@0.125 | fp8kv@8 | -0.0011 [-0.0029, 0.0007] (n=5) | -0.0010 [-0.0069, 0.0048] (n=5) | 0.012 [0.000, 0.024] (n=5) | 0.344 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | dtail4ux_v4@0.125 | fp8kv@8 | -0.0018 [-0.0030, -0.0006] (n=5) | -0.0036 [-0.0084, 0.0012] (n=5) | 0.012 [0.000, 0.024] (n=5) | 0.344 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | dtail6ux_v4@0.125 | fp8kv@8 | -0.0021 [-0.0036, -0.0007] (n=5) | -0.0028 [-0.0079, 0.0019] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.398 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | dtail8ux_v4@0.125 | fp8kv@8 | -0.0011 [-0.0028, 0.0004] (n=5) | -0.0018 [-0.0061, 0.0021] (n=5) | 0.000 [0.000, 0.000] (n=5) | 0.453 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | fp_noise@0 | fp8kv@8 | 0.0000 [-0.0022, 0.0019] (n=5) | -0.0027 [-0.0079, 0.0019] (n=5) | 0.012 [0.000, 0.024] (n=5) | 1.000 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | tail4x_v4@0.125 | fp8kv@8 | -0.0007 [-0.0012, -0.0004] (n=10) | -0.0095 [-0.0158, -0.0036] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | dtail4x_v4@0.125 | fp8kv@8 | 0.0008 [-0.0002, 0.0018] (n=10) | -0.0066 [-0.0164, 0.0021] (n=10) | 0.010 [0.000, 0.030] (n=10) | 0.344 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | dtail4ux_v4@0.125 | fp8kv@8 | -0.0013 [-0.0020, -0.0005] (n=10) | -0.0062 [-0.0129, -0.0006] (n=10) | 0.000 [0.000, 0.000] (n=10) | 0.344 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | dtail6ux_v4@0.125 | fp8kv@8 | -0.0017 [-0.0026, -0.0008] (n=10) | -0.0055 [-0.0098, -0.0012] (n=10) | 0.000 [-0.040, 0.040] (n=10) | 0.398 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | dtail8ux_v4@0.125 | fp8kv@8 | -0.0021 [-0.0029, -0.0013] (n=10) | -0.0038 [-0.0086, 0.0010] (n=10) | 0.020 [0.000, 0.060] (n=10) | 0.453 / 0.500 |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | fp_noise@0 | fp8kv@8 | 0.0011 [-0.0005, 0.0031] (n=10) | -0.0037 [-0.0095, 0.0018] (n=10) | 0.020 [0.000, 0.060] (n=10) | 1.000 / 0.500 |

### R6's draft test at 1.25 x FP8 + 0.0002 (point estimate / 90% upper bound of mean(KL − ρ·KL_FP8) ≤ κ)

| model | family | arm | n | mean | upper | point | upper |
|---|---|---|---|---|---|---|---|
| llama31-8b | icl | tail4x_v4 | 10 | -0.00408 | -0.00143 | yes | yes |
| llama31-8b | icl | dtail4x_v4 | 10 | -0.00402 | -0.00128 | yes | yes |
| llama31-8b | icl | dtail4ux_v4 | 10 | -0.00412 | -0.00139 | yes | yes |
| llama31-8b | icl | dtail6ux_v4 | 10 | -0.00406 | -0.00132 | yes | yes |
| llama31-8b | icl | dtail8ux_v4 | 10 | -0.00400 | -0.00131 | yes | yes |
| llama31-8b | icl | uniform+v4 | 10 | +0.00093 | +0.00319 | no | no |
| llama31-8b | icl | fp_noise | 10 | -0.00389 | -0.00115 | yes | yes |
| llama31-8b | multivalue | tail4x_v4 | 5 | -0.00674 | -0.00398 | yes | yes |
| llama31-8b | multivalue | dtail4x_v4 | 5 | -0.00668 | -0.00372 | yes | yes |
| llama31-8b | multivalue | dtail4ux_v4 | 5 | -0.00685 | -0.00401 | yes | yes |
| llama31-8b | multivalue | dtail6ux_v4 | 5 | -0.00698 | -0.00406 | yes | yes |
| llama31-8b | multivalue | dtail8ux_v4 | 5 | -0.00703 | -0.00401 | yes | yes |
| llama31-8b | multivalue | uniform+v4 | 5 | +0.01062 | +0.01894 | no | no |
| llama31-8b | multivalue | fp_noise | 5 | -0.00673 | -0.00378 | yes | yes |
| llama31-8b | rag | tail4x_v4 | 10 | -0.00181 | -0.00113 | yes | yes |
| llama31-8b | rag | dtail4x_v4 | 10 | -0.00166 | -0.00086 | yes | yes |
| llama31-8b | rag | dtail4ux_v4 | 10 | -0.00117 | +0.00065 | yes | no |
| llama31-8b | rag | dtail6ux_v4 | 10 | -0.00189 | -0.00106 | yes | yes |
| llama31-8b | rag | dtail8ux_v4 | 10 | -0.00237 | -0.00161 | yes | yes |
| llama31-8b | rag | uniform+v4 | 10 | +0.00108 | +0.00279 | no | no |
| llama31-8b | rag | fp_noise | 10 | -0.00257 | -0.00170 | yes | yes |
| llama31-8b | rerank | tail4x_v4 | 5 | +0.00012 | +0.00426 | yes | no |
| llama31-8b | rerank | dtail4x_v4 | 5 | -0.00100 | +0.00122 | yes | no |
| llama31-8b | rerank | dtail4ux_v4 | 5 | -0.00240 | +0.00013 | yes | yes |
| llama31-8b | rerank | dtail6ux_v4 | 5 | -0.00564 | -0.00253 | yes | yes |
| llama31-8b | rerank | dtail8ux_v4 | 5 | -0.00870 | -0.00281 | yes | yes |
| llama31-8b | rerank | uniform+v4 | 5 | +0.00527 | +0.01266 | no | no |
| llama31-8b | rerank | fp_noise | 5 | -0.00840 | -0.00268 | yes | yes |
| llama31-8b | retrieval | tail4x_v4 | 10 | -0.00060 | -0.00020 | yes | yes |
| llama31-8b | retrieval | dtail4x_v4 | 10 | -0.00059 | -0.00024 | yes | yes |
| llama31-8b | retrieval | dtail4ux_v4 | 10 | -0.00062 | -0.00023 | yes | yes |
| llama31-8b | retrieval | dtail6ux_v4 | 10 | -0.00061 | -0.00023 | yes | yes |
| llama31-8b | retrieval | dtail8ux_v4 | 10 | -0.00063 | -0.00023 | yes | yes |
| llama31-8b | retrieval | uniform+v4 | 10 | +0.00178 | +0.00345 | no | no |
| llama31-8b | retrieval | fp_noise | 10 | -0.00053 | -0.00013 | yes | yes |
| llama31-8b | tracking | tail4x_v4 | 5 | -0.00282 | -0.00190 | yes | yes |
| llama31-8b | tracking | dtail4x_v4 | 5 | -0.00281 | -0.00190 | yes | yes |
| llama31-8b | tracking | dtail4ux_v4 | 5 | -0.00280 | -0.00188 | yes | yes |
| llama31-8b | tracking | dtail6ux_v4 | 5 | -0.00292 | -0.00199 | yes | yes |
| llama31-8b | tracking | dtail8ux_v4 | 5 | -0.00294 | -0.00194 | yes | yes |
| llama31-8b | tracking | uniform+v4 | 5 | +0.00741 | +0.01085 | no | no |
| llama31-8b | tracking | fp_noise | 5 | -0.00278 | -0.00190 | yes | yes |
| qwen3-30b-a3b-2507 | aggregation | tail4x_v4 | 10 | -0.00146 | -0.00090 | yes | yes |
| qwen3-30b-a3b-2507 | aggregation | dtail4x_v4 | 10 | +0.00009 | +0.00122 | yes | no |
| qwen3-30b-a3b-2507 | aggregation | dtail4ux_v4 | 10 | -0.00197 | -0.00098 | yes | yes |
| qwen3-30b-a3b-2507 | aggregation | dtail6ux_v4 | 10 | -0.00241 | -0.00129 | yes | yes |
| qwen3-30b-a3b-2507 | aggregation | dtail8ux_v4 | 10 | -0.00281 | -0.00187 | yes | yes |
| qwen3-30b-a3b-2507 | aggregation | uniform+v4 | 10 | +0.00790 | +0.01241 | no | no |
| qwen3-30b-a3b-2507 | aggregation | fp_noise | 10 | +0.00041 | +0.00236 | no | no |
| qwen3-30b-a3b-2507 | multivalue | tail4x_v4 | 5 | +0.00059 | +0.00139 | no | no |
| qwen3-30b-a3b-2507 | multivalue | dtail4x_v4 | 5 | -0.00048 | +0.00010 | yes | yes |
| qwen3-30b-a3b-2507 | multivalue | dtail4ux_v4 | 5 | -0.00053 | +0.00002 | yes | yes |
| qwen3-30b-a3b-2507 | multivalue | dtail6ux_v4 | 5 | -0.00049 | +0.00005 | yes | yes |
| qwen3-30b-a3b-2507 | multivalue | dtail8ux_v4 | 5 | -0.00053 | -0.00000 | yes | yes |
| qwen3-30b-a3b-2507 | multivalue | uniform+v4 | 5 | +0.00074 | +0.00233 | no | no |
| qwen3-30b-a3b-2507 | multivalue | fp_noise | 5 | -0.00053 | +0.00002 | yes | yes |
| qwen3-30b-a3b-2507 | retrieval | tail4x_v4 | 10 | -0.00000 | +0.00000 | yes | yes |
| qwen3-30b-a3b-2507 | retrieval | dtail4x_v4 | 10 | -0.00000 | +0.00000 | yes | yes |
| qwen3-30b-a3b-2507 | retrieval | dtail4ux_v4 | 10 | -0.00000 | -0.00000 | yes | yes |
| qwen3-30b-a3b-2507 | retrieval | dtail6ux_v4 | 10 | +0.00000 | +0.00000 | yes | yes |
| qwen3-30b-a3b-2507 | retrieval | dtail8ux_v4 | 10 | -0.00000 | -0.00000 | yes | yes |
| qwen3-30b-a3b-2507 | retrieval | uniform+v4 | 10 | +0.00001 | +0.00003 | yes | yes |
| qwen3-30b-a3b-2507 | retrieval | fp_noise | 10 | -0.00000 | -0.00000 | yes | yes |
| qwen3-30b-a3b-2507 | tracking | tail4x_v4 | 5 | -0.00255 | -0.00105 | yes | yes |
| qwen3-30b-a3b-2507 | tracking | dtail4x_v4 | 5 | -0.00168 | +0.00030 | yes | no |
| qwen3-30b-a3b-2507 | tracking | dtail4ux_v4 | 5 | -0.00237 | -0.00085 | yes | yes |
| qwen3-30b-a3b-2507 | tracking | dtail6ux_v4 | 5 | -0.00265 | -0.00097 | yes | yes |
| qwen3-30b-a3b-2507 | tracking | dtail8ux_v4 | 5 | -0.00173 | +0.00016 | yes | yes |
| qwen3-30b-a3b-2507 | tracking | uniform+v4 | 5 | +0.00100 | +0.00428 | no | no |
| qwen3-30b-a3b-2507 | tracking | fp_noise | 5 | -0.00057 | +0.00150 | yes | no |

### R6's draft test at 1.5 x FP8 + 0.0005 (point estimate / 90% upper bound of mean(KL − ρ·KL_FP8) ≤ κ)

| model | family | arm | n | mean | upper | point | upper |
|---|---|---|---|---|---|---|---|
| llama31-8b | icl | tail4x_v4 | 10 | -0.00496 | -0.00177 | yes | yes |
| llama31-8b | icl | dtail4x_v4 | 10 | -0.00490 | -0.00161 | yes | yes |
| llama31-8b | icl | dtail4ux_v4 | 10 | -0.00500 | -0.00172 | yes | yes |
| llama31-8b | icl | dtail6ux_v4 | 10 | -0.00494 | -0.00166 | yes | yes |
| llama31-8b | icl | dtail8ux_v4 | 10 | -0.00488 | -0.00164 | yes | yes |
| llama31-8b | icl | uniform+v4 | 10 | +0.00005 | +0.00224 | yes | no |
| llama31-8b | icl | fp_noise | 10 | -0.00477 | -0.00149 | yes | yes |
| llama31-8b | multivalue | tail4x_v4 | 5 | -0.00820 | -0.00486 | yes | yes |
| llama31-8b | multivalue | dtail4x_v4 | 5 | -0.00814 | -0.00459 | yes | yes |
| llama31-8b | multivalue | dtail4ux_v4 | 5 | -0.00831 | -0.00491 | yes | yes |
| llama31-8b | multivalue | dtail6ux_v4 | 5 | -0.00844 | -0.00494 | yes | yes |
| llama31-8b | multivalue | dtail8ux_v4 | 5 | -0.00849 | -0.00489 | yes | yes |
| llama31-8b | multivalue | uniform+v4 | 5 | +0.00916 | +0.01772 | no | no |
| llama31-8b | multivalue | fp_noise | 5 | -0.00819 | -0.00468 | yes | yes |
| llama31-8b | rag | tail4x_v4 | 10 | -0.00242 | -0.00165 | yes | yes |
| llama31-8b | rag | dtail4x_v4 | 10 | -0.00226 | -0.00141 | yes | yes |
| llama31-8b | rag | dtail4ux_v4 | 10 | -0.00178 | +0.00000 | yes | yes |
| llama31-8b | rag | dtail6ux_v4 | 10 | -0.00250 | -0.00159 | yes | yes |
| llama31-8b | rag | dtail8ux_v4 | 10 | -0.00298 | -0.00202 | yes | yes |
| llama31-8b | rag | uniform+v4 | 10 | +0.00048 | +0.00220 | yes | no |
| llama31-8b | rag | fp_noise | 10 | -0.00317 | -0.00213 | yes | yes |
| llama31-8b | rerank | tail4x_v4 | 5 | -0.00183 | +0.00221 | yes | no |
| llama31-8b | rerank | dtail4x_v4 | 5 | -0.00296 | -0.00002 | yes | yes |
| llama31-8b | rerank | dtail4ux_v4 | 5 | -0.00436 | -0.00101 | yes | yes |
| llama31-8b | rerank | dtail6ux_v4 | 5 | -0.00760 | -0.00323 | yes | yes |
| llama31-8b | rerank | dtail8ux_v4 | 5 | -0.01066 | -0.00349 | yes | yes |
| llama31-8b | rerank | uniform+v4 | 5 | +0.00331 | +0.01010 | no | no |
| llama31-8b | rerank | fp_noise | 5 | -0.01036 | -0.00339 | yes | yes |
| llama31-8b | retrieval | tail4x_v4 | 10 | -0.00073 | -0.00025 | yes | yes |
| llama31-8b | retrieval | dtail4x_v4 | 10 | -0.00072 | -0.00029 | yes | yes |
| llama31-8b | retrieval | dtail4ux_v4 | 10 | -0.00075 | -0.00028 | yes | yes |
| llama31-8b | retrieval | dtail6ux_v4 | 10 | -0.00074 | -0.00029 | yes | yes |
| llama31-8b | retrieval | dtail8ux_v4 | 10 | -0.00076 | -0.00029 | yes | yes |
| llama31-8b | retrieval | uniform+v4 | 10 | +0.00165 | +0.00334 | no | no |
| llama31-8b | retrieval | fp_noise | 10 | -0.00067 | -0.00019 | yes | yes |
| llama31-8b | tracking | tail4x_v4 | 5 | -0.00346 | -0.00236 | yes | yes |
| llama31-8b | tracking | dtail4x_v4 | 5 | -0.00345 | -0.00236 | yes | yes |
| llama31-8b | tracking | dtail4ux_v4 | 5 | -0.00344 | -0.00233 | yes | yes |
| llama31-8b | tracking | dtail6ux_v4 | 5 | -0.00357 | -0.00245 | yes | yes |
| llama31-8b | tracking | dtail8ux_v4 | 5 | -0.00359 | -0.00239 | yes | yes |
| llama31-8b | tracking | uniform+v4 | 5 | +0.00676 | +0.01022 | no | no |
| llama31-8b | tracking | fp_noise | 5 | -0.00343 | -0.00236 | yes | yes |
| qwen3-30b-a3b-2507 | aggregation | tail4x_v4 | 10 | -0.00217 | -0.00141 | yes | yes |
| qwen3-30b-a3b-2507 | aggregation | dtail4x_v4 | 10 | -0.00062 | +0.00065 | yes | no |
| qwen3-30b-a3b-2507 | aggregation | dtail4ux_v4 | 10 | -0.00268 | -0.00149 | yes | yes |
| qwen3-30b-a3b-2507 | aggregation | dtail6ux_v4 | 10 | -0.00312 | -0.00182 | yes | yes |
| qwen3-30b-a3b-2507 | aggregation | dtail8ux_v4 | 10 | -0.00352 | -0.00238 | yes | yes |
| qwen3-30b-a3b-2507 | aggregation | uniform+v4 | 10 | +0.00719 | +0.01155 | no | no |
| qwen3-30b-a3b-2507 | aggregation | fp_noise | 10 | -0.00031 | +0.00163 | yes | no |
| qwen3-30b-a3b-2507 | multivalue | tail4x_v4 | 5 | +0.00047 | +0.00109 | yes | no |
| qwen3-30b-a3b-2507 | multivalue | dtail4x_v4 | 5 | -0.00060 | +0.00008 | yes | yes |
| qwen3-30b-a3b-2507 | multivalue | dtail4ux_v4 | 5 | -0.00065 | +0.00000 | yes | yes |
| qwen3-30b-a3b-2507 | multivalue | dtail6ux_v4 | 5 | -0.00061 | +0.00003 | yes | yes |
| qwen3-30b-a3b-2507 | multivalue | dtail8ux_v4 | 5 | -0.00065 | -0.00002 | yes | yes |
| qwen3-30b-a3b-2507 | multivalue | uniform+v4 | 5 | +0.00062 | +0.00221 | no | no |
| qwen3-30b-a3b-2507 | multivalue | fp_noise | 5 | -0.00065 | +0.00000 | yes | yes |
| qwen3-30b-a3b-2507 | retrieval | tail4x_v4 | 10 | -0.00000 | -0.00000 | yes | yes |
| qwen3-30b-a3b-2507 | retrieval | dtail4x_v4 | 10 | -0.00000 | +0.00000 | yes | yes |
| qwen3-30b-a3b-2507 | retrieval | dtail4ux_v4 | 10 | -0.00000 | -0.00000 | yes | yes |
| qwen3-30b-a3b-2507 | retrieval | dtail6ux_v4 | 10 | -0.00000 | +0.00000 | yes | yes |
| qwen3-30b-a3b-2507 | retrieval | dtail8ux_v4 | 10 | -0.00000 | -0.00000 | yes | yes |
| qwen3-30b-a3b-2507 | retrieval | uniform+v4 | 10 | +0.00001 | +0.00002 | yes | yes |
| qwen3-30b-a3b-2507 | retrieval | fp_noise | 10 | -0.00000 | -0.00000 | yes | yes |
| qwen3-30b-a3b-2507 | tracking | tail4x_v4 | 5 | -0.00314 | -0.00129 | yes | yes |
| qwen3-30b-a3b-2507 | tracking | dtail4x_v4 | 5 | -0.00226 | +0.00014 | yes | yes |
| qwen3-30b-a3b-2507 | tracking | dtail4ux_v4 | 5 | -0.00296 | -0.00110 | yes | yes |
| qwen3-30b-a3b-2507 | tracking | dtail6ux_v4 | 5 | -0.00324 | -0.00122 | yes | yes |
| qwen3-30b-a3b-2507 | tracking | dtail8ux_v4 | 5 | -0.00232 | -0.00009 | yes | yes |
| qwen3-30b-a3b-2507 | tracking | uniform+v4 | 5 | +0.00041 | +0.00389 | yes | no |
| qwen3-30b-a3b-2507 | tracking | fp_noise | 5 | -0.00116 | +0.00114 | yes | no |

### Theorem R in the probe (dithered tier; read sets from the Lloyd-Max store; FP's teacher-forced queries)

Read = rows read exactly (tailx = the vote's; _p128/_p32/_p8 = C/128, C/32, C/8 more), p = plain / u = unbiased tail, k = key bits. Coverage is a guarantee: it must be 1.

| model | cell | read | head-steps | coverage (min) | median bound / error | certified within FP8 | rel. error |
|---|---|---|---|---|---|---|---|
| llama31-8b | h54llama128_hm | tailx_p128_p_k6 | 304128 | 1.0000 (1.000) | 16.2 | 0.893 | 0.0003 |
| llama31-8b | h54llama128_hm | tailx_p128_p_k8 | 304128 | 1.0000 (1.000) | 7.9 | 0.988 | 0.0002 |
| llama31-8b | h54llama128_hm | tailx_p128_u_k6 | 304128 | 1.0000 (1.000) | 15.7 | 0.918 | 0.0003 |
| llama31-8b | h54llama128_hm | tailx_p128_u_k8 | 304128 | 1.0000 (1.000) | 7.8 | 0.988 | 0.0002 |
| llama31-8b | h54llama128_hm | tailx_p32_p_k6 | 304128 | 1.0000 (1.000) | 13.1 | 0.983 | 0.0001 |
| llama31-8b | h54llama128_hm | tailx_p32_p_k8 | 304128 | 1.0000 (1.000) | 7.1 | 0.999 | 0.0001 |
| llama31-8b | h54llama128_hm | tailx_p32_u_k6 | 304128 | 1.0000 (1.000) | 12.6 | 0.991 | 0.0001 |
| llama31-8b | h54llama128_hm | tailx_p32_u_k8 | 304128 | 1.0000 (1.000) | 6.9 | 0.999 | 0.0001 |
| llama31-8b | h54llama128_hm | tailx_p8_p_k6 | 304128 | 1.0000 (1.000) | 11.1 | 0.999 | 0.0000 |
| llama31-8b | h54llama128_hm | tailx_p8_p_k8 | 304128 | 1.0000 (1.000) | 6.5 | 1.000 | 0.0000 |
| llama31-8b | h54llama128_hm | tailx_p8_u_k6 | 304128 | 1.0000 (1.000) | 10.8 | 1.000 | 0.0000 |
| llama31-8b | h54llama128_hm | tailx_p8_u_k8 | 304128 | 1.0000 (1.000) | 6.4 | 1.000 | 0.0000 |
| llama31-8b | h54llama128_hm | tailx_p_k6 | 304128 | 1.0000 (1.000) | 38.2 | 0.443 | 0.0007 |
| llama31-8b | h54llama128_hm | tailx_p_k8 | 304128 | 1.0000 (1.000) | 15.1 | 0.700 | 0.0006 |
| llama31-8b | h54llama128_hm | tailx_u_k6 | 304128 | 1.0000 (1.000) | 37.8 | 0.461 | 0.0007 |
| llama31-8b | h54llama128_hm | tailx_u_k8 | 304128 | 1.0000 (1.000) | 15.0 | 0.701 | 0.0006 |
| llama31-8b | h54llama128_r3a | tailx_p128_p_k6 | 354304 | 1.0000 (1.000) | 14.9 | 0.944 | 0.0002 |
| llama31-8b | h54llama128_r3a | tailx_p128_p_k8 | 354304 | 1.0000 (1.000) | 7.7 | 0.993 | 0.0001 |
| llama31-8b | h54llama128_r3a | tailx_p128_u_k6 | 354304 | 1.0000 (1.000) | 14.7 | 0.963 | 0.0002 |
| llama31-8b | h54llama128_r3a | tailx_p128_u_k8 | 354304 | 1.0000 (1.000) | 7.5 | 0.993 | 0.0001 |
| llama31-8b | h54llama128_r3a | tailx_p32_p_k6 | 354304 | 1.0000 (1.000) | 12.1 | 0.988 | 0.0001 |
| llama31-8b | h54llama128_r3a | tailx_p32_p_k8 | 354304 | 1.0000 (1.000) | 6.9 | 0.998 | 0.0001 |
| llama31-8b | h54llama128_r3a | tailx_p32_u_k6 | 354304 | 1.0000 (1.000) | 11.9 | 0.994 | 0.0001 |
| llama31-8b | h54llama128_r3a | tailx_p32_u_k8 | 354304 | 1.0000 (1.000) | 6.7 | 0.998 | 0.0001 |
| llama31-8b | h54llama128_r3a | tailx_p8_p_k6 | 354304 | 1.0000 (1.000) | 10.3 | 0.999 | 0.0000 |
| llama31-8b | h54llama128_r3a | tailx_p8_p_k8 | 354304 | 1.0000 (1.000) | 6.4 | 1.000 | 0.0000 |
| llama31-8b | h54llama128_r3a | tailx_p8_u_k6 | 354304 | 1.0000 (1.000) | 10.2 | 1.000 | 0.0000 |
| llama31-8b | h54llama128_r3a | tailx_p8_u_k8 | 354304 | 1.0000 (1.000) | 6.2 | 1.000 | 0.0000 |
| llama31-8b | h54llama128_r3a | tailx_p_k6 | 354304 | 1.0000 (1.000) | 37.2 | 0.504 | 0.0005 |
| llama31-8b | h54llama128_r3a | tailx_p_k8 | 354304 | 1.0000 (1.000) | 15.0 | 0.793 | 0.0004 |
| llama31-8b | h54llama128_r3a | tailx_u_k6 | 354304 | 1.0000 (1.000) | 37.2 | 0.524 | 0.0005 |
| llama31-8b | h54llama128_r3a | tailx_u_k8 | 354304 | 1.0000 (1.000) | 14.9 | 0.794 | 0.0004 |
| qwen3-30b-a3b-2507 | h54qwen32_r3a | tailx_p128_p_k6 | 806400 | 1.0000 (1.000) | 33.0 | 0.600 | 0.0005 |
| qwen3-30b-a3b-2507 | h54qwen32_r3a | tailx_p128_p_k8 | 806400 | 1.0000 (1.000) | 13.2 | 0.848 | 0.0003 |
| qwen3-30b-a3b-2507 | h54qwen32_r3a | tailx_p128_u_k6 | 806400 | 1.0000 (1.000) | 35.4 | 0.616 | 0.0004 |
| qwen3-30b-a3b-2507 | h54qwen32_r3a | tailx_p128_u_k8 | 806400 | 1.0000 (1.000) | 13.1 | 0.851 | 0.0003 |
| qwen3-30b-a3b-2507 | h54qwen32_r3a | tailx_p32_p_k6 | 806400 | 1.0000 (1.000) | 26.0 | 0.709 | 0.0003 |
| qwen3-30b-a3b-2507 | h54qwen32_r3a | tailx_p32_p_k8 | 806400 | 1.0000 (1.000) | 11.3 | 0.914 | 0.0002 |
| qwen3-30b-a3b-2507 | h54qwen32_r3a | tailx_p32_u_k6 | 806400 | 1.0000 (1.000) | 28.7 | 0.726 | 0.0002 |
| qwen3-30b-a3b-2507 | h54qwen32_r3a | tailx_p32_u_k8 | 806400 | 1.0000 (1.000) | 11.1 | 0.916 | 0.0002 |
| qwen3-30b-a3b-2507 | h54qwen32_r3a | tailx_p8_p_k6 | 806400 | 1.0000 (1.000) | 20.5 | 0.822 | 0.0001 |
| qwen3-30b-a3b-2507 | h54qwen32_r3a | tailx_p8_p_k8 | 806400 | 1.0000 (1.000) | 9.7 | 0.970 | 0.0001 |
| qwen3-30b-a3b-2507 | h54qwen32_r3a | tailx_p8_u_k6 | 806400 | 1.0000 (1.000) | 23.4 | 0.836 | 0.0001 |
| qwen3-30b-a3b-2507 | h54qwen32_r3a | tailx_p8_u_k8 | 806400 | 1.0000 (1.000) | 9.6 | 0.972 | 0.0001 |
| qwen3-30b-a3b-2507 | h54qwen32_r3a | tailx_p_k6 | 806400 | 1.0000 (1.000) | 55.8 | 0.375 | 0.0009 |
| qwen3-30b-a3b-2507 | h54qwen32_r3a | tailx_p_k8 | 806400 | 1.0000 (1.000) | 20.1 | 0.646 | 0.0006 |
| qwen3-30b-a3b-2507 | h54qwen32_r3a | tailx_u_k6 | 806400 | 1.0000 (1.000) | 58.4 | 0.385 | 0.0008 |
| qwen3-30b-a3b-2507 | h54qwen32_r3a | tailx_u_k8 | 806400 | 1.0000 (1.000) | 20.0 | 0.648 | 0.0006 |
| qwen3-30b-a3b-2507 | h54qwen32_r3b | tailx_p128_p_k6 | 477696 | 1.0000 (1.000) | 26.1 | 0.571 | 0.0007 |
| qwen3-30b-a3b-2507 | h54qwen32_r3b | tailx_p128_p_k8 | 477696 | 1.0000 (1.000) | 12.1 | 0.825 | 0.0004 |
| qwen3-30b-a3b-2507 | h54qwen32_r3b | tailx_p128_u_k6 | 477696 | 1.0000 (1.000) | 29.1 | 0.589 | 0.0005 |
| qwen3-30b-a3b-2507 | h54qwen32_r3b | tailx_p128_u_k8 | 477696 | 1.0000 (1.000) | 12.0 | 0.827 | 0.0004 |
| qwen3-30b-a3b-2507 | h54qwen32_r3b | tailx_p32_p_k6 | 477696 | 1.0000 (1.000) | 20.9 | 0.693 | 0.0004 |
| qwen3-30b-a3b-2507 | h54qwen32_r3b | tailx_p32_p_k8 | 477696 | 1.0000 (1.000) | 10.4 | 0.908 | 0.0002 |
| qwen3-30b-a3b-2507 | h54qwen32_r3b | tailx_p32_u_k6 | 477696 | 1.0000 (1.000) | 24.1 | 0.713 | 0.0003 |
| qwen3-30b-a3b-2507 | h54qwen32_r3b | tailx_p32_u_k8 | 477696 | 1.0000 (1.000) | 10.4 | 0.911 | 0.0002 |
| qwen3-30b-a3b-2507 | h54qwen32_r3b | tailx_p8_p_k6 | 477696 | 1.0000 (1.000) | 16.6 | 0.830 | 0.0002 |
| qwen3-30b-a3b-2507 | h54qwen32_r3b | tailx_p8_p_k8 | 477696 | 1.0000 (1.000) | 9.1 | 0.976 | 0.0001 |
| qwen3-30b-a3b-2507 | h54qwen32_r3b | tailx_p8_u_k6 | 477696 | 1.0000 (1.000) | 19.8 | 0.848 | 0.0001 |
| qwen3-30b-a3b-2507 | h54qwen32_r3b | tailx_p8_u_k8 | 477696 | 1.0000 (1.000) | 9.0 | 0.977 | 0.0001 |
| qwen3-30b-a3b-2507 | h54qwen32_r3b | tailx_p_k6 | 477696 | 1.0000 (1.000) | 41.2 | 0.401 | 0.0010 |
| qwen3-30b-a3b-2507 | h54qwen32_r3b | tailx_p_k8 | 477696 | 1.0000 (1.000) | 16.9 | 0.652 | 0.0006 |
| qwen3-30b-a3b-2507 | h54qwen32_r3b | tailx_u_k6 | 477696 | 1.0000 (1.000) | 45.0 | 0.413 | 0.0008 |
| qwen3-30b-a3b-2507 | h54qwen32_r3b | tailx_u_k8 | 477696 | 1.0000 (1.000) | 16.9 | 0.654 | 0.0006 |

### The accuracy–cost frontier (against FP, per span token; sorted by traffic)

| model / cell / family | arm@B | traffic | KL/tok | dP/tok | ΔAcc (FP right) |
|---|---|---|---|---|---|
| llama31-8b/h54llama128_hm/icl | uniform+v4@4 | 0.250 | 0.0053 [0.0018, 0.0097] (n=10) | -0.0167 [-0.0458, 0.0060] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_hm/icl | dtail4ux_v4@0.125 | 0.344 | 0.0003 [0.0001, 0.0004] (n=10) | -0.0001 [-0.0035, 0.0034] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_hm/icl | dtail4x_v4@0.125 | 0.344 | 0.0004 [0.0002, 0.0006] (n=10) | 0.0025 [-0.0002, 0.0060] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_hm/icl | tail4x_v4@0.125 | 0.344 | 0.0003 [0.0001, 0.0005] (n=10) | -0.0032 [-0.0074, 0.0005] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_hm/icl | dtail6ux_v4@0.125 | 0.398 | 0.0003 [0.0001, 0.0005] (n=10) | 0.0025 [-0.0012, 0.0065] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_hm/icl | dtail8ux_v4@0.125 | 0.453 | 0.0004 [0.0001, 0.0007] (n=10) | 0.0005 [-0.0037, 0.0048] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_hm/icl | fp8kv@8 | 0.500 | 0.0035 [0.0013, 0.0060] (n=10) | -0.0186 [-0.0405, -0.0011] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_hm/icl | uniform@4 | 0.625 | 0.0041 [0.0017, 0.0066] (n=10) | -0.0160 [-0.0372, 0.0025] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_hm/icl | fp_noise@0 | 1.000 | 0.0005 [0.0002, 0.0008] (n=10) | 0.0013 [-0.0033, 0.0061] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_hm/rag | uniform+v4@4 | 0.250 | 0.0041 [0.0023, 0.0061] (n=10) | 0.0251 [0.0093, 0.0441] (n=10) | 0.000 [0.000, 0.000] (n=4) |
| llama31-8b/h54llama128_hm/rag | dtail4ux_v4@0.125 | 0.344 | 0.0018 [0.0004, 0.0043] (n=10) | 0.0075 [0.0023, 0.0132] (n=10) | 0.000 [0.000, 0.000] (n=4) |
| llama31-8b/h54llama128_hm/rag | dtail4x_v4@0.125 | 0.344 | 0.0014 [0.0006, 0.0025] (n=10) | 0.0013 [-0.0045, 0.0072] (n=10) | 0.000 [0.000, 0.000] (n=4) |
| llama31-8b/h54llama128_hm/rag | tail4x_v4@0.125 | 0.344 | 0.0012 [0.0006, 0.0021] (n=10) | 0.0061 [0.0011, 0.0112] (n=10) | 0.000 [0.000, 0.000] (n=4) |
| llama31-8b/h54llama128_hm/rag | dtail6ux_v4@0.125 | 0.398 | 0.0011 [0.0004, 0.0022] (n=10) | 0.0052 [-0.0060, 0.0171] (n=10) | 0.000 [0.000, 0.000] (n=4) |
| llama31-8b/h54llama128_hm/rag | dtail8ux_v4@0.125 | 0.453 | 0.0007 [0.0005, 0.0008] (n=10) | -0.0009 [-0.0085, 0.0050] (n=10) | 0.000 [0.000, 0.000] (n=4) |
| llama31-8b/h54llama128_hm/rag | fp8kv@8 | 0.500 | 0.0024 [0.0017, 0.0032] (n=10) | 0.0049 [-0.0083, 0.0187] (n=10) | 0.000 [0.000, 0.000] (n=4) |
| llama31-8b/h54llama128_hm/rag | uniform@4 | 0.625 | 0.0039 [0.0022, 0.0057] (n=10) | 0.0249 [0.0096, 0.0441] (n=10) | 0.000 [0.000, 0.000] (n=4) |
| llama31-8b/h54llama128_hm/rag | fp_noise@0 | 1.000 | 0.0005 [0.0003, 0.0006] (n=10) | 0.0032 [0.0008, 0.0059] (n=10) | 0.000 [0.000, 0.000] (n=4) |
| llama31-8b/h54llama128_hm/rerank | uniform+v4@4 | 0.250 | 0.0151 [0.0057, 0.0264] (n=5) | 0.0111 [0.0003, 0.0235] (n=5) | — |
| llama31-8b/h54llama128_hm/rerank | dtail4ux_v4@0.125 | 0.344 | 0.0074 [0.0035, 0.0117] (n=5) | 0.0028 [-0.0053, 0.0119] (n=5) | — |
| llama31-8b/h54llama128_hm/rerank | dtail4x_v4@0.125 | 0.344 | 0.0088 [0.0040, 0.0139] (n=5) | 0.0068 [-0.0025, 0.0173] (n=5) | — |
| llama31-8b/h54llama128_hm/rerank | tail4x_v4@0.125 | 0.344 | 0.0099 [0.0025, 0.0192] (n=5) | 0.0019 [-0.0093, 0.0134] (n=5) | — |
| llama31-8b/h54llama128_hm/rerank | dtail6ux_v4@0.125 | 0.398 | 0.0041 [0.0010, 0.0074] (n=5) | -0.0011 [-0.0050, 0.0029] (n=5) | — |
| llama31-8b/h54llama128_hm/rerank | dtail8ux_v4@0.125 | 0.453 | 0.0011 [0.0007, 0.0015] (n=5) | -0.0057 [-0.0100, -0.0011] (n=5) | — |
| llama31-8b/h54llama128_hm/rerank | fp8kv@8 | 0.500 | 0.0078 [0.0028, 0.0129] (n=5) | 0.0067 [-0.0119, 0.0288] (n=5) | — |
| llama31-8b/h54llama128_hm/rerank | uniform@4 | 0.625 | 0.0151 [0.0053, 0.0264] (n=5) | 0.0080 [-0.0009, 0.0169] (n=5) | — |
| llama31-8b/h54llama128_hm/rerank | fp_noise@0 | 1.000 | 0.0014 [0.0007, 0.0022] (n=5) | -0.0015 [-0.0062, 0.0027] (n=5) | — |
| llama31-8b/h54llama128_r3a/multivalue | uniform+v4@4 | 0.250 | 0.0179 [0.0113, 0.0254] (n=5) | 0.0002 [-0.0424, 0.0395] (n=5) | 0.000 [0.000, 0.000] (n=1) |
| llama31-8b/h54llama128_r3a/multivalue | dtail4ux_v4@0.125 | 0.344 | 0.0005 [0.0003, 0.0006] (n=5) | -0.0013 [-0.0017, -0.0009] (n=5) | 0.000 [0.000, 0.000] (n=1) |
| llama31-8b/h54llama128_r3a/multivalue | dtail4x_v4@0.125 | 0.344 | 0.0006 [0.0004, 0.0008] (n=5) | -0.0015 [-0.0055, 0.0017] (n=5) | 0.000 [0.000, 0.000] (n=1) |
| llama31-8b/h54llama128_r3a/multivalue | tail4x_v4@0.125 | 0.344 | 0.0006 [0.0003, 0.0008] (n=5) | 0.0014 [0.0006, 0.0022] (n=5) | 0.000 [0.000, 0.000] (n=1) |
| llama31-8b/h54llama128_r3a/multivalue | dtail6ux_v4@0.125 | 0.398 | 0.0003 [0.0002, 0.0004] (n=5) | 0.0028 [-0.0004, 0.0070] (n=5) | 0.000 [0.000, 0.000] (n=1) |
| llama31-8b/h54llama128_r3a/multivalue | dtail8ux_v4@0.125 | 0.453 | 0.0003 [0.0002, 0.0004] (n=5) | 0.0038 [-0.0000, 0.0105] (n=5) | 0.000 [0.000, 0.000] (n=1) |
| llama31-8b/h54llama128_r3a/multivalue | fp8kv@8 | 0.500 | 0.0058 [0.0035, 0.0087] (n=5) | -0.0020 [-0.0167, 0.0075] (n=5) | 0.000 [0.000, 0.000] (n=1) |
| llama31-8b/h54llama128_r3a/multivalue | uniform@4 | 0.625 | 0.0177 [0.0101, 0.0245] (n=5) | -0.0080 [-0.0566, 0.0351] (n=5) | 0.000 [0.000, 0.000] (n=1) |
| llama31-8b/h54llama128_r3a/multivalue | fp_noise@0 | 1.000 | 0.0006 [0.0004, 0.0007] (n=5) | 0.0020 [0.0002, 0.0038] (n=5) | 0.000 [0.000, 0.000] (n=1) |
| llama31-8b/h54llama128_r3a/retrieval | uniform+v4@4 | 0.250 | 0.0024 [0.0010, 0.0041] (n=10) | 0.0043 [-0.0005, 0.0097] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_r3a/retrieval | dtail4ux_v4@0.125 | 0.344 | 0.0000 [0.0000, 0.0001] (n=10) | 0.0001 [-0.0003, 0.0005] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_r3a/retrieval | dtail4x_v4@0.125 | 0.344 | 0.0001 [0.0000, 0.0001] (n=10) | 0.0000 [-0.0014, 0.0012] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_r3a/retrieval | tail4x_v4@0.125 | 0.344 | 0.0001 [0.0000, 0.0001] (n=10) | 0.0003 [-0.0002, 0.0008] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_r3a/retrieval | dtail6ux_v4@0.125 | 0.398 | 0.0001 [0.0000, 0.0001] (n=10) | -0.0007 [-0.0014, -0.0001] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_r3a/retrieval | dtail8ux_v4@0.125 | 0.453 | 0.0000 [0.0000, 0.0001] (n=10) | -0.0001 [-0.0005, 0.0003] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_r3a/retrieval | fp8kv@8 | 0.500 | 0.0005 [0.0002, 0.0009] (n=10) | 0.0011 [-0.0018, 0.0051] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_r3a/retrieval | uniform@4 | 0.625 | 0.0022 [0.0010, 0.0037] (n=10) | 0.0034 [-0.0010, 0.0087] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_r3a/retrieval | fp_noise@0 | 1.000 | 0.0001 [0.0001, 0.0002] (n=10) | 0.0004 [-0.0008, 0.0014] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| llama31-8b/h54llama128_r3a/tracking | uniform+v4@4 | 0.250 | 0.0106 [0.0080, 0.0142] (n=5) | 0.0117 [0.0071, 0.0164] (n=5) | -0.038 [-0.077, 0.000] (n=2) |
| llama31-8b/h54llama128_r3a/tracking | dtail4ux_v4@0.125 | 0.344 | 0.0004 [0.0003, 0.0006] (n=5) | 0.0001 [-0.0008, 0.0008] (n=5) | 0.000 [0.000, 0.000] (n=2) |
| llama31-8b/h54llama128_r3a/tracking | dtail4x_v4@0.125 | 0.344 | 0.0004 [0.0003, 0.0006] (n=5) | 0.0002 [-0.0015, 0.0019] (n=5) | 0.000 [0.000, 0.000] (n=2) |
| llama31-8b/h54llama128_r3a/tracking | tail4x_v4@0.125 | 0.344 | 0.0004 [0.0004, 0.0005] (n=5) | 0.0007 [-0.0027, 0.0042] (n=5) | 0.000 [0.000, 0.000] (n=2) |
| llama31-8b/h54llama128_r3a/tracking | dtail6ux_v4@0.125 | 0.398 | 0.0003 [0.0003, 0.0004] (n=5) | -0.0015 [-0.0036, 0.0008] (n=5) | 0.000 [0.000, 0.000] (n=2) |
| llama31-8b/h54llama128_r3a/tracking | dtail8ux_v4@0.125 | 0.453 | 0.0003 [0.0002, 0.0003] (n=5) | -0.0023 [-0.0040, -0.0005] (n=5) | 0.000 [0.000, 0.000] (n=2) |
| llama31-8b/h54llama128_r3a/tracking | fp8kv@8 | 0.500 | 0.0026 [0.0018, 0.0032] (n=5) | 0.0008 [-0.0056, 0.0053] (n=5) | -0.038 [-0.077, 0.000] (n=2) |
| llama31-8b/h54llama128_r3a/tracking | uniform@4 | 0.625 | 0.0098 [0.0074, 0.0126] (n=5) | 0.0098 [0.0026, 0.0165] (n=5) | -0.154 [-0.308, 0.000] (n=2) |
| llama31-8b/h54llama128_r3a/tracking | fp_noise@0 | 1.000 | 0.0005 [0.0003, 0.0005] (n=5) | -0.0006 [-0.0012, -0.0001] (n=5) | 0.000 [0.000, 0.000] (n=2) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | uniform+v4@4 | 0.250 | 0.0013 [0.0004, 0.0029] (n=5) | 0.0010 [-0.0003, 0.0030] (n=5) | -0.008 [-0.025, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | dtail4ux_v4@0.125 | 0.344 | 0.0001 [0.0000, 0.0002] (n=5) | -0.0001 [-0.0001, 0.0000] (n=5) | 0.000 [0.000, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | dtail4x_v4@0.125 | 0.344 | 0.0001 [0.0000, 0.0003] (n=5) | -0.0001 [-0.0002, 0.0000] (n=5) | 0.000 [0.000, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | tail4x_v4@0.125 | 0.344 | 0.0012 [0.0001, 0.0029] (n=5) | 0.0010 [-0.0001, 0.0026] (n=5) | -0.008 [-0.025, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | dtail6ux_v4@0.125 | 0.398 | 0.0001 [0.0000, 0.0002] (n=5) | -0.0001 [-0.0002, -0.0000] (n=5) | 0.000 [0.000, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | dtail8ux_v4@0.125 | 0.453 | 0.0001 [0.0000, 0.0001] (n=5) | -0.0001 [-0.0003, 0.0000] (n=5) | 0.000 [0.000, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | fp8kv@8 | 0.500 | 0.0005 [0.0001, 0.0012] (n=5) | 0.0004 [0.0000, 0.0010] (n=5) | 0.000 [0.000, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | uniform@4 | 0.625 | 0.0005 [0.0001, 0.0011] (n=5) | 0.0003 [-0.0002, 0.0011] (n=5) | 0.000 [0.000, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/multivalue | fp_noise@0 | 1.000 | 0.0001 [0.0000, 0.0002] (n=5) | -0.0000 [-0.0001, 0.0001] (n=5) | 0.000 [0.000, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | uniform+v4@4 | 0.250 | 0.0000 [0.0000, 0.0000] (n=10) | -0.0000 [-0.0001, 0.0000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | dtail4ux_v4@0.125 | 0.344 | 0.0000 [0.0000, 0.0000] (n=10) | -0.0000 [-0.0000, 0.0000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | dtail4x_v4@0.125 | 0.344 | 0.0000 [0.0000, 0.0000] (n=10) | 0.0000 [-0.0000, 0.0000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | tail4x_v4@0.125 | 0.344 | 0.0000 [0.0000, 0.0000] (n=10) | 0.0000 [0.0000, 0.0001] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | dtail6ux_v4@0.125 | 0.398 | 0.0000 [0.0000, 0.0000] (n=10) | -0.0000 [-0.0001, 0.0000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | dtail8ux_v4@0.125 | 0.453 | 0.0000 [0.0000, 0.0000] (n=10) | 0.0000 [0.0000, 0.0001] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | fp8kv@8 | 0.500 | 0.0000 [0.0000, 0.0000] (n=10) | 0.0000 [-0.0000, 0.0001] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | uniform@4 | 0.625 | 0.0000 [0.0000, 0.0001] (n=10) | -0.0001 [-0.0002, 0.0000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/retrieval | fp_noise@0 | 1.000 | 0.0000 [0.0000, 0.0000] (n=10) | -0.0000 [-0.0000, 0.0000] (n=10) | 0.000 [0.000, 0.000] (n=10) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | uniform+v4@4 | 0.250 | 0.0039 [0.0015, 0.0080] (n=5) | 0.0041 [0.0013, 0.0071] (n=5) | -0.012 [-0.024, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | dtail4ux_v4@0.125 | 0.344 | 0.0006 [0.0001, 0.0011] (n=5) | -0.0001 [-0.0006, 0.0004] (n=5) | 0.000 [0.000, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | dtail4x_v4@0.125 | 0.344 | 0.0013 [0.0004, 0.0022] (n=5) | 0.0025 [0.0007, 0.0049] (n=5) | 0.000 [0.000, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | tail4x_v4@0.125 | 0.344 | 0.0004 [0.0001, 0.0006] (n=5) | 0.0015 [-0.0001, 0.0031] (n=5) | -0.012 [-0.024, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | dtail6ux_v4@0.125 | 0.398 | 0.0003 [0.0002, 0.0004] (n=5) | 0.0007 [0.0003, 0.0012] (n=5) | -0.012 [-0.024, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | dtail8ux_v4@0.125 | 0.453 | 0.0012 [0.0006, 0.0018] (n=5) | 0.0017 [0.0003, 0.0031] (n=5) | -0.012 [-0.024, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | fp8kv@8 | 0.500 | 0.0023 [0.0010, 0.0040] (n=5) | 0.0035 [-0.0014, 0.0087] (n=5) | -0.012 [-0.024, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | uniform@4 | 0.625 | 0.0053 [0.0015, 0.0100] (n=5) | 0.0035 [-0.0001, 0.0075] (n=5) | 0.000 [0.000, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3a/tracking | fp_noise@0 | 1.000 | 0.0024 [0.0010, 0.0042] (n=5) | 0.0008 [-0.0015, 0.0031] (n=5) | 0.000 [0.000, 0.000] (n=5) |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | uniform+v4@4 | 0.250 | 0.0115 [0.0070, 0.0164] (n=10) | 0.0086 [-0.0020, 0.0197] (n=10) | 0.000 [0.000, 0.000] (n=4) |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | dtail4ux_v4@0.125 | 0.344 | 0.0016 [0.0012, 0.0020] (n=10) | 0.0003 [-0.0039, 0.0046] (n=10) | 0.000 [0.000, 0.000] (n=4) |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | dtail4x_v4@0.125 | 0.344 | 0.0037 [0.0028, 0.0046] (n=10) | -0.0001 [-0.0061, 0.0053] (n=10) | 0.000 [0.000, 0.000] (n=4) |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | tail4x_v4@0.125 | 0.344 | 0.0021 [0.0016, 0.0027] (n=10) | -0.0030 [-0.0060, 0.0002] (n=10) | 0.000 [0.000, 0.000] (n=4) |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | dtail6ux_v4@0.125 | 0.398 | 0.0012 [0.0007, 0.0017] (n=10) | 0.0011 [-0.0030, 0.0058] (n=10) | -0.050 [-0.100, 0.000] (n=4) |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | dtail8ux_v4@0.125 | 0.453 | 0.0008 [0.0005, 0.0010] (n=10) | 0.0027 [0.0009, 0.0047] (n=10) | 0.000 [0.000, 0.000] (n=4) |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | fp8kv@8 | 0.500 | 0.0029 [0.0020, 0.0038] (n=10) | 0.0065 [0.0016, 0.0118] (n=10) | 0.000 [0.000, 0.000] (n=4) |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | uniform@4 | 0.625 | 0.0126 [0.0070, 0.0190] (n=10) | 0.0097 [0.0005, 0.0192] (n=10) | -0.100 [-0.200, 0.000] (n=4) |
| qwen3-30b-a3b-2507/h54qwen32_r3b/aggregation | fp_noise@0 | 1.000 | 0.0040 [0.0021, 0.0061] (n=10) | 0.0028 [-0.0006, 0.0062] (n=10) | 0.000 [0.000, 0.000] (n=4) |
