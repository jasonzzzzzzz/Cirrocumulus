# R7a — the theory at the paper's scale (reader: read_r7a.py)


## Predictions (stated before the run)

| prediction | verdict |
|---|---|
| P1_coverage | MET |
| P2_tightness | MET |
| P3_certified | MET |
| P5_certread | MET |
| P6_p8_over_v2 | MET |
| P4_theorem_m | MET |
| P7_seed_robust | MET |
| P8_share2_orders_kl | MET |

## Panel A: Llama-3.1-8B, 128K (1,254,400 head-steps, 75 units)

| key bits | σ (median) | part 8 / error (p10–p90) | v2 / error | part 8 / v2 | certified within FP8 (p8 / v2) | coverage p8 (misses / checks) | coverage v2 | Theorem M exact (95% CI) | Theorem M first order | tail / FP8 (pred.) | certified read: rows (traffic), oracle, met per query head |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.343 | 166.72 (70.26–488.54) | 427.5 | 0.448 | 0.059 / 0.024 | COVER_OK (0 / 2,508,800) | COVER_OK (0) | 1.000 (0.998–1.002) | 0.999 | 0.062 | — |
| 6 | 0.082 | 5.59 (4.11–8.41) | 25.5 | 0.217 | 0.919 / 0.633 | COVER_OK (0 / 6,272,000) | COVER_OK (0) | 1.000 (1.000–1.001) | 1.000 | 0.024 | 0.126 (0.399), 0.125, 1.0000 (COVER_OK) |
| 8 | 0.020 | 2.05 (1.78–2.50) | 13.0 | 0.157 | 0.986 / 0.811 | COVER_OK (0 / 3,763,200) | COVER_OK (0) | 1.000 (0.999–1.000) | 1.000 | 0.019 | 0.125 (0.453), 0.125, 1.0000 (COVER_OK) |

| family | head-steps | unread share of Σw²‖v−o‖² (mean) | part 8 certified at 6 bits |
|---|---|---|---|
| aggregation | 316,416 | 0.0115 | 0.906 |
| icl | 30,720 | 0.0105 | 0.882 |
| multivalue | 261,120 | 0.0101 | 0.945 |
| rag | 78,848 | 0.0175 | 0.872 |
| rerank | 142,336 | 0.0296 | 0.824 |
| retrieval | 114,688 | 0.0067 | 0.977 |
| tracking | 310,272 | 0.0096 | 0.950 |

## Panel A: Qwen3-30B-A3B, 128K (500,736 head-steps, 30 units)

| key bits | σ (median) | part 8 / error (p10–p90) | v2 / error | part 8 / v2 | certified within FP8 (p8 / v2) | coverage p8 (misses / checks) | coverage v2 | Theorem M exact (95% CI) | Theorem M first order | tail / FP8 (pred.) | certified read: rows (traffic), oracle, met per query head |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.403 | 454.32 (69.28–296971.25) | 1372.6 | 0.428 | 0.026 / 0.007 | COVER_OK (0 / 1,001,472) | COVER_OK (0) | 0.990 (0.980–1.004) | 0.983 | 0.125 | — |
| 6 | 0.096 | 8.02 (4.50–35.60) | 42.6 | 0.211 | 0.679 / 0.382 | COVER_OK (0 / 2,503,680) | COVER_OK (0) | 1.000 (0.999–1.001) | 1.000 | 0.039 | 0.161 (0.423), 0.125, 1.0000 (COVER_OK) |
| 8 | 0.024 | 2.40 (1.88–4.03) | 17.1 | 0.149 | 0.912 / 0.642 | COVER_OK (0 / 1,502,208) | COVER_OK (0) | 1.001 (1.000–1.001) | 1.001 | 0.028 | 0.126 (0.454), 0.125, 1.0000 (COVER_OK) |

| family | head-steps | unread share of Σw²‖v−o‖² (mean) | part 8 certified at 6 bits |
|---|---|---|---|
| icl | 62,976 | 0.0291 | 0.731 |
| rag | 142,848 | 0.0314 | 0.776 |
| rerank | 294,912 | 0.0722 | 0.621 |

## Panel A: Qwen3-30B-A3B, 32K (2,029,056 head-steps, 50 units)

| key bits | σ (median) | part 8 / error (p10–p90) | v2 / error | part 8 / v2 | certified within FP8 (p8 / v2) | coverage p8 (misses / checks) | coverage v2 | Theorem M exact (95% CI) | Theorem M first order | tail / FP8 (pred.) | certified read: rows (traffic), oracle, met per query head |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.379 | 240.87 (39.16–143791.18) | 724.0 | 0.411 | 0.118 / 0.047 | COVER_OK (0 / 4,058,112) | COVER_OK (0) | 0.994 (0.989–0.999) | 0.987 | 0.088 | — |
| 6 | 0.090 | 7.22 (3.93–33.57) | 37.1 | 0.212 | 0.724 / 0.483 | COVER_OK (0 / 10,145,280) | COVER_OK (0) | 1.000 (1.000–1.001) | 1.000 | 0.029 | 0.235 (0.474), 0.126, 1.0000 (COVER_OK) |
| 8 | 0.022 | 2.34 (1.81–4.07) | 16.6 | 0.149 | 0.934 / 0.699 | COVER_OK (0 / 6,087,168) | COVER_OK (0) | 1.000 (1.000–1.001) | 1.000 | 0.020 | 0.130 (0.456), 0.125, 1.0000 (COVER_OK) |

| family | head-steps | unread share of Σw²‖v−o‖² (mean) | part 8 certified at 6 bits |
|---|---|---|---|
| aggregation | 451,584 | 0.0420 | 0.680 |
| multivalue | 491,520 | 0.0323 | 0.717 |
| retrieval | 649,728 | 0.0230 | 0.766 |
| tracking | 436,224 | 0.0347 | 0.714 |

## Panel B: Llama-3.1-8B, 128K, the design's KL across dither seeds

| family | units × seeds | mean y (seed 0) | common seed effect: share of variance | U (seed 0 = R6's) | U per seed | U (one seed per replicate) | U (two-level) | label |
|---|---|---|---|---|---|---|---|---|
| multivalue | 10 × 5 | -0.00751 | 0.000 | -0.00286 | -0.00286, -0.00287, -0.00287, -0.00281, -0.00291 | -0.00286 | -0.00287 | SEED_ROBUST |
| rerank | 10 × 5 | -0.00656 | 0.000 | -0.00422 | -0.00422, -0.00391, -0.00406, -0.00427, -0.00420 | -0.00419 | -0.00413 | SEED_ROBUST |
| retrieval | 10 × 5 | -0.00037 | 0.000 | -0.00019 | -0.00019, -0.00018, -0.00019, -0.00018, -0.00019 | -0.00018 | -0.00019 | SEED_ROBUST |

## Panel B: Qwen3-30B-A3B, 32K, the design's KL across dither seeds

| family | units × seeds | mean y (seed 0) | common seed effect: share of variance | U (seed 0 = R6's) | U per seed | U (one seed per replicate) | U (two-level) | label |
|---|---|---|---|---|---|---|---|---|
| multivalue | 10 × 5 | -0.00002 | 0.000 | 0.00002 | 0.00002, -0.00001, 0.00001, -0.00000, -0.00001 | 0.00001 | 0.00000 | SEED_ROBUST |
| retrieval | 10 × 5 | -0.00001 | 0.002 | -0.00000 | -0.00000, -0.00000, -0.00000, -0.00000, -0.00000 | -0.00000 | -0.00000 | SEED_ROBUST |
