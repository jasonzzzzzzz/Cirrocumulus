# R14 Stage 1d (read_stage1d.py; rules frozen in its docstring)

## qwen3-30b-a3b-2507@32768 — FP 0.750 ({'niah_multikey': 1.0, 'niah_multivalue': 0.0, 'niah_single': 1.0, 'vt': 1.0}), 40 prompts, 120 prompt-tasks with an answer value (dropped 0), fp replay 0.996, answer-value coverage 1.00

| lens | seq2 | topn | hvah | qread | sieve | pool | seq | union |
|---|---|---|---|---|---|---|---|---|
| V16 | NO_POINT | NO_POINT | NO_POINT | WIN qread_v16@0.25 ρ 0.25 | WIN router_calib@3 ρ 0.65 | NO_POINT | NO_POINT | NO_POINT |
| V4 | NO_POINT | NO_POINT | NO_POINT | NO_POINT | NO_POINT | NO_POINT | NO_POINT | NO_POINT |

| arm@B | lens | bytes | ρ | dA vs FP | dA vs D [90%] | tail | Δtail [90%] | label | label (1c metric) | score | lossless | kept width | evicted | needle kept |
|---|---|---:|---:|---|---|---:|---|---|---|---:|---|---:|---:|---:|
| qread_v16@0.25 | V16 | 78.1 | 0.25 | +0.032 [+0.016, +0.048] | +0.002 [-0.006, +0.011] | 0.000 | +0.000 [+0.000, +0.000] | MATCHED | MATCHED | 1.000 | yes | 3.00 | 75.0% | 0.868 |
| mech_p@2.5 | V16 | 163.4 | 0.53 | +1.338 [+0.590, +2.268] | +1.309 [+0.568, +2.230] | 0.108 | +0.108 [+0.067, +0.158] | WORSE | WORSE | 0.903 | no | 5.22 | 52.7% | 0.600 |
| mech_q@2.5 | V16 | 163.4 | 0.53 | +1.375 [+0.613, +2.323] | +1.345 [+0.586, +2.290] | 0.125 | +0.125 [+0.075, +0.175] | WORSE | WORSE | 0.903 | no | 5.22 | 52.7% | 0.600 |
| router_pool_calib@2.5 | V16 | 163.4 | 0.53 | +2.122 [+1.060, +3.419] | +2.093 [+1.035, +3.387] | 0.158 | +0.158 [+0.100, +0.225] | WORSE | WORSE | 0.868 | no | 5.22 | 52.7% | 0.600 |
| mech_a@2.5 | V16 | 163.4 | 0.53 | +0.688 [+0.400, +1.027] | +0.659 [+0.373, +0.997] | 0.100 | +0.100 [+0.058, +0.150] | WORSE | WORSE | 0.883 | no | 5.22 | 52.7% | 0.600 |
| router_seq2_calib@2.5 | V16 | 165.3 | 0.54 | +0.384 [+0.209, +0.581] | +0.355 [+0.181, +0.550] | 0.075 | +0.075 [+0.033, +0.117] | WORSE | WORSE | 0.933 | no | 5.12 | 51.9% | 0.607 |
| router_pool_oracle@2.5 | V16 | 167.2 | 0.54 | +1.212 [+0.638, +1.888] | +1.182 [+0.613, +1.853] | 0.125 | +0.125 [+0.083, +0.167] | WORSE | WORSE | 0.960 | no | 5.05 | 51.1% | 0.641 |
| router_union_calib@2.5 | V16 | 192.3 | 0.63 | +0.116 [+0.058, +0.181] | +0.087 [+0.028, +0.152] | 0.017 | +0.017 [+0.000, +0.033] | INCONCLUSIVE | MATCHED | 0.993 | yes | 3.98 | 40.8% | 0.704 |
| router_calib@3 | V16 | 200.8 | 0.65 | +0.041 [+0.005, +0.078] | +0.011 [-0.019, +0.045] | 0.000 | +0.000 [+0.000, +0.000] | MATCHED | MATCHED | 1.000 | yes | 5.13 | 41.5% | 0.714 |
| router_pool_calib@3 | V16 | 207.1 | 0.67 | +0.083 [+0.014, +0.195] | +0.053 [-0.015, +0.167] | 0.008 | +0.008 [+0.000, +0.025] | INCONCLUSIVE | MATCHED | 0.992 | yes | 4.92 | 39.0% | 0.733 |
| router_seq2_calib@3 | V16 | 207.1 | 0.67 | +0.083 [+0.014, +0.195] | +0.053 [-0.015, +0.167] | 0.008 | +0.008 [+0.000, +0.025] | INCONCLUSIVE | MATCHED | 0.992 | yes | 4.92 | 39.0% | 0.733 |
| router_pool_oracle@3 | V16 | 208.3 | 0.68 | +0.176 [+0.022, +0.453] | +0.147 [-0.010, +0.422] | 0.008 | +0.008 [+0.000, +0.025] | INCONCLUSIVE | MATCHED | 0.998 | yes | 4.92 | 38.5% | 0.782 |
| uniform@3 | V16 | 307.5 | — | +0.029 [+0.012, +0.049] | (D) | 0.000 | — | — | — | 1.000 | yes | 3.00 | 0.0% | 1.000 |
| uniform@4 | V16 | 323.4 | 1.05 | +0.019 [+0.009, +0.029] | -0.011 [-0.028, +0.007] | 0.000 | +0.000 [+0.000, +0.000] | MATCHED | MATCHED | 1.000 | yes | 4.00 | 0.0% | 1.000 |

**QM**: 4 labels change with the metric fix (router_pool_calib@3, router_pool_oracle@3, router_seq2_calib@3, router_union_calib@2.5); post-answer continuation = 0.71 of D's Stage-1c-metric damage
**QN** ladder router_seq2_calib@2.5 → router_union_calib@2.5: smallest MATCHED NONE
**QR** smallest MATCHED r: qread_v4 NONE, qreadp_v4 NONE, qreadr_v4 NONE, qreadpr_v4 NONE, qread_v16 0.25
**QX** B=2.5: SPLIT (full rescue +1.738 [+0.804, +2.907]; f_q 0.43 [0.33, 0.61]; f_a 0.82 [0.62, 0.93]; f_p 0.45 [0.34, 0.66]); PATCH_PARTIAL; key-term mass, critical / other heads 3.79 [3.58, 4.03] → KEY_READERS; answer-value rank median 103 of 192 → LOW_ANSWER_ATTENTION
**QX** B=3: MISSING_ARMS
Calibration: B=2.5: {'prompt_tasks': 30, 'fail': 8, 'searched': 8, 'dense_R0': 11, 'dense_seq2': 14, 'dense_union': 55}; B=3: {'prompt_tasks': 30, 'fail': 0, 'searched': 0, 'dense_R0': 39, 'dense_seq2': 39, 'dense_union': 98}

## Qwen replication

- R1: {'2.5': {'added': 3, 'share': 0.015625, 'label': 'SMALL_SET'}, '3': {'added': 0, 'share': 0.0, 'label': 'NO_CRITICAL_HEADS'}}
- R3: {'2.5': 'LOW_ANSWER_ATTENTION', '3': 'none'}
- R4: {'2.5': ('SPLIT', 'PATCH_PARTIAL'), '3': ('MISSING_ARMS', None)}
- R2: {'gap_closed': (1.9094188446568112, 1.3922864431415878, 3.4631021851462145), 'label': 'SEQ_CLOSES'}
- R5: {'n32': 3, 'n8': 3, 'overlap': [(30, 3)], 'share': 0.3333333333333333, 'label': 'LENGTH_UNSTABLE'}
