# R14 Stage 1d (read_stage1d.py; rules frozen in its docstring)

**Decision: GO_KERNEL** · V4 WINs: seq2 @ llama31-8b@131072, topn @ llama31-8b@131072, qread @ llama31-8b@131072, qread @ llama31-8b@32768 · V16 WINs: seq2 @ llama31-8b@131072, topn @ llama31-8b@131072, hvah @ llama31-8b@131072, qread @ llama31-8b@131072, sieve @ llama31-8b@131072, union @ llama31-8b@131072, qread @ llama31-8b@32768

## llama31-8b@131072 — FP 0.996 ({'niah_multikey': 1.0, 'niah_multivalue': 0.9938, 'niah_single': 1.0, 'vt': 0.99}), 40 prompts, 160 prompt-tasks with an answer value (dropped 0), fp replay 0.998, answer-value coverage 1.00

| lens | seq2 | topn | hvah | qread | sieve | pool | seq | union |
|---|---|---|---|---|---|---|---|---|
| V16 | WIN router_seq2_calib@3 ρ 0.62 | WIN router_top32_calib@3 ρ 0.63 | WIN hvah_v16@3 ρ 0.57 | WIN qread_v16@0.25 ρ 0.25 | WIN router_calib@4 ρ 0.77 | NO_POINT | NO_POINT | WIN router_union_calib@3 ρ 0.71 |
| V4 | WIN router_seq2_calib+v4@3 ρ 0.74 | WIN router_top32_calib+v4@3 ρ 0.75 | NO_POINT | WIN qread_v4@0.125 ρ 0.13 | TIE router_calib+v4@4 ρ 0.94 | NO_POINT | NO_POINT | TIE router_union_calib+v4@3 ρ 0.80 |

| arm@B | lens | bytes | ρ | dA vs FP | dA vs D [90%] | tail | Δtail [90%] | label | label (1c metric) | score | lossless | kept width | evicted | needle kept |
|---|---|---:|---:|---|---|---:|---|---|---|---:|---|---:|---:|---:|
| qread_v16@0.25 | V16 | 77.0 | 0.25 | +0.227 [+0.166, +0.291] | -0.032 [-0.053, -0.011] | 0.025 | -0.013 [-0.025, +0.000] | MATCHED | MATCHED | 0.975 | no | 3.00 | 75.0% | 0.853 |
| hvah_v16@3 | V16 | 175.7 | 0.57 | +0.198 [+0.132, +0.270] | -0.061 [-0.122, +0.001] | 0.019 | -0.019 [-0.037, +0.000] | MATCHED | INCONCLUSIVE | 0.974 | no | 4.00 | 45.6% | 0.852 |
| mech_a@3 | V16 | 183.1 | 0.60 | +0.420 [+0.268, +0.602] | +0.162 [+0.013, +0.344] | 0.056 | +0.019 [-0.019, +0.062] | INCONCLUSIVE | INCONCLUSIVE | 0.966 | no | 5.76 | 47.9% | 0.846 |
| mech_p@3 | V16 | 183.2 | 0.60 | +1.452 [+0.971, +1.993] | +1.194 [+0.731, +1.708] | 0.144 | +0.106 [+0.050, +0.163] | WORSE | WORSE | 0.923 | no | 5.76 | 47.9% | 0.846 |
| mech_q@3 | V16 | 183.2 | 0.60 | +1.467 [+0.987, +2.007] | +1.209 [+0.748, +1.719] | 0.156 | +0.119 [+0.062, +0.175] | WORSE | WORSE | 0.915 | no | 5.76 | 47.9% | 0.846 |
| router_pool_calib@3 | V16 | 183.2 | 0.60 | +1.741 [+1.150, +2.413] | +1.482 [+0.911, +2.138] | 0.181 | +0.144 [+0.081, +0.212] | WORSE | WORSE | 0.900 | no | 5.76 | 47.9% | 0.846 |
| router_seq_calib@3 | V16 | 186.5 | 0.61 | +0.342 [+0.236, +0.461] | +0.084 [-0.006, +0.183] | 0.044 | +0.006 [-0.019, +0.031] | INCONCLUSIVE | INCONCLUSIVE | 0.977 | yes | 5.62 | 46.6% | 0.850 |
| router_seq2_calib@3 | V16 | 189.1 | 0.62 | +0.254 [+0.182, +0.336] | -0.004 [-0.061, +0.056] | 0.025 | -0.013 [-0.031, +0.006] | MATCHED | INCONCLUSIVE | 0.981 | yes | 5.51 | 45.6% | 0.852 |
| router_top32_calib@3 | V16 | 194.4 | 0.63 | +0.208 [+0.141, +0.281] | -0.051 [-0.109, +0.008] | 0.025 | -0.013 [-0.037, +0.013] | MATCHED | INCONCLUSIVE | 0.988 | yes | 5.31 | 43.5% | 0.860 |
| router_top64_calib@3 | V16 | 210.0 | 0.69 | +0.185 [+0.128, +0.246] | -0.073 [-0.105, -0.043] | 0.019 | -0.019 [-0.037, +0.000] | MATCHED | MATCHED | 0.981 | yes | 4.80 | 37.5% | 0.875 |
| router_union_calib@3 | V16 | 217.0 | 0.71 | +0.210 [+0.150, +0.271] | -0.048 [-0.071, -0.026] | 0.013 | -0.025 [-0.044, -0.006] | MATCHED | MATCHED | 0.986 | yes | 4.60 | 34.8% | 0.878 |
| hvah_v16@4 | V16 | 220.4 | 0.72 | +0.253 [+0.117, +0.425] | -0.005 [-0.146, +0.174] | 0.037 | +0.000 [-0.031, +0.031] | INCONCLUSIVE | INCONCLUSIVE | 0.964 | no | 4.00 | 31.7% | 0.929 |
| router_calib@4 | V16 | 235.2 | 0.77 | +0.105 [+0.063, +0.151] | -0.154 [-0.206, -0.102] | 0.006 | -0.031 [-0.050, -0.013] | MATCHED | MATCHED | 0.993 | yes | 6.05 | 33.9% | 0.905 |
| router_pool_calib@4 | V16 | 239.8 | 0.78 | +0.222 [+0.094, +0.380] | -0.037 [-0.170, +0.128] | 0.037 | +0.000 [-0.031, +0.031] | INCONCLUSIVE | INCONCLUSIVE | 0.980 | yes | 5.90 | 32.1% | 0.928 |
| router_seq2_calib@4 | V16 | 240.9 | 0.79 | +0.193 [+0.062, +0.362] | -0.065 [-0.204, +0.114] | 0.031 | -0.006 [-0.037, +0.025] | INCONCLUSIVE | INCONCLUSIVE | 0.981 | yes | 5.86 | 31.7% | 0.929 |
| uniform@3 | V16 | 306.3 | — | +0.258 [+0.196, +0.323] | (D) | 0.037 | — | — | — | 0.979 | yes | 3.00 | 0.0% | 1.000 |
| uniform@4 | V16 | 322.3 | 1.05 | +0.043 [+0.020, +0.065] | -0.216 [-0.277, -0.156] | 0.000 | -0.037 [-0.062, -0.019] | MATCHED | MATCHED | 0.988 | yes | 4.00 | 0.0% | 1.000 |
| qread_v4@0.125 | V4 | 15.0 | 0.13 | +0.236 [+0.177, +0.297] | -0.040 [-0.074, -0.008] | 0.044 | +0.019 [+0.000, +0.037] | MATCHED | INCONCLUSIVE | 0.990 | yes | 3.00 | 87.5% | 0.786 |
| qreadp_v4@0.125 | V4 | 20.1 | 0.17 | +0.236 [+0.175, +0.298] | -0.040 [-0.074, -0.006] | 0.013 | -0.013 [-0.031, +0.006] | MATCHED | INCONCLUSIVE | 0.981 | yes | 3.00 | 83.1% | 0.787 |
| qreadr_v4@0.125 | V4 | 21.2 | 0.18 | +0.234 [+0.177, +0.293] | -0.042 [-0.074, -0.012] | 0.031 | +0.006 [-0.006, +0.025] | MATCHED | INCONCLUSIVE | 0.990 | yes | 3.00 | 87.5% | 0.791 |
| qreadpr_v4@0.125 | V4 | 26.4 | 0.23 | +0.233 [+0.176, +0.291] | -0.043 [-0.073, -0.011] | 0.013 | -0.013 [-0.031, +0.006] | MATCHED | INCONCLUSIVE | 0.980 | yes | 3.00 | 83.1% | 0.792 |
| qread_v4@0.25 | V4 | 29.5 | 0.25 | +0.249 [+0.183, +0.319] | -0.027 [-0.047, -0.009] | 0.031 | +0.006 [+0.000, +0.019] | MATCHED | MATCHED | 0.989 | yes | 3.00 | 75.0% | 0.853 |
| qreadp_v4@0.25 | V4 | 33.9 | 0.29 | +0.240 [+0.182, +0.300] | -0.036 [-0.056, -0.016] | 0.019 | -0.006 [-0.019, +0.000] | MATCHED | MATCHED | 0.977 | yes | 3.00 | 71.2% | 0.853 |
| qread_v4@0.5 | V4 | 58.5 | 0.50 | +0.260 [+0.198, +0.325] | -0.016 [-0.027, -0.005] | 0.025 | +0.000 [+0.000, +0.000] | MATCHED | MATCHED | 0.978 | yes | 3.00 | 50.0% | 0.932 |
| hvah_v4@3 | V4 | 63.6 | 0.55 | +0.399 [+0.294, +0.523] | +0.123 [+0.051, +0.209] | 0.069 | +0.044 [+0.019, +0.075] | INCONCLUSIVE | WORSE | 0.973 | no | 3.00 | 45.6% | 0.852 |
| hvah_v4@4 | V4 | 79.7 | 0.68 | +0.557 [+0.359, +0.787] | +0.281 [+0.095, +0.505] | 0.087 | +0.062 [+0.025, +0.100] | INCONCLUSIVE | WORSE | 0.962 | no | 3.00 | 31.7% | 0.929 |
| router_seq2_calib+v4@3 | V4 | 85.7 | 0.74 | +0.272 [+0.201, +0.354] | -0.004 [-0.067, +0.061] | 0.019 | -0.006 [-0.031, +0.019] | MATCHED | INCONCLUSIVE | 0.980 | yes | 5.51 | 45.6% | 0.852 |
| router_top32_calib+v4@3 | V4 | 87.1 | 0.75 | +0.224 [+0.157, +0.298] | -0.052 [-0.111, +0.008] | 0.025 | +0.000 [-0.025, +0.025] | MATCHED | INCONCLUSIVE | 0.986 | yes | 5.31 | 43.5% | 0.860 |
| router_top64_calib+v4@3 | V4 | 91.2 | 0.78 | +0.204 [+0.145, +0.264] | -0.072 [-0.104, -0.043] | 0.025 | +0.000 [-0.013, +0.013] | MATCHED | MATCHED | 0.991 | yes | 4.80 | 37.5% | 0.875 |
| router_union_calib+v4@3 | V4 | 93.1 | 0.80 | +0.231 [+0.172, +0.292] | -0.045 [-0.069, -0.021] | 0.019 | -0.006 [-0.019, +0.000] | MATCHED | MATCHED | 0.987 | yes | 4.60 | 34.8% | 0.878 |
| router_calib+v4@4 | V4 | 109.6 | 0.94 | +0.124 [+0.079, +0.172] | -0.152 [-0.205, -0.099] | 0.013 | -0.013 [-0.031, +0.006] | MATCHED | MATCHED | 0.987 | yes | 6.05 | 33.9% | 0.905 |
| router_seq2_calib+v4@4 | V4 | 111.2 | 0.96 | +0.209 [+0.079, +0.373] | -0.067 [-0.207, +0.110] | 0.037 | +0.013 [-0.019, +0.044] | INCONCLUSIVE | MATCHED | 0.981 | yes | 5.86 | 31.7% | 0.929 |
| uniform+v4@3 | V4 | 116.3 | — | +0.276 [+0.211, +0.343] | (D) | 0.025 | — | — | — | 0.979 | yes | 3.00 | 0.0% | 1.000 |
| uniform+v4@4 | V4 | 132.3 | 1.14 | +0.052 [+0.028, +0.076] | -0.224 [-0.289, -0.162] | 0.000 | -0.025 [-0.044, -0.006] | MATCHED | MATCHED | 0.996 | yes | 4.00 | 0.0% | 1.000 |
| fp+v4@0 | V4 | 322.3 | 2.77 | +0.010 [+0.001, +0.018] | -0.266 [-0.332, -0.203] | 0.000 | -0.025 [-0.044, -0.006] | MATCHED | MATCHED | 0.994 | yes | 16.00 | 0.0% | 1.000 |

**QM**: 12 labels change with the metric fix (hvah_v16@3, hvah_v4@3, hvah_v4@4, qread_v4@0.125, qreadp_v4@0.125, qreadpr_v4@0.125, qreadr_v4@0.125, router_seq2_calib@3, router_seq2_calib+v4@3, router_seq2_calib+v4@4, router_top32_calib@3, router_top32_calib+v4@3); post-answer continuation = 0.08 of D's Stage-1c-metric damage
**QH** hvah_v16@3: hvah − seq2 -0.056 [-0.090, -0.024] → HVAH_HELPS
**QH** hvah_v4@3: hvah − seq2 +0.128 [+0.079, +0.181] → HVAH_HURTS
**QH** hvah_v16@4: hvah − seq2 +0.060 [+0.039, +0.080] → HVAH_HURTS
**QH** hvah_v4@4: hvah − seq2 +0.348 [+0.254, +0.453] → HVAH_HURTS
**QN** ladder router_seq2_calib@3 → router_top32_calib@3 → router_top64_calib@3 → router_union_calib@3: smallest MATCHED router_seq2_calib@3
**QR** qreadp_v4 − qread_v4 @0.125: -0.000 [-0.020, +0.021] → PROT_NO_EFFECT
**QR** qreadr_v4 − qread_v4 @0.125: -0.002 [-0.017, +0.013] → RESEL_NO_EFFECT
**QR** qreadpr_v4 − qread_v4 @0.125: -0.003 [-0.028, +0.022] → BOTH_NO_EFFECT
**QR** smallest MATCHED r: qread_v4 0.125, qreadp_v4 0.125, qreadr_v4 0.125, qreadpr_v4 0.125, qread_v16 0.25
**QX** B=3: ANSWER_TIME (full rescue +1.487 [+0.945, +2.102]; f_q 0.18 [0.10, 0.26]; f_a 0.89 [0.82, 0.96]; f_p 0.19 [0.11, 0.28]); PATCH_NO; key-term mass, critical / other heads 3.78 [3.67, 3.89] → KEY_READERS; answer-value rank median 41 of 256 → HIGH_ANSWER_ATTENTION
Calibration: B=3: {'prompt_tasks': 40, 'fail': 8, 'searched': 8, 'dense_R0': 9, 'dense_seq2': 21, 'dense_union': 78}; B=4: {'prompt_tasks': 40, 'fail': 2, 'searched': 2, 'dense_R0': 30, 'dense_seq2': 33, 'dense_union': 105}

## llama31-8b@32768 — FP 0.994 ({'niah_multikey': 1.0, 'niah_multivalue': 0.975, 'niah_single': 1.0, 'vt': 1.0}), 40 prompts, 159 prompt-tasks with an answer value (dropped 0), fp replay 0.996, answer-value coverage 1.00

| lens | seq2 | topn | hvah | qread | sieve | pool | seq | union |
|---|---|---|---|---|---|---|---|---|
| V16 | NO_POINT | NO_POINT | NO_POINT | WIN qread_v16@0.25 ρ 0.25 | NO_POINT | NO_POINT | NO_POINT | NO_POINT |
| V4 | NO_POINT | NO_POINT | NO_POINT | WIN qreadr_v4@0.125 ρ 0.19 | NO_POINT | NO_POINT | NO_POINT | NO_POINT |

| arm@B | lens | bytes | ρ | dA vs FP | dA vs D [90%] | tail | Δtail [90%] | label | label (1c metric) | score | lossless | kept width | evicted | needle kept |
|---|---|---:|---:|---|---|---:|---|---|---|---:|---|---:|---:|---:|
| qread_v16@0.25 | V16 | 78.0 | 0.25 | +0.127 [+0.090, +0.168] | +0.017 [-0.012, +0.055] | 0.006 | +0.008 [+0.000, +0.025] | MATCHED | INCONCLUSIVE | 0.988 | yes | 3.00 | 75.0% | 0.901 |
| mech_a@2.5 | V16 | 164.9 | 0.54 | +0.922 [+0.673, +1.180] | +0.812 [+0.567, +1.068] | 0.157 | +0.158 [+0.104, +0.212] | WORSE | WORSE | 0.940 | no | 5.21 | 52.2% | 0.798 |
| mech_p@2.5 | V16 | 164.9 | 0.54 | +2.515 [+1.802, +3.252] | +2.405 [+1.693, +3.137] | 0.302 | +0.302 [+0.237, +0.365] | WORSE | WORSE | 0.820 | no | 5.21 | 52.2% | 0.798 |
| router_pool_calib@2.5 | V16 | 164.9 | 0.54 | +3.277 [+2.417, +4.149] | +3.167 [+2.305, +4.043] | 0.377 | +0.379 [+0.294, +0.460] | WORSE | WORSE | 0.781 | no | 5.21 | 52.2% | 0.798 |
| mech_q@2.5 | V16 | 164.9 | 0.54 | +2.646 [+1.920, +3.383] | +2.536 [+1.813, +3.274] | 0.302 | +0.304 [+0.237, +0.369] | WORSE | WORSE | 0.801 | no | 5.21 | 52.2% | 0.798 |
| hvah_v16@2.5 | V16 | 170.2 | 0.55 | +0.153 [+0.095, +0.220] | +0.043 [-0.014, +0.104] | 0.013 | +0.013 [+0.000, +0.025] | INCONCLUSIVE | INCONCLUSIVE | 0.982 | yes | 4.00 | 47.6% | 0.819 |
| router_seq_calib@2.5 | V16 | 173.5 | 0.56 | +0.392 [+0.262, +0.546] | +0.281 [+0.156, +0.432] | 0.050 | +0.050 [+0.025, +0.081] | WORSE | WORSE | 0.983 | yes | 4.79 | 48.7% | 0.812 |
| router_seq2_calib@2.5 | V16 | 176.1 | 0.57 | +0.331 [+0.241, +0.429] | +0.221 [+0.140, +0.309] | 0.044 | +0.044 [+0.019, +0.069] | WORSE | WORSE | 0.985 | yes | 4.67 | 47.6% | 0.819 |
| router_top32_calib@2.5 | V16 | 178.6 | 0.58 | +0.315 [+0.207, +0.436] | +0.205 [+0.107, +0.317] | 0.038 | +0.037 [+0.013, +0.062] | WORSE | WORSE | 0.986 | yes | 4.56 | 46.6% | 0.828 |
| router_union_calib@2.5 | V16 | 187.6 | 0.61 | +0.292 [+0.210, +0.379] | +0.182 [+0.101, +0.269] | 0.038 | +0.037 [+0.013, +0.062] | WORSE | WORSE | 0.993 | yes | 4.20 | 42.9% | 0.850 |
| router_top64_calib@2.5 | V16 | 193.1 | 0.63 | +0.248 [+0.180, +0.319] | +0.138 [+0.077, +0.200] | 0.025 | +0.025 [+0.006, +0.044] | INCONCLUSIVE | WORSE | 0.996 | yes | 4.01 | 40.6% | 0.856 |
| hvah_v16@3 | V16 | 197.5 | 0.64 | +0.141 [+0.074, +0.220] | +0.031 [-0.039, +0.112] | 0.013 | +0.013 [+0.000, +0.025] | INCONCLUSIVE | INCONCLUSIVE | 0.992 | yes | 4.00 | 39.1% | 0.885 |
| router_calib@3 | V16 | 199.6 | 0.65 | +0.313 [+0.197, +0.438] | +0.203 [+0.089, +0.322] | 0.063 | +0.062 [+0.037, +0.094] | INCONCLUSIVE | INCONCLUSIVE | 0.979 | yes | 5.17 | 41.9% | 0.859 |
| router_pool_calib@3 | V16 | 205.5 | 0.67 | +0.244 [+0.105, +0.431] | +0.134 [-0.005, +0.322] | 0.019 | +0.019 [+0.006, +0.031] | INCONCLUSIVE | INCONCLUSIVE | 0.978 | yes | 4.97 | 39.7% | 0.882 |
| router_seq2_calib@3 | V16 | 206.8 | 0.67 | +0.143 [+0.072, +0.222] | +0.033 [-0.035, +0.110] | 0.013 | +0.013 [+0.000, +0.025] | INCONCLUSIVE | INCONCLUSIVE | 0.993 | yes | 4.93 | 39.1% | 0.885 |
| uniform@3 | V16 | 307.4 | — | +0.110 [+0.081, +0.141] | (D) | 0.000 | — | — | — | 0.994 | yes | 3.00 | 0.0% | 1.000 |
| uniform@4 | V16 | 323.4 | 1.05 | +0.024 [+0.010, +0.038] | -0.086 [-0.120, -0.054] | 0.000 | +0.000 [+0.000, +0.000] | MATCHED | MATCHED | 0.994 | yes | 4.00 | 0.0% | 1.000 |
| qread_v4@0.125 | V4 | 16.0 | 0.14 | +0.174 [+0.118, +0.236] | +0.063 [+0.013, +0.121] | 0.013 | +0.008 [-0.013, +0.029] | INCONCLUSIVE | WORSE | 0.988 | yes | 3.00 | 87.5% | 0.844 |
| qreadr_v4@0.125 | V4 | 22.3 | 0.19 | +0.140 [+0.099, +0.187] | +0.029 [-0.006, +0.076] | 0.006 | +0.002 [-0.013, +0.019] | MATCHED | WORSE | 0.986 | yes | 3.00 | 87.5% | 0.850 |
| qreadp_v4@0.125 | V4 | 25.5 | 0.22 | +0.107 [+0.077, +0.138] | -0.004 [-0.018, +0.010] | 0.000 | -0.006 [-0.019, +0.000] | MATCHED | WORSE | 0.994 | yes | 3.00 | 79.3% | 0.849 |
| qread_v4@0.25 | V4 | 30.5 | 0.26 | +0.131 [+0.093, +0.175] | +0.020 [-0.012, +0.062] | 0.006 | +0.002 [-0.013, +0.019] | MATCHED | INCONCLUSIVE | 0.986 | yes | 3.00 | 75.0% | 0.900 |
| qreadpr_v4@0.125 | V4 | 31.8 | 0.27 | +0.103 [+0.074, +0.133] | -0.008 [-0.023, +0.006] | 0.000 | -0.006 [-0.019, +0.000] | MATCHED | INCONCLUSIVE | 0.993 | yes | 3.00 | 79.3% | 0.853 |
| qreadp_v4@0.25 | V4 | 38.7 | 0.33 | +0.096 [+0.069, +0.125] | -0.015 [-0.024, -0.005] | 0.000 | -0.006 [-0.019, +0.000] | MATCHED | MATCHED | 0.994 | yes | 3.00 | 68.0% | 0.903 |
| qread_v4@0.5 | V4 | 59.5 | 0.51 | +0.102 [+0.074, +0.132] | -0.009 [-0.015, -0.003] | 0.000 | -0.006 [-0.019, +0.000] | MATCHED | MATCHED | 0.994 | yes | 3.00 | 50.0% | 0.953 |
| hvah_v4@2.5 | V4 | 62.3 | 0.53 | +0.253 [+0.186, +0.328] | +0.142 [+0.086, +0.206] | 0.031 | +0.025 [+0.006, +0.050] | INCONCLUSIVE | WORSE | 0.981 | yes | 3.00 | 47.6% | 0.819 |
| hvah_v4@3 | V4 | 72.1 | 0.61 | +0.248 [+0.159, +0.343] | +0.137 [+0.057, +0.223] | 0.044 | +0.037 [+0.013, +0.062] | INCONCLUSIVE | WORSE | 0.976 | yes | 3.00 | 39.1% | 0.885 |
| router_seq2_calib+v4@2.5 | V4 | 76.6 | 0.65 | +0.340 [+0.245, +0.441] | +0.229 [+0.143, +0.319] | 0.044 | +0.037 [+0.019, +0.062] | WORSE | WORSE | 0.987 | yes | 4.67 | 47.6% | 0.819 |
| router_top32_calib+v4@2.5 | V4 | 77.1 | 0.66 | +0.325 [+0.218, +0.445] | +0.214 [+0.116, +0.323] | 0.044 | +0.037 [+0.013, +0.062] | WORSE | WORSE | 0.988 | yes | 4.56 | 46.6% | 0.828 |
| router_union_calib+v4@2.5 | V4 | 79.1 | 0.67 | +0.307 [+0.221, +0.399] | +0.196 [+0.112, +0.286] | 0.044 | +0.037 [+0.013, +0.062] | WORSE | WORSE | 0.990 | yes | 4.20 | 42.9% | 0.850 |
| router_top64_calib+v4@2.5 | V4 | 80.3 | 0.68 | +0.257 [+0.188, +0.330] | +0.146 [+0.083, +0.210] | 0.025 | +0.019 [+0.006, +0.037] | INCONCLUSIVE | WORSE | 0.997 | yes | 4.01 | 40.6% | 0.856 |
| router_calib+v4@3 | V4 | 89.3 | 0.76 | +0.325 [+0.204, +0.455] | +0.214 [+0.098, +0.336] | 0.063 | +0.056 [+0.025, +0.087] | INCONCLUSIVE | INCONCLUSIVE | 0.979 | yes | 5.17 | 41.9% | 0.859 |
| router_seq2_calib+v4@3 | V4 | 91.2 | 0.78 | +0.158 [+0.079, +0.249] | +0.047 [-0.027, +0.133] | 0.019 | +0.013 [+0.000, +0.025] | INCONCLUSIVE | INCONCLUSIVE | 0.993 | yes | 4.93 | 39.1% | 0.885 |
| uniform+v4@3 | V4 | 117.4 | — | +0.111 [+0.082, +0.142] | (D) | 0.006 | — | — | — | 0.994 | yes | 3.00 | 0.0% | 1.000 |
| uniform+v4@4 | V4 | 133.4 | 1.14 | +0.025 [+0.010, +0.041] | -0.086 [-0.119, -0.054] | 0.000 | -0.006 [-0.019, +0.000] | MATCHED | MATCHED | 1.000 | yes | 4.00 | 0.0% | 1.000 |
| fp+v4@0 | V4 | 323.4 | 2.75 | +0.001 [-0.004, +0.006] | -0.110 [-0.140, -0.082] | 0.000 | -0.006 [-0.019, +0.000] | MATCHED | MATCHED | 0.994 | yes | 16.00 | 0.0% | 1.000 |

**QM**: 10 labels change with the metric fix (hvah_v4@2.5, hvah_v4@3, qread_v16@0.25, qread_v4@0.125, qread_v4@0.25, qreadp_v4@0.125, qreadpr_v4@0.125, qreadr_v4@0.125, router_top64_calib@2.5, router_top64_calib+v4@2.5); post-answer continuation = 0.38 of D's Stage-1c-metric damage
**QH** hvah_v16@2.5: hvah − seq2 -0.179 [-0.247, -0.114] → HVAH_HELPS
**QH** hvah_v4@2.5: hvah − seq2 -0.087 [-0.156, -0.022] → HVAH_HELPS
**QH** hvah_v16@3: hvah − seq2 -0.002 [-0.040, +0.033] → HVAH_NO_EFFECT
**QH** hvah_v4@3: hvah − seq2 +0.090 [+0.054, +0.125] → HVAH_HURTS
**QN** ladder router_seq2_calib@2.5 → router_top32_calib@2.5 → router_top64_calib@2.5 → router_union_calib@2.5: smallest MATCHED NONE
**QR** qreadp_v4 − qread_v4 @0.125: -0.067 [-0.120, -0.022] → PROT_HELPS
**QR** qreadr_v4 − qread_v4 @0.125: -0.034 [-0.068, -0.008] → RESEL_NO_EFFECT
**QR** qreadpr_v4 − qread_v4 @0.125: -0.071 [-0.126, -0.025] → BOTH_HELPS
**QR** smallest MATCHED r: qread_v4 0.25, qreadp_v4 0.125, qreadr_v4 0.125, qreadpr_v4 0.125, qread_v16 0.25
**QX** B=2.5: ANSWER_TIME (full rescue +2.946 [+2.115, +3.793]; f_q 0.21 [0.13, 0.30]; f_a 0.80 [0.75, 0.85]; f_p 0.26 [0.18, 0.35]); PATCH_PARTIAL; key-term mass, critical / other heads 2.67 [2.62, 2.72] → KEY_READERS; answer-value rank median 90 of 256 → HIGH_ANSWER_ATTENTION
Calibration: B=2.5: {'prompt_tasks': 40, 'fail': 20, 'searched': 20, 'dense_R0': 5, 'dense_seq2': 27, 'dense_union': 50}; B=3: {'prompt_tasks': 40, 'fail': 3, 'searched': 3, 'dense_R0': 33, 'dense_seq2': 36, 'dense_union': 118}

