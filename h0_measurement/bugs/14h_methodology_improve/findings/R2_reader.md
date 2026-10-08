# R14 Stage 1h — R2, Qwen3-30B-A3B at 32K (read_stage1h_r2.py; rules frozen in its docstring)

EXPLORATORY run: labels guide the design; they are not claims.

- **SYSTEM**: vs FP at m_FP 0.050: NEAR_FP (-0.007 [-0.025, +0.010]), EQUIV; vs D_V4 MATCHED; lost 0
- **VOTE_LOSS_EXACT**: VOTE_LOSS_EXACT_NO_EFFECT
- **VOTE_LOSS_4**: VOTE_LOSS_4_NO_EFFECT
- **EXACT_KV**: EXACT_KV_NO_EFFECT
- **SYS_VS_UNIFORM_V4**: SYS_VS_UNIFORM_V4_NO_EFFECT
- **SYS_VS_KIVI4_V4**: SYS_VS_KIVI4_V4_NO_EFFECT
- **SYS_VS_KVQUANT4_V4**: SYS_VS_KVQUANT4_V4_NO_EFFECT
- **SYS_VS_BEST**: SYS_VS_BEST_NO_EFFECT
- **SYS_VS_R1_BEST**: SYS_VS_R1_BEST_NO_EFFECT
- **REQ1**: REQ1_NO_EFFECT
- **STORE4**: STORE4_HELPS
- **SYS_VS_TT3**: SYS_VS_TT3_NO_EFFECT
- **FLOOR_SYS**: FLOOR_SYS_NO_EFFECT
- **FLOOR_EXACT**: FLOOR_EXACT_NO_EFFECT
- **FLOOR_4**: FLOOR_4_NO_EFFECT
- **BEST_DENSE_R2**: kvquant4_v4@4
- **SEQUENTIAL**: PASS after 10 prompts
- **NOISE**: NOISE_MEASURED
- **FP8_COST (Qwen)**: dP -0.005 [-0.014, +0.004]
- **FIX 8215/niah_multivalue system**: NOT_REPRODUCED
- **FIX 8218/niah_multivalue system**: NOT_REPRODUCED
- **FIX 8234/niah_multikey system**: FIXED
- **FIX 8235/niah_multivalue system**: NOT_REPRODUCED
- **FIX 8830/vt system**: NOT_FIXED
- **VOTE_LOSES_NEEDLE 8234**: False

## r2 — FP 1.000 ({'niah_multikey': 1.0, 'niah_multivalue': 1.0, 'niah_single': 1.0, 'vt': 1.0}), 10 prompts, 40 units; m_FP 0.050 (from R1)

| arm@B | lens | ρ | ρ mem | dP vs FP [90%] | NEAR_FP | EQUIV | KL span [90%] | dP vs D_L [90%] | MATCHED | lost / gained (McNemar p) | lost types | dS vs FP | min bias | score | peak GiB |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| qoraclefp_v16@0.125 | V16 | 0.21 | 1.67 | -0.019 [-0.028, -0.010] | NEAR_FP | EQUIV | +0.008 [+0.006, +0.009] | -0.126 [-0.244, -0.047] | MATCHED | 0 / 0 (1.000) | — | -0.019 [-0.028, -0.010] | +0.000 | 1.000 | 62.8 |
| qreadfp_v16@0.125 | V16 | 0.21 | 1.67 | -0.015 [-0.030, -0.001] | NEAR_FP | EQUIV | +0.018 [+0.012, +0.024] | -0.122 [-0.235, -0.049] | MATCHED | 0 / 0 (1.000) | — | -0.015 [-0.030, -0.001] | +0.000 | 1.000 | 62.9 |
| qreadfp_v16@0.5 | V16 | 0.84 | 1.67 | -0.011 [-0.018, -0.004] | NEAR_FP | EQUIV | +0.003 [+0.002, +0.003] | -0.118 [-0.241, -0.038] | MATCHED | 0 / 0 (1.000) | — | -0.011 [-0.018, -0.004] | +0.000 | 1.000 | 62.1 |
| uniform@3 | V16 | — | — | +0.107 [+0.030, +0.226] | INCONCLUSIVE | NOT_EQUIV | +0.114 [+0.061, +0.195] | (D) | — | 0 / 0 (1.000) | — | +0.043 [+0.022, +0.066] | -0.064 | 1.000 | 62.8 |
| uniform@4 | V16 | 1.05 | 1.05 | +0.035 [+0.016, +0.055] | NEAR_FP | EQUIV | +0.016 [+0.011, +0.022] | -0.072 [-0.172, +0.004] | MATCHED | 0 / 0 (1.000) | — | +0.035 [+0.016, +0.055] | +0.000 | 1.000 | 62.8 |
| fp_noise@0 | V16 | 1.67 | 1.67 | -0.005 [-0.011, +0.000] | NEAR_FP | EQUIV | +0.003 [+0.003, +0.004] | -0.112 [-0.228, -0.036] | MATCHED | 0 / 0 (1.000) | — | -0.005 [-0.011, +0.000] | +0.000 | 1.000 | 62.9 |
| qread_v4@0.125 | V4 | 0.14 | 1.00 | +0.145 [+0.049, +0.251] | INCONCLUSIVE | NOT_EQUIV | +0.169 [+0.086, +0.260] | +0.027 [-0.066, +0.132] | INCONCLUSIVE | 1 / 0 (1.000) | {'incomplete': 1} | +0.075 [+0.017, +0.165] | -0.069 | 0.980 | 63.5 |
| qoracle4_v4@0.125 | V4 | 0.15 | 1.14 | +0.015 [-0.016, +0.056] | NEAR_FP | EQUIV | +0.023 [+0.008, +0.046] | -0.103 [-0.195, -0.036] | MATCHED | 0 / 0 (1.000) | — | +0.002 [-0.016, +0.022] | -0.013 | 1.000 | 63.5 |
| qread4_v4@0.125 | V4 | 0.15 | 1.14 | +0.047 [+0.002, +0.095] | NEAR_FP | EQUIV | +0.058 [+0.028, +0.092] | -0.071 [-0.167, -0.001] | MATCHED | 0 / 0 (1.000) | — | +0.032 [-0.002, +0.068] | -0.015 | 1.000 | 63.5 |
| qread4q_v4@0.125 | V4 | 0.15 | 1.14 | +0.075 [+0.050, +0.102] | INCONCLUSIVE | NOT_EQUIV | +0.058 [+0.042, +0.076] | -0.043 [-0.155, +0.030] | MATCHED | 0 / 0 (1.000) | — | +0.069 [+0.048, +0.088] | -0.007 | 1.000 | 63.5 |
| qread2t4kq_v4@0.125 | V4 | 0.36 | 1.14 | -0.007 [-0.025, +0.010] | NEAR_FP | EQUIV | +0.029 [+0.019, +0.041] | -0.125 [-0.252, -0.044] | MATCHED | 0 / 0 (1.000) | — | -0.007 [-0.025, +0.010] | +0.000 | 1.000 | 63.5 |
| qread2t4q_v4@0.125 | V4 | 0.56 | 1.14 | -0.014 [-0.029, +0.005] | NEAR_FP | EQUIV | +0.023 [+0.015, +0.033] | -0.132 [-0.274, -0.041] | MATCHED | 0 / 0 (1.000) | — | -0.014 [-0.029, +0.005] | +0.000 | 1.000 | 63.5 |
| qread2t_v4@0.125 | V4 | 0.56 | 1.00 | -0.006 [-0.026, +0.016] | NEAR_FP | EQUIV | +0.044 [+0.029, +0.062] | -0.124 [-0.240, -0.051] | MATCHED | 0 / 0 (1.000) | — | -0.006 [-0.026, +0.016] | +0.000 | 1.000 | 63.5 |
| qread4_v4@0.5 | V4 | 0.58 | 1.14 | +0.027 [+0.008, +0.046] | NEAR_FP | EQUIV | +0.015 [+0.011, +0.020] | -0.091 [-0.214, -0.010] | MATCHED | 0 / 0 (1.000) | — | +0.027 [+0.008, +0.046] | +0.000 | 1.000 | 63.5 |
| uniform+v4@3 | V4 | — | — | +0.118 [+0.034, +0.256] | INCONCLUSIVE | NOT_EQUIV | +0.115 [+0.059, +0.213] | (D) | — | 0 / 0 (1.000) | — | +0.043 [+0.023, +0.063] | -0.075 | 1.000 | 63.5 |
| uniform+v4@4 | V4 | 1.14 | 1.14 | +0.025 [+0.004, +0.048] | NEAR_FP | EQUIV | +0.016 [+0.011, +0.022] | -0.093 [-0.208, -0.016] | MATCHED | 0 / 0 (1.000) | — | +0.025 [+0.004, +0.048] | +0.000 | 1.000 | 63.5 |
| kivi4_v4@4 | V4 | 1.15 | 1.15 | +0.007 [-0.004, +0.018] | NEAR_FP | EQUIV | +0.011 [+0.006, +0.017] | -0.111 [-0.246, -0.026] | MATCHED | 0 / 0 (1.000) | — | +0.007 [-0.004, +0.018] | +0.000 | 1.000 | 63.5 |
| kvquant4_v4@4 | V4 | 1.16 | 1.16 | -0.002 [-0.014, +0.011] | NEAR_FP | EQUIV | +0.010 [+0.006, +0.016] | -0.120 [-0.259, -0.031] | MATCHED | 0 / 0 (1.000) | — | -0.002 [-0.014, +0.011] | +0.000 | 1.000 | 63.5 |
| qread2t4kq_v4@0.5 | V4 | 1.38 | 1.14 | -0.006 [-0.011, -0.000] | NEAR_FP | EQUIV | +0.005 [+0.003, +0.007] | -0.123 [-0.259, -0.042] | MATCHED | 0 / 0 (1.000) | — | -0.006 [-0.011, -0.000] | +0.000 | 1.000 | 63.5 |
| fp+v4@0 | V4 | 2.75 | 2.75 | -0.011 [-0.020, -0.002] | NEAR_FP | EQUIV | +0.003 [+0.002, +0.005] | -0.129 [-0.261, -0.046] | MATCHED | 0 / 0 (1.000) | — | -0.011 [-0.020, -0.002] | +0.000 | 1.000 | 63.5 |
| fp8kv@8 | V8 | 2.19 | 2.19 | -0.005 [-0.014, +0.004] | NEAR_FP | EQUIV | +0.007 [+0.005, +0.009] | -0.123 [-0.266, -0.034] | MATCHED | 0 / 0 (1.000) | — | -0.005 [-0.014, +0.004] | +0.000 | 1.000 | 63.5 |

| arm@B | NEAR_FP at m_FP | EQUIV at m_FP |
|---|---|---|
| fp+v4@0 | NEAR_FP | EQUIV |
| fp8kv@8 | NEAR_FP | EQUIV |
| fp_noise@0 | NEAR_FP | EQUIV |
| kivi4_v4@4 | NEAR_FP | EQUIV |
| kvquant4_v4@4 | NEAR_FP | EQUIV |
| qoracle4_v4@0.125 | INCONCLUSIVE | NOT_EQUIV |
| qoraclefp_v16@0.125 | NEAR_FP | EQUIV |
| qread2t4kq_v4@0.125 | NEAR_FP | EQUIV |
| qread2t4kq_v4@0.5 | NEAR_FP | EQUIV |
| qread2t4q_v4@0.125 | NEAR_FP | EQUIV |
| qread2t_v4@0.125 | NEAR_FP | EQUIV |
| qread4_v4@0.125 | INCONCLUSIVE | NOT_EQUIV |
| qread4_v4@0.5 | NEAR_FP | EQUIV |
| qread4q_v4@0.125 | INCONCLUSIVE | NOT_EQUIV |
| qread_v4@0.125 | INCONCLUSIVE | NOT_EQUIV |
| qreadfp_v16@0.125 | NEAR_FP | EQUIV |
| qreadfp_v16@0.5 | NEAR_FP | EQUIV |
| uniform+v4@3 | INCONCLUSIVE | NOT_EQUIV |
| uniform+v4@4 | NEAR_FP | EQUIV |
| uniform@3 | INCONCLUSIVE | NOT_EQUIV |
| uniform@4 | INCONCLUSIVE | NOT_EQUIV |

**VOTE_LOSS_EXACT** (qreadfp_v16@0.125 − qoraclefp_v16@0.125): +0.004 [-0.007, +0.014] → VOTE_LOSS_EXACT_NO_EFFECT
**VOTE_LOSS_4** (qread4_v4@0.125 − qoracle4_v4@0.125): +0.032 [+0.001, +0.069] → VOTE_LOSS_4_NO_EFFECT
**EXACT_KV** (qread2t4q_v4@0.125 − qread2t4kq_v4@0.125): -0.007 [-0.022, +0.007] → EXACT_KV_NO_EFFECT
**SYS_VS_UNIFORM_V4** (qread2t4kq_v4@0.125 − uniform+v4@4): -0.032 [-0.052, -0.014] → SYS_VS_UNIFORM_V4_NO_EFFECT
**SYS_VS_KIVI4_V4** (qread2t4kq_v4@0.125 − kivi4_v4@4): -0.014 [-0.032, +0.004] → SYS_VS_KIVI4_V4_NO_EFFECT
**SYS_VS_KVQUANT4_V4** (qread2t4kq_v4@0.125 − kvquant4_v4@4): -0.005 [-0.022, +0.012] → SYS_VS_KVQUANT4_V4_NO_EFFECT
**SYS_VS_BEST** (qread2t4kq_v4@0.125 − kvquant4_v4@4): -0.005 [-0.022, +0.012] → SYS_VS_BEST_NO_EFFECT
**SYS_VS_R1_BEST** (qread2t4kq_v4@0.125 − kivi4_v4@4): -0.014 [-0.032, +0.004] → SYS_VS_R1_BEST_NO_EFFECT
**REQ1** (qread4q_v4@0.125 − qread4_v4@0.125): +0.028 [+0.001, +0.054] → REQ1_NO_EFFECT
**STORE4** (qread4_v4@0.125 − qread_v4@0.125): -0.098 [-0.181, -0.017] → STORE4_HELPS
**SYS_VS_TT3** (qread2t4kq_v4@0.125 − qread2t_v4@0.125): -0.001 [-0.017, +0.015] → SYS_VS_TT3_NO_EFFECT
**FLOOR_SYS** (qread2t4kq_v4@0.5 − qread2t4kq_v4@0.125): +0.001 [-0.013, +0.016] → FLOOR_SYS_NO_EFFECT
**FLOOR_EXACT** (qreadfp_v16@0.5 − qreadfp_v16@0.125): +0.004 [-0.010, +0.016] → FLOOR_EXACT_NO_EFFECT
**FLOOR_4** (qread4_v4@0.5 − qread4_v4@0.125): -0.020 [-0.060, +0.015] → FLOOR_4_NO_EFFECT

## Regression units across rotation seeds [0, 1, 2] (dP vs FP)

- 8215/niah_multivalue (reference uniform+v4@3): qread2t4kq_v4@0.125 **NOT_REPRODUCED**, qread2t4q_v4@0.125 **NOT_REPRODUCED**, qread4_v4@0.125 **NOT_REPRODUCED**, qread4q_v4@0.125 **NOT_REPRODUCED**, qreadfp_v16@0.125 **NOT_REPRODUCED**, qoraclefp_v16@0.125 **NOT_REPRODUCED**, uniform+v4@4 **NOT_REPRODUCED**, kivi4_v4@4 **NOT_REPRODUCED**, kvquant4_v4@4 **NOT_REPRODUCED**, fp8kv@8 **NOT_REPRODUCED**
- 8218/niah_multivalue (reference uniform+v4@3): qread2t4kq_v4@0.125 **NOT_REPRODUCED**, qread2t4q_v4@0.125 **NOT_REPRODUCED**, qread4_v4@0.125 **NOT_REPRODUCED**, qread4q_v4@0.125 **NOT_REPRODUCED**, qreadfp_v16@0.125 **NOT_REPRODUCED**, qoraclefp_v16@0.125 **NOT_REPRODUCED**, uniform+v4@4 **NOT_REPRODUCED**, kivi4_v4@4 **NOT_REPRODUCED**, kvquant4_v4@4 **NOT_REPRODUCED**, fp8kv@8 **NOT_REPRODUCED**
- 8234/niah_multikey (reference uniform+v4@3): qread2t4kq_v4@0.125 **FIXED**, qread2t4q_v4@0.125 **FIXED**, qread4_v4@0.125 **FIXED**, qread4q_v4@0.125 **NOT_FIXED**, qreadfp_v16@0.125 **FIXED**, qoraclefp_v16@0.125 **FIXED**, uniform+v4@4 **FIXED**, kivi4_v4@4 **FIXED**, kvquant4_v4@4 **FIXED**, fp8kv@8 **FIXED**
- 8235/niah_multivalue (reference uniform+v4@3): qread2t4kq_v4@0.125 **NOT_REPRODUCED**, qread2t4q_v4@0.125 **NOT_REPRODUCED**, qread4_v4@0.125 **NOT_REPRODUCED**, qread4q_v4@0.125 **NOT_REPRODUCED**, qreadfp_v16@0.125 **NOT_REPRODUCED**, qoraclefp_v16@0.125 **NOT_REPRODUCED**, uniform+v4@4 **NOT_REPRODUCED**, kivi4_v4@4 **NOT_REPRODUCED**, kvquant4_v4@4 **NOT_REPRODUCED**, fp8kv@8 **NOT_REPRODUCED**
- 8830/vt (reference qread2t_v4@0.125): qread2t4kq_v4@0.125 **NOT_FIXED**, qread2t4q_v4@0.125 **FIXED**, qread4_v4@0.125 **FIXED**, qread4q_v4@0.125 **FIXED**, qreadfp_v16@0.125 **FIXED**, qoraclefp_v16@0.125 **FIXED**, uniform+v4@4 **FIXED**, kivi4_v4@4 **FIXED**, kvquant4_v4@4 **FIXED**, fp8kv@8 **FIXED**

- 8215/niah_multivalue: uniform+v4@3 {0: 0.02, 1: 0.04, 2: -0.0}; qread2t_v4@0.125 {0: 0.15, 1: 0.14, 2: 0.06}; qread2t4kq_v4@0.125 {0: 0.06, 1: 0.08, 2: 0.06}; qread2t4q_v4@0.125 {0: 0.06, 1: 0.03, 2: 0.08}; qread4_v4@0.125 {0: 0.06, 1: 0.11, 2: 0.17}; qread4q_v4@0.125 {0: 0.13, 1: 0.08, 2: 0.14}; qreadfp_v16@0.125 {0: 0.02, 1: 0.02, 2: 0.02}; qoraclefp_v16@0.125 {0: 0.03, 1: 0.03, 2: 0.03}; uniform+v4@4 {0: -0.02, 1: 0.02, 2: 0.09}; kivi4_v4@4 {0: 0.02, 1: -0.02, 2: 0.02}; kvquant4_v4@4 {0: -0.02, 1: -0.02, 2: -0.02}; fp8kv@8 {0: 0.04, 1: 0.04, 2: 0.04}
- 8218/niah_multivalue: uniform+v4@3 {0: -0.04, 1: -0.08, 2: -0.06}; qread2t_v4@0.125 {0: -0.01, 1: 0.03, 2: -0.04}; qread2t4kq_v4@0.125 {0: 0.02, 1: 0.0, 2: 0.03}; qread2t4q_v4@0.125 {0: 0.02, 1: 0.05, 2: 0.02}; qread4_v4@0.125 {0: 0.01, 1: 0.01, 2: -0.01}; qread4q_v4@0.125 {0: 0.01, 1: 0.01, 2: -0.03}; qreadfp_v16@0.125 {0: 0.0, 1: 0.0, 2: 0.0}; qoraclefp_v16@0.125 {0: 0.02, 1: 0.02, 2: 0.02}; uniform+v4@4 {0: -0.03, 1: -0.01, 2: -0.03}; kivi4_v4@4 {0: 0.03, 1: 0.0, 2: 0.01}; kvquant4_v4@4 {0: 0.0, 1: -0.01, 2: -0.02}; fp8kv@8 {0: 0.05, 1: 0.05, 2: 0.05}
- 8234/niah_multikey: uniform+v4@3 {0: 9.13, 1: -0.0, 2: -0.0}; qread2t_v4@0.125 {0: 9.0, 1: -0.0, 2: -0.0}; qread2t4kq_v4@0.125 {0: 0.01, 1: 0.01, 2: 0.47}; qread2t4q_v4@0.125 {0: -0.0, 1: 0.02, 2: 2.15}; qread4_v4@0.125 {0: 0.04, 1: 0.0, 2: -0.0}; qread4q_v4@0.125 {0: 3.17, 1: 4.88, 2: -0.0}; qreadfp_v16@0.125 {0: 0.0, 1: 0.0, 2: 0.0}; qoraclefp_v16@0.125 {0: 0.0, 1: 0.0, 2: 0.0}; uniform+v4@4 {0: 0.03, 1: 0.0, 2: -0.0}; kivi4_v4@4 {0: 0.01, 1: 0.1, 2: 0.1}; kvquant4_v4@4 {0: -0.0, 1: -0.0, 2: -0.0}; fp8kv@8 {0: 0.16, 1: 0.16, 2: 0.16}
- 8235/niah_multivalue: uniform+v4@3 {0: 0.14, 1: 0.07, 2: -0.02}; qread2t_v4@0.125 {0: 0.03, 1: 0.03, 2: -0.02}; qread2t4kq_v4@0.125 {0: -0.0, 1: 0.03, 2: -0.03}; qread2t4q_v4@0.125 {0: 0.06, 1: 0.03, 2: 0.03}; qread4_v4@0.125 {0: -0.03, 1: 0.0, 2: -0.03}; qread4q_v4@0.125 {0: -0.03, 1: 0.06, 2: 0.09}; qreadfp_v16@0.125 {0: -0.0, 1: -0.0, 2: -0.0}; qoraclefp_v16@0.125 {0: -0.0, 1: -0.0, 2: -0.0}; uniform+v4@4 {0: -0.07, 1: -0.05, 2: -0.05}; kivi4_v4@4 {0: -0.05, 1: -0.0, 2: -0.03}; kvquant4_v4@4 {0: -0.03, 1: -0.03, 2: -0.05}; fp8kv@8 {0: 0.0, 1: 0.0, 2: 0.0}
- 8830/vt: uniform+v4@3 {0: 0.13, 1: 0.41, 2: 0.55}; qread2t_v4@0.125 {0: 8.8, 1: 9.84, 2: 14.83}; qread2t4kq_v4@0.125 {0: 9.82, 1: -0.04, 2: -0.02}; qread2t4q_v4@0.125 {0: -0.03, 1: -0.04, 2: -0.04}; qread4_v4@0.125 {0: 0.04, 1: -0.0, 2: 0.08}; qread4q_v4@0.125 {0: 0.05, 1: 0.04, 2: 0.03}; qreadfp_v16@0.125 {0: -0.02, 1: -0.02, 2: -0.02}; qoraclefp_v16@0.125 {0: -0.03, 1: -0.03, 2: -0.03}; uniform+v4@4 {0: 0.13, 1: 0.07, 2: 0.2}; kivi4_v4@4 {0: 0.08, 1: 0.12, 2: 0.12}; kvquant4_v4@4 {0: -0.04, 1: -0.05, 2: -0.05}; fp8kv@8 {0: -0.03, 1: -0.03, 2: -0.03}

## Beside Stage 1f (seed 0; dP R2, dP Stage 1f)

- 8215/niah_multivalue: {'fp+v4@0': (0.036, 0.036), 'uniform@3': (0.002, 0.002), 'uniform+v4@3': (0.017, 0.017), 'qread_v4@0.125': (0.032, 0.032), 'qreadfp_v16@0.125': (0.016, 0.016), 'qread2t_v4@0.125': (0.147, 0.147)}
- 8218/niah_multivalue: {'fp+v4@0': (-0.02, -0.02), 'uniform@3': (-0.025, -0.025), 'uniform+v4@3': (-0.038, -0.038), 'qread_v4@0.125': (0.019, 0.019), 'qreadfp_v16@0.125': (0.001, 0.001), 'qread2t_v4@0.125': (-0.012, -0.012)}
- 8234/niah_multikey: {'fp+v4@0': (0.009, 0.009), 'uniform@3': (9.751, 9.751), 'uniform+v4@3': (9.126, 9.126), 'qread_v4@0.125': (8.631, 8.631), 'qreadfp_v16@0.125': (0.001, 0.001), 'qread2t_v4@0.125': (9.004, 9.004)}
- 8235/niah_multivalue: {'fp+v4@0': (-0.026, -0.026), 'uniform@3': (0.062, 0.062), 'uniform+v4@3': (0.137, 0.137), 'qread_v4@0.125': (0.221, 0.221), 'qreadfp_v16@0.125': (-0.001, -0.001), 'qread2t_v4@0.125': (0.028, 0.028)}
- 8830/vt: {'fp+v4@0': (0.004, 0.004), 'uniform@3': (0.203, 0.203), 'uniform+v4@3': (0.128, 0.128), 'qread_v4@0.125': (0.144, 0.144), 'qreadfp_v16@0.125': (-0.022, -0.022), 'qread2t_v4@0.125': (8.799, 8.799)}

