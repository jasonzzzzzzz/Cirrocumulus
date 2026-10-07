# R14 Stage 1h — R1, calibration and bridge (read_stage1h.py; rules frozen in its docstring)

EXPLORATORY run: labels guide the design; they are not claims.

- **VOTE_LOSS_EXACT**: VOTE_LOSS_EXACT_NO_EFFECT, NOT_EQUIV
- **VOTE_LOSS_4**: VOTE_LOSS_4_NO_EFFECT, EQUIV
- **EXACT_KV**: EXACT_KV_NO_EFFECT
- **NOISE**: NOISE_MEASURED: |dP| 95th pct 0.1368, KL span 95th pct 0.0184
- **FP8_COST**: dP +0.000 [-0.018, +0.018], NEAR_FP
- **M_FP (R2 on)**: 0.050
- **BEST_DENSE**: kivi4_v4@4
- **SYSTEM**: vs FP NEAR_FP (+0.016 [-0.033, +0.069]), EQUIV; vs D_V4 MATCHED; lost 0
- **BRIDGE**: BRIDGE_OK
- **CONFUSION 8109/niah_multikey**: CONFUSION_SEED_DEPENDENT
- **CONFUSION 8901/niah_multikey**: CONFUSION_SEED_DEPENDENT
- **CONFUSION 8937/niah_multikey**: CONFUSION_SEED_DEPENDENT

## r1 — FP 0.988 ({'niah_multikey': 1.0, 'niah_multivalue': 0.9625, 'niah_single': 1.0, 'vt': 0.99}), 20 prompts, 80 units; primary dP = a_span_nll, co-primary KL span

| arm@B | lens | ρ | ρ mem | dP vs FP [90%] | NEAR_FP | EQUIV | KL span [90%] | dP vs D_L [90%] | MATCHED | lost / gained (McNemar p) | lost types | dS vs FP | min bias | score | peak GiB |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| qreadfp_v16@0.125 | V16 | 0.21 | 1.67 | +0.011 [-0.041, +0.067] | NEAR_FP | EQUIV | +0.061 [+0.047, +0.078] | -0.319 [-0.453, -0.192] | MATCHED | 1 / 2 (1.000) | {'incomplete': 1} | -0.026 [-0.066, +0.018] | -0.036 | 0.986 | 45.6 |
| qoraclefp_v16@0.125 | V16 | 0.21 | 1.67 | -0.006 [-0.025, +0.013] | NEAR_FP | EQUIV | +0.021 [+0.018, +0.025] | -0.336 [-0.451, -0.227] | MATCHED | 0 / 0 (1.000) | — | -0.016 [-0.036, +0.005] | -0.010 | 0.988 | 45.5 |
| uniform@3 | V16 | — | — | +0.330 [+0.219, +0.447] | FAR_FROM_FP | NOT_EQUIV | +0.286 [+0.234, +0.343] | (D) | — | 3 / 3 (1.000) | {'incomplete': 3} | +0.288 [+0.176, +0.405] | -0.042 | 0.976 | 45.5 |
| uniform@4 | V16 | 1.05 | 1.05 | +0.073 [+0.036, +0.113] | INCONCLUSIVE | NOT_EQUIV | +0.079 [+0.066, +0.094] | -0.257 [-0.385, -0.137] | MATCHED | 0 / 1 (1.000) | — | +0.070 [+0.034, +0.107] | -0.004 | 0.991 | 45.5 |
| fp_noise@0 | V16 | 1.67 | 1.67 | +0.013 [+0.003, +0.023] | NEAR_FP | EQUIV | +0.005 [+0.004, +0.006] | -0.317 [-0.434, -0.206] | MATCHED | 0 / 0 (1.000) | — | +0.013 [+0.003, +0.023] | +0.000 | 0.988 | 45.6 |
| qread_v4@0.125 | V4 | 0.13 | 1.00 | +0.309 [+0.193, +0.424] | FAR_FROM_FP | NOT_EQUIV | +0.308 [+0.254, +0.362] | -0.051 [-0.107, +0.001] | MATCHED | 4 / 2 (0.688) | {'incomplete': 4} | +0.225 [+0.109, +0.340] | -0.084 | 0.970 | 48.2 |
| qread4_v4@0.125 | V4 | 0.15 | 1.14 | +0.093 [+0.033, +0.156] | INCONCLUSIVE | NOT_EQUIV | +0.137 [+0.115, +0.161] | -0.266 [-0.405, -0.132] | MATCHED | 3 / 3 (1.000) | {'incomplete': 3} | +0.066 [+0.012, +0.121] | -0.027 | 0.981 | 48.3 |
| qoracle4_v4@0.125 | V4 | 0.15 | 1.14 | +0.103 [+0.053, +0.155] | INCONCLUSIVE | NOT_EQUIV | +0.098 [+0.082, +0.117] | -0.257 [-0.383, -0.137] | MATCHED | 2 / 2 (1.000) | {'incomplete': 2} | +0.102 [+0.052, +0.154] | -0.001 | 0.981 | 48.2 |
| qread2t4kq_v4@0.125 | V4 | 0.35 | 1.14 | +0.016 [-0.033, +0.069] | NEAR_FP | EQUIV | +0.070 [+0.054, +0.086] | -0.344 [-0.475, -0.219] | MATCHED | 0 / 2 (0.500) | — | -0.012 [-0.054, +0.032] | -0.028 | 0.994 | 48.3 |
| qread2t4q_v4@0.125 | V4 | 0.55 | 1.14 | -0.012 [-0.053, +0.035] | NEAR_FP | EQUIV | +0.058 [+0.045, +0.073] | -0.371 [-0.501, -0.245] | MATCHED | 1 / 1 (1.000) | {'incomplete': 1} | -0.042 [-0.079, -0.002] | -0.030 | 0.984 | 48.3 |
| uniform+v4@3 | V4 | — | — | +0.359 [+0.242, +0.479] | FAR_FROM_FP | NOT_EQUIV | +0.303 [+0.252, +0.357] | (D) | — | 3 / 2 (1.000) | {'incomplete': 3} | +0.293 [+0.174, +0.411] | -0.067 | 0.974 | 48.1 |
| uniform+v4@4 | V4 | 1.14 | 1.14 | +0.107 [+0.063, +0.152] | INCONCLUSIVE | NOT_EQUIV | +0.095 [+0.077, +0.115] | -0.253 [-0.383, -0.128] | MATCHED | 2 / 3 (1.000) | {'incomplete': 2} | +0.106 [+0.063, +0.152] | -0.000 | 0.984 | 48.1 |
| kivi4_v4@4 | V4 | 1.15 | 1.15 | +0.071 [+0.041, +0.101] | INCONCLUSIVE | NOT_EQUIV | +0.069 [+0.059, +0.080] | -0.288 [-0.398, -0.181] | MATCHED | 0 / 0 (1.000) | — | +0.066 [+0.039, +0.093] | -0.005 | 0.988 | 48.1 |
| kvquant4_v4@4 | V4 | 1.16 | 1.16 | +0.123 [+0.086, +0.161] | INCONCLUSIVE | NOT_EQUIV | +0.077 [+0.068, +0.087] | -0.236 [-0.346, -0.127] | MATCHED | 2 / 0 (0.500) | {'incomplete': 2} | +0.103 [+0.072, +0.132] | -0.021 | 0.975 | 48.1 |
| fp+v4@0 | V4 | 2.77 | 2.77 | +0.035 [+0.019, +0.051] | NEAR_FP | EQUIV | +0.013 [+0.010, +0.015] | -0.324 [-0.441, -0.211] | MATCHED | 1 / 0 (1.000) | {'incomplete': 1} | +0.035 [+0.019, +0.051] | +0.000 | 0.986 | 48.1 |
| fp8kv@8 | V8 | 2.20 | 2.20 | +0.000 [-0.018, +0.018] | NEAR_FP | EQUIV | +0.027 [+0.022, +0.032] | -0.359 [-0.479, -0.244] | MATCHED | 1 / 0 (1.000) | {'incomplete': 1} | +0.000 [-0.018, +0.018] | +0.000 | 0.981 | 48.1 |

**VOTE_LOSS_EXACT** (qreadfp_v16@0.125 − qoraclefp_v16@0.125): +0.016 [-0.023, +0.060] → VOTE_LOSS_EXACT_NO_EFFECT, NOT_EQUIV within ±0.05
**VOTE_LOSS_4** (qread4_v4@0.125 − qoracle4_v4@0.125): -0.010 [-0.043, +0.026] → VOTE_LOSS_4_NO_EFFECT, EQUIV within ±0.05
**EXACT_KV** (qread2t4q_v4@0.125 − qread2t4kq_v4@0.125): -0.027 [-0.043, -0.011] → EXACT_KV_NO_EFFECT
**NOISE** (NOISE_MEASURED, 80 units): |dP| mean 0.0346, 95th percentile 0.1368; dP +0.013 [+0.003, +0.023]; KL span +0.005 [+0.004, +0.006], 95th percentile 0.0184; lost 0
**FP8_COST**: dP +0.000 [-0.018, +0.018] (NEAR_FP); KL span +0.027 [+0.022, +0.032]; lost 1. **M_FP** (from R2 on) = 0.050
**BEST_DENSE**: kivi4_v4@4 (mean dP {'uniform+v4@4': 0.10666799999999999, 'kivi4_v4@4': 0.07096387499999998, 'kvquant4_v4@4': 0.12322487499999997})
**SYSTEM vs D_V4 (lost answers)**: 0 vs 3, McNemar p 0.250

**FP-FAILED stratum**: 4 of 80 units ({'niah_multikey': 0, 'niah_multivalue': 3, 'niah_single': 0, 'vt': 1}); mean dP there: fp+v4@0 +0.03, fp8kv@8 -0.14, fp_noise@0 +0.05, kivi4_v4@4 -0.03, kvquant4_v4@4 +0.22, qoracle4_v4@0.125 +0.19, qoraclefp_v16@0.125 +0.12, qread2t4kq_v4@0.125 +0.34, qread2t4q_v4@0.125 +0.24, qread4_v4@0.125 +0.25, qread_v4@0.125 +0.67, qreadfp_v16@0.125 +0.25, uniform@3 +0.57, uniform@4 +0.07, uniform+v4@3 +0.63, uniform+v4@4 +0.09

## Bridge to Stage 1g (80 units; jobs ['22465811', '22465812']): **BRIDGE_OK**

| arm@B | dP R1 | dP Stage 1g | R1 − 1g [90%] |
|---|---:|---:|---|
| fp+v4@0 | +0.035 | +0.035 | +0.000 [+0.000, +0.000] |
| uniform@3 | +0.330 | +0.330 | +0.000 [+0.000, +0.000] |
| uniform+v4@3 | +0.359 | +0.359 | +0.000 [+0.000, +0.000] |
| uniform@4 | +0.073 | +0.073 | +0.000 [+0.000, +0.000] |
| uniform+v4@4 | +0.107 | +0.107 | +0.000 [+0.000, +0.000] |
| qread_v4@0.125 | +0.309 | +0.309 | +0.000 [+0.000, +0.000] |
| qreadfp_v16@0.125 | +0.011 | +0.011 | +0.000 [+0.000, +0.000] |
| qread4_v4@0.125 | +0.093 | +0.093 | +0.000 [+0.000, +0.000] |
| qread2t4kq_v4@0.125 | +0.016 | +0.016 | +0.000 [+0.000, +0.000] |

## Key confusions across rotation seeds [0, 1, 2] (dP vs FP)

- 8109/niah_multikey: **CONFUSION_SEED_DEPENDENT**
- 8901/niah_multikey: **CONFUSION_SEED_DEPENDENT**
- 8937/niah_multikey: **CONFUSION_SEED_DEPENDENT**

- 8109/niah_multikey: uniform@3 {0: 4.74, 1: 4.61, 2: 0.22}; uniform+v4@3 {0: 6.03, 1: 5.65, 2: 0.2}; qread_v4@0.125 {0: 6.4, 1: 6.39, 2: 0.22}; uniform+v4@4 {0: 0.59, 1: 0.13, 2: -0.03}; kivi4_v4@4 {0: 0.07, 1: 0.01, 2: -0.0}; kvquant4_v4@4 {0: 0.13, 1: 0.06, 2: 0.01}; fp8kv@8 {0: -0.02, 1: -0.02, 2: -0.02}; fp+v4@0 {0: 0.04, 1: -0.0, 2: -0.03}; qread4_v4@0.125 {0: 0.79, 1: 0.17, 2: -0.04}; qread2t4kq_v4@0.125 {0: 0.05, 1: 0.03, 2: -0.03}; qread2t4q_v4@0.125 {0: 0.02, 1: 0.03, 2: 0.02}; qreadfp_v16@0.125 {0: 0.02, 1: 0.02, 2: 0.02}; qoraclefp_v16@0.125 {0: 0.03, 1: 0.03, 2: 0.03}; qoracle4_v4@0.125 {0: 0.72, 1: 0.17, 2: -0.03}
- 8901/niah_multikey: uniform@3 {0: 3.74, 1: 0.01, 2: 1.11}; uniform+v4@3 {0: 3.38, 1: 0.04, 2: 0.95}; qread_v4@0.125 {0: 2.9, 1: -0.0, 2: 0.67}; uniform+v4@4 {0: 0.02, 1: -0.04, 2: 2.67}; kivi4_v4@4 {0: -0.02, 1: -0.04, 2: -0.04}; kvquant4_v4@4 {0: -0.04, 1: -0.05, 2: -0.04}; fp8kv@8 {0: -0.05, 1: -0.05, 2: -0.05}; fp+v4@0 {0: 0.01, 1: 0.01, 2: -0.01}; qread4_v4@0.125 {0: -0.01, 1: -0.05, 2: 2.32}; qread2t4kq_v4@0.125 {0: -0.01, 1: -0.01, 2: -0.03}; qread2t4q_v4@0.125 {0: -0.02, 1: -0.02, 2: -0.02}; qreadfp_v16@0.125 {0: -0.02, 1: -0.02, 2: -0.02}; qoraclefp_v16@0.125 {0: -0.02, 1: -0.02, 2: -0.02}; qoracle4_v4@0.125 {0: -0.01, 1: -0.04, 2: 2.21}
- 8937/niah_multikey: uniform@3 {0: 3.09, 1: -0.05, 2: 0.9}; uniform+v4@3 {0: 3.09, 1: -0.06, 2: 0.61}; qread_v4@0.125 {0: 2.85, 1: -0.08, 2: 0.48}; uniform+v4@4 {0: 0.76, 1: 0.7, 2: 0.15}; kivi4_v4@4 {0: 0.03, 1: 0.03, 2: -0.01}; kvquant4_v4@4 {0: 0.28, 1: 0.31, 2: 0.09}; fp8kv@8 {0: 0.15, 1: 0.15, 2: 0.15}; fp+v4@0 {0: 0.02, 1: 0.05, 2: -0.04}; qread4_v4@0.125 {0: 0.55, 1: 0.67, 2: 0.1}; qread2t4kq_v4@0.125 {0: -0.03, 1: 0.01, 2: -0.06}; qread2t4q_v4@0.125 {0: -0.02, 1: -0.02, 2: -0.01}; qreadfp_v16@0.125 {0: -0.02, 1: -0.02, 2: -0.02}; qoraclefp_v16@0.125 {0: -0.02, 1: -0.02, 2: -0.02}; qoracle4_v4@0.125 {0: 0.62, 1: 0.8, 2: 0.17}

