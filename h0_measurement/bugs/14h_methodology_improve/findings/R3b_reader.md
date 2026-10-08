# R14 Stage 1h — R3b, aggregation and latent-association tasks (read_stage1h_r3b.py; rules frozen in its docstring)

EXPLORATORY run: labels guide the design; they are not claims.

- **ACC_SYSTEM AGG llama**: ACC_INCONCLUSIVE (-0.033 [-0.150, +0.083])
- **SYS_VS_BEST_ACC AGG llama**: SYS_VS_BEST_ACC_NO_EFFECT, NOT_EQUIV (-0.083 [-0.200, +0.017])
- **SIMPLE_VS_BEST_ACC AGG llama**: SIMPLE_VS_BEST_ACC_NO_EFFECT, NOT_EQUIV (-0.067 [-0.183, +0.050])
- **SYS_VS_D_ACC AGG llama**: SYS_VS_D_ACC_HURTS, NOT_EQUIV (-0.133 [-0.250, -0.017])
- **REQ1_ACC AGG llama**: REQ1_ACC_NO_EFFECT, NOT_EQUIV (+0.033 [-0.033, +0.100])
- **VOTE_LOSS_ACC AGG llama**: VOTE_LOSS_ACC_NO_EFFECT, NOT_EQUIV (-0.017 [-0.083, +0.050])
- **BEST_DENSE_ACC AGG llama**: uniform+v4@4
- **FP accuracy llama**: {'cwe': 0.48, 'fwe': 0.6, 'FP_LOW': ['cwe']}
- **SYSTEM NLL llama**: FAR_FROM_FP at the cell's margin 0.050 (+1.374 [+0.985, +1.811]); vs D_V4 WORSE
- **SYS_VS_BEST llama**: SYS_VS_BEST_HURTS
- **REQ1 llama**: REQ1_HURTS
- **STORE4 llama**: STORE4_HURTS
- **ACC_SYSTEM AGG qwen**: ACC_LOSS (-0.096 [-0.135, -0.057])
- **SYS_VS_BEST_ACC AGG qwen**: SYS_VS_BEST_ACC_HURTS, NOT_EQUIV (-0.108 [-0.145, -0.073])
- **SIMPLE_VS_BEST_ACC AGG qwen**: SIMPLE_VS_BEST_ACC_HURTS, NOT_EQUIV (-0.116 [-0.148, -0.085])
- **SYS_VS_D_ACC AGG qwen**: SYS_VS_D_ACC_HURTS, NOT_EQUIV (-0.073 [-0.122, -0.027])
- **REQ1_ACC AGG qwen**: REQ1_ACC_NO_EFFECT, EQUIV (+0.000 [-0.005, +0.005])
- **VOTE_LOSS_ACC AGG qwen**: VOTE_LOSS_ACC_HURTS, NOT_EQUIV (-0.126 [-0.158, -0.094])
- **BEST_DENSE_ACC AGG qwen**: kivi4_v4@4
- **FP accuracy qwen**: {'cwe': 0.8, 'fwe': 0.883}
- **SYSTEM NLL qwen**: FAR_FROM_FP at the cell's margin 0.100 (+6.919 [+5.035, +8.963]); vs D_V4 WORSE
- **SYS_VS_BEST qwen**: SYS_VS_BEST_HURTS
- **REQ1 qwen**: REQ1_HURTS
- **STORE4 qwen**: STORE4_HELPS
- **FLOOR_SYS qwen**: FLOOR_SYS_HELPS

## r3bllama — tasks cwe,fwe, `freq_cw=100,alpha=2`; FP {'cwe': 0.48, 'fwe': 0.6}

### AGG (cwe, fwe): 20 units

| arm@B | accuracy [90%] | Δ vs FP [90%] | by task |
|---|---|---|---|
| uniform@3 | +0.750 [+0.650, +0.850] | +0.150 [+0.067, +0.233] | {'fwe': 0.75} |
| uniform+v4@3 | +0.700 [+0.600, +0.817] | +0.100 [+0.033, +0.183] | {'fwe': 0.7} |
| uniform@4 | +0.683 [+0.567, +0.800] | +0.083 [+0.017, +0.167] | {'fwe': 0.683} |
| fp8kv@8 | +0.650 [+0.533, +0.767] | +0.050 [-0.033, +0.133] | {'fwe': 0.65} |
| qread_v4@0.125 | +0.650 [+0.550, +0.750] | +0.050 [-0.050, +0.150] | {'fwe': 0.65} |
| uniform+v4@4 | +0.650 [+0.550, +0.750] | +0.050 [-0.017, +0.117] | {'fwe': 0.65} |
| kvquant4_v4@4 | +0.633 [+0.533, +0.750] | +0.033 [-0.017, +0.100] | {'fwe': 0.633} |
| qread4q_v4@0.125 | +0.617 [+0.517, +0.717] | +0.017 [-0.117, +0.133] | {'fwe': 0.617} |
| fp@0 | +0.600 [+0.500, +0.717] | +0.000 [+0.000, +0.000] | {'fwe': 0.6} |
| fp_noise@0 | +0.600 [+0.500, +0.700] | +0.000 [-0.033, +0.033] | {'fwe': 0.6} |
| qoraclefp_v16@0.125 | +0.600 [+0.517, +0.683] | -0.000 [-0.117, +0.117] | {'fwe': 0.6} |
| qread2t4q_v4@0.125 | +0.600 [+0.483, +0.700] | -0.000 [-0.117, +0.117] | {'fwe': 0.6} |
| kivi4_v4@4 | +0.583 [+0.483, +0.683] | -0.017 [-0.067, +0.033] | {'fwe': 0.583} |
| qread4_v4@0.125 | +0.583 [+0.483, +0.683] | -0.017 [-0.150, +0.117] | {'fwe': 0.583} |
| qreadfp_v16@0.125 | +0.583 [+0.483, +0.683] | -0.017 [-0.133, +0.100] | {'fwe': 0.583} |
| qread2t4kq_v4@0.125 | +0.567 [+0.467, +0.667] | -0.033 [-0.150, +0.083] | {'fwe': 0.567} |

**ACC_SYSTEM** (qread2t4kq_v4@0.125 − fp@0): -0.033 [-0.150, +0.083] → ACC_INCONCLUSIVE
**SYS_VS_BEST_ACC** (qread2t4kq_v4@0.125 − uniform+v4@4): -0.083 [-0.200, +0.017] → SYS_VS_BEST_ACC_NO_EFFECT, NOT_EQUIV
**SIMPLE_VS_BEST_ACC** (qread4_v4@0.125 − uniform+v4@4): -0.067 [-0.183, +0.050] → SIMPLE_VS_BEST_ACC_NO_EFFECT, NOT_EQUIV
**SYS_VS_D_ACC** (qread2t4kq_v4@0.125 − uniform+v4@3): -0.133 [-0.250, -0.017] → SYS_VS_D_ACC_HURTS, NOT_EQUIV
**REQ1_ACC** (qread4q_v4@0.125 − qread4_v4@0.125): +0.033 [-0.033, +0.100] → REQ1_ACC_NO_EFFECT, NOT_EQUIV
**VOTE_LOSS_ACC** (qreadfp_v16@0.125 − qoraclefp_v16@0.125): -0.017 [-0.083, +0.050] → VOTE_LOSS_ACC_NO_EFFECT, NOT_EQUIV

### Per task and arm: accuracy, needle_keep, distractor

| task arm@B | values |
|---|---|
| cwe fp@0 | {'acc': 0.48, 'needle_keep': 1.0, 'distractor': 0.85} |
| cwe fp8kv@8 | {'acc': 0.53, 'needle_keep': 1.0, 'distractor': 0.85} |
| cwe fp_noise@0 | {'acc': 0.47, 'needle_keep': 1.0, 'distractor': 0.85} |
| cwe kivi4_v4@4 | {'acc': 0.47, 'needle_keep': 1.0, 'distractor': 0.8} |
| cwe kvquant4_v4@4 | {'acc': 0.325, 'needle_keep': 1.0, 'distractor': 0.7} |
| cwe qoraclefp_v16@0.125 | {'acc': 0.575, 'needle_keep': 0.257, 'distractor': 0.7} |
| cwe qread2t4kq_v4@0.125 | {'acc': 0.59, 'needle_keep': 0.246, 'distractor': 0.4} |
| cwe qread2t4q_v4@0.125 | {'acc': 0.6, 'needle_keep': 0.246, 'distractor': 0.4} |
| cwe qread4_v4@0.125 | {'acc': 0.62, 'needle_keep': 0.246, 'distractor': 0.3} |
| cwe qread4q_v4@0.125 | {'acc': 0.615, 'needle_keep': 0.246, 'distractor': 0.25} |
| cwe qread_v4@0.125 | {'acc': 0.61, 'needle_keep': 0.24, 'distractor': 0.3} |
| cwe qreadfp_v16@0.125 | {'acc': 0.61, 'needle_keep': 0.248, 'distractor': 0.3} |
| cwe uniform@3 | {'acc': 0.66, 'needle_keep': 1.0, 'distractor': 0.65} |
| cwe uniform@4 | {'acc': 0.47, 'needle_keep': 1.0, 'distractor': 0.8} |
| cwe uniform+v4@3 | {'acc': 0.69, 'needle_keep': 1.0, 'distractor': 0.65} |
| cwe uniform+v4@4 | {'acc': 0.48, 'needle_keep': 1.0, 'distractor': 0.85} |
| fwe fp@0 | {'acc': 0.6, 'needle_keep': 1.0, 'distractor': 0.3} |
| fwe fp8kv@8 | {'acc': 0.65, 'needle_keep': 1.0, 'distractor': 0.35} |
| fwe fp_noise@0 | {'acc': 0.6, 'needle_keep': 1.0, 'distractor': 0.3} |
| fwe kivi4_v4@4 | {'acc': 0.583, 'needle_keep': 1.0, 'distractor': 0.35} |
| fwe kvquant4_v4@4 | {'acc': 0.633, 'needle_keep': 1.0, 'distractor': 0.4} |
| fwe qoraclefp_v16@0.125 | {'acc': 0.6, 'needle_keep': 0.124, 'distractor': 0.8} |
| fwe qread2t4kq_v4@0.125 | {'acc': 0.567, 'needle_keep': 0.126, 'distractor': 0.75} |
| fwe qread2t4q_v4@0.125 | {'acc': 0.6, 'needle_keep': 0.126, 'distractor': 0.75} |
| fwe qread4_v4@0.125 | {'acc': 0.583, 'needle_keep': 0.126, 'distractor': 0.75} |
| fwe qread4q_v4@0.125 | {'acc': 0.617, 'needle_keep': 0.126, 'distractor': 0.85} |
| fwe qread_v4@0.125 | {'acc': 0.65, 'needle_keep': 0.126, 'distractor': 0.65} |
| fwe qreadfp_v16@0.125 | {'acc': 0.583, 'needle_keep': 0.126, 'distractor': 0.8} |
| fwe uniform@3 | {'acc': 0.75, 'needle_keep': 1.0, 'distractor': 0.45} |
| fwe uniform@4 | {'acc': 0.683, 'needle_keep': 1.0, 'distractor': 0.35} |
| fwe uniform+v4@3 | {'acc': 0.7, 'needle_keep': 1.0, 'distractor': 0.45} |
| fwe uniform+v4@4 | {'acc': 0.65, 'needle_keep': 1.0, 'distractor': 0.45} |

NLL / KL points:

| arm@B | lens | ρ | ρ mem | dP vs FP [90%] | NEAR_FP | EQUIV | KL span [90%] | dP vs D_L [90%] | MATCHED | lost / gained (McNemar p) | lost types | dS vs FP | min bias | score | peak GiB |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| qreadfp_v16@0.125 | V16 | 0.21 | 1.67 | +1.296 [+0.931, +1.703] | FAR_FROM_FP | NOT_EQUIV | +0.617 [+0.447, +0.801] | +1.260 [+0.881, +1.683] | WORSE | 6 / 6 (1.000) | {'confused': 6} | +1.274 [+0.913, +1.682] | -0.021 | 0.583 | 45.7 |
| qoraclefp_v16@0.125 | V16 | 0.21 | 1.67 | +0.773 [+0.456, +1.124] | FAR_FROM_FP | NOT_EQUIV | +0.421 [+0.314, +0.541] | +0.738 [+0.381, +1.124] | WORSE | 6 / 5 (1.000) | {'incomplete': 1, 'confused': 5} | +0.773 [+0.456, +1.124] | +0.000 | 0.600 | 45.6 |
| uniform@3 | V16 | — | — | +0.036 [-0.033, +0.107] | INCONCLUSIVE | NOT_EQUIV | +0.028 [+0.018, +0.038] | (D) | — | 0 / 6 (0.031) | — | +0.036 [-0.033, +0.107] | +0.000 | 0.750 | 40.7 |
| uniform@4 | V16 | 1.05 | 1.05 | +0.055 [+0.018, +0.094] | NEAR_FP | EQUIV | +0.009 [+0.006, +0.012] | +0.019 [-0.035, +0.080] | MATCHED | 0 / 3 (0.250) | — | +0.055 [+0.018, +0.094] | +0.000 | 0.683 | 45.6 |
| fp_noise@0 | V16 | 1.67 | 1.67 | +0.007 [-0.023, +0.039] | NEAR_FP | EQUIV | +0.004 [+0.003, +0.006] | -0.029 [-0.090, +0.026] | MATCHED | 1 / 1 (1.000) | {'incomplete': 1} | +0.007 [-0.023, +0.039] | +0.000 | 0.600 | 45.7 |
| qread_v4@0.125 | V4 | 0.13 | 1.00 | +1.203 [+0.863, +1.584] | FAR_FROM_FP | NOT_EQUIV | +0.478 [+0.341, +0.630] | +1.138 [+0.807, +1.501] | WORSE | 3 / 7 (0.344) | {'incomplete': 2, 'confused': 1} | +1.126 [+0.821, +1.469] | -0.078 | 0.650 | 48.4 |
| qread4q_v4@0.125 | V4 | 0.15 | 1.14 | +1.619 [+1.190, +2.090] | FAR_FROM_FP | NOT_EQUIV | +0.729 [+0.537, +0.943] | +1.554 [+1.124, +2.017] | WORSE | 6 / 8 (0.791) | {'confused': 6} | +1.595 [+1.165, +2.070] | -0.025 | 0.617 | 48.4 |
| qread4_v4@0.125 | V4 | 0.15 | 1.14 | +1.367 [+0.986, +1.794] | FAR_FROM_FP | NOT_EQUIV | +0.598 [+0.434, +0.777] | +1.302 [+0.921, +1.722] | WORSE | 6 / 7 (1.000) | {'confused': 5, 'incomplete': 1} | +1.342 [+0.965, +1.769] | -0.025 | 0.583 | 48.4 |
| qread2t4kq_v4@0.125 | V4 | 0.35 | 1.14 | +1.374 [+0.985, +1.811] | FAR_FROM_FP | NOT_EQUIV | +0.606 [+0.439, +0.790] | +1.309 [+0.919, +1.737] | WORSE | 6 / 5 (1.000) | {'confused': 6} | +1.348 [+0.963, +1.784] | -0.026 | 0.567 | 48.5 |
| qread2t4q_v4@0.125 | V4 | 0.55 | 1.14 | +1.293 [+0.925, +1.706] | FAR_FROM_FP | NOT_EQUIV | +0.589 [+0.424, +0.771] | +1.228 [+0.858, +1.635] | WORSE | 5 / 6 (1.000) | {'confused': 5} | +1.277 [+0.911, +1.687] | -0.016 | 0.600 | 48.5 |
| uniform+v4@3 | V4 | — | — | +0.065 [-0.006, +0.141] | INCONCLUSIVE | NOT_EQUIV | +0.040 [+0.025, +0.057] | (D) | — | 0 / 4 (0.125) | — | +0.065 [-0.006, +0.141] | +0.000 | 0.700 | 48.3 |
| uniform+v4@4 | V4 | 1.14 | 1.14 | +0.092 [+0.034, +0.153] | INCONCLUSIVE | NOT_EQUIV | +0.014 [+0.009, +0.020] | +0.027 [-0.029, +0.083] | MATCHED | 1 / 3 (0.625) | {'confused': 1} | +0.092 [+0.034, +0.153] | +0.000 | 0.650 | 48.3 |
| kivi4_v4@4 | V4 | 1.15 | 1.15 | +0.000 [-0.065, +0.059] | NEAR_FP | EQUIV | +0.019 [+0.013, +0.027] | -0.065 [-0.124, -0.012] | MATCHED | 2 / 1 (1.000) | {'incomplete': 1, 'confused': 1} | +0.000 [-0.065, +0.059] | +0.000 | 0.583 | 48.3 |
| kvquant4_v4@4 | V4 | 1.16 | 1.16 | -0.007 [-0.048, +0.036] | NEAR_FP | EQUIV | +0.014 [+0.009, +0.019] | -0.072 [-0.125, -0.018] | MATCHED | 1 / 2 (1.000) | {'confused': 1} | -0.007 [-0.048, +0.036] | +0.000 | 0.633 | 48.3 |
| fp8kv@8 | V8 | 2.20 | 2.20 | +0.002 [-0.028, +0.035] | NEAR_FP | EQUIV | +0.007 [+0.005, +0.010] | -0.063 [-0.150, +0.021] | MATCHED | 1 / 3 (0.625) | {'incomplete': 1} | +0.002 [-0.028, +0.035] | +0.000 | 0.650 | 48.3 |

## r3bqwen — tasks cwe,fwe, `freq_cw=30,alpha=2`; FP {'cwe': 0.8, 'fwe': 0.883}

### AGG (cwe, fwe): 40 units

| arm@B | accuracy [90%] | Δ vs FP [90%] | by task |
|---|---|---|---|
| fp_noise@0 | +0.857 [+0.827, +0.887] | +0.015 [+0.003, +0.028] | {'cwe': 0.83, 'fwe': 0.883} |
| kivi4_v4@4 | +0.854 [+0.828, +0.882] | +0.013 [+0.002, +0.023] | {'cwe': 0.825, 'fwe': 0.883} |
| qoraclefp_v16@0.125 | +0.854 [+0.829, +0.880] | +0.013 [-0.002, +0.028] | {'cwe': 0.825, 'fwe': 0.883} |
| qread2t4kq_v4@0.5 | +0.852 [+0.828, +0.877] | +0.010 [-0.002, +0.023] | {'cwe': 0.82, 'fwe': 0.883} |
| fp8kv@8 | +0.847 [+0.818, +0.876] | +0.005 [-0.002, +0.012] | {'cwe': 0.81, 'fwe': 0.883} |
| uniform@4 | +0.843 [+0.818, +0.871] | +0.002 [-0.009, +0.013] | {'cwe': 0.82, 'fwe': 0.867} |
| fp@0 | +0.842 [+0.812, +0.872] | +0.000 [+0.000, +0.000] | {'cwe': 0.8, 'fwe': 0.883} |
| kvquant4_v4@4 | +0.842 [+0.814, +0.871] | +0.000 [-0.005, +0.005] | {'cwe': 0.8, 'fwe': 0.883} |
| uniform+v4@4 | +0.839 [+0.812, +0.867] | -0.002 [-0.010, +0.005] | {'cwe': 0.795, 'fwe': 0.883} |
| uniform+v4@3 | +0.819 [+0.783, +0.857] | -0.022 [-0.044, -0.002] | {'cwe': 0.805, 'fwe': 0.833} |
| uniform@3 | +0.816 [+0.780, +0.853] | -0.026 [-0.052, -0.000] | {'cwe': 0.815, 'fwe': 0.817} |
| qread2t4kq_v4@0.125 | +0.746 [+0.707, +0.786] | -0.096 [-0.135, -0.057] | {'cwe': 0.675, 'fwe': 0.817} |
| qread2t4q_v4@0.125 | +0.743 [+0.705, +0.782] | -0.098 [-0.137, -0.062] | {'cwe': 0.67, 'fwe': 0.817} |
| qread4_v4@0.125 | +0.738 [+0.703, +0.775] | -0.103 [-0.140, -0.069] | {'cwe': 0.66, 'fwe': 0.817} |
| qread4q_v4@0.125 | +0.738 [+0.703, +0.774] | -0.103 [-0.140, -0.069] | {'cwe': 0.66, 'fwe': 0.817} |
| qreadfp_v16@0.125 | +0.728 [+0.689, +0.768] | -0.113 [-0.153, -0.075] | {'cwe': 0.64, 'fwe': 0.817} |
| qread_v4@0.125 | +0.701 [+0.658, +0.745] | -0.141 [-0.177, -0.106] | {'cwe': 0.635, 'fwe': 0.767} |

**ACC_SYSTEM** (qread2t4kq_v4@0.125 − fp@0): -0.096 [-0.135, -0.057] → ACC_LOSS
**SYS_VS_BEST_ACC** (qread2t4kq_v4@0.125 − kivi4_v4@4): -0.108 [-0.145, -0.073] → SYS_VS_BEST_ACC_HURTS, NOT_EQUIV
**SIMPLE_VS_BEST_ACC** (qread4_v4@0.125 − kivi4_v4@4): -0.116 [-0.148, -0.085] → SIMPLE_VS_BEST_ACC_HURTS, NOT_EQUIV
**SYS_VS_D_ACC** (qread2t4kq_v4@0.125 − uniform+v4@3): -0.073 [-0.122, -0.027] → SYS_VS_D_ACC_HURTS, NOT_EQUIV
**REQ1_ACC** (qread4q_v4@0.125 − qread4_v4@0.125): +0.000 [-0.005, +0.005] → REQ1_ACC_NO_EFFECT, EQUIV
**VOTE_LOSS_ACC** (qreadfp_v16@0.125 − qoraclefp_v16@0.125): -0.126 [-0.158, -0.094] → VOTE_LOSS_ACC_HURTS, NOT_EQUIV

### Per task and arm: accuracy, needle_keep, distractor

| task arm@B | values |
|---|---|
| cwe fp@0 | {'acc': 0.8, 'needle_keep': 1.0, 'distractor': 0.1} |
| cwe fp8kv@8 | {'acc': 0.81, 'needle_keep': 1.0, 'distractor': 0.2} |
| cwe fp_noise@0 | {'acc': 0.83, 'needle_keep': 1.0, 'distractor': 0.15} |
| cwe kivi4_v4@4 | {'acc': 0.825, 'needle_keep': 1.0, 'distractor': 0.15} |
| cwe kvquant4_v4@4 | {'acc': 0.8, 'needle_keep': 1.0, 'distractor': 0.2} |
| cwe qoraclefp_v16@0.125 | {'acc': 0.825, 'needle_keep': 0.347, 'distractor': 0.35} |
| cwe qread2t4kq_v4@0.125 | {'acc': 0.675, 'needle_keep': 0.17, 'distractor': 0.65} |
| cwe qread2t4kq_v4@0.5 | {'acc': 0.82, 'needle_keep': 0.664, 'distractor': 0.35} |
| cwe qread2t4q_v4@0.125 | {'acc': 0.67, 'needle_keep': 0.17, 'distractor': 0.6} |
| cwe qread4_v4@0.125 | {'acc': 0.66, 'needle_keep': 0.17, 'distractor': 0.6} |
| cwe qread4q_v4@0.125 | {'acc': 0.66, 'needle_keep': 0.17, 'distractor': 0.6} |
| cwe qread_v4@0.125 | {'acc': 0.635, 'needle_keep': 0.167, 'distractor': 0.65} |
| cwe qreadfp_v16@0.125 | {'acc': 0.64, 'needle_keep': 0.169, 'distractor': 0.5} |
| cwe uniform@3 | {'acc': 0.815, 'needle_keep': 1.0, 'distractor': 0.5} |
| cwe uniform@4 | {'acc': 0.82, 'needle_keep': 1.0, 'distractor': 0.2} |
| cwe uniform+v4@3 | {'acc': 0.805, 'needle_keep': 1.0, 'distractor': 0.5} |
| cwe uniform+v4@4 | {'acc': 0.795, 'needle_keep': 1.0, 'distractor': 0.25} |
| fwe fp@0 | {'acc': 0.883, 'needle_keep': 1.0, 'distractor': 0.35} |
| fwe fp8kv@8 | {'acc': 0.883, 'needle_keep': 1.0, 'distractor': 0.35} |
| fwe fp_noise@0 | {'acc': 0.883, 'needle_keep': 1.0, 'distractor': 0.35} |
| fwe kivi4_v4@4 | {'acc': 0.883, 'needle_keep': 1.0, 'distractor': 0.35} |
| fwe kvquant4_v4@4 | {'acc': 0.883, 'needle_keep': 1.0, 'distractor': 0.35} |
| fwe qoraclefp_v16@0.125 | {'acc': 0.883, 'needle_keep': 0.125, 'distractor': 0.35} |
| fwe qread2t4kq_v4@0.125 | {'acc': 0.817, 'needle_keep': 0.123, 'distractor': 0.45} |
| fwe qread2t4kq_v4@0.5 | {'acc': 0.883, 'needle_keep': 0.488, 'distractor': 0.35} |
| fwe qread2t4q_v4@0.125 | {'acc': 0.817, 'needle_keep': 0.123, 'distractor': 0.45} |
| fwe qread4_v4@0.125 | {'acc': 0.817, 'needle_keep': 0.123, 'distractor': 0.45} |
| fwe qread4q_v4@0.125 | {'acc': 0.817, 'needle_keep': 0.123, 'distractor': 0.45} |
| fwe qread_v4@0.125 | {'acc': 0.767, 'needle_keep': 0.124, 'distractor': 0.5} |
| fwe qreadfp_v16@0.125 | {'acc': 0.817, 'needle_keep': 0.123, 'distractor': 0.45} |
| fwe uniform@3 | {'acc': 0.817, 'needle_keep': 1.0, 'distractor': 0.4} |
| fwe uniform@4 | {'acc': 0.867, 'needle_keep': 1.0, 'distractor': 0.4} |
| fwe uniform+v4@3 | {'acc': 0.833, 'needle_keep': 1.0, 'distractor': 0.4} |
| fwe uniform+v4@4 | {'acc': 0.883, 'needle_keep': 1.0, 'distractor': 0.35} |

NLL / KL points:

| arm@B | lens | ρ | ρ mem | dP vs FP [90%] | NEAR_FP | EQUIV | KL span [90%] | dP vs D_L [90%] | MATCHED | lost / gained (McNemar p) | lost types | dS vs FP | min bias | score | peak GiB |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| qreadfp_v16@0.125 | V16 | 0.21 | 1.67 | +7.211 [+5.237, +9.338] | FAR_FROM_FP | NOT_EQUIV | +6.286 [+4.783, +7.929] | +4.631 [+2.356, +6.980] | WORSE | 18 / 1 (0.000) | {'confused': 10, 'incomplete': 8} | +6.976 [+4.925, +9.173] | -0.235 | 0.728 | 62.8 |
| qoraclefp_v16@0.125 | V16 | 0.21 | 1.67 | +1.308 [+0.953, +1.671] | FAR_FROM_FP | NOT_EQUIV | +1.722 [+1.392, +2.092] | -1.271 [-2.528, -0.204] | MATCHED | 3 / 6 (0.508) | {'incomplete': 2, 'confused': 1} | +0.438 [+0.123, +0.765] | -0.870 | 0.854 | 62.8 |
| uniform@3 | V16 | — | — | +2.580 [+1.644, +3.677] | FAR_FROM_FP | NOT_EQUIV | +2.282 [+1.291, +3.478] | (D) | — | 9 / 5 (0.424) | {'confused': 5, 'incomplete': 4} | +1.881 [+1.052, +2.860] | -0.699 | 0.816 | 62.0 |
| uniform@4 | V16 | 1.05 | 1.05 | +0.304 [+0.185, +0.421] | FAR_FROM_FP | NOT_EQUIV | +0.247 [+0.196, +0.303] | -2.276 [-3.433, -1.283] | MATCHED | 3 / 5 (0.727) | {'confused': 2, 'incomplete': 1} | +0.185 [+0.078, +0.298] | -0.119 | 0.843 | 62.8 |
| fp_noise@0 | V16 | 1.67 | 1.67 | +0.097 [+0.006, +0.191] | INCONCLUSIVE | NOT_EQUIV | +0.194 [+0.156, +0.234] | -2.482 [-3.612, -1.515] | MATCHED | 1 / 5 (0.219) | {'confused': 1} | -0.028 [-0.127, +0.072] | -0.125 | 0.857 | 62.8 |
| qread_v4@0.125 | V4 | 0.14 | 1.00 | +9.274 [+7.267, +11.355] | FAR_FROM_FP | NOT_EQUIV | +7.988 [+6.376, +9.722] | +6.628 [+4.903, +8.479] | WORSE | 23 / 1 (0.000) | {'confused': 15, 'incomplete': 8} | +9.088 [+7.004, +11.241] | -0.186 | 0.701 | 63.5 |
| qread4q_v4@0.125 | V4 | 0.15 | 1.14 | +7.269 [+5.389, +9.319] | FAR_FROM_FP | NOT_EQUIV | +6.305 [+4.854, +7.953] | +4.623 [+2.407, +6.985] | WORSE | 18 / 1 (0.000) | {'confused': 10, 'incomplete': 8} | +6.931 [+4.952, +9.088] | -0.338 | 0.738 | 63.5 |
| qread4_v4@0.125 | V4 | 0.15 | 1.14 | +7.047 [+5.156, +9.127] | FAR_FROM_FP | NOT_EQUIV | +6.145 [+4.671, +7.823] | +4.401 [+2.129, +6.806] | WORSE | 18 / 1 (0.000) | {'confused': 11, 'incomplete': 7} | +6.682 [+4.663, +8.874] | -0.365 | 0.738 | 63.5 |
| qread2t4kq_v4@0.125 | V4 | 0.36 | 1.14 | +6.919 [+5.035, +8.963] | FAR_FROM_FP | NOT_EQUIV | +6.134 [+4.683, +7.795] | +4.272 [+2.046, +6.601] | WORSE | 18 / 2 (0.000) | {'confused': 11, 'incomplete': 7} | +6.538 [+4.540, +8.692] | -0.381 | 0.746 | 63.5 |
| qread2t4q_v4@0.125 | V4 | 0.56 | 1.14 | +6.982 [+5.089, +9.029] | FAR_FROM_FP | NOT_EQUIV | +6.163 [+4.716, +7.798] | +4.336 [+2.135, +6.653] | WORSE | 18 / 2 (0.000) | {'confused': 10, 'incomplete': 8} | +6.592 [+4.583, +8.754] | -0.390 | 0.743 | 63.5 |
| uniform+v4@3 | V4 | — | — | +2.646 [+1.649, +3.764] | FAR_FROM_FP | NOT_EQUIV | +2.365 [+1.342, +3.571] | (D) | — | 7 / 3 (0.344) | {'confused': 5, 'incomplete': 2} | +1.957 [+1.043, +2.989] | -0.689 | 0.819 | 63.4 |
| uniform+v4@4 | V4 | 1.14 | 1.14 | +0.317 [+0.199, +0.429] | FAR_FROM_FP | NOT_EQUIV | +0.258 [+0.210, +0.309] | -2.329 [-3.525, -1.274] | MATCHED | 3 / 2 (1.000) | {'incomplete': 1, 'confused': 2} | +0.163 [+0.028, +0.298] | -0.154 | 0.839 | 63.4 |
| kivi4_v4@4 | V4 | 1.15 | 1.15 | +0.036 [-0.020, +0.092] | NEAR_FP | EQUIV | +0.082 [+0.067, +0.097] | -2.610 [-3.744, -1.603] | MATCHED | 0 / 3 (0.250) | — | -0.189 [-0.317, -0.075] | -0.225 | 0.854 | 63.4 |
| kvquant4_v4@4 | V4 | 1.16 | 1.16 | +0.063 [-0.010, +0.139] | INCONCLUSIVE | NOT_EQUIV | +0.127 [+0.103, +0.152] | -2.583 [-3.718, -1.580] | MATCHED | 1 / 1 (1.000) | {'confused': 1} | -0.037 [-0.144, +0.063] | -0.099 | 0.842 | 63.4 |
| qread2t4kq_v4@0.5 | V4 | 1.38 | 1.14 | +0.229 [+0.060, +0.416] | INCONCLUSIVE | NOT_EQUIV | +0.324 [+0.238, +0.419] | -2.417 [-3.605, -1.345] | MATCHED | 2 / 5 (0.453) | {'confused': 2} | +0.134 [-0.042, +0.328] | -0.095 | 0.852 | 63.5 |
| fp8kv@8 | V8 | 2.19 | 2.19 | +0.058 [-0.018, +0.132] | INCONCLUSIVE | NOT_EQUIV | +0.079 [+0.067, +0.091] | -2.588 [-3.726, -1.587] | MATCHED | 1 / 3 (0.625) | {'incomplete': 1} | -0.039 [-0.169, +0.074] | -0.096 | 0.847 | 63.4 |

