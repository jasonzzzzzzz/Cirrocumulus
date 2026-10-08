# R14 Stage 1h — R3a, harder synthetic tasks (read_stage1h_r3.py; rules frozen in its docstring)

EXPLORATORY run: labels guide the design; they are not claims.

- **ACC_SYSTEM llama**: ACC_NEAR_FP (+0.022 [-0.011, +0.056])
- **SYS_VS_BEST_ACC llama**: SYS_VS_BEST_ACC_NO_EFFECT, NOT_EQUIV (-0.002 [-0.037, +0.036])
- **SIMPLE_VS_BEST_ACC llama**: SIMPLE_VS_BEST_ACC_NO_EFFECT, NOT_EQUIV (-0.005 [-0.056, +0.040])
- **SYS_VS_D_ACC llama**: SYS_VS_D_ACC_NO_EFFECT, NOT_EQUIV (+0.032 [-0.017, +0.082])
- **REQ1_ACC llama**: REQ1_ACC_NO_EFFECT, NOT_EQUIV (-0.021 [-0.052, +0.010])
- **VOTE_LOSS_ACC llama**: VOTE_LOSS_ACC_HELPS, NOT_EQUIV (+0.028 [+0.006, +0.052])
- **BEST_DENSE_ACC llama**: kvquant4_v4@4
- **FP accuracy llama**: {'mk_panel': 1.0, 'niah_multikey': 0.95, 'niah_multivalue': 0.744, 'vt': 0.796}
- **SYSTEM NLL llama**: FAR_FROM_FP at the cell's margin 0.100 (+0.441 [+0.238, +0.687]); vs D_V4 MATCHED
- **SYS_VS_BEST llama**: SYS_VS_BEST_NO_EFFECT
- **REQ1 llama**: REQ1_HURTS
- **STORE4 llama**: STORE4_HELPS
- **ACC_SYSTEM qwen**: ACC_LOSS (-0.135 [-0.166, -0.107])
- **SYS_VS_BEST_ACC qwen**: SYS_VS_BEST_ACC_HURTS, NOT_EQUIV (-0.138 [-0.170, -0.109])
- **SIMPLE_VS_BEST_ACC qwen**: SIMPLE_VS_BEST_ACC_HURTS, NOT_EQUIV (-0.158 [-0.194, -0.126])
- **SYS_VS_D_ACC qwen**: SYS_VS_D_ACC_HURTS, NOT_EQUIV (-0.122 [-0.158, -0.087])
- **REQ1_ACC qwen**: REQ1_ACC_NO_EFFECT, EQUIV (+0.009 [-0.004, +0.026])
- **VOTE_LOSS_ACC qwen**: VOTE_LOSS_ACC_HURTS, NOT_EQUIV (-0.141 [-0.176, -0.110])
- **BEST_DENSE_ACC qwen**: kivi4_v4@4
- **FP accuracy qwen**: {'mk_panel': 1.0, 'niah_multikey': 1.0, 'niah_multivalue': 0.969, 'vt': 0.982}
- **SYSTEM NLL qwen**: FAR_FROM_FP at the cell's margin 0.100 (+18.735 [+16.248, +21.342]); vs D_V4 WORSE
- **SYS_VS_BEST qwen**: SYS_VS_BEST_HURTS
- **REQ1 qwen**: REQ1_NO_EFFECT
- **STORE4 qwen**: STORE4_HELPS
- **FLOOR_SYS qwen**: FLOOR_SYS_HELPS

## r3llama — n_keys=64,n_values=16,n_hops=12; FP accuracy {'mk_panel': 1.0, 'niah_multikey': 0.95, 'niah_multivalue': 0.744, 'vt': 0.796}; 80 units

| arm@B | accuracy [90%] | Δ vs FP [90%] | by task |
|---|---|---|---|
| qreadfp_v16@0.125 | +0.898 [+0.864, +0.932] | +0.026 [-0.003, +0.057] | {'mk_panel': 1.0, 'niah_multikey': 0.95, 'niah_multivalue': 0.812, 'vt': 0.831} |
| kvquant4_v4@4 | +0.896 [+0.858, +0.931] | +0.024 [-0.003, +0.057] | {'mk_panel': 1.0, 'niah_multikey': 0.95, 'niah_multivalue': 0.784, 'vt': 0.85} |
| qread2t4kq_v4@0.125 | +0.894 [+0.864, +0.924] | +0.022 [-0.011, +0.056] | {'mk_panel': 1.0, 'niah_multikey': 1.0, 'niah_multivalue': 0.803, 'vt': 0.773} |
| qread4_v4@0.125 | +0.891 [+0.847, +0.930] | +0.019 [-0.007, +0.047] | {'mk_panel': 1.0, 'niah_multikey': 1.0, 'niah_multivalue': 0.784, 'vt': 0.781} |
| kivi4_v4@4 | +0.886 [+0.846, +0.922] | +0.014 [-0.011, +0.042] | {'mk_panel': 1.0, 'niah_multikey': 0.95, 'niah_multivalue': 0.741, 'vt': 0.854} |
| fp_noise@0 | +0.873 [+0.834, +0.908] | +0.000 [-0.004, +0.006] | {'mk_panel': 1.0, 'niah_multikey': 0.95, 'niah_multivalue': 0.738, 'vt': 0.804} |
| qread2t4q_v4@0.125 | +0.873 [+0.832, +0.911] | +0.000 [-0.026, +0.027] | {'mk_panel': 1.0, 'niah_multikey': 0.95, 'niah_multivalue': 0.759, 'vt': 0.781} |
| fp@0 | +0.872 [+0.831, +0.910] | +0.000 [+0.000, +0.000] | {'mk_panel': 1.0, 'niah_multikey': 0.95, 'niah_multivalue': 0.744, 'vt': 0.796} |
| qoraclefp_v16@0.125 | +0.871 [+0.830, +0.909] | -0.002 [-0.027, +0.023] | {'mk_panel': 1.0, 'niah_multikey': 0.95, 'niah_multivalue': 0.759, 'vt': 0.773} |
| fp8kv@8 | +0.871 [+0.830, +0.907] | -0.002 [-0.015, +0.011] | {'mk_panel': 1.0, 'niah_multikey': 0.95, 'niah_multivalue': 0.744, 'vt': 0.788} |
| qread4q_v4@0.125 | +0.871 [+0.826, +0.910] | -0.002 [-0.029, +0.023] | {'mk_panel': 1.0, 'niah_multikey': 1.0, 'niah_multivalue': 0.694, 'vt': 0.788} |
| qread_v4@0.125 | +0.866 [+0.828, +0.901] | -0.007 [-0.042, +0.030] | {'mk_panel': 1.0, 'niah_multikey': 1.0, 'niah_multivalue': 0.694, 'vt': 0.769} |
| uniform@4 | +0.863 [+0.820, +0.902] | -0.010 [-0.030, +0.008] | {'mk_panel': 1.0, 'niah_multikey': 0.95, 'niah_multivalue': 0.697, 'vt': 0.804} |
| uniform+v4@3 | +0.862 [+0.818, +0.902] | -0.010 [-0.044, +0.024] | {'mk_panel': 1.0, 'niah_multikey': 0.95, 'niah_multivalue': 0.722, 'vt': 0.777} |
| uniform+v4@4 | +0.860 [+0.817, +0.899] | -0.013 [-0.033, +0.005] | {'mk_panel': 1.0, 'niah_multikey': 0.95, 'niah_multivalue': 0.684, 'vt': 0.804} |
| uniform@3 | +0.849 [+0.806, +0.889] | -0.024 [-0.049, +0.004] | {'mk_panel': 1.0, 'niah_multikey': 0.95, 'niah_multivalue': 0.688, 'vt': 0.758} |

**ACC_SYSTEM** (qread2t4kq_v4@0.125 − fp@0): +0.022 [-0.011, +0.056] → ACC_NEAR_FP
**SYS_VS_BEST_ACC** (qread2t4kq_v4@0.125 − kvquant4_v4@4): -0.002 [-0.037, +0.036] → SYS_VS_BEST_ACC_NO_EFFECT, NOT_EQUIV
**SIMPLE_VS_BEST_ACC** (qread4_v4@0.125 − kvquant4_v4@4): -0.005 [-0.056, +0.040] → SIMPLE_VS_BEST_ACC_NO_EFFECT, NOT_EQUIV
**SYS_VS_D_ACC** (qread2t4kq_v4@0.125 − uniform+v4@3): +0.032 [-0.017, +0.082] → SYS_VS_D_ACC_NO_EFFECT, NOT_EQUIV
**REQ1_ACC** (qread4q_v4@0.125 − qread4_v4@0.125): -0.021 [-0.052, +0.010] → REQ1_ACC_NO_EFFECT, NOT_EQUIV
**VOTE_LOSS_ACC** (qreadfp_v16@0.125 − qoraclefp_v16@0.125): +0.028 [+0.006, +0.052] → VOTE_LOSS_ACC_HELPS, NOT_EQUIV
**mk_panel** (accuracy, confusion): {'fp@0': {'acc': 1.0, 'confused': 0.0}, 'fp8kv@8': {'acc': 1.0, 'confused': 0.0}, 'fp_noise@0': {'acc': 1.0, 'confused': 0.0}, 'kivi4_v4@4': {'acc': 1.0, 'confused': 0.0}, 'kvquant4_v4@4': {'acc': 1.0, 'confused': 0.0}, 'qoraclefp_v16@0.125': {'acc': 1.0, 'confused': 0.0}, 'qread2t4kq_v4@0.125': {'acc': 1.0, 'confused': 0.0}, 'qread2t4q_v4@0.125': {'acc': 1.0, 'confused': 0.0}, 'qread4_v4@0.125': {'acc': 1.0, 'confused': 0.0}, 'qread4q_v4@0.125': {'acc': 1.0, 'confused': 0.0}, 'qread_v4@0.125': {'acc': 1.0, 'confused': 0.0}, 'qreadfp_v16@0.125': {'acc': 1.0, 'confused': 0.0}, 'uniform@3': {'acc': 1.0, 'confused': 0.0}, 'uniform@4': {'acc': 1.0, 'confused': 0.0}, 'uniform+v4@3': {'acc': 1.0, 'confused': 0.0}, 'uniform+v4@4': {'acc': 1.0, 'confused': 0.0}}

NLL / KL points:

| arm@B | lens | ρ | ρ mem | dP vs FP [90%] | NEAR_FP | EQUIV | KL span [90%] | dP vs D_L [90%] | MATCHED | lost / gained (McNemar p) | lost types | dS vs FP | min bias | score | peak GiB |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| qreadfp_v16@0.125 | V16 | 0.21 | 1.67 | +0.374 [+0.180, +0.587] | FAR_FROM_FP | NOT_EQUIV | +0.551 [+0.376, +0.739] | -1.164 [-1.487, -0.836] | MATCHED | 5 / 13 (0.096) | {'incomplete': 5} | +0.268 [+0.043, +0.509] | -0.105 | 0.898 | 45.8 |
| qoraclefp_v16@0.125 | V16 | 0.21 | 1.67 | +0.274 [+0.132, +0.428] | FAR_FROM_FP | NOT_EQUIV | +0.260 [+0.188, +0.340] | -1.264 [-1.632, -0.897] | MATCHED | 7 / 6 (1.000) | {'incomplete': 7} | +0.232 [+0.063, +0.403] | -0.043 | 0.871 | 45.8 |
| uniform@3 | V16 | — | — | +1.538 [+1.233, +1.840] | FAR_FROM_FP | NOT_EQUIV | +1.393 [+1.163, +1.623] | (D) | — | 14 / 7 (0.189) | {'incomplete': 14} | +1.366 [+1.026, +1.695] | -0.175 | 0.849 | 41.1 |
| uniform@4 | V16 | 1.05 | 1.05 | +0.384 [+0.227, +0.560] | FAR_FROM_FP | NOT_EQUIV | +0.363 [+0.268, +0.471] | -1.154 [-1.429, -0.868] | MATCHED | 8 / 7 (1.000) | {'incomplete': 8} | +0.340 [+0.165, +0.525] | -0.045 | 0.863 | 45.8 |
| fp_noise@0 | V16 | 1.67 | 1.67 | +0.011 [-0.012, +0.033] | NEAR_FP | EQUIV | +0.019 [+0.013, +0.026] | -1.527 [-1.830, -1.218] | MATCHED | 2 / 1 (1.000) | {'incomplete': 2} | -0.021 [-0.060, +0.011] | -0.031 | 0.873 | 45.9 |
| qread_v4@0.125 | V4 | 0.13 | 1.00 | +1.687 [+1.257, +2.119] | FAR_FROM_FP | NOT_EQUIV | +1.625 [+1.320, +1.944] | +0.116 [-0.069, +0.306] | INCONCLUSIVE | 17 / 8 (0.108) | {'incomplete': 17} | +1.475 [+1.043, +1.904] | -0.215 | 0.866 | 48.9 |
| qread4_v4@0.125 | V4 | 0.15 | 1.14 | +0.684 [+0.419, +0.954] | FAR_FROM_FP | NOT_EQUIV | +0.822 [+0.631, +1.025] | -0.887 [-1.244, -0.513] | MATCHED | 8 / 7 (1.000) | {'incomplete': 8} | +0.442 [+0.131, +0.760] | -0.246 | 0.891 | 48.9 |
| qread4q_v4@0.125 | V4 | 0.15 | 1.14 | +0.746 [+0.481, +1.020] | FAR_FROM_FP | NOT_EQUIV | +0.861 [+0.664, +1.073] | -0.825 [-1.189, -0.440] | MATCHED | 10 / 7 (0.629) | {'incomplete': 10} | +0.550 [+0.248, +0.851] | -0.199 | 0.871 | 48.9 |
| qread2t4kq_v4@0.125 | V4 | 0.35 | 1.14 | +0.441 [+0.238, +0.687] | FAR_FROM_FP | NOT_EQUIV | +0.555 [+0.407, +0.718] | -1.130 [-1.496, -0.751] | MATCHED | 8 / 10 (0.815) | {'incomplete': 8} | +0.291 [+0.060, +0.557] | -0.152 | 0.894 | 49.0 |
| qread2t4q_v4@0.125 | V4 | 0.55 | 1.14 | +0.401 [+0.202, +0.637] | FAR_FROM_FP | NOT_EQUIV | +0.566 [+0.404, +0.741] | -1.170 [-1.539, -0.794] | MATCHED | 7 / 11 (0.481) | {'incomplete': 7} | +0.241 [+0.009, +0.506] | -0.163 | 0.873 | 49.0 |
| uniform+v4@3 | V4 | — | — | +1.571 [+1.251, +1.889] | FAR_FROM_FP | NOT_EQUIV | +1.407 [+1.178, +1.634] | (D) | — | 14 / 7 (0.189) | {'incomplete': 14} | +1.337 [+1.008, +1.660] | -0.238 | 0.862 | 48.8 |
| uniform+v4@4 | V4 | 1.14 | 1.14 | +0.439 [+0.266, +0.635] | FAR_FROM_FP | NOT_EQUIV | +0.401 [+0.295, +0.517] | -1.132 [-1.419, -0.839] | MATCHED | 9 / 6 (0.607) | {'incomplete': 9} | +0.324 [+0.114, +0.546] | -0.117 | 0.860 | 48.8 |
| kivi4_v4@4 | V4 | 1.15 | 1.15 | +0.353 [+0.254, +0.450] | FAR_FROM_FP | NOT_EQUIV | +0.282 [+0.244, +0.323] | -1.218 [-1.525, -0.919] | MATCHED | 8 / 7 (1.000) | {'incomplete': 8} | +0.273 [+0.147, +0.390] | -0.081 | 0.886 | 48.7 |
| kvquant4_v4@4 | V4 | 1.16 | 1.16 | +0.604 [+0.455, +0.755] | FAR_FROM_FP | NOT_EQUIV | +0.407 [+0.344, +0.470] | -0.968 [-1.273, -0.652] | MATCHED | 9 / 10 (1.000) | {'incomplete': 9} | +0.386 [+0.201, +0.558] | -0.205 | 0.896 | 48.7 |
| fp8kv@8 | V8 | 2.20 | 2.20 | +0.180 [+0.082, +0.290] | INCONCLUSIVE | NOT_EQUIV | +0.160 [+0.117, +0.216] | -1.392 [-1.721, -1.064] | MATCHED | 9 / 4 (0.267) | {'incomplete': 9} | +0.132 [+0.017, +0.255] | -0.048 | 0.871 | 48.8 |

## r3qwen — n_keys=64,n_values=24,n_hops=16; FP accuracy {'mk_panel': 1.0, 'niah_multikey': 1.0, 'niah_multivalue': 0.969, 'vt': 0.982}; 80 units

| arm@B | accuracy [90%] | Δ vs FP [90%] | by task |
|---|---|---|---|
| kivi4_v4@4 | +0.991 [+0.984, +0.997] | +0.003 [+0.000, +0.007] | {'mk_panel': 1.0, 'niah_multikey': 1.0, 'niah_multivalue': 0.969, 'vt': 0.994} |
| uniform@4 | +0.990 [+0.984, +0.995] | +0.003 [-0.003, +0.008] | {'mk_panel': 1.0, 'niah_multikey': 1.0, 'niah_multivalue': 0.973, 'vt': 0.988} |
| fp_noise@0 | +0.990 [+0.983, +0.996] | +0.002 [-0.001, +0.005] | {'mk_panel': 1.0, 'niah_multikey': 1.0, 'niah_multivalue': 0.969, 'vt': 0.991} |
| qoraclefp_v16@0.125 | +0.989 [+0.982, +0.995] | +0.001 [-0.003, +0.006] | {'mk_panel': 1.0, 'niah_multikey': 1.0, 'niah_multivalue': 0.971, 'vt': 0.985} |
| fp8kv@8 | +0.989 [+0.981, +0.995] | +0.001 [-0.002, +0.003] | {'mk_panel': 1.0, 'niah_multikey': 1.0, 'niah_multivalue': 0.969, 'vt': 0.985} |
| fp@0 | +0.988 [+0.981, +0.994] | +0.000 [+0.000, +0.000] | {'mk_panel': 1.0, 'niah_multikey': 1.0, 'niah_multivalue': 0.969, 'vt': 0.982} |
| qread2t4kq_v4@0.5 | +0.987 [+0.980, +0.994] | -0.001 [-0.004, +0.004] | {'mk_panel': 1.0, 'niah_multikey': 1.0, 'niah_multivalue': 0.954, 'vt': 0.994} |
| uniform+v4@4 | +0.986 [+0.979, +0.993] | -0.001 [-0.004, +0.001] | {'mk_panel': 1.0, 'niah_multikey': 1.0, 'niah_multivalue': 0.969, 'vt': 0.976} |
| kvquant4_v4@4 | +0.985 [+0.978, +0.992] | -0.003 [-0.005, -0.001] | {'mk_panel': 1.0, 'niah_multikey': 1.0, 'niah_multivalue': 0.965, 'vt': 0.976} |
| uniform@3 | +0.976 [+0.953, +0.991] | -0.012 [-0.035, +0.003] | {'mk_panel': 1.0, 'niah_multikey': 1.0, 'niah_multivalue': 0.965, 'vt': 0.938} |
| uniform+v4@3 | +0.975 [+0.952, +0.990] | -0.013 [-0.036, +0.002] | {'mk_panel': 1.0, 'niah_multikey': 1.0, 'niah_multivalue': 0.96, 'vt': 0.938} |
| qread2t4q_v4@0.125 | +0.855 [+0.823, +0.884] | -0.133 [-0.163, -0.106] | {'mk_panel': 0.85, 'niah_multikey': 1.0, 'niah_multivalue': 0.617, 'vt': 0.953} |
| qread2t4kq_v4@0.125 | +0.853 [+0.819, +0.882] | -0.135 [-0.166, -0.107] | {'mk_panel': 0.85, 'niah_multikey': 1.0, 'niah_multivalue': 0.61, 'vt': 0.95} |
| qreadfp_v16@0.125 | +0.848 [+0.812, +0.880] | -0.140 [-0.173, -0.110] | {'mk_panel': 0.85, 'niah_multikey': 1.0, 'niah_multivalue': 0.604, 'vt': 0.938} |
| qread4q_v4@0.125 | +0.842 [+0.805, +0.875] | -0.146 [-0.180, -0.116] | {'mk_panel': 0.85, 'niah_multikey': 1.0, 'niah_multivalue': 0.602, 'vt': 0.915} |
| qread4_v4@0.125 | +0.833 [+0.795, +0.867] | -0.155 [-0.189, -0.125] | {'mk_panel': 0.8, 'niah_multikey': 1.0, 'niah_multivalue': 0.604, 'vt': 0.926} |
| qread_v4@0.125 | +0.828 [+0.785, +0.867] | -0.160 [-0.200, -0.123] | {'mk_panel': 0.85, 'niah_multikey': 1.0, 'niah_multivalue': 0.569, 'vt': 0.894} |

**ACC_SYSTEM** (qread2t4kq_v4@0.125 − fp@0): -0.135 [-0.166, -0.107] → ACC_LOSS
**SYS_VS_BEST_ACC** (qread2t4kq_v4@0.125 − kivi4_v4@4): -0.138 [-0.170, -0.109] → SYS_VS_BEST_ACC_HURTS, NOT_EQUIV
**SIMPLE_VS_BEST_ACC** (qread4_v4@0.125 − kivi4_v4@4): -0.158 [-0.194, -0.126] → SIMPLE_VS_BEST_ACC_HURTS, NOT_EQUIV
**SYS_VS_D_ACC** (qread2t4kq_v4@0.125 − uniform+v4@3): -0.122 [-0.158, -0.087] → SYS_VS_D_ACC_HURTS, NOT_EQUIV
**REQ1_ACC** (qread4q_v4@0.125 − qread4_v4@0.125): +0.009 [-0.004, +0.026] → REQ1_ACC_NO_EFFECT, EQUIV
**VOTE_LOSS_ACC** (qreadfp_v16@0.125 − qoraclefp_v16@0.125): -0.141 [-0.176, -0.110] → VOTE_LOSS_ACC_HURTS, NOT_EQUIV
**mk_panel** (accuracy, confusion): {'fp@0': {'acc': 1.0, 'confused': 0.0}, 'fp8kv@8': {'acc': 1.0, 'confused': 0.0}, 'fp_noise@0': {'acc': 1.0, 'confused': 0.0}, 'kivi4_v4@4': {'acc': 1.0, 'confused': 0.0}, 'kvquant4_v4@4': {'acc': 1.0, 'confused': 0.0}, 'qoraclefp_v16@0.125': {'acc': 1.0, 'confused': 0.0}, 'qread2t4kq_v4@0.125': {'acc': 0.85, 'confused': 0.0}, 'qread2t4kq_v4@0.5': {'acc': 1.0, 'confused': 0.0}, 'qread2t4q_v4@0.125': {'acc': 0.85, 'confused': 0.0}, 'qread4_v4@0.125': {'acc': 0.8, 'confused': 0.0}, 'qread4q_v4@0.125': {'acc': 0.85, 'confused': 0.0}, 'qread_v4@0.125': {'acc': 0.85, 'confused': 0.0}, 'qreadfp_v16@0.125': {'acc': 0.85, 'confused': 0.0}, 'uniform@3': {'acc': 1.0, 'confused': 0.0}, 'uniform@4': {'acc': 1.0, 'confused': 0.0}, 'uniform+v4@3': {'acc': 1.0, 'confused': 0.0}, 'uniform+v4@4': {'acc': 1.0, 'confused': 0.0}}

NLL / KL points:

| arm@B | lens | ρ | ρ mem | dP vs FP [90%] | NEAR_FP | EQUIV | KL span [90%] | dP vs D_L [90%] | MATCHED | lost / gained (McNemar p) | lost types | dS vs FP | min bias | score | peak GiB |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| qreadfp_v16@0.125 | V16 | 0.21 | 1.67 | +19.019 [+16.507, +21.567] | FAR_FROM_FP | NOT_EQUIV | +19.204 [+16.661, +21.754] | +18.074 [+15.425, +20.736] | WORSE | 34 / 1 (0.000) | {'incomplete': 33, 'other': 1} | +18.933 [+16.388, +21.496] | -0.086 | 0.848 | 63.1 |
| qoraclefp_v16@0.125 | V16 | 0.21 | 1.67 | +0.168 [-0.130, +0.467] | INCONCLUSIVE | NOT_EQUIV | +0.629 [+0.255, +1.086] | -0.777 [-1.305, -0.280] | MATCHED | 4 / 4 (1.000) | {'incomplete': 4} | -0.016 [-0.244, +0.167] | -0.184 | 0.989 | 63.1 |
| uniform@3 | V16 | — | — | +0.945 [+0.445, +1.477] | FAR_FROM_FP | NOT_EQUIV | +1.442 [+0.919, +2.029] | (D) | — | 10 / 4 (0.180) | {'incomplete': 10} | +0.581 [+0.198, +0.989] | -0.364 | 0.976 | 62.4 |
| uniform@4 | V16 | 1.05 | 1.05 | +0.026 [-0.216, +0.246] | INCONCLUSIVE | NOT_EQUIV | +0.408 [+0.146, +0.741] | -0.919 [-1.341, -0.540] | MATCHED | 3 / 4 (1.000) | {'incomplete': 3} | -0.033 [-0.281, +0.195] | -0.059 | 0.990 | 63.1 |
| fp_noise@0 | V16 | 1.67 | 1.67 | -0.002 [-0.190, +0.159] | INCONCLUSIVE | NOT_EQUIV | +0.191 [+0.061, +0.348] | -0.947 [-1.369, -0.563] | MATCHED | 1 / 3 (0.625) | {'incomplete': 1} | -0.080 [-0.264, +0.045] | -0.078 | 0.990 | 63.1 |
| qread_v4@0.125 | V4 | 0.14 | 1.00 | +22.100 [+19.598, +24.656] | FAR_FROM_FP | NOT_EQUIV | +22.388 [+19.743, +25.141] | +21.180 [+18.624, +23.844] | WORSE | 34 / 1 (0.000) | {'incomplete': 33, 'other': 1} | +21.879 [+19.404, +24.417] | -0.222 | 0.828 | 64.0 |
| qread4_v4@0.125 | V4 | 0.16 | 1.13 | +19.357 [+16.801, +22.018] | FAR_FROM_FP | NOT_EQUIV | +19.621 [+17.091, +22.223] | +18.437 [+15.789, +21.134] | WORSE | 36 / 2 (0.000) | {'incomplete': 35, 'other': 1} | +19.347 [+16.787, +22.008] | -0.010 | 0.833 | 64.0 |
| qread4q_v4@0.125 | V4 | 0.16 | 1.13 | +19.598 [+16.999, +22.282] | FAR_FROM_FP | NOT_EQUIV | +19.814 [+17.232, +22.462] | +18.678 [+16.014, +21.420] | WORSE | 37 / 0 (0.000) | {'incomplete': 36, 'other': 1} | +19.596 [+16.995, +22.281] | -0.001 | 0.842 | 63.9 |
| qread2t4kq_v4@0.125 | V4 | 0.36 | 1.13 | +18.735 [+16.248, +21.342] | FAR_FROM_FP | NOT_EQUIV | +19.014 [+16.541, +21.556] | +17.815 [+15.243, +20.449] | WORSE | 32 / 1 (0.000) | {'incomplete': 31, 'other': 1} | +18.605 [+16.068, +21.250] | -0.130 | 0.853 | 64.0 |
| qread2t4q_v4@0.125 | V4 | 0.56 | 1.13 | +18.974 [+16.537, +21.523] | FAR_FROM_FP | NOT_EQUIV | +19.274 [+16.848, +21.778] | +18.054 [+15.538, +20.654] | WORSE | 32 / 1 (0.000) | {'incomplete': 31, 'other': 1} | +18.815 [+16.325, +21.419] | -0.159 | 0.855 | 64.0 |
| uniform+v4@3 | V4 | — | — | +0.920 [+0.464, +1.399] | FAR_FROM_FP | NOT_EQUIV | +1.382 [+0.899, +1.930] | (D) | — | 11 / 4 (0.118) | {'incomplete': 11} | +0.598 [+0.220, +1.004] | -0.322 | 0.975 | 63.9 |
| uniform+v4@4 | V4 | 1.13 | 1.13 | +0.133 [-0.061, +0.339] | INCONCLUSIVE | NOT_EQUIV | +0.375 [+0.191, +0.581] | -0.788 [-1.144, -0.455] | MATCHED | 4 / 2 (0.688) | {'incomplete': 4} | +0.033 [-0.160, +0.241] | -0.100 | 0.986 | 63.9 |
| kivi4_v4@4 | V4 | 1.15 | 1.15 | +0.090 [-0.017, +0.209] | INCONCLUSIVE | NOT_EQUIV | +0.312 [+0.162, +0.484] | -0.830 [-1.264, -0.418] | MATCHED | 0 / 2 (0.500) | — | -0.023 [-0.081, +0.037] | -0.114 | 0.991 | 63.9 |
| kvquant4_v4@4 | V4 | 1.16 | 1.16 | +0.182 [-0.074, +0.408] | INCONCLUSIVE | NOT_EQUIV | +0.352 [+0.221, +0.494] | -0.738 [-1.094, -0.403] | MATCHED | 5 / 1 (0.219) | {'incomplete': 5} | +0.001 [-0.228, +0.209] | -0.181 | 0.985 | 63.9 |
| qread2t4kq_v4@0.5 | V4 | 1.38 | 1.13 | +1.112 [+0.380, +1.987] | FAR_FROM_FP | NOT_EQUIV | +1.501 [+0.680, +2.465] | +0.192 [-0.656, +1.109] | INCONCLUSIVE | 8 / 3 (0.227) | {'incomplete': 8} | +0.967 [+0.239, +1.839] | -0.146 | 0.987 | 64.0 |
| fp8kv@8 | V8 | 2.18 | 2.18 | +0.001 [-0.180, +0.144] | INCONCLUSIVE | NOT_EQUIV | +0.159 [+0.093, +0.234] | -0.919 [-1.334, -0.535] | MATCHED | 2 / 3 (1.000) | {'incomplete': 2} | -0.067 [-0.237, +0.067] | -0.068 | 0.989 | 63.9 |

