# R14 Stage 1e — E1/E2 (read_stage1e.py; rules frozen in its docstring)

- **E1 llama31-8b@32768 r=0.125**: NO_GAIN
- **E1 llama31-8b@32768 r=0.25**: NO_GAIN
- **E2 llama31-8b@32768**: TAIL_FIXED
- **E2 ROUTER_32K**: WIN: nest3

## llama31-8b@32768 — FP 0.988 ({'niah_multikey': 0.975, 'niah_multivalue': 0.975, 'niah_single': 1.0, 'vt': 1.0}), 40 prompts, 159 prompt-tasks (dropped 0), fp replay 0.994, answer-value coverage 1.00

| lens | qread | seq2 | nest2 | seq3 | nest3 | sieve | pool | snapq |
|---|---|---|---|---|---|---|---|---|
| V16 | WIN qread_v16@0.125 ρ 0.13 | NO_POINT | WIN router_nest2_calib@3 ρ 0.69 | NO_POINT | WIN router_nest3_calib@3 ρ 0.69 | NO_POINT | NO_POINT | NO_POINT |
| V4 | WIN qreadp_v4@0.125 ρ 0.22 | NO_POINT | NO_POINT | NO_POINT | WIN router_nest3_calib+v4@3 ρ 0.79 | NO_POINT | NO_POINT | NO_POINT |

| arm@B | lens | bytes | ρ | ρ mem | dA vs FP | dA vs D [90%] | tail | Δtail [90%] | label | cat>3 | max dA | score | evicted | needle kept |
|---|---|---:|---:|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|
| qread_v16@0.125 | V16 | 39.8 | 0.13 | 1.00 | +0.153 [+0.125, +0.182] | +0.074 [+0.050, +0.100] | 0.000 | +0.000 [+0.000, +0.000] | MATCHED | 0 | 1.69 | 0.994 | 87.5% | 0.844 |
| qreadfp_v16@0.125 | V16 | 65.5 | 0.21 | 1.67 | +0.075 [+0.057, +0.094] | -0.005 [-0.030, +0.021] | 0.000 | +0.000 [+0.000, +0.000] | MATCHED | 0 | 0.80 | 0.991 | 87.5% | 0.850 |
| qread_v16@0.25 | V16 | 78.0 | 0.25 | 1.00 | +0.116 [+0.094, +0.139] | +0.036 [+0.022, +0.052] | 0.000 | +0.000 [+0.000, +0.000] | MATCHED | 0 | 1.22 | 0.994 | 75.0% | 0.898 |
| qreadfp_v16@0.25 | V16 | 129.5 | 0.42 | 1.67 | +0.038 [+0.028, +0.049] | -0.041 [-0.064, -0.019] | 0.000 | +0.000 [+0.000, +0.000] | MATCHED | 0 | 0.49 | 0.987 | 75.0% | 0.901 |
| router_seq2_calib@2.5 | V16 | 175.8 | 0.57 | 0.57 | +0.427 [+0.301, +0.564] | +0.348 [+0.216, +0.490] | 0.082 | +0.083 [+0.050, +0.119] | WORSE | 6 | 8.61 | 0.980 | 47.7% | 0.812 |
| router_seq3_calib@2.5 | V16 | 177.6 | 0.58 | 0.58 | +0.474 [+0.349, +0.614] | +0.394 [+0.267, +0.535] | 0.063 | +0.062 [+0.037, +0.094] | WORSE | 7 | 10.07 | 0.978 | 47.0% | 0.814 |
| router_pool_calib@3 | V16 | 205.2 | 0.67 | 0.67 | +0.327 [+0.181, +0.498] | +0.248 [+0.097, +0.425] | 0.031 | +0.031 [+0.013, +0.056] | INCONCLUSIVE | 3 | 10.87 | 0.981 | 39.7% | 0.874 |
| router_seq3_calib@3 | V16 | 206.2 | 0.67 | 0.67 | +0.188 [+0.114, +0.278] | +0.109 [+0.034, +0.199] | 0.031 | +0.031 [+0.013, +0.056] | INCONCLUSIVE | 2 | 5.45 | 0.983 | 39.4% | 0.875 |
| router_seq2_calib@3 | V16 | 206.6 | 0.67 | 0.67 | +0.166 [+0.088, +0.261] | +0.087 [+0.000, +0.192] | 0.013 | +0.013 [+0.000, +0.025] | INCONCLUSIVE | 2 | 8.27 | 0.987 | 39.2% | 0.876 |
| router_nest2_calib@3 | V16 | 213.0 | 0.69 | 0.69 | +0.070 [+0.041, +0.101] | -0.010 [-0.040, +0.022] | 0.000 | +0.000 [+0.000, +0.000] | MATCHED | 0 | 1.51 | 0.991 | 36.7% | 0.883 |
| router_nest3_calib@3 | V16 | 213.4 | 0.69 | 0.69 | +0.059 [+0.031, +0.090] | -0.020 [-0.048, +0.008] | 0.000 | +0.000 [+0.000, +0.000] | MATCHED | 0 | 1.56 | 0.992 | 36.6% | 0.882 |
| uniform@3 | V16 | 307.4 | — | — | +0.079 [+0.058, +0.102] | (D) | 0.000 | — | — | — | 1.38 | 0.992 | 0.0% | 1.000 |
| uniform@4 | V16 | 323.4 | 1.05 | 1.05 | +0.025 [+0.007, +0.043] | -0.055 [-0.081, -0.030] | 0.000 | +0.000 [+0.000, +0.000] | MATCHED | 0 | 0.89 | 0.991 | 0.0% | 1.000 |
| qread_v4@0.125 | V4 | 16.0 | 0.14 | 1.00 | +0.169 [+0.141, +0.199] | +0.083 [+0.059, +0.110] | 0.006 | +0.006 [+0.000, +0.019] | INCONCLUSIVE | 0 | 2.01 | 0.992 | 87.5% | 0.844 |
| qreadp_v4@0.125 | V4 | 25.5 | 0.22 | 1.00 | +0.119 [+0.093, +0.145] | +0.033 [+0.011, +0.054] | 0.000 | +0.000 [+0.000, +0.000] | MATCHED | 0 | 1.21 | 0.991 | 79.3% | 0.849 |
| router_seq3_calib+v4@2.5 | V4 | 76.9 | 0.65 | 0.65 | +0.477 [+0.353, +0.617] | +0.391 [+0.266, +0.530] | 0.069 | +0.069 [+0.037, +0.106] | WORSE | 6 | 9.88 | 0.980 | 47.0% | 0.814 |
| router_seq2_calib+v4@3 | V4 | 91.1 | 0.78 | 0.78 | +0.171 [+0.091, +0.267] | +0.085 [-0.002, +0.190] | 0.013 | +0.013 [+0.000, +0.025] | INCONCLUSIVE | 2 | 8.28 | 0.980 | 39.2% | 0.876 |
| router_nest3_calib+v4@3 | V4 | 92.9 | 0.79 | 0.79 | +0.064 [+0.036, +0.094] | -0.022 [-0.050, +0.006] | 0.000 | +0.000 [+0.000, +0.000] | MATCHED | 0 | 1.31 | 0.992 | 36.6% | 0.882 |
| uniform+v4@3 | V4 | 117.4 | — | — | +0.086 [+0.065, +0.108] | (D) | 0.000 | — | — | — | 1.12 | 0.992 | 0.0% | 1.000 |
| fp+v4@0 | V4 | 323.4 | 2.75 | 2.75 | +0.007 [+0.003, +0.010] | -0.079 [-0.101, -0.059] | 0.000 | +0.000 [+0.000, +0.000] | MATCHED | 0 | 0.19 | 0.986 | 0.0% | 1.000 |

**E1** r=0.125: G_q -0.074 [-0.100, -0.050], G_fp -0.075 [-0.094, -0.057], Q +0.001 [-0.019, +0.021] → **NO_GAIN**; qreadfp vs FP MATCHED
**E1** r=0.25: G_q -0.036 [-0.052, -0.022], G_fp -0.038 [-0.049, -0.028], Q +0.002 [-0.009, +0.014] → **NO_GAIN**; qreadfp vs FP MATCHED
**E2** B_t=3: catastrophes {'seq2': 2, 'nest2': 0, 'seq3': 2, 'nest3': 0}; labels {'seq2': 'INCONCLUSIVE', 'nest2': 'MATCHED', 'seq3': 'INCONCLUSIVE', 'nest3': 'MATCHED'}; ρ {'seq2': 0.672, 'nest2': 0.693, 'seq3': 0.671, 'nest3': 0.694}; nesting -0.113 [-0.194, -0.043] → NEST_HELPS; calibration +0.006 [-0.043, +0.055] → CAL_NO_EFFECT; nest3 − seq2 -0.107 [-0.208, -0.024] → **TAIL_FIXED**
**Regression** (dA − dA_D per Stage 1d catastrophic prompt-task): fp+v4@0 {'7112/niah_multikey': -0.006, '7113/niah_multikey': -0.051, '7117/niah_multikey': -0.012}; qread_v16@0.125 {'7112/niah_multikey': -0.0, '7113/niah_multikey': 0.001, '7117/niah_multikey': 0.009}; qread_v16@0.25 {'7112/niah_multikey': -0.0, '7113/niah_multikey': -0.005, '7117/niah_multikey': 0.002}; qread_v4@0.125 {'7112/niah_multikey': 0.005, '7113/niah_multikey': -0.009, '7117/niah_multikey': 0.011}; qreadfp_v16@0.125 {'7112/niah_multikey': -0.008, '7113/niah_multikey': -0.041, '7117/niah_multikey': -0.008}; qreadfp_v16@0.25 {'7112/niah_multikey': -0.008, '7113/niah_multikey': -0.043, '7117/niah_multikey': -0.011}; qreadp_v4@0.125 {'7112/niah_multikey': 0.002, '7113/niah_multikey': -0.014, '7117/niah_multikey': 0.001}; router_nest2_calib@3 {'7112/niah_multikey': -0.003, '7113/niah_multikey': 0.406, '7117/niah_multikey': 0.408}; router_nest3_calib@3 {'7112/niah_multikey': -0.01, '7113/niah_multikey': 0.06, '7117/niah_multikey': -0.004}; router_nest3_calib+v4@3 {'7112/niah_multikey': -0.008, '7113/niah_multikey': 0.057, '7117/niah_multikey': 0.001}; router_pool_calib@3 {'7112/niah_multikey': 4.369, '7113/niah_multikey': 0.602, '7117/niah_multikey': 1.329}; router_seq2_calib@2.5 {'7112/niah_multikey': 0.074, '7113/niah_multikey': 1.536, '7117/niah_multikey': 0.79}; router_seq2_calib@3 {'7112/niah_multikey': 3.793, '7113/niah_multikey': 0.579, '7117/niah_multikey': 0.264}; router_seq2_calib+v4@3 {'7112/niah_multikey': 4.235, '7113/niah_multikey': 0.588, '7117/niah_multikey': 0.335}; router_seq3_calib@2.5 {'7112/niah_multikey': 0.358, '7113/niah_multikey': 0.458, '7117/niah_multikey': 0.13}; router_seq3_calib@3 {'7112/niah_multikey': 0.816, '7113/niah_multikey': 0.77, '7117/niah_multikey': 0.58}; router_seq3_calib+v4@2.5 {'7112/niah_multikey': 0.401, '7113/niah_multikey': 0.517, '7117/niah_multikey': 0.233}; uniform@4 {'7112/niah_multikey': -0.004, '7113/niah_multikey': -0.053, '7117/niah_multikey': -0.01}
Calibration: {'2.5': {'prompt_tasks': 70, 'fail': 29, 'fail_by_task': {'niah_multikey': 17, 'niah_multivalue': 6, 'niah_single': 4, 'vt': 2}, 'dense_R0': 5, 'dense_seq3': 31, 'dense_nest3': 31}, '3': {'prompt_tasks': 70, 'fail': 4, 'fail_by_task': {'niah_multikey': 3, 'niah_multivalue': 1, 'niah_single': 0, 'vt': 0}, 'dense_R0': 33, 'dense_seq3': 35, 'dense_nest3': 51}}

