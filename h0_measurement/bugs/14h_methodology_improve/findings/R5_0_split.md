# R5.0 — where the system's loss comes from (R1–R4, exploratory)

Change against FP per unit, averaged. dP = span NLL (nats), /tok = per span token; split at r = 1/8: budget = oracle − FP, vote = vote(exact) − oracle, store = system − vote(exact).

## R1 Llama 128K RULER (jobs 1032160, 1032161)

| task | n | FP acc | span tok | budget | vote | store | budget/tok | vote/tok | store/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| niah_multikey | 20 | 1.000 | 3 | -0.00 | +0.00 | +0.00 | -0.000 | +0.001 | +0.001 |
| niah_multivalue | 20 | 0.963 | 17 | +0.03 | +0.06 | +0.01 | +0.002 | +0.003 | +0.001 |
| niah_single | 20 | 1.000 | 3 | -0.00 | +0.00 | +0.01 | -0.000 | +0.002 | +0.002 |
| vt | 20 | 0.990 | 25 | -0.05 | +0.00 | -0.00 | -0.002 | -0.000 | -0.000 |

| task | arm | dP | dP/tok | KL | KL/tok | dAcc |
|---|---|---:|---:|---:|---:|---:|
| niah_multikey | FP8 KV | -0.00 | -0.001 | +0.00 | +0.001 | +0.000 |
| niah_multikey | noise | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | dense KIVI-4 | +0.00 | +0.000 | +0.00 | +0.001 | +0.000 |
| niah_multikey | dense KVQuant-4 | +0.02 | +0.006 | +0.01 | +0.004 | +0.000 |
| niah_multikey | dense uniform-4 | +0.01 | +0.003 | +0.01 | +0.002 | +0.000 |
| niah_multikey | oracle 1/8 (exact) | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | oracle 1/8 (4-bit) | +0.01 | +0.003 | +0.01 | +0.002 | +0.000 |
| niah_multikey | vote 1/8 (exact) | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | vote 1/8 (4-bit) | +0.01 | +0.004 | +0.01 | +0.002 | +0.000 |
| niah_multikey | system 1/8 | +0.00 | +0.001 | +0.00 | +0.000 | +0.000 |
| niah_multivalue | FP8 KV | -0.00 | -0.001 | +0.05 | +0.003 | +0.000 |
| niah_multivalue | noise | +0.04 | +0.003 | +0.01 | +0.001 | +0.000 |
| niah_multivalue | dense KIVI-4 | +0.15 | +0.009 | +0.12 | +0.007 | +0.000 |
| niah_multivalue | dense KVQuant-4 | +0.31 | +0.018 | +0.13 | +0.008 | -0.013 |
| niah_multivalue | dense uniform-4 | +0.15 | +0.008 | +0.19 | +0.011 | +0.025 |
| niah_multivalue | oracle 1/8 (exact) | +0.03 | +0.002 | +0.04 | +0.002 | +0.000 |
| niah_multivalue | oracle 1/8 (4-bit) | +0.15 | +0.009 | +0.18 | +0.010 | +0.013 |
| niah_multivalue | vote 1/8 (exact) | +0.09 | +0.005 | +0.10 | +0.005 | +0.013 |
| niah_multivalue | vote 1/8 (4-bit) | +0.15 | +0.009 | +0.23 | +0.013 | +0.013 |
| niah_multivalue | system 1/8 | +0.10 | +0.007 | +0.12 | +0.007 | +0.025 |
| niah_single | FP8 KV | +0.00 | +0.001 | +0.00 | +0.000 | +0.000 |
| niah_single | noise | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_single | dense KIVI-4 | +0.01 | +0.002 | +0.00 | +0.001 | +0.000 |
| niah_single | dense KVQuant-4 | +0.01 | +0.003 | +0.00 | +0.001 | +0.000 |
| niah_single | dense uniform-4 | +0.01 | +0.003 | +0.00 | +0.001 | +0.000 |
| niah_single | oracle 1/8 (exact) | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_single | oracle 1/8 (4-bit) | +0.01 | +0.002 | +0.00 | +0.001 | +0.000 |
| niah_single | vote 1/8 (exact) | +0.00 | +0.001 | +0.00 | +0.001 | +0.000 |
| niah_single | vote 1/8 (4-bit) | +0.01 | +0.005 | +0.01 | +0.002 | +0.000 |
| niah_single | system 1/8 | +0.01 | +0.003 | +0.00 | +0.001 | +0.000 |
| vt | FP8 KV | +0.00 | +0.000 | +0.05 | +0.002 | -0.030 |
| vt | noise | +0.01 | +0.000 | +0.01 | +0.000 | +0.000 |
| vt | dense KIVI-4 | +0.13 | +0.005 | +0.15 | +0.006 | +0.000 |
| vt | dense KVQuant-4 | +0.15 | +0.007 | +0.16 | +0.006 | -0.040 |
| vt | dense uniform-4 | +0.26 | +0.011 | +0.18 | +0.007 | -0.040 |
| vt | oracle 1/8 (exact) | -0.05 | -0.002 | +0.04 | +0.002 | +0.000 |
| vt | oracle 1/8 (4-bit) | +0.25 | +0.010 | +0.21 | +0.008 | -0.040 |
| vt | vote 1/8 (exact) | -0.05 | -0.002 | +0.15 | +0.006 | -0.020 |
| vt | vote 1/8 (4-bit) | +0.20 | +0.008 | +0.31 | +0.012 | -0.040 |
| vt | system 1/8 | -0.05 | -0.002 | +0.15 | +0.006 | +0.000 |

## R2 Qwen 32K RULER (jobs 1032366)

| task | n | FP acc | span tok | budget | vote | store | budget/tok | vote/tok | store/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| niah_multikey | 10 | 1.000 | 7 | -0.00 | -0.00 | +0.00 | -0.000 | -0.000 | +0.000 |
| niah_multivalue | 10 | 1.000 | 40 | +0.01 | +0.01 | +0.01 | +0.000 | +0.000 | +0.000 |
| niah_single | 10 | 1.000 | 7 | +0.00 | +0.00 | +0.00 | +0.000 | +0.000 | +0.000 |
| vt | 10 | 1.000 | 23 | -0.09 | +0.00 | +0.02 | -0.004 | +0.000 | +0.001 |

| task | arm | dP | dP/tok | KL | KL/tok | dAcc |
|---|---|---:|---:|---:|---:|---:|
| niah_multikey | FP8 KV | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | noise | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | dense KIVI-4 | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | dense KVQuant-4 | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | dense uniform-4 | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | oracle 1/8 (exact) | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | oracle 1/8 (4-bit) | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | vote 1/8 (exact) | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | vote 1/8 (4-bit) | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | system 1/8 | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | system floor r=1/2 | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_multivalue | FP8 KV | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_multivalue | noise | -0.02 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_multivalue | dense KIVI-4 | -0.00 | -0.000 | +0.01 | +0.000 | +0.000 |
| niah_multivalue | dense KVQuant-4 | -0.01 | -0.000 | +0.01 | +0.000 | +0.000 |
| niah_multivalue | dense uniform-4 | -0.04 | -0.001 | +0.02 | +0.000 | +0.000 |
| niah_multivalue | oracle 1/8 (exact) | +0.01 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_multivalue | oracle 1/8 (4-bit) | -0.01 | -0.000 | +0.01 | +0.000 | +0.000 |
| niah_multivalue | vote 1/8 (exact) | +0.02 | +0.001 | +0.01 | +0.000 | +0.000 |
| niah_multivalue | vote 1/8 (4-bit) | +0.06 | +0.002 | +0.08 | +0.002 | +0.000 |
| niah_multivalue | system 1/8 | +0.03 | +0.001 | +0.01 | +0.000 | +0.000 |
| niah_multivalue | system floor r=1/2 | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_single | FP8 KV | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_single | noise | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_single | dense KIVI-4 | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_single | dense KVQuant-4 | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_single | dense uniform-4 | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_single | oracle 1/8 (exact) | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_single | oracle 1/8 (4-bit) | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_single | vote 1/8 (exact) | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_single | vote 1/8 (4-bit) | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_single | system 1/8 | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_single | system floor r=1/2 | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| vt | FP8 KV | -0.02 | -0.001 | +0.02 | +0.001 | +0.000 |
| vt | noise | -0.00 | -0.000 | +0.01 | +0.000 | +0.000 |
| vt | dense KIVI-4 | +0.03 | +0.001 | +0.04 | +0.002 | +0.000 |
| vt | dense KVQuant-4 | +0.00 | +0.000 | +0.03 | +0.001 | +0.000 |
| vt | dense uniform-4 | +0.14 | +0.006 | +0.05 | +0.002 | +0.000 |
| vt | oracle 1/8 (exact) | -0.09 | -0.004 | +0.03 | +0.001 | +0.000 |
| vt | oracle 1/8 (4-bit) | +0.07 | +0.003 | +0.08 | +0.004 | +0.000 |
| vt | vote 1/8 (exact) | -0.08 | -0.003 | +0.06 | +0.003 | +0.000 |
| vt | vote 1/8 (4-bit) | +0.12 | +0.005 | +0.15 | +0.006 | +0.000 |
| vt | system 1/8 | -0.06 | -0.003 | +0.11 | +0.004 | +0.000 |
| vt | system floor r=1/2 | -0.02 | -0.001 | +0.02 | +0.001 | +0.000 |

## R3a Llama 128K harder RULER (jobs 22594818, 22594819)

| task | n | FP acc | span tok | budget | vote | store | budget/tok | vote/tok | store/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mk_panel | 20 | 1.000 | 3 | -0.00 | +0.00 | +0.01 | -0.000 | +0.001 | +0.002 |
| niah_multikey | 20 | 0.950 | 3 | -0.00 | -0.00 | -0.00 | -0.001 | -0.001 | -0.000 |
| niah_multivalue | 20 | 0.744 | 58 | +0.84 | +0.20 | +0.31 | +0.017 | +0.002 | +0.005 |
| vt | 20 | 0.796 | 54 | +0.24 | +0.13 | +0.02 | +0.004 | +0.002 | +0.001 |

| task | arm | dP | dP/tok | KL | KL/tok | dAcc |
|---|---|---:|---:|---:|---:|---:|
| mk_panel | FP8 KV | +0.01 | +0.002 | +0.00 | +0.001 | +0.000 |
| mk_panel | noise | -0.00 | -0.001 | +0.00 | +0.000 | +0.000 |
| mk_panel | dense KIVI-4 | +0.01 | +0.005 | +0.01 | +0.002 | +0.000 |
| mk_panel | dense KVQuant-4 | +0.02 | +0.005 | +0.01 | +0.003 | +0.000 |
| mk_panel | dense uniform-4 | +0.02 | +0.005 | +0.01 | +0.003 | +0.000 |
| mk_panel | oracle 1/8 (exact) | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| mk_panel | vote 1/8 (exact) | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| mk_panel | vote 1/8 (4-bit) | +0.01 | +0.004 | +0.01 | +0.002 | +0.000 |
| mk_panel | system 1/8 | +0.01 | +0.002 | +0.00 | +0.001 | +0.000 |
| niah_multikey | FP8 KV | +0.01 | +0.002 | +0.00 | +0.001 | +0.000 |
| niah_multikey | noise | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | dense KIVI-4 | +0.02 | +0.007 | +0.01 | +0.003 | +0.000 |
| niah_multikey | dense KVQuant-4 | +0.03 | +0.009 | +0.01 | +0.004 | +0.000 |
| niah_multikey | dense uniform-4 | +0.04 | +0.014 | +0.02 | +0.005 | +0.000 |
| niah_multikey | oracle 1/8 (exact) | -0.00 | -0.001 | +0.00 | +0.000 | +0.000 |
| niah_multikey | vote 1/8 (exact) | -0.00 | -0.001 | +0.00 | +0.000 | +0.000 |
| niah_multikey | vote 1/8 (4-bit) | +0.04 | +0.014 | +0.01 | +0.005 | +0.050 |
| niah_multikey | system 1/8 | -0.01 | -0.002 | +0.00 | +0.001 | +0.050 |
| niah_multivalue | FP8 KV | +0.42 | +0.007 | +0.42 | +0.006 | +0.000 |
| niah_multivalue | noise | -0.02 | +0.001 | +0.05 | +0.001 | -0.006 |
| niah_multivalue | dense KIVI-4 | +0.71 | +0.037 | +0.65 | +0.015 | -0.003 |
| niah_multivalue | dense KVQuant-4 | +1.72 | +0.039 | +1.00 | +0.021 | +0.041 |
| niah_multivalue | dense uniform-4 | +1.19 | +0.009 | +1.04 | +0.017 | -0.059 |
| niah_multivalue | oracle 1/8 (exact) | +0.84 | +0.017 | +0.82 | +0.012 | +0.016 |
| niah_multivalue | vote 1/8 (exact) | +1.04 | +0.019 | +1.49 | +0.022 | +0.069 |
| niah_multivalue | vote 1/8 (4-bit) | +1.84 | +0.024 | +2.20 | +0.034 | +0.041 |
| niah_multivalue | system 1/8 | +1.36 | +0.025 | +1.59 | +0.023 | +0.059 |
| vt | FP8 KV | +0.23 | +0.003 | +0.18 | +0.003 | -0.008 |
| vt | noise | +0.06 | +0.000 | +0.02 | +0.000 | +0.008 |
| vt | dense KIVI-4 | +0.63 | +0.015 | +0.43 | +0.008 | +0.058 |
| vt | dense KVQuant-4 | +0.63 | +0.014 | +0.57 | +0.011 | +0.054 |
| vt | dense uniform-4 | +0.46 | +0.009 | +0.49 | +0.009 | +0.008 |
| vt | oracle 1/8 (exact) | +0.24 | +0.004 | +0.20 | +0.004 | -0.023 |
| vt | vote 1/8 (exact) | +0.37 | +0.006 | +0.61 | +0.010 | +0.035 |
| vt | vote 1/8 (4-bit) | +0.79 | +0.015 | +0.98 | +0.018 | -0.015 |
| vt | system 1/8 | +0.39 | +0.007 | +0.57 | +0.010 | -0.023 |

## R3a Qwen 32K harder RULER (jobs 22594820, 22594821)

| task | n | FP acc | span tok | budget | vote | store | budget/tok | vote/tok | store/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mk_panel | 20 | 1.000 | 7 | +0.00 | +1.72 | -0.11 | +0.000 | +0.246 | -0.015 |
| niah_multikey | 20 | 1.000 | 7 | +0.00 | -0.00 | -0.00 | +0.000 | -0.000 | -0.000 |
| niah_multivalue | 20 | 0.969 | 262 | +0.44 | +69.10 | -1.22 | +0.002 | +0.266 | -0.004 |
| vt | 20 | 0.982 | 83 | +0.23 | +4.58 | +0.20 | +0.003 | +0.056 | +0.003 |

| task | arm | dP | dP/tok | KL | KL/tok | dAcc |
|---|---|---:|---:|---:|---:|---:|
| mk_panel | FP8 KV | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| mk_panel | noise | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| mk_panel | dense KIVI-4 | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| mk_panel | dense KVQuant-4 | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| mk_panel | dense uniform-4 | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| mk_panel | oracle 1/8 (exact) | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| mk_panel | vote 1/8 (exact) | +1.72 | +0.246 | +1.72 | +0.246 | -0.150 |
| mk_panel | vote 1/8 (4-bit) | +1.11 | +0.158 | +1.11 | +0.158 | -0.200 |
| mk_panel | system 1/8 | +1.61 | +0.230 | +1.61 | +0.230 | -0.150 |
| mk_panel | system floor r=1/2 | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | FP8 KV | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | noise | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | dense KIVI-4 | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | dense KVQuant-4 | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | dense uniform-4 | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | oracle 1/8 (exact) | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | vote 1/8 (exact) | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | vote 1/8 (4-bit) | +0.00 | +0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | system 1/8 | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_multikey | system floor r=1/2 | -0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| niah_multivalue | FP8 KV | -0.22 | -0.001 | +0.45 | +0.002 | +0.000 |
| niah_multivalue | noise | -0.20 | -0.001 | +0.60 | +0.002 | +0.000 |
| niah_multivalue | dense KIVI-4 | +0.21 | +0.001 | +0.95 | +0.004 | +0.000 |
| niah_multivalue | dense KVQuant-4 | +0.08 | +0.000 | +0.88 | +0.003 | -0.004 |
| niah_multivalue | dense uniform-4 | +0.22 | +0.001 | +1.14 | +0.004 | +0.000 |
| niah_multivalue | oracle 1/8 (exact) | +0.44 | +0.002 | +2.16 | +0.008 | +0.002 |
| niah_multivalue | vote 1/8 (exact) | +69.54 | +0.268 | +70.51 | +0.272 | -0.365 |
| niah_multivalue | vote 1/8 (4-bit) | +70.71 | +0.272 | +71.89 | +0.277 | -0.365 |
| niah_multivalue | system 1/8 | +68.32 | +0.263 | +69.44 | +0.268 | -0.358 |
| niah_multivalue | system floor r=1/2 | +4.16 | +0.017 | +5.69 | +0.023 | -0.015 |
| vt | FP8 KV | +0.23 | +0.003 | +0.19 | +0.002 | +0.003 |
| vt | noise | +0.19 | +0.002 | +0.17 | +0.002 | +0.009 |
| vt | dense KIVI-4 | +0.15 | +0.002 | +0.30 | +0.004 | +0.012 |
| vt | dense KVQuant-4 | +0.65 | +0.008 | +0.52 | +0.006 | -0.006 |
| vt | dense uniform-4 | +0.31 | +0.004 | +0.36 | +0.004 | -0.006 |
| vt | oracle 1/8 (exact) | +0.23 | +0.003 | +0.36 | +0.004 | +0.003 |
| vt | vote 1/8 (exact) | +4.81 | +0.059 | +4.59 | +0.056 | -0.044 |
| vt | vote 1/8 (4-bit) | +5.61 | +0.069 | +5.49 | +0.068 | -0.056 |
| vt | system 1/8 | +5.01 | +0.062 | +5.00 | +0.062 | -0.032 |
| vt | system floor r=1/2 | +0.29 | +0.004 | +0.32 | +0.004 | +0.012 |

## R3b Llama 128K cwe/fwe (jobs 22560817, 22560818)

| task | n | FP acc | span tok | budget | vote | store | budget/tok | vote/tok | store/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cwe | 20 | 0.480 | 34 | +1.02 | +1.53 | +0.19 | +0.030 | +0.047 | +0.007 |
| fwe | 20 | 0.600 | 15 | +0.77 | +0.52 | +0.08 | +0.060 | +0.056 | +0.003 |

| task | arm | dP | dP/tok | KL | KL/tok | dAcc |
|---|---|---:|---:|---:|---:|---:|
| cwe | FP8 KV | -0.01 | +0.000 | +0.03 | +0.001 | +0.050 |
| cwe | noise | -0.04 | -0.000 | +0.01 | +0.000 | -0.010 |
| cwe | dense KIVI-4 | +0.25 | +0.009 | +0.09 | +0.003 | -0.010 |
| cwe | dense KVQuant-4 | +0.42 | +0.012 | +0.16 | +0.004 | -0.155 |
| cwe | dense uniform-4 | +0.20 | +0.005 | +0.08 | +0.002 | +0.000 |
| cwe | oracle 1/8 (exact) | +1.02 | +0.030 | +0.57 | +0.017 | +0.095 |
| cwe | vote 1/8 (exact) | +2.55 | +0.076 | +1.81 | +0.057 | +0.130 |
| cwe | vote 1/8 (4-bit) | +2.94 | +0.087 | +1.96 | +0.062 | +0.140 |
| cwe | system 1/8 | +2.74 | +0.083 | +1.87 | +0.060 | +0.110 |
| fwe | FP8 KV | +0.00 | +0.001 | +0.01 | +0.001 | +0.050 |
| fwe | noise | +0.01 | -0.001 | +0.00 | +0.000 | +0.000 |
| fwe | dense KIVI-4 | +0.00 | +0.001 | +0.02 | +0.001 | -0.017 |
| fwe | dense KVQuant-4 | -0.01 | -0.003 | +0.01 | +0.001 | +0.033 |
| fwe | dense uniform-4 | +0.09 | +0.008 | +0.01 | +0.001 | +0.050 |
| fwe | oracle 1/8 (exact) | +0.77 | +0.060 | +0.42 | +0.040 | -0.000 |
| fwe | vote 1/8 (exact) | +1.30 | +0.116 | +0.62 | +0.049 | -0.017 |
| fwe | vote 1/8 (4-bit) | +1.37 | +0.122 | +0.60 | +0.048 | -0.017 |
| fwe | system 1/8 | +1.37 | +0.119 | +0.61 | +0.048 | -0.033 |

## R3b Qwen 32K cwe/fwe (jobs 22560819, 22560820)

| task | n | FP acc | span tok | budget | vote | store | budget/tok | vote/tok | store/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cwe | 20 | 0.800 | 34 | +2.68 | +11.41 | -0.70 | +0.073 | +0.304 | -0.015 |
| fwe | 20 | 0.883 | 11 | -0.06 | +0.39 | +0.12 | -0.006 | +0.031 | +0.009 |

| task | arm | dP | dP/tok | KL | KL/tok | dAcc |
|---|---|---:|---:|---:|---:|---:|
| cwe | FP8 KV | +0.10 | +0.003 | +0.14 | +0.004 | +0.010 |
| cwe | noise | +0.22 | +0.006 | +0.38 | +0.011 | +0.030 |
| cwe | dense KIVI-4 | +0.10 | +0.003 | +0.14 | +0.004 | +0.025 |
| cwe | dense KVQuant-4 | +0.13 | +0.003 | +0.24 | +0.007 | +0.000 |
| cwe | dense uniform-4 | +0.60 | +0.018 | +0.41 | +0.012 | -0.005 |
| cwe | oracle 1/8 (exact) | +2.68 | +0.073 | +3.33 | +0.094 | +0.025 |
| cwe | vote 1/8 (exact) | +14.09 | +0.377 | +11.93 | +0.321 | -0.160 |
| cwe | vote 1/8 (4-bit) | +13.59 | +0.372 | +11.43 | +0.313 | -0.140 |
| cwe | system 1/8 | +13.39 | +0.362 | +11.48 | +0.312 | -0.125 |
| cwe | system floor r=1/2 | +0.49 | +0.013 | +0.63 | +0.017 | +0.020 |
| fwe | FP8 KV | +0.02 | +0.000 | +0.02 | +0.002 | +0.000 |
| fwe | noise | -0.03 | -0.002 | +0.01 | +0.001 | +0.000 |
| fwe | dense KIVI-4 | -0.03 | -0.003 | +0.02 | +0.002 | +0.000 |
| fwe | dense KVQuant-4 | -0.01 | -0.002 | +0.02 | +0.002 | +0.000 |
| fwe | dense uniform-4 | +0.04 | +0.003 | +0.10 | +0.009 | +0.000 |
| fwe | oracle 1/8 (exact) | -0.06 | -0.006 | +0.11 | +0.011 | +0.000 |
| fwe | vote 1/8 (exact) | +0.33 | +0.026 | +0.64 | +0.054 | -0.067 |
| fwe | vote 1/8 (4-bit) | +0.50 | +0.041 | +0.86 | +0.073 | -0.067 |
| fwe | system 1/8 | +0.45 | +0.035 | +0.78 | +0.066 | -0.067 |
| fwe | system floor r=1/2 | -0.03 | -0.004 | +0.02 | +0.001 | +0.000 |

## R4 Llama 128K LongBench v2 (jobs 1036792, 1036793)

| task | n | FP acc | span tok | budget | vote | store | budget/tok | vote/tok | store/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lbv2 | 40 | 0.425 | 1 | +0.01 | -0.01 | -0.00 | +0.006 | -0.014 | -0.005 |

| task | arm | dP | dP/tok | KL | KL/tok | dAcc |
|---|---|---:|---:|---:|---:|---:|
| lbv2 | FP8 KV | -0.00 | -0.001 | +0.00 | +0.003 | +0.000 |
| lbv2 | noise | +0.00 | +0.003 | +0.00 | +0.001 | -0.025 |
| lbv2 | dense KIVI-4 | +0.04 | +0.039 | +0.01 | +0.009 | +0.050 |
| lbv2 | dense KVQuant-4 | +0.00 | +0.003 | +0.00 | +0.004 | +0.000 |
| lbv2 | dense uniform-4 | +0.02 | +0.018 | +0.00 | +0.004 | +0.025 |
| lbv2 | oracle 1/8 (exact) | +0.01 | +0.006 | +0.00 | +0.001 | +0.000 |
| lbv2 | vote 1/8 (exact) | -0.01 | -0.008 | +0.00 | +0.002 | -0.025 |
| lbv2 | vote 1/8 (4-bit) | +0.01 | +0.013 | +0.00 | +0.005 | +0.050 |
| lbv2 | system 1/8 | -0.01 | -0.013 | +0.00 | +0.003 | -0.025 |
| lbv2 | system floor (R4) | -0.00 | -0.003 | +0.00 | +0.002 | +0.000 |
| lbv2 | Quest exact 1/8 | -0.00 | -0.001 | +0.00 | +0.004 | -0.050 |
| lbv2 | Quest 4-bit 1/8 | +0.02 | +0.022 | +0.01 | +0.006 | -0.025 |

## R4 Llama 128K HELMET (jobs 1036925, 1036926)

| task | n | FP acc | span tok | budget | vote | store | budget/tok | vote/tok | store/tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| icl_banking77 | 10 | 1.000 | 1 | +0.01 | +0.01 | -0.00 | +0.011 | +0.011 | -0.001 |
| icl_trec_coarse | 10 | 1.000 | 1 | +0.00 | -0.05 | -0.01 | +0.003 | -0.046 | -0.008 |
| kilt_hotpotqa | 10 | 0.600 | 7 | +0.11 | +0.02 | +0.08 | +0.005 | +0.010 | +0.007 |
| kilt_nq | 10 | 0.500 | 11 | +0.50 | +1.87 | -0.22 | +0.069 | +0.184 | -0.126 |
| msmarco_rerank_psg | 10 | 0.166 | 103 | +0.48 | +28.60 | -0.49 | +0.008 | +0.448 | -0.006 |

| task | arm | dP | dP/tok | KL | KL/tok | dAcc |
|---|---|---:|---:|---:|---:|---:|
| icl_banking77 | FP8 KV | -0.00 | -0.005 | +0.00 | +0.002 | +0.000 |
| icl_banking77 | noise | +0.00 | +0.003 | +0.00 | +0.000 | +0.000 |
| icl_banking77 | dense KIVI-4 | +0.00 | +0.002 | +0.00 | +0.002 | +0.000 |
| icl_banking77 | dense KVQuant-4 | +0.01 | +0.009 | +0.00 | +0.002 | +0.000 |
| icl_banking77 | dense uniform-4 | -0.01 | -0.007 | +0.00 | +0.002 | +0.000 |
| icl_banking77 | oracle 1/8 (exact) | +0.01 | +0.011 | +0.00 | +0.002 | +0.000 |
| icl_banking77 | vote 1/8 (exact) | +0.02 | +0.022 | +0.01 | +0.006 | +0.000 |
| icl_banking77 | vote 1/8 (4-bit) | +0.01 | +0.012 | +0.01 | +0.006 | +0.000 |
| icl_banking77 | system 1/8 | +0.02 | +0.021 | +0.01 | +0.006 | +0.000 |
| icl_banking77 | system floor (R4) | +0.02 | +0.018 | +0.00 | +0.005 | +0.000 |
| icl_banking77 | Quest exact 1/8 | +0.02 | +0.020 | +0.01 | +0.010 | +0.000 |
| icl_banking77 | Quest 4-bit 1/8 | -0.01 | -0.006 | +0.01 | +0.012 | +0.000 |
| icl_trec_coarse | FP8 KV | -0.02 | -0.020 | +0.00 | +0.003 | +0.000 |
| icl_trec_coarse | noise | -0.00 | -0.001 | +0.00 | +0.001 | +0.000 |
| icl_trec_coarse | dense KIVI-4 | -0.02 | -0.016 | +0.00 | +0.002 | +0.000 |
| icl_trec_coarse | dense KVQuant-4 | -0.00 | -0.004 | +0.00 | +0.002 | +0.000 |
| icl_trec_coarse | dense uniform-4 | +0.02 | +0.021 | +0.01 | +0.007 | +0.000 |
| icl_trec_coarse | oracle 1/8 (exact) | +0.00 | +0.003 | +0.01 | +0.009 | +0.000 |
| icl_trec_coarse | vote 1/8 (exact) | -0.04 | -0.043 | +0.02 | +0.018 | +0.000 |
| icl_trec_coarse | vote 1/8 (4-bit) | -0.02 | -0.020 | +0.02 | +0.023 | +0.000 |
| icl_trec_coarse | system 1/8 | -0.05 | -0.051 | +0.02 | +0.021 | +0.000 |
| icl_trec_coarse | system floor (R4) | -0.05 | -0.045 | +0.02 | +0.017 | +0.000 |
| icl_trec_coarse | Quest exact 1/8 | +0.05 | +0.048 | +0.03 | +0.028 | +0.000 |
| icl_trec_coarse | Quest 4-bit 1/8 | +0.07 | +0.066 | +0.04 | +0.043 | -0.100 |
| kilt_hotpotqa | FP8 KV | +0.03 | -0.003 | +0.01 | +0.003 | +0.100 |
| kilt_hotpotqa | noise | +0.00 | -0.000 | +0.00 | +0.000 | +0.000 |
| kilt_hotpotqa | dense KIVI-4 | +0.14 | +0.050 | +0.05 | +0.017 | -0.100 |
| kilt_hotpotqa | dense KVQuant-4 | +0.02 | -0.006 | +0.02 | +0.003 | +0.000 |
| kilt_hotpotqa | dense uniform-4 | +0.13 | +0.025 | +0.03 | +0.006 | +0.000 |
| kilt_hotpotqa | oracle 1/8 (exact) | +0.11 | +0.005 | +0.02 | +0.002 | +0.100 |
| kilt_hotpotqa | vote 1/8 (exact) | +0.14 | +0.015 | +0.08 | +0.015 | +0.000 |
| kilt_hotpotqa | vote 1/8 (4-bit) | +0.35 | +0.041 | +0.13 | +0.016 | +0.000 |
| kilt_hotpotqa | system 1/8 | +0.22 | +0.022 | +0.11 | +0.018 | +0.000 |
| kilt_hotpotqa | system floor (R4) | +0.26 | +0.031 | +0.11 | +0.017 | +0.000 |
| kilt_hotpotqa | Quest exact 1/8 | +0.18 | +0.020 | +0.07 | +0.008 | +0.100 |
| kilt_hotpotqa | Quest 4-bit 1/8 | +0.35 | +0.058 | +0.11 | +0.018 | +0.000 |
| kilt_nq | FP8 KV | +0.14 | +0.020 | +0.03 | +0.002 | +0.000 |
| kilt_nq | noise | +0.05 | +0.009 | +0.01 | +0.001 | +0.000 |
| kilt_nq | dense KIVI-4 | +0.18 | +0.012 | +0.05 | +0.007 | +0.000 |
| kilt_nq | dense KVQuant-4 | +0.27 | +0.022 | +0.07 | +0.005 | +0.000 |
| kilt_nq | dense uniform-4 | +0.13 | +0.014 | +0.04 | +0.005 | +0.000 |
| kilt_nq | oracle 1/8 (exact) | +0.50 | +0.069 | +0.10 | +0.019 | +0.000 |
| kilt_nq | vote 1/8 (exact) | +2.36 | +0.253 | +1.69 | +0.169 | -0.100 |
| kilt_nq | vote 1/8 (4-bit) | +2.25 | +0.144 | +1.59 | +0.101 | +0.000 |
| kilt_nq | system 1/8 | +2.15 | +0.127 | +1.54 | +0.092 | +0.000 |
| kilt_nq | system floor (R4) | +2.33 | +0.234 | +1.66 | +0.153 | -0.100 |
| kilt_nq | Quest exact 1/8 | +0.91 | +0.073 | +0.43 | +0.046 | -0.100 |
| kilt_nq | Quest 4-bit 1/8 | +1.16 | +0.098 | +0.50 | +0.052 | -0.100 |
| msmarco_rerank_psg | FP8 KV | +0.37 | +0.006 | +0.38 | +0.005 | +0.018 |
| msmarco_rerank_psg | noise | -0.01 | -0.002 | +0.08 | +0.001 | -0.018 |
| msmarco_rerank_psg | dense KIVI-4 | +1.27 | +0.019 | +0.80 | +0.011 | -0.128 |
| msmarco_rerank_psg | dense KVQuant-4 | +0.56 | +0.007 | +0.59 | +0.009 | -0.111 |
| msmarco_rerank_psg | dense uniform-4 | +0.23 | +0.003 | +0.88 | +0.013 | +0.003 |
| msmarco_rerank_psg | oracle 1/8 (exact) | +0.48 | +0.008 | +0.60 | +0.009 | -0.008 |
| msmarco_rerank_psg | vote 1/8 (exact) | +29.08 | +0.456 | +26.44 | +0.412 | -0.166 |
| msmarco_rerank_psg | vote 1/8 (4-bit) | +28.27 | +0.451 | +25.96 | +0.408 | -0.166 |
| msmarco_rerank_psg | system 1/8 | +28.59 | +0.450 | +26.14 | +0.407 | -0.166 |
| msmarco_rerank_psg | system floor (R4) | +25.37 | +0.395 | +23.87 | +0.366 | -0.087 |
| msmarco_rerank_psg | Quest exact 1/8 | +13.85 | +0.217 | +12.14 | +0.187 | -0.085 |
| msmarco_rerank_psg | Quest 4-bit 1/8 | +14.22 | +0.233 | +13.18 | +0.210 | -0.052 |

