# R14 Stage 1h — R3a ladder (read_stage1h_r3.py; rules frozen in its docstring)

Excluded from every result: it only chooses each task's difficulty for the main cells.

## llama31-8b → h3llama with `--task-cfg n_keys=64,n_values=16,n_hops=12`

| task | FP score by level | D_V4 − FP by level | chosen | label |
|---|---|---|---:|---|
| niah_multikey | {1: 1.0, 2: 1.0, 3: 1.0} | {1: 0.0, 2: 0.0, 3: 0.0} | 3 | CEILING_REMAINS |
| niah_multivalue | {1: 1.0, 2: 0.825, 3: 0.8666666666666668} | {1: -0.075, 2: 0.087, 3: -0.35} | 2 | HEADROOM |
| vt | {1: 0.9777777777777779, 2: 0.9384615384615385, 3: 0.9058823529411765} | {1: -0.067, 2: -0.092, 3: -0.129} | 2 | HEADROOM |
| mk_panel | 1.000 | +0.000 | — | no headroom |

## qwen3-30b-a3b-2507 → h3qwen with `--task-cfg n_keys=64,n_values=24,n_hops=16`

| task | FP score by level | D_V4 − FP by level | chosen | label |
|---|---|---|---:|---|
| niah_multikey | {1: 1.0, 2: 1.0, 3: 1.0} | {1: 0.0, 2: 0.0, 3: 0.0} | 3 | CEILING_REMAINS |
| niah_multivalue | {1: 1.0, 2: 1.0, 3: 0.9583333333333333} | {1: -0.025, 2: -0.075, 3: -0.025} | 3 | CEILING_REMAINS |
| vt | {1: 1.0, 2: 1.0, 3: 0.9764705882352942} | {1: 0.0, 2: 0.0, 3: 0.012} | 3 | CEILING_REMAINS |
| mk_panel | 1.000 | +0.000 | — | no headroom |

