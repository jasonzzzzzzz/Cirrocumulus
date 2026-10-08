# R14 Stage 1h — R3b ladder (read_stage1h_r3b.py; rules frozen in its docstring)

Excluded from every result: it only chooses each task's difficulty, and which tasks run, for the main cells.

## llama31-8b → h3bllama: RUN, tasks cwe,fwe, `--task-cfg freq_cw=100,alpha=2`

| task | FP score by level | D_V4 − FP by level | chosen | label | enters |
|---|---|---|---:|---|---|
| cwe | {1: 0.0, 2: 0.712, 3: 1.0} | {1: 0.0, 2: 0.062, 3: 0.0} | 2 | HEADROOM | True |
| fwe | {1: 0.708, 2: 0.708, 3: 0.667} | {1: -0.083, 2: 0.0, 3: 0.0} | 1 | HEADROOM | True |
| nolima | {1: 0.062} | {1: -0.062} | — | TOO_HARD | False |
| nolima_direct | {1: 0.875} | {1: -0.062} | — | (with nolima) | False |

## qwen3-30b-a3b-2507 → h3bqwen: RUN, tasks cwe,fwe, `--task-cfg freq_cw=30,alpha=2`

| task | FP score by level | D_V4 − FP by level | chosen | label | enters |
|---|---|---|---:|---|---|
| cwe | {1: 0.825, 2: 0.963, 3: 1.0} | {1: 0.025, 2: 0.0, 3: -0.012} | 1 | HEADROOM | True |
| fwe | {1: 0.917, 2: 0.833, 3: 0.875} | {1: 0.0, 2: -0.042, 3: -0.083} | 1 | HEADROOM | True |
| nolima | {1: 0.375} | {1: -0.125} | — | TOO_HARD | False |
| nolima_direct | {1: 1.0} | {1: 0.0} | — | (with nolima) | False |

