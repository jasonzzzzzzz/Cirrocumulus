# Stage 1h status (rewritten, not appended)

**Phase:** R1 and R2 queued on Trillium; R3a's code is ready and tested (ladder not submitted). Updated 2026-10-04.

## M0 checklist (before R1 is submitted)
- [x] Docs: `CLAUDE.md`, `STATUS.md`, `glossary.md`, `plan.md`
- [x] `s1h_lib.py`: R1 presets (`h1cal`, `h1pilot`, `h1regress`, `h1smoke`), arm
  grammar, plan (17 arms), R1 labels
- [x] `metrics_s1h.py`: KL, lost answers, McNemar, equivalence, Holm, sequential rule,
  failure types
- [x] `run_s1h.py`, the run-time items:
  - KL columns;
  - `fp_noise`;
  - `fp8kv`;
  - `kivi4_v4` and `kvquant4_v4`;
  - `qoraclefp_v16` and `qoracle4_v4`;
  - per-arm peak memory;
  - `qread2t4q_v4` in the preset;
  - seeds through `--override rot_seed=N`.
- [x] `test_r14_stage1h.py`: all checks pass, both `--fast` and the default:
  - with the wrappers installed, Stage 1g's arms give identical tokens and log-probs;
  - every new view matches its quantizer and replays greedily;
  - the oracle keeps floor(r C) rows per KV head;
  - the noise arm works;
  - the driver smoke runs all 17 arms on 2 tasks.
- [x] `submit_s1h.slurm`, `script_stage1h.sh`: the R1 chain (pilot → gate → 2 blocks +
  3 seed regression jobs → reader). The preflight refuses until `read_stage1h.py`
  exists.

- [x] `read_stage1h.py`: R1's rules frozen in its docstring (2026-10-04, before any
  output), the gate (with the NOISE_DEGENERATE check), the R1 read; synthetic-block
  tests pass; checked on real CPU driver output.

**Don't edit files R1's jobs import while they are pending or running:** `s1h_lib.py`,
`run_s1h.py`, `metrics_s1h.py`, `read_stage1h.py`, `test_r14_stage1h.py`,
`submit_s1h.slurm`, and R2's `s1h2_lib.py`, `run_s1h2.py`, `read_stage1h_r2.py`, `test_r14_stage1h_r2.py`, `submit_s1h2.slurm`.
Once R3a is submitted, also `s1h3_lib.py`, `run_s1h3.py`, `read_stage1h_r3.py`, `test_r14_stage1h_r3.py`,
`submit_s1h3.slurm`. New work goes in new files.

## Runs
| run | jobs | state |
|---|---|---|
| R1 calibration and bridge | Trillium: pilot 1032158, gate 1032159, h1cal 1032160 (9100–9109) and 1032161 (9110–9119), seeds 1032162–1032164 (rot_seed 0, 1, 2), reader 1032165 (writes `findings/R1_reader.{json,md}`) | submitted 2026-10-04 21:05; **gate PASS** (noise measured: fp_noise KL ~0.005 on the pilot; peak 48 GiB; projected block 1.7 h); seed 0 done; blocks queued |
| R2 Qwen 32K | Trillium: pilot 1032364, gate 1032365, h2qwen32 1032366 (9300–9309) and 1032367 (9310–9319), seeds 1032368–1032370, reader 1032371 (after R1's reader; writes `findings/R2_reader.{json,md}`) | submitted 2026-10-04 |
| R3a harder synthetic tasks (both models) | step 1 ladder: 6 jobs (3 levels × Llama 128K / Qwen 32K, prompts 3200–3204, FP / D / D_V4) + ladder reader (writes `findings/R3a_levels.{json,md}`); step 2: `--run-r3a <R1 reader job>` (h3llama 9400–9419, h3qwen 9420–9439) | code ready: fast tests, the CPU driver smokes (16K ladder with mk_panel; all 16 main arms at 4K) and the trig-login01 preflight (`--run-r3-ladder-dry`) pass; ladder not submitted |
| R3b, R4, R5 | — | — |

## Notes
- On CPU (float32), `fp_noise` was bit-identical to FP. On the GPU it should differ:
  a smaller chunk moves rows from the causal flash kernel to the masked SDPA kernel.
  The gate checks this.
