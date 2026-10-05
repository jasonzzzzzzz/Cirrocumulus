# Stage 1h status (rewritten, not appended)

**Phase:** R1, R2 and R3a's ladder queued on Trillium; R3b's code is ready and tested (not
submitted). Updated 2026-10-05.

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

**Don't edit files that queued or running jobs import:**
- R1: `s1h_lib.py`, `run_s1h.py`, `metrics_s1h.py`, `read_stage1h.py`, `test_r14_stage1h.py`,
  `submit_s1h.slurm`;
- R2: `s1h2_lib.py`, `run_s1h2.py`, `read_stage1h_r2.py`, `test_r14_stage1h_r2.py`,
  `submit_s1h2.slurm`;
- R3a: `s1h3_lib.py`, `run_s1h3.py`, `read_stage1h_r3.py`, `test_r14_stage1h_r3.py`,
  `submit_s1h3.slurm`;
- once R3b is submitted: `tasks_s1h.py`, `data/r3b/`, `s1h3b_lib.py`, `run_s1h3b.py`,
  `read_stage1h_r3b.py`, `test_r14_stage1h_r3b.py`, `submit_s1h3b.slurm`.

New work goes in new files.

## Runs
| run | jobs | state |
|---|---|---|
| R1 calibration and bridge | Trillium: pilot 1032158, gate 1032159, h1cal 1032160 (9100–9109) and 1032161 (9110–9119), seeds 1032162–1032164 (rot_seed 0, 1, 2), reader 1032165 (writes `findings/R1_reader.{json,md}`) | submitted 2026-10-04 21:05; **gate PASS** (noise measured: fp_noise KL ~0.005 on the pilot; peak 48 GiB; projected block 1.7 h); seeds done; h1cal blocks queued |
| R2 Qwen 32K | Trillium: pilot 1032364, gate 1032365, h2qwen32 1032366 (9300–9309) and 1032367 (9310–9319), seeds 1032368–1032370, reader 1032371 (after R1's reader; writes `findings/R2_reader.{json,md}`) | submitted 2026-10-04; gate done; blocks queued |
| R3a harder synthetic tasks (both models) | step 1 ladder, Trillium: Llama 1032502 / 1032504 / 1032506, Qwen 1032503 / 1032505 / 1032507 (levels 1–3, prompts 3200–3204, FP / D / D_V4), ladder reader 1032508 (writes `findings/R3a_levels.{json,md}`); step 2: `--run-r3a 1032165` after the ladder is read (h3llama 9400–9419, h3qwen 9420–9439) | ladder submitted 2026-10-04 ~22:05 |
| R3b aggregation and latent association (both models) | step 1: `script_stage1h.sh --run-r3b-ladder` (8 jobs: per model, `cwe` + `fwe` at levels 1–3 on 3210–3217, and `nolima` + `nolima_direct` on 3210–3225; FP / D / D_V4; then the ladder reader → `findings/R3b_levels.{json,md}`); step 2: `--run-r3b 1032165` (h3bllama 9440–9459, h3bqwen 9460–9479, the tasks that entered) | code ready: fast tests, the CPU driver smokes (ladder on all four tasks; 16 arms on `cwe` + `nolima` at 4K) and the chain logic (dry, mock ladder output) pass; not submitted |
| R4, R5 | — | — |

## Notes
- On CPU (float32), `fp_noise` was bit-identical to FP. On the GPU it should differ:
  a smaller chunk moves rows from the causal flash kernel to the masked SDPA kernel.
  The gate checks this.
- R3b needs no result from R1–R3a to be submitted. Its main cells wait only for its own
  ladder, and its reader for R1's (m_FP).
- R3b's NoLiMa data is under the Adobe Research License (non-commercial research);
  `data/r3b/nolima_LICENSE`.
- In the 1B CPU smoke at 4K, FP answered `nolima` with a character from the PG-19
  haystack ("Heathcliff") and got `nolima_direct` right: the failure the task is meant to
  show. NoLiMa's own table puts Llama-3.1-8B at 0.14 by 32K, so `nolima` may not enter
  Llama's 128K cell.
