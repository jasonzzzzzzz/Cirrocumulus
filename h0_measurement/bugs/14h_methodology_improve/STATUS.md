# Stage 1h status (rewritten, not appended)

**Phase:** M0, methodology code. The run-time code is done and tested. No GPU job
submitted. Updated 2026-10-04.

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

## Before R1 is submitted
- [ ] `read_stage1h.py`: R1's frozen rules (`plan.md` §5) in its docstring; the gate
  (`--pilot`, including the NOISE_DEGENERATE check); the read (`--r1`, `--r1-seeds`);
  synthetic-block tests added to `test_r14_stage1h.py`.

## Runs
| run | jobs | state |
|---|---|---|
| R1 calibration and bridge | — | not submitted |
| R2 Qwen | — | outline only (`plan.md` §6) |
| R3–R5 | — | — |

## Notes
- On CPU (float32), `fp_noise` was bit-identical to FP. On the GPU it should differ:
  a smaller chunk moves rows from the causal flash kernel to the masked SDPA kernel.
  The gate checks this.
