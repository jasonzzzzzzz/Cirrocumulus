# Stage 1h status (rewritten, not appended)

**Phase:** R1 running on Trillium; R2's code is ready and tested (not submitted). Updated 2026-10-04.

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
`submit_s1h.slurm`. R2's code goes in new files.

## Runs
| run | jobs | state |
|---|---|---|
| R1 calibration and bridge | Trillium: pilot 1032158, gate 1032159, h1cal 1032160 (9100–9109) and 1032161 (9110–9119), seeds 1032162–1032164 (rot_seed 0, 1, 2), reader 1032165 (writes `findings/R1_reader.{json,md}`) | submitted 2026-10-04 21:05; **gate PASS** (noise measured: fp_noise KL ~0.005 on the pilot; peak 48 GiB; projected block 1.7 h); seed 0 done; blocks queued |
| R2 Qwen 32K | — | code written and tested; rules frozen (`read_stage1h_r2.py`; `plan.md` Amendments); submit with `script_stage1h.sh --run-r2 1032165` once R1's gate passes |
| R3–R5 | — | — |

## Notes
- On CPU (float32), `fp_noise` was bit-identical to FP. On the GPU it should differ:
  a smaller chunk moves rows from the causal flash kernel to the masked SDPA kernel.
  The gate checks this.
