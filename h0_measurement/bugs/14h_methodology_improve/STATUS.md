# Stage 1h status (rewritten, not appended)

**Phase:** R1 and R2 queued on Trillium. R3a's first ladder is void (amendment R3a2); R3a2, R3b and
R4b are coded and tested, none submitted. Updated 2026-10-05.

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
  `read_stage1h_r3b.py`, `test_r14_stage1h_r3b.py`, `submit_s1h3b.slurm`;
- shared by R3a2, R3b and R4b: `stops_s1h.py`;
- once R4 is submitted: `tasks_s1h4.py`, `make_r4_manifest.py`, `data/r4/` and `.h0_corpus/helmet/r4_items/` (pinned by sha256),
  `s1h4_lib.py`, `run_s1h4.py`, `read_stage1h_r4.py`, `test_r14_stage1h_r4.py`, `submit_s1h4.slurm`
  (R4's jobs also import `tasks_s1h.py`, R3b's).

New work goes in new files.

## Runs
| run | jobs | state |
|---|---|---|
| R1 calibration and bridge | Trillium: pilot 1032158, gate 1032159, h1cal 1032160 (9100–9109) and 1032161 (9110–9119), seeds 1032162–1032164 (rot_seed 0, 1, 2), reader 1032165 (writes `findings/R1_reader.{json,md}`) | submitted 2026-10-04 21:05; **gate PASS**; seeds done; h1cal blocks queued |
| R2 Qwen 32K | Trillium: pilot 1032364, gate 1032365, h2qwen32 1032366 (9300–9309) and 1032367 (9310–9319), seeds 1032368–1032370, reader 1032371 (after R1's reader; writes `findings/R2_reader.{json,md}`) | submitted 2026-10-04; gate done; blocks queued |
| R3a harder synthetic tasks (both models) | first ladder 1032502–1032508 **void** (amendment R3a2: 1032504 cancelled by the system so the reader failed; Qwen's multivalue answers cut by the cap; Llama stopped at `:\n\n`). Next: `script_stage1h.sh --run-r3-ladder` again (R3a2: stop rule `r8list` for both models, per-unit caps), then `--run-r3a` | R3a2 code ready: fast tests and the CPU driver smokes pass (the 1B model's multivalue answer now scores 1.0 where `:\n\n` had cut it); not resubmitted |
| R3b aggregation and latent association (both models) | `--run-r3b-ladder`, then `--run-r3b` | fixed 2026-10-05 (`r8list` for Qwen too, the cell's own margin, the OTHER_NAME report); fast tests and smokes pass; not submitted |
| R4b real tasks at 128K (both models) | `script_stage1h.sh --run-r4`: per cell a pilot → gate → 2 blocks; cells: LongBench v2 at Llama 128K (1 GPU) and Qwen 128K (2 GPUs), 40 items each; HELMET (kilt_nq, kilt_hotpotqa, msmarco_rerank_psg, icl_trec_coarse, icl_banking77) at Llama and Qwen 128K, 10 items × 5 tasks; then the reader → `findings/R4_reader.{json,md}` | rebuilt 2026-10-05 (replaces the LongBench v1 design, which never ran): manifest `data/r4/manifest_r4.json` pinned; fast tests, the CPU driver smokes (every arm on lbv2, kilt_nq and icl_trec_coarse at 4K) and the trig-login01 preflight + `--run-r4-dry` pass; not submitted. About 18 GPU-h (Qwen's 2-GPU cells about 10) |
| R5 | — | — |

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
- R4 depends on no result of R1–R3b either. It reuses bug 9's LongBench v2 partitions that were
  already exposed (V4–V6 qualification + development; V7, a closed study, qualification + development)
  and leaves both confirmation partitions (45 + 45 items) untouched for the confirmatory campaign.
- The 2026-10-05 review's fixes (plan.md, amendments R3a2 and R4b): the `r8list` stop rule for both
  models; per-unit caps (R3a); every cell's own FP8 margin; on LongBench v2 the vote observes the
  question and its choices; a closed-book arm and the context-dependent stratum; a pilot and gate
  per R4 cell; labels per task family; one bootstrap cluster per unit; a split that tokenizes as
  the whole prompt; the gold choice's log-probability reported.
- `.h0_corpus/longbench_v1/` (114 MB) is no longer used by any run.
