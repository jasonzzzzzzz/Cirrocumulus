# Stage 1h status (rewritten, not appended)

**Phase:** R1 and R2 queued on Trillium. R3a's first ladder is void (amendment R3a2); R3a2 and R3b
are coded and tested, not submitted. R4b: lb2llama and hmllama done (Trillium); the Qwen cells and the reader were cancelled; `script_stage1h_temp.sh` submits only what is missing. Updated 2026-10-08: findings for R1, R2 (re-read on block 1032366 + seeds; sequential PASS at 10 prompts), R3a and R3b are all in findings/ (R3a/R3b copied from Rorqual; the current readers reproduce all 25 labels of each). Open: R3b's Qwen ladder ran with stop rule eos_only (accept with a note, or re-run); R4's Qwen blocks and reader.

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
| R4b real tasks at 128K (both models) | `script_stage1h.sh --run-r4`: per cell a pilot → gate → 2 blocks; cells: LongBench v2 at Llama 128K (1 GPU) and Qwen 128K (2 GPUs), 40 items each; HELMET (kilt_nq, kilt_hotpotqa, msmarco_rerank_psg, icl_trec_coarse, icl_banking77) at Llama and Qwen 128K, 10 items × 5 tasks; then the reader → `findings/R4_reader.{json,md}` | rebuilt 2026-10-05 (replaces the LongBench v1 design, which never ran): manifest `data/r4/manifest_r4.json` pinned; fast tests, the CPU driver smokes (every arm on lbv2, kilt_nq and icl_trec_coarse at 4K) and the trig-login01 preflight + `--run-r4-dry` pass. **lb2llama submitted** 2026-10-05: pilot 1036790, gate 1036791, blocks 1036792 1036793 (no reader yet). The chain then stopped: Trillium refused the 2-GPU Qwen pilot (1 GPU or whole nodes only). Fixed (plan.md R4b execution note: a Qwen node job runs both blocks at once). Pre-submit audit: the gate's answer-mask check failed any pilot FP got wrong (fixed, plan.md R4b validity fix; reaches lb2llama's queued gate 1036791, which loads the reader at run time); Qwen3-0.6B driver smokes added and passing. **The other three cells submitted** 2026-10-05 (`R4_CELLS=hmllama,lb2qwen,hmqwen R4_EXTRA="--lb2llama 1036792 1036793" --run-r4`): lb2qwen pilot 1036920, gate 1036921, node job 1036922 (blocks read as 1036922_0/_1); hmllama pilot 1036923, gate 1036924, blocks 1036925 1036926; hmqwen pilot 1036927, gate 1036928, node job 1036929 (1036929_0/_1); reader 1036930 (after all six block jobs, lb2llama's included) → `findings/R4_reader.{json,md}`; gates write `findings/R4_gate_<cell>.json`. Status: `--status 1036792 1036793 1036922 1036925 1036926 1036929 1036930`. **2026-10-07:** lb2llama (1036790–93) and hmllama (1036923–26) COMPLETED, gates PASS, the reader reads them cleanly; 1036920–22 and 1036927–30 cancelled before running. Rorqual pilots: lb2qwen 22650148 (gate PASS, read here), hmqwen 22651018 (killed at its 2 h limit while loading Qwen's weights, 357/531 tensors after 74 min), duplicates 22650144 / 22651014. Resume ON RORQUAL: `bash $DIR/script_stage1h_temp.sh [--dry]` (lb2qwen blocks as two 2-GPU jobs; hmqwen pilot, gate, two 2-GPU blocks; the reader over all four cells if Trillium's block results are copied there). Jobs copy the model to $SLURM_TMPDIR first (worker S1H_STAGE=1); walls: pilot 4 h, blocks 6 h / 8 h. **2026-10-08:** hmqwen pilot 22694021 (Rorqual) finished, gate PASS (42 GiB per GPU, block 2.0 h). Resume ON TRILLIUM: `bash $DIR/script_stage1h_temp2.sh [--dry]` on trig-login01 — finds finished pilots and blocks from results/ (never resubmits them), submits the missing Qwen blocks as 4-GPU node jobs (two blocks at once) and the reader over all four cells. Cancel any R4 Qwen jobs still queued on Rorqual first. About 18 GPU-h (Qwen's 2-GPU cells about 10) |
| R5 (draft, plan.md amendment "R5") | R5.0 done: `findings/R5_0_split.md` (the vote, not the budget or the store, is the failure). Code: `cert_s1h5.py`, `probe_s1h5.py`, `s1h5_lib.py`, `run_s1h5.py` | theory checked by brute force; CPU driver smokes pass (probe = FP, bounds hold on real attention, injected missed mass on target); worst-case certificate loose on the 1B smoke (median 11×), high-probability variant added. **R5 pilot submitted 2026-10-08 (Trillium):** 1059459 (Llama 128K r3), 1059460 (Llama 128K r4 HELMET), 1059461 (Qwen 32K r3), 1059462 (Qwen 32K r3b), reader 1059463 → `findings/R5_pilot.{json,md}` (`script_stage1h_r5.sh --pilot-r5`). Pilot PASS (validity on both models and all suites; peak 49 / 64 GiB). Fix before freezing: delta injection (the pilot's KL floor at small eps was the fp32-vs-bf16 recompute). **R5 frozen and submitted 2026-10-08:** blocks 1059655–1059665 (Llama 128K: r1 ×2, R3a ×2, cwe/fwe, HELMET, LongBench v2; Qwen 32K: RULER, R3a ×2, cwe/fwe), check reader 1059666 → `findings/R5_blocks_check`; ≈ 25 GPU-h. Qwen 128K cells wait for R4's Qwen. **R5 read 2026-10-08:** `findings/R5.md` (tables `R5_reader.md`): 10 blocks valid, LongBench v2 excluded by V (bf16 decode nondeterminism; appendix); the tail design (nothing evicted, unselected rows from the 4-bit tier) is near FP8 everywhere and removes the vote's failure. **R5.3 (the tail design's scan) frozen and submitted 2026-10-08:** blocks 1060323–1060329, reader 1060330 → `findings/R5_3_reader`; ≈ 8 GPU-h; test models downloaded (gpt-oss-20b, DeepSeek-V2-Lite-Chat, Moonlight-16B-A3B-Instruct, gemma-3-12b-it, granite-4.0-h-tiny, Kimi-Linear-48B-A3B-Instruct) |
| R5 theory part 3, adapters, R6 draft (2026-10-09, CPU only, no jobs) | `cert_s1h5.py` (Lemmas 4–5: `tail_bound_det`, `tail_bound_conc` with the MLA coupled form and a score-bias allowance), probe columns (`_tail_certs`, `_tail_err`: `l4_*`, `l5_*`, `l5b_*`, `errd_*`, `tailx_p*`), `adapters_s1h5.py` + `test_adapters_s1h5.py`, `size_confirm_r5.py` | proofs in plan.md "R5 theory, part 3"; R6 confirmatory plan in plan.md "R6 (draft)" with `findings/R5_sizing.md`; adapters pass on all six downloaded models + Llama-1B (`findings/R5_adapters.md`; Kimi's MLA layer runs alone: its KDA layers need fla). Found: tier-1 score errors regress to the mean (≈ 0.25σ, coherent), which made Lemma 5 (no allowance) miss on 12–18 of 16,384 head-steps (one row group) at the r3b CPU smoke's extra-row reads; with a 0.5σ allowance it misses nowhere; the probe records both. Next: R5.3's read picks R6's arm; rerun `size_confirm_r5.py` with every R5.3 block; freeze R6 |

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
