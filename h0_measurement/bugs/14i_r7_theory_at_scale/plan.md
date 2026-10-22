# R7a plan: the theory at the paper's scale (FROZEN 2026-10-12)

The question, the measurements and the predictions are in `theory_r7.md`; this file is the run. Drafted 2026-10-11,
reviewed end to end and extended to Qwen 128K on 2026-10-12, then frozen on the user's go-ahead:
1. `r7_lib.AMEND_R7A = "R7a"`;
2. `bash script_r7a.sh --pin-code` (`findings/R7a_code.sha256`);
3. `bash script_r7a.sh --run-r7a-dry`;
4. `bash script_r7a.sh --run-r7a` from trig-login01 (§10).

Nothing below changes while any R7a job has output. Every job re-checks the code pin at its start.

## 1. What R7a decides

Whether the theory section's tight results hold on R6's models and lengths (Llama-3.1-8B at 128K, Qwen3-30B-A3B at
32K and at 128K: all three of R6's claim groups). These are the ledger's T3, T4 and T6, CPU-only so far:
- part 8's certificate;
- the certified adaptive read;
- Theorem M.

It also decides whether R6's primary statistic depends on the dither seed (C1's gap). It is a verification run: it
tests no design claim, and its labels say whether the predictions stated in advance (`theory_r7.md` §4) hold.

## 2. Design

### Panel (a): the probe, on R6's seven probe kinds with fresh prompts, and Qwen 128K
Arms per unit, in one process: `fp`, then R7a's probe (`probe_r7`), which leaves FP's outputs unchanged.

| block | tag | preset | tasks | prompts | units | wall |
|---|---|---|---|---|---|---|
| l-r1 | r7llama128_r1 | r7llama128 | niah_single, niah_multikey, niah_multivalue, vt | 9700–9704 | 20 | 6 h |
| l-r3a | r7llama128_r3a | r7llama128 | niah_multikey, niah_multivalue, vt, mk_panel (R6's R3a level) | 9705–9709 | 20 | 6 h |
| l-agg | r7llama128_r3b | r7llama128 | cwe, fwe (R6's R3b level) | 9710–9714 | 10 | 5 h |
| l-hm | r7llama128_hm | r7llama128 | kilt_nq, kilt_hotpotqa, msmarco_rerank_psg, icl_trec_coarse, icl_banking77 | R6 manifest items 16–20 | 25 | 7 h |
| q-r1 | r7qwen32_r1 | r7qwen32 | as l-r1 | 9750–9754 | 20 | 4 h |
| q-r3a | r7qwen32_r3a | r7qwen32 | as l-r3a (Qwen's level) | 9755–9759 | 20 | 4 h |
| q-agg | r7qwen32_r3b | r7qwen32 | cwe, fwe | 9760–9764 | 10 | 3 h |
| w-hm | r7qwen128_hm | r7qwen128 | as l-hm | R6 manifest items 16–18 and 19–21: one 4-GPU node job, two 2-GPU blocks | 30 | 6 h |

- **Fresh units.** The RULER prompts and the pilots appear in none of the 4,242 earlier (task, prompt) entries under
  `results/` (`test_run_r7.test_fresh`).
- **HELMET.** HELMET has no unused items: R6 used items 11–40. Items 16–20 (Llama) were run by R6's non-probe block
  `l-hmb`, and items 16–21 (Qwen 128K) by R6's Qwen 128K blocks, which had no probe at all. So they are fresh to the
  probe. The theorems hold for every input, so reuse does not bias coverage.
- **Qwen 128K** is R6's third claim group, which R6 never probed (it would have cost node jobs). Trillium refuses 2-GPU
  jobs, so its two blocks share one 4-GPU node job (R6's and R4's node split), read as `JOB_0` and `JOB_1`.

**The probe's settings** (`probe_r7`, recorded in every sidecar):
- **Keys.** 4/6/8 bits, with 2/5/3 dither draws. The design's 6 bits gets the ≥ 5 draws of seed panel (a).
- **Values.** 4 bits.
- **R6's tier.** Rotations seeds 50/150; draw seeds 1000003·draw + 2·layer (draw 0 is R6's probe draw); unbiased tail; the
  vote's rows (the question's vote at 1/8 over the coin-free 4/4 store) read exactly.
- **Bounds.** δ = 10⁻³ per head-step:
  - v2 (R6's column);
  - part 8 (HKZ key sum, P-exact on 16 rows per head-step, remedy 1 on the sketch's 16 heaviest unread rows, Bennett as
    a second form at 4 bits), with the exact HKZ matrix.
- **Theorem M** at each width, draw-free. **The unread shares.**
- **The certified read.** At 6 and 8 bits, on the first chunk of 4 measured steps of every layer, draw 0. It uses the
  FINE schedule (16 coin-free sets in the sketch's order, each at δ/16), FP8's per-head error as the target, the exact
  matrix and Lemma B's brackets. Its guarantee holds per query head (`theory_r7.md` §2.6), which is what the reader tests.
- **Coverage of steps.** Up to 32 measured steps per unit (R6's rule), every layer, every head; float64 throughout.

### Panel (b): the design's KL across dither seeds, on R6's own units
Arms per unit, in one process:
- `fp`;
- `dtail6ux_v4` at R6's seed 7000;
- `dtail6ux_v4_s1` … `_s4` at seeds 7000 + 100003·k;
- `fp8kv`;
- `fp_noise` last.

| block | tag | preset | tasks | units (R6's) | wall |
|---|---|---|---|---|---|
| lk-rr | r7kllama128_hm | r7kllama128 | msmarco_rerank_psg | items 11–20 (10) | 6 h |
| lk-mv | r7kllama128_r1 | r7kllama128 | niah_multivalue | 9500–9509 (10) | 6 h |
| lk-ret | r7kllama128_r1 | r7kllama128 | niah_single, niah_multikey | 9500–9504 (10) | 6 h |
| qk-mv | r7kqwen32_r1 | r7kqwen32 | niah_multivalue | 9600–9609 (10) | 3 h |
| qk-ret | r7kqwen32_r1 | r7kqwen32 | niah_single, niah_multikey | 9600–9604 (10) | 3 h |

The families were chosen as `RESEARCH_PLAN.md` §4 proposed: re-ranking, the weak family; multi-value; and retrieval.

### Pilots and gate (as R6)
Pilots (`r7_lib.R7_PILOT`): one Llama 128K unit with the probe (niah_multivalue, prompt 3120) and one with the seed panel
(prompt 3121), 1 GPU each.

`read_r7a.py --gate` checks:
- validity;
- FP's KL 0;
- the pilot's misses within the theorem's δ (the same COVER_OK rule as the reader);
- peak ≤ 76 GiB per GPU;
- every 128K block's projected time ≤ 90% of its wall. Qwen 128K's node job is projected from the probe pilot at 1.5×
  per unit (48 layers against 32, with the same per-layer tensors) plus 60 s.

Every block waits on the gate (`afterok`).

## 3. Reader (`read_r7a.py`; rules in its docstring)
- **Panel (a), per group and width:**
  - coverage: misses against δ·checks, COVER_OK or COVER_EXCESS;
  - bound / error, part 8 and v2, and their ratio; the certified share within FP8;
  - part 8's K/V/D composition;
  - Theorem M: R_exact, the identity check, and R_first, the first-order reading, with unit-bootstrap intervals;
  - the per-layer map √m_pred / e₈;
  - the certified read (rows, traffic, oracle, fallback; query heads met, against δ per head);
  - the unread share and the certified share per family.
- **Panel (b), per group and family:** the two-way decomposition (unit, common seed effect, residual); R6's statistic at
  each seed; with one seed per bootstrap replicate; two-level; SEED_ROBUST, SEED_SENSITIVE or SEED0_NOT_PASSING.
- **Predictions P1–P8:** MET, NOT_MET or NOT_READ (`theory_r7.md` §4). P8 needs R6's reader JSON (`--r6-json`).
- **Validity:**
  - an R7a block whose tag, preset, suite and tasks are a layout row; the preset's plan; every arm on every unit;
    R7a's units;
  - the probe's pass is FP's, with frozen settings;
  - dtail arms evict nothing; peak ≤ 76 GiB; R6's HELMET manifest;
  - identical code manifests.

## 4. Cost (estimates; the gate measures them)
There is no GPU timing yet; the CPU smokes only show that the probe's cost is part 8's bound and the certified read. A
GPU estimate from flop counts, per Llama 128K unit (32 layers × 32 heads × ≤ 32 steps, C = 131K):
- part 8 at the vote's rows, 10 (width, draw) passes × 256 chunks × ≈ 50–80 ms, ≈ 2–3.5 min (the HKZ matrix, the P-exact
  series and the elementwise moments dominate);
- the certified read, 2 widths × 32 chunks × 16 sets ≈ 1 min;
- Theorem M ≈ 10 s;
- FP's generation, the vote and the 4/4 store ≈ 1–2 min.

| part | units | per unit | GPU-h |
|---|---|---|---|
| Llama 128K probe (4 blocks) | 75 | 4–8 min | 5–10 |
| Qwen 32K probe (3 blocks) | 50 | 2–4 min | 1.7–3.3 |
| Llama 128K seed panel (3 blocks) | 30 | 8 arms × 1.1–1.5 min (R6's rate) | 4.5–6 |
| Qwen 32K seed panel (2 blocks) | 20 | 8 arms × 0.4–0.6 min | 1–1.6 |
| Qwen 128K probe (1 node job, 2 blocks in parallel) | 30 | 6–10 min (1.5× Llama's probe + 2-GPU prefill) | 4 GPUs × 1.6–2.7 h ≈ 6.5–11 |
| pilots, gate, reader | — | — | ≈ 2 |
| **total** | | | **≈ 21–34** |

`RESEARCH_PLAN.md` §1.1 expected 12–20 without Qwen 128K, and D5 about +4 GPU-h for it; the node job is charged for 4 GPUs.
R6's measurements support the per-unit figures: R6's probe took 67–95 s per Llama 128K unit, R6's Qwen 128K blocks
113–121 s per unit with 8 arms (peak 44 GB per GPU), and R7a's probe holds about 11 GB of float64 state per chunk
against R6's ≈ 24 GB probe overhead (peak 54–57 GB on Llama).

## 5. Predictions (`theory_r7.md` §4; `read_r7a.PRED`)
- **P1:** coverage COVER_OK everywhere.
- **P2:** part 8 / error at 6 bits in [4, 7] (Llama), [6, 13] (Qwen).
- **P3:** certified within FP8 at 6 bits ≥ 0.80 (Llama), ≥ 0.65 (Qwen).
- **P4:** Theorem M exact consistent with 1, first order ±5% at 6–8 bits.
- **P5:** the certified read at 6 bits, traffic ≤ 0.45 (Llama) and ≤ 0.50 (Qwen), with met ≥ 1 − 2δ.
- **P6:** part 8 / v2 ≤ 0.30.
- **P7:** SEED_ROBUST.
- **P8:** the unread share orders R6's families by the design's KL.

## 6. What R7a reads and what it may not change
- **Read-only:**
  - R6's library and its whole chain (`run_s1h5`, `s1h5_lib`, Stage 1g, `run_r8`, `sievelib`);
  - R6's HELMET manifest and items, R6's R3a/R3b levels;
  - R6's probe results (for the predictions only).

  `findings/R7a_code.sha256` pins all of it, with this folder's code and fixtures (119 files). Every preflight verifies it.
- **R6's files are not edited.** If they change after R6's final read (e.g. a port into `cert_s1h5`), R7a's preflight
  and every R7a job's start (the worker re-checks the pin) fail by design. Re-pin deliberately only before any R7a output.
- **The bounds R7a measures are its own** (`cert_r7`, `tier_r7`). An edit of `cert_s1h5` or `rquant_s1h5` cannot move
  R7a's numbers; `test_cert_r7` reports whether the copies still match.
- **R7a needs no `plan.md` amendment in `../14h_methodology_improve/`.** The bounds reach the run through this folder's
  run-time modules. This is `RESEARCH_PLAN.md` §5's proposed rule (decision D6) in practice.

## 7. Decisions for the user
1. **Approve and freeze R7a** (panels a and b, or panel a only). Nothing is submitted without your go-ahead.
2. **D5: Qwen 128K in the probe:** decided by the user (2026-10-12): yes, as block `w-hm`.
3. **Order with R6's final read.** R7a reads R6's files read-only and does not wait for R6's reader. Its preflight
   refuses to run if R6's files change after the pin.

## 8. Files
| file | role |
|---|---|
| `cert_r7.py`, `tier_r7.py` | the bounds and the tier (copies plus the GPU forms; `theory_r7.md` §2) |
| `probe_r7.py` | the measurement hook (panel a) |
| `r7_lib.py` | presets, the seed-suffixed design arms, `R7_LAYOUT`, `R7_PILOT` |
| `run_r7.py` | the driver: `run_s1h5.main` with `r7_lib` and `probe_r7` swapped in at run time |
| `read_r7a.py` | the reader and the gate |
| `submit_r7.slurm`, `script_r7a.sh` | the worker and the chain (`--run-r7a[-dry]`, `--pin-code`, `--list-r7a`) |
| `test_cert_r7.py`, `test_run_r7.py` | the bounds' checks (`--quick` in the worker) and the run side's (`--fast` in the worker; driver smokes without it) |
| `explore_r7.py`, `findings/explore_r7.md` | the CPU exploration behind the predictions |

## 9. Commands
```bash
cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14i_r7_theory_at_scale/test_cert_r7.py --fast
OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14i_r7_theory_at_scale/test_run_r7.py        # + driver smokes
bash h0_measurement/bugs/14i_r7_theory_at_scale/script_r7a.sh --run-r7a-dry
# after the run:
.venv/bin/python -u h0_measurement/bugs/14i_r7_theory_at_scale/read_r7a.py \
    --r7a $(bash h0_measurement/bugs/14i_r7_theory_at_scale/script_r7a.sh --list-r7a) \
    --r6-json h0_measurement/bugs/14h_methodology_improve/findings/R6_reader.json
```

## 10. Submission (2026-10-12, Trillium, from trig-login01; the user's go-ahead)
Preflight passed: code manifest verified (119 files), `test_cert_r7 --quick` and `test_run_r7 --fast` passed,
`AMEND_R7A = "R7a"`.

| job | what | depends on |
|---|---|---|
| 1080289 | pilot `p-probe` (r7llama128_pilot, niah_multivalue 3120) | — |
| 1080290 | pilot `p-kl` (r7kllama128_pilot, niah_multivalue 3121) | — |
| 1080291 | gate (`read_r7a.py --gate`) → `findings/R7a_gate.json` | afterok 1080289, 1080290 |
| 1080292–1080298 | probe blocks l-r1, l-r3a, l-agg, l-hm, q-r1, q-r3a, q-agg (1 GPU each) | afterok 1080291 |
| 1080299 | probe node job w-hm (Qwen 128K, 4 GPUs; blocks `1080299_0`, `1080299_1`) | afterok 1080291 |
| 1080300–1080304 | seed-panel blocks lk-rr, lk-mv, lk-ret, qk-mv, qk-ret (1 GPU each) | afterok 1080291 |
| 1080309 | reader (`read_r7a.py --r7a $(script_r7a.sh --list-r7a)`) → `findings/R7a_reader.{json,md}` | afterany 1080292–1080304 |

The chain script's own reader submission failed: Trillium's sbatch refused a `--wrap` naming 14 TAG:JOB pairs ("parameter
too long"). The reader was resubmitted by hand as 1080309, listing the finished blocks when it runs, and the script now
does the same. No block was affected.
- **If the gate fails,** the blocks never start (`afterok`) and the reader waits on them: cancel 1080292–1080304 and
  1080309, read `findings/R7a_gate.json`, and fix before any block runs.
- **After the run:** rerun the reader with `--r6-json <R6's final reader JSON>` once R6's final read exists (P8).

## 11. Outcome (added 2026-10-22; §1–§10 are the frozen record)
- **Trillium.** The pilots and the gate ran as planned: 1080291 PASSED at 08:52 on 2026-10-12. The blocks 1080292–1080304
  and the reader 1080309 were then cancelled while still pending (09:17–09:23, from the user's account).
- **Rorqual.** The run was resubmitted there; the reason for the move is not recorded.

  | job | what |
  |---|---|
  | 23020651, 23020652 | pilots (probe, seed panel) |
  | 23020654–23020660 | probe blocks l-r1, l-r3a, l-agg, l-hm (23020657), q-r1, q-r3a, q-agg |
  | 23020661 | Qwen 128K node job w-hm (`_0`, `_1`) |
  | 23020662–23020666 | seed-panel blocks lk-rr, lk-mv, lk-ret, qk-mv, qk-ret |
  | 23020667 | reader: **did not run** |

  Every block finished. l-hm, the last one, was copied back on 2026-10-22. The blocks took about 16–17 GPU-h in all,
  against §4's 21–34.
- **The read.** Job 23020667 did not run, so the reader was run here instead, with the same frozen `read_r7a.py`:
  `findings/R7a_reader.{md,json}`, from all 14 blocks.
  - All blocks are valid and their code manifests are identical.
  - **All eight predictions are MET.** The write-up is `findings/R7a.md`.
- **P8 is closed on R6's partial read of 2026-10-18** (`../14h_methodology_improve/findings/R6_reader_partial_1018.json`),
  by the user's decision of 2026-10-22, not on R6's final read as above.
  - P8 uses only R6's Llama 128K rows. All of R6's Llama blocks had run by then, and those rows are identical in R6's
    2026-10-11 and 2026-10-18 reads.
  - R6's final read waits only on a Qwen 128K block.
- **Freeze.** After the read, R7a's files other than `cert_r7.py` and `tier_r7.py` were unfrozen. Those two stay frozen
  for R7b. The code that produced the numbers is kept in `findings/R7a_code_snapshot_1022.tar.gz`, which verifies
  against `findings/R7a_code.sha256`.
