# 14i_r7_theory_at_scale/r7b status (rewritten, not appended)

**2026-10-22: interim read with Qwen 128K's first blocks** (`findings/R7b_reader_partial_1022.{md,json}`).
- **New from Rorqual** (copied here on 10-22, 12:27–12:29):
  - Qwen 128K's pilot 23453079 and 6 of its 8 core blocks: w-hma/b/c .0 and .1, jobs 23453082–86 and 88, the RAG and
    ICL families;
  - 4 Rorqual copies of Qwen 32K's r1 blocks, 23453109–11 and 13. These are duplicates: Trillium had already run them.
- **Still missing for Qwen 128K:**
  - the re-ranking blocks w-rr.0/.1 (23453089–90);
  - calibration wc1-hm.0/.1 and wc2-hm.0/.1;
  - exploratory wx-hm.0/.1;
  - Rorqual's gate file, ledger and logs, none of which were copied.
- **Validity, all 49 core blocks:** every block valid; **code manifests identical across Trillium and Rorqual**, so
  `rorqual_r7b.sh`'s path setup works; nothing excluded. The four duplicates are read once, from Trillium's copy (the
  reader's first-by-job rule). The pin verifies.
- **The Qwen 128K gate**, recomputed here from the pilot into a scratch folder (Rorqual's gate file was not copied): PASS.
  136 s per unit, peak 44.3 GiB, the longest block projected at 1.69 h (w-rr) against 4 h.
- **Llama 128K and Qwen 32K: unchanged** from 10-19.

| Qwen 128K family | primary test | units | design / FP8 / fp_noise KL per token | E4 (ratio of means) |
|---|---|---|---|---|
| rag | PASS (U −0.0044) | 60 | 0.0025 / 0.0067 / 0.0035 | 1.20 (0.74–1.78) |
| icl | NOT_TESTABLE: design U +0.00003, but fp_noise U +0.00097 ≥ 5e-4 | 60 | 0.0010 / 0.0013 / 0.0016 | 1.67 (0.70–3.33) |
| rerank | no data (w-rr missing) | 0 | | |

- **The group reads INSUFFICIENT for now:** 1 of 3 families is testable. If re-ranking passes, 2 of 3 are testable and
  the group reads NONINFERIOR.
- **Gates:** KIVI-4, dense 4/4 and Quest are BEATEN (p 0.0005, 0.0005, 0.0030). Dense 7/6 is NOT_SHOWN (p 0.95; mean d
  +0.00043, so dense 7/6 is slightly better on RAG and ICL here).
- **Accuracy: ACC_FAIL at δ = 0.02** (n 86, mean 0.0000, lower −0.0233); ACC_PASS at 0.05.
  - The cause is one discordant unit each way: kilt_nq 46 (the design right, FP8 wrong) and kilt_nq 65 (the design
    wrong; FP8, E4 and fp_noise right).
  - At n = 86, one such unit puts the bootstrap's lower bound past −0.02. The rule calls that FAIL at any n; P-b3 had
    predicted ACC_UNDERPOWERED. Quest and dense 7/6 get ACC_FAIL the same way.
  - Re-ranking adds no FP-correct units: FP never gets a re-ranking item fully right.
  - Only the optional `extra` stage, about 85 more FP-correct units, would power the test. Deciding on it after this
    result is data-dependent, so its read would be a sensitivity analysis, not the confirmatory result.
- **Cross-folder note (R7a was unfrozen on 10-22, except `cert_r7.py` and `tier_r7.py`):** R7b's start-up tests
  (`test_run_r7b.py`, run by every R7b job) import R7a's `r7_lib.py`, which R7b's pin does not cover. Don't copy a
  changed `r7_lib.py` to Rorqual while R7b jobs are queued there.
- **Next:**
  1. When Qwen 128K's last blocks finish, run `rorqual_r7b.sh results` on Rorqual and copy that list here. It includes
     the gate file, the ledger and the logs.
  2. Then the full three-group read (`script_r7b.sh read-core`, `read-explore`).
  3. Cancel any Rorqual copies of Trillium's blocks still queued.
  4. The user decides whether to run `extra`.

**2026-10-19: interim read, Llama 128K and Qwen 32K** (2 of 3 claim groups; Qwen 128K runs on Rorqual).
- **All 24 Trillium jobs COMPLETED.** The 13 that had waited since 2026-10-12 ran Oct 18 08:14 to Oct 19 07:33.
- **The readers:**
  - inputs: all 39 core blocks of the two groups (52 Trillium result folders in all) and their 5 exploratory blocks;
  - outputs: `findings/R7b_reader_partial_1019.{md,json}` and `findings/R7b_explore_partial_1019.{md,json}`;
  - validity: every block valid, code manifests identical, nothing excluded, no duplicate units; the pin verifies.
- **Calibration stage 2 (from the exploratory jobs' pre-steps):** both groups bracketed, exploratory gate PASS. The
  levels equal the earlier copy-based computation: Llama t1/t2 B = 0.217 / 0.0302, Qwen 32K 4.35 / 0.538.

| claim group | primary (single store vs FP8) | accuracy δ = 0.02 | KIVI-4 / dense 4/4 / Quest | dense 7/6 |
|---|---|---|---|---|
| Llama 128K | NONINFERIOR, 7 of 7 families PASS | ACC_PASS (n 180, lower −0.0145) | BEATS ×3 | NOT_SHOWN (p 0.23) |
| Qwen 32K | NONINFERIOR, 3 of 4 PASS (aggregation NOT_TESTABLE: fp_noise U +0.0019, as in R6) | ACC_PASS (n 416, lower −0.0003) | BEATS ×3 | NOT_SHOWN (p 0.33) |

- **Qwen 32K multivalue at 300 units:** PASS (U −0.00002). The memory claim (0.32 at rest, 0.45 during a question, vs
  FP8's 0.50) is confirmatory in these two groups.
- **E4 (single store / R6's design):**
  - Llama: ratio of means 1.00 [0.84, 1.16], geometric mean 0.89 (families 0.61–1.08);
  - Qwen 32K: ratio of means 1.29 [1.00, 1.65], geometric mean 0.97.

  Qwen 32K multivalue's ratio of means, 1.87, comes from a few perturbation-sensitive units, mostly at R6's hard level.
  Prompt 9872 has design 8.5e-3 against E4 6.6e-6, with fp_noise 3.5e-2 on that same unit.
- **Against the predictions:**
  - **P-b1** met (2 groups). **P-b3** met.
  - **P-b2:** per unit, better than predicted (geometric means 0.89 / 0.97 against 1.1–1.4). On mean KL, met for Llama
    (1.00) but not for Qwen 32K (1.29).
  - **P-b4:** KIVI-4, dense 4/4 and Quest BEAT in both groups; **dense 7/6 does not** (NOT_SHOWN, mean d ≈ 0).
    - On Llama, the design's KL is lower on most units of multivalue, tracking (0.90), retrieval (0.79) and ICL (0.63).
    - On re-ranking, the one family clearly above the noise floor, dense 7/6 is 2.6× lower (8.9e-4 against 2.3e-3; the
      design lower on 0.23 of units), so the pooled test cancels.
  - **P-b7 (E1):**
    - traffic met: Llama 0.4007 / 0.4086, Qwen 32K 0.4166 / 0.4246; each level at its target, floor + 0.002 / + 0.01;
    - coverage met: 0 misses in 1,338,368 + 1,503,744 per-query-head checks (COVER_OK);
    - the predicted 4–20× lower KL is not resolvable: KL / design 0.63–0.89.
  - **P-b8, P-b9 (E2, E3) not resolvable.** E2 4+3/4+2: Llama 1.34 (geometric mean 1.23), Qwen 32K 0.53 (0.82). E3
    water-filled / 4+1: Llama 0.37 (0.76), Qwen 32K 1.16 (0.93).
  - **Why "not resolvable":** the design is at FP's rerun noise floor.
    - Llama's exploratory units: design mean KL 3.7e-4 against fp_noise 3.5e-4; 75% of units within 1.5× fp_noise.
    - Qwen 32K's: 7.8e-4 against 2.1e-3; 80% of units within 1.5× fp_noise.
    - So nothing below the design is measurable here; the CPU predictions had no such floor.
- **Rorqual:**
  - submitted with an earlier `rorqual_r7b.sh`: Qwen 128K 23453079–23453094; copies of Trillium's blocks 23453096–101
    (l-hm) and 23453109–134 (q-*);
  - cancelled by the user (they had already run on Trillium): the six calibration re-runs 23453102–05, 135, 146, and the
    exploratory jobs that waited on them, 23453106–08, 148–149;
  - the copies are now redundant. Cancel them; the reader would keep Trillium's in any case.
- **Next:** Qwen 128K's results from Rorqual (`rorqual_r7b.sh results`, copy back), then the full three-group read.

**2026-10-15, 10:00 (checked from tri-login06).**
- Finished on Trillium:
  - both pilots, and both gates PASS: Llama 148 s per unit, peak 55 GiB; Qwen 32K 95 s, 64 GiB;
  - calibration stages 1 and 2 for both groups;
  - Llama core jobs 1083801–03, 12 of its 18 blocks. All 12 are valid. Their measured hours are 0.6–1.3× the cost
    model: R3a above, standard below, because R6's per-task times pool both levels.
- Waiting with reason Priority since 2026-10-12: Llama core 1083804–05; Qwen 32K core 1083807–12; explore
  1083817–21.
  - FairShare is 0.14 (def-enright has no allocation). Priority is about 716k, against about 1.36M at the top of the
    queue.
  - The age factor adds only about 5k, so these jobs start only by backfill.
- Calibration stage 2 computed on a copy (not written): both groups are inside the scan and their exploratory gates
  pass.
  - E1 costs 1.4–9.3 s per answer token across the scan, and about 4.6–7 s at the frozen levels.
  - The exploratory blocks fit their walls, because their standard-level answers average 21–28 tokens.
  - Note: `explore_projection` projects from the calibration answers' length, not each exploratory block's tasks. It
    did not matter here (cwe in `lx-agg` is the longest, about 2.3 h of 4 h).
- No R7b result of Qwen 128K and no Rorqual ledger are here.

**Updated 2026-10-12, 14:55.** R7b is FROZEN (`AMEND_R7B = "R7b"`; `findings/R7b_code.sha256` pins 126 files) and its
Trillium half is SUBMITTED from trig-login01 (the user asked). The ledger is `findings/R7b_jobs_trillium.tsv`.

| stage | jobs | waits on |
|---|---|---|
| `pilot` | 1083794 (Llama 128K), 1083795 (Qwen 32K); node jobs that run the gate | — |
| `calib1` | 1083796 (`lc1`), 1083797 (`lc1-hm`), 1083799 (`qc1`) | — |
| `core` | Llama 1083801–1083805; Qwen 32K 1083807–1083811 and 1083812 (`q-aggc`, 1 GPU) | their group's pilot |
| `calib2` | 1083813 (`lc2`), 1083814 (`lc2-hm`), 1083815 (`qc2`) | calib1 |
| `explore` | 1083817–1083819 (Llama), 1083820–1083821 (Qwen 32K) | calib2 |

- At 14:53, 1083796 and 1083797 were running. Their worker preflight passed on the GPU node (pin, both test suites).
- The pilots were estimated to start at 15:05 and 15:16.
- **Rorqual's half (Qwen 128K) is not submitted.** To submit it:
  1. Copy `--list-sync`'s files to Rorqual.
  2. There, run `pilot`, `calib1`, `core`, `calib2`, `explore`, and `extra` if chosen.
- **Nothing in the pin may change while these jobs are queued.** That covers R7b's `*.py`, the worker, R7a's `cert_r7` and
  `tier_r7`, every `14h_methodology_improve/*.py`, `14_kernel_tpot/*.py`, `sievelib`, `run_h0.py`, `run_r8.py` and the
  HELMET data. A changed file makes every later job stop at its pin check.
- A gate FAIL leaves that group's core waiting forever: `--cancel-orphans`.
- Fixes made just before the freeze:
  - the preflight's dry-run worker test no longer depends on the node's GPU count or on the job's exported `R7B_*`
    settings; either would have stopped 1-GPU or packed jobs;
  - Llama's 3 h walls and `q-mv`'s are now 4 h, so each gate passes a pilot up to 2.4–3.2× R6's mean time.

## Decisions applied, and submission by stage (2026-10-12, late evening; the user asked)
**The decisions** (`plan_r7b.md` §8):
- Qwen 32K multivalue is 300 units: `q-mva-j`, 10 blocks of 26, prompts 10100–10359, fresh.
- The dense 7/6 gate is tested in every group (`read_r7b.GATE_DENSE_GROUPS`).
- Re-ranking reuses R6's items 11–40; the reader already marks them "R6 units".
- E1's rule and its loose layers are as built.
- TurboQuant's product variant stays out.
- `R7B_EXTRA` stays optional, as its own stage.

**Stages** (`script_r7b.sh STAGE [--dry]`): `pilot`, `core`, `extra`, `calib1`, `calib2`, `explore`, then `read-core`
and `read-explore`.
- Each is one command. A stage may be submitted as soon as the one it follows is queued: its jobs wait on that stage's
  jobs (`afterok`).
- The planner `jobs_r7b.py` decides what a stage submits; the script submits only what it prints.

New pieces:
- **One pilot and gate per claim group** (`lib_r7b.R7B_PILOTS`; `findings/R7b_gate_<group>.json`). The pilot's own job
  runs the gate (`submit_r7b.slurm R7B_POST=gate`), so the gate costs no queue wait. Each cluster's half is then
  self-contained.
- **The calibration steps run inside the next stage's jobs** (`R7B_PRE`).
  - Each stage-2 job first runs `calib_r7b.py stage1 --blocks auto`.
  - Each exploratory job first runs `stage2 --blocks auto --strict`, which stops the job unless the levels and the
    exploratory gate are ready.
  - Both skip work already done and write atomically. No hand step, no extra queue wait.
- **Two clusters, no duplicates.** The default `R7B_ASSIGN` is R6's split: Trillium runs Llama 128K and Qwen 32K;
  Rorqual runs Qwen 128K, as 2-GPU jobs with the model staged and `--mem=96G`. Each group runs on one cluster.
  - A cluster submits only its own groups.
  - A block goes out only if it is neither finished (`results/`) nor queued (this cluster's ledger and `squeue`).
  - A group with jobs in another cluster's ledger is refused (`R7B_MOVE=1` after moving it).
  - Sync lists: `--list-sync` (code to Rorqual), `--list-results` (results back).
- **Packing on Trillium.** On 2026-10-12, whole nodes were estimated to start in 14 min, single GPUs in 2 h 20 min
  (`sbatch --test-only`).
  - The pilot, core and extra stages pack a group's blocks four to a node, the longest first (worker `R7B_B<i>`).
  - Packing costs about 5 GPU-h on the core.
  - The calibration and exploratory stages stay 1-GPU jobs: packing them would cost about 26 GPU-h of idle GPUs.
- **Utilities:** `--freeze` (AMEND, tests, pin), `--status`, `--probe-queue`, `--cancel-orphans` (jobs left waiting on a
  failed gate or step). Trillium's Slurm 25.11 refuses `--kill-on-invalid-dep`, hence `--cancel-orphans`.
- **Cost** (`explore_r7b.py cost`, the plan as packed): Trillium 60.5 / 52.4 GPU-h billed, Rorqual 24.0 / 23.1, and +8.5
  with `extra` (E1's upper / lower rate).
- **Checks:**
  - `test_run_r7b.py --fast` passes, with the new `test_stages`: blocks, packing, cluster ownership, waits, the gate
    under the current pin, duplicates, calibration discovery, the worker's packed and stepped jobs in dry-run mode.
  - Dry runs of every stage pass on trig-login01, for Trillium's groups and for Rorqual's (`R7B_CLUSTER=rorqual`).
  - The CPU driver smokes (job 2544236) ran before this change. The worker's new modes are tested dry; the first GPU
    jobs exercise them for real.

## The review before the freeze (2026-10-12; the user asked for recommendations, and fixes for what is missing)
**Recommendations** (`plan_r7b.md` §8, with the evidence):
1. Sizes: raise Qwen 32K multivalue from 150 to 300 units (+9 GPU-h). Keep the rest. Add `R7B_EXTRA` only if Qwen
   128K's accuracy at δ = 0.02 matters.
2. Re-ranking: reuse R6's items 11–40, flagged as R6's units.
3. Dense 7/6: a test in every group (applied later the same day, with the other decisions).
4. E1: every group; the KL-budget rule; two traffic-matched levels.
5. Loose layers: 4+4 keys in Qwen's 14.
6. TurboQuant's product variant: leave it out.
7. Budget: about 70 GPU-h, plus 8.5 for `R7B_EXTRA` and 9 for decision 1 (since then: 76–85 billed, as packed).

**Fixed** (all in `r7b/`; `test_run_r7b.py --fast` passes):
1. **The code manifest no longer holds the calibration outputs** (`data/tau_*`, `alloc_*`). They appear mid-run: with
   them in it, core blocks that start after stage 1 would carry a different manifest from those before, and the reader
   would flag them. Each exploratory block's sidecar records them (`r7b.data_sha256`), and the exploratory reader checks
   that they agree per group.
2. **The preflight's queued-job check is per stage**, so the core, both calibration stages and the exploratory stage
   can run in parallel.
3. **`R7B_EXTRA` is 100 units (was 80).** It adds ICL items 91–100: about 170 FP-correct Qwen 128K units are expected,
   where 80 gave about 150, enough only half the time. The Qwen 128K exploratory items moved to 105–110.
4. **E1's loose layers have a default** (`lib_r7b.LOOSE_DEFAULT`): Qwen3-30B-A3B's 14 at 32K; none for Llama-3.1-8B,
   which R7a's probe pilots confirm at 128K. The planned source, R7a's probe blocks, was cancelled.
5. **An exploratory gate.** Stage 2 projects each exploratory block (E1 per answer token × the 90th-percentile answer)
   and checks the peak memory. Now each exploratory job checks it first (`stage2 --strict`), and the `explore` stage
   refuses once it has failed.
6. **The worker's preflight runs `test_r7b.py --quick`** (13 s, not 75 s).
7. **The core gate projects per task** (`read_r7b.gate_projection`, `GATE_UNIT_S` from R6's measured times). The
   per-group factor under-projected the multivalue-only blocks by about 30%.
8. **Walls.**
   - `qx-r1` and stage 2's `qc2` now have 6 h. At E1's upper rate, E1 on Qwen 32K multivalue's 164-token answers needs
     3.5 h.
   - The core's `q-mv` blocks now have 4 h (and Llama's 3 h blocks too). At 2 h, a pilot 30% slower than R6's times
     would have failed the gate for the whole core. Now each gate passes a pilot up to 2.4–3.2× R6's mean time for its
     task: one pilot unit's answer length varies that much.
9. **The cost, measured** (`explore_r7b.py cost`, `findings/cost_r7b.md`): the core is 45 GPU-h, not 22–26. The
   earlier estimates used R6's median unit time and too few arms.
10. **Power by level** (`explore_r7b.py power-mv`, `findings/power_mv_r7b.md`), behind decision 1.
11. **The pin no longer holds the calibration outputs**, and the plan no longer re-pins after calibration. With
    `data/*.json` in the pin, a re-pin between two groups' stage 2 would have stopped every core block that started
    afterwards. Instead, each block hashes the calibration outputs at its start and end (`r7b.data_sha256`,
    `data_changed`), and the exploratory reader excludes a block whose outputs changed while it ran. `--list-pin` prints
    the pinned files.
12. **New tests**: `test_run_r7b.test_freeze_guards` for 1, 3, 4, 5, 7 and 11 (the pin covers every file of the code
    manifest); the exploratory smoke checks the data hashes and runs the exploratory reader.

## Done before the review
- **Theory, port and tests:**
  - `theory_r7b.md`;
  - `store_r7b.py` and `cert_r7b.py`: the single store bit for bit with `../../14h_theory/nested_s1h5`; R7a's part 8 and
    certified read re-bound to per-coordinate steps; Theorem M by Lemma P′;
  - `test_r7b.py --fast` passes.
- **CPU job 2542204** (32K, every layer of R6's two models; `findings/depth_r7b.md`):
  - the single store's KL proxy is 1.307× R6's design on Llama-3.1-8B and 1.235× on Qwen3-30B-A3B;
  - E1's loose layers are 14 of Qwen3-30B-A3B's 48 and none of Llama-3.1-8B's (`../../14h_theory/findings/findings_nested.md`
    §7);
  - in `theory_r7b.md` §5.4b and §7.
- **The run side** (`plan_r7b.md` §11; `CLAUDE.md` lists the files):
  - presets, layouts, per-unit seeds; the arms; the driver; R7b's HELMET manifest (CPU job 2543842, pinned);
  - the reader and gate; the calibration stages; the worker and chain; the tests.
- **Checks:**
  - 979 planned units, all fresh except re-ranking;
  - the single store and E4 share value coins bit for bit;
  - the streamed debias is within 1.2·10⁻⁷ of the float64 reference;
  - the masked attention is within 1.2·10⁻¹⁰ of a direct computation;
  - the run-time store's key error at 1–4 residual bits follows the step model;
  - `cert_layer` equals a direct certified read exactly;
  - the Holm gates behave as specified.
- **CPU driver smokes**, Llama-3.2-1B at 4K (Trillium debug jobs):

  | job | what ran | result |
  |---|---|---|
  | 2543867 | the core plan on RULER, and the reader | every arm ran; KL/token: single store 1.9e-4, E4 1.0e-4, dense 7/6 1.0e-4, FP8 7.3e-4, Quest 7.0e-3, uniform 5.7e-3, uniform+v4 8.9e-3, KIVI-4 1.3e-2, fp_noise 0 |
  | 2543872 | calibration stages 1 and 2 with `calib_r7b`, then the exploratory plan | all ran |
  | 2543899 | the core plan on HELMET (suite r4, `kilt_nq`, fresh item 41), and the reader; calibration and exploratory again | every arm ran; floor 0.4019 against the loosest level's 0.4018; t1 at B = 0.0039 (bracketed); t2 at the scan's end, flagged |
  | 2543964 | the exploratory plan on 8 units of `niah_multivalue`, each with its own seed | E2 / design: geometric mean 0.84, ratio of means 1.11, per unit 0.14–2.6×; one draw per unit is too noisy to resolve E2's expected 0.6–0.8×. E3 0.35× its control; E1 at t1 / t2 0.026× / 0.018× the design |

## Changed from the spec while building
- **E1's levels are measured from E1's own traffic floor**, not from the design's 0.398. The floor is the design plus
  the loose layers' two extra key bits, about +0.016 on Qwen3-30B-A3B at 32K. Measured from 0.398, both of Qwen's
  targets would fall below its floor and collapse to one level. A target outside the scan is flagged
  (`bracketed: false`), and gets a new level before the exploratory blocks.
- **Re-ranking cannot be fresh.** msmarco has 41 queries and R4–R6 ran all of them (decision 2).

## R7a, as R7b's evidence
On trig, R7a's pilots 1080289 and 1080290 completed and its gate 1080291 passed (08:52). Its 13 blocks 1080292–1080304
and reader 1080309 were cancelled while pending (09:17–09:23); R7a's `STATUS.md` does not record why. Probe pilots also
exist under 8-digit job IDs (`results/r14r7a_*_23020651`, `_23020652`). R7b uses the Llama 128K pilot for the loose
layers (part 8 at 3.2–6.3× the error in every layer). Nothing in R7b waits on R7a.

## Next
- Rorqual (2026-10-17): `rorqual_r7b.sh submit`, run on Rorqual (`plan_r7b.md` §9). It submits Qwen 128K's half
  and the 32 blocks still pending on Trillium. The Llama and Qwen 32K exploratory blocks rerun their calibration there
  first; no pilot is rerun, and nothing is copied or cancelled on Trillium.
  - Tested here in simulation: `check`, and `submit --dry` both with Trillium's results present and with none (the
    calibration chain then goes out, with `afterok` waits).
  - The earlier mirror version (copy lists, race tracking) was dropped at the user's request.
  - Duplicate review (2026-10-17, the user asked). Fixed:
    - `rest` no longer reruns the six calibration blocks that finished on Trillium. The exploratory blocks need Trillium's
      calibration outputs copied (10 files, added to `findings/R7b_rorqual_copy.txt`).
    - A failed `squeue` now stops the script. It had passed as an empty queue: the planning ran inside `$( ... )`, where
      bash ignores `set -e` (now `inherit_errexit`, plus explicit checks; `script_r7b.sh`'s queue read stops too).
    - A failed ledger write after `sbatch` stops loudly.
    - `--skip-trillium`, and a confirmation that Trillium's 13 jobs are still pending.
  - Tested on a simulated Rorqual root, with fake `squeue` / `sbatch` (Trillium's exported `sbatch` function removed):
    a fresh submit gives 27 core jobs and refuses the exploratory blocks until the calibration files are present; skips,
    finished and queued blocks are left out; the same submit run twice adds nothing; Qwen 128K's finished pilot is not
    resubmitted. One test first reached Trillium's real `sbatch` (an exported function shadows `PATH`); both calls were
    refused, and no job was created on either Trillium side (checked).
  - On Rorqual the R7b files are in `14i_r7b_theory_at_scale` (user, 2026-10-17). The script links the pinned path to
    them. Tested on three simulated layouts: R7b in `14i_r7b_theory_at_scale/r7b/`; R7b's files directly in
    `14i_r7b_theory_at_scale/` with R7a's modules; a stray `.py` in the pinned folder (refused). In the first two the
    start-up tests pass, and the code manifest equals Trillium's blocks' (f62dbf10…).
1. Watch: the pilots' gates (`findings/R7b_gate_<group>.json`), then the core; `bash script_r7b.sh --status` from
   trig-login01.
2. Rorqual's half when the user asks: copy `--list-sync`'s files, then the stages there (`plan_r7b.md` §9).
3. When every block is in: copy Rorqual's `--list-results` here; `read-core`, `read-explore`.
