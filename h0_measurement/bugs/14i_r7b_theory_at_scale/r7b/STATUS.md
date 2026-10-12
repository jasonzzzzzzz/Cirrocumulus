# 14i_r7_theory_at_scale/r7b status (rewritten, not appended)

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
1. Watch: the pilots' gates (`findings/R7b_gate_<group>.json`), then the core; `bash script_r7b.sh --status` from
   trig-login01.
2. Rorqual's half when the user asks: copy `--list-sync`'s files, then the stages there (`plan_r7b.md` §9).
3. When every block is in: copy Rorqual's `--list-results` here; `read-core`, `read-explore`.
