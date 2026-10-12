# R7b plan: the deployed design (the single store), confirmatory (2026-10-12; frozen, Trillium submitted)

The question, the theory and the predictions are in `theory_r7b.md`; the spec is `../../14h_theory/RESEARCH_PLAN.md` §4
"R7b" and §8. This file is the run. The user took §8's decisions as recommended (2026-10-12). FROZEN 2026-10-12
(`AMEND_R7B = "R7b"`, `findings/R7b_code.sha256`); Trillium's stages submitted the same day (`STATUS.md`), Rorqual's
not yet. R7a is frozen; nothing here touches its files.

## 1. What R7b decides
- **Primary, confirmatory (D7).** In each claim group (Llama-3.1-8B at 128K, Qwen3-30B-A3B at 32K, Qwen3-30B-A3B at
  128K), the single store `dnest42ux_v4` is non-inferior to FP8 KV on KL per answer token, under R6's rules. The single
  store is the design R6 tested (`dtail6ux_v4`) with its separate vote store removed: one store of 4+2-bit keys and
  4-bit values serves both the vote and the tail. That makes the memory claim confirmatory: 0.32 of FP16 K+V at rest and
  0.45 during a question, against FP8's 0.50 (R6's design: 0.58 / 0.70), at R6's traffic, 0.398.
- **Key secondary:** accuracy, under R6's rule.
- **Gates (D3):** paired one-sided tests against baselines at no more than the design's traffic, Holm-corrected.
- **Exploratory, reported but not tested:** E1 the certified read with a run-time τ; E2 the 4+3 store; E3 per-layer
  water-filling; E4 R6's design, paired in every unit.

## 2. Design

### 2.1 The arms of every unit (one process; `lib_r7b.build_plan`, presets `r7b<group>`)

| arm | what | traffic | role |
|---|---|---|---|
| `fp` | FP16 KV | 1 | reference |
| `dnest42ux_v4`@1/8 | **the single store** (below) | 0.398 | primary |
| `dtail6ux_v4`@1/8 | R6's design: uniform dithered 6/4 tier, its own Lloyd-Max 4/4 vote store | 0.398 | E4 (paired) |
| `ddense7u_v6`@7 | every context row from a dithered 7/6 tier, unbiased; nothing exact | 0.406 | gate: "two tiers beat one" |
| `uniform`@4, `uniform+v4`@4 | dense Lloyd-Max 4-bit keys (FP16 values), then 4/4 (TurboQuant-MSE); the twin needs its base | 0.625, 0.25 | gate (4/4); base reported |
| `kivi4_v4`@4 | KIVI-4 | 0.25 | gate |
| `quest_v16`@0.3359375 | Quest at the design's traffic (2/32 of metadata + B) | 0.398 | gate |
| `fp8kv`@8 | FP8 E4M3 K and V | 0.5 | the comparator |
| `fp_noise` | FP rerun | 1 | calibration (R6's) |

**The single store** (`arms_r7b`; `store_r7b`, a bit-for-bit copy of `../../14h_theory/nested_s1h5`).
- **Keys:** a 4-bit Lloyd-Max base (`quant.quantize_keys` in the tier's rotation, seed 50) plus a 2-bit
  subtractive-dithered residual inside the base cell, the outer cells closed at ±max.
- **Values:** R6's 4-bit dithered tier (rotation 150).
- **The vote:** R6's question pass and SnapKV vote (`run_s1e.select_all_w`), run over the store itself: the base keys and
  the tier's own dithered values.
- **The read:** the vote's 1/8 exact (keys and values); every other context row from the store.
- **Unbiased:** the per-coordinate debias, ldb = ½(lmu_lo + lmu_hi) of the per-coordinate coin moments, is subtracted from
  every unread row's score as an additive mask. It is streamed over rows from the stored base index, row maximum and norm.
- R6's own steps are reused unchanged: `run_s1h5._tail5_steps` and `_tf_tail5`, with `_tier1` swapped for the store's
  vote view during the arm.

**Seeds** (`theory_r7b.md` §6.1).
- Each unit draws its own seed, `lib_r7b.unit_seed(task, prompt)`, with R6's per-layer offsets: + 2·layer for keys,
  + 2·layer + 1 for values.
- The unit bootstrap then covers dither draws (C1's estimand).
- The single store and E4 share their value coins (common random numbers; checked bit for bit in
  `test_run_r7b.test_store_arm`). Their paired difference measures the key quantizer and the vote alone.

### 2.2 What is not built
- TurboQuant's product variant (a 1-bit QJL residual on keys; optional in the spec).

## 3. Units (`lib_r7b.R7B_LAYOUT`)

| group | family | units | blocks (prompts) |
|---|---|---|---|
| Llama 128K | retrieval / multivalue / tracking | 60 / 30 / 30 | `l-r1a-c` (standard, 9800–9814), `l-r3aa-c` (R6's R3a level, 9815–9829) |
| | aggregation | 30 | `l-agga-c` (cwe, fwe at R6's R3b level, 9830–9844) |
| | rag / icl | 60 / 60 | `l-hma-f`: HELMET items 41–70 of the RAG and ICL tasks |
| | rerank | 30 | `l-rra-c`: msmarco items 11–40 (**R6's**, see below) |
| Qwen 32K | retrieval / multivalue / tracking | 80 / 40 / 40 | `q-r1a-d` (9850–9869), `q-r3aa-d` (9870–9889) |
| | multivalue (extension) | +260 | `q-mva-j` (niah_multivalue, 10100–10359, 26 per block): the binding family, 300 units, 280 at the standard level (§8 decision 1) |
| | aggregation | 30 | `q-agga-c` (9890–9904) |
| Qwen 128K | rag / icl | 60 / 60 | `w-hma-c`: items 41–70, each row two 2-GPU blocks |
| | rerank | 30 | `w-rr`: items 11–40 (R6's) |
| Qwen 128K (optional) | rag / icl | +40 / +60 | `w-hmxa-b`: items 71–90 of the RAG and ICL tasks; `w-hmxc`: ICL items 91–100 (the accuracy secondary's size; stage `extra`; about 170 FP-correct units expected with the core) |

- **Fresh units.** No RULER (task, prompt) appears in any earlier sidecar under `results/` or in R7a's planned blocks
  (`test_run_r7b.test_layout`; 1,131 planned units checked, pilots included).
- **HELMET.** R6's manifest had 41 items per task, all run. R7b's manifest (`data/manifest_r7b.json`,
  `make_manifest_r7b.py`, built 2026-10-12) extends it in R6's own order to 111 items for the RAG and ICL tasks; items
  41–110 are fresh. Its first 41 items render exactly as R6's manifest recorded, for both models (checked by the builder).
  The two 128K models share items, as in R6.
- **Re-ranking cannot be fresh.** msmarco's test file has 41 distinct queries, and R4–R6 ran all of them. R7b's
  re-ranking units are R6's items 11–40 (the record file is R6's, byte for byte). The reader marks the family "R6 units".
  This is decision 2 (§8).

## 4. Pilots and gates (one per claim group, as R6's Qwen 128K had its own)
- **The pilots** (`lib_r7b.R7B_PILOTS`): one unit of the group's core plan each. Llama 128K: niah_multivalue, prompt
  9999. Qwen 32K: niah_multivalue, prompt 9998. Qwen 128K: kilt_nq, item 101 (2 GPUs).
- **The pilot's own job runs the gate** after the unit (`submit_r7b.slurm R7B_POST=gate`), so the gate costs no queue
  wait. It writes `findings/R7b_gate_<group>.json` (with the job and the code pin's sha256), and the job's exit code is
  the verdict.
- **`read_r7b.py --gate` checks:**
  - validity in pilot mode;
  - FP's KL 0 and `fp_noise` not degenerate;
  - peak ≤ 76 GiB per GPU;
  - every core and extra block of the group projected within 90% of its wall (`read_r7b.gate_projection`). Per task,
    the block's units × the pilot's seconds per unit × the task's ratio to the pilot's task in R6's measured times with
    R7b's arms (`GATE_UNIT_S`, from `explore_r7b.py cost`), × 1.1, plus 15 minutes. A task's answer length sets its time,
    which varies 5× across tasks: Qwen 32K multivalue's answers average 164 tokens.
- A group's core and extra blocks wait on its pilot's job (`afterok`): they start only after a PASS.

## 5. Calibration and exploratory blocks
They run on separate prompts, never test units, and freeze their outputs before the exploratory blocks. The two CPU
steps run inside the next stage's jobs, so the chain needs no hand step and no extra queue wait. Each stage-2 job first
runs `calib_r7b.py stage1 --blocks auto`, and each exploratory job first runs `stage2 --blocks auto --strict`
(`submit_r7b.slurm R7B_PRE`). Both steps skip work already done from the same blocks, and write atomically.
1. **Stage 1** (`r7bc1<group>`, `lib_r7b.R7B_CALIB_LAYOUT`): 8 calibration units per group across families. Per unit:
   the design, then the single-layer arms `kcal_{tail,key,val}_l{li}`. In each, layer li alone reads the store's tail
   (keys and values / keys only / values only), every other layer is exact, the question is FP's, and the arm records
   layer li's realized output error.
2. **`calib_r7b.py stage1`:**
   - c_l = 2·KL / err², pooled over the units, gives E1's per-layer τ_l = √(2B/(L·H·c_l)) at the scan levels
     B = 10 … 10⁻³;
   - E3's allocation: the per-layer Lagrangian choice over residual bits r ∈ {1..4} and value bits v ∈ {3..6}, from the
     single-layer key-only and value-only KLs scaled by the step ratios, with inputs weighted equally, at the uniform
     4+1/4 control's traffic, 0.371;
   - E1's loose layers (4+4 keys): from R7a's probe blocks of the same group (`--loose-from-r7a`), the layers whose
     median part-8 bound / error at 6-bit keys exceeds 14 (about 20× at 4+2); or `--loose`; or else
     `lib_r7b.LOOSE_DEFAULT`, the measured lists. At 32K on CPU that is 14 of Qwen3-30B-A3B's 48 layers
     (`../../14h_theory/findings/findings_nested.md` §7), used for Qwen 128K too. Llama-3.1-8B has none, confirmed at
     128K on the GPU by R7a's probe pilots (part 8 at 3.2–6.3× the error in every layer). `data/tau_<group>.json`
     records the source (`loose_source`).
3. **Stage 2** (`r7bc2<group>`): E1 at the five scan levels on the same units. **`calib_r7b.py stage2`** sets the
   levels t1 and t2 to the B whose mean traffic is E1's floor + 0.002 and + 0.01.
   - The floor is E1's traffic with only the vote's rows read: the design's 0.398 plus the loose layers' two extra key
     bits, about +0.016 on Qwen3-30B-A3B at 32K. Measured from 0.398, both of Qwen's targets would sit below its floor
     and collapse to one level.
   - A target outside the scan takes the nearest end and is flagged (`bracketed: false` in `data/tau_<group>.json`). Add
     a level and rerun stage 2 before the exploratory blocks.
   - **The exploratory gate** (`calib_r7b.explore_projection`, written to `data/tau_<group>.json` as `explore_gate`):
     stage 2's peak GPU memory (≤ 76 GiB), and each exploratory block's projected hours (≤ 90% of its wall). E1 is
     projected per answer token, from the costliest level's measured seconds per token times the 90th percentile of the
     calibration answers' length, because E1's cost grows with the answer and the exploratory blocks hold longer
     answers (cwe) than some calibration units.
   - An exploratory job stops before its block unless both levels are inside the scan and the exploratory gate passes
     (`--strict`). The `explore` stage refuses to submit once stage 2 has been computed and fails either check.
4. **Exploratory blocks** (`r7bx<group>`, `R7B_EXPLORE_LAYOUT`): 104 units (Llama 50, Qwen 32K 30, Qwen 128K 24; its
   HELMET items are 105–110, so that `R7B_EXTRA` can use ICL items 91–100). The walls hold `explore_r7b.py cost`'s
   projection at E1's upper rate. `qx-r1` and stage 2's `qc2` have 6 h because E1 runs every answer step.
   Per unit: fp, the design, E2 `dnest43ux_v4`, E3 `wnest_v4` and its control `dnest41ux_v4`, E1 `cnest42ux_v4_t1`
   and `_t2`, FP8, `fp_noise`. The reader is `read_r7b.py --explore`.

E1 is teacher-forced only. In FP's answer pass, every layer's attention is replaced by the certified read:
- per KV head and answer step, R7a's `certified_read` with part 8 at the store's per-coordinate steps (`cert_r7b`) over
  the FINE schedule;
- it stops at the first set whose bound meets τ_l for every query head of the group, else reads exactly;
- it records rows read, fallbacks and query heads met. The guarantee is per query head: misses ≤ δ.

## 6. Reader (`read_r7b.py`; rules in its docstring)
- **Primary test, calibration, group verdict, key secondary:** R6's, copied with the design replaced.
- **Gates:** per group and baseline, d_u = KL(design) − KL(baseline) over all units. The one-sided bootstrap p-value
  goes through Holm's step-down at 0.05: BEATS or NOT_SHOWN.
  - Tested everywhere: KIVI-4, dense Lloyd-Max 4/4, Quest at 0.398.
  - Dense dithered 7/6 is tested in every group as well (§8 decision 3; `theory_r7b.md` §6.3, §5.4b).
- **Reported:** E4's KL ratio per family (ratio of means with a 95% interval; geometric mean of per-unit ratios); the
  primary statistic for every other arm; accuracy for every arm; the arm table.
- **Validity:**
  - layout rows, plans, every arm on every unit, R7b's units;
  - tier arms evict nothing, and every dithered arm uses its unit's seed;
  - peak ≤ 76 GiB; R7b's HELMET manifest; identical code manifests;
  - exploratory blocks: the calibration outputs unchanged while the block ran, and one version per group.
- **The readers run once**, on the cluster that holds every group's results (`script_r7b.sh read-core`,
  `read-explore`, §9).

## 7. Predictions (`theory_r7b.md` §7, stated before the run)
- **P-b1:** NONINFERIOR in Llama 128K and Qwen 128K. Qwen 32K NONINFERIOR with probability 0.98–1.00 at R7b's 300
  multivalue units (ρ = 1.3; 0.84–0.95 at the draft's 150; decision 1).
- **P-b2 (E4):** KL(single store) / KL(R6's design) is 1.1–1.4 per unit in geometric mean, and 0.9–1.2 on mean KL. CPU
  measured 1.12–1.27 (8 seeds); the 32K KL proxy measured 1.24–1.31.
- **P-b3:** accuracy ACC_PASS in Llama 128K and Qwen 32K; in Qwen 128K ACC_UNDERPOWERED without `R7B_EXTRA` (about 90
  FP-correct units), testable with it (about 170).
- **P-b4 (gates):** KIVI-4, dense 4/4, Quest and dense 7/6 BEATS in every group (dense 7/6 over every layer at 32K:
  10.7× the design's MSE on Llama, 136× on Qwen; least sure on Llama 128K).
- **P-b7 (E1):** traffic ≤ 0.41 on Llama 128K with KL 4–20× below the design's; misses within δ per query head.
- **P-b8 (E2):** 0.4–0.8× the design's KL.
- **P-b9 (E3):** at most about 20% below the uniform control at equal traffic.

## 8. Decisions (reviewed 2026-10-12; the user took every recommendation the same day)
1. **Sizes. Decided: Qwen 32K multivalue raised from 150 to 300 units (+9 GPU-h); the other sizes as R6's. `R7B_EXTRA`
   stays optional: the `extra` stage, submitted only if Qwen 128K's accuracy at δ = 0.02 matters.**
   - Llama 128K and Qwen 128K: P(NONINFERIOR) ≈ 1.00 at R6's sizes (`findings/power_r7b.md`), even if the single store
     had twice R6's design's excess KL.
   - Qwen 32K is decided by multivalue. There FP8's KL is tiny, so the margin is κ alone. One of R6's 40 units (prompt
     9638, at R6's hard level) had design KL 0.0125 from its dither draw (R6's 4-bit arm: 0.0003).
   - R7b draws a seed per unit, so such draws can occur at either level. The family's P(pass) at ρ = 1.14 / 1.3 / 1.5
     (the CPU's median; the 32K KL proxy; pessimistic), from `findings/power_mv_r7b.md`:

     | multivalue units | such draws at R6's overall rate (1 in 40) | only at R6's hard level |
     |---|---|---|
     | 150 (the draft) | 0.93 / 0.84 / 0.69 | 0.98 / 0.95 / 0.94 |
     | 300 | 1.00 / 0.98 / 0.89 | 1.00 / 1.00 / 1.00 |

   - A miss would be a real FAIL: `fp_noise` passes there, so the family is testable. R7a's seed panel on Qwen 32K
     multivalue (units 9600–9609) would measure how often such draws occur. It has no results yet: its Trillium block
     was cancelled.
   - Applied: `q-mv` as 10 blocks of 26 prompts (10100–10359; fresh, checked; 4 h walls, the gate projects 1.7 h).
   - `R7B_EXTRA` is 100 Qwen 128K units: RAG and ICL items 71–90, and ICL items 91–100 (+8.5 GPU-h).
     - The accuracy secondary needs 150 FP-correct units outside aggregation. R6's FP-correct rates there are ICL 0.98,
       RAG 0.52, re-ranking 0.
     - The core gives about 90. The 80-unit extension first drafted gave about 150, enough only half the time. The 100
       units give about 170.
     - Without it, Qwen 128K's accuracy is ACC_UNDERPOWERED at δ = 0.02 and reported at 0.05, as in R6.
2. **Re-ranking. Decided: reuse R6's items 11–40, flagged "R6 units" in the reader (done) and the paper.**
   - msmarco has no fresh query: its 41 were all run by R4–R6.
   - Re-ranking is the design's weakest family. Dropping it would look like selection.
   - The single store's parameters come from the theory and R6's 6-bit tier, not from these units.
   - Each unit gets a fresh dither seed, and E4 replicates R6's design on the same units in the same process.
   - A new re-ranking set from MS MARCO's dev queries would be a new data pipeline, and still not comparable with R6.
3. **Dense 7/6 gate. Decided: a test in every group** (the draft tested it on Llama 128K only).
   - Over every layer at 32K, dense 7/6 has 10.7× (Llama) and 136× (Qwen) the design's MSE (`theory_r7b.md` §5.4b).
     Qwen's flat layers carry under 0.3% of the design's error.
   - Holm costs the other gates nothing: their p ≈ 1/2,001 is below α/4.
   - The least certain group is Llama 128K, which the draft tests. There 81% of the design's MSE sits in layers 8 and 5,
     where dense 7/6 has 0.9× and 0.4× the design's MSE. A NOT_SHOWN there would mean those layers are the sensitive
     ones.
   - Applied: `read_r7b.GATE_DENSE_GROUPS` holds all three groups; `test_run_r7b.test_gates` checks it.
4. **E1's scope and rule. Decided: E1 in every group; the KL-budget rule with stage 1's sensitivities; two
   traffic-matched levels, no KL target claimed** (as built).
   - Skipping Qwen 128K's E1 would save about 2 GPU-h (§10). Its exploratory units are HELMET RAG and ICL, with answers
     of 3–20 tokens.
   - The rule is 14h_theory's final recommendation (`../../14h_theory/findings/findings_klbridge.md` §6.5). A flat τ
     is 1.35–1.4× worse on Qwen3-1.7B. The FP8 quantile is 1.6–1.8× worse in shape on Qwen, and is the fallback only
     if stage 1 is cut.
5. **E1's loose layers. Decided: 4+4 keys in Qwen3-30B-A3B's 14 loose layers** (the default,
   `lib_r7b.LOOSE_DEFAULT`).
   - Without them, E1 on Qwen mostly measures a vacuous certificate: 0.27–1.0 of the rows read in those layers at 32K.
   - Llama needs none: part 8 is at 3.2–6.3× the error in every layer at 128K (R7a's pilots).
   - Qwen 128K uses the 32K list. Nothing measured it at 128K: R7a's Qwen 128K probe block was cancelled. A layer that is
     loose at 128K but not on the list makes E1 fall back to exact reads there. That costs traffic, which is reported;
     the guarantee still holds.
6. **TurboQuant's product variant. Decided: left out, and discussed in related work.**
   - It is a dense quantizer at matched traffic, so Corollary G covers it as it covers dense 7/6, which R7b tests.
   - It is unbiased in scores, not in attention weights (E e^{ŝ} ≠ e^{s}). It is not the unbiased comparator it seems.
   - Building QJL with its tests is new run-time code just before a freeze.
7. **The GPU budget. Decided: about 76–85 GPU-h billed, 84–93 with the `extra` stage** (§10: the submission plan as
   packed, from R6's measured times; the range is E1's cost).
   - The confirmatory part (pilots and core, with the 300 multivalue units) is about 60 billed. The calibration and
     exploratory part is 15–24.
   - The earlier estimates were low: 22 for the core (the draft), then 26 (this review's first pass). Both used R6's
     median unit time and too few arms. R6's mean time per unit is 1.2–1.5× its median, and R7b's core runs ten arms
     (R6's blocks: eight or nine).
   - Packing Trillium's core into whole-node jobs adds about 5 GPU-h (§9, §10), for shorter queue waits.
   - If the budget is capped: run the pilots and the core first. The exploratory part needs nothing from the core, and
     can follow as its own allocation.
   - The pilot measures the core's time before any block starts. Stage 2's exploratory gate measures E1 before the
     exploratory blocks.

## 9. Submitting by stage, on two clusters (`script_r7b.sh`, planner `jobs_r7b.py`)

**Where each claim group runs** (`R7B_ASSIGN`, the default in `script_r7b.sh`; keep it the same on both clusters). Every
block of a group runs on one cluster, as in R6, so no family's test pools units across clusters.
- **Trillium** (submit from trig-login01): Llama 128K and Qwen 32K, 1-GPU blocks.
  - Trillium gives a GPU job 1 GPU or a whole node of 4. On 2026-10-12, whole nodes were estimated to start in 14
    minutes, single GPUs in 2 h 20 min (`sbatch --test-only`).
  - So the pilot, core and extra stages pack a group's blocks four to a node, the longest first. A pilot takes a node
    alone: it is short, and the core waits on it.
  - The calibration and exploratory stages stay 1-GPU jobs: their 1–3 uneven blocks per group would leave a node
    mostly idle.
- **Rorqual**: Qwen 128K, 2-GPU blocks as 2-GPU jobs (Trillium would need whole nodes). The model is staged to node-local
  disk (`R7B_STAGE=1`, R6's fix for Rorqual's slow shared cache), with `--mem=96G` per block. This is R6's split.
- **Rebalancing.** Moving Qwen 32K to Rorqual gives Trillium about 27 GPU-h and Rorqual about 55, against about 60 and
  24 now (§10).
  - To move it, change `R7B_ASSIGN` on both clusters before that group's first submission.
  - `--probe-queue` prints each cluster's start estimates for these job shapes.

**No duplicated runs** (`jobs_r7b.plan`; `test_run_r7b.test_stages`):
- A cluster submits only its own groups.
- A block is submitted only if it is neither finished nor queued:
  - "finished" means its results are in `results/`, from any cluster once copied;
  - "queued" means it is in this cluster's ledger `findings/R7b_jobs_<cluster>.tsv` with its job in `squeue`.
  So resubmitting a stage fills only the gaps.
- A group that has jobs in another cluster's ledger is refused. The ledgers travel with the code and the results.
  `R7B_MOVE=1` overrides, only after cancelling that group's jobs there and copying its finished results here.

**The stages and what each waits on.** A stage may be submitted as soon as the one it follows is queued: its jobs wait on
that stage's jobs (`afterok`). Each stage is one command, `bash script_r7b.sh STAGE` (add `--dry` to print the jobs).

| stage | its jobs | wait on | before their blocks run |
|---|---|---|---|
| `pilot` | one unit of the group's core plan, then the gate | — | — |
| `core` | the group's core blocks | the group's pilot (its gate must PASS) | — |
| `extra` (Qwen 128K, optional) | `R7B_EXTRA_LAYOUT` | the pilot | — |
| `calib1` | calibration stage 1 | — | — |
| `calib2` | calibration stage 2 | `calib1` | `calib_r7b.py stage1` |
| `explore` | the exploratory blocks | `calib2` | `calib_r7b.py stage2 --strict` (stops the job unless ready) |
| `read-core`, `read-explore` | the readers, on the login node | every block of every group in `results/` here | — |

**The order:**
0. **Freeze**, once, on Trillium: `bash script_r7b.sh --freeze`. This sets `AMEND_R7B = "R7b"`, runs the tests and pins
   the code.
   - Then copy the code to Rorqual: `bash script_r7b.sh --list-sync > r7b_sync.txt`, then
     `rsync -a --files-from=r7b_sync.txt ./ <rorqual>:<project root>/` from the project root.
   - Rorqual's preflight verifies the pin.
1. **On each cluster, submit `pilot` and `calib1`.** They are independent.
2. **Right after, submit `core` and `calib2`.** Their jobs wait on step 1's jobs.
3. **Right after, submit `explore`;** on Rorqual also `extra`, if chosen.
4. **Check progress** with `--status` on each cluster (on Trillium, from trig-login01).
5. **When Rorqual's blocks are done**, copy them to Trillium: `bash script_r7b.sh --list-results` on Rorqual lists the
   result directories, the gate and calibration files, its ledger and the logs.
6. **On Trillium, run `read-core`, then `read-explore`.** Each refuses while any block is missing (`R7B_PARTIAL=1` reads
   what is there; add `R7B_EXTRA=1` if the extra stage ran).

**If something fails:**
- **A gate FAIL** (`findings/R7b_gate_<group>.json`) leaves that group's core and extra jobs waiting forever.
  `--cancel-orphans` cancels them.
- **A calibration step that stops** (a level outside the scan, or the exploratory gate) does the same to the exploratory
  jobs.
- **A failed block:** resubmit its stage. Only the missing blocks go.

**What must not change once a job runs:**
- The pin: R7b's code, the parts of R6 and R7a it imports, the worker, the HELMET data. `--pin-code` refuses once a ledger
  has jobs.
- The calibration outputs are not pinned, because they are written mid-run. Each block records their hashes at its start
  and end.

**R7a.** R7a's Trillium blocks were cancelled after its gate passed (2026-10-12). Its probe pilots are R7b's 128K
evidence for Llama's loose layers: Trillium 1080289, and 23020651, an 8-digit ID (presumably Rorqual). Nothing else in
R7b waits on R7a.

## 10. Cost (`explore_r7b.py cost`, `findings/cost_r7b.md`: R6's measured per-arm, per-task times)
How the times are modelled:
- Per unit: R6's prefill, then each arm's measured time on the same group and task. The single store, E4, dense 7/6,
  E2 and E3 are timed as R6's `dtail6ux_v4`, and Quest as KIVI-4.
- Per block: × 1.1, plus setup. R6's blocks spent 1.1× their summed unit times, and 3–4 minutes (Llama) or 6–11
  minutes (Qwen) of setup (sacct).
- E1 runs at 0.02–0.075 s per layer per answer token. R7a's pilot read 0.05 s per layer-step at Llama 128K, × 1.5 for
  the single store's looser bound gives the upper rate.
- **Billing.** Each job is billed its GPUs until its longest block ends. That applies to Trillium's whole-node packs
  and to a pilot's node.

The submission plan (§9: the default `R7B_ASSIGN` and packing), GPU-h billed at E1's upper / lower rate:

| cluster | pilot | core | extra (optional) | calib1 | calib2 | explore | total |
|---|---|---|---|---|---|---|---|
| Trillium (Llama 128K, Qwen 32K) | 2 jobs, 1.4 | 11 jobs, 42.4 | — | 3 jobs, 1.9 | 3 jobs, 4.9 / 1.7 | 5 jobs, 9.9 / 5.0 | 60.5 / 52.4 |
| Rorqual (Qwen 128K) | 1 job, 0.5 | 8 jobs, 16.0 | 6 jobs, 8.5 | 2 jobs, 3.0 | 2 jobs, 1.6 / 1.2 | 2 jobs, 2.9 / 2.4 | 32.5 / 31.6 |

- **Total:** about 85 / 76 GPU-h without `extra`, and 93 / 84 with it. The readers run on a login node.
- **The core** is 53.6 GPU-h unpacked: Llama 15.8, Qwen 32K 21.9 with its 300 multivalue units, Qwen 128K 16.0.
  Trillium's whole-node packs bill 42.4 for its 37.7.
- **Where the time goes.** Seconds per core unit range from 43 (Qwen 32K, short answers) to 331 (Llama re-ranking).
  Each answer token costs every arm a decoding step. Qwen 32K multivalue's 164-token answers make it 177 s per unit.
- **E1 is the main uncertainty.**
  - Its upper rate assumes no saving from Qwen's fewer rows (32K) and KV heads (4).
  - At that rate, a 164-token Qwen 32K multivalue answer costs about 10 minutes per E1 arm. The walls hold it.

## 11. Files and commands
`lib_r7b.py` (presets, arms, layouts, pilots, seeds, `blocks_of`), `arms_r7b.py` (the arms), `run_r7b.py` (the driver),
`tasks_r7b.py` and `make_manifest_r7b.py` (HELMET), `read_r7b.py` (reader and gates), `calib_r7b.py` (calibration),
`jobs_r7b.py` (the planner: stages, packing, waits, ledgers, status), `submit_r7b.slurm` (the worker: one block, or
several packed; the calibration pre-step; the gate post-step) and `script_r7b.sh` (stages and utilities),
`test_run_r7b.py` (tests; `--fast` in the worker's preflight), with the theory's `store_r7b.py`, `cert_r7b.py`,
`test_r7b.py` (`--quick` in the preflight) and `explore_r7b.py` (`power`, `power-mv` and `cost` for §8 and §10).

```bash
cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14i_r7_theory_at_scale/r7b/test_run_r7b.py --fast
bash h0_measurement/bugs/14i_r7_theory_at_scale/r7b/script_r7b.sh core --dry       # prints the jobs; submits nothing
bash h0_measurement/bugs/14i_r7_theory_at_scale/r7b/script_r7b.sh --status
```
