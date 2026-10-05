# R14h · Accuracy-first Stage 1h — plan

Written 2026-10-04, after R14 Stage 1g (`../14_kernel_tpot/report.md`, Parts I–L) and
the methodology review (`../../../weekly_reports/Oct4_next_steps.md`).

**Status:** M0 in progress. The R1 rules in §5 are frozen when `read_stage1h.py` is
written, which must happen before any R1 job has output. R2–R5 are outlined in §6 and
are frozen run by run, each before that run's output exists.

## 1. Question

Stage 1g found a design that looks near full precision (FP) at Llama 128K:
- a 4-bit store;
- question-time reads of 1/8 of the rows;
- exact keys fetched once per question;
- the question run a second time.

Two problems weaken that finding:
- **The measurement can't resolve it.** RULER accuracy is at its ceiling (FP 0.984,
  system 0.987), and the 0.10-nat margin is arbitrary. There is no noise floor and no
  FP8 reference.
- **The baselines and benchmarks are too easy.** The only dense baseline is
  TurboQuant, and there are four synthetic tasks.

Stage 1h has two jobs:
- make the accuracy measurement able to show a gain, or a loss, reliably;
- use it to improve the design's accuracy.

System speed is a bonus track (§8); no accuracy claim depends on it.

## 2. Run sequence

| run | what | cells | GPU-h (est.) |
|---|---|---|---:|
| M0 | Methodology code (this folder); no GPU | — | 0 |
| R1 | Calibration and bridge: noise floor, FP8 cost, same-memory 4-bit baselines, oracle selection, rotation seeds | Llama-3.1-8B 128K: 20 prompts (Stage 1g's 9100–9119) + key-confusion prompts × 3 seeds | 5–7 |
| R2 | Second model | Qwen3-30B-A3B: regression block × seeds, then 32K × 40 fresh prompts | 6–8 |
| R3 | Stress tasks | Llama 128K: many-distractor multikey, common/frequent-word extraction, NoLiMa | 8–10 |
| R4 | Real tasks | LongBench v2 (+ a HELMET or LongBench v1 subset) | 6–10 |
| R5 | Accuracy fixes for what R3/R4 break; read-floor sweep | chosen after R3/R4 | 8–10 |
| Conf. | Confirmatory campaign, design frozen | 100–120 prompts per cell, about 6 arms | later |

Every run R1–R5 is **exploratory**: its labels guide design choices and never become
paper claims. Only the confirmatory campaign produces claims.

## 3. Methodology changes (M0)

**Run-time items** (computed while the model runs; cannot be recovered afterwards):
1. **KL to FP.** Per-token KL(FP ‖ arm) of the next-token distributions under teacher
   forcing on FP's answer, aligned with the existing NLL columns. Stored as columns
   `kl_all`, `kl_span`, `kl_val`, `kl_span_max` and the per-token list `tf_kl`.
2. **Noise arm `fp_noise`.** FP again, with the context re-prefilled at half the
   chunk size: mathematically identical and numerically different. It is always the
   last arm of a prompt, because it replaces the cache.
3. **FP8 KV arm `fp8kv`.** Dense FP8 (E4M3) keys and values, one scale per layer and KV
   head. This is the format deployments treat as lossless.
4. **Same-memory 4-bit baselines `kivi4_v4`, `kvquant4_v4`.** The draft's per-channel
   key quantizers (KIVI with G = 128; KVQuant-style), at 4 bits, with the same 4-bit
   values as every V4 arm.
5. **Oracle selection `qoraclefp_v16`, `qoracle4_v4`.** Read the floor(r C) rows per KV
   head that FP itself attends to most while answering (summed over the answer's query
   rows and the KV group), over the exact store or the 4-bit store. It uses future
   information, so it is an upper bound on selection at that r.
6. **The exact-K+V system `qread2t4q_v4`.** Fetches values as well as keys. The arm
   already exists in Stage 1g's code but has not been run.
7. **Rotation seeds.** `--override rot_seed=N`, applied to the key-confusion prompts.
8. **Peak GPU memory per arm.** Columns `peak_gib_arm` and `base_gib_arm`. The
   per-prompt `peak_gib` keeps its meaning.

**Reader-side items** (`read_stage1h.py`, frozen before each run's output exists):
1. Span NLL **without** the best-of-two-orders minimum (`a_span_nll`) is the primary
   NLL. `s_set_nll` is reported beside it for continuity, together with the bias of
   the minimum (`s_set_nll − a_span_nll`) per arm.
2. KL over the span (`kl_span`) is co-primary.
3. Lost answers (units on which the arm scores below FP: Stages 1f–1g's "below FP")
   are tested with an exact McNemar test against FP and against D.
4. Equivalence tests (TOST, expressed as a 90% CI inside ±m) back "near FP" and
   "no effect" claims.
5. Each lost answer gets a failure type by rule: confused (states a distractor's
   value), incomplete (a strict subset of the expected values, or a prefix of one),
   or other.
6. Units FP fails are a separate stratum, not dropped.
7. The Holm correction applies to the confirmatory campaign's secondary comparisons
   only. Exploratory runs are unadjusted and labelled exploratory.
8. Sequential prompt blocks (§7).

## 4. Rules for every run

- **Metric freeze.** The primary metrics are `a_span_nll` and `kl_span`, both relative
  to FP in the same process. They stay fixed from R1 through the confirmatory campaign.
- **Amendments.** A run's rules may change only while none of its jobs has output. A
  later change is an exploratory analysis and is reported as one.
- **Pairing.** Every compared arm runs in one process on the same prompts. Forward
  passes are not reproducible across processes.
- **Fresh prompts.** Each run's main cell uses a fresh range:
  - R2: 9300–9339;
  - R3: 9400–9439;
  - R5: 9500+.

  The exceptions are the bridges:
  - R1 reruns 9100–9119 (Stage 1g);
  - R2 reruns the Stage 1e/1f Qwen failures.
- **Nothing in `../14_kernel_tpot/` is edited.** This folder imports it. A Stage 1g
  function that needs a change is overridden at run time from `run_s1h.py`.

## 5. R1 — calibration and bridge (frozen with `read_stage1h.py`)

**Cell.** Llama-3.1-8B, C = 128K, prompts 9100–9119 (the first two blocks of Stage 1g's
g128 cell), tasks niah_single, niah_multikey, niah_multivalue and vt. Seed 0.

**Regression block.** 8109, 8901 and 8937 (niah_multikey: Stage 1e/1f's key
confusions), at rotation seeds 0, 1 and 2.

**Arms** (preset `h1cal`, 17 arms, one process per block):

| arm | role |
|---|---|
| `fp`, `fp+v4` | reference; FP keys with 4-bit values |
| `uniform@3`, `uniform+v4@3` | D (TurboQuant-3) and D_V4 |
| `uniform@4`, `uniform+v4@4` | dense TurboQuant-4 |
| `qread_v4@1/8` | single-tier reads over the 3-bit store |
| `qreadfp_v16@1/8` | reads over the exact store |
| `qread4_v4@1/8` | the simple design: 4-bit store, 1/8 reads |
| `qread2t4kq_v4@1/8` | the Stage 1g system |
| `qread2t4q_v4@1/8` | the system with exact K+V fetched |
| `fp8kv` | FP8 KV |
| `kivi4_v4`, `kvquant4_v4` | same-memory 4-bit baselines |
| `qoraclefp_v16@1/8`, `qoracle4_v4@1/8` | oracle selection over the exact / 4-bit store |
| `fp_noise` | noise floor (last) |

**Quantities and labels.** All are paired per unit (prompt × task), with prompt-clustered
bootstrap 90% CIs.
1. **Noise floor.** The mean and 95th percentile of |dS| and of KL for `fp_noise`, and
   its lost answers. This is NOISE_FLOOR.
2. **FP8 cost.** The mean dS of `fp8kv` vs FP with its CI, and its lost answers. This
   is FP8_COST.
3. **The margin for R2 onward (rule frozen now; values from R1):**
   `m_FP = clip(hi90(dS_fp8kv), 0.05, 0.10)` nats.
   - NEAR_FP(arm) iff hi90(dS) ≤ m_FP and the > 2-nat tail share's hi90 ≤ 0.05.
   - EQUIV_FP iff the 90% CI lies inside [−m_FP, +m_FP].
   - R1's own labels use the Stage 1g margin of 0.10, for continuity.
4. **Bridge.** For each arm that R1 shares with Stage 1g on prompts 9100–9119 (`fp+v4`,
   D, D_V4, TurboQuant-4 ± v4, `qread_v4`, `qreadfp_v16`, `qread4_v4`,
   `qread2t4kq_v4`), compare the per-unit dS of R1 with Stage 1g's.
   - BRIDGE_OK if every arm's 90% CI of the mean difference lies inside ±0.05 nats;
     otherwise DRIFT, naming the arms.
   - The comparison is across processes, so it measures drift and run-to-run noise
     together.
5. **Best same-memory dense quantizer.** Among `uniform+v4@4`, `kivi4_v4` and
   `kvquant4_v4`, the arm with the lowest mean dS. It becomes the dense comparator from
   R2 onward. A difference smaller than 0.02 nats keeps TurboQuant-4.
6. **Selection loss (oracle).**
   - VOTE_LOSS_EXACT = dS(`qreadfp_v16`) − dS(`qoraclefp_v16`);
   - VOTE_LOSS_4 = dS(`qread4_v4`) − dS(`qoracle4_v4`).

   Effect labels as in Stage 1g (|mean| ≥ 0.05 and a CI excluding 0), plus equivalence
   within ±0.05.
7. **Exact-K+V system.** dS(`qread2t4q_v4`) − dS(`qread2t4kq_v4`), as an effect label.
8. **Key confusions across seeds.**
   - CONFUSION_SYSTEMATIC if, on a prompt, D_V4 is > 2 nats from FP at all three seeds
     while every ≥ 4-bit or exact arm is within 2 nats at all three.
   - CONFUSION_SEED_DEPENDENT if D_V4 fails at only some seeds.
   - NOT_REPRODUCED if D_V4 fails at none.
9. **Lost answers.** Count and exact McNemar p for each arm vs FP, and for the system
   vs D_V4. Each lost answer gets a failure type.
10. **Continuity.** `s_set_nll` labels as in Stage 1g, and the minimum's bias per arm.

**Pilot and gate.** Preset `h1pilot`: one prompt (3112: niah_multivalue, niah_single)
with every arm once. The gate checks:
- every arm ran, and the A2 self-checks are within tolerance;
- `kl_all = 0` for FP;
- `fp_noise` differs from FP somewhere: some token's log-prob differs, or `kl_all > 0`.
  On the CPU smoke the two prefills were bit-identical. If they are also identical on
  the GPU, the gate fails with NOISE_DEGENERATE, and the noise arm is amended (for
  example, to another attention kernel) before any R1 block runs;
- peak memory per device stays under the limit;
- the projected block time is within 90% of the 6-hour job limit.

## 6. R2–R5 (outlines; each is frozen as an amendment before its output exists)

**R2 — Qwen3-30B-A3B, 32K.**
- **Regression block** first: 8234 (multikey key confusion) and 8830 (vt tier
  mismatch), plus 8215, 8218 and 8235, at seeds 0–2.
- **Main cell:** 40 fresh prompts (9300–9339), with R1's arms plus the second question
  pass for single-tier reads, at r = 1/8 and at the floor.
- **Margins:** from R1 (§5.3).
- **Prediction:** the 4-bit tier and the second question pass fix 8830. They fix 8234
  only if the 4-bit vote keeps the true needle's rows.
- **Fallback:** if 8234 is not fixed by the system but is fixed by exact-store reads,
  add a larger-r or exact-key vote arm.

**R3 — stress tasks, Llama 128K (9400–9439).**
- **Tasks:** multikey with about 11 near-duplicate distractors (sievelib's
  `build_multikey_panel`), common- and frequent-word extraction, and NoLiMa.
- **Arms:** FP, noise, FP8, the best dense 4-bit quantizer, the simple design, the
  system, exact-store reads, the oracle, and protected heads (`qreadp`).
- **Outcomes:** free-running accuracy is co-primary, and it finds where 1/8 reads
  break.

**R4 — real tasks.** LongBench v2 (`sievelib/tasks_longbench_v2.py`), plus a HELMET or
LongBench v1 subset, at natural lengths. Accuracy is primary, units FP fails form a
stratum, and the arms are as in R3. A Quest-style per-step page-selection baseline is
added before R4.

**R5 — fixes.** Arms that target what R3 and R4 break: an adaptive read fraction, a
read-floor sweep (K_MIN from 4K to 32K rows per head, at 32K and 64K), protected heads,
and re-selection for long outputs.

## 7. Sequential prompt blocks (R2 onward)

Start a cell at 20 prompts (2 blocks). After each pair of blocks, add a 10-prompt block
only while the deciding comparison's 90% CI straddles its margin, up to a cap of 60
prompts. The deciding comparisons are the system vs FP (NEAR_FP) and the system vs the
best dense 4-bit quantizer. The reader reports the stopping point.

## 8. Systems (bonus track)

Kernel fusion, the GPU-side fetch, the 4-bit decode and end-to-end TTFT/TPOT/throughput
run on spare GPU time. They follow `../14_kernel_tpot/report.md`, Part L.

## 9. Files

The files are listed in `CLAUDE.md`. Results go to
`h0_measurement/results/r14s1h_<tag>_<job>/`, logs to `h0_measurement/logs/`, and each
run's read to `findings/R<n>.md`.

## Amendments

(none yet)
