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
  - R3: 9400–9439 (R3a), 9440–9479 (R3b);
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

**2026-10-04: R2 frozen.** Written after R1 was submitted (jobs 1032158–1032165) and
before any R2 output existed. The rules are in `read_stage1h_r2.py`'s docstring. R1's
rules (§5) are unchanged.

R2's code is in new files, because R1's jobs import the Stage 1h modules while they run:
- `s1h2_lib.py`, `run_s1h2.py`, `read_stage1h_r2.py`;
- `test_r14_stage1h_r2.py`, `submit_s1h2.slurm`;
- `--run-r2` in `script_stage1h.sh`.

**The run.** Qwen3-30B-A3B at 32K, with the stop rule eos_only:
- the main cell (`h2qwen32`) is 22 arms on prompts 9300–9319 (2 blocks);
- the regression block covers 8234, 8830, 8215, 8218 and 8235 at rotation seeds 0, 1
  and 2.

**The new arm, `qread4q_v4`:** a single-tier read over the 4-bit store with a second
question pass. The question is prefilled again over the selected rows only, so its keys
and values come from the same view the answer reads.

**The other added arms:**
- the floor arms at r = 1/2: the system, exact-store reads and 4-bit-store reads;
- Stage 1f's 3-bit two-tier read, `qread2t_v4`.

**Labels:**
- NEAR_FP and EQUIV at R1's m_FP. R2 can't be read until R1 has been.
- SYS_VS_DENSE4: the system against each same-memory dense 4-bit arm, and against R2's
  own best dense quantizer (the strongest baseline on this model).
- REQ1 (the second pass), STORE4, SYS_VS_TT3, and the FLOOR_* labels.
- SEQUENTIAL: add 10-prompt blocks from 9320 while the system's interval straddles m_FP,
  up to 60 prompts.
- FIX, per seed and regression unit. The failing reference is `qread2t_v4` for 8830 and
  D_V4 otherwise. VOTE_LOSES_NEEDLE is flagged if exact-store reads fix 8234 and the
  system doesn't.
- Reported beside Stage 1f's numbers: the seed-0 regression units.

**2026-10-04: R3a frozen** (harder synthetic tasks, both models). Written after R2 was
submitted (jobs 1032364–1032371) and before any R3a output existed. The rules are in
`read_stage1h_r3.py`'s docstring.

The code is in new files:
- `s1h3_lib.py`, `run_s1h3.py`, `read_stage1h_r3.py`;
- `test_r14_stage1h_r3.py`, `submit_s1h3.slurm`;
- `--run-r3-ladder` and `--run-r3a` in `script_stage1h.sh`.

R3a replaces the R3 outline in §6 for the RULER-style part. R3b (common- and
frequent-word extraction, NoLiMa) follows separately (its amendment is below).

**Why.** On the default tasks FP scores 0.98–1.00, so accuracy can't show a loss or a
gain.

**Tasks.** The RULER generators' own difficulty knobs, plus sievelib's contrastive
multikey panel:
- `niah_multikey`: n_keys;
- `niah_multivalue`: n_values;
- `vt`: n_hops;
- `mk_panel`: 48 needles in 4 clusters of 12 near-duplicate keys; one of its four
  questions per prompt.

**Step 1, the ladder (excluded from every result).**
- Arms: FP, D and D_V4.
- Levels (n_keys, n_values, n_hops): (16, 8, 8), (32, 16, 12), (64, 24, 16).
- Prompts 3200–3204, on Llama 128K and Qwen 32K.
- `choose_level` picks each task's level: the lowest level with FP in [0.5, 0.95]. If FP
  is above 0.95 at every level, the highest (CEILING_REMAINS). If below 0.5 at every
  level, the lowest (TOO_HARD). Otherwise the level whose FP is closest to 0.75.
- The rule uses FP only, so it can't favour the design over the baselines.

**Step 2, the main cells.**
- Llama 128K on prompts 9400–9419 (16 arms) and Qwen 32K on 9420–9439 (17 arms), at the
  chosen levels.
- FP_MIN is 0.5, because the tasks are hard on purpose.
- Accuracy is co-primary:
  - ACC_SYSTEM: the system is near FP in accuracy if the interval's lower bound is at
    least −0.03;
  - the system against the best same-memory dense 4-bit arm (chosen by accuracy) and
    against D_V4;
  - the simple design against the best dense 4-bit arm;
  - the second question pass, and vote loss.
- R2's NLL and KL labels apply at R1's m_FP.
- A block run at any other difficulty is INVALID.

**Prompt fit.** The haystack fills 92% of the context, so even the hardest level fits:
64 keys at 32K leaves about 1,100–1,400 tokens spare. Only 2K CPU smokes overflow; the
smokes therefore run at 4K and 16K.

**2026-10-05: R3b frozen** (aggregation and latent-association tasks, both models).
Written after R3a's ladder was submitted (jobs 1032502–1032508) and before any R3b output
existed. The rules are in `read_stage1h_r3b.py`'s docstring. R3b depends on no result of
R1–R3a: R1's m_FP is read at read time, as in R2 and R3a.

The code is in new files:
- `tasks_s1h.py` (the tasks; no model), with its data in `data/r3b/` pinned by sha256;
- `s1h3b_lib.py`, `run_s1h3b.py`, `read_stage1h_r3b.py`;
- `test_r14_stage1h_r3b.py`, `submit_s1h3b.slurm`;
- `--run-r3b-ladder` and `--run-r3b` in `script_stage1h.sh`.

**Why.** Every task so far is retrieval: the question names the needle. The design reads
1/8 of the rows, chosen by the question's attention. Two kinds of task can break that
while retrieval does not:
- **aggregation**: the answer is a count over the whole context;
- **latent association**: the question shares no word with the needle, so its attention
  may not find it.

**Tasks** (`tasks_s1h.py`):
- `cwe`, `fwe`: RULER's common- and frequent-word extraction (RULER@c3f5e3b4, templates
  verbatim);
- `nolima`: NoLiMa's one-hop questions (its 32 needle–test pairs; Adobe Research License,
  non-commercial research);
- `nolima_direct`: the same prompt with NoLiMa's direct question, which repeats the
  needle's words. This is the control for the vote.

**Deviations from the sources:**
- raw text, as every R14 task;
- the context fills 92% of C;
- `cwe` uses wonderwords' 8,050 lowercase single words;
- scoring matches whole words;
- `nolima` uses the project's PG-19 haystack, with a character whose name the window
  does not contain;
- the stop rule is `r8list` for both models (`stops_s1h.py`): no stop before a line with
  content, a line holding a single list item does not stop the answer, anything else stops
  at its first line. Without a newline stop, raw-text Llama runs on past a one-line answer
  and Qwen never emits EOS, and the run-on can state list words by chance. (Qwen had
  eos_only until the 2026-10-05 review; no R3b output existed.)

**Step 1, the ladder (excluded from every result).**
- Arms: FP, D and D_V4.
- `cwe` and `fwe` at three levels, on prompts 3210–3217. Level 1 is RULER's setting
  (`freq_cw` 30, `alpha` 2.0); then (100, 1.5) and (300, 1.2). `cwe` gets easier, `fwe`
  harder.
- `nolima` and `nolima_direct` at their one setting, on prompts 3210–3225.
- `choose_level_r3b`: the lowest level with FP in [0.5, 0.95]; otherwise the level whose
  FP is closest to 0.75.
- A task enters step 2 only if FP scores ≥ 0.5 at its level. `nolima_direct` enters with
  `nolima`. NoLiMa's own table puts Llama-3.1-8B at 0.14 by 32K, so `nolima` may not enter
  at Llama 128K.

**Step 2, the main cells.**
- Llama 128K on prompts 9440–9459 and Qwen 32K on 9460–9479, with R3a's arms (16 / 17).
- The accuracy labels are R3a's, per family: AGG (`cwe` + `fwe`), LATENT, DIRECT.
- R2's NLL and KL labels apply at the cell's own FP8 margin (as R3a2), with R1's m_FP
  reported beside it.
- **LEX_VOTE**: the vote's loss against the oracle on `nolima` minus on `nolima_direct`,
  over prompts FP gets right on both. **LEX_SYS**: the same for the system.
- A task whose FP falls below 0.5 in the cell leaves the labels (FP_LOW).
- Reported: on the NoLiMa tasks, the share of answers naming someone outside NoLiMa's
  character set (OTHER_NAME), such as a character of the PG-19 book the haystack comes from.

**2026-10-05: R3a2** (R3a's harness, after its first ladder; no main-cell output exists).
The first ladder (jobs 1032502–1032508) is void:
- its level-2 Llama job was cancelled by the system, so the ladder reader failed;
- Qwen writes multivalue as a numbered list, one value per line, and never emits EOS on raw
  text, so every answer ran into the cap (sized for a comma list) mid-list: FP scored exactly
  14/16 and 19/24 on every prompt, which the level rule would have read as headroom;
- on one level-3 Llama item the first token was `:\n\n`, and R8's newline stop ended the
  answer there.

The fixes (`s1h3_lib.py`, `run_s1h3.py`, `read_stage1h_r3.py`; their docstrings):
- the stop rule `r8list` for both models (`stops_s1h.py`, shared with R3b and R4);
- per-unit caps for multivalue and vt: the larger of tasks_ruler's limit and 16 + 1.5 × the
  tokens of the expected answer written as a numbered list in the run's tokenizer;
- NEAR_FP and EQUIV at the cell's own margin, clip(hi90 of FP8 KV's mean dP in the cell,
  0.05, 0.10), with R1's m_FP reported beside it. R1's margin was measured on Llama's RULER
  answer values at 128K, and R2–R4 are other models, tasks and answer spans.
- the ladder is rerun (`--run-r3-ladder`), and blocks are tagged `R3a2`.

**2026-10-05: R4b frozen** (real tasks at 128K), replacing the first R4 design (LongBench v2 +
a LongBench v1 subset at 32K), which never ran. The rules are in `read_stage1h_r4.py`'s
docstring. R4 depends on no result of R1–R3b.

**Why the change.** At their natural lengths most LongBench v1 items are short enough that
the system's floor (16,384 rows) reads everything, so they do not test its reads; the
LongBench v2 Qwen cell at 32K sat in the same regime. R4b runs both suites at 128K, where
the reads matter.

**Tasks** (`tasks_s1h4.py`):
- **LongBench v2**, scored by forced choice as bug 9's V5–V7.
- **HELMET** (code @aeadacc6, data @dddb209d), its 128K configs, no chat template, a newline
  stop:
  - RAG: kilt_nq and kilt_hotpotqa (1,000 passages, 2 demonstrations; substring exact match);
  - re-ranking: msmarco_rerank_psg (1,000 passages; NDCG@10, 200-token rankings);
  - many-shot ICL: trec_coarse (6,600 shots) and banking77 (5,900), labels mapped to random
    integers per item, so a closed-book answer is at chance.

**Items** (`make_r4_manifest.py` → `data/r4/manifest_r4.json`, pinned):
- LongBench v2: bug 9's V7 pool (qualification + development; V7's confirmation untouched),
  rendered per model at 131,072 tokens;
- HELMET ICL: HELMET's own test selection; RAG and re-ranking: a salted-hash order (HELMET's
  sampling depends on its global random state); HELMET's demonstrations and its end-of-context
  truncation.

**Method fixes from the 2026-10-05 review:**
- **The vote on LongBench v2** observes 32 rows spread evenly over the question and its four
  choices. The question's last 32 rows are the official format instruction, the chat
  template's tail and the response prefix; even the last 32 rows of the user message are
  mostly the format line and the end of choice D. RULER and HELMET keep the last rows (their
  questions are short).
- **A closed-book arm** (an empty document, or empty passages and demonstrations, the same
  question): the context-dependent units are those where FP's accuracy beats it by ≥ 0.5,
  and every accuracy label is also given on them.
- **A pilot and a gate per cell** (validity, noise, ≤ 76 GiB per device, the projected block
  within 90% of its wall time) before its blocks.
- **Margins** per cell and family from FP8 KV's own cost.
- **Labels per family** (lbv2, RAG, re-ranking, ICL), never pooled across metrics; every unit
  its own bootstrap cluster.
- **The split** is the first token boundary after the document where the separate
  tokenization equals the whole prompt's.
- **Reported**: the change in the gold choice's log-probability (lbv2).

**New arms:** `quest_v16@1/8`, `quest4_v4@1/8`, the floor system `qread2t4kqF_v4` (as the first
design) and `closedbook`: 20 arms per cell, stop rule `r8list`.

**Cells**, a pilot, a gate and two blocks each:
- LongBench v2: Llama 128K (1 GPU) and Qwen3-30B-A3B 128K (2 GPUs), 40 items each;
- HELMET: Llama 128K and Qwen 128K, 10 items × 5 tasks each.

About 18 GPU-h, of which Qwen's 2-GPU cells are about 10.

**Execution note (2026-10-05, before any R4 output):** Trillium gives a GPU job 1 GPU or whole
4-GPU nodes; the 2-GPU Qwen pilot was refused. A Qwen cell's two blocks now run at once in one
node job (2 GPUs each; the same items, arms and processes per block as two jobs; results
`r14s1h_<tag>_<job>_0/` and `_1/`, read as job IDs `<job>_0`, `<job>_1`). Its pilot takes a node
but runs on 2 GPUs, so the gate's per-GPU peak is a block's; 2 GPUs sit idle for the pilot's
≤ 2 h. No rule of the read changes.

**Validity fix (2026-10-05, before any R4 output):** R1's answer-value mask check (the mask
found in ≥ 90% of FP's correct answers) scored a unit set with no correct FP answer as 0% and
failed it. That would have failed most 1-item LongBench v2 pilots (FP is right on about a third
of items) and HELMET pilots whose re-ranking NDCG is below 1. In R4 the check now holds
vacuously when FP answers no unit correctly (`read_stage1h_r4.validate_h4`). The Qwen paths
(Quest, closed book, forced choice, the vote span, `r8list`) now also have a CPU smoke on
Qwen3-0.6B, Qwen's `validate_with` model.

**2026-10-08: R3b Qwen ladder accepted with a known deviation.** Qwen's R3b ladder jobs
(22517276, 22517280, 22517282, 22517284, Rorqual) ran with stop rule `eos_only`, before Qwen
moved to `r8list` (2026-10-05); the current reader rejects them, and `findings/R3b_levels.json`
was written by an older reader that accepted them. The levels it chose for Qwen (cwe
`freq_cw=30`, fwe `alpha=2`) are kept: the main cell (22560819–20) ran with `r8list` and still
has headroom (FP 0.80 on cwe, 0.88 on fwe), so its results stand on their own; only the choice
of difficulty was made under the old stop rule. Not re-run, to give R5 the GPU time. Future
ladders must run with the main cell's stop rule.

**2026-10-08: R5 theory (draft, not frozen).** What R1–R4 say about the selector, why, and
the design R5 should test. Nothing here changes a frozen rule.

*Part 1 — does the design work in theory?*

For one head at one step, attending only to rows S (renormalized) gives an exact identity:

```
o − o_S = ε · (ō_unread − ō_read)      ε = attention mass on the unread rows,
                                       ō = attention-weighted mean value
```

The current design (fixed r = 1/8, one vote at question time, unread rows evicted) picks S
before the answer's queries exist. Unless the read keys surround all the others, some query
puts almost all its mass outside S, so no advance selection carries a guarantee. It is correct
exactly when two conditions hold: **C1, concentration** (a small set holds almost all of the
answer-time attention) and **C2, predictability** (the question's attention already points to
that set). R1–R4 match this:

| Workload | C1 | C2 | Theory predicts | Observed |
|---|---|---|---|---|
| Single needle (R1, R2) | yes | yes | near FP | near FP, 0 answers lost |
| Many values (R3a multivalue) | yes, per step | no | vote fails, oracle fine | Qwen 32K: oracle +0.6, vote +69 nats |
| Aggregation (R3b cwe, fwe) | no (flat) | — | even the oracle fails | oracle +0.75 to +2.6 |
| Enumeration (R4 HELMET re-ranking) | partly | no (drifts) | re-selection wins | Quest beats the system by 14.7 nats |

The two-tier store is the right foundation: a guarantee needs a cheap view of every row (to
bound what goes unread) and an exact fallback (to fix what the bound flags), which is what tier
1 (4-bit, rotated, all rows) and tier 2 (exact) are. R1 shows why: dense 4-bit costs +0.07 to
+0.12 nats on needles, the two-tier system +0.016, because tier 2 makes the high-mass rows exact.

The proposed design is a **certified controller**: per head and step, read rows exactly from
tier 2 until a computable upper bound on the error is below a target τ; where attention is
flat, read the remaining rows from tier 1 instead of evicting them. Accuracy is then provably
within τ at every head and step on any model; GPU memory is that of a 4-bit cache plus a small
exact working set. Speed is not guaranteed. Per-step memory traffic, as a fraction of FP16 dense
attention: a full 4-bit score scan for the bound ≈ 1/8, exact reads of 1/8 of the rows ≈ 0.08,
total ≈ 0.20 against 0.25 for a plain 4-bit cache. Page-level bounds (per-channel min/max per
16 rows) cost ≈ 1/16–1/32 instead (total ≈ 0.11–0.14) if they are tight enough on real
attention. Flat heads cost about a 4-bit cache: no gain, no loss.

Conclusions:
1. The current design is the special case with the certificate off, valid where C1 and C2 hold.
2. Keep the two tiers.
3. A certified controller makes attention-level correctness independent of the model.
4. Savings are limited by attention concentration and by how tight a cheap bound is; whether
   the design pays is an empirical question (R5 measures bound tightness from FP logs).
5. The question-time vote becomes a warm start: a good vote lowers cost; correctness no longer
   depends on it.

*Part 2 — the claim, precise and proved.*

Setting: one softmax head, query q, scores s_i = ⟨q, k_i⟩/√d + β_i (β_i any exactly known bias:
sinks, ALiBi, masks). Tier 1 stores k̂_i, v̂_i for all rows; the quantizer guarantees
‖k_i − k̂_i‖ ≤ η_i and ‖v_i − v̂_i‖ ≤ ν, computable from its step sizes (rotation preserves
norms). V = max‖v_i‖ after prefill; T = unread rows; read rows use exact keys (with 4-bit
values, as now, every bound gains an additive ν).

- **Lemma 1 (evicting).** o − o_S = ε(ō_T − ō_S), so ‖o − o_S‖ ≤ 2Vε. *Proof:* o = (1 − ε)ō_S +
  εō_T and the renormalized output over S is ō_S.
- **Lemma 2 (certificate).** With b_i = ‖q‖η_i/√d, |s_i − ŝ_i| ≤ b_i (Cauchy–Schwarz). With
  M_in = Σ_S e^{s_i} (exact) and U = Σ_T e^{ŝ_i + b_i}: ε ≤ ε̄ = U/(M_in + U). *Proof:* the true
  unread mass is ≤ U and x/(M_in + x) increases in x.
- **Lemma 3 (unread rows read at 4 bits).** With b = max_T b_i:
  ‖o − õ‖ ≤ ε[(e^{2b} − 1)(3V + ν) + ν] ≈ ε(6bV + ν). *Proof:* o − õ = (ε̃ − ε)(ō_S − õ_T) +
  ε(ō_T − õ_T); scores on S are exact, each weight in T moves by a factor in [e^{−2b}, e^{2b}],
  so |ε̃ − ε| ≤ ε(e^{2b} − 1) and ‖ō_T − õ_T‖ ≤ (e^{2b} − 1)V + ν. The error is a product: unread
  mass × 4-bit error.
- **Theorem.** Start from any S (e.g. the vote's); compute the Lemma 1 or Lemma 3 bound with ε̄
  for ε; while it exceeds τ, fetch exactly the unread row with the largest e^{ŝ_i + b_i}. For any
  model with softmax attention over cached rows, any input, layer, head and step, the output is
  within τ of exact, or the controller reports that its row cap was reached. *Proof:* the bound
  uses only computed quantities (exact scores on S, tier-1 scores, step sizes, ‖q‖, V) and only
  softmax algebra, Cauchy–Schwarz and the quantizer's guarantee; each fetched row moves its term
  from U to M_in, so the bound only falls; at S = all rows, U = 0 and the error is ≤ ν.

Model properties change only the cost (how long the loop runs): flat attention; large ‖q‖ or
key outliers (larger b_i); rows shared by many heads (GQA ratio; MLA shares one row set across
all heads); answers that drift (more re-fetches). None can push the output past τ unreported.

End-to-end fidelity cannot be model-independent: for any τ > 0 some model and input exist where
a size-τ perturbation in one head flips the greedy token (near-tied logits); this holds for FP8
and every KV format. So τ is set per model by measurement, not tuning: a high quantile of the
per-head attention error FP8 KV causes on that model (from FP runs), optionally weighted per
layer by measured sensitivity, and checked by injecting known ε and fitting KL against it.
**Corrected claim:** a model changes how much the design saves and how strict τ must be (one
measured number); it cannot make the design exceed its attention-level target unreported.

*Models downloaded to test the architecture adapters (2026-10-08, `.hf_cache/hub`).* The theorem
depends on structure, not size, so one small member per family validates its adapter; large
SOTA models then only need their savings measured.

| Family | Model (repo) | Adapter tested |
|---|---|---|
| local/global, learned sinks | `openai/gpt-oss-20b` | global layers only; sinks as exact score terms |
| latent attention (MLA) | `deepseek-ai/DeepSeek-V2-Lite-Chat`, `moonshotai/Moonlight-16B-A3B-Instruct` | row = token latent shared by all heads; absorbed queries; values from the latent |
| local/global, QK-norm | `google/gemma-3-12b-it` | global layers only; 5 local : 1 global |
| hybrid SSM / linear + softmax | `ibm-granite/granite-4.0-h-tiny`, `moonshotai/Kimi-Linear-48B-A3B-Instruct` | softmax layers only; Mamba-2 / KDA state untouched |

Llama-3.1-8B (full GQA, 4 query heads per KV head, no QK-norm) and Qwen3-30B-A3B (8 per KV
head, QK-norm, MoE) remain the design models; these six come in only after R5 freezes the
controller.

**2026-10-08: R5 (frozen 2026-10-08, after the pilot, before any R5 cell's output; see "R5 freeze" below).** Code: `cert_s1h5.py`
(the theory's math, checked by brute force in `test_r14_stage1h_r5.py`), `probe_s1h5.py` (the
measurement and injected-error hooks), `s1h5_lib.py` (presets, arms), `run_s1h5.py` (driver on any
suite's units: `--suite r3 | r3b | r4`), `consolidate_r5.py` (R5.0).

*R5.0 result* (`findings/R5_0_split.md`, no GPU). The system's loss at r = 1/8, split into
budget (best fixed set of floor(rC) rows, exact keys, minus FP), vote (the question's vote instead
of that set) and store (the two-tier store instead of exact keys on the vote's rows):
- the store adds ≈ 0 in every cell (−1.2 to +0.3 nats);
- the vote is the failure wherever there is one: Qwen 32K multivalue +69, cwe +11.4, vt +4.6,
  mk_panel +1.7; Llama HELMET re-ranking +28.6, kilt_nq +1.9;
- the budget is modest: at most +2.7 (Qwen cwe), +1.0 (Llama cwe), +0.5 (re-ranking).
So the estimate (which rows), not the budget (how many), is R5's main lever; the certificate's
per-step check targets exactly that. (Part 1's table said aggregation fails "even with the oracle";
the budget does cost there, but the vote costs more.)

*R5.1 + R5.2, one run per cell* (preset `h5*`, stop `r8list`, 19–21 arms): fp; `probe`; `inj_top`
at ε ∈ {0.003, 0.01, 0.03, 0.1, 0.3} and `inj_rnd` at ε ∈ {0.01, 0.1} (teacher-forced only); the
tail arm `qread2t4kqT_v4` at 1/8 (and the floor); references: dense 4-bit keys and +v4, the
3-bit single-tier read, the exact-store read, the system at 1/8 (and the floor), FP8, KIVI-4, the
exact oracle; `fp_noise` last. Cells (the same units as R1–R4):

| cell | suite | units |
|---|---|---|
| Llama 128K RULER | r3 (default difficulty) | 9100–9109 × 4 tasks |
| Llama 128K harder RULER | r3 (R3a levels) | 9400–9409 × 4 tasks |
| Llama 128K cwe/fwe | r3b (R3b levels) | 9440–9449 × 2 tasks |
| Llama 128K HELMET, LongBench v2 | r4 | HELMET items 0–4 × 5 tasks; lbv2 0–19 |
| Qwen 32K RULER + harder | r3 | 9300–9309 and 9420–9429 × 4 tasks |
| Qwen 32K cwe/fwe | r3b | 9460–9469 × 2 tasks |
| Qwen 128K HELMET, LongBench v2 | r4 | after R4's Qwen cells, if they show the same split |

*Hypotheses (exploratory; each with its check):*
- **V (validity).** The probe's pass is FP (its KL to FP is 0); Lemma 1's and Lemma 3's bounds
  hold on every measured head and step (`probe_l1_ok` = `probe_l3_ok` = 1). Failure = a bug.
- **H1 (the link).** Per unit, the teacher-forced `kl_span` of `inj_top` grows with the injected
  missed mass as KL ≈ a_M ε² at small ε, one constant a_M per model; at the same ε, `inj_rnd`'s KL
  differs from `inj_top`'s as Lemma 1's value term predicts; the per-head relative output error
  (recorded) explains KL at least as well as ε. Read: fit quality per model and task family.
- **H2 (certificate tightness).** On needle tasks, the certified budget (full 4-bit scan, cold) at
  ε = 0.01 is at most 2× the shared-set oracle budget, and the page-certified budget at most 4×;
  warm-starting from the vote's rows lowers both. Read: `s1h5_probe_groups`.
- **H3 (cost forecast).** The shared-set oracle budget at ε predicts the certified budget across
  task families and both models (rank correlation), so concentration alone forecasts cost.
- **H4 (Lemma 3 in practice).** The tail arm is never worse than the system at the same r and
  recovers most of the vote's loss on the failing cells (Qwen 32K multivalue and cwe, HELMET
  re-ranking).

*Arms (2026-10-08 revision).* Already measured in R1–R4 and kept as references: FP, `fp_noise`,
FP8 KV, dense 4-bit (`uniform@4`, `uniform+v4@4`), KIVI-4 (KVQuant-4 dropped: never the best
dense arm by enough to matter), the 3-bit single-tier vote `qread_v4` (SnapKV-style question-time
eviction), the exact-store vote `qreadfp_v16`, the system `qread2t4kq_v4` (and its floor), the
static exact oracle `qoraclefp_v16`; Quest (`quest_v16`, `quest4_v4`) was R4-only and now runs in
every R5 cell (run_s1h4's arm, unchanged). New: `probe`; `inj_top`, `inj_rnd`; `inj_top_l{0..3}`
(ε = 0.1 in one layer quarter: per-layer sensitivity and additivity); the tail arm
`qread2t4kqT_v4`. 25 arms at Llama 128K, 27 at Qwen 32K. Still for R5.3 (after R5.1/R5.2 are
read): the stateful controller as a generating arm, top-p uncertified, a sampling estimator.

*Metrics, and where each is reported.*

| metric | where |
|---|---|
| NLL, summed and per token | `a_span_nll`, `a_span_nll_tok` (÷ `a_span_ntok`, the A2 span) on every row |
| KL, summed and per token | `kl_span`, `kl_span_tok`, `kl_all`, `kl_all_tok` on every row |
| accuracy | `score` (lbv2: `fc_correct`; injected arms are teacher-forced only, `tf_only`) |
| memory traffic share | `traffic_frac` (analytic, per decode step, of FP16 K+V; `s1h5_lib.traffic_frac`); the controller's: `probe_ctl_*_traffic_frac` |
| exact rows read | `exact_rows_frac`; the controller's: `probe_ctl_*_exact_rows_frac`, trace `F_sum / steps` |
| re-fetch rate | `probe_ctl_*_refetch_rate` (fetch events per step), `*_fetched_rows_frac`; trace `fetch_ev`, `fetched` |
| row-cap hits | `probe_ctl_*_cap_rate` (steps above 25% of the context); trace `cap_steps` |
| B_cert / B_min (tightness) | `probe_*_over_union_*_med`; groups `cert_*`, `cert_hp_*`, `page_*` against `union_*`; heads `epsbar_vote1 / eps_vote1` |
| missed mass of each selection | heads `eps_vote1`, `eps_votef`, `eps_static1`, `eps_step1` |
| output error of each design, and its bound | heads `rel_sys`, `rel_tail`, `rel_fp8`, `rel_d4`, `rel_static1`; `l1_bound`, `l3_bound` |
| the τ anchor | heads `err_fp8`, `rel_fp8` |
| concentration | heads `bmin_*`, `entropy`, `mass_ctx`; groups `union_*` |

*Justifying the theory for arbitrary models.* The proposal's claim is structural, so R5 checks each
link on real attention rather than assuming it:
1. the bounds hold (V: Lemmas 1–3, the certificate, the score bound, on every measured head and
   step of every task family and both models);
2. the theorem holds in practice: the controller emulated on FP's path with the worst-case bound
   never leaves its target (trace `viol` = 0), and the high-probability bound's violations are
   measured (`hp_viol`, trace `viol` for `hp`);
3. the link from attention error to the output is the second-order one the corrected claim relies
   on: KL ∝ ε² (log-log slope ≈ 2 over the injected levels), random-order eviction at the same ε
   costs what Lemma 1's value term predicts, and the four layer quarters' KL add up to the
   all-layer KL (the quadratic form has no large cross-layer terms);
4. what a model changes is cost: concentration (B_min, the shared-set budget) forecasts the
   certified budget, the traffic and the re-fetches (H2, H3), on two models that differ in GQA
   ratio (4 vs 8), QK-norm, MoE and context length.
The six downloaded models (plan.md "R5 theory") repeat 1–4 after the design is frozen.

*Pilot (`script_stage1h_r5.sh --pilot-r5`).* Four 1-GPU jobs on the longest-answer units outside
R5's units (Llama 128K: 9410 multivalue + vt at R3a's levels, HELMET item 5 re-ranking + kilt_nq;
Qwen 32K: 9430 multivalue + vt, 9470 cwe), then `read_stage1h_r5.py --pilot` →
`findings/R5_pilot.{json,md}`: validity, peak memory, seconds per unit and arm, the projected GPU-h
of each cell, a first look. The R5 cells are submitted only after the amendment is frozen.

R5.3 (the stateful certified controller, per-step re-selection, page bounds, top-p uncertified,
Quest and SnapKV-style references) is designed after R5.1/R5.2 are read.

**2026-10-08: R5 freeze** (after the pilot 1059459–62, reader 1059463, `findings/R5_pilot.md`; no
R5 cell has output). Everything above under "R5" holds, with these results of the pilot:

*Pilot.* Validity (V) passed on both models and all three suites: the probe's pass is FP; Lemma 1
and 3 bounds, the certificate and the score bound hold on every measured head and step; the
worst-case and the high-probability controller traces never left their targets. Peak GPU memory
49.4 GiB (Llama 128K), 64.3 GiB (Qwen 32K, one GPU). Pilot units are excluded from every result.

*Fix: delta injection.* The pilot's `inj_top` KL levelled off at small ε (Llama multivalue: KL 0.020
at ε = 0.003 and 0.018 at 0.01; Qwen multivalue 0.010 at both): the injected arms recomputed the
answer rows' attention in fp32 while FP's runs in bf16, and that recompute, not the eviction, set the
floor. The injection now adds to FP's own output only the eviction's effect, computed by one fp32
path for both terms (`probe_s1h5._inject_layer`): nothing evicted gives FP's output exactly (tested:
`inj_top@0` has KL 0 in the CPU smokes). The probe never changed outputs and is unaffected.

*Cost estimate.* From the pilot, seconds per unit ≈ 210 + 2.4 T at Llama 128K and 37 + 2.3 T at Qwen
32K (T = FP's answer tokens); with each cell's answer lengths from R1–R4:

| cell | units | GPU-h |
|---|---:|---:|
| Llama 128K RULER (9100–9109) | 40 | 2.8 |
| Llama 128K harder RULER (9400–9409, R3a levels) | 40 | 4.2 |
| Llama 128K cwe/fwe (9440–9449, R3b levels) | 20 | 2.2 |
| Llama 128K HELMET (items 0–4) | 25 | 1.8 |
| Llama 128K LongBench v2 (0–19) | 20 | 1.2 |
| Qwen 32K RULER (9300–9309) | 40 | 1.5 |
| Qwen 32K harder RULER (9420–9429, R3a levels) | 40 | 3.2 |
| Qwen 32K cwe/fwe (9460–9469, R3b levels) | 20 | 0.8 |

≈ 17.7 GPU-h of compute; ≈ 25 GPU-h with each job's setup (tests, model load) and a 1.25× margin.
Qwen 128K (HELMET, LongBench v2) waits for R4's Qwen cells, as planned.

*Blocks* (`script_stage1h_r5.sh --run-r5`; 1 GPU each, rot_seed 0, stop `r8list`): Llama RULER and
harder RULER in two blocks of 5 prompts each; Llama cwe/fwe, HELMET, LongBench v2, Qwen RULER and
Qwen cwe/fwe in one block each; Qwen harder RULER in two blocks of 5 — 11 jobs, wall about twice
the projection. A reader (afterany) checks V on every block and writes `findings/R5_blocks_check`.

*Read rules (frozen; exploratory — no labels, every number with a unit-clustered bootstrap 90%
interval; each model × task family separately, never pooled across families):*
- **V** as in the pilot reader (`read_stage1h_r5.validity`): a block failing any check is
  INVALID and excluded.
- **H1.** Per unit, the slope of log `kl_all` on log ε over `inj_top` at ε ∈ {0.003, 0.01, 0.03,
  0.1, 0.3} (units with KL > 0 at ≥ 3 levels); median per model × family. The quarters' sum:
  Σ_q `kl_all(inj_top_l{q}@0.1)` / `kl_all(inj_top@0.1)`. `inj_rnd` / `inj_top` KL at ε = 0.01 and
  0.1. The link: per unit and injected arm, R² of log `kl_all` on log mean realized ε and on log mean
  `rel_inj`.
- **H2.** Medians of `cert` / `union`, `cert_hp` / `union`, `page` / `union` at ε ∈ {0.01, 0.1} over
  (unit, layer, KV head, step); the controller traces' exact-rows share, traffic, re-fetch rate,
  cap rate and violation rate per bound and ε.
- **H3.** Spearman ρ between the shared-set oracle budget and the certified budgets over (unit,
  layer, KV head, step), and between each unit's mean oracle budget and its controller exact-rows
  share.
- **H4.** Paired differences (per span token) of dP, `kl_span` and accuracy: the tail arm against
  the system at the same r; against the exact oracle, Quest (exact and 4-bit), dense 4-bit
  (`uniform+v4`) and FP8 — each beside its `traffic_frac`.
- **The accuracy–cost table:** every arm's dP/token, KL/token, accuracy change (FP-failed units a
  separate stratum) and `traffic_frac` / `exact_rows_frac`, per model × family.
R5.3 (the stateful controller as a generating arm, tighter certificates, top-p, a sampling
estimator) is designed from these.

**2026-10-08: R5 read.** `findings/R5.md` (tables: `R5_reader.md`, reader `read_stage1h_r5.py --r5`).
Validity held everywhere except LongBench v2 (1059661), excluded by the frozen rule (probe KL
0.001–0.002 on 4 of 20 single-token answers: bf16 nondeterminism on the decode path, not the probe;
appendix). The tail design (`qread2t4kqT_v4`: nothing evicted, the unselected rows from the 4-bit
tier) is near FP8 in all 17 groups and removes the vote's failure; its error sits at FP8's level
because its selected rows keep 4-bit values; Lemma 3's worst-case bound never certifies it at FP8
level. H1's frozen slope (0.4–1.0) is flattened by a bf16 floor at small ε (post hoc, floor removed:
1.2–2.0). Not frozen, post hoc: the floor correction and the tail-against-FP8 shares.

**2026-10-08: R5.3 frozen** (the tail design's scan; before any R5.3 output). R5.3 asks what the
tail design needs, now that R5 found it near FP8: exact values on the selected rows, how coarse the
tier may be, how many rows must be exact, and whether selection still matters.

*Arms* (family `tail5`, `tail<kb>[x][o]_v<vb>` at read fraction r: tier-1 keys at kb bits and
values at vb bits; tier 2's exact keys on the selected rows, and exact values with `x`; the
question again; nothing evicted; `o` = the static oracle's rows instead of the vote's):
`tail4_v4`, `tail4x_v4`, `tail3x_v3`, `tail2x_v2`, `tail2x_v4`, `tail2_v2` at 1/8; `tail2x_v2` at
1/16 and 1/4; `tail2xo_v2` at 1/8. References: the R5 tail arm `qread2t4kqT_v4` (identical to
`tail4_v4` by construction: an implementation check), the system at 1/8, the exact oracle, dense
4/4 and 2/2 (`uniform+v4@4`, `uniform+v2@2`, and their key-only bases), `qread_v4`, `qreadfp_v16`,
FP8, `fp_noise` last — 21 arms (presets `h53llama128`, `h53qwen32`; stop `r8list`; rot_seed 0).
Traffic per decode step (of FP16 K+V): ((1 − r)(kb + vb) + r(16 + (16 if x else vb))) / 32 —
e.g. `tail2_v2` 0.180, `tail2x_v2` 0.234, `tail4x_v4` 0.344 (FP8 0.50, dense 4/4 0.25).

*Cells* (R5's units on the families where designs differed, and a no-regression slice): Llama 128K
RULER 9100–9102 × 4 tasks, harder RULER (R3a levels) 9400–9404 × 4, cwe/fwe (R3b levels)
9440–9444, HELMET items 0–4 × 5 tasks; Qwen 32K RULER 9300–9302 × 4, harder RULER 9420–9424 × 4,
cwe/fwe 9460–9464 — 109 units, 7 jobs (`script_stage1h_r5.sh --run-r53`), ≈ 8 GPU-h from R5's
per-unit times (no pilot: every code path is R5's tail arm with other widths, checked on CPU).

*Read rules (frozen; exploratory, no labels; unit-clustered bootstrap 90% intervals; paired per unit;
each model × cell × family separately; `read_stage1h_r5.py --r53`):*
- **V.** Every planned arm on every unit; tail arms evict nothing; `tail4_v4` equals
  `qread2t4kqT_v4` (max |Δ log-prob| ≤ 1e-6 per unit); peak ≤ 76 GiB per GPU. A failing block is
  excluded and reported in an appendix.
- **Q1** exact values: `tail4x_v4` − `tail4_v4` (KL/token, dP/token, accuracy).
- **Q2** tier precision: each tail arm at 1/8 − FP8; `tail4x_v4` − dense 4/4, `tail2x_v2` − dense 2/2.
- **Q3** the read fraction: `tail2x_v2` at 1/16 and 1/4 − at 1/8.
- **Q4** selection with nothing evicted: `tail2xo_v2` − `tail2x_v2`.
- **Q5** against eviction: `tail2x_v2` − the system, − the exact oracle (at 1/8).
- **The frontier:** every arm's traffic, KL/token, dP/token and accuracy change (FP-correct units).

**2026-10-09: R5 theory, part 3 — a certificate for the tail design (draft; no run uses it yet).**
Code: `cert_s1h5.py` (`tail_bound_det`, `tail_bound_conc`, `bernstein_radius`, `trunc_z`,
`center_dist`), tests `test_r14_stage1h_r5.py` (`test_cert_tail`), probe columns `probe_s1h5.py`
(`_tail_certs`).

*Why.* R5: Lemma 3 never certified the tail at FP8's error (0% of head-steps). R5's head records
show why no worst-case bound can: the Cauchy–Schwarz score bound b_i = ‖q‖η_i/√d (largest per head
and step) has median 8.1 nats on Llama 128K and 8.0 on Qwen 32K (p99 12 and 30), so any factor
e^{b} is about 3,000; the high-probability bound (z = 5) is about 3.5 nats, still a factor of 33. Yet
the actual score errors are small and unbiased in direction (the z = 5 bound is exceeded on 0.45%
of head-steps). The certificate has to use the errors' independence, not their worst case.

*Setting.* One head; S = rows read with exact keys (values exact with `x`, else from tier 1),
T = unread rows from tier 1, fixed rows exact (weight W_F). w_i = e^{s_i}, ŵ_i = e^{ŝ_i},
δ_i = s_i − ŝ_i, e_i = v_i − v̂_i, ν_i = ‖e_i‖, õ = the design's output, Z = W_F + Σ_S w + Σ_T w.
Per row the design stores η_i = ‖k_i − k̂_i‖ and ν_i (one scalar each, computed at write time; an
8-bit log code rounded up keeps every bound valid: 16 bits on a 4/4-bit row of 1,024 at d = 128,
+1.6%).

- **Identity.** Z(o − õ) = Σ_S w_i(v_i − ṽ_i) + Σ_T w_i e_i + Σ_T (w_i − ŵ_i)(v̂_i − õ), ṽ_i the
  value the design reads on S. *Proof:* Zo and Ẑõ (Ẑ = W_F + Σ_S w + Σ_T ŵ) are the two weighted
  sums; subtract Zõ = Ẑõ + (Z − Ẑ)õ. Checked exactly in the tests (deviation 2e-15).
- **Lemma 4 (deterministic, per row).** With |δ_i| ≤ b_i:
  ‖o − õ‖ ≤ [Σ_S w_i ν_i + Σ_T ŵ_i(e^{b_i}ν_i + (e^{b_i} − 1)‖v̂_i − õ‖)] / (W_F + Σ_S w_i +
  Σ_T ŵ_i e^{−b_i}), the S sum only with 4-bit read values. *Proof:* triangle inequality on the
  identity; w_i ∈ [ŵ_i e^{−b_i}, ŵ_i e^{b_i}] gives |w_i − ŵ_i| ≤ ŵ_i(e^{b_i} − 1), w_i ≤ ŵ_i e^{b_i}
  and Z ≥ the denominator. Lemma 3 with each row's own b_i, ν_i and its distance from the output
  (the tail's mass-weighted value spread) instead of maxima; never above Lemma 3 in the tests
  (median 0.05 of it), still carries e^{b_i}.
- **Model M.** Given everything stored and the rows read (S chosen from those, e.g. the vote):
  (M1) rows' errors are independent, and each row's key and value errors are independent;
  (M2) E[e_i | δ_i] = 0, ‖e_i‖ = ν_i; (M3) δ_i is symmetric and sub-Gaussian with proxy σ_i², i.e.
  E e^{λδ_i} ≤ e^{λ²σ_i²/2}. A uniformly random error direction gives σ_i = b_i/√d (the uniform
  sphere's coordinates are sub-Gaussian with proxy 1/d). Lloyd-Max's centroid condition (each level
  is its cell's conditional mean under the design distribution) is what makes (M2)/(M3)'s zero
  mean plausible for a rotated store. R5's hp record (the z = 5 bound exceeded on 0.45% of
  head-steps, fewer than a Gaussian with σ_i = b_i/√d would give at 128K rows) suggests that σ_i
  is, if anything, an overestimate.
- **Lemma 5 (concentration).** Under M, with probability ≥ 1 − δ: ‖o − õ‖ ≤ (m + r)/D with
  t_i = zσ_i, z = √(2 ln(4|T|/δ)); D = W_F + Σ_S w_i + Σ_T ŵ_i e^{−t_i};
  m = Σ_T ŵ_i(e^{min(σ_i²/2, t_i)} − 1)‖v̂_i − õ‖;
  B² = Σ_S (w_iν_i)² + Σ_T ŵ_i²[g_iν_i² + (g_i − 1)‖v̂_i − õ‖²], g_i = e^{min(2σ_i², 2t_i)};
  a = max(max_S w_iν_i, max_T ŵ_i(e^{t_i}ν_i + (e^{t_i} − 1)‖v̂_i − õ‖));
  r = La/3 + √((La/3)² + 2LB²), L = ln(4/δ); the S terms only with 4-bit read values.
  *Proof.* (i) Truncation: P(|δ_i| > t_i) ≤ 2e^{−z²/2} per row, so all of T stay within t_i except
  with probability ≤ δ/2; on that event Z ≥ D. (ii) Write the identity's right side as Σ X_i,
  X_i = w_i e_i on S, X_i = ŵ_i[e^{δ_i}e_i + (e^{δ_i} − 1)u_i] on T, u_i = v̂_i − õ (fixed given
  the conditioning; õ uses no unread row's true key or value, nor, with 4-bit read values, a read
  row's true value). Conditioned on the event, the X_i stay independent. Mean: E X_i = 0 on S;
  on T, E X_i = ŵ_i(E[e^{δ_i}] − 1)u_i, and since δ_i is symmetric, E[e^{δ_i} | |δ_i| ≤ t_i] =
  E[cosh δ_i | ·] ≤ min(E cosh δ_i, cosh t_i) ≤ e^{min(σ_i²/2, t_i)}, and ≥ 1 by Jensen; so
  ‖Σ E X_i‖ ≤ m. Second moment: the cross term vanishes by (M2), and E[e^{2δ}] ≤ g_i, Var(e^δ) ≤
  g_i − 1, so Σ E‖X_i − EX_i‖² ≤ B². Range: |e^{δ} − E e^{δ}| ≤ e^{t} − 1 on the event, so
  ‖X_i − EX_i‖ ≤ a. (iii) Pinelis' Bernstein inequality for independent zero-mean vectors in a
  Hilbert space (Ann. Probab. 22, 1994, Thm 3.4): P(‖Σ(X_i − EX_i)‖ ≥ r) ≤
  2exp(−r²/(2(B² + ra/3))) = δ/2 at the r above. (iv) ‖o − õ‖ ≤ (‖Σ EX_i‖ + ‖Σ(X_i − EX_i)‖)/Z. ∎
  Tests (synthetic data drawn from M, 2,400 head cases): no failure at δ = 0.3, 0.1, 0.01, 0.001;
  median bound/error 14 (conservative), 0.1 of Lemma 4. Negative control: with every key error
  along +q and value errors aligned, Lemma 5 fails on 332 of 640 cases and Lemma 4 on none, so the
  independence assumption carries real weight and has to be measured, not assumed.
- **One-pass form.** Replacing Σ a_i‖u_i‖ by √(Σa_i · Σa_i‖u_i‖²) and ‖u_i‖ ≤ ‖v̂_i‖ + ‖õ‖ in the
  max, every sum expands into running sums the attention pass already visits (‖u_i‖² = ‖v̂_i‖² −
  2⟨v̂_i, õ⟩ + ‖õ‖²), so a kernel can emit the bound with its output. Never below the two-pass form.

*What it predicts.* (1) With 4-bit values on S (`tail4_v4`, R5's tail arm) the S term ≈ the
read rows' own 4-bit error, about 3× FP8's per-row error, so the bound cannot reach FP8's level:
certifying at FP8 needs exact values on the read rows (`x`). (2) With `x`, the variance term
shrinks like 1/√(effective tail rows), so a flat tail certifies; the range term a is set by the
largest single tail row's e^{t}-inflated weight, so reading the rows of largest e^{ŝ_i + t_i}
exactly (the theorem's controller, with Lemma 5 as its bound) is what certifies. (3) The mean term
m is a real bias, not slack: E w_i = ŵ_i E e^{δ_i} ≥ ŵ_i, so unbiased key errors under-weight the
tail by ≈ σ_i²/2 on average. A design could add σ_i²/2 to tier-1 scores (computable from η_i and
‖q‖); it would remove the bias from the error but not from the bound (M gives only an upper bound
on E e^{δ}). Not tested.

*The probe's new columns* (per head and answer step; side file `s1h5_probe_heads`):
`err_tailx`/`rel_tailx` (the vote's rows with exact keys and values, the rest from tier 1);
`l4_tail`, `l4_tailx` (Lemma 4, worst-case b); `l5_tail`, `l5_tailx` (Lemma 5, σ = b/√d,
δ = 1e-3) with their parts `l5mean_*`, `l5B_*`, `l5a_*` (÷ D: the bound at another δ is
mean + bernstein_radius(B, a, δ/2)); `l5p_tailx` (one-pass); `trunc_viol` (share of unread rows
outside t_i); and, for `x` with C/128, C/32 or C/8 more rows read exactly per step (largest
e^{ŝ + t} share over the group), `err_tailx_p128|p32|p8` and `l5_tailx_p128|p32|p8`. Row
summaries in the results file: `probe_l4_*_ok`, `probe_l5_*_cover`, `probe_l5_*_fp8` (share of
head-steps where the bound is within FP8's own error there), `probe_err_*_fp8`,
`probe_trunc_viol_any`.

*For the run that uses it (to be frozen as its own amendment):* V = Lemma 4 holds on every
head-step; Lemma 5's coverage (error ≤ bound) per model × cell, with `trunc_viol` and the sign of
the realized mean term separating which part of M fails if coverage is below 1 − δ; the answer =
`probe_l5_*_fp8` at the vote's rows and with each extra-row read, against the extra rows' traffic.
Probe cost: about one more attention-sized product per design and per extra-row read (8 per
chunk), and about 2 GB of float64 temporaries at 128K (Llama, 4 steps per chunk).
