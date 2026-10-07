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
