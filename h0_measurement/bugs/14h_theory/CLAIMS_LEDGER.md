# Claims ledger: the two-tier design paper (draft, 2026-10-10)

This is step 2 of `STATUS.md`'s ranking. It lists every claim the design paper could make, the evidence for each, and
whether the protocol that produced the evidence meets the assumptions of the theory the claim leans on. The paper
itself has no draft yet (proposed in `../../../weekly_reports/Oct4.md` §6). `latex/main.tex` is the understanding
paper, and §7 below divides results between the two.

Status labels:
- **CONF-PENDING:** a confirmatory test, frozen, whose result is not in yet (R6);
- **EXPLORATORY:** a frozen-reader run whose labels "guide the design and are not claims" (R1–R5.4). It can motivate a
  claim, or appear in a figure labelled exploratory, but cannot be the claim;
- **PROVED:** a theorem with a complete proof in `theory.md`, checked by brute force;
- **CPU-ONLY:** validated on CPU only (Llama-3.2-1B and three other architectures' layers, ≤ 8K tokens), not at the
  paper's models and lengths;
- **MISSING:** no evidence.

"Step n" refers to `STATUS.md`'s ranked next steps.

## 1. Working thesis

Store every row of the KV cache in a low-bit **randomized** (dithered) tier, read the rows the question's vote selects
exactly, and read everything else from the tier with weights corrected for the dither's bias. Nothing is evicted.
Three results support it:
- end to end, this is non-inferior to FP8 KV at 0.40 of FP16 traffic per decode step (R6);
- per attention call, its error is within a bound computed at run time, for any input and model (Theorem R);
- an exact formula predicts its error, and says when the design will work (Theorem M).

The design under test is `dtail6ux_v4`@1/8: 6-bit keys, 4-bit values, the unbiased tail, the vote's 1/8 exact
(`../14h_methodology_improve/glossary.md`).

## 2. Empirical claims

| id | claim | evidence (run · metric · statistic) | status | gaps, and the step that closes them |
|---|---|---|---|---|
| C1 | **Primary.** In each claim group (Llama-3.1-8B 128K; Qwen3-30B-A3B 32K; Qwen3-30B-A3B 128K), `dtail6ux_v4` is non-inferior to FP8 KV on KL per answer token in every testable family, at traffic 0.398 against FP8's 0.50 | R6 (`../14h_methodology_improve/plan.md`, "R6 frozen"; reader `read_stage1h_r6.py`) · KL per span token, teacher-forced · per claim group × family, one-sided 95% unit-bootstrap upper bound of mean(KL_design − 1.5·KL_FP8) < 5e-4; a family where `fp_noise` fails the same test is NOT_TESTABLE; the group is NONINFERIOR if every testable family passes and at least half the planned families are testable. 640 fresh units. Support: R5.4, where `dtail6ux_v4` passed 10/10 groups at this margin on 5–10 units each (`findings/R5_4.md` §5) | CONF-PENDING (Trillium half queued; Qwen 128K half not submitted) | Qwen 128K needs submitting [1]. One dither seed for every unit and arm (`run_s1h5.DITHER_SEED`), so the CI covers prompts, not dither draws; a heavy row at a fixed position (e.g. the attention sink) sees the same perturbation in all 640 units [3: seed panel] |
| C2 | **Key secondary.** No accuracy loss against FP8 on FP-correct units outside aggregation | R6 · free-running answer accuracy · one-sided 95% lower bound of mean(acc_design − acc_FP8) > −0.02, with ≥ 3/δ units, else ACC_UNDERPOWERED; also reported at 0.05 | CONF-PENDING | Qwen 128K is expected to be ACC_UNDERPOWERED at 0.02 (about 90 FP-correct units); testable at 0.05 |
| C3 | **Evicting nothing removes the vote's failures.** The old controller (vote-selected reads, rest evicted) loses on re-ranking (NDCG −0.17 Llama, −0.26 Qwen at 128K), multi-value and aggregation, and the loss is the vote's (re-ranking: vote +28.6 / +39.8 nats against a budget of +0.48 / +0.77). The tail design keeps the same selection and stays at FP8's level | R4 (`findings/R4.md`), the loss split (`findings/R5_0_split.md`), R5 and R5.3 (`findings/R5.md`, `R5_3.md`) | EXPLORATORY for the comparison; R6 adds confirmatory tail-design numbers in the same families, but has no old-controller arm | Present the old controller's failure as motivation (exploratory, labelled), and R6's tail-design result as the claim |
| C4 | **Against dense 4-bit stores** (uniform 4/4, KIVI-4; traffic 0.25), the tail design is closer to FP: dense 4/4 passes R6's margin in 1/10 R5.4 groups, the tail design 10/10 | R5.4 (exploratory); R6 reports dense 4/4, dense 4-bit keys and KIVI-4 (reported, not tested) | EXPLORATORY; R6's numbers will be descriptive | A claim against dense stores needs the paired test R6 does not run; state it as a frontier (traffic × KL) from R6's reported arms |
| C5 | **Against Quest at the same 1/8 read fraction**, the tail design has 15–80× lower KL per token on Llama 128K multi-value, tracking, aggregation, re-ranking, RAG and ICL (3–5× on retrieval), at higher traffic (0.297–0.398 against 0.094–0.188) | R5 (`findings/R5_reader.md`, "Accuracy and cost per arm"), with the Lloyd-Max tail (4-bit values on the read rows) | EXPLORATORY only; Quest is not in R6 | The Oct 4 report made "beats Quest" a gate for the design paper. As it stands it is an exploratory frontier point at unequal traffic. A matched-traffic comparison needs a run (not in the ranking) |
| C6 | **The dithered tier fixes the Lloyd-Max tier's coherent-bias failures** (Qwen multi-value unit 9422: KL/token 0.0049 → 0.00002; Llama re-ranking 0.0099 → 0.0074 at 4 bits, 0.0041 at 6) | R5.4 (`findings/R5_4.md` §3), a rerun of R5.3's prompts | EXPLORATORY; R6 has no Lloyd-Max tail arm | Motivation for randomizing tier 1, beside Theorem R; labelled exploratory |
| C7 | **Key bits trade traffic for accuracy** smoothly: 4 / 6 / 8-bit keys at traffic 0.344 / 0.398 / 0.453; re-ranking and aggregation improve with each step, and at 8 bits every group is at `fp_noise`'s level | R5.4 §4 (exploratory); R6 runs 6 (tested) and 4 (reported) | EXPLORATORY | Theorem M predicts this split; checking it at 128K is step 3, and linking it to KL is step 7 |

## 3. Theory claims

| id | statement (`theory.md`) | assumptions, checked against the protocol | validation | reported metric | gaps → step |
|---|---|---|---|---|---|
| T1 | **Eviction can't be certified in advance** (Lemma 1, §2): o − o_S = ε(ō_T − ō_S); a set chosen before the answer's queries exist can't bound ε. It works exactly when attention is concentrated and the question predicts it | none | R1–R4's pattern (single needles fine; multi-value, aggregation and re-ranking fail) | — | none; it motivates C3 |
| T2 | **Theorem R** (§6): with a dithered tier 1, the tail design's attention error at every head and step is ≤ a bound computed at run time, with probability ≥ 1 − δ, for every cache content and model with softmax attention over cached rows | (R1) rows read exactly are chosen without the dither's coins: **met in R5.4 and R6** (the vote reads a separate Lloyd-Max 4/4 store), **not met by a single-store deployment** whose vote reads the dithered tier [5]. (R2) queries don't depend on this layer's coins: met in teacher-forced passes by Lemma C [T5]. The seed is independent of the input: met (fresh prompts, fixed seed) | PROVED; brute force on adversarial data at every δ (`test_theory*`); 0 misses in 31,080,448 GPU head-step checks (R5.4, v1, 6- and 8-bit keys); R6's probe adds v1/v2 at 6 bits on 7 blocks; CPU coverage 1.000 on Qwen3-1.7B and R6's Qwen3-30B-A3B (`findings/findings_qwen.md`) | coverage (must be 1); bound / error; share of head-steps certified within FP8's error | GPU numbers exist for v1 and v2 only. R5.4, v1, 6-bit, vote's rows: bound/error 37–38 (Llama), 41–58 (Qwen); certified within FP8 0.44–0.52 / 0.38–0.41. The tight forms (T3) are CPU-ONLY [3] |
| T3 | **Tightened forms** (v3, sure sums, mirror split, and part 8's tilted tail with a Hsu–Kakade–Zhang key sum; §6, §6c–g): valid for every input; part 8 is tightest at every width (with Bennett kept as a second form at 4 bits) | as T2; the split and part 8 need independent key and value dithers (GQA: met); part 8's exact-law proxy uses the rotated query and the per-row steps, both available at run time | PROVED (part 8's proxies also checked against exact laws by quadrature); brute force, 0 failures; CPU real layers, part 8: 6-bit 6.7–7.3× the error and 0.41–0.59 certified within FP8 (Llama-1B L1/L3, 1.6K–8K; split 17–19× and 0.13–0.21); 8-bit 2.0–2.6× and 0.89–0.95. On Qwen's typical layers: 6-bit 4.0–6.5×; its high-σ early layers are 11–900× | as T2 | CPU-ONLY [3]. What is left at 6 bits is the sure weights (e^t ≈ 1.75); research, with [9] |
| T4 | **Theorem M** (§10): the tail's MSE, exactly, E‖N‖² = Σ_T w_i²(Var Z_i‖v_i − o‖² + E Z_i² E‖e_i‖²) + bias², with E‖o − õ‖² ≤ E‖N‖²/M² always. The unread rows' share of Σ w²‖v − o‖², not their mass, sets the error | independent key and value dithers (GQA: met); coin-free reads and queries (as T2) | PROVED; Monte Carlo agreement within 4.5 s.e.; CPU real layers: measured/predicted 0.98–1.00 at 6–8-bit keys on Llama, Gemma 3, GPT-OSS and Qwen3 (4-bit: 0.91–1.00 where σ ≤ 1.1, and 0.2–0.3 where σ ≈ 2, which is conservative); its per-layer map predicts R6's 6-bit design at ≤ FP8's per-head error in 32/32 Llama-8B and 47/48 Qwen3-30B layers; tail/dense MSE predicted 0.0186, measured 0.0187 (Llama-1B L1) | measured/predicted MSE; unread second-moment share; tail RMS / FP8 | CPU-ONLY [3]; no link to KL per token, the paper's primary metric [7]; MLA: done (coupled form, ≤ 0.1% coupling term) |
| T5 | **Lemma C and Theorem DD** (§11): with per-layer independent dithers, Theorem R holds layer by layer in any prefill or teacher-forced pass; in free-running decode, up to the first token that differs from the exact model's | per-layer seeds: met (`DITHER_SEED + 2·layer`, independent streams); context compressed after an exact prefill: met | PROVED; consistent with `adaptive_s1h5.py` (design-substituted queries changed no coverage on Llama-1B) | — | R6's KL is teacher-forced (Lemma C applies); its accuracy is free-running (DD applies up to the first divergence). State both in the paper |
| T6 | **Certified adaptive read** (§6e): a coin-free nested schedule, a union bound at δ/m, stop at the first set whose bound meets the target. With part 8's bound: FP8-level per-head error at traffic 0.433–0.499 with 6-bit keys (Llama-1B L1/L3 at 1.6K–8K, a 15-set schedule; part 7: 0.50–0.62) and 0.456–0.460 with 8-bit keys (the oracle stop: 0.453) | as T2, plus a coin-free schedule (sketch order: met). The target used so far, FP8's per-head error, is an oracle, not computable at run time | PROVED; brute force; CPU-ONLY evidence | rows read; traffic against 0.5 | at 6 bits it is at FP8's traffic, not clearly below it; L0 (flat attention) needs 0.73. A run-time τ [6]; GPU check [3] |
| T7 | **The 4-bit limit** (§12): with 4-bit keys no proof in this family reaches FP8's level (best: ≈ 100× the oracle at δ = 0.05), although the information is there (the oracle certifies 62%). The 4-bit design's accuracy is explained by T4, not certified | — | CPU (`findings/findings_4bit.md`) | — | state as a limitation; research [9] |
| T8 | **Generality across architectures**: for any softmax attention over a per-layer cache of rows, Theorems R and M hold for every input, given a subtractive-dithered tier, read sets chosen without its coins, and a dither per layer. That covers MHA, MQA, GQA, MLA, local:global, sinks, biases, gates, and the softmax layers of recurrent hybrids. Theorem M predicts the error of every key width, value width and read set from one FP pass, so bits are chosen per layer (Corollary M-B) and checked per call by Theorem R | as T2; excluded until extended: a sparse-attention selector that reads the dithered tier (R1), cross-layer KV sharing (R2: a later layer's query depends on the same coins), non-softmax or differential attention, recurrent states | CPU: 21 models; part 8 / certified read coverage 1.000; Theorem M 0.990–1.003 (GQA) and 0.98–1.01 (MLA); off-diagonal (k, v) within 2% where k ≥ 5 (`findings/findings_architectures.md`, `findings_bits.md`). `MODEL_FAMILY.md` collects it by family: which bits each kind of layer and model needs, and which attention mechanisms the theorems cover | bound / error; tail / FP8 per layer; allocation traffic | CPU-ONLY, ≤ 1.6K tokens; the per-layer choice depends on the input (calibrate on several; the run-time certificate covers the rest); no single uniform configuration (not even 6/6) reaches FP8's per-head error on every model: 6/6 misses 29 of 313 layers (`MODEL_FAMILY.md` §3) |

## 4. Systems claims

| id | claim | evidence | status | gap |
|---|---|---|---|---|
| S1 | **Traffic** per decode step: 0.398 (design), 0.344 (4-bit keys), against FP8's 0.50 | analytic model `s1h5_lib.traffic_frac`; excludes the question-time vote ("one 4-bit pass per question, not amortized") | analytic | The tail design has **no measured time**. The packed low-bit decode was compute-bound and slower than FlashAttention on FP16 rows (Stage 1e/1f, `../14_kernel_tpot/report.md`), and every Stage 1g timing is of the old evicting system. Traffic is not time; the paper must say so, or a kernel must exist |
| S2 | **GPU memory.** Tier 1 at 6/4 bits is 0.31 of FP16 K+V. A coin-free vote store adds 0.125 (4-bit keys only) to 0.25 (4/4, as R6 allocates it). The exact rows fetched for an active question add 0.125. That gives about 0.56–0.69 during a question, against FP8's 0.50, and 0.44–0.56 for a stored context | accounting from R6's arms; no reported metric includes the vote store | MISSING as a reported metric | Claim traffic, not memory; or remove the vote store (vote on exact keys at the question's prefill, or prove Theorem R for a vote over the dithered tier) [5] |
| S3 | **Host memory** holds the exact tier (FP16 K+V, 1.0) | design | stated | the paper's cost table must show it |

## 5. Theory against protocol: the checks this ledger ran

| check | verdict | action |
|---|---|---|
| R6's KL is teacher-forced over FP's answer span: queries depend on the coins of lower layers only | ✓ Lemma C covers it | state it in the theory section |
| R6's accuracy is free-running | ✓ Theorem DD, up to the first token that differs from FP's | state the scope; after a divergence the per-call guarantee needs the log-cost extension |
| The vote never sees the dither (R1) | ✓ in R6 (Lloyd-Max 4/4 store); ✗ in a deployment with one tier | step 5; until then, the paper states R1 as a deployment requirement and counts the store's memory |
| The seed is independent of the input | ✓ fresh prompts, fixed seed | state "seed drawn per request" as a deployment requirement |
| Dither variability is in the CIs | ✗ one seed for all units and arms | step 3's seed panel; state it in the methods |
| Theorem R's δ: per call (10⁻³ in the probe) or per decode | per call | a whole-decode statement runs each of N calls at δ/N (z grows from ≈ 3.9 to ≈ 6 at N = 10⁵); the paper states the per-call form and the N-call corollary |
| The certification target | FP8's per-head error, computed from exact data: right for an analysis metric, not computable by a deployed controller | step 6 for the certified read |
| Reported certificate numbers match the theory section's forms | ✗ GPU: v1/v2 only; theory: part 8 (and v3/split) | step 3 (step 4 is done) |
| Theorem M's scope covers the paper's models | ✓ GQA (Llama, Qwen3 MoE with QK-norm); MLA too, in its coupled form (§10; validated on three MLA models) | restrict the statement to independent key/value dithers, or step 8 |

## 6. Not claimed

- End-to-end quality from attention-level bounds. A size-τ change in one head can flip a near-tied token in any format,
  FP8 included; R6 measures end-to-end quality directly.
- Speed, until a kernel for the tail design exists (S1).
- FP8-level certificates at 4-bit keys (T7).
- That one configuration reaches FP8's per-head error on every architecture (T8). 6/4 fails flat layers; 6/6 fails
  layers with noisy scores (GPT-OSS needs 8/4) and some of the flattest (three Granite-4.2 layers need 4/8). The bits
  are chosen per layer (`MODEL_FAMILY.md` §3–§5).
- MLA, sliding-window and hybrid layers: the adapters cover them (`adapters_s1h5.py`), but no end-to-end run does.

## 7. Which paper owns what

The October 4 report warned that two papers under review must not present the same results as contributions.
- **Understanding paper** (`latex/main.tex`): the interior and SIEVE (F1–F7), plus measurement findings the report
  ranked for it: the question-time contract and reuse, the sequence proxy, the 3-bit key confusions, the likelihood
  metric, scoring order as a verdict-mover, and total bytes with 4-bit values. It must not present the two-tier design
  as a contribution.
- **Design paper**: the tail design (C1–C7), the theory (T1–T7) and the systems accounting (S1–S3). It cites the
  understanding paper for the question-time contract and the likelihood metric.

## 8. Decisions for the user

1. Submit R6's Qwen 128K half (C1, C2).
2. Is the certified adaptive read (T6) a contribution? Step 4 (done) brought it to about FP8's traffic at R6's key
   width on CPU. A run-time target (step 6) and the GPU check (step 3) remain.
3. Is "beats Quest" (C5) still a gate? If so, a matched-traffic run is needed; nothing in the ranking provides it.
4. Memory (S2): claim traffic only, or close step 5 first?
