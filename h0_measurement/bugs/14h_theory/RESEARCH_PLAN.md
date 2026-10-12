# Research plan after R6 (2026-10-11)

*14h_theory. This is a recommendation for the user to approve or change. If approved, it replaces the ranking in
`STATUS.md` ("Next steps"), where steps 1, 2, 4 and 8 are done. It covers three things: a check of the colleague's three
proposed steps (§1), one ranked plan (§2), and the theory agenda (§3) and experiments (§4) behind it. It also says where
each part lives (§5), the decisions only the user can make (§6), and the order of work (§7).*

## 0. Where things stand
- **R6, the confirmatory run.** The design `dtail6ux_v4` uses 6-bit dithered keys and 4-bit values, reads the vote's
  1/8 exactly, and costs 0.398 of FP16 traffic. It is NONINFERIOR to FP8 in all three claim groups:
  - Llama-3.1-8B at 128K: 7 of 7 families;
  - Qwen3-30B-A3B at 128K: 3 of 3;
  - Qwen3-30B-A3B at 32K: 3 of 4 testable.

  Accuracy is ACC_PASS on Llama. Qwen is underpowered at δ = 0.02 and passes at 0.05.
  - The design is usually far below FP8: on Llama its KL per token is 3–22× lower.
  - The weak spot is re-ranking. On Llama the design's KL is 0.0021, against 0.0008 for rerunning FP and 0.0051 for
    FP8.
  - The 4-bit tail `dtail4ux_v4` (traffic 0.344) fails only re-ranking, on both models.

  Source: `../14h_methodology_improve/findings/R6_reader_partial_1011.md`.
- **Theorem R on the GPU.**
  - Coverage is 1.000, with no misses in 46.6 M checks: 16 read forms × 2.9 M head-steps.
  - The reported form, v2, is loose: median bound / error is 25.4× on Llama 128K and 37.9× on Qwen 32K.
  - 0.625 and 0.487 of head-steps are certified within FP8's per-head error.
- **Theory on the CPU.**
  - Part 8's bound is 6.7–7.3× the error at 6-bit keys (Llama-1B).
  - The certified read reaches FP8's per-head error at traffic 0.433–0.499 at 6 bits.
  - Theorem R's coverage is checked on 21 models, and Theorem M against Monte Carlo on 19 of them.
  - Corollary M-B gives bits per layer.
  - The model families are in `MODEL_FAMILY.md`.
- **Gaps the ledger tracks** (`CLAIMS_LEDGER.md`):
  - the tight bound forms, Theorem M and the certified read are CPU-only (T3, T4, T6);
  - R6 used one dither seed (C1);
  - the vote's store is in no memory number (S2);
  - the certified read's target is an oracle (T6);
  - there is no measured time (S1).

## 1. The colleague's three steps, checked

### 1.1 "Run R7: the tight theory at full scale, at 4/6/8-bit keys, with a dither-seed panel"
**What holds:**
- **It is the most important next step.** The theory section's headline numbers (part 8, Theorem M, the certified read)
  exist only on CPU proxies (1–2B models, ≤ 8K tokens). R7 puts them on the paper's own models and lengths.
- **Porting through a plan amendment** matches the folder rules. §5 proposes a new run folder instead of editing R6's
  frozen files.
- **A dither-seed panel** closes the ledger's C1 gap. R6's confidence intervals cover prompts, not dither draws.

**What does not hold, or needs care:**
- **"R6 has measured the gap this must close: 25–38× at 128K against 7× on the CPU."** This compares two different
  bounds.
  - Like for like, on Llama-3.1-8B itself, v2 is 27× the error at 1.6K, 24.6× at 16K (CPU) and 25.4× at 128K (GPU,
    R6). Scale does not hurt, but it doesn't help either. (This plan first compared Llama-1B on CPU with Llama-8B on GPU;
    corrected 2026-10-12 from `../14i_r7_theory_at_scale/STATUS.md`.)
  - On Llama-3.1-8B, part 8 / v2 is 0.19–0.22 at 6 bits.
  - Part 8 at 128K should therefore be about 5–6× on Llama. R7a's prediction P2 is 4–7× on Llama and 6–13× on Qwen
    32K. Qwen's noisy early layers will be much looser (11–900× on the CPU).

  So R7 confirms and reports; it is not expected to close a gap. That makes it a low-risk, high-value run.
- **"Port Theorem M's per-head prediction into the run-time code."** Theorem M needs the exact weights and values of the
  unread rows. It therefore belongs in the probe, which computes exact attention, not in the controller. A run-time
  version would be a plug-in estimate from tier-1 data. That is a new estimator, worth testing as a candidate for the
  run-time target τ (§3, τ).
- **"≈ 6–10 GPU-hours."** That is STATUS step 3's estimate. It was made before three costs were counted:
  - Qwen 128K's node jobs, about 4 GPU-h (R6's probe skipped them);
  - the 16-set certified read;
  - an end-to-end KL seed panel.

  R6's probe already cost 4–8× an arm (`s1h5_lib.py`), and part 8 is heavier than v2. Expect nearer 12–20 GPU-h,
  and pin it with a gate job, as in R6.
- **4-bit keys.** Run them for the frontier figure, but part 8 is 92–554× the error there on the CPU (T7). Don't expect
  any FP8-level certification at 4 bits.

**Verdict:** rank 1. Split it in two:
- **R7a:** the probe plus the seed panel, which nothing blocks;
- **R7b:** the design arms, after the CPU items they depend on.

### 1.2 "An exploratory per-layer bit-width arm in R7, aimed at re-ranking (≈ 0.36 traffic beats 0.398)"
**What holds:**
- Corollary M-B supports choosing bits per layer.
- The CPU allocation for R6's models is 0.354 (Llama-3.1-8B) and 0.374 (Qwen3-30B-A3B) at the p90 target, so "≈ 0.36"
  is right.
- It must be exploratory, and confirming it needs its own pre-registered run.

**What does not hold:**
- **"Aimed at re-ranking" points the arm at the one family it is most likely to fail.**
  - Re-ranking is the only family where 4-bit keys fail, on both models.
  - The CPU allocation puts 4-bit keys in 23 of Llama-3.1-8B's 32 layers. It was calibrated against FP8's per-head
    error, on PG-19, at 1.6K tokens.
  - FP8's per-head error is too weak a target for re-ranking. The design passes there at 6/4 with KL only 2.4× below
    FP8's, while 4/4 fails.
  - The input study already shows the PG-19 choice is fragile: on the code input it misses the target in 9 of 16
    Llama-3.1-8B layers (`MODEL_FAMILY.md` §5).
- **"Passes re-ranking ⇒ beats the 0.398 design."** To beat the design it must pass every family, and an exploratory pass
  cannot carry a "beats" claim.

**How to make it a real test:**
- calibrate at the run's length on separate calibration prompts from every family, re-ranking included, and take the
  per-layer maximum;
- use a KL-weighted objective once the KL bridge exists (§3, KL and WF);
- add uniform 5/4 (0.371) as a control, which separates "fewer bits" from "allocation";
- state the success criterion in advance.

**Verdict:** keep it, redesigned, as an exploratory arm in R7b (rank 5).

### 1.3 "Close the vote-store gap on the CPU before any memory claim; put the run-time τ in the same step"
**What holds:**
- R6 supports a traffic claim (0.398 against FP8's 0.5), not a memory claim.
- The coin-free store the vote reads is in no reported number. R6 allocates it at 4/4, which is 0.25 of FP16 K+V; the
  6/4 tier is 0.31 (ledger S2).
- Neither is the question-time vote pass in the traffic number. It is small: one 4-bit pass per question, so about
  +0.25/N per step for an N-token answer.

**What needs more:**
- **CPU accounting does not close the gap. It needs a design change or a theorem.** A concrete design closes it with
  only a small lemma: the **nested coin-free store** (§3, N).
  - Each row's 4-bit keys become a deterministic Lloyd-Max base, which the vote reads, plus a 2-bit residual quantized
    with subtractive dither. The tail reads base + residual: 6-bit keys, as now.
  - R1 then holds by construction. Memory is 0.31 at rest and about 0.44 during a question, against FP8's 0.50.
- **τ** is a separate problem: what the certified read aims at. It matters only if the certified read is a claimed
  contribution (decision D2). It falls out of the KL bridge as a per-head error budget.

**Verdict:** rank 2 for the nested store; τ is part of rank 3.

## 2. Ranked plan

The order is by importance to the design paper. Where a dependency forces a different order, §7 says so.

*Updated by §7.4 (2026-10-12, afternoon):*
- ranks 2 and 3 are done;
- R7b is built;
- the generalization stages of `ARCHITECTURES_PLAN.md` are in the ranking.

| rank | item | kind | folder | what it decides for the paper | cost | needs |
|---|---|---|---|---|---|---|
| 0 | **Close R6 formally**: the frozen reader over the final block set, archived | record | `14h_methodology_improve` | the final R6 numbers the paper cites | one reader run | the last blocks |
| 1 | **R7a: the theory at the paper's scale.** Part 8's bound, the certified read and Theorem M per head-step, in the probe, at 4/6/8-bit keys; a seed panel (§4) | GPU verification | new `14i_r7_theory_at_scale` | the theory section's numbers on its own models (T3, T4, T6); C1's seed gap | port and tests 3–5 days; 12–20 GPU-h | nothing |
| 2 | **N: the nested coin-free store**, with its lemma and Theorems R and M at per-coordinate steps (§3) | theory + CPU | `14h_theory` | a memory claim (0.31 at rest, about 0.44 during a question, against 0.50); R1 holds in the deployed system; the threshold read rule becomes provable | 2–4 days on CPU | nothing |
| 3 | **KL: bridge per-head error to KL per token**, with **τ** and the **water-filling allocation** (§3) | theory + CPU | `14h_theory` | explains R6's pattern (re-ranking, 4-bit keys); gives the allocation objective and a deployable τ | 1–2 weeks on CPU | nothing |
| 4 | **Draft the design paper**: the method, R6's table, and the theory section from `paper_theory_section.md`, with slots for R7 and R8 | writing | a new design-paper LaTeX folder | shows early which claims still lack evidence | ongoing | R6 |
| 5 | **R7b: the deployed design, confirmatory** (D7). The single store against FP8 under R6's rules; gates against SOTA baselines at matched traffic; exploratory arms: the certified read with run-time τ, the 4+3 store, water-filling, R6's design as reference (§4) | GPU confirmatory + exploratory | `14i_r7_theory_at_scale` (new `*_r7b` files) | the memory claim, confirmatory; the gates; whether the certified read is a contribution | about 45–55 GPU-h | 2, 3; R7a's gate; R6's final read |
| 6 | **R8: the cross-architecture prediction test**, pre-registered (§4) | GPU confirmatory | new `14j_r8_cross_architecture` | T8 end to end: Theorem M chooses bits on models R6 never saw, and the end-to-end result follows | 20–40 GPU-h plus run-time support for windows and sinks | 3; R7a's port |
| 7 | **Measured time**: a kernel for the tier-1 tail read | systems | `../14_kernel_tpot` or `../14b_*` | S1. A systems venue needs it; an ML venue can claim traffic and state the limitation | weeks | D1 |
| 8 | **Theory research** (§3): escalation (ESC), scaling with context length (SC), coin-dependent selection (SEL), tightness (TI), lower bounds (LB), decode beyond divergence (DD+), uncovered mechanisms (EXT) | theory | `14h_theory` | depth now; a follow-up theory paper later | open | R7a's data, for SC |

**Why this order:**
- **Rank 1:** R7a carries no design risk, needs no CPU prerequisite, and fills the most visible hole.
- **Ranks 2–3:** each is a few days of CPU work that changes what R7b and R8 should test. Run them before freezing R7b.
- **Ranks 5–6:** R8 is the bigger contribution, but it needs engineering in the run-time library (sliding windows,
  sinks, per-layer bit widths) and an objective from rank 3. R7b reuses R7a's port.
- **Rank 7:** venue-dependent. For a systems venue it moves to rank 2.

## 3. Theory agenda (all in `14h_theory`)

Each item gives the question, the approach, the CPU check, what counts as success, the risk, and which paper it serves.
The ones the design paper needs are **N, KL, τ and WF**. The rest build toward a follow-up theory paper.

### N. A nested coin-free store (design, plus Lemma N)
- **The design.**
  - Store each rotated key coordinate as a deterministic 4-bit Lloyd-Max base index plus a 2-bit residual, quantized
    with subtractive dither inside the base cell.
  - The residual's step is the cell's width / 3. The outer cell is closed by the row's stored maximum.
  - The vote, the threshold read rule and the certified read's sketch order read only the base, which is coin-free.
  - The tail reads base + residual: 6-bit keys, as now.
- **The vote's question pass.** R6's vote runs the question over a 4-bit store of keys *and* values
  (`run_s1h5.vote_sets`).
  - In the nested store it reads the base keys and the tier's own 4-bit dithered values.
  - R1 still holds layer by layer, because each layer has its own independent dither. Layer l's read set depends on the
    coins of lower layers only, which is Lemma C's argument.
  - The vote's quality must be checked: its keys are R6's, but its values are dithered rather than Lloyd-Max.
- **Lemma N.** Given the base index, each residual error is uniform on ±D_j/2. The errors are independent across
  coordinates and rows, and independent of the data.
  - This is Schuchman's theorem again. There is no overload, because the residual grid covers the cell plus a dither
    margin.
  - Theorems R and M then hold with per-coordinate steps D_j. Their proofs use the steps only through
    Σ_j (Rq)_j² D_j², Σ_j |(Rq)_j| D_j and the per-coordinate uniform laws.
- **What it closes:**
  - the vote store: memory is 0.31 of FP16 K+V at rest, and about 0.44 during a question with the exact 1/8 fetched,
    against FP8's 0.50;
  - R1 in the deployed system;
  - the unproven threshold read rule (theory.md §9).
- **What it costs in precision.** A back-of-envelope estimate, for Gaussian rotated coordinates: the RMS residual step
  is ≈ 0.11σ, against 0.09–0.11σ for a 6-bit uniform dithered key. That is at most about 0.3 bit. A 3-bit residual
  (7-bit keys, +0.027 traffic) is strictly finer.
- **CPU work:**
  - generalize `exact_coin_moments`, `second_moment` and part 8's proxies to per-coordinate steps;
  - compute Theorem M's prediction plus a 200-draw Monte Carlo on layers of the R6 models, GPT-OSS and Gemma 3;
  - check part 8's coverage at per-coordinate steps by brute force;
  - compare the vote's selection with R6's.
- **Success:**
  - the tail's error is within 10% of the 6-bit uniform tier's (or better at 4 + 3);
  - coverage is 1.000;
  - the vote's selection overlaps R6's on ≥ 95% of the read mass.
- **Risk:** low. **For:** the design paper.

### KL. From per-head error to KL per token
- **The question.** Does E[KL] ≈ ½·Σ_{l,h} c_{l,h}·MSE_{l,h} hold?
  - MSE_{l,h} comes from Theorem M.
  - c_{l,h} is the head's sensitivity, measured by injecting small isotropic noise into its output.
  - The formula is the second-order expansion of the KL in the logits, with linear propagation. Errors from different
    layers are independent and zero-mean (each layer has its own dither), so cross terms vanish in expectation.
- **CPU work:**
  - on Llama-3.2-1B (then Llama-3.1-8B at ≤ 8K if time allows), on task-like prompts;
  - measure KL per token for the dithered tail at 4, 6 and 8 bits, FP8 and dense 4/4, and compare with the prediction;
  - measure c per layer by noise injection, then per head by random probing.
  - FP8 and dense errors are deterministic, so their prediction needs the squared-mean term.
- **Success:** the prediction is within 2× of the measurement and orders configurations and families correctly.
- **What it pays off:**
  - it explains R6: why re-ranking is the weak family, and why 4-bit keys fail it (larger c, larger MSE, or both);
  - it gives the allocation objective (WF) and τ;
  - it gives a second-order corollary: per-head certificates imply a KL bound.
- **Risk:** medium. The expansion may be nonlinear at 4 bits. **For:** the design paper.

### τ. A run-time target for the certified read
- **Candidates:**
  1. a per-head error budget from the KL bridge, τ_h² = 2·B·s_h / c_h, with the budget B split over heads by WF;
  2. a per-layer quantile of FP8's per-head error from calibration prompts (a table lookup);
  3. a relative target, τ = ρ·‖o‖;
  4. the plug-in Theorem M estimate from tier-1 data (§1.1).
- **CPU check:** which candidate keeps end-to-end KL at or below FP8's with the fewest rows read (Llama-1B).
- **For:** the design paper, if D2 says the certified read is a contribution.
- **Status (2026-10-12, `findings/findings_klbridge.md` §6):**
  - the KL budget (candidate 1, per layer) is the rule;
  - a flat τ ties it on Llama-1B but loses 1.35–1.4× on Qwen3-1.7B;
  - the FP8 quantile (candidate 2) holds up end to end, 1.0–1.33× the KL budget's KL on Llama-1B, but its shape is
    1.6–1.8× worse on Qwen;
  - sensitivities can be pooled across families, but the level that meets a KL target must be calibrated per family;
  - noisy layers need 8-bit keys, or the certificate is vacuous there. Part 8's slack grows like e^{14σ_w}; at 32K,
    Qwen3-30B-A3B has 14 such layers (`findings/findings_nested.md` §7).

### WF. Allocation as reverse water-filling
- **The result.** Corollary M-B with the KL weights turns allocation into a convex program in the continuous relaxation:
  minimize traffic subject to Σ_l c_l·MSE_l ≤ B. With c(x) ≈ 4^(−x), its solution is
  k_l* = ½·log₂(c_l·a_l/θ) and v_l* = ½·log₂(c_l·b_l/θ) for one common water level θ, clipped to the available
  widths.
- This is rate-distortion's reverse water-filling, here for attention caches. Bits go where log-sensitivity is high,
  and none go where it is below the water level.
- **CPU check:** compare water-filling, the current FP8-target allocation and uniform widths, in predicted and measured
  KL at equal traffic.
- **What it gives:** R7b's per-layer arm its objective. **For:** the design paper, a short theorem.

### ESC. Escalating precision through residual planes (after N)
- More dithered residual planes per row, and a coin-free nested schedule over rows × planes. Theorems R and M hold for
  any per-row mix.
- **CPU check:** the certified read's traffic when it escalates precision, against escalating rows only.
- **For:** the design paper if it lowers the certified read's traffic; otherwise the follow-up.

### SEL. A certificate under coin-dependent selection (general research)
- **The setting:** the read set depends on the dithered tier. Examples are a vote over tier 1, or Quest-like page
  selection on quantized metadata.
- **The idea:** under a threshold rule, row i's selection depends only on row i's own coins. The unread rows' coins are
  then independent with truncated laws. Debias by the conditional mean and redo the concentration for bounded,
  conditionally independent variables.
- **Top-k rules** need a bias bound on the boundary rows.
- N makes this unnecessary for the design; it is a general result. **For:** the follow-up.

### SC. Scaling with context length
- **The observation.** On Llama-3.1-8B, v2's bound / error stays at 24.6–27× from 1.6K to 128K, while part 8's is flat at
  4.2–5.5× (Llama) and 11–15× (Qwen) from 1.6K to 16K (`../14i_r7_theory_at_scale/findings/explore_r7.md`).
- **The task.** Explain why the ratio is flat while certification improves with length (the vote leaves a smaller share
  of Σ w²‖v − o‖² unread). Predict how bound / error and Theorem M's terms scale with the context length C (the
  effective number of unread rows; the union terms' log C).
- **Data:** check on R7a's 32K and 128K results. **For:** both papers, wherever 1M-token claims come up.

### TI. Tightness
- At 6 bits, part 8's remaining slack is mainly the sure weights' e^t ≈ 1.75 (STATUS step 9).
- **Approach:** pay one row's deviation instead of a union (summand-level truncation, or inverting a confidence set).
- **Risk:** uncertain. **For:** the follow-up.

### LB. Lower bounds
- (a) A minimax lower bound for any certificate computed from tier-1 data at b bits. It would turn T7's "no proof in
  this family" into "no certificate can".
- (b) A lower bound on the traffic needed for FP8-level per-head error, given flatness and σ. It would show whether the
  two-tier design is near-optimal.
- **Risk:** uncertain. (b) is high value if it works. **For:** the follow-up.

### DD+. Free-running decode past the first divergence
- **The gap.** C2's accuracy and long reasoning outputs (`../14b_*`) decode past the first token that differs from the
  exact model's, where Lemma C and Theorem DD stop.
- **Approach:** a log-cost union over histories, or a coupling argument. **For:** the follow-up; it becomes relevant to
  the design paper if reasoning models enter it.

### EXT. Mechanisms not yet covered (`MODEL_FAMILY.md` §6)
- **Sparse selectors (DSA):** R1 holds if their index keys are not dithered. This is a short corollary.
- **Cross-layer KV sharing:** R2 needs an adaptive-query bound. `adaptive_s1h5.py` found no change in coverage
  empirically.
- **Differential and non-softmax attention:** a joint second moment.
- **MLA's bit grid:** the coupled Corollary M-B (`layermap_s1h5.py` takes GQA views only).
- **For:** the follow-up.

## 4. Experiments

### R7a: the theory at the paper's scale (new folder `14i_r7_theory_at_scale`)
- **The port.** New modules with unique names and self-contained brute-force tests; nothing imports `14h_theory`. It
  carries:
  - part 8's bound: the HKZ key sum, the exact proxy, and remedy 1, with Bennett as a second form at 4 bits;
  - the certified read with the 16-set schedule;
  - Theorem M per head-step (probe only);
  - the unread share of Σ w²‖v − o‖² per head-step.

  GPU notes are in STATUS step 3: power iteration for ‖M‖_op, and the exact proxy for 16 rows per head-step in fp64.
- **Blocks.** Fresh prompts on R6's seven probe kinds (Llama 128K, Qwen 32K). Qwen 128K is optional (D5).
- **Key widths.** 4, 6 and 8 bits, in the probe: one forward pass serves many dither draws.
- **Seed panel:**
  - (a) coverage over ≥ 5 dither draws per head-step, in the probe, which is cheap;
  - (b) KL over 4 extra seeds for the design arm on a subset of units (e.g. re-ranking, multi-value and retrieval, about
    10 units each per group). This gives the variance split and a two-level bootstrap of C1's statistic.
  - Compared arms run in one process; forward passes are not reproducible across processes.
- **Expectations, stated before the run so that it is a test:**
  - coverage 1.000;
  - part 8's median bound / error at 6 bits about 4–6× on Llama 128K and 6–10× on Qwen 32K;
  - a certified share well above v2's 0.625 / 0.487;
  - Theorem M's measured / predicted 0.98–1.02 at 6–8 bits;
  - the unread second-moment share orders the families by the design's KL, re-ranking highest.
- **Gate first, as R6 did.** About 12–20 GPU-h.

### R7b: the deployed design, confirmatory (spec for the R7b session; decided 2026-10-12)
This section is written so that a session can build R7b from it directly. §8 lists the build.

**What R7b decides.**
- **Primary, confirmatory (D7, user, 2026-10-12).** The single store, the nested coin-free store of theory.md §13, is
  non-inferior to FP8 KV on KL per answer token in each claim group, under R6's rules. At R6's traffic (0.398) and the
  store's memory (0.31 at rest, about 0.44 during a question, against FP8's 0.50), this makes the paper's memory claim
  confirmatory.
- **Key secondary:** accuracy, under R6's rule.
- **Gates (D3):** pre-registered secondary tests against SOTA baselines at matched traffic. Recommended below; the R7b
  session freezes them.
- **Exploratory, reported but not tested:** the certified read with a run-time τ, the 4+3 store, per-layer
  water-filling, and R6's own design as a paired reference.

**The tested arm.** Name it in R6's grammar, e.g. `dnest42ux_v4`.
- **Keys.** Per row, γ = ‖k‖ and y = (k/γ)Rᵀ, where R is the tier's rotation.
  - Base: the 4-bit Lloyd-Max index, using `quant.quantize_keys`'s codebook and midpoint thresholds.
  - Residual: a 2-bit subtractive-dithered uniform quantizer on the base cell, with the outer cells closed at
    ±max_j |y_j|.
  - The dither comes from the counter-based generator, keyed as R6's tier's.
- **Values:** R6's 4-bit subtractive-dithered tier (`rquant`).
- **Read set:** the vote's 1/8, by R6's rule (`run_s1h5.vote_sets`).
  - The question's last 32 rows run over the store, and SnapKV's score picks the top C/8 rows per KV head
    (`s1d_lib.rows_scores`).
  - The store is now the base keys (norm-corrected Lloyd-Max, the same quantizer as R6's vote store) plus the tier's own
    4-bit dithered values. There is no separate vote store.
- **The tail:** unbiased, debiased with per-coordinate steps (theory.md §13). Fixed rows are exact, and the read set's
  rows come exact from tier 2.
- **Traffic:** 0.398, as R6's design. The store keeps one more fp16 scalar per row (the norm γ) than R6's tier.
- **Memory:** tier 1 is 10 bits per element pair (0.3125 of FP16 K+V) plus per-row scalars. It is about 0.44 during a
  question with the exact 1/8 fetched, against FP8's 0.50. Report it as a measured accounting.

**Rules: copy R6's (`read_stage1h_r6.py`, frozen 2026-10-10).**
- **Claim groups:** Llama-3.1-8B at 128K, Qwen3-30B-A3B at 32K, Qwen3-30B-A3B at 128K.
- **Primary test.** Per claim group × family, the one-sided 95% unit-bootstrap upper bound of
  mean(KL_design − 1.5·KL_FP8) must be below 5·10⁻⁴.
  - A family is NOT_TESTABLE if `fp_noise` fails the same test.
  - A group is NONINFERIOR if every testable family passes and at least half the planned families are testable.
- **Key secondary.** The lower bound of mean(acc_design − acc_FP8) over FP-correct units must exceed −0.02, with at least
  3/δ units (else ACC_UNDERPOWERED). Also report it at 0.05.
- **Prompts:** fresh, with no (task, prompt) from R1–R7a. HELMET takes the next unused manifest items.
- **Sizes:** from R6's per-unit variance. R6 had at least 30 per family; the CPU ratio of KL to R6's design is 1.14×.
- **Seeds (recommended).** Draw the dither seed per unit, so that the confidence interval covers dither draws: C1's gap,
  which R7a's panel (b) measures.
- **One process per unit:** FP, FP8, `fp_noise` and the design together, because forward passes are not reproducible
  across processes.

**Gates (D3), recommended as pre-registered secondary tests.**
- **The test.** Per claim group, over all its units, the paired one-sided 95% upper bound of
  mean(KL_design − KL_baseline) must be below 0. Holm-correct over the baselines. The paper says "beats X" only where
  the test passes.
- **(ii) Dense quantizers at no more than the design's traffic:**
  - dense dithered 7/6 (0.406), a new arm: "two tiers beat one". On CPU, dense 6/6 had 3.9× the design's KL;
  - KIVI-4 (existing);
  - dense 4/4 Lloyd-Max, which is TurboQuant's MSE variant (existing).
- **(iii) Quest at matched traffic 0.398** (the existing `quest` family, with its budget set by `traffic_frac`).
- **Optional:** TurboQuant's product variant (a 1-bit QJL residual on keys), the closest unbiased comparator. Report it
  if built.

**Exploratory arms, on a unit subset (e.g. 10 per family).**
- **E1, the certified read with a run-time τ, on the single store.** It walks the coin-free FINE schedule (R7a's
  `cert_r7` has the certified read and part 8). It stops each (KV head, step) at the first set whose part-8 bound meets
  τ_l for every query head of the group, because the group shares the rows read. Check that R7a's port stops the same
  way. The CPU checks of 2026-10-12 (`findings/findings_klbridge.md` §6.5) set the rest:
  - **τ rule: the KL budget, τ_l² = 2B/(L·H·c_l).**
    - c_l is the design's own tail sensitivity, pooled over 5–10 calibration prompts across families.
    - A flat τ ties it on Llama-1B but is 1.35–1.4× worse on Qwen3-1.7B.
    - The FP8 quantile, which needs no calibration, is the fallback: 1.0–1.33× the KL budget's KL on Llama-1B, and
      1.6–1.8× worse in shape on Qwen.
  - **Levels: two fixed B per model,** chosen before the test blocks for about +0.002 and +0.01 traffic above E1's
    floor (the design plus the loose layers' extra key bits). Report the
    (traffic, KL) pairs. Claim no KL target: the KL a given B reaches varies about 10× across families.
  - **Noisy layers:** 8-bit keys (a 4+4 store) where part 8 at the store's width exceeds about 20× the error on
    calibration prompts at the run's length (σ_w ≳ 0.15 at 6-bit keys; `nested_s1h5.py depth`). From R7b's 32K
    captures (`findings/findings_nested.md` §7):
    - Llama-3.1-8B has none (3.6–7.6× everywhere);
    - Qwen3-30B-A3B has 14 of 48: L0, 1, 3, 5, 7, 8, 9, 12, 16, 19, 20, 28, 32, 40, at 20–1,960×. With 4+4 every one
      is within 8.6×;
    - recheck the list at 128K.
    - Without them, part 8 is vacuous there and the read falls back to exact rows. On Qwen3-1.7B the traffic was
      0.444–0.480 instead of 0.413–0.421.
    - The cost on Qwen3-30B-A3B is +0.016 traffic at the vote and about +0.02 of FP16 K+V in memory, and E1's store
      then differs from the primary arm's in those layers.
  - Report KL and traffic against the single-store design and FP8.
  - **CPU, the deployed combination:**
    - Llama-1B: 5.6× and 25× below the single store's KL at +0.001 and +0.007 traffic;
    - Qwen3-1.7B with 8-bit noisy layers: 2.9× and 5.3× below at +0.001 and +0.009.
- **E2:** the 4+3 store (traffic 0.426). On CPU it has 0.57× the KL of uniform 6/4.
- **E3:** per-layer water-filling, with inputs weighted equally and calibrated on separate prompts, against a uniform
  5/4 control. CPU: −16% KL at equal traffic.
- **E4:** R6's design `dtail6ux_v4` (uniform tier plus separate vote store), as the paired reference for the single
  store's cost (CPU: 1.14×).

**Calibration stage** (for E1's KL-budget τ and for E3). It runs on separate calibration prompts, never test units, per
claim group:
- E1: the per-layer sensitivity c_l of the design's own tail, one single-layer arm per layer (L arms per prompt). It is
  pooled over 5–10 prompts across families: per-family c_l gains at most 10% on CPU;
- E3: c^K and c^V, from single-layer key-only and value-only dithers (2L arms per prompt; theory.md §14);
- Theorem M's per-layer MSE.

It outputs the per-layer τ table and the water-filling allocation, frozen before the test blocks.
- The FP8-quantile τ, E1's fallback, needs no calibration run: R7a's probe gives the per-head FP8 errors per layer at
  scale.
- A KL target, if ever wanted, would need B fitted per family on at least 8–10 prompts each. E1 reports a frontier
  instead.

**Predictions** (from CPU, 8 seeds, each design with its own vote; `findings/findings_nested.md` §6):
- the single store's KL is about 1.1–1.3× R6's design per unit in geometric mean (Llama-1B 1.27×, Qwen3-1.7B 1.12×),
  and about 0.9–1.15× on mean KL, the quantity the primary test uses. At 32K on R7b's own models, Theorem M's KL proxy
  gives 1.31× (Llama-3.1-8B) and 1.24× (Qwen3-30B-A3B) (`../14i_r7_theory_at_scale/r7b/findings/depth_r7b.md`);
- its vote with dithered values reads as R6's Lloyd-Max-store vote does: 98.7–98.9% of rows in common, and KL 0.99–1.01×;
- R6's design was 3–22× below FP8 on Llama, so NONINFERIOR is expected in every group;
- the risk is re-ranking: R6's design was 2.4× below FP8 there, and with ×1.3 it is still about 1.8× below.

**Dependencies.** Build now; freeze after three things:
- (a) R7a's pilots and gate, which give the certified read's GPU cost per unit, and so whether E1 fits;
- (b) this session's CPU checks: **done 2026-10-12.** The vote's question pass with dithered values is neutral, and
  E1's τ rule, levels and noisy-layer keys are set (§8);
- (c) R6's final read.

Submission waits for the user's go-ahead.

**Cost.** R6 was about 38 GPU-h for 6 arms. The confirmatory core (FP, FP8, `fp_noise`, the design) on R6-size units is
about 25 GPU-h. Gate arms on all units add 15–20, and exploratory arms on subsets add 5–10. The total, about 45–55
GPU-h, is to be measured by R7b's pilots.

### R8: the cross-architecture prediction test (new folder `14j_r8_cross_architecture`)
- **Models:** Gemma-3-12B (flat global layers) and GPT-OSS-20B (noisy scores, sinks), at 32K, with R6's families and
  margin.
- **Pre-registered predictions,** from a calibration map at 32K:
  - Gemma 3: 6/4 fails and 6/6 passes;
  - GPT-OSS: 6/6 fails and 8/4 passes;
  - the allocation passes at its predicted traffic.
- **Needs** run-time support for sliding windows, sinks and per-layer bit widths.
- **Confirms** T8's "the formula picks the bits". About 20–40 GPU-h plus the port.
- **After D7** (`ARCHITECTURES_PLAN.md` §5, G2), R8 should test the deployed single store, not the uniform tier.
  - Restated predictions: Gemma 3 passes at 4+2/6 and fails at 4+2/4; GPT-OSS passes at 4+4/4 and fails at 4+2/6.
  - This is an amendment for the user.
  - `ARCHITECTURES_PLAN.md` is the plan for extending R7a and R7b to every architecture.

## 5. Where each part lives

| part | kind | folder |
|---|---|---|
| N, KL, τ, WF, ESC, SEL, SC, TI, LB, DD+, EXT: proofs, CPU checks, tests, findings | theory and its CPU checks | `14h_theory` (the user's rule of 2026-10-10) |
| R7a/R7b: the run-time port (new module names), GPU tests, plan, gate, submit scripts, reader, findings | GPU verification and exploration | **new** `14i_r7_theory_at_scale`; it imports R6's frozen library read-only |
| R8: run-time support for new architectures, pre-registration, runs, reader | GPU confirmatory | **new** `14j_r8_cross_architecture` |
| R6's code, plan and findings | the confirmatory record | `14h_methodology_improve`, untouched (`findings/R6_code.sha256`) |
| the kernel and measured time | systems | `../14_kernel_tpot` (Stage 1e/1f) or `../14b_*` |
| the design paper | writing | a new LaTeX folder (`latex/main.tex` is the understanding paper) |

**This needs a rule change.** The current rule (this folder's and METH's `CLAUDE.md`) says a new bound reaches
`cert_s1h5` only through a `plan.md` amendment in METH. A new run folder keeps R6's frozen files byte-identical. The
rule would then read: "a new bound reaches the run-time library through a run folder's plan". The isolation stays: no
run folder imports `14h_theory`, and module names stay unique across folders. This is decision D6.

## 6. Decisions (user, 2026-10-11)
- **D1 Venue:** ML venues. Measured time (rank 7) becomes a stated limitation, not a gate.
- **D2 The certified read as a contribution:** not decided. Rank 3 gives it a run-time τ (theory.md §14), so it can be
  one.
- **D3 Gates:** "beats Quest" is a gate if it makes sense, alongside other SOTA baselines. Recommended for R7b, all at
  matched traffic (or matched memory for eviction):
  - **Already in the run-time library:** FP8 KV; dense 4-bit (rotated Lloyd-Max, which is TurboQuant's MSE variant);
    KIVI-4; KVQuant (sievelib); Quest (R4/R5's `quest` family).
  - **To add:**
    - a dense tier at the design's traffic (dithered 7/6), which tests "two tiers beat one";
    - TurboQuant's product variant (1-bit QJL residual for unbiased inner products), the closest unbiased comparator;
    - SnapKV, for eviction at matched memory (R12 had it).
  - **Related work rather than arms** (implementation cost): MagicPIG (LSH sampling, unbiased estimation) and ShadowKV
    (low-rank keys on GPU, offloaded values, sparse fetch).
  - **Proposed gates:**
    - (i) non-inferior to FP8 (R6: done);
    - (ii) lower KL than every dense quantizer at no more than its traffic;
    - (iii) lower KL than Quest at matched traffic.

    CPU evidence for (ii): dense 6/6, at about the same traffic as the design, has 3.9× its KL (Llama-1B, 8 units).
- **D4 Memory:** claim memory and traffic, through the nested store (rank 2, done on CPU: `findings/findings_nested.md`).
- **D5:** Qwen 128K is in R7a's probe.
- **D6:** new run folders, with R7 split into R7a and R7b.
- **D7 (2026-10-12):** R7b's single-store arm is **confirmatory**, under R6's rules (§4 R7b).

**Status of ranks 2 and 3 (CPU, 2026-10-11):**
- **Rank 2 (the nested store).**
  - Lemma N and Theorem N are proved and checked: simulation, brute force, 13 real layers.
  - End to end on Llama-1B, its KL is 1.14× uniform 6/4's and 1.5–18× below FP8's.
  - Memory 0.31 / 0.44. CLAIMS_LEDGER T9, S2.
- **Rank 3 (KL, WF, τ).**
  - Proposition KL: the structured bridge predicts KL within a median 1.00–1.34×.
  - Corollary WF: −16% KL at equal traffic, held out, with inputs weighted equally.
  - Key errors cost 2× value errors per unit MSE.
  - τ: the KL-budget rule meets its budget end to end within 15%. It cuts the 6/4 design's KL 3.7–14× for
    +0.0004–0.003 traffic (`findings/findings_klbridge.md` §4). CLAIMS_LEDGER T10.
  - Qwen3-1.7B: the bridge holds with keys ≥ 5 bits and fails at 4-bit keys (Theorem M's first-order limit).
- **What this changes for R7b's arms:**
  - the nested-store arm is now the memory claim's end-to-end test;
  - the per-layer allocation arm should use water-filling with inputs weighted equally. Expect at most about 20% lower
    KL at equal traffic, against a uniform 5/4 control;
  - the probe should record the bridge's per-layer sensitivities, so that §14 can be checked at 128K;
  - add a certified-read arm with the KL-budget τ. On CPU it is the cheapest way to cut KL, and it would make the
    certified read (D2) a contribution.

## 7. Order of work and dependencies (updated 2026-10-12, afternoon)
This replaces the morning's order. It includes the generalization stages G0–G5 of `ARCHITECTURES_PLAN.md`.

### 7.1 Where each step stands

| step | state | blocked on |
|---|---|---|
| R6's final read | 31 of 32 blocks in `results/` | Rorqual's Qwen 128K block w-hmc.1 (prompt offset 36). The user checks Rorqual and copies it, or reruns it; there is no SSH route from Trillium |
| R7a | frozen; its gate passed (1080291, 08:52) | its 13 blocks and reader were cancelled from the user's account at 09:23, before starting. Resubmitting needs only the user's go-ahead (the code pin verifies) |
| R7b | run side built and CPU-tested, a DRAFT (`../14i_r7_theory_at_scale/r7b/`) | the user's decisions in `r7b/plan_r7b.md` §8 |
| G0–G5 | planned (`ARCHITECTURES_PLAN.md`) | — |
| the design paper | `latex_paper/icml27/` holds only the ICML template | — |

### 7.2 Dependencies
Two kinds:
- **Hard:** the step cannot start, or its result would be invalid, without it.
- **Soft:** the step is better with it, or a substitute exists.

| step | hard | soft (and its substitute) | unblocks |
|---|---|---|---|
| R6's final read | the Rorqual block | — | the paper's final R6 numbers; R7b's freeze (soft) |
| R7a's blocks | the user's go-ahead | — | the theory section at scale (T3, T4, T6) and the seed panel (C1); SC; R7b's E1 inputs (soft) |
| R7b's decisions | — | — | R7b's freeze |
| R7b's core: pilot, gate, 750 units, reader | the decisions; TurboQuant's product variant built, if it is chosen | R6's final read: cheap, so do it first | the memory claim and the gates; which design R8 tests |
| R7b's calibration and exploratory stages (E1–E3) | R7b's freeze | R7a's probe blocks, for two inputs: E1's loose layers at 128K (substitutes: a CPU capture at 128K, or the 32K list) and E1's GPU cost (substitute: an E1 pilot) | D2: whether the certified read is a contribution |
| G0, the CPU map at 32K | code: a 32K capture driver for the adapters, and Theorem M's grid in the store's form (2–3 days) | — | R8's predictions (G0a); G3's and G4's maps; the slack law across architectures |
| G1, the run-time adapter | — for its design-agnostic parts: global-layer selection, the sink column, refusals, truncated-model smokes | G0 (the widths it must support); R7b's core read (the design it ports) | R8 |
| G2 (R8) | G0a (Gemma-3-12B, GPT-OSS-20B); G1; the user's approval of the restated predictions; R7b's core read, which decides the design R8 pre-registers (if the single store fails, R8 tests R6's design with its original predictions) | R7a: whether R8 carries the probe | T8 end to end |
| G3 (MLA) | the MLA theory items (the coupled grid, Lemma N for a latent); G0c; a latent-cache module; G1 | R8's outcome (go or no-go) | MLA end to end |
| G4 (hybrids) | G1; G0's map of the hybrid model | R8's outcome | hybrids end to end |
| G5 (theory) | the slack law across architectures needs G0's captures; SC needs R7a's data; the rest needs nothing | — | the follow-up theory paper; G3, through the MLA items |
| the design paper | nothing, to start | the final numbers: R6's final read, R7a, R7b's core (also R7b's exploratory stage for E1, and R8 for T8 end to end) | — |

- **G0–G5 do not need R7a's or R7b's results.** The theorems already transfer layer by layer (`ARCHITECTURES_PLAN.md`
  §1), so G0 measures validity and the per-layer design point on CPU without any GPU result. R7b's result decides only
  which design G1 and R8 carry.
- **A constraint, not a dependency.**
  - R7a's code pin covers 48 files of `../14h_methodology_improve/` and 19 of `sievelib/`. R7b's code manifest covers
    R6's run-time chain.
  - So after R6's final read, those files stay byte-identical until R7a's and R7b's last blocks have run.
  - G1 wraps them, as R7a and R7b do.

### 7.3 Order
Two lanes run in parallel: GPU runs, and CPU, engineering or writing work. The user's decisions gate the GPU lane.
1. **Now, the user:**
   - copy Rorqual's last R6 block, then run the frozen reader once (R6's final read);
   - resubmit R7a, or say why it was cancelled;
   - decide R7b's open items (`r7b/plan_r7b.md` §8); then R7b is frozen and its core submitted.
2. **Now, no GPU:**
   - G0's code, then G0a (R8's two models), then G0b (the other GQA and MHA models);
   - the design paper's skeleton, with R6's numbers;
   - the MLA theory items, which gate G0c and G3.
3. **When R7a's blocks have run:**
   - R7b's calibration and exploratory stages, with R7a's 128K loose layers and E1's measured cost;
   - SC.
4. **When R7b's core read is in:**
   - G1's design-specific parts;
   - R8's freeze (with G0a and the user's approval of its predictions), then R8.
5. **After R8:** G3 and G4 on a go; the theory follow-up (G5).

GPU over the next weeks: R7a 21–34 GPU-h, R7b about 58 (its core 22), R8 20–40.

### 7.4 Ranking by importance and impact
The ranking is for the design paper at an ML venue (D1). Impact is what the step decides for the paper's claims.

| rank | step | impact | cost | ready |
|---|---|---|---|---|
| 1 | R6's final read | the paper's only confirmatory numbers, final | one reader run | after the Rorqual copy |
| 2 | R7b's core | the memory claim (0.32 of FP16 K+V at rest), confirmatory; the gates against SOTA baselines at matched traffic | 22 GPU-h | after the decisions |
| 3 | R7a | the theory section's numbers on the paper's models and lengths; the seed panel | 21–34 GPU-h | now |
| 4 | the design paper's draft | shows which claims still lack evidence; the long pole before a deadline | ongoing | now |
| 5 | G0 | T8's validity on 21 models at 32K instead of 1.6K; R8's predictions | 2–3 days of code, 1–3 CPU node-days | now |
| 6 | R7b's exploratory stage | D2 (E1: the certified read as a contribution), the allocation (E3), the 4+3 store (E2) | 36 GPU-h | after R7a, or with the substitutes |
| 7 | G1, then R8 | "the formula picks the bits" end to end on models R6 never saw: the generality contribution | 1–2 weeks, plus 20–40 GPU-h | after G0a and R7b's core read |
| 8 | the theory items | MLA (unblocks G3); the slack law across architectures (after G0); SC (after R7a); the rest for a follow-up | open | mostly now |
| 9 | G3 (MLA), G4 (hybrids) | end-to-end coverage beyond GQA | weeks, plus 25–35 GPU-h | after R8 |
| — | measured time (a kernel); R14b's reasoning models | not needed for this paper | — | — |

- Measured time: D1 makes it a stated limitation.
- R14b's reasoning models need decode-time compression of generated tokens, which the protocol does not do.
- R7a ranks below R7b's core by impact, but it is ready now and needs no decision, so it goes first.

## 8. Handoff: building R7b
Start from §4 "R7b" (what to run and the rules), §6 (D3 gates, D7), and the R7 folder's `CLAUDE.md` (its isolation
rules).

**Built 2026-10-12: R7b's run side, in `../14i_r7_theory_at_scale/r7b/` (a DRAFT, `r7b/plan_r7b.md`; nothing
submitted).**
- **Files:**
  - `lib_r7b` (presets, layout, seeds);
  - `arms_r7b` (the single store with its vote and debias, dense 7/6, E1's certified read, the calibration arms);
  - `run_r7b` (the driver; it wraps `run_s1h5` as R7a's does);
  - `tasks_r7b` and `make_manifest_r7b` (HELMET: R6's 41 items plus fresh items 41–110; msmarco has only R6's 41);
  - `read_r7b` (R6's rules, the Holm gates and E4's ratios; `--gate`, `--explore`);
  - `calib_r7b` (stage 1: c_l, the loose layers, E3's allocation; stage 2: E1's levels);
  - `submit_r7b.slurm` and `script_r7b.sh` (the chain);
  - `test_run_r7b`.
- **Location:** a subfolder, not top-level `_r7b` files, because R7a's code manifest hashes the parent's top-level
  `*.py` (`r7b/CLAUDE.md`).
- **Checks:**
  - `test_run_r7b --fast` passes;
  - the CPU driver smokes on Llama-3.2-1B at 4K pass (`r7b/STATUS.md`): the core plan on RULER and on HELMET (R7b's
    manifest), calibration stages 1–2 with `calib_r7b`, the exploratory plan and the readers.
- **One change to §4's E1 levels.** "+0.002 and +0.01" are measured from E1's own floor, not from the design's 0.398. The
  floor is the design plus the loose layers' two extra key bits: about +0.016 on Qwen3-30B-A3B at 32K. Measured from
  0.398, both targets would sit below Qwen's floor and collapse to one level.
- **Before the freeze:** the user's decisions in `r7b/plan_r7b.md` §8.

**Changed since the R7b session last read this (2026-10-12, later the same day):**
- **E1 is now specified** (§4): the KL-budget τ with pooled sensitivities, two fixed levels, the stop rule per KV
  group, and 8-bit keys in the loose layers.
- **The loose-layer list comes from part 8 on `r7b/.views`** (R7b's own 32K captures, read-only;
  `findings/findings_nested.md` §7):
  - Qwen3-30B-A3B has 14 loose layers, not 2;
  - Llama-3.1-8B has none.
- **The predictions use 8 seeds:** the single store at 1.1–1.3× R6's design per unit. `r7b/findings/depth_r7b.md`'s
  32K KL proxy agrees: 1.31× and 1.24×.
- **The inputs below are done.**

**Where.**
- New files with an `_r7b` suffix in `../14i_r7_theory_at_scale/`, so R7b can reuse R7a's `cert_r7` and `tier_r7` (the
  certified read, part 8, Theorem M, the GPU forms).
- R7a's 119 pinned files (`findings/R7a_code.sha256`) must not change until R7a has output.
- Nothing there may import `14h_theory`. Port by copying, naming the source, and checking parity on frozen fixtures, as
  R7a did.

**Port from `14h_theory` (sources and what they do):**
- `nested_s1h5.py`:
  - `codebook`;
  - `nested_quantize` (the store: base index, cell, dithered residual, the base sketch, per-coordinate steps);
  - `coin_moments_pc`, `score_sigma_pc`, `score_bound_pc` and `exact_coin_moments_pc` (the per-coordinate coin
    quantities every bound and the debias need).
- `theory5_s1h5.exact_rows` (2026-10-11 version): part 8's exact-law proxy with per-coordinate steps (the `step.dim() == 3`
  branch). R7a's copy has the scalar path only.
- `test_nested_s1h5.py`: Lemma N by simulation, the per-row reduction (6·10⁻¹⁶), Monte Carlo of the moments, and brute
  force for every bound form on the nested tier. Copy it as R7b's tests.
- `e2e_s1h5.py`:
  - `Unit._cert`: the certified read as a substituting arm, with per-layer τ. It stops per (KV head, position) once
    every query head of the group meets τ; from 2026-10-12, later sets are computed only for the positions still
    waiting (verified identical);
  - `Unit.store_vote` (2026-10-12): R6's question-pass vote over the single store.

  These are CPU references for the GPU implementation's smokes.

**Build in the run-time** (R7a's `run_r7.py`, which wraps `run_s1h5`, is the model):
1. The single store as tier 1:
   - keys by the nested quantizer, values by R6's dithered tier;
   - the debias by `coin_moments_pc`;
   - the memory accounting.
2. The vote over the single store: R6's `vote_sets` with the store's base keys and dithered values in place of the 4/4
   Lloyd-Max store (`_store4`, `_vfn4`).
3. The primary arm (`dnest42ux_v4`) and E4 (`dtail6ux_v4`, R6's), with FP, FP8 and `fp_noise` in the same process.
4. Gate arms: dense dithered 7/6; KIVI-4 and dense 4/4 (existing); Quest at 0.398; TurboQuant's product variant
   (optional).
5. Exploratory arms:
   - E1: the certified read with per-layer τ (the KL budget), as a substituting arm, with 8-bit keys in the noisy
     layers (§4);
   - E2: the 4+3 store;
   - E3: per-layer widths.
6. The calibration stage: for E1, single-layer tail arms (L per prompt); for E3, single-layer key-only and value-only
   dither arms (2L per prompt). Both run on 5–10 calibration prompts pooled across families. A script turns them into
   the τ table and the allocation.
7. `read_r7b.py` with R6's rules (copied) plus the gate tests; sizing from R6's variance; pilots and gate (R6's gate
   logic); the chain; the code pin.
8. Tests: copies and parity, the nested brute force, and driver smokes on Llama-3.2-1B (the probe's pass is FP's; KL
   against the CPU references).

**Inputs from 14h_theory before the freeze: done 2026-10-12.** The results are in `findings/findings_nested.md` §6 and
`findings/findings_klbridge.md` §6; the runs are in `findings/klbridge_runs/`, summarized by its `checks_summary.py`.
1. **The vote's question pass with dithered values is neutral.**
   - It shares 98.7–98.9% of its rows with R6's Lloyd-Max-store pass and reads the same attention mass, within 0.002
     per layer.
   - Its KL is 0.99–1.01× (Llama-1B and Qwen3-1.7B, 8 seeds).
   - The single store against R6's design, each with its own vote: 1.27× and 1.12× in geometric mean, 1.15× and 0.91×
     on mean KL.
2. **The deployed combination works.**
   - Llama-1B: 5.6× and 25× below the single store's KL at +0.001 and +0.007 traffic, at or below the uniform tier's
     certified read.
   - Qwen3-1.7B: it needs 8-bit keys in its noisy layers. Without them part 8 is vacuous there, and the read costs
     +0.046–0.082 traffic. With them: 2.9× and 5.3× lower KL at +0.001 and +0.009.
   - At 32K on R7b's models (R7b's captures, every layer): Llama-3.1-8B is certifiable with the single store as is;
     Qwen3-30B-A3B needs 8-bit keys in 14 of 48 layers (`findings/findings_nested.md` §7).
3. **Calibrating τ.**
   - The shape needs per-layer sensitivities on models like Qwen (a flat τ is 1.35–1.4× worse there; it ties on
     Llama-1B). They can be pooled across families from 5–10 prompts.
   - The level that meets a KL target must be fitted per family, on at least 8–10 prompts: the KL at a given B varies
     about 10× across families.
   - E1 should report two fixed levels instead.
4. **The FP8 quantile holds up end to end**: 1.0–1.33× the KL budget's KL on Llama-1B. Its shape is 1.6–1.8× worse on
   Qwen, so it is E1's fallback if the calibration stage is cut, not the first choice.

**Open for the user at the freeze:**
- E1: the KL budget with a calibration stage (recommended), or the FP8 quantile with none; whether E1 on Qwen3-30B-A3B
  uses 8-bit keys in its 14 loose layers at 32K (recommended; +0.016 traffic) or the primary store with the
  exact-read fallback reported;
- whether the gates are tests or reports;
- whether TurboQuant's product variant is in;
- the GPU-hour budget.
