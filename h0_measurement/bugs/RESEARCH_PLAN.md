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
  - Compared like for like, v2 is 39–42× the error on the CPU (Llama-1B L1/L3 at 1.6K, `findings/findings_part7.md`) and
    25× on the GPU at 128K. Scale did not hurt.
  - On the same CPU layers, part 8 is about 5.7× tighter than v2.
  - Extrapolated, part 8 at 128K should be about 4–5× on Llama and 6–10× (median) on Qwen 32K. Qwen's noisy early
    layers will be much looser (11–900× on the CPU).

  So R7 confirms and reports; it is not expected to close a gap. That makes it a low-risk, high-value run.
- **"Port Theorem M's per-head prediction into the run-time code."** Theorem M needs the exact weights and values of the
  unread rows. It therefore belongs in the probe, which computes exact attention, not in the controller. A run-time
  version would be a plug-in estimate from tier-1 data. That is a new estimator, worth testing as a candidate for the
  run-time target τ (§3, τ).
- **"≈ 6–10 GPU-hours."** That is STATUS step 3's estimate. It was made before three costs were counted:
  - Qwen 128K's node jobs, about 4 GPU-h (R6's probe skipped them);
  - the 15-set certified read;
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

| rank | item | kind | folder | what it decides for the paper | cost | needs |
|---|---|---|---|---|---|---|
| 0 | **Close R6 formally**: the frozen reader over the final block set, archived | record | `14h_methodology_improve` | the final R6 numbers the paper cites | one reader run | the last blocks |
| 1 | **R7a: the theory at the paper's scale.** Part 8's bound, the certified read and Theorem M per head-step, in the probe, at 4/6/8-bit keys; a seed panel (§4) | GPU verification | new `14i_r7_theory_at_scale` | the theory section's numbers on its own models (T3, T4, T6); C1's seed gap | port and tests 3–5 days; 12–20 GPU-h | nothing |
| 2 | **N: the nested coin-free store**, with its lemma and Theorems R and M at per-coordinate steps (§3) | theory + CPU | `14h_theory` | a memory claim (0.31 at rest, about 0.44 during a question, against 0.50); R1 holds in the deployed system; the threshold read rule becomes provable | 2–4 days on CPU | nothing |
| 3 | **KL: bridge per-head error to KL per token**, with **τ** and the **water-filling allocation** (§3) | theory + CPU | `14h_theory` | explains R6's pattern (re-ranking, 4-bit keys); gives the allocation objective and a deployable τ | 1–2 weeks on CPU | nothing |
| 4 | **Draft the design paper**: the method, R6's table, and the theory section from `paper_theory_section.md`, with slots for R7 and R8 | writing | a new design-paper LaTeX folder | shows early which claims still lack evidence | ongoing | R6 |
| 5 | **R7b: design arms**, exploratory or reported: the nested store; the per-layer allocation (1.2, redesigned); a uniform 5/4 control; a dense tier at the design's traffic; the KL bridge's sensitivities at scale | GPU exploration | `14i_r7_theory_at_scale` | the nested store end to end (needed for the memory claim); whether a theory-chosen allocation beats 0.398 | 10–15 GPU-h on unit subsets | 2, 3 |
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
- **The observation.** v2's bound / error fell from 39–42× at 1.6K to 25× at 128K.
- **The task.** Explain it, and predict how bound / error and Theorem M's terms scale with the context length C (the
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
  - the certified read with the 15-set schedule;
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

### R7b: design arms (same folder; after N, KL and WF)
- **The nested store**, at 6/4 traffic and about 0.44 memory. Theorem M predicts it within a few percent of R6's design;
  the arm checks this end to end.
- **The per-layer allocation (exploratory).**
  - Calibrated at the run's length on separate calibration prompts from every family.
  - Objective: WF/KL; the fallback is FP8's error at the p90 target, taking the per-layer maximum over families.
  - Traffic about 0.36.
- **Uniform 5/4 (0.371):** the control.
- **A dense tier at the design's traffic** (e.g. dithered 7/6, 0.41), reported. The tail's MSE is the unread share
  times a dense tier's at the same widths. So Theorem M predicts the two-tier design's MSE to be tens of times lower
  than a dense tier's at equal traffic in concentrated layers.
- **The KL bridge's sensitivities** on a few units per family.
- **Exploratory throughout.** Any "beats 0.398" claim needs its own pre-registered run. About 10–15 GPU-h on subsets.

### R8: the cross-architecture prediction test (new folder `14j_r8_cross_architecture`)
- **Models:** Gemma-3-12B (flat global layers) and GPT-OSS-20B (noisy scores, sinks), at 32K, with R6's families and
  margin.
- **Pre-registered predictions,** from a calibration map at 32K:
  - Gemma 3: 6/4 fails and 6/6 passes;
  - GPT-OSS: 6/6 fails and 8/4 passes;
  - the allocation passes at its predicted traffic.
- **Needs** run-time support for sliding windows, sinks and per-layer bit widths.
- **Confirms** T8's "the formula picks the bits". About 20–40 GPU-h plus the port.

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

## 6. Decisions for the user
- **D1. Venue.** For a systems venue, measured time moves to rank 2. For an ML venue, traffic plus the stated
  limitation can stand.
- **D2. Is the certified read a contribution?** If yes, τ stays in rank 3. If no, the certified read becomes an
  analysis tool.
- **D3. Is "beats Quest" still a gate?** If yes, R7b needs a Quest arm at matched traffic.
- **D4. Memory claim.** Go through the nested store (recommended), or claim traffic only.
- **D5. Qwen 128K in R7a's probe?** It adds about 4 GPU-h of node jobs.
- **D6. R7 in a new folder** (recommended) or as a METH `plan.md` amendment. R7 split into R7a and R7b (recommended) or
  run as one campaign.

## 7. Order of work
- **Phase A (now, about a week, in parallel):**
  - close R6;
  - write R7a's port, tests and gate (`14i_r7_theory_at_scale`);
  - N: the lemma and the CPU check;
  - start the KL bridge;
  - outline the paper.
- **Phase B:**
  - submit R7a once the user gives the go-ahead;
  - finish the KL bridge, τ and WF;
  - freeze R7b's plan, including the nested store if N passed.
- **Phase C:**
  - run R7b;
  - port R8's run-time support;
  - write R8's pre-registered predictions from calibration maps at 32K.
- **Phase D:**
  - run R8;
  - start the kernel if D1 calls for it;
  - pursue the theory research items (SC uses R7a's data).
