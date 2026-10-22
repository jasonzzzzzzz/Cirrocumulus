# R7b's theory: what the run tests, why the single store's forms are valid, and what it should find (2026-10-12)

R7b is rank 5 of `../../14h_theory/RESEARCH_PLAN.md` (§4 "R7b", §8). It tests the deployed design: the **single store**
(`../../14h_theory/theory.md` §13) as tier 1, read with the vote's 1/8 exact, against FP8 under R6's rules. The single
store keeps keys as a deterministic 4-bit Lloyd-Max base plus a 2-bit subtractive-dithered residual, and values as R6's
4-bit dithered tier. The vote reads the base keys and the tier's own values, so there is no separate vote store.
- **Primary, confirmatory (D7):** the single store is non-inferior to FP8 KV on KL per answer token in each claim group
  (Llama-3.1-8B 128K, Qwen3-30B-A3B 32K and 128K), under R6's rules. This makes the memory claim confirmatory.
- **Key secondary:** accuracy, under R6's rule.
- **Gates (D3):** paired tests against baselines at no more than the design's traffic.
- **Exploratory:** E1 the certified read with a run-time target; E2 the 4+3 store; E3 water-filling; E4 R6's design as a
  paired reference.

This file is the theory for that run, written while R7a's jobs are pending. It is CPU work on R6's own models; nothing
here is a GPU run, and nothing outside `r7b/` was changed. It adds:
1. which measurement tests which statement, and what each assumes (§1);
2. the single store as run-time code: Lemma N in a deployment form, the read-time steps and debias, the accounting (§2);
3. the bounds at per-coordinate steps: R7b's bounds are R7a's code, and Lemma P′ gives Theorem M's moments (§3);
4. the vote and the certified read on the single store: R1 by a filtration (Lemma C′), the question pass's perturbation
   (Lemma V), E1's guarantee with a run-time target, and its stopping bias (Proposition S) (§4);
5. the single store on R6's models at 1.6K–16K (§5);
6. the statistics: seeds per unit, power under R6's rules, the gates' tests and when two tiers beat one (Corollary G) (§6);
7. the predictions, stated before the run (§7), what each outcome would mean (§8), and the limits (§9).

Notation as theory.md and `../theory_r7.md`: weights w_i = e^{s_i}, the read set S (the vote's 1/8), the unread rows T,
the exact output o, the design's õ, FP8's per-head error e₈, u_i = v_i − o, and the unread share
share2 = Σ_T w_i²‖u_i‖² / Σ_i w_i²‖u_i‖².

## 1. What each measurement tests

| statement | what R7b measures | what it assumes | what would contradict it |
|---|---|---|---|
| **Primary (D7).** The single store `dnest42ux_v4` is non-inferior to FP8 per claim group × family | R6's statistic on fresh units: U = 95th percentile of 2,000 unit-bootstrap means of KL_design − 1.5·KL_FP8, PASS if U < 5·10⁻⁴; `fp_noise` calibrates | nothing from the theory: it is an empirical test | U ≥ 5·10⁻⁴ in a testable family |
| Key secondary: accuracy | R6's rule on FP-correct units outside aggregation | — | lower bound ≤ −0.02 with ≥ 150 units |
| S2: memory 0.324 at rest and 0.449 during a question, against FP8's 0.50 | the accounting (§2.6). The GPU arm emulates the store in a dequantized cache, so its measured peak is not the claim | the stored layout of §2.2 | — (arithmetic) |
| T9 end to end: the single store costs little | E4 paired in one process: KL(single store) / KL(R6's design), per family | — | a ratio far above the head-level prediction (§5) |
| R1 in the deployed system | nothing: it holds by construction (Lemma C′, §4.1) | per-layer independent coins; teacher-forced queries | — (a theorem) |
| The vote is unchanged by dithered values in the question pass (Lemma V) | the vote's rows and the KL gap of E4, per family | — | a KL gap concentrated where the vote changes |
| Gates (D3) | per group, the paired upper bound of mean(KL_design − KL_baseline) < 0, Holm over the baselines (§6.3) | — | — (each gate is its own test) |
| E1: the certified read with a run-time target τ | KL and traffic of the substituting arm; per-head misses if probed | Theorem N; τ fixed before the run (calibration) | misses above δ per query head; KL far above its budget times Proposition S's factor |
| E2, E3 | KL and traffic | — | — (exploratory) |
| C1's seed gap | a dither seed drawn per unit (§6.1) | — | — |

## 2. The single store at run time

**2.1 The store** (`store_r7b.py`; a copy of `nested_s1h5`'s, bit for bit: `fixtures/`). For a key row x of one KV head
and the tier's key rotation R (seed 50, R6's):
- γ = ‖x‖ and y = (x/γ)Rᵀ; M = max_j |y_j|;
- **base:** m_j, the 4-bit Lloyd-Max index of y_j (sievelib's N(0, 1) codebook / √d, midpoint thresholds). Its
  norm-corrected reconstruction γ·(lv[m]/‖lv[m]‖)·R is `quant.quantize_keys(x, 4, R)`: the vote's keys, coin-free;
- **cell:** [lo_j, hi_j], the decision interval of m_j, with the two outer cells closed at ±M;
- **residual:** r bits of subtractive dither on the cell: Dn_j = (hi_j − lo_j)/(2^r − 1),
  c_j = round((y_j − lo_j + u_j)/Dn_j), u_j ~ U(±Dn_j/2) from R6's counter-based generator (never stored);
- **the tail** reads ŷ_j = lo_j + c_j·Dn_j − u_j and x̂ = γ·ŷ·R: keys at 4 + r bits, unbiased.

**Lemma N** (theory.md §13). Given the base indices, the errors ŷ_j − y_j are independent across coordinates and rows,
each uniform on ±Dn_j/2, and independent of the data. So the key error has per-coordinate steps D_ij = γ_i·Dn_ij in the
rotated basis.

**2.2 What a deployment stores, and Lemma N′.** Per row: the base indices (4 bits per coordinate), the residual codes
(r bits), and three fp16 scalars: γ̃, M̃ and the value step. Rounding the scalars needs care, and one rule is enough:

**Lemma N′.** Let the store keep any γ̃ > 0 (e.g. fp16(γ)) and any M̃ ≥ max_j |y_j| with y = (x/γ̃)Rᵀ. Encode the base
and the residual on this y, with the outer cells closed at ±M̃. Then x̂ = γ̃·ŷ·R is unbiased, and Lemma N holds with
Dn_j computed from the stored m_j and M̃.

*Proof.* Lemma N's proof uses two facts: y_j lies in its clipped cell (true, since |y_j| ≤ M̃), and the cell is a
function of stored values (m_j, M̃). Nothing uses ‖y‖ = 1. Unbiasedness: E x̂ = γ̃·(E ŷ)·R = γ̃·y·R = x. ∎

Two consequences for a deployment:
- **Round M up** (fp16 toward +∞). If M̃ < |y_j| at the largest coordinate, that coordinate lies outside its clipped
  outer cell, the quantizer clamps it, and its error is no longer uniform: Lemma N fails there.
- **Normalize by the stored γ̃**, not the exact γ. Then fp16 rounding of γ costs nothing. Normalizing by γ but
  reconstructing with γ̃ would add a deterministic key error (γ̃/γ − 1)·x, a score bias of up to 2⁻¹¹·|s_i|.

**2.3 The steps at read time.** D_ij = γ̃_i·Dn(m_ij, M̃_i). The 14 inner cells have fixed widths; the two outer ones
depend on M̃_i. `store_r7b.cell_steps` computes them from what the store keeps. They equal the encoder's bit for bit
(`test_r7b.test_store_read`), and so do the keys rebuilt from the codes and the regenerated dither (`reconstruct`).

**2.4 The debias at read time.** The unbiased tail subtracts ldb_i(q) = ½(lmu_lo + lmu_hi) from each unread row's score,
with A2_i = (sc/2)²·Σ_j (Rq)_j²·D_ij² and A4_i = (sc/2)⁴·Σ_j (Rq)_j⁴·D_ij⁴. R6's uniform tier has one step per row, so its
A2 is rank one, a_i²·‖q‖². The single store needs two matmuls per (KV head, block of queries): (Rq)^∘2 against
(D^∘2)ᵀ, and the same at the 4th power.
- `store_r7b.debias_from_codes` streams them over chunks of rows from the codes, without holding the [G, C, d] step
  tensor. It equals the full computation to 0 in the test.
- Cost at 128K: about 137 GFLOP per layer per 64 teacher-forced queries in fp32 (32 query heads), a few ms on an H100.
- Precision: fp32's relative error of about 10⁻⁶ in A2 moves ldb by about 10⁻⁹. Against σ² ≈ 10⁻³ that is negligible.

**2.5 Emulation in R7b's GPU arm.** R6's arms emulate tier 1: they store x̂ dequantized in the cache's dtype (bf16), and
the debias is an additive attention mask. R7b's arm would do the same, with the nested quantizer and the per-coordinate
debias. bf16 storage of x̂ adds a deterministic rounding of at most 2⁻⁹ per coordinate. Its score noise is about 4% of
the 6-bit dither's in standard deviation (0.2% in variance). The same holds for R6's design, so E4's paired comparison
is unaffected.

**2.6 Accounting** (`store_r7b.accounting`; fractions of FP16 K+V; d = 128; the 1/8 read; fp16 per-row scalars).

| design | at rest | during a question (exact 1/8 fetched) | traffic per decode step | question pass (once per question) |
|---|---|---|---|---|
| single store (4+2 keys / 4-bit values; γ̃, M̃, value step) | 0.3242 (0.3125 without scalars) | 0.4492 (0.4375) | 0.3984 | 0.2578 |
| R6's design (tier 6/4 with key and value steps, plus a 4/4 Lloyd-Max vote store with key and value norms) | 0.5781 (0.5625) | 0.7031 (0.6875) | 0.3984 | 0.2578 |
| FP8 (one scale per KV head) | 0.5000 | 0.5000 | 0.5000 | — |

- Traffic per step follows R6's model (`s1h5_lib.traffic_frac`): tier 1 for the unread rows, FP16 for the read 1/8,
  per-row scalars not counted. Counting them adds 0.0103.
- The question pass reads the base keys and the values once per question: +0.258/N per step for an N-token answer
  (+0.008 at N = 32), the same as R6's vote store pass.
- E1 adds the rows it reads beyond the vote: +0.001 to +0.007 on the CPU with the single store
  (`../../14h_theory/findings/findings_klbridge.md` §6.1).

## 3. The bounds at per-coordinate steps

**3.1 What changes and what does not.** Theorem N (theory.md §13): every Theorem R form and Theorem M hold with the
per-coordinate steps D_ij. The proofs use the key steps only through three quantities, each computable per row and query:
- the exact coin moments E e^{−kδ_i} = ∏_j sinhc(k·a_ij), with a_ij = sc·|(Rq)_j|·D_ij/2 (Theorem M, part 8's exact-law
  proxy P-exact, the debias);
- the sub-Gaussian proxy σ_i² = sc²·Σ_j (Rq)_j²·D_ij²/12;
- the almost-sure bound sc·Σ_j |(Rq)_j|·D_ij/2.

The value side is R6's. R7a's GPU forms carry over unchanged:
- Lemma S (the HKZ matrix from heavy rows) and Lemma B (the certified read decided by brackets) bound the HKZ matrix's
  norms. The matrix is built from per-row proxies and the observed values, never from the key steps;
- the batching of the certified read (`../theory_r7.md` §2.3) is likewise unchanged.

**3.2 R7b's bounds are R7a's code.** `cert_r7b.exact_rows` takes [G, C] steps (it then calls R7a's `cert_r7.exact_rows`)
or [G, C, d] steps (theory5's per-coordinate branch). R7a's `p8_terms`, `bound_p8` and `certified_read` are re-bound to
it. They are the same code objects, resolving `exact_rows` in a namespace where it is R7b's (`cert_r7b._relink`).
`cert_r7` is neither copied nor patched. Checked (`test_r7b.py`):
- **with [G, C] steps,** the re-bound functions equal R7a's bit for bit: bound, stop and output;
- **with [G, C, d] steps that repeat one step per row,** every quantity equals R7a's scalar path: 0 for P-exact and
  part 8, ≤ 10⁻¹⁵ for the coin quantities, 5·10⁻¹⁶ for Theorem M;
- **against 14h_theory's reference on frozen inputs** (`fixtures/`: 12 synthetic cases at 4+2, 4+3 and 3+2 bits, and a
  slice of Llama-3.1-8B L1):
  - the store and the per-coordinate coin quantities match exactly;
  - part 8 in R7a's five configurations at two δ to 1.2·10⁻¹⁵ (2,560 values);
  - Theorem M to 6.8·10⁻¹³ (the reference forms E Z² − (E Z)²; Lemma P′ forms (E Z)²·expm1(Δ));
  - the certified read's stop in 24 of 24 groups;
- **by brute force** (4 kinds of adversarial data; bases of 2–4 bits, residuals of 1–3 bits; 1,152 head cases per form
  and δ): v2, the split, HKZ, P-exact, part 8 and part 8 with Bennett have no failures at any δ. The median bound /
  error at 6–7 bits is v2 17.4, the split 7.0 and part 8 2.4;
- the certified read's stop equals the naive rule's with and without Lemma B's brackets.

**3.3 Lemma P′ (Theorem M's exact moments at per-coordinate steps, separable).** With g(x) = log(sinh x / x),
a_ij = α_j·D_ij (α_j = sc·|(Rq)_j|/2), A = max_j α_j and D̄_i = max_j D_ij: wherever 2A·D̄_i ≤ 2,

  Σ_j g(k·a_ij) = Σ_{m≥1} c_m·k^{2m}·y_i^m·S_m(i),  y_i = (A·D̄_i)² ≤ 1,  S_m(i) = Σ_j (α_j/A)^{2m}·(D_ij/D̄_i)^{2m},

with c_m = 2^{2m}B_{2m}/(2m(2m)!) (1/6, −1/180, 1/2835, …). Δ_i = Σ_j [g(2a_ij) − 2g(a_ij)] is the same series with
c_m(4^m − 2), and Var Z_i = (E Z_i)²·expm1(Δ_i).

*Proof.* g's Taylor series converges absolutely for |x| < π, and every |k·a_ij| ≤ 2·A·D̄_i ≤ 2. Sum it termwise over the
finitely many j. ∎

- **The form.** Each S_m is one [queries, d] × [d, rows] matmul of entrywise powers. Every factor is ≤ 1, so nothing
  overflows. The terms fall like (2/π)^{2m}, so 40 terms leave a tail below 10⁻¹⁷·d. Rows outside the range use the
  closed form.
- **Exactness.** The series is within 7·10⁻¹⁶ of 40-digit arithmetic (`test_r7b.test_lemma_p_prime`). The closed form
  it replaces loses up to 10⁻¹³ relative just above |x| = 0.1 (log1p's cancellation). That is why the fixture gap above
  is 10⁻¹³, not 10⁻¹⁶.
- **Cost.** At 128K: 40 matmuls of [128, 128] × [128, 131K] per KV head and layer, about 0.17 TFLOP in fp64. It is
  needed only where Theorem M is computed (the calibration stage, a probe), never in an arm.
- **Measured against Monte Carlo** (`test_r7b.test_moments_mc`, 1,200 draws): E Z, E Z² and Theorem M's E‖N‖²/Z² are
  within 3.6 and 2.3 standard errors.

## 4. The vote and the certified read on the single store

**4.1 R1 in the deployed system (Lemma C′).** Number the layers 1..L. Each layer draws its own coins U_l (key residual
and value dithers), independent across layers and of the cache. Let F_l = σ(U_1, …, U_{l−1}).

**Lemma C′.** In a teacher-forced pass with the single store, the following are F_l-measurable:
- the vote's read set S_l (the question pass over the base keys and the dithered values);
- the certified read's schedule at layer l (the base sketch's priority for the current query);
- every query at layer l;
- any target τ_l fixed before the run.

U_l is independent of F_l. So, given F_l, layer l's tier is a fresh draw on a fixed (cache, query, read set), and every
per-call statement of Theorems R and N holds at layer l with probability ≥ 1 − δ, given F_l and hence unconditionally.

*Proof.*
- The hidden states entering layer l, in the question pass and in the answer pass, are functions of the cache and of
  the outputs of layers < l, hence of U_{<l}.
- The base keys are a deterministic function of the cache. The vote and the schedule are deterministic functions of
  these and of the queries.
- U_l enters only layer l's outputs, which feed layers > l. In particular, the question pass at layer l reads U_l's value
  coins, but that affects only S_{l'} for l' > l, and U_l is part of F_{l'}. ∎

This is theory.md §13's "R1 layer by layer", stated as a filtration. It also gives Proposition KL's zero-mean condition
for fixed read sets: E[e_l | F_l] is Theorem M's bias term, about 0. Free-running decode is covered up to the first token
that differs from FP's (Theorem DD); R7b's KL is teacher-forced.

**4.2 Lemma V (the question pass's value perturbation).** At layer l, the question pass gives question row t the context
weights p_ti (softmax over the base-key scores and the question rows). Its value error is e_t = Σ_i p_ti·(v̂_i − v_i).
- **Single store:** E e_t = 0 and E‖e_t‖² = Σ_i p_ti²·d·D_i²/12, exactly (the rows' errors are independent and uniform).
- **R6's vote store:** e_t = Σ_i p_ti·b_i, with b_i the deterministic TurboQuant-4 errors.

Both are dense errors over every context row. They reach the vote only through higher layers' question queries. So the
vote is unchanged to first order if the two perturbations have the same size. They do:
- on R6's models the dithered pass's RMS error is 0.94–1.28× the Lloyd-Max pass's (§5.1);
- that matches the per-row MSE of a 4-bit uniform quantizer against Lloyd-Max on near-Gaussian rotated rows (1.3
  measured);
- per layer, the base's vote reads the same attention mass as R6's store vote within 0.0012 on every view (§5.1).

End to end, 14h_theory's CPU check finds the vote and the KL unchanged: 98.7–98.9% of rows in common, KL ratio
0.99–1.01 (`../../14h_theory/findings/findings_nested.md` §6). E4 measures it at scale.

**4.3 E1: the certified read with a run-time target.**

*(a) The guarantee needs no condition on the target.* Let E be the event that every set's bound holds, B_j^h ≥ err_j^h
for all sets j at query head h. Then P(E^c) ≤ m·δ/m = δ (`../theory_r7.md` §2.6). On E, the stop J satisfies
B_J^h ≤ τ^h, so err_J^h ≤ τ^h, whatever τ is and however J was chosen; at J = m (the exact read) the error is 0. So
P(err_J^h > τ^h) ≤ δ at each query head, for any target, even one that depends on the coins. A run-time τ (the KL budget
τ_l² = 2B/(L·H·c_l), or a per-layer FP8 quantile) is a calibration constant, so it is F_l-measurable anyway (Lemma C′).

*(b) The stop biases the output.* Proposition KL assumes per-layer errors of mean zero, so that layers add. The certified
read's output õ_J is not unbiased: J depends on the coins (optional stopping). Take any F_l-measurable reference stop J*
(e.g. the most frequent stop) and write e_J = e_{J*} + X, with X = (e_J − e_{J*})·1{J ≠ J*}. Then:
- E[e_{J*} | F_l] is a fixed set's bias (Theorem M's bias term, about 0);
- ‖E[X | F_l]‖² ≤ E[‖X‖² | F_l]·P(J ≠ J* | F_l) (Cauchy–Schwarz).

So the stopping bias is small when the stop rarely moves off J*. It moves only where a bound sits near its target. A
constant per-layer target makes the bound's coin dependence (mostly through ‖õ − c‖) the only source.

*(c) Proposition S (how much the stopping bias can cost).* Let a_l = F^{1/2}J_l·e_l be layer l's error in the logits'
Fisher metric (Proposition KL's second-order setting, with J_l FP's Jacobian), KL_l = ½E‖a_l‖², and
β_m = E[e_m | F_m] the conditional bias. Suppose E‖F^{1/2}J_m·β_m‖² ≤ β·E‖a_m‖² at every layer m. Then, to second order,

  E KL ≤ (1 + √β·(L − 1))·Σ_l KL_l.

*Proof.* E KL = ½E‖Σ_l a_l‖² = Σ_l KL_l + Σ_{l<m} E[a_lᵀa_m]. For l < m, a_l is F_m-measurable, so
E[a_lᵀa_m] = E[a_lᵀ·F^{1/2}J_m·β_m] ≤ √(E‖a_l‖²)·√(β·E‖a_m‖²) = 2√β·√(KL_l·KL_m). Summing over pairs,
Σ_{l<m} 2√(KL_l·KL_m) ≤ (L − 1)·Σ_l KL_l. ∎

With β = 0 (fixed read sets) this is Proposition KL's additivity. The bound is worst case: it lets every layer's bias
align with every lower layer's error. β = 10⁻⁴ gives at most 1.31× the sum at L = 32 and 1.47× at L = 48.

*Remark S′ (only the systematic part enters at first order).* Write b_m = F^{1/2}J_m·β_m = b̄_m + b̃_m, with
b̄_m = E b_m (the bias's systematic part) and b̃_m its fluctuation with the lower layers' coins. Then
E a_l = b̄_l, and the cross term splits:

  E[a_lᵀ b_m] = b̄_lᵀ b̄_m + Cov(a_l, b̃_m).
- The first part sums to at most β·(L − 1)·Σ_l KL_l.
- If the lower layers' coins move layer m's bias by a relative η (Var b_m ≤ η²·E‖b_m‖²), the second part sums to at most
  η·√β·(L − 1)·Σ_l KL_l.

So E KL ≤ (1 + min(√β, β + η·√β)·(L − 1))·Σ_l KL_l. η is small: b̃_m comes only from the lower layers' perturbation of
layer m's queries, a relative 10⁻³–10⁻² in the design. With β = 3·10⁻³ and L = 32 the factor is 2.7 by Proposition S,
1.26 at η = 0.1 and 1.11 at η = 0.01. The CPU end-to-end check met its budget within 15% (findings_klbridge §4). §5.5
measures β and P(J ≠ J*) on R6's models.

*(d) Which τ at scale.* 14h_theory's CPU checks (`../../14h_theory/findings/findings_klbridge.md` §6, 2026-10-12)
settle most of it:
- **The deployed combination works.** The single store plus the certified read with the KL-budget τ cuts the single
  store's KL 5.6× for +0.001 traffic and 25× for +0.007 on Llama-1B, ending 22–96× below FP8 (§6.1 there).
- **The shape matters little.** One flat τ for every layer, with no sensitivities, is within 5% of the calibrated
  KL-budget shape on Llama-1B and better on Qwen3-1.7B at 5–10% extra rows. The FP8-quantile shape is 1.2–1.4× worse on
  Llama and about 2× on Qwen (§6.3 there).
- **The level must be calibrated per family.** The KL a given τ reaches differs up to 10× between texts. Plan on 8–10
  calibration prompts per family: a family mean then lands within about ±20% of its level.
- **The FP8 quantile needs no sensitivities and its level is stable.** End to end on Llama-1B it matches the KL budget
  at 0.3994 (§6.4 there), and R7a's probe records the per-layer FP8 errors it needs at the paper's scale.
- **Noisy layers make the certificate vacuous.** In Qwen3-1.7B's layers with score noise σ ≥ 0.25 at 6-bit keys, part
  8's bound is 10³× the error, and the certified read falls back to the exact read: +0.05–0.08 traffic (§6.2 there).
  The same holds at scale in Qwen3-30B's L1 (§5.3: certified share ≤ 0.07; the read takes 0.94–1.0 of rows at 1.6K–16K).
  The remedy is per-layer key widths: 4+4 keys where σ ≥ 0.25 (σ falls 4×, +0.014 traffic on Qwen3-1.7B). Otherwise
  E1 should skip those layers and leave them to the vote, since the guarantee is per call and needs no coverage of
  every layer.

So for R7b's E1:
- a flat τ, or the FP8 quantile from R7a's probe output, with its level calibrated per family on 8–10 separate prompts;
- on Qwen, either 4+4 keys in its noisy layers (Theorem M's σ map picks them) or no certified read there.

**4.4 E2 and E3 on the single store.** The base must stay at 4 bits, because the vote and the schedule read it. Only the
residual's width r and the value width can change.
- **E2, the 4+3 store** (7-bit keys, traffic 0.426). Its residual has 8 levels per cell instead of 4, so its key steps
  are 3/7 of 4+2's and its key term 0.18 of 4+2's. The total falls by the key share of the MSE (§5.2). On the CPU, KL is
  0.39–0.57× uniform 6/4's (findings_nested §4, findings_klbridge §5).
- **E3, per-layer widths** (4 + r_l, v_l), with r_l ∈ {1, 2, 3} and v_l ∈ {3, …, 6}. This is Corollary WF on the grid:
  - each layer takes the argmin of traffic + λ·KL_l (`cert_r7b.waterfill`, `frontier`);
  - the surface is Theorem M's per-layer MSE at each width (Lemma P′ handles the nested steps), weighted by c_l^K and
    c_l^V from the calibration stage;
  - every calibration input is weighted equally (`equal_weight_surface`): the raw average does not transfer
    (findings_klbridge §3);
  - the control is uniform 4+1/4 (traffic 0.371), which separates "fewer bits" from "allocation";
  - CPU: 16% lower KL than uniform at equal traffic, held out.
- **The optional TurboQuant product variant** (Lloyd-Max plus a 1-bit QJL residual) is unbiased in scores, not in
  weights: E e^{ŝ} = e^{s}·E e^{δ} ≠ e^{s}. The single store debiases the weights (ldb). As a dense quantizer at matched
  traffic it falls under Corollary G (§6.3).

## 5. The single store on R6's models (CPU, `explore_r7b.py`, `findings/explore_r7b.md`)

Setup: R7a's views (`../.views`: PG-19, the last 48 positions fixed, the first 32 vote, the last 16 are measured):
- Llama-3.1-8B L1 and L3, and Qwen3-30B-A3B L1 (a flat layer) and L12, at 1.6K–16K tokens;
- Llama-3.2-1B L1 at 1.6K for reference.

Tier 1 has nested 4+r or uniform k-bit keys and 4-bit dithered values, with R6's rotations; δ = 10⁻³; 3 draws. The
single store reads its own vote (the base keys, rotation 50); R6's design reads R6's (TurboQuant-4 store, rotation 0).
Full tables: `findings/explore_r7b_tables.md`.

**5.1 The vote and the question pass (Lemma V).**

| model, layer | context | attention mass read: base / R6's store / exact-key vote | rows in common, base and R6 | question pass: dithered / Lloyd-Max RMS error (p10–p90) | per-row value MSE, dithered / Lloyd-Max |
|---|---|---|---|---|---|
| Llama-8B L1 | 1.6K → 16K | 0.7487 / 0.7488 / 0.7493 → 0.7323 / 0.7322 / 0.7333 | 0.79 → 0.82 | 1.15 (0.94–1.40) → 1.14 | 1.30 |
| Llama-8B L3 | 1.6K → 16K | 0.8018 / 0.8019 / 0.8026 → 0.8162 / 0.8162 / 0.8166 | 0.84 → 0.89 | 1.08 (0.95–1.60) → 1.09 | 1.32–1.33 |
| Qwen-30B L1 (flat) | 1.6K → 16K | 0.3110 / 0.3100 / 0.3163 → 0.3807 / 0.3795 / 0.3873 | 0.80–0.82 | 1.01 (0.50–1.19) → 0.94 | 1.31–1.32 |
| Qwen-30B L12 | 1.6K → 16K | 0.5637 / 0.5646 / 0.5670 → 0.6519 / 0.6516 / 0.6539 | 0.70–0.75 | 1.28 (1.09–1.51) → 1.25 | 1.31 |

- **The base's vote reads the same attention mass as R6's store vote,** within 0.0012 on every view. The rows differ
  only among near-ties, because the rotations differ. The exact-key vote reads at most 0.007 more.
- **The question pass's dithered values perturb it by 0.94–1.28× Lloyd-Max's RMS error,** matching the per-row MSE
  ratio of 1.3 (a uniform 4-bit quantizer against Lloyd-Max on near-Gaussian rotated rows).
- End to end, 14h_theory's CPU check finds the KL unchanged (0.99–1.01; findings_nested §6).

**5.2 Theorem N at scale: what the single store costs, and its certificate.** "Sum" is over head-steps, the bridge's
weighting with equal sensitivities. The KL proxy weights the value term by c^V/c^K = 0.46.

| model, layer | context | MSE, single store / uniform 6/4 (same vote; median, sum) | single store / R6's design (each its own vote; sum, KL proxy) | key share of the MSE | Theorem M measured / predicted (mean) | part 8 coverage (misses) | part 8 / error: single / uniform | certified within FP8: single / uniform |
|---|---|---|---|---|---|---|---|---|
| Llama-8B L1 | 1.6K | 1.15, 1.11 | 1.11, 1.18 | 0.44 | 1.00 | 1.0000 (0) | 5.96 / 5.08 | 0.22 / 0.35 |
| Llama-8B L1 | 16K | 1.16, 1.15 | 1.27, 1.35 | 0.46 | 0.97 | 1.0000 (0) | 6.85 / 5.49 | 0.49 / 0.65 |
| Llama-8B L3 | 1.6K | 1.12, 1.10 | 1.16, 1.23 | 0.36 | 1.03 | 1.0000 (0) | 4.77 / 4.20 | 0.53 / 0.62 |
| Llama-8B L3 | 16K | 1.14, 1.13 | 1.12, 1.20 | 0.40 | 0.97 | 1.0000 (0) | 5.46 / 4.48 | 0.92 / 0.97 |
| Qwen-30B L1 (flat) | 1.6K | 1.34, 1.24 | 1.21, 1.33 | 0.70 | 0.97 | 1.0000 (0) | 16.7 / 11.3 | 0.00 / 0.03 |
| Qwen-30B L1 (flat) | 16K | 1.35, 1.31 | 1.32, 1.44 | 0.72 | 1.00 | 1.0000 (0) | 19.3 / 12.5 | 0.07 / 0.09 |
| Qwen-30B L12 | 1.6K | 1.31, 1.29 | 1.39, 1.44 | 0.78 | 0.98 | 1.0000 (0) | 18.8 / 13.5 | 0.36 / 0.43 |
| Qwen-30B L12 | 16K | 1.31, 1.29 | 1.26, 1.33 | 0.79 | 1.03 | 1.0000 (0) | 21.0 / 15.1 | 0.68 / 0.77 |

The 4K and 8K rows (`findings/explore_r7b_tables.md`) lie between.
- **The single store costs about 0.2 bit of key precision, and the cost is flat in context.**
  - Its MSE is 1.10–1.15× the uniform 6-bit tier's on Llama and 1.24–1.31× on Qwen at every length. The key term alone
    is 1.3–1.4×.
  - Qwen's ratio is larger because its MSE is 70–80% keys, and the nested quantizer changes only the keys.
  - With each design's own vote the ratio is 1.08–1.27 (Llama) and 1.21–1.48 (Qwen); in the KL proxy, 1.15–1.35 and
    1.33–1.58.
  - 14h_theory's end-to-end CPU check gives 1.12–1.27× per unit and 0.91–1.15× on mean KL (findings_nested §6).
- **E2:** the 4+3 store matches uniform 7-bit keys (1.00–1.04×). Its MSE is 0.64–0.76× the 4+2 store's on Llama-8B and
  0.42–0.61× on Qwen (keys dominate there).
- **Theorem N holds at scale.** Measured / predicted MSE is 0.97–1.04 (mean over head-steps, 3 draws).
- **Part 8 covers every case.** There are 0 misses in all 15 views × 3 draws.
  - On the single store it is 1.1–1.4× looser than on the uniform tier, from the outer cells' wider steps and the
    larger error.
  - The certified share within FP8 falls accordingly: 0.92 against 0.97 on Llama L3 at 16K, 0.68 against 0.77 on
    Qwen L12.
  - It still rises with context, as in R7a's exploration (`../findings/explore_r7.md`).

**5.3 The certified read on the single store** (FP8's per-head error as an oracle target, draw 0; traffic in brackets).

| model, layer | 1.6K: single / uniform | 16K: single / uniform | oracle stop at 16K | stops at the vote's set at 16K (single) |
|---|---|---|---|---|
| Llama-8B L1 | 0.558 (0.696) / 0.467 (0.633) | 0.251 (0.485) / 0.209 (0.456) | 0.125 | 0.08 |
| Llama-8B L3 | 0.268 (0.497) / 0.228 (0.469) | 0.134 (0.404) / 0.128 (0.400) | 0.125 | 0.73 |
| Qwen-30B L1 (flat) | 1.000 / 0.994 | 0.941 (0.960) / 0.764 (0.838) | 0.145 | 0.00 |
| Qwen-30B L12 | 0.683 (0.782) / 0.547 (0.689) | 0.332 (0.541) / 0.245 (0.481) | 0.125 | 0.00 |

- Every query head met its target (1.000).
- The single store's looser bound costs rows: +0.006 of rows on Llama L3 at 16K, where the vote's own set is certified
  for 73% of groups, and +0.09 on Qwen L12.
- FP8's per-head error is a demanding target (§4.3(d)). With a per-layer run-time target the read is cheap where the
  vote is good: 0.130 of rows on Llama L3 at 16K, cutting the per-head MSE 8× (§5.5).
- Qwen's flat L1 is the exception at every length: the read takes 0.94–1.0 of rows. That is §4.3(d)'s vacuous
  certificate in a noisy layer, and it calls for 4+4 keys there or no certified read.

**5.4 The gates per head-step** (each baseline's MSE over the single store's, summed over head-steps; Theorem M for the
dithered tiers, exact errors for the deterministic ones).

| model, layer | context | dense dithered 7/6 (0.406) | dense dithered 6/6 (0.375) | FP8 (0.500) | Quest (0.398) | KIVI-4 (0.250) | dense Lloyd-Max 4/4 (0.250) | unread share2 (median) |
|---|---|---|---|---|---|---|---|---|
| Llama-8B L1 | 1.6K | 1.8 | 3.3 | 2.3 | dense layer | 111 | 41 | 0.0012 |
| Llama-8B L1 | 16K | 28 | 55 | 68 | dense layer | 1,484 | 729 | 0.0003 |
| Llama-8B L3 | 1.6K | 3.3 | 6.5 | 12 | 553 | 87 | 63 | 0.0022 |
| Llama-8B L3 | 16K | 46 | 89 | 217 | 1,248 | 1,475 | 1,076 | 0.0001 |
| Qwen-30B L1 (flat) | 1.6K | 0.43 | 1.5 | 0.61 | dense layer | 2.1 | 30 | 0.32 |
| Qwen-30B L1 (flat) | 16K | 0.69 | 2.5 | 1.5 | dense layer | 4.5 | 84 | 0.13 |
| Qwen-30B L12 | 1.6K | 4.2 | 15 | 27 | 124 | 20 | 218 | 0.0031 |
| Qwen-30B L12 | 16K | 38 | 128 | 719 | 803 | 201 | 1,998 | 0.0005 |

("dense layer": R4's Quest reads layers 0–1 exactly, so its error there is 0.)

- **Two tiers beat one by a margin that grows with context, as Corollary G says.** The design's error falls with the
  unread share, which falls with context; a dense tier's does not.
  - The median over head-steps is 15–170× for dense 7/6.
  - The sums, dominated by the flattest heads, are smaller: 1.8 → 28 (Llama L1), 3.3 → 46 (Llama L3), 4.2 → 38 (Qwen L12).
- **The exception is the flat layer.** In Qwen's L1 the vote leaves 13–32% of the second moment unread, and dense 7/6
  has 0.3–0.7× the design's MSE at every length. FP8 has 0.6–1.5×.
  - The gate's outcome on Qwen therefore depends on how much of the KL its flat early layers (L0–L3; `../../14h_theory/
    findings/findings_qwen.md`) carry.
  - The full-depth CPU job (`depth_r7b.md`, 32K) gives the totals over layers with equal sensitivities.
- **Quest at matched traffic** loses by 124–1,248× wherever it selects (layers ≥ 2), because it drops two thirds of the
  rows.
- **KIVI-4 and Lloyd-Max 4/4** lose by 2–2,000×: least in Qwen's flat L1 (KIVI-4 2.1–4.5×), most at 16K. This is
  consistent with R6's end-to-end margins (§6.3).

**5.4b Every layer at 32K** (CPU job 2542204: every layer of both models captured at 32K and analyzed as above;
`findings/depth_r7b.md`). Totals are sums over layers and head-steps, i.e. the bridge with equal sensitivities.

| model | single store / R6's design (MSE; KL proxy) | 4+3 / 4+2 | dense 7/6 / design | FP8 / design | Quest / design | the design's MSE concentrated in |
|---|---|---|---|---|---|---|
| Llama-3.1-8B (32 layers) | 1.21; 1.31 | 0.77 | 10.7 | 106 | 16,324 | L8 0.47, L5 0.34 |
| Qwen3-30B-A3B (48 layers) | 1.15; 1.24 | 0.62 | 136 | 1,419 | 1,626 | L47 0.20, L42 0.17 |

- **At full depth both models' totals favour the design by an order of magnitude or more.** Qwen's flat layers 0–1,
  the worry of §5.4, carry under 0.3% of the design's MSE at 32K.
- **On Llama the design's error is concentrated.** Layers 8 and 5 carry 81% of it, and there dense 7/6 has 0.9× and
  0.4× the design's MSE. So the Llama gate depends on how sensitive those two layers are.
- **Part 8 covers every layer of both models** (1.0000).
  - The certified read at FP8's oracle target needs about the vote's set (0.125–0.15 of rows) in nearly every Llama layer.
  - Qwen's 14 loose layers need 0.27–1.0 of rows (E1's 4+4 keys, §4.3(d)).

**5.5 E1's stopping bias** (the single store, 2 KV heads × 4 steps per view; J* from 20 separate draws, 80 evaluated draws).

| view | target | rows read | P(J ≠ J*) | stopping part β (debiased ± s.e.) | its Cauchy–Schwarz bound | MSE, certified read / vote alone |
|---|---|---|---|---|---|---|
| Llama-8B L1, 1.6K | FP8 per head (oracle) | 0.481 | 0.061 | −0.00001 ± 0.00004 | 0.0042 | 0.068 |
| Llama-8B L1, 1.6K | layer's FP8 median (run-time) | 0.500 | 0.052 | +0.00007 ± 0.00016 | 0.0069 | 0.073 |
| Qwen-30B L12, 1.6K | oracle | 0.872 | 0.052 | +0.00037 ± 0.00039 | 0.0094 | 0.003 |
| Qwen-30B L12, 1.6K | run-time | 0.571 | 0.044 | +0.00037 ± 0.00047 | 0.0077 | 0.009 |
| Llama-8B L3, 16K | oracle | 0.129 | 0.027 | +0.0013 ± 0.0009 | 0.0061 | 0.147 |
| Llama-8B L3, 16K | run-time | 0.130 | 0.034 | +0.0034 ± 0.0016 | 0.0163 | 0.126 |

- **The stop rarely moves:** it leaves the reference stop J* in 3–6% of draws. The stopping bias is zero within noise
  at 1.6K (|β| ≤ 4·10⁻⁴).
- **At 16K it is (1–3)·10⁻³, about 2 s.e.** That is small but perhaps real, and the CPU job doubles the draws.
  - Remark S′ turns β = 3·10⁻³ into a KL factor of about 1.1–1.3 at L = 32.
  - Proposition S's worst case gives 2.7.
- **The fixed vote set's β is 0 within noise,** as Theorem M's bias term says.
- **E1 is cheap where the vote is good.** On Llama L3 at 16K it reads 0.4% more rows than the vote and cuts the per-head
  MSE 7–8×.

## 6. The statistics

**6.1 Seeds per unit.** R7a's seed panel model (`../theory_r7.md` §6) is y_us = μ + a_u + c_s + f_us, with y the primary
statistic's per-unit term, u the unit and s the dither seed.
- **R6 used one seed for every unit.** Its interval covers μ + c_{s₀}: R6's design at that seed.
- **With an independent seed per unit** (a fixed hash of (task, prompt)), the y_u are i.i.d. with mean μ (E c_s = 0), so
  R6's unit bootstrap covers the design's expected KL over dither draws. That is C1's estimand. The cost is var(c_s)/n in
  the variance, negligible if R7a's panel finds c_s small.
- **Every layer keeps its own stream**, as R6 (seed + 2l for keys, + 1 for values), so Lemma C′ holds.
- **Common random numbers with E4.** R6's design and the single store use the same value quantizer. Giving them the
  same value seed per unit makes the paired difference measure the key quantizer and the vote alone. The key coins
  cannot be shared (different quantizers).

**6.2 Power under R6's rules** (`explore_r7b.py power`, `findings/power_r7b.md`). R6's per-unit KL of its design, FP8
and `fp_noise` (30 of R6's 32 blocks, 605 units; read-only) are resampled. The simulation:
- models the single store's KL as R6's design's, scaled by ρ above the family's `fp_noise` floor:
  KL′ = KL + (ρ − 1)·max(KL − floor, 0);
- applies R6's rule exactly (2,000-sample bootstrap percentile);
- labels each family jointly with `fp_noise` on the same units (PASS / FAIL / NOT_TESTABLE), then forms the group
  verdict.

| claim group | P(NONINFERIOR) at ρ = 1.0 (R6's design) | ρ = 1.14 (CPU median) | ρ = 1.5 | ρ = 2.0 | where it fails |
|---|---|---|---|---|---|
| Llama-3.1-8B, 128K | 1.00 | 1.00 | 1.00 | 1.00 | — |
| Qwen3-30B-A3B, 128K | 1.00 | 1.00 | 1.00 | 1.00 | — |
| Qwen3-30B-A3B, 32K | 0.64 | 0.58 | 0.38 | 0.37 | multivalue, every failure |

R6's planned sizes; 200–400 simulated runs per cell.

- **The binding family is Qwen 32K multivalue, not re-ranking.** There FP8's KL is tiny (0.0006 mean; most units
  ≈ 3·10⁻⁵), so the margin is essentially κ = 5·10⁻⁴ alone.
  - One unit of R6's 40 (prompt 9638) has design KL 0.0125, against FP8's 0.0002 and `fp_noise`'s 0.0033. R6's 4-bit arm
    had 0.0003 on the same unit, so this is a perturbation-sensitive unit with a heavy-tailed outcome across draws,
    not a precision effect.
  - Without it, the family passes with probability 1.00 at every n.
- **The lever is that family's size.** At ρ = 1.14 the pass probability is 0.55 at 40 units, 0.75 at 80, 0.89 at 120,
  0.95 at 200 and 0.99 at 300. It is testable (fp_noise passes at ≥ 0.97), so a miss would be a real FAIL.
  - Recommendation for R7b's freeze: about 150–200 multivalue units in Qwen 32K (revised to 300 below).
  - R7a's seed panel (b) on Qwen 32K multivalue units 9600–9609 would show how common such outcomes are across seeds.
    It has no results yet: its Trillium block was cancelled.
- **By level** (`explore_r7b.py power-mv`, `findings/power_mv_r7b.md`; 400 runs per cell).
  - R6's sensitive unit was at its hard level (R3a). R7b's 150 multivalue units are 130 at the standard level and 20
    at the hard level.
  - Two readings, with P(pass) at ρ = 1.14 / 1.3 / 1.5:

    | multivalue units | drawn per level from R6's units of that level | drawn from all 40 of R6's units |
    |---|---|---|
    | 150 | 0.98 / 0.95 / 0.94 | 0.93 / 0.84 / 0.69 |
    | 300 | 1.00 / 1.00 / 1.00 | 1.00 / 0.98 / 0.89 |

  - The second reading fits R7b better: the outcome was the dither draw's, and R7b draws a seed per unit.
  - Recommendation (revised 2026-10-12), taken by the user the same day: 300 multivalue units, +150 at the standard
    level, about +9 GPU-h (`findings/cost_r7b.md`; `lib_r7b.R7B_LAYOUT` `q-mva-j`).
- **Re-ranking is safe.** The design's KL there is 2.4× below FP8's on Llama (1.8× on Qwen 128K), and FP8's KL is large
  enough that the margin scales with it.
- **The accuracy secondary binds the Qwen sizes.** It needs 150 FP-correct units outside aggregation (3/δ). R6's
  FP-correct rates are 0.64 (Llama 128K), 0.60 (Qwen 128K) and 0.93 (Qwen 32K), so about 233, 250 and 162 units. R6 had
  270, 125 and 160 (ACC_UNDERPOWERED in both Qwen groups). Qwen 128K at 250 units is the costly part: node jobs.

**6.3 The gates (D3).**

*The test.* Per claim group and baseline b: d_u = KL_design,u − KL_b,u over all the group's units. The one-sided
bootstrap p-value is p_b = (#{bootstrap means ≥ 0} + 1)/(2,000 + 1), and Holm's step-down runs over the group's
baselines at α = 0.05: the i-th smallest p is compared with α/(k − i + 1). "Beats b" is claimed only where H₀ is
rejected. R6's rules have no gates, so this is R7b's addition. Each group is its own claim, as in R6.

*KIVI-4 and dense Lloyd-Max 4/4 (both at traffic 0.25), from R6's data* (design = R6's `dtail6ux_v4`, all units):

| group | baseline | units | mean(KL_design − KL_baseline) | one-sided 95% upper bound | units where the design is lower |
|---|---|---|---|---|---|
| Llama 128K | KIVI-4 | 289 | −0.0053 | −0.0045 | 0.99 |
| Llama 128K | dense 4/4 | 289 | −0.0074 | −0.0060 | 0.97 |
| Qwen 128K | KIVI-4 | 125 | −0.0040 | −0.0029 | 0.80 |
| Qwen 128K | dense 4/4 | 125 | −0.0064 | −0.0048 | 0.92 |
| Qwen 32K | KIVI-4 | 180 | −0.0016 | −0.0010 | 0.87 |
| Qwen 32K | dense 4/4 | 180 | −0.0027 | −0.0017 | 0.93 |

Their p-values are below 1/2,001 at R6's sizes, so both gates pass after Holm in every group, with a factor of 1.14 or
even 2 on the design's KL.

*When two tiers beat one (Corollary G).* Take a dense dithered tier with no exact rows, and the design reading S exactly.
To first order (Theorem M's key and value terms, small σ), at one head-step:

  MSE_dense / MSE_design = (ρ_K·K + ρ_V·V) / (s_K·K + s_V·V) ≥ min(ρ_K/s_K, ρ_V/s_V),

where:
- K = Σ_i w_i²·σ_i²·‖u_i‖² and V = Σ_i w_i²·n2_i are the dense key and value terms at the design's steps, over all rows;
- s_K and s_V are their unread shares (s_K is share2 when σ_i is the same across rows);
- ρ_K and ρ_V are the dense tier's step² over the design's.

*Proof.* Theorem M's terms are linear in the per-row variances, and the mediant inequality bounds the ratio. ∎

For dense dithered 7/6 against the single store 4+2/4: ρ_K ≈ (63/127)²/1.1–1.3 ≈ 0.19–0.22 and ρ_V = (15/63)² = 0.057.
So the design wins at a head-step when its unread shares are below about 0.06–0.2.
- That holds at almost every head-step: the median dense 7/6 / design MSE is 15–170 on every view except Qwen's flat
  L1 (§5.4).
- **It fails at the flat heads,** where the vote leaves much of the second moment unread. Those heads dominate the sum
  over head-steps:
  - dense 7/6's total MSE is only 0.4–4.2× the design's at 1.6K;
  - at 16K it is 28–46× on Llama and 38× on Qwen L12;
  - it stays below 1 on Qwen's flat L1 at every length (0.3–0.7).
- So gate (ii) against dense 7/6 is decided by how sensitive the flat heads are, which no single-layer view measures.
  The one end-to-end number (Llama-1B, 1.6K, CPU) has dense 6/6 at 3.9× the design's KL. Dense 7/6's keys are 4× finer.
  The design's total improves with context while dense 7/6's does not (§5.4).

Recommendation for R7b's freeze (revised after the full-depth totals, §5.4b), taken by the user on 2026-10-12: **test
the dense 7/6 gate in every group** (`read_r7b.GATE_DENSE_GROUPS`).
- Summed over every layer at 32K, dense 7/6 has 10.7× (Llama) and 136× (Qwen) the design's MSE. Qwen's flat layers
  carry under 0.3% of the design's error.
- Holm costs the other gates nothing. KIVI-4, Lloyd-Max 4/4 and Quest have p ≈ 1/2,001, below α/4, so they are
  rejected first whatever dense 7/6's p is.
- The least certain group is Llama 128K. There, 81% of the design's MSE sits in two layers where dense 7/6 is 0.4–0.9×
  as good, so a NOT_SHOWN there would mean those layers are the sensitive ones.

*Quest at 0.398.* Quest's metadata costs 2/32 of the traffic, so it reads 0.336 of the rows. R4's Quest reads the first
two layers densely and pages of 16 rows elsewhere: 0.29 per selected layer on Llama, 0.31 on Qwen. Its error is the
dropped rows' (deterministic, biased).
- On concentrated layers its per-head MSE is 124–1,248× the design's (§5.4). On Llama 128K's multivalue and
  tracking, R5 measured Quest at 1/8 with KL 0.02–0.14 against the design's 0.0003.
- Its exact dense layers 0–1 are where the design is weakest, so the gate is decided in layers ≥ 2.
- **Prediction:** PASS in every group, by a margin far larger than the dense 7/6 gate's.

## 7. Predictions for R7b (stated before the run)

| id | prediction | derivation |
|---|---|---|
| P-b1 | **Primary.** NONINFERIOR in Llama 128K and Qwen 128K. Qwen 32K NONINFERIOR with probability about 0.6 at R6's sizes; 0.84–0.95 with 150 multivalue units and 0.98–1.00 with R7b's 300, at ρ = 1.3 | §6.2: R6's per-unit data under R6's rule, with the single store's KL at ρ = 0.9–1.5 × R6's design above the noise floor; by level, `findings/power_mv_r7b.md` |
| P-b2 | **E4.** KL(single store) / KL(R6's design) is 1.1–1.4 per unit (geometric mean) and 0.9–1.2 on mean KL | §5.2: head-level sums 1.08–1.48 (KL proxy 1.15–1.58); every layer at 32K: 1.21 / 1.15 (KL proxy 1.31 / 1.24) on Llama / Qwen (§5.4b); CPU end to end 1.12–1.27 per unit, 0.91–1.15 on mean KL (findings_nested §6). Mean KL is diluted by the `fp_noise` floor: in R6 the design's KL is within 1.5× of `fp_noise`'s in 6 of 7 Llama families (all but re-ranking) |
| P-b3 | **Accuracy.** ACC_PASS in Llama 128K. ACC_UNDERPOWERED in a Qwen group unless it has ≥ 150 FP-correct units outside aggregation (about 250 units in Qwen 128K, 162 in Qwen 32K) | §6.2 |
| P-b4 | **Gates.** KIVI-4 and dense Lloyd-Max 4/4 pass after Holm in every group. Quest at 0.398 passes in every group. Dense dithered 7/6 passes in every group, least surely on Llama 128K | §6.3 (R6's data); §5.4, §5.4b: dense 7/6 has 10.7× (Llama) and 136× (Qwen) the design's MSE over every layer at 32K; on Llama the design's error sits in two layers where dense 7/6 is close (Corollary G) |
| P-b5 | **Memory.** 0.324 at rest and 0.449 during a question (0.3125 / 0.4375 without the per-row scalars), against FP8's 0.50 and R6's design's 0.578 / 0.703 | §2.6 (accounting, exact) |
| P-b6 | **R1** holds in the deployed system; nothing to test | §4.1 (Lemma C′) |
| P-b7 | **E1.** On Llama 128K, with a flat or FP8-quantile τ at a level a few times below the design's own KL: traffic ≤ 0.41, KL 4–20× below the single store's and within about 1.1–1.3× of its calibrated level. On Qwen, traffic +0.05–0.1 unless its noisy layers get 4+4 keys or skip the read. Everywhere, per-query-head misses within δ | CPU: +0.001 to +0.007 traffic for 5.6–25× lower KL (findings_klbridge §6.1); §5.3, §5.5 (Llama L3 at 16K: 0.130 of rows; β ≈ 10⁻³); §4.3 (Remark S′; Qwen's vacuous layers) |
| P-b8 | **E2.** KL(4+3) / KL(4+2) is 0.4–0.8, for +0.027 traffic | §5.2: MSE 0.42–0.76; CPU 0.39–0.57 against uniform 6/4 |
| P-b9 | **E3.** Water-filling against uniform 4+1/4 at equal traffic: at most about 20% lower KL | CPU 16%, held out (findings_klbridge §3) |

Not predicted to hold: FP8-level per-head certificates in Qwen's loose layers at 4+2 (L1: 0.07 at 16K; §5.4b), which
is why E1 gives them 4+4 keys.

## 8. What each outcome would mean

| outcome | meaning, and what follows |
|---|---|
| Primary NONINFERIOR in all three groups | the memory claim (0.324 at rest, 0.449 during a question, against FP8's 0.50) is confirmatory, at R6's traffic |
| Qwen 32K NOT_SHOWN from multivalue alone | expected with probability 0.4 at R6's size, 0.05–0.16 at 150 units and ≤ 0.02 at 300 (ρ = 1.3; §6.2): the sensitive-unit tail, not the store. Read the failing units' `fp_noise` and the E4 pair before attributing it to the single store |
| E4 ratio within the head-level prediction | Theorem N's cost (about 0.2 bit of key precision) transfers end to end at scale |
| E4 ratio far above it in some family | look at the vote first (rows in common with R6's store vote, Lemma V), then at the flat layers |
| Dense 7/6 gate fails on Qwen | Corollary G's flat-head regime. "Two tiers beat one" holds per head, not in total, on flat-attention models. The remedy is more bits or a larger read in the flat layers (E3, E1), not a different bound |
| E1 KL above its budget by more than Proposition S's factor at the measured β | the second-order bridge fails at scale (Qwen's compounding layers; findings_klbridge §5), not the certificate |
| E1 misses above δ per query head | a code or theorem failure: stop and find the head-steps |

## 9. Limits

- **Single layers, at most 16K.** The views cover two layers per model. Totals over layers, and 128K, are extrapolations.
  The share2 trend with context (§5) is the basis.
- **No sensitivities at scale.** Head-level MSE ratios are not KL ratios. The KL weighting uses c^V/c^K = 0.46 from
  Llama-1B. The deterministic baselines' (FP8, KIVI-4, Lloyd-Max 4/4, Quest) per-unit-MSE harm differs from the tail's
  (0.2–1.6× on the CPU).
- **The vote's quality** is measured per layer with exact queries. Its compounding through the question pass over
  layers is measured only end to end.
- **The power analysis** resamples R6's units: R6's tasks and difficulties, one seed per unit, 30 of 32 blocks.
  R7b's fresh prompts and per-unit seeds may differ. ρ models the single store as a scaled design.
- **The stopping bias** is measured on 2 KV heads × 4 steps per view. β's estimates carry the noise of 80 draws.
- **Emulation:** bf16 storage of the dequantized tier (§2.5), as R6.
