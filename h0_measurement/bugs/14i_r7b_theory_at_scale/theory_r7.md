# R7a's theory: what the run tests, why its GPU forms are valid, and what it should find (2026-10-11)

R7a is rank 1 of `../14h_theory/RESEARCH_PLAN.md` (§1.1, §4). It puts the theory section's headline results on the
paper's own models and lengths (Llama-3.1-8B at 128K, Qwen3-30B-A3B at 32K and 128K: all three of R6's claim groups),
where they now exist only on CPU proxies (1–2B models, ≤ 8K tokens):
- part 8's certificate (theory.md §6f–§6g);
- the certified adaptive read (§6e);
- Theorem M (§10).

It also closes the claims ledger's C1 gap: R6 used one dither seed.

R7a tests no design claim. The statements are `../14h_theory/theory.md`'s, unchanged. This file adds five things:
1. which measurement tests which statement (§1);
2. three lemmas that make the GPU forms valid, or exact against the reference (§2);
3. how the theory's quantities scale with context on R6's own models, measured on CPU (§3);
4. the predictions, stated before the run, and how each was derived (§4);
5. the seed panel's model (§6).

Notation as theory.md: weights w_i = e^{s_i}, tier-1 weights w̃_i, Z_i = w̃_i/w_i, the read set S (the vote's 1/8), the
unread rows T, the exact output o, the design's output õ, its normaliser Z̃, FP8's per-head error e₈.

## 1. What each measurement tests

The protocol meets every assumption:
- **(R1) Coin-free reads.** The vote's rows, the certified read's schedule and remedy 1's heavy rows all come from the
  Lloyd-Max 4/4 store, a deterministic quantizer.
- **(R2) Coin-free queries.** The probe measures on FP's teacher-forced pass and changes no output, so every query is
  FP's and depends on no coin. This is the oblivious case. For the design's own queries, which depend on lower layers'
  coins, Lemma C gives the same guarantee given those coins; the CPU's `adaptive_s1h5` found no measurable difference.
- **The seeds.** A fixed formula (1000003·draw + 2·layer), on fresh inputs.

| statement | what R7a measures (probe_r7 / read_r7a) | what would contradict it |
|---|---|---|
| Theorem R, part 8 form and v2 form (T2, T3) | per head-step, over D draws: the share of draws with ‖o − õ‖ ≤ bound (`cov_p8`, `cov_v2`), at 4/6/8-bit keys (D = 2/5/3) | more misses than δ·checks (δ = 10⁻³): a bug or a false theorem |
| T3 tightness | median bound / error (`rat_*`); the share certified within FP8's per-head error (`f8_*`); part 8's K/V/D composition | — (descriptive; P2, P3, P6 in §4) |
| T6, the certified adaptive read | per (KV head, step) of the first chunk, draw 0, 6 and 8 bits: the stop J over the FINE schedule (16 coin-free sets, each at δ/16), rows read, the oracle stop, how many query heads' outputs met FP8's error | more query heads missing than δ·heads (beyond noise); the guarantee is per query head (§2.6) |
| T4, Theorem M, exact | `nn` = mean over draws of (Z̃/Z)²‖o − õ‖²: unbiased for m_pred = E‖N‖²/Z² **exactly** (N = Z̃(o − õ) is an identity) | mean nn/m_pred inconsistent with 1: an implementation error at scale |
| T4, its first-order reading | `mse` = mean ‖o − õ‖² against m_pred | a drift beyond ±5% at 6–8 bits |
| "The unread second moment sets the error" | `share2` = Σ_T w²‖v − o‖² / Σ w²‖v − o‖² by family, against R6's design KL per family | no positive rank correlation |
| T4's layer map at scale | per layer: √m_pred / e₈ at 4/6/8 bits | — (descriptive) |
| C1's seed gap | panel (b): KL of the design at R6's seed and four more, on R6's units, in one process with FP8 | a family that passes at R6's seed but fails at another seed (§6) |

## 2. The GPU forms, and why each keeps the guarantee

R7a computes the CPU study's bounds, unchanged in meaning. Three steps that were Python loops or memory-bound on CPU
needed new forms.

**2.1 Copies.** The pieces of `cert_s1h5`/`rquant_s1h5` that part 8 builds on are copied into `cert_r7`/`tier_r7`:
- `test_cert_r7.test_copies` checks them bit for bit;
- part 8, the certified read and Theorem M match `../14h_theory`'s reference outputs to 5·10⁻¹⁴ on frozen inputs
  (`fixtures/`), including a slice of Llama-3.1-8B layer 1.

**2.2 Lemma S (the HKZ matrix from its heavy rows).** Let M = Σ_{i∈T} cw_i u_i u_iᵀ with cw_i ≥ 0. Split T into any H
and L = T \ H, with M_H and M_L their parts. Then

  ‖M‖_op ≤ ‖M_H‖_op + tr M_L  and  ‖M‖_F ≤ ‖M_H‖_F + tr M_L.

*Proof.* Both norms obey the triangle inequality. M_L is positive semidefinite, so ‖M_L‖_op ≤ ‖M_L‖_F ≤ tr M_L,
because Σλ² ≤ (Σλ)² when every λ ≥ 0. ∎

The HKZ radius √(tr M + 2√(‖M‖_F²x) + 2‖M‖_op x) increases in both norms, so the bound stays valid. The inequality is
deterministic and applies to the computed matrix (sure weights). The partition may therefore depend on anything,
the coins included; no new event is spent.

Cost: the matrix needs H rows instead of C. Measured inflation of the bound (Llama-3.1-8B L1, 6-bit keys):
- with H = 4,096: 1.0008× at 8K and 1.008× at 16K;
- with H = 1,024: 1.05× at 8K and 1.10× at 16K.

The inflation grows with C, so **R7a keeps the exact matrix**: it is one batched fp64 matmul per chunk of rows,
≈ 5·10¹¹ flops per 128 head-steps at 128K. The split stays available (`CR_GRAM_ROWS`) if the gate shows the
certified read too slow.

**2.3 Lemma B (deciding the certified read before forming the matrix).** For a positive semidefinite d×d matrix with
trace τ:

  τ/d ≤ ‖M‖_op ≤ τ  and  τ²/d ≤ ‖M‖_F² ≤ τ².

Part 8's bound B(‖M‖_F², ‖M‖_op) is non-decreasing in both. So B(τ²/d, τ/d) ≤ B ≤ B(τ², τ).

The rule's decision at set j is "B_j ≤ target at every query head". It is settled without M in two cases:
- if the upper bracket meets the target at every query head, the set is accepted;
- if the lower bracket misses it at some query head, the set is rejected.

M is formed only for the undecided (KV head, step) groups, and only while the group's stop is still open. **The stop J
is the naive rule's, up to floating point.** The one difference: P-exact is computed for the open groups' rows only, so
its batches differ, and a batch's series length is its longest row's. A longer series only tightens a proxy, by far
less than 10⁻¹² here, so the two rules can differ only where a bound sits on the target to that precision. Both are valid
bounds. `test_cert_r7` checks identical stops against a naive implementation with brackets on and off, and against the
reference's `certified_read` on the fixtures. The brackets saved 75% of the matrices in the synthetic tests and 50–90%
on real layers.

**2.4 Lemma P (Theorem M's exact moments in O(C) per head-step).** With g(x) = log(sinh x / x) and a_ij = α_j D_i
(α_j = scaling·|(Rq)_j|/2, D_i the row's step):

  Σ_j g(k α_j D_i) = Σ_{m≥1} c_m (k D_i)^{2m} Σ_j α_j^{2m}  for k·max_j α_j D_i < π,

with c_m = 2^{2m}B_{2m}/(2m(2m)!) = (−1)^{m+1}ζ(2m)/(m π^{2m}); the first are 1/6, −1/180 and 1/2835.

Var Z_i = (E Z_i)²·expm1(Δ_i), with Δ_i = Σ_j [g(2a_ij) − 2g(a_ij)] = Σ_m c_m(4^m − 2)(α_j D_i)^{2m} summed as a series,
so nothing cancels. The series is used where 2·max_j a_ij ≤ 2; there its tail after 40 terms is
below d·(2/π)^{82}/0.6 ≈ 10⁻¹⁴. The other rows use the closed form.

Checked against the closed form to 2.7·10⁻¹³ from near-zero scores to past the range. This cures the near-zero-score
precision issue that theory3's `_log_sinhc` fixed separately. It is a numerical method for a prediction, not a
certificate input.

**2.5 Theorem M's identity as an exactness check.** Z̃(o − õ) = N holds exactly. So (Z̃/Z)²‖o − õ‖² averaged over
draws estimates E‖N‖²/Z² = m_pred without bias, at any σ and any key width. The reader's R_exact = mean(nn/m_pred) must be
consistent with 1. Its interval comes from a unit bootstrap: head-steps of one unit and layer share their draws. Only
R_first = mean(mse/m_pred) involves the first-order step Z̃ ≈ Z.

**2.6 The certified read's guarantee is per query head.** Each set's bound B_j holds per head-step at δ/m. For one
query head h, the event E_h that every B_j^h ≥ err_j^h has P(E_h) ≥ 1 − m·δ/m = 1 − δ. On E_h, whatever stop J the coins
choose, err_J^h ≤ B_J^h ≤ τ_J^h, because J meets the target at every head. So P(err_J^h > τ_J^h) ≤ δ for each query
head, and for all r heads of a KV group at once only ≥ 1 − rδ (union bound).

`../14h_theory/theory.md` §6e's "with probability ≥ 1 − δ, … at every query head" should read "at each query head"; the
joint statement needs δ/(m·r) per bound. R7a tests the per-head statement: misses against δ·heads with the coverage
rule's noise allowance. It reports the share of groups with every head met beside it.

## 3. Scaling with context on R6's own models (CPU, `explore_r7.py`, `findings/explore_r7.md`)

Setup: the CPU study's (PG-19; the last 48 positions fixed; 32 voters over a Lloyd-Max 4-bit sketch pick the vote's 1/8; 16
measured queries; δ = 10⁻³; 2 draws).
- Models: Llama-3.1-8B L1 and L3, and Qwen3-30B-A3B L1 (a flat layer) and L12 (a typical one), each truncated.
- Contexts: 1.6K, 4K, 8K and 16K tokens.
- Coverage is 1.000 in all 22 cells.

| model, layer | context | key bits | part 8 / error | v2 / error | certified within FP8 (p8 / v2) | unread share of Σw²‖v−o‖² | tail / FP8 (pred.) | certified read: rows (traffic) | oracle stop |
|---|---|---|---|---|---|---|---|---|---|
| Llama-8B L1 | 1.6K | 6 | 5.11 | 27.2 | 0.35 / 0.00 | 0.0012 | 0.26 | 0.459 (0.628) | 0.131 |
| Llama-8B L1 | 4K | 6 | 5.09 | 24.4 | 0.47 / 0.05 | 0.0011 | 0.21 | 0.349 (0.552) | 0.126 |
| Llama-8B L1 | 8K | 6 | 5.39 | 24.6 | 0.60 / 0.10 | 0.0006 | 0.16 | 0.230 (0.470) | 0.125 |
| Llama-8B L1 | 16K | 6 | 5.45 | 24.6 | 0.66 / 0.12 | 0.0003 | 0.14 | 0.209 (0.456) | 0.125 |
| Llama-8B L3 | 16K | 6 | 4.60 | 24.0 | 0.97 / 0.43 | 0.0001 | 0.05 | 0.128 (0.400) | 0.125 |
| Llama-8B L1 | 16K | 8 | 2.05 | 12.6 | 0.96 / 0.34 | 0.0003 | 0.11 | 0.132 (0.457) | 0.125 |
| Llama-8B L1 | 16K | 4 | 183 | 528 | 0 / 0 | 0.0003 | 0.39 | — | — |
| Qwen-30B L12 | 16K | 6 | 14.8 | 62.5 | 0.78 / 0.38 | 0.0005 | 0.02 | 0.244 (0.480) | 0.125 |
| Qwen-30B L12 | 16K | 8 | 2.92 | 18.2 | 0.99 / 0.81 | 0.0005 | 0.01 | 0.126 (0.453) | 0.125 |
| Qwen-30B L1 (flat) | 16K | 6 | 11.9 | 50.3 | 0.09 / 0.04 | 0.129 | 0.63 | 0.748 (0.827) | 0.137 |

What it shows:
1. **Part 8's tightness is flat in context.** Llama L1 goes 5.1 → 5.45, L3 4.2 → 4.6, Qwen L12 13.3 → 14.8. The bound
   carries the sure weights' e^{t}, t = zσ, and z = √(2 ln(4|T|/δ)) grows like √ln C: from 5.6 at 1.4K rows to 6.0 at 16K
   and 6.3 at 128K. That is +3% from 16K to 128K at σ = 0.084. σ itself depends on the key width and the query norm,
   not on C.
2. **v2 is flat too, on the same model.** It is 27× at 1.6K, 24.6× at 16K, and 25.4× at 128K on the GPU (R6).
   `RESEARCH_PLAN.md` §1.1's "39–42× on the CPU against 25× on the GPU" compared Llama-3.2-1B with Llama-3.1-8B. Scale
   does not hurt, but the gap was a model difference, not a length effect. Part 8 / v2 is 0.19–0.22 on Llama and 0.24 on
   Qwen at 6 bits, and 0.14–0.16 at 8 bits.
3. **Certification gets easier with context.** The bound / error ratio is flat, but the tail's error falls relative to
   FP8's (0.26 → 0.14 on L1, 6 bits). By Theorem M, the tail's MSE is the unread rows' share of Σ w²‖v − o‖² times the
   tier's per-row error, and that share falls as the vote's 1/8 captures more of the second moment (0.0012 → 0.0003).
   FP8's error is a dense error: its second moment runs over every row's w², so no read set cuts it. The tail / FP8
   ratio therefore scales roughly like √share2, and the certified share P(err/e₈ ≤ 1/ratio) rises: 0.35 → 0.66 on L1
   and 0.62 → 0.97 on L3.
4. **The certified read approaches the oracle stop.** At 6 bits, L1 needs 0.459 → 0.209 of rows. L3 at 16K needs 0.128
   (traffic 0.400 against the oracle's 0.398), because the vote's own set is certified for most head-steps.
5. **Flat layers stay the exception.** Qwen's L1 leaves 13–33% of the second moment unread. There the certified read at
   6 bits needs 0.75–0.99 of rows against the oracle's 0.14–0.31 (MODEL_FAMILY's flat-layer limit). Longer context helps
   (share2 0.33 → 0.13), slowly.
6. **Theorem M holds at every length.** Mean measured / predicted is 0.98–1.01 at 6–8 bits. At 4 bits it is 1.01–1.14
   with 2 draws: the noise of the first-order reading where σ ≈ 0.35.

## 4. Predictions for R7a (stated before the run; `read_r7a.py` labels each MET or NOT_MET)

Sources:
- §3's CPU factors;
- R6's GPU probe at the paper's scale: six of its seven probe blocks, 2.9 M head-steps. They give, per head-step, v2's
  bound at the vote's rows and at C/128, C/32 and C/8 more, and FP8's error (`findings/predictions_from_r6.json`).

The R6-based numbers scale each head-step's v2 bound by the CPU's part 8 / v2 factor f. They assume f transfers. R7a
measures it.

| id | prediction | derivation |
|---|---|---|
| P1 | coverage COVER_OK for part 8 and v2 at every width and group (0 misses expected) | the bounds are 2–15× the error at 6–8 bits; the brute force and 46.6 M R6 checks never failed |
| P2 | part 8's median bound / error at 6 bits in [4, 7] on Llama 128K, [6, 13] on Qwen 32K and Qwen 128K | 25.4 × f(0.20–0.22) = 5.1–5.6; 37.9 × f(0.24) ≈ 9.1; §3.1's +3% drift. Qwen 128K: tightness is flat in context (§3.1), so Qwen 32K's band |
| P3 | part 8 certified within FP8 at 6 bits ≥ 0.80 (Llama), ≥ 0.65 (Qwen 32K, Qwen 128K) | R6's head-steps with f = 0.18–0.25: Llama 0.88–0.91 (v2: 0.625), Qwen 0.71–0.76 (v2: 0.487). Qwen's weakest layers: L3 (0.003), L1 (0.06), L2, L0, L16. Qwen 128K: certification improves with context (§3.3), so at least Qwen 32K's threshold; extrapolated, as R6 did not probe Qwen 128K |
| P4 | Theorem M: R_exact's 95% interval meets [0.98, 1.02] at every width; R_first in [0.95, 1.05] at 6 and 8 bits | §2.5 (exact); §3.6 |
| P5 | certified read at 6 bits: traffic ≤ 0.45 (Llama), ≤ 0.50 (Qwen 32K, Qwen 128K); per-query-head misses COVER_OK (§2.6) | R6's coarse sets with 1.15·f (δ/16 per set): Llama 0.127 rows, traffic 0.400, stopping at the vote's set in 70–73% of groups; Qwen 0.25 rows, 0.49 |
| P6 | part 8 / v2 ≤ 0.30 (median, 6 bits) | §3.2 |
| P7 | SEED_ROBUST in every family of panel (b) | §6 |
| P8 | the unread share orders R6's families by the design's KL (Spearman > 0) | Theorem M: the tail's MSE is the unread share times a dense tier's |

Not predicted to pass: FP8-level certification at 4-bit keys. Part 8 there is 100–500× the error (§3; T7).

## 5. What each outcome would mean

| outcome | meaning, and what follows |
|---|---|
| P1 NOT_MET | a theorem or code failure; the brute force never failed. Stop and find the head-steps; nothing else of R7a is reported until it is understood |
| P4 exact NOT_MET | an implementation error at scale (the identity is exact); Theorem M's GPU numbers are withheld |
| P2/P3/P6 better than predicted | the theory section quotes R7a's numbers; f does not transfer, in the good direction |
| P2/P3 worse | the same, and theory.md's CPU claims carry the caveat; check the K/V/D composition (`pK`, `pV`, `pD`) and σ per layer for where the slack goes |
| P5 MET | "certified FP8-level per-head error at below FP8's traffic, at 6-bit keys" holds on the paper's models with an oracle target (T6). The run-time target τ is still open (RESEARCH_PLAN §3 τ) |
| P5 NOT_MET (Qwen) | expected first in the flat early layers L0–L3. Report the per-layer cost; the flat-layer remedy is more value bits, not a tighter bound |
| Qwen 128K differs from Qwen 32K | the HELMET families (RAG, ICL, re-ranking) rather than RULER's; and context (§3). Read it per family, against Llama 128K's HELMET block, before attributing it to length |
| P7 NOT_MET | C1 holds for R6's seed, not for the design. Restate C1 conditionally, or run R6's statistic with seed draws |
| P8 NOT_MET | the unread share is not the family-level predictor at 128K. Look at the value term (flat attention) and at c_h (the KL bridge, RESEARCH_PLAN §3 KL) |

## 6. The seed panel's model (panel b)

On R6's own units (re-ranking 10, multi-value 10, retrieval 10 at Llama 128K; multi-value 10, retrieval 10 at Qwen 32K),
one process runs:
- fp;
- the design at R6's seed (7000) and at four more (7000 + 100003·k);
- FP8 and fp_noise.

With y_us = KL_tok(design, seed s) − 1.5·KL_tok(FP8):

  y_us = μ + a_u + c_s + f_us.

- **a_u, the unit, and f_us, the rest, are already in R6's confidence interval.** A fixed seed still gives every unit
  its own effective draw, because the dither acts on that unit's cache.
- **c_s, a seed effect common to every unit, is not.** It would come from rows that are nearly the same in every input
  and sit in the unread tail. The attention sink is the obvious candidate, but it is almost always among the vote's
  rows, which are read exactly, so it is not perturbed. Formatting and BOS-like tokens in the tail remain.

The reader estimates var(c_s) by two-way ANOVA. It computes R6's statistic U_s at each seed, and U with one seed drawn
per bootstrap replicate. **SEED_ROBUST** means every seed passes R6's margin wherever R6's seed does.

Expectation: c_s is small and every family is SEED_ROBUST. In R6, the panel's families have mean y of −0.0056
(re-ranking), −0.0098 (multi-value) and −0.0029 (retrieval) on Llama, against a margin of +5·10⁻⁴. To cross it at one
seed, in every unit at once, the design's KL would have to rise about 4× on re-ranking (0.0021 → 0.0082) and about 35× on
the other two.

The Llama-3.2-1B smoke shows why the panel is still needed. On one unit, the design's KL per token ranged from
5·10⁻⁵ to 6.4·10⁻⁴ across the five seeds. Per-unit seed variability is large; whether any of it is common to all
units is what R7a measures.

## 7. Limits of these predictions

- **Different inputs.** The CPU exploration uses PG-19 and a sketch vote; R7a uses RULER and HELMET with the question's
  vote. The predictions transfer ratios (bound / error, part 8 / v2), not absolute errors. The R6-based ones use the
  real inputs.
- **Few layers.** The CPU covers two layers per model; R7a covers every layer. Qwen's QK-norm makes σ vary by layer
  (`../14h_theory/findings/findings_qwen.md`).
- **The target is an oracle.** The certified read is measured against FP8's per-head error, as on the CPU (T6).
- **One draw for the certified read** (draw 0, the first chunk of steps). Its met share is still a guarantee per query
  head (§2.6).
- **Qwen 128K's predictions are extrapolated.** R6 has no probe there, so P2, P3 and P5 for Qwen 128K reuse Qwen 32K's
  thresholds, justified by §3's flat tightness and improving certification.
- **FP's queries.** The coverage is measured on the oblivious pass (§1). The design's own queries are covered by Lemma C,
  not measured here.
