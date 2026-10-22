# Certified two-tier KV reads: the theory (updated 2026-10-10, parts 6–8; §13–§14 2026-10-11, CPU checks 2026-10-12; R7a's GPU numbers and §6e's per-head correction 2026-10-22)

This file states what is proved about the two-tier KV design, under which assumptions, and how
strong each result is in practice. **Every R14 theory update goes in this folder** (since 2026-10-10; `CLAUDE.md`).

The run-time library stays in `../14h_methodology_improve/`, because GPU runs import it and freeze it per run:
`cert_s1h5.py` (every bound through Theorem R v2, Bennett and the coupled MLA form) and `rquant_s1h5.py` (the
randomized tier). This folder:
- `theory2_s1h5.py`: the instrumented Theorem R (v2's development form);
- `adapters_s1h5.py`: the architectures;
- `adaptive_s1h5.py` and `oracle_s1h5.py`: the adaptive-query and oracle-width experiments;
- part 6: `theory3_s1h5.py` (Theorem R v3, Theorem M) and `exp4bit_s1h5.py` (the 4-bit study);
- part 7: `theory4_s1h5.py` (the sure-sum bound, the mirror split, the certified adaptive read);
- part 8: `theory5_s1h5.py` (the tilted tail, the Hsu–Kakade–Zhang key sum and its proxies);
- `layermap_s1h5.py`: Theorem M's per-layer map of a model from one FP pass;
- `mla_m_s1h5.py`: Theorem M for MLA (the coupled key and value errors), with its Monte Carlo check.

Findings: `findings/` (`findings_4bit`, `findings_part7`, `findings_part8`, `findings_qwen`, `findings_architectures`,
`findings_mla_m`, `layermap_*`, `R5_adapters`, `R5_theory_mla`). Tests here:
`test_theory_s1h5.py`, `test_theory3_s1h5.py`, `test_theory4_s1h5.py`, `test_theory5_s1h5.py`, `test_adapters_s1h5.py`; the run-time bounds'
brute force is `../14h_methodology_improve/test_r14_stage1h_r5.py`. Run records: `../14h_methodology_improve/plan.md`,
amendments "R5 theory" parts 1–6 (that folder's part 6 is this file's part 7).

> **Correction (2026-10-12): every Gemma-3-12B number in this file before that date came from a randomly initialized model.**
> `adapters_s1h5._load` built `Gemma3ForCausalLM` from the multimodal checkpoint. transformers 5.16 does not map that checkpoint's `language_model.*` keys onto it, so every text weight was freshly initialized, and that branch skipped the missing-weight check. Random weights give flat attention.
> - Gemma 3's "flat, value-leaning" class (family E), its 6/6 choice, L5 as the flat-attention example (66% of the second moment unread) and R8's Gemma 3 predictions all describe the random model.
> - With the checkpoint's weights, L5 at 1,600 tokens has 7.6% of the second moment unread (between, not flat; unread mass 0.20, against 0.77) and a key share of 0.86 at 6/6 (key-leaning, against 0.37). Uniform 6/4 meets FP8's error at the median head-step (0.37×), not at p90 (1.36×).
> - The theorems' checks on those views (coverage, Theorem M's accuracy) remain valid statements about those inputs, but they are not evidence about Gemma 3. Rows marked † below are the random model's; the explanatory examples now use Granite-4.2 L1 (a real flat layer) and Gemma 3's real numbers.
> - Fixed: the loader uses the multimodal class and refuses any missing weight; `g0map_s1h5.provenance` compares loaded weights with the checkpoint's. No other model was affected: the 18 GQA/MHA models and the two MLA models load with no missing weight. G0 (`findings/findings_g0.md`) recomputes Gemma 3 at 1,600 tokens and at 32K.

## Summary

| Result | Holds for | Assumes | Useful on real models? |
|---|---|---|---|
| Lemma 1: eviction error = missed mass × value gap | every input | nothing | explains R1–R5 |
| Lemma 2: a computable bound on the missed mass | every input | per-row key-error norms stored | yes (2–4× the oracle's budget at ε = 0.01) |
| Theorem D: the controller stays within τ or reports | every input and model | as Lemma 2 | correct, but vacuous for the tail at 4 bits |
| Lemmas 3–4: tail error, deterministic | every input | as Lemma 2, plus value-error norms | no: they carry e^b with b ≈ 8 nats |
| Lemma 5: tail error, concentration | inputs that fit an error model | zero-mean, independent tier-1 errors | ≈ 15–60× tight, but real keys break the model |
| **Theorem R** (v3): tail error, randomized tier | **every input and model** | read rows don't see tier 1's coins; seed independent of the input | with 6–8-bit tier-1 keys: 6–17× the best possible certificate (GPT-OSS at 6 bits: 39×) (§7); at 4 bits ≈ 50–160× on Llama, 12× on Gemma 3, ≈ 7,500× on GPT-OSS |
| **Part 7: sure sums, the mirror split** (§6c–§6d) | every input; the split needs independent key and value dithers (GQA) | as Theorem R | Llama-1B 8-bit 5.6–5.7× the error (v3: 9.8–10.4×); 6-bit about 18× |
| **Part 8: the tilted tail, a Hsu–Kakade–Zhang key sum** (§6f–§6g) | every input; GQA (as the split) | as Theorem R | the tightest certificates at every width: Llama-1B 6-bit 6.7–7.3× the error (split 17–19×), 0.41–0.59 of head-steps certified at FP8's level (split 0.13–0.21); 8-bit 2.0–2.6× (0.89–0.95 certified on L1/L3); 4-bit 92–554×, still far from FP8. **At the paper's scale** (R7a, GPU): 6-bit 5.6× on Llama-3.1-8B at 128K and 7.2–8.0× on Qwen3-30B-A3B at 32K/128K, certifying 0.92 / 0.72 / 0.68 of head-steps; no miss in 37.8 M checks (§7) |
| **Certified adaptive read** (§6e, corollary) | every input; the guarantee is per query head | a coin-free nested schedule of read sets | the guarantee at any target, at a cost the coins decide. With part 8's bound and FP8's per-head error as an (oracle) target: traffic 0.456–0.460 at 8-bit keys (the oracle stop is 0.453) and 0.433–0.499 at 6-bit keys on Llama-1B L1/L3 at 1.6K–8K (FP8: 0.50) (§7). **At the paper's scale** (R7a): 6-bit traffic 0.399 on Llama-3.1-8B at 128K (the oracle's 0.399), 0.474 / 0.423 on Qwen3-30B-A3B at 32K / 128K; no query head missed in 774,656 checks |
| **Theorem M**: the tail's mean squared error, exactly | **every input and model** (GQA; MLA in its coupled form, §10) | as Theorem R | predicts the measured MSE within 1% at 6–8-bit keys on every layer tested (Llama, Gemma 3, GPT-OSS, Qwen3); its first-order reading overpredicts where σ ≈ 2 (4-bit keys in two Qwen layers) (§10). At the paper's scale (R7a, 3.78 M head-steps): exact, 1.000 ± 0.001 at 6–8-bit keys; the unread share ranks Llama's task families by measured KL (Spearman 0.857). It explains why the design works and maps which layers need which bits (`layermap_s1h5.py`) |
| **Lemma N, Theorem N**: one store for the vote and the tail (§13) | every input; GQA | the vote reads only the deterministic base | Theorems R and M hold with per-coordinate steps; R1 by construction; memory 0.31 at rest, about 0.44 during a question (FP8: 0.50); about 0.1–0.3 bit of key precision. On CPU (8 seeds, each design with its own vote) end-to-end KL is 1.27× R6's design on Llama-1B and 1.12× on Qwen3-1.7B, in geometric mean; the vote's question pass with dithered values is neutral |
| **Proposition KL, Corollary WF, run-time τ** (§14) | second order; per-layer independent dithers | sensitivities measured once per layer | predicts every tail configuration's KL within a median 1.00–1.34× (Llama-1B; Qwen3-1.7B with keys ≥ 5 bits); key errors cost 2× value errors; the KL-budget τ makes the certified read deployable, cutting KL 3.7–14× for +0.0004–0.003 traffic. With the single store: 5.6–25× on Llama-1B; on Qwen3-1.7B only with 8-bit keys in its noisy layers. The level needs per-family calibration; the sensitivities can be pooled |
| **Theorem DD**: free-running decode | every input | per-layer independent dithers | Theorem R holds at every decode step up to the first token that differs from the exact model's (§11) |

**The claim that survives:** with a randomized (dithered) tier 1, the tail design's attention error
at every head and step is within a bound computed at run time, with probability ≥ 1 − δ, for any model
with softmax attention over cached rows and any input. This now includes free-running decode up to the
first token that differs from the exact model's. A model changes only what the guarantee costs:
the tier-1 precision it needs and how many rows must be read exactly.

**Why the design works (Theorem M):** the tier's noise enters the output through the unread rows' share
of Σ w_i²‖v_i − o‖², not through their share of the mass. Reading the few heavy rows exactly removes
almost all of it:
- on Llama-1B L1 the vote leaves 2.4% of the mass unread but 0.18% of that second moment, and the tail's MSE is
  1.9% of a dense 4-bit tier's (predicted 1.86%, measured 1.87%);
- where attention is flat the gain shrinks with it: on Granite-4.2's L1 (attention multiplier 1/128) the vote leaves
  80% of the second moment unread, and the tail's MSE is 79% of a dense tier's.
The same formula therefore says when the design will work, on any model, from one FP pass.

**Not covered:**
- end-to-end quality (R6 measures it);
- decode steps after the first differing token, except at a log-size cost (§11);
- FP8-level certificates at 4-bit keys: the information is there (the oracle certifies 62% of Llama head-steps),
  but no proof in this family gets within the needed 2–3× of the oracle (§12).

## 1. Setting

One softmax attention head at one step:
- query q, rows i with keys k_i and values v_i, scores s_i = scaling·⟨q, k_i⟩ + β_i, where β_i is any
  exactly known bias (sinks, ALiBi, masks), and weights w_i = e^{s_i};
- **fixed rows** (protected window, question, generated tokens) are always exact: total weight W_F,
  mean value o_F;
- **tier 1** stores every context row approximately (k̂_i, v̂_i, scores ŝ_i, weights ŵ_i = e^{ŝ_i});
  **tier 2** holds the exact rows;
- a design reads a set S of rows exactly (from tier 2) and either evicts the rest, T, or reads T from
  tier 1 (the **tail design**);
- exact output o = (W_F o_F + Σ w_i v_i)/Z, with Z = W_F + Σ_i w_i; missed mass ε = Σ_T w_i / Z.

Per-row tier-1 errors: δ_i = s_i − ŝ_i (score) and e_i = v_i − v̂_i (value), with ν_i = ‖e_i‖.

## 2. Evicting, and why no advance selection can be certified

**Lemma 1.** Evicting T gives o − o_S = ε(ō_T − ō_S), so ‖o − o_S‖ ≤ 2Vε, where V = max‖v_i‖.
*Proof:* o = (1 − ε)ō_S + εō_T. ∎

A set chosen before the answer's queries exist can't bound ε for queries it hasn't seen. Eviction
works exactly when (i) a small set holds almost all of the answer-time attention and (ii) the question
already points at it. R1–R4 match this:
- single needles: near FP;
- many values: the vote fails while the oracle is fine;
- aggregation: even the oracle fails;
- re-ranking: re-selection wins.

## 3. A computable bound on the missed mass, and the controller

**Lemma 2.** If |δ_i| ≤ b_i (Cauchy–Schwarz gives b_i = scaling·‖q‖·η_i with η_i = ‖k_i − k̂_i‖ stored
per row), then

  ε ≤ ε̄ = U/(M + U),  with U = Σ_T e^{ŝ_i + b_i} and M = W_F + Σ_S w_i.

*Proof:* the true unread mass is at most U, and x/(M + x) increases in x. ∎

**Theorem D (deterministic controller).** Start from any S. While the Lemma 1 bound (with ε̄ in place
of ε) exceeds τ, read the unread row with the largest e^{ŝ_i + b_i}. The result is within τ of exact,
or the controller reports its row cap.
*Proof:*
- the bound uses only computed quantities;
- each read row moves its term from U to M, so the bound only falls;
- with every row read, U = 0. ∎

On R5's data the certified budget is 2–3.7× the oracle's at ε = 0.01, but loose at ε = 0.1.

## 4. The tail design: deterministic bounds

**Identity.** Let õ be the tail design's output and ṽ_i the value it reads on S (exact, or v̂_i). Then

  Z(o − õ) = Σ_S w_i(v_i − ṽ_i) + Σ_T w_i e_i + Σ_T (w_i − ŵ_i)(v̂_i − õ).

*Proof:* Zo and Ẑõ are the two weighted sums; subtract Zõ = Ẑõ + (Z − Ẑ)õ. ∎ (Checked to 2e-15.)

**Lemma 4.** If |δ_i| ≤ b_i and ν_i ≤ ν̄_i, then

  ‖o − õ‖ ≤ [Σ_S w_iν_i + Σ_T ŵ_i(e^{b_i}ν̄_i + (e^{b_i} − 1)‖v̂_i − õ‖)] / (W_F + Σ_S w_i + Σ_T ŵ_i e^{−b_i}),

with the S term present only when S's values come from tier 1.
*Proof:* the triangle inequality on the identity, with w_i ∈ [ŵ_i e^{−b_i}, ŵ_i e^{b_i}]. ∎

Lemma 3, the earlier version with max b and max V, is weaker: Lemma 4 is 0.05× it at the median.

**Why deterministic bounds cannot certify.** On real attention at 128K, b_i is about 8 nats (R5:
median of the per-head maximum 8.1 on Llama, 8.0 on Qwen), so e^{b} ≈ 3,000. The deterministic bound
is valid but useless. Any useful bound must use the fact that tier-1 errors cancel across rows.

## 5. Lemma 5: concentration under an error model

**Model M.** Given what is stored and the rows read:
- rows' errors are independent;
- E[e_i | δ_i] = 0;
- δ_i = β_i + ξ_i, where |β_i| ≤ μ_i is a bias allowance and ξ_i is symmetric and sub-Gaussian with
  proxy σ_i² (σ_i = b_i/√d for a random-direction key error).

**Lemma 5.** Under M, with probability ≥ 1 − δ, ‖o − õ‖ ≤ (m + r)/D. Here:
- m is a mean term in the moment bounds M1 = e^{μ}·min(e^{σ²/2}, cosh zσ);
- r is Pinelis' Bernstein radius;
- D is the denominator from truncating every ξ_i at zσ_i, with z = √(2 ln(4|T|/δ)).

The proof: truncation, then the identity read as a sum of independent vectors, then Pinelis (1994,
Thm 3.4). A coupled variant covers MLA, where one latent error drives both the key and the value
error. Full statement and proof: `cert_s1h5.tail_bound_conc` and plan.md, part 3.

**Why it isn't enough.** M is a statement about the data. Real tier-1 scores regress to the mean:
- on Llama-3.2-1B, rows in the top decile of tier-1 score are 0.08–0.11 nats lower in truth and the
  bottom decile 0.016–0.019 higher (σ ≈ 0.4);
- the bias is coherent across rows;
- with μ = 0, Lemma 5 failed on 12–18 of 16,384 head-steps in one row group of a CPU smoke;
- a 0.5σ allowance fixed it, but only as a calibration, not a proof.

## 6. The randomized tier and Theorem R (the main result)

**Randomized tier 1** (`rquant_s1h5.dither_quantize`). Each row x (key or value, per KV head; or one
block of an MLA latent) is stored as follows:
1. y = Rx, with R a fixed rotation that only spreads outliers;
2. step D = 2·max_j|y_j|/(2^bits − 1), stored per row (the same side information as today's norm);
3. dither u_j ~ U(−D/2, D/2), regenerated from (seed, layer, head, row, channel) and never stored;
4. code c_j = ⌊(y_j + u_j)/D⌋ (no clipping, by the choice of D);
5. reconstruction ŷ_j = (c_j + ½)D − u_j.

By Schuchman's theorem (subtractive dither, no overload), ŷ_j − y_j is uniform on ±D/2 and
independent of y, *for every y*. Storage is unchanged, and the MSE is 1.03–1.15× the store's 4-bit
Lloyd-Max. For any query q and row i, over the coins:
- δ_i is symmetric and sub-Gaussian with proxy σ_i² = scaling²‖q‖²D_i²/12 (sinh(x)/x ≤ e^{x²/6}),
  and |δ_i| ≤ scaling·(D_i/2)‖Rq‖₁;
- E e^{−kδ_i} = ∏_j sinh(k a_ij)/(k a_ij), with a_ij = scaling·|(Rq)_j|D_i/2. Its log lies in
  [k²A₂/6 − k⁴A₄/180, k²A₂/6], with A₂ = a²‖Rq‖₂² and A₄ = a⁴‖Rq‖₄⁴: two statistics per query and one
  step per row;
- the value error is zero-mean, independent of δ_i (separate dither), with E‖e_i‖² = d D_v²/12 and
  ‖e_i‖ ≤ √d D_v/2;
- rows are independent.

**The unbiased tail.** E[ŵ_i] = w_i·E e^{−δ_i} > w_i: every tier-1 tail over-weights itself by about
σ²/2. Dividing each unread row's tier-1 weight by E e^{−δ_i}(q) makes the tail's sums unbiased. In
measurements this design is more accurate than the Lloyd-Max tail (§7).

**Theorem R.** Assume:
- (R1) the rows read exactly (the vote and any extra rows) are chosen without looking at tier 1's
  coins, e.g. from exact prefill attention, from exact-key page metadata, or from a separate sketch;
- (R2) the query does not depend on the coins of this layer's tier 1. In any prefill or teacher-forced pass this
  holds by Lemma C (§11), given the coins of the layers below. In free-running decode it holds up to the first
  token that differs from the exact model's (Theorem DD, §11).

Then for **every cache content** (no assumption on keys, values or queries), with probability ≥ 1 − δ
over the coins, the tail design (exact keys and values on S, tier-1 weights w̃_i = ŵ_i/debias_i on T,
plain or unbiased) satisfies

  ‖o − õ‖ ≤ [m_Y + r(B_Y, a_Y) + ‖õ − c‖·(m_D + r(B_D, a_D))] / D_Z,

with every term computed at run time (`cert_s1h5.tail_bound_rand`).

*Proof.*
1. **Identity.** Let c be the exact output of the read and fixed rows; by (R1) it is coin-free. Let
   M be their mass. Then Z(o − õ) = Σ_T Y_i − (õ − c)·Σ_T D_i, where Y_i = w_i(v_i − c) −
   w̃_i(v̂_i − c) and D_i = w_i − w̃_i. Each depends only on row i's coins, so they are independent
   across rows.
2. **Truncation.** Every |δ_i| ≤ t_i = min(zσ_i, a.s. bound), z = √(2 ln(4|T|/δ)), fails with
   probability ≤ δ/2. The truncated variables stay independent.
3. **Moments.** With Z_i = w̃_i/w_i = e^{−δ_i}/debias_i, the exact moment interval, corrected for
   truncation, gives:
   - E Z_i ∈ [lo, hi], E Z_i² ≤ hi₂, E Z_i⁴ ≤ hi₄, and |Z_i − E Z_i| ≤ rng;
   - from these: ‖E Y_i‖ ≤ w_iκ_iρ_i, |E D_i| ≤ w_iκ_i (κ = max(hi − 1, 1 − lo): zero up to
     truncation for the unbiased tail), variances w_i²[(hi₂ − lo²)ρ_i² + hi₂E‖e_i‖²] and
     w_i²(hi₂ − lo²), and ranges w_i[rng·ρ_i + e^{t}ν̄_i] and w_i·rng, where ρ_i = ‖v_i − c‖.
4. **Self-bounding.** The w_i and ρ_i are unknown. On the truncation event, ρ_i ≤ ‖v̂_i − c‖ + ν̄_i,
   and w_i ≤ w̃_i·debias_i·e^{t_i} (used only in maxima). Each unknown sum Σ w_i g_i or Σ w_i² g_i is
   bounded from its observed tier-1 version. The tool is Maurer's lower-tail inequality for sums of
   independent non-negative variables, P(E S − S ≥ x) ≤ exp(−x²/(2ΣE X_i²)), solved as a quadratic.
5. **Concentration.** Pinelis' Bernstein inequality in Hilbert space applies to each sum: with
   L = ln(2/δ'), r(B, a) = La/3 + √((La/3)² + 2LB²). The radius only grows in B and a, so substituting
   upper bounds keeps the probability statement.
6. **Denominator.** Z ≥ D_Z = M + max(0, Σ_T w̃_i − |Σ_T D_i|), and |Σ D| is already bounded by
   step 5's events.

Six events at δ/12 each and the truncation at δ/2 complete the proof. ∎

**Tightening without new assumptions (v2, `theory2_s1h5.tail_bound_rand_v2`).**
- The self-bounds weight the coins' moments row by row: Σ a_i²m_i ≤ max_i(a_i m_i)·Σ a_i.
- The variance sum is the smaller of its fourth-moment self-bound and max_i(w_i√g_i)·Σ_i w_i√g_i,
  which needs only second moments (one extra event; budget δ/14).
- Both sums use the exact Bennett form of Pinelis' Theorem 3.4,
  P ≤ 2exp(−(B²/a²)h(ra/B²)) with h(u) = (1 + u)ln(1 + u) − u, inverted numerically. Bernstein is its
  relaxation; Bennett's radius is several times smaller when a single row's range dominates the
  standard deviation, which is the heavy-row regime of a coarse tier.

**Tightening further (v3, `theory3_s1h5.tail_bound_rand_v3`).** No new assumptions; three changes.
1. *Regroup by noise source.* The identity in step 1 is
   Z(o − õ) = K_c − (õ − c)K₂ + V, with K_c = Σ_T D_i(v_i − c) and K₂ = Σ_T D_i (key noise only) and
   V = Σ_T w̃_i e_i (value noise). Given the key coins, the value rotation turns V into d independent coordinates
   Σ_i w̃_i ε_ij. Each is sub-Gaussian with proxy s² = Σ_T w̃_i² D_{v,i}²/12, so E e^{λ‖V‖²} ≤ (1 − 2λs²)^{−d/2}, and
   Laurent and Massart's χ² tail gives ‖V‖ ≤ s·√(d + 2√(dL) + 2L) with probability ≥ 1 − e^{−L}. Every input is
   observed, so the values need no self-bound. The norm's dimension is used, which a dimension-free bound cannot do.
2. *One-row maxima.* **Lemma (one row).** If a_i ≥ 0 are fixed and P(a_i > ā_i) ≤ δ₁ for each i, then
   P(max_i a_i > max_i ā_i) ≤ δ₁. *Proof:* the argmax of the fixed a_i is a fixed row i*, and
   {max a > max ā} ⊆ {a_{i*} > ā_{i*}}. ∎ Every maximum in the proof (Bennett's ranges, the self-bounds' maxima) is
   a max over rows of an unknown fixed quantity w_i·c_i. So inside it, one row's deviation t₁ = σ√(2 ln(1/δ₁))
   (≈ 3.6σ) replaces the union level t = zσ (z ≈ 4.8 for 1,500 rows at δ = 0.05). This is used when the largest
   unread σ is ≥ 0.15, a switch that does not see the coins.
3. *Variance proxy.* E‖v̂_i − c‖² = ρ_i² + E‖e_i‖². So Σ w̃_i²·var_i·‖v̂_i − c‖² self-bounds
   Σ w_i²·var_i·(ρ_i² + E‖e_i‖²), which is at least the true variance, without the cross term of (‖v̂_i − c‖ + ν̄_i)².
   Maurer's second moment needs E‖v̂_i − c‖⁴. For the symmetric, isotropic dither error it is
   ρ⁴ + 2ρ²E‖e‖² + 4ρ²E‖e‖²/d + E‖e‖⁴ ≤ (ρ² + E‖e‖²)(ρ² + ν̄²) when d ≥ 2, so the per-row factor is ρ̄² + ν̄².

Events: the truncation takes δ/2; eight events share the other δ/2: V, and for K_c and K₂ the mean and variance
self-bounds and the Bennett radii, with v2's second-moment alternative for K_c. When change 2 is on, eight
one-row events share it too.

*Brute force* (`test_theory3_s1h5.py`): 19,200 head cases (random, spiky, 30× outlier and adversarially aligned
data; 2-, 4- and 6-bit keys; plain and unbiased tails). There is no failure at any δ from 0.3 to 10⁻³, and v3 is
never more than 5% above v2. Real layers: §7.

**Part 7 (merged 2026-10-10; developed in parallel with part 6 in `14h_methodology_improve/certread_s1h5.py` (deleted after the merge), now
`theory4_s1h5.py`).**

**§6c. Sure sums on the truncation event.** On the truncation event that every version already pays for,
w_i = w̃_i·e^{δ_i + ldebias_i} ≤ w̄_i := w̃_i·e^{t_i + ldebias_i} for every unread row. So every unknown weighted sum
Σ w_i^p g_i (p = 1, 2, g_i an almost-sure per-row factor) is bounded *surely* by Σ w̄_i^p g_i, with no new event:
Bennett's radius is increasing in B, so any variance bound that holds on the truncation event may be used.
- The self-bound pays ln(1/δ) × the heaviest row.
- The sure sum pays e^{p t_i} on every row.
- Taking the smaller is valid.

On Llama-1B L1 at 8-bit keys the self-bound inflated the value-and-key variance sum 7–17×, against e^{2t} ≈ 1.3 for
the sure sum. The sure sums need only per-row factors that hold almost surely: κρ̄, var·ρ̄², √var·ρ̄, κ, var, with
ρ̄ = ‖v̂ − c‖ + ν̄ ≥ ‖v − c‖.

Implemented two ways:
- `theory4_s1h5.tail_bound_rand_sure`: v2 plus sure sums;
- `theory3_s1h5.tail_bound_rand_v3(..., sure=True)`: part 6's v3 plus sure sums, never above v3.

**§6d. The mirror split by coin source** (`theory4_s1h5.tail_bound_rand_split`; GQA). Part 6 writes
Y_i = D_i(v_i − c) − w̃_i e_i, i.e. observed weights on the value noise and unknown vectors on the key noise (D_i =
w_i − w̃_i, e_i = v̂_i − v_i). The same identity can be grouped the other way:

  Y_i = D_i(v̂_i − c) − w_i e_i,  so  Z(o − õ) = K − V − (õ − c)·Σ_T D_i,  K = Σ_T D_i(v̂_i − c),  V = Σ_T w_i e_i.

- **V** has fixed weights (w_i = e^{s_i} is a property of the data) and involves the value coins only. It is a
  sub-Gaussian vector with proxy v = Σ w_i²D_{v,i}²/12, the same argument as part 6's. Hsu–Kakade–Zhang (2012, Thm 1,
  A = I) gives ‖V‖² ≤ v(d + 2√(dx) + 2x) with probability ≥ 1 − e^{−x}; that is Laurent–Massart's χ² form, as in part
  6. Here v ≤ Σ w̄_i² E‖e_i‖²/d surely, which costs e^{2t} against part 6's observed w̃.
- **K, given the value coins** (independent of the key coins), is a sum of independent vectors D_i·u_i with
  u_i = v̂_i − c *observed*. Its mean is ≤ Σ w̄ κ‖u_i‖. Its fluctuation takes Bennett with B² = Σ w̄² var ‖u_i‖² and
  range max w̄·rng·‖u_i‖, using sure sums throughout. Part 6's K_c has unknown vectors v_i − c, so it must widen by
  ν̄ (ρ̄ = ‖v̂ − c‖ + ν̄) or self-bound ‖v̂ − c‖².
- **Σ_T D_i** as in Theorem R, with sure sums.

Events: the truncation at δ/2; V, K and Σ D at δ/6 each.

**Which grouping is tighter depends on σ.**
- At 6–8-bit keys (e^{2t} ≈ 1.1–1.3) the split's observed vectors win: Llama-1B 8-bit 5.6–5.7× the error, against
  v3 + sure sums 6.5–6.6× and v3 9.8–10.4×; Gemma 3 L5† 2.8× against 3.2× and 3.7×.
- At 4 bits (e^{2t} ≫ 1) part 6's observed weights win: v3 + sure sums 484–505×, split 516–547×.

Both need independent key and value dithers, so MLA keeps the coupled form.

*A negative result.* A dimension-aware bound for the whole sum Z(o − õ) was also tried: Bousquet (2002), driven by the
weak variance sup_u Σ Var⟨u, X_i⟩, which divides the value part by d. It gains nothing on real or synthetic data, because
its 2·range·E‖S‖ term costs what the weak variance saves. Only the split (or part 6's regrouping) separates the
isotropic value noise cleanly.

**§6e. Corollary: the certified adaptive read** (`theory4_s1h5.certified_read`). Fix a nested schedule
S_0 ⊂ S_1 ⊂ … ⊂ S_m = all rows, each set chosen without the coins. For example: the vote's rows, then rows in the order
of a separate Lloyd-Max sketch's upper-bound share, up to context fractions f_j. Let B_j be any Theorem R form for S_j
at confidence δ/m (S_m is exact and needs no bound). Read sets in order and stop at

  J = min{ j : B_j ≤ τ_j for every query head of the KV head },

where τ_j may depend on anything observed (e.g. ε·‖õ(S_j)‖). Return õ(S_J). Then **at each query head h**, with
probability ≥ 1 − δ, ‖o_h − õ_h(S_J)‖ ≤ τ_J:
- each B_j bounds one query head's output at confidence δ/m, so by the union bound over the m sets, every B_j of head h
  holds at once with probability ≥ 1 − δ;
- on that event the chosen one holds, whatever the stopping rule, because J meets the target at every head.

For all r query heads of a KV head at once the same argument gives only ≥ 1 − rδ. A joint statement at 1 − δ needs
δ/(m·r) in each bound. (Corrected 2026-10-22: this said "at every query head" with probability ≥ 1 − δ.
R7a, `../14i_r7_theory_at_scale/theory_r7.md` §2.6, found the gap and tested the per-head form at scale.)

The coins may decide *how many* sets are read, but not *which* sets. The price is δ/m inside each bound (a ln m term).
This certifies a controller that reads more only where the bound is loose. §9's open threshold rule ("read every row
with w̃_i > θ") is the harder case where the sets themselves depend on the coins.

**Part 8 (2026-10-10; `theory5_s1h5.py`, tests `test_theory5_s1h5.py`, findings `findings/findings_part8.md`).** Two
changes to the split (§6d), each valid on its own. Together they take 6-bit keys from about 18× the error to about 7×
(§7, part 8 table). Notation: Z_i = w̃_i/w_i = e^{−δ_i − ldb_i}; t_i the truncation level; Z̃_i = e^{−clip(δ_i, ±t_i) − ldb_i}
(= Z_i on the truncation event); σ_i² = Var δ_i.

**§6f. The tilted tail.** Every version so far bounds E[Z_i; |δ_i| > t_i] by Cauchy–Schwarz, √(E Z_i²·p_i) ≈ √p_i.
At δ = 10⁻³ and |T| = 1,400 that is 6·10⁻⁴ per row. It enters every row's bias κ_i and adds 1.2·10⁻³ to Bennett's
variance, which is more than σ² itself at 8-bit keys (σ² ≈ 6·10⁻⁴). A Chernoff bound on the tilted tail is of order
p_i instead. For λ ≥ 0, 1{x > t} ≤ e^{λ(x − t)}; with x = −δ (proxy σ²) and λ = t/σ² − 1:

  E[e^{−δ}; −δ > t] ≤ e^{t − t²/(2σ²)},  E[e^{−δ}; δ > t] ≤ e^{−t}·e^{−t²/(2σ²)}.

Call their sum, times e^{−ldb}, τ_i. Then:
- E[Z | |δ| ≤ t] ≥ E Z − τ;
- Var(Z | |δ| ≤ t) ≤ E(Z − 1)²/P(|δ| ≤ t);
- E Z̃ ∈ [E Z − τ, E Z + τ].

That alone takes the split from 5.8× to 4.7× the error at 8-bit keys (Llama-1B L1).

**§6g. The key sum by Hsu–Kakade–Zhang.** Given the value coins, K − E K̃ = −Σ_i w_i(Z̃_i − E Z̃_i)·u_i, with u_i = v̂_i − c
fixed. This is a sum of independent scalars along fixed vectors, so HKZ (2012, Thm 1) applies once each Z̃_i − E Z̃_i
has a sub-Gaussian proxy s_i²:

  ‖K − E K̃‖² ≤ tr M + 2√(‖M‖_F²·x) + 2‖M‖_op·x  with probability ≥ 1 − e^{−x},  M = Σ_i w_i²s_i²·u_i u_iᵀ.

The radius increases with every w_i in the Loewner order, so on the truncation event the sure weights w̄_i (§6c) may
replace w_i. The true M is fixed given the value coins, so the HKZ event is about a fixed matrix. The weight sum
Σ_T D_i is the scalar case: mean + √(2 ln(2/δ')·Σ w̄²s²). Against Bennett, HKZ has no range term (Bennett's range is
w̄(e^t − 1)‖u‖, which dominates when one row carries the sum) and uses the key term's dimension. That is the
"Hanson–Wright-type" step §9 asked for. Two proxies:

*(P-LSI, closed form.)* s_i² = e^{−2ldb_i}·σ_i²·(1 + c_L(e^{t_i} − 1))², with c_L = √12/π. Proof:
- write e^{−y} = 1 − y + r(y) with y = clip(δ);
- clip(δ) is symmetric with |clip δ| ≤ |δ|, so E e^{λ clip δ} = E cosh(λ clip δ) ≤ E e^{λδ} ≤ e^{λ²σ²/2};
- r(clip δ) is (e^t − 1)·scaling‖q‖-Lipschitz in the rotated key error, which is uniform on the cube of side D;
- the uniform law on [0, 1] satisfies Ent(f²) ≤ (2/π²)E f′². This is the circle's sharp log-Sobolev inequality
  (Weissler 1980; constant = spectral gap) carried over by even extension. It tensorizes to Ent ≤ 2(D²/π²)E|∇f|² on
  the cube, and Herbst's argument then gives proxy (12/π²)σ²(e^t − 1)² for r;
- square roots of proxies add (Hölder).

It is tight as t → 0 (8-bit keys: 1.36σ²). It is loose when t is large: 3.3σ² at 6 bits, 129σ² at 4 bits.

*(P-exact, numerically rigorous; `exact_key_proxy`.)* The law of δ_i is exact:
- δ_i = Σ_j a_ij U_j, with U_j uniform on [−1, 1] and a_ij = scaling·|(Rq)_j|·D_i/2;
- L(κ) = log E e^{κδ} = Σ_j log(sinh(κa_j)/(κa_j)) is known in closed form.

λ < 0: one-sided Herbst. Since |∇Z̃| ≤ Z̃·scaling‖q‖, the log-Sobolev inequality gives
Ent(e^{θZ̃}) ≤ (θ²/2)(12/π²)σ² E[Z̃²e^{θZ̃}]. For θ < 0, Z̃² and e^{θZ̃} move in opposite directions (Chebyshev), so
this is ≤ (θ²/2)(12/π²)σ² E Z̃²·E e^{θZ̃}. Integrating Herbst's (H/θ)′ over [λ, 0] gives proxy (12/π²)σ²·E Z̃² ≈ 1.22σ²
for every λ < 0, with no clip penalty.

λ > 0: E e^{λZ̃} ≤ Σ_k (λ^k/k!)·E[Z^k; δ ≥ −t] + (e^{λe^{t−ldb}} + e^{λe^{−t−ldb}})·P(δ < −t). Each piece comes from L:
- let κ* solve L′(κ*) = t (any κ ≥ 0 gives a valid bound);
- for k ≤ κ*, E[e^{−kδ}; δ ≥ −t] ≤ e^{L(k)};
- for k > κ*, E[e^{−kδ}; δ ≥ −t] ≤ e^{L(κ*) + (k − κ*)t}·min(1, 1/((k − κ*)√L″(κ*)));
- P(δ < −t) ≤ e^{L(κ*) − κ*t}·min(1, 1/(κ*√L″(κ*))).

The factors 1/(c√L″) hold because a tilted uniform sum is log-concave, so its density is at most 1/(its standard
deviation) (Bobkov–Chistyakov). The series' tail uses Z ≤ e^{t − ldb} on δ ≥ −t. Then ψ(λ) = log E e^{λZ̃} − λE Z̃ is
convex, and the supremum of 2ψ/λ² comes in three ranges:
- on a grid of λσ in [10⁻², 20], below chords between grid points;
- below the grid, ψ ≤ (λ²/2)·Var Z̃·e^{λ₁(Zmax − E Z̃)};
- above it, ψ ≤ λ(Zmax − E Z̃).

Every step is an inequality, so the result is a valid proxy for every row; floating point is the only approximation.
Against the optimum sup_λ 2ψ/λ² computed by convolution (64 channels, z = 5.6):
- σ = 0.1 (6-bit): 1.68σ² against 1.57σ² (P-LSI 3.32σ²);
- σ = 0.42 (4-bit): 14.0σ² against 13.3σ² (P-LSI 129σ²);
- σ = 0.025 (8-bit): 1.22σ², where the λ < 0 side binds, against 1.08σ² (P-LSI 1.36σ²).

The optimum sits at λσ ≈ 4: the clipped edge, a rare event that a few hundred Monte Carlo draws never sample. The
1.25σ² estimated earlier from 300 draws was optimistic, just as §12's earlier Monte Carlo estimate was. P-exact is
computed for each head-step's 16 largest contributors to tr M; every other row keeps P-LSI. Any per-row choice is
valid, since both proxies bound the same fixed quantity.

**The bound** (`tail_bound_rand_split_hkz`): Z‖o − õ‖ ≤ K-term + V-term + |Σ D|-term·‖õ − c‖, divided by
M + (Σ w̃ − |Σ D|-term)₊. The pieces:
- the K-term: Σ w̄κ̃‖u‖ + the HKZ radius;
- the V-term: as in §6d;
- the |Σ D|-term: Σ w̄κ̃ + the scalar sub-Gaussian radius;
- κ̃ = the clipped law's bias, max(E Z + τ − 1, 1 − E Z + τ).

Events: the truncation at δ/2, then V, K and Σ D at δ/6 each. With Bennett as a second form for K and Σ D, each sum
takes the smaller form, at δ/10 per event. That pays only at 4-bit keys on GPT-OSS, where P-exact is not computed
(its series would need > 6,000 terms).

**The colleague's remedies, with P-exact** (Llama-1B L1, 6-bit keys, one draw):
- *Truncation budget to the vote's sketch-heavy rows* (remedy 1; `prep(heavy=, share=)`): the heavy rows get `share`
  of δ/2 and the rest 1 − share. With 16 rows at share 0.5, z falls from 5.6 to 4.7 on those rows, and the bound from
  6.9× to 6.5× the error. z grows like √(ln 1/p), so moving the budget buys little.
- *The heaviest rows in a deterministic term* (remedy 2; `det_heavy=`): |w_i − w̃_i| ≤ w̃_i·max(e^{ldb+t} − 1,
  1 − e^{ldb−t}) on the truncation event, taken out of the HKZ sum. This makes the bound worse: 8.4× with one row,
  11.5× with four. It adds e^t − 1 per row linearly, while HKZ adds them in quadrature with near-exact proxies.

The diagnosis behind both (the unknown weight of a few heavy rows costs most) is right. The gain came instead from the
proxy (P-exact) and the range-free HKZ form.

**What is left at 6 bits** (Llama-1B L1, from non-rigorous substitutions): the sure weights w̄ ≥ w cost about 1.75×
(e^t with t ≈ 0.56), on K and on V. P-exact is within 1.07× of the optimal proxy. A self-bound on M (lower tail of
Σ w̃²…) does not help: the median head-step's heaviest unread row carries 21% of M's trace, and a self-bound wins only
below about 1%.

**Proof details used by every version.**
- *The Hilbert-space Bennett inequality.* Pinelis (1994), Thm 3.4 with D = 1, follows from
  E cosh(λ‖x + d‖) ≤ cosh(λ‖x‖)(1 + E φ(λ‖d‖)) for zero-mean d, where φ(u) = e^u − 1 − u. This holds because
  the Hessian of cosh(λ‖·‖) has operator norm ≤ λ²·cosh(λ‖y‖), and cosh(λ‖x + td‖) ≤ cosh(λ‖x‖)·e^{λt‖d‖}.
  Induction over the rows and Markov's inequality then give P(‖Σd_i‖ ≥ r) ≤ 2 exp(−λr + Σ E φ(λ‖d_i‖)).
  With ‖d_i‖ ≤ a, E φ(λ‖d_i‖) ≤ (E‖d_i‖²/a²)·φ(λa).
- *The radius may take upper bounds.* Bennett's radius r(B, a) is increasing in B and in a. Both partial
  derivatives of the exponent (B²/a²)·h(ra/B²) are ≤ 0, because ln(1 + u) ≤ u and (2 + u)·ln(1 + u) ≥ 2u.
- *The seed.* The probability is over the dither for a fixed cache. The seed must therefore be drawn
  independently of the input (per request in a deployment): an input chosen with the seed in view can defeat
  any randomized guarantee.

**MLA (coupled), in head space.** One latent draw makes both errors of a row. The concentration runs on
each head's output vectors W_h·(latent):
- the value error W_h·e has E‖W_h e‖² = ‖W_h‖_F²·D²/12 exactly (e = Rᵀε with ε uniform and
  independent, so E[e eᵀ] = (D²/12)·I), and ‖W_h e‖ ≤ ‖W_h‖₂·√d·D/2;
- the mean term gains ‖E[Z·W_h e]‖ ≤ √E(Z − 1)²·√E‖W_h e‖² (joint symmetry of (e, δ));
- the variance term becomes (√var·ρ + √hi₂·ν̄)²;
- ρ_i = ‖W_h(v_i − c)‖ is bracketed by the observed ‖W_h(v̂_i − c)‖ ± ‖W_h‖₂ν̄_i. Self-bounds that
  allow this coin dependence (`coupled_self_bound1/2`) charge only a correction Σ w ν̄ (or Σ w²ν̄²).

This replaces a latent-space bound multiplied by ‖W_h‖₂ with a coin-free ρ, which overstated the error
by the ratio of W's top singular value to its average (§7).

**Deterministic companion.** Lemma 4 with the dither's a.s. bounds (b_i = scaling·(D_i/2)‖Rq‖₁,
ν_i ≤ √d D_v/2) holds always and needs no stored error norms.

**A whole decode.** Run each of N head-steps at δ/N. Every constant grows like √ln N (z from about 3.9
to about 6 for N = 10⁵).

## 7. How strong it is in practice (CPU, 2026-10-09)

**Validity.** Theorem R never failed:
- brute force on random, spiky, 30×-outlier and adversarially aligned data, at δ from 0.3 to 1e-3:
  3,200 head cases × 2 designs, plus 1,600 coupled;
- all seven test models at 4-bit keys, 8 draws per layer: coverage 1.000 on every layer;
- the r3b unit where Lemma 5 failed, at 4-, 6- and 8-bit keys: coverage 1.000, including that head.

**Tightness is set by the score noise σ = scaling·‖q‖D/√12.** A rigorous bound must cover a row whose
true weight is e^{zσ} times its estimate, and needs fourth moments e^{8σ²}. With 4-bit keys, real
queries give σ ≈ 0.5–1.4 nats; each extra key bit halves σ.

The r3b unit (Llama-3.2-1B, 4K, the probe on FP's queries, read sets from the Lloyd-Max store; Theorem R v1). Shares
certified within FP8's error are at the vote's rows / with C/32 more rows read exactly / with C/8 more:

| tier-1 key bits | bound / error | certified within FP8 | tail's relative error | traffic vs FP16 (vote / +C/32 / +C/8) |
|---|---|---|---|---|
| 4 | 1,500–3,200 | 0 / 0 / 0 | 0.0068 (Lloyd-Max tail 0.0121) | 0.34 / 0.37 / 0.44 |
| 6 | 28–50 | 0.08 / 0.24 / 0.52 | 0.0024 | 0.40 / 0.42 / 0.48 |
| 8 | 11.5–18 | 0.29 / 0.61 / 0.85 | 0.0018 | 0.45 / 0.47 / 0.53 |

(Values stay at 4 bits; FP8's traffic is 0.50.)

Across architectures (`findings/R5_adapters.md`, early layers, 1,600 tokens):

| model | bound / error at 4 → 6 → 8 key bits | what limits it |
|---|---|---|
| Llama-1B (L1) | 2,826 → 59 → 21 | key noise |
| GPT-OSS | 4.3 × 10⁶ → 275 → 36 | large ‖q‖‖k‖: key noise |
| Gemma 3† (global) | 32 → 11 → 9 | 4-bit values (QK-norm keeps key noise small) |
| Moonlight (MLA) | 929 → 188 → 236 | the coupled bound's coin-free distance |

**MLA in head space** (Moonlight, DeepSeek-V2-Lite, Kimi-Linear; coverage 1 at every width). Median
bound/error at 8 bits: 246 → 74, 166 → 55 and 260 → 80 (latent space with a coin-free ρ → head space).

**How far from the best possible certificate?** For a fixed cache and query, no certificate at 1 − δ
can be narrower than the (1 − δ)-quantile of the error over the coins: the *oracle width*, computable
only from the exact data. 200 draws per head-step, δ = 0.05, unbiased tail, Theorem R v2 (coverage 1
everywhere). Columns: v2 bound / oracle width (median), and the share of head-steps within FP8's error
for v2 / for the oracle:

| model (layer) | 4-bit keys | 6-bit | 8-bit |
|---|---|---|---|
| Llama-1B (L1) | 206 · 0 / 0.62 | 20 · 0.09 / 0.95 | 14 · 0.28 / 0.98 |
| GPT-OSS (L1) | 42,500 · 0 / 0.08 | 60 · 0.01 / 0.39 | 19 · 0.03 / 0.87 |
| Gemma 3† (L5) | 14 · 0 / 0 | 9 · 0 / 0 | 8 · 0 / 0 |

Three readings:
- **Llama at 4 bits:** the tier holds enough information to certify 62% of head-steps at FP8's
  level, so the proof is the bottleneck. Its range term covers a single tail row whose true weight is
  e^{t} times its estimate (t = zσ ≈ 2.4 nats).
- **Gemma 3†:** on that random model's flat layer the 4-bit values were the limit, and no certificate of any kind
  could reach FP8's level without more value bits. The real L5 at 1,600 tokens leaves 7.6% of Σ w²‖v − o‖² unread,
  and its tail sits at 0.91× FP8's error with 4-bit keys and 0.37× with 6-bit keys
  (`findings/g0/evaluate_gemma3_L5_1600_real.json`). A real flat layer, Granite-4.2's L1 (85% of the mass and 80% of
  the second moment unread), shows the reading: its tail is 1.77× FP8's error at every key width, because the
  4-bit values are the limit (`findings/g0/evaluate_granite42_L1_1600.json`).
- **6–8 bits:** the remaining factor (8–20×) is the constants of a high-confidence, worst-case-over-rows
  argument.

**Theorem R v3 on the same setup** (`exp4bit_s1h5.py evaluate`, 2026-10-10). Bound / oracle width, median,
v2 → v3. In brackets: the share of head-steps certified within FP8's error, v2 → v3. Coverage is 1 everywhere.

| view (context rows) | 4-bit keys | 6-bit | 8-bit |
|---|---|---|---|
| Llama-1B L0 (1,552) | 80 → 52 (0 → 0) | 13.8 → 11.5 (0.002 → 0.004) | 10.8 → 8.5 (0.006 → 0.014) |
| Llama-1B L1 (1,552) | 201 → 114.5 (0 → 0) | 20.2 → 17.3 (0.09 → 0.10) | 14.3 → 11.1 (0.28 → 0.37) |
| Llama-1B L3 (1,552) | 217 → 124 (0 → 0) | 19.1 → 16.2 (0.12 → 0.16) | 13.3 → 10.5 (0.37 → 0.47) |
| Llama-1B L1 (8,144) | 337 → 159 (0 → 0) | 18.5 → 14.9 (0.11 → 0.16) | 11.2 → 7.7 (0.31 → 0.46) |
| GPT-OSS L1, sinks (1,552) | 26,057 → 7,485 (0 → 0) | 51.7 → 39.3 (0.014 → 0.014) | 18.3 → 14.6 (0.035 → 0.050) |
| Gemma 3† L5, global (1,552) | 14.2 → 11.6 (0 → 0) | 9.2 → 6.9 (0 → 0) | 8.5 → 6.3 (0 → 0) |

v3 gains 1.5–3.5× at 4-bit keys where the one-row maxima switch on (σ ≥ 0.15; Gemma 3†, with σ ≈ 0.12, gains 1.2×)
and 1.2–1.45× at 6–8 bits. The gain grows with the context (8,144 rows). It does not make 4-bit keys certifiable at
FP8's level; §12 explains why.

**The tail's error against FP8's** (Theorem M's measured RMS error / FP8's error, median; dense = a dithered tier
with no exact rows):

| view | tail, 4 / 6 / 8-bit keys | dense, 4-bit keys |
|---|---|---|
| Llama-1B L0 | 1.39 / 0.58 / 0.47 | 3.4 |
| Llama-1B L1 | 0.52 / 0.16 / 0.11 | 3.4 |
| Llama-1B L3 | 0.38 / 0.12 / 0.09 | 4.3 |
| Llama-1B L1, 8K | 0.57 / 0.18 / 0.13 | 3.8 |
| GPT-OSS L1 | 4.1 / 0.91 / 0.43 | 9.2 |
| Gemma 3† L5 | 3.4 / 2.7 / 2.7 | 4.2 |

**Adaptive queries (R2), measured** (`adaptive_s1h5.py`). Setup: full Llama-3.2-1B, 2,048 tokens, 4
draws, and the design's output substituted at every answer position and layer, so every later query
depends on the coins. Theorem R covered all 49,152 head-steps at 4 and 6 bits. The worst error/bound
(0.051, 0.138) was identical to the oblivious pass on FP's queries.

**The unbiased tail is more accurate than the Lloyd-Max tail in most layers.** Relative error:

| case | unbiased dithered | Lloyd-Max |
|---|---|---|
| r3b unit, vote's rows | 0.0068 | 0.0121 |
| Kimi (MLA) | 0.010 | 0.032 |
| Llama L0 | 0.018 | 0.022 |
| Moonlight L0 | 0.008 | 0.010 |

On GPT-OSS it halves the plain dithered tail's error.

**What the guarantee costs.** A read certified at FP8's level costs about FP8's traffic. The 4-bit
design is at or below FP8's KL in most families (R5.3) but cannot be certified.

**Part 7 on real weights** (`findings/findings_part7.md`; δ = 10⁻³ per head-step, the vote's rows read; median bound / error ·
share certified within FP8):

| view, key bits | v2 | v3 (part 6) | v3 + sure sums | split |
|---|---|---|---|---|
| Llama-1B L1, 6 | 41.7 · 0.04 | 33.0 · 0.06 | 21.6 · 0.10 | 18.3 · 0.13 |
| Llama-1B L1, 8 | 19.3 · 0.24 | 10.4 · 0.43 | 6.6 · 0.61 | 5.7 · 0.66 |
| Llama-1B L3, 8 | 18.2 · 0.32 | 9.8 · 0.56 | 6.5 · 0.69 | 5.6 · 0.75 |
| Gemma 3† L5, 8 | 9.0 · 0 | 3.7 · 0 | 3.2 · 0 | 2.8 · 0 |

At 4 bits: Llama-1B L1 1,054 / 637 / 484 / 516, nothing certified.

The certified adaptive read at 8-bit keys (split form) reads 14–18% of rows exactly at traffic 0.46–0.49 (FP8: 0.50) on:
- Llama-1B L1–L3, with FP8's per-head error as the target (an oracle target);
- Gemma 3† L5, with 1% of the output's norm as the target.

An oracle stop would read 12.5–13%. Gemma 3† at FP8's error and Granite-4.0-H at 1% cannot be met by any read set:
their 4-bit values are the limit.

**Part 8 on real weights** (`findings/findings_part8.md`; same setup, 3 draws; median bound / error · share certified
within FP8; "part 8" = the tilted tail, HKZ with P-exact on each head-step's 16 largest contributors, and remedy 1):

| view | key bits | split (part 7) | part 8 |
|---|---|---|---|
| Llama-1B L1 (1.6K / 8K) | 6 | 18.0 · 0.13 / 19.3 · 0.15 | 6.75 · 0.47 / 7.34 · 0.41 |
| Llama-1B L3 | 6 | 17.4 · 0.21 | 6.72 · 0.59 |
| Llama-1B L0 | 6 | 10.3 · 0 | 4.60 · 0.08 |
| Gemma 3† L5 (global) | 6 | 4.04 · 0 | 2.05 · 0 |
| GPT-OSS L1 | 6 | 77.1 · 0.02 | 32.4 · 0.05 |
| Llama-1B L1 (1.6K / 8K) | 8 | 5.77 · 0.66 / 5.30 · 0.70 | 2.57 · 0.89 / 2.59 · 0.90 |
| Llama-1B L3 | 8 | 5.64 · 0.75 | 2.51 · 0.95 |
| Llama-1B L1 (1.6K / 8K) | 4 | 522 / 875 | 287 / 554 |

Coverage is 1.000 everywhere. At 6 bits, the range-free HKZ form gives about 1.9×, P-exact about 1.3× more, and
remedy 1 about 5%. GPT-OSS at 4 bits stays ≥ 10⁵× the error with every form (σ ≈ 0.9).

The certified adaptive read with part 8's bound (FP8's per-head error as the target; δ = 10⁻³; rows read exactly
(traffic)):
- **6-bit keys:** 0.244 (0.480) on L1 and 0.271 (0.499) on L3, with a finer 16-set coin-free schedule, and 0.175 (0.433)
  on L1 at 8K. That is at or below FP8's 0.50. Part 7 needed 0.432 (0.609), 0.449 (0.621) and 0.279 (0.505). The oracle
  stop is 0.125 (0.399).
- **8-bit keys:** 0.129–0.135 (0.456–0.460), against the oracle stop's 0.125 (0.453).
- **L0** stays far above: 0.605 (0.728) at 6 bits and 0.241 (0.526) at 8.

**Qwen** (`findings/findings_qwen.md`; Qwen3-1.7B and R6's Qwen3-30B-A3B-Instruct-2507, same setup). Every theorem holds:
- the architecture checks pass, and coverage is 1.000 everywhere;
- Theorem M is within 1% at 6–8-bit keys.

In typical layers part 8 transfers unchanged: 4.0–6.5× the error at 6-bit keys and 1.9–2.2× at 8. The certified read
meets FP8's per-head error at traffic 0.403–0.423 at 6-bit keys, against the oracle stop's 0.398–0.399. What differs is
the score noise: Qwen3's QK-norm gives some layers 2–7× Llama's σ at the same key width.
- **Where the tail exceeds FP8's error.** Theorem M's per-layer map (`layermap_s1h5.py`, every layer, no sampling) puts
  the 6-bit tail's median per-head error above FP8's in no Llama-3.1-8B layer, in Qwen3-1.7B's L0–L2, and in R6's
  Qwen L1 only (1.25×).
- **Where no certificate helps.** In Qwen's flat early layers, FP8's per-head error is out of reach at FP8's traffic
  even for the oracle stop (0.51–0.70 at 6 bits).

**Every available architecture** (`findings/findings_architectures.md`, 2026-10-11). That is 11 new models: Llama-3.3-70B,
Mistral-7B, Mistral-Small-3.2, Granite-4.2, Qwen1.5 and Qwen1.5-MoE (MHA with bias), GLM-4.5-Air, MiniMax-M2 (FP8),
Qwen3.5/3.8 (gated attention in recurrent hybrids), and Gemma 4 (global head dim 512). MLA's Theorem M was checked
on three more models.
- **Every theorem holds:**
  - the architecture checks pass on 38 layers;
  - part 8 and the certified read cover every draw on 22 views;
  - Theorem M is within 1% on every GQA view at 4, 6 and 8-bit keys, and within 2% on five MLA layers.
- **Part 8 is tight across the board:** 1.3–7.8× the error at 6-bit keys (median ≈ 3.5) and 1.3–2.8× at 8. It is
  2.0–3.3× tighter than the split everywhere.
- **The certified read sits at the oracle** in concentrated layers on six architectures: traffic 0.400–0.441 at 6
  bits, against 0.398.
- **What varies is the design's own error.** In flat-attention models the vote leaves 12–64% of Σ w²‖v − o‖² unread,
  and 4-bit values hold the tail above FP8's per-head error at any key width. These are Gemma 4, Granite-4.2
  (attention multiplier 1/128), MiniMax-M2's first layers, and Qwen3.5/3.8's global layers.
- **Theorem M's map names the fix:** 6-bit values. They bring the tail's median below FP8's error in every mapped
  layer, at traffic 0.453, the same as 8/4. (Two Granite-4.2 exceptions first reported here were a rounding artifact,
  §10b.) At the stricter p90 target 6/6 is not universal (§10b, `MODEL_FAMILY.md`).

**At the paper's scale** (R7a, GPU, 2026-10-22; `../14i_r7_theory_at_scale/findings/R7a.md`).
- **Setup.** R6's models and lengths:
  - Llama-3.1-8B at 128K (75 units);
  - Qwen3-30B-A3B at 32K (50 units) and at 128K (30 units);
  - RULER and HELMET tasks, every layer and head;
  - 3.78 M head-steps in all, at 4/6/8-bit keys, on FP's pass, with δ = 10⁻³.
- **Coverage.** Part 8 and v2 missed in none of 37.8 M checks each.
- **Part 8's tightness.** At 6-bit keys, part 8 is 5.6× the error on Llama and 7.2× / 8.0× on Qwen at 32K / 128K
  (median). It certifies FP8-level per-head error on 0.92 / 0.72 / 0.68 of head-steps, and is 0.21–0.22 of v2's bound.
  At 8 bits it is 2.0–2.4× the error and certifies 0.99 / 0.93 / 0.91.
  - v2 is 25.5× the error at 128K on Llama-3.1-8B, against 27× at 1.6K and 24.6× at 16K on the CPU: flat in context.
  - The weakest layers are Llama's L30, L8 and L1, and Qwen's flat L3, L1, L0, L2 and L16.
- **The certified read** (FP8's per-head error as the oracle target, the 16-set fine schedule) met its target at all
  774,656 query-head checks at 6 bits. It reads:
  - 0.126 of rows on Llama (traffic 0.399, the oracle's 0.399);
  - 0.235 on Qwen 32K (0.474);
  - 0.161 on Qwen 128K (0.423).

  Qwen's extra rows are its flat early layers.

## 8. The general approach this implies

For any model with softmax attention over cached rows:
1. **Find the rows.** Use the global softmax layers. Rows are per KV head (GQA), per token latent
   (MLA, coupled), and sinks enter as exact fixed rows. Local windows and SSM/linear layers are left
   alone (`adapters_s1h5.py`).
2. **Store tier 1 randomized, sized by Theorem M.** Use dithered keys and values with per-row steps. On an
   FP pass, Theorem M splits the tail's MSE into a key term Σ_T w²·Var Z·‖v − o‖² (Var Z ≈ σ²; each key bit halves σ)
   and a value term Σ_T w²·E‖e‖² (each value bit divides it by 4). Choose bits per model so that the predicted
   error sits below FP8's:
   - Llama-1B: 4-bit keys and values already put the tail at 0.4–0.6× FP8's error;
   - GPT-OSS: σ ≈ 0.9 at 4-bit keys, so it needs 6–8-bit keys;
   - Granite-4.2 (L1, attention multiplier 1/128): its attention is flat, so the vote leaves 80% of the second moment
     unread. Its 4-bit values then hold the tail at 1.77× FP8's error at any key width, and it needs more value bits
     or a larger exact set. (Gemma 3's L5 stood here; that was a random-weight model.)
   - Qwen3: the score noise varies by layer, and a few layers need more key bits. `layermap_s1h5.py` computes the
     map for every layer from one FP pass. On R6's Qwen3-30B-A3B, 6-bit keys keep the tail at or below FP8's
     per-head error in 47 of 48 layers; 8-bit keys in the first four layers would cost about +0.005 traffic.
   - Flat attention (Gemma 4, Granite-4.2, MiniMax-M2's first layers, Qwen3.5/3.8's global layers): the value
     term is 85–98% of the tail's error. Use 6-bit values, not 8-bit keys; 6/6 costs the same traffic as 8/4
     (0.453) (`findings/findings_architectures.md` §3).
3. **Choose the exact rows without the coins.** Exact prefill attention (the question's vote computed on exact keys
   before compression), exact-key page metadata, or a separate sketch qualify. A vote over the dithered tier
   itself does not.
4. **Read the tail unbiased:** divide tier-1 weights by E e^{−δ}, computable from two per-query
   statistics.
5. **Certify per call** with Theorem R v3 where it is affordable: 6–8-bit tier-1 keys, plus a few extra exact rows.
   R5.4, with v1, certified 98–99% of Llama-8B head-steps at FP8's level at 128K with 6-bit keys and C/32
   extra rows. Anchor τ to FP8's per-head error on that model. At 4-bit keys the certificate is valid but
   loose (§12); use Theorem M to predict the error and R6 to measure it.
6. **Draw the dither seed per request** (independent of the input). Certificates then hold through
   free-running decode until the first token that differs from the exact model's (Theorem DD).
7. **Measure end-to-end quality** with a confirmatory run (R6). No attention-level bound can stand in
   for it: a size-τ change in one head can flip a near-tied token in any format, FP8 included.

A model changes the cost (σ, concentration, value precision), never the validity.

## 9. Open problems

- **Adaptive decode (R2).** Closed up to the first token that differs from the exact model's (Theorem DD, §11).
  Beyond it, the cost is k·(ln N + ln|V|) in each event's log-probability per allowed divergence.
- **Tightness at 4 bits** (§12). Diagnosed:
  - of v2's 201×, about 10× is the unknown weight of the heaviest unread rows (e^{zσ} inside Bennett's
    range and the self-bounds' maxima), about 3× the self-bounded sums, and about 10× constants;
  - the two remedies proposed for it (truncation budget by the vote's sketch; the heaviest rows in a
    deterministic term) and v3's one-row maxima each recover 1.4–2×, and 2× together (201 → 98–114×).
  The information is there (the oracle certifies 62% of head-steps at FP8's level). What is missing is a proof
  that pays one row's deviation instead of a union over rows for the truncation, too (§12).
- **Constants at 6–8 bits.** Part 8 (§6f–§6g) is the Hanson–Wright-type step this item asked for: the key sum by
  Hsu–Kakade–Zhang with a near-exact proxy. It brings 6-bit keys from about 18× the error to 6.7–7.3× on Llama-1B, and
  8-bit keys from 5.6–5.8× to 2.0–2.6×. What is left at 6 bits is mainly the sure weights w̄ ≥ w, worth e^t ≈ 1.75
  (t ≈ 0.56) on both the key and value sums. Self-bounds do not help, because one unread row carries 21% of the sum at the
  median head-step. Paying one row's deviation instead of the union's (the two families below) would also address it.
- **Reads chosen from the dithered tier.** Consider the threshold rule "also read every row with w̃_i > θ", with
  θ coin-free. (A controller whose read *sets* are coin-free and whose *stopping* depends on the coins is already
  covered: the certified adaptive read, §6e.) Each unread summand still depends only on its own row's coins, so Theorem R's proof goes
  through with Bennett's ranges capped by θ·e^{t}. It needs one more term, the selection bias
  Σ_i w_i·|E[(1 − Z_i)·1{w_i Z_i > θ}]|, which on the truncation event is bounded by the rows observed within
  e^{±2t} of θ. This would certify a controller whose extra reads look at the dithered tier. Not implemented.
- **MLA in Theorem M:** done (§10, `mla_m_s1h5.py`). The coupling term is exact and ≤ 0.1% on Moonlight,
  DeepSeek-V2-Lite and Kimi-Linear.
- **End-to-end:** R6.

## 10. Theorem M: the second moment (why the design works)

**The identity.** Let Z̃ = M + Σ_T w̃_i, the design's own normaliser: observed, and ≥ M. Then exactly

  Z̃(o − õ) = N := Σ_T w_i(1 − Z_i)(v_i − o) + Σ_T w_i Z_i e_i,

with Z_i = w̃_i/w_i. *Proof:* Z̃(o − õ) = M(o − c) + Σ_T w̃_i(o − v̂_i), and Σ_T w_i(v_i − o) = M(o − c) because
Zo = Mc + Σ_T w_i v_i. Substitute v̂_i = v_i − e_i. ∎ (Checked to 10⁻¹³ on random and adversarial data.)

**Theorem M.** Fix any cache, query and coin-free read set. Take a dithered tier 1 whose key and value draws are
independent (GQA). Then exactly

  E‖N‖² = Σ_T w_i²·[Var(Z_i)·‖v_i − o‖² + E(Z_i²)·E‖e_i‖²] + ‖Σ_T w_i(1 − E Z_i)(v_i − o)‖²,

and hence
- E‖o − õ‖² ≤ E‖N‖²/M², always;
- E‖o − õ‖² = E‖N‖²/Z² to first order in the fluctuation of Σ_T w̃_i.

The coin moments are exact: E Z^k = ∏_j sinh(k·a_j)/(k·a_j)·e^{−k·ldb}, with a_j = scaling·|(Rq)_j|·D_i/2, and
E‖e_i‖² = d_v·D_{v,i}²/12. For the unbiased tail E Z_i = 1 up to the debias interval, so the last term vanishes.
*Proof:* N's summands are independent across rows. E[Z_i e_i] = E Z_i·E e_i = 0, because the key and value draws are
independent. Expand the square. ∎ (`theory3_s1h5.second_moment`; the Monte Carlo check is in `test_theory3_s1h5.py`.)

**Evidence** (`exp4bit_s1h5.py evaluate`; oracle_s1h5's setup, 200 draws, medians over head-steps). The columns:
- the unread mass and the unread share of Σ w_i²‖v_i − o‖² (the vote's 1/8 read exactly);
- the tail's MSE over a dense dithered tier's at 4-bit keys, predicted / measured;
- measured MSE / Theorem M's first-order prediction at 4 / 6 / 8-bit keys;
- the eviction error (the unread rows dropped) over FP8's.

| view | unread mass | unread share of Σw²‖v−o‖² | tail / dense MSE (4-bit) | measured / predicted MSE | eviction / FP8 |
|---|---|---|---|---|---|
| Llama-1B L0 | 0.240 | 0.198 | 0.220 / 0.222 | 0.996 / 1.000 / 1.003 | 18.4 |
| Llama-1B L1 | 0.024 | 0.0018 | 0.0186 / 0.0187 | 0.991 / 0.994 / 0.999 | 3.35 |
| Llama-1B L3 | 0.029 | 0.0009 | 0.0095 / 0.0090 | 0.981 / 1.003 / 0.999 | 2.53 |
| Llama-1B L1, 8K | 0.049 | 0.0015 | 0.0223 / 0.0226 | 0.990 / 1.002 / 1.002 | 6.03 |
| GPT-OSS L1 | 0.121 | 0.199 | 0.227 / 0.229 | 0.922 / 0.995 / 1.001 | 8.58 |
| Gemma 3 L5 (real weights; corrected 2026-10-12) | 0.203 | 0.076 | 0.077 / 0.074 | 0.994 / 0.999 / 1.002 | 6.9 |
| Granite-4.2 L1 (attention multiplier 1/128) | 0.853 | 0.798 | 0.787 / 0.790 | 1.003 / 1.003 / 1.003 | 408 |
| Qwen3-1.7B L0 | 0.144 | 0.471 | 0.449 / 0.484 | 0.199 / 0.990 / 0.998 | 22.1 |
| Qwen3-1.7B L1 | 0.188 | 0.238 | 0.238 / 0.246 | 0.914 / 0.998 / 0.997 | 22.1 |
| Qwen3-1.7B L13 | 0.052 | 0.0031 | 0.0027 / 0.0030 | 0.991 / 0.999 / 0.998 | 4.16 |
| Qwen3-1.7B L26 | 0.047 | 0.0036 | 0.0015 / 0.0014 | 0.999 / 1.002 / 1.001 | 2.65 |
| Qwen3-30B-A3B (R6) L1 | 0.221 | 0.331 | 0.397 / 0.387 | 0.985 / 1.001 / 1.003 | 32.8 |
| Qwen3-30B-A3B (R6) L3 | 0.066 | 0.0051 | 0.518 / 0.486 | 0.311 / 0.998 / 1.003 | 2.18 |
| Qwen3-30B-A3B (R6) L24 | 0.021 | 0.0003 | 0.0006 / 0.0006 | 0.987 / 0.996 / 0.999 | 1.21 |

(More Qwen views: `findings/findings_qwen.md`.)

The first-order form is within 1% at 6–8-bit keys on every view, Llama, Gemma 3, Granite-4.2, GPT-OSS and Qwen3 alike. At 4-bit
keys it is off by 8–9% where σ ≈ 1 (GPT-OSS; Qwen3-1.7B L1). Where σ ≈ 2 (Qwen3-1.7B L0, R6's Qwen L3) it overpredicts
the error 3–5×: the random normaliser Z̃ then moves too far for Z to stand in for it. The M² bound and the tail-to-dense
ratio still hold there (0.449 predicted, 0.484 measured).

**At the paper's scale** (R7a, GPU, 2026-10-22; `../14i_r7_theory_at_scale/findings/R7a.md`). The test is the identity
Z̃(o − õ) = N: the mean of (Z̃/Z)²‖o − õ‖² estimates E‖N‖²/Z² = m_pred without bias. On 3.78 M head-steps
(Llama-3.1-8B at 128K, Qwen3-30B-A3B at 32K and 128K), measured / predicted is 1.000 at 6 and 8-bit keys in every
group, with 95% intervals within ±0.001; the first-order reading is 1.000 there too.
- At 4 bits Llama is 1.000.
- On Qwen at 4 bits the ratio is 0.994 (32K) and 0.990 (128K). The ratio is heavy-tailed there (maximum ≈ 2,000 against a
  mean of 1), and 2 draws per head-step fall short of its expectation.
- **The unread share orders the families.** Across Llama's seven task families it ranks them as R6's measured KL does
  (Spearman 0.857, permutation p = 0.012), and it does so on Qwen too.

**What it explains.**
1. *Why reading the heavy rows exactly works.* The tier's noise enters through Σ_T w_i², the unread rows' share
   of the attention's second moment, not through their mass. On Llama-1B L1:
   - the vote leaves 2.4% of the mass unread, but 0.18% of Σ w_i²‖v_i − o‖²;
   - the formula predicts the tail's MSE at 1.86% of a dense dithered 4-bit tier's; measured, 1.87%.

   Where attention is flat, the share is large and the gain small. On Granite-4.2's L1 the vote leaves 85% of the
   mass and 80% of the second moment unread; the tail's MSE is 79% of a dense tier's, and evicting instead costs 408×
   FP8's error. (Gemma 3's L5 stood here; that was a random-weight model. With its real weights, the share is 7.6%,
   the tail 7.6% of a dense tier, and eviction 6.9× FP8's error.) The share is the predictor of whether the two tiers help, and it comes from one FP pass.
2. *Why the tail beats eviction.* Eviction's error is a first-order bias ε_T(ō_T − ō_S) (Lemma 1). The tail's is a
   standard deviation, about σ·ε_T·(spread)/√n_eff, which averages over the unread rows: on Llama-1B L1, n_eff ≈ 124.
   Measured, eviction is 3.35× FP8's error and the 4-bit tail 0.52×.
3. *Which bits a model needs.* The key term scales with Var Z ≈ σ² and the value term with E‖e‖²; the formula splits
   them per model (§8, step 2; the RMS / FP8 table in §7).
4. *A design calculator.* From one FP pass, the formula predicts the tail's MSE for any read set and any key and
   value bits, with no dither draws.

**MLA** (`mla_m_s1h5.py`, 2026-10-10). One latent draw makes both errors of a row. The tier dithers the row
[c_i, k_rope,i] block by block (rotation R_b, step D_b). So Z_i = e^{−ldb_i}·∏_j e^{β_j ε_j}, with β = scaling·(R q):
a product over independent uniform coordinates. Head h's value error e_i = A_h ε_0 (A_h = W_UV,h R_0ᵀ) shares ε_0
with Z_i. With e = v̂ − v, Y_i = w_i[(1 − Z_i)u_i − Z_i e_i] and u_i = W_h(c_i − o_lat). Each coordinate's tilted
moments are closed forms (L(x) = coth x − 1/x, a_j = D_0/2):
- E[Z ε_j] = E Z·a_j L(a_jβ_j);
- E[Z² ε_j] = E Z²·m2_j, with m2_j = a_j L(2a_jβ_j);
- E[Z² ε_j ε_k] = E Z²(m2_j m2_k + 1{j = k}·v2_j), with v2_j = a_j²(1 − 2coth(x)/x + 2/x²) − m2_j² and x = 2a_jβ_j.

Hence:
- E Y_i = w_i[(1 − E Z)u_i − E Z·A m1];
- E‖Y_i‖² = w_i²[(1 − 2E Z + E Z²)‖u_i‖² − 2⟨u_i, A(E Z·m1 − E Z²·m2)⟩ + E Z²(Σ_j v2_j‖A_{:j}‖² + ‖A m2‖²)];
- E‖N‖² = Σ_i(E‖Y_i‖² − ‖E Y_i‖²) + ‖Σ_i E Y_i‖², exactly.

Validated by 200 dither draws in head space (`findings/findings_mla_m.json`). Measured over predicted MSE at 4/6/8-bit
keys:

| layer | 4 / 6 / 8 bits |
|---|---|
| Moonlight L1 | 0.99 / 1.00 / 0.98 |
| Moonlight L13 | 0.99 / 1.00 / 1.00 |
| DeepSeek-V2-Lite L1 | 0.98 / 1.01 / 0.98 |
| DeepSeek-V2-Lite L13 | 0.99 / 1.00 / 1.00 |
| Kimi-Linear's MLA layer | 1.00 / 1.01 / 1.00 |

The coupling term is ≤ 0.1% of the key and value terms on every layer: the GQA formula is accurate for MLA in
practice. The term is first order in the alignment of u with W_h's image of the query, which is small in d_v = 512.

**Limit.** Theorem M is a second moment over the coins for a fixed cache. It explains and predicts the design's
average accuracy; it is not a high-probability certificate (Theorem R is).

**§10b. Corollary M-B: choosing the bits** (`layermap_s1h5.py --grid`, 2026-10-11). Theorem M gives the tail's
error for any key width k, value width v and exact read set S, from one FP pass:

  E‖o − õ‖² ≈ [K_S(k) + V_S(k, v) + bias_S(k)]/Z²,  K_S(k) = Σ_{i∉S} w_i²·Var Z_i(k)·‖v_i − o‖²,
  V_S(k, v) = Σ_{i∉S} w_i²·E Z_i²(k)·E‖e_i‖²(v).

Every term is exact: the step D(b) = 2‖R x‖_∞/(2^b − 1) fixes both moments. So the error *surface* over (k, v, S) is
known per layer and head without sampling.

Measured over predicted on seven views at 5/5, 4/6, 6/6, 8/6, 4/8 and 6/8 (200 draws): 0.98–1.005 wherever k ≥ 5.
The views are Llama-1B L1, Gemma 3 L5†, GPT-OSS L1, Granite-4.2 L1, Qwen3-1.7B L1, MiniMax-M2 L1 and R6's Qwen L1. At
4-bit keys on high-σ layers (GPT-OSS L1, Qwen3-1.7B L1) the median is 0.92: the formula overpredicts, which is
conservative. Near-zero scores (Granite-4.2 L4/L6, |s| ≈ 10⁻⁹) need the moments' series form: the closed form of
log(sinh x/x) loses every digit at x ≈ 10⁻¹² (fixed 2026-10-11 in `theory3_s1h5._log_sinhc`). With it, measured
over predicted is 0.99–1.00 there too.

*The closed form.* For small σ, Var Z ≈ σ² and E Z² ≈ 1. Then
  MSE(k, v) ≈ [a·c(k) + b·c(v)]/Z², with c(x) = (2^x − 1)^{−2},
  a = Σ_T w_i²·(scaling²‖q‖²/12)(2‖R k_i‖_∞)²·‖v_i − o‖² (key sensitivity),
  b = Σ_T w_i²·(d_v/12)(2‖R_v v_i‖_∞)² (value sensitivity).
Meeting a target ε with the fewest bits gives:
- each tier's bits: k* = log₂(1 + √(2a/(εZ²))) and v* = log₂(1 + √(2b/(εZ²)));
- the split: k* − v* ≈ ½·log₂(a/b).

So each tier needs one more bit per 4× of its own sensitivity, and the split depends only on the layer's key-to-value
sensitivity ratio. At equal widths that ratio is the key term's share of the error. The read set enters through a and
b: reading row i exactly removes w_i²(…) from both, so the heaviest rows are the most valuable reads.

*What it manages.* 4/4, 4/6, 6/4, 6/6 or any per-layer or per-head mix is one point on the surface. Three mechanisms
build on it:
- an offline allocator (per layer or head, from a few calibration prompts): the cheapest (k, v, |S|) meeting a target;
- the run-time certificate (Theorem R, part 8), which checks each head-step on the actual input;
- a run-time escalation: the certified read's nested coin-free schedule (§6e), extended from "read more rows exactly" to
  "fetch more precision". A residual plane of a subtractive-dithered code, re-dithered with step D/(2^r − 1), leaves an
  error uniform on that finer cell (Schuchman again, no overload). So base + residual is itself a dithered tier with an
  exact law, and Theorem R and Theorem M hold for any per-row mix of planes chosen coin-free.

Measured allocation on 18 models (`findings/findings_bits.md`; target: 90% of head-steps at or below FP8's per-head
error). The families of layers and models it implies are in `MODEL_FAMILY.md`.
- **Per-layer allocation:**
  - 0.35–0.39 traffic on models whose layers are mostly concentrated (Llama 1B/8B/70B, Mistral, Qwen1.5, Qwen3, GLM).
    That is mostly 4/4 or 5/4, with 6/4–8/5 in a few layers, and below uniform 6/4's 0.398, which misses 0–8 layers;
  - 0.43–0.47 on flat ones (Gemma 4, Qwen3.5/3.8, Granite-4.2, MiniMax-M2), and 0.42 on Gemma 3 (between, key-heavy);
  - 0.46 on GPT-OSS, whose noisy scores need 8/4.
- **Uniform 6/6 is not universal:** it fails 31 of 313 layers, all 12 GPT-OSS full layers among them. 23 have noisy
  scores and need 8-bit keys; 4 have near-constant scores and need 8-bit values; in 4 flat ones both terms are large.
- **The split** k* − v* runs from +2.7 (GPT-OSS) to −2.3 (Granite-4.2, where several layers' key term is ≈ 0).
  Gemma 3 is +1.8 (the −0.6 reported first was the random model's), Gemma 4 +0.3, and R6's models +1.4 to +1.8.
- **The choice depends on the input.** On the code input, PG-19's allocation misses in 0–10 layers per model (3–10 on
  four of six). The per-layer maximum over four inputs costs 0.39–0.48, below FP8's 0.50.

## 11. Free-running decode: Lemma C and Theorem DD

**Lemma C (layer by layer).** Assume:
- each layer's tier-1 dither is drawn independently, and independently of the input;
- the context's exact keys and values are computed before compression;
- each layer's read set depends only on the input, coin-free side information and lower layers' coins.

Then in a pass over any fixed token sequence (prefill or teacher forcing), layer ℓ's queries, fixed rows and read
sets depend only on the dithers of layers below ℓ. So Theorem R (any version) holds for every head-step of layer ℓ
with probability ≥ 1 − δ given those dithers, hence unconditionally, and a union over layers, positions and heads
covers the whole pass.
*Proof:* layer ℓ's query and fixed rows at a position come from the residual stream after layer ℓ − 1. Layer ℓ's
own dither reaches the stream only through layer ℓ's outputs, which feed the layers above ℓ. ∎

**Theorem DD (decode up to divergence).** Let y* be the exact model's decode: greedy, or sampled with the same
sampling randomness the design uses. It does not depend on the dithers. With probability ≥ 1 − δ_total over all
dithers, every head-step of every layer is within its certificate at every decode step t ≤ τ, where τ is the first
step whose token differs from y*_τ.
*Proof:* until τ, the free-running computation receives exactly the tokens y*_{<t} and the same coins as a
teacher-forced pass over the fixed sequence y*. So its head-steps up to τ are head-steps of that pass, which Lemma C
covers with total failure ≤ δ_total. ∎

**Beyond the first divergence.** Run the same argument over every sequence that differs from the exact model's
choice at ≤ k steps. It covers every step before the (k + 1)-th divergence, at a cost of ln|P_k| ≤ k·(ln N + ln|V|)
in each event's log-probability (N decode steps, |V| the vocabulary): about +17 per divergence at N = 128 and
|V| = 128K. That is affordable at 8-bit keys. At 4 bits the k = 0 statement is the useful one.

**What it changes.** R2 is no longer an assumption for decode. The guarantee covers every run that reproduces the
exact model's tokens (the "lossless" regime R6 tests) and, in every other run, all steps up to and including the
first differing token. This agrees with `adaptive_s1h5.py`, where the dependence changed nothing measurable.

## 12. Why 4-bit keys are not certified at FP8's level, and what would do it

Setup: oracle_s1h5's: Llama-3.2-1B L1, 1,552 context rows, the vote's 1/8 exact, 16 queries × 32 heads, δ = 0.05.
Details: `findings/findings_4bit.md`.

**Where v2's 201× comes from.** Counterfactuals that use true quantities (not certificates):

| v2 with … | bound / oracle |
|---|---|
| nothing changed | 201 |
| the true weights inside every max (Bennett's ranges, the self-bounds' maxima) | 19.6 |
| the true self-bounded sums | 71.6 |
| both | 11.3 |
| both, and the true distances ‖v_i − c‖ | 9.8 |

The tail is not one dominant row. At the median head-step the heaviest unread row carries 3.9% of the unread mass,
and the unread rows are worth about 124 equal rows. What costs 10× is that every row's weight is known only up to
e^{zσ} ≈ 7.5 (z ≈ 4.8, σ ≈ 0.42). This enters Bennett's range twice (the unknown weight × the range of Z) and the
self-bounds' maxima squared.

**The proposed remedies, made rigorous** (`exp4bit_s1h5.py remedies`; bound / oracle, coverage 1):

| certificate | Llama-1B L1 | Llama-1B L3 |
|---|---|---|
| v2 | 201 | 217 |
| truncation budget to the vote's sketch-heavy rows (share 0.5 / 0.9) | 143 / 133 | 159 / 149 |
| heaviest rows (true weight > κM) in a deterministic term, κ = 0.01 / 0.001 / a grid of 7 | 153 / 124 / 121 | 188 / 154 / 154 |
| both (grid + budget 0.9) | 98 | 126 |
| v3 (one-row maxima, regrouped identity, χ² values) | 114.5 | 124 |

Each recovers 1.4–2×, and they do not compound beyond about 2× (a side run of v3 with the sketch budget gave 105× on L1).
- The deterministic term pays e^{t} − 1 per heavy row with no cancellation between rows: alone it is 52× the oracle.
- The budget lowers z only from 4.8 to about 3.3 for the favoured rows, because z grows like √(ln 1/p).

**The exact law of Z does not rescue Bennett.** Z = e^{−δ} is lognormal-like, and its moment generating function
exists only because |δ| is bounded almost surely (≈ 11σ for d = 64). Computed rigorously (`exp4bit_s1h5.py lawcheck`, true
weights, the dominant key sum):
- untruncated, the exact-law exponent explodes at the λ the radius needs, so every Chernoff-type argument must
  truncate every unread row;
- truncated, the exact law improves on Bennett only 1.3× (6.9 vs 9.2 × the true 95% quantile), and only with per-row
  scales (99× with one common scale).

An earlier Monte Carlo estimate (2.2×) was optimistic: it never sampled the far tail.

**In a sub-Gaussian form, the exact law does help** (part 8, §6g). Bennett needs a range, and the range of a lognormal-like
Z is what costs. HKZ needs only a sub-Gaussian proxy. The exact law's proxy at 4-bit-like noise is 14.0σ², against
129σ² from the log-Sobolev Lipschitz bound and 13.3σ² for the optimum. With it, the key sum costs 287–359× the error on
Llama-1B L1/L3 (1.6K, δ = 10⁻³), against 484–505× for v3 + sure sums. That is a real gain, but far from the 2–3× of the
oracle needed for FP8-level certificates, so the conclusion below stands.

**The ceiling.** The oracle itself is at 0.74× FP8's error at the median head-step. A certificate at 2× the oracle
would certify 34% of head-steps at FP8's level; at 5×, 7%; at 10×, 2.7%. So FP8-level certification at 4-bit keys
needs a near-oracle certificate. At 6–8-bit keys σ is 4–16× smaller, e^{zσ} ≈ 1, and the same proof is 6–17× the
oracle. That certifies 37–47% of Llama-1B L1/L3 head-steps at FP8's level with 8-bit keys (1.6K–8K rows). Per R5.4 (v1,
Llama-8B, 128K), it certified 70–79% with 8-bit keys and 98–99% with 6-bit keys plus C/32 extra exact rows.

**What would do it.** The information is there: the oracle certifies 62%. A two-point argument says a row that hides
a weight e^{x} above its estimate must be covered only when P(δ ≥ x) > δ, that is, x ≤ σ√(2 ln 1/δ). So the best
certificate pays one row's deviation (z ≈ 2.4), not the union's (z ≈ 4.8). Two candidate proof families:
- *Truncation at the level of the summands* (Fuk–Nagaev style). Truncate w_i|1 − Z_i|·‖v_i − c‖ at one level y.
  Its failure Σ_i P(w_i|1 − Z_i|·ρ_i > y) automatically falls on the rows that can exceed y. With the homogenisation
  P(·) ≤ (s_i/U)²·P(|1 − Z| > y/U), it is bounded by a self-bounded Σ w²ρ² and one-row maxima. This is the rigorous
  form of "give the budget to the heavy rows", with no sketch needed.
- *Confidence-set inversion.* Take the sup of the error over all pivots δ' in a coin-free set of probability ≥ 1 − δ,
  with the true weights reconstructed as w̃·e^{δ'+ldb}. Theorem R is one such set (truncation box ∩ concentration
  events). A set shaped by the error itself would approach the oracle, if its sup can be computed.

Until then, the actionable answers are on the design side: 6–8-bit tier-1 keys, or a few extra exact rows, make the
certificate useful (§8, step 5), and Theorem M explains and predicts the 4-bit design's accuracy.

**At scale, the slack is exponential in the score noise** (2026-10-12, `findings/findings_nested.md` §7; part 8 on
every layer of Llama-3.1-8B and Qwen3-30B-A3B at 32K, uniform 6- and 8-bit keys, 160 cases):

  ln(bound / error) = 0.51 + 13.9·σ_w   (R² = 0.987),

where σ_w is the attention-weighted score noise of the unread rows.
- This matches the diagnosis above: the union's deviation z = √(2 ln(C·m/δ)) ≈ 6.3 (C = 32K rows, m = 16 sets,
  δ = 10⁻³), paid twice, e^{2zσ} = e^{12.7σ}.
- Two consequences:
  - the slack grows only like e^{c·σ·√ln C} with context;
  - one extra key bit, which halves σ, removes most of it wherever σ_w ≳ 0.15.
- Qwen3-30B-A3B has 14 such layers at 32K (bound 20–1,960× the error with 6-bit keys). With 8-bit keys every one is
  within 8.6×.
- A certificate that pays one row's deviation instead of the union's (above) would halve the exponent.

## 13. One store for the vote and the tail: the nested coin-free store (2026-10-11)

**Why.** Theorem R needs the rows read exactly to be chosen without tier 1's coins (R1).
- R6 met R1 with a separate Lloyd-Max 4/4 store for the vote. That store is 0.25 of FP16 K+V, and no reported memory
  number counts it (`CLAIMS_LEDGER.md` S2).
- So R6 supports a traffic claim but not a memory claim.

**The store** (`nested_s1h5.py`). Take a key row x of one KV head and the tier's rotation R:
- γ = ‖x‖ and y = (x/γ)Rᵀ, a unit vector;
- **base:** m_j, the 4-bit Lloyd-Max index of y_j (the N(0, 1/d) codebook, with midpoint thresholds). Its norm-corrected
  reconstruction is exactly `quant.quantize_keys(x, 4, R)`, the vote's sketch;
- **cell:** [lo_j, hi_j], the decision interval of m_j. The outer cells are closed at ±M, where M = max_j |y_j| is stored
  per row;
- **residual:** an r-bit subtractive-dithered uniform quantizer on the cell. It has L = 2^r levels lo_j + iD_j, with
  D_j = (hi_j − lo_j)/(L − 1), and reconstructs ŷ_j = lo_j + D_j·round((y_j − lo_j + u_j)/D_j) − u_j, with
  u_j ~ U(−D_j/2, D_j/2) drawn from the seed;
- **the tail** reads x̂ = γ·ŷR: keys at 4 + r bits, unbiased. Values stay the 4-bit dithered tier.

**Lemma N.** Fix the cache. Given the base indices, the errors ŷ_j − y_j are independent across coordinates and rows.
Each is uniform on [−D_j/2, D_j/2] and independent of the data.

*Proof.*
- y_j + u_j lies in [lo_j − D_j/2, hi_j + D_j/2]. That is the L-level quantizer's range without overload, so
  Q(y_j + u_j) − (y_j + u_j) is uniform on one step and independent of y_j (subtractive dither; Schuchman 1964).
- D_j depends only on the stored base index and M, not on the coins.
- The coins are independent across coordinates and rows. ∎

**Theorem N.** For the nested tier, Theorem R in every form (v1, v2, v3, the split, part 8) and Theorem M hold with the
per-coordinate key steps step_ij = γ_i·D_ij in place of the per-row step.

*Proof.* The key error is e_i = γ_i(ŷ_i − y_i)R. So the score error is δ_i = scaling·Σ_j (Rq)_j ε_ij, with the ε_ij
independent and uniform on ±step_ij/2. Every proof uses the key steps only through three quantities:
- (i) the exact moments E e^{−kδ_i} = ∏_j sinhc(k·a_ij), with a_ij = scaling·|(Rq)_j|·step_ij/2. These feed Theorem M,
  part 8's exact-law proxy, and the coin-moment bounds with A2 = Σ_j a_ij² and A4 = Σ_j a_ij⁴;
- (ii) the sub-Gaussian proxy σ_i² = scaling²·Σ_j (Rq)_j²·step_ij²/12. A uniform is strictly sub-Gaussian, so each
  summand's proxy is its variance;
- (iii) the almost-sure bound |δ_i| ≤ scaling·Σ_j |(Rq)_j|·step_ij/2.

Each holds term by term for unequal steps. Each is computable at read time from the stored indices, γ and M. The value
side is unchanged. ∎

**R1 by construction.**
- The vote, the threshold read rule (§9) and the certified read's sketch order read only the base. The base is a
  deterministic function of the cache, so every read set is coin-free.
- The vote may also run the question through the store, as R6's question pass reads keys and values. It then reads the
  base keys and the tier's dithered values. Layer l's read set depends on the coins of lower layers only, which are
  independent of layer l's coins (the per-layer dithers of Lemma C). So R1 holds layer by layer. On CPU the dithered
  values leave the vote unchanged (evidence below).

**Evidence** (`findings/findings_nested.md`):
- **Lemma N by simulation** (400 draws): the error's moments are a uniform's to 10⁻⁴, it never exceeds D/2, and it is
  uncorrelated with y.
- **Brute force** (`test_nested_s1h5.py`):
  - with one step per row, the per-coordinate code reproduces the per-row code to 6·10⁻¹⁶;
  - E Z, E Z² and Theorem M are within 4.5 s.e. of Monte Carlo;
  - every Theorem R form has no failure at any δ, on adversarial instances with bases of 2–4 bits and residuals of 2–3
    bits.
- **13 real layers** (R6's two models among them):
  - Theorem M's measured / predicted is 0.994–1.004;
  - coverage is 1.000 for every form;
  - the base's vote reads the same attention mass as R6's sketch, within 0.011.
- **End to end** (Llama-3.2-1B, CPU, 8 units): KL per token of nested 4+2 is a median 1.14× uniform 6/4's (0.62–1.84),
  at the same traffic, from 2 seeds. FP8's is 1.5–18× higher (median 4.9×). Nested 4+3 halves uniform 6/4's KL.
- **The vote's question pass over the single store** (2026-10-12, `findings_nested.md` §6; Llama-3.2-1B, 8 units, and
  Qwen3-1.7B, 4 units; 8 seeds each):
  - dithered values in R6's question pass leave the vote unchanged: 98.7–98.9% of its rows are the Lloyd-Max store's,
    it reads the same attention mass within 0.002 per layer, and the KL ratio is 0.99–1.01;
  - the single store with its own vote, against R6's design with its own: 1.27× in geometric mean on Llama-1B (1.15×
    on mean KL) and 1.12× on Qwen3-1.7B (0.91× on mean KL), at the same traffic.
- **Part 8 at 32K on R7b's models** (`findings_nested.md` §7; every layer, coverage 1.000):
  - Llama-3.1-8B: 3.6–7.6× the error in every layer with the single store;
  - Qwen3-30B-A3B: 14 of 48 layers above 20× (up to 1,960× in L3). The 4+4 store brings every layer within 8.6×
    (§12, the slack law).

**Cost and gain.**
- **Precision:** about 0.1–0.3 bit of key precision. Nested 4+2 raises the tail's RMS error by 0–22% over uniform 6-bit
  keys, at the same traffic. Nested 4+3 matches uniform 7-bit keys. Part 8's certificate is 1.05–1.5× looser.
- **Memory:** one store, 0.31 of FP16 K+V at rest, and about 0.44 during a question with the exact 1/8 fetched, against
  FP8's 0.50. The 0.25 vote store disappears.
- **Traffic:** unchanged, 0.398 at f = 1/8.

## 13b. One store for a latent cache (MLA) (2026-10-12, G0c; `g0mla_s1h5.py`)

**Setting.** MLA caches one row per token, the latent x_i = [c_i, k^R_i] (blocks 0 and 1; DeepSeek-V2-Lite: d_c = 512,
d_R = 64). Head h scores scaling·⟨q'_h, x_i⟩ with the absorbed query q'_h = [W_UK,hᵀ q^nope_h, q^R_h], and reads the
value W_UV,h c_i. One store therefore serves both the keys and the values. Each block b is stored in §13's nested
form with its own rotation R_b (the rope block may instead stay exact, or use the uniform tier).

**Lemma N_lat.** Fix the cache. Given every block's base indices, the residual errors ε_bij are independent across
blocks, coordinates and rows. Each is uniform on ±D_bij/2, with D_bij = γ_bi·Dn_bij, and independent of the data.

*Proof.* Lemma N, block by block. Its proof uses nothing about how the row is read. ∎

**Consequences.**
- *Scores.* Head h's score error is δ_hi = Σ_b ⟨β_hb, ε_bi⟩, with β_hb = scaling·R_b q'_hb. Theorem N's three per-row
  quantities therefore hold summed over the blocks: the exact moments multiply, and σ² and the almost-sure bound add.
- *Values.* Head h's value error is A_h ε_0i, with A_h = W_UV,h R_0ᵀ. It shares ε_0 with δ: the coupled case of §10's
  MLA form.
- *The coupled certificate.* (e, δ) is jointly symmetric under ε → −ε. That is the one property
  `cert_s1h5.tail_bound_rand_coupled` (head space, §6 "MLA") uses beyond its per-row inputs, and those inputs have
  per-coordinate forms:
  - E‖W_h e‖² = Σ_j ‖A_{:j}‖²·D_0j²/12;
  - ‖W_h e‖ ≤ min(‖W_h‖₂·(Σ_j D_0j²)^{1/2}/2, Σ_j ‖A_{:j}‖·D_0j/2).
- *R1.* The vote reads scaling·⟨q'_h, base(x_i)⟩, which is coin-free. In the question pass, layer l's dithered latent
  reaches only higher layers, whose votes read other coins (Lemma C's filtration), as for GQA.

**Theorem M_lat.** For the latent store and any coin-free read set,

  E‖N_h‖² = Σ_i (E‖Y_i‖² − ‖E Y_i‖²) + ‖Σ_i E Y_i‖²,

with Y_i = w_i[(1 − Z_i)u_i − Z_i A_h ε_0i] and u_i = W_h(c_i − ō). The moments are §10's MLA moments at per-coordinate
half-widths a_ij = D_0ij/2:
- E Z_i = e^{−ldb_i}·∏_{b,j} sinhc(a_bij β_bj);
- m1_ij = a_ij L(a_ij β_j) and m2_ij = a_ij L(2a_ij β_j);
- v2_ij = a_ij² T(2a_ij β_j) − m2_ij².

*Proof.* §10's MLA proof, coordinate by coordinate: each coordinate's tilted moments are closed forms in its own
half-width. ∎

`g0mla_s1h5.theorem_m_mla` evaluates it by blocks of rows, without [head-step, row, d] tensors. A latent width's tilted
moments are computed once for every rope option and read set.

**Evidence** (`test_g0mla_s1h5.py`):
- with one step per row, it equals `mla_m_s1h5.second_moment_mla` to 4·10⁻¹¹;
- against 200 draws of the nested latent store (synthetic, 4+1 to 4+3, rope same, 8-bit or exact), measured over
  predicted is 0.983–1.003;
- the coupled certificate covers every draw;
- DeepSeek-V2-Lite (real weights, 600 tokens, L1): the latent read through `expand_kv` reproduces the module's keys and
  values exactly; the view reconstructs the layer's output to 8·10⁻⁷; the coupling term is 3·10⁻⁴ of the error.
- At 32K: G0c (`findings/findings_g0.md`).

**Accounting.** Traffic is (1 − f)(d_c·b_c + d_R·b_R)/(16(d_c + d_R)) + f, against the FP16 latent row. There is no
separate value width. A 4+2 latent with a 4+2 rope block costs 0.453 at f = 1/8, as GQA's 6/6 does. GQA's 4+2/4
traffic (0.398) corresponds to a 4+1 latent.

**Open.** Part 8 (§12) has no coupled form. The coupled certificate is v1-type: 60–130× the error on DeepSeek-V2-Lite
at 600 tokens, against part 8's 3–8× for GQA. A coupled split (key part and value part sharing ε_0) is G5's MLA item.

## 14. From per-head error to KL per token: the bridge, water-filling, run-time targets (2026-10-11)

**Why.** Theorems R and M bound and predict each attention call's error, but the paper's metric is KL per answer token.
The bridge connects the two, and it does three jobs:
- it predicts end-to-end KL from Theorem M;
- it gives the bit allocation an end-to-end objective;
- it gives the certified read a target it can compute at run time.

**Proposition KL (second order).**

*Setting:*
- a teacher-forced pass, the unbiased tail, and an independent dither per layer and per KV head;
- z is the logits at an answer position, p = softmax(z), and F = diag(p) − ppᵀ;
- layer l's attention-output errors (all heads and answer positions) split as e_l = e_l^K + e_l^V, the parts due to its
  key and its value dither. Their mean squared sizes are M_l^K and M_l^V, Theorem M's key and value terms.

*Statement.* To second order in the errors,

  E KL(p ‖ p̃) = ½·Σ_l (c_l^K·M_l^K + c_l^V·M_l^V),  with c = E[eᵀJ_lᵀFJ_l e] / E‖e‖² for each part,

where J_l maps layer l's attention outputs to the logits.

The sensitivities depend on each error's shape, not its size. To first order in the coins:
- e_l^K = Σ_T w_i(1 − Z_i)(v_i − o)/Z, with 1 − Z_i linear in the key coins;
- e_l^V = Σ_T w̃_i e_i/Z, with e_i the value coins.

So each part's covariance is a fixed matrix times its size: the width scales the step, and the read set fixes the rows.

*Proof.*
- KL(p‖p̃) = ½ΔzᵀFΔz + O(‖Δz‖³) (the softmax family's Fisher metric), and Δz = Σ_l J_l e_l + O(‖e‖²).
- Different layers' coins are independent, and each e_l has mean zero to first order (the unbiased tail). So the cross
  terms between layers vanish in expectation.
- Within a layer, the key and value coins are independent and each part has mean zero, so their cross term vanishes too.
- The two remaining terms are the quadratic forms above. ∎

*What it buys.* The KL of any configuration can be predicted from one FP pass plus 2L calibration runs. This covers any
key or value width, read set, uniform or nested store, or per-layer mix.
- Theorem M gives M_l^K and M_l^V.
- Each calibration run dithers one part of one layer alone and gives c = 2·KL / realized MSE.

The sensitivities must be measured with the design's own error shapes. They then absorb how the errors correlate across
answer positions (the same coins at every step) and across a KV group's query heads. Isotropic Gaussian noise misses
both.

**Evidence** (`findings/findings_klbridge.md`; Llama-3.2-1B on CPU, `e2e_s1h5.py`). There are 8 units: 4 texts × 2
windows, each with 1,536 context rows, a 32-token question and a 32-token answer.
- **KL is quadratic in the error.** Isotropic noise at 3× the size gives 8.3–11.8× the KL (9 if exactly quadratic),
  in every layer and unit.
- **The key and value parts add.** Keys-only plus values-only gives 0.93–1.49× the full design's KL (median 1.06).
- **Key errors cost about twice as much KL as value errors of the same size.**
  - c_l^V / c_l^K has a median of 0.46 (p10–p90 0.14–0.96) over layers and units.
  - At 6/4 the keys carry 32% of Theorem M's MSE but 57% of the KL.
  - A likely reason, not tested here: a key error moves the weights, so its output error lies along the value
    differences v_i − o that the model uses. The rotated value error is isotropic, so only part of it falls in the
    directions the logits are sensitive to.
  - So the KL-optimal split is about ½·log₂(1/0.46) ≈ 0.56 bit further toward keys than Corollary M-B's MSE-based
    split.
- **Layers nearly add.** At 6/4 the single-layer KLs sum to 0.82–1.64× (median 1.17) the all-layer KL.
- **The structured bridge** (no draws of the configuration it predicts) gives predicted / measured with a median of
  1.00–1.34 for every tail configuration from 4/4 to 8/8, the nested store, keys-only and values-only. Its p10–p90 is
  within about 0.6–2.6, and it errs on the safe side.
- **The isotropic bridge does not transfer across inputs:** its medians are 0.45–1.23 and its p10–p90 0.06–6.
- **FP8's deterministic errors** are 0.2–1.6× as harmful per unit MSE as the tail's (worst on the code input). The
  bridge's sensitivities are for dithered errors.

**Corollary WF (reverse water-filling).**

*Statement.* Let each part's MSE scale as M_l^K(k) = a_l·c(k) and M_l^V(v) = b_l·c(v) (Corollary M-B), with c(x) ≈ 4^(−x).
Minimize traffic, which is affine in Σ_l (k_l + v_l), subject to E KL ≤ B, in the continuous relaxation. Then every
unclipped (layer, part) component contributes the same KL θ:
- k_l* = ½·log₂(c_l^K·a_l/(2θ)) and v_l* = ½·log₂(c_l^V·b_l/(2θ));
- θ is set by the budget.

*Proof.* The Lagrangian derivative in k_l is 1 − λ·ln4·½c_l^K a_l 4^(−k_l). It vanishes when each component's KL is
1/(λ ln 4). ∎

This is rate-distortion's reverse water-filling, here for attention caches. On the grid, the per-layer Lagrangian choice
argmin_(k,v) traffic + λ·KL_l solves it.

*Evidence* (Llama-3.2-1B, calibrated on window 0, tested on window 1):
- **The surface depends on the input,** because the per-layer sensitivities move with the text.
- **The raw KL average does not transfer.** It is dominated by the hardest inputs. In sample it matches uniform 6/4's KL
  at 0.383 traffic, but held out it is 1.6× uniform 6/4's KL.
- **With every input weighted equally** (each unit's surface divided by its own uniform-6/4 KL), it does transfer:
  - held out, 16% lower KL than uniform 6/4 at the same traffic (0.398); 18% was predicted;
  - at 0.371, equal to uniform 5/4 on mean KL and 11% better in geometric mean.
- **The FP8-target allocation** (per-head p90 against FP8's error, `findings/findings_bits.md`) is 1.7–1.8× worse than
  uniform at equal traffic.
- **Conclusion:** per-layer allocation gains are real but modest on this key-heavy model, at most about 20% KL at equal
  traffic. They need calibration on the deployment's input mix, with inputs weighted equally.

**Where the bridge breaks: 4-bit keys on noisy layers.** Qwen3-1.7B (`findings/findings_klbridge.md` §5):
- With keys of ≥ 5 bits the structured bridge holds, at a median ratio of 0.73–0.96. Qwen's layers compound: the
  all-layer KL is about 1.6× the sum of the single-layer KLs (0.55–0.90× the other way round). So the bridge errs low
  here, where on Llama it erred high.
- At 4-bit keys it overpredicts by about 10⁶.
  - Where σ ≈ 2, a few rows' E Z² is astronomically large.
  - Theorem M's first-order form divides by the exact normalizer instead of the design's random one.
  - The design's own normalization keeps the actual error bounded.
- There the bridge needs the realized MSE (which works, 0.76) or a second-order Theorem M. This is §10's first-order
  limit, seen end to end.

**Run-time targets for the certified read (τ).** The certified read's target so far, FP8's per-head error, is an oracle:
it needs the exact rows. Proposition KL turns a KL budget B into per-head targets computable at run time:

  τ_l² = 2B / (L·H·c_l), so that ½·Σ_l c_l·Σ_h τ_l² = B.

If every head's error is certified within τ_l, the predicted KL on that input is at most B. This holds to second order,
with probability ≥ 1 − δ per call.

*Evidence* (Llama-3.2-1B, held-out units, part 8's bound, 6-bit keys). Each rule's parameter was swept.
- **The KL-budget rule** reaches every predicted-KL level at the least traffic. It beats a per-layer quantile of FP8's
  error, a relative target ρ‖õ‖ and the oracle itself.
- **The oracle is the wrong currency.** Heads differ in how much their error moves the logits. The certificate is also
  about 8× the error, so the old target buys far more accuracy than needed: traffic 0.475 for a predicted KL of 10⁻⁶.
- **Large reductions cost almost no traffic.** Against the vote alone (predicted KL 2.1·10⁻⁴ at 0.398), the KL-budget
  rule cuts the predicted KL 7× at 0.399 and 20× at 0.402. It reads extra rows only where the certificate flags a large
  error in a sensitive head.
- **End to end** (the certified read inside the forward pass, 4 held-out inputs): it meets its budget within 15%.
  - Budget 3·10⁻⁵: measured 3.2·10⁻⁵ against 2.9·10⁻⁵ predicted.
  - Budget 10⁻⁵: measured 9.2·10⁻⁶ against 7.9·10⁻⁶ predicted.
  - It cuts the 6/4 design's KL 3.7× at +0.0004 traffic and 14× at +0.003, against FP8's KL of about 4·10⁻⁴.
  - The certified read thus becomes a deployable contribution: a certified, query-aware patch to the vote.

*Refinement* (2026-10-12, `findings/findings_klbridge.md` §6). Llama-3.2-1B and Qwen3-1.7B, held out. Offline, each
unit's KL is predicted with its own sensitivities; the key comparisons were also run end to end.
- **A target rule is a shape across layers times one global level.** The ranking above used the calibration units'
  c_l both to build the rule and to score it, which favors the KL budget.
- **Shape.**
  - On Llama-1B, one flat τ for every layer does as well as the KL budget, offline and end to end (0.91–1.06× its KL at
    the same traffic).
  - On Qwen3-1.7B, whose sensitivities fall by up to 3,300× with depth, the KL budget beats flat by 1.35–1.4× end to end.
    Flat spends rows in late layers that barely move the logits.
  - Each input's own c_l would be only 10–25% better than calibrated ones, and c_l pooled across families loses at most
    10% to per-family c_l.
  - Shapes built on FP8's per-head error are 1.2–1.8× worse than the KL budget.
  - So the KL budget is the rule, and its sensitivities can be pooled.
- **Level.** The KL a given B reaches varies about 10× across input families and about ±2× within one. B bounds the
  predicted KL, and inputs differ in how far below τ their errors fall. A KL target therefore needs B calibrated per
  family, on at least 8–10 prompts; reporting a frontier at fixed B needs no calibration.
- **The deployed combination** (the single store, its own question-pass vote, and this certified read) on Llama-1B:
  - 5.6× and 25× below the single store's KL, at +0.001 and +0.007 traffic;
  - equal to or below the uniform tier's certified read at the same B.
- **Where the certificate is vacuous, the read costs traffic.**
  - In Qwen3-1.7B's noisy layers (σ ≥ 0.25 at 6-bit keys), part 8 exceeds the error by more than 10³, and the read
    falls back to exact rows (89% of layer 0's). The certified read then costs +0.046–0.082 traffic.
  - 8-bit keys in those 7 layers fix it: +0.001–0.009 over that store, for 2.9–5.3× lower KL.
  - This extends Corollary M-B: a noisy layer needs more key bits for its certificate even where the tail's MSE is
    acceptable.
  - At 32K, Qwen3-30B-A3B has 14 such layers of 48, and Llama-3.1-8B none. The slack grows like e^{14σ_w} (§12).
- **The FP8-quantile rule holds up end to end on Llama-1B.** It equals the KL budget at its cheapest point, and has
  1.33× its KL at 1.5% extra rows.


## References

- L. Schuchman, "Dither signals and their effect on quantization noise," IEEE Trans. Commun.
  Technology, 1964.
- I. Pinelis, "Optimum bounds for the distributions of martingales in Banach spaces," Ann. Probab. 22
  (1994), Thm 3.4.
- A. Maurer, "A bound on the deviation probability for sums of non-negative random variables,"
  J. Inequal. Pure Appl. Math. 4 (2003).
- B. Laurent and P. Massart, "Adaptive estimation of a quadratic functional by model selection," Ann. Statist. 28
  (2000), Lemma 1.
- S. V. Nagaev, "Large deviations of sums of independent random variables," Ann. Probab. 7 (1979) (truncation at the
  level of the summands, §12).
- D. Hsu, S. Kakade, T. Zhang, "A tail inequality for quadratic forms of subgaussian random vectors,"
  Electron. Commun. Probab. 17 (2012), Thm 1.
- O. Bousquet, "A Bennett concentration inequality and its application to suprema of empirical processes,"
  C. R. Acad. Sci. Paris 334 (2002).
- F. B. Weissler, "Logarithmic Sobolev inequalities and hypercontractive estimates on the circle," J. Funct. Anal. 37
  (1980) (the circle's sharp log-Sobolev constant; §6g's P-LSI and its λ < 0 side).
- M. Ledoux, "Concentration of measure and logarithmic Sobolev inequalities," Séminaire de Probabilités XXXIII,
  LNM 1709 (1999) (Herbst's argument, §6g).
- S. G. Bobkov and G. P. Chistyakov, "On concentration functions of random variables," J. Theoret. Probab. 28 (2015)
  (a log-concave density is at most 1/its standard deviation; §6g's P-exact).
