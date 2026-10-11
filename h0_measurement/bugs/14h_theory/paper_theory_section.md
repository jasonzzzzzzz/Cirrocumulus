# Theory section: skeleton for the design paper (draft, 2026-10-10)

Step 2 of `STATUS.md`, with `CLAIMS_LEDGER.md`. It gives the section's structure, statements in paper form, the proof
route for each (full proofs: `theory.md`, which becomes the appendix), and an evidence slot per statement:
- a filled slot holds a number that exists today, with its source;
- **[step n]** marks a number to come from that step in `STATUS.md`'s ranking.

Author notes are in brackets.

[Length: about two pages in the main text. Statements and one-line proof ideas go there; proofs, brute-force checks and
the 4-bit study go to the appendix.]

---

## 4. Why evicting fails, and why reading the tail works

### 4.1 Setting

Take one attention head at one decoding step:
- a query q;
- context rows i with keys k_i, values v_i, scores s_i = ⟨q, k_i⟩/√d (plus any known bias) and weights w_i = e^{s_i};
- fixed rows (attention sinks, the question, generated tokens) with total weight W_F and mean value o_F, always exact.

The exact output is o = (W_F o_F + Σ_i w_i v_i)/Z, with Z = W_F + Σ_i w_i. A design reads a set S of rows exactly; T is
the rest.
- **Evicting** drops T: o_S is the softmax over S and the fixed rows.
- **The tail design** reads T from a low-bit tier: weights w̃_i and values v̂_i.

Write M = W_F + Σ_S w_i for the exactly read mass and c for the output of the exactly read rows alone.

### 4.2 Eviction cannot be certified before the answer is seen

**Lemma 1.** o − o_S = ε(ō_T − ō_S), where ε = Σ_T w_i/Z is the unread mass, and ō_T, ō_S are the unread and read rows'
mean values.

*Proof idea:* o is the ε-mixture of the two means.

So evicting is accurate exactly when the answer's attention is concentrated on S. A set chosen at question time cannot
bound ε for queries that do not exist yet. [One paragraph tying this to the measurements:
- single needles: near FP;
- multi-value, aggregation, re-ranking: the vote fails, and the loss split puts the vote, not the budget, at fault
  (`../14h_methodology_improve/findings/R4.md`, `R5_0_split.md`; exploratory, `CLAIMS_LEDGER.md` C3);
- hence nothing is evicted in our design.]

### 4.3 A randomized low-bit tier

Each cached row x (a key or a value, per KV head) is stored as follows:
1. rotate it, y = Rx;
2. set the step D = 2‖y‖_∞/(2^b − 1), stored per row;
3. draw a dither u ~ U(−D/2, D/2)^d from a counter-based generator keyed by (seed, layer, head, row), never stored;
4. keep the code ⌊(y + u)/D⌋;
5. reconstruct ŷ = D(⌊(y + u)/D⌋ + ½) − u.

Storage equals a plain b-bit store plus one scale per row. Because the dither is subtracted and the code never clips,
ŷ − y is uniform on [−D/2, D/2]^d **independently of y** (Schuchman, 1964). Three consequences are used below; all hold
for every cache content:
- (P1) row errors are independent of each other and of the data;
- (P2) the score error δ_i = s_i − ŝ_i is symmetric and sub-Gaussian with proxy σ_i² = ‖q‖²D_i²/(12d), and its moments
  are exact: E e^{−kδ_i} = ∏_j sinh(k a_ij)/(k a_ij), with a_ij = |(Rq)_j|D_i/(2√d);
- (P3) the value error is isotropic, with E‖e_i‖² = d_v D_{v,i}²/12.

The tail divides each tier-1 weight by E e^{−δ_i}. This makes the tail's sums unbiased (the **unbiased tail**); without
it every low-bit tail over-weights itself by about σ²/2.
[Evidence for that bias: `findings/R5_4.md` §2; exploratory.]

### 4.4 The certificate

**Theorem R.** Assume:
- the read set S is chosen without the tier's dither, e.g. from exact attention at prefill or from a separate sketch;
- the dither seed is drawn independently of the input.

Then for every cache content, query and δ ∈ (0, 1), with probability at least 1 − δ over the dither,

  ‖o − õ‖ ≤ B_δ,

where B_δ is computed at run time from the query, the exactly read rows, the tier-1 rows and their steps.

*Proof idea.*
- **Identity:** Z(o − õ) = Σ_T D_i (v_i − c) − (õ − c) Σ_T D_i + Σ_T w̃_i e_i, with D_i = w_i − w̃_i. Each sum has
  independent summands (P1).
- **Truncation:** truncate every score error at a union level.
- **Concentration:** the value sum has an isotropic form and gets a χ² tail (P3). The key sum, given the value coins,
  is a sum of independent scalars along observed vectors. It gets the Hsu–Kakade–Zhang quadratic-form bound, with a
  sub-Gaussian proxy for each row's weight ratio computed from the dither's exact law (P2). The unknown true weights
  are bounded by sums that hold surely on the truncation event. Bennett's inequality in Hilbert space is the simpler
  alternative.
- **Maxima:** every maximum over rows is attained at a fixed row, so one row's deviation suffices inside it.
- **Union:** a union bound over the finitely many events.

[Full proof: `theory.md` §6. State the v1 form as the simple version, and part 8 (§6f–§6g: the split with a tilted tail
and the HKZ key sum) as the tightest valid form at every width.]

**Corollary (layers and decoding).** Suppose each layer's dither is drawn independently. Then Theorem R holds for every
head-step of every layer of a teacher-forced pass, since a layer's queries depend only on the layers below. It also
holds in free-running decoding, up to the first token that differs from the exact model's decode. Before that token,
free-running decoding is a teacher-forced pass over a sequence that does not depend on the dither.
[`theory.md` §11, Lemma C and Theorem DD. The protocol matches: R6's KL is teacher-forced, and its accuracy is
free-running.]

[Remark on δ: the statement is per call. A whole decode of N head-steps runs each at δ/N; the constants grow like
√ln N.]

*Evidence.*

| | value now | source | to come |
|---|---|---|---|
| validity | 0 misses in 31,080,448 head-step checks (Llama-3.1-8B 128K, Qwen3-30B-A3B 32K; 6- and 8-bit keys; v1); no failure in brute force on adversarial data at every δ | `findings/R5_4.md` §1; `test_theory*.py` | R6's probe (v1, v2; 6-bit; 7 blocks) [1] |
| tightness, the design's width (6-bit keys), the vote's rows | bound / error 37–38 (Llama), 41–58 (Qwen) with v1 | R5.4 | part 8 at 128K [3] |
| share of head-steps certified within FP8's per-head error | 0.44–0.52 (Llama), 0.38–0.41 (Qwen), v1, 6-bit; 0.98–0.99 (Llama) with C/32 more exact rows | R5.4 | part 8 at 128K [3] |
| CPU, part 8, 6-bit keys | 6.7–7.3× the error, 0.41–0.59 certified within FP8 (Llama-1B L1/L3, 1.6K–8K; v1-era forms 17–19× and 0.13–0.21) | `findings/findings_part8.md` | — |
| CPU, part 8, 8-bit keys | 2.0–2.6× the error, 0.89–0.95 certified within FP8 (Llama-1B L1/L3) | same | — |

[Table T1 in the paper: coverage, bound/error and certified share per model × key width × read set. It is filled from
the R7 probe (step 3); CPU numbers go to the appendix.]

### 4.5 Why it works: the error is a second moment of the unread attention

**Theorem M.** Let the tier have independent key and value dithers (grouped-query attention). With Z̃ = M + Σ_T w̃_i (the
design's own normaliser), exactly

  Z̃(o − õ) = N := Σ_T w_i(1 − Z_i)(v_i − o) + Σ_T w_i Z_i e_i,  Z_i = w̃_i/w_i,

and

  E‖N‖² = Σ_T w_i² [Var(Z_i)·‖v_i − o‖² + E(Z_i²)·E‖e_i‖²] + ‖Σ_T w_i(1 − E Z_i)(v_i − o)‖².

The last term vanishes for the unbiased tail. Hence E‖o − õ‖² ≤ E‖N‖²/M², with equality to first order with Z in place
of M. Every quantity is exact: (P2) gives the moments and (P3) the value term.

*Proof idea:* N's summands are independent with known means; expand the square.

**Consequences** (one sentence each in the text):
1. **What sets the error.** The tier's noise enters through Σ_T w_i², the unread rows' share of the attention's second
   moment, not through their mass. Reading the few heavy rows exactly removes most of it.
2. **Against eviction.** Eviction's error is a first-order bias ε(ō_T − ō_S). The tail's is a standard deviation of
   about σ·ε·spread/√n_eff, which averages over the n_eff effective unread rows.
3. **Key bits against value bits.** The key term scales with Var Z ≈ σ², which each key bit divides by 4; the value
   term scales with E‖e‖², which each value bit divides by 4. The formula separates the two per model and layer.
4. **When the design fails.** Flat attention (a large unread second moment) makes the tail no better than a dense
   low-bit store.

*Evidence.*

| | value now | source | to come |
|---|---|---|---|
| exactness | Monte Carlo within 4.5 s.e.; the identity to 10⁻¹³ | `test_theory3_s1h5.py` | — |
| accuracy on real layers | measured/predicted MSE 0.98–1.00 (GPT-OSS at 4-bit keys: 0.92) on Llama-1B L0/L1/L3, GPT-OSS, Gemma 3, at 1.6K and 8K | `findings/findings_4bit.md` §5 | at 128K on the paper's models [3] |
| mechanism | Llama-1B L1: 2.4% of the mass but 0.18% of the second moment unread; tail/dense MSE predicted 0.0186, measured 0.0187 | same | the unread share against each family's KL in R6 [3] |
| failure case | Gemma 3 global layer: 66% of the second moment unread; tail/dense MSE 0.66 | same | — |
| link to the reported metric | — | — | E[KL] ≈ ½ Σ_h c_h·MSE_h against R6's KL per family [7] |

[Figure F-M: the unread second-moment share per family (x) against KL per token (y), for the tail and dense arms.
It shows the theory predicting which tasks are hard. Needs step 3's columns on R6's units.]

### 4.6 A certified adaptive read (optional; decision 2 in `CLAIMS_LEDGER.md`)

**Corollary.** Take a nested sequence of read sets S_0 ⊂ … ⊂ S_m, each chosen without the dither. Read them in order and
stop at the first whose Theorem R bound, at confidence δ/m, meets a target τ. With probability ≥ 1 − δ the returned
output is within τ, whatever the stopping rule.

*Proof idea:* a union bound over the m sets; the coins choose how many sets are read, never which.

*Evidence* (CPU, Llama-1B, part 8's bound):
- at 8-bit keys, FP8's per-head error is met at traffic 0.456–0.460 on L1/L3, against the oracle stop's 0.453;
- at 6-bit keys it is met at 0.480 (L1) and 0.499 (L3) with a 15-set schedule, and 0.433 on L1 at 8K, against
  FP8's 0.50;
- the first layer (flat attention) needs 0.73 at 6 bits.

The target used so far, FP8's per-head error, is an oracle; a run-time τ is step 6. GPU: [3].

### 4.7 Limits

- **End to end.** No attention-level bound implies an output-level one in general: a size-τ change in one head can
  flip a near-tied token in any format, FP8 included. We therefore test end-to-end quality directly (§5, R6).
- **4-bit keys.** Theorem R holds but is not tight enough to certify FP8-level error. The information exists (the
  oracle certifies 62% of head-steps on Llama-1B), but every bound of this family pays a union over rows inside a
  lognormal-like weight error (`theory.md` §12). Theorem M still explains the 4-bit design's accuracy.
- **Deployment requirements.** The read set must not see the dither, and the seed is drawn per request. A vote over the
  dithered tier itself breaks the first; a vote on exact keys at the question's prefill or a separate sketch meets it,
  and its cost belongs in the memory accounting (`CLAIMS_LEDGER.md` S2, step 5).
- **Architectures.** Theorem R covers latent attention (MLA) in a coupled form (`theory.md` §6). Theorem M's MLA form
  has an extra cross term that we do not evaluate [8].

---

## Appendix plan

| appendix | content | from |
|---|---|---|
| A. Proofs | Lemma 1; the randomized tier's facts; Theorem R (v1, then v3, the sure sums, the split); Lemma C and Theorem DD; Theorem M; the certified read | `theory.md` §2, §6, §10, §11 |
| B. Validity | brute force on random, spiky, outlier and adversarial data at every δ; GPU coverage | `test_theory*.py`; R5.4; R6; R7 |
| C. Tightness | bound / error and the oracle width on CPU; the 4-bit study (where the slack comes from, what does not close it) | `findings/findings_4bit.md`, `findings_part7.md` |
| D. Architectures | the adapters (GQA, sinks, local/global, MLA, hybrids) and their certificates | `findings/R5_adapters.md`, `R5_theory_mla.json` |
