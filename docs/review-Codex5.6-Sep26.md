# Review of latex/main.tex: correctness, novelty, and paper positioning

**Review date:** 2026-09-22  
**Manuscript reviewed:** [latex/main.tex](../latex/main.tex) and [latex/appendix.tex](../latex/appendix.tex)  
**Other evidence checked:** the current R3--R8 status in [h0_measurement/ROADMAP.md](../h0_measurement/ROADMAP.md), the figure-generation code, and the co-design reports.

## Executive decision

I would change the emphasis of my earlier recommendation, but not abandon it.

1. **I agree that the paper should not begin with RDKV.** It should begin with our problem and result. RDKV belongs in a concise concurrent-work paragraph, not in the first sentence or title.
2. **I agree that ICLR 2027 policy gives real protection for arXiv-only work.** The official reviewer guidance says authors are not required to compare against work available only on arXiv and that the absence of such a comparison cannot itself be a basis for rejection.
3. **I do not agree that this makes a broad “first joint eviction and quantization” claim strategically safe.** The policy protects authors from a required comparison; it does not make readers forget a very close idea or remove ICLR's requirement that a paper contribute new knowledge. More importantly, DiffKV is archival SOSP 2025 prior work and already jointly uses high precision, low precision, and pruning.
4. **The joint formulation should remain prominent.** It is the conceptual foundation and can still be presented as our independently developed formulation. But the paper-specific headline should be what the formulation reveals: **when an interior allocation is useful, when it collapses to a corner, and how context length moves a model between those regimes.**
5. **The current draft is not yet defensible as written.** The central eviction derivation needs correction; the plotted experiment does not match the described symmetric experiment; the three-regime “phase diagram” is only partially identified by \(D_2\); the memory budget is not a total-KV matched budget; and Sieve's end-task effectiveness is not yet established.
6. **The project itself is considerably stronger than the manuscript.** R3--R7 and the co-design study have already answered several of the draft's TBDs. The honest symmetric advantage survives, the boundary is architecture-conditioned rather than universal, the routing decision is stable while the allocation goes stale, the GQA cost is moderate, and a 4-bit cascade recovers most of the lag penalty. These results support a good characterization/design-principle paper if they replace the stale claims and figures.

My recommended title is:

> **When Does Joint Key-Cache Allocation Pay? Context-Dependent Regimes in Quantization and Eviction**

If the submitted method genuinely compresses and budgets both keys and values, “KV-Cache” is appropriate. Under the current experiment, “Key-Cache” is the accurate scope.

The one-sentence thesis I would sell is:

> Quantization and eviction can be treated as actions in one rate-allocation problem, but the interior of that problem is not uniformly valuable: a derived tier-viability statistic predicts when mixed allocation helps, and context length can move the same model toward an eviction-like regime.

That framing keeps the joint idea at the front without making the paper depend on exclusive priority for the joint idea.

---

## 1. Response to the three RDKV considerations

### 1.1 “RDKV is an arXiv paper, not yet justified by the community”

This is formally relevant but should be expressed carefully.

As of this review, [RDKV](https://arxiv.org/abs/2605.08317) is publicly listed as an arXiv preprint submitted on May 8, 2026; its arXiv record does not list an archival venue. It has therefore not received the evidentiary status of an accepted peer-reviewed paper. That supports calling it **concurrent arXiv work**, not ordinary established prior work.

I would not say in the paper that RDKV is “unjustified by the community.” Peer review is not a correctness oracle, and reviewers may regard that phrase as dismissive. The useful distinction is simply:

> Concurrent with our work, RDKV independently studies joint rate allocation for eviction and quantization. We cite it for completeness and distinguish the scientific question addressed here.

We do not have to organize our introduction around it. We should nevertheless cite it, because we know it exists and because doing so lets us state our differentiation before a reviewer states it for us.

### 1.2 What the ICLR 2027 policy does and does not say

Your narrow reading is correct. The [ICLR 2027 Reviewer Guidelines](https://iclr.cc/Conferences/2027/ReviewerGuidelines) say that:

- arXiv is not a peer-reviewed venue;
- authors are not required to compare against papers available only on arXiv;
- citation is nevertheless strongly encouraged when the work is known; and
- failure to make such a comparison cannot, by itself, be a rejection basis.

The [ICLR 2027 Author Guidelines](https://www.iclr.cc/Conferences/2027/AuthorGuidelines) also permit papers on non-peer-reviewed sites such as arXiv and give the actual full-paper deadline as September 25, 2026, 11:59 PM AoE.

There is an internal policy-page inconsistency worth knowing about. The reviewer FAQ still mentions a September 16 deadline and a July 17 two-month cutoff, while the [Area Chair Guide](https://iclr.cc/Conferences/2027/AreaChairGuidelines) says contemporaneous peer-reviewed work means work from the last four months. This looks like template drift. It does **not** change the clearer rule for arXiv-only work: comparison is encouraged but not required.

The safe interpretation is:

> A reviewer should not reject the submission merely because it fails to cite or compare with RDKV as an arXiv-only paper.

The unsafe interpretation is:

> RDKV cannot influence novelty, significance, reviewer enthusiasm, or the final decision in any way.

The same reviewer guide asks whether a submission contributes “new knowledge,” is well placed in the literature, and presents novel findings. A conscientious reviewer may follow the contemporaneous-work rule and still conclude that the independently valuable part of our paper is the regime analysis rather than the shared allocator. An area chair may also see both ICLR submissions during discussion. Thus the policy is a shield against an unfair priority test, not a substitute for a clear intellectual distinction.

### 1.3 Should we still lead with joint allocation?

**Yes as the conceptual hook; no as the sole novelty claim.**

I would open with the joint question:

> Eviction and quantization spend the same memory budget using different actions. What does an allocator choose when both actions are available, and when is that extra choice useful?

Then immediately give our distinctive answer:

> The mixed interior is regime-dependent. It is valuable for some heads and contexts, but disappears when finite-bit tiers become dominated; increasing context can move the same model toward that collapse.

This lets the paper “start with joint” in the meaningful sense. It does not start with RDKV, and it does not spend its novelty budget on a priority contest over a shared formulation.

The formulation can remain Section 2 and can remain one of the contributions, described as an **independently developed foundation**. It should not be worded as “the first unification,” “the first zero-rate view,” or “the first KV rate-distortion allocator.”

---

## 2. Position relative to the closest papers

### 2.1 RDKV

[RDKV](https://arxiv.org/abs/2605.08317) has the largest narrative overlap. It explicitly:

- casts KV compression as rate-distortion allocation;
- describes quantization and eviction as endpoints of one allocation curve;
- uses reverse water-filling and a zero-bit action;
- discretizes over \(\{0,2,4,8,16\}\);
- allocates value tokens and key channels;
- includes downstream benchmarks and a packed decoding implementation.

Therefore, the following are not good exclusive novelty claims, even if RDKV is formally concurrent:

- first to unify eviction and quantization;
- first to recognize eviction as zero rate;
- first reverse-water-filling allocator for KV cache;
- first “phase transition” between zero and finite rate;
- first discrete joint allocator.

The strongest genuine distinction is the question being answered:

> **RDKV asks how to build and realize a joint allocator. This project asks when the allocator's interior is useful, whether that can be predicted before deployment, and how the answer changes with model architecture and context length.**

There are also concrete scope differences:

- RDKV's implemented allocation is token-wise for values and channel-wise for keys; this project studies token-wise key precision and token eviction through attention-output sensitivity.
- RDKV applies allocation after prefill from prefill statistics. This project measures score information, decode-time staleness, and context-dependent regime movement.
- RDKV demonstrates a method and kernel. This project's distinctive evidence is the dead-tier diagnostic, the architecture-conditioned boundary, context trajectories, and the two-timescale result: routes remain stable while allocations require refreshing.

Recommended wording:

> Concurrent RDKV also casts eviction and quantization as joint rate allocation. Our focus is complementary: we characterize when a nontrivial token-precision interior improves over both endpoint policies and show that this regime changes systematically with context length.

That is a strong distinction. It neither concedes the paper nor pretends the overlap is absent.

### 2.2 RateQuant

[RateQuant](https://arxiv.org/abs/2605.06675), first posted April 22, 2026, also uses rate-distortion theory and reverse water-filling for KV-cache mixed precision. Its allocation is principally across K/V heads, with an offline loss-gradient sensitivity, quantizer-specific distortion \(D(b)=\alpha\beta^{-b}\), and a positive minimum precision rather than token eviction. It also derives an AM/GM sensitivity ratio as a predictor of when mixed precision helps.

The clean distinction is one of allocation axes and actions:

- **RateQuant:** across-head allocation, K/V separation, finite-bit quantization, loss sensitivity, offline calibration.
- **This project:** within-head token allocation, an explicit zero-rate eviction action, local attention-output distortion, and context-dependent tier survival.

This is more than a cosmetic distinction, but the paper must compare \(D_2\) against RateQuant-style heterogeneity measures such as \(\operatorname{var}(\log w)\) or the AM/GM ratio. Otherwise a reviewer can reasonably ask whether dead-tier fraction is simply another monotone summary of sensitivity dispersion.

Safe wording:

> Our token-level, zero-inclusive allocation is orthogonal to RateQuant's across-head finite-bit allocation; the two could be composed.

Unsafe wording:

> We are the first to apply rate-distortion or water-filling to KV-cache quantization.

### 2.3 CAOTE and ReST-KV

[CAOTE](https://arxiv.org/abs/2504.14051) directly minimizes attention-output change for eviction. For a single removed token \(j\), it derives the exact score

\[
\frac{a_j}{1-a_j}\lVert v_j-o\rVert.
\]

This is directly relevant to the central derivation in the current draft. CAOTE is eviction-only: it does not solve finite-bit allocation, derive tier viability, or characterize when mixed precision beats the corners. Thus it is a foundation/adjacent method rather than a substitute for this paper.

The scientifically accurate relationship is:

> CAOTE derives exact singleton attention-output eviction error. We use a stated separable, small-attention approximation to place the zero-rate action inside a mixed-rate allocator and study the resulting regime structure.

Do not claim first use of attention-output error or first value-aware eviction criterion. Also cite [ReST-KV](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8be9c134bb193d8bd3827d4df8488228-Abstract-Conference.html), an archival ICLR 2026 paper that models attention redistribution and output reconstruction for eviction.

### 2.4 DiffKV

[DiffKV](https://arxiv.org/abs/2412.03131) is the most important non-concurrent obstacle to a generic “joint” claim because it is an archival [SOSP 2025 paper](https://dblp.org/rec/conf/sosp/ZhangHZLC25.html). It explicitly calls pruning an extreme case of quantization and assigns important tokens to a high-precision tier, moderately important tokens to a low-precision tier, and unimportant tokens to pruning. It adapts by request, head, and sequence length and includes a real GPU memory manager.

Consequently, the draft should not say that the literature cleanly splits into methods that either quantize every token or evict tokens at full precision. That taxonomy is useful historically, but false as a description of the current literature. DiffKV also blocks claims of first hybrid tiering, first per-head adaptive mixture, and first length-adaptive mixture.

The important distinction is that DiffKV uses profiled threshold rules, whereas this project attempts to derive a distortion objective and a predictive criterion for whether the intermediate tier is useful.

Recommended wording:

> DiffKV demonstrates that hierarchical high-/low-precision/pruned storage is effective and implementable. We provide an output-distortion account of when such an intermediate tier should exist and an architecture- and context-dependent diagnostic of when it improves over the endpoints.

This is an especially good use of the current observations: the paper can explain a design class that DiffKV demonstrates empirically.

### 2.5 Attention-Aware Transform Coding (AATC)

[KV Cache Compression Through the Lens of Transform Coding](https://arxiv.org/abs/2608.14191), posted August 14, 2026, derives an attention-output distortion decomposition under a white-noise model and uses reverse water-filling for Attention-Aware Transform Coding (AATC). It includes both key and value effects, transformed-channel allocation, downstream results, and zero-bit transformed dimensions corresponding to rank reduction.

Its implemented allocation is channel-wise; the paper explicitly treats token-wise allocation/eviction as orthogonal and outside its method. This creates a clean complementarity:

- **AATC:** which transformed channels deserve bits, globally calibrated over K and V;
- **this project:** which tokens deserve key bits or eviction, and when the token-wise interior is useful at a given context.

Safe wording:

> AATC develops the channel axis of attention-aware transform coding; we develop and characterize the context-dependent token axis. These axes are composable.

Unsafe wording:

> We are the first to optimize attention-output distortion with water-filling or to permit zero-bit components.

### 2.6 Compact positioning table

| Work | Allocation unit/action | What it already establishes | What this paper can own |
|---|---|---|---|
| DiffKV, SOSP 2025 | token high/low precision/pruning; per request/head | hybrid tiering, length adaptation, GPU system | derived criterion explaining when the middle tier pays |
| CAOTE; ReST-KV | token eviction | output-aware eviction and redistribution | finite-rate extension plus regime characterization |
| RateQuant, concurrent arXiv | K/V head, finite bits | KV rate-distortion, reverse water-filling, heterogeneity predictor | within-head zero-inclusive token allocation and context dependence |
| RDKV, concurrent arXiv | V tokens, K channels, 0--16 bits | joint allocator, zero rate, discrete solver, kernel | predictive “when does the interior help?” map and context traversal |
| AATC, concurrent arXiv | transformed K/V channels, including zero | output-aware transform coding and channel allocation | token-axis tier viability and eviction/quantization regime |

The related-work message should be “orthogonal axes plus a new predictive question,” not “all previous work optimizes within only one family.”

---

## 3. What is actually novel enough to sell

The following observations appear both interesting and materially differentiated:

1. **The value of joint allocation is predictable rather than universal.** Most hybrid papers propose a mixture. This project asks when the mixture adds anything over the better endpoint.
2. **Tier extinction gives a mechanistic regime coordinate.** Comparing absolute finite-bit logit-noise cost with a zero-rate cost is more interpretable than tuning a high/low/prune ratio.
3. **Context length changes the preferred topology within a fixed model.** The same architecture can move from allocation-friendly toward eviction-like behavior as context grows. This is stronger than another cross-model leaderboard.
4. **The boundary is architecture-conditioned, not universal.** R6's negative result improves the science: \(D_2\) strongly orders behavior, but an identical global threshold is not justified.
5. **Route and allocation live on different timescales.** The head-level route remains stable over thousands of tokens, while a frozen token allocation becomes stale. This is a useful design principle that neither “calibrate once” nor “recompute everything” captures.
6. **GQA and score acquisition have measured, moderate costs.** The group constraint and cascade results turn apparent deployment objections into quantified tradeoffs.

These can support an ICLR paper if the manuscript treats them as the primary knowledge contribution. ICLR's own reviewer guidance explicitly says a paper need not be state of the art if it convincingly contributes relevant, impactful new knowledge. That favors a rigorous characterization paper when method-level effectiveness is incomplete.

The following are enabling results, not sufficient headline novelty by themselves:

- reverse water-filling;
- \(b_i^\star=\log_2 a_i+c\) under an exponential high-rate model;
- treating zero bits as an available action;
- using attention-output error;
- a hybrid of eviction and quantization.

---

## 4. Correctness of the central derivation

### 4.1 Quantization expansion: valid only under stated assumptions

For a logit perturbation vector \(\delta\), the first-order output perturbation is

\[
\Delta o \approx \sum_i a_i(v_i-o)\delta_i.
\]

The separable expected squared error

\[
\mathbb E\lVert\Delta o\rVert^2
\approx
\sum_i a_i^2\lVert v_i-o\rVert^2\operatorname{Var}(\delta_i)
\]

follows if the perturbations are zero mean and cross-token covariance terms vanish. Real low-bit quantization errors can be biased, correlated with logits, and correlated across coordinates/tokens, so this is a surrogate whose accuracy must be validated—not an identity.

The project does something important and correct here: it uses the surrogate to choose allocations but scores their error by exact recomputation. That separation should be prominent.

### 4.2 The eviction equality in the draft is incorrect

Let \(E\) be the evicted set and \(A_E=\sum_{i\in E}a_i\). After deleting \(E\) and renormalizing,

\[
o_{\setminus E}
=\frac{o-\sum_{i\in E}a_i v_i}{1-A_E},
\]

so

\[
o_{\setminus E}-o
=-\frac{\sum_{i\in E}a_i(v_i-o)}{1-A_E}.
\]

Therefore,

\[
\lVert o_{\setminus E}-o\rVert^2
=\frac{1}{(1-A_E)^2}
\left\lVert\sum_{i\in E}a_i(v_i-o)\right\rVert^2.
\]

This is not generally

\[
\sum_{i\in E}a_i^2\lVert v_i-o\rVert^2.
\]

The exact set error contains the renormalization denominator and cross-token inner products. For one token \(j\), the exact magnitude is

\[
\frac{a_j}{1-a_j}\lVert v_j-o\rVert,
\]

which is CAOTE's result. The draft's expression is a small-mass, cross-term-free surrogate—not an exact equivalence.

This correction does not destroy the paper. It changes the theorem claim to:

> We include eviction as a zero-rate action under a separable local distortion surrogate, and then validate all selected allocations using exact recomputation.

That is honest and still useful. If a stronger set-level theorem is desired, it must explicitly control \(A_E\) and the cross terms.

### 4.3 The zero-rate constant uses inconsistent units

The main text writes

\[
\operatorname{Var}(\delta_i)=\tau^2c_{b_i}
\]

and then uses \(\tau^2c_0=1\) while calling \(c_0=1\). Both cannot hold unless \(\tau=1\).

There are two clean conventions:

1. **Absolute cost:** define \(d_b=\operatorname{Var}(\delta_b)\), with the normalized eviction surrogate \(d_0=1\). Then compare \(d_b\) with \(d_0\).
2. **Relative cost:** retain \(d_b=\tau^2c_b\). Then the zero-rate relative constant is \(c_0=1/\tau^2\), while the absolute zero-rate cost remains \(d_0=1\).

The implementation in [sievelib/alloc.py](../sievelib/alloc.py) already uses the clearer absolute convention (sig2[0] = 1.0). The paper should follow the implementation and write \(d_0=1\), \(d_b=\operatorname{Var}(\delta_b)\). The tier test is then \(d_b>d_0\), without mixing relative and absolute quantities.

### 4.4 Softmax removes only common-mode bias

“Bias costs nothing” is too broad. Softmax is invariant to adding the same scalar to every logit. A token-dependent mean error or a bias correlated with \(s_i\) changes attention. The correct statement is:

> Common-mode logit shift is exactly canceled; after removing that component, residual mean and covariance may still matter.

The implementation's fitted temperature component is useful evidence here, but it does not justify discarding arbitrary bias.

### 4.5 The continuous water-filling rule is not the discrete optimizer

Under independent noise and a continuous distortion law proportional to \(4^{-b}\), the stationary solution is indeed

\[
b_i^\star=\log_2a_i+\log_2\lVert v_i-o\rVert+c.
\]

For the actual discrete tier set, the optimizer is instead

\[
b_i^\star(\lambda)
=\arg\min_{b\in\mathcal B}
\left\{w_i d_b+\lambda b\right\},
\qquad
w_i=a_i^2\lVert v_i-o\rVert^2.
\]

That is what the code implements. Present the logarithmic expression as the ideal continuous relaxation and the per-tier argmin as the actual allocation rule.

### 4.6 Be precise about “dead” tiers and the convex envelope

If a positive-bit tier has \(d_b\ge d_0\), it is strictly dominated by eviction: it consumes more bits and has no lower surrogate distortion. This is a valid sufficient condition for extinction.

The converse is false. A tier with \(d_b<d_0\) may still lie above a chord between two other tiers and never minimize \(w_i d_b+\lambda b\) for any \(\lambda\). Thus:

- \(d_b\ge d_0\) is sufficient for a tier to be dead;
- \(d_b<d_0\) is necessary but not sufficient for lower-convex-envelope membership.

The appendix's “if and only if” formulation should be removed.

### 4.7 The value-term claim overstates what was measured

The manuscript says \(\log_2\lVert v_i-o\rVert\) is at most \(0.035\) bits. The implementation actually compares two **standard deviations**:

- ladder_bits = std(log2(\(a_i\lVert v_i-o\rVert\))), and
- ladder_bits_a_only = std(log2(\(a_i\))).

A small difference between these two dispersions does not imply that every per-token value term is at most \(0.035\) bits, nor that removing it never changes tier assignments. It shows that the value term changes the aggregate ladder-width statistic little in the measured data.

The safe claim is:

> Adding the value factor changes the measured ladder-width statistic by at most 0.035 bits in our configurations.

A claim that the value term is operationally negligible needs a direct ablation: allocate with and without it and compare tier assignments and exactly recomputed error.

### 4.8 The ladder-width relation is an identity

Because

\[
\log_2a_i=\frac{s_i}{\ln2}-\log_2 Z,
\]

the equality \(\operatorname{std}(\log_2a)=\tau/\ln2\) is algebraic, not empirical validation. The code now documents this correctly. Keep it as an explanatory identity or consistency check, not as evidence that the model predicts real allocation behavior.

---

## 5. The “phase diagram” needs two coordinates

The manuscript claims three regimes:

1. diffuse / uniform;
2. productive mixed ladder;
3. sharp / eviction-like.

But \(D_2\) principally measures the sharp-side boundary: whether a low-bit tier is dominated by the zero-rate action. The diffuse boundary is a different condition—whether the spread in token importance is too small for nonuniform allocation to matter.

A defensible regime map therefore needs two conceptual coordinates:

- **tier viability**, such as \(d_2/d_0\) or the number of finite-bit tiers on the lower convex envelope;
- **allocation pressure/dispersion**, such as the spread of \(\log a_i\), the spread of \(\log w_i\), or the gain predicted from weight heterogeneity.

The current data appear to cover mainly the productive-to-eviction side. A strong negative correlation between \(D_2\) and gain does not by itself demonstrate the diffuse/uniform corner. If the data do not sample that corner, say so and present a **regime trajectory** rather than a complete empirical three-phase diagram.

The physics language should also be moderated:

- “phase diagram” is acceptable as an analogy if explicitly identified as one;
- “order parameter” suggests a universal critical boundary that R6 does not support;
- “derived regime indicator” or “tier-extinction coordinate” is safer;
- “architecture-conditioned boundary” accurately reflects the observed result.

R6 is important here. Across 25 cells, the association remains strong (\(\rho=-0.978\), or \(-0.915\) after accounting for model identity), but the crossing is pinned per model rather than universal. That is a scientifically interesting result, not something to hide behind a broad universal claim.

---

## 6. Methodology and reporting

### 6.1 The symmetric experiment now exists, but the paper still plots the old one

The main text says the primary comparison is symmetric, with allocation and the corner using the same information. The current paper figures do not implement that statement. [latex/figures/make_paper_figs.py](../latex/figures/make_paper_figs.py) plots routed_or and band_or, i.e. the all-oracle contrast, while its comments/captions call the result symmetric.

This was originally an experimental validity gap. It is now principally a manuscript synchronization problem because R3 has completed the honest symmetric comparison:

- the interior advantage survives at roughly half its earlier size;
- 5 of 16 reported cells are STOP and 5 are NARROW;
- the dead-tier/band association remains strong (\(\rho=-0.985\)).

Those are credible results. Replace every headline number and figure with the symmetric cell and show the oracle result only as an upper-bound contrast.

### 6.2 “Routed gain” is currently an oracle portfolio

The manuscript defines routed gain as the geometric mean of \(\max(\mathrm{gain},1)\) and says this is what a per-head router can realize. That maximum is selected after observing which policy has lower exact error. Until a prospective routing rule chooses the arm without seeing the outcome, this metric is an **oracle-routed upper bound**.

Use one of these labels:

- oracle portfolio gain;
- hindsight-routed gain;
- achievable upper envelope.

Reserve “router gain” for router_calib evaluated on disjoint data. The new R5 result supports the feasibility of such a router—the route is stable—but does not retroactively make the max operator an achieved method.

### 6.3 The budget is matched on key bits, not total KV memory

At \(B=3\), the draft compares uniform 3-bit keys with retaining \(3L/8\) tokens at 8-bit keys. This matches the nominal **key rate**. It does not match total resident cache bytes because:

- values remain stored for every nonzero-tier token;
- eviction also removes the associated values;
- indices, tier maps, scales, codebooks, alignment, and packing metadata are omitted;
- 8-bit retained keys are not “full precision.”

This matters because a zero-bit token saves both its key and value, while moving a retained key from 8 to 3 bits does not save its value. The paper must either:

1. state that the experiment is a controlled key-rate comparison and avoid total-KV/system claims; or
2. budget actual K+V bytes including metadata.

Until the second exists, use “matched key-bit budget,” not “matched memory budget,” and use “8-bit retained keys,” not “full precision.”

### 6.4 The current scope is key-token compression

The theory and code quantize keys and retain unquantized values for nonzero tiers. RDKV and AATC cover both K and V on different axes. The title, abstract, and contributions should not imply a complete joint KV allocation unless value precision is also in the optimization.

A narrow and accurate scope is a strength:

> We study token-wise key-cache rate allocation with joint token eviction; value-cache compression and channel allocation are orthogonal axes.

### 6.5 GQA changes the implementable allocation unit

The earlier figures report allocations per query head, but in GQA multiple query heads share one KV head and cannot store different key bit-widths for the same token. The co-design experiment now quantifies this rather than leaving it as a fatal concern:

- \(1.00\times\) cost for n_rep = 1;
- about \(1.14\times\) for n_rep = 4;
- about \(1.32\times\) for n_rep = 8.

This should be reported as part of the methodology. The deployable unit is the KV-head group; query-head results are an unconstrained upper bound.

### 6.6 \(D_2\) has no explicit \(L\), but it is not length invariant

\(D_2\) is computed from \(\tau\) and tier cost at a particular model, input distribution, query position, and context length. It does not contain the symbol \(L\), but its measured value changes with \(L\). Therefore the sentence saying a new deployment length can be re-phased “with no new measurement” is unsupported and conflicts with the main result that context changes \(\tau\).

The correct claim is:

> \(D_2\) uses no explicit length feature; length acts through measured head statistics. A calibration at the intended deployment length is required unless a separately validated model predicts those statistics across length.

The new timing result allows a more interesting statement: routing can be calibrated once **for a deployment context/phase**, while token allocations should be refreshed during decoding.

### 6.7 Statistical unit and generalization need clearer treatment

The abstract says 10,240 heads, while the setup says 46,000 head measurements. Neither makes clear how many are unique architectural heads versus repeated head-context-prompt-step observations. The independent units for a cross-architecture claim are much closer to the number of models/configurations than to the raw row count.

Report separately:

- number of models;
- number of model-context cells;
- number of unique query heads and KV heads;
- prompts/seeds/steps per cell;
- total correlated observations.

The within-model correlations of \(-1.00\) are descriptive but should not be sold as overwhelming statistical evidence: each model has only a few monotonically ordered context points, and both variables share context-related mechanics. The partial correlation is better; model-cluster bootstrap intervals and leave-one-model-out prediction are better still.

The “held-out 70B at 128k” claim is useful only if the predictor, threshold, and held-out choice were fixed before looking at that cell. Otherwise call it a leave-one-cell-out check. A whole held-out architecture is a stronger test, particularly now that R6 finds architecture-specific boundaries.

### 6.8 The 2x band is interpretable but arbitrary

The share of heads with gain at least \(2\times\) is useful for visualization and a practical GO/NARROW/STOP rule, but it is sensitive to density near the threshold and to the corner set. R7 shows:

- prompt-block standard deviation of roughly 0.4--3.0 band points;
- changing the corner set moves the band by roughly 5--9 points.

Use a continuous robust gain statistic as the primary response and band fraction as a secondary operational summary. Always name the corner set and information set beside a band number.

### 6.9 “Scoring is solved” is too broad

The measured oracle-versus-lagged gap of roughly \(1.02--1.41\times\) is useful. It supports:

> Within the evaluated scoring family and workload, improving token scores contributes less than changing the allocation topology.

It does not establish that scoring is solved across tasks, future queries, or SOTA evictors. Ada-KV, DropKV, OBCache, and LaProx are built in the R8 harness but not yet run. Use the narrower claim.

### 6.10 Exact recomputation is a real strength

The study does not score its own allocations with the approximation used to choose them. It quantizes real keys, recomputes attention and output, and reports exact output error. This should be emphasized earlier because it protects the empirical findings even after the analytical surrogate is weakened.

The claimed predicted/measured range \(0.79--1.34\times\) is encouraging, but “forward prediction” should specify whether tier costs and regression quantities were estimated on the same prompts/heads. If so, call it model-fit validation. If calibration and evaluation are disjoint, state the split prominently.

### 6.11 Remove projected numeric panels

Both current paper figures contain hard-coded projected end-task/router curves. They are visibly stamped “PROJECTED,” which avoids literal misrepresentation, but invented numeric axes still reduce trust and occupy scarce space. Use a nonnumeric conceptual diagram or leave the panel out until R8 produces measurements.

---

## 7. What the completed experiments now support

The manuscript predates several results. The paper should use the following current, more defensible picture.

### R3: honest symmetric comparison

The advantage is smaller but real. Both the mixed allocator and the eviction corner use lagged/deployable information; the edge survives at roughly half magnitude, with 5/16 STOP cells. This is a better story than the oracle result because it demonstrates that the regime phenomenon is not caused entirely by information asymmetry.

### R4: absolute length versus RoPE limit

The updated result is not simply “hitting the trained window causes the transition.” \(\tau\) is mostly convex in \(\log L\), indicating a broad absolute-length effect, with a real additional \(+0.21\) \(\tau\) excess at the RoPE cap in some Qwen cells and no comparable excess in Llama-3.1. Sell this as a decomposition, not a single universal mechanism.

### R5: two timescales

The route remains stable over 4,096 decode tokens (reported p90 regret \(1.00\) in six cells), while freezing the allocation costs \(2.5--4.3\times\) relative to a fresh allocation. This directly corrects the draft's “one calibration pass does everything” implication:

> choose the compression family slowly; refresh token allocation quickly.

That is a strong design principle and may be more memorable than the name Sieve.

### R6: ordered but nonuniversal boundary

Across 25 cells the association is strong, but the crossing is model-specific. Report both facts. Do not force a universal 40--60% threshold when two architectures can differ by about eight band points at nearly the same \(D_2\).

### R7: uncertainty and corner-set sensitivity

Prompt uncertainty is modest relative to the corner-definition shift. This means the main methodological disclosure is not simply an error bar; it is the exact set of admissible corner policies. Put that definition in every result table/figure caption.

### Co-design: GQA and cascade

The shared-KV-head constraint costs about \(1.14\times/1.32\times\) at GQA ratios 4/8 rather than destroying the advantage. A 4-bit current-query cascade closes a median 89% of the lag gap. These results make a deployable interpretation plausible, but they do not by themselves establish end-task accuracy or throughput.

### R8: effectiveness remains open

P0 found question-aware SnapKV perfect at every tested budget, revealing a protocol shortcut rather than establishing the router. P0b's question-agnostic experiment is the correct next comparison, and P2 is built but not yet evaluated. Until it lands:

- do not claim Sieve improves end-task accuracy;
- do not show projected accuracy values;
- describe routing as a design implication or prospective method;
- keep the paper viable as characterization plus prediction.

If R8 is strong, Sieve can return as the practical culmination. If R8 is null, the regime result can still be the paper, but the method claims must disappear.

---

## 8. Recommended claim hierarchy

### Headline

**The preferred cache-compression topology is a property of model × context, not of the model alone.**

### Primary scientific contribution

**A derived tier-extinction coordinate predicts when the interior of a joint token-rate allocator improves over uniform quantization and eviction.**

### Mechanistic contribution

**Increasing context changes logit spread and finite-tier viability, moving heads toward an eviction-like topology; architecture shifts the boundary.**

### Validation contribution

**The surrogate chooses allocations, while exact attention-output recomputation evaluates them; the resulting regime relationship survives a symmetric information comparison, model controls, and several robustness checks.**

### Practical implication

**Routing and allocation operate on separate timescales: route per head/context slowly, re-budget tokens more frequently.**

### Supporting foundation

**We independently formulate key-token quantization and eviction as actions in one output-distortion rate-allocation problem.**

The foundation should be near the front, but it should be the last item in the novelty hierarchy, or explicitly labeled as concurrently developed.

### Claims to avoid

- “first joint eviction and quantization”;
- “all prior work treats them separately”;
- “eviction is exactly independent zero-bit quantization”;
- “\(D_2\) is a universal order parameter”;
- “\(D_2\) is independent of context length”;
- “Sieve realizes routed gain” before prospective evaluation;
- “matched KV-memory budget” for the current key-bit experiment;
- “full precision” for 8-bit retained keys;
- “scoring is solved.”

### Claims that are strong and supportable after the stated corrections

- “We study eviction and finite precision inside one token-rate action space.”
- “The mixed interior is useful only in a measurable subset of heads/configurations.”
- “A tier-extinction statistic strongly orders that usefulness across the tested models and contexts.”
- “The threshold is architecture-conditioned rather than universal.”
- “Context length can change a method ranking within one model.”
- “The symmetric/deployable comparison preserves a substantial part of the oracle advantage.”
- “Routing is stable on a slower timescale than token allocation.”

---

## 9. Recommended paper structure after accounting for concurrent work

I still recommend a structure centered on regimes, but I would now give the joint formulation more prominence than in my first recommendation.

### 1. Introduction: one budget, three actions, one missing question

Do not open with RDKV. Open with eviction, uniform quantization, and mixed allocation as competing ways to spend a cache budget. Then state the missing question:

> When does the mixed interior buy anything over its two endpoints?

Lead with the most memorable within-model reversal, not only the pooled correlation. Introduce the joint formulation as the tool that makes this question precise. End with three contributions: regime criterion, context traversal, predictive/symmetric validation.

Include one sentence near the end of the introduction:

> Concurrent work also derives joint KV rate allocation; our contribution is the context-dependent characterization of when the interior is useful.

### 2. Joint token-rate allocation

Keep this section compact and exact:

- state the key-token scope;
- derive the first-order quantization surrogate and assumptions;
- give the exact eviction formula;
- define the separable zero-rate approximation explicitly;
- use absolute tier costs \(d_0=1,d_b=\operatorname{Var}(\delta_b)\);
- present the continuous relaxation and actual discrete argmin;
- explain overlap with RDKV, RateQuant, CAOTE, and AATC in one paragraph.

This section is a foundation, not a priority polemic.

### 3. From allocation to an architecture-conditioned regime map

This is the primary theory/claim section:

- diffuse-side condition: insufficient importance dispersion;
- interior condition: multiple finite tiers survive and nonuniformity matters;
- sharp-side condition: low-bit tiers are dominated by zero rate;
- define both regime coordinates;
- define \(D_2\) as the tested sharp-side summary;
- state falsifiable predictions before showing results.

Call it a three-regime hypothesis if the diffuse corner has not been measured.

### 4. Measurement protocol

Give this a standalone section. State:

- models, contexts, prompts, seeds, decode steps;
- query heads versus unique KV heads;
- key-only precision and treatment of values;
- discrete tier set and exact key-bit accounting;
- oracle, lagged, and cascade information sets;
- uniform and eviction corner sets;
- exact output recomputation;
- calibration/evaluation split;
- primary continuous metric, secondary 2x band;
- clustered/leave-model-out statistical protocol.

### 5. When does the interior pay?

Lead with the honest symmetric comparison and the robust continuous response. Show:

- \(D_2\) versus measured gain;
- model-conditioned trajectories;
- comparison to \(\tau\), entropy, \(n_{95}/L\), and RateQuant-style dispersion predictors;
- uncertainty and corner-set sensitivity;
- held-out model or leave-model-out results.

The all-oracle result should be a secondary upper bound.

### 6. Context moves models through the regime map

Show each model as a trajectory. Lead with the strongest within-model change. Separate:

- absolute-\(L\) trend;
- architecture effect;
- additional RoPE-cap residual.

The correct conclusion is not that every model crosses one universal boundary. It is that context systematically changes tier viability and can change the preferred policy.

### 7. What the theory predicts—and where it fails

Show predicted versus exact recomputed gain, residuals, value-term ablation, and failure cases. This is where the \(0.79--1.34\times\) result belongs. It demonstrates utility of the surrogate without treating it as exact.

### 8. Deployment implications

Report, rather than speculate about:

- symmetric lagged information;
- KV-head/GQA grouping cost;
- cascade recovery;
- slow route / fast allocation timescales.

If R8 lands, end this section with achieved router-on/off end-task results. If it does not, describe Sieve as a decision rule suggested by the map and do not make it a numbered contribution.

### 9. Related and concurrent work

Organize by allocation axis rather than a binary eviction/quantization taxonomy:

- token selection/output-aware eviction;
- token precision and hybrid pruning/quantization;
- head/layer allocation;
- channel/transform allocation;
- concurrent rate-distortion formulations.

A compact overlap table will make the differentiation much clearer than defensive prose.

### 10. Limitations and conclusion

State plainly:

- attention-output error is not end-task accuracy;
- current allocation is key-token only;
- the total-KV/system budget is not yet matched;
- the boundary is architecture-conditioned;
- the full diffuse side is less validated;
- throughput requires a packed kernel/system implementation.

These limitations do not invalidate the regime result. They define its scope.

---

## 10. Suggested abstract framing

The following is the narrative I would target; numerical details should be updated from the final symmetric/R6 tables.

> KV-cache compressors are usually evaluated as if a model has one preferred policy: retain every token at low precision or retain a subset at high precision. We study both as actions in a joint token-rate allocation problem and ask a different question: when is the mixed interior worth using? Under an attention-output distortion surrogate, finite-bit tiers become dominated when their absolute logit-noise cost exceeds the zero-rate cost, while nearly uniform token importance removes the benefit of unequal allocation. These conditions yield an architecture-conditioned regime map and a measurable tier-extinction statistic. Across six LLMs and contexts from 8k to 128k, evaluated by exact attention-output recomputation, this statistic strongly orders the gain of mixed allocation over the better endpoint, including under a symmetric lagged-information comparison. Context length systematically changes tier viability and can move the same model from allocation-friendly toward eviction-like behavior, while the crossing remains architecture dependent. The resulting picture explains when joint allocation is useful, when a simpler endpoint suffices, and why compression policies calibrated at one context need not transfer to another.

If R8 succeeds, add one final sentence with achieved end-task benefit on disjoint routing calibration. If R8 does not succeed, do not mention Sieve in the abstract.

Suggested contributions:

1. **A common action space and a regime question.** We independently formulate token-wise key quantization and eviction under attention-output distortion, and derive conditions under which the optimizer approaches uniform, mixed, or eviction-like allocations.
2. **A predictive, architecture-conditioned regime map.** A tier-extinction statistic strongly orders when the mixed interior improves over both endpoints across models and context lengths; the boundary is model-conditioned rather than universal.
3. **Context-dependent traversal and deployment timescales.** Context changes tier viability within a model, while measured decode experiments show a stable route but a rapidly stale token allocation.

This contribution list retains the joint foundation without claiming that the existence of a joint allocator is exclusively new.

---

## 11. Figure hierarchy

1. **Teaser:** one model's measured trajectory across context, plus the broader architecture-conditioned regime map. No projected accuracy.
2. **Primary evidence:** tier-extinction statistic versus symmetric exact gain, with uncertainty and a held-out/leave-model-out marker.
3. **Mechanism:** context \(\rightarrow\tau\)/absolute tier cost \(\rightarrow\) tier extinction, separating absolute length and RoPE residual.
4. **Prediction/robustness:** surrogate-predicted versus exact error; alternative predictors; oracle versus lagged versus cascade; GQA grouping.
5. **End task, only if measured:** router calibration and evaluation on disjoint prompts/tasks.

The current synthetic Gaussian panel can remain as a small explanatory schematic, but not as evidence for the empirical regime map. Both hard-coded projected panels should be removed.

---

## 12. Priority order for the manuscript

These are writing/evidence priorities, not proposals for a new design.

### Must fix before submission

1. Correct the exact eviction derivation and the zero-rate cost units.
2. Replace all oracle-labeled-as-symmetric figures and numbers with R3/R6 results.
3. Cite and distinguish DiffKV, CAOTE/ReST-KV, RDKV, RateQuant, and AATC.
4. Remove exclusive “first joint,” exact-equivalence, universal-boundary, and length-free-calibration claims.
5. Remove invented projected numeric panels.
6. Disclose key-only rate accounting, full value treatment, metadata omissions, and the actual GQA allocation unit.
7. Rename hindsight \(\max(\mathrm{gain},1)\) as an oracle portfolio until a prospective router is evaluated.
8. Reconcile 10,240 heads, 46,000 measurements, 24/25 configurations, and the independent statistical unit.

### Strongly desirable

1. Use continuous gain as primary and the 2x band as secondary.
2. Compare \(D_2\) to RateQuant-style heterogeneity and simple concentration baselines.
3. Report leave-one-model-out performance and model-cluster uncertainty.
4. Integrate R4--R7 and co-design results instead of leaving them as TBDs.
5. Add R8 only if calibration and evaluation are disjoint and the result is complete.

### Final novelty/effectiveness verdict

**Novelty:** The paper is potentially novel enough for ICLR if it is claimed as an architecture- and context-dependent characterization of when joint token allocation helps. It is not safely novel as “the first joint eviction/quantization formulation,” because archival DiffKV already combines the actions and concurrent RDKV overlaps closely with the rate-distortion derivation.

**Methodological effectiveness:** The exact-output measurement program is strong, and the symmetric result confirms that the main phenomenon survives deployable information. The current manuscript nevertheless overstates budget matching, routing, universality, and mathematical exactness.

**Practical effectiveness:** Not established yet. R8 end-task accuracy and an eventual packed/runtime implementation determine whether Sieve should be sold as an effective method. They are not necessary for a carefully scoped characterization paper, but they are necessary for the current method/system language.

**Recommended selling decision:** Keep the joint formulation at the front as the lens. Sell the paper on the answer that follows from it:

> **The important question is not whether eviction and quantization can be combined. It is when their combination has a useful interior—and that answer changes with context.**
