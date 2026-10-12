# Generalizing R7a and R7b to every model architecture (plan, 2026-10-12)

R7a measures the theory at the paper's scale: part 8's certificate, the certified read and Theorem M. R7b tests the
deployed single store against FP8. Both run on two GQA models, Llama-3.1-8B and Qwen3-30B-A3B. This note covers how
both extend to every architecture with a softmax KV cache:
- what is already architecture-independent;
- what each attention mechanism needs;
- the per-model inputs;
- the order of work;
- what the paper can claim at each stage.

It builds on two sources:
- `MODEL_FAMILY.md`: where the theorems hold, by mechanism, on 21 models; and which bits each kind of layer needs, on
  18 of them;
- `adapters_s1h5.py`: the certificate's view of each architecture, on CPU.

This is a plan. Nothing below is built unless it says so.

## 1. What already holds for every architecture
- **Theorems R and M are per layer.**
  - They hold for every input, for any softmax attention over a per-layer cache of rows: Theorem R is the certificate,
    Theorem M the exact error.
  - Three conditions are needed (`MODEL_FAMILY.md` §6–7, `CLAIMS_LEDGER.md` T8):
    - a subtractive-dithered tier;
    - read sets chosen without its coins;
    - a separate dither per layer.
  - Positional schemes, QK-norm, biases, gates and attention multipliers do not matter, because the scores stay bilinear
    in a stored row.
  - Sinks and sliding windows add only exact rows.
- **The single store (Lemma N, Theorem N; `theory.md` §13) is stated for GQA.**
  - It is defined per key row of one KV head, for any head dimension and number of heads.
  - Its vote is R6's question pass, which needs only softmax scores over the store's keys.
  - So it carries over to any cache of key rows.
  - The exception is MLA, whose one latent feeds both keys and values. It needs the coupled form (§4).
- **R6's statistics carry over in form:**
  - the paired bootstrap;
  - U against 1.5 × FP8 plus an additive term;
  - fp_noise;
  - the gates with Holm;
  - the reader.

  The additive term (5e-4 nats per token) was set on R6's two models. A new model's is fixed in advance from its own
  FP8 and fp_noise KL.
- **The calibration's methods carry over; its numbers do not.**
  - Sensitivities come from single-layer arms (`theory.md` §14).
  - The slack law maps each layer's score noise σ_w to its certificate's tightness: part 8 / error ≈ 1.7·e^{14σ_w}
    (`theory.md` §12; R² 0.99 on 160 layer-width cases at 32K).

So generalizing needs:
1. a per-architecture adapter at run time (the CPU one exists);
2. a per-model width map in place of "4+2/4 everywhere";
3. run-time support for the mechanisms the GPU chain lacks;
4. theory for the mechanisms outside the theorems.

## 2. Per mechanism: what R7a and R7b need

| mechanism | examples | theory | R7a (the probe) | R7b (store, vote, arms) | run-time engineering |
|---|---|---|---|---|---|
| full softmax: MHA, MQA, GQA; RoPE, QK-norm, biases, gates | Llama 3.x, Mistral, Qwen 1.5–3, GLM-4.5, MiniMax-M2, Granite-4.2 | verified | none | per-layer widths from the map (§3); the arms otherwise unchanged | per-layer widths exist (R7b's `wnest`). Its allocation searches values up to 6 bits; a few flat layers need 8 (Granite-4.2's L2, L3, L8) |
| learned sinks | GPT-OSS | verified: a sink is an exact fixed row with value 0 | add each head's sink logit to the fixed rows' log-sum-exp | the attention and the debias mask carry one extra logit column per head; the vote ignores it | **`sieve_compress_attention` ignores HF's `s_aux` without an error:** through `sievelib`, GPT-OSS runs without its sinks, even in the FP arm. Add the column, or refuse |
| local : global (sliding or chunked windows) | Gemma 2/3/4, GPT-OSS, Llama 4, Command A | verified on the global layers | global layers only; the windows are exact | store, vote and tail on global layers only; the windows stay exact and enter the memory count at their fixed size | `sievelib` puts every layer in the store, and its decode path refuses a layer whose cache is shorter than the context (a sliding layer). Select the global layers; leave local ones on exact attention |
| softcapped logits | Gemma 2 | **not covered:** tanh breaks the linear score error and the debias identity | refuse | refuse | `sievelib` already refuses |
| latent cache (MLA) | DeepSeek-V2/V3/R1, Kimi K2, Moonlight | verified in the coupled forms (Theorem R; Theorem M with coupling ≤ 0.1%) | per head, q′ = W_ukᵀq over the latent, with coupled key and value errors; R7a's GPU forms are GQA only: port `cert_s1h5`'s coupled form and add part 8's | store the latent as the nested base (the vote's) plus the dithered residual; the RoPE key part stays exact or gets its own store; the debias uses q′, and the value error is W_uv times the latent's | **the largest item:** HF's MLA modules cache the up-projected K and V. A custom module must cache, quantize and read the latent |
| hybrids: recurrent or linear layers plus softmax layers | tested: Granite-4.0-H, Qwen3.5/3.8, Kimi-Linear; untested: Qwen3-Next, Nemotron-H, Jamba | softmax layers verified | softmax layers only | softmax layers only; the recurrent state stays exact, and memory counts only the KV share | as local:global; map `layer_idx` to the softmax layers |
| sparse attention with a learned selector | DeepSeek-V3.2 (DSA), NSA, MoBA | covered if the selector reads a store other than the dithered tier (R1) | the selected rows are the read set; certify the rest only if the model attends to them | the store becomes a memory format: the selector's index keys exact or separately quantized, the selected rows fetched exact, no tail when unselected rows are not attended | the selector must not read the dithered tier |
| cross-layer KV sharing | Gemma 3n, YOCO, CLA | **needs an extension:** R2 fails, because a later layer reads the same dithered rows with a query that depends on their dither | not yet | re-dither per consuming layer (a memory cost), or a new theorem | §4 |
| non-softmax or differential attention | sigmoid attention, Differential Transformer | **needs new formulas** | — | — | §4 |
| MoE (attention unchanged) | Qwen3-30B-A3B, GPT-OSS, DeepSeek | no change | none | none | forward passes are not reproducible across processes: compared arms run in one process (already the rule) |

## 3. The per-model inputs that replace "4+2/4 everywhere"
For each model at its run length, from calibration prompts:
1. **Widths.** Theorem M's grid picks each layer's key and value bits (Corollary M-B; `layermap_s1h5.py --grid`).
   - At 1,600 tokens it found three families (`MODEL_FAMILY.md` §4):
     - mostly concentrated models do well at 6/4, except their first one to three layers (flat) and their noisy-score
       layers (8-bit keys; e.g. Qwen3-30B-A3B's L0, L1, L3, L8, L16);
     - flat, balanced or value-leaning models need 6-bit values: Gemma-3-12B, Gemma-4-31B, Granite-4.2,
       MiniMax-M2, Qwen3.5/3.8's attention layers;
     - GPT-OSS-20B, flat with noisy scores, needs 8-bit keys (8/4 in 10 of its 12 full layers).
   - In the single store, the base stays at 4 bits for the vote; the residual and value bits vary per layer. The grid
     must be rerun in the store's per-coordinate form (Theorem N), since 4+r keys lose about 0.1–0.3 bit against
     uniform 4+r.
2. **Certification widths.** σ_w predicts each layer's part-8 slack by the slack law. A layer whose slack exceeds about
   20× at the store's width gets 4+4 keys wherever the certified read runs.
   - At 32K: 14 of Qwen3-30B-A3B's 48 layers; none of Llama-3.1-8B's.
   - R7b reads the list off R7a's probe as a 6-bit ratio above 14, which is about 20× at 4+2 (`r7b/plan_r7b.md` §3).
3. **Sensitivities.** c_l from single-layer arms gives E1's τ shape. Per-layer key-only and value-only KLs give E3's
   allocation. The KL budget's level needs calibration per family (`findings/findings_klbridge.md` §6.3).

Item 1 needs one FP pass per calibration prompt: Theorem M's moments are exact, so there are no dither draws. Item 2
adds one dither draw.
- At 32K this is cheap on CPU. R7b's captures of Llama-3.1-8B and Qwen3-30B-A3B took 2–3 minutes each on one node, and
  part 8's depth pass over both took 9 minutes.
- At 128K, R7a's GPU probe measures the same quantities. A CPU capture there costs about 16× 32K's attention and 4× its
  view size.

Item 3 needs GPU arms at the run's length.

## 4. Theory still to do (`RESEARCH_PLAN.md` §3 EXT)
- **MLA's bit grid and single store:**
  - the coupled Corollary M-B (`layermap_s1h5.py` takes GQA views only; `mla_m_s1h5.py` has the coupled Theorem M);
  - Lemma N for a latent, whose base the vote reads through W_uk.
- **Cross-layer KV sharing.** R2 needs an adaptive-query bound: later layers' queries depend on the shared dither.
  `adaptive_s1h5.py` found no change in coverage empirically.
- **Selector-based sparse attention.** R1 holds when the selector's index keys are not dithered: a short corollary.
  Check it on DSA's open weights.
- **Softcapped logits.** tanh is 1-Lipschitz, so the certificate's score errors contract. The unbiased tail's
  correction, however, needs E e^{cap(s + δ)}. Derive it, or leave such models out.
- **Differential and non-softmax attention:** a joint second moment for two softmax maps over the same dithered keys.
- **The slack law across architectures.** It is fitted on two GQA models. Check it on sinks, Gemma's global layers and
  MLA from G0's captures. It is the cheap way to pre-register each certificate's tightness.

## 5. Order of work

| stage | what | where | needs | cost |
|---|---|---|---|---|
| **G0** (CPU, now) | `MODEL_FAMILY.md`'s map at 32K instead of 1,600 tokens (its §8 long-context check), on 5 calibration prompts per model. Per model: per-layer σ_w, Theorem M's grid in the store's form, part 8's slack at 6/8-bit keys (`nested_s1h5.py depth`), the slack law per architecture. The largest models (Llama-3.3-70B, GLM-4.5-Air, MiniMax-M2) at partial depth, as in §2 of that note. MLA after §4's coupled grid | `14h_theory`, on CPU nodes as R7b's depth job | the adapters; code: a 32K capture driver and the grid in the store's form. `depth` and the grid take GQA views today (sinks and windows reduce to them; MLA does not). No GPU result | 2–3 days of code, then about 1–3 node-days |
| **G1** (run time) | the GPU chain's adapter: global-layer selection (local:global, hybrids), the sink column, per-layer widths (R7b's `wnest`), refusals for unhandled kwargs (`s_aux` until supported), softcapping, cross-layer sharing and differential attention; a truncated-model option for CPU smokes | new `14j_r8_cross_architecture` (`RESEARCH_PLAN.md` §4 R8); `sievelib` stays frozen, wrapped as R7a and R7b wrap it | R7b's run side as the template. Nothing else for the design-agnostic parts; G0 (widths) and R7b's core read (the design to port) for the rest | 1–2 weeks |
| **G2** (R8) | Gemma-3-12B (local:global, flat) and GPT-OSS-20B (sinks, local:global, noisy, MoE) at 32K. R8's predictions, restated for the single store: Gemma 3 passes at 4+2/6 and fails at 4+2/4; GPT-OSS passes at 4+4/4 and fails at 4+2/6; each map's allocation passes at its predicted traffic. Also R7a's probe on both | `14j_r8_cross_architecture` | G0a (its two models), G1, R7b's core read, the user's approval of the restated predictions | about 20–40 GPU-h |
| **G3** (MLA) | the latent-cache module; the coupled probe; R7a and R7b on DeepSeek-V2-Lite or Moonlight-16B-A3B at 32K | `14j_r8_*` or its own folder | G1; §4's MLA items; G0c; R8's outcome (go or no-go) | 2–3 weeks; 15–25 GPU-h |
| **G4** (hybrids) | Granite-4.0-H or Qwen3.5: softmax layers only | as G2 | G1; G0's map of the model; R8's outcome | about 10 GPU-h |
| **G5** (theory) | the rest of §4 | `14h_theory` | the slack law across architectures needs G0's captures; the rest needs nothing | open |

The order and dependencies across every next step (R6, R7a, R7b, G0–G5, the paper) are in `RESEARCH_PLAN.md` §7.

`RESEARCH_PLAN.md` §4 states R8's predictions for the uniform tier (6/4, 6/6, 8/4); they predate D7. G2 restates them
for the deployed design. That restatement is an amendment for the user to approve before R8 is frozen.

Each family needs a CPU driver smoke, playing the role Llama-3.2-1B plays in R7a and R7b. The candidates are local
models truncated to a few layers:
- Gemma-3-12B (local:global);
- GPT-OSS-20B (sinks);
- DeepSeek-V2-Lite (MLA);
- Granite-4.0-H-Tiny (hybrid).

## 6. What the paper can claim at each stage

| after | claim |
|---|---|
| R7a, R7b | GQA (Llama-3.1-8B, Qwen3-30B-A3B): the certificate at scale; the single store non-inferior to FP8 at 0.32 of FP16's memory at rest and 0.45 during a question, per-row scalars included (`../14i_r7_theory_at_scale/r7b/theory_r7b.md` §2.6), confirmatory |
| G0 | validity on 21 models across every open mechanism, and the per-layer design point (widths, certification widths) at 32K; the slack law across architectures (CPU) |
| G2 (R8) | the formula picks the bits on models R6 never saw, end to end and pre-registered (T8's "picks the bits") |
| G3, G4 | MLA and hybrids end to end |
| G5 | theorems for cross-layer sharing and selector-based sparse attention |

Until G2, "every architecture" is a statement about validity: the certificate holds and Theorem M is exact. It is not
a statement about the end-to-end design point. That is `MODEL_FAMILY.md` §7's scoped claim.

## 7. Risks
- **MLA's engineering.** HF's implementations differ across versions, and a latent cache changes the memory accounting.
- **Flat-attention models.** They need 6-bit values, so the single store's traffic rises to about 0.45
  ((7/8)·12/32 + 1/8), and the margin over FP8 narrows.
- **Local:global.** The memory claim's denominator changes, because the windows are small and stay exact.
- **The slack law.** If it fails on sinks or MLA, G0's certification widths must come from part 8 itself (one draw per
  layer, as `nested_s1h5.py depth`), at more CPU cost.
- **Silent errors at run time.** HF passes mechanism-specific kwargs (`s_aux`, `sliding_window`) that an attention
  function can ignore. G1 refuses any it does not handle.
