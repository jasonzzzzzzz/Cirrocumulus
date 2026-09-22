# R9 — four more eviction baselines: Ada-KV, DropKV, OBCache, LaProx

**Implementation audit and run contract:** `audit.md`. It records the baseline-to-paper judgments, the corrected R8 observation window and budget checks, and what still needs a main-model run.

Today every comparison is against **H2O** (`evict_h2o`, and the H0 `accum` corner)
and **SnapKV** (`evict`, and the H0 `window` corner). This adds four published
methods to the same harness, faithfully, so each can be run as an arm next to
the existing ones:

| paper | what it changes vs SnapKV | venue |
|---|---|---|
| Ada-KV (Feng et al., arXiv 2407.11550v5) | **budget**: keep-counts per KV head, adaptive within a layer | NeurIPS 2025 |
| DropKV (Zhang et al.) | **score**: exact single-token output perturbation, value-aware | ICML 2026 |
| OBCache (Gu et al., arXiv 2510.07651) | **score**: OBD second-order pruning error, value/key/joint | ICML 2026 |
| LaProx (Mai & Kim, arXiv 2605.07234) | **score** ‖A[:,i]‖·‖v_i W_O‖ **and budget** model-wide across layers and heads | preprint 2026 |

Sources read: the four PDFs in `docs/`, plus the official code where it exists:
`FFY0/AdaKV` (`adaptive_snapkv/monkeypatch/snapkv_utils.py`),
`DreamSoul-AI/OBCache` (`kvpress/presses/obcache_press.py`), and the DropKV
first author's kvpress PR (NVIDIA/kvpress#285, `dropkv_press.py`). LaProx has no
public code.

PDF path:
(1) /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/docs/ada-kv.pdf
(2) /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/docs/dropkv.pdf
(3) /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/docs/laprox.pdf
(4) /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/docs/obcache.pdf

Note that now we omit RDKV, which will be discussed later. 

---

## 1. Where they go: R8 prefill eviction, not the H0 corner

All four papers are **prefill-time, question-aware, one-shot** eviction: score
the context with the last `w` prompt queries (the observation window), keep a
budget, decode from what survives. That is exactly R8's `evict` / `evict_h2o`
path (`compress.py` + `router.py` + `run_r8.py`): full-precision prefill, the
protected last-W window, one allocation at decode start, end-task accuracy
scored. They land there.

The H0 corner (`evict.py`, `run_h0`) is a different object: a **per-query-head,
decode-time, lagged** score fed one attention vector per step, with a per-head
corner budget. Two of the four do not fit it at all. Ada-KV's contribution is the
split of a *layer's* budget across heads, and LaProx's is a *model-wide* split
across layers. A per-head corner with a fixed per-head budget cannot express
either. DropKV and OBCache could be adapted as decode-time corners, but that
would be our adaptation, not their method. Porting it changes `evict.py`'s
`observe(a, fin)` contract (they need V and the output o), so it needs the
user's OK (outside the R8 exception). **§7 proposes it as a separate phase.**

## 2. Design: score × allocator × select

Every eviction baseline, old and new, factors into three pieces:

```
score      per layer: [Hkv, C] importance of each context token, per KV head
allocator  keep-count per (layer, KV head), summing to the arm's budget
select     per (layer, KV head): top-count tokens by score -> maxb bits, rest 0
```

| arm (preset) | score | allocator |
|---|---|---|
| `evict` (SnapKV, unchanged) | window vote, group-summed, max-pool 7 | uniform |
| `evict_h2o` (H2O, unchanged) | all-prefill-query sum | uniform |
| `snapkv` | same as `evict`, but options can be set (e.g. obs=8, pool_k=11 as in DropKV's setup) | uniform |
| `adakv` | SnapKV's (identical to `evict`) | `ada` (α = 0.2) |
| `dropkv` | Σ_t (p/(1−p+ε))²‖a_t−v_j‖², max-pool 11, obs 8 | uniform |
| `obcache_v` / `obcache_k` / `obcache_vk` | OBD value / key / joint, max-pool 7, obs 16 | uniform |
| `laprox` | ‖A[:,i]‖₂·‖v_i W_Oʰ‖₂, avg-pool 7, obs 32 | `global` (layer-normalised, model-wide) |

Because the three pieces are independent, every preset takes `:key=value`
overrides. Ablations that the papers themselves run then come for free, e.g.
`obcache_k:alloc=ada` (the OBCache README's headline "AdaKV + OBCache-K"),
`laprox:alloc=layer` (LaProx Table 5, head-flatten only) and
`laprox:alloc=uniform`. Syntax follows `evict.make`:
`name[:k=v[:k=v...]][@label]`. The label becomes the `arm` value in the parquet,
and it needs no commas or quotes, so it passes through `--arms a,b,c` unchanged.

**Budget.** Every arm keeps exactly `Σ counts = n_layers · Hkv · keep_count(B, C, maxb)`
tokens at `maxb`, so every arm spends the same bits in total. `uniform` and `ada`
also match per layer, and `global` matches only in total (that is LaProx's
point). `bits_audit` checks the total, as it already does.

**The harness conventions that apply to every arm equally** stay unchanged and
are recorded in the sidecar. Each paper protects its own last-`w` tokens; here the
last W = 32 prompt tokens are protected at full precision in every arm, and only
the C context tokens before them are candidates. A kept token is stored at `maxb`
bits (8), not fp16. That is the same for all arms and is what makes
`keep = B·C/maxb` budget-matched with quantization.

## 3. Each method, exactly as implemented, and every judgment call

Common: window queries `qwin[li]` (post-RoPE, already captured in prefill), the
cache's full-precision K/V. For window query t at position C+t, softmax is taken
over **context + window keys, causal** (as in every paper: attention over the
whole cache). Scores are float32; they are used only for ranking.

### 3.1 Ada-KV
* Score = SnapKV's: Σ over window queries and the KV group, then max-pool k=7 over
  context positions (official order: mean over window → GQA mean → pool; mean vs
  sum is one constant for every head, so ranking and the cross-head flatten are
  identical). **Bit-identical to `evict`'s ranking**, which is pinned by a test.
* Algorithm 1: flatten the layer's [Hkv, C] scores, take the top `Hkv·k`, and
  count per head `f_g`.
* Safeguard: `B_g = (1−α)·f_g + α·k`, α = 0.2.
  * **Paper vs code discrepancy, resolved to the code.** Algorithm 2 line 8
    prints `α·B* + (1−α)·B/h`, but the paper's own α-robustness appendix says
    "a smaller α allows more aggressive allocation", which only holds if α is
    the *uniform* share. The official code agrees:
    `round(f·(1−floor_alpha) + floor_capacity)`.
* Rounding: the official `round` can drift the total by up to Hkv/2 tokens. We use
  largest-remainder rounding, so the total is exactly `Hkv·k` and the arm stays
  budget-matched. That differs from the code by at most one token per head.
* Not ported: `skip` / `normalize` / `pyram_mode` (code-only knobs, not in the
  paper's method; Pyramid is a different baseline).

### 3.2 DropKV
* Paper Algorithm 1: `score_j = Σ_{t∈last w} (p_tj/(1−p_tj+ε))² ‖a_t − v_j‖²`, where
  `a_t = p_t V`. Then max-pool κ over the full length, and keep the largest (evict
  the smallest).
* Computed without the w×C×d tensor via ‖a−v‖² = ‖a‖²+‖v‖²−2⟨a,v⟩ (the paper's
  own Triton identity).
* Defaults are the paper's: **w = 8, max-pool κ = 11**, ε = 1e-6 (Kernel B).
  * The authors' kvpress PR uses avg-pool, w = 32, κ = 7, which are kvpress's
    SnapKV defaults. `pool=avg:obs=32:pool_k=7` reproduces it.
* GQA: the paper is silent. We follow the authors' PR and **average over the
  group's query heads**.
* Pool over context + window, then slice to the context (the PR pools the full
  length, then overwrites the window).

### 3.3 OBCache
* Eq. 4–6 over the last `w` window queries with logits Z = q·k/√d, A = softmax(Z),
  o_t = A_t V (full cache):
  * value: Σ_t A²‖v‖²
  * key: Σ_t (A·Z)²‖v−o_t‖²
  * joint: value + key + 2 Σ_t A²·Z·(‖v‖² − ⟨v,o_t⟩)
* Integrated into SnapKV as the paper's main prefill setting (App. C.2.1): **w = 16,
  max-pool k = 7** over context positions, window protected.
* GQA: **App. B.5, Eq. 33–35: sum of per-query-head scores over the group**
  (`gqa=sum`).
  * The official repo defaults to `pre_redc` (average A and Z over the group,
    then score once), which is not what the paper derives. `gqa=pre` reproduces
    it, and `gqa=mean|max` give the code's post-reduction modes.
  * The paper's experiments kept all query heads' caches separately, which a GQA
    cache cannot do (one K row per KV head). B.5 is the paper's own answer to that.
* The official repo's defaults (w = 64, avg-pool 5) differ from the paper's
  (w = 16, max-pool 7). We use the paper's; both are options.
* Three presets, one per score. OBCache-K is the one the authors' scripts run as
  their headline.

### 3.4 LaProx
* Algorithm 1: per query head, `p_i = ‖A[:, i]‖₂ · ‖v_i W_Oʰ‖₂` with A the last
  w = 32 window queries' attention. ‖v W_Oʰ‖ is computed as √(vᵀ G_h v) with
  `G_h = W_Oʰᵀ W_Oʰ` (dh×dh, computed once per layer from `o_proj.weight`), so V W_O
  is never materialised.
* GQA: App. A says "mean attention weight within each query group". Since W_Oʰ is
  per query head, we compute the score per query head and average over the group.
  This is **judgment call 1 of 2**, forced by the paper not specifying it.
* Pooling: App. A says every non-SLLM method uses "average pooling with a kernel
  size of 7". We apply it to the final score (it is linear, so the order relative
  to the group mean does not matter). This is **judgment call 2 of 2**.
* Algorithm 2: flatten per layer, normalise `s = p / Σ_layer p`, then take the top-K
  across **all layers** with K = n_layers·Hkv·k. Implemented as a pre-pass that
  derives per-(layer, head) counts. Global top-K = per-(layer, head) top-count, so
  the selection is identical.
* `alloc=layer` (no cross-layer step) and `alloc=uniform` are the paper's Table 5 / 6
  ablations. `norm=0` gives its "raw layer-flatten" ablation.
* Needs `o_proj` in `model.model.layers[i].self_attn`. It refuses loudly on any
  other layout.

## 4. Code changes

New files:
* **`sievelib/baselines.py`**: the scorers, allocators, `select`, the preset
  registry and the spec parser. Pure tensor code over a `LayerCtx`; no model
  access other than the W_O Gram helper.
* **`tests/test_baselines.py`**: §5.

Shared files: all changes are **additive and default-off** (R8 exception; a
queued P0/P2 job behaves identically):
* `router.py`:
  * `LayerCtx` gains `qwin` and `wo_gram` (default None).
  * `build_layer_ctx` gains `need_noise=True` and `need_quant=True`, so a
    baseline-only run can skip the noise model and the 7-width quantization it
    never reads. Both default to today's behaviour.
* `run_r8.py`:
  * Baseline arms route through the P2 precompute path, with a model-wide
    pre-pass when an arm's allocator is `global`.
  * W_O Gram matrices are built only if some arm needs them.
  * The sidecar gets a `baselines` block holding each arm's full effective config
    (`config_record`).
  * `evict` / `evict_h2o` / P2 arms are untouched.
* `bugs/8_router_endtask/read_r8.py`: preset names are appended to `ARM_ORDER` so
  the tables show them. This is a no-op on every existing parquet.

Flagged, not changed: `evict.py`, `run_h0.py`, `report.py`, `alloc.py` and every
`submit_*.slurm`.

## 5. Tests: the correctness anchors

1. `adakv` scores == `evict`'s pooled vote, and `snapkv` preset bits == `router.allocate("evict")`, bit for bit.
2. Ada-KV: α = 0 is pure flattened top-T, and α = 1 is uniform. Counts are brute-forced on a toy. The sum is exactly Hkv·k, and no head gets more than C.
3. DropKV **is exact**: with w = 1 and pool off, `score_j` equals ‖a_{−j} − a‖², computed by actually deleting token j and recomputing the attention (Lemma 1).
4. OBCache **is its paper**: on a toy head, the value/key/joint scores equal ½·vᵀ[H]v terms computed by `torch.autograd` Hessians of the pruning error L(V̂, K̂) (Eq. 2–3). The GQA sum equals the sum of per-head scores.
5. LaProx: the Gram form equals the explicit ‖v W_Oʰ‖, and the global allocator equals a brute-force flatten → normalise → top-K over all layers.
6. Budget: every preset spends B bits/token in total, for B ∈ 1..4 and three context lengths.
7. End to end: a tiny model generates through each arm (skipped with `--fast`).
8. `tests/test_r8.py` still passes unchanged.

## 6. How to run

```
python h0_measurement/run_r8.py --model llama31-8b --ctx 32768 \
   --arms fp,uniform,evict,evict_h2o,adakv,dropkv,obcache_k,laprox --budgets 1,2,3,4 ...
# paper ablations / code-default variants:
   --arms obcache_k:alloc=ada@obck_ada,laprox:alloc=layer@laprox_layer,dropkv:pool=avg:obs=32:pool_k=7@dropkv_pr
```

## 7. Proposed, NOT done: decode-time H0 corners (needs an OK)

The H0 corner could gain lagged DropKV/OBCache scores (`observe(a, fin, v, o)`)
and an Ada-style cross-head split of the per-head corner budget. It changes
`evict.py`'s contract, the RAM model (`state_bytes_per_slot`) and alloc.py's
per-head corner loop, and it would be our adaptation (the papers are
prefill-time). Recommend deciding after R9 shows whether any of them moves R8's
end-task numbers.

## 8. Status: built and CPU-validated (2026-09-21)

* `tests/test_baselines.py`: all anchors pass.
  * DropKV matches brute-force deletion to rel 1e-6.
  * OBCache Eq. 4/5/6 match autograd Hessians to rel 1e-6.
  * LaProx matches an explicit V·W_O and a brute-force model-wide top-K.
  * Ada-KV counts match a brute-force count.
  * The `snapkv` preset equals `evict` bit for bit.
  * On Llama-3.2-1B, every preset is budget-matched, and every arm at B = maxb
    answers token-for-token like fp.
* `tests/test_r8.py` still passes unchanged.
* CLI smoke run: qwen3-1.7b @2k on CPU, 1 prompt, niah_single, B = 2, with
  `--head-error`. All arms ran and scored 1.00, the sidecar carries the full
  `baselines` config, and `read_r8.py` tabulates them. This checks the plumbing
  only; one easy prompt separates nothing.
* Run tests with `OMP_NUM_THREADS=8`. This node caps CPU time at 3600 s, and
  192 torch threads exhaust it in seconds.
