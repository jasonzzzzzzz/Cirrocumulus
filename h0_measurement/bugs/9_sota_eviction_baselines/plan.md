# R9 — four more eviction baselines: Ada-KV, DropKV, OBCache, LaProx

**Implementation audit and run contract:** `audit.md`. **Main-model results:**
`report.md`. All five evaluation cells completed on 2026-09-22.

**Next iteration status (2026-09-24):** Steps 1 and 2 are complete. Held-out
job 980414 initially confirmed Llama 32K/B=2 multikey at k16/v4/h4, but Step 3
development job 981481 then found uniform at 20/20, leaving zero candidate
envelope headroom. V2-A jobs 982121 (k24) and 982122 (k32) completed and
authenticated: k24 is ceilinged at 1.000 and k32 reaches 0.825, so neither
passes the frozen [0.50, 0.75] uniform band. The preregistered ``both too easy''
k40 branch completed as job 982613 with uniform 0.925, and the single final
k48 bracket completed as job 982702 with uniform 0.850. K48 fails the frozen
uniform band and half-block gate, so the bounded key-count search is complete
with no selected operating point. V2-B is blocked and must not be submitted.
V3's separately versioned contrastive multikey panel passed implementation,
production-tokenizer, corpus, and provenance checks, but qualification job
983199 returned ``stop_panel``: FP is 0.981, uniform is 0.850, and the first
uniform half is 0.863. The mean and half gates fail, so prompts 740--819 remain
untouched and panel policy development is prohibited. The synthetic operating
point search is closed. V4 then found a measurable natural LongBench-v2 cell,
but job 983715 stopped because two FP generations were censored by its frozen
128-token answer contract. Separately versioned V5 removed that censoring with
a forced-choice endpoint. Its development job 983888 passed competence but
returned `stop_no_opportunity`: all eight compressed policies scored 19/52 and
their itemwise oracle recovered only 3 additional rows. V5 is closed, its proxy
outcome is suppressed, and its 45-item confirmation split remains untouched.
V6 changed the architecture once to pinned Qwen3-30B-A3B at the same B=2.
Qualification job 984224 passed, but the authenticated development job 984370
returned `stop_v6_invalid_operating_point`: FP was 24/52 = 0.462, below the
frozen 0.50 floor. Uniform was the best fixed policy at 27/52 = 0.519, while
the itemwise compressed-policy oracle reached only 29/52. Thus V6 is closed,
G is suppressed, no confirmation lock exists, and the 45 confirmation rows
remain untouched.

V7 then preregistered one structured-query mechanism test on a fresh 128K
partition. Its authenticated qualification job 984886 returned
`stop_v7_qualification`: FP scored 9/20 = 0.450 (bootstrap q05 0.250) and
uniform scored 7/20 = 0.350 (q05 0.200). Uniform passed only its point-range
gate; FP and both strict lower-bound gates failed. Qualification ran only FP
and uniform, so it did not exercise the structured-query scorer. No V7
development ledger, development job, confirmation lock, or confirmation job
exists. V7 is closed without an outcome-guided rerun.

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

## 8. Status: main-model comparison complete (2026-09-22)

The implementation checks below still hold. Jobs 978479--978489 completed all
five evaluation cells at Llama 8K/32K/128K and Qwen 8K/32K. See `report.md` for
the validity exclusions and results. In brief, OBCache-K + Ada-KV and LaProx are
the strongest new eviction arms, but uniform quantization wins 34/36 valid
task/budget cells. The grid is ceiling limited and
must not be presented as satisfying R8's P0 budget gate. Oracle diagnostic
979308 then showed both failures: fixed calibration loses +0.298 accuracy of
prompt-specific headroom, while the output-error oracle still trails uniform by
0.201 overall despite much lower measured output error.


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


## 9. Next iteration: task provenance, non-ceiling screen, and policy diagnostics

### Step 1 — task-difficulty interface (implemented 2026-09-23)

The public controls are `--n-keys`, `--n-values`, and `--n-hops`, with historical
defaults 4/4/4. Slurm exposes the matching `R8_N_KEYS`, `R8_N_VALUES`, and
`R8_N_HOPS` variables. Counts must be positive integers and are validated before
the model loads.

The exact tuple is now present in:

- every accuracy row and per-head-error row;
- the result sidecar and route metadata;
- the first job-log line;
- nondefault result/head filenames (`..._kN_vN_hN`);
- route compatibility checks and R8/R9 completion guards; and
- reader grouping, deduplication, CSV rows, and printed section labels.

Legacy rows and routes with no task metadata mean exactly k4/v4/h4. They remain
compatible with default runs and are rejected for every harder configuration.
The runner now fails when a generated prompt exceeds the context instead of
writing a partial parquet. The fixed `prompt_block=[0,9]` route expectation was
removed; route calibration and evaluation must still have disjoint prompt
blocks, and every other route field remains exact.

The initial hard screen exposed a second provenance dimension: incomplete VT
answers at h8 all stopped at the old 64-token generation cap. The interface now
uses generation-limit contract `difficulty_v1`, preserves every k4/v4/h4 limit,
and adds 8 tokens per value and 16 per VT hop above four. Rows record
`max_new_tokens` and `reached_max_new`; sidecars record the version and per-task
limits. The reader reconstructs legacy limits, reports cap rates, rejects mixed
limits, and invalidates any incomplete FP answer that reaches its cap.

Verification before a GPU submission:

- Python compilation passed for the runner, task generator, reader, and tests;
- `bash -n` passed for the worker and all three R8/R9 drivers;
- `OMP_NUM_THREADS=8 .venv/bin/python tests/test_r8.py --fast` passed, including
  legacy-default, filename, positive-count, route-config, and overlap checks;
- the updated reader reproduced job 979308 while labeling it k4/v4/h4; and
- the Step 2 dry run prints all three task controls for both jobs.

### Step 2 — find a non-ceiling operating point (completed 2026-09-23)

The old k4/v4/h4 result is the easy lower bracket: at Llama 32K/B=2, uniform is
1.00 on multikey, 0.95 on multivalue, and 0.90 on variable tracking. Screen these
two configurations independently on the same development prompt block:

| level | n_keys | n_values | n_hops | prompts | arms | tasks |
|---|---:|---:|---:|---|---|---|
| easy-hard | 8 | 6 | 6 | 400--409 | FP, uniform | multikey, multivalue, VT |
| hard | 16 | 8 | 8 | 400--409 | FP, uniform | multikey, multivalue, VT |

**Completed 2026-09-23:** job 980284 is k8/v6/h6 and job 980285 is
k16/v8/h8. Both passed the row/provenance/budget gates. Multikey at k16 passes
(FP 1.00, uniform 0.80). Multivalue at v8 genuinely fails its FP gate (the one
zero-score FP answer stopped after one token). VT's apparent FP failure is
censored: every incomplete FP answer reached the legacy 64-token limit, so the
h8 VT decision is void pending a cap-fixed rerun.

Both use Llama-3.1-8B, 32K, question-agnostic compression, and B=2. Single NIAH
is omitted because none of the knobs changes it. Each job must contain exactly
60 rows: 10 prompts x 3 tasks x 2 arms. Preflight tokenization on the real corpus
put every planned prompt below 32K. Tokenizing the expected answers below 64
tokens did not bound verbose model generations; the observed VT truncation is
why the generation contract and row-level cap audit were added.

Acceptance is task-specific:

1. no missing/skipped rows, one real-corpus SHA, and a passing bit audit;
2. FP mean at least 0.95;
3. uniform mean in [0.50, 0.80], or its 90% screening interval overlaps that
   band; and
4. choose `n_keys`, `n_values`, and `n_hops` independently from the smallest
   level meeting the gate.

The completed screen selects `n_keys=16` for multikey. For multivalue and VT,
level 6 is too easy while level 8 has FP below 0.95. Before looking at an
intermediate result, preregister the only missing integer point:

| follow-up | n_keys | n_values | n_hops | tasks | prompts | expected rows |
|---|---:|---:|---:|---|---|---:|
| legacy midpoint | 16 | 7 | 7 | multivalue, VT | 400--409 | 40 |

**Submitted 2026-09-23:** job 980342 used the legacy 64-token cap. Its
multivalue rows can diagnose direction, but any capped row and the complete VT
comparison are excluded. The following cap-fixed jobs are preregistered:

| rerun | configuration | tasks | limits | prompts | rows |
|---|---|---|---|---|---:|
| midpoint-v1 | k16/v7/h7 | multivalue, VT | 88 / 112 | 400--409 | 40 |
| vt-hard-v1 | k16/v7/h8 | VT | 128 | 400--409 | 20 |

**Submitted 2026-09-23:** midpoint-v1 is job 980356 and vt-hard-v1 is job
980355. These are development-screen jobs; their outcomes select the fixed
task configurations for one held-out confirmation on prompts 420--439.

**Development decision, 2026-09-23:** both jobs completed with their exact
generation contracts and row counts. Multivalue v7 has FP 0.986 and uniform
0.871, so it is too easy. VT h7 is invalid because its only incomplete FP
answer reaches the 112-token limit. VT h8 has FP 1.000 and uniform 0.967, so it
is valid but too easy. Capped incomplete uniform answers cannot reverse either
“too easy” decision because additional output can only add expected hits.
Consequently, multikey at `n_keys=16` is the only task selected by this
development block.

For multivalue v7: accept only if uncapped FP >=0.95 and uniform is in
[0.50,0.80] (or its screening interval overlaps); if FP passes and uniform is
above 0.80, counts 6--8 contain no usable B=2 point because v8 FP fails. For VT,
apply the same gate separately at h7 and h8 after verifying no incomplete FP
answer reaches its new cap. Do not run v10/h10 after multivalue FP has already
failed at 8. Confirm every selected task once on held-out prompts 420--439 before
a router/SOTA comparison.

The held-out confirmation is preregistered as follows. It uses the canonical
irrelevant knobs (`n_values=n_hops=4`) so the artifact states only the selected
multikey difficulty:

| confirmation | configuration | tasks | arms | budget | prompts | rows |
|---|---|---|---|---:|---|---:|
| multikey | k16/v4/h4 | multikey | FP, uniform | 2 | 420--439 | 40 |

It passes only if all 40 rows and the real-corpus provenance are present, the
bit audit passes, FP is at least 0.95 with no incomplete capped FP answer, and
uniform is in [0.50, 0.80]. If it fails, do not change `n_keys` using prompts
420--439; any redesign must return to a new development block.

**Completed 2026-09-23:** held-out job 980414 has all 40 rows and exact
provenance. FP is 1.000 (20/20), with no FP cap hit; uniform B=2 is 0.800
(16/20), with a 90% prompt-bootstrap interval [0.65, 0.95]. FP minus uniform is
+0.200 [0.05, 0.35] under the paired 90% bootstrap. One uniform answer reaches
24 tokens but already scores 1.0; all four uniform failures are uncapped. The
operating point passes at the prespecified upper boundary. Freeze k16/v4/h4 and
do not tune it on prompts 420--439.

The completed workflow and its reproducible commands are owned by `steps.sh`:

```bash
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --difficulty-dry
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --difficulty-submit
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --difficulty-status JOB [JOB...]
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --difficulty-read JOB [JOB...]
```

The post-screen midpoint is:

```bash
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --difficulty-dry mid
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --difficulty-submit mid
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --difficulty-dry vt-hard
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --difficulty-submit vt-hard
```

The frozen held-out confirmation was run and can be audited in this order:

```bash
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --confirm-dry
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --confirm-submit
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --confirm-status JOB
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --confirm-read JOB
```

### Step 3 — whole-policy headroom and logit-proxy diagnostic

**State (2026-09-23):** implementation and CPU contracts pass. Development
job **981481** completed and authenticated on prompts 440--459, but uniform was
1.00 on all 20 prompts, giving H=G=0 and decision `revise_candidates`. Because
an envelope containing a perfect uniform policy has algebraically zero
headroom, do not run expanded candidates on this block. Prompts 420--439 remain
the earlier difficulty-confirmation block and prompts 460--499 remain untouched
for a future locked policy confirmation.

#### 3A. Fresh diagnostic-development block

Use prompts 440--459 and the complete policies `fp`, `uniform`, `evict`,
and `interior`. The primary candidate set is exactly
`{uniform, evict, interior}`. There are 80 accuracy rows and 60 candidate
diagnostic rows. Do not include the calibrated router, the per-head output-error
oracle, or head-error routing in this job.

The two selectors are diagnostics over complete policies, not allocation arms:

- **End-task oracle:** retain every candidate tying for the maximum independently
  greedy-decoded RULER score on each prompt. This label-seeing envelope measures
  candidate-set opportunity and costs no extra GPU once the accuracy rows exist.
- **Policy-logit oracle:** replay up to the first eight FP greedy decisions under
  the identical FP teacher-forced prefix for every candidate. Stream exact
  float32 full-vocabulary KL(FP || candidate); never feed a candidate's argmax
  back into the trace. Select the lowest mean-KL complete policy, with the stable
  declared candidate order breaking exact numeric ties, then join that policy's
  independently greedy RULER score.

Raw logits are transient. A separate diagnostic parquet and sidecar store
mean/max KL, cross entropy on the FP-chosen token (not gold-answer NLL), top-1
agreement, trace length, selected policy, timing, candidate order, trace-rule
version, task configuration, prompt block, model/cache configuration, and the
joined accuracy-artifact identity.

Report, with paired prompt-bootstrap intervals:

- `H = mean(end-task envelope - uniform)`;
- `G = mean(KL-selected policy - uniform)`;
- regret `H-G` and captured opportunity `G/H` when `H>0`;
- uniform failures rescued, uniform successes harmed, selector counts, and
  whether the KL selection belongs to the end-task-optimal tie set.

The development decisions are fixed before viewing prompts 440--459:

- `H >= 0.10`: the primary candidates have useful complementarity;
- `H <= 0.05`: revise the candidates; an intermediate result needs more
  development prompts before a design decision;
- when `H >= 0.10`, `G >= 0.05` and `G/H >= 0.5` advance the mean-KL rule;
- `G <= 0` or `G/H < 0.25` with useful `H` is evidence against mean
  final-logit KL; intermediate proxy outcomes need more development prompts.

At 20 binary multikey prompts, scores move in 0.05 increments. These are
development gates, not final evidence.

#### 3B. Locked confirmation

If a candidate set and proxy rule advance, freeze both and run once on prompts
460--499 (40 prompts). Require `H >= 0.10`, `G >= 0.05`,
`G/H >= 0.5`, and a paired 90% lower bound for `G` of at least zero.
Do not tune on this block.

#### 3C. Conditional design branches

1. **Little end-task headroom.** On the development block only, add complete
   policies in nested sets: `interior_pool`, `interior_cascade`, plain
   `obcache_k`, `obcache_k:alloc=ada@obck_ada` (resolved label `obck_ada`), and
   `laprox`. Collect the full superset once on the same prompts, then evaluate
   the preregistered prefixes offline in exactly that order; this avoids five
   redundant GPU prefills. For every prefix, discard the full-set recorded
   selector and recompute its stable selection after filtering. Freeze the
   smallest prefix reaching `H >= 0.10` before confirmation. Whole-policy
   selection can use model-wide policies without pretending they are per-head
   splices. If the expanded envelope still lacks headroom, stop router training
   and redesign the allocations or retention objective.
2. **Headroom but mean KL fails.** Use only the already stored aggregates on
   prompts 440--459: minimize FP-token cross entropy, minimize maximum per-step
   KL, or maximize top-1 agreement. Each rule resolves exact numeric ties by the
   declared candidate order and may not use another metric as a hidden
   tie-breaker. Report the same G, regret, captured opportunity, paired interval,
   rescue/harm, oracle-hit, selector-count, and selector-tie statistics for all
   three rules using shared bootstrap draws. An alternate advances at G >= 0.05
   and G/H >= 0.5. If several pass, freeze the largest-G rule; an exact G tie uses
   this preregistered order: FP-token cross entropy, maximum KL, top-1 agreement.
   If none passes, replace the proxy with a task-relevant downstream-sensitivity
   objective. These three rules need no new GPU data; per-token weighting or a
   new composite does.
3. **Both diagnostics gain and confirm.** Train a deployable prefill-only
   selector on new disjoint data. It may imitate the KL selector but cannot use
   FP decode logits. Keep training, threshold selection, and final testing
   disjoint, and report regret to both oracle bounds.

Do not launch another model/context/SOTA grid before Step 3B passes.

#### 3D. Clean implementation boundary

Add a small pure `sievelib/policy_diagnostic.py` module for divergence metrics
and deterministic selection. Make `run_r8.py` opt in through explicit
`--policy-diagnostic-out`, `--policy-candidates`, and
`--policy-trace-steps` flags so it can reuse the same prefill and candidate
allocations. Write `r8policy*.parquet` plus its own sidecar; neither oracle
becomes an arm, a route, or code in `router.py`. A separate reader performs an
exact one-to-one join on model, context, task configuration, task, prompt,
budget, and candidate.

Tests must pin KL numerics, zero KL for FP versus itself, shared teacher-forced
token IDs, refusal to feed candidate argmax tokens, cache crop isolation,
deterministic ties, exact row/join provenance, absence of raw logits, and
completion counts.

#### 3E. Implementation and submission record (2026-09-23)

The implementation follows the boundary in 3D:

- `sievelib/policy_diagnostic.py` owns teacher-forced replay metrics and stable
  selection, with no task, route, allocation, or file logic;
- `run_r8.py` enables the diagnostic only through the three explicit policy
  flags, reuses complete precomputed arm allocations, crops to the identical
  context boundary before every replay, and writes a separate authenticated
  artifact with no raw logits;
- `read_policy.py` verifies both sidecars, the linked accuracy SHA-256, exact
  one-to-one candidate joins, shared trace identity, and declared selection
  before computing the two oracles; and
- `submit_r8.slurm` plus `steps.sh` provide opt-in Slurm plumbing, fixed
  development/confirmation splits, exact completion counts, and a confirmation
  gate tied to the development decision.

Before submission, Python compilation, shell syntax, the full fast R8 and R9
baseline suites, seven policy-diagnostic tests, and fifteen reader checks all
passed. The regression tests include exact full-vocabulary float32 KL, FP zero
KL, FP-only teacher forcing, deterministic ties, repeated crop isolation,
artifact identity, shared-trace provenance, and rejection of raw logits.

Development job **981481** was submitted from `trig-login01` with prompts
440--459, 20 prompts, four accuracy arms, three diagnostic candidates, B=2,
and an at-most eight-token FP trace. It completed in 5:24 with all 80 accuracy
rows and 60 diagnostic rows in `h0_measurement/results/r8policy_dev_981481/`.
The authenticated result is FP 1.00, uniform 1.00, eviction 0.25, interior
0.30, H=0, G=0, and `revise_candidates`. Mean KL selects uniform 18 times and
interior twice; every choice belongs to an end-task-optimal tie. No raw logits
were written and every FP replay argmax check passes.

Do not submit the locked confirmation or the expanded candidate set from 3C on
prompts 440--459. Since uniform is correct on all 20, adding candidates cannot
increase the end-task envelope on even one prompt. The next step must repair the
operating point on fresh data; the result and rationale are recorded in
`report.md`.


#### 3F. V2 operating-point repair after the ceilinged development block

**Frozen before viewing any prompts at 500 or above.** Job 981481 makes the
planned same-block candidate expansion uninformative: its uniform score is 1 on
every prompt, so adding policies cannot change H. The earlier k16 selection was
also made at the upper 0.80 boundary using only 10 prompts. Across all observed
k16 blocks, uniform is 44/50 = 0.88. Return to task selection on fresh data
without changing model, context, budget, task family, compression point, or
scorer.

##### V2-A: one full-corpus-cycle difficulty screen

Run two independent FP+uniform cells on prompts 500--539:

| cell | configuration | model/context | task/budget | prompts | rows |
|---|---|---|---|---|---:|
| op2-k24 | k24/v4/h4 | Llama-3.1-8B / 32K | QA multikey / B=2 | 500--539 | 80 |
| op2-k32 | k32/v4/h4 | Llama-3.1-8B / 32K | QA multikey / B=2 | 500--539 | 80 |

**Implementation record (2026-09-23, before submission).**
`sievelib/tasks_ruler.py` now records the zero-based queried-needle insertion
rank and depth without changing prompt bytes or RNG order; `run_r8.py` carries
these fields into every accuracy/policy/head row and versions them in sidecars.
`read_op2.py` requires the exact k24+k32 pair before reporting outcomes, checks
all 80 rows per cell, the full 40-document cycle and identical source offsets,
the queried needle against `needle_depths`, sidecar versions, task/budget/cap
provenance, and the gates below. It also reports `first_ok` without using it for
selection and enforces the k24 FP-stop rule. `steps.sh --op2-*` is the sole
submission/read interface. Focused reader tests, the full fast R8 suite, policy
diagnostic/reader tests, Python compilation, shell syntax, and both dry-run
commands pass. No prompt at or above 500 was scored during implementation.
The frozen workflow then submitted k24 as Slurm job **982121** and k32 as
**982122**; reporting remains blocked until both exact artifacts authenticate.

Forty consecutive prompt IDs cover one full 40-book corpus cycle. Authenticate
both artifacts before examining outcomes. A cell is eligible only if:

1. FP mean is at least 0.95 and no incomplete FP answer reaches its cap;
2. uniform mean is in [0.50, 0.75];
3. uniform is in [0.35, 0.85] separately on prompts 500--519 and 520--539;
4. the paired 90% prompt-bootstrap lower bound for FP minus uniform is greater
   than 0.05; and
5. row count, unique keys, real-corpus SHA, B=2 bits, question-agnostic flag,
   k/v/h configuration, and `difficulty_v1` generation contract all pass.

Choose the smallest eligible `n_keys`. Report standard RULER substring score as
the primary outcome and `first_ok` as a prespecified secondary safety outcome.
Do not pool prompts 400--459 into selection. If neither cell passes, do not run
a policy diagnostic: k24 too hard implies a fresh k20 bracket; both cells too
easy imply k40; k24 easy and k32 too hard imply k28; an FP failure at k24 stops
higher-key screening.

**Authenticated V2-A outcome (jobs 982121 and 982122).** Both artifacts pass
all row, corpus, target, cap, bit-budget, and sidecar-provenance checks. The
primary and secondary scores agree exactly:

| cell | FP | uniform | halves | paired 90% CI for FP-uniform | decision |
|---|---:|---:|---:|---:|---|
| k24 | 1.000 | 1.000 | 1.000 / 1.000 | [0.000, 0.000] | too easy |
| k32 | 1.000 | 0.825 | 0.800 / 0.850 | [0.075, 0.275] | too easy |

K32 passes the FP, cap, half-block, and positive-difference gates but fails the
frozen uniform upper bound of 0.75. Do not relax the bound after seeing 0.825.
There is no selected operating point, so V2-B is not allowed on prompts
540--579 under the original split.

##### V2-A2: bounded fresh k40 follow-up

**Frozen after authenticating V2-A and before viewing any outcome at prompt 540
or above.** Follow the already declared ``both cells too easy imply k40'' branch
with one independent k40/v4/h4 FP+uniform cell on prompts 540--579. Keep
Llama-3.1-8B, 32K, QA multikey, B=2, the question-agnostic compression point,
40 documents, answer contract, seed schedule, corpus, and all five V2-A gates
unchanged. Require exactly 80 rows. Authenticate and report this cell by itself;
do not pool the adaptive k24/k32 screen into its estimates.

The branch after k40 is bounded before observing it:

1. If k40 passes every V2-A gate, freeze k40 and move V2-B to prompts 580--619.
2. If FP, cap, corpus, task, budget, or provenance validity fails, stop this task
   search. A mean inside [0.50, 0.75] that fails a half-block or paired-CI gate
   also stops for instability.
3. If every non-band gate passes but uniform remains above 0.75, run one final
   fresh k48 bracket on prompts 580--619. If uniform falls below 0.50, run one
   final fresh k36 bracket on that block. Apply the identical gates. Freeze it
   only if it passes; otherwise stop. Do not try another key count or combine
   adaptive cells to manufacture a pass.
4. If the one final bracket selects k36 or k48, move V2-B to prompts 620--659.

This spends at most one additional bracket after k40 and preserves one full
fresh corpus cycle for policy development. It also makes the split consequence
explicit before any k40 outcome is known.

**V2-A2 implementation record (2026-09-23, before submission).**
`read_op2_k40.py` is a standalone strict reader for the single k40 cell. It
binds prompts 540--579, the exact model/task/configuration and corpus SHA,
40 distinct unspliced source documents, task and queried-needle versions, the
complete target-depth vector, P0 decode path, B=2 bits, caps, sidecar, the
10,000-draw seed-0 paired bootstrap, and all frozen gates. Valid ineligible
outcomes exit successfully; provenance errors do not. `steps.sh` exposes only
`--op2-k40-{dry,submit,status,read}` for this cell. Dry, submit, and read first
reauthenticate exact V2-A jobs 982121/982122, recompute and byte-compare their
canonical summary, and require no selection, both uniform means above 0.75, and
no k24 FP stop. The new focused reader suite, positive and negative workflow
guards, Python compilation, and shell syntax all pass. No prompt at or above
540 was viewed while implementing or testing this branch. The authenticated
workflow submitted this exact cell as Slurm job **982613** on the H100 debug
partition with a 12-minute walltime.

**Authenticated V2-A2 outcome (job 982613).** The strict reader accepts all 80
rows and their provenance. FP is 1.000, uniform is 0.925, the two uniform halves
are 0.950/0.900, FP-uniform is 0.075 with paired 90% interval [0.025, 0.150],
and no incomplete FP answer reaches the cap. `first_ok` equals the primary
score. The cell is directionally too easy in both halves and selects no
operating point.

**Branch clarification before viewing prompts 580 or above.** In branch item 3,
``non-band'' means the validity conditions enumerated in item 2: FP/cap,
corpus, task, budget, and provenance. A half-block or difference failure stops
when the mean is inside the target band because that indicates instability or
insufficient separation. When the mean and both halves are above their upper
bounds, as here, those failures point in the same prespecified ``too easy''
direction and cannot become eligibility evidence; the lower difference bound
also shrinks mechanically with only three uniform failures. Therefore the
already named final k48 branch applies. Run exactly one k48/v4/h4 FP+uniform
cell on prompts 580--619 under the identical contract. It must independently
pass all original V2-A gates. If it fails for any reason, stop the operating
point search; there is no k44 interpolation, pooling, or threshold change.
No outcome at prompt 580 or above was viewed when making this clarification.

**V2-A3 implementation record (2026-09-23, before submission).**
`read_op2_k48.py` independently binds k48/v4/h4 and prompts 580--619 to the
same strict 80-row, corpus, target, P0, cap, bit, sidecar, score, and bootstrap
contract. `steps.sh --op2-k48-{dry,submit,status,read}` first authenticates exact
k40 job 982613, recomputes and byte-compares its canonical summary, and requires
FP/cap validity, no selection, uniform above 0.75, and both halves above 0.85.
The job command is fixed to the H100 debug partition with a 12-minute walltime.
The 38-check reader suite, k40 regression suite, Python compilation, shell
syntax, guarded dry run, and an independent workflow audit pass. No prompt at
or above 580 was scored during implementation or testing. The guarded workflow
then submitted this exact cell as Slurm job **982702**.

**Authenticated V2-A3 outcome (job 982702).** All 80 rows and their strict
provenance pass. FP is 1.000, uniform is 0.850, the two uniform halves are
0.900/0.800, FP-uniform is 0.150 with paired 90% interval [0.075, 0.250], and
no incomplete FP answer reaches the cap. `first_ok` again equals the primary
score. K48 passes FP/cap and the paired-difference gate, but fails the frozen
[0.50, 0.75] uniform band and the first half's 0.85 upper bound. It is not
eligible. Per the pre-outcome terminal rule, stop: do not try k44/k56/k64, pool
adaptive blocks, relax a boundary, or run V2-B on prompts 620--659.

##### V2-B: separate expanded-policy development

**State after V2-A3: blocked.** No operating point passed every gate. The
workflow below remains a design specification only and no expanded-policy job
is authorized from the completed screen.

Only after V2-A/V2-A2/V2-A3 freezes `n_keys`, use the development block assigned by
the branch above: prompts 580--619 after a direct k40 selection, or prompts
620--659 after a final k36/k48 bracket. Collect the complete ordered superset
once:

`uniform, evict, interior, interior_pool, interior_cascade, obcache_k, obck_ada, laprox`

The raw arm for resolved label `obck_ada` is
`obcache_k:alloc=ada@obck_ada`. Include one FP row per prompt. The full artifact
has 360 independently decoded accuracy rows and 320 diagnostic rows. Before any
prefix analysis require FP >= 0.95, no incomplete capped FP, uniform in
[0.50, 0.80], and all artifact/provenance/budget checks. If this independent
block misses the operating-point gate, declare the difficulty non-replicating;
do not interpret H/G or tune `n_keys` on prompts 540--579.

Analyze the authenticated artifact as fixed nested prefixes of size 3 through
8 in the order above. Filter both frames, discard the full-set recorded
selection, and recompute each selector with stable candidate-order ties. For
each prefix, freeze `best_fixed_dev` as the candidate with the largest mean
independently greedy development score, using candidate order for an exact tie.
Retain the original uniform-relative H/G for continuity, but use routing-value
metrics relative to that strongest fixed policy:

- `H_F = mean(end-task envelope - best_fixed_dev)`;
- `G_F = mean(selector - best_fixed_dev)`.

Freeze the smallest prefix with H_F >= 0.10, a paired 90% lower bound for H_F
above zero, and H_F >= 0.05 in each fixed 20-prompt half. Mean KL advances only
with G_F >= 0.05, G_F/H_F >= 0.5, and G_F >= 0 in both halves. If it formally
fails, apply the preregistered aggregate alternatives from 3C to that same
prefix. If the full prefix has H_F <= 0.05, stop candidate routing. An
intermediate result may use one fresh 40-prompt extension immediately after its
assigned development block (620--659 on the direct-k40 path or 660--699 after a
final bracket), followed by a single pooled n=80 decision; no other repeated
extensions are allowed.

Write a lock manifest containing the selected task configuration, ordered
candidate prefix, `best_fixed_dev`, selector metric/direction, trace rule and
length, development artifact paths/SHA-256, and the locked confirmation block.

##### V2-C: locked confirmation remains untouched

Prompts 460--499 remain unopened and reserved. Run exactly the frozen manifest.
For prefix size m, require 40(m+1) accuracy rows and 40m diagnostic rows. The
primary gates are FP >= 0.95 with no incomplete capped FP, H_F >= 0.10,
G_F >= 0.05, G_F/H_F >= 0.5, and a paired 90% lower bound for G_F of at least
zero. Keep `best_fixed_dev` fixed; do not reselect the comparator on
confirmation. A failure ends this selector iteration with no tuning on these
prompts.

#### 3G. V3: contrastive multikey query panel

**State (2026-09-23): qualification complete; ``stop_panel`` is
binding.** Job 983199 authenticated all 320 rows but failed the frozen uniform
mean and first-half gates. No policy-development or confirmation job is
authorized. V2 remains terminal; V3 was a new task contract rather than another
key-count bracket. It was motivated by the observed failure mechanism:
14 of the 16 uniform failures across k32/k40/k48 retrieve a registered
distractor, while target rank and depth do not distinguish failures.

##### 3G.1 Task contract

Keep the registered task as `niah_multikey` and add the explicit
`task_variant=multikey_panel_v1` contract without changing legacy RULER prompt
bytes or metadata. Each prompt has one 32K context and exactly 48 needles:
four contrastive clusters of 12 keys. Keys in one cluster share a fixed literal
prefix and differ by one tokenizer-validated suffix token. Each cluster has one
target and 11 registered distractors with unique seven-digit values.

Ask four questions independently from the same context cache and the same
compressed allocation, one target per cluster. Target depths are fixed at
0.15, 0.38, 0.62, and 0.85; rotate the cluster-to-depth assignment by
`prompt_idx mod 4`. Use separate named RNG streams for haystack selection,
target identities/values, distractors, and placement. Extending a distractor
pool must not change any target key, value, depth, or question. Record
`task_variant`, panel version, `query_idx`, cluster, target key/value/rank/depth,
distractor identities, and RNG versions in rows and sidecars.

The primary query outcome is binary `first_ok`. The primary prompt outcome is
the mean of its four query outcomes, in {0, 0.25, 0.5, 0.75, 1}. Retain RULER
substring match, distractor retrieval, generation length, and cap state as
query-level diagnostics. All intervals resample 40 contexts and carry all four
queries together; never treat 160 correlated query rows as independent.

##### 3G.2 Frozen splits and qualification

| phase | prompts | arms/policies | expected accuracy rows |
|---|---|---|---:|
| qualification | 700--739 | FP, uniform B=2 | 320 |
| policy development | 740--779 | FP + the frozen eight candidates | 1,440 |
| locked confirmation | 780--819 | exact locked prefix | 40 x 4 x (m+1) |

Qualification must pass every condition before policy development is submitted:

1. exact panel/task/corpus/cache/bit provenance, 40 distinct real documents,
   four unique query slots per prompt/arm, and one shared allocation identity;
2. FP micro accuracy >= 0.95, FP accuracy >= 0.90 in every query slot, and no
   incomplete capped FP query;
3. uniform mean prompt score in [0.50, 0.75] and in [0.35, 0.85] separately on
   prompts 700--719 and 720--739;
4. uniform query-slot accuracy spread `max(slot)-min(slot) <= 0.25`; and
5. the paired 10,000-draw seed-0 context-bootstrap lower 90% bound for
   FP-minus-uniform is greater than 0.05.

If qualification fails, stop. Do not change the cluster size, key template,
depths, budget, prompt split, or score, and do not filter prompts. B=1 and any
further random-k bracket remain prohibited.

Policy development uses the ordered candidates
`uniform, evict, interior, interior_pool, interior_cascade, obcache_k,
obck_ada, laprox`. It must independently repeat the qualification operating
gates on prompts 740--779 before any H/G quantity is read. Aggregate each
complete policy's four independently decoded query scores to one prompt score,
then apply the existing nested-prefix, `best_fixed_dev`, H_F/G_F, selector,
lock-manifest, and confirmation rules. Confirmation on prompts 780--819 keeps
the comparator and selector fixed and bootstraps contexts.

##### 3G.3 Implementation boundary

- `tasks_ruler.py` owns a panel builder returning one shared context plus four
  immutable query records; the legacy `build()` path stays byte-identical.
- `run_r8.py` prefills and allocates once per context, then restores the exact
  context boundary and reapplies identical bits for each independent question.
  `query_idx` becomes part of every accuracy and whole-policy diagnostic key.
  Per-head errors and router arms are rejected in panel mode because their
  measurements depend on the question and cannot share the frozen allocation.
- A separate strict reader authenticates the panel and aggregates query rows to
  prompt scores. A separate workflow gate prevents policy submission unless the
  exact qualification artifact passes.
- Tests pin byte determinism, one-token suffix contrast, target invariance,
  fixed depth strata, shared allocation identity, cache isolation across
  questions, row counts, context-clustered bootstrap, and provenance rejection.

This interface reduces binary prompt variance and controls the associative
confusion seen in V2. It is a stress-test task, so any eventual paper claim must
name that scope; success here alone would not restore a broad SIEVE claim.

##### 3G.4 Implemented qualification contract and preflight

The implementation uses `build_multikey_panel()` for one question-free context
and four immutable query records. The runner hashes the UTF-8 context and every
actual per-layer uint8 allocation, prefills once, computes the uniform allocation
once, crops to the same context boundary before all eight FP/uniform decodes, and
records the allocation hash on every query row. Rows and sidecars carry the
canonical task/panel/RNG versions; the strict reader additionally authenticates
the exact model, corpus SHA, 320 row keys, 40 distinct documents, target
rank/depth rotation, local distractor identities, bit spending, caps, prompt
token accounting, and P0 provenance before applying any outcome gate.

The production Llama-3.1 tokenizer and the staged PG-19 corpus preflight over
prompts 700--739 found 40 distinct unspliced real documents with corpus SHA
`0a26bc1e05a1eea8`. Contexts contain 31,185--31,202 tokens and the longest
context-plus-question contains 31,241 tokens, below 32,768. Every prompt has
four queries and 48 needles. This preflight inspected construction and
provenance only; it did not execute the model or reveal qualification outcomes.

Focused generator (5), runner (3), and strict-reader tests pass, as do the
legacy `test_r8.py --fast`, `test_baselines.py --fast` (with one host thread),
the completed k48 strict reader, and shell/Python syntax checks. The qualification
workflow is:

```bash
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --panel-dry
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --panel-submit
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --panel-status JOB_ID
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh --panel-read JOB_ID
```

Qualification job 983199 was submitted through this exact workflow on the
debug partition with a 20-minute limit and completed in 6:31 (271 seconds in
the model runner). No policy-development job was implemented or submitted.

##### 3G.5 Authenticated qualification outcome and terminal decision

The strict reader accepts the exact 320-row artifact and reports:

| quantity | result | frozen gate |
|---|---:|---|
| FP query accuracy | 157/160 = 0.981 | pass |
| FP slots at depths .15/.38/.62/.85 | 1.000 / 1.000 / 0.950 / 0.975 | pass |
| uniform query/prompt mean | 136/160 = 0.850 | **fail: above 0.75** |
| uniform halves | 0.863 / 0.838 | **fail: first half above 0.85** |
| uniform slots | 0.950 / 0.850 / 0.800 / 0.800 | pass: spread 0.150 |
| FP minus uniform | 0.131 [0.088, 0.181] | pass |
| incomplete capped FP failures | 0 | pass |
| decision | ``stop_panel`` | binding |

Across the 160 paired queries, 136 are correct under both arms, 21 are correct
only under FP, three fail under both, and none is correct only under uniform.
Of the 24 uniform failures, 11 return a registered distractor in the target
cluster, seven return a registered value from another cluster, three are
one-digit target mutations, and three are other numeric answers. Thus 18/24
(75%) are exact registered alternatives. Standard substring score and
`first_ok` agree on all 320 rows, and no failed answer reaches its cap. The
observed gap is therefore a real associative-retrieval loss under B=2, not a
cap, parser, provenance, or single-slot artifact.

The panel improved measurement: four queries give graded prompt outcomes,
cluster rotation balances identity against depth, and the paired interval is
well separated from zero. It did not produce the preregistered 25--50% uniform
failure rate. The positive FP-minus-uniform gap is an upper bound on possible
recovery; it does not establish complementarity among compressed policies or
validate a logit selector because those policies were never run.

As an explicitly post-outcome diagnostic, requiring all four queries to be
correct would score FP at 38/40=0.950 and uniform at 23/40=0.575. That statistic
is not the frozen primary score. Section 3G.2 explicitly forbids changing the
score after a failed qualification, so it cannot reopen V3 or justify using
prompts 740--819. Do not rerun 983199, filter to deeper slots, change cluster
size/depth/budget, or submit panel policy development.

#### 3H. Design direction after V3: complementarity before routing

**State: design direction only; no dataset, thresholds, code, or GPU run is
authorized.** Stop adapting synthetic NIAH until an accuracy band appears. For
the current paper, retain the calibrated/per-head router as a negative result
and center supported claims on characterization, fixed co-designed policies,
and the SOTA comparisons.

If routing remains a research objective, V4 should be a separately versioned
study on one pinned public natural long-context task with official deterministic
scoring. A multiple-choice task is preferable for the first pass because it
removes answer-extraction and generation-cap ambiguity. Before any model
output, a CPU-only data audit must freeze the exact release checksum, prompt
template, parser, eligible example IDs, token-length and category strata, and
document-disjoint qualification/development/test splits. Include every
untruncated eligible example; do not choose examples, lengths, or categories
from FP or compressed outcomes. LongBench-v2 is a candidate to audit, not a
selected dataset.

V4 must separate three questions and artifacts:

1. **End-task opportunity.** Run an ordered, equal-memory menu of complete
   policies and freeze `F*`, the strongest mean fixed policy on development
   data. The label-seeing end-task oracle chooses the highest official score per
   example. Its gain over `F*` is `H`. If H lacks a positive paired interval
   and a practically useful minimum fixed before the run, stop: the candidate
   set has no routing opportunity.
2. **Proxy transfer.** Only after H passes, run the frozen whole-policy logit
   rule. It may execute every policy and compare its logits with FP, but it sees
   no labels. Its end-task gain `G`, captured opportunity `G/H`, rescue/harm
   counts, and fixed-half stability decide whether the proxy transfers. H pass
   with G fail rejects the proxy; it is not repaired with threshold tuning.
3. **Deployable selection.** Only after H and G pass, train a low-capacity
   selector from features available after prompt prefill and before generation.
   Candidate outputs, labels, FP decode logits, and per-policy logits are
   unavailable features. Freeze it with cross-fitting on development data and
   evaluate once on the untouched test split against `F*`, the end-task
   oracle, and the whole-policy logit upper bound.

Qualification should run FP plus uniform only to authenticate the natural-task
pipeline and reject truncation, format failure, an FP floor, or an unusable
uniform extreme. Development should run all eight already ordered complete
policies; test must keep the candidate menu, `F*`, proxy, feature set, parser,
and thresholds fixed. Bootstrap examples while preserving any document or
question grouping. A later architecture or dataset is external validation, not
a fallback after a failed gate.


#### 3I. V4: pinned natural multiple-choice complementarity study

**State (2026-09-23): qualification job 983715 completed with strict decision
`stop_v4`; development and confirmation remain unopened.** Protocol and
acceptance rules were frozen before model output. Preflight job 983708 was
cancelled after 32 seconds during tests, before model inference or output, while
a suspected cache-hook issue was checked. The existing compressed multi-token
path and its R8 regression resolved that concern; qualification was resubmitted
unchanged as job 983715. V4
is a new experiment. It does not reopen V2 or V3 and does not use prompts
740--819. Its purpose is to answer, in order, whether complete compressed
policies have end-task complementarity on a natural task, whether a label-free
whole-policy logit proxy captures it, and only then whether a deployable
selector is justified.

##### 3I.1 Public task and immutable provenance

Select LongBench v2 under this exact local protocol:

- dataset `zai-org/LongBench-v2` (the current target of the official
  `THUDM/LongBench-v2` name), revision
  `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`; `data.json` is 465,490,535
  bytes with SHA-256
  `15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2`;
- official code `THUDM/LongBench` at
  `2e00731f8d0bff23dc4325161044d0ed8af94c1e`;
- exact `prompts/0shot.txt`, including no final newline, SHA-256
  `68a162252bc9ff71d5d7abca3d69bb31aac3c35f832d657a2866f2018b8a6950`;
- official direct-answer parser: remove `*`, then case-sensitive
  `The correct answer is \(([A-D])\)`, with the official no-parentheses
  fallback; invalid output is retained and scored zero;
- `meta-llama/Llama-3.1-8B-Instruct` tokenizer/model revision
  `0e9e39f249a16976918f6564b8830bc894c89659`, whose chat-template SHA-256 is
  `e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65`;
  record `transformers==5.16.1` and `tokenizers==0.23.1`; and
- render the official prompt with stripped field substitution, then one user
  message and `add_generation_prompt=True` from the pinned Llama template.

The official runner samples at temperature 0.1 and middle-truncates overlength
inputs. V4 instead uses greedy decoding so policy differences cannot exploit
sampling noise, retains the official 128-token cap, and forbids truncation.
Describe results as **LongBench v2 with the official prompt/parser and a greedy
controlled decode**, not as an exact leaderboard reproduction.

##### 3I.2 Eligibility and frozen splits

Reserve all 128 possible answer tokens inside the 32,768-token experiment
window. An item is eligible exactly when its official length label is `short`
and its fully rendered Llama chat input has at most 32,640 tokens. This
pre-output rule admits 117 of 503 rows, with 10,110--32,592 input tokens. The
sorted `_id<TAB>input_tokens\n` eligibility list has SHA-256
`d87774ba198ad16645bb9364bd96e9fa80e5e5c8e5a7924ac9c5677ae370d3d0`.
All 117 happen to be in the official short word-count bin; no category, answer,
FP outcome, or compressed outcome participates in eligibility.

The released data has no source-document identifier. Build the strongest
verifiable leakage groups as connected components: two rows belong to one
component when they share either SHA-256 of `context.strip()` or SHA-256 of
`question.strip()`, transitively. This yields 108 components. For each
component, take the minimum over its member context hashes of
`SHA256(b"longbench_v2_sieve_v4_group_split_v1\x00" || context_hash_ascii)`, sort
by that key, and assign components 0--18 to qualification, 19--62 to
development, and 63--107 to confirmation. The sets have zero overlap in IDs,
exact contexts, and exact questions.

| phase | components | rows | token range | sorted `id\n` SHA-256 |
|---|---:|---:|---:|---|
| qualification | 19 | 20 | 10,110--30,051 | `e1a2b0a72f6e783369af4cb743d91a4725c373e5baa9a8db24b232fa45a3ca38` |
| development | 44 | 52 | 10,727--29,984 | `98bf6531b62e474a05b66f488b298d51b7c1ecb53e4e0e97c23495e8dacace35` |
| confirmation | 45 | 45 | 12,701--32,592 | `31c9cd599df77ba9bada01ce601f1b04fa1e5c0316dd3198e1b20ba147231abe` |

The corresponding sorted `_id<TAB>input_tokens\n` hashes are qualification
`6648d0b34a7ef13184c2c7384973443158754fb5dd8440992cf2975e0fe4e47f`,
development `820ea99fdea86db20e4e1e9d297c7c174eeb78a48b6325968c755f4cd4eb893a`,
and confirmation `fb21285dd3296458e9b2e68a1d424f1df342071bbc3f4f2289fa979f4d949244`.

The eligible set is naturally imbalanced: 61 single-document, 33
multi-document, 7 long-ICL, 12 dialogue-history, 3 code, and 1 structured-data
row. The primary estimand is therefore paired micro accuracy. Domain and
difficulty summaries are descriptive. Bootstrap the 108 connected components,
carrying all questions in a component together. Fixed-half checks split the
ordered components within a phase once at the midpoint; they never split a
component.

This is an internal development/confirmation split of a public single
benchmark split. Its protection is procedural, not hidden-test secrecy. The
manifest contains IDs, grouping, tokens, and public metadata but no answer
field. The authenticated dataset remains the only source of labels at scoring
time.

##### 3I.3 Clean execution boundary

Use a separate `run_longbench_v2.py`; do not add LongBench conditionals to the
RULER registry or main loop. The runner may reuse the tested R8 prefill,
allocation, bit-application, and baseline components. It must consume explicit
manifest IDs, emit every one exactly once per arm, and fail on token, prompt,
model, split, or checksum drift.

Compression is query-aware at the end of the complete official chat prompt.
Prefill all but its final token once, protect the last 32 prompt tokens, build
one immutable allocation per complete policy at B=2, then restore the same
cache boundary before each greedy decode. The final prompt token produces the
first answer distribution through the compressed cache. Head-error,
answer-mass, calibrated-route, question-agnostic, truncation, sampling, and
outcome filtering interfaces are unavailable in V4.

The ordered equal-memory menu is fixed to:

`uniform, evict, interior, interior_pool, interior_cascade, obcache_k, obck_ada, laprox`

where raw `obck_ada` is `obcache_k:alloc=ada@obck_ada`. FP is a ceiling and
proxy reference, not a selectable B=2 policy. The runner writes end-task rows
and a separate scalar-only choice-logit artifact; no raw logits or answer labels
enter allocation or selection.

##### 3I.4 Whole-policy choice-logit proxy

The official requested response gives a model-independent decision location.
For the pinned tokenizer, `The correct answer is (` is token IDs
`[791, 4495, 4320, 374, 320]`, followed by four distinct one-token branches
A/B/C/D = `[32, 33, 34, 35]` and the same closing suffix. For each complete
policy, restore its full allocation, teacher-force that canonical prefix, and
retain only its four branch logits. Normalize over all four choices. The frozen
proxy selects the candidate minimizing
`KL(p_FP(A,B,C,D) || p_candidate(A,B,C,D))`; an exact tie uses candidate order.
This rule sees every policy and FP logits, so it is an unavailable diagnostic
upper bound rather than a deployable router. It never sees the answer label.

Before using this proxy, qualification must show that FP's canonical-choice
argmax agrees with its parsed greedy answer on at least 80% of valid FP rows.
Failure rejects this elicitation interface; it does not authorize switching to
trajectory KL, output error, another prompt, or a threshold sweep.

##### 3I.5 Frozen sequential gates

**Qualification (20 rows).** Run only FP and uniform B=2, while recording the
FP/uniform canonical choice diagnostics. Advance only if all row/provenance/bit
checks pass, every manifest item is untruncated and retained, FP has at least
18/20 valid parsed answers, uniform has at least 16/20, FP accuracy is in
[0.15, 0.70], uniform accuracy is in [0.10, 0.70], neither arm has a missing
row, no invalid FP answer terminates only at the 128-token cap, and the FP
canonical/direct agreement is at least 0.80 among valid FP rows. There is no
gate on the sign of FP-minus-uniform and no task, budget, or subset selection.
A failure stops V4 without tuning qualification.

**Development (52 rows).** Only after qualification passes, run FP plus all
eight ordered candidates once. Freeze `F*` as the candidate with the largest
mean official accuracy, with candidate order resolving an exact tie. The
label-seeing end-task oracle scores one when any candidate is correct. Define
`H = oracle_accuracy - F*_accuracy`. Opportunity passes only if H >= 0.10, the
5th percentile of a 10,000-draw seed-0 paired component bootstrap is above
zero, and H >= 0.05 in both fixed component halves. If this fails, stop: this
candidate menu provides no established routing opportunity.

If H passes, evaluate the already frozen minimum-choice-KL selector. Define
`G = selector_accuracy - F*_accuracy`. Proxy transfer passes only if G >= 0.05,
G/H >= 0.50, the paired component-bootstrap 5th percentile for G is at least
zero, G >= 0 in each fixed half, and canonical/direct FP agreement remains at
least 0.80. Report rescue, harm, both-correct, and both-wrong counts. H pass with
G fail rejects this proxy; do not tune a KL threshold or choose a recorded
secondary metric.

**Confirmation (45 rows).** Run it only after both development gates pass.
Carry forward the exact candidate menu/order, `F*`, choice proxy, B=2, prompt,
parser, decode, grouping, and thresholds. Require the same H and G gates on the
untouched confirmation components without reselecting `F*`. A failure closes
this iteration. A pass authorizes a separately specified deployable-selector
study; it does not by itself establish a deployable SIEVE router.

The ordered workflow is CPU audit/manifest, qualification, development H,
development G, locked confirmation, and only then deployable selection. Later
models or 64K/128K subsets are external validation after this sequence, not
fallbacks for a failed gate.

##### 3I.6 Frozen qualification outcome and stop (job 983715)

The strict artifact reader reports FP valid 18/20, uniform valid 18/20, FP and
uniform official accuracy both 0.450, and FP canonical/direct agreement
17/18 = 0.944. Those gates pass. Two invalid FP responses terminate at the
128-token cap, so the required zero-censored-FP gate fails and the frozen
decision is `stop_v4`. Both failed rows belong to one question-linked component
and both FP and uniform spend the cap calculating an answer without emitting
the official phrase. This diagnoses the generative answer interface; it does
not authorize increasing the cap, changing the parser, excluding dialogue, or
using canonical-choice accuracy inside V4.

No V4 development or confirmation output may be generated. At the V4 stop, the
52 development and 45 confirmation items were untouched and could be used only
by a separately versioned protocol whose scorer and proxy were frozen before
opening either split. V5 subsequently opened the 52 development rows under that
separate protocol; the 45 confirmation rows remain untouched.

#### 3J. V5: forced-choice opportunity and branch-blind whole-policy proxy

**State (2026-09-23): completed and closed as `stop_no_opportunity` by
authenticated development job 983888.** V4 remains terminal `stop_v4`. V5 is a
separately named experiment and never reclassifies job 983715 as a pass. The 20
exposed qualification rows are a disclosed design pilot only: their saved
canonical choices score FP at 11/20 and uniform at 10/20. They are excluded
from every V5 estimate, threshold, and claim. V5 opened the 52-row development
split once. Its opportunity gate failed, so the 45-row confirmation split is
still untouched and must not be opened by V5.

##### 3J.1 Endpoint and fixed policy menu

Retain the pinned dataset, model and tokenizer revisions, official zero-shot
prompt, complete untruncated inputs, existing context/question components,
32,768-token window, query-aware compression at the complete prompt end,
protected 32-token observation window, B=2, allocator settings, and this exact
ordered compressed-policy menu:

`uniform, evict, interior, interior_pool, interior_cascade, obcache_k, obck_ada, laprox`.

FP is a reference and competence check, not an admissible B=2 policy. The raw
arm spelling for `obck_ada` remains `obcache_k:alloc=ada@obck_ada`.

V5 performs no free generation and does not invoke the official answer parser
or its 128-token cap. For FP and every complete policy, restore the identical
cache boundary immediately before the final prompt token and teacher-force
`[last_prompt_token, 791, 4495, 4320, 374, 320]`. The five fixed tokens spell
the canonical scaffold `The correct answer is (` under the pinned tokenizer.
At output position 5, restrict the logits to the one-token A/B/C/D branches
`[32, 33, 34, 35]`, apply a temperature-one four-way softmax, and select the
argmax with A/B/C/D order resolving an exact tie. Accuracy is equality with the
authenticated released answer. This endpoint is **LongBench-v2 forced-choice
accuracy**; it is not official LongBench generative accuracy.

The runner must never consult the answer when it allocates, executes, or
selects a policy. It writes choices and scalar confidence summaries without a
gold answer or score. Only the strict reader authenticates the dataset and
joins answers after it has recomputed the label-free selection.

##### 3J.2 Primary proxy and descriptive FP-choice fidelity

The primary rule is `branch_blind_scaffold_kl_v1`. In the same forward pass,
positions 0--4 are the full-vocabulary distributions that predict the five
fixed scaffold tokens. For candidate policy p define

`D_p = (1/5) sum_t KL(softmax(z_FP,t) || softmax(z_p,t))`, for t=0,...,4,

using temperature one and float32 arithmetic over the complete vocabulary.
Position 5, which predicts A/B/C/D and defines end-task correctness, is
explicitly excluded. Select the candidate with minimum `D_p`; exact numerical
ties use the frozen menu order. This rule sees neither the answer label nor the
candidate's final choice. It still executes FP and every complete policy, so it
is an unavailable whole-policy diagnostic rather than a deployable router.

The four-choice FP-to-candidate KL may be retained as a separately named scalar
diagnostic, but it cannot select V5's primary policy or enter a gate. Minimum
A/B/C/D KL is label-free in a literal sense, yet it consumes the statistic that
defines the forced-choice endpoint and primarily imitates FP. With FP at 0.55
in the exposed pilot, preserving its decision can also preserve its mistakes.
Do not replace the branch-blind rule with A/B/C/D KL, reverse a KL, or add a
threshold after development output.

##### 3J.3 Frozen splits, halves, and artifacts

Development contains 44 connected components and 52 rows (ID hash
`98bf6531b62e474a05b66f488b298d51b7c1ecb53e4e0e97c23495e8dacace35`,
ID/token hash
`820ea99fdea86db20e4e1e9d297c7c174eeb78a48b6325968c755f4cd4eb893a`).
Order components by the already frozen audit key
`min(sha256(split_namespace || member_context_hash_ascii))`, with component ID
as the exact tie break. Its first 22 components contain 28 rows (ID hash
`b006a353c33488d661b87932cf1c4abf8ce1e7aff63e688810a74b13a85184b8`)
and its second 22 contain 24 (ID hash
`6adf6c4b4e378a4f4605a4a959df0766fad7329f384bf27c8734abaae1296a1b`).

Confirmation contains 45 one-row components (ID hash
`31c9cd599df77ba9bada01ce601f1b04fa1e5c0316dd3198e1b20ba147231abe`,
ID/token hash
`fb21285dd3296458e9b2e68a1d424f1df342071bbc3f4f2289fa979f4d949244`).
Its ordered 22/23-component halves have 22/23 rows and ID hashes
`b0e905a214648f35d9c38a615c5e0d6811a07f69d9558209e224e023efaa8e08`
and `c95acca30821a88fa4c6a8ff6a885c9a05c867fdbb00c2b9be10659f84753d53`.

Use a new runner/protocol/result namespace
`longbench_v2_sieve_v5_forced_choice_v1` / `lbv2_v5_*`. Write two
cross-hashed artifacts:

1. a prediction table keyed by `(item_id, arm)`, with 52 x 9 development rows
   (or 45 x 9 confirmation rows), forced choice, restricted choice entropy,
   margin and maximum probability, allocation identity, bit audit, and full
   provenance; and
2. a proxy table keyed by `(item_id, candidate)`, with 52 x 8 development rows
   (or 45 x 8 confirmation rows), the five-step scalar scaffold summaries,
   declared-order primary selection, allocation identity, and any explicitly
   descriptive four-choice divergence scalars.

Store no gold answer, correctness score, response text, or raw logit vector in
either artifact. Sidecars must authenticate and cross-hash both tables. Tests
must pin cache restoration and the 0--4 versus 5 position boundary, and show
that gold permutation or arbitrary changes confined to the excluded A/B/C/D
logits cannot change the primary selector.

##### 3J.4 Development analysis and sequential gates

After exact row, provenance, no-truncation, allocation, and B=2 checks, join
the authenticated answers. Freeze `F*` as the compressed candidate with the
largest 52-row forced-choice accuracy; exact ties use menu order. Define the
label-seeing candidate oracle as correct when any of the eight compressed
policies is correct, and

`H = oracle_accuracy - F*_accuracy`.

First require a coherent four-choice operating point: both FP and `F*` must
have point accuracy at most 0.75, and each must have a component-bootstrap 5th
percentile above chance 0.25. Then opportunity passes only when H is at least
0.10, its component-bootstrap 5th percentile is strictly above zero, and H is
at least 0.05 in each frozen half. Failure returns
`stop_invalid_operating_point` or `stop_no_opportunity` and suppresses all G
outcomes in the reader's report.

Only after H passes, evaluate the already frozen scaffold selector. Define

`G = scaffold_selector_accuracy - F*_accuracy`.

Proxy transfer passes only when G is at least 0.05, G/H is at least 0.50, the
component-bootstrap 5th percentile of G is at least zero, and G is nonnegative
in both frozen halves. Report rescue, harm, both-correct, and both-wrong counts
against `F*`. Failure returns `reject_scaffold_proxy`; passing returns
`advance_confirmation`.

Every bootstrap uses 10,000 draws from NumPy `Generator(PCG64(0))`, samples
phase components uniformly with replacement, carries every row of a sampled
component, computes micro means, and uses
`quantile(0.05, method="linear")`. `F*` is the development-selected policy and
is not reselected inside bootstrap draws or halves.

##### 3J.5 Locked confirmation

An `advance_confirmation` reader writes a lock containing the development
artifact hashes, exact runner and reader hashes, menu/configuration, `F*`,
proxy/version, split/half hashes, and all gates. The confirmation worker must
require and authenticate that lock before model loading. It uses the same menu,
endpoint, proxy, settings, competence gates, H/G thresholds, bootstrap, and
half rules on the untouched 45 rows, with the development-selected `F*`; it
does not reselect a fixed policy. A failure closes V5. A pass establishes a
nondeployable whole-policy diagnostic and authorizes a separately specified
cross-fitted selector using only features available before answer generation.
It does not by itself validate the existing SIEVE router.

##### 3J.6 Authenticated development outcome and stop (job 983888)

The strict reader authenticated the two reciprocal artifacts and sidecars, the
pinned dataset, manifest, model and tokenizer revision, complete input hashes,
52 items in 44 components, all 468 prediction rows, all 416 proxy rows, unique
allocation identities, and exact B=2 spending. The Slurm job exited 0:0 after
9:47 (8:51 in the GPU step).

| frozen quantity | result | gate |
|---|---:|---|
| FP forced-choice accuracy | 19/52 = 0.365 | competence pass; q05 = 0.255 |
| every compressed candidate | 19/52 = 0.365 | `F* = uniform` by frozen tie order; q05 = 0.260 |
| candidate oracle | 22/52 = 0.423 | descriptive ceiling |
| `H = oracle - F*` | 3/52 = 0.058 | **fail:** required at least 0.10 |
| H component-bootstrap q05 | 0.017 | pass: strictly above zero |
| H fixed halves | 0.071 / 0.042 | **fail:** half 2 required at least 0.05 |
| strict decision | `stop_no_opportunity` | binding |

The equal marginal accuracies hide only two endpoint behaviors. Uniform differs
from FP on ten items. Every other candidate -- eviction, all three interiors,
OBCache-K, OBCache-K plus Ada-KV, and LaProx -- exactly matches FP's forced
choice on all 52 items and therefore exactly matches every other nonuniform
candidate. Against uniform, that common behavior produces three rescues, three
harms, and four changed choices for which both are wrong. The allocations are
not aliases: all eight candidates have distinct allocation IDs on every item,
and their confidence summaries differ. The collapse occurs at the final answer
argmax.

The candidate oracle's bootstrap excludes exactly zero, but it recovers only
three rows and misses both the preregistered practical magnitude and second-half
stability requirements. This is an upstream policy-opportunity failure. It does
not authorize lowering H, selecting a subset, or retuning the proxy. The strict
reader correctly suppresses every G outcome. No confirmation lock exists and
V5 confirmation is forbidden.

#### 3K. V6: stronger-model qualification before another policy study

**State (2026-09-23): the frozen Qwen qualification completed as job 984224 and
returned `advance_v6_development`. The qualification artifact, sidecar, source
seal, model snapshot, and advance lock authenticate. The conditional development
implementation is now the active branch; no Qwen development output has been
generated or inspected.** V6 does not alter
V5's terminal decision. It asks whether V5's endpoint collapse was dominated by
the barely-above-chance Llama task competence rather than by the policy family.
It changes the model once, keeps B=2, and starts with a small disclosed
qualification. It does not sweep models, budgets, prompts, or subsets.

##### 3K.1 Why Qwen3-30B at B=2

Do not make Llama compression harsher at B=1. V5's FP endpoint is only 19/52,
and all seven sparse candidates already preserve FP choices exactly at B=2.
A lower budget would manufacture diversity by damaging a marginally competent
endpoint and risks turning policy opportunity into multiple-choice lottery.

Use the strongest locally pinned model already exercised by the cache stack:
`Qwen/Qwen3-30B-A3B-Instruct-2507` at revision
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`. Prior H0/co-design runs establish
that this architecture, revision, custom attention path, and one-H100 loading
path execute in this repository. Keep B=2, W=32, bfloat16, norm correction,
rotation seed 0, maxb 8, and the same official LongBench-v2 prompt and source
items. This is an architecture replication and competence repair, not a new
compression stress sweep.

##### 3K.2 Gold-free Qwen source contract

Reuse the already fixed V4 component/ID partition: 20 qualification rows in 19
components, 52 development rows in 44 components, and 45 untouched confirmation
rows in 45 components. The qualification labels were previously exposed and
serve only as a disclosed model/interface pilot. Qwen development outputs have
not been generated; confirmation outputs remain untouched. The Qwen manifest
must contain IDs, component IDs, source hashes, metadata, Qwen token counts and
prompt-token hashes, but no answers.

Apply the pinned Qwen chat template with one user message,
`add_generation_prompt=True`, and `enable_thinking=False`. The template SHA-256
is `64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326`.
The canonical response scaffold tokenizes as
`[785, 4396, 4226, 374, 320]`; the one-token A/B/C/D branches remain
`[32, 33, 34, 35]`. V6 uses the same forced-choice endpoint and no free
generation.

Set the experiment window to 40,960 so every preselected item remains complete.
The CPU-only pre-output audit gives these ranges and hashes:

| split | rows/components | Qwen input-token range | sorted `id\tinput_tokens\n` SHA-256 |
|---|---:|---:|---|
| qualification | 20 / 19 | 10,150--30,588 | `bb453f5c5a27694cf2c9214d9752f4af0670f0a33a24ed2881caf78d0f5c2d51` |
| development | 52 / 44 | 10,873--31,848 | `0e2db1d967d93da4a2f937423f9f330031753f989740252f90b32c68cd691cf8` |
| confirmation | 45 / 45 | 12,677--33,438 | `3b24f41058ae272268b465f2d4ce7847ed3c38e07684f5fa2322c406eb1344e6` |

The source-ID hashes remain V4's
`e1a2b0a72f6e783369af4cb743d91a4725c373e5baa9a8db24b232fa45a3ca38`,
`98bf6531b62e474a05b66f488b298d51b7c1ecb53e4e0e97c23495e8dacace35`, and
`31c9cd599df77ba9bada01ce601f1b04fa1e5c0316dd3198e1b20ba147231abe`;
the per-prompt token hashes are also binding in the manifest. No row is selected or
removed from a Qwen outcome.

##### 3K.3 Qualification first: FP plus uniform only

Run exactly the 20 qualification rows with FP and uniform B=2. The runner is
label blind and writes exactly 40 scalar forced-choice rows plus an adjacent
sidecar. It stores no answer, correctness flag, response text, raw logits, or
probability vector. The strict reader authenticates the dataset, Qwen manifest,
model/tokenizer/template, split and prompt hashes, schema, allocation identity,
no truncation, and bit audit before joining labels in memory.

Use a 10,000-draw seed-0 component bootstrap, sampling the 19 components with
replacement and carrying the two linked rows together. Advance only when all
provenance checks pass and:

1. FP forced-choice accuracy is at least 0.50 and its bootstrap q05 is strictly
   above chance 0.25;
2. uniform accuracy lies in [0.30, 0.80] and its bootstrap q05 is strictly above
   0.25; and
3. no row is missing, truncated, over budget, or outside the exact one-token
   branch contract.

Qualification does not select a task, item subset, budget, or model. Failure
returns `stop_v6_qualification` and closes V6. Passing returns
`advance_v6_development` and writes an authenticated lock. The development
worker must validate the lock contents, qualification artifact hash, executed
runner hash, reader hash, manifest hash, model/menu/configuration and thresholds
before loading the model. A nonempty lock path alone is insufficient.

##### 3K.4 Conditional development and confirmation

Only after qualification passes, run FP plus the unchanged ordered candidate
menu on all 52 development rows:

`uniform, evict, interior, interior_pool, interior_cascade, obcache_k, obck_ada, laprox`.

Freeze `F*` by development accuracy with menu order for an exact tie. Require
FP accuracy at least 0.50 with bootstrap q05 above 0.25; require `F*` accuracy
in [0.30, 0.80] with q05 above 0.25. Then apply V5's unchanged opportunity
gate: H at least 0.10, component-bootstrap q05 strictly above zero, and H at
least 0.05 in each frozen 22-component half. Report answer-vector equivalence
classes descriptively. H failure closes routing work for this menu before any
proxy claim.

The primary proxy is frozen now, before Qwen output, to V5's branch-blind
five-position scaffold KL with Qwen's scaffold token IDs. Its G outcome is
suppressed unless every H gate passes. If reached, retain V5's thresholds:
G at least 0.05, G/H at least 0.50, bootstrap q05 at least zero, and G
nonnegative in both halves. Only H and G passing may write a second lock and
open the 45 confirmation rows once. Confirmation carries the exact model,
40,960 window, B=2, menu/order, development-selected `F*`, proxy, hashes,
bootstrap and gates. It never reselects `F*`.

##### 3K.5 Implemented qualification contract and preflight

The qualification implementation was completed and frozen before submission.
It uses protocol `longbench_v2_sieve_v6_qwen_qualification_v1`, runner
`longbench_v2_qwen_qualification_runner_v1`, task
`longbench_v2_0shot_qwen3_2507_no_thinking_v1`, endpoint
`canonical_abcd_forced_choice_v1`, and manifest
`longbench_v2_sieve_v6_qwen30_manifest_v1`. The exact context rule is
`input_tokens + 5 <= 40,960`, so the last prompt token plus all five scaffold
tokens fit in the model call. The largest frozen input is 33,438 tokens.

The gold-free Qwen manifest has canonical content SHA-256
`c6a951bb5433dd5c918ca6c8c964eb2ed4a4c26f9a079f53fb9111e9a3279f5b`
and file SHA-256
`a85b94fe636ce00ccddbc4ffa0f371f4d666aa0aa02750841444a811250a1946`.
It contains exactly 117 rows and preserves the full V4 ID/component partition.
The runner authenticates the pinned local model snapshot before loading: 24
content-addressed entries, inventory SHA-256
`4ac169930fba4989c196232eae1528dfea473cc695a439061741aa61df3510c2`,
plus fixed hashes for the config, weight index, tokenizer metadata, tokenizer,
merges, vocabulary, and generation config. It also requires the exact 48-layer,
32-query-head, 4-KV-head, 128-dimensional attention architecture and rejects
CPU or disk offload.

For every item, uniform must contain all 48 layer tensors with shape
`[4, input_tokens - 1 - 32]`, uint8 dtype, and every entry exactly 2. The runner
and reader independently compute its deterministic
`sha256_layer_uint8_v1` allocation identity. Mean-bit checks alone cannot make a
heterogeneous 1/3-bit allocation pass as uniform.

The source seal is captured before model loading and recomputed before artifact
write. It covers the runner, strict reader, Slurm worker, Qwen auditor, model
registry, `run_h0.py`, `run_r8.py`, and the executed `compress`, `forced_choice`,
`policy_diagnostic`, `probe`, `quant`, `router`, and LongBench-v2 task modules.
The snapshot attestation is likewise repeated after inference. Any change
during the job aborts output. The artifact contains no answer, correctness,
response, raw logit, or probability vector. The authenticated source JSON
contains released labels, but the runner never uses them for prompt rendering,
allocation, execution, or serialization; the strict reader joins them only
after artifact authentication.

Frozen implementation hashes at preflight were:

| file | SHA-256 |
|---|---|
| `sievelib/forced_choice.py` | `9b1c82561bd37eb6e4a27b7918a0534401806f91ea0e0b836e97fc0f4b2d0e2a` |
| `h0_measurement/audit_longbench_v2_qwen.py` | `4670004b1233440d880c94cdfa4a0bd8786325a800b31567b37b8d19004c2bf7` |
| `h0_measurement/run_longbench_v2_qwen_qualification.py` | `0c0546261961253b8c5e4b5fa35b4ba7271102149ace283a44b7dc5763ff855c` |
| `read_longbench_v2_qwen_qualification.py` | `2c2a8a3d6109e3ded6d166eceac0fbb21e46d0fa67daf2df8c4b3692f18f1991` |
| `submit_longbench_v2_qwen_qualification.slurm` | `8881598548e22f7b9e0809bb62fa3e49619a8948c5157514a2dd41f842f0ba5c` |
| `tests/test_longbench_v2_qwen_qualification.py` | `563615b426bc2b0fb84dc5c633ec279e7e851a457a028578f5e29ad0ea50e315` |

Preflight passed the 12-test V6 suite, Python compilation, both shell syntax
checks, the complete R8 fast suite, manifest/snapshot authentication, and the
local and remote dry-run interface. The qualification-only submission is:

```bash
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh \
  --lbv2-v6-qual-submit
```

It requests one H100 on the debug partition for at most two hours and writes
`h0_measurement/results/lbv2_v6_qwen_qualification_JOB/`. The worker creates an
exact completion marker only after the strict reader accepts the 40-row
artifact. No V6 development or confirmation submission mode exists at this
stage.

##### 3K.6 Authenticated qualification outcome (job 984224)

Job `984224` completed in 8:49 (GPU step 7:26, exit 0:0, peak RSS about
59.7 GiB). The strict reader accepted all 40 rows, all 20 frozen items in 19
components, the exact FP/uniform menu, complete prompts, deterministic uniform
allocations, exact B=2 spending, reciprocal provenance, all source hashes, and
the pinned Qwen snapshot. FP and uniform each score 12/20 = 0.600. Their
10,000-draw seed-0 component-bootstrap q05 values are 0.421 and 0.429,
respectively, so every frozen qualification gate passes.

FP and uniform agree on 18/20 choices. The two differences consist of one
uniform rescue and one uniform harm. This is a competence and non-ceiling pass;
it is not evidence of policy opportunity or proxy utility. The authenticated
decision is `advance_v6_development`, and the qualification lock is
`longbench_v2_v6_qualification_lock_984224.json` with canonical content SHA-256
`4ede1d969fc9ad4905587baa9000eae30f670e93752622c9ea516d90ec576491`.
That lock authorizes only the already frozen 52-row development branch in
3K.4. Development must still pass competence and H before G may be disclosed.


##### 3K.7 Frozen development implementation before Qwen development output

The lock-authorized development implementation and local/remote preflight are
complete. No Qwen development output had been generated or inspected when this
contract was frozen. The runner authenticates the exact job-984224 lock and its
qualification artifacts, the qualification source seal, dataset, source and
Qwen manifests, configuration, and pinned model snapshot before loading either
the tokenizer or model. The worker repeats this authorization as its first
Python action. The runner rechecks the source seal, snapshot, and qualification
lock after inference and before writing artifacts.

The development protocol is
`longbench_v2_sieve_v6_qwen_development_v1`, runner version
`longbench_v2_qwen_development_runner_v1`, reader version
`longbench_v2_qwen_development_reader_v1`, and completion marker
`longbench_v2_v6_qwen_development_v1`. It writes exactly
`longbench_v2_v6_qwen_forced_choice_development.parquet` with 468 rows and
`longbench_v2_v6_qwen_scaffold_proxy_development.parquet` with 416 rows, plus
reciprocally hashed JSON sidecars. Both artifact schemas remain scalar and
label free. The five full-vocabulary scaffold distributions are transient;
only five scalar KL values and their mean are serialized.

Uniform is authenticated as all 2-bit uint8 storage at every position in all 48
layers and four KV heads, including its deterministic allocation hash. Sparse
policies use the inherited feasible budget rule: integer keep counts may
underfill B=2 slightly but must never overspend. This distinction is covered by
a regression test because prior valid sparse allocations range slightly below
2 bits/token.

The reader fixes `F*` once by 52-row accuracy and menu order, carries that choice
through bootstraps and halves, and uses the four exact terminal decisions
`stop_v6_invalid_operating_point`, `stop_v6_no_opportunity`,
`reject_v6_scaffold_proxy`, and `advance_v6_confirmation`. G fields remain null
unless competence and every H gate pass. Only `advance_v6_confirmation` writes
an idempotent confirmation lock; a non-advance decision refuses a stale lock.
The reader also reports answer-vector equivalence classes descriptively.

Frozen implementation hashes are:

| file | SHA-256 |
|---|---|
| `h0_measurement/run_longbench_v2_qwen_development.py` | `7fc23d902e043c63260215456aa61a70e0bac70ab2931c16df2c1a0ecabc5d38` |
| `read_longbench_v2_qwen_development.py` | `20b58de59401833289acc95ec8f74abd2cba04789635839ec4ae1d2f7032a219` |
| `submit_longbench_v2_qwen_development.slurm` | `afb5e0fde7cf8c93327b71c52ea07483c44ac036fab10a78980077780ec0f20e` |
| `tests/test_longbench_v2_qwen_development.py` | `03343db334112b16a34d18dcbf7b8933d91d1f33cd236e88ecc9851b7cd0b4c2` |
| `steps.sh` | `a3a2bc9b9fe94cb9794215d3de90719d98f71a2c41114318a68e0603f12622b9` |
| qualification lock 984224 | `1deaad5ceb35f943301262fd81c2dfa58b1e5a0dbff2ceaa0975af8257ecebd9` |

Preflight passed the six-test development suite, the unchanged 12-test
qualification suite, the complete R8 and SOTA-baseline fast suites, Python and
shell syntax checks, exact runner/reader sidecar-schema comparison, real lock
authentication, and local plus remote dry runs. The single authorized submission
is:

```bash
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh \
  --lbv2-v6-dev-submit
```

It requests one H100 on the debug partition for at most two hours and writes
`h0_measurement/results/lbv2_v6_qwen_development_JOB/`. The 45 confirmation rows
remain inaccessible from this worker.

Development job `984370` was submitted through this frozen interface on
2026-09-23 and completed successfully. Section 3K.8 records its binding stop;
no V6 confirmation work is authorized.

##### 3K.8 Authenticated development outcome and terminal stop (job 984370)

Job `984370` completed in 16:38 (GPU step 15:25, exit 0:0, peak Python-step
RSS 63,125,752 KiB, about 60.2 GiB). The worker and strict reader authenticated
the job-984224 qualification lock, pinned dataset and manifests, complete model
snapshot, executed source seal, 52 items in 44 leakage components, all 468
prediction rows and 416 proxy rows, reciprocal sidecars, exact all-2 uniform
allocations, feasible sparse spending, and the completion marker.

| frozen quantity | result | gate |
|---|---:|---|
| FP forced-choice accuracy | 24/52 = 0.462 | **fail:** required at least 0.50; q05 = 0.340 passes |
| best fixed compressed policy `F*` | uniform, 27/52 = 0.519 | point and q05 = 0.404 pass |
| every nonuniform candidate | 24/52 = 0.462 | descriptive |
| itemwise compressed-policy oracle | 29/52 = 0.558 | descriptive ceiling |
| observed `H = oracle - F*` | 2/52 = 0.038 | below 0.10 |
| H component-bootstrap q05 | 0.000 | fails the strict-above-zero gate |
| H frozen halves | 0.036 / 0.042 | both below 0.05 |
| proxy G | suppressed | sequential competence/H gates did not pass |
| strict decision | `stop_v6_invalid_operating_point` | binding; V6 closed |

The formal first failure is the preregistered FP competence floor. It cannot be
repaired by lowering 0.50 after observing 24/52. The observed policy envelope
also independently gives no reason to continue this menu: uniform changes 11
FP choices, rescuing five FP errors and harming two FP successes, but every
nonuniform candidate except LaProx exactly matches FP on all 52 items. LaProx
differs from FP once and is still wrong on that row. The two oracle rescues over
uniform are precisely the two rows on which uniform harms FP.

This is again an endpoint collapse rather than an allocation alias. Every item
has eight distinct candidate allocation IDs. Interior variants evict roughly
61--65% of context keys and 8-bit sparse baselines roughly 75%, yet eviction,
the three interiors, OBCache-K, and OBCache-K plus Ada-KV share FP's complete
answer vector. The answer-vector classes are (1) FP plus those six candidates,
(2) uniform, and (3) LaProx. Allocation diversity therefore does not produce
enough end-task decision diversity.

No `longbench_v2_v6_confirmation_lock_984370.json` exists. The 45 confirmation
rows cannot be used to rescue competence, enlarge H, tune the scaffold proxy,
or test a changed menu. A later iteration must be separately versioned and must
use fresh development evidence for any changed operating point or policy
mechanism. Repeating B=2 with another scorer, changing the FP threshold, or
revealing G would be outcome-dependent and is prohibited.

Authenticated outputs:

- `h0_measurement/results/lbv2_v6_qwen_development_984370/`
- `h0_measurement/logs/lbv2v6d_984370.out`
- `longbench_v2_v6_development_984370.txt`
- `longbench_v2_v6_development_984370_summary.csv`


#### 3L. V7: structured-query mechanism repair on a fresh 128K block

**State (2026-09-24, terminal after qualification): V6 is terminal. V7 was a
separately versioned, bounded mechanism test whose source partition, mechanism,
qualification and conditional-development implementations, thresholds, runtime,
and complete qualification source ledger were frozen before model output.
Authenticated job 984886 returned `stop_v7_qualification`. Qualification ran
only FP and uniform, so the structured-query mechanism was never evaluated.
No V7 development or confirmation phase is open.**

##### 3L.1 Why one more mechanism test is justified

V5 and V6 rule out proxy tuning for the existing menu: their label-seeing
candidate envelopes add only 3/52 and 2/52 over the best fixed policy. Adding
another importance scorer with the same observation interface is not justified.
A label-free prompt audit identified a different upstream defect. Under the
official LongBench-v2 prompt, the last 32 prompt queries usually cover the end
of choice D plus response-format text; they do not cover the question and all
four choices. The code comment that the protected tail ``holds the question''
does not describe this task interface.

V7 changes that observation once. It compares the old tail-32 score with an
equal-span score over the exact question and A/B/C/D content token spans. It
keeps the model, endpoint, B=2 budget, protected tail, quantizer, tier set,
water-filling rule, and task source fixed. This is a scorer-mechanism test, not
a budget, model, prompt, or proxy sweep.

##### 3L.2 Fresh source partition selected without labels or model output

The pinned 503-row LongBench-v2 source contains too few unused rows at the V6
40,960-token window: only 14 rows in components disjoint from the old 117 pass
all exact checks. A CPU-only Qwen tokenizer audit considered the fixed windows
40,960, 49,152, 65,536, 81,920, 98,304, and 131,072. It used IDs, exact
context/question hashes, rendered-token counts, and prompt hashes only. It did
not inspect answers or generate model output.

At 131,072 tokens, 184 unused complete rows belong to 160 components disjoint
from every old component; 152 of those components are singletons. This is the
first audited window with at least 117 disjoint singleton components (the
98,304-token window has only 105). V7 therefore fixes Qwen's native-supported
131,072-token window and selects exactly 117 singleton rows by one salted
SHA-256 ordering. It takes the first 20 for qualification, the next 52 for
development, and the last 45 for a new untouched confirmation split. Selection
cannot filter on domain, difficulty, length, answer, or any model result. Rows
linked by either exact stripped context or exact stripped question to V4--V6
are excluded transitively before selection.

The V7 manifest must contain no answer. For each selected row it records the
exact rendered prompt hash and five nonempty token-index groups: question, A,
B, C, and D. Token indices are derived from fast-tokenizer character offsets
and must reproduce the exact no-thinking chat-template tokenization. Within a
group of `n` content-overlap tokens, retain all tokens when `n <= 8`; otherwise
retain eight positions `floor(j*(n-1)/7)` for `j=0,...,7`. At scoring time each
of the five groups has weight 1/5 and each retained token shares its group's
weight equally. Thus long choices cannot dominate simply by containing more
tokens. All selected query indices must occur in the cached `input_ids[:-1]`.

##### 3L.3 Clean policy interface and fixed menu

Implement the new score in a separate `structured_policy` module. Its attention
wrapper records weighted attention for the selected absolute query positions,
then delegates the actual attention and compressed-cache behavior to the
existing `sieve_compress` implementation. It must be causal and invariant to
prefill chunk boundaries. V4--V6 sealed sources remain byte-identical so their
artifacts stay independently authenticatable.

The module exposes two allocations from the same structured score:

- `structured_evict`: top-count 8-bit retention using the existing eviction
  allocator;
- `structured_interior`: the existing group water-filling allocator with the
  structured per-query-head importance distribution.

The interior's quantizer-noise curve remains the existing last-prompt-query
curve. V7 changes only the importance observation; it does not silently add a
second noise-model change. The protected last 32 keys remain full precision in
every compressed arm and remain outside the B=2 context budget exactly as in
V6.

The complete ordered development menu is fixed to:

`fp, uniform, tail_evict, tail_interior, structured_evict, structured_interior`.

No SOTA-score variants, calibrated router, output-error oracle, scaffold proxy,
or label-dependent feature is present. The 2 x 2 score-by-allocator comparison
isolates whether structured observation changes either endpoint enough to
create useful task behavior.

##### 3L.4 Qualification and sequential development gates

**Qualification (20 fresh singleton rows).** Run only FP and exact all-2
uniform. Use Qwen3-30B-A3B-2507, no thinking, B=2, W=32, bfloat16, norm
correction, rotation seed 0, maxb 8, and the canonical forced-choice endpoint.
Advance only if all provenance, placement, no-truncation, and bit checks pass,
FP accuracy is at least 0.50 with component-bootstrap q05 strictly above 0.25,
and uniform accuracy lies in [0.30, 0.80] with q05 strictly above 0.25. Use
10,000 `Generator(PCG64(0))` draws and `quantile(..., method="linear")`.
Failure returns `stop_v7_qualification` and closes V7.

**Development (52 fresh singleton rows).** It is lock-authorized only after
qualification. Freeze `F*` once as the most accurate compressed policy in menu
order and define the label-seeing candidate oracle over all five compressed
policies. Competence requires FP accuracy at least 0.50 with q05 above 0.25 and
`F*` accuracy in [0.30, 0.80] with q05 above 0.25. Define

`H = all-candidate-oracle accuracy - F* accuracy`.

Retain V5/V6's opportunity gates: H at least 0.10, H bootstrap q05 strictly
above zero, and H at least 0.05 in each fixed 26-row half.

To attribute any opportunity to the mechanism change, also define the old menu
as `uniform, tail_evict, tail_interior` and

`S = all-candidate-oracle accuracy - old-menu-oracle accuracy`.

Require S at least 0.05. Since adding candidates makes S itemwise nonnegative,
also report its bootstrap q05 and both halves descriptively; do not manufacture
extra pass conditions that are true by construction. Report paired
structured-minus-tail accuracy for eviction and interior, all answer-vector
classes, allocation identities, bits, and selection-free rescue counts.

Apply decisions in this order:

1. competence failure -> `stop_v7_invalid_operating_point`;
2. S below 0.05 -> `stop_v7_no_mechanism_effect`;
3. any H gate failure -> `stop_v7_no_opportunity`;
4. otherwise -> `advance_v7_confirmation` and write one authenticated lock.

There is no G or proxy result in V7. A mechanism must first create candidate
headroom before another selector is designed.

##### 3L.5 One locked confirmation, or stop

Only `advance_v7_confirmation` may open V7's new 45-row confirmation split.
Carry the development-selected `F*` without reselection and keep the exact menu,
settings, S/H definitions, bootstrap, and thresholds. Confirmation must pass
competence, S >= 0.05, H >= 0.10, H q05 > 0, and H >= 0.05 in each fixed half.
Failure closes V7 without tuning. Passing supports a structured-query candidate
mechanism and a nondeployable label-seeing envelope; it does not validate a
router. A later deployable selector would require a separately specified and
fresh external test set.

##### 3L.6 Frozen pre-output implementation ledger

The freeze below completed before any V7 model output. The canonical label-free
manifest contains 20/52/45 singleton components for qualification, development,
and confirmation. Its content SHA-256 is
`9692c79c22888820fed069ba45be93fa32059e35a148f4524530e4c6b4dfaa84` and
its file SHA-256 is
`27757f06ff49ffa232a20e59866e0779689b0cdd5f20db23106cd42c44750a1c`.
The salted within-split ordered-ID hashes are respectively
`bf095ff6ece61d26f7a75a6d13ea6dfb19ce6bedc07ef7e4611b91bb14735834`,
`b613e954483add55fb2e15c2259d6320fb09d9714cf7af8f5018a734ea5b53d2`,
and `7fa48077464830452e66f050ded1be8726bb8ac5cd96c58a48c86de43c5177fc`.
Prompt lengths are 35,100--125,491, 34,089--120,669, and 35,780--130,291
tokens. The manifest contains no answer or model result and binds every prompt
and question/A/B/C/D position plan.

The source audit found 184 unused complete rows in 160 old-disjoint components
at 131,072 tokens, including 152 singleton components; 117 were selected by the
single frozen salted ordering. The source-audit script and report hashes are
`b26dfc0f90f0f024f5e9fb0c04d43bda936cc62b7d7c4eab1e1c3ca8ff5821ee`
and `6c32c8af86296a2f344df5273d2864b77d49ba92f7426505de86c46cf0a9e47a`.

The frozen implementation hashes are:

| file | SHA-256 |
|---|---|
| `h0_measurement/audit_longbench_v2_qwen_v7.py` | `d59426b8e138bea1015f63d08471d50ea46ea31ab82c49588b09e74a4e974093` |
| `sievelib/structured_policy.py` | `a5e3f7d7f41f5c16ef30ef06664cbb02b433345e773beb1c933c7dbb20d3901d` |
| `h0_measurement/run_longbench_v2_qwen_v7_qualification.py` | `dd3213678cc4c127f25044426b67b12e1ed082aa41aa52f46e2e0d803846e099` |
| `read_longbench_v2_qwen_v7_qualification.py` | `08a0c7a2f97348d596fb5537385fd4173b1948b24af5d06f3113100c3fa82233` |
| `submit_longbench_v2_qwen_v7_qualification.slurm` | `8a2b8c2a57347434ebc4664194202b1410e39f620199c70b5365f8be429602d8` |
| `h0_measurement/run_longbench_v2_qwen_v7_development.py` | `0664d6c3cba34d6e29ff3b0f60d45a2c7b69df1e0342e870b20c07adb0186608` |
| `read_longbench_v2_qwen_v7_development.py` | `a9a451969068d147c219a9406065455a88229f4755e256df93eef540b8032100` |
| `submit_longbench_v2_qwen_v7_development.slurm` | `2758d7f564e4541656c1d0f33e6306e834fa8a251a195fbbdf369a2f1827a512` |

The canonical source ledger seals those files, the complete 16-file local
`sievelib` import closure, `run_h0.py`, `run_r8.py`, and `models.yaml`: 26
sources plus the ledger itself at execution. Its content SHA-256 is
`c3dcbcc409df7aafa9ce2b66615e6013c9a7612ac8d5200ee01dac61fdbc8090`;
its file SHA-256 is
`55854620752449321119d21d1b77df00863f66af2195311306618e73e979b81e`.
Both workers authenticate the ledger before tokenizer/model load, and the
runners repeat it before model load and artifact write. Qualification also
freezes the conditional development sources, so a passing outcome cannot be
used to change the mechanism or decision code.

The runtime fails closed unless NumPy is 2.4.2, pandas 3.0.0, PyArrow is absent,
the fastparquet distribution is `2025.12.0+computecanada`, PyTorch is 2.13.0,
Transformers is 5.16.1, and tokenizers is 0.23.1. The snapshot attestation binds
the pinned revision, symlink-target inventory, and pinned metadata hashes; it
does not claim to rehash every weight blob.

Preflight passed 6 manifest tests, 13 qualification contract tests, 10
synthetic-label development contract tests, 5 structured-policy brute-force and
chunk-boundary tests, the complete R8 and R9 fast tensor suites, Python compile,
shell syntax, duplicate-key AST checks, full local-import-closure comparison,
and an independent ledger audit. Production workers run no test or analysis
that joins hidden answers. Artifact rows contain scalar forced-choice summaries
only; the post-job reader joins answers after validating provenance.

The only authorized qualification command was frozen as:

```bash
bash h0_measurement/bugs/9_sota_eviction_baselines/steps.sh \
  --lbv2-v7-qual-submit
```

It submitted only the 20-row FP/uniform qualification on one GPU with a
two-hour limit. The authenticated result was the terminal
`stop_v7_qualification` branch. Because no `advance_v7_development` lock was
written, the separate deterministic development ledger was not created and
the frozen 52-row mechanism job was not submitted.

##### 3L.7 Authenticated qualification outcome and terminal stop (job 984886)

Remote dry-run authentication passed on `trig-login01`. The single frozen
qualification was submitted as Slurm job `984886` on the debug partition with a
two-hour limit. Its only permitted result directory is
`h0_measurement/results/lbv2_v7_qwen_qualification_984886/`; stdout is
`h0_measurement/logs/lbv2v7q_984886.out`. The job completed in 27:54 (GPU step 18:27) with exit code 0:0, and
the strict reader authenticated the dataset, answer-free manifest, exact 20
singleton qualification IDs, model and tokenizer snapshot, no-thinking prompt
template, complete untruncated inputs, CUDA placement, source ledger, exact
all-2 uniform allocations, 40 prediction rows, reciprocal artifacts, and the
completion marker.

| frozen qualification check | result | gate |
|---|---:|---|
| FP forced-choice accuracy | 9/20 = 0.450 | **fail:** required at least 0.50 |
| FP component-bootstrap q05 | 0.250 | **fail:** required strictly above 0.25 |
| uniform B=2 accuracy | 7/20 = 0.350 | pass: within [0.30, 0.80] |
| uniform component-bootstrap q05 | 0.200 | **fail:** required strictly above 0.25 |
| missing, truncated, or invalid rows | 0 | pass |
| strict decision | `stop_v7_qualification` | V7 closed |

This is an operating-point competence failure. The qualification menu contained
only FP and uniform; it collected no tail or structured allocation, score, or
end-task result. It therefore provides no comparison of tail-32 against the
question/A/B/C/D observation and cannot establish whether the proposed
structured-query mechanism helps. The preregistered sequential rule stops
before that question is asked.

No `advance_v7_development` lock was written. Consequently no development
source ledger was created, no 52-row development job was submitted, and no
development-selected `F*`, S, H, or proxy result exists. The new 45-row V7
confirmation partition remains untouched; no confirmation lock or job exists.
The old V4--V6 45-row confirmation partition also remains untouched.

Job 984886 exhausts V7. Do not rerun this 20-row block, relax its gates, choose
a subset after inspecting the outcomes, or use development/confirmation rows
to repair qualification. Any later mechanism study needs a separately justified
operating-point protocol and a new frozen partition; it is not a continuation
or tuned rerun of V7.

Authenticated artifacts:

- `h0_measurement/results/lbv2_v7_qwen_qualification_984886/`
- `h0_measurement/logs/lbv2v7q_984886.out`
- `longbench_v2_v7_qualification_984886.txt`
- `longbench_v2_v7_qualification_984886_summary.csv`
