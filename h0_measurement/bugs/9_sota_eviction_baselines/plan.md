# R9 — four more eviction baselines: Ada-KV, DropKV, OBCache, LaProx

**Implementation audit and run contract:** `audit.md`. **Main-model results:**
`report.md`. All five evaluation cells completed on 2026-09-22.

**Next iteration status (2026-09-23):** Steps 1 and 2 are complete. Held-out
job 980414 initially confirmed Llama 32K/B=2 multikey at k16/v4/h4, but Step 3
development job 981481 then found uniform at 20/20, leaving zero candidate
envelope headroom. The V2-A full-cycle operating-point repair below is now
implemented and locally validated. Jobs 982121 (k24) and 982122 (k32) were
submitted together on 2026-09-23 and are the only V2-A outcome artifacts.

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

##### V2-B: separate expanded-policy development

Only after V2-A freezes `n_keys`, use prompts 540--579. Collect the complete
ordered superset once:

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
intermediate result may use one fresh extension at prompts 580--619 and then a
single pooled n=80 decision; no other repeated extensions are allowed.

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
