# R9: K\*-derived KV-head budgets

**Protocol frozen 2026-09-24, before any R9 model output.**  This is roadmap
R9.  The older `bugs/9_sota_eviction_baselines/` directory used the folder
number 9 for the R8 baseline/end-task investigation; it is a separate, closed
branch.  Nothing from its LongBench-v2 labels, prompts, routes, or outcomes is
an input here.

## 1. Why this iteration exists

The legacy E1 diagnostic cannot support the sentence “K\*=100% means every
token in the fractional budget is necessary.” It evaluated the union of 12
geometrically spaced keep counts and the configured absolute-policy count. At
B=3, the actual preceding tested count occasionally reached 97.5% of the full
fractional budget, but only 13.5% of saturated rows used that extra checkpoint.
Its median among saturated rows was 48.5%, 45.6%, 42.8%, 40.2%, and 37.7% at
8K, 16K, 32K, 64K, and 128K, and every configuration's median predecessor
remained the geometric point. Because exact output error is not monotone in keep
count, neither this coarse candidate set nor a binary search identifies the
smallest acceptable count.

A CPU audit of the existing 24 cells also rules out two tempting shortcuts:

* the recorded K\* is per **query head**, while a GQA cache stores one allocation
  per **KV head**;
* taking the maximum query-head K\* inside each GQA group nearly collapses to
  the fixed fractional count: group means are 91.9--99.9% at 32K, with four of
  five GQA models above 98% in the authenticated audit.

Those are stop results for the old statistic and the max aggregation, not for a
KV-group objective that has never been measured.  This iteration asks one
bounded question: does a directly measured per-(layer, KV-head) demand profile
exist, remain stable across calibration prompts, and move enough budget to be a
meaningful allocator?

## 2. Frozen research question

On disjoint calibration and evaluation prompts, can a K\* measured for the
physical KV storage unit redistribute an **exact fixed total** of retained
8-bit context-key tokens and reduce exact attention-output error relative to a
fixed fraction and existing budget allocators?

This is an output-error/cache-allocation experiment.  It uses no answer labels,
does not measure end-task accuracy, and is not a rescue attempt for V7.

## 3. Storage unit, score, and exact loss

### 3.1 Context contract

Qualification uses exactly 8,192 token IDs under the repository's R8
convention: IDs 0..8190 are prefilled and ID 8191 is the held-out next-token
query.  The last `W=32` prefilled tokens are protected in every arm.  Therefore

```
L0 = 8191 cached tokens
C  = L0 - W = 8159 selectable context tokens
B  = 3 bits per selectable token
maxb = 8
k0 = floor(B*C/maxb) = 3059 retained context tokens per KV head
```

Prompt construction uses one frozen `builder_ctx=9000` for every family and
prompt ID, then crops only the earliest text tokens to the exact length while
preserving any leading tokenizer special prefix. This keeps the underlying H0
haystack source paired across families. It must preserve the query suffix and
NIAH needle span, audit one corpus document/offset per prompt ID across
families, and record the builder context, special-prefix count, pre-crop
length, crop offset, final-token SHA-256, corpus provenance, and the four
quantities above. Any row with another live length is invalid.

### 3.2 Fixed ranker

All primary arms use the same canonical SnapKV ranker already shipped in
`sievelib/router.py`: attention received from the 32 observation queries,
pooled across the query heads that share a KV head, then max pooled over context
positions with kernel 7.  The rank order is one order for each physical
`u=(layer, KV head)`.  Fixing the ranker makes this an allocator experiment.

The protected 32-token tail remains full precision and participates in both the
reference and compressed outputs.  A selected context key is quantized by the
existing 8-bit key path and keeps its associated value under the project's
current output-error convention; an unselected context key/value row is absent.

### 3.3 Dense physical-group curve

Let `G` query heads share storage unit `u`.  For every prompt/family observation
row `r`, query head `h`, and integer `k in [1,k0]`, apply the same top-k context
set from the unit's SnapKV order and compute

```
e[r,h,u](k) = ||o_compressed(k) - o_FP||_2 / max(||o_FP||_2, 1e-12)
E_u(k)      = mean_{calibration r,h} e[r,h,u](k)^2 .
```

`o_FP` attends to all `C+W` cached tokens.  `o_compressed(k)` attends to the
selected k context rows plus the common protected tail.  Accumulation is
float64.  The implementation evaluates **every integer k**, using chunked
prefix sums; it must not impose monotonicity or use binary search.

The calibration demand is

```
K*_u = min { k in [1,k0] : E_u(k) <= 1.10 * E_u(k0) + 1e-12 }.
```

For `G=1`, the group curve, K\*, and loss must reduce to the corresponding
single-query-head definition by construction.  This is measured with the MHA
control as well as pinned by unit tests.

## 4. Budget policies and equal-memory contract

Let `U = n_layers*n_kv_heads` and `T=U*k0`. K\*/shrinkage/kappa counts are
integers in `[1,C]`; the shipped layer-flat/global controls retain their
method-faithful `[0,C]` range because they may assign no old-context token to
one KV head. Every equal-memory arm must sum to exactly `T`; largest-remainder
ties are resolved by `(layer, kv_head)` order where the method calls for that
projection. The reader audits both retained-token totals and
`8*sum(counts)/(U*C)`. The common protected tail is outside this comparison.

Primary matched-interface arms, all with the fixed SnapKV ranker and all
calibrated on prompts 0 and 1 before their counts are frozen:

1. **fixed:** the fixed-fraction reference, `k_u=k0`.
2. **kstar_prop:** start from calibration `K*_u`, then deterministically project
   the vector onto the bounded integer simplex with total `T`.
3. **kstar_shrink20:** start from `0.8*K*_u + 0.2*k0`, then use the same exact
   projection. This preregistered shrinkage protects against noisy extremes.
4. **kappa4_rebalanced_calibrated:** first average the six scalar `n95_u`
   measurements from prompts 0--1 and the three families for each storage unit,
   then form the historical `min(k0, max(4*mean_n95_u, 256))` demand and project
   it to total `T`. This is the mean of scalar supports; it is not n95 recomputed
   from an averaged score tensor.
5. **adakv_alpha02_calibrated, alpha=.2:** average calibration SnapKV score tensors,
   then apply the shipped layer-wise allocator once with exact-total rounding.
6. **layer_flat_calibrated** and **snapkv_global_calibrated:** apply the shipped
   allocators once to the same average calibration scores.

Averaging scores and then allocating is frozen; integer count vectors are not
averaged. On held-out prompts every calibrated arm uses its frozen count and
that prompt's own SnapKV token order.

Secondary natural-spend controls are explicit. **kappa4_natural_calibrated**
freezes `min(k0, max(4*mean_n95_u, 256))` from the same six calibration scalar
supports without equal-total projection. **kappa4_natural_dynamic** applies that
historical rule to the current held-out prompt. Both may underspend and neither
can be credited as an equal-memory win or enter a qualification gate.

The other method-faithful dynamic controls are
**kappa4_rebalanced_dynamic**, **adakv_alpha02_dynamic**,
**layer_flat_dynamic**, and **snapkv_global_dynamic**. They derive counts from
each held-out prompt's score/n95 exactly as a deployed prefill-time method does
and are clearly separated from the calibrated controls. Native **LaProx** is
reported separately if its W_O scorer is run, because it changes both scorer
and allocator; its absence cannot fail a qualification gate.

`n95_u` is computed from the same unpooled group-summed observation-window
attention before positional max pooling. No end-task result or legacy
query-head K\* enters a count.

## 5. Qualification: the only run currently authorized

### 5.1 Cells and split

Two independent one-GPU jobs, B=3 and exact 8K:

| model | grouping role | prompt IDs | families |
|---|---|---:|---|
| `llama31-8b` | GQA, `G=4`; the actual physical-group test | 0,1 calibration; 2,3 held out | `niah,qa,cont` |
| `qwen15-moe-a2.7b` | MHA, `G=1`; reduction control | 0,1 calibration; 2,3 held out | `niah,qa,cont` |

Prompt IDs 0 and 1 are used only to build demand profiles. Stability is prompt
0 versus prompt 1, with the three families averaged within each prompt. Primary
counts are derived from the average of both prompts and then frozen. On IDs 2
and 3 each matched-interface arm gets the prompt's own fixed SnapKV token
ranking but its calibrated per-unit count; no held-out curve, loss, or answer
may change those counts. The distinctly named dynamic controls may use only
that held-out prompt's score, as their native method requires. Families share
their prompt-ID haystack as in H0.

### 5.2 Frozen qualification gates

The strict reader applies these in order.  A numerical/provenance failure makes
the artifact invalid and permits a repaired rerun on the same split.  A valid
scientific failure returns `stop_r9_qualification`; thresholds are not relaxed
and the split is not rerun.

* **Q1 — implementation and budget:** both artifacts pass source/provenance,
  schema, finite-value, prompt-length, count-bound, exact-total, and bit audits.
  On the Qwen MHA control, the dense group curve matches the single-head curve
  with maximum absolute difference `<=1e-10` and yields identical K\*.
* **Q2 — calibration stability:** on Llama, Spearman correlation across all
  physical storage units between profiles built independently from prompt 0
  and prompt 1 (each averaged over all three families and observation queries)
  is finite and `>=0.50`.
* **Q3 — nontrivial redistribution:** for Llama `kstar_prop`,
  `sum_u |k_u-k0|/(2*T) >= 0.05`.
* **Q4 — resolution:** fewer than 95% of Llama units have calibration
  `K*_u == k0`.

All four gates must pass to write an authenticated `advance_development` lock.
Held-out output errors are reported with paired prompt/family summaries, but
they do not alter these feasibility gates and do not select a policy at this
stage.  This keeps qualification from becoming an unregistered performance
screen.

## 6. Conditional development and confirmation

These partitions remain closed until the qualification lock exists.

### 6.1 Development, conditional

Use `llama31-8b`, `qwen3-8b`, and `qwen15-moe-a2.7b` at 32K on fresh prompt
IDs fixed in the advance lock.  Compare every equal-memory arm in section 4.
Select between `kstar_prop` and `kstar_shrink20` by mean held-out exact squared
relative output error only, matching the frozen group loss. Advance one policy
only if its equal-cell geometric mean error ratio to `fixed` is `<=0.97` and no
model/family cell exceeds
`1.10`.  Primary inference uses paired log error, equal model/family-cell
weight, and a prompt-cluster bootstrap.  No threshold or candidate is added
after seeing development output.

### 6.2 Confirmation, conditional

Run the selected policy once on fresh prompts and include a fresh GQA-8
architecture (`llama33-70b`) to test transfer.  Confirmation reports the frozen
paired-log-error endpoint, exact memory audit, and per-cell harms.  It does not
reselect the policy.  The precise prompt block, sample count, and uncertainty
gate must be written here and source-sealed before any confirmation output.

## 7. Clean implementation boundary

R9 is additive.  It must not modify the sealed V7 source set or the existing
`run_h0.py`, `run_r8.py`, `models.yaml`, or imported `sievelib` modules.

New components:

* `sievelib/group_error_curve.py`: dense exact group curve and exact K\* only;
* `sievelib/budget_policies.py`: bounded deterministic count projection and the
  K\*/kappa control policies only;
* `h0_measurement/run_kstar_budget_qualification.py`: the isolated runner;
* `h0_measurement/submit_kstar_budget_qualification.slurm`: qualification-only
  worker;
* this directory's audit, strict reader, source ledger, report, and `steps.sh`.

The runner writes no raw logits, raw K/V tensors, labels, or generated answers.
It writes scalar losses, counts/profiles, prompt hashes, the exact effective
configuration, model snapshot identity, package versions, and SHA-256 hashes of
the complete import closure. The immutable qualification input manifest hashes
all 40 PG19 files and every byte in both pinned model snapshots. The source
ledger seals the runner, worker, strict reader, command interface, full local
import closure, the exact Lloyd-Max cache, and
`qualification_protocol_frozen.md`, a pre-output copy of this protocol. This
working `plan.md` remains outside that ledger so its status can be updated after
the result without invalidating qualification authentication.

The runner also records the exact 8-bit level tensor, random rotation, CUDA
device/runtime, and math-mode settings. The worker refuses an existing result
directory, runs the strict reader in task-validation mode before writing
`COMPLETE`, and the offline reader authenticates `COMPLETE` before making a
decision.

## 8. Stop rules

Stop R9 without a development campaign if any valid qualification gate fails.
In particular, do not substitute query-head K\*, max-pool query-head demands,
smooth the curve after inspection, widen 10%, lower the stability threshold, or
search another budget/context on the qualification prompts.  A stopped R9 is a
useful result: the apparent tail-demand signal was a coarse-grid diagnostic and
does not form a stable physical-cache budget rule.

