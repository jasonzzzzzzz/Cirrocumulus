# R8/R9 question-agnostic campaign report

**Dates:** main campaign 2026-09-22; non-ceiling follow-up 2026-09-23  
**Jobs audited:** 978479--978489  
**Status:** all five evaluation cells complete. The results are diagnostic
because the campaign did not satisfy the P0 budget-selection gate.

## 1. What completed

All completed evaluations used 20 evaluation prompts starting at offset 100,
four tasks, budgets 2 and 3, question-agnostic compression, and the same nine
arms:

`fp, uniform, evict, interior_cascade, router_calib, adakv, dropkv,
obcache_k:alloc=ada@obck_ada, laprox`.

| model and context | calibration | evaluation | result |
|---|---:|---:|---|
| Llama-3.1-8B, 8K | 978479 | 978480 | complete |
| Llama-3.1-8B, 32K | 978481 | 978483 | complete |
| Llama-3.1-8B, 128K | 978484 | 978485 | complete |
| Qwen3-8B, 8K | 978486 | 978487 | complete |
| Qwen3-8B, 32K | 978488 | 978489 | complete |

Job 978482 has no campaign artifact and is not used. The five completed
evaluation files contain 6,800 rows. Their sidecars agree on the comparison
contract, the reader's bits audit passes with zero overspend, and calibration
and evaluation use disjoint prompt blocks.

## 2. Validity gates

The FP ceiling passes on all twelve Llama model/context/task blocks, including
0.975 on multivalue NIAH at 128K. Qwen's multivalue NIAH is invalid: FP scores
0.2375 at 8K and 0.0375 at 32K. All Qwen multivalue rows are therefore excluded
from every aggregate and claim below. The remaining 36 model/context/task/budget cells are valid.

The budget gate did **not** pass. P0b showed that Llama uniform quantization
jumped from near zero at 1 bit to 0.95--1.00 at 2 bits and explicitly called for
an intermediate regime. The full campaign nevertheless used B=2 and B=3. In
the completed data, uniform is at least 0.95 in 31/36 valid cells. This makes
most comparisons ceiling limited and means this run is not the clean router
test specified in `plan.md` section 5.

## 3. Main result

Equal-weight means over valid tasks, both budgets, and prompts are:

| model and context | uniform | best new SOTA arm | router | router minus strongest fixed arm |
|---|---:|---:|---:|---:|
| Llama-3.1-8B, 8K | 0.998 | OBCache-K + Ada-KV 0.624 | 0.677 | -0.321 |
| Llama-3.1-8B, 32K | 0.996 | OBCache-K + Ada-KV 0.700 | 0.726 | -0.270 |
| Llama-3.1-8B, 128K | 0.962 | OBCache-K + Ada-KV 0.832 | 0.559 | -0.402 |
| Qwen3-8B, 8K | 0.893 | LaProx 0.370 | 0.982 | +0.087 |
| Qwen3-8B, 32K | 0.842 | LaProx 0.543 | 0.997 | +0.090 |

The router is architecture dependent. At B=2 it loses clearly to uniform on
all twelve Llama task/context cells, by 0.25--0.78 accuracy. At 128K it also
loses clearly on all four B=3 tasks, by 0.17--0.40. On Qwen it is near or above
uniform, with the clearest gains on VT: +0.53 at 8K and +0.59 at 32K.

The calibration routes most KV heads to the interior in every cell:

| model and context | B=2 | B=3 |
|---|---:|---:|
| Llama-3.1-8B, 8K | 96% | 87% |
| Llama-3.1-8B, 32K | 96% | 89% |
| Llama-3.1-8B, 128K | 99% | 96% |
| Qwen3-8B, 8K | 89% | 82% |
| Qwen3-8B, 32K | 90% | 83% |

Thus the Llama failure is not a small routing edge case. The output-error
calibration chooses the interior for almost the whole model even though uniform
quantization preserves task accuracy much better.

## 4. Preregistered decisions

| prediction | result on the five complete cells | verdict |
|---|---|---|
| P-1: router is at least the strongest fixed arm within noise | 19/36 valid cells; seventeen clear losses, all on Llama | **fails** |
| P-2: router gain is largest in GO and near zero in STOP | cell gains are -0.321, -0.270, -0.402, +0.087, +0.090 as band falls 54.5 -> 11.5; Spearman = -0.70 | **fails in the opposite direction** |
| P-3: router beats interior everywhere in STOP | router is higher in all 12 valid Qwen task/budget cells | **holds on the two observed Qwen cells** |
| P-4: output-error gain predicts accuracy gain with Spearman >0.7 | router point estimates range from -0.62 to +0.20 by error statistic; no statistic establishes >0.7 | **not supported** |
| P-5: oracle minus calibrated router is small | job 979308: mean +0.298, maximum +0.450; oracle still trails uniform by 0.201 overall | **fails; calibration and proxy both contribute** |
| P-6: compressed and full observation agree | ablation was not run | untested |

For P-4, prompt split-half reliability is 0.95 for router accuracy gain, so the
weak relationship cannot be dismissed as an obviously unreliable accuracy
measurement. The block bootstrap intervals remain wide with five
model/context blocks. The evidence does not establish the
prespecified transfer claim or the population correlation.

### P-5 oracle diagnostic (job 979308)

On ten new Llama 32K/B=2 prompts, the output-error oracle improves mean accuracy
from 0.464 for calibrated routing to 0.761: +0.298 [0.139, 0.459] under a paired
prompt-block bootstrap. Uniform remains higher at 0.963; oracle minus uniform is
-0.201 [-0.324, -0.085]. Thus fixed offline calibration loses substantial
prompt-specific signal, while perfect knowledge of the current local
output-error proxy still does not recover task accuracy.

The proxy mismatch is direct: mean relative head-output error is 0.135 for the
oracle and 0.652 for uniform, yet uniform has higher accuracy. The oracle still
routes 93% of heads to the interior. See
`../9_sota_eviction_baselines/oracle_979308.txt` and that folder's `report.md`
for the task table and design implications.

A separate label-seeing oracle over complete policies has little opportunity in
this ceiling block. Taking the best per-prompt RULER score over uniform, eviction,
and interior yields 0.984 versus uniform 0.963: +0.021 with a 95% prompt-block
interval [0.000, 0.044]. Allowing interior-cascade raises this to only +0.031
[0.000, 0.073]. This supports building the non-ceiling task regime before a
whole-policy router; it is not a deployable routing result.

## 5. What the SOTA arms say

Across the 36 valid task/budget cells:

| arm | mean accuracy | change from SnapKV (`evict`) |
|---|---:|---:|
| uniform | 0.946 | +0.547 |
| router calibrated | 0.766 | +0.367 |
| OBCache-K + Ada-KV | 0.602 | +0.204 |
| LaProx | 0.579 | +0.181 |
| Ada-KV | 0.503 | +0.105 |
| interior cascade | 0.465 | +0.067 |
| DropKV | 0.418 | +0.019 |
| SnapKV | 0.398 | reference |

The new baselines improve substantially over SnapKV on average, especially
OBCache-K + Ada-KV and LaProx, but they do not close the gap to keeping every
token at low precision. Uniform is the strongest fixed arm in 34/36 aggregate
task/budget cells. The exceptions are Qwen VT at B=2: interior cascade is 0.47
versus uniform 0.46 at 8K, and LaProx is 0.79 versus uniform 0.40 at 32K.

The `obck_ada` result is the combined OBCache-K score plus Ada-KV allocator. The
campaign did not run plain OBCache-K, the DropKV code-default variant, or the
LaProx layer-only ablation, so it cannot isolate scorer and allocator effects.

These are matched key-bit accuracy comparisons in the R8 simulator. Values
remain exact, kept keys use 8 bits, and no packed-cache memory or throughput is
measured. They should not be described as reproductions of the papers' reported
end-to-end systems.

## 6. Decision and next design iteration

### Confirmed non-ceiling operating point

The provenance-safe development screen selected multikey at `n_keys=16`.
Cap-fixed follow-ups did not select the other tasks: multivalue v7 is too easy
(FP 0.986, uniform 0.871), VT h7 has an incomplete FP answer at its 112-token
cap, and VT h8 is valid but too easy (FP 1.000, uniform 0.967). Jobs 980356 and
980355 contain those follow-ups.

Held-out job 980414 confirms k16/v4/h4 at Llama 32K/B=2 on prompts 420--439.
All 40 rows and provenance checks pass. FP is 1.000 with no cap hits; uniform is
0.800 with a 90% interval [0.65,0.95]. The paired FP-minus-uniform difference is
+0.200 [0.05,0.35]. Its four uniform failures are uncapped wrong answers, so
the point supplies real but modest recoverable headroom. See
`../9_sota_eviction_baselines/confirmation_980414.txt`. Freeze k16/v4/h4 and
do not tune on the confirmation prompts.

This result validates the task interface and operating point. It does not
validate the existing calibrated router, whose negative result remains
unchanged.

### Ordered next branch

1. On fresh prompts 440--459, run FP plus the complete policies uniform,
   eviction, and interior. Keep the per-head router arms out.
2. Compute the label-seeing complete-policy envelope and the complete-policy
   mean-logit-KL selector over a shared eight-step FP teacher-forced trace.
3. Let `H` be end-task-envelope gain over uniform and `G` be KL-selector
   gain. Require `H >= 0.10` for candidate headroom; require
   `G >= 0.05` and `G/H >= 0.5` for the KL rule to advance.
4. Low `H` triggers candidate-set revision with complete policies. Useful
   `H` with failed `G` rejects mean final-logit KL as the proxy. If both
   pass, lock the candidates and selector and confirm once on prompts 460--499.
5. Build a deployable prefill-only router and expand models or contexts only
   after that locked confirmation passes.

The whole-policy diagnostics are separate artifacts and selectors. They are
not allocation arms or calibration routes. The detailed schema, tests, and
conditional candidate expansions are in the R9 `plan.md` Step 3.

## 7. Reproduction

The statistics above come from:

```bash
OMP_NUM_THREADS=8 .venv/bin/python \
  h0_measurement/bugs/8_router_endtask/read_r8.py \
  h0_measurement/results/r9job978480/r8_llama31-8b_8192.parquet \
  h0_measurement/results/r9job978483/r8_llama31-8b_32768.parquet \
  h0_measurement/results/r9job978485/r8_llama31-8b_131072.parquet \
  h0_measurement/results/r9job978487/r8_qwen3-8b_8192.parquet \
  h0_measurement/results/r9job978489/r8_qwen3-8b_32768.parquet \
  --p2
```
