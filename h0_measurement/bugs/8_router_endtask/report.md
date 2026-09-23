# R8/R9 question-agnostic campaign report

**Run date:** 2026-09-22  
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
| P-5: oracle minus calibrated router is small | oracle was not run | untested |
| P-6: compressed and full observation agree | ablation was not run | untested |

For P-4, prompt split-half reliability is 0.95 for router accuracy gain, so the
weak relationship cannot be dismissed as an obviously unreliable accuracy
measurement. The block bootstrap intervals remain wide with five
model/context blocks. The evidence does not establish the
prespecified transfer claim or the population correlation.

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

## 6. Decision and next run

The current grid is complete. Before extending it, choose a regime that passes
the existing P0 rule:

1. harden the Llama tasks until uniform at B=2 lies in the planned 0.50--0.80
   range, while FP remains at least 0.95; or
2. define and test a budget-matched intermediate uniform baseline between the
   current 1-bit failure and 2-bit ceiling.

Then run a small Llama/Qwen pilot with `uniform`, `interior_cascade`, and
`router_calib`. If the router's architecture split remains, add the oracle arm
before a new full grid; it separates a bad calibration rule from a bad
interior allocation. The completed grid remains useful as a negative result
and as the first main-model comparison of the R9 eviction baselines.

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
