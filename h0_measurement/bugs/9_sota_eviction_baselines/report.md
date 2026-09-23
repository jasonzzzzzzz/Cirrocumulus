# R9 main-model baseline results

**Date:** 2026-09-22  
**Status:** all five evaluation cells complete.  
**Full R8/router analysis:** `../8_router_endtask/report.md`.

## Comparison

The completed campaign compares Ada-KV, DropKV, OBCache-K with the Ada-KV
allocator (`obck_ada`), and LaProx against SnapKV (`evict`), uniform key
quantization, the cascade interior, and the calibrated router. Every arm uses
the same prompts, protected window, question-agnostic compression point,
budgets B=2/3, exact values, and total key-bit budget. The runtime bits audit
passes in all five result files.

Qwen multivalue NIAH is excluded because the FP ceiling is only 0.2375 at 8K
and 0.0375 at 32K. This leaves 36 valid model/context/task/budget cells.

| method | valid mean accuracy | delta from SnapKV |
|---|---:|---:|
| uniform quantization | 0.946 | +0.547 |
| OBCache-K + Ada-KV | 0.602 | +0.204 |
| LaProx | 0.579 | +0.181 |
| Ada-KV | 0.503 | +0.105 |
| DropKV | 0.418 | +0.019 |
| SnapKV | 0.398 | reference |

Among the four new arms, OBCache-K + Ada-KV is strongest on Llama (0.719
across 8K/32K/128K), while LaProx is strongest on Qwen (0.457). Uniform remains
the strongest fixed arm in 34/36 task/budget cells. LaProx's Qwen 32K VT result at
B=2 is the substantive exception: 0.79 versus uniform 0.40, SnapKV 0.28, and
OBCache-K + Ada-KV 0.61.

## Interpretation limits

The data support two narrow conclusions:

1. The new score/allocation methods improve the old SnapKV baseline in this
   harness, with the largest average improvements from `obck_ada` and LaProx.
2. Sparse 8-bit retention is usually worse than retaining all keys at 2 or 3
   bits on these retrieval tasks and budgets.

The run does not isolate the OBCache scorer from Ada-KV allocation because only
their combined arm was included. It also omits the DropKV PR-default and LaProx
layer-only ablations. R8 simulates key compression for accuracy; values are
exact and it does not measure packed memory or throughput. These numbers are a
controlled comparison under R8's contract, not a reproduction of the original
papers' system results.

The broad average is ceiling limited: uniform is at least 0.95 in 31/36 valid
cells. A follow-up intended to rank algorithms should first create a regime in
which FP passes and uniform lies between 0.50 and 0.80.

## Does this mean SIEVE is effective?

### Current verdict

The evaluated calibrated router is **not effective enough to support a general
end-task claim**. It is not a safe drop-in policy: it loses clearly to the
strongest fixed arm in 17/36 valid task/budget cells, all on Llama, and routes
87--99% of Llama KV heads to the interior despite uniform quantization having
much higher task accuracy. The predeclared claims that routing never costs
accuracy (P-1) and that its gain is largest in GO cells (P-2) fail.

This does not reject every component called SIEVE. The experiment distinguishes
several claims:

| claim | verdict from this campaign | evidence |
|---|---|---|
| The calibrated SIEVE router is a robust general policy | **fails** | large, clear Llama losses at B=2 and at 128K/B=3 |
| The output-error phase predicts where routing improves task accuracy | **fails on this grid** | phase-band/gain Spearman is -0.70, opposite the predicted sign |
| SIEVE can improve particular end-task cells | **supported narrowly** | Qwen VT gains over uniform are +0.53 at 8K/B=2 and +0.59 at 32K/B=2; Qwen 32K multikey gains +0.25 |
| Output error is a reliable proxy for end-task accuracy | **not validated** | P-4 point estimates are -0.62 to +0.20 for the calibrated router; none establishes the predeclared >0.7 correlation |
| The allocation theorem minimizes its stated output-distortion objective | **not tested by R8 and not contradicted** | R8 asks whether that objective transfers to generated-answer accuracy |
| SIEVE is a memory- or speed-efficient system | **untested** | keys are simulated, values remain exact, and no packed-cache memory or throughput is measured |

The ceiling-limited task regime does not explain away the observed Llama
failures. A ceiling prevents measuring a positive improvement when uniform is
already correct, but it does not cause the router to fall from uniform near
1.00 to 0.23--0.60. It does limit broad method ranking and the precision of a
positive gain estimate.

The scope is also narrower than the full proposed system. R8 measures one-shot
post-prefill key compression, exact values, short retrieval answers, and no
allocation refresh during generation. It therefore rejects the evaluated
router and its downstream phase prediction under this contract; it does not by
itself reject long-generation re-budgeting, the storage format, or every
possible downstream-aware router.

### Why the pending oracle run matters

The current deployable router uses one fixed route per budget, layer, and KV
head, learned by averaging output errors over ten calibration prompts and all
four tasks. The pending `steps.sh --oracle-submit` run evaluates two routers on
new prompts:

- `router_calib`: those fixed offline routes;
- `router_oracle`: for each evaluation prompt and KV head, chooses among
  `interior`, `uniform`, and `evict` using the actual relative attention-output
  errors measured on the first eight FP answer queries.

`router_oracle` is an oracle for the **output-error proxy**, not for task
accuracy. It has unavailable evaluation-prompt information, but still chooses
policies by local output fidelity rather than by whether the final answer is
correct. It diagnoses the failure as follows:

| oracle result | diagnosis | next action |
|---|---|---|
| oracle approximately matches uniform and clearly beats `router_calib` | the proxy has prompt-specific routing information; averaging it into fixed routes loses that information | build a conditional, conservative router and validate it on a third disjoint prompt block |
| oracle is also far below uniform | better offline calibration cannot rescue this formulation; the local proxy, independent per-head composition, or both are misaligned with task accuracy | stop threshold sweeps and revise the objective or composition rule |
| oracle improves only a few points | routing headroom is too small to justify another full grid | report the negative result and redirect effort |

Even oracle failure would apply specifically to the present relative
per-head-output error, eight-query measurement, independent KV-head routing,
and candidate set. It would be strong mechanistic evidence against this
formulation, not a theorem that all representation-error objectives must fail.

## How to improve SIEVE

The oracle result decides which improvement path is justified.

### If the output-error oracle succeeds

The problem is mainly route prediction. Improve it without looking at the final
evaluation block:

1. Condition routes on model, context, budget, and task family instead of
   averaging all tasks into one fixed route.
2. Predict routes from prefill-available features and retain uniform as a
   conservative fallback when the predicted margin is small or uncertain.
3. Select the margin threshold on a validation block separate from both route
   fitting and final evaluation. Report calibration-to-oracle regret.
4. Test whether routing whole layers or whole allocations is more stable than
   independently composing every KV head. Independent local choices need not
   minimize the error of the composed network.
5. Reconsider the candidate set. The current router chooses only among
   `interior`, `uniform`, and SnapKV; it cannot select the stronger R9 policies.
   Add a candidate only where its allocation scope permits a budget-correct
   composition. LaProx's model-wide allocator, for example, cannot simply be
   spliced per head without changing the method.

### If the output-error oracle fails

The objective needs revision before router engineering:

1. Replace or augment local head-output norm with a signal closer to generation,
   such as next-token logit/KL divergence, sequence negative log likelihood, or
   downstream sensitivity from a layer output to final logits.
2. Test answer-relevance weighting. The harness already records the attention
   mass each head places on answer tokens; a small number of retrieval heads may
   matter more than the median head error.
3. Model cross-layer and cross-head interaction. A collection of locally best
   choices can be globally poor after residual mixing and later layers.
4. Re-evaluate the phase claim using the revised downstream-linked statistic.
   If no reliable relationship appears, present the existing phase diagram as
   descriptive of output distortion rather than predictive of task value.

### Improvements required in either branch

1. Use a non-ceiling pilot before another grid. Prefer harder tasks at the valid
   B=2 quantizer width; require FP >=0.95 and uniform accuracy around 0.50--0.80.
   This improves power to detect gains but does not alter the existing losses.
2. Add a provenance-safe task-difficulty interface. Record key/value/hop counts
   in rows, sidecars, route metadata, filenames, completion guards, and reader
   compatibility checks. Do not change hidden task defaults in place.
3. Exclude Qwen multivalue NIAH until its FP ceiling passes. Compression results
   on a task the full-precision model cannot solve are not interpretable.
4. Isolate scorer and allocator effects for the strongest R9 methods: plain
   OBCache-K versus OBCache-K + Ada-KV, and LaProx global versus layer-only.

## What remains to validate if a revised SIEVE looks effective

A revised design should pass gates written before another full campaign:

1. **Safety:** no statistically clear loss to the strongest fixed policy in any
   primary cell.
2. **Value:** positive mean gain with paired confidence intervals on
   non-ceiling cells, not only ties at accuracy 1.00.
3. **Routing quality:** calibrated-router accuracy within a few points of the
   output-error oracle, with routes fitted and tuned on disjoint data.
4. **Objective transfer:** a stable relationship between the routing objective
   and end-task gain across model/context blocks. If this does not hold, remove
   the claim that the phase predicts downstream value.
5. **Generalization:** held-out architectures, context lengths, prompt blocks,
   and task families. The present Llama/Qwen sign reversal makes this necessary.
6. **Runtime behavior:** long generations, allocation refresh, and the
   compressed-observation versus full-observation ablation (P-6).
7. **Full-cache accuracy:** compress values as well as keys, or explicitly scope
   the paper to key-only compression.
8. **System evidence:** packed memory including metadata, kernel throughput,
   latency/TPOT, and comparison at equal realized memory rather than simulated
   key bits alone.

Until those gates pass, the defensible paper statement is that SIEVE's
output-error framework exposes strong model-dependent structure, but its current
calibrated router does not transfer reliably to end-task accuracy. The R9
baseline comparison remains useful under its stated key-bit simulation
contract.
