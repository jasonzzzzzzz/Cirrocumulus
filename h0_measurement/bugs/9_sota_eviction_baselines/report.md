# R9 main-model baseline results

**Dates:** main campaign 2026-09-22; design follow-ups through 2026-09-24  
**Status:** all five evaluation cells complete. The first whole-policy
implementation run (job 981481) is authenticated, but its fresh block returned
to a uniform ceiling. V2-A jobs 982121/982122, V2-A2 job 982613, and final
V2-A3 job 982702 are authenticated. None selects a stable synthetic non-ceiling
operating point, so V2-B remains blocked. V3 job 983199 ended `stop_panel`, V4
job 983715 ended `stop_v4`, and V5 forced-choice development job 983888 ended
`stop_no_opportunity`. V5 confirmation remains untouched and closed. V6 Qwen
qualification job 984224 passed, but development job 984370 ended
`stop_v6_invalid_operating_point`: FP missed its frozen competence floor and
observed oracle headroom was only 2/52. V6 confirmation remains untouched and
closed. V7 qualification job 984886 ended `stop_v7_qualification`: FP was
9/20 = 0.450 and uniform was 7/20 = 0.350, with neither strict bootstrap lower
bound above 0.25. V7 development and confirmation were never opened.  
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

| method | valid mean task score | delta from SnapKV | mean relative head-output error (lower is better) |
|---|---:|---:|---:|
| full precision | 0.999 | +0.600 | -- |
| uniform quantization | 0.946 | +0.547 | 0.445 |
| SIEVE calibrated router | 0.766 | +0.367 | **0.120** |
| OBCache-K + Ada-KV | 0.602 | +0.204 | 0.260 |
| LaProx | 0.579 | +0.181 | 0.264 |
| Ada-KV | 0.503 | +0.105 | 0.258 |
| SIEVE cascade | 0.465 | +0.067 | 0.133 |
| DropKV | 0.418 | +0.019 | 0.271 |
| SnapKV | 0.398 | reference | 0.261 |

Among the four new arms, OBCache-K + Ada-KV is strongest on Llama (0.719
across 8K/32K/128K), while LaProx is strongest on Qwen (0.457). Including the
fixed SIEVE cascade, uniform is strictly strongest in 32/36 cells and tied for
strongest in two more. The two losses are Qwen VT at B=2: cascade scores 0.47
versus uniform 0.46 at 8K, and LaProx scores 0.79 versus uniform 0.40 at 32K.
Against the external baseline set alone, uniform is therefore at least tied for
strongest in 35/36 cells.

### Per-configuration end-task scores

Each number is the arithmetic mean RULER task score over the 20 held-out
prompts 100--119. The score is `hits / n_expected`: single and multikey are
exact-hit accuracy, while multivalue and VT permit fractional partial credit.
FP is the B=0 competence ceiling and is repeated beside B=2 and B=3 for
comparison. The five source jobs are 978480 (Llama 8K), 978483 (Llama 32K),
978485 (Llama 128K), 978487 (Qwen 8K), and 978489 (Qwen 32K). All compressed
arms use the same question-agnostic point, exact values, protected W=32 tail,
and matched total key-bit budget.

| Model | Context | Task | B | FP | Uniform | H2O | SnapKV | SIEVE cascade | SIEVE router | AdaKV | DropKV | OBCache-K + AdaKV | LaProx |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 8K | multikey | 2 | 1.000 | 1.000 | — | 0.200 | 0.350 | 0.300 | 0.250 | 0.300 | 0.400 | 0.250 |
| Llama-3.1-8B | 8K | multikey | 3 | 1.000 | 1.000 | — | 0.350 | 0.600 | 1.000 | 0.550 | 0.550 | 0.750 | 0.550 |
| Llama-3.1-8B | 8K | multivalue | 2 | 1.000 | 1.000 | — | 0.138 | 0.150 | 0.225 | 0.188 | 0.163 | 0.388 | 0.250 |
| Llama-3.1-8B | 8K | multivalue | 3 | 1.000 | 1.000 | — | 0.338 | 0.400 | 0.988 | 0.425 | 0.425 | 0.588 | 0.463 |
| Llama-3.1-8B | 8K | single | 2 | 1.000 | 1.000 | — | 0.450 | 0.300 | 0.350 | 0.500 | 0.550 | 0.700 | 0.600 |
| Llama-3.1-8B | 8K | single | 3 | 1.000 | 1.000 | — | 0.750 | 0.600 | 1.000 | 0.750 | 0.650 | 0.750 | 0.800 |
| Llama-3.1-8B | 8K | VT | 2 | 1.000 | 0.980 | — | 0.450 | 0.430 | 0.600 | 0.420 | 0.490 | 0.540 | 0.400 |
| Llama-3.1-8B | 8K | VT | 3 | 1.000 | 1.000 | — | 0.600 | 0.670 | 0.950 | 0.620 | 0.640 | 0.880 | 0.590 |
| Llama-3.1-8B | 32K | multikey | 2 | 1.000 | 1.000 | — | 0.300 | 0.400 | 0.450 | 0.350 | 0.200 | 0.550 | 0.450 |
| Llama-3.1-8B | 32K | multikey | 3 | 1.000 | 1.000 | — | 0.450 | 0.600 | 1.000 | 0.700 | 0.350 | 0.700 | 0.700 |
| Llama-3.1-8B | 32K | multivalue | 2 | 1.000 | 0.975 | — | 0.125 | 0.163 | 0.225 | 0.200 | 0.062 | 0.350 | 0.263 |
| Llama-3.1-8B | 32K | multivalue | 3 | 1.000 | 1.000 | — | 0.425 | 0.400 | 0.900 | 0.562 | 0.225 | 0.700 | 0.613 |
| Llama-3.1-8B | 32K | single | 2 | 1.000 | 1.000 | — | 0.600 | 0.550 | 0.600 | 0.650 | 0.350 | 0.850 | 0.750 |
| Llama-3.1-8B | 32K | single | 3 | 1.000 | 1.000 | — | 0.700 | 0.650 | 1.000 | 0.950 | 0.600 | 0.900 | 0.950 |
| Llama-3.1-8B | 32K | VT | 2 | 1.000 | 0.990 | — | 0.280 | 0.460 | 0.630 | 0.460 | 0.370 | 0.690 | 0.470 |
| Llama-3.1-8B | 32K | VT | 3 | 1.000 | 1.000 | — | 0.650 | 0.640 | 1.000 | 0.820 | 0.550 | 0.860 | 0.870 |
| Llama-3.1-8B | 128K | multikey | 2 | 1.000 | 1.000 | — | 0.300 | 0.350 | 0.400 | 0.500 | 0.400 | 0.650 | 0.600 |
| Llama-3.1-8B | 128K | multikey | 3 | 1.000 | 0.950 | — | 0.450 | 0.500 | 0.550 | 0.800 | 0.500 | 0.850 | 0.700 |
| Llama-3.1-8B | 128K | multivalue | 2 | 0.975 | 0.938 | — | 0.412 | 0.338 | 0.338 | 0.575 | 0.350 | 0.637 | 0.588 |
| Llama-3.1-8B | 128K | multivalue | 3 | 0.975 | 0.988 | — | 0.537 | 0.550 | 0.688 | 0.787 | 0.550 | 0.850 | 0.812 |
| Llama-3.1-8B | 128K | single | 2 | 1.000 | 1.000 | — | 0.900 | 0.450 | 0.450 | 1.000 | 0.750 | 0.950 | 0.950 |
| Llama-3.1-8B | 128K | single | 3 | 1.000 | 1.000 | — | 0.950 | 0.650 | 0.650 | 1.000 | 0.850 | 0.950 | 1.000 |
| Llama-3.1-8B | 128K | VT | 2 | 1.000 | 0.830 | — | 0.480 | 0.500 | 0.580 | 0.740 | 0.420 | 0.810 | 0.810 |
| Llama-3.1-8B | 128K | VT | 3 | 1.000 | 0.990 | — | 0.670 | 0.790 | 0.820 | 0.900 | 0.730 | 0.960 | 0.940 |
| Qwen3-8B | 8K | multikey | 2 | 1.000 | 0.950 | — | 0.000 | 0.150 | 0.900 | 0.000 | 0.050 | 0.050 | 0.050 |
| Qwen3-8B | 8K | multikey | 3 | 1.000 | 1.000 | — | 0.100 | 0.500 | 1.000 | 0.150 | 0.200 | 0.100 | 0.300 |
| Qwen3-8B | 8K | multivalue † | 2 | 0.2375 | 0.6875 | — | 0.000 | 0.050 | 0.725 | 0.000 | 0.050 | 0.0125 | 0.0125 |
| Qwen3-8B | 8K | multivalue † | 3 | 0.2375 | 0.4875 | — | 0.000 | 0.275 | 0.5875 | 0.000 | 0.075 | 0.100 | 0.2125 |
| Qwen3-8B | 8K | single | 2 | 1.000 | 0.950 | — | 0.100 | 0.200 | 1.000 | 0.050 | 0.100 | 0.150 | 0.250 |
| Qwen3-8B | 8K | single | 3 | 1.000 | 1.000 | — | 0.250 | 0.600 | 1.000 | 0.250 | 0.350 | 0.400 | 0.550 |
| Qwen3-8B | 8K | VT | 2 | 1.000 | 0.460 | — | 0.190 | 0.470 | 0.990 | 0.280 | 0.280 | 0.430 | 0.410 |
| Qwen3-8B | 8K | VT | 3 | 1.000 | 1.000 | — | 0.630 | 0.810 | 1.000 | 0.620 | 0.630 | 0.680 | 0.660 |
| Qwen3-8B | 32K | multikey | 2 | 1.000 | 0.750 | — | 0.000 | 0.100 | 1.000 | 0.050 | 0.150 | 0.150 | 0.150 |
| Qwen3-8B | 32K | multikey | 3 | 1.000 | 1.000 | — | 0.100 | 0.400 | 1.000 | 0.100 | 0.300 | 0.250 | 0.300 |
| Qwen3-8B | 32K | multivalue † | 2 | 0.0375 | 0.750 | — | 0.000 | 0.0125 | 0.125 | 0.000 | 0.000 | 0.000 | 0.050 |
| Qwen3-8B | 32K | multivalue † | 3 | 0.0375 | 0.425 | — | 0.000 | 0.000 | 0.075 | 0.000 | 0.0375 | 0.000 | 0.025 |
| Qwen3-8B | 32K | single | 2 | 1.000 | 0.950 | — | 0.200 | 0.350 | 1.000 | 0.200 | 0.300 | 0.250 | 0.400 |
| Qwen3-8B | 32K | single | 3 | 1.000 | 1.000 | — | 0.300 | 0.450 | 1.000 | 0.500 | 0.550 | 0.550 | 0.700 |
| Qwen3-8B | 32K | VT | 2 | 1.000 | 0.400 | — | 0.280 | 0.460 | 0.990 | 0.450 | 0.400 | 0.610 | 0.790 |
| Qwen3-8B | 32K | VT | 3 | 1.000 | 0.950 | — | 0.690 | 0.760 | 0.990 | 0.770 | 0.700 | 0.810 | 0.920 |

† Invalid task-score cell: Qwen FP competence is only 0.2375 at 8K and 0.0375 at 32K. Exclude all four Qwen-multivalue budget cells from aggregate claims. H2O is an em dash because it was not run in these five held-out files.

The `H2O` column is deliberately empty: `evict_h2o` was not an arm in these
five held-out jobs. Historical H2O jobs used prompts 0--19, question-aware
compression, and the older 31-query capture contract, so their task scores
cannot be inserted into this paired table. A matched H2O rerun is still
missing.

### Budget and output-error measurements

The companion measurements below use exactly the same valid cells: 18 cells
and 360 prompt rows for each method at each budget. Head-output error is the
relative attention-output error per query head, averaged over the first eight
FP answer queries, heads, prompts, and cells. Evicted fraction and audited bits
per token come from the runtime result rows.

| method | B | mean task score | mean head-output error (lower is better) | mean evicted fraction | mean audited key bits/token |
|---|---:|---:|---:|---:|---:|
| uniform | 2 | 0.898 | 0.679 | 0.0% | 2.0000 |
| uniform | 3 | 0.993 | 0.212 | 0.0% | 3.0000 |
| SnapKV | 2 | 0.300 | 0.308 | 75.0% | 1.9998 |
| SnapKV | 3 | 0.497 | 0.213 | 62.5% | 2.9998 |
| SIEVE cascade | 2 | 0.343 | 0.168 | 63.0% | 2.0000 |
| SIEVE cascade | 3 | 0.587 | 0.098 | 47.6% | 3.0000 |
| SIEVE router | 2 | 0.613 | **0.158** | 59.9% | 2.0000 |
| SIEVE router | 3 | 0.919 | **0.083** | 42.3% | 3.0000 |
| AdaKV | 2 | 0.381 | 0.308 | 75.0% | 1.9998 |
| AdaKV | 3 | 0.625 | 0.208 | 62.5% | 2.9998 |
| DropKV | 2 | 0.316 | 0.320 | 75.0% | 1.9998 |
| DropKV | 3 | 0.519 | 0.223 | 62.5% | 2.9998 |
| OBCache-K + Ada-KV | 2 | 0.509 | 0.311 | 75.0% | 1.9998 |
| OBCache-K + Ada-KV | 3 | 0.696 | 0.210 | 62.5% | 2.9998 |
| LaProx | 2 | 0.468 | 0.316 | 75.0% | 1.9998 |
| LaProx | 3 | 0.690 | 0.212 | 62.5% | 2.9998 |

Every compressed arm stays within B. These are logical key-bit and sparsity
measurements from the simulator, not packed-memory or latency measurements.
The router has the lowest measured output error at both budgets, yet uniform
has the highest end-task score. The output-error metric therefore diagnoses
representation distortion; it does not establish downstream accuracy.

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

### Oracle diagnosis: both calibration and proxy transfer fail

Job 979308 completed the P-5 diagnostic on Llama-3.1-8B at 32K and B=2:
ten new prompts (300--309), disjoint from route calibration (0--9) and the main
evaluation (100--119), with FP, uniform, eviction, both interiors, the calibrated
router, and the output-error oracle. All task FP ceilings pass and the bits audit
has zero overspend.

| task | uniform | calibrated router | output-error oracle | oracle minus calibrated | oracle minus uniform |
|---|---:|---:|---:|---:|---:|
| multikey NIAH | 1.000 | 0.300 | 0.600 | +0.300 | -0.400 |
| multivalue NIAH | 0.950 | 0.175 | 0.625 | +0.450 | -0.325 |
| single NIAH | 1.000 | 0.600 | 0.900 | +0.300 | -0.100 |
| variable tracking | 0.900 | 0.780 | 0.920 | +0.140 | +0.020 |
| **equal-task mean** | **0.963** | **0.464** | **0.761** | **+0.298** | **-0.201** |

A paired bootstrap that resamples whole prompt blocks gives oracle minus
calibrated **+0.298 [0.139, 0.459]** and oracle minus uniform
**-0.201 [-0.324, -0.085]**. The oracle therefore recovers about 60% of the
calibrated router's 0.499-point deficit to uniform, while a substantial and
statistically clear 0.201-point deficit remains. P-5 predicted an oracle gap of
only a few points; its mean gap is 0.298 and its largest task gap is 0.450, so
**P-5 fails**.

The failure is especially informative because the oracle achieves the objective
it is given. Averaged over all measured heads, its relative output error is
0.135, compared with 0.142 for the calibrated router and 0.652 for uniform.
Nevertheless, uniform accuracy is 0.963 and oracle accuracy is 0.761. The
answer-attention-weighted error has the same ordering, so that existing weighting
does not repair the mismatch on this block.

Both parts of the design need attention:

1. **Calibration loses real prompt-specific information.** Oracle routing gains
   0.298 over fixed offline routes. Averaging ten prompts and four tasks into one
   route per model/context/budget/layer/head is too coarse.
2. **The local output-error proxy remains insufficient for task accuracy.** The
   oracle has access to the actual evaluation-prompt FP answer queries and still
   loses clearly to uniform on multikey and multivalue NIAH and on the overall
   mean. It routes 93% of KV heads to the interior, versus 96% for calibrated
   routing, even though task accuracy usually favors uniform.

`router_oracle` is still an oracle only for the present proxy. It independently
chooses among `interior`, `uniform`, and `evict` per KV head using relative
attention-output error over eight FP answer queries. It does not search policy
combinations for the final answer score. The result is strong evidence against
this proxy-and-composition formulation; it is not a statement that every
representation-error objective must fail.

### Complete-policy end-task envelope: little headroom in the easy block

A second diagnostic can be computed from job 979308 without another GPU run.
For each prompt and task, the **end-task policy oracle** takes the largest stored
RULER score among complete, budget-matched policies. It sees the answer score,
retains ties, and is therefore a label-seeing opportunity bound rather than a
deployable router.

| task | uniform | best of uniform/evict/interior | also allowing interior-cascade |
|---|---:|---:|---:|
| multikey NIAH | 1.000 | 1.000 | 1.000 |
| multivalue NIAH | 0.950 | 0.975 | 0.975 |
| single NIAH | 1.000 | 1.000 | 1.000 |
| variable tracking | 0.900 | 0.960 | 1.000 |
| **equal-task mean** | **0.963** | **0.984** | **0.994** |

The base three-policy envelope gains only **+0.021** over uniform, with a 95%
prompt-block bootstrap interval **[0.000, 0.044]**. Adding interior-cascade raises
the point estimate to **+0.031 [0.000, 0.073]**. Most prompt blocks have no gain
because uniform is already correct. Thus this block has too little complete-policy
headroom to develop or judge a new router. It motivates the non-ceiling task
screen; it does not show that a policy-logit selector will achieve these bounds.

The planned policy-logit oracle is a different diagnostic. It will select a
complete policy by full-vocabulary KL to FP on one shared FP teacher-forced
trajectory, without seeing task labels. Keeping the two selectors separate will
show whether failure comes from the downstream proxy or from the candidate set.

### Non-ceiling screen: multikey succeeds; values and hops hit the FP boundary

Jobs 980284 and 980285 screened the preregistered difficulty levels on prompts
400--409 at Llama 32K/B=2, question-agnostic, with FP and uniform only. Both
jobs completed all expected rows, used corpus SHA `0a26bc1e05a1eea8`, and spent
at most exactly 2 bits per token.

| configuration | task | FP | uniform B=2 | gate |
|---|---|---:|---:|---|
| k8/v6/h6 | multikey | 1.000 | 0.900 | too easy |
| k8/v6/h6 | multivalue | 1.000 | 1.000 | too easy |
| k8/v6/h6 | variable tracking | 1.000 | 0.943 | too easy |
| k16/v8/h8 | multikey | 1.000 | 0.800 | **passes** |
| k16/v8/h8 | multivalue | 0.900 | 0.738 | invalid: FP < 0.95 |
| k16/v8/h8 | variable tracking | 0.911 | 0.933 | **censored:** all incomplete FP rows hit 64 tokens |

Thus `n_keys=16` supplies a valid development operating point for multikey:
uncompressed accuracy remains 1.00 while uniform reaches the upper edge of the
planned 0.50--0.80 band. The held-out confirmation reported below passes.
Multivalue v8 cannot be used even though uniform falls into the target range:
its FP mean is 0.90, and the failed FP response stopped after one token rather
than at the generation cap. The VT conclusion is different. Every incomplete
h8 FP response has `gen_len=64`, so the apparent 0.911 ceiling is a harness
censoring error rather than evidence that the model cannot solve h8.

The next bounded work was preregistered before rerun. A difficulty-aware answer
contract preserves the historical limits at 4, then adds 8 tokens per extra
value and 16 per extra VT hop. It records the limit and cap event per row. The
cap-fixed k16/v7/h7 job tests multivalue and VT with limits 88/112; a separate
h8 VT job uses 128. Multivalue is accepted only if FP >=0.95 and uniform reaches
the target band. VT is judged only after incomplete FP outputs no longer reach
the cap. A harder multivalue extension is not justified after genuine FP failure
at v8.

### Cap-fixed development result: only multikey advances

Jobs 980356 and 980355 completed the preregistered cap-fixed follow-up. Their
sidecars record `difficulty_v1`, the exact limits below, the real corpus SHA,
and all 60 expected rows; the B=2 audit passes.

| configuration | task | limit | FP | uniform B=2 | decision |
|---|---|---:|---:|---:|---|
| k16/v7/h7 | multivalue | 88 | 0.986 | 0.871 | too easy |
| k16/v7/h7 | variable tracking | 112 | 0.988 | 0.975 | invalid: incomplete FP at cap |
| k16/v7/h8 | variable tracking | 128 | 1.000 | 0.967 | too easy |

For multivalue, the only incomplete FP answer stops before the cap, while its
capped FP answer already contains all expected values. Six uniform answers hit
the cap; extending them can only recover more expected values, so uniform's
true score cannot move down into the target band. At VT h7, every answer reaches
112 tokens and one FP answer contains only seven of eight variables, so that
cell remains censored. At VT h8, all FP answers already contain all nine
expected variables. Uniform also reaches the cap on every prompt, but its two
incomplete answers could only increase its already high 0.967 score.

The development block therefore selects only multikey at `n_keys=16`.
Values and hops return to their canonical defaults because those knobs do not
affect the multikey task.

### Held-out confirmation passes

Job 980414 confirms k16/v4/h4 on prompts 420--439:

| check | result |
|---|---:|
| expected rows | 40/40 |
| FP | 1.000 (20/20) |
| uniform B=2 | 0.800 (16/20) |
| uniform 90% prompt-bootstrap interval | [0.65, 0.95] |
| paired FP minus uniform, 90% interval | +0.200 [0.05, 0.35] |
| incomplete capped FP answers | 0 |
| uniform bit audit | exactly 2.0 bits/token |

The one capped uniform answer already scores 1.0. All four uniform failures are
uncapped six-token wrong answers; three name a distractor. Thus this is a clean
retention-failure regime rather than decoding truncation. It passes at the
prespecified upper boundary and supplies 20 points of recoverable accuracy, but
only four uniform failures at n=20. Treat it as modest headroom, with 0.05 score
granularity, rather than a precise estimate.

The exact reader output is
[confirmation_980414.txt](confirmation_980414.txt), and the source artifact is
`h0_measurement/results/r8confirm_k16_v4_h4_980414/`. Freeze this task
configuration and do not tune on prompts 420--439. This validates the harder-task
interface and operating point; it does not establish that SIEVE or a new router
is effective.

### Next iteration: complete-policy headroom, then proxy quality

Use fresh prompts 440--459 at the confirmed cell. Run FP plus the primary
complete candidates uniform, eviction, and interior. Compute two distinct
diagnostics: the label-seeing end-task envelope and the lowest mean
full-vocabulary KL policy on the shared first eight FP teacher-forced decisions.
Join the latter to each policy's independently greedy task score.

Predeclared development quantities are `H`, the end-task-envelope gain over
uniform, and `G`, the mean-KL selector gain over uniform. Useful candidate
opportunity requires `H >= 0.10`. Mean KL advances only with
`G >= 0.05` and `G/H >= 0.5`. Low `H` triggers nested complete-policy
expansion; useful `H` with failed `G` is evidence against final-logit KL;
both passing trigger a locked 40-prompt confirmation on prompts 460--499. Do
not launch a broad model/context grid before that confirmation. The precise
metrics, artifact boundary, tests, and conditional candidate set are frozen in
`plan.md` Step 3.

## How to improve SIEVE after the oracle result

### Priority 1: connect the objective to task-relevant computation

Threshold tuning alone cannot establish this. Uniform has roughly five times
the oracle's mean local output error while producing higher accuracy, so the
magnitude of local representation distortion is poorly ordered with retrieval
success.

1. Measure next-token logit or KL divergence, sequence negative log likelihood,
   or downstream sensitivity from a layer/head output to final logits.
2. Focus on rare task-critical failures rather than the mean head. Test tail and
   worst-group objectives and identify which heads retain the requested values
   or variable-chain links.
3. Improve answer relevance beyond raw attention mass. The recorded
   answer-mass-weighted error still ranks the oracle ahead of uniform while task
   accuracy ranks them oppositely.
4. Model cross-head and cross-layer interaction. Independently selecting the
   smallest-error allocation for each KV head need not minimize the composed
   network's error after residual mixing and later layers.
5. Rebuild the phase analysis on any revised downstream-linked statistic. If no
   stable relationship appears, scope the existing phase diagram to output
   distortion and remove its downstream-prediction claim.

### Priority 2: retain the useful prompt-specific routing signal

The +0.298 oracle gain shows that calibration also leaves substantial value on
the table. After choosing a better objective:

1. Condition routes on model, context, budget, and task family instead of
   averaging all tasks into one fixed route.
2. Predict routes from prefill-available features and fall back to uniform when
   the predicted gain is small or uncertain.
3. Fit routes on one prompt block, choose thresholds on a second block, and
   report once on a third block. Report calibration-to-oracle regret.
4. Test whole-layer or whole-allocation routing against independent KV-head
   composition.
5. Reconsider the candidate set. The current router cannot choose the stronger
   R9 policies. Add a candidate only when its allocation scope permits a
   budget-correct composition; LaProx's model-wide allocation cannot be spliced
   per head without changing the method.

### Experimental requirements before another full grid

1. Use a non-ceiling pilot. Prefer harder tasks at the valid B=2 quantizer width;
   require FP >=0.95 and uniform accuracy around 0.50--0.80. This improves power
   to detect gains but does not alter the existing losses.
2. **Completed 2026-09-23:** the provenance-safe task-difficulty interface
   records key/value/hop counts in rows, sidecars, route metadata, filenames,
   completion guards, and reader identities while preserving legacy 4/4/4.
3. Exclude Qwen multivalue NIAH until its FP ceiling passes.
4. Isolate scorer and allocator effects for the strongest R9 methods: plain
   OBCache-K versus OBCache-K + Ada-KV, and LaProx global versus layer-only.
5. Start with one Llama 32K diagnostic cell. Expand only if the revised proxy
   orders uniform and the routed policy consistently with task accuracy.

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

## Whole-policy diagnostic: development job 981481

Job 981481 is the first implementation of the Step 3 whole-policy diagnostic.
It uses Llama-3.1-8B at 32K, question-agnostic multikey NIAH at k16/v4/h4,
B=2, and fresh prompts 440--459. The accuracy artifact has exactly 80 rows for
`fp`, `uniform`, `evict`, and `interior`; the separate diagnostic artifact has
exactly 60 rows for the three non-FP candidates. Both sidecars, the linked
accuracy SHA-256, real-corpus identity, row keys, candidate order, shared trace
hashes, bit budgets, and absence of raw logits pass the strict reader.

| quantity | result |
|---|---:|
| FP accuracy | 1.000 (20/20) |
| uniform accuracy | 1.000 (20/20) |
| eviction accuracy | 0.250 (5/20) |
| interior accuracy | 0.300 (6/20) |
| end-task envelope | 1.000 |
| `H = envelope - uniform` | **0.000 [0.000, 0.000]** |
| mean-KL-selected accuracy | 1.000 |
| `G = KL-selected - uniform` | **0.000 [0.000, 0.000]** |
| mean-KL selections | uniform 18, eviction 0, interior 2 |
| selector in end-task-optimal tie | 20/20 |
| preregistered decision | **`revise_candidates`** |

The result does not test whether mean logit KL can capture useful policy
opportunity: no such opportunity exists on this block. Uniform is already at
the normalized maximum on every prompt, so for any expanded candidate set that
still contains uniform,

`max(candidate score) - uniform score = 1 - 1 = 0`

prompt by prompt. Running the preregistered expanded policies again on prompts
440--459 therefore cannot make H reach 0.10 and must not be submitted.

The weak candidates are also nested rather than complementary here. All five
eviction successes are interior successes; interior adds one further success.
The logit measurements nevertheless look coherent: uniform has the lowest mean
KL on 18 prompts, interior has it on two prompts where all three candidates are
correct, trace lengths equal `min(8, FP generation length)`, every candidate in
a prompt shares one trace hash, and every replay reproduces the FP argmax.
Within eviction and interior separately, the stored KL statistics cleanly
separate their successes from their failures. That is descriptive mechanism
evidence, not evidence for gain over uniform.

### Why the previously confirmed point was not stable enough

The k16/B=2 task produced uniform accuracy 0.80 on prompts 400--409, 0.80 on
held-out prompts 420--439, and 1.00 on prompts 440--459: 44/50 = 0.88 overall.
The 0.80-versus-1.00 difference between the two 20-prompt blocks has a
conditional two-sided exact probability of about 0.106, so the current sample
does not distinguish a structural block shift from ordinary binary prompt
variation. Reconstructing the queried needle location also finds no depth
explanation: mean queried depth is 0.519 versus 0.596, while the four earlier
uniform failures span depths 0.292--0.900. Provenance, prompt IDs, corpus rows,
bit spending, generation caps, and task configuration are all correct.

The practical conclusion is that selecting a point at the upper 0.80 boundary
with 10 prompts and confirming it with only 20 prompts did not leave enough
headroom for a later oracle test. This is an experimental-design failure, not a
diagnostic implementation failure and not evidence that the logit proxy works.
The next iteration must establish a harder point on a larger fresh block before
collecting any expanded policy diagnostic.

Artifacts:

- `h0_measurement/results/r8policy_dev_981481/`
- `policy_981481.txt`
- `policy_981481_summary.csv`
- `policy_981481_prompts.csv`

## V2-A full-cycle operating-point screen: jobs 982121 and 982122

The repair screen uses Llama-3.1-8B at 32K, question-agnostic multikey NIAH,
B=2, and one complete 40-document corpus cycle on prompts 500--539. Job 982121
runs k24/v4/h4 and job 982122 runs k32/v4/h4. Each artifact has exactly 80
independently decoded FP/uniform rows. The strict paired reader authenticates
the real corpus and source offsets, queried-needle rank and depth, task and
answer-contract versions, model/cache/task configuration, B=2 bit audit, row
keys, prompt split, generation caps, and both sidecars before reporting scores.

| cell | FP | uniform | uniform halves | FP-uniform, paired 90% CI | first_ok | decision |
|---|---:|---:|---:|---:|---:|---|
| k24 | 1.000 (40/40) | 1.000 (40/40) | 1.000 / 1.000 | 0.000 [0.000, 0.000] | same | too easy |
| k32 | 1.000 (40/40) | 0.825 (33/40) | 0.800 / 0.850 | 0.175 [0.075, 0.275] | same | too easy |

No incomplete FP answer reaches its generation cap. K24 fails the uniform-band
and positive-difference gates because it is fully ceilinged. K32 has seven
clean uniform failures, balanced across the two halves, and a positive paired
interval. It fails only the preregistered primary requirement that uniform lie
in [0.50, 0.75]. Its 0.825 result is useful evidence that increasing requested
keys moves difficulty in the intended direction, but it is not a reason to
relax a boundary after observing the data. The seven k32 failures are short
six- or seven-token wrong answers and none reaches the cap; six retrieve a
known distractor and one is a near-digit error. They span queried ranks 2--31
and depths 0.1265--0.8811, with essentially the same mean rank and depth as the
successful rows. This points to associative-load failures rather than a cap or
target-position artifact. Standard RULER substring scoring and the
prespecified `first_ok` score agree on every row, so the decision is not a
scorer artifact.

The screen therefore selects no operating point and does not authorize a policy
diagnostic. The next action follows the frozen ``both too easy imply k40''
branch: run one fresh k40/v4/h4 FP+uniform cell on prompts 540--579 with the same
gates. The plan records, before those outcomes are viewed, at most one final
k36 or k48 bracket if k40 cleanly misses only the primary band. V2-B remains
blocked until a fresh cell passes every operating-point gate.

Artifacts:

- `h0_measurement/results/r8op2_k24_982121/`
- `h0_measurement/results/r8op2_k32_982122/`
- `op2_982121_982122.txt`
- `op2_982121_982122_summary.csv`

## V2-A2 k40 follow-up: job 982613

Job 982613 follows the declared ``both too easy imply k40'' branch on a fresh
full corpus cycle, prompts 540--579. It keeps every V2-A setting fixed except
`n_keys=40`. Slurm completed in 5:06 (Python artifact time 177 seconds) and
wrote exactly 80 rows. The strict reader authenticates the exact corpus SHA,
40 distinct unspliced source documents, source metadata across arms, queried
needle rank/depth and full depth vector, task/generation versions, P0 decode
path, B=2 accounting, caps, row keys, and sidecar.

| quantity | result |
|---|---:|
| FP | 1.000 (40/40) |
| uniform B=2 | 0.925 (37/40) |
| uniform halves | 0.950 / 0.900 |
| FP minus uniform | 0.075 [0.025, 0.150] |
| incomplete capped FP | 0 |
| `first_ok` | identical to primary score |
| decision | **too easy; no selection** |

K40 fails the primary [0.50, 0.75] band, both half-block upper bounds, and the
strict lower-bound requirement for FP minus uniform. These failures all point
in the same too-easy direction rather than showing an unstable split: both
halves exceed 0.85. The three uniform failures are uncapped six-token answers,
and all retrieve a registered distractor. They occur at queried ranks 36, 35,
and 8 and depths 0.8948, 0.8443, and 0.2725. The cap, score definition, and a
single target-position region therefore do not explain the result.

K40 is easier than the independent k32 block (0.925 versus 0.825), but the
failure-count difference is compatible with block sampling; it does not support
a monotonic key-count claim. The final k48 action is a single bounded
development screen, not evidence for SIEVE. The pre-k40 phrase ``every non-band
gate passes'' was ambiguous because the half and delta gates also fail when a
cell is strongly ceilingward. After observing k40 but before viewing any prompt
at 580 or above, the plan records the operational interpretation: valid FP and
provenance plus both halves above 0.85 take the already named k48 branch. K48
must independently pass every original gate. If it fails, the multikey/key-count
search stops with no interpolation, pooling, or relaxed threshold.

Artifacts:

- `h0_measurement/results/r8op2_k40_982613/`
- `op2_k40_982613.txt`
- `op2_k40_982613_summary.csv`

## V2-A3 final k48 bracket: job 982702

Job 982702 is the single final development bracket permitted after k40 was
directionally too easy. It uses k48/v4/h4 on the untouched prompts 580--619 and
keeps the model, 32K context, QA compression point, B=2, FP/uniform arms,
corpus, generation contract, and all acceptance gates fixed. Slurm completed in
5:08 (Python artifact time 181 seconds) and wrote exactly 80 rows. The strict
reader accepts the full row, source, target, bit, cap, P0, sidecar, and corpus
contract.

| quantity | result |
|---|---:|
| FP | 1.000 (40/40) |
| uniform B=2 | 0.850 (34/40) |
| uniform halves | 0.900 / 0.800 |
| FP minus uniform | 0.150 [0.075, 0.250] |
| incomplete capped FP | 0 |
| `first_ok` | identical to primary score |
| decision | **ineligible; no selection** |

K48 passes FP/cap validity and the paired-difference gate. It fails the frozen
uniform [0.50, 0.75] band, and its first half exceeds the half-block upper bound
(0.900 > 0.85). Five of six uniform failures are registered distractor
retrievals. Five failures are short uncapped answers; prompt 619 is incomplete
at the uniform generation cap, which could only leave the score unchanged or
raise the already too-high uniform mean if extended. Failure and success target
positions have nearly the same means (rank 25.2 versus 23.8; depth 0.514 versus
0.506), so target location does not explain the decision.

### Decision after the bounded search

No tested point provides the stable 25--50% uniform failure rate required for a
well-powered complete-policy diagnostic:

| development block | k | uniform | decision |
|---|---:|---:|---|
| prompts 500--539 | 24 | 1.000 | too easy |
| prompts 500--539 | 32 | 0.825 | above band |
| prompts 540--579 | 40 | 0.925 | too easy |
| prompts 580--619 | 48 | 0.850 | above band; half gate fails |

Because key count and prompt block change together after k32, these means cannot
separate a key-count effect from block variation; the unpaired interface did
not reliably control difficulty. They do not establish that asking for more
keys makes uniform accuracy worse. The terminal rule therefore applies:
do not run k44, k56, k64, combine the adaptive blocks, relax the 0.75 boundary,
or launch V2-B on prompts 620--659. Doing so would turn operating-point search
into outcome-guided selection.

This does not add evidence that SIEVE is effective or ineffective; it says the
current random multikey interface did not produce a stable test bed for the
next proxy comparison. Further progress needs a new, separately specified
difficulty construction with a reproducible per-prompt hardness control and a
fresh selection/confirmation split. B=1 is not the repair: prior P0b evidence
places uniform near failure there, and the roadmap already treats the 1-bit
tier as unusable. Until such an interface is validated, the existing negative
router and proxy conclusions remain the scientifically supported result.


Artifacts:

- `h0_measurement/results/r8op2_k48_982702/`
- `op2_k48_982702.txt`
- `op2_k48_982702_summary.csv`

## V3 contrastive multikey-panel qualification: job 983199

The versioned panel replaces one random target per context with four fixed depth
slots over one shared 48-needle context and one shared allocation. Job 983199
ran Llama-3.1-8B-Instruct at 32K, question-agnostic compression, B=2, prompts
700--739, and exactly the FP and uniform arms. It completed in 6:31 (271 seconds
inside the runner) and wrote 320 rows: 40 contexts x 4 queries x 2 arms.

The strict reader authenticates 40 distinct unspliced PG-19 documents and
context hashes, corpus SHA `0a26bc1e05a1eea8`, the exact task/panel/RNG
versions, four rotated target clusters and fixed depths, registered distractor
identities, one allocation hash per context/arm across all four questions,
B=2 spending, prompt-token accounting, cap state, and sidecars before computing
an outcome.

### Frozen qualification result

| quantity | result | gate |
|---|---:|---|
| FP query accuracy | 157/160 = 0.981 | pass |
| FP slots at depths .15/.38/.62/.85 | 1.000 / 1.000 / 0.950 / 0.975 | pass |
| uniform query/prompt mean | 136/160 = 0.850 | **fail: above 0.75** |
| uniform halves | 0.863 / 0.838 | **fail: first half above 0.85** |
| uniform slots | 0.950 / 0.850 / 0.800 / 0.800 | pass: spread 0.150 |
| FP minus uniform | 0.131 [0.088, 0.181] | pass |
| incomplete capped FP failures | 0 | pass |
| strict decision | `stop_panel` | binding |

The bootstrap uses 10,000 seed-0 draws of 40 contexts, carrying all four queries
together. It never treats 160 query rows as independent. Standard RULER
substring score equals the prespecified `first_ok` outcome on every row.

The four-query interface does reduce the all-or-nothing noise of the earlier
task. Uniform prompt scores are 1.00 on 23 contexts, 0.75 on 11, 0.50 on five,
and 0.25 on one. FP is perfect on 38 contexts, 0.75 on one, and 0.50 on one.
Nevertheless, uniform makes only a 15% query error rate, below the frozen
25--50% target. Its first half exceeds the stability upper bound by one query.

### Paired behavior and failure mechanism

Across the 160 paired queries:

| FP | uniform | count |
|---|---|---:|
| correct | correct | 136 |
| correct | wrong | 21 |
| wrong | wrong | 3 |
| wrong | correct | 0 |

Uniform therefore introduces 21 clean additional failures and never repairs an
FP failure. Its 24 failures break down as follows:

| uniform failure | count | share |
|---|---:|---:|
| exact distractor from the target cluster | 11 | 45.8% |
| exact registered value from another cluster | 7 | 29.2% |
| one-digit mutation of the target | 3 | 12.5% |
| other numeric answer | 3 | 12.5% |

Eighteen of 24 failures (75%) are exact registered alternatives. All three FP
failures are also registered-alternative retrievals and remain failures under
uniform. No failed output is empty or reaches the generation cap. This confirms
the intended associative-confusion mechanism and a real B=2 loss; it is not a
parser, cap, score, source, or allocation-identity artifact.

Uniform accuracy falls descriptively from 0.950 at depth 0.15 to 0.800 at both
0.62 and 0.85. Because slot and depth are the same factor, this is not a causal
depth estimate or permission to keep only deeper queries. Cluster rotation
does its job: cluster accuracies are 0.925 for alpha and 0.825 for each other
cluster, while the slot spread remains within the frozen bound.

### What this says about SIEVE

The panel implementation is reliable and the experiment demonstrates stable
quantization-induced associative retrieval loss. It does **not** show that
SIEVE, any adaptive policy, the end-task oracle, or the whole-policy logit rule
is effective. Only FP and uniform ran. Candidate complementarity `H`, selector
gain `G`, and deployable-router regret are therefore unidentified. The
positive FP-minus-uniform difference is available headroom, not evidence that
another compressed policy can recover it.

As a post-outcome diagnostic, requiring all four questions to be correct would
give FP 38/40=0.950 and uniform 23/40=0.575, with uniform halves 0.600/0.550.
That score lands in the desired band, but it was not the registered primary
outcome. Changing the score now is precisely the outcome-guided repair that the
plan forbids. It cannot reopen V3 or authorize a fresh panel run.

The binding action is:

- do not rerun job 983199;
- do not change cluster size, depths, budget, score, or select deeper slots;
- do not open prompts 740--819; and
- do not implement or submit panel policy development.

The current paper should keep the calibrated/per-head router and proxy transfer
as negative results. Supported positive claims remain the characterization,
fixed co-design findings, and SOTA baseline comparison.

### Design direction after closing V3

A future routing study should stop tuning synthetic NIAH difficulty. Use a
separately versioned, pinned public natural long-context task with official
deterministic scoring and untouched qualification/development/test splits.
Audit the release checksum, prompt template, parser, IDs, length/category
strata, grouping, and absence of truncation on CPU before any model output.
LongBench-v2 multiple choice is a candidate for that audit, not yet a selected
dataset.

The experiment should test three claims in order:

1. **Whole-policy opportunity:** compare the ordered equal-memory candidate menu
   with its strongest fixed member `F*`; require a preregistered positive
   label-seeing end-task-oracle gain `H`.
2. **Proxy transfer:** only if H passes, test whether a frozen label-free
   whole-policy logit rule gains over `F*`, captures a fixed fraction of H,
   and is stable across fixed halves.
3. **Deployable routing:** only if both pass, cross-fit a low-capacity selector
   from pre-generation features and evaluate it once on the untouched test
   split. Candidate outputs, labels, FP decode logits, and per-policy logits
   cannot be deployable features.

If the candidate menu lacks H, abandon routing for that menu. If H exists but
the logit rule lacks G, reject the proxy. This ordering prevents another router
from being trained where there is no recoverable policy complementarity.

Artifacts:

- `h0_measurement/results/r8panel_qual_983199/`
- `panel_qual_983199.txt`
- `panel_qual_983199_summary.csv`
- `h0_measurement/logs/r8_983199.out`


## V4 CPU audit: selecting an untruncated natural task

V3 closed because its frozen primary score remained ceilingward. V4 therefore
changes the task family rather than tuning the synthetic generator again. A
CPU-only, pre-output audit selects LongBench v2 multiple choice under a pinned
32K protocol. No model output was generated or inspected during this audit.

### Reproducible source contract

The local `data.json` is the official 503-row release at revision
`2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`, 465,490,535 bytes, SHA-256
`15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2`.
The study pins the official zero-shot prompt and parser from code revision
`2e00731f8d0bff23dc4325161044d0ed8af94c1e`; the 262-byte prompt has SHA-256
`68a162252bc9ff71d5d7abca3d69bb31aac3c35f832d657a2866f2018b8a6950`.
It also pins the Llama-3.1-8B-Instruct snapshot
`0e9e39f249a16976918f6564b8830bc894c89659` and its chat template.

The official evaluator samples at temperature 0.1 and may middle-truncate an
overlength prompt. This controlled study instead uses greedy generation to
remove sampling noise between policies and rejects truncation. It keeps the
official prompt, answer parser, and 128-token output cap. These results will not
be described as an exact LongBench leaderboard reproduction.

### Eligibility and leakage control

The full rendered chat inputs span 10,110 to 4,144,620 Llama tokens. Reserving
128 answer tokens inside the 32,768-token window gives a maximum input length of
32,640. Exactly 117 rows pass that rule; their actual range is
10,110--32,592. All are in the official `short` word-count category. The
canonical sorted ID/token list hashes to
`d87774ba198ad16645bb9364bd96e9fa80e5e5c8e5a7924ac9c5677ae370d3d0`.
No FP or compressed result was available to this selection.

The dataset supplies no source-document ID. Exact stripped contexts and exact
stripped questions are therefore joined transitively into leakage components.
The resulting 108 components are deterministically ordered by a salted context
hash and divided before inference:

| phase | components | rows | token range |
|---|---:|---:|---:|
| qualification | 19 | 20 | 10,110--30,051 |
| development | 44 | 52 | 10,727--29,984 |
| untouched confirmation | 45 | 45 | 12,701--32,592 |

The phases have zero common IDs, exact contexts, or exact questions. This is a
procedural split of a public benchmark rather than a secret test. The released
data lacks source provenance, so near-duplicate source documents cannot be
ruled out.

The 117 rows are imbalanced across natural categories: 61 single-document, 33
multi-document, 12 dialogue-history, 7 long in-context-learning, 3 code, and 1
structured-data example. The primary analysis is paired micro accuracy with
component-clustered resampling; category breakdowns will be descriptive.

### What changed in the design

The experiment now tests complete policies on a non-ceiling natural end task.
It first runs a 20-row FP/uniform qualification that can only authenticate the
pipeline, output format, operating range, and a canonical choice-logit
interface. It cannot select a task, budget, or subset. If qualification passes,
the 52-row development phase compares the fixed eight-policy B=2 menu.

The label-seeing end-task oracle and the label-free logit proxy are separate.
The former identifies whether any compressed-policy complementarity exists.
The latter teacher-forces the requested response prefix and compares each
policy's A/B/C/D distribution with FP by four-choice KL. This targets the
benchmark decision instead of generic response wording. It still executes all
policies and sees FP logits, so it is only a diagnostic upper bound. The exact
H/G thresholds and the untouched 45-row confirmation rule were recorded in
`plan.md` before model output.

This audit is evidence that the next experiment is measurable and
outcome-independent. It is not evidence that SIEVE, the candidate menu, or the
new proxy is effective. Those claims remain gated on qualification,
development complementarity H, proxy transfer G, and locked confirmation in
that order.

## V4 qualification: natural operating point passes, generative answer contract stops the iteration

Slurm job 983715 completed the frozen 20-item LongBench v2 qualification in
15 minutes. The strict reader authenticated the pinned dataset, manifest,
model and tokenizer revision, official prompt and parser, chat template,
split hashes, all 40 end-task rows, all 20 choice rows, the cross-file hashes,
and the B=2 allocation records. Inputs span 10,110--30,051 tokens; every input
is complete and untruncated. Preflight job 983708 was cancelled during its CPU
tests before model inference and produced no result artifact.

| frozen qualification check | result | gate |
|---|---:|---:|
| FP valid official parses | 18/20 | pass, at least 18 |
| uniform valid official parses | 18/20 | pass, at least 16 |
| FP official accuracy | 0.450 | pass, [0.15, 0.70] |
| uniform B=2 official accuracy | 0.450 | pass, [0.10, 0.70] |
| invalid FP answers at the 128-token cap | 2 | **fail, required 0** |
| FP canonical choice agrees with valid greedy parse | 17/18 = 0.944 | pass, at least 0.80 |
| strict decision | `stop_v4` | development remains closed |

The failed rows are not missing data or compression failures. They are the two
Long-dialogue History Understanding questions in one leakage component: they
share the same question, use two related contexts, and both FP and uniform spend
all 128 tokens calculating player utilities without emitting the requested
answer phrase. The official parser therefore assigns zero. Six other FP rows
and seven other uniform rows also reach 128 tokens but contain a valid parsed
choice, so the failure is specifically uncensored answer elicitation on this
component rather than a general inability to decode at the cap.

The paired official outcomes contain seven rows correct under both policies,
two FP-only, two uniform-only, and nine wrong under both. Equal aggregate
accuracy therefore hides real policy changes, but FP is a ceiling reference
rather than an admissible B=2 policy, so this 2-versus-2 exchange is not evidence
of a deployable routing gain.

The scalar choice diagnostic clarifies the interface failure. On the same
exposed qualification block, FP's canonical A/B/C/D argmax scores 11/20 and
uniform's scores 10/20, compared with 9/20 for each official generative score.
One capped invalid FP row has the correct canonical choice, and the sole valid
canonical/direct disagreement changes a wrong greedy parse to the correct
canonical choice. These are descriptive post-result observations only. They
motivate a separately versioned forced-choice study; they do not retroactively
change V4's end-task metric or pass its gate.

V4 therefore stopped before its 52-item development split. At that decision
point, no expanded-policy or confirmation job had been submitted and no output
from those 97 items had been inspected. V5 later opened the 52 development
items under a separately frozen forced-choice protocol; the 45 confirmation
items remain untouched. The V4 result establishes a useful non-ceiling natural task
and a coherent canonical decision point, while rejecting the frozen
128-token generative parser as a sufficiently uncensored interface for this
study. The authenticated reader output is
[`longbench_v2_qualification_983715.txt`](longbench_v2_qualification_983715.txt),
and the artifacts are in
`h0_measurement/results/lbv2_qualification_983715/`.

## V5 forced-choice development: competence passes, policy opportunity fails

V5 removed V4's generative parser and cap from the endpoint without changing
the source items, model, B=2 memory budget, or ordered policy menu. Each policy
teacher-forced the fixed response scaffold and selected one of A/B/C/D at the
next position. Its primary branch-blind proxy used only the five preceding
scaffold distributions and could not see the answer branch or gold label.

Job 983888 completed in 9:47, including 8:51 in the GPU step, with exit code
0:0. The strict reader authenticated the pinned dataset and 52-row development
split, 44 leakage components, model/tokenizer revisions, complete untruncated
prompts, reciprocal sidecars, 468 prediction rows, 416 proxy rows, allocation
identities, and exact B=2 spending.

| frozen check | result | decision |
|---|---:|---|
| FP forced-choice accuracy | 19/52 = 0.365 | competence pass; bootstrap q05 0.255 |
| uniform / every other candidate | 19/52 = 0.365 | `F* = uniform` by frozen order; q05 0.260 |
| itemwise candidate oracle | 22/52 = 0.423 | three extra correct rows |
| `H = oracle - F*` | 3/52 = 0.058 | **fail:** required at least 0.10 |
| H component-bootstrap q05 | 0.017 | pass: above zero |
| H frozen halves | 0.071 / 0.042 | **fail:** second half required at least 0.05 |
| proxy G | suppressed | sequential H gate failed |
| strict result | `stop_no_opportunity` | V5 closed |

The bootstrap result and the stop decision are consistent. The three oracle
rescues are repeatable enough that the 5th percentile is above zero, but they
are too few to meet the prespecified practical gain and second-half stability
requirements. The confirmation split cannot be used to enlarge this estimate,
and lowering H after seeing 3/52 would make the criterion outcome dependent.

### Why eight allocations became two endpoint behaviors

The marginal tie is stronger than it first appears. On 42/52 items every
candidate makes the same choice. On the remaining ten, uniform makes one choice
and all seven nonuniform policies make another. Eviction, interior,
interior-pool, interior-cascade, OBCache-K, OBCache-K plus Ada-KV, and LaProx
match each other and FP on all 52 items.

| uniform versus the shared FP-like behavior | rows |
|---|---:|
| FP-like correct, uniform wrong | 3 |
| uniform correct, FP-like wrong | 3 |
| both wrong with different choices | 4 |
| same choice | 42 |

This is an endpoint collapse, not an implementation alias. Every item has eight
distinct candidate allocation IDs, and policy confidence summaries differ even
when their final argmax agrees. The candidate oracle therefore has only one
effective decision: choose uniform or choose the common FP-like answer. No
selector over this menu can exceed the observed 3/52 opportunity on these
items.

The gold-free scaffold rule reflects the same mechanism without revealing its
gated accuracy. It never selects uniform, its selected answer equals FP on all
52 items, and uniform has the worst scaffold-KL rank on 50/52. This is expected
for an FP-fidelity objective. Because H failed first, the reader does not
compute or report selector accuracy, G, rescue, or harm. There is no V5
confirmation lock.

Artifacts:

- `h0_measurement/results/lbv2_v5_development_983888/`
- `h0_measurement/logs/lbv2v5_983888.out`
- [`longbench_v2_v5_development_983888.txt`](longbench_v2_v5_development_983888.txt)
- [`longbench_v2_v5_development_983888_summary.csv`](longbench_v2_v5_development_983888_summary.csv)

### Next bounded iteration

Do not tune the scaffold proxy, add more B=2 Llama policies, lower H, or run V5
confirmation. Section 3K of `plan.md` froze one architecture change:
Qwen3-30B-A3B-2507 at 40,960 tokens, still B=2. Its qualification result follows.

## V6 Qwen qualification: competence and non-ceiling gates pass

Job 984224 completed the frozen 20-item qualification in 8:49, with 7:26 in the
GPU step and exit code 0:0. The strict reader authenticated the pinned dataset,
117-row gold-free Qwen manifest, 20 qualification IDs in 19 leakage components,
model/tokenizer revision, no-thinking chat template, complete prompt hashes,
CUDA-only placement, exact uniform allocations, source seal, snapshot inventory,
and all 40 scalar prediction rows. No proxy artifact was generated in this
phase.

| frozen qualification check | result | gate |
|---|---:|---|
| FP forced-choice accuracy | 12/20 = 0.600 | pass, at least 0.50 |
| FP component-bootstrap q05 | 0.421 | pass, above 0.25 |
| uniform B=2 accuracy | 12/20 = 0.600 | pass, within [0.30, 0.80] |
| uniform component-bootstrap q05 | 0.429 | pass, above 0.25 |
| missing, truncated, or invalid rows | 0 | pass |
| strict decision | `advance_v6_development` | development authorized |

FP and uniform agree on 18/20 choices. Of the two changed choices, uniform
rescues one FP error and introduces one error, producing the same marginal
accuracy. The result therefore establishes a competent, non-ceiling operating
point. It does not establish that the expanded policy menu has useful
complementarity, and it does not test the scaffold proxy.

The authenticated lock
[`longbench_v2_v6_qualification_lock_984224.json`](longbench_v2_v6_qualification_lock_984224.json)
binds the qualification artifact, source hashes, model snapshot, exact menu,
thresholds, and untouched development/confirmation partitions. The next
permitted experiment is the one frozen 52-row, 44-component development run.
It first tests competence and the label-seeing candidate envelope H. The reader
will suppress the proxy outcome G unless all H gates pass. The 45 confirmation
rows remain closed.



## V6 Qwen development: stronger architecture does not repair policy collapse

Job 984370 completed the frozen Qwen development protocol in 16:38, including a
15:25 GPU step, with exit code 0:0 and peak Python-step RSS of about 60.2 GiB.
The strict reader authenticated the qualification lock, pinned dataset and
manifests, Qwen snapshot and source seal, 52 rows in 44 leakage components, all
468 forced-choice predictions and 416 scaffold-proxy measurements, reciprocal
artifact hashes, exact uniform B=2 storage, feasible sparse budgets, and the
completion marker.

| frozen development check | result | gate |
|---|---:|---|
| FP forced-choice accuracy | 24/52 = 0.462 | **fail:** at least 0.50; q05 0.340 passes |
| best fixed compressed policy | uniform, 27/52 = 0.519 | point pass; q05 0.404 passes |
| each of the seven nonuniform policies | 24/52 = 0.462 | descriptive |
| itemwise compressed-policy oracle | 29/52 = 0.558 | descriptive ceiling |
| `H = oracle - uniform` | 2/52 = 0.038 | **fail:** at least 0.10 |
| H component-bootstrap q05 | 0.000 | **fail:** strictly above zero |
| H frozen halves | 0.036 / 0.042 | **fail:** each at least 0.05 |
| scaffold-proxy G | suppressed | competence and H did not pass |
| strict result | `stop_v6_invalid_operating_point` | V6 closed |

The formal stop is the preregistered FP competence condition. The 20-row
qualification estimate of 12/20 did not transfer to the larger development
block: FP scored 24/52, two correct answers short of its 0.50 floor. This is not
a provenance, truncation, placement, or budget failure. Its component bootstrap
still excludes chance, but the frozen rule required both the point floor and
the bootstrap condition.

The policy result gives a second, independent reason not to continue. Uniform
changes 11 FP choices: it rescues five FP errors, harms two FP successes, and
changes four other wrong answers. All three interiors, eviction, OBCache-K, and
OBCache-K plus Ada-KV exactly match FP on every item. LaProx differs from FP on
one item, where neither answer is correct. The only two candidate-oracle rescues
over uniform are the two rows on which uniform harms FP. Thus the observed
headroom is two items, below every frozen magnitude and stability gate.

The branch-blind scaffold rule supplies a label-free mechanism check only. It
selects interior 16 times, interior-cascade 9, interior-pool 8, OBCache-K 7,
OBCache-K plus Ada-KV 7, and LaProx 5; it never selects uniform or eviction.
Its selected forced choice equals FP on all 52 rows and equals uniform on 41.
This is consistent with an FP-fidelity objective preferring the FP-like class.
No selector accuracy, G, G/H, rescue, or harm statistic is computed because the
sequential competence and H gates failed.

### Distinct allocations, almost identical decisions

This is not an implementation alias. Each item has eight distinct candidate
allocation IDs. Interior allocations spend just under B=2 while evicting about
61--65% of context keys; the sparse 8-bit policies evict about 75%; uniform
stores every key at two bits. Despite those differences, the 52-item answer
vectors form only three classes:

1. FP, eviction, all three interiors, OBCache-K, and OBCache-K plus Ada-KV;
2. uniform;
3. LaProx, which differs from the first class once.

The V5 Llama result and V6 Qwen result therefore agree on the main design
problem: changing the scorer and allocation geometry creates diverse caches but
very little useful answer-level complementarity. A selector cannot recover gain
that its candidate policies do not create.

### Consequence for the next design

Do not rerun V6, lower the FP or H gates, inspect G, or open the 45 confirmation
rows. No confirmation lock was created. Changing only the scaffold proxy cannot
help because failure occurs before proxy evaluation. Adding another B=2 scorer
to this same sparse family is also weakly motivated: six distinct nonuniform
allocators already share FP's entire answer vector.

Section 3L of `plan.md` froze one bounded mechanism test before any new
model output. The tail-32 observation usually covers the end of choice D and
response-format text rather than the question and all four choices. V7 was
designed to compare that score with an equal-span question/A/B/C/D score under
the same Qwen model, B=2 budget, protected tail, quantizer, endpoint, and
water-filling rule. A label-free tokenizer audit found 152 unused singleton
components at the first sufficiently large audited window, 131,072 tokens. A
salted selection supplied a wholly new 20/52/45 qualification, development, and
confirmation partition. As reported below, qualification stopped V7 before the
structured policy comparison ran. The present V4--V6 45-row confirmation split
stays untouched.

Artifacts:

- `h0_measurement/results/lbv2_v6_qwen_development_984370/`
- `h0_measurement/logs/lbv2v6d_984370.out`
- [`longbench_v2_v6_development_984370.txt`](longbench_v2_v6_development_984370.txt)
- [`longbench_v2_v6_development_984370_summary.csv`](longbench_v2_v6_development_984370_summary.csv)

## V7 128K qualification: operating-point competence fails before the mechanism test

V7 changed the proposed mechanism once: its frozen development design would
compare the existing tail-32 importance observation with an equal-weight score
over exact question and A/B/C/D content positions. A label-free source audit
selected a fresh 20/52/45 partition of singleton LongBench-v2 components at
Qwen's native-supported 131,072-token window. The qualification phase was
deliberately narrower. It ran only FP and exact all-2 uniform on the 20
qualification items to determine whether this fresh operating point was
competent enough to expose the mechanism comparison.

Job 984886 completed in 27:54 (GPU step 18:27) with exit code 0:0. The strict reader authenticated the pinned
dataset, answer-free manifest, exact qualification IDs and order, Qwen model
and tokenizer snapshot, no-thinking prompt template, complete untruncated
inputs, CUDA placement, source ledger, exact B=2 uniform allocations, all 40
prediction rows, reciprocal artifacts, and the completion marker.

| frozen qualification check | result | gate |
|---|---:|---|
| FP forced-choice accuracy | 9/20 = 0.450 | **fail:** required at least 0.50 |
| FP component-bootstrap q05 | 0.250 | **fail:** required strictly above 0.25 |
| uniform B=2 accuracy | 7/20 = 0.350 | pass: within [0.30, 0.80] |
| uniform component-bootstrap q05 | 0.200 | **fail:** required strictly above 0.25 |
| missing, truncated, or invalid rows | 0 | pass |
| strict result | `stop_v7_qualification` | V7 closed |

The point estimate for uniform is non-ceiling and lies inside its frozen range,
but the experiment does not satisfy the complete competence rule. FP misses
its 0.50 floor, its q05 lands exactly on the excluded boundary, and uniform's
q05 is 0.200. These failures reflect performance on the prespecified fresh
sample rather than missing data, truncation, placement, budget, or provenance
errors.

The two policies make the same forced choice on 16/20 items. Their correctness
pairs are seven both correct, eleven both wrong, two FP-only correct, and zero
uniform-only correct, so uniform supplies no rescue on this block. FP accuracy
by ascending prompt-token quartile is 0.60/0.20/0.40/0.60 and uniform is
0.40/0.20/0.40/0.40. The failure is therefore not isolated to the longest
prompts or monotone in prompt length. These are descriptive post-result
aggregates and cannot justify selecting a subset or length range.

### Scope of the negative result

This result rejects the V7 operating point under its prespecified qualification
rule. It does not show that the structured-query score is ineffective. No
structured-query or tail-32 compressed policy ran in qualification, so there is
no structured-versus-tail end-task comparison and no measurement of mechanism
effect S or candidate opportunity H. The stop occurs before the mechanism
question becomes identifiable.

No `advance_v7_development` lock was written. No development source ledger or
52-row development job exists, no `F*`, S, H, or proxy result was selected, and
the 45-row V7 confirmation partition remains untouched. No confirmation lock
or job exists. V7 is terminal under its frozen sequential rule.

Do not rerun the exposed qualification block, relax its gates, select favorable
items, or use the reserved development or confirmation rows to choose a repair.
The defensible next step is outside V7: either close this end-task mechanism
branch or preregister a new operating-point construction with a fresh partition
before testing the structured observation. The V7 outcome alone supplies no
basis for choosing that construction from these labels.

Artifacts:

- `h0_measurement/results/lbv2_v7_qwen_qualification_984886/`
- `h0_measurement/logs/lbv2v7q_984886.out`
- [`longbench_v2_v7_qualification_984886.txt`](longbench_v2_v7_qualification_984886.txt)
- [`longbench_v2_v7_qualification_984886_summary.csv`](longbench_v2_v7_qualification_984886_summary.csv)

