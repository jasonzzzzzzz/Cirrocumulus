# R9 main-model baseline results

**Dates:** main campaign 2026-09-22; non-ceiling follow-up 2026-09-23  
**Status:** all five evaluation cells complete; the non-ceiling task interface
is held-out confirmed in job 980414.  
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
