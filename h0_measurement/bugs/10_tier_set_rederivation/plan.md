# R10: tier-set re-derivation and the base-rate decision

**Protocol frozen:** 2026-09-24, before any R10 GPU output.  
**Roadmap role:** R10, prerequisite for R11 nested-code overhead.  
**Status:** implementation in progress; no R10 model result exists yet.

The older `bugs/10_kstar_budget/` directory is roadmap R9 and is closed. This
folder is roadmap R10. It does not reopen K-star budgeting or the stopped R8/V7
end-task branch.

## 1. Question

Under the design choices already fixed by co-design -- one allocation per
physical KV head, lagged `accum` output estimate, and current-query cascade
attention measured from 4-bit keys -- which discrete key-rate ladder preserves
the exact-output benefit of the current dense ladder at an exact B=3 budget?
B=2 is a prespecified secondary scope check.

This experiment selects ideal per-token rates under independently measured
TurboQuant at each width. R11 separately asks whether one finite-dimensional
nested code realizes those rate points cheaply. R10 does not measure end-task
accuracy, packed memory, or latency.

## 2. Frozen ladder panel

Every ladder is evaluated inside one run from the same prompts, FP logits,
values, rotations, and quantized-logit dictionary. Tier 0 is eviction.

| label | offered tiers | purpose |
|---|---|---|
| `full` | 0,1,2,3,4,5,6,8 | current control |
| `no1` | 0,2,3,4,5,6,8 | isolate removal of the reportedly dead 1-bit rung |
| `base3_dense` | 0,3,4,5,6,8 | isolate removal of the 2-bit rung |
| `nested3` | 0,3,4,6,8 | proposed 3+1+2+2 ladder; also removes 5 bits |
| `nested4` | 0,4,6,8 | directly re-budgetable 4+2+2 ladder; also removes 3 bits |

The full quantization measurement remains `[1,2,3,4,5,6,8]`; each alternative
only restricts the noise-cost dictionaries handed to the same physical-group
water filler. No alternative triggers another quantizer call.

### 2.1 The 3-bit/4-bit distinction

Co-design selected cascade observation width `bc=4`: it closes 82--91% of the
lag penalty, median 89%. A token stored only at 3 bits cannot later supply a
4-bit cascade score unless its first refinement bit is also retained and
counted. Therefore:

- `nested3` is the primary ideal allocation ladder requested by the roadmap;
- `nested4` is the ladder immediately compatible with repeated 4-bit rescoring;
- if only `nested3` passes, R11/R14 must price an always-retained observation
  refinement or change the scorer before the design is called deployable.

## 3. Fixed measurement contract

- Full measured widths: `[1,2,3,4,5,6,8]`.
- Budgets: B=2 and B=3; B=3 is primary.
- Protected/unseen positions: existing `floor_maxb` rule.
- Allocation unit: one `(layer, KV head)` allocation shared by its query heads.
- Interior score: `accum`.
- Cascade observation: `bc=4`, lagged `accum` output estimate.
- Corner policy: fractional, with `oracle,accum` measured; endpoint quantities
  use the matched group corner.
- Families: `niah,qa,cont`.
- Decode: 8 tokens, measurement steps 0 and 4; the reader uses step 4 only.
- Quantizer rotation seed: 2.
- Values remain exact, as in H0/co-design.
- Real PG-19 corpus only; synthetic input invalidates the experiment.

For each ladder and budget, record exact relative attention-output error,
realized mean key bits per selectable token, eviction fraction, and the fraction
assigned to every offered tier. Tier occupancy is de-duplicated by
`(prompt,family,step,layer,kv_head)` because one physical allocation is repeated
across the query heads in that group.

## 4. Implementation pilot (excluded from scientific evidence)

Run both tasks before the primary array:

| task | cell | prompts | purpose |
|---:|---|---:|---|
| 0 | Qwen3-1.7B at 2K, `cont` only | 1 | column, tier, and budget smoke test |
| 1 | Qwen1.5-MoE-A2.7B at 8K, `cont` only | 1 | `n_rep=1` reduction control |

Pilot acceptance is conjunctive:

1. The `full` panel duplicate equals the existing
   `grp_csv_b4_accum` allocation and exact error per row.
2. Every alternative uses only its declared tiers.
3. No allocation overspends B and unused budget is at most one 8-bit token,
   `B - mean_bits <= 8/L + 1e-9`.
4. For `n_rep=1`, restricted group allocation is identical to the corresponding
   restricted per-head allocation by construction.
5. Sidecar fields and all required columns are present; no primary value is NaN.

An authentication or pilot failure is a code/input failure. Fix it before the
same excluded pilot is rerun. It has no scientific stop implication.

## 5. Locked primary cells

All cells and prompt blocks were selected before the pilot or primary output.
They are disjoint from co-design wave 4.

| array task | model | context | prompts | offset | role |
|---:|---|---:|---:|---:|---|
| 0 | Llama-3.1-8B | 32K | 6 | 18 | GQA-4, shorter-context boundary |
| 1 | Llama-3.1-8B | 128K | 6 | 18 | GQA-4, long context |
| 2 | Qwen3-8B | 8K | 6 | 18 | GQA-4, largest measured cascade gain |
| 3 | Qwen3-30B-A3B-Instruct-2507 | 8K | 4 | 12 | GQA-8 architecture control |

No ladder, cell, prompt block, threshold, or aggregation rule is selected from
the pilot. The four primary cells form one confirmation panel rather than a
development/test search.

## 6. Statistics

For prompt block `p`, ladder `T`, budget `B`, and one model/context cell, define

```text
E_T(p) = sqrt(mean(err_T^2 over families, layers, and query heads at step 4))
r_T(p) = E_T(p) / E_full(p)
R_T(cell) = exp(mean_p(log(r_T(p))))
```

This treats a prompt, rather than a head row, as the resampling unit. The macro
result gives each of the four cells equal weight. A 10,000-draw PCG64(0)
bootstrap resamples prompt blocks independently within each cell and reports a
90% interval; gates use the frozen point estimates below.

For routed gain, each row uses the matched group endpoint:

```text
gain_T = min(error_uniform, error_group_corner) / error_T
routed_gain_T = geometric mean(max(gain_T, 1))
band_T = fraction(gain_T >= 2)
```

Aggregate first within prompts, then equally across prompts and cells. Report
`routed_gain_T / routed_gain_full` and `band_T - band_full`.

## 7. Validity gates

All are required for a scientific verdict:

- **V1 full identity:** the panel's `full` allocation summaries and exact errors
  match the existing full-ladder group+cascade path per row within numerical
  tolerance.
- **V2 tier/budget:** no absent tier is used, no overspend occurs, and unused
  budget is at most `8/L + 1e-9` for every physical group, ladder, and budget.
- **V3 provenance:** exact source ledger, model/context, prompt block, families,
  corpus hash, rotation, bit list, budgets, group allocation, `bc=4`, `accum`,
  `floor_maxb`, and non-synthetic sidecar fields authenticate.
- **V4 completeness:** every expected task artifact and row exists, all primary
  errors are finite, and every NIAH prompt retrieves its needle.
- **V5 control:** the excluded MHA pilot passes the `n_rep=1` identity.

A V-gate failure produces `invalid_r10`, not a design verdict.

## 8. Frozen B=3 decision gates

### 8.1 Primary ideal ladder: `nested3` versus `full`

All must pass:

- **G1:** median of the four cell-level `R_nested3` values <= 1.05.
- **G2:** worst cell `R_nested3` <= 1.10.
- **G3:** macro routed-gain retention >= 0.95.
- **G4:** every cell's routed-gain retention >= 0.90.
- **G5:** no cell loses more than 5 absolute percentage points of productive
  band.

Passing G1--G5 says the ideal 3+1+2+2 allocation ladder preserves the current
storable design within the frozen tolerance.

### 8.2 Directly re-budgetable ladder: `nested4` versus `nested3`

Select `nested4` only if all hold:

- median cell error ratio `nested4/nested3` <= 1.02;
- every cell ratio <= 1.05;
- macro routed-gain retention versus `nested3` >= 0.98; and
- `nested3` assigns tier 3 to less than 5% of physical tokens in every cell.

If `nested3` passes but `nested4` fails, retain `nested3` only as the ideal
ladder and carry the extra 4-bit observation-storage cost into R11/R14.

### 8.3 Sequential attribution

Use the fixed chain `full -> no1 -> base3_dense -> nested3` and the same
5% median/10% worst-cell error tolerances at each transition:

- failure at `full -> no1`: removing 1 bit is not exact-error safe;
- first failure at `no1 -> base3_dense`: the 2-bit rung matters;
- first failure at `base3_dense -> nested3`: the 5-bit rung matters.

This attribution is diagnostic. It does not authorize tuning another ladder on
these prompts.

## 9. B=2 scope rule

Apply G1--G5 at B=2 as a secondary result.

- Pass: one selected ladder covers B=2 and B=3.
- Fail B=2 while B=3 passes: scope the ladder to B>=3 or retain tier 2 under a
  separately confirmed budget-conditioned design.
- Fail B=3: stop before R11 regardless of B=2.

Do not average B=2 and B=3 into one pass.

## 10. Failure branches and next dependency

- `no1` failure: retain the full ladder; investigate why the old dead-tier proxy
  did not imply exact-error safety. R11 is blocked.
- `base3_dense` failure after `no1` passes: reject a 3-bit minimum. A
  budget-conditioned ladder requires a new prompt block before R11.
- `nested3` failure after `base3_dense` passes: the missing 5-bit rung is the
  cause. A `{0,3,4,5,6,8}` code requires a new confirmation before R11.
- `nested3` pass and `nested4` failure: R11 must measure and count the retained
  +1 observation refinement, or the online cascade must change.
- both pass: select `nested4`; R11 tests a 4+2+2 nested code.
- validity failure: repair implementation/provenance and rerun only the excluded
  pilot or invalid task; do not interpret it as design evidence.

R11 begins only after this report selects and scopes a ladder. Its primary
quantity is nested-code rate overhead against monolithic widths at matched
distortion: 10% is the target and 20% is the architecture stop line. R14 follows
only if R11 establishes a representable ladder.
