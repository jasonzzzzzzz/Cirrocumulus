# R9: K*-derived physical KV-head budgets

**Status: closed at qualification.** Gate 0 completed on CPU. The source-sealed
GPU qualification completed as Slurm array **985183**; Q1, Q2, and Q4 passed,
but Q3 failed. The frozen decision is **stop_r9_qualification**, no advance lock
exists, and no development run is permitted.

## Gate 0 - audit of the existing K* evidence

## Decision

The existing campaign supports a weaker result than the old sentence “K* is
100% of the budget in 24/24 runs.” The audit reproduces that sentence exactly
under its original aggregation, but the statistic is a median over a coarse
candidate ladder. It does not show that the integer-resolution K* equals the
full budget.

The existing rows do establish two useful design facts:

1. Individual query heads often have apparent slack. At 32K, mean
   `K*/budget` is 0.720–0.978 across the six models.
2. Taking the maximum requirement across query heads that share one physical
   KV head removes almost all of that slack for GQA. This **kills the naive GQA
   policy** “measure each query head independently, then give its KV group the
   maximum.” Four of five GQA models have a group mean above 0.98; the remaining
   Qwen3-8B cell has group mean 0.919 and 87.0% exact saturation. The MHA
   `n_rep=1` control is unchanged at 0.800.

The max calculation is a diagnostic, not a directly measured group K*. The
current artifacts cannot reconstruct direct physical-group curves. A new
experiment must measure a shared group ranking and group error as a function of
every integer keep count.

## Authenticated scope

[`input_manifest.json`](input_manifest.json) freezes the canonical 24 cells from
`bugs/2_towards_real_evictor/report.md`: all six models at 8K, 16K, and 32K,
plus Llama-3.1-8B, Llama-3.3-70B, and Qwen3-30B at 64K and 128K. It records the
exact Parquet and sidecar path, byte length, SHA-256, Arrow-schema fingerprint,
row count, model ID, prompt count, and GQA replication factor for every cell.

The audit fails closed on:

- the manifest's canonical-content SHA;
- the source `bugs/2` report SHA;
- all 24 Parquet hashes and all 24 sidecar hashes;
- exact Arrow schema fingerprints and required column types;
- sidecar model, context, row count, prompt count, corpus, schedule, budgets,
  bit widths, and effective config;
- the exact corner config: `oracle,accum,window,recency`, `frac,abs`,
  `kappa=4`, `floor=256`, `kstar=true`, 12 points, and 10% tolerance;
- row-level fractional and absolute keep-count arithmetic, candidate membership,
  stored K* fraction, and finite full-budget error.

The manifest canonical-content SHA is
`f082668eb5f170884897dc83dd0660334945b7726c1ca66506e95db2bf4000bd`.
The sealed source-report SHA is
`1d47fc574413dd91c60c724dc507b892b142868107e2a8e31c01fc4bf2a5f274`.

## Population and definitions

The population follows the original report's complete-corner convention:

```text
quantized == true and n_practical == 3 and kstar3 is present
```

This selects 629,760 B=3 query-head observations. One observation is one
`(family, prompt, decode step, layer, query head)` row.

For query head `q`, define

```text
r_q = kstar3 / corner_tokens3_frac
mean = arithmetic mean(r_q)
slack = 1 - mean
saturation = fraction with r_q == 1
```

For the physical-group proxy, query heads are grouped by
`(family, prompt, step, layer, head // n_rep)` and the group value is
`max_q r_q`. Every group is checked to contain exactly `n_rep` contiguous query
heads. The architecture-derived replication factors are 1, 4, or 8.

The old 24/24 claim used a different summary: median `r_q` within each
`(layer, query head)`, followed by a median across heads in a cell. The audit
reproduces **1.0 in all 24 cells** under that definition.

## 32K result

| model | `n_rep` | query rows | query mean | query slack | query saturation | physical groups | group-max mean | group-max slack | group-max saturation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 4 | 18,432 | 0.873029 | 0.126971 | 80.697% | 4,608 | 0.983139 | 0.016861 | 97.092% |
| Llama-3.3-70B | 8 | 61,440 | 0.977575 | 0.022425 | 95.871% | 7,680 | 0.999349 | 0.000651 | 99.935% |
| Qwen3-30B-A3B-2507 | 8 | 18,432 | 0.767414 | 0.232586 | 67.128% | 2,304 | 0.990538 | 0.009462 | 98.307% |
| Qwen3-8B | 4 | 20,736 | 0.719854 | 0.280146 | 60.687% | 5,184 | 0.919002 | 0.080998 | 86.960% |
| Mistral-7B | 4 | 18,432 | 0.933055 | 0.066945 | 88.721% | 4,608 | 0.993547 | 0.006453 | 99.023% |
| Qwen1.5-MoE-A2.7B | 1 | 6,912 | 0.800453 | 0.199547 | 70.197% | 6,912 | 0.800453 | 0.199547 | 70.197% |

The `n_rep=1` row is the control: query and physical-group columns are identical.
The large jump at 4 and 8 query heads is therefore caused by max aggregation,
not a change in the K* definition.

This proxy cannot be promoted to a physical policy result. Each legacy query
head used its own oracle sensitivity ranking. Query heads in one GQA group share
the stored K and V tensors, so a deployable group has to choose one kept-token
set. The Parquets contain the selected scalar K*, the full-budget error, and no
intermediate curve errors, rankings, logits, or values. Distinct query-head
rankings cannot be combined after the fact. The max is neither a direct group
curve nor proof that one shared set attains the reported per-query errors.

## The 12-point grid is too coarse for the old saturation claim

The legacy implementation constructs a 12-point geometric grid from 1 to the
full fractional keep count. It also evaluates the `abs` policy keep count, so
the actual candidate set is:

```text
geometric_grid(full, 12) union {corner_tokens3_abs}
```

All 629,760 selected K* values belong to that exact set. Their sources are
478,286 geometric-only, 17,753 absolute-policy-only, and 133,721 points that
belong to both.

The table below audits the last geometric interval using actual per-prompt `L`
from the authenticated rows. “Effective penultimate” includes the extra
absolute-policy point when it lies closer to full and is summarized only on
rows whose recorded K* is full.

| nominal context | rows | geometric penultimate/full min | mean | max | full-K* rows | effective penultimate/full median | mean | max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 144,384 | 0.485351 | 0.485432 | 0.485553 | 126,006 | 0.485502 | 0.507865 | 0.972897 |
| 16K | 144,384 | 0.455781 | 0.455881 | 0.455960 | 123,444 | 0.455862 | 0.482424 | 0.957597 |
| 32K | 144,384 | 0.427991 | 0.428049 | 0.428091 | 119,939 | 0.428053 | 0.459298 | 0.973135 |
| 64K | 98,304 | 0.401892 | 0.401912 | 0.401928 | 84,238 | 0.401910 | 0.432994 | 0.961436 |
| 128K | 98,304 | 0.377374 | 0.377383 | 0.377394 | 74,702 | 0.377385 | 0.426747 | 0.974907 |

For the median saturated observation, “K*=full” means only that roughly
38–49% of full failed and full passed. The entire interval between those points
was never evaluated. The occasional absolute-policy point makes the maximum
resolution much finer, but it does not repair the median case.

Therefore the legacy result is **resolution-limited**. The defensible statement
is: “the median query head did not tolerate the large reduction represented by
the penultimate sampled candidate in 24/24 cells.” It is not evidence that no
5%, 10%, or 20% budget reduction would pass.

## Required next measurement

Gate 0 rejects both shortcuts:

- Do not turn the query-head scalar into a GQA budget by taking a max.
- Do not reuse the 12-point search to claim saturation.

The next experiment must construct one physical KV-group ranking, evaluate a
group loss curve, and scan every integer K from 1 through the B=3 full keep
count. The held-out test should report direct group K*, savings, error ratio at
K*, and the complete full-to-K* region. This requires fresh logits/values or
newly recorded sufficient statistics; the current Parquets cannot supply it.

## Reproduction

```bash
cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant

# Dependency-free arithmetic smoke test
python h0_measurement/bugs/10_kstar_budget/audit_existing.py --self-check

# Full authenticated CPU audit
module --force purge
module load StdEnv/2026 python/3.14 arrow/25.0.1
python h0_measurement/bugs/10_kstar_budget/audit_existing.py
```

The full audit printed PASS for all 24 artifact pairs and all 629,760 selected
rows, reproduced 24/24 legacy medians, and emitted the two tables above.

---

## Physical-group qualification - job 985183

### Authenticated execution

The pre-output protocol is qualification_protocol_frozen.md. Its source-ledger
content SHA-256 is
fcbdeb48d88b16e9026d8a8f36d4b58378f1b5b29095eefbd6d2b041a924336a;
the complete input-manifest content SHA-256 is
33e6e064970085571db3e4e2e041316b97c160cd537d24b2286cd0d7deaeb8c2.

One two-task array ran the exact 8K/B=3 qualification:

| task | model and role | Slurm state | wall time | inference peak RSS |
|---:|---|---|---:|---:|
| 0 | Llama-3.1-8B, GQA-4 physical-group test | completed, exit 0 | 2m55s | 17.7 GB |
| 1 | Qwen1.5-MoE-A2.7B, MHA G=1 reduction control | completed, exit 0 | 4m00s | 30.4 GB |

Both result directories passed runner validation and a second independent task
read before their exact COMPLETE markers were written. The offline reader then
rehashed all 25 sealed sources, all 40 PG19 files, the complete 16.072 GB Llama
snapshot, the complete 28.644 GB Qwen snapshot, and all ten result artifacts.
The two directories contain exactly the expected seven files.

The reader prints a PyTorch warning when wrapping a read-only Pandas array.
This is code hygiene only: the count functions allocate new tensors and never
write to the inputs. A writable-copy recomputation produced byte-identical
inputs and identical outputs for all five recomputed policies.

### Frozen decision

| gate | frozen criterion | result | status |
|---|---|---:|---|
| Q1 implementation and budget | both tasks authenticate; G=1 max absolute curve difference <= 1e-10; identical K* | max 1.954e-14; all K* identical | pass |
| Q2 calibration stability | Llama p0/p1 K* Spearman >= 0.50 | 0.751734 | pass |
| Q3 nontrivial redistribution | Llama K*-prop movement >= 0.05 | 0.002319 | **fail** |
| Q4 integer resolution | fewer than 95% of Llama units have K*=k0 | 0/256 = 0 | pass |

All gates were conjunctive. The authenticated decision is
**stop_r9_qualification**. The file qualification_advance_lock_985183.json does
not exist. A valid Q3 failure is a scientific stop, so the threshold is not
relaxed and the same prompts are not used to search a new budget, context,
tolerance, or amplified K* rule.

### Why Q3 failed

The direct physical-group curve fixed the old measurement problem: it scans
every integer count, reduces correctly at G=1, and produces a stable rank
ordering on Llama. The resulting scalar demand is nevertheless almost common
mode.

| model | physical units | calibration K* mean / k0 | K* SD | K* range | projected-count range | movement |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 256 | 2948.82 / 3059 = 96.40% | 17.53 | 2867-2984 | 2974-3095 | 0.2319% |
| Qwen1.5-MoE-A2.7B | 384 | 2940.79 / 3059 = 96.14% | 91.37 | 1673-3004 | 1740-3125 | 0.5384% |

For Llama, raw K* sums to 754,897, which is 3.60% below the fixed total
783,104. The equal-total projection faithfully rescales this vector by
1.037365; no bound binds and rank correlation with raw K* is 0.999986. It has
116 groups below, 9 at, and 131 above k0. Its L1 movement is 3,632 slots, so
only 1,816 slots move in one direction. The gate required at least 39,156.
Shrinkage moves still less, 0.1843%.

This is not a projection bug. The raw Llama profile has coefficient of
variation 0.00594, and prompt-specific projected movements are only 0.2511%
and 0.2448%. Averaging the two prompts narrows this to 0.2319%. There is some
repeatable ordering, which explains Q2, but too little amplitude to form a
budget allocator.

The more serious issue is the scalar definition. K* compares every unit to
1.10 times that unit's E_full, discarding the absolute size of E_full. Llama
full-budget errors span about 4,869x, yet K* and full-budget error have Spearman
-0.8009. The 116 projected donors contain 84.96% of calibration
full-budget-error mass, while the 131 receivers contain 11.44%. On the held-out
prompts every donor's aggregate error rises and every receiver's falls, but
donor harm dominates receiver gain. Qwen has the same direction: Spearman
-0.4862, with 75.62% of full-budget-error mass in its donors.

Thus the own-baseline relative threshold is useful as a per-unit tolerance
diagnostic but points the wrong way for a fixed-sum global objective.

### Descriptive held-out results

Held-out prompts 2 and 3 did not enter Q1-Q4 and cannot select a policy. The
table reports pooled mean squared relative output-error ratios to fixed and the
number of improved prompt/family cells out of six.

| equal-memory policy | Llama ratio | Llama wins | Qwen ratio | Qwen wins |
|---|---:|---:|---:|---:|
| K* proportional, calibrated | 1.0121 | 0/6 | 1.0238 | 0/6 |
| K* shrink-20%, calibrated | 1.0093 | 0/6 | 1.0194 | 0/6 |
| kappa=4 rebalanced, calibrated | 1.6234 | 0/6 | 0.8155 | 6/6 |
| AdaKV alpha=0.2, calibrated | 1.0298 | 1/6 | 0.6257 | 6/6 |
| layer-flat, calibrated | 1.1744 | 0/6 | 0.7171 | 6/6 |
| SnapKV global, calibrated | 1.6486 | 0/6 | 0.8308 | 6/6 |
| AdaKV alpha=0.2, dynamic | 1.0137 | 3/6 | 0.5802 | 6/6 |

The two K* arms lose all 12 model/prompt/family cells; shrinkage reduces but
does not reverse the harm. More aggressive existing allocators show an
architecture sign reversal: they consistently help the Qwen MHA control and
harm Llama GQA. Dynamic AdaKV is the strongest descriptive alternative, but it
is an existing baseline and this two-prompt qualification cannot promote it
into the proposed design.

Natural kappa controls are not equal-memory comparisons. Calibrated natural
kappa spends 83.02% of the target on Llama and 71.31% on Qwen, yet its error
ratios are 2.0825 and 1.4925. Dynamic natural kappa also loses on both models.
The raw K* sum suggests a separate hypothesis of roughly 3.6-3.9% common
memory slack, but that is a memory-saving question requiring a new protocol; it
does not rescue equal-total redistribution.

### Design conclusion and next branch

This result rejects the **K*-proportional budget subdesign**. It does not by
itself reject per-KV-head scoring, the cascade score, tiering, or the whole
Sieve architecture. It establishes:

1. direct physical-group measurement and the G=1 reduction are correct;
2. the K* rank is repeatable enough for Q2 and is no longer grid-saturated;
3. its dynamic range is about 22 times too small for the prespecified movement
   gate; and
4. its relative normalization allocates away from the units that dominate the
   global output-error objective.

R9 is closed without development. The roadmap now returns to R1 document
reconciliation and then R10 tier-set re-derivation.

If the budget rule is revisited later, the next defensible hypothesis changes
the allocator objective rather than the failed gate. A separate preregistered
branch should:

- extend calibration curves above k0, ideally over 0..C, because the current
  curves stop at k0 while an equal-total allocator gives some recipients more
  than k0;
- choose counts from a whole-policy constrained objective that minimizes the
  sum of E_u(k_u) subject to the exact total, retaining absolute curve
  magnitudes or frozen marginal gains instead of an own-baseline K* threshold;
- freeze how nonmonotone curves are handled, the exact optimizer, movement and
  stability gates, prompt IDs, and confirmation split before output; and
- compare against fixed and AdaKV on fresh prompts and architectures, followed
  by end-task validation if output-error transfer again appears promising.

The current qualification artifacts may motivate that future protocol, but
they cannot serve as its test set.

### Reproduction

    bash h0_measurement/bugs/10_kstar_budget/steps.sh --qual-status 985183
    bash h0_measurement/bugs/10_kstar_budget/steps.sh --qual-read 985183

The authenticated outputs are:

- qualification_985183.txt
- qualification_985183_summary.csv
- results/r9_kstar_qualification_985183_0/
- results/r9_kstar_qualification_985183_1/

