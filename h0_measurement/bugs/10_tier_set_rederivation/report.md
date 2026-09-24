# R10 report: tier-set re-derivation and the base-rate decision

**Status (2026-09-24): done, valid (`valid_r10`).** The frozen decision is
`retain_nested3_price_4bit_observation_b2_b3`.

**Headline.** The 3+1+2+2 ladder `{0,3,4,6,8}` (`nested3`) keeps the current
dense ladder's benefit at both budgets. Every frozen gate G1–G5 passes at B=3
and at B=2. The directly re-budgetable 4+2+2 ladder `{0,4,6,8}` (`nested4`) fails
its selection test against `nested3` at both budgets. So R11 must measure and
count the retained +1 observation refinement that the `bc=4` cascade needs, or
the online cascade has to change (plan §10).

Protocol: `plan.md`, frozen before any R10 output. Reader:
`read_tier_set.py`. Raw analysis: `main_986347.{txt,json}`,
`main_986347_summary.csv`.

## 1. What ran

| step | Slurm job | result |
|---|---|---|
| excluded pilot (qwen3-1.7b @2K; qwen15-moe @8K, n_rep=1) | 986344_[0-1] | COMPLETED; 1m49s / 4m01s |
| pilot gate (reader `--pilot`) | 986346 | PASS V1–V5; wrote `pilot_advance_lock_986344.json` |
| locked primary array | 986347_[0-3] | all COMPLETED; 45m / 1h07m / 42m / 39m |
| main analysis (reader `--main`) | 986350 | `valid_r10` |

Source ledger sha256 `0330be93…ba1688` (13 sources). Every worker and both
reader jobs verified it. Corpus: real PG-19, `corpus_sha=0a26bc1e05a1eea8`.

| task | cell | prompts | role |
|---:|---|---|---|
| 0 | llama31-8b @32K | 6 @ 18 | GQA-4, shorter context |
| 1 | llama31-8b @128K | 6 @ 18 | GQA-4, long context |
| 2 | qwen3-8b @8K | 6 @ 18 | GQA-4, largest cascade gain |
| 3 | qwen3-30b-a3b-2507 @8K | 4 @ 12 | GQA-8 control |

Fixed design: one allocation per physical KV head, `accum` interior score,
`bc=4` cascade, the `floor_maxb` rule, rotation seed 2, step-4 rows.

## 2. Validity (V1–V5): all pass

- **V1:** the panel's `full` arm reproduces the existing `grp_csv_b4_accum`
  allocation exactly: zero mismatches and zero bit differences, with identical
  exact error per row.
- **V2:** no absent tier is used, nothing overspends, and unused budget is at
  most 8/L in every physical group, ladder and budget.
- **V3:** provenance authenticates for every task: RUN_INFO, sidecar,
  task-local ledger and the pilot lock.
- **V4:** the grids are complete, all primary values are finite, and every NIAH
  prompt retrieves its needle.
- **V5:** on the n_rep=1 control (qwen15-moe), restricted group allocation is
  identical to restricted per-head allocation for all 5 ladders × 2 budgets.

## 3. Primary result: B=3

Error ratio is prompt-RMS exact attention-output error relative to `full`,
geometric mean over prompts. Retention is routed gain relative to `full`.

| cell | no1 | base3_dense | **nested3** | nested4 |
|---|---|---|---|---|
| llama31-8b @32K | 0.982 | 1.002 | **1.016** | 1.258 |
| llama31-8b @128K | 1.003 | 1.022 | **1.031** | 1.117 |
| qwen3-8b @8K | 1.000 | 1.003 | **1.018** | 1.059 |
| qwen3-30b @8K | 1.000 | 1.001 | **1.014** | 1.027 |
| **macro** (90% CI) | | | **1.019** [1.018, 1.021] | 1.112 [1.106, 1.117] |

**`nested3` gates (plan §8.1): PASS**

| gate | threshold | observed | |
|---|---|---|---|
| G1 median cell error ratio | ≤ 1.05 | 1.017 | PASS |
| G2 worst cell error ratio | ≤ 1.10 | 1.031 (llama @128K) | PASS |
| G3 macro routed-gain retention | ≥ 0.95 | 0.986 | PASS |
| G4 each cell's retention | ≥ 0.90 | min 0.984 (qwen3-30b) | PASS |
| G5 each cell's band loss | ≤ 5 pts | worst −1.25 pts (qwen3-30b) | PASS |

**Sequential attribution (plan §8.3): no step fails.** Each removal costs
about 1%:

| transition | median | worst | |
|---|---|---|---|
| full → no1 | 1.000 | 1.003 | PASS |
| no1 → base3_dense | 1.011 | 1.020 | PASS |
| base3_dense → nested3 | 1.013 | 1.015 | PASS |

**`nested4` versus `nested3` (plan §8.2): FAIL; do not select**

| criterion | threshold | observed | |
|---|---|---|---|
| median cell error ratio n4/n3 | ≤ 1.02 | 1.062 | FAIL |
| every cell's ratio | ≤ 1.05 | worst 1.239 (llama @32K) | FAIL |
| macro retention vs nested3 | ≥ 0.98 | 0.960 | FAIL |
| nested3 tier-3 occupancy < 5% in every cell | < 0.05 | 0.052 (llama @32K) | FAIL |

The fourth criterion misses narrowly, but the first three fail by wide
margins, so the conclusion does not depend on it. Removing the 3-bit rung is
costly: a 24% error increase in llama31-8b @32K and 8% at 128K. The two Qwen
cells are hurt less (1–4%).

## 4. Secondary result: B=2 (plan §9)

`nested3` passes G1–G5 at B=2 as well, so one ladder covers B=2 and B=3.

| gate | observed |
|---|---|
| G1 median error ratio | 1.036 |
| G2 worst | 1.070 (llama @32K) |
| G3 macro retention | 0.985 |
| G4 min retention | 0.982 |
| G5 worst band loss | −1.19 pts |

The margin is thinner than at B=3. The Llama cells cost 6–7%, against a 10%
worst-cell limit. Most of that cost comes from removing the 2-bit rung
(`no1 → base3_dense`: llama @32K 1.061, llama @128K 1.046). At B=2 the 2-bit rung
does real work for Llama, but not enough to fail a frozen gate. `nested4` fails
again: median n4/n3 1.148, worst 1.254, retention 0.956. Its tier-3 criterion
passes at B=2 (max 4.8%).

## 5. Tier occupancy (physical tokens, L-weighted, `full` ladder)

| cell, B=3 | t0 (evict) | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| llama @32K | 46.5% | 0.07% | 0.32% | 4.9% | 11.6% | 10.7% | 11.1% | 14.7% |
| llama @128K | 50.5% | 0.05% | 0.25% | 2.5% | 8.9% | 8.8% | 10.1% | 19.0% |
| qwen3-8b @8K | 48.8% | 0.00% | 0.09% | 3.6% | 9.5% | 10.2% | 11.2% | 16.6% |
| qwen3-30b @8K | 47.4% | 0.00% | 0.19% | 4.3% | 9.4% | 11.7% | 12.8% | 14.2% |

- **The 1-bit tier is dead**, as the roadmap stated: at most 0.1% of physical
  tokens in the Llama cells and exactly 0 in the Qwen cells. For the Qwen cells,
  `no1` is therefore bit-identical to `full` (ratio 1.0000).
- **The 2-bit tier is nearly dead** (0.1–0.4%), yet removing it still costs 1–2%
  at B=3 and up to 6% at B=2 in Llama. The tokens are few but high-leverage.
- **Tier 3 is small (2.5–5%) but not removable.** When `nested4` removes it,
  mass shifts to eviction (+1–2 pts) and to 4 bits, and error rises by
  3–26% at B=3.
- **Eviction stays inside the allocator** at 47–51% (B=3) and 62–65% (B=2),
  consistent with C3.

## 6. Observations outside the frozen gates

- **Removing an option lowered error once.** `no1` in llama @32K, B=3, has error
  ratio 0.982 (CI [0.980, 0.985]). The water filler optimises a lagged `accum`
  proxy, not the exact error it is scored on, so a smaller action set can land
  closer to the exact optimum. The same effect is already recorded as "provably
  monotone is false per head" (ROADMAP §6). It is diagnostic only and selects
  nothing.
- **The bootstrap intervals are very narrow** (±0.1–2%). The intervals resample
  prompts within a cell, and every ladder shares the same prompts, rotations and
  quantized logits, so they measure paired variation between ladders. They are
  not a spread across architectures. The architecture spread shows in the
  per-cell ratios: Llama loses more than Qwen at every step of the removal chain.
- **Llama @32K is the binding cell** for `nested4` at both budgets and for
  `nested3` at B=2. If a later campaign widens the model panel, this regime is
  where to look for a failure.

## 7. Decision and consequences (plan §10)

**Frozen decision:** `retain_nested3_price_4bit_observation_b2_b3`.

1. **Selected ideal ladder: `{0,3,4,6,8}`**, i.e. a 3-bit base with
   refinements of +1, +2 and +2. It is scoped to B=2 and B=3. The 1-, 2- and
   5-bit rungs are removed.
2. **`nested4` is rejected** on these prompts. Do not retune a 4-bit-base
   ladder on this prompt block.
3. **R11 inherits a cost item.** A token stored at only 3 bits cannot supply
   the `bc=4` cascade's 4-bit rescoring. R11 must either:
   - (a) price an always-retained first refinement bit for every token that is
     not evicted. At B=3 this adds up to about 1 bit on the 2.5–5% of physical
     tokens at tier 3, so at most about 0.05 bits/token; or
   - (b) test a `bc=3` cascade instead. Co-design measured `bc=4` as closing a
     median 89% of the lag gap, and `bc=3`'s closure would need to be checked
     against that before switching.

   Option (a) looks cheap enough that it is the likely default.
4. **R11 is unblocked.** Its question is now concrete: does a nested 3+1+2+2
   code cost more than 10% (target) or 20% (architecture stop) in rate against
   monolithic widths at matched distortion?

R10 does not measure end-task accuracy, packed memory or latency. Those belong
to R8, R11 and R14.

## 8. Engineering notes

- `script.sh` gained `--seal` (writes `source_ledger.json`, which was missing,
  so preflight could not pass before), `--run-dry`, `--run` (submits
  pilot → gate → main → analysis, chained with `afterok`), and `--chain-status`
  / `--cancel-chain`.
- **Known defect in `--run`:** it submits the analysis job to `debug`. The trig
  debug QOS allows only one queued job per user, so the analysis job was
  rejected (`QOSMaxSubmitJobPerUserLimit`) while the gate job was queued there.
  The analysis job 986350 was resubmitted by hand on `compute` with the same
  command. The fix is `REPORTER_SLURM=(--partition=compute …)`, but it was not
  applied: `script.sh` is a sealed ledger source, and editing it would make
  `--main-read 986347` fail re-authentication with a source-drift error. Apply
  the fix and `--seal` again only for a new campaign.
- `tests/test_r10_tier_set_reader.py`: the quantized-rows negative test used
  `frame.head`, which is the pandas method, not the `head` column. The mutation
  was a no-op and the test failed. Fixed to `frame["head"]`. The file is not in
  the ledger.
