# R11: nested-code rate overhead for the R10 ladder

**Protocol frozen:** 2026-09-24, before any R11 GPU output.
**Roadmap role:** R11 ("way this dies" #2). Its input is R10 (job 986347),
which selected the ideal ladder `{0,3,4,6,8}` (`nested3`) for B=2 and B=3. R10
also left a cost item for R11: a 3-bit token cannot supply the `bc=4` cascade
observation.

## 1. Question

R10 chose rates assuming each width had its own monolithic Lloyd-Max code. A
deployable tiered cache needs one successively refinable code: a 3-bit base
plus 1, 2 and 2 refinement bits, where any prefix decodes. That way a token can
be demoted or promoted without re-encoding.

**At matched attention-output error, how much more rate does the nested
3+1+2+2 code need than monolithic codes?** The architecture is killed if the
overhead exceeds 20%; the target is 10% (ROADMAP R11).

## 2. The nested code (implemented in `sievelib/quant.py`, default off)

- **Pipeline:** unchanged TurboQuant-MSE: norm, fixed rotation, per-coordinate
  scalar code, norm correction.
- **Base (3 bits):** the monolithic 3-bit Lloyd-Max codebook itself, bitwise
  identical.
- **Refinement:** each stage splits every current cell into `2^(b_k − b_{k−1})`
  sub-cells by Lloyd-Max on the N(0,1) density restricted to that cell. This is
  a greedy tree-structured design: coarse boundaries are frozen, and the new
  boundaries and all reconstruction points are optimised. Every partition
  refines the coarser ones exactly (pinned by
  `tests/test_r11_nested_codebook.py`), so an 8-bit index truncates to any
  width in the chain.
- **Other widths (1, 2, 5)** stay monolithic. They appear only as uniform
  endpoints and as pairing controls, never inside the `nested3` allocation.
- **Side information:** none beyond what the monolithic code already stores
  (the per-token norm, which is shared). The storage difference between the
  two arms is therefore purely distortion at a given rate.

**Offline Gaussian design check** (grid-exact, recorded before any run): MSE
ratio nested/monolithic is 1.000 / 1.010 / 1.057 / 1.099 at 3/4/6/8 bits. The
equivalent rate overhead is 0.00 / 0.19 / 0.68 / 0.86%.

## 3. Design: paired arms

Each cell runs twice with identical configuration, prompts, rotation, decode
and evictor state. Only `codebook` differs:

| arm | `codebook` | widths 3,4,6,8 | widths 1,2,5 |
|---|---|---|---|
| mono | `lloyd` | monolithic | monolithic |
| nested | `nested3` | 3 = monolithic; 4, 6, 8 nested | monolithic |

Fixed contract for both arms:
- `bit_list=1,2,3,4,5,6,8`, `budgets=extra_budgets=1,2,3,4`
- `tier_panel=nested3`, allocated on the R10 path (`group_alloc`, `coarse_bits=4`
  on the lagged `accum` interior, `floor_maxb`)
- `evictors=oracle,accum`, `corner_policies=frac`, `rot_seed=2`
- `n_decode=8`, `quant_every=4`, families `niah,qa,cont`; the reader uses step 4

In the nested arm the cascade observes nested 4-bit keys, which is the
deployed design.

## 4. Cells

These are the four R10 cells, on a prompt block disjoint from every earlier
run of each cell (earlier blocks reach 23 at most).

| main tasks | cell | prompts |
|---|---|---|
| 0 mono, 1 nested | llama31-8b @32K | 6 @ 24 (24–29) |
| 2, 3 | llama31-8b @128K | 6 @ 24 |
| 4, 5 | qwen3-8b @8K | 6 @ 24 |
| 6, 7 | qwen3-30b-a3b-2507 @8K | 4 @ 24 (24–27) |

**Excluded pilot:** qwen3-1.7b @2K, `cont`, 1 prompt @0, both arms (tasks 0 and
1). It is an implementation check only.

## 5. Validity gates (a failure means `invalid_r11`, not a verdict)

- **V1 pairing identity.** The two arms have identical step-4 row keys.
  Codebook-independent quantities agree to a maximum relative difference of
  1e-6 on every row. Those quantities are `L`, `tau`, `c{1,2,3,5}_abs` (width 3
  is the shared base) and `err_uniform{1,2,3}`. This proves both arms saw the
  same keys and queries.
- **V2 nesting applied.** In the nested arm, `c4_abs`, `c6_abs` and `c8_abs`
  differ from the mono arm on at least 99% of rows. This rules out a silent
  fallback to the monolithic code.
- **V3 budget.** No `nested3` allocation overspends B, and the tier fractions
  sum to 1.
- **V4 provenance.** The source-ledger hash in RUN_INFO equals the current
  ledger, and the ledger verifies. The sidecar confirms codebook, chain, tier
  panel, budgets, bits, prompt block, a real (non-synthetic) corpus and
  `group_alloc`.
- **V5 completeness.** Every task exists, and all primary values are finite and
  positive.

NIAH retrieval is reported but is not a gate: both arms share one generation,
and R11 measures code distortion, not task success.

## 6. Statistics

For arm `a`, budget `B`, prompt `p` (step-4 rows):
`E_a(p,B) = sqrt(mean err²)` of `err_wf_grp_tier_nested3_csv_b4_accum_B`, taken
over families, layers and query heads.

- **Direct ratio:** `E_nested(p,B) / E_mono(p,B)`.
- **Matched rate `B*(p,B)`:** the budget at which the mono curve `log E_mono(p,·)`,
  piecewise linear over B ∈ {1,2,3,4}, equals `log E_nested(p,B)`. Beyond
  [1,4] the nearest segment is extrapolated and flagged.
- **Nesting overhead:** `O_nest = B / B* − 1`.
- **Observation cost:** `f3(p,B)` is the fraction of physical tokens (one per
  KV group, L-weighted) that the nested arm assigns to tier 3. Each such token
  must also keep its first refinement bit for the 4-bit cascade.
  `O_total = (B + f3) / B* − 1`. This is conservative: the extra bit is charged
  but the lower error it would buy is not credited.
- **Aggregation:** mean over prompts within a cell, then equal weight across
  the four cells (macro). The 90% interval comes from 10,000 PCG64(0) draws
  that resample prompts within cells.
- **Code-level (secondary):** the per-width logit-noise ratio
  `c{b}_abs(nested)/c{b}_abs(mono)` for b ∈ {4,6,8} (per-prompt geometric
  mean), and the equivalent width `r_b` from the mono `c{w}_abs` curve over
  w ∈ {1,…,6,8}, giving `O_code(b) = b/r_b − 1`.

## 7. Frozen decision at B=3 (primary)

| decision | condition |
|---|---|
| `stop_architecture` | macro `O_total(3)` > 0.20 |
| `scope_per_model` | macro ≤ 0.20, but some cell > 0.20 |
| `pass_priced` | 0.10 < macro ≤ 0.20, every cell ≤ 0.20 |
| `pass_target` | macro ≤ 0.10, every cell ≤ 0.20 |

B=2 uses the same rule as a secondary scope result. It is not averaged with
B=3. R14 (kernel / TPOT) proceeds only on `pass_*`.
