# R11-ext: MoE budget-2 tail and two new architectures

**Protocol:** `r11_nested_code_ext_v1`. **Frozen:** 2026-09-25, before any
R11-ext GPU output.
**Parent:** R11 (`plan.md` + A1/A2), frozen verdict `pass_target` at B=3 and
B=2 from job 988607.
**Why this exists:** R11 rests on 22 prompts in 4 cells and 3 architectures.
Its B=2 Qwen3-30B cell is carried by one prompt (27, +58%), so it cannot
estimate how often such tail events occur (1/4 has a 95% CI of 0.6–81%). R11
also cannot say whether its result carries over to other architectures.

## 1. What is already known (exploratory, from existing data; `knife_edge.txt`)

These numbers informed the design below. They are not evidence for this
protocol's verdict.

- **Allocator knife-edges are common at the KV-group level in every cell.**
  Under the nested codebook's small change to the noise table (median |log|
  0.06–0.10), 1.3–4.8% of physical KV groups move ≥2× up and a similar
  fraction move ≥2× down. Under R10's `full→no1` change, only the groups that
  used tier 1 change (1.1% in Llama, 0% in Qwen), some by up to 5.9×. Groups
  whose options did not change stay bit-identical, so the allocator is
  deterministic: the flips are discontinuities, not noise.
- **In dense cells the flips cancel.** Each prompt's largest KV group holds
  2–8% of its total squared error, so the ≥2× flips average out. The prompt
  spread at B=3 is −2% to +6.4%. A slot-stratified bootstrap gives
  P(prompt O_total > 10%) ≈ 0 in all three dense cells.
- **In Qwen3-30B they do not cancel.** Layer 3 holds a prompt's largest KV
  group, with 8–22% of its squared error at B=3. At B=2 a single layer-3 group
  carries 83% of the cell's total |ΔSSE|. The tail is caused by the
  concentration of error mass in one layer, not by a higher flip rate (Qwen3-30B's
  flip rate is the lowest of the four cells).
- **Implication:** a cell's tail risk should be predictable from its error
  concentration (the top KV group's share of each prompt's squared error),
  which is measurable in the monolithic arm alone.

## 2. Questions

- **Q1 (MoE B=2 tail).** On fresh prompts, how often does Qwen3-30B @8K
  produce a prompt with O_total > 20% at B=2? Does the cell mean clear 10% and
  20%?
- **Q2 (generalization).** Does the R11 result, nested 3+1+2+2 costing ≤10% at
  B=3, hold in two architectures not in R11?
  - **Mistral-7B-Instruct-v0.3:** dense, GQA-4 (32 query / 8 KV heads), 32K
    native context, full attention (`sliding_window: null`).
  - **Qwen1.5-MoE-A2.7B-Chat:** a second MoE, and the registry's only MHA model
    (n_rep=1), so no KV group is shared by several query heads.
- **Q3 (mechanism, secondary).** Does per-prompt error concentration predict
  the prompt-level tail?

## 3. Design (R11 A2 layout, unchanged contract)

Each array task is **one** `run_h0` process with `codebook=lloyd
codebook_ab=nested3`, so both codebooks are measured from the same captured
tensors. Every other setting matches R11 exactly:

- `bit_list=1,2,3,4,5,6,8`, `budgets=extra_budgets=1,2,3,4`
- `tier_panel=nested3`, `group_alloc`, `coarse_bits=4`, `interior_scores=accum`
- `evictors=oracle,accum`, `corner_policies=frac`, `rot_seed=2`
- `n_decode=8`, `quant_every=4`, families `niah,qa,cont`; the reader uses step 4

**Pilot (excluded; implementation only).** Array 0–1, run through
`run_h0` + A2 before any main task:

| pilot task | cell | prompts |
|---|---|---|
| 0 | mistral-7b @2048, `cont` | 1 @ 0 |
| 1 | qwen15-moe-a2.7b @2048, `cont` | 1 @ 0 |

Mistral has never been run through `run_h0`. Both pilot tasks must pass
V1–V5.

**Main (locked).** Array 0–9. A cell may span several tasks; prompt blocks
never overlap.

| cell | tasks | model @ ctx | prompts (offset) | role |
|---|---|---|---|---|
| X1 | 0–5 | qwen3-30b-a3b-2507 @8K | 6 × 4 = **24** (28, 32, 36, 40, 44, 48) | Q1: MoE B=2 tail, fresh prompts |
| X2 | 6–7 | qwen15-moe-a2.7b @8K | 2 × 6 = **12** (24, 30) | Q2: second MoE, n_rep=1 |
| X3 | 8 | mistral-7b @8K | **6** (24) | Q2: new dense GQA-4 |
| X4 | 9 | mistral-7b @32K | **6** (24) | Q2: new dense GQA-4, longer context |

**Prompt blocks.**
- Qwen3-30B @8K has used 0, 4–15 and 24–27 before, so 28–51 is fresh.
- Qwen1.5-MoE has used 0 (8K) and 0–5 (16K). Mistral has never been run.
- At 8K every corpus book qualifies as a full window, so book = `key mod 40`.
  Keys 40–51 therefore reuse the books of keys 0–11, but at key-seeded offsets,
  so every window is new text.

**Sample sizes.**
- **X1 (24 prompts).** Section 1's mixture suggests a per-prompt tail rate
  near 0.2, with events around +60% and the rest near −2%. That gives a
  per-prompt SD of about 24 points, so n=24 gives a standard error of about
  5 points. This separates the 20% stop line from a cell mean near 10%, but
  may not separate 10% from, say, 7%.
- **X2 (12 prompts):** the second MoE, where a tail is possible.
- **X3/X4 (6 prompts each):** dense cells. R11's dense cells had per-prompt SD
  ≤3.2 points, so 6 prompts already bound the mean within ~3 points.

## 4. Validity gates (a failure means `invalid_r11_ext`, not a verdict)

**V1–V5 exactly as in R11 §5,** applied per task, since pairing is within a
process:

- **V1** pairing identity at relative 1e-6
- **V2** nesting applied to ≥99% of rows at widths 4, 6, 8
- **V3** budget and tier fractions
- **V4** provenance: ledger, sidecar, `codebook_ab` layout, prompt block, real corpus
- **V5** finite, positive errors

**One addition.** Every task of a cell must report the same model, context,
families and contract. The prompt sets across tasks must be disjoint and must
union to the planned block.

## 5. Statistics (per cell; prompt = resampling unit)

**Per-prompt O_total.** These are R11 §6's definitions exactly: prompt-RMS
error per arm, matched rate B* on the monolithic curve over B ∈ {1,2,3,4}, and
`O_total = (B + f3)/B* − 1`.

**One pre-specified change.** R11's reader declares a prompt invalid if its
monolithic curve is not strictly decreasing in B. Here, B* is instead computed
on the curve's running minimum over B (the best error available at ≤B), and
the prompt is flagged and counted. This change is made so that one odd prompt
in a new architecture cannot void a cell. It does not change any value on a
strictly decreasing curve.

**Reported per cell and budget:**
- mean O_total, with a 90% interval from 10,000 PCG64(0) prompt resamples
- median O_total
- `tail20` = fraction of prompts with O_total > 20%, with a Clopper-Pearson
  95% interval
- max prompt O_total
- f3 (the tier-3 share)
- nested/monolithic error ratio

## 6. Frozen decisions

These use cells X1–X4 only. R11's own cells keep their verdict and are not
pooled into these gates.

**B=3 (primary, Q2):**

| label | condition |
|---|---|
| `b3_stop` | macro mean O_total(X1–X4) > 0.20 |
| `b3_scope` | macro ≤ 0.20, but some cell mean > 0.20 |
| `b3_priced` | every cell ≤ 0.20 and 0.10 < macro ≤ 0.20 |
| `b3_generalizes` | every cell ≤ 0.20 and macro ≤ 0.10 |

**B=2 (Q1 and scope):**

| label | condition |
|---|---|
| `b2_fail` | some dense cell (X3 or X4) mean > 0.20 |
| `b2_dense_only` | X3 and X4 ≤ 0.20, some MoE cell (X1 or X2) > 0.20 |
| `b2_priced` | every cell ≤ 0.20, some cell > 0.10 |
| `b2_all` | every cell ≤ 0.10 |

**Q1 secondary.** X1's `tail20` and its Clopper-Pearson interval at B=2 are
reported whatever the label. If X1's cell mean at B=2 is ≤ 0.20 but its
`tail20` upper bound exceeds 0.25, the paper states the B=2 MoE result as
"passes on average with a per-prompt tail".

## 7. Mechanism (Q3, secondary, pre-specified)

**Per prompt:**
- `top_share`: the largest physical KV group's share of that prompt's squared
  error in the monolithic arm at the same B;
- the ≥2× flip rate of its KV groups.

**Prediction (stated now):**
- In X3 and X4, `top_share` ≤ 0.10 in every prompt, and no prompt has
  |O_total| > 10% at B=3.
- Across all X-cell prompts at B=2, prompts with O_total > 20% occur only when
  `top_share` ≥ 0.08.

Both are reported as confirmed or refuted. Neither gates anything.

## 8. What follows

- **`b3_generalizes`, with B=2 at `b2_all` or `b2_priced`:** the nested-code
  claim is architecture-general at B=2–3. R14 proceeds on the 3+1+2+2 code.
- **`b2_dense_only`:** the paper states B=2 for dense models only. MoE models
  run at B ≥ 3, or the allocator gets a smoothing or hysteresis fix (a separate
  experiment).
- **`b3_scope` or `b3_stop`:** R14 is blocked for the failing architecture
  class until the cause is diagnosed.
