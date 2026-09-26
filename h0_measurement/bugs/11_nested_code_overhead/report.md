# R11 report: nested-code rate overhead for the R10 ladder

**Status (2026-09-25): done, valid (`valid_r11`), job 988607.**
**Frozen decision: `pass_target` at B=3 (primary) and at B=2.**

At matched attention-output error, storing the R10 ladder `{0,3,4,6,8}` as one
successively refinable 3+1+2+2 code costs **+0.34% rate across the four cells**
at B=3 [90% CI −0.33, +0.99]. The worst cell is **+2.9%**, against a 10% target
and a 20% kill line (ROADMAP R11). The code itself costs **at most 0.42% rate**
at any width. Most of the end-to-end overhead is the bit that tier-3 tokens
must keep for the 4-bit cascade, which R10 handed to R11.

**Open, with an extension running:** the B=2 result for Qwen3-30B rests on one
prompt out of four. The extension (`plan_ext.md`, jobs 991618–991621) re-tests
that tail on 24 fresh prompts and adds Mistral-7B and Qwen1.5-MoE.

- **Protocol:** `plan.md` (frozen before any R11 output) plus amendments A1
  and A2.
- **Frozen reader:** `read_nested_code.py`; output in
  `main_988607.{txt,json}`.

## 1. What was measured

**The code** (`sievelib/quant.py`, default off):
- The pipeline is TurboQuant-MSE with a fixed rotation and a per-coordinate
  scalar code.
- The 3-bit base is the monolithic Lloyd-Max codebook itself, bit for bit.
- Each refinement stage splits every cell by Lloyd-Max on the cell-restricted
  N(0,1) density. The coarse boundaries are frozen, so an 8-bit index truncates
  to 6, 4 or 3 bits with no re-encoding (pinned by
  `tests/test_r11_nested_codebook.py`).
- There is no side information beyond the per-token norm, which both codes
  store identically.

**The comparison (amendment A2).** Each cell is one `run_h0` process with
`codebook=lloyd codebook_ab=nested3`. At every measured step, both codebooks are
evaluated on the same captured q, K and V, the same mask and the same lagged
evictor scores, before the evictors observe that step:
- Widths 1, 2, 3 and 5 are shared tensors.
- Widths 4, 6 and 8 are re-quantized with the nested code.
- Both arms allocate on the R10 path: nested3 panel, one allocation per
  physical KV group, `accum` interior, `bc=4` cascade, `floor_maxb`.

**Cells:** the four R10 cells, on a fresh prompt block (24–29).

| cell | prompts | wall time |
|---|---|---|
| llama31-8b @32K | 6 | 2h05m |
| llama31-8b @128K | 6 | 2h49m |
| qwen3-8b @8K | 6 | 2h06m |
| qwen3-30b-a3b-2507 @8K | 4 | 1h49m |

Budgets were B=1–4, used for rate matching; the analysis uses step 4 rows. All
22 NIAH prompts retrieved their needle.

**Statistic (plan §6):**
- `O_total = (B + f3) / B* − 1`.
- `B*` is the budget at which the monolithic curve's prompt-RMS error equals
  the nested arm's error at B.
- `f3` is the tier-3 share of physical tokens, each charged +1 bit for 4-bit
  rescoring. This is conservative: the bit is charged, but the lower error it
  would buy is not credited.
- Prompt is the resampling unit; the average gives each cell equal weight.

## 2. Validity: V1–V5 pass

- **V1:** the two arms are bit-identical on every codebook-free column at
  relative 1e-6 on every row, which proves they saw the same inputs.
- **V2:** the nested code was applied on ≥99% of rows at widths 4, 6 and 8.
- **V3:** no allocation overspends, and tier fractions sum to 1.
- **V4:** the ledger (`3ab7351e…f497`, 11 sources), sidecars, `codebook_ab`
  layout and real PG-19 corpus all authenticate.
- **V5:** all errors are finite and positive.

## 3. Results

### 3.1 B=3 (primary)

| cell | nested / mono error | B* | f3 | O_nest | **O_total [90% CI]** |
|---|---|---|---|---|---|
| llama31-8b @32K | 1.018 | 2.976 | 5.2% | +0.82% | **+2.55%** [+2.11, +2.98] |
| llama31-8b @128K | 1.032 | 2.946 | 2.8% | +1.93% | **+2.88%** [+0.83, +4.82] |
| qwen3-8b @8K | 1.005 | 3.004 | 3.8% | −0.10% | **+1.17%** [−0.23, +2.52] |
| qwen3-30b @8K | 0.949 | 3.213 | 4.4% | −6.60% | **−5.22%** [−6.31, −4.18] |
| **average of 4 cells** | 1.001 | | 4.1% | −0.99% | **+0.34%** [−0.33, +0.99] |

Per prompt, the dense cells range from −2.1% to +6.4%. No prompt comes near
10%.

### 3.2 B=2 (secondary)

| cell | nested / mono error | O_total [90% CI] | per prompt |
|---|---|---|---|
| llama31-8b @32K | 1.007 | +2.78% [+2.64, +2.92] | +2.5 … +3.1% |
| llama31-8b @128K | 1.018 | +2.23% [+1.70, +2.73] | +0.8 … +3.2% |
| qwen3-8b @8K | 1.005 | +2.01% [+1.51, +2.49] | +1.0 … +3.1% |
| qwen3-30b @8K | 1.162 | **+12.91%** [−2.32, +42.91] | −1.9, −1.9, −2.8, **+58.1%** (prompt 27) |
| **average of 4 cells** | 1.046 | **+4.98%** [+1.09, +12.29] | |

B=2 passes the frozen rule: every cell is ≤20% and the average is ≤10%. The
Qwen3-30B cell, however, is carried by one prompt (§4).

### 3.3 Code level (no allocation)

This is the per-width logit-noise variance, `c{b}_abs`, nested vs. monolithic,
converted to a rate on the monolithic curve.

| cell | 4 bits | 6 bits | 8 bits |
|---|---|---|---|
| llama31-8b @32K | +0.03% | +0.33% | +0.38% |
| llama31-8b @128K | +0.06% | +0.36% | +0.42% |
| qwen3-8b @8K | −0.01% | +0.26% | +0.29% |
| qwen3-30b @8K | −0.09% | +0.25% | +0.21% |
| Gaussian design (offline) | +0.19% | +0.68% | +0.86% |

The nested code costs less on real rotated keys than on a Gaussian source. At
every width it is at least 20× below the 10% target.

### 3.4 Where the end-to-end overhead comes from (dense cells, B=3)

- **Nesting:** −0.1 to +1.9 points (`O_nest`).
- **The observation bit:** f3/B* ≈ 1.0–1.7 points. At B=3 this is the larger
  of the two terms.

## 4. Allocator knife-edges (exploratory; `knife_edge.py`, `knife_edge.txt`)

**The Qwen3-30B numbers are not code distortion.** The Qwen3-30B code-level
overhead is at most 0.25%, yet its allocation error moves −5% at B=3 and
+58% on one prompt at B=2. To understand why, two small perturbations measured
on identical inputs were analysed, one physical KV group at a time:
- **P1:** the nested codebook, which changes the per-width noise table by a
  median |log| of 0.06–0.10.
- **P2:** R10's `full → no1`, which removes a tier used by ≤0.1% of tokens.

| | llama @32K | llama @128K | qwen3-8b | qwen3-30b |
|---|---|---|---|---|
| P1 groups ≥2× worse, B=3 | 3.5% | 4.8% | 4.5% | 2.4% |
| P1 groups ≥2× better, B=3 | 3.1% | 4.0% | 3.5% | 2.2% |
| largest group's share of a prompt's err², B=3 | 5–7% | 2–3% | 4–6% | **8–22%** (always layer 3) |
| top group's share of the cell's \|ΔSSE\|, B=2 | 0.8% | 2.1% | 1.4% | **83%** |

**Findings:**
1. **The allocator is deterministic but discontinuous.** Under P2, groups whose
   options did not change stayed bit-identical. Only the 1.1% of Llama groups
   that used tier 1 moved, some by up to 5.9×. The water filler optimises a
   lagged `accum` proxy with a 4-bit cascade estimate, so small changes to its
   noise table can push it across a discontinuity. R10's `no1` beating `full`
   on llama @32K is the same effect.
2. **Flips are common, and in dense models they cancel.** No KV group holds
   more than about 8% of a prompt's error, so ≥2× flips in either direction
   average out. A slot-stratified resampling check (each KV-group slot drawn
   from a random observed prompt) gives P(prompt O_total > 10%) ≈ 0 in all
   three dense cells.
3. **In Qwen3-30B they do not cancel.** Layer 3 concentrates error: its largest
   group holds 8–22% of each prompt's squared error. One such group flipped at
   B=2 on prompt 27 (layer 3, KV head 3, `cont`: error 0.59 → 4.25, identical
   inputs, identical eviction fraction). The tail comes from **error
   concentration**, not from a higher flip rate: Qwen3-30B's flip rate is the
   lowest of the four cells.
4. **The knife-edge is a property of the allocator, not the code.** It will
   appear under any change to the noise table. R14, or a follow-up, needs an
   allocator with hysteresis or a smoother score. Nesting does not make this
   worse.

## 5. How much to trust what (22 prompts, 4 cells)

| claim | trust | why |
|---|---|---|
| code-level ≤0.42% rate | **high** | exactly paired, hundreds of thousands of rows, identical across cells, steps and the invalid and valid runs |
| B=3 dense cells: 1–3% | **high** | per-prompt SD ≤3.2 points; worst one-sided 95% upper bound ≈5.5%; no prompt >6.4% |
| B=3 passes overall | **high** | worst cell's upper interval is +4.8%, far below 10% |
| Qwen3-30B B=3 (−5%) | medium | all 4 prompts agree in sign, but it is an allocator effect, not a code gain |
| B=2 passes for dense models | high | all 18 dense prompts +0.8 … +3.2% |
| **B=2 passes for MoE** | **low** | 1 of 4 prompts at +58%; tail rate 95% CI 0.6–81%; the extension re-tests it |
| generality beyond 3 architectures | not established | the extension adds Mistral-7B and Qwen1.5-MoE |

**Safe wording for the paper:** "The nested 3+1+2+2 code costs ≤0.42% rate at
every width on real keys. End to end at B=3 it costs +0.34% on average (worst
cell +2.9%), mostly the retained 4-bit observation bit. B=2 passes on average;
one MoE prompt shows an allocator discontinuity."

## 6. History of the run (why three main attempts)

| job | layout | outcome |
|---|---|---|
| 987079 / 987080 | arms as separate array tasks | pilot V1 failed: arms on different nodes diverged from decode step 4 |
| 987134 | both arms in one job (A1) | cancelled by root after 3 s; no artifact |
| 987151 / 987153 / 987154 | A1 | 2K pilot passed; **main `invalid_r11`**: even on one GPU, 3–14% of rows differed at step 0 and 34–60% by step 4 |
| 988603 / 988607 / 988608 | in-process A/B (A2) | **valid** |

**Lesson:** `run_h0`'s forward pass is not bit-reproducible run to run at
≥8K, even on the same GPU; MoE is worst (up to 3.6% on a codebook-free
per-prompt error). Any row-paired A/B must compute both arms inside one
process, as R10's tier panel and R11 A2 do.

The exploratory analysis of the invalid run (`explore_987153.json`) estimated
the dense cells at +1.2 to +2.7% at B=3. The valid run gives +1.2 to +2.9%;
every cell moved less than 0.4 points.

## 7. Consequences

1. **R11 → R14 is open.** The nested 3+1+2+2 code is the storage format, with
   the first refinement bit retained for tier-3 tokens so that the `bc=4`
   cascade can read a 4-bit prefix. It is priced at 1.0–1.7 points of rate at
   B=3.
2. **Scope for now:** B=3 is architecture-general across the tested models. B=2
   is dense-only until the extension reports.
3. **Two follow-ups:**
   - Observation bit: is `bc=3` rescoring good enough to drop the retained bit?
     This would be a config-only in-process A/B (`coarse_bits=3,4`).
   - Allocator smoothing (hysteresis or damping) against the knife-edge
     documented in §4.

## 8. Files

| file | contents |
|---|---|
| `plan.md` | frozen protocol plus A1 (same job) and A2 (in-process A/B) |
| `read_nested_code.py` | frozen reader; `--main-ab 988607` reproduces `main_988607.{txt,json}` |
| `knife_edge.py`, `knife_edge.{txt,json}` | §4 allocator diagnostic (exploratory) |
| `explore_paired_subset.py`, `explore_987153.json` | analysis of the invalid run (superseded; kept for the record) |
| `plan_ext.md`, `read_nested_ext.py`, `script_ext.sh`, `h0_measurement/submit_r11_ext.slurm` | R11-ext (running: 991618 pilot passed, main 991620) |
| `sievelib/quant.py` | `NESTED_CHAINS`, `design_nested`, `nested_codebook`, `codebook_differs`, `quantize_keys(codebook=)` |
| `h0_measurement/run_h0.py` | `codebook=` and `codebook_ab=` (in-process second codebook); both off by default |
| `tests/test_r11_nested_codebook.py` | default path unchanged; nesting; shared base; the A/B pass is independent and does not mutate its inputs |
| `script.sh` | `--seal`, `--run-dry`, `--run` (pilot → gate → main → analysis), `--status`, `--cancel`, `--pilot-read`, `--main-read` |
