# R12-paper: matched baselines for the paper's main end-task table

**Written 2026-09-25, before any job ran.** This folder is new; `bugs/12_*` does
not collide with ROADMAP R12 (GQA union), which co-design answered.

## Why

The R9 table (`bugs/9_sota_eviction_baselines/report.md`) has three gaps a
reviewer will find:

1. **No matched H2O.** `evict_h2o` was not an arm in jobs 978480–978489.
2. **Only one quantization baseline.** R8's `uniform` arm is TurboQuant_mse
   (`sievelib/quant.py`), so a quantization baseline exists, but KIVI and
   KVQuant (the two most-cited key-quantization families) do not.
3. **Ceiling.** TurboQuant is ≥ 0.95 in 31/36 valid cells, so the table can
   rank methods mainly on Qwen and on the eviction side.

## What runs

**A. Main grid, fully paired.** The same five cells, prompts 100–119, the same
four tasks, B = 2, 3, question-agnostic, W = 32, and the existing router
calibrations (`results/r8_routes/*_qa.json`, prompts 0–9). **Every arm runs in
one process** (the forward is not bit-reproducible across runs; ROADMAP methods
rule), so the new table replaces the R9 table instead of being spliced onto it:

`fp, uniform (TurboQuant), evict (SnapKV), evict_h2o (H2O), interior_cascade,
router_calib (SIEVE), adakv, dropkv, obck_ada, laprox, kivi, kivi_g128, kvquant`

**B. Non-ceiling cell.** Llama-3.1-8B @ 32K, `niah_multikey` at k32/v4/h4, the
operating point where op2 job 982122 measured FP 1.000 and TurboQuant 0.825
(Δ 90% CI [0.075, 0.275], 40 prompts). The same arm list and budgets.
- Router calibration: prompts 1100–1109 at k32/v4/h4, `niah_multikey` only.
- Evaluation: **fresh prompts 1000–1059** (60). Every index below 820 is used
  or reserved by R8/R9; 460–499 and 740–819 are closed R9 partitions and stay
  unopened.

## New code (additive, R8 exception)

- `sievelib/kv_quant_baselines.py` (new): KIVI (per-channel, groups of 32 or
  128 tokens), and KVQuant-style (pre-RoPE per-channel NUQ, 1% dense-and-sparse
  outliers, exact sink token). Stated deviations are in its docstring. Online
  calibration can only favour the baselines.
- `sievelib/compress.py`: `apply_bits(..., keys_fn=None)`; None is the old path.
- `sievelib/baselines.py`: three labels added to `RESERVED`.
- `h0_measurement/run_r8.py`: the arms, their per-head errors (`eval_heads` on
  the arm's own keys), a `side_bits` column, and a `quant_baselines` sidecar
  record. The panel and policy-diagnostic paths refuse these arms.
- `tests/test_kv_quant_baselines.py` (new), including the RoPE inverse against
  Llama-3.2-1B's real cache (max error 1.9e-6, llama3 rope scaling).

No file in the R11-ext ledger (`bugs/11_nested_code_overhead/source_ledger_ext.json`)
was touched.

## Budget accounting (decided before the run)

Every arm spends B **code** bits per context key element (audited at run time).
Side information differs and is reported next to it, never hidden:

| arm | side info, bit/elem (d = 128) |
|---|---|
| TurboQuant (`uniform`) | 0.125 (fp16 norm per token) |
| KIVI g32 / g128 | 1.0 / 0.25 (fp16 scale + zero per channel-group) |
| KVQuant-style | ≈0.32 (1% outliers × 32 bits) |
| eviction arms | 0.125 × kept fraction (norm per kept 8-bit token) |
| SIEVE | 0.125 × kept fraction + width map (≤ 3 bits per token = 0.023) |

KIVI-g32 therefore has the most side information. The table reports KIVI-g128
as the closest to matched memory and KIVI-g32 as the paper default.

## Readouts, fixed now

1. **Main table:** mean task score per arm over the valid cells (FP ≥ 0.9 per
   task/cell; Qwen multivalue stays excluded, FP 0.24/0.04), and per model.
2. **Head-to-head counts:** SIEVE vs each baseline, per cell, win/tie/loss at a
   0.05 margin, with a paired prompt-bootstrap 90% CI for the mean difference.
3. **Non-ceiling gate (B):** the cell counts as non-ceiling only if FP ≥ 0.95
   and TurboQuant at B = 2 lies in [0.50, 0.90]. If TurboQuant returns ≥ 0.90,
   report the cell as still at ceiling. **Do not tune on these prompts.**
4. **Failure case:** Llama-3.1-8B @ 128K, all four tasks, reported with the
   route shares, evicted fraction, and head-error vs accuracy for every arm.

## Failure-case diagnostics (added before any job started)

Reading the R9 128K parquet (job 978485) showed the SIEVE router's wrong
answers are **not distractors** (distractor rate 0.025, lowest of all arms, vs
0.06–0.11 for the eviction baselines). They are corrupted numbers
("1111006", "801530m", "976, 976, 976, …"). The needle was found and its
value was damaged. Every baseline that beats SIEVE at 128K smooths its score
over neighbouring positions (SnapKV/Ada-KV/OBCache max-pool 7, LaProx
avg-pool 7). SIEVE's interior allocates per token on the unpooled window
attention.

Hypothesis H-pool: token-granular allocation fragments multi-token answers
(the key is attended, the value digits next to it are not, so they get few
bits or are evicted). Two diagnostic arms test it, and neither is a method:

- `interior_pool`: the same water-fill on SnapKV-pooled attention (exists in
  run_r8 since P2). **H-pool is supported** if `interior_pool` recovers at
  least half of the gap between `interior_cascade` and `obck_ada` at 128K,
  B = 2, averaged over the four tasks. It is **refuted** if it recovers under
  a fifth.
- `router_oracle`: routes on the measured per-head output error (P-5's upper
  bound). If the oracle also fails at 128K, the failure is the output-error
  proxy, not the offline calibration.
