# Part 7: sure sums, the mirror split, the certified adaptive read (CPU, 2026-10-10)

> **Correction (2026-10-12): every Gemma-3-12B number in this file before that date came from a randomly initialized model.**
> `adapters_s1h5._load` built `Gemma3ForCausalLM` from the multimodal checkpoint. transformers 5.16 does not map that checkpoint's `language_model.*` keys onto it, so every text weight was freshly initialized, and that branch skipped the missing-weight check. Random weights give flat attention.
> - Gemma 3's "flat, value-leaning" class (family E), its 6/6 choice, L5 as the flat-attention example (66% of the second moment unread) and R8's Gemma 3 predictions all describe the random model.
> - With the checkpoint's weights, L5 at 1,600 tokens has 7.6% of the second moment unread (between, not flat; unread mass 0.20, against 0.77) and a key share of 0.86 at 6/6 (key-leaning, against 0.37). Uniform 6/4 meets FP8's error at the median head-step (0.37×), not at p90 (1.36×).
> - The theorems' checks on those views (coverage, Theorem M's accuracy) remain valid statements about those inputs, but they are not evidence about Gemma 3.
> - Fixed: the loader uses the multimodal class and refuses any missing weight; `g0map_s1h5.provenance` compares loaded weights with the checkpoint's. No other model was affected: the 18 GQA/MHA models and the two MLA models load with no missing weight. G0 (`findings/findings_g0.md`) recomputes Gemma 3 at 1,600 tokens and at 32K.

This was developed in parallel with part 6 in `14h_methodology_improve/certread_s1h5.py` (its "R5 theory, part 6";
that write-up is `archive/R5_theory6.md`) and merged here.
- Code: `../theory4_s1h5.py` (`compare`, `certread`) and `../theory3_s1h5.tail_bound_rand_v3(..., sure=True)`.
- Tests: `../test_theory4_s1h5.py`.
- Theory: `../theory.md` §6c–§6e.
- Data: `findings_part7.json`, next to this file.

## Validity

Brute force (random, spiky, 30×-outlier and adversarial data; 2–8-bit keys; δ from 0.5 to 10⁻³): the sure-sum bound,
the split and v3 + sure sums have 0 failures. v3 + sure sums is never above v3. The certified read meets its target in
every case with every bound, and stops before the exact read. On real weights, coverage is 1.000 everywhere below.

## Every certificate on the same draws

Setup: the vote's 1/8 read exactly, δ = 10⁻³ per head-step, 3 draws, unbiased tail. Each cell is the median bound /
error · the share of head-steps certified within FP8's error.

| view | key bits | v2 | v3 (part 6) | v3 + sure sums | v2 + sure sums | split |
|---|---|---|---|---|---|---|
| Llama-1B L1 | 4 | 1,054 · 0 | 637 · 0 | **484 · 0** | 636 · 0 | 516 · 0 |
| | 6 | 41.7 · 0.04 | 33.0 · 0.06 | 21.6 · 0.10 | 23.9 · 0.09 | **18.3 · 0.13** |
| | 8 | 19.3 · 0.24 | 10.4 · 0.43 | 6.6 · 0.61 | 9.9 · 0.45 | **5.7 · 0.66** |
| Llama-1B L3 | 4 | 1,080 · 0 | 624 · 0 | **505 · 0** | 678 · 0 | 547 · 0 |
| | 6 | 38.8 · 0.04 | 30.4 · 0.08 | 21.1 · 0.15 | 23.4 · 0.12 | **17.9 · 0.20** |
| | 8 | 18.2 · 0.32 | 9.8 · 0.56 | 6.5 · 0.69 | 9.7 · 0.57 | **5.6 · 0.75** |
| Gemma 3 L5 (global) | 4 | 24.2 · 0 | 18.0 · 0 | 17.6 · 0 | 20.6 · 0 | **14.9 · 0** |
| | 6 | 10.5 · 0 | 5.2 · 0 | 4.6 · 0 | 7.7 · 0 | **4.0 · 0** |
| | 8 | 9.0 · 0 | 3.7 · 0 | 3.2 · 0 | 6.1 · 0 | **2.8 · 0** |

(Part 6's `findings_4bit.md` reports bound / *oracle* at δ = 0.05, so its numbers are not on this scale.)

- **The split is tightest at 6–8 bits**, about 1.8× tighter than v3 at 8 bits. Its key sum has observed vectors, and
  its sure sums cost only e^{2t} ≈ 1.1–1.3.
- **At 4 bits v3 + sure sums is tightest.** There e^{2t} is large, and part 6's observed weights on the value sum win.
- **Gemma 3 certifies nothing at FP8's level at any width.** Its 4-bit values (flat attention: 77% of the mass unread)
  leave the error itself above FP8's; part 6's Theorem M explains it.

## The certified adaptive read

Setup:
- schedule: the vote, then 0, 1/256, …, 1/2 more rows by the sketch, then everything;
- δ = 10⁻³ per head-step, 2 draws;
- "rows" = the share read exactly, and traffic = (1 − f)(k + 4)/32 + f;
- oracle stop = the first set whose true error meets the target (a lower limit).

| view | key bits | target | v2: rows (traffic) | v3 + sure sums | split | oracle stop |
|---|---|---|---|---|---|---|
| Llama-1B L1 | 8 | FP8's error | 0.355 (0.597) | 0.178 (0.486) | **0.167 (0.479)** | 0.125 |
| Llama-1B L2 | 8 | FP8's error | 0.268 (0.542) | 0.148 (0.468) | **0.143 (0.464)** | 0.125 |
| Llama-1B L3 | 8 | FP8's error | 0.330 (0.581) | 0.194 (0.496) | **0.180 (0.488)** | 0.125 |
| Llama-1B L0 | 8 | FP8's error | 0.815 (0.884) | 0.491 (0.682) | 0.449 (0.656) | 0.134 |
| Llama-1B L1–L3 | 6 | FP8's error | 0.50–0.60 | 0.38–0.49 | 0.35–0.45 (0.55–0.62) | 0.125 |
| Llama-1B L1–L3 | 8 | 1% of ‖õ‖ | 0.51–0.57 | 0.29–0.33 | **0.26–0.29 (0.54–0.56)** | 0.126–0.129 |
| Gemma 3 L5 | 8 | 1% of ‖õ‖ | 0.962 | 0.202 (0.501) | **0.160 (0.475)** | 0.125 |
| Gemma 3 L5 | 6, 8 | FP8's error | 1.0 | 1.0 | 1.0 | 0.64: the values cannot |
| Granite-4.0-H L5 | 8 | 1% of ‖õ‖ | 1.0 | — | 0.994 | 0.881: the values cannot |

Readings:
- **At 8-bit keys a per-head-step guarantee costs no more than FP8's traffic**, on Llama-1B L1–L3 (FP8's own error as
  the target) and on Gemma 3's global layer (a 1% run-time target). The rule reads 14–18% of rows exactly; traffic is
  0.46–0.49 against FP8's 0.50.
- **FP8's per-head error is an oracle target.** A deployment needs a run-time τ, e.g. a relative one, or one calibrated
  offline from FP8's per-head error distribution.
- **Where 4-bit values cannot reach the target, no read set can.** That is Gemma 3 at FP8's error and Granite at 1%.
  More value bits are the remedy.

## Open

- The key part at 6 bits (16–18× the error) and at 4 bits. A sub-Gaussian proxy for e^{−δ} − E e^{−δ} would give K a
  Hsu–Kakade–Zhang bound like V's.
- A GPU check at 128K. R6's probe is frozen and reports v1 and v2 only, so the split and the certified read need a probe
  column set after R6.
