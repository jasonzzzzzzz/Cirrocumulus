# R7a's CPU exploration: the theory on R6's own models at growing context (2026-10-11)

`../explore_r7.py` (capture, analyze, summary, predict). Data:
- `explore/*.json`, one per (view, key width);
- `explore_r7.json` and `explore_r7_table.md`, the full table;
- `predictions_from_r6.json`.

Views are in `../.views/` (re-captured here, not copied). The 1.6K Llama-3.1-8B L1 capture equals
`../../14h_theory/.views/view_llama8b_L1_1600.pt` to fp32 rounding (2.4·10⁻⁷). (That view moved to
`../../14_logs/14h_theory/views/` on 2026-10-22. R7a's own `../.views/` is no longer on disk.)

## Setup

The CPU study's setup (`../../14h_theory/exp4bit_s1h5.load_setup`), re-implemented in `explore_r7.load_view`:
- the input is PG-19, from the first book's character 20,000 on;
- the last 48 positions are fixed rows; their first 32 queries vote over a Lloyd-Max 4-bit sketch of the context's keys
  (the top C/8 rows: a coin-free read set); the last 16 are measured;
- tier 1 is `tier_r7`'s dithered keys at 4/6/8 bits with 4-bit values, the unbiased tail;
- δ = 10⁻³ per head-step, 2 draws per width;
- the certified read uses draw 0 and the FINE schedule (16 sets at δ/16), with FP8's per-head error as the target.

Models are truncated, fp32 SDPA:
- Llama-3.1-8B-Instruct L1 and L3;
- Qwen3-30B-A3B-Instruct-2507 L1 (flat attention) and L12 (typical).

Contexts are 1.6K, 4K, 8K and 16K tokens. Every cell is 512 head-steps.

## The table

| model | layer | context | key bits | σ (median) | part 8 / error | v2 / error | certified within FP8 (p8 / v2) | Theorem M measured / predicted (median) | unread mass | unread share of Σw²‖v−o‖² | tail RMS / FP8 (pred.) | certified read: rows (traffic) | oracle stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama8b | 1 | 1600 | 4 | 0.341 | 100.6 | 346 | 0.00 / 0.00 | 1.011 | 0.026 | 0.0012 | 0.71 | — | — |
| llama8b | 1 | 1600 | 6 | 0.081 | 5.11 | 27.2 | 0.35 / 0.00 | 0.977 | 0.026 | 0.0012 | 0.26 | 0.459 (0.628) | 0.131 (0.402) |
| llama8b | 1 | 1600 | 8 | 0.020 | 2.04 | 14.8 | 0.84 / 0.07 | 0.983 | 0.026 | 0.0012 | 0.21 | 0.177 (0.486) | 0.128 (0.455) |
| llama8b | 1 | 4096 | 6 | 0.082 | 5.09 | 24.4 | 0.47 / 0.05 | 0.977 | 0.035 | 0.0011 | 0.21 | 0.349 (0.552) | 0.126 (0.399) |
| llama8b | 1 | 4096 | 8 | 0.020 | 2.00 | 13.4 | 0.93 / 0.15 | 0.993 | 0.035 | 0.0011 | 0.16 | 0.145 (0.466) | 0.125 (0.453) |
| llama8b | 1 | 8192 | 6 | 0.084 | 5.39 | 24.6 | 0.60 / 0.10 | 0.975 | 0.035 | 0.0006 | 0.16 | 0.230 (0.470) | 0.125 (0.399) |
| llama8b | 1 | 8192 | 8 | 0.021 | 2.03 | 13.0 | 0.94 / 0.29 | 1.006 | 0.035 | 0.0006 | 0.13 | 0.127 (0.454) | 0.125 (0.453) |
| llama8b | 1 | 16384 | 4 | 0.354 | 183.1 | 528 | 0.00 / 0.00 | 0.955 | 0.039 | 0.0003 | 0.39 | — | — |
| llama8b | 1 | 16384 | 6 | 0.084 | 5.45 | 24.6 | 0.66 / 0.12 | 0.999 | 0.039 | 0.0003 | 0.14 | 0.209 (0.456) | 0.125 (0.398) |
| llama8b | 1 | 16384 | 8 | 0.021 | 2.05 | 12.6 | 0.96 / 0.34 | 0.997 | 0.039 | 0.0003 | 0.11 | 0.132 (0.457) | 0.125 (0.453) |
| llama8b | 3 | 1600 | 6 | 0.071 | 4.19 | 24.3 | 0.62 / 0.07 | 0.966 | 0.030 | 0.0024 | 0.17 | 0.232 (0.472) | 0.125 (0.398) |
| llama8b | 3 | 16384 | 6 | 0.075 | 4.60 | 24.0 | 0.97 / 0.43 | 0.997 | 0.018 | 0.0001 | 0.05 | 0.128 (0.400) | 0.125 (0.398) |
| qwen30b | 1 | 1600 | 6 | 0.136 | 11.0 | 53.8 | 0.03 / 0.00 | 0.985 | 0.221 | 0.3309 | 1.26 | 0.994 (0.996) | 0.311 (0.526) |
| qwen30b | 1 | 1600 | 8 | 0.034 | 2.83 | 19.0 | 0.14 / 0.03 | 1.003 | 0.221 | 0.3309 | 0.84 | 0.494 (0.684) | 0.210 (0.506) |
| qwen30b | 1 | 8192 | 6 | 0.136 | 11.5 | 51.0 | 0.04 / 0.00 | 0.984 | 0.207 | 0.2310 | 0.84 | 0.955 (0.969) | 0.175 (0.433) |
| qwen30b | 1 | 16384 | 6 | 0.135 | 11.9 | 50.3 | 0.09 / 0.04 | 0.995 | 0.181 | 0.1289 | 0.63 | 0.748 (0.827) | 0.137 (0.407) |
| qwen30b | 1 | 16384 | 8 | 0.033 | 2.78 | 16.5 | 0.45 / 0.09 | 1.012 | 0.181 | 0.1289 | 0.42 | 0.193 (0.495) | 0.128 (0.455) |
| qwen30b | 12 | 1600 | 6 | 0.168 | 13.3 | 70.3 | 0.44 / 0.11 | 0.928 | 0.036 | 0.0029 | 0.10 | 0.549 (0.690) | 0.130 (0.402) |
| qwen30b | 12 | 1600 | 8 | 0.042 | 3.03 | 22.1 | 0.89 / 0.43 | 0.989 | 0.036 | 0.0029 | 0.06 | 0.148 (0.468) | 0.125 (0.453) |
| qwen30b | 12 | 8192 | 6 | 0.171 | 14.5 | 62.9 | 0.69 / 0.20 | 0.976 | 0.046 | 0.0013 | 0.05 | 0.261 (0.492) | 0.125 (0.399) |
| qwen30b | 12 | 16384 | 6 | 0.172 | 14.8 | 62.5 | 0.78 / 0.38 | 0.960 | 0.034 | 0.0005 | 0.02 | 0.244 (0.480) | 0.125 (0.398) |
| qwen30b | 12 | 16384 | 8 | 0.043 | 2.92 | 18.2 | 0.99 / 0.81 | 0.996 | 0.034 | 0.0005 | 0.01 | 0.126 (0.453) | 0.125 (0.453) |

Coverage is 1.000 for part 8 and for v2 in every cell. Theorem M's *mean* measured / predicted, which is unbiased where
the median of 2-draw means is not, is 0.976–1.014 at 6–8 bits and 1.01–1.14 at 4 bits.

## Readings

1. **Part 8's tightness does not depend on context.**
   - Llama L1 5.1 → 5.45× (6 bits) and 2.0× (8 bits); L3 4.2 → 4.6×; Qwen L12 13.3 → 14.8×; Qwen L1 11.0 → 11.9×.
   - The only C-dependence is the truncation level z = √(2 ln(4|T|/δ)) inside e^{zσ}: about +3% from 16K to 128K.
2. **v2 is flat on the same model.** It is 27× at 1.6K and 24.6× at 16K on Llama-3.1-8B, and R6 measured 25.4× at 128K.
   `../../14h_theory/RESEARCH_PLAN.md` §1.1 compared v2's 39–42× (Llama-3.2-1B, CPU) with 25× (Llama-3.1-8B, GPU). The
   difference is the model, not the length. Part 8 / v2 is 0.19–0.22 (Llama) and 0.24 (Qwen) at 6 bits, 0.14–0.16 at
   8 bits.
3. **Certification at FP8's level gets easier as context grows.**
   - The vote's 1/8 captures more of Σ w²‖v − o‖² at long context: the unread share is 0.0012 → 0.0003 on Llama L1 and
     0.0029 → 0.0005 on Qwen L12.
   - FP8's dense error does not shrink, so the tail's predicted error falls relative to FP8's: 0.26 → 0.14 (L1),
     0.17 → 0.05 (L3).
   - The certified share rises: Llama L1 0.35 → 0.66, L3 0.62 → 0.97, Qwen L12 0.44 → 0.78 (6 bits).
4. **The certified read approaches the oracle stop.**
   - At 6 bits: Llama L1 0.459 → 0.209 of rows (traffic 0.628 → 0.456); Llama L3 at 16K 0.128 (0.400, against the
     oracle's 0.398); Qwen L12 0.549 → 0.244 (0.690 → 0.480).
   - At 8 bits the rule reads what the oracle reads by 8K–16K.
5. **Flat layers are the exception.**
   - Qwen L1 leaves 13–33% of the second moment unread. At 6 bits the certified read needs 0.75–0.99 of rows, against
     the oracle's 0.14–0.31.
   - Its predicted tail error is above FP8's at 1.6K (1.26) and below it at 16K (0.63).
   - This is MODEL_FAMILY's flat-layer limit; more value bits, not a tighter bound, is the remedy.
6. **4-bit keys.** Part 8 is 100–180× the error on Llama L1 and nothing is certified within FP8 (T7). The tail's
   predicted error is still 0.39–0.71× FP8's (T4).

## The GPU forms on real layers

**Lemma S** bounds the matrix's norms from its H heaviest rows. Its inflation of part 8's bound, median (max), with the
light rows' share of tr M:

| view, key bits | H = 1,024 | H = 4,096 |
|---|---|---|
| Llama L1 4K, 6 | 1.022 (1.36) | — |
| Llama L1 8K, 6 | 1.052 (1.46), light 0.034 | 1.0008 (1.09), light 0.0005 |
| Llama L1 16K, 6 | 1.097 (1.52), light 0.065 | 1.008 (1.20), light 0.0052 |
| Llama L1 16K, 8 | 1.031 (1.20) | 1.0026 (1.08) |
| Qwen L12 16K, 6 | 1.082 (1.41) | 1.0045 (1.08) |

The inflation grows with context, so R7a uses the exact matrix (`probe_r7.GRAM_ROWS = CR_GRAM_ROWS = None`).

**Lemma B** decides the certified read's sets before forming the matrix. On real layers it skips 50–90% of them: the
matrix was formed for 274 of 591 group evaluations (Llama L1 16K, 6 bits) and 61 of 207 (8 bits).

## Predictions for R7a from R6's GPU probe (`predictions_from_r6.json`)

R6's six probe blocks hold 1.33 M Llama 128K head-steps and 1.58 M Qwen 32K head-steps (q-agga, job 1072665, was
pending on 2026-10-11). Each head-step's v2 bound at 6 bits, scaled by f = part 8 / v2:

| group | v2 certified (measured) | part 8 certified, f = 0.18 / 0.20 / 0.22 / 0.24 / 0.25 | certified read on R6's coarse sets: rows (traffic) | weakest layers (f = 0.22) |
|---|---|---|---|---|
| Llama 128K | 0.625 | 0.910 / 0.900 / 0.891 / 0.882 / 0.877 | 0.127 (0.400) at every f; stops at the vote's set in 70–73% of groups | L30 0.79, L1 0.83, L8 0.84 |
| Qwen 32K | 0.487 | 0.757 / 0.743 / 0.731 / 0.719 / 0.713 | 0.233–0.263 (0.473–0.494) | L3 0.003, L1 0.06, L2 0.32, L0 0.36, L16 0.39 |

At 128K the design's median per-head error is 2.4% of FP8's on Llama and 2.7% on Qwen 32K. Certification is limited by
the bound on the hard head-steps, not by the design's error.
