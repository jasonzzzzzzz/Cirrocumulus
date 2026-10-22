# R7a — the theory at the paper's scale (reader: read_r7a.py)


## Predictions (stated before the run)

| prediction | verdict |
|---|---|
| P1_coverage | MET |
| P2_tightness | MET |
| P3_certified | MET |
| P5_certread | MET |
| P6_p8_over_v2 | MET |
| P4_theorem_m | MET |
| P7_seed_robust | MET |
| P8_share2_orders_kl | MET |

All eight of R7a's predictions are MET on the 13 finished blocks. I ran R7a's own frozen reader on them: every block passed its validity checks, the code manifests are identical, and the 119-file code pin still matches. Only l-hm (23020657) is missing. That means Llama 128K is read on RULER alone so far: 50 units, about 1.0 M head-steps.

| | prediction | measured (Llama 128K / Qwen 32K / Qwen 128K) |
|---|---|---|
| P1 | coverage OK | **0 misses** in about 35 M checks for each bound (part 8 and v2) |
| P2 | part 8 / error at 6 bits: [4, 7] Llama, [6, 13] Qwen | 5.65 / 7.22 / 8.02 |
| P3 | certified within FP8 at 6 bits ≥ 0.80 / 0.65 / 0.65 | 0.938 / 0.724 / 0.679 |
| P4 | Theorem M ratio consistent with 1 | 1.000 at 6 and 8 bits in every group; at 4 bits 1.000 / 0.994 / 0.990 |
| P5 | certified read traffic ≤ 0.45 / 0.50 / 0.50 | 0.399 / 0.474 / 0.423; no query head missed its target |
| P6 | part 8 / v2 ≤ 0.30 | 0.224 / 0.212 / 0.211 |
| P7 | the seed doesn't matter (SEED_ROBUST) | 5 of 5 families; the seed effect common to all units is ≤ 0.2% of the variance; all margins far below 5e-4 |
| P8 | unread share ranks Llama's families by KL | Spearman 0.8, but over 4 families only |

**What else the data show:**
- **The weak layers are the predicted ones.** Llama's three worst are L1, L8 and L30 (0.87–0.88 certified), exactly those predicted from R6. Qwen 32K's worst are L3 (0.000), L1, L0, L2 and L16, again the predicted set. At 128K Qwen's early layers do better (L1 0.03 → 0.41), as the CPU study expected, but L3 stays near 0.
- **The certified read costs almost nothing on Llama:** it reads 0.126 of rows against the oracle's 0.125. On Qwen the extra reading comes from those same early layers: at 32K, L3 is read in full, and 5% of groups fall back to an exact read.
- **v2 is 25.0× the error on Llama at 128K.** That matches R6's 25.4× and the CPU's flat-in-context finding. So the note to carry back to RESEARCH_PLAN §1.1 stands: v2's tightness doesn't change with length.
- **Most of part 8's slack is in the key term:** about 0.7 of the bound, with values about 0.25 and the rest under 0.03.
- **The design's predicted error is about 2% of FP8's** on Llama at 6 bits, matching R6's 2.4%.
- **The same family ranking holds on Qwen, outside P8:** Spearman 0.8 at 32K and 1.0 at 128K, where rerank has both the largest unread share and the largest KL.

**Caveats:**
- **Theorem M at 4 bits on Qwen:** the ratio is 0.990–0.994, and Qwen 32K's interval just misses 1. The quantity is heavy-tailed: the maximum is 2,255 against a mean of 1, and with only 2 draws per head-step a sample mean usually falls short. At 6 and 8 bits it is 1.000. I'd call 4 bits "consistent, heavy-tailed", not exact to the third digit.
- **Qwen 128K certifies less than Qwen 32K (0.68 vs 0.72), driven by rerank (0.62).** Its block is HELMET while Qwen 32K's are RULER, so the families differ. The plan says to compare with Llama's HELMET block, l-hm, before blaming context length.
- **The rerank seed panel is a re-run, not R6's own numbers.** 8 of 10 items match R6 exactly. On items 13 and 15, FP's own generated answer came out differently, the known run-to-run nondeterminism at 128K. Panel B compares all its arms within one process, so it's unaffected.
- **P8 is read against R6's 10-11 partial read,** which is final for Llama, because R6's final read doesn't exist yet. I don't see R6's last block, 22943967, in `results/` here.

**Left to do:** when l-hm runs, it adds ICL, RAG and rerank to Llama's results and raises P8 to 7 families. Reader job 23020667 then runs automatically on Rorqual. After that, copy back its `findings/R7a_reader.*`. Once R6's final read exists, rerun the reader with R6's final JSON, which is what P8 calls for.

I saved this interim read as `findings/R7a_reader_partial_1017.md` and `.json`, with the diagnostics script beside it. The names don't clash with the final reader's output.

## Panel A: Llama-3.1-8B, 128K (1,002,496 head-steps, 50 units)

| key bits | σ (median) | part 8 / error (p10–p90) | v2 / error | part 8 / v2 | certified within FP8 (p8 / v2) | coverage p8 (misses / checks) | coverage v2 | Theorem M exact (95% CI) | Theorem M first order | tail / FP8 (pred.) | certified read: rows (traffic), oracle, met per query head |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.343 | 167.15 (70.00–489.48) | 419.1 | 0.462 | 0.066 / 0.027 | COVER_OK (0 / 2,004,992) | COVER_OK (0) | 1.000 (0.998–1.002) | 0.999 | 0.057 | — |
| 6 | 0.082 | 5.65 (4.11–8.51) | 25.0 | 0.224 | 0.938 / 0.662 | COVER_OK (0 / 5,012,480) | COVER_OK (0) | 1.000 (1.000–1.001) | 1.000 | 0.022 | 0.126 (0.399), 0.125, 1.0000 (COVER_OK) |
| 8 | 0.020 | 2.06 (1.78–2.51) | 12.8 | 0.160 | 0.992 / 0.839 | COVER_OK (0 / 3,007,488) | COVER_OK (0) | 1.000 (1.000–1.000) | 1.000 | 0.018 | 0.125 (0.453), 0.125, 1.0000 (COVER_OK) |

| family | head-steps | unread share of Σw²‖v−o‖² (mean) | part 8 certified at 6 bits |
|---|---|---|---|
| aggregation | 316,416 | 0.0115 | 0.906 |
| multivalue | 261,120 | 0.0101 | 0.945 |
| retrieval | 114,688 | 0.0067 | 0.977 |
| tracking | 310,272 | 0.0096 | 0.950 |

## Panel A: Qwen3-30B-A3B, 128K (500,736 head-steps, 30 units)

| key bits | σ (median) | part 8 / error (p10–p90) | v2 / error | part 8 / v2 | certified within FP8 (p8 / v2) | coverage p8 (misses / checks) | coverage v2 | Theorem M exact (95% CI) | Theorem M first order | tail / FP8 (pred.) | certified read: rows (traffic), oracle, met per query head |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.403 | 454.32 (69.28–296971.25) | 1372.6 | 0.428 | 0.026 / 0.007 | COVER_OK (0 / 1,001,472) | COVER_OK (0) | 0.990 (0.980–1.004) | 0.983 | 0.125 | — |
| 6 | 0.096 | 8.02 (4.50–35.60) | 42.6 | 0.211 | 0.679 / 0.382 | COVER_OK (0 / 2,503,680) | COVER_OK (0) | 1.000 (0.999–1.001) | 1.000 | 0.039 | 0.161 (0.423), 0.125, 1.0000 (COVER_OK) |
| 8 | 0.024 | 2.40 (1.88–4.03) | 17.1 | 0.149 | 0.912 / 0.642 | COVER_OK (0 / 1,502,208) | COVER_OK (0) | 1.001 (1.000–1.001) | 1.001 | 0.028 | 0.126 (0.454), 0.125, 1.0000 (COVER_OK) |

| family | head-steps | unread share of Σw²‖v−o‖² (mean) | part 8 certified at 6 bits |
|---|---|---|---|
| icl | 62,976 | 0.0291 | 0.731 |
| rag | 142,848 | 0.0314 | 0.776 |
| rerank | 294,912 | 0.0722 | 0.621 |

## Panel A: Qwen3-30B-A3B, 32K (2,029,056 head-steps, 50 units)

| key bits | σ (median) | part 8 / error (p10–p90) | v2 / error | part 8 / v2 | certified within FP8 (p8 / v2) | coverage p8 (misses / checks) | coverage v2 | Theorem M exact (95% CI) | Theorem M first order | tail / FP8 (pred.) | certified read: rows (traffic), oracle, met per query head |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.379 | 240.87 (39.16–143791.18) | 724.0 | 0.411 | 0.118 / 0.047 | COVER_OK (0 / 4,058,112) | COVER_OK (0) | 0.994 (0.989–0.999) | 0.987 | 0.088 | — |
| 6 | 0.090 | 7.22 (3.93–33.57) | 37.1 | 0.212 | 0.724 / 0.483 | COVER_OK (0 / 10,145,280) | COVER_OK (0) | 1.000 (1.000–1.001) | 1.000 | 0.029 | 0.235 (0.474), 0.126, 1.0000 (COVER_OK) |
| 8 | 0.022 | 2.34 (1.81–4.07) | 16.6 | 0.149 | 0.934 / 0.699 | COVER_OK (0 / 6,087,168) | COVER_OK (0) | 1.000 (1.000–1.001) | 1.000 | 0.020 | 0.130 (0.456), 0.125, 1.0000 (COVER_OK) |

| family | head-steps | unread share of Σw²‖v−o‖² (mean) | part 8 certified at 6 bits |
|---|---|---|---|
| aggregation | 451,584 | 0.0420 | 0.680 |
| multivalue | 491,520 | 0.0323 | 0.717 |
| retrieval | 649,728 | 0.0230 | 0.766 |
| tracking | 436,224 | 0.0347 | 0.714 |

## Panel B: Llama-3.1-8B, 128K, the design's KL across dither seeds

| family | units × seeds | mean y (seed 0) | common seed effect: share of variance | U (seed 0 = R6's) | U per seed | U (one seed per replicate) | U (two-level) | label |
|---|---|---|---|---|---|---|---|---|
| multivalue | 10 × 5 | -0.00751 | 0.000 | -0.00286 | -0.00286, -0.00287, -0.00287, -0.00281, -0.00291 | -0.00286 | -0.00287 | SEED_ROBUST |
| rerank | 10 × 5 | -0.00656 | 0.000 | -0.00422 | -0.00422, -0.00391, -0.00406, -0.00427, -0.00420 | -0.00419 | -0.00413 | SEED_ROBUST |
| retrieval | 10 × 5 | -0.00037 | 0.000 | -0.00019 | -0.00019, -0.00018, -0.00019, -0.00018, -0.00019 | -0.00018 | -0.00019 | SEED_ROBUST |

## Panel B: Qwen3-30B-A3B, 32K, the design's KL across dither seeds

| family | units × seeds | mean y (seed 0) | common seed effect: share of variance | U (seed 0 = R6's) | U per seed | U (one seed per replicate) | U (two-level) | label |
|---|---|---|---|---|---|---|---|---|
| multivalue | 10 × 5 | -0.00002 | 0.000 | 0.00002 | 0.00002, -0.00001, 0.00001, -0.00000, -0.00001 | 0.00001 | 0.00000 | SEED_ROBUST |
| retrieval | 10 × 5 | -0.00001 | 0.002 | -0.00000 | -0.00000, -0.00000, -0.00000, -0.00000, -0.00000 | -0.00000 | -0.00000 | SEED_ROBUST |


