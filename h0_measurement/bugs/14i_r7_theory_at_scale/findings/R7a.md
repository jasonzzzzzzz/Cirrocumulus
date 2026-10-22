# R7a: the theory at the paper's scale — the read (2026-10-22)

R7a measured three of the theory section's results on R6's models and lengths, where they existed only on CPU proxies
(1–2B models, ≤ 16K tokens): part 8's certificate, the certified adaptive read, and Theorem M. It also tested whether
R6's verdict depends on the dither seed (the claims ledger's C1 gap). It is a verification run: it tests no design
claim, and its predictions were stated before the run (`../theory_r7.md` §4, `../plan.md` §5).

**Verdict: all eight predictions are MET, on all 14 blocks.** The guarantees held with no miss in about 38 M checks per
bound, Theorem M is exact at scale, the certified read reaches the oracle stop on Llama, and the dither seed has no
common effect on R6's statistic.

## 1. The run and the read

The pilots and the gate ran on Trillium on 2026-10-12 (1080289, 1080290 → gate 1080291 PASS, `R7a_gate.json`). The
Trillium blocks were then cancelled while still pending (09:17–09:23, from the user's account) and the run was
resubmitted on Rorqual; the reason is not recorded. On Rorqual: pilots 23020651/52, blocks 23020654–66, reader 23020667.

| block | group | tasks | units | job (Rorqual) | driver time | peak per GPU |
|---|---|---|---|---|---|---|
| l-r1 | Llama-3.1-8B, 128K | niah_single, niah_multikey, niah_multivalue, vt | 20 | 23020654 | 1.55 h | 56.0 GiB |
| l-r3a | Llama 128K | multikey, multivalue, vt, mk_panel (R6's R3a level) | 20 | 23020655 | 1.67 h | 56.3 GiB |
| l-agg | Llama 128K | cwe, fwe (R6's R3b level) | 10 | 23020656 | 1.37 h | 56.0 GiB |
| l-hm | Llama 128K | kilt_nq, kilt_hotpotqa, msmarco_rerank_psg, icl_trec_coarse, icl_banking77 (items 16–20) | 25 | 23020657 | 1.16 h | 58.8 GiB |
| q-r1 | Qwen3-30B-A3B, 32K | as l-r1 | 20 | 23020658 | 1.65 h | 66.2 GiB |
| q-r3a | Qwen 32K | as l-r3a (Qwen's level) | 20 | 23020659 | 1.72 h | 66.7 GiB |
| q-agg | Qwen 32K | cwe, fwe | 10 | 23020660 | 0.94 h | 66.2 GiB |
| w-hm | Qwen3-30B-A3B, 128K | as l-hm (items 16–21), two 2-GPU blocks | 30 | 23020661_0, _1 | 1.30, 1.10 h | 49.8 GiB |
| lk-rr | Llama 128K, seed panel | msmarco_rerank_psg (R6's items 11–20) | 10 | 23020662 | 0.42 h | 44.1 GiB |
| lk-mv | Llama 128K, seed panel | niah_multivalue (R6's 9500–9509) | 10 | 23020663 | 0.30 h | 50.8 GiB |
| lk-ret | Llama 128K, seed panel | niah_single, niah_multikey (R6's 9500–9504) | 10 | 23020664 | 0.21 h | 48.4 GiB |
| qk-mv | Qwen 32K, seed panel | niah_multivalue (R6's 9600–9609) | 10 | 23020665 | 0.12 h | 63.8 GiB |
| qk-ret | Qwen 32K, seed panel | niah_single, niah_multikey (R6's 9600–9604) | 10 | 23020666 | 0.08 h | 63.5 GiB |

Driver time is the sidecar's `elapsed_s`. Each job's preflight and model staging come on top of it. In all, the blocks
took about 16–17 GPU-h, with the Qwen 128K node job billed at 4 GPUs; the plan estimated 21–34 GPU-h.

**The read.** `../read_r7a.py` (rules frozen with the run) on all 14 blocks, run here on 2026-10-22 with
`--r6-json ../../14h_methodology_improve/findings/R6_reader_partial_1018.json`:
- every block passes the reader's validity checks; none is excluded;
- the blocks' code manifests are identical, and the 119-file code pin (`R7a_code.sha256`) still verifies;
- 3.78 M measured head-steps: Llama 128K 1,254,400 (75 units), Qwen 32K 2,029,056 (50 units), Qwen 128K 500,736 (30 units).

Outputs:
- `R7a_reader.md` and `.json`: the final read. It is byte-identical to `R7a_reader_1022.*`, from which it was rerun on
  2026-10-22 because Rorqual's reader job, 23020667, did not run.
- `R7a_diag_1022.py.txt`: the read-only diagnostics behind §5.
- `R7a_reader_partial_1017.*` and `R7a_diag_partial_1017.py.txt`: the interim read of 13 blocks, kept. Its Qwen numbers
  are unchanged here, since the Qwen blocks are the same.

The code that produced the numbers is in `R7a_code_snapshot_1022.tar.gz`: the 119 pinned files and the pin, which it
verifies against.

## 2. The predictions

| | prediction (stated before the run) | measured: Llama 128K / Qwen 32K / Qwen 128K | verdict |
|---|---|---|---|
| P1 | coverage COVER_OK for part 8 and v2 at every width and group | **0 misses** in 37.8 M checks per bound (part 8 and v2) | MET |
| P2 | part 8's median bound / error at 6 bits in [4, 7] (Llama), [6, 13] (Qwen) | 5.59 / 7.22 / 8.02 | MET |
| P3 | part 8 certified within FP8 at 6 bits ≥ 0.80 / 0.65 / 0.65 (from R6: Llama 0.88–0.91, Qwen 32K 0.71–0.76) | 0.919 / 0.724 / 0.679 | MET |
| P4 | Theorem M: R_exact's 95% interval meets [0.98, 1.02] at every width; R_first in [0.95, 1.05] at 6 and 8 bits | R_exact 1.000 at 6 and 8 bits in every group; at 4 bits 1.000 / 0.994 / 0.990 | MET |
| P5 | certified read at 6 bits: traffic ≤ 0.45 / 0.50 / 0.50; per-query-head misses COVER_OK | traffic 0.399 / 0.474 / 0.423; 0 misses in 774,656 query-head checks | MET |
| P6 | part 8 / v2 ≤ 0.30 (median, 6 bits) | 0.217 / 0.212 / 0.211 | MET |
| P7 | SEED_ROBUST in every family of panel (b) | 5 of 5 families; the common seed effect is ≤ 0.2% of the variance | MET |
| P8 | the unread share orders Llama 128K's families by the design's KL (Spearman > 0) | **Spearman 0.857 over all 7 families** (exact permutation p = 0.012, one-sided) | MET (closed on R6's 2026-10-18 read, §7) |

## 3. Panel (a): the bounds, per group and key width

| group | key bits | σ (median) | part 8 / error (p10–p90) | v2 / error | part 8 / v2 | certified within FP8 (p8 / v2) | misses / checks | Theorem M exact (95% CI) | first order | tail / FP8 (pred.) | part 8's K / V / D shares |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 128K | 4 | 0.343 | 167 (70–489) | 428 | 0.448 | 0.059 / 0.024 | 0 / 2,508,800 | 1.000 (0.998–1.002) | 0.999 | 0.062 | 0.97 / 0.02 / 0.01 |
| | 6 | 0.082 | 5.59 (4.11–8.41) | 25.5 | 0.217 | 0.919 / 0.633 | 0 / 6,272,000 | 1.000 (1.000–1.001) | 1.000 | 0.024 | 0.68 / 0.30 / 0.01 |
| | 8 | 0.020 | 2.05 (1.78–2.50) | 13.0 | 0.157 | 0.986 / 0.811 | 0 / 3,763,200 | 1.000 (0.999–1.000) | 1.000 | 0.019 | 0.30 / 0.70 / 0.00 |
| Qwen 32K | 4 | 0.379 | 241 (39–143,791) | 724 | 0.411 | 0.118 / 0.047 | 0 / 4,058,112 | 0.994 (0.989–0.999) | 0.987 | 0.088 | 0.96 / 0.01 / 0.02 |
| | 6 | 0.090 | 7.22 (3.93–33.6) | 37.1 | 0.212 | 0.724 / 0.483 | 0 / 10,145,280 | 1.000 (1.000–1.001) | 1.000 | 0.029 | 0.73 / 0.23 / 0.02 |
| | 8 | 0.022 | 2.34 (1.81–4.07) | 16.6 | 0.149 | 0.934 / 0.699 | 0 / 6,087,168 | 1.000 (1.000–1.001) | 1.000 | 0.020 | 0.37 / 0.61 / 0.01 |
| Qwen 128K | 4 | 0.403 | 454 (69–296,971) | 1,373 | 0.428 | 0.026 / 0.007 | 0 / 1,001,472 | 0.990 (0.980–1.004) | 0.983 | 0.125 | 0.96 / 0.01 / 0.02 |
| | 6 | 0.096 | 8.02 (4.50–35.6) | 42.6 | 0.211 | 0.679 / 0.382 | 0 / 2,503,680 | 1.000 (0.999–1.001) | 1.000 | 0.039 | 0.73 / 0.22 / 0.03 |
| | 8 | 0.024 | 2.40 (1.88–4.03) | 17.1 | 0.149 | 0.912 / 0.642 | 0 / 1,502,208 | 1.001 (1.000–1.001) | 1.001 | 0.028 | 0.37 / 0.61 / 0.01 |

"misses / checks" is part 8's; v2 also has 0 misses everywhere. "tail / FP8 (pred.)" is the median of √m_pred / e₈, the
tail's predicted RMS error over FP8's per-head error. The K / V / D shares are part 8's key, value and remainder terms
(medians, so they need not sum to 1).

**The certified read** (FINE schedule, 16 coin-free sets at δ/16; FP8's per-head error as the target; first chunk of
steps, draw 0):

| group | key bits | rows read (rule) | rows (oracle stop) | traffic (rule / oracle) | exact-read fallback | query-head checks | misses |
|---|---|---|---|---|---|---|---|
| Llama 128K | 6 | 0.126 | 0.125 | 0.399 / 0.399 | 0 | 296,960 | 0 |
| | 8 | 0.125 | 0.125 | 0.453 / 0.453 | 0 | 296,960 | 0 |
| Qwen 32K | 6 | 0.235 | 0.126 | 0.474 / 0.399 | 0.051 | 307,200 | 0 |
| | 8 | 0.130 | 0.125 | 0.456 / 0.453 | 0.001 | 307,200 | 0 |
| Qwen 128K | 6 | 0.161 | 0.125 | 0.423 / 0.399 | 0.010 | 170,496 | 0 |
| | 8 | 0.126 | 0.125 | 0.454 / 0.453 | 0 | 170,496 | 0 |

**By family** (6-bit keys; the R6 column is the design's mean KL per token in R6's read, P8's other variable):

| group | family | head-steps | unread share of Σw²‖v−o‖² (mean) | part 8 certified within FP8 | R6 design KL / token |
|---|---|---|---|---|---|
| Llama 128K | retrieval | 114,688 | 0.0067 | 0.977 | 9.0e-5 |
| | tracking | 310,272 | 0.0096 | 0.950 | 3.2e-4 |
| | multivalue | 261,120 | 0.0101 | 0.945 | 2.7e-4 |
| | icl | 30,720 | 0.0105 | 0.882 | 6.8e-4 |
| | aggregation | 316,416 | 0.0115 | 0.906 | 3.3e-4 |
| | rag | 78,848 | 0.0175 | 0.872 | 6.5e-4 |
| | rerank | 142,336 | 0.0296 | 0.824 | 2.1e-3 |
| Qwen 32K | retrieval | 649,728 | 0.0230 | 0.766 | 6.1e-6 |
| | multivalue | 491,520 | 0.0323 | 0.717 | 5.7e-4 |
| | tracking | 436,224 | 0.0347 | 0.714 | 4.2e-4 |
| | aggregation | 451,584 | 0.0420 | 0.680 | 1.3e-3 |
| Qwen 128K | icl | 62,976 | 0.0291 | 0.731 | 2.0e-4 |
| | rag | 142,848 | 0.0314 | 0.776 | 1.9e-3 |
| | rerank | 294,912 | 0.0722 | 0.621 | 1.9e-3 |

## 4. Panel (b): the design's KL across dither seeds

y = KL_tok(design at seed s) − 1.5 · KL_tok(FP8), on R6's own units, all arms in one process; seed 0 is R6's (7000). U is
R6's statistic (one-sided 95% unit-bootstrap upper bound of the mean y); R6's margin is 5·10⁻⁴.

| group | family | units × seeds | mean y (seed 0) | common seed effect: share of variance | U at seed 0 | U per seed (0–4) | U, one seed per replicate | label |
|---|---|---|---|---|---|---|---|---|
| Llama 128K | multivalue | 10 × 5 | −0.00751 | 0.000 | −0.00286 | −0.00286, −0.00287, −0.00287, −0.00281, −0.00291 | −0.00286 | SEED_ROBUST |
| | rerank | 10 × 5 | −0.00656 | 0.000 | −0.00422 | −0.00422, −0.00391, −0.00406, −0.00427, −0.00420 | −0.00419 | SEED_ROBUST |
| | retrieval | 10 × 5 | −0.00037 | 0.000 | −0.00019 | −0.00019, −0.00018, −0.00019, −0.00018, −0.00019 | −0.00018 | SEED_ROBUST |
| Qwen 32K | multivalue | 10 × 5 | −0.00002 | 0.000 | +0.00002 | +0.00002, −0.00001, +0.00001, −0.00000, −0.00001 | +0.00001 | SEED_ROBUST |
| | retrieval | 10 × 5 | −0.00001 | 0.002 | −0.00000 | all −0.00000 | −0.00000 | SEED_ROBUST |

## 5. Findings

1. **The guarantees hold at the paper's scale.** Part 8 and v2 missed in none of 37.8 M checks each, at 4, 6 and 8-bit
   keys, in all three groups. The certified read met FP8's per-head error at all 1.55 M query-head checks (6 and 8 bits).
   Every query was FP's (the oblivious pass); the design's own queries are covered by Lemma C, not measured here.

2. **Part 8's tightness is what the CPU study predicted, and v2 does not change with length.** At 6-bit keys part 8 is
   5.6× the error on Llama and 7.2–8.0× on Qwen, against 25.5× and 37–43× for v2 (part 8 / v2 = 0.21–0.22). On Llama-3.1-8B,
   v2 is 27× at 1.6K and 24.6× at 16K on the CPU and 25.5× at 128K here: flat in context.

3. **Which term of part 8 dominates depends on the key width.** The key term is about 0.96 of the bound at 4 bits and
   0.68–0.73 at 6 bits; at 8 bits the value term is 0.61–0.70. Going from 6 to 8 key bits still cuts the bound from
   5.6× to 2.1× the error on Llama, but past 8 bits more key bits would buy little: the value term is then the larger part.

4. **The layers that are hard to certify are the ones predicted from R6's probe.**

   | group | weakest layers, measured (share certified within FP8, 6 bits) | predicted from R6 (f = 0.22) |
   |---|---|---|
   | Llama 128K | L30 0.829, L8 0.866, L1 0.871, L25 0.891, L27 0.893 | L30 0.79, L1 0.83, L8 0.84 |
   | Qwen 32K | L3 0.000, L1 0.033, L0 0.195, L2 0.340, L16 0.383 | L3 0.003, L1 0.06, L2 0.32, L0 0.36, L16 0.39 |
   | Qwen 128K | L3 0.012, L1 0.405, L2 0.422, L0 0.527, L43 0.541 | (R6 did not probe Qwen 128K) |

   On Llama's HELMET block alone, L30 falls to 0.62. Qwen's early layers do better at 128K than at 32K (L1 0.03 → 0.41,
   L0 0.20 → 0.53), but L3 stays near 0. These are the flat-attention layers; the remedy there is more value bits, not a
   tighter bound.

5. **On Llama the certified read stops at the oracle.** It reads 0.126 of rows against the oracle's 0.125 (traffic
   0.3993 against 0.3985), between 0.1256 and 0.1273 in every family, with no exact-read fallback. On Qwen the extra rows
   come from the same early layers: at 32K, L3 is read in full (rows 1.000), L0 0.68 and L1 0.57, and 5.1% of groups fall
   back to an exact read; at 128K, L3 reads 0.71 and 1.0% of groups fall back. At 8-bit keys every group reads within
   0.005 of the oracle.

6. **Theorem M is exact at scale.** R_exact is 1.000 at 6 and 8 bits in every group, with 95% intervals within ±0.001.
   The first-order reading (Z̃ ≈ Z) is also 1.000 there. At 4 bits Llama is 1.000 too (mean 1.0001; largest single ratio
   43.6). On Qwen at 4 bits it is 0.994 (32K) and 0.990 (128K), and Qwen 32K's interval (0.989–0.999) just misses 1. The
   quantity is heavy-tailed there: nn / m_pred has median 0.87–0.88, maximum 1,297 (32K) and 2,255 (128K) against a mean
   of 1, and its top 0.1% carries 2.5% of the sum. Qwen 32K's L3 alone averages 0.71. With 2 draws per head-step, the
   sample mean of such a variable usually falls short of its expectation. On Qwen at 6 and 8 bits the largest ratios
   are 10–14 and 3–4, and the deficit is gone.

7. **The unread second moment orders the families by the design's KL.** Across Llama 128K's seven families the unread
   share of Σw²‖v − o‖² and R6's design KL have Spearman 0.857 (exact permutation p = 0.012); retrieval is lowest and
   rerank highest on both. Outside P8, the same holds on Qwen: Spearman 0.8 at 32K (4 families) and 1.0 at 128K
   (3 families), where rerank again has both the largest unread share (0.072) and the largest KL. This is Theorem M's
   reading at the family level: the tail's error is the unread share times a dense tier's per-row error.

8. **Qwen 128K certifies less than Qwen 32K because of its tasks, not its length.** R7a probed Qwen 128K only on HELMET
   and Qwen 32K only on RULER. HELMET is harder on Llama too:

   | family | Llama 128K certified | Qwen 128K certified | Llama unread share | Qwen 128K unread share |
   |---|---|---|---|---|
   | icl | 0.882 | 0.731 | 0.0105 | 0.0291 |
   | rag | 0.872 | 0.776 | 0.0175 | 0.0314 |
   | rerank | 0.824 | 0.621 | 0.0296 | 0.0722 |

   On the same items (16–20) Llama certifies 0.846 and Qwen 0.684. Llama's RULER families certify 0.906–0.977, about
   0.1 above its HELMET ones. Within Qwen, R7a cannot separate length from task; a Qwen run with the same families at
   both lengths would.

9. **The dither seed has no common effect.** In all five families the seed effect shared by every unit is at most 0.2%
   of the variance, and every seed's U is far below the 5·10⁻⁴ margin (Llama: −0.0039 to −0.0043 on rerank, the weakest
   family; Qwen 32K multivalue: at most +2·10⁻⁵). R6's verdict does not depend on its seed.

10. **R6's units reproduce, except where FP's own answer changed.** At seed 0 the panel re-runs R6's arms. On the RULER
    units the design's and FP8's KL match R6's exactly (correlation 1.00, the same means). On rerank, 8 of 10 items
    match exactly; on items 13 and 15 FP's own greedy answer came out differently (FP's answer NLL 53.9 vs 39.5 and
    11.49 vs 11.08), so the KL is measured on a different span (design 3.5·10⁻³ vs 2.3·10⁻³ and 1.3·10⁻³ vs 9.7·10⁻³).
    This is the known run-to-run nondeterminism at 128K. Panel (b) compares its arms within one process, so it is not
    affected.

## 6. What the theory section can now state (carried to `../../14h_theory/` on 2026-10-22: `theory.md`, `CLAIMS_LEDGER.md`, `paper_theory_section.md`)

- **T3 (part 8).** At 6-bit keys, part 8 is 5.6× the error on Llama-3.1-8B at 128K and 7.2–8.0× on Qwen3-30B-A3B at 32K
  and 128K (median; 2.0–2.4× at 8 bits). It certifies FP8-level per-head error on 0.92, 0.72 and 0.68 of head-steps
  (0.99, 0.93, 0.91 at 8 bits), with no miss in 37.8 M checks. Part 8 is 0.21–0.22 of v2's bound.
- **T4 (Theorem M).** Exact at the paper's scale: the measured / predicted ratio is 1.000 ± 0.001 at 6 and 8 bits on
  3.78 M head-steps; at 4 bits it is within 1% with a heavy tail. The unread share of the second moment orders the
  families by the design's KL (Spearman 0.857 on Llama's seven families).
- **T6 (the certified adaptive read).** With FP8's per-head error as the target, the certified read at 6-bit keys reads
  0.126 of rows on Llama (traffic 0.399, the oracle's 0.399), 0.235 on Qwen 32K (0.474) and 0.161 on Qwen 128K (0.423),
  with no query head missing its target in 774,656 checks. The guarantee is per query head. The target is an oracle;
  the run-time target τ is still open (`RESEARCH_PLAN.md` §3).
- **C1 (the seed).** R6's primary statistic does not depend on the dither seed: across five seeds the common seed effect
  is ≤ 0.2% of the variance in every family read, and every seed passes R6's margin with room (§4).

## 7. Limits

- **The target is an oracle.** The certified read is measured against FP8's per-head error, as on the CPU.
- **FP's queries.** Coverage is measured on the oblivious pass; the design's own queries rely on Lemma C.
- **The certified read covers the first chunk of 4 steps per layer, draw 0.** Its met share is still a per-query-head
  guarantee.
- **Theorem M at 4 bits** rests on 2 draws per head-step of a heavy-tailed quantity (§5.6).
- **Length and task are confounded within Qwen** (§5.8).
- **P8 is closed on R6's partial read of 2026-10-18,** by the user's decision of 2026-10-22, not on R6's final read as
  the plan said.
  - Its Llama 128K rows, the only ones P8 uses, are final: all of R6's Llama blocks had run, and the rows are identical
    to the 2026-10-11 read.
  - R6's final read waits only on a Qwen 128K block.
- **Panel (b) covers five families** (Llama: multivalue, rerank, retrieval; Qwen 32K: multivalue, retrieval), not Qwen
  128K.

## 8. Corrections carried back to `../../14h_theory/` (2026-10-22)

- `theory.md` §6e: the certified read holds "at each query head" (P ≥ 1 − δ), not "at every query head" jointly, which
  needs δ/(m·r) per bound (`../theory_r7.md` §2.6). R7a confirms the per-head form.
- `RESEARCH_PLAN.md` §1.1: "v2 39–42× on the CPU against 25× on the GPU" compared Llama-3.2-1B with Llama-3.1-8B. On
  Llama-3.1-8B v2 is 27× at 1.6K, 24.6× at 16K and 25.5× at 128K: scale does not hurt, and length does not help.
- Part 8 at 128K is 5.6× the error on Llama at 6 bits, not the plan's 4–5×.
- The fine schedule is 16 sets at δ/16, not 15 at δ/15. (No "15 sets" text remained in `../../14h_theory/` by 2026-10-22;
  nothing to change.)

Where each went: `theory.md` §6e (the per-head statement, with a correction note), §7 and §10 (the at-scale evidence),
the summary table; `RESEARCH_PLAN.md` §1.1 (already corrected 2026-10-12; the measured 5.6× added), the ranking and
§4's R7a outcome; `CLAIMS_LEDGER.md` (T2, T3, T4, T6, C1's seed gap, §5's checks); `paper_theory_section.md` (§4.4–§4.7
evidence slots); `STATUS.md`; `ARCHITECTURES_PLAN.md` (a pointer); `findings/findings_klbridge.md` (the 128K loose-layer
recheck and the stop-rule check); the paper's findings slides.

## 9. Closed

R7a has nothing open (2026-10-22):
- **The read.** Rorqual's reader job, 23020667, did not run; the final read is `R7a_reader.*`, run here with the frozen
  reader.
- **P8** is closed on R6's 2026-10-18 read (§7).
- **The freeze.** R7a's files are unfrozen except `cert_r7.py` and `tier_r7.py`, which stay frozen until R7b's final
  read. The code that produced these numbers is `R7a_code_snapshot_1022.tar.gz`.
- **Carried back.** The statements in §6 and the corrections in §8 went to `../../14h_theory/` and to the paper's
  slides.

To reproduce the read, extract the snapshot at the repository root and run:

```bash
cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14i_r7_theory_at_scale/read_r7a.py \
    --r7a $(bash h0_measurement/bugs/14i_r7_theory_at_scale/script_r7a.sh --list-r7a) \
    --out-stem <out> --r6-json h0_measurement/bugs/14h_methodology_improve/findings/R6_reader_partial_1018.json
```
