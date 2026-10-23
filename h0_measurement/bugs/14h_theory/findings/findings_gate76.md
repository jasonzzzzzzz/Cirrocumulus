# The dense 7/6 gate: why two tiers lose on re-ranking, and the next design (CPU, 2026-10-22/23)

*Code: `../gate76_s1h5.py` (units, theory, arms, reads), `../gate76sum_s1h5.py` (summary), `../test_gate76_s1h5.py`
(checks, all pass), `../submit_gate76_cpu.slurm`. Results: `gate76/<model>/<unit>/` (meta, theory, arms, layers,
reads), the G0 control in `gate76/g0/`, tables in `gate76/tables_gate76.md`, everything in `gate76/summary.json`.
All jobs are done (§7); `../../../.venv/bin/python gate76sum_s1h5.py` regenerates the tables.*

## 0. The question and the answer

R7b's gate (ii), "two tiers beat one", asks whether the single store (4+2/4 keys/values, the vote's 1/8 read exactly,
traffic 0.398) beats dense dithered 7/6 on every row (traffic 0.406). It is NOT_SHOWN in all three claim groups
(p 0.23 / 0.33 / 0.95). On Llama re-ranking, dense 7/6 has 2.6× lower KL. The task: find out why, and pick the next
design among G0's per-layer allocation, 6-bit values in flat layers, and a 1/4 read.

1. **It is not flat heads. The question's vote reads the wrong rows.**
   - On Llama re-ranking at 32K, the design's predicted excess over dense 7/6 splits by head-step as follows:

     | head-steps | share of the excess | share of the design's KL |
     |---|---|---|
     | vote miss: the per-step oracle 1/8 leaves ≈ 0 unread, the vote leaves 20–50% | **107%** | **99%** |
     | flat | **0%** | 0% |
   - The heads involved are copying the passage IDs that FP's ranking names.
   - The vote cannot know those passages: it is cast by a question of about 10 tokens before the answer exists.
   - The same tier with the read set chosen from the answer's own attention is 2,000× below dense 7/6 (§3).
2. **None of the three named variants beats dense 7/6 at ≤ 0.406, predicted or measured.** Each adds bits or rows to
   a read set that is wrong. Predicted at the budget, as the geometric mean of KL over dense 7/6's on 15 re-ranking
   units:

   | variant at ≤ 0.406 | KL / dense 7/6's |
   |---|---|
   | G0's allocation | 27.7× |
   | 6-bit values in the 4 best layers | 3.5× |
   | a 1/4 read in the 2 best layers | 4.0× |
   | the best water-filled allocation over residual bits, value bits and read fraction | 1.4× |

   Measured at uniform widths, over budget, they are 3–28× (§3.3).
3. **The next design: the single store with its vote re-cast at every answer step.**
   - The vote is R6's own row-buffer vote (s1d_lib.RowBuffer): SnapKV over the last 32 query rows, question then
     answer, on the single store's base keys, re-selected at every step instead of once.
   - Traffic stays at 0.398 (§4.3).
   - The read set at layer l depends only on base keys and layer l's earlier queries, so it is F_l-measurable
     (Lemma C′), and R1 and Theorems R and M hold at every step (§4.2).
   - Measured on CPU at 32K, KL over dense 7/6's (geometric mean, units below dense 7/6):

     | units | KL / dense 7/6's | units below |
     |---|---|---|
     | Llama re-ranking | 0.29× | 16 of 16 (8 of them held out, arms fixed before any 32K result) |
     | Llama RAG and ICL | 0.50–0.81× | 6 of 8 |
     | Qwen re-ranking, RAG and ICL | 0.07–0.85× | 6 of 7 |

   - It replaces 1.2–2.0% of its 1/8 per step.
   - The window-8 answer re-vote is stronger on re-ranking (0.08–0.13×) but gives back more of the design's lead on
     ICL. It is the alternate arm.
   - Re-selecting only every 8 steps (Stage 1d's qreadr) does not beat dense 7/6 (1.3×).
4. **Prediction for the GPU** (R7b's own bootstrap on its 289 Llama 128K units): with re-ranking's excess removed,
   gate (ii) passes at p = 0.0005 (mean d −2.0e-4); with only half of it removed, p = 0.010 (§4.5).

## 1. What R7b measured (GPU, `../../14i_r7_theory_at_scale/r7b/findings/R7b_reader_partial_1019/1022.*` and the per-unit parquet)

d = KL_design − KL_dense76 per unit (KL per answer token over FP's first line).

| Llama 128K family | units | design | dense 7/6 | fp_noise | FP8 | sum of d | units where the design is lower |
|---|---|---|---|---|---|---|---|
| aggregation | 21 | 2.9e-4 | 2.8e-4 | 3.4e-4 | 6.1e-4 | +0.0002 | 0.38 |
| icl | 60 | 2.4e-4 | 5.8e-4 | 4.3e-4 | 2.4e-3 | −0.0202 | 0.63 |
| multivalue | 30 | 2.4e-4 | 7.9e-4 | 5.7e-4 | 4.9e-3 | −0.0163 | 0.90 |
| rag | 60 | 6.5e-4 | 7.5e-4 | 6.6e-4 | 3.4e-3 | −0.0061 | 0.57 |
| **rerank** | 30 | **2.3e-3** | **9.0e-4** | 8.4e-4 | 5.3e-3 | **+0.0418** | 0.23 |
| retrieval | 58 | 5.4e-5 | 2.2e-4 | 1.4e-4 | 1.1e-3 | −0.0094 | 0.79 |
| tracking | 30 | 3.0e-4 | 4.6e-4 | 3.9e-4 | 2.4e-3 | −0.0049 | 0.90 |

- **Llama.** Re-ranking is the one family above the noise floor, and there dense 7/6 sits on it (9.0e-4 against
  fp_noise's 8.4e-4). Re-ranking's +0.042 cancels the design's −0.057 over the other six families (mean d −5e-5,
  p 0.23).
- **Qwen 32K** (mean d −1.6e-5, p 0.33) is at the floor everywhere. FP8 itself is at 1.2e-4 on multivalue against
  fp_noise's 2.1e-4.
- **Qwen 128K** (RAG and ICL so far, mean d +4.3e-4, p 0.95):
  - the design is lower on 53–62% of units, but five units carry 59% of the positive d;
  - those units have 2–3-token answers, and fp_noise on them is as large as the design's KL (up to 3.8e-2);
  - on them the single store is 3–5× R6's design, the per-unit dither lottery of `power_mv_r7b.md`.

  Its re-ranking blocks are still missing; §3.1 predicts them.

The threat to the design paper is one mechanism on one family, and a CPU run without a noise floor can isolate it.

## 2. The CPU study at 32K

- **Units.**
  - Re-ranking: R7b's own items (R6's msmarco items 11–18 for the analysis, 21–28 held out), rendered exactly as R6
    renders them (`tasks_s1h4.render_hm` with R7b's manifest). HELMET's end-of-context cut to 32,768 tokens keeps
    347–455 of the 1,000 passages.
  - Controls: kilt_nq 41–44, icl_banking77 41–42, icl_trec_coarse 41–42.
  - Models: Llama-3.1-8B (all units) and Qwen3-30B-A3B (re-ranking 11–13, kilt_nq 41–42, icl_banking77 41–42), fp32
    on CPU.
- **FP's answer:** greedy, `run_r8._decode`'s rule (EOS, or the first newline after content; 200 tokens at most). For
  Llama re-ranking it is 49–200 tokens of passage IDs ("8760867 > 4517034 > …"), the same answer on every re-run.
- **Arms** are teacher-forced from the question's last token, as R6's pass is, with KL over FP's first line, in one
  process (no noise floor).
  - Every layer's context rows are read the arm's way; question and answer rows stay exact. The GPU arms recompute
    the question rows under the arm; this does not.
  - The vote is R6's exactly (checked bit for bit against `s1c_lib.qread_keep`): the question's first n − 1 rows over
    the single store's base keys and dithered values, SnapKV's max-pool of 7, the top C/8 per KV head.
  - The tiers are R6's and R7b's code (`nested_s1h5`, `rquant_s1h5`, e2e_s1h5's arms), with R7b's seed scheme.
- **Theory part:** Theorem M per head-step at every answer step, for the store's 4+r keys (r = 1–4) at the vote's 1/8
  and 1/4, the oracle sets, the re-votes, and dense dithered keys at 6/7/8 bits. The value terms at other widths
  follow exactly (one step per row). The diagnostics:
  - Corollary G's unread share s under each read set;
  - the per-step oracle 1/8, which tests flatness;
  - attention on passage-ID tokens, and on the IDs FP names.
- **Sensitivities:** per layer, c_l = 2 KL_l / err2_l from single-layer arms (the store's keys alone, its values alone,
  dense 7/6), averaged over items 11–18 (R7b's calibration rule).
- **Reads part:** per re-vote rule and layer:
  - the rows new since the last step (churn);
  - the share of a step's rows outside the question's set;
  - the distinct rows outside it over the answer.
- **Checks** (`../test_gate76_s1h5.py`, all pass):
  - the fourth-order coin moments against `g0map_s1h5.coin_terms`: relative error ≤ 3e-5;
  - Theorem M against Monte Carlo draws of both tiers: 0.997 and 1.009;
  - the exact-attention prefill against eager attention: 2e-5;
  - chunked arm layers equal a single call;
  - the re-votes are causal and match R6's row-buffer rule.
- **Theorem M holds per layer at 32K.** Measured / predicted err2 over 480 Llama layer×unit pairs:

  | arm | median | p10–p90 | pooled |
  |---|---|---|---|
  | design | 0.99 | 0.91–1.10 | 1.00 |
  | dense 7/6 | 1.00 | 0.93–1.08 | 0.99 |
  | window-8 re-vote | 0.99 | 0.87–1.11 | 0.98 |
  | 1/4 read | 0.98 | 0.87–1.12 | 0.99 |
  | oracle | 1.00 | 0.97–1.04 | 1.00 |

  Qwen: 0.98–1.01, except dense 7/6 pooled 0.84.
- **The KL bridge on Llama re-ranking** (items 11–18's c, applied to all units): predicted / measured median 1.10
  (design), 1.53 (dense 7/6), 1.34 (window-8 re-vote). It predicts the design/dense gap at 2.6–5.9× against a measured
  2.9–10.1×, so it understates the problem.
  - Re-ranking's c does not transfer to the 2–6-token RAG and ICL answers (about 20× off).
  - On Qwen, one sensitivity unit dominated by L0 (c 552) overpredicts every arm 3–30×.
  - So the controls and Qwen are read end to end only.

## 3. Results

### 3.1 End to end: KL / dense 7/6's (geometric mean over units; units below dense 7/6)

| arm | traffic | Llama rerank 11–18 | Llama rerank 21–28 (held out) | Llama kilt_nq | Llama icl_banking77 | Llama icl_trec | Qwen rerank 11–13 | Qwen kilt_nq | Qwen icl_banking77 |
|---|---|---|---|---|---|---|---|---|---|
| design 4+2/4, question vote | 0.398 | 5.14 (0/8) | 6.29 (0/8) | 0.91 (3/4) | 0.58 (2/2) | 0.28 (2/2) | 2.02 (0/3) | 0.10 (2/2) | 1.06 (1/2) |
| R6's design (6/4, Lloyd-Max vote) | 0.398 | 3.9 (0/8) | 4.14 (0/8) | 0.71 (3/4) | 0.39 (2/2) | 1.2 (1/2) | 1.64 (0/3) | 0.08 (2/2) | 0.51 (2/2) |
| FP8 | 0.500 | 13.1 | 15.8 | 8.4 | 10.8 | 119 | 5.1 | 3.2 | 4.6 |
| 4+2/6 (6-bit values everywhere) | 0.453 | 3.10 (0/8) | 3.13 (0/8) | 0.55 | 0.82 | 0.51 | 1.57 (1/3) | 0.06 | 0.13 |
| 4+2/4 at 1/4 (everywhere) | 0.484 | 4.04 (0/8) | 4.24 (0/8) | 0.26 | 0.20 | 0.17 | 2.11 (0/3) | 0.02 | 0.24 |
| 4+3/4 | 0.426 | 2.64 (0/8) | — | — | — | — | 1.49 (1/3) | — | — |
| G0's allocation (FP8 target) | 0.348 / 0.368 | 28.4 (0/8) | 28.2 (0/8) | 4.4 | 21.8 | 1.4 | 12.0 (0/3) | 4.4 | 2.0 |
| **R6's row buffer, every step** (32 rows of question + answer) | 0.398 | **0.288 (8/8)** | **0.287 (8/8)** | **0.81 (3/4)** | **0.50 (2/2)** | **0.56 (1/2)** | **0.44 (3/3)** | **0.07 (2/2)** | **0.85 (1/2)** |
| R6's row buffer, every 8th step (Stage 1d's qreadr) | 0.398 | 1.27 (2/8) | 1.39 (1/8) | 0.91 | 0.58 | 0.28 | 0.98 (1/3) | 0.11 | 1.01 |
| answer re-vote, window 8 (answer rows only) | 0.398 | 0.080 (8/8) | 0.134 (8/8) | 0.85 (3/4) | 0.90 (1/2) | 0.58 (1/2) | 0.54 (2/3) | 0.07 (2/2) | 1.21 (1/2) |
| answer re-vote, window 1 | 0.398 | 0.126 (8/8) | 0.144 (8/8) | 0.85 (3/4) | 0.90 (1/2) | 0.58 (1/2) | 0.31 (3/3) | 0.21 (1/1) | 1.53 (1/2) |
| window 8 mixed with the question's vote | 0.398 | 0.201 (8/8) | 0.24 (8/8) | 0.99 (2/4) | 0.42 (2/2) | 0.47 (1/2) | 0.41 (3/3) | 0.05 (2/2) | 0.21 (2/2) |
| window 8 over 4+1/3 widths | 0.344 | 0.45 (8/8) | 0.64 (7/8) | 5.1 (0/4) | 12.8 (0/2) | 5.2 (0/2) | 0.84 (2/3) | 1.21 | 1.38 |
| window 8 over G0's allocation | 0.348 / 0.368 | 0.41 (8/8) | 0.57 (7/8) | 4.6 (0/4) | 12.6 (0/2) | 2.6 (0/2) | 0.45 (3/3) | 0.67 | 0.61 |
| oracle read set (answer-pooled 1/8; not deployable) | 0.398 | 5.6e-4 (8/8) | — | — | — | — | 0.13 (3/3) | — | — |

- **The CPU reproduces the GPU's picture without its noise floor.**
  - On re-ranking the design is 2.9–10.1× dense 7/6 on every Llama unit.
  - On RAG and ICL the design is mostly ahead of dense 7/6, as at 128K.
  - FP8 is far behind everywhere.
- **The oracle row is the diagnosis.** The same tier with the read set chosen from the answer's own attention is
  2,000× below dense 7/6 on Llama re-ranking. The tier's precision is not what loses; the read set is.
- **Qwen re-ranking** has the same direction with a smaller gap (2.0×). Its oracle leaves a residual (0.13× dense 7/6,
  against Llama's 5.6e-4×).
  - The cause is not located. With equal layer weights, the oracle's predicted error sits in layers 43–47, which are
    not flat, and there it is about 0.01× dense 7/6's.
  - So the residual is in how sensitive a layer is (Qwen's c are unreliable, §2), or in the CPU arm's exact question
    rows.
- **Prediction for R7b's pending Qwen 128K re-ranking blocks (w-rr):** dense 7/6 below the design there too.

### 3.2 Where dense 7/6 wins (Llama re-ranking, Theorem M per head-step × the layer's c, summed over 15 units)

- **Dense 7/6 is lower in 251 of 1,024 heads, and those heads carry all of the design's predicted KL.** 82 heads carry
  90% of the excess.
- The design's excess by read regime (head-steps):

  | regime | definition | head-steps | share of the excess | share of the design's KL |
  |---|---|---|---|---|
  | flat | the per-step oracle 1/8 leaves > 5% unread | 8,887 | 0.00 | 0.00 |
  | **vote miss** | not flat; the vote leaves > 5%, the answer-pooled oracle < 1% | 136,294 (8%) | **1.07** | **0.99** |
  | other | the rest | 1.5 M | −0.07 | 0.01 |

  Qwen (3 units, its unreliable c): vote-miss head-steps hold 87% of the design's KL and flat ones 0%. Its excess split
  is ill-conditioned there, because the two totals nearly cancel.
- **The top heads**: Corollary G's s under each read set, and attention on passage IDs (means over steps and units).

  | layer | KV | q | design | dense 7/6 | re-vote 8 | s, vote 1/8 | s, vote 1/4 | s, oracle 1/8 | s, per-step 1/8 | s, re-vote 8 | ID mass | named-ID mass | named-ID mass unread |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 16 | 5 | 0 | 2.2e-3 | 2.9e-4 | 6.4e-5 | 0.28 | 0.23 | 0.000 | 0.000 | 0.011 | 0.23 | 0.13 | 0.13 |
  | 2 | 5 | 1 | 1.9e-3 | 3.5e-4 | 7.9e-5 | 0.22 | 0.20 | 0.000 | 0.000 | 0.011 | 0.11 | 0.09 | 0.09 |
  | 10 | 0 | 2 | 1.8e-3 | 2.5e-4 | 1.6e-5 | 0.36 | 0.28 | 0.000 | 0.000 | 0.005 | 0.29 | 0.21 | 0.19 |
  | 14 | 4 | 3 | 1.6e-3 | 1.1e-4 | 2.1e-5 | 0.31 | 0.21 | 0.000 | 0.000 | 0.005 | 0.25 | 0.19 | 0.17 |
  | 8 | 2 | 3 | 1.6e-3 | 1.7e-4 | 3.1e-5 | 0.49 | 0.34 | 0.000 | 0.000 | 0.013 | 0.30 | 0.24 | 0.18 |
  | 16 | 0 | 1 | 1.7e-3 | 3.2e-4 | 5.0e-6 | 0.27 | 0.14 | 0.000 | 0.000 | 0.006 | 0.30 | 0.13 | 0.09 |
  | 17 | 6 | 0 | 1.2e-3 | 2.1e-4 | 1.3e-5 | 0.35 | 0.21 | 0.001 | 0.000 | 0.015 | 0.13 | 0.07 | 0.06 |
  | 15 | 7 | 2 | 9.5e-4 | 1.4e-4 | 8.2e-6 | 0.20 | 0.15 | 0.000 | 0.000 | 0.003 | 0.24 | 0.17 | 0.13 |

  - These heads spend 11–30% of their attention on passage-ID tokens, most of it on the IDs FP's ranking names, and the
    vote leaves 70–100% of that unread.
  - Their s under the vote is 0.2–0.5, above Corollary G's threshold (about 0.06–0.2), yet the per-step oracle leaves
    nothing: each step reads one passage's ID, a different one each time.
  - Over heads, the excess ranks with s under the vote (Spearman +0.65), with the ID mass (+0.49), and least with the
    per-step share (+0.35).
- **The excess is spread over layers.**
  - Measured single-layer KLs: the design's is 3.5–10.6× dense 7/6's in each of the top 12 layers;
  - no layer carries more than 17% (L16 0.17; then L27, L24, L10, L25, L8, L17, L14 at 0.06–0.07 each);
  - keys or values dominate depending on the layer (key/value KL 0.15–5.0).

  So no fix of a few layers can close the gap.
- **The G0 control** (text-continuation steps on five prompt types, R7b's GPU c_l as layer weights; `gate76/g0/`):
  - **Llama:** dense 7/6 / design is 6.9 (PG-19), 5.4 (agent log), 1.4 (code), 1.15 (news) and 0.90 (table).
    Dense 7/6 is lower in 6–13 layers on code, news and tables, and those layers carry 56–86% of the design's KL.
  - **Qwen:** dense 7/6 / design is 1.0–56.
  - So R7b's 10.7× (PG-19, equal weights) came from the most favourable text. On structured text the same staleness
    of a vote cast before the measured steps appears, and the window-8 re-vote is 0.04–0.28× dense 7/6 (Llama) and
    0.01–0.62× (Qwen) on every prompt type.

### 3.3 The three named variants

Corollary G per head-step: MSE_dense / MSE_design ≥ min(ρ_K/s_K, ρ_V/s_V), with ρ_K ≈ 0.2 and ρ_V = 0.057 for dense
7/6. In a vote-miss head-step s ≈ 0.2–0.5 and the unread rows carry the step's attention, so:
- **6-bit values** cut only the value term, by (15/63)² = 0.057, and leave the key term on rows that should have been
  exact;
- **a 1/4 read** adds rows by the same question vote. In the top heads s drops only from 0.20–0.49 to 0.14–0.34
  (the oracle's 1/4 leaves ≤ 1e-4). Measured, a 1/4 read everywhere at 0.484 is still 4.0–4.2× dense 7/6;
- **G0's allocation** targets FP8's per-head error (4+1/3 in most Llama layers, traffic 0.348). FP8 is 13–16× dense
  7/6 here, so meeting FP8 is not enough.

Predicted at the budget (`gate76sum_s1h5.py`; Theorem M per layer × c; the upgraded layers chosen where the predicted
gain per unit of traffic is largest; the design's own predicted ratio is 4.19× with these c):

| variant at ≤ 0.406 | traffic | predicted KL / dense 7/6's: geometric mean (range) over 15 units |
|---|---|---|
| G0's allocation | 0.348 | 27.7 (16.5–38.8) |
| 6-bit values in 4 layers (8, 10, 14, 16) | 0.405 | 3.48 (2.21–4.74) |
| a 1/4 read in 2 layers (14, 16) | 0.404 | 3.96 (2.35–5.32) |
| 4+3 keys in 9 layers | 0.406 | 2.65 (1.53–4.18) |
| water-filled over (r ∈ 1–4, v ∈ 3–8, 1/8 or 1/4) | 0.406 | 1.39 (0.84–2.10) |
| answer re-vote, window 8 (uniform 4+2/4) | 0.398 | 0.097 (0.054–0.22) |

The water-filled optimum takes no 1/4 read in any layer. It moves bits from insensitive layers (4+1/3 in 8 layers) to
sensitive ones (4+3/5 in 9) and still predicts dense 7/6 below it on 14 of 15 units.

**Verdict on the three: none of them beats dense 7/6 at ≤ 0.406 on re-ranking, predicted or measured.**

### 3.4 The re-votes
- **Every re-vote that re-selects at every step beats dense 7/6 on every Llama re-ranking unit, held-out ones
  included** (§3.1). The order on re-ranking is:

  | rule | KL / dense 7/6's |
  |---|---|
  | answer rows only, window 8 | 0.08–0.13× |
  | answer rows only, window 1 | 0.13–0.14× |
  | window 8 mixed with the question's vote | 0.20–0.24× |
  | R6's row buffer (32 rows, question first) | 0.29× |

  Re-selecting every 8th step (Stage 1d's qreadr) does not beat dense 7/6 (1.3–1.4×): the copying heads move to a new
  passage every 3–4 tokens.
- **On RAG and ICL** the re-votes keep part of the design's lead and stay below dense 7/6 on average:

  | model | rule | kilt_nq | icl_banking77 | icl_trec_coarse |
  |---|---|---|---|---|
  | Llama | R6's row buffer | 0.81× | 0.50× | 0.56× |
  | Llama | answer re-vote, window 8 | 0.85× | 0.90× | 0.58× |
  | Qwen | R6's row buffer | 0.07× | 0.85× (1/2 below) | — |
  | Qwen | answer re-vote, window 8 | 0.07× | 1.21× (1/2 below) | — |
  | Qwen | mixed | 0.05× | 0.21× | — |

  These KLs are 1e-6–1e-4, below R7b's GPU floor, so on GPU they would read as ties. Re-ranking decides the gate.
- **Fewer bits under a re-vote win on re-ranking but lose the short-answer tasks.** 4+1/3 (0.344) and G0's allocation
  (0.348) are 0.4–0.6× dense 7/6 on re-ranking but 2.6–12.8× on RAG and ICL. The bits stay at 4+2/4.
- **Churn** (`reads.json`, means over layers, Llama re-ranking, 15 units):

  | rule | rows new per step (share of the 1/8) | share of a step's rows outside the question's set | distinct rows outside it over the answer (× the 1/8) |
  |---|---|---|---|
  | R6's row buffer, every step | 0.012–0.019 | 0.28–0.43 | 0.57–1.15 |
  | R6's row buffer, every 8th step | 0.009–0.014 | 0.25–0.42 | 0.53–1.09 |
  | answer re-vote, window 8 | 0.035–0.065 | 0.42–0.48 | 1.08–1.91 |
  | answer re-vote, window 1 | 0.25–0.30 | 0.46–0.52 | 2.0–3.3 |

  Qwen: the same within 0.01 (row buffer 0.019–0.020 per step).

## 4. The next design: the single store with its vote re-cast at every answer step

### 4.1 Definition
For layer l, answer step t ≥ 1 and KV head g, the read set is

  S_t = top_{C/8} pool₇( Σ_{rows ρ ∈ last 32 of [question rows, answer rows < t]} Σ_{j ∈ group g} softmax_i( q_{ρ,j} · k̄_i · sc ; fixed rows ) ),

where k̄_i are the single store's base keys (4-bit Lloyd-Max, the tier's rotation). This is R6's question-time vote
(`s1d_lib.rows_scores` + `s1c_lib.qread_keep`), fed by `s1d_lib.RowBuffer`'s last 32 rows and recomputed before every
answer step (RowBuffer with k = 1). Step 0 is the question's vote. Everything else is R7b's design: the tier, 4+2
keys and 4-bit values on the unread rows, the per-coordinate debias, and the 1/8 read exactly.

The alternate is the answer-only window of 8. It is better on re-ranking (0.08–0.13× against 0.29×), worse on ICL,
and churns 3–4× more.

### 4.2 Theory (what carries over)
- **R1 by filtration (Lemma C′).** S_t at layer l is a function of the base keys (coin-free) and of layer l's queries at
  steps < t (and the question's). Layer l's query at step t′ depends on the coins of layers < l only: layer l's own
  output reaches step t′ only through attention at layers > l. So S_t is F_l-measurable, teacher-forced. Free-running
  decode is covered up to the first token that differs from FP's (Theorem DD), as for the question's vote.
- **Theorems R and M hold per step** with S_t in place of S. Both are statements for a fixed, F_l-measurable read set
  at one query. Measured / predicted per layer for a re-vote: 0.99 median, 0.98 pooled (§2).
- **Corollary G, restated for the paper.** The share s that decides two tiers against one is the unread second moment
  under the read set actually used at that step. Flatness, where even the per-step oracle leaves s large, is one way
  to make s large. On R6's models at 32K it carries nothing. A read set chosen before the answer exists is the other
  way, and it carries all of the excess. Re-casting the vote per step removes it: in the top heads s falls from
  0.2–0.5 to 0.003–0.015 (window 8).
- E1's certified read already escalates per step by the answer query's base-sketch priority, so it addresses the same
  failure with a certificate. R7b ran no E1 on re-ranking.

### 4.3 Accounting
- **HBM traffic per step** is unchanged in R6's model (0.398): the 1/8 is read in FP16 and the rest from the tier.
- **The vote's scores** come from base keys the tail reads anyway. R6's row buffer re-scores its 32 rows each step:
  compute, not cache traffic. The answer-only window would need a per-row score accumulator, +0.008 at 16 bits.
- **The exact rows move during the answer.** R7b's memory claim (0.449 during a question) fetches the exact 1/8 once
  per question. With the row-buffer rule each step brings in 1.2–2% of the 1/8 (0.15–0.25% of the context's FP16).
  Over an answer of 50–200 tokens the distinct new rows total 0.57–1.15× the 1/8. If FP16 rows come from host memory,
  that is the extra transfer per answer: about 1–2× the question-time fetch, spread over the answer.

### 4.4 Implementation (GPU)
R14 Stage 1d's `qreadr` arms already re-select from `RowBuffer`'s last 32 rows every `RESEL_K` answer steps
(`../../14_kernel_tpot/s1d_lib.RowBuffer.due`, `run_s1d._reselect`; teacher-forced replay in segments of k rows). For
the single store:
- `RESEL_K = 1` (segments of one row in the teacher-forced pass);
- `_reselect` must score against the base keys (`store_r7b.base_sketch`), not `C.STATE.kdeq`, which after `nest_swap`
  holds the dithered reconstruction and would break R1;
- the per-coordinate debias mask must follow the new S_t.

R7b's pinned files cannot change, so this is a new run (R7c) or a plan amendment, for the user to decide. Arms: the
row-buffer re-vote (primary), the answer-only window-8 re-vote (alternate), and the design and dense 7/6 as controls,
in R7b's three claim groups.

### 4.5 Predictions for a GPU run
- **Llama 128K re-ranking:** the re-vote's KL at fp_noise's floor (about 8–9e-4), against the design's 2.3e-3.
- **Gate (ii) on Llama 128K under R7b's rule** (2,000 unit-bootstrap means of d over R7b's 289 units, re-ranking's d
  replaced):

  | scenario | mean d | p |
  |---|---|---|
  | as measured | −5.1e-5 | 0.23 (NOT_SHOWN) |
  | re-ranking's d = 0 (the re-vote ties dense 7/6 at the floor) | −2.0e-4 | 0.0005 |
  | the same, with ICL's lead halved (the CPU's give-back) | −1.6e-4 | 0.0005 |
  | re-ranking's excess only halved | −1.2e-4 | 0.010 |

  Dense 7/6 is the last baseline under Holm, so its level is 0.05. **Prediction: BEATS.**
- **Qwen 32K:** at the floor; not resolvable either way.
- **Qwen 128K re-ranking:** the re-vote below dense 7/6 (CPU 0.44×). RAG and ICL stay the lottery of §1.
- **The primary test against FP8** can only improve: re-ranking was the design's weakest family against FP8 too
  (2.3e-3 against 5.3e-3).

## 5. Limits
- **32K, not 128K.** The cut keeps a third of the passages. The mechanism, copying IDs the question never names, does
  not depend on length.
- **The CPU arms keep the question rows exact.** The GPU arms recompute them under the arm.
- **Few units.** 16 Llama re-ranking units, 3 Qwen. Every one shows the design above dense 7/6 and the every-step
  re-votes below it.
- **The controls are short-answer tasks with 2 to 4 units each.** No RULER family was run on CPU here; they are at the
  GPU floor, and G0's continuation steps stand in for longer answers.
- **The bridge's sensitivities are re-ranking's own.** They predict re-ranking within about 1.5×, not the 2–6-token
  answers, and not Qwen.

## 6. Corrections for R7b's documents (that folder is pinned; recorded here)
- `theory_r7b.md` §6.3, "It fails at the flat heads": it fails where the read set misses the answer's rows. On R6's
  models these are copying heads in re-ranking, not flat heads.
- §5.4b's "dense 7/6 has 10.7× the design's MSE over every layer at 32K" holds for PG-19. On G0's code, news and table
  prompts the ratio is 0.9–1.4 (Llama, R7b's c_l as weights).

## 7. Jobs (Trillium compute, CPU)

| job | what | state |
|---|---|---|
| 2636677 | Llama re-ranking 11–18: theory, arms, single-layer arms | OUT_OF_MEMORY at 53 min after 7 of 8 units (item 13's 200-step arms; arms are now chunked over steps); item 13's arms ran in 2637115 |
| 2636678 | held-out, controls, G0 | NODE_FAIL at 16 min (8 parallel 32K prefills); resubmitted as 2636769 |
| 2636769 | Llama re-ranking 21–28 (core arms) and controls (theory, arms); G0 grid on llama8b and qwen3_30b (400 views) | done |
| 2636679 | Qwen re-ranking 11–13 (single-layer arms on 11), kilt_nq 41–42, icl_banking77 41–42 | OUT_OF_MEMORY at 2.6 h (one Qwen process at 302 GB RSS; 3 at a time is too many); finished units kept, kilt_nq 42's remaining arms in 2637469 |
| 2637115 | follow-up arms (`gate76/arms_followup_llama8b.json`: R6's row buffer every step and every 8th, fewer bits under the re-vote) and reads on every Llama unit | done (2.0 h) |
| 2637469 | follow-up arms and reads on every Qwen unit, 2 at a time | done (2.1 h) |
