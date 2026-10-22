# 14i_r7_theory_at_scale status (rewritten, not appended)

**Updated 2026-10-22. R7a is complete: run and read.** All 14 blocks finished on Rorqual and were copied back. The frozen
reader passes every block, the code manifests are identical, the 119-file code pin still verifies, and **all eight
predictions are MET**. The write-up is `findings/R7a.md`; the final read is `findings/R7a_reader.{md,json}` (identical
to `findings/R7a_reader_1022.*`). R7a's files are unfrozen, except `cert_r7.py` and `tier_r7.py`, which stay frozen for
R7b (`CLAUDE.md`, "Frozen and unfrozen").

## The verdicts (`findings/R7a.md` §2)

| | prediction (stated before the run) | measured: Llama 128K / Qwen 32K / Qwen 128K | verdict |
|---|---|---|---|
| P1 | coverage OK for part 8 and v2 | 0 misses in 37.8 M checks per bound | MET |
| P2 | part 8 / error at 6 bits in [4, 7] / [6, 13] / [6, 13] | 5.59 / 7.22 / 8.02 | MET |
| P3 | certified within FP8 at 6 bits ≥ 0.80 / 0.65 / 0.65 | 0.919 / 0.724 / 0.679 | MET |
| P4 | Theorem M consistent with 1 (exact), first order within ±5% at 6–8 bits | 1.000 at 6 and 8 bits everywhere; at 4 bits 1.000 / 0.994 / 0.990 | MET |
| P5 | certified read traffic at 6 bits ≤ 0.45 / 0.50 / 0.50, misses per query head OK | 0.399 / 0.474 / 0.423; 0 misses in 774,656 head checks | MET |
| P6 | part 8 / v2 ≤ 0.30 | 0.217 / 0.212 / 0.211 | MET |
| P7 | SEED_ROBUST in every family | 5 of 5; common seed effect ≤ 0.2% of the variance | MET |
| P8 | unread share orders Llama's families by the design's KL | Spearman 0.857 over 7 families (permutation p = 0.012) | MET |

## The findings in brief (details and tables in `findings/R7a.md` §5)
- The guarantees hold at scale: no miss for part 8, v2 or the certified read.
- Part 8 is 5.6× the error on Llama and 7.2–8.0× on Qwen at 6 bits, 0.21–0.22 of v2. v2 does not change with length
  (25.5× at 128K on Llama, 27× at 1.6K on the CPU).
- The key term dominates part 8 at 4–6 bits, the value term at 8 bits.
- The weakest layers are the ones predicted from R6's probe: Llama L30, L8, L1; Qwen L3, L1, L0, L2, L16.
- The certified read is at the oracle on Llama (0.126 of rows against 0.125, no fallback). On Qwen the extra rows come
  from the early flat layers.
- Theorem M is exact (1.000 ± 0.001 at 6–8 bits). On Qwen at 4 bits the ratio is heavy-tailed, so 2 draws fall short by
  ≤ 1%.
- The unread second moment orders the families by KL on all three groups. Rerank is the top family on both measures.
- Qwen 128K certifies less than Qwen 32K because R7a probed it on HELMET, which is harder on Llama too (0.82–0.88 against
  0.91–0.98 on RULER). Within Qwen, length and task are confounded.
- The dither seed has no common effect; R6's verdict does not depend on it.
- At seed 0 the panel reproduces R6's RULER units exactly. On rerank, 2 of 10 items differ because FP's own answer
  changed between runs.

## How it ran
- 2026-10-12, Trillium: pilots 1080289 and 1080290, then gate 1080291, which PASSED (`findings/R7a_gate.json`). The
  blocks 1080292–1080304 and the reader 1080309 were cancelled while pending (09:17–09:23, from the user's account).
- Resubmitted on Rorqual (the reason for the move is not recorded):
  - pilots 23020651 and 23020652;
  - probe blocks 23020654–60 and Qwen 128K 23020661 (`_0`, `_1`);
  - seed-panel blocks 23020662–66;
  - reader 23020667, which did not run.

  l-hm (23020657) was the last block; its results arrived 2026-10-22.
- Block run times were 0.08–1.72 h, about 16–17 GPU-h in all (the plan estimated 21–34). Peak memory per GPU was
  56–59 GiB (Llama), 66.7 GiB (Qwen 32K) and 49.8 GiB (Qwen 128K, on 2 GPUs).
- The reads: an interim read of 13 blocks on 2026-10-17 (`findings/R7a_reader_partial_1017.*`), then the complete read
  on 2026-10-22 (`findings/R7a_reader_1022.*`, rerun as the final `findings/R7a_reader.*`, byte-identical). All were
  run here with R6's partial reader JSON for P8: 10-11 for the first, 10-18 for the others, with identical Llama rows. The diagnostics behind `findings/R7a.md` §5 are in
  `findings/R7a_diag_partial_1017.py.txt` and `findings/R7a_diag_1022.py.txt`.

## Closed (2026-10-22, the user's decisions)
1. **The reader.** Rorqual's reader job 23020667 did not run, so the frozen reader was run here:
   `findings/R7a_reader.{md,json}`.
   - If 23020667 is still in Rorqual's queue, it can be cancelled.
2. **P8** is closed on R6's partial read of 2026-10-18, not on R6's final read.
   - P8 uses only R6's Llama 128K rows. Those are final, and identical in the 10-11 and 10-18 reads.
   - R6's final read waits only on a Qwen 128K block.
3. **The code snapshot:** `findings/R7a_code_snapshot_1022.tar.gz`.
   - It holds the 119 pinned files plus the pin, verified after extraction; its sha256 is in the `.sha256` file beside it.
4. **The freeze is lifted for R7a's own files.**
   - `cert_r7.py` and `tier_r7.py` stay frozen until R7b's final read: R7b imports and pins them.
   - R6's files stay frozen for R6 and R7b.
5. **Carried back** (the user asked). R7a's at-scale numbers and corrections went into:
   - `../14h_theory/theory.md`, `CLAIMS_LEDGER.md`, `RESEARCH_PLAN.md`, `paper_theory_section.md`, `STATUS.md`;
   - a pointer in `ARCHITECTURES_PLAN.md` and a note in `findings/findings_klbridge.md`;
   - the paper's slides (`latex_paper/icml27_understanding/slides/findings_slides.html`).

## Open
- Nothing for R7a itself.
- `cert_r7.py` and `tier_r7.py` can be unfrozen once R7b's final read exists (R7b's Qwen 128K half runs on Rorqual).

## Corrections carried back to `../14h_theory/` (2026-10-22)
- `../14h_theory/theory.md` §6e: the certified read holds "at each query head" (P ≥ 1 − δ), not "at every query head";
  the joint statement needs δ/(m·r) per bound. R7a tested and confirmed the per-head form.
- `../14h_theory/RESEARCH_PLAN.md` §1.1 compares v2 on Llama-3.2-1B (39–42×, CPU) with Llama-3.1-8B (25×, GPU): a model
  difference. On Llama-3.1-8B, v2 is 27× at 1.6K, 24.6× at 16K and 25.5× at 128K.
- Part 8 at 128K is 5.6× the error on Llama at 6 bits, not the plan's 4–5×.
- The fine schedule is 16 sets at δ/16, not "15 at δ/15". (No such text remained there; nothing to change.)
- `RESEARCH_PLAN.md` §1.1's v2 comparison had already been corrected on 2026-10-12; the measured 5.6× is added.

## The review before the freeze (2026-10-12)
No bug was found in the bounds' mathematics: the copies match bit for bit, and the port matches `../14h_theory`'s
reference to 5·10⁻¹⁴. The review changed six things:
1. The certified read's guarantee is per query head. The probe records `met_heads`, and the reader tests misses
   against δ·heads.
2. The gate's coverage check uses the reader's COVER_OK rule instead of requiring coverage exactly 1.
3. "Identical stops" is stated precisely (`theory_r7.md` §2.3).
4. `tail_err` is tested directly.
5. The worker re-verifies the code pin at every job start.
6. The scope of coverage is stated: FP's queries; the design's own queries rely on Lemma C.

## What was built (unchanged since the freeze)
- **The port:** `cert_r7.py` and `tier_r7.py` (part 8, the certified read with the FINE schedule, Theorem M, v2), with
  parity against frozen fixtures (`fixtures/`).
- **The GPU forms:**
  - the batched, centred HKZ matrix;
  - Lemma S (the HKZ matrix from heavy rows; available but off);
  - Lemma B (bracketed decisions in the certified read; same stop);
  - Lemma P (Theorem M's moments as a separable series).
- **The run side:** `probe_r7.py`, `r7_lib.py`, `run_r7.py`, `read_r7a.py`, `submit_r7.slurm`, `script_r7a.sh`.
- **Tests:** `test_cert_r7.py`, `test_run_r7.py`.
- **The CPU exploration** behind the predictions: `findings/explore_r7.md`, `findings/predictions_from_r6.json`.

## R7b: in `r7b/`, not here
R7b (the single store) has its own theory, run and status in `r7b/` (read `r7b/STATUS.md`). It adds no file to this
folder's top level, so R7a's code manifest is unaffected.
