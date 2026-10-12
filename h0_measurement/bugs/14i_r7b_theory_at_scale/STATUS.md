# 14i_r7_theory_at_scale status (rewritten, not appended)

**Updated 2026-10-12.**
- R7a was reviewed end to end on 2026-10-12 (the user asked): every file, the theory claims and the protocol.
- Extended to Qwen 128K, R6's third claim group, at the user's request.
- Frozen and submitted to Trillium on 2026-10-12 (the user asked), from trig-login01 (`plan.md` §10):
  - pilots 1080289 (probe) and 1080290 (seed panel);
  - gate 1080291;
  - probe blocks 1080292–1080298;
  - Qwen 128K node job 1080299 (blocks `_0`, `_1`);
  - seed-panel blocks 1080300–1080304;
  - reader 1080309.

## Review (2026-10-12)
No bug in the bounds' mathematics. The copies match bit for bit, and the port matches the reference to 5·10⁻¹⁴. The
review changed six things:
1. **The certified read's guarantee is per query head** (`theory_r7.md` §2.6): P(miss) ≤ δ for each head, ≤ rδ for all r
   heads of a group.
   - The probe now records the met count per group (`met_heads`).
   - The reader tests misses against δ·heads; it had compared all-heads-met groups against 1 − 2δ.
   - `../14h_theory/theory.md` §6e says "at every query head" where it should say "at each query head" (to carry back).
2. **The gate's coverage check** now uses the reader's COVER_OK rule. It had required coverage exactly 1, which one
   legitimate miss (the theorem allows rate δ) would have failed.
3. **"Identical stops"** is stated precisely (`theory_r7.md` §2.3). P-exact's batching depends on which groups are
   open; the measured effect on the proxy is 0 to the last bit.
4. **`tail_err` is now checked directly**: the error to 4.5·10⁻¹³ and Z̃/Z to 4.5·10⁻¹⁶ (`test_cert_r7.test_tail_err`).
5. **The worker re-verifies the 119-file code pin at every job start**, so a change to R6's library while R7a is queued
   stops the job instead of silently changing the code.
6. **The scope of the coverage numbers is stated** (`theory_r7.md` §1, §7): FP's queries (the oblivious pass); the design's
   own queries are covered by Lemma C, not measured.

Checked and found right:
- the probe's step indexing (one teacher-forced call; every measured step once);
- the GQA head grouping, the seeds and rotations (R6's), the debias, v2's arguments (R6's probe);
- the vote and the sketch (coin-free);
- Theorem M's identity and moments, re-derived;
- Lemmas S, B and P, re-derived;
- the HKZ validity when the P-exact rows are chosen with coin-dependent scores (the proxies themselves are coin-free);
- the panel (b) model, the reader's bootstraps, R6's reader JSON format (P8).

## Done
- **The port** (`cert_r7.py`, `tier_r7.py`). These are the bounds R7a measures:
  - part 8's certificate;
  - the certified adaptive read with the FINE schedule;
  - Theorem M per head-step;
  - v2 (R6's column).

  They import nothing outside torch. The pieces of `cert_s1h5`/`rquant_s1h5` they build on are copies, checked bit for
  bit (`test_cert_r7.test_copies`). Part 8, the certified read and Theorem M equal `../14h_theory`'s reference to
  5·10⁻¹⁴ on frozen inputs (`fixtures/`).
- **The GPU forms** (`theory_r7.md` §2), each keeping the guarantee:
  - the batched, centred HKZ matrix;
  - Lemma S, the matrix from its heavy rows (available, off: its inflation grows with context);
  - Lemma B, the certified read decided by brackets before forming the matrix (same stop; skips 50–90% of matrices);
  - Lemma P, Theorem M's exact moments as a separable series (2.7·10⁻¹³).
- **Tests.**
  - `test_cert_r7.py --fast` (3 min): brute force (1,280 head cases, 0 failures), quadrature, Monte Carlo, the
    certified read's stop against a naive rule, parity.
  - `--quick` (24 s): the worker's preflight.
  - `test_run_r7.py --fast`: presets, layout, fresh prompts, the probe against direct computations (exact).
  - Driver smokes on Llama-3.2-1B at 4K: the probe's pass is FP's, coverage 1, part 8 6.0× the error at 6 bits; the seed
    panel's five seeds give five KLs; the reader reads both. All pass.
- **The run side.**
  - `r7_lib.py`: presets, seed-suffixed design arms, `R7_LAYOUT` (13 blocks; Qwen 128K as a 4-GPU node job with two
    2-GPU blocks).
  - `probe_r7.py`.
  - `run_r7.py`: `run_s1h5.main` with R7a's library and probe swapped in at run time; no R6 file edited.
  - `read_r7a.py`: rules, the gate, predictions P1–P8.
  - `submit_r7.slurm` (node split, pin re-check), `script_r7a.sh`: the dry run assembles 2 pilots, the gate, 13 block
    jobs (14 blocks) and the reader.
  - `findings/R7a_code.sha256` pins 119 files.
- **The exploration** (`findings/explore_r7.md`; Llama-3.1-8B L1/L3, Qwen3-30B-A3B L1/L12; 1.6K–16K; CPU):
  - part 8's tightness is flat in context (Llama 4.2–5.5×, Qwen 11–15× at 6 bits);
  - certification within FP8 and the certified read improve steadily with context, because the vote leaves a smaller
    share of Σ w²‖v − o‖² unread;
  - Llama L3 at 16K is certified on 0.97 of head-steps and its certified read is at the oracle (traffic 0.400 vs 0.398);
  - Qwen's flat L1 is the exception;
  - Theorem M is within 2% at 6–8 bits;
  - coverage is 1.000 in all 22 cells.
- **Predictions from R6's GPU probe** (`findings/predictions_from_r6.json`):
  - part 8 certified within FP8 at 6 bits: 0.88–0.91 on Llama 128K (v2: 0.625), 0.71–0.76 on Qwen 32K (v2: 0.487);
  - the certified read at 6 bits: about 0.127 of rows (traffic 0.40) on Llama, 0.25 (0.49) on Qwen.

## A correction to carry back (not edited there: isolation)
`../14h_theory/RESEARCH_PLAN.md` §1.1 reads "v2 is 39–42× the error on the CPU (Llama-1B …) and 25× on the GPU at 128K.
Scale did not hurt." That compares two models. On Llama-3.1-8B itself, v2 is 27× at 1.6K, 24.6× at 16K and 25.4× at
128K. Scale does not hurt, but nothing improved with length either. Part 8 / v2 is 0.19–0.22 on Llama-3.1-8B at 6 bits,
so part 8 at 128K should be about 5–6×, not the plan's 4–5×.

The docs also call part 8's fine schedule "15 sets at δ/15". The code (theory5 and here) has 16 sets at δ/16. Valid
either way; R7a uses 16.

**A parallel session works in `../14h_theory`** (the nested store N, the KL bridge). On 2026-10-11 at 21:09,
`theory5_s1h5.exact_rows` gained per-coordinate steps ([G, C, d], for the nested store). Its scalar-step path, the one R7a
uses, is unchanged: the fixtures were generated from that version (sha 28a6ae9d) and parity holds to 5·10⁻¹⁴. R7b's
nested-store arm will need the per-coordinate form ported.

## R7b (2026-10-12): in `r7b/`, not here
R7b's theory, the per-coordinate port (done: `r7b/cert_r7b.py`, `r7b/store_r7b.py`), its tests and its CPU exploration
live in `r7b/` (read `r7b/STATUS.md`). R7a's 119 pinned files and this folder's top-level `*.py` are unchanged (the pin
verifies), so R7a's code manifests stay identical across its blocks.

## Next
1. Watch the chain: pilots → gate → blocks → reader (`plan.md` §10).
2. After the run: `read_r7a.py --r7a $(script_r7a.sh --list-r7a) --r6-json <R6's final reader JSON>`. Then
   `findings/R7a.md`; the theory section's numbers (T3, T4, T6) and C1's seed statement come from it.
3. If the gate projects a block over its wall, set `probe_r7.CR_GRAM_ROWS = 4096` (Lemma S) or trim the 4-bit draws,
   before any block runs.
