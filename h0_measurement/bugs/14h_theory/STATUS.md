# 14h_theory status (rewritten, not appended)

**Updated 2026-10-10, evening.** `theory.md` covers parts 1–7. The folder is isolated from `../14h_methodology_improve/`
(`ISOLATION_PLAN.md`, executed), and every test here passes (the four `--fast` suites, rerun after the isolation).

## What holds
- **Theorem R** (v1, v2, v3, the sure sums, the split): valid for every input. 0 misses in 31 million GPU checks (R5.4,
  v1) and in every brute-force and CPU test.
- **Tightest certificates:**
  - 6–8-bit keys: the mirror split (Llama-1B, 8 bits: 5.6–5.7× the error);
  - 4-bit keys: v3 + sure sums (≈ 480–500×), nothing certified at FP8's level.
- **Certified adaptive read:** at 8-bit keys it meets FP8's per-head error on Llama-1B L1–L3 at traffic 0.46–0.49, below
  FP8's 0.50 (CPU, early layers, one passage). At 6-bit keys, R6's width, it needs 35–45% of rows (traffic 0.55–0.62).
- **Theorem M** predicts the tail's MSE within 2% on every layer tested (8% for GPT-OSS at 4-bit keys).
- **Theorem DD** carries the guarantee through free-running decode up to the first token that differs from the exact
  model's.

## Next steps, ranked by importance to the design paper

The paper these results feed is the design paper proposed in `../../../weekly_reports/Oct4.md` §6. It has no draft or
claims ledger yet; `latex/main.tex` is still the understanding paper. R6 tests the design's primary claim.

| rank | step | tests | impact on the paper | cost, dependency |
|---|---|---|---|---|
| 1 | **Finish R6**: submit the Qwen 128K half on Rorqual, then run the frozen reader once over every block | the primary claim: per claim group × family, `dtail6ux_v4` non-inferior to FP8 on KL per span token (upper bound of mean(KL_design − 1.5·KL_FP8) < 5e-4, `fp_noise` calibration); key secondary: FP-correct accuracy (lower bound > −0.02); reported arms; Theorem R's GPU coverage and certified share (v1, v2) at 6-bit keys | the main results table and the only confirmatory numbers; without Qwen 128K one of three claim groups is missing, and it is the HELMET group where the old vote failed hardest | ≈ 16 GPU-h on Rorqual; **needs the user's go-ahead** |
| 2 | **Claims ledger and theory-section skeleton** | whether every intended claim has evidence, and whether each theorem's assumptions match the protocol that produced it (see "Known gaps" below) | keeps the theory section from claiming tightness the reported GPU metrics don't show; decides which paper owns which result (the Oct 4 report's double-counting warning) | 0 GPU; **done 2026-10-10**: `CLAIMS_LEDGER.md`, `paper_theory_section.md` |
| 3 | **Validate the tight theory at the paper's scale** (an R7 probe): port v3 + sure sums, the split, the certified read and Theorem M's per-head prediction into the run-time library by a plan amendment, then run probe blocks on R6's three groups at 4/6/8-bit keys, with a dither-seed panel | whether the 1B, ≤ 8K CPU tightness holds at 128K on 8B/30B models; the share of the R6 design's head-steps certified within FP8 (R5.4, v1, 6-bit: 0.44–0.52 Llama, 0.38–0.41 Qwen); the certified read's rows and traffic against 0.5; Theorem M's accuracy at 128K and whether the unread share of Σ w²‖v − o‖² ranks families by KL; the dither seed's effect on KL | the theory section's headline numbers come from the paper's own models and lengths, not a CPU proxy | stage the port and its tests now (here); run ≈ 6–10 GPU-h **after R6's reader** (`cert_s1h5` and the probe are frozen until then); step 4 should land first |
| 4 | **Tighten the key part at 6-bit keys**: a sub-Gaussian proxy for the truncated 1 − Z_i (its law is exact) and a Hsu–Kakade–Zhang bound for K = Σ D_i u_i (observed vectors in the split form) | whether the 6-bit split bound falls from ≈ 18× the error toward ≈ 5–8×, with no brute-force failure | the R6 design's certified share; whether its certified read drops below FP8's traffic at 6 bits (now 0.55–0.62), which would let the paper claim certified FP8-level error at ≤ FP8 traffic at its own key width | CPU, 1–2 days; **in progress** (after step 2, as the user asked) |
| 5 | **Close the R1 gap for deployment.** Theorem R needs a vote that never sees the dither; R6's vote reads a separate Lloyd-Max 4/4 store that no reported memory number counts (≈ +0.125–0.25 of FP16 K+V, against the 6/4 tier's 0.31) | CPU first: the selection bias and coverage loss of a vote over the dithered tier itself, against a vote on exact keys at the question's prefill (one exact-key pass per question) | the design's memory/capacity claim against FP8 (0.5); whether the theorem covers the deployed system or only the experimental one | CPU, then a design decision |
| 6 | **Bridge per-head error to KL per token**: E[KL] ≈ ½ Σ_h c_h·MSE_h for unbiased, independent per-head errors, with MSE_h from Theorem M and c_h from injected noise | whether it predicts measured KL across dtail 4/6/8, FP8 and dense 4/4 (Llama-1B on CPU first, then R6's units) | explains the R6 table (the 4- vs 6-bit gap, why re-ranking and aggregation are hardest); a possible second-order corollary: every head certified at FP8's level implies KL ≤ FP8's | CPU first |
| 7 | **A run-time target τ for the certified read** (relative, or calibrated offline from FP8's per-head errors) | which τ keeps end-to-end KL ≤ FP8's at the fewest rows | whether the certified read is a deployable contribution or an oracle analysis | CPU, then the R7 probe |
| 8 | **MLA in Theorem M** (its cross term), checked on Moonlight, DeepSeek-V2-Lite and Kimi | whether the explanation holds for latent attention | only if the paper claims "any softmax attention"; R6's models are GQA | CPU |
| 9 | **4-bit keys at FP8's level** (summand-level truncation or confidence-set inversion, §12) | whether a proof can pay one row's deviation instead of the union's | only the reported, untested `dtail4ux_v4` arm; the paper can state the limit from §12 | research, uncertain |

**Known gaps the ledger tracks** (`CLAIMS_LEDGER.md` §5 has the full checks), each closed by the step in brackets:
- R6's probe reports Theorem R v1 and v2 only; theory.md's tight forms (v3, sure sums, the split) are CPU-only [3, 4];
- Theorem M is validated on CPU only (1B models and three architectures' layers, ≤ 8K) [3];
- R6 uses one dither seed for every unit and arm, so dither variability is not in its confidence intervals [3];
- the certified read's target (FP8's per-head error) is an oracle, not available at run time [7];
- the vote's coin-free store is in no memory number, and a vote over the dithered tier breaks R1 [5];
- the threshold read rule (read sets chosen from the dithered tier itself, §9) is unproven [5].

**Guardrails.** R6's frozen files (`../14h_methodology_improve/plan.md`, "R6 frozen") may not change until its reader has
run; step 3's port lands only after that, through a plan amendment. New theory code goes here.

## Related run
R6 (in `../14h_methodology_improve/`): frozen; Trillium half (Llama 128K, Qwen 32K) submitted 2026-10-10, 26 jobs
1072639–1072667, all pending at 18:49; Rorqual half (Qwen 128K) not submitted. It tests the design Theorem R covers,
`dtail6ux_v4`, against FP8 end to end.
