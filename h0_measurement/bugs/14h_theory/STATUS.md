# 14h_theory status (rewritten, not appended)

**Updated 2026-10-10, night.** `theory.md` covers parts 1–8. The folder is isolated from `../14h_methodology_improve/`
(`ISOLATION_PLAN.md`, executed). Every test here passes: the five `--fast` suites, including part 8's
`test_theory5_s1h5.py`. Cached views for the CPU harnesses are in `.views/`.

## What holds
- **Theorem R** (v1, v2, v3, the sure sums, the split, part 8): valid for every input. 0 misses in 31 million GPU
  checks (R5.4, v1) and in every brute-force and CPU test.
- **Tightest certificates (part 8, `theory5_s1h5.py`):** the split with a tilted tail and a Hsu–Kakade–Zhang key sum,
  using a sub-Gaussian proxy from the dither's exact law. CPU, Llama-1B, the vote's rows read, δ = 10⁻³:

  | key bits | bound / error | share certified within FP8 | part 7's split |
  |---|---|---|---|
  | 6 | 6.7–7.3× (L1/L3, 1.6K–8K) | 0.41–0.59 | 17–19×, 0.13–0.21 |
  | 8 | 2.0–2.6× | 0.89–0.95 (L1/L3) | 5.3–5.8× |
  | 4 | 92–554× | none | — |

  The 4-bit figure beats v3 + sure sums' ≈ 480–500×, but is still far from FP8's level.
- **Certified adaptive read** (part 8's bound, FP8's per-head error as an oracle target):
  - 6-bit keys, R6's width: traffic 0.480 (L1), 0.499 (L3) and 0.433 (L1 at 8K), with a 15-set schedule, against FP8's 0.50;
  - 8-bit keys: 0.456–0.460, against the oracle stop's 0.453;
  - L0 (flat attention) still needs 0.73 at 6 bits.
- **Theorem M** predicts the tail's MSE within 2% on every layer tested (8% for GPT-OSS at 4-bit keys).
- **Theorem DD** carries the guarantee through free-running decode up to the first token that differs from the exact
  model's.

## Next steps, ranked by importance to the design paper

The paper these results feed is the design paper proposed in `../../../weekly_reports/Oct4.md` §6. Its claims are in
`CLAIMS_LEDGER.md` and its theory section is sketched in `paper_theory_section.md`; `latex/main.tex` is still the
understanding paper. R6 tests the design's primary claim.

Step 4 is done. It moved two things:
- step 3 must now port part 8;
- the run-time target (now rank 6) is the only thing between the CPU certified-read result and a deployable claim.

| rank | step | tests | impact on the paper | cost, dependency |
|---|---|---|---|---|
| 1 | **Finish R6**: submit the Qwen 128K half on Rorqual, then run the frozen reader once over every block | the primary claim: per claim group × family, `dtail6ux_v4` non-inferior to FP8 on KL per span token (upper bound of mean(KL_design − 1.5·KL_FP8) < 5e-4, `fp_noise` calibration); key secondary: FP-correct accuracy (lower bound > −0.02); reported arms; Theorem R's GPU coverage and certified share (v1, v2) at 6-bit keys | the main results table and the only confirmatory numbers; without Qwen 128K one of three claim groups is missing, and it is the HELMET group where the old vote failed hardest | ≈ 16 GPU-h on Rorqual; **needs the user's go-ahead** |
| 2 | **Claims ledger and theory-section skeleton** | whether every intended claim has evidence, and whether each theorem's assumptions match the protocol that produced it | keeps the theory section from claiming tightness the reported GPU metrics don't show; decides which paper owns which result | **done 2026-10-10**: `CLAIMS_LEDGER.md`, `paper_theory_section.md` (updated for step 4) |
| 3 | **Validate the tight theory at the paper's scale** (an R7 probe). Port into the run-time library by a plan amendment: part 8's bound (`tail_bound_rand_split_hkz` with the exact-law proxy; Bennett kept as a second form at 4 bits), the certified read with the 15-set schedule, and Theorem M's per-head prediction. Then run probe blocks on R6's three groups at 4/6/8-bit keys, with a dither-seed panel | whether the 1B, ≤ 8K CPU numbers hold at 128K on 8B/30B models. Measures: the share of head-steps certified within FP8 (R5.4, v1, 6-bit: 0.44–0.52 Llama, 0.38–0.41 Qwen); the certified read's rows and traffic against 0.5; Theorem M's accuracy at 128K and whether the unread share of Σ w²‖v − o‖² ranks families by KL; the dither seed's effect on KL | the theory section's headline numbers come from the paper's own models and lengths, not a CPU proxy | stage the port and its tests here; run ≈ 6–10 GPU-h **after R6's reader** (`cert_s1h5` and the probe are frozen until then). GPU notes: the d×d Gram per head-step over C rows (or power iteration for ‖M‖_op, with ‖M‖_F² ≤ ‖M‖_op·tr M); the exact proxy for 16 rows per head-step in fp64 |
| 4 | **Tighten the key part at 6-bit keys** | whether the 6-bit bound falls from ≈ 18× the error toward ≈ 5–8× with no brute-force failure, and whether the certified read gets below FP8's traffic at 6 bits | the R6 design's certified share and certified read | **done 2026-10-10** (part 8, `findings/findings_part8.md`): 6.7–7.3× and 0.41–0.59 certified; the certified read at 0.433–0.499 on L1/L3 (1.6K–8K) |
| 5 | **Close the R1 gap for deployment.** Theorem R needs a vote that never sees the dither. R6's vote reads a separate Lloyd-Max 4/4 store that no reported memory number counts (≈ +0.125–0.25 of FP16 K+V, against the 6/4 tier's 0.31) | CPU first: the selection bias and coverage loss of a vote over the dithered tier itself, against a vote on exact keys at the question's prefill (one exact-key pass per question). The certified read's sketch order needs the same | the design's memory/capacity claim against FP8 (0.5); whether the theorem covers the deployed system or only the experimental one | CPU, then a design decision |
| 6 | **A run-time target τ for the certified read** (relative, or calibrated offline from FP8's per-head errors) | which τ keeps end-to-end KL ≤ FP8's at the fewest rows | whether "certified FP8-level per-head error at about FP8's traffic, at 6-bit keys" is a deployable contribution or an oracle analysis | CPU, then the R7 probe. Moved up from 7: step 4 made the oracle-target result strong enough to be worth making deployable |
| 7 | **Bridge per-head error to KL per token**: E[KL] ≈ ½ Σ_h c_h·MSE_h for unbiased, independent per-head errors, with MSE_h from Theorem M and c_h from injected noise | whether it predicts measured KL across dtail 4/6/8, FP8 and dense 4/4 (Llama-1B on CPU first, then R6's units) | explains the R6 table (the 4- vs 6-bit gap, why re-ranking and aggregation are hardest); a possible second-order corollary: every head certified at FP8's level implies KL ≤ FP8's | CPU first |
| 8 | **MLA in Theorem M** (its cross term), checked on Moonlight, DeepSeek-V2-Lite and Kimi | whether the explanation holds for latent attention | only if the paper claims "any softmax attention"; R6's models are GQA | CPU |
| 9 | **Paying one row's deviation instead of the union's** (summand-level truncation or confidence-set inversion, §12) | whether a proof can avoid the sure weights' e^t: ≈ 1.75× at 6 bits (where the rest is now within ≈ 1.1× of the optimal proxy) and ≈ 10× at 4 bits | the remaining 6-bit slack (the certified read's 0.24–0.27 rows against the oracle's 0.125); FP8-level certificates at 4 bits for the reported `dtail4ux_v4` arm | research, uncertain |

**Known gaps the ledger tracks** (`CLAIMS_LEDGER.md` §5 has the full checks), each closed by the step in brackets:
- R6's probe reports Theorem R v1 and v2 only; theory.md's tight forms (part 8, v3, the split) are CPU-only [3];
- Theorem M is validated on CPU only (1B models and three architectures' layers, ≤ 8K) [3];
- R6 uses one dither seed for every unit and arm, so dither variability is not in its confidence intervals [3];
- the certified read's target (FP8's per-head error) is an oracle, not available at run time [6];
- the vote's coin-free store is in no memory number, and a vote over the dithered tier breaks R1 [5];
- the threshold read rule (read sets chosen from the dithered tier itself, §9) is unproven [5].

**Guardrails.** R6's frozen files (`../14h_methodology_improve/plan.md`, "R6 frozen") may not change until its reader has
run. Step 3's port lands only after that, through a plan amendment. New theory code goes here.

## Related run
R6 (in `../14h_methodology_improve/`): frozen. The Trillium half (Llama 128K, Qwen 32K) was submitted 2026-10-10 as
26 jobs, 1072639–1072667, all pending at 18:49. The Rorqual half (Qwen 128K) is not submitted. R6 tests the design
Theorem R covers, `dtail6ux_v4`, against FP8 end to end.
