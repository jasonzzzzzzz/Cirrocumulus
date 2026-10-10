# 14h_theory status (rewritten, not appended)

**Updated 2026-10-10.** `theory.md` covers parts 1–7. The folder is isolated from `../14h_methodology_improve/`
(`ISOLATION_PLAN.md`, executed), and every test here passes.

## What holds
- **Theorem R** (v1, v2, v3, the sure sums, the split): valid for every input. 0 misses in 31 million GPU checks (R5.4,
  v1) and in every brute-force and CPU test.
- **Tightest certificates:**
  - 6–8-bit keys: the mirror split (Llama-1B, 8 bits: 5.6–5.7× the error);
  - 4-bit keys: v3 + sure sums (≈ 480–500×), nothing certified at FP8's level.
- **Certified adaptive read:** at 8-bit keys it meets FP8's per-head error on Llama-1B L1–L3 at traffic 0.46–0.49, below
  FP8's 0.50 (CPU, early layers, one passage).
- **Theorem M** predicts the tail's MSE within 2% on every layer tested.
- **Theorem DD** carries the guarantee through free-running decode up to the first token that differs from the exact
  model's.

## Open
- **The key part at 4–6 bits.** A sub-Gaussian proxy for e^{−δ} − E e^{−δ} would give it a Hsu–Kakade–Zhang bound.
  Alternatively, a confidence-set or Fuk–Nagaev-style truncation that pays one row's deviation, not the union's (§12).
- **A GPU check of part 7 at 128K.** R6's probe is frozen and reports v1 and v2 only. The split and the certified read
  need a probe column set, ported into `cert_s1h5` by a plan amendment after R6.
- **MLA in Theorem M**: its cross term.
- **The threshold read rule**: read sets chosen from the dithered tier itself (§9).
- **A run-time target τ for the certified read.** FP8's per-head error is an oracle target; a relative target works
  (Gemma 3: 1%).

## Related run
R6 (in `../14h_methodology_improve/`, frozen, not submitted) tests the design Theorem R covers, `dtail6ux_v4`, against
FP8 end to end.
