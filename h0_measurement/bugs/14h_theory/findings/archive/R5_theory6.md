> **Archived (2026-10-10).** Superseded by `../findings_part7.md`: the same runs, plus part 6's bounds. Its code,
> `14h_methodology_improve/certread_s1h5.py`, became `../../theory4_s1h5.py`.

# R5 theory, part 6: a tighter Theorem R and the certified adaptive read (CPU, 2026-10-10)

Code: `certread_s1h5.py` (new module; R6's frozen files are untouched). Proofs: `theory.md` §6. Tests:
`test_theory_s1h5.py` (`test_split`, `test_certified_read`). Data: `R5_theory6_certread.json`.

## What is new

1. **v3: sure sums.** On the truncation event that Theorem R already uses, w_i ≤ w̄_i, so every unknown weighted
   sum is bounded surely by its w̄ version, at no extra probability cost. The self-bound inflated the variance sum
   7–17× on Llama-1B L1, because a few heavy rows carry the tail.
2. **The split by coin source** (GQA: independent key and value dithers):
   - Z(o − õ) = K − V − (õ − c)Σ(w − w̃), where V = Σ w_i e_i has fixed weights and value coins only.
   - A dithered value error is isotropic and strictly sub-Gaussian, so Hsu–Kakade–Zhang bounds ‖V‖ at about 1.4 ×
     its standard deviation. Bernstein's Hilbert-space bound cost about 4.2×.
   - K and the weight sum keep Bennett, with sure sums.
3. **Certified adaptive read** (corollary):
   - nested read sets chosen without the coins, a union bound at δ/m, and a stop at the first set whose bound meets
     the target;
   - with probability ≥ 1 − δ, the output meets its target;
   - the coins may decide how many rows are read, but not which.
4. **A negative result.** A dimension-aware bound for the whole sum (Bousquet) gains nothing: its 2·range·E‖S‖ term
   costs what it saves.

## Validity

Brute force: random, spiky, 30×-outlier and adversarial data at 2–8-bit keys, δ from 0.5 to 10⁻³.
- v3 and the split form: 0 failures in 12,800 cases; the worst error/bound at δ = 0.5 was 0.20.
- The certified read: 0 misses, with stops before the exact read.
- On real weights: coverage 1.000 in every layer, key width and target below.

## Tightness on real weights

Llama-3.2-1B L1, 1,600 tokens, unbiased tail. Each cell gives the median bound / error and the share of head-steps
certified within FP8's error:

| key bits | read | v2 | v3 (sure) | split |
|---|---|---|---|---|
| 6 | vote | 41.7 · 0.04 | 23.9 · 0.09 | 18.3 · 0.13 |
| 6 | + C/32 | 30.6 · 0.20 | 20.6 · 0.34 | 16.0 · 0.44 |
| 8 | vote | 19.3 · 0.24 | 9.9 · 0.45 | **5.7 · 0.66** |
| 8 | + C/128 | 16.2 · 0.43 | 9.2 · 0.65 | 5.4 · 0.80 |
| 8 | + C/32 | 14.6 · 0.60 | 8.8 · 0.77 | **5.3 · 0.88** |

At 4 bits every form is still 500–1,000×: the key part dominates.

## The certified adaptive read

Setup:
- schedule: the vote's 1/8, then 0, 1/256, …, 1/2 more rows by a Lloyd-Max sketch's order, then everything;
- δ = 10⁻³ per head-step, 2 draws;
- "rows read" = the share read exactly, and traffic = (1 − f)(k + 4)/32 + f;
- oracle stop = the first set whose true error meets the target (needs the exact data; a lower limit).

Target FP8's own per-head error (an oracle target, not available at run time):

| model / layer | key bits | v2: rows (traffic) | split: rows (traffic) | oracle stop |
|---|---|---|---|---|
| Llama-1B L1 | 8 | 0.355 (0.597) | **0.167 (0.479)** | 0.125 |
| Llama-1B L2 | 8 | 0.268 (0.542) | **0.143 (0.464)** | 0.125 |
| Llama-1B L3 | 8 | 0.330 (0.581) | **0.180 (0.488)** | 0.125 |
| Llama-1B L0 | 8 | 0.815 (0.884) | 0.449 (0.656) | 0.134 |
| Llama-1B L1–L3 | 6 | 0.50–0.60 | 0.35–0.45 (0.55–0.62) | 0.125 |
| Gemma 3 L5 | 6, 8 | 1.0 | 1.0 | 0.63: the 4-bit values cannot reach FP8 here |

Target 1% of the output's norm (available at run time):

| model / layer | key bits | v2: rows (traffic) | split: rows (traffic) | oracle stop |
|---|---|---|---|---|
| Llama-1B L1–L3 | 8 | 0.51–0.57 (0.70–0.73) | **0.26–0.29 (0.54–0.56)** | 0.126–0.129 |
| Gemma 3 L5 | 8 | 0.962 (0.976) | **0.169 (0.481)** | 0.125 |
| Gemma 3 L5 | 6 | 1.0 | 0.380 (0.574) | 0.125 |
| Granite-4.0-H L5 | 8 | 1.0 | 0.994 | 0.881: 4-bit values miss 1% almost everywhere |

## Reading

- **At 8-bit keys the guarantee costs no more than FP8's traffic.** On Llama-1B layers 1–3 (FP8's error as the
  target) and Gemma 3's global layer (1% target), the certified read reads 14–18% of rows exactly and its traffic
  is 0.46–0.49, against FP8's 0.50. With v2 the same guarantee cost 0.54–0.98.
- **The value bits set the floor, per model.** Gemma 3 cannot reach FP8's error with 4-bit values at any read, and
  Granite cannot reach 1%. There the design needs more value bits, not a better proof.
- **Layer 0 is harder** (0.45 of rows). The first layer's attention is broad, and the key part dominates.
- **4 bits remains out of reach**, now because of the key part alone (theory.md §9).

## Next

- **A GPU check.** These are CPU numbers on early layers, one 1,600-token passage. A GPU probe with the split form
  and the certified read on R6-like units would give the 128K picture. It would need a new probe column set after
  R6 (whose probe files are frozen).
- **The key part.** A sub-Gaussian proxy for e^{−δ} − E e^{−δ} would give K a Hsu–Kakade–Zhang bound too. That is
  the remaining route to 6 bits, and perhaps 4.
