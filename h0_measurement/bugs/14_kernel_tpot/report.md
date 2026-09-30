# R14 · Kernel, iso-budget, TPOT — report

Protocol: `plan.md` for Stages 0 and 1 (amendment A1). For Stage 1b, the rules
were frozen in `read_stage1b.py`'s docstring before any Stage 1b output existed
(amendment A2).

## Decision (2026-09-29)

**`STOP_SYSTEMS` (Stage 1b's frozen D2), recorded with a caveat. The kernel
(Stage 2) is not built.**

- **Under the frozen rule, no byte win can be certified.** No calibrated router
  (SIEVE or pooled) is lossless in either cell, with exact or 4-bit values. The
  one lossless joint point is a hybrid with 2-bit values at 128K, which is
  non-gating.
- **The caveat: the verdict is decided at the noise floor.**
  - A change that leaves full precision untouched (quantizing the values) moves
    compressed arms' scores by up to ±0.024.
  - The decisive calls miss the −0.02 margin by 0.001–0.002.
  - Pooled with Stage 1's prompts (40 at 128K), SIEVE-4 and TurboQuant-3 are
    both lossless, and Stage 1's `BYTES_TIE` stands (Part C, C2).
- **Robust findings:**
  - Span pooling removes the deletion failures (`POOL_FIXES`).
  - A teacher-forced answer likelihood predicts the failures that per-head
    output error cannot see (`SEQUENCE_PROXY`, AUC 0.90).
  - Values compress to 4 bits almost free.
- **Next** (Part D): measure the design ideas first, with at least 40 prompts per
  cell, a continuous quality metric and a tail metric:
  - a value-aware hybrid;
  - a sequence-calibrated pooled router;
  - dense storage with question-time reads.

| stage | jobs | decision | where |
|---|---|---|---|
| 0: bytes frontier from R12 rows (0 GPU) | — | `GO_KERNEL` (3 of 7 units WIN, all at ≤ 32K); 128K `NO_POINT` | Part A |
| 1: Llama @128K, B = 2–4, FP8 | 996008–996013 | `REENTER_STAGE0` → 128K `BYTES_TIE` (ρ 0.768 / 0.845) | Part B |
| 1b: hybrids, value twins, pooled router, sequence proxy | 999908–999916 | `STOP_SYSTEMS` (resolution-limited), `POOL_FIXES`, `SEQUENCE_PROXY` | Part C |

---

# Part A · Stage 0 — the bytes-read frontier

Run 2026-09-27 on the login node (CPU, under one second):
`bytes_model.py` → `stage0.{json,md}`. It used the rules in `plan.md` §4 and §8
without change, and no amendment was needed. `test_r14.py` passes 9/9 checks,
including R12's side-bit rule, the model shapes against the cached HF configs,
and the decision mapping.

Inputs: the R12 A grid (6 jobs), the Mistral D grid (3 jobs) and the hard cell
(reported only). All are corpus b524da5e, with 20 prompts per unit (60 for the
hard cell) and every arm paired within its job. Bytes are per context token,
per KV head, per layer, with the BF16 tail (window, question, answer) included.

## A1. Verdicts

**Stage 0 decision: `GO_KERNEL`. Stage 1 is needed: Llama @128K is `NO_POINT`.**

| unit | S\* | D\* | ρ, V16 | ρ, V8 | S\* vs FP8, V8 | verdict |
|---|---|---|---:|---:|---:|---|
| Llama-3.1-8B @8K | SIEVE-3 | TurboQuant-2 | 0.725 | **0.804** | 0.515 | TIE (lens-dependent) |
| Qwen3-8B @8K | SIEVE-3 | KIVI-32-2 | 0.683 | 0.734 | 0.509 | **WIN** |
| Llama-3.1-8B @32K | SIEVE-3 | TurboQuant-2 | 0.684 | 0.766 | 0.486 | **WIN** |
| Qwen3-8B @32K | — | KIVI-32-2 | — | — | — | NO_POINT |
| Llama-3.1-8B @128K | — | TurboQuant-3 | — | — | — | NO_POINT |
| Mistral-7B @8K | SIEVE-3 | TurboQuant-3 | 0.743 | 0.781 | 0.548 | **WIN** |
| Mistral-7B @32K | — | TurboQuant-3 | — | — | — | NO_POINT |
| Llama @32K, hard cell (not gated) | SIEVE-3 | KIVI-32-2 | 0.693 | 0.740 | 0.510 | (WIN) |

Here S\* is SIEVE's cheapest lossless point and D\* is the cheapest lossless
dense quantizer. ρ = bytes(S\*) / bytes(D\*), and ρ ≤ 0.80 is a WIN.

Where SIEVE is lossless at 3 bits, it reads **0.68–0.77×** the bytes of the
cheapest lossless dense quantizer and **about 0.5×** FP8 KV's. Where it is not
lossless within 3 bits, no byte statement is possible. It is never lossless at
2 bits.

## A2. What the verdicts rest on

- **The whole advantage is value bytes that eviction never reads.** At its
  lossless point SIEVE spends *more* key bits than D\* (3.10 against 2.13–3.00)
  and evicts 31–42% of tokens. The break-even value width v\* makes this
  concrete:
  - on Llama, a dense 2-bit quantizer that also stored its values below about
    2.3–2.6 bits would read no more than SIEVE;
  - on Qwen and Mistral, D\* already needs 3+ key bits, and v\* ≈ 0.
- **Three verdicts sit on a threshold:**
  - Llama @8K, V8: ρ = 0.804 against the 0.80 line, which makes the unit
    lens-dependent.
  - Qwen3-8B @32K: SIEVE-3 scores 0.980 (Δ = −0.020, exactly the point
    margin), but its 90% lower bound is −0.053 against −0.05, so it is not
    lossless.
  - The hard cell's D\* (KIVI-32-2) passes with a lower bound of exactly
    −0.050.

  All three follow the frozen rule and stay as they are. A different sample
  could move any of them.
- **The oracle router** (a diagnostic) is lossless in two units where the
  calibrated router is not: Qwen @32K (at 2 and 3 bits) and Mistral @32K (at 3
  bits). Calibration, not the allocator, is what misses there.
- **FP8 KV is assumed lossless** (pending Stage 1), as is FP8 V under every arm
  in the V8 lens. The V16 lens is fully measured, and it gives the same verdicts
  except for Llama @8K.

## A3. Predictions (plan.md §3), checked

| prediction | outcome |
|---|---|
| P1: Llama @128K has no lossless SIEVE point at ≤ 3 bits | **held.** SIEVE-3 0.721 (Δ −0.263). The oracle router reaches 0.956 (lower bound −0.061), also not lossless |
| P2: WIN in Llama 8K/32K and Qwen 8K/32K, NO_POINT on Mistral | **partly wrong.** Llama @8K is a lens-dependent TIE (0.804), Qwen @32K is NO_POINT (lower bound −0.053), and Mistral @8K is a WIN (0.743 / 0.781). Predicted ρ 0.65–0.71 (V8); measured 0.73–0.78, because SIEVE-3 evicts only 31–42% at ≤ 32K, not the ~50% assumed |
| P3: FP8 is no longer the bar | **held.** Every lossless dense D\* reads 0.64–0.70× FP8's bytes (V8) |
| P4: a per-step bc = 4 scan erases D-rebudget's saving | **held.** At k = 1, ρ = 1.09–1.19 (V8, revivable eviction). ρ ≤ 0.80 needs k ≥ 8–16 (V8) or k ≥ 2–4 (V16) |
| P5: batch 1 dilutes everything | **held.** At 32K, batch 1, S\* against D\* per step (weights included) is 16.0 vs 16.3 GB on Llama (V8). The same figure at batch 4 is 18.9 vs 20.1 GB |

## A4. Whole-model context (not gated)

Capacity assumes one H100-80GB, BF16 weights and a 6 GB reserve. At 32K on
Llama (V8), the number of sequences that fit is:

| format | sequences |
|---|---:|
| FP8 KV | 32 |
| TurboQuant-2 (D\*) | 50 |
| SIEVE-3 (D-once, stored compacted) | 65 |

At 8K the KV cache is a small fraction of each step's bytes, and any gain
there is capacity only.

## A5. Consequences at the time

*Point 1 is superseded by the Stage 1b decision above: no kernel yet. Stage 1b
also did not replicate the Llama @32K WIN (C1).*

1. **`GO_KERNEL`.** Stage 2 has three units to realize, all at ≤ 32K:
   - Llama @32K: ρ = 0.68 / 0.77. The primary cell, and the one at the most
     memory-bound length.
   - Qwen @8K and Mistral @8K.

   plan.md §6 names Llama @32K and Qwen @32K as the Stage 2 cells. Qwen @32K
   is NO_POINT, so its place goes to Qwen @8K or Mistral @8K. §8 ("Stage 2 on
   those units") already implies this; it is not a rule change.
2. **Stage 1 is needed.** The ROADMAP's cell, TPOT at 128K, cannot be judged on
   R12's budgets. The byte model gives Stage 1 a concrete target. SIEVE-4 at
   Llama @128K must reach a lossless score (≥ 0.964, lower bound ≥ −0.05). It
   must also read ≤ 0.80 × 178.3 = 143 bytes (V8), or ≤ 245 bytes (V16), which
   at 4.1 bits means evicting at least ~41% of tokens (V8) or ~30% (V16).
3. **For the paper,** a claim within scope, with no kernel needed: *on total
   bytes read (values counted), SIEVE at 3 key bits matches full-precision
   accuracy while reading 0.68–0.77× the bytes of the cheapest lossless dense
   quantizer in 3 of 7 units. It does not reach full precision within 3 bits in
   the other 3 units, including 128K*. This reverses R12's key-bit verdict in
   those units, and the reversal comes entirely from value bytes that eviction
   never reads.
4. **D-rebudget,** the paper's two-clock design, keeps the byte advantage only
   when it re-budgets every ≥ 8 steps under FP8 values. F3 found that
   allocations go stale within steps.

---

# Part B · Stage 1 — Llama-3.1-8B @128K at B = 2, 3, 4

Jobs:
- pilot 996008 (excluded) and gate 996009 (pass);
- calibration 996010 on prompts 0–9 (56 min);
- main blocks 996011 and 996012 on prompts 100–119 (2.97 h each);
- reader 996013, which wrote `stage1.{json,md}`.

Corpus 0a26bc1e. All 30 arms of a block ran in one process. The total was about
7.5 GPU-hours.

## B1. Verdicts (plan.md §5, amendment A1)

**`REENTER_STAGE0`: SIEVE is lossless at B = 4, and the 128K unit is `BYTES_TIE`.**

| point | score (FP 0.994) | lossless | evicted | bytes V16 / V8 |
|---|---:|---|---:|---:|
| SIEVE router, B = 2 | 0.429 | no | 64% | 125 / 79 |
| SIEVE router, B = 3 | 0.667 | no | 49% | 181 / 115 |
| **SIEVE router, B = 4 (S\*)** | **0.991** | yes | 34% | **235 / 151** |
| **TurboQuant-3 (D\*)** | 0.982 | yes | 0 | **306 / 178** |
| FP8 KV | 0.994 (Δ 0.000) | yes | 0 | 384 / 256 |

- ρ = 0.768 with BF16 values (WIN) and 0.845 with FP8 values (TIE). The two
  lenses disagree, so the frozen rule gives `BYTES_TIE`.
- FP8 KV, and FP8 values under every arm, are lossless. The largest FP8-value
  change is 0.009, and no V8 lens was withdrawn.
- **Checks:**
  - An independent recomputation from the raw rows reproduces the headline.
  - The FP8 arms changed 6–20% of the generated texts, so FP8 was applied, but
    changed at most 3 of 80 scores.
  - Cell means agree with R12's 128K cell within the cross-corpus spread.
    SIEVE router at B = 2 / 3: 0.43 / 0.67 against 0.51 / 0.72. TurboQuant:
    0.94 / 0.98 against 0.93 / 0.98.
  - Prefill took 19 s against R12's 170 s, because R12 also captured H2O
    scores.

## B2. Mechanism (analysis of the same rows, not gated)

- **Eviction deletes digits; quantization substitutes them.**
  - Dense arms (TurboQuant and KIVI-128, B = 2–4): 75 of 76 wrong needle values
    are substitutions.
  - SIEVE at B = 3: 48 deletions or length changes against 18 substitutions.
  - SIEVE at B = 4: no deletions.
- **Per-head output error cannot see this.**
  - In all 31 prompt-tasks where SIEVE-3 fails and TurboQuant-2 answers,
    TurboQuant-2 has the *larger* error on the per-head median, 99th percentile,
    maximum and answer-weighted error. On the most answer-attending heads, this
    holds in 30 of 31.
  - The same holds in every such pair at ≤ 32K in R12's rows: about 300 cases
    on three models.
- **SIEVE buys coverage, not precision.** Its kept tokens average 5.6, 5.9 and
  6.1 bits at B = 2, 3 and 4. A larger budget mostly decides how many tokens
  survive.
- **Routing.** At B = 3 the oracle router scores 0.936 against the calibrated
  router's 0.667.
  - The two differ on 5.9% of KV heads, but on about 20% of the top 1% of
    answer-attending heads.
  - On those heads the oracle more often keeps them dense.
- **The byte edge is value reads.** SIEVE-4 spends more key bits than
  TurboQuant-3. With values at 16 / 8 / 4 / 2 bits, the byte model gives ratios
  of 0.77 / 0.84 / 0.95 / 1.06, a break-even value width of 2.9 bits.

---

# Part C · Stage 1b — hybrids, value twins, pooled router, sequence proxy

Design: `s1b_lib.py`. Frozen rules: `read_stage1b.py` (amendment A2).
Llama-3.1-8B at 128K (prompts 4000–4019) and 32K (prompts 4100–4119), both on
fresh prompts. Calibration used prompts 0–9. Every arm of a block ran in one
process.

Jobs:
- pilot 999908 (excluded) and gate 999909 (pass, peak 51.9 GiB);
- calibrations 999910 and 999911 (29 and 10 min);
- 128K blocks 999912 and 999913 (2.4 and 2.1 h);
- 32K blocks 999914 and 999915 (0.74 h each);
- reader 999916, which wrote `stage1b.{json,md}`.

The total was about 7 GPU-hours. Validity checks V1–V4 pass, and the replay of
full precision agrees with its own answer on 0.995 / 0.997 of steps.

The arms:
- **dense:** TurboQuant at B = 2–4;
- **routers:** the SIEVE router and the pooled router at the top budget, one bit
  lower and at a half-bit budget, plus the pooled oracle as a diagnostic;
- **hybrids:** SIEVE's own eviction mask, or SnapKV's selection with the same
  per-head keep counts, with every kept token at one TurboQuant width;
- **value twins:** the same keys and eviction as the base arm, with values
  through TurboQuant-MSE at 4 or 2 bits.

Every arm was also replayed teacher-forced on the full-precision answer.

## C1. Frozen verdicts

| rule | 128K | 32K |
|---|---|---|
| D2 bytes (gating) | NO_POINT with exact or 4-bit values. WIN with 2-bit values (hybrid 66.6 against 84.3 bytes, ρ 0.79; not gating) | NO_POINT with any values |
| **decision** | **`STOP_SYSTEMS`** | |
| D1 hybrid vs SIEVE | NEITHER (exact, 4-bit); HYBRID_SUFFICES (2-bit) | NEITHER |
| D3 pooling | **`POOL_FIXES`**: +0.143 [+0.085, +0.203] at B = 3; deletions 10 → 0 | **`POOL_FIXES`**: +0.356 [+0.263, +0.450] at B = 2; deletions 32 → 14 |
| D4 proxy | **`SEQUENCE_PROXY`** (both cells pooled): AUC 0.903 for the worst answer token's log-probability, against 0.56 / 0.69 / 0.69 for per-head median / p99 / answer-weighted error, over 1,520 rows. Per cell: 0.906 and 0.896 | |

No calibrated router point (SIEVE or pooled) is lossless in any lens; the
pooled oracle, a diagnostic, is (C4). The lossless points are:
- **128K:** TurboQuant-4; TurboQuant-3 and -4 with 4-bit or 2-bit values; and
  the hybrid above;
- **32K:** TurboQuant-3 only.

Stage 0's Llama @32K win did not replicate. SIEVE-3 scores 0.972 here against
0.994 on R12's prompts.

## C2. The caveat: the frozen D2 was decided at the noise floor

- **Quantizing values alone is harmless.** The full-precision twins move the
  score by 0.000 to −0.003. The same change moves compressed arms by up to
  ±0.024, in both directions (median |Δ| 0.003), because their greedy decodes
  sit near a flip.
- **The decisive calls miss the −0.02 margin by a hair:**
  - 128K: SIEVE-4 Δ −0.022, TurboQuant-3 Δ −0.021;
  - 32K: TurboQuant-2 Δ −0.021, SIEVE-3 Δ −0.028.
- **Pooled with Stage 1 at 128K** (40 prompts; each Δ paired within its own
  run), both are lossless and Stage 1's `BYTES_TIE` stands:
  - SIEVE-4: Δ −0.013 [−0.034, 0.000];
  - TurboQuant-3: Δ −0.016 [−0.032, −0.003];
  - TurboQuant-4: Δ −0.005.
- **Cell means, Stage 1 → Stage 1b:** full precision 0.994 → 0.994,
  TurboQuant-3 0.982 → 0.974, SIEVE-4 0.991 → 0.972.

## C3. The continuous view (teacher-forced answer NLL; D4 validates it)

Byte ratio against dense TurboQuant-3 with the same value format, at roughly
matched mean increase in answer NLL. Brackets give the policy's NLL minus
TurboQuant-3's.

| cell | policy | exact values | 4-bit values | 2-bit values |
|---|---|---:|---:|---:|
| 128K | SIEVE-4 | 0.77 | 0.94 | 1.05 |
| 128K | hybrid: SIEVE-4 mask, 4-bit kept keys | 0.70 (+0.06) | 0.76 (+0.06) | 0.79 (+0.03) |
| 32K | SIEVE-3 | 0.65 | 0.76 (+0.03) | 0.83 (+0.07) |
| 32K | hybrid: SIEVE-3 mask, 3-bit kept keys | 0.58 (+0.18) | 0.59 (+0.24) | 0.59 (+0.31) |

- **SIEVE's own allocator loses its edge as values get cheaper,** as the byte
  model predicted (0.95 / 1.06 predicted, 0.94 / 1.05 observed). A hybrid that
  lowers the kept keys' precision keeps a stable ratio.
- **Risk profile at 128K.**
  - SIEVE-4's median increase in answer NLL is half TurboQuant-3's (0.027
    against 0.058 nats).
  - Its worst case is 4× larger (14.9 against 3.6 nats).
  - The worst 5% of prompt-tasks carry 76% of its total damage, against 33% for
    TurboQuant-3.
  - At 32K the tails are similar (SIEVE-3 max 3.5, TurboQuant-3 2.6).

## C4. Pooling, routing, widths, values

- **Pooling** changes which tokens are evicted, not how many.
  - 128K: +0.118 at B = 3.5 (deletions 6 → 0).
  - 32K: +0.297 at B = 2.5 (deletions 32 → 6).
  - No effect at the top budget: −0.001 at 128K, −0.013 at 32K.
- **Routing is the largest remaining gap.** After pooling, the calibrated router
  still trails its oracle:
  - 128K, B = 3: +0.91 nats [+0.16, +2.17], score −0.048;
  - 32K, B = 2: +2.34 nats [+1.20, +3.61], score −0.122.

  The oracle keeps about 1.6× as many KV heads dense (5.7% against 3.5%; 7.9%
  against 4.7%). The pooled oracle at B = 3 is lossless at 128K (0.982) at 186
  bytes, with the same answer NLL as TurboQuant-3 at 0.61× its bytes.
- **The widths matter, but second-order.** Keeping SIEVE's mask with one width
  for all kept keys costs likelihood but not task score:

  | cell | kept-key width | NLL cost | score change | bytes saved |
  |---|---|---|---|---:|
  | 128K | 4-bit | +0.087 nats [+0.044, +0.130] | −0.009, n.s. | 9% |
  | 128K | 3-bit | +0.35 nats | −0.015, n.s. | 14% |
  | 32K | 3-bit | +0.19 nats | −0.006, n.s. | 10% |

  SnapKV's selection at SIEVE's per-head keep counts is about as good as SIEVE's
  own mask. With 2-bit kept keys at 32K it deletes fewer digits (2 against 8).
- **Values.**
  - 4-bit values are nearly free: +0.02 to +0.06 nats.
  - 2-bit values cost about as much likelihood as 3-bit keys (+0.19 to +0.31
    nats) but no task score.
  - Dense TurboQuant-3 with 2-bit values is lossless in both cells, at 84–85
    bytes: 0.33× FP8 KV.

## C5. Expected and unexpected

**Expected:**
- pooling removes the deletions;
- the sequence proxy works;
- compressing the values erodes SIEVE's edge in the proportions the byte model
  predicted.

**Unexpected:**
- single runs of 20 prompts cannot resolve the −0.02 lossless margin;
- the mixed widths matter more than Stage 1 suggested;
- SIEVE's typical error is smaller than dense quantization's, and all of its
  damage sits in the tail;
- a re-balanced hybrid keeps a 0.6–0.8 byte ratio at every value width.

---

# Part D · Decision and requirements for the next step

**Recorded decision.**
- `STOP_SYSTEMS` stands as the pre-registered outcome of Stage 1b, with the
  caveat in C2.
- The kernel (plan.md §6–§7) is not built.
- Stage 0's `GO_KERNEL` (A5.1) is superseded.

**What the stages show, in one place.**
- **Unified in rate, not in risk.** Eviction trades a smaller typical error for
  a heavy tail. Which method wins depends on the quality metric as well as on
  the byte accounting:
  - mean likelihood and output error favour eviction-heavy policies;
  - exact-match tasks and the worst answer token favour dense quantization.
- **Span integrity comes first, then coverage, then precision.** Pooling fixes
  the failures at the same coverage. Kept-key precision is worth 0.09–0.35
  nats, not nothing.
- **Once spans are protected, the remaining gap is information:** which heads
  the question will need. The oracle has it; question-agnostic compression does
  not.
- **Values are the cheap axis.** Once they are compressed, the precision of kept
  keys becomes the dominant cost.

**The design ideas to measure next:**
1. **A value-aware hybrid.** Price a kept token as its key bits plus its value
   bits, which lowers kept-key precision as values get cheaper (C3).
2. **A sequence-calibrated pooled router.** Pool by default. Choose routes on
   the worst answer token's log-probability (D4) instead of mean per-head error,
   and keep a head dense if it is critical on any calibration prompt (C4).
3. **Dense storage with question-time reads.** Store every token
   dense-quantized, with 2–4-bit values, so nothing is erased. Select tokens
   only when reading, once the question is known. R12's question-visible SnapKV
   scoring 1.00 supports this; it is untested here.

**Requirements before any of these supports a claim.** The thresholds are to be
frozen in the next plan before any output.
- **At least 40 prompts per cell,** with every compared arm in the same process.
  At Stage 1b's rate this costs about 9 GPU-hours per 128K cell and about 3 per
  32K cell.
- **A continuous metric:** the teacher-forced increase in answer NLL over
  content tokens, paired against full precision with a prompt-bootstrap
  interval. "Matched quality" becomes an interval condition on this difference,
  not a binary lossless call.
- **A tail metric:** the share of prompt-tasks whose answer NLL rises by more
  than 2 nats, and the share of total damage carried by the worst 5%. Task
  score and the binary lossless rule stay as reported statistics.
