# R14 · Kernel, iso-budget, TPOT — report

Protocol: `plan.md` for Stages 0 and 1 (amendment A1). For Stages 1b–1e, the
rules were frozen in the reader's docstring before any output of that stage
existed: `read_stage1b.py` (amendment A2), `read_stage1c.py`, `read_stage1d.py`
and `read_stage1e.py`; the kernel rule is in `s1e_kernel.py`.

## Decision (2026-10-03)

**`GO_KERNEL` stands (Stages 1c and 1d). Stage 1e built the kernel. It is correct
and its decode time scales with the read fraction, but the packed 3-bit decode is
slower than FlashAttention on 16-bit data. The speed comes from reading fewer
rows; the low-bit store buys memory capacity (Part G).**

- **Question-time reads hold on two models and both lengths.** Each question
  selects 1/8 of the rows once and the answer reads only those. With 4-bit values
  this is MATCHED against dense TurboQuant-3 at 13–14% of its read bytes on
  Llama-3.1-8B at 128K and Qwen3-30B-A3B at 32K and 128K. At Llama 32K the plain
  read is borderline (INCONCLUSIVE), and the read with the critical heads read
  in full is MATCHED at 22%.
- **Calibrated quant+evict routers save memory once their dense sets nest.**
  - Llama 32K: the nested router is MATCHED with no catastrophe, at ρ 0.69 (exact
    values) and 0.79 (4-bit values). This is the first router WIN at 32K under
    the answer-value metric.
  - Qwen 32K: SIEVE's router is better than TurboQuant-3 at ρ 0.65 / 0.76.
- **The kernel is the bottleneck.** The bit-plane decode reaches at most 22% of
  HBM bandwidth (4% with 4-bit values). FlashAttention on the compacted 16-bit
  rows is 3–7× faster than on all rows at 128K.
- **Pending:**
  - the Llama 128K tail cell (jobs 1018902–1018904, queued): Stage 1e's test of
    whether reads beat TurboQuant-3 at 128K, and nesting at B = 4;
  - two questions per context at 128K, which the frozen reader returned INVALID
    (FP scores 0.895 on vt).
- The 2026-09-29 decision below (`STOP_SYSTEMS`, Stage 1b) was superseded by
  Stage 1c's `GO_KERNEL` (Part E).

## Decision (2026-09-29, superseded)

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
| 1c: value-aware hybrid, sequence-calibrated router, question-time reads | 1005224–1005236 | `GO_KERNEL` (reads WIN at 128K with 4-bit values; 32K router WINs later retracted) | Part E |
| 1d: answer-value metric, second-round designs, setup-head test, Qwen | Rorqual 22113462–22113507 | `GO_KERNEL` (reads at both lengths; seq2 and top-32 routers at 128K) | Part F |
| 1e: exact-store reads, router tail, two questions, Qwen follow-ups, kernel | Rorqual 22224148–22224554; Trillium 1015405–1015430, 1018902–1018904 | kernel correct and `KERNEL_SCALES` but slower than FlashAttention; `TAIL_FIXED` and a router WIN at 32K; reads WIN on Qwen at 32K and 128K; `REUSE_MIXED`; 128K tail pending | Part G |

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

*Superseded: Stage 1c implemented these requirements (Part E). The current
decision is in Part H.*

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

---

# Part E · Stage 1c — the three designs of Part D

Design: `s1c_lib.py`. Frozen rules: `read_stage1c.py`. Llama-3.1-8B at 128K
(prompts 5000–5039) and 32K (5100–5139), 40 prompts per cell, calibration on
prompts 0–9. Jobs 1005224 (pilot) to 1005236 (reader), which wrote
`stage1c.{json,md}`.

Every design is compared with dense TurboQuant-3 (D) in the same value lens:
- **metric:** the teacher-forced NLL of FP's answer over all its content tokens;
- **MATCHED:** the 90% upper bound of the paired mean difference from D is ≤ 0.10
  nats, and the upper bound of the difference in > 2-nat tail share is ≤ 0.05;
- **verdict:** WIN if ρ ≤ 0.80, where ρ is bytes read relative to D.

## E1. Verdicts

**Decision: `GO_KERNEL`** (some family WINs with 4-bit values in at least one
cell).

| lens | 128K | 32K |
|---|---|---|
| exact values | WIN: SIEVE router at B = 4 (ρ 0.77), union router at B = 3 (ρ 0.71) | WIN: sequence-calibrated router (0.67), SIEVE (0.65), union (0.79), all at B = 3 |
| 4-bit values | WIN: question-time reads at r = 1/4 (ρ 0.25); TIE: SIEVE at B = 4 (0.94) | WIN: reads at r = 1/2 (0.51), sequence-calibrated router (0.78), SIEVE (0.76) |

- **The value-aware hybrid is WORSE everywhere.**
  - Its proxy kept 8-bit keys in every head (7.96–8.00 bits).
  - The fixed 3- and 4-bit versions are also WORSE: evicting the same share of
    every head fails.
- **Sequence calibration closes the pooled router's gap at the tight budget.**
  - Against the pooled router: −2.06 nats at 128K, B = 3, closing 1.03× of the
    gap to the oracle.
  - −2.57 nats at 32K, B = 2.5, closing 1.44× of it.
  - No effect at 128K, B = 4.
- **Question-time reads:** their cost against D is +0.05 nats at r = 1/4 (128K)
  and +0.15 at 32K.

## E2. What Stage 1d corrected

- **The metric also scored FP's text after the answer.** On vt that is up to 64
  copied tokens. The answer-value metric of Stage 1d removed it, and the 32K
  router WINs did not survive (Part F).
- **The hybrid's label `LOWERS_PRECISION` is an artifact.** The chosen widths
  fell by only 0.01–0.03 bits.

---

# Part F · Stage 1d — the answer-value metric, second-round designs, the setup-head test, Qwen

Design: `s1d_lib.py`. Frozen rules: `read_stage1d.py`. It ran on Rorqual (jobs
22113462–22113507):
- Llama-3.1-8B at 128K (prompts 7000–7039) and 32K (7100–7139);
- Qwen3-30B-A3B-Instruct-2507 at 32K (7100–7139), calibrated at 8K and 32K.

The metric is dA: the teacher-forced NLL of the answer-value tokens of FP's
answer, up to the end of the answer span. The MATCHED rule is unchanged.
Validity checks V1–V7 pass, and FP's replay agrees with its own answer on ≥ 99.6%
of steps.

## F1. Results at a glance

| cell | D vs FP | WINs (cheapest MATCHED point) | routers |
|---|---|---|---|
| Llama 128K | +0.26 | reads at r = 1/8, ρ 0.13, **−0.040 [−0.074, −0.008]** vs D (4-bit values) | seq2 ρ 0.74, top-32 ρ 0.75 (4-bit values) |
| Llama 32K | +0.11 | reads with protected heads at r = 1/8, ρ 0.22, −0.004 (4-bit values) | WORSE or INCONCLUSIVE |
| Qwen 32K | +0.03 | reads at r = 1/4, ρ 0.25, +0.002 (exact values); SIEVE router at B = 3, ρ 0.65 | seq2 WORSE at B = 2.5 |

**Decision: `GO_KERNEL`.** The 4-bit-value WINs are the reads at both lengths,
and seq2 and top-32 at 128K.

## F2. Findings

- **Setup heads vs answer-time heads depends on the task.** Each cell below gives
  the share of the full rescue recovered by making the critical heads dense only
  during the question (f_q) or only during the answer (f_a):

  | task | Llama 128K, B = 3 | Llama 32K, B = 2.5 | Qwen 32K, B = 2.5 |
  |---|---|---|---|
  | multivalue | 0.01 / 1.00 | 0.01 / 0.97 | — |
  | vt | 0.01 / 1.00 | −0.07 / 0.98 | 0.19 / 0.92 |
  | multikey | 0.66 / 0.64 | 0.66 / 0.48 | 0.87 / 0.68 |
  | single | 1.00 / −0.17 | 0.31 / 0.66 | 0.80 / 0.32 |

  Patching the critical heads' question-time outputs from FP reproduces f_q in
  every cell.
- **The critical heads are key readers,** with 2.7–3.8× the key-term attention
  of other heads, and come in two kinds:
  - early heads with no answer-time attention to the value: (2,5), (5,2), (10,0);
  - mid and late retrieval heads that read both key and value: (27,5), (24,6),
    (13,1).

  The strongest value-copying heads are not critical. A critical head is a blind
  spot of question-agnostic scoring, not the model's most important head.
- **At 128K, reads gain as they read less.** Against D: −0.016, −0.027 and
  −0.040 nats at r = 1/2, 1/4 and 1/8. At 32K the trend reverses unless the
  critical heads are read in full, which improves dA by 0.067 nats. Re-selecting
  rows during the answer has no effect.
- **Routers win at 128K and not at 32K** because D's own damage grows with
  length (0.110 → 0.258 nats), not because eviction gets easier.
- **The router tail.**
  - At B ≥ 3, every catastrophic deletion (dA more than 3 nats above D) was on
    multikey, at a budget whose calibration found ≤ 3 critical heads.
  - The dense sets did not nest: seq2 at B = 4 (128K) kept only 4 of B = 3's 12
    critical heads dense.
- **Kept-row width is second-order:** ±0.06–0.35 nats, against 1.2–3.3 nats for
  protecting the lookup heads.
- **Qwen.**
  - The critical set is tiny (3 heads at B = 2.5, none at 3) and changes with
    length: the 8K and 32K sets share 1 of 3.
  - Sequence calibration beats the per-head-error oracle (gap closed 1.9×).
  - FP's multivalue score of 0 was a harness artifact: Qwen's merged ':\n\n'
    first token tripped run_r8's stop rule.
- **The metric fix.** Post-answer continuation had been 8% (128K), 38% (32K) and
  71% (Qwen) of D's damage under Stage 1c's metric. Removing it retracts Stage
  1c's 32K router WINs.

## F3. Implications as drawn after Stage 1d (revisited in G9)

- **Design:**
  - Make question-time reads the lead design. Re-selection does nothing, so the
    kernel is "compact once per question, then decode densely".
  - Protect the critical heads in the reads.
  - Make router dense sets nest across budgets.
  - Treat kept-row width as a second-order knob.
- **Understanding:**
  - *When* rows are chosen (question-agnostic vs question-aware) matters more
    than how bits are split.
  - Lookup heads explain the task-dependent timing.
  - Quantization damage grows with length while eviction damage is rare
    deletions, so the best quant/evict mix depends on length.
  - Criticality is not importance.
- **Selling:**
  - Lead with the reads as a time-per-token claim.
  - Make the memory claim at 128K only.
  - Tell the lookup-head story.
  - Retire the 32K router claims.

---

# Part G · Stage 1e — five follow-ups and the kernel

Design: `s1e_lib.py` (E1–E4) and `s1e_kernel.py` (E5). Frozen rules:
`read_stage1e.py`; the kernel rule is in `s1e_kernel.py`. All prompts are fresh
(the 8000 range).

Jobs:
- **Rorqual:**
  - Qwen 22224148–22224160;
  - kernel 22224224;
  - tail 22224541–22224554: pilot, both calibrations, the 128K block on prompts
    8100–8109, all four 32K blocks and both regression blocks.
- **Trillium:**
  - two questions per context: 1015423–1015430;
  - the three missing 128K tail blocks: 1018902–1018904, queued.

**Read on 2026-10-03:**
- both Qwen cells;
- the Llama 32K tail cell, with its regression block and calibration 22224544;
- two questions per context at 32K;
- the kernel.

**Not read:**
- the 128K tail cell (one block of four);
- two questions per context at 128K, which the frozen reader returned INVALID
  (FP scores 0.895 on vt, below 0.9).

**Validity:**
- All four reads pass V1–V7.
- FP's replay agrees with its own answer on ≥ 99.2% of steps, and the
  answer-value mask covers every correct FP answer.
- Routes files were matched across clusters by sha256 (V5 amended 2026-10-03 for
  file identity only).
- The kernel matches the float64 reference within 2.3e-3 of the largest output.

## G1. Results at a glance

| question | cell | frozen verdict | main numbers |
|---|---|---|---|
| E1: reads over an exact store | Llama 32K | `NO_GAIN` (r = 1/8 and 1/4) | selection costs the same over both stores: −0.074 vs −0.075 nats, interaction +0.001 |
| | Qwen 32K | `NO_GAIN` (1/8), `QUANT_NOISE` (1/4) | interaction +0.016 [+0.001, +0.036] at 1/4 |
| | Qwen 128K | `NO_GAIN` | reads are 0.02–0.03 nats worse than D |
| | Llama 128K | pending | — |
| E2: router tail | Llama 32K | `TAIL_FIXED`; router WIN at 32K | nested router MATCHED at ρ 0.69 (exact values) / 0.79 (4-bit values), 0 catastrophes; `NEST_HELPS` −0.11, `CAL_NO_EFFECT` |
| | Llama 128K | pending | — |
| E3: two questions per context | Llama 32K | `REUSE_MIXED` | on the second question, SnapKV-with-question minus reads = +0.49 nats at r = 1/8, all of it on vt (+0.99) |
| | Llama 128K | INVALID (FP vt 0.895) | — |
| E4: Qwen | 32K | `STOP_FIX` WORKS; reads WIN with 4-bit values (ρ 0.14); SIEVE router WIN (ρ 0.65 / 0.76); `QWEN_TAIL` `TAIL_NOT_FIXED` | FP 1.000 on all four tasks |
| | 128K | reads WIN with 4-bit values (ρ 0.13) | — |
| E5: kernel | H100 | `CORRECT`; `KERNEL_SCALES` (5.2× at r = 1/8, 3.3× at 1/4) | but 3–5× slower than FlashAttention on 16-bit K/V |

## G2. E1 — the cost of reading fewer rows

A positive gain means the sparse read is better. G_q compares reads over the
3-bit store with D; G_fp compares reads over the exact store with FP; Q = G_q −
G_fp is their interaction.

| cell | r | G_q | G_fp | Q |
|---|---|---|---|---|
| Llama 32K | 1/8 | −0.074 [−0.100, −0.050] | −0.075 [−0.094, −0.057] | +0.001 [−0.019, +0.021] |
| Llama 32K | 1/4 | −0.036 [−0.052, −0.022] | −0.038 [−0.049, −0.028] | +0.002 [−0.009, +0.014] |
| Qwen 32K | 1/8 | −0.008 [−0.037, +0.020] | −0.005 [−0.011, +0.000] | −0.003 [−0.032, +0.026] |
| Qwen 32K | 1/4 | +0.015 [+0.001, +0.034] | −0.001 [−0.005, +0.002] | +0.016 [+0.001, +0.036] |
| Qwen 128K | 1/8 | −0.027 [−0.052, −0.003] | −0.046 [−0.083, −0.016] | +0.019 [−0.016, +0.060] |
| Qwen 128K | 1/4 | −0.017 [−0.031, −0.006] | −0.005 [−0.014, +0.004] | −0.013 [−0.025, +0.000] |

- **The two damages add.** Selecting rows costs the same whether the store is
  quantized or exact; the interaction is within ±0.02 nats in every cell read.
- **No dilution in FP here.** Reading fewer exact rows never beats FP
  (G_fp ≤ 0), so none of these cells shows the attention dilution that FP would
  also suffer.
- **The selection cost is multi-hop.** It sits mostly in vt: −0.205 of G_fp's
  −0.075 average (Llama 32K, r = 1/8); −0.172 (Qwen 128K).
- **At equal read bytes, a few exact rows beat more 3-bit rows:**

  | cell | exact keys, r = 1/8 | 3-bit keys, r = 1/4 |
  |---|---|---|
  | Llama 32K | 65.5 bytes, +0.075 vs FP | 78.0 bytes, +0.116 |
  | Qwen 32K | 65.7 bytes, +0.005 | 78.2 bytes, +0.112 |
  | Qwen 128K | 64.5 bytes, +0.046 | 77.0 bytes, +0.084 |

  On Qwen at 32K, reading 1/8 of the exact rows is as good as FP. The exact store
  costs 1.67× D's memory.
- **The open test** is the Llama 128K cell, where Stage 1d saw reads beat D.

## G3. E2 — the router tail (Llama 32K, B = 3)

| router | catastrophes | label (exact values) | ρ |
|---|---:|---|---|
| seq2: Stage 1d's routes | 2 | INCONCLUSIVE | 0.67 |
| seq3: the new calibration | 2 | INCONCLUSIVE | 0.67 |
| nest2: Stage 1d's, nested | 0 | MATCHED | 0.69 |
| nest3: new, nested | 0 | MATCHED | 0.69; 0.79 with 4-bit values (WIN) |

- **Nesting is the fix; the data is not.** Nesting effect −0.113 [−0.194,
  −0.043] nats (`NEST_HELPS`); calibration effect +0.006 [−0.043, +0.055]
  (`CAL_NO_EFFECT`).
- **The known failures are fixed.** Stage 1d's routes still fail prompt 7112
  (+3.79 nats over D); the full fix is within 0.06 nats on all three regression
  prompt-tasks.
- **The catastrophes were on lookup tasks:** multivalue (prompts 8201, 8218) and
  multikey (8237) this time.
- **Why the larger calibration did not help.**
  - At B = 3 it found 4 failures in 70 prompt-tasks and added 2 heads.
  - At B = 2.5 it found 29 failures (multikey 17, multivalue 6, single 4, vt 2)
    and added 26 heads.
  - Nesting carries the tight budget's information up.
- **Protection has redundancy beyond a stable core.**
  - The two calibrations (prompts 0–9 and 8000–8039) share 8 of their 22 / 26
    critical heads at B = 2.5. These include the early key readers (2,5), (4,4),
    (5,2) and (10,0) seen in Stages 1c and 1d.
  - Their nested sets overlap in only 9 of 24 / 26 heads, yet both remove every
    catastrophe.
- **Cost:** nesting adds 16 dense heads at B = 3 (35 → 51), +3% bytes.

## G4. E3 — two questions per stored context (Llama 32K)

| arm (exact values) | first question vs D | second question vs D | stored bytes (ρ mem) |
|---|---|---|---|
| reads, r = 1/8 | +0.038, INCONCLUSIVE | +0.028, MATCHED | 1.00 |
| SnapKV-with-question, r = 1/8 | identical to reads | +0.523 [+0.295, +0.783], WORSE | 0.13 |
| reads, r = 1/4 | −0.007, MATCHED | −0.012, MATCHED | 1.00 |
| SnapKV-with-question, r = 1/4 | identical to reads | +0.129 [+0.044, +0.230], INCONCLUSIVE | 0.25 |

With 4-bit values at r = 1/8 the picture is the same: reads +0.030 (MATCHED)
and SnapKV +0.440 (WORSE) on the second question.

- **The reuse loss is task-asymmetric.** On the second question, SnapKV minus
  reads is +0.49 [+0.27, +0.75] nats at r = 1/8: vt +0.99 [+0.62, +1.42],
  multikey −0.00.
  - A selection made for the multikey question drops vt's chain links.
  - A selection made for vt keeps the multikey needle.
- **The frozen label is `REUSE_MIXED` because of the rule, not reuse.**
  - The reads' first question at r = 1/8 is INCONCLUSIVE from one prompt (8401,
    vt: 2.46 nats against D's 1.21).
  - With 40 units per role, a single > 2-nat event fails the tail test.
  - The first question is the same computation for both arms, so it carries no
    information about reuse.
- **At 128K** FP itself scores 0.83 on vt as the second question, and the cell is
  INVALID.

## G5. E4 — Qwen3-30B-A3B

- **The stop fix works.** FP scores 1.000 on all four tasks at 32K (multivalue
  was 0.0 in Stage 1d).
- **Reads WIN with 4-bit values at both lengths:** at r = 1/8, −0.001 vs D at
  32K (ρ 0.14) and +0.021 at 128K (ρ 0.13).
- **SIEVE's router (unpooled) beats D at B = 3:** −0.057 [−0.102, −0.018] with
  exact values (ρ 0.65) and −0.070 with 4-bit values (ρ 0.76).
- **The pooled routers fail on multivalue** (`QWEN_TAIL` `TAIL_NOT_FIXED`).
  - Stage 1d's routes, flat or nested with its 3 heads, each have one multivalue
    catastrophe (prompt 8218, 6.4–8.2 nats).
  - Stage 1d's Qwen calibration contained no multivalue prompt-tasks, because FP
    failed them then. Its critical set cannot cover multivalue's lookups.
  - Prompt 8218's multivalue also breaks Llama's un-nested routers (both models
    read prompts 8200–8239), so it is a structurally hard prompt.
- **Quantization can confuse near-duplicate keys.**
  - On prompt 8234 (multikey) FP answers 9203285. TurboQuant-3 and TurboQuant-4
    answer a distractor's 7299132 (9.75 and 6.25 nats).
  - Every 3-bit arm fails there; reads over the exact store answer correctly.
- **D's damage is flat with length on Qwen,** unlike Llama: +0.127 at 32K
  (+0.066 without prompt 8234) and +0.066 at 128K.

## G6. E5 — the kernel

- **Correct and scaling.** Maximum error 2.3e-3 against the float64 reference.
  `KERNEL_SCALES`: the compacted read is 5.16× faster at r = 1/8 and 3.27× at
  r = 1/4 (128K, batch 1, 4-bit values, Llama shapes).
- **Slow in absolute terms.** Per layer, Llama-3.1-8B at 128K:

  | read | batch 1 | batch 16 |
  |---|---:|---:|
  | FlashAttention, 16-bit K/V, all rows | 0.199 ms | 2.69 ms |
  | FlashAttention, 16-bit, compacted r = 1/8 | 0.049 | 0.362 |
  | packed: 3-bit keys + 16-bit values, all rows | 0.609 | 7.67 |
  | packed: 3-bit keys + 16-bit values, r = 1/8 | 0.125 | 1.02 |
  | packed: 3-bit keys + 4-bit values, all rows (= D) | 1.010 | 14.27 |
  | packed: 3-bit keys + 4-bit values, r = 1/8 | 0.196 | 1.86 |
  | unfused torch merge, added per call | +0.14–0.37 | +0.03–0.16 |

- **Compute-bound.** The packed kernel reaches at most 22% of HBM bandwidth with
  16-bit values and 4% with 4-bit values, against the measured 3.04 TB/s;
  FlashAttention reaches 61–105%.
  - Unpacking the bit-planes and looking up the levels dominate.
  - At batch 16 the 4-bit-value read at r = 1/8 runs 23× slower than its byte
    roofline (0.08 ms).
- **Sparsity alone runs at full speed:** FlashAttention on compacted 16-bit rows
  at r = 1/8 is 4.1× (batch 1) to 7.4× (batch 16) faster than on all rows for
  Llama at 128K, and 3.0× to 7.0× for Qwen.
- **Capacity forces the low-bit store.** At batch 16 × 128K on Llama-8B:

  | format | memory | fits one H100? |
  |---|---:|---|
  | 16-bit K/V | 275 GB | no |
  | 3-bit keys + 16-bit values | 164 GB | no |
  | 3-bit keys + 4-bit values | 62 GB | yes |

- **Per-question cost** (per layer, 128K, batch 1): top-k selection 0.30 ms,
  compaction 0.08 ms.
- **Rough time per token at batch 1, 128K** (weights 5.3 ms at 3.04 TB/s):

  | read | time per token |
  |---|---:|
  | FlashAttention, 16-bit, all rows | 11.7 ms |
  | FlashAttention, compacted 16-bit rows | 6.9 ms |
  | packed 4-bit values, r = 1/8 | 11.6 ms |
  | packed D | 37.6 ms |

## G7. Expected and unexpected

**Expected:**
- nesting removes the tail;
- the stop fix recovers Qwen's multivalue;
- reads generalize to Qwen, to 4-bit values and to 128K;
- no read gain at 32K;
- the kernel is correct and scales with r.

**Unexpected:**
- the larger, multikey-weighted calibration does nothing on its own;
- a router WIN at 32K;
- a few exact rows beat more quantized rows at equal bytes, and on Qwen 1/8 of
  the exact rows is as good as FP;
- quantization makes a catastrophic key confusion on Qwen, even at 4 bits;
- reads do not beat D on Qwen at 128K;
- the reuse loss falls only on multi-hop second questions;
- the packed decode is compute-bound and slower than FlashAttention on 16-bit
  data.

## G8. Corrections to the instruments

- **Kernel report, bandwidth column.** "% read BW" used half the copy rate as its
  reference. The right reference is the full 3.04 TB/s (FlashAttention reads at
  3.2 TB/s): halve that column. The weights-per-token line is also 2× too high
  (Llama 5.3 ms, Qwen 2.2 ms).
- **Kernel rule.** It compared the kernel only with itself. The next kernel rule
  needs an absolute bar: FlashAttention on the same rows.
- **Reuse label.** It includes the first question, which is identical for both
  arms, and its 40-unit tail test fails on one event. It needs a
  second-question-only label and ≥ 80 units per role.

## G9. Implications, on top of Stages 1–1d

**Design.**
1. **Nest by default; calibrate tight, deploy loose.**
   - The 2 × 2 shows the tail is fixed by nesting, not by more calibration data.
   - A tight budget is a stress test: there, failures are common and expose the
     lookup heads (29 of 70 at B = 2.5 against 4 of 70 at B = 3). Nesting
     carries those heads to the looser budgets for +3% bytes.
   - The calibration must cover every task's lookup schedule. Qwen's lacked
     multivalue, and nesting could not help.
2. **When the question is known, spend bytes on precision, not coverage.** Exact
   rows at r = 1/8 beat 3-bit rows at r = 1/4 on both models. This suggests a
   two-tier store:
   - a low-bit tier (3-bit keys, 4-bit values) resident for capacity and for the
     question-time vote;
   - exact (16-bit or FP8) rows for the selected r C rows, fetched or compacted
     once per question (from host memory or an 8-bit tier).

   Because selection happens once per question, the transfer is per question,
   not per decode step.
3. **The low-bit decode needs real kernel work before it can save time:**
   - nibble-aligned layouts (4-bit keys, or a 2 + 1 split) instead of three
     bit-planes;
   - dequantization through register or shared-memory tables, feeding
     tensor-core MMA;
   - a fused split merge;
   - or FP8 keys on the read path.

   Until then, the time win runs on a 16-bit or FP8 read format and the low-bit
   store is a capacity device.
4. **Multi-hop questions are where selection loses** (vt in every cell). Give
   chain-like questions a larger read floor or protected heads.
   - Protection is model-specific: it helped Llama at 32K and hurt Qwen once.
   - On Qwen, the unpooled SIEVE router beats D, while the pooled routers fail.

**Understanding the unified problem.**
1. **Quality loss ≈ quantization loss + selection loss.** The interaction is
   ≤ 0.02 nats wherever it was measured. "Reads beat D" (Stage 1d, Llama 128K) is
   the only candidate for a real interaction, and it did not appear on Qwen at
   128K.
2. **The tail is not eviction-only.**
   - Quantization can flip a discrimination between near-duplicate keys (Qwen
     prompt 8234, even at 4 bits); eviction deletes.
   - Exact-row reads avoid both.
   - D's length scaling is model-specific: Llama's damage doubles from 32K to
     128K, Qwen's stays flat.
3. **The timing of information decides where bits go.**
   - Without the question, coverage and lookup-head protection come first.
     This agrees with Stage 1 (SIEVE buys coverage), Stage 1b (span integrity,
     then coverage, then precision) and Stage 1d (width is second-order).
   - With the question, precision comes first (Stage 1e).
4. **Criticality belongs to the model, router, task mix and budget together.**
   - A stable core of early key readers recurs across three calibrations; beyond
     it, different sets suffice.
   - The reuse loss follows the second question's lookup schedule.
5. **Bytes are an upper bound on speed, not a forecast.** At ≤ 4 bits a naive
   decode is compute-bound. The byte ratios ρ of Stages 0–1e are achievable only
   with a near-roofline kernel.

**Selling.**
1. **Reframe the speed claim.**
   - "Select once per question and read 1/8 of the rows" is the speed mechanism
     (3–7× attention speedup at 128K on a fast format).
   - Its quality now holds on two models and both lengths: MATCHED vs
     TurboQuant-3 at ρ 0.13–0.14 (Llama 128K, Qwen 32K and 128K), and at ρ 0.22
     with the critical heads read in full (Llama 32K).
   - The low-bit store is the capacity mechanism (62 GB against 275 GB at batch
     16 × 128K).
   - Say plainly that combining the two needs a roofline low-bit kernel.
2. **Do not claim that reads beat TurboQuant-3** until the Llama 128K cell is
   read.
3. **The memory claim now covers both lengths with nesting:** Llama 32K at
   ρ 0.69 / 0.79 with no catastrophe, and SIEVE on Qwen 32K better than D at
   ρ 0.65 / 0.76. The nested router at 128K is pending; Stage 1d's un-nested
   seq2 was MATCHED there at ρ 0.74.
4. **A method contribution:** nesting, with a 2 × 2 showing it, not calibration
   data, removes the tail.
5. **Tell reuse with multi-hop questions.** SnapKV-with-question loses about 1
   nat on a second multi-hop question while reads match TurboQuant-3. Quote its
   8× memory saving as the trade-off.
6. **The two-tier exact-row read is the forward design:** on Qwen, exact reads at
   1/8 are within 0.005 nats of FP.
7. **Avoid two claims:** that the packed 3-bit decode is fast, and that
   quantization is safe while only eviction is risky.

---

# Part H · Decision and next steps (2026-10-03)

**Decision.** `GO_KERNEL` stands. The kernel exists and is correct, but its
low-bit decode does not yet save time against FlashAttention on 16-bit data.

| claim | status |
|---|---|
| question-time reads at r = 1/8 match TurboQuant-3 at 13–14% of its read bytes (4-bit values) | holds: Llama 128K (Stage 1d), Qwen 32K and 128K (Stage 1e); Llama 32K needs the critical heads read in full (ρ 0.22, Stages 1d and 1e) |
| reading fewer rows speeds up decode attention | holds on a 16-bit format: 3–7× at 128K |
| the packed low-bit decode saves time | not yet: compute-bound, 3–5× slower than FlashAttention |
| calibrated quant+evict routers save memory at matched quality | holds with nesting: Llama 32K ρ 0.69 / 0.79; Qwen 32K SIEVE ρ 0.65 / 0.76; Llama 128K pending |
| reads beat TurboQuant-3 at 128K | open: Llama 128K pending; not on Qwen |
| reads keep a stored context reusable where SnapKV-with-question does not | shown at 32K for multi-hop second questions; frozen label `REUSE_MIXED`; 128K invalid |

**Next steps.**
1. **Read the 128K tail cell** when jobs 1018902–1018904 finish. It settles E1
   at 128K and tests nesting at B = 4. Its blocks span two clusters, and the
   reader matches routes by hash.
2. **Kernel v2.**
   - Nibble or 2 + 1 layout, table-driven dequantization with tensor-core MMA,
     and a fused merge.
   - Target: at least half of HBM bandwidth.
   - Add an absolute rule against FlashAttention on the same rows.
3. **A two-tier exact-row read.** Quality is already measured (exact reads at
   1/8); measure the per-question fetch cost from host memory or an FP8 tier.
4. **Reuse, again.**
   - ≥ 80 units per role and a second-question-only label.
   - A pre-registered rule for prompt-questions FP fails.
   - The nested router as the question-agnostic arm.
5. **Qwen.** Recalibrate with multivalue (now valid), nest, and test the unpooled
   SIEVE router with nesting.
6. **A calibration-free shortcut.** Test whether protecting the early key readers
   plus nesting is enough.
