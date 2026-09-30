# R13 · The K-channel axis: report

Job 995550 (6 cells, array 0–5) · pilot 995548 · reader run locally
(`read_r13.py --main 995550`; reader job 995551 hit `NODE_FAIL`) ·
2026-09-27. Protocol: `plan.md`, with amendments A1–A5 (§6), all made before
any pilot or main output. Raw output: `main_995550.{json,txt}`, per-cell
parquet in `h0_measurement/results/r13_main_995550_*`.

**Status: `valid`.** All 6 cells are COMPLETE. All validity gates pass:

| gate | what it checks | result |
|---|---|---|
| V1 | the rotated uniform arm reproduces TurboQuant without norm correction | ≤ 1.1e-6 relative logit error |
| V2 | 8-bit per-channel key error | < 1% |
| V3 | every water-fill spends its budget | spend exact |
| V4 | L2 capture fidelity | worst 3.3e-3 |
| V5 | row count per cell | 13,824 to 41,472 head rows, all as expected |

All cells ran the current sources (checked against `SOURCES.sha256`). Each
cell took 12–16 minutes on one H100.

## 1. Verdicts (frozen decision table, plan.md §5)

| Q | question | verdict | numbers |
|---|---|---|---|
| Q1 | does the raw channel ladder have C1-dead rungs? | **C1_DEAD_CH**, but see §3: this is an artifact of query weighting | raw dead-2 = 14.6–18.0% in 6/6 cells |
| Q2 | is channel deadness driven by context length? | **MIXED** by the letter of the rule; in substance **flat** | channel dead-2 moves +0.7 / −2.0 pts (8K→32K); the token dead-2 control moves +6.2 / +7.1 |
| Q3 | does deployable channel allocation pay (raw basis, B=2)? | **PAYS** | `ch_xcal` 1.29–1.50× over per-channel uniform in 6/6 cells |
| Q4 | does the channel axis survive TurboQuant's rotation? | **CONSUMED** | `rc_xcal` 0.81–1.01× at B=3 in 6/6 cells |
| Q5 | is the channel axis orthogonal to the token axis? | **COUPLED**, weakly | Spearman ρ = +0.31 to +0.40 in 5/6 cells, +0.13 on Mistral |

## 2. What this means for §1 — the sentence to write

"Orthogonal and composable" is **not true as written for our quantizer.** The
supported statement is:

> We allocate along the token axis. Per-channel key treatment (KIVI, KVQuant)
> is a different axis, but our quantizer's random rotation already spends it.
> In the rotated basis, a channel allocation calibrated on one prompt and used
> on others is no better than uniform (0.81–1.01× at 3 bits). Per-channel
> methods are therefore an alternative treatment of outlier channels, not an
> addition to ours. In the unrotated basis, channel allocation does pay
> (1.3–1.5× at 2 bits). Its dead rungs, however, do not move with context
> length (≤2 points from 8K to 32K, against +6–7 points for the token axis's
> dead-2), so the order parameter C1 does not carry over to the channel axis.

No C1-style claim on a second axis is allowed. Plan §5 permits one only for
C1_DEAD_CH together with L_DRIVEN, and Q2 is not L_DRIVEN.

## 3. Findings

### 3.1 Q1/Q2: C1's mechanism is absent on the channel axis

The table gives dead fractions in %. `ch` is the raw basis, `rc` the rotated
basis, and `tok` the token axis (the fraction of heads with `sig2_b > 1`).

| cell | τ | ch d1 / d2 / d3 | rc d1 / d2 / d3 | tok d1 / d2 / d3 |
|---|---|---|---|---|
| llama31-8b @8K | 2.09 | 81.6 / 16.5 / 1.0 | 17.9 / 3.2 / 0.3 | 98.4 / 23.2 / 0.0 |
| llama31-8b @32K | 2.47 | 80.8 / 17.3 / 1.4 | 18.9 / 3.7 / 0.4 | 98.5 / 29.3 / 0.1 |
| qwen3-8b @8K | 2.51 | 88.2 / 18.0 / 1.0 | 31.7 / 10.7 / 2.2 | 99.8 / 49.8 / 9.6 |
| qwen3-8b @32K | 3.09 | 89.1 / 16.0 / 0.7 | 31.5 / 10.3 / 2.0 | 100.0 / 56.8 / 9.8 |
| mistral-7b @8K | 1.99 | 82.9 / 14.6 / 0.8 | 16.9 / 2.3 / 0.2 | 96.9 / 13.3 / 0.0 |
| qwen15-moe @8K | 2.42 | 82.4 / 16.9 / 1.4 | 22.6 / 6.6 / 1.4 | 94.2 / 30.9 / 0.3 |

- **Token dead-2 behaves like an order parameter.** It spans 13–57% across
  models, tracks τ, and rises with L: +6.2 pts for Llama (prompt SE about 3–4,
  so noisy) and +7.1 pts for Qwen3 (SE about 1, so clear). This is the
  positive control, and it moved.
- **Raw channel dead-2 does not.** It sits at 15–18% in every model regardless
  of τ, and moves +0.7 (Llama) and −2.0 (Qwen3) pts from 8K to 32K. The prompt
  SE is about 1 pt per cell, so both moves are within 0.5–1.6 SE of zero and
  have opposite signs.
  - The frozen rule calls this MIXED only because −2.0 misses the 1.5-pt
    "flat" bar.
  - That matches plan prediction P2: the channel axis has no saturating
    zero-rate action, so nothing couples deadness to L.
- **The "dead-2 = 15–18%" that triggers Q1 comes from query weighting, not from
  the quantizer.** Compare the query-free version of the same test (unweighted
  variance over all tokens):

  | basis | dead-1 | dead-2 | dead-3 |
  |---|---|---|---|
  | raw | 97–100% | **0.0%** in every cell | 0.0% |
  | rotated | 0–4.5% | ≤1.2% | ≤0.6% |

  At decode, attention weight sits on a handful of tokens. Measured over those
  few tokens, the mean-replacement error can come out smaller than the 2-bit
  rounding error. So the rung is dead for this query, not for the channel.
  P1 (only the 1-bit rung is dead, which is a min-max artifact) holds for the
  key statistics.
- **The allocator agrees:**
  - The 1-bit rung is unused on the raw basis (0.0–0.4% of channels), as
    expected for min-max 1-bit.
  - The 2-bit rung carries 6–11% of channels, so it is not dead in aggregate.
  - The top rung (8 bits) is essentially unused on both channel bases
    (≤1.5%).
  - On the token axis, the water-fill evicts 41–51% of tokens at B=3. Rung 1
    is used by ≤0.2% and rung 2 by ≤1.1%, which is the familiar C1 picture.

### 3.2 Q3: channel allocation pays in the raw basis, mostly at 2 bits

Median gain over own-family uniform (per-channel min-max, all channels at B
bits); each entry is B=2 / B=3 / B=4:

| cell | `ch_wf` (oracle) | `ch_cal` (same prompt) | `ch_xcal` (cross-prompt) | `ch_ks` (key stats only) |
|---|---|---|---|---|
| llama31-8b @8K | 2.53 / 2.06 / 1.93 | 1.35 / 1.01 / 0.97 | **1.30** / 1.01 / 1.02 | 0.33 / 0.56 / 0.95 |
| llama31-8b @32K | 2.59 / 2.07 / 1.96 | 1.44 / 1.06 / 1.02 | **1.29** / 1.04 / 1.06 | 0.40 / 0.85 / 1.00 |
| qwen3-8b @8K | 2.68 / 2.17 / 2.05 | 1.58 / 1.16 / 1.12 | **1.48** / 1.16 / 1.21 | 0.90 / 0.91 / 1.12 |
| qwen3-8b @32K | 2.65 / 2.29 / 2.20 | 1.57 / 1.22 / 1.20 | **1.50** / 1.28 / 1.40 | 1.18 / 1.15 / 1.27 |
| mistral-7b @8K | 2.62 / 2.13 / 2.06 | 1.60 / 1.21 / 1.17 | **1.49** / 1.16 / 1.19 | 0.49 / 0.76 / 0.99 |
| qwen15-moe @8K | 4.08 / 3.61 / 3.23 | 1.19 / 0.75 / 0.51 | **1.39** / 1.06 / 0.97 | 0.37 / 0.56 / 1.03 |

- **Calibrated allocation pays at B=2 and fades at B=3–4** (1.0–1.4×). There
  is real per-channel structure: the oracle gets 2–4×. About a third of it
  survives calibration on a different prompt.
- **Key statistics alone are harmful** (0.33× on Llama at B=2). Allocating by
  key variance hands bits to high-variance channels, but a channel's cost is
  `q_c² × variance`. The channels the queries weight heavily are not the
  high-variance ones, so the query has to enter the allocation. This last
  point is an inference from the cost form; it was not isolated.
- **`xcal` ≈ `cal`.** Calibrating on the same prompt is no better than
  calibrating on a disjoint one; on the MoE model it is worse. The channel
  allocation is a stable, prompt-independent property of a head, which is
  consistent with C4's "one calibration pass".

### 3.3 Q4: TurboQuant's rotation consumes the channel axis

Rotated-basis gains over rotated uniform (B=2 / B=3 / B=4):

| cell | `rc_wf` (oracle) | `rc_xcal` | `rc_ks` |
|---|---|---|---|
| llama31-8b @8K | 1.92 / 1.75 / 1.56 | 1.33 / **1.01** / 0.96 | 0.78 / 1.12 / 1.04 |
| llama31-8b @32K | 1.86 / 1.76 / 1.56 | 1.32 / **0.97** / 0.93 | 0.82 / 1.13 / 1.05 |
| qwen3-8b @8K | 1.49 / 1.56 / 1.55 | 0.93 / **0.84** / 0.90 | 1.00 / 1.00 / 1.00 |
| qwen3-8b @32K | 1.42 / 1.47 / 1.47 | 0.89 / **0.81** / 0.88 | 1.00 / 1.00 / 1.00 |
| mistral-7b @8K | 1.79 / 1.64 / 1.49 | 1.20 / **0.99** / 0.96 | 0.88 / 1.09 / 1.03 |
| qwen15-moe @8K | 2.42 / 2.47 / 2.26 | 1.11 / **0.82** / 0.70 | 0.83 / 1.09 / 1.04 |

- **Key statistics are nearly flat after rotation.** `rc_ks` is about 1.0, and
  exactly 1.00 on Qwen3, meaning the allocation is uniform. The rotation has
  equalised per-coordinate key variance, so the channel axis has nothing left
  to act on from the key side.
- **The oracle still finds 1.4–2.5×, but this comes from the current query.**
  The spread of `(Rq)_c²` across coordinates is what the oracle exploits, and
  that pattern does not transfer across prompts. `rc_xcal` at B=3 is
  0.81–1.01×, below uniform in 4/6 cells.
- **One nuance: at B=2, `rc_xcal` keeps 1.20–1.33× on Llama and Mistral** (the
  Qwen3 models get none). The frozen question is read at B=3, where it is gone.
  "Consumed" is accurate for the operating point, not for every bit-width.

### 3.4 Q5: weakly coupled, not independent

Across the (layer, KV head) units of a cell, the median log gain of channel
water-fill correlates with that of token water-fill:

- ρ = +0.31 to +0.40 on Llama, Qwen3 and Qwen1.5-MoE;
- ρ = +0.13 on Mistral.

This is COUPLED by the frozen bar (|ρ| < 0.3 in ≥4/6 cells), but only just.
The heads with dispersed error along one axis tend mildly to have it along the
other too. "Orthogonal" should not appear in §1 as an empirical claim.

### 3.5 Context, not a verdict: absolute errors across families

Median relative output error of each family's uniform arm (B=2 / B=3 / B=4, at
equal payload bits, metadata not counted):

| family | error |
|---|---|
| raw per-channel min-max (`ch_u`) | 0.31–0.36 / 0.12–0.14 / 0.05–0.07 |
| TurboQuant token-axis, norm-corrected (`t_u`) | 0.42–0.49 / 0.18–0.21 / 0.08–0.10 |
| rotated, no norm correction (`rc_u`) | 0.48–0.62 / 0.20–0.25 / 0.09–0.13 |

KIVI-style per-channel groups of 128 are better per payload bit at a
single decode step. Metadata is not counted:
- KIVI's per-group scale and zero point add about 0.25 bit/element;
- TurboQuant's per-token norm adds about 0.125.

No verdict above depends on this cross-family comparison. It is a
reminder that the paper's baseline question (R12, R14) should include a
per-channel quantizer, not a statement that one family wins.

For scale:
- the paper's token-axis oracle water-fill (`t_wf`) gets 12–48× over its own
  uniform;
- the channel oracle gets 2–4×.

The token axis, with eviction as tier 0, remains by far the larger lever. Both
of these are oracle numbers.

## 4. What was not established

- The channel test uses two context lengths (8K and 32K) on two models. A
  128K point would make Q2's "flat" more convincing; the token control moved
  clearly only on Qwen3.
- Keys are post-RoPE, as stored in the HF cache (KIVI's setting). KVQuant
  quantizes pre-RoPE keys, which have stronger channel structure. The raw-basis
  gains could be larger there.
- Only single-step attention-output error is measured, at decode steps 3, 5
  and 7. There is no end-task number (R8's regime problem applies here as
  well).
- The raw-basis tier 0 stores group means, and all channel families leave
  metadata out of B.

## 5. Run history

The chain reached a complete run on the third submission:

| chain | jobs | outcome |
|---|---|---|
| 1 | 995473–995476 | Cancelled. The pilot died on an `srun` controller timeout before Python started, and the code-review amendments A1–A5 were made at the same time. |
| 2 | 995534–995537 | Lost to a Trillium node incident (`prolog.chk.sssd.cache.expired`; 23 compute nodes draining). No R13 code ran. |
| 3 | 995548–995551 | Pilot, gate and all 6 main cells COMPLETED. The reader job hit `NODE_FAIL` and was run locally with identical code. |

## 6. Suggested follow-ups (not run)

- Replace the ROADMAP's R13 sentence with §2 and close R13.
- Add a KIVI/KVQuant per-channel arm to the paper's baseline table, with
  metadata counted, given §3.5.
- If the channel axis goes beyond one sentence in the paper, the natural
  follow-up is a 2-bit composition test: `rc_xcal` at B=2 on Llama and Mistral,
  stacked on the token tiers. That is the only place the rotated channel axis
  showed a deployable gain.
