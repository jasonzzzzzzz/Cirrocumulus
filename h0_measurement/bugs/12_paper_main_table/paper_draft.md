<!--
DRAFT, 2026-09-25. Numbers are from the R9 held-out campaign (jobs 978480,
978483, 978485, 978487, 978489; prompts 100-119), read by
bugs/12_paper_main_table/read_main_table.py. The paired rerun with H2O, KIVI and
KVQuant (jobs 991902-991906) and the non-ceiling cell (991907) will REPLACE
every number below, because the forward pass is not bit-reproducible across
runs. Items marked [PENDING jobid] have no data yet.
Build: python build_draft.py  (substitutes the appendix grid).
-->

# 5 End-task evaluation

## 5.1 Setup

We ask whether the allocation that minimizes attention-output distortion also
preserves what a model *retrieves*. We compress a cache once, after the context
has been prefilled and before the question arrives, as a prefix cache or a
multi-turn session would. Compressing after the question lets every
observation-window method (SnapKV and its descendants) read the question and
score 1.00 at every budget, so it cannot separate methods.

**Models and tasks.** Llama-3.1-8B-Instruct at 8K, 32K and 128K tokens, and
Qwen3-8B at 8K and 32K. The two models share size and GQA ratio (4) and lie at
opposite ends of our phase map (§4). Four RULER-style tasks run on a PG-19
haystack: single-needle and multi-key retrieval, multi-value retrieval, and
variable tracking. There are 20 held-out prompts per cell, disjoint from the 10
prompts used to calibrate the router. We exclude a (model, context, task) block
when the uncompressed model scores below 0.9. This removes only Qwen3-8B
multi-value (0.24 at 8K, 0.04 at 32K) and leaves 36 model/context/task/budget
cells.

**Matched budgets.** Every method spends B ∈ {2, 3} code bits per context key
element, audited at run time. Values stay in full precision, and the last 32
context tokens and all generated tokens are uncompressed in every method.
Eviction methods keep ⌊B·C/8⌋ tokens at 8 bits. Side information (norms,
scales, outlier indices, width maps) differs by method and is reported next to
the code budget in Table 1 (Appendix B).

**Baselines.** Quantization: TurboQuant-MSE (random rotation + Lloyd-Max +
per-token norm), KIVI [PENDING 991902-6] and KVQuant [PENDING 991902-6].
Eviction: H2O [PENDING 991902-6], SnapKV, DropKV, Ada-KV, LaProx, and
OBCache-K with Ada-KV allocation. All eviction scores are pooled over the KV
group, as a GQA cache requires. Implementation details and our deviations from
each paper are in Appendix B.

## 5.2 Main results

**Table 1.** Mean task score over the 36 valid cells (equal weight per cell).
Head-output error is the relative attention-output error per query head,
averaged over the first eight answer tokens. Effective bits = code bits + side
information.

| family | method | mean | Llama-3.1-8B | Qwen3-8B | B = 2 | B = 3 | head-output error ↓ | tokens evicted | effective bits (B=2 / 3) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| — | Full precision | 0.999 | | | | | — | 0% | 16 |
| quantization | TurboQuant-MSE | **0.946** | **0.985** | 0.867 | **0.898** | **0.993** | 0.445 | 0% | 2.12 / 3.12 |
| quantization | KIVI (g = 128) | [PENDING] | | | | | | 0% | 2.25 / 3.25 |
| quantization | KVQuant | [PENDING] | | | | | | 0% | ≈2.32 / 3.32 |
| eviction | H2O | [PENDING] | | | | | | 69% | 2.04 / 3.05 |
| eviction | SnapKV | 0.398 | 0.479 | 0.237 | 0.300 | 0.497 | 0.261 | 69% | 2.04 / 3.05 |
| eviction | DropKV | 0.418 | 0.459 | 0.334 | 0.316 | 0.519 | 0.271 | 69% | 2.04 / 3.05 |
| eviction | Ada-KV | 0.503 | 0.612 | 0.285 | 0.381 | 0.625 | 0.258 | 69% | 2.04 / 3.05 |
| eviction | LaProx | 0.579 | 0.640 | 0.457 | 0.468 | 0.690 | 0.264 | 69% | 2.04 / 3.05 |
| eviction | OBCache-K + Ada-KV | 0.602 | 0.719 | 0.369 | 0.509 | 0.696 | 0.260 | 69% | 2.04 / 3.05 |
| mixed | SIEVE interior only | 0.465 | 0.479 | 0.438 | 0.343 | 0.587 | 0.133 | 55% | 2.07 / 3.09 |
| mixed | **SIEVE** | 0.766 | 0.654 | **0.989** | 0.613 | 0.919 | **0.120** | 51% | 2.07 / 3.10 |

Three results stand out.

1. **SIEVE is the strongest token-selective method.** It averages 0.766
   against 0.602 for the best eviction baseline (OBCache-K + Ada-KV), a gain of
   +0.16 (90% CI [+0.13, +0.20], prompt bootstrap). It beats every eviction
   baseline on average (Table 2) and has less than half their attention-output
   error.
2. **On Qwen3-8B, SIEVE beats every method, including dense quantization.**
   It averages 0.989 against 0.867 for TurboQuant. The gap is largest where
   uniform 2-bit keys break down: variable tracking at B = 2 scores 0.99 vs
   0.46 at 8K and 0.99 vs 0.40 at 32K, and multi-key at 32K scores 1.00 vs 0.75.
3. **On Llama-3.1-8B, keeping every token at low precision wins.** TurboQuant
   averages 0.985 and SIEVE 0.654. SIEVE's losses concentrate at B = 2 and at
   128K (§5.4). Averaged over Llama's B = 2 cells, every token-selective
   method trails dense 2-bit quantization by at least 0.35: the best is
   OBCache-K + Ada-KV at 0.626 against TurboQuant's 0.976, and SIEVE scores 0.429.

**Table 2.** SIEVE against each baseline, per valid cell (win or loss = a
difference above 0.05).

| SIEVE vs | wins | ties | losses | mean Δ | 90% CI |
|---|---:|---:|---:|---:|---|
| TurboQuant-MSE | 5 | 13 | 18 | −0.180 | [−0.208, −0.152] |
| KIVI (g = 32 / 128) | [PENDING] | | | | |
| KVQuant | [PENDING] | | | | |
| H2O | [PENDING] | | | | |
| SnapKV | 31 | 1 | 4 | +0.367 | [+0.338, +0.398] |
| DropKV | 30 | 3 | 3 | +0.348 | [+0.314, +0.384] |
| Ada-KV | 23 | 3 | 10 | +0.262 | [+0.236, +0.290] |
| LaProx | 22 | 4 | 10 | +0.187 | [+0.154, +0.221] |
| OBCache-K + Ada-KV | 21 | 0 | 15 | +0.163 | [+0.129, +0.197] |
| best eviction method, per cell | 21 | 0 | 15 | +0.127 | — |

All 15 losses to the best eviction method are on Llama: 7 of the 8 cells at
B = 2 and 8K/32K, and all 8 cells at 128K.

**A harder regime.** TurboQuant is at 0.95 or above in 31 of 36 cells, which
limits how finely Table 1 can rank methods on Llama. We therefore add one
Llama-3.1-8B cell chosen *before* running any compressed method: multi-key
retrieval with 32 keys at 32K, where uncompressed accuracy is 1.00 and
TurboQuant at B = 2 was 0.825 on a screening block (40 prompts). We evaluate
all methods on 60 fresh prompts. [PENDING 991907: table, and whether the
pre-registered gate (FP ≥ 0.95, TurboQuant in [0.50, 0.90]) held.]

## 5.3 Output error ranks budgets, not methods

SIEVE minimizes attention-output distortion, and it does so: it has the lowest
head-output error of any method (0.120). Yet TurboQuant, with the *highest*
error (0.445), has the highest accuracy. Across all (cell, method) pairs,
output error and accuracy correlate at Spearman ρ = −0.35. Within the eviction
family the correlation reaches −0.69, but that comes from budget and context
length: lower budgets and longer contexts raise error and lower accuracy
together. **Within a cell, where methods compete at a fixed budget, the mean
rank correlation is −0.05 over 36 cells (−0.19 for eviction methods alone).**
Output distortion predicts *how hard* a setting is. It does not predict *which
method* wins.

This separates two claims that the KV-compression literature usually makes
together. The allocation theorem (§3) holds for its own objective: SIEVE
minimizes the distortion it targets. What fails is the transfer from that
objective to retrieval. An oracle router that picks each head's method from
its *measured* output error on the test prompt closes only part of the gap:
at Llama-3.1-8B 32K, B = 2 it scores 0.761 against 0.464 for the calibrated
router, and still trails TurboQuant (0.963) by 0.20 (90% CI [−0.32, −0.09]).
The residual gap is therefore the proxy, not the calibration.

## 5.4 Where SIEVE fails: long-context Llama

SIEVE's weakest setting is Llama-3.1-8B at 128K. It loses all eight cells to
the best eviction method, by 0.14–0.55, and at B = 2 it trails OBCache-K +
Ada-KV by 0.32 (0.442 vs 0.762).

**Table 3.** Llama-3.1-8B @ 128K, B = 2. The error types are for single- and
multi-key retrieval at B = 2 and 3, where an answer is one number: *truncated*
means the prediction is a proper prefix (≥ 3 digits) of the correct number.

| method | score | truncated | distractor | tokens evicted | head-output error | KV heads routed to SIEVE interior |
|---|---:|---:|---:|---:|---:|---:|
| TurboQuant-MSE | 0.942 | 0% | 0% | 0% | 0.706 | — |
| OBCache-K + Ada-KV | 0.762 | 4% | 10% | 75% | 0.182 | — |
| LaProx | 0.737 | — | — | 75% | 0.187 | — |
| SnapKV | 0.523 | — | — | 75% | 0.187 | — |
| SIEVE | 0.442 | 36% | 4% | 64% | **0.094** | 99% |
| SIEVE interior only | 0.409 | 39% | 6% | 65% | 0.096 | 100% |
| SIEVE interior, pooled score (diagnostic) | [PENDING 991904] | | | | | |
| SIEVE, oracle router (diagnostic) | [PENDING 991904] | | | | | |

<!-- truncated/distractor for LaProx and SnapKV at 128K are not yet split;
fill from read_main_table.py after the rerun -->

**What goes wrong.** SIEVE's wrong answers have a distinctive form. Asked for
the number 4929054 it answers "492"; for 9465170, "946"; for 6022964,
"602296". Llama tokenizes digits in groups of three, so SIEVE keeps the key and
the first token of the value and loses the tokens after it. Truncation makes
up 36% of SIEVE's single- and multi-key answers at 128K, against 4% for
OBCache-K + Ada-KV. The eviction methods fail differently: when they fail they
more often return a *different* needle's value (10% distractors), meaning a
whole needle was lost rather than part of one. Truncation is not specific to
128K: pooled over Llama's three context lengths, 40% of the SIEVE interior's
single- and multi-key answers are truncated, against 12–20% for the eviction
methods.

**Why the router does not help.** SIEVE's router can send a KV head to dense
quantization or to eviction instead of the interior. At 128K the offline
calibration sends 99% of Llama's KV heads (96% at B = 3) to the interior,
against 87–96% at 8K. The calibration minimizes output error, and by that
measure the interior *is* the best choice: at 0.094 it has half the error of
any eviction method. A truncated answer barely registers in output error. The
error is averaged over heads and over eight answer tokens, and losing the
second token of one needle's value moves that average very little while
changing the answer completely. This is §5.3's proxy failure, concentrated in
one setting.

**Why the eviction methods avoid it (hypothesis under test).** Every eviction
method that beats SIEVE here smooths its token score over neighbouring
positions before selecting (SnapKV, Ada-KV and OBCache max-pool over 7
positions; LaProx averages over 7). A kept token therefore protects its
neighbours, and a needle survives as a phrase. SIEVE's interior allocates each
token independently from unpooled attention, so the value tokens next to an
attended key can drop to few bits or be evicted. We pre-registered a test:
the same water-filling allocator on pooled attention supports this explanation
if it recovers at least half of the 128K, B = 2 gap to OBCache-K + Ada-KV, and
refutes it if it recovers less than a fifth. [PENDING 991904: result.] The
oracle router tests the other half of the account: if routing on the true
per-head error also fails at 128K, the calibration is not at fault.

**What this implies.** Output-distortion allocation at token granularity is
the wrong objective for retrieving multi-token spans. Both obvious repairs
(positional smoothing of the allocation score, or a span-level unit of
allocation) change the allocator's input and leave the water-filling theorem
unchanged.

---

# Appendix A. Full end-task grid

Mean task score over 20 held-out prompts per cell. Bold marks the best
non-diagnostic method in each row. † marks cells excluded from every aggregate
because the uncompressed model scores below 0.9.

| model | ctx | task | B | FP | TurboQuant-MSE | SnapKV | DropKV | Ada-KV | LaProx | OBCache-K + Ada-KV | SIEVE interior only | SIEVE (router) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama31-8b | 8K | niah_multikey | 2 | 1.00 | **1.00** | 0.20 | 0.30 | 0.25 | 0.25 | 0.40 | 0.35 | 0.30 |
| llama31-8b | 8K | niah_multikey | 3 | 1.00 | **1.00** | 0.35 | 0.55 | 0.55 | 0.55 | 0.75 | 0.60 | **1.00** |
| llama31-8b | 8K | niah_multivalue | 2 | 1.00 | **1.00** | 0.14 | 0.16 | 0.19 | 0.25 | 0.39 | 0.15 | 0.23 |
| llama31-8b | 8K | niah_multivalue | 3 | 1.00 | **1.00** | 0.34 | 0.42 | 0.42 | 0.46 | 0.59 | 0.40 | 0.99 |
| llama31-8b | 8K | niah_single | 2 | 1.00 | **1.00** | 0.45 | 0.55 | 0.50 | 0.60 | 0.70 | 0.30 | 0.35 |
| llama31-8b | 8K | niah_single | 3 | 1.00 | **1.00** | 0.75 | 0.65 | 0.75 | 0.80 | 0.75 | 0.60 | **1.00** |
| llama31-8b | 8K | vt | 2 | 1.00 | **0.98** | 0.45 | 0.49 | 0.42 | 0.40 | 0.54 | 0.43 | 0.60 |
| llama31-8b | 8K | vt | 3 | 1.00 | **1.00** | 0.60 | 0.64 | 0.62 | 0.59 | 0.88 | 0.67 | 0.95 |
| llama31-8b | 32K | niah_multikey | 2 | 1.00 | **1.00** | 0.30 | 0.20 | 0.35 | 0.45 | 0.55 | 0.40 | 0.45 |
| llama31-8b | 32K | niah_multikey | 3 | 1.00 | **1.00** | 0.45 | 0.35 | 0.70 | 0.70 | 0.70 | 0.60 | **1.00** |
| llama31-8b | 32K | niah_multivalue | 2 | 1.00 | **0.97** | 0.12 | 0.06 | 0.20 | 0.26 | 0.35 | 0.16 | 0.23 |
| llama31-8b | 32K | niah_multivalue | 3 | 1.00 | **1.00** | 0.42 | 0.23 | 0.56 | 0.61 | 0.70 | 0.40 | 0.90 |
| llama31-8b | 32K | niah_single | 2 | 1.00 | **1.00** | 0.60 | 0.35 | 0.65 | 0.75 | 0.85 | 0.55 | 0.60 |
| llama31-8b | 32K | niah_single | 3 | 1.00 | **1.00** | 0.70 | 0.60 | 0.95 | 0.95 | 0.90 | 0.65 | **1.00** |
| llama31-8b | 32K | vt | 2 | 1.00 | **0.99** | 0.28 | 0.37 | 0.46 | 0.47 | 0.69 | 0.46 | 0.63 |
| llama31-8b | 32K | vt | 3 | 1.00 | **1.00** | 0.65 | 0.55 | 0.82 | 0.87 | 0.86 | 0.64 | **1.00** |
| llama31-8b | 128K | niah_multikey | 2 | 1.00 | **1.00** | 0.30 | 0.40 | 0.50 | 0.60 | 0.65 | 0.35 | 0.40 |
| llama31-8b | 128K | niah_multikey | 3 | 1.00 | **0.95** | 0.45 | 0.50 | 0.80 | 0.70 | 0.85 | 0.50 | 0.55 |
| llama31-8b | 128K | niah_multivalue | 2 | 0.97 | **0.94** | 0.41 | 0.35 | 0.57 | 0.59 | 0.64 | 0.34 | 0.34 |
| llama31-8b | 128K | niah_multivalue | 3 | 0.97 | **0.99** | 0.54 | 0.55 | 0.79 | 0.81 | 0.85 | 0.55 | 0.69 |
| llama31-8b | 128K | niah_single | 2 | 1.00 | **1.00** | 0.90 | 0.75 | **1.00** | 0.95 | 0.95 | 0.45 | 0.45 |
| llama31-8b | 128K | niah_single | 3 | 1.00 | **1.00** | 0.95 | 0.85 | **1.00** | **1.00** | 0.95 | 0.65 | 0.65 |
| llama31-8b | 128K | vt | 2 | 1.00 | **0.83** | 0.48 | 0.42 | 0.74 | 0.81 | 0.81 | 0.50 | 0.58 |
| llama31-8b | 128K | vt | 3 | 1.00 | **0.99** | 0.67 | 0.73 | 0.90 | 0.94 | 0.96 | 0.79 | 0.82 |
| qwen3-8b | 8K | niah_multikey | 2 | 1.00 | **0.95** | 0.00 | 0.05 | 0.00 | 0.05 | 0.05 | 0.15 | 0.90 |
| qwen3-8b | 8K | niah_multikey | 3 | 1.00 | **1.00** | 0.10 | 0.20 | 0.15 | 0.30 | 0.10 | 0.50 | **1.00** |
| qwen3-8b | 8K | niah_multivalue † | 2 | 0.24 | 0.69 | 0.00 | 0.05 | 0.00 | 0.01 | 0.01 | 0.05 | **0.72** |
| qwen3-8b | 8K | niah_multivalue † | 3 | 0.24 | 0.49 | 0.00 | 0.07 | 0.00 | 0.21 | 0.10 | 0.28 | **0.59** |
| qwen3-8b | 8K | niah_single | 2 | 1.00 | 0.95 | 0.10 | 0.10 | 0.05 | 0.25 | 0.15 | 0.20 | **1.00** |
| qwen3-8b | 8K | niah_single | 3 | 1.00 | **1.00** | 0.25 | 0.35 | 0.25 | 0.55 | 0.40 | 0.60 | **1.00** |
| qwen3-8b | 8K | vt | 2 | 1.00 | 0.46 | 0.19 | 0.28 | 0.28 | 0.41 | 0.43 | 0.47 | **0.99** |
| qwen3-8b | 8K | vt | 3 | 1.00 | **1.00** | 0.63 | 0.63 | 0.62 | 0.66 | 0.68 | 0.81 | **1.00** |
| qwen3-8b | 32K | niah_multikey | 2 | 1.00 | 0.75 | 0.00 | 0.15 | 0.05 | 0.15 | 0.15 | 0.10 | **1.00** |
| qwen3-8b | 32K | niah_multikey | 3 | 1.00 | **1.00** | 0.10 | 0.30 | 0.10 | 0.30 | 0.25 | 0.40 | **1.00** |
| qwen3-8b | 32K | niah_multivalue † | 2 | 0.04 | **0.75** | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 | 0.01 | 0.12 |
| qwen3-8b | 32K | niah_multivalue † | 3 | 0.04 | **0.42** | 0.00 | 0.04 | 0.00 | 0.03 | 0.00 | 0.00 | 0.07 |
| qwen3-8b | 32K | niah_single | 2 | 1.00 | 0.95 | 0.20 | 0.30 | 0.20 | 0.40 | 0.25 | 0.35 | **1.00** |
| qwen3-8b | 32K | niah_single | 3 | 1.00 | **1.00** | 0.30 | 0.55 | 0.50 | 0.70 | 0.55 | 0.45 | **1.00** |
| qwen3-8b | 32K | vt | 2 | 1.00 | 0.40 | 0.28 | 0.40 | 0.45 | 0.79 | 0.61 | 0.46 | **0.99** |
| qwen3-8b | 32K | vt | 3 | 1.00 | 0.95 | 0.69 | 0.70 | 0.77 | 0.92 | 0.81 | 0.76 | **0.99** |

# Appendix B. Baselines, budgets and side information

| method | what is kept | score / quantizer | side information (bit per key element, d = 128) |
|---|---|---|---|
| TurboQuant-MSE | every token at B bits | random rotation, Lloyd-Max, per-token norm | 0.125 |
| KIVI (g = 32 / 128) | every token at B bits | per-channel asymmetric integer, min/max per g tokens, post-RoPE | 1.0 / 0.25 |
| KVQuant | every token at B bits | pre-RoPE per-channel non-uniform codebook; 1% outliers and the first token exact | ≈ 0.32 |
| H2O | ⌊B·C/8⌋ tokens at 8 bits | attention summed over every prefill query | ≈ 0.05 |
| SnapKV | same | observation window (32), max-pool 7 | ≈ 0.05 |
| DropKV / Ada-KV / OBCache-K / LaProx | same | as published; Ada-KV head allocation for Ada-KV and OBCache-K; LaProx global top-K | ≈ 0.05 |
| SIEVE | per-token width in {0, 1, 2, 3, 4, 5, 6, 8} | output-distortion water-filling per KV head; offline per-head router | 0.07–0.09 |

**Stated deviations.** KIVI keeps a partial final group quantized rather than
in full precision; a 32-token uncompressed window, shared by all methods, plays
the role of its residual. KVQuant calibrates its outlier thresholds and
non-uniform codebook offline with Fisher-weighted k-means; we fit both online
on each prompt's own context without weighting, which can only favour it. All
methods use simulated quantization (the model reads dequantized keys), so this
section measures accuracy, not memory traffic or latency.
