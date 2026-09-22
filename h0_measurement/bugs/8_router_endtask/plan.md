# R8 — router-on vs router-off on an end task


> **Status 2026-09-22: IN PROGRESS. Next job: P0b** —
> `bash h0_measurement/bugs/8_router_endtask/script.sh --p0b` (~1 GPU-h, not yet
> submitted).
>
> | phase | state | where |
> |---|---|---|
> | build + tests (P0 path, P2 path, question-agnostic mode, B = 0.5) | done; `tests/test_r8.py` all pass | §9, §10, §12 |
> | **P0** question-aware budget pilot, llama31-8b @32k | **done**, job 21529825 | **§11** — SnapKV 1.00 at every budget: with the question in its window, eviction is an oracle on these tasks |
> | **P0b** question-agnostic + B = 0.5 | **built, next to submit** | **§12** (design, checks, CPU pilot, decision table) |
> | P2 interior + routers (pilot → calibration → evaluation, 5 cells) | built, waits on P0b's read | §10 |
>
> The sections below are in the order they were written (§0–§8 the original
> plan, §9 onward the build and results). §11–§12 are current.

**Plan, 2026-09-21. P0 is BUILT and CPU-validated (§9); nothing has run on the cluster.** Every file below is NEW; no
shared file (`run_h0.py`, `alloc.py`, `probe.py`, `quant.py`, `evict.py`,
`test_units.py`) is edited, because wave 4 is about to run on exactly that code.
R8 only *imports* from it.

ROADMAP R8: *"Every number in the paper is an output-error ratio. A reader's
fair question is 'why should I believe this matters', and we currently have no
answer."* That is the job. Everything below is scoped to it.

---

## 0. Does R8 wait for waves 3 and 4?

**The build does not wait. Half the GPU runs do not wait. The other half must.**

| | waits on | why |
|---|---|---|
| **the compressed-inference path** (§3) | nothing | it is the same code whatever wave 4 decides |
| **the budget pilot** (§5 P0) | nothing | it measures where *uniform* breaks, which involves no router |
| **baseline arms** — full precision, uniform, evict | nothing | no interior, no routing, no GQA question |
| **interior-everywhere and router arms** | **wave 4** | wave 4 decides the two things these arms are made of |
| anything | **wave 3** | nothing — N16 tightens one residual in R4; N17/N18 are R5 controls |

Wave 4 decides exactly two design parameters, and the harness takes both as
**config flags**, so the outcome is a flag change, not a rewrite:

| wave-4 question | its outcome selects | R8 flag |
|---|---|---|
| is per-head routing storable under GQA? (R12) | `head` or `kv_head` allocation | `--granularity` |
| does the cascade close the lag gap? | lagged `accum` or cascade at bc 3/4 | `--interior-score` |

> **✅ Wave 4 has decided both (2026-09-21, `../co-design/report.md` §7).**
> `--granularity kv_head` — per-head routing is storable at a discount (in-band
> grouping cost 1.136× at n_rep 4, 1.316× at n_rep 8, exact 1.000 at n_rep 1), and
> the group constraint hurts eviction more than the interior. `--interior-score
> cascade --coarse-bits 4` — the base-tier score removes a median 89% of the lag
> penalty in-band, and its deployable form (no V read) matches its bound. **P2 is
> unblocked.** One consequence for P2's baselines: under the realizable constraint
> the eviction arm must be group-constrained too, or it is the same unfair
> comparison co-design plan B2 exists to prevent.

Running the router arms *before* wave 4 would produce an end-task number for an
allocation that may not be storable (if the group cost is large) or may be the
wrong score (if the cascade pays). The ROADMAP's own warning: *"R8 isn't blocked,
but it would produce an end-task number for an interior you're about to
redefine."* The pilot at n_rep 2 leans both ways already (grouping marginal
1.005–1.078×, cascade closing 0.72–0.81 at bc = 4), but n_rep 2 is not a headline
ratio.

---

## 1. The fact that sizes R8

**No code in this repository has ever fed an allocation into the model's
forward pass.** Read from `sievelib/probe.py:88-102`:

```python
out = _sdpa(query, key, value, attention_mask, scaling, q_len > 1)   # full precision
if STATE.enabled and q_len == 1:
    STATE.q[li] = query[0, :, 0]...                                   # only RECORDS q
return out, None
```

The model always attends over **full-precision** keys; `run_h0.py` then
computes *offline* what the error *would* be under each allocation. That is the
right design for a measurement — and it is why R8 is not "run the pipeline on
RULER". It needs a **compressed-inference path**: attention that actually reads
quantized and evicted keys, so the model generates from them and the answer is
scored.

This is why R8 is **~5–7 days to build**, not the ROADMAP's 3–5: the ROADMAP
entry was written assuming a path that does not exist.

The good news: the hook point does. `probe.py` registers its function through
`ALL_ATTENTION_FUNCTIONS["sieve_probe"]` and `run_h0.py:479` loads the model with
`attn_implementation="sieve_probe"`. R8 registers a sibling,
`"sieve_compress"`, in a new file. `probe.py` is not touched.

---

## 2. Design decisions — and the six that are not obvious

### 2.1 Simulated quantization, compressed at decode, prefill at full precision

Quantize-dequantize each key at its assigned width and attend in full precision
over the result. Accuracy depends only on the dequantized values; bit-packing
changes speed and memory, which is R14's job, not R8's. Prefill runs at full
precision, the cache is compressed, decode reads the compressed cache — which
is how deployed systems work, and exactly the population `run_h0` measured
(decode steps). K only, V exact, matching the paper (co-design plan B5).

### 2.2 ⚠ The lagged score is built from COMPRESSED attention — a feedback loop nothing has measured

Every R3–R7 number built its lagged `accum` score from **full-precision**
attention history, because the probe observes the full-precision forward. A
deployed system has no full-precision attention: it observes the attention it
*actually computed*, over compressed keys. If an eviction error shifts attention,
the shifted attention feeds the next allocation, and errors can **compound**.

R8 is the first place this loop exists. It is the same kind of asymmetry R3 was
built to remove, and it should be measured rather than assumed: an ablation arm
that observes full-precision attention (the R3 condition) against one that
observes compressed attention (the deployed condition). If they differ, the
paper's output-error numbers are optimistic by that gap.

### 2.3 ⚠ Re-budgeting from full precision allows upgrades a cache cannot perform

Simulated quantization keeps the full-precision K, so a re-budget can move a
token from 2 bits *up* to 8. A real cache that stored the token at 2 bits has
discarded what it would need. Re-budgeting-with-upgrade is realizable only if the
refinement tiers are retained — which is the **nested code** of ROADMAP R11, and
not yet measured.

**v1 therefore allocates once, at the start of decode, and never upgrades.** That
is fully honest, and it costs little here: see 2.4.

### 2.4 ⚠ Short-answer tasks do not test R5's two-timescale result

A RULER answer is 5–20 tokens. R5 found the **route** is stable over 4,096 steps
and the **allocation** goes stale (`froz/lag1` 2.5–3.4×) — but over *thousands*
of steps. On a 20-token answer the allocation is essentially prefill-time, so R8
tests the **router** and the **allocation shape**, and says nothing about
re-budget cadence. Validating the two-timescale design needs a long-generation
task with a judge. That is scope creep for v1; it is flagged, not done.

### 2.5 ⚠ A better model pair than the ROADMAP's

ROADMAP: qwen3-30b (high dead-tier) vs llama33-70b (low). That pair confounds the
phase variable with **size** (30B vs 70B), **architecture** (MoE vs dense) and
cost (llama33-70b is 140 GB bf16 and genuinely needs 2+ GPUs).

**Proposed primary pair: llama31-8b vs qwen3-8b.**

| | llama31-8b | qwen3-8b |
|---|---|---|
| params | 8B | 8B |
| heads | 32 q / 8 kv | 32 q / 8 kv |
| **n_rep** | **4** | **4** |
| dead-2 @8k | 18.8% | 50.5% |
| symmetric band @8k | **54.5% — GO** | **16.5% — NARROW/STOP** |
| GPUs | 1 | 1 |

Same size, same GQA ratio, same head count; they differ in the phase variable.
That isolates what C1 claims — and both fit on one card.

**Plus a context sweep within llama31-8b** — 8k (GO, 54.5) → 32k (NARROW, 34.0)
→ 128k (NARROW, 20.2). One model traversing the phase diagram on context alone
is C2's claim, and it gives R8 its figure: accuracy against context, router
against every fixed policy, with the phase boundary marked. qwen3-30b @32k
(STOP, 8.1, 1 GPU per X1) is the optional third point.

### 2.6 ⚠ Single-needle NIAH will saturate

The in-house `niah` family is one 5-digit code scored by substring match
(`validity.task_level_gate`), and it passed **6/6, 4/4, 3/3 on every cell** at
full precision. Single-needle retrieval is known to stay near 100% under
aggressive KV compression, so it cannot separate the arms.

It is the right **smoke test** — the compressed path must reproduce its 100% at
a generous budget — and the wrong **measurement**. R8's number needs RULER's
discriminating tasks: **multi-key NIAH, multi-value NIAH, variable tracking**.
`qa` is open-ended summarisation with no automatic scorer and is not usable.

### 2.7 Prefill once, share it across every arm

Prefill is the cost at long context; decoding 20 tokens is cheap. Prefill each
prompt once at full precision, then fork the cache for each (arm, budget). This
is what makes six arms × three budgets affordable (§6).

---

## 3. Code — all new files

| file | LOC | reuses (import only) | does |
|---|---|---|---|
| `sievelib/compress.py` | ~200 | `quant.quantize_keys`, `probe._sdpa` | `mixed_quantize_keys()`; the `sieve_compress` attention function; `COMPRESS_STATE`; `install()` |
| `sievelib/router.py` | ~120 | `alloc.waterfill`, `waterfill_floor`, `waterfill_group`, `evict.*` | per-head allocation for each arm; `load_routes()` from a calibration parquet |
| `sievelib/tasks_ruler.py` | ~150 | `prompts._build_haystack`, `prompts._seed` | MK-NIAH, MV-NIAH, variable tracking on the PG-19 haystack, with scorers |
| `h0_measurement/run_r8.py` | ~250 | `run_h0.load_cfg`, `next_token`, prompt plumbing | prefill once → fork per arm → decode → score → parquet |
| `tests/test_r8.py` | ~200 | — | T-R8-1…7 (§4) — a **new** file, so `test_units.py` is untouched |
| `bugs/8_router_endtask/script.sh`, `read_r8.py` | ~250 | — | submission sheet with guards; the reader |

### 3.1 `mixed_quantize_keys` — reuse, don't reimplement

`quant.quantize_keys(K, bits, R)` quantizes every token at **one** width. Per-token
mixed widths: call it once per distinct width on that width's token subset and
scatter back; width 0 marks eviction. No change to `quant.py`, and a test pins
that a uniform-width call is bit-identical to the original (T-R8-3).

### 3.2 `sieve_compress` — the attention function

At each decode step, per (layer, head): look up the allocation in
`COMPRESS_STATE`, dequantize K per token at its width, mask evicted positions to
−∞, softmax, attend over V. Then `observe()` the attention it computed, to build
the next lagged score (the 2.2 loop). The allocation is computed once at decode
start (2.3).

### 3.3 The arms — all at matched bits

| arm | per head | spends |
|---|---|---|
| **FP** | full precision | ceiling |
| **uniform** | every token at B bits (KIVI-like) | B·L |
| **evict** | SnapKV: keep B·L/maxb tokens at maxb, ranked by the observation window's attention pooled over the KV group's heads and max-pooled over positions (kernel 7) | B·L |
| **evict_h2o** | H2O: same keep count, ranked by attention from every prefill query. A literature check, not the fair baseline (§9.3) | B·L |
| **interior** | water-fill every head on the deployable score | B·L |
| **router** | interior if calibrated honest gain > θ, else argmin(uniform, evict) for that head | B·L |
| **oracle router** | route by the head's *true* gain on this prompt | B·L — upper bound on routing |

Every arm spends exactly B·L per head, so they are budget-matched by
construction; T-R8-4 asserts it.

### 3.4 The router is calibrated with the existing pipeline

The router needs each head's honest gain **before** generation — one offline pass
(C4), which R5 showed is stable (p90 regret 1.00 over 4,096 steps). That pass
already exists: it is `run_h0`. Run it on a calibration prompt set, read
`gain_pp3_<score>` per (layer, head), route where it exceeds θ.

Calibration and evaluation prompts must be **disjoint**, or the router is fitted
on its test set. `prompt_offset` (R7) provides exactly that. Calibration = block
0; evaluation = a disjoint block.

**θ is a design knob, not the band's 2×.** The band's 2× is a reporting
convention. As a routing threshold it sends 1.5× heads to the baseline and
leaves gain on the table. Sweep θ ∈ {1.0, 1.25, 1.5, 2.0}.

---

## 4. Tests — the correctness anchors (`tests/test_r8.py`)

| id | pins | why it matters |
|---|---|---|
| **T-R8-1** | compression disabled ⇒ `sieve_compress` output **bit-identical** to `sieve_probe` / sdpa | the new path adds nothing when off |
| **T-R8-2** | B = maxb, no eviction ⇒ generated tokens match full precision (within 8-bit noise) | the anchor: a generous budget must reproduce FP |
| **T-R8-3** | `mixed_quantize_keys` at one uniform width == `quantize_keys` at that width, bit for bit | reuse is exact, not approximate |
| **T-R8-4** | every arm's allocation spends B·L ± waterfill's one-tier tolerance | matched bits is the whole comparison |
| **T-R8-5** | evicted positions receive exactly zero attention mass | eviction is eviction |
| **T-R8-6** | the allocation used at generation == `alloc.waterfill_floor` on the same inputs | **R8 measures THE design the paper measured**, not a reimplementation that drifts |
| **T-R8-7** | `kv_head` granularity ⇒ one allocation shared by the `n_rep` heads; n_rep = 1 ⇒ identical to `head` | reuses co-design's `waterfill_group` and its exact control |

T-R8-6 is the one that matters most. Without it, R8 could measure a subtly
different allocator and nobody would know.

---

## 5. Experiments, in order

### P0 — the budget pilot. **Run this first; it can kill R8's design.**

If uniform at 3 bits/token already scores ~100%, every arm scores ~100% and R8
shows nothing. The paper's B = 3 may simply be too generous for an end task.

| | |
|---|---|
| cell | llama31-8b @32k |
| arms | FP, uniform, evict — **no router, runs before wave 4** |
| budgets | B ∈ {1, 2, 3, 4} |
| tasks | MK-NIAH, MV-NIAH, variable tracking; single-needle as the smoke test |
| prompts | 20 per task |
| output | the budget where **uniform falls to 50–80%** — that is where R8 must run |

Accept only if FP ≈ 100% (the path works) and uniform degrades somewhere in
B ∈ {1…4}. If uniform is ~100% even at B = 1, the tasks are too easy: add harder
RULER variants (more keys, more distractors) before going further.

### P1 — baselines on the full grid (before wave 4)

llama31-8b @ 8k/32k/128k and qwen3-8b @ 8k/32k, the P0 budgets, arms FP / uniform
/ evict. No dependence on wave 4.

### P2 — interior and router arms (after wave 4)

Same grid, arms interior / router / oracle router, with wave 4's `--granularity`
and `--interior-score`. θ sweep on one cell first, then the chosen θ everywhere.

**Superseded in detail by §10** (built): per KV head, window score with cascade as
an ablation arm; `script.sh --p2-pilot / --p2-cal / --p2`. θ: the calibration run
writes routes at one θ; a sweep is `--theta=` on `--p2-cal` into separate route
files, only if P-5 (oracle − calibrated) comes out large.

### P3 — the two ablations

- **observe compressed vs observe full** (2.2) — prices the feedback loop.
- **granularity `head` vs `kv_head`** — the end-task version of wave 4's R12
  number. If wave 4 says the group cost is small, this should agree.

### P4 — the payoff analysis (0 GPU)

§7's prediction P-4: does the per-cell **output-error** gain predict the per-cell
**end-task** gain?

---

## 6. Cost

Prefill once per prompt, then 6 arms × 3 budgets × ~20 decode steps. For an 8B
model at 32k on one H100, roughly 1–2 minutes per prompt for all arms together.

| phase | cells | ≈ GPU-h |
|---|---|---|
| P0 budget pilot | 1 | ~1.5 |
| P1 baselines | 5 | ~5 |
| P2 interior + router | 5 | ~5 |
| P3 ablations | 2 | ~2 |
| **total** | | **~13** |

These are estimates, not measurements; P0 measures the real per-prompt rate and
every later header is rescaled from it. At 128k prefill dominates, so size that
cell on P0's rate × (L)^0.3, as the co-design sheet does.

---

## 7. Decision table, written before any run

| id | prediction | if it holds | if it fails |
|---|---|---|---|
| **P-1** | router ≥ max(uniform, evict, interior) at every (model, ctx, B), within noise | the router never costs accuracy — the minimum claim | find the cell; it is either a calibration failure (check P-5) or the 2.2 loop |
| **P-2** | router − best fixed policy is **largest in GO cells**, ~0 in STOP | the gain tracks the phase diagram | the diagram does not predict end-task value; C1 is descriptive only |
| **P-3** | interior-everywhere < router **in STOP cells** | "the router recovers accuracy exactly where the diagram predicts the allocator degenerates" — **the ROADMAP's sentence, verified** | the interior does not degenerate where predicted; restate C1/C4 |
| **P-4** | Spearman(per-cell output-error routed gain, per-cell end-task gain) **> 0.7** | **output error is a valid proxy for accuracy — every number in the paper is retroactively justified** | the paper's metric does not transfer; the strongest possible reason to lead with R8 |
| **P-5** | oracle router − router ≤ a few points | one calibration pass is enough (C4) | a better router is worth building; the gap says how much |
| **P-6** | observe-compressed ≈ observe-full | the output-error numbers are not optimistic | they are optimistic by the gap; report it |

**P-4 is the reason to run R8.** P-1 to P-3 defend the router. P-4 defends the
paper's *metric*, which is what the reviewer's question — "why should I believe
this matters" — is actually about.

**What would invalidate a cell:** FP below ~95% on a task (the model cannot do the
task uncompressed, so compression cannot be measured on it), or P0 finding no
budget where uniform degrades.

---

## 8. Limits of v1

- **Short answers only** (2.4): says nothing about re-budget cadence.
- **No upgrades** (2.3): allocate once. Re-budgeting-with-upgrade waits on R11.
- **K only, V exact**, matching the paper; a deployed system quantizes V too.
- **Simulated quantization**: accuracy, not speed or memory (R14).
- **RULER-style tasks on PG-19**, generated natively on the validated haystack.
  If a reviewer requires the official RULER harness, that is an addition, not a
  redesign — the tasks are the same definitions.
- **Two architectures** in the primary pair. qwen3-30b @32k is the optional third.


---

## 9. P0 — built and CPU-validated (2026-09-21)

### 9.1 What exists

| file | what |
|---|---|
| `sievelib/compress.py` | the `sieve_compress` attention; `mixed_quantize_keys`; observation-window and H2O capture; `crop_to` |
| `sievelib/router.py` | per-arm widths: `uniform`, `evict` (SnapKV), `evict_h2o` (H2O) |
| `sievelib/tasks_ruler.py` | `niah_single`, `niah_multikey`, `niah_multivalue`, `vt` on the PG-19 haystack, RULER templates and scoring |
| `h0_measurement/run_r8.py` | prefill once, crop between arms, greedy decode, score, parquet + sidecar |
| `h0_measurement/submit_r8.slurm` | the cluster launcher — a new file, so wave 4's `submit_h0.slurm` is untouched |
| `tests/test_r8.py` | 50+ checks, below |
| `bugs/8_router_endtask/{script.sh, read_r8.py}` | the P0 sheet with gates and guard; the reader with the P0 rule |

No shared file was edited. Everything imports from `quant`, `probe`, `prompts` and
`run_h0` without changing them.

### 9.2 The anchors — all pass

| test | pins |
|---|---|
| T-R8-1 | compression off == `sieve_probe`, **bit for bit**, decode and prefill, with the capture running |
| T-R8-2 | 8-bit keys within 1% of FP (tensor); on Llama-3.2-1B, the uniform-8 answer == the FP answer, token for token |
| T-R8-3 | `mixed_quantize_keys` == `quant.quantize_keys`, bit for bit, every width |
| T-R8-4 | every arm spends ≤ B bits/token, three context lengths |
| T-R8-5 | an evicted position has **exactly zero** influence (poisoning its value moves the output by 0 ulp) |
| end to end | FP arm == `model.generate` token for token; FP repeats exactly after a crop; the FP ceiling is correct |
| capture | window score chunking-invariant (4 layouts); H2O == brute-force causal column sum (3 layouts, 3.8e-6) |
| `crop_to` | truncates on every transformers version (see 9.4) |

### 9.3 CPU pilot — qwen3-1.7b @2,048, 2 prompts, all four arms

Accuracy (RULER string match), mean of 2 prompts. **Two prompts: a mechanism
check, not a measurement.**

| arm | B | single | multikey | multivalue | vt |
|---|---|---|---|---|---|
| fp | — | 1.00 | 1.00 | 1.00 | 1.00 |
| uniform | 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| uniform | 2 | 0.00 | 0.00 | 0.00 | 0.10 |
| uniform | 3 | 1.00 | 0.50 | 0.63 | 0.60 |
| uniform | 4 | 1.00 | 1.00 | 1.00 | 1.00 |
| **evict** (SnapKV) | 1 | **1.00** | **1.00** | 0.63 | 0.60 |
| **evict** (SnapKV) | 2 | 1.00 | 1.00 | 1.00 | 0.60 |
| **evict** (SnapKV) | 4 | 1.00 | 1.00 | 1.00 | 1.00 |
| evict_h2o | 1–2 | 0.00 | 0.00 | 0.00 | 0.20 |
| evict_h2o | 3–4 | 0.00 | 0.50 | 0.38 | 0.50 |

Bits audit exact: uniform spends B; evict spends B − ε with evict fraction exactly
1 − B/8.

**What it shows:**

1. **Uniform has a cliff, not a slope** — dead at 1–2 bits, borderline at 3, fine
   at 4. The dead 1-bit tier is the project's own measurement (1-bit costs more
   than eviction in 95–100% of heads); here it is an end-task fact.
2. **Eviction dominates uniform at low budget** — the project's C1/C3 thesis on an
   end task: at 1 bit/token, keeping 1/8 of the tokens at 8 bits beats keeping all
   of them at 1.
3. **They fail differently.** Uniform corrupts *content*: `3191729 → 3591729`, the
   right needle with one wrong digit. SnapKV drops *structure*: on `vt` it answered
   with the first chain variable five times, because the question mentions the
   value but not the intermediate links, so the links score low and are evicted.
   **Different failure modes are exactly the room a router or an interior has.**
4. **Faithful SnapKV mattered.** Without its positional pool, `evict` scored 0.50
   on `niah_single` and 0.13 on `niah_multivalue` at B = 1; with it, 1.00 and 0.63.
   The first version was a weaker baseline than the method it was named after.

### 9.4 Three defects caught before any GPU time

| # | defect | would have caused | caught by |
|---|---|---|---|
| 1 | **H2O capture crashed on every early prefill chunk** — an early chunk's keys stop at `k_len < ctx_len`, and the slice was narrower than the accumulator | every prompt of the GPU P0 dying on its first 4,096-token chunk | the brute-force test |
| 2 | **`DynamicCache.crop(positive)` changes meaning at transformers 5.18** — positive = "keep first N" up to 5.17, removed after | on a newer cluster, every arm after the first mis-cropped, silently reading the previous arm's generated tokens | the deprecation warning in a test log; `crop_to` now always removes a counted number |
| 3 | **the end-to-end test skipped the newline stop** — `run_arm` was called without `tok` | the production answer-stopping path untested | reading the test's own output (`'5715821.\n\nI will quiz…'`) |

### 9.5 H2O: the multi-hop hypothesis, refuted

The hypothesis was that H2O, which counts attention from *every* query and not just
the question, would credit chain links and rescue `vt`. Measured on Llama-3.2-1B,
4 prompts:

| | Spearman(score, position) | needle's rank percentile |
|---|---|---|
| SnapKV | +0.22 … +0.33 | **0.4 – 0.6%** |
| H2O | **−0.40 … −0.47** | **31 – 64%** |

H2O's raw prefill sum is dominated by **position**: under causal attention token
*j* is seen by every later query, so the sum decays like ln(L/j). A needle at depth
0.93 ranks at the 61st percentile — evicted even when half the context is kept.
That is SnapKV's published advantage over H2O, reproduced, which is a useful check
that both baselines behave like their papers.

**Consequence:** `evict` (SnapKV) is the fair baseline. `evict_h2o` runs in the GPU
P0 as a literature check and is dropped from P1/P2 unless it wins somewhere.

**Correction to 9.1's first draft:** `evict_h2o` is **not** the paper's `accum`
corner. `accum` sums over *decode* steps, where every query sees every token, so it
has no causal-count bias. The prefill sum does.

### 9.6 The prediction P0 can test on its own

C1 says the dead 2-bit fraction orders the heads. qwen3-1.7b's is ~78%, and
uniform at 2 bits scored **0.00**. llama31-8b @32k's is ~27–31%, so ~70% of its
heads keep a live 2-bit tier. **Uniform at B = 2 on llama31-8b should be clearly
above zero.** If it is ~0 as well, the dead-tier fraction does not predict where
the end-task cliff falls — the most important thing P0 can find, because it is the
link between C1 and accuracy that P-4 is built on.

### 9.7 Open, for P2

SnapKV's `vt` failure has a direct consequence for the interior. The water-fill
allocates on `w² = (a·‖v−o‖)²`, and a chain link the question never mentions has
low `a` — so the interior may **evict the same links** SnapKV does, and fail `vt`
the same way. Whether a mixed-precision interior rescues multi-hop depends on
whether low-score tokens land at *some* bits rather than zero. That is an
empirical question P2 answers, and the reason `vt` stays in the task set.

---

## 10. P2 — built and CPU-validated (2026-09-21), waiting on P0's budgets

Wave 4 fixed both open design parameters (../co-design/report.md §7): allocation
**per KV head** (the group cost is small, and a real cache stores one K per KV
head), and **cascade scoring at bc = 4**. R8 v1 allocates once, at the first answer
token, while the full-precision keys still exist — so the interior's default score
is the window's full-precision attention, and the cascade (current query × 4-bit
keys) is kept as the ablation arm `interior_cascade`, not the default.

### 10.1 What exists

| piece | where | what |
|---|---|---|
| interior allocator | `sievelib/router.py` `alloc_interior` | per KV group: `alloc._sens` → `_rel` (n_rep > 1) → `alloc.waterfill_group`; same code path as co-design S1 |
| arms | `router.base_bits` | `interior`, `interior_pool` (SnapKV's positional max-pool applied to the window score), `interior_cascade` (bc = 4 keys) |
| per-head output error | `router.eval_heads` | vectorised over (queries × group heads); mean over the FP arm's first `n_q` = 8 decode queries |
| router | `router.route` / `compose` / `calibrate_routes` | per KV head, candidates interior / uniform / evict, margin θ; `router_oracle` routes on this prompt's own errors, `router_calib` on a routes file from a disjoint calibration block |
| driver | `run_r8.py --head-error --n-q --theta --cascade-bits --routes --write-routes` | FP arm first (captures the decode queries), one precompute per prompt, then every arm × B; writes `r8heads_*.parquet` (per layer × head error + answer mass) |
| guards | `run_r8.load_routes` | refuses routes built for another model / ctx, calibration prompts that overlap the eval block, and missing budgets — before the model loads |
| submit | `submit_r8.slurm` | `R8_HEAD_ERROR`, `R8_N_Q`, `R8_THETA`, `R8_CASCADE_BITS`, `R8_ROUTES` (must exist), `R8_WRITE_ROUTES` |
| campaign | `script.sh --p2-pilot / --p2-cal / --p2` | `--budgets=` required (from P0, no default); `--eval-prompts=N` (walls scale N/20); `--theta=` |
| reader | `read_r8.py --p2 [--p2-csv] [--p2-verbose]` | paired bootstraps, P-1 … P-5, split-half reliability ceiling for P-4 |
| tests | `tests/test_r8.py` | T-R8-6 `alloc_interior` == `waterfill_group` by hand (bit-for-bit); T-R8-7 n_rep = 1 == `waterfill`; vectorised `eval_heads` == `alloc.exact_error` loop (max diff ~1e-16); answer-span; route/compose |

The P0 path (`p2 = False`) is byte-for-byte what P0 is running: re-run on the CPU
pilot, 104 rows identical.

### 10.2 The campaign

| mode | cells | prompts | arms | output |
|---|---|---|---|---|
| `--p2-pilot` | llama31-8b @32k | 200–201 | fp, uniform, evict, interior, interior_pool, interior_cascade, router_oracle | the per-prompt rate → rescale every header below |
| `--p2-cal` | llama31-8b @8k/32k/128k, qwen3-8b @8k/32k | 0–9 | fp, uniform, evict, interior | `results/r8_routes/<model>_<ctx>.json` |
| `--p2` | same five | 100–(100+N−1), N = 20 default | all eight | r8 + r8heads parquets; the decision table |

Order: P0 read → `--p2-pilot --budgets=…` → rescale walls → `--p2-cal` → `--p2`.
`evict_h2o` is dropped from P2 (§9.5: a weak baseline with a positional bias)
unless P0 finds it winning somewhere.

**128k is unmeasured on every path.** The precompute quantizes each layer's
context keys at 7 widths (~0.5 GB each in fp32 at 131k for 8 KV heads), one layer
at a time, so it should fit on one H100 — but no run has confirmed it. If the
pilot's 32k rate looks fine, the 128k cal cell (5 h header) is the first job to
watch.

### 10.3 CPU finding 1 — the answer span and step 0

The first decode query (step 0) attends mostly to the *key* the question names,
not the value that must be copied out. On a multikey prompt the answer digits sat
at the 7.3th percentile of step-0 attention and the named key at the 0.7th. An
interior scored on the window vote therefore protects the key better than the
answer, and at B = 2 it put 32% of the answer tokens into the evicted tier against
SnapKV's 27%. Two principled responses, both kept as arms, not as a default:

- `interior_pool` — SnapKV's max-pool over positions spreads a key's score onto its
  neighbours, which is where the value sits in a `key: value` haystack line.
- the per-head errors average over **8** decode queries, not 1 — later queries
  attend to the value being copied.

Both fixed that prompt. Neither is evidence yet: it is one prompt at a knife-edge.

### 10.4 CPU finding 2 — P-4 cannot be tested per prompt

On the same prompts, no statistic of the per-head error — median, mean, p99, max,
or answer-mass-weighted — separated the arms that passed from those that failed.
Near the cliff one prompt's pass/fail is decided by whether a handful of answer
tokens survive, which an average over 1,000+ heads cannot see. That is not a
failure of the metric; it is the wrong unit. **P-4 is tested per cell** —
(model, ctx, task, B), accuracy as a rate over prompts, error gain as the
geometric mean over prompts of err_uniform / err_arm — which is what §7 says and
what `read_r8.py --p2` does.

### 10.5 A power problem in P-4's threshold, and what the reader does about it

A cell's accuracy gain is the mean of ~20 paired pass/fail differences, with
binomial noise of ~0.1–0.15, the same size as the gains being ranked. That noise
caps the Spearman any error statistic can reach at about √reliability, where
reliability is the split-half correlation of the per-cell gains
(Spearman–Brown corrected). On synthetic data built with a *perfect* proxy and
realistic noise, the reader measured ρ = 0.33–0.51 against a ceiling of 0.32–0.42:
a literal "ρ > 0.7" would have rejected a perfect metric.

So the reader reports three numbers for P-4, for each of four error statistics
(mean — what the water-fill minimises —, median, p99, answer-weighted):
ρ with a 90% bootstrap over cells, the reliability ceiling, and ρ / ceiling.
Verdicts: **HOLDS** (lower end of ρ's interval > 0.7), **HOLDS after
disattenuation** (ρ / ceiling > 0.7 and ρ clearly > 0), otherwise consistent or
FAILS. The paper states the raw ρ and the ceiling; never the disattenuated value
alone.

**What to do before P2:** run `read_r8.py --p2` on P0's output. Evict vs uniform
over 4 tasks × 4 budgets are cells too, so it prints the reliability at N = 20
before any P2 GPU time. If the ceiling is well under ~0.85, raise
`--eval-prompts` (reliability grows roughly like N / (N + c)); 40 prompts doubles
every eval wall.

### 10.6 Cost, per prompt × task at 2k on CPU

P0 path 45 s, P2 path 48 s, of which the precompute is 7 s (vectorising
`eval_heads` cut its Python calls ~32×). The precompute grows linearly in the
context and the decodes do not, so the P2/P0 ratio at 32k–128k is unknown until
the pilot runs.

### 10.7 Selection effect in "best fixed"

P-1 and P-2 compare the router with the best of uniform / evict / interior
**chosen on the same prompts**. That choice is biased toward whichever fixed
policy got lucky, so "router − best fixed" is conservative: P-1 failures should
be re-checked against each fixed policy (`--p2-verbose` prints them all).

---

## 11. P0 result — job 21529825 (2026-09-22): the tasks are solved by SnapKV

llama31-8b @32,768 (30.2k prompt tokens), 20 prompts × 4 tasks, maxb = 8, W = 32.
Bits audit OK. Wall 38.5 min (header 1:30): **29 s per prompt × task**, of which
prefill 11.3 s and ~1.2–1.6 s per arm decode. Reader output:
`h0_measurement/reports/r8_p0.csv`.

| task | FP | uniform B=1 / 2 / 3 / 4 | evict (SnapKV) B=1…4 | evict_h2o B=1 / 2 / 3 / 4 |
|---|---|---|---|---|
| niah_single | 1.00 | 0.25 / 1.00 / 1.00 / 1.00 | **1.00 at every B** | 0.00 / 0.00 / 0.00 / 0.10 |
| niah_multikey | 1.00 | 0.35 / 1.00 / 1.00 / 1.00 | **1.00** | 0.20 / 0.45 / 0.65 / 0.80 |
| niah_multivalue | 0.99 | 0.06 / 1.00 / 1.00 / 0.99 | **1.00** | 0.11 / 0.33 / 0.45 / 0.54 |
| vt | 1.00 | 0.20 / 1.00 / 1.00 / 1.00 | **1.00** | 0.61 / 0.78 / 0.76 / 0.85 |

### 11.1 What it says

1. **§9.6's C1 prediction holds.** Uniform at B = 2 is 1.00 on llama31-8b (dead-2
   ≈ 27–31%), where qwen3-1.7b (≈ 78%) scored 0.00. The cliff sits between 1 and 2
   bits: 1-bit keys corrupt the value (`650417` → other digits), 2-bit keys are exact.
2. **SnapKV is at ceiling at B = 1** — keeping 12.5% of the context (~3.8k tokens)
   at 8 bits — on every task, **including vt**. §9's CPU hypothesis (SnapKV drops
   the chain links the question never names) is refuted at this scale: the 32
   question tokens in the window vote every hop onto the kept set.
3. **H2O is the weak baseline §9.5 predicted**, and fails in a telling way: it keeps
   the *first* digits of the needle and drops the rest (`650417` → `650.`) — the
   causal-count bias acts even inside a 6-token number. Drop it from P2.
4. **No budget puts uniform in [0.50, 0.80]**, and the arm the router must beat
   (evict) is at 1.00 with zero variance at every B. P2 as planned would measure
   ties: no end-task gain exists for P-1…P-5 to rank, and P-4 has no variance to
   correlate. **P2 is on hold until the regime is fixed.**

### 11.2 Why: the question is in the window at compression time

Every arm compresses *after* prefill of context + question, and the window vote
that scores tokens is the question itself. On retrieval tasks that makes eviction
an oracle: the question points at the needle. This is the published blind spot
of query-aware eviction (compress-once-query-many, prefix caching, multi-turn):
the compressor does not know the question when it compresses. It is also exactly
the regime where a *unified* quantize-or-evict allocator should matter — mixed
precision keeps everything at some bits, so a needle the scorer did not foresee
degrades rather than vanishes.

### 11.3 Options for P0b

| option | change | expected | verdict |
|---|---|---|---|
| **A. query-agnostic compression** | score and compress the context **before** the question: the window = the last W context tokens; the question is prefilled after compression and kept FP | SnapKV loses the needle at low B (its vote no longer points at it); uniform unchanged (1→2-bit cliff); the interior and router have room between them | **recommended** — the realistic deployment regime and the one C1/C4 are about |
| B. sub-1-bit budgets (0.25, 0.5) | allow float budgets; uniform undefined below 1 | SnapKV at 1.5–3% of tokens still likely keeps a ~60-token needle set; interior ≈ evict | cheap but probably still at ceiling |
| C. harder tasks (16 keys, 8 values, 8 hops) | `tasks_ruler.build` knobs | question-aware SnapKV still points at the needle | does not attack the cause |

A and B combine: A defines the regime, B (0.5 bits) extends the budget axis
below uniform's cliff. **Chosen (2026-09-22): A + B → P0b, §12.**

---

## 12. P0b — question-agnostic compression + a 0.5-bit budget (built 2026-09-22)

### 12.1 What changes, and what does not

| | P0 (question-aware) | P0b (`--question-agnostic`) |
|---|---|---|
| tokenization | `tok(context + question)` | `tok(context)` ⧺ `tok(question, add_special_tokens=False)` — as a prefix cache holds them |
| prefilled at full precision | context + question (all but the last token) | **context only** |
| observation window (SnapKV vote, interior score, noise model) | last W = 32 prompt tokens = **the question** | last W context tokens = **the haystack's tail** |
| compressed | context before the window | same |
| question | inside the window, full precision | prefilled **per arm, through the compressed cache**; full precision itself |
| decode | from the last prompt token, over the compressed context | same |
| FP arm | plain generation | plain generation (the question prefill is uncompressed) |

Mechanism: `compress.sieve_compress_attention` gained one branch — a multi-token
call while `STATE.enabled` (only ever the question prefill) substitutes the
compressed context and applies an explicit causal + evicted + caller mask
(`_question_view`). The context prefill never takes it (`STATE.enabled` is False
there), and the question-aware path is untouched.

Budgets may be fractional. At **B = 0.5** eviction keeps 1/16 of the context at
8 bits and the interior water-fills 0.5 bits per token; **uniform is skipped**
(there is no 0.5-bit quantizer). The routes file keys budgets with
`router.bkey` ("2", "0.5"); routers drop the uniform candidate where it does not
exist. The reader compares against evict where uniform is absent.

The R9 SOTA baselines (`bugs/9_sota_eviction_baselines`) read the same prefill
capture, so `--question-agnostic` applies to them unchanged — in this mode they
are blind in the same way SnapKV is.

### 12.2 Verification

| check | result |
|---|---|
| `tests/test_r8.py` (full, CPU) | **ALL PASS**, incl. 3 new tests: question prefill over a compressed context == row-by-row decode (max diff 1.2e-7), evicted positions have zero influence on the question, causal inside the question, bool-mask safe; B = 0.5 budget/route pieces; end to end on Llama-3.2-1B — QA fp arm == `model.generate(context + question)` token for token, u8 == fp, evict 0.5 spends ≤ 0.5 b/tok |
| `tests/test_baselines.py` (R9) | ALL PASS |
| question-aware path unchanged | old vs new `run_r8.py` on qwen3-1.7b @2k (fp/uniform/evict, B 1,3, 2 tasks): **identical parquets** except timing columns |
| mechanism (Llama-3.2-1B @2k, niah_single) | blind SnapKV's kept set shifts toward recent tokens (mean kept position 0.61 → 0.75 of the context at B = 1); needle-token survival 0.92 → 0.59 when the needle is not at the very end |

### 12.3 CPU pilot, qwen3-1.7b @2,048 (partial; plumbing + direction only)

Partial: the scratch output was lost when the session ended. These are the
per-prompt log lines that had printed (5 of 12 prompt × task pairs in the P0b-shaped
run, plus the first 3 tasks of the P2-shaped run). qwen3-1.7b is the high-dead-tier
toy model (dead-2 ≈ 78%), so absolute levels do **not** transfer to llama31-8b.
Each entry is a single prompt, so every score is 0/1 or a fraction of the expected
answers.

P0b-shaped (`--arms fp,uniform,evict --budgets 0.5,1,2,3,4 --question-agnostic`):

| prompt · task (needle depth) | fp | uniform 1 / 2 / 3 / 4 | evict 0.5 / 1 / 2 / 3 / 4 |
|---|---|---|---|
| p0 niah_single (0.45) | 1.00 | 0 / 0 / 0 / **1.00** | 0 / 0 / 0 / 0 / **0** |
| p0 niah_multikey | 1.00 | 0 / 0 / 0 / 1.00 | 0 / 0 / 0 / 0 / 0 |
| p0 niah_multivalue | 1.00 | 0 / 0 / 0.75 / 1.00 | 0 / 0 / 0 / 0 / 0 |
| p0 vt | 1.00 | 0 / 0 / 0 / 1.00 | 0 / 0 / 0 / 0.20 / 0.80 |
| p1 niah_single (deep) | 1.00 | 0 / 0 / 0 / 1.00 | 0 / 0 / **1.00 / 1.00 / 1.00** |

P2-shaped (`+ interior, interior_pool, router_oracle --head-error`, B 0.5 / 1 / 3), prompt 0:

| task | uniform 1 / 3 | evict 0.5 / 1 / 3 | interior 0.5 / 1 / 3 | interior_pool 0.5 / 1 / 3 | router_oracle 0.5 / 1 / 3 |
|---|---|---|---|---|---|
| niah_single | 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / **1.00** | 0 / 0 / **1.00** |
| niah_multikey | 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / **1.00** | 0 / 0 / **1.00** |
| niah_multivalue | 0 / 0.75 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / **1.00** |

What this is and is not evidence for:
- **The regime discriminates.** Blind SnapKV fails whenever the needle is not near
  the end (p0, depth 0.45: 0 even while keeping half the context at B = 4) and
  passes when it is (p1). That is the mechanism measured on Llama-3.2-1B (§12.2).
- **Uniform's cliff moves up** (P0's CPU run: 3 bits; here: 4), because the question
  now reads compressed keys too.
- On prompt 0 at B = 3, **interior_pool and the oracle router answer where uniform,
  SnapKV and the plain interior all fail.** That is one prompt on a toy model: a
  plumbing check that also points in a direction, not a result.
- The P2 path ran end to end in this mode (60 rows, uniform correctly absent at
  B = 0.5, 25,088 per-head errors), and `read_r8.py --p2` read it.

**Bits audit tolerance (reader).** The water-fill arms (interior, routers)
overspend by up to ~0.05% of B: the bisection on λ cannot hit B exactly with
discrete widths. Eviction and uniform spend ≤ B exactly. `read_r8.py` now flags
only overspends above 0.2% of B and prints the worst one. `alloc.waterfill_group`
is left unchanged: H0's campaigns and golden tests depend on it.

### 12.4 Known approximation in QA mode (P2 only)

`router.eval_heads` measures each head's output error over **context + window**
keys; in QA mode the decode queries also attend to the question's (exact) keys,
which the per-head error leaves out. The error is therefore relative to a
slightly smaller softmax than the model uses. It affects the per-head error
columns (P-4's x-axis) only, not accuracy, and is the same for every arm.

### 12.5 The P0b run

`bash h0_measurement/bugs/8_router_endtask/script.sh --p0b` — llama31-8b @32,768,
arms fp/uniform/evict, budgets 0.5/1/2/3/4, 20 prompts × 4 tasks, `R8_QA=1`,
header 1:00:00 (estimate ~35 min from P0's 29 s per prompt × task).

Decision table (written before the run):

| P0b shows | then |
|---|---|
| evict < ~0.9 at some B with FP ≥ 0.95 | the regime discriminates → P2 with `--qa`, at the budgets where uniform and evict differ most (0.5 counts: interior vs evict there) |
| evict still ~1.00 at B = 0.5 | eviction is not the bottleneck even blind → rethink the task set before P2 |
| uniform ≪ P0's uniform at the same B | the question now reads compressed keys too; expected, and it moves uniform's cliff — read it per task |
| FP < 0.95 | the separate tokenization broke the prompt → debug before reading arms |
