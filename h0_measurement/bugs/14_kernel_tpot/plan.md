# R14 · Kernel, iso-budget, TPOT — plan (frozen before any R14 output)

Written 2026-09-27. There was no `bugs/14_*` folder before this one.
**Status (2026-09-29): Stage 1b's frozen decision is `STOP_SYSTEMS`, recorded
with a caveat: it was decided at the noise floor (report.md, Decision and C2).
The kernel (§6–§7) is not built. Next: §13.**
History:
- Stage 0: `GO_KERNEL` (2026-09-27).
- Stage 1: 128K `BYTES_TIE` (amendment A1; jobs 996008–996013).
- Stage 1b: amendment A2; jobs 999908–999916.

ROADMAP §5 R14 reads: "Kernel, iso-budget, TPOT vs dense FP8 (old E7) · ~4
weeks · FP8 currently beats every quantization variant on the Pareto frontier,
which is the bar. Kill the systems claim if TPOT at 128k does not beat dense FP8
at equal accuracy."

That entry predates R10–R13. §0 lists what is now wrong with it. The rest of the
plan re-scopes R14 into stages, each of which stops early and cheaply when its
answer is already determined.

## 0. What is wrong with the R14 entry as written

The entry rests on the proposal-v5 cost model (`docs/proposal-sieve-v5.html`
§5). That model has SIEVE reading 0.423·Ld bytes per step against 2.0·Ld for
dense k8v4, hence "4.7× fewer bytes at identical memory". Every input to it has
since been measured or changed:

| v5 cost-model input | now | source |
|---|---|---|
| a 2-bit pass-0 scan over every token | the base tier is 3 bits, and the cascade reads a 4-bit prefix (bc = 4) | R10, R11 |
| 5% of tokens keep their V | SIEVE evicts 64% / 49% at 128K (B = 2 / 3), so 36–51% keep V | R12 failure table |
| re-budget every step from stored 8-bit codes | the only SIEVE with end-task accuracy **allocates once at decode start and never upgrades**. It uses the full ladder {0,1,2,3,4,5,6,8} with a separate (monolithic) Lloyd codebook per width, not R10's tier set or R11's nested code | `sievelib/compress.py` header; all 16 R12 run records (`bit_list` [1..6, 8], `codebook` None, `cascade_bits` 4) |
| "at equal accuracy" | Llama-3.1-8B @128K: SIEVE router 0.506 / 0.721 at B = 2 / 3, against FP 0.983 and TurboQuant 0.931 / 0.984 | R12 |
| FP8 is the bar | a dense 3-bit key quantizer is as accurate as FP on most cells (TurboQuant-3 grid mean 0.995 vs FP 0.996) and reads about 0.70× FP8's bytes, so it is the bar. R13 adds that the baseline set needs a per-channel quantizer | R12, R13 |
| a systems claim to defend | the paper makes none: "The experiments therefore establish matched key-bit output error, not matched total memory or throughput." Its claim-boundary table lists "total-KV memory savings, or realized throughput" as not supported | `latex/appendix.tex` |

This has two consequences.

1. **At 128K the kill criterion is already decided on measured budgets, before
   any kernel exists.** SIEVE has no budget ≤ 3 bits at which it matches FP
   accuracy there, and no kernel can change an accuracy number. The only open
   question at 128K is whether B = 4 reaches it (Stage 1).
2. **The comparison the paper has never made is on total bytes.** Every accuracy
   number is keys only with values exact, and every budget counts key code bits.
   Evicting a token also removes its value read; a dense key quantizer reads
   every value. At 2–3 key bits the values (16 bits BF16, 8 bits FP8) are most of
   the bytes. The byte ranking can therefore differ from the key-bit ranking, and
   any difference favors eviction. The Limitations sentence "budgets match key
   code bits rather than total KV bytes" names exactly this gap. Closing it needs
   no GPU (Stage 0).

## 1. Questions, in the order they are answered

| stage | question | cost | runs if |
|---|---|---|---|
| 0 | On total bytes read per decode step, is SIEVE (as R12 measured it) cheaper than the cheapest dense quantizer at full-precision accuracy? | 0 GPU, ~1 day | always |
| 1 | Does SIEVE reach full-precision accuracy at B = 4 on Llama @128K? Is FP8 KV lossless on the R12 tasks, and is FP8 V lossless under every arm? | ~1 day of code, ~6 node-h; additive shared-code edits that need the user's OK | Stage 0 finds no lossless SIEVE point at 128K |
| 2 | Does the byte saving turn into attention-kernel time on an H100? | 1–2 weeks of kernel work, ~3 node-h | ≥ 1 unit is BYTES_WIN in Stage 0 |
| 3 | TPOT end to end | 1–2 weeks | Stage 2 is KERNEL_WIN; not planned in detail here |

## 2. The two designs, and the byte model

**D-once** is the configuration R12 measured. The allocation happens once, after
the context prefill. A deployed cache stores each kept token at its assigned
width only: nothing re-reads a token at another width, so evicted tokens are not
stored and no refinement bits are kept. For D-once, memory and bytes read per
step are the same number.

Two properties make D-once cheap to lay out:
- **Attention over the compressed context is permutation-invariant.** Keys are
  cached post-RoPE, and every context token is visible to every later query
  (decode, and the question prefill in question-agnostic mode). The kept tokens
  of a KV head can therefore be sorted by width into contiguous single-width
  segments at allocation time. The kernel then reads dense arrays, with no
  per-token gather and no per-token width index.
- **It needs neither the nested code (R11) nor the retained observation bit.**
  Nothing is upgraded or re-scored. Each segment uses the codebook R12 used for
  its width.

**D-rebudget** is the paper's two-clock design. It stores every token's full
8-bit nested code and its V, reads each kept token's prefix at its current
tier, and re-scores with a bc = 4 scan every k steps. It has output-error
evidence (co-design) but **no end-task accuracy**, so it cannot enter an
iso-accuracy comparison. R14 prices it analytically only (Stage 0, secondary).

**Byte model.** The unit is bytes per context token, per KV head, per layer, at
d = 128. `B_eff` is code bits plus side bits per key element, `f` is the evicted
fraction of context tokens, and `v` is the value width in bits.

| format | K (codes + side) | V | example: B = 3, V FP8 |
|---|---|---|---|
| BF16 KV | 256 | 256 | 512 |
| FP8 KV (E4M3, static per-(layer, KV head) scale) | 128 | 16·v | 256 |
| dense key quantizer (TurboQuant, KIVI-G, KVQuant) | 16·B_eff | 16·v | TurboQuant: 178 |
| eviction, kept tokens at 8 bits | 16·B_eff | (1−f)·16·v | f = 0.63: 96 |
| SIEVE, D-once | 16·B_eff | (1−f)·16·v | f = 0.49: 115 |

D-rebudget adds 64·g/k bytes per token per step for the bc = 4 scan, with
g = 1 if evicted tokens can come back and g = 1−f if eviction is permanent. It
stores 128 + 16·v bytes for every stored token.

**Side bits** follow R12's rule (`bugs/12_paper_main_table/read_main_table.py`):
- TurboQuant: 16/d.
- KIVI-G: 32/G.
- KVQuant: its row's `side_bits`.
- evictors: (1−f)·16/d + 1/d.
- SIEVE: (1−f)·16/d + 3/d.

D-once's sorted layout needs no width index or keep bitmap, so R12's rule
overcharges SIEVE and the evictors by at most 0.4 bytes per token. Stage 0 uses
R12's rule unchanged, which is the conservative direction.

**Uncompressed tail.** The protected window (W = 32), the question and the
generated tokens stay BF16 in every arm, at 512 bytes per tail token. They are
identical across arms and are under 1% at ≥ 32K. Stage 0 includes them.

**Per step, whole model:** n_layers × n_kv_heads × (C × per-token bytes + tail).
Weights are added only where a batch-1 or batch-4 share of TPOT is reported.

**Two V lenses.** All accuracy is measured with V exact, so every arm uses the
same V format within a lens:
- **V16:** V in BF16 in every arm. This is the contract R12 measured. The FP8
  comparator becomes FP8 keys with BF16 values, 384 bytes.
- **V8:** V in FP8 in every arm, so FP8 KV costs 256 bytes. This is the
  deployment default. Its accuracy is assumed equal to V16's until Stage 1
  measures it.

A verdict holds unconditionally only if both lenses agree.

## 3. Predictions (written before Stage 0; not blind)

Stage 0 reads R12 numbers that are already in the repo
(`bugs/12_paper_main_table/tables_r12_paired.md`). What is pre-registered here is
the accounting and the decision mapping, not ignorance of the inputs. The table
below is computed by hand from that file for the ROADMAP's own cell,
Llama-3.1-8B @128K (bytes per context token per KV head per layer):

| method | B | score | evicted | V16 bytes | V8 bytes |
|---|---|---|---|---|---|
| full precision (BF16) | 16 | 0.983 | 0 | 512 | 512 |
| FP8 KV | 8 | ≈ FP (Stage 1) | 0 | 384 | 256 |
| TurboQuant | 2 / 3 | 0.931 / 0.984 | 0 | 290 / 306 | 162 / 178 |
| KIVI-128 | 2 / 3 | 0.899 / 0.984 | 0 | 292 / 308 | 164 / 180 |
| KVQuant | 2 / 3 | 0.904 / 0.971 | 0 | 293 / 309 | 165 / 181 |
| SIEVE router | 2 / 3 | 0.506 / 0.721 | 64% / 49% | 125 / 180 | 79 / 115 |
| OBCache-K + Ada-KV | 2 / 3 | 0.774 / 0.914 | 75% / 63% | 97 / 144 | 65 / 96 |

- **P1 (Llama @128K): NO_POINT, then KILL_128K.** SIEVE has no lossless point
  at B ≤ 3. The cheapest lossless dense point is TurboQuant-3 (0.984; 178 bytes
  at V8). At an assumed 40% eviction, SIEVE at B = 4 costs 16·4.1 + 0.6·128 ≈
  142 bytes (V8). That is ρ ≈ 0.80 against TurboQuant-3, on the WIN/TIE line
  even if SIEVE-4 is lossless. Being lossless needs a score ≥ 0.963, above the
  oracle router's 0.956 at B = 3. Predicted: NO_POINT at B = 4 as well
  (KILL_128K), or at best a lens-dependent tie.
- **P2 (Llama 8K/32K, Qwen3-8B 8K/32K): BYTES_WIN.** SIEVE-3 is lossless on the
  valid tasks (0.985, 0.993, 1.00, 0.98). The cheapest lossless dense point is
  TurboQuant-2 on Llama (162 bytes at V8), and KVQuant-2 or KIVI-32-2 on Qwen
  (165–176 bytes; TurboQuant-2 fails Qwen's vt). Suppose SIEVE-3 evicts about
  half its tokens there, as it does at 128K. Then it costs ≈ 115 bytes, giving
  ρ ≈ 0.65–0.71 (V8) and 0.59–0.62 (V16), so BYTES_WIN in 4 units. Mistral
  8K/32K probably has no lossless SIEVE point (router mean 0.699).
- **P3: FP8 is no longer the bar.** The cheapest lossless dense points already
  read 0.63–0.70× FP8's bytes (V8).
- **P4: D-rebudget's scan eats the saving.** A bc = 4 scan costs 64 bytes per
  token (revivable eviction), more than SIEVE-3's whole saving over
  TurboQuant-2 at Llama 32K (162 − 115 = 47). Keeping ρ ≤ 0.80 needs k ≥ 5
  (revivable) or k ≥ 3 (permanent eviction). F3 found that allocations go stale
  within steps, which argues against intervals that long.
- **P5: batch 1 dilutes everything.** At 128K the model's weights (~16 GB) are
  read every step. SIEVE-3 and TurboQuant-3 (V8) read 20.0 vs 22.1 GB per step
  at batch 1 (0.90×) and 31.5 vs 40.0 GB at batch 4 (0.79×). SIEVE-3 is not
  lossless at 128K (P1); the point is the dilution. Any systems statement
  concerns long context with batch > 1, or capacity.
- **P6 (Stage 2): V skipping is realized almost fully** (values are a plain
  read). SIEVE's key path costs the same per byte as dense TurboQuant's (the same
  unpack and table lookup). Predicted: SIEVE's bandwidth efficiency within 10%
  of TurboQuant's, and both below FP8's.

## 4. Stage 0 — the bytes frontier (0 GPU)

**Inputs.**
- The R12 paired grid exactly as `tables_r12_paired.md` read it (A:
  `r12job21832327`, `21832330`, `21841737`, `21841738`, `21850510`, `21850511`).
- The Mistral grid (D: `r12job21840797`, `21840803`, `21840804`).
- All of these are corpus b524da5e, and every arm of a job ran in one process,
  so rows pair within a job.
- The hard cell (`r12job21850534`) is reported, not gated.
- Calibration-only runs and the failed Trillium attempts are excluded.
- R12's validity rule carries over: a task cell with FP < 0.9 is dropped (the
  Qwen3 multivalue cells).

**Units.** There are seven (model, ctx) units: Llama 8K/32K/128K, Qwen3-8B
8K/32K, and Mistral-7B 8K/32K. A unit's score for an (arm, B) is the macro mean,
over its valid tasks, of the per-task mean.

**Lossless.** An (arm, B) is lossless in a unit iff both of the following hold:
- its score is ≥ FP's − 0.02;
- the paired prompt-bootstrap 90% interval of (arm − FP) has a lower bound
  ≥ −0.05, which is R12's tie margin.

The bootstrap resamples prompts within a job, all tasks of a prompt together,
with 10,000 draws and seed 14. FP8 KV has no R12 row: it is treated as lossless,
and every FP8 statement carries "pending Stage 1".

**Comparators.**
- **D\***: the lossless dense-quantizer point with the fewest bytes, over
  TurboQuant, KIVI-128, KIVI-32 and KVQuant at B ∈ {2, 3}. KIVI-32 counts at its
  effective bits, so its extra side information is paid for. If no dense point
  is lossless, D\* is FP8 KV.
- **FP8 KV**: reported for continuity with the ROADMAP criterion; not gating.
- **S\***: the lossless `router_calib` point with the fewest bytes. That arm is
  the paper's SIEVE. `router_oracle` and `interior_pool` are diagnostics and are
  reported only.

**Statistic.** ρ = bytes(S\*) / bytes(D\*), per unit and lens. Bytes come from
each row's audited `bits_per_token`, `evict_frac` and the §2 side rule,
averaged over the unit's rows.

**Also written, not gated:**
- the full (bytes, score) scatter and Pareto set, per unit and lens;
- per-step bytes including weights, at batch 1 and batch 4;
- the number of sequences that fit on an 80 GB GPU at the unit's ctx;
- D-rebudget's ρ for k ∈ {1, 2, 4, 8, 16, 64}, with revivable and with
  permanent eviction;
- the break-even value width v\*, below which a dense quantizer that also stores
  v-bit values matches SIEVE's bytes. No arm measures value quantization's
  accuracy, so v\* is a byte statement only.

Outputs: `stage0.json` and `stage0.md`.

## 5. Stage 1 — the missing accuracy points (GPU; needs a shared-code OK)

Stage 1 runs only if Stage 0 finds Llama @128K to be NO_POINT (predicted).

**Design.**
- One process per job, with all arms paired (ROADMAP methods rule).
- Llama-3.1-8B @128K, question-agnostic, with R12's task config.
- 20 prompts in two jobs of 10 (100–109 and 110–119). These are R12 A's
  indices, so the cell means can be sanity-checked against R12. Row-level
  comparison with R12 is invalid: it would cross runs and corpora (0a26bc1e on
  Trillium, b524da5e on the other cluster).

**Arms.**
- `fp` and `fp8kv`;
- `uniform` (TurboQuant) and `kivi_g128`, each at B = 2, 3, 4;
- `router_calib` at B = 2, 3, 4, and `router_oracle` at B = 3, 4 (diagnostic);
- V-FP8 variants (`--v-fp8`) of `fp`, `uniform`-3, `kivi_g128`-3 and
  `router_calib`-3/4.

**Calibration.** Routes at B = 4 on prompts 0–9, written to a new routes file.
`results/r8_routes/llama31-8b_131072_qa.json`, which R12 used, is not
overwritten.

**Shared-code edits.** Each is additive and off by default, and each is pinned
by a test that the default path is bit-identical:

| file | change | blast radius |
|---|---|---|
| `sievelib/compress.py` | `STATE.vdeq`: a per-layer substitute context V, like `kdeq`. None leaves behaviour unchanged | every R8/R12 arm passes through `sieve_compress_attention` |
| `sievelib/kv_quant_baselines.py` | `fp8kv` arm: K and V to E4M3 with a static per-(layer, KV head) amax scale | new arm only |
| `h0_measurement/run_r8.py` | accept `fp8kv`, and `--v-fp8`, which applies FP8 V to every arm. B = 4 already works through `--budgets` | CLI only; defaults unchanged |
| `h0_measurement/submit_r8.slurm` | forward the new flag | none by default |

None of these is made until the user approves. The R8 standing exception does
not cover R14.

**Stage 1 decisions** (§4's lossless rule throughout):
- **SIEVE lossless at B = 4:** Llama @128K re-enters Stage 0 with B ∈ {2, 3, 4},
  and ρ is recomputed.
- **Not lossless at B = 4:** **KILL_128K.** No TPOT-at-128K claim is possible
  at these budgets.
- **FP8 KV not lossless:** FP8 drops out as a comparator, and the ROADMAP
  criterion is restated against D\*.
- **FP8 V lowers any arm's score by more than 0.02:** the V8 lens is withdrawn
  for that arm, and its verdicts use V16.

## 6. Stage 2 — the kernel microbenchmark (GPU)

Stage 2 runs if Stage 0 has at least one BYTES_WIN unit. It measures whether
bytes become time. It makes no accuracy claim of its own; it inherits R12's
accuracy only through the equality gate K1.

**Toolchain.**
- Triton 3.6.0 from the Alliance wheelhouse
  (`wheelhouse/generic/triton-3.6.0+computecanada-cp311`).
- Installed with `pip install --no-index --no-deps --target` into a separate
  directory that only R14 jobs put on `PYTHONPATH`, with `TRITON_CACHE_DIR` on
  /scratch.
- The shared `.venv` is not modified. It has torch 2.13.0+computecanada on CUDA
  13.2 and no Triton, flash-attn, FlashInfer or vLLM.
- The cluster has no `cuda` module, so there is no CUDA C++ fallback. If Triton
  fails K0, the stage stops and reports.

**Layout (D-once, per sequence, layer and KV head).**
- Kept context tokens are sorted by width into segments [8 | 6 | 5 | 4 | 3 | 2 |
  1], covering whatever widths R12's `bit_list` produced.
- Codes are stored bit-sliced. A b-bit token is b planes of d/8 = 16 bytes;
  plane j holds bit j of every channel's index. Every width therefore costs
  exactly 16·b bytes, with no padding.
- Each token has one fp16 scale, γ' = γ/‖ŷ‖, which folds TurboQuant's norm
  correction in.
- V is stored for the same tokens in the same order, in BF16 or FP8.
- A BF16 tail segment holds the window, the question and the generated tokens.
- Each (layer, KV head) keeps a table of at most 8 segment offsets.

**Kernel.**
- Split-K decode attention (flash-decoding) runs over a flat work list of
  (sequence, KV head, segment, chunk) items, each carrying its width.
- One program serves all n_rep query heads of its KV head, so K and V are read
  once per GQA group.
- Logits are computed in TurboQuant's rotated basis. The kernel forms q' = Rq
  once per head, then s_i = scale · γ'_i · Σ_c q'_c · L_b[idx_ic].
- A second kernel merges the partial (max, sum, acc) results.
- That is two launches per layer, captured in a CUDA graph. Launch overhead
  per width would otherwise eat a large share of the saving at these sizes.

**Formats, all in the same framework, so the format is the only variable:**

| format | notes |
|---|---|
| BF16 | the reference for K2 |
| FP8 E4M3 K and V | static per-(layer, KV head) scale; the ROADMAP comparator |
| TurboQuant-b, dense, b = 2, 3, 4 | the SIEVE kernel with one segment and f = 0 |
| KIVI-g128 and KIVI-g32 | per-channel scale and zero point per token group; bit-sliced like TurboQuant |
| evictors (kept at 8 bits) | the SIEVE kernel with one 8-bit segment |
| SIEVE `router_calib`, D-once | real per-(layer, KV head) allocations |
| KVQuant | bytes model only; its sparse-outlier path is not implemented |

**Inputs: real allocations.** The bench job:
1. loads the model;
2. builds a question-agnostic, R12-style prompt from a fresh block (3000–3001);
3. prefills the context;
4. calls `run_r8.prefill` and `run_r8.precompute` with the existing routes
   (import only, no edit);
5. keeps `bits[(arm, B)][layer]` and the post-RoPE K and V, then frees the model.

**Cells.**
- Llama @32K and Qwen3-8B @32K: the predicted BYTES_WIN units, at a length where
  decode attention is memory-bound.
- Llama @128K at B = 2, 3: reported as realization only, not as an iso-accuracy
  comparison.

Batches of 1, 4 and 16 sequences reuse one allocation per cell.

**Timing.**
- Each measurement is one decode step's attention over all layers (q' rotation,
  split kernel, merge), replayed as one CUDA graph.
- 50 warm-up and 200 timed replays, reported as median and p10–p90, on two
  GPUs (different nodes if the queue allows).
- Every measured size touches far more than the 50 MB L2 except 32K at batch 1,
  which is flagged as latency-bound.
- Also measured once: the one-time compression cost at the end of prefill
  (allocate, quantize, sort, pack) for SIEVE, TurboQuant and FP8. It is reported
  as time-to-first-token overhead and is not gated.

**Validity gates.** A failure means `invalid_r14_kernel`: fix it, do not report.
- **K0, toolchain:** Triton compiles and runs on the H100, and a reference copy
  kernel reaches ≥ 85% of torch's copy bandwidth.
- **K1, equality:**
  - For every format and cell, dequantizing the packed codes reproduces
    `compress.mixed_quantize_keys` (fp32, before its bf16 cast) to ≤ 1e-3
    relative per key. The fp16 γ' is the only rounding.
  - The kernel's attention output matches an fp32 reference over the same
    dequantized keys, V and eviction mask to ≤ 5e-3 relative L2 per (layer,
    query head). That is below the simulation's own bf16 rounding of the keys,
    which is what lets R12's accuracy transfer to the kernel.
  - The unsorted reference agrees to the same tolerance, which checks the
    permutation argument of §2.
- **K2, BF16 kernel quality:** at 128K and batch 4, our BF16 kernel reaches
  ≥ 80% of measured stream-read bandwidth. It is also no slower than the fastest
  torch SDPA backend at the same shape (recorded). Otherwise every compressed
  format would be compared against a weak baseline.
- **K3, FP8 quality:** at 128K and batch 4, our FP8 kernel's bandwidth
  efficiency is ≥ 0.9× our BF16 kernel's, so FP8 is not a strawman.
- **K4, accounting:** each packed cache's allocated bytes equal Stage 0's model
  for that allocation to within 1%, plus the segment tables.

## 7. Stage 3 — TPOT (not planned in detail)

Stage 3 opens only on KERNEL_WIN (§8).
- **Integration:** the kernel is registered as an HF attention function, as
  `compress.install` does, with a static cache and a CUDA-graphed decode step.
- **Measurement:** TPOT at batch 1, at FP8's capacity and at the format's
  capacity.
- **Check:** greedy tokens match the simulated arm on the R12 prompts (≥ 99%).
- **At 128K:** it runs only if Llama @128K is BYTES_WIN after Stage 1.

A D-rebudget kernel (bit-plane prefixes, a periodic scan, upgrades) is not part
of R14. It waits on the observation-bit and allocator-smoothing results, and on
any end-task accuracy for re-budgeting.

## 8. Decision table (frozen)

**Stage 0, per unit and lens:**

| verdict | rule |
|---|---|
| **BYTES_WIN** | S\* exists and ρ ≤ 0.80 |
| **BYTES_TIE** | S\* exists and 0.80 < ρ ≤ 1.00 |
| **BYTES_LOSS** | S\* exists and ρ > 1.00 |
| **NO_POINT** | SIEVE has no lossless point at the measured budgets |

A unit's verdict is unconditional if both lenses agree. Otherwise it is reported
as lens-dependent and counts as BYTES_TIE for gating.

Why 0.80: a kernel does not realize all of an ideal byte saving, because of
unpacking, table lookups and uneven per-head lengths. At ρ = 0.80 with half
realized, the kernel gain is about 10%, which is within the spread between
competent implementations of one format. Below that margin, a systems sentence
does not survive "compare against a better-tuned FP8 kernel".

| outcome | next |
|---|---|
| no unit BYTES_WIN (unconditional) | **STOP_BYTES**: R14 ends. The paper keeps "no memory or throughput claim" and may gain the bytes table as an appendix result |
| ≥ 1 unit BYTES_WIN | **GO_KERNEL**: Stage 2 on those units |
| Llama @128K is NO_POINT | Stage 1, independent of the rows above |
| still NO_POINT at B = 4 | **KILL_128K**: the ROADMAP's TPOT-at-128K criterion fails on accuracy |

**Stage 2, per BYTES_WIN unit**, at its ctx and batch 4 (primary; batches 1 and
16 are reported). t is attention time per decode step. D\*_k is the cheapest
lossless dense point that has a kernel, which excludes KVQuant.

| verdict | rule |
|---|---|
| **KERNEL_WIN** | t(S\*) ≤ 0.85 · t(D\*_k) |
| **KERNEL_TIE** | 0.85 < ratio ≤ 1.15 |
| **KERNEL_LOSS** | ratio > 1.15 |

Also recorded, not gated: each format's bandwidth efficiency relative to our
BF16 kernel, and the time ratio against FP8.

**Paper consequences** (no `latex/` edit without the user):
- **STOP_BYTES:** keep "no memory or throughput claim", optionally with the
  bytes table.
- **GO_KERNEL and KERNEL_WIN:** one attention-kernel sentence, scoped to the
  winning units. For example: "at full-precision accuracy, SIEVE's attention
  reads ρ× the bytes of D\* and takes t× its time on an H100 at 32K, batch 4".
  Still no TPOT claim.
- **KERNEL_TIE or KERNEL_LOSS:** the bytes table plus the realized ratio.
- **KILL_128K:** state that SIEVE does not reach full-precision accuracy on
  Llama @128K within 4 key bits.

## 9. Files and shared code

New files, all in `h0_measurement/bugs/14_kernel_tpot/`:

| file | stage | role |
|---|---|---|
| `bytes_model.py` | 0 | pure byte-accounting functions and the R12 frontier reader; writes `stage0.{json,md}` |
| `test_r14.py` | 0, 2 | CPU tests: byte identities (BF16 = 512, FP8 = 256, one-width SIEVE at f = 0 equals dense TurboQuant); the lossless rule on synthetic rows; packer round trip against `mixed_quantize_keys`; sorted vs unsorted reference attention |
| `r14pack.py` | 2 | turns an allocation into sorted, bit-sliced segments plus γ', V and the tail |
| `r14kern.py` | 2 | Triton kernels (split and merge) for every format in §6 |
| `bench_r14.py` | 2 | captures and allocates (imports `run_r8` and `sievelib`), packs, runs gates K0–K4 and the timings; writes `stage2_<job>.{json,txt}` |
| `submit_r14.slurm`, `script.sh` | 2 | `--smoke` (debug) and `--bench` (compute). Trillium rules apply: submit from trig-login01, pass settings as arguments not env, no `--mem` |

**Shared code.** Stages 0 and 2 change none: `run_r8`, `sievelib.compress` and
`sievelib.quant` are only imported. Stage 1's edits are listed in §5 and wait
for approval. This plan does not edit `ROADMAP.md`, `README.md` or `latex/`.

## 10. Budget and order

| stage | person time | GPU |
|---|---|---|
| 0 | ~1 day | 0 |
| 1 | ~1 day of code and tests | ~6 node-h (Llama 128K calibration + two evaluation jobs) |
| 2 | 1–2 weeks | debug-partition iterations (1 h each, one queued at a time) + ~2–3 node-h of bench |
| 3 | 1–2 weeks | ~1 day |

The ROADMAP puts R14 after the observation-bit A/B and allocator smoothing.
Both matter only for D-rebudget:
- D-once never re-scores, so it has no observation bit to keep.
- After sorting, token- and span-granular allocations give the same kind of
  layout.

Stages 0–2 can therefore run now, and D-rebudget stays behind those two items.
If the ICLR submission is close, Stage 0 is the only part of R14 that can land
in it.

## 11. Known limits

- **Stage 0 is not blind** (§3).
- **Few budget points.** Each method has two budgets (three for the 128K cell
  after Stage 1). ρ compares measured points; it does not interpolate a
  frontier.
- **Only measured methods are on the frontier.** The obvious hybrid, a dense
  2–3-bit quantizer applied to the tokens SnapKV keeps, is not in R12. If SIEVE
  wins on bytes, that hybrid is a reviewer's first question; one extra Stage 1
  arm could answer it.
- **No value width below 8 bits** is measured for any arm. The break-even width
  v\* (§4) says where the eviction advantage would disappear.
- **R12's scope carries over:** four synthetic retrieval tasks and three 7–8B
  models.
- **Our kernels, not production ones.** K2 and K3 bound how weak our baselines
  can be, but a production FP8 decode kernel (FlashInfer, vLLM) may still beat
  ours. FlashInfer 0.7.0 is on PyPI, but it JIT-compiles CUDA and the cluster
  has no CUDA toolkit module.
- **H100 only.** NVFP4 needs Blackwell, and the B200 nodes are reserved.

## 12. Amendments

### A1 (2026-09-27, before any Stage 1 output)

These were made while building Stage 1. The only Stage 1 outputs so far are CPU
smoke runs of the new code on Llama-3.2-1B at ≤ 2K tokens. They are excluded by
construction and exist to exercise the code.

1. **More comparator arms.** The dense set is TurboQuant, KIVI-128, KIVI-32 and
   KVQuant at B = 2, 3, 4, so D\* is defined over the same four quantizers as in
   Stage 0. `router_oracle` (diagnostic) also runs at B = 2. The `+v8`
   variants run at B = 2, 3, 4 for `fp`, `uniform`, `kivi_g128` and
   `router_calib`, so that a D\* or S\* at any budget has a measured V8 twin
   (a CPU smoke run showed D\* can land on a B = 2 point). All of these run in one process per block.
2. **Routes are recalibrated at B = 2, 3, 4** on this cluster's prompts 0–9
   (corpus 0a26bc1e), into a new file
   (`r8_routes/r14_llama31-8b_131072_qa_b234.json`). Adding only B = 4 to R12's
   routes would have mixed two corpora inside one router. R12's routes file is
   not touched.
3. **How the new arms are built.**
   - `fp8kv` is a fixed-width arm, run once with B recorded as 8. It is not a
     per-budget quantizer arm.
   - Each `<arm>+v8` runs directly after `<arm>` and reuses its keys, eviction
     mask and audit (the same tensors), so each pair differs in its values
     alone.
   - `fp+v8` has exact keys and FP8 values, under fp's B = 0 label.
   - FP8 uses float8_e4m3fn with one amax/448 scale per (layer, KV head) over
     the context. The window, the question and generated tokens stay exact, as
     they do for the keys.
4. **Pilot and gate added** (mechanics only; excluded).
   - The pilot is Llama @128K, prompt 3100, `niah_single`, arms `fp`,
     `uniform`, `kivi_g128` and `router_oracle` at B = 3, 4, with every R14
     arm.
   - The gate (`read_stage1.py --pilot`) checks that every planned row is
     present once, fp8kv is audited at 8 bits with nothing evicted, and every
     `+v8` row is audited exactly like its base. It also requires peak GPU
     memory ≤ 76 GiB and a projected main block ≤ 9 h.
   - The calibration uses no R14 option (it is R12's unchanged path), so it runs
     in parallel with the pilot.
5. **The V8-lens rule, made operational.**
   - A pair whose point Δ(+v8 − base) is below −0.02 withdraws the V8 lens for
     that arm.
   - If S\* or D\* is withdrawn, the unit verdict uses V16 alone.
   - If S\* or D\* has no measured pair (KIVI-32, KVQuant), its
     V8 lens stays assumed, and the output says so.
   - FP8 KV is lossless iff `fp8kv` passes §4's lossless rule against `fp`.
6. **Cost corrected.** R12's own 128K logs (jobs 21850510/11, H100 80 GB on the
   other cluster) show about 170 s of prefill per (prompt, task). That makes the
   calibration about 3 h and each main block about 4.5 h, roughly 12 GPU-h in
   total (§10 said ~6).
7. **Shared-code edits** were made as listed in §5, after the user's go-ahead
   (2026-09-27). The pre-edit copies are kept read-only in
   `pre_stage1_originals/`. `test_r14_stage1.py` pins the default attention path
   bit-identical to the pre-edit `compress.py`, and `test_r8`, `test_baselines`
   and `test_kv_quant_baselines` all still pass.
8. **Submission** is by the user from trig-login01:
   `bash h0_measurement/bugs/14_kernel_tpot/script_stage1.sh --run`.

### A2 (2026-09-28, before any Stage 1b output): Stage 1b added

Stage 1's analysis (report.md B2) left questions that §8 cannot answer:
- whether an eviction mask plus one dense width does as well as SIEVE;
- whether the byte edge survives values below 8 bits;
- whether pooling repairs the deletions;
- whether any proxy sees them.

Stage 1b's rules were frozen in `read_stage1b.py`'s docstring before any Stage
1b output existed. They are copied below unchanged.

1. **Arms**, from the `s1b_lib.py` presets `main128` and `main32`, one process
   per block:
   - dense TurboQuant;
   - the calibrated SIEVE router;
   - a pooled router, with `interior_pool` as its interior candidate,
     calibrated on the same prompts;
   - both routers also at a half-bit budget, whose dense candidate runs at
     floor(B);
   - a pooled oracle (diagnostic);
   - hybrids: SIEVE's eviction mask, or SnapKV's selection with the same
     per-KV-head keep counts, with every kept token at one TurboQuant width;
   - `+v4` / `+v2` twins: the base arm's keys and eviction, with values through
     TurboQuant-MSE at 4 / 2 bits (value rotation seed = rot_seed + 101);
   - every arm replayed teacher-forced on the full-precision answer, in one
     multi-token call.
2. **Cells.**
   - Llama-3.1-8B @128K, prompts 4000–4019, and @32K, prompts 4100–4119, both
     fresh.
   - Calibration on prompts 0–9, written to
     `r8_routes/r14s1b_llama31-8b_{131072,32768}_routes.json`.
   - At 128K the SIEVE router at B = 3 and 4 keeps Stage 1's routes.
   - The pilot (excluded) is prompt 3101.
3. **Rules.**
   - The lossless rule is §4's. The lenses are exact, 4-bit and 2-bit values.
   - D\* is the cheapest lossless dense point, S\* the cheapest over the SIEVE
     and pooled routers, H\* the cheapest hybrid, and J\* the cheaper of S\* and
     H\*.
   - **D1**, r1 = S\*/H\*: ≤ 0.90 is SIEVE_EARNS, ≥ 1.00 is HYBRID_SUFFICES.
   - **D2**, ρ = J\*/D\*, a WIN at ≤ 0.80:
     - `GO_KERNEL` if any cell is a WIN with 4-bit values;
     - `SCOPE_EXACT_V` if a WIN appears only with exact values;
     - `STOP_SYSTEMS` otherwise;
     - 2-bit values are reported, not gating.
   - **D3**, pooled against standard router at the lowest pooled budget:
     `POOL_FIXES` needs Δ ≥ +0.10, a lower bound > 0 and deletions down by
     ≥ 50%.
   - **D4**: `SEQUENCE_PROXY` needs AUC ≥ 0.85 for the worst answer token's
     log-probability, and ≥ 0.15 above the best per-head statistic.
4. **Files.**
   - `s1b_lib.py`, `run_s1b.py`, `read_stage1b.py`, `test_r14_stage1b.py`,
     `submit_s1b.slurm`, `script_stage1b.sh`.
   - No shared code was edited; `sievelib` and `run_r8` are imported.
   - CPU smoke runs on Llama-3.2-1B at 2K exercised the whole chain, and their
     outputs were deleted.
5. **Submission** was by the user from trig-login01:
   `bash h0_measurement/bugs/14_kernel_tpot/script_stage1b.sh --run`.

## 13. Decision after Stage 1b (2026-09-29)

- **`STOP_SYSTEMS` is recorded** as Stage 1b's pre-registered outcome, with the
  caveat that it was decided at the noise floor (report.md C2).
- **Stage 0's `GO_KERNEL` is superseded.** Stage 2 (§6) and Stage 3 (§7) are not
  started.
- **The next measurement tests three design ideas:**
  - a value-aware hybrid;
  - a sequence-calibrated pooled router;
  - dense storage with question-time reads.
- **Before it supports any claim, it must have:**
  - at least 40 prompts per cell, with every compared arm in one process;
  - a continuous metric: the teacher-forced increase in answer NLL, paired
    against full precision, with an interval;
  - a tail metric: the share of prompt-tasks whose answer NLL rises by more
    than 2 nats, and the share of damage in the worst 5%.
- **Its plan,** with thresholds, is written and frozen before any of its output.
  Details: report.md Part D.
