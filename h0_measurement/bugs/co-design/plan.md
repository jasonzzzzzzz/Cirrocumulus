# Co-design — two measurement columns: GQA-group allocation and the cascade score

**Plan, 2026-09-20. Nothing here is applied.** Every shared-file change below
waits for an explicit OK. Q1–Q3 (bugs/4, jobs 21444721/22/24) are queued and read
`run_h0.py` and `sievelib/` **when they start, not when they were submitted**, so
every change is additive and default-off, and §5 states what each one does to a
job that is already waiting. The only file written so far is `script.sh` beside
this one, which holds the FIVE cells back (wave 4) until these edits are synced.

Answers to: *"which designs should be added before the FIVE runs, and what does
adding them touch?"* Measured evidence behind the choice is in §1; the sheet that
carries the columns is `script.sh` wave 4.

---

## 1. Why these two, and what the code actually does

### B1 · The measured gains are per **query** head; storage is per **KV** head

Read from `run_h0.py` (lines as of today):

| what | where | per what |
|---|---|---|
| K is quantized | `:614-621` `quant.quantize_keys(K, b, …)` | **KV head**, once, shared by `n_rep` query heads |
| logits `s`, `shat[b]` | `:615-621` `logits_gqa` | query head (against the shared quantized K) |
| V read | `:658` `V[h // n_rep]` | KV head |
| evictor state, lagged scores | `:669-671` `evs[(li, h)]` | **query head** |
| bit allocation, corner keep-set, `exact_error` | `alloc.quant_metrics` | **query head**, independently |

So every gain in R3–R7 assumes each of the `n_rep` query heads sharing a KV head
may pick its **own** per-token bit-widths and its **own** kept set. A stored
tiered cache has one bit-width per (KV head, token). `n_rep` is 4 for llama31-8b,
mistral-7b, qwen3-8b; 8 for llama33-70b and (32 q / 4 kv) qwen3-30b; **1 for
qwen15-moe**. Five of six models are GQA. This is ROADMAP R12, which is currently
unmeasured and is the only open item that can shrink the headline.

### B2 · The corner has the same constraint

An H2O/TOVA keep-set in a GQA model is per KV head (attention summed over the
group). Constraining only the interior would compare a realizable interior to an
unrealizable per-head corner and **understate** the gain. The group column needs
a group corner too (§3.1).

### B3 · The gap the cascade would attack, measured

From `../2_towards_real_evictor/R3-cells.csv`, band % symmetric (`band_pp`) against
E2 (`band_pr`, oracle interior, lagged corner), same rows:

| | mean over 16 cells | range |
|---|---|---|
| E2 − symmetric, band points | **12.1** | 3.9 (70B @128k) … 24.0 (qwen3-8b @8k) |
| routed gain, symmetric vs E2 | 3.21× vs 4.77× at best | |
| interior lag cost at k=1 (R3 sweep) | median 1.05–1.20×, in-band 1.37–1.56× | grows to 1.26–2.62× at k=8 |

The largest gaps are on the qwen3 models, i.e. the ones sitting on the STOP line.
This is the ceiling for anything that adds current-query information; `E2` is
not reachable by a deployable system.

### B4 · The cascade's premise is weak, and that is why it is one column, not a project

ROADMAP R10 records that the coarse pass **identifies the head region but does not
order it** (Spearman 0.52–0.77 against 0.9). A rank correlation is not an error
ratio, so it does not say how much of the B3 gap survives, but it is the reason to
expect partial recovery at best. The plan therefore measures a bound (`cs`) and a
deployable variant (`csv`) at several base-tier widths, and decides in §8.

### B5 · V is exact in this model

The sweep quantizes K only; V is read unquantized. Group columns price the **K
bit-width and eviction** constraint. V's tier is not modelled (it would share the
same per-KV-head allocation). Stated as a limit in §10.

### B6 · Reader hazards — new column names must avoid these prefixes

| reader | line | matches | consequence of a collision |
|---|---|---|---|
| `report.py` | 383 | `startswith("gain_pp3_")` | new column pulled into the band table |
| `report.py` | 354 | `startswith("gain_e3_")` | same |
| `report.py` | 457 | `startswith("corner_bits_used3_")` | same |
| `report.py` | 571 | `c*` and `*_rel` | same |
| `boundary.py` | 114 | `startswith("gain_pp")` | pooled into the symmetric target |
| `drift.py` | 344 | `startswith("interior_lag_cost{B}_first")` | mis-read as `froz` |

All new columns use the prefixes `grp_`, `cs_`, `csv_`, `gain_grp`, `gain_cs`,
`gain_csv`, `err_wf_grp`, `err_wf_cs`, `err_wf_csv`, `err_e<B>_grp`, `err_e<B>_cs`.
None starts with a matched prefix.

### B7 · The corner tag must not change

Every guard keys on it (`or-ac_f` in bugs/2, 4, 6; `or-la-ac-wi-re_f` in bugs/7;
`or-la-fi_f` in bugs/5) and `boundary.py --evictors` filters on it. The new knobs
therefore live in the run-level config `c`, exactly as `prompt_offset` does, and
**not** in `CornerSpec`; `evict.py` is not edited. A run that carries the columns
is identified by a sidecar marker (`"group_alloc": true`), and `script.sh` wave 4
requires it (a block measured without the columns must not satisfy the guard).

---

## 2. What ships — the ladder, per head, at B = 3

`err_*` is relative output error; `gain_* = min(uniform, corner) / interior`.

| rung | interior decides from | constraint | columns | status |
|---|---|---|---|---|
| R0 uniform | — | — | `err_uniform3` | exists |
| R1 per-head oracle (the ideal) | current step, exact | per query head | `err_wf3` | exists |
| R2 per-head lagged (R3's symmetric cell) | lagged `accum` | per query head | `err_wf_pp3_accum`, `gain_pp3_accum` | exists |
| **R3 group, oracle info** | current step | **per KV head** | `err_wf_grp_or3`, `grp_cost_or3`, `gain_grp_or3` | **new** — the pure GQA cost |
| **R4 group, lagged** (realizable today) | lagged `accum` | per KV head | `err_wf_grp_pp3_accum`, `grp_cost_pp3_accum`, `gain_grp_pp3_accum` | **new** |
| **R5 cascade, bound** | current query × base-tier keys, exact `o` | per query head | `err_wf_cs3_b<bc>`, `cs_cost3_b<bc>`, `gain_cs3_b<bc>`, `gain_cs_sym3_b<bc>` | **new** |
| **R6 cascade, deployable** | current query × base-tier keys, **lagged** `o` | per query head | `err_wf_csv3_b<bc>`, `csv_cost3_b<bc>`, `gain_csv3_b<bc>` | **new** |
| **R7 group + cascade** (the likely design) | as R6 | per KV head | `err_wf_grp_csv3_b<bc0>`, `grp_cost_csv3_b<bc0>`, `gain_grp_csv3_b<bc0>` | **new** |

Group corners: `err_e3_grp_or_frac` (ranked by group-summed oracle sensitivity)
and `err_e3_grp_accum_frac` (ranked by group-summed lagged attention). Coarse-ranked
corner for the symmetric cascade cell: `err_e3_cs<bc>_frac`. Per-row stamps, only
when the feature is on: `n_rep`, `kv_head`.

Aggregate direction only, never per head: `waterfill` minimises the first-order
proxy while the reported error is exact recomputation, so a constrained allocation
can beat an unconstrained one on a single head (R3-report §2.6: 9–32% of head-rows
already violate the analogous "less information" bound). No test asserts it per
head.

---

## 3. Design

### 3.1 G — group allocation

For each KV head g with query heads H_g (|H_g| = `n_rep`), token i, tier b:

```
cost_g[i, b] = Σ_{h ∈ H_g} w²_h[i] · σ²_h[b]        (σ²_h[0] = 1, eviction)
b_g[i]       = argmin_b  cost_g[i, b] + λ · b          λ bisected to spend B·L
```

— exactly `waterfill`, with the per-token cost a sum over the group's heads
instead of one head's `w² · σ²`. `w²_h` is the oracle `(a·‖v−o‖)²` for mode `or`,
the lagged `w2p` for mode `pp`, the cascade `w²` for mode `csv`. σ²_h stays
per-head (the noise fit is per query head). Positions a lagged score has never
seen are floored at `maxb` (`waterfill_floor` semantics) for the union of the
group's unseen sets. The group **corner** keeps the top `B·L/maxb` tokens by the
group-summed score. Each head is then evaluated by `exact_error` /
`evict_error_curve` against **its own** logits, with the **shared** allocation.

`n_rep = 1` ⇒ the group is one head ⇒ every group column equals its per-head
twin **exactly**. That identity is the control (§7 A1), and qwen15-moe is the
model that carries it.

### 3.2 C — the cascade score

`shat[bc]` (quantized logits at base width `bc`, current query) is already in
hand for every head at every quantized step, because the sweep computes it for
the noise model. So:

```
ac  = softmax(shat[bc])                                   current-query attention, coarse keys
cs  : oc = ac @ V            w²_c  = (ac · ‖v − oc‖)²     exact o  -> an upper bound on the cascade
csv : op = ap @ V (lagged)   w²_cv = (ac · ‖v − op‖)²     lagged o -> no V read at the current step
```

then `waterfill` on `w²_c` (`w²_cv`); no unseen set is needed (the current step
covers every position, the fresh token included). `gain_cs_sym` ranks the
corner by the same `w²_c`, mirroring R3's `gain_pp_sym`: it isolates whether the
edge is mixed-precision *shape* or score information the corner lacks. At `bc =
maxb` the logits are exact and `cs` must reproduce `err_wf` (§7 A5).

`csv` is the deployable variant; `cs` is the bound. The systems cost (K3 read =
3/16 of K bytes per step; V read for `cs` only) is **not** measured here — this
plan prices error, not bandwidth.

---

## 4. Configurations

Knobs (run-level config, the `prompt_offset` pattern; unset ⇒ off):

| env (as `sbatch` argument) | config key | default | meaning |
|---|---|---|---|
| `SIEVE_GROUP_ALLOC=1` | `group_alloc` | off | G columns |
| `SIEVE_COARSE_BITS=3,4` | `coarse_bits` | `[]` | C columns, one set per width; must be in `bit_list` |
| `SIEVE_EXTRA_BUDGETS=3` | `extra_budgets` | `[3]` | budgets the new columns run at (subset of `budgets`); B = 3 is the headline and all 4 would multiply the cost |

Carriers (`script.sh` wave 4, marker required by its guards):

| id | cell | why it carries the columns | GPU-h today → with columns |
|---|---|---|---|
| N7 ×2 | llama31-8b @131072, five-corner blocks | n_rep 4, the ctx the phase story rests on | 5.0 → ~6–7.5 |
| N5 ×2 | llama31-8b @32768 | crosses GO | 3.5 → ~4.2–5.3 |
| N6 ×2 | qwen3-8b @8192 | crosses STOP, n_rep 4, largest E2 gap | 3.0 → ~3.6–4.5 |
| N14 | qwen3-30b @8192 (n=8, offset 4) | n_rep 8, largest E2 gap | 8.0 → ~9.6–12 |
| N11 | qwen15-moe @16384, LEAN | **n_rep = 1 control** | 0.75 → ~0.9–1.1 |
| P1–P3 | pilots, 1 prompt, §7 | validate before any of the above | ~0.4 (+~3 optional) |

The pass count per (head, quantized step) goes from about 24 (LEAN) / 36 (FIVE) to
+12 at one budget: ~8 `exact_error` and ~4 `evict_error_curve` (cascade: 2 widths
× 2 `exact_error` + 2 coarse-ranked corner curves; group: 3 `exact_error` + 2
group corner curves). That is **+33–50% of the metrics passes**, an upper bound
on wall time because prefill and the bit sweep are unchanged. The estimate is
**+20–50%**; P1's s/unit replaces it and the wave-4 walltimes are rescaled before
submission. Total added: **~4–10 GPU-h**.

---

## 5. Shared code — blast radius

**Invariants every edit must keep**

| | invariant | how it is pinned |
|---|---|---|
| I1 | flags unset ⇒ `head_metrics` returns a **bit-identical dict**, run rows gain **no** column, the sidecar gains **no** key | golden-dict test on the synthetic head (§6 T1); CPU smoke diff (§7) |
| I2 | **mixed-version safe**: `run_h0.py` passes the new kwargs **only when a knob is on**; `alloc.py`'s new kwargs default to `None`. Old `alloc` + new `run_h0`, and new `alloc` + old `run_h0`, both behave as today for a default job | T2 (call with and without kwargs); a job that starts mid-sync cannot break |
| I3 | corner tag / `config_record` unchanged | `evict.py` not edited; T3 asserts `corner_tag` for the five shipped configs |
| I4 | no existing statement reordered or rewritten; new code is appended or in new functions | diff review; the only line changes inside existing code are listed below |
| I5 | flag on + any failure in the new path ⇒ **loud** `SystemExit`, never a NaN column | T5 — the "silent empty column" class already cost three campaigns (bugs/2 P0) |

**Proposed edits** (none applied)

| id | file | change | LOC | blast radius on a **queued** job (default, no knob) | on a knob-on job |
|---|---|---|---|---|---|
| **S1** | `sievelib/alloc.py` | new `waterfill_group(w2_stack, sig2_list, budget, maxb, floor)` (§3.1) | ~35 | none: nothing calls it | new path |
| **S2** | `sievelib/alloc.py` | new `group_allocations(...)`: per KV group → interior bits per mode + group corner rankings | ~50 | none: not called | new path |
| **S3** | `sievelib/alloc.py` | new `extra_metrics(s, shat, V, base_out, *, coarse_bits, extra_budgets, group, …)` returning the §2 columns **from the dict `quant_metrics` already produced** (`err_wf`, `err_uniform`, `err_practical`, `err_wf_pp_*`) — so **`quant_metrics` itself is not edited**; plus `head_metrics` gains two optional kwargs and a 3-line `if extra is not None: m.update(extra_metrics(...))` after the existing `m.update(quant_metrics(...))` | ~120 + 5 | the 5 lines in `head_metrics` are the **only** edit inside a function a queued job executes; both kwargs default `None`, so the branch is not taken | new columns |
| **S4** | `h0_measurement/run_h0.py` | (a) read the three knobs from `c` (default off); (b) **only if on**: a per-layer pre-pass before `for h in range(s_all.shape[0])` that, per KV group, builds the group inputs and calls `group_allocations` once; (c) pass `extra=` to `head_metrics` only if on; (d) stamp `n_rep`, `kv_head` per row and `group_alloc`/`coarse_bits`/`extra_budgets` in the sidecar **only if on**; (e) validate `coarse_bits ⊂ bit_list`, `extra_budgets ⊂ budgets`, and that `fin` is identical across a group's heads, at startup / first layer | ~90 | (a) is one `c.get` per knob; (b)–(d) sit behind `if group_alloc or coarse_bits:`. **This is the highest-risk edit**: a Python syntax/NameError in the file breaks *every* job that starts afterwards, queued or not. Mitigation §7 steps 1–3 | pre-pass runs the evictors' `score()` a second time per step; safe by the existing regression "a second `score()` in one step must not move the state", re-pinned by T4 |
| **S5** | `submit_h0.slurm`, `submit_h0_large_models.slurm` | add the three names to the existing `for … SIEVE_PROMPT_OFFSET; do` forward list (`:159` / `:178`), the `OVERRIDES+=` lines (`:170` / `:189`), line-1 echo and RUN_INFO (only when set) | ~12 each | Slurm keeps the batch script it was given at submission, so **queued jobs run their spooled copy** — to confirm: `scontrol write batch_script <jobid> -` on one of 21444721/22/24. Even if it did re-read, unset names add nothing | pattern-checked like `SIEVE_PROMPT_OFFSET` |
| **S6** | `tests/test_units.py` | T1–T6 (§6) | ~150 | none (tests are not run by jobs) | — |
| **S7** | `co-design/script.sh` | wave 4 + `have_block`'s `NEED_GROUP` marker check | done | new file, none | — |
| **S8** | `co-design/gqa_cascade.py` (new) | the reader: per-rung medians, the ladder, `closed` fraction, group band, dose-response in `n_rep` | ~200 | new file, 0 GPU | — |

**Flagged, not changed** (each is a reader whose output is being quoted now, or
a file no edit here needs)

| file | why it stays |
|---|---|
| `evict.py` | `CornerSpec` / `corner_tag` / `config_record` untouched ⇒ I3 by construction |
| `report.py`, `boundary.py`, `errorbars.py`, `drift.py` | column prefixes avoided (B6); nothing to change. If the phase figure later takes the **group band** as its y-axis, `boundary.py` needs a `--target` for it — after R6's own read |
| `models.yaml` | no default added; absent keys mean off |
| `bugs/4/script.sh` (queued Q1–Q3) | untouched; do not re-run it (unguarded) |

**Sync procedure** — `run_h0.py` and `alloc.py` must land together, atomically:

```
cp sievelib/alloc.py sievelib/alloc.py.new && cp h0_measurement/run_h0.py h0_measurement/run_h0.py.new
python -c "import sievelib.alloc, importlib.util"     # import check on the .new copies first
mv sievelib/alloc.py.new sievelib/alloc.py && mv h0_measurement/run_h0.py.new h0_measurement/run_h0.py
```

Order is irrelevant for a default job because of I2. The slurm scripts and
`co-design/script.sh` follow.

---

## 6. Tests (`tests/test_units.py`, S6)

| id | pins |
|---|---|
| **T1** | golden: `head_metrics` on the existing synthetic head with the new kwargs absent returns a dict equal (keys **and** values) to the pre-edit output; with them present, a strict superset |
| **T2** | I2: `head_metrics(..., extra=None)` and the pre-edit signature agree; `run_h0`'s call site builds no new kwarg when both knobs are off (AST/inspect check on the built kwargs) |
| **T3** | I3: `corner_tag` and `config_record["tag"]` for `or-ac_f`, `or-la-ac-wi-re_f`, `or-la-fi_f`, `oracle`, `oracle,accum,last_step` unchanged |
| **T4** | `group_allocations`: (i) `n_rep=1` ⇒ group bits == per-head `waterfill_floor` bits, and every group column equals its per-head twin; (ii) `n_rep=4` with four identical heads ⇒ same; (iii) the allocation is identical across the group by construction; (iv) budget-matched to `waterfill`'s own tolerance; (v) unseen positions floored; (vi) a second `score()` per step does not move evictor state with the pre-pass on |
| **T5** | I5: `coarse_bits` not in `bit_list`, `extra_budgets` not in `budgets`, and non-identical `fin` across a group each raise at start-up / first layer with a message naming the knob |
| **T6** | `extra_metrics` cascade: at `bc = maxb` `err_wf_cs ≈ err_wf` (median over synthetic heads within 2%); `csv ≥ cs` in aggregate direction only; `gain_cs_sym` uses the same `w²_c` ranking |
| **T7** | reader hazards (B6): no new column name starts with any prefix in the table |

Existing tests that must still pass unmodified: `test_practical_interior`,
`test_unseen_floor`, `test_first_evictor`, `test_rescore_is_idempotent`,
`test_corner_provenance`, `test_prompt_offset`, `test_decode_plan` — `script.sh`
already gates on all of them.

---

## 7. Pilots and acceptance — before wave 4

**Pre-sync, on CPU (the R5 path: Llama-3.2-1B, a real PG-19 prompt):**

1. Full unit suite green.
2. Default run, flags **off**, old code vs new code, same seed, **byte-identical**
   parquet (CPU is deterministic; GPU is not — R3-report finding 5 records up to
   80% per-head nondeterminism, so this check is CPU-only).
3. Flags **on**: columns present, all §2 identities hold (A1, A2, A5 below).

**On the cluster, after the sync** (`LEAN` config + `SIEVE_GROUP_ALLOC=1
SIEVE_COARSE_BITS=2,3,4,6,8 SIEVE_EXTRA_BUDGETS=3`, 1 prompt, 3 units):

| pilot | cell | cost | purpose |
|---|---|---|---|
| P1 | llama31-8b @8k | ~0.3 GPU-h | first real GQA numbers; s/unit for the walltime rescale; the base-width curve (2/3/4/6/8) |
| P2 | qwen15-moe @8k | ~0.1 GPU-h | the n_rep=1 identity on real attention |
| P3 (optional) | llama33-70b @8k, large | ~3 GPU-h | n_rep=8; only if P1 shows `grp_cost_pp` above ~1.15 |

| check | must hold | if it fails |
|---|---|---|
| **A1** identity | P2: every `gain_grp_*` equals its per-head twin, per row, to float tolerance | bug in S2/S3 — do not read anything else |
| **A2** budget | every group allocation spends `B·L` to `waterfill`'s tolerance (3.0005 b/token seen in T4) | bug in S1 |
| **A3** log | line 1 echoes the three knobs; sidecar carries `group_alloc: true`, `coarse_bits`, `extra_budgets`; `interior_unseen_policy: floor_maxb` still present | S5 not synced — `scancel` |
| **A4** order | aggregate: median `grp_cost_or3` ≥ 1.0 (a constraint cannot help in aggregate) | if < 1, the proxy/exact mismatch is larger than assumed — say so, do not hide |
| **A5** cascade exact | P1: median `err_wf_cs3_b8 / err_wf3` within 2% of 1.0; `b6` within ~5% | S3 mis-indexes `shat` |
| **A6** cost | s/unit within the +20–50% estimate; rescale wave-4 headers by the measured factor | as measured |
| **A7** fidelity | `cs_cost` non-increasing in `bc` in aggregate | mis-ordered widths |

---

## 8. Decision table, written before the run

**G — is per-head routing realizable?** Read on **in-band** heads
(`gain_pp3_accum ≥ 2`), because out-of-band heads never use the interior
(R3-report §2.2: lag cost 1.00–1.04× there).

| outcome | reading |
|---|---|
| median `grp_cost_pp3_accum ≤ 1.15` and group band within ~3 pts of `band_pp` | **Realizable.** C4's per-head router stands as written; report the group band beside the per-head band |
| 1.15–2.0×, or group band 50–90% of the per-head band | **Realizable at a discount.** Every headline gain shrinks by that factor; report the **group** band as the paper's y-axis, and re-fit the phase law on it (dead-2 is unchanged) |
| > 2.0×, or group band < 50% of the per-head band | **Per-head routing is partly fictional** for GQA models (ROADMAP R12's stated threat). The router must decide per **KV** head from group-summed w²; R8 must be run with the group allocation, or it measures something a cache cannot store |
| `grp_cost` grows with `n_rep` (1 → 4 → 8: qwen15-moe, llama31-8b/qwen3-8b, qwen3-30b) | the mechanism is head-heterogeneity inside a group — the dose-response is the figure |
| `grp_cost_or3 ≈ 1` but `grp_cost_pp3 ≫ 1` | the loss is an interaction of lag and grouping, not grouping alone — heads in a group differ in *where their attention moves* |

**C — is the cascade worth building?** `closed = (interior_lag_cost − cs_cost) /
(interior_lag_cost − 1)`, median over in-band heads, per `bc`.

| outcome | reading |
|---|---|
| `closed(csv, bc=3) ≥ 0.5` | **Build it.** Half of the lag penalty goes away using only the base tier; then price the systems cost (K3 read each step) against re-budgeting more often |
| 0.2–0.5 | **Hybrid or not at all**: blend lagged and coarse, or spend the base tier only on in-band heads |
| < 0.2 | **Drop it.** The lag-only design with re-budgeting every few steps (R3 §2.3) stands; the 12-pt gap is not reachable from the base tier |
| `closed(cs)` high but `closed(csv)` low | the value term `‖v − o‖` needs the current `o`, i.e. a V read — the deployable cascade is not the bound |
| `gain_cs_sym` ≈ `gain_cs` and both ≈ `gain_pp_sym` | the corner gets the same information gain, so the edge is **shape, not information** — R3's central claim, now with the cascade's information given to both sides |
| `closed` rises steeply between bc = 2 and 4 | the base layer's width is a design knob; R10's "3-bit base layer" is then a measured choice |

**Both together** (`gain_grp_csv3_b*` vs `gain_pp3_accum`): the distance between
today's honest number and the design a cache can actually store. That ratio is what
R8 should be expected to recover, replacing the E2-derived expectation in R6 §7.

---

## 9. Effort and schedule

Estimates from reading `alloc.py`'s whole `quant_metrics`, `run_h0.py:575-760` and
the sidecar block, and the reader prefixes; **not** from having run any of it, so
±2×. They are higher than the first estimates given in conversation (2–4 days
group, 1–2 days cascade) because §B2 adds a group corner.

| step | days | note |
|---|---|---|
| S1 + S2 (`waterfill_group`, `group_allocations`) | 0.75 | a `waterfill` with a per-token cost matrix |
| S3 (`extra_metrics`, `head_metrics` hook) | 1.0 | cascade is ~40 of the ~120 lines |
| S4 (`run_h0.py` pre-pass, stamps, validation) | 1.5 | the risky one; only place the head loop is touched |
| S5 (two slurm scripts) | 0.25 | mechanical, the `prompt_offset` pattern |
| S6 (T1–T7) | 1.0 | T4 and T1 are the load-bearing ones |
| CPU smoke (§7 steps 1–3) | 0.5 | |
| S8 (reader) | 0.5 | 0 GPU, can overlap S1–S5 |
| **total, both** | **~5.5 days** | **G alone ~3.5–4; C alone ~2** (they share S4–S6) |
| pilots + read | 0.5 | ~0.4 GPU-h |

Order: S6/T1–T3 first (they fix the invariants before any code moves), then S1–S3,
S4, S5, CPU smoke, sync, pilots, wave 4.

**If time is short:** ship G alone. It is the one that can change the headline
and needs no assumption about the base tier. C is one more column set once S4 exists.

---

## 10. Limits and open questions

- **V is exact** in this model (B5); group columns price K bits and eviction only.
- **Group corner = accum only** (summed over the group). H2O/SnapKV variants differ
  (max vs sum; per-KV-head vs per-group budgets); the five-corner min is not
  reproduced at group level, so `gain_grp_*` is "against the accum corner" like
  every d\* in R6.
- **Group-summed score for eviction** assumes the cache keeps one token set per KV
  head, which is how the shipped GQA cache layouts work; a per-query-head cache
  would need `n_rep`× the memory and is not a deployment target.
- **`fin` identical across a group's heads** — verified by reading: the mask is one
  `[L]` vector per layer (`probe.py:99`, `attention_mask[0, 0, 0]`) added to every
  head (`run_h0.py:623-627`), so the finite set can differ between heads only if a
  logit is itself non-finite. S4 still asserts it on the first layer and stops
  loudly, because a violation would be structural.
- **Cost model is a bound, not a benchmark.** Bytes read per step and kernel time
  are R14; this plan reports error.
- **Slurm spooling** (§5 S5) is a standard behaviour I have not confirmed on this
  cluster; the `scontrol` check takes a minute.
- **One prompt** in the pilots is enough for identities and s/unit, not for a
  statistic. All readings come from the wave-4 blocks (4–6 prompts each,
  `errorbars.py`-compatible).

---

## 11. Files here and dependencies

| file | state |
|---|---|
| `plan.md` | this file |
| `script.sh` | written, dry-run clean; waves 1–3 submit today; **wave 4 refuses to run until S4/S5 are synced** and its guards require `group_alloc: true` |
| `gqa_cascade.py` | to write (S8), after S1–S5 |

- **R7 — carrier.** N5–N7, N14 are R7's blocks; `errorbars.py` is unaffected (B6).
  The columns exist only on those blocks, not on R3's reference block.
- **R3 — interpretation.** R3's `interior_lag_cost` is the baseline the cascade is
  measured against; R3's `gain_pp3_accum` is rung R2.
- **R6 — soft.** If the group band replaces the per-head band on the y-axis, the
  fitted law is re-fitted, not re-run (dead-2 is unchanged).
- **R8 — hard for interpretation.** The router-on/off design must use the
  allocation a cache can store; §8 decides whether that is per-head or per-KV-head
  and which score it runs on. Run R8 after this.
- **R4, R5 — none.** Different decode shape / different readout; wave 1–3 do not wait.
