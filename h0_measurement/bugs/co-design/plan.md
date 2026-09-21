# Co-design — two measurement columns: GQA-group allocation and the cascade score

> **Status 2026-09-20: S1–S6 are WRITTEN AND APPLIED on this machine, and the CPU
> smoke test passed (§7.0). Nothing is synced to the cluster and nothing is
> submitted.** The sync in §5 is the user's step, and §7's cluster pilots (P1, P2)
> come before wave 4. Waves 1–3 of `script.sh` carry no co-design knob (verified:
> 13 cells, zero occurrences) and are unaffected.

**Plan, 2026-09-20.** Every shared-file change below was made only after an
explicit OK. Q1–Q3 (bugs/4, jobs 21444721/22/24) are queued and read
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

Column names below are **as emitted** (verified against a real run: 44 new
columns at `group_alloc=1 coarse_bits=3,4`, 69 at four widths). `<s>` is the
interior score, `accum` everywhere here; `<bc>` a base-tier width.

| rung | interior decides from | constraint | columns | status |
|---|---|---|---|---|
| R0 uniform | — | — | `err_uniform3` | exists |
| R1 per-head oracle (the ideal) | current step, exact | per query head | `err_wf3` | exists |
| R2 per-head lagged (R3's symmetric cell) | lagged `accum` | per query head | `err_wf_pp3_<s>`, `gain_pp3_<s>`, `interior_lag_cost3_<s>` | exists |
| **R3 group, oracle info** | current step | **per KV head** | `err_wf_grp_or_3`, `grp_or_cost3`, `gain_u_grp_or_3` | **new** — the pure GQA cost |
| **R4 group, lagged** (realizable today) | lagged `accum` | per KV head | `err_wf_grp_pp_<s>_3`, **`grp_pp_<s>_over_head3`**, `grp_pp_<s>_cost3`, `gain_grp_pp_<s>_3` | **new** |
| **R5 cascade, bound** | current query × base-tier keys, exact `o` | per query head | `err_wf_cs_b<bc>_3`, `cs_b<bc>_cost3`, `gain_cs_b<bc>_3`, `gain_cs_sym_b<bc>_3` | **new** |
| **R6 cascade, deployable** | current query × base-tier keys, **lagged** `o` | per query head | `err_wf_csv_b<bc>_<s>_3`, `csv_b<bc>_<s>_cost3`, `gain_csv_b<bc>_<s>_3` | **new** |
| **R7 group + cascade** (the likely design) | as R6 | per KV head | `err_wf_grp_csv_b<bc>_<s>_3`, **`grp_csv_b<bc>_<s>_over_head3`**, `gain_grp_csv_b<bc>_<s>_3` | **new** |

**`*_over_head3` is the R12 number, `*_cost3` is not.** `cost` divides by
`err_wf` and so carries R3's lag cost on top of the grouping; `over_head`
divides by the matching per-head allocation and isolates the grouping alone
(§7.0b). `gain_u_grp_or_3` is named apart because the `or` rung has no
deployable corner to compare against and is reported against uniform.

Group corners: `err_e3_grp_or_frac` (ranked by group-summed oracle sensitivity)
and `err_e3_grp_pp_<s>_frac` (group-summed lagged attention) — the latter is the
competitor every group `gain_` uses. Coarse-ranked corner for the symmetric
cascade cell: `err_e3_cs<bc>_frac`. Also emitted: `evict_frac_*` per rung, and
the stamps `n_rep`, `kv_head`, `grp_size` — only when the feature is on.

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
| **S1** ✅ | `sievelib/alloc.py` | new `waterfill_group(...)` (§3.1). **G = 1 delegates to `waterfill`/`waterfill_floor`** so the n_rep = 1 control is exact by construction: the group form is the same argmin multiplied through by w², and `a*b/a != a` in IEEE754 parted the two by one tier (4.5e-6 in the reported error) when it re-derived | ~50 | none: nothing calls it | new path |
| **S2** ✅ | `sievelib/alloc.py` | new `group_allocations(...)`: per KV group → interior bits per mode + group corner rankings | ~50 | none: not called | new path |
| **S3** ✅ | `sievelib/alloc.py` | new `extra_metrics(s, shat, V, base_out, *, coarse_bits, extra_budgets, group, …)` returning the §2 columns **from the dict `quant_metrics` already produced** (`err_wf`, `err_uniform`, `err_practical`, `err_wf_pp_*`) — so **`quant_metrics` itself is not edited**; plus `head_metrics` gains two optional kwargs and a 3-line `if extra is not None: m.update(extra_metrics(...))` after the existing `m.update(quant_metrics(...))` | ~120 + 5 | the 5 lines in `head_metrics` are the **only** edit inside a function a queued job executes; both kwargs default `None`, so the branch is not taken | new columns |
| **S4** ✅ | `h0_measurement/run_h0.py` | (a) read the three knobs from `c` (default off); (b) **only if on**: a per-layer pre-pass before `for h in range(s_all.shape[0])` that, per KV group, builds the group inputs and calls `group_allocations` once; (c) pass `extra=` to `head_metrics` only if on; (d) stamp `n_rep`, `kv_head` per row and `group_alloc`/`coarse_bits`/`extra_budgets` in the sidecar **only if on**; (e) validate `coarse_bits ⊂ bit_list`, `extra_budgets ⊂ budgets`, and that `fin` is identical across a group's heads, at startup / first layer | ~90 | (a) is one `c.get` per knob; (b)–(d) sit behind `if group_alloc or coarse_bits:`. **This is the highest-risk edit**: a Python syntax/NameError in the file breaks *every* job that starts afterwards, queued or not. Mitigation §7 steps 1–3 | pre-pass runs the evictors' `score()` a second time per step; safe by the existing regression "a second `score()` in one step must not move the state", re-pinned by T4 |
| **S5** ✅ | `submit_h0.slurm`, `submit_h0_large_models.slurm` | add the three names to the existing `for … SIEVE_PROMPT_OFFSET; do` forward list (`:159` / `:178`), the `OVERRIDES+=` lines (`:170` / `:189`), line-1 echo and RUN_INFO (only when set) | ~12 each | Slurm keeps the batch script it was given at submission, so **queued jobs run their spooled copy** — to confirm: `scontrol write batch_script <jobid> -` on one of 21444721/22/24. Even if it did re-read, unset names add nothing | pattern-checked like `SIEVE_PROMPT_OFFSET` |
| **S6** ✅ | `tests/test_units.py` | T1–T6 (§6) | ~150 | none (tests are not run by jobs) | — |
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

### 7.0 Done — S1–S6 are written and the CPU smoke test passed (2026-09-20)

Run on this machine, not the cluster. `qwen3-1.7b` (28L, **16 query heads / 8 KV
heads, so n_rep = 2**), ctx 2,048, 1 prompt, `cont`, dense 2 steps, float32 CPU,
`oracle,accum` + interior `accum` — the identical command twice, knobs off and on.

| check | result |
|---|---|
| **I1 at run level** — every shared numeric column, quantized rows | **225 / 225 bit-identical** between the knobs-off and knobs-on runs |
| **I1 vs a real campaign** — columns of a knobs-off run vs `job21421769` | identical but for `prompt_offset`, `rot_seed`, `decode_rep_penalty`, `decode_no_repeat_ngram`, all of which landed with R5/R7 **before** this work. **No co-design column appears.** |
| sidecar marker | `group_alloc/coarse_bits/extra_budgets` present only with the knobs on; absent otherwise, so the wave-4 guard discriminates |
| columns added | 69 with `group_alloc=1 coarse_bits=2,3,4,8`; none lost |
| **A5** cascade at `bc = maxb` | `cs_b8_cost3 = 1.000` on real attention |
| **A2** budget | every group allocation budget-matched to `waterfill`'s one-tier-step tolerance |
| **A6** cost | 31 s → 48 s wall (**+55 %**) with **four** coarse widths + group; the wave-4 config uses two, so the +20–50 % estimate stands. Re-measure with P1. |
| unit tests | T1–T7 green, and every gate test the `bugs/*` sheets run (`test_practical_interior`, `test_unseen_floor`, `test_first_evictor`, `test_rescore_is_idempotent`, `test_corner_provenance`, `test_prompt_offset`, `test_decode_plan`, `test_override_lists`, `test_ban_eos`) still green |

### 7.0b Review pass — three bugs found in the first implementation, all fixed

Found by re-reading the code against what the columns are supposed to *mean*,
before any batch ran. Each is pinned by a test that fails on the old behaviour.

| # | bug | why it mattered | fix |
|---|---|---|---|
| **1** | the group allocator summed **raw** `w2` across a KV group, i.e. the sum of **absolute** squared errors, while `exact_error` reports **relative** error | the head with the largest `‖o‖` captured the shared allocation and its neighbours were starved — in exactly the per-head relative numbers the band counts. Measured: scaling one head's V by 1000 (which changes no relative error at all) moved **574 of 1024 tokens**; after the fix, 1 | `_rel()` divides each head's `w2` by `‖o_h‖²` before any cross-head sum, making the group objective the sum of *relative* squared errors. Skipped at n_rep = 1, where a rescale cannot change the argmin but *can* move the finite bisection by one tier — that keeps the control bit-exact. T4g |
| **2** | `gain_*` silently fell back to comparing against uniform when its corner was absent | a lagged corner has no history on early decode steps, so the column would change denominator between rows and any reader would average the two. `quant_metrics` withholds `gain_best_practical` for exactly this reason | the `gain_` column is withheld; the uniform-only variant is emitted under its own name `gain_u_grp_or_<B>`. The `cost` column, which needs only `err_wf`, is always emitted |
| **3** | the group and cascade corners assumed the `frac` policy without checking it was configured | a run with `abs` only would get columns named `_frac` computed under a policy the rest of the run never used | corner columns are withheld unless `frac` is in `corner.policies` |

**A fourth thing that was not a bug but was being reported wrongly.**
`grp_pp_*cost3` divides by `err_wf`, so it carries the **lag** cost as well as
the grouping. The lag tail is heavy and is R3's measurement, not R12's: at
n_rep = 2 the conflated ratio reads median 1.165 / max 124×, while the
grouping's *own* marginal cost reads **1.005 / max 4.99×**, and
`corr(log grp_pp, log per-head lag) = 0.919`. A new column
`grp_<name>_over_head<B>` divides by the matching per-head allocation, and
`gqa_cascade.py` leads with it.

**First real-attention numbers — a pilot, not a result.** One debug-tier model, one
prompt, ctx 2,048, 448 head-rows, band 9.8 % (near STOP), and **n_rep = 2, the
smallest real GQA ratio in the registry**. Every headline cell is n_rep 4 or 8.

| quantity | all heads | in band (n=44) |
|---|---|---|
| per-head lagged interior (R3's number) | 1.131× | 1.401× |
| **group's own marginal cost** `grp_pp_accum_over_head3` | **1.005×** | **1.064×** |
| group, oracle information `grp_or_cost3` | 1.016× | 1.150× |
| cascade `cs / csv` at bc = 3 | 1.054 / 1.055× | 1.214× |
| cascade at bc = 4 | 1.021 / 1.020× | 1.080× |
| gap closed, in-band, bc = 2 / 3 / 4 | | **−0.43 / +0.31 / +0.81** |

Four things this already says, each to be re-tested at n_rep 4 and 8:

1. **The group constraint is nearly free at n_rep = 2** (1.6 % under oracle
   information, and it adds ~1.7 % on top of lag). The synthetic worst case in
   development was 4.16×, so real heads inside a KV group are far more alike than
   independent draws. If this survives n_rep 8, R12 stops being a threat to C4.
2. **The deployable cascade costs nothing against its own bound**: `csv ≈ cs` to
   three decimals at every width, so the lagged `o` is as good as the exact one and
   the current-step V read is not needed.
3. **A 2-bit base tier is worse than doing nothing** (−0.43: the coarse score is a
   worse allocator than last step's attention), 3-bit is marginal, 4-bit closes
   81 %. That is a sharper statement of R10's "the base layer moves to 3 bits" —
   for the cascade, 3 bits is the edge of useful.
4. **The group band went UP, 9.8 % → 13.2 %** — because the group corner is
   constrained too (B2), and the constraint costs the *corner* more than the
   interior. Real, and it makes the design look better, but it means
   `gain_grp_pp` and `gain_pp3_accum` are **not the same statistic**: §8 must
   compare group-to-group, and the paper must report both, named.

### 7.0c Wave-4 readiness pass (2026-09-20, after waves 1–3 landed)

Four things found by testing wave 4's **actual** config rather than LEAN, and by
re-deriving its walltimes from wave 1's measured rates.

| # | finding | fix |
|---|---|---|
| **1** | **FIVE+ had never been run.** §7.0 tested LEAN + the columns; wave 4 runs the five-corner set. Now tested end to end on qwen3-1.7b: corner tag still `or-la-ac-wi-re_f`, marker present only when on, **261/261 shared numeric columns bit-identical**, +44 columns, group marginals 1.008–1.020. Column overhead **+35%** (34 s → 46 s) with two coarse widths, against +55% with four. | none needed — it works |
| **2** | **The two bands used different corner sets.** The group corner is ranked by the group-summed *interior* score, i.e. `accum` alone, while the five-corner per-head competitor `err_practical<B>` is a min over five. The group side got the easier corner — the same class of error as B2, pointing the other way. | `gqa_cascade.py` now reconstructs the **accum-only** per-head corner exactly (R7's identity) and prints both bands matched. On the test cell: 10.5% vs 12.1% matched, against 10.3% mismatched. The R12 headline is untouched — it is a ratio of two interiors with no corner in it. |
| **3** | **Three of five cells would have overrun and lost everything.** Re-derived from wave 1's measured LEAN rates × (L)^0.3 × 1.54 (FIVE/LEAN) × 1.35 (FIVE+/FIVE): N7 needed 173 min against a 150-min header, N5 116 against 105, N14 173 against 120. `run_h0` writes its parquet once at the end. | N7 → 03:15, N5 → 02:15, N6 → 01:45, N14 → 01:45 per block |
| **4** | **N14 was one 24-unit job.** It was merged to buy a single 4-GPU queue wait; X1 moved the cell to one GPU, so that saving vanished while the risk doubled. | split into N14a/N14b, one per 4-prompt block — R7 §4's own rule |

**n_rep coverage is complete.** Wave 4 spans the whole dose-response axis:

| cell | model | heads | **n_rep** |
|---|---|---|---|
| N11 | qwen15-moe-a2.7b | 16 q / 16 kv | **1** (the exactness control) |
| N5, N7 | llama31-8b | 32 q / 8 kv | **4** |
| N6 | qwen3-8b | 32 q / 8 kv | **4** |
| N14 | qwen3-30b-a3b-2507 | 32 q / 4 kv | **8** |

So §8's "does the group cost grow with n_rep" test is answerable from wave 4
alone, at 1 / 4 / 8 — which is what turns a number into a mechanism.

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

**Read `grp_pp_accum_cost3`, not the band, as the primary.** §7.0 found the group
band is *higher* than the per-head band (13.2 vs 9.8) because the group corner is
constrained too: the two bands use different competitors and are not the same
statistic. The cost ratio has one meaning; the bands must be compared
group-to-group and both reported by name.

| outcome | reading |
|---|---|
| median `grp_pp_accum_cost3 ≤ 1.15` at n_rep 4 **and** 8 | **Realizable.** C4's per-head router stands as written; report the group band beside the per-head band. This is what n_rep = 2 already shows (1.150×) |
| 1.15–2.0× | **Realizable at a discount.** Every headline gain shrinks by that factor; report the **group** band as the paper's y-axis, and re-fit the phase law on it (dead-2 is unchanged) |
| > 2.0× | **Per-head routing is partly fictional** for GQA models (ROADMAP R12's stated threat). The router must decide per **KV** head from group-summed w²; R8 must be run with the group allocation, or it measures something a cache cannot store |
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
| S4 (`run_h0.py` pre-pass, stamps, validation) | 1.5 | the risky one; only place the head loop is touched. **Found while writing it:** `coarse_bits=3,4` arrives from `--override` as a STRING, so iterating it yields `'3', ',', '4'` — the same defect `load_cfg`'s comment already records for `families=cont`. Both new keys joined that coercion list |
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
| `script.sh` | written, dry-run clean (S7 ✅). Waves 1–3 submit today and carry no co-design knob; `--pilot` runs P1/P2; **wave 4 refuses to run until S4/S5 are synced** and its guards require `group_alloc: true` |
| `gqa_cascade.py` | **written (S8)** ✅ — 0 GPU. Leads with the grouping's marginal cost and the in-band column, prints the §8 verdict for both designs, and refuses cleanly on parquets without the columns |
| `tests/golden_head_metrics.json` | written — T1's golden, captured from the code **before** S1–S3. Do not regenerate it from edited code, or T1 becomes tautological |

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
