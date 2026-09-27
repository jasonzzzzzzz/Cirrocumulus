# Roadmap: remaining experiments, ranked

Written after the E1/E2 campaign (`bugs/2_towards_real_evictor/report.md`).
Supersedes the "Remaining experiments" list in proposal v8 §7, which was drawn up
before that campaign landed and still lists items it completed or refuted.
**The status board below is current as of 2026-09-27;** the sections after it are
the original ranked plan, each headed by its current status.

## Status board (2026-09-27)

Each part's working folder is `h0_measurement/bugs/<N>_*/`; its report there is
the source of truth. "Done" means measured and written up, not merely submitted.

| part | question | status | result in one line | where |
|---|---|---|---|---|
| R1 | reconcile documents, demote τ/ln2 | **open** (0 GPU) | proposal v8 / pitch v5 (`docs/*-h0v2.html`, 2026-09-17) predate R3–R8 and do not carry their results | `docs/` |
| R2 | robust statistic + partial ρ | done (marked ✓ in the graph below) | routed gain is the robust y-axis; band kept secondary | R6 report uses it |
| R3 | symmetric cell (interior and corner both lagged) | **done** (job214217\*) | edge survives at ~half size; **5/16 cells STOP**; ρ(dead-2, band) = −0.985 | `bugs/2_towards_real_evictor/R3-report.md` |
| R4 | 128k drop: attention or RoPE limit? | **done** (8 cells + co-design N16) | mostly ABSOLUTE-L: τ is convex in log L; a real **+0.21 τ excess at rope_frac 1.00** remains | `bugs/4_rope_limit_or_mechanism/report.md` §8–9, `bugs/co-design/report.md` §7.6 |
| R5 | does a head's phase drift over 4,096 decode steps? | **done** for C4; τ-drift sub-question **open** | two timescales: the **route** is one-pass (p90 regret 1.00, 6 cells), the **allocation** goes stale (2.5–4.3×); within-generation Δτ underpowered (needs ~20 prompts/cell) | `bugs/co-design/report.md` §3, §7.7 |
| R6 | pin the GO/STOP boundary | **done** (25 cells) | pooled crossings GO 29.0% / STOP 54.1% dead-2; ρ = −0.978 (−0.915 partialled on model); but **pinned per model, not universal** — an 8-pt band gap across 0.6 pts of dead-2 between two architectures | `bugs/6_pin_sharp_boundary/report.md`, `bugs/co-design/report.md` §4 |
| R7 | error bars | **done** (wave 4) | prompt-block sd 0.4–3.0 band pts; **the corner set moves the band −5 to −9 pts** — name it | `bugs/co-design/report.md` §7.5 |
| co-design | GQA group allocation + cascade score | **done** (waves 1–4) | per-KV-head allocation costs **1.00 / 1.14 / 1.32×** at n_rep 1/4/8 (answers R12); cascade at bc = 4 closes **89%** of the lag gap | `bugs/co-design/report.md` §7; per-head data `wave1.csv`…`wave4.csv` |
| **R8** | router on/off on an end task | **done; gate failed** | 5/5 cells complete; P-1/P-2/P-5 fail, P-4 is unsupported, and the budget gate failed because uniform is ≥0.95 in 31/36 valid cells | `bugs/8_router_endtask/report.md` |
| R9 (roadmap) | K*-derived KV-head budget | **done; stopped at qualification** | job 985183 passed implementation, stability, and resolution, but moved only 0.2319% of Llama budget versus the frozen 5% gate; no development lock | bugs/10_kstar_budget/report.md |
| R9 (folder) | SOTA baselines and end-task mechanism diagnosis | **done; V7 stopped at qualification** | uniform wins 34/36 baseline cells; V5/V6 candidate oracles add only 3/52 and 2/52; V7 job 984886 failed its frozen FP and bootstrap competence gates before running the structured-query mechanism | `bugs/9_sota_eviction_baselines/report.md`, `plan.md` §3L |
| **R10** | tier set and 3-bit base | **done** (job 986347) | `{0,3,4,6,8}` (3+1+2+2) passes all frozen gates at B=3 and B=2 (worst cell error 1.031 / 1.070 of full); `{0,4,6,8}` rejected (up to 1.26×); 1-bit tier dead (≤0.1% of tokens). Decision `retain_nested3_price_4bit_observation_b2_b3` | `bugs/10_tier_set_rederivation/report.md` |
| **R11** | nested-code overhead | **done; `pass_target`** (job 988607); **extension done: `b3_generalizes`, `b2_all`** (job 992939) | nested code ≤0.42% rate per width; end to end +0.34% across 4 cells, worst cell +2.9% at B=3 (target 10%, kill 20%), mostly the tier-3 observation bit. Extension (Qwen3-30B 24 fresh prompts, Qwen1.5-MoE, Mistral-7B 8k/32k; in-process `codebook_ab`): B=3 macro mean **+1.48%** (cells −3.6% to +3.8%), B=2 cells −1.9% to +4.9%, **0 prompts past the 20% tail** at either B, worst prompt +10.2%. The Qwen3-30B B=2 tail did not recur on fresh prompts. Mechanism prediction 1 refuted (one Mistral 8k prompt's top KV-group share 0.116 > 0.10), prediction 2 confirmed | `bugs/11_nested_code_overhead/report.md`, `main_ext_992939.txt` |
| R13 | K-channel axis | **planned** (plan frozen 2026-09-27; not run) | — | `bugs/13_channel_axis/plan.md` |
| R14–R15 | kernel/TPOT, related work | **not started** (related work drafted in the paper, `latex/main.tex`, Related work) | — | — |
| R12 | GQA union | **answered by co-design** | 1.14× (n_rep 4), 1.32× (n_rep 8) — far below the ">3×" kill line | `bugs/co-design/report.md` §7.2 |
| **R12 (folder)** | paper main table: matched quantization + eviction baselines, all arms paired | **done** (A 21832327/30, 21841737/38, 21850510/11; B 21850534; C 21834051/52/54/55 + 992195; D 21840797/803/804) | 36 valid cells, 12 methods + 2 diagnostics in one process per cell. Router 0.764 beats every evictor (best LaProx 0.596, +0.17 [+0.13, +0.20]) but a budget-matched dense quantizer beats the router on every model (KVQuant 0.973, KIVI-128 0.967, TurboQuant 0.946; Mistral: TurboQuant 0.951 vs router 0.699). **H-pool supported**: pooled-score interior recovers 110% of the 128k B=2 gap (0.531 → 0.799 vs OBCache 0.774), truncation 34% → 1%. **Hard cell B failed its gate** (TurboQuant 0.917 > 0.90; reported as at ceiling). C: SnapKV → 1.00 in all 5 cells when the question is visible | `bugs/12_paper_main_table/plan.md`, `tables_r12_paired.md`; paper `latex/main.tex` F4–F7 |

**Design state after R11 and the R12 paper table:**
- Settled from before: per-KV-head allocation, cascade score at bc = 4, two
  timescales.
- **Tier set (R10):** `{0,3,4,6,8}`, a 3-bit base.
- **Storage format (R11):** one successively refinable 3+1+2+2 code. Tier-3
  tokens retain their first refinement bit so the bc = 4 cascade can read a
  4-bit prefix (priced at 1.0–1.7 points of rate at B=3).
- K*-proportional budgeting is rejected; the general budget rule remains open.
- **New issue (R11 §4):** the water filler is deterministic but discontinuous.
  1–5% of KV groups flip ≥2× under small changes to the noise table. The flips
  cancel in dense models but not when error concentrates in one layer, as in
  Qwen3-30B's layer 3. The R11 extension did not reproduce the B=2 tail on 24
  fresh Qwen3-30B prompts (0/24 past 20%), so it is rare, not absent.
- **End task (R12 folder):** a model-matched dense quantizer beats every eviction
  and mixed policy at 2–3 key code bits; SIEVE's main failure is truncating
  multi-token answers, which positional pooling of its token scores repairs.

**Next actions, in order:**
1. Paper (ICLR 2027, `latex/`): trim the body to 9 pages, refresh
   `submission/supplementary_code.zip`, and fix Table 4's 128k question-visible
   row (job 992195 ran on corpus 0a26bc1e; the paired 128k grid on b524da5e).
2. Finish R1 against the measured R3–R12 results, and add the R13 channel-axis
   sentence.
3. Run the observation-bit follow-up: is bc = 3 good enough to drop the
   retained bit? This is an in-process A/B, `coarse_bits=3,4`.
4. Design allocator smoothing (hysteresis or damping) against the R11
   knife-edge before R14. R12's pooled-score result says to smooth over
   positions as well: allocate spans, not tokens.
5. Then R14.

R9 development is closed because its Q3 lock gate failed. Do not rerun V7 or R9
on their frozen prompts, and do not open their reserved development or
confirmation partitions.

**Methods rule (R11, 2026-09-24):** `run_h0`'s forward pass is not
bit-reproducible run to run at ≥8K, even on one GPU; MoE is worst. A row-paired
A/B must compute both arms inside one process (R10 tier panel, R11
`codebook_ab`, the R12 paired grid). Treat cross-run row comparisons (reruns,
seeds, R7's "rerun ≤0.6 pt") as carrying this noise.

**Corpus rule (R12, 2026-09-27):** the two clusters hold different PG-19 copies
(`corpus_sha` 0a26bc1e on Trillium, b524da5e on the other cluster). The same
`prompt_idx` then gives the same needle depths but different haystack text and
needle values. Compare runs prompt by prompt only when `corpus_sha` matches
(`script_temp.sh --check` prints it); R9-folder full grid (0a26bc1e) vs the R12
paired grid (b524da5e) is a cell-mean comparison only (median |Δ| 0.04, p90 0.15).

---

## 0. One correction found while reviewing our own code

**The τ/ln2 result is an algebraic identity, not a prediction.** Verified to
machine precision against `sievelib/alloc.py`:

```
L=131072   ladder_a_only = 4.3369449159   tau/ln2 = 4.3369449159   rel diff = 1.2e-15
L= 32768   ladder_a_only = 3.1634293428   tau/ln2 = 3.1634293428   rel diff = 9.8e-16
```

`ladder_bits_a_only = std_i(log₂ aᵢ)` and `log₂ aᵢ = sᵢ/ln2 − log₂Z`, so the
`log₂Z` is a constant offset and `std(log₂ a) = τ/ln2` **exactly**. The only thing
standing between prediction and tautology is the value term `‖vᵢ−o‖` — which we
separately measured at ≤0.035 bits and describe as decorative. So "confirmed to
1.4%" is measuring the size of a term we already told the reader is negligible,
and Fig 5-right plots x against x.

This is not a data problem; every number is still correct. It is a *claim* problem,
and it is load-bearing because the proposal currently calls it "the strongest
single piece of evidence in the paper". One sharp reviewer ends the paper's
credibility with it in two sentences.

**The replacement is already measured and stronger:** `lin_ratio3 = 0.80–1.35`.
That is a first-principles cost model predicting a *measured* output-error ratio
within ±35% across 24 configurations — a genuine forward prediction of the same
theory, and it was buried in §3 as a repair note. Promote it.

Everything else below is about finishing the project, not repairing it.

---

## 1. What we sell — and what each task defends

| # | claim | evidence today | defended by |
|---|---|---|---|
| **C1** | **dead-tier fraction is an order parameter** — derived from τ²c_b vs c₀=1, contains no L | ρ=−0.95 over 24 configs; invariant to a 6–29 pt shift in its own target when the corner was redefined; predicted the 70B's 37-pt collapse at 128k out of sample | R2, R4, R6 |
| **C2** | **context length is a phase variable, acting through τ** — per-model verdicts are ill-posed | one model traverses GO→NARROW→STOP on L alone; τ rises monotonically with L in 6/6; −3.7 to −15.9 band-pts per doubling | R3, R5, R6 |
| **C3** | **the objective substitution and the allocation theorem** — output distortion, reverse water-filling, eviction as the zero-rate case with `c₀=1` derived | `lin_ratio3 = 0.80–1.35`; interior chooses to evict 40–54% of tokens on its own, so tier 0 is inside the allocator | R1, R3, R7 |
| **C4** | **a router that reads each head's phase from one L-free calibration pass** | ladder width correlates with realised gain in 24/24 (−0.41 to −0.77); n₉₉₅ inverts on the 70B, so ladder is the only signal that works everywhere | R3, R5, R7, R8 |
| **C5** | **scoring is measured; the K*-budget hypothesis is rejected** - oracle beats deployable H2O by only 1.02-1.41x; dense physical-group K* is stable but nearly common mode and points away from high-error units | 24/24 legacy runs + authenticated R9 job 985183 | R8, R9 |

Tasks are tagged `[SELL]` where they defend or extend C1–C5, `[DEBT]` where they
are not our novelty but must be paid for the paper to stand.

**One contribution to restate, not delete.** Proposal contribution #6 currently
reads "the field's eviction baselines are oracles". That is wrong about the field
— H2O, SnapKV and StreamingLLM all score on *observed* past attention, which is
exactly what we implemented as `accum`/`window`/`recency`. The oracle corner was
**our own** construction. The accurate and still-interesting version is the one
`bugs/2/report.md` already states: *our* corner's scoring rule was an oracle,
correcting it moved every verdict by 6–29 points, and the price of using
decode-time-available information rather than current-step truth is 1.02–1.41× in
error, concentrated on diffuse heads. Fix the wording in R1; the result stands.

---

## 2. Tier 0 — blocking, cheap, 0 GPU (≈2–3 days)

### R1 · Reconcile the documents and demote τ/ln2 `[DEBT]`
**0 GPU · 1–2 days · nothing else is safe to write on top of**

- τ/ln2 → appendix consistency check, with the identity stated openly. Promote
  `lin_ratio3 = 0.80–1.35` into the C3 slot as the theory's real forward
  prediction. Redraw Fig 5-right or drop it — as plotted it is x against x.
- Restate contribution #6 per §1 above.
- Collapse ten contributions to the five in §1. Everything else — the value term,
  QJL, the φ retraction, the Gaussian-simulation result, the 1-bit tier, the
  defect record — becomes a *Findings and negative results* subsection. Ten
  contributions reads as a lab notebook; reviewers read four or five.
- **The pitch is stale and contradicts the proposal.** It still ships the refuted
  budget mis-specification as C6 "new, testable"; still lists "the eviction corner
  is an oracle" under *what is not yet settled*; still says "at 128k only
  llama3.3-70B holds NARROW; everything else is STOP" when no configuration is
  STOP any more; its header says 22 configs / 39,424 heads against 20 / 33,920 in
  its own abstract; and it quotes the 70B at 79%/59% where the tables say
  82.1/61.4 (oracle) or 92.6/75.4 (practical).
- Cut "we replace our results table entirely, for the second time" and the roll
  call of named defects. Rigour is demonstrated by the validity gates we now run
  — keep those in the main text and move the bisects to an appendix. The
  confession invites the reviewer to assume there is one more defect.

### R2 · Replot on the robust statistic, and partial out model identity `[SELL]` C1
**0 GPU · ~half a day**

`report.md` §3 already concedes the band fraction is threshold-fragile: a
1.02–1.41× change in corner error moves it 6–29 points, because the per-head gain
distribution is dense right at the 2× threshold. So stop making it the y-axis.
Replot the phase panel with **median routed gain** on y and demote band fraction
to a secondary panel. That removes the fragility objection and the unstable
"+6 to +29 points" framing in one move.

Also report ρ **partialled on model identity**: 24 configurations nested in six
models are not 24 independent points. If the within-model partial correlation
stays strongly negative, C1 becomes much harder to attack; if it collapses, we
need to know now rather than in a rebuttal. Both are free — the data is on disk.

---

## 3. Tier 1 — close our own honesty gaps (≈1.5 weeks)

These defend C1–C4 directly, and every one is a question a careful reader asks.

### R3 · E2b — the practical interior, the last asymmetry `[SELL]` C3, C4 — **DONE (job214217\*; `bugs/2/R3-report.md`)**
**~2 days + one campaign · the highest-value experiment remaining**

Still open from `bugs/2`, and the one place the comparison is not yet honest. We
demoted the *corner* to lagged attention but the *interior* still water-fills on
oracle sensitivity `w² = (a·‖v−o‖)²` computed from the current step. Having just
published that a corner scored on current-step truth is not a fair baseline, we
cannot leave our own interior scored that way.

Build `w2p` from the lagged score (`ō = pa @ V`, computable from cache), waterfill
on `w2p`, evaluate on true logits — decision lagged, evaluation honest, the same
asymmetry a deployed system actually faces. `practical_scores` is the plumbing and
it exists. Report the corner ranked both by raw `pa` (literature-faithful) and by
`w2p` (isolates whether the edge comes from mixed-precision *shape* rather than
from score information the corner lacks).

**Both outcomes are publishable, which is why it is safe to run now:**
- edge survives with both sides practical → the strongest GO this framework can
  produce, and no reviewer can attribute it to information asymmetry;
- edge collapses → the finding is that allocation needs current-query information,
  which is exactly the motivation for the cascade's cheap first pass, and turns
  C4 into a claim about *when to re-budget* rather than *how to score*.

### R4 · The 64k→128k step: mechanism or RoPE artifact? `[SELL]` C1, C2 — **DONE (`bugs/4/report.md` §8–9; co-design §7.6)**
**~1 day of GPU · the most interesting open question in the study**

Every model that reaches 128k drops sharply there (llama33-70b −36.9, qwen3-30b
−5.7, llama31-8b −5.4) with τ and n₉₅ both jumping. Two readings fit the existing
data **equally well** and they are not the same claim: the decay is a property of
attention (C2 stands, the reach really ends near 64k), or part of the slope is an
artifact of measuring models at their trained limit (C2 must be restated).

The confound is structural: the two models that dropped are **at** their RoPE cap
(131,072 of 131,072); the one with 2× headroom (qwen3-30b-a3b-2507, 131,072 of
262,144) dropped least. Pushing that model to 192k/256k moves rope_frac
0.50 → 1.00 while absolute L moves 128k → 256k, and the two hypotheses then make
**opposite within-model predictions** about where its steepest drop falls.

**Landed (0 GPU):**
- `models.yaml` — `native_ctx` per model, the README ctx audit made
  machine-readable and verified against every live config.
- the ctx guard — now caps `SIEVE_CTX` at `native_ctx`, not at the default `ctx`.
  It previously refused 192k for qwen3-30b-a3b-2507 on the stated grounds that
  "the registry values are native RoPE limits" — true for six of eight models and
  **false for the only one this experiment needs**. Past the window is still
  refused; past the default now prints a headroom note.
- `run_h0.py` — stamps `native_ctx`, `rope_frac`, `rope_type` per row, read from
  the live config and cross-checked against the registry. Without `rope_frac` in
  the parquet the question cannot be asked after the fact.
- `report.py` — `page_rope` plots each model against **both** x-axes and prints
  where each model's own steepest per-octave drop falls.
- `tests/test_units.py::test_rope_window` — 10 checks, pins the headroom.
- corpus capacity verified: 22/40 books clear 256k, against n_prompts 2–3.

**To submit:** `bugs/4_rope_limit_or_mechanism/script.sh` (four groups, with the
decision table written before the run). The cheap control is qwen3-1.7b, which has
5× headroom.


### R5 · Phase drift across decode steps `[SELL]` C4, C2 — **DONE for C4 (co-design §3, §7.7); τ-drift sub-question open**

> Current (2026-09-21): campaign 2 ran on six cells (co-design waves 1–3). The
> route survives 4,096 tokens (p90 regret 1.00 everywhere); the allocation goes
> stale (froz/lag1 2.5–4.3×). The within-generation Δτ test is underpowered.
> The note below is the 2026-09-19 status, kept for the record.
>
> Status 2026-09-19: llama31-8b at 8k/32k/128k (sampled), the bridge and the
> greedy control are done (job21406669–71, 73, 74). qwen3-30b @32k (job21406672)
> died on the preflight `interior_scores` crash, now fixed; `script.sh --run`
> submits only that cell. The fresh-token defect (bugs/2 R3-report §2) affects
> only these runs' interior `lag` column, which `drift.py` blanks; the route,
> band, τ and bridge columns are valid.
**~1 day of GPU (pilot first) · cheap, and it does double duty**

C4 claims "one offline calibration pass". Nobody has checked whether a head's
phase — its route (interior vs baseline), dead tiers, τ — *drifts* during a long
generation. Every campaign so far measured decode steps 0 and 4.

Two mechanisms can move it, and the design separates them:
- **content** — the query distribution changes as the generated text moves away
  from the prompt; independent of L;
- **length** — τ rises with L in 6/6 models, so generating G tokens should move τ
  by slope × log₂(1+G/L). That is a *within-generation* test of C2, which has only
  ever been measured *across* prompts of different length.

**Corrected from the earlier version of this entry:**
- *"One config at `quant_every=1, n_decode=32` settles it"* — it cannot. 32 tokens
  grow a 128k cache by 0.02% and an 8k cache by 0.4%, so the length half is
  unmeasurable at any ctx, and quantizing all 32 steps costs ~16× a campaign unit
  for a 32-token window. The run now uses the **sparse schedule**
  (`measure_steps`): decode to step 4,096 with the probe off in between and
  quantize only 14 log-spaced steps.
- *"provably mis-specified by the end of a long generation"* — not proved. The
  τ–L slope is across prompts; whether it holds inside one generation is what R5
  measures. If it does not, C2 must be restated as a statement about the context
  a model is given, not about how long it has been running.
- R5 is not R3's staleness sweep. `lag:k` (bug 2 §C) prices re-budgeting *inside*
  a head over ≤8 steps, and R3-report.md already shows it goes stale fast (lag
  cost 1.36–1.58× at k=1 on in-band heads). R5 asks whether the cheap per-head
  *router* survives thousands of tokens.

**Landed (0 GPU):**
- `submit_h0*.slurm` — forward `SIEVE_MEASURE_STEPS / FAMILIES / DECODE_TEMPERATURE
  / DECODE_TOP_P / DECODE_SEED / DECODE_BAN_EOS`. run_h0.py already read the
  first five, but nothing forwarded them, so a sparse run could not be submitted.
- `run_h0.py` — opt-in `decode_ban_eos` (min_new_tokens). With no chat template
  an instruct model ends a continuation within a few hundred tokens and every
  later row is post-EOS; `past_eos` could only flag those rows. Stamped per row.
  Pinned by `test_units.py::test_ban_eos`.
- `bugs/5_phase_drift_across_decode/drift.py` — the reader. `report.py` medians
  over steps and averages drift away. Per step: route flip rate against its
  noise floor, rank stability of gain, the regret of keeping the calibration
  route, and predicted vs measured Δτ from the across-ctx slope.

**Constraints the design has to respect:** `accum` cannot run sparse (it sums
every step), so R5's corner and interior are `last_step` (TOVA), and a dense
bridge run measures the accum→last_step substitution on the same rows. The
interior must be set explicitly (`SIEVE_INTERIOR_SCORES=last_step`), because the
`accum` default is silently dropped when accum is absent. There is no practical
gain at step 0, so calibration is step 1.

**To submit:** `bugs/5_phase_drift_across_decode/script.sh --pilot`, then `--run`
(llama31-8b at 8k/32k/128k plus qwen3-30b at 32k; a dense accum/last_step bridge;
a greedy control), with the decision table written before the run.

### R6 · Pin the sharp boundary `[SELL]` C1 — **DONE: pinned per model, not universal (co-design §4)**
**~10 GPU-h (+~70 with the 70B) · reader landed, boundary already fitted**

For an order-parameter paper the boundary *location* is part of the result: at
what dead-2 does the band cross GO (35%) and STOP (15%), with what uncertainty,
and is that location the same for every architecture?

**Where it stands** — `boundary.py` on R3's 16 measured symmetric cells
(`bugs/2/R3-report.md` §2.1), i.e. on the comparison with no information
asymmetry:

| line | monotone d* | bracket (no cell inside) | per-model crossings |
|---|---|---|---|
| **GO** (35%) | 26.8% dead-2 | [25.8, 27.0], 1.2 pts | llama31-8b 26.6 · qwen15-moe 32.1 · llama33-70b 32.5 |
| **STOP** (15%) | 57.4% dead-2 | [53.7, 59.8], 6.1 pts | qwen3-8b 53.2 · llama31-8b still above at 53.7 · qwen3-30b already below at 62.0 |

So the *pooled* location is close to pinned. What is **not** pinned is whether
it is the same across architectures — which is the actual C1 claim. Every
per-model crossing above is interpolated across a wide gap in that model's own
curve (llama31-8b 18.8→27.0, qwen15-moe 25.8→36.8, llama33-70b 9.6→41.4,
qwen3-8b 50.5→59.8), and the one place two models can be compared at matched
dead-2 they disagree: at ~53.7% llama31-8b reads 20.2% in band while qwen3-8b
has already crossed 15% by 53.2%. The architecture interval (90%) is 10.3 pts at
GO and 11.2 at STOP against measurement intervals of 8.4 and 8.5 — **architecture,
not noise, is what R6 has to buy down.**

**Corrected from the earlier version of this entry:**
- *"bugs/2 puts it at 57–69% dead tiers in the two Qwen3 models"* — no crossing
  lives there; 57–69% is where the Qwen3 models *sit*.
- *"above the 45–50% previously guessed"* — for the STOP line the guess was
  **low** (57.4%), and 45–50% is roughly the *oracle* cell's crossing (45.8% on
  the E2 campaign), a different comparison.
- *"the 70B lands at 41.4% dead-2 → 25% band"* — that is the asymmetric E2 band.
  Symmetric: 21.3%, still NARROW; the 70B never reaches STOP.
- *"the 40–60% region carries few points"* — coverage is not the problem (five
  cells sit in 41–60). No model has two **adjacent** cells straddling a line.
- *"12k, 24k, 48k"* — half right, for one model: 24k is a good qwen3-8b cell,
  12k lands below the gap, and 48k is past three models' RoPE windows and
  outside both gaps for the llamas.
- *"the same mechanism that produced the sweep"* — `submit_h0_ctx_sweep.slurm`
  does not parse `SIEVE_*=value` arguments, so on Trillium it would silently run
  the defaults. R6 uses `submit_h0.slurm`, one cell per job, as R3–R5 do; the
  sweep script is left unchanged.
- *"keep `page_phase`'s hatch data-driven"* — it is, but on the wrong quantity:
  it hatches the widest gap in dead-2 *coverage*, only past 12 pts, so it never
  draws here. The gap that matters is the one at the *crossing*. `report.py` is
  untouched; `boundary.py` prints the crossing bracket.
- *"a fitted interval with its uncertainty"* — implemented, and the first thing
  it showed is that a smooth fit is the wrong tool: the logistic puts GO at
  34.6% when no cell exists between 25.8 and 27.0 to support it.

**Landed (0 GPU):**
- `bugs/6_pin_sharp_boundary/boundary.py` — per target (symmetric / E2 /
  oracle): a **monotone (isotonic)** crossing, which cannot leave the bracket,
  plus the logistic as a cross-check with a warning when it does; two 90%
  intervals (layer-cluster bootstrap = measurement; + model resampling =
  architecture); the model-free bracket (gap vs overlap); each model's own
  crossing; routed gain at d*. The symmetric target is read only from
  `floor_maxb` runs. Reproduces `bugs/2/report.md` and R3-cells.csv exactly.

**To submit:** `bugs/6_pin_sharp_boundary/script.sh --run` — six cells in R3's
exact config, so they pool with R3's 16. The sheet refuses to spend GPU time
until a finished R3 parquet shows the fresh-token floor working (lag cost < 2×).
- **the STOP gap:** qwen3-8b at 16k and 24k (≈54 and ≈57, inside its own
  crossing), and qwen3-30b at 4k (≈59) to turn its one-sided bound into a
  crossing — a third architecture in the gap.
- **the GO gap:** llama31-8b at 16k (≈23.6) and qwen15-moe at 16k (≈34.1), so
  both crossings straddle the line instead of being interpolated; plus
  llama31-8b at 64k, the largest remaining hole in any model's curve (27.0→53.7)
  and a point R4 and C2 want anyway.
- `--with-70b`: llama33-70b at 96k and 112k, inside its 32-point jump — the
  widest interpolation in the study, and the crossing furthest from
  llama31-8b's. ~70 GPU-h, worth it only if the GO line's architecture spread
  goes in the paper.

The decision table (pinned-and-universal / pinned-per-model / non-monotone) is
in the script, written before the run. R3's caveat carries over: these corners
are `accum` alone, and a five-corner `min` moves the STOP line by up to 6 points.

### R7 · Error bars, and the cells whose verdict is not reportable `[DEBT]` → protects C1 — **DONE (co-design wave 4, §7.5)**
**~11 GPU-h · the reader is 0 GPU · `bugs/7_error_bars_and_seeds/`**

The R3 re-run landed (job21421769–74, `floor_maxb`, interior lag cost 1.23×), so
the **symmetric cell** — interior and corner both lagged, the honest headline —
exists for the first time. Read with `errorbars.py` at each cell's reference
block size, layer-cluster interval only:

| cell | `sym_acc` | 90% (layers) | `e2_acc` | at risk |
|---|---|---|---|---|
| llama31-8b @32k | 34.0 | [29.2, 38.7] | 45.1 | **crosses GO** |
| qwen3-8b @8k | 16.5 | [13.5, 19.5] | 40.5 | **crosses STOP and GO** |
| qwen3-30b @8k | 14.1 | [10.5, 17.7] | 34.8 | **crosses STOP and GO** |
| llama31-8b @128k | 20.2 | [15.6, 24.9] | 29.2 | widest prompt spread (sd 9.6 pts) |
| llama33-70b @128k | 21.6 | [17.5, 25.7] | 26.1 | clear |
| qwen3-30b @128k | **7.5** | [4.9, 10.2] | 20.8 | **clear STOP** |

**Corrected from the earlier version of this entry:**
- *"qwen3-30b @128k sits at 18.6%, 3.6 points above the STOP line — the single
  cell where a modest measurement change flips a verdict"* — that was an E2
  number for a cell that reads **7.5%** on the symmetric target, five points of
  interval clear of STOP. The cells actually at risk are three cheap ones (one
  at 32k, two at 8k), and they cross **both** lines, not one.
- *"n_prompts and rot_seed are not reachable through `SIEVE_*`"* — fixed
  2026-09-18. The knob that was missing, `prompt_offset`, landed 2026-09-20
  (default 0 ⇒ every earlier run unchanged).
- *"two extra seeds is enough"* — `rot_seed` is a **quantizer** seed: prompts are
  keyed on `prompt_idx` alone, so a second `rot_seed` re-reads the same books at
  the same offsets with the same needles. Measured component sizes are
  **prompts sd 0.6–9.6 pts**, **layers ±3–5**, **corner set 3.1–6.8**, **rotation
  and rerun ≤0.6** — the one the entry proposed is the smallest by an order of
  magnitude, and the corner set (which R3-report §3.4 asked for) is larger than
  the gap the entry called decisive.
- **New, and it affects R4 and R6:** the band is a count of per-head *medians*,
  so it is **not invariant to `n_prompts`** (21.0% at one prompt vs 18.9% at
  four, same run). Replicates need equal block sizes, and R4's 2–3-prompt ctx
  points are biased ~+0.5 to +2 points against this grid's 4–6-prompt cells when
  R6 pools them.
- **Also new:** the haystack is keyed on **(corpus_sha, prompt index)** —
  `prompts.py` shuffles the book order with the corpus sha, so re-staging the
  corpus makes prompt 0 a different novel. Any "same prompts" comparison across
  corpora (including R3-report §3.5's across-campaign agreement) is a prompt
  comparison, not a rerun one.

**Landed (0 GPU):**
- `run_h0.py` — `prompt_offset`; `prompt_block` and `rot_seed` stamped per row
  and in the sidecar, so a replicate's sample is recorded rather than inferred.
- `submit_h0*.slurm` — forward `SIEVE_PROMPT_OFFSET`; the first log line echoes
  `prompts=<n>@<offset> rot_seed=<s>`.
- `tests/test_units.py::test_prompt_offset` — offset 0 is byte-identical to
  every earlier run; blocks are disjoint; prompt identity ignores every seed;
  band(1 prompt) > band(6 prompts).
- `bugs/7_error_bars_and_seeds/errorbars.py` — the reader: equal-size prompt
  blocks, six targets (symmetric / E2 / oracle × accum-only / five-corner min),
  the four components separately, and a verdict line that says when an interval
  straddles STOP or GO. Its screen of the R3 grid is `reports/r7_grid.csv`.

**To submit:** `bugs/7_error_bars_and_seeds/script.sh --run [--near-line]` —
llama31-8b @32k, qwen3-8b @8k, qwen3-30b @8k and llama31-8b @128k, two disjoint
prompt blocks each, five corners so both corner-set versions come from one run.
Gated on the R3 fix (G), the `prompt_offset` knob (P) and the corpus (C). The
decision table is in the script, written before the run.

## 4. Tier 2 — the binding constraint (≈1 week)

### R8 · Router-on vs router-off on an end task `[DEBT]` → unlocks C4, C5 — **DONE (gate failed); superseded for the paper by the R12-folder paired grid**
**~3–5 days · the one experiment we still owe**

> **Current state (2026-09-22)** — full detail in `bugs/8_router_endtask/plan.md`.
>
> - **Design as built** (differs from the paragraph below): model pair
>   **llama31-8b vs qwen3-8b** (same size, same n_rep = 4, opposite phase) plus a
>   llama31-8b context sweep 8k/32k/128k, instead of qwen3-30b vs llama33-70b
>   (which confounds phase with size and architecture). Tasks: RULER-style
>   niah_single / niah_multikey / niah_multivalue / vt on the PG-19 haystack.
>   Arms at matched bits: fp, uniform, evict (SnapKV), interior (the paper's
>   water-fill, per KV head), router_calib (offline routes from a disjoint
>   calibration block), router_oracle (upper bound). Simulated quantization,
>   keys only, protected 32-token window. Code: `sievelib/{compress,router,
>   tasks_ruler}.py`, `h0_measurement/run_r8.py`, `submit_r8.slurm`,
>   `tests/test_r8.py`.
> - **P0 done** (job 21529825, llama31-8b @32k, 20 prompts × 4 tasks): FP 0.99–1.00;
>   uniform fails at 1 bit and is perfect from 2 bits; **SnapKV scores 1.00 at
>   every budget on every task.** With the question inside SnapKV's observation
>   window, eviction is an oracle on retrieval, so nothing can beat it (plan §11).
> - **P0b done** (job 978352): question-agnostic compression makes eviction
>   discriminate, but uniform jumps from failure at B=1 to 0.95--1.00 at B=2;
>   the preregistered reader asks for an intermediate regime.
> - **P2 + R9 campaign done** (jobs 978479--978489): all five evaluation cells
>   completed. P-1, P-2, and P-5 fail, and P-4 is unsupported
>   (`bugs/8_router_endtask/report.md`).
> - **R12 folder (2026-09-27)** re-ran all five cells with every arm in one
>   process, adding KIVI, KVQuant, H2O and two diagnostics, plus Mistral-7B, a
>   question-visible control, and a pre-registered hard cell. The paper's end-task
>   numbers come from there (`bugs/12_paper_main_table/`); see the R12-folder
>   entry below.

The current answer to the output-error transfer question is negative evidence:
the measured gain does not establish the prespecified correlation with task
accuracy, and the phase prediction reverses across the five complete cells.
Because uniform is at ceiling in 31/36 valid cells, first repair the regime;
then run a small Llama/Qwen pilot with the calibrated and oracle routers before
spending on another full grid.

This is not our novelty — end-task evaluation is table stakes — but it is the
difference between "interesting map" and "I would run this", and it is the one
result that makes C1 actionable rather than descriptive. Resist scope creep into a
full benchmark sweep; one model pair on one benchmark is enough for the claim.

### R9 - K*-derived KV-head budget [SELL] C5 - **DONE; STOPPED AT Q3 (JOB 985183)**
**~1 week · a second, independent contribution — and the insurance policy**

> Note: the folder `bugs/9_sota_eviction_baselines/` is a **different** task that
> took the R9 slot: four published eviction baselines (Ada-KV, DropKV, OBCache,
> LaProx) implemented as extra R8 arms. All five main-model cells are measured;
> OBCache-K + Ada-KV and LaProx lead the new eviction arms, while uniform wins
> 34/36 valid cells (`bugs/9_sota_eviction_baselines/report.md`). The K\*-budget
> below is now isolated in `bugs/10_kstar_budget/`.

**Correction (2026-09-24).** The old K* result used the union of 12 geometric
keep counts and the configured absolute-policy count. At B=3 the extra count
occasionally made the preceding checkpoint as high as 97.5% of full, but the
median predecessor among saturated rows still falls from 48.5% at 8K to 37.7%
at 128K, and every configuration's median predecessor is the geometric point.
Exact error is nonmonotone, so `K*=100%` does not prove that every retained
token is necessary. Its K*/n₉₅ spread remains a description of that coarse
diagnostic, not an exact budget profile.

The existing rows are also per query head. A 24-cell CPU Gate 0 found that
maximising them within each physical GQA group gives 91.9--99.9% of the fixed
fractional count at 32K, with four of five GQA models above 98%. That shortcut
and the naive query-head policy are stopped.

The direct 8K/B=3 qualification completed as source-sealed array 985183.
Implementation and G=1 reduction passed (max difference 1.954e-14), Llama
prompt-half stability passed (Spearman 0.752), and no K* landed exactly at k0.
The allocation gate failed: equal-total K*-prop moved only 0.2319% of Llama
memory against the frozen 5% minimum. Its relative threshold also points in the
wrong direction for a global objective: Spearman(K*, E_full) is -0.801, and
donors contain 84.96% of E_full mass. K*-prop and shrink both lose all 12
descriptive held-out model/prompt/family cells. R9 is closed, no development
lock exists, and the detailed result is in bugs/10_kstar_budget/report.md.

---

## 5. Tier 3 — not our novelty, must be justified (≈3–6 weeks)

Ordered by how much the paper loses if a reviewer asks and we have nothing.

### R10 · Tier-set re-derivation and the 3-bit base layer (old E3) `[DEBT]` — **DONE (job 986347; `bugs/10_tier_set_rederivation/report.md`)**
**~2 days · prerequisite for R11**

> **Result (2026-09-24):**
> - The nested 3+1+2+2 ladder `{0,3,4,6,8}` passes all frozen gates G1–G5 at
>   B=3 (median cell error 1.017 of full, worst 1.031, routed gain kept 0.986)
>   and at B=2 (worst 1.070). Removing the 1-, 2- and 5-bit rungs each costs
>   about 1%.
> - The re-budgetable `{0,4,6,8}` fails against it (error ratio up to 1.26×).
> - The 1-bit tier is dead (≤0.1% of physical tokens; 0 in Qwen).
> - Decision `retain_nested3_price_4bit_observation_b2_b3`. The retained 4-bit
>   observation bit went to R11.

H2 failed at its stated threshold — Spearman 0.52–0.77 against 0.9 — so the coarse
pass identifies the head region but does not order it; the base layer moves to 3
bits. Fold in the measured fact that the **1-bit tier is dead in 95–100% of
heads**: remove it from the offered set rather than allocating a tier that is
never chosen. Both are repairs to our own architecture.

### R11 · Nested coding overhead (old E4) `[DEBT]` · "way this dies" #2 — **DONE: `pass_target` (job 988607; `bugs/11_nested_code_overhead/report.md`)**
**~1 week**

> **Result (2026-09-25):**
> - The nested code costs **≤0.42% rate** per width on real keys (Gaussian
>   design: ≤0.86%).
> - End to end at B=3, measured in-process on identical inputs:
>   **+0.34% across the 4 cells** (worst +2.9%), mostly the tier-3 observation
>   bit. The architecture is not killed.
> - B=2 passes on average (+5.0%), but Qwen3-30B's cell rests on one prompt at
>   +58%. That is an allocator knife-edge in a layer that concentrates error,
>   not code distortion.
> - **Extension done (job 992939, 2026-09-27; `main_ext_992939.txt`).** Four
>   cells, both codes in one process (`codebook_ab: nested3`): Qwen3-30B-A3B 8k
>   (24 fresh prompts), Qwen1.5-MoE 8k (12), Mistral-7B 8k and 32k (6 each).
>   Frozen decisions **`b3_generalizes`** (macro mean +1.48%, cell means −3.57%
>   to +3.81%, worst prompt +10.2%) and **`b2_all`** (cell means −1.91% to
>   +4.86%). No prompt passes the 20% tail at either B, so the original
>   Qwen3-30B B=2 +58% prompt did not recur. Mechanism prediction 1 is refuted
>   (Mistral 8k top KV-group share 0.116 > 0.10); prediction 2 is confirmed.
>   Earlier extension attempts (991618–991621 on Trillium; 21843344 on the other
>   cluster, whose `run_h0.py` lacked `codebook_ab`) are not used.
> - Two earlier attempts were invalid because two `run_h0` processes are not
>   bit-reproducible (methods rule in the status board).

Equitz–Cover is asymptotic and assumes optimal vector quantization; we use
per-coordinate Lloyd-Max at d=128 with three refinement boundaries. Kill the
architecture if the nested 3+1+2+2 code costs more than ~10–20% rate against a
monolithic 8-bit code. Successive refinability is classical and not ours, but the
architecture rests on it, so it must be measured rather than cited.

### R12 · GQA union (old E6) `[DEBT]` → threatens C4's granularity — **ANSWERED by co-design wave 4: 1.14× (n_rep 4), 1.32× (n_rep 8)**
**~1 week**

> Note: the folder `bugs/12_paper_main_table/` is a **different** task that took
> the R12 slot (as `bugs/9_*` did for R9): the paper's end-task main table.
> **Done 2026-09-27.** Plan and pre-registered rules: `plan.md`; tables:
> `tables_r12_paired.md`; reader: `read_main_table.py`.
> - **A, paired grid** (Llama-3.1-8B 8k/32k/128k, Qwen3-8B 8k/32k; prompts
>   100–119; B = 2, 3; question-agnostic; 15 arms in one process per cell).
>   36 valid cells. Means: KIVI-32 0.990 (3.00 eff. bits, not matched),
>   KVQuant 0.973, KIVI-128 0.967, TurboQuant 0.946; SIEVE router 0.764;
>   LaProx 0.596, OBCache-K+Ada-KV 0.572, Ada-KV 0.510, interior 0.493,
>   DropKV 0.397, SnapKV 0.388, H2O 0.338. Router vs best evictor per cell
>   16 W / 10 T / 10 L (margin 0.15; all losses on Llama); vs every
>   budget-matched quantizer −0.18 to −0.21, no wins against KIVI-128 or KVQuant.
>   TurboQuant wins Llama (0.983); on Qwen it fails variable tracking at B=2
>   (0.42 / 0.44) and the per-channel quantizers win (0.986 / 0.987).
>   Within-cell ρ(output error, score) −0.16 (12 methods), −0.38 token-selective.
> - **H-pool (pre-registered, 128k B=2): supported.** `interior_pool` 0.799 vs
>   `interior_cascade` 0.531 and `obck_ada` 0.774 = 110% of the gap (≥50%
>   needed). Llama truncation 39/44/34% → 9/11/1% at 8k/32k/128k. The oracle
>   router reaches only 0.661 there, so the failure is the token-separable
>   objective, not the calibration. H2O, the one unpooled evictor, truncates 52%.
> - **B, hard cell** (Llama 32k, 32 keys/4 values/4 hops, fresh prompts
>   1000–1059): **gate failed** (FP 1.00, TurboQuant 0.917 > 0.90). Reported as
>   still at ceiling; eviction ≤0.20, router 0.38 / 1.00 at B = 2 / 3.
> - **C, question visible** (5 cells, fp/uniform/SnapKV/H2O): SnapKV → 1.00 in
>   every cell; H2O moves ≤0.08 at 8k/32k. The 128k run (992195) used the other corpus.
> - **D, Mistral-7B** (8k/32k, 10 valid cells): TurboQuant 0.951, KIVI-32 0.930,
>   KVQuant 0.861, KIVI-128 0.825, router 0.699, best evictor 0.139. Pooling
>   cuts the interior's truncation only from 39% to 24%.
> - Fixes on the way: KVQuant k-means index overflow at ≥32k (float64 index,
>   jobs 992065/992068 failed before it).

Heads sharing a KV head cannot be given independent rates. Five of our six models
are GQA at ratio 4 or 8; qwen1.5-MoE is the n_rep=1 control, which is precisely
why it is in the registry. **If the G=8 union costs more than ~3× the single-head
footprint, per-head routing is partly fictional for most production models** — and
per-head routing *is* C4. This is a correctness-of-claim issue wearing engineering
clothes; do not leave it to the rebuttal.

### R13 · Scope the K-channel axis `[DEBT]`
**a sentence now, optionally a week later**

We study the **token** axis: per-token K quantization. Per-channel K treatment is
a different axis and is orthogonal and composable with ours. State that explicitly
in §1 — silence reads as an oversight. The optional upside is to ask whether the
channel-side ladder has its own dead tiers, which would extend C1 to a second
axis, but that is a follow-up paper, not a blocker.

### R14 · Kernel, iso-budget, TPOT vs dense FP8 (old E7) `[DEBT]`
**~4 weeks · highest cost, highest risk, schedule last**

FP8 currently beats every quantization variant on the Pareto frontier, which is
the bar. Kill the systems claim if TPOT at 128k does not beat dense FP8 at equal
accuracy. For a submission, R8's accuracy result matters more than TPOT — be
prepared to scope the paper as characterization-plus-router if this does not land.

### R15 · Related-work sweep before submission `[DEBT]`
**~1 day · ordinary hygiene, but do it deliberately**

The formulation this project rests on — output distortion as the objective,
water-filling as the solution, eviction as the zero-rate endpoint — is natural
enough that concurrent work may reach it independently. Before submission, sweep
arXiv for the last 12 months on rate allocation / output-distortion KV
compression, and for per-head budget allocators. Whatever it finds, C1, C2 and C5
are about *when* the interior pays and are not displaced by another derivation of
the allocation itself; budget an afternoon to position rather than discovering it
in review.

---

## 6. Explicitly dropped — do not spend time here

| item | why |
|---|---|
| **E1 / the `abs` budget policy** | κ=4 is refuted as shipped (1.9–7.4× the error). The accompanying `K*=100%` statement is resolution-limited by its coarse geometric-plus-absolute candidate set and cannot prove every token is necessary; R9 replaces it with a dense physical-KV-group test. |
| **`window` (SnapKV) and `recency` (StreamingLLM) in future campaigns** | 18 of the 23 bytes per layer-head-token of host state, for 6–22% and 1–12% of head wins. Keep `oracle` (free, and the bound the legacy columns are defined against) and `accum`. |
| **τ/ln2 as a headline** | identity, verified in §0. Appendix consistency check only. |
| **"the field's eviction baselines are oracles"** | wrong about the field; restated in §1 as a claim about our own corner and the price of decode-time-available information. |
| **Re-running the n₉₅ discontinuity** | resolved, not open — two models show it at the same step and one reproduces the prior campaign's L^1.81 figure to 1%. |
| **"provably monotone" as a bug detector** | false per head: a practical corner beats the oracle on 15.9% of head-rows, because ranking by the first-order proxy is not the argmin of the exact error. Aggregate direction only. Two test assertions encoding this were already removed. |

---

## 7. Order from here, with decision gates (updated 2026-09-27)

The original order (R1/R2 → R3–R7 campaign → R8 → R9…) has run through R11 and
the R12-folder paper table. What remains:

```
done       R9 dense physical-KV-group qualification     STOP at Q3
           R10 tier set {0,3,4,6,8}                     all gates pass
           R11 nested 3+1+2+2 code                      pass_target (B=3 +0.34% avg, +2.9% worst)
           R11-ext 4 new cells (job 992939)             b3_generalizes, b2_all (0 tail prompts)
           R12-folder paper table (paired grid, A-D)    dense quantizer wins; H-pool supported;
                                                        hard cell gate failed

now        paper: page limit, supplementary zip, Table 4 128k corpus note
           R1 documents against the measured R3-R12 results; R13 sentence
then       observation bit: bc = 3 vs bc = 4 (in-process A/B)
           allocator smoothing against the R11 knife-edge (design, then A/B)

later      R14 kernel / TPOT on the 3+1+2+2 code, R15 related-work sweep
           (R12 is answered)
```

## Dependency graph (updated 2026-09-25)

```
R1 R2 ──► R3 R4 R6 R7 (one campaign) ──► co-design waves 1-4 ──► design fixed:
  ✓        ✓  ✓  ✓  ✓        R5 ✓ ─────┘   (GQA per KV head,         per KV head,
                                            cascade bc = 4, R12 ✓)    cascade, 2 timescales
                                                                           │
                                                                           ▼
                                               R8  P0 ✓ ─► P0b ✓ ─► P2 5/5 ─► R12-folder paired
                                                                           │    grid A-D ✓
                         R9 K*-budget STOP at Q3 ──────────────┤
                                                                           ▼
                                               R10 tier set ✓ ─► R11 nesting ✓ ─► R14 kernel
                                                                    │                  ▲
                                                                    ├─ R11-ext ✓        │
                                                                    ├─ obs. bit bc 3/4 ─┤
                                                                    └─ alloc smoothing ─┘
```

## Design state

**Settled:**
- the objective (output distortion);
- the allocation rule, with eviction as tier 0 inside the allocator;
- per-KV-head allocation (co-design, R12);
- the interior's score (cascade at bc = 4);
- two timescales (router offline once, allocator re-budgeted; R5);
- the tier set `{0,3,4,6,8}` (R10);
- the storage format: a nested 3+1+2+2 code, with tier-3 tokens retaining their
  first refinement bit for the bc = 4 cascade (R11, ≤3% rate at B=3).

**Open:**
- **The general budget rule.** R9 rejects K*-proportional allocation under the
  direct physical-group objective.
- **Generality beyond the tested architectures.** The R11 extension adds
  Mistral-7B and Qwen1.5-MoE and finds no B=2 tail on 24 fresh Qwen3-30B
  prompts; dense 70B-class and other families remain untested.
- **Whether bc = 3 can drop the retained observation bit.**
- **Allocator discontinuity.** It matters wherever error concentrates in a few
  KV groups (R11 §4).
- **End-task transfer.** Answered negatively for method choice (R12 folder):
  output error ranks budgets and contexts but not methods, and a model-matched
  dense quantizer beats SIEVE on all three end-task models. Open: a span-level
  allocation objective (pooled scores help Llama fully, Mistral partly), and a
  non-ceiling end-task regime (the pre-registered hard cell failed its gate).
