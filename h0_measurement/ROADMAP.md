# Roadmap: remaining experiments, ranked

Written after the E1/E2 campaign (`bugs/2_towards_real_evictor/report.md`).
Supersedes the "Remaining experiments" list in proposal v8 §7, which was drawn up
before that campaign landed and still lists items it completed or refuted.

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
| **C4** | **a router that reads each head's phase from one L-free calibration pass** | ladder width correlates with realised gain in 24/24 (−0.41 to −0.77); n₉₉₅ inverts on the 70B, so ladder is the only signal that works everywhere | R3, R7, R8 |
| **C5** | **two measured facts that redirect design** — scoring is solved (oracle beats deployable H2O by only 1.02–1.41×, on heads the router sends to the interior anyway); tail dependence is architectural (K*/n₉₅ spans 37×, not ordered by support) | 24/24 runs | R8, R9 |

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

### R3 · E2b — the practical interior, the last asymmetry `[SELL]` C3, C4
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

### R4 · The 64k→128k step: mechanism or RoPE artifact? `[SELL]` C1, C2
**~1 day · the most interesting open question in the study**

Every model that reaches 128k drops sharply there — llama33-70b −36.9, qwen3-30b
−5.7, llama31-8b −5.4 — with τ and n₉₅ both jumping (70B: n₉₅ 117 → 4,854, τ 2.06
→ 3.31). Two explanations with very different consequences for C2:

- **genuine attention regime change** → the decay accelerates past 128k, and the
  method's reach narrows honestly but the mechanism is confirmed;
- **RoPE behaviour near the trained limit** → part of the steep slope is an
  artifact of running models at their cap, and C2's slope needs restating.

There is already a hint worth chasing: the two models at **exactly** their RoPE
cap (llama31-8b and llama33-70b, both 131072) show the drops, while
qwen3-30b-a3b-2507 — native window **262144**, so 128k is half of it — shows the
smallest. Test directly:
- add **96k** for the llama pair, which brackets the step from below;
- push qwen3-30b-a3b-2507 to **192k**, using headroom it actually has.

If the collapse tracks *fraction of the RoPE window consumed* rather than absolute
L, that is a new finding and it reframes our own limit. Note `run_h0` refuses
contexts above the native cap by design — the qwen3 extension works within it, so
consult the guard rather than bypassing it.

### R5 · Phase drift across decode steps `[SELL]` C4
**~1 day · cheap, and it does double duty**

C4 claims "one offline calibration pass". Nobody has checked whether a head's
phase *drifts* during a long generation. The `step` column exists, but
`quant_every: 4` with `n_decode: 8` gives only steps 0 and 4 — far too coarse. One
config at `quant_every=1, n_decode=32` settles it.

Double duty, and this is the part that matters: τ rises monotonically with L, so
an allocation calibrated at prefill is **provably mis-specified by the end of a
long generation**. If drift is measurable, that is a mechanism-backed argument for
re-budgeting during decode — which is the strongest route from "we have a map" to
"we have a method". If drift is negligible, the one-pass calibration claim gets a
number instead of an assumption. Either way we need it before claiming the router
is cheap.

### R6 · Pin the sharp boundary `[SELL]` C1
**~1–2 days**

For an order-parameter paper the boundary *location* is part of the result, and
ours is not pinned. `bugs/2` puts it at 57–69% dead tiers in the two Qwen3 models,
above the 45–50% previously guessed, while the 70B lands at 41.4% dead-2 → 25%
band. The 40–60% region carries few points.

Populate it with intermediate contexts (12k, 24k, 48k) on the models whose dead-2
lands in the gap — the same mechanism that produced the sweep, no new
infrastructure. Then state the boundary as a fitted interval with its uncertainty
rather than a number, and keep `page_phase`'s hatch data-driven.

### R7 · Error bars, and the one cell that flips `[DEBT]` → protects C1
**~1 day**

qwen3-30b @128k sits at 18.6%, **3.6 points above the STOP line** — the single
cell where a modest measurement change turns "no configuration is STOP" into one
that is, which is the E2 headline. More broadly: we concede in print that the band
fraction is fragile, so we must ship error bars or the concession becomes the
attack.

`n_prompts` and `rot_seed` are not reachable through `SIEVE_*` (noted in
`bugs/2/report.md` §7); closing that in the submission scripts is small and
unlocks seeds everywhere. Two extra seeds on the 128k-capable cells is enough.

---

## 4. Tier 2 — the binding constraint (≈1 week)

### R8 · Router-on vs router-off on an end task `[DEBT]` → unlocks C4, C5
**~3–5 days · the one experiment we still owe**

Our own "ways this dies" list has this at #3 and it is right. Every number in the
paper is an **output-error** ratio. A reader's fair question is "why should I
believe this matters", and we currently have no answer.

Cheapest decisive design: pick the **highest** and **lowest** dead-tier models in
the set (qwen3-30b, and llama33-70b at ≤64k), run allocation with our router
enabled and disabled at matched total bits on RULER. **Show the router recovers
accuracy exactly where the diagram predicts the allocator degenerates.**

This is not our novelty — end-task evaluation is table stakes — but it is the
difference between "interesting map" and "I would run this", and it is the one
result that makes C1 actionable rather than descriptive. Resist scope creep into a
full benchmark sweep; one model pair on one benchmark is enough for the claim.

### R9 · K*-derived per-head budget `[SELL]` C5
**~1 week · a second, independent contribution — and the insurance policy**

The K* result says attention concentration is the **wrong** signal for budgeting:
the 70B has the smallest n₉₅ in the study (40–117 tokens at ≤64k) and the largest
tail dependence (K*/n₉₅ up to 214.7), a 37× spread that n₉₅ does not order. K* is
computable in one cumulative pass over the ranked tokens (`fix.md` §1).

Build a per-head keep-count from measured K* and measure it against the
fixed-fraction and fixed-κ rules that per-head budget allocators use. Small,
self-contained, and **orthogonal to whether the interior pays** — which makes it
the best insurance on this list. If R3 or R8 come back weak, this still stands as
a result.

---

## 5. Tier 3 — not our novelty, must be justified (≈3–6 weeks)

Ordered by how much the paper loses if a reviewer asks and we have nothing.

### R10 · Tier-set re-derivation and the 3-bit base layer (old E3) `[DEBT]`
**~2 days · prerequisite for R11**

H2 failed at its stated threshold — Spearman 0.52–0.77 against 0.9 — so the coarse
pass identifies the head region but does not order it; the base layer moves to 3
bits. Fold in the measured fact that the **1-bit tier is dead in 95–100% of
heads**: remove it from the offered set rather than allocating a tier that is
never chosen. Both are repairs to our own architecture.

### R11 · Nested coding overhead (old E4) `[DEBT]` · "way this dies" #2
**~1 week**

Equitz–Cover is asymptotic and assumes optimal vector quantization; we use
per-coordinate Lloyd-Max at d=128 with three refinement boundaries. Kill the
architecture if the nested 3+1+2+2 code costs more than ~10–20% rate against a
monolithic 8-bit code. Successive refinability is classical and not ours, but the
architecture rests on it, so it must be measured rather than cited.

### R12 · GQA union (old E6) `[DEBT]` → threatens C4's granularity
**~1 week**

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
| **E1 / the `abs` budget policy** | refuted. K* = 100% of budget in 24/24 runs; κ=4 costs 1.9–7.4× the error. Withdraw the mis-specification claim and report K*/n₉₅ as the tail-dependence measurement it turned out to be (R9). |
| **`window` (SnapKV) and `recency` (StreamingLLM) in future campaigns** | 18 of the 23 bytes per layer-head-token of host state, for 6–22% and 1–12% of head wins. Keep `oracle` (free, and the bound the legacy columns are defined against) and `accum`. |
| **τ/ln2 as a headline** | identity, verified in §0. Appendix consistency check only. |
| **"the field's eviction baselines are oracles"** | wrong about the field; restated in §1 as a claim about our own corner and the price of decode-time-available information. |
| **Re-running the n₉₅ discontinuity** | resolved, not open — two models show it at the same step and one reproduces the prior campaign's L^1.81 figure to 1%. |
| **"provably monotone" as a bug detector** | false per head: a practical corner beats the oracle on 15.9% of head-rows, because ranking by the first-order proxy is not the argmin of the exact error. Aggregate direction only. Two test assertions encoding this were already removed. |

---

## 7. Suggested order, with decision gates

```
days 1-3   R1 doc reconciliation + tau/ln2 demotion    0 GPU
           R2 replot on routed gain + partial rho      0 GPU
                                                       GATE: does rho survive
                                                       partialling on model?
                                                       no -> C1 needs restating
                                                       before anything else runs

week 1-2   R3 practical interior (campaign)            GATE: does the edge survive
           R4 96k + RoPE headroom                       with both sides practical?
           R5 phase drift                               no -> pivot C3/C4 to the
           R6 pin the boundary                          cascade / re-budgeting story
           R7 seeds + error bars

week 3     R8 router on/off on RULER                   GATE: does the router recover
                                                       accuracy where predicted?
                                                       no -> paper is characterization;
                                                       shift weight to R9

week 4-5   R9  K*-derived per-head budget              insurance: stands alone
           R10 tier set + 3-bit base
           R13 channel-axis scoping sentence

week 6+    R11 nesting overhead
           R12 GQA union
           R14 kernel / TPOT
           R15 related-work sweep
```

**Two things matter most on this page.** R1 and R2 cost no GPU time and remove the
one claim that is currently indefensible — do them before writing anything else.
And R3 is the highest-value experiment left: it is the last place our own
comparison is asymmetric, we have just made a public point of exactly that kind of
asymmetry, and both of its outcomes are publishable.
