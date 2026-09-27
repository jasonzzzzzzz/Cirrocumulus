# SIEVE — file manifest

Rate allocation for KV caches. Two studies share one core library.

| | |
|---|---|
| **H1 — simulation** | COMPLETE. Synthetic study; produced the figures in `docs/`. Given a fixed total memory budget for the KV cache, is it better to (a) give every token the same number of bits, (b) keep a few tokens at full precision and throw the rest away entirely, or (c) give different tokens different numbers of bits based on how important each one is? |
| **H0 — measurement** | Measurement campaigns **done** through R11 and the R12-folder paper table; the ICLR 2027 draft is in `latex/`. Open: R1 (documents), R13 (planned), R14–R15. Per-part status: the board at the top of **`h0_measurement/ROADMAP.md`**. |
| **Current documents** | proposal **v8** (`docs/proposal-sieve-v5.1-h0v2.html`), pitch **v5** (`docs/pitch-sieve-v5.1-h0v2.html`), both 2026-09-17 — they **predate** the R3–R8 results (reconciling them is ROADMAP R1, open). Earlier post-H0 versions (`*-h0v0` = v6/v3, `*-h0v1` = v7/v4) and the pre-H0 v5 pair are kept alongside; everything older is in `docs/deprecated/`. `docs/review-sieve-vs-rdkv.html` is a referee-style review of proposal v8 / pitch v4. |

## Status at a glance (2026-09-27)

| part | status | result | report |
|---|---|---|---|
| H1 simulation | done | figures in `docs/` | `h1_simulation/README.md` |
| R3 symmetric cell | done | interior's edge survives at ~half size; 5/16 cells STOP; ρ(dead-2, band) = −0.985 | `h0_measurement/bugs/2_towards_real_evictor/R3-report.md` |
| R4 · R5 · R6 · R7 | done | τ convex in log L (+0.21 at the RoPE cap) · route one-pass, allocation re-budgeted · boundary pinned per model · corner set dominates the error bar | `h0_measurement/bugs/co-design/report.md` §7 |
| co-design (GQA, cascade) | done | per-KV-head allocation costs 1.14× / 1.32× at n_rep 4 / 8; cascade score at bc = 4 closes 89% of the lag gap | `h0_measurement/bugs/co-design/report.md` §7.2–7.4 |
| R8 end task | done; gate failed | P-1/P-2/P-5 fail, P-4 unsupported; uniform at ceiling in 31/36 cells | `h0_measurement/bugs/8_router_endtask/report.md` |
| SOTA baselines (folder `bugs/9_…`) | done | Ada-KV, DropKV, OBCache, LaProx as R8 arms; uniform wins 34/36 cells | `h0_measurement/bugs/9_sota_eviction_baselines/report.md` |
| R9 (K\*-budget) | done; stopped at qualification | moved 0.23% of budget vs the 5% gate | `h0_measurement/bugs/10_kstar_budget/report.md` |
| R10 tier set · R11 nested code | done | `{0,3,4,6,8}` · 3+1+2+2 code +0.34% at B=3; **extension (job 992939): `b3_generalizes`, `b2_all`**, +1.48% over 4 new cells, no tail prompts | `h0_measurement/bugs/10_tier_set_rederivation/report.md`, `h0_measurement/bugs/11_nested_code_overhead/report.md` |
| **R12-folder paper table** | **done** | paired grid, 36 cells: dense quantizer (model-dependent) > SIEVE router 0.764 > best evictor 0.596; pooled-score test supported at 128k (110% of gap); hard cell gate failed; Mistral-7B agrees | `h0_measurement/bugs/12_paper_main_table/` |
| R13 channel axis | planned, not run | — | `h0_measurement/bugs/13_channel_axis/plan.md` |
| R1, R14–R15 | not started | — | `h0_measurement/ROADMAP.md` |

---

## Tree

```
.
├── README.md                     <- you are here
├── .gitignore                    ignores *.parquet, .venv/, .hf_cache/, .hf_token, .locks/, __pycache__/
├── .hf_token                     HF access token, ONE line, untracked. Every slurm
│                                 script reads HF_TOKEN from here (override: HF_TOKEN_FILE).
├── .hf_cache/                    HF_HOME — all model weights land here ($HF_HOME/hub)
├── .venv/                        the virtualenv every script activates
├── .locks/                       prefetch.py inter-process download locks
│
├── docs/                         THE DELIVERABLES
│   ├── proposal-sieve-v5.1-h0v2.html   CURRENT · v8 · "A Phase Diagram for KV Cache
│   │                             Compression" (2026-09-17; predates R3–R8 — ROADMAP R1)
│   ├── pitch-sieve-v5.1-h0v2.html      CURRENT · v5 · the phase-diagram pitch
│   ├── proposal/pitch-sieve-v5.1-h0v1.html   v7 / v4
│   ├── proposal/pitch-sieve-v5.1-h0v0.html   v6 / v3 — first post-H0 rewrite
│   ├── review-sieve-vs-rdkv.html referee-style review of proposal v8 / pitch v4
│   ├── fig5_phase.png            the phase figure (+ fig5_phase.txt)
│   ├── *.pdf (ada-kv, dropkv, obcache, laprox, palu, RDKV-2026)   papers the baselines follow
│   ├── proposal-sieve-v5.html    pre-H0 · v5 · full proposal, scoring, kill gates
│   ├── pitch-sieve-v5.html       pre-H0 · abstract, design space, novel claims (body marked v2)
│   ├── h0_expected_outputs.pdf   3 mock H0 reports: expected / best / worst
│   ├── figures_h1_v0/            archived v0 figures (fig1..4, incl. the dropped fig2_alloc)
│   └── deprecated/               proposal v3/v4, naive proposal, old pitch
│   NOTE: the HTMLs load fig1_curves.png / fig3_envelope.png / fig4_tau.png as
│   docs/ siblings, but those are NOT checked in at docs/ root. Regenerate:
│   `python h1_simulation/run_h1.py` (fig1+fig4); fig3 from the envelope code.
│
├── sievelib/                     SHARED CORE — used by both studies
│   ├── __init__.py
│   ├── quant.py                  TurboQuant_mse quantizer (rotation, Lloyd-Max,
│   │                             norm correction). Levels disk-cached.
│   ├── alloc.py                  water-filling, exact-recompute error, per-head
│   │                             metrics, band membership. The heart of both studies.
│   ├── probe.py                  attention capture + KV-cache access   (H0 only)
│   ├── validate.py               3 independent probe validation levels (H0 only)
│   ├── validity.py               input-validity checks for the H0 haystack
│   ├── evict.py                  the H0 eviction corners (oracle / H2O / SnapKV /
│   │                             StreamingLLM / TOVA on lagged attention) and their budgets
│   ├── prompts.py                niah / qa / cont prompt families over a window of
│   │                             real text (H0_CORPUS). Families share a haystack per
│   │                             prompt index, so niah-vs-cont is a paired test.
│   ├── compress.py               R8: the attention function that makes the model GENERATE
│   │                             from simulated-quantized / evicted keys; question-agnostic mode
│   ├── router.py                 R8: which width each context token gets, per arm (uniform,
│   │                             SnapKV, H2O, the water-fill interior, routers, calibration)
│   ├── tasks_ruler.py            R8: RULER-style tasks on the PG-19 haystack + scoring
│   ├── baselines.py              R8 arms from four papers: Ada-KV, DropKV, OBCache, LaProx
│   └── .lloyd_cache.pt           precomputed quantizer levels (~50 s to rebuild)
│
├── h1_simulation/                COMPLETE — synthetic study
│   ├── README.md                 what H1 tests and why
│   ├── run_h1.py                 regenerates docs/fig1_curves.png + docs/fig4_tau.png
│   │                             and prints the τ table
│   └── superseded_v1/            the three pre-audit scripts (h1_sim/h1_robust/h1_tau)
│
├── h0_measurement/               IN PROGRESS — real-model measurement
│   ├── README.md                 experiment plan + ctx methodology + validate_with
│   │                             rationale + how to run R8. Read first.
│   ├── ROADMAP.md                STATUS BOARD for every research part (R1–R15), then the plan
│   ├── bugs/<N>_*/               one folder per research part: plan, script.sh, reader, report
│   ├── run_r8.py, submit_r8.slurm   R8's end-task driver and its one-cell job
│   ├── models.yaml               model registry — add a model here, nothing else.
│   │                             Tiers: debug / main / large.
│   ├── prefetch.py               stage weights on the LOGIN node — model + its
│   │                             validate_with proxy (compute nodes have no internet)
│   ├── run_h0.py                 the measurement (--model TAG --out-dir DIR
│   │                             [--override k=v ...] [--validate-only])
│   ├── report.py                 multi-page PDF + go/no-go verdict + per-head CSV
│   ├── mock_report.py            regenerates docs/h0_expected_outputs.pdf
│   ├── quick_test.sh             LOGIN-node smoke test for one model, <10 min
│   ├── quick_test_all.slurm      CPU-cluster: prefetch + quick_test.sh for a LIST
│   │                             of models, one at a time
│   ├── submit_h0.slurm           MAIN campaign. sbatch array, 1×H100/task, self-
│   │                             chains the CPU report job (afterok on the array)
│   ├── submit_h0_large_models.slurm   LARGE-tier campaign. Same structure, 4 GPUs/task
│   │                             (70B / 30B-MoE do not fit one H100)
│   ├── report.slurm              SUPERSEDED — the report is now a self-resubmission
│   │                             of submit_h0*.slurm (SIEVE_ROLE=report)
│   ├── logs/                     all SLURM .out/.err + per-model quick_test logs
│   ├── results/<RUN_ID>/         *.parquet (one per model) + RUN_INFO.txt
│   ├── reports/                  h0_report_<RUN_ID>_<date>.pdf + _per_head.csv
│   └── results_smoke/            throwaway scratch for quick_test.sh
│
├── tests/                        CPU-only. Run with OMP_NUM_THREADS=8 (the login node caps CPU time)
│   ├── test_units.py             regression checks — one per audit bug. Run before any H0 GPU job.
│   ├── test_r8.py                R8 anchors (--fast = tensor only; full = + Llama-3.2-1B end to end)
│   └── test_baselines.py         the four SOTA baselines against brute force / autograd
│
├── reports/                      repo-root smoke scratch (quick_test.sh writes
│                                 smoke.pdf + smoke_per_head.csv here)
└── results_smoke/                repo-root smoke scratch
```

---

## Run, in order

All commands are from the **repo root**.

```bash
# 0. once — venv + deps
python -m venv .venv && source .venv/bin/activate
# fastparquet, NOT pyarrow: on this cluster `pip install pyarrow` resolves to a
# dummy wheel that refuses to build. fastparquet is a real wheel and needs no module.
pip install -U "torch>=2.4" "transformers>=4.48" accelerate \
               pandas fastparquet pyyaml matplotlib huggingface_hub

# 0b. once — HF token. ONE line, no newline fuss (scripts strip whitespace).
printf '%s' hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx > .hf_token

# HF_HOME is the single cache knob — every script derives HF_HUB_CACHE as $HF_HOME/hub.
export HF_HOME=$PWD/.hf_cache


```

Run h1 -- check the folder h1_simulation

Run h0 in -- check the folder h0_measurement


### Environment knobs

| var | used by | meaning |
|---|---|---|
| `HF_HOME` | all | the one cache knob; `HF_HUB_CACHE` is always `$HF_HOME/hub`. Default `$PROJECT_ROOT/.hf_cache`. |
| `HF_TOKEN_FILE` | slurm scripts | path to the token file. Default `$PROJECT_ROOT/.hf_token`. |
| `HF_HUB_OFFLINE` | slurm measure stage | forced to `1` on compute nodes — they have no outbound internet; a cache HEAD request there burns GPU time through five backoff rounds per file. |
| `PROJECT_ROOT` | slurm scripts | absolute repo root. Edit it (and `submit_h0*.slurm`'s hard-coded default) if the repo moves. |
| `SIEVE_MODELS` | slurm scripts | colon-separated model list (survives `sbatch --export`; a space list can arrive truncated). |
| `SIEVE_VENV` | `submit_h0*.slurm` | venv to activate. Default `$PROJECT_ROOT/.venv`. |
| `SIEVE_NO_REPORT=1` | `submit_h0*.slurm` | skip chaining the report job. |
| `SIEVE_ROLE` / `SIEVE_RUN_ID` | internal | set by the self-resubmission for the report stage — do not set by hand. |
| `H0_CORPUS` | `sievelib/prompts.py` | directory of real text for the haystack, staged by `h0_measurement/prefetch_corpus.py`. **Unset ⇒ tier main/large refuses to run** (the alternative is 8 sentences tiled ~970× at ctx 131072 — see `h0_measurement/README.md` § synthetic-haystack confound). |
| `H0_ALLOW_SYNTHETIC=1` | `h0_measurement/run_h0.py` | run main/large on filler anyway; `report.py` stamps the verdict UNKNOWN. |

---

## Which file answers which question

| Question | File |
|---|---|
| What is the project and is it worth doing? | `docs/pitch-sieve-v5.1-h0v2.html` (v5) |
| What exactly gets claimed, scored, and killed? | `docs/proposal-sieve-v5.1-h0v2.html` (v8) |
| **Where does every research part stand, and what runs next?** | **`h0_measurement/ROADMAP.md`, the status board at the top** |
| Does output error translate into end-task accuracy? | `h0_measurement/bugs/12_paper_main_table/tables_r12_paired.md` (paired grid; R8 history in `bugs/8_router_endtask/report.md`) |
| What bugs were found and what did they change? | `tests/test_units.py` — one regression check per audit bug |
| Does context length change the conclusion? | `h0_measurement/README.md` § Context length methodology |
| What will the result look like? | `docs/h0_expected_outputs.pdf` |
| What did the first real run show? | `h0_measurement/reports/` (PDFs + `analysis_from_fable.md`) — superseded as a verdict by R3 |
| Where is the allocation theorem implemented? | `sievelib/alloc.py` — `waterfill`, `exact_error` |
| Where is the noise actually measured? | `sievelib/quant.py` + `alloc.noise_model` |
| How do I add a model? | `h0_measurement/models.yaml`, append an entry |

---

## Version notes

Kept deliberately short since there is no VCS here.

- **2026-09-22 — R8 end-task harness.** `sievelib/{compress,router,tasks_ruler}.py`,
  `h0_measurement/run_r8.py`, `submit_r8.slurm`, `tests/test_r8.py`. P0 (job
  21529825) showed question-aware SnapKV is perfect on these tasks at every budget;
  added a question-agnostic mode (`--question-agnostic`, `R8_QA=1`) and fractional
  budgets. The question-aware path is unchanged (verified: identical parquets).
- **2026-09-21 — co-design waves 1–4** (`h0_measurement/bugs/co-design/`): GQA
  per-KV-head allocation (`alloc.waterfill_group`) and the cascade score landed in
  `alloc.py` / `run_h0.py` behind knobs that default off, with regression tests in
  `tests/test_units.py`.

- **proposal v6 / pitch v3** (`docs/*-v5.1-h0v0.html`) rewrite both documents around
  the completed H0: allocation beats both corners on ~38% of all heads (53% median
  model, 2.06× geo-mean routed gain), and the failures split into two predictable
  phases at opposite ends of the attention-concentration axis. The framing shifts
  from "our method wins" to a phase diagram of when each method wins. The pre-H0 v5
  pair is kept for reference.
- **proposal v5** supersedes v4. v4 claimed the allocation gain grows with logit
  spread τ. Measured against the *best* of both corners it is **non-monotonic** —
  peaks near τ≈1.25 at ~13×, gone by τ≈2. v5 revises Fig 1 and Fig 4, adds the
  per-head routing table, and moves acceptance from 60–68% to 45–55%. v3/v4 and
  the naive proposal are in `docs/deprecated/`.
- **pitch** (`docs/pitch-sieve-v5.html`, body still marked v2) carries the same
  correction and states the retraction explicitly.
- **H1 code** was three scripts (`h1_sim.py`, `h1_robust.py`, `h1_tau.py`) built on
  the pre-audit cost model. They are kept only under `h1_simulation/superseded_v1/`.
  `h1_simulation/run_h1.py` replaces them and uses the corrected `sievelib` core,
  so the figures and the H0 code now share one implementation instead of two that
  could drift.
- **Figures.** `fig2_alloc.png` (the allocation staircase) was generated under the
  superseded relative-cost model and is dropped. `fig1`, `fig3`, `fig4` are current;
  `fig1`/`fig4` are reproducible from `h1_simulation/run_h1.py`, `fig3` from the
  envelope code. The v0 set is archived in `docs/figures_h1_v0/`. The proposal/pitch
  HTMLs load `fig{1,3,4}` as `docs/` siblings — regenerate them into `docs/` if the
  images render broken.
- **Ten defects** were found in a second audit pass and fixed; each has a regression
  test in `tests/test_units.py`. The two that would have inverted the conclusion:
  eviction and quantization costs were in different units (6× error at τ=2.5), and
  Lloyd-Max was 4× from optimal at 8 bits. Each `check(...)` in `test_units.py`
  names the bug it locks down.
- **H0 SLURM layout.** The single `submit.slurm` + `submit_h0.sh` driver is replaced
  by three self-contained scripts under `h0_measurement/`: `quick_test_all.slurm`
  (CPU pre-flight), `submit_h0.slurm` (main tier, 1 GPU/task), and
  `submit_h0_large_models.slurm` (large tier, 4 GPUs/task). Each embeds its own
  report stage as a `SIEVE_ROLE=report` self-resubmission, so `report.slurm` is
  vestigial. All logs/results/reports now live under `h0_measurement/`, not the
  repo root.
