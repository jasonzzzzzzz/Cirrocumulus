# SIEVE — file manifest

Rate allocation for KV caches. Two studies share one core library.

| | |
|---|---|
| **H1 — simulation** | COMPLETE. Synthetic study; produced the figures in `docs/`. Given a fixed total memory budget for the KV cache, is it better to (a) give every token the same number of bits, (b) keep a few tokens at full precision and throw the rest away entirely, or (c) give different tokens different numbers of bits based on how important each one is? |
| **H0 — measurement** | IN PROGRESS. Measures real models. A first pass of both tiers has run — see `h0_measurement/results/` and `h0_measurement/reports/`. |
| **Current documents** | proposal **v6** (`docs/proposal-sieve-v5.1-h0v0.html` — the phase diagram, post-H0), pitch **v3** (`docs/pitch-sieve-v5.1-h0v0.html`). The pre-H0 v5 pair (`*-v5.html`) is kept alongside; everything earlier is in `docs/deprecated/`. |

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
│   ├── proposal-sieve-v5.1-h0v0.html   CURRENT · v6 · "A Phase Diagram for KV Cache
│   │                             Compression" — rewritten around the completed H0 (6 models, 10,240 heads)
│   ├── pitch-sieve-v5.1-h0v0.html      CURRENT · v3 · the phase-diagram pitch, measured
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
│   ├── prompts.py                niah / qa / cont prompt families over a window of
│   │                             real text (H0_CORPUS). Families share a haystack per
│   │                             prompt index, so niah-vs-cont is a paired test.
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
│   │                             rationale. Read first.
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
├── tests/
│   └── test_units.py             CPU-only regression checks — one per audit bug.
│                                 Run before any GPU job.
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
| What is the project and is it worth doing? | `docs/pitch-sieve-v5.1-h0v0.html` (pre-H0: `pitch-sieve-v5.html`) |
| What exactly gets claimed, scored, and killed? | `docs/proposal-sieve-v5.1-h0v0.html` (pre-H0: `proposal-sieve-v5.html`) |
| What bugs were found and what did they change? | `tests/test_units.py` — one regression check per audit bug |
| Does context length change the conclusion? | `h0_measurement/README.md` § Context length methodology |
| What will the result look like? | `docs/h0_expected_outputs.pdf` |
| What did the first real run show? | `h0_measurement/reports/` (PDFs + `analysis_from_fable.md`) |
| Where is the allocation theorem implemented? | `sievelib/alloc.py` — `waterfill`, `exact_error` |
| Where is the noise actually measured? | `sievelib/quant.py` + `alloc.noise_model` |
| How do I add a model? | `h0_measurement/models.yaml`, append an entry |

---

## Version notes

Kept deliberately short since there is no VCS here.

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
