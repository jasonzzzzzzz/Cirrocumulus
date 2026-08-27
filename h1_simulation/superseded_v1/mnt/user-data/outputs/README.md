# SIEVE — file manifest

Rate allocation for KV caches. Two studies share one core library.

| | |
|---|---|
| **H1 — simulation** | COMPLETE. Synthetic study; produced the figures in `docs/`. |
| **H0 — measurement** | NEXT. Measures real models. One day of GPU time. |
| **Current documents** | proposal **v5**, pitch **v2**. Everything earlier is superseded and deleted. |

---

## Tree

```
.
├── README.md                     <- you are here
│
├── docs/                         THE DELIVERABLES
│   ├── proposal-sieve.html       v5 · full proposal, scoring, kill gates
│   ├── pitch-sieve.html          v2 · abstract, design space, novel claims
│   ├── fig1_curves.png           }
│   ├── fig3_envelope.png         }  referenced by the HTMLs — keep as siblings
│   ├── fig4_tau.png              }
│   └── h0_expected_outputs.pdf   3 mock H0 reports: expected / best / worst
│
├── sievelib/                     SHARED CORE — used by both studies
│   ├── quant.py                  TurboQuant_mse quantizer (rotation, Lloyd-Max,
│   │                             norm correction). Levels disk-cached.
│   ├── alloc.py                  water-filling, exact-recompute error, per-head
│   │                             metrics, band membership. The heart of both studies.
│   ├── probe.py                  attention capture + KV-cache access   (H0 only)
│   ├── validate.py               3 independent probe validation levels (H0 only)
│   ├── prompts.py                niah / qa / cont prompt families      (H0 only)
│   └── .lloyd_cache.pt           precomputed quantizer levels (~50 s to rebuild)
│
├── h1_simulation/                COMPLETE — synthetic study
│   ├── run_h1.py                 CURRENT. regenerates fig1 + fig4, prints τ table
│   └── superseded_v1/            ARCHIVE — the original first-pass scripts
│       ├── README.md             what is wrong with them (read before using)
│       ├── h1_sim.py             base simulation + convex envelope
│       ├── h1_robust.py          fig1, fig2, fig3 + cascade / correlated-value checks
│       ├── h1_tau.py             fig4 + τ sweep
│       └── fig*_ORIGINAL.png     the exact figures they produced
│
├── h0_measurement/               NEXT — real-model measurement
│   ├── README.md                 full experiment plan + AUDIT FINDINGS. Read first.
│   ├── models.yaml               model registry — add a model here, nothing else
│   ├── prefetch.py               stage weights on the LOGIN node (compute nodes
│   │                             have no internet — this is not optional)
│   ├── run_h0.py                 the measurement
│   ├── report.py                 multi-page PDF + go/no-go verdict
│   ├── mock_report.py            regenerates docs/h0_expected_outputs.pdf
│   ├── quick_test.sh             login-node smoke test, <10 min
│   └── submit.slurm              sbatch array over models
│
└── tests/
    └── test_units.py             35 CPU-only checks. Regression tests for every
                                  bug found in the audit — run before any GPU job.
```

---

## Run, in order

All commands are from the **repo root**.

```bash
# 0. once
python -m venv .venv && source .venv/bin/activate
pip install -U "torch>=2.4" "transformers>=4.48" accelerate \
               pandas pyarrow pyyaml matplotlib huggingface_hub
export HF_HOME=$PWD/.hf_cache

# 1. login node, <10 min: env + weights + tests + 3 validations + tiny run + PDF
./h0_measurement/quick_test.sh qwen3-1.7b

# 2. login node: stage ALL weights (compute nodes have no internet)
python h0_measurement/prefetch.py

# 3. submit
sbatch h0_measurement/submit.slurm

# 4. read the verdict
python h0_measurement/report.py "results/*.parquet" -o reports/h0_report.pdf
```

Regenerating things without a GPU:

```bash
python tests/test_units.py               # 35 checks, ~30 s
python h1_simulation/run_h1.py           # rebuilds docs/fig1 + docs/fig4
python h0_measurement/mock_report.py     # rebuilds docs/h0_expected_outputs.pdf
```

---

## Which file answers which question

| Question | File |
|---|---|
| What is the project and is it worth doing? | `docs/pitch-sieve.html` |
| What exactly gets claimed, scored, and killed? | `docs/proposal-sieve.html` |
| What bugs were found and what did they change? | `h0_measurement/README.md` § AUDIT FINDINGS |
| What will the result look like? | `docs/h0_expected_outputs.pdf` |
| Where is the allocation theorem implemented? | `sievelib/alloc.py` — `waterfill`, `exact_error` |
| Where is the noise actually measured? | `sievelib/quant.py` + `alloc.noise_model` |
| How do I add a model? | `h0_measurement/models.yaml`, append an entry |

---

## Version notes

Kept deliberately short since there is no VCS here.

- **proposal v5** supersedes v4. v4 claimed the allocation gain grows with logit
  spread τ. Measured against the *best* of both corners it is **non-monotonic** —
  peaks near τ≈1.25 at ~13×, gone by τ≈2. v5 revises Fig 1 and Fig 4, adds the
  per-head routing table, and moves acceptance from 60–68% to 45–55%.
- **pitch v2** carries the same correction and states the retraction explicitly.
- **H1 code** was three scripts (`h1_sim.py`, `h1_robust.py`, `h1_tau.py`) built on
  the pre-audit cost model. They are archived under
  `h1_simulation/superseded_v1/` with a README listing their three flaws.
  `h1_simulation/run_h1.py` replaces them and uses the corrected `sievelib` core,
  so the figures and the H0 code share one implementation instead of two that
  could drift.
- **Figures.** `docs/fig1`, `docs/fig3`, `docs/fig4` are current. The original
  first-pass versions of all four, including `fig2_alloc` (the allocation
  staircase, generated under the superseded relative-cost model), are kept as
  `*_ORIGINAL.png` in the archive.
- **Ten defects** were found in a second audit pass and fixed; each has a regression
  test. The two that would have inverted the conclusion: eviction and quantization
  costs were in different units (6× error at τ=2.5), and Lloyd-Max was 4× from
  optimal at 8 bits. Details in `h0_measurement/README.md`.

## Decision rule, one line

H0 asks **what fraction of heads sit in the productive band** — not whether τ is
large, and not what the median head does. `report.py` prints
`heads in band` (gain over the best corner ≥ 2×) and `routed gain`; below 15% is
STOP, above 35% is GO.
