# Measurement H0, Motivation — file manifest

Rate allocation for KV caches. Two studies share one core library.

## Decision rule, one line

H0 asks **what fraction of heads sit in the productive band** — not whether τ is
large, and not what the median head does. `report.py` prints
`heads in band` (gain over the best corner ≥ 2×) and `routed gain`; below 15% is
STOP, above 35% is GO.


---

## Tree

```
.
├── h0_measurement/               <- you are here — real-model measurement
│   ├── README.md                 this file: experiment plan + ctx methodology + validate_with rationale
│   ├── models.yaml               model registry (tiers: debug / main / large) — add a model here, nothing else
│   ├── prefetch.py               stage weights on the LOGIN node — model + its
│   │                             validate_with proxy (compute nodes have no internet)
│   ├── run_h0.py                 the measurement
│   │                               --model TAG --out-dir DIR [--override k=v ...] [--validate-only]
│   ├── report.py                 multi-page PDF + go/no-go verdict + per-head CSV
│   ├── mock_report.py            regenerates docs/h0_expected_outputs.pdf
│   ├── quick_test.sh             LOGIN-node smoke test for ONE model, <10 min
│   ├── quick_test_all.slurm      CPU cluster: prefetch + quick_test.sh for a LIST of models
│   ├── submit_h0.slurm           MAIN campaign — array, 1×H100/task, self-chains the CPU report
│   ├── submit_h0_large_models.slurm   LARGE tier — same structure, 4 GPUs/task (70B / 30B-MoE)
│   ├── report.slurm              SUPERSEDED — report is now a SIEVE_ROLE=report resubmission
│   ├── logs/                     SLURM .out/.err + per-model quick_test logs
│   ├── results/<RUN_ID>/         *.parquet (one per model) + RUN_INFO.txt
│   ├── reports/                  h0_report_<RUN_ID>_<date>.pdf + _per_head.csv
│   └── results_smoke/            throwaway scratch for quick_test.sh
```

---

## Run, in order

All commands are from the **repo root**.

```bash
# 0. once
python -m venv .venv && source .venv/bin/activate
# fastparquet, not pyarrow: on this cluster `pip install pyarrow` resolves to a
# dummy wheel that refuses to build and tells you to `module load arrow`, which
# then has to be loaded before the venv on every compute node. fastparquet ships
# as a real wheel in the local wheelhouse and needs no module.
pip install -U "torch>=2.4" "transformers>=4.48" accelerate \
               pandas fastparquet pyyaml matplotlib huggingface_hub
printf '%s' hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx > .hf_token   # ONE line
# HF_HOME is the single cache knob -- every script derives HF_HUB_CACHE from it
# as $HF_HOME/hub. Point it at shared scratch to share weights across runs.
export HF_HOME=$PWD/.hf_cache

# 1. LOGIN node, <10 min: env + weights + tests + 3 validations + tiny run + PDF.
#    Run once now, and again whenever the venv changes. Not per submission.
./h0_measurement/quick_test.sh qwen3-1.7b

# 2. LOGIN node: stage weights for the models you are about to submit
#    (compute nodes have NO internet). Omit -m to stage every model in the registry.
python h0_measurement/prefetch.py -m qwen3-8b llama31-8b mistral-7b qwen15-moe-a2.7b

# 2b. OPTIONAL — CPU cluster: run prefetch + quick_test.sh for each model you plan
#     to submit, catching a broken tokenizer / gated proxy per model in minutes on
#     CPU instead of at hour 3 of a paid H100 allocation.
sbatch h0_measurement/quick_test_all.slurm qwen3-8b llama31-8b mistral-7b

# 3. ONCE — sbatch cannot create its own --output directory
mkdir -p h0_measurement/{logs,results,reports}

# 4. submit. Each script runs a measurement array (one model per task) and
#    self-chains a CPU-only report job with --dependency=afterok on the array.
sbatch h0_measurement/submit_h0.slurm                       # main tier, 4 models, 1×H100/task
sbatch h0_measurement/submit_h0.slurm qwen3-8b mistral-7b   # ...but pass --array to match:
sbatch --array=0-1 h0_measurement/submit_h0.slurm qwen3-8b mistral-7b

sbatch h0_measurement/submit_h0_large_models.slurm          # large tier: qwen3-30b-a3b-2507, llama33-70b (4 GPUs/task)

# 5. read the verdict. RUN_ID = job<arrayjobid>, printed by step 4 and in RUN_INFO.txt
open h0_measurement/reports/h0_report_<RUN_ID>_<date>.pdf
```


Steps 0–1 are one-time setup. Steps 2 and 4 are what you repeat per campaign;
`pip install` is deliberately *not* folded into submission — a `-U` on every run
could upgrade `transformers` mid-campaign and change what the probe measures.

Every submission is scoped by a **RUN_ID** (`job<arrayjobid>`, identical across the
array's sibling tasks): parquet lands in `h0_measurement/results/<RUN_ID>/`, the
report in `h0_measurement/reports/h0_report_<RUN_ID>_<date>.pdf` plus a
`..._per_head.csv` beside it. Reruns never mix.

Run the report by hand (e.g. after a partial array):

```bash
python h0_measurement/report.py "h0_measurement/results/<RUN_ID>/*.parquet" \
       -o h0_measurement/reports/h0_report_<RUN_ID>.pdf
```

Regenerating things without a GPU:

```bash
python tests/test_units.py               # regression checks, ~30 s
python h1_simulation/run_h1.py           # rebuilds docs/fig1_curves.png + docs/fig4_tau.png
python h0_measurement/mock_report.py     # rebuilds docs/h0_expected_outputs.pdf
```

### Environment knobs

| var | meaning |
|---|---|
| `HF_HOME` | the one cache knob; `HF_HUB_CACHE` is always `$HF_HOME/hub`. Default `$PROJECT_ROOT/.hf_cache`. |
| `HF_TOKEN_FILE` | token file path read by the slurm scripts. Default `$PROJECT_ROOT/.hf_token`. |
| `HF_HUB_OFFLINE` | forced to `1` in the measure stage — compute nodes have no internet. |
| `SIEVE_MODELS` | colon-separated model list (survives `sbatch --export`). |
| `SIEVE_VENV` | venv to activate. Default `$PROJECT_ROOT/.venv`. |
| `SIEVE_NO_REPORT=1` | do not chain the report job. |
| `H0_CORPUS` | directory of real haystack text. Unset ⇒ synthetic filler (see § synthetic-haystack confound). |

---

## Version notes

Kept deliberately short since there is no VCS here.

- **proposal v6 / pitch v3** (`docs/*-v5.1-h0v0.html`) rewrite both documents around
  the completed H0: allocation beats both corners on ~38% of all heads (53% median
  model, 2.06× geo-mean routed gain), and the failures split into two predictable
  phases at opposite ends of the attention-concentration axis. Framing shifts from
  "our method wins" to a phase diagram of when each method wins. The pre-H0 v5 pair
  is kept for reference.
- **proposal v5** supersedes v4. v4 claimed the allocation gain grows with logit
  spread τ. Measured against the *best* of both corners it is **non-monotonic** —
  peaks near τ≈1.25 at ~13×, gone by τ≈2. v5 revises Fig 1 and Fig 4, adds the
  per-head routing table, and moves acceptance from 60–68% to 45–55%.
- **pitch** (`docs/pitch-sieve-v5.html`, body still marked v2) carries the same
  correction and states the retraction explicitly. v3/v4 and the naive proposal
  live in `docs/deprecated/`.
- **H1 code** was three scripts (`h1_sim.py`, `h1_robust.py`, `h1_tau.py`) built on
  the pre-audit cost model. They are kept only under `h1_simulation/superseded_v1/`.
  `h1_simulation/run_h1.py` replaces them and uses the corrected `sievelib` core, so
  the figures and the H0 code now share one implementation instead of two that could drift.
- **Figures.** `fig2_alloc.png` (the allocation staircase) was generated under the
  superseded relative-cost model and is dropped. `fig1`, `fig3`, `fig4` are current
  and reproducible from `h1_simulation/run_h1.py` (1 and 4) and the envelope code;
  the v0 set is archived in `docs/figures_h1_v0/`.
- **H0 SLURM layout.** `submit.slurm` + the `submit_h0.sh` driver are replaced by
  `quick_test_all.slurm` (CPU pre-flight), `submit_h0.slurm` (main tier, 1 GPU/task),
  and `submit_h0_large_models.slurm` (large tier, 4 GPUs/task). Each embeds its report
  stage as a `SIEVE_ROLE=report` self-resubmission, so `report.slurm` is vestigial.
  Logs/results/reports now live under `h0_measurement/`, not the repo root.
- **Ten defects** were found in a second audit pass and fixed; each has a regression
  test. The two that would have inverted the conclusion: eviction and quantization
  costs were in different units (6× error at τ=2.5), and Lloyd-Max was 4× from
  optimal at 8 bits. One regression check per defect in `tests/test_units.py`.





## Context length methodology

**Per-model ctx audit** (checked against live `max_position_embeddings` / `rope_scaling`):

| tag | ctx | max_pos | rope_type | verdict |
|---|---|---|---|---|
| qwen3-1.7b | 8,192 | 40,960 | default | ok (debug) |
| qwen3-8b | 40,960 | 40,960 | default | at limit |
| llama31-8b | 131,072 | 131,072 | llama3, factor 8 | ok — genuinely extended |
| mistral-7b | 32,768 | 32,768 | default | at limit |
| qwen15-moe-a2.7b | 32,768 | 32,768 | default | at limit |
| qwen3-30b-a3b | 40,960 | 40,960 | default | at limit |
| qwen3-30b-a3b-2507 | 131,072 | 262,144 | default | ok, headroom |
| llama33-70b | 131,072 | 131,072 | llama3, factor 8 | ok |

No entry currently exceeds its native window, and only the two Llama entries carry
real `rope_scaling`; every Qwen/Mistral entry is `rope_type: "default"`, so their
`ctx` is a hard cap, not advisory.

**Does ctx affect the methodology?** Yes, on two axes.

*What's L-invariant.* In `sievelib/alloc.py` the budget is a rate — `budget_bits *
w2.numel()` — and the eviction corner keeps a fraction — `B * L / maxb` tokens.
Every reported gain is a dimensionless error ratio, so comparing models at
different ctx is not meaningless on its face.

*What's not.* Two quantities in `sensitivity_metrics` move systematically with L
and both feed the go/no-go statistic:
- `tau = sd.std()` grows with L. proposal v5 already found gain-over-best-corner is
  **non-monotonic in tau** (peaks ~1.25, gone by ~2) — so ctx slides each head along
  exactly the axis the headline result is non-monotonic in.
- `eff_frac = n95 / L` falls with L as attention concentrates, shrinking the
  eviction corner's cost and pushing `in_band` toward 0.

Both effects push the same direction: **longer ctx -> fewer heads in band -> closer
to STOP.** ctx is therefore a second independent variable sitting on top of
`F_STOP=0.15` / `F_GO=0.35`, not a nuisance knob.

**Higher-priority issue found while checking this: the synthetic-haystack
confound.** `prompts.build` falls back to synthetic filler whenever `H0_CORPUS` is
unset, and the filler pool is 8 sentences / 124 tokens of unique text. Measured
repetition counts:

```
     ctx  prompt tok  synthetic   x each sentence repeats
    8192       7,536       True                      61x
   32768      30,146       True                     243x
   40960      37,683       True                     304x
  131072     120,586       True                     972x
```

At `ctx: 131072` the model attends over 124 tokens tiled ~972x — induction heads
lock onto the repeats, so `tau` and `eff_frac` are computed on a degenerate
distribution, and the artifact scales with ctx, perfectly confounded with the real
L-dependence above. The code records `synthetic` per row and prints a warning, but
nothing blocks a full run on synthetic data. **Set `H0_CORPUS` before choosing ctx
values for a real campaign** — comparisons made before that point aren't
informative about long-context behavior.

**Recommendation.**
- Match the `main` tier to the largest ctx all four main-tier models support
  natively (currently 32,768) — `report.py`'s cross-model boxplot compares models
  at whatever ctx each ran at, so leaving them mismatched (32,768 vs 131,072) means
  a difference in `gain_best3` can't be attributed to architecture vs. context length.
- Turn L into a deliberate axis on one model (e.g. llama31-8b at {8k, 32k, 128k})
  instead of an incidental difference across models — this also gives an on-hardware
  test of the v5 tau non-monotonicity claim.
- `report.py`'s `page_compare` currently groups by `model` only
  (`ph.groupby("model")`) and labels the panel with `g['ctx'].iloc[0]` — a per-model
  ctx sweep needs `groupby(["model", "ctx"])` there or two ctx values for one model
  will silently pool into a single mislabeled page.


# Methodology: Validating with a smaller model

What validate_with is actually for
It's instrument calibration, not a scientific shortcut. Worth being precise, because the naming invites a misreading:

- L1/L3 validate the probe — that sieve_probe is a transparent drop-in (L1: identical LM logits vs sdpa) and that its recomputed attention weights match HF's independent eager path (L3: catches wrong GQA expansion, scaling, mask, softmax axis).

- The measurement itself always runs on the real model. No H0 result is ever extrapolated from the 0.6B.

So the design isn't claiming "0.6B head sensitivity ≈ 8B head sensitivity." It's claiming "my measuring device is wired correctly, and wiring correctness doesn't depend on parameter count." That claim is sound in kind, and the cost argument is real — L1 loads the model twice in float32, L3 twice more. For the 70B that's ~280GB per load; infeasible. The fp32 + tight-tolerance trade (1e-4 vs the 5e-2 bf16 forces) is also a genuine win — 5e-2 on attention weights would hide real indexing bugs.