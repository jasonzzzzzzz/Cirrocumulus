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
│   ├── prefetch_corpus.py        stage the HAYSTACK on the LOGIN node — PG-19 books
│   │                             from Project Gutenberg + MANIFEST.json (--verify is
│   │                             offline, so compute nodes can check it)
│   ├── run_h0.py                 the measurement
│   │                               --model TAG --out-dir DIR [--override k=v ...] [--validate-only]
│   ├── report.py                 multi-page PDF + go/no-go verdict + per-head CSV
│   ├── mock_report.py            regenerates docs/h0_expected_outputs.pdf
│   ├── diagnose_generation.py    NOT part of the measurement — a bisect for the
│   │                             0/26 needle-retrieval failure: sdpa vs probe x
│   │                             single vs chunked prefill x raw vs chat template
│   ├── quick_test.sh             LOGIN-node smoke test for ONE model, <10 min
│   ├── quick_test_all.slurm      CPU cluster: prefetch + quick_test.sh for a LIST of models
│   ├── submit_h0.slurm           MAIN campaign — array, 1×H100/task, self-chains the CPU report
│   ├── submit_h0_large_models.slurm   LARGE tier — same structure, 4 GPUs/task (70B / 30B-MoE)
│   ├── submit_h0_ctx_sweep.slurm ONE model swept across CONTEXT LENGTHS — array
│   │                             axis is ctx, not model (§ matched context & ctx sweep)
│   ├── submit_validity.slurm     INPUT-validity probe — niah only, no bit sweep,
│   │                             minutes/model. Validates an EXISTING run, because
│   │                             the haystack is seeded on prompt_idx alone.
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

# 2a. LOGIN node: stage the HAYSTACK. Without this, tier main/large REFUSES to run
#     (see § synthetic-haystack confound). ~25 MB, one minute, once per machine.
python h0_measurement/prefetch_corpus.py --check-ctx 131072 --n-prompts 6
export H0_CORPUS=$PWD/.h0_corpus/pg19          # the slurm scripts default to this

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

#    Optional eviction-corner knobs (§ Eviction corner). Unset = models.yaml
#    defaults, which is what the headers' --mem/--time are sized for.
SIEVE_EVICTORS='oracle,last_step,accum' \
  sbatch h0_measurement/submit_h0.slurm                     # a cheaper corner set
SIEVE_CORNER_POLICIES=frac \
  sbatch h0_measurement/submit_h0.slurm                     # drop the E1 abs policy

# 4b. the INPUT-validity probe. Answers "did the haystack induce retrieval?" --
#     niah only, no bit sweep, minutes per model. The haystack is seeded on
#     prompt_idx alone, so at the same ctx this reads byte-identical text to the
#     run in step 4 and its verdict applies to that run. report.py stamps the
#     verdict UNKNOWN until it has this evidence.
sbatch --array=0-3 h0_measurement/submit_validity.slurm \
       qwen3-8b llama31-8b mistral-7b qwen15-moe-a2.7b

# 5. read the verdict. RUN_ID = job<arrayjobid>, printed by step 4 and in RUN_INFO.txt
#    Pass the validity parquet too -- report.py keeps the frames separate and
#    uses the probe only for the gate.
python h0_measurement/report.py \
       "h0_measurement/results/<RUN_ID>/*.parquet" \
       "h0_measurement/results/validity<VJOBID>/*.parquet" \
       -o h0_measurement/reports/h0_report_<RUN_ID>.pdf
```

**Submit every GPU job from `trig-login01`.** This cluster rejects a GPU request
made from a CPU login node (`tri-login*`) with an error that does not name the
script: `GPU resources requested from a CPU login node`.


Steps 0–1 are one-time setup. Steps 2 and 4 are what you repeat per campaign;
`pip install` is deliberately *not* folded into submission — a `-U` on every run
could upgrade `transformers` mid-campaign and change what the probe measures.

Every submission is scoped by a **RUN_ID** (`job<arrayjobid>`, identical across the
array's sibling tasks): parquet lands in `h0_measurement/results/<RUN_ID>/`, the
report in `h0_measurement/reports/h0_report_<RUN_ID>_<date>.pdf` plus a
`..._per_head.csv` beside it. Reruns never mix.

Run the report by hand (e.g. after a partial array):

```bash
python h0_measurement/report.py \
       "h0_measurement/results/job852849/*.parquet" \
       "h0_measurement/results/job852851/*.parquet" \
       -o h0_measurement/reports/h0_report_job852849+job852851.pdf

```

---

## A complete run, start to finish

Every command below is literal — no placeholders except the two RUN_IDs, which
the tooling prints. Copy-paste from a clean checkout on `trig-login01`.

```bash
cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant

# ---- 1. environment (once per machine) ----------------------------------------
python -m venv .venv && source .venv/bin/activate
pip install -U "torch>=2.4" "transformers>=4.48" accelerate \
               pandas fastparquet pyyaml matplotlib huggingface_hub
printf '%s' hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx > .hf_token
export HF_HOME=$PWD/.hf_cache

# ---- 2. CPU-only checks (no weights, no GPU, ~1 min) --------------------------
python tests/test_units.py

# ---- 3. stage weights + haystack (LOGIN node: compute nodes have no internet) --
python h0_measurement/prefetch.py -m qwen3-1.7b llama31-8b
python h0_measurement/prefetch_corpus.py --check-ctx 131072 --n-prompts 6
export H0_CORPUS=$PWD/.h0_corpus/pg19

# ---- 4. smoke the whole path on the debug tier (<10 min, CPU or 1 GPU) --------
./h0_measurement/quick_test.sh qwen3-1.7b

# ---- 5. a real single-model run by hand, before spending array hours ----------
#     n_decode=4 quant_every=1 makes 3 of the 4 decode steps carry full lagged
#     history, which is what you want when checking the practical corner (§ below).
mkdir -p h0_measurement/{logs,results,reports,results_smoke}
python h0_measurement/run_h0.py --model qwen3-1.7b \
       --out-dir h0_measurement/results_smoke \
       --override n_decode=4 quant_every=1

# ---- 6. confirm the practical corner actually populated -----------------------
python - <<'PY'
import glob, pandas as pd
df = pd.concat([pd.read_parquet(f)
                for f in glob.glob("h0_measurement/results_smoke/h0_*.parquet")])
q = df[df.quantized]
print("rows", len(q), "| corners:", q["evictors"].iloc[0])
print("practical evictors scored per row:", q["n_practical"].value_counts().to_dict())
print("gain_best_practical3 null frac: %.3f" % q["gain_best_practical3"].isna().mean())
print("band   oracle %.1f%%  ->  practical %.1f%%"
      % (100*(q.gain_best3 >= 2).mean(), 100*(q.gain_best_practical3 >= 2).mean()))
print("winning corner:", q["best_evictor3"].value_counts().to_dict())
print("K* median %.0f tok = %.1f%% of the corner's budget"
      % (q.kstar3.median(), 100*q.kstar_frac3.median()))
PY
# EXPECT: null frac 0.000, and the practical band fraction well above the oracle
# one. A null frac of 1.000 means no practical corner ran -- that is the bug this
# replaced, and it is the one thing to check before spending array hours.
#
# Real output from the command above (qwen3-1.7b, ctx 2048, CPU, ~15 s):
#   rows 1344 | corners: oracle,last_step,accum,window,recency
#   practical evictors scored per row: {4: 896, 1: 448}
#   gain_best_practical3 null frac: 0.000
#   band   oracle 2.5%  ->  practical 32.6%
#   winning corner: {'last_step': 527, 'recency': 495, 'accum': 233, 'window': 89}
#   K* median 718 tok = 100.0% of the corner's budget
#
# `{4: 896, 1: 448}` is expected, not a fault: step 0 has no lagged history, so
# only `recency` scores there. Filter on `n_practical` for a clean comparison.
# K* = 100% is also expected at ctx 2048 -- the budget only has slack to detect
# at long context, which is the whole point of E1.

# ---- 7. submit the campaign --------------------------------------------------
sbatch h0_measurement/submit_h0.slurm                  # prints RUN_ID = job<arrayjobid>
sbatch --array=0-3 h0_measurement/submit_validity.slurm \
       qwen3-8b llama31-8b mistral-7b qwen15-moe-a2.7b # prints VJOBID

# ---- 8. the verdict ----------------------------------------------------------
python h0_measurement/report.py \
       "h0_measurement/results/<RUN_ID>/*.parquet" \
       "h0_measurement/results/validity<VJOBID>/*.parquet" \
       -o h0_measurement/reports/h0_report_<RUN_ID>.pdf
```

Step 8 prints one line per `(model, ctx)`; the bracket names which corner the
band fraction was measured against:

```
  llama31-8b                128k  band  31.4% [practical] routed 2.71x  -> NARROW
```

`[practical]` = a deployable evictor. `[oracle]` = that parquet has no practical
columns (a pre-fix run), and the number is against a baseline no system can field.

---

## Eviction corner: which baseline the verdict uses (bug 2)

The corner the interior is compared against has two axes, both configurable in
`models.yaml` `defaults:` or per run with `--override`
(`bugs/2_towards_real_evictor/fix.md`).

**WHO — `evictors`.** `oracle` ranks by the current step's `a·‖v−o‖`; it needs the
attention weights eviction exists to avoid computing, so it is an upper bound, not
a baseline. It is kept and reported as that bound, but the **verdict keys off the
practical corners**.

| name | score | anchor |
|---|---|---|
| `oracle` | current-step `a·‖v−o‖` | upper bound (not deployable) |
| `last_step` | previous step's attention | TOVA |
| `accum` | running sum of attention received | H2O |
| `window` | last `window` steps, max-pooled over `pool` neighbours | SnapKV |
| `recency` | sinks + newest, no attention stats at all | StreamingLLM (floor) |

**HOW MUCH — `corner_policies`.** `frac` keeps `B·L/maxb` tokens (the literature's
definition, linear in `L`); `abs` caps that at `max(corner_kappa·n95, corner_floor)`,
a multiple of the head's own measured support.

```bash
# the shipped default: all five corners, both budget policies
python h0_measurement/run_h0.py --model qwen3-1.7b --out-dir h0_measurement/results_smoke

# reproduce the PRE-FIX baseline exactly (oracle only, fractional budget)
python h0_measurement/run_h0.py --model qwen3-1.7b --out-dir h0_measurement/results_smoke \
       --override evictors=oracle corner_policies=frac kstar=false

# fastest honest run: oracle bound + H2O, fractional budget only
python h0_measurement/run_h0.py --model qwen3-8b --out-dir h0_measurement/results/manual \
       --override evictors=oracle,accum corner_policies=frac

# SnapKV with a longer window and wider pooling (quote it: ';' is a shell metachar)
python h0_measurement/run_h0.py --model qwen3-8b --out-dir h0_measurement/results/manual \
       --override 'evictors=oracle;accum;window:window=8,pool=13'

# two SnapKV variants in one run -- @alias keeps their columns apart
python h0_measurement/run_h0.py --model qwen3-8b --out-dir h0_measurement/results/manual \
       --override 'evictors=oracle;window:window=2@w2;window:window=16@w16'

# tune the absolute-support cap, drop the K* ladder
python h0_measurement/run_h0.py --model qwen3-8b --out-dir h0_measurement/results/manual \
       --override corner_policies=frac,abs corner_kappa=8 corner_floor=512 kstar=false

# paper names work as aliases: tova/h2o/snapkv/streamingllm
python h0_measurement/run_h0.py --model qwen3-8b --out-dir h0_measurement/results/manual \
       --override evictors=oracle,tova,h2o,snapkv
```

Whole-campaign overrides go through the slurm scripts the same way as `SIEVE_CTX`
— pass them on the `run_h0.py` line inside the script, or edit `models.yaml`
`defaults:` once and submit normally.

A bad name, option or policy fails **before** the tokenizer loads and long before
a GPU allocation is held:

```bash
$ python h0_measurement/run_h0.py --model qwen3-1.7b --override evictors=h2o_typo
bad eviction-corner config: unknown evictor 'h2o_typo'; available: ['oracle', 'accum', 'last_step', 'recency', 'window']
  evictors='h2o_typo'  corner_policies=['frac', 'abs']

$ python h0_measurement/run_h0.py --model qwen3-1.7b --override corner_policies=nope
bad eviction-corner config: unknown corner policy 'nope'; use frac / abs
  evictors=['oracle', 'last_step', 'accum', 'window', 'recency']  corner_policies='nope'

$ python h0_measurement/run_h0.py --model qwen3-1.7b --override 'evictors=oracle;window:pool=4'
bad eviction-corner config: pool must be odd and >= 1, got 4
  evictors='oracle;window:pool=4'  corner_policies=['frac', 'abs']
```

### Reading the new report output

`report.py` gains three blocks per model. `oracle_adv` is what each evictor loses
to the oracle — a publishable per-head number in its own right:

```
    HEADS IN BAND   31.4%  (gain over best PRACTICAL corner >= 2.0x at 3 b/token)
    vs ORACLE corner (upper bound, NOT deployable):  12.5% in band, median 1.31x
                                                     -> honest corner moves it +18.9 pts
    CORNER GRID @3b   (median gain of the interior over each corner; higher = weaker corner)
                          frac         abs     oracle_adv
      accum              7.41x       7.902x        5.66x
      last_step          6.98x       7.11x         5.33x
      window             8.02x       8.30x         6.13x
      recency           11.40x      11.62x         8.71x
      oracle             1.31x       1.42x
    CORNER SPEND @3b  frac: 3.00 b/tok (49,152 tok)   abs: 0.33 b/tok (5,400 tok)
    K* @3b         median 4,610 tokens (9.4% of the corner's budget)   = 3.1x n95
    VERDICT: NARROW — [vs accum/frac; oracle bound 5.66x]  31% of heads are in band ...
```

### On the cluster — `SIEVE_EVICTORS` / `SIEVE_CORNER_POLICIES`

The three measurement scripts take the same two axes as environment variables,
exactly like `SIEVE_CTX`. `submit_validity.slurm` deliberately has **no** corner
knobs: `--validity-only` forms no corners at all, so it prints
`eviction corners: (none)` and writes empty `evictors` / `corner_policies`
columns — a validity parquet can never be mistaken for one that measured a corner.

```bash
# ---- main tier -----------------------------------------------------------------
sbatch h0_measurement/submit_h0.slurm                        # models.yaml defaults
SIEVE_EVICTORS='oracle,last_step,accum' \
  sbatch h0_measurement/submit_h0.slurm
SIEVE_CORNER_POLICIES=frac \
  sbatch h0_measurement/submit_h0.slurm
# reproduce the PRE-FIX campaign exactly (oracle corner, fractional budget only)
SIEVE_EVICTORS=oracle SIEVE_CORNER_POLICIES=frac \
  sbatch h0_measurement/submit_h0.slurm
# combines with SIEVE_CTX -- both land in ONE --override, so neither is dropped
SIEVE_CTX=32768 SIEVE_EVICTORS='oracle,accum' \
  sbatch h0_measurement/submit_h0.slurm

# a spec with options: quote it, and use ';' between corners (',' is inside the spec)
SIEVE_EVICTORS='oracle;accum;window:window=8,pool=13' \
  sbatch h0_measurement/submit_h0.slurm

# ---- large tier ----------------------------------------------------------------
sbatch h0_measurement/submit_h0_large_models.slurm           # 4×H100, --mem=320G
SIEVE_EVICTORS='oracle,last_step,accum,recency' \
  sbatch h0_measurement/submit_h0_large_models.slurm         # drops `window`: 28 B/slot -> 11

# ---- ctx sweep -----------------------------------------------------------------
sbatch h0_measurement/submit_h0_ctx_sweep.slurm llama31-8b
SIEVE_CORNER_POLICIES=frac,abs \
  sbatch h0_measurement/submit_h0_ctx_sweep.slurm llama31-8b # E1 is a ctx question

# ---- validity probe: no corner knobs, unchanged resources ----------------------
sbatch --array=0-3 h0_measurement/submit_validity.slurm \
       qwen3-8b llama31-8b mistral-7b qwen15-moe-a2.7b
```

`SIEVE_EVICTORS` is **not** forwarded to the chained report job. `sbatch --export`
takes a comma-separated list, so `SIEVE_EVICTORS='oracle,accum'` would arrive as
two variable assignments — the same trap `SIEVE_MODELS` colon-encodes around. The
report does not need it: every parquet row carries `evictors` and
`corner_policies`, and `report.py` names the corner on each verdict line from
those. `RUN_INFO.txt` records them too.

### Which corner did THIS parquet use?

Three places record it, in increasing authority.

**1. The `.json` sidecar beside every parquet — always written, authoritative.**
One per result file, written by the task that produced it, so it records what was
actually resolved and run (not what the submitter asked for):

```bash
$ ls h0_measurement/results/job861234/
h0_llama31-8b_131072.parquet   h0_llama31-8b_131072.json   RUN_INFO.txt

$ python -c "import json;print(json.load(open('h0_measurement/results/job861234/h0_llama31-8b_131072.json'))['corner'])"
{'evictors': ['oracle','last_step','accum','window','recency'],
 'practical': ['last_step','accum','window','recency'], 'policies': ['frac','abs'],
 'kappa': 4.0, 'floor': 256, 'kstar': True, 'kstar_tol': 0.1,
 'state_bytes_per_layer_head_token': 28, 'tag': 'or-la-ac-wi-re_fa'}
```

The sidecar also carries the full merged config (`budgets`, `bit_list`,
`quant_every`, `corpus_sha`, …), so a result is reproducible from its own output.

**2. `RUN_INFO.txt` — per array, now resolved rather than echoed.** It used to
print `evictors: none (models.yaml defaults)`, which does not say what those
defaults *were*. It now records the knob **and** the effective corner per model:

```
evictors knob: unset
corner policies knob: unset

effective corner per model (authoritative copy: the .json beside each parquet)
  qwen3-8b               ctx=40960   or-la-ac-wi-re_fa
                         evictors=oracle,last_step,accum,window,recency  policies=frac,abs  kappa=4 floor=256  kstar_tol=0.1
```

**3. The filename — opt-in, off by default.** Set `corner_in_filename: true`
(models.yaml or `--override`) to put the fingerprint in the name. Enable it when
one results dir holds several corner configurations: without it, two configs of
the same model at the same ctx produce the **same filename** and collide.

```bash
python h0_measurement/run_h0.py --model qwen3-8b --out-dir results/sweep \
       --override corner_in_filename=true
# -> h0_qwen3-8b_40960__or-la-ac-wi-re_fa.parquet   (+ the matching .json)

SIEVE_EVICTORS='oracle,accum' \
  sbatch h0_measurement/submit_h0.slurm      # add corner_in_filename to models.yaml
                                             # defaults to tag a whole campaign
```

Tag grammar: two letters per evictor, one per policy, then only the parameters
that actually applied — `or-la-ac-wi-re_fa`, `or-ac_f`, `or-ac_fa_k8_f512`,
`..._noks` when K* is off. It disambiguates; it does not reconstruct (use the
sidecar for that). Off by default because turning it on changes every filename.

Everything is also on **every row** of the parquet — `evictors`,
`corner_policies`, `corner_kappa`, `corner_floor`, `kstar_tol` — which is what
`report.py` reads; it never consults RUN_INFO, models.yaml, or the environment.
Pool two runs with different settings and it warns rather than majority-voting.

### Resources: the corner costs host RAM

The lagged evictor state is **host** RAM, per (layer, head), linear in ctx:

```
bytes = n_layers × n_heads × ctx × slot        slot = Σ (4·n_bufs + 1) per evictor
```

`slot` is 28 B for the shipped default set; `window` alone is 17 of those.

| corner set | slot | llama31-8b @128k | llama33-70b @128k |
|---|---|---|---|
| default (`oracle,last_step,accum,window,recency`) | 28 B | 1.3 GB | **18.8 GB** |
| `oracle,last_step,accum,recency` | 11 B | 0.5 GB | 7.4 GB |
| `oracle` | 0 B | 0 | 0 |

The headers were raised for the default set — `--mem=96G` on the main and sweep
scripts, `--mem=320G` on the large one, and `--time` up ~50% for the extra corner
passes. Both the large and sweep scripts preflight this and **refuse to start**
rather than getting OOM-killed an hour in:

```
  corners=['oracle', 'last_step', 'accum', 'window', 'recency']  lagged state=18.8 GB host RAM (28 B per layer-head-token)
ERROR: the lagged evictor state needs 18.8 GB of host RAM, more than a quarter of
  the 31 GB allocated, and the weights need the rest. Raise --mem, or shrink the
  corner set: SIEVE_EVICTORS='oracle,last_step,accum,recency' drops `window`
  (17 of 28 B/slot), SIEVE_EVICTORS=oracle drops all of it.
```

**If you override resources on the sbatch line, override `--mem` too.** The sweep
header is sized for a 1-GPU model; a tier-large model swept to 128k needs the
large-tier memory as well as its GPUs:

```bash
sbatch --gpus-per-node=4 --cpus-per-task=32 --mem=320G --time=03:30:00 --array=0-2 \
       h0_measurement/submit_h0_ctx_sweep.slurm llama33-70b 32768 65536 131072
```

### Caveats that change how you configure a run

- **`n_decode` must be ≥ 2.** Lagged evictors have no history on step 0, so only
  `recency` scores there. `--validate-only` sets `n_decode=1` and produces no
  lagged columns — that is a probe check, not a measurement.
- **`quant_every` thins the comparison.** Practical columns only exist on
  quantization steps, and `do_quant = step % quant_every == 0` makes step 0 always
  a quant step — the one step with no history. With the defaults (`n_decode: 8`,
  `quant_every: 4`) the quant steps are 0 and 4, so half the quant rows carry only
  `recency`. Filter on `n_practical`, or run `--override quant_every=2`.
- **`accum` accumulates from decode, not prefill** (the probe only captures decode
  queries), so it is weaker than a deployed H2O and biases gains *up*. Raise
  `n_decode` when it is the headline.
- **RAM.** Each evictor holds `ctx × (4·n_bufs + 1)` bytes per (layer, head);
  `window` holds `window` times what the others do. On llama33-70b @ 128k the
  default set is ~18.8 GB of host RAM. Drop `window` first:
  `--override evictors=oracle,last_step,accum,recency`.
- **`gain_best_practical ≥ gain_best` is NOT a per-head theorem.** `oracle` is an
  oracle only w.r.t. the first-order proxy `w² = (a·‖v−o‖)²`; the reported error
  is exact recomputation, and the proxy ranking is not the argmin of the exact
  error. On the real qwen3-1.7b run above a practical corner beat the oracle on
  **15.9% of head-rows**, by up to 4×. The direction holds in *aggregate* (median
  `err_practical/err_evict` = 1.19, band 2.5% → 32.6%) — so compare distributions,
  and do not treat a single falling head as a bug.

## Matched context & ctx sweep (bug 3)

The φ-window couples ctx into the verdict: eviction keeps a fixed *fraction* of
tokens, so a head whose support is ~constant in absolute tokens gets relatively
cheaper to evict as ctx grows. Comparing models measured at different ctx
therefore confounds architecture with context length
(`bugs/3_context_sweep_and_reports/`). Two campaigns fix the comparison:

### 1. Matched-ctx campaign — `SIEVE_CTX`

`SIEVE_CTX` makes **every** model in a submission run at one context length via
`--override ctx=`. Unset means today's per-model registry ctx. 32,768 is the
largest value all main-tier models support natively (see the ctx audit below);
a `SIEVE_CTX` above any submitted model's registry ctx fails that task fast.

```bash
# the matched-32k cross-model table: both tiers at ctx 32768
SIEVE_CTX=32768 sbatch h0_measurement/submit_h0.slurm
SIEVE_CTX=32768 sbatch h0_measurement/submit_h0_large_models.slurm

# a subset, same rules as before (--array must match the model count)
SIEVE_CTX=32768 sbatch --array=0-1 h0_measurement/submit_h0.slurm qwen3-8b mistral-7b

# WRONG — mistral-7b caps at 32768, so this task exits with a clear error:
SIEVE_CTX=65536 sbatch h0_measurement/submit_h0.slurm
```

Each script still chains its own per-tier report. For the **single** matched-32k
table across both tiers, combine the two runs by hand (RUN_IDs are printed at
submission and recorded in each `RUN_INFO.txt`):

```bash
python h0_measurement/report.py \
       "h0_measurement/results/<SMALL_RUN_ID>/*.parquet" \
       "h0_measurement/results/<LARGE_RUN_ID>/*.parquet" \
       -o h0_measurement/reports/h0_matched32k.pdf
```

### 2. Ctx sweep — `submit_h0_ctx_sweep.slurm`

One model, one array task per **ctx value**. Arg 1 = model tag, remaining args =
ctx list (default `32768 65536 131072`, matching the header's `--array=0-2`).
Output parquet is `h0_<tag>_<ctx>.parquet`, so all tasks share one results dir;
the chained report includes the **“Band fraction vs context length”** page — the
slope figure this sweep exists to draw (it appears whenever a model shows up at
≥ 2 ctx values).

The registry ctx values are native RoPE limits, not advisory, and the script
rejects any ctx above the model's cap. Per model:

```bash
# ---- small tier (1×H100/task, the header as-is) --------------------------------
# llama31-8b — cap 131,072; the ONLY main-tier model that can sweep past 40,960,
# hence the flagship sweep. Default = the plan's 32k/64k/128k:
sbatch h0_measurement/submit_h0_ctx_sweep.slurm llama31-8b
# ...or the full 4k -> 128k ladder (everything below 32k combined costs less
# than the 32k point alone; cost scales ~quadratically with ctx):
sbatch --array=0-5 h0_measurement/submit_h0_ctx_sweep.slurm \
       llama31-8b 4096 8192 16384 32768 65536 131072

# qwen3-8b — cap 40,960 (native, no rope_scaling):
sbatch --array=0-4 h0_measurement/submit_h0_ctx_sweep.slurm \
       qwen3-8b 4096 8192 16384 32768 40960

# mistral-7b — cap 32,768:
sbatch --array=0-3 h0_measurement/submit_h0_ctx_sweep.slurm \
       mistral-7b 4096 8192 16384 32768

# qwen15-moe-a2.7b — cap 32,768 (the ratio-1 GQA control):
sbatch --array=0-3 h0_measurement/submit_h0_ctx_sweep.slurm \
       qwen15-moe-a2.7b 4096 8192 16384 32768

# qwen3-1.7b — debug tier, cap 8,192; only for smoke-testing the sweep plumbing:
sbatch --array=0-1 h0_measurement/submit_h0_ctx_sweep.slurm qwen3-1.7b 4096 8192

# ---- large tier: override resources on the sbatch line (header is 1-GPU) ------
# --mem MUST be overridden too: the header's 96G is sized for a 1-GPU model, and
# the lagged evictor state alone is ~19 GB for the 70B at 128k (§ Resources).
# llama33-70b — cap 131,072; needs 4×H100 at 128k (weights 141 GB + KV 43 GB):
sbatch --gpus-per-node=4 --cpus-per-task=32 --mem=320G --time=03:30:00 --array=0-2 \
       h0_measurement/submit_h0_ctx_sweep.slurm llama33-70b 32768 65536 131072

# qwen3-30b-a3b-2507 — cap 131,072 (262k native); 2×H100 is the real floor.
# NOTE: --array must match the ctx COUNT -- five values below, so 0-4, not 0-2.
sbatch --gpus-per-node=2 --cpus-per-task=32 --mem=200G --time=03:30:00 --array=0-4 \
       h0_measurement/submit_h0_ctx_sweep.slurm qwen3-30b-a3b-2507 8192 16384 32768 65536 131072

# qwen3-30b-a3b — cap 40,960 (no YaRN; use the -2507 entry for real 128k):
sbatch --gpus-per-node=2 --cpus-per-task=32 --mem=200G --time=03:30:00 --array=0-4 \
       h0_measurement/submit_h0_ctx_sweep.slurm qwen3-30b-a3b 4096 8192 16384 32768 40960
```

`--gpus-per-node=4 --array=0-2` also works on a *small* model (e.g. llama31-8b —
`device_map=auto` shards to whatever it is given), but it parks 3 idle H100s per
task; a VRAM preflight in the script tells you when extra cards are actually
required rather than wasted. The mismatch trap is the same as the sibling
scripts: **`--array` must be `0-(N_ctx - 1)`** — a wrong range is caught before
any GPU work.

### 3. Getting the report

Each submission self-chains its report (skip with `SIEVE_NO_REPORT=1`):

- matched-ctx runs → `reports/h0_report_<RUN_ID>_<date>.pdf` per tier, plus the
  combined-table command above;
- the sweep → `reports/h0_ctxsweep_<tag>_<RUN_ID>_<date>.pdf`, whose
  ctx-slope page is the figure.

By hand — any mix of globs works, and pooling a sweep with a matched-ctx
campaign puts *both* the cross-model table and the slope page in one PDF:

```bash
python h0_measurement/report.py \
       "h0_measurement/results/<SWEEP_RUN_ID>/*.parquet" \
       "h0_measurement/results/<MATCHED_RUN_ID>/*.parquet" \
       -o h0_measurement/reports/h0_ctx_study.pdf
```

---

Regenerating things without a GPU:

```bash
python tests/test_units.py               # regression checks
python h1_simulation/run_h1.py           # rebuilds docs/fig1_curves.png + docs/fig4_tau.png
python h0_measurement/mock_report.py     # rebuilds docs/h0_expected_outputs.pdf
```

Run these **inside the venv** (`source .venv/bin/activate`, or call
`.venv/bin/python` directly) — the bare login-node interpreter has no `pandas`,
`torch` or `transformers`, and the failure looks like a missing module rather than
a missing venv.

The login node also enforces a CPU-time rlimit that can kill a single full
`test_units.py` run partway (it exits 152 with no summary line). Split it:

```bash
.venv/bin/python -c "
import sys; sys.argv=['x']
import tests.test_units as T
for t in (T.test_lloyd_max, T.test_rotation_and_chunking, T.test_gqa_mapping,
          T.test_chunked_prefill, T.test_monotone_error, T.test_units_regression): t()
print('FAILS:', T.fails)"

.venv/bin/python -c "
import sys; sys.argv=['x']
import tests.test_units as T
for t in (T.test_bias_regression, T.test_waterfill_budget, T.test_exact_error_guards,
          T.test_end_to_end, T.test_p0_alignment, T.test_e2_registry,
          T.test_e1_budget_policy, T.test_corner_columns, T.test_corpus_prompts,
          T.test_family_gate, T.test_probe_chunked_prefill, T.test_needle_span,
          T.test_validity_gate): t()
print('FAILS:', T.fails)"
```

### Environment knobs

| var | meaning |
|---|---|
| `HF_HOME` | the one cache knob; `HF_HUB_CACHE` is always `$HF_HOME/hub`. Default `$PROJECT_ROOT/.hf_cache`. |
| `HF_TOKEN_FILE` | token file path read by the slurm scripts. Default `$PROJECT_ROOT/.hf_token`. |
| `HF_HUB_OFFLINE` | forced to `1` in the measure stage — compute nodes have no internet. |
| `SIEVE_MODELS` | colon-separated model list (survives `sbatch --export`). |
| `SIEVE_CTX` | matched-context override for `submit_h0.slurm` / `submit_h0_large_models.slurm`: every model runs at this ctx via `--override ctx=`. Rejected per task if above a model's registry ctx. Unset ⇒ per-model registry ctx (§ matched context & ctx sweep). |
| `SIEVE_MODEL`, `SIEVE_CTXS` | `submit_h0_ctx_sweep.slurm` internals: the swept model tag and colon-separated ctx list, carried into the report resubmission. Set them via the command line, not by hand. |
| `SIEVE_EVICTORS` | eviction-corner override for the three MEASUREMENT scripts — `--override evictors=`. Quote it; use `;` between corners when a spec carries `k=v` options. NOT forwarded to the chained report (`sbatch --export` is comma-separated); the parquet's `evictors` column carries it instead. No effect on `submit_validity.slurm`, which measures no corner. Unset ⇒ models.yaml defaults (§ Eviction corner). |
| `SIEVE_CORNER_POLICIES` | budget-policy override, `frac` / `abs` / `frac,abs` — `--override corner_policies=`. Same scope and non-forwarding as `SIEVE_EVICTORS`. |
| `SIEVE_VENV` | venv to activate. Default `$PROJECT_ROOT/.venv`. |
| `SIEVE_NO_REPORT=1` | do not chain the report job. |
| `H0_CORPUS` | directory of real haystack text. Staged by `prefetch_corpus.py`; the slurm scripts default it to `$PROJECT_ROOT/.h0_corpus/pg19`. Unset ⇒ **tier main/large refuses to start** (see § synthetic-haystack confound). |
| `H0_ALLOW_SYNTHETIC=1` | run main/large on filler anyway. Same as `run_h0.py --allow-synthetic`. `report.py` still stamps the verdict UNKNOWN. |

### Registry knobs (`models.yaml`, or `--override k=v`)

Not environment variables — config keys. `defaults:` applies to every model; a
model entry overrides it; `--override` overrides both.

| key | default | meaning |
|---|---|---|
| `evictors` | `[oracle, last_step, accum, window, recency]` | WHO the eviction corner is. `oracle` is the bound and stays configurable; the verdict keys off the practical ones. `name:k=v,k=v` for options, `@alias` to disambiguate. `none` ⇒ oracle only. |
| `corner_policies` | `[frac, abs]` | HOW MUCH it may keep. `frac` = `B·L/maxb`; `abs` = capped at `max(corner_kappa·n95, corner_floor)`. |
| `corner_kappa` | `4.0` | `abs`: keep this many times the head's `n95` support. |
| `corner_floor` | `256` | `abs`: never fewer than this many tokens. |
| `kstar` | `true` | the slack diagnostic: smallest keep-count within `kstar_tol` of the full-budget corner. Nearly free — rides the same cumulative pass. |
| `kstar_points` | `12` | resolution of the geometric K ladder. |
| `kstar_tol` | `0.10` | "within 10% of the full-budget corner". |
| `quant_every` | `4` | the expensive bit sweep every N decode steps. Practical columns only exist on those steps — see the caveats above. |
| `budgets`, `bit_list`, `maxb` | `[1,2,3,4]`, `[1,2,3,4,5,6,8]`, `8` | every budget must be in `bit_list` or the run refuses to start. |

---

## Version notes

Kept deliberately short since there is no VCS here.

- **Matched ctx + ctx sweep (bug 3).** The two Recommendation bullets below are
  now runnable: `SIEVE_CTX` in both campaign scripts pins every model to one
  context length (`--override ctx=`, validated against each model's registry
  cap), and `submit_h0_ctx_sweep.slurm` arrays one model over a ctx list.
  `report.py` gains `page_ctx_slope` — band fraction, the quantize/evict
  crossover, and `eff_frac = n95/L` each plotted against ctx, drawn whenever a
  model appears at ≥ 2 ctx values and skipped otherwise, so single-ctx reports
  are unchanged. The sink-excluded φ piece of bug 3 is deliberately NOT in this
  round; see `bugs/3_context_sweep_and_reports/plan.md`. Results are in `h0_measurement/reports/h0_report_ctx_pooled.pdf`. 

- **Validity gate replaced (bug 1, round two).** The niah-vs-cont ladder gate was
  unpassable by construction — the ladder is a bulk second moment and a needle is
  one token in 131,072, so a perfect retrieval moves it ~3x less than the 0.1 b
  threshold demanded, and the median over heads cannot move at all. It stamped
  UNKNOWN on all five PG-19 models. New `sievelib/validity.py` checks retrieval
  where it actually lives: task-level (does the model emit the code — enforced)
  and head-level (attention mass on the needle span, a max over heads — advisory
  until `MIN_MASS` is calibrated). `run_h0.py --validity-only` +
  `submit_validity.slurm` run that probe for minutes per model instead of hours,
  and validate an existing run because the haystack is seeded on `prompt_idx`.
  `report.py` gains `page_phase`, which retires phi = n95/L (it divides by context
  length) for ladder width x dead-2-bit-tier fraction, both derived from
  tau^2*c_b vs c0 = 1. Run `python -m sievelib.validity` for the arithmetic.

- **Real haystack (bug 1).** `H0_CORPUS` went from an optional knob to a gate.
  New `prefetch_corpus.py` (PG-19 via Project Gutenberg, no new pip dependency);
  `prompts.build` returns a provenance dict instead of a `synthetic` bool, seeks a
  window inside one book instead of concatenating whole files, and shares that
  window across families at each prompt index so the niah-vs-cont check is paired
  per head; `run_h0.py` refuses main/large on filler; `report.py` gains
  `family_gate` and the UNKNOWN verdict. `report.py` also now groups by
  `(model, ctx)`, which was flagged below as a latent mislabelling and becomes
  live the moment the ctx sweep runs.

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
L-dependence above. **FIXED** — see `bugs/1_from_synthetic_to_real_corpus/fix.md`.
The warning is now a gate at three points:

- `prefetch_corpus.py` stages PG-19 books and reports whether the corpus supports
  `n_prompts` distinct full-ctx windows.
- `run_h0.py` refuses to start a `tier: main`/`large` model without a corpus, and
  proves by tokenising at the real ctx that it yields a full haystack — before the
  tokenizer's first GPU byte, not at hour 3. `--allow-synthetic` overrides.
- `report.py` stamps the verdict **UNKNOWN** unless the haystack was real *and*
  niah's ladder beats cont's on a paired per-head test.

All six runs in `results/job847127` and `results/job847130` were measured on filler
and now read UNKNOWN; their band fractions describe the prompt, not the model.

**Recommendation.**
- Match the `main` tier to the largest ctx all four main-tier models support
  natively (currently 32,768) — `report.py`'s cross-model boxplot compares models
  at whatever ctx each ran at, so leaving them mismatched (32,768 vs 131,072) means
  a difference in `gain_best3` can't be attributed to architecture vs. context length.
- Turn L into a deliberate axis on one model (e.g. llama31-8b at {8k, 32k, 128k})
  instead of an incidental difference across models — this also gives an on-hardware
  test of the v5 tau non-monotonicity claim.
- ~~`report.py`'s `page_compare` groups by `model` only~~ — **fixed**: every panel,
  page and terminal line now groups by `(model, ctx)` and labels the ctx, so a
  per-model ctx sweep can no longer pool into one mislabeled page.


# Methodology: Validating with a smaller model

What validate_with is actually for
It's instrument calibration, not a scientific shortcut. Worth being precise, because the naming invites a misreading:

- L1/L3 validate the probe — that sieve_probe is a transparent drop-in (L1: identical LM logits vs sdpa) and that its recomputed attention weights match HF's independent eager path (L3: catches wrong GQA expansion, scaling, mask, softmax axis).

- The measurement itself always runs on the real model. No H0 result is ever extrapolated from the 0.6B.

So the design isn't claiming "0.6B head sensitivity ≈ 8B head sensitivity." It's claiming "my measuring device is wired correctly, and wiring correctness doesn't depend on parameter count." That claim is sound in kind, and the cost argument is real — L1 loads the model twice in float32, L3 twice more. For the 70B that's ~280GB per load; infeasible. The fp32 + tight-tolerance trade (1e-4 vs the 5e-2 bf16 forces) is also a genuine win — 5e-2 on attention weights would hide real indexing bugs.