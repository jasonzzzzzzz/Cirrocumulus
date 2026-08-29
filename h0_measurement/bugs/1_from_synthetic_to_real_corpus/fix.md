# Fix: synthetic filler → real long-context corpus

Status: **implemented and verified.** All six existing runs (`job847127`,
`job847130`) now report `UNKNOWN` instead of STOP/NARROW/GO, because they were
measured on filler.

The problem, restated in one line: every metric H0 reports is computed from the
attention distribution, which is a function of the model *and the prompt*, so a
campaign on 8 sentences tiled ~970× measures how a model degenerates on
repetition. `H0_CORPUS` already existed but was an optional knob that failed open.
This change makes real text the default, makes a missing or undersized corpus an
error rather than a silent downgrade, and refuses to issue a verdict on text that
never induced retrieval behaviour.

---

## 1. How you use it

### One-time, on the LOGIN node (compute nodes have no internet)

```bash
python h0_measurement/prefetch_corpus.py --check-ctx 131072 --n-prompts 6
export H0_CORPUS=$PWD/.h0_corpus/pg19
```

~25 MB, about a minute. It fetches 40 pre-1919 Project Gutenberg books (PG-19 *is*
pre-1919 Gutenberg), strips the licence header and footer, and writes a
`MANIFEST.json` with a sha256 per book. It ends by telling you whether the corpus
can actually support the campaign you intend:

```
  ctx 131,072  needs ~602,931 chars/prompt  (6 prompts)
  books >= that size : 36 / 40
  total corpus       : 58,101,444 chars
  OK -- 6 prompts get distinct single-book windows
```

The slurm scripts already default `H0_CORPUS` to `$PROJECT_ROOT/.h0_corpus/pg19`,
so if you stage it there the `export` is optional.

### Then submit exactly as before

```bash
sbatch h0_measurement/submit_h0.slurm
sbatch h0_measurement/submit_h0_large_models.slurm
```

No new flags, and PG-19 is now the default input for both scripts — they set
`H0_CORPUS` to `$PROJECT_ROOT/.h0_corpus/pg19` unless you override it, and each
measure task verifies the corpus (offline sha256 against `MANIFEST.json`) before
it chains a report job or asks for GPUs. A missing or truncated corpus stops the
task there:

```
ERROR: no usable PG-19 haystack at H0_CORPUS=/...
  stage it on the LOGIN node (this node has no internet):
    python h0_measurement/prefetch_corpus.py --check-ctx 131072 --n-prompts 6
  or resubmit with H0_ALLOW_SYNTHETIC=1 to measure filler anyway
```

That check is fatal rather than advisory because every model in either script is
`tier: main` or `tier: large`, so a missing corpus is a *certain* failure — and in
`submit_h0_large_models.slurm` what it would waste is a 4×H100 allocation.
`H0_ALLOW_SYNTHETIC=1` downgrades it to a warning and runs anyway.

What changed inside the job is what happens when the corpus is missing: a
`tier: main` or `tier: large` model now refuses to start, in seconds, before the
tokenizer loads and long before a GPU is held.

```
FATAL: tier 'main' requires a real haystack, but H0_CORPUS is unset or not a
directory (H0_CORPUS=None).
  stage it:  python h0_measurement/prefetch_corpus.py --check-ctx 40960 --n-prompts 6
  then:      export H0_CORPUS=<that dir>
  override (results will be stamped UNKNOWN):  --allow-synthetic
```

`tier: debug` and `--validate-only` stay exempt, so `quick_test.sh` still works on
a fresh checkout with no corpus.

A successful run prints the provenance of the text it read:

```
haystack: 40 books (59,198,464 chars, corpus_sha=0a26bc1e05a1eea8), 36 of them
cover a 131072-token window on their own
[0/niah] prefill 131072 tok  [pg19_01399_anna-karenina.txt@519712] ...
[0/qa]   prefill 131072 tok  [pg19_01399_anna-karenina.txt@519712] ...
[0/cont] prefill 131072 tok  [pg19_01399_anna-karenina.txt@519712] ...
[1/niah] prefill 131072 tok  [pg19_03300_wealth-of-nations.txt@226889] ...
```

Note the same book and offset across all three families at prompt 0, and a
different book at prompt 1. Both are deliberate — see §2.

### How to specify a shorter context

You do not stage a second corpus. Context length is the only knob; the builder
takes a window of `ctx × 0.92` tokens out of a book, so a 32k prompt is a
smaller slice of the same PG-19 text as a 128k prompt.

```bash
# one model, three context lengths, same corpus — the ctx axis the README wants
python h0_measurement/run_h0.py --model llama31-8b --override ctx=32768
python h0_measurement/run_h0.py --model llama31-8b --override ctx=65536
python h0_measurement/run_h0.py --model llama31-8b --override ctx=131072
```

Or permanently, per model, in `models.yaml` (`ctx:` is already a per-entry key).
`report.py` now groups every panel and terminal line by `(model, ctx)`, so those
three runs produce three labelled pages instead of silently pooling into one.

Two things worth knowing when you pick a ctx:

- The corpus must supply `n_prompts` windows at that size. Check before running:
  `prefetch_corpus.py --check-ctx 65536 --n-prompts 6`. Smaller ctx is always
  safer; 128k is the demanding case.
- Below ~4k tokens the window comes from a single page of one novel, which is
  fine for smoke tests but is not a long-context measurement.

### Reading the verdict

`report.py` prints an input-validity block before the numbers, and the verdict is
`UNKNOWN` unless the block passes:

```
llama33-70b  (ctx 131,072,  5,120 heads)
    INPUT VALIDITY FAIL   haystack SYNTHETIC
      ladder niah 1.834 b   cont 1.833 b   paired delta +0.001 b (need >= 0.1)
      heads with niah > cont  50.4%  (need >= 60%, n=5,120)
    HEADS IN BAND   28.4%
    VERDICT: UNKNOWN — input-validity gate failed (...)
```

To deliberately run on filler anyway (e.g. to reproduce the old numbers):
`--allow-synthetic` or `H0_ALLOW_SYNTHETIC=1`. The results are still stamped
UNKNOWN — the override lets you run, not conclude.

---

## 2. Interface

```
 LOGIN NODE (internet)                          COMPUTE NODE (offline)
 ─────────────────────                          ──────────────────────

 prefetch.py ──────────► $HF_HOME/hub ─────────────────┐
   weights + proxy                                     │
                                                       ▼
 prefetch_corpus.py ───► $H0_CORPUS/ ──────►  ┌──────────────────────┐
   gutenberg.org           pg19_*.txt         │      run_h0.py       │
   strip boilerplate       MANIFEST.json      │                      │
   sha256 + capacity            │             │ ① tier gate          │  no corpus
   report                       │             │   (before tokenizer) │──► SystemExit
                                │             │ ② preflight:         │    (seconds,
   --verify (offline) ──────────┘             │   TOKENISE at ctx    │──► no GPU held)
                                              │ ③ per-prompt build   │
                                              └──────────┬───────────┘
                                                         │ prompts.build(...)
                                                         ▼
                                      ┌──────────────────────────────────┐
                                      │        sievelib/prompts.py       │
                                      │                                  │
                                      │  corpus_window(dir, need, key)   │
                                      │    key = prompt_idx              │
                                      │    ├ prefer ONE book ≥ need      │
                                      │    │   seek to random offset     │
                                      │    └ else splice (recorded)      │
                                      │                                  │
                                      │  haystack ids ── shared ─────┐   │
                                      │        ┌──────────┬──────────┤   │
                                      │      niah        qa        cont  │
                                      │    +needle   +question    (bare) │
                                      │    seeded on (prompt_idx, family)│
                                      └──────────────────┬───────────────┘
                                                         │ (text, meta)
                                                         ▼
                                          h0_<tag>_<ctx>.parquet
                                          per-head metrics
                                          + synthetic, corpus_source,
                                            corpus_doc, corpus_offset,
                                            corpus_spliced, corpus_sha,
                                            needle_tok
                                                         │
                                                         ▼
                                      ┌──────────────────────────────────┐
                                      │           report.py              │
                                      │  family_gate(raw)  per (model,ctx)│
                                      │    synthetic_frac == 0           │
                                      │    AND paired median             │
                                      │        ladder(niah)-ladder(cont) │
                                      │        ≥ 0.10 bits               │
                                      │    AND ≥ 60% of heads niah>cont  │
                                      │              │                   │
                                      │       pass ──┴── fail            │
                                      │        │          │              │
                                      │   STOP/NARROW/GO  UNKNOWN        │
                                      └──────────────────────────────────┘
```

### Upward interface (what you call)

| surface | interface |
|---|---|
| `prefetch_corpus.py` | `--out --books --ids --min-chars --check-ctx --n-prompts --chars-per-token --verify --force` |
| `run_h0.py` | unchanged, plus `--allow-synthetic` |
| env | `H0_CORPUS` (now load-bearing), `H0_ALLOW_SYNTHETIC=1` |
| `models.yaml` | unchanged; `tier` now selects gate strictness, `corpus:` is an optional per-model override (also reachable as `--override corpus=/path`) |
| parquet / `_per_head.csv` | 6 new provenance columns (above) |

### Downward interface (what it depends on)

| dependency | note |
|---|---|
| `gutenberg.org` over HTTPS, login node only | stdlib `urllib` — **no new pip package**. `datasets` is not installed in this venv and the pyarrow episode in the README argued against adding one. Two mirror URL patterns are tried per book; 404s are skipped, not fatal. |
| pinned Gutenberg ID list | 70 candidates in `CANDIDATES`, ordered long-first. All 40 requested resolved on the verification run. |
| `MANIFEST.json` | the only thing `--verify` needs, so integrity checking works offline on a compute node. |
| `corpus_index` via `os.path.getsize` | file sizes are bytes, window reads are bytes; for mostly-ASCII prose bytes ≥ chars, so the estimate errs toward *more* text than needed. |
| `raw` DataFrame columns `family`, `synthetic`, `ladder_bits`, `layer`, `head`, `ctx` | `family_gate` runs on `raw`, not `per_head()`, which groups `family` away. |

### One breaking change

`prompts.build` returns `(text, meta: dict)` instead of `(text, synthetic: bool)`.
There was exactly one caller in the repo and no test referenced it. `meta` carries
`synthetic, source, doc, offset, spliced, corpus_sha, family, prompt_idx,
needle_tok, n_haystack_tokens` (+ `needle_frac` for niah).

---

## 3. What changed, and why each piece is load-bearing

### `h0_measurement/prefetch_corpus.py` (new, ~230 lines)

Stages the haystack the way `prefetch.py` stages weights, for the same reason.
Strips the Gutenberg licence header/footer — left in, that ~1 KB header would be
the one passage repeated across all 40 books, a miniature of the artifact this
change exists to remove.

### `sievelib/prompts.py` (rewritten)

**Windows instead of whole files.** The old `_corpus_text` read entire files and
concatenated until it had enough characters, always starting at byte 0. Different
prompts therefore landed on near-identical text. It now seeks to a per-prompt
offset inside a book chosen round-robin from a corpus-level shuffle, so six
prompts get six *different* books — verified, not assumed.

**Single-book windows preferred.** Splicing two unrelated novels puts an
artificial topic boundary in the middle of the context, which is itself an
attention artifact. Splicing happens only when no single book covers the window,
and is recorded as `corpus_spliced` in the parquet.

**Families share a haystack — deliberately.** This was previously true *by
accident*: the seed was `1000*p + len(family)`, and `len("niah") == len("cont")
== 4`, so those two paired while `qa` silently got different text. The niah-vs-cont
gate is a paired per-(layer, head) comparison, which is what lets the thresholds
be strict; independent text per family would leave the gate working but much
noisier. It is now explicit, commented as such in the module docstring, and
covered by a `[REGRESSION]` test — the obvious "cleanup" of giving each family its
own seed would quietly weaken the gate.

**A too-small corpus is an error.** The old code returned filler whenever it
gathered less than 60% of its character estimate. Since the corpus is *sized in
characters* but *consumed in tokens*, a character estimate is not evidence; the
builder now tokenises, grows the window up to 4× if the text is denser than
estimated, and raises under `require_real`.

**Deterministic seeding.** Seeds are derived through sha256 of a string.
`random.Random()` falls back to `hash()` for tuples, and `hash()` of a `str` is
salted per process, which would have made prompts irreproducible across jobs.

### `h0_measurement/run_h0.py`

The tier gate sits after config load and **before** `P.install()`, the tokenizer,
the L1/L3 proxy loads and the model — the old warning printed mid-run, after the
allocation was already held. The tokenising preflight runs right after the
tokenizer health check. Six provenance columns are written per row so a result can
be traced to the exact text the model read.

### `h0_measurement/report.py`

`family_gate` + the UNKNOWN verdict + the validity block, thresholds
`GATE_DELTA_BITS = 0.10` and `GATE_PAIRED_FRAC = 0.60`. Also fixes the
`groupby("model")` mislabelling the README flagged as latent — it becomes live as
soon as the ctx sweep runs.

### `submit_h0.slurm` and `submit_h0_large_models.slurm`

PG-19 is now the input by default. Each script exports
`H0_CORPUS=${H0_CORPUS:-$PROJECT_ROOT/.h0_corpus/pg19}` and
`H0_ALLOW_SYNTHETIC=${H0_ALLOW_SYNTHETIC:-0}` alongside the existing `HF_HOME`
block, and the measure stage runs `prefetch_corpus.py --verify` as a **fatal**
precondition.

Placement matters and is the reason this is not just an `export`. The check sits
after the model-tag validation but **before** the `sbatch --dependency=afterok`
chaining and before `srun`, so a missing corpus cannot leave a report job queued
against an array that can never succeed, and cannot consume the allocation. The
report stage also now explains what an UNKNOWN verdict means, since that is the
line a user will actually see in the log.

`quick_test.sh` exports the same default and verifies the corpus when one exists,
but the debug tier stays exempt, so a fresh checkout with no corpus still smoke
tests end to end.

### Tests

`test_corpus_prompts` and `test_family_gate` in `tests/test_units.py`, 16 checks,
all passing. They use a reversible 4-chars-per-token `FakeTok`, so they need no
model and no network. Four are marked `[REGRESSION]`, matching the file's
convention for bugs that must not silently return.

---

## 4. Verification performed

| check | result |
|---|---|
| 40/40 pinned Gutenberg IDs resolve | ok, 58.1 M chars, 36 books cover a 128k window alone |
| preflight at ctx 131072 on real corpus | `synthetic=False`, `corpus_sha=0a26bc1e05a1eea8` |
| niah/qa/cont share doc+offset at one prompt_idx | ok |
| 6 prompt indices → 6 distinct books, no splicing | ok |
| `require_real` on empty and on undersized corpus | raises in both |
| determinism across processes | ok |
| main tier without corpus | refuses in seconds, no model load |
| main tier with corpus | passes, prints provenance |
| `report.py` on the 6 existing synthetic runs | all six → UNKNOWN |
| new unit tests | 16/16 pass |
| both slurm scripts, `bash -n` | clean |
| slurm measure stage, corpus missing | exits 1 with staging instructions, **before** report chaining and before `srun` (verified with stubbed `nvidia-smi`/`sbatch`/`srun`; no job was queued) |
| slurm measure stage, corpus present | verify passes, proceeds to chaining and measurement |
| slurm measure stage, `H0_ALLOW_SYNTHETIC=1` | warns and proceeds |

Two pre-existing, unrelated problems surfaced while testing on the login node and
were **not** touched: `Qwen/Qwen3-1.7B`'s tokenizer files are missing from
`.hf_cache` (re-run `prefetch.py -m qwen3-1.7b`), and the full `tests/test_units.py`
suite plus L1 validation exceed the login node's CPU/memory limits — L1 loads the
model twice in float32, as the README notes.

---

## 4b. Round two: the gate itself was wrong (see `fix_further.md`)

The corpus fix above was correct. **The gate I attached to it was not.** The PG-19
re-run (`job852849` + `job852851`) stamped UNKNOWN on all five models, and the
cause was the gate, not the data.

**The retired check was unpassable by construction.** It demanded the
median-over-heads ladder width on `niah` exceed `cont` by ≥ 0.10 b. But
`ladder_bits = std(log₂ aᵢ) ≈ τ/ln2` is a **bulk second moment over all L logits**,
and a needle is one token in 131,072. Reproduce with `python -m sievelib.validity`:

| needle tokens | needle logit | Δ ladder | vs the 0.1 b gate |
|---|---|---|---|
| 1 | 10·τ | +0.0016 b | FAIL |
| 20 (the real needle) | 10·τ | +0.0325 b | FAIL |
| 100 | 10·τ | +0.1600 b | pass |

A *perfect* retrieval falls ~3× short of the threshold. Compounding it, the
**median over ~5,120 heads** is dominated by heads that never do retrieval, so it
cannot move however well retrieval works. The observed deltas (−0.010 to +0.061 b)
are exactly what success looks like under this statistic. Setting 0.10 b without
checking what the phenomenon can physically produce was my error.

**The replacement** (`sievelib/validity.py`) asks two questions matched to the
phenomenon, either sufficient:

- **task-level, enforced** — does the model emit the needle code in its greedy
  decode? Behavioural ground truth, needs no calibration.
- **head-level, advisory** — do *any* heads put real attention mass on the needle
  span? A **max over heads, never a median**.

`MIN_MASS = 0.05` / `MIN_HEADS = 4` are **not yet calibrated**, so
`ENFORCE_HEAD_LEVEL = False`. Enforcing a second plausible-looking-but-unmeasured
attention threshold is precisely the mistake that produced the retired gate.
Calibrate from the reported distribution, then flip the switch. `synthetic == False`
remains hard, and `retired_ladder_gate` is kept, printed, and never decisive.

**`--validity-only`, so this costs minutes not hours.** The probe needs attention
and generation only — no bit sweep, no value tensors, niah only. Because the
haystack is seeded on `prompt_idx` alone, it reads byte-identical text to a full
run at the same ctx, so **it validates the measurement you already have**:

```bash
# FROM trig-login01 -- GPU requests are rejected from CPU login nodes
sbatch --array=0-3 h0_measurement/submit_validity.slurm \
       qwen3-8b llama31-8b mistral-7b qwen15-moe-a2.7b
sbatch --array=0-0 --gpus-per-node=4 --time=01:30:00 \
       h0_measurement/submit_validity.slurm qwen3-30b-a3b-2507

python h0_measurement/report.py \
       "h0_measurement/results/job852849/*.parquet" \
       "h0_measurement/results/job852851/*.parquet" \
       "h0_measurement/results/validity<JOBID>/*.parquet" \
       -o h0_measurement/reports/h0_report_gated.pdf
```

`report.py` keeps the two frames separate — `validity_*.parquet` has no gain
columns, so pooling it would drag every per-head median.

**The phase diagram moved off φ.** φ = n₉₅/L divides by context length, and these
models ran at 32k / 40k / 128k, so the same head lands in different places
depending on ctx — disqualifying for a phase claim regardless of fit. (The
"non-monotone" argument in `fix_further.md` is weaker than it looks: it rests on a
0.3-point inversion, which is noise. The ctx confound is the real disqualifier.)
The new `page_phase` uses two axes that fall out of comparing τ²c_b against the
derived c₀ = 1, so the boundaries are **derived rather than fitted**: ladder width
owns the diffuse edge, and the dead-2-bit-tier fraction (`evict_beats_b2`, already
measured) owns the sharp edge — the only axis separating qwen3-8B from qwen3-30B,
0.13 b apart in ladder but 36 points apart here. The region between 39% and 75%
dead tiers is unconstrained by data and is hatched, not drawn as a line.

Verified: 30/30 corpus + validity unit checks pass; needle spans round-trip
exactly through decode→concat→re-tokenise against the real Qwen tokenizer (17
tokens, always containing the code, at both 4k and 32k); the report reaches GO on
a model with retrieval evidence and UNKNOWN with a *meaningful* reason without it.
**Not yet done: the calibration run.** GPU jobs cannot be submitted from
`tri-login*`, so `submit_validity.slurm` is written and syntax-checked but must be
launched from `trig-login01`.

## 5. What this does not fix

Only the input. The other findings in `reports/analysis_from_fable.md` stand: the
practical-evictor off-by-one, the φ-window making ctx a direct input to the
verdict, and the τ/ln2-vs-ladder agreement being algebraic rather than
confirmatory. This change is what makes fixing those worth doing — a de-biased
verdict on filler is still a verdict about filler.

The first real campaign should therefore be treated as new data, not as a
correction of the old table. The two pathologies documented in `why.md` push in
opposite directions and both should relax, so both currently-STOP models may move
up and the "bigger → less benefit" trend could attenuate or invert.
