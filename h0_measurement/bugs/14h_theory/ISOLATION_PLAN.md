# Plan: isolate R14's theory in `14h_theory/` (proposed and EXECUTED 2026-10-10)

Rule (user, 2026-10-10): every R14 theory update goes in `14h_theory/`. `14h_methodology_improve/` (METH below) keeps
the run-time code that GPU jobs import, and the run records. Executed on 2026-10-10: see "Execution record" at the end.

## Constraint: R6

R6 is frozen and about to be submitted. Its workers import `cert_s1h5`, `rquant_s1h5`, `probe_s1h5`, `run_s1h5`,
`s1h5_lib`, `tasks_s1h4/6`, `read_stage1h_r5/r6`, `test_r14_stage1h_r5/r6`. These stay in METH, unchanged, until R6's
reader has run.

**Step 0: DONE (2026-10-10, before R6's submission).** `test_r14_stage1h_r5.test_cert_rand` imported `theory2_s1h5`
(a v2-agreement check), and R6's workers run that test before every job.
- The check now lives in `14h_theory/test_theory4_s1h5.py` (`test_v2_agreement`). The copy of
  `test_theory_s1h5.py` here is stale and cannot import `cert_s1h5`, so it was not used.
- `test_r14_stage1h_r5.py` imports no theory module.

Verified:
- no R6 file imports a theory module (grep);
- the R5 and R6 fast suites pass with every theory module blocked from import;
- the R6 dry run on trig-login01 passes (32 submissions, 32 blocks read).

Every theory file can now move without touching R6.

## What goes where

| METH file | action | why |
|---|---|---|
| `adapters_s1h5.py`, `test_adapters_s1h5.py` | **move** to 14h_theory | architecture views for CPU theory experiments; no run imports them |
| `oracle_s1h5.py`, `adaptive_s1h5.py` | **move** | CPU theory experiments |
| `theory2_s1h5.py` | **replace** 14h_theory's older copy with METH's current one, then delete METH's | two copies now (14h_theory has the pre-port full version); METH's re-exports v2 from `cert_s1h5` (one implementation); `theory3_s1h5` works with either |
| `test_theory_s1h5.py` | **replace** 14h_theory's older copy with METH's newer one, minus `test_split` / `test_certified_read` (now in `test_theory4_s1h5.py`), then delete METH's | two copies now; same-name modules in both folders shadow each other on `sys.path` |
| `certread_s1h5.py` | **delete** | merged into `14h_theory/theory4_s1h5.py` |
| `theory.md` | **delete** (or keep a 3-line pointer) | merged into `14h_theory/theory.md` (parts 1–7); it already carries a "superseded" note |
| `findings/R5_adapters.{md,json}`, `findings/R5_theory_mla.json` | **move** to `14h_theory/findings/` | theory evidence; `test_theory_s1h5` writes `R5_theory_mla.json` next to itself |
| `findings/R5_theory6.md`, `findings/R5_theory6_certread.json` | **move** to `14h_theory/findings/archive/` | superseded by `findings_part7.*` (same runs, plus part 6's bounds) |
| `cert_s1h5.py`, `rquant_s1h5.py`, `probe_s1h5.py`, `test_r14_stage1h_r5.py` | **keep** in METH | the run-time library and its brute-force tests; GPU jobs import them. 14h_theory imports them read-only; a new bound reaches them only through a plan.md amendment before a run, as v2 did |
| `size_confirm_r5.py`, `consolidate_r5.py`, `plan.md` | **keep** | run sizing and analysis, and the run records (its "R5 theory" amendments stay as history, with a pointer) |

In `14h_theory/` itself:
- `findings_4bit.*` and `findings_part7.*` → `findings/`;
- add `CLAUDE.md` (role; the import rule: `sys.path.append(METH)` for `cert`, `rquant` and the test helpers; the file
  table; test commands);
- add `STATUS.md` (open problems: the key part at 4–6 bits, the GPU check of part 7, MLA in Theorem M, the threshold
  read rule).

## Steps (after R6 is submitted, or after step 0)

1. **Copy, don't move yet.** Copy the "move"/"replace" files into 14h_theory. Give the moved scripts a METH path line
   (`sys.path.append(METH)`), because they now live outside METH and still import `cert_s1h5` / `rquant_s1h5` /
   `test_r14_stage1h_r5`.
2. **Re-point imports and outputs.** `oracle`, `adaptive`, `theory4`, `exp4bit` and the tests import `adapters_s1h5`
   from HERE. `test_theory_s1h5` writes to `14h_theory/findings/`. Drop the duplicated tests (step "replace" above).
3. **Verify in 14h_theory.** `test_theory_s1h5.py --fast`, `test_theory3_s1h5.py --fast`, `test_theory4_s1h5.py --fast`,
   `test_adapters_s1h5.py --fast` (then without `--fast` once).
4. **Delete METH's copies** (`rm -f`; `rm` is aliased to `rm -i` here) and its `__pycache__`.
5. **Verify METH.** All fast suites the worker runs (R6, R5, R4, R3b, R3, R2, R1, Stage 1g, R8); the R6 dry run on
   trig-login01; and a grep showing no METH file imports a moved module:
   `grep -nE "import (theory2|adapters|oracle|adaptive|certread)" 14h_methodology_improve/*.py`.
6. **Docs.** METH `CLAUDE.md`: replace the theory rows of the file table with one pointer. METH `STATUS.md` and
   `plan.md`: a pointer line. 14h_theory `CLAUDE.md` / `STATUS.md`.

## Risks

- **Moving a file R6 imports breaks R6's pre-run tests.** Only `theory2_s1h5` is affected (step 0).
- **Same-name modules in both folders.** Today `theory2_s1h5.py` and `test_theory_s1h5.py` exist twice, and which one
  loads depends on the caller's folder. The plan leaves one of each.
- **Paths baked into outputs.** `test_theory_s1h5` writes `findings/R5_theory_mla.json` relative to itself.

## Execution record (2026-10-10)

Backups of every replaced or deleted file are in that session's scratchpad (`isolation_backup/`).

- **Steps 1–2.**
  - Copied `adapters_s1h5`, `test_adapters_s1h5`, `oracle_s1h5`, `adaptive_s1h5`, `theory2_s1h5` (the re-export
    version, replacing this folder's pre-port copy) and `test_theory_s1h5` (the newer copy; `test_split` /
    `test_certified_read` dropped, as `test_theory4_s1h5` covers them).
  - Each got a METH path line. `exp4bit` now writes to `findings/`.
  - Moved `R5_adapters.*` and `R5_theory_mla.json` to `findings/`, `R5_theory6.*` to `findings/archive/`, and this
    folder's `findings_4bit.*` / `findings_part7.*` into `findings/`.
  - Fixed the references in `theory.md` and the findings notes.
- **Step 3.** The four suites here passed with `--fast`, with the moved modules loading from this folder.
- **Step 4.** Deleted METH's copies and their `__pycache__` entries. METH's `theory.md` is now a 3-line pointer.
- **Step 5.**
  - No METH, Stage 1g or R8 file imports a moved module.
  - All nine worker fast suites pass (R6, R5, R4, R3b, R3, R2, R1, Stage 1g, R8).
  - The R6 dry run on trig-login01 passes (32 submissions, 32 blocks). R6's files are unchanged.
  - The four suites here also pass at full size (real models).
- **Step 6.** `CLAUDE.md` and `STATUS.md` added here. METH's `CLAUDE.md`, `STATUS.md` and `plan.md` point here.

**Note on the regenerated findings.** The full runs rewrote `findings/R5_theory_mla.json` and `findings/R5_adapters.*`
with the current code. Both originals predate later edits:
- `R5_theory_mla.json` was written at 00:27, before the coupled MLA form used Bennett. Its `*_new` / `*_head` medians
  are now 2–25% tighter (e.g. Moonlight 4-bit 1,051 → 813). The `*_old` (v1) entries are unchanged.
- `R5_adapters` was written 3 minutes before `adapters_s1h5.py`'s last edit. It shifts by under 1%.

The move itself changed no number: the ported coupled bound equals the original to 0.0 relative gap on 144 cases.

