# 14i_r7_theory_at_scale — R7a: R14's theory at the paper's scale

Read `STATUS.md` first, then `theory_r7.md` (what R7a measures, the lemmas behind the GPU forms, the predictions), then
`plan.md` (the run: blocks, columns, seed panels, gate, cost).

R7a is rank 1 of `../14h_theory/RESEARCH_PLAN.md` (§1.1, §4): part 8's certificate, the certified adaptive read and
Theorem M, measured per head-step on R6's models and lengths (Llama-3.1-8B at 128K, Qwen3-30B-A3B at 32K and 128K), at
4/6/8-bit keys, with a dither-seed panel. Frozen and submitted 2026-10-12 (`plan.md` §10): R7a's files may not change
while any R7a job has output, and every job re-checks the code pin.

## Isolation (user, 2026-10-11): development and exploration stay in this folder
- **Nothing outside this folder is edited.** Not `../14h_theory/`, not `../14h_methodology_improve/` (R6's files are
  frozen until its final reader has run, and stay R6's record after), not `sievelib/`, `run_r8.py`, `run_h0.py`.
- **Nothing here imports `../14h_theory/`.** The theory this folder needs is copied into its own modules
  (`cert_r7.py`, `tier_r7.py`), with the source named in each docstring. Parity with the originals is checked against
  frozen fixtures (`fixtures/`), never by importing them.
- **`../14h_methodology_improve/` is read-only, and only for the run-time chain.** The driver (`run_r7.py`) wraps
  `run_s1h5` and its chain (Stage 1g, `run_r8`, `sievelib`) without editing them; `r7_lib.py` builds on `s1h5_lib`.
  The bounds and the tier the probe computes come from this folder's copies, so a later edit of `cert_s1h5` or
  `rquant_s1h5` cannot move R7a's numbers (`test_cert_r7.py` reports whether the copies still match).
- **Nothing outside this folder may import a module from here.**
- **Module names end in `_r7` (or `_r7a`) and are unique** across every folder on `sys.path`.
- Exploration outputs (captured views, JSON) go in `findings/` or `.views/` here; temporary files in the scratchpad.

## Files
| file | role |
|---|---|
| `theory_r7.md` | the theory of R7a: which measurement tests which statement; the GPU forms and their lemmas (S: the HKZ matrix from heavy rows; B: the certified read decided by brackets; P: Theorem M's moments as a series); scaling with context on R6's models; the predictions P1–P8 and their derivations; the seed panel's model |
| `plan.md` | the run (DRAFT until the user approves; nothing submitted): panel (a) probe blocks, panel (b) seed panel, pilots and gate, reader, cost, what is frozen, decisions |
| `STATUS.md` | where things stand, and a correction to carry back to `../14h_theory/RESEARCH_PLAN.md` |
| `tier_r7.py` | the randomized tier 1: a copy of `rquant_s1h5` (+ `dither_step`) |
| `cert_r7.py` | the bounds: copies of `cert_s1h5`'s pieces and Theorem R v2; part 8 (`bound_p8`, `p8_terms`/`p8_combine`); the certified read (`schedule`, `certified_read`); Theorem M (`theorem_m`); the GPU forms |
| `probe_r7.py` | panel (a)'s measurement hook (R6's probe interface; R7a's columns) |
| `r7_lib.py` | presets (`r7*`), the seed-suffixed design arms, `R7_LAYOUT` (11 fields; split 2 = a 4-GPU node job of two 2-GPU blocks), `R7_PILOT`, `r7_units` |
| `run_r7.py` | the driver: `run_s1h5.main` with `r7_lib`, `probe_r7` and the seed arms swapped in at run time |
| `read_r7a.py` | the reader (panels a and b, predictions P1–P8, validity) and the gate |
| `submit_r7.slurm`, `script_r7a.sh` | the worker (re-checks the code pin, runs `test_cert_r7 --quick` and `test_run_r7 --fast` first; `R7_SPLIT`/`R7_GPUS` node split) and the chain (`--run-r7a[-dry]`, `--pin-code`, `--list-r7a`) |
| `test_cert_r7.py` | the bounds: copies (bit for bit), GPU forms, quadrature, brute force, the certified read's stop, Theorem M by Monte Carlo, fixture parity (`--fast`, `--quick`) |
| `test_run_r7.py` | the run side: presets, layout, fresh prompts, the probe against direct computations; without `--fast`, driver smokes and the reader (`R7_SMOKES=r7smoke` runs one) |
| `explore_r7.py` | CPU exploration on R6's own models (`capture`, `analyze`, `summary`, `predict` from R6's GPU probe, read-only) |
| `fixtures/` | frozen reference outputs of `../14h_theory`'s part 8 / Theorem M, with provenance (`fixtures/README.md`) |
| `findings/` | `explore_r7.md` (+ `explore/`, `explore_r7_table.md`, `explore_r7.json`), `predictions_from_r6.json`, `R7a_code.sha256`; after the run `R7a_reader.*`, `R7a_gate.json` |
| `.views/` | captured q/K/V views (exploration only) |
| `r7b/` | **R7b** (the single store, `../14h_theory/RESEARCH_PLAN.md` §4, §8): its theory, port, tests and CPU exploration, in a subfolder with its own `CLAUDE.md`. Not part of R7a: `run_r7.code_manifest` hashes this folder's top-level `*.py`, so R7b adds no file here |

## Commands
```bash
cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14i_r7_theory_at_scale/test_cert_r7.py --fast
OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14i_r7_theory_at_scale/test_run_r7.py --fast
bash h0_measurement/bugs/14i_r7_theory_at_scale/script_r7a.sh --run-r7a-dry      # prints the chain; submits nothing
```
Local runs use `OMP_NUM_THREADS=8` and `python -u` (login nodes cap CPU time at 3,600 s). Don't modify the shared
`.venv`. `cp`/`rm` are aliased to `-i`: use `command cp -f` / `/bin/rm -f` in scripts.

GPU jobs are submitted only when the user asks, from trig-login01 (`ssh -o BatchMode=yes trig-login01`); sbatch drops
the environment, so settings are `R7_NAME=value` arguments. Forward passes are not reproducible across processes:
compared arms run in one process.
