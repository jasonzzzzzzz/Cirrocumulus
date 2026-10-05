# 14h_methodology_improve — R14 Stage 1h (accuracy first)

Read `STATUS.md` first, then `plan.md`. `glossary.md` decodes arm names and labels.
Don't read `../14_kernel_tpot/report.md` (1,800 lines) unless a task needs Stage 1g's
history; its Parts K–L are the summary.

## What this folder is
- Stage 1h's methodology work (M0) and runs R1–R5, building on Stage 1g's code.
- New files only. Stage 1g's modules in `../14_kernel_tpot/` (`run_s1g`, `s1g_lib`,
  `run_s1f`, `run_s1e`, …) are imported and never edited. If a Stage 1g function needs
  a change, override it at run time from `run_s1h.py`.
- New logic that doesn't need the model (metrics, statistics, tasks) goes in standalone
  modules (`metrics_s1h.py`, `tasks_s1h.py`) that don't import the runner chain.

## Files
| file | role |
|---|---|
| `plan.md` | the question, the run sequence, frozen rules, amendments |
| `s1h_lib.py` | presets (`h1cal`, `h1pilot`, `h1smoke`), arm grammar, plan builder, byte and side-bit rules |
| `metrics_s1h.py` | KL, McNemar, equivalence tests, Holm, the sequential rule, failure types (pure functions) |
| `run_s1h.py` | the driver: `run_s1e.main` with Stage 1g's read paths plus Stage 1h's arms and columns |
| `test_r14_stage1h.py` | `--fast` (CPU, no model); the default adds Llama-3.2-1B on CPU and a driver smoke |
| `read_stage1h.py` | the frozen reader (written before R1's output exists) |
| `submit_s1h.slurm`, `script_stage1h.sh` | the worker and the submission chains |
| `*_r2`, `s1h2_lib.py`, `run_s1h2.py` | R2 (Qwen 32K); the same roles |
| `*_r3`, `s1h3_lib.py`, `run_s1h3.py` | R3a (harder RULER tasks, `mk_panel`) |
| `*_r3b`, `s1h3b_lib.py`, `run_s1h3b.py` | R3b (`cwe`, `fwe`, `nolima`) |
| `tasks_s1h.py`, `data/r3b/` | R3b's tasks (no model) and their pinned data (NoLiMa: Adobe Research License) |
| `findings/R<n>.md` | each run's tables and read |

## Rules
- Results go to `h0_measurement/results/r14s1h_<tag>_<job>/`, logs to `h0_measurement/logs/`.
- GPU jobs are submitted from trig-login01 (`ssh -o BatchMode=yes trig-login01`). sbatch
  drops the environment, so pass settings as `S1H_NAME=value` arguments. tri-login's
  `squeue` can't see GPU jobs.
- Local tests run with `OMP_NUM_THREADS=8` and `python -u`; login nodes have a CPU-time
  limit.
- Forward passes are not reproducible across processes. Compared arms must run in one
  process.
- Don't modify the shared `.venv` (it has no pytest; the tests are scripts).
- A run's rules may change only while none of its jobs has output (`plan.md` §4).

## Commands
```bash
cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h.py --fast
OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h.py
```
