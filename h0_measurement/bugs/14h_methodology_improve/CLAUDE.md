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
| `*_r4`, `s1h4_lib.py`, `run_s1h4.py` | R4b (LongBench v2 and HELMET at 128K, Quest, the floor system, closed book, pilots and gates) |
| `stops_s1h.py` | the `r8list` stop rule (R3a2, R3b, R4b) |
| `tasks_s1h4.py`, `make_r4_manifest.py`, `data/r4/` | R4's tasks, the item manifest and its builder, HELMET's ICL data (the large data under `.h0_corpus/longbench_v2/` and `.h0_corpus/helmet/`) |
| `cert_s1h5.py` | R5's theory as code: missed mass, the certificate, certified / page-certified selection, Lemma 1 and 3 bounds (pure torch) |
| `probe_s1h5.py`, `s1h5_lib.py`, `run_s1h5.py` | R5.1/R5.2: the measurement and injected-error hooks, presets and arms (probe, inj_top/inj_rnd, the tail arm), the driver on any suite (`--suite r3 \| r3b \| r4`) |
| `test_r14_stage1h_r5.py`, `consolidate_r5.py` | R5's tests (the lemmas by brute force; driver smokes) and R5.0's loss split (`findings/R5_0_split.md`) |
| `submit_s1h5.slurm`, `script_stage1h_r5.sh`, `read_stage1h_r5.py` | R5's worker (`S1H_SUITE`, `S1H_TASK_CFG`), its chains (`--pilot-r5`, `--run-r5`, `--run-r53`) and its readers (`--pilot`, `--r5`, `--r53`); R5.3's tail scan is family `tail5` in `s1h5_lib.py` / `run_s1h5.py` |
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
