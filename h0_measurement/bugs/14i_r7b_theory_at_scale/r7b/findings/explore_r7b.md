# R7b's CPU exploration: the single store on R6's models (2026-10-12)

`../explore_r7b.py`, on R7a's views (`../../.views`: Llama-3.1-8B L1 and L3, Qwen3-30B-A3B L1 and L12, at 1.6K–16K
tokens; Llama-3.2-1B L1 at 1.6K), 3 draws per view; and R6's per-unit results (read-only) for the power analysis.
`../theory_r7b.md` §5–§7 has the read in context; this file lists what was run and the headline numbers.

| output | what |
|---|---|
| `explore_r7b_tables.md` | the tables: votes and the question pass; Theorem N (single store against uniform 6/4 and R6's design); the certified read; the gates per head-step; E1's stopping bias; memory |
| `explore_r7b.json` | the same, with every view's full record (`explore/<view>_r7b.json`) |
| `explore/bias_<view>_p<k>.pt` | E1's stopping-bias draws (part 0: the reference stop J*; parts 1–4: evaluated) |
| `power_r7b.md`, `power_r7b.json` | R7b's primary test, gates and accuracy sizes under R6's rules, from R6's 605 units |
| `depth_r7b.md` (after CPU job 2542204) | every layer of both models at 32K, totals over layers |

Commands (one process per step on a login node; the full set runs in `../submit_r7b_cpu.slurm`):
```bash
cd h0_measurement/bugs/14i_r7_theory_at_scale/r7b
OMP_NUM_THREADS=8 ../../../../.venv/bin/python -u explore_r7b.py analyze --view ../.views/r7view_llama8b_L3_16384.pt --draws 3
OMP_NUM_THREADS=4 ../../../../.venv/bin/python -u explore_r7b.py bias --view ../.views/r7view_llama8b_L3_16384.pt --draws 20 --part 0   # .. 4
../../../../.venv/bin/python -u explore_r7b.py summary
../../../../.venv/bin/python -u explore_r7b.py power
```

## The read
- **The vote is unchanged.** The base's vote reads the same attention mass as R6's store vote within 0.0012 on all 15
  views. The question pass's dithered values perturb it by 0.94–1.28× Lloyd-Max's RMS error.
- **The single store costs about 0.2 bit of key precision, flat in context.**
  - Its MSE is 1.10–1.15× uniform 6/4's on Llama and 1.24–1.31× on Qwen (keys are 70–80% of Qwen's MSE).
  - Its key term alone is 1.3–1.4×.
  - With each design's own vote it is 1.08–1.48× R6's design.
- **Theorem N holds at scale.** Measured / predicted MSE is 0.97–1.04. Part 8 has 0 misses on all 15 views, and is
  1.1–1.4× looser than on the uniform tier.
- **Two tiers beat one by a margin that grows with context** (Corollary G).
  - Dense dithered 7/6 has 1.8 → 28× (Llama L1) and 3.3 → 46× (Llama L3) the design's summed MSE from 1.6K to 16K, and
    4.2 → 38× on Qwen L12.
  - The exception is Qwen's flat L1, where it has 0.3–0.7×.
  - Quest at 0.398 loses by 124–1,248× wherever it selects.
- **E1's stopping bias is small.** The stop leaves its reference set in 3–6% of draws. β is 0 within noise at 1.6K and
  (1–3)·10⁻³ (about 2 s.e.) at 16K.
- **Power under R6's rules.**
  - Llama 128K and Qwen 128K pass with P ≈ 1.00.
  - Qwen 32K passes with P ≈ 0.58, entirely because of multivalue's perturbation-sensitive units. Size that family at
    150–200 units.
  - The accuracy secondary needs about 250 Qwen 128K units.

## Limits
Two layers per model, at most 16K tokens; head-level MSE, not KL. Totals over layers come from CPU job 2542204
(`depth_r7b.md`, 32K). The bias estimates carry the noise of 80 draws. See `../theory_r7b.md` §9.
