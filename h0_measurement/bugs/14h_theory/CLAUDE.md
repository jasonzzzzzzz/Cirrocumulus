# 14h_theory — R14's theory of certified two-tier KV reads

Read `STATUS.md` first, then `theory.md` (the canonical statement of every result, with proofs and evidence).

## Rule
**Every R14 theory update goes in this folder** (user, 2026-10-10). That covers new bounds, proofs, `theory.md`, CPU
theory experiments, and their tests and findings.

The run-time library stays in `../14h_methodology_improve/` because GPU runs import it and freeze it per run:
- `cert_s1h5.py` (every bound through Theorem R v2, Bennett, the coupled MLA form);
- `rquant_s1h5.py` (the randomized tier);
- `probe_s1h5.py`, and its brute-force tests (`test_r14_stage1h_r5.py`).

How the two folders connect:
- **This folder imports them read-only.** Each file does `sys.path.append(METH)`, with
  METH = `../14h_methodology_improve`.
- **Never edit them from here.** A new bound reaches `cert_s1h5` only through a `../14h_methodology_improve/plan.md`
  amendment before a run, as Theorem R v2 did.
- **Nothing in METH may import a module from here.** The GPU workers' tests must run without this folder.
- **Module names must be unique across the two folders.** Both are on `sys.path`, so a name present in both shadows
  depending on the caller.

## Files
| file | role |
|---|---|
| `theory.md` | parts 1–8 and §13–§14: Lemmas 1–5, Theorem D, Theorem R (v1, v2, v3, the sure sums, the mirror split, the tilted tail and HKZ key sum), the certified adaptive read, Theorem M, Lemma C and Theorem DD, the 4-bit study, the nested coin-free store (Lemma N, Theorem N), the KL bridge (Proposition KL, Corollary WF, run-time τ), evidence, open problems |
| `theory2_s1h5.py` | part 5's instrumented Theorem R (`tail_bound_rand_diag`; v2's development form). Its bounds re-export `cert_s1h5`'s |
| `theory3_s1h5.py` | part 6: Theorem R v3 (by noise source, one-row maxima, χ² values; `sure=True` adds part 7's sure sums), Theorem M (the exact second moment) |
| `theory4_s1h5.py` | part 7: `tail_bound_rand_sure`, `tail_bound_rand_split` (GQA), `schedule` and `certified_read`; CPU harnesses `compare` and `certread` |
| `theory5_s1h5.py` | part 8: the split with the tilted tail and a Hsu–Kakade–Zhang key sum (`tail_bound_rand_split_hkz`); the key proxies (`key_proxy`: log-Sobolev, closed form; `exact_key_proxy`: exact law); per-row truncation (`prep`); a generic `certified_read`; CPU harnesses `compare` and `certread` on cached views (`TH_CACHE`) |
| `adapters_s1h5.py` | (2026-10-12: Gemma 3 now loads through the multimodal class and any missing weight is refused; before, Gemma 3 was randomly initialized) the certificate's view of each architecture's global softmax layers (GQA, GPT-OSS sinks, Gemma 3 global, Granite-4.0-H attention, MLA for DeepSeek-V2/V3, Moonlight, Kimi-Linear); loads the test models on CPU, truncated or to any depth (`_load(key, k)`): the six architectures plus Qwen3-1.7B, R6's two models (`llama8b`, `qwen3_30b`) and 11 more (Llama-3.3-70B, Mistral 7B and Small 3.2, Granite-4.2, Qwen1.5 and Qwen1.5-MoE, GLM-4.5-Air, MiniMax-M2, Qwen3.5/3.8, Gemma 4); multimodal checkpoints load their text model, block-FP8 dequantizes, a Mistral tekken BPE in pure Python |
| `layermap_s1h5.py` | Theorem M's per-layer map of a model from one FP pass: σ, the tail's predicted error against FP8's at 4/6/8-bit keys, the unread shares (`findings/layermap_<model>.json`); `--grid`: every (key, value) width in {4, 5, 6, 8}² and read fraction in {1/8, 1/4} (Corollary M-B; `findings/findings_bits.*`). `H0_TEXT=<file>` switches the input text |
| `mla_m_s1h5.py` | Theorem M for MLA (coupled key and value errors, exact), with its Monte Carlo check (`findings/findings_mla_m.json`) |
| `nested_s1h5.py` | theory.md §13: the nested coin-free store (a deterministic Lloyd-Max base, which the vote reads, plus a dithered residual); per-coordinate coin moments, sigma, a.s. bound and exact moments; CPU checks `lemma`, `mse`, `cover`, `vote`, and `depth` (part 8 at 6/8-bit keys and the 4+2/4+4 store on any GQA view file, e.g. R7b's 32K captures; `findings/depth32k/`) (`findings/findings_nested.*`) |
| `e2e_s1h5.py` | the end-to-end CPU harness: KL per answer token of any per-layer attention arm against FP (tail, nested, keys-only, values-only, dense, FP8, noise), with Theorem M's per-layer prediction and the certified read's material (`--plan configs | sens | arms | tau | vote`); the certified read as an arm (`cert`: uniform tier, `certn`: the single store) and R6's question-pass vote over a store (`--vote exact | lm | dither`); texts: PG-19 and three LongBench v2 contexts |
| `klbridge_s1h5.py` | theory.md §14: the KL bridge (`bridge`), water-filling allocation (`alloc`), run-time targets for the certified read (`tau`, `taufront`), how to calibrate them (`calib`: level by protocol; `calibfront`: held-out frontiers by shape; `cvar`: the sensitivities' spread), from `e2e_s1h5.py`'s runs (`findings/findings_klbridge.*`, `findings/klbridge_runs/`; that folder's `checks_summary.py` summarizes the 2026-10-12 checks) |
| `oracle_s1h5.py`, `adaptive_s1h5.py` | the oracle width (the best any certificate could do); Theorem R under adaptive (design-substituted) queries |
| `exp4bit_s1h5.py` | part 6's 4-bit study (`capture`, `diagnose`, `remedies`, `lawcheck`, `evaluate`; views cached in `../14_logs/14h_theory/views/`, moved from `.views/` on 2026-10-22; `TH_CACHE` overrides) |
| `test_theory_s1h5.py`, `test_theory3_s1h5.py`, `test_theory4_s1h5.py`, `test_theory5_s1h5.py`, `test_adapters_s1h5.py`, `test_nested_s1h5.py` | brute-force and real-weight checks for parts 5, 6, 7, 8 and the adapters (part 8's also check the proxies against exact laws by quadrature) |
| `findings/` | `findings_4bit`, `findings_part7`, `findings_part8`, `findings_qwen`, `findings_architectures`, `findings_mla_m`, `findings_bits`, `layermap_*`, `R5_adapters`, `R5_theory_mla`; `archive/` holds superseded write-ups |
| `ISOLATION_PLAN.md` | how this folder was separated from `../14h_methodology_improve/` (executed 2026-10-10) |
| `CLAIMS_LEDGER.md` | the design paper's claims (empirical C1–C7, theory T1–T8, systems S1–S3): evidence, status, theory-against-protocol checks, which paper owns what |
| `RESEARCH_PLAN.md` | the post-R6 plan: the colleague's three steps checked, one ranked plan, the theory agenda (N, KL, τ, WF, ESC, SEL, SC, TI, LB, DD+, EXT), R7a/R7b/R8, which folder each part lives in, the user's decisions D1–D7; **§4 R7b and §8 are R7b's spec and build list; §8 also records the build (2026-10-12)** |
| `ARCHITECTURES_PLAN.md` | how R7a and R7b extend to every architecture: what is already architecture-independent, per mechanism what the probe, the single store and the run time need, the per-model inputs (widths, certification widths, sensitivities), stages G0–G5, the claims per stage, risks |
| `MODEL_FAMILY.md` | self-contained: which key and value bits each kind of layer and model needs (families by flatness and key/value split, from Theorem M's grid on 18 models), which attention mechanisms the theorems cover, and the scoped generality claim (T8) |
| `g0map_s1h5.py`, `g0mla_s1h5.py`, `g0sum_s1h5.py` | G0 (`ARCHITECTURES_PLAN.md` §5), the 32K map in the single store's form: `capture` (exact attention swapped into every layer, sinks and windows included, unknown kwargs refused; loaded weights checked against the checkpoint), `analyze` (Theorem M's store and uniform grids, part 8 at 6/8-bit keys; MLA: Lemma N_lat, Theorem M_lat and the coupled certificate, theory.md §13b), `g0sum` (families, widths, certification widths, the slack law: `findings/g0/tables_g0.md`); views in `../14_logs/14h_theory/g0views/` (581 GB; moved from `.g0views/` on 2026-10-22; `G0_VIEWS` overrides; `../14_logs/README.md`) |
| `test_g0map_s1h5.py`, `test_g0mla_s1h5.py`, `submit_g0_cpu.slurm` | G0's checks (`--fast`: synthetic; without it, every architecture's truncated real model) and its Trillium CPU job (`G0_MODELS=`, `G0_STAGES=`; resumes from disk; analyze reserves memory per view) |
| `paper_theory_section.md` | the design paper's theory section as a skeleton: statements in paper form, proof routes, evidence slots |
| `gate76_s1h5.py`, `gate76sum_s1h5.py`, `test_gate76_s1h5.py`, `submit_gate76_cpu.slurm` | why dense 7/6 beats R7b's design on re-ranking, and the answer-time re-vote (`findings/findings_gate76.md`): R7b's HELMET units at 32K on CPU (`RUnit` = e2e_s1h5.Unit with an exact-attention prefill, FP's greedy answer, R6's pooled vote; parts theory / arms / layers / reads), the G0 views as a control, the summary (bridge, heads, read regimes, variants at <= 0.406); outputs `findings/gate76/` |

## Commands
```bash
cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
for t in test_theory_s1h5 test_theory3_s1h5 test_theory4_s1h5 test_theory5_s1h5 test_adapters_s1h5 test_nested_s1h5 test_g0map_s1h5 test_g0mla_s1h5 test_gate76_s1h5; do
  OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14h_theory/$t.py --fast; done
```
Without `--fast`, the tests load real models on CPU (Llama-3.2-1B, and the six adapter models): minutes to tens of
minutes. Local runs use `OMP_NUM_THREADS=8` and `python -u` (login nodes have a CPU-time limit). Don't modify the
shared `.venv`.
