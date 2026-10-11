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
| `theory.md` | parts 1–8: Lemmas 1–5, Theorem D, Theorem R (v1, v2, v3, the sure sums, the mirror split, the tilted tail and HKZ key sum), the certified adaptive read, Theorem M, Lemma C and Theorem DD, the 4-bit study, evidence, open problems |
| `theory2_s1h5.py` | part 5's instrumented Theorem R (`tail_bound_rand_diag`; v2's development form). Its bounds re-export `cert_s1h5`'s |
| `theory3_s1h5.py` | part 6: Theorem R v3 (by noise source, one-row maxima, χ² values; `sure=True` adds part 7's sure sums), Theorem M (the exact second moment) |
| `theory4_s1h5.py` | part 7: `tail_bound_rand_sure`, `tail_bound_rand_split` (GQA), `schedule` and `certified_read`; CPU harnesses `compare` and `certread` |
| `theory5_s1h5.py` | part 8: the split with the tilted tail and a Hsu–Kakade–Zhang key sum (`tail_bound_rand_split_hkz`); the key proxies (`key_proxy`: log-Sobolev, closed form; `exact_key_proxy`: exact law); per-row truncation (`prep`); a generic `certified_read`; CPU harnesses `compare` and `certread` on cached views (`TH_CACHE`) |
| `adapters_s1h5.py` | the certificate's view of each architecture's global softmax layers (GQA, GPT-OSS sinks, Gemma 3 global, Granite-4.0-H attention, MLA for DeepSeek-V2/V3, Moonlight, Kimi-Linear); loads the six test models truncated, on CPU |
| `oracle_s1h5.py`, `adaptive_s1h5.py` | the oracle width (the best any certificate could do); Theorem R under adaptive (design-substituted) queries |
| `exp4bit_s1h5.py` | part 6's 4-bit study (`capture`, `diagnose`, `remedies`, `lawcheck`, `evaluate`; views cached in `.views/`) |
| `test_theory_s1h5.py`, `test_theory3_s1h5.py`, `test_theory4_s1h5.py`, `test_theory5_s1h5.py`, `test_adapters_s1h5.py` | brute-force and real-weight checks for parts 5, 6, 7, 8 and the adapters (part 8's also check the proxies against exact laws by quadrature) |
| `findings/` | `findings_4bit`, `findings_part7`, `findings_part8`, `R5_adapters`, `R5_theory_mla`; `archive/` holds superseded write-ups |
| `ISOLATION_PLAN.md` | how this folder was separated from `../14h_methodology_improve/` (executed 2026-10-10) |
| `CLAIMS_LEDGER.md` | the design paper's claims (empirical C1–C7, theory T1–T7, systems S1–S3): evidence, status, theory-against-protocol checks, which paper owns what |
| `paper_theory_section.md` | the design paper's theory section as a skeleton: statements in paper form, proof routes, evidence slots |

## Commands
```bash
cd /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant
for t in test_theory_s1h5 test_theory3_s1h5 test_theory4_s1h5 test_theory5_s1h5 test_adapters_s1h5; do
  OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14h_theory/$t.py --fast; done
```
Without `--fast`, the tests load real models on CPU (Llama-3.2-1B, and the six adapter models): minutes to tens of
minutes. Local runs use `OMP_NUM_THREADS=8` and `python -u` (login nodes have a CPU-time limit). Don't modify the
shared `.venv`.
