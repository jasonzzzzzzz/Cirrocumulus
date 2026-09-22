# R9 baseline audit and R8 comparison contract

Status: code and CPU checks completed; no main-model R8 comparison has run. This audit supplements `plan.md`. It distinguishes a tested implementation of a paper's score from a reproduction of that paper's reported accuracy.

## Method fidelity

| Arm | Verified core | Explicit deviation or unresolved choice |
|---|---|---|
| Ada-KV | SnapKV score; flattened layer top-K counts; alpha=0.2 safeguard. The official implementation uses alpha as the uniform share. | Largest-remainder rounding enforces the exact shared budget; the official code rounds per head and can drift. Our common protected window and 8-bit kept keys are R8 conventions. |
| DropKV | The score equals the squared output change from deleting one token on a toy, including the observation window. Defaults: 8 queries, max-pool 11. | Its paper does not specify GQA reduction. We use the authors' kvpress PR's group mean. `dropkv:pool=avg:obs=32:pool_k=7` is the PR setting, not the paper setting. |
| OBCache | Value/key/joint scores match the corresponding autograd Hessian terms. `gqa=sum` implements the paper appendix. Defaults: 16 queries, max-pool 7. | The official repository defaults to `pre_redc`, which computes a different score. `gqa=pre` exposes that variant. `obcache_k:alloc=ada` reproduces its AdaKV + OBCache-K combination, subject to R8's common cache conventions. |
| LaProx | Per-query-head score matches explicit projected values; model-wide layer-normalized top-K matches brute force. Defaults: 32 queries, avg-pool 7. | No official code is available in this project's evidence. The paper says to average attention within a GQA group but does not fully specify how to combine it with the query-head-specific output projection. Our mean of complete per-head scores is a documented choice, not a verified reproduction. |

Primary sources: [Ada-KV paper](https://arxiv.org/abs/2407.11550), [Ada-KV code](https://github.com/FFY0/AdaKV), [DropKV paper](https://openreview.net/pdf/23d67de8f260d8b808a83b2de462980c88175072.pdf), [OBCache paper](https://arxiv.org/abs/2510.07651), [OBCache code](https://github.com/DreamSoul-AI/OBCache), [LaProx paper](https://arxiv.org/abs/2605.07234). The checked copies are `docs/{ada-kv,dropkv,obcache,laprox}.pdf`.

## One comparison contract

`run_r8.py` resolves all arm specs before loading a model. Baseline objects in `sievelib/baselines.py` own the score and their layer/model allocation; the driver handles only model-wide prepass versus layer-by-layer execution. Unknown or repeated arms, invalid options, and observation windows larger than `--window` fail early. After allocation, every baseline must keep exactly `n_layers * Hkv * floor(B*C/maxb)` context tokens (with the `keep_count` minimum of one), and every R8 arm must spend at most B context-key bits per token. The runtime bits audit enforces that again after decode.

`--window 32` now captures **32 prefill queries and protects 32 tokens**. Earlier P0 job 21529825 captured 31 because the final prompt token was reserved for first decode; its results remain historical and must not be pooled with new runs as if this setting were identical. The water-filling bisection now returns its feasible side, avoiding the formerly permitted one-tier overspend. This also changes a small number of allocations if H0 is rerun; the existing wave CSVs are historical and are not rewritten.

At a matched budget, compare each candidate with **every fixed arm actually run**. `read_r8.py --p2` now uses that set for its strongest-fixed comparator, including custom-labeled paper variants and fixed SIEVE interiors. Router arms are excluded from the fixed set. Include `fp` as the validity ceiling. Use one prompt block, task set, question-awareness mode, context, model, protected-window size, `maxb`, and quantizer configuration for every arm in a comparison. The sidecar records the effective baseline specs, actual observation-query counts, and budget-rule version. The reader rejects mixed window, quantizer, seed, or allocation-rule versions within a cell and mismatched corpus identities for paired prompts. P-4 computes correlations separately for each arm and resamples whole model/context blocks for its confidence interval, so several policies or tasks on one model cannot masquerade as independent configurations. The parity test confirms that the configurable `snapkv` preset gives the same bits as `evict`.

For the default-window comparison, an arm list is:

```
fp,uniform,evict,interior_cascade,router_calib,adakv,dropkv,obcache_k,obcache_k:alloc=ada@obck_ada,laprox
```

`router_calib` requires a disjoint calibration run and `--routes` pointing to its routes file; see the R8 plan's P2 sequence. For a one-command plumbing smoke, replace it with `router_oracle`, which is an upper bound and must not be reported as a deployable method.

To isolate DropKV's *score* from its observation/pooling settings, also run `snapkv:obs=8:pool_k=11@snap8`. To compare OBCache-K scoring at its reported AdaKV setting, include `obck_ada` above rather than only plain `obcache_k`. For LaProx, report the GQA choice and an `alloc=layer` ablation if the model-wide budget is important to the conclusion.

## Limits of the evidence

The four baselines pass score, allocation, budget, GQA, and tiny-model end-to-end CPU checks. They have **not** been run on the main R8 cells, so relative end-task accuracy is unknown. R8 performs one-shot post-prefill compression and simulates quantized keys in a full-precision cache; it establishes generation accuracy, not packed-cache memory use or throughput. The matched B counts context **key** bits. Values remain exact in this study, and the protected window is excluded from the variable-rate budget. These conventions must accompany any paper table; they are adaptations of the published eviction systems, which normally keep retained KV pairs at full precision.
