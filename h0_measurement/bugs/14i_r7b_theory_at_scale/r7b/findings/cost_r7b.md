R7b's GPU-hours from R6's blocks (results/r14s1h_h6*): t_prefill + t_precompute per unit, t_arm + t_tf per arm; x 1.1 slack, + setup {'llama128': 0.07, 'qwen32': 0.15, 'qwen128': 0.2} h per block (R6's measured);
a node job (split 2) counts 4 GPUs. E1 at 0.075 s and 0.02 s per layer per answer token.

| stage | GPU-h | blocks over 90% of their wall |
|---|---|---|
| core | 53.6 | — |
| R7B_EXTRA | 8.5 | — |
| calibration stage 1 | 4.8 | — |
| calibration stage 2, E1 0.075 s | 6.5 | — |
| calibration stage 2, E1 0.02 s | 2.9 | — |
| exploratory, E1 0.075 s | 12.7 | — |
| exploratory, E1 0.02 s | 7.4 | — |

The submission plan (llama128=trillium,qwen32=trillium,qwen128=rorqual; jobs_r7b.DEFAULT_PACK): jobs and GPU-hours billed, at E1's 0.075 / 0.02 s per layer per answer token.

| cluster | pilot | core | extra | calib1 | calib2 | explore | total |
|---|---|---|---|---|---|---|---|
| rorqual | 1 jobs, 0.5 | 8 jobs, 16.0 | 6 jobs, 8.5 | 2 jobs, 3.0 | 2 jobs, 1.6 / 1.2 | 2 jobs, 2.9 / 2.4 | 32.5 / 31.6 |
| trillium | 2 jobs, 1.4 | 11 jobs, 42.4 | 0 jobs, 0.0 | 3 jobs, 1.9 | 3 jobs, 4.9 / 1.7 | 5 jobs, 9.9 / 5.0 | 60.5 / 52.4 |

| group | seconds per unit of the core plan, by task |
|---|---|
| llama128 | cwe 298, fwe 149, icl_banking77 70, icl_trec_coarse 68, kilt_hotpotqa 99, kilt_nq 107, mk_panel 86, msmarco_rerank_psg 331, niah_multikey 89, niah_multivalue 247, niah_single 89, vt 261 |
| qwen32 | cwe 67, fwe 47, mk_panel 46, niah_multikey 44, niah_multivalue 177, niah_single 43, vt 75 |
| qwen128 | icl_banking77 91, icl_trec_coarse 81, kilt_hotpotqa 123, kilt_nq 121, msmarco_rerank_psg 280 |
