# R5 adapters — the certificate on each architecture (CPU smokes, real weights)

Written by `test_adapters_s1h5.py` (code: `adapters_s1h5.py`). Each model truncated to its first
global softmax layers, float32 on CPU, one 1,600-token pg19 passage; context = the first 1,552 rows,
the next 32 positions vote (tier-1 attention summed over each row group: 1/8 of the rows), the last 16
are measured. Tier 1 = the store's rotated Lloyd-Max at 4 bits (MLA: the latent, once). Medians over
heads and steps; shares are of head-steps. A smoke, not a result: one passage, early layers only.

| model | layer | G x r | row (k/v) | recon | B_min .01 | vote eps | cert/oracle | hp/oracle | b max | rel tailx | rel FP8 | rel 4/4 | tailx<=FP8 | L5 cover | L5/err | L5<=FP8 | +C/32 | +C/8 | L5b cover | L5b<=FP8 +C/8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama1b | 0 | 8 x 4 | 64/64 | 6e-07 | 0.595 | 0.232 | 1.2 | 1.1 | 4.2 | 0.0215 | 0.0139 | 0.0489 | 0.22 | 1.000 | 26 | 0.00 | 0.00 | 0.00 | 1.000 | 0.00 |
| llama1b | 1 | 8 x 4 | 64/64 | 4e-06 | 0.209 | 0.023 | 1.6 | 1.5 | 6.6 | 0.0101 | 0.0211 | 0.0861 | 0.78 | 1.000 | 79 | 0.00 | 0.00 | 0.20 | 1.000 | 0.07 |
| llama1b | 2 | 8 x 4 | 64/64 | 2e-06 | 0.167 | 0.021 | 2.0 | 1.7 | 6.1 | 0.0092 | 0.0322 | 0.0958 | 0.85 | 1.000 | 79 | 0.00 | 0.02 | 0.17 | 1.000 | 0.07 |
| llama1b | 3 | 8 x 4 | 64/64 | 3e-06 | 0.198 | 0.030 | 2.2 | 1.9 | 5.7 | 0.0077 | 0.0221 | 0.0884 | 0.86 | 1.000 | 72 | 0.00 | 0.01 | 0.14 | 1.000 | 0.05 |
| gptoss | 1 | 8 x 8 | 64/64 | 1e-06 | 0.300 | 0.120 | 1.3 | 1.3 | 11.3 | 0.0556 | 0.0183 | 0.1638 | 0.16 | 1.000 | 981 | 0.00 | 0.00 | 0.00 | 1.000 | 0.00 |
| dsv2lite | 0 | 1 x 16 | 576/512 | 3e-06 | 0.315 | 0.212 | 1.1 | 1.1 | 6.4 | 0.0194 | 0.0113 | 0.0489 | 0.20 | 1.000 | 414 | 0.00 | 0.00 | 0.00 | 1.000 | 0.00 |
| dsv2lite | 1 | 1 x 16 | 576/512 | 1e-06 | 0.576 | 0.321 | 1.1 | 1.0 | 5.5 | 0.0238 | 0.0101 | 0.0460 | 0.05 | 1.000 | 273 | 0.00 | 0.00 | 0.00 | 1.000 | 0.00 |
| moonlight | 0 | 1 x 16 | 576/512 | 8e-07 | 0.607 | 0.294 | 1.0 | 1.0 | 3.7 | 0.0100 | 0.0093 | 0.0270 | 0.37 | 1.000 | 110 | 0.00 | 0.00 | 0.00 | 1.000 | 0.00 |
| moonlight | 1 | 1 x 16 | 576/512 | 6e-07 | 0.527 | 0.163 | 1.0 | 1.0 | 3.2 | 0.0062 | 0.0053 | 0.0162 | 0.32 | 1.000 | 112 | 0.00 | 0.00 | 0.00 | 1.000 | 0.00 |
| gemma3 | 5 | 8 x 2 | 256/256 | 3e-07 | 0.932 | 0.764 | 1.0 | 1.0 | 2.3 | 0.0083 | 0.0019 | 0.0104 | 0.00 | 1.000 | 8 | 0.00 | 0.00 | 0.00 | 1.000 | 0.00 |
| granite | 5 | 4 x 3 | 128/128 | 6e-07 | 0.973 | 0.813 | 1.0 | 1.0 | 0.2 | 0.0179 | 0.0053 | 0.0226 | 0.06 | 1.000 | 4 | 0.01 | 0.04 | 0.05 | 1.000 | 0.01 |
| kimilinear | 3 | 1 x 32 | 576/512 | 1e-06 | 0.949 | 0.838 | 1.0 | 1.0 | 1.3 | 0.0320 | 0.0093 | 0.0361 | 0.00 | 1.000 | 13 | 0.00 | 0.00 | 0.00 | 1.000 | 0.00 |

Columns: G x r = row groups x query heads sharing one row set; row = key/value width per group (MLA: the 576-wide latent, values its first 512); recon = max relative difference between the adapter's output and the model's own; B_min .01 = rows per head for missed mass <= 0.01 (share of context); vote eps = the vote's missed mass; cert/oracle, hp/oracle = certified rows (worst-case, z = 5 bound) over the shared-set oracle's at 0.01; b max = the largest worst-case score bound (nats); rel = output error / |o| (tailx = the vote's rows exact, the rest at 4 bits; 4/4 = dense 4-bit); tailx<=FP8 = share of head-steps with tailx's error within FP8's; L5 cover = share with the error within Lemma 5's bound (should be >= 0.999); L5/err = median bound over error; L5<=FP8 = share certified at FP8's level at the vote's rows, +C/32 and +C/8 with that many more rows read exactly; L5b = Lemma 5 with a score-bias allowance of 0.5 sigma (its worst coverage over the three reads, and its share certified at FP8's level with C/8 more rows).
