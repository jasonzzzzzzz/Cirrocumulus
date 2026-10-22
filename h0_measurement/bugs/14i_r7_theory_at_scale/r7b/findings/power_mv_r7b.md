Qwen 32K multivalue: R6's units {'std': 20, 'hard': 20} (sensitive, design KL > 0.01: prompts [9638]); 400 simulated runs per cell; P(pass) (P(FAIL and testable)) by R6's rule.

| units | fp_noise passes | rho = 1.0 | rho = 1.14 | rho = 1.3 | rho = 1.5 | rho = 2.0 |
|---|---|---|---|---|---|---|
| R6 (20 std + 20 hard) | 0.99 | 0.60 (0.38) | 0.51 (0.48) | 0.41 (0.58) | 0.36 (0.62) | 0.34 (0.65) |
| R7b (130 std + 20 hard), stratified | 1.00 | 0.99 (0.01) | 0.98 (0.02) | 0.95 (0.05) | 0.94 (0.06) | 0.72 (0.28) |
| R7b, pooled (150 from all of R6's) | 1.00 | 0.96 (0.04) | 0.93 (0.07) | 0.84 (0.16) | 0.69 (0.31) | 0.38 (0.62) |
| 280 std + 20 hard, stratified | 1.00 | 1.00 (0.00) | 1.00 (0.00) | 1.00 (0.00) | 1.00 (0.00) | 0.98 (0.01) |
| 300, pooled | 1.00 | 1.00 (0.00) | 1.00 (0.00) | 0.98 (0.01) | 0.89 (0.11) | 0.49 (0.51) |
