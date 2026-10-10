# R14 Stage 1h — R4, real tasks at 128K (read_stage1h_r4.py; rules frozen in its docstring)

EXPLORATORY run: labels guide the design; they are not claims. lbv2 accuracy is the forced choice.

- **FP accuracy lb2llama/lbv2**: 0.425 (40 units, 23 FP-failed; closed-book 0.35, 6 need the context)
- **ACC_SYSTEM lb2llama/lbv2**: ACC_INCONCLUSIVE (-0.025 [-0.075, +0.000])
- **ACC_FLOOR lb2llama/lbv2**: ACC_INCONCLUSIVE (+0.000 [-0.050, +0.050])
- **ACC_QUEST lb2llama/lbv2**: ACC_INCONCLUSIVE (-0.050 [-0.100, +0.000])
- **SYS_VS_QUEST_ACC lb2llama/lbv2**: SYS_VS_QUEST_ACC_NO_EFFECT, NOT_EQUIV (+0.025 [+0.000, +0.075])
- **FLOOR_VS_QUEST_ACC lb2llama/lbv2**: FLOOR_VS_QUEST_ACC_NO_EFFECT, NOT_EQUIV (+0.050 [+0.000, +0.100])
- **SCHEDULE_ACC lb2llama/lbv2**: SCHEDULE_ACC_HELPS, NOT_EQUIV (+0.075 [+0.025, +0.150])
- **SYS_VS_BEST_ACC lb2llama/lbv2**: SYS_VS_BEST_ACC_HURTS, NOT_EQUIV (-0.075 [-0.150, -0.025])
- **SYS_VS_D_ACC lb2llama/lbv2**: SYS_VS_D_ACC_NO_EFFECT, NOT_EQUIV (-0.050 [-0.100, +0.000])
- **REQ1_ACC lb2llama/lbv2**: REQ1_ACC_HURTS, NOT_EQUIV (-0.125 [-0.225, -0.050])
- **VOTE_LOSS_ACC lb2llama/lbv2**: VOTE_LOSS_ACC_NO_EFFECT, NOT_EQUIV (-0.025 [-0.075, +0.000])
- **ACC_SYSTEM_CTX lb2llama/lbv2**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **ACC_FLOOR_CTX lb2llama/lbv2**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **ACC_QUEST_CTX lb2llama/lbv2**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **SYS_VS_QUEST_ACC_CTX lb2llama/lbv2**: SYS_VS_QUEST_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **FLOOR_VS_QUEST_ACC_CTX lb2llama/lbv2**: FLOOR_VS_QUEST_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SCHEDULE_ACC_CTX lb2llama/lbv2**: SCHEDULE_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SYS_VS_BEST_ACC_CTX lb2llama/lbv2**: SYS_VS_BEST_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SYS_VS_D_ACC_CTX lb2llama/lbv2**: SYS_VS_D_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **REQ1_ACC_CTX lb2llama/lbv2**: REQ1_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **VOTE_LOSS_ACC_CTX lb2llama/lbv2**: VOTE_LOSS_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SYS_VS_QUEST lb2llama/lbv2**: SYS_VS_QUEST_NO_EFFECT (-0.012 [-0.029, +0.006])
- **SCHEDULE lb2llama/lbv2**: SCHEDULE_NO_EFFECT (-0.009 [-0.021, +0.004])
- **FLOOR_VS_FIXED lb2llama/lbv2**: FLOOR_VS_FIXED_NO_EFFECT (+0.009 [-0.003, +0.022])
- **SYSTEM NLL lb2llama/lbv2**: NEAR_FP at the cell's margin 0.050 (-0.013 [-0.026, +0.001]); vs D_V4 MATCHED; at R1's m_FP 0.050: NEAR_FP
- **FP accuracy lb2qwen/lbv2**: 0.45 (40 units, 22 FP-failed; closed-book 0.225, 11 need the context)
- **ACC_SYSTEM lb2qwen/lbv2**: ACC_INCONCLUSIVE (-0.050 [-0.100, +0.000])
- **ACC_FLOOR lb2qwen/lbv2**: ACC_INCONCLUSIVE (-0.025 [-0.075, +0.000])
- **ACC_QUEST lb2qwen/lbv2**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **SYS_VS_QUEST_ACC lb2qwen/lbv2**: SYS_VS_QUEST_ACC_NO_EFFECT, NOT_EQUIV (-0.050 [-0.100, +0.000])
- **FLOOR_VS_QUEST_ACC lb2qwen/lbv2**: FLOOR_VS_QUEST_ACC_NO_EFFECT, NOT_EQUIV (-0.025 [-0.075, +0.000])
- **SCHEDULE_ACC lb2qwen/lbv2**: SCHEDULE_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **SYS_VS_BEST_ACC lb2qwen/lbv2**: SYS_VS_BEST_ACC_HURTS, NOT_EQUIV (-0.100 [-0.175, -0.025])
- **SYS_VS_D_ACC lb2qwen/lbv2**: SYS_VS_D_ACC_NO_EFFECT, NOT_EQUIV (-0.100 [-0.200, +0.000])
- **REQ1_ACC lb2qwen/lbv2**: REQ1_ACC_NO_EFFECT, NOT_EQUIV (+0.000 [-0.075, +0.075])
- **VOTE_LOSS_ACC lb2qwen/lbv2**: VOTE_LOSS_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **ACC_SYSTEM_CTX lb2qwen/lbv2**: ACC_INCONCLUSIVE (-0.182 [-0.364, +0.000])
- **ACC_FLOOR_CTX lb2qwen/lbv2**: ACC_INCONCLUSIVE (-0.091 [-0.273, +0.000])
- **ACC_QUEST_CTX lb2qwen/lbv2**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **SYS_VS_QUEST_ACC_CTX lb2qwen/lbv2**: SYS_VS_QUEST_ACC_NO_EFFECT (-0.182 [-0.364, +0.000])
- **FLOOR_VS_QUEST_ACC_CTX lb2qwen/lbv2**: FLOOR_VS_QUEST_ACC_NO_EFFECT (-0.091 [-0.273, +0.000])
- **SCHEDULE_ACC_CTX lb2qwen/lbv2**: SCHEDULE_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SYS_VS_BEST_ACC_CTX lb2qwen/lbv2**: SYS_VS_BEST_ACC_NO_EFFECT (-0.182 [-0.364, +0.000])
- **SYS_VS_D_ACC_CTX lb2qwen/lbv2**: SYS_VS_D_ACC_NO_EFFECT (-0.182 [-0.364, +0.000])
- **REQ1_ACC_CTX lb2qwen/lbv2**: REQ1_ACC_NO_EFFECT (-0.091 [-0.273, +0.000])
- **VOTE_LOSS_ACC_CTX lb2qwen/lbv2**: VOTE_LOSS_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SYS_VS_QUEST lb2qwen/lbv2**: SYS_VS_QUEST_NO_EFFECT (+0.040 [+0.008, +0.079])
- **SCHEDULE lb2qwen/lbv2**: SCHEDULE_NO_EFFECT (-0.011 [-0.025, +0.001])
- **FLOOR_VS_FIXED lb2qwen/lbv2**: FLOOR_VS_FIXED_NO_EFFECT (-0.035 [-0.081, +0.002])
- **SYSTEM NLL lb2qwen/lbv2**: INCONCLUSIVE at the cell's margin 0.050 (+0.036 [-0.003, +0.084]); vs D_V4 MATCHED; at R1's m_FP 0.050: INCONCLUSIVE
- **FP accuracy hmllama/icl**: 1.0 (20 units, 0 FP-failed; closed-book 0.0, 20 need the context)
- **ACC_SYSTEM hmllama/icl**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **ACC_FLOOR hmllama/icl**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **ACC_QUEST hmllama/icl**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **SYS_VS_QUEST_ACC hmllama/icl**: SYS_VS_QUEST_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **FLOOR_VS_QUEST_ACC hmllama/icl**: FLOOR_VS_QUEST_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **SCHEDULE_ACC hmllama/icl**: SCHEDULE_ACC_NO_EFFECT, NOT_EQUIV (+0.050 [+0.000, +0.150])
- **SYS_VS_BEST_ACC hmllama/icl**: SYS_VS_BEST_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **SYS_VS_D_ACC hmllama/icl**: SYS_VS_D_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **REQ1_ACC hmllama/icl**: REQ1_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **VOTE_LOSS_ACC hmllama/icl**: VOTE_LOSS_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **ACC_SYSTEM_CTX hmllama/icl**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **ACC_FLOOR_CTX hmllama/icl**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **ACC_QUEST_CTX hmllama/icl**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **SYS_VS_QUEST_ACC_CTX hmllama/icl**: SYS_VS_QUEST_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **FLOOR_VS_QUEST_ACC_CTX hmllama/icl**: FLOOR_VS_QUEST_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SCHEDULE_ACC_CTX hmllama/icl**: SCHEDULE_ACC_NO_EFFECT (+0.050 [+0.000, +0.150])
- **SYS_VS_BEST_ACC_CTX hmllama/icl**: SYS_VS_BEST_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SYS_VS_D_ACC_CTX hmllama/icl**: SYS_VS_D_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **REQ1_ACC_CTX hmllama/icl**: REQ1_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **VOTE_LOSS_ACC_CTX hmllama/icl**: VOTE_LOSS_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SYS_VS_QUEST hmllama/icl**: SYS_VS_QUEST_NO_EFFECT (-0.049 [-0.112, +0.005])
- **SCHEDULE hmllama/icl**: SCHEDULE_NO_EFFECT (-0.034 [-0.099, +0.021])
- **FLOOR_VS_FIXED hmllama/icl**: FLOOR_VS_FIXED_NO_EFFECT (+0.001 [-0.008, +0.010])
- **SYSTEM NLL hmllama/icl**: NEAR_FP at the cell's margin 0.050 (-0.015 [-0.054, +0.019]); vs D_V4 MATCHED; at R1's m_FP 0.050: NEAR_FP
- **FP accuracy hmllama/rag**: 0.55 (20 units, 9 FP-failed; closed-book 0.45, 4 need the context)
- **ACC_SYSTEM hmllama/rag**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **ACC_FLOOR hmllama/rag**: ACC_INCONCLUSIVE (-0.050 [-0.150, +0.000])
- **ACC_QUEST hmllama/rag**: ACC_INCONCLUSIVE (+0.000 [-0.100, +0.100])
- **SYS_VS_QUEST_ACC hmllama/rag**: SYS_VS_QUEST_ACC_NO_EFFECT, NOT_EQUIV (+0.000 [-0.100, +0.100])
- **FLOOR_VS_QUEST_ACC hmllama/rag**: FLOOR_VS_QUEST_ACC_NO_EFFECT, NOT_EQUIV (-0.050 [-0.200, +0.100])
- **SCHEDULE_ACC hmllama/rag**: SCHEDULE_ACC_NO_EFFECT, NOT_EQUIV (+0.050 [+0.000, +0.150])
- **SYS_VS_BEST_ACC hmllama/rag**: SYS_VS_BEST_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **SYS_VS_D_ACC hmllama/rag**: SYS_VS_D_ACC_NO_EFFECT, NOT_EQUIV (+0.000 [-0.100, +0.100])
- **REQ1_ACC hmllama/rag**: REQ1_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **VOTE_LOSS_ACC hmllama/rag**: VOTE_LOSS_ACC_NO_EFFECT, NOT_EQUIV (-0.100 [-0.200, +0.000])
- **ACC_SYSTEM_CTX hmllama/rag**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **ACC_FLOOR_CTX hmllama/rag**: ACC_INCONCLUSIVE (-0.250 [-0.500, +0.000])
- **ACC_QUEST_CTX hmllama/rag**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **SYS_VS_QUEST_ACC_CTX hmllama/rag**: SYS_VS_QUEST_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **FLOOR_VS_QUEST_ACC_CTX hmllama/rag**: FLOOR_VS_QUEST_ACC_NO_EFFECT (-0.250 [-0.500, +0.000])
- **SCHEDULE_ACC_CTX hmllama/rag**: SCHEDULE_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SYS_VS_BEST_ACC_CTX hmllama/rag**: SYS_VS_BEST_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SYS_VS_D_ACC_CTX hmllama/rag**: SYS_VS_D_ACC_NO_EFFECT (+0.250 [+0.000, +0.500])
- **REQ1_ACC_CTX hmllama/rag**: REQ1_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **VOTE_LOSS_ACC_CTX hmllama/rag**: VOTE_LOSS_ACC_NO_EFFECT (-0.250 [-0.500, +0.000])
- **SYS_VS_QUEST hmllama/rag**: SYS_VS_QUEST_NO_EFFECT (+0.639 [-0.276, +2.205])
- **SCHEDULE hmllama/rag**: SCHEDULE_NO_EFFECT (+0.545 [-0.306, +1.985])
- **FLOOR_VS_FIXED hmllama/rag**: FLOOR_VS_FIXED_NO_EFFECT (+0.108 [-0.020, +0.328])
- **SYSTEM NLL hmllama/rag**: FAR_FROM_FP at the cell's margin 0.100 (+1.184 [+0.154, +3.110]); vs D_V4 INCONCLUSIVE; at R1's m_FP 0.050: FAR_FROM_FP
- **FP accuracy hmllama/rerank**: 0.166 (10 units, 10 FP-failed; closed-book 0.0, 1 need the context)
- **ACC_SYSTEM hmllama/rerank**: ACC_LOSS (-0.166 [-0.280, -0.060])
- **ACC_FLOOR hmllama/rerank**: ACC_INCONCLUSIVE (-0.087 [-0.193, +0.007])
- **ACC_QUEST hmllama/rerank**: ACC_INCONCLUSIVE (-0.085 [-0.165, -0.016])
- **SYS_VS_QUEST_ACC hmllama/rerank**: SYS_VS_QUEST_ACC_HURTS, NOT_EQUIV (-0.081 [-0.138, -0.025])
- **FLOOR_VS_QUEST_ACC hmllama/rerank**: FLOOR_VS_QUEST_ACC_NO_EFFECT, NOT_EQUIV (-0.002 [-0.093, +0.095])
- **SCHEDULE_ACC hmllama/rerank**: SCHEDULE_ACC_HURTS, NOT_EQUIV (-0.114 [-0.205, -0.031])
- **SYS_VS_BEST_ACC hmllama/rerank**: SYS_VS_BEST_ACC_HURTS, NOT_EQUIV (-0.169 [-0.297, -0.059])
- **SYS_VS_D_ACC hmllama/rerank**: SYS_VS_D_ACC_NO_EFFECT, NOT_EQUIV (-0.118 [-0.235, +0.000])
- **REQ1_ACC hmllama/rerank**: REQ1_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **VOTE_LOSS_ACC hmllama/rerank**: VOTE_LOSS_ACC_HURTS, NOT_EQUIV (-0.157 [-0.275, -0.054])
- **SYS_VS_QUEST hmllama/rerank**: SYS_VS_QUEST_HURTS (+14.749 [+1.052, +29.217])
- **SCHEDULE hmllama/rerank**: SCHEDULE_NO_EFFECT (+14.051 [-0.537, +28.943])
- **FLOOR_VS_FIXED hmllama/rerank**: FLOOR_VS_FIXED_HELPS (-3.219 [-4.599, -1.847])
- **SYSTEM NLL hmllama/rerank**: FAR_FROM_FP at the cell's margin 0.100 (+28.594 [+14.687, +43.223]); vs D_V4 WORSE; at R1's m_FP 0.050: FAR_FROM_FP
- **FP accuracy hmqwen/icl**: 1.0 (20 units, 0 FP-failed; closed-book 0.0, 20 need the context)
- **ACC_SYSTEM hmqwen/icl**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **ACC_FLOOR hmqwen/icl**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **ACC_QUEST hmqwen/icl**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **SYS_VS_QUEST_ACC hmqwen/icl**: SYS_VS_QUEST_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **FLOOR_VS_QUEST_ACC hmqwen/icl**: FLOOR_VS_QUEST_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **SCHEDULE_ACC hmqwen/icl**: SCHEDULE_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **SYS_VS_BEST_ACC hmqwen/icl**: SYS_VS_BEST_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **SYS_VS_D_ACC hmqwen/icl**: SYS_VS_D_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **REQ1_ACC hmqwen/icl**: REQ1_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **VOTE_LOSS_ACC hmqwen/icl**: VOTE_LOSS_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **ACC_SYSTEM_CTX hmqwen/icl**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **ACC_FLOOR_CTX hmqwen/icl**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **ACC_QUEST_CTX hmqwen/icl**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **SYS_VS_QUEST_ACC_CTX hmqwen/icl**: SYS_VS_QUEST_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **FLOOR_VS_QUEST_ACC_CTX hmqwen/icl**: FLOOR_VS_QUEST_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SCHEDULE_ACC_CTX hmqwen/icl**: SCHEDULE_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SYS_VS_BEST_ACC_CTX hmqwen/icl**: SYS_VS_BEST_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SYS_VS_D_ACC_CTX hmqwen/icl**: SYS_VS_D_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **REQ1_ACC_CTX hmqwen/icl**: REQ1_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **VOTE_LOSS_ACC_CTX hmqwen/icl**: VOTE_LOSS_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SYS_VS_QUEST hmqwen/icl**: SYS_VS_QUEST_NO_EFFECT (+0.000 [-0.005, +0.005])
- **SCHEDULE hmqwen/icl**: SCHEDULE_NO_EFFECT (+0.001 [-0.007, +0.010])
- **FLOOR_VS_FIXED hmqwen/icl**: FLOOR_VS_FIXED_NO_EFFECT (+0.002 [-0.001, +0.007])
- **SYSTEM NLL hmqwen/icl**: NEAR_FP at the cell's margin 0.050 (+0.001 [-0.004, +0.006]); vs D_V4 MATCHED; at R1's m_FP 0.050: NEAR_FP
- **FP accuracy hmqwen/rag**: 0.5 (20 units, 10 FP-failed; closed-book 0.4, 3 need the context)
- **ACC_SYSTEM hmqwen/rag**: ACC_NEAR_FP (+0.050 [+0.000, +0.150])
- **ACC_FLOOR hmqwen/rag**: ACC_NEAR_FP (+0.050 [+0.000, +0.150])
- **ACC_QUEST hmqwen/rag**: ACC_NEAR_FP (+0.050 [+0.000, +0.150])
- **SYS_VS_QUEST_ACC hmqwen/rag**: SYS_VS_QUEST_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **FLOOR_VS_QUEST_ACC hmqwen/rag**: FLOOR_VS_QUEST_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **SCHEDULE_ACC hmqwen/rag**: SCHEDULE_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **SYS_VS_BEST_ACC hmqwen/rag**: SYS_VS_BEST_ACC_NO_EFFECT, NOT_EQUIV (+0.050 [+0.000, +0.150])
- **SYS_VS_D_ACC hmqwen/rag**: SYS_VS_D_ACC_NO_EFFECT, NOT_EQUIV (+0.000 [-0.100, +0.100])
- **REQ1_ACC hmqwen/rag**: REQ1_ACC_NO_EFFECT, NOT_EQUIV (+0.050 [+0.000, +0.150])
- **VOTE_LOSS_ACC hmqwen/rag**: VOTE_LOSS_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **ACC_SYSTEM_CTX hmqwen/rag**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **ACC_FLOOR_CTX hmqwen/rag**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **ACC_QUEST_CTX hmqwen/rag**: ACC_NEAR_FP (+0.000 [+0.000, +0.000])
- **SYS_VS_QUEST_ACC_CTX hmqwen/rag**: SYS_VS_QUEST_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **FLOOR_VS_QUEST_ACC_CTX hmqwen/rag**: FLOOR_VS_QUEST_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SCHEDULE_ACC_CTX hmqwen/rag**: SCHEDULE_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SYS_VS_BEST_ACC_CTX hmqwen/rag**: SYS_VS_BEST_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SYS_VS_D_ACC_CTX hmqwen/rag**: SYS_VS_D_ACC_NO_EFFECT (+0.333 [+0.000, +0.667])
- **REQ1_ACC_CTX hmqwen/rag**: REQ1_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **VOTE_LOSS_ACC_CTX hmqwen/rag**: VOTE_LOSS_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **SYS_VS_QUEST hmqwen/rag**: SYS_VS_QUEST_HURTS (+0.219 [+0.065, +0.390])
- **SCHEDULE hmqwen/rag**: SCHEDULE_HURTS (+0.183 [+0.051, +0.320])
- **FLOOR_VS_FIXED hmqwen/rag**: FLOOR_VS_FIXED_NO_EFFECT (+0.003 [-0.077, +0.082])
- **SYSTEM NLL hmqwen/rag**: INCONCLUSIVE at the cell's margin 0.100 (+0.228 [+0.074, +0.406]); vs D_V4 MATCHED; at R1's m_FP 0.050: FAR_FROM_FP
- **FP accuracy hmqwen/rerank**: 0.266 (10 units, 10 FP-failed; closed-book 0.0, 2 need the context)
- **ACC_SYSTEM hmqwen/rerank**: ACC_LOSS (-0.261 [-0.410, -0.121])
- **ACC_FLOOR hmqwen/rerank**: ACC_LOSS (-0.266 [-0.415, -0.127])
- **ACC_QUEST hmqwen/rerank**: ACC_INCONCLUSIVE (+0.018 [-0.099, +0.145])
- **SYS_VS_QUEST_ACC hmqwen/rerank**: SYS_VS_QUEST_ACC_HURTS, NOT_EQUIV (-0.279 [-0.453, -0.123])
- **FLOOR_VS_QUEST_ACC hmqwen/rerank**: FLOOR_VS_QUEST_ACC_HURTS, NOT_EQUIV (-0.283 [-0.456, -0.128])
- **SCHEDULE_ACC hmqwen/rerank**: SCHEDULE_ACC_HURTS, NOT_EQUIV (-0.272 [-0.443, -0.122])
- **SYS_VS_BEST_ACC hmqwen/rerank**: SYS_VS_BEST_ACC_HURTS, NOT_EQUIV (-0.313 [-0.467, -0.166])
- **SYS_VS_D_ACC hmqwen/rerank**: SYS_VS_D_ACC_NO_EFFECT, NOT_EQUIV (-0.056 [-0.128, +0.000])
- **REQ1_ACC hmqwen/rerank**: REQ1_ACC_NO_EFFECT, EQUIV (+0.000 [+0.000, +0.000])
- **VOTE_LOSS_ACC hmqwen/rerank**: VOTE_LOSS_ACC_HURTS, NOT_EQUIV (-0.132 [-0.225, -0.044])
- **ACC_SYSTEM_CTX hmqwen/rerank**: ACC_LOSS (-0.725 [-0.725, -0.725])
- **ACC_FLOOR_CTX hmqwen/rerank**: ACC_LOSS (-0.725 [-0.725, -0.725])
- **ACC_QUEST_CTX hmqwen/rerank**: ACC_NEAR_FP (+0.032 [+0.032, +0.032])
- **SYS_VS_QUEST_ACC_CTX hmqwen/rerank**: SYS_VS_QUEST_ACC_HURTS (-0.757 [-0.757, -0.757])
- **FLOOR_VS_QUEST_ACC_CTX hmqwen/rerank**: FLOOR_VS_QUEST_ACC_HURTS (-0.757 [-0.757, -0.757])
- **SCHEDULE_ACC_CTX hmqwen/rerank**: SCHEDULE_ACC_HURTS (-0.757 [-0.757, -0.757])
- **SYS_VS_BEST_ACC_CTX hmqwen/rerank**: SYS_VS_BEST_ACC_HURTS (-0.725 [-0.725, -0.725])
- **SYS_VS_D_ACC_CTX hmqwen/rerank**: SYS_VS_D_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **REQ1_ACC_CTX hmqwen/rerank**: REQ1_ACC_NO_EFFECT (+0.000 [+0.000, +0.000])
- **VOTE_LOSS_ACC_CTX hmqwen/rerank**: VOTE_LOSS_ACC_HURTS (-0.290 [-0.290, -0.290])
- **SYS_VS_QUEST hmqwen/rerank**: SYS_VS_QUEST_HURTS (+36.812 [+19.448, +56.924])
- **SCHEDULE hmqwen/rerank**: SCHEDULE_HURTS (+37.224 [+19.296, +57.754])
- **FLOOR_VS_FIXED hmqwen/rerank**: FLOOR_VS_FIXED_NO_EFFECT (-0.863 [-1.976, +0.126])
- **SYSTEM NLL hmqwen/rerank**: FAR_FROM_FP at the cell's margin 0.100 (+37.864 [+19.988, +58.614]); vs D_V4 WORSE; at R1's m_FP 0.050: FAR_FROM_FP

## lb2llama / lbv2 — FP 0.425; 40 units (23 FP-failed, 6 need the context); margin 0.050

| arm@B | accuracy [90%] | Δ vs FP [90%] | Δ on FP-failed | by task |
|---|---|---|---:|---|
| kivi4_v4@4 | +0.475 [+0.350, +0.600] | +0.050 [+0.000, +0.100] | 0.087 | {'lbv2': 0.475} |
| qread4_v4@0.125 | +0.475 [+0.350, +0.600] | +0.050 [+0.000, +0.100] | 0.087 | {'lbv2': 0.475} |
| qread_v4@0.125 | +0.450 [+0.325, +0.575] | +0.025 [+0.000, +0.075] | 0.043 | {'lbv2': 0.45} |
| uniform@3 | +0.450 [+0.325, +0.575] | +0.025 [+0.000, +0.075] | 0.043 | {'lbv2': 0.45} |
| uniform@4 | +0.450 [+0.325, +0.575] | +0.025 [-0.050, +0.100] | 0.087 | {'lbv2': 0.45} |
| uniform+v4@3 | +0.450 [+0.325, +0.575] | +0.025 [+0.000, +0.075] | 0.043 | {'lbv2': 0.45} |
| uniform+v4@4 | +0.450 [+0.325, +0.575] | +0.025 [-0.050, +0.100] | 0.087 | {'lbv2': 0.45} |
| fp@0 | +0.425 [+0.300, +0.550] | +0.000 [+0.000, +0.000] | 0.0 | {'lbv2': 0.425} |
| fp8kv@8 | +0.425 [+0.300, +0.550] | +0.000 [-0.050, +0.050] | 0.043 | {'lbv2': 0.425} |
| kvquant4_v4@4 | +0.425 [+0.300, +0.550] | +0.000 [-0.050, +0.050] | 0.043 | {'lbv2': 0.425} |
| qoraclefp_v16@0.125 | +0.425 [+0.300, +0.550] | +0.000 [-0.050, +0.050] | 0.043 | {'lbv2': 0.425} |
| qread2t4kqF_v4@0.125 | +0.425 [+0.300, +0.550] | +0.000 [-0.050, +0.050] | 0.043 | {'lbv2': 0.425} |
| fp_noise@0 | +0.400 [+0.275, +0.525] | -0.025 [-0.075, +0.000] | 0.0 | {'lbv2': 0.4} |
| qread2t4kq_v4@0.125 | +0.400 [+0.275, +0.525] | -0.025 [-0.075, +0.000] | 0.0 | {'lbv2': 0.4} |
| qread2t4q_v4@0.125 | +0.400 [+0.275, +0.525] | -0.025 [-0.075, +0.000] | 0.0 | {'lbv2': 0.4} |
| qreadfp_v16@0.125 | +0.400 [+0.275, +0.525] | -0.025 [-0.075, +0.000] | 0.0 | {'lbv2': 0.4} |
| quest4_v4@0.125 | +0.400 [+0.275, +0.525] | -0.025 [-0.100, +0.050] | 0.043 | {'lbv2': 0.4} |
| quest_v16@0.125 | +0.375 [+0.250, +0.500] | -0.050 [-0.100, +0.000] | 0.0 | {'lbv2': 0.375} |
| closedbook@0 | +0.350 [+0.225, +0.475] | -0.075 [-0.200, +0.050] | 0.13 | {'lbv2': 0.35} |
| qread4q_v4@0.125 | +0.350 [+0.225, +0.475] | -0.075 [-0.150, -0.025] | 0.0 | {'lbv2': 0.35} |

**ACC_SYSTEM** (qread2t4kq_v4@0.125 − fp@0): -0.025 [-0.075, +0.000] → ACC_INCONCLUSIVE
**ACC_FLOOR** (qread2t4kqF_v4@0.125 − fp@0): +0.000 [-0.050, +0.050] → ACC_INCONCLUSIVE
**ACC_QUEST** (quest_v16@0.125 − fp@0): -0.050 [-0.100, +0.000] → ACC_INCONCLUSIVE
**SYS_VS_QUEST_ACC** (qread2t4kq_v4@0.125 − quest_v16@0.125): +0.025 [+0.000, +0.075] → SYS_VS_QUEST_ACC_NO_EFFECT, NOT_EQUIV
**FLOOR_VS_QUEST_ACC** (qread2t4kqF_v4@0.125 − quest_v16@0.125): +0.050 [+0.000, +0.100] → FLOOR_VS_QUEST_ACC_NO_EFFECT, NOT_EQUIV
**SCHEDULE_ACC** (qread4_v4@0.125 − quest4_v4@0.125): +0.075 [+0.025, +0.150] → SCHEDULE_ACC_HELPS, NOT_EQUIV
**SYS_VS_BEST_ACC** (qread2t4kq_v4@0.125 − kivi4_v4@4): -0.075 [-0.150, -0.025] → SYS_VS_BEST_ACC_HURTS, NOT_EQUIV
**SYS_VS_D_ACC** (qread2t4kq_v4@0.125 − uniform+v4@3): -0.050 [-0.100, +0.000] → SYS_VS_D_ACC_NO_EFFECT, NOT_EQUIV
**REQ1_ACC** (qread4q_v4@0.125 − qread4_v4@0.125): -0.125 [-0.225, -0.050] → REQ1_ACC_HURTS, NOT_EQUIV
**VOTE_LOSS_ACC** (qreadfp_v16@0.125 − qoraclefp_v16@0.125): -0.025 [-0.075, +0.000] → VOTE_LOSS_ACC_NO_EFFECT, NOT_EQUIV
**SYS_VS_QUEST** (NLL, qread2t4kq_v4@0.125 − quest_v16@0.125): -0.012 [-0.029, +0.006] → SYS_VS_QUEST_NO_EFFECT
**SCHEDULE** (NLL, qread4_v4@0.125 − quest4_v4@0.125): -0.009 [-0.021, +0.004] → SCHEDULE_NO_EFFECT
**FLOOR_VS_FIXED** (NLL, qread2t4kqF_v4@0.125 − qread2t4kq_v4@0.125): +0.009 [-0.003, +0.022] → FLOOR_VS_FIXED_NO_EFFECT

Quest read fraction (all layers): {'quest4_v4': 0.125, 'quest_v16': 0.125}; floor r: {'mean': 0.292, 'min': 0.149, 'max': 0.515}
4-choice KL to FP: {'closedbook@0': 0.7237, 'fp8kv@8': 0.0031, 'fp_noise@0': 0.001, 'kivi4_v4@4': 0.0088, 'kvquant4_v4@4': 0.0036, 'qoraclefp_v16@0.125': 0.0007, 'qread2t4kqF_v4@0.125': 0.0019, 'qread2t4kq_v4@0.125': 0.0031, 'qread2t4q_v4@0.125': 0.0024, 'qread4_v4@0.125': 0.0048, 'qread4q_v4@0.125': 0.0382, 'qread_v4@0.125': 0.0128, 'qreadfp_v16@0.125': 0.0017, 'quest4_v4@0.125': 0.0058, 'quest_v16@0.125': 0.0037, 'uniform@3': 0.0124, 'uniform@4': 0.0029, 'uniform+v4@3': 0.0134, 'uniform+v4@4': 0.0036}
Δ gold log-prob vs FP: closedbook@0 -0.074 [-0.530, +0.378], fp8kv@8 +0.029 [+0.005, +0.055], fp_noise@0 +0.007 [-0.017, +0.031], kivi4_v4@4 +0.046 [-0.006, +0.103], kvquant4_v4@4 +0.003 [-0.028, +0.034], qoraclefp_v16@0.125 +0.009 [-0.007, +0.026], qread2t4kqF_v4@0.125 +0.022 [-0.003, +0.046], qread2t4kq_v4@0.125 +0.003 [-0.022, +0.027], qread2t4q_v4@0.125 +0.000 [-0.031, +0.029], qread4_v4@0.125 +0.027 [-0.012, +0.067], qread4q_v4@0.125 -0.014 [-0.103, +0.075], qread_v4@0.125 +0.031 [-0.031, +0.093], qreadfp_v16@0.125 -0.002 [-0.021, +0.017], quest4_v4@0.125 -0.000 [-0.046, +0.048], quest_v16@0.125 -0.027 [-0.061, +0.003], uniform@3 +0.030 [-0.027, +0.086], uniform@4 +0.027 [-0.001, +0.056], uniform+v4@3 +0.037 [-0.020, +0.093], uniform+v4@4 +0.029 [-0.003, +0.063]
Greedy-letter accuracy: {'closedbook@0': 0.35, 'fp@0': 0.425, 'fp8kv@8': 0.425, 'fp_noise@0': 0.4, 'kivi4_v4@4': 0.475, 'kvquant4_v4@4': 0.425, 'qoraclefp_v16@0.125': 0.425, 'qread2t4kqF_v4@0.125': 0.425, 'qread2t4kq_v4@0.125': 0.4, 'qread2t4q_v4@0.125': 0.4, 'qread4_v4@0.125': 0.475, 'qread4q_v4@0.125': 0.35, 'qread_v4@0.125': 0.45, 'qreadfp_v16@0.125': 0.4, 'quest4_v4@0.125': 0.4, 'quest_v16@0.125': 0.375, 'uniform@3': 0.45, 'uniform@4': 0.45, 'uniform+v4@3': 0.45, 'uniform+v4@4': 0.475}

NLL / KL points:

| arm@B | lens | ρ | ρ mem | dP vs FP [90%] | NEAR_FP | EQUIV | KL span [90%] | dP vs D_L [90%] | MATCHED | lost / gained (McNemar p) | lost types | dS vs FP | min bias | score | peak GiB |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| closedbook@0 | V16 | 0.01 | 0.01 | +0.856 [+0.519, +1.211] | FAR_FROM_FP | NOT_EQUIV | +0.724 [+0.499, +0.968] | +0.811 [+0.472, +1.168] | WORSE | 6 / 3 (0.508) | {'confused': 6} | +0.856 [+0.519, +1.211] | +0.000 | 0.350 | 31.3 |
| qoraclefp_v16@0.125 | V16 | 0.22 | 1.67 | +0.006 [+0.000, +0.013] | NEAR_FP | EQUIV | +0.001 [+0.000, +0.001] | -0.039 [-0.076, -0.005] | MATCHED | 1 / 1 (1.000) | {'confused': 1} | +0.006 [+0.000, +0.013] | +0.000 | 0.425 | 31.2 |
| qreadfp_v16@0.125 | V16 | 0.22 | 1.67 | -0.008 [-0.019, +0.003] | NEAR_FP | EQUIV | +0.002 [+0.001, +0.002] | -0.053 [-0.090, -0.018] | MATCHED | 1 / 0 (1.000) | {'confused': 1} | -0.008 [-0.019, +0.003] | +0.000 | 0.400 | 31.2 |
| quest_v16@0.125 | V16 | 0.32 | 1.77 | -0.001 [-0.017, +0.015] | NEAR_FP | EQUIV | +0.004 [+0.003, +0.005] | -0.046 [-0.087, -0.008] | MATCHED | 2 / 0 (0.500) | {'confused': 2} | -0.001 [-0.017, +0.015] | +0.000 | 0.375 | 31.3 |
| uniform@3 | V16 | — | — | +0.045 [+0.012, +0.082] | NEAR_FP | EQUIV | +0.012 [+0.008, +0.017] | (D) | — | 0 / 1 (1.000) | — | +0.045 [+0.012, +0.082] | +0.000 | 0.450 | 29.3 |
| uniform@4 | V16 | 1.05 | 1.05 | +0.014 [-0.001, +0.029] | NEAR_FP | EQUIV | +0.003 [+0.002, +0.004] | -0.031 [-0.060, -0.003] | MATCHED | 1 / 2 (1.000) | {'confused': 1} | +0.014 [-0.001, +0.029] | +0.000 | 0.450 | 31.2 |
| fp_noise@0 | V16 | 1.67 | 1.67 | +0.003 [-0.004, +0.010] | NEAR_FP | EQUIV | +0.001 [+0.001, +0.001] | -0.042 [-0.079, -0.009] | MATCHED | 1 / 0 (1.000) | {'confused': 1} | +0.003 [-0.004, +0.010] | +0.000 | 0.400 | 31.2 |
| qread_v4@0.125 | V4 | 0.15 | 1.00 | +0.038 [+0.004, +0.077] | NEAR_FP | EQUIV | +0.013 [+0.008, +0.018] | -0.006 [-0.017, +0.004] | MATCHED | 0 / 1 (1.000) | — | +0.038 [+0.004, +0.077] | +0.000 | 0.450 | 33.4 |
| qread4_v4@0.125 | V4 | 0.16 | 1.13 | +0.013 [-0.003, +0.030] | NEAR_FP | EQUIV | +0.005 [+0.004, +0.006] | -0.031 [-0.065, +0.001] | MATCHED | 0 / 2 (0.500) | — | +0.013 [-0.003, +0.030] | +0.000 | 0.475 | 33.5 |
| qread4q_v4@0.125 | V4 | 0.16 | 1.13 | +0.067 [+0.019, +0.116] | INCONCLUSIVE | NOT_EQUIV | +0.038 [+0.026, +0.051] | +0.023 [-0.044, +0.083] | MATCHED | 3 / 0 (0.250) | {'confused': 3} | +0.067 [+0.019, +0.116] | +0.000 | 0.350 | 33.5 |
| qread2t4kq_v4@0.125 | V4 | 0.36 | 1.13 | -0.013 [-0.026, +0.001] | NEAR_FP | EQUIV | +0.003 [+0.002, +0.004] | -0.057 [-0.097, -0.020] | MATCHED | 1 / 0 (1.000) | {'confused': 1} | -0.013 [-0.026, +0.001] | +0.000 | 0.400 | 33.5 |
| quest4_v4@0.125 | V4 | 0.43 | 1.39 | +0.022 [+0.005, +0.039] | NEAR_FP | EQUIV | +0.006 [+0.004, +0.007] | -0.022 [-0.057, +0.011] | MATCHED | 2 / 1 (1.000) | {'confused': 2} | +0.022 [+0.005, +0.039] | +0.000 | 0.400 | 33.9 |
| qread2t4q_v4@0.125 | V4 | 0.56 | 1.13 | -0.013 [-0.024, -0.002] | NEAR_FP | EQUIV | +0.002 [+0.002, +0.003] | -0.057 [-0.099, -0.020] | MATCHED | 1 / 0 (1.000) | {'confused': 1} | -0.013 [-0.024, -0.002] | +0.000 | 0.400 | 33.5 |
| qread2t4kqF_v4@0.125 | V4 | 0.82 | 1.13 | -0.003 [-0.014, +0.008] | NEAR_FP | EQUIV | +0.002 [+0.001, +0.002] | -0.047 [-0.081, -0.016] | MATCHED | 1 / 1 (1.000) | {'confused': 1} | -0.003 [-0.014, +0.008] | +0.000 | 0.425 | 33.5 |
| uniform+v4@3 | V4 | — | — | +0.044 [+0.008, +0.085] | NEAR_FP | EQUIV | +0.013 [+0.009, +0.019] | (D) | — | 0 / 1 (1.000) | — | +0.044 [+0.008, +0.085] | +0.000 | 0.450 | 33.4 |
| uniform+v4@4 | V4 | 1.13 | 1.13 | +0.018 [+0.002, +0.035] | NEAR_FP | EQUIV | +0.004 [+0.003, +0.005] | -0.026 [-0.056, +0.003] | MATCHED | 1 / 2 (1.000) | {'other': 1} | +0.018 [+0.002, +0.035] | +0.000 | 0.450 | 33.4 |
| kivi4_v4@4 | V4 | 1.15 | 1.15 | +0.039 [+0.011, +0.069] | NEAR_FP | EQUIV | +0.009 [+0.006, +0.012] | -0.005 [-0.031, +0.021] | MATCHED | 0 / 2 (0.500) | — | +0.039 [+0.011, +0.069] | +0.000 | 0.475 | 33.4 |
| kvquant4_v4@4 | V4 | 1.16 | 1.16 | +0.003 [-0.011, +0.019] | NEAR_FP | EQUIV | +0.004 [+0.002, +0.006] | -0.041 [-0.074, -0.011] | MATCHED | 1 / 1 (1.000) | {'confused': 1} | +0.003 [-0.011, +0.019] | +0.000 | 0.425 | 33.4 |
| fp8kv@8 | V8 | 2.18 | 2.18 | -0.001 [-0.014, +0.012] | NEAR_FP | EQUIV | +0.003 [+0.002, +0.004] | -0.045 [-0.085, -0.010] | MATCHED | 1 / 1 (1.000) | {'confused': 1} | -0.001 [-0.014, +0.012] | +0.000 | 0.425 | 33.4 |

## lb2qwen / lbv2 — FP 0.45; 40 units (22 FP-failed, 11 need the context); margin 0.050

| arm@B | accuracy [90%] | Δ vs FP [90%] | Δ on FP-failed | by task |
|---|---|---|---:|---|
| uniform+v4@3 | +0.500 [+0.375, +0.625] | +0.050 [-0.025, +0.125] | 0.136 | {'lbv2': 0.5} |
| uniform+v4@4 | +0.500 [+0.375, +0.625] | +0.050 [+0.000, +0.100] | 0.091 | {'lbv2': 0.5} |
| kvquant4_v4@4 | +0.475 [+0.350, +0.600] | +0.025 [+0.000, +0.075] | 0.045 | {'lbv2': 0.475} |
| qread4_v4@0.125 | +0.475 [+0.350, +0.600] | +0.025 [-0.050, +0.100] | 0.091 | {'lbv2': 0.475} |
| qread4q_v4@0.125 | +0.475 [+0.350, +0.600] | +0.025 [-0.075, +0.125] | 0.182 | {'lbv2': 0.475} |
| qread_v4@0.125 | +0.475 [+0.350, +0.600] | +0.025 [-0.050, +0.100] | 0.091 | {'lbv2': 0.475} |
| quest4_v4@0.125 | +0.475 [+0.350, +0.600] | +0.025 [-0.050, +0.100] | 0.091 | {'lbv2': 0.475} |
| uniform@3 | +0.475 [+0.350, +0.600] | +0.025 [-0.050, +0.100] | 0.091 | {'lbv2': 0.475} |
| uniform@4 | +0.475 [+0.350, +0.600] | +0.025 [-0.050, +0.100] | 0.091 | {'lbv2': 0.475} |
| fp@0 | +0.450 [+0.325, +0.575] | +0.000 [+0.000, +0.000] | 0.0 | {'lbv2': 0.45} |
| fp8kv@8 | +0.450 [+0.325, +0.575] | +0.000 [+0.000, +0.000] | 0.0 | {'lbv2': 0.45} |
| qoraclefp_v16@0.125 | +0.450 [+0.325, +0.575] | +0.000 [+0.000, +0.000] | 0.0 | {'lbv2': 0.45} |
| qreadfp_v16@0.125 | +0.450 [+0.325, +0.575] | +0.000 [+0.000, +0.000] | 0.0 | {'lbv2': 0.45} |
| quest_v16@0.125 | +0.450 [+0.325, +0.575] | +0.000 [+0.000, +0.000] | 0.0 | {'lbv2': 0.45} |
| fp_noise@0 | +0.425 [+0.300, +0.550] | -0.025 [-0.075, +0.000] | 0.0 | {'lbv2': 0.425} |
| kivi4_v4@4 | +0.425 [+0.300, +0.550] | -0.025 [-0.075, +0.000] | 0.0 | {'lbv2': 0.425} |
| qread2t4kqF_v4@0.125 | +0.425 [+0.300, +0.550] | -0.025 [-0.075, +0.000] | 0.0 | {'lbv2': 0.425} |
| qread2t4q_v4@0.125 | +0.425 [+0.300, +0.550] | -0.025 [-0.075, +0.000] | 0.0 | {'lbv2': 0.425} |
| qread2t4kq_v4@0.125 | +0.400 [+0.275, +0.525] | -0.050 [-0.100, +0.000] | 0.0 | {'lbv2': 0.4} |
| closedbook@0 | +0.225 [+0.125, +0.325] | -0.225 [-0.375, -0.100] | 0.091 | {'lbv2': 0.225} |

**ACC_SYSTEM** (qread2t4kq_v4@0.125 − fp@0): -0.050 [-0.100, +0.000] → ACC_INCONCLUSIVE
**ACC_FLOOR** (qread2t4kqF_v4@0.125 − fp@0): -0.025 [-0.075, +0.000] → ACC_INCONCLUSIVE
**ACC_QUEST** (quest_v16@0.125 − fp@0): +0.000 [+0.000, +0.000] → ACC_NEAR_FP
**SYS_VS_QUEST_ACC** (qread2t4kq_v4@0.125 − quest_v16@0.125): -0.050 [-0.100, +0.000] → SYS_VS_QUEST_ACC_NO_EFFECT, NOT_EQUIV
**FLOOR_VS_QUEST_ACC** (qread2t4kqF_v4@0.125 − quest_v16@0.125): -0.025 [-0.075, +0.000] → FLOOR_VS_QUEST_ACC_NO_EFFECT, NOT_EQUIV
**SCHEDULE_ACC** (qread4_v4@0.125 − quest4_v4@0.125): +0.000 [+0.000, +0.000] → SCHEDULE_ACC_NO_EFFECT, EQUIV
**SYS_VS_BEST_ACC** (qread2t4kq_v4@0.125 − uniform+v4@4): -0.100 [-0.175, -0.025] → SYS_VS_BEST_ACC_HURTS, NOT_EQUIV
**SYS_VS_D_ACC** (qread2t4kq_v4@0.125 − uniform+v4@3): -0.100 [-0.200, +0.000] → SYS_VS_D_ACC_NO_EFFECT, NOT_EQUIV
**REQ1_ACC** (qread4q_v4@0.125 − qread4_v4@0.125): +0.000 [-0.075, +0.075] → REQ1_ACC_NO_EFFECT, NOT_EQUIV
**VOTE_LOSS_ACC** (qreadfp_v16@0.125 − qoraclefp_v16@0.125): +0.000 [+0.000, +0.000] → VOTE_LOSS_ACC_NO_EFFECT, EQUIV
**SYS_VS_QUEST** (NLL, qread2t4kq_v4@0.125 − quest_v16@0.125): +0.040 [+0.008, +0.079] → SYS_VS_QUEST_NO_EFFECT
**SCHEDULE** (NLL, qread4_v4@0.125 − quest4_v4@0.125): -0.011 [-0.025, +0.001] → SCHEDULE_NO_EFFECT
**FLOOR_VS_FIXED** (NLL, qread2t4kqF_v4@0.125 − qread2t4kq_v4@0.125): -0.035 [-0.081, +0.002] → FLOOR_VS_FIXED_NO_EFFECT

Quest read fraction (all layers): {'quest4_v4': 0.125, 'quest_v16': 0.125}; floor r: {'mean': 0.279, 'min': 0.138, 'max': 0.491}
4-choice KL to FP: {'closedbook@0': 5.176, 'fp8kv@8': 0.0136, 'fp_noise@0': 0.019, 'kivi4_v4@4': 0.0247, 'kvquant4_v4@4': 0.0182, 'qoraclefp_v16@0.125': 0.0012, 'qread2t4kqF_v4@0.125': 0.0128, 'qread2t4kq_v4@0.125': 0.0195, 'qread2t4q_v4@0.125': 0.0179, 'qread4_v4@0.125': 0.2001, 'qread4q_v4@0.125': 0.2345, 'qread_v4@0.125': 0.4676, 'qreadfp_v16@0.125': 0.004, 'quest4_v4@0.125': 0.2122, 'quest_v16@0.125': 0.0031, 'uniform@3': 0.4242, 'uniform@4': 0.1501, 'uniform+v4@3': 0.4658, 'uniform+v4@4': 0.208}
Δ gold log-prob vs FP: closedbook@0 -4.185 [-5.727, -2.673], fp8kv@8 -0.006 [-0.165, +0.156], fp_noise@0 +0.047 [-0.080, +0.181], kivi4_v4@4 +0.021 [-0.092, +0.135], kvquant4_v4@4 -0.089 [-0.286, +0.101], qoraclefp_v16@0.125 +0.040 [-0.013, +0.098], qread2t4kqF_v4@0.125 +0.068 [-0.062, +0.197], qread2t4kq_v4@0.125 +0.005 [-0.132, +0.151], qread2t4q_v4@0.125 -0.021 [-0.154, +0.112], qread4_v4@0.125 +0.205 [-0.038, +0.457], qread4q_v4@0.125 -0.174 [-0.665, +0.284], qread_v4@0.125 +0.254 [-0.240, +0.736], qreadfp_v16@0.125 -0.061 [-0.121, +0.005], quest4_v4@0.125 +0.153 [-0.106, +0.414], quest_v16@0.125 -0.043 [-0.098, +0.016], uniform@3 +0.355 [-0.192, +0.854], uniform@4 +0.138 [-0.078, +0.362], uniform+v4@3 +0.312 [-0.187, +0.795], uniform+v4@4 +0.226 [-0.022, +0.480]
Greedy-letter accuracy: {'closedbook@0': 0.225, 'fp@0': 0.45, 'fp8kv@8': 0.45, 'fp_noise@0': 0.425, 'kivi4_v4@4': 0.425, 'kvquant4_v4@4': 0.475, 'qoraclefp_v16@0.125': 0.45, 'qread2t4kqF_v4@0.125': 0.425, 'qread2t4kq_v4@0.125': 0.4, 'qread2t4q_v4@0.125': 0.425, 'qread4_v4@0.125': 0.475, 'qread4q_v4@0.125': 0.475, 'qread_v4@0.125': 0.475, 'qreadfp_v16@0.125': 0.45, 'quest4_v4@0.125': 0.475, 'quest_v16@0.125': 0.45, 'uniform@3': 0.475, 'uniform@4': 0.475, 'uniform+v4@3': 0.5, 'uniform+v4@4': 0.5}

NLL / KL points:

| arm@B | lens | ρ | ρ mem | dP vs FP [90%] | NEAR_FP | EQUIV | KL span [90%] | dP vs D_L [90%] | MATCHED | lost / gained (McNemar p) | lost types | dS vs FP | min bias | score | peak GiB |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| closedbook@0 | V16 | 0.01 | 0.01 | +5.753 [+4.191, +7.422] | FAR_FROM_FP | NOT_EQUIV | +5.224 [+3.822, +6.710] | +5.218 [+3.776, +6.715] | WORSE | 11 / 2 (0.022) | {'confused': 10, 'other': 1} | +5.753 [+4.191, +7.422] | +0.000 | 0.225 | 70.1 |
| qoraclefp_v16@0.125 | V16 | 0.22 | 1.67 | -0.006 [-0.014, +0.002] | NEAR_FP | EQUIV | +0.001 [+0.001, +0.002] | -0.541 [-1.003, -0.185] | MATCHED | 0 / 0 (1.000) | — | -0.006 [-0.014, +0.002] | +0.000 | 0.450 | 71.1 |
| qreadfp_v16@0.125 | V16 | 0.22 | 1.67 | +0.001 [-0.015, +0.016] | NEAR_FP | EQUIV | +0.004 [+0.001, +0.008] | -0.534 [-0.990, -0.182] | MATCHED | 0 / 0 (1.000) | — | +0.001 [-0.015, +0.016] | +0.000 | 0.450 | 71.1 |
| quest_v16@0.125 | V16 | 0.32 | 1.77 | -0.004 [-0.018, +0.009] | NEAR_FP | EQUIV | +0.003 [+0.001, +0.006] | -0.540 [-1.000, -0.185] | MATCHED | 0 / 0 (1.000) | — | -0.004 [-0.018, +0.009] | +0.000 | 0.450 | 71.4 |
| uniform@3 | V16 | — | — | +0.535 [+0.180, +0.998] | FAR_FROM_FP | NOT_EQUIV | +0.426 [+0.140, +0.806] | (D) | — | 1 / 2 (1.000) | {'confused': 1} | +0.535 [+0.180, +0.998] | +0.000 | 0.475 | 71.0 |
| uniform@4 | V16 | 1.05 | 1.05 | +0.140 [-0.006, +0.329] | INCONCLUSIVE | NOT_EQUIV | +0.150 [+0.027, +0.323] | -0.396 [-0.777, -0.103] | MATCHED | 1 / 2 (1.000) | {'confused': 1} | +0.140 [-0.006, +0.329] | +0.000 | 0.475 | 71.0 |
| fp_noise@0 | V16 | 1.67 | 1.67 | -0.000 [-0.028, +0.028] | NEAR_FP | EQUIV | +0.019 [+0.008, +0.032] | -0.536 [-0.997, -0.174] | MATCHED | 1 / 0 (1.000) | {'confused': 1} | -0.000 [-0.028, +0.028] | +0.000 | 0.425 | 69.9 |
| qread_v4@0.125 | V4 | 0.15 | 1.00 | +0.593 [+0.221, +1.051] | FAR_FROM_FP | NOT_EQUIV | +0.472 [+0.178, +0.840] | +0.001 [-0.022, +0.024] | MATCHED | 1 / 2 (1.000) | {'confused': 1} | +0.593 [+0.221, +1.051] | +0.000 | 0.475 | 74.3 |
| qread4_v4@0.125 | V4 | 0.16 | 1.13 | +0.198 [+0.006, +0.430] | INCONCLUSIVE | NOT_EQUIV | +0.200 [+0.036, +0.408] | -0.393 [-0.709, -0.127] | MATCHED | 1 / 2 (1.000) | {'confused': 1} | +0.198 [+0.006, +0.430] | +0.000 | 0.475 | 74.3 |
| qread4q_v4@0.125 | V4 | 0.16 | 1.13 | +0.242 [+0.110, +0.396] | FAR_FROM_FP | NOT_EQUIV | +0.235 [+0.112, +0.384] | -0.349 [-0.784, -0.009] | MATCHED | 3 / 4 (1.000) | {'confused': 3} | +0.242 [+0.110, +0.396] | +0.000 | 0.475 | 74.3 |
| qread2t4kq_v4@0.125 | V4 | 0.36 | 1.13 | +0.036 [-0.003, +0.084] | NEAR_FP | EQUIV | +0.020 [+0.006, +0.036] | -0.556 [-0.996, -0.189] | MATCHED | 2 / 0 (0.500) | {'confused': 2} | +0.036 [-0.003, +0.084] | +0.000 | 0.400 | 74.3 |
| quest4_v4@0.125 | V4 | 0.43 | 1.39 | +0.209 [+0.009, +0.447] | INCONCLUSIVE | NOT_EQUIV | +0.213 [+0.043, +0.424] | -0.382 [-0.693, -0.120] | MATCHED | 1 / 2 (1.000) | {'confused': 1} | +0.209 [+0.009, +0.447] | +0.000 | 0.475 | 74.6 |
| qread2t4q_v4@0.125 | V4 | 0.56 | 1.13 | +0.011 [-0.020, +0.051] | NEAR_FP | EQUIV | +0.018 [+0.006, +0.032] | -0.580 [-1.021, -0.217] | MATCHED | 1 / 0 (1.000) | {'confused': 1} | +0.011 [-0.020, +0.051] | +0.000 | 0.425 | 74.3 |
| qread2t4kqF_v4@0.125 | V4 | 0.78 | 1.13 | +0.000 [-0.024, +0.028] | NEAR_FP | EQUIV | +0.013 [+0.006, +0.021] | -0.591 [-1.031, -0.231] | MATCHED | 1 / 0 (1.000) | {'confused': 1} | +0.000 [-0.024, +0.028] | +0.000 | 0.425 | 74.3 |
| uniform+v4@3 | V4 | — | — | +0.591 [+0.229, +1.036] | FAR_FROM_FP | NOT_EQUIV | +0.468 [+0.181, +0.826] | (D) | — | 1 / 3 (0.625) | {'confused': 1} | +0.591 [+0.229, +1.036] | +0.000 | 0.500 | 74.2 |
| uniform+v4@4 | V4 | 1.13 | 1.13 | +0.205 [+0.012, +0.437] | INCONCLUSIVE | NOT_EQUIV | +0.208 [+0.043, +0.417] | -0.387 [-0.699, -0.123] | MATCHED | 0 / 2 (0.500) | — | +0.205 [+0.012, +0.437] | +0.000 | 0.500 | 74.2 |
| kivi4_v4@4 | V4 | 1.15 | 1.15 | +0.020 [-0.018, +0.059] | NEAR_FP | EQUIV | +0.025 [+0.011, +0.042] | -0.572 [-1.019, -0.199] | MATCHED | 1 / 0 (1.000) | {'confused': 1} | +0.020 [-0.018, +0.059] | +0.000 | 0.425 | 74.2 |
| kvquant4_v4@4 | V4 | 1.16 | 1.16 | -0.001 [-0.033, +0.033] | NEAR_FP | EQUIV | +0.018 [+0.007, +0.034] | -0.593 [-1.042, -0.226] | MATCHED | 0 / 1 (1.000) | — | -0.001 [-0.033, +0.033] | +0.000 | 0.475 | 74.2 |
| fp8kv@8 | V8 | 2.18 | 2.18 | -0.000 [-0.018, +0.018] | NEAR_FP | EQUIV | +0.014 [+0.007, +0.021] | -0.591 [-1.037, -0.228] | MATCHED | 0 / 0 (1.000) | — | -0.000 [-0.018, +0.018] | +0.000 | 0.450 | 74.2 |

## hmllama / icl — FP 1.0; 20 units (0 FP-failed, 20 need the context); margin 0.050

| arm@B | accuracy [90%] | Δ vs FP [90%] | Δ on FP-failed | by task |
|---|---|---|---:|---|
| fp@0 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| fp8kv@8 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| fp_noise@0 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| kivi4_v4@4 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| kvquant4_v4@4 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| qoraclefp_v16@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| qread2t4kqF_v4@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| qread2t4kq_v4@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| qread2t4q_v4@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| qread4_v4@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| qread4q_v4@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| qread_v4@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| qreadfp_v16@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| quest_v16@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| uniform@3 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| uniform@4 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| uniform+v4@3 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| uniform+v4@4 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| quest4_v4@0.125 | +0.950 [+0.850, +1.000] | -0.050 [-0.150, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 0.9} |
| closedbook@0 | +0.000 [+0.000, +0.000] | -1.000 [-1.000, -1.000] | nan | {'icl_banking77': 0.0, 'icl_trec_coarse': 0.0} |

**ACC_SYSTEM** (qread2t4kq_v4@0.125 − fp@0): +0.000 [+0.000, +0.000] → ACC_NEAR_FP
**ACC_FLOOR** (qread2t4kqF_v4@0.125 − fp@0): +0.000 [+0.000, +0.000] → ACC_NEAR_FP
**ACC_QUEST** (quest_v16@0.125 − fp@0): +0.000 [+0.000, +0.000] → ACC_NEAR_FP
**SYS_VS_QUEST_ACC** (qread2t4kq_v4@0.125 − quest_v16@0.125): +0.000 [+0.000, +0.000] → SYS_VS_QUEST_ACC_NO_EFFECT, EQUIV
**FLOOR_VS_QUEST_ACC** (qread2t4kqF_v4@0.125 − quest_v16@0.125): +0.000 [+0.000, +0.000] → FLOOR_VS_QUEST_ACC_NO_EFFECT, EQUIV
**SCHEDULE_ACC** (qread4_v4@0.125 − quest4_v4@0.125): +0.050 [+0.000, +0.150] → SCHEDULE_ACC_NO_EFFECT, NOT_EQUIV
**SYS_VS_BEST_ACC** (qread2t4kq_v4@0.125 − uniform+v4@4): +0.000 [+0.000, +0.000] → SYS_VS_BEST_ACC_NO_EFFECT, EQUIV
**SYS_VS_D_ACC** (qread2t4kq_v4@0.125 − uniform+v4@3): +0.000 [+0.000, +0.000] → SYS_VS_D_ACC_NO_EFFECT, EQUIV
**REQ1_ACC** (qread4q_v4@0.125 − qread4_v4@0.125): +0.000 [+0.000, +0.000] → REQ1_ACC_NO_EFFECT, EQUIV
**VOTE_LOSS_ACC** (qreadfp_v16@0.125 − qoraclefp_v16@0.125): +0.000 [+0.000, +0.000] → VOTE_LOSS_ACC_NO_EFFECT, EQUIV
**SYS_VS_QUEST** (NLL, qread2t4kq_v4@0.125 − quest_v16@0.125): -0.049 [-0.112, +0.005] → SYS_VS_QUEST_NO_EFFECT
**SCHEDULE** (NLL, qread4_v4@0.125 − quest4_v4@0.125): -0.034 [-0.099, +0.021] → SCHEDULE_NO_EFFECT
**FLOOR_VS_FIXED** (NLL, qread2t4kqF_v4@0.125 − qread2t4kq_v4@0.125): +0.001 [-0.008, +0.010] → FLOOR_VS_FIXED_NO_EFFECT

Quest read fraction (all layers): {'quest4_v4': 0.125, 'quest_v16': 0.125}; floor r: {'mean': 0.153, 'min': 0.151, 'max': 0.155}

NLL / KL points:

| arm@B | lens | ρ | ρ mem | dP vs FP [90%] | NEAR_FP | EQUIV | KL span [90%] | dP vs D_L [90%] | MATCHED | lost / gained (McNemar p) | lost types | dS vs FP | min bias | score | peak GiB |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| closedbook@0 | V16 | 0.00 | 0.00 | +5.363 [+4.572, +6.160] | FAR_FROM_FP | NOT_EQUIV | +5.095 [+4.328, +5.872] | +5.341 [+4.538, +6.153] | WORSE | 20 / 0 (0.000) | {'other': 20} | +5.363 [+4.572, +6.160] | +0.000 | 0.000 | 42.2 |
| qoraclefp_v16@0.125 | V16 | 0.21 | 1.67 | +0.007 [-0.011, +0.024] | NEAR_FP | EQUIV | +0.005 [+0.003, +0.008] | -0.015 [-0.056, +0.021] | MATCHED | 0 / 0 (1.000) | — | +0.007 [-0.011, +0.024] | +0.000 | 1.000 | 42.1 |
| qreadfp_v16@0.125 | V16 | 0.21 | 1.67 | -0.011 [-0.046, +0.021] | NEAR_FP | EQUIV | +0.012 [+0.007, +0.019] | -0.033 [-0.096, +0.019] | MATCHED | 0 / 0 (1.000) | — | -0.011 [-0.046, +0.021] | +0.000 | 1.000 | 42.2 |
| quest_v16@0.125 | V16 | 0.31 | 1.78 | +0.034 [-0.002, +0.072] | NEAR_FP | EQUIV | +0.019 [+0.010, +0.030] | +0.012 [-0.027, +0.049] | MATCHED | 0 / 0 (1.000) | — | +0.034 [-0.002, +0.072] | +0.000 | 1.000 | 42.2 |
| uniform@3 | V16 | — | — | +0.022 [-0.010, +0.058] | NEAR_FP | EQUIV | +0.014 [+0.009, +0.019] | (D) | — | 0 / 0 (1.000) | — | +0.022 [-0.010, +0.058] | +0.000 | 1.000 | 37.5 |
| uniform@4 | V16 | 1.05 | 1.05 | +0.012 [-0.008, +0.036] | NEAR_FP | EQUIV | +0.004 [+0.002, +0.007] | -0.010 [-0.028, +0.006] | MATCHED | 0 / 0 (1.000) | — | +0.012 [-0.008, +0.036] | +0.000 | 1.000 | 42.1 |
| fp_noise@0 | V16 | 1.67 | 1.67 | +0.001 [-0.006, +0.008] | NEAR_FP | EQUIV | +0.001 [+0.000, +0.001] | -0.022 [-0.054, +0.009] | MATCHED | 0 / 0 (1.000) | — | +0.001 [-0.006, +0.008] | +0.000 | 1.000 | 42.1 |
| qread_v4@0.125 | V4 | 0.13 | 1.00 | +0.026 [-0.004, +0.056] | NEAR_FP | EQUIV | +0.020 [+0.012, +0.028] | +0.005 [-0.023, +0.033] | MATCHED | 0 / 0 (1.000) | — | +0.026 [-0.004, +0.056] | +0.000 | 1.000 | 44.3 |
| qread4_v4@0.125 | V4 | 0.14 | 1.14 | -0.004 [-0.035, +0.025] | NEAR_FP | EQUIV | +0.015 [+0.008, +0.023] | -0.025 [-0.070, +0.015] | MATCHED | 0 / 0 (1.000) | — | -0.004 [-0.035, +0.025] | +0.000 | 1.000 | 44.3 |
| qread4q_v4@0.125 | V4 | 0.14 | 1.14 | +0.027 [-0.004, +0.057] | NEAR_FP | EQUIV | +0.021 [+0.013, +0.030] | +0.005 [-0.035, +0.044] | MATCHED | 0 / 0 (1.000) | — | +0.027 [-0.004, +0.057] | +0.000 | 1.000 | 44.4 |
| qread2t4kq_v4@0.125 | V4 | 0.35 | 1.14 | -0.015 [-0.054, +0.019] | NEAR_FP | EQUIV | +0.014 [+0.007, +0.022] | -0.036 [-0.099, +0.016] | MATCHED | 0 / 0 (1.000) | — | -0.015 [-0.054, +0.019] | +0.000 | 1.000 | 44.4 |
| quest4_v4@0.125 | V4 | 0.42 | 1.40 | +0.030 [-0.017, +0.085] | NEAR_FP | EQUIV | +0.027 [+0.014, +0.042] | +0.009 [-0.026, +0.046] | MATCHED | 1 / 0 (1.000) | {'other': 1} | +0.030 [-0.017, +0.085] | +0.000 | 0.950 | 45.1 |
| qread2t4kqF_v4@0.125 | V4 | 0.43 | 1.14 | -0.014 [-0.047, +0.016] | NEAR_FP | EQUIV | +0.011 [+0.006, +0.016] | -0.035 [-0.090, +0.011] | MATCHED | 0 / 0 (1.000) | — | -0.014 [-0.047, +0.016] | +0.000 | 1.000 | 44.4 |
| qread2t4q_v4@0.125 | V4 | 0.55 | 1.14 | -0.011 [-0.043, +0.018] | NEAR_FP | EQUIV | +0.010 [+0.005, +0.015] | -0.032 [-0.088, +0.015] | MATCHED | 0 / 0 (1.000) | — | -0.011 [-0.043, +0.018] | +0.000 | 1.000 | 44.4 |
| uniform+v4@3 | V4 | — | — | +0.021 [-0.008, +0.053] | NEAR_FP | EQUIV | +0.013 [+0.009, +0.018] | (D) | — | 0 / 0 (1.000) | — | +0.021 [-0.008, +0.053] | +0.000 | 1.000 | 44.3 |
| uniform+v4@4 | V4 | 1.14 | 1.14 | +0.007 [-0.014, +0.030] | NEAR_FP | EQUIV | +0.005 [+0.002, +0.007] | -0.014 [-0.028, -0.000] | MATCHED | 0 / 0 (1.000) | — | +0.007 [-0.014, +0.030] | +0.000 | 1.000 | 44.3 |
| kivi4_v4@4 | V4 | 1.15 | 1.15 | -0.007 [-0.019, +0.003] | NEAR_FP | EQUIV | +0.002 [+0.001, +0.004] | -0.028 [-0.064, +0.003] | MATCHED | 0 / 0 (1.000) | — | -0.007 [-0.019, +0.003] | +0.000 | 1.000 | 44.2 |
| kvquant4_v4@4 | V4 | 1.16 | 1.16 | +0.002 [-0.008, +0.014] | NEAR_FP | EQUIV | +0.002 [+0.001, +0.003] | -0.019 [-0.047, +0.008] | MATCHED | 0 / 0 (1.000) | — | +0.002 [-0.008, +0.014] | +0.000 | 1.000 | 44.2 |
| fp8kv@8 | V8 | 2.20 | 2.20 | -0.012 [-0.026, -0.000] | NEAR_FP | EQUIV | +0.003 [+0.002, +0.004] | -0.034 [-0.070, -0.002] | MATCHED | 0 / 0 (1.000) | — | -0.012 [-0.026, -0.000] | +0.000 | 1.000 | 44.3 |

## hmllama / rag — FP 0.55; 20 units (9 FP-failed, 4 need the context); margin 0.100

| arm@B | accuracy [90%] | Δ vs FP [90%] | Δ on FP-failed | by task |
|---|---|---|---:|---|
| fp8kv@8 | +0.600 [+0.450, +0.800] | +0.050 [+0.000, +0.150] | 0.111 | {'kilt_hotpotqa': 0.7, 'kilt_nq': 0.5} |
| qoraclefp_v16@0.125 | +0.600 [+0.450, +0.800] | +0.050 [+0.000, +0.150] | 0.111 | {'kilt_hotpotqa': 0.7, 'kilt_nq': 0.5} |
| fp@0 | +0.550 [+0.400, +0.700] | +0.000 [+0.000, +0.000] | 0.0 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| fp_noise@0 | +0.550 [+0.400, +0.700] | +0.000 [+0.000, +0.000] | 0.0 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| kvquant4_v4@4 | +0.550 [+0.400, +0.700] | +0.000 [+0.000, +0.000] | 0.0 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| qread2t4kq_v4@0.125 | +0.550 [+0.400, +0.700] | +0.000 [+0.000, +0.000] | 0.0 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| qread2t4q_v4@0.125 | +0.550 [+0.400, +0.700] | +0.000 [+0.000, +0.000] | 0.0 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| qread4_v4@0.125 | +0.550 [+0.400, +0.700] | +0.000 [+0.000, +0.000] | 0.0 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| qread4q_v4@0.125 | +0.550 [+0.400, +0.700] | +0.000 [+0.000, +0.000] | 0.0 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| quest_v16@0.125 | +0.550 [+0.350, +0.750] | +0.000 [-0.100, +0.100] | 0.111 | {'kilt_hotpotqa': 0.7, 'kilt_nq': 0.4} |
| uniform@3 | +0.550 [+0.350, +0.750] | +0.000 [-0.100, +0.100] | 0.111 | {'kilt_hotpotqa': 0.7, 'kilt_nq': 0.4} |
| uniform@4 | +0.550 [+0.400, +0.700] | +0.000 [+0.000, +0.000] | 0.0 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| uniform+v4@3 | +0.550 [+0.350, +0.750] | +0.000 [-0.100, +0.100] | 0.111 | {'kilt_hotpotqa': 0.7, 'kilt_nq': 0.4} |
| uniform+v4@4 | +0.550 [+0.400, +0.700] | +0.000 [+0.000, +0.000] | 0.0 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| kivi4_v4@4 | +0.500 [+0.300, +0.700] | -0.050 [-0.150, +0.000] | 0.0 | {'kilt_hotpotqa': 0.5, 'kilt_nq': 0.5} |
| qread2t4kqF_v4@0.125 | +0.500 [+0.300, +0.700] | -0.050 [-0.150, +0.000] | 0.0 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.4} |
| qread_v4@0.125 | +0.500 [+0.300, +0.700] | -0.050 [-0.150, +0.000] | 0.0 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.4} |
| qreadfp_v16@0.125 | +0.500 [+0.300, +0.700] | -0.050 [-0.150, +0.000] | 0.0 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.4} |
| quest4_v4@0.125 | +0.500 [+0.300, +0.700] | -0.050 [-0.150, +0.000] | 0.0 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.4} |
| closedbook@0 | +0.450 [+0.300, +0.600] | -0.100 [-0.300, +0.100] | 0.222 | {'kilt_hotpotqa': 0.4, 'kilt_nq': 0.5} |

**ACC_SYSTEM** (qread2t4kq_v4@0.125 − fp@0): +0.000 [+0.000, +0.000] → ACC_NEAR_FP
**ACC_FLOOR** (qread2t4kqF_v4@0.125 − fp@0): -0.050 [-0.150, +0.000] → ACC_INCONCLUSIVE
**ACC_QUEST** (quest_v16@0.125 − fp@0): +0.000 [-0.100, +0.100] → ACC_INCONCLUSIVE
**SYS_VS_QUEST_ACC** (qread2t4kq_v4@0.125 − quest_v16@0.125): +0.000 [-0.100, +0.100] → SYS_VS_QUEST_ACC_NO_EFFECT, NOT_EQUIV
**FLOOR_VS_QUEST_ACC** (qread2t4kqF_v4@0.125 − quest_v16@0.125): -0.050 [-0.200, +0.100] → FLOOR_VS_QUEST_ACC_NO_EFFECT, NOT_EQUIV
**SCHEDULE_ACC** (qread4_v4@0.125 − quest4_v4@0.125): +0.050 [+0.000, +0.150] → SCHEDULE_ACC_NO_EFFECT, NOT_EQUIV
**SYS_VS_BEST_ACC** (qread2t4kq_v4@0.125 − uniform+v4@4): +0.000 [+0.000, +0.000] → SYS_VS_BEST_ACC_NO_EFFECT, EQUIV
**SYS_VS_D_ACC** (qread2t4kq_v4@0.125 − uniform+v4@3): +0.000 [-0.100, +0.100] → SYS_VS_D_ACC_NO_EFFECT, NOT_EQUIV
**REQ1_ACC** (qread4q_v4@0.125 − qread4_v4@0.125): +0.000 [+0.000, +0.000] → REQ1_ACC_NO_EFFECT, EQUIV
**VOTE_LOSS_ACC** (qreadfp_v16@0.125 − qoraclefp_v16@0.125): -0.100 [-0.200, +0.000] → VOTE_LOSS_ACC_NO_EFFECT, NOT_EQUIV
**SYS_VS_QUEST** (NLL, qread2t4kq_v4@0.125 − quest_v16@0.125): +0.639 [-0.276, +2.205] → SYS_VS_QUEST_NO_EFFECT
**SCHEDULE** (NLL, qread4_v4@0.125 − quest4_v4@0.125): +0.545 [-0.306, +1.985] → SCHEDULE_NO_EFFECT
**FLOOR_VS_FIXED** (NLL, qread2t4kqF_v4@0.125 − qread2t4kq_v4@0.125): +0.108 [-0.020, +0.328] → FLOOR_VS_FIXED_NO_EFFECT

Quest read fraction (all layers): {'quest4_v4': 0.125, 'quest_v16': 0.125}; floor r: {'mean': 0.136, 'min': 0.125, 'max': 0.15}

NLL / KL points:

| arm@B | lens | ρ | ρ mem | dP vs FP [90%] | NEAR_FP | EQUIV | KL span [90%] | dP vs D_L [90%] | MATCHED | lost / gained (McNemar p) | lost types | dS vs FP | min bias | score | peak GiB |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| closedbook@0 | V16 | 0.00 | 0.00 | +4.556 [+2.413, +7.193] | FAR_FROM_FP | NOT_EQUIV | +5.144 [+3.599, +6.981] | +4.040 [+2.152, +6.360] | WORSE | 4 / 2 (0.688) | {'other': 4} | +4.556 [+2.413, +7.193] | +0.000 | 0.450 | 45.7 |
| qreadfp_v16@0.125 | V16 | 0.21 | 1.67 | +1.250 [+0.139, +3.143] | FAR_FROM_FP | NOT_EQUIV | +0.884 [+0.073, +2.294] | +0.734 [-0.272, +2.327] | INCONCLUSIVE | 1 / 0 (1.000) | {'other': 1} | +1.250 [+0.139, +3.143] | +0.000 | 0.500 | 45.6 |
| qoraclefp_v16@0.125 | V16 | 0.21 | 1.67 | +0.306 [+0.158, +0.483] | FAR_FROM_FP | NOT_EQUIV | +0.057 [+0.027, +0.093] | -0.211 [-0.445, -0.006] | MATCHED | 0 / 1 (1.000) | — | +0.306 [+0.158, +0.483] | +0.000 | 0.600 | 45.6 |
| quest_v16@0.125 | V16 | 0.31 | 1.78 | +0.545 [+0.244, +0.900] | FAR_FROM_FP | NOT_EQUIV | +0.250 [+0.123, +0.409] | +0.028 [-0.275, +0.305] | INCONCLUSIVE | 1 / 1 (1.000) | {'other': 1} | +0.545 [+0.244, +0.900] | +0.000 | 0.550 | 45.7 |
| uniform@3 | V16 | — | — | +0.517 [+0.220, +0.861] | FAR_FROM_FP | NOT_EQUIV | +0.147 [+0.080, +0.221] | (D) | — | 1 / 1 (1.000) | {'other': 1} | +0.517 [+0.220, +0.861] | +0.000 | 0.550 | 40.4 |
| uniform@4 | V16 | 1.05 | 1.05 | +0.126 [+0.053, +0.207] | INCONCLUSIVE | NOT_EQUIV | +0.032 [+0.019, +0.047] | -0.391 [-0.734, -0.096] | MATCHED | 0 / 0 (1.000) | — | +0.126 [+0.053, +0.207] | +0.000 | 0.550 | 45.6 |
| fp_noise@0 | V16 | 1.67 | 1.67 | +0.029 [+0.011, +0.050] | NEAR_FP | EQUIV | +0.005 [+0.003, +0.007] | -0.488 [-0.838, -0.187] | MATCHED | 0 / 0 (1.000) | — | +0.029 [+0.011, +0.050] | +0.000 | 0.550 | 45.6 |
| qread_v4@0.125 | V4 | 0.13 | 1.00 | +1.536 [+0.370, +3.394] | FAR_FROM_FP | NOT_EQUIV | +0.965 [+0.169, +2.356] | +1.005 [+0.083, +2.570] | INCONCLUSIVE | 1 / 0 (1.000) | {'other': 1} | +1.536 [+0.370, +3.394] | +0.000 | 0.500 | 48.1 |
| qread4q_v4@0.125 | V4 | 0.15 | 1.14 | +1.575 [+0.358, +3.469] | FAR_FROM_FP | NOT_EQUIV | +0.963 [+0.156, +2.401] | +1.045 [+0.061, +2.640] | INCONCLUSIVE | 0 / 0 (1.000) | — | +1.575 [+0.358, +3.469] | +0.000 | 0.550 | 48.2 |
| qread4_v4@0.125 | V4 | 0.15 | 1.14 | +1.301 [+0.230, +3.218] | FAR_FROM_FP | NOT_EQUIV | +0.862 [+0.097, +2.313] | +0.771 [-0.140, +2.361] | INCONCLUSIVE | 0 / 0 (1.000) | — | +1.301 [+0.230, +3.218] | +0.000 | 0.550 | 48.2 |
| qread2t4kq_v4@0.125 | V4 | 0.35 | 1.14 | +1.184 [+0.154, +3.110] | FAR_FROM_FP | NOT_EQUIV | +0.825 [+0.080, +2.249] | +0.654 [-0.261, +2.234] | INCONCLUSIVE | 0 / 0 (1.000) | — | +1.184 [+0.154, +3.110] | +0.000 | 0.550 | 48.2 |
| qread2t4kqF_v4@0.125 | V4 | 0.38 | 1.14 | +1.292 [+0.194, +3.178] | FAR_FROM_FP | NOT_EQUIV | +0.882 [+0.089, +2.281] | +0.762 [-0.191, +2.318] | INCONCLUSIVE | 1 / 0 (1.000) | {'other': 1} | +1.292 [+0.194, +3.178] | +0.000 | 0.500 | 48.2 |
| quest4_v4@0.125 | V4 | 0.42 | 1.40 | +0.756 [+0.364, +1.232] | FAR_FROM_FP | NOT_EQUIV | +0.308 [+0.159, +0.497] | +0.226 [-0.085, +0.514] | INCONCLUSIVE | 1 / 0 (1.000) | {'other': 1} | +0.756 [+0.364, +1.232] | +0.000 | 0.500 | 49.0 |
| qread2t4q_v4@0.125 | V4 | 0.55 | 1.14 | +1.144 [+0.139, +3.036] | FAR_FROM_FP | NOT_EQUIV | +0.812 [+0.076, +2.215] | +0.614 [-0.293, +2.150] | INCONCLUSIVE | 0 / 0 (1.000) | — | +1.144 [+0.139, +3.036] | +0.000 | 0.550 | 48.2 |
| uniform+v4@3 | V4 | — | — | +0.530 [+0.209, +0.905] | FAR_FROM_FP | NOT_EQUIV | +0.156 [+0.082, +0.240] | (D) | — | 1 / 1 (1.000) | {'other': 1} | +0.530 [+0.209, +0.905] | +0.000 | 0.550 | 48.1 |
| uniform+v4@4 | V4 | 1.14 | 1.14 | +0.130 [+0.040, +0.234] | INCONCLUSIVE | NOT_EQUIV | +0.036 [+0.021, +0.053] | -0.400 [-0.776, -0.085] | MATCHED | 0 / 0 (1.000) | — | +0.130 [+0.040, +0.234] | +0.000 | 0.550 | 48.1 |
| kivi4_v4@4 | V4 | 1.15 | 1.15 | +0.161 [+0.064, +0.271] | INCONCLUSIVE | NOT_EQUIV | +0.049 [+0.025, +0.079] | -0.369 [-0.702, -0.075] | MATCHED | 1 / 0 (1.000) | {'other': 1} | +0.161 [+0.064, +0.271] | +0.000 | 0.500 | 48.0 |
| kvquant4_v4@4 | V4 | 1.16 | 1.16 | +0.142 [+0.022, +0.332] | INCONCLUSIVE | NOT_EQUIV | +0.045 [+0.017, +0.087] | -0.388 [-0.637, -0.167] | MATCHED | 0 / 0 (1.000) | — | +0.142 [+0.022, +0.332] | +0.000 | 0.550 | 48.0 |
| fp8kv@8 | V8 | 2.20 | 2.20 | +0.081 [+0.005, +0.182] | INCONCLUSIVE | NOT_EQUIV | +0.020 [+0.012, +0.030] | -0.449 [-0.740, -0.196] | MATCHED | 0 / 1 (1.000) | — | +0.081 [+0.005, +0.182] | +0.000 | 0.600 | 48.1 |

## hmllama / rerank — FP 0.166; 10 units (10 FP-failed, 1 need the context); margin 0.100

| arm@B | accuracy [90%] | Δ vs FP [90%] | Δ on FP-failed | by task |
|---|---|---|---:|---|
| fp8kv@8 | +0.184 [+0.065, +0.319] | +0.018 [-0.006, +0.047] | 0.018 | {'msmarco_rerank_psg': 0.184} |
| uniform+v4@4 | +0.169 [+0.059, +0.297] | +0.003 [-0.027, +0.031] | 0.003 | {'msmarco_rerank_psg': 0.169} |
| fp@0 | +0.166 [+0.060, +0.280] | +0.000 [+0.000, +0.000] | 0.0 | {'msmarco_rerank_psg': 0.166} |
| qoraclefp_v16@0.125 | +0.157 [+0.054, +0.275] | -0.008 [-0.033, +0.013] | -0.008 | {'msmarco_rerank_psg': 0.157} |
| uniform@4 | +0.151 [+0.048, +0.276] | -0.014 [-0.044, +0.015] | -0.014 | {'msmarco_rerank_psg': 0.151} |
| fp_noise@0 | +0.148 [+0.050, +0.260] | -0.018 [-0.041, -0.000] | -0.018 | {'msmarco_rerank_psg': 0.148} |
| uniform@3 | +0.132 [+0.000, +0.264] | -0.034 [-0.099, +0.029] | -0.034 | {'msmarco_rerank_psg': 0.132} |
| uniform+v4@3 | +0.118 [+0.000, +0.235] | -0.048 [-0.109, +0.008] | -0.048 | {'msmarco_rerank_psg': 0.118} |
| quest4_v4@0.125 | +0.114 [+0.031, +0.205] | -0.052 [-0.180, +0.047] | -0.052 | {'msmarco_rerank_psg': 0.114} |
| quest_v16@0.125 | +0.081 [+0.025, +0.138] | -0.085 [-0.165, -0.016] | -0.085 | {'msmarco_rerank_psg': 0.081} |
| qread2t4kqF_v4@0.125 | +0.078 [+0.000, +0.160] | -0.087 [-0.193, +0.007] | -0.087 | {'msmarco_rerank_psg': 0.078} |
| kvquant4_v4@4 | +0.055 [+0.015, +0.104] | -0.111 [-0.223, -0.019] | -0.111 | {'msmarco_rerank_psg': 0.055} |
| kivi4_v4@4 | +0.037 [+0.000, +0.082] | -0.128 [-0.237, -0.041] | -0.128 | {'msmarco_rerank_psg': 0.037} |
| qread_v4@0.125 | +0.016 [+0.000, +0.047] | -0.150 [-0.267, -0.044] | -0.15 | {'msmarco_rerank_psg': 0.016} |
| closedbook@0 | +0.000 [+0.000, +0.000] | -0.166 [-0.280, -0.060] | -0.166 | {'msmarco_rerank_psg': 0.0} |
| qread2t4kq_v4@0.125 | +0.000 [+0.000, +0.000] | -0.166 [-0.280, -0.060] | -0.166 | {'msmarco_rerank_psg': 0.0} |
| qread2t4q_v4@0.125 | +0.000 [+0.000, +0.000] | -0.166 [-0.280, -0.060] | -0.166 | {'msmarco_rerank_psg': 0.0} |
| qread4_v4@0.125 | +0.000 [+0.000, +0.000] | -0.166 [-0.280, -0.060] | -0.166 | {'msmarco_rerank_psg': 0.0} |
| qread4q_v4@0.125 | +0.000 [+0.000, +0.000] | -0.166 [-0.280, -0.060] | -0.166 | {'msmarco_rerank_psg': 0.0} |
| qreadfp_v16@0.125 | +0.000 [+0.000, +0.000] | -0.166 [-0.280, -0.060] | -0.166 | {'msmarco_rerank_psg': 0.0} |

**ACC_SYSTEM** (qread2t4kq_v4@0.125 − fp@0): -0.166 [-0.280, -0.060] → ACC_LOSS
**ACC_FLOOR** (qread2t4kqF_v4@0.125 − fp@0): -0.087 [-0.193, +0.007] → ACC_INCONCLUSIVE
**ACC_QUEST** (quest_v16@0.125 − fp@0): -0.085 [-0.165, -0.016] → ACC_INCONCLUSIVE
**SYS_VS_QUEST_ACC** (qread2t4kq_v4@0.125 − quest_v16@0.125): -0.081 [-0.138, -0.025] → SYS_VS_QUEST_ACC_HURTS, NOT_EQUIV
**FLOOR_VS_QUEST_ACC** (qread2t4kqF_v4@0.125 − quest_v16@0.125): -0.002 [-0.093, +0.095] → FLOOR_VS_QUEST_ACC_NO_EFFECT, NOT_EQUIV
**SCHEDULE_ACC** (qread4_v4@0.125 − quest4_v4@0.125): -0.114 [-0.205, -0.031] → SCHEDULE_ACC_HURTS, NOT_EQUIV
**SYS_VS_BEST_ACC** (qread2t4kq_v4@0.125 − uniform+v4@4): -0.169 [-0.297, -0.059] → SYS_VS_BEST_ACC_HURTS, NOT_EQUIV
**SYS_VS_D_ACC** (qread2t4kq_v4@0.125 − uniform+v4@3): -0.118 [-0.235, +0.000] → SYS_VS_D_ACC_NO_EFFECT, NOT_EQUIV
**REQ1_ACC** (qread4q_v4@0.125 − qread4_v4@0.125): +0.000 [+0.000, +0.000] → REQ1_ACC_NO_EFFECT, EQUIV
**VOTE_LOSS_ACC** (qreadfp_v16@0.125 − qoraclefp_v16@0.125): -0.157 [-0.275, -0.054] → VOTE_LOSS_ACC_HURTS, NOT_EQUIV
**SYS_VS_QUEST** (NLL, qread2t4kq_v4@0.125 − quest_v16@0.125): +14.749 [+1.052, +29.217] → SYS_VS_QUEST_HURTS
**SCHEDULE** (NLL, qread4_v4@0.125 − quest4_v4@0.125): +14.051 [-0.537, +28.943] → SCHEDULE_NO_EFFECT
**FLOOR_VS_FIXED** (NLL, qread2t4kqF_v4@0.125 − qread2t4kq_v4@0.125): -3.219 [-4.599, -1.847] → FLOOR_VS_FIXED_HELPS

Quest read fraction (all layers): {'quest4_v4': 0.125, 'quest_v16': 0.125}; floor r: {'mean': 0.187, 'min': 0.169, 'max': 0.218}

NLL / KL points:

| arm@B | lens | ρ | ρ mem | dP vs FP [90%] | NEAR_FP | EQUIV | KL span [90%] | dP vs D_L [90%] | MATCHED | lost / gained (McNemar p) | lost types | dS vs FP | min bias | score | peak GiB |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| closedbook@0 | V16 | 0.00 | 0.00 | +51.195 [+31.601, +72.498] | FAR_FROM_FP | NOT_EQUIV | +51.403 [+36.151, +67.847] | +47.614 [+29.248, +67.522] | WORSE | 5 / 0 (0.062) | {'other': 5} | +51.195 [+31.601, +72.498] | +0.000 | 0.000 | 37.7 |
| qoraclefp_v16@0.125 | V16 | 0.21 | 1.67 | +0.480 [-0.102, +1.088] | INCONCLUSIVE | NOT_EQUIV | +0.597 [+0.429, +0.767] | -3.101 [-5.229, -1.176] | MATCHED | 2 / 1 (1.000) | {'other': 1, 'incomplete': 1} | +0.480 [-0.102, +1.088] | +0.000 | 0.157 | 37.3 |
| qreadfp_v16@0.125 | V16 | 0.21 | 1.67 | +29.081 [+15.133, +43.768] | FAR_FROM_FP | NOT_EQUIV | +26.445 [+14.611, +38.854] | +25.499 [+13.283, +38.238] | WORSE | 5 / 0 (0.062) | {'other': 3, 'incomplete': 2} | +29.081 [+15.133, +43.768] | +0.000 | 0.000 | 37.4 |
| quest_v16@0.125 | V16 | 0.32 | 1.78 | +13.845 [+7.503, +21.431] | FAR_FROM_FP | NOT_EQUIV | +12.136 [+6.601, +18.663] | +10.264 [+3.849, +18.220] | WORSE | 4 / 2 (0.688) | {'other': 3, 'incomplete': 1} | +13.845 [+7.503, +21.431] | +0.000 | 0.081 | 37.4 |
| uniform@3 | V16 | — | — | +3.581 [+1.402, +6.060] | FAR_FROM_FP | NOT_EQUIV | +2.688 [+1.504, +3.982] | (D) | — | 3 / 2 (1.000) | {'other': 3} | +3.581 [+1.402, +6.060] | +0.000 | 0.132 | 34.0 |
| uniform@4 | V16 | 1.05 | 1.05 | -0.062 [-0.479, +0.332] | INCONCLUSIVE | NOT_EQUIV | +0.833 [+0.455, +1.240] | -3.644 [-6.029, -1.543] | MATCHED | 3 / 1 (0.625) | {'other': 1, 'incomplete': 2} | -0.062 [-0.479, +0.332] | +0.000 | 0.151 | 37.4 |
| fp_noise@0 | V16 | 1.67 | 1.67 | -0.011 [-0.110, +0.103] | INCONCLUSIVE | NOT_EQUIV | +0.081 [+0.043, +0.127] | -3.592 [-6.082, -1.409] | MATCHED | 3 / 0 (0.250) | {'incomplete': 3} | -0.011 [-0.110, +0.103] | +0.000 | 0.148 | 37.3 |
| qread_v4@0.125 | V4 | 0.13 | 1.00 | +29.110 [+14.312, +44.723] | FAR_FROM_FP | NOT_EQUIV | +26.107 [+13.923, +38.889] | +25.298 [+12.420, +38.951] | WORSE | 4 / 0 (0.125) | {'other': 4} | +29.110 [+14.312, +44.723] | +0.000 | 0.016 | 39.7 |
| qread4_v4@0.125 | V4 | 0.15 | 1.14 | +28.270 [+14.213, +43.024] | FAR_FROM_FP | NOT_EQUIV | +25.964 [+14.018, +38.458] | +24.458 [+12.300, +37.254] | WORSE | 5 / 0 (0.062) | {'other': 4, 'incomplete': 1} | +28.270 [+14.213, +43.024] | +0.000 | 0.000 | 39.7 |
| qread4q_v4@0.125 | V4 | 0.15 | 1.14 | +28.547 [+14.706, +43.165] | FAR_FROM_FP | NOT_EQUIV | +25.965 [+14.004, +38.469] | +24.735 [+12.684, +37.372] | WORSE | 5 / 0 (0.062) | {'other': 2, 'incomplete': 3} | +28.547 [+14.706, +43.165] | +0.000 | 0.000 | 39.7 |
| qread2t4kq_v4@0.125 | V4 | 0.35 | 1.14 | +28.594 [+14.687, +43.223] | FAR_FROM_FP | NOT_EQUIV | +26.137 [+14.246, +38.509] | +24.781 [+12.698, +37.434] | WORSE | 5 / 0 (0.062) | {'other': 3, 'incomplete': 2} | +28.594 [+14.687, +43.223] | +0.000 | 0.000 | 39.8 |
| quest4_v4@0.125 | V4 | 0.42 | 1.40 | +14.219 [+6.868, +23.716] | FAR_FROM_FP | NOT_EQUIV | +13.179 [+6.627, +21.274] | +10.407 [+2.994, +19.783] | WORSE | 3 / 2 (1.000) | {'other': 2, 'incomplete': 1} | +14.219 [+6.868, +23.716] | +0.000 | 0.114 | 39.8 |
| qread2t4kqF_v4@0.125 | V4 | 0.53 | 1.14 | +25.375 [+11.977, +39.500] | FAR_FROM_FP | NOT_EQUIV | +23.870 [+12.464, +35.732] | +21.562 [+10.014, +33.634] | WORSE | 3 / 2 (1.000) | {'incomplete': 2, 'other': 1} | +25.375 [+11.977, +39.500] | +0.000 | 0.078 | 39.7 |
| qread2t4q_v4@0.125 | V4 | 0.56 | 1.14 | +28.269 [+14.356, +42.867] | FAR_FROM_FP | NOT_EQUIV | +26.033 [+14.172, +38.423] | +24.457 [+12.368, +37.208] | WORSE | 5 / 0 (0.062) | {'other': 4, 'incomplete': 1} | +28.269 [+14.356, +42.867] | +0.000 | 0.000 | 39.8 |
| uniform+v4@3 | V4 | — | — | +3.812 [+1.609, +6.286] | FAR_FROM_FP | NOT_EQUIV | +2.757 [+1.574, +4.055] | (D) | — | 3 / 2 (1.000) | {'other': 3} | +3.812 [+1.609, +6.286] | +0.000 | 0.118 | 39.6 |
| uniform+v4@4 | V4 | 1.14 | 1.14 | +0.227 [-0.170, +0.663] | INCONCLUSIVE | NOT_EQUIV | +0.876 [+0.484, +1.290] | -3.585 [-5.898, -1.539] | MATCHED | 1 / 3 (0.625) | {'incomplete': 1} | +0.227 [-0.170, +0.663] | +0.000 | 0.169 | 39.6 |
| kivi4_v4@4 | V4 | 1.16 | 1.16 | +1.272 [+0.750, +1.800] | FAR_FROM_FP | NOT_EQUIV | +0.799 [+0.498, +1.126] | -2.540 [-4.697, -0.589] | MATCHED | 5 / 0 (0.062) | {'other': 2, 'incomplete': 3} | +1.272 [+0.750, +1.800] | +0.000 | 0.037 | 39.6 |
| kvquant4_v4@4 | V4 | 1.16 | 1.16 | +0.559 [-0.131, +1.252] | INCONCLUSIVE | NOT_EQUIV | +0.595 [+0.387, +0.815] | -3.254 [-5.571, -1.137] | MATCHED | 4 / 0 (0.125) | {'other': 2, 'incomplete': 2} | +0.559 [-0.131, +1.252] | +0.000 | 0.055 | 39.5 |
| fp8kv@8 | V8 | 2.20 | 2.20 | +0.374 [-0.013, +0.934] | INCONCLUSIVE | NOT_EQUIV | +0.382 [+0.199, +0.600] | -3.438 [-5.648, -1.513] | MATCHED | 1 / 3 (0.625) | {'incomplete': 1} | +0.374 [-0.013, +0.934] | +0.000 | 0.184 | 39.6 |

## hmqwen / icl — FP 1.0; 20 units (0 FP-failed, 20 need the context); margin 0.050

| arm@B | accuracy [90%] | Δ vs FP [90%] | Δ on FP-failed | by task |
|---|---|---|---:|---|
| fp@0 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| fp8kv@8 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| fp_noise@0 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| kivi4_v4@4 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| kvquant4_v4@4 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| qoraclefp_v16@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| qread2t4kqF_v4@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| qread2t4kq_v4@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| qread2t4q_v4@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| qread4_v4@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| qread4q_v4@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| qread_v4@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| qreadfp_v16@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| quest4_v4@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| quest_v16@0.125 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| uniform@3 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| uniform@4 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| uniform+v4@3 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| uniform+v4@4 | +1.000 [+1.000, +1.000] | +0.000 [+0.000, +0.000] | nan | {'icl_banking77': 1.0, 'icl_trec_coarse': 1.0} |
| closedbook@0 | +0.000 [+0.000, +0.000] | -1.000 [-1.000, -1.000] | nan | {'icl_banking77': 0.0, 'icl_trec_coarse': 0.0} |

**ACC_SYSTEM** (qread2t4kq_v4@0.125 − fp@0): +0.000 [+0.000, +0.000] → ACC_NEAR_FP
**ACC_FLOOR** (qread2t4kqF_v4@0.125 − fp@0): +0.000 [+0.000, +0.000] → ACC_NEAR_FP
**ACC_QUEST** (quest_v16@0.125 − fp@0): +0.000 [+0.000, +0.000] → ACC_NEAR_FP
**SYS_VS_QUEST_ACC** (qread2t4kq_v4@0.125 − quest_v16@0.125): +0.000 [+0.000, +0.000] → SYS_VS_QUEST_ACC_NO_EFFECT, EQUIV
**FLOOR_VS_QUEST_ACC** (qread2t4kqF_v4@0.125 − quest_v16@0.125): +0.000 [+0.000, +0.000] → FLOOR_VS_QUEST_ACC_NO_EFFECT, EQUIV
**SCHEDULE_ACC** (qread4_v4@0.125 − quest4_v4@0.125): +0.000 [+0.000, +0.000] → SCHEDULE_ACC_NO_EFFECT, EQUIV
**SYS_VS_BEST_ACC** (qread2t4kq_v4@0.125 − uniform+v4@4): +0.000 [+0.000, +0.000] → SYS_VS_BEST_ACC_NO_EFFECT, EQUIV
**SYS_VS_D_ACC** (qread2t4kq_v4@0.125 − uniform+v4@3): +0.000 [+0.000, +0.000] → SYS_VS_D_ACC_NO_EFFECT, EQUIV
**REQ1_ACC** (qread4q_v4@0.125 − qread4_v4@0.125): +0.000 [+0.000, +0.000] → REQ1_ACC_NO_EFFECT, EQUIV
**VOTE_LOSS_ACC** (qreadfp_v16@0.125 − qoraclefp_v16@0.125): +0.000 [+0.000, +0.000] → VOTE_LOSS_ACC_NO_EFFECT, EQUIV
**SYS_VS_QUEST** (NLL, qread2t4kq_v4@0.125 − quest_v16@0.125): +0.000 [-0.005, +0.005] → SYS_VS_QUEST_NO_EFFECT
**SCHEDULE** (NLL, qread4_v4@0.125 − quest4_v4@0.125): +0.001 [-0.007, +0.010] → SCHEDULE_NO_EFFECT
**FLOOR_VS_FIXED** (NLL, qread2t4kqF_v4@0.125 − qread2t4kq_v4@0.125): +0.002 [-0.001, +0.007] → FLOOR_VS_FIXED_NO_EFFECT

Quest read fraction (all layers): {'quest4_v4': 0.125, 'quest_v16': 0.125}; floor r: {'mean': 0.149, 'min': 0.144, 'max': 0.153}

NLL / KL points:

| arm@B | lens | ρ | ρ mem | dP vs FP [90%] | NEAR_FP | EQUIV | KL span [90%] | dP vs D_L [90%] | MATCHED | lost / gained (McNemar p) | lost types | dS vs FP | min bias | score | peak GiB |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| closedbook@0 | V16 | 0.00 | 0.00 | +3.899 [+3.188, +4.619] | FAR_FROM_FP | NOT_EQUIV | +3.823 [+3.113, +4.540] | +3.880 [+3.161, +4.608] | WORSE | 20 / 0 (0.000) | {'other': 20} | +3.899 [+3.188, +4.619] | +0.000 | 0.000 | 78.3 |
| qoraclefp_v16@0.125 | V16 | 0.21 | 1.67 | +0.011 [+0.006, +0.016] | NEAR_FP | EQUIV | +0.002 [+0.001, +0.003] | -0.009 [-0.025, +0.009] | MATCHED | 0 / 0 (1.000) | — | +0.011 [+0.006, +0.016] | +0.000 | 1.000 | 78.2 |
| qreadfp_v16@0.125 | V16 | 0.21 | 1.67 | +0.002 [-0.003, +0.006] | NEAR_FP | EQUIV | +0.001 [+0.001, +0.002] | -0.018 [-0.037, +0.001] | MATCHED | 0 / 0 (1.000) | — | +0.002 [-0.003, +0.006] | +0.000 | 1.000 | 78.3 |
| quest_v16@0.125 | V16 | 0.31 | 1.78 | +0.001 [-0.005, +0.008] | NEAR_FP | EQUIV | +0.002 [+0.001, +0.004] | -0.019 [-0.035, -0.003] | MATCHED | 0 / 0 (1.000) | — | +0.001 [-0.005, +0.008] | +0.000 | 1.000 | 78.3 |
| uniform@3 | V16 | — | — | +0.020 [+0.002, +0.038] | NEAR_FP | EQUIV | +0.022 [+0.010, +0.037] | (D) | — | 0 / 0 (1.000) | — | +0.020 [+0.002, +0.038] | +0.000 | 1.000 | 76.9 |
| uniform@4 | V16 | 1.05 | 1.05 | +0.009 [-0.003, +0.026] | NEAR_FP | EQUIV | +0.005 [+0.001, +0.011] | -0.010 [-0.030, +0.008] | MATCHED | 0 / 0 (1.000) | — | +0.009 [-0.003, +0.026] | +0.000 | 1.000 | 78.3 |
| fp_noise@0 | V16 | 1.67 | 1.67 | +0.000 [-0.003, +0.003] | NEAR_FP | EQUIV | +0.001 [+0.000, +0.001] | -0.019 [-0.038, -0.001] | MATCHED | 0 / 0 (1.000) | — | +0.000 [-0.003, +0.003] | +0.000 | 1.000 | 78.3 |
| qread_v4@0.125 | V4 | 0.13 | 1.00 | +0.022 [+0.004, +0.042] | NEAR_FP | EQUIV | +0.018 [+0.010, +0.027] | -0.000 [-0.010, +0.008] | MATCHED | 0 / 0 (1.000) | — | +0.022 [+0.004, +0.042] | +0.000 | 1.000 | 82.2 |
| qread4_v4@0.125 | V4 | 0.14 | 1.14 | +0.005 [+0.002, +0.010] | NEAR_FP | EQUIV | +0.002 [+0.001, +0.002] | -0.017 [-0.033, -0.001] | MATCHED | 0 / 0 (1.000) | — | +0.005 [+0.002, +0.010] | +0.000 | 1.000 | 82.3 |
| qread4q_v4@0.125 | V4 | 0.14 | 1.14 | +0.013 [+0.003, +0.026] | NEAR_FP | EQUIV | +0.004 [+0.001, +0.009] | -0.010 [-0.026, +0.007] | MATCHED | 0 / 0 (1.000) | — | +0.013 [+0.003, +0.026] | +0.000 | 1.000 | 82.3 |
| qread2t4kq_v4@0.125 | V4 | 0.35 | 1.14 | +0.001 [-0.004, +0.006] | NEAR_FP | EQUIV | +0.001 [+0.001, +0.002] | -0.021 [-0.042, -0.001] | MATCHED | 0 / 0 (1.000) | — | +0.001 [-0.004, +0.006] | +0.000 | 1.000 | 82.3 |
| qread2t4kqF_v4@0.125 | V4 | 0.42 | 1.14 | +0.003 [+0.001, +0.006] | NEAR_FP | EQUIV | +0.001 [+0.000, +0.001] | -0.019 [-0.037, -0.001] | MATCHED | 0 / 0 (1.000) | — | +0.003 [+0.001, +0.006] | +0.000 | 1.000 | 82.3 |
| quest4_v4@0.125 | V4 | 0.42 | 1.40 | +0.004 [-0.005, +0.013] | NEAR_FP | EQUIV | +0.005 [+0.002, +0.008] | -0.018 [-0.034, -0.004] | MATCHED | 0 / 0 (1.000) | — | +0.004 [-0.005, +0.013] | +0.000 | 1.000 | 82.8 |
| qread2t4q_v4@0.125 | V4 | 0.55 | 1.14 | +0.003 [-0.002, +0.007] | NEAR_FP | EQUIV | +0.001 [+0.001, +0.002] | -0.020 [-0.041, +0.001] | MATCHED | 0 / 0 (1.000) | — | +0.003 [-0.002, +0.007] | +0.000 | 1.000 | 82.3 |
| uniform+v4@3 | V4 | — | — | +0.022 [+0.004, +0.042] | NEAR_FP | EQUIV | +0.023 [+0.011, +0.037] | (D) | — | 0 / 0 (1.000) | — | +0.022 [+0.004, +0.042] | +0.000 | 1.000 | 82.2 |
| uniform+v4@4 | V4 | 1.14 | 1.14 | +0.004 [-0.004, +0.012] | NEAR_FP | EQUIV | +0.002 [+0.001, +0.004] | -0.019 [-0.033, -0.005] | MATCHED | 0 / 0 (1.000) | — | +0.004 [-0.004, +0.012] | +0.000 | 1.000 | 82.2 |
| kivi4_v4@4 | V4 | 1.15 | 1.15 | -0.006 [-0.013, -0.001] | NEAR_FP | EQUIV | +0.003 [+0.000, +0.006] | -0.029 [-0.050, -0.009] | MATCHED | 0 / 0 (1.000) | — | -0.006 [-0.013, -0.001] | +0.000 | 1.000 | 82.2 |
| kvquant4_v4@4 | V4 | 1.16 | 1.16 | +0.006 [+0.001, +0.012] | NEAR_FP | EQUIV | +0.001 [+0.000, +0.002] | -0.016 [-0.032, +0.000] | MATCHED | 0 / 0 (1.000) | — | +0.006 [+0.001, +0.012] | +0.000 | 1.000 | 82.2 |
| fp8kv@8 | V8 | 2.20 | 2.20 | -0.002 [-0.006, +0.001] | NEAR_FP | EQUIV | +0.001 [+0.000, +0.002] | -0.024 [-0.044, -0.005] | MATCHED | 0 / 0 (1.000) | — | -0.002 [-0.006, +0.001] | +0.000 | 1.000 | 82.2 |

## hmqwen / rag — FP 0.5; 20 units (10 FP-failed, 3 need the context); margin 0.100

| arm@B | accuracy [90%] | Δ vs FP [90%] | Δ on FP-failed | by task |
|---|---|---|---:|---|
| qread4q_v4@0.125 | +0.600 [+0.400, +0.750] | +0.100 [+0.000, +0.200] | 0.2 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.6} |
| qread_v4@0.125 | +0.600 [+0.400, +0.800] | +0.100 [+0.000, +0.200] | 0.2 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.6} |
| qoraclefp_v16@0.125 | +0.550 [+0.350, +0.750] | +0.050 [+0.000, +0.150] | 0.1 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| qread2t4kqF_v4@0.125 | +0.550 [+0.350, +0.750] | +0.050 [+0.000, +0.150] | 0.1 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| qread2t4kq_v4@0.125 | +0.550 [+0.350, +0.750] | +0.050 [+0.000, +0.150] | 0.1 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| qread2t4q_v4@0.125 | +0.550 [+0.350, +0.750] | +0.050 [+0.000, +0.150] | 0.1 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| qread4_v4@0.125 | +0.550 [+0.350, +0.750] | +0.050 [+0.000, +0.150] | 0.1 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| qreadfp_v16@0.125 | +0.550 [+0.350, +0.750] | +0.050 [+0.000, +0.150] | 0.1 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| quest4_v4@0.125 | +0.550 [+0.350, +0.750] | +0.050 [+0.000, +0.150] | 0.1 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| quest_v16@0.125 | +0.550 [+0.350, +0.750] | +0.050 [+0.000, +0.150] | 0.1 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| uniform@3 | +0.550 [+0.350, +0.750] | +0.050 [-0.100, +0.200] | 0.2 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| uniform+v4@3 | +0.550 [+0.350, +0.750] | +0.050 [-0.100, +0.200] | 0.2 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.5} |
| fp@0 | +0.500 [+0.300, +0.700] | +0.000 [+0.000, +0.000] | 0.0 | {'kilt_hotpotqa': 0.5, 'kilt_nq': 0.5} |
| fp8kv@8 | +0.500 [+0.300, +0.700] | +0.000 [+0.000, +0.000] | 0.0 | {'kilt_hotpotqa': 0.5, 'kilt_nq': 0.5} |
| fp_noise@0 | +0.500 [+0.300, +0.700] | +0.000 [+0.000, +0.000] | 0.0 | {'kilt_hotpotqa': 0.5, 'kilt_nq': 0.5} |
| kivi4_v4@4 | +0.500 [+0.300, +0.700] | +0.000 [+0.000, +0.000] | 0.0 | {'kilt_hotpotqa': 0.5, 'kilt_nq': 0.5} |
| kvquant4_v4@4 | +0.500 [+0.300, +0.700] | +0.000 [+0.000, +0.000] | 0.0 | {'kilt_hotpotqa': 0.5, 'kilt_nq': 0.5} |
| uniform+v4@4 | +0.500 [+0.300, +0.700] | +0.000 [-0.100, +0.100] | 0.1 | {'kilt_hotpotqa': 0.6, 'kilt_nq': 0.4} |
| uniform@4 | +0.450 [+0.250, +0.650] | -0.050 [-0.150, +0.000] | 0.0 | {'kilt_hotpotqa': 0.5, 'kilt_nq': 0.4} |
| closedbook@0 | +0.400 [+0.250, +0.600] | -0.100 [-0.250, +0.050] | 0.1 | {'kilt_hotpotqa': 0.4, 'kilt_nq': 0.4} |

**ACC_SYSTEM** (qread2t4kq_v4@0.125 − fp@0): +0.050 [+0.000, +0.150] → ACC_NEAR_FP
**ACC_FLOOR** (qread2t4kqF_v4@0.125 − fp@0): +0.050 [+0.000, +0.150] → ACC_NEAR_FP
**ACC_QUEST** (quest_v16@0.125 − fp@0): +0.050 [+0.000, +0.150] → ACC_NEAR_FP
**SYS_VS_QUEST_ACC** (qread2t4kq_v4@0.125 − quest_v16@0.125): +0.000 [+0.000, +0.000] → SYS_VS_QUEST_ACC_NO_EFFECT, EQUIV
**FLOOR_VS_QUEST_ACC** (qread2t4kqF_v4@0.125 − quest_v16@0.125): +0.000 [+0.000, +0.000] → FLOOR_VS_QUEST_ACC_NO_EFFECT, EQUIV
**SCHEDULE_ACC** (qread4_v4@0.125 − quest4_v4@0.125): +0.000 [+0.000, +0.000] → SCHEDULE_ACC_NO_EFFECT, EQUIV
**SYS_VS_BEST_ACC** (qread2t4kq_v4@0.125 − uniform+v4@4): +0.050 [+0.000, +0.150] → SYS_VS_BEST_ACC_NO_EFFECT, NOT_EQUIV
**SYS_VS_D_ACC** (qread2t4kq_v4@0.125 − uniform+v4@3): +0.000 [-0.100, +0.100] → SYS_VS_D_ACC_NO_EFFECT, NOT_EQUIV
**REQ1_ACC** (qread4q_v4@0.125 − qread4_v4@0.125): +0.050 [+0.000, +0.150] → REQ1_ACC_NO_EFFECT, NOT_EQUIV
**VOTE_LOSS_ACC** (qreadfp_v16@0.125 − qoraclefp_v16@0.125): +0.000 [+0.000, +0.000] → VOTE_LOSS_ACC_NO_EFFECT, EQUIV
**SYS_VS_QUEST** (NLL, qread2t4kq_v4@0.125 − quest_v16@0.125): +0.219 [+0.065, +0.390] → SYS_VS_QUEST_HURTS
**SCHEDULE** (NLL, qread4_v4@0.125 − quest4_v4@0.125): +0.183 [+0.051, +0.320] → SCHEDULE_HURTS
**FLOOR_VS_FIXED** (NLL, qread2t4kqF_v4@0.125 − qread2t4kq_v4@0.125): +0.003 [-0.077, +0.082] → FLOOR_VS_FIXED_NO_EFFECT

Quest read fraction (all layers): {'quest4_v4': 0.125, 'quest_v16': 0.125}; floor r: {'mean': 0.132, 'min': 0.125, 'max': 0.148}

NLL / KL points:

| arm@B | lens | ρ | ρ mem | dP vs FP [90%] | NEAR_FP | EQUIV | KL span [90%] | dP vs D_L [90%] | MATCHED | lost / gained (McNemar p) | lost types | dS vs FP | min bias | score | peak GiB |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| closedbook@0 | V16 | 0.00 | 0.00 | +6.747 [+4.288, +9.441] | FAR_FROM_FP | NOT_EQUIV | +7.228 [+4.744, +9.843] | +6.027 [+3.688, +8.578] | WORSE | 3 / 1 (0.625) | {'other': 3} | +6.747 [+4.288, +9.441] | +0.000 | 0.400 | 81.1 |
| qreadfp_v16@0.125 | V16 | 0.21 | 1.67 | +0.327 [+0.143, +0.539] | FAR_FROM_FP | NOT_EQUIV | +0.308 [+0.158, +0.473] | -0.393 [-0.713, -0.075] | MATCHED | 0 / 1 (1.000) | — | +0.327 [+0.143, +0.539] | +0.000 | 0.550 | 81.0 |
| qoraclefp_v16@0.125 | V16 | 0.21 | 1.67 | +0.144 [+0.053, +0.244] | INCONCLUSIVE | NOT_EQUIV | +0.076 [+0.042, +0.114] | -0.577 [-0.882, -0.302] | MATCHED | 0 / 1 (1.000) | — | +0.144 [+0.053, +0.244] | +0.000 | 0.550 | 81.0 |
| quest_v16@0.125 | V16 | 0.31 | 1.78 | +0.008 [-0.077, +0.090] | NEAR_FP | EQUIV | +0.176 [+0.095, +0.262] | -0.712 [-1.078, -0.383] | MATCHED | 0 / 1 (1.000) | — | +0.008 [-0.077, +0.090] | +0.000 | 0.550 | 81.1 |
| uniform@3 | V16 | — | — | +0.721 [+0.387, +1.083] | FAR_FROM_FP | NOT_EQUIV | +0.689 [+0.427, +0.958] | (D) | — | 1 / 2 (1.000) | {'other': 1} | +0.721 [+0.387, +1.083] | +0.000 | 0.550 | 79.7 |
| uniform@4 | V16 | 1.05 | 1.05 | +0.057 [-0.015, +0.142] | INCONCLUSIVE | NOT_EQUIV | +0.118 [+0.062, +0.179] | -0.663 [-1.004, -0.346] | MATCHED | 1 / 0 (1.000) | {'other': 1} | +0.057 [-0.015, +0.142] | +0.000 | 0.450 | 81.0 |
| fp_noise@0 | V16 | 1.67 | 1.67 | -0.001 [-0.052, +0.052] | NEAR_FP | EQUIV | +0.027 [+0.016, +0.037] | -0.722 [-1.060, -0.402] | MATCHED | 0 / 0 (1.000) | — | -0.001 [-0.052, +0.052] | +0.000 | 0.500 | 81.0 |
| qread_v4@0.125 | V4 | 0.13 | 1.00 | +0.764 [+0.378, +1.195] | FAR_FROM_FP | NOT_EQUIV | +0.873 [+0.511, +1.250] | +0.048 [-0.121, +0.221] | INCONCLUSIVE | 0 / 2 (0.500) | — | +0.764 [+0.378, +1.195] | +0.000 | 0.600 | 85.7 |
| qread4q_v4@0.125 | V4 | 0.15 | 1.14 | +0.450 [+0.259, +0.647] | FAR_FROM_FP | NOT_EQUIV | +0.443 [+0.223, +0.700] | -0.266 [-0.565, +0.032] | MATCHED | 0 / 2 (0.500) | — | +0.450 [+0.259, +0.647] | +0.000 | 0.600 | 85.7 |
| qread4_v4@0.125 | V4 | 0.15 | 1.14 | +0.334 [+0.192, +0.473] | FAR_FROM_FP | NOT_EQUIV | +0.421 [+0.219, +0.649] | -0.383 [-0.765, -0.027] | MATCHED | 0 / 1 (1.000) | — | +0.334 [+0.192, +0.473] | +0.000 | 0.550 | 85.7 |
| qread2t4kq_v4@0.125 | V4 | 0.35 | 1.14 | +0.228 [+0.074, +0.406] | INCONCLUSIVE | NOT_EQUIV | +0.291 [+0.142, +0.466] | -0.489 [-0.854, -0.156] | MATCHED | 0 / 1 (1.000) | — | +0.228 [+0.074, +0.406] | +0.000 | 0.550 | 85.8 |
| qread2t4kqF_v4@0.125 | V4 | 0.37 | 1.14 | +0.230 [+0.119, +0.357] | FAR_FROM_FP | NOT_EQUIV | +0.259 [+0.140, +0.389] | -0.486 [-0.828, -0.169] | MATCHED | 0 / 1 (1.000) | — | +0.230 [+0.119, +0.357] | +0.000 | 0.550 | 85.8 |
| quest4_v4@0.125 | V4 | 0.42 | 1.40 | +0.151 [+0.025, +0.281] | INCONCLUSIVE | NOT_EQUIV | +0.331 [+0.189, +0.479] | -0.566 [-0.983, -0.183] | MATCHED | 0 / 1 (1.000) | — | +0.151 [+0.025, +0.281] | +0.000 | 0.550 | 86.3 |
| qread2t4q_v4@0.125 | V4 | 0.55 | 1.14 | +0.299 [+0.124, +0.507] | FAR_FROM_FP | NOT_EQUIV | +0.312 [+0.156, +0.489] | -0.418 [-0.763, -0.099] | MATCHED | 0 / 1 (1.000) | — | +0.299 [+0.124, +0.507] | +0.000 | 0.550 | 85.8 |
| uniform+v4@3 | V4 | — | — | +0.716 [+0.348, +1.124] | FAR_FROM_FP | NOT_EQUIV | +0.672 [+0.402, +0.959] | (D) | — | 1 / 2 (1.000) | {'other': 1} | +0.716 [+0.348, +1.124] | +0.000 | 0.550 | 85.6 |
| uniform+v4@4 | V4 | 1.14 | 1.14 | +0.049 [-0.008, +0.116] | INCONCLUSIVE | NOT_EQUIV | +0.113 [+0.058, +0.172] | -0.667 [-1.031, -0.335] | MATCHED | 1 / 1 (1.000) | {'other': 1} | +0.049 [-0.008, +0.116] | +0.000 | 0.500 | 85.6 |
| kivi4_v4@4 | V4 | 1.15 | 1.15 | -0.007 [-0.068, +0.055] | NEAR_FP | EQUIV | +0.065 [+0.037, +0.095] | -0.723 [-1.122, -0.362] | MATCHED | 0 / 0 (1.000) | — | -0.007 [-0.068, +0.055] | +0.000 | 0.500 | 85.6 |
| kvquant4_v4@4 | V4 | 1.16 | 1.16 | -0.029 [-0.100, +0.043] | NEAR_FP | EQUIV | +0.108 [+0.068, +0.149] | -0.746 [-1.135, -0.393] | MATCHED | 0 / 0 (1.000) | — | -0.029 [-0.100, +0.043] | +0.000 | 0.500 | 85.6 |
| fp8kv@8 | V8 | 2.20 | 2.20 | +0.110 [+0.017, +0.210] | INCONCLUSIVE | NOT_EQUIV | +0.072 [+0.044, +0.102] | -0.607 [-0.937, -0.304] | MATCHED | 0 / 0 (1.000) | — | +0.110 [+0.017, +0.210] | +0.000 | 0.500 | 85.6 |

## hmqwen / rerank — FP 0.266; 10 units (10 FP-failed, 2 need the context); margin 0.100

| arm@B | accuracy [90%] | Δ vs FP [90%] | Δ on FP-failed | by task |
|---|---|---|---:|---|
| uniform+v4@4 | +0.317 [+0.173, +0.471] | +0.052 [-0.008, +0.159] | 0.052 | {'msmarco_rerank_psg': 0.317} |
| fp_noise@0 | +0.308 [+0.156, +0.466] | +0.042 [-0.028, +0.139] | 0.042 | {'msmarco_rerank_psg': 0.308} |
| quest_v16@0.125 | +0.283 [+0.128, +0.456] | +0.018 [-0.099, +0.145] | 0.018 | {'msmarco_rerank_psg': 0.283} |
| quest4_v4@0.125 | +0.276 [+0.125, +0.447] | +0.011 [-0.085, +0.119] | 0.011 | {'msmarco_rerank_psg': 0.276} |
| fp@0 | +0.266 [+0.127, +0.415] | +0.000 [+0.000, +0.000] | 0.0 | {'msmarco_rerank_psg': 0.266} |
| fp8kv@8 | +0.229 [+0.090, +0.383] | -0.037 [-0.133, +0.024] | -0.037 | {'msmarco_rerank_psg': 0.229} |
| uniform@4 | +0.176 [+0.069, +0.290] | -0.090 [-0.260, +0.000] | -0.09 | {'msmarco_rerank_psg': 0.176} |
| kvquant4_v4@4 | +0.159 [+0.063, +0.267] | -0.107 [-0.276, +0.065] | -0.107 | {'msmarco_rerank_psg': 0.159} |
| qoraclefp_v16@0.125 | +0.137 [+0.046, +0.233] | -0.129 [-0.263, +0.006] | -0.129 | {'msmarco_rerank_psg': 0.137} |
| kivi4_v4@4 | +0.122 [+0.036, +0.217] | -0.144 [-0.274, -0.016] | -0.144 | {'msmarco_rerank_psg': 0.122} |
| uniform@3 | +0.119 [+0.020, +0.218] | -0.147 [-0.331, +0.030] | -0.147 | {'msmarco_rerank_psg': 0.119} |
| uniform+v4@3 | +0.061 [+0.000, +0.142] | -0.205 [-0.364, -0.063] | -0.205 | {'msmarco_rerank_psg': 0.061} |
| qread2t4kq_v4@0.125 | +0.005 [+0.000, +0.015] | -0.261 [-0.410, -0.121] | -0.261 | {'msmarco_rerank_psg': 0.005} |
| qread2t4q_v4@0.125 | +0.005 [+0.000, +0.015] | -0.261 [-0.410, -0.121] | -0.261 | {'msmarco_rerank_psg': 0.005} |
| qread4_v4@0.125 | +0.005 [+0.000, +0.015] | -0.261 [-0.410, -0.121] | -0.261 | {'msmarco_rerank_psg': 0.005} |
| qread4q_v4@0.125 | +0.005 [+0.000, +0.015] | -0.261 [-0.410, -0.121] | -0.261 | {'msmarco_rerank_psg': 0.005} |
| qreadfp_v16@0.125 | +0.005 [+0.000, +0.015] | -0.261 [-0.410, -0.121] | -0.261 | {'msmarco_rerank_psg': 0.005} |
| closedbook@0 | +0.000 [+0.000, +0.000] | -0.266 [-0.415, -0.127] | -0.266 | {'msmarco_rerank_psg': 0.0} |
| qread2t4kqF_v4@0.125 | +0.000 [+0.000, +0.000] | -0.266 [-0.415, -0.127] | -0.266 | {'msmarco_rerank_psg': 0.0} |
| qread_v4@0.125 | +0.000 [+0.000, +0.000] | -0.266 [-0.415, -0.127] | -0.266 | {'msmarco_rerank_psg': 0.0} |

**ACC_SYSTEM** (qread2t4kq_v4@0.125 − fp@0): -0.261 [-0.410, -0.121] → ACC_LOSS
**ACC_FLOOR** (qread2t4kqF_v4@0.125 − fp@0): -0.266 [-0.415, -0.127] → ACC_LOSS
**ACC_QUEST** (quest_v16@0.125 − fp@0): +0.018 [-0.099, +0.145] → ACC_INCONCLUSIVE
**SYS_VS_QUEST_ACC** (qread2t4kq_v4@0.125 − quest_v16@0.125): -0.279 [-0.453, -0.123] → SYS_VS_QUEST_ACC_HURTS, NOT_EQUIV
**FLOOR_VS_QUEST_ACC** (qread2t4kqF_v4@0.125 − quest_v16@0.125): -0.283 [-0.456, -0.128] → FLOOR_VS_QUEST_ACC_HURTS, NOT_EQUIV
**SCHEDULE_ACC** (qread4_v4@0.125 − quest4_v4@0.125): -0.272 [-0.443, -0.122] → SCHEDULE_ACC_HURTS, NOT_EQUIV
**SYS_VS_BEST_ACC** (qread2t4kq_v4@0.125 − uniform+v4@4): -0.313 [-0.467, -0.166] → SYS_VS_BEST_ACC_HURTS, NOT_EQUIV
**SYS_VS_D_ACC** (qread2t4kq_v4@0.125 − uniform+v4@3): -0.056 [-0.128, +0.000] → SYS_VS_D_ACC_NO_EFFECT, NOT_EQUIV
**REQ1_ACC** (qread4q_v4@0.125 − qread4_v4@0.125): +0.000 [+0.000, +0.000] → REQ1_ACC_NO_EFFECT, EQUIV
**VOTE_LOSS_ACC** (qreadfp_v16@0.125 − qoraclefp_v16@0.125): -0.132 [-0.225, -0.044] → VOTE_LOSS_ACC_HURTS, NOT_EQUIV
**SYS_VS_QUEST** (NLL, qread2t4kq_v4@0.125 − quest_v16@0.125): +36.812 [+19.448, +56.924] → SYS_VS_QUEST_HURTS
**SCHEDULE** (NLL, qread4_v4@0.125 − quest4_v4@0.125): +37.224 [+19.296, +57.754] → SCHEDULE_HURTS
**FLOOR_VS_FIXED** (NLL, qread2t4kqF_v4@0.125 − qread2t4kq_v4@0.125): -0.863 [-1.976, +0.126] → FLOOR_VS_FIXED_NO_EFFECT

Quest read fraction (all layers): {'quest4_v4': 0.125, 'quest_v16': 0.125}; floor r: {'mean': 0.177, 'min': 0.158, 'max': 0.205}

NLL / KL points:

| arm@B | lens | ρ | ρ mem | dP vs FP [90%] | NEAR_FP | EQUIV | KL span [90%] | dP vs D_L [90%] | MATCHED | lost / gained (McNemar p) | lost types | dS vs FP | min bias | score | peak GiB |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| closedbook@0 | V16 | 0.00 | 0.00 | +62.757 [+37.618, +90.997] | FAR_FROM_FP | NOT_EQUIV | +75.829 [+50.766, +104.244] | +59.875 [+35.368, +87.374] | WORSE | 6 / 0 (0.031) | {'other': 6} | +62.757 [+37.618, +90.997] | +0.000 | 0.000 | 75.5 |
| qreadfp_v16@0.125 | V16 | 0.21 | 1.67 | +40.588 [+22.925, +61.181] | FAR_FROM_FP | NOT_EQUIV | +47.227 [+29.574, +68.297] | +37.706 [+21.365, +56.808] | WORSE | 6 / 0 (0.031) | {'other': 5, 'incomplete': 1} | +40.588 [+22.925, +61.181] | +0.000 | 0.005 | 75.4 |
| qoraclefp_v16@0.125 | V16 | 0.21 | 1.67 | +0.769 [-0.027, +1.633] | INCONCLUSIVE | NOT_EQUIV | +0.544 [+0.339, +0.779] | -2.114 [-3.504, -0.866] | MATCHED | 2 / 2 (1.000) | {'other': 2} | +0.769 [-0.027, +1.633] | +0.000 | 0.137 | 75.3 |
| quest_v16@0.125 | V16 | 0.32 | 1.78 | +1.052 [+0.282, +1.831] | FAR_FROM_FP | NOT_EQUIV | +1.588 [+1.103, +2.098] | -1.831 [-2.895, -0.779] | MATCHED | 3 / 3 (1.000) | {'other': 2, 'incomplete': 1} | +1.052 [+0.282, +1.831] | +0.000 | 0.283 | 75.0 |
| uniform@3 | V16 | — | — | +2.883 [+1.316, +4.505] | FAR_FROM_FP | NOT_EQUIV | +4.578 [+3.005, +6.234] | (D) | — | 4 / 2 (0.688) | {'incomplete': 1, 'other': 3} | +2.883 [+1.316, +4.505] | +0.000 | 0.119 | 75.2 |
| uniform@4 | V16 | 1.05 | 1.05 | +0.013 [-0.381, +0.408] | INCONCLUSIVE | NOT_EQUIV | +0.768 [+0.486, +1.085] | -2.870 [-4.647, -1.204] | MATCHED | 3 / 1 (0.625) | {'other': 2, 'incomplete': 1} | +0.013 [-0.381, +0.408] | +0.000 | 0.176 | 75.3 |
| fp_noise@0 | V16 | 1.67 | 1.67 | -0.078 [-0.356, +0.179] | INCONCLUSIVE | NOT_EQUIV | +0.156 [+0.113, +0.200] | -2.961 [-4.476, -1.497] | MATCHED | 1 / 2 (1.000) | {'incomplete': 1} | -0.078 [-0.356, +0.179] | +0.000 | 0.308 | 75.0 |
| qread_v4@0.125 | V4 | 0.13 | 1.00 | +41.709 [+24.006, +62.341] | FAR_FROM_FP | NOT_EQUIV | +49.514 [+31.909, +70.226] | +38.385 [+21.997, +57.531] | WORSE | 6 / 0 (0.031) | {'other': 6} | +41.709 [+24.006, +62.341] | +0.000 | 0.000 | 79.8 |
| qread4_v4@0.125 | V4 | 0.15 | 1.14 | +38.244 [+20.043, +59.184] | FAR_FROM_FP | NOT_EQUIV | +45.690 [+27.573, +67.126] | +34.920 [+18.201, +54.494] | WORSE | 6 / 0 (0.031) | {'other': 4, 'incomplete': 2} | +38.244 [+20.043, +59.184] | +0.000 | 0.005 | 79.8 |
| qread4q_v4@0.125 | V4 | 0.15 | 1.14 | +38.229 [+20.096, +59.105] | FAR_FROM_FP | NOT_EQUIV | +45.453 [+27.369, +66.918] | +34.905 [+18.247, +54.280] | WORSE | 6 / 0 (0.031) | {'other': 6} | +38.229 [+20.096, +59.105] | +0.000 | 0.005 | 79.7 |
| qread2t4kq_v4@0.125 | V4 | 0.35 | 1.14 | +37.864 [+19.988, +58.614] | FAR_FROM_FP | NOT_EQUIV | +44.988 [+27.068, +66.258] | +34.540 [+18.035, +53.848] | WORSE | 6 / 0 (0.031) | {'other': 5, 'incomplete': 1} | +37.864 [+19.988, +58.614] | +0.000 | 0.005 | 79.8 |
| quest4_v4@0.125 | V4 | 0.42 | 1.39 | +1.021 [+0.278, +1.742] | FAR_FROM_FP | NOT_EQUIV | +2.440 [+1.636, +3.287] | -2.303 [-3.839, -0.942] | INCONCLUSIVE | 2 / 3 (1.000) | {'other': 2} | +1.021 [+0.278, +1.742] | +0.000 | 0.276 | 78.9 |
| qread2t4kqF_v4@0.125 | V4 | 0.49 | 1.14 | +37.001 [+19.046, +57.760] | FAR_FROM_FP | NOT_EQUIV | +43.385 [+25.593, +64.430] | +33.677 [+17.082, +53.115] | WORSE | 6 / 0 (0.031) | {'other': 4, 'incomplete': 2} | +37.001 [+19.046, +57.760] | +0.000 | 0.000 | 79.8 |
| qread2t4q_v4@0.125 | V4 | 0.55 | 1.14 | +38.097 [+20.189, +58.854] | FAR_FROM_FP | NOT_EQUIV | +45.030 [+27.064, +66.312] | +34.773 [+18.217, +54.145] | WORSE | 6 / 0 (0.031) | {'other': 5, 'incomplete': 1} | +38.097 [+20.189, +58.854] | +0.000 | 0.005 | 79.8 |
| uniform+v4@3 | V4 | — | — | +3.324 [+1.655, +5.090] | FAR_FROM_FP | NOT_EQUIV | +4.871 [+3.230, +6.600] | (D) | — | 4 / 1 (0.375) | {'incomplete': 1, 'other': 3} | +3.324 [+1.655, +5.090] | +0.000 | 0.061 | 79.6 |
| uniform+v4@4 | V4 | 1.14 | 1.14 | -0.178 [-0.676, +0.305] | INCONCLUSIVE | NOT_EQUIV | +0.803 [+0.517, +1.127] | -3.502 [-5.479, -1.734] | MATCHED | 2 / 2 (1.000) | {'other': 1, 'incomplete': 1} | -0.178 [-0.676, +0.305] | +0.000 | 0.317 | 79.6 |
| kivi4_v4@4 | V4 | 1.15 | 1.15 | +0.632 [+0.182, +1.090] | FAR_FROM_FP | NOT_EQUIV | +0.766 [+0.515, +1.040] | -2.692 [-4.149, -1.339] | MATCHED | 4 / 0 (0.125) | {'incomplete': 3, 'other': 1} | +0.632 [+0.182, +1.090] | +0.000 | 0.122 | 79.6 |
| kvquant4_v4@4 | V4 | 1.16 | 1.16 | +0.154 [-0.422, +0.698] | INCONCLUSIVE | NOT_EQUIV | +1.237 [+0.917, +1.597] | -3.170 [-4.757, -1.775] | MATCHED | 4 / 1 (0.375) | {'incomplete': 2, 'other': 2} | +0.154 [-0.422, +0.698] | +0.000 | 0.159 | 79.6 |
| fp8kv@8 | V8 | 2.20 | 2.20 | +0.551 [+0.101, +1.051] | FAR_FROM_FP | NOT_EQUIV | +0.389 [+0.228, +0.569] | -2.773 [-4.182, -1.453] | MATCHED | 2 / 1 (1.000) | {'incomplete': 1, 'other': 1} | +0.551 [+0.101, +1.051] | +0.000 | 0.229 | 79.6 |

