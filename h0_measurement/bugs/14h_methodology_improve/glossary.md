# Glossary (R14 Stages 1c–1h)

## References and units
- **FP** — the uncompressed model: BF16 keys and values. Every metric is paired against
  FP in the same process.
- **D** — dense TurboQuant-3 keys (`uniform@3`). **D_V4** — D with 4-bit values
  (`uniform+v4@3`), the comparator for the MATCHED rule.
- **Unit** — one prompt × task. CIs are 90% bootstrap intervals, clustered by prompt.
- **r** — read fraction: the answer reads floor(r·C) rows per KV head. **C** is the
  context length in tokens.
- **ρ** — bytes read per decode step relative to D in the same value lens.
  **ρ GPU mem** — stored GPU bytes relative to D.
- **Lens V16 / V4** — the value precision an arm reads (exact, or 4-bit TurboQuant-MSE).

## Arms (name@B; B is a width for dense arms and r for reads)
| arm | meaning |
|---|---|
| `fp` | uncompressed |
| `fp+v4` | exact keys, 4-bit values |
| `uniform@b` (`+v4`) | dense TurboQuant-b keys (and 4-bit values) |
| `qread_v{v}@r` | question-time reads over the 3-bit store: the question's SnapKV vote picks floor(rC) rows; the answer reads only those |
| `qreadp_v4@r` | the same, with critical heads read in full |
| `qreadfp_v16@r` | question-time reads over the exact store |
| `qread4_v4@r` | the **simple design**: reads over a 4-bit store |
| `qread2t[4][k][8][q]_v{v}@r` | two-tier reads. The vote runs over tier 1 (3-bit keys, or 4-bit with `4`). The selected rows come from tier 2 (exact, or FP8 with `8`): keys and values, or keys only with `k`. `q` runs the question again over the fetched rows. |
| `qread2t4kq_v4` | the Stage 1g **system**: 4-bit tier, exact keys only, the question again |
| `qread2t4q_v4` | the system fetching exact keys **and** values |
| `fp_noise` | (1h) FP with the context re-prefilled at half the chunk size: the noise floor |
| `fp8kv` | (1h) dense FP8 keys and values, one scale per layer and KV head |
| `kivi4_v4` | (1h) KIVI keys (per-channel, G = 128) at 4 bits, 4-bit values |
| `kvquant4_v4` | (1h) KVQuant-style keys (pre-RoPE non-uniform, 1% outliers) at 4 bits, 4-bit values |
| `qoraclefp_v16@r`, `qoracle4_v4@r` | (1h) oracle selection: the rows FP attends to most while answering, over the exact or 4-bit store |
| `quest_v16@r`, `quest4_v4@r` | (R4) Quest: per-step selection of 16-row pages by min/max key bounds, over the exact / 4-bit store; first two layers dense; the rows read over all layers equal r |
| `qread2t4kqF_v4` | (R4) the system at its per-item read floor, r = max(1/8, 16384 / C) |
| `closedbook` | (R4b) FP on the prompt without the document (or passages and demonstrations): which units need the context |
| `router_*_calib@B` | calibrated quant-and-evict routers (Stages 1b–1f); `nest` = nested dense sets |
| `snapq_v{v}@r` | SnapKV-with-question: the first question's selection made permanent (reuse runs) |

## Tasks
- **`niah_single`, `niah_multikey`, `niah_multivalue`, `vt`** — the RULER retrieval tasks
  (`sievelib/tasks_ruler.py`). R3a raises their difficulty (n_keys, n_values, n_hops) and adds
  **`mk_panel`**, 48 near-duplicate keys.
- **`cwe`, `fwe`** — (R3b) RULER's common- and frequent-word extraction: the answer is a
  count over the whole context. Difficulty: `freq_cw` (repeats of the 10 common words; RULER
  30) and `alpha` (the Zipf exponent; RULER 2.0). `tasks_s1h.py`.
- **`nolima`** — (R3b) NoLiMa one-hop: the question shares no word with the needle
  ("Which character has been to Dresden?" for "Yuki lives next to the Semper Opera House").
  **`nolima_direct`** — the same prompt with the question in the needle's own words.

- **`lbv2`** — (R4) LongBench v2, multiple choice, scored by forced choice (`fc_correct`: the argmax of the teacher-forced A/B/C/D log-probabilities).
- **`kilt_nq`, `kilt_hotpotqa`, `msmarco_rerank_psg`, `icl_trec_coarse`, `icl_banking77`** — (R4b) HELMET at 128K: RAG (substring exact match), re-ranking (NDCG@10), many-shot ICL with random-integer labels (exact match).

## Metrics (per unit, minus FP's value in the same process)
- **`a_sum_nll`** (dA) — teacher-forced NLL of FP's answer-value tokens. This was the
  Stage 1d–1e primary. It is order-sensitive and blind to truncation.
- **`a_span_nll`** — NLL of every token from FP's first answer-value token to the end
  of the answer span. **Primary from Stage 1h on.**
- **`s_set_nll`** (dS, A2) — the minimum of `a_span_nll` and the same NLL with FP's
  answer rewritten into the arm's order. This was Stage 1f–1g's primary. The minimum
  biases arms near FP by about −0.02 nats.
- **`kl_span`**, `kl_all`, `kl_val`, `kl_span_max`, `tf_kl` — (1h) KL(FP ‖ arm) of the
  next-token distributions under teacher forcing. Co-primary from Stage 1h.
- **score** — free-running greedy task score (RULER's).
- **Lost answer** — the arm scores below FP on a unit (Stages 1f–1g's "below FP").
- **Below FP: X / D** — lost answers of arm X and of D.

## Labels
- **MATCHED** (vs D_V4) — hi90(mean dS − dS_D) ≤ 0.10 and hi90(Δ share of > 2-nat
  units) ≤ 0.05.
- **NEAR_FP** — the same rule against FP. From R2 the margin is `m_FP`, which R1
  measures (`plan.md` §5.3).
- **EQUIV_FP** — (1h) the 90% CI lies inside ±m_FP.
- **X_HELPS / X_HURTS / X_NO_EFFECT** — |mean| ≥ 0.05 and a CI excluding 0. These are
  Stage 1g's design factors:
  - QPASS3/4 — the question pass over a 3-/4-bit tier;
  - TIER4 — the 4-bit tier;
  - STORE4 — the 4-bit store;
  - KONLY / KONLY8 — a keys-only tier 2, exact or FP8;
  - REQ — the question run again;
  - FLOOR — the read floor.
- **CONFUSION FIXED / NOT_FIXED** — on the key-confusion prompts, within 2 nats of FP
  wherever D is not.
- **VOTE_LOSS_*** — (1h) the vote's selection minus the oracle's selection.
- **BRIDGE_OK / DRIFT** — (1h) arms rerun on Stage 1g's prompts agree within ±0.05 nats.
- **HEADROOM / CEILING_REMAINS / TOO_HARD / CLOSEST** — (R3a, R3b ladders) how a task's
  difficulty level was chosen from FP's score. In R3b a task **ENTERS** the main cell only if
  FP scores ≥ 0.5 at its chosen level.
- **ACC_*** — (R3a, R3b) the accuracy labels: ACC_NEAR_FP if the system's accuracy interval
  stays above −0.03; the `_ACC` effect labels use ±0.02.
- **LEX_VOTE / LEX_SYS** — (R3b) the vote loss, or the system's loss, on `nolima` minus on
  `nolima_direct`: _HURTS = the read loses more when the question shares no word with the
  needle.
- **SYS_VS_QUEST / FLOOR_VS_QUEST / SCHEDULE / FLOOR_VS_FIXED** — (R4) the system or the floor system against Quest; the once-per-question vote against Quest's per-step pages over the same 4-bit store; the floor against a fixed 1/8.
- **`r8list`** — (R3a2, R3b, R4b) the stop rule for both models (`stops_s1h.py`): no stop before a line with content, a single list item does not stop the answer, anything else stops at its first line.
- **m_cell** — (R3a2 on) the NEAR_FP / EQUIV margin of a cell (and family): FP8 KV's own cost there, clip(hi90 mean dP, 0.05, 0.10). R1's m_FP is reported beside it.
- **Vote span** — (R4b) on LongBench v2 the question-time vote observes 32 rows spread over the question and its choices, not the question's last 32 rows (the format line and the chat template's tail).
- **CTX units** — (R4b) units where FP's accuracy beats the closed-book arm's by ≥ 0.5; the `_CTX` labels.
