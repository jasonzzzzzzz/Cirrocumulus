# V7 source-capacity audit

This is a label-blind CPU audit for a possible fresh LongBench-v2 V7. It uses
the pinned Qwen3-30B tokenizer, no-thinking chat template, official zero-shot
prompt, and five-token forced-choice scaffold. A row is complete at context cap
`C` exactly when `input_tokens + 5 <= C`.

The 117 IDs already assigned to the V4--V6 qualification, development, and
untouched confirmation partitions are unavailable as fresh V7 data. The final
column also removes any otherwise unused row in a connected component that
contains one of those 117 IDs. Components are the frozen connected components
of equal stripped-context SHA-256 or equal stripped-question SHA-256.

| context cap | all complete rows/components | unused rows/components | old-linked rows removed | leakage-clean rows/components |
|---:|---:|---:|---:|---:|
| 40,960 | 133/122 | 16/16 | 2 | 14/14 |
| 49,152 | 158/146 | 41/40 | 2 | 39/38 |
| 65,536 | 184/171 | 67/65 | 2 | 65/63 |
| 81,920 | 213/197 | 96/92 | 3 | 93/89 |
| 98,304 | 249/219 | 132/114 | 3 | 129/111 |
| 131,072 | 304/268 | 187/163 | 3 | 184/160 |

At 131,072, the 160 leakage-clean components comprise 152 singletons and eight
linked components of sizes 6, 6, 4, 4, 4, 3, 3, and 2.

## Truncation sentinel

The largest eligible prompt has 131,067 tokens because the scaffold consumes
five positions. Tokenization therefore truncates at 131,068, one token beyond
eligibility. A returned length below 131,068 is the exact full length. A
returned length equal to 131,068 means the true length is at least 131,068 and
the row cannot satisfy `input_tokens + 5 <= 131,072`. The audit observed 199
rows at this sentinel. Excluding them is exact for every cap in the table; it
does not discard an ambiguous eligible row.

The normalized 503-row length table has SHA-256
`a668a476f09dbb151af09a0228f4d2c74a0cfede0292262d0b2703a1571e2517`.

## Reproduction

Run from the repository root:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python \
  h0_measurement/bugs/9_sota_eviction_baselines/audit_v7_source.py
```

The script authenticates the dataset, original source manifest, frozen Qwen
manifest, tokenizer metadata, chat template, forced-choice token contract, and
local model-snapshot inventory before counting. It renders all 503 prompts with
truncation only at the ineligibility sentinel. It never reads an answer to
select, filter, group, or count a row.

For the table above, the exact tokenization output was retained temporarily as
`/tmp/v7_qwen_lengths.json`. The following command replays only the counting
and component logic from that cache after authenticating all pinned inputs:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python \
  h0_measurement/bugs/9_sota_eviction_baselines/audit_v7_source.py \
  --legacy-length-cache /tmp/v7_qwen_lengths.json
```

At 65,536, the clean pool has only 63 components. It can support a bounded
qualification plus development split, but it cannot also provide a well-sized
fresh confirmation split. The existing 45 confirmation rows remain untouched.
