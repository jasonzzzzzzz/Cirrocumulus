# R7b, the exploratory arms (E1-E3; reported, not tested)

- every block valid

## Llama-3.1-8B, 128K

- E2 (4+3 / 4+2): 1.338 (0.923–1.733), n 48
- E3 (water-filled / uniform 4+1, equal traffic): 0.367 (0.174–0.902)
- E1 level t1: KL / design 0.625, traffic 0.4007, rows 0.1282, fallback 0.0000, misses 0 of 1338368 query-head checks (COVER_OK)
- E1 level t2: KL / design 0.894, traffic 0.4086, rows 0.1397, fallback 0.0000, misses 0 of 1338368 query-head checks (COVER_OK)

## Qwen3-30B-A3B, 32K

- E2 (4+3 / 4+2): 0.532 (0.254–1.113), n 30
- E3 (water-filled / uniform 4+1, equal traffic): 1.164 (0.755–1.725)
- E1 level t1: KL / design 0.731, traffic 0.4166, rows 0.1282, fallback 0.0000, misses 0 of 1503744 query-head checks (COVER_OK)
- E1 level t2: KL / design 0.739, traffic 0.4246, rows 0.1400, fallback 0.0002, misses 0 of 1503744 query-head checks (COVER_OK)
