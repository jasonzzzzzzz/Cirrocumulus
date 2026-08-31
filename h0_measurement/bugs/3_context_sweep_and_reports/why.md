
# φ-window couples ctx into the verdict (characterize, don't "fix")

Why third, and why it's different in kind. Part of this is a real property of the tradeoff, not an artifact: eviction keeps a fixed fraction B/maxb of tokens, so if a head's support is absolute (~constant token count), eviction genuinely does get relatively better as ctx grows. The artifact is only in the comparison: attributing differences between a 32k run and a 128k run to model size. So the fix is experimental design + reporting, and the right goal is to measure the ctx-slope, not to define it away. It ranks third because within a (model, ctx) cell the verdict is still meaningful once P1/P2 land — only the cross-model table is corrupted.

How results would change if fixed.

- A matched-32k cross-model table: sharp models' low-φ heads slide right toward the band window at shorter ctx, so qwen3's numbers rise relative to their 128k values; near-uniform heads stay pinned at φ≈1 regardless, so llama-70B's diffuse penalty persists. Expect the size trend to attenuate but the 70B-diffuseness component to survive — that split (mechanical vs real) is itself the insight.
- The llama31-8b 32k/64k/128k sweep gives you the band-fraction-vs-ctx slope directly — this becomes a figure, and arguably a contribution ("the quantize/evict crossover moves with context length" is a claim TurboQuant-adjacent work doesn't make).
- Sink-excluded φ will reshape the Qwen3 histograms dramatically; per your analysis, most of their sharp mass is one sink token.