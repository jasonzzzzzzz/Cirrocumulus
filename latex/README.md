# Draft 

## Version: Sept. 18

I've written the 4-page draft in main.tex. The body ends at the bottom of page 4, references start on page 5, and page 6 has an outline of the planned appendix. It compiles with no undefined references.

How the draft argues

Intro: quantization and eviction are two ends of one problem, so the real question is when each one wins, not which is better.

§2 Framework: once you measure error on the attention output, the best bit-width per token is b* = log₂aᵢ + c, and eviction is just the 0-bit case, with a cost c₀=1 that falls out of the maths. The same inequality says when a precision tier is "dead" (costs more than evicting), and the share of heads with a dead 2-bit tier becomes the regime variable.

§3 Measurements: three findings.
- The dead-tier fraction orders the gain across all 24 configurations.
- Context length moves a model from one regime to another.
- The cost model predicts the measured gains to within 0.79–1.34×.

§4 SIEVE: the per-head router, plus a table of six measured facts and the design decision each one forces.

§5: related work and limitations.
Figures (script: make_paper_figs.py)

- Fig 1(a): the three optimal bit profiles (flat, staircase, keep/evict). It uses the real water-filling rule on synthetic logits, with one assumed cost constant chosen so the 2-bit tier dies around τ≈2.3.
- Fig 1(b) and 2(a–b): real data. I recomputed them from the E2 campaign parquet files; the per-configuration numbers are saved in h0_configs.csv.
- Fig 1(c) and 2(c): projected, with illustrative numbers I made up. They are watermarked "PROJECTED", drawn with dashed axes, and marked TBD in the captions: end-task accuracy against context, and the router on/off comparison.
- 
Decisions you should check

- Headline numbers use the comparison where both sides score tokens with the same information, not E2's. The R3 report found E2's mismatched setup (real evictor, but an allocator with perfect information) inflated the band by 12–19 points. It also found the two matched setups agree to within about 5 points. So the draft's numbers are lower than the proposal's (for example, Llama-3.3-70B at 128k is 18.1% rather than 25.0%), and it drops the "no configuration is STOP" claim. Recomputed on this basis: ρ = −0.963 pooled, −0.944 with model identity removed, and −1.00 within every model.
- The τ/ln2 result is not presented as evidence. The draft calls it an algebraic identity and moves it to the appendix; the 0.79–1.34× prediction takes its place.
- The lag-cost finding (1.4–1.6× on in-band heads) is only a TBD. R3 is marked provisional and the llama31-8b re-run (job934606) is still running, so it isn't stated as a result.
- RDKV is left out, as you asked.

TBD anchors. Each is orange and tagged with its roadmap number so it maps to ROADMAP.md:

- R3: the fully deployable matched comparison
- R4: whether the 128k drop comes from attention or from the RoPE window limit
- R5: how often to re-budget during decoding
- R6: where exactly the regime boundary sits
- R7: error bars
- R8: RULER router on/off
- R9: per-head K* budget
- R11/12/14: systems work (nested-code overhead, GQA, kernel speed)
- R15: related-work sweep
- the abstract's end-task sentence
- The bibliography is in a new sieve.bib (the template's bib is untouched). I wrote the entries from memory, so check each one before submission.

To rebuild: cd latex/figures && ../../.venv/bin/python make_paper_figs.py, then cd .. && latexmk -pdf main.tex.
