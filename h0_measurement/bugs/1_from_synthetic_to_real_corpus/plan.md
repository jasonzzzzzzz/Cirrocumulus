

# Contexts

This is for H0 measurement

- /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/run_h0.py 
- /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/h0_measurement/report.py
- /scratch/jczhao20/ondemand/Cirrocumulus/contexts/unified-kv-quant-evict-TurboQuant/sievelib/prompts.py

# Implementation plan

The mechanism already exists (prompts.py:25-46 honors H0_CORPUS). Work needed:

- Stage a corpus (PG-19 books + concatenated code covers both regimes) — a prefetch_corpus.py sibling to the existing prefetch.py, since compute nodes are offline. ~50 lines, half a day including download. At 128k ctx the builder wants ~600KB of text per prompt, so a handful of books suffices, but with only 6 prompts you should also add a per-prompt file/offset rotation in _corpus_text (currently only the file order is shuffled, so different prompts can land on near-identical text) 

- Make the kill-switch a real gate: in run_h0.py, refuse to run tier: main/large with synthetic=True unless overridden; in report.py, print the niah-vs-cont ladder comparison per model and stamp the verdict UNKNOWN if it fails.