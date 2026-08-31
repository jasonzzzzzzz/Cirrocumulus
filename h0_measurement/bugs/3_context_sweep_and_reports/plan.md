

Fix and effort — three separable pieces:

- Matched-ctx runs + ctx sweep: pure config — --override ctx=32768 in slurm, or add registry entries in models.yaml. S (an hour). GPU cost is the real line item, but 32k runs are ~4× cheaper per prompt than 128k.

- Reporting: φ-binned in-band panel and "in band at any budget" in report.py — doable today on existing data (n95, L, per-budget columns are all stored), and worth doing before the rerun as a dry run of the analysis. S–M (half a day).

- [defered, ask me next time] Sink-excluded φ: n95_nosink in sensitivity_metrics (drop top-k or position-0 mass before the cumsum) — ~6 lines in alloc.py, but needs the rerun to populate. S. A more radical option — redefining the band around absolute support rather than fractional budget — I'd defer; it changes the question H0 asks and should be a v3 decision made after seeing the ctx sweep.