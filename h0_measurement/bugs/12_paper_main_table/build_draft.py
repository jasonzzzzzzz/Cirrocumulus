#!/usr/bin/env python3
"""paper_draft.src.md + the reader's appendix grid -> paper_draft.md.

    python build_draft.py [tables.md]     (default: tables_r9_existing.md)
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
tables = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "tables_r9_existing.md")
txt = open(tables).read()
grid = txt[txt.index("## Appendix"):].split("\n", 1)[1].strip()
src = open(os.path.join(HERE, "paper_draft.src.md")).read()
with open(os.path.join(HERE, "paper_draft.md"), "w") as fh:
    fh.write(src.replace("<<GRID>>", grid))
print("wrote paper_draft.md")
