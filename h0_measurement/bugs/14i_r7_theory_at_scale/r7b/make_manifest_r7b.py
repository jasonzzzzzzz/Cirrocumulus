#!/usr/bin/env python3
"""R7b's HELMET item manifest, data/manifest_r7b.json (tasks_r7b.py): R6's manifest (R4's rules, tasks_s1h4.py's docstring)
extended in the same order. Tokenizers only, no model, no labels. A copy of ../../14h_methodology_improve/make_r6_manifest.py
(read-only) with R7b's item counts: N_ITEMS per task (tasks_r7b.N_ITEMS), its first 41 items checked equal to R6's for
both models, the RAG / re-rank records written to data/helmet_items/<task>.jsonl.

    .venv/bin/python -u h0_measurement/bugs/14i_r7_theory_at_scale/r7b/make_manifest_r7b.py

prints the sha256 of the manifest and of every record file, to pin in tasks_r7b.py (MANIFEST_SHA256, HM_SHA256_R7B).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(os.path.dirname(HERE)), "14h_methodology_improve")
for _p in (HERE, METH):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import tasks_s1h4 as T  # noqa: E402
import make_r4_manifest as MK  # noqa: E402
import tasks_s1h6 as T6  # noqa: E402
import tasks_r7b as T7B  # noqa: E402

CTX = 131072


def main():
    from transformers import AutoTokenizer
    toks = {m: AutoTokenizer.from_pretrained(i) for m, i in T.MODEL_IDS.items()}
    if T._sha(T6.MANIFEST) != T6.MANIFEST_SHA256:
        raise SystemExit(f"{T6.MANIFEST}: not R6's pinned manifest")
    r6 = json.load(open(T6.MANIFEST))
    keys, pins = {}, {}
    os.makedirs(T7B.ITEMS_DIR, exist_ok=True)
    for task in T.TASKS_HM:
        n = T7B.N_ITEMS[task]
        if T.FAMILY[task] == "icl":
            keys[task] = [T.icl_item_id(s) for s in T.icl_test(task)][:n]
        else:
            test = os.path.join(T.HM_DIR, T.HM_FILES[task][0])
            if T._sha(test) != T.HM_TEST_SHA256[T.HM_FILES[task][0]]:
                raise SystemExit(f"{test}: not the pinned HELMET test file")
            offs = MK.key_offsets(task, test)
            keys[task] = T.salted_order(task, offs)[:n]
            path = os.path.join(T7B.ITEMS_DIR, f"{task}.jsonl")
            with open(path, "w", encoding="utf-8") as fh, open(test, "rb") as srcf:
                for k in keys[task]:
                    rows = offs[k]
                    srcf.seek(rows[T.depth_row(task, k, len(rows))])
                    rec = json.loads(srcf.readline())
                    if str(T._key(task, rec)) != k:
                        raise SystemExit(f"{task}: offset of {k!r} reads another record")
                    fh.write(json.dumps(rec) + "\n")
            pins[f"{T7B.HM_ITEMS}/{task}.jsonl"] = T._sha(path)
        if len(keys[task]) < n:
            raise SystemExit(f"{task}: only {len(keys[task])} items, {n} wanted")
        for m in T.MODEL_IDS:
            old = [x[0] for x in r6["cells"][T.cell_key("helmet", m, CTX)][task]]
            if len(old) != T7B.R6_N_ITEMS or keys[task][:len(old)] != old:
                raise SystemExit(f"{task}: the first {len(old)} items differ from R6's manifest ({m})")
        print(f"{task}: {len(keys[task])} items; the first {T7B.R6_N_ITEMS} are R6's", flush=True)
    T7B.install(pins=pins, check_manifest=False)          # read the new records (not yet pinned) for rendering
    cells = {}
    try:
        for m, tok in toks.items():
            per = {}
            for task in T.TASKS_HM:
                out = []
                for k in keys[task]:
                    r = T.render(tok, task, T.item_record(task, k))
                    out.append([k, T.n_tokens(tok, r), r["truncated_tokens"]])
                per[task] = out
                print(f"helmet {m} {task}: {len(out)} items, tokens {min(x[1] for x in out)}-{max(x[1] for x in out)}, "
                      f"cut {sum(x[2] > 0 for x in out)}", flush=True)
            cells[T.cell_key("helmet", m, CTX)] = per
    finally:
        T7B.uninstall()
    for m in T.MODEL_IDS:                                  # R6's 41 items render exactly as R6's manifest recorded them
        ck = T.cell_key("helmet", m, CTX)
        for task in T.TASKS_HM:
            if cells[ck][task][:T7B.R6_N_ITEMS] != r6["cells"][ck][task]:
                raise SystemExit(f"{m} {task}: R6's items render differently than R6's manifest recorded")
    man = dict(version=T7B.GEN_VERSION, rules="tasks_s1h4.py docstring (R4's), with N_ITEMS per task: tasks_r7b.py",
               base_manifest=dict(path="h0_measurement/bugs/14h_methodology_improve/data/r6/manifest_r6.json",
                                  sha256=T6.MANIFEST_SHA256, first_items_equal=T7B.R6_N_ITEMS),
               helmet_sources=T.HM_SOURCES, helmet_test_sha256=T.HM_TEST_SHA256, data_sha256=T.DATA_SHA256,
               model_ids=T.MODEL_IDS, salt=T.SALT.decode("latin-1"), n_items=T7B.N_ITEMS,
               fresh_items=[T7B.FRESH_ITEMS.start, T7B.FRESH_ITEMS.stop], cells=cells)
    os.makedirs(os.path.dirname(T7B.MANIFEST), exist_ok=True)
    with open(T7B.MANIFEST, "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    print("HM_SHA256_R7B = {")
    for k, v in pins.items():
        print(f'    "{k}": "{v}",')
    print("}")
    print(f"MANIFEST_SHA256 = \"{hashlib.sha256(open(T7B.MANIFEST, 'rb').read()).hexdigest()}\"")


if __name__ == "__main__":
    main()
