#!/usr/bin/env python3
"""R6's HELMET item manifest, data/r6/manifest_r6.json (plan.md amendment "R6 frozen"; R4's rules, tasks_s1h4.py's
docstring, with more items). Tokenizers only, no model, no labels.

R4's manifest holds 20 items per HELMET task, and items 0-10 have been run (R4: 0-9; R5's pilot: 5 and 10). R6 needs
30 fresh re-ranking queries, and msmarco's test file has 41 distinct queries, so this manifest takes N_ITEMS = 41
items per task in R4's order (RAG / re-rank: the salted order; ICL: HELMET's test order). Its first 20 items are R4's,
checked here, so an item's index means the same in both manifests and R6's items 11-40 are new. LongBench v2 is not
in R6 (R4: no arm differs from FP; R5's validity rule excluded its decode path).

    .venv/bin/python -u h0_measurement/bugs/14h_methodology_improve/make_r6_manifest.py

writes the RAG / re-rank records to .h0_corpus/helmet/r6_items/<task>.jsonl and prints every sha256 for tasks_s1h6.py.
"""
from __future__ import annotations
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import tasks_s1h4 as T  # noqa: E402
import make_r4_manifest as MK  # noqa: E402
import tasks_s1h6 as T6  # noqa: E402

CTX = 131072


def main():
    from transformers import AutoTokenizer
    toks = {m: AutoTokenizer.from_pretrained(i) for m, i in T.MODEL_IDS.items()}
    r4 = T.manifest()
    keys, pins = {}, {}
    items_dir = os.path.join(T.HM_DIR, T6.HM_ITEMS)
    os.makedirs(items_dir, exist_ok=True)
    for task in T.TASKS_HM:
        if T.FAMILY[task] == "icl":
            keys[task] = [T.icl_item_id(s) for s in T.icl_test(task)][:T6.N_ITEMS]
        else:
            test = os.path.join(T.HM_DIR, T.HM_FILES[task][0])
            if T._sha(test) != T.HM_TEST_SHA256[T.HM_FILES[task][0]]:
                raise SystemExit(f"{test}: not the pinned HELMET test file")
            offs = MK.key_offsets(task, test)
            keys[task] = T.salted_order(task, offs)[:T6.N_ITEMS]
            path = os.path.join(items_dir, f"{task}.jsonl")
            with open(path, "w", encoding="utf-8") as fh, open(test, "rb") as srcf:
                for k in keys[task]:
                    rows = offs[k]
                    srcf.seek(rows[T.depth_row(task, k, len(rows))])
                    rec = json.loads(srcf.readline())
                    if str(T._key(task, rec)) != k:
                        raise SystemExit(f"{task}: offset of {k!r} reads another record")
                    fh.write(json.dumps(rec) + "\n")
            pins[f"{T6.HM_ITEMS}/{task}.jsonl"] = T._sha(path)
        if len(keys[task]) < T6.N_ITEMS:
            raise SystemExit(f"{task}: only {len(keys[task])} items")
        for m in T.MODEL_IDS:
            old = [x[0] for x in r4["cells"][T.cell_key("helmet", m, CTX)][task]]
            if keys[task][:len(old)] != old:
                raise SystemExit(f"{task}: the first {len(old)} items differ from R4's manifest ({m})")
        print(f"{task}: {len(keys[task])} items; the first 20 are R4's", flush=True)
    T6.install(pins=pins, check_manifest=False)          # read the new records (not yet pinned) for rendering
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
        T6.uninstall()
    man = dict(version=T6.GEN_VERSION, rules="tasks_s1h4.py docstring (R4's), with N_ITEMS per task: tasks_s1h6.py",
               base_manifest=dict(path="data/r4/manifest_r4.json", sha256=T.MANIFEST_SHA256, first_items_equal=20),
               helmet_sources=T.HM_SOURCES, helmet_test_sha256=T.HM_TEST_SHA256, data_sha256=T.DATA_SHA256,
               model_ids=T.MODEL_IDS, salt=T.SALT.decode("latin-1"), n_items=T6.N_ITEMS, r6_items=list(T6.R6_ITEMS),
               cells=cells)
    os.makedirs(os.path.dirname(T6.MANIFEST), exist_ok=True)
    with open(T6.MANIFEST, "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    print("HM_SHA256_R6:")
    for k, v in pins.items():
        print(f'    "{k}": "{v}",')
    print(f"wrote {T6.MANIFEST} sha256 {hashlib.sha256(open(T6.MANIFEST, 'rb').read()).hexdigest()}")


if __name__ == "__main__":
    main()
