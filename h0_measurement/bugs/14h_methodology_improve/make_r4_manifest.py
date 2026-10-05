#!/usr/bin/env python3
"""R4's frozen item manifest, data/r4/manifest_r4.json (rules: tasks_s1h4.py's docstring).
Tokenizers only, no model, no labels: an item's eligibility depends on its rendered length
alone. Run on a login node; tasks_s1h4.MANIFEST_SHA256 and HM_SHA256 then pin the outputs.

    .venv/bin/python -u h0_measurement/bugs/14h_methodology_improve/make_r4_manifest.py --extract   # HELMET files
    .venv/bin/python -u h0_measurement/bugs/14h_methodology_improve/make_r4_manifest.py              # the manifest

--extract takes HELMET's 128K RAG / re-rank files out of .h0_corpus/helmet/data-dddb209d.tar.gz
(11.3 GB; one pass) into .h0_corpus/helmet/data/. The manifest step writes the selected RAG /
re-rank test records to .h0_corpus/helmet/r4_items/<task>.jsonl and prints every sha256.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import tasks_s1h4 as T  # noqa: E402

BUG9 = os.path.join(os.path.dirname(HERE), "9_sota_eviction_baselines")
LBV2_POOL = ("longbench_v2_qwen30_v7_manifest.json", ("qualification", "development"))   # V7: confirmation untouched
CTX = 131072
TARBALL = os.path.join(T.HM_DIR, "data-dddb209d.tar.gz")
TARBALL_BYTES = 11_271_916_108


_KEY_RE = {"question": re.compile(rb'"question":\s*("(?:[^"\\]|\\.)*")'),
           "qid": re.compile(rb'"qid":\s*("(?:[^"\\]|\\.)*")')}


def key_offsets(task, path) -> dict:
    """key -> byte offsets of its rows (one pass; the key read from the line's JSON prefix,
    checked against a full parse on the first rows)."""
    field = "qid" if task == "msmarco_rerank_psg" else "question"
    out, off, checked = {}, 0, 0
    with open(path, "rb") as fh:
        for line in fh:
            if line.strip():
                m = _KEY_RE[field].search(line[:20000])
                k = json.loads(m.group(1)) if m else None
                if checked < 20:
                    full = str(T._key(task, json.loads(line)))
                    if full != k:
                        raise SystemExit(f"{task}: key prefix {k!r} differs from the parsed key {full!r}")
                    checked += 1
                out.setdefault(str(k), []).append(off)
            off += len(line)
    return out


def extract():
    if os.path.getsize(TARBALL) != TARBALL_BYTES:
        raise SystemExit(f"{TARBALL}: {os.path.getsize(TARBALL)} bytes, expected {TARBALL_BYTES}")
    want = {p for pair in T.HM_FILES.values() for p in pair}
    got = set()
    with tarfile.open(TARBALL, "r:gz") as tf:
        for m in tf:
            name = m.name[2:] if m.name.startswith("./") else m.name
            if name in want:
                try:
                    tf.extract(m, T.HM_DIR, filter="data")
                except TypeError:                     # Python without extraction filters
                    tf.extract(m, T.HM_DIR)
                got.add(name)
                print("extracted", name, flush=True)
                if got == want:
                    break
    if got != want:
        raise SystemExit(f"missing from the tarball: {sorted(want - got)}")
    for p in sorted(want):
        print(f'    "{p}": "{T._sha(os.path.join(T.HM_DIR, p))}",')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    a = ap.parse_args()
    if a.extract:
        return extract()
    from transformers import AutoTokenizer
    toks = {m: AutoTokenizer.from_pretrained(i) for m, i in T.MODEL_IDS.items()}
    for m, tok in toks.items():
        got = hashlib.sha256(tok.chat_template.encode()).hexdigest()
        if got != T.CHAT_TEMPLATE_SHA256[m]:
            raise SystemExit(f"{m}: chat template {got[:12]} is not the pinned one")
    cells, pins = {}, {}
    src = json.load(open(os.path.join(BUG9, LBV2_POOL[0])))
    rows = [e for e in src["examples"] if e["split"] in LBV2_POOL[1]]
    lb2 = T.lb2_items()
    for m, tok in toks.items():
        out = []
        for e in rows:
            r = T.render(tok, T.LBV2, lb2[e["id"]])
            n = T.n_tokens(tok, r)
            if n + T.generation_limit(T.LBV2) <= CTX:
                out.append([e["id"], n])
        cells[T.cell_key("lbv2", m, CTX)] = {T.LBV2: out}
        print(f"lbv2 {m}: {len(out)} of {len(rows)} fit; tokens {min(x[1] for x in out)}-{max(x[1] for x in out)}",
              flush=True)
    # HELMET: the RAG / re-rank items in salted order (their records saved), ICL in HELMET's order
    keys, items_dir = {}, os.path.join(T.HM_DIR, T.HM_ITEMS)
    os.makedirs(items_dir, exist_ok=True)
    for task in T.TASKS_HM:
        if T.FAMILY[task] == "icl":
            ids = [T.icl_item_id(s) for s in T.icl_test(task)]
            if len(set(ids)) != len(ids):
                raise SystemExit(f"{task}: duplicate test texts")
            keys[task] = ids[:T.N_ITEMS_HM]
            continue
        test = os.path.join(T.HM_DIR, T.HM_FILES[task][0])
        if T._sha(test) != T.HM_TEST_SHA256[T.HM_FILES[task][0]]:
            raise SystemExit(f"{test}: not the pinned HELMET test file")
        offs = key_offsets(task, test)
        keys[task] = T.salted_order(task, offs)[:T.N_ITEMS_HM]
        path = os.path.join(items_dir, f"{task}.jsonl")
        with open(path, "w", encoding="utf-8") as fh, open(test, "rb") as srcf:
            for k in keys[task]:
                rows = offs[k]
                srcf.seek(rows[T.depth_row(task, k, len(rows))])
                rec = json.loads(srcf.readline())
                if str(T._key(task, rec)) != k:
                    raise SystemExit(f"{task}: offset of {k!r} reads another record")
                fh.write(json.dumps(rec) + "\n")
        pins[f"{T.HM_ITEMS}/{task}.jsonl"] = T._sha(path)
        T.jsonl.cache_clear()
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
    man = dict(version=T.GEN_VERSION, rules="tasks_s1h4.py docstring", lbv2_data_sha256=T.LB2.DATASET_SHA256,
               helmet_sources=T.HM_SOURCES, helmet_test_sha256=T.HM_TEST_SHA256, data_sha256=T.DATA_SHA256,
               model_ids=T.MODEL_IDS,
               chat_template_sha256=T.CHAT_TEMPLATE_SHA256, salt=T.SALT.decode("latin-1"),
               lbv2_pool=dict(source=f"bug 9: {LBV2_POOL[0]}", content_sha256=src["content_sha256"],
                              splits=list(LBV2_POOL[1])), cells=cells)
    with open(T.MANIFEST, "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    print("HM_SHA256 additions:")
    for k, v in pins.items():
        print(f'    "{k}": "{v}",')
    print(f"wrote {T.MANIFEST} sha256 {T._sha(T.MANIFEST)}")


if __name__ == "__main__":
    main()
