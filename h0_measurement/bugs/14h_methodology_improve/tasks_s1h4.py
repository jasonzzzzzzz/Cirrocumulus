"""tasks_s1h4.py -- R14 Stage 1h, R4: real tasks at 128K for the project's harness:
LongBench v2 and a HELMET subset. Standalone (tokenizer only, no model): run_s1h4.py
installs them into sievelib.tasks_ruler for one run. Items come from the frozen manifest
data/r4/manifest_r4.json (make_r4_manifest.py), pinned by sha256 here.

TASKS
  lbv2   LongBench v2 (zai-org/LongBench-v2@2b48e494; sievelib.tasks_longbench_v2: the
         official 0-shot prompt in the model's chat template, as bug 9's V4-V7), scored by
         FORCED CHOICE as V5-V7: the question ends with 'The correct answer is (' and every
         arm's choice is the argmax of its teacher-forced A/B/C/D logits (run_s1h4 records
         them). One generated token.
  kilt_nq, kilt_hotpotqa      HELMET RAG at 128K (princeton-nlp/HELMET@aeadacc6, data
         dddb209d): 1,000 passages per question (k1000; dep6 / dep3), 2 demonstrations;
         substring exact match (SubEM) on the raw and the 'Answer:'-parsed output, max over
         the gold answers. 20 generated tokens.
  msmarco_rerank_psg          HELMET re-ranking: 1,000 passages, 2 demonstrations; NDCG@10 of
         the parsed ranking against the graded labels (trec_eval's linear gain). 200 tokens.
  icl_trec_coarse, icl_banking77  HELMET many-shot ICL, balanced demonstrations (6,600 /
         5,900 shots) with labels mapped to random integers per item, so an answer without
         the context is at chance; exact match after parsing 'label:'. 20 tokens.
  All HELMET tasks: no chat template (HELMET's configs), prompt = user template + '\\n' +
  'Answer:' / 'Ranking:' / 'label:', BOS; the context is cut from its end to fit 131,072
  tokens minus the generation cap, as HELMET's tokenize() does.
ITEMS (manifest; FP and labels never seen):
  - lbv2: bug 9's V7 pool (singletons that fit Qwen at 131K), its exposed qualification
    rows and its development rows (V7 is closed), in V7's selection order, those that fit
    the model at 131,072 tokens. V7's 45 confirmation rows stay untouched.
  - HELMET ICL: HELMET's own test selection (trec: the 500 test items in order; banking77:
    the test set shuffled with seed 42 and label-balanced to 500, as load_icl does).
  - HELMET RAG and re-rank: the test file's distinct questions in sha256(salt + key) order
    (HELMET samples 100 with Python's global random state, which depends on the order its
    config loads datasets in; not replicated). The KILT files hold each question at several
    gold-passage depths (dep6 / dep3) and HELMET evaluates all of them; R4 takes one row per
    question, chosen by sha256(salt + 'depth' + key) mod the question's rows, so depths are
    spread over items. Demonstrations as HELMET (a per-question sha256-seeded shuffle of the
    demo file, duplicates dropped, the first 2).
THE SPLIT. The context segment ends with the document (LB v2), or the passages and
  demonstrations (HELMET); the question segment is the rest. The cut is the first token
  boundary at or after the document's end at which the harness's separate tokenization
  (context with BOS, question without) equals the whole prompt's tokenization, so the
  model reads the same tokens as an uncut prompt (split_consistent).
THE VOTE SPAN (lbv2). The question-time vote observes 32 rows spread evenly over the
  question and its four choices (vote_span), not the question segment's last 32 rows, which
  are the official format instruction, the chat template's tail and the response prefix.
  HELMET's questions are short: their vote keeps the last rows (the whole question).
CLOSED BOOK. Every prompt has a closed-book twin with an empty document (LB v2) or empty
  passages and demonstrations (HELMET), and the same question segment (cb_ctx_text).
THE ANSWER SPAN for the NLL / KL metrics is FP's whole first line (answer_span).
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import random
import re
import string
import sys
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from sievelib import tasks_longbench_v2 as LB2  # noqa: E402

DATA = os.path.join(HERE, "data", "r4")
LBV2_JSON = os.path.join(ROOT, ".h0_corpus", "longbench_v2", "data-2b48e494.json")
HM_DIR = os.path.join(ROOT, ".h0_corpus", "helmet")
HM_SOURCES = dict(code="princeton-nlp/HELMET@aeadacc6593e9a222a84b0660b186a4e23d2efda",
                  data="HF princeton-nlp/HELMET data.tar.gz@dddb209d03e38f1f0faf76d6d05ef4ccf96240ee",
                  trec="HF CogComp/trec refs/convert/parquet@65752bf53af25bc935a0dce92fb5b6c930728450",
                  banking77="PolyAI-LDN/task-specific-datasets@57ec275d8078af65b7731c2a98be812d844a6d6b (CSV) + "
                            "HF PolyAI/banking77@90d4e2ee (label names)")
HM_FILES = {   # task -> (test file, demo file) under HM_DIR (HELMET's 128K configs)
    "kilt_nq": ("data/kilt/nq-dev-multikilt_1000_k1000_dep6.jsonl", "data/kilt/nq-train-multikilt_1000_k3_dep6.jsonl"),
    "kilt_hotpotqa": ("data/kilt/hotpotqa-dev-multikilt_1000_k1000_dep3.jsonl",
                      "data/kilt/hotpotqa-train-multikilt_1000_k3_dep3.jsonl"),
    "msmarco_rerank_psg": ("data/msmarco/test_reranking_data_k1000_dep3.jsonl",
                           "data/msmarco/test_reranking_data_k10_dep3.jsonl"),
}
DATA_SHA256 = {   # data/r4/ (ICL) and HM_DIR (RAG, re-rank), checked when loaded
    "helmet_icl/trec_train.parquet": "5ffb07b1cbd45e47b4dd1dd113dadd19e44eb260721114c6147e0ce6c1e47590",
    "helmet_icl/trec_test.parquet": "6aa889b0289e02760644caa22ac9aa833e79e34fb1f1ced69f745a078550c56e",
    "helmet_icl/banking77_train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "helmet_icl/banking77_test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "helmet_icl/banking77_labels.json": "b5ffecc77b559dbe88a88086401988bdcaf379fc798f1f27c1e0344de8417b35",
}
HM_SHA256 = {     # the demo files and the selected test items (make_r4_manifest.py prints them)
    "data/kilt/nq-train-multikilt_1000_k3_dep6.jsonl": "78ce24e766fa40bb7aa784e41393d67cdf8efef088668c0818554c2da9614bae",
    "data/kilt/hotpotqa-train-multikilt_1000_k3_dep3.jsonl": "52bab6190ea5aca655e2816fbbc2388205e8e2a0a6b2ba86551d882bc016cdcc",
    "data/msmarco/test_reranking_data_k10_dep3.jsonl": "05e872b2ad2c8fdfb36e82665d1724cbc130ce992d99d765145df76dc2d6ac1f",
    "r4_items/kilt_nq.jsonl": "adcf442413b256dbf441a64e1dacd3ba13241e95deedc163622e62d5f14a7bd1",
    "r4_items/kilt_hotpotqa.jsonl": "a7515f06f5f9cc1474fde001fc90363aa4c62cd5fe4c3073a5528df08e899664",
    "r4_items/msmarco_rerank_psg.jsonl": "b6898b0ffd96318f1303b3208489cce215211dff097d0555266e9227c6296cae",
}
HM_TEST_SHA256 = {    # the full test files (read by make_r4_manifest.py only)
    "data/kilt/nq-dev-multikilt_1000_k1000_dep6.jsonl": "d39073c4915041e4319d313aa558d9a6cee1571dabdf84a53155abb9fedd342a",
    "data/kilt/hotpotqa-dev-multikilt_1000_k1000_dep3.jsonl": "10cfc8abf743a1b9683b6244db7978fbe0af307d88ee0e5f0a8cec2efad92a7d",
    "data/msmarco/test_reranking_data_k1000_dep3.jsonl": "2697bac97ad4bdfad5b837824eafaf5df5c92d99341ddb41d766bbb31c26b025",
}
HM_ITEMS = "r4_items"   # under HM_DIR: the manifest's RAG / re-rank test records, one jsonl per task
N_ITEMS_HM = 20         # items per HELMET task in the manifest
MANIFEST = os.path.join(DATA, "manifest_r4.json")
MANIFEST_SHA256 = "d1af8acfcb641827be629eda179dc6bd8d8053f1ba571575cf7faef2e0d988f7"   # make_r4_manifest.py, 2026-10-05
GEN_VERSION = "s1h_r4_v2"

LBV2 = "lbv2"
TASKS_HM = ("kilt_nq", "kilt_hotpotqa", "msmarco_rerank_psg", "icl_trec_coarse", "icl_banking77")
TASKS = (LBV2,) + TASKS_HM
FAMILY = {LBV2: "lbv2", "kilt_nq": "rag", "kilt_hotpotqa": "rag", "msmarco_rerank_psg": "rerank",
          "icl_trec_coarse": "icl", "icl_banking77": "icl"}
GEN = {"lbv2": 1, "rag": 20, "rerank": 200, "icl": 20}
SHOTS = {"rag": 2, "rerank": 2}
ICL_SHOTS = {"icl_trec_coarse": 6600, "icl_banking77": 5900}
HM_MAX_LEN = 131072
HM_SEED = 42
SCAFFOLD = "The correct answer is ("
SALT = b"s1h_r4_helmet_order_v1\0"
VOTE_ROWS = 32
MODEL_IDS = {"llama31-8b": "meta-llama/Llama-3.1-8B-Instruct", "qwen3-30b-a3b-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507"}
CHAT_TEMPLATE_SHA256 = {"llama31-8b": "e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65",
                        "qwen3-30b-a3b-2507": "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326"}

# HELMET's templates (data.py), verbatim
RAG_USER = ("Use the given documents to write a concise and short answer to the question. Write your answer in the "
            "following format:\nAnswer: [answer]\n\n{demos}{context}\n\nQuestion: {question}")
RAG_DEMO = "{documents}\n\nQuestion: {question}\nAnswer: {answer}"
RAG_PASSAGE = "Document (Title: {title}): {text}"
RERANK_USER = ("You are provided with a list of documents, each indicated by their ID. Rank each document based on "
               "their relevance to the question in descending order from most relelvant to least relevant texts. "
               "Include all documents in the rankings. Write your answer using the unique IDs, with the following "
               "format:\nRanking: ID3 > ID1 > ID2\n\n{demos}{context}\n\nQuery: {question}")
ICL_USER = ("Use the provided mapping from the text to label to assign a label to the text. Only output "
            "\"label: {{label}}\" and nothing else. \n\n{context}\n\n{question}")
ICL_ITEM = "{text}\nlabel: {label}"
SYSTEM = {"rag": "Answer:", "rerank": "Ranking:", "icl": "label:"}


def cell_key(suite: str, model: str, ctx: int) -> str:
    return f"{suite}/{model}/{int(ctx)}"


def suite_of(task: str) -> str:
    return "lbv2" if task == LBV2 else "helmet"


# -------------------------------------------------------------------- data
def _sha(path) -> str:
    return LB2.sha256_file(path)


@lru_cache(maxsize=None)
def _verified(rel: str) -> str:
    if rel in DATA_SHA256:
        p, want = os.path.join(DATA, rel), DATA_SHA256[rel]
    else:
        p, want = os.path.join(HM_DIR, rel), HM_SHA256.get(rel)
    if want is not None and _sha(p) != want:
        raise RuntimeError(f"{p}: sha256 is not the pinned {want[:12]}")
    return p


@lru_cache(maxsize=None)
def lb2_items() -> dict:
    return LB2.index_by_id(LB2.load_dataset(LBV2_JSON, authenticate=True))


@lru_cache(maxsize=None)
def jsonl(rel: str) -> tuple:
    with open(_verified(rel), encoding="utf-8") as fh:
        return tuple(json.loads(x) for x in fh if x.strip())


@lru_cache(maxsize=None)
def manifest() -> dict:
    if MANIFEST_SHA256 is not None and _sha(MANIFEST) != MANIFEST_SHA256:
        raise RuntimeError(f"{MANIFEST} is not the frozen R4 manifest")
    return json.load(open(MANIFEST))


def _key(task, sample):
    if task == "msmarco_rerank_psg":
        return sample["qid"] if "qid" in sample else sample["query"]
    return sample["id"] if "id" in sample else sample["question"]


def salted_order(task, keys) -> list:
    """RAG / re-rank: the distinct keys in sha256(salt + task + key) order."""
    keys = list(dict.fromkeys(str(k) for k in keys))
    return sorted(keys, key=lambda k: hashlib.sha256(SALT + task.encode() + b"\0" + k.encode()).hexdigest())


def depth_row(task, key, n_rows: int) -> int:
    """Which of a question's rows (its depths) R4 uses."""
    return int(hashlib.sha256(SALT + b"depth\0" + task.encode() + b"\0" + key.encode()).hexdigest(), 16) % n_rows


# --------------------------------------------------------------- HELMET ICL
@lru_cache(maxsize=None)
def icl_data(task) -> tuple:
    """(train, test, n_labels, text field, label field) as HELMET's load_icl loads them."""
    if task == "icl_trec_coarse":
        import pandas as pd
        tr = pd.read_parquet(_verified("helmet_icl/trec_train.parquet")).to_dict("records")
        te = pd.read_parquet(_verified("helmet_icl/trec_test.parquet")).to_dict("records")
        return tuple(tr), tuple(te), 6, "text", "coarse_label"
    names = json.load(open(_verified("helmet_icl/banking77_labels.json")))

    def rows(rel):
        with open(_verified(rel), encoding="utf-8") as f:
            r = csv.reader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
            next(r)
            return tuple({"text": t, "label": names.index(c)} for t, c in r)
    return rows("helmet_icl/banking77_train.csv"), rows("helmet_icl/banking77_test.csv"), 77, "text", "label"


def balance_labels(data, shots, seed, label_field):
    """HELMET's load_icl.balance_labels, verbatim in effect."""
    rand = random.Random(seed)
    label_mapping = {x[label_field]: [] for x in data}
    for x in data:
        label_mapping[x[label_field]].append(x)
    num_rounds = math.ceil(shots / len(label_mapping))
    new_data = [[] for _ in range(num_rounds)]
    for _, samples in label_mapping.items():
        indices = rand.sample(range(len(samples)), num_rounds % len(samples))
        while len(indices) < num_rounds:
            indices += rand.sample(range(len(samples)), min(num_rounds - len(indices), len(samples)))
        for i, idx in enumerate(indices):
            new_data[i].append(samples[idx])
    for i in range(len(new_data)):
        rand.shuffle(new_data[i])
    return [item for sublist in new_data for item in sublist][:shots]


@lru_cache(maxsize=None)
def icl_test(task) -> tuple:
    """HELMET's test selection: max_test_sample 500; if the test set is larger, shuffled with
    seed 42 (datasets' Dataset.shuffle = numpy default_rng permutation) and label-balanced."""
    _, te, _, _, lf = icl_data(task)
    if len(te) > 500:
        perm = np.random.default_rng(HM_SEED).permutation(len(te))
        te = balance_labels([te[int(i)] for i in perm], 500, HM_SEED, lf)
    return tuple(te)


def icl_item(task, sample) -> dict:
    """HELMET's load_icl.preprocess for one test sample: balanced demos with labels mapped to
    a per-item random permutation of the integers."""
    tr, _, n_labels, tf, lf = icl_data(task)
    local_seed = (int(hashlib.sha256(sample[tf].encode("utf-8")).hexdigest(), 16) + HM_SEED) % 2 ** 31
    demos = balance_labels(tr, ICL_SHOTS[task], local_seed, lf)
    label_mapping = list(range(n_labels))
    rng = random.Random(local_seed)                  # HELMET seeds the global random; the same stream
    rng.shuffle(label_mapping)
    context = "\n\n".join(ICL_ITEM.format(text=d[tf], label=str(label_mapping[int(d[lf])])) for d in demos)
    return dict(_id=hashlib.sha256(sample[tf].encode()).hexdigest()[:16], context=context, demos="",
                question=sample[tf], answer=str(label_mapping[int(sample[lf])]), n_labels=n_labels)


# -------------------------------------------------------- HELMET RAG, re-rank
def _demo_pick(task, demos, key, h):
    perm = np.random.default_rng(h).permutation(len(demos))
    out, seen = [], set()
    for i in perm:
        d = demos[int(i)]
        k = str(_key(task, d))
        if k in seen:
            continue
        seen.add(k)
        out.append(d)
        if len(out) == SHOTS[FAMILY[task]]:
            break
    return out


def hm_item(task, key) -> dict:
    """One RAG or re-rank item, HELMET's update() (demos, passages, answer / qrels); the test
    record from the manifest's subset file (the full test files are hundreds of MB)."""
    _, demo_rel = HM_FILES[task]
    sample = next(s for s in jsonl(f"{HM_ITEMS}/{task}.jsonl") if str(_key(task, s)) == key)
    if task == "msmarco_rerank_psg":
        demos = [d for d in jsonl(demo_rel) if d["qid"] != sample["qid"]]
        h = abs(int(hashlib.sha256(sample["qid"].encode("utf-8")).hexdigest(), 16) % 2 ** 31)
        tmpl = "[ID: {id}] Document (Title: {title}): {text}" if "title" in sample["ctxs"][0] else "[ID: {id}] Document: {text}"
        demo_text = ""
        for d in _demo_pick(task, demos, "qid", h):
            ranking = " > ".join(x["id"] for x in sorted(d["ctxs"], key=lambda x: x["label"], reverse=True))
            demo_text += ("\n\n".join(tmpl.format(**c) for c in d["ctxs"]) + f"\n\nQuery: {d['query']}\nRanking: "
                          f"{ranking}" + "\n\n")
        return dict(_id=str(sample["qid"]), demos=demo_text, context="\n\n".join(tmpl.format(**c) for c in sample["ctxs"]),
                    question=sample["query"], qrels={str(c["id"]): int(c["label"]) for c in sample["ctxs"]})
    h = int(hashlib.sha256(str(_key(task, sample)).encode("utf-8")).hexdigest(), 16) % 2 ** 31
    demo_text = "\n\n".join(RAG_DEMO.format(documents="\n\n".join(RAG_PASSAGE.format(**c) for c in d["ctxs"]),
                                            question=d["question"], answer=d["answers"][0])
                            for d in _demo_pick(task, jsonl(demo_rel), None, h)) + "\n\n"
    return dict(_id=str(_key(task, sample)), demos=demo_text,
                context="\n\n".join(RAG_PASSAGE.format(**c) for c in sample["ctxs"]) if sample["ctxs"] else "",
                question=sample["question"], answers=[str(a) for a in sample["answers"]])


# --------------------------------------------------------------- rendering
def _adds_bos(tok) -> bool:
    bos = getattr(tok, "bos_token_id", None)
    return bos is not None and tok("a").input_ids[:1] == [bos]


def _chat(tok, prompt: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def harness_text(tok, rendered: str, chat: bool) -> str:
    """The text as the harness tokenizes it (the tokenizer adds BOS; a chat template's own BOS
    text is removed)."""
    if chat and _adds_bos(tok) and tok.bos_token and rendered.startswith(tok.bos_token):
        return rendered[len(tok.bos_token):]
    return rendered


def split_consistent(tok, full: str, cut: int, tries: int = 12) -> tuple:
    """(context text, question text): the first token boundary at or after character `cut`
    where tok(context) + tok(question, no special tokens) == tok(full) (module docstring)."""
    enc = tok(full, return_offsets_mapping=True)
    ids, off = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    cands = [i for i, (a, b) in enumerate(off) if a >= cut and b > a]
    for i in cands[:tries]:
        c = off[i][0]
        if (tok(full[:c]).input_ids == ids[:i] and tok(full[c:], add_special_tokens=False).input_ids == ids[i:]):
            return full[:c], full[c:]
    raise ValueError("no token boundary near the document's end tokenizes the same split as whole")


def _span_tokens(tok, question: str, a: int, b: int) -> tuple:
    """Question-token indices [c0, c1) covering characters [a, b) of the question."""
    off = tok(question, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    c0 = next(i for i, (s, e) in enumerate(off) if e > a)
    c1 = next((i for i, (s, e) in enumerate(off) if s >= b), len(off))
    return int(c0), int(c1)


def render_lbv2(tok, item: dict) -> dict:
    full = harness_text(tok, _chat(tok, LB2.render_prompt(item)), True) + SCAFFOLD
    doc = item["context"].strip()
    i = full.find(doc)
    if i < 0 or full.find(doc, i + 1) >= 0:
        raise ValueError("the document does not occur exactly once in the rendered prompt")
    ctx, q = split_consistent(tok, full, i + len(doc))
    a = q.find("What is the correct answer to this question:")
    b = q.find("\n\nFormat your response as follows:")
    if a < 0 or b < a:
        raise ValueError("lbv2: the question / choices span was not found in the question segment")
    empty = dict(item, context="")
    cb_full = harness_text(tok, _chat(tok, LB2.render_prompt(empty)), True) + SCAFFOLD
    if not cb_full.endswith(q):
        raise ValueError("lbv2: the closed-book prompt does not end with the same question segment")
    return dict(ctx=ctx, question=q, vote_span=_span_tokens(tok, q, a, b), cb_ctx=cb_full[:len(cb_full) - len(q)],
                truncated_tokens=0)


def _hm_prompt(fam, it, context, demos):
    user = {"rag": RAG_USER, "rerank": RERANK_USER, "icl": ICL_USER}[fam]
    return user.format(demos=demos, context=context, question=it["question"]) + "\n" + SYSTEM[fam]


def render_hm(tok, task: str, it: dict, max_len: int = HM_MAX_LEN) -> dict:
    """HELMET's prompt (no chat template, BOS), its end-of-context truncation to max_len -
    the generation cap (model_utils.tokenize), the consistent split at the context's end."""
    fam = FAMILY[task]
    gen = GEN[fam]
    context, cut_tokens = it["context"], 0
    full = _hm_prompt(fam, it, context, it["demos"])
    for _ in range(5):           # HELMET cuts once; a re-tokenized boundary can leave a token or two over
        n = len(tok(full).input_ids)
        if n <= max_len - gen:
            break
        t = n - (max_len - gen)
        off = tok([context], return_offsets_mapping=True)["offset_mapping"][0]
        context, cut_tokens = context[:off[-t][0]], cut_tokens + t
        full = _hm_prompt(fam, it, context, it["demos"])
    else:
        raise ValueError(f"{task} {it['_id']}: the context cannot be cut to fit {max_len - gen} tokens")
    head = _hm_prompt(fam, it, "\x00", it["demos"]).split("\x00")[0]
    ctx, q = split_consistent(tok, full, len(head) + len(context))
    cb_full = _hm_prompt(fam, it, "", "")
    if not cb_full.endswith(q):
        raise ValueError(f"{task}: the closed-book prompt does not end with the same question segment")
    return dict(ctx=ctx, question=q, vote_span=None, cb_ctx=cb_full[:len(cb_full) - len(q)], truncated_tokens=cut_tokens)


def item_record(task: str, key: str) -> dict:
    if task == LBV2:
        return lb2_items()[key]
    if FAMILY[task] == "icl":
        return next(icl_item(task, s) for s in icl_test(task) if icl_item_id(s) == key)
    return hm_item(task, key)


def icl_item_id(sample) -> str:
    return hashlib.sha256(sample["text"].encode()).hexdigest()[:16]


def render(tok, task: str, rec: dict) -> dict:
    return render_lbv2(tok, rec) if task == LBV2 else render_hm(tok, task, rec)


def n_tokens(tok, r: dict) -> int:
    return len(tok(r["ctx"]).input_ids) + len(tok(r["question"], add_special_tokens=False).input_ids)


def generation_limit(task: str) -> int:
    return GEN[FAMILY[task]]


def _truncate_middle(tok, ctx_text: str, keep: int) -> str:
    """Smokes only: the middle of the context cut out so that `keep` tokens remain."""
    ids = tok(ctx_text, add_special_tokens=False).input_ids
    if len(ids) <= keep:
        return ctx_text
    h = keep // 2
    return tok.decode(ids[:h]) + tok.decode(ids[len(ids) - (keep - h):])


def build(tok, task, ctx, *, prompt_idx, cell=None, model=None, rec=None, allow_truncate=False):
    """One prompt: (text, meta); text = context + meta['question']. `rec` overrides the
    manifest (tests). allow_truncate: smokes only."""
    if task not in TASKS:
        raise ValueError(f"unknown R4 task {task!r}; one of {TASKS}")
    if rec is None:
        ids = manifest()["cells"][cell][task]
        if not 0 <= int(prompt_idx) < len(ids):
            raise ValueError(f"{cell} {task}: prompt {prompt_idx} outside the manifest's {len(ids)} items")
        rec = item_record(task, ids[int(prompt_idx)][0])
    if task == LBV2 and model in CHAT_TEMPLATE_SHA256 and not allow_truncate:
        got = hashlib.sha256((tok.chat_template or "").encode()).hexdigest()
        if got != CHAT_TEMPLATE_SHA256[model]:
            raise RuntimeError(f"{model}: chat template sha256 {got[:12]} is not the pinned one")
    r = render(tok, task, rec)
    max_new = generation_limit(task)
    n = n_tokens(tok, r)
    truncated = r["truncated_tokens"] > 0
    if n + max_new > ctx:
        if not allow_truncate:
            raise ValueError(f"{task} {rec['_id']}: {n} + {max_new} tokens exceed ctx {ctx}")
        keep = ctx - max_new - len(tok(r["question"], add_special_tokens=False).input_ids) - 16
        r["ctx"], truncated = _truncate_middle(tok, r["ctx"], keep), True
        n = n_tokens(tok, r)
    fam = FAMILY[task]
    if task == LBV2:
        pre, _ = LB2.choice_logit_contract(tok)
        qi = tok(r["question"], add_special_tokens=False).input_ids
        if qi[-len(pre):] != list(pre):
            raise RuntimeError(f"the question does not end with the canonical response prefix {pre}")
        expected, distractors = [rec["answer"]], [c for c in "ABCD" if c != rec["answer"]]
    elif fam == "rag":
        expected, distractors = list(rec["answers"]), []
    elif fam == "icl":
        expected, distractors = [rec["answer"]], []
    else:
        expected, distractors = [f"{k}:{v}" for k, v in rec["qrels"].items() if v > 0], []
    meta = dict(synthetic=False, source=f"r4_{suite_of(task)}", doc=rec["_id"], offset=0, spliced=False,
                corpus_sha=LB2.DATASET_SHA256 if task == LBV2 else "helmet:" + HM_SOURCES["data"][-8:],
                family=f"r4_{task}", task=task, prompt_idx=int(prompt_idx), expected=expected, distractors=distractors,
                question=r["question"], target_needle_depth=None, target_needle_rank=None, query_term="",
                lb_id=rec["_id"], n_input_tokens=n, truncated=truncated, truncated_tokens=r["truncated_tokens"],
                vote_span=r["vote_span"], cb_ctx_text=r["cb_ctx"])
    return r["ctx"] + r["question"], meta


# ------------------------------------------------------------ the metrics
def normalize_answer(s):
    """HELMET's (DrQA's) normalize_answer."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def parse_output(output, prefix="Answer:"):
    """HELMET's utils.parse_output."""
    def lstrip_string(s, sub):
        return re.sub(f"^{re.escape(sub)}", "", s, flags=re.IGNORECASE)
    for pat in (re.compile(f"(?:{prefix})(.*)(?:\n|$)", flags=re.IGNORECASE), re.compile(r"(?:^)(.*)(?:\n|$)")):
        m = pat.search(output)
        if m is not None:
            return lstrip_string(m[1].strip(), prefix).strip()
    return None


def parse_rankings(output):
    """HELMET's utils.parse_rankings."""
    output = re.sub(r"[\[\]:]", "", output)
    output = output.lower().replace("id", "")
    longest = ""
    for m in re.finditer(r"(\d+)(?:\s*>\s*(\d+))*", output):
        if len(m.group(0)) > len(longest):
            longest = m.group(0)
    rankings = [x.strip() for x in longest.split(">") if x.strip().isdigit()] if longest else [output]
    results = {}
    for i, rank in enumerate(rankings):
        if rank not in results:
            results[rank] = len(rankings) - i
    return results


def ndcg_at(results: dict, qrels: dict, k: int = 10) -> float:
    """trec_eval's ndcg_cut_k (linear gain, log2 discount), as pytrec_eval computes it."""
    ranked = sorted(results.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    dcg = sum(qrels.get(d, 0) / math.log2(i + 2) for i, (d, _) in enumerate(ranked))
    ideal = sorted((v for v in qrels.values() if v > 0), reverse=True)[:k]
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def lbv2_choice(pred: str):
    m = re.match(r"\s*([A-D])", str(pred))
    return m.group(1) if m else None


def score(task, pred, meta) -> dict:
    exp = [str(x) for x in meta["expected"]]
    p = str(pred)
    fam = FAMILY[task]
    if fam == "lbv2":
        c = lbv2_choice(p)
        s = float(c == exp[0])
        return {"score": s, "hits": int(s), "n_expected": 1, "distractor": bool(c and c != exp[0]),
                "first_ok": float(c is not None)}
    if fam == "rag":
        cands = [p] + ([parse_output(p, SYSTEM["rag"])] if parse_output(p, SYSTEM["rag"]) is not None else [])
        s = float(any(normalize_answer(g) in normalize_answer(c) for c in cands for g in exp))
    elif fam == "icl":
        parsed = parse_output(p, SYSTEM["icl"]) or ""
        s = float(any(normalize_answer(parsed) == normalize_answer(g) for g in exp))
    else:
        qrels = {x.split(":")[0]: int(x.split(":")[1]) for x in exp}
        s = ndcg_at(parse_rankings(p), qrels, 10)
    return {"score": s, "hits": int(s >= 1.0), "n_expected": 1, "distractor": False, "first_ok": float("nan")}


# ---------------------------------------------------- masks (the runner's helpers)
def answer_span(tok, ids) -> dict:
    """s1d_lib.answer_tokens' record for a natural-task answer: the value tokens are the
    content tokens (any letter or digit) of FP's first line, the span ends at that line's
    end; found = [the line]. Returns dict(vmask, span_end, found, text)."""
    ids = [int(x) for x in ids]
    n = len(ids)
    full = tok.decode(ids) if n else ""
    starts, ends, prev = [], [], 0
    for i in range(n):
        e = len(tok.decode(ids[:i + 1]))
        starts.append(min(prev, e))
        ends.append(e)
        prev = e
    lead = len(full) - len(full.lstrip())
    line = full[lead:].split("\n")[0].rstrip()
    if not line.strip():
        return dict(vmask=[False] * n, span_end=0, found=[], text=full)
    a, b = lead, lead + len(line)
    span_end = sum(1 for s in starts if s < b)
    vmask = [i < span_end and starts[i] < b and ends[i] > a and any(ch.isalnum() for ch in tok.decode([ids[i]]))
             for i in range(n)]
    return dict(vmask=vmask, span_end=span_end, found=[line] if any(vmask) else [], text=full)


def vote_rows(span, n_rows: int = VOTE_ROWS) -> list:
    """The question-token indices the vote observes: n_rows spread evenly over [c0, c1)
    (all of them if fewer), as floor(j (n - 1) / (m - 1))."""
    c0, c1 = int(span[0]), int(span[1])
    n = c1 - c0
    if n <= n_rows:
        return list(range(c0, c1))
    return sorted({c0 + (j * (n - 1)) // (n_rows - 1) for j in range(n_rows)})
