#!/usr/bin/env python3
"""R14 Stage 1h R4 anchors (LongBench v2 and HELMET at 128K, Quest, the floor system, the
closed-book arm, the vote span). CPU only; same PASS/FAIL convention as test_r8.

    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h_r4.py --fast
    .venv/bin/python h0_measurement/bugs/14h_methodology_improve/test_r14_stage1h_r4.py   # + Llama-3.2-1B driver smokes
"""
import json, math, os, shutil, subprocess, sys, tempfile
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S1G_DIR = os.path.join(os.path.dirname(HERE), "14_kernel_tpot")
H0 = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(H0)
for _p in (HERE, S1G_DIR, H0, ROOT, os.path.join(ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import s1c_lib as L1C  # noqa: E402
import s1h3_lib as L3  # noqa: E402
import s1h4_lib as L  # noqa: E402
import tasks_s1h4 as T  # noqa: E402

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0
CORPUS = os.environ.get("H0_CORPUS") or os.path.join(ROOT, ".h0_corpus/pg19")


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


def _raises(f):
    try:
        f()
    except (ValueError, KeyError, RuntimeError):
        return True
    return False


class _Enc(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class CharTok:
    def __call__(self, text, add_special_tokens=True, **kw):
        return _Enc(input_ids=([0] if add_special_tokens else []) + [ord(c) for c in text])

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i)


# ------------------------------------------------------------------- plans
def test_plans():
    print("\n[S1h R4] presets, the new arms, budgets, cells")
    pl, pq = L.build_plan(L.PRESETS["h4llama128"]), L.build_plan(L.PRESETS["h4qwen128"])
    base = L3.build_plan(dict(L3.PRESETS["h3llama"]))
    extra = [("quest_v16", 0.125), ("quest4_v4", 0.125), (L.FLOOR_ARM, 0.125), ("closedbook", 0)]
    check("Llama and Qwen at 128K: R3a's 16 arms + Quest exact/4-bit + the floor system + closed book (20), "
          "fp_noise last; stop r8list",
          pl == base[:-1] + extra + [("fp_noise", 0)] and pq == pl and L.PRESETS["h4qwen128"]["ctx"] == 131072
          and all(L.PRESETS[k]["stop"] == "r8list" for k in ("h4llama128", "h4qwen128")), f"({len(pl)} arms)")
    pa, pf, pc = L.parse_arm("quest4_v4"), L.parse_arm(L.FLOOR_ARM), L.parse_arm("closedbook")
    check("arm grammar: quest stores and value widths; the floor arm is the system with floor; closedbook",
          L.parse_arm("quest_v16")["store"] == 16 and pa["store"] == 4 and pa["family"] == "quest"
          and pf["family"] == "qread2t" and pf["floor"] and pf["base"] == "qread2t4kq_v4"
          and pc["family"] == "closedbook" and L.key_side_bits("closedbook", 1.0) == 0.0
          and L.key_side_bits("quest", 0.9) == 2.0 and _raises(lambda: L.parse_arm("quest_v4")))
    rows = L.quest_budget_rows(0.125, 100000, 32)
    tot = 2 * 100000 + 30 * L.quest_pages(rows, 100000) * L.QUEST_PAGE
    check("Quest budget: rows over all layers equal floor(r C) per layer (within a page); floor r 1/8 at 128K, "
          "1/2 at 32K, 1 below 16K rows",
          abs(tot / 32 - L1C.qread_keep_count(0.125, 100000)) <= L.QUEST_PAGE
          and _raises(lambda: L.quest_budget_rows(0.125, 4000, 16))
          and [L.floor_r(c) for c in (131072, 32768, 12000)] == [0.125, 0.5, 1.0])
    check("cells: LongBench v2 and HELMET on Llama (1 GPU) and Qwen (2 GPUs) at 128K; pilots outside the blocks",
          {c: (x["preset"], x["gpus"]) for c, x in L.CELLS.items()} == {
              "lb2llama": ("h4llama128", 1), "lb2qwen": ("h4qwen128", 2), "hmllama": ("h4llama128", 1),
              "hmqwen": ("h4qwen128", 2)}
          and all(p >= L.CELLS[c]["per"] * L.CELLS[c]["blocks"] for c in L.CELLS for p, _ in L.pilot_units(c)))


# ------------------------------------------------------------------- tasks
def test_tasks():
    print("\n[S1h R4] tasks_s1h4: manifest, HELMET items, the split, the vote span, closed book, metrics")
    man = T.manifest()
    c = man["cells"]
    check("manifest: pinned; 72 lbv2 items per model at 128K (V7 qualification + development, confirmation "
          "untouched); 20 items per HELMET task and model",
          T._sha(T.MANIFEST) == T.MANIFEST_SHA256 and all(len(c[f"lbv2/{m}/131072"]["lbv2"]) == 72 for m in T.MODEL_IDS)
          and all(len(c[f"helmet/{m}/131072"][t]) == T.N_ITEMS_HM for m in T.MODEL_IDS for t in T.TASKS_HM)
          and _no_confirmation(c))
    for task in ("icl_trec_coarse", "icl_banking77"):
        te = T.icl_test(task)
        rec = T.icl_item(task, te[0])
        import re
        found = re.findall(r"\nlabel: (\d+)(?=\n\n|$)", rec["context"])
        labs = Counter(int(x) for x in found)
        n_lab = 6 if "trec" in task else 77
        check(f"{task}: HELMET's 500 test items (balanced); {T.ICL_SHOTS[task]} balanced demos with labels mapped "
              f"to a permutation of 0..{n_lab - 1}; the answer is the test label's mapped integer",
              len(te) == 500 and len(found) == T.ICL_SHOTS[task] and set(labs) == set(range(n_lab))
              and max(labs.values()) - min(labs.values()) <= 1 and 0 <= int(rec["answer"]) < n_lab
              and (task != "icl_banking77" or max(Counter(x["label"] for x in te).values()) <= math.ceil(500 / 77)),
              f"({dict(list(labs.items())[:3])})")
    for task in ("kilt_nq", "msmarco_rerank_psg"):
        k = c["helmet/llama31-8b/131072"][task][0][0]
        rec = T.item_record(task, k)
        n_doc = rec["context"].count("Document")
        check(f"{task}: the manifest's item with HELMET's 2 demonstrations and 1,000 passages",
              n_doc >= 999 and rec["demos"].count("Query:" if "rerank" in task else "Question:") == 2
              and (task != "msmarco_rerank_psg" or len(rec["qrels"]) >= 999), f"({n_doc} passages)")
    from transformers import AutoTokenizer
    ok, det = True, []
    for m in T.MODEL_IDS:
        tok = AutoTokenizer.from_pretrained(T.MODEL_IDS[m])
        for task in (T.LBV2, "kilt_hotpotqa", "icl_trec_coarse"):
            cell = T.cell_key(T.suite_of(task), m, 131072)
            text, meta = T.build(tok, task, 131072, prompt_idx=0, cell=cell, model=m)
            q = meta["question"]
            ctx = text[:len(text) - len(q)]
            joint = tok(text).input_ids
            good = (tok(ctx).input_ids + tok(q, add_special_tokens=False).input_ids == joint
                    and meta["n_input_tokens"] == man["cells"][cell][task][0][1] == len(joint)
                    and meta["cb_ctx_text"] + q != text and (meta["cb_ctx_text"] + q).endswith(q))
            if task == T.LBV2:
                qi = tok(q, add_special_tokens=False).input_ids
                rows = T.vote_rows(meta["vote_span"])
                seen = tok.decode([qi[i] for i in range(meta["vote_span"][0], meta["vote_span"][1])])
                good &= (len(rows) == 32 and max(rows) < len(qi) - 1 and "What is the correct answer" in seen
                         and "(D)" in seen and "Format your response" not in seen and q.endswith(T.SCAFFOLD))
            else:
                good &= meta["vote_span"] is None and "<|" not in text
            ok &= good
            det.append(f"{m[:5]} {task} {meta['n_input_tokens']}")
    check("rendering: the split tokenizes as the whole prompt; token counts as the manifest; the closed-book twin "
          "keeps the question; lbv2's vote span covers the question and choices (not the format line); HELMET "
          "has no chat template", ok, f"({det})")
    s = T.score
    qrels = {"7": 3, "2": 1, "9": 0}
    exp_ndcg = (3 / math.log2(2) + 1 / math.log2(4)) / (3 / math.log2(2) + 1 / math.log2(3))
    check("metrics: SubEM on raw / parsed output; ICL exact match after 'label:'; NDCG@10 (trec_eval's linear gain); "
          "HELMET's parse_rankings; lbv2's letter",
          s("kilt_nq", " Paris, the capital.", dict(expected=["paris"]))["score"] == 1.0
          and s("kilt_nq", " London", dict(expected=["Paris"]))["score"] == 0.0
          and s("icl_banking77", " 17\n", dict(expected=["17"]))["score"] == 1.0
          and s("icl_banking77", " 17\nlabel: 3", dict(expected=["17"]))["score"] == 0.0   # HELMET parses the 'label:' line
          and s("icl_banking77", " 170", dict(expected=["17"]))["score"] == 0.0
          and T.parse_rankings("[ID: 7] > [ID: 9] > 2") == {"7": 3, "9": 2, "2": 1}
          and abs(s("msmarco_rerank_psg", " 7 > 9 > 2", dict(expected=[f"{k}:{v}" for k, v in qrels.items() if v]))
                  ["score"] - exp_ndcg) < 1e-12
          and s("lbv2", "B", dict(expected=["B"]))["score"] == 1.0, f"(ndcg {exp_ndcg:.4f})")
    a = T.answer_span(CharTok(), [ord(x) for x in "\n Paris, France.\nExtra"])
    check("answer span: FP's first line, content tokens only", a["found"] == ["Paris, France."] and a["span_end"] == 16)
    check("vote rows: 32 spread evenly over the span (all of a shorter one)",
          T.vote_rows((3, 370)) == sorted({3 + (j * 366) // 31 for j in range(32)}) and T.vote_rows((2, 10)) == list(range(2, 10)))


def _no_confirmation(cells):
    src = json.load(open(os.path.join(os.path.dirname(HERE), "9_sota_eviction_baselines",
                                      "longbench_v2_qwen30_v7_manifest.json")))
    conf = {e["id"] for e in src["examples"] if e["split"] == "confirmation"}
    return not any(i in conf for k, cl in cells.items() if k.startswith("lbv2") for t in cl.values() for i, *_ in t)


# ------------------------------------------------------------ the driver's parts
def test_driver_parts():
    print("\n[S1h R4] the driver's parts (no model): Quest, the vote filter, forced choice, plumbing")
    import torch
    import run_s1h4 as D
    from sievelib import compress as C
    import s1d_lib
    Hkv, rep, d, Cn = 2, 3, 8, 70
    g = torch.Generator().manual_seed(0)
    K = torch.randn(Hkv, Cn + 5, d, generator=g)
    C.STATE.reset_prompt()
    C.STATE.ctx_len = Cn
    C.STATE.evict = {li: torch.zeros(Hkv, Cn, dtype=torch.bool) for li in range(4)}
    C.STATE.bits = {li: torch.full((Hkv, Cn), 16, dtype=torch.long) for li in range(4)}
    try:
        D.Q.n_pages, D.Q.width, D.Q.reads = 2, 16, []
        P = L.QUEST_PAGE
        Np = math.ceil(Cn / P)
        Kc = torch.cat([K[:, :Cn], K[:, Cn - 1:Cn].expand(-1, Np * P - Cn, -1)], 1).reshape(Hkv, Np, P, d)
        D.Q.meta = {2: (Kc.amin(2), Kc.amax(2))}
        q = torch.randn(Hkv * rep, d, generator=g)
        D.quest_select(2, q)
        keep = ~C.STATE.evict[2]
        ok = True
        for h in range(Hkv):
            ub = [max(float(sum(max(q[h * rep + i, j] * Kc[h, p, :, j].min(), q[h * rep + i, j] * Kc[h, p, :, j].max())
                                for j in range(d))) for i in range(rep)) for p in range(Np)]
            ok &= sorted({int(i) // P for i in torch.nonzero(keep[h]).flatten()}) == sorted(sorted(range(Np),
                                                                                              key=lambda p: -ub[p])[:2])
        check("Quest: the top pages by the group's largest page bound (brute force); bits and masks per row",
              ok and int(keep[0].sum()) == 2 * P and torch.equal(C.STATE.bits[2] > 0, keep))
    finally:
        C.STATE.reset_prompt()
    # the vote filter: a question of 40 tokens at positions 100..138 (the last token is decode step 0), a
    # vote span over tokens 5..29, then two answer rows at 139..140 (the oracle's second call)
    D.install_tasks()
    try:
        D.CUR.n_q = 40
        D.CUR.vote = torch.zeros(40, dtype=torch.bool)
        D.CUR.vote[T.vote_rows((5, 30))] = True
        buf = s1d_lib.RowBuffer(rows=1 << 20)
        H, dd = 4, 2
        buf.append(0, torch.randn(39, H, dd), torch.arange(100, 139))
        buf.append(0, torch.randn(2, H, dd), torch.arange(139, 141))
        kept = buf.pos[0].tolist()
        buf2 = s1d_lib.RowBuffer(rows=32)
        D.CUR.vote = None
        buf2.append(0, torch.randn(39, H, dd), torch.arange(100, 139))
        last32 = buf2.pos[0].tolist()
    finally:
        D.uninstall_tasks()
    check("vote filter: of the question prefill, only the span's rows (25); later rows (the oracle's answer rows) "
          "pass; without a span, the last 32 rows as before",
          kept == [100 + i for i in range(5, 30)] + [139, 140] and last32 == list(range(107, 139)), f"({kept[:3]}..)")
    D.CUR.task, D.CUR.gold, D.CUR.branches, D.CUR.vote = T.LBV2, "C", [10, 11, 12, 13], torch.ones(5, dtype=torch.bool)
    lg = torch.zeros(1, 20)
    lg[0, 12], lg[0, 11] = 5.0, 1.0
    orig = D._ORIG["tfm_of_h"]
    D._ORIG["tfm_of_h"] = lambda lg, a, b, c: {"x": 1}
    try:
        out = D.tfm_of_r4(lg, [12], [True], {})
    finally:
        D._ORIG["tfm_of_h"] = orig
        D.CUR.task = D.CUR.gold = D.CUR.branches = D.CUR.vote = None
    check("forced choice and the vote record: A/B/C/D log-probabilities, the argmax and its correctness; vote_rows",
          out["fc_choice"] == "C" and out["fc_correct"] == 1.0 and out["fc_lp"][2] > out["fc_lp"][1] > out["fc_lp"][0]
          and out["vote_rows"] == 5)
    from sievelib import tasks_ruler as TR
    import run_r8 as RR
    import run_s1h as RH
    before = (TR.TASKS, TR.build, TR.score, TR.generation_limit, s1d_lib.answer_tokens, RR.answer_positions,
              RH.tfm_of_h, s1d_lib.RowBuffer.append)
    D.install_tasks()
    try:
        inside = (set(T.TASKS) <= set(TR.TASKS) and TR.generation_limit("lbv2", {}) == 1
                  and TR.generation_limit("msmarco_rerank_psg", {}) == 200 and RH.tfm_of_h is D.tfm_of_r4)
        D.CUR.task = "icl_banking77"
        z = RR.answer_positions(CharTok(), "abc", ["1"], 4)
        D.CUR.task = None
    finally:
        D.uninstall_tasks()
    after = (TR.TASKS, TR.build, TR.score, TR.generation_limit, s1d_lib.answer_tokens, RR.answer_positions,
             RH.tfm_of_h, s1d_lib.RowBuffer.append)
    check("plumbing: installed serves R4's tasks; no needle mask outside RAG; uninstalled restores everything",
          inside and not bool(z.any()) and before == after)


# ------------------------------------------------------------- the reader
def _write(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _fake_block(root, job, cell, pts, effect, acc_of, peak=64.0):
    """One R4 block of `cell`: dP = effect(arm, B, task, p); accuracy = acc_of(arm, B, task, p)."""
    import pandas as pd
    cc = L.CELLS[cell]
    preset = cc["preset"]
    pr = L.PRESETS[preset]
    plan = L.build_plan(pr)
    man = T.manifest()["cells"][L.manifest_cell(preset, cc["suite"])]
    rows = []
    for p, task in pts:
        Cn = 60000 + 1000 * p
        for ai, (arm, B) in enumerate(plan):
            pa = L.parse_arm(arm)
            fam, v = pa["family"], pa["v_bits"]
            extra = {}
            if fam in ("fp", "fpnoise") or arm == "fp+v4":
                kb, f, stored = 16.0, 0.0, None
            elif fam == "closedbook":
                kb, f, stored = 0.0, 1.0, (0.0, 1.0)
            elif fam == "dense":
                kb, f, stored = float(B), 0.0, None
            elif fam in ("fp8kv", "kivi", "kvquant"):
                kb, f, stored = float(pa["store"]), 0.0, None
            else:
                r = L.floor_r(Cn) if pa.get("floor") else B
                rf = L1C.qread_keep_count(r, Cn) / Cn
                w = float(pa["tier2_bits"]) if fam == "qread2t" else float(pa["store"])
                kb, f, stored = w * rf, 1 - rf, (float(pa["store"]), 0.0)
                if fam == "qread2t":
                    extra = dict(tier2=pa["tier2"], tier2_bits=pa["tier2_bits"], read_v_bits=pa["read_v_bits"],
                                 keys_only=not pa["kv"], requestion=pa["requestion"], store_width=pa["store"])
                if pa.get("floor"):
                    extra["floor_r"] = r
                if fam == "quest":
                    extra = dict(quest_read_frac_all=B + 0.001, quest_pages=10)
            dn = effect(arm, B, task, p)
            sc = acc_of(arm, B, task, p)
            kls = 0.0 if arm == "fp" else abs(dn) / 2 + 1e-4
            row = dict(model=pr["model"], ctx=pr["ctx"], task=task, prompt_idx=p, q_role="", ctx_len=Cn, window=32,
                       n_question_tokens=300, n_context_tokens=Cn, corpus_sha="c0ffee", corpus_doc=man[task][p][0],
                       head_dim=128, t_prefill=60.0, rot_seed=0, arm=arm, B=B, family=fam, base_arm=pa["base"],
                       twin=pa["twin"], lens=pa["lens"], v_bits=float(v), v_side=L.v_side(v), bits_per_token=kb,
                       evict_frac=f, key_side=L.key_side_bits(fam, f), read_frac=1 - f,
                       kept_width=(kb / (1 - f)) if f < 1 else 0.0, needle_keep=float("nan"),
                       stored_bits_per_token=stored[0] if stored else kb, stored_evict_frac=stored[1] if stored else f,
                       score=sc, hits=int(sc >= 1), n_expected=1, distractor=False,
                       pred="ABCD"[int(p % 4)] if task == "lbv2" else "x", gen_len=1, fp_gen_len=1, t_arm=2.0,
                       t_tf=0.3, t_own=0.0, t_precompute=10.0, tf_len=1, tf_top1=1.0, a_len=1, a_sum_nll=0.4 + dn,
                       tf_sum_nll=0.4 + dn + 0.001 * ai + 0.002 * bool(pa["twin"]), peak_gib=peak, peak_gib_arm=63.0,
                       base_gib_arm=62.0, stop_rule=pr["stop"], tf_logp=[-0.4 - dn - (0.001 if fam == "fpnoise" else 0)],
                       tf_vmask="1", span_end=1, a_span_nll=0.41 + dn, a_own_nll=0.4 + dn, a_span_own_nll=0.41 + dn,
                       a_set_nll=0.4 + dn, s_set_nll=0.41 + dn, ans_order="0", fp_ans_order="0", ans_reordered=False,
                       own_replay=False, a2_check=float("nan"), kl_all=kls * 2, kl_span=kls, kl_val=kls, kl_mean=kls,
                       kl_span_max=kls, tf_kl=[kls], vote_rows=32 if task == "lbv2" else -1, **extra)
            if task == "lbv2":
                ch = "ABCD"[int(p % 4)] if (sc >= 1 or arm == "fp") else "ABCD"[int((p + 1) % 4)]
                gold = "ABCD"[int(p % 4)] if acc_of("fp", 0.0, task, p) >= 1 else "ABCD"[int((p + 2) % 4)]
                lp = [-3.0] * 4
                lp["ABCD".index(ch)] = -0.1
                row.update(fc_lp=lp, fc_choice=ch, fc_correct=float(ch == gold), fc_gold_logp=lp["ABCD".index(gold)])
            if fam == "fpnoise":
                row["noise_chunk"] = 2048
            rows.append(row)
    d = os.path.join(root, f"r14s1h_{preset}_{job}")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(d, "s1h_evaluate_x_1.parquet"))
    _write(os.path.join(d, "s1h_evaluate_x_1.json"),
           dict(plan=[list(x) for x in plan], preset=pr, preset_name=preset, model=pr["model"], ctx=pr["ctx"],
                prompt_tasks=[list(x) for x in pts], stop_rule=pr["stop"], n_kv_heads=8, peak_gib_dev_max=[peak],
                parquet="s1h_evaluate_x_1.parquet", stage="1h", amend=L.AMEND, amend_1h=L.AMEND_1H,
                amend_r4=L.AMEND_R4, suite=cc["suite"], manifest_cell=L.manifest_cell(preset, cc["suite"]),
                manifest_sha256=T.MANIFEST_SHA256, smoke=False, tasks=list(cc["tasks"]), rot_seed=0))


def test_reader_synthetic():
    print("\n[S1h R4] read_stage1h_r4.py on synthetic blocks with known answers")
    import read_stage1h_r4 as R
    tmp = tempfile.mkdtemp(prefix="s1h4_reader_")
    try:
        # lbv2: FP right on p % 5 in (0, 1); closed book right only on p % 10 == 0; the system keeps FP's choices,
        # Quest loses every other FP-right item; the 3-bit dense arms lose all of them
        def acc_of(a, B, t, p):
            fp_ok = p % 5 in (0, 1)
            if a == "closedbook":
                return 1.0 if p % 10 == 0 else 0.0
            if a == "quest_v16" and fp_ok and p % 2 == 0:
                return 0.0
            if float(B) == 3.0 and a.startswith("uniform"):
                return 0.0
            return 1.0 if fp_ok else 0.0

        def eff(a, B, t, p):
            if a == "fp":
                return 0.0
            if a == "closedbook":
                return 3.0 + 0.1 * (p % 3)
            if a.startswith("quest"):
                return 0.5 + 0.01 * (p % 3)
            return 0.25 if float(B) == 3.0 else 0.01 + 0.001 * (p % 4)

        for job, off in (("1", 0), ("2", 20)):
            _fake_block(tmp, job, "lb2llama", [(off + k, "lbv2") for k in range(20)], eff, acc_of)
        stem = os.path.join(tmp, "findings", "R4_reader")
        rc = R.read_r4({"lb2llama": ["1", "2"]}, stem, root=tmp, r1_json=os.path.join(tmp, "absent.json"))
        out = json.load(open(stem + ".json"))
        s = out["summary"]
        f = out["cells"]["lb2llama"]["families"]["lbv2"]
        check("lbv2: FP 0.4 kept (no FP floor); the system near FP and better than Quest; CTX units (FP right, closed "
              "book wrong: 12) labelled; the gold log-prob change and the 4-choice KL reported; the cell's own margin "
              "(R1's read absent)",
              rc == 0 and abs(f["accuracy"]["fp"] - 0.4) < 1e-9 and f["n_ctx"] == 12
              and s["ACC_SYSTEM lb2llama/lbv2"].startswith("ACC_NEAR_FP")
              and s["SYS_VS_QUEST_ACC lb2llama/lbv2"].startswith("SYS_VS_QUEST_ACC_HELPS")
              and s["SYS_VS_QUEST lb2llama/lbv2"].startswith("SYS_VS_QUEST_HELPS")
              and "SYS_VS_QUEST_ACC_CTX lb2llama/lbv2" in s and f["gold_logp"] and f["fc_kl"]
              and "cell's margin" in s["SYSTEM NLL lb2llama/lbv2"] and "R1's" not in s["SYSTEM NLL lb2llama/lbv2"],
              f"({ {k: v for k, v in s.items() if 'QUEST' in k or 'FP accuracy' in k} })")
        # HELMET: RAG kept near FP by the floor; ICL and re-rank families labelled separately
        acc_h = lambda a, B, t, p: 0.0 if a == "closedbook" else 1.0  # noqa: E731
        eff_h = lambda a, B, t, p: 0.0 if a == "fp" else (2.0 if a == "closedbook" else 0.02)  # noqa: E731
        for job, off in (("11", 0), ("12", 5)):
            _fake_block(tmp, job, "hmllama", [(off + k, t) for k in range(5) for t in T.TASKS_HM], eff_h, acc_h)
        rc = R.read_r4({"hmllama": ["11", "12"]}, stem + "h", root=tmp, r1_json=os.path.join(tmp, "absent.json"))
        oh = json.load(open(stem + "h.json"))
        check("HELMET: one analysis per family (rag, rerank, icl), never pooled; every unit needs the context; one "
              "cluster per (task, item)",
              rc == 0 and set(oh["cells"]["hmllama"]["families"]) == {"rag", "rerank", "icl"}
              and oh["cells"]["hmllama"]["families"]["rag"]["n_ctx"] == 20
              and oh["cells"]["hmllama"]["families"]["rag"]["accuracy"]["n_units"] == 20)
        _fake_block(tmp, "p1", "hmllama", [(10, "msmarco_rerank_psg"), (10, "icl_banking77")], eff_h, acc_h)
        g_ok = R.read_gate("hmllama", "p1", root=tmp, out_dir=tmp) == 0
        _fake_block(tmp, "p2", "hmllama", [(10, "msmarco_rerank_psg"), (10, "icl_banking77")], eff_h, acc_h, peak=80.0)
        g_bad = R.read_gate("hmllama", "p2", root=tmp, out_dir=tmp) == 1
        check("gate: PASS within memory and wall time; FAIL above 76 GiB per device", g_ok and g_bad)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------- driver smoke
def _drive(args, log):
    env = dict(os.environ, OMP_NUM_THREADS="8", H0_CORPUS=CORPUS, PYTHONUNBUFFERED="1")
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "run_s1h4.py")] + args, stdout=fh,
                           stderr=subprocess.STDOUT, env=env, cwd=ROOT, timeout=3300)
    return r.returncode


def test_driver_smoke():
    print("\n[S1h R4] driver smokes (CPU, Llama-3.2-1B at 4K, items cut in the middle): lbv2; kilt_nq + icl_trec_coarse")
    import read_stage1h_r4  # noqa: F401  (installs R4's validity into read_stage1h)
    import read_stage1h as R1R
    tmp = tempfile.mkdtemp(prefix="s1h4_drive_")
    lov = ["--model", "llama31-8b", "--override", "id=meta-llama/Llama-3.2-1B-Instruct", "dtype=float32", "tier=smoke"]
    try:
        for nm, tasks in (("lb2", "lbv2"), ("hm", "kilt_nq,icl_trec_coarse")):
            rc = _drive(["--mode", "evaluate", "--preset", "h4smoke", "--ctx", "4096", "--n-prompts", "1",
                         "--prompt-offset", "0", "--tasks", tasks, "--out-dir", os.path.join(tmp, f"r14s1h_h4smoke_{nm}")]
                        + lov, os.path.join(tmp, f"{nm}.log"))
            if rc != 0:
                check(f"{nm} smoke ran", False, f"(rc {rc})")
                print(open(os.path.join(tmp, f"{nm}.log")).read()[-4000:])
                continue
            d, side = R1R.load_run("h4smoke", nm, tmp)
            probs = []
            R1R.validate_h(d, [side], probs, main=False)
            R1R.validate_a2_h(d, probs, "smoke", self_check=False)
            fp = d[d.arm == "fp"]
            q = d[d.family == "quest"]
            cb = d[d.arm == "closedbook"]
            good = (len(side["plan"]) == len(L.build_plan(L.PRESETS["h4smoke"])) and not probs and (fp.kl_all == 0).all() and side.get("amend_r4") == "R4b"
                    and side.get("smoke") and len(q) == 2 * len(fp) and (q.quest_steps >= 1).all()
                    and (np.abs(q.quest_read_frac_all - q.B) <= 0.01).all() and len(cb) == len(fp)
                    and (cb.kl_all > 0).all()
                    and (cb.closedbook_ctx_tokens.to_numpy() < fp.n_context_tokens.to_numpy()).all()
                    and (fp.a_len > 0).all())
            if nm == "lb2":
                good &= (d.fc_correct.notna().all() and (fp.max_new_tokens == 1).all() and (d.vote_rows == 32).all())
            check(f"{nm} smoke: every arm (the 4K preset adds the floor system at r = 1); validity passes; Quest at its budget; the closed-book arm on its own short "
                  f"cache, away from FP" + ("; forced choice and a 32-row vote span" if nm == "lb2" else ""),
                  good, f"({probs}; FP {fp.set_index('task').score.round(2).to_dict()}; "
                        f"closed book {cb.set_index('task').score.round(2).to_dict()})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fast = "--fast" in sys.argv
    tests = [test_plans, test_tasks, test_driver_parts, test_reader_synthetic]
    if not fast:
        tests += [test_driver_smoke]
    for t in tests:
        t()
    print(f"\n{'ALL R14 STAGE-1H R4 TESTS PASSED' if not fails else f'{fails} R14 STAGE-1H R4 TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)
