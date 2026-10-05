#!/usr/bin/env python3
"""R14 Stage 1h R4 driver (design: s1h4_lib.py, plan.md; tasks: tasks_s1h4.py; frozen
rules: read_stage1h_r4.py).

run_s1h2.py's driver (every R1-R3 arm, KL, per-arm memory) with s1h4_lib's presets,
tasks_s1h4's tasks, and three new arm kinds (s1h4_lib's docstring). Installed for the run
in this process only and removed after it (sievelib, run_r8 and the Stage 1g modules are
not edited):
  tasks        tasks_ruler.TASKS / build / score / generation_limit also serve lbv2 and the
               HELMET tasks (only these may be run here); items from the manifest cell of
               the preset's model and length;
  masks        s1d_lib.answer_tokens = tasks_s1h4.answer_span (FP's first line);
               run_r8.answer_positions = the gold answers' whole-word occurrences (RAG; none
               for the other tasks);
  the vote     s1d_lib.RowBuffer.append keeps, of a question prefill, only the rows of the
               unit's vote span (tasks_s1h4.vote_rows; lbv2), and every later row (the
               oracle's answer rows);
  forced choice  run_s1h.tfm_of_h also records, for lbv2, the log-probabilities of A/B/C/D
               at the answer position of the arm's teacher-forced replay (fc_lp), its choice
               (fc_choice) and whether it is the gold (fc_correct);
  stop rule    'r8list' (stops_s1h.py);
  quest arms   a per-step page selection inside the attention call (decode only);
               teacher-forced one token at a time;
  floor arm    the system at r = s1h4_lib.floor_r(C);
  closedbook   FP on the closed-book prompt, on its own cache.

    python run_s1h4.py --mode evaluate --preset h4llama128 --ctx 131072 --tasks lbv2 --n-prompts 20 \\
        --prompt-offset 0 --out-dir DIR
"""
from __future__ import annotations
import gc, json, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import run_s1h2 as RH2  # noqa: E402  (sets up every other path; imports run_s1h)
import run_s1h as RH  # noqa: E402
import run_r8 as RR  # noqa: E402
import run_s1e as S1E  # noqa: E402
import run_s1f as S1F  # noqa: E402
import s1b_lib as L1B  # noqa: E402
import s1d_lib  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
from sievelib import compress as C, tasks_ruler as TR, tasks_longbench_v2 as LB2  # noqa: E402
from sievelib.probe import cache_kv  # noqa: E402
import s1h4_lib as L  # noqa: E402
import stops_s1h as STOPS  # noqa: E402
import tasks_s1h4 as T  # noqa: E402
import tasks_s1h as T3  # noqa: E402  (R3b's whole-word positions)

_ORIG = dict(TASKS=TR.TASKS, build=TR.build, score=TR.score, generation_limit=TR.generation_limit,
             answer_tokens=s1d_lib.answer_tokens, query_term=s1d_lib.query_term,
             answer_positions=RR.answer_positions, tfm_of_h=RH.tfm_of_h, append=s1d_lib.RowBuffer.append)


class _Cur:
    """The unit being run (run_s1e.main builds, then runs every arm of, one unit at a time)."""
    task = gold = branches = None
    vote = None             # bool mask over the question's tokens, or None (the last rows)
    n_q = 0
    cb_ctx = None
    preset = model = None
    smoke = False


CUR = _Cur()


def _cell(task):
    suite = T.suite_of(task)
    p = L.PRESETS[CUR.preset]
    return T.cell_key(suite, p["model"], L.CTX_R4 if CUR.smoke else p["ctx"])


def build_h(tok, task, ctx, *, prompt_idx, corpus_dir=None, require_real=False, **ruler_cfg):
    if task not in T.TASKS:
        return _ORIG["build"](tok, task, ctx, prompt_idx=prompt_idx, corpus_dir=corpus_dir,
                              require_real=require_real, **ruler_cfg)
    text, meta = T.build(tok, task, ctx, prompt_idx=prompt_idx, cell=_cell(task), model=CUR.model,
                         allow_truncate=CUR.smoke)
    CUR.task, CUR.gold, CUR.cb_ctx = task, meta["expected"][0], meta["cb_ctx_text"]
    CUR.branches = [LB2.choice_logit_contract(tok)[1][c] for c in "ABCD"] if task == T.LBV2 else None
    CUR.n_q = len(tok(meta["question"], add_special_tokens=False).input_ids)
    CUR.vote = None
    if meta["vote_span"] is not None:
        m = torch.zeros(CUR.n_q, dtype=torch.bool)
        m[T.vote_rows(meta["vote_span"])] = True
        CUR.vote = m
    return text, meta


def score_h(task, pred, meta):
    return T.score(task, pred, meta) if task in T.TASKS else _ORIG["score"](task, pred, meta)


def generation_limit_h(task, config):
    return T.generation_limit(task) if task in T.TASKS else _ORIG["generation_limit"](task, config)


def answer_tokens_h(tok, ids, expected):
    return T.answer_span(tok, ids) if CUR.task in T.TASKS else _ORIG["answer_tokens"](tok, ids, expected)


def query_term_h(task, meta):
    return "" if task in T.TASKS else _ORIG["query_term"](task, meta)


def answer_positions_h(tok, text, expected, ctx_len):
    if CUR.task not in T.TASKS:
        return _ORIG["answer_positions"](tok, text, expected, ctx_len)
    if T.FAMILY[CUR.task] != "rag":
        return torch.zeros(ctx_len, dtype=torch.bool)
    return T3.answer_positions_wb(tok, text, expected, ctx_len)


def append_h(self, li, q, pos):
    """s1d_lib.RowBuffer.append; of a question prefill, only the vote span's rows (module
    docstring). The first row a buffer sees on a layer is the question's first token."""
    if CUR.vote is not None and self.phase == "question":
        base = self.__dict__.setdefault("_vbase", {}).setdefault(li, int(pos[0]))
        rel = pos.to("cpu") - base
        inq = rel < CUR.n_q - 1
        keep = ~inq
        keep[inq] = CUR.vote[rel[inq]]
        if not bool(keep.any()):
            return None
        keep = keep.to(q.device)
        q, pos = q[keep], pos[keep.to(pos.device)]
    return _ORIG["append"](self, li, q, pos)


def tfm_of_r4(lg, fp_gen, content, am):
    """run_s1h.tfm_of_h, plus the lbv2 forced choice and the vote span (module docstring)."""
    out = _ORIG["tfm_of_h"](lg, fp_gen, content, am)
    if out is None:
        return out
    out["vote_rows"] = int(CUR.vote.sum()) if CUR.vote is not None else -1
    if CUR.task == T.LBV2 and lg is not None and len(lg):
        lp = torch.log_softmax(lg[0].float(), -1)[torch.tensor(CUR.branches, device=lg.device)]
        fc = [round(float(x), 5) for x in lp.tolist()]
        ch = "ABCD"[int(max(range(4), key=lambda i: fc[i]))]
        out.update(fc_lp=fc, fc_choice=ch, fc_correct=float(ch == CUR.gold), fc_gold_logp=fc["ABCD".index(CUR.gold)])
    return out


# ---------------------------------------------------------------- Quest
class _Quest:
    def __init__(self):
        self.active, self.meta, self.n_pages, self.width, self.reads, self.rows = False, {}, 0, 16, [], 0


Q = _Quest()


def quest_setup(past, nL, r, width):
    """Page metadata (min, max key per page and channel) of the selected layers, from the
    cache's exact keys, and the page budget (s1h4_lib.quest_budget_rows)."""
    Cn = C.STATE.ctx_len
    P = L.QUEST_PAGE
    Q.rows = L.quest_budget_rows(r, Cn, nL)
    Q.n_pages, Q.width, Q.meta, Q.reads = L.quest_pages(Q.rows, Cn), int(width), {}, []
    Np = -(-Cn // P)
    for li in range(L.QUEST_DENSE_LAYERS, nL):
        K, _ = cache_kv(past, li)
        Kc = K[:, :Cn]
        if Np * P > Cn:
            Kc = torch.cat([Kc, Kc[:, -1:].expand(-1, Np * P - Cn, -1)], 1)
        Kp = Kc.reshape(Kc.shape[0], Np, P, Kc.shape[-1])
        Q.meta[li] = (Kp.amin(2), Kp.amax(2))


def quest_select(li, q):
    """Quest's selection for one layer at one decode step; q [H, d] the step's queries."""
    kmin, kmax = Q.meta[li]
    Hkv, Np, d = kmin.shape
    Cn = C.STATE.ctx_len
    qf = q.float().reshape(Hkv, q.shape[0] // Hkv, d).to(kmin.device)
    ub = (torch.einsum("grd,gpd->grp", qf.clamp(min=0), kmax.float())
          + torch.einsum("grd,gpd->grp", qf.clamp(max=0), kmin.float())).amax(1)        # [Hkv, Np]
    top = ub.topk(min(Q.n_pages, Np), dim=-1).indices
    pm = torch.zeros(Hkv, Np, dtype=torch.bool, device=ub.device)
    pm.scatter_(1, top, True)
    keep = pm.repeat_interleave(L.QUEST_PAGE, dim=1)[:, :Cn]
    dev = C.STATE.evict[li].device
    C.STATE.evict[li] = ~keep.to(dev)
    C.STATE.bits[li] = torch.where(keep, Q.width, 0).long().to(dev)
    Q.reads.append(float(keep.float().mean()))


def _attn_quest(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kwargs):
    li = C._layer(module)
    if Q.active and query.shape[2] == 1 and C.STATE.enabled and li in Q.meta:
        quest_select(li, query[0, :, 0, :])
    return C.sieve_compress_attention(module, query, key, value, attention_mask=attention_mask, scaling=scaling,
                                      dropout=dropout, **kwargs)


def _quest_store(pa, alloc):
    return None if int(pa["store"]) == L.EXACT_WIDTH else alloc[("uniform", L.STORE4)]


def _dense_view(width):
    for li in list(C.STATE.evict):
        C.STATE.evict[li] = torch.zeros_like(C.STATE.evict[li])
        C.STATE.bits[li] = torch.full_like(C.STATE.bits[li], int(width))


def run_quest(model, past, q_ids, store, width, r, R, norm_correct, vfn, eos, max_new, L0, tok, nL):
    C.crop_to(past, L0)
    C.STATE.reset_arm()
    S1E.apply_store(past, store, width, R, norm_correct, vfn, nL)
    quest_setup(past, nL, r, width)
    ALL_ATTENTION_FUNCTIONS[C.IMPL] = _attn_quest
    try:
        past = RR._question(model, past, q_ids)          # dense: Quest selects while decoding only
        Q.active = True
        gen, past = RR._decode(model, past, q_ids[0, -1], max_new, eos, tok)
    finally:
        Q.active = False
        C.install()
        C.STATE.enabled = False
    return gen, past


def tf_quest(model, past, L0, q_ids, ids, width):
    """The same arm teacher-forced on `ids`, one token per call (a selection per step)."""
    if not ids:
        return None
    C.crop_to(past, L0)
    _dense_view(width)
    C.STATE.enabled = True
    ALL_ATTENTION_FUNCTIONS[C.IMPL] = _attn_quest
    lgs = []
    try:
        with torch.no_grad():
            if q_ids.shape[1] > 1:
                model(q_ids[:, :-1], past_key_values=past, use_cache=True)
            Q.active = True
            for t in [int(q_ids[0, -1])] + [int(x) for x in ids[:-1]]:
                inp = torch.tensor([[t]], device=q_ids.device, dtype=q_ids.dtype)
                lgs.append(model(inp, past_key_values=past, use_cache=True).logits[0].float())
    finally:
        Q.active = False
        C.install()
        C.STATE.enabled = False
    C.crop_to(past, L0)
    return torch.cat(lgs, 0)[:len(ids)]


def run_quest_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, norm_correct, eos, max_new,
                  L0, tok):
    """One Quest arm (run_s1e.run_arm's return tuple), with A2's columns."""
    width = int(pa["store"])
    store = _quest_store(pa, alloc)
    vfn = L1B.v_quantizer(pa["v_bits"], Rv, norm_correct) if pa["v_bits"] < 16 else None
    t1 = time.time()
    gen, past = run_quest(model, past, q_ids, store, width, B, R, norm_correct, vfn, eos, max_new, L0, tok, nL)
    t_arm = time.time() - t1
    reads, rows, n_pages = list(Q.reads), Q.rows, Q.n_pages
    au = C.bits_audit()
    nk_state = {li: e.clone() for li, e in C.STATE.evict.items()}
    tt = time.time()
    lg = tf_quest(model, past, L0, q_ids, fp_gen, width)
    tfm = RH.tfm_of_h(lg, fp_gen, content, am)
    del lg
    t_tf = time.time() - tt
    S1F.a2_defaults(tfm, am)
    replay = lambda ids: tf_quest(model, past, L0, q_ids, ids, width)  # noqa: E731
    tfm.update(S1F.a2_columns(tok, fp_gen, am, gen, tfm, replay, (pa["family"], pa["twin"], False, arm)))
    stored = dict(stored_bits_per_token=float(width), stored_evict_frac=0.0)
    n_sel = nL - L.QUEST_DENSE_LAYERS
    mean_sel = float(sum(reads) / len(reads)) if reads else float("nan")
    extra = dict(store_width=width, quest_page=L.QUEST_PAGE, quest_pages=n_pages, quest_rows_layer=rows,
                 quest_dense_layers=L.QUEST_DENSE_LAYERS, quest_steps=len(reads) // max(n_sel, 1),
                 quest_read_frac_sel=mean_sel, quest_read_frac_all=(n_sel * mean_sel + L.QUEST_DENSE_LAYERS) / nL)
    return gen, past, tfm, stored, extra, None, t_arm, t_tf, (arm, B), au, nk_state


# ------------------------------------------------------------- closed book
def run_closedbook_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, eos, max_new, tok):
    """FP on the closed-book prompt (the unit's cb_ctx_text, then the same question tokens),
    on its own cache; the main cache is not touched. Its teacher-forced replay of FP's answer
    goes through the original tf_phased, which does not overwrite FP's KL reference."""
    saved = (C.STATE.enabled, C.STATE.capture, C.STATE.h2o)
    C.STATE.enabled, C.STATE.capture, C.STATE.h2o = False, False, False
    cb_ids = tok(CUR.cb_ctx, return_tensors="pt").input_ids.to(q_ids.device)
    try:
        t1 = time.time()
        with torch.no_grad():
            pcb = model(cb_ids, use_cache=True).past_key_values
        lcb = C.cache_len(pcb)
        pcb = RR._question(model, pcb, q_ids)
        gen, pcb = RR._decode(model, pcb, q_ids[0, -1], max_new, eos, tok)
        t_arm = time.time() - t1
        tt = time.time()
        lg = RH._ORIG["tf_phased"](model, pcb, lcb, q_ids, fp_gen, False)
        tfm = RH.tfm_of_h(lg, fp_gen, content, am)
        del lg
        t_tf = time.time() - tt
        S1F.a2_defaults(tfm, am)
        replay = lambda ids: RH._ORIG["tf_phased"](model, pcb, lcb, q_ids, ids, False)  # noqa: E731
        tfm.update(S1F.a2_columns(tok, fp_gen, am, gen, tfm, replay, ("closedbook", "", False, arm)))
    finally:
        C.STATE.enabled, C.STATE.capture, C.STATE.h2o = saved
        C.STATE.enabled = False
    del pcb
    gc.collect()
    au = {"bits_per_token": 0.0, "evict_frac": 1.0}
    stored = dict(stored_bits_per_token=0.0, stored_evict_frac=1.0)
    nk = {0: torch.ones(1, max(C.STATE.ctx_len, 1), dtype=torch.bool)}
    return gen, past, tfm, stored, dict(closedbook_ctx_tokens=int(lcb)), None, t_arm, t_tf, (arm, B), au, nk


def run_arm_h4(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, protect, nL, R, Rv, norm_correct,
               eos, max_new, L0, tok, last, sel_q1=None):
    """run_s1h2.run_arm_h2, a Quest arm, the floor system or the closed-book arm, with
    per-arm memory."""
    if pa.get("floor"):
        r = L.floor_r(C.STATE.ctx_len)
        base = L.SYSTEM + "_v4"
        res = RH2.run_arm_h2(model, past, base, r, L.parse_arm(base), q_ids, fp_gen, content, am, alloc, protect, nL,
                             R, Rv, norm_correct, eos, max_new, L0, tok, last, sel_q1)
        res[4].update(floor_r=r)
        return res
    if pa["family"] not in ("quest", "closedbook"):
        return RH2.run_arm_h2(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, protect, nL, R, Rv,
                              norm_correct, eos, max_new, L0, tok, last, sel_q1)
    RH._fold()
    RH._ORIG["reset_peaks"]()
    base = (sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30
            if torch.cuda.is_available() else float("nan"))
    if pa["family"] == "quest":
        res = run_quest_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, alloc, nL, R, Rv, norm_correct, eos,
                            max_new, L0, tok)
        Q.meta = {}
    else:
        res = run_closedbook_arm(model, past, arm, B, pa, q_ids, fp_gen, content, am, eos, max_new, tok)
    pk = RH._dev_peaks()
    RH._fold()
    res[4].update(peak_gib_arm=sum(pk) if pk else float("nan"), base_gib_arm=base)
    return res


# ------------------------------------------------------------- install
def install_tasks():
    TR.TASKS = tuple(_ORIG["TASKS"]) + T.TASKS
    TR.build, TR.score, TR.generation_limit = build_h, score_h, generation_limit_h
    s1d_lib.answer_tokens, s1d_lib.query_term = answer_tokens_h, query_term_h
    s1d_lib.RowBuffer.append = append_h
    RR.answer_positions = answer_positions_h
    RH.tfm_of_h = tfm_of_r4                       # before RH.install(), which binds S1E.tfm_of / S1F.tfm_of_f to it


def uninstall_tasks():
    TR.TASKS, TR.build, TR.score, TR.generation_limit = (_ORIG["TASKS"], _ORIG["build"], _ORIG["score"],
                                                         _ORIG["generation_limit"])
    s1d_lib.answer_tokens, s1d_lib.query_term = _ORIG["answer_tokens"], _ORIG["query_term"]
    s1d_lib.RowBuffer.append = _ORIG["append"]
    RR.answer_positions = _ORIG["answer_positions"]
    RH.tfm_of_h = _ORIG["tfm_of_h"]
    CUR.task = CUR.gold = CUR.branches = CUR.vote = CUR.cb_ctx = None


def _arg(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def _mark_outputs(out_dir, mode, tasks):
    if not os.path.isdir(out_dir):
        return
    for f in os.listdir(out_dir):
        if f.startswith(f"s1h_{mode}_") and f.endswith(".json"):
            p = os.path.join(out_dir, f)
            side = json.load(open(p))
            side.update(lib="s1h4_lib", amend_r4=L.AMEND_R4, suite=T.suite_of(tasks[0]), manifest_cell=_cell(tasks[0]),
                        manifest_sha256=T.MANIFEST_SHA256, tasks_gen=T.GEN_VERSION, smoke=CUR.smoke,
                        quest=dict(page=L.QUEST_PAGE, dense_layers=L.QUEST_DENSE_LAYERS, meta_bits=L.QUEST_META_BITS),
                        floor=dict(k_min=L.K_MIN, nominal_r=L.FLOOR_NOMINAL_R), vote_rows=T.VOTE_ROWS,
                        driver="run_s1h4.py (run_s1h2.py's driver with s1h4_lib presets, LongBench v2 and HELMET, "
                               "Quest, the floor system and the closed-book arm)")
            with open(p, "w") as fh:
                json.dump(side, fh, indent=1, default=str)


def main():
    argv = sys.argv[1:]
    mode, out_dir, preset = _arg(argv, "--mode"), _arg(argv, "--out-dir"), _arg(argv, "--preset")
    if mode and mode != "evaluate":
        raise SystemExit("Stage 1h runs evaluation blocks only (no calibration, no reuse)")
    tasks = [t for t in (_arg(argv, "--tasks") or "").split(",") if t]
    plist = _arg(argv, "--prompt-list")
    if plist:
        tasks += [x.split(":")[1] for x in plist.split(",") if x]
    tasks = list(dict.fromkeys(tasks))
    if not tasks or set(tasks) - set(T.TASKS) or len({T.suite_of(t) for t in tasks}) != 1:
        raise SystemExit(f"run_s1h4.py runs R4's tasks only, one suite per block ({T.TASKS}); pass --tasks")
    if preset not in L.PRESETS or "closedbook" not in L.PRESETS[preset]:
        raise SystemExit(f"--preset must be an R4 preset (h4*), not {preset!r}")
    CUR.preset, CUR.model = preset, L.PRESETS[preset]["model"]
    CUR.smoke = preset.endswith("smoke") or any(x.startswith("tier=smoke") for x in argv)
    S1F._A2_CHECKED.clear()
    install_tasks()
    STOPS.install(S1E, RR)
    RH.install()
    S1E.L, S1E.run_arm = L, run_arm_h4
    try:
        S1E.main()
    finally:
        RH.uninstall()
        STOPS.uninstall(S1E, RR)
        uninstall_tasks()
    if out_dir and mode:
        RH._rename_outputs(out_dir, mode)
        _mark_outputs(out_dir, mode, tasks)


if __name__ == "__main__":
    main()
