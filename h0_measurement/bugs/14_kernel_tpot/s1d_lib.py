"""s1d_lib.py -- R14 Stage 1d building blocks (pure functions, CPU-testable).

Stage 1c (jobs 1005224-1005236) ended GO_KERNEL under its frozen rule. Three
things in its analysis shape this stage.
  - Its continuous metric also scored FP's text AFTER the answer. For vt, FP
    always continues past the five names, copying book text for up to 64 tokens.
    So the metric rewarded copying fidelity as well as answer quality.
  - Protecting a few structural KV heads fixed most router failures. Those
    heads attend little to the answer at the answer step.
  - Evicting the same share of every head (the value-aware hybrid as built)
    failed; head-adaptive masks did not.
Driver: run_s1d.py. Frozen rules: read_stage1d.py.

1  METRIC (the fix). The primary quality measure is the teacher-forced NLL of
   the ANSWER VALUES in FP's greedy answer. These are the tokens that spell an
   expected answer string (the needle numbers; the vt variable names), counted
   up to the point where every expected value FP gives has appeared ("the
   answer span"). Format words, reorderings and everything after the span are
   excluded from it and reported separately:
   - s_*    content tokens inside the span;
   - post_* content tokens after it;
   - tf_c_* Stage 1c's all-content metric, for comparison.
   Every row also stores the per-token log-probabilities and the masks, so any
   later metric can be recomputed without a GPU. The sequence calibration's
   failure statistic becomes the worst answer-VALUE token (a_min_logp) instead
   of the worst content token.

2  DESIGNS (Llama-3.1-8B, 128K and 32K, fresh prompts).
   - router_seq2_calib: Stage 1c's sequence-calibrated router, recalibrated with
     the answer-value statistic (base R0 = Stage 1b's routes_pool, prompts 0-9).
   - router_topN_calib: a head BUDGET instead of a rescue threshold. The N
     dense heads are R0's dense heads, then the sequence-critical heads (by
     recurrence, then rescue gain), then the heads the per-prompt oracle keeps
     dense most often. N = 32 and 64 of 256 KV heads. This interpolates between
     seq2 (~16) and the union router (~80), which was matched in Stage 1c.
   - hvah_v16 / hvah_v4 (B = the router budget): head-aware value-aware hybrid.
     It keeps exactly router_seq2_calib@B's tokens (its head-adaptive mask), all
     at one width chosen by the value format: 4 bits with BF16 values, 3 bits
     with 4-bit values. That is where Stage 1c's fixed-width points put the
     budget line's optimum.
   - Question-time reads over the dense TurboQuant-3 store, in three variants:
     - qreadp_v*: the protected heads (the union of seq2's critical heads)
       read everything;
     - qreadr_v*: re-selection every RESEL_K answer steps, from the attention
       of the last RESEL_ROWS query rows (question rows, then generated
       tokens), which follows multi-hop chains the question never names. The
       scan it needs is charged: (d/8)(3 + 16/d)/k bytes per token per step;
     - qreadpr_v*: both.

3  MECHANISM of the critical heads (the setup-head hypothesis). The critical
   set is seq2's added heads at B_low. There are three diagnostic arms, each
   R0@B_low except for those heads:
   - mech_q: the critical heads are dense only while the QUESTION is prefilled
     (q_ids[:-1]);
   - mech_a: they are dense only while the ANSWER is generated (from the last
     question token on);
   - mech_p: the view is R0 throughout, but the critical heads' attention
     OUTPUTS at the question rows are replaced by the full-precision run's
     outputs (causal patching).
   With router_pool_calib@B_low (no rescue) and router_seq2_calib@B_low (rescue
   in both phases), each arm gives a rescue fraction. Setup heads predict
   f_q ~ 1 and f_a ~ 0; answer-time heads the reverse; global stabilisers both.
   The FP question pass also records every head's attention on the question's
   key term, the answer values, other needles, sink, prefix, window and
   question tokens (the "anatomy"), and the same at the first answer step.

4  SECOND MODEL: Qwen3-30B-A3B-Instruct-2507 at 32K (MoE; 48 x 4 KV heads, each
   shared by 8 query heads), with the same calibration, routers and mechanism
   arms. A calibration at 8K checks whether the critical heads are stable
   across lengths.

No shared file is edited. sievelib, run_r8 and the Stage 1b/1c modules are imported.
"""
from __future__ import annotations
import os
import re
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from sievelib import router  # noqa: E402
import s1b_lib as L1B  # noqa: E402
import s1c_lib as L1C  # noqa: E402

D_DEFAULT = 128
REF_WIDTH = L1C.REF_WIDTH                  # D = TurboQuant-3
STORE_WIDTH = L1C.QREAD_STORE_WIDTH
QREAD_ROWS = L1C.QREAD_ROWS                # rows in a (re-)selection vote = SnapKV's window
RESEL_K = 8                                # re-select every 8 answer steps
HVAH_WIDTH = {16: 4, 4: 3, 2: 3}           # kept-key width by value width (Stage 1c's budget line)
TAU_FAIL, TAU_CRIT, K_MAX = L1C.TAU_FAIL, L1C.TAU_CRIT, L1C.K_MAX
LENS = L1C.LENS
TWINS = dict(L1C.TWINS)
ANAT_CATS = ("value", "key", "other", "sink", "prefix", "rest", "window", "question")
SINK_TOKENS, PREFIX_TOKENS = 4, 32         # positions [0,4) = sink, [4,32) = instruction prefix

FAMILY_OF = {"fp": "fp", "uniform": "dense", "router_calib": "sieve",
             "router_pool_calib": "pool", "router_seq_calib": "seq",
             "router_seq2_calib": "seq2", "router_union_calib": "union",
             "router_pool_oracle": "oracle"}
ROUTERS = ("sieve", "pool", "seq", "seq2", "topn", "union", "oracle")
DESIGNS = ("seq2", "topn", "hvah", "qread")
REFERENCES = ("sieve", "pool", "seq", "union")
DIAGNOSTICS = ("oracle", "mech")

# ------------------------------------------------------------------ presets
# Frozen before any Stage 1d output. B for routers = key bits (fractional B uses
# uniform at floor(B) as the dense candidate); dense B = width; hvah B = the
# budget of the seq2 mask it copies; qread B = read fraction r; mech B = the
# budget whose R0 / seq2 pair it splits. 'calib' = the calibration budgets.
_Q4 = [(0.125, 4), (0.25, 4), (0.5, 4), (0.25, 16)]
PRESETS = {
    "main128": dict(
        model="llama31-8b", ctx=131072, B_low=3, B_top=4, calib=[3, 4],
        dense=[3, 4], dense_twins=["+v4"], fp_twins=["+v4"],
        sieve=[4], sieve_twins=["+v4"], pool=[3, 4], pool_twins=[],
        seq=[3], seq2=[3, 4], seq2_twins=["+v4"],
        topn=[32, 64], topn_twins=["+v4"], union=[3], union_twins=["+v4"], oracle=[],
        hvah=[(3, 16), (3, 4), (4, 16), (4, 4)], qread=_Q4,
        qreadp=[(0.125, 4), (0.25, 4)], qreadr=[(0.125, 4)], qreadpr=[(0.125, 4)], mech=[3]),
    "main32": dict(
        model="llama31-8b", ctx=32768, B_low=2.5, B_top=3, calib=[2.5, 3],
        dense=[3, 4], dense_twins=["+v4"], fp_twins=["+v4"],
        sieve=[3], sieve_twins=["+v4"], pool=[2.5, 3], pool_twins=[],
        seq=[2.5], seq2=[2.5, 3], seq2_twins=["+v4"],
        topn=[32, 64], topn_twins=["+v4"], union=[2.5], union_twins=["+v4"], oracle=[],
        hvah=[(2.5, 16), (2.5, 4), (3, 16), (3, 4)], qread=_Q4,
        qreadp=[(0.125, 4), (0.25, 4)], qreadr=[(0.125, 4)], qreadpr=[(0.125, 4)], mech=[2.5]),
    # mechanics only (excluded): one arm of every code path
    "pilot128": dict(
        model="llama31-8b", ctx=131072, B_low=3, B_top=4, calib=[3],
        dense=[3], dense_twins=["+v4"], fp_twins=["+v4"],
        sieve=[4], sieve_twins=[], pool=[3], pool_twins=[],
        seq=[3], seq2=[3], seq2_twins=["+v4"],
        topn=[32], topn_twins=["+v4"], union=[3], union_twins=[], oracle=[3],
        hvah=[(3, 4)], qread=[(0.25, 4)],
        qreadp=[(0.25, 4)], qreadr=[(0.125, 4)], qreadpr=[(0.125, 4)], mech=[3]),
    # Qwen3-30B-A3B-2507 replication (routers + mechanism; exact values only)
    "qwen32": dict(
        model="qwen3-30b-a3b-2507", ctx=32768, B_low=2.5, B_top=3, calib=[2.5, 3],
        dense=[3, 4], dense_twins=[], fp_twins=[],
        sieve=[3], sieve_twins=[], pool=[2.5, 3], pool_twins=[],
        seq=[], seq2=[2.5, 3], seq2_twins=[],
        topn=[], topn_twins=[], union=[2.5], union_twins=[], oracle=[2.5, 3],
        # mech at both budgets: Qwen's keys have outlier channels (R12/R13), so the rescue
        # candidate at 2.5 (TurboQuant-2) may not rescue; at 3 it is TurboQuant-3
        hvah=[], qread=[(0.25, 16)], qreadp=[], qreadr=[], qreadpr=[], mech=[2.5, 3]),
    "qwenpilot": dict(
        model="qwen3-30b-a3b-2507", ctx=32768, B_low=2.5, B_top=3, calib=[2.5],
        dense=[3], dense_twins=[], fp_twins=[],
        sieve=[], sieve_twins=[], pool=[2.5], pool_twins=[],
        seq=[], seq2=[2.5], seq2_twins=[],
        topn=[], topn_twins=[], union=[2.5], union_twins=[], oracle=[2.5],
        hvah=[], qread=[(0.25, 16)], qreadp=[], qreadr=[], qreadpr=[], mech=[2.5]),
    # calibration only: is the critical set stable across lengths?
    "qwen8cal": dict(
        model="qwen3-30b-a3b-2507", ctx=8192, B_low=2.5, B_top=3, calib=[2.5, 3],
        dense=[], dense_twins=[], fp_twins=[], sieve=[], sieve_twins=[], pool=[],
        pool_twins=[], seq=[], seq2=[], seq2_twins=[], topn=[], topn_twins=[], union=[],
        union_twins=[], oracle=[], hvah=[], qread=[], qreadp=[], qreadr=[], qreadpr=[], mech=[]),
}
PRESETS["smoke"] = dict(PRESETS["pilot128"], ctx=2048)          # CPU, Llama-3.2-1B; never submitted
PRESETS["qwensmoke"] = dict(PRESETS["qwenpilot"], ctx=2048)     # CPU, Qwen3-0.6B (Qwen3 attention); never submitted


def norm_b(B):
    return L1B.norm_b(B)


def floor_width(B) -> int:
    return L1B.floor_width(B)


def bk(B) -> str:
    return router.bkey(B)


# ------------------------------------------------------------------- arms
_QREAD_VAR = {"qread": (False, False), "qreadp": (True, False), "qreadr": (False, True),
              "qreadpr": (True, True)}


def parse_arm(arm: str) -> dict:
    """Family, value width and the design parameters an arm name carries."""
    s = L1B.twin_suffix(arm)
    base = arm[:-len(s)] if s else arm
    out = dict(base=base, twin=s, topn=None, protect=False, resel=False, phase=None,
               hvah_width=None)
    m = re.fullmatch(r"router_top(\d+)_calib", base)
    if m:
        v = L1C.TWINS[s] if s else 16
        return dict(out, family="topn", v_bits=v, lens=LENS[v], topn=int(m.group(1)))
    if base in FAMILY_OF:
        v = L1C.TWINS[s] if s else 16
        return dict(out, family=FAMILY_OF[base], v_bits=v, lens=LENS[v])
    m = re.fullmatch(r"(hvah|qread|qreadp|qreadr|qreadpr)_v(\d+)", base)
    if m and not s:
        v = int(m.group(2))
        if v not in HVAH_WIDTH and v != 16:
            raise ValueError(f"bad value width in {arm!r}")
        if m.group(1) == "hvah":
            return dict(out, family="hvah", v_bits=v, lens=LENS[v], hvah_width=HVAH_WIDTH[v])
        p, r = _QREAD_VAR[m.group(1)]
        return dict(out, family="qread", v_bits=v, lens=LENS[v], protect=p, resel=r)
    m = re.fullmatch(r"mech_(q|a|p)", base)
    if m and not s:
        return dict(out, family="mech", v_bits=16, lens="V16", phase=m.group(1))
    raise ValueError(f"unknown arm {arm!r}")


def family(arm: str) -> str:
    return parse_arm(arm)["family"]


def build_plan(p: dict) -> list:
    """The frozen decode order: fp (and twins) first; every twin right after
    its base; then the arms that build their own views."""
    groups = [("fp", [0], p["fp_twins"]), ("uniform", p["dense"], p["dense_twins"]),
              ("router_calib", p["sieve"], p["sieve_twins"]),
              ("router_pool_calib", p["pool"], p["pool_twins"]),
              ("router_seq_calib", p["seq"], []),
              ("router_seq2_calib", p["seq2"], p["seq2_twins"])]
    groups += [(f"router_top{n}_calib", [p["B_low"]], p["topn_twins"]) for n in p["topn"]]
    groups += [("router_union_calib", p["union"], p["union_twins"]),
               ("router_pool_oracle", p["oracle"], [])]
    plan = []
    for arm, budgets, twins in groups:
        for B in budgets:
            B = norm_b(B)
            if arm == "uniform" and not float(B).is_integer():
                raise ValueError(f"uniform needs an integer width, got {B}")
            plan.append((arm, B))
            plan += [(arm + t, B) for t in twins]
    for B, v in p["hvah"]:
        if norm_b(B) not in [norm_b(x) for x in p["seq2"]]:
            raise ValueError(f"hvah@{B} copies router_seq2_calib@{B}, which is not planned")
        plan.append((f"hvah_v{int(v)}", norm_b(B)))
    for key in ("qread", "qreadp", "qreadr", "qreadpr"):
        for r, v in p[key]:
            plan.append((f"{key}_v{int(v)}", norm_b(r)))
    for B in p["mech"]:
        if norm_b(B) not in [norm_b(x) for x in p["pool"]] or \
                norm_b(B) not in [norm_b(x) for x in p["seq2"]]:
            raise ValueError(f"mech@{B} needs router_pool_calib@{B} and router_seq2_calib@{B}")
        plan += [(f"mech_{ph}", norm_b(B)) for ph in ("q", "a", "p")]
    if len(set(plan)) != len(plan):
        raise ValueError("duplicate (arm, B) in the plan")
    for arm, _ in plan:
        parse_arm(arm)
    return plan


def precompute_want(p: dict) -> list:
    """The base allocations every planned arm needs: the dense widths, the
    qread store, and the three candidates of every pooled-router budget
    (pool, seq, seq2, top-N, union, oracle, mech) and of SIEVE's budgets."""
    pooled = {*p["pool"], *p["seq"], *p["seq2"], *p["union"], *p["oracle"], *p["mech"]}
    if p["topn"]:
        pooled.add(p["B_low"])
    q = dict(dense=list(p["dense"]),
             qread=any(p[k] for k in ("qread", "qreadp", "qreadr", "qreadpr")),
             sieve=list(p["sieve"]), pool=sorted(pooled, key=float), seq=[], union=[], oracle=[])
    return L1C.precompute_want(q)


# ------------------------------------------------------------ the metric
def answer_tokens(tok, ids, expected) -> dict:
    """Which tokens of FP's answer spell an expected answer string.

    Token i covers the characters [start_i, end_i) of the decoded answer, from
    prefix decodes. The span ends at the last character of the expected value
    that FP completes last; FP's text after that (continuation) is outside.
    Value tokens are the span's tokens that overlap any occurrence of any
    expected string. Returns dict(vmask, span_end, found, text)."""
    ids = [int(x) for x in ids]
    n = len(ids)
    full = tok.decode(ids) if n else ""
    starts, ends, prev = [], [], 0
    for i in range(n):
        e = len(tok.decode(ids[:i + 1]))
        starts.append(min(prev, e))
        ends.append(e)
        prev = e
    occ, span_char, found = [], 0, []
    for x in expected:
        x = str(x)
        if not x:
            continue
        f = full.find(x)
        if f < 0:
            continue
        found.append(x)
        span_char = max(span_char, f + len(x))
        j = f
        while j >= 0:
            occ.append((j, j + len(x)))
            j = full.find(x, j + 1)
    occ = [(s, e) for s, e in occ if s < span_char]
    span_end = sum(1 for s in starts if s < span_char) if found else 0
    vmask = [bool(found) and i < span_end and any(starts[i] < e and ends[i] > s for s, e in occ)
             for i in range(n)]
    return dict(vmask=vmask, span_end=span_end, found=found, text=full)


def tf_metrics2(logits: torch.Tensor, targets, content, vmask, span_end) -> dict:
    """Stage 1c's metrics (tf_*, tf_c_*: all content tokens) plus the answer-value
    metric (a_*), the content inside / after the answer span (s_*, post_*), and
    the per-token log-probabilities (tf_logp) for later re-scoring."""
    out = L1B.tf_metrics(logits, targets, content)
    T = len(targets)
    lg = logits.float()
    t = torch.tensor([int(x) for x in targets], device=lg.device)
    lp = torch.log_softmax(lg, -1)[torch.arange(T, device=lg.device), t]
    top1 = lg.argmax(-1) == t
    c = torch.tensor([bool(x) for x in content], device=lg.device)
    v = torch.tensor([bool(x) for x in vmask], device=lg.device)
    idx = torch.arange(T, device=lg.device)
    nan = float("nan")
    if bool(v.any()):
        out.update(a_len=int(v.sum()), a_sum_nll=float(-lp[v].sum()), a_min_logp=float(lp[v].min()),
                   a_top1=float(top1[v].float().mean()), a_all_top1=bool(top1[v].all()))
    else:
        out.update(a_len=0, a_sum_nll=nan, a_min_logp=nan, a_top1=nan, a_all_top1=False)
    sp = c & (idx < int(span_end))
    po = c & (idx >= int(span_end))
    out.update(span_end=int(span_end), s_len=int(sp.sum()),
               s_sum_nll=float(-lp[sp].sum()) if bool(sp.any()) else 0.0,
               post_len=int(po.sum()), post_sum_nll=float(-lp[po].sum()) if bool(po.any()) else 0.0,
               tf_logp=[round(float(x), 5) for x in lp.tolist()],
               tf_vmask="".join("1" if x else "0" for x in vmask),
               tf_cmask="".join("1" if x else "0" for x in content))
    return out


def query_term(task: str, meta: dict) -> str:
    """The string the question names and the model must look up: the key for
    the niah tasks, the chain's value for vt."""
    q = meta["question"]
    m = (re.search(r"assigned the value (\d+)", q) if task == "vt" else
         re.search(r"special magic numbers? for (\S+) mentioned", q))
    return m.group(1) if m else ""


def category_index(C: int, L0: int, k_len: int, value, key, other) -> torch.Tensor:
    """Anatomy category of every key position [0, k_len) as an index into
    ANAT_CATS. Context positions [0, C): value > key > other > sink > prefix >
    rest; [C, L0) = the protected window; [L0, k_len) = question / answer tokens."""
    cat = torch.full((k_len,), ANAT_CATS.index("rest"), dtype=torch.long)
    cat[:min(PREFIX_TOKENS, C)] = ANAT_CATS.index("prefix")
    cat[:min(SINK_TOKENS, C)] = ANAT_CATS.index("sink")
    for name, m in (("other", other), ("key", key), ("value", value)):
        if m is not None:
            mm = torch.as_tensor(m, dtype=torch.bool)[:C]
            cat[:mm.numel()][mm] = ANAT_CATS.index(name)
    cat[C:L0] = ANAT_CATS.index("window")
    cat[L0:] = ANAT_CATS.index("question")
    return cat


def anatomy_masses(q: torch.Tensor, K: torch.Tensor, pos: torch.Tensor, scaling: float,
                   cat: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """Mean attention mass per category, per query head, over the given rows.
    q [H, n, d] rows at absolute positions pos [n]; K [Hkv, k_len, d] the keys
    they read (causal: row at p sees keys <= p); cat [k_len]. -> [H, n_cat]."""
    H, n, d = q.shape
    Hkv, k_len, _ = K.shape
    r = H // Hkv
    ncat = len(ANAT_CATS)
    onehot = torch.nn.functional.one_hot(cat.to(K.device), ncat).float()      # [k_len, n_cat]
    acc = torch.zeros(H, ncat, dtype=torch.float64, device=K.device)
    j = torch.arange(k_len, device=K.device)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        qq = q[:, a:b].float().reshape(Hkv, r, b - a, d)
        s = torch.einsum("grnd,gkd->grnk", qq, K.float()) * scaling
        pp = pos[a:b].to(K.device)
        s = s.masked_fill(j.view(1, 1, 1, -1) > pp.view(1, 1, -1, 1), float("-inf"))
        w = torch.softmax(s, -1)                                                # [Hkv, r, m, k]
        acc += (w @ onehot).sum(2).reshape(H, ncat).double()
    return (acc / n).float()


# ------------------------------------------------ question-time selection
def rows_scores(qrows: torch.Tensor, pos: torch.Tensor, kd: torch.Tensor, key: torch.Tensor,
                C: int, scaling: float, ev: torch.Tensor | None = None) -> torch.Tensor:
    """SnapKV's vote of arbitrary query rows over the context: qrows [n, H, d] at
    absolute positions pos [n]; the context keys are the stored ones (kd
    [Hkv, C, d]) and everything after them comes from the cache (key
    [Hkv, k_len, d]). Each row's softmax covers the keys it sees (causal);
    its context part is summed over rows and the KV group -> [Hkv, C]."""
    n, H, d = qrows.shape
    k_len = int(pos.max()) + 1
    K = torch.cat([kd.to(key.dtype), key[:, C:k_len, :]], dim=1).float()      # [Hkv, k_len, d]
    Hkv = K.shape[0]
    r = H // Hkv
    q = qrows.float().permute(1, 0, 2).reshape(Hkv, r, n, d)
    s = torch.einsum("grnd,gkd->grnk", q, K) * scaling
    j = torch.arange(k_len, device=s.device)
    s = s.masked_fill(j.view(1, 1, 1, -1) > pos.to(s.device).view(1, 1, -1, 1), float("-inf"))
    if ev is not None and bool(ev.any()):
        m = torch.zeros(Hkv, k_len, dtype=torch.bool, device=s.device)
        m[:, :C] = ev.to(s.device)
        s = s.masked_fill(m.view(Hkv, 1, 1, k_len), float("-inf"))
    return torch.softmax(s, -1)[..., :C].sum(dim=(1, 2))


def select_keep(score: torch.Tensor, r: float, protect=()) -> torch.Tensor:
    """floor(r C) tokens per KV head by the pooled vote; protected KV heads keep all."""
    keep = L1C.qread_keep(score, r)
    for g in protect:
        keep[int(g)] = True
    return keep


class RowBuffer:
    """The last `rows` query rows per layer ([H, d] each, with absolute positions),
    and how many answer rows each layer has processed: what a (re-)selection
    votes with, identically in the decode path and the segmented replay."""

    def __init__(self, rows: int = QREAD_ROWS, k: int = 0):
        self.rows, self.k = int(rows), int(k)
        self.q: dict = {}
        self.pos: dict = {}
        self.steps: dict = {}
        self.phase = "question"

    def append(self, li: int, q: torch.Tensor, pos: torch.Tensor):
        """q [n, H, d] at positions pos [n]; keeps the last `rows`."""
        qq = torch.cat([self.q[li], q.float()], 0) if li in self.q else q.float()
        pp = torch.cat([self.pos[li], pos.to(qq.device)], 0) if li in self.pos else pos.to(qq.device)
        self.q[li], self.pos[li] = qq[-self.rows:], pp[-self.rows:]
        if self.phase == "answer":
            self.steps[li] = self.steps.get(li, 0) + q.shape[0]

    def due(self, li: int) -> bool:
        s = self.steps.get(li, 0)
        return self.k > 0 and self.phase == "answer" and s > 0 and s % self.k == 0


def scan_bytes(k: int, d: int = D_DEFAULT) -> float:
    """Bytes per context token per KV head per step for re-selecting every k
    steps from the stored keys (codes + norm; values are not read)."""
    return (d / 8.0) * (STORE_WIDTH + 16.0 / d) / k if k else 0.0


# ------------------------------------------------------ routes and heads
def topn_dense(r0_B: dict, critical, oracle, n_total: int) -> list:
    """The head budget's dense set: R0's dense heads, then sequence-critical
    heads (more prompt-tasks first, then larger total rescue gain), then the
    heads the per-prompt oracle keeps dense most often, until n_total.
    critical rows: [layer, kv, count, gain_sum]; oracle rows: [layer, kv, count]."""
    dense = list(L1C.dense_heads(r0_B))
    seen = set(dense)
    ranked = [(int(li), int(g)) for li, g, *_ in sorted(critical, key=lambda x: (-x[2], -x[3], x[0], x[1]))]
    ranked += [(int(li), int(g)) for li, g, *_ in sorted(oracle, key=lambda x: (-x[2], x[0], x[1]))]
    for h in ranked:
        if len(dense) >= n_total:
            break
        if h not in seen:
            dense.append(h)
            seen.add(h)
    return sorted(dense)


def heads_by_layer(heads) -> dict:
    out: dict = {}
    for li, g in heads:
        out.setdefault(int(li), []).append(int(g))
    return {li: sorted(set(v)) for li, v in out.items()}


def hvah_bits(src: torch.Tensor, width: int) -> torch.Tensor:
    """Keep exactly the source allocation's tokens, every one at `width`."""
    return L1B.hybrid_bits(src, width)


# ----------------------------------------------------------- byte rules
def key_side_bits(fam: str, evict_frac: float, d: int = D_DEFAULT) -> float:
    if fam in ("seq2", "topn", "seq", "mech"):
        return L1C.key_side_bits("sieve", evict_frac, d)
    if fam in ("hvah", "qread"):
        return L1C.key_side_bits("vah", evict_frac, d)
    return L1C.key_side_bits(fam, evict_frac, d)


def v_side(v_bits, d: int = D_DEFAULT) -> float:
    return L1C.v_side(v_bits, d)


# ------------------------------------------------------------ statistics
def rescue_fraction(pool, rescue, arm):
    """Share of the full rescue (pool - seq2) that `arm` recovers, from mean
    per-prompt differences; nan when there is nothing to rescue."""
    den = float(np.mean(pool - rescue))
    return float(np.mean(pool - arm) / den) if den > 0 else float("nan")


def effect_label(mean: float, lo: float, hi: float, name: str, eps: float = 0.05) -> str:
    """HELPS / HURTS need |effect| >= eps nats AND an interval excluding 0."""
    if mean <= -eps and hi < 0:
        return f"{name}_HELPS"
    if mean >= eps and lo > 0:
        return f"{name}_HURTS"
    return f"{name}_NO_EFFECT"
