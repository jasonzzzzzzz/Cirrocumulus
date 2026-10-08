"""R14 Stage 1h R5 design (frozen 2026-10-08; plan.md amendment "R5"): presets, the arms of R5.1 (the probe),
R5.2 (injected errors) and the first R5.3 arms, on top of s1h4_lib (every R1-R4 arm keeps its
name and meaning).

NEW ARMS
  probe          FP's answer, teacher-forced once more with probe_s1h5's measurement hook (the
                 output is FP's: its KL to FP must be 0). Measures, per head and step, the vote's,
                 the static oracle's and the per-step oracle's missed mass, certificates, budgets,
                 output errors with their bounds; and runs the certified controller's emulation
                 on FP's path (rows read, re-fetches, cap hits, true missed mass).
  inj_top@eps    FP's answer teacher-forced with every answer row's attention read exactly over
  inj_rnd@eps    a row set per KV head whose missed mass is at most eps for every query head of
                 the group: the least important rows evicted first (top) or rows in a random
                 order (rnd). B carries eps. Teacher-forced only: KL and dP against FP are the
                 outcome; the score columns are FP's (tf_only).
  inj_top_l{q}@eps   inj_top in layer quarter q only (q = 0..3): per-layer sensitivity, and the
                 quadratic theory's additivity (the four quarters' KL against all layers').
  qread2t4kqT_v4@r   the system's selection and tiers (qread2t4kq_v4), but the answer reads the
                 unselected rows from tier 1 instead of evicting them (Lemma 3's design).
  quest_v16, quest4_v4   Quest (run_s1h4, unchanged), now in every R5 cell.

PER-ROW METRICS ADDED TO EVERY ROW (run_s1h5 post-processing):
  a_span_ntok, a_span_nll_tok, kl_span_tok, kl_all_tok   per-token NLL / KL (A2 span; all tokens)
  traffic_frac       context bytes read per decode step / FP16 K+V (traffic_frac() below)
  exact_rows_frac    share of context rows read with exact keys

PRESETS (stop r8list): h5llama128, h5qwen32, h5qwen128, h5smoke (Llama-3.2-1B on CPU at 4K,
excluded).
"""
from __future__ import annotations
import re

from s1h4_lib import *  # noqa: F401,F403  (run_s1e reads every earlier name through this module)
import s1c_lib as L1C  # noqa: E402
import s1e_lib as L1E  # noqa: E402
import s1h3_lib as L3  # noqa: E402
import s1h4_lib as L4  # noqa: E402
import s1h_lib as L1H  # noqa: E402
import stops_s1h as STOPS  # noqa: E402

AMEND_R5 = "R5"                              # frozen 2026-10-08 (plan.md amendment "R5")
PROBE, INJ_TOP, INJ_RND = "probe", "inj_top", "inj_rnd"
TAIL = L1H.SYSTEM + "T_v4"                   # 'qread2t4kqT_v4'
INJ_EPS = (0.003, 0.01, 0.03, 0.1, 0.3)
INJ_RND_EPS = (0.01, 0.1)
INJ_LAYER_EPS = (0.1,)                       # the per-quarter injections
N_QUARTERS = 4
TRACE_EPS = (0.01, 0.1)                      # the controller emulation's targets
CAP_FRAC = 0.25                              # the emulated row cap (share of the context)
NEW_FAMILIES = L4.NEW_FAMILIES + ("probe", "inject", "tail", "tail5")
# R5.3 (plan.md amendment "R5.3", draft): the tail design's precision / budget scan
TAIL5_RE = r"tail([234])(x?)(o?)_v([234])"
AMEND_R53 = "R5.3"                            # frozen 2026-10-08 (plan.md amendment "R5.3")


def _plain(arm, family, **kw):
    return dict(base=arm, twin="", protect=False, store=None, family=family, v_bits=16, lens=L1E.LENS[16],
                tier2=None, tier2_bits=None, kv=None, requestion=False, read_v_bits=16, floor=False, **kw)


def parse_arm(arm: str) -> dict:
    if arm == PROBE:
        return _plain(arm, "probe")
    m = re.fullmatch(r"inj_(top|rnd)(?:_l(\d))?", arm)
    if m:
        q = None if m.group(2) is None else int(m.group(2))
        if q is not None and (m.group(1) != "top" or not 0 <= q < N_QUARTERS):
            raise ValueError(f"bad injection arm {arm!r}")
        return _plain(arm, "inject", policy=m.group(1), quarter=q)
    if arm == TAIL:
        pa = dict(L4.parse_arm(L1H.SYSTEM + "_v4"))
        pa.update(base=arm, family="tail", floor=False)
        return pa
    m = re.fullmatch(TAIL5_RE, arm)
    if m:
        kb, vb, x, o = int(m.group(1)), int(m.group(4)), bool(m.group(2)), bool(m.group(3))
        return dict(base=arm, twin="", protect=False, store=kb, family="tail5", v_bits=vb, lens=L1E.LENS.get(vb, f"V{vb}"),
                    tier2="exact", tier2_bits=16, kv=x, requestion=True, read_v_bits=16 if x else vb, floor=False,
                    tier1_key_bits=kb, oracle_sel=o)
    return L4.parse_arm(arm)


def family(arm: str) -> str:
    return parse_arm(arm)["family"]


def key_side_bits(fam: str, evict_frac: float, d: int = L1E.D_DEFAULT) -> float:
    if fam in ("probe", "inject"):
        return float("nan")
    if fam == "tail":
        return float(L1H.STORE4)
    if fam == "tail5":
        return float("nan")
    return L4.key_side_bits(fam, evict_frac, d)


def quarter_layers(q: int, nL: int) -> set:
    return set(range(q * nL // N_QUARTERS, (q + 1) * nL // N_QUARTERS))


# ------------------------------------------------------------ cost model
FP16_KV = 32.0                               # bits per context row and channel: 16-bit key + 16-bit value


def _read_r(pa: dict, B: float, C: int) -> float:
    """The share of context rows a selecting arm reads: floor(r C) / C (the floor system: r = floor_r(C))."""
    r = floor_r(C) if pa.get("floor") else float(B)
    return L1C.qread_keep_count(r, C) / C


def traffic_frac(arm: str, B: float, C: int) -> float:
    """Context bits read per decode step and channel, over FP16 K+V (32 bits per row and channel):
    an analytic model of each arm's design, per answer step (the question-time vote, one 4-bit
    pass per question, is not amortized here; Quest's metadata is read at every step).
    probe / inject: nan (not a design)."""
    pa = parse_arm(arm)
    fam = pa["family"]
    v, rv = float(pa["v_bits"]), float(pa.get("read_v_bits") or pa["v_bits"])
    if fam in ("probe", "inject"):
        return float("nan")
    if fam in ("fp", "fpnoise"):
        return 1.0
    if fam == "closedbook":
        return 0.0
    if fam == "dense":
        return (float(B) + v) / FP16_KV
    if fam in ("fp8kv", "kivi", "kvquant"):
        return (float(pa["store"]) + v) / FP16_KV
    if fam == "quest":
        return L4.QUEST_META_BITS / FP16_KV + float(B) * (float(pa["store"]) + v) / FP16_KV
    k = _read_r(pa, B, C)
    if fam in ("qread", "qreadfp", "qoracle", "qoraclefp", "qreadq"):
        return k * (float(pa["store"]) + v) / FP16_KV
    if fam == "qread2t":
        return k * (float(pa["tier2_bits"]) + rv) / FP16_KV
    if fam == "tail":
        return ((1 - k) * (L1H.STORE4 + v) + k * (float(pa["tier2_bits"]) + rv)) / FP16_KV
    if fam == "tail5":
        return ((1 - k) * (float(pa["store"]) + v) + k * (float(pa["tier2_bits"]) + rv)) / FP16_KV
    return float("nan")


def exact_rows_frac(arm: str, B: float, C: int) -> float:
    """Share of context rows read with exact (16-bit) keys at each decode step."""
    pa = parse_arm(arm)
    fam = pa["family"]
    if fam in ("probe", "inject"):
        return float("nan")
    if fam in ("fp", "fpnoise"):
        return 1.0
    if fam == "quest":
        return float(B) if int(pa["store"]) == L1E.EXACT_WIDTH else 0.0
    if fam in ("qreadfp", "qoraclefp") or (fam in ("qread2t", "tail", "tail5") and int(pa["tier2_bits"]) == 16):
        return _read_r(pa, B, C)
    return 0.0


# ------------------------------------------------------------------ presets
def _r5_preset(model, ctx, quest_r=0.125):
    rf = L1H.floor_r(ctx)
    base = L3._main_preset(model, ctx, STOPS.STOP_LINE)
    g2t = [(L1H.SYSTEM, 0.125, 4)] + ([(L1H.SYSTEM, rf, 4)] if rf > 0.125 else [])
    return dict(base, dense=[(4, ["+v4"])], qread4=[], readq=[], kq=[("kivi", 4, 4)], oracle=[("fp", 0.125, 16)],
                g2t=g2t, quest=[("quest_v16", quest_r), ("quest4_v4", quest_r)], probe=True,
                inject=[(INJ_TOP, INJ_EPS), (INJ_RND, INJ_RND_EPS)]
                + [(f"{INJ_TOP}_l{q}", INJ_LAYER_EPS) for q in range(N_QUARTERS)],
                tail=[0.125] + ([rf] if rf > 0.125 else []))


PRESETS = dict(L4.PRESETS)
PRESETS.update({
    "h5llama128": _r5_preset("llama31-8b", 131072),
    "h5qwen32": _r5_preset("qwen3-30b-a3b-2507", 32768),
    "h5qwen128": _r5_preset("qwen3-30b-a3b-2507", 131072),
})
# CPU smoke: 16 layers, so Quest's two dense layers would spend the whole 1/8 (as R4's smoke: 1/4)
PRESETS["h5smoke"] = dict(_r5_preset("llama31-8b", 4096, quest_r=0.25),
                          inject=[(INJ_TOP, (0.0, 0.01, 0.1)), (INJ_RND, (0.1,))]
                          + [(f"{INJ_TOP}_l{q}", INJ_LAYER_EPS) for q in range(N_QUARTERS)])
R4_PRESET_OF = {"h5llama128": "h4llama128", "h5qwen128": "h4qwen128", "h5smoke": "h4smoke",
                "h53llama128": "h4llama128", "h53qwen128": "h4qwen128", "h53smoke": "h4smoke"}

# R5.3: the tail design (Lemma 3: nothing evicted, unread rows from tier 1) across tier precision,
# exact values on the selected rows, the read fraction, and vote vs oracle selection
TAIL5_ARMS = [("tail4_v4", 0.125), ("tail4x_v4", 0.125), ("tail3x_v3", 0.125), ("tail2x_v2", 0.125),
              ("tail2x_v4", 0.125), ("tail2_v2", 0.125), ("tail2x_v2", 0.0625), ("tail2x_v2", 0.25),
              ("tail2xo_v2", 0.125)]


def _r53_preset(model, ctx, tail5=TAIL5_ARMS):
    base = L3._main_preset(model, ctx, STOPS.STOP_LINE)
    return dict(base, dense=[(4, ["+v4"]), (2, ["+v2"])], qread4=[], readq=[], kq=[], oracle=[("fp", 0.125, 16)],
                g2t=[(L1H.SYSTEM, 0.125, 4)], quest=[], tail=[0.125], tail5=list(tail5))


PRESETS.update({
    "h53llama128": _r53_preset("llama31-8b", 131072),
    "h53qwen32": _r53_preset("qwen3-30b-a3b-2507", 32768),
    "h53qwen128": _r53_preset("qwen3-30b-a3b-2507", 131072),
    "h53smoke": _r53_preset("llama31-8b", 4096),
})


def build_plan(p: dict) -> list:
    """fp, then R5's arms (probe, injected errors, the tail arm), then the reference arms in R4's
    order (L4.build_plan: R3a's arms, Quest), fp_noise last."""
    if not (p.get("probe") or p.get("inject") or p.get("tail") or p.get("tail5")):
        return L4.build_plan(p)
    base = L4.build_plan(dict(p, noise=False))
    if base[0] != ("fp", 0):
        raise ValueError(f"the reference plan must start with fp, not {base[0]}")
    plan = [base[0]]
    if p.get("probe"):
        plan.append((PROBE, 0))
    for arm, eps_list in p.get("inject", []):
        parse_arm(arm)
        for e in eps_list:
            if not 0 <= float(e) < 1:
                raise ValueError(f"{arm}: target missed mass {e}")
            plan.append((arm, float(e)))
    for r in p.get("tail", []):
        if (L1H.SYSTEM + "_v4", L1E.norm_b(r)) not in base:
            raise ValueError(f"the tail arm at r={r} needs the system at the same r planned")
        plan.append((TAIL, L1E.norm_b(r)))
    for arm, r in p.get("tail5", []):
        if parse_arm(arm)["family"] != "tail5" or not 0 < float(r) <= 1:
            raise ValueError(f"bad R5.3 tail arm {arm}@{r}")
        plan.append((arm, L1E.norm_b(r)))
    plan += base[1:]
    if p.get("noise"):
        plan.append((L1H.NOISE, 0))
    if len(set(plan)) != len(plan):
        raise ValueError("duplicate (arm, B) in the plan")
    return plan
