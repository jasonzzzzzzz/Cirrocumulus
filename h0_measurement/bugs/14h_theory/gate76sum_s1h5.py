#!/usr/bin/env python3
"""Summary of gate76_s1h5's results (findings/gate76/) -> findings/gate76/summary.json and tables_gate76.md.

Per model:
  measured   per unit and arm, KL per token over FP's first line (mean over seeds), and its ratio to dense 7/6's;
  sens       per layer, c_l = 2 KL_l / err2_l of the store's keys, its values and dense 7/6 (single-layer arms), the mean
             over the units that ran them (R7b's calibration rule, calib_r7b);
  bridge     each arm's predicted KL, 1/2 sum_l (c^K_l M^K_l + c^V_l M^V_l), with Theorem M's per-layer terms from the
             unit's own theory part (theory.md §14), against the measured KL;
  layers     the design's and dense 7/6's predicted KL per layer, and the measured single-layer KLs;
  heads      per (layer, KV head, query head): the predicted KL of the design and of dense 7/6 (Theorem M per head-step,
             weighted by the layer's c), summed over the units; the heads that carry the design's excess, with their read
             diagnostics (Corollary G's unread share s under the vote, under the answer-pooled oracle and the per-step
             oracle; attention on passage-ID tokens, and the share of it the vote leaves unread);
  variants   predicted KL at <= 0.406 traffic, over dense 7/6's prediction, per unit: the uniform design, G0's per-layer
             allocation, 6-bit values in the k layers where they gain most, a 1/4 read in the k layers where it gains
             most (k as the budget allows), the water-filled allocation over (residual bits, value bits, read fraction)
             with the vote, and the answer-time re-votes;
  g0         per model and G0 prompt: the same predictions with R7b's calibrated c_l (data/tau_<group>.json), the
             non-re-ranking control.

    python gate76sum_s1h5.py [--models llama8b,qwen3_30b]
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate76_s1h5 as T  # noqa: E402
import g0map_s1h5 as G  # noqa: E402

OUT = T.OUT
BUDGET = 13 / 32                                         # dense 7/6's traffic, 0.40625
R7B_TAU = {"llama8b": "tau_llama128.json", "qwen3_30b": "tau_qwen32.json"}
GRID_R, GRID_V, GRID_F = (1, 2, 3, 4), (3, 4, 5, 6, 8), (8, 4)


def _mean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(x) / len(x) if x else float("nan")


def load_units(model: str) -> dict:
    out = {}
    for d in sorted(glob.glob(os.path.join(OUT, model, "*_*"))):
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        u = dict(meta=json.load(open(os.path.join(d, "meta.json"))), dir=d)
        for part in ("theory", "arms", "layers", "reads"):
            p = os.path.join(d, part + ".json")
            if os.path.exists(p):
                u[part] = json.load(open(p))
        out[os.path.basename(d)] = u
    return out


# ------------------------------------------------------------------ sensitivities
def sensitivities(units: dict, n_layers: int) -> dict:
    """{kind: [c_l]}: the mean over units of 2 KL_l / err2_l (kinds keys, vals, dense76)."""
    acc = {k: [[] for _ in range(n_layers)] for k in ("keys", "vals", "dense76")}
    for u in units.values():
        for li, rec in u.get("layers", {}).get("layers", {}).items():
            for k in acc:
                e2 = rec[k]["err2"]
                if e2 > 0:
                    acc[k][int(li)].append(2.0 * rec[k]["kl_span"] / e2)
    return {k: [_mean(v) for v in vs] for k, vs in acc.items()} | {"n_units": sum(1 for u in units.values() if "layers" in u)}


# ------------------------------------------------------------------ Theorem M terms of a configuration
def cfg_terms(sums: dict, cfg) -> tuple | None:
    """(M^K, M^V) of one layer's configuration from that layer's theory sums, or None when the grid lacks it."""
    kind = cfg[0]
    if kind == "nested" or kind == "grid":
        r, vb, f = cfg[1], cfg[2], cfg[3]
        rd = cfg[4] if len(cfg) > 4 else f"vote{f}"
        nm = f"store:{r}|{rd}"
        if nm not in sums:
            return None
        return sums[nm]["K"], sums[nm]["V"] * T.value_scale("store:", vb)
    if kind == "nrv":
        nm = f"store:{cfg[1]}|rv{cfg[4]}{'q' if cfg[5] else ''}"
        if nm not in sums or cfg[2] != 4:
            return None
        return sums[nm]["K"], sums[nm]["V"]
    if kind == "dense":
        nm = f"dense:{cfg[1]}|none"
        if nm not in sums:
            return None
        return sums[nm]["K"], sums[nm]["V"] * T.value_scale("dense:", cfg[2])
    return None


def pred_kl(theory_layers: dict, arm: dict, c: dict, oracle: bool = False, dense_c: str = "kv") -> tuple:
    """(predicted KL, per-layer list) of an arm {layer: cfg}; c = sensitivities; dense_c "kv" uses the store's c^K, c^V
    for a dense layer too, "own" dense 7/6's measured c."""
    tot, per = 0.0, []
    for li in range(len(theory_layers)):
        cfg = arm[li]
        sums = theory_layers[str(li)] if str(li) in theory_layers else theory_layers[li]
        if oracle and cfg[0] == "nested":
            cfg = ("grid", cfg[1], cfg[2], 8, "oracle8")
        mt = cfg_terms(sums, cfg)
        if mt is None:
            return float("nan"), []
        cK, cV = c["keys"][li], c["vals"][li]
        if cfg[0] == "dense" and dense_c == "own":
            x = 0.5 * c["dense76"][li] * (mt[0] + mt[1])
        else:
            x = 0.5 * (cK * mt[0] + cV * mt[1])
        per.append(x)
        tot += x
    return tot, per


def arm_of(rec: dict) -> dict:
    return {int(li): tuple(v) for li, v in rec["cfg"].items()}


# ------------------------------------------------------------------ allocations at the budget
def options(sums: dict) -> list:
    """Every (cfg, traffic) of the grid at one layer: store 4+r / v at the vote's 1/8 or 1/4."""
    out = []
    for r in GRID_R:
        for vb in GRID_V:
            for f in GRID_F:
                if f"store:{r}|vote{f}" in sums:
                    out.append((("grid", r, vb, f), G.traffic(4 + r, vb, f)))
    return out


def waterfill(thy: list, c: dict, weights: list, budget: float = BUDGET, base=("grid", 2, 4, 8)) -> dict:
    """Corollary WF on the grid: per layer the argmin of sum_u w_u KL_{u,l}(x) + lam traffic(x), lam swept (bisection)
    to the largest objective drop with mean traffic <= budget. thy: per unit its theory layers; weights per unit."""
    n_layers = len(thy[0])
    opts = [options(thy[0][str(li)]) for li in range(n_layers)]

    def cost(li, cfg):
        s = 0.0
        for w_, tl in zip(weights, thy):
            mt = cfg_terms(tl[str(li)], cfg)
            s += w_ * 0.5 * (c["keys"][li] * mt[0] + c["vals"][li] * mt[1])
        return s
    table = [[(cost(li, cfg), tr, cfg) for cfg, tr in opts[li]] for li in range(n_layers)]

    def pick(lam):
        ch = [min(row, key=lambda x: x[0] + lam * x[1]) for row in table]
        return ch, sum(x[1] for x in ch) / n_layers, sum(x[0] for x in ch)
    lo, hi = 0.0, 1.0
    while pick(hi)[1] > budget:
        hi *= 4.0
        if hi > 1e12:
            break
    for _ in range(80):
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 1e6
        if pick(mid)[1] > budget:
            lo = mid
        else:
            hi = mid
    ch, tr, obj = pick(hi)
    return dict(alloc={li: list(x[2]) for li, x in enumerate(ch)}, traffic=tr, objective=obj, lam=hi)


def greedy_upgrade(thy: list, c: dict, weights: list, up, budget: float = BUDGET, base=("grid", 2, 4, 8)) -> dict:
    """Upgrade the base configuration to `up` in the layers with the largest predicted gain per unit of extra traffic,
    as many as the budget allows (the "6-bit values in flat layers" and "1/4 read in a few layers" variants)."""
    n_layers = len(thy[0])
    t0, t1 = G.traffic(4 + base[1], base[2], base[3]), G.traffic(4 + up[1], up[2], up[3])
    k = int(math.floor((budget - t0) * n_layers / (t1 - t0) + 1e-9))
    gain = []
    for li in range(n_layers):
        g_ = 0.0
        for w_, tl in zip(weights, thy):
            a, b = cfg_terms(tl[str(li)], base), cfg_terms(tl[str(li)], up)
            g_ += w_ * 0.5 * (c["keys"][li] * (a[0] - b[0]) + c["vals"][li] * (a[1] - b[1]))
        gain.append((g_, li))
    top = sorted(gain, reverse=True)[:max(k, 0)]
    alloc = {li: list(up if any(li == j for _, j in top) else base) for li in range(n_layers)}
    return dict(alloc=alloc, layers=sorted(j for _, j in top), k=k,
                traffic=(k * t1 + (n_layers - k) * t0) / n_layers)


# ------------------------------------------------------------------ heads
def heads_table(units: dict, c: dict, top: int = 15) -> dict:
    """Per (layer, KV head, query head), summed over units: predicted KL of the design and of dense 7/6 (mean over
    steps of Theorem M's head-step terms, times 1/2 c_l), and the read diagnostics (means over steps and units)."""
    acc = {}
    diag_names = ("s2_vote8", "s2_oracle8", "s2_step8", "s2_rv8", "mass_vote8", "mass_ctx", "mass_id", "mass_named",
                  "mass_named_unread8")
    n_u = 0
    for nm, u in units.items():
        p = os.path.join(u["dir"], "theory_heads.pt")
        if not os.path.exists(p):
            continue
        n_u += 1
        H = torch.load(p, weights_only=False)["heads"]
        for li, hd in H.items():
            li = int(li)
            cK, cV = c["keys"][li], c["vals"][li]
            des = 0.5 * (cK * hd["store:2|vote8"]["K"] + cV * hd["store:2|vote8"]["V"]).mean(-1)      # [G, r]
            den = 0.5 * (cK * hd["dense:7|none"]["K"] + cV * hd["dense:7|none"]["V"]).mean(-1)
            rv8 = 0.5 * (cK * hd["store:2|rv8"]["K"] + cV * hd["store:2|rv8"]["V"]).mean(-1) if "store:2|rv8" in hd else None
            Gk, r = des.shape
            for g in range(Gk):
                for j in range(r):
                    a = acc.setdefault((li, g, j), dict(design=0.0, dense=0.0, rv8=0.0, n=0, **{d: 0.0 for d in diag_names}))
                    a["design"] += float(des[g, j])
                    a["dense"] += float(den[g, j])
                    a["rv8"] += float(rv8[g, j]) if rv8 is not None else float("nan")
                    a["n"] += 1
                    for d in diag_names:
                        if d in hd:
                            a[d] += float(hd[d][g, j].mean())
    if not acc:
        return {}
    rows = []
    for (li, g, j), a in acc.items():
        n = max(a["n"], 1)
        rows.append(dict(layer=li, kv=g, q=j, design=a["design"], dense=a["dense"], rv8=a["rv8"], excess=a["design"] - a["dense"],
                         **{d: a[d] / n for d in diag_names}))
    tot_d = sum(x["design"] for x in rows)
    tot_n = sum(x["dense"] for x in rows)
    rows.sort(key=lambda x: -x["excess"])
    win = [x for x in rows if x["dense"] < x["design"]]
    cum, k90 = 0.0, None
    pos = sum(max(x["excess"], 0) for x in rows)
    for i, x in enumerate(rows):
        cum += max(x["excess"], 0)
        if k90 is None and cum >= 0.9 * pos:
            k90 = i + 1
    by_layer = {}
    for x in rows:
        b = by_layer.setdefault(x["layer"], dict(design=0.0, dense=0.0))
        b["design"] += x["design"]
        b["dense"] += x["dense"]
    return dict(units=n_u, n_heads=len(rows), design_total=tot_d, dense_total=tot_n,
                dense_wins_heads=len(win), dense_wins_share_of_design=sum(x["design"] for x in win) / max(tot_d, 1e-300),
                heads_for_90pct_of_excess=k90, top=rows[:top], by_layer=by_layer,
                corr={d: _spearman([x[d] for x in rows], [x["excess"] for x in rows]) for d in diag_names})


def regimes(units: dict, c: dict, flat=0.05, miss=0.05, oracle_ok=0.01) -> dict:
    """The design's predicted excess over dense 7/6 (1/2 c_l (K + V) per head-step, design minus dense), split by the
    head-step's read regime (Corollary G's s): "flat" = even the per-step oracle 1/8 leaves > `flat` of the second
    moment unread; "vote miss" = not flat, the vote leaves > `miss` unread while the answer-pooled oracle 1/8 leaves
    < `oracle_ok`; "other" = the rest. Also the share of the design's own predicted KL in each regime."""
    tot = dict(flat=[0.0, 0.0], miss=[0.0, 0.0], other=[0.0, 0.0])
    n_hs = dict(flat=0, miss=0, other=0)
    for u in units.values():
        p = os.path.join(u["dir"], "theory_heads.pt")
        if not os.path.exists(p):
            continue
        H = torch.load(p, weights_only=False)["heads"]
        for li, hd in H.items():
            li = int(li)
            cK, cV = c["keys"][li], c["vals"][li]
            des = 0.5 * (cK * hd["store:2|vote8"]["K"] + cV * hd["store:2|vote8"]["V"]).double()
            den = 0.5 * (cK * hd["dense:7|none"]["K"] + cV * hd["dense:7|none"]["V"]).double()
            n = des.shape[-1]
            fl = hd["s2_step8"] > flat
            ms = (~fl) & (hd["s2_vote8"] > miss) & (hd["s2_oracle8"] < oracle_ok)
            ot = ~(fl | ms)
            for nm, m in (("flat", fl), ("miss", ms), ("other", ot)):
                tot[nm][0] += float(((des - den) * m).sum()) / n
                tot[nm][1] += float((des * m).sum()) / n
                n_hs[nm] += int(m.sum())
    ex = sum(v[0] for v in tot.values())
    dd = sum(v[1] for v in tot.values())
    return {nm: dict(excess_share=v[0] / ex if ex else float("nan"), design_share=v[1] / dd if dd else float("nan"),
                     head_steps=n_hs[nm]) for nm, v in tot.items()} | dict(thresholds=dict(flat=flat, miss=miss, oracle_ok=oracle_ok))


def _spearman(a, b):
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = a[ok].argsort().argsort(), b[ok].argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


ARMS_ORDER = ("design", "dense76", "r6", "fp8", "dense66", "v6", "read4", "k3", "g0alloc", "rv1", "rv8", "rv8q", "rvb32k1",
              "rvb32k8", "rv8_k1v3", "rv8_g0", "design_oracle")


def unit_set(meta: dict) -> str:
    """Which set a unit belongs to: re-ranking items 11-18 (the analysis set: sensitivities, the variants'
    predictions), 21-28 (held out: core arms fixed before any 32K result), or a control task."""
    if meta.get("name", "rerank") != "rerank":
        return meta["name"]
    return "rerank 11-18" if meta["item"] <= 18 else "rerank 21-28 (held out)"


def pooled(units: dict, meas: dict) -> dict:
    """Per unit set and arm: units, geometric mean of KL / dense 7/6's, units where the arm is below dense 7/6, mean KL,
    and the arm's traffic."""
    out = {}
    for nm, row in meas.items():
        st = unit_set(units[nm]["meta"])
        d = row.get("dense76")
        for arm in ARMS_ORDER:
            x = row.get(arm)
            if x is None or not d:
                continue
            e = out.setdefault(st, {}).setdefault(arm, dict(logs=[], below=0, kl=[], traffic=row["_traffic"][arm]))
            e["logs"].append(math.log(max(x, 1e-300) / d))
            e["below"] += int(x < d)
            e["kl"].append(x)
    return {st: {arm: dict(n=len(e["logs"]), geo_ratio=math.exp(_mean(e["logs"])), below=e["below"], mean_kl=_mean(e["kl"]),
                           traffic=e["traffic"]) for arm, e in arms.items()} for st, arms in out.items()}


# ------------------------------------------------------------------ per model
def model_summary(model: str) -> dict:
    units = load_units(model)
    if not units:
        return {}
    any_t = next((u for u in units.values() if "theory" in u), None)
    n_layers = any_t["meta"]["n_layers"] if any_t else next(iter(units.values()))["meta"]["n_layers"]
    rr = {k: u for k, u in units.items() if u["meta"].get("name", "rerank") == "rerank"}
    c = sensitivities(rr, n_layers)
    meas = {}
    for nm, u in units.items():
        if "arms" not in u:
            continue
        a = u["arms"]["arms"]
        row = {arm: _mean([x["kl_span"] for x in rec["runs"]]) for arm, rec in a.items()}
        row["_seeds"] = {arm: [x["kl_span"] for x in rec["runs"]] for arm, rec in a.items()}
        row["_traffic"] = {arm: rec["traffic"] for arm, rec in a.items()}
        meas[nm] = row
    bridge = {}
    if c["n_units"]:
        for nm, u in units.items():
            if "theory" not in u or "arms" not in u:
                continue
            tl = u["theory"]["layers"]
            b = {}
            for arm, rec in u["arms"]["arms"].items():
                ar = arm_of(rec)
                p, _ = pred_kl(tl, ar, c, oracle=(rec["vote"] == "oracle"))
                po, _ = pred_kl(tl, ar, c, dense_c="own") if arm.startswith("dense") else (float("nan"), None)
                b[arm] = dict(pred=p, pred_own_c=po, meas=meas[nm].get(arm))
            bridge[nm] = b
    layers = {}
    for nm, u in units.items():
        if "theory" not in u or not c["n_units"]:
            continue
        tl = u["theory"]["layers"]
        _, pd = pred_kl(tl, {li: ("nested", 2, 4, 8) for li in range(n_layers)}, c)
        _, pn = pred_kl(tl, {li: ("dense", 7, 6) for li in range(n_layers)}, c)
        lm = u.get("layers", {}).get("layers", {})
        layers[nm] = dict(design=pd, dense=pn,
                          meas_design={li: lm[li]["keys"]["kl_span"] + lm[li]["vals"]["kl_span"] for li in lm},
                          meas_dense={li: lm[li]["dense76"]["kl_span"] for li in lm},
                          median={li: tl[li]["median"] for li in tl})
    heads = heads_table(rr, c) if c["n_units"] else {}
    if heads:
        heads["regimes"] = regimes(rr, c)
    variants = {}
    thy_rr = [(nm, u["theory"]["layers"]) for nm, u in rr.items() if "theory" in u]
    if c["n_units"] and thy_rr:
        dense_arm = {li: ("dense", 7, 6) for li in range(n_layers)}
        dpred = [pred_kl(tl, dense_arm, c)[0] for _, tl in thy_rr]
        w = [1.0 / max(x, 1e-300) for x in dpred]
        thy = [tl for _, tl in thy_rr]
        cand = {"design 4+2/4 (vote 1/8)": {li: ("grid", 2, 4, 8) for li in range(n_layers)}}
        try:
            cand["G0 allocation (FP8 target)"] = {li: ("grid",) + tuple(v[1:4]) for li, v in T.g0_alloc(model, n_layers).items()}
        except FileNotFoundError:
            pass
        v6 = greedy_upgrade(thy, c, w, ("grid", 2, 6, 8))
        r4 = greedy_upgrade(thy, c, w, ("grid", 2, 4, 4))
        k3 = greedy_upgrade(thy, c, w, ("grid", 3, 4, 8))
        wf = waterfill(thy, c, w)
        cand[f"6-bit values in {v6['k']} layers {v6['layers']}"] = {li: tuple(v) for li, v in v6["alloc"].items()}
        cand[f"1/4 read in {r4['k']} layers {r4['layers']}"] = {li: tuple(v) for li, v in r4["alloc"].items()}
        cand[f"4+3 keys in {k3['k']} layers"] = {li: tuple(v) for li, v in k3["alloc"].items()}
        cand["water-filled (r, v, 1/f) at <= 0.406"] = {li: tuple(v) for li, v in wf["alloc"].items()}
        cand["re-vote W=1 (4+2/4, 1/8)"] = {li: ("nrv", 2, 4, 8, 1, 0) for li in range(n_layers)}
        cand["re-vote W=8 (4+2/4, 1/8)"] = {li: ("nrv", 2, 4, 8, 8, 0) for li in range(n_layers)}
        cand["oracle read (4+2/4, 1/8; not deployable)"] = {li: ("grid", 2, 4, 8, "oracle8") for li in range(n_layers)}

        def trf(arm):
            ts = [G.traffic(4 + v[1], v[2], v[3]) for v in arm.values()]
            return sum(ts) / len(ts)
        for nm_, arm in cand.items():
            per = []
            for (un, tl), dp in zip(thy_rr, dpred):
                p, _ = pred_kl(tl, arm, c)
                per.append(p / dp)
            variants[nm_] = dict(traffic=trf(arm), ratio_to_dense=per, mean_ratio=_mean(per),
                                 geo_ratio=math.exp(_mean([math.log(x) for x in per if x > 0])), alloc={li: list(v) for li, v in arm.items()})
        variants["_wf"] = wf
    reads = {nm: u["reads"]["mean"] for nm, u in units.items() if "reads" in u}
    return dict(model=model, n_layers=n_layers, units={k: u["meta"] for k, u in units.items()}, sens=c, measured=meas,
                pooled=pooled(units, meas), bridge=bridge, layers=layers, heads=heads, variants=variants, reads=reads)


def g0_summary(model: str) -> dict:
    """Per G0 prompt: predicted KL (R7b's c_l, the same for keys and values) of the design, dense 7/6, the re-votes and
    the oracle read, summed over layers; and the per-layer ratio dense / design."""
    tau = os.path.join(T.R7B, "data", R7B_TAU.get(model, "-"))
    if not os.path.exists(tau):
        return {}
    cl = {int(k): v for k, v in json.load(open(tau))["c_l"].items()}
    res = {}
    for p in sorted(glob.glob(os.path.join(OUT, "g0", model, "*.json"))):
        r = json.load(open(p))
        prompt = r["tag"]
        s = r["sums"]
        li = int(r["layer"])
        e = res.setdefault(prompt, dict(layers={}))
        row = {}
        for nm in ("store:2|vote8", "dense:7|none", "store:2|rv1", "store:2|rv8", "store:2|oracle8", "store:2|vote4"):
            if nm in s:
                v = s[nm]["V"] * (T.value_scale("dense:", 6) if nm.startswith("dense") else 1.0)
                row[nm] = 0.5 * cl[li] * (s[nm]["K"] + v)
        row["s2_vote8"] = s["median"]["s2_vote8"]
        e["layers"][li] = row
    for prompt, e in res.items():
        tot = {}
        for row in e["layers"].values():
            for nm, v in row.items():
                if not nm.startswith("s2"):
                    tot[nm] = tot.get(nm, 0.0) + v
        e["total"] = tot
        d = tot.get("store:2|vote8", float("nan"))
        e["dense_over_design"] = tot.get("dense:7|none", float("nan")) / d if d else float("nan")
        e["rv8_over_dense"] = tot.get("store:2|rv8", float("nan")) / tot.get("dense:7|none", float("nan"))
        e["n_layers"] = len(e["layers"])
    return res


# ------------------------------------------------------------------ tables
def _f(x, p=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{p}e}" if (abs(x) < 1e-2 or abs(x) >= 1e4) and x != 0 else f"{x:.{p + 1}g}"


def tables(S: dict) -> str:
    L = ["# gate76: why dense 7/6 is not beaten (tables; gate76sum_s1h5.py)", ""]
    for model, M in S["models"].items():
        if not M:
            continue
        L += [f"## {model}", ""]
        P = M.get("pooled", {})
        if P:
            L += ["### Pooled: KL / dense 7/6's (geometric mean over units) and units below dense 7/6", "",
                  "| arm | traffic | " + " | ".join(P) + " |", "|---|---|" + "---|" * len(P)]
            for arm in ARMS_ORDER:
                cells = [f"{_f(P[st][arm]['geo_ratio'])} ({P[st][arm]['below']}/{P[st][arm]['n']})" if arm in P[st] else "—"
                         for st in P]
                tr = next((P[st][arm]["traffic"] for st in P if arm in P[st]), None)
                if tr is None:
                    continue
                L.append(f"| {arm} | {tr:.4f} | " + " | ".join(cells) + " |")
            L.append("")
        arms = list(ARMS_ORDER)
        L += ["### Measured KL per token (FP's first line), mean over seeds", "",
              "| unit | na | " + " | ".join(arms) + " | design / dense76 | rv8 / dense76 |",
              "|" + "---|" * (len(arms) + 4)]
        for nm, row in sorted(M["measured"].items()):
            na = M["units"][nm]["na"]
            dd = row.get("design", float("nan")) / row.get("dense76", float("nan")) if row.get("dense76") else float("nan")
            rd = row.get("rv8", float("nan")) / row.get("dense76", float("nan")) if row.get("dense76") else float("nan")
            L.append(f"| {nm} | {na} | " + " | ".join(_f(row.get(a)) for a in arms) + f" | {_f(dd)} | {_f(rd)} |")
        L.append("")
        if M["bridge"]:
            L += ["### The KL bridge on these units: predicted / measured", "",
                  "| unit | " + " | ".join(a for a in arms if a not in ("r6", "fp8")) + " |", "|" + "---|" * (len(arms) - 1)]
            for nm, b in sorted(M["bridge"].items()):
                L.append(f"| {nm} | " + " | ".join(_f(b[a]["pred"] / b[a]["meas"]) if a in b and b[a]["meas"] else "—"
                                                    for a in arms if a not in ("r6", "fp8")) + " |")
            L.append("")
        h = M.get("heads")
        if h:
            L += ["### Where dense 7/6 wins (heads; predicted KL summed over the units)", "",
                  f"- design total {_f(h['design_total'])}, dense 7/6 total {_f(h['dense_total'])}; dense lower in "
                  f"{h['dense_wins_heads']} of {h['n_heads']} heads, which carry {h['dense_wins_share_of_design']:.2f} of the "
                  f"design's KL; {h['heads_for_90pct_of_excess']} heads carry 90% of the design's excess.",
                  "- Spearman (excess, diagnostic) over heads: " + ", ".join(f"{k} {v:+.2f}" for k, v in h["corr"].items()),
                  "- The design's excess by read regime (head-steps; flat = per-step oracle 1/8 leaves > 5% unread; vote miss ="
                  " not flat, vote leaves > 5%, answer-pooled oracle < 1%): " + ", ".join(
                      f"{nm} {h['regimes'][nm]['excess_share']:.2f} of the excess ({h['regimes'][nm]['design_share']:.2f} of the "
                      f"design's KL, {h['regimes'][nm]['head_steps']} head-steps)" for nm in ("flat", "miss", "other")), "",
                  "| layer | KV | q | design | dense 7/6 | re-vote 8 | s (vote 1/8) | s (oracle 1/8) | s (per-step 1/8) | s (re-vote 8) | ID mass | named-ID mass | named-ID unread |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for x in h["top"]:
                L.append(f"| {x['layer']} | {x['kv']} | {x['q']} | {_f(x['design'])} | {_f(x['dense'])} | {_f(x['rv8'])} | "
                         f"{x['s2_vote8']:.3f} | {x['s2_oracle8']:.3f} | {x['s2_step8']:.3f} | {x['s2_rv8']:.3f} | "
                         f"{x['mass_id']:.3f} | {x['mass_named']:.3f} | {x['mass_named_unread8']:.3f} |")
            L.append("")
        if M.get("reads"):
            rules = sorted({r for v in M["reads"].values() for r in v})
            L += ["### Re-vote read sets (means over layers): churn per step / share outside the question's set / distinct "
                  "extra rows over the answer, as a multiple of the 1/8", "",
                  "| unit | " + " | ".join(rules) + " |", "|---|" + "---|" * len(rules)]
            for nm, v in sorted(M["reads"].items()):
                L.append(f"| {nm} | " + " | ".join(f"{v[r]['churn']:.3f} / {v[r]['outside_q']:.2f} / {v[r]['union_over_read']:.2f}"
                                                    for r in rules) + " |")
            L.append("")
        if M["variants"]:
            L += ["### Predicted KL over dense 7/6's (re-ranking units; Theorem M x per-layer c)", "",
                  "| variant | traffic | mean ratio | geometric mean | per unit |", "|---|---|---|---|---|"]
            for nm, v in M["variants"].items():
                if nm.startswith("_"):
                    continue
                L.append(f"| {nm} | {v['traffic']:.4f} | {_f(v['mean_ratio'])} | {_f(v['geo_ratio'])} | "
                         + ", ".join(_f(x) for x in v["ratio_to_dense"]) + " |")
            L.append("")
        g0 = S["g0"].get(model)
        if g0:
            L += ["### G0 prompts (R7b's c_l): predicted KL summed over layers", "",
                  "| prompt | layers | design | dense 7/6 | re-vote 1 | re-vote 8 | oracle 1/8 | dense / design | re-vote 8 / dense |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for pr, e in sorted(g0.items()):
                t = e["total"]
                L.append(f"| {pr} | {e['n_layers']} | {_f(t.get('store:2|vote8'))} | {_f(t.get('dense:7|none'))} | "
                         f"{_f(t.get('store:2|rv1'))} | {_f(t.get('store:2|rv8'))} | {_f(t.get('store:2|oracle8'))} | "
                         f"{_f(e['dense_over_design'])} | {_f(e['rv8_over_dense'])} |")
            L.append("")
    return "\n".join(L) + "\n"


def main():
    argv = sys.argv[1:]
    models = argv[argv.index("--models") + 1].split(",") if "--models" in argv else ["llama8b", "qwen3_30b"]
    S = dict(models={m: model_summary(m) for m in models}, g0={m: g0_summary(m) for m in models})
    T._atomic_json(S, os.path.join(OUT, "summary.json"))
    md = tables(S)
    with open(os.path.join(OUT, "tables_gate76.md") + ".tmp", "w") as fh:
        fh.write(md)
    os.replace(os.path.join(OUT, "tables_gate76.md") + ".tmp", os.path.join(OUT, "tables_gate76.md"))
    print(md)


if __name__ == "__main__":
    main()
