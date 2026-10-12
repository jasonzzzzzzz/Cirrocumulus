#!/usr/bin/env python3
"""The KL bridge, water-filling allocation and run-time targets (RESEARCH_PLAN.md §3 KL, WF, τ; theory.md §14), from
e2e_s1h5.py's runs.

Proposition KL (theory.md §14). For the unbiased tail with an independent dither per layer, to second order in the
attention errors, the expected KL per token is additive over layers. Each layer's term is a key part proportional to
its key-term MSE plus a value part proportional to its value-term MSE:
    E KL ~ 1/2 sum_l (c_l^K M_l^K + c_l^V M_l^V).
c_l^K and c_l^V are the layer's sensitivities to its own key-dither and value-dither error shapes. They are measured
once, by dithering only the keys (or only the values) of that layer alone, as 2 KL / realized MSE. The isotropic
bridge uses one c_l, from isotropic Gaussian noise of the same per-coordinate size.

Corollary WF. With M_l^K ~ a_l c(k) and M_l^V ~ b_l c(v), c(x) ~ 4^-x, traffic is affine in sum (k_l + v_l). The traffic
that meets a KL budget is lowest when every (layer, tier) component contributes the same KL theta (reverse
water-filling): k_l* = 1/2 log2(c_l^K a_l / (2 theta)) and v_l* likewise, clipped to the available widths. On the grid,
the allocation is the Lagrangian per-layer choice argmin_(k,v) traffic(k, v) + lambda KL_l(k, v).

    python klbridge_s1h5.py bridge --model llama1b [--dir DIR]
    python klbridge_s1h5.py alloc --model llama1b --calib 0 --out arms.json    # allocations from window-0 units
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.environ.get("KL_DIR", os.path.join(HERE, "findings", "klbridge_runs"))
KB, VB = (4, 5, 6, 8), (4, 5, 6, 8)
FRAC = 8


def traffic(kb, vb, frac=FRAC):
    return (1 - 1 / frac) * (kb + vb) / 32 + 1 / frac


def load(model, d=DIR):
    """{(text, window): {"configs": ..., "sens": ...}} from the run files."""
    units = {}
    for p in sorted(glob.glob(os.path.join(d, f"{model}_*_w*_*.json"))):
        r = json.load(open(p))
        plan = os.path.basename(p).rsplit("_", 1)[1][:-5]
        units.setdefault((r["text"], r["window"]), {})[plan] = r
    return units


def sens(u):
    """Per layer: c_iso (isotropic noise), c_key, c_val (own key / value dither at 4 bits), c_tail (the 6/4 design),
    each 2 KL / realized MSE; and the noise's linearity ratio KL(3s) / KL(s) (9 in the quadratic regime)."""
    out = {}
    for li, r in u["sens"]["sens"].items():
        c = lambda k: 2.0 * r[k]["kl"] / max(r[k]["err2"], 1e-300)            # noqa: E731
        out[int(li)] = dict(iso=c("noise1"), iso3=c("noise3"), key=c("keys4"), val=c("vals4"), tail=c("tail64"),
                            lin=r["noise3"]["kl"] / max(r["noise1"]["kl"], 1e-300), kl_tail64=r["tail64"]["kl"])
    return out


def cfg_parts(pred_l, nm):
    """(key MSE, value MSE) per layer for a configuration name, from Theorem M's predictions."""
    if nm.startswith("tail") or nm.startswith("nested") or nm.startswith("keys") or nm.startswith("vals"):
        if nm.startswith("nested"):
            rb = int(nm.split("+")[1].split("/")[0])
            return pred_l["key"][f"n{rb}/{FRAC}"], pred_l["val"][f"n{rb}/4/{FRAC}"]
        if nm.startswith("keys"):
            kb = int(nm.split()[1])
            return pred_l["key"][f"{kb}/{FRAC}"], 0.0
        if nm.startswith("vals"):
            vb = int(nm.split()[1])
            return 0.0, pred_l["val"][f"8/{vb}/{FRAC}"] if f"8/{vb}/{FRAC}" in pred_l["val"] else pred_l["val"][f"6/{vb}/{FRAC}"]
        kb, vb = (int(x) for x in nm.split()[1].split("/"))
        return pred_l["key"][f"{kb}/{FRAC}"], pred_l["val"][f"{kb}/{vb}/{FRAC}"]
    return None


def bridge(model, d=DIR) -> dict:
    """Measured KL per configuration against the isotropic and the structured bridge (Theorem M's MSE, no draws), and
    the realized-error bridges; additivity over layers; per unit and pooled."""
    units = load(model, d)
    rows, add = [], []
    for (t, w), u in sorted(units.items()):
        if "configs" not in u or "sens" not in u:
            continue
        S = sens(u)
        pred = {int(k): v for k, v in u["configs"]["predict"].items()}
        L = len(pred)
        # additivity: the 6/4 design in all layers against the sum of its single-layer KLs
        meas_all = st.mean(r["kl_mean"] for r in u["configs"]["configs"]["tail 6/4"])
        add.append(dict(text=t, window=w, all_layers=meas_all, sum_single=sum(S[li]["kl_tail64"] for li in range(L))))
        for nm, runs in u["configs"]["configs"].items():
            meas = st.mean(r["kl_mean"] for r in runs)
            real = [st.mean(r["layers"][str(li)]["err2"] if str(li) in r["layers"] else r["layers"][li]["err2"]
                            for r in runs) for li in range(L)]
            row = dict(text=t, window=w, config=nm, measured=meas,
                       real_iso=0.5 * sum(S[li]["iso"] * real[li] for li in range(L)),
                       real_tail=0.5 * sum(S[li]["tail"] * real[li] for li in range(L)))
            parts = cfg_parts(pred[0], nm) if nm != "fp8" and not nm.startswith("dense") else None
            if parts is not None:
                kv = [cfg_parts(pred[li], nm) for li in range(L)]
                row["pred_iso"] = 0.5 * sum(S[li]["iso"] * (kv[li][0] + kv[li][1]) for li in range(L))
                row["pred_struct"] = 0.5 * sum(S[li]["key"] * kv[li][0] + S[li]["val"] * kv[li][1] for li in range(L))
            if nm == "fp8":
                row["pred_iso"] = 0.5 * sum(S[li]["iso"] * pred[li]["fp8"] for li in range(L))
            rows.append(row)
    return dict(rows=rows, additivity=add)


def summarize(res) -> dict:
    """Median (and 10th-90th percentile) of predicted / measured per configuration, pooled over units; the additivity
    ratio; the linearity of the noise probes."""
    by = {}
    for r in res["rows"]:
        for k in ("pred_struct", "pred_iso", "real_iso", "real_tail"):
            if k in r and r["measured"] > 0:
                by.setdefault(r["config"], {}).setdefault(k, []).append(r[k] / r["measured"])
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))]            # noqa: E731
    out = {c: {k: dict(med=st.median(v), p10=q(v, 0.1), p90=q(v, 0.9), n=len(v)) for k, v in d.items()}
           for c, d in by.items()}
    ad = [a["sum_single"] / a["all_layers"] for a in res["additivity"]]
    return dict(ratios=out, additivity=dict(med=st.median(ad), min=min(ad), max=max(ad), n=len(ad)) if ad else None)


# ------------------------------------------------------------------ water-filling
def surface(units, windows, norm=False):
    """Per layer, the KL surface KL_l(k, v) = 1/2 (c_key M_key(k) + c_val M_val(k, v)) averaged over the units of the
    given windows (calibration), and the per-layer FP8-target choice for comparison. norm: each unit's surface is first
    divided by its own predicted uniform 6/4 KL, so that every input counts equally (relative KL)."""
    acc, n = {}, 0
    fp8_choice = {}
    for (t, w), u in units.items():
        if w not in windows or "configs" not in u or "sens" not in u:
            continue
        S = sens(u)
        pred = {int(k): v for k, v in u["configs"]["predict"].items()}
        n += 1
        z = 1.0
        if norm:
            z = sum(0.5 * (S[li]["key"] * p["key"][f"6/{FRAC}"] + S[li]["val"] * p["val"][f"6/4/{FRAC}"])
                    for li, p in pred.items())
        for li, p in pred.items():
            for kb in KB:
                for vb in VB:
                    k_ = p["key"][f"{kb}/{FRAC}"]
                    v_ = p["val"][f"{kb}/{vb}/{FRAC}"]
                    acc.setdefault(li, {}).setdefault((kb, vb), 0.0)
                    acc[li][(kb, vb)] += 0.5 * (S[li]["key"] * k_ + S[li]["val"] * v_) / z
            # FP8-target (per head, p90) choice on this unit: cheapest grid point whose per-head MSE <= FP8's at p90
            f8 = p["fp8_head"]
            best = None
            for kb in KB:
                for vb in VB:
                    mh = [a + b for a, b in zip(p["key_head"][f"{kb}/{FRAC}"], p["val_head"][f"{kb}/{vb}/{FRAC}"])]
                    ok = sorted(m <= f for m, f in zip(mh, f8))
                    if sum(ok) >= 0.9 * len(ok):
                        t_ = traffic(kb, vb)
                        if best is None or t_ < best[0] - 1e-12:
                            best = (t_, (kb, vb))
            fp8_choice.setdefault(li, []).append(best[1] if best else (8, 8))
    surf = {li: {kv: v / n for kv, v in d.items()} for li, d in acc.items()}
    fp8c = {li: max(v, key=lambda kv: traffic(*kv)) for li, v in fp8_choice.items()}     # per-layer max over units
    return surf, fp8c


def lagrange(surf, lam):
    """Each layer's argmin traffic + lam KL_l over the grid; returns (allocation, mean traffic, predicted KL)."""
    alloc = {}
    for li, d in surf.items():
        alloc[li] = min(d, key=lambda kv: traffic(*kv) + lam * d[kv])
    L = len(surf)
    return alloc, sum(traffic(*alloc[li]) for li in surf) / L, sum(surf[li][alloc[li]] for li in surf)


def allocations(model, calib=(0,), d=DIR, norm=False) -> dict:
    """Water-filled allocations from the calibration units: (a) the cheapest that meets uniform 6/4's predicted KL,
    (b) the lowest predicted KL at uniform 5/4's traffic (0.371), (c) at 0.36; with the FP8-target allocation (per-head
    p90, per-layer max over calibration units) and uniform 5/4 and 6/4 for reference. Arms for e2e_s1h5.py --plan arms."""
    units = load(model, d)
    surf, fp8c = surface(units, set(calib), norm)
    L = len(surf)
    pred_uniform = {f"{kb}/{vb}": sum(surf[li][(kb, vb)] for li in surf) for kb in KB for vb in VB}
    frontier = []
    for e in range(-60, 61):
        lam = 10 ** (e / 10)
        a, t_, kl = lagrange(surf, lam)
        frontier.append((t_, kl, lam, a))
    frontier.sort(key=lambda x: (x[0], x[1]))
    tgt = pred_uniform["6/4"]
    best_kl = min((f for f in frontier if f[1] <= tgt), key=lambda f: f[0])
    at_371 = min((f for f in frontier if f[0] <= traffic(5, 4) + 1e-9), key=lambda f: f[1])
    at_36 = min((f for f in frontier if f[0] <= 0.36 + 1e-9), key=lambda f: f[1])
    fp8_t = sum(traffic(*fp8c[li]) for li in surf) / L
    fp8_kl = sum(surf[li][fp8c[li]] for li in surf)
    arms = {
        "wf_match64": {li: ["tail", *best_kl[3][li], FRAC] for li in surf},
        "wf_371": {li: ["tail", *at_371[3][li], FRAC] for li in surf},
        "wf_360": {li: ["tail", *at_36[3][li], FRAC] for li in surf},
        "fp8target": {li: ["tail", *fp8c[li], FRAC] for li in surf},
        "uniform54": {li: ["tail", 5, 4, FRAC] for li in surf},
        "uniform64": {li: ["tail", 6, 4, FRAC] for li in surf},
    }
    summary = dict(pred_uniform=pred_uniform, wf_match64=dict(traffic=best_kl[0], pred=best_kl[1]),
                   wf_371=dict(traffic=at_371[0], pred=at_371[1]), wf_360=dict(traffic=at_36[0], pred=at_36[1]),
                   fp8target=dict(traffic=fp8_t, pred=fp8_kl),
                   alloc={nm: {str(li): list(a[li][1:3]) for li in a} for nm, a in arms.items()})
    return dict(arms=arms, summary=summary)


# ------------------------------------------------------------------ run-time targets for the certified read
def _tau_units(model, d=DIR):
    """{(text, window): {layer: tau material}} merged over the run files' layer halves."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, f"{model}_*_w*_tau*.json"))):
        r = json.load(open(p))
        out.setdefault((r["text"], r["window"]), {}).update({int(k): v for k, v in r["tau"].items()})
    return out


def _apply(lay, tau_fn, c_l, kb, vb):
    """One layer's certified read under a target rule: the first set whose bound meets the target for every query head
    of the KV head (else the exact read). Returns (rows read, sum over heads of the mean squared error, the predicted
    KL 1/2 c_l err2)."""
    import torch
    G, r, n = lay["shape"]
    sets = lay["sets"]
    m = len(sets)
    J = torch.full((G, n), m, dtype=torch.long)
    E, Sz = [], []
    for j, st_ in enumerate(sets):
        bd = torch.tensor(st_["bound"]).view(G, r, n)
        tau = tau_fn(torch.tensor(lay["e8"]).view(G, r, n), torch.tensor(st_["onorm"]).view(G, r, n))
        ok = (bd <= tau).all(1)
        J = torch.where((J == m) & ok, torch.full_like(J, j), J)
        E.append(torch.tensor(st_["err"]).view(G, r, n))
        Sz.append(torch.tensor(st_["size"]).view(G, n))
    E.append(torch.zeros(G, r, n))
    Sz.append(torch.ones(G, n))
    E, Sz = torch.stack(E), torch.stack(Sz)
    err = E.gather(0, J.view(1, G, 1, n).expand(1, G, r, n)).squeeze(0)
    rows = float(Sz.gather(0, J.view(1, G, n)).mean())
    e2 = float((err ** 2).sum((0, 1)).mean())
    return rows, e2, 0.5 * c_l * e2


def tau(model, d=DIR, calib=(0,), test=(1,), kb=6, vb=4, target="oracle") -> dict:
    """Run-time targets for the certified read (theory.md §14). Four rules against the oracle (FP8's per-head error at
    that head-step, which needs exact data):
    - fp8q: one number per layer, a quantile of FP8's per-head error over the calibration units;
    - klb: the KL budget split equally over (layer, head) components with the layer's tail sensitivity,
      tau_l^2 = 2 B / (L H c_l);
    - rel: tau = rho |o~| (the design's own output norm);
    - vote: no certified read (the vote's 1/8 only).
    Each parameter is calibrated so that the rule's predicted KL on the calibration units equals the oracle rule's,
    then applied to the test units. Per rule and unit: rows read, traffic, and the predicted KL
    1/2 sum_l c_l err2_l, with c_l the 6/4 tail's sensitivity (the calibration units' mean)."""
    import torch
    units, tus = load(model, d), _tau_units(model, d)
    cs = {}
    for (t, w), u in units.items():
        if w in calib and "sens" in u:
            for li, v in sens(u).items():
                cs.setdefault(li, []).append(v["tail"])
    c_l = {li: st.mean(v) for li, v in cs.items()}
    L = len(c_l)

    def run(unit_keys, rule, p=None):
        res = {}
        for key in unit_keys:
            lays = tus[key]
            rows, kl = [], 0.0
            for li, lay in lays.items():
                G, r, n = lay["shape"]
                H = G * r
                if rule == "oracle":
                    fn = lambda e8, on: e8                                             # noqa: E731
                elif rule == "fp8q":
                    fn = lambda e8, on, li=li: torch.full_like(e8, q8[li](p))          # noqa: E731
                elif rule == "klb":
                    fn = lambda e8, on, li=li: torch.full_like(e8, math.sqrt(2.0 * p / (L * H * c_l[li])))  # noqa: E731
                elif rule == "rel":
                    fn = lambda e8, on: p * on                                         # noqa: E731
                else:                                                                  # vote: the first set only
                    fn = lambda e8, on: torch.full_like(e8, math.inf)                  # noqa: E731
                rw, e2, k_ = _apply(lay, fn, c_l[li], kb, vb)
                rows.append(rw)
                kl += k_
            rd = st.mean(rows)
            res[key] = dict(rows=rd, traffic=(1 - rd) * (kb + vb) / 32 + rd, pred_kl=kl)
        return res

    e8s = {}
    for key, lays in tus.items():
        if key[1] in calib:
            for li, lay in lays.items():
                e8s.setdefault(li, []).extend(lay["e8"])
    srt = {li: sorted(v) for li, v in e8s.items()}
    q8 = {li: (lambda qq, v=v: v[min(len(v) - 1, max(0, int(qq * len(v))))]) for li, v in srt.items()}
    cal = [k for k in tus if k[1] in calib]
    tst = [k for k in tus if k[1] in test]
    if target == "oracle":
        target = st.mean(v["pred_kl"] for v in run(cal, "oracle").values())
    else:                                              # a multiple of FP8's measured KL on the calibration units, e.g. "fp8*1"
        mult = float(target.split("*")[1]) if "*" in target else 1.0
        target = mult * st.mean(st.mean(r["kl_mean"] for r in u["configs"]["configs"]["fp8"])
                                for (t, w), u in units.items() if w in calib and "configs" in u)

    def solve(rule, lo, hi, geometric=True):
        for _ in range(40):
            mid = math.sqrt(lo * hi) if geometric else (lo + hi) / 2
            if st.mean(v["pred_kl"] for v in run(cal, rule, mid).values()) > target:
                hi = mid
            else:
                lo = mid
        return lo
    params = dict(fp8q=solve("fp8q", 0.0, 1.0, geometric=False), klb=solve("klb", 1e-12, 1.0), rel=solve("rel", 1e-6, 10.0))
    out = dict(target_pred_kl=target, params=params, c_l=c_l, calib={}, test={})
    for rule in ("oracle", "fp8q", "klb", "rel", "vote"):
        for nm, keys in (("calib", cal), ("test", tst)):
            r_ = run(keys, rule, params.get(rule))
            out[nm][rule] = dict(units={f"{k[0]}_w{k[1]}": v for k, v in r_.items()},
                                 traffic=st.mean(v["traffic"] for v in r_.values()),
                                 pred_kl=st.mean(v["pred_kl"] for v in r_.values()))
        print(f"{rule}: test traffic {out['test'][rule]['traffic']:.3f}, predicted KL {out['test'][rule]['pred_kl']:.2e} "
              f"(calibration {out['calib'][rule]['traffic']:.3f}, {out['calib'][rule]['pred_kl']:.2e})", flush=True)
    return out


def tau_frontier(model, d=DIR, calib=(0,), test=(1,), kb=6, vb=4) -> dict:
    """The certified read's traffic against its predicted KL for each target rule, over a wide sweep of the rule's
    parameter. The rules are tau() 's, plus oracle_m (m x FP8's per-head error). The parameters come from the
    calibration units; the frontier is measured on the test units. Reports, per rule, the least test traffic at which
    the predicted KL falls below each of several levels."""
    import torch
    units, tus = load(model, d), _tau_units(model, d)
    cs = {}
    for (t, w), u in units.items():
        if w in calib and "sens" in u:
            for li, v in sens(u).items():
                cs.setdefault(li, []).append(v["tail"])
    c_l = {li: st.mean(v) for li, v in cs.items()}
    L = len(c_l)
    e8s = {}
    for key, lays in tus.items():
        if key[1] in calib:
            for li, lay in lays.items():
                e8s.setdefault(li, []).extend(lay["e8"])
    srt = {li: sorted(v) for li, v in e8s.items()}
    tst = [k for k in tus if k[1] in test]

    def point(rule, p):
        tr, kl = [], []
        for key in tst:
            rows, k_ = [], 0.0
            for li, lay in tus[key].items():
                G, r, n = lay["shape"]
                H = G * r
                if rule == "oracle_m":
                    fn = lambda e8, on, p=p: p * e8                                             # noqa: E731
                elif rule == "fp8q":
                    v = srt[li][min(len(srt[li]) - 1, int(p * len(srt[li])))]
                    fn = lambda e8, on, v=v: torch.full_like(e8, v)                             # noqa: E731
                elif rule == "klb":
                    v = math.sqrt(2.0 * p / (L * H * c_l[li]))
                    fn = lambda e8, on, v=v: torch.full_like(e8, v)                             # noqa: E731
                else:
                    fn = lambda e8, on, p=p: p * on                                             # noqa: E731
                rw, e2, kk = _apply(lay, fn, c_l[li], kb, vb)
                rows.append(rw)
                k_ += kk
            rd = st.mean(rows)
            tr.append((1 - rd) * (kb + vb) / 32 + rd)
            kl.append(k_)
        return st.mean(tr), st.mean(kl)
    grids = dict(oracle_m=[2 ** (e / 2) for e in range(0, 13)], fp8q=[i / 20 for i in range(1, 21)],
                 klb=[10 ** (e / 4) for e in range(-36, 9)], rel=[10 ** (e / 4) for e in range(-14, 5)])
    front = {rule: sorted(point(rule, p_) + (p_,) for p_ in g) for rule, g in grids.items()}
    vote = point("rel", 1e9)
    levels = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
    best = {rule: {f"{lv:g}": min((t for t, k, _ in f if k <= lv), default=None) for lv in levels}
            for rule, f in front.items()}
    for rule, b in best.items():
        print(f"{rule}: least test traffic for predicted KL <= " + ", ".join(
            f"{lv}: {('%.3f' % t) if t is not None else '-'}" for lv, t in b.items()), flush=True)
    print(f"vote only: traffic {vote[0]:.3f}, predicted KL {vote[1]:.2e}")
    return dict(front={r: [list(x) for x in f] for r, f in front.items()}, best=best, vote=list(vote), c_l=c_l)


def main():
    argv = sys.argv[1:]
    cmd = argv[0]
    opts = {argv[i]: argv[i + 1] for i in range(1, len(argv) - 1, 2) if argv[i].startswith("--")}
    model, d = opts.get("--model", "llama1b"), opts.get("--dir", DIR)
    if cmd == "bridge":
        res = bridge(model, d)
        sm = summarize(res)
        for c, v in sm["ratios"].items():
            print(f"{c}: " + "; ".join(f"{k} {x['med']:.2f} ({x['p10']:.2f}-{x['p90']:.2f})" for k, x in v.items()))
        print("additivity (sum of single-layer KLs / all-layer KL, 6/4):", sm["additivity"])
        res["summary"] = sm
    elif cmd == "taufront":
        res = tau_frontier(model, d)
    elif cmd == "tau":
        res = tau(model, d, target=opts.get("--target", "oracle"))
    elif cmd == "alloc":
        res = allocations(model, tuple(int(x) for x in opts.get("--calib", "0").split(",")), d, opts.get("--norm") == "1")
        print(json.dumps(res["summary"], indent=1)[:3000])
        json.dump(res["arms"], open(opts["--arms"], "w"))
    else:
        raise SystemExit(__doc__)
    if "--out" in opts:
        json.dump(res, open(opts["--out"], "w"), indent=1)


if __name__ == "__main__":
    main()
