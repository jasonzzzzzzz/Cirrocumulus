#!/usr/bin/env python3
"""G0's summary (ARCHITECTURES_PLAN.md §5): tables from findings/g0/<model>/*.json (g0map_s1h5 and g0mla_s1h5 analyze).

Per model at 32K (the five calibration prompts):
  - families of layers (MODEL_FAMILY.md §1.4, §4.1): flatness = the unread share of sum w^2 |v - o|^2 at the vote's 1/8
    read (concentrated <= 1%, flat >= 10%), split k* - v* = 1/2 log2(a / b) from the key share at uniform 6/6
    (key-heavy >= 1.5, key-leaning 0.5..1.5, balanced -0.5..0.5, value-leaning <= -0.5), the score noise sigma;
  - widths in the single store's form: per layer the cheapest store configuration (4+r keys, values, read 1/8 or 1/4)
    meeting FP8's per-head error at p90, on every prompt (the per-layer maximum, as MODEL_FAMILY.md §5);
  - misses of fixed configurations (a layer misses when p90 > 1 on any prompt);
  - certification widths: layers where part 8 at the store's 4+2 exceeds 20x the error (4+4 keys there for the
    certified read, ARCHITECTURES_PLAN.md §3 item 2);
  - the slack law ln(bound / error) = a + b sigma_w per model and pooled by mechanism.
Controls: PG-19 at 1,600 tokens raw (MODEL_FAMILY.md's input) and with the chat template's first token, against 32K.

    python g0sum_s1h5.py [--models a,b] [--out findings/g0]
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("G0_OUT", os.path.join(HERE, "findings", "g0"))
L_MAIN = 32768
MAIN = ("pg19", "code", "table", "agent", "news")
MECH = {   # mechanism label for the slack law's pooling (MODEL_FAMILY.md §2)
    "llama1b": "GQA", "llama8b": "GQA", "llama70b": "GQA", "mistral7b": "GQA", "mistralsmall": "GQA",
    "granite42": "GQA (multiplier 1/128)", "qwen15": "MHA + bias", "qwen15moe": "MHA + bias",
    "qwen3_1p7b": "GQA + QK-norm", "qwen3_30b": "GQA + QK-norm", "minimaxm2": "GQA + QK-norm",
    "glm45air": "GQA, partial RoPE + bias", "gptoss": "sinks (full layers)", "gemma3": "local:global (global layers)",
    "gemma4": "local:global (global layers)", "qwen35": "hybrid (attention layers)", "qwen38": "hybrid (attention layers)",
    "granite": "hybrid (attention layers)", "dsv2lite": "MLA", "moonlight": "MLA"}
NAMES = {"llama1b": "Llama-3.2-1B", "llama8b": "Llama-3.1-8B (R6)", "llama70b": "Llama-3.3-70B", "mistral7b": "Mistral-7B",
         "mistralsmall": "Mistral-Small-3.2", "granite42": "Granite-4.2-30B", "qwen15": "Qwen1.5-0.5B",
         "qwen15moe": "Qwen1.5-MoE", "qwen3_1p7b": "Qwen3-1.7B", "qwen3_30b": "Qwen3-30B-A3B (R6)",
         "glm45air": "GLM-4.5-Air", "minimaxm2": "MiniMax-M2", "qwen35": "Qwen3.5-27B", "qwen38": "Qwen3.8-27B",
         "gemma4": "Gemma-4-31B", "gptoss": "GPT-OSS-20B", "gemma3": "Gemma-3-12B", "granite": "Granite-4.0-H-Tiny",
         "dsv2lite": "DeepSeek-V2-Lite", "moonlight": "Moonlight-16B-A3B"}
MEASURED = {"store:2/4/8": "nested 4+2/4", "store:4/4/8": "nested 4+4/4", "uniform:6/4/8": "uniform 6/4",
            "uniform:8/4/8": "uniform 8/4"}    # configurations the depth draws measured (2 draws, 4-bit values)
FIXED = ("store:1/4/8", "store:2/4/8", "store:2/5/8", "store:2/6/8", "store:3/4/8", "store:4/4/8", "store:4/6/8",
         "uniform:4/4/8", "uniform:6/4/8", "uniform:6/6/8", "uniform:8/4/8")


def traffic(k, v, f):
    return (1.0 - 1.0 / f) * (k + v) / 32.0 + 1.0 / f


def cfg_traffic(nm):
    form, rest = nm.split(":")
    kx, vb, f = (int(x) for x in rest.split("/"))
    return traffic(4 + kx if form == "store" else kx, vb, f)


UNMET = traffic(8, 8, 4)


def choose(cfg, form="store", target="p90"):
    ok = [(cfg_traffic(nm), v["med"], nm) for nm, v in cfg.items() if nm.startswith(form + ":") and v[target] <= 1.0]
    return min(ok)[2] if ok else None


def split_of(share):
    share = min(max(share, 1e-12), 1 - 1e-12)
    return 0.5 * math.log2(share / (1.0 - share))


def flat_class(x):
    return "concentrated" if x <= 0.01 else ("flat" if x >= 0.10 else "between")


def split_class(x):
    return "key-heavy" if x >= 1.5 else ("key-leaning" if x >= 0.5 else ("balanced" if x > -0.5 else "value-leaning"))


def family(flat, split):
    fc, sc = flat_class(flat), split_class(split)
    if fc == "between":
        return "between"
    if fc == "concentrated":
        return {"key-leaning": "A", "key-heavy": "B"}.get(sc, "A*")       # A*: concentrated, balanced/value-leaning
    return {"key-heavy": "C", "key-leaning": "D", "balanced": "D", "value-leaning": "E"}[sc]


def load(models=None):
    R = collections.defaultdict(lambda: collections.defaultdict(dict))
    for path in sorted(glob.glob(os.path.join(OUT, "*", "*.json"))):
        r = json.load(open(path))
        if "model" not in r or (models and r["model"] not in models):
            continue
        R[r["model"]][r["tag"]][r["layer"]] = r
    return R


def pct(x):
    """A share as a percentage with two significant digits below 10%."""
    return f"{100 * x:.2f}%" if x < 0.1 else f"{100 * x:.0f}%"


def _med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def per_model(m, by_tag, L=L_MAIN):
    main = [t for t in by_tag if t.endswith(f"_{L}") and t.split("_")[0] in MAIN]
    layers = sorted(set().union(*[set(by_tag[t]) for t in main])) if main else []
    rows = {}
    for li in layers:
        rs = {t: by_tag[t][li] for t in main if li in by_tag[t]}
        fl = {t: r["flat"]["8"]["unread_second_moment_share"] for t, r in rs.items()}
        sp = {t: split_of(r["cfg"]["uniform:6/6/8"]["key_share"]) for t, r in rs.items()}
        s6 = {t: r["sigma"]["uniform:6"]["median_unread"] for t, r in rs.items()}
        ch = {t: choose(r["cfg"]) for t, r in rs.items()}
        tr = {t: (cfg_traffic(c) if c else UNMET) for t, c in ch.items()}
        worst = max(tr, key=tr.get)
        names = [nm for nm in next(iter(rs.values()))["cfg"] if nm.startswith("store:")]
        ok_all = [(cfg_traffic(nm), max(r["cfg"][nm]["med"] for r in rs.values()), nm) for nm in names
                  if all(r["cfg"][nm]["p90"] <= 1.0 for r in rs.values())]
        joint = min(ok_all)[2] if ok_all else None
        p8 = {t: r["depth"].get("nested 4+2/4", {}).get("ratio_med") for t, r in rs.items()}
        rows[li] = dict(prompts=sorted(rs), flat=fl, split=sp, sigma6=s6, family={t: family(fl[t], sp[t]) for t in rs},
                        choice=ch, traffic=tr, permax=ch[worst], permax_traffic=tr[worst],
                        calibrated=joint, calibrated_traffic=cfg_traffic(joint) if joint else UNMET,
                        misses={c: sorted(t for t, r in rs.items() if r["cfg"][c]["p90"] > 1.0) for c in FIXED},
                        agg={c: (max(r["cfg"][c]["agg"] for r in rs.values())
                                 if all("agg" in r["cfg"][c] for r in rs.values()) else None) for c in FIXED},
                        agg_meas={c: (max(r["cfg"][c]["agg"] * r["depth"][d_]["mse_meas_over_pred"] for r in rs.values())
                                      if all("agg" in r["cfg"][c] and d_ in r["depth"] for r in rs.values()) else None)
                                  for c, d_ in MEASURED.items()},
                        p8_42=p8, p8_44={t: r["depth"].get("nested 4+4/4", {}).get("ratio_med") for t, r in rs.items()},
                        mse_ratio={t: r["depth"].get("nested 4+2/4", {}).get("mse_meas_over_pred") for t, r in rs.items()},
                        store_cost={t: (r["cfg"]["store:2/4/8"]["med"] / max(r["cfg"]["uniform:6/4/8"]["med"], 1e-30),
                                        r["cfg"]["store:4/4/8"]["med"] / max(r["cfg"]["uniform:8/4/8"]["med"], 1e-30))
                                    for t, r in rs.items()},
                        coverage=min((d_["coverage"] for r in rs.values() for d_ in r["depth"].values()), default=None))
    return dict(layers=layers, prompts=sorted(main), rows=rows)


def control(m, by_tag, tag):
    """One prompt (e.g. pg19raw_1600): per layer flatness, split, uniform-grid choice, store choice."""
    out = {}
    for li, r in by_tag.get(tag, {}).items():
        out[li] = dict(flat=r["flat"]["8"]["unread_second_moment_share"],
                       split=split_of(r["cfg"]["uniform:6/6/8"]["key_share"]), uni=choose(r["cfg"], "uniform"),
                       store=choose(r["cfg"]), miss64=r["cfg"]["uniform:6/4/8"]["p90"] > 1.0,
                       miss66=r["cfg"]["uniform:6/6/8"]["p90"] > 1.0)
    return out


def slack_rows(R):
    rows = []
    for m, by_tag in R.items():
        for t, by_l in by_tag.items():
            if not t.endswith(f"_{L_MAIN}"):
                continue
            for li, r in by_l.items():
                for nm, d_ in (r.get("depth") or {}).items():
                    if d_.get("ratio_med", 0) > 0 and d_.get("sigma_w_med") is not None:
                        rows.append(dict(model=m, mech=MECH.get(m, "?"), layer=li, tag=t, cfg=nm,
                                         sigma_w=d_["sigma_w_med"], ln_ratio=math.log(d_["ratio_med"])))
    return rows


def fit(xs):
    import numpy as np
    if len(xs) < 3:
        return None
    s = np.array([x["sigma_w"] for x in xs])
    y = np.array([x["ln_ratio"] for x in xs])
    A = np.stack([np.ones_like(s), s], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    ref = y - (0.51 + 13.9 * s)
    return dict(n=len(xs), a=float(coef[0]), b=float(coef[1]),
                r2=float(1 - (res ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300)),
                max_abs_resid=float(abs(res).max()), ref_resid_med=float(np.median(ref)),
                ref_resid_max=float(abs(ref).max()), sigma_range=[float(s.min()), float(s.max())])


def mla_model(m, by_tag, L):
    main = [t for t in by_tag if t.endswith(f"_{L}") and t.split("_")[0] in MAIN]
    layers = sorted(set().union(*[set(by_tag[t]) for t in main])) if main else []
    rows = {}
    for li in layers:
        rs = {t: by_tag[t][li] for t in main if li in by_tag[t]}
        ch = {}
        for t, r in rs.items():
            ok = [(v["traffic"], v["med"], nm) for nm, v in r["cfg"].items() if nm.startswith("store:") and v["p90"] <= 1]
            ch[t] = min(ok)[2] if ok else None
        trs = {t: (rs[t]["cfg"][c]["traffic"] if c else 1.0) for t, c in ch.items()}
        worst = max(trs, key=trs.get)
        rows[li] = dict(flat={t: r["flat"]["8"]["unread_second_moment_share"] for t, r in rs.items()}, choice=ch,
                        calibrated=ch[worst], calibrated_traffic=trs[worst],
                        cross_share=max(r["cfg"]["store:2/same/8"]["cross_share"] for r in rs.values()),
                        miss42=sorted(t for t, r in rs.items() if r["cfg"]["store:2/same/8"]["p90"] > 1),
                        miss44=sorted(t for t, r in rs.items() if r["cfg"]["store:4/same/8"]["p90"] > 1),
                        cert={t: r["cert"] for t, r in rs.items()})
    return dict(layers=layers, prompts=sorted(main), rows=rows)


def main():
    argv = sys.argv[1:]
    models = argv[argv.index("--models") + 1].split(",") if "--models" in argv else None
    W = argv[argv.index("--out") + 1] if "--out" in argv else OUT          # where the tables go (results are read from OUT)
    os.makedirs(W, exist_ok=True)
    R = load(models)
    S, md = {}, []
    md.append("# G0 tables (generated by g0sum_s1h5.py; findings_g0.md reads them)\n")
    md.append("## 1. Per model at 32K, single store's form (p90 target, FP8's per-head error; read 1/8 unless chosen)\n")
    md.append("| model | layers × prompts | flatness: median (range) | split k*−v*: median | σ at 6 bits | families | "
              "calibrated store traffic (per-prompt max) | 4+2/4 misses | 4+2/6 | 4+4/4 | 4+4/6 | uniform 6/4 | 6/6 | 8/4 | part 8 > 20× at 4+2 | coverage |")
    md.append("|" + "---|" * 16)
    for m in sorted(R, key=lambda k: list(NAMES).index(k) if k in NAMES else 99):
        if any("cert" in r for t in R[m].values() for r in t.values()):
            continue                                                   # MLA: §6
        P = per_model(m, R[m])
        S[m] = P
        rows = P["rows"]
        if not rows:
            continue
        fl = [x for r in rows.values() for x in r["flat"].values()]
        sp = [_med(list(r["split"].values())) for r in rows.values()]
        s6 = [x for r in rows.values() for x in r["sigma6"].values()]
        fam = collections.Counter(collections.Counter(r["family"].values()).most_common(1)[0][0] for r in rows.values())
        trf = sum(r["calibrated_traffic"] for r in rows.values()) / len(rows)
        trp = sum(r["permax_traffic"] for r in rows.values()) / len(rows)
        mis = {c: sum(bool(r["misses"][c]) for r in rows.values()) for c in FIXED}
        loose = [li for li, r in rows.items() if any(x is not None and x > 20 for x in r["p8_42"].values())]
        cov = min(r["coverage"] for r in rows.values() if r["coverage"] is not None)
        md.append(f"| {NAMES.get(m, m)} | {len(rows)} × {len(P['prompts'])} | {pct(_med(fl))} ({pct(min(fl))}–"
                  f"{pct(max(fl))}) | {_med(sp):+.2f} | {min(s6):.2f}–{max(s6):.2f} | "
                  f"{', '.join(f'{k} {v}' for k, v in sorted(fam.items()))} | {trf:.3f} ({trp:.3f}) | {mis['store:2/4/8']} | "
                  f"{mis['store:2/6/8']} | {mis['store:4/4/8']} | {mis['store:4/6/8']} | {mis['uniform:6/4/8']} | "
                  f"{mis['uniform:6/6/8']} | {mis['uniform:8/4/8']} | {len(loose)} {loose if len(loose) <= 12 else ''} | "
                  f"{cov:.3f} |")
    md.append("\nFamilies: the layer's modal family over the prompts (A concentrated key-leaning, B concentrated key-heavy, "
              "C flat key-heavy, D flat key-leaning or balanced, E flat value-leaning, A* concentrated balanced or "
              "value-leaning). Calibrated traffic: per layer the cheapest store configuration meeting p90 on every prompt "
              "(0.625 if none), averaged over layers; in brackets MODEL_FAMILY §5's rule, the most expensive of the "
              "prompts' own cheapest choices; (1 − f)(k + v)/32 + f.\n")
    # 2. controls
    md.append("## 2. 1,600 tokens (MODEL_FAMILY.md's input) against 32K, PG-19\n")
    md.append("| model | flatness median: 1.6K raw / 1.6K with first token / 32K | uniform-grid traffic (p90): 1.6K raw / 32K | "
              "store traffic: 1.6K raw / 32K | 6/4 misses: 1.6K raw / 32K | 6/6 misses: 1.6K raw / 32K |")
    md.append("|---|---|---|---|---|---|")
    for m in S:
        raw, pre = control(m, R[m], "pg19raw_1600"), control(m, R[m], "pg19_1600")
        big = control(m, R[m], f"pg19_{L_MAIN}")
        if not raw or not big:
            continue
        tu = lambda c: sum(cfg_traffic(x["uni"]) if x["uni"] else UNMET for x in c.values()) / len(c)     # noqa: E731
        ts = lambda c: sum(cfg_traffic(x["store"]) if x["store"] else UNMET for x in c.values()) / len(c)  # noqa: E731
        md.append(f"| {NAMES.get(m, m)} | {pct(_med([x['flat'] for x in raw.values()]))} / "
                  f"{pct(_med([x['flat'] for x in pre.values()]))} / {pct(_med([x['flat'] for x in big.values()]))} | "
                  f"{tu(raw):.3f} / {tu(big):.3f} | {ts(raw):.3f} / {ts(big):.3f} | "
                  f"{sum(x['miss64'] for x in raw.values())} / {sum(x['miss64'] for x in big.values())} | "
                  f"{sum(x['miss66'] for x in raw.values())} / {sum(x['miss66'] for x in big.values())} |")
    # 3. R8's models, per layer
    md.append("\n## 3. R8's models per layer (32K, every prompt): flatness, split, σ, choice, part 8 at 4+2\n")
    for m in ("gemma3", "gptoss"):
        if m not in S:
            continue
        md.append(f"\n**{NAMES[m]}**\n")
        md.append("| layer | flatness (min–max) | split (median) | σ at 6 bits (max) | families | choice per prompt | "
                  "calibrated | misses 4+2/4 · 4+2/6 · 4+4/4 | part 8 at 4+2 (max) |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for li, r in S[m]["rows"].items():
            md.append(f"| {li} | {pct(min(r['flat'].values()))}–{pct(max(r['flat'].values()))} | "
                      f"{_med(list(r['split'].values())):+.2f} | {max(r['sigma6'].values()):.2f} | "
                      f"{''.join(sorted(set(r['family'].values())))} | "
                      f"{', '.join((r['choice'][t] or 'unmet').replace('store:', '') for t in r['prompts'])} | "
                      f"{(r['calibrated'] or 'unmet').replace('store:', '')} | "
                      f"{len(r['misses']['store:2/4/8'])} · {len(r['misses']['store:2/6/8'])} · "
                      f"{len(r['misses']['store:4/4/8'])} | "
                      f"{max(x for x in r['p8_42'].values() if x is not None):.1f} |")
    # 3a. aggregate MSE over FP8's, per layer (the KL-facing view of each configuration)
    AGG = ("store:2/4/8", "store:2/6/8", "store:4/4/8", "store:4/6/8", "uniform:6/4/8")
    md.append("\n## 3a. Aggregate error per layer: Σ predicted MSE / Σ FP8 MSE over head-steps, worst prompt\n")
    md.append("Per configuration: the worst layer, the median layer, and the layers above 1. A sum over head-steps is what "
              "the KL bridge (theory.md §14) weighs, so this is the end-to-end reading; the p90 target of §1 is per head.\n")
    md.append("| model | " + " | ".join(c.replace("store:", "store ").replace("uniform:", "uniform ").replace("/8", "")
                                         for c in AGG) + " |")
    md.append("|---|" + "---|" * len(AGG))
    for m, P in S.items():
        cells = []
        for c in AGG:
            xs = [r["agg"][c] for r in P["rows"].values()]
            if any(x is None for x in xs) or not xs:
                cells.append("n/a")
                continue
            xs = sorted(xs)
            cells.append(f"{xs[-1]:.2f} · {xs[len(xs) // 2]:.3f} · {sum(x > 1 for x in xs)}")
        md.append(f"| {NAMES.get(m, m)} | " + " | ".join(cells) + " |")
    md.append("\nMeasured aggregate (the depth draws: predicted aggregate × pooled measured / predicted MSE, same head-steps; "
              "2 draws), worst layer · median layer · layers above 1. Theorem M's first-order sum overpredicts where a few "
              "head-steps have large score noise (σ ≳ 0.4, GPT-OSS): use these columns there.\n")
    md.append("| model | measured store 2/4 | measured store 4/4 | measured uniform 6/4 | measured uniform 8/4 |")
    md.append("|---|---|---|---|---|")
    for m, P in S.items():
        cells = []
        for c in MEASURED:
            xs = [r["agg_meas"][c] for r in P["rows"].values()]
            if any(x is None for x in xs) or not xs:
                cells.append("n/a")
                continue
            xs = sorted(xs)
            cells.append(f"{xs[-1]:.2f} · {xs[len(xs) // 2]:.3f} · {sum(x > 1 for x in xs)}")
        md.append(f"| {NAMES.get(m, m)} | " + " | ".join(cells) + " |")
    # 3b. validity at 32K and the store's precision cost
    md.append("\n## 3b. Validity at 32K, and the single store's precision cost\n")
    md.append("| model | part 8 coverage (every config, draw, layer, prompt) | Theorem M measured / predicted at 4+2/4 "
              "(median over views; 10th–90th) | store 4+2 / uniform 6 (median RMS ratio) | store 4+4 / uniform 8 |")
    md.append("|---|---|---|---|---|")
    for m, P in S.items():
        rows_ = P["rows"]
        mr = sorted(x for r in rows_.values() for x in r["mse_ratio"].values() if x is not None)
        c2 = sorted(x[0] for r in rows_.values() for x in r["store_cost"].values())
        c4 = sorted(x[1] for r in rows_.values() for x in r["store_cost"].values())
        cov = min(r["coverage"] for r in rows_.values() if r["coverage"] is not None)
        if mr:
            md.append(f"| {NAMES.get(m, m)} | {cov:.3f} | {mr[len(mr) // 2]:.3f} ({mr[len(mr) // 10]:.3f}–"
                      f"{mr[(9 * len(mr)) // 10]:.3f}) | {c2[len(c2) // 2]:.2f} | {c4[len(c4) // 2]:.2f} |")
    # 4. slack law
    rows = slack_rows({m: R[m] for m in S})
    law = {}
    md.append("\n## 4. The slack law, ln(part 8 bound / error) = a + b·σ_w (32K; uniform 6/4 and 8/4, nested 4+2/4 and 4+4/4)\n")
    md.append("| group | n | a | b | R² | max residual | median residual against 0.51 + 13.9σ_w (largest) | σ_w range |")
    md.append("|---|---|---|---|---|---|---|---|")
    groups = collections.defaultdict(list)
    for x in rows:
        groups[("all", "*")].append(x)
        groups[("mech", x["mech"])].append(x)
        groups[("model", x["model"])].append(x)
        if x["cfg"].startswith("uniform"):
            groups[("uniform", "*")].append(x)
    for (kind, k_), xs in sorted(groups.items()):
        f_ = fit(xs)
        if f_:
            law[f"{kind}:{k_}"] = f_
            md.append(f"| {kind}: {NAMES.get(k_, k_)} | {f_['n']} | {f_['a']:.2f} | {f_['b']:.1f} | {f_['r2']:.3f} | "
                      f"{f_['max_abs_resid']:.2f} | {f_['ref_resid_med']:+.2f} ({f_['ref_resid_max']:.2f}) | "
                      f"{f_['sigma_range'][0]:.3f}–{f_['sigma_range'][1]:.3f} |")
    # 5. widths for G1
    for m, P in S.items():
        widths = {str(li): dict(calibrated=r["calibrated"], traffic=r["calibrated_traffic"],
                                loose_4p2=any(x is not None and x > 20 for x in r["p8_42"].values()))
                  for li, r in P["rows"].items()}
        json.dump(widths, open(os.path.join(W, f"widths_{m}.json"), "w"), indent=1)
    # 6. MLA
    mla = {}
    for m in R:
        if not any("cert" in r for t in R[m].values() for r in t.values()):
            continue
        L = max(int(t.rsplit("_", 1)[1]) for t in R[m])
        mla[m] = P = mla_model(m, R[m], L)
        if not P["rows"]:
            continue
        md.append(f"\n## 6. MLA: {NAMES.get(m, m)} at {L} tokens ({len(P['rows'])} layers × {len(P['prompts'])} prompts)\n")
        md.append("| layer | flatness (min–max) | store choice (calibrated) | traffic | 4+2 misses · 4+4 | coupling share (max) | "
                  "coupled Theorem R at 4+2: coverage, bound / error (median over prompts) |")
        md.append("|---|---|---|---|---|---|---|")
        for li, r in P["rows"].items():
            c42 = [c["store:2/same"] for c in r["cert"].values() if "store:2/same" in c]
            md.append(f"| {li} | {pct(min(r['flat'].values()))}–{pct(max(r['flat'].values()))} | "
                      f"{r['calibrated'] or 'unmet'} | {r['calibrated_traffic']:.3f} | {len(r['miss42'])} · {len(r['miss44'])} | "
                      f"{r['cross_share']:.1e} | {min(x['coverage'] for x in c42):.3f}, {_med([x['ratio_med'] for x in c42]):.0f}× |")
    json.dump(dict(models=S, mla=mla, slack_law=law, slack_rows=rows), open(os.path.join(W, "summary_g0.json"), "w"),
              indent=1, default=str)
    open(os.path.join(W, "tables_g0.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
