"""Summaries of the 2026-10-12 CPU checks (findings_nested.md §6, findings_klbridge.md §6), from the run files in this
folder (klbridge_runs). From the repo root:
    .venv/bin/python h0_measurement/bugs/14h_theory/findings/klbridge_runs/checks_summary.py \
        h0_measurement/bugs/14h_theory/findings/klbridge_runs <out.json>
The run files and the earlier runs (the first argument) are both this folder."""
import glob
import json
import math
import os
import statistics as st
import sys

CHK = os.path.dirname(os.path.abspath(__file__))
KLR = sys.argv[1]
OUT = sys.argv[2]
FRAC = 8


def traffic(rows, kb=6, vb=4):
    return (1 - rows) * (kb + vb) / 32 + rows


def gmean(xs):
    return math.exp(st.mean(math.log(x) for x in xs))


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


res = {}
# ------------------------------------------------------------------ check 1: the vote
vote = {}
for model in ("llama1b", "qwen3_1p7b"):
    units = {}
    for f in sorted(glob.glob(f"{CHK}/{model}_*_w[01]_vote.json") + glob.glob(f"{CHK}/{model}_*_w[01]_vote_s2-7.json")):
        r = json.load(open(f))
        u = (r["text"], r["window"])
        d = units.setdefault(u, dict(kl={}, layers=[]))
        for k, v in r["vote"]["kl"].items():
            d["kl"].setdefault(k, []).extend(v)
        d["layers"].append(r["vote"]["layers"])
    stats = {}
    for u, d in units.items():
        lay = d["layers"][0]
        L = len(lay)
        for nm in ("exact", "lm", "dither0"):
            for key in ("mass", "unread_sm", "common_with_exact"):
                stats.setdefault(f"{nm} {key}", []).append(st.mean(lay[str(li)][nm][key] for li in range(L)))
        stats.setdefault("lm~dither common", []).append(st.mean(lay[str(li)]["common_lm_dither0"] for li in range(L)))
        dm = [lay[str(li)]["dither0"]["mass"] - lay[str(li)]["lm"]["mass"] for li in range(L)]
        stats.setdefault("dither-lm mass, worst layer", []).append(max(dm, key=abs))
    km = {u: {k: st.mean(v) for k, v in d["kl"].items()} for u, d in units.items()}
    nseed = {u: len(d["kl"]["nested 4+2/4 | dither"]) for u, d in units.items()}
    ratios = {}
    for base in ("nested 4+2/4", "tail 6/4"):
        for a, b in (("dither", "lm"), ("dither", "exact"), ("lm", "exact")):
            ratios[f"{base}: {a} / {b} vote"] = [km[u][f"{base} | {a}"] / km[u][f"{base} | {b}"] for u in km]
    # the deployed comparison: the single store (nested, its own dither vote) against R6's design (uniform 6/4, LM vote)
    dep = [km[u]["nested 4+2/4 | dither"] / km[u]["tail 6/4 | lm"] for u in km]
    same = [km[u]["nested 4+2/4 | exact"] / km[u]["tail 6/4 | exact"] for u in km]
    rom = st.mean(km[u]["nested 4+2/4 | dither"] for u in km) / st.mean(km[u]["tail 6/4 | lm"] for u in km)
    rom_same = st.mean(km[u]["nested 4+2/4 | exact"] for u in km) / st.mean(km[u]["tail 6/4 | exact"] for u in km)
    vote[model] = dict(
        units={f"{u[0]}_w{u[1]}": dict(kl=km[u], seeds=nseed[u]) for u in km},
        layer_stats={k: dict(mean=st.mean(v), min=min(v), max=max(v)) for k, v in stats.items()},
        ratios={k: dict(med=st.median(v), gmean=gmean(v), min=min(v), max=max(v)) for k, v in ratios.items()},
        deployed=dict(med=st.median(dep), gmean=gmean(dep), min=min(dep), max=max(dep), ratio_of_means=rom),
        same_vote=dict(med=st.median(same), gmean=gmean(same), min=min(same), max=max(same), ratio_of_means=rom_same))
    print(f"== check 1, {model}: {len(km)} units, seeds per unit {sorted(set(nseed.values()))}")
    for k, v in vote[model]["layer_stats"].items():
        print(f"  {k:32s} {v['mean']:.4f} ({v['min']:.4f} to {v['max']:.4f})")
    for k, v in vote[model]["ratios"].items():
        print(f"  {k:38s} median {v['med']:.3f}, geometric mean {v['gmean']:.3f} ({v['min']:.3f}-{v['max']:.3f})")
    print(f"  single store (own vote) / R6 design (own vote): {vote[model]['deployed']}")
    print(f"  single store / R6 design under the exact vote:  {vote[model]['same_vote']}")
res["vote"] = vote

# ------------------------------------------------------------------ check 2: the deployed combination
def arm_runs(model, text, nm, d=CHK, pre=""):
    out = []
    for f in sorted(glob.glob(f"{d}/{model}_{text}_w1_{pre}{nm}_s*.json")):
        r = json.load(open(f))
        for x in r["arms"][nm]:
            out.append((x["kl_mean"], st.mean(x["rows"].values()) if x["rows"] else 1 / FRAC))
    return out


def configs(model, text, w=1):
    p = f"{KLR}/{model}_{text}_w{w}_configs.json"
    r = json.load(open(p))
    return {k: st.mean(x["kl_mean"] for x in v) for k, v in r["configs"].items()}


old = json.load(open(f"{KLR}/cert_e2e_summary.json"))
dep = {}
for model, texts, pre in (("llama1b", ("pg19", "code", "table", "agent"), ""), ("qwen3_1p7b", ("pg19", "code"), "qwen_")):
    rows = {}
    for t in texts:
        cf = configs(model, t)
        vk = vote[model]["units"][f"{t}_w1"]["kl"]
        row = dict(fp8=cf["fp8"], r6=vk["tail 6/4 | lm"], single=vk["nested 4+2/4 | dither"])
        for lv in ("3e-05", "1e-05"):
            rr = arm_runs(model, t, f"certn_klb_{lv}", pre=pre)
            if rr:
                row[f"certn {lv}"] = dict(kl=st.mean(x[0] for x in rr), rows=st.mean(x[1] for x in rr), n=len(rr))
            if model == "llama1b":
                o = old[t].get(f"cert_klb_{lv}", [])
                extra = arm_runs(model, t, f"cert_klb_{lv}")       # the re-run missing seed (table, 1e-05, seed 1)
                o = o + [list(x) for x in extra if all(abs(x[0] - y[0]) > 1e-15 for y in o)]
                if o:
                    row[f"cert {lv}"] = dict(kl=st.mean(x[0] for x in o), rows=st.mean(x[1] for x in o), n=len(o))
        rows[t] = row
    dep[model] = rows
    print(f"== check 2, {model}: KL per token (2 seeds), traffic")
    for t, row in rows.items():
        s = f"  {t:6s} FP8 {row['fp8']:.2e} | R6 design {row['r6']:.2e} | single store {row['single']:.2e}"
        for k in ("cert 3e-05", "certn 3e-05", "cert 1e-05", "certn 1e-05"):
            if k in row:
                s += f" | {k} {row[k]['kl']:.2e} @{traffic(row[k]['rows']):.4f} (n={row[k]['n']})"
        print(s)
    for k in ("certn 3e-05", "certn 1e-05", "cert 3e-05", "cert 1e-05"):
        xs = [r[k] for r in rows.values() if k in r]
        if len(xs) == len(rows):
            g = gmean([r[k]["kl"] / r["single"] for r in rows.values()])
            g6 = gmean([r[k]["kl"] / r["r6"] for r in rows.values()])
            g8 = gmean([r["fp8"] / r[k]["kl"] for r in rows.values()])
            print(f"  {k}: mean KL {st.mean(x['kl'] for x in xs):.2e}, traffic {traffic(st.mean(x['rows'] for x in xs)):.4f}; "
                  f"geometric mean over units of KL / single store {g:.3f}, / R6 design {g6:.3f}; FP8 / it {g8:.1f}")
    print(f"  single store mean KL {st.mean(r['single'] for r in rows.values()):.2e}; R6 design {st.mean(r['r6'] for r in rows.values()):.2e}; "
          f"FP8 {st.mean(r['fp8'] for r in rows.values()):.2e}")
res["deployed"] = dep

# ------------------------------------------------------------------ check 4: fp8q against klb at matched traffic (Llama-1B)
tf = json.load(open(f"{KLR}/taufront_llama1b.json"))
pts = {}
for nm, rule, p in (("cert_flat_a", "flat", 0.1625), ("cert_flat_b", "flat", 0.02935),
                    ("cert_fp8q_3e-05", "fp8q", 1.0), ("cert_fp8q_1e-05", "fp8q", 0.95),
                    ("cert_klb_3e-05m", "klb", 10 ** -1.5), ("cert_klb_1e-05m", "klb", 10 ** -3.0),
                    ("cert_klb_3e-05", "klb", 10 ** -0.75), ("cert_klb_1e-05", "klb", 10 ** -2.25)):
    fr = min(tf["front"][rule], key=lambda x: abs(math.log(x[2] / p))) if rule in tf["front"] else (float("nan"),) * 3
    per = {}
    for t in ("pg19", "code", "table", "agent"):
        rr = arm_runs("llama1b", t, nm)
        if nm in ("cert_klb_3e-05", "cert_klb_1e-05"):                 # the earlier runs, plus any re-run seed
            o = [tuple(x) for x in old[t].get(nm, [])]
            rr = o + [x for x in rr if all(abs(x[0] - y[0]) > 1e-15 for y in o)]
        if rr:
            per[t] = dict(kl=st.mean(x[0] for x in rr), rows=st.mean(x[1] for x in rr), n=len(rr))
    if len(per) == 4:
        pts[nm] = dict(rule=rule, param=p, front_traffic=fr[0], front_pred=fr[1], units=per,
                       kl=st.mean(v["kl"] for v in per.values()), traffic=traffic(st.mean(v["rows"] for v in per.values())))
print("== check 4, Llama-1B test units: mean KL / traffic, against the frontier's prediction")
for nm, v in sorted(pts.items(), key=lambda x: x[1]["traffic"]):
    print(f"  {nm:18s} traffic {v['traffic']:.4f} (front {v['front_traffic']:.4f}); KL {v['kl']:.2e} (predicted {v['front_pred']:.2e}, "
          f"x{v['kl'] / v['front_pred']:.2f}); per unit " + ", ".join(f"{t} {x['kl']:.1e}" for t, x in v["units"].items()))
res["fp8q_vs_klb"] = pts
json.dump(res, open(OUT, "w"), indent=1)
print("->", OUT)

# ------------------------------------------------------------------ check 2b: Qwen, 8-bit keys in the noisy layers (σ ≥ 0.25 at 6 bits)
NOISY = {0, 1, 2, 3, 4, 5, 10}


def traffic_mixed(rows_by_layer, L=28):
    tot = 0.0
    for li in range(L):
        r = rows_by_layer.get(str(li), rows_by_layer.get(li, 1 / FRAC)) if rows_by_layer else 1 / FRAC
        kb = 8 if li in NOISY else 6
        tot += (1 - r) * (kb + 4) / 32 + r
    return tot / L


q8 = {}
for nm in ("certn8_klb_3e-05", "certn8_klb_1e-05", "certn8_flat_c", "certn8_flat_a", "certn8_flat_b", "nested8_vote",
           "certn_klb_3e-05", "certn_klb_1e-05"):
    per = {}
    for t in ("pg19", "code"):
        xs = []
        for f in sorted(glob.glob(f"{CHK}/qwen3_1p7b_{t}_w1_qwen_{nm}_s*.json")):
            x = json.load(open(f))["arms"][nm][0]
            tr = traffic_mixed(x["rows"]) if nm.startswith(("certn8", "nested8")) else traffic(st.mean(x["rows"].values()))
            xs.append((x["kl_mean"], tr, x["rows"]))
        if xs:
            per[t] = dict(kl=st.mean(x[0] for x in xs), traffic=st.mean(x[1] for x in xs), n=len(xs),
                          rows={li: st.mean(float(x[2][li]) for x in xs) for li in xs[0][2]} if xs[0][2] else {})
    if len(per) == 2:
        q8[nm] = dict(units=per, kl=st.mean(v["kl"] for v in per.values()), traffic=st.mean(v["traffic"] for v in per.values()))
print("== check 2b, Qwen3-1.7B, held-out units (2 seeds): KL per token, traffic")
for nm, v in q8.items():
    print(f"  {nm:18s} mean KL {v['kl']:.2e} traffic {v['traffic']:.4f}; " + ", ".join(
        f"{t} {x['kl']:.1e} @{x['traffic']:.4f} (n={x['n']})" for t, x in v["units"].items()))
    if v["units"]["pg19"]["rows"]:
        r_ = v["units"]["pg19"]["rows"]
        print("     pg19 rows by layer: " + " ".join(f"{float(r_[k]):.3f}" for k in sorted(r_, key=int)))
res["qwen_8bit_noisy"] = q8
json.dump(res, open(OUT, "w"), indent=1)
