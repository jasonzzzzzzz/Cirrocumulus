#!/usr/bin/env python3
"""Authenticate and analyse the frozen R11-ext experiment (plan_ext.md).

Reuses R11's frozen pair validation (V1-V3) and per-prompt definitions from
read_nested_code.py; adds multi-task cells, the running-minimum B* rule,
tail rates with Clopper-Pearson intervals, the section 6 decisions and the
section 7 mechanism checks.

  --seal | --preflight | --validate-task-dir D --phase P --task T --job J
  --pilot JOB | --main JOB
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
_spec = importlib.util.spec_from_file_location("r11_frozen", HERE / "read_nested_code.py")
R = importlib.util.module_from_spec(_spec)
sys.modules["r11_frozen"] = R
_spec.loader.exec_module(R)

PROTOCOL = "r11_nested_code_ext_v1"
LEDGER = HERE / "source_ledger_ext.json"
RESULTS = PROJECT / "h0_measurement" / "results"
LEDGER_SOURCES = (
    "sievelib/quant.py", "sievelib/alloc.py", "sievelib/evict.py",
    "sievelib/prompts.py", "sievelib/probe.py",
    "h0_measurement/run_h0.py", "h0_measurement/models.yaml",
    "h0_measurement/submit_r11_ext.slurm",
    "h0_measurement/bugs/11_nested_code_overhead/plan_ext.md",
    "h0_measurement/bugs/11_nested_code_overhead/read_nested_ext.py",
    "h0_measurement/bugs/11_nested_code_overhead/read_nested_code.py",
    ".h0_corpus/pg19/MANIFEST.json",
)
FAMS = ("niah", "qa", "cont")
BUDGETS = R.BUDGETS
DRAWS = 10_000


@dataclass(frozen=True)
class Task:
    cell: str
    model: str
    ctx: int
    n_prompts: int
    offset: int
    families: tuple[str, ...]
    dense: bool


PILOT = {0: Task("P0", "mistral-7b", 2048, 1, 0, ("cont",), True),
         1: Task("P1", "qwen15-moe-a2.7b", 2048, 1, 0, ("cont",), False)}
MAIN = {**{t: Task("X1", "qwen3-30b-a3b-2507", 8192, 4, 28 + 4 * t, FAMS, False) for t in range(6)},
        6: Task("X2", "qwen15-moe-a2.7b", 8192, 6, 24, FAMS, False),
        7: Task("X2", "qwen15-moe-a2.7b", 8192, 6, 30, FAMS, False),
        8: Task("X3", "mistral-7b", 8192, 6, 24, FAMS, True),
        9: Task("X4", "mistral-7b", 32768, 6, 24, FAMS, True)}
CELL_PROMPTS = {"X1": set(range(28, 52)), "X2": set(range(24, 36)),
                "X3": set(range(24, 30)), "X4": set(range(24, 30))}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg):
    raise R.InvalidR11(msg)


def seal():
    src = {rel: sha(PROJECT / rel) for rel in LEDGER_SOURCES}
    tmp = LEDGER.with_name(LEDGER.name + ".tmp")
    tmp.write_text(json.dumps({"protocol": PROTOCOL, "source_sha256": src},
                              indent=2, sort_keys=True) + "\n")
    tmp.replace(LEDGER)
    print(f"sealed {len(src)} sources -> {LEDGER} sha256={sha(LEDGER)}")


def verify_ledger() -> str:
    if not LEDGER.is_file() or LEDGER.is_symlink():
        fail(f"missing {LEDGER}; run script_ext.sh --seal")
    led = json.loads(LEDGER.read_text())
    if led.get("protocol") != PROTOCOL or set(led.get("source_sha256", {})) != set(LEDGER_SOURCES):
        fail("ext ledger protocol or source set mismatch")
    for rel, want in led["source_sha256"].items():
        got = sha(PROJECT / rel)
        if got != want:
            fail(f"source drift for {rel}: ledger={want[:12]} current={got[:12]}")
    return sha(LEDGER)


def task_dir(phase, job, task) -> Path:
    return RESULTS / f"r11x_{phase}_{job}_{task}"


def load_task(phase: str, job: str, task: int, lsha: str, d: Path | None = None):
    spec = (PILOT if phase == "pilot" else MAIN)[task]
    d = d or task_dir(phase, job, task)
    if not (d / "COMPLETE").is_file() and not (d / "COMPLETE.pending").is_file():
        fail(f"{d.name}: no COMPLETE marker")
    info = json.loads((d / "RUN_INFO.json").read_text())
    for k, v in (("protocol", PROTOCOL), ("layout", "in_process_ab"), ("phase", phase),
                 ("task", task), ("job", str(job)), ("model", spec.model),
                 ("ctx", spec.ctx), ("codebook", "lloyd"), ("codebook_ab", "nested3"),
                 ("source_ledger_sha256", lsha)):
        if info.get(k) != v:
            fail(f"{d.name}: RUN_INFO {k}={info.get(k)!r}, expected {v!r}")
    pq = list(d.glob("h0_*.parquet"))
    if len(pq) != 1:
        fail(f"{d.name}: expected one parquet, found {len(pq)}")
    side = json.loads(pq[0].with_suffix(".json").read_text())
    cfg = side.get("config", {})
    checks = {
        "model": (side.get("model"), spec.model), "ctx": (int(side.get("ctx", -1)), spec.ctx),
        "prompt_block": (side.get("prompt_block"), [spec.offset, spec.offset + spec.n_prompts - 1]),
        "budgets": (tuple(side.get("budgets", ())), BUDGETS),
        "bit_list": (tuple(side.get("bit_list", ())), R.BIT_LIST),
        "extra_budgets": (tuple(side.get("extra_budgets", ())), BUDGETS),
        "coarse_bits": (tuple(side.get("coarse_bits", ())), (4,)),
        "group_alloc": (side.get("group_alloc"), True),
        "tier_panel": (side.get("tier_panel"), R.PANEL),
        "synthetic": (side.get("synthetic"), False), "rot_seed": (side.get("rot_seed"), 2),
        "families": (tuple(cfg.get("families", ())), spec.families),
        "codebook": (side.get("codebook", "lloyd"), "lloyd"),
        "codebook_ab": (side.get("codebook_ab"), "nested3"),
        "codebook_ab_chain": (tuple(side.get("codebook_ab_chain", ())), R.CHAIN),
        "codebook_ab_suffix": (side.get("codebook_ab_suffix"), R.AB_SUFFIX),
        "codebook_ab_widths": (tuple(side.get("codebook_ab_widths", ())), R.CHAIN[1:]),
        "interior": (side.get("interior_unseen_policy"), "floor_maxb"),
    }
    for k, (got, want) in checks.items():
        if got != want:
            fail(f"{d.name}: sidecar {k}={got!r}, expected {want!r}")
    df = pd.read_parquet(pq[0])
    df4 = df[df["step"].astype(int) == R.STEP].copy()
    want_p = set(range(spec.offset, spec.offset + spec.n_prompts))
    if df4.empty or set(df4["prompt"].astype(int)) != want_p:
        fail(f"{d.name}: step-4 prompt set differs from {sorted(want_p)}")
    if set(df4["family"].astype(str)) != set(spec.families) or df4.duplicated(R.KEY).any():
        fail(f"{d.name}: family set differs or duplicate row keys")
    ab_cols = [c for c in df4.columns if c.endswith(R.AB_SUFFIX)]
    mono = df4.drop(columns=ab_cols)
    nest = mono.copy()
    for c in ab_cols:
        nest[c[: -len(R.AB_SUFFIX)]] = df4[c].to_numpy()
    cellobj = R.Cell(spec.model, spec.ctx, spec.n_prompts, spec.offset, spec.families)
    m = {"cell": cellobj, "frame": mono}
    n = {"cell": cellobj, "frame": nest}
    R.validate_pair(m, n)                                  # V1-V3, frozen
    return spec, m["frame"], n["frame"], sha(pq[0])


# ----------------------------------------------------------------- statistics
def bstar_runmin(logE_mono: np.ndarray, target: float) -> tuple[float, bool]:
    """B* on the running minimum of the mono curve (plan_ext 5)."""
    y = np.minimum.accumulate(np.asarray(logE_mono, float))
    flagged = not bool((np.diff(np.asarray(logE_mono, float)) < 0).all())
    bs = np.asarray(BUDGETS, float)
    for i in range(len(bs) - 1):
        if y[i] >= target >= y[i + 1] and y[i] > y[i + 1]:
            t = (y[i] - target) / (y[i] - y[i + 1])
            return float(bs[i] + t * (bs[i + 1] - bs[i])), flagged
    # beyond the curve: extrapolate on the nearest strictly decreasing segment
    segs = [i for i in range(len(bs) - 1) if y[i] > y[i + 1]]
    if not segs:
        fail("mono error curve is flat across every budget")
    i = segs[0] if target > y[0] else segs[-1]
    t = (y[i] - target) / (y[i] - y[i + 1])
    return float(bs[i] + t * (bs[i + 1] - bs[i])), True


def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def _solve(f, lo=0.0, hi=1.0) -> float:
    for _ in range(200):                  # f is monotone decreasing in p
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def cp(k: int, n: int) -> tuple[float, float]:
    """Exact Clopper-Pearson 95% interval (binomial tails, no scipy)."""
    lo = 0.0 if k == 0 else _solve(lambda p: 0.025 - (1 - _binom_cdf(k - 1, n, p)))
    hi = 1.0 if k == n else _solve(lambda p: _binom_cdf(k, n, p) - 0.025)
    return lo, hi


def top_share(frame: pd.DataFrame, B: int) -> pd.Series:
    e = frame[f"err_wf_{R._pfx(B)}"].astype(float) ** 2
    g = e.groupby([frame["prompt"], frame["family"], frame["layer"], frame["kv_head"]]).sum()
    tot = g.groupby(level="prompt").sum()
    return (g.groupby(level="prompt").max() / tot)


def flip_rate(m: pd.DataFrame, n: pd.DataFrame, B: int) -> pd.Series:
    key = ["prompt", "family", "layer", "kv_head"]
    col = f"err_wf_{R._pfx(B)}"
    a = (m[col].astype(float) ** 2).groupby([m[k] for k in key]).mean() ** 0.5
    b = (n[col].astype(float) ** 2).groupby([n[k] for k in key]).mean() ** 0.5
    r = (b / a)
    return ((r >= 2) | (r <= 0.5)).groupby(level="prompt").mean()


def cell_stats(cell: str, mono: pd.DataFrame, nest: pd.DataFrame, rng) -> dict:
    Em = pd.DataFrame({B: R.prompt_E(mono, B) for B in BUDGETS})
    En = pd.DataFrame({B: R.prompt_E(nest, B) for B in BUDGETS})
    out = {"n_prompts": int(len(Em)), "budgets": {}}
    for B in (3, 2):
        f3 = R.tier3_frac(nest, B)
        ts = top_share(mono, B)
        fr = flip_rate(mono, nest, B)
        rows = []
        for p in Em.index:
            bs, flag = bstar_runmin(np.log(Em.loc[p].to_numpy()), math.log(En.loc[p, B]))
            rows.append({"prompt": int(p), "ratio": float(En.loc[p, B] / Em.loc[p, B]),
                         "bstar": bs, "flagged": flag, "f3": float(f3.loc[p]),
                         "O_total": (B + float(f3.loc[p])) / bs - 1,
                         "top_share": float(ts.loc[p]), "flip_rate": float(fr.loc[p])})
        t = pd.DataFrame(rows)
        O = t.O_total.to_numpy()
        draws = O[rng.integers(0, len(O), size=(DRAWS, len(O)))].mean(1)
        k20 = int((O > 0.20).sum())
        out["budgets"][str(B)] = {
            "mean_O_total": float(O.mean()),
            "mean_O_total_ci90": tuple(float(x) for x in np.quantile(draws, (.05, .95))),
            "median_O_total": float(np.median(O)), "max_O_total": float(O.max()),
            "tail20_k": k20, "tail20_rate": k20 / len(O), "tail20_cp95": cp(k20, len(O)),
            "error_ratio_gmean": float(np.exp(np.log(t.ratio).mean())),
            "f3_mean": float(t.f3.mean()), "flagged_prompts": int(t.flagged.sum()),
            "top_share_max": float(t.top_share.max()), "flip_rate_mean": float(t.flip_rate.mean()),
            "per_prompt": t.to_dict("records"),
        }
    return out


def decide(cells: dict) -> dict:
    m3 = {c: v["budgets"]["3"]["mean_O_total"] for c, v in cells.items()}
    m2 = {c: v["budgets"]["2"]["mean_O_total"] for c, v in cells.items()}
    macro3 = float(np.mean(list(m3.values())))
    if macro3 > 0.20:
        b3 = "b3_stop"
    elif max(m3.values()) > 0.20:
        b3 = "b3_scope"
    elif macro3 > 0.10:
        b3 = "b3_priced"
    else:
        b3 = "b3_generalizes"
    dense = [c for c in m2 if MAIN_DENSE[c]]
    moe = [c for c in m2 if not MAIN_DENSE[c]]
    if any(m2[c] > 0.20 for c in dense):
        b2 = "b2_fail"
    elif any(m2[c] > 0.20 for c in moe):
        b2 = "b2_dense_only"
    elif max(m2.values()) > 0.10:
        b2 = "b2_priced"
    else:
        b2 = "b2_all"
    x1 = cells["X1"]["budgets"]["2"]
    q1_tail_note = bool(x1["mean_O_total"] <= 0.20 and x1["tail20_cp95"][1] > 0.25)
    # section 7 predictions
    p1 = all(cells[c]["budgets"]["3"]["top_share_max"] <= 0.10 and
             all(abs(r["O_total"]) <= 0.10 for r in cells[c]["budgets"]["3"]["per_prompt"])
             for c in ("X3", "X4"))
    tails = [r for c in cells for r in cells[c]["budgets"]["2"]["per_prompt"] if r["O_total"] > 0.20]
    p2 = all(r["top_share"] >= 0.08 for r in tails)
    return {"B3": b3, "B2": b2, "macro_B3": macro3, "cell_mean_B3": m3, "cell_mean_B2": m2,
            "q1_passes_on_average_with_tail": q1_tail_note,
            "mech_pred1_dense_concentration_and_no_tail": p1,
            "mech_pred2_tails_need_top_share_ge_0.08": p2, "n_tail_prompts_B2": len(tails)}


MAIN_DENSE = {"X1": False, "X2": False, "X3": True, "X4": True}
NAMES = {"X1": "qwen3-30b-a3b-2507@8192", "X2": "qwen15-moe-a2.7b@8192",
         "X3": "mistral-7b@8192", "X4": "mistral-7b@32768"}


def render(res: dict) -> str:
    L = [f"R11-ext analysis ({PROTOCOL})", "Validity: valid_r11_ext (V1-V5 + cell assembly passed)",
         "Layout: in-process A/B (both codebooks from the same captured tensors)", ""]
    for B in ("3", "2"):
        L.append(f"B={B}: cell | n | E ratio | mean O_total [90% CI] | median | max | tail>20% k/n [CP95] | f3 | top_share max | flip rate | flagged")
        for c, v in res["cells"].items():
            b = v["budgets"][B]
            L.append(f"  {c} {NAMES[c]} | {v['n_prompts']} | {b['error_ratio_gmean']:.4f} | "
                     f"{100*b['mean_O_total']:+.2f}% [{100*b['mean_O_total_ci90'][0]:+.2f}, {100*b['mean_O_total_ci90'][1]:+.2f}] | "
                     f"{100*b['median_O_total']:+.2f}% | {100*b['max_O_total']:+.1f}% | "
                     f"{b['tail20_k']}/{v['n_prompts']} [{100*b['tail20_cp95'][0]:.0f}, {100*b['tail20_cp95'][1]:.0f}]% | "
                     f"{b['f3_mean']:.4f} | {b['top_share_max']:.3f} | {100*b['flip_rate_mean']:.2f}% | {b['flagged_prompts']}")
        L.append("")
    d = res["decision"]
    L += [f"macro mean O_total B=3 (X1-X4): {100*d['macro_B3']:+.2f}%",
          f"Frozen decision B=3: {d['B3']}", f"Frozen decision B=2: {d['B2']}",
          f"Q1 'passes on average with a per-prompt tail': {d['q1_passes_on_average_with_tail']}",
          f"Mechanism prediction 1 (dense: top_share<=0.10, |O|<=10% at B=3): "
          f"{'CONFIRMED' if d['mech_pred1_dense_concentration_and_no_tail'] else 'REFUTED'}",
          f"Mechanism prediction 2 (B=2 tails need top_share>=0.08; {d['n_tail_prompts_B2']} tail prompts): "
          f"{'CONFIRMED' if d['mech_pred2_tails_need_top_share_ge_0.08'] else 'REFUTED'}"]
    return "\n".join(L) + "\n"


def write(path: Path, text: str):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def run(phase: str, job: str):
    lsha = verify_ledger()
    table = PILOT if phase == "pilot" else MAIN
    loaded = {t: load_task(phase, job, t, lsha) for t in sorted(table)}
    if phase == "pilot":
        lock = {"protocol": PROTOCOL, "layout": "in_process_ab", "decision": "advance_main",
                "pilot_job": job, "source_ledger_sha256": lsha,
                "artifacts": {str(t): x[3] for t, x in loaded.items()}}
        write(HERE / f"pilot_ext_advance_lock_{job}.json", json.dumps(lock, indent=2) + "\n")
        txt = f"R11-ext excluded pilot {job}: PASS (V1-V5 on mistral-7b and qwen15-moe @2K)\n"
        write(HERE / f"pilot_ext_{job}.txt", txt)
        print(txt, end="")
        return
    # cell assembly check (plan_ext 4)
    cells_m, cells_n = {}, {}
    for t, (spec, m, n, _) in loaded.items():
        cells_m.setdefault(spec.cell, []).append(m)
        cells_n.setdefault(spec.cell, []).append(n)
    rng = np.random.Generator(np.random.PCG64(0))
    res = {"protocol": PROTOCOL, "main_job": job, "cells": {}}
    for c in ("X1", "X2", "X3", "X4"):
        m = pd.concat(cells_m[c], ignore_index=True)
        n = pd.concat(cells_n[c], ignore_index=True)
        got = set(m["prompt"].astype(int))
        if got != CELL_PROMPTS[c] or m.duplicated(R.KEY).any():
            fail(f"cell {c}: prompts {sorted(got)} != planned or overlap across tasks")
        res["cells"][c] = cell_stats(c, m, n, rng)
    res["decision"] = decide(res["cells"])
    res["validity"] = "valid_r11_ext"
    text = render(res)
    write(HERE / f"main_ext_{job}.txt", text)
    write(HERE / f"main_ext_{job}.json", json.dumps(res, indent=2, default=float) + "\n")
    print(text, end="")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seal", action="store_true")
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--pilot")
    g.add_argument("--main")
    g.add_argument("--validate-task-dir")
    ap.add_argument("--phase")
    ap.add_argument("--task", type=int)
    ap.add_argument("--job")
    a = ap.parse_args(argv)
    try:
        if a.seal:
            seal()
        elif a.preflight:
            print(f"PASS ext source ledger sha256={verify_ledger()}")
        elif a.pilot:
            run("pilot", a.pilot)
        elif a.main:
            run("main", a.main)
        else:
            spec, m, _, _ = load_task(a.phase, a.job, a.task, verify_ledger(), Path(a.validate_task_dir))
            print(f"PASS {a.phase} task {a.task} ({spec.cell} {spec.model}@{spec.ctx}): "
                  f"{len(m)} step-4 rows, V1-V3 hold in-process")
        return 0
    except R.InvalidR11 as exc:
        print(f"INVALID_R11_EXT: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
