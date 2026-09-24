#!/usr/bin/env python3
"""Authenticate and analyse the frozen R11 nested-code experiment (plan.md).

Entry points:
  --seal                  write source_ledger.json from the current source bytes
  --preflight             verify the ledger (worker start, before model load)
  --validate-task-dir D   authenticate one finished task (worker end)
  --pilot JOB             V1-V5 on the excluded pilot pair; writes the lock
  --main JOB              V1-V5 on four pairs, then the frozen statistics

A validity failure exits 2 as INVALID_R11; a design failure is a result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT))
from sievelib import quant as Q  # noqa: E402

PROTOCOL = "r11_nested_code_v1"
LEDGER = HERE / "source_ledger.json"
RESULTS = PROJECT / "h0_measurement" / "results"
LEDGER_SOURCES = (
    "sievelib/quant.py", "sievelib/alloc.py", "sievelib/evict.py",
    "sievelib/prompts.py", "sievelib/probe.py",
    "h0_measurement/run_h0.py", "h0_measurement/models.yaml",
    "h0_measurement/submit_r11_nested_code.slurm",
    "h0_measurement/bugs/11_nested_code_overhead/plan.md",
    "h0_measurement/bugs/11_nested_code_overhead/read_nested_code.py",
    ".h0_corpus/pg19/MANIFEST.json",
)
CHAIN = (3, 4, 6, 8)
BUDGETS = (1, 2, 3, 4)
BIT_LIST = (1, 2, 3, 4, 5, 6, 8)
PANEL = {"nested3": [0, 3, 4, 6, 8]}
STEP = 4
DRAWS = 10_000
IDENTITY_COLS = ["L", "tau", "c1_abs", "c2_abs", "c3_abs", "c5_abs",
                 "err_uniform1", "err_uniform2", "err_uniform3"]
KEY = ["prompt", "family", "step", "layer", "head"]


@dataclass(frozen=True)
class Cell:
    model: str
    ctx: int
    n_prompts: int
    offset: int
    families: tuple[str, ...]


PILOT = {0: (Cell("qwen3-1.7b", 2048, 1, 0, ("cont",)), "lloyd"),
         1: (Cell("qwen3-1.7b", 2048, 1, 0, ("cont",)), "nested3")}
_MAIN_CELLS = [Cell("llama31-8b", 32768, 6, 24, ("niah", "qa", "cont")),
               Cell("llama31-8b", 131072, 6, 24, ("niah", "qa", "cont")),
               Cell("qwen3-8b", 8192, 6, 24, ("niah", "qa", "cont")),
               Cell("qwen3-30b-a3b-2507", 8192, 4, 24, ("niah", "qa", "cont"))]
MAIN = {2 * i + a: (c, ("lloyd", "nested3")[a])
        for i, c in enumerate(_MAIN_CELLS) for a in (0, 1)}


class InvalidR11(RuntimeError):
    pass


def fail(msg: str):
    raise InvalidR11(msg)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------- ledger
def seal() -> None:
    src = {rel: sha(PROJECT / rel) for rel in LEDGER_SOURCES}
    tmp = LEDGER.with_name(LEDGER.name + ".tmp")
    tmp.write_text(json.dumps({"protocol": PROTOCOL, "source_sha256": src},
                              indent=2, sort_keys=True) + "\n")
    tmp.replace(LEDGER)
    print(f"sealed {len(src)} sources -> {LEDGER} sha256={sha(LEDGER)}")


def verify_ledger() -> str:
    if not LEDGER.is_file() or LEDGER.is_symlink():
        fail(f"missing source ledger {LEDGER}; run script.sh --seal")
    led = json.loads(LEDGER.read_text())
    if led.get("protocol") != PROTOCOL:
        fail("ledger protocol mismatch")
    src = led.get("source_sha256", {})
    if set(src) != set(LEDGER_SOURCES):
        fail("ledger source set differs from the frozen list")
    for rel, want in src.items():
        got = sha(PROJECT / rel)
        if got != want:
            fail(f"source drift for {rel}: ledger={want[:12]} current={got[:12]}")
    return sha(LEDGER)


# ----------------------------------------------------------------- tasks
def task_dir(phase: str, job: str, task: int) -> Path:
    return RESULTS / f"r11_nested_{phase}_{job}_{task}"


def load_task(phase: str, job: str, task: int, ledger_sha: str) -> dict:
    cell, codebook = (PILOT if phase == "pilot" else MAIN)[task]
    d = task_dir(phase, job, task)
    if not (d / "COMPLETE").is_file() and not (d / "COMPLETE.pending").is_file():
        fail(f"{d.name}: no COMPLETE marker")
    info = json.loads((d / "RUN_INFO.json").read_text())
    for k, v in (("protocol", PROTOCOL), ("phase", phase), ("task", task),
                 ("model", cell.model), ("ctx", cell.ctx), ("codebook", codebook),
                 ("source_ledger_sha256", ledger_sha)):
        if info.get(k) != v:
            fail(f"{d.name}: RUN_INFO {k}={info.get(k)!r}, expected {v!r}")
    pq = list(d.glob("h0_*.parquet"))
    if len(pq) != 1:
        fail(f"{d.name}: expected one parquet, found {len(pq)}")
    side = json.loads(pq[0].with_suffix(".json").read_text())
    cfg = side.get("config", {})
    checks = {
        "model": (side.get("model"), cell.model),
        "ctx": (int(side.get("ctx", -1)), cell.ctx),
        "prompt_block": (side.get("prompt_block"),
                         [cell.offset, cell.offset + cell.n_prompts - 1]),
        "budgets": (tuple(side.get("budgets", ())), BUDGETS),
        "bit_list": (tuple(side.get("bit_list", ())), BIT_LIST),
        "extra_budgets": (tuple(side.get("extra_budgets", ())), BUDGETS),
        "coarse_bits": (tuple(side.get("coarse_bits", ())), (4,)),
        "group_alloc": (side.get("group_alloc"), True),
        "tier_panel": (side.get("tier_panel"), PANEL),
        "synthetic": (side.get("synthetic"), False),
        "rot_seed": (side.get("rot_seed"), 2),
        "families": (tuple(cfg.get("families", ())), cell.families),
        "codebook": (side.get("codebook", "lloyd"), codebook),
        "config codebook": (cfg.get("codebook", "lloyd"), codebook),
        "interior": (side.get("interior_unseen_policy"), "floor_maxb"),
    }
    if codebook != "lloyd":
        checks["chain"] = (tuple(side.get("codebook_chain", ())), CHAIN)
    for k, (got, want) in checks.items():
        if got != want:
            fail(f"{d.name}: sidecar {k}={got!r}, expected {want!r}")
    df = pd.read_parquet(pq[0])
    df4 = df[df["step"].astype(int) == STEP].copy()
    if df4.empty:
        fail(f"{d.name}: no step-4 rows")
    want_p = set(range(cell.offset, cell.offset + cell.n_prompts))
    if set(df4["prompt"].astype(int)) != want_p:
        fail(f"{d.name}: prompt set {sorted(set(df4['prompt']))} != {sorted(want_p)}")
    if set(df4["family"].astype(str)) != set(cell.families):
        fail(f"{d.name}: family set differs")
    if df4.duplicated(KEY).any():
        fail(f"{d.name}: duplicate row keys")
    return {"cell": cell, "codebook": codebook, "dir": d, "frame": df4,
            "parquet_sha256": sha(pq[0])}


def _pfx(B: int) -> str:
    return f"grp_tier_nested3_csv_b4_accum_{B}"


def validate_pair(mono: dict, nest: dict) -> pd.DataFrame:
    name = f"{mono['cell'].model}@{mono['cell'].ctx}"
    a = mono["frame"].set_index(KEY).sort_index()
    b = nest["frame"].set_index(KEY).sort_index()
    if not a.index.equals(b.index):
        fail(f"V1 {name}: arms have different step-4 row keys")
    # V1 pairing identity
    for col in IDENTITY_COLS:
        x, y = a[col].to_numpy(float), b[col].to_numpy(float)
        rel = np.abs(x - y) / np.maximum(np.abs(x), 1e-30)
        if not np.isfinite(rel).all() or rel.max() > 1e-6:
            fail(f"V1 {name}: {col} differs between arms (max rel {np.nanmax(rel):.3e})")
    # V2 nesting applied
    for w in CHAIN[1:]:
        diff = (a[f"c{w}_abs"].to_numpy(float) != b[f"c{w}_abs"].to_numpy(float)).mean()
        if diff < 0.99:
            fail(f"V2 {name}: nested c{w}_abs equals mono on {1-diff:.1%} of rows")
    # V3 budget + V5 finiteness
    for arm, fr in (("mono", a), ("nested", b)):
        for B in BUDGETS:
            p = _pfx(B)
            e, mb = fr[f"err_wf_{p}"].to_numpy(float), fr[f"mean_bits_{p}"].to_numpy(float)
            if not (np.isfinite(e).all() and (e > 0).all()):
                fail(f"V5 {name}/{arm}: non-finite or non-positive error at B={B}")
            if (mb > B + 1e-9).any():
                fail(f"V3 {name}/{arm}: overspend at B={B}")
            tf = fr[[f"tier_frac_{p}_b{t}" for t in PANEL["nested3"]]].to_numpy(float)
            if not np.allclose(tf.sum(1), 1.0, atol=2e-9):
                fail(f"V3 {name}/{arm}: tier fractions do not sum to 1 at B={B}")
    return b


# ----------------------------------------------------------------- statistics
def prompt_E(frame: pd.DataFrame, B: int) -> pd.Series:
    e = frame[f"err_wf_{_pfx(B)}"].astype(float)
    return (e * e).groupby(frame["prompt"]).mean().pow(0.5)


def tier3_frac(frame: pd.DataFrame, B: int) -> pd.Series:
    phys = frame.sort_values(KEY).drop_duplicates(
        ["prompt", "family", "step", "layer", "kv_head"])
    col = f"tier_frac_{_pfx(B)}_b3"
    return (phys[col] * phys["L"]).groupby(phys["prompt"]).sum() / \
        phys["L"].groupby(phys["prompt"]).sum()


def matched_rate(budgets, logE_mono, logE_target) -> tuple[float, bool]:
    """B* with piecewise-linear log E_mono(B*) = target; (value, extrapolated)."""
    bs = np.asarray(budgets, float)
    y = np.asarray(logE_mono, float)
    if not (np.diff(y) < 0).all():
        fail(f"mono error curve is not strictly decreasing in B: {np.exp(y)}")
    for i in range(len(bs) - 1):
        if y[i] >= logE_target >= y[i + 1]:
            t = (y[i] - logE_target) / (y[i] - y[i + 1])
            return float(bs[i] + t * (bs[i + 1] - bs[i])), False
    i = 0 if logE_target > y[0] else len(bs) - 2
    t = (y[i] - logE_target) / (y[i] - y[i + 1])
    return float(bs[i] + t * (bs[i + 1] - bs[i])), True


def gaussian_theory() -> dict[str, Any]:
    x = torch.linspace(-9, 9, 400_001, dtype=torch.float64)
    p = torch.exp(-0.5 * x ** 2)
    p /= p.sum()

    def mse(lv, bd):
        return float((p * (x - lv[torch.bucketize(x, bd)]) ** 2).sum())
    mono = {}
    for w in range(1, 10):
        lv = Q.levels_for(w, "cpu").double()
        mono[w] = mse(lv, (lv[1:] + lv[:-1]) / 2)
    design = Q.design_nested(CHAIN)
    out = {}
    ws = sorted(mono)
    for b in CHAIN:
        D = mse(*design[b])
        r, _ = matched_rate(ws, [math.log(mono[w]) for w in ws], math.log(D))
        out[str(b)] = {"mse_ratio": D / mono[b], "equiv_rate": r,
                       "rate_overhead": b / r - 1}
    return out


def boot(values: dict[str, np.ndarray], rng) -> tuple[dict, tuple]:
    per = {}
    for c in sorted(values):
        v = np.asarray(values[c], float)
        idx = rng.integers(0, len(v), size=(DRAWS, len(v)))
        per[c] = v[idx].mean(1)
    stack = np.stack([per[c] for c in sorted(per)], 1)
    q = lambda d: tuple(float(z) for z in np.quantile(d, (0.05, 0.95)))  # noqa
    return {c: q(d) for c, d in per.items()}, q(stack.mean(1))


def analyse(pairs: list[tuple[dict, dict]]) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(0))
    res: dict[str, Any] = {"protocol": PROTOCOL, "budgets": {},
                           "gaussian_theory": gaussian_theory(), "cells": {}}
    frames = {}
    for mono, nest in pairs:
        name = f"{mono['cell'].model}@{mono['cell'].ctx}"
        frames[name] = (mono["frame"], nest["frame"])
        niah = nest["frame"][nest["frame"]["family"] == "niah"]
        res["cells"][name] = {
            "prompts": sorted(int(p) for p in nest["frame"]["prompt"].unique()),
            "niah_hit_prompts": (int(niah.groupby("prompt")["needle_hit"].all().sum())
                                 if len(niah) else None),
            "rows_step4": int(len(nest["frame"])),
        }
    Em = {n: pd.DataFrame({B: prompt_E(f[0], B) for B in BUDGETS}) for n, f in frames.items()}
    En = {n: pd.DataFrame({B: prompt_E(f[1], B) for B in BUDGETS}) for n, f in frames.items()}
    for B in (3, 2):
        cells, per_o, per_ot, per_r = {}, {}, {}, {}
        for n in frames:
            f3 = tier3_frac(frames[n][1], B)
            rows = []
            for p in Em[n].index:
                bstar, ext = matched_rate(BUDGETS, np.log(Em[n].loc[p].to_numpy()),
                                          math.log(En[n].loc[p, B]))
                rows.append({"prompt": int(p), "ratio": En[n].loc[p, B] / Em[n].loc[p, B],
                             "bstar": bstar, "extrapolated": ext, "f3": float(f3.loc[p]),
                             "O_nest": B / bstar - 1,
                             "O_total": (B + float(f3.loc[p])) / bstar - 1})
            t = pd.DataFrame(rows)
            per_o[n], per_ot[n], per_r[n] = t.O_nest.values, t.O_total.values, t.ratio.values
            cells[n] = {"error_ratio": float(np.exp(np.log(t.ratio).mean())),
                        "bstar": float(t.bstar.mean()),
                        "any_extrapolated": bool(t.extrapolated.any()),
                        "f3": float(t.f3.mean()),
                        "O_nest": float(t.O_nest.mean()),
                        "O_total": float(t.O_total.mean()),
                        "per_prompt": t.to_dict("records")}
        ci_o, mci_o = boot(per_o, rng)
        ci_ot, mci_ot = boot(per_ot, rng)
        for n in cells:
            cells[n]["O_nest_ci90"], cells[n]["O_total_ci90"] = ci_o[n], ci_ot[n]
        macro = {k: float(np.mean([cells[n][k] for n in cells]))
                 for k in ("O_nest", "O_total", "f3")}
        macro["error_ratio"] = float(np.exp(np.mean([math.log(cells[n]["error_ratio"])
                                                     for n in cells])))
        macro["O_nest_ci90"], macro["O_total_ci90"] = mci_o, mci_ot
        worst = max(cells[n]["O_total"] for n in cells)
        if macro["O_total"] > 0.20:
            dec = "stop_architecture"
        elif worst > 0.20:
            dec = "scope_per_model"
        elif macro["O_total"] > 0.10:
            dec = "pass_priced"
        else:
            dec = "pass_target"
        res["budgets"][str(B)] = {"cells": cells, "macro": macro,
                                  "worst_cell_O_total": worst, "decision": dec}
    # code-level: per-width logit-noise ratio and equivalent width
    code = {}
    for n, (m, q) in frames.items():
        ws = [1, 2, 3, 4, 5, 6, 8]
        gm = {w: np.exp(np.log(m[f"c{w}_abs"].clip(lower=1e-300)).groupby(m["prompt"]).mean())
              for w in ws}
        out = {}
        for b in CHAIN[1:]:
            gq = np.exp(np.log(q[f"c{b}_abs"].clip(lower=1e-300)).groupby(q["prompt"]).mean())
            rs = [matched_rate(ws, [math.log(gm[w].loc[p]) for w in ws], math.log(gq.loc[p]))[0]
                  for p in gq.index]
            out[str(b)] = {"noise_ratio": float(np.exp(np.log(gq / gm[b]).mean())),
                           "equiv_width": float(np.mean(rs)),
                           "O_code": float(np.mean([b / r - 1 for r in rs]))}
        code[n] = out
    res["code_level"] = code
    res["decision"] = res["budgets"]["3"]["decision"]
    res["validity"] = "valid_r11"
    return res


def render(res: dict) -> str:
    L = [f"R11 nested-code analysis ({PROTOCOL})", "Validity: valid_r11 (V1-V5 passed)", "",
         "Gaussian design check (width: MSE ratio, rate overhead):"]
    for b, v in res["gaussian_theory"].items():
        L.append(f"  {b}: {v['mse_ratio']:.4f}  {100*v['rate_overhead']:+.2f}%")
    for B in ("3", "2"):
        bd = res["budgets"][B]
        L += ["", f"B={B}: cell | E_nested/E_mono | B* | f3 | O_nest | O_total [90% CI]"]
        for n, c in bd["cells"].items():
            L.append(f"  {n} | {c['error_ratio']:.4f} | {c['bstar']:.4f}"
                     f"{' (extrap)' if c['any_extrapolated'] else ''} | {c['f3']:.4f} | "
                     f"{100*c['O_nest']:+.2f}% | {100*c['O_total']:+.2f}% "
                     f"[{100*c['O_total_ci90'][0]:+.2f}, {100*c['O_total_ci90'][1]:+.2f}]")
        m = bd["macro"]
        L.append(f"  macro | {m['error_ratio']:.4f} | | {m['f3']:.4f} | "
                 f"{100*m['O_nest']:+.2f}% | {100*m['O_total']:+.2f}% "
                 f"[{100*m['O_total_ci90'][0]:+.2f}, {100*m['O_total_ci90'][1]:+.2f}]")
        L.append(f"  worst cell O_total {100*bd['worst_cell_O_total']:+.2f}%  "
                 f"-> decision B={B}: {bd['decision']}")
    L += ["", "Code-level logit noise (cell: width noise_ratio equiv_width O_code):"]
    for n, v in res["code_level"].items():
        L.append("  " + n + "  " + "  ".join(
            f"{b}:{x['noise_ratio']:.3f}/{x['equiv_width']:.3f}/{100*x['O_code']:+.2f}%"
            for b, x in v.items()))
    L += ["", f"Frozen decision (B=3): {res['decision']}"]
    return "\n".join(L) + "\n"


def write(path: Path, text: str):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def run_phase(phase: str, job: str) -> None:
    lsha = verify_ledger()
    tasks = PILOT if phase == "pilot" else MAIN
    loaded = {t: load_task(phase, job, t, lsha) for t in sorted(tasks)}
    pairs = [(loaded[t], loaded[t + 1]) for t in sorted(tasks) if t % 2 == 0]
    for m, n in pairs:
        validate_pair(m, n)
    if phase == "pilot":
        lock = {"protocol": PROTOCOL, "decision": "advance_main", "pilot_job": job,
                "source_ledger_sha256": lsha,
                "artifacts": {str(t): x["parquet_sha256"] for t, x in loaded.items()}}
        write(HERE / f"pilot_advance_lock_{job}.json", json.dumps(lock, indent=2) + "\n")
        txt = (f"R11 excluded pilot {job}: PASS (V1 pairing, V2 nesting, V3 budget, "
               f"V4 provenance, V5 completeness)\n")
        write(HERE / f"pilot_{job}.txt", txt)
        print(txt, end="")
        return
    res = analyse(pairs)
    text = render(res)
    write(HERE / f"main_{job}.txt", text)
    write(HERE / f"main_{job}.json", json.dumps(res, indent=2, default=float) + "\n")
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
            print(f"PASS source ledger sha256={verify_ledger()}")
        elif a.pilot:
            run_phase("pilot", a.pilot)
        elif a.main:
            run_phase("main", a.main)
        else:
            x = load_task(a.phase, a.job, a.task, verify_ledger())
            print(f"PASS {a.phase} task {a.task}: {len(x['frame'])} step-4 rows")
        return 0
    except InvalidR11 as exc:
        print(f"INVALID_R11: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
