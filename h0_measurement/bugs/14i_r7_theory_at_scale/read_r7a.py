#!/usr/bin/env python3
"""R7a's reader and gate (plan.md). Written 2026-10-11 before any R7a output existed; the rules freeze with the run
(r7_lib.AMEND_R7A = 'R7a'). R7a is a VERIFICATION run: its labels say whether the theory's predictions (theory_r7.md §4,
plan.md §5, stated before the run) hold on the paper's models and lengths. It tests no design claim.

    python read_r7a.py --r7a TAG:JOB [TAG:JOB ...] [--out-stem findings/R7a_reader] [--r6-json PATH]
    python read_r7a.py --gate TAG:JOB [TAG:JOB]      (the pilots, r7_lib.R7_PILOT; writes findings/R7a_gate.json)

UNITS. Prompt x task; families as R6 (read_stage1h_r5.FAMILY). Groups: llama128 (Llama-3.1-8B, 128K), qwen32
(Qwen3-30B-A3B, 32K), qwen128 (Qwen3-30B-A3B, 128K; HELMET, the probe only). A split row's blocks are read as JOB_0, JOB_1.
Bootstraps resample UNITS (head-steps of a unit share its draws), 2,000 times, numpy default_rng(0).

PANEL A (probe blocks; probe_r7's columns), per group and key width k in {4, 6, 8}, over every measured head-step:
  COVERAGE (Theorem R: a guarantee). checks = sum of draws; misses = sum (1 - cov) x draws, for part 8 and for v2.
    COVER_OK if misses <= delta checks + 3 sqrt(delta checks) + 3 (delta = 1e-3), else COVER_EXCESS.
  TIGHTNESS. Median (p10, p90) over head-steps of bound / error, part 8 and v2, and of their ratio.
  CERTIFIED. Mean over head-steps of the share of draws with the bound within FP8's per-head error.
  THEOREM M. Over head-steps with m_pred > 0: R_exact = mean of nn / m_pred (E (Z~/Z)^2 |o - o~|^2 = m_pred exactly, so
    it estimates 1) and R_first = mean of mse / m_pred (the first-order reading), with 95% unit-bootstrap intervals.
  LAYER MAP. Per layer: the median of sqrt(m_pred) / FP8's error (the tail's predicted RMS error over FP8's) and the share
    of head-steps above 1; the unread mass and the unread share of sum w^2 |v - o|^2 (share2).
  CERTIFIED READ (k in {6, 8}, the first chunk of steps, draw 0; target FP8's per-head error): mean rows read by the
    rule and by the oracle stop, traffic (1 - f)(k + 4)/32 + f, the exact-read fallback share, and the share of QUERY
    HEADS whose output met the target. That is the guarantee: each set's bound holds per head-step at delta / m, so by
    the union bound over the m sets P(miss) <= delta for each query head (for all r heads of a group at once only
    <= r delta); labelled with the coverage rule above (misses against delta x heads). The share of groups with every
    head met is reported beside it.
  FAMILIES. share2 and part 8's certified share per family; with --r6-json (R6's reader JSON), the Spearman correlation of
    share2 with the design's KL per family.
PANEL B (seed-panel blocks), per group and family: y_us = KL_tok(design at seed s) - 1.5 KL_tok(FP8), s = 0..4 (s = 0 is
  R6's seed 7000). Model y_us = mu + a_u + c_s + f_us: a_u the unit, c_s a seed effect COMMON to every unit (e.g. a
  near-constant heavy row, such as the attention sink, perturbed alike in every unit), f_us the rest (a fixed seed still
  gives each unit its own effective draw, so f_us is already inside R6's unit bootstrap). Two-way ANOVA without
  replication gives var_unit, var_seed_common, var_residual. U_s = R6's statistic at seed s (one-sided 95% unit-bootstrap
  upper bound of mean_u y_us; U_0 = R6's seed); U_seed_level = the bootstrap that draws units and ONE seed per replicate
  (the common effect); U_two = units, then a seed per drawn unit. SEED_ROBUST if U_0 < 5e-4 and every U_s and U_seed_level
  are < 5e-4; SEED_SENSITIVE if U_0 < 5e-4 and one is not; SEED0_NOT_PASSING if U_0 >= 5e-4.
PREDICTIONS (plan.md §5; theory_r7.md §4): P1 coverage COVER_OK everywhere; P2 part 8's median bound / error at 6 bits
  in [4, 7] (Llama 128K), [6, 13] (Qwen 32K and, extrapolated, Qwen 128K); P3 part 8 certified within FP8 at 6 bits
  >= 0.80 (Llama), >= 0.65 (Qwen 32K, Qwen 128K); P4 Theorem M: R_exact's 95% interval meets [0.98, 1.02] at every width
  (consistent with the exact identity), R_first in [0.95, 1.05] at 6 and 8 bits; P5 the certified read at 6 bits reaches
  traffic <= 0.45 (Llama), <= 0.50 (Qwen), and its per-query-head misses are COVER_OK; P6 part 8 / v2 <= 0.30 (median,
  6 bits); P7 SEED_ROBUST in every family; P8 (with --r6-json) share2 correlates positively with the design's KL across
  Llama 128K's families (Spearman > 0). Each is MET or NOT_MET (or NOT_READ if its data are missing).
VALIDITY (per block; a failing block is excluded and reported): an R7a block (sidecar r7a.amend = AMEND_R7A, or the
  draft for smokes) whose tag, preset, suite and tasks are a row of R7_LAYOUT; the preset's plan; every arm on every unit
  and nothing else; R7a's units for the tag; the probe ran, its pass is FP's (|kl_all| <= 1e-6) and its settings are
  probe_r7.settings(); the dtail arms evict nothing; peak <= 76 GiB per device; HELMET blocks carry R6's manifest; the
  code manifest is the same in every block.
GATE (the pilots; excluded): validity in pilot mode; FP's KL to itself 0; the probe's misses COVER_OK; peak <= 76 GiB
  per device; each 128K block's projected time (its units per process x the pilot's seconds per unit of the same preset
  kind, plus 15 minutes; Qwen 128K at 1.5 x the Llama probe pilot's + 60 s) within 90% of its wall. Exits 1 on FAIL,
  so the blocks (afterok) never start.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if METH not in sys.path:
    sys.path.append(METH)
import r7_lib as L7  # noqa: E402
import probe_r7 as PR  # noqa: E402

H0 = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(H0, "results")
FINDINGS = os.path.join(HERE, "findings")
N_BOOT, SEED, ALPHA = 2000, 0, 0.05
PEAK_LIMIT, PROBE_KL = 76.0, 1e-6
GATE_SETUP_H, GATE_WALL_FRAC = 0.25, 0.9
RHO, KAPPA = L7.R6_MARGIN
UKEY = ["prompt_idx", "task"]
GROUPS = {"llama128": "Llama-3.1-8B, 128K", "qwen32": "Qwen3-30B-A3B, 32K", "qwen128": "Qwen3-30B-A3B, 128K"}
PRED = dict(P2={"llama128": (4.0, 7.0), "qwen32": (6.0, 13.0), "qwen128": (6.0, 13.0)},
            P3={"llama128": 0.80, "qwen32": 0.65, "qwen128": 0.65},
            P4_exact=(0.98, 1.02), P4_first=(0.95, 1.05), P5={"llama128": 0.45, "qwen32": 0.50, "qwen128": 0.50}, P6=0.30)
GATE_Q128_SCALE, GATE_Q128_EXTRA_S = 1.5, 60.0     # Qwen 128K's probe unit: 48 layers against Llama's 32, same tensors


def family_of():
    import read_stage1h_r5 as R5
    return R5.FAMILY


# ------------------------------------------------------------------ loading and validity
def load_block(tag, job, root=RESULTS):
    d = os.path.join(root, f"r14r7a_{tag}_{job}")
    sp = glob.glob(os.path.join(d, "s1h_evaluate_*.json"))
    if len(sp) != 1:
        return None, None, None, None
    side = json.load(open(sp[0]))
    main = pd.read_parquet(os.path.join(d, side["parquet"]))
    files = (side.get("r7a") or {}).get("files") or {}
    heads = pd.read_parquet(os.path.join(d, files["r7a_heads"])) if "r7a_heads" in files else None
    groups = pd.read_parquet(os.path.join(d, files["r7a_groups"])) if "r7a_groups" in files else None
    return side, main, heads, groups


def _row(tag, preset, suite, tasks, layout=None):
    for row in (L7.R7_LAYOUT if layout is None else layout):
        if row[1] == tag and row[2] == preset and row[4] == suite and row[6] == tasks:
            return row
    return None


def validity(tag, side, d, heads, problems, pilot=False, smoke=False):
    r7 = side.get("r7a") or {}
    preset = side.get("preset_name")
    amend_ok = r7.get("amend") == L7.AMEND_R7A or (smoke and r7.get("amend", "").startswith("R7a"))
    if not amend_ok or preset not in L7.R7_PRESETS or (preset.endswith("smoke") and not smoke):
        problems.append(f"not an R7a block (r7a.amend {r7.get('amend')!r}, preset {preset!r})")
        return
    tasks = ",".join(side.get("tasks") or [])
    if not smoke and _row(tag, preset, side.get("suite"), tasks, L7.R7_PILOT if pilot else None) is None:
        problems.append(f"({tag}, {preset}, {side.get('suite')}, {tasks}) is not a row of the {'pilots' if pilot else 'layout'}")
    plan = [tuple(p) for p in side.get("plan", [])]
    want = [tuple(p) for p in L7.build_plan(L7.PRESETS[preset])]
    if [(a, float(b)) for a, b in plan] != [(a, float(b)) for a, b in want]:
        problems.append(f"plan {plan} is not the preset's {want}")
    units = d[d.arm == "fp"][UKEY].drop_duplicates()
    for a, b in want:
        n = d[(d.arm == a) & np.isclose(d.B.astype(float), float(b))][UKEY].drop_duplicates().shape[0]
        if n != len(units):
            problems.append(f"{a}@{b:g} on {n} of {len(units)} units")
    if len(d) != len(units) * len(want):
        problems.append(f"{len(d)} rows for {len(units)} units x {len(want)} arms")
    if not (pilot or smoke):
        pu = L7.r7_units()
        bad = [(int(p), t) for p, t in units.itertuples(index=False) if int(p) not in pu.get((tag, t), set())]
        if bad:
            problems.append(f"units outside R7a's plan for {tag}: {bad[:5]}")
    dt = d[d.arm.str.startswith("dtail")]
    if len(dt) and float(dt.evict_frac.fillna(0).abs().max()) > 0:
        problems.append("a dtail arm evicted rows")
    if L7.PRESETS[preset]["r7a"] == "probe":
        pr = d[d.arm == "probe"]
        if not len(pr) or float(pr.kl_all.abs().max()) > PROBE_KL:
            problems.append(f"the probe's pass is not FP's (max |kl_all| {float(pr.kl_all.abs().max()) if len(pr) else 'none'})")
        if heads is None or not len(heads):
            problems.append("no probe records")
        if not smoke and r7.get("probe") != PR.settings():
            problems.append("the probe's settings are not probe_r7's (frozen with the run)")
    peaks = [float(v) for v in side.get("peak_gib_dev_max") or []]
    if int(side.get("n_gpus") or 0) > 0 and (not peaks or max(peaks) > PEAK_LIMIT):
        problems.append(f"peak GPU memory per device {peaks}")
    if side.get("suite") == "r4":
        import tasks_s1h6 as T6
        if side.get("manifest_sha256") != T6.MANIFEST_SHA256:
            problems.append(f"HELMET manifest {str(side.get('manifest_sha256'))[:12]} is not R6's")


# ------------------------------------------------------------------ statistics
def _boot_ratio(num_by_unit, den_by_unit):
    """A ratio of sums over units and its 95% unit-bootstrap interval."""
    num, den = np.asarray(num_by_unit, float), np.asarray(den_by_unit, float)
    if not len(num) or den.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    idx = np.random.default_rng(SEED).integers(0, len(num), (N_BOOT, len(num)))
    bs = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-300)
    return float(num.sum() / den.sum()), float(np.quantile(bs, ALPHA / 2)), float(np.quantile(bs, 1 - ALPHA / 2))


def cover_label(misses, checks, delta):
    """COVER_OK if the misses are within delta x checks plus binomial noise (the theorem allows a miss rate of delta)."""
    return "COVER_OK" if misses <= delta * checks + 3.0 * math.sqrt(delta * checks) + 3.0 else "COVER_EXCESS"


def _q(x, p):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, p)) if len(x) else float("nan")


def panel_a(h, gr, delta=PR.DELTA, fam=None):
    """Panel A on one group's head records h and certified-read records gr (both with a 'unit' column)."""
    out = dict(head_steps=int(len(h)), units=int(h.unit.nunique()), bits={})
    for kb in PR.BITS:
        if f"cov_p8_k{kb}" not in h:
            continue
        r = {}
        nd = h[f"ndraw_k{kb}"].astype(float)
        checks = float(nd.sum())
        for b in ("p8", "v2"):
            miss = float(((1.0 - h[f"cov_{b}_k{kb}"]) * nd).sum())
            rat = h[f"rat_{b}_k{kb}"].replace([np.inf, -np.inf], np.nan)
            r[b] = dict(checks=int(round(checks)), misses=int(round(miss)), label=cover_label(miss, checks, delta),
                        ratio_med=_q(rat, 0.5), ratio_p10=_q(rat, 0.1), ratio_p90=_q(rat, 0.9),
                        certified_fp8=float(h[f"f8_{b}_k{kb}"].mean()))
        r["p8_over_v2_med"] = _q(h[f"rat_p8_k{kb}"] / h[f"rat_v2_k{kb}"], 0.5)
        r["p8_parts_K_V_D_med"] = [_q(h[f"p{x}_k{kb}"], 0.5) for x in ("K", "V", "D")]
        ok = h[f"m_pred_k{kb}"] > 0
        hu = h[ok].assign(rn=h.loc[ok, f"nn_k{kb}"] / h.loc[ok, f"m_pred_k{kb}"],
                          rf=h.loc[ok, f"mse_k{kb}"] / h.loc[ok, f"m_pred_k{kb}"])
        gu = hu.groupby("unit").agg(sn=("rn", "sum"), sf=("rf", "sum"), n=("rn", "size"))
        r["M_exact"] = _boot_ratio(gu.sn, gu.n)
        r["M_first"] = _boot_ratio(gu.sf, gu.n)
        r["M_first_med"] = _q(hu.rf, 0.5)
        r["sigma_med"] = _q(h[f"sig_k{kb}"], 0.5)
        tf8 = np.sqrt(h[f"m_pred_k{kb}"]) / h.err_fp8.clip(lower=1e-300)
        lay = h.assign(t=tf8).groupby("layer").t
        r["layers"] = {int(li): dict(tail_over_fp8_med=float(v.median()), above_fp8=float((v > 1).mean()))
                       for li, v in lay}
        r["layers_above_fp8_med"] = sorted(li for li, v in r["layers"].items() if v["tail_over_fp8_med"] > 1)
        r["tail_over_fp8_med"] = _q(tf8, 0.5)
        if gr is not None and len(gr) and kb in set(gr.bits):
            g = gr[gr.bits == kb]
            n_g = len(g)
            heads = float(g.r.sum())
            miss_h = heads - float(g.met_heads.sum())               # the guarantee: P(miss) <= delta per query head
            r["certread"] = dict(groups=int(n_g), rows_rule=float(g.rows.mean()), rows_oracle=float(g.rows_or.mean()),
                                 traffic_rule=float(((1 - g.rows) * (kb + 4) / 32 + g.rows).mean()),
                                 traffic_oracle=float(((1 - g.rows_or) * (kb + 4) / 32 + g.rows_or).mean()),
                                 met_heads=float(g.met_heads.sum() / heads), met_groups=float(g.met.mean()),
                                 head_checks=int(heads), head_misses=int(round(miss_h)),
                                 label=cover_label(miss_h, heads, delta), fallback=float(g.fallback.mean()),
                                 by_layer={int(li): float(v.rows.mean()) for li, v in g.groupby("layer")})
        out["bits"][str(kb)] = r
    out["unread_mass_med"] = _q(h.unread_mass, 0.5)
    out["share2_med"] = _q(h.share2, 0.5)
    if fam is not None:
        hf = h.assign(family=h.task.map(fam))
        out["families"] = {f: dict(share2_mean=float(v.share2.mean()), share2_med=float(v.share2.median()),
                                   certified_p8_k6=float(v.f8_p8_k6.mean()) if "f8_p8_k6" in v else float("nan"),
                                   head_steps=int(len(v))) for f, v in hf.groupby("family")}
    return out


def panel_b(rows, fam):
    """Panel B on one group's main rows (seed-panel blocks): the unit x seed table of y = KL_tok(design) - 1.5 KL_tok(FP8),
    its two-way decomposition y_us = mu + a_u + c_s + f_us (unit, seed common to every unit, residual), the per-seed
    upper bounds, and the two-level bootstrap."""
    arms = [L7.DESIGN] + [f"{L7.DESIGN}_s{k}" for k in range(1, L7.SEEDS_EXTRA + 1)]
    x = rows.assign(family=rows.task.map(fam))
    out = {}
    for f, g in x.groupby("family"):
        e = g[g.arm == "fp8kv"].set_index("unit").kl_span_tok
        Y = pd.concat([g[g.arm == a].set_index("unit").kl_span_tok.rename(a) for a in arms], axis=1)
        idx = Y.index.intersection(e.index)
        Y = Y.loc[idx].sub(RHO * e.loc[idx], axis=0)
        Y = Y[np.isfinite(Y.values).all(1)]
        if len(Y) < 2:
            out[f] = dict(units=int(len(Y)))
            continue
        y = Y.values                                           # [units n, seeds m]
        n, m = y.shape
        mu, a_u, c_s = y.mean(), y.mean(1), y.mean(0)
        res = y - a_u[:, None] - c_s[None, :] + mu
        ms_u = m * float(((a_u - mu) ** 2).sum()) / (n - 1)
        ms_s = n * float(((c_s - mu) ** 2).sum()) / (m - 1)
        ms_r = float((res ** 2).sum()) / ((n - 1) * (m - 1))
        var_c = max((ms_s - ms_r) / n, 0.0)                    # the seed effect common to every unit
        var_a = max((ms_u - ms_r) / m, 0.0)
        rng = np.random.default_rng(SEED)
        iu = rng.integers(0, n, (N_BOOT, n))
        U_by_seed = [float(np.quantile(y[iu, s_].mean(1), 1 - ALPHA)) for s_ in range(m)]
        js = rng.integers(0, m, (N_BOOT, n))
        U_two = float(np.quantile(y[iu, js].mean(1), 1 - ALPHA))
        s_blk = rng.integers(0, m, N_BOOT)                     # one seed for every unit of a replicate (the common effect)
        U_blk = float(np.quantile(y[iu, s_blk[:, None]].mean(1), 1 - ALPHA))
        U0 = U_by_seed[0]
        label = ("SEED0_NOT_PASSING" if U0 >= KAPPA else "SEED_ROBUST" if max(U_by_seed) < KAPPA and U_blk < KAPPA
                 else "SEED_SENSITIVE")
        kl = {a: float(g[g.arm == a].kl_span_tok.mean()) for a in arms + ["fp8kv", "fp_noise"]}
        out[f] = dict(units=int(n), seeds=int(m), mean_y_seed0=float(y[:, 0].mean()), seed_means=[float(v) for v in c_s],
                      var_unit=var_a, var_seed_common=var_c, var_residual=ms_r,
                      seed_common_share=var_c / max(var_a + var_c + ms_r, 1e-300), U_unit=U0, U_two=U_two,
                      U_seed_level=U_blk, U_by_seed=U_by_seed, label=label, kl_mean=kl)
    return out


def predictions(A, B, spearman=None):
    out = {}

    def cell(g, k):
        return (A.get(g) or {}).get("bits", {}).get(str(k))
    got = [cell(g, k) for g in A for k in PR.BITS if cell(g, k)]
    out["P1_coverage"] = ("NOT_READ" if not got else "MET" if all(c[b]["label"] == "COVER_OK" for c in got for b in ("p8", "v2"))
                          else "NOT_MET")
    for nm, fn in (("P2_tightness", lambda g, c: PRED["P2"][g][0] <= c["p8"]["ratio_med"] <= PRED["P2"][g][1]),
                   ("P3_certified", lambda g, c: c["p8"]["certified_fp8"] >= PRED["P3"][g]),
                   ("P5_certread", lambda g, c: "certread" in c and c["certread"]["traffic_rule"] <= PRED["P5"][g]
                    and c["certread"]["label"] == "COVER_OK"),
                   ("P6_p8_over_v2", lambda g, c: c["p8_over_v2_med"] <= PRED["P6"])):
        res = {g: bool(fn(g, cell(g, 6))) for g in A if cell(g, 6) and g in PRED["P2"]}
        out[nm] = "NOT_READ" if not res else "MET" if all(res.values()) else f"NOT_MET {sorted(g for g, v in res.items() if not v)}"
    m = [(g, k, cell(g, k)) for g in A for k in PR.BITS if cell(g, k)]
    ok = all(c["M_exact"][1] <= PRED["P4_exact"][1] and c["M_exact"][2] >= PRED["P4_exact"][0] for _, _, c in m) and \
        all(PRED["P4_first"][0] <= c["M_first"][0] <= PRED["P4_first"][1] for _, k, c in m if k >= 6)
    out["P4_theorem_m"] = "NOT_READ" if not m else "MET" if ok else "NOT_MET"
    labs = [v.get("label") for g in B for v in B[g].values() if v.get("label") in ("SEED_ROBUST", "SEED_SENSITIVE")]
    out["P7_seed_robust"] = "NOT_READ" if not labs else "MET" if all(x == "SEED_ROBUST" for x in labs) else "NOT_MET"
    out["P8_share2_orders_kl"] = ("NOT_READ" if spearman is None else "MET" if spearman > 0 else "NOT_MET")
    return out


def r6_design_kl(r6: dict, group: str) -> dict:
    """The design's mean KL per token per family in R6's reader JSON (read_stage1h_r6: groups[g]['arms'] entries with
    arm, B, family and KL_tok = [mean, lo, hi, n])."""
    out = {}
    for x in (r6.get("groups", {}).get(group, {}).get("arms") or []):
        if x.get("arm") == L7.DESIGN and abs(float(x.get("B", 0)) - 0.125) < 1e-9 and x.get("KL_tok"):
            out[x["family"]] = float(x["KL_tok"][0])
    return out


def _spearman(a, b):
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")


def read_r7a(pairs, out_stem, root=RESULTS, r6_json=None, smoke=False):
    fam = family_of()
    excluded, heads, groups, mains, codes = {}, {}, {}, {}, {}
    for tag, job in pairs:
        side, d, h, gr = load_block(tag, job, root)
        problems = []
        if side is None:
            excluded[f"{tag}:{job}"] = ["cannot load"]
            continue
        validity(tag, side, d, h, problems, smoke=smoke)
        if problems:
            excluded[f"{tag}:{job}"] = problems
            continue
        preset = side["preset_name"]
        grp = L7.R7_GROUP_OF.get(preset, "smoke")
        codes[f"{tag}:{job}"] = json.dumps((side.get("r7a") or {}).get("code_sha256"), sort_keys=True)
        unit = lambda x: f"{tag}:{job}:" + x.prompt_idx.astype(str) + ":" + x.task.astype(str)   # noqa: E731
        mains.setdefault(grp, []).append(d.assign(unit=unit(d), block=f"{tag}:{job}", kind=L7.PRESETS[preset]["r7a"]))
        if h is not None and len(h):
            heads.setdefault(grp, []).append(h.assign(unit=unit(h)))
        if gr is not None and len(gr):
            groups.setdefault(grp, []).append(gr.assign(unit=unit(gr)))
    if len(set(codes.values())) > 1:
        print("WARNING: the blocks' code manifests differ", flush=True)
    A, B = {}, {}
    for grp in sorted(set(heads) | set(mains)):
        if grp in heads:
            A[grp] = panel_a(pd.concat(heads[grp], ignore_index=True),
                             pd.concat(groups[grp], ignore_index=True) if grp in groups else None, fam=fam)
        m = pd.concat(mains[grp], ignore_index=True)
        mk = m[m.kind == "kl"]
        if len(mk):
            B[grp] = panel_b(mk, fam)
    sp = None
    if r6_json and os.path.exists(r6_json) and "llama128" in A and "families" in A["llama128"]:
        r6 = json.load(open(r6_json))
        kl = {}
        kl = r6_design_kl(r6, "llama128")
        fams = sorted(set(kl) & set(A["llama128"]["families"]))
        if len(fams) > 2:
            sp = _spearman([A["llama128"]["families"][f]["share2_mean"] for f in fams], [kl[f] for f in fams])
    out = dict(rules="read_r7a.py (frozen with the run)", amend=L7.AMEND_R7A, blocks=[f"{t}:{j}" for t, j in pairs],
               excluded=excluded, code_manifests_identical=len(set(codes.values())) <= 1, panel_a=A, panel_b=B,
               spearman_share2_vs_r6_kl_llama=sp, predictions=predictions(A, B, sp))
    os.makedirs(os.path.dirname(out_stem) or ".", exist_ok=True)
    with open(out_stem + ".json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    with open(out_stem + ".md", "w") as fh:
        fh.write(_markdown(out))
    print(_markdown(out))
    return 0


def _f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _markdown(out):
    L_ = ["# R7a — the theory at the paper's scale (reader: read_r7a.py)", ""]
    if out["excluded"]:
        L_.append("Excluded blocks: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in out["excluded"].items()))
    L_ += ["", "## Predictions (stated before the run)", "", "| prediction | verdict |", "|---|---|"]
    L_ += [f"| {k} | {v} |" for k, v in out["predictions"].items()]
    for grp, A in out["panel_a"].items():
        L_ += ["", f"## Panel A: {GROUPS.get(grp, grp)} ({A['head_steps']:,} head-steps, {A['units']} units)", "",
               "| key bits | σ (median) | part 8 / error (p10–p90) | v2 / error | part 8 / v2 | certified within FP8 (p8 / v2) | "
               "coverage p8 (misses / checks) | coverage v2 | Theorem M exact (95% CI) | Theorem M first order | tail / FP8 (pred.) | "
               "certified read: rows (traffic), oracle, met per query head |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for kb, r in A["bits"].items():
            cr = r.get("certread")
            L_.append(f"| {kb} | {_f(r['sigma_med'])} | {_f(r['p8']['ratio_med'], 2)} ({_f(r['p8']['ratio_p10'], 2)}–"
                      f"{_f(r['p8']['ratio_p90'], 2)}) | {_f(r['v2']['ratio_med'], 1)} | {_f(r['p8_over_v2_med'])} | "
                      f"{_f(r['p8']['certified_fp8'])} / {_f(r['v2']['certified_fp8'])} | {r['p8']['label']} "
                      f"({r['p8']['misses']} / {r['p8']['checks']:,}) | {r['v2']['label']} ({r['v2']['misses']}) | "
                      f"{_f(r['M_exact'][0])} ({_f(r['M_exact'][1])}–{_f(r['M_exact'][2])}) | {_f(r['M_first'][0])} | "
                      f"{_f(r['tail_over_fp8_med'])} | "
                      + (f"{_f(cr['rows_rule'])} ({_f(cr['traffic_rule'])}), {_f(cr['rows_oracle'])}, {_f(cr['met_heads'], 4)} "
                         f"({cr['label']}) |"
                         if cr else "— |"))
        if A.get("families"):
            L_ += ["", "| family | head-steps | unread share of Σw²‖v−o‖² (mean) | part 8 certified at 6 bits |", "|---|---|---|---|"]
            L_ += [f"| {f} | {v['head_steps']:,} | {_f(v['share2_mean'], 4)} | {_f(v['certified_p8_k6'])} |"
                   for f, v in A["families"].items()]
    for grp, B in out["panel_b"].items():
        L_ += ["", f"## Panel B: {GROUPS.get(grp, grp)}, the design's KL across dither seeds", "",
               "| family | units × seeds | mean y (seed 0) | common seed effect: share of variance | U (seed 0 = R6's) | "
               "U per seed | U (one seed per replicate) | U (two-level) | label |", "|---|---|---|---|---|---|---|---|---|"]
        for f, v in B.items():
            if "label" not in v:
                L_.append(f"| {f} | {v['units']} | — | — | — | — | — | — | too few units |")
                continue
            L_.append(f"| {f} | {v['units']} × {v['seeds']} | {_f(v['mean_y_seed0'], 5)} | {_f(v['seed_common_share'])} | "
                      f"{_f(v['U_unit'], 5)} | {', '.join(_f(u, 5) for u in v['U_by_seed'])} | {_f(v['U_seed_level'], 5)} | "
                      f"{_f(v['U_two'], 5)} | {v['label']} |")
    return "\n".join(L_) + "\n"


# ------------------------------------------------------------------ the gate
def read_gate(pairs, root=RESULTS, out_dir=FINDINGS):
    problems, per_unit, peaks, cover = [], {}, [], {}
    for tag, job in pairs:
        side, d, h, gr = load_block(tag, job, root)
        if side is None:
            problems.append(f"{tag}:{job}: cannot load")
            continue
        validity(tag, side, d, h, problems, pilot=True)
        if not (d[d.arm == "fp"].kl_all.fillna(0) == 0).all():
            problems.append(f"{tag}: FP's KL to itself is not 0")
        if h is not None and len(h):
            for c in [c for c in h.columns if c.startswith("cov_")]:
                kb = c.rsplit("_k", 1)[1]
                nd = h[f"ndraw_k{kb}"].astype(float)
                miss, checks = float(((1.0 - h[c]) * nd).sum()), float(nd.sum())
                cover.setdefault(tag, {})[c] = [int(round(miss)), int(round(checks))]
                if cover_label(miss, checks, PR.DELTA) != "COVER_OK":
                    problems.append(f"{tag}: {c} misses {miss:.0f} of {checks:.0f} checks (beyond delta)")
        peaks += [float(x) for x in side.get("peak_gib_dev_max") or []]
        kind = L7.PRESETS[side["preset_name"]]["r7a"]
        t = []
        for _, g in d.groupby(UKEY):
            t.append(float(g.t_prefill.iloc[0] + g.t_precompute.fillna(0).iloc[0] + (g.t_arm.fillna(0) + g.t_tf.fillna(0)).sum()))
        per_unit[kind] = float(np.mean(t)) if t else float("nan")
    if peaks and max(peaks) > PEAK_LIMIT:
        problems.append(f"peak {max(peaks):.1f} GiB > {PEAK_LIMIT}")
    proj = {}
    for row in L7.R7_LAYOUT:
        if row[3] != 131072:                                  # the 32K blocks take a quarter of the context's work
            continue
        kind = L7.PRESETS[row[2]]["r7a"]
        if kind not in per_unit or not np.isfinite(per_unit[kind]):
            problems.append(f"no pilot time for {kind} blocks")
            continue
        n_units = row[7] * len(row[6].split(","))             # per process: a split row runs its blocks in parallel
        sec = per_unit[kind]
        if row[2].startswith("r7qwen128"):                    # projected from Llama's probe pilot (48 / 32 layers)
            sec = GATE_Q128_SCALE * sec + GATE_Q128_EXTRA_S
        hh, mm, ss = (int(x) for x in row[9].split(":"))
        wall = hh + mm / 60 + ss / 3600
        proj[row[0]] = round(n_units * sec / 3600 + GATE_SETUP_H, 2)
        if proj[row[0]] > GATE_WALL_FRAC * wall:
            problems.append(f"{row[0]}: projected {proj[row[0]]:.2f} h > {GATE_WALL_FRAC} x {wall:.1f} h")
    verdict = "PASS" if not problems else "FAIL"
    out = dict(verdict=verdict, problems=problems, pilots=[f"{t}:{j}" for t, j in pairs], seconds_per_unit=per_unit,
               projected_block_h=proj, peak_gib=max(peaks) if peaks else None, coverage_min=cover)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "R7a_gate.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0 if verdict == "PASS" else 1


def _pairs(xs):
    return [tuple(x.rsplit(":", 1)) for x in xs]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--r7a", nargs="+", metavar="TAG:JOB")
    g.add_argument("--gate", nargs="+", metavar="TAG:JOB")
    ap.add_argument("--out-stem", default=os.path.join(FINDINGS, "R7a_reader"))
    ap.add_argument("--results-root", default=RESULTS)
    ap.add_argument("--r6-json", default=None)
    a = ap.parse_args()
    if a.gate:
        sys.exit(read_gate(_pairs(a.gate), a.results_root))
    sys.exit(read_r7a(_pairs(a.r7a), a.out_stem, a.results_root, a.r6_json))


if __name__ == "__main__":
    main()
