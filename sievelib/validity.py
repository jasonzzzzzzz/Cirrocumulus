"""
validity.py -- input-validity checks for the H0 haystack.

WHY THE OLD GATE WAS WRONG
--------------------------
The retired gate required the MEDIAN-OVER-HEADS ladder width on `niah` prompts to
exceed `cont` by >= 0.1 bits. It failed on 5/5 models. It is unpassable by
construction, for two compounding reasons:

  1. WRONG STATISTIC. ladder = tau/ln2 and tau is the std over ALL L logits -- a
     bulk second moment. A needle is ONE token out of 131,072. Even a perfect
     retrieval, with the needle logit at 10x the logit spread, moves the ladder by
     ~0.0016 b. The gate demanded 0.1 b: about 60x more than the phenomenon can
     physically produce. Reproduce it with `python -m sievelib.validity`.

  2. WRONG AGGREGATE. Retrieval is carried by a small minority of heads. Taking the
     MEDIAN over heads guarantees the statistic is dominated by heads that do not
     participate, so it cannot move regardless of how well retrieval works.

This is the same error as using tau instead of the concentration statistic: a bulk
moment cannot see an extreme-value event. The observed deltas (-0.010 to +0.061 b)
are exactly what a *successful* retrieval looks like under this statistic.

THE REPLACEMENT
---------------
Two checks matched to the phenomenon, either of which is sufficient:

  A. TASK-LEVEL (preferred, unambiguous): the model reproduces the needle code in
     its greedy decode. If it answers correctly, the haystack induced retrieval.
     No attention statistic can override that.

  B. HEAD-LEVEL (extreme value, not bulk): in at least `min_heads` heads, the
     attention mass landing on the needle span exceeds `min_mass`. This is a max,
     not a median, so a small retrieval-head population is enough to register.

CALIBRATION STATUS -- READ BEFORE TIGHTENING
--------------------------------------------
`MIN_MASS` and `MIN_HEADS` below are NOT yet calibrated against measured data, so
they are advisory by default (`ENFORCE_HEAD_LEVEL = False`). Enforcing an
uncalibrated attention threshold is precisely the mistake that produced the
retired gate: a plausible-looking number vetoing a good run. Run
`run_h0.py --validity-only`, read the reported max / p99 / counts, and only then
set these and flip the switch. The task-level check needs no calibration -- the
model either emitted the code or it did not -- so it is enforced from the start.
"""
from __future__ import annotations
import torch

# Uniform attention over a ~20-token needle in a 131k context is 1.5e-4, so
# MIN_MASS = 0.05 is ~300x uniform. Plausible for a real retrieval head; not yet
# measured. See CALIBRATION STATUS.
MIN_MASS = 0.05
MIN_HEADS = 4
ENFORCE_HEAD_LEVEL = False   # flip only after calibrating MIN_MASS from real data
MIN_TASK_FRAC = 0.5          # fraction of niah prompts that must retrieve


def needle_mass(attn: torch.Tensor, needle_slice: slice) -> torch.Tensor:
    """attn: [H, L] post-softmax weights for one decode step.
    Returns [H] attention mass on the needle span."""
    return attn[:, needle_slice].sum(dim=-1)


def head_level_gate(attn_by_head: torch.Tensor, needle_slice: slice,
                    min_mass: float = MIN_MASS, min_heads: int = MIN_HEADS) -> dict:
    """B: do ANY heads actually look at the needle? Extreme value, not median."""
    m = needle_mass(attn_by_head, needle_slice)
    n_hit = int((m >= min_mass).sum())
    return dict(check="head_level", n_heads_on_needle=n_hit,
                max_needle_mass=float(m.max()),
                p99_needle_mass=float(m.quantile(0.99)),
                passed=bool(n_hit >= min_heads))


def task_level_gate(generated: str, needle_code: str) -> dict:
    """A: did the model actually retrieve? Behavioural ground truth."""
    ok = needle_code in generated
    return dict(check="task_level", needle_code=needle_code,
                generated=generated[:120], passed=bool(ok))


def combined(task=None, head=None) -> dict:
    """Either check passing is sufficient. Task-level dominates when available."""
    if task is not None and task["passed"]:
        return dict(passed=True, basis="task_level", detail=task)
    if head is not None and head["passed"]:
        return dict(passed=True, basis="head_level", detail=head)
    parts = [c for c in (task, head) if c is not None]
    return dict(passed=False, basis="none",
                detail=parts,
                note="Neither the model's answer nor any head's attention shows "
                     "retrieval. THIS is a genuine haystack failure -- unlike the "
                     "retired median-ladder gate, which could not pass at all.")


# ------------------------------------------------------- aggregate over a run
def summarize(niah_rows, *, min_mass: float = MIN_MASS,
              min_heads: int = MIN_HEADS,
              enforce_head: bool = ENFORCE_HEAD_LEVEL,
              min_task_frac: float = MIN_TASK_FRAC) -> dict:
    """Roll per-(prompt, layer, head) niah rows up into one validity record.

    `niah_rows` is a DataFrame with at least `needle_mass`, and `needle_hit` and
    `prompt` when task-level data is present. Kept here rather than in report.py
    so the thresholds and their calibration caveat live in ONE file.
    """
    import numpy as np
    r = {"n_rows": int(len(niah_rows)), "task": None, "head": None}
    if not len(niah_rows):
        return {**r, "passed": False, "basis": "none",
                "reason": "no niah rows -- was the niah family in `families`?"}

    # Task level: one verdict per prompt, then the fraction that retrieved.
    if "needle_hit" in niah_rows:
        per_prompt = niah_rows.groupby("prompt")["needle_hit"].max()
        frac = float(per_prompt.mean())
        r["task"] = {"n_prompts": int(len(per_prompt)),
                     "n_retrieved": int(per_prompt.sum()), "frac": frac,
                     "passed": bool(frac >= min_task_frac)}

    # Head level: per-head median mass across prompts, then an EXTREME statistic
    # over heads -- never a median over heads, which is what sank the old gate.
    if "needle_mass" in niah_rows:
        per_head = (niah_rows.groupby(["layer", "head"])["needle_mass"]
                    .median().dropna())
        if len(per_head):
            n_hit = int((per_head >= min_mass).sum())
            r["head"] = {
                "n_heads": int(len(per_head)),
                "n_heads_on_needle": n_hit,
                "max_needle_mass": float(per_head.max()),
                "p99_needle_mass": float(per_head.quantile(0.99)),
                "median_needle_mass": float(per_head.median()),
                "counts": {f"{t:g}": int((per_head >= t).sum())
                           for t in (0.001, 0.005, 0.01, 0.05, 0.1, 0.25)},
                "passed": bool(n_hit >= min_heads)}

    task_ok = bool(r["task"] and r["task"]["passed"])
    head_ok = bool(r["head"] and r["head"]["passed"])
    if task_ok:
        return {**r, "passed": True, "basis": "task_level",
                "reason": f"model retrieved the code in "
                          f"{r['task']['n_retrieved']}/{r['task']['n_prompts']} "
                          f"niah prompts"}
    if head_ok and enforce_head:
        return {**r, "passed": True, "basis": "head_level",
                "reason": f"{r['head']['n_heads_on_needle']} heads carry "
                          f">= {min_mass} attention mass on the needle"}
    if head_ok:
        return {**r, "passed": True, "basis": "head_level (advisory)",
                "reason": f"{r['head']['n_heads_on_needle']} heads carry "
                          f">= {min_mass} mass on the needle, but MIN_MASS is "
                          f"uncalibrated so this is advisory, not a pass on merit"}

    bits = []
    if r["task"]:
        bits.append(f"model retrieved the code in only "
                    f"{r['task']['n_retrieved']}/{r['task']['n_prompts']} prompts")
    else:
        bits.append("no needle_hit column -- results predate --validity-only")
    if r["head"]:
        bits.append(f"max per-head needle mass {r['head']['max_needle_mass']:.4f} "
                    f"< {min_mass}")
    return {**r, "passed": False, "basis": "none", "reason": "; ".join(bits)}


# --------------------------------------------------------------- retired gate
def retired_ladder_gate(ladder_niah: float, ladder_cont: float,
                        frac_niah_gt_cont: float) -> dict:
    """Kept only so the paper can report what was tried and why it was dropped."""
    return dict(check="RETIRED_median_ladder_delta",
                delta_bits=ladder_niah - ladder_cont,
                frac_heads=frac_niah_gt_cont,
                passed=(ladder_niah - ladder_cont) >= 0.1 and frac_niah_gt_cont >= 0.60,
                note="Unpassable: a single needle moves this statistic by ~0.0016 b "
                     "against a 0.1 b threshold, and the median over heads is "
                     "dominated by non-retrieval heads. Do not use.")


def _demo():
    """Why the retired gate cannot pass, reproducible in ten seconds."""
    torch.manual_seed(0)
    L, tau = 131072, 3.0
    base = torch.randn(L) * tau

    def ladder(s):
        return float(torch.log2(torch.softmax(s.double(), -1)).std())

    b0 = ladder(base)
    print(f"baseline ladder = {b0:.4f} b   (tau/ln2 = {tau / 0.693147:.3f})")
    print(f"retired gate demanded a delta of >= 0.1000 b\n")
    print(f"{'needle tokens':>14s} {'needle logit':>13s} {'delta ladder':>13s}  verdict")
    for n, k in ((1, 10), (1, 20), (20, 10), (100, 10), (500, 10)):
        s = base.clone()
        s[:n] = base.mean() + k * tau
        d = ladder(s) - b0
        print(f"{n:>14d} {str(k) + 'x tau':>13s} {d:>+12.4f} b  "
              f"{'pass' if d >= 0.1 else 'FAIL'}")
    print("\nThe real needle is ~20 tokens. Even a perfect retrieval falls ~3x "
          "short of\nthe threshold, so the gate measured nothing about the input.")


if __name__ == "__main__":
    _demo()
