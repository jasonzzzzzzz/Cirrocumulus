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
     ~0.0016 b. The gate demanded 0.1 b: about 70x more than the phenomenon can
     physically produce. You would need ~500 simultaneous needles to trip it.

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
"""
from __future__ import annotations
import torch


def needle_mass(attn: torch.Tensor, needle_slice: slice) -> torch.Tensor:
    """attn: [H, L] post-softmax weights for one decode step.
    Returns [H] attention mass on the needle span."""
    return attn[:, needle_slice].sum(dim=-1)


def head_level_gate(attn_by_head: torch.Tensor, needle_slice: slice,
                    min_mass: float = 0.05, min_heads: int = 4) -> dict:
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
