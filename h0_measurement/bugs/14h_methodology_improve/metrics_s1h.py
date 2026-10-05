"""metrics_s1h.py -- R14 Stage 1h metrics and statistics (pure functions).

No model, no runner chain: the driver (run_s1h.py) and the reader (read_stage1h.py)
both import this module, and later stages can import it directly.

  kl_columns      KL(FP || arm) of the next-token distributions under teacher
                  forcing on FP's answer, per token, summed over the same positions
                  as the NLL columns (all tokens; the A2 span from FP's first
                  answer-value token to the span end; the answer-value tokens).
  lost_mask       units on which an arm scores below FP ("below FP" of Stages 1f-1g).
  mcnemar_exact   two-sided exact McNemar test on two arms' lost answers.
  equivalent      TOST at alpha = 0.05 each side, as a 90% CI inside +-margin.
  holm            Holm-Bonferroni adjusted p-values.
  sequential_step the sequential prompt-block rule (plan.md section 7).
  failure_type    a lost answer's failure type, by rule.
"""
from __future__ import annotations
import math

import numpy as np

# --------------------------------------------------------------------- KL
def kl_columns(lg, ref_logp, vmask, span_end, digits: int = 6) -> dict:
    """lg: the arm's teacher-forced logits [T, V] on FP's answer; ref_logp: FP's
    log-probabilities on the same positions [T, V]; vmask / span_end as the NLL
    columns (s1d_lib.answer_tokens). Position t scores the prediction of token t.
    Returns kl_all, kl_mean, kl_span, kl_val, kl_span_max and the per-token tf_kl
    (nan where there is no answer value)."""
    import torch
    T = int(min(lg.shape[0], ref_logp.shape[0]))
    nan = float("nan")
    if T == 0:
        return dict(kl_all=nan, kl_mean=nan, kl_span=nan, kl_val=nan, kl_span_max=nan, tf_kl=[])
    lp = torch.log_softmax(lg[:T].float(), -1)
    rp = ref_logp[:T].float().to(lp.device)
    p = rp.exp()
    terms = torch.where(p > 0, p * (rp - lp), torch.zeros_like(p))
    kl = terms.sum(-1).clamp_min(0.0).double()                         # [T]
    out = dict(kl_all=float(kl.sum()), kl_mean=float(kl.mean()),
               tf_kl=[round(float(x), digits) for x in kl.tolist()])
    v = np.zeros(T, dtype=bool)
    m = [str(x) in ("1", "True") for x in vmask][:T]
    v[:len(m)] = m
    if not v.any():
        out.update(kl_span=nan, kl_val=nan, kl_span_max=nan)
        return out
    i0, se = int(np.argmax(v)), int(min(int(span_end), T))
    kk = kl.cpu().numpy()
    span = kk[i0:se] if se > i0 else kk[i0:i0 + 1]
    out.update(kl_span=float(span.sum()), kl_val=float(kk[v].sum()), kl_span_max=float(span.max()))
    return out


# ------------------------------------------------------------- lost answers
def lost_mask(fp_score, arm_score, eps: float = 1e-9) -> np.ndarray:
    """Units on which the arm scores below FP."""
    return np.asarray(arm_score, dtype=float) < np.asarray(fp_score, dtype=float) - eps


def mcnemar_exact(lost_a, lost_b) -> dict:
    """Two-sided exact McNemar test on paired binary outcomes: b = units A loses
    and B does not, c = the reverse. p = min(1, 2 * P(X <= min(b, c))), X ~
    Binomial(b + c, 1/2)."""
    a = np.asarray(lost_a, dtype=bool)
    bb = np.asarray(lost_b, dtype=bool)
    b, c = int((a & ~bb).sum()), int((~a & bb).sum())
    n = b + c
    if n == 0:
        return dict(b=b, c=c, p=1.0)
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return dict(b=b, c=c, p=float(min(1.0, 2.0 * tail)))


# ---------------------------------------------------------------- equivalence
def equivalent(lo: float, hi: float, margin: float) -> bool:
    """TOST at alpha = 0.05 each side: the 90% CI lies inside [-margin, +margin]."""
    return float(lo) >= -float(margin) and float(hi) <= float(margin)


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values, same keys."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * float(p)))
        out[k] = run
    return out


# ------------------------------------------------------------ sequential rule
def sequential_step(lo: float, hi: float, margin: float, n_prompts: int, cap: int = 60) -> str:
    """plan.md section 7, for a non-inferiority comparison: PASS if the 90% CI's
    upper bound is <= margin, FAIL if its lower bound is > margin, CAP at the
    prompt cap, otherwise ADD (one more 10-prompt block)."""
    if float(hi) <= float(margin):
        return "PASS"
    if float(lo) > float(margin):
        return "FAIL"
    if int(n_prompts) >= int(cap):
        return "CAP"
    return "ADD"


# ------------------------------------------------------------- failure types
def failure_type(pred: str, expected, distractors=(), min_prefix: int = 3) -> str:
    """A lost answer's type, by rule:
      'confused'    states a distractor's value (another needle's) -- key confusion
                    or a wrong needle;
      'incomplete'  states some but not all expected values, or a proper prefix (at
                    least min_prefix characters) of a missing one -- truncation;
      'other'       anything else;
      'none'        every expected value is stated (not a lost answer by content)."""
    pred = str(pred)
    exp = [str(x) for x in expected if str(x)]
    dis = [str(x) for x in (distractors or []) if str(x) and str(x) not in exp]
    have = [x for x in exp if x in pred]
    if exp and len(have) == len(exp):
        return "none"
    if any(x in pred for x in dis):
        return "confused"
    if have:
        return "incomplete"
    for x in exp:
        for k in range(len(x) - 1, min_prefix - 1, -1):
            if x[:k] in pred:
                return "incomplete"
    return "other"
