"""Pure helpers for the R8 whole-policy diagnostics.

The policy-logit diagnostic compares *complete* cache policies along one
shared, teacher-forced FP trajectory.  This module deliberately knows nothing
about allocations, routes, tasks, or result files.  Logits enter as transient
tensors and the public metric function returns scalar summaries only.

All divergence arithmetic is performed over the complete vocabulary in
``float32``.  ``teacher_forced_logits`` obtains every candidate distribution by
feeding the supplied FP choices; a candidate's own argmax is never fed back to
the model.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


TRACE_RULE_VERSION = "fp_teacher_forced_v1"
METRIC_KEYS = (
    "mean_kl",
    "max_kl",
    "fp_token_cross_entropy",
    "top1_agreement",
    "trace_length",
)


def _token_list(tokens: Sequence[int] | torch.Tensor) -> list[int]:
    """Return a one-dimensional token sequence as ordinary Python integers."""
    if isinstance(tokens, torch.Tensor):
        if tokens.ndim != 1:
            raise ValueError(f"tokens must be one-dimensional, got {tuple(tokens.shape)}")
        return [int(x) for x in tokens.detach().cpu().tolist()]
    try:
        out = [int(x) for x in tokens]
    except (TypeError, ValueError) as exc:
        raise ValueError("tokens must be a one-dimensional integer sequence") from exc
    # Refuse silent truncation such as 1.5 -> 1.  bool is also not a token ID.
    for raw, value in zip(tokens, out):
        if isinstance(raw, bool) or raw != value:
            raise ValueError(f"invalid token ID {raw!r}")
    return out


def token_hash(tokens: Sequence[int] | torch.Tensor) -> str:
    """SHA-256 of a canonical token-ID sequence.

    JSON makes the digest independent of tensor dtype, device, and host byte
    order.  The full digest is cheap and avoids turning the hash into an
    accidental raw-token surrogate in diagnostic artifacts.
    """
    payload = json.dumps(_token_list(tokens), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def teacher_forced_logits(model: Any, past: Any, first_token: int | torch.Tensor,
                          fp_tokens: Sequence[int] | torch.Tensor
                          ) -> tuple[torch.Tensor, Any]:
    """Replay a candidate on the FP greedy trajectory.

    ``fp_tokens`` contains the FP choices whose distributions are being
    compared.  For a trace of length ``T``, the model is called ``T`` times:
    ``first_token`` produces the distribution for ``fp_tokens[0]`` and then
    ``fp_tokens[0:T-1]`` are fed to produce the remaining distributions.  The
    last FP token need not be fed because no later distribution is requested.

    The returned tensor is ``[T, vocab]`` and float32.  It is intentionally not
    moved to CPU or serialized; callers compute summaries and discard it.  The
    returned cache is the cache produced by the last step.
    """
    chosen = _token_list(fp_tokens)
    if not chosen:
        raise ValueError("a teacher-forced trace needs at least one FP token")

    if isinstance(first_token, torch.Tensor):
        if first_token.numel() != 1:
            raise ValueError("first_token must contain exactly one token ID")
        if first_token.dtype == torch.bool:
            raise ValueError("first_token must be an integer token ID")
        device = first_token.device
        dtype = first_token.dtype
        cur = first_token.detach().reshape(1, 1)
    else:
        if isinstance(first_token, bool) or int(first_token) != first_token:
            raise ValueError(f"invalid first token ID {first_token!r}")
        # A model normally receives a tensor first token from run_r8.  This
        # fallback keeps the helper useful for small CPU tests and toy models.
        device, dtype = torch.device("cpu"), torch.long
        cur = torch.tensor([[int(first_token)]], device=device, dtype=dtype)

    rows: list[torch.Tensor] = []
    with torch.no_grad():
        for step in range(len(chosen)):
            out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                shape = getattr(logits, "shape", None)
                raise ValueError(f"model logits must have shape [batch, seq, vocab], got {shape}")
            if logits.shape[0] != 1 or logits.shape[1] < 1 or logits.shape[2] < 1:
                raise ValueError(f"model logits must have shape [1, seq>=1, vocab>=1], got "
                                 f"{tuple(logits.shape)}")
            rows.append(logits[0, -1].detach().to(dtype=torch.float32))
            if step + 1 < len(chosen):
                # This is the defining invariant: use the supplied FP choice,
                # never logits.argmax() from the candidate just evaluated.
                cur = torch.tensor([[chosen[step]]], device=device, dtype=dtype)
    return torch.stack(rows, dim=0), past


def divergence_metrics(ref_logits: torch.Tensor, cand_logits: torch.Tensor,
                       fp_tokens: Sequence[int] | torch.Tensor) -> dict[str, float | int]:
    """Summarize one candidate against FP on a shared trajectory.

    KL is ``KL(softmax(FP) || softmax(candidate))`` at each step, reduced over
    the entire vocabulary in float32.  Cross entropy is evaluated on the token
    FP actually chose at that step; it is not answer-token or gold-label NLL.
    ``top1_agreement`` compares the two distributions' deterministic argmaxes.
    """
    if not isinstance(ref_logits, torch.Tensor) or not isinstance(cand_logits, torch.Tensor):
        raise TypeError("ref_logits and cand_logits must be torch tensors")
    if ref_logits.ndim != 2 or cand_logits.ndim != 2:
        raise ValueError("logits must each have shape [trace, vocab]")
    if ref_logits.shape != cand_logits.shape:
        raise ValueError(f"logit shapes differ: {tuple(ref_logits.shape)} vs "
                         f"{tuple(cand_logits.shape)}")
    trace, vocab = ref_logits.shape
    if trace < 1 or vocab < 1:
        raise ValueError("logits need a nonempty trace and vocabulary")
    chosen = _token_list(fp_tokens)
    if len(chosen) != trace:
        raise ValueError(f"got {len(chosen)} FP tokens for a {trace}-step trace")
    if any(x < 0 or x >= vocab for x in chosen):
        raise ValueError(f"FP token IDs must lie in [0, {vocab})")
    if ref_logits.device != cand_logits.device:
        raise ValueError("reference and candidate logits must be on the same device")

    ref = ref_logits.detach().to(dtype=torch.float32)
    cand = cand_logits.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(ref).all()) or not bool(torch.isfinite(cand).all()):
        raise ValueError("diagnostic logits must be finite")

    ref_logp = torch.log_softmax(ref, dim=-1, dtype=torch.float32)
    cand_logp = torch.log_softmax(cand, dim=-1, dtype=torch.float32)
    # Keep the multiplication and both reductions in float32.  Writing the
    # formula directly also guarantees bit-exact zero for FP compared with
    # itself (the log-probability difference is exactly zero).
    step_kl = torch.sum(ref_logp.exp() * (ref_logp - cand_logp), dim=-1,
                        dtype=torch.float32)
    ids = torch.tensor(chosen, device=cand.device, dtype=torch.long).view(-1, 1)
    fp_ce = -cand_logp.gather(1, ids).squeeze(1)
    agreement = (ref.argmax(dim=-1) == cand.argmax(dim=-1)).to(torch.float32)

    return {
        "mean_kl": float(step_kl.mean(dtype=torch.float32).item()),
        "max_kl": float(step_kl.max().item()),
        "fp_token_cross_entropy": float(fp_ce.mean(dtype=torch.float32).item()),
        "top1_agreement": float(agreement.mean(dtype=torch.float32).item()),
        "trace_length": int(trace),
    }


def _ordered_values(values: Mapping[str, Any], order: Sequence[str]) -> tuple[str, ...]:
    names = tuple(order)
    if not names:
        raise ValueError("candidate order must not be empty")
    if len(set(names)) != len(names):
        raise ValueError(f"candidate order contains duplicates: {names!r}")
    if set(values) != set(names):
        missing = sorted(set(names) - set(values))
        extra = sorted(set(values) - set(names))
        raise ValueError(f"candidate mapping/order mismatch: missing={missing}, extra={extra}")
    return names


def _mean_kl(value: Any) -> float:
    if isinstance(value, Mapping):
        if "mean_kl" not in value:
            raise ValueError("candidate metric mapping has no 'mean_kl'")
        value = value["mean_kl"]
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"mean KL must be finite, got {value!r}")
    return out


def select_min_mean(values: Mapping[str, Any], order: Sequence[str]) -> str:
    """Select the least mean-KL policy; exact ties use declared order.

    Values may be the dictionaries returned by :func:`divergence_metrics` or
    plain mean-KL scalars.  No tolerance is used: a numerically smaller value
    wins, while an exactly equal value leaves the earlier policy selected.
    """
    names = _ordered_values(values, order)
    best = names[0]
    best_value = _mean_kl(values[best])
    for name in names[1:]:
        value = _mean_kl(values[name])
        if value < best_value:
            best, best_value = name, value
    return best


def end_task_ties(scores: Mapping[str, float], order: Sequence[str]) -> tuple[str, ...]:
    """Return every exact end-task-score maximizer in declared order."""
    names = _ordered_values(scores, order)
    vals: dict[str, float] = {}
    for name in names:
        value = float(scores[name])
        if not math.isfinite(value):
            raise ValueError(f"end-task score must be finite, got {scores[name]!r}")
        vals[name] = value
    maximum = max(vals.values())
    return tuple(name for name in names if vals[name] == maximum)

