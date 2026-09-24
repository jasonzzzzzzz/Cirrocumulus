"""Structured-query attention capture and allocations for the V7 design.

This module is an opt-in wrapper around :mod:`sievelib.compress`.  The wrapper
always delegates the actual attention and compression behavior to
``compress.sieve_compress_attention``.  During a full-precision prompt prefill
it can additionally score context keys from five frozen groups of absolute
query positions.  Every group contributes one fifth of the score and every
query position within a group has equal weight.

The capture is independent of chunk boundaries.  Absolute query position zero
is the first token in the actually prefilled prompt.  A query can only score
keys at or before its own position; scores for later context keys remain zero.
No task labels or answer values enter this interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import torch

from . import compress, router


IMPL = "sieve_structured"
CAPTURE_VERSION = "structured_absolute_query_capture_v1"
QUERY_GROUP_COUNT = 5


@dataclass(frozen=True)
class AbsoluteQueryPlan:
    """Validated absolute prompt queries and their frozen global weights."""

    groups: tuple[tuple[int, ...], ...]
    weights: tuple[tuple[float, ...], ...]
    prompt_length: int

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(index for group in self.groups for index in group)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def validate_absolute_query_groups(
    query_groups: Sequence[Sequence[int]],
    *,
    prompt_length: int,
    query_weights: Sequence[Sequence[float]] | None = None,
) -> AbsoluteQueryPlan:
    """Validate five nonempty, disjoint groups of absolute prompt indices.

    The only accepted weight rule is the preregistered one: a position in group
    ``g`` has weight ``1 / (5 * len(group_g))``.  Callers may omit weights and
    receive those values, or provide them for strict provenance validation.
    """
    length = _positive_int(prompt_length, "prompt_length")
    if isinstance(query_groups, (str, bytes)) or not isinstance(
        query_groups, Sequence
    ):
        raise ValueError("query_groups must be a sequence of five groups")
    if len(query_groups) != QUERY_GROUP_COUNT:
        raise ValueError(
            f"query_groups must contain exactly {QUERY_GROUP_COUNT} groups"
        )

    groups: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for group_index, raw_group in enumerate(query_groups):
        if isinstance(raw_group, (str, bytes)) or not isinstance(
            raw_group, Sequence
        ):
            raise ValueError(f"query group {group_index} must be a sequence")
        if not raw_group:
            raise ValueError(f"query group {group_index} must be nonempty")
        group: list[int] = []
        for raw_index in raw_group:
            if isinstance(raw_index, bool) or not isinstance(raw_index, Integral):
                raise ValueError(
                    f"query group {group_index} contains a non-integer index"
                )
            index = int(raw_index)
            if not 0 <= index < length:
                raise ValueError(
                    f"absolute query index {index} lies outside [0, {length})"
                )
            group.append(index)
        if any(right <= left for left, right in zip(group, group[1:])):
            raise ValueError(
                f"query group {group_index} must be strictly increasing"
            )
        overlap = seen.intersection(group)
        if overlap:
            raise ValueError(
                f"absolute query indices occur in multiple groups: {sorted(overlap)}"
            )
        seen.update(group)
        groups.append(tuple(group))

    expected_weights = tuple(
        tuple(1.0 / (QUERY_GROUP_COUNT * len(group)) for _ in group)
        for group in groups
    )
    if query_weights is None:
        weights = expected_weights
    else:
        if isinstance(query_weights, (str, bytes)) or not isinstance(
            query_weights, Sequence
        ) or len(query_weights) != QUERY_GROUP_COUNT:
            raise ValueError("query_weights must contain exactly five groups")
        validated: list[tuple[float, ...]] = []
        for group_index, (raw_weights, expected) in enumerate(
            zip(query_weights, expected_weights)
        ):
            if isinstance(raw_weights, (str, bytes)) or not isinstance(
                raw_weights, Sequence
            ) or len(raw_weights) != len(expected):
                raise ValueError(
                    f"query weight group {group_index} has the wrong length"
                )
            group_weights: list[float] = []
            for raw_weight, expected_weight in zip(raw_weights, expected):
                if isinstance(raw_weight, bool) or not isinstance(raw_weight, Real):
                    raise ValueError("query weights must be real numbers")
                weight = float(raw_weight)
                if not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError("query weights must be finite and positive")
                if not math.isclose(
                    weight, expected_weight, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(
                        "query weights must give each group mass 1/5 and each "
                        "position equal mass within its group"
                    )
                group_weights.append(weight)
            validated.append(tuple(group_weights))
        weights = tuple(validated)

    if not math.isclose(
        sum(weight for group in weights for weight in group),
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("validated query weights do not sum to one")
    return AbsoluteQueryPlan(tuple(groups), weights, length)


def aggregate_equal_group_attention(group_scores: torch.Tensor) -> torch.Tensor:
    """Return the direct weighted per-query-head sum from ``[5, H, C]``.

    ``group_scores`` already contains weights ``1 / (5 * group_size)``.  The
    five tensors are therefore summed directly.  There is deliberately no
    per-group context renormalization: doing that would change the frozen query
    weights according to how much attention happened to remain inside the
    compressible context.  The interior allocation performs one row
    normalization after this sum; eviction ranks the raw weighted score.
    """
    if not isinstance(group_scores, torch.Tensor) or group_scores.ndim != 3:
        raise ValueError("group_scores must be a [5, H, C] tensor")
    if group_scores.shape[0] != QUERY_GROUP_COUNT:
        raise ValueError("group_scores must contain exactly five query groups")
    if group_scores.shape[1] <= 0 or group_scores.shape[2] <= 0:
        raise ValueError("group_scores must have positive head and context axes")
    values = group_scores.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("group_scores must be finite")
    if bool((values < 0).any()):
        raise ValueError("group_scores must be nonnegative")
    output = values.sum(dim=0)
    output_mass = output.sum(dim=-1)
    if not bool(torch.isfinite(output_mass).all()) or bool((output_mass <= 0).any()):
        raise ValueError("every query head must put positive weighted mass on context")
    return output


class StructuredCaptureState:
    """Prompt-local state for the opt-in structured attention wrapper."""

    def __init__(self) -> None:
        self.reset_prompt()

    def reset_prompt(self) -> None:
        self.capture = False
        self.plan: AbsoluteQueryPlan | None = None
        self.ctx_len = 0
        self.seen: dict[int, set[int]] = {}
        self.n_rep: dict[int, int] = {}
        # Each selected query is multiplied by its frozen global weight before
        # it is added here.  Persisting only this direct [H, C] sum avoids the
        # fivefold prompt-memory cost of retaining one tensor per query group.
        self.score_h: dict[int, torch.Tensor] = {}
        self.score: dict[int, torch.Tensor] = {}

    def begin(
        self,
        query_groups: Sequence[Sequence[int]],
        *,
        prompt_length: int,
        ctx_len: int,
        query_weights: Sequence[Sequence[float]] | None = None,
    ) -> AbsoluteQueryPlan:
        context = _positive_int(ctx_len, "ctx_len")
        plan = validate_absolute_query_groups(
            query_groups,
            prompt_length=prompt_length,
            query_weights=query_weights,
        )
        if context > plan.prompt_length:
            raise ValueError("ctx_len cannot exceed prompt_length")
        self.reset_prompt()
        self.plan = plan
        self.ctx_len = context
        self.capture = True
        return plan

    def expected_indices(self) -> frozenset[int]:
        if self.plan is None:
            return frozenset()
        return frozenset(self.plan.indices)

    def intersects(self, base: int, q_len: int) -> bool:
        end = int(base) + int(q_len)
        return any(base <= index < end for index in self.expected_indices())

    def _finalize_layer(self, layer: int) -> None:
        expected = self.expected_indices()
        observed = self.seen.get(int(layer), set())
        if observed != set(expected):
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise RuntimeError(
                f"layer {layer} structured capture is incomplete: "
                f"missing={missing}, extra={extra}"
            )
        score_h = self.score_h.get(int(layer))
        if score_h is None:
            raise RuntimeError(f"layer {layer} has no structured query scores")
        if (
            score_h.ndim != 2
            or score_h.shape[1] != self.ctx_len
            or not bool(torch.isfinite(score_h).all())
            or bool((score_h < 0).any())
            or bool((score_h.sum(dim=-1) <= 0).any())
        ):
            raise RuntimeError(f"layer {layer} has invalid structured query scores")
        repeats = self.n_rep[int(layer)]
        heads, context = score_h.shape
        if heads % repeats:
            raise RuntimeError(
                f"layer {layer} has {heads} query heads not divisible by n_rep={repeats}"
            )
        score_kv = score_h.reshape(heads // repeats, repeats, context).sum(dim=1)
        self.score[int(layer)] = score_kv

    def layer_scores(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._finalize_layer(int(layer))
        return self.score_h[int(layer)], self.score[int(layer)]

    def end(
        self, expected_layers: Sequence[int] | None = None
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        if self.plan is None:
            raise RuntimeError("structured capture was not configured")
        layers = (
            tuple(sorted(self.score_h))
            if expected_layers is None
            else tuple(int(layer) for layer in expected_layers)
        )
        if not layers or len(layers) != len(set(layers)):
            raise RuntimeError("expected_layers must contain unique captured layers")
        for layer in layers:
            self._finalize_layer(layer)
        unexpected = set(self.score_h) - set(layers)
        if expected_layers is not None and unexpected:
            raise RuntimeError(
                f"structured capture contains unexpected layers: {sorted(unexpected)}"
            )
        self.capture = False
        return (
            {layer: self.score_h[layer] for layer in layers},
            {layer: self.score[layer] for layer in layers},
        )


STATE = StructuredCaptureState()


def begin_capture(
    query_groups: Sequence[Sequence[int]],
    *,
    prompt_length: int,
    ctx_len: int,
    query_weights: Sequence[Sequence[float]] | None = None,
) -> AbsoluteQueryPlan:
    """Configure the module singleton for one prompt."""
    return STATE.begin(
        query_groups,
        prompt_length=prompt_length,
        ctx_len=ctx_len,
        query_weights=query_weights,
    )


def end_capture(
    expected_layers: Sequence[int] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Validate completeness, stop capture, and return query/KV-head scores."""
    return STATE.end(expected_layers)


def _selected_attention_mask(
    attention_mask: torch.Tensor,
    *,
    local_indices: Sequence[int],
    q_len: int,
    k_len: int,
    n_heads: int,
) -> torch.Tensor:
    mask = attention_mask.detach()
    if mask.ndim == 2:  # [batch, keys]
        mask = mask[:, None, None, :]
    elif mask.ndim == 3:  # [batch, queries, keys]
        mask = mask[:, None, :, :]
    elif mask.ndim != 4:
        raise ValueError("attention_mask must have 2, 3, or 4 dimensions")
    if mask.shape[0] != 1 or mask.shape[-1] < k_len:
        raise ValueError("attention_mask batch/key shape is incompatible with capture")
    mask = mask[..., :k_len]
    query_axis = mask.shape[-2]
    if query_axis == 1:
        mask = mask.expand(*mask.shape[:-2], len(local_indices), k_len)
    elif query_axis >= q_len:
        offset = query_axis - q_len
        rows = torch.tensor(
            [offset + int(index) for index in local_indices],
            dtype=torch.long,
            device=mask.device,
        )
        mask = mask.index_select(-2, rows)
    else:
        raise ValueError("attention_mask query axis is incompatible with capture")
    mask_heads = mask.shape[1]
    if mask_heads == 1:
        mask = mask.expand(1, n_heads, len(local_indices), k_len)
    elif mask_heads != n_heads:
        raise ValueError("attention_mask head axis is incompatible with capture")
    return mask[0]


def _capture_weighted_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None,
) -> None:
    if not STATE.capture:
        return
    plan = STATE.plan
    if plan is None:
        raise RuntimeError("structured capture is enabled without a query plan")
    if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor):
        raise ValueError("query and key must be tensors")
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must have shape [1, heads, tokens, dim]")
    if query.shape[0] != 1 or key.shape[0] != 1:
        raise ValueError("structured capture supports batch size one")
    if query.shape[-1] != key.shape[-1] or query.shape[2] <= 0:
        raise ValueError("query/key dimensions are incompatible")
    q_len, k_len = int(query.shape[2]), int(key.shape[2])
    base = k_len - q_len
    if base < 0:
        raise ValueError("key length cannot be shorter than query length")
    selected = [
        index for index in plan.indices if base <= index < base + q_len
    ]
    if not selected:
        return
    if compress.STATE.enabled:
        raise RuntimeError(
            "structured prompt capture must run before compression is enabled"
        )
    layer = compress._layer(module)
    observed = STATE.seen.setdefault(layer, set())
    duplicates = observed.intersection(selected)
    if duplicates:
        raise RuntimeError(
            f"layer {layer} captured absolute queries more than once: "
            f"{sorted(duplicates)}"
        )

    n_heads = int(query.shape[1])
    n_kv_heads = int(key.shape[1])
    if n_heads <= 0 or n_kv_heads <= 0 or n_heads % n_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    repeats = n_heads // n_kv_heads
    previous_repeats = STATE.n_rep.setdefault(layer, repeats)
    if previous_repeats != repeats:
        raise RuntimeError(f"layer {layer} changed its GQA repetition factor")
    scale = module.head_dim ** -0.5 if scaling is None else float(scaling)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("attention scaling must be finite and positive")

    local_indices = [index - base for index in selected]
    local_tensor = torch.tensor(
        local_indices, dtype=torch.long, device=query.device
    )
    q_selected = query[0].index_select(1, local_tensor).detach().float()
    keys = key[0].detach().float()
    n_selected = len(selected)
    logits = torch.einsum(
        "grnd,gkd->grnk",
        q_selected.reshape(n_kv_heads, repeats, n_selected, query.shape[-1]),
        keys,
    ).reshape(n_heads, n_selected, k_len)
    logits.mul_(scale)
    positions = torch.tensor(selected, dtype=torch.long, device=logits.device)
    key_positions = torch.arange(k_len, device=logits.device)
    logits.masked_fill_(
        key_positions.view(1, 1, k_len) > positions.view(1, n_selected, 1),
        float("-inf"),
    )
    if attention_mask is not None:
        selected_mask = _selected_attention_mask(
            attention_mask,
            local_indices=local_indices,
            q_len=q_len,
            k_len=k_len,
            n_heads=n_heads,
        ).to(logits.device)
        if selected_mask.dtype == torch.bool:
            logits.masked_fill_(~selected_mask, float("-inf"))
        else:
            additive = selected_mask.float()
            if not bool(torch.isfinite(additive).all()):
                allowed = torch.isfinite(additive) | torch.isneginf(additive)
                if not bool(allowed.all()):
                    raise ValueError("attention_mask contains invalid values")
            logits.add_(additive)
    attention = torch.softmax(logits, dim=-1, dtype=torch.float32)
    if not bool(torch.isfinite(attention).all()):
        raise ValueError("selected query attention is not finite")

    weighted_score = STATE.score_h.get(layer)
    if weighted_score is None:
        weighted_score = torch.zeros(
            n_heads,
            STATE.ctx_len,
            dtype=torch.float32,
            device=attention.device,
        )
        STATE.score_h[layer] = weighted_score
    elif (
        tuple(weighted_score.shape) != (n_heads, STATE.ctx_len)
        or weighted_score.device != attention.device
    ):
        raise RuntimeError(f"layer {layer} structured score shape/device changed")

    location: dict[int, tuple[int, int]] = {
        index: (group_index, position)
        for group_index, group in enumerate(plan.groups)
        for position, index in enumerate(group)
    }
    context_seen = min(STATE.ctx_len, k_len)
    for selected_position, absolute_index in enumerate(selected):
        group_index, within_group = location[absolute_index]
        weight = plan.weights[group_index][within_group]
        weighted_score[:, :context_seen].add_(
            attention[:, selected_position, :context_seen] * weight
        )
    observed.update(selected)
    if observed == set(STATE.expected_indices()):
        STATE._finalize_layer(layer)


def structured_policy_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
):
    """Delegate attention/compression, then record the configured query scores."""
    q_len, k_len = int(query.shape[2]), int(key.shape[2])
    base = k_len - q_len
    if (
        STATE.capture
        and STATE.intersects(base, q_len)
        and compress.STATE.enabled
    ):
        raise RuntimeError(
            "structured prompt capture must run before compression is enabled"
        )
    output = compress.sieve_compress_attention(
        module,
        query,
        key,
        value,
        attention_mask=attention_mask,
        scaling=scaling,
        dropout=dropout,
        **kwargs,
    )
    _capture_weighted_attention(
        module, query, key, attention_mask, scaling
    )
    return output


def install() -> None:
    """Register the wrapper while retaining the underlying implementation."""
    compress.install()
    compress.ALL_ATTENTION_FUNCTIONS[IMPL] = structured_policy_attention


def _validate_budget(B: float, maxb: int) -> tuple[float, int]:
    if isinstance(B, bool) or not isinstance(B, Real):
        raise ValueError("B must be a real number")
    budget = float(B)
    width = _positive_int(maxb, "maxb")
    if not math.isfinite(budget) or not 0.0 < budget <= width:
        raise ValueError("B must be finite and lie in (0, maxb]")
    return budget, width


def _validate_score(
    score: torch.Tensor,
    *,
    label: str,
    expected_shape: tuple[int, int] | None = None,
    normalized: bool = False,
) -> torch.Tensor:
    if not isinstance(score, torch.Tensor) or score.ndim != 2:
        raise ValueError(f"{label} must be a [heads, C] tensor")
    if score.shape[0] <= 0 or score.shape[1] <= 0:
        raise ValueError(f"{label} must have positive axes")
    if expected_shape is not None and tuple(score.shape) != expected_shape:
        raise ValueError(
            f"{label} has shape {tuple(score.shape)}, expected {expected_shape}"
        )
    values = score.float()
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError(f"{label} must be finite and nonnegative")
    mass = values.sum(dim=-1)
    if bool((mass <= 0).any()):
        raise ValueError(f"every row of {label} must have positive mass")
    if normalized and not torch.allclose(
        mass, torch.ones_like(mass), rtol=1e-5, atol=1e-6
    ):
        raise ValueError(f"every row of {label} must sum to one")
    return values


def allocate_structured_evict(
    score_kv: torch.Tensor,
    B: float,
    maxb: int,
    *,
    pool: int = router.SNAPKV_POOL,
) -> torch.Tensor:
    """Allocate the structured eviction corner through the existing router."""
    budget, width = _validate_budget(B, maxb)
    score = _validate_score(score_kv, label="structured KV-head score")
    if isinstance(pool, bool) or not isinstance(pool, Integral) or int(pool) <= 0:
        raise ValueError("pool must be a positive integer")
    bits = router.allocate("evict", budget, score, width, pool=int(pool))
    if bits is None:  # defensive: the evict arm always returns a tensor
        raise RuntimeError("router.allocate('evict') returned no allocation")
    return bits


def allocate_structured_interior(
    ctx: router.LayerCtx,
    score_h: torch.Tensor,
    B: float,
    maxb: int,
) -> torch.Tensor:
    """Run the existing interior with structured ``ap`` and unchanged ``sig2``.

    ``ctx.sig2`` is the existing last-prefill-query quantizer noise model built
    by :func:`router.build_layer_ctx`; this function substitutes only the
    attention distribution passed as ``ap``.
    """
    budget, width = _validate_budget(B, maxb)
    expected = (ctx.snap.shape[0] * int(ctx.n_rep), ctx.snap.shape[1])
    score = _validate_score(
        score_h,
        label="structured query-head score",
        expected_shape=expected,
    ).to(ctx.Vc.device)
    ap = score.double()
    ap = ap / ap.sum(dim=-1, keepdim=True)
    if ctx.sig2 is None or len(ctx.sig2) != expected[0]:
        raise ValueError(
            "structured interior requires the existing per-query-head noise model"
        )
    return router.alloc_interior(ctx, budget, width, ap=ap)


def allocate_structured(
    arm: str,
    B: float,
    maxb: int,
    *,
    score_h: torch.Tensor | None = None,
    score_kv: torch.Tensor | None = None,
    ctx: router.LayerCtx | None = None,
    pool: int = router.SNAPKV_POOL,
) -> torch.Tensor:
    """Dispatch the two V7 structured policies without reimplementing them."""
    if arm == "evict":
        if score_kv is None:
            raise ValueError("structured evict requires score_kv")
        return allocate_structured_evict(score_kv, B, maxb, pool=pool)
    if arm == "interior":
        if ctx is None or score_h is None:
            raise ValueError("structured interior requires ctx and score_h")
        return allocate_structured_interior(ctx, score_h, B, maxb)
    raise ValueError("structured arm must be 'evict' or 'interior'")
