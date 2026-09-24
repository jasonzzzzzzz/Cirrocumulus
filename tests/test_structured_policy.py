#!/usr/bin/env python3
"""Focused CPU contracts for the isolated V7 structured-query policy."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sievelib import compress as C  # noqa: E402
from sievelib import router  # noqa: E402
from sievelib import structured_policy as SP  # noqa: E402


def expect_error(action, error_type, contains: str) -> None:
    try:
        action()
        raise AssertionError("invalid contract was accepted")
    except error_type as caught:
        assert contains in str(caught), str(caught)


def groups() -> tuple[tuple[int, ...], ...]:
    return ((0, 1), (2,), (3, 5), (7,), (8, 10))


def brute_group_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    query_groups,
    weights,
    *,
    ctx_len: int,
    scaling: float,
) -> torch.Tensor:
    H, Hkv = query.shape[1], key.shape[1]
    repeats = H // Hkv
    length = key.shape[2]
    out = torch.zeros(5, H, ctx_len, dtype=torch.float32)
    for group_index, group in enumerate(query_groups):
        for within_group, absolute_index in enumerate(group):
            q = query[0, :, absolute_index]
            logits = torch.einsum(
                "grd,gkd->grk",
                q.reshape(Hkv, repeats, -1).float(),
                key[0].float(),
            ).reshape(H, length)
            logits *= scaling
            logits[:, absolute_index + 1 :] = float("-inf")
            attention = torch.softmax(logits, dim=-1)
            out[group_index] += (
                attention[:, :ctx_len] * weights[group_index][within_group]
            )
    return out


def test_group_validation_and_equal_mass_aggregation() -> None:
    plan = SP.validate_absolute_query_groups(groups(), prompt_length=12)
    assert len(plan.groups) == 5
    assert math.isclose(sum(sum(group) for group in plan.weights), 1.0)
    for indices, weights in zip(plan.groups, plan.weights):
        assert len(set(weights)) == 1
        assert math.isclose(sum(weights), 0.2)

    supplied = tuple(tuple(value for value in row) for row in plan.weights)
    assert SP.validate_absolute_query_groups(
        groups(), prompt_length=12, query_weights=supplied
    ) == plan
    malformed = [
        (groups()[:4], "exactly 5"),
        (((0,), (1,), (), (2,), (3,)), "nonempty"),
        (((1, 0), (2,), (3,), (4,), (5,)), "strictly increasing"),
        (((0,), (0,), (1,), (2,), (3,)), "multiple groups"),
        (((0,), (1,), (2,), (3,), (12,)), "outside"),
    ]
    for value, message in malformed:
        expect_error(
            lambda value=value: SP.validate_absolute_query_groups(
                value, prompt_length=12
            ),
            ValueError,
            message,
        )
    bad_weights = [list(row) for row in supplied]
    bad_weights[0][0] += 0.01
    expect_error(
        lambda: SP.validate_absolute_query_groups(
            groups(), prompt_length=12, query_weights=bad_weights
        ),
        ValueError,
        "mass 1/5",
    )

    raw = torch.zeros(5, 2, 4)
    for group_index in range(5):
        raw[group_index, :, group_index % 4] = 0.2 * (group_index + 1)
    aggregate = SP.aggregate_equal_group_attention(raw)
    expected = raw.sum(dim=0)
    assert torch.equal(aggregate, expected)
    # A field with less context attention keeps less context mass; it is not
    # silently renormalized back to one fifth.
    changed = raw.clone()
    changed[0] *= 0.25
    assert torch.equal(
        SP.aggregate_equal_group_attention(changed), changed.sum(dim=0)
    )
    zero = torch.zeros_like(raw)
    expect_error(
        lambda: SP.aggregate_equal_group_attention(zero),
        ValueError,
        "positive weighted mass",
    )


def test_bruteforce_capture_causality_delegation_and_registration() -> None:
    generator = torch.Generator().manual_seed(91)
    H, Hkv, length, dim, context = 4, 2, 12, 8, 9
    query = torch.randn(1, H, length, dim, generator=generator)
    key = torch.randn(1, Hkv, length, dim, generator=generator)
    value = torch.randn(1, Hkv, length, dim, generator=generator)
    module = SimpleNamespace(layer_idx=0, head_dim=dim)
    scaling = dim ** -0.5

    C.STATE.reset_prompt()
    SP.STATE.reset_prompt()
    expected_output, _ = C.sieve_compress_attention(
        module, query, key, value, scaling=scaling
    )
    plan = SP.begin_capture(
        groups(), prompt_length=length, ctx_len=context
    )
    actual_output, returned_weights = SP.structured_policy_attention(
        module, query, key, value, scaling=scaling
    )
    assert returned_weights is None
    assert torch.equal(actual_output, expected_output)

    brute_groups = brute_group_scores(
        query,
        key,
        plan.groups,
        plan.weights,
        ctx_len=context,
        scaling=scaling,
    )
    # Capture retains only the direct weighted [H,C] sum.  This is exactly
    # equivalent to summing the five brute-force group tensors, while using
    # one fifth of their persistent storage.
    assert not hasattr(SP.STATE, "group_score_h")
    assert SP.STATE.score_h[0].numel() * SP.QUERY_GROUP_COUNT == brute_groups.numel()
    assert torch.allclose(
        SP.STATE.score_h[0], brute_groups.sum(dim=0), rtol=1e-6, atol=1e-7
    )
    # Group zero contains only queries 0 and 1, so causality makes every key
    # after absolute position one exactly unreachable.
    assert torch.count_nonzero(brute_groups[0, :, 2:]) == 0
    expected_h = brute_groups.sum(dim=0)
    expected_kv = expected_h.reshape(Hkv, H // Hkv, context).sum(1)
    score_h, score_kv = SP.STATE.layer_scores(0)
    assert torch.allclose(score_h, expected_h, rtol=1e-6, atol=1e-7)
    assert torch.allclose(score_kv, expected_kv, rtol=1e-6, atol=1e-7)
    all_h, all_kv = SP.end_capture([0])
    assert torch.equal(all_h[0], score_h) and torch.equal(all_kv[0], score_kv)

    SP.install()
    assert C.ALL_ATTENTION_FUNCTIONS[SP.IMPL] is SP.structured_policy_attention
    assert C.ALL_ATTENTION_FUNCTIONS[C.IMPL] is C.sieve_compress_attention


def capture_in_chunks(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    boundaries: tuple[int, ...],
    *,
    ctx_len: int,
):
    module = SimpleNamespace(layer_idx=3, head_dim=query.shape[-1])
    C.STATE.reset_prompt()
    SP.begin_capture(groups(), prompt_length=query.shape[2], ctx_len=ctx_len)
    outputs = []
    start = 0
    for end in boundaries:
        output, _ = SP.structured_policy_attention(
            module,
            query[:, :, start:end],
            key[:, :, :end],
            value[:, :, :end],
            scaling=query.shape[-1] ** -0.5,
        )
        outputs.append(output)
        start = end
    scores = tuple(value.clone() for value in SP.STATE.layer_scores(3))
    SP.end_capture([3])
    return torch.cat(outputs, dim=1), scores


def test_chunk_equivalence_across_group_boundaries() -> None:
    generator = torch.Generator().manual_seed(123)
    H, Hkv, length, dim, context = 8, 2, 12, 16, 9
    query = torch.randn(1, H, length, dim, generator=generator)
    key = torch.randn(1, Hkv, length, dim, generator=generator)
    value = torch.randn(1, Hkv, length, dim, generator=generator)
    full_output, full_scores = capture_in_chunks(
        query, key, value, (length,), ctx_len=context
    )
    chunk_output, chunk_scores = capture_in_chunks(
        query, key, value, (3, 6, 8, 9, length), ctx_len=context
    )
    assert torch.allclose(chunk_output, full_output, rtol=1e-5, atol=1e-6)
    assert torch.allclose(chunk_scores[0], full_scores[0], rtol=1e-6, atol=1e-7)
    assert torch.allclose(chunk_scores[1], full_scores[1], rtol=1e-6, atol=1e-7)


def test_incomplete_and_malformed_capture_rejected() -> None:
    SP.STATE.reset_prompt()
    SP.begin_capture(groups(), prompt_length=12, ctx_len=9)
    module = SimpleNamespace(layer_idx=0, head_dim=4)
    query = torch.randn(1, 4, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)
    C.STATE.reset_prompt()
    SP.structured_policy_attention(module, query, key, value)
    expect_error(lambda: SP.end_capture([0]), RuntimeError, "incomplete")

    SP.STATE.reset_prompt()
    SP.begin_capture(groups(), prompt_length=12, ctx_len=9)
    C.STATE.reset_prompt()
    C.STATE.enabled = True
    try:
        expect_error(
            lambda: SP.structured_policy_attention(module, query, key, value),
            RuntimeError,
            "before compression",
        )
    finally:
        C.STATE.reset_prompt()
        SP.STATE.reset_prompt()

    expect_error(
        lambda: SP.allocate_structured("unknown", 2, 8),
        ValueError,
        "'evict' or 'interior'",
    )
    expect_error(
        lambda: SP.allocate_structured_evict(torch.tensor([[float("nan")]]), 2, 8),
        ValueError,
        "finite",
    )


def make_layer_ctx(seed: int = 77) -> router.LayerCtx:
    generator = torch.Generator().manual_seed(seed)
    Hkv, repeats, context, dim = 2, 2, 96, 8
    H = Hkv * repeats
    Kc = torch.randn(Hkv, context, dim, generator=generator)
    Vc = torch.randn(Hkv, context, dim, generator=generator)
    Kw = torch.randn(Hkv, 8, dim, generator=generator)
    Vw = torch.randn(Hkv, 8, dim, generator=generator)
    snap = torch.rand(Hkv, context, generator=generator)
    ap = torch.rand(H, context, generator=generator)
    ap /= ap.sum(-1, keepdim=True)
    bit_list = (1, 2, 3, 4, 5, 6, 8)
    sig2 = [
        {bit: (1.0 + 0.15 * head) / (bit * bit) for bit in bit_list}
        for head in range(H)
    ]
    return router.LayerCtx(
        li=0,
        n_rep=repeats,
        scaling=dim ** -0.5,
        Kc=Kc,
        Vc=Vc,
        Kw=Kw,
        Vw=Vw,
        snap=snap,
        ap=ap,
        sig2=sig2,
    )


def test_structured_allocations_budget_noise_and_distinctness() -> None:
    context = 97
    legacy_kv = torch.arange(context, dtype=torch.float32).repeat(2, 1)
    structured_kv = legacy_kv.flip(-1)
    evict = SP.allocate_structured_evict(
        structured_kv, 2, 8, pool=1
    )
    legacy_evict = router.allocate("evict", 2, legacy_kv, 8, pool=1)
    assert evict.shape == structured_kv.shape
    assert float(evict.double().mean()) <= 2.0 + 1e-12
    assert float(evict.double().mean()) >= 2.0 - 8.0 / context - 1e-12
    assert not torch.equal(evict, legacy_evict)

    ctx = make_layer_ctx()
    H, Cn = ctx.snap.shape[0] * ctx.n_rep, ctx.snap.shape[1]
    structured_h = torch.full((H, Cn), 1e-6)
    for head in range(H):
        start = (head * 19) % (Cn - 12)
        structured_h[head, start : start + 12] = 1.0
    structured_h /= structured_h.sum(-1, keepdim=True)
    sig2_before = copy.deepcopy(ctx.sig2)
    sig2_identity = id(ctx.sig2)
    interior = SP.allocate_structured_interior(ctx, structured_h, 2, 8)
    legacy_interior = router.alloc_interior(ctx, 2, 8)
    assert interior.shape == ctx.snap.shape
    assert float(interior.double().mean()) <= 2.0 + 1e-9
    assert set(torch.unique(interior).tolist()) <= {0, 1, 2, 3, 4, 5, 6, 8}
    assert not torch.equal(interior, legacy_interior)
    assert id(ctx.sig2) == sig2_identity and ctx.sig2 == sig2_before

    via_dispatch = SP.allocate_structured(
        "interior", 2, 8, ctx=ctx, score_h=structured_h
    )
    assert torch.equal(via_dispatch, interior)
    # Interior performs one per-head normalization after the direct weighted
    # sum, so a common positive scale cannot change its allocation.
    scaled = SP.allocate_structured_interior(ctx, structured_h * 2.0, 2, 8)
    assert torch.equal(scaled, interior)


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS: {len(tests)} structured-policy tests")
