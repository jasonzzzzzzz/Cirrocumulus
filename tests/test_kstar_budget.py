"""Focused tests for the isolated R9 group-K* and count-policy mathematics."""
from __future__ import annotations

import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sievelib.budget_policies import (
    bounded_proportional_projection,
    fixed_uniform_counts,
    kappa4_raw_counts,
    kappa4_rebalanced_counts,
    kstar_prop_counts,
    shrink20_counts,
)
from sievelib.group_error_curve import (
    dense_group_error_curve,
    exact_kstar,
    group_error_checkpoints,
)


def _assert_raises(error, match, fn):
    try:
        fn()
    except error as exc:
        assert match in str(exc), str(exc)
    else:
        raise AssertionError(f"expected {error.__name__} containing {match!r}")


def _brute_curve(full, qctx, values, order, k_max=None, eps=1e-12):
    """Independent per-K construction used only as the test oracle."""
    N, C = full.shape[-1], qctx.shape[-1]
    k_max = C if k_max is None else k_max
    f = full.reshape(-1, N).double()
    q = qctx.reshape(-1, C).double().clone()
    q[torch.isneginf(f[:, :C])] = -torch.inf
    v = values.double()
    ref = torch.softmax(f, -1) @ v
    scale = ref.norm(dim=-1).clamp_min(eps).square()
    losses = []
    for k in range(1, k_max + 1):
        keep = order[:k]
        logits = torch.cat([q[:, keep], f[:, C:]], -1)
        vv = torch.cat([v[:C][keep], v[C:]], 0)
        out = torch.softmax(logits, -1) @ vv
        losses.append(((out - ref).square().sum(-1) / scale).mean())
    return torch.stack(losses)


def _brute_checkpoints(full, qctx, values, order, counts, eps=1e-12):
    N, C = full.shape[-1], qctx.shape[-1]
    f = full.reshape(-1, N).double()
    q = qctx.reshape(-1, C).double().clone()
    q[torch.isneginf(f[:, :C])] = -torch.inf
    v = values.double()
    ref = torch.softmax(f, -1) @ v
    scale = ref.norm(dim=-1).clamp_min(eps).square()
    losses = []
    for count in counts:
        keep = order[:count]
        logits = torch.cat([q[:, keep], f[:, C:]], -1)
        vv = torch.cat([v[:C][keep], v[C:]], 0)
        out = torch.softmax(logits, -1) @ vv
        losses.append(((out - ref).square().sum(-1) / scale).mean())
    return torch.stack(losses)


def test_dense_curve_matches_brute_with_tail_and_causal_masks():
    torch.manual_seed(17)
    rows, C, tail, d = 6, 13, 5, 7
    full = torch.randn(rows, C + tail, dtype=torch.float64)
    qctx = full[:, :C] + 0.17 * torch.randn(rows, C, dtype=torch.float64)
    values = torch.randn(C + tail, d, dtype=torch.float64)
    order = torch.randperm(C)

    # Different causal visibility in the protected window and explicit masked
    # context entries.  qctx is intentionally left finite at two context masks:
    # the exact-logit mask must remain authoritative.
    full[0, C + 3:] = -torch.inf
    full[1, C + 1:] = -torch.inf
    full[2, C + 4:] = -torch.inf
    full[3, order[0]] = -torch.inf
    full[4, order[3]] = -torch.inf
    qctx[5, order[1]] = -torch.inf

    got = dense_group_error_curve(full, qctx, values, order, chunk_size=3)
    want = _brute_curve(full, qctx, values, order)
    torch.testing.assert_close(got, want, rtol=2e-12, atol=2e-13)

    # Chunk boundaries must be semantically invisible.
    one_chunk = dense_group_error_curve(full, qctx, values, order, chunk_size=C)
    torch.testing.assert_close(got, one_chunk, rtol=2e-12, atol=2e-13)


def test_protected_tail_is_always_retained():
    # The protected row carries almost all probability.  Keeping it makes every
    # K close to the full output; dropping it would put the output near zero.
    full = torch.tensor([[0.0, -1.0, -2.0, 12.0]], dtype=torch.float64)
    qctx = full[:, :3] + torch.tensor([[0.02, -0.03, 0.01]])
    values = torch.tensor([[0.0], [0.0], [0.0], [9.0]], dtype=torch.float64)
    order = torch.tensor([2, 0, 1])
    got = dense_group_error_curve(full, qctx, values, order, chunk_size=2)
    want = _brute_curve(full, qctx, values, order)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-15)
    assert float(got.max()) < 1e-8


def test_nonmonotone_dense_curve_and_exact_minimum():
    # Adding token 1 gives its badly perturbed logit half the mass and moves the
    # scalar output far from FP.  Adding token 2 cancels it again.  This pins the
    # reason exact K* scans every integer rather than binary-searching.
    full = torch.tensor([[10.0, 0.0, 0.0, -5.0]], dtype=torch.float64)
    qctx = torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float64)
    values = torch.tensor([[0.0], [10.0], [-10.0], [1.0]], dtype=torch.float64)
    order = torch.tensor([0, 1, 2])
    curve = dense_group_error_curve(full, qctx, values, order, chunk_size=2)
    torch.testing.assert_close(curve, _brute_curve(full, qctx, values, order))
    assert float(curve[1]) > float(curve[0])
    assert float(curve[1]) > float(curve[2])

    # K=2 lies exactly on the 10% threshold, K=3 gets worse, and K=1 fails.
    explicit = torch.tensor([1.1000001, 1.1, 9.0, 1.0], dtype=torch.float64)
    assert exact_kstar(explicit, 4) == 2
    assert exact_kstar({1: 7.0, 2: 1.1, 3: 5.0, 4: 1.0}, 4) == 2


def test_g1_reduces_exactly_to_the_single_row_curve():
    torch.manual_seed(4)
    C, tail, d = 9, 3, 5
    full = torch.randn(C + tail, dtype=torch.float64)
    qctx = full[:C] + 0.09 * torch.randn(C, dtype=torch.float64)
    values = torch.randn(C + tail, d, dtype=torch.float64)
    order = torch.randperm(C)
    flat = dense_group_error_curve(full, qctx, values, order, chunk_size=4)
    grouped = dense_group_error_curve(
        full.view(1, 1, -1), qctx.view(1, 1, -1), values, order, chunk_size=4
    )
    torch.testing.assert_close(flat, grouped, rtol=0, atol=0)
    torch.testing.assert_close(flat, _brute_curve(full, qctx, values, order), rtol=2e-12, atol=2e-13)


def test_checkpoint_curve_matches_brute_at_zero_duplicates_and_above_k0():
    torch.manual_seed(29)
    rows, C, tail, d = 5, 17, 4, 6
    full = torch.randn(rows, C + tail, dtype=torch.float64)
    qctx = full[:, :C] + 0.13 * torch.randn(rows, C, dtype=torch.float64)
    values = torch.randn(C + tail, d, dtype=torch.float64)
    order = torch.randperm(C)
    full[0, C + 2:] = -torch.inf
    full[1, C + 1:] = -torch.inf
    full[2, order[0]] = -torch.inf
    counts = torch.tensor([C, 0, 5, 5, 1, 13, 0, 9])
    got = group_error_checkpoints(
        full, qctx, values, order, counts, chunk_size=3
    )
    want = _brute_checkpoints(full, qctx, values, order, counts.tolist())
    assert got.shape == counts.shape
    torch.testing.assert_close(got, want, rtol=3e-12, atol=3e-13)
    assert float(got[1]) == float(got[6])
    assert float(got[2]) == float(got[3])

    dense = dense_group_error_curve(full, qctx, values, order, chunk_size=3)
    for i, count in enumerate(counts.tolist()):
        if count:
            torch.testing.assert_close(
                got[i], dense[count - 1], rtol=3e-12, atol=3e-13
            )


def test_checkpoint_g1_extreme_logits_and_validation():
    full = torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float64)
    qctx = torch.tensor([[-1000.0, 0.0]], dtype=torch.float64)
    values = torch.tensor([[3.0], [7.0], [5.0]], dtype=torch.float64)
    order = torch.tensor([0, 1])
    counts = torch.tensor([[2, 0], [1, 2]])
    got = group_error_checkpoints(full, qctx, values, order, counts, chunk_size=2)
    want = _brute_checkpoints(
        full, qctx, values, order, counts.reshape(-1).tolist()
    )
    torch.testing.assert_close(
        got.reshape(-1), want, rtol=2e-12, atol=2e-13
    )
    assert bool(torch.isfinite(got).all())
    _assert_raises(
        ValueError,
        "counts must lie",
        lambda: group_error_checkpoints(full, qctx, values, order, [-1, 2]),
    )


def test_positive_infinity_has_softmax_limit_semantics():
    full = torch.tensor([[0.0, -torch.inf, 1.0]], dtype=torch.float64)
    qctx = torch.tensor([[torch.inf, 99.0]], dtype=torch.float64)
    values = torch.tensor([[2.0], [100.0], [4.0]], dtype=torch.float64)
    # The full -inf mask on context token 1 overrides its finite qctx value.
    got = dense_group_error_curve(full, qctx, values, torch.tensor([1, 0]))
    assert math.isfinite(float(got[0]))
    assert math.isfinite(float(got[1]))


def test_projection_exact_sum_caps_and_zero_fallback():
    got = bounded_proportional_projection([100.0, 1.0, 1.0], 8, 4)
    assert got.tolist() == [4, 2, 2]
    assert int(got.sum()) == 8
    assert bool(((got >= 1) & (got <= 4)).all())

    # Once the positive demand saturates, a feasible residual must go to zero
    # demands rather than being lost.
    zero = bounded_proportional_projection([5.0, 0.0, 0.0], 8, [3, 4, 4])
    assert zero.tolist() == [3, 3, 2]
    assert int(zero.sum()) == 8

    _assert_raises(
        ValueError,
        "infeasible",
        lambda: bounded_proportional_projection([1.0, 2.0], 7, 3),
    )


def test_projection_ties_use_stable_keys_not_input_position():
    keys = [(1, 0), (0, 1), (0, 0)]
    got = bounded_proportional_projection([1.0, 1.0, 1.0], 4, 3, keys=keys)
    assert got.tolist() == [1, 1, 2]  # lexicographically first key gets remainder

    perm = torch.tensor([2, 0, 1])
    raw2 = torch.ones(3)[perm]
    keys2 = [keys[i] for i in perm.tolist()]
    got2 = bounded_proportional_projection(raw2, 4, 3, keys=keys2)
    remapped = torch.empty_like(got2)
    remapped[perm] = got2
    assert torch.equal(remapped, got)


def test_policy_helpers_preserve_total_and_historical_kappa_rule():
    keys = [(0, 0), (0, 1), (1, 0)]
    assert fixed_uniform_counts(3, 4, 8, keys=keys).tolist() == [4, 4, 4]

    ks = torch.tensor([1.0, 4.0, 7.0])
    prop = kstar_prop_counts(ks, 4, 8, keys=keys)
    shrink = shrink20_counts(ks, 4, 8, keys=keys)
    assert prop.tolist() == [1, 4, 7]
    assert shrink.tolist() == [2, 4, 6]
    assert int(prop.sum()) == int(shrink.sum()) == 12

    # Historical abs rule: int(min(k0, max(4*n95, 256))).  Float means are
    # accepted and truncated exactly as the old Python int conversion did.
    n95 = torch.tensor([10.5, 80.9, 400.0], dtype=torch.float64)
    raw = kappa4_raw_counts(n95, k0=1000, capacity=1200)
    assert raw.tolist() == [256, 323, 1000]
    reb = kappa4_rebalanced_counts(n95, k0=1000, capacity=1200, keys=keys)
    assert int(reb.sum()) == 3000
    assert bool(((reb >= 1) & (reb <= 1200)).all())



if __name__ == "__main__":
    tests = [
        test_dense_curve_matches_brute_with_tail_and_causal_masks,
        test_protected_tail_is_always_retained,
        test_nonmonotone_dense_curve_and_exact_minimum,
        test_g1_reduces_exactly_to_the_single_row_curve,
        test_checkpoint_curve_matches_brute_at_zero_duplicates_and_above_k0,
        test_checkpoint_g1_extreme_logits_and_validation,
        test_positive_infinity_has_softmax_limit_semantics,
        test_projection_exact_sum_caps_and_zero_fallback,
        test_projection_ties_use_stable_keys_not_input_position,
        test_policy_helpers_preserve_total_and_historical_kappa_rule,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} K* budget tests")
