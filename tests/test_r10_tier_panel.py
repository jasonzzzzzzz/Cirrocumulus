"""Focused CPU tests for R10's opt-in paired tier-set panel."""
from __future__ import annotations

import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sievelib import evict
from sievelib.alloc import (
    R10_TIER_PANEL,
    TIER_PANEL_REGISTRY,
    group_prepass,
    head_metrics,
    resolve_tier_panel,
)

BITS = (1, 2, 3, 4, 5, 6, 8)
BUDGETS = (2, 3)


def _assert_raises(error, text, fn):
    try:
        fn()
    except error as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError(f"expected {error.__name__} containing {text!r}")


def _head(seed: int, length: int = 384, dim: int = 16):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(length, generator=g, dtype=torch.float64) * 2.2
    V = torch.randn(length, dim, generator=g, dtype=torch.float64)
    # One common direction with a strictly decreasing magnitude gives a stable,
    # non-degenerate noise model while keeping this test independent of the key
    # quantizer implementation.
    z = torch.randn(length, generator=g, dtype=torch.float64)
    scales = {1: 1.30, 2: .62, 3: .29, 4: .13,
              5: .060, 6: .026, 8: .005}
    shat = {b: s + scales[b] * z for b in BITS}
    raw = torch.softmax(s + .35 * torch.randn(
        length, generator=g, dtype=torch.float64), -1)
    unseen = torch.zeros(length, dtype=torch.bool)
    return dict(s=s, shat=shat, V=V,
                raw={"accum": raw}, unseen={"accum": unseen})


def _metrics(head, extra):
    spec = evict.CornerSpec(
        evictors=("oracle", "accum"), policies=("frac",), kstar=False,
        interior_scores=("accum",)
    )
    raw = head["raw"]["accum"]
    return head_metrics(
        head["s"], head["shat"], head["V"], budgets=BUDGETS, maxb=8,
        practical_scores={"accum": raw}, corner=spec,
        interior_raw={"accum": raw}, interior_unseen=head["unseen"],
        extra=extra,
    )


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


def test_registry_and_parser_are_explicit_and_strict():
    panel = resolve_tier_panel("r10", BITS)
    assert tuple(panel) == R10_TIER_PANEL
    assert panel == TIER_PANEL_REGISTRY
    assert panel["full"] == (0, 1, 2, 3, 4, 5, 6, 8)
    assert panel["no1"] == (0, 2, 3, 4, 5, 6, 8)
    assert panel["base3_dense"] == (0, 3, 4, 5, 6, 8)
    assert panel["nested3"] == (0, 3, 4, 6, 8)
    assert panel["nested4"] == (0, 4, 6, 8)
    assert resolve_tier_panel(None, BITS) == {}
    assert tuple(resolve_tier_panel("nested3,full", BITS)) == ("nested3", "full")
    assert resolve_tier_panel({"custom": [3, 6, 8]}, BITS) == {
        "custom": (0, 3, 6, 8)
    }
    _assert_raises(ValueError, "unknown tier_panel", lambda: resolve_tier_panel("x", BITS))
    _assert_raises(ValueError, "unmeasured tiers", lambda: resolve_tier_panel(
        {"x": [3, 7, 8]}, BITS))
    _assert_raises(ValueError, "must retain maxb=8", lambda: resolve_tier_panel(
        {"x": [3, 4, 6]}, BITS))
    _assert_raises(ValueError, "stable column names", lambda: resolve_tier_panel(
        {"bad-label": [3, 8]}, BITS))


def test_panel_metrics_full_duplicate_and_nrep1_controls():
    head = _head(11)
    panel = resolve_tier_panel("r10", BITS)
    extra = group_prepass(
        [head], n_rep=1, budgets=BUDGETS, maxb=8, coarse_bits=(4,),
        group=True, tier_panel=panel,
    )[0]
    metrics = _metrics(head, extra)
    length = head["s"].numel()

    for budget in BUDGETS:
        old_err = f"err_wf_grp_csv_b4_accum_{budget}"
        old_evict = f"evict_frac_grp_csv_b4_accum_{budget}"
        assert old_err in metrics and old_evict in metrics
        for label, tiers in panel.items():
            stem = f"grp_tier_{label}_csv_b4_accum_{budget}"
            err = f"err_wf_{stem}"
            mean = f"mean_bits_{stem}"
            evict_col = f"evict_frac_{stem}"
            assert math.isfinite(metrics[err])
            assert 0 <= metrics[mean] <= budget + 8 / length
            assert 0 <= metrics[evict_col] <= 1
            fracs = {b: metrics[f"tier_frac_{stem}_b{b}"] for b in tiers}
            assert abs(sum(fracs.values()) - 1.0) < 1e-12
            assert abs(sum(b * f for b, f in fracs.items()) - metrics[mean]) < 1e-12
            assert fracs[0] == metrics[evict_col]
            assert not any(
                key.startswith(f"tier_frac_{stem}_b")
                for key in metrics
                if key.rsplit("b", 1)[-1].isdigit()
                and int(key.rsplit("b", 1)[-1]) not in tiers
            )
            assert metrics[
                f"alloc_mismatch_grp_vs_head_tier_{label}_csv_b4_accum_{budget}"
            ] == 0
            assert metrics[
                f"max_bit_diff_grp_vs_head_tier_{label}_csv_b4_accum_{budget}"
            ] == 0

        full = f"grp_tier_full_csv_b4_accum_{budget}"
        assert metrics[f"err_wf_{full}"] == metrics[old_err]
        assert metrics[f"evict_frac_{full}"] == metrics[old_evict]
        assert metrics[f"alloc_mismatch_full_vs_existing_csv_b4_accum_{budget}"] == 0
        assert metrics[f"max_bit_diff_full_vs_existing_csv_b4_accum_{budget}"] == 0


def test_panel_is_additive_and_default_off_is_unchanged():
    head = _head(23)
    panel = resolve_tier_panel("r10", BITS)
    off_extra = group_prepass(
        [head], n_rep=1, budgets=BUDGETS, maxb=8, coarse_bits=(4,),
        group=True,
    )[0]
    on_extra = group_prepass(
        [head], n_rep=1, budgets=BUDGETS, maxb=8, coarse_bits=(4,),
        group=True, tier_panel=panel,
    )[0]
    off = _metrics(head, off_extra)
    on = _metrics(head, on_extra)
    added = set(on) - set(off)
    assert added
    assert all("tier_" in key or "full_vs_existing" in key for key in added)
    assert set(off) <= set(on)
    for key in off:
        assert _same(off[key], on[key]), (key, off[key], on[key])
    assert not any("grp_tier_" in key or "full_vs_existing" in key for key in off)


def test_group_allocations_are_shared_and_use_only_offered_tiers():
    heads = [_head(31), _head(32)]
    panel = resolve_tier_panel("r10", BITS)
    extras = group_prepass(
        heads, n_rep=2, budgets=BUDGETS, maxb=8, coarse_bits=(4,),
        group=True, tier_panel=panel,
    )
    for budget in BUDGETS:
        for label, tiers in panel.items():
            a = extras[0]["group"]["tier_bits"][(label, budget)]
            b = extras[1]["group"]["tier_bits"][(label, budget)]
            assert torch.equal(a, b)
            assert set(int(x) for x in torch.unique(a)) <= set(tiers)
            assert int(a.sum()) <= budget * a.numel()
    # Group-vs-head is an n_rep=1 pilot control, not a claim for a real GQA
    # group; no misleading zero columns are emitted here.
    assert extras[0]["group"]["tier_controls"] == {}


if __name__ == "__main__":
    for test in (
        test_registry_and_parser_are_explicit_and_strict,
        test_panel_metrics_full_duplicate_and_nrep1_controls,
        test_panel_is_additive_and_default_off_is_unchanged,
        test_group_allocations_are_shared_and_use_only_offered_tiers,
    ):
        test()
        print(f"PASS {test.__name__}")
