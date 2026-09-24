"""R11 nested codebook: default path unchanged, true nesting, base identity."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sievelib import quant as Q  # noqa: E402

CHAIN = Q.NESTED_CHAINS["nested3"]


def _keys(seed=0, L=3000, d=128):
    g = torch.Generator().manual_seed(seed)
    # heavy-ish tails and a nonzero mean, like real keys
    return (torch.randn(2, L, d, generator=g) * 1.7 + 0.3).float()


def test_default_path_is_bitwise_unchanged():
    K = _keys()
    R = Q.random_rotation(128, "cpu", seed=2)
    for b in (1, 2, 3, 4, 5, 6, 8):
        ref = Q.quantize_keys(K, b, R, True)
        assert torch.equal(ref, Q.quantize_keys(K, b, R, True, codebook="lloyd"))
        assert torch.equal(ref, Q.quantize_keys(K, b, R, True, codebook=None))


def test_widths_outside_chain_and_base_are_monolithic():
    K = _keys(1)
    R = Q.random_rotation(128, "cpu", seed=2)
    for b in (1, 2, 5, CHAIN[0]):
        assert torch.equal(Q.quantize_keys(K, b, R, True),
                           Q.quantize_keys(K, b, R, True, codebook="nested3"))
    for b in CHAIN[1:]:
        assert not torch.equal(Q.quantize_keys(K, b, R, True),
                               Q.quantize_keys(K, b, R, True, codebook="nested3"))


def test_partitions_nest_and_codebooks_are_well_formed():
    y = torch.randn(400_000, generator=torch.Generator().manual_seed(3))
    idx = {}
    for b in CHAIN:
        if b == CHAIN[0]:
            lv = Q.levels_for(b, "cpu")
            bd = (lv[1:] + lv[:-1]) / 2
        else:
            lv, bd = Q.nested_codebook("nested3", b, "cpu")
        assert lv.numel() == 2 ** b and bd.numel() == 2 ** b - 1
        assert bool((lv[1:] > lv[:-1]).all()) and bool((bd[1:] > bd[:-1]).all())
        # each boundary separates its two neighbouring levels
        assert bool((lv[:-1] < bd).all()) and bool((bd < lv[1:]).all())
        idx[b] = torch.bucketize(y, bd)
    for a in CHAIN:
        for c in CHAIN:
            if c > a:
                assert torch.equal(idx[c] // 2 ** (c - a), idx[a]), (a, c)


def test_nested_distortion_is_close_to_but_not_below_monolithic():
    x = torch.linspace(-9, 9, 400_001, dtype=torch.float64)
    p = torch.exp(-0.5 * x ** 2)
    p /= p.sum()

    def mse(lv, bd):
        return float((p * (x - lv[torch.bucketize(x, bd)]) ** 2).sum())

    design = Q.design_nested(CHAIN)
    last = float("inf")
    for b in CHAIN:
        lv, bd = design[b]
        m = Q.levels_for(b, "cpu").double()
        dm = mse(m, (m[1:] + m[:-1]) / 2)
        dn = mse(lv, bd)
        assert dn >= dm * (1 - 1e-6), (b, dn, dm)       # constrained >= optimal
        assert dn <= dm * 1.25, (b, dn, dm)             # and not wildly worse
        assert dn < last
        last = dn


def test_bad_codebook_is_refused():
    try:
        Q.check_codebook("nested9")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown codebook accepted")


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    for t in (test_default_path_is_bitwise_unchanged,
              test_widths_outside_chain_and_base_are_monolithic,
              test_partitions_nest_and_codebooks_are_well_formed,
              test_nested_distortion_is_close_to_but_not_below_monolithic,
              test_bad_codebook_is_refused):
        t()
        print(f"PASS {t.__name__}")
