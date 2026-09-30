#!/usr/bin/env python3
"""CPU invariants for r13lib (run before any submission)."""
from __future__ import annotations
import pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

import torch
from sievelib import quant
import r13lib as L13


def keys(L=700, d=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(L, d, generator=g)
    K[:, 3] = K[:, 3] * 20 + 5            # an outlier channel with an offset
    K[:, 7] = torch.randn(L, generator=g).exp()   # a skewed channel
    return K


def test_chan_quant():
    K = keys()
    m0 = L13.chan_quant(K, 0)
    gid = torch.arange(K.shape[0]) // L13.GROUP
    for gi in range(int(gid.max()) + 1):
        assert torch.allclose(m0[gid == gi], K[gid == gi].mean(0, keepdim=True).expand_as(m0[gid == gi]), atol=1e-4)
    e8 = float((K - L13.chan_quant(K, 8)).norm() / K.norm())
    assert e8 < 0.01, e8
    prev = float("inf")
    for b in (1, 2, 3, 4, 5, 6, 8):
        e = float((K - L13.chan_quant(K, b)).pow(2).sum())
        assert e < prev, (b, e, prev)
        prev = e


def test_mix_is_per_channel_gather():
    K = keys()
    S = L13.tier_stack(K, "raw")
    idx = torch.randint(0, len(L13.TIERS), (K.shape[1],), generator=torch.Generator().manual_seed(1))
    M = L13.mix(S, idx)
    for c in range(K.shape[1]):
        assert torch.equal(M[:, c], S[idx[c], :, c])


def test_rot_uniform_equals_turboquant():
    K = keys()
    R = quant.random_rotation(K.shape[1], "cpu", seed=2)
    y, gam = L13.rot_frame(K, R)
    q = torch.randn(4, K.shape[1])
    for B in (2, 3, 4):
        ref = q @ quant.quantize_keys(K, B, R, norm_correct=False).T
        got = (q @ R.T) @ L13.rot_quant(y, gam, B).T
        rel = float((got - ref).abs().max() / ref.abs().max())
        assert rel < 1e-4, (B, rel)


def test_waterfill_cost():
    torch.manual_seed(0)
    n, T = 128, len(L13.TIERS)
    bt = torch.tensor(L13.TIERS, dtype=torch.float64)
    sens = torch.rand(n, dtype=torch.float64) ** 4 * 100
    cost = sens[:, None] * torch.pow(4.0, -bt)[None, :]
    for B in (2, 3, 4):
        idx = L13.waterfill_cost(cost, B)
        spent = float(bt[idx].sum())
        assert spent <= B * n + 1e-9
        assert spent >= B * n - 8, (B, spent)
        # beats uniform on its own objective
        u = cost[:, L13.TIERS.index(B)].sum()
        assert float(cost.gather(1, idx[:, None]).sum()) <= float(u) + 1e-12
    # equal sensitivities -> (near-)uniform
    idx = L13.waterfill_cost(torch.ones(n, 1, dtype=torch.float64) * torch.pow(4.0, -bt)[None], 3)
    assert float(bt[idx].mean()) >= 2.95


def test_channel_costs_forms():
    K = keys(L=300)
    S = L13.tier_stack(K, "raw")
    ks = L13.channel_costs(K, S, None, None)
    assert ks.shape == (K.shape[1], len(L13.TIERS))
    q = torch.randn(2, K.shape[1])
    rw = torch.rand(2, K.shape[0])
    c = L13.channel_costs(K, S, q, rw)
    E = (K - S[2]).double()
    man = 0
    for h in range(2):
        w = rw[h].double()
        mu = (w @ E) / w.sum()
        man = man + q[h].double() ** 2 * (w @ (E - mu) ** 2)
    assert torch.allclose(c[:, 2], man, rtol=1e-6)
    # a constant per-channel shift costs nothing (softmax shift-invariance)
    c_shift = L13.channel_costs(K, S.double() - 3.0, q, rw)   # shift in f64: an f32 shift re-rounds S
    assert torch.allclose(c_shift, c, rtol=1e-6, atol=1e-9)
    # tier 0 (mean replacement) costs the same as zeroing the channel
    Z = S.clone(); Z[0] = 0
    assert torch.allclose(L13.channel_costs(K, Z, q, rw)[:, 0],
                          L13.channel_costs(K, torch.stack([K.mean(0).expand_as(K)] * 8), q, rw)[:, 0],
                          rtol=1e-6)


def test_rel_output_error_zero():
    s = torch.randn(3, 200); V = torch.randn(200, 16)
    o = torch.softmax(s.double(), -1) @ V.double()
    assert float(L13.rel_output_error(s, V, o).max()) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"ok  {name}")
    print("all R13 tests passed")
