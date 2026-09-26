"""R11 nested codebook: default path unchanged, true nesting, base identity."""
from __future__ import annotations

import math
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


def test_codebook_differs_exactly_on_the_nested_refinement_widths():
    diff = [b for b in (1, 2, 3, 4, 5, 6, 8) if Q.codebook_differs("lloyd", "nested3", b)]
    assert diff == [4, 6, 8]
    assert not any(Q.codebook_differs("nested3", "nested3", b) for b in range(1, 9))


def _ab_heads(n_rep=2, length=384, dim=16):
    """Heads as run_h0 builds them; the second codebook shares widths 1,2,3,5
    as the SAME tensors and perturbs 4,6,8 (amendment A2)."""
    heads, heads_ab = [], []
    for i in range(n_rep):
        g = torch.Generator().manual_seed(10 + i)
        s = torch.randn(length, generator=g, dtype=torch.float64) * 2.2
        V = torch.randn(length, dim, generator=g, dtype=torch.float64)
        z = torch.randn(length, generator=g, dtype=torch.float64)
        z2 = torch.randn(length, generator=g, dtype=torch.float64)
        sc = {1: 1.30, 2: .62, 3: .29, 4: .13, 5: .060, 6: .026, 8: .005}
        shat = {b: s + sc[b] * z for b in sc}
        shat_ab = {b: (s + sc[b] * (0.9 * z + 0.45 * z2) if b in (4, 6, 8) else shat[b])
                   for b in sc}
        raw = {"accum": torch.softmax(s + .35 * torch.randn(
            length, generator=g, dtype=torch.float64), -1)}
        unseen = {"accum": torch.zeros(length, dtype=torch.bool)}
        heads.append(dict(s=s, shat=shat, V=V, raw=raw, unseen=unseen))
        heads_ab.append(dict(s=s, shat=shat_ab, V=V, raw=raw, unseen=unseen))
    return heads, heads_ab


def _metrics_for(heads, extras):
    from sievelib import evict
    from sievelib.alloc import head_metrics
    spec = evict.CornerSpec(evictors=("oracle", "accum"), policies=("frac",),
                            kstar=False, interior_scores=("accum",))
    return [head_metrics(h["s"], h["shat"], h["V"], budgets=(1, 2, 3, 4), maxb=8,
                         practical_scores=dict(h["raw"]), corner=spec,
                         interior_raw=h["raw"], interior_unseen=h["unseen"],
                         extra=e) for h, e in zip(heads, extras)]


def _same_rec(a, b):
    assert a.keys() == b.keys()
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and math.isnan(x):
            assert isinstance(y, float) and math.isnan(y), k
        else:
            assert x == y, (k, x, y)


def test_in_process_ab_is_independent_and_shares_identical_widths():
    """The premise of amendment A2: running the second codebook on the same
    inputs neither changes the primary result nor mutates the shared inputs,
    and codebook-free / shared-width quantities are bitwise equal."""
    from sievelib.alloc import group_prepass, resolve_tier_panel
    heads, heads_ab = _ab_heads()
    panel = resolve_tier_panel("nested3", (1, 2, 3, 4, 5, 6, 8))
    kw = dict(n_rep=2, budgets=(1, 2, 3, 4), maxb=8, coarse_bits=(4,),
              group=True, tier_panel=panel)
    snap = [{k: (v.clone() if torch.is_tensor(v) else
                 {kk: vv.clone() for kk, vv in v.items()}) for k, v in h.items()}
            for h in heads]

    ref = _metrics_for(heads, group_prepass(heads, **kw))
    ab = _metrics_for(heads_ab, group_prepass(heads_ab, **kw))
    again = _metrics_for(heads, group_prepass(heads, **kw))
    for r, a2 in zip(ref, again):
        _same_rec(r, a2)                        # primary unaffected by the A/B pass
    for h, s0 in zip(heads, snap):              # shared inputs never mutated
        for k, v in s0.items():
            if torch.is_tensor(v):
                assert torch.equal(h[k], v), k
            else:
                for kk, vv in v.items():
                    assert torch.equal(h[k][kk], vv), (k, kk)
    for r, a in zip(ref, ab):
        for k in ("tau", "c1_abs", "c2_abs", "c3_abs", "c5_abs",
                  "err_uniform1", "err_uniform2", "err_uniform3"):
            assert r[k] == a[k], k               # shared widths: bitwise equal
        for k in ("c4_abs", "c6_abs", "c8_abs"):
            assert r[k] != a[k], k               # the codebook really differs
        assert any(r[f"err_wf_grp_tier_nested3_csv_b4_accum_{B}"] !=
                   a[f"err_wf_grp_tier_nested3_csv_b4_accum_{B}"] for B in (1, 2, 3, 4))


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
              test_codebook_differs_exactly_on_the_nested_refinement_widths,
              test_in_process_ab_is_independent_and_shares_identical_widths,
              test_bad_codebook_is_refused):
        t()
        print(f"PASS {t.__name__}")
