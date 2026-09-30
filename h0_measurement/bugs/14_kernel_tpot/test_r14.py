#!/usr/bin/env python3
"""CPU tests for R14 Stage 0 (plan.md §9). Run:

    OMP_NUM_THREADS=8 ../../../.venv/bin/python -u test_r14.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bytes_model as BM  # noqa: E402


def test_byte_identities():
    assert BM.token_bytes(16, 0, 0, 16) == 512          # BF16 KV
    assert BM.fp8_bytes(8) == 256 and BM.fp8_bytes(16) == 384
    # plan §3 hand-computed rows (Llama @128K)
    assert abs(BM.token_bytes(3, 16 / 128, 0, 8) - 178) < 0.01      # TurboQuant-3, V8
    assert abs(BM.token_bytes(2, 0.25, 0, 16) - 292) < 0.01         # KIVI-128-2, V16
    # one-width SIEVE with nothing evicted is dense TurboQuant
    for v in (8, 16):
        assert BM.token_bytes(3, 16 / 128, 0.0, v) == BM.token_bytes(3, 0.125, 0, v)
    # evicting a token removes its value bytes and nothing else
    assert np.isclose(BM.token_bytes(3, 0.1, 0.5, 16), BM.token_bytes(3, 0.1, 0, 16) - 128)


def test_side_rule_is_r12s():
    rows = [dict(arm="uniform", evict_frac=0.0, side_bits=np.nan),
            dict(arm="router_calib", evict_frac=0.49, side_bits=np.nan),
            dict(arm="obck_ada", evict_frac=0.63, side_bits=np.nan),
            dict(arm="kivi_g128", evict_frac=0.0, side_bits=0.25)]
    got = [BM.RMT.side_bits(pd.Series(r)) for r in rows]
    want = [16 / 128, 0.51 * 16 / 128 + 3 / 128, 0.37 * 16 / 128 + 1 / 128, 0.25]
    assert np.allclose(got, want)


def test_model_shapes_match_hf_configs():
    hub = os.path.join(BM.H0, "..", ".hf_cache", "hub")
    ids = {"llama31-8b": "meta-llama--Llama-3.1-8B-Instruct", "qwen3-8b": "Qwen--Qwen3-8B",
           "mistral-7b": "mistralai--Mistral-7B-Instruct-v0.3"}
    for tag, mid in ids.items():
        f = glob.glob(os.path.join(hub, f"models--{mid}", "snapshots", "*", "config.json"))
        if not f:
            print(f"  (skip {tag}: no cached config)")
            continue
        c = json.load(open(f[0]))
        m = BM.MODELS[tag]
        assert (c["num_hidden_layers"], c["num_key_value_heads"], c["hidden_size"],
                c["intermediate_size"], c["vocab_size"], c["num_attention_heads"]) == \
            (m["layers"], m["kv"], m["hidden"], m["inter"], m["vocab"], m["heads"]), tag
        hd = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
        assert hd == BM.D, tag


def test_weight_bytes():
    # Llama-3.1-8B: 8.03e9 params, minus a 0.525e9 embedding table, at 2 bytes
    assert abs(BM.weight_bytes("llama31-8b") / 1e9 - 15.01) < 0.05


def test_verdict_thresholds():
    assert BM.verdict(None) == "NO_POINT"
    assert BM.verdict(0.80) == "BYTES_WIN"
    assert BM.verdict(0.8001) == "BYTES_TIE" and BM.verdict(1.0) == "BYTES_TIE"
    assert BM.verdict(1.01) == "BYTES_LOSS"


def test_boot_weights_resample_within_job():
    idx = pd.MultiIndex.from_tuples([("a", i) for i in range(10)] + [("b", i) for i in range(5)],
                                    names=["job", "prompt_idx"])
    w = BM.boot_weights(idx, reps=500, seed=0)
    assert np.all(w[:, :10].sum(axis=1) == 10) and np.all(w[:, 10:].sum(axis=1) == 5)
    assert np.array_equal(w, BM.boot_weights(idx, reps=500, seed=0))


def _matrix(fp, arm):
    idx = pd.MultiIndex.from_tuples([("j", i) for i in range(len(fp))],
                                    names=["job", "prompt_idx"])
    cols = pd.MultiIndex.from_tuples([("fp", 0), ("x", 3)], names=["arm", "B"])
    return pd.DataFrame(np.c_[fp, arm], index=idx, columns=cols)


def test_lossless_rule():
    n = 40
    fp = np.ones(n)
    m = _matrix(fp, np.ones(n))                              # identical: lossless
    assert BM.lossless_table(m, BM.boot_weights(m.index))[("x", 3)]["lossless"]
    arm = np.ones(n); arm[:2] = 0.0                          # Δ = -0.05: fails point rule
    m = _matrix(fp, arm)
    assert not BM.lossless_table(m, BM.boot_weights(m.index))[("x", 3)]["lossless"]
    arm = np.ones(n); arm[:1] = 0.2                          # Δ = -0.02: point passes
    m = _matrix(fp, arm)
    r = BM.lossless_table(m, BM.boot_weights(m.index))[("x", 3)]
    assert abs(r["delta"] + 0.02) < 1e-12 and r["lossless"] == (r["lo"] >= -0.05)


def test_break_even_and_pareto():
    # k_s + (1 - f) 16 v = k_d + 16 v at v*
    k_s, f, k_d = 49.6, 0.49, 33.9
    v = BM.break_even_v(k_s, f, k_d)
    assert abs(k_s + (1 - f) * 16 * v - (k_d + 16 * v)) < 1e-9
    pts = [dict(label="a", bytes=100, score=0.9), dict(label="b", bytes=120, score=0.8),
           dict(label="c", bytes=150, score=0.95)]
    assert BM.pareto(pts) == ["a", "c"]


def test_decide_mapping():
    def res(unit, v16, v8):
        return dict(unit=unit, lenses={"V16": {"verdict": v16}, "V8": {"verdict": v8}})
    r = [res("llama31-8b@128K", "NO_POINT", "NO_POINT"),
         res("llama31-8b@32K", "BYTES_WIN", "BYTES_WIN"),
         res("qwen3-8b@32K", "BYTES_WIN", "BYTES_TIE")]
    d = BM.decide(r)
    assert d["stage0"] == "GO_KERNEL" and d["bytes_win_units"] == ["llama31-8b@32K"]
    assert d["units"]["qwen3-8b@32K"] == dict(verdict="BYTES_TIE", lens_dependent=True)
    assert d["stage1_needed"]
    assert BM.decide([res("llama31-8b@32K", "BYTES_TIE", "BYTES_TIE")])["stage0"] == "STOP_BYTES"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:          # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
