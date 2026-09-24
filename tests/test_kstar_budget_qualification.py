#!/usr/bin/env python3
"""CPU-only contracts for the isolated R9 K*-budget qualification runner."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "h0_measurement"))

from h0_measurement import run_kstar_budget_qualification as RUN  # noqa: E402
from sievelib import group_error_curve as GEC  # noqa: E402


def expect_error(action, error_type, contains: str) -> None:
    try:
        action()
        raise AssertionError("invalid contract was accepted")
    except error_type as caught:
        assert contains in str(caught), str(caught)


def test_frozen_grid_split_and_off_by_one_contract() -> None:
    assert RUN.CTX == 8192 and RUN.BUILDER_CTX == 9000
    assert RUN.PREFILL_TOKENS == 8191 and RUN.WINDOW == 32
    assert RUN.CONTEXT_TOKENS == 8159 and RUN.K0 == 3059
    assert RUN.CALIBRATION_PROMPT_IDS == (0, 1)
    assert RUN.HELDOUT_PROMPT_IDS == (2, 3)
    assert RUN.SPLITS == {
        "calibration": (0, 1), "cal_p0": (0,), "cal_p1": (1,)}
    assert RUN.CALIBRATION_UNIT_ORDER == tuple(
        (prompt, family) for prompt in (0, 1)
        for family in ("niah", "qa", "cont"))
    assert RUN.HELDOUT_UNIT_ORDER == tuple(
        (prompt, family) for prompt in (2, 3)
        for family in ("niah", "qa", "cont"))
    assert set(RUN.MODELS) == {"llama31-8b", "qwen15-moe-a2.7b"}
    assert RUN.INPUT_MANIFEST_VERSION == \
        "r9_kstar_budget_qualification_input_manifest_v1"
    assert "h0_measurement/bugs/10_kstar_budget/qualification_protocol_frozen.md" \
        in RUN.SOURCE_FILES
    assert "h0_measurement/bugs/10_kstar_budget/plan.md" not in RUN.SOURCE_FILES
    assert "h0_measurement/bugs/10_kstar_budget/report.md" not in RUN.SOURCE_FILES
    assert RUN.QUANTIZER_CACHE_RELATIVE in RUN.SOURCE_FILES
    for model in RUN.MODELS:
        cfg = RUN.experiment_config(model)
        assert cfg["ctx"] == 8192 and cfg["context_tokens"] == 8159
        assert cfg["B"] == 3 and cfg["maxb"] == 8 and cfg["k0"] == 3059
        assert cfg["execution"] == {
            "dtype": "bfloat16", "device_map": "auto", "chunk": 4096,
            "norm_correct": True, "rot_seed": 0,
        }


class _FakeTokenizer:
    is_fast = True

    def __call__(self, text, return_offsets_mapping=False, **_kwargs):
        # One BOS plus one token per character makes the crop boundary exact.
        ids = [1] + [2 + (i % 101) for i in range(len(text))]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(0, 0)] + [
                (i, i + 1) for i in range(len(text))]
        return out


def _real_meta(needle_start: int = 4000):
    return {
        "synthetic": False,
        "corpus_sha": "corpus-test",
        "doc": "book.txt",
        "offset": 17,
        "spliced": False,
        "needle_char_start": needle_start,
        "needle_char_end": needle_start + 12,
    }


def test_exact_prompt_uses_one_fixed_builder_keeps_bos_suffix_and_needle() -> None:
    original = RUN.prompts.build
    calls = []

    def fake_build(_tok, family, builder_ctx, **kwargs):
        calls.append((family, builder_ctx, kwargs["prompt_idx"]))
        return "x" * 9000, _real_meta()

    RUN.prompts.build = fake_build
    try:
        ids, provenance = RUN.build_exact_prompt(
            _FakeTokenizer(), "niah", 2, "/real/corpus")
    finally:
        RUN.prompts.build = original
    assert ids.shape == (1, RUN.CTX)
    assert calls == [("niah", RUN.BUILDER_CTX, 2)]
    assert int(ids[0, 0]) == 1
    assert provenance["pre_crop_tokens"] == 9001
    assert provenance["crop_offset"] == 9001 - RUN.CTX
    assert provenance["special_prefix_tokens"] == 1
    assert provenance["synthetic"] is False
    for field in ("special_prefix_sha256", "query_suffix_sha256",
                  "prompt_token_sha256"):
        assert len(provenance[field]) == 64

    def removes_needle(_tok, _family, builder_ctx, **_kwargs):
        assert builder_ctx == RUN.BUILDER_CTX
        return "x" * 9000, _real_meta(needle_start=10)

    RUN.prompts.build = removes_needle
    try:
        expect_error(
            lambda: RUN.build_exact_prompt(
                _FakeTokenizer(), "niah", 0, "/real/corpus"),
            RuntimeError, "remove NIAH needle")
    finally:
        RUN.prompts.build = original


def test_shared_haystack_audit_requires_all_four_prompt_blocks() -> None:
    records = []
    for p, family in RUN.UNIT_ORDER:
        records.append({
            "prompt_idx": p, "family": family, "builder_ctx": RUN.BUILDER_CTX,
            "corpus_sha": "sha", "corpus_doc": f"doc{p}",
            "corpus_offset": p * 7, "corpus_spliced": False,
        })
    RUN._assert_shared_haystack(records)
    bad = [dict(x) for x in records]
    bad[1]["corpus_offset"] += 1
    expect_error(lambda: RUN._assert_shared_haystack(bad), RuntimeError,
                 "corpus_offset")


def test_curve_scalars_scans_every_integer_and_accepts_nonmonotonicity() -> None:
    curve = torch.tensor([2.0, 1.05, 1.7, 1.0], dtype=torch.float64)
    old_tol, old_atol = RUN.KSTAR_TOLERANCE, RUN.KSTAR_ATOL
    try:
        RUN.KSTAR_TOLERANCE = 0.10
        RUN.KSTAR_ATOL = 0.0
        got = RUN.curve_scalars(curve, k0=4)
    finally:
        RUN.KSTAR_TOLERANCE, RUN.KSTAR_ATOL = old_tol, old_atol
    assert got["kstar"] == 2
    assert got["E_kstar"] == 1.05 and got["E_full"] == 1.0
    assert math.isclose(got["threshold"], 1.1)


def _synthetic_layer_ctx(repeats: int = 2) -> SimpleNamespace:
    gen = torch.Generator().manual_seed(19)
    groups, dim = 1, 4
    Cn, W = RUN.CONTEXT_TOKENS, RUN.WINDOW
    return SimpleNamespace(
        n_rep=repeats,
        scaling=dim ** -0.5,
        Kc=torch.randn(groups, Cn, dim, generator=gen),
        Vc=torch.randn(groups, Cn, dim, generator=gen),
        Kw=torch.randn(groups, W, dim, generator=gen),
        Vw=torch.randn(groups, W, dim, generator=gen),
        Kq={RUN.MAXB: torch.randn(groups, Cn, dim, generator=gen)},
        qwin=torch.randn(groups * repeats, W, dim, generator=gen),
        snap=torch.linspace(0.001, 1.0, Cn).view(1, Cn),
    )


def test_canonical_order_uses_pooled_score_but_n95_uses_unpooled_mass() -> None:
    Cn = RUN.CONTEXT_TOKENS
    pooled = torch.zeros(1, Cn)
    pooled[0, 17] = 10
    pooled[0, 3] = 9
    raw = torch.zeros(1, Cn)
    raw[0, :20] = 1
    order, n95 = RUN.canonical_order_n95_from_scores(pooled, raw)
    assert order[0, :2].tolist() == [17, 3]
    assert int(n95[0]) == 19
    pooled2 = pooled.roll(100, dims=-1)
    _, n95_2 = RUN.canonical_order_n95_from_scores(pooled2, raw)
    assert torch.equal(n95, n95_2)


def test_group_curve_and_k0_checkpoint_integration() -> None:
    ctx = _synthetic_layer_ctx()
    order, n95 = RUN.canonical_order_and_n95(ctx)
    assert order.shape == (1, RUN.CONTEXT_TOKENS)
    assert torch.equal(torch.sort(order[0]).values,
                       torch.arange(RUN.CONTEXT_TOKENS))
    assert n95.shape == (1,) and 1 <= int(n95[0]) <= RUN.CONTEXT_TOKENS
    curve = RUN.group_curve(ctx, 0, order[0], k0=7)
    full, qctx, values = RUN.group_inputs(ctx, 0)
    checkpoints = GEC.group_error_checkpoints(
        full, qctx, values, order[0], [0, 1, 7], eps=RUN.KSTAR_ATOL)
    assert curve.shape == (7,) and curve.dtype == torch.float64
    assert torch.allclose(checkpoints[1:], curve[[0, 6]], atol=1e-12, rtol=1e-12)
    assert bool(torch.isfinite(checkpoints).all()) and bool((checkpoints >= 0).all())


def test_independent_g1_reference_is_prefix_stable_and_honors_masks() -> None:
    # A much larger late-ranked logit made the old one-global-shift reference
    # underflow the first prefix.  One qctx entry is finite where the full mask
    # is -inf; the full-logit mask must win.
    Cn, tail, dim = 5, 2, 3
    full = torch.tensor([[[-1000.0, -999.0, -998.0, -997.0, -float("inf"),
                           -1001.0, -1002.0],
                          [-10.0, -9.0, -8.0, -7.0, -float("inf"), -11.0, -12.0]]],
                        dtype=torch.float64)
    qctx = torch.tensor([[[-1000.0, -999.0, -998.0, 1000.0, 5000.0],
                           [-10.0, -9.0, -8.0, 900.0, 7000.0]]],
                        dtype=torch.float64)
    values = torch.tensor([
        [1.0, -2.0, 0.5], [-3.0, 1.0, 2.0], [0.3, 0.7, -1.0],
        [2.0, 3.0, -4.0], [9.0, 9.0, 9.0], [0.2, -0.1, 1.0],
        [-0.3, 0.4, 0.8]], dtype=torch.float64)
    order = torch.tensor([0, 1, 2, 3, 4])
    direct = RUN.independent_single_head_curve(
        full, qctx, values, order, k_max=Cn)
    group = GEC.dense_group_error_curve(
        full, qctx, values, order, k_max=Cn, chunk_size=2,
        eps=RUN.KSTAR_ATOL)
    assert torch.isfinite(direct).all()
    assert torch.allclose(direct, group, atol=1e-10, rtol=1e-10), \
        float(torch.max(torch.abs(direct - group)))


def test_projected_policies_have_exact_budget_bounds_and_stable_keys() -> None:
    geometry = {"layers": 2, "kv_heads": 3}
    kstars = torch.tensor([[30, 100, 500], [1000, 2000, 3000]])
    n95 = torch.tensor([[10.5, 20.0, 40.0], [80.0, 160.0, 320.0]])
    counts = RUN.policy_counts(kstars, n95, geometry)
    target = kstars.numel() * RUN.K0
    for name in ("uniform", "prop", "shrink20", "n95"):
        assert counts[name].dtype == torch.long
        assert int(counts[name].sum()) == target
        assert bool((counts[name] >= 1).all())
        assert bool((counts[name] <= RUN.CONTEXT_TOKENS).all())
    assert bool((counts["uniform"] == RUN.K0).all())
    assert not torch.equal(counts["prop"], counts["uniform"])
    assert torch.equal(counts["prop"],
                       RUN.policy_counts(kstars, n95, geometry)["prop"])
    assert bool((counts["n95_raw"] >= 1).all())
    assert int(counts["n95_raw"].sum()) <= target
    summary = RUN.policy_summary(counts)
    for name in ("uniform", "prop", "shrink20", "n95"):
        assert summary[name]["budget_delta"] == 0
        assert 0.0 <= summary[name]["movement_fraction"] <= 1.0


def test_frozen_and_dynamic_policy_interfaces_allocate_once() -> None:
    geometry = {"layers": 2, "kv_heads": 3}
    kstars = torch.tensor([[30, 100, 500], [1000, 2000, 3000]])
    n95 = torch.tensor([[50, 100, 200], [300, 400, 500]])
    gen = torch.Generator().manual_seed(5)
    scores = {li: torch.rand(3, RUN.CONTEXT_TOKENS, generator=gen)
              for li in range(2)}
    frozen = RUN.freeze_calibrated_counts(kstars, n95, scores, geometry)
    dynamic = RUN.heldout_dynamic_counts(n95, scores, geometry)
    assert tuple(frozen) == RUN.FROZEN_POLICIES
    assert tuple(dynamic) == RUN.DYNAMIC_POLICIES
    target = 2 * 3 * RUN.K0
    for name, value in {**frozen, **dynamic}.items():
        assert value.shape == (2, 3) and value.dtype == torch.long
        if "natural" in name:
            assert int(value.sum()) <= target
        else:
            assert int(value.sum()) == target
        assert bool((value >= 0).all()) and bool((value <= RUN.CONTEXT_TOKENS).all())


def test_stability_summary_uses_calibration_prompt_zero_and_one() -> None:
    p0 = torch.tensor([[1, 2], [3, 4]])
    p1 = torch.tensor([[1, 3], [2, 4]])
    split_k = {"cal_p0": p0, "cal_p1": p1}
    c0 = {name: p0 + 100 for name in ("uniform", "prop", "shrink20", "n95")}
    c1 = {name: p1 + 100 for name in ("uniform", "prop", "shrink20", "n95")}
    out = RUN.stability_summary(split_k, {"cal_p0": c0, "cal_p1": c1})
    assert math.isclose(out["kstar_half_exact_fraction"], 0.5)
    assert out["kstar_half_median_abs_delta"] == 0.5
    assert out["prop_half_spearman"] is not None


def test_strict_json_and_input_schema_fail_closed_without_network() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "duplicate.json"
        path.write_text('{"a":1,"a":2}', encoding="utf-8")
        expect_error(lambda: RUN.strict_json_load(path), RuntimeError,
                     "duplicate JSON key")
    manifest = RUN.strict_json_load(ROOT / RUN.INPUT_MANIFEST_RELATIVE)
    assert manifest["content_sha256"] == RUN.INPUT_MANIFEST_CONTENT_SHA256
    for tag in RUN.MODELS:
        RUN._validate_manifest_snapshot_schema(
            tag, manifest["model_snapshots"][tag])
    broken = dict(manifest["model_snapshots"]["llama31-8b"])
    broken["extra"] = 1
    expect_error(
        lambda: RUN._validate_manifest_snapshot_schema("llama31-8b", broken),
        RuntimeError, "schema mismatch")


def test_registry_and_runtime_contracts_are_complete() -> None:
    good = {
        "id": RUN.MODELS["llama31-8b"]["id"], "native_ctx": 131072,
        "dtype": "bfloat16", "device_map": "auto", "chunk": 4096,
        "norm_correct": True, "rot_seed": 0, "maxb": 8,
    }
    original = RUN.run_h0.load_cfg
    RUN.run_h0.load_cfg = lambda *_args, **_kwargs: dict(good)
    try:
        assert RUN.validate_registry_config("unused", "llama31-8b") == good
        RUN.run_h0.load_cfg = lambda *_args, **_kwargs: {**good, "chunk": 2048}
        expect_error(
            lambda: RUN.validate_registry_config("unused", "llama31-8b"),
            RuntimeError, "chunk")
    finally:
        RUN.run_h0.load_cfg = original
    for package in ("pyyaml", "accelerate", "safetensors", "huggingface_hub"):
        assert package in RUN.EXPECTED_RUNTIME_VERSIONS


def test_quantizer_cache_and_serialization_boundaries() -> None:
    attestation = RUN.validate_quantizer_cache()
    assert attestation["file_sha256"] == RUN.QUANTIZER_CACHE_SHA256
    assert attestation["level_tensor_sha256"] == RUN.QUANTIZER_LEVEL8_SHA256
    assert attestation["levels"] == 256 and attestation["dtype"] == "float32"
    lowered = {x.lower() for x in (
        RUN.UNIT_COLUMNS + RUN.GROUP_COLUMNS + RUN.COUNT_COLUMNS +
        RUN.HELDOUT_COLUMNS)}
    forbidden = {"answer", "label", "correct", "prediction", "response",
                 "raw_logits", "raw_attention", "attention_matrix"}
    assert lowered.isdisjoint(forbidden)
    source = Path(RUN.__file__).read_text(encoding="utf-8")
    assert "tasks_longbench" not in source and "forced_choice" not in source


def test_geometry_contract_and_per_unit_g1_schema() -> None:
    cfg = SimpleNamespace(
        num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, hidden_size=4096, max_position_embeddings=131072,
        model_type="llama")
    assert RUN.validate_live_geometry(cfg, "llama31-8b") == \
        RUN.MODELS["llama31-8b"]["geometry"]
    cfg.num_key_value_heads = 7
    expect_error(lambda: RUN.validate_live_geometry(cfg, "llama31-8b"),
                 RuntimeError, "geometry mismatch")
    for field in ("g1_reference_available", "g1_max_abs", "g1_kstar",
                  "g1_kstar_match"):
        assert field in RUN.UNIT_COLUMNS and field in RUN.GROUP_COLUMNS


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} K* qualification runner tests")
