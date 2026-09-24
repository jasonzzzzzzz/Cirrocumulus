#!/usr/bin/env python3
"""Focused CPU contracts for V7 structured-query development."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "h0_measurement"))
from h0_measurement import run_longbench_v2_qwen_v7_development as RUN  # noqa: E402
from sievelib import tasks_longbench_v2 as LB  # noqa: E402

READER_PATH = ROOT / "h0_measurement/bugs/9_sota_eviction_baselines/read_longbench_v2_qwen_v7_development.py"
SPEC = importlib.util.spec_from_file_location("read_longbench_v2_qwen_v7_development", READER_PATH)
READER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(READER)
DATA = ROOT / ".h0_corpus/longbench_v2/data-2b48e494.json"
LEGACY_MANIFEST = ROOT / "h0_measurement/bugs/9_sota_eviction_baselines/longbench_v2_qwen30_manifest.json"
MANIFEST = ROOT / "h0_measurement/bugs/9_sota_eviction_baselines/longbench_v2_qwen30_v7_manifest.json"
CONFIG = ROOT / "h0_measurement/models.yaml"
SNAPSHOT = ROOT / ".hf_cache/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots" / RUN.MODEL_REVISION


def expect_error(action, error_type, contains: str) -> None:
    try:
        action()
        assert False, "invalid contract was accepted"
    except error_type as caught:
        assert contains in str(caught), str(caught)


def manifest_object():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_exact_menu_scalar_schema_canonical_order_and_label_blind_source():
    assert RUN.ARMS == (
        "fp", "uniform", "tail_evict", "tail_interior",
        "structured_evict", "structured_interior",
    )
    assert RUN.CANDIDATES == RUN.ARMS[1:]
    assert RUN.OLD_MENU == ("uniform", "tail_evict", "tail_interior")
    assert RUN.STRUCTURED_MENU == ("structured_evict", "structured_interior")
    assert READER.ARMS == RUN.ARMS and READER.CANDIDATES == RUN.CANDIDATES
    assert RUN.EXPECTED_ITEMS == RUN.EXPECTED_COMPONENTS == 52
    assert RUN.EXPECTED_PREDICTION_ROWS == 312
    assert "observed_queries" not in RUN.PREDICTION_COLUMNS
    assert {"score_query_source", "score_query_count"} <= set(RUN.PREDICTION_COLUMNS)
    assert RUN.SCORE_QUERY_SOURCES == {
        "fp": "none", "uniform": "none",
        "tail_evict": "tail_32", "tail_interior": "tail_32",
        "structured_evict": "structured_question_choices",
        "structured_interior": "structured_question_choices",
    }
    assert RUN.CTX == 131_072 and RUN.WINDOW == 32 and RUN.BUDGET == 2
    assert RUN.PARQUET_ENGINE == "fastparquet"
    assert RUN.NUMPY_VERSION == RUN.QUAL.NUMPY_VERSION == RUN.np.__version__
    assert RUN.PANDAS_VERSION == RUN.QUAL.PANDAS_VERSION == RUN.pd.__version__
    assert RUN.PYARROW_VERSION == RUN.QUAL.PYARROW_VERSION == "not_installed"
    assert RUN.TORCH_VERSION == RUN.QUAL.TORCH_VERSION == RUN.torch.__version__
    assert RUN.TRANSFORMERS_VERSION == RUN.QUAL.TRANSFORMERS_VERSION
    assert RUN.TOKENIZERS_VERSION == RUN.QUAL.TOKENIZERS_VERSION
    assert RUN.FASTPARQUET_VERSION == RUN.QUAL.FASTPARQUET_VERSION
    assert RUN.FASTPARQUET_VERSION != "not_installed"
    assert {"numpy_version", "pandas_version", "pyarrow_version", "fastparquet_version", "parquet_engine"} <= READER.SIDECAR_FIELDS
    assert "engine=PARQUET_ENGINE" in Path(RUN.__file__).read_text(encoding="utf-8")
    assert "engine=RUN.PARQUET_ENGINE" in READER_PATH.read_text(encoding="utf-8")
    assert RUN.development_execution()["attn_impl"] == RUN.SP.IMPL
    assert RUN.development_execution()["no_proxy"] is True
    assert not ({column.lower() for column in RUN.PREDICTION_COLUMNS} & RUN.FORBIDDEN_RESULT_COLUMNS)
    source = Path(RUN.__file__).read_text(encoding="utf-8")
    assert 'item["answer"]' not in source and "item['answer']" not in source
    manifest = manifest_object()
    entries = RUN.canonical_entries(manifest)
    assert [row["id"] for row in entries] == RUN.QA.partition_order(entries)
    order, halves = READER.frozen_halves(entries)
    assert len(order) == 52 and [len(half) for half in halves] == [26, 26]
    assert all(len({row["group_id"] for row in entries}) == 52 for _ in [0])


def test_manifest_query_groups_are_exact_five_group_absolute_plan():
    manifest = manifest_object()
    for entry in RUN.canonical_entries(manifest):
        groups = RUN.query_groups(entry, int(entry["input_tokens"]) - 1)
        assert len(groups) == 5
        assert all(1 <= len(group) <= 8 for group in groups)
        assert len({position for group in groups for position in group}) == sum(map(len, groups))
        assert max(position for group in groups for position in group) < int(entry["input_tokens"]) - 1
    bad = copy.deepcopy(RUN.canonical_entries(manifest)[0])
    bad["content_token_positions"]["question"] = [bad["input_tokens"]]
    expect_error(
        lambda: RUN.query_groups(bad, bad["input_tokens"] - 1),
        RUN.QwenV7DevelopmentRunnerError,
        "invalid structured query groups",
    )


def test_all_five_allocations_share_one_layer_context_and_structured_sig2():
    originals = {
        "n_layers": RUN.N_LAYERS, "n_h": RUN.N_ATTENTION_HEADS,
        "n_kv": RUN.N_KV_HEADS, "crop": RUN.C.crop_to,
        "build": RUN.router.build_layer_ctx, "base": RUN.router.base_bits,
        "sev": RUN.SP.allocate_structured_evict,
        "sint": RUN.SP.allocate_structured_interior,
        "ctx_len": RUN.C.STATE.ctx_len,
    }
    RUN.N_LAYERS, RUN.N_ATTENTION_HEADS, RUN.N_KV_HEADS = 2, 4, 2
    RUN.C.STATE.ctx_len = 3
    calls, interior_sig2 = [], []
    class Ctx:
        n_rep = 2
        sig2 = [{1: 1.0}] * 4
    def build(layer, *args, **kwargs):
        calls.append(layer)
        return Ctx()
    def base(arm, budget, ctx, maxb):
        fill = {"uniform": 2, "evict": 0, "interior": 1}[arm]
        return torch.full((2, 3), fill, dtype=torch.long)
    def structured_evict(score, budget, maxb):
        return torch.full((2, 3), 2, dtype=torch.long)
    def structured_interior(ctx, score, budget, maxb):
        interior_sig2.append(ctx.sig2)
        return torch.full((2, 3), 1, dtype=torch.long)
    RUN.C.crop_to = lambda past, length: None
    RUN.router.build_layer_ctx = build
    RUN.router.base_bits = base
    RUN.SP.allocate_structured_evict = structured_evict
    RUN.SP.allocate_structured_interior = structured_interior
    score_h = {layer: torch.ones(4, 3) for layer in range(2)}
    score_kv = {layer: torch.full((2, 3), 2.0) for layer in range(2)}
    try:
        got = RUN.build_allocations(
            object(), cache_length=5, rotation=torch.eye(1),
            structured_score_h=score_h, structured_score_kv=score_kv,
        )
    finally:
        RUN.N_LAYERS = originals["n_layers"]
        RUN.N_ATTENTION_HEADS = originals["n_h"]
        RUN.N_KV_HEADS = originals["n_kv"]
        RUN.C.crop_to = originals["crop"]
        RUN.router.build_layer_ctx = originals["build"]
        RUN.router.base_bits = originals["base"]
        RUN.SP.allocate_structured_evict = originals["sev"]
        RUN.SP.allocate_structured_interior = originals["sint"]
        RUN.C.STATE.ctx_len = originals["ctx_len"]
    assert calls == [0, 1]
    assert len(interior_sig2) == 2 and all(value is Ctx.sig2 for value in interior_sig2)
    assert set(got) == set(RUN.CANDIDATES)
    assert all(set(layers) == {0, 1} for layers in got.values())
    assert all(bits.dtype == torch.uint8 for layers in got.values() for bits in layers.values())


def _allocation(fill: int = 2, replacement: int | None = None):
    by_layer = {
        layer: torch.full((RUN.N_KV_HEADS, 17), fill, dtype=torch.uint8)
        for layer in range(RUN.N_LAYERS)
    }
    if replacement is not None:
        by_layer[0][0, 0] = replacement
    return by_layer


def test_exact_uniform_and_feasible_allocation_contracts():
    uniform = _allocation()
    allocation_id, audit = RUN.validate_allocation(uniform, ctx_len=17, arm="uniform")
    assert allocation_id == RUN.QUAL.expected_uniform_allocation_id(17)
    assert audit == {"bits_per_token": 2.0, "evict_frac": 0.0}
    expect_error(
        lambda: RUN.validate_allocation(_allocation(replacement=1), ctx_len=17, arm="uniform"),
        RUN.QwenV7DevelopmentRunnerError, "exact all-2",
    )
    expect_error(
        lambda: RUN.validate_allocation(_allocation(replacement=3), ctx_len=17, arm="tail_interior"),
        RUN.QwenV7DevelopmentRunnerError, "outside",
    )
    sparse = _allocation(fill=0)
    _, sparse_audit = RUN.validate_allocation(sparse, ctx_len=17, arm="tail_evict")
    assert sparse_audit == {"bits_per_token": 0.0, "evict_frac": 1.0}


def _synthetic_analysis_entries():
    return [
        {"id": f"synthetic-item-{index:02d}", "group_id": f"synthetic-group-{index:02d}"}
        for index in range(52)
    ]


def _analysis_frame(mode: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    order = [str(row["id"]) for row in _synthetic_analysis_entries()]
    choices = []
    uniform = set(range(20))
    if mode == "advance":
        tail_evict, tail_interior = set(range(18)), set(range(2, 20))
        structured_evict = (uniform - {0, 1, 2}) | {20, 21, 46}
        structured_interior = (uniform - {3, 4, 5}) | {22, 47, 48}
    elif mode == "no_mechanism":
        tail_evict = (uniform - set(range(6))) | {20, 21, 22, 46, 47, 48}
        tail_interior = set(uniform)
        structured_evict = set(tail_evict)
        structured_interior = set(uniform)
    elif mode == "no_opportunity":
        tail_evict, tail_interior = set(uniform), set(uniform)
        structured_evict = (uniform - {0, 1, 2}) | {20, 21, 46}
        structured_interior = set(uniform)
    else:
        raise AssertionError(mode)
    sets = {
        "fp": set(range(30)), "uniform": uniform,
        "tail_evict": tail_evict, "tail_interior": tail_interior,
        "structured_evict": structured_evict,
        "structured_interior": structured_interior,
    }
    for index, item_id in enumerate(order):
        gold, wrong = "A", "B"
        for arm in RUN.ARMS:
            choices.append({
                "item_id": item_id, "arm": arm,
                "forced_choice": gold if index in sets[arm] else wrong,
                "allocation_id": f"{RUN.ARMS.index(arm) + 1:064x}",
                "bits_per_token": 16.0 if arm == "fp" else 2.0,
                "evict_frac": 0.0 if arm in {"fp", "uniform"} else 0.5,
            })
    bootstrap = {
        "fp": np.full(100, 0.40), "F_star": np.full(100, 0.35),
        "all_oracle": np.full(100, 0.5), "old_oracle": np.full(100, 0.4),
        "H": np.full(100, 0.04), "S": np.full(100, 0.03),
    }
    return pd.DataFrame(choices), bootstrap


def test_f_star_tie_order_and_sequential_S_then_H_decisions():
    manifest = {"synthetic": True}
    entries = _synthetic_analysis_entries()
    order = tuple(str(entry["id"]) for entry in entries)
    halves = (frozenset(order[:26]), frozenset(order[26:]))
    original = READER.component_bootstrap
    original_load_dataset = READER.LB.load_dataset
    original_entries = READER.canonical_entries
    original_halves = READER.frozen_halves
    synthetic_dataset = [
        {"_id": str(entry["id"]), "answer": "A"} for entry in entries
    ]
    READER.LB.load_dataset = lambda *args, **kwargs: synthetic_dataset
    READER.canonical_entries = lambda ignored: entries
    READER.frozen_halves = lambda ignored: (order, halves)
    try:
        frame, bootstrap = _analysis_frame("advance")
        READER.component_bootstrap = lambda *args, **kwargs: bootstrap
        summary = READER.analyze(frame, manifest=manifest, dataset_path=DATA)
        assert summary["F_star_policy"] == "uniform"
        assert math.isclose(summary["H"], 6 / 52)
        assert math.isclose(summary["S"], 6 / 52)
        assert summary["H_half1"] >= 3 / 26 and summary["H_half2"] >= 3 / 26
        assert summary["decision"] == "advance_v7_confirmation"
        assert "G" not in summary and "proxy" not in "".join(summary)

        frame, bootstrap = _analysis_frame("no_mechanism")
        READER.component_bootstrap = lambda *args, **kwargs: bootstrap
        no_mechanism = READER.analyze(frame, manifest=manifest, dataset_path=DATA)
        assert no_mechanism["S"] == 0.0
        assert no_mechanism["H"] >= 0.10
        assert no_mechanism["decision"] == "stop_v7_no_mechanism_effect"

        frame, bootstrap = _analysis_frame("no_opportunity")
        READER.component_bootstrap = lambda *args, **kwargs: bootstrap
        no_opportunity = READER.analyze(frame, manifest=manifest, dataset_path=DATA)
        assert no_opportunity["S"] >= 0.05 and no_opportunity["H"] < 0.10
        assert no_opportunity["decision"] == "stop_v7_no_opportunity"

        bad_bootstrap = dict(bootstrap, fp=np.full(100, 0.25))
        READER.component_bootstrap = lambda *args, **kwargs: bad_bootstrap
        invalid = READER.analyze(frame, manifest=manifest, dataset_path=DATA)
        assert invalid["decision"] == "stop_v7_invalid_operating_point"
    finally:
        READER.component_bootstrap = original
        READER.LB.load_dataset = original_load_dataset
        READER.canonical_entries = original_entries
        READER.frozen_halves = original_halves


def test_singleton_PCG64_bootstrap_is_deterministic():
    entries = RUN.canonical_entries(manifest_object())
    order = [str(row["group_id"]) for row in entries]
    outcomes = pd.DataFrame({
        "item_id": [str(row["id"]) for row in entries],
        "group_id": order, "x": np.arange(52) % 2,
    })
    first = READER.component_bootstrap(outcomes, order, draws=100, seed=0)["x"]
    second = READER.component_bootstrap(outcomes, order, draws=100, seed=0)["x"]
    assert np.array_equal(first, second)
    indices = np.random.Generator(np.random.PCG64(0)).integers(0, 52, size=(100, 52))
    expected = outcomes["x"].to_numpy(dtype=float)[indices].mean(axis=1)
    assert np.array_equal(first, expected)


def test_reviewed_full_repo_local_import_closure_is_sealed():
    observed = {
        str(path.resolve().relative_to(ROOT.resolve()))
        for path in RUN.sealed_source_paths(CONFIG)
    }
    expected = {
        "h0_measurement/run_longbench_v2_qwen_v7_development.py",
        "h0_measurement/audit_longbench_v2_qwen_v7.py",
        "h0_measurement/run_longbench_v2_qwen_v7_qualification.py",
        "h0_measurement/run_h0.py", "h0_measurement/run_r8.py",
        "h0_measurement/models.yaml",
        "h0_measurement/bugs/9_sota_eviction_baselines/read_longbench_v2_qwen_v7_development.py",
        "h0_measurement/submit_longbench_v2_qwen_v7_development.slurm",
        *RUN.QUAL.SEALED_SIEVELIB_RELATIVE_PATHS,
    }
    assert observed == expected
    assert len(RUN.QUAL.SEALED_SIEVELIB_RELATIVE_PATHS) == 16
    assert not any(path.startswith("tests/") for path in observed)


def test_development_source_ledger_interface_and_default_path_contract():
    assert RUN.DEFAULT_SOURCE_LEDGER.name == "longbench_v2_qwen_v7_development_source_ledger.json"
    ledger = RUN.source_ledger_object(CONFIG)
    with tempfile.TemporaryDirectory(dir=ROOT / "h0_measurement/results_smoke") as directory:
        path = Path(directory) / "development-ledger.json"
        path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        hashes = RUN.executed_source_hashes(CONFIG, path)
        attestation = RUN.verify_source_ledger(path, hashes)
        assert attestation["content_sha256"] == ledger["content_sha256"]
        assert hashes[attestation["relative_path"]] == LB.sha256_file(path)
        bad = copy.deepcopy(ledger)
        bad["source_sha256"][next(iter(bad["source_sha256"]))] = "0" * 64
        bad["content_sha256"] = RUN.source_ledger_content_sha256(bad)
        path.write_text(json.dumps(bad, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        hashes[attestation["relative_path"]] = LB.sha256_file(path)
        expect_error(
            lambda: RUN.verify_source_ledger(path, hashes),
            RUN.QwenV7DevelopmentRunnerError, "mapping",
        )


def test_nonadvancing_lock_rejected_before_tokenizer_or_model_boundary():
    value = {
        "lock_version": RUN.QUALIFICATION_LOCK_VERSION,
        "protocol_version": RUN.QUAL.PROTOCOL_VERSION,
        "runner_version": RUN.QUAL.RUNNER_VERSION,
        "reader_version": RUN.QUALIFICATION_READER_VERSION,
        "task_version": RUN.TASK_VERSION, "endpoint_version": RUN.ENDPOINT_VERSION,
        "decision": "stop_v7_qualification", "qualification": {},
        "provenance": {}, "partitions": {}, "execution": RUN._expected_execution(),
        "thresholds": RUN.QUALIFICATION_THRESHOLDS,
    }
    value["content_sha256"] = RUN._content_sha256(value)
    with tempfile.TemporaryDirectory() as directory:
        lock = Path(directory) / "stop.json"
        lock.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        calls = {"tokenizer": 0}
        original = RUN.QA.load_tokenizer
        RUN.QA.load_tokenizer = lambda *args, **kwargs: calls.__setitem__("tokenizer", calls["tokenizer"] + 1)
        try:
            args = SimpleNamespace(
                config=str(CONFIG), source_ledger=str(RUN.DEFAULT_SOURCE_LEDGER),
                qualification_lock=str(lock), dataset=str(DATA),
                legacy_manifest=str(LEGACY_MANIFEST), manifest=str(MANIFEST),
                model_source=str(SNAPSHOT), out_dir=str(Path(directory) / "out"),
            )
            expect_error(lambda: RUN.run(args), RUN.QwenV7DevelopmentRunnerError, "lock decision")
        finally:
            RUN.QA.load_tokenizer = original
        assert calls == {"tokenizer": 0}


def test_confirmation_lock_is_advance_only_idempotent_and_stale_safe():
    lock = {
        "lock_version": READER.CONFIRMATION_LOCK_VERSION,
        "decision": "advance_v7_confirmation", "F_star_policy": "uniform",
    }
    lock["content_sha256"] = READER._content_sha256(lock)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "confirmation-lock.json"
        READER.write_confirmation_lock(path, lock)
        READER.write_confirmation_lock(path, lock)
        assert json.loads(path.read_text()) == lock
        changed = dict(lock, F_star_policy="structured_evict")
        changed["content_sha256"] = READER._content_sha256(changed)
        expect_error(
            lambda: READER.write_confirmation_lock(path, changed),
            READER.QwenV7DevelopmentReaderError, "different existing",
        )
        expect_error(
            lambda: READER.write_confirmation_lock(path, None),
            READER.QwenV7DevelopmentReaderError, "stale confirmation",
        )


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS: {len(tests)} V7 development tests")
