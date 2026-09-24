#!/usr/bin/env python3
"""Focused CPU contracts for the lock-authorized V6 Qwen development study."""
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

from h0_measurement import run_longbench_v2_qwen_development as RUN  # noqa: E402

READER_PATH = (
    ROOT
    / "h0_measurement/bugs/9_sota_eviction_baselines"
    / "read_longbench_v2_qwen_development.py"
)
SPEC = importlib.util.spec_from_file_location(
    "read_longbench_v2_qwen_development", READER_PATH
)
READER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(READER)

DATA = ROOT / ".h0_corpus/longbench_v2/data-2b48e494.json"
SOURCE_MANIFEST = (
    ROOT
    / "h0_measurement/bugs/9_sota_eviction_baselines"
    / "longbench_v2_manifest.json"
)
MANIFEST = (
    ROOT
    / "h0_measurement/bugs/9_sota_eviction_baselines"
    / "longbench_v2_qwen30_manifest.json"
)
CONFIG = ROOT / "h0_measurement/models.yaml"
QUALIFICATION_LOCK = (
    ROOT
    / "h0_measurement/bugs/9_sota_eviction_baselines"
    / "longbench_v2_v6_qualification_lock_984224.json"
)
SNAPSHOT = (
    ROOT
    / ".hf_cache/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots"
    / RUN.MODEL_REVISION
)


def expect_error(action, error_type, contains: str) -> None:
    try:
        action()
        assert False, "invalid contract was accepted"
    except error_type as caught:
        assert contains in str(caught), str(caught)


def test_exact_qwen_menu_schema_and_label_blind_runner():
    arms = (
        "fp", "uniform", "evict", "interior", "interior_pool",
        "interior_cascade", "obcache_k", "obck_ada", "laprox",
    )
    assert RUN.ARM_LABELS == arms
    assert RUN.CANDIDATES == arms[1:]
    assert READER.ARMS == arms
    assert READER.CANDIDATES == arms[1:]
    assert RUN.EXPECTED_ITEMS == 52
    assert RUN.EXPECTED_COMPONENTS == 44
    assert RUN.EXPECTED_PREDICTION_ROWS == 468
    assert RUN.EXPECTED_PROXY_ROWS == 416
    assert tuple(READER.PREDICTION_COLUMNS) == tuple(RUN.PREDICTION_COLUMNS)
    assert tuple(READER.PROXY_COLUMNS) == tuple(RUN.PROXY_COLUMNS)
    assert RUN.MODEL_ID == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert RUN.CTX == 40_960 and RUN.WINDOW == 32 and RUN.BUDGET == 2
    assert tuple(RUN.QA.SCAFFOLD_TOKEN_IDS) == (785, 4396, 4226, 374, 320)
    assert tuple(RUN.QA.CHOICE_BRANCH_IDS[x] for x in "ABCD") == (32, 33, 34, 35)
    assert not ({column.lower() for column in RUN.PREDICTION_COLUMNS} & READER.FORBIDDEN_RESULT_COLUMNS)
    assert not ({column.lower() for column in RUN.PROXY_COLUMNS} & READER.FORBIDDEN_RESULT_COLUMNS)
    source = Path(RUN.__file__).read_text(encoding="utf-8")
    assert 'item["answer"]' not in source and "item['answer']" not in source
    RUN.resolve_menu()  # Also enforces exact public labels and W-bounded scorers.


def test_qualification_lock_tamper_rejected_before_tokenizer_or_model_load():
    lock, attestation = RUN.authenticate_qualification_lock(
        QUALIFICATION_LOCK,
        dataset_path=DATA,
        source_manifest_path=SOURCE_MANIFEST,
        manifest_path=MANIFEST,
        config_path=CONFIG,
        model_source=SNAPSHOT,
    )
    assert lock["decision"] == "advance_v6_development"
    assert attestation["lock_content_sha256"] == RUN.QUALIFICATION_LOCK_CONTENT_SHA256
    assert attestation["lock_file_sha256"] == RUN.QUALIFICATION_LOCK_FILE_SHA256
    reader_lock, reader_path = READER.authenticate_qualification_lock(QUALIFICATION_LOCK)
    assert reader_lock["decision"] == "advance_v6_development"
    assert reader_path == QUALIFICATION_LOCK.resolve()

    tampered = copy.deepcopy(lock)
    tampered["decision"] = "advance_v6_development_tampered"
    tampered["content_sha256"] = RUN._content_sha256(tampered)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bad_lock = root / "tampered.json"
        bad_lock.write_text(
            json.dumps(tampered, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        calls = {"tokenizer": 0}
        original = RUN.QA.load_tokenizer

        def forbidden_tokenizer(*args, **kwargs):
            calls["tokenizer"] += 1
            raise AssertionError("tokenizer/model boundary was crossed")

        RUN.QA.load_tokenizer = forbidden_tokenizer
        try:
            args = SimpleNamespace(
                config=str(CONFIG),
                qualification_lock=str(bad_lock),
                dataset=str(DATA),
                source_manifest=str(SOURCE_MANIFEST),
                manifest=str(MANIFEST),
                model_source=str(SNAPSHOT),
                out_dir=str(root / "out"),
            )
            expect_error(
                lambda: RUN.run(args),
                RUN.QwenDevelopmentRunnerError,
                "qualification lock file hash",
            )
        finally:
            RUN.QA.load_tokenizer = original
        assert calls == {"tokenizer": 0}


def _allocation(fill: int = 2, *, changed: int = 0, replacement: int = 2):
    ctx_len = 1000
    by_layer = {
        layer: torch.full(
            (RUN.N_KV_HEADS, ctx_len), fill, dtype=torch.uint8
        )
        for layer in range(RUN.N_LAYERS)
    }
    if changed:
        by_layer[0].view(-1)[:changed] = replacement
    return ctx_len, by_layer


def test_feasible_sparse_underfill_uniform_exactness_and_overspend_rejection():
    # 48 * 4 * 1000 entries; changing 96 entries by one bit changes the
    # average by exactly 0.0005 bit/token.
    ctx_len, underfilled = _allocation(changed=96, replacement=1)
    _, audit = RUN.validate_allocation(
        underfilled, ctx_len=ctx_len, arm="evict"
    )
    assert math.isclose(audit["bits_per_token"], 1.9995, abs_tol=1e-12)

    expect_error(
        lambda: RUN.validate_allocation(
            underfilled, ctx_len=ctx_len, arm="uniform"
        ),
        RUN.QwenDevelopmentRunnerError,
        "uniform allocation is not exact all-2",
    )

    _, overspent = _allocation(changed=96, replacement=3)
    expect_error(
        lambda: RUN.validate_allocation(
            overspent, ctx_len=ctx_len, arm="evict"
        ),
        RUN.QwenDevelopmentRunnerError,
        "outside feasible",
    )

    _, uniform = _allocation()
    allocation_id, uniform_audit = RUN.validate_allocation(
        uniform, ctx_len=ctx_len, arm="uniform"
    )
    assert uniform_audit == {"bits_per_token": 2.0, "evict_frac": 0.0}
    assert allocation_id == RUN.QUAL.expected_uniform_allocation_id(ctx_len)


def test_proxy_and_fixed_policy_ties_follow_frozen_menu_order():
    reader_values = {candidate: 0.5 for candidate in READER.CANDIDATES}
    runner_values = {
        candidate: {"scaffold_mean_kl": 0.5}
        for candidate in RUN.CANDIDATES
    }
    assert READER.select_scaffold_policy(reader_values) == "uniform"
    assert RUN.select_scaffold_policy(runner_values) == "uniform"
    reader_values["evict"] = np.nextafter(0.5, 0.0)
    runner_values["evict"]["scaffold_mean_kl"] = np.nextafter(0.5, 0.0)
    assert READER.select_scaffold_policy(reader_values) == "evict"
    assert RUN.select_scaffold_policy(runner_values) == "evict"

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    development = [
        entry for entry in manifest["examples"]
        if entry["split"] == "development"
    ]
    item_ids = [str(entry["id"]) for entry in development]
    predictions = []
    for index, item_id in enumerate(item_ids):
        choice = "A" if index < 26 else "B"
        for arm in READER.ARMS:
            predictions.append({
                "item_id": item_id,
                "arm": arm,
                "forced_choice": choice,
            })
    proxy = pd.DataFrame({
        "item_id": item_ids,
        "selected_policy": ["uniform"] * len(item_ids),
    })
    data = {item_id: {"answer": "A"} for item_id in item_ids}
    _, fixed, accuracies = READER._scored_table(
        pd.DataFrame(predictions),
        proxy,
        development=development,
        data_by_id=data,
    )
    assert set(accuracies.values()) == {0.5}
    assert fixed == "uniform"


def _gate_table(*, opportunity: bool = True) -> tuple[pd.DataFrame, tuple[str, ...], tuple[frozenset[str], frozenset[str]]]:
    groups = tuple(f"g{index}" for index in range(52))
    half1, half2 = frozenset(groups[:28]), frozenset(groups[28:])
    fixed = np.zeros(52, dtype=bool)
    fixed[:14] = True
    fixed[28:40] = True
    oracle = fixed.copy()
    selector = fixed.copy()
    if opportunity:
        oracle[[14, 15, 16, 40, 41, 42]] = True
        selector[[14, 15, 40]] = True
    table = pd.DataFrame({
        "item_id": [f"i{index}" for index in range(52)],
        "group_id": groups,
        "fp": fixed,
        "fixed": fixed,
        "oracle": oracle,
        "selector": selector,
    })
    return table, groups, (half1, half2)


def test_sequential_competence_opportunity_and_proxy_gates_suppress_G():
    table, order, halves = _gate_table(opportunity=True)
    state = {
        "table": table,
        "bootstrap": {
            "fp": np.full(100, 0.40),
            "fixed": np.full(100, 0.40),
            "oracle": np.full(100, 0.52),
            "selector": np.full(100, 0.46),
            "H": np.full(100, 0.02),
            "G": np.zeros(100),
        },
    }
    candidates = {candidate: 0.5 for candidate in READER.CANDIDATES}
    originals = (
        READER.frozen_halves,
        READER._scored_table,
        READER.component_bootstrap,
    )
    READER.frozen_halves = lambda development: (order, halves)
    READER._scored_table = lambda *args, **kwargs: (
        state["table"], "uniform", candidates
    )
    READER.component_bootstrap = lambda *args, **kwargs: state["bootstrap"]
    development = [
        {"id": f"i{index}", "group_id": f"g{index}"}
        for index in range(52)
    ]
    predictions = pd.DataFrame([
        {"item_id": f"i{index}", "arm": arm, "forced_choice": "A"}
        for index in range(52)
        for arm in READER.ARMS
    ])
    proxy = pd.DataFrame(index=range(READER.EXPECTED_PROXY_ROWS))
    try:
        passed = READER.analyze_frames(
            predictions, proxy, development=development, data_by_id={}
        )
        assert passed["decision"] == "advance_v6_confirmation"
        assert math.isclose(passed["H"], 6 / 52)
        assert math.isclose(passed["G"], 3 / 52)
        assert math.isclose(passed["G_over_H"], 0.5)
        assert passed["G_bootstrap_q05"] == 0.0  # inclusive by protocol

        state["bootstrap"] = dict(state["bootstrap"])
        state["bootstrap"]["fp"] = np.full(100, 0.25)
        incompetent = READER.analyze_frames(
            predictions, proxy, development=development, data_by_id={}
        )
        assert incompetent["decision"] == "stop_v6_invalid_operating_point"
        assert incompetent["G"] is None
        assert incompetent["selector_accuracy"] is None
        assert incompetent["proxy_transfer_gate"] is None

        no_opportunity, _, _ = _gate_table(opportunity=False)
        state["table"] = no_opportunity
        state["bootstrap"] = {
            "fp": np.full(100, 0.40),
            "fixed": np.full(100, 0.40),
            "oracle": np.full(100, 0.40),
            "selector": np.full(100, 0.40),
            "H": np.zeros(100),
            "G": np.zeros(100),
        }
        stopped = READER.analyze_frames(
            predictions, proxy, development=development, data_by_id={}
        )
        assert stopped["decision"] == "stop_v6_no_opportunity"
        assert stopped["G"] is None and stopped["proxy_transfer_gate"] is None

        state["table"] = table
        state["bootstrap"] = {
            "fp": np.full(100, 0.40),
            "fixed": np.full(100, 0.40),
            "oracle": np.full(100, 0.52),
            "selector": np.full(100, 0.46),
            "H": np.full(100, 0.02),
            "G": np.full(100, -1e-3),
        }
        rejected = READER.analyze_frames(
            predictions, proxy, development=development, data_by_id={}
        )
        assert rejected["decision"] == "reject_v6_scaffold_proxy"
        assert rejected["G"] is not None
        assert rejected["G_bootstrap_gate"] is False
        assert rejected["proxy_transfer_gate"] is False
    finally:
        (
            READER.frozen_halves,
            READER._scored_table,
            READER.component_bootstrap,
        ) = originals


def test_confirmation_lock_is_idempotent_and_stale_safe():
    lock = {
        "lock_version": READER.CONFIRMATION_LOCK_VERSION,
        "decision": "advance_v6_confirmation",
        "fixed_policy": "uniform",
    }
    lock["content_sha256"] = READER._lock_content_hash(lock)
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "confirmation-lock.json"
        READER.write_confirmation_lock(destination, lock)
        READER.write_confirmation_lock(destination, lock)
        assert json.loads(destination.read_text(encoding="utf-8")) == lock

        changed = dict(lock, fixed_policy="evict")
        changed["content_sha256"] = READER._lock_content_hash(changed)
        expect_error(
            lambda: READER.write_confirmation_lock(destination, changed),
            READER.LongBenchForcedChoiceError,
            "different confirmation lock",
        )
        expect_error(
            lambda: READER.write_confirmation_lock(destination, None),
            READER.LongBenchForcedChoiceError,
            "stale confirmation lock",
        )


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS: {len(tests)} V6 Qwen development tests")
