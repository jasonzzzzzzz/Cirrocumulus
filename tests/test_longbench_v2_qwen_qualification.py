#!/usr/bin/env python3
"""Focused CPU contracts for V6 Qwen manifest, qualification, and gates."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
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
from h0_measurement import audit_longbench_v2_qwen as QA  # noqa: E402
from h0_measurement import run_longbench_v2_qwen_qualification as RUN  # noqa: E402
from sievelib import forced_choice as FC  # noqa: E402
from sievelib import policy_diagnostic as PD  # noqa: E402
from sievelib import tasks_longbench_v2 as LB  # noqa: E402

READER_PATH = ROOT / "h0_measurement/bugs/9_sota_eviction_baselines/read_longbench_v2_qwen_qualification.py"
SPEC = importlib.util.spec_from_file_location("read_longbench_v2_qwen_qualification", READER_PATH)
READER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(READER)
DATA = ROOT / ".h0_corpus/longbench_v2/data-2b48e494.json"
SOURCE_MANIFEST = ROOT / "h0_measurement/bugs/9_sota_eviction_baselines/longbench_v2_manifest.json"
MANIFEST = ROOT / "h0_measurement/bugs/9_sota_eviction_baselines/longbench_v2_qwen30_manifest.json"
SNAPSHOT = ROOT / ".hf_cache/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots" / QA.MODEL_REVISION


def expect_error(action, error_type, contains: str) -> None:
    try:
        action()
        assert False, "invalid contract was accepted"
    except error_type as caught:
        assert contains in str(caught), str(caught)


def test_frozen_manifest_and_actual_qwen_tokenizer_authenticate():
    assert LB.sha256_file(MANIFEST) == QA.FROZEN_MANIFEST_FILE_SHA256
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["content_sha256"] == QA.FROZEN_MANIFEST_CONTENT_SHA256
    assert QA.manifest_content_sha256(manifest) == manifest["content_sha256"]
    assert '"answer"' not in MANIFEST.read_text()
    tokenizer = QA.load_tokenizer(str(SNAPSHOT))
    authenticated, file_hash = QA.authenticate_manifest(
        MANIFEST,
        dataset_path=DATA,
        source_manifest_path=SOURCE_MANIFEST,
        tokenizer=tokenizer,
    )
    assert file_hash == QA.FROZEN_MANIFEST_FILE_SHA256
    assert authenticated["split_counts"]["qualification"]["id_prompt_sha256"] == READER.QUALIFICATION_ID_PROMPT_SHA256
    for split, expected in QA.EXPECTED_SPLITS.items():
        got = authenticated["split_counts"][split]
        for field, value in expected.items():
            assert got[field] == value


def test_render_passes_explicit_no_thinking_flag():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "prompt"}]
            assert kwargs == {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return {"input_ids": [[1, 2, 3]]}
    assert FC.render_user_prompt(Tokenizer(), "prompt", enable_thinking=False) == [1, 2, 3]


def test_choice_summary_tie_order_and_scalar_only_schema():
    summary = FC.choice_summary(torch.ones(4))
    assert summary["forced_choice"] == "A"
    assert summary["choice_index"] == 0
    assert summary["choice_max_probability"] == 0.25
    assert summary["choice_margin"] == 0.0
    assert math.isclose(summary["choice_entropy"], math.log(4), rel_tol=1e-6)
    lowered = {column.lower() for column in RUN.PREDICTION_COLUMNS}
    assert not (READER.FORBIDDEN_RESULT_COLUMNS & lowered)
    source = Path(RUN.__file__).read_text()
    assert 'item["answer"]' not in source and "item['answer']" not in source
    assert RUN.ARMS == ("fp", "uniform")


def test_policy_forward_restores_cache_applies_bits_and_uses_position_five():
    class Past:
        length = 10
    class Model:
        def __init__(self):
            self.inputs = []
        def __call__(self, input_ids, *, past_key_values, use_cache):
            assert use_cache is True
            self.inputs.append(input_ids.tolist())
            logits = torch.zeros(1, 6, 64)
            logits[0, :5, :] = 1000.0
            logits[0, 5, [32, 33, 34, 35]] = torch.tensor([-2.0, -1.0, 8.0, 1.0])
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)
    model, cropped, applied = Model(), [], []
    originals = (FC.C.crop_to, FC.C.apply_bits, FC.C.bits_audit)
    FC.C.crop_to = lambda past, length: cropped.append(length)
    def apply_bits(past, bits, rotation, norm_correct):
        applied.append((sorted(bits), norm_correct))
        FC.C.STATE.enabled = True
    FC.C.apply_bits = apply_bits
    FC.C.bits_audit = lambda: {"bits_per_token": 2.0, "evict_frac": 0.0}
    try:
        logits, audit = FC.policy_choice_logits(
            model,
            Past(),
            last_prompt_token=9,
            prefix_tokens=QA.SCAFFOLD_TOKEN_IDS,
            branch_ids=QA.CHOICE_BRANCH_IDS,
            bits_by_layer={0: torch.ones(1, 10, dtype=torch.long)},
            rotation=torch.eye(1),
            norm_correct=True,
            cache_length=10,
            device="cpu",
        )
    finally:
        FC.C.crop_to, FC.C.apply_bits, FC.C.bits_audit = originals
        FC.C.STATE.enabled = False
    assert cropped == [10]
    assert applied == [([0], True)]
    assert model.inputs == [[[9, *QA.SCAFFOLD_TOKEN_IDS]]]
    assert FC.choice_summary(logits)["forced_choice"] == "C"
    assert audit == {"bits_per_token": 2.0, "evict_frac": 0.0}
    assert FC.C.STATE.enabled is False


def _manifest_and_rows():
    manifest = json.loads(MANIFEST.read_text())
    file_hash = LB.sha256_file(MANIFEST)
    rows = []
    entries = [row for row in manifest["examples"] if row["split"] == RUN.SPLIT]
    for entry in entries:
        for arm_order, arm in enumerate(RUN.ARMS):
            values = {
                "item_id": entry["id"], "group_id": entry["group_id"],
                "split": RUN.SPLIT, **entry["metadata"],
                "context_hash": entry["context_hash"],
                "question_hash": entry["question_hash"],
                "prompt_token_hash": entry["prompt_token_hash"],
                "input_tokens": entry["input_tokens"],
                "n_prompt_tokens": entry["input_tokens"], "truncated": False,
                "dataset_sha256": LB.DATASET_SHA256,
                "manifest_content_sha256": manifest["content_sha256"],
                "manifest_file_sha256": file_hash,
                "task_version": RUN.TASK_VERSION,
                "endpoint_version": RUN.ENDPOINT_VERSION,
                "model": RUN.MODEL_TAG, "model_id": RUN.MODEL_ID,
                "model_revision": RUN.MODEL_REVISION,
                "tokenizer_revision": RUN.TOKENIZER_REVISION,
                "ctx": RUN.CTX, "window": RUN.WINDOW,
                "B": 0 if arm == "fp" else RUN.BUDGET,
                "arm": arm, "arm_order": arm_order,
                "forced_choice": "A", "choice_index": 0,
                "choice_entropy": 1.0, "choice_margin": 0.1,
                "choice_max_probability": 0.4,
                "scaffold_token_hash": PD.token_hash(QA.SCAFFOLD_TOKEN_IDS),
                "choice_branch_ids_hash": PD.token_hash([QA.CHOICE_BRANCH_IDS[x] for x in "ABCD"]),
                "bits_per_token": 16.0 if arm == "fp" else 2.0,
                "evict_frac": 0.0,
                "allocation_id": READER.FP_ALLOCATION_ID if arm == "fp" else RUN.expected_uniform_allocation_id(entry["input_tokens"] - 1 - RUN.WINDOW),
                "ctx_len": entry["input_tokens"] - 1 - RUN.WINDOW,
                "observed_queries": RUN.WINDOW, "maxb": RUN.MAXB,
                "rot_seed": RUN.ROT_SEED, "norm_correct": True,
            }
            rows.append({column: values[column] for column in RUN.PREDICTION_COLUMNS})
    return manifest, file_hash, rows


def test_exact_40_row_schema_budget_and_tamper_rejection():
    manifest, file_hash, rows = _manifest_and_rows()
    assert len(rows) == 40
    item_ids = [row["id"] for row in manifest["examples"] if row["split"] == RUN.SPLIT]
    RUN.validate_output_rows(rows, item_ids)
    frame = pd.DataFrame(rows, columns=RUN.PREDICTION_COLUMNS)
    READER._validate_rows(frame, manifest=manifest, manifest_file_hash=file_hash)
    bad = frame.copy(deep=True)
    bad.loc[bad["arm"].eq("uniform"), "bits_per_token"] = 1.999
    expect_error(
        lambda: READER._validate_rows(bad, manifest=manifest, manifest_file_hash=file_hash),
        READER.QwenQualificationReaderError,
        "underspend",
    )
    bad = frame.copy(deep=True)
    bad["answer"] = "A"
    expect_error(
        lambda: READER._validate_rows(bad, manifest=manifest, manifest_file_hash=file_hash),
        READER.QwenQualificationReaderError,
        "schema drifted",
    )


def test_component_bootstrap_is_deterministic_and_carries_linked_rows():
    manifest = json.loads(MANIFEST.read_text())
    entries = [row for row in manifest["examples"] if row["split"] == RUN.SPLIT]
    order = READER._ordered_components(entries)
    assert len(order) == 19
    sizes = {group: sum(row["group_id"] == group for row in entries) for group in order}
    assert sorted(sizes.values()) == [1] * 18 + [2]
    outcomes = pd.DataFrame({
        "item_id": [row["id"] for row in entries],
        "group_id": [row["group_id"] for row in entries],
        "fp": [index % 2 == 0 for index in range(20)],
        "uniform": [index % 3 == 0 for index in range(20)],
    })
    got = READER._component_bootstrap(outcomes, order, draws=100, seed=0)
    again = READER._component_bootstrap(outcomes, order, draws=100, seed=0)
    assert np.array_equal(got["fp"], again["fp"])
    grouped = outcomes.groupby("group_id", sort=False)
    component_sizes = grouped.size().reindex(order).to_numpy(dtype=np.int64)
    sums = grouped["fp"].sum().reindex(order).to_numpy(dtype=float)
    indices = np.random.Generator(np.random.PCG64(0)).integers(0, 19, size=(100, 19))
    expected = sums[indices].sum(axis=1) / component_sizes[indices].sum(axis=1)
    assert np.array_equal(got["fp"], expected)


def test_qualification_point_bounds_are_inclusive_and_q05_is_strict():
    manifest = json.loads(MANIFEST.read_text())
    entries = [row for row in manifest["examples"] if row["split"] == RUN.SPLIT]
    data = LB.index_by_id(LB.load_dataset(DATA, authenticate=True))
    records = []
    for index, entry in enumerate(entries):
        answer = data[entry["id"]]["answer"]
        wrong = "ABCD"[("ABCD".index(answer) + 1) % 4]
        for arm, limit in (("fp", 10), ("uniform", 6)):
            records.append({
                "item_id": entry["id"], "arm": arm,
                "forced_choice": answer if index < limit else wrong,
            })
    frame = pd.DataFrame(records)
    original = READER._component_bootstrap
    try:
        READER._component_bootstrap = lambda *args, **kwargs: {
            "fp": np.full(READER.BOOTSTRAP_DRAWS, 0.26),
            "uniform": np.full(READER.BOOTSTRAP_DRAWS, 0.26),
        }
        summary = READER.analyze(frame, manifest=manifest, dataset_path=DATA)
        assert summary["fp_accuracy"] == 0.50
        assert summary["uniform_accuracy"] == 0.30
        assert summary["decision"] == "advance_v6_development"
        READER._component_bootstrap = lambda *args, **kwargs: {
            "fp": np.full(READER.BOOTSTRAP_DRAWS, 0.25),
            "uniform": np.full(READER.BOOTSTRAP_DRAWS, 0.25),
        }
        summary = READER.analyze(frame, manifest=manifest, dataset_path=DATA)
        assert summary["decision"] == "stop_v6_qualification"
        assert not summary["fp_bootstrap_gate"]
        assert not summary["uniform_bootstrap_gate"]
    finally:
        READER._component_bootstrap = original


def test_complete_sidecar_contract_authenticates():
    manifest, file_hash, rows = _manifest_and_rows()
    frame = pd.DataFrame(rows, columns=RUN.PREDICTION_COLUMNS)
    config = ROOT / "h0_measurement/models.yaml"
    with tempfile.TemporaryDirectory() as directory:
        parquet = Path(directory) / f"{RUN.ARTIFACT_BASENAME}.parquet"
        frame.to_parquet(parquet, index=False)
        sidecar = RUN._sidecar(
            manifest=manifest,
            manifest_path=MANIFEST.resolve(),
            manifest_file_hash=file_hash,
            source_manifest_path=SOURCE_MANIFEST.resolve(),
            frame=frame,
            parquet_path=parquet,
            elapsed_seconds=1.0,
            device_placements=["cuda:0"],
            config_path=config.resolve(),
            source_hashes=RUN.executed_source_hashes(config.resolve()),
            snapshot_attestation=QA.authenticate_snapshot(SNAPSHOT),
        )
        assert set(sidecar) == READER.SIDECAR_FIELDS
        READER._check_sidecar(
            sidecar,
            predictions_path=parquet.resolve(),
            manifest_path=MANIFEST.resolve(),
            manifest=manifest,
            manifest_file_hash=file_hash,
            config_path=config.resolve(),
        )
        bad = copy.deepcopy(sidecar)
        bad["manifest_content_sha256"] = "0" * 64
        expect_error(
            lambda: READER._check_sidecar(
                bad,
                predictions_path=parquet.resolve(),
                manifest_path=MANIFEST.resolve(),
                manifest=manifest,
                manifest_file_hash=file_hash,
                config_path=config.resolve(),
            ),
            READER.QwenQualificationReaderError,
            "manifest content hash",
        )


def test_exact_uniform_tensor_contract_and_deterministic_id():
    ctx_len = 17
    bits = {
        layer: torch.full((RUN.N_KV_HEADS, ctx_len), RUN.BUDGET, dtype=torch.uint8)
        for layer in range(RUN.N_LAYERS)
    }
    expected = RUN.expected_uniform_allocation_id(ctx_len)
    assert RUN.validate_uniform_allocation(bits, ctx_len=ctx_len) == expected
    assert RUN.R8.allocation_id(bits) == expected
    tampered = {layer: value.clone() for layer, value in bits.items()}
    tampered[7][0, 0] = 1
    tampered[7][0, 1] = 3
    expect_error(
        lambda: RUN.validate_uniform_allocation(tampered, ctx_len=ctx_len),
        RUN.QwenQualificationRunnerError,
        "bit width other than",
    )
    missing = dict(bits)
    missing.pop(47)
    expect_error(
        lambda: RUN.validate_uniform_allocation(missing, ctx_len=ctx_len),
        RUN.QwenQualificationRunnerError,
        "expected 0..47",
    )


def test_snapshot_inventory_authentication_rejects_symlink_tamper():
    attestation = QA.authenticate_snapshot(SNAPSHOT)
    assert attestation == {
        "revision": QA.MODEL_REVISION,
        "file_count": QA.SNAPSHOT_FILE_COUNT,
        "inventory_sha256": QA.SNAPSHOT_INVENTORY_SHA256,
        "metadata_sha256": QA.SNAPSHOT_METADATA_SHA256,
    }
    with tempfile.TemporaryDirectory() as directory:
        model_root = Path(directory) / "model-cache"
        copied = model_root / "snapshots" / QA.MODEL_REVISION
        blobs = model_root / "blobs"
        copied.mkdir(parents=True)
        blobs.mkdir(parents=True)
        for source in sorted(SNAPSHOT.iterdir()):
            target = os.readlink(source)
            destination_blob = blobs / Path(target).name
            if not destination_blob.exists():
                if source.name in QA.SNAPSHOT_METADATA_SHA256:
                    destination_blob.write_bytes(source.read_bytes())
                else:
                    destination_blob.write_bytes(b"")
            os.symlink(target, copied / source.name)
        assert QA.authenticate_snapshot(copied) == attestation
        (copied / "config.json").unlink()
        os.symlink(os.readlink(SNAPSHOT / "tokenizer_config.json"), copied / "config.json")
        expect_error(
            lambda: QA.authenticate_snapshot(copied),
            QA.QwenManifestError,
            "snapshot inventory SHA-256",
        )


def test_source_hash_authentication_and_cuda_offload_rejection():
    config = ROOT / "h0_measurement/models.yaml"
    hashes = RUN.executed_source_hashes(config)
    runner_key = str(Path(RUN.__file__).resolve().relative_to(ROOT))
    sidecar = {
        "executed_source_sha256": hashes,
        "runner_source": runner_key,
        "runner_source_sha256": hashes[runner_key],
    }
    READER._check_source_hashes(sidecar, config)
    bad = copy.deepcopy(sidecar)
    bad["executed_source_sha256"][runner_key] = "0" * 64
    expect_error(
        lambda: READER._check_source_hashes(bad, config),
        READER.QwenQualificationReaderError,
        "executed source hash",
    )
    cpu_model = SimpleNamespace(
        hf_device_map={"": "cpu"},
        parameters=lambda: iter([torch.zeros(1)]),
    )
    expect_error(
        lambda: RUN.validate_cuda_placement(cpu_model),
        RUN.QwenQualificationRunnerError,
        "not wholly resident on CUDA",
    )


def test_lock_content_hash_and_atomic_idempotence():
    value = {"lock_version": READER.LOCK_VERSION, "decision": "advance_v6_development"}
    value["content_sha256"] = READER._content_sha256(value)
    assert value["content_sha256"] == READER._content_sha256(value)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "advance.lock.json"
        READER._write_atomic_once(path, value)
        READER._write_atomic_once(path, value)
        changed = dict(value, decision="stop_v6_qualification")
        changed["content_sha256"] = READER._content_sha256(changed)
        expect_error(
            lambda: READER._write_atomic_once(path, changed),
            READER.QwenQualificationReaderError,
            "stale/different",
        )


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS: {len(tests)} V6 Qwen qualification tests")
