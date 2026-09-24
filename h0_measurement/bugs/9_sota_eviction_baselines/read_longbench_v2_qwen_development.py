#!/usr/bin/env python3
"""Strict reader for V6 Qwen LongBench-v2 development artifacts.

The development runner is label blind.  It writes one scalar-only prediction
table and one scalar-only branch-blind scaffold-proxy table.  This reader
authenticates the frozen Qwen manifest, the pinned dataset, the advancing V6
qualification lock, both parquets and both sidecars before joining released
answers in memory and applying the preregistered sequential gates.

An authenticated experimental decision exits zero.  Artifact or provenance
drift exits two.  A confirmation lock is written idempotently only when the
competence, opportunity, and proxy-transfer gates all pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h0_measurement import audit_longbench_v2_qwen as QA  # noqa: E402
from h0_measurement import run_longbench_v2_qwen_development as RUN  # noqa: E402
from sievelib import policy_diagnostic as PD  # noqa: E402
from sievelib import tasks_longbench_v2 as LB  # noqa: E402


SPLIT = RUN.SPLIT
MODEL = RUN.MODEL_TAG
MODEL_ID = RUN.MODEL_ID
CTX = RUN.CTX
MAX_INPUT = RUN.MAX_PROMPT_TOKENS
WINDOW = RUN.WINDOW
BUDGET = float(RUN.BUDGET)
MAXB = RUN.MAXB
PROTOCOL_VERSION = RUN.PROTOCOL_VERSION
RUNNER_VERSION = RUN.RUNNER_VERSION
READER_VERSION = "longbench_v2_qwen_development_reader_v1"
CONFIRMATION_LOCK_VERSION = "longbench_v2_sieve_v6_qwen_confirmation_lock_v1"
QUALIFICATION_LOCK_VERSION = "longbench_v2_sieve_v6_qwen_qualification_lock_v1"
QUALIFICATION_LOCK_CONTENT_SHA256 = "4ede1d969fc9ad4905587baa9000eae30f670e93752622c9ea516d90ec576491"
QUALIFICATION_LOCK_FILE_SHA256 = "1deaad5ceb35f943301262fd81c2dfa58b1e5a0dbff2ceaa0975af8257ecebd9"
ENDPOINT_VERSION = RUN.ENDPOINT_VERSION
PROXY_RULE_VERSION = RUN.PROXY_RULE_VERSION
DECODE = "forced_choice_teacher_forced_argmax"
COMPRESS_AT = "official_chat_prompt_end"
ALLOCATION_ID_ALGORITHM = RUN.R8.ALLOCATION_ID_ALGORITHM
ARMS = tuple(RUN.ARM_LABELS)
CANDIDATES = tuple(RUN.CANDIDATES)
RAW_ARMS = tuple(RUN.RAW_ARMS)
EXPECTED_ROWS = RUN.EXPECTED_ITEMS
EXPECTED_COMPONENTS = RUN.EXPECTED_COMPONENTS
EXPECTED_PREDICTION_ROWS = EXPECTED_ROWS * len(ARMS)
EXPECTED_PROXY_ROWS = EXPECTED_ROWS * len(CANDIDATES)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
FP_ALLOCATION_ID = hashlib.sha256(b"fp16").hexdigest()
MANIFEST_CONTENT_SHA256 = QA.FROZEN_MANIFEST_CONTENT_SHA256
MANIFEST_FILE_SHA256 = QA.FROZEN_MANIFEST_FILE_SHA256
SPLIT_ID_SHA256 = QA.EXPECTED_SPLITS[SPLIT]["id_sha256"]
SPLIT_ID_TOKEN_SHA256 = QA.EXPECTED_SPLITS[SPLIT]["id_token_sha256"]
SPLIT_ID_PROMPT_SHA256 = "befd2f05d107a41e8bee2cec59be3ae42cbf0e0f7359c393c1c948df9c49df71"
HALF_COMPONENT_COUNTS = (22, 22)
HALF_ROW_COUNTS = (28, 24)
HALF_ID_SHA256 = (
    "b006a353c33488d661b87932cf1c4abf8ce1e7aff63e688810a74b13a85184b8",
    "6adf6c4b4e378a4f4605a4a959df0766fad7329f384bf27c8734abaae1296a1b",
)
SCAFFOLD_TOKEN_IDS = tuple(QA.SCAFFOLD_TOKEN_IDS)
CHOICE_BRANCH_IDS = dict(QA.CHOICE_BRANCH_IDS)
SCAFFOLD_TOKEN_HASH = PD.token_hash(SCAFFOLD_TOKEN_IDS)
CHOICE_BRANCH_IDS_HASH = PD.token_hash(
    [CHOICE_BRANCH_IDS[choice] for choice in "ABCD"]
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")

COMMON_COLUMNS = tuple(RUN.COMMON_COLUMNS)
PREDICTION_COLUMNS = tuple(RUN.PREDICTION_COLUMNS)
SCAFFOLD_KL_COLUMNS = tuple(f"scaffold_kl_pos{index}" for index in range(5))
PROXY_COLUMNS = tuple(RUN.PROXY_COLUMNS)

# Exact row schemas reject these automatically.  Keeping an explicit list gives
# a useful error when a runner accidentally serializes transient or labelled
# values.
FORBIDDEN_RESULT_COLUMNS = {
    "answer", "gold", "gold_answer", "label", "correct", "correctness",
    "score", "response", "parsed_answer", "logit", "logits", "raw_logits",
    "fp_logits", "candidate_logits", "probability", "probabilities", "probs",
    "fp_probs", "candidate_probs", "choice_logits", "choice_probs",
}

SIDECAR_COMMON_FIELDS = (
    'runner_version', 'protocol_version', 'task_version', 'endpoint_version',
    'proxy_rule_version', 'dataset_repo', 'dataset_revision', 'dataset_sha256',
    'dataset_bytes', 'official_code_repo', 'official_code_revision', 'official_prompt_sha256',
    'source_manifest', 'source_manifest_version', 'source_manifest_content_sha256', 'source_manifest_file_sha256',
    'manifest', 'manifest_version', 'manifest_content_sha256', 'manifest_file_sha256',
    'split', 'split_count', 'split_components', 'component_order',
    'half_component_counts', 'half_row_counts', 'half_id_sha256', 'split_ids_sha256',
    'split_id_token_sha256', 'split_id_prompt_sha256', 'model', 'model_id',
    'model_revision', 'tokenizer_revision', 'chat_template_sha256', 'add_generation_prompt',
    'enable_thinking', 'transformers_version', 'tokenizers_version', 'torch_version',
    'snapshot_revision', 'snapshot_file_count', 'snapshot_inventory_sha256', 'snapshot_metadata_sha256',
    'snapshot_seal_timing', 'qualification_lock', 'qualification_lock_version', 'qualification_lock_file_sha256',
    'qualification_lock_content_sha256', 'qualification_predictions_sha256', 'qualification_sidecar_sha256', 'ctx',
    'max_prompt_tokens', 'window', 'budget', 'allocator_budget_rule',
    'cascade_bits',
    'dtype', 'bit_list', 'maxb', 'rot_seed',
    'norm_correct', 'chunk', 'n_layers', 'n_attention_heads',
    'n_kv_heads', 'head_dim', 'attn_impl', 'device_placements',
    'decode', 'temperature', 'compress_at', 'raw_arms',
    'arms', 'candidates', 'allocation_id_algorithm', 'scaffold_token_ids',
    'scaffold_token_hash', 'choice_branch_ids', 'choice_branch_ids_hash', 'proxy_positions',
    'choice_position', 'no_truncation', 'no_free_generation', 'no_raw_logits',
    'no_labels', 'proxy_dtype', 'uniform_allocation_contract', 'gold_unused_for_prompt_allocation_execution_and_output',
    'source_seal_timing', 'executed_source_sha256', 'runner_source', 'runner_source_sha256',
    'elapsed_seconds', 'item_ids_sha256', 'item_count', 'artifact',
    'parquet', 'parquet_sha256', 'rows', 'expected_rows',
    'row_key', 'paired_artifact', 'paired_parquet', 'paired_parquet_sha256',
    'paired_rows',
)


class LongBenchForcedChoiceError(ValueError):
    """A V6 Qwen artifact violates the frozen development contract."""


def _fail(message: str) -> None:
    raise LongBenchForcedChoiceError(message)


def _python_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _same(expected: Any, actual: Any, description: str) -> None:
    actual = _python_scalar(actual)
    if expected != actual:
        _fail(f"{description}: expected {expected!r}, got {actual!r}")


def _sha256_file(path: str | Path) -> str:
    return LB.sha256_file(path)


def _regular(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        _fail(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        _fail(f"{label} is not a regular file: {resolved}")
    return resolved


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    candidate = _regular(path, label)
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {label} {candidate}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must contain one JSON object")
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def authenticate_qualification_lock(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Authenticate the single advancing V6 qualification authorization."""
    lock_path = _regular(path, "qualification lock")
    lock = _load_json(lock_path, "qualification lock")
    exact_top = {
        "lock_version", "protocol_version", "runner_version", "reader_version",
        "task_version", "endpoint_version", "proxy_rule_version", "decision",
        "qualification", "provenance", "partitions", "execution", "thresholds",
        "content_sha256",
    }
    _same(exact_top, set(lock), "qualification lock top-level schema")
    _same(QUALIFICATION_LOCK_FILE_SHA256, _sha256_file(lock_path), "qualification lock file hash")
    _same(QUALIFICATION_LOCK_CONTENT_SHA256, lock.get("content_sha256"), "qualification lock frozen content hash")
    _same(_content_sha256(lock), lock.get("content_sha256"), "qualification lock canonical content hash")
    for expected, actual, label in (
        (QUALIFICATION_LOCK_VERSION, lock.get("lock_version"), "version"),
        ("longbench_v2_sieve_v6_qwen_qualification_v1", lock.get("protocol_version"), "protocol"),
        ("longbench_v2_qwen_qualification_runner_v1", lock.get("runner_version"), "runner"),
        ("longbench_v2_qwen_qualification_reader_v1", lock.get("reader_version"), "reader"),
        (RUN.TASK_VERSION, lock.get("task_version"), "task"),
        (ENDPOINT_VERSION, lock.get("endpoint_version"), "endpoint"),
        (PROXY_RULE_VERSION, lock.get("proxy_rule_version"), "proxy rule"),
        ("advance_v6_development", lock.get("decision"), "decision"),
    ):
        _same(expected, actual, f"qualification lock {label}")

    execution = lock.get("execution")
    if not isinstance(execution, dict):
        _fail("qualification lock execution is absent")
    expected_execution = {
        "model": MODEL, "model_id": MODEL_ID,
        "model_revision": RUN.MODEL_REVISION,
        "tokenizer_revision": RUN.TOKENIZER_REVISION,
        "chat_template_sha256": QA.CHAT_TEMPLATE_SHA256,
        "enable_thinking": False, "ctx": CTX, "window": WINDOW,
        "budget": RUN.BUDGET, "dtype": RUN.DTYPE_NAME,
        "bit_list": list(RUN.BIT_LIST), "cascade_bits": RUN.CASCADE_BITS,
        "maxb": MAXB, "rot_seed": RUN.ROT_SEED,
        "norm_correct": RUN.NORM_CORRECT, "chunk": RUN.CHUNK,
        "n_layers": RUN.N_LAYERS, "n_attention_heads": RUN.N_ATTENTION_HEADS,
        "n_kv_heads": RUN.N_KV_HEADS, "head_dim": RUN.HEAD_DIM,
        "uniform_allocation_contract": "48_layers_x_4_kv_x_ctx_len_all_uint8_2",
        "attn_impl": RUN.C.IMPL,
        "qualification_arms": ["fp", "uniform"],
        "development_raw_arms": list(RAW_ARMS),
        "development_arms": list(ARMS),
        "development_candidates": list(CANDIDATES),
        "endpoint_version": ENDPOINT_VERSION,
        "proxy_rule_version": PROXY_RULE_VERSION,
        "scaffold_token_ids": list(SCAFFOLD_TOKEN_IDS),
        "choice_branch_ids": CHOICE_BRANCH_IDS,
    }
    _same(expected_execution, execution, "qualification lock execution")

    thresholds = lock.get("thresholds")
    expected_thresholds = {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_rng": "numpy.Generator(PCG64)",
        "bootstrap_quantile": 0.05,
        "bootstrap_quantile_method": "linear",
        "chance": 0.25,
        "qualification_fp_accuracy_min": 0.50,
        "qualification_fp_q05_strictly_above": 0.25,
        "qualification_uniform_accuracy_interval": [0.30, 0.80],
        "qualification_uniform_q05_strictly_above": 0.25,
        "development_fp_accuracy_min": 0.50,
        "development_fp_q05_strictly_above": 0.25,
        "development_fixed_accuracy_interval": [0.30, 0.80],
        "development_fixed_q05_strictly_above": 0.25,
        "development_H_min": 0.10,
        "development_H_q05_strictly_above": 0.0,
        "development_H_each_half_min": 0.05,
        "development_G_min": 0.05,
        "development_G_over_H_min": 0.50,
        "development_G_q05_min": 0.0,
        "development_G_each_half_min": 0.0,
    }
    _same(expected_thresholds, thresholds, "qualification lock thresholds")

    partitions = lock.get("partitions")
    if not isinstance(partitions, dict):
        _fail("qualification lock partitions are absent")
    development_partition = partitions.get("development")
    expected_development = {
        **dict(QA.EXPECTED_SPLITS[SPLIT]),
        "id_prompt_sha256": SPLIT_ID_PROMPT_SHA256,
        "half_component_counts": list(HALF_COMPONENT_COUNTS),
        "half_row_counts": list(HALF_ROW_COUNTS),
        "half_id_sha256": list(HALF_ID_SHA256),
    }
    _same(expected_development, development_partition, "qualification lock development partition")
    _same(
        "min_sha256_split_namespace_context_hash_then_component_id_v1",
        partitions.get("component_order"), "qualification lock component order",
    )

    provenance = lock.get("provenance")
    if not isinstance(provenance, dict):
        _fail("qualification lock provenance is absent")
    for expected, actual, label in (
        (LB.DATASET_REPO, provenance.get("dataset_repo"), "dataset repo"),
        (LB.DATASET_REVISION, provenance.get("dataset_revision"), "dataset revision"),
        (LB.DATASET_SHA256, provenance.get("dataset_sha256"), "dataset hash"),
        (LB.DATASET_BYTES, provenance.get("dataset_bytes"), "dataset bytes"),
        (LB.OFFICIAL_PROMPT_SHA256, provenance.get("official_prompt_sha256"), "official prompt"),
        (QA.SOURCE_MANIFEST_VERSION, provenance.get("source_manifest_version"), "source manifest version"),
        (QA.SOURCE_MANIFEST_CONTENT_SHA256, provenance.get("source_manifest_content_sha256"), "source manifest content hash"),
        (QA.SOURCE_MANIFEST_FILE_SHA256, provenance.get("source_manifest_file_sha256"), "source manifest file hash"),
        (QA.MANIFEST_VERSION, provenance.get("qwen_manifest_version"), "Qwen manifest version"),
        (MANIFEST_CONTENT_SHA256, provenance.get("qwen_manifest_content_sha256"), "Qwen manifest content hash"),
        (MANIFEST_FILE_SHA256, provenance.get("qwen_manifest_file_sha256"), "Qwen manifest file hash"),
        (QA.MODEL_REVISION, provenance.get("snapshot_revision"), "snapshot revision"),
        (QA.SNAPSHOT_FILE_COUNT, provenance.get("snapshot_file_count"), "snapshot file count"),
        (QA.SNAPSHOT_INVENTORY_SHA256, provenance.get("snapshot_inventory_sha256"), "snapshot inventory"),
        (dict(QA.SNAPSHOT_METADATA_SHA256), provenance.get("snapshot_metadata_sha256"), "snapshot metadata"),
    ):
        _same(expected, actual, f"qualification lock {label}")
    source_hashes = provenance.get("executed_source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        _fail("qualification lock executed source seal is absent")
    for relative, expected_hash in source_hashes.items():
        source = _regular(ROOT / str(relative), f"qualification executed source {relative}")
        try:
            source.relative_to(ROOT.resolve())
        except ValueError:
            _fail(f"qualification source path escapes repository: {relative}")
        _same(expected_hash, _sha256_file(source), f"qualification executed source {relative}")

    qualification = lock.get("qualification")
    if not isinstance(qualification, dict):
        _fail("qualification lock evidence is absent")
    for path_key, hash_key, label in (
        ("predictions", "predictions_sha256", "qualification predictions"),
        ("sidecar", "sidecar_sha256", "qualification sidecar"),
    ):
        evidence = _regular(str(qualification.get(path_key, "")), label)
        _same(qualification.get(hash_key), _sha256_file(evidence), f"{label} hash")
    summary = qualification.get("summary")
    if not isinstance(summary, dict) or summary.get("decision") != "advance_v6_development":
        _fail("qualification lock does not contain an advancing summary")
    required_true = (
        "fp_point_gate", "fp_bootstrap_gate", "uniform_point_gate",
        "uniform_bootstrap_gate",
    )
    if any(summary.get(gate) is not True for gate in required_true):
        _fail("qualification lock summary does not pass every qualification gate")
    return lock, lock_path


def _id_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    payload = "".join(
        f"{item_id}\n" for item_id in sorted(str(row["id"]) for row in rows)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _split_id_token_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    payload = "".join(
        f"{item_id}\t{tokens}\n"
        for item_id, tokens in sorted(
            (str(row["id"]), int(row["input_tokens"])) for row in rows
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordered_components(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return the frozen salted component order for a manifest phase."""
    members: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        members.setdefault(str(row["group_id"]), []).append(row)

    def sort_key(group_id: str) -> tuple[bytes, str]:
        salted = min(
            hashlib.sha256(
                LB.SPLIT_NAMESPACE + str(row["context_hash"]).encode("ascii")
            ).digest()
            for row in members[group_id]
        )
        return salted, group_id

    return tuple(sorted(members, key=sort_key))


def frozen_halves(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[frozenset[str], frozenset[str]]]:
    order = ordered_components(rows)
    if len(order) != EXPECTED_COMPONENTS:
        _fail(
            f"development has {len(order)} components, expected {EXPECTED_COMPONENTS}"
        )
    halves = (frozenset(order[:22]), frozenset(order[22:]))
    for index, groups in enumerate(halves):
        half_rows = [row for row in rows if str(row["group_id"]) in groups]
        _same(
            HALF_COMPONENT_COUNTS[index], len(groups),
            f"development half {index + 1} component count",
        )
        _same(
            HALF_ROW_COUNTS[index], len(half_rows),
            f"development half {index + 1} row count",
        )
        _same(
            HALF_ID_SHA256[index], _id_hash(half_rows),
            f"development half {index + 1} ID SHA-256",
        )
    return order, halves


def authenticate_manifest(
    path: str | Path,
    *,
    dataset_path: str | Path,
    source_manifest_path: str | Path,
    tokenizer: Any,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Authenticate the complete frozen Qwen manifest and development split."""
    try:
        manifest, file_hash = QA.authenticate_manifest(
            path,
            dataset_path=dataset_path,
            source_manifest_path=source_manifest_path,
            tokenizer=tokenizer,
        )
    except (OSError, ValueError) as exc:
        _fail(f"Qwen manifest authentication failed: {exc}")
    _same(MANIFEST_CONTENT_SHA256, manifest.get("content_sha256"), "manifest content hash")
    _same(MANIFEST_FILE_SHA256, file_hash, "manifest file hash")
    development = [row for row in manifest["examples"] if row["split"] == SPLIT]
    _same(EXPECTED_ROWS, len(development), "development manifest row count")
    _same(SPLIT_ID_SHA256, QA.id_sha256(development), "development ID hash")
    _same(SPLIT_ID_TOKEN_SHA256, QA.id_token_sha256(development), "development ID/token hash")
    _same(SPLIT_ID_PROMPT_SHA256, QA.id_prompt_sha256(development), "development ID/prompt hash")
    split_record = manifest["split_counts"][SPLIT]
    for field, expected in (
        ("rows", EXPECTED_ROWS), ("components", EXPECTED_COMPONENTS),
        ("id_sha256", SPLIT_ID_SHA256),
        ("id_token_sha256", SPLIT_ID_TOKEN_SHA256),
        ("id_prompt_sha256", SPLIT_ID_PROMPT_SHA256),
    ):
        _same(expected, split_record.get(field), f"development manifest {field}")
    frozen_halves(development)
    return manifest, file_hash, development


def authenticate_dataset(
    path: str | Path, manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    try:
        data = LB.load_dataset(path, authenticate=True)
    except (OSError, ValueError) as exc:
        _fail(f"dataset authentication failed: {exc}")
    by_id = LB.index_by_id(data)
    for row in manifest_rows:
        item_id = str(row["id"])
        if item_id not in by_id:
            _fail(f"manifest item {item_id!r} is absent from pinned dataset")
        item = by_id[item_id]
        metadata = {
            key: item[key]
            for key in ("domain", "sub_domain", "difficulty", "length")
        }
        _same(row["metadata"], metadata, f"manifest metadata for {item_id}")
        _same(row["context_hash"], LB.context_hash(item), f"context hash for {item_id}")
        _same(
            row["question_hash"], hashlib.sha256(item["question"].strip().encode("utf-8")).hexdigest(),
            f"question hash for {item_id}",
        )
    return by_id


def _require_exact_schema(
    frame: pd.DataFrame, expected: Sequence[str], label: str,
) -> None:
    columns = tuple(str(column) for column in frame.columns)
    expected_tuple = tuple(expected)
    extra = [column for column in columns if column not in expected_tuple]
    forbidden = sorted(
        column for column in extra
        if column.lower() in FORBIDDEN_RESULT_COLUMNS
        or "logit" in column.lower()
        or column.lower().startswith(("gold_", "raw_"))
    )
    if forbidden:
        _fail(f"{label} contains forbidden label/raw columns: {forbidden}")
    if columns != expected_tuple:
        missing = [column for column in expected_tuple if column not in columns]
        _fail(
            f"{label} does not have the exact ordered schema: "
            f"missing={missing}, extra={extra}"
        )
    for column in columns:
        if frame[column].dtype != object:
            continue
        for value in frame[column]:
            if isinstance(value, (list, tuple, dict, np.ndarray)):
                _fail(f"{label}.{column} contains a nonscalar value")


def _numeric(
    frame: pd.DataFrame, column: str, label: str, *, integer: bool = False,
) -> pd.Series:
    try:
        values = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as exc:
        _fail(f"{label}.{column} is not numeric: {exc}")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        _fail(f"{label}.{column} contains a non-finite value")
    if integer and not np.equal(array, np.floor(array)).all():
        _fail(f"{label}.{column} must contain integers")
    return values.astype(np.int64) if integer else values.astype(float)


def _bool(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    values = frame[column]
    valid = values.map(lambda value: isinstance(value, (bool, np.bool_)))
    if not bool(valid.all()):
        _fail(f"{label}.{column} must contain booleans")
    return values.astype(bool)


def _all_sha256(frame: pd.DataFrame, column: str, label: str) -> None:
    valid = frame[column].map(
        lambda value: isinstance(value, str) and bool(SHA256_RE.fullmatch(value))
    )
    if not bool(valid.all()):
        _fail(f"{label}.{column} must contain lowercase SHA-256 values")


def _check_development_source_seal(
    sidecar: Mapping[str, Any], *, config_path: Path,
) -> None:
    observed = sidecar.get("executed_source_sha256")
    if not isinstance(observed, dict) or not observed:
        _fail("development executed-source seal is absent")
    expected = RUN.executed_source_hashes(config_path)
    _same(expected, observed, "development executed-source seal")
    runner_key = str(Path(RUN.__file__).resolve().relative_to(ROOT.resolve()))
    _same(runner_key, sidecar.get("runner_source"), "development runner source path")
    _same(expected[runner_key], sidecar.get("runner_source_sha256"), "development runner source hash")
    for relative, expected_hash in observed.items():
        raw = ROOT / str(relative)
        try:
            raw.resolve().relative_to(ROOT.resolve())
        except ValueError:
            _fail(f"development source path escapes repository: {relative}")
        source = _regular(raw, f"development source {relative}")
        _same(expected_hash, _sha256_file(source), f"development source hash {relative}")


def _sidecar_constant_checks(
    sidecar: Mapping[str, Any], *, artifact: str, parquet: Path,
    paired_artifact: str, paired_parquet: Path, manifest_path: Path,
    manifest: Mapping[str, Any], manifest_file_hash: str,
    expected_rows: int, row_key: Sequence[str], config_path: Path,
    qualification_attestation: Mapping[str, Any],
    snapshot_attestation: Mapping[str, Any],
) -> None:
    missing = sorted(set(SIDECAR_COMMON_FIELDS) - set(sidecar))
    extra = sorted(set(sidecar) - set(SIDECAR_COMMON_FIELDS))
    if missing or extra:
        _fail(
            f"{artifact} sidecar does not have the exact schema: "
            f"missing={missing}, extra={extra}"
        )
    checks = (
        (RUNNER_VERSION, sidecar["runner_version"], "runner version"),
        (PROTOCOL_VERSION, sidecar["protocol_version"], "protocol version"),
        (RUN.TASK_VERSION, sidecar["task_version"], "task version"),
        (ENDPOINT_VERSION, sidecar["endpoint_version"], "endpoint version"),
        (PROXY_RULE_VERSION, sidecar["proxy_rule_version"], "proxy rule"),
        (LB.DATASET_REPO, sidecar["dataset_repo"], "dataset repo"),
        (LB.DATASET_REVISION, sidecar["dataset_revision"], "dataset revision"),
        (LB.DATASET_SHA256, sidecar["dataset_sha256"], "dataset SHA-256"),
        (LB.DATASET_BYTES, sidecar["dataset_bytes"], "dataset bytes"),
        (LB.OFFICIAL_CODE_REPO, sidecar["official_code_repo"], "official repo"),
        (LB.OFFICIAL_CODE_REVISION, sidecar["official_code_revision"], "official revision"),
        (LB.OFFICIAL_PROMPT_SHA256, sidecar["official_prompt_sha256"], "official prompt"),
        (QA.SOURCE_MANIFEST_ID, sidecar["source_manifest"], "source manifest ID"),
        (QA.SOURCE_MANIFEST_VERSION, sidecar["source_manifest_version"], "source manifest version"),
        (QA.SOURCE_MANIFEST_CONTENT_SHA256, sidecar["source_manifest_content_sha256"], "source manifest content hash"),
        (QA.SOURCE_MANIFEST_FILE_SHA256, sidecar["source_manifest_file_sha256"], "source manifest file hash"),
        (str(manifest_path), sidecar["manifest"], "manifest path"),
        (QA.MANIFEST_VERSION, sidecar["manifest_version"], "manifest version"),
        (manifest["content_sha256"], sidecar["manifest_content_sha256"], "manifest content hash"),
        (manifest_file_hash, sidecar["manifest_file_sha256"], "manifest file hash"),
        (SPLIT, sidecar["split"], "split"),
        (EXPECTED_ROWS, sidecar["split_count"], "split count"),
        (EXPECTED_COMPONENTS, sidecar["split_components"], "component count"),
        (SPLIT_ID_SHA256, sidecar["split_ids_sha256"], "split ID hash"),
        (SPLIT_ID_TOKEN_SHA256, sidecar["split_id_token_sha256"], "split ID/token hash"),
        (SPLIT_ID_PROMPT_SHA256, sidecar["split_id_prompt_sha256"], "split ID/prompt hash"),
        ("min_sha256_split_namespace_context_hash_then_component_id_v1", sidecar["component_order"], "component order"),
        (list(HALF_COMPONENT_COUNTS), sidecar["half_component_counts"], "half component counts"),
        (list(HALF_ROW_COUNTS), sidecar["half_row_counts"], "half row counts"),
        (list(HALF_ID_SHA256), sidecar["half_id_sha256"], "half ID hashes"),
        (MODEL, sidecar["model"], "model tag"),
        (MODEL_ID, sidecar["model_id"], "model ID"),
        (RUN.MODEL_REVISION, sidecar["model_revision"], "model revision"),
        (RUN.TOKENIZER_REVISION, sidecar["tokenizer_revision"], "tokenizer revision"),
        (QA.CHAT_TEMPLATE_SHA256, sidecar["chat_template_sha256"], "chat template"),
        (True, sidecar["add_generation_prompt"], "generation prompt flag"),
        (False, sidecar["enable_thinking"], "thinking flag"),
        (QA.TRANSFORMERS_VERSION, sidecar["transformers_version"], "transformers version"),
        (QA.TOKENIZERS_VERSION, sidecar["tokenizers_version"], "tokenizers version"),
        (torch.__version__, sidecar["torch_version"], "torch version"),
        (QA.MODEL_REVISION, sidecar["snapshot_revision"], "snapshot revision"),
        (QA.SNAPSHOT_FILE_COUNT, sidecar["snapshot_file_count"], "snapshot file count"),
        (QA.SNAPSHOT_INVENTORY_SHA256, sidecar["snapshot_inventory_sha256"], "snapshot inventory"),
        (dict(QA.SNAPSHOT_METADATA_SHA256), sidecar["snapshot_metadata_sha256"], "snapshot metadata"),
        ("before_tokenizer_and_rechecked_before_artifact_write", sidecar["snapshot_seal_timing"], "snapshot seal timing"),
        (qualification_attestation["lock_path"], sidecar["qualification_lock"], "qualification lock path"),
        (RUN.QUALIFICATION_LOCK_VERSION, sidecar["qualification_lock_version"], "qualification lock version"),
        (qualification_attestation["lock_file_sha256"], sidecar["qualification_lock_file_sha256"], "qualification lock file hash"),
        (qualification_attestation["lock_content_sha256"], sidecar["qualification_lock_content_sha256"], "qualification lock content hash"),
        (qualification_attestation["qualification_predictions_sha256"], sidecar["qualification_predictions_sha256"], "qualification prediction hash"),
        (qualification_attestation["qualification_sidecar_sha256"], sidecar["qualification_sidecar_sha256"], "qualification sidecar hash"),
        (CTX, sidecar["ctx"], "context window"),
        (MAX_INPUT, sidecar["max_prompt_tokens"], "max prompt tokens"),
        (WINDOW, sidecar["window"], "window"),
        (BUDGET, sidecar["budget"], "budget"),
        (RUN.CASCADE_BITS, sidecar["cascade_bits"], "cascade bits"),
        (RUN.DTYPE_NAME, sidecar["dtype"], "dtype"),
        (list(RUN.BIT_LIST), sidecar["bit_list"], "bit list"),
        (MAXB, sidecar["maxb"], "maxb"),
        (RUN.ROT_SEED, sidecar["rot_seed"], "rotation seed"),
        (RUN.NORM_CORRECT, sidecar["norm_correct"], "norm correction"),
        (RUN.CHUNK, sidecar["chunk"], "chunk"),
        (RUN.N_LAYERS, sidecar["n_layers"], "layer count"),
        (RUN.N_ATTENTION_HEADS, sidecar["n_attention_heads"], "attention-head count"),
        (RUN.N_KV_HEADS, sidecar["n_kv_heads"], "KV-head count"),
        (RUN.HEAD_DIM, sidecar["head_dim"], "head dimension"),
        (RUN.C.IMPL, sidecar["attn_impl"], "attention implementation"),
        (DECODE, sidecar["decode"], "decode"),
        (1.0, sidecar["temperature"], "temperature"),
        (COMPRESS_AT, sidecar["compress_at"], "compression boundary"),
        (list(RAW_ARMS), sidecar["raw_arms"], "raw arm menu"),
        (list(ARMS), sidecar["arms"], "arm menu"),
        (list(CANDIDATES), sidecar["candidates"], "candidate menu"),
        (ALLOCATION_ID_ALGORITHM, sidecar["allocation_id_algorithm"], "allocation algorithm"),
        (list(SCAFFOLD_TOKEN_IDS), sidecar["scaffold_token_ids"], "scaffold IDs"),
        (SCAFFOLD_TOKEN_HASH, sidecar["scaffold_token_hash"], "scaffold hash"),
        (CHOICE_BRANCH_IDS, sidecar["choice_branch_ids"], "choice branch IDs"),
        (CHOICE_BRANCH_IDS_HASH, sidecar["choice_branch_ids_hash"], "choice branch hash"),
        ([0, 1, 2, 3, 4], sidecar["proxy_positions"], "proxy positions"),
        (5, sidecar["choice_position"], "choice position"),
        (True, sidecar["no_truncation"], "no-truncation flag"),
        (True, sidecar["no_free_generation"], "no-generation flag"),
        (True, sidecar["no_raw_logits"], "no-raw-logits flag"),
        (True, sidecar["no_labels"], "no-labels flag"),
        ("float32", sidecar["proxy_dtype"], "proxy dtype"),
        ("48_layers_x_4_kv_x_ctx_len_all_uint8_2", sidecar["uniform_allocation_contract"], "uniform allocation contract"),
        ("feasible", sidecar["allocator_budget_rule"], "allocator budget rule"),
        (True, sidecar["gold_unused_for_prompt_allocation_execution_and_output"], "gold isolation flag"),
        ("before_tokenizer_and_rechecked_before_artifact_write", sidecar["source_seal_timing"], "source seal timing"),
        (SPLIT_ID_SHA256, sidecar["item_ids_sha256"], "item ID hash"),
        (EXPECTED_ROWS, sidecar["item_count"], "item count"),
        (artifact, sidecar["artifact"], "artifact kind"),
        (parquet.name, sidecar["parquet"], "parquet filename"),
        (_sha256_file(parquet), sidecar["parquet_sha256"], "parquet hash"),
        (expected_rows, sidecar["rows"], "row count"),
        (expected_rows, sidecar["expected_rows"], "expected row count"),
        (list(row_key), sidecar["row_key"], "row key"),
        (paired_artifact, sidecar["paired_artifact"], "paired artifact kind"),
        (paired_parquet.name, sidecar["paired_parquet"], "paired parquet filename"),
        (_sha256_file(paired_parquet), sidecar["paired_parquet_sha256"], "paired parquet hash"),
        (EXPECTED_PROXY_ROWS if artifact == "qwen_forced_choice_development" else EXPECTED_PREDICTION_ROWS, sidecar["paired_rows"], "paired rows"),
    )
    for expected, actual, description in checks:
        _same(expected, actual, f"{artifact} sidecar {description}")
    elapsed = float(sidecar["elapsed_seconds"])
    if not math.isfinite(elapsed) or elapsed < 0.0:
        _fail(f"{artifact} sidecar elapsed time is invalid")
    placements = sidecar["device_placements"]
    if not isinstance(placements, list) or not placements:
        _fail(f"{artifact} sidecar device placements are absent")
    if any(str(value).lower().startswith(("cpu", "disk")) for value in placements):
        _fail(f"{artifact} sidecar records CPU/disk model placement: {placements}")
    _same(snapshot_attestation, qualification_attestation["snapshot"], "qualification/development snapshot")
    _check_development_source_seal(sidecar, config_path=config_path)


def validate_sidecars(
    prediction_sidecar: Mapping[str, Any], proxy_sidecar: Mapping[str, Any], *,
    prediction_path: Path, proxy_path: Path, manifest_path: Path,
    manifest: Mapping[str, Any], manifest_file_hash: str, config_path: Path,
    qualification_attestation: Mapping[str, Any],
    snapshot_attestation: Mapping[str, Any],
) -> None:
    _sidecar_constant_checks(
        prediction_sidecar, artifact="qwen_forced_choice_development",
        parquet=prediction_path, paired_artifact="qwen_scaffold_proxy_development",
        paired_parquet=proxy_path, manifest_path=manifest_path,
        manifest=manifest, manifest_file_hash=manifest_file_hash,
        expected_rows=EXPECTED_PREDICTION_ROWS, row_key=("item_id", "arm"),
        config_path=config_path, qualification_attestation=qualification_attestation,
        snapshot_attestation=snapshot_attestation,
    )
    _sidecar_constant_checks(
        proxy_sidecar, artifact="qwen_scaffold_proxy_development",
        parquet=proxy_path, paired_artifact="qwen_forced_choice_development",
        paired_parquet=prediction_path, manifest_path=manifest_path,
        manifest=manifest, manifest_file_hash=manifest_file_hash,
        expected_rows=EXPECTED_PROXY_ROWS, row_key=("item_id", "candidate"),
        config_path=config_path, qualification_attestation=qualification_attestation,
        snapshot_attestation=snapshot_attestation,
    )
    artifact_fields = {
        "artifact", "parquet", "parquet_sha256", "rows", "expected_rows",
        "row_key", "paired_artifact", "paired_parquet",
        "paired_parquet_sha256", "paired_rows",
    }
    for field in set(SIDECAR_COMMON_FIELDS) - artifact_fields:
        _same(prediction_sidecar[field], proxy_sidecar[field], f"paired sidecar {field}")



def select_scaffold_policy(values: Mapping[str, Any]) -> str:
    """Select the exact minimum saved mean KL, breaking ties by menu order."""
    if set(values) != set(CANDIDATES):
        _fail("scaffold selector values do not cover the exact candidate menu")
    selected = CANDIDATES[0]
    try:
        best = float(values[selected])
    except (TypeError, ValueError) as exc:
        _fail(f"invalid scaffold mean KL for {selected}: {exc}")
    if not math.isfinite(best):
        _fail(f"invalid scaffold mean KL for {selected}")
    for candidate in CANDIDATES[1:]:
        try:
            value = float(values[candidate])
        except (TypeError, ValueError) as exc:
            _fail(f"invalid scaffold mean KL for {candidate}: {exc}")
        if not math.isfinite(value):
            _fail(f"invalid scaffold mean KL for {candidate}")
        if value < best:
            selected, best = candidate, value
    return selected


def validate_frames(
    predictions: pd.DataFrame, proxy: pd.DataFrame, *,
    manifest: Mapping[str, Any], manifest_file_hash: str,
    development: Sequence[Mapping[str, Any]],
    data_by_id: Mapping[str, Mapping[str, str]],
) -> None:
    _require_exact_schema(predictions, PREDICTION_COLUMNS, "prediction parquet")
    _require_exact_schema(proxy, PROXY_COLUMNS, "proxy parquet")
    _same(EXPECTED_PREDICTION_ROWS, len(predictions), "prediction parquet rows")
    _same(EXPECTED_PROXY_ROWS, len(proxy), "proxy parquet rows")

    manifest_by_id = {str(row["id"]): row for row in development}
    item_ids = tuple(manifest_by_id)
    expected_prediction_keys = {
        (item_id, arm) for item_id in item_ids for arm in ARMS
    }
    prediction_keys = list(zip(
        predictions["item_id"].astype(str), predictions["arm"].astype(str)
    ))
    if len(set(prediction_keys)) != len(prediction_keys):
        _fail("prediction parquet has duplicate item_id/arm keys")
    if set(prediction_keys) != expected_prediction_keys:
        _fail("prediction parquet does not contain exactly 52 manifest IDs x 9 arms")

    expected_proxy_keys = {
        (item_id, candidate) for item_id in item_ids for candidate in CANDIDATES
    }
    proxy_keys = list(zip(
        proxy["item_id"].astype(str), proxy["candidate"].astype(str)
    ))
    if len(set(proxy_keys)) != len(proxy_keys):
        _fail("proxy parquet has duplicate item_id/candidate keys")
    if set(proxy_keys) != expected_proxy_keys:
        _fail("proxy parquet does not contain exactly 52 manifest IDs x 8 candidates")

    for label, frame in (("prediction", predictions), ("proxy", proxy)):
        for column in ("input_tokens", "n_prompt_tokens", "ctx", "window"):
            frame[column] = _numeric(frame, column, label, integer=True)
        frame["truncated"] = _bool(frame, "truncated", label)
        if bool(frame["truncated"].any()):
            _fail(f"{label} contains a truncated row")
        for column, expected in (
            ("split", SPLIT), ("dataset_sha256", LB.DATASET_SHA256),
            ("manifest_content_sha256", manifest["content_sha256"]),
            ("manifest_file_sha256", manifest_file_hash),
            ("task_version", RUN.TASK_VERSION),
            ("endpoint_version", ENDPOINT_VERSION), ("model", MODEL),
            ("model_id", MODEL_ID), ("model_revision", RUN.MODEL_REVISION),
            ("tokenizer_revision", RUN.MODEL_REVISION), ("ctx", CTX),
            ("window", WINDOW),
        ):
            values = frame[column].drop_duplicates().tolist()
            if values != [expected]:
                _fail(f"{label}.{column} must be exactly {expected!r}, got {values!r}")
        for column in (
            "context_hash", "question_hash", "prompt_token_hash",
            "manifest_content_sha256", "manifest_file_sha256",
        ):
            _all_sha256(frame, column, label)

    for item_id in item_ids:
        prediction_rows = predictions.loc[
            predictions["item_id"].astype(str).eq(item_id)
        ]
        proxy_rows = proxy.loc[proxy["item_id"].astype(str).eq(item_id)]
        entry = manifest_by_id[item_id]
        item = data_by_id[item_id]
        expected_fields = {
            "group_id": entry["group_id"], "split": SPLIT,
            "domain": item["domain"], "sub_domain": item["sub_domain"],
            "difficulty": item["difficulty"], "length": item["length"],
            "context_hash": entry["context_hash"],
            "question_hash": hashlib.sha256(
                item["question"].strip().encode("utf-8")
            ).hexdigest(),
            "prompt_token_hash": entry["prompt_token_hash"],
            "input_tokens": int(entry["input_tokens"]),
            "n_prompt_tokens": int(entry["input_tokens"]),
        }
        for column, expected in expected_fields.items():
            for label, block in (
                ("prediction", prediction_rows), ("proxy", proxy_rows)
            ):
                if block[column].drop_duplicates().tolist() != [expected]:
                    _fail(
                        f"{label} {item_id}.{column} disagrees with manifest/dataset"
                    )
        for column in COMMON_COLUMNS:
            if prediction_rows[column].nunique(dropna=False) != 1:
                _fail(f"prediction arms differ for {item_id}.{column}")
            if proxy_rows[column].nunique(dropna=False) != 1:
                _fail(f"proxy candidates differ for {item_id}.{column}")
            if prediction_rows[column].iloc[0] != proxy_rows[column].iloc[0]:
                _fail(f"prediction/proxy provenance differs for {item_id}.{column}")

    predictions["B"] = _numeric(predictions, "B", "prediction")
    for column in (
        "arm_order", "choice_index", "ctx_len", "observed_queries", "maxb",
        "rot_seed",
    ):
        predictions[column] = _numeric(
            predictions, column, "prediction", integer=True
        )
    for column in (
        "choice_entropy", "choice_margin", "choice_max_probability",
        "bits_per_token", "evict_frac",
    ):
        predictions[column] = _numeric(predictions, column, "prediction")
    predictions["norm_correct"] = _bool(
        predictions, "norm_correct", "prediction"
    )
    expected_arm_order = predictions["arm"].astype(str).map(
        {arm: index for index, arm in enumerate(ARMS)}
    )
    if expected_arm_order.isna().any() or not predictions["arm_order"].equals(
        expected_arm_order.astype(np.int64)
    ):
        _fail("prediction arm_order disagrees with the frozen arm menu")
    if (~predictions["choice_index"].between(0, 3)).any():
        _fail("prediction choice_index must lie in 0..3")
    expected_choice = predictions["choice_index"].map(
        lambda value: "ABCD"[int(value)]
    )
    if not predictions["forced_choice"].astype(str).equals(expected_choice):
        _fail("prediction forced_choice disagrees with choice_index")
    if (
        (predictions["choice_entropy"] < -1e-6)
        | (predictions["choice_entropy"] > math.log(4.0) + 1e-6)
    ).any():
        _fail("prediction choice_entropy lies outside [0, log(4)]")
    if (
        (predictions["choice_margin"] < -1e-7)
        | (predictions["choice_margin"] > 1.0 + 1e-7)
    ).any():
        _fail("prediction choice_margin lies outside [0, 1]")
    if (
        (predictions["choice_max_probability"] < 0.25 - 1e-7)
        | (predictions["choice_max_probability"] > 1.0 + 1e-7)
    ).any():
        _fail("prediction choice_max_probability lies outside [0.25, 1]")
    if (
        predictions["choice_margin"]
        > predictions["choice_max_probability"] + 1e-7
    ).any():
        _fail("prediction choice margin exceeds its maximum probability")
    if (
        (predictions["evict_frac"] < 0.0)
        | (predictions["evict_frac"] > 1.0)
    ).any():
        _fail("prediction evict_frac lies outside [0, 1]")
    expected_ctx_len = predictions["input_tokens"] - 1 - WINDOW
    if not predictions["ctx_len"].equals(expected_ctx_len.astype(np.int64)):
        _fail("prediction ctx_len is inconsistent with input_tokens/window")
    if not predictions["observed_queries"].eq(WINDOW).all():
        _fail("prediction observed_queries must equal the protected window")
    if (
        not predictions["maxb"].eq(MAXB).all()
        or not predictions["rot_seed"].eq(0).all()
    ):
        _fail("prediction maxb/rotation seed drifted")
    if not predictions["norm_correct"].all():
        _fail("prediction norm correction must be enabled")
    for column in (
        "scaffold_token_hash", "choice_branch_ids_hash", "allocation_id"
    ):
        _all_sha256(predictions, column, "prediction")
    if not predictions["scaffold_token_hash"].eq(SCAFFOLD_TOKEN_HASH).all():
        _fail("prediction scaffold token hash drifted")
    if not predictions["choice_branch_ids_hash"].eq(
        CHOICE_BRANCH_IDS_HASH
    ).all():
        _fail("prediction choice branch IDs drifted")

    fp = predictions.loc[predictions["arm"].astype(str).eq("fp")]
    compressed = predictions.loc[~predictions["arm"].astype(str).eq("fp")]
    if (
        not fp["B"].eq(0.0).all()
        or not fp["bits_per_token"].eq(16.0).all()
    ):
        _fail("FP prediction rows must use B=0 and 16 bits/token")
    if (
        not fp["evict_frac"].eq(0.0).all()
        or not fp["allocation_id"].eq(FP_ALLOCATION_ID).all()
    ):
        _fail("FP allocation provenance is not the frozen fp16 sentinel")
    if not compressed["B"].eq(BUDGET).all():
        _fail("compressed prediction rows must use B=2")
    if (
        (compressed["bits_per_token"] < 0.0)
        | (compressed["bits_per_token"] > BUDGET + 1e-7)
    ).any():
        _fail("compressed bits_per_token lies outside the feasible B=2 budget")
    if compressed["allocation_id"].eq(FP_ALLOCATION_ID).any():
        _fail("compressed allocation cannot equal the FP sentinel")
    uniform = compressed.loc[compressed["arm"].astype(str).eq("uniform")]
    if len(uniform) != EXPECTED_ROWS:
        _fail("uniform arm does not cover every development item exactly once")
    if not np.isclose(
        uniform["bits_per_token"].to_numpy(dtype=float), BUDGET,
        rtol=0.0, atol=1e-7,
    ).all():
        _fail("uniform rows must spend exactly B=2")
    if not np.isclose(
        uniform["evict_frac"].to_numpy(dtype=float), 0.0,
        rtol=0.0, atol=1e-12,
    ).all():
        _fail("uniform rows must not evict tokens")
    for row in uniform.itertuples(index=False):
        expected_allocation = RUN.QUAL.expected_uniform_allocation_id(int(row.ctx_len))
        _same(
            expected_allocation, str(row.allocation_id),
            f"uniform deterministic allocation for {row.item_id}",
        )

    proxy["B"] = _numeric(proxy, "B", "proxy")
    proxy["candidate_order"] = _numeric(
        proxy, "candidate_order", "proxy", integer=True
    )
    for column in SCAFFOLD_KL_COLUMNS + ("scaffold_mean_kl",):
        proxy[column] = _numeric(proxy, column, "proxy")
        if (proxy[column] < -1e-6).any():
            _fail(
                f"proxy.{column} must be nonnegative apart from float32 roundoff"
            )
    if not proxy["B"].eq(BUDGET).all():
        _fail("proxy rows must use B=2")
    expected_candidate_order = proxy["candidate"].astype(str).map(
        {candidate: index for index, candidate in enumerate(CANDIDATES)}
    )
    if (
        expected_candidate_order.isna().any()
        or not proxy["candidate_order"].equals(
            expected_candidate_order.astype(np.int64)
        )
    ):
        _fail("proxy candidate_order disagrees with the frozen candidate menu")
    recomputed_mean = proxy[list(SCAFFOLD_KL_COLUMNS)].mean(axis=1)
    if not np.isclose(
        proxy["scaffold_mean_kl"].to_numpy(), recomputed_mean.to_numpy(),
        rtol=1e-6, atol=1e-7,
    ).all():
        _fail("proxy scaffold_mean_kl disagrees with its five saved positions")
    if not proxy["proxy_rule_version"].eq(PROXY_RULE_VERSION).all():
        _fail("proxy rule version drifted")
    for column in (
        "scaffold_token_hash", "choice_branch_ids_hash", "allocation_id"
    ):
        _all_sha256(proxy, column, "proxy")
    if not proxy["scaffold_token_hash"].eq(SCAFFOLD_TOKEN_HASH).all():
        _fail("proxy scaffold token hash drifted")
    if not proxy["choice_branch_ids_hash"].eq(CHOICE_BRANCH_IDS_HASH).all():
        _fail("proxy choice branch IDs drifted")
    for column in (
        "selected_policy", "fp_canonical_choice", "candidate_canonical_choice"
    ):
        if proxy[column].isna().any():
            _fail(f"proxy.{column} contains a missing value")
    if not proxy["fp_canonical_choice"].astype(str).isin(tuple("ABCD")).all():
        _fail("proxy fp_canonical_choice must lie in A..D")
    if not proxy["candidate_canonical_choice"].astype(str).isin(
        tuple("ABCD")
    ).all():
        _fail("proxy candidate_canonical_choice must lie in A..D")

    prediction_lookup = predictions.set_index(["item_id", "arm"])
    for item_id in item_ids:
        block = proxy.loc[proxy["item_id"].astype(str).eq(item_id)]
        values = {
            str(row.candidate): float(row.scaffold_mean_kl)
            for row in block.itertuples(index=False)
        }
        selected = select_scaffold_policy(values)
        if block["selected_policy"].drop_duplicates().tolist() != [selected]:
            _fail(
                f"proxy selected_policy was not recomputed correctly for {item_id}"
            )
        fp_choice = str(
            prediction_lookup.loc[(item_id, "fp"), "forced_choice"]
        )
        for row in block.itertuples(index=False):
            candidate = str(row.candidate)
            prediction_row = prediction_lookup.loc[(item_id, candidate)]
            if str(row.fp_canonical_choice) != fp_choice:
                _fail(
                    f"proxy FP choice disagrees with predictions for {item_id}"
                )
            if str(row.candidate_canonical_choice) != str(
                prediction_row["forced_choice"]
            ):
                _fail(
                    f"proxy candidate choice disagrees with predictions for "
                    f"{item_id}/{candidate}"
                )
            if str(row.allocation_id) != str(
                prediction_row["allocation_id"]
            ):
                _fail(
                    f"proxy allocation ID disagrees with predictions for "
                    f"{item_id}/{candidate}"
                )


def _scored_table(
    predictions: pd.DataFrame, proxy: pd.DataFrame, *,
    development: Sequence[Mapping[str, Any]],
    data_by_id: Mapping[str, Mapping[str, str]],
) -> tuple[pd.DataFrame, str, dict[str, float]]:
    """Join labels in memory and return item-level binary outcomes."""
    item_ids = [str(row["id"]) for row in development]
    groups = {str(row["id"]): str(row["group_id"]) for row in development}
    choices = predictions.pivot(
        index="item_id", columns="arm", values="forced_choice"
    ).reindex(index=item_ids, columns=ARMS)
    if choices.isna().any().any():
        _fail("validated predictions could not be pivoted over the full arm menu")
    gold = pd.Series(
        {item_id: str(data_by_id[item_id]["answer"]) for item_id in item_ids},
        name="gold",
    )
    correct = choices.eq(gold, axis=0)
    candidate_accuracy = {
        candidate: float(correct[candidate].mean())
        for candidate in CANDIDATES
    }
    fixed = CANDIDATES[0]
    fixed_accuracy = candidate_accuracy[fixed]
    for candidate in CANDIDATES[1:]:
        accuracy = candidate_accuracy[candidate]
        if accuracy > fixed_accuracy:
            fixed, fixed_accuracy = candidate, accuracy

    selected = (
        proxy[["item_id", "selected_policy"]]
        .drop_duplicates()
        .set_index("item_id")["selected_policy"]
        .reindex(item_ids)
    )
    selected_correct = np.fromiter(
        (
            bool(correct.loc[item_id, str(selected.loc[item_id])])
            for item_id in item_ids
        ),
        dtype=bool,
        count=len(item_ids),
    )
    table = pd.DataFrame({
        "item_id": item_ids,
        "group_id": [groups[item_id] for item_id in item_ids],
        "fp": correct["fp"].to_numpy(dtype=bool),
        "fixed": correct[fixed].to_numpy(dtype=bool),
        "oracle": correct[list(CANDIDATES)].any(axis=1).to_numpy(dtype=bool),
        "selector": selected_correct,
    })
    return table, fixed, candidate_accuracy


def component_bootstrap(
    table: pd.DataFrame, component_order: Sequence[str], *,
    draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    """Paired component bootstrap with micro, rather than component, means."""
    metric_columns = ("fp", "fixed", "oracle", "selector")
    if draws <= 0:
        raise ValueError("bootstrap draw count must be positive")
    order = tuple(component_order)
    if len(order) != len(set(order)):
        raise ValueError("component order contains duplicates")
    got_groups = set(table["group_id"].astype(str))
    if got_groups != set(order):
        raise ValueError("bootstrap table does not cover the component order exactly")
    grouped = table.assign(
        **{column: table[column].astype(float) for column in metric_columns}
    ).groupby("group_id", sort=False)
    sizes = grouped.size().reindex(order).to_numpy(dtype=np.int64)
    sums = {
        column: grouped[column].sum().reindex(order).to_numpy(dtype=float)
        for column in metric_columns
    }
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled = rng.integers(0, len(order), size=(draws, len(order)))
    denominators = sizes[sampled].sum(axis=1).astype(float)
    out = {
        column: values[sampled].sum(axis=1) / denominators
        for column, values in sums.items()
    }
    out["H"] = out["oracle"] - out["fixed"]
    out["G"] = out["selector"] - out["fixed"]
    return out


def _q05(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.05, method="linear"))


def _half_delta(
    table: pd.DataFrame, groups: frozenset[str], numerator: str,
) -> float:
    rows = table.loc[table["group_id"].astype(str).isin(groups)]
    difference = int(rows[numerator].astype(bool).sum()) - int(
        rows["fixed"].astype(bool).sum()
    )
    return difference / len(rows)


def analyze_frames(
    predictions: pd.DataFrame, proxy: pd.DataFrame, *,
    development: Sequence[Mapping[str, Any]],
    data_by_id: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    component_order, halves = frozen_halves(development)
    table, fixed, candidate_accuracy = _scored_table(
        predictions, proxy, development=development, data_by_id=data_by_id
    )
    bootstrap = component_bootstrap(table, component_order)
    item_ids = [str(row["id"]) for row in development]
    choices = predictions.pivot(
        index="item_id", columns="arm", values="forced_choice"
    ).reindex(index=item_ids, columns=ARMS)
    vector_classes: dict[tuple[str, ...], list[str]] = {}
    for arm in ARMS:
        vector = tuple(choices[arm].astype(str).tolist())
        vector_classes.setdefault(vector, []).append(arm)
    equivalence_classes = list(vector_classes.values())
    fp_vector = choices["fp"].astype(str)
    agreement_with_fp = {
        arm: int(choices[arm].astype(str).eq(fp_vector).sum()) for arm in CANDIDATES
    }

    fp_accuracy = float(table["fp"].mean())
    fixed_accuracy = float(table["fixed"].mean())
    oracle_accuracy = float(table["oracle"].mean())
    headroom_count = int(table["oracle"].sum()) - int(table["fixed"].sum())
    H = headroom_count / len(table)
    fp_q05 = _q05(bootstrap["fp"])
    fixed_q05 = _q05(bootstrap["fixed"])
    H_q05 = _q05(bootstrap["H"])
    H_halves = tuple(_half_delta(table, groups, "oracle") for groups in halves)

    competence_gates = {
        "fp_point_gate": fp_accuracy >= 0.50,
        "fp_bootstrap_chance_gate": fp_q05 > 0.25,
        "fixed_point_gate": 0.30 <= fixed_accuracy <= 0.80,
        "fixed_bootstrap_chance_gate": fixed_q05 > 0.25,
    }
    h_gates = {
        "H_point_gate": H >= 0.10,
        "H_bootstrap_gate": H_q05 > 0.0,
        "H_half1_gate": H_halves[0] >= 0.05,
        "H_half2_gate": H_halves[1] >= 0.05,
    }
    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "split": SPLIT,
        "n_items": EXPECTED_ROWS,
        "n_components": EXPECTED_COMPONENTS,
        "prediction_rows": len(predictions),
        "proxy_rows": len(proxy),
        "fixed_policy": fixed,
        "answer_vector_equivalence_class_count": len(equivalence_classes),
        "answer_vector_equivalence_classes_json": json.dumps(
            equivalence_classes, separators=(",", ":")
        ),
        "candidate_agreement_with_fp_json": json.dumps(
            agreement_with_fp, separators=(",", ":")
        ),
        **{
            f"candidate_accuracy_{candidate}": candidate_accuracy[candidate]
            for candidate in CANDIDATES
        },
        "fp_accuracy": fp_accuracy,
        "fixed_accuracy": fixed_accuracy,
        "oracle_accuracy": oracle_accuracy,
        "H": H,
        "fp_bootstrap_q05": fp_q05,
        "fixed_bootstrap_q05": fixed_q05,
        "H_bootstrap_q05": H_q05,
        "H_half1": H_halves[0],
        "H_half2": H_halves[1],
        **competence_gates,
        "competence_gate": all(competence_gates.values()),
        **h_gates,
        "opportunity_gate": all(h_gates.values()),
        "selector_accuracy": None,
        "G": None,
        "G_over_H": None,
        "G_bootstrap_q05": None,
        "G_half1": None,
        "G_half2": None,
        "G_point_gate": None,
        "G_capture_gate": None,
        "G_bootstrap_gate": None,
        "G_half1_gate": None,
        "G_half2_gate": None,
        "proxy_transfer_gate": None,
        "selector_rescue": None,
        "selector_harm": None,
        "selector_both_correct": None,
        "selector_both_wrong": None,
    }
    if not summary["competence_gate"]:
        summary["decision"] = "stop_v6_invalid_operating_point"
        return summary
    if not summary["opportunity_gate"]:
        summary["decision"] = "stop_v6_no_opportunity"
        return summary

    selector_accuracy = float(table["selector"].mean())
    gain_count = int(table["selector"].sum()) - int(table["fixed"].sum())
    G = gain_count / len(table)
    ratio = gain_count / headroom_count
    G_q05 = _q05(bootstrap["G"])
    G_halves = tuple(_half_delta(table, groups, "selector") for groups in halves)
    g_gates = {
        "G_point_gate": G >= 0.05,
        "G_capture_gate": ratio >= 0.50,
        "G_bootstrap_gate": G_q05 >= 0.0,
        "G_half1_gate": G_halves[0] >= 0.0,
        "G_half2_gate": G_halves[1] >= 0.0,
    }
    fixed_correct = table["fixed"].astype(bool)
    selector_correct = table["selector"].astype(bool)
    summary.update({
        "selector_accuracy": selector_accuracy,
        "G": G,
        "G_over_H": ratio,
        "G_bootstrap_q05": G_q05,
        "G_half1": G_halves[0],
        "G_half2": G_halves[1],
        **g_gates,
        "proxy_transfer_gate": all(g_gates.values()),
        "selector_rescue": int((~fixed_correct & selector_correct).sum()),
        "selector_harm": int((fixed_correct & ~selector_correct).sum()),
        "selector_both_correct": int((fixed_correct & selector_correct).sum()),
        "selector_both_wrong": int((~fixed_correct & ~selector_correct).sum()),
    })
    summary["decision"] = (
        "advance_v6_confirmation"
        if summary["proxy_transfer_gate"]
        else "reject_v6_scaffold_proxy"
    )
    return summary


def load_pair(
    prediction_path: str | Path,
    proxy_path: str | Path,
    *,
    manifest_path: str | Path,
    source_manifest_path: str | Path,
    dataset_path: str | Path,
    model_source: str | Path,
    config_path: str | Path,
    qualification_lock_path: str | Path,
    analyze: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    prediction_path = _regular(prediction_path, "prediction parquet")
    proxy_path = _regular(proxy_path, "proxy parquet")
    manifest_path = _regular(manifest_path, "Qwen manifest")
    source_manifest_path = _regular(source_manifest_path, "source manifest")
    dataset_path = _regular(dataset_path, "dataset")
    model_source_path = Path(model_source).resolve()
    config_path = _regular(config_path, "model config")
    if prediction_path.suffix != ".parquet" or proxy_path.suffix != ".parquet":
        _fail("both result artifacts must use the .parquet suffix")
    prediction_sidecar_path = _regular(
        prediction_path.with_suffix(".json"), "prediction sidecar"
    )
    proxy_sidecar_path = _regular(
        proxy_path.with_suffix(".json"), "proxy sidecar"
    )

    frozen_lock, frozen_lock_path = authenticate_qualification_lock(
        qualification_lock_path
    )
    try:
        runner_lock, qualification_attestation = RUN.authenticate_qualification_lock(
            frozen_lock_path,
            dataset_path=dataset_path,
            source_manifest_path=source_manifest_path,
            manifest_path=manifest_path,
            config_path=config_path,
            model_source=model_source_path,
        )
    except (OSError, ValueError, RUN.QwenDevelopmentRunnerError) as exc:
        _fail(f"runner qualification-lock authentication failed: {exc}")
    _same(frozen_lock, runner_lock, "reader/runner qualification lock")
    snapshot_attestation = QA.authenticate_snapshot(model_source_path)
    _same(
        qualification_attestation["snapshot"], snapshot_attestation,
        "qualification/development snapshot attestation",
    )

    tokenizer = QA.load_tokenizer(str(model_source_path))
    manifest, manifest_file_hash, development = authenticate_manifest(
        manifest_path,
        dataset_path=dataset_path,
        source_manifest_path=source_manifest_path,
        tokenizer=tokenizer,
    )
    data_by_id = authenticate_dataset(dataset_path, development)
    try:
        predictions = pd.read_parquet(prediction_path)
        proxy = pd.read_parquet(proxy_path)
    except Exception as exc:
        _fail(f"cannot read parquet artifacts: {exc}")
    prediction_sidecar = _load_json(prediction_sidecar_path, "prediction sidecar")
    proxy_sidecar = _load_json(proxy_sidecar_path, "proxy sidecar")
    validate_sidecars(
        prediction_sidecar,
        proxy_sidecar,
        prediction_path=prediction_path,
        proxy_path=proxy_path,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_file_hash=manifest_file_hash,
        config_path=config_path,
        qualification_attestation=qualification_attestation,
        snapshot_attestation=snapshot_attestation,
    )
    validate_frames(
        predictions,
        proxy,
        manifest=manifest,
        manifest_file_hash=manifest_file_hash,
        development=development,
        data_by_id=data_by_id,
    )
    summary = (
        analyze_frames(
            predictions, proxy, development=development, data_by_id=data_by_id
        )
        if analyze
        else {}
    )
    return summary, predictions, proxy


def _lock_content_hash(value: Mapping[str, Any]) -> str:
    return _content_sha256(value)


def build_confirmation_lock(
    summary: Mapping[str, Any], *,
    prediction_path: str | Path,
    proxy_path: str | Path,
    manifest_path: str | Path,
    source_manifest_path: str | Path,
    dataset_path: str | Path,
    config_path: str | Path,
    qualification_lock_path: str | Path,
    model_source: str | Path,
    runner_path: str | Path,
) -> dict[str, Any] | None:
    if summary.get("decision") != "advance_v6_confirmation":
        return None
    for gate in ("competence_gate", "opportunity_gate", "proxy_transfer_gate"):
        if summary.get(gate) is not True:
            _fail(f"advancing summary does not pass {gate}")
    prediction_path = _regular(prediction_path, "confirmation-lock prediction parquet")
    proxy_path = _regular(proxy_path, "confirmation-lock proxy parquet")
    prediction_sidecar_path = _regular(
        prediction_path.with_suffix(".json"), "confirmation-lock prediction sidecar"
    )
    proxy_sidecar_path = _regular(
        proxy_path.with_suffix(".json"), "confirmation-lock proxy sidecar"
    )
    manifest_path = _regular(manifest_path, "confirmation-lock Qwen manifest")
    source_manifest_path = _regular(source_manifest_path, "confirmation-lock source manifest")
    dataset_path = _regular(dataset_path, "confirmation-lock dataset")
    config_path = _regular(config_path, "confirmation-lock config")
    qualification_lock_path = _regular(
        qualification_lock_path, "confirmation-lock qualification authorization"
    )
    runner_path = _regular(runner_path, "confirmation-lock runner")
    reader_path = _regular(__file__, "confirmation-lock reader")
    _same(Path(RUN.__file__).resolve(), runner_path, "confirmation-lock runner path")
    prediction_sidecar = _load_json(prediction_sidecar_path, "prediction sidecar")
    proxy_sidecar = _load_json(proxy_sidecar_path, "proxy sidecar")
    _same(
        prediction_sidecar["executed_source_sha256"],
        proxy_sidecar["executed_source_sha256"],
        "confirmation-lock paired source seals",
    )
    source_seal = dict(prediction_sidecar["executed_source_sha256"])
    _same(
        RUN.executed_source_hashes(config_path), source_seal,
        "confirmation-lock live development source seal",
    )
    qualification_lock, qualification_attestation = RUN.authenticate_qualification_lock(
        qualification_lock_path,
        dataset_path=dataset_path,
        source_manifest_path=source_manifest_path,
        manifest_path=manifest_path,
        config_path=config_path,
        model_source=model_source,
    )
    frozen_lock, _ = authenticate_qualification_lock(qualification_lock_path)
    _same(frozen_lock, qualification_lock, "confirmation-lock qualification authorization")
    snapshot = QA.authenticate_snapshot(model_source)
    _same(qualification_attestation["snapshot"], snapshot, "confirmation-lock snapshot")
    manifest = _load_json(manifest_path, "Qwen manifest")
    confirmation = dict(manifest["split_counts"]["confirmation"])
    confirmation.update({
        "half_component_counts": [22, 23],
        "half_row_counts": [22, 23],
        "half_id_sha256": [
            "b0e905a214648f35d9c38a615c5e0d6811a07f69d9558209e224e023efaa8e08",
            "c95acca30821a88fa4c6a8ff6a885c9a05c867fdbb00c2b9be10659f84753d53",
        ],
    })
    thresholds = {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_rng": "numpy.Generator(PCG64)",
        "bootstrap_quantile": 0.05,
        "bootstrap_quantile_method": "linear",
        "fp_accuracy_min": 0.50,
        "fp_q05_strictly_above": 0.25,
        "fixed_accuracy_interval": [0.30, 0.80],
        "fixed_q05_strictly_above": 0.25,
        "H_min": 0.10,
        "H_q05_strictly_above": 0.0,
        "H_each_half_min": 0.05,
        "G_min": 0.05,
        "G_over_H_min": 0.50,
        "G_q05_min": 0.0,
        "G_each_half_min": 0.0,
    }
    lock: dict[str, Any] = {
        "lock_version": CONFIRMATION_LOCK_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "runner_version": RUNNER_VERSION,
        "reader_version": READER_VERSION,
        "task_version": RUN.TASK_VERSION,
        "endpoint_version": ENDPOINT_VERSION,
        "proxy_rule_version": PROXY_RULE_VERSION,
        "decision": "advance_v6_confirmation",
        "qualification_authorization": {
            "path": str(qualification_lock_path),
            "file_sha256": _sha256_file(qualification_lock_path),
            "content_sha256": qualification_lock["content_sha256"],
            "qualification_predictions_sha256": qualification_attestation["qualification_predictions_sha256"],
            "qualification_sidecar_sha256": qualification_attestation["qualification_sidecar_sha256"],
        },
        "development": {
            "prediction_parquet": str(prediction_path),
            "prediction_parquet_sha256": _sha256_file(prediction_path),
            "prediction_sidecar": str(prediction_sidecar_path),
            "prediction_sidecar_sha256": _sha256_file(prediction_sidecar_path),
            "proxy_parquet": str(proxy_path),
            "proxy_parquet_sha256": _sha256_file(proxy_path),
            "proxy_sidecar": str(proxy_sidecar_path),
            "proxy_sidecar_sha256": _sha256_file(proxy_sidecar_path),
            "fixed_policy": summary["fixed_policy"],
            "analysis": {
                str(key): _python_scalar(value) for key, value in summary.items()
            },
        },
        "provenance": {
            "dataset_repo": LB.DATASET_REPO,
            "dataset_revision": LB.DATASET_REVISION,
            "dataset_sha256": LB.DATASET_SHA256,
            "dataset_bytes": LB.DATASET_BYTES,
            "official_prompt_sha256": LB.OFFICIAL_PROMPT_SHA256,
            "source_manifest": str(source_manifest_path),
            "source_manifest_content_sha256": QA.SOURCE_MANIFEST_CONTENT_SHA256,
            "source_manifest_file_sha256": _sha256_file(source_manifest_path),
            "qwen_manifest": str(manifest_path),
            "qwen_manifest_version": QA.MANIFEST_VERSION,
            "qwen_manifest_content_sha256": MANIFEST_CONTENT_SHA256,
            "qwen_manifest_file_sha256": _sha256_file(manifest_path),
            "config": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "runner": str(runner_path),
            "runner_sha256": _sha256_file(runner_path),
            "reader": str(reader_path),
            "reader_sha256": _sha256_file(reader_path),
            "executed_source_sha256": source_seal,
            "snapshot_revision": snapshot["revision"],
            "snapshot_file_count": snapshot["file_count"],
            "snapshot_inventory_sha256": snapshot["inventory_sha256"],
            "snapshot_metadata_sha256": dict(snapshot["metadata_sha256"]),
        },
        "partitions": {
            "development": {
                **dict(manifest["split_counts"][SPLIT]),
                "half_component_counts": list(HALF_COMPONENT_COUNTS),
                "half_row_counts": list(HALF_ROW_COUNTS),
                "half_id_sha256": list(HALF_ID_SHA256),
            },
            "confirmation": confirmation,
            "component_order": "min_sha256_split_namespace_context_hash_then_component_id_v1",
        },
        "execution": {
            "model": MODEL,
            "model_id": MODEL_ID,
            "model_revision": RUN.MODEL_REVISION,
            "tokenizer_revision": RUN.TOKENIZER_REVISION,
            "chat_template_sha256": QA.CHAT_TEMPLATE_SHA256,
            "enable_thinking": False,
            "ctx": CTX,
            "window": WINDOW,
            "budget": RUN.BUDGET,
            "allocator_budget_rule": "feasible",
            "uniform_allocation_contract": "48_layers_x_4_kv_x_ctx_len_all_uint8_2",
            "dtype": RUN.DTYPE_NAME,
            "bit_list": list(RUN.BIT_LIST),
            "cascade_bits": RUN.CASCADE_BITS,
            "maxb": MAXB,
            "rot_seed": RUN.ROT_SEED,
            "norm_correct": RUN.NORM_CORRECT,
            "chunk": RUN.CHUNK,
            "n_layers": RUN.N_LAYERS,
            "n_attention_heads": RUN.N_ATTENTION_HEADS,
            "n_kv_heads": RUN.N_KV_HEADS,
            "head_dim": RUN.HEAD_DIM,
            "attn_impl": RUN.C.IMPL,
            "raw_arms": list(RAW_ARMS),
            "arms": list(ARMS),
            "candidates": list(CANDIDATES),
            "fixed_policy": summary["fixed_policy"],
            "decode": DECODE,
            "compress_at": COMPRESS_AT,
            "proxy_rule_version": PROXY_RULE_VERSION,
            "proxy_dtype": "float32",
            "scaffold_token_ids": list(SCAFFOLD_TOKEN_IDS),
            "choice_branch_ids": CHOICE_BRANCH_IDS,
        },
        "thresholds": thresholds,
    }
    lock["content_sha256"] = _lock_content_hash(lock)
    _same(lock["content_sha256"], _lock_content_hash(lock), "confirmation lock content hash")
    return lock


def write_confirmation_lock(path: str | Path, lock: Mapping[str, Any] | None) -> None:
    destination = Path(path)
    if destination.is_symlink():
        _fail(f"refusing confirmation-lock symlink: {destination}")
    if lock is None:
        if destination.exists():
            _fail(f"refusing stale confirmation lock after non-advance: {destination}")
        return
    expected = dict(lock)
    _same(
        _lock_content_hash(expected), expected.get("content_sha256"),
        "confirmation lock content SHA-256",
    )
    if destination.exists():
        current = _load_json(destination, "confirmation lock")
        if current != expected:
            _fail(f"refusing to overwrite different confirmation lock: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(expected, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    authenticated = _load_json(destination, "confirmation lock")
    _same(expected, authenticated, "confirmation lock round trip")
    _same(
        authenticated["content_sha256"], _lock_content_hash(authenticated),
        "written confirmation lock content hash",
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "suppressed"
    number = float(value)
    return "nan" if not math.isfinite(number) else f"{number:.3f}"


def print_summary(summary: Mapping[str, Any]) -> None:
    print("V6 Qwen LongBench-v2 forced-choice development (frozen section 3K)")
    print("provenance      PASS (pinned dataset/manifest/model/prompt/split)")
    print(
        f"rows            {summary['prediction_rows']} predictions, "
        f"{summary['proxy_rows']} proxy; {summary['n_items']} items / "
        f"{summary['n_components']} components"
    )
    print(
        f"fixed policy    {summary['fixed_policy']} "
        f"accuracy {_fmt(summary['fixed_accuracy'])}"
    )
    print(
        "answer vectors  "
        f"{summary['answer_vector_equivalence_class_count']} classes "
        f"{summary['answer_vector_equivalence_classes_json']}"
    )
    print(
        f"competence      FP {_fmt(summary['fp_accuracy'])} "
        f"(q05 {_fmt(summary['fp_bootstrap_q05'])}), "
        f"F* {_fmt(summary['fixed_accuracy'])} "
        f"(q05 {_fmt(summary['fixed_bootstrap_q05'])}) "
        f"[{'PASS' if summary['competence_gate'] else 'FAIL'}]"
    )
    print(
        f"opportunity H   {_fmt(summary['H'])}, "
        f"q05 {_fmt(summary['H_bootstrap_q05'])}, "
        f"halves {_fmt(summary['H_half1'])}/{_fmt(summary['H_half2'])} "
        f"[{'PASS' if summary['opportunity_gate'] else 'FAIL'}]"
    )
    if summary["G"] is None:
        print("proxy G         suppressed (sequential H gate did not pass)")
    else:
        print(
            f"proxy G         {_fmt(summary['G'])}, "
            f"G/H {_fmt(summary['G_over_H'])}, "
            f"q05 {_fmt(summary['G_bootstrap_q05'])}, "
            f"halves {_fmt(summary['G_half1'])}/{_fmt(summary['G_half2'])} "
            f"[{'PASS' if summary['proxy_transfer_gate'] else 'FAIL'}]"
        )
        print(
            f"selector counts rescue {summary['selector_rescue']}, "
            f"harm {summary['selector_harm']}, "
            f"both-correct {summary['selector_both_correct']}, "
            f"both-wrong {summary['selector_both_wrong']}"
        )
    print(f"decision        {summary['decision']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="V6 development forced-choice parquet")
    parser.add_argument("proxy", help="V6 development scaffold-proxy parquet")
    parser.add_argument("--manifest", required=True, help="frozen gold-free Qwen manifest")
    parser.add_argument("--source-manifest", required=True, help="frozen source partition manifest")
    parser.add_argument("--dataset", required=True, help="pinned LongBench-v2 data JSON")
    parser.add_argument("--model-source", required=True, help="pinned local Qwen snapshot")
    parser.add_argument("--qualification-lock", required=True, help="advancing V6 qualification lock")
    parser.add_argument(
        "--config", default=str(ROOT / "h0_measurement/models.yaml"),
        help="frozen model registry",
    )
    parser.add_argument("--csv", help="write the one-row analysis summary CSV")
    parser.add_argument(
        "--lock",
        help=(
            "write an idempotent confirmation lock only on advance; refuse a "
            "pre-existing stale lock after any non-advance decision"
        ),
    )
    parser.add_argument(
        "--runner",
        default=str(ROOT / "h0_measurement/run_longbench_v2_qwen_development.py"),
        help="V6 Qwen development runner source to authenticate",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="authenticate artifacts and provenance without joining labels",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.validate_only and (args.csv or args.lock):
            _fail("--validate-only forbids --csv and --lock")
        summary, predictions, proxy = load_pair(
            args.predictions,
            args.proxy,
            manifest_path=args.manifest,
            source_manifest_path=args.source_manifest,
            dataset_path=args.dataset,
            model_source=args.model_source,
            config_path=args.config,
            qualification_lock_path=args.qualification_lock,
            analyze=not args.validate_only,
        )
        if args.validate_only:
            print(
                f"PASS V6 Qwen development artifacts: {len(predictions)} predictions, "
                f"{len(proxy)} proxy rows; {EXPECTED_ROWS} items / "
                f"{EXPECTED_COMPONENTS} components"
            )
            return 0
        if args.csv:
            destination = Path(args.csv)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([summary]).to_csv(destination, index=False)
        if args.lock:
            lock = build_confirmation_lock(
                summary,
                prediction_path=args.predictions,
                proxy_path=args.proxy,
                manifest_path=args.manifest,
                source_manifest_path=args.source_manifest,
                dataset_path=args.dataset,
                config_path=args.config,
                qualification_lock_path=args.qualification_lock,
                model_source=args.model_source,
                runner_path=args.runner,
            )
            write_confirmation_lock(args.lock, lock)
            if lock is not None:
                print(f"lock            {Path(args.lock).resolve()}")
        print_summary(summary)
        return 0
    except (
        LongBenchForcedChoiceError,
        QA.QwenManifestError,
        RUN.QwenDevelopmentRunnerError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
