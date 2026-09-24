#!/usr/bin/env python3
"""Strict reader and advance-lock writer for V7 Qwen qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h0_measurement import audit_longbench_v2_qwen_v7 as QA  # noqa: E402
from h0_measurement import run_longbench_v2_qwen_v7_qualification as RUN  # noqa: E402
from sievelib import policy_diagnostic as PD  # noqa: E402
from sievelib import tasks_longbench_v2 as LB  # noqa: E402


READER_VERSION = "longbench_v2_qwen_v7_qualification_reader_v1"
LOCK_VERSION = "longbench_v2_sieve_v7_qwen131072_qualification_lock_v1"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
BOOTSTRAP_QUANTILE = 0.05
FP_ACCURACY_MIN = 0.50
UNIFORM_ACCURACY_MIN = 0.30
UNIFORM_ACCURACY_MAX = 0.80
CHANCE = 0.25
DEVELOPMENT_ARMS = (
    "fp", "uniform", "tail_evict", "tail_interior",
    "structured_evict", "structured_interior",
)
DEVELOPMENT_CANDIDATES = DEVELOPMENT_ARMS[1:]
OLD_MENU = ("uniform", "tail_evict", "tail_interior")
STRUCTURED_MENU = ("structured_evict", "structured_interior")
FP_ALLOCATION_ID = hashlib.sha256(b"fp16").hexdigest()

SIDECAR_FIELDS = {
    "artifact", "runner_version", "protocol_version", "task_version",
    "endpoint_version", "dataset_repo", "dataset_revision", "dataset_sha256",
    "dataset_bytes", "official_code_repo", "official_code_revision",
    "official_prompt_sha256", "legacy_manifest", "legacy_manifest_version",
    "legacy_manifest_content_sha256", "legacy_manifest_file_sha256", "manifest",
    "manifest_version", "manifest_content_sha256", "manifest_file_sha256",
    "split", "split_count", "split_components", "split_ids_sha256",
    "split_id_token_sha256", "split_id_prompt_sha256",
    "split_id_positions_sha256", "split_ordered_id_sha256",
    "within_split_order", "component_sampling",
    "model", "model_id", "model_revision", "tokenizer_revision",
    "chat_template_sha256", "add_generation_prompt", "enable_thinking",
    "transformers_version", "tokenizers_version", "torch_version",
    "numpy_version", "pandas_version", "pyarrow_version",
    "fastparquet_version", "parquet_engine",
    "model_snapshot_attestation_scope", "model_snapshot_inventory_revision",
    "model_snapshot_symlink_file_count",
    "model_snapshot_symlink_inventory_sha256",
    "model_snapshot_pinned_metadata_sha256", "ctx",
    "max_prompt_tokens", "window", "budget", "dtype",
    "bit_list", "maxb", "rot_seed", "norm_correct", "chunk",
    "n_layers", "n_attention_heads", "n_kv_heads", "head_dim",
    "uniform_allocation_contract", "qualification_attn_impl",
    "structured_policy_impl", "structured_capture_version",
    "structured_policy_source", "structured_policy_source_sha256",
    "device_placements", "decode", "temperature", "compress_at", "arms",
    "allocation_id_algorithm", "scaffold_token_ids", "scaffold_token_hash",
    "choice_branch_ids", "choice_branch_ids_hash", "choice_position",
    "no_truncation", "no_free_generation", "no_raw_logits", "no_labels",
    "source_ledger", "source_ledger_version",
    "source_ledger_content_sha256", "source_ledger_file_sha256",
    "executed_source_sha256", "runner_source", "runner_source_sha256",
    "parquet", "parquet_sha256", "rows", "expected_rows", "row_key",
    "elapsed_seconds",
}
FORBIDDEN_RESULT_COLUMNS = {
    "answer", "gold", "gold_answer", "label", "correct", "correctness",
    "score", "response", "parsed_answer", "logit", "logits", "raw_logits",
    "probability", "probabilities", "probs", "choice_logits", "choice_probs",
}


class QwenQualificationReaderError(ValueError):
    """A V7 qualification artifact or provenance check failed."""


def _fail(message: str) -> None:
    raise QwenQualificationReaderError(message)


def _same(expected: Any, actual: Any, description: str) -> None:
    if isinstance(actual, (np.integer, np.floating, np.bool_)):
        actual = actual.item()
    if expected != actual:
        _fail(f"{description}: expected {expected!r}, got {actual!r}")


def _regular(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        _fail(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        _fail(f"{label} is not a regular file: {resolved}")
    return resolved


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
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


def _check_source_hashes(
    sidecar: Mapping[str, Any], config_path: Path, source_ledger_path: Path
) -> None:
    observed = sidecar.get("executed_source_sha256")
    if not isinstance(observed, dict) or not observed:
        _fail("sidecar executed_source_sha256 must be a nonempty object")
    expected = RUN.executed_source_hashes(config_path, source_ledger_path)
    expected_paths = set(expected)
    if set(observed) != expected_paths:
        _fail(
            "executed source set drifted: "
            f"missing={sorted(expected_paths - set(observed))}, "
            f"extra={sorted(set(observed) - expected_paths)}"
        )
    for relative, expected_hash in observed.items():
        path = ROOT / str(relative)
        try:
            path.resolve().relative_to(ROOT.resolve())
        except ValueError:
            _fail(f"executed source path escapes repository: {relative}")
        path = _regular(path, f"executed source {relative}")
        got = LB.sha256_file(path)
        _same(str(expected_hash), got, f"executed source hash {relative}")
        _same(str(expected[relative]), got, f"current sealed source hash {relative}")
    ledger_attestation = RUN.verify_source_ledger(source_ledger_path, observed)
    _same(ledger_attestation["path"], sidecar.get("source_ledger"), "source ledger path")
    _same(ledger_attestation["ledger_version"], sidecar.get("source_ledger_version"), "source ledger version")
    _same(ledger_attestation["content_sha256"], sidecar.get("source_ledger_content_sha256"), "source ledger content hash")
    _same(ledger_attestation["file_sha256"], sidecar.get("source_ledger_file_sha256"), "source ledger file hash")
    runner_key = str(sidecar.get("runner_source"))
    _same(
        str(observed.get(runner_key)), sidecar.get("runner_source_sha256"),
        "runner source SHA-256",
    )
    _same(
        str(Path(RUN.__file__).resolve().relative_to(ROOT.resolve())),
        runner_key,
        "runner source path",
    )


def _check_sidecar(
    sidecar: Mapping[str, Any],
    *,
    predictions_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    manifest_file_hash: str,
    config_path: Path,
    source_ledger_path: Path,
) -> None:
    if set(sidecar) != SIDECAR_FIELDS:
        _fail(
            "sidecar schema drifted: "
            f"missing={sorted(SIDECAR_FIELDS - set(sidecar))}, "
            f"extra={sorted(set(sidecar) - SIDECAR_FIELDS)}"
        )
    checks = (
        ("qwen_forced_choice_qualification", sidecar["artifact"], "artifact"),
        (RUN.RUNNER_VERSION, sidecar["runner_version"], "runner version"),
        (RUN.PROTOCOL_VERSION, sidecar["protocol_version"], "protocol version"),
        (RUN.TASK_VERSION, sidecar["task_version"], "task version"),
        (RUN.ENDPOINT_VERSION, sidecar["endpoint_version"], "endpoint version"),
        (LB.DATASET_REPO, sidecar["dataset_repo"], "dataset repo"),
        (LB.DATASET_REVISION, sidecar["dataset_revision"], "dataset revision"),
        (LB.DATASET_SHA256, sidecar["dataset_sha256"], "dataset SHA-256"),
        (LB.DATASET_BYTES, sidecar["dataset_bytes"], "dataset bytes"),
        (LB.OFFICIAL_CODE_REPO, sidecar["official_code_repo"], "official repo"),
        (LB.OFFICIAL_CODE_REVISION, sidecar["official_code_revision"], "official revision"),
        (LB.OFFICIAL_PROMPT_SHA256, sidecar["official_prompt_sha256"], "official prompt"),
        (QA.LEGACY_MANIFEST_ID, sidecar["legacy_manifest"], "legacy manifest ID"),
        (QA.LEGACY_MANIFEST_VERSION, sidecar["legacy_manifest_version"], "legacy manifest version"),
        (QA.LEGACY_MANIFEST_CONTENT_SHA256, sidecar["legacy_manifest_content_sha256"], "legacy manifest content hash"),
        (QA.LEGACY_MANIFEST_FILE_SHA256, sidecar["legacy_manifest_file_sha256"], "legacy manifest file hash"),
        (str(manifest_path), sidecar["manifest"], "manifest path"),
        (QA.MANIFEST_VERSION, sidecar["manifest_version"], "manifest version"),
        (manifest["content_sha256"], sidecar["manifest_content_sha256"], "manifest content hash"),
        (manifest_file_hash, sidecar["manifest_file_sha256"], "manifest file hash"),
        (RUN.SPLIT, sidecar["split"], "split"),
        (RUN.EXPECTED_ITEMS, sidecar["split_count"], "split count"),
        (RUN.EXPECTED_COMPONENTS, sidecar["split_components"], "component count"),
        (manifest["split_counts"][RUN.SPLIT]["id_sha256"], sidecar["split_ids_sha256"], "split ID hash"),
        (manifest["split_counts"][RUN.SPLIT]["id_token_sha256"], sidecar["split_id_token_sha256"], "split ID/token hash"),
        (manifest["split_counts"][RUN.SPLIT]["id_prompt_sha256"], sidecar["split_id_prompt_sha256"], "split ID/prompt hash"),
        (manifest["split_counts"][RUN.SPLIT]["id_positions_sha256"], sidecar["split_id_positions_sha256"], "split ID/positions hash"),
        (manifest["split_counts"][RUN.SPLIT]["ordered_id_sha256"], sidecar["split_ordered_id_sha256"], "split ordered-ID hash"),
        (RUN.WITHIN_SPLIT_ORDER, sidecar["within_split_order"], "within-split order"),
        ("fresh_leakage_clean_singletons_without_replacement", sidecar["component_sampling"], "component sampling"),
        (RUN.MODEL_TAG, sidecar["model"], "model tag"),
        (RUN.MODEL_ID, sidecar["model_id"], "model ID"),
        (RUN.MODEL_REVISION, sidecar["model_revision"], "model revision"),
        (RUN.TOKENIZER_REVISION, sidecar["tokenizer_revision"], "tokenizer revision"),
        (QA.CHAT_TEMPLATE_SHA256, sidecar["chat_template_sha256"], "chat template"),
        (True, sidecar["add_generation_prompt"], "generation prompt flag"),
        (QA.ENABLE_THINKING, sidecar["enable_thinking"], "thinking flag"),
        (RUN.TRANSFORMERS_VERSION, sidecar["transformers_version"], "transformers version"),
        (RUN.TOKENIZERS_VERSION, sidecar["tokenizers_version"], "tokenizers version"),
        (RUN.TORCH_VERSION, sidecar["torch_version"], "torch version"),
        (RUN.NUMPY_VERSION, sidecar["numpy_version"], "numpy version"),
        (RUN.PANDAS_VERSION, sidecar["pandas_version"], "pandas version"),
        (RUN.PYARROW_VERSION, sidecar["pyarrow_version"], "pyarrow version"),
        (RUN.FASTPARQUET_VERSION, sidecar["fastparquet_version"], "fastparquet version"),
        (RUN.PARQUET_ENGINE, sidecar["parquet_engine"], "parquet engine"),
        (RUN.MODEL_SNAPSHOT_ATTESTATION_SCOPE, sidecar["model_snapshot_attestation_scope"], "model snapshot attestation scope"),
        (QA.MODEL_REVISION, sidecar["model_snapshot_inventory_revision"], "model snapshot inventory revision"),
        (QA.SNAPSHOT_FILE_COUNT, sidecar["model_snapshot_symlink_file_count"], "model snapshot symlink file count"),
        (QA.SNAPSHOT_INVENTORY_SHA256, sidecar["model_snapshot_symlink_inventory_sha256"], "model snapshot symlink inventory hash"),
        (dict(QA.SNAPSHOT_METADATA_SHA256), sidecar["model_snapshot_pinned_metadata_sha256"], "model snapshot pinned metadata hashes"),
        (RUN.CTX, sidecar["ctx"], "context window"),
        (RUN.MAX_PROMPT_TOKENS, sidecar["max_prompt_tokens"], "max prompt tokens"),
        (RUN.WINDOW, sidecar["window"], "window"),
        (float(RUN.BUDGET), sidecar["budget"], "budget"),
        (RUN.DTYPE_NAME, sidecar["dtype"], "dtype"),
        (list(RUN.BIT_LIST), sidecar["bit_list"], "bit list"),
        (RUN.MAXB, sidecar["maxb"], "maxb"),
        (RUN.ROT_SEED, sidecar["rot_seed"], "rotation seed"),
        (RUN.NORM_CORRECT, sidecar["norm_correct"], "norm correction"),
        (RUN.CHUNK, sidecar["chunk"], "chunk"),
        (RUN.N_LAYERS, sidecar["n_layers"], "layer count"),
        (RUN.N_ATTENTION_HEADS, sidecar["n_attention_heads"], "attention head count"),
        (RUN.N_KV_HEADS, sidecar["n_kv_heads"], "KV head count"),
        (RUN.HEAD_DIM, sidecar["head_dim"], "head dimension"),
        ("48_layers_x_4_kv_x_ctx_len_all_uint8_2", sidecar["uniform_allocation_contract"], "uniform allocation contract"),
        (RUN.C.IMPL, sidecar["qualification_attn_impl"], "qualification attention implementation"),
        (RUN.SP.IMPL, sidecar["structured_policy_impl"], "structured policy implementation"),
        (RUN.SP.CAPTURE_VERSION, sidecar["structured_capture_version"], "structured capture version"),
        (str(Path(RUN.SP.__file__).resolve().relative_to(ROOT.resolve())), sidecar["structured_policy_source"], "structured policy source"),
        (LB.sha256_file(RUN.SP.__file__), sidecar["structured_policy_source_sha256"], "structured policy source hash"),
        ("forced_choice_teacher_forced_argmax", sidecar["decode"], "decode"),
        (1.0, sidecar["temperature"], "temperature"),
        ("official_chat_prompt_end", sidecar["compress_at"], "compression boundary"),
        (list(RUN.ARMS), sidecar["arms"], "qualification arms"),
        (RUN.R8.ALLOCATION_ID_ALGORITHM, sidecar["allocation_id_algorithm"], "allocation ID algorithm"),
        (list(QA.SCAFFOLD_TOKEN_IDS), sidecar["scaffold_token_ids"], "scaffold token IDs"),
        (PD.token_hash(QA.SCAFFOLD_TOKEN_IDS), sidecar["scaffold_token_hash"], "scaffold token hash"),
        (dict(QA.CHOICE_BRANCH_IDS), sidecar["choice_branch_ids"], "branch IDs"),
        (PD.token_hash([QA.CHOICE_BRANCH_IDS[x] for x in "ABCD"]), sidecar["choice_branch_ids_hash"], "branch ID hash"),
        (len(QA.SCAFFOLD_TOKEN_IDS), sidecar["choice_position"], "choice position"),
        (True, sidecar["no_truncation"], "no-truncation flag"),
        (True, sidecar["no_free_generation"], "no-generation flag"),
        (True, sidecar["no_raw_logits"], "no-raw-logits flag"),
        (True, sidecar["no_labels"], "no-labels flag"),
        (predictions_path.name, sidecar["parquet"], "parquet basename"),
        (LB.sha256_file(predictions_path), sidecar["parquet_sha256"], "parquet SHA-256"),
        (RUN.EXPECTED_ITEMS * len(RUN.ARMS), sidecar["rows"], "row count"),
        (RUN.EXPECTED_ITEMS * len(RUN.ARMS), sidecar["expected_rows"], "expected row count"),
        (["item_id", "arm"], sidecar["row_key"], "row key"),
    )
    for expected, actual, description in checks:
        _same(expected, actual, description)
    elapsed = float(sidecar["elapsed_seconds"])
    if not math.isfinite(elapsed) or elapsed < 0:
        _fail("sidecar elapsed_seconds must be finite and nonnegative")
    placements = sidecar["device_placements"]
    if not isinstance(placements, list) or not placements:
        _fail("sidecar device placements must be a nonempty list")
    if any(str(value).lower().startswith(("cpu", "disk")) for value in placements):
        _fail(f"sidecar records CPU/disk model placement: {placements}")
    _check_source_hashes(sidecar, config_path, source_ledger_path)


def _exact_prediction_schema(frame: pd.DataFrame) -> None:
    columns = tuple(str(column) for column in frame.columns)
    if columns != RUN.PREDICTION_COLUMNS:
        extra = [column for column in columns if column not in RUN.PREDICTION_COLUMNS]
        missing = [column for column in RUN.PREDICTION_COLUMNS if column not in columns]
        _fail(f"prediction schema drifted: missing={missing}, extra={extra}")
    for column in columns:
        lowered = column.lower()
        if lowered in FORBIDDEN_RESULT_COLUMNS or lowered.startswith(("gold_", "raw_")):
            _fail(f"prediction artifact contains forbidden column {column!r}")
        if frame[column].dtype == object:
            for value in frame[column]:
                if isinstance(value, (dict, list, tuple, np.ndarray)):
                    _fail(f"prediction {column} contains a nonscalar value")


def _validate_rows(
    frame: pd.DataFrame,
    *,
    manifest: Mapping[str, Any],
    manifest_file_hash: str,
) -> tuple[list[str], dict[str, str]]:
    _exact_prediction_schema(frame)
    entries = RUN.canonical_entries(manifest)
    item_ids = [str(row["id"]) for row in entries]
    entry_by_id = {str(row["id"]): row for row in entries}
    expected_order = [(item_id, arm) for item_id in item_ids for arm in RUN.ARMS]
    expected_keys = set(expected_order)
    observed_keys = list(zip(frame["item_id"].astype(str), frame["arm"].astype(str)))
    if len(observed_keys) != len(expected_keys) or set(observed_keys) != expected_keys or len(observed_keys) != len(set(observed_keys)):
        _fail("prediction keys differ from exact qualification IDs x arms")
    if observed_keys != expected_order:
        _fail("prediction rows are not in canonical item-major/arm order")
    if frame["truncated"].dtype != bool:
        _fail("prediction truncated column must be Boolean")
    if frame["norm_correct"].dtype != bool:
        _fail("prediction norm_correct column must be Boolean")

    allocation_by_item: dict[str, str] = {}
    for item_id in item_ids:
        block = frame.loc[frame["item_id"].astype(str).eq(item_id)]
        if block["arm"].astype(str).tolist() != list(RUN.ARMS):
            _fail(f"arm order drifted for {item_id}")
        entry = entry_by_id[item_id]
        uniform_allocation = ""
        for row in block.itertuples(index=False):
            arm = str(row.arm)
            shared = {
                "group_id": str(entry["group_id"]), "split": RUN.SPLIT,
                "context_hash": str(entry["context_hash"]),
                "question_hash": str(entry["question_hash"]),
                "prompt_token_hash": str(entry["prompt_token_hash"]),
                "input_tokens": int(entry["input_tokens"]),
                "n_prompt_tokens": int(entry["input_tokens"]),
                "truncated": False,
                "dataset_sha256": LB.DATASET_SHA256,
                "manifest_content_sha256": str(manifest["content_sha256"]),
                "manifest_file_sha256": manifest_file_hash,
                "task_version": RUN.TASK_VERSION,
                "endpoint_version": RUN.ENDPOINT_VERSION,
                "model": RUN.MODEL_TAG, "model_id": RUN.MODEL_ID,
                "model_revision": RUN.MODEL_REVISION,
                "tokenizer_revision": RUN.TOKENIZER_REVISION,
                "ctx": RUN.CTX, "window": RUN.WINDOW,
            }
            for field, expected in shared.items():
                _same(expected, getattr(row, field), f"{item_id}/{arm} {field}")
            for field in ("domain", "sub_domain", "difficulty", "length"):
                _same(entry["metadata"][field], getattr(row, field), f"{item_id}/{arm} {field}")
            _same(RUN.ARMS.index(arm), int(row.arm_order), f"{item_id}/{arm} order")
            _same(0 if arm == "fp" else RUN.BUDGET, int(row.B), f"{item_id}/{arm} B")
            choice = str(row.forced_choice)
            if choice not in "ABCD":
                _fail(f"invalid forced choice for {item_id}/{arm}")
            _same("ABCD".index(choice), int(row.choice_index), f"{item_id}/{arm} choice index")
            for field in ("choice_entropy", "choice_margin", "choice_max_probability"):
                value = float(getattr(row, field))
                if not math.isfinite(value):
                    _fail(f"non-finite {field} for {item_id}/{arm}")
            if not 0 <= float(row.choice_margin) <= 1:
                _fail(f"invalid choice margin for {item_id}/{arm}")
            if not 0.25 <= float(row.choice_max_probability) <= 1:
                _fail(f"invalid max choice probability for {item_id}/{arm}")
            _same(PD.token_hash(QA.SCAFFOLD_TOKEN_IDS), row.scaffold_token_hash, "scaffold hash")
            _same(PD.token_hash([QA.CHOICE_BRANCH_IDS[x] for x in "ABCD"]), row.choice_branch_ids_hash, "branch hash")
            _same(RUN.MAXB, int(row.maxb), f"{item_id}/{arm} maxb")
            _same(RUN.ROT_SEED, int(row.rot_seed), f"{item_id}/{arm} rotation seed")
            _same(True, bool(row.norm_correct), f"{item_id}/{arm} norm correction")
            _same(RUN.WINDOW, int(row.observed_queries), f"{item_id}/{arm} observed queries")
            _same(int(entry["input_tokens"]) - 1 - RUN.WINDOW, int(row.ctx_len), f"{item_id}/{arm} ctx_len")
            allocation = str(row.allocation_id)
            if len(allocation) != 64 or any(ch not in "0123456789abcdef" for ch in allocation):
                _fail(f"invalid allocation ID for {item_id}/{arm}")
            if arm == "fp":
                _same(16.0, float(row.bits_per_token), f"{item_id} FP bits")
                _same(0.0, float(row.evict_frac), f"{item_id} FP eviction")
                _same(FP_ALLOCATION_ID, allocation, f"{item_id} FP allocation")
            else:
                if not math.isclose(float(row.bits_per_token), RUN.BUDGET, rel_tol=0, abs_tol=1e-7):
                    _fail(f"uniform bits exceed or underspend B=2 for {item_id}")
                if not math.isclose(float(row.evict_frac), 0.0, rel_tol=0, abs_tol=1e-12):
                    _fail(f"uniform evicted tokens for {item_id}")
                expected_allocation = RUN.expected_uniform_allocation_id(int(row.ctx_len))
                _same(expected_allocation, allocation, f"{item_id} deterministic uniform allocation")
                uniform_allocation = allocation
        allocation_by_item[item_id] = uniform_allocation
    return item_ids, allocation_by_item


def _component_bootstrap(
    outcomes: pd.DataFrame,
    component_order: Sequence[str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    order = tuple(component_order)
    if len(order) != len(set(order)) or len(order) != RUN.EXPECTED_COMPONENTS:
        _fail("qualification component order is not the frozen 20-singleton set")
    grouped = outcomes.groupby("group_id", sort=False)
    if set(grouped.groups) != set(order):
        _fail("outcomes do not cover the exact qualification components")
    sizes = grouped.size().reindex(order).to_numpy(dtype=np.int64)
    sums = {
        arm: grouped[arm].sum().reindex(order).to_numpy(dtype=float)
        for arm in RUN.ARMS
    }
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(order), size=(draws, len(order)))
    denominators = sizes[indices].sum(axis=1)
    return {
        arm: values[indices].sum(axis=1) / denominators
        for arm, values in sums.items()
    }


def _ordered_components(entries: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    members: dict[str, list[Mapping[str, Any]]] = {}
    for row in entries:
        members.setdefault(str(row["group_id"]), []).append(row)
    id_order = QA.partition_order(entries)
    group_by_id = {str(row["id"]): str(row["group_id"]) for row in entries}
    order = tuple(group_by_id[item_id] for item_id in id_order)
    if len(order) != len(set(order)):
        _fail("qualification manifest does not contain singleton components")
    return order


def analyze(
    frame: pd.DataFrame,
    *,
    manifest: Mapping[str, Any],
    dataset_path: Path,
) -> dict[str, Any]:
    data = LB.load_dataset(dataset_path, authenticate=True)
    by_id = LB.index_by_id(data)
    entries = RUN.canonical_entries(manifest)
    item_ids = [str(row["id"]) for row in entries]
    choices = frame.pivot(index="item_id", columns="arm", values="forced_choice").reindex(index=item_ids, columns=RUN.ARMS)
    if choices.isna().any().any():
        _fail("validated predictions do not pivot over every item and arm")
    gold = pd.Series({item_id: str(by_id[item_id]["answer"]) for item_id in item_ids})
    correct = choices.eq(gold, axis=0)
    groups = {str(row["id"]): str(row["group_id"]) for row in entries}
    outcomes = pd.DataFrame({
        "item_id": item_ids,
        "group_id": [groups[item_id] for item_id in item_ids],
        "fp": correct["fp"].to_numpy(dtype=bool),
        "uniform": correct["uniform"].to_numpy(dtype=bool),
    })
    bootstrap = _component_bootstrap(outcomes, _ordered_components(entries))
    fp_accuracy = float(outcomes["fp"].mean())
    uniform_accuracy = float(outcomes["uniform"].mean())
    fp_q05 = float(np.quantile(bootstrap["fp"], BOOTSTRAP_QUANTILE, method="linear"))
    uniform_q05 = float(np.quantile(bootstrap["uniform"], BOOTSTRAP_QUANTILE, method="linear"))
    gates = {
        "fp_point_gate": fp_accuracy >= FP_ACCURACY_MIN,
        "fp_bootstrap_gate": fp_q05 > CHANCE,
        "uniform_point_gate": UNIFORM_ACCURACY_MIN <= uniform_accuracy <= UNIFORM_ACCURACY_MAX,
        "uniform_bootstrap_gate": uniform_q05 > CHANCE,
    }
    advance = all(gates.values())
    return {
        "protocol_version": RUN.PROTOCOL_VERSION,
        "split": RUN.SPLIT,
        "n_items": len(item_ids),
        "n_components": len(set(outcomes["group_id"])),
        "prediction_rows": len(frame),
        "fp_correct": int(outcomes["fp"].sum()),
        "fp_accuracy": fp_accuracy,
        "fp_bootstrap_q05": fp_q05,
        "uniform_correct": int(outcomes["uniform"].sum()),
        "uniform_accuracy": uniform_accuracy,
        "uniform_bootstrap_q05": uniform_q05,
        **gates,
        "decision": "advance_v7_development" if advance else "stop_v7_qualification",
    }


def _thresholds() -> dict[str, Any]:
    return {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_rng": "numpy.Generator(PCG64)",
        "bootstrap_quantile": BOOTSTRAP_QUANTILE,
        "bootstrap_quantile_method": "linear",
        "chance": CHANCE,
        "qualification_fp_accuracy_min": FP_ACCURACY_MIN,
        "qualification_fp_q05_strictly_above": CHANCE,
        "qualification_uniform_accuracy_interval": [UNIFORM_ACCURACY_MIN, UNIFORM_ACCURACY_MAX],
        "qualification_uniform_q05_strictly_above": CHANCE,
        "development_fp_accuracy_min": 0.50,
        "development_fp_q05_strictly_above": 0.25,
        "development_F_star_accuracy_interval": [0.30, 0.80],
        "development_F_star_q05_strictly_above": 0.25,
        "development_H_min": 0.10,
        "development_H_q05_strictly_above": 0.0,
        "development_H_each_half_min": 0.05,
        "development_S_min": 0.05,
    }


def _write_atomic_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        _fail(f"refusing symlink lock target {path}")
    if path.exists():
        if not path.is_file():
            _fail(f"lock target is not a regular file: {path}")
        if path.read_bytes() == payload:
            return
        _fail(f"refusing stale/different existing lock {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _half_partition_record(
    manifest: Mapping[str, Any], *, split: str, first_half: int
) -> dict[str, Any]:
    entries = [row for row in manifest["examples"] if row["split"] == split]
    ordered_ids = QA.partition_order(entries)
    expected = QA.SPLIT_ROWS[split]
    if len(ordered_ids) != expected or len(ordered_ids) != len(set(ordered_ids)):
        _fail(f"{split} partition does not contain the frozen unique IDs")
    if not 0 < first_half < expected:
        _fail(f"invalid {split} half boundary {first_half}")
    halves = (ordered_ids[:first_half], ordered_ids[first_half:])
    def digest(ids: Sequence[str]) -> str:
        payload = "".join(f"{item_id}\n" for item_id in ids).encode("ascii")
        return hashlib.sha256(payload).hexdigest()
    return {
        **dict(manifest["split_counts"][split]),
        "half_order": "sha256(selection_namespace_plus_item_id_ascii)_ascending_v1",
        "half_row_counts": [len(values) for values in halves],
        "half_component_counts": [len(values) for values in halves],
        "half_ordered_id_sha256": [digest(values) for values in halves],
    }


def write_advance_lock(
    path: Path,
    *,
    summary: Mapping[str, Any],
    predictions_path: Path,
    sidecar_path: Path,
    sidecar: Mapping[str, Any],
    manifest_path: Path,
    legacy_manifest_path: Path,
    config_path: Path,
    source_ledger_path: Path,
) -> None:
    if summary["decision"] != "advance_v7_development":
        if path.exists() or path.is_symlink():
            _fail(f"failed qualification refuses existing lock {path}")
        return
    reader_path = _regular(__file__, "qualification reader")
    manifest_object = _load_object(manifest_path, "Qwen manifest for lock")
    lock: dict[str, Any] = {
        "lock_version": LOCK_VERSION,
        "protocol_version": RUN.PROTOCOL_VERSION,
        "runner_version": RUN.RUNNER_VERSION,
        "reader_version": READER_VERSION,
        "task_version": RUN.TASK_VERSION,
        "endpoint_version": RUN.ENDPOINT_VERSION,
        "decision": "advance_v7_development",
        "qualification": {
            "predictions": str(predictions_path),
            "predictions_sha256": LB.sha256_file(predictions_path),
            "sidecar": str(sidecar_path),
            "sidecar_sha256": LB.sha256_file(sidecar_path),
            "summary": dict(summary),
        },
        "provenance": {
            "dataset_repo": LB.DATASET_REPO,
            "dataset_revision": LB.DATASET_REVISION,
            "dataset_sha256": LB.DATASET_SHA256,
            "dataset_bytes": LB.DATASET_BYTES,
            "official_prompt_sha256": LB.OFFICIAL_PROMPT_SHA256,
            "legacy_manifest": QA.LEGACY_MANIFEST_ID,
            "legacy_manifest_version": QA.LEGACY_MANIFEST_VERSION,
            "legacy_manifest_content_sha256": QA.LEGACY_MANIFEST_CONTENT_SHA256,
            "legacy_manifest_file_sha256": LB.sha256_file(legacy_manifest_path),
            "qwen_manifest": str(manifest_path),
            "qwen_manifest_version": QA.MANIFEST_VERSION,
            "qwen_manifest_content_sha256": str(manifest_object["content_sha256"]),
            "qwen_manifest_file_sha256": LB.sha256_file(manifest_path),
            "source_ledger": str(source_ledger_path),
            "source_ledger_version": sidecar["source_ledger_version"],
            "source_ledger_content_sha256": sidecar["source_ledger_content_sha256"],
            "source_ledger_file_sha256": LB.sha256_file(source_ledger_path),
            "executed_source_sha256": dict(sidecar["executed_source_sha256"]),
            "runner_source": str(sidecar["runner_source"]),
            "runner_source_sha256": str(sidecar["runner_source_sha256"]),
            "reader_source": str(reader_path.relative_to(ROOT.resolve())),
            "reader_source_sha256": LB.sha256_file(reader_path),
            "config": str(config_path.relative_to(ROOT.resolve())),
            "config_sha256": LB.sha256_file(config_path),
            "model_snapshot_attestation_scope": sidecar["model_snapshot_attestation_scope"],
            "model_snapshot_inventory_revision": sidecar["model_snapshot_inventory_revision"],
            "model_snapshot_symlink_file_count": sidecar["model_snapshot_symlink_file_count"],
            "model_snapshot_symlink_inventory_sha256": sidecar["model_snapshot_symlink_inventory_sha256"],
            "model_snapshot_pinned_metadata_sha256": dict(
                sidecar["model_snapshot_pinned_metadata_sha256"]
            ),
        },
        "partitions": {
            "qualification": dict(manifest_object["split_counts"]["qualification"]),
            "development": _half_partition_record(
                manifest_object, split="development", first_half=26
            ),
            "confirmation": _half_partition_record(
                manifest_object, split="confirmation", first_half=22
            ),
            "within_split_order": RUN.WITHIN_SPLIT_ORDER,
        },
        "execution": {
            "model": RUN.MODEL_TAG,
            "model_id": RUN.MODEL_ID,
            "model_revision": RUN.MODEL_REVISION,
            "tokenizer_revision": RUN.TOKENIZER_REVISION,
            "chat_template_sha256": QA.CHAT_TEMPLATE_SHA256,
            "enable_thinking": QA.ENABLE_THINKING,
            "ctx": RUN.CTX,
            "window": RUN.WINDOW,
            "budget": RUN.BUDGET,
            "dtype": RUN.DTYPE_NAME,
            "bit_list": list(RUN.BIT_LIST),
            "maxb": RUN.MAXB,
            "rot_seed": RUN.ROT_SEED,
            "norm_correct": RUN.NORM_CORRECT,
            "chunk": RUN.CHUNK,
            "n_layers": RUN.N_LAYERS,
            "n_attention_heads": RUN.N_ATTENTION_HEADS,
            "n_kv_heads": RUN.N_KV_HEADS,
            "head_dim": RUN.HEAD_DIM,
            "uniform_allocation_contract": "48_layers_x_4_kv_x_ctx_len_all_uint8_2",
            "qualification_attn_impl": RUN.C.IMPL,
            "development_attn_impl": RUN.SP.IMPL,
            "structured_policy_impl": RUN.SP.IMPL,
            "structured_capture_version": RUN.SP.CAPTURE_VERSION,
            "structured_policy_source": sidecar["structured_policy_source"],
            "structured_policy_source_sha256": sidecar["structured_policy_source_sha256"],
            "qualification_arms": list(RUN.ARMS),
            "development_arms": list(DEVELOPMENT_ARMS),
            "development_candidates": list(DEVELOPMENT_CANDIDATES),
            "old_menu": list(OLD_MENU),
            "structured_menu": list(STRUCTURED_MENU),
            "endpoint_version": RUN.ENDPOINT_VERSION,
            "scaffold_token_ids": list(QA.SCAFFOLD_TOKEN_IDS),
            "choice_branch_ids": dict(QA.CHOICE_BRANCH_IDS),
            "numpy_version": sidecar["numpy_version"],
            "pandas_version": sidecar["pandas_version"],
            "pyarrow_version": sidecar["pyarrow_version"],
            "fastparquet_version": sidecar["fastparquet_version"],
            "parquet_engine": sidecar["parquet_engine"],
        },
        "thresholds": _thresholds(),
    }
    lock["content_sha256"] = _content_sha256(lock)
    _same(lock["content_sha256"], _content_sha256(lock), "lock content hash")
    _write_atomic_once(path, lock)
    authenticated = _load_object(path, "advance lock")
    _same(set(lock), set(authenticated), "advance lock schema")
    _same(authenticated["content_sha256"], _content_sha256(authenticated), "advance lock authentication")
    _same(lock, authenticated, "advance lock round trip")


def _write_csv(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(path, index=False)


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    predictions_path = _regular(args.predictions, "prediction parquet")
    sidecar_path = predictions_path.with_suffix(".json")
    sidecar = _load_object(sidecar_path, "prediction sidecar")
    manifest_path = _regular(args.manifest, "Qwen manifest")
    legacy_manifest_path = _regular(args.legacy_manifest, "legacy manifest")
    dataset_path = _regular(args.dataset, "dataset")
    config_path = _regular(args.config, "model config")
    source_ledger_path = _regular(args.source_ledger, "source ledger")
    if source_ledger_path != RUN.DEFAULT_SOURCE_LEDGER.resolve():
        _fail(
            f"source ledger must be the canonical path {RUN.DEFAULT_SOURCE_LEDGER.resolve()}"
        )
    runner_path = _regular(args.runner, "qualification runner")
    _same(Path(RUN.__file__).resolve(), runner_path, "runner argument")
    tokenizer = QA.load_tokenizer(args.tokenizer)
    manifest, manifest_file_hash = QA.authenticate_manifest(
        manifest_path,
        dataset_path=dataset_path,
        legacy_manifest_path=legacy_manifest_path,
        tokenizer=tokenizer,
    )
    _check_sidecar(
        sidecar,
        predictions_path=predictions_path,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_file_hash=manifest_file_hash,
        config_path=config_path,
        source_ledger_path=source_ledger_path,
    )
    try:
        frame = pd.read_parquet(
            predictions_path, engine=RUN.PARQUET_ENGINE
        )
    except Exception as exc:
        _fail(f"cannot read prediction parquet: {exc}")
    _validate_rows(frame, manifest=manifest, manifest_file_hash=manifest_file_hash)
    if args.validate_only:
        if args.lock or args.csv:
            _fail("--validate-only forbids --lock and --csv")
        print(
            f"PASS V7 Qwen qualification artifact: {len(frame)} rows, "
            f"{RUN.EXPECTED_ITEMS} items / {RUN.EXPECTED_COMPONENTS} components"
        )
        return None

    summary = analyze(frame, manifest=manifest, dataset_path=dataset_path)
    print("V7 Qwen qualification (frozen section 3L)")
    print("provenance      PASS (dataset/manifest/model/template/source hashes)")
    print(f"rows            {summary['prediction_rows']} predictions; {summary['n_items']} items / {summary['n_components']} components")
    print(f"FP              {summary['fp_correct']}/20 = {summary['fp_accuracy']:.3f}, q05={summary['fp_bootstrap_q05']:.3f}")
    print(f"uniform         {summary['uniform_correct']}/20 = {summary['uniform_accuracy']:.3f}, q05={summary['uniform_bootstrap_q05']:.3f}")
    print(f"decision        {summary['decision']}")
    if args.csv:
        _write_csv(Path(args.csv), summary)
    if args.lock:
        write_advance_lock(
            Path(args.lock),
            summary=summary,
            predictions_path=predictions_path,
            sidecar_path=sidecar_path,
            sidecar=sidecar,
            manifest_path=manifest_path,
            legacy_manifest_path=legacy_manifest_path,
            config_path=config_path,
            source_ledger_path=source_ledger_path,
        )
        if summary["decision"] == "advance_v7_development":
            print(f"lock            {Path(args.lock).resolve()}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--legacy-manifest", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", default=str(ROOT / "h0_measurement/models.yaml"))
    parser.add_argument("--source-ledger", required=True)
    parser.add_argument("--runner", default=str(ROOT / "h0_measurement/run_longbench_v2_qwen_v7_qualification.py"))
    parser.add_argument("--csv", default="")
    parser.add_argument("--lock", default="")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (QwenQualificationReaderError, QA.V7ManifestError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
