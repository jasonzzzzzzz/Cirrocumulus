#!/usr/bin/env python3
"""Strict reader for the frozen V7 Qwen structured-query development study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "h0_measurement") not in sys.path:
    sys.path.insert(0, str(ROOT / "h0_measurement"))

from h0_measurement import audit_longbench_v2_qwen_v7 as QA  # noqa: E402
from h0_measurement import run_longbench_v2_qwen_v7_development as RUN  # noqa: E402
from sievelib import policy_diagnostic as PD  # noqa: E402
from sievelib import tasks_longbench_v2 as LB  # noqa: E402


READER_VERSION = "longbench_v2_qwen_v7_development_reader_v1"
CONFIRMATION_LOCK_VERSION = "longbench_v2_sieve_v7_qwen131072_confirmation_lock_v1"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
BOOTSTRAP_QUANTILE = 0.05
FP_ACCURACY_MIN = 0.50
F_STAR_ACCURACY_MIN = 0.30
F_STAR_ACCURACY_MAX = 0.80
CHANCE = 0.25
H_MIN = 0.10
H_HALF_MIN = 0.05
S_MIN = 0.05
ARMS = tuple(RUN.ARMS)
CANDIDATES = tuple(RUN.CANDIDATES)
OLD_MENU = tuple(RUN.OLD_MENU)
STRUCTURED_MENU = tuple(RUN.STRUCTURED_MENU)
PREDICTION_COLUMNS = tuple(RUN.PREDICTION_COLUMNS)
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
    "split_id_positions_sha256", "split_ordered_id_sha256", "within_split_order",
    "model", "model_id", "model_revision", "tokenizer_revision",
    "chat_template_sha256", "add_generation_prompt", "enable_thinking",
    "transformers_version", "tokenizers_version", "torch_version",
    "numpy_version", "pandas_version", "pyarrow_version",
    "fastparquet_version", "parquet_engine", "model_snapshot_attestation_scope",
    "model_snapshot_inventory_revision", "model_snapshot_symlink_file_count",
    "model_snapshot_symlink_inventory_sha256",
    "model_snapshot_pinned_metadata_sha256", "snapshot_seal_timing", "qualification_lock",
    "qualification_lock_version", "qualification_lock_file_sha256",
    "qualification_lock_content_sha256", "qualification_predictions_sha256",
    "qualification_sidecar_sha256", "qualification_source_ledger_file_sha256",
    "qualification_source_ledger_content_sha256", "ctx", "max_prompt_tokens",
    "window", "budget", "dtype", "bit_list", "maxb", "rot_seed",
    "norm_correct", "chunk", "n_layers", "n_attention_heads", "n_kv_heads",
    "head_dim", "uniform_allocation_contract", "attn_impl",
    "underlying_attn_impl", "structured_capture_version",
    "structured_query_groups", "structured_max_tokens_per_group",
    "structured_group_weight", "score_query_sources", "tail_score_query_count",
    "structured_score_query_count_rule", "structured_interior_noise_query_source",
    "structured_capture_before_single_prefill",
    "layer_context_builds_per_layer", "device_placements", "decode",
    "temperature", "compress_at", "arms", "candidates", "old_menu",
    "structured_menu", "allocation_id_algorithm", "allocation_summary",
    "scaffold_token_ids", "scaffold_token_hash", "choice_branch_ids",
    "choice_branch_ids_hash", "choice_position", "no_truncation",
    "no_free_generation", "no_raw_logits", "no_proxy", "no_labels",
    "gold_unused_for_prompt_allocation_execution_and_output", "source_ledger",
    "source_ledger_version", "source_ledger_content_sha256",
    "source_ledger_file_sha256", "executed_source_sha256", "runner_source",
    "runner_source_sha256", "source_seal_timing", "parquet", "parquet_sha256",
    "rows", "expected_rows", "row_key", "item_ids_sha256", "elapsed_seconds",
}


class QwenV7DevelopmentReaderError(ValueError):
    """A V7 development artifact or decision contract failed validation."""


def _fail(message: str) -> None:
    raise QwenV7DevelopmentReaderError(message)


def _scalar(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _same(expected: Any, actual: Any, description: str) -> None:
    actual = _scalar(actual)
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


def _hash_ids(values: Iterable[str]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in values).encode("ascii")
    ).hexdigest()


def canonical_entries(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return RUN.canonical_entries(manifest)


def frozen_halves(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[frozenset[str], frozenset[str]]]:
    order = tuple(QA.partition_order(entries))
    if len(order) != RUN.EXPECTED_ITEMS or len(set(order)) != len(order):
        _fail("development canonical order does not contain 52 unique IDs")
    halves = (frozenset(order[:26]), frozenset(order[26:]))
    if [len(half) for half in halves] != [26, 26]:
        _fail("development halves are not the frozen 26/26 partition")
    return order, halves


def _check_source_hashes(
    sidecar: Mapping[str, Any], *, config_path: Path, source_ledger_path: Path
) -> dict[str, Any]:
    observed = sidecar.get("executed_source_sha256")
    if not isinstance(observed, dict) or not observed:
        _fail("sidecar executed_source_sha256 must be a nonempty object")
    try:
        expected = RUN.executed_source_hashes(config_path, source_ledger_path)
    except RUN.QwenV7DevelopmentRunnerError as exc:
        _fail(str(exc))
    _same(set(expected), set(observed), "development executed source path set")
    for relative, expected_hash in observed.items():
        path = ROOT / str(relative)
        try:
            path.resolve().relative_to(ROOT.resolve())
        except ValueError:
            _fail(f"development source path escapes repository: {relative}")
        source = _regular(path, f"development source {relative}")
        got = LB.sha256_file(source)
        _same(str(expected_hash), got, f"recorded development source hash {relative}")
        _same(expected[relative], got, f"current development source hash {relative}")
    try:
        ledger = RUN.verify_source_ledger(source_ledger_path, observed)
    except RUN.QwenV7DevelopmentRunnerError as exc:
        _fail(str(exc))
    for key, field in (
        ("path", "source_ledger"), ("ledger_version", "source_ledger_version"),
        ("content_sha256", "source_ledger_content_sha256"),
        ("file_sha256", "source_ledger_file_sha256"),
    ):
        _same(ledger[key], sidecar[field], f"development {field}")
    runner_key = str(sidecar["runner_source"])
    _same(str(Path(RUN.__file__).resolve().relative_to(ROOT.resolve())), runner_key, "development runner source path")
    _same(observed[runner_key], sidecar["runner_source_sha256"], "development runner source hash")
    return ledger


def _expected_allocation_summary(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm in ARMS:
        values = frame.loc[frame["arm"].eq(arm)]
        summary[arm] = {
            "unique_allocation_ids": int(values["allocation_id"].nunique()),
            "mean_bits_per_token": float(values["bits_per_token"].mean()),
            "mean_evict_frac": float(values["evict_frac"].mean()),
        }
    return summary


def _check_sidecar(
    sidecar: Mapping[str, Any], *, predictions_path: Path,
    manifest_path: Path, manifest: Mapping[str, Any], manifest_file_hash: str,
    legacy_manifest_path: Path, config_path: Path, source_ledger_path: Path,
    lock_attestation: Mapping[str, Any], snapshot: Mapping[str, Any],
    frame: pd.DataFrame,
) -> None:
    if set(sidecar) != SIDECAR_FIELDS:
        _fail(
            "development sidecar schema drifted: "
            f"missing={sorted(SIDECAR_FIELDS - set(sidecar))}, "
            f"extra={sorted(set(sidecar) - SIDECAR_FIELDS)}"
        )
    checks = (
        ("qwen_forced_choice_v7_development", sidecar["artifact"], "artifact"),
        (RUN.RUNNER_VERSION, sidecar["runner_version"], "runner version"),
        (RUN.PROTOCOL_VERSION, sidecar["protocol_version"], "protocol version"),
        (RUN.TASK_VERSION, sidecar["task_version"], "task version"),
        (RUN.ENDPOINT_VERSION, sidecar["endpoint_version"], "endpoint version"),
        (LB.DATASET_REPO, sidecar["dataset_repo"], "dataset repo"),
        (LB.DATASET_REVISION, sidecar["dataset_revision"], "dataset revision"),
        (LB.DATASET_SHA256, sidecar["dataset_sha256"], "dataset hash"),
        (LB.DATASET_BYTES, sidecar["dataset_bytes"], "dataset bytes"),
        (LB.OFFICIAL_CODE_REPO, sidecar["official_code_repo"], "official code repo"),
        (LB.OFFICIAL_CODE_REVISION, sidecar["official_code_revision"], "official code revision"),
        (LB.OFFICIAL_PROMPT_SHA256, sidecar["official_prompt_sha256"], "official prompt hash"),
        (QA.LEGACY_MANIFEST_ID, sidecar["legacy_manifest"], "legacy manifest ID"),
        (QA.LEGACY_MANIFEST_VERSION, sidecar["legacy_manifest_version"], "legacy manifest version"),
        (QA.LEGACY_MANIFEST_CONTENT_SHA256, sidecar["legacy_manifest_content_sha256"], "legacy manifest content hash"),
        (LB.sha256_file(legacy_manifest_path), sidecar["legacy_manifest_file_sha256"], "legacy manifest file hash"),
        (str(manifest_path), sidecar["manifest"], "manifest path"),
        (QA.MANIFEST_VERSION, sidecar["manifest_version"], "manifest version"),
        (manifest["content_sha256"], sidecar["manifest_content_sha256"], "manifest content hash"),
        (manifest_file_hash, sidecar["manifest_file_sha256"], "manifest file hash"),
        (RUN.SPLIT, sidecar["split"], "split"),
        (RUN.EXPECTED_ITEMS, sidecar["split_count"], "split count"),
        (RUN.EXPECTED_COMPONENTS, sidecar["split_components"], "split components"),
        (manifest["split_counts"][RUN.SPLIT]["id_sha256"], sidecar["split_ids_sha256"], "split ID hash"),
        (manifest["split_counts"][RUN.SPLIT]["id_token_sha256"], sidecar["split_id_token_sha256"], "split ID/token hash"),
        (manifest["split_counts"][RUN.SPLIT]["id_prompt_sha256"], sidecar["split_id_prompt_sha256"], "split prompt hash"),
        (manifest["split_counts"][RUN.SPLIT]["id_positions_sha256"], sidecar["split_id_positions_sha256"], "split positions hash"),
        (manifest["split_counts"][RUN.SPLIT]["ordered_id_sha256"], sidecar["split_ordered_id_sha256"], "split ordered ID hash"),
        ("sha256(selection_namespace_plus_item_id_ascii)_ascending_v1", sidecar["within_split_order"], "within-split order"),
        (RUN.MODEL_TAG, sidecar["model"], "model tag"),
        (RUN.MODEL_ID, sidecar["model_id"], "model ID"),
        (RUN.MODEL_REVISION, sidecar["model_revision"], "model revision"),
        (RUN.TOKENIZER_REVISION, sidecar["tokenizer_revision"], "tokenizer revision"),
        (QA.CHAT_TEMPLATE_SHA256, sidecar["chat_template_sha256"], "chat-template hash"),
        (True, sidecar["add_generation_prompt"], "generation prompt"),
        (False, sidecar["enable_thinking"], "thinking flag"),
        (RUN.TRANSFORMERS_VERSION, sidecar["transformers_version"], "transformers version"),
        (RUN.TOKENIZERS_VERSION, sidecar["tokenizers_version"], "tokenizers version"),
        (RUN.TORCH_VERSION, sidecar["torch_version"], "torch version"),
        (RUN.NUMPY_VERSION, sidecar["numpy_version"], "numpy version"),
        (RUN.PANDAS_VERSION, sidecar["pandas_version"], "pandas version"),
        (RUN.PYARROW_VERSION, sidecar["pyarrow_version"], "pyarrow version"),
        (RUN.FASTPARQUET_VERSION, sidecar["fastparquet_version"], "fastparquet version"),
        (RUN.PARQUET_ENGINE, sidecar["parquet_engine"], "parquet engine"),
        (RUN.CTX, sidecar["ctx"], "context window"),
        (RUN.MAX_PROMPT_TOKENS, sidecar["max_prompt_tokens"], "max prompt tokens"),
        (RUN.WINDOW, sidecar["window"], "protected window"),
        (float(RUN.BUDGET), float(sidecar["budget"]), "budget"),
        (RUN.DTYPE_NAME, sidecar["dtype"], "dtype"),
        (list(RUN.BIT_LIST), sidecar["bit_list"], "bit list"),
        (RUN.MAXB, sidecar["maxb"], "maxb"),
        (RUN.ROT_SEED, sidecar["rot_seed"], "rotation seed"),
        (RUN.NORM_CORRECT, sidecar["norm_correct"], "norm correction"),
        (RUN.CHUNK, sidecar["chunk"], "chunk"),
        (RUN.N_LAYERS, sidecar["n_layers"], "layers"),
        (RUN.N_ATTENTION_HEADS, sidecar["n_attention_heads"], "attention heads"),
        (RUN.N_KV_HEADS, sidecar["n_kv_heads"], "KV heads"),
        (RUN.HEAD_DIM, sidecar["head_dim"], "head dim"),
        ("48_layers_x_4_kv_x_ctx_len_all_uint8_2", sidecar["uniform_allocation_contract"], "uniform contract"),
        (RUN.SP.IMPL, sidecar["attn_impl"], "structured attention implementation"),
        (RUN.C.IMPL, sidecar["underlying_attn_impl"], "underlying attention implementation"),
        (RUN.SP.CAPTURE_VERSION, sidecar["structured_capture_version"], "structured capture version"),
        (list(QA.CONTENT_GROUPS), sidecar["structured_query_groups"], "structured query groups"),
        (QA.MAX_CONTENT_TOKENS_PER_GROUP, sidecar["structured_max_tokens_per_group"], "structured group token cap"),
        (1.0 / len(QA.CONTENT_GROUPS), float(sidecar["structured_group_weight"]), "structured group weight"),
        (dict(RUN.SCORE_QUERY_SOURCES), sidecar["score_query_sources"], "score query sources"),
        (RUN.WINDOW, sidecar["tail_score_query_count"], "tail score query count"),
        ("sum_manifest_question_A_B_C_D_group_lengths", sidecar["structured_score_query_count_rule"], "structured score query count rule"),
        ("existing_last_prompt_query_sig2", sidecar["structured_interior_noise_query_source"], "structured interior noise query source"),
        (True, sidecar["structured_capture_before_single_prefill"], "structured capture timing"),
        (1, sidecar["layer_context_builds_per_layer"], "layer context build count"),
        ("forced_choice_teacher_forced_argmax", sidecar["decode"], "decode"),
        (1.0, float(sidecar["temperature"]), "temperature"),
        ("official_chat_prompt_end", sidecar["compress_at"], "compression boundary"),
        (list(ARMS), sidecar["arms"], "arms"),
        (list(CANDIDATES), sidecar["candidates"], "candidates"),
        (list(OLD_MENU), sidecar["old_menu"], "old menu"),
        (list(STRUCTURED_MENU), sidecar["structured_menu"], "structured menu"),
        (RUN.R8.ALLOCATION_ID_ALGORITHM, sidecar["allocation_id_algorithm"], "allocation ID algorithm"),
        (list(QA.SCAFFOLD_TOKEN_IDS), sidecar["scaffold_token_ids"], "scaffold tokens"),
        (PD.token_hash(QA.SCAFFOLD_TOKEN_IDS), sidecar["scaffold_token_hash"], "scaffold hash"),
        (dict(QA.CHOICE_BRANCH_IDS), sidecar["choice_branch_ids"], "choice branches"),
        (PD.token_hash([QA.CHOICE_BRANCH_IDS[c] for c in "ABCD"]), sidecar["choice_branch_ids_hash"], "choice branches hash"),
        (len(QA.SCAFFOLD_TOKEN_IDS), sidecar["choice_position"], "choice position"),
        (True, sidecar["no_truncation"], "no truncation"),
        (True, sidecar["no_free_generation"], "no free generation"),
        (True, sidecar["no_raw_logits"], "no raw logits"),
        (True, sidecar["no_proxy"], "no proxy"),
        (True, sidecar["no_labels"], "no labels"),
        (True, sidecar["gold_unused_for_prompt_allocation_execution_and_output"], "gold isolation"),
        ("before_tokenizer_and_rechecked_before_artifact_write", sidecar["snapshot_seal_timing"], "snapshot seal timing"),
        ("before_model_load_and_rechecked_before_artifact_write", sidecar["source_seal_timing"], "source seal timing"),
        (predictions_path.name, sidecar["parquet"], "parquet name"),
        (LB.sha256_file(predictions_path), sidecar["parquet_sha256"], "parquet hash"),
        (RUN.EXPECTED_PREDICTION_ROWS, sidecar["rows"], "rows"),
        (RUN.EXPECTED_PREDICTION_ROWS, sidecar["expected_rows"], "expected rows"),
        (["item_id", "arm"], sidecar["row_key"], "row key"),
        (QA.id_sha256(canonical_entries(manifest)), sidecar["item_ids_sha256"], "item ID hash"),
    )
    for expected, actual, description in checks:
        _same(expected, actual, description)
    if not isinstance(sidecar["device_placements"], list) or not sidecar["device_placements"] or any(not str(value).startswith("cuda") for value in sidecar["device_placements"]):
        _fail("development sidecar does not attest CUDA-only placement")
    elapsed = float(sidecar["elapsed_seconds"])
    if not math.isfinite(elapsed) or elapsed <= 0:
        _fail("development elapsed time must be finite and positive")
    _same(RUN.QUAL.MODEL_SNAPSHOT_ATTESTATION_SCOPE, sidecar["model_snapshot_attestation_scope"], "model snapshot attestation scope")
    _same(snapshot["revision"], sidecar["model_snapshot_inventory_revision"], "model snapshot inventory revision")
    _same(snapshot["file_count"], sidecar["model_snapshot_symlink_file_count"], "model snapshot symlink file count")
    _same(snapshot["inventory_sha256"], sidecar["model_snapshot_symlink_inventory_sha256"], "model snapshot symlink inventory hash")
    _same(snapshot["metadata_sha256"], sidecar["model_snapshot_pinned_metadata_sha256"], "model snapshot pinned metadata hashes")
    for key, field in (
        ("lock_path", "qualification_lock"),
        ("lock_file_sha256", "qualification_lock_file_sha256"),
        ("lock_content_sha256", "qualification_lock_content_sha256"),
        ("qualification_predictions_sha256", "qualification_predictions_sha256"),
        ("qualification_sidecar_sha256", "qualification_sidecar_sha256"),
        ("qualification_source_ledger_file_sha256", "qualification_source_ledger_file_sha256"),
        ("qualification_source_ledger_content_sha256", "qualification_source_ledger_content_sha256"),
    ):
        _same(lock_attestation[key], sidecar[field], f"sidecar {field}")
    _same(RUN.QUALIFICATION_LOCK_VERSION, sidecar["qualification_lock_version"], "qualification lock version")
    _same(_expected_allocation_summary(frame), sidecar["allocation_summary"], "allocation summary")
    _check_source_hashes(sidecar, config_path=config_path, source_ledger_path=source_ledger_path)


def _validate_frame(
    frame: pd.DataFrame, *, manifest: Mapping[str, Any], manifest_file_hash: str
) -> None:
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        _fail(
            "prediction schema drifted: "
            f"got={tuple(frame.columns)}, expected={PREDICTION_COLUMNS}"
        )
    try:
        RUN.validate_output_rows(
            frame.to_dict(orient="records"),
            [str(entry["id"]) for entry in canonical_entries(manifest)],
        )
    except RUN.QwenV7DevelopmentRunnerError as exc:
        _fail(str(exc))
    entries = canonical_entries(manifest)
    entry_by_id = {str(entry["id"]): entry for entry in entries}
    expected_order = [(str(entry["id"]), arm) for entry in entries for arm in ARMS]
    got_order = list(zip(frame["item_id"].astype(str), frame["arm"].astype(str)))
    _same(expected_order, got_order, "canonical prediction row order")
    for row in frame.to_dict(orient="records"):
        item_id, arm = str(row["item_id"]), str(row["arm"])
        entry = entry_by_id[item_id]
        metadata = entry["metadata"]
        expected = {
            "group_id": entry["group_id"], "split": RUN.SPLIT,
            "domain": metadata["domain"], "sub_domain": metadata["sub_domain"],
            "difficulty": metadata["difficulty"], "length": metadata["length"],
            "context_hash": entry["context_hash"], "question_hash": entry["question_hash"],
            "prompt_token_hash": entry["prompt_token_hash"],
            "input_tokens": entry["input_tokens"], "n_prompt_tokens": entry["input_tokens"],
            "truncated": False, "dataset_sha256": LB.DATASET_SHA256,
            "manifest_content_sha256": manifest["content_sha256"],
            "manifest_file_sha256": manifest_file_hash,
            "task_version": RUN.TASK_VERSION, "endpoint_version": RUN.ENDPOINT_VERSION,
            "model": RUN.MODEL_TAG, "model_id": RUN.MODEL_ID,
            "model_revision": RUN.MODEL_REVISION,
            "tokenizer_revision": RUN.TOKENIZER_REVISION,
            "ctx": RUN.CTX, "window": RUN.WINDOW,
            "ctx_len": int(entry["input_tokens"]) - 1 - RUN.WINDOW,
            "maxb": RUN.MAXB,
            "rot_seed": RUN.ROT_SEED, "norm_correct": RUN.NORM_CORRECT,
        }
        for field, value in expected.items():
            _same(value, row[field], f"{item_id}/{arm} {field}")
        _same(RUN.SCORE_QUERY_SOURCES[arm], row["score_query_source"], f"{item_id}/{arm} score query source")
        structured_count = sum(
            len(entry["content_token_positions"][group]) for group in QA.CONTENT_GROUPS
        )
        expected_query_count = (
            0 if arm in {"fp", "uniform"} else
            RUN.WINDOW if arm.startswith("tail_") else structured_count
        )
        _same(expected_query_count, int(row["score_query_count"]), f"{item_id}/{arm} score query count")
        allocation = str(row["allocation_id"])
        if len(allocation) != 64 or any(char not in "0123456789abcdef" for char in allocation):
            _fail(f"{item_id}/{arm} allocation ID is not lowercase SHA-256")
        if arm == "fp":
            _same(FP_ALLOCATION_ID, allocation, f"{item_id} FP allocation ID")
        elif arm == "uniform":
            _same(
                RUN.QUAL.expected_uniform_allocation_id(expected["ctx_len"]),
                allocation, f"{item_id} uniform allocation ID",
            )


def component_bootstrap(
    outcomes: pd.DataFrame, component_order: Sequence[str], *,
    draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    if draws <= 0:
        _fail("bootstrap draws must be positive")
    order = tuple(str(value) for value in component_order)
    if len(order) != RUN.EXPECTED_COMPONENTS or len(set(order)) != len(order):
        _fail("bootstrap component order must contain 52 unique components")
    grouped = outcomes.groupby("group_id", sort=False)
    sizes = grouped.size().reindex(order)
    if sizes.isna().any() or not np.array_equal(sizes.to_numpy(dtype=int), np.ones(len(order), dtype=int)):
        _fail("V7 development bootstrap requires singleton components")
    numeric = [column for column in outcomes.columns if column not in {"item_id", "group_id"}]
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled = rng.integers(0, len(order), size=(draws, len(order)))
    denominator = sizes.to_numpy(dtype=float)[sampled].sum(axis=1)
    result: dict[str, np.ndarray] = {}
    for column in numeric:
        sums = grouped[column].sum().reindex(order).to_numpy(dtype=float)
        result[column] = sums[sampled].sum(axis=1) / denominator
    return result


def _q05(values: np.ndarray) -> float:
    return float(np.quantile(values, BOOTSTRAP_QUANTILE, method="linear"))


def _half_mean(outcomes: pd.DataFrame, ids: frozenset[str], column: str) -> float:
    selected = outcomes.loc[outcomes["item_id"].isin(ids), column]
    if len(selected) != 26:
        _fail(f"half {column} contains {len(selected)} rows, expected 26")
    return float(selected.mean())


def _json_counts(values: Iterable[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return json.dumps(dict(sorted(counts.items())), sort_keys=True, separators=(",", ":"))


def analyze(
    frame: pd.DataFrame, *, manifest: Mapping[str, Any], dataset_path: Path,
) -> dict[str, Any]:
    entries = canonical_entries(manifest)
    order, halves = frozen_halves(entries)
    data = LB.index_by_id(LB.load_dataset(dataset_path, authenticate=True))
    choices = frame.pivot(index="item_id", columns="arm", values="forced_choice").reindex(index=order, columns=ARMS)
    if choices.isna().any().any():
        _fail("validated predictions do not pivot over every development item/arm")
    gold = pd.Series({item_id: str(data[item_id]["answer"]) for item_id in order})
    correct = choices.eq(gold, axis=0)
    candidate_accuracy = {arm: float(correct[arm].mean()) for arm in CANDIDATES}
    f_star_policy = max(CANDIDATES, key=lambda arm: candidate_accuracy[arm])
    all_oracle = correct[list(CANDIDATES)].any(axis=1)
    old_oracle = correct[list(OLD_MENU)].any(axis=1)
    outcomes = pd.DataFrame({
        "item_id": list(order),
        "group_id": [str(next(row["group_id"] for row in entries if str(row["id"]) == item_id)) for item_id in order],
        "fp": correct["fp"].to_numpy(dtype=float),
        "F_star": correct[f_star_policy].to_numpy(dtype=float),
        "all_oracle": all_oracle.to_numpy(dtype=float),
        "old_oracle": old_oracle.to_numpy(dtype=float),
    })
    outcomes["H"] = outcomes["all_oracle"] - outcomes["F_star"]
    outcomes["S"] = outcomes["all_oracle"] - outcomes["old_oracle"]
    bootstrap = component_bootstrap(outcomes, [str(next(row["group_id"] for row in entries if str(row["id"]) == item_id)) for item_id in order])
    fp_accuracy = float(outcomes["fp"].mean())
    f_star_accuracy = float(outcomes["F_star"].mean())
    all_oracle_accuracy = float(outcomes["all_oracle"].mean())
    old_oracle_accuracy = float(outcomes["old_oracle"].mean())
    H = float(outcomes["H"].mean())
    S = float(outcomes["S"].mean())
    H_halves = [_half_mean(outcomes, half, "H") for half in halves]
    S_halves = [_half_mean(outcomes, half, "S") for half in halves]
    fp_q05, f_star_q05 = _q05(bootstrap["fp"]), _q05(bootstrap["F_star"])
    H_q05, S_q05 = _q05(bootstrap["H"]), _q05(bootstrap["S"])
    competence = {
        "fp_point_gate": fp_accuracy >= FP_ACCURACY_MIN,
        "fp_bootstrap_gate": fp_q05 > CHANCE,
        "F_star_point_gate": F_STAR_ACCURACY_MIN <= f_star_accuracy <= F_STAR_ACCURACY_MAX,
        "F_star_bootstrap_gate": f_star_q05 > CHANCE,
    }
    mechanism_gate = S >= S_MIN
    opportunity = {
        "H_point_gate": H >= H_MIN,
        "H_bootstrap_gate": H_q05 > 0.0,
        "H_half1_gate": H_halves[0] >= H_HALF_MIN,
        "H_half2_gate": H_halves[1] >= H_HALF_MIN,
    }
    if not all(competence.values()):
        decision = "stop_v7_invalid_operating_point"
    elif not mechanism_gate:
        decision = "stop_v7_no_mechanism_effect"
    elif not all(opportunity.values()):
        decision = "stop_v7_no_opportunity"
    else:
        decision = "advance_v7_confirmation"

    paired: dict[str, Any] = {}
    for allocator in ("evict", "interior"):
        tail, structured = f"tail_{allocator}", f"structured_{allocator}"
        paired[f"structured_minus_tail_{allocator}_accuracy"] = candidate_accuracy[structured] - candidate_accuracy[tail]
        paired[f"structured_{allocator}_rescues_over_tail"] = int((correct[structured] & ~correct[tail]).sum())
        paired[f"structured_{allocator}_harms_vs_tail"] = int((~correct[structured] & correct[tail]).sum())
        paired[f"structured_{allocator}_different_answer_count"] = int((choices[structured] != choices[tail]).sum())

    vectors = choices.apply(lambda row: "".join(str(row[arm]) for arm in ARMS), axis=1)
    allocation_stats: dict[str, Any] = {}
    for arm in ARMS:
        values = frame.loc[frame["arm"].eq(arm)].set_index("item_id").reindex(order)
        allocation_stats[arm] = {
            "unique_allocation_ids": int(values["allocation_id"].nunique()),
            "mean_bits_per_token": float(values["bits_per_token"].mean()),
            "mean_evict_frac": float(values["evict_frac"].mean()),
        }
    for allocator in ("evict", "interior"):
        left = frame.loc[frame["arm"].eq(f"tail_{allocator}")].set_index("item_id").reindex(order)["allocation_id"]
        right = frame.loc[frame["arm"].eq(f"structured_{allocator}")].set_index("item_id").reindex(order)["allocation_id"]
        allocation_stats[f"structured_equals_tail_{allocator}_count"] = int(left.eq(right).sum())

    return {
        "protocol_version": RUN.PROTOCOL_VERSION,
        "split": RUN.SPLIT,
        "n_items": RUN.EXPECTED_ITEMS,
        "n_components": RUN.EXPECTED_COMPONENTS,
        "prediction_rows": len(frame),
        "F_star_policy": f_star_policy,
        "fp_correct": int(outcomes["fp"].sum()), "fp_accuracy": fp_accuracy,
        "fp_bootstrap_q05": fp_q05,
        "F_star_correct": int(outcomes["F_star"].sum()), "F_star_accuracy": f_star_accuracy,
        "F_star_bootstrap_q05": f_star_q05,
        "all_candidate_oracle_correct": int(outcomes["all_oracle"].sum()),
        "all_candidate_oracle_accuracy": all_oracle_accuracy,
        "old_menu_oracle_correct": int(outcomes["old_oracle"].sum()),
        "old_menu_oracle_accuracy": old_oracle_accuracy,
        "H": H, "H_bootstrap_q05": H_q05,
        "H_half1": H_halves[0], "H_half2": H_halves[1],
        "S": S, "S_bootstrap_q05": S_q05,
        "S_half1": S_halves[0], "S_half2": S_halves[1],
        **competence, "S_point_gate": mechanism_gate, **opportunity,
        "candidate_accuracies_json": json.dumps(candidate_accuracy, sort_keys=True, separators=(",", ":")),
        "answer_vector_classes_json": _json_counts(vectors),
        "allocation_stats_json": json.dumps(allocation_stats, sort_keys=True, separators=(",", ":")),
        **paired,
        "decision": decision,
    }


def thresholds() -> dict[str, Any]:
    return dict(RUN.QUALIFICATION_THRESHOLDS)


def build_confirmation_lock(
    *, summary: Mapping[str, Any], predictions_path: Path,
    sidecar_path: Path, sidecar: Mapping[str, Any], manifest_path: Path,
    legacy_manifest_path: Path, config_path: Path,
    qualification_lock_path: Path, source_ledger_path: Path,
) -> dict[str, Any] | None:
    if summary["decision"] != "advance_v7_confirmation":
        return None
    manifest = _load_object(manifest_path, "V7 manifest for confirmation lock")
    qualification_lock = _load_object(qualification_lock_path, "qualification lock for confirmation lock")
    reader_path = _regular(__file__, "V7 development reader")
    source_ledger = _regular(source_ledger_path, "V7 development source ledger")
    lock: dict[str, Any] = {
        "lock_version": CONFIRMATION_LOCK_VERSION,
        "protocol_version": RUN.PROTOCOL_VERSION,
        "runner_version": RUN.RUNNER_VERSION,
        "reader_version": READER_VERSION,
        "task_version": RUN.TASK_VERSION,
        "endpoint_version": RUN.ENDPOINT_VERSION,
        "decision": "advance_v7_confirmation",
        "F_star_policy": summary["F_star_policy"],
        "development": {
            "predictions": str(predictions_path),
            "predictions_sha256": LB.sha256_file(predictions_path),
            "sidecar": str(sidecar_path),
            "sidecar_sha256": LB.sha256_file(sidecar_path),
            "summary": dict(summary),
        },
        "authorization": {
            "qualification_lock": str(qualification_lock_path),
            "qualification_lock_file_sha256": LB.sha256_file(qualification_lock_path),
            "qualification_lock_content_sha256": qualification_lock["content_sha256"],
        },
        "provenance": {
            "dataset_sha256": LB.DATASET_SHA256,
            "dataset_bytes": LB.DATASET_BYTES,
            "legacy_manifest": str(legacy_manifest_path),
            "legacy_manifest_file_sha256": LB.sha256_file(legacy_manifest_path),
            "qwen_manifest": str(manifest_path),
            "qwen_manifest_version": QA.MANIFEST_VERSION,
            "qwen_manifest_content_sha256": manifest["content_sha256"],
            "qwen_manifest_file_sha256": LB.sha256_file(manifest_path),
            "config": str(config_path), "config_sha256": LB.sha256_file(config_path),
            "development_source_ledger": str(source_ledger),
            "development_source_ledger_file_sha256": LB.sha256_file(source_ledger),
            "development_source_ledger_content_sha256": sidecar["source_ledger_content_sha256"],
            "executed_source_sha256": dict(sidecar["executed_source_sha256"]),
            "runner_source": sidecar["runner_source"],
            "runner_source_sha256": sidecar["runner_source_sha256"],
            "reader_source": str(reader_path.relative_to(ROOT.resolve())),
            "reader_source_sha256": LB.sha256_file(reader_path),
            "model_snapshot_attestation_scope": sidecar["model_snapshot_attestation_scope"],
            "model_snapshot_inventory_revision": sidecar["model_snapshot_inventory_revision"],
            "model_snapshot_symlink_file_count": sidecar["model_snapshot_symlink_file_count"],
            "model_snapshot_symlink_inventory_sha256": sidecar["model_snapshot_symlink_inventory_sha256"],
            "model_snapshot_pinned_metadata_sha256": sidecar["model_snapshot_pinned_metadata_sha256"],
        },
        "partitions": {
            "development": RUN._partition_record(manifest, split="development", first_half=26),
            "confirmation": RUN._partition_record(manifest, split="confirmation", first_half=22),
            "within_split_order": "sha256(selection_namespace_plus_item_id_ascii)_ascending_v1",
        },
        "execution": RUN.development_execution(),
        "thresholds": thresholds(),
    }
    lock["content_sha256"] = _content_sha256(lock)
    return lock


def write_confirmation_lock(path: str | Path, lock: Mapping[str, Any] | None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        _fail(f"refusing symlink confirmation-lock target {destination}")
    if lock is None:
        if destination.exists():
            _fail(f"refusing stale confirmation lock after non-advance: {destination}")
        return
    _same("advance_v7_confirmation", lock.get("decision"), "confirmation-lock decision")
    _same(_content_sha256(lock), lock.get("content_sha256"), "confirmation-lock content hash")
    payload = (json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    if destination.exists():
        if destination.is_file() and destination.read_bytes() == payload:
            return
        _fail(f"refusing different existing confirmation lock {destination}")
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def print_summary(summary: Mapping[str, Any]) -> None:
    print("V7 Qwen structured-query development (frozen section 3L)")
    print("provenance      PASS (dataset/manifest/model/lock/source hashes)")
    print(f"rows            {summary['prediction_rows']} predictions; {summary['n_items']} singleton items")
    print(f"FP              {summary['fp_correct']}/52 = {summary['fp_accuracy']:.3f}, q05={summary['fp_bootstrap_q05']:.3f}")
    print(f"F*              {summary['F_star_policy']}: {summary['F_star_correct']}/52 = {summary['F_star_accuracy']:.3f}, q05={summary['F_star_bootstrap_q05']:.3f}")
    print(f"all oracle      {summary['all_candidate_oracle_correct']}/52 = {summary['all_candidate_oracle_accuracy']:.3f}")
    print(f"old oracle      {summary['old_menu_oracle_correct']}/52 = {summary['old_menu_oracle_accuracy']:.3f}")
    print(f"S mechanism     {summary['S']:.3f}, q05={summary['S_bootstrap_q05']:.3f}, halves={summary['S_half1']:.3f}/{summary['S_half2']:.3f}")
    print(f"H opportunity   {summary['H']:.3f}, q05={summary['H_bootstrap_q05']:.3f}, halves={summary['H_half1']:.3f}/{summary['H_half2']:.3f}")
    print(f"structured-tail evict    {summary['structured_minus_tail_evict_accuracy']:+.3f}; rescues/harms={summary['structured_evict_rescues_over_tail']}/{summary['structured_evict_harms_vs_tail']}")
    print(f"structured-tail interior {summary['structured_minus_tail_interior_accuracy']:+.3f}; rescues/harms={summary['structured_interior_rescues_over_tail']}/{summary['structured_interior_harms_vs_tail']}")
    print(f"decision        {summary['decision']}")


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    predictions_path = _regular(args.predictions, "V7 development prediction parquet")
    sidecar_path = _regular(predictions_path.with_suffix(".json"), "V7 development sidecar")
    dataset_path = _regular(args.dataset, "LongBench-v2 dataset")
    legacy_manifest_path = _regular(args.legacy_manifest, "legacy V6 manifest")
    manifest_path = _regular(args.manifest, "V7 Qwen manifest")
    config_path = _regular(args.config, "model config")
    runner_path = _regular(args.runner, "V7 development runner")
    _same(Path(RUN.__file__).resolve(), runner_path, "development runner argument")
    qualification_lock_path = _regular(args.qualification_lock, "V7 qualification lock")
    source_ledger_path = _regular(args.source_ledger, "V7 development source ledger")
    if source_ledger_path != RUN.DEFAULT_SOURCE_LEDGER.resolve():
        _fail(
            f"source ledger must be the canonical path {RUN.DEFAULT_SOURCE_LEDGER.resolve()}"
        )
    try:
        _, lock_attestation = RUN.authenticate_qualification_lock(
            qualification_lock_path, dataset_path=dataset_path,
            legacy_manifest_path=legacy_manifest_path, manifest_path=manifest_path,
            config_path=config_path, model_source=args.tokenizer,
        )
    except RUN.QwenV7DevelopmentRunnerError as exc:
        _fail(str(exc))
    tokenizer = QA.load_tokenizer(args.tokenizer)
    manifest, manifest_file_hash = QA.authenticate_manifest(
        manifest_path, dataset_path=dataset_path,
        legacy_manifest_path=legacy_manifest_path, tokenizer=tokenizer,
    )
    snapshot = QA.authenticate_snapshot(args.tokenizer)
    try:
        frame = pd.read_parquet(predictions_path, engine=RUN.PARQUET_ENGINE)
    except Exception as exc:
        _fail(f"cannot read V7 development parquet: {exc}")
    _validate_frame(frame, manifest=manifest, manifest_file_hash=manifest_file_hash)
    sidecar = _load_object(sidecar_path, "V7 development sidecar")
    _check_sidecar(
        sidecar, predictions_path=predictions_path, manifest_path=manifest_path,
        manifest=manifest, manifest_file_hash=manifest_file_hash,
        legacy_manifest_path=legacy_manifest_path, config_path=config_path,
        source_ledger_path=source_ledger_path, lock_attestation=lock_attestation,
        snapshot=snapshot, frame=frame,
    )
    if args.validate_only:
        if args.csv or args.lock:
            _fail("--validate-only forbids --csv and --lock")
        print(f"PASS V7 development artifact: {len(frame)} rows, 52 singleton items")
        return None
    summary = analyze(frame, manifest=manifest, dataset_path=dataset_path)
    print_summary(summary)
    if args.csv:
        destination = Path(args.csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{key: _csv_value(value) for key, value in summary.items()}]).to_csv(destination, index=False)
    if args.lock:
        lock = build_confirmation_lock(
            summary=summary, predictions_path=predictions_path,
            sidecar_path=sidecar_path, sidecar=sidecar,
            manifest_path=manifest_path, legacy_manifest_path=legacy_manifest_path,
            config_path=config_path, qualification_lock_path=qualification_lock_path,
            source_ledger_path=source_ledger_path,
        )
        write_confirmation_lock(args.lock, lock)
        if lock is not None:
            print(f"lock            {Path(args.lock).resolve()}")
        else:
            print("lock            none (development did not advance)")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--legacy-manifest", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--qualification-lock", required=True)
    parser.add_argument("--source-ledger", default=str(RUN.DEFAULT_SOURCE_LEDGER))
    parser.add_argument("--config", default=str(ROOT / "h0_measurement/models.yaml"))
    parser.add_argument("--runner", default=str(ROOT / "h0_measurement/run_longbench_v2_qwen_v7_development.py"))
    parser.add_argument("--csv", default="")
    parser.add_argument("--lock", default="")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    try:
        run(parser.parse_args())
    except (QwenV7DevelopmentReaderError, QA.V7ManifestError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
