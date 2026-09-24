#!/usr/bin/env python3
"""Strict reader for V5 LongBench-v2 forced-choice development artifacts.

The runner is deliberately label blind.  It writes one scalar-only prediction
table and one scalar-only branch-blind proxy table.  This reader authenticates
the frozen manifest and dataset, joins the released answers in memory, and
applies the preregistered competence, opportunity, and proxy-transfer gates.

A valid experiment decision exits zero.  Artifact or provenance drift exits
two.  No result artifact contains a gold answer, correctness label, response,
or raw logit/probability vector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h0_measurement import audit_longbench_v2 as AUDIT  # noqa: E402
from sievelib import policy_diagnostic as PD  # noqa: E402
from sievelib import tasks_longbench_v2 as LB  # noqa: E402


SPLIT = "development"
MODEL = "llama31-8b"
MODEL_ID = LB.MODEL_ID
CTX = LB.CONTEXT_WINDOW
MAX_INPUT = LB.MAX_INPUT_TOKENS
WINDOW = 32
BUDGET = 2.0
MAXB = 8
PROTOCOL_VERSION = "longbench_v2_sieve_v5_forced_choice_v1"
RUNNER_VERSION = "longbench_v2_forced_choice_runner_v1"
ENDPOINT_VERSION = "canonical_abcd_forced_choice_v1"
PROXY_RULE_VERSION = "branch_blind_scaffold_kl_v1"
DECODE = "forced_choice_teacher_forced_argmax"
COMPRESS_AT = "official_chat_prompt_end"
ALLOCATION_ID_ALGORITHM = "sha256_layer_uint8_v1"
ARMS = (
    "fp", "uniform", "evict", "interior", "interior_pool",
    "interior_cascade", "obcache_k", "obck_ada", "laprox",
)
CANDIDATES = ARMS[1:]
RAW_ARMS = (
    "fp", "uniform", "evict", "interior", "interior_pool",
    "interior_cascade", "obcache_k",
    "obcache_k:alloc=ada@obck_ada", "laprox",
)
EXPECTED_ROWS = 52
EXPECTED_COMPONENTS = 44
EXPECTED_PREDICTION_ROWS = EXPECTED_ROWS * len(ARMS)
EXPECTED_PROXY_ROWS = EXPECTED_ROWS * len(CANDIDATES)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
FP_ALLOCATION_ID = hashlib.sha256(b"fp16").hexdigest()
MANIFEST_CONTENT_SHA256 = (
    "ee951a2c7b2c256125e73de4fba6f7e14485a566bbf6096c14c6e4f23e6407ff"
)
MANIFEST_FILE_SHA256 = (
    "bd7b358a91081fd5657a2aa822a4c814ebfeb85f410d55cee25aa9f54223798e"
)
SPLIT_ID_SHA256 = (
    "98bf6531b62e474a05b66f488b298d51b7c1ecb53e4e0e97c23495e8dacace35"
)
SPLIT_ID_TOKEN_SHA256 = (
    "820ea99fdea86db20e4e1e9d297c7c174eeb78a48b6325968c755f4cd4eb893a"
)
HALF_COMPONENT_COUNTS = (22, 22)
HALF_ROW_COUNTS = (28, 24)
HALF_ID_SHA256 = (
    "b006a353c33488d661b87932cf1c4abf8ce1e7aff63e688810a74b13a85184b8",
    "6adf6c4b4e378a4f4605a4a959df0766fad7329f384bf27c8734abaae1296a1b",
)
SCAFFOLD_TOKEN_IDS = (791, 4495, 4320, 374, 320)
CHOICE_BRANCH_IDS = {"A": 32, "B": 33, "C": 34, "D": 35}
SCAFFOLD_TOKEN_HASH = PD.token_hash(SCAFFOLD_TOKEN_IDS)
CHOICE_BRANCH_IDS_HASH = PD.token_hash(
    [CHOICE_BRANCH_IDS[choice] for choice in "ABCD"]
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")

COMMON_COLUMNS = (
    "item_id", "group_id", "split", "domain", "sub_domain", "difficulty",
    "length", "context_hash", "question_hash", "prompt_token_hash",
    "input_tokens", "n_prompt_tokens", "truncated", "dataset_sha256",
    "manifest_content_sha256", "manifest_file_sha256", "task_version",
    "endpoint_version", "model", "model_id", "model_revision",
    "tokenizer_revision", "ctx", "window",
)

PREDICTION_COLUMNS = COMMON_COLUMNS + (
    "B", "arm", "arm_order", "forced_choice", "choice_index",
    "choice_entropy", "choice_margin", "choice_max_probability",
    "scaffold_token_hash", "choice_branch_ids_hash", "bits_per_token",
    "evict_frac", "allocation_id", "ctx_len", "observed_queries", "maxb",
    "rot_seed", "norm_correct",
)

SCAFFOLD_KL_COLUMNS = tuple(f"scaffold_kl_pos{index}" for index in range(5))
PROXY_COLUMNS = COMMON_COLUMNS + (
    "B", "candidate", "candidate_order",
) + SCAFFOLD_KL_COLUMNS + (
    "scaffold_mean_kl", "proxy_rule_version", "scaffold_token_hash",
    "choice_branch_ids_hash", "allocation_id", "selected_policy",
    "fp_canonical_choice", "candidate_canonical_choice",
)

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
    "runner_version", "protocol_version", "dataset_repo", "dataset_revision",
    "dataset_sha256", "dataset_bytes", "official_code_repo",
    "official_code_revision", "official_prompt_sha256", "task_version",
    "endpoint_version", "context_hash_version", "question_hash_version",
    "manifest", "manifest_version", "manifest_content_sha256",
    "manifest_file_sha256", "eligible_list_sha256", "split", "split_count",
    "split_components", "split_ids_sha256", "split_id_token_sha256",
    "component_order", "half_component_counts", "half_row_counts",
    "half_id_sha256", "model", "model_id", "model_revision",
    "tokenizer_revision", "ctx", "max_input_tokens", "decode", "temperature",
    "window", "budget", "cascade_bits", "allocator_budget_rule",
    "compress_at", "raw_arms", "arms", "candidates",
    "allocation_id_algorithm", "chat_template_sha256", "dtype", "bit_list",
    "maxb", "rot_seed", "norm_correct", "chunk", "attn_impl",
    "transformers_version", "tokenizers_version", "no_truncation",
    "no_raw_logits", "no_labels", "no_free_generation", "proxy_dtype",
    "scaffold_token_ids", "scaffold_token_hash", "choice_branch_ids",
    "choice_branch_ids_hash", "proxy_rule_version", "proxy_positions",
    "choice_position", "artifact", "parquet", "parquet_sha256", "rows",
    "expected_rows", "row_key", "paired_artifact", "paired_parquet",
    "paired_parquet_sha256", "paired_rows",
)


class LongBenchForcedChoiceError(ValueError):
    """A V5 artifact violates the frozen forced-choice contract."""


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


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {label} {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must contain one JSON object")
    return value


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


def _protocol(manifest: Mapping[str, Any]) -> None:
    try:
        protocol = manifest["protocol"]
        dataset = protocol["dataset"]
        official = protocol["official_evaluation"]
        model = protocol["model"]
        eligibility = protocol["eligibility"]
        grouping = protocol["grouping"]
    except (KeyError, TypeError) as exc:
        _fail(f"manifest protocol schema is incomplete: {exc}")
    checks = (
        (LB.DATASET_REPO, dataset.get("repo"), "manifest dataset repo"),
        (LB.DATASET_REVISION, dataset.get("revision"), "manifest dataset revision"),
        (LB.DATASET_SHA256, dataset.get("sha256"), "manifest dataset SHA-256"),
        (LB.DATASET_BYTES, dataset.get("bytes"), "manifest dataset bytes"),
        (LB.OFFICIAL_CODE_REPO, official.get("code_repo"), "manifest official code repo"),
        (LB.OFFICIAL_CODE_REVISION, official.get("code_revision"), "manifest official revision"),
        (LB.OFFICIAL_PROMPT_SHA256, official.get("prompt_sha256"), "manifest prompt SHA-256"),
        (LB.TASK_VERSION, official.get("task_version"), "manifest task version"),
        (LB.MODEL_ID, model.get("id"), "manifest model id"),
        (LB.MODEL_REVISION, model.get("revision"), "manifest model revision"),
        (LB.TRANSFORMERS_VERSION, model.get("transformers_version"), "manifest transformers version"),
        (LB.TOKENIZERS_VERSION, model.get("tokenizers_version"), "manifest tokenizers version"),
        (LB.CHAT_TEMPLATE_SHA256, model.get("chat_template_sha256"), "manifest chat-template SHA-256"),
        (CTX, eligibility.get("context_window"), "manifest context window"),
        (MAX_INPUT, eligibility.get("max_rendered_input_tokens"), "manifest max input"),
        (LB.CONTEXT_HASH_VERSION, grouping.get("context_hash_version"), "manifest context-hash version"),
        (LB.QUESTION_HASH_VERSION, grouping.get("question_hash_version"), "manifest question-hash version"),
        (LB.SPLIT_VERSION, grouping.get("split_version"), "manifest split version"),
        (LB.SPLIT_NAMESPACE.decode("utf-8"), grouping.get("split_namespace_utf8"), "manifest split namespace"),
    )
    for expected, actual, description in checks:
        _same(expected, actual, description)


def authenticate_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    manifest_path = Path(path).resolve()
    manifest = _load_json(manifest_path, "manifest")
    try:
        AUDIT.verify_manifest_content_hash(manifest)
    except (KeyError, TypeError, ValueError) as exc:
        _fail(f"manifest content authentication failed: {exc}")
    _same(LB.MANIFEST_VERSION, manifest.get("manifest_version"), "manifest version")
    _same(
        MANIFEST_CONTENT_SHA256, manifest.get("content_sha256"),
        "frozen manifest content SHA-256",
    )
    file_hash = _sha256_file(manifest_path)
    _same(MANIFEST_FILE_SHA256, file_hash, "frozen manifest file SHA-256")
    _protocol(manifest)
    serialized = json.dumps(manifest, sort_keys=True)
    if '"answer"' in serialized or '"gold_answer"' in serialized:
        _fail("manifest must be gold-free")

    examples = manifest.get("examples")
    if not isinstance(examples, list) or len(examples) != LB.EXPECTED_ELIGIBLE_ROWS:
        _fail(
            f"manifest must contain exactly {LB.EXPECTED_ELIGIBLE_ROWS} eligible rows"
        )
    _same(
        LB.EXPECTED_ELIGIBLE_LIST_SHA256,
        manifest.get("eligible_list_sha256"),
        "manifest eligible-list SHA-256",
    )
    if AUDIT.eligible_list_sha256(examples) != LB.EXPECTED_ELIGIBLE_LIST_SHA256:
        _fail("manifest rows do not reproduce the eligible-list SHA-256")

    exact_keys = {"id", "group_id", "context_hash", "input_tokens", "metadata", "split"}
    seen: set[str] = set()
    group_splits: dict[str, str] = {}
    for index, row in enumerate(examples):
        if not isinstance(row, dict) or set(row) != exact_keys:
            _fail(f"manifest example {index} does not have the exact gold-free schema")
        item_id = row["id"]
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            _fail(f"manifest contains empty/duplicate item id {item_id!r}")
        seen.add(item_id)
        group_id = row["group_id"]
        if not isinstance(group_id, str) or not SHA256_RE.fullmatch(group_id):
            _fail(f"manifest item {item_id} has invalid group id")
        if group_id in group_splits and group_splits[group_id] != row["split"]:
            _fail(f"manifest component {group_id} crosses splits")
        group_splits[group_id] = row["split"]
        if row["split"] not in AUDIT.SPLIT_NAMES:
            _fail(f"manifest item {item_id} has invalid split {row['split']!r}")
        if not isinstance(row["context_hash"], str) or not SHA256_RE.fullmatch(
            row["context_hash"]
        ):
            _fail(f"manifest item {item_id} has invalid context hash")
        tokens = row["input_tokens"]
        if isinstance(tokens, bool) or not isinstance(tokens, int) or not 0 < tokens <= MAX_INPUT:
            _fail(f"manifest item {item_id} has invalid input_tokens {tokens!r}")

    development = [row for row in examples if row["split"] == SPLIT]
    _same(EXPECTED_ROWS, len(development), "development manifest row count")
    _same(SPLIT_ID_SHA256, _id_hash(development), "development ID SHA-256")
    _same(
        SPLIT_ID_TOKEN_SHA256, _split_id_token_hash(development),
        "development id/token SHA-256",
    )
    split_record = manifest.get("split_counts", {}).get(SPLIT)
    if not isinstance(split_record, dict):
        _fail("manifest development split summary is absent")
    for field, expected in (
        ("rows", EXPECTED_ROWS), ("groups", EXPECTED_COMPONENTS),
        ("id_sha256", SPLIT_ID_SHA256),
        ("id_token_sha256", SPLIT_ID_TOKEN_SHA256),
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
        if row["metadata"] != metadata:
            _fail(f"manifest metadata disagrees with dataset for {item_id}")
        if LB.context_hash(item) != row["context_hash"]:
            _fail(f"manifest context hash disagrees with dataset for {item_id}")
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


def _sidecar_constant_checks(
    sidecar: Mapping[str, Any], *, artifact: str, parquet: Path,
    paired_artifact: str, paired_parquet: Path, manifest_path: Path,
    manifest: Mapping[str, Any], manifest_file_hash: str,
    expected_rows: int, row_key: Sequence[str],
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
        (LB.DATASET_REPO, sidecar["dataset_repo"], "dataset repo"),
        (LB.DATASET_REVISION, sidecar["dataset_revision"], "dataset revision"),
        (LB.DATASET_SHA256, sidecar["dataset_sha256"], "dataset SHA-256"),
        (LB.DATASET_BYTES, sidecar["dataset_bytes"], "dataset bytes"),
        (LB.OFFICIAL_CODE_REPO, sidecar["official_code_repo"], "official code repo"),
        (LB.OFFICIAL_CODE_REVISION, sidecar["official_code_revision"], "official revision"),
        (LB.OFFICIAL_PROMPT_SHA256, sidecar["official_prompt_sha256"], "prompt SHA-256"),
        (LB.TASK_VERSION, sidecar["task_version"], "task version"),
        (ENDPOINT_VERSION, sidecar["endpoint_version"], "endpoint version"),
        (LB.CONTEXT_HASH_VERSION, sidecar["context_hash_version"], "context-hash version"),
        (LB.QUESTION_HASH_VERSION, sidecar["question_hash_version"], "question-hash version"),
        (LB.MANIFEST_VERSION, sidecar["manifest_version"], "manifest version"),
        (manifest["content_sha256"], sidecar["manifest_content_sha256"], "manifest content hash"),
        (manifest_file_hash, sidecar["manifest_file_sha256"], "manifest file hash"),
        (LB.EXPECTED_ELIGIBLE_LIST_SHA256, sidecar["eligible_list_sha256"], "eligible-list hash"),
        (SPLIT, sidecar["split"], "split"),
        (EXPECTED_ROWS, sidecar["split_count"], "split count"),
        (EXPECTED_COMPONENTS, sidecar["split_components"], "split component count"),
        (SPLIT_ID_SHA256, sidecar["split_ids_sha256"], "split ID SHA-256"),
        (SPLIT_ID_TOKEN_SHA256, sidecar["split_id_token_sha256"], "split id/token SHA-256"),
        ("min_sha256_split_namespace_context_hash_then_component_id_v1", sidecar["component_order"], "component order"),
        (list(HALF_COMPONENT_COUNTS), sidecar["half_component_counts"], "half component counts"),
        (list(HALF_ROW_COUNTS), sidecar["half_row_counts"], "half row counts"),
        (list(HALF_ID_SHA256), sidecar["half_id_sha256"], "half ID hashes"),
        (MODEL, sidecar["model"], "model tag"),
        (MODEL_ID, sidecar["model_id"], "model id"),
        (LB.MODEL_REVISION, sidecar["model_revision"], "model revision"),
        (LB.MODEL_REVISION, sidecar["tokenizer_revision"], "tokenizer revision"),
        (CTX, sidecar["ctx"], "context length"),
        (MAX_INPUT, sidecar["max_input_tokens"], "maximum input tokens"),
        (DECODE, sidecar["decode"], "decode"),
        (1.0, sidecar["temperature"], "temperature"),
        (WINDOW, sidecar["window"], "protected window"),
        (BUDGET, sidecar["budget"], "budget"),
        (4, sidecar["cascade_bits"], "cascade bits"),
        ("feasible", sidecar["allocator_budget_rule"], "allocator rule"),
        (COMPRESS_AT, sidecar["compress_at"], "compression point"),
        (list(RAW_ARMS), sidecar["raw_arms"], "raw arm order"),
        (list(ARMS), sidecar["arms"], "arm order"),
        (list(CANDIDATES), sidecar["candidates"], "candidate order"),
        (ALLOCATION_ID_ALGORITHM, sidecar["allocation_id_algorithm"], "allocation ID algorithm"),
        (LB.CHAT_TEMPLATE_SHA256, sidecar["chat_template_sha256"], "chat-template hash"),
        ("bfloat16", sidecar["dtype"], "model dtype"),
        ([1, 2, 3, 4, 5, 6, 8], sidecar["bit_list"], "bit-width list"),
        (MAXB, sidecar["maxb"], "maximum bit width"),
        (0, sidecar["rot_seed"], "rotation seed"),
        (True, sidecar["norm_correct"], "norm correction"),
        (4096, sidecar["chunk"], "prefill chunk"),
        ("sieve_compress", sidecar["attn_impl"], "attention implementation"),
        (LB.TRANSFORMERS_VERSION, sidecar["transformers_version"], "transformers version"),
        (LB.TOKENIZERS_VERSION, sidecar["tokenizers_version"], "tokenizers version"),
        (True, sidecar["no_truncation"], "no-truncation flag"),
        (True, sidecar["no_raw_logits"], "no-raw-logits flag"),
        (True, sidecar["no_labels"], "no-labels flag"),
        (True, sidecar["no_free_generation"], "no-free-generation flag"),
        ("float32", sidecar["proxy_dtype"], "proxy arithmetic dtype"),
        (list(SCAFFOLD_TOKEN_IDS), sidecar["scaffold_token_ids"], "scaffold token IDs"),
        (SCAFFOLD_TOKEN_HASH, sidecar["scaffold_token_hash"], "scaffold token hash"),
        (CHOICE_BRANCH_IDS, sidecar["choice_branch_ids"], "choice branch IDs"),
        (CHOICE_BRANCH_IDS_HASH, sidecar["choice_branch_ids_hash"], "choice branch hash"),
        (PROXY_RULE_VERSION, sidecar["proxy_rule_version"], "proxy rule version"),
        ([0, 1, 2, 3, 4], sidecar["proxy_positions"], "proxy positions"),
        (5, sidecar["choice_position"], "choice position"),
        (artifact, sidecar["artifact"], "artifact kind"),
        (parquet.name, sidecar["parquet"], "parquet filename"),
        (_sha256_file(parquet), sidecar["parquet_sha256"], "parquet SHA-256"),
        (expected_rows, sidecar["rows"], "row count"),
        (expected_rows, sidecar["expected_rows"], "expected row count"),
        (list(row_key), sidecar["row_key"], "row key"),
        (paired_artifact, sidecar["paired_artifact"], "paired artifact kind"),
        (paired_parquet.name, sidecar["paired_parquet"], "paired parquet filename"),
        (_sha256_file(paired_parquet), sidecar["paired_parquet_sha256"], "paired parquet SHA-256"),
        (
            EXPECTED_PROXY_ROWS if artifact == "forced_choice_predictions"
            else EXPECTED_PREDICTION_ROWS,
            sidecar["paired_rows"], "paired row count",
        ),
    )
    for expected, actual, description in checks:
        _same(expected, actual, f"{artifact} sidecar {description}")
    if Path(str(sidecar["manifest"])).resolve() != manifest_path.resolve():
        _fail(f"{artifact} sidecar points at a different manifest")


def validate_sidecars(
    prediction_sidecar: Mapping[str, Any], proxy_sidecar: Mapping[str, Any], *,
    prediction_path: Path, proxy_path: Path, manifest_path: Path,
    manifest: Mapping[str, Any], manifest_file_hash: str,
) -> None:
    _sidecar_constant_checks(
        prediction_sidecar, artifact="forced_choice_predictions",
        parquet=prediction_path, paired_artifact="scaffold_proxy",
        paired_parquet=proxy_path, manifest_path=manifest_path,
        manifest=manifest, manifest_file_hash=manifest_file_hash,
        expected_rows=EXPECTED_PREDICTION_ROWS, row_key=("item_id", "arm"),
    )
    _sidecar_constant_checks(
        proxy_sidecar, artifact="scaffold_proxy", parquet=proxy_path,
        paired_artifact="forced_choice_predictions", paired_parquet=prediction_path,
        manifest_path=manifest_path, manifest=manifest,
        manifest_file_hash=manifest_file_hash, expected_rows=EXPECTED_PROXY_ROWS,
        row_key=("item_id", "candidate"),
    )


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
            ("task_version", LB.TASK_VERSION),
            ("endpoint_version", ENDPOINT_VERSION), ("model", MODEL),
            ("model_id", MODEL_ID), ("model_revision", LB.MODEL_REVISION),
            ("tokenizer_revision", LB.MODEL_REVISION), ("ctx", CTX),
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
        "fp_point_ceiling_gate": fp_accuracy <= 0.75,
        "fixed_point_ceiling_gate": fixed_accuracy <= 0.75,
        "fp_bootstrap_chance_gate": fp_q05 > 0.25,
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
        summary["decision"] = "stop_invalid_operating_point"
        return summary
    if not summary["opportunity_gate"]:
        summary["decision"] = "stop_no_opportunity"
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
        "advance_confirmation"
        if summary["proxy_transfer_gate"]
        else "reject_scaffold_proxy"
    )
    return summary


def load_pair(
    prediction_path: str | Path, proxy_path: str | Path, *,
    manifest_path: str | Path, dataset_path: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    prediction_path = Path(prediction_path).resolve()
    proxy_path = Path(proxy_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    dataset_path = Path(dataset_path).resolve()
    for path, label in (
        (prediction_path, "prediction parquet"),
        (proxy_path, "proxy parquet"),
        (manifest_path, "manifest"),
        (dataset_path, "dataset"),
    ):
        if not path.is_file():
            _fail(f"missing {label}: {path}")
    if prediction_path.suffix != ".parquet" or proxy_path.suffix != ".parquet":
        _fail("both result artifacts must use the .parquet suffix")
    prediction_sidecar_path = prediction_path.with_suffix(".json")
    proxy_sidecar_path = proxy_path.with_suffix(".json")
    if not prediction_sidecar_path.is_file() or not proxy_sidecar_path.is_file():
        _fail("both parquets require strict adjacent JSON sidecars")

    manifest, manifest_file_hash, development = authenticate_manifest(manifest_path)
    data_by_id = authenticate_dataset(dataset_path, development)
    try:
        predictions = pd.read_parquet(prediction_path)
        proxy = pd.read_parquet(proxy_path)
    except Exception as exc:
        _fail(f"cannot read parquet artifacts: {exc}")
    prediction_sidecar = _load_json(
        prediction_sidecar_path, "prediction sidecar"
    )
    proxy_sidecar = _load_json(proxy_sidecar_path, "proxy sidecar")
    validate_sidecars(
        prediction_sidecar, proxy_sidecar,
        prediction_path=prediction_path, proxy_path=proxy_path,
        manifest_path=manifest_path, manifest=manifest,
        manifest_file_hash=manifest_file_hash,
    )
    validate_frames(
        predictions, proxy, manifest=manifest,
        manifest_file_hash=manifest_file_hash, development=development,
        data_by_id=data_by_id,
    )
    summary = analyze_frames(
        predictions, proxy, development=development, data_by_id=data_by_id
    )
    return summary, predictions, proxy


def _lock_content_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("content_sha256", None)
    payload = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_confirmation_lock(
    summary: Mapping[str, Any], *, prediction_path: str | Path,
    proxy_path: str | Path, manifest_path: str | Path,
    runner_path: str | Path,
) -> dict[str, Any] | None:
    if summary.get("decision") != "advance_confirmation":
        return None
    prediction_path = Path(prediction_path).resolve()
    proxy_path = Path(proxy_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    runner_path = Path(runner_path).resolve()
    reader_path = Path(__file__).resolve()
    required_files = (
        prediction_path, prediction_path.with_suffix(".json"),
        proxy_path, proxy_path.with_suffix(".json"), manifest_path,
        runner_path, reader_path,
    )
    for path in required_files:
        if not path.is_file():
            _fail(f"confirmation lock source is missing: {path}")
    lock: dict[str, Any] = {
        "lock_version": "longbench_v2_v5_confirmation_lock_v1",
        "protocol_version": PROTOCOL_VERSION,
        "runner_version": RUNNER_VERSION,
        "endpoint_version": ENDPOINT_VERSION,
        "proxy_rule_version": PROXY_RULE_VERSION,
        "decision": "advance_confirmation",
        "development_split": SPLIT,
        "development_ids_sha256": SPLIT_ID_SHA256,
        "development_id_token_sha256": SPLIT_ID_TOKEN_SHA256,
        "development_half_id_sha256": list(HALF_ID_SHA256),
        "development_half_component_counts": list(HALF_COMPONENT_COUNTS),
        "development_half_row_counts": list(HALF_ROW_COUNTS),
        "prediction_parquet": str(prediction_path),
        "prediction_parquet_sha256": _sha256_file(prediction_path),
        "prediction_sidecar": str(prediction_path.with_suffix(".json")),
        "prediction_sidecar_sha256": _sha256_file(
            prediction_path.with_suffix(".json")
        ),
        "proxy_parquet": str(proxy_path),
        "proxy_parquet_sha256": _sha256_file(proxy_path),
        "proxy_sidecar": str(proxy_path.with_suffix(".json")),
        "proxy_sidecar_sha256": _sha256_file(proxy_path.with_suffix(".json")),
        "manifest": str(manifest_path),
        "manifest_content_sha256": MANIFEST_CONTENT_SHA256,
        "manifest_file_sha256": _sha256_file(manifest_path),
        "runner": str(runner_path),
        "runner_sha256": _sha256_file(runner_path),
        "reader": str(reader_path),
        "reader_sha256": _sha256_file(reader_path),
        "dataset_repo": LB.DATASET_REPO,
        "dataset_revision": LB.DATASET_REVISION,
        "dataset_sha256": LB.DATASET_SHA256,
        "model_id": MODEL_ID,
        "model_revision": LB.MODEL_REVISION,
        "tokenizer_revision": LB.MODEL_REVISION,
        "task_version": LB.TASK_VERSION,
        "official_prompt_sha256": LB.OFFICIAL_PROMPT_SHA256,
        "ctx": CTX,
        "window": WINDOW,
        "budget": BUDGET,
        "arms": list(ARMS),
        "raw_arms": list(RAW_ARMS),
        "candidates": list(CANDIDATES),
        "scaffold_token_ids": list(SCAFFOLD_TOKEN_IDS),
        "scaffold_token_hash": SCAFFOLD_TOKEN_HASH,
        "choice_branch_ids": CHOICE_BRANCH_IDS,
        "choice_branch_ids_hash": CHOICE_BRANCH_IDS_HASH,
        "proxy_positions": [0, 1, 2, 3, 4],
        "choice_position": 5,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_rng": "numpy.Generator(PCG64)",
        "bootstrap_quantile": "0.05_linear",
        "fixed_policy": summary["fixed_policy"],
        "thresholds": {
            "point_accuracy_ceiling": 0.75,
            "competence_q05_strictly_above": 0.25,
            "H_min": 0.10,
            "H_q05_strictly_above": 0.0,
            "H_half_min": 0.05,
            "G_min": 0.05,
            "G_over_H_min": 0.50,
            "G_q05_min": 0.0,
            "G_half_min": 0.0,
        },
        "development_analysis": {
            str(key): _python_scalar(value) for key, value in summary.items()
        },
    }
    lock["content_sha256"] = _lock_content_hash(lock)
    return lock


def write_confirmation_lock(path: str | Path, lock: Mapping[str, Any] | None) -> None:
    destination = Path(path)
    if destination.is_symlink():
        _fail(f"refusing confirmation-lock symlink: {destination}")
    if lock is None:
        if destination.exists():
            _fail(
                f"refusing stale confirmation lock after non-advance: {destination}"
            )
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
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    if temporary.exists() or temporary.is_symlink():
        _fail(f"temporary confirmation-lock path already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fmt(value: Any) -> str:
    if value is None:
        return "suppressed"
    number = float(value)
    return "nan" if not math.isfinite(number) else f"{number:.3f}"


def print_summary(summary: Mapping[str, Any]) -> None:
    print("V5 LongBench-v2 forced-choice development (frozen section 3J)")
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
    parser.add_argument("predictions", help="development forced-choice parquet")
    parser.add_argument("proxy", help="development scaffold-proxy parquet")
    parser.add_argument("--manifest", required=True, help="frozen gold-free manifest")
    parser.add_argument("--dataset", required=True, help="pinned LongBench-v2 data JSON")
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
        default=str(ROOT / "h0_measurement/run_longbench_v2_forced_choice.py"),
        help="runner source to authenticate in --lock (default: V5 runner)",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="authenticate silently; do not print the report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary, _, _ = load_pair(
            args.predictions, args.proxy, manifest_path=args.manifest,
            dataset_path=args.dataset,
        )
        if args.csv:
            destination = Path(args.csv)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([summary]).to_csv(destination, index=False)
        if args.lock:
            lock = build_confirmation_lock(
                summary, prediction_path=args.predictions,
                proxy_path=args.proxy, manifest_path=args.manifest,
                runner_path=args.runner,
            )
            write_confirmation_lock(args.lock, lock)
        if not args.validate_only:
            print_summary(summary)
        return 0
    except LongBenchForcedChoiceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
