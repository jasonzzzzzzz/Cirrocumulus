#!/usr/bin/env python3
"""Strict reader for the frozen V4 LongBench-v2 qualification.

The reader accepts exactly the qualification split, Llama-3.1-8B at 32K,
FP plus uniform B=2 accuracy, and one uniform choice-logit diagnostic per
manifest item.  It authenticates the pinned dataset and gold-free manifest,
recomputes official scores from the saved responses, and applies only the
qualification gates frozen in plan.md section 3I.5.

A well-formed but ineligible qualification exits zero and prints
``decision  stop_v4``.  Artifact/provenance drift exits two.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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


SPLIT = "qualification"
MODEL = "llama31-8b"
MODEL_ID = LB.MODEL_ID
CTX = LB.CONTEXT_WINDOW
MAX_NEW = LB.RESERVED_OUTPUT_TOKENS
MAX_INPUT = LB.MAX_INPUT_TOKENS
WINDOW = 32
BUDGET = 2.0
MAXB = 8
ARMS = ("fp", "uniform")
CANDIDATES = ("uniform",)
EXPECTED_ROWS = 20
EXPECTED_ACCURACY_ROWS = EXPECTED_ROWS * len(ARMS)
EXPECTED_CHOICE_ROWS = EXPECTED_ROWS * len(CANDIDATES)
FP_ALLOCATION_ID = hashlib.sha256(b"fp16").hexdigest()
ALLOCATION_ID_ALGORITHM = "sha256_layer_uint8_v1"
RUNNER_VERSION = "longbench_v2_runner_v1"
MANIFEST_CONTENT_SHA256 = "ee951a2c7b2c256125e73de4fba6f7e14485a566bbf6096c14c6e4f23e6407ff"
MANIFEST_FILE_SHA256 = "bd7b358a91081fd5657a2aa822a4c814ebfeb85f410d55cee25aa9f54223798e"
DECODE = "greedy_eos_or_cap"
COMPRESS_AT = "official_chat_prompt_end"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
CHOICE_PREFIX_IDS = (791, 4495, 4320, 374, 320)
CHOICE_BRANCH_IDS = {"A": 32, "B": 33, "C": 34, "D": 35}
CHOICE_PREFIX_HASH = PD.token_hash(CHOICE_PREFIX_IDS)
CHOICE_BRANCH_HASH = PD.token_hash([CHOICE_BRANCH_IDS[choice] for choice in "ABCD"])

ACCURACY_COLUMNS = (
    "item_id", "group_id", "split", "domain", "sub_domain", "difficulty",
    "length", "context_hash", "question_hash", "prompt_token_hash",
    "input_tokens", "n_prompt_tokens", "truncated", "dataset_sha256",
    "manifest_content_sha256", "manifest_file_sha256", "task_version",
    "parser_version", "model", "model_id", "model_revision",
    "tokenizer_revision", "ctx", "max_new_tokens", "window", "arm", "B",
    "gold_answer", "response", "parsed_answer", "valid", "score", "gen_len",
    "reached_max_new", "bits_per_token", "evict_frac", "allocation_id",
    "ctx_len", "observed_queries", "maxb", "rot_seed", "norm_correct",
)

CHOICE_COLUMNS = (
    "item_id", "group_id", "split", "domain", "sub_domain", "difficulty",
    "length", "context_hash", "question_hash", "prompt_token_hash",
    "input_tokens", "n_prompt_tokens", "truncated", "dataset_sha256",
    "manifest_content_sha256", "manifest_file_sha256", "task_version",
    "parser_version", "model", "model_id", "model_revision",
    "tokenizer_revision", "ctx", "max_new_tokens", "window", "B",
    "candidate", "candidate_order", "choice_kl", "choice_top1_agreement",
    "fp_choice_index", "candidate_choice_index", "fp_choice_entropy",
    "candidate_choice_entropy", "choice_rule_version", "choice_logit_version",
    "choice_prefix_token_hash", "choice_branch_ids_hash", "allocation_id",
    "selected_policy", "fp_canonical_choice", "candidate_canonical_choice",
    "fp_greedy_parsed_answer", "fp_greedy_valid",
    "fp_canonical_agrees_with_greedy",
)

COMMON_COLUMNS = (
    "item_id", "group_id", "split", "domain", "sub_domain", "difficulty",
    "length", "context_hash", "question_hash", "prompt_token_hash",
    "input_tokens", "n_prompt_tokens", "truncated", "dataset_sha256",
    "manifest_content_sha256", "manifest_file_sha256", "task_version",
    "parser_version", "model", "model_id", "model_revision",
    "tokenizer_revision", "ctx", "max_new_tokens", "window",
)

# Names which would serialize the transient branch logits/probability vectors.
# Scalar summaries such as choice_kl and fp_choice_entropy are intentional.
RAW_VECTOR_COLUMNS = {
    "logits", "raw_logits", "fp_logits", "candidate_logits", "choice_logits",
    "probabilities", "probs", "fp_probs", "candidate_probs", "choice_probs",
    "fp_choice_logits", "candidate_choice_logits", "fp_choice_probs",
    "candidate_choice_probs",
}


class LongBenchQualificationError(ValueError):
    """A file violates the frozen qualification contract."""


def _fail(message: str) -> None:
    raise LongBenchQualificationError(message)


def _same(expected: Any, actual: Any, description: str) -> None:
    if isinstance(actual, (np.integer, np.floating, np.bool_)):
        actual = actual.item()
    if expected != actual:
        _fail(f"{description}: expected {expected!r}, got {actual!r}")


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        _fail(f"{label} is missing required columns: {', '.join(missing)}")
    forbidden = sorted(set(frame.columns) & RAW_VECTOR_COLUMNS)
    if forbidden:
        _fail(f"{label} contains raw-logit/probability columns: {forbidden}")


def _numeric(frame: pd.DataFrame, column: str, label: str, *, integer: bool = False
             ) -> pd.Series:
    try:
        values = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as exc:
        _fail(f"{label}.{column} is not numeric: {exc}")
    raw = values.to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        _fail(f"{label}.{column} contains a non-finite value")
    if integer and not np.equal(raw, np.floor(raw)).all():
        _fail(f"{label}.{column} must contain integers")
    return values.astype(np.int64) if integer else values.astype(float)


def _bool(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    values = frame[column]
    allowed = values.map(lambda value: isinstance(value, (bool, np.bool_)))
    if not bool(allowed.all()):
        _fail(f"{label}.{column} must contain booleans")
    return values.astype(bool)


def _only(frame: pd.DataFrame, column: str, label: str) -> Any:
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        _fail(f"{label} mixes {column}: {values!r}")
    value = values[0]
    return value.item() if isinstance(value, (np.integer, np.floating, np.bool_)) else value


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


def _protocol(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
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
        (LB.OFFICIAL_CODE_REVISION, official.get("code_revision"), "manifest official code revision"),
        (LB.OFFICIAL_PROMPT_SHA256, official.get("prompt_sha256"), "manifest prompt SHA-256"),
        (LB.TASK_VERSION, official.get("task_version"), "manifest task version"),
        (LB.PARSER_VERSION, official.get("parser_version"), "manifest parser version"),
        (LB.MODEL_ID, model.get("id"), "manifest model id"),
        (LB.MODEL_REVISION, model.get("revision"), "manifest model revision"),
        (LB.TRANSFORMERS_VERSION, model.get("transformers_version"), "manifest transformers version"),
        (LB.TOKENIZERS_VERSION, model.get("tokenizers_version"), "manifest tokenizers version"),
        (LB.CHAT_TEMPLATE_SHA256, model.get("chat_template_sha256"), "manifest chat-template SHA-256"),
        (CTX, eligibility.get("context_window"), "manifest context window"),
        (MAX_NEW, eligibility.get("reserved_output_tokens"), "manifest reserved output"),
        (MAX_INPUT, eligibility.get("max_rendered_input_tokens"), "manifest max input"),
        (LB.CONTEXT_HASH_VERSION, grouping.get("context_hash_version"), "manifest context-hash version"),
        (LB.QUESTION_HASH_VERSION, grouping.get("question_hash_version"), "manifest question-hash version"),
        (LB.SPLIT_VERSION, grouping.get("split_version"), "manifest split version"),
    )
    for expected, actual, description in checks:
        _same(expected, actual, description)
    return protocol


def authenticate_manifest(path: str | Path) -> tuple[dict[str, Any], str,
                                                       list[dict[str, Any]]]:
    manifest_path = Path(path).resolve()
    manifest = _load_json(manifest_path, "manifest")
    try:
        AUDIT.verify_manifest_content_hash(manifest)
    except (KeyError, TypeError, ValueError) as exc:
        _fail(f"manifest content authentication failed: {exc}")
    _same(LB.MANIFEST_VERSION, manifest.get("manifest_version"), "manifest version")
    _same(MANIFEST_CONTENT_SHA256, manifest.get("content_sha256"),
          "frozen manifest content SHA-256")
    _same(MANIFEST_FILE_SHA256, _sha256_file(manifest_path),
          "frozen manifest file SHA-256")
    _protocol(manifest)
    serialized = json.dumps(manifest, sort_keys=True)
    if '"answer"' in serialized or '"gold_answer"' in serialized:
        _fail("manifest must be gold-free")
    examples = manifest.get("examples")
    if not isinstance(examples, list) or len(examples) != LB.EXPECTED_ELIGIBLE_ROWS:
        _fail(f"manifest must contain exactly {LB.EXPECTED_ELIGIBLE_ROWS} eligible rows")
    if manifest.get("eligible_list_sha256") != LB.EXPECTED_ELIGIBLE_LIST_SHA256:
        _fail("manifest eligible-list SHA-256 drifted")
    if AUDIT.eligible_list_sha256(examples) != LB.EXPECTED_ELIGIBLE_LIST_SHA256:
        _fail("manifest eligible rows do not reproduce the frozen list SHA-256")

    exact_keys = {"id", "group_id", "context_hash", "input_tokens", "metadata", "split"}
    seen: set[str] = set()
    groups: dict[str, str] = {}
    for index, row in enumerate(examples):
        if not isinstance(row, dict) or set(row) != exact_keys:
            _fail(f"manifest example {index} does not have the exact gold-free schema")
        item_id = row["id"]
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            _fail(f"manifest contains empty/duplicate item id {item_id!r}")
        seen.add(item_id)
        if row["split"] not in AUDIT.SPLIT_NAMES:
            _fail(f"manifest item {item_id} has invalid split {row['split']!r}")
        group_id = row["group_id"]
        if not isinstance(group_id, str) or not SHA256_RE.fullmatch(group_id):
            _fail(f"manifest item {item_id} has invalid group id")
        if group_id in groups and groups[group_id] != row["split"]:
            _fail(f"manifest group {group_id} crosses splits")
        groups[group_id] = row["split"]
        if not isinstance(row["context_hash"], str) or not SHA256_RE.fullmatch(row["context_hash"]):
            _fail(f"manifest item {item_id} has invalid context hash")
        tokens = row["input_tokens"]
        if isinstance(tokens, bool) or not isinstance(tokens, int) or not 0 < tokens <= MAX_INPUT:
            _fail(f"manifest item {item_id} has invalid input_tokens {tokens!r}")

    qualification = [row for row in examples if row["split"] == SPLIT]
    if len(qualification) != EXPECTED_ROWS:
        _fail(f"manifest qualification has {len(qualification)} rows, expected {EXPECTED_ROWS}")
    expected = LB.EXPECTED_SPLITS[SPLIT]
    split_record = manifest.get("split_counts", {}).get(SPLIT)
    if not isinstance(split_record, dict):
        _fail("manifest qualification split summary is absent")
    _same(expected["rows"], split_record.get("rows"), "qualification manifest row count")
    _same(expected["groups"], split_record.get("groups"), "qualification manifest group count")
    _same(expected["id_sha256"], split_record.get("id_sha256"),
          "qualification ID SHA-256")
    _same(expected["id_token_sha256"], split_record.get("id_token_sha256"),
          "qualification id/token SHA-256")
    got_split_sha = hashlib.sha256(
        "".join(f"{row['id']}\t{int(row['input_tokens'])}\n"
                for row in sorted(qualification, key=lambda value: value["id"]))
        .encode("utf-8")
    ).hexdigest()
    _same(expected["id_token_sha256"], got_split_sha,
          "recomputed qualification id/token SHA-256")
    return manifest, _sha256_file(manifest_path), qualification


def authenticate_dataset(path: str | Path, manifest_rows: Sequence[Mapping[str, Any]]
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
        expected_metadata = {key: item[key] for key in
                             ("domain", "sub_domain", "difficulty", "length")}
        if row["metadata"] != expected_metadata:
            _fail(f"manifest metadata disagrees with dataset for {item_id}")
        if LB.context_hash(item) != row["context_hash"]:
            _fail(f"manifest context hash disagrees with dataset for {item_id}")
    return by_id


def _validate_sidecar_base(sidecar: Mapping[str, Any], *, artifact: str,
                           parquet: Path, manifest_path: Path,
                           manifest: Mapping[str, Any], manifest_file_hash: str,
                           ids: Sequence[str]) -> None:
    required = {
        "runner_version", "dataset_repo", "dataset_revision", "dataset_sha256",
        "dataset_bytes", "official_code_repo", "official_code_revision",
        "official_prompt_sha256", "task_version", "parser_version",
        "context_hash_version", "choice_logit_version", "manifest",
        "manifest_version", "manifest_content_sha256", "manifest_file_sha256",
        "eligible_list_sha256", "split", "split_count", "split_ids_sha256",
        "model", "model_id", "model_revision", "tokenizer_revision", "ctx",
        "max_input_tokens", "max_new_tokens", "decode", "temperature", "window",
        "budget", "cascade_bits", "allocator_budget_rule", "compress_at",
        "raw_arms", "arms", "allocation_id_algorithm", "chat_template_sha256",
        "dtype", "bit_list", "maxb", "rot_seed", "norm_correct", "chunk",
        "attn_impl",
        "transformers_version", "tokenizers_version", "no_truncation",
        "no_raw_logits", "artifact", "parquet", "rows", "expected_rows",
        "row_key",
    }
    missing = sorted(required - set(sidecar))
    if missing:
        _fail(f"{artifact} sidecar is missing: {', '.join(missing)}")
    model_protocol = manifest["protocol"]["model"]
    checks = (
        (RUNNER_VERSION, sidecar["runner_version"], "runner version"),
        (LB.DATASET_REPO, sidecar["dataset_repo"], "dataset repo"),
        (LB.DATASET_REVISION, sidecar["dataset_revision"], "dataset revision"),
        (LB.DATASET_SHA256, sidecar["dataset_sha256"], "dataset SHA-256"),
        (LB.DATASET_BYTES, sidecar["dataset_bytes"], "dataset bytes"),
        (LB.OFFICIAL_CODE_REPO, sidecar["official_code_repo"], "official code repo"),
        (LB.OFFICIAL_CODE_REVISION, sidecar["official_code_revision"], "official code revision"),
        (LB.OFFICIAL_PROMPT_SHA256, sidecar["official_prompt_sha256"], "prompt SHA-256"),
        (LB.TASK_VERSION, sidecar["task_version"], "task version"),
        (LB.PARSER_VERSION, sidecar["parser_version"], "parser version"),
        (LB.CONTEXT_HASH_VERSION, sidecar["context_hash_version"], "context-hash version"),
        (LB.CHOICE_LOGIT_VERSION, sidecar["choice_logit_version"], "choice-logit version"),
        (LB.MANIFEST_VERSION, sidecar["manifest_version"], "manifest version"),
        (manifest["content_sha256"], sidecar["manifest_content_sha256"], "manifest content hash"),
        (manifest_file_hash, sidecar["manifest_file_sha256"], "manifest file hash"),
        (LB.EXPECTED_ELIGIBLE_LIST_SHA256, sidecar["eligible_list_sha256"], "eligible-list hash"),
        (SPLIT, sidecar["split"], "split"),
        (EXPECTED_ROWS, sidecar["split_count"], "split count"),
        (MODEL, sidecar["model"], "model tag"),
        (MODEL_ID, sidecar["model_id"], "model id"),
        (LB.MODEL_REVISION, sidecar["model_revision"], "model revision"),
        (LB.MODEL_REVISION, sidecar["tokenizer_revision"], "tokenizer revision"),
        (CTX, sidecar["ctx"], "context length"),
        (MAX_INPUT, sidecar["max_input_tokens"], "maximum input tokens"),
        (MAX_NEW, sidecar["max_new_tokens"], "generation cap"),
        (DECODE, sidecar["decode"], "decode"),
        (0.0, sidecar["temperature"], "temperature"),
        (WINDOW, sidecar["window"], "window"),
        (BUDGET, sidecar["budget"], "budget"),
        (4, sidecar["cascade_bits"], "cascade bits"),
        ("feasible", sidecar["allocator_budget_rule"], "allocator rule"),
        (COMPRESS_AT, sidecar["compress_at"], "compression point"),
        (list(ARMS), sidecar["raw_arms"], "raw arm order"),
        (list(ARMS), sidecar["arms"], "resolved arm order"),
        (None, sidecar.get("baselines"), "baseline override record"),
        (ALLOCATION_ID_ALGORITHM, sidecar["allocation_id_algorithm"], "allocation id algorithm"),
        ("bfloat16", sidecar["dtype"], "model dtype"),
        ([1, 2, 3, 4, 5, 6, 8], sidecar["bit_list"], "bit-width list"),
        (MAXB, sidecar["maxb"], "maximum bit width"),
        (0, sidecar["rot_seed"], "rotation seed"),
        (True, sidecar["norm_correct"], "norm correction"),
        (4096, sidecar["chunk"], "prefill chunk"),
        ("sieve_compress", sidecar["attn_impl"], "attention implementation"),
        (LB.CHAT_TEMPLATE_SHA256, sidecar["chat_template_sha256"], "chat-template hash"),
        (LB.TRANSFORMERS_VERSION, sidecar["transformers_version"], "transformers version"),
        (LB.TOKENIZERS_VERSION, sidecar["tokenizers_version"], "tokenizers version"),
        (True, sidecar["no_truncation"], "no-truncation flag"),
        (True, sidecar["no_raw_logits"], "no-raw-logits flag"),
        (artifact, sidecar["artifact"], "artifact kind"),
        (parquet.name, sidecar["parquet"], "parquet filename"),
    )
    for expected, actual, description in checks:
        _same(expected, actual, f"{artifact} sidecar {description}")
    _same(model_protocol["revision"], sidecar["model_revision"],
          f"{artifact} sidecar model/manifest revision")
    if Path(str(sidecar["manifest"])).resolve() != manifest_path.resolve():
        _fail(f"{artifact} sidecar points at a different manifest")
    split_ids_hash = hashlib.sha256(
        "".join(f"{item_id}\n" for item_id in sorted(ids)).encode("utf-8")
    ).hexdigest()
    _same(split_ids_hash, sidecar["split_ids_sha256"],
          f"{artifact} sidecar split ID SHA-256")


def _normalize_parsed(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def validate_frames(accuracy: pd.DataFrame, choice: pd.DataFrame, *,
                    manifest: Mapping[str, Any], manifest_file_hash: str,
                    qualification: Sequence[Mapping[str, Any]],
                    data_by_id: Mapping[str, Mapping[str, str]]) -> None:
    _require_columns(accuracy, ACCURACY_COLUMNS, "accuracy parquet")
    _require_columns(choice, CHOICE_COLUMNS, "choice parquet")
    if len(accuracy) != EXPECTED_ACCURACY_ROWS:
        _fail(f"accuracy parquet has {len(accuracy)} rows, expected {EXPECTED_ACCURACY_ROWS}")
    if len(choice) != EXPECTED_CHOICE_ROWS:
        _fail(f"choice parquet has {len(choice)} rows, expected {EXPECTED_CHOICE_ROWS}")
    for label, frame in (("accuracy parquet", accuracy), ("choice parquet", choice)):
        for column in frame.columns:
            if frame[column].dtype == object:
                for value in frame[column]:
                    if isinstance(value, (list, tuple, dict, np.ndarray)):
                        _fail(f"{label}.{column} contains a nonscalar value")

    manifest_by_id = {str(row["id"]): row for row in qualification}
    ordered_ids = [str(row["id"]) for row in qualification]
    expected_accuracy = {(item_id, arm) for item_id in ordered_ids for arm in ARMS}
    accuracy_keys = list(zip(accuracy.item_id.astype(str), accuracy.arm.astype(str)))
    if len(set(accuracy_keys)) != len(accuracy_keys):
        _fail("accuracy parquet has duplicate item_id/arm keys")
    if set(accuracy_keys) != expected_accuracy:
        _fail("accuracy parquet does not contain exactly 20 manifest IDs x FP/uniform")
    choice_keys = list(zip(choice.item_id.astype(str), choice.candidate.astype(str)))
    expected_choice = {(item_id, "uniform") for item_id in ordered_ids}
    if len(set(choice_keys)) != len(choice_keys):
        _fail("choice parquet has duplicate item_id/candidate keys")
    if set(choice_keys) != expected_choice:
        _fail("choice parquet does not contain exactly 20 manifest IDs x uniform")

    numeric_int = ("input_tokens", "n_prompt_tokens", "ctx", "max_new_tokens",
                   "window")
    for label, frame in (("accuracy", accuracy), ("choice", choice)):
        for column in numeric_int:
            frame[column] = _numeric(frame, column, label, integer=True)
        frame["truncated"] = _bool(frame, "truncated", label)
        if bool(frame.truncated.any()):
            _fail(f"{label} contains a truncated row")
        for column, expected in (
            ("split", SPLIT), ("dataset_sha256", LB.DATASET_SHA256),
            ("manifest_content_sha256", manifest["content_sha256"]),
            ("manifest_file_sha256", manifest_file_hash),
            ("task_version", LB.TASK_VERSION), ("parser_version", LB.PARSER_VERSION),
            ("model", MODEL), ("model_id", MODEL_ID),
            ("model_revision", LB.MODEL_REVISION),
            ("tokenizer_revision", LB.MODEL_REVISION), ("ctx", CTX),
            ("max_new_tokens", MAX_NEW), ("window", WINDOW),
        ):
            values = frame[column].drop_duplicates().tolist()
            if values != [expected]:
                _fail(f"{label}.{column} must be exactly {expected!r}, got {values!r}")
        if not frame.prompt_token_hash.astype(str).map(
                lambda value: bool(SHA256_RE.fullmatch(value))).all():
            _fail(f"{label}.prompt_token_hash must be lowercase SHA-256")

    # The two arms and choice replay must use identical immutable input records.
    for item_id in ordered_ids:
        a = accuracy.loc[accuracy.item_id.astype(str) == item_id]
        c = choice.loc[choice.item_id.astype(str) == item_id]
        entry = manifest_by_id[item_id]
        item = data_by_id[item_id]
        expected_fields = {
            "group_id": entry["group_id"], "split": SPLIT,
            "domain": item["domain"], "sub_domain": item["sub_domain"],
            "difficulty": item["difficulty"], "length": item["length"],
            "context_hash": entry["context_hash"],
            "question_hash": hashlib.sha256(item["question"].strip().encode("utf-8")).hexdigest(),
            "input_tokens": int(entry["input_tokens"]),
            "n_prompt_tokens": int(entry["input_tokens"]),
        }
        for column, expected in expected_fields.items():
            for label, block in (("accuracy", a), ("choice", c)):
                values = block[column].drop_duplicates().tolist()
                if values != [expected]:
                    _fail(f"{label} {item_id}.{column} disagrees with manifest/dataset")
        for column in COMMON_COLUMNS:
            if a[column].iloc[0] != c[column].iloc[0]:
                _fail(f"accuracy/choice provenance differs for {item_id}.{column}")
        if a.prompt_token_hash.nunique(dropna=False) != 1:
            _fail(f"accuracy arms use different prompt token hashes for {item_id}")

    accuracy["B"] = _numeric(accuracy, "B", "accuracy")
    accuracy["score"] = _numeric(accuracy, "score", "accuracy")
    accuracy["gen_len"] = _numeric(accuracy, "gen_len", "accuracy", integer=True)
    accuracy["bits_per_token"] = _numeric(accuracy, "bits_per_token", "accuracy")
    accuracy["evict_frac"] = _numeric(accuracy, "evict_frac", "accuracy")
    accuracy["ctx_len"] = _numeric(accuracy, "ctx_len", "accuracy", integer=True)
    accuracy["observed_queries"] = _numeric(
        accuracy, "observed_queries", "accuracy", integer=True)
    accuracy["maxb"] = _numeric(accuracy, "maxb", "accuracy", integer=True)
    accuracy["rot_seed"] = _numeric(accuracy, "rot_seed", "accuracy", integer=True)
    accuracy["valid"] = _bool(accuracy, "valid", "accuracy")
    accuracy["reached_max_new"] = _bool(
        accuracy, "reached_max_new", "accuracy")
    accuracy["norm_correct"] = _bool(accuracy, "norm_correct", "accuracy")
    if not set(accuracy.score.unique()).issubset({0.0, 1.0}):
        _fail("accuracy scores must be binary")
    if (accuracy.gen_len < 0).any() or (accuracy.gen_len > MAX_NEW).any():
        _fail("accuracy generation length lies outside [0,128]")
    if not np.array_equal(accuracy.reached_max_new.to_numpy(),
                          accuracy.gen_len.eq(MAX_NEW).to_numpy()):
        _fail("reached_max_new disagrees with generation length")
    if not accuracy.observed_queries.eq(WINDOW).all():
        _fail("accuracy observed_queries must equal the protected window")
    expected_ctx = accuracy.input_tokens - 1 - WINDOW
    if not accuracy.ctx_len.eq(expected_ctx).all():
        _fail("accuracy ctx_len is inconsistent with input_tokens/window boundary")
    if not accuracy.maxb.eq(MAXB).all() or not accuracy.rot_seed.eq(0).all():
        _fail("accuracy maxb/rotation seed drifted")
    if not accuracy.norm_correct.all():
        _fail("accuracy norm correction must be enabled")
    if not accuracy.allocation_id.astype(str).map(
            lambda value: bool(SHA256_RE.fullmatch(value))).all():
        _fail("accuracy allocation_id must be lowercase SHA-256")

    fp = accuracy.loc[accuracy.arm.astype(str) == "fp"]
    uniform = accuracy.loc[accuracy.arm.astype(str) == "uniform"]
    if not fp.B.eq(0.0).all() or not fp.bits_per_token.eq(16.0).all():
        _fail("FP rows must use B=0 and 16 bits/token")
    if not fp.evict_frac.eq(0.0).all() or not fp.allocation_id.eq(FP_ALLOCATION_ID).all():
        _fail("FP allocation provenance is not the frozen fp16 sentinel")
    if not uniform.B.eq(BUDGET).all():
        _fail("uniform rows must use B=2")
    if not np.allclose(uniform.bits_per_token, BUDGET, rtol=0.0, atol=1e-7):
        _fail("uniform rows do not spend exactly 2 bits/token")
    if not uniform.evict_frac.eq(0.0).all():
        _fail("uniform rows unexpectedly evict tokens")
    if uniform.allocation_id.eq(FP_ALLOCATION_ID).any():
        _fail("uniform allocation cannot equal the FP sentinel")

    for row in accuracy.itertuples(index=False):
        item = data_by_id[str(row.item_id)]
        if str(row.gold_answer) != item["answer"]:
            _fail(f"saved gold answer disagrees with dataset for {row.item_id}")
        if not isinstance(row.response, str):
            _fail(f"saved response is not text for {row.item_id}/{row.arm}")
        rescored = LB.score_response(row.response, item["answer"])
        if _normalize_parsed(row.parsed_answer) != rescored["parsed_answer"]:
            _fail(f"official parser output disagrees for {row.item_id}/{row.arm}")
        if bool(row.valid) != bool(rescored["valid"]):
            _fail(f"valid flag disagrees with official parser for {row.item_id}/{row.arm}")
        if float(row.score) != float(rescored["score"]):
            _fail(f"score disagrees with pinned gold/parser for {row.item_id}/{row.arm}")

    choice["B"] = _numeric(choice, "B", "choice")
    choice["candidate_order"] = _numeric(
        choice, "candidate_order", "choice", integer=True)
    for column in ("choice_kl", "fp_choice_entropy", "candidate_choice_entropy"):
        choice[column] = _numeric(choice, column, "choice")
    for column in ("fp_choice_index", "candidate_choice_index"):
        choice[column] = _numeric(choice, column, "choice", integer=True)
    choice["choice_top1_agreement"] = _numeric(
        choice, "choice_top1_agreement", "choice")
    choice["fp_greedy_valid"] = _bool(choice, "fp_greedy_valid", "choice")
    choice["fp_canonical_agrees_with_greedy"] = _bool(
        choice, "fp_canonical_agrees_with_greedy", "choice")
    if not choice.B.eq(BUDGET).all() or not choice.candidate.eq("uniform").all():
        _fail("choice rows must contain only uniform at B=2")
    if not choice.candidate_order.eq(0).all() or not choice.selected_policy.eq("uniform").all():
        _fail("choice candidate order/selection must be the singleton uniform menu")
    if (choice.choice_kl < -1e-6).any():
        _fail("choice KL must be nonnegative")
    if ((choice.fp_choice_entropy < -1e-6) |
            (choice.fp_choice_entropy > math.log(4) + 1e-6) |
            (choice.candidate_choice_entropy < -1e-6) |
            (choice.candidate_choice_entropy > math.log(4) + 1e-6)).any():
        _fail("choice entropy lies outside [0,log(4)]")
    if (~choice.fp_choice_index.between(0, 3) |
            ~choice.candidate_choice_index.between(0, 3)).any():
        _fail("choice indices must lie in 0..3")
    expected_agree = choice.fp_choice_index.eq(choice.candidate_choice_index).astype(float)
    if not np.array_equal(choice.choice_top1_agreement.to_numpy(),
                          expected_agree.to_numpy()):
        _fail("choice_top1_agreement disagrees with saved choice indices")
    fp_letters = choice.fp_choice_index.astype(int).map(lambda value: "ABCD"[value])
    candidate_letters = choice.candidate_choice_index.astype(int).map(
        lambda value: "ABCD"[value])
    if not choice.fp_canonical_choice.astype(str).equals(fp_letters.astype(str)):
        _fail("fp_canonical_choice disagrees with fp_choice_index")
    if not choice.candidate_canonical_choice.astype(str).equals(
            candidate_letters.astype(str)):
        _fail("candidate_canonical_choice disagrees with candidate_choice_index")
    if not choice.choice_rule_version.eq(PD.CHOICE_RULE_VERSION).all():
        _fail("choice rule version drifted")
    if not choice.choice_logit_version.eq(LB.CHOICE_LOGIT_VERSION).all():
        _fail("choice-logit version drifted")
    for column in ("choice_prefix_token_hash", "choice_branch_ids_hash", "allocation_id"):
        if not choice[column].astype(str).map(
                lambda value: bool(SHA256_RE.fullmatch(value))).all():
            _fail(f"choice.{column} must be lowercase SHA-256")
        if column != "allocation_id" and choice[column].nunique(dropna=False) != 1:
            _fail(f"choice.{column} drifted")
    uniform_alloc = uniform.set_index("item_id").allocation_id.astype(str)
    choice_alloc = choice.set_index("item_id").allocation_id.astype(str)
    if not uniform_alloc.sort_index().equals(choice_alloc.sort_index()):
        _fail("choice uniform allocation IDs disagree with accuracy allocations")
    fp_direct = fp.set_index("item_id")
    for row in choice.itertuples(index=False):
        direct = fp_direct.loc[str(row.item_id)]
        parsed = _normalize_parsed(direct.parsed_answer)
        if bool(row.fp_greedy_valid) != bool(direct.valid):
            _fail(f"choice FP-valid provenance disagrees for {row.item_id}")
        if _normalize_parsed(row.fp_greedy_parsed_answer) != parsed:
            _fail(f"choice FP parsed answer disagrees for {row.item_id}")
        expected = bool(direct.valid and row.fp_canonical_choice == parsed)
        if bool(row.fp_canonical_agrees_with_greedy) != expected:
            _fail(f"choice canonical/direct flag disagrees for {row.item_id}")


def analyze_frames(accuracy: pd.DataFrame, choice: pd.DataFrame) -> dict[str, Any]:
    fp = accuracy.loc[accuracy.arm.astype(str) == "fp"].set_index("item_id")
    uniform = accuracy.loc[accuracy.arm.astype(str) == "uniform"].set_index("item_id")
    diagnostic = choice.set_index("item_id")
    fp_valid = int(fp.valid.sum())
    uniform_valid = int(uniform.valid.sum())
    fp_accuracy = float(fp.score.mean())
    uniform_accuracy = float(uniform.score.mean())
    invalid_fp_cap = int((~fp.valid & fp.reached_max_new).sum())
    valid_ids = fp.index[fp.valid]
    canonical = diagnostic.loc[valid_ids, "fp_canonical_choice"].astype(str)
    direct = fp.loc[valid_ids, "parsed_answer"].astype(str)
    agreement_count = int(canonical.eq(direct).sum())
    agreement = agreement_count / fp_valid if fp_valid else float("nan")
    gates = {
        "fp_valid_gate": fp_valid >= 18,
        "uniform_valid_gate": uniform_valid >= 16,
        "fp_accuracy_gate": 0.15 <= fp_accuracy <= 0.70,
        "uniform_accuracy_gate": 0.10 <= uniform_accuracy <= 0.70,
        "fp_cap_gate": invalid_fp_cap == 0,
        "fp_canonical_direct_gate": fp_valid > 0 and agreement >= 0.80,
    }
    return {
        "split": SPLIT,
        "n_items": EXPECTED_ROWS,
        "accuracy_rows": len(accuracy),
        "choice_rows": len(choice),
        "fp_valid": fp_valid,
        "uniform_valid": uniform_valid,
        "fp_accuracy": fp_accuracy,
        "uniform_accuracy": uniform_accuracy,
        "fp_minus_uniform": fp_accuracy - uniform_accuracy,
        "invalid_fp_at_cap": invalid_fp_cap,
        "fp_canonical_direct_agree_n": agreement_count,
        "fp_canonical_direct_valid_n": fp_valid,
        "fp_canonical_direct_agreement": agreement,
        **gates,
        "decision": "advance_development" if all(gates.values()) else "stop_v4",
    }


def load_pair(accuracy_path: str | Path, choice_path: str | Path, *,
              manifest_path: str | Path, dataset_path: str | Path
              ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    accuracy_path = Path(accuracy_path).resolve()
    choice_path = Path(choice_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    for path, label in ((accuracy_path, "accuracy parquet"),
                        (choice_path, "choice parquet"),
                        (manifest_path, "manifest"),
                        (Path(dataset_path).resolve(), "dataset")):
        if not path.is_file():
            _fail(f"missing {label}: {path}")
    accuracy_sidecar_path = accuracy_path.with_suffix(".json")
    choice_sidecar_path = choice_path.with_suffix(".json")
    if not accuracy_sidecar_path.is_file() or not choice_sidecar_path.is_file():
        _fail("both parquets require adjacent JSON sidecars")

    manifest, manifest_file_hash, qualification = authenticate_manifest(manifest_path)
    data_by_id = authenticate_dataset(dataset_path, qualification)
    try:
        accuracy = pd.read_parquet(accuracy_path)
        choice = pd.read_parquet(choice_path)
    except Exception as exc:
        _fail(f"cannot read parquet artifacts: {exc}")
    accuracy_sidecar = _load_json(accuracy_sidecar_path, "accuracy sidecar")
    choice_sidecar = _load_json(choice_sidecar_path, "choice sidecar")
    ids = [str(row["id"]) for row in qualification]
    _validate_sidecar_base(
        accuracy_sidecar, artifact="end_task_accuracy", parquet=accuracy_path,
        manifest_path=manifest_path, manifest=manifest,
        manifest_file_hash=manifest_file_hash, ids=ids)
    _validate_sidecar_base(
        choice_sidecar, artifact="choice_logit_diagnostic", parquet=choice_path,
        manifest_path=manifest_path, manifest=manifest,
        manifest_file_hash=manifest_file_hash, ids=ids)
    _same(_sha256_file(accuracy_path), accuracy_sidecar.get("parquet_sha256"),
          "accuracy sidecar parquet SHA-256")
    _same(choice_path.name, accuracy_sidecar.get("choice_parquet"),
          "accuracy sidecar choice parquet")
    _same(_sha256_file(choice_path), accuracy_sidecar.get("choice_sha256"),
          "accuracy sidecar choice SHA-256")
    _same(EXPECTED_CHOICE_ROWS, accuracy_sidecar.get("choice_rows"),
          "accuracy sidecar choice rows")
    _same(EXPECTED_ACCURACY_ROWS, accuracy_sidecar["rows"], "accuracy sidecar rows")
    _same(EXPECTED_ACCURACY_ROWS, accuracy_sidecar["expected_rows"],
          "accuracy sidecar expected rows")
    _same(["item_id", "arm"], accuracy_sidecar["row_key"], "accuracy row key")
    _same(True, accuracy_sidecar.get("official_invalid_scores_zero"),
          "accuracy invalid-score contract")
    _same(True, accuracy_sidecar.get("choice_diagnostic"),
          "accuracy choice-diagnostic flag")
    _same(_sha256_file(choice_path), choice_sidecar.get("parquet_sha256"),
          "choice sidecar parquet SHA-256")
    _same(EXPECTED_CHOICE_ROWS, choice_sidecar["rows"], "choice sidecar rows")
    _same(EXPECTED_CHOICE_ROWS, choice_sidecar["expected_rows"],
          "choice sidecar expected rows")
    _same(["item_id", "candidate"], choice_sidecar["row_key"], "choice row key")
    _same(list(CANDIDATES), choice_sidecar.get("candidates"), "choice candidates")
    _same("minimum_choice_kl_exact_ties_declared_order",
          choice_sidecar.get("selector"), "choice selector")
    _same(PD.CHOICE_RULE_VERSION, choice_sidecar.get("choice_rule_version"),
          "choice rule version")
    _same(list(LB.CANONICAL_RESPONSES), choice_sidecar.get("canonical_responses"),
          "canonical responses")
    _same(CHOICE_PREFIX_HASH, choice_sidecar.get("choice_prefix_token_hash"),
          "choice sidecar prefix-token SHA-256")
    _same(CHOICE_BRANCH_IDS, choice_sidecar.get("choice_branch_ids"),
          "choice sidecar branch token IDs")
    _same(accuracy_path.name, choice_sidecar.get("accuracy_parquet"),
          "choice sidecar accuracy parquet")
    _same(EXPECTED_ACCURACY_ROWS, choice_sidecar.get("accuracy_rows"),
          "choice sidecar accuracy rows")
    _same(_sha256_file(accuracy_path), choice_sidecar.get("accuracy_sha256"),
          "choice sidecar accuracy SHA-256")
    _same(True, choice_sidecar.get("no_raw_logits"), "choice no-raw-logits flag")
    if not choice.choice_prefix_token_hash.astype(str).eq(CHOICE_PREFIX_HASH).all():
        _fail("choice rows use an unexpected canonical-prefix token hash")
    if not choice.choice_branch_ids_hash.astype(str).eq(CHOICE_BRANCH_HASH).all():
        _fail("choice rows use unexpected A/B/C/D branch token IDs")
    validate_frames(
        accuracy, choice, manifest=manifest, manifest_file_hash=manifest_file_hash,
        qualification=qualification, data_by_id=data_by_id)
    return analyze_frames(accuracy, choice), accuracy, choice


def _fmt_float(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.3f}"


def print_summary(summary: Mapping[str, Any]) -> None:
    print("V4 LongBench-v2 qualification (frozen section 3I.5)")
    print("provenance      PASS (pinned dataset/manifest/model/prompt/parser/chat/split)")
    print(f"rows            {summary['accuracy_rows']} accuracy, "
          f"{summary['choice_rows']} choice; {summary['n_items']} manifest items")
    print(f"valid parses    FP {summary['fp_valid']}/20 "
          f"[{'PASS' if summary['fp_valid_gate'] else 'FAIL'}], "
          f"uniform {summary['uniform_valid']}/20 "
          f"[{'PASS' if summary['uniform_valid_gate'] else 'FAIL'}]")
    print(f"accuracy        FP {_fmt_float(float(summary['fp_accuracy']))} "
          f"[{'PASS' if summary['fp_accuracy_gate'] else 'FAIL'}], "
          f"uniform {_fmt_float(float(summary['uniform_accuracy']))} "
          f"[{'PASS' if summary['uniform_accuracy_gate'] else 'FAIL'}]")
    print(f"FP-uniform      {_fmt_float(float(summary['fp_minus_uniform']))} "
          "(descriptive only; no gap gate)")
    print(f"invalid FP cap  {summary['invalid_fp_at_cap']} "
          f"[{'PASS' if summary['fp_cap_gate'] else 'FAIL'}]")
    print(f"canonical/direct {summary['fp_canonical_direct_agree_n']}/"
          f"{summary['fp_canonical_direct_valid_n']} = "
          f"{_fmt_float(float(summary['fp_canonical_direct_agreement']))} "
          f"[{'PASS' if summary['fp_canonical_direct_gate'] else 'FAIL'}]")
    print(f"decision        {summary['decision']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accuracy", help="qualification end-task parquet")
    parser.add_argument("choice", help="qualification choice-diagnostic parquet")
    parser.add_argument("--manifest", required=True, help="frozen gold-free manifest")
    parser.add_argument("--dataset", required=True, help="pinned LongBench-v2 data JSON")
    parser.add_argument("--csv", help="write the one-row summary CSV")
    parser.add_argument("--validate-only", action="store_true",
                        help="authenticate silently; print no report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary, _, _ = load_pair(
            args.accuracy, args.choice, manifest_path=args.manifest,
            dataset_path=args.dataset)
        if args.csv:
            destination = Path(args.csv)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([summary]).to_csv(destination, index=False)
        if not args.validate_only:
            print_summary(summary)
        return 0
    except LongBenchQualificationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
