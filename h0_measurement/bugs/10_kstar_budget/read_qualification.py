#!/usr/bin/env python3
"""Strict offline reader for the frozen R9 K*-budget qualification.

The reader has three explicit modes:

* ``--preflight`` authenticates code, dependencies, PG19, and both model
  snapshots before a GPU job may be submitted;
* ``--validate-task-dir`` validates one array task before its worker writes
  ``COMPLETE``;
* ``--job-id`` authenticates both completed array tasks, applies Q1--Q4, and
  writes an advance lock only when every frozen gate passes.

A malformed or unauthenticated artifact exits 2.  A valid scientific gate
failure exits 0 with ``stop_r9_qualification`` and never creates a lock.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "h0_measurement"))

from h0_measurement import run_kstar_budget_qualification as RUN  # noqa: E402


READER_VERSION = "r9_kstar_budget_qualification_reader_v1"
INPUT_MANIFEST_VERSION = "r9_kstar_budget_qualification_input_manifest_v1"
LOCK_VERSION = "r9_kstar_budget_qualification_advance_lock_v1"
RESULT_PREFIX = "r9_kstar_qualification"
TASK_MODELS = {0: "llama31-8b", 1: "qwen15-moe-a2.7b"}
COMPLETE_BYTES = (RUN.PROTOCOL + "\n").encode("ascii")

INPUT_MANIFEST = ROOT / "h0_measurement/bugs/10_kstar_budget/qualification_input_manifest.json"
SOURCE_LEDGER = ROOT / "h0_measurement/bugs/10_kstar_budget/source_ledger.json"
CORPUS_ROOT = ROOT / ".h0_corpus/pg19"
CONFIG_PATH = ROOT / "h0_measurement/models.yaml"

ARTIFACTS = {
    "units": "kstar_budget_units.parquet",
    "groups": "kstar_budget_groups.parquet",
    "counts": "kstar_budget_frozen_counts.parquet",
    "heldout": "kstar_budget_heldout.parquet",
    "curves": "kstar_budget_curves.npz",
}
SUMMARY_NAME = "kstar_budget_summary.json"

SUMMARY_FIELDS = {
    "protocol", "runner_version", "model", "model_id",
    "experiment_config", "experiment_config_sha256", "models_yaml_sha256",
    "model_config_sha256", "effective_registry", "input_manifest",
    "model_snapshot", "quantizer_cache", "quantizer_levels8", "rotation",
    "cuda_environment", "placement", "corpus_sha",
    "corpus_preflight", "prompt_records", "source_ledger",
    "source_hashes", "source_set_sha256", "runtime_versions", "row_counts",
    "profile_shape", "g1_reference_shape", "g1_unit_audit",
    "frozen_policy_summaries",
    "calibration_profile_stability", "heldout_policy_summaries",
    "heldout_policy_totals", "elapsed_seconds", "artifact_sha256",
}

PROMPT_RECORD_FIELDS = {
    "prompt_idx", "family", "builder_ctx", "pre_crop_tokens", "crop_offset",
    "special_prefix_tokens", "special_prefix_sha256", "query_suffix_sha256",
    "prompt_token_sha256", "corpus_sha", "corpus_doc", "corpus_offset",
    "corpus_spliced", "synthetic", "n_prompt_tokens", "prefill_tokens",
    "context_tokens", "prefill_seconds",
}

FORBIDDEN_COLUMN_FRAGMENTS = (
    "answer", "correct", "label", "logit", "hidden_state", "attention_matrix",
    "raw_key", "raw_value", "decoded", "response_text",
)

Q2_MIN_SPEARMAN = 0.50
Q3_MIN_MOVEMENT = 0.05
Q4_MAX_SATURATION = 0.95
Q1_G1_MAX_ABS = 1e-10


class ReaderError(RuntimeError):
    """Authentication, schema, or invariant failure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReaderError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _regular(path: Path, label: str) -> Path:
    _require(path.is_file() and not path.is_symlink(),
             f"{label} is missing, nonregular, or symlinked: {path}")
    return path


def _directory(path: Path, label: str) -> Path:
    _require(path.is_dir() and not path.is_symlink(),
             f"{label} is missing, nondirectory, or symlinked: {path}")
    return path


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ReaderError(f"JSON object contains duplicate key {key!r}")
        out[key] = value
    return out


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _regular(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           object_pairs_hook=_no_duplicate_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReaderError(f"{label} is not strict JSON: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    got = set(value)
    _require(got == expected_set,
             f"{label} schema drift: missing={sorted(expected_set-got)}, "
             f"extra={sorted(got-expected_set)}")


def _safe_relative(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value and "\\" not in value,
             f"{label} must be a nonempty POSIX relative path")
    p = Path(value)
    _require(not p.is_absolute() and ".." not in p.parts and p.as_posix() == value,
             f"{label} is not a canonical relative path: {value!r}")
    return value


def _valid_sha(value: Any, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and value == value.lower()
             and all(c in "0123456789abcdef" for c in value),
             f"{label} is not a lowercase SHA-256")
    return value


def _as_nonnegative_int(value: Any, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
             f"{label} must be a nonnegative integer")
    return int(value)


def verify_input_manifest(
    path: Path = INPUT_MANIFEST,
    *,
    live_models: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Authenticate the sealed PG19 and pinned-snapshot byte inventory."""
    path = Path(path)
    _regular(path, "qualification input manifest")
    _regular(INPUT_MANIFEST, "canonical qualification input manifest")
    _require(path.resolve() == INPUT_MANIFEST.resolve(),
             f"input manifest must be the canonical path {INPUT_MANIFEST}")
    value = _load_json(path, "qualification input manifest")
    _exact_keys(value, {"manifest_version", "corpus", "model_snapshots",
                        "content_sha256"}, "qualification input manifest")
    _require(value["manifest_version"] == INPUT_MANIFEST_VERSION,
             "qualification input manifest version mismatch")
    body = {key: value[key] for key in
            ("manifest_version", "corpus", "model_snapshots")}
    _require(value["content_sha256"] == _canonical_sha(body),
             "qualification input manifest content hash mismatch")

    corpus = value["corpus"]
    _require(isinstance(corpus, dict), "input corpus record must be an object")
    _exact_keys(corpus, {"root", "manifest", "file_count", "total_bytes",
                         "inventory_sha256", "files"}, "input corpus record")
    _require(corpus["root"] == ".h0_corpus/pg19", "input corpus root drifted")
    corpus_root = _directory(ROOT / corpus["root"], "PG19 corpus root")
    manifest = corpus["manifest"]
    _require(isinstance(manifest, dict), "PG19 manifest attestation must be an object")
    _exact_keys(manifest, {"path", "bytes", "sha256", "declared_corpus_sha",
                           "declared_n_books", "declared_total_chars"},
                "PG19 manifest attestation")
    _require(manifest["path"] == ".h0_corpus/pg19/MANIFEST.json",
             "PG19 MANIFEST path drifted")
    manifest_path = _regular(ROOT / manifest["path"], "PG19 MANIFEST.json")
    _require(manifest_path.stat().st_size == _as_nonnegative_int(
        manifest["bytes"], "PG19 manifest bytes"), "PG19 manifest byte size mismatch")
    _require(_sha256(manifest_path) == _valid_sha(manifest["sha256"],
                                                  "PG19 manifest hash"),
             "PG19 manifest byte hash mismatch")
    declared = _load_json(manifest_path, "PG19 MANIFEST.json")
    for field, frozen_field in (("corpus_sha", "declared_corpus_sha"),
                                ("n_books", "declared_n_books"),
                                ("total_chars", "declared_total_chars")):
        _require(declared.get(field) == manifest[frozen_field],
                 f"PG19 declared {field} mismatch")

    files = corpus["files"]
    _require(isinstance(files, list), "PG19 files must be a list")
    _require(len(files) == _as_nonnegative_int(corpus["file_count"],
                                                "PG19 file_count") == 40,
             "PG19 file count must be exactly 40")
    expected_names: list[str] = []
    declared_files = declared.get("files")
    _require(isinstance(declared_files, list), "PG19 MANIFEST files must be a list")
    declared_by_name = {item.get("file"): item for item in declared_files
                        if isinstance(item, dict)}
    _require(len(declared_by_name) == len(declared_files) == 40,
             "PG19 MANIFEST has duplicate or malformed file records")
    total_bytes = 0
    for index, entry in enumerate(files):
        _require(isinstance(entry, dict), f"PG19 file entry {index} is not an object")
        _exact_keys(entry, {"path", "bytes", "sha256"}, f"PG19 file entry {index}")
        name = _safe_relative(entry["path"], f"PG19 file path {index}")
        _require(Path(name).parent == Path("."), "PG19 files must be direct children")
        expected_names.append(name)
        file_path = _regular(corpus_root / name, f"PG19 file {name}")
        size = _as_nonnegative_int(entry["bytes"], f"PG19 bytes {name}")
        digest = _valid_sha(entry["sha256"], f"PG19 hash {name}")
        _require(file_path.stat().st_size == size, f"PG19 byte size mismatch: {name}")
        _require(_sha256(file_path) == digest, f"PG19 byte hash mismatch: {name}")
        _require(name in declared_by_name
                 and declared_by_name[name].get("sha256") == digest,
                 f"PG19 frozen/declared inventory mismatch: {name}")
        total_bytes += size
    _require(expected_names == sorted(expected_names)
             and len(expected_names) == len(set(expected_names)),
             "PG19 frozen file list is not sorted and unique")
    live_names = sorted(p.name for p in corpus_root.iterdir())
    _require(live_names == sorted(["MANIFEST.json", *expected_names]),
             "PG19 directory contains missing or extra entries")
    _require(total_bytes == _as_nonnegative_int(corpus["total_bytes"],
                                                 "PG19 total_bytes"),
             "PG19 total byte count mismatch")
    _require(corpus["inventory_sha256"] == _canonical_sha({"files": files}),
             "PG19 inventory hash mismatch")

    snapshots = value["model_snapshots"]
    _require(isinstance(snapshots, dict), "model_snapshots must be an object")
    _require(set(snapshots) == set(RUN.MODELS), "input-manifest model set mismatch")
    selected = set(RUN.MODELS if live_models is None else live_models)
    _require(selected <= set(RUN.MODELS), f"unknown live-model selection {selected}")
    for tag, frozen in snapshots.items():
        _require(isinstance(frozen, dict), f"snapshot {tag} must be an object")
        _exact_keys(frozen, {"model_id", "revision", "root", "directories",
                             "file_count", "total_bytes", "inventory_sha256", "files"},
                    f"snapshot {tag}")
        spec = RUN.MODELS[tag]
        _require(frozen["model_id"] == spec["id"], f"{tag} model ID mismatch")
        _require(frozen["revision"] == spec["revision"], f"{tag} revision mismatch")
        expected_root = (f".hf_cache/hub/{spec['cache_dir']}/snapshots/"
                         f"{spec['revision']}")
        _require(frozen["root"] == expected_root, f"{tag} snapshot root mismatch")
        directories = frozen["directories"]
        files = frozen["files"]
        _require(isinstance(directories, list) and isinstance(files, list),
                 f"{tag} snapshot inventory must use lists")
        _require(directories == sorted(directories)
                 and len(directories) == len(set(directories)),
                 f"{tag} directory inventory is not sorted and unique")
        for index, name in enumerate(directories):
            _safe_relative(name, f"{tag} directory {index}")
        _require(len(files) == _as_nonnegative_int(frozen["file_count"],
                                                    f"{tag} file_count"),
                 f"{tag} file count mismatch")
        frozen_names: list[str] = []
        frozen_total = 0
        for index, entry in enumerate(files):
            _require(isinstance(entry, dict), f"{tag} file {index} is not an object")
            _exact_keys(entry, {"path", "kind", "symlink_target", "content_blob_id",
                                "bytes", "sha256"}, f"{tag} file {index}")
            name = _safe_relative(entry["path"], f"{tag} file path {index}")
            frozen_names.append(name)
            _require(entry["kind"] in {"symlink", "regular"},
                     f"{tag}/{name}: invalid file kind")
            if entry["kind"] == "symlink":
                _require(isinstance(entry["symlink_target"], str)
                         and isinstance(entry["content_blob_id"], str),
                         f"{tag}/{name}: incomplete symlink identity")
            else:
                _require(entry["symlink_target"] is None
                         and entry["content_blob_id"] is None,
                         f"{tag}/{name}: regular file has symlink fields")
            frozen_total += _as_nonnegative_int(entry["bytes"], f"{tag}/{name} bytes")
            _valid_sha(entry["sha256"], f"{tag}/{name} hash")
        _require(frozen_names == sorted(frozen_names)
                 and len(frozen_names) == len(set(frozen_names)),
                 f"{tag} file inventory is not sorted and unique")
        _require(frozen_total == _as_nonnegative_int(frozen["total_bytes"],
                                                      f"{tag} total_bytes"),
                 f"{tag} total byte count mismatch")
        _require(frozen["inventory_sha256"] == _canonical_sha(
            {"directories": directories, "files": files}),
            f"{tag} inventory hash mismatch")
        if tag not in selected:
            continue
        snapshot_root = _directory(ROOT / frozen["root"], f"{tag} snapshot root")
        live_dirs: list[str] = []
        live_files: list[str] = []
        for item in sorted(snapshot_root.rglob("*"),
                           key=lambda x: x.relative_to(snapshot_root).as_posix()):
            relative = item.relative_to(snapshot_root).as_posix()
            if item.is_dir() and not item.is_symlink():
                live_dirs.append(relative)
            else:
                live_files.append(relative)
        _require(live_dirs == directories, f"{tag} live directory inventory drifted")
        _require(live_files == frozen_names, f"{tag} live file inventory drifted")
        blob_root = (ROOT / ".hf_cache/hub" / spec["cache_dir"] / "blobs").resolve()
        for entry in files:
            item = snapshot_root / entry["path"]
            if entry["kind"] == "symlink":
                _require(item.is_symlink(), f"{tag}/{entry['path']} is no longer a symlink")
                _require(os.readlink(item) == entry["symlink_target"],
                         f"{tag}/{entry['path']} symlink target drifted")
            else:
                _require(item.is_file() and not item.is_symlink(),
                         f"{tag}/{entry['path']} is no longer regular")
            try:
                resolved = item.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ReaderError(f"{tag}/{entry['path']} cannot be resolved: {exc}") from exc
            _regular(resolved, f"{tag}/{entry['path']} content blob")
            if entry["kind"] == "symlink":
                try:
                    resolved.relative_to(blob_root)
                except ValueError as exc:
                    raise ReaderError(f"{tag}/{entry['path']} escapes its blob store") from exc
                _require(resolved.name == entry["content_blob_id"],
                         f"{tag}/{entry['path']} blob identity drifted")
            _require(resolved.stat().st_size == entry["bytes"],
                     f"{tag}/{entry['path']} byte size drifted")
            _require(_sha256(resolved) == entry["sha256"],
                     f"{tag}/{entry['path']} byte hash drifted")

    return {
        "path": RUN.INPUT_MANIFEST_RELATIVE,
        "manifest_version": value["manifest_version"],
        "content_sha256": value["content_sha256"],
        "file_sha256": _sha256(path),
        "corpus": value["corpus"],
        "model_snapshots": value["model_snapshots"],
    }


def verify_source_ledger(path: Path = SOURCE_LEDGER) -> dict[str, Any]:
    path = Path(path)
    _regular(path, "R9 source ledger")
    _regular(SOURCE_LEDGER, "canonical R9 source ledger")
    _require(path.resolve() == SOURCE_LEDGER.resolve(),
             f"source ledger must be the canonical path {SOURCE_LEDGER}")
    value = _load_json(path, "R9 source ledger")
    _exact_keys(value, {"ledger_version", "source_sha256", "content_sha256"},
                "R9 source ledger")
    _require(value["ledger_version"] == RUN.SOURCE_LEDGER_VERSION,
             "source ledger version mismatch")
    hashes = value["source_sha256"]
    _require(isinstance(hashes, dict) and set(hashes) == set(RUN.SOURCE_FILES),
             "source ledger does not contain the exact runner source closure")
    required_decision_sources = {
        "h0_measurement/bugs/10_kstar_budget/qualification_protocol_frozen.md",
        "h0_measurement/bugs/10_kstar_budget/read_qualification.py",
        "h0_measurement/bugs/10_kstar_budget/steps.sh",
    }
    _require(required_decision_sources <= set(hashes),
             "source closure omits the frozen protocol, reader, or command interface")
    mutable_docs = {
        "h0_measurement/bugs/10_kstar_budget/plan.md",
        "h0_measurement/bugs/10_kstar_budget/report.md",
    }
    _require(not (mutable_docs & set(hashes)),
             "source closure incorrectly seals a mutable plan/report document")
    for relative, digest in hashes.items():
        _safe_relative(relative, "source-ledger path")
        _valid_sha(digest, f"source-ledger hash {relative}")
        source = _regular(ROOT / relative, f"sealed source {relative}")
        _require(_sha256(source) == digest, f"sealed source hash drifted: {relative}")
    body = {"ledger_version": value["ledger_version"], "source_sha256": hashes}
    _require(value["content_sha256"] == _canonical_sha(body),
             "source ledger content hash mismatch")
    return {
        "path": RUN.SOURCE_LEDGER_RELATIVE,
        "ledger_version": value["ledger_version"],
        "content_sha256": value["content_sha256"],
        "file_sha256": _sha256(path),
    }


def preflight(input_manifest: Path, source_ledger: Path) -> None:
    versions = RUN.validate_runtime_dependencies()
    ledger = verify_source_ledger(source_ledger)
    inputs = verify_input_manifest(input_manifest)
    for model in TASK_MODELS.values():
        RUN.validate_registry_config(str(CONFIG_PATH), model)
    print(f"PASS preflight protocol={RUN.PROTOCOL}")
    print(f"runtime={json.dumps(versions, sort_keys=True)}")
    print(f"source_ledger={ledger['content_sha256']}")
    print(f"input_manifest={inputs['content_sha256']}")
    print("models=llama31-8b,qwen15-moe-a2.7b corpus_files=40")


def _read_parquet(path: Path, columns: Sequence[str], label: str) -> pd.DataFrame:
    _regular(path, label)
    try:
        frame = pd.read_parquet(path, engine="fastparquet")
    except Exception as exc:
        raise ReaderError(f"could not read {label}: {exc}") from exc
    _require(tuple(map(str, frame.columns)) == tuple(columns),
             f"{label} schema/order drifted")
    for column in frame.columns:
        lower = str(column).lower()
        _require(not any(fragment in lower for fragment in FORBIDDEN_COLUMN_FRAGMENTS),
                 f"{label} contains forbidden raw/label column {column!r}")
    return frame


def _check_constant(frame: pd.DataFrame, column: str, expected: Any, label: str) -> None:
    _require(column in frame and bool((frame[column] == expected).all()),
             f"{label}.{column} drifted from {expected!r}")


def _check_prompt_records(summary: Mapping[str, Any], model: str) -> dict[tuple[int, str], dict[str, Any]]:
    records = summary.get("prompt_records")
    _require(isinstance(records, list) and len(records) == len(RUN.UNIT_ORDER),
             f"{model}: prompt_records has the wrong length")
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for index, record in enumerate(records):
        _require(isinstance(record, dict), f"{model}: prompt record {index} is not an object")
        _exact_keys(record, PROMPT_RECORD_FIELDS, f"{model}: prompt record {index}")
        key = (int(record["prompt_idx"]), str(record["family"]))
        _require(key == RUN.UNIT_ORDER[index], f"{model}: prompt record order drifted")
        _require(key not in out, f"{model}: duplicate prompt record {key}")
        _require(record["builder_ctx"] == RUN.BUILDER_CTX
                 and record["n_prompt_tokens"] == RUN.CTX
                 and record["prefill_tokens"] == RUN.PREFILL_TOKENS
                 and record["context_tokens"] == RUN.CONTEXT_TOKENS,
                 f"{model}: prompt length provenance drifted for {key}")
        _require(record["corpus_sha"] and record["corpus_doc"]
                 and record["synthetic"] is False,
                 f"{model}: incomplete/ synthetic prompt provenance for {key}")
        for field in ("special_prefix_sha256", "query_suffix_sha256",
                      "prompt_token_sha256"):
            _valid_sha(record[field], f"{model}/{key}/{field}")
        for field in ("pre_crop_tokens", "crop_offset", "special_prefix_tokens",
                      "corpus_offset"):
            _as_nonnegative_int(record[field], f"{model}/{key}/{field}")
        elapsed = float(record["prefill_seconds"])
        _require(math.isfinite(elapsed) and elapsed >= 0,
                 f"{model}/{key}: prefill_seconds is invalid")
        out[key] = record
    return out


def _invoke_runner_validation(out: Path, model: str, source_ledger: Path,
                              input_manifest: Path) -> None:
    """Call the independently sealed structural validator without rehashing inputs."""
    import inspect
    signature = inspect.signature(RUN.validate_artifacts)
    kwargs: dict[str, Any] = {"verify_environment": False}
    if "source_ledger" in signature.parameters:
        kwargs["source_ledger"] = str(source_ledger)
    if "input_manifest" in signature.parameters:
        kwargs["input_manifest"] = str(input_manifest)
    try:
        RUN.validate_artifacts(out, model, **kwargs)
    except Exception as exc:
        raise ReaderError(f"{model}: runner structural validation failed: {exc}") from exc


def validate_task(
    out: Path,
    model: str,
    *,
    input_manifest: Path,
    source_ledger: Path,
    require_complete: bool,
    verify_live_inputs: bool,
    input_attestation: Mapping[str, Any] | None = None,
    ledger_attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _directory(out, f"{model} task directory")
    expected_files = set(ARTIFACTS.values()) | {SUMMARY_NAME}
    if require_complete:
        expected_files.add("COMPLETE")
    live_files = {item.name for item in out.iterdir()}
    _require(live_files == expected_files,
             f"{model}: task directory schema drift: "
             f"missing={sorted(expected_files-live_files)}, "
             f"extra={sorted(live_files-expected_files)}")
    for name in expected_files:
        _regular(out / name, f"{model} task file {name}")
    if require_complete:
        _require((out / "COMPLETE").read_bytes() == COMPLETE_BYTES,
                 f"{model}: COMPLETE marker mismatch")

    if ledger_attestation is None:
        ledger_attestation = verify_source_ledger(source_ledger)
    if input_attestation is None:
        input_attestation = verify_input_manifest(
            input_manifest, live_models=(model,) if verify_live_inputs else ())
    RUN.validate_runtime_dependencies()
    _invoke_runner_validation(out, model, source_ledger, input_manifest)

    summary_path = out / SUMMARY_NAME
    summary = _load_json(summary_path, f"{model} summary")
    _exact_keys(summary, SUMMARY_FIELDS, f"{model} summary")
    _require(summary["protocol"] == RUN.PROTOCOL
             and summary["runner_version"] == RUN.RUNNER_VERSION,
             f"{model}: protocol/runner mismatch")
    _require(summary["model"] == model
             and summary["model_id"] == RUN.MODELS[model]["id"],
             f"{model}: model identity mismatch")
    config = RUN.experiment_config(model)
    _require(summary["experiment_config"] == config
             and summary["experiment_config_sha256"] == _canonical_sha(config),
             f"{model}: experiment config mismatch")
    _require(summary["models_yaml_sha256"] == _sha256(CONFIG_PATH),
             f"{model}: models.yaml hash mismatch")
    _require(summary["source_ledger"] == ledger_attestation,
             f"{model}: source-ledger attestation mismatch")
    _require(summary["input_manifest"] == input_attestation,
             f"{model}: full input-manifest attestation mismatch")
    current_sources = {relative: _sha256(ROOT / relative)
                       for relative in RUN.SOURCE_FILES}
    _require(summary["source_hashes"] == current_sources
             and summary["source_set_sha256"] == _canonical_sha(current_sources),
             f"{model}: executed source closure mismatch")
    _require(summary["runtime_versions"] == RUN.validate_runtime_dependencies(),
             f"{model}: runtime version attestation mismatch")
    elapsed = float(summary["elapsed_seconds"])
    _require(math.isfinite(elapsed) and elapsed >= 0,
             f"{model}: elapsed_seconds is invalid")

    _require(summary["model_snapshot"]
             == input_attestation["model_snapshots"][model],
             f"{model}: selected full snapshot declaration mismatch")
    registry = RUN.validate_registry_config(str(CONFIG_PATH), model)
    registry_keys = ("dtype", "device_map", "chunk", "norm_correct", "rot_seed", "maxb")
    _require(summary["effective_registry"] == {key: registry[key] for key in registry_keys},
             f"{model}: effective registry attestation mismatch")
    quantizer_cache = RUN.validate_quantizer_cache()
    _require(summary["quantizer_cache"] == quantizer_cache,
             f"{model}: quantizer-cache attestation mismatch")
    expected_levels = {
        "shape": [256], "dtype": "float32",
        "sha256": quantizer_cache["level_tensor_sha256"],
    }
    _require(summary["quantizer_levels8"] == expected_levels,
             f"{model}: live 8-bit level attestation mismatch")
    rotation = RUN.quant.random_rotation(
        RUN.MODELS[model]["geometry"]["head_dim"], "cpu", RUN.torch.float32,
        seed=0)
    _require(summary["rotation"] == RUN.float32_tensor_attestation(rotation),
             f"{model}: rotation attestation mismatch")
    cuda = summary["cuda_environment"]
    _require(isinstance(cuda, dict) and set(cuda) == {
        "torch_cuda_runtime", "device_index", "device_name", "compute_capability",
        "total_memory_bytes", "matmul_allow_tf32", "cudnn_allow_tf32",
        "float32_matmul_precision"}, f"{model}: CUDA attestation schema mismatch")
    _require(isinstance(cuda["torch_cuda_runtime"], str)
             and cuda["torch_cuda_runtime"] == str(RUN.torch.version.cuda)
             and isinstance(cuda["device_index"], int)
             and isinstance(cuda["device_name"], str) and cuda["device_name"]
             and isinstance(cuda["compute_capability"], list)
             and len(cuda["compute_capability"]) == 2
             and all(isinstance(x, int) and x >= 0 for x in cuda["compute_capability"])
             and isinstance(cuda["total_memory_bytes"], int)
             and cuda["total_memory_bytes"] > 0
             and isinstance(cuda["matmul_allow_tf32"], bool)
             and isinstance(cuda["cudnn_allow_tf32"], bool)
             and cuda["float32_matmul_precision"] in {"highest", "high", "medium"},
             f"{model}: CUDA execution attestation values are invalid")
    placement = summary["placement"]
    _require(isinstance(placement, dict)
             and isinstance(placement.get("parameter_devices"), list)
             and placement["parameter_devices"]
             and all(str(device).startswith("cuda")
                     for device in placement["parameter_devices"]),
             f"{model}: placement attestation is not fully CUDA")
    _require(summary["corpus_sha"] == input_attestation["corpus"]["manifest"]
             ["declared_corpus_sha"], f"{model}: corpus identity mismatch")
    prompt_records = _check_prompt_records(summary, model)

    paths = {key: out / name for key, name in ARTIFACTS.items()}
    _require(set(summary["artifact_sha256"]) == set(ARTIFACTS),
             f"{model}: artifact hash map schema drifted")
    for key, artifact_path in paths.items():
        _require(summary["artifact_sha256"][key] == _sha256(artifact_path),
                 f"{model}: {key} artifact hash mismatch")
    units = _read_parquet(paths["units"], RUN.UNIT_COLUMNS, f"{model} units")
    groups = _read_parquet(paths["groups"], RUN.GROUP_COLUMNS, f"{model} groups")
    counts = _read_parquet(paths["counts"], RUN.COUNT_COLUMNS, f"{model} counts")
    heldout = _read_parquet(paths["heldout"], RUN.HELDOUT_COLUMNS, f"{model} heldout")
    expected_rows = {"units": len(units), "groups": len(groups),
                     "counts": len(counts), "heldout": len(heldout)}
    _require(summary["row_counts"] == expected_rows,
             f"{model}: row-count summary mismatch")
    for frame_name, frame in (("units", units), ("groups", groups),
                              ("counts", counts), ("heldout", heldout)):
        for column, expected in (("protocol", RUN.PROTOCOL),
                                 ("runner_version", RUN.RUNNER_VERSION),
                                 ("model", model),
                                 ("model_id", RUN.MODELS[model]["id"])):
            _check_constant(frame, column, expected, f"{model} {frame_name}")

    # Recompute all count policies identifiable from the serialized K*/n95
    # signals; the published score allocators are audited by exact totals and by
    # frozen reuse because their dense score tensors are intentionally absent.
    geometry = RUN.MODELS[model]["geometry"]
    n_groups = int(geometry["layers"]) * int(geometry["kv_heads"])
    target = n_groups * RUN.K0
    calibration = (groups[groups.split == "calibration"]
                   .sort_values(["layer", "kv_head"]).reset_index(drop=True))
    count_sorted = counts.sort_values(["layer", "kv_head"]).reset_index(drop=True)
    _require(np.array_equal(count_sorted.kstar.to_numpy(dtype=np.int64),
                            calibration.kstar.to_numpy(dtype=np.int64))
             and np.allclose(count_sorted.n95_mean.to_numpy(dtype=np.float64),
                             calibration.n95_mean.to_numpy(dtype=np.float64),
                             rtol=0.0, atol=1e-14),
             f"{model}: frozen count signals differ from calibration profiles")
    _require(bool((counts.calibration_prompt_ids
                   == json.dumps(list(RUN.CALIBRATION_PROMPT_IDS))).all())
             and bool((counts.calibration_units == len(RUN.CALIBRATION_UNIT_ORDER)).all()),
             f"{model}: frozen count calibration metadata mismatch")
    shape = (int(geometry["layers"]), int(geometry["kv_heads"]))
    core = RUN.policy_counts(
        RUN.torch.from_numpy(calibration.kstar.to_numpy(dtype=np.int64).reshape(shape)),
        RUN.torch.from_numpy(calibration.n95_mean.to_numpy(dtype=np.float64).reshape(shape)),
        geometry)
    core_columns = {
        "uniform": "count_fixed",
        "prop": "count_kstar_prop_calibrated",
        "shrink20": "count_kstar_shrink20_calibrated",
        "n95": "count_kappa4_rebalanced_calibrated",
        "n95_raw": "count_kappa4_natural_calibrated",
    }
    for core_name, column in core_columns.items():
        _require(np.array_equal(
            count_sorted[column].to_numpy(dtype=np.int64),
            core[core_name].cpu().numpy().reshape(-1)),
            f"{model}: deterministic frozen policy drifted: {column}")
    lower_zero = {"adakv_alpha02_calibrated", "layer_flat_calibrated",
                  "snapkv_global_calibrated"}
    matched = [policy for policy in RUN.FROZEN_POLICIES if "natural" not in policy]
    budget_ok = True
    for policy in matched:
        values = counts["count_" + policy].to_numpy(dtype=np.int64)
        lower = 0 if policy in lower_zero else 1
        budget_ok &= (int(values.sum()) == target and bool(np.all(values >= lower))
                      and bool(np.all(values <= RUN.CONTEXT_TOKENS)))
    natural = counts["count_kappa4_natural_calibrated"].to_numpy(dtype=np.int64)
    budget_ok &= (int(natural.sum()) <= target and bool(np.all(natural >= 1))
                  and bool(np.all(natural <= RUN.CONTEXT_TOKENS)))
    _require(budget_ok, f"{model}: frozen exact-budget audit failed")

    for (prompt_idx, family, policy), block in heldout.groupby(
            ["prompt_idx", "family", "policy"], sort=False, observed=True):
        values = block.keep_count.to_numpy(dtype=np.int64)
        is_natural = policy in {"kappa4_natural_calibrated",
                                "kappa4_natural_dynamic"}
        total = int(values.sum())
        _require((is_natural and total <= target) or
                 (not is_natural and total == target),
                 f"{model}/p{prompt_idx}/{family}/{policy}: token total mismatch")
        expected_bits = values.astype(np.float64) * RUN.MAXB / RUN.CONTEXT_TOKENS
        got_bits = block.bits_per_context_token.to_numpy(dtype=np.float64)
        _require(bool(np.allclose(got_bits, expected_bits, rtol=0.0, atol=1e-12)),
                 f"{model}/p{prompt_idx}/{family}/{policy}: bit audit mismatch")
        _require(bool((block.budget_matched == (not is_natural)).all()),
                 f"{model}/p{prompt_idx}/{family}/{policy}: budget flag mismatch")
    for policy_index, policy in enumerate(RUN.HELDOUT_POLICIES):
        block = heldout[heldout.policy == policy]
        _require(bool((block.policy_order == policy_index).all()),
                 f"{model}/{policy}: policy order mismatch")
        _require(bool((block.calibrated == (policy in RUN.FROZEN_POLICIES)).all())
                 and bool((block.dynamic == (policy in RUN.DYNAMIC_POLICIES)).all()),
                 f"{model}/{policy}: calibrated/dynamic flags mismatch")
    errors = heldout.squared_relative_output_error.to_numpy(dtype=np.float64)
    _require(bool(np.isfinite(errors).all()) and bool(np.all(errors >= 0)),
             f"{model}: held-out errors are nonfinite or negative")
    is_g1 = int(geometry["query_heads"]) == int(geometry["kv_heads"])
    expected_g1_audit = {
        "reference_available": is_g1,
        "rows": len(units),
        "max_abs": float(units.g1_max_abs.max()),
        "all_kstar_match": bool(units.g1_kstar_match.all()) if is_g1 else False,
    }
    _require(summary["g1_unit_audit"] == expected_g1_audit,
             f"{model}: per-unit G=1 summary mismatch")

    with np.load(paths["curves"], allow_pickle=False) as archive:
        _require(set(archive.files) == {"curves", "g1_reference_curves", "split_names"},
                 f"{model}: curve archive member schema drifted")
        curves = archive["curves"]
        references = archive["g1_reference_curves"]
        split_names = archive["split_names"].tolist()
    _require(split_names == list(RUN.SPLITS), f"{model}: split-name archive drifted")
    _require(summary["profile_shape"] == list(curves.shape)
             and summary["g1_reference_shape"] == list(references.shape),
             f"{model}: profile shape summary mismatch")
    return {
        "model": model, "out": out, "summary": summary,
        "summary_sha256": _sha256(summary_path), "units": units,
        "groups": groups, "counts": counts, "heldout": heldout,
        "curves": curves, "g1_references": references,
        "prompt_records": prompt_records, "budget_ok": bool(budget_ok),
        "artifact_sha256": {name: _sha256(out / name) for name in expected_files},
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    a, b = _average_ranks(left), _average_ranks(right)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def _cross_task_checks(tasks: Mapping[str, Mapping[str, Any]]) -> None:
    llama, qwen = tasks["llama31-8b"], tasks["qwen15-moe-a2.7b"]
    for field in ("source_ledger", "source_hashes", "source_set_sha256",
                  "runtime_versions", "corpus_sha", "input_manifest"):
        _require(llama["summary"][field] == qwen["summary"][field],
                 f"cross-task {field} differs")
    provenance_fields = ("builder_ctx", "corpus_sha", "corpus_doc",
                         "corpus_offset", "corpus_spliced")
    for key in RUN.UNIT_ORDER:
        a = llama["prompt_records"][key]
        b = qwen["prompt_records"][key]
        for field in provenance_fields:
            _require(a[field] == b[field],
                     f"cross-model shared-haystack provenance differs for {key}/{field}")


def compute_decision(tasks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    llama = tasks["llama31-8b"]
    qwen = tasks["qwen15-moe-a2.7b"]

    qwen_curves = np.asarray(qwen["curves"], dtype=np.float64)
    qwen_refs = np.asarray(qwen["g1_references"], dtype=np.float64)
    averaged_g1_max_abs = float(np.max(np.abs(qwen_curves - qwen_refs)))
    qwen_units = qwen["units"]
    unit_g1_max_abs = float(qwen_units.g1_max_abs.max())
    unit_g1_kstar_match = bool(qwen_units.g1_reference_available.all()
                               and qwen_units.g1_kstar_match.all())
    qwen_group = qwen["groups"]
    averaged_g1_kstar_match = bool(qwen_group.g1_reference_available.all()
                                    and qwen_group.g1_kstar_match.all())
    g1_max_abs = max(unit_g1_max_abs, averaged_g1_max_abs)
    g1_kstar_match = unit_g1_kstar_match and averaged_g1_kstar_match
    q1 = bool(llama["budget_ok"] and qwen["budget_ok"]
              and g1_max_abs <= Q1_G1_MAX_ABS and g1_kstar_match)

    groups = llama["groups"]
    p0 = (groups[groups.split == "cal_p0"].sort_values(["layer", "kv_head"])
          .kstar.to_numpy(dtype=np.float64))
    p1 = (groups[groups.split == "cal_p1"].sort_values(["layer", "kv_head"])
          .kstar.to_numpy(dtype=np.float64))
    stability = _spearman(p0, p1)
    q2 = bool(math.isfinite(stability) and stability >= Q2_MIN_SPEARMAN)

    counts = llama["counts"]
    prop = counts.count_kstar_prop_calibrated.to_numpy(dtype=np.float64)
    target = len(prop) * RUN.K0
    movement = float(np.abs(prop - RUN.K0).sum() / (2.0 * target))
    q3 = bool(movement >= Q3_MIN_MOVEMENT)

    calibration = groups[groups.split == "calibration"]
    saturation = float(np.mean(calibration.kstar.to_numpy(dtype=np.int64) == RUN.K0))
    q4 = bool(saturation < Q4_MAX_SATURATION)
    advance = q1 and q2 and q3 and q4
    return {
        "q1_implementation_budget": q1,
        "q1_g1_max_abs": g1_max_abs,
        "q1_g1_unit_max_abs": unit_g1_max_abs,
        "q1_g1_averaged_max_abs": averaged_g1_max_abs,
        "q1_g1_kstar_match": g1_kstar_match,
        "q2_calibration_stability": q2,
        "q2_llama_p0_p1_spearman": stability,
        "q3_nontrivial_redistribution": q3,
        "q3_llama_prop_movement_fraction": movement,
        "q4_resolution": q4,
        "q4_llama_kstar_eq_k0_fraction": saturation,
        "decision": "advance_development" if advance else "stop_r9_qualification",
    }


def heldout_rows(job_id: str, tasks: Mapping[str, Mapping[str, Any]],
                 decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return model aggregates and paired prompt/family held-out summaries."""
    rows: list[dict[str, Any]] = []
    for task_index, model in TASK_MODELS.items():
        heldout = tasks[model]["heldout"]
        cells: list[tuple[str, int | None, str | None, pd.DataFrame]] = [
            ("model", None, None, heldout)
        ]
        for prompt_idx in RUN.HELDOUT_PROMPT_IDS:
            for family in RUN.FAMILIES:
                block = heldout[(heldout.prompt_idx == prompt_idx)
                                & (heldout.family == family)]
                cells.append(("prompt_family", prompt_idx, family, block))
        for scope, prompt_idx, family, cell in cells:
            fixed = cell[cell.policy == "fixed"]
            _require(len(fixed) > 0,
                     f"{model}/{scope}/p{prompt_idx}/{family}: missing fixed rows")
            fixed_mean = float(fixed.squared_relative_output_error.mean())
            for policy in RUN.HELDOUT_POLICIES:
                values = cell[cell.policy == policy]
                _require(len(values) > 0,
                         f"{model}/{scope}/p{prompt_idx}/{family}: missing {policy} rows")
                mean_error = float(values.squared_relative_output_error.mean())
                rows.append({
                    "job_id": job_id, "array_task": task_index, "model": model,
                    "scope": scope, "prompt_idx": prompt_idx, "family": family,
                    "policy": policy, "heldout_rows": len(values),
                    "heldout_mean_squared_relative_output_error": mean_error,
                    "heldout_ratio_to_fixed_mean": (
                        mean_error / fixed_mean if fixed_mean != 0.0 else None),
                    **decision,
                })
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    _require(bool(rows), "cannot serialize an empty summary CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_once(path: Path, payload: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.is_symlink(), f"refusing symlink {label} target {path}")
    if path.exists():
        _require(path.is_file() and path.read_bytes() == payload,
                 f"refusing different existing {label} {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _lock_payload(job_id: str, tasks: Mapping[str, Mapping[str, Any]],
                  decision: Mapping[str, Any], csv_path: Path,
                  input_attestation: Mapping[str, Any],
                  ledger_attestation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "lock_version": LOCK_VERSION,
        "protocol": RUN.PROTOCOL,
        "runner_version": RUN.RUNNER_VERSION,
        "reader_version": READER_VERSION,
        "decision": "advance_development",
        "job_id": job_id,
        "qualification_gates": dict(decision),
        "tasks": {
            model: {
                "array_task": task,
                "directory": str(tasks[model]["out"].relative_to(ROOT)),
                "summary_sha256": tasks[model]["summary_sha256"],
                "files_sha256": tasks[model]["artifact_sha256"],
            }
            for task, model in TASK_MODELS.items()
        },
        "source_ledger": dict(ledger_attestation),
        "input_manifest": {
            key: input_attestation[key]
            for key in ("path", "manifest_version", "content_sha256", "file_sha256")
        },
        "reader_source": str(Path(__file__).resolve().relative_to(ROOT)),
        "reader_source_sha256": _sha256(Path(__file__).resolve()),
        "summary_csv": str(csv_path.relative_to(ROOT)),
        "summary_csv_sha256": _sha256(csv_path),
    }


def read_job(args: argparse.Namespace) -> None:
    job_id = str(args.job_id)
    _require(job_id.isdigit() and int(job_id) > 0, "job ID must be a positive integer")
    results_root = _directory(args.results_root.resolve(), "results root")
    expected_dirs = {
        results_root / f"{RESULT_PREFIX}_{job_id}_{task}" for task in TASK_MODELS}
    found_dirs = set(results_root.glob(f"{RESULT_PREFIX}_{job_id}_*"))
    _require(found_dirs == expected_dirs,
             f"job {job_id} task directory set mismatch: "
             f"missing={sorted(map(str, expected_dirs-found_dirs))}, "
             f"extra={sorted(map(str, found_dirs-expected_dirs))}")

    ledger_attestation = verify_source_ledger(args.source_ledger)
    input_attestation = verify_input_manifest(args.input_manifest)
    RUN.validate_runtime_dependencies()
    tasks: dict[str, dict[str, Any]] = {}
    for task, model in TASK_MODELS.items():
        out = results_root / f"{RESULT_PREFIX}_{job_id}_{task}"
        tasks[model] = validate_task(
            out, model, input_manifest=args.input_manifest,
            source_ledger=args.source_ledger, require_complete=True,
            verify_live_inputs=False, input_attestation=input_attestation,
            ledger_attestation=ledger_attestation)
    _cross_task_checks(tasks)
    decision = compute_decision(tasks)
    rows = heldout_rows(job_id, tasks, decision)
    csv_payload = _csv_bytes(rows)
    _write_once(args.csv.resolve(), csv_payload, "qualification summary CSV")

    if decision["decision"] == "advance_development":
        lock = _lock_payload(job_id, tasks, decision, args.csv.resolve(),
                             input_attestation, ledger_attestation)
        lock_payload = (json.dumps(lock, indent=2, sort_keys=True,
                                   ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
        _write_once(args.lock.resolve(), lock_payload, "qualification advance lock")
    else:
        _require(not args.lock.exists() and not args.lock.is_symlink(),
                 f"stopped qualification refuses existing advance lock {args.lock}")

    print(f"R9 K* QUALIFICATION job={job_id}")
    print("authentication  PASS (2/2 exact array tasks; source/input provenance sealed)")
    print(f"Q1 implementation+budget  {'PASS' if decision['q1_implementation_budget'] else 'FAIL'} "
          f"(G=1 max_abs={decision['q1_g1_max_abs']:.3e}, "
          f"K* match={decision['q1_g1_kstar_match']})")
    print(f"Q2 calibration stability  {'PASS' if decision['q2_calibration_stability'] else 'FAIL'} "
          f"(Llama p0/p1 Spearman={decision['q2_llama_p0_p1_spearman']:.6f}; "
          f"minimum={Q2_MIN_SPEARMAN:.2f})")
    print(f"Q3 redistribution         {'PASS' if decision['q3_nontrivial_redistribution'] else 'FAIL'} "
          f"(movement={decision['q3_llama_prop_movement_fraction']:.6f}; "
          f"minimum={Q3_MIN_MOVEMENT:.2f})")
    print(f"Q4 integer resolution     {'PASS' if decision['q4_resolution'] else 'FAIL'} "
          f"(K*=k0 fraction={decision['q4_llama_kstar_eq_k0_fraction']:.6f}; "
          f"must be <{Q4_MAX_SATURATION:.2f})")
    print(f"decision        {decision['decision']}")
    print("held-out errors are descriptive and did not enter Q1--Q4")
    for model in TASK_MODELS.values():
        model_rows = [row for row in rows
                      if row["model"] == model and row["scope"] == "model"]
        rendered = ", ".join(
            f"{row['policy']}={row['heldout_mean_squared_relative_output_error']:.6g}"
            for row in model_rows)
        print(f"{model}: {rendered}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--validate-task-dir", type=Path)
    mode.add_argument("--job-id")
    parser.add_argument("--model", choices=tuple(RUN.MODELS))
    parser.add_argument("--results-root", type=Path,
                        default=ROOT / "h0_measurement/results")
    parser.add_argument("--input-manifest", type=Path, default=INPUT_MANIFEST)
    parser.add_argument("--source-ledger", type=Path, default=SOURCE_LEDGER)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args(argv)
    if args.validate_task_dir is not None:
        parser.error("--validate-task-dir requires --model") if args.model is None else None
        if args.csv is not None or args.lock is not None:
            parser.error("task validation does not accept --csv/--lock")
    elif args.job_id is not None:
        if args.model is not None or args.csv is None or args.lock is None:
            parser.error("--job-id requires --csv and --lock and does not accept --model")
    else:
        if args.model is not None or args.csv is not None or args.lock is not None:
            parser.error("--preflight does not accept --model/--csv/--lock")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.preflight:
            preflight(args.input_manifest, args.source_ledger)
        elif args.validate_task_dir is not None:
            ledger = verify_source_ledger(args.source_ledger)
            inputs = verify_input_manifest(args.input_manifest, live_models=(args.model,))
            validate_task(
                args.validate_task_dir.resolve(), args.model,
                input_manifest=args.input_manifest, source_ledger=args.source_ledger,
                require_complete=False, verify_live_inputs=False,
                input_attestation=inputs, ledger_attestation=ledger)
            print(f"PASS {RUN.PROTOCOL} task-validation model={args.model} "
                  f"out={args.validate_task_dir}")
        else:
            read_job(args)
    except (ReaderError, RuntimeError, OSError, ValueError, TypeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
