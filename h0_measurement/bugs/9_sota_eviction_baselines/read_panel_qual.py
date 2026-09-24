#!/usr/bin/env python3
"""Authenticate and summarize the frozen V3 multikey-panel qualification.

This reader deliberately accepts one artifact only: Llama-3.1-8B at 32K,
question-agnostic decoding, B=2, k48/v4/h4, prompts 700--739, and the
``multikey_panel_v1`` task variant with FP and uniform arms.  Every context has
four independently decoded questions over one shared context and allocation.

The primary query outcome is ``first_ok``.  Query outcomes are averaged within
each context before halves and paired bootstrap intervals are computed.  A
valid but ineligible qualification exits zero; schema/provenance failures exit
two.
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
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


MODEL = "llama31-8b"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
CTX = 32_768
NATIVE_CTX = 131_072
TASK = "niah_multikey"
TASK_VARIANT = "multikey_panel_v1"
PANEL_VERSION = "contrastive_multikey_panel_v1"
PANEL_RNG_VERSION = "named_streams_v1"
PANEL_RNG_NAMESPACES = {
    "haystack": "niah_multikey_panel_v1/haystack",
    "target": "niah_multikey_panel_v1/target",
    "distractor": "niah_multikey_panel_v1/distractor",
    "placement": "niah_multikey_panel_v1/placement",
}
KEY_SUFFIX_CONTRACT = (
    "token_ids(key)==common_prefix_token_ids+[unique_suffix_token_id]"
)
PROMPTS = tuple(range(700, 740))
HALF_1 = tuple(range(700, 720))
HALF_2 = tuple(range(720, 740))
QUERY_SLOTS = tuple(range(4))
TARGET_RANKS = (5, 17, 30, 42)
TARGET_DEPTHS = (0.15, 0.38, 0.62, 0.85)
N_CLUSTERS = 4
CLUSTER_SIZE = 12
QUERY_COUNT = 4
CLUSTER_STEMS = (
    "panel cluster alpha key",
    "panel cluster bravo key",
    "panel cluster charlie key",
    "panel cluster delta key",
)
N_KEYS = 48
N_VALUES = 4
N_HOPS = 4
BUDGET = 2.0
WINDOW = 32
MAXB = 8
GENERATION_LIMIT_VERSION = "difficulty_v1"
MAX_NEW_TOKENS = 24
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 0
EXPECTED_CORPUS_SHA = "0a26bc1e05a1eea8"
TASK_GENERATION_VERSION = "contrastive_multikey_panel_v1"
TARGET_PROVENANCE_VERSION = "queried_needle_v1"
CONTEXT_HASH_ALGORITHM = "sha256_utf8"
ALLOCATION_ID_ALGORITHM = "sha256_layer_uint8_v1"
FP_ALLOCATION_ID = hashlib.sha256(b"fp16").hexdigest()

SHA256_RE = re.compile(r"[0-9a-f]{64}")
REAL_CORPUS_RE = re.compile(r"[0-9a-f]{16,64}")
VALUE_RE = re.compile(r"[0-9]{7}")

REQUIRED_COLUMNS = (
    "model", "model_id", "ctx", "native_ctx", "task", "task_variant",
    "panel_version", "panel_rng_version", "panel_rng_namespaces",
    "prompt_idx", "query_idx", "target_cluster", "target_key",
    "target_value", "target_needle_rank", "target_needle_depth",
    "panel_cluster_size", "panel_query_count", "panel_context_hash",
    "allocation_id", "panel_distractor_keys", "panel_distractor_values",
    "arm", "B", "score", "hits", "n_expected", "first_ok", "n_keys",
    "n_values", "n_hops", "task_n_needles", "max_new_tokens",
    "reached_max_new", "gen_len", "bits_per_token", "n_prompt_tokens",
    "ctx_len", "n_question_tokens", "question_agnostic", "window",
    "observed_queries", "allocator_budget_rule", "maxb", "needle_depths",
    "corpus_sha", "corpus_doc", "corpus_offset", "corpus_spliced",
    "synthetic", "rot_seed", "norm_correct",
)

SIDECAR_FIELDS = (
    "parquet", "model", "model_id", "ctx", "native_ctx", "tasks", "arms",
    "task_config", "task_variant", "panel", "generation_limit_version",
    "generation_limits", "budgets", "n_prompts", "prompt_offset", "window",
    "allocator_budget_rule", "maxb", "rows", "rot_seed", "norm_correct",
    "attn_impl", "compress_from", "compress_at", "question_agnostic",
    "observation_queries",
    "corpus_sha", "baselines", "p2", "task_generation_version",
    "target_needle_provenance_version",
)


class PanelQualificationError(ValueError):
    """An input violates the frozen panel qualification contract."""


def _fail(message: str) -> None:
    raise PanelQualificationError(message)


def _same(expected, actual, description: str) -> None:
    if isinstance(actual, (np.integer, np.floating)):
        actual = actual.item()
    if expected != actual:
        _fail(f"{description}: expected {expected!r}, got {actual!r}")


def _only(frame: pd.DataFrame, column: str):
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        _fail(f"parquet mixes {column}: {values!r}")
    value = values[0]
    return value.item() if isinstance(value, (np.integer, np.floating)) else value


def _numeric(frame: pd.DataFrame, column: str, *, integer: bool = False) -> pd.Series:
    try:
        values = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as exc:
        _fail(f"{column} is not numeric: {exc}")
    raw = values.to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        _fail(f"{column} contains a non-finite value")
    if integer and not np.equal(raw, np.floor(raw)).all():
        _fail(f"{column} must contain integers")
    return values.astype(np.int64) if integer else values.astype(float)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        _fail("parquet is missing required columns: " + ", ".join(missing))


def _parse_json_list(raw, description: str, length: int) -> list:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        _fail(f"{description} is not valid JSON: {exc}")
    if not isinstance(value, list) or len(value) != length:
        _fail(f"{description} must be a JSON list of length {length}")
    return value


def _validate_query_provenance(frame: pd.DataFrame, label: str) -> None:
    """Validate immutable panel records and cross-arm identity."""
    query_fields = [
        "target_cluster", "target_key", "target_value", "target_needle_rank",
        "target_needle_depth", "panel_distractor_keys", "panel_distractor_values",
        "n_prompt_tokens", "n_question_tokens",
    ]
    per_query = frame.groupby(["prompt_idx", "query_idx"])[query_fields].nunique(
        dropna=False
    )
    if per_query.ne(1).any().any():
        _fail(f"{label} query provenance differs between FP and uniform")

    context_fields = [
        "panel_context_hash", "corpus_doc", "corpus_offset", "ctx_len",
        "needle_depths",
    ]
    per_context = frame.groupby("prompt_idx")[context_fields].nunique(dropna=False)
    if per_context.ne(1).any().any():
        _fail(f"{label} shared-context provenance differs across arms or queries")

    contexts = frame.drop_duplicates("prompt_idx")
    if contexts.panel_context_hash.nunique() != len(PROMPTS):
        _fail(f"{label} must contain 40 distinct panel context hashes")
    if contexts.corpus_doc.astype(str).str.len().eq(0).any():
        _fail(f"{label}.corpus_doc contains an empty identity")
    if contexts.corpus_doc.nunique() != len(PROMPTS):
        _fail(f"{label} must traverse 40 distinct real corpus documents")
    if (contexts.corpus_offset < 0).any():
        _fail(f"{label}.corpus_offset must be nonnegative")

    # Each arm must reuse one real allocation for all four questions.  The hash
    # is over the actual bit tensors, rather than a method label.
    allocation_counts = frame.groupby(["prompt_idx", "arm"]).allocation_id.nunique(
        dropna=False
    )
    if allocation_counts.ne(1).any():
        _fail(f"{label} allocation_id changes across queries")
    if not frame.allocation_id.astype(str).map(lambda x: bool(SHA256_RE.fullmatch(x))).all():
        _fail(f"{label}.allocation_id must be a lowercase SHA-256 hex digest")
    fp_ids = frame.loc[frame.arm.astype(str) == "fp", "allocation_id"]
    if not fp_ids.eq(FP_ALLOCATION_ID).all():
        _fail(f"{label} FP allocation_id is not the frozen fp16 sentinel hash")
    uniform_ids = frame.loc[frame.arm.astype(str) == "uniform", "allocation_id"]
    if uniform_ids.eq(FP_ALLOCATION_ID).any():
        _fail(f"{label} uniform allocation_id cannot equal the FP sentinel hash")

    for prompt, block in frame.groupby("prompt_idx", sort=True):
        one_arm = block[block.arm.astype(str) == "fp"].sort_values("query_idx")
        if set(one_arm.query_idx.astype(int)) != set(QUERY_SLOTS):
            _fail(f"{label} prompt {prompt} lacks four unique query slots")
        if set(one_arm.target_cluster.astype(int)) != set(range(N_CLUSTERS)):
            _fail(f"{label} prompt {prompt} must query every target cluster once")
        ranks = tuple(sorted(one_arm.target_needle_rank.astype(int)))
        if ranks != TARGET_RANKS:
            _fail(f"{label} prompt {prompt} target ranks are {ranks!r}, expected {TARGET_RANKS!r}")
        depths = tuple(sorted(float(x) for x in one_arm.target_needle_depth))
        if not np.allclose(depths, TARGET_DEPTHS, rtol=0.0, atol=1e-8):
            _fail(f"{label} prompt {prompt} does not contain the four fixed depths")
        if one_arm.target_key.astype(str).nunique() != QUERY_COUNT:
            _fail(f"{label} prompt {prompt} target keys are not unique")
        if one_arm.target_value.astype(str).nunique() != QUERY_COUNT:
            _fail(f"{label} prompt {prompt} target values are not unique")

        depth_by_cluster = {
            int(row.target_cluster): float(row.target_needle_depth)
            for row in one_arm.itertuples(index=False)
        }
        expected_depth_by_cluster = {
            cluster: TARGET_DEPTHS[(cluster + int(prompt)) % QUERY_COUNT]
            for cluster in range(N_CLUSTERS)
        }
        for cluster, expected in expected_depth_by_cluster.items():
            if not math.isclose(depth_by_cluster[cluster], expected, rel_tol=0.0, abs_tol=1e-8):
                _fail(
                    f"{label} prompt {prompt} violates the cluster-to-depth rotation "
                    f"for cluster {cluster}"
                )

        all_panel_keys = []
        all_panel_values = []
        for row in one_arm.itertuples(index=False):
            query_idx = int(row.query_idx)
            expected_rank = TARGET_RANKS[query_idx]
            expected_depth = TARGET_DEPTHS[query_idx]
            expected_cluster = (query_idx - int(prompt) % QUERY_COUNT) % QUERY_COUNT
            if (int(row.target_needle_rank) != expected_rank or
                    not math.isclose(float(row.target_needle_depth), expected_depth,
                                     rel_tol=0.0, abs_tol=1e-8)):
                _fail(f"{label} prompt {prompt} query {query_idx} rank/depth slot drift")
            if int(row.target_cluster) != expected_cluster:
                _fail(f"{label} prompt {prompt} query {query_idx} cluster rotation drift")
            target_value = str(row.target_value)
            if not VALUE_RE.fullmatch(target_value):
                _fail(f"{label} target values must be seven decimal digits")
            keys = [str(x) for x in _parse_json_list(
                row.panel_distractor_keys,
                f"{label}.panel_distractor_keys p{prompt} q{row.query_idx}",
                CLUSTER_SIZE - 1,
            )]
            values = [str(x) for x in _parse_json_list(
                row.panel_distractor_values,
                f"{label}.panel_distractor_values p{prompt} q{row.query_idx}",
                CLUSTER_SIZE - 1,
            )]
            target_key = str(row.target_key)
            if len(set(keys)) != CLUSTER_SIZE - 1 or target_key in keys:
                _fail(f"{label} distractor keys must be unique and exclude the target")
            if (len(set(values)) != CLUSTER_SIZE - 1 or target_value in values or
                    not all(VALUE_RE.fullmatch(value) for value in values)):
                _fail(f"{label} distractor values must be unique seven-digit values and exclude target")
            stem = CLUSTER_STEMS[int(row.target_cluster)]
            cluster_keys = [target_key, *keys]
            if not all(key.startswith(stem + " ") and len(key) > len(stem) + 1
                       for key in cluster_keys):
                _fail(f"{label} cluster keys violate the frozen literal-prefix contract")
            all_panel_keys.extend(cluster_keys)
            all_panel_values.extend([target_value, *values])
        if len(set(all_panel_keys)) != N_KEYS:
            _fail(f"{label} prompt {prompt} does not contain 48 unique registered keys")
        if len(set(all_panel_values)) != N_KEYS:
            _fail(f"{label} prompt {prompt} does not contain 48 unique registered values")

        depth_vector = _parse_json_list(
            one_arm.iloc[0].needle_depths,
            f"{label}.needle_depths p{prompt}",
            N_KEYS,
        )
        try:
            depth_vector = [float(x) for x in depth_vector]
        except (TypeError, ValueError) as exc:
            _fail(f"{label}.needle_depths contains a nonnumeric value: {exc}")
        if not all(math.isfinite(x) and .05 <= x <= .95 for x in depth_vector):
            _fail(f"{label}.needle_depths entries must be finite and in [.05,.95]")
        if any(a > b for a, b in zip(depth_vector, depth_vector[1:])):
            _fail(f"{label}.needle_depths must be nondecreasing")
        for row in one_arm.itertuples(index=False):
            rank = int(row.target_needle_rank)
            if not math.isclose(depth_vector[rank], float(row.target_needle_depth),
                                rel_tol=0.0, abs_tol=5e-5):
                _fail(f"{label} target depth does not match needle_depths[target rank]")


def validate_frame(frame: pd.DataFrame, *, label: str = "parquet") -> None:
    frame = frame.copy()
    _require_columns(frame, REQUIRED_COLUMNS)
    if len(frame) != 320:
        _fail(f"{label} has {len(frame)} rows; expected exactly 320")
    nulls = [column for column in REQUIRED_COLUMNS if frame[column].isna().any()]
    if nulls:
        _fail(f"{label} has null values in: {', '.join(nulls)}")

    for column in (
        "ctx", "native_ctx", "prompt_idx", "query_idx", "target_cluster",
        "target_needle_rank", "panel_cluster_size", "panel_query_count",
        "n_keys", "n_values", "n_hops", "task_n_needles", "max_new_tokens",
        "gen_len", "n_prompt_tokens", "ctx_len", "n_question_tokens", "window",
        "observed_queries", "maxb", "corpus_offset", "rot_seed", "hits",
        "n_expected",
    ):
        frame[column] = _numeric(frame, column, integer=True)
    for column in ("target_needle_depth", "B", "score", "first_ok", "bits_per_token"):
        frame[column] = _numeric(frame, column)

    singleton_expectations = {
        "model": MODEL, "model_id": MODEL_ID, "ctx": CTX,
        "native_ctx": NATIVE_CTX, "task": TASK, "task_variant": TASK_VARIANT,
        "panel_version": PANEL_VERSION, "panel_rng_version": PANEL_RNG_VERSION,
        "panel_rng_namespaces": json.dumps(
            PANEL_RNG_NAMESPACES, sort_keys=True, separators=(",", ":")
        ),
        "panel_cluster_size": CLUSTER_SIZE,
        "panel_query_count": QUERY_COUNT, "n_keys": N_KEYS,
        "n_values": N_VALUES, "n_hops": N_HOPS, "task_n_needles": N_KEYS,
        "max_new_tokens": MAX_NEW_TOKENS, "n_expected": 1,
        "question_agnostic": True, "window": WINDOW, "observed_queries": WINDOW,
        "allocator_budget_rule": "feasible", "maxb": MAXB,
        "corpus_sha": EXPECTED_CORPUS_SHA, "corpus_spliced": False,
        "synthetic": False, "rot_seed": 0, "norm_correct": True,
    }
    for column, expected in singleton_expectations.items():
        _same(expected, _only(frame, column), f"{label}.{column}")
    for column in (
        "reached_max_new", "question_agnostic", "corpus_spliced", "synthetic",
        "norm_correct",
    ):
        if not pd.api.types.is_bool_dtype(frame[column].dtype):
            _fail(f"{label}.{column} must have boolean dtype")

    corpus_sha = str(_only(frame, "corpus_sha"))
    if not REAL_CORPUS_RE.fullmatch(corpus_sha) or corpus_sha == "0" * len(corpus_sha):
        _fail(f"{label}.corpus_sha is not a real corpus identity")
    if set(frame.prompt_idx) != set(PROMPTS):
        _fail(f"{label} must contain exactly prompts 700..739")
    if set(frame.query_idx) != set(QUERY_SLOTS):
        _fail(f"{label} query_idx must be exactly 0..3")
    if set(frame.arm.astype(str)) != {"fp", "uniform"}:
        _fail(f"{label} arms must be exactly fp and uniform")
    keys = frame.groupby(["prompt_idx", "arm", "query_idx"], dropna=False).size()
    expected_keys = {
        (prompt, arm, query)
        for prompt in PROMPTS for arm in ("fp", "uniform") for query in QUERY_SLOTS
    }
    if set(keys.index.tolist()) != expected_keys or not keys.eq(1).all():
        _fail(f"{label} must have one row per prompt, arm, and query slot")

    fp = frame[frame.arm.astype(str) == "fp"]
    uniform = frame[frame.arm.astype(str) == "uniform"]
    if not np.equal(fp.B.to_numpy(float), 0.0).all():
        _fail(f"{label} FP rows must use B=0")
    if not np.equal(uniform.B.to_numpy(float), BUDGET).all():
        _fail(f"{label} uniform rows must use B=2")
    if not np.isclose(fp.bits_per_token, 16.0, rtol=0.0, atol=1e-7).all():
        _fail(f"{label} FP rows must record 16 bits/token")
    if not np.isclose(uniform.bits_per_token, BUDGET, rtol=0.0, atol=1e-7).all():
        _fail(f"{label} uniform rows must realize exactly B=2")
    if ((frame.score < 0.0) | (frame.score > 1.0)).any():
        _fail(f"{label}.score must be in [0,1]")
    if not frame.first_ok.isin([0.0, 1.0]).all():
        _fail(f"{label}.first_ok must be binary")
    if not frame.hits.isin([0, 1]).all() or not frame.n_expected.eq(1).all():
        _fail(f"{label} hits must be binary with n_expected=1")
    if not np.equal(frame.score.to_numpy(float), frame.hits.to_numpy(float)).all():
        _fail(f"{label}.score must equal hits for each one-answer query")
    if (frame.first_ok > frame.score).any():
        _fail(f"{label}.first_ok cannot exceed substring score")
    expected_cap = frame.gen_len.to_numpy(int) >= frame.max_new_tokens.to_numpy(int)
    if not np.array_equal(frame.reached_max_new.astype(bool), expected_cap):
        _fail(f"{label} violates reached_max_new == (gen_len >= max_new_tokens)")
    if ((frame.gen_len < 0) | (frame.gen_len > frame.max_new_tokens)).any():
        _fail(f"{label}.gen_len must be between zero and max_new_tokens")
    if ((frame.n_prompt_tokens <= 0) | (frame.n_prompt_tokens > CTX)).any():
        _fail(f"{label}.n_prompt_tokens must be in (0,ctx]")
    if ((frame.ctx_len <= 0) | (frame.n_question_tokens <= 0)).any():
        _fail(f"{label}.ctx_len and n_question_tokens must be positive")
    reconstructed_prompt = (
        frame.ctx_len + frame.observed_queries + frame.n_question_tokens
    )
    if not frame.n_prompt_tokens.eq(reconstructed_prompt).all():
        _fail(
            f"{label}.n_prompt_tokens must equal "
            "ctx_len + observed_queries + n_question_tokens"
        )
    if not frame.panel_context_hash.astype(str).map(
            lambda x: bool(SHA256_RE.fullmatch(x))).all():
        _fail(f"{label}.panel_context_hash must be a lowercase SHA-256 hex digest")

    _validate_query_provenance(frame, label)


def _read_sidecar(path: Path) -> dict:
    if not path.is_file():
        _fail(f"missing sidecar: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read sidecar {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"sidecar must contain a JSON object: {path}")
    missing = sorted(set(SIDECAR_FIELDS) - set(value))
    if missing:
        _fail(f"sidecar is missing: {', '.join(missing)}")
    return value


def validate_sidecar(parquet: Path, frame: pd.DataFrame, sidecar: Mapping) -> None:
    expected = {
        "parquet": parquet.name, "model": MODEL, "model_id": MODEL_ID,
        "ctx": CTX, "native_ctx": NATIVE_CTX, "tasks": [TASK],
        "arms": ["fp", "uniform"],
        "task_config": {"n_keys": N_KEYS, "n_values": N_VALUES, "n_hops": N_HOPS},
        "task_variant": TASK_VARIANT,
        "generation_limit_version": GENERATION_LIMIT_VERSION,
        "generation_limits": {TASK: MAX_NEW_TOKENS}, "budgets": [2],
        "n_prompts": len(PROMPTS), "prompt_offset": PROMPTS[0],
        "window": WINDOW, "allocator_budget_rule": "feasible", "maxb": MAXB,
        "rows": 320, "rot_seed": 0, "norm_correct": True,
        "attn_impl": "sieve_compress", "compress_from": "first_answer_token",
        "compress_at": ("context_end: context prefilled and scored alone "
                        "(window = its last W tokens), question prefilled "
                        "through the compressed cache"),
        "question_agnostic": True, "observation_queries": [WINDOW],
        "corpus_sha": EXPECTED_CORPUS_SHA,
        "task_generation_version": TASK_GENERATION_VERSION,
        "target_needle_provenance_version": TARGET_PROVENANCE_VERSION,
        "baselines": None,
    }
    for field, value in expected.items():
        _same(value, sidecar[field], f"sidecar.{field}")

    panel = sidecar["panel"]
    if not isinstance(panel, Mapping):
        _fail("sidecar.panel must be an object")
    panel_expected = {
        "version": PANEL_VERSION, "clusters": N_CLUSTERS,
        "cluster_size": CLUSTER_SIZE, "query_count": QUERY_COUNT,
        "target_ranks": list(TARGET_RANKS), "target_depths": list(TARGET_DEPTHS),
        "context_hash_algorithm": CONTEXT_HASH_ALGORITHM,
        "allocation_id_algorithm": ALLOCATION_ID_ALGORITHM,
        "expected_accuracy_rows": 320, "expected_policy_rows": 0,
        "fp_allocation_sentinel": FP_ALLOCATION_ID,
    }
    for field, value in panel_expected.items():
        if field not in panel:
            _fail(f"sidecar.panel is missing {field}")
        _same(value, panel[field], f"sidecar.panel.{field}")
    for field, value in {
        "rng_version": PANEL_RNG_VERSION,
        "rng_namespaces": PANEL_RNG_NAMESPACES,
        "key_suffix_contract": KEY_SUFFIX_CONTRACT,
    }.items():
        if field not in panel:
            _fail(f"sidecar.panel is missing {field}")
        _same(value, panel[field], f"sidecar.panel.{field}")

    p2 = sidecar["p2"]
    if not isinstance(p2, Mapping):
        _fail("sidecar.p2 must be an object")
    for field, value in {
        "enabled": False, "want": ["uniform"], "routers": [],
        "head_error": False, "routes": None, "routes_meta": None,
    }.items():
        if field not in p2:
            _fail(f"sidecar.p2 is missing {field}")
        _same(value, p2[field], f"sidecar.p2.{field}")


def load_artifact(path: os.PathLike[str] | str) -> pd.DataFrame:
    parquet = Path(path)
    if not parquet.is_file():
        _fail(f"missing parquet: {parquet}")
    if parquet.suffix != ".parquet":
        _fail(f"input must be a .parquet file: {parquet}")
    try:
        frame = pd.read_parquet(parquet)
    except (OSError, ValueError) as exc:
        _fail(f"cannot read parquet {parquet}: {exc}")
    validate_frame(frame, label=str(parquet))
    sidecar = _read_sidecar(parquet.with_suffix(".json"))
    validate_sidecar(parquet, frame, sidecar)
    return frame


def _paired_interval(context_differences: np.ndarray) -> tuple[float, float]:
    differences = np.asarray(context_differences, dtype=float)
    if differences.shape != (len(PROMPTS),):
        _fail(f"paired bootstrap expected {len(PROMPTS)} context differences")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.integers(0, len(differences),
                        size=(BOOTSTRAP_REPLICATES, len(differences)))
    means = differences[draw].mean(axis=1)
    return float(np.quantile(means, .05)), float(np.quantile(means, .95))


def analyze_frame(frame: pd.DataFrame) -> dict:
    validate_frame(frame)
    query = frame.pivot(index=["prompt_idx", "query_idx"], columns="arm",
                        values="first_ok").sort_index()
    prompt = query.groupby(level="prompt_idx").mean().loc[list(PROMPTS)]
    differences = (prompt.fp - prompt.uniform).to_numpy(float)
    delta_lo, delta_hi = _paired_interval(differences)
    fp_rows = frame[frame.arm.astype(str) == "fp"]
    incomplete_capped_fp = int(((fp_rows.first_ok < 1.0) &
                                fp_rows.reached_max_new.astype(bool)).sum())
    fp_slots = query.fp.groupby(level="query_idx").mean().reindex(QUERY_SLOTS)
    uniform_slots = query.uniform.groupby(level="query_idx").mean().reindex(QUERY_SLOTS)
    uniform_spread = float(uniform_slots.max() - uniform_slots.min())
    first = prompt.loc[list(HALF_1)]
    second = prompt.loc[list(HALF_2)]
    standard = frame.groupby("arm").score.mean()

    fp_micro = float(query.fp.mean())
    uniform_mean = float(prompt.uniform.mean())
    fp_gate = fp_micro >= .95 and bool((fp_slots >= .90).all())
    cap_gate = incomplete_capped_fp == 0
    uniform_gate = .50 <= uniform_mean <= .75
    half_gate = (.35 <= float(first.uniform.mean()) <= .85 and
                 .35 <= float(second.uniform.mean()) <= .85)
    spread_gate = uniform_spread <= .25 + 1e-12
    delta_gate = delta_lo > .05
    eligible = fp_gate and cap_gate and uniform_gate and half_gate and spread_gate and delta_gate
    result = {
        "task_variant": TASK_VARIANT,
        "n_contexts": len(PROMPTS), "n_queries": len(query),
        "fp_micro_first_ok": fp_micro,
        "uniform_prompt_mean_first_ok": uniform_mean,
        "fp_standard_score_micro": float(standard["fp"]),
        "uniform_standard_score_micro": float(standard["uniform"]),
        "uniform_700_719": float(first.uniform.mean()),
        "uniform_720_739": float(second.uniform.mean()),
        "uniform_slot_spread": uniform_spread,
        "delta_fp_minus_uniform": float(differences.mean()),
        "delta_lo90": delta_lo, "delta_hi90": delta_hi,
        "incomplete_capped_fp": incomplete_capped_fp,
        "gate_fp_micro_ge_095_and_slots_ge_090": bool(fp_gate),
        "gate_no_incomplete_capped_fp": bool(cap_gate),
        "gate_uniform_050_075": bool(uniform_gate),
        "gate_each_half_035_085": bool(half_gate),
        "gate_uniform_slot_spread_le_025": bool(spread_gate),
        "gate_delta_lo_gt_005": bool(delta_gate),
        "eligible": bool(eligible), "advance_policy_development": bool(eligible),
    }
    for slot in QUERY_SLOTS:
        result[f"fp_slot_{slot}"] = float(fp_slots.loc[slot])
        result[f"uniform_slot_{slot}"] = float(uniform_slots.loc[slot])
    return result


def analyze_artifact(path: os.PathLike[str] | str) -> tuple[pd.DataFrame, bool]:
    frame = load_artifact(path)
    summary = pd.DataFrame([analyze_frame(frame)])
    return summary, bool(summary.iloc[0].eligible)


def _fmt(value: float) -> str:
    return f"{float(value):.3f}"


def print_summary(summary: pd.DataFrame, eligible: bool) -> None:
    row = summary.iloc[0]
    print("# V3 contrastive multikey-panel qualification")
    print("# primary: first_ok; four queries aggregated within each of 40 contexts")
    print("# paired context bootstrap: 90%, 10000 replicates, seed 0")
    print(
        f"FP micro={_fmt(row.fp_micro_first_ok)}  "
        f"uniform prompt mean={_fmt(row.uniform_prompt_mean_first_ok)}  "
        f"halves={_fmt(row.uniform_700_719)}/{_fmt(row.uniform_720_739)}"
    )
    print(
        "FP slots=" + "/".join(_fmt(row[f"fp_slot_{i}"]) for i in QUERY_SLOTS) +
        "  uniform slots=" + "/".join(
            _fmt(row[f"uniform_slot_{i}"]) for i in QUERY_SLOTS) +
        f"  spread={_fmt(row.uniform_slot_spread)}"
    )
    print(
        f"FP-uniform={_fmt(row.delta_fp_minus_uniform)} "
        f"[{_fmt(row.delta_lo90)}, {_fmt(row.delta_hi90)}]  "
        f"incomplete-capped-FP={int(row.incomplete_capped_fp)}"
    )
    print(
        "gates: "
        f"FP={'pass' if row.gate_fp_micro_ge_095_and_slots_ge_090 else 'fail'}, "
        f"FP-cap={'pass' if row.gate_no_incomplete_capped_fp else 'fail'}, "
        f"uniform[.50,.75]={'pass' if row.gate_uniform_050_075 else 'fail'}, "
        f"halves[.35,.85]={'pass' if row.gate_each_half_035_085 else 'fail'}, "
        f"slot-spread<=.25={'pass' if row.gate_uniform_slot_spread_le_025 else 'fail'}, "
        f"delta-lo>.05={'pass' if row.gate_delta_lo_gt_005 else 'fail'}"
    )
    print("decision: " + ("advance_policy_development" if eligible else "stop_panel"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", help="the frozen panel qualification parquet")
    parser.add_argument("--csv", help="optional path for the one-row summary CSV")
    parser.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            if args.csv:
                _fail("--csv is unavailable with --validate-only")
            load_artifact(args.parquet)
            print("validated 1 panel qualification artifact")
            return 0
        summary, eligible = analyze_artifact(args.parquet)
        print_summary(summary, eligible)
        if args.csv:
            output = Path(args.csv)
            output.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(output, index=False)
            print(f"wrote {output}")
        return 0
    except (PanelQualificationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
