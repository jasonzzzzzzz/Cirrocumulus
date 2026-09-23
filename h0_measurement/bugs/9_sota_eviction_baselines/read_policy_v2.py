#!/usr/bin/env python3
"""Authenticate and analyze the preregistered V2-B expanded-policy run.

The reader is intentionally specific to the V2-B development experiment in
``plan.md``: Llama-3.1-8B at 32K, QA multikey, B=2, prompts 540--579, one FP
arm, and the ordered eight-policy superset.  It authenticates both parquets and
both sidecars before looking at outcomes, evaluates the nested prefixes of
sizes 3 through 8, and recomputes every selector after filtering a prefix.

Usage::

    .venv/bin/python read_policy_v2.py ACCURACY.parquet DIAGNOSTIC.parquet \
        --n-keys SELECTED_K --csv SUMMARY.csv --prompt-csv PROMPTS.csv \
        --lock-out policy_v2_lock.json

``--lock-out`` writes an immutable confirmation manifest only when the smallest
headroom-qualifying prefix has an advancing selector.  A valid experiment with
no advancing selector is still a successful read (exit status 0).  Contract or
provenance failures use exit status 2.  ``--validate-only`` authenticates an
artifact pair without reporting scores and is intended for job completion
guards.
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
N_VALUES = 4
N_HOPS = 4
PROMPTS = tuple(range(540, 580))
HALVES = (tuple(range(540, 560)), tuple(range(560, 580)))
BUDGET = 2.0
WINDOW = 32
MAXB = 8
MAX_NEW_TOKENS = 24
GENERATION_LIMIT_VERSION = "difficulty_v1"
TASK_GENERATION_VERSION = "ruler_pg19_v1"
TARGET_PROVENANCE_VERSION = "queried_needle_v1"
TRACE_RULE_VERSION = "fp_teacher_forced_v1"
TRACE_STEPS = 8
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 0

CANDIDATES = (
    "uniform", "evict", "interior", "interior_pool", "interior_cascade",
    "obcache_k", "obck_ada", "laprox",
)
ARMS = ("fp", *CANDIDATES)
PREFIX_SIZES = tuple(range(3, len(CANDIDATES) + 1))
BASELINE_SPECS = {
    "obcache_k": "obcache_k",
    "obck_ada": "obcache_k:alloc=ada@obck_ada",
    "laprox": "laprox",
}

_OBCACHE_DESCRIPTION = (
    "OBCache: OBD pruning error of V (Eq.4) / K (Eq.5) / joint (Eq.6) "
    "over the window, pooled"
)
_UNIFORM_DESCRIPTION = "uniform keep-count per KV head"
_ADA_DESCRIPTION = (
    "Ada-KV Alg.1 + safeguard: B_g = (1-alpha) f_g + alpha k "
    "(largest-remainder rounding)"
)
_LAPROX_DESCRIPTION = (
    "LaProx: ||A[:,i]||_2 * ||v_i W_O^h||_2 per query head, group mean, pooled"
)
_GLOBAL_DESCRIPTION = (
    "LaProx Alg.2: layer-normalised scores, top-K over every layer and head"
)
EXPECTED_BASELINES = {
    "obcache_k": {
        "preset": "obcache_k", "score": "obcache", "variant": "k",
        "obs": 16, "pool": "max", "pool_k": 7, "gqa": "sum",
        "alloc": "uniform",
        "describe": f"{_OBCACHE_DESCRIPTION}; {_UNIFORM_DESCRIPTION}",
    },
    "obck_ada": {
        "preset": "obcache_k", "score": "obcache", "variant": "k",
        "obs": 16, "pool": "max", "pool_k": 7, "gqa": "sum",
        "alloc": "ada", "alpha": 0.2,
        "describe": f"{_OBCACHE_DESCRIPTION}; {_ADA_DESCRIPTION}",
    },
    "laprox": {
        "preset": "laprox", "score": "laprox", "obs": 32,
        "pool": "avg", "pool_k": 7, "gqa": "mean", "alloc": "global",
        "norm": True,
        "describe": f"{_LAPROX_DESCRIPTION}; {_GLOBAL_DESCRIPTION}",
    },
}

ACCURACY_COLUMNS = (
    "model", "model_id", "ctx", "native_ctx", "task", "prompt_idx", "arm",
    "B", "score", "hits", "n_expected", "first_ok", "n_keys", "n_values",
    "n_hops", "task_n_needles", "max_new_tokens", "reached_max_new",
    "gen_len", "bits_per_token", "n_prompt_tokens", "ctx_len", "window",
    "observed_queries", "allocator_budget_rule", "maxb", "needle_depths",
    "target_needle_rank", "target_needle_depth", "corpus_doc", "corpus_offset",
    "corpus_spliced", "corpus_sha", "synthetic", "rot_seed", "norm_correct",
    "question_agnostic",
)
DIAGNOSTIC_COLUMNS = (
    "model", "model_id", "ctx", "native_ctx", "task", "prompt_idx", "B",
    "n_keys", "n_values", "n_hops", "candidate", "candidate_order",
    "trace_rule_version", "trace_steps_requested", "trace_len",
    "trace_token_hash", "vocab_size", "mean_kl", "max_kl", "fp_token_ce",
    "top1_agreement", "fp_argmax_verified", "selected_policy",
    "question_agnostic", "window", "maxb", "allocator_budget_rule",
    "target_needle_rank", "target_needle_depth", "corpus_doc", "corpus_offset",
    "corpus_spliced", "corpus_sha", "synthetic", "rot_seed", "norm_correct",
    "t_policy_trace",
)
ROW_SHARED = (
    "model", "model_id", "ctx", "native_ctx", "task", "prompt_idx", "B",
    "n_keys", "n_values", "n_hops", "question_agnostic", "window", "maxb",
    "allocator_budget_rule", "target_needle_rank", "target_needle_depth",
    "corpus_doc", "corpus_offset", "corpus_spliced", "corpus_sha", "synthetic",
    "rot_seed", "norm_correct",
)
REAL_CORPUS_RE = re.compile(r"[0-9a-f]{16,64}")
RAW_LOGIT_RE = re.compile(r"logits?", re.IGNORECASE)

SELECTORS = {
    "mean_kl": ("mean_kl", "min"),
    "fp_token_ce": ("fp_token_ce", "min"),
    "max_kl": ("max_kl", "min"),
    "top1_agreement": ("top1_agreement", "max"),
}
ALTERNATE_ORDER = ("fp_token_ce", "max_kl", "top1_agreement")


class PolicyV2ReaderError(ValueError):
    """An artifact does not satisfy the frozen V2-B contract."""


def _fail(message: str) -> None:
    raise PolicyV2ReaderError(message)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        _fail(f"{label} is missing required columns: {', '.join(missing)}")
    nulls = [column for column in required if frame[column].isna().any()]
    if nulls:
        _fail(f"{label} has null values in: {', '.join(nulls)}")


def _numeric(frame: pd.DataFrame, column: str, label: str, *, integer: bool = False,
             minimum: float | None = None) -> pd.Series:
    try:
        values = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as exc:
        _fail(f"{label}.{column} is not numeric: {exc}")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        _fail(f"{label}.{column} contains a non-finite value")
    if integer and not np.equal(array, np.floor(array)).all():
        _fail(f"{label}.{column} must contain integers")
    if minimum is not None and (array < minimum).any():
        _fail(f"{label}.{column} must be >= {minimum}")
    return values.astype(np.int64) if integer else values.astype(float)


def _only(frame: pd.DataFrame, column: str, label: str):
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        _fail(f"{label} mixes {column}: {values!r}")
    value = values[0]
    return value.item() if isinstance(value, (np.integer, np.floating)) else value


def _same(expected, actual, description: str) -> None:
    if isinstance(actual, (np.integer, np.floating)):
        actual = actual.item()
    if expected != actual:
        _fail(f"{description}: expected {expected!r}, got {actual!r}")


def _sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sidecar_path(parquet: os.PathLike[str] | str) -> Path:
    return Path(parquet).with_suffix(".json")


def _read_sidecar(path: Path, label: str) -> dict:
    if not path.is_file():
        _fail(f"missing {label} sidecar: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {label} sidecar {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} sidecar must contain a JSON object")
    return value


def _parse_needles(frame: pd.DataFrame, n_keys: int, label: str) -> None:
    parsed: list[tuple[float, ...]] = []
    for row in frame.itertuples(index=False):
        try:
            raw = json.loads(row.needle_depths)
        except (TypeError, json.JSONDecodeError) as exc:
            _fail(f"{label}.needle_depths is not valid JSON: {exc}")
        if not isinstance(raw, list) or len(raw) != n_keys:
            _fail(f"{label}.needle_depths must contain exactly n_keys entries")
        try:
            values = tuple(float(value) for value in raw)
        except (TypeError, ValueError) as exc:
            _fail(f"{label}.needle_depths contains a nonnumeric entry: {exc}")
        if (not all(math.isfinite(x) and .05 <= x <= .95 for x in values) or
                any(a > b for a, b in zip(values, values[1:]))):
            _fail(f"{label}.needle_depths must be sorted finite values in [.05, .95]")
        rank = int(row.target_needle_rank)
        if rank < 0 or rank >= n_keys:
            _fail(f"{label}.target_needle_rank must be in [0, n_keys)")
        if not math.isclose(values[rank], float(row.target_needle_depth),
                            rel_tol=0.0, abs_tol=5e-5):
            _fail(f"{label}.target_needle_depth does not match needle_depths[rank]")
        parsed.append(values)
    audit = pd.DataFrame({"prompt_idx": frame.prompt_idx, "depths": parsed})
    if audit.groupby("prompt_idx").depths.nunique(dropna=False).ne(1).any():
        _fail(f"{label}.needle_depths differs across arms for a prompt")


def validate_frames(accuracy: pd.DataFrame, diagnostic: pd.DataFrame,
                    n_keys: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Strictly validate the two in-memory frames and return their exact join."""
    if isinstance(n_keys, bool) or not isinstance(n_keys, int) or n_keys < 1:
        _fail("n_keys must be a positive integer")
    accuracy, diagnostic = accuracy.copy(), diagnostic.copy()
    _require_columns(accuracy, ACCURACY_COLUMNS, "accuracy parquet")
    _require_columns(diagnostic, DIAGNOSTIC_COLUMNS, "diagnostic parquet")
    if len(accuracy) != 360:
        _fail(f"accuracy parquet has {len(accuracy)} rows; expected exactly 360")
    if len(diagnostic) != 320:
        _fail(f"diagnostic parquet has {len(diagnostic)} rows; expected exactly 320")
    forbidden = sorted(column for column in diagnostic.columns if RAW_LOGIT_RE.search(column))
    if forbidden:
        _fail("diagnostic parquet contains forbidden raw-logit column(s): " +
              ", ".join(forbidden))
    reserved = sorted({"score", "arm", "pred"}.intersection(diagnostic.columns))
    if reserved:
        _fail("diagnostic parquet duplicates independently decoded fields: " +
              ", ".join(reserved))

    int_accuracy = (
        "ctx", "native_ctx", "prompt_idx", "hits", "n_expected", "n_keys",
        "n_values", "n_hops", "task_n_needles", "max_new_tokens", "gen_len",
        "n_prompt_tokens", "ctx_len", "window", "observed_queries", "maxb",
        "target_needle_rank", "corpus_offset", "rot_seed",
    )
    for column in int_accuracy:
        accuracy[column] = _numeric(accuracy, column, "accuracy parquet", integer=True,
                                    minimum=0)
    for column in ("B", "score", "first_ok", "bits_per_token", "target_needle_depth"):
        accuracy[column] = _numeric(accuracy, column, "accuracy parquet")
    int_diagnostic = (
        "ctx", "native_ctx", "prompt_idx", "n_keys", "n_values", "n_hops",
        "candidate_order", "trace_steps_requested", "trace_len", "vocab_size",
        "window", "maxb", "target_needle_rank", "corpus_offset", "rot_seed",
    )
    for column in int_diagnostic:
        diagnostic[column] = _numeric(diagnostic, column, "diagnostic parquet",
                                      integer=True, minimum=0)
    for column in ("B", "mean_kl", "max_kl", "fp_token_ce", "top1_agreement",
                   "target_needle_depth", "t_policy_trace"):
        diagnostic[column] = _numeric(diagnostic, column, "diagnostic parquet")

    expected_singletons = {
        "model": MODEL, "model_id": MODEL_ID, "ctx": CTX,
        "native_ctx": NATIVE_CTX, "task": TASK, "n_keys": n_keys,
        "n_values": N_VALUES, "n_hops": N_HOPS, "max_new_tokens": MAX_NEW_TOKENS,
        "n_expected": 1, "task_n_needles": n_keys, "question_agnostic": True,
        "window": WINDOW, "observed_queries": WINDOW,
        "allocator_budget_rule": "feasible", "maxb": MAXB,
        "corpus_spliced": False, "synthetic": False, "rot_seed": 0,
        "norm_correct": True,
    }
    for field, expected in expected_singletons.items():
        _same(expected, _only(accuracy, field, "accuracy parquet"),
              f"accuracy parquet.{field}")
    diagnostic_singletons = {key: value for key, value in expected_singletons.items()
                             if key not in {"max_new_tokens", "n_expected",
                                            "task_n_needles", "observed_queries"}}
    diagnostic_singletons["trace_rule_version"] = TRACE_RULE_VERSION
    diagnostic_singletons["trace_steps_requested"] = TRACE_STEPS
    for field, expected in diagnostic_singletons.items():
        _same(expected, _only(diagnostic, field, "diagnostic parquet"),
              f"diagnostic parquet.{field}")

    corpus_sha = str(_only(accuracy, "corpus_sha", "accuracy parquet"))
    if (not REAL_CORPUS_RE.fullmatch(corpus_sha) or
            corpus_sha == "0" * len(corpus_sha)):
        _fail(f"accuracy parquet has no real corpus identity: {corpus_sha!r}")
    _same(corpus_sha, _only(diagnostic, "corpus_sha", "diagnostic parquet"),
          "accuracy/diagnostic corpus_sha")
    if set(accuracy.prompt_idx.tolist()) != set(PROMPTS):
        _fail("accuracy parquet must contain exactly prompts 540..579")
    if set(diagnostic.prompt_idx.tolist()) != set(PROMPTS):
        _fail("diagnostic parquet must contain exactly prompts 540..579")

    accuracy["arm"] = accuracy.arm.astype(str)
    diagnostic["candidate"] = diagnostic.candidate.astype(str)
    if set(accuracy.arm) != set(ARMS):
        _fail(f"accuracy arms must be exactly {list(ARMS)!r}")
    if set(diagnostic.candidate) != set(CANDIDATES):
        _fail(f"diagnostic candidates must be exactly {list(CANDIDATES)!r}")
    acc_counts = accuracy.groupby(["prompt_idx", "arm"], dropna=False).size()
    expected_acc = {(prompt, arm) for prompt in PROMPTS for arm in ARMS}
    if set(acc_counts.index.tolist()) != expected_acc or not acc_counts.eq(1).all():
        _fail("accuracy parquet must have one unique FP/candidate row per prompt")
    diag_counts = diagnostic.groupby(["prompt_idx", "candidate"], dropna=False).size()
    expected_diag = {(prompt, candidate) for prompt in PROMPTS for candidate in CANDIDATES}
    if set(diag_counts.index.tolist()) != expected_diag or not diag_counts.eq(1).all():
        _fail("diagnostic parquet must have one unique row per prompt/candidate")

    mapping_rows = diagnostic[["candidate", "candidate_order"]].drop_duplicates()
    if len(mapping_rows) != len(CANDIDATES):
        _fail("candidate_order is not one-to-one")
    mapping = dict(zip(mapping_rows.candidate, mapping_rows.candidate_order.astype(int)))
    expected_mapping = {candidate: order for order, candidate in enumerate(CANDIDATES)}
    if mapping != expected_mapping:
        _fail(f"candidate order is {mapping!r}; expected {expected_mapping!r}")

    fp = accuracy[accuracy.arm == "fp"]
    candidates = accuracy[accuracy.arm != "fp"]
    if not fp.B.eq(0.0).all() or not candidates.B.eq(BUDGET).all():
        _fail("FP rows must use B=0 and every candidate row must use B=2")
    if not diagnostic.B.eq(BUDGET).all():
        _fail("every diagnostic row must use B=2")
    if ((accuracy.score < 0) | (accuracy.score > 1)).any():
        _fail("accuracy score must be in [0, 1]")
    if not accuracy.hits.isin([0, 1]).all() or not accuracy.n_expected.eq(1).all():
        _fail("multikey hits must be binary with n_expected=1")
    if not np.equal(accuracy.score.to_numpy(float), accuracy.hits.to_numpy(float)).all():
        _fail("multikey score must equal hits")
    if (not accuracy.first_ok.isin([0.0, 1.0]).all() or
            (accuracy.first_ok > accuracy.score).any()):
        _fail("first_ok must be binary and cannot exceed substring score")
    expected_cap = accuracy.gen_len.to_numpy(int) >= accuracy.max_new_tokens.to_numpy(int)
    if not np.array_equal(expected_cap, accuracy.reached_max_new.astype(bool).to_numpy()):
        _fail("reached_max_new must equal (gen_len >= max_new_tokens)")
    if ((accuracy.gen_len < 0) | (accuracy.gen_len > accuracy.max_new_tokens)).any():
        _fail("gen_len must be between zero and max_new_tokens")
    if not np.isclose(fp.bits_per_token.to_numpy(float), 16.0,
                      rtol=0.0, atol=1e-7).all():
        _fail("FP rows must record 16 bits/token")
    uniform_bits = accuracy.loc[accuracy.arm == "uniform", "bits_per_token"].to_numpy(float)
    if not np.isclose(uniform_bits, BUDGET, rtol=0.0, atol=1e-7).all():
        _fail("uniform rows must realize exactly 2 bits/token")
    lossy_bits = candidates.bits_per_token.to_numpy(float)
    if (lossy_bits <= 0).any() or (lossy_bits > BUDGET + 1e-7).any():
        _fail("candidate rows must spend in (0, B] bits/token")

    for column in ("mean_kl", "max_kl", "fp_token_ce", "t_policy_trace"):
        if (diagnostic[column] < -1e-6).any():
            _fail(f"diagnostic {column} must be nonnegative")
    if (diagnostic.max_kl + 1e-6 < diagnostic.mean_kl).any():
        _fail("diagnostic max_kl is smaller than mean_kl")
    if ((diagnostic.top1_agreement < 0) | (diagnostic.top1_agreement > 1)).any():
        _fail("diagnostic top1_agreement must be in [0, 1]")
    if ((diagnostic.trace_len < 1) |
            (diagnostic.trace_len > diagnostic.trace_steps_requested)).any():
        _fail("trace_len must be between 1 and trace_steps_requested")
    if not diagnostic.trace_token_hash.astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        _fail("trace_token_hash must be a lowercase SHA-256 digest")
    if not diagnostic.fp_argmax_verified.eq(True).all():
        _fail("every diagnostic replay must verify the FP argmax")
    for prompt, group in diagnostic.groupby("prompt_idx", sort=False):
        for field in ("trace_len", "trace_token_hash", "vocab_size"):
            if group[field].nunique(dropna=False) != 1:
                _fail(f"prompt {prompt} mixes shared FP trace field {field}")
        selected = min(CANDIDATES,
                       key=lambda candidate: (float(group.loc[
                           group.candidate == candidate, "mean_kl"].iloc[0]),
                                              mapping[candidate]))
        if group.selected_policy.nunique(dropna=False) != 1 or str(
                group.selected_policy.iloc[0]) != selected:
            _fail(f"prompt {prompt} has an invalid full-set selected_policy")

    # Every policy for one prompt must use the same generated task and source.
    same_prompt_fields = (
        "target_needle_rank", "target_needle_depth", "needle_depths", "corpus_doc",
        "corpus_offset", "corpus_spliced", "corpus_sha", "n_prompt_tokens", "ctx_len",
    )
    for field in same_prompt_fields:
        if accuracy.groupby("prompt_idx")[field].nunique(dropna=False).ne(1).any():
            _fail(f"accuracy parquet mixes {field} across arms for a prompt")
    diag_prompt_fields = (
        "target_needle_rank", "target_needle_depth", "corpus_doc", "corpus_offset",
        "corpus_spliced", "corpus_sha",
    )
    for field in diag_prompt_fields:
        if diagnostic.groupby("prompt_idx")[field].nunique(dropna=False).ne(1).any():
            _fail(f"diagnostic parquet mixes {field} across candidates for a prompt")
    if (accuracy.corpus_doc.astype(str).str.len() == 0).any() or (accuracy.corpus_offset < 0).any():
        _fail("accuracy parquet has invalid corpus document/offset provenance")
    fp_sources = fp[["prompt_idx", "corpus_doc"]].drop_duplicates()
    if fp_sources.corpus_doc.nunique() != len(PROMPTS):
        _fail("the development block must cover 40 distinct corpus documents")
    if ((accuracy.target_needle_rank < 0) | (accuracy.target_needle_rank >= n_keys)).any():
        _fail("accuracy target_needle_rank is outside [0, n_keys)")
    if ((diagnostic.target_needle_rank < 0) | (diagnostic.target_needle_rank >= n_keys)).any():
        _fail("diagnostic target_needle_rank is outside [0, n_keys)")
    if ((accuracy.target_needle_depth < .05) | (accuracy.target_needle_depth > .95)).any():
        _fail("accuracy target_needle_depth is outside [.05, .95]")
    _parse_needles(accuracy, n_keys, "accuracy parquet")

    acc_candidates = candidates.rename(columns={"arm": "candidate"})
    join_keys = ["prompt_idx", "candidate"]
    acc_columns = [*join_keys, "score", "first_ok", *ROW_SHARED]
    # prompt_idx and B occur in ROW_SHARED; retain each column once.
    acc_columns = list(dict.fromkeys(acc_columns))
    joined = diagnostic.merge(acc_candidates[acc_columns], on=join_keys, how="outer",
                              validate="one_to_one", indicator=True,
                              suffixes=("", "_accuracy"))
    if not joined._merge.eq("both").all():
        _fail("accuracy and diagnostic candidate rows do not form an exact one-to-one join")
    joined = joined.drop(columns="_merge")
    for field in ROW_SHARED:
        if field in join_keys:
            continue
        other = f"{field}_accuracy"
        left, right = joined[field], joined[other]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            same = np.isclose(left.to_numpy(float), right.to_numpy(float),
                              rtol=0.0, atol=0.0)
        else:
            same = left.astype(str).to_numpy() == right.astype(str).to_numpy()
        if not same.all():
            _fail(f"accuracy and diagnostic provenance disagree for {field}")
        joined = joined.drop(columns=other)
    return joined, mapping


def _require_sidecar_fields(sidecar: Mapping, fields: Iterable[str], label: str) -> None:
    missing = sorted(set(fields) - set(sidecar))
    if missing:
        _fail(f"{label} sidecar is missing: {', '.join(missing)}")


def validate_sidecars(accuracy_path: Path, diagnostic_path: Path,
                      accuracy: pd.DataFrame, diagnostic: pd.DataFrame,
                      n_keys: int) -> tuple[dict, dict]:
    """Authenticate file identity, versions, run config, and baseline configs."""
    accuracy_side = _read_sidecar(_sidecar_path(accuracy_path), "accuracy")
    diag_side = _read_sidecar(_sidecar_path(diagnostic_path), "diagnostic")
    accuracy_required = (
        "parquet", "model", "model_id", "ctx", "native_ctx", "tasks", "arms",
        "task_config", "task_generation_version", "target_needle_provenance_version",
        "generation_limit_version", "generation_limits", "budgets", "n_prompts",
        "prompt_offset", "window", "observation_queries", "allocator_budget_rule",
        "maxb", "rows", "rot_seed", "norm_correct", "attn_impl", "compress_from",
        "question_agnostic", "baselines", "corpus_sha", "p2",
    )
    diagnostic_required = (
        "parquet", "model", "model_id", "ctx", "native_ctx", "tasks", "task_config",
        "task_generation_version", "target_needle_provenance_version",
        "generation_limit_version", "generation_limits", "budgets", "n_prompts",
        "prompt_offset", "window", "maxb", "question_agnostic",
        "allocator_budget_rule", "candidates", "trace_rule_version",
        "trace_steps_requested", "teacher", "metric_dtype", "vocabulary", "rows",
        "expected_rows", "corpus_sha", "rot_seed", "norm_correct",
        "accuracy_parquet", "accuracy_sidecar", "accuracy_sha256", "accuracy_rows",
        "no_raw_logits",
    )
    _require_sidecar_fields(accuracy_side, accuracy_required, "accuracy")
    _require_sidecar_fields(diag_side, diagnostic_required, "diagnostic")
    corpus_sha = str(_only(accuracy, "corpus_sha", "accuracy parquet"))
    expected_accuracy = {
        "parquet": accuracy_path.name, "model": MODEL, "model_id": MODEL_ID,
        "ctx": CTX, "native_ctx": NATIVE_CTX, "tasks": [TASK], "arms": list(ARMS),
        "task_config": {"n_keys": n_keys, "n_values": N_VALUES, "n_hops": N_HOPS},
        "task_generation_version": TASK_GENERATION_VERSION,
        "target_needle_provenance_version": TARGET_PROVENANCE_VERSION,
        "generation_limit_version": GENERATION_LIMIT_VERSION,
        "generation_limits": {TASK: MAX_NEW_TOKENS}, "budgets": [2],
        "n_prompts": len(PROMPTS), "prompt_offset": PROMPTS[0], "window": WINDOW,
        "observation_queries": [WINDOW], "allocator_budget_rule": "feasible",
        "maxb": MAXB, "rows": 360, "rot_seed": 0, "norm_correct": True,
        "attn_impl": "sieve_compress", "compress_from": "first_answer_token",
        "question_agnostic": True, "baselines": EXPECTED_BASELINES,
        "corpus_sha": corpus_sha,
    }
    for field, value in expected_accuracy.items():
        _same(value, accuracy_side[field], f"accuracy sidecar.{field}")
    p2 = accuracy_side["p2"]
    if not isinstance(p2, Mapping):
        _fail("accuracy sidecar.p2 must be an object")
    expected_p2 = {
        "enabled": True,
        "want": ["evict", "interior", "interior_cascade", "interior_pool", "uniform"],
        "routers": [], "head_error": False, "routes": None, "routes_meta": None,
    }
    for field, value in expected_p2.items():
        if field not in p2:
            _fail(f"accuracy sidecar.p2 is missing {field}")
        _same(value, p2[field], f"accuracy sidecar.p2.{field}")

    expected_diag = {
        "parquet": diagnostic_path.name, "model": MODEL, "model_id": MODEL_ID,
        "ctx": CTX, "native_ctx": NATIVE_CTX, "tasks": [TASK],
        "task_config": {"n_keys": n_keys, "n_values": N_VALUES, "n_hops": N_HOPS},
        "task_generation_version": TASK_GENERATION_VERSION,
        "target_needle_provenance_version": TARGET_PROVENANCE_VERSION,
        "generation_limit_version": GENERATION_LIMIT_VERSION,
        "generation_limits": {TASK: MAX_NEW_TOKENS}, "budgets": [2],
        "n_prompts": len(PROMPTS), "prompt_offset": PROMPTS[0], "window": WINDOW,
        "maxb": MAXB, "question_agnostic": True,
        "allocator_budget_rule": "feasible", "candidates": list(CANDIDATES),
        "trace_rule_version": TRACE_RULE_VERSION, "trace_steps_requested": TRACE_STEPS,
        "teacher": "fp_greedy", "metric_dtype": "float32", "vocabulary": "full",
        "rows": 320, "expected_rows": 320, "corpus_sha": corpus_sha,
        "rot_seed": 0, "norm_correct": True, "accuracy_parquet": accuracy_path.name,
        "accuracy_sidecar": _sidecar_path(accuracy_path).name,
        "accuracy_sha256": _sha256(accuracy_path), "accuracy_rows": 360,
        "no_raw_logits": True,
    }
    for field, value in expected_diag.items():
        _same(value, diag_side[field], f"diagnostic sidecar.{field}")
    return accuracy_side, diag_side


def load_pair(accuracy_path: os.PathLike[str] | str,
              diagnostic_path: os.PathLike[str] | str,
              n_keys: int) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Load and strictly authenticate one V2-B artifact pair."""
    accuracy_path, diagnostic_path = Path(accuracy_path), Path(diagnostic_path)
    for path, label in ((accuracy_path, "accuracy"), (diagnostic_path, "diagnostic")):
        if not path.is_file():
            _fail(f"missing {label} parquet: {path}")
        if path.suffix != ".parquet":
            _fail(f"{label} input must be a .parquet file: {path}")
    try:
        accuracy = pd.read_parquet(accuracy_path)
        diagnostic = pd.read_parquet(diagnostic_path)
    except (OSError, ValueError) as exc:
        _fail(f"cannot read artifact pair: {exc}")
    validate_frames(accuracy, diagnostic, n_keys)
    accuracy_side, diag_side = validate_sidecars(
        accuracy_path, diagnostic_path, accuracy, diagnostic, n_keys)
    return accuracy, diagnostic, accuracy_side, diag_side


def _selector(group: pd.DataFrame, prefix: Sequence[str], metric: str,
              mapping: Mapping[str, int]) -> str:
    column, direction = SELECTORS[metric]
    values = {candidate: float(group.loc[group.candidate == candidate, column].iloc[0])
              for candidate in prefix}
    if direction == "min":
        return min(prefix, key=lambda candidate: (values[candidate], mapping[candidate]))
    return min(prefix, key=lambda candidate: (-values[candidate], mapping[candidate]))


def _interval(values: np.ndarray, draws: np.ndarray) -> tuple[float, float]:
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, .05)), float(np.quantile(means, .95))


def _ratio_interval(numerator: np.ndarray, denominator: np.ndarray,
                    draws: np.ndarray) -> tuple[float, float]:
    num = numerator[draws].mean(axis=1)
    den = denominator[draws].mean(axis=1)
    ratios = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    finite = ratios[np.isfinite(ratios)]
    if not len(finite):
        return math.nan, math.nan
    return float(np.quantile(finite, .05)), float(np.quantile(finite, .95))


def _prefix_rows(accuracy: pd.DataFrame, diagnostic: pd.DataFrame,
                 prefix: Sequence[str], mapping: Mapping[str, int]) -> pd.DataFrame:
    scores = (accuracy[accuracy.arm.isin(prefix)]
              .pivot(index="prompt_idx", columns="arm", values="score")
              .loc[list(PROMPTS), list(prefix)])
    means = scores.mean(axis=0)
    best_fixed = min(prefix, key=lambda candidate: (-float(means[candidate]),
                                                     mapping[candidate]))
    rows: list[dict] = []
    for prompt in PROMPTS:
        group = diagnostic[(diagnostic.prompt_idx == prompt) &
                           diagnostic.candidate.isin(prefix)]
        prompt_scores = scores.loc[prompt]
        envelope = float(prompt_scores.max())
        ties = tuple(candidate for candidate in prefix
                     if float(prompt_scores[candidate]) == envelope)
        row = {
            "prefix_size": len(prefix), "prompt_idx": prompt,
            "prefix": "|".join(prefix), "best_fixed_dev": best_fixed,
            "uniform_score": float(prompt_scores["uniform"]),
            "best_fixed_score": float(prompt_scores[best_fixed]),
            "envelope_score": envelope, "endtask_ties": "|".join(ties),
            "H": envelope - float(prompt_scores["uniform"]),
            "H_F": envelope - float(prompt_scores[best_fixed]),
        }
        for metric in SELECTORS:
            selected = _selector(group, prefix, metric, mapping)
            score = float(prompt_scores[selected])
            metric_column, _ = SELECTORS[metric]
            selected_value = float(group.loc[group.candidate == selected,
                                             metric_column].iloc[0])
            selector_ties = sum(
                float(group.loc[group.candidate == candidate, metric_column].iloc[0])
                == selected_value for candidate in prefix)
            row[f"{metric}_candidate"] = selected
            row[f"{metric}_score"] = score
            row[f"{metric}_G"] = score - float(prompt_scores["uniform"])
            row[f"{metric}_G_F"] = score - float(prompt_scores[best_fixed])
            row[f"{metric}_tie_hit"] = selected in ties
            row[f"{metric}_selector_tie_count"] = selector_ties
            row[f"{metric}_rescued"] = bool(
                float(prompt_scores["uniform"]) < 1.0 and
                score > float(prompt_scores["uniform"]))
            row[f"{metric}_harmed"] = bool(
                float(prompt_scores["uniform"]) >= 1.0 and
                score < float(prompt_scores["uniform"]))
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_frames(accuracy: pd.DataFrame, diagnostic: pd.DataFrame, n_keys: int,
                   n_boot: int = BOOTSTRAP_REPLICATES,
                   seed: int = BOOTSTRAP_SEED) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Validate frames, analyze all prefixes, and return decision metadata."""
    _, mapping = validate_frames(accuracy, diagnostic, n_keys)
    if n_boot < 1:
        _fail("bootstrap replicate count must be positive")
    fp = accuracy[accuracy.arm == "fp"].set_index("prompt_idx").loc[list(PROMPTS)]
    uniform = accuracy[accuracy.arm == "uniform"].set_index("prompt_idx").loc[list(PROMPTS)]
    fp_mean = float(fp.score.mean())
    uniform_mean = float(uniform.score.mean())
    incomplete_capped_fp = int(((fp.score < 1.0) & fp.reached_max_new.astype(bool)).sum())
    operating_gate = bool(fp_mean >= .95 and incomplete_capped_fp == 0 and
                          .50 <= uniform_mean <= .80)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(PROMPTS), size=(n_boot, len(PROMPTS)))
    all_prompts: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for size in PREFIX_SIZES:
        prefix = CANDIDATES[:size]
        prompts = _prefix_rows(accuracy, diagnostic, prefix, mapping)
        all_prompts.append(prompts)
        h = prompts.H.to_numpy(float)
        hf = prompts.H_F.to_numpy(float)
        h_lo, h_hi = _interval(h, draws)
        hf_lo, hf_hi = _interval(hf, draws)
        row: dict = {
            "n_keys": n_keys, "prefix_size": size, "prefix": "|".join(prefix),
            "n_prompts": len(PROMPTS), "fp_mean": fp_mean,
            "uniform_mean": uniform_mean,
            "fp_first_ok": float(fp.first_ok.mean()),
            "uniform_first_ok": float(uniform.first_ok.mean()),
            "incomplete_capped_fp": incomplete_capped_fp,
            "gate_operating_point": operating_gate,
            "best_fixed_dev": str(prompts.best_fixed_dev.iloc[0]),
            "best_fixed_mean": float(prompts.best_fixed_score.mean()),
            "envelope_mean": float(prompts.envelope_score.mean()),
            "H": float(h.mean()), "H_lo90": h_lo, "H_hi90": h_hi,
            "H_F": float(hf.mean()), "H_F_lo90": hf_lo, "H_F_hi90": hf_hi,
            "H_F_half1": float(prompts.iloc[:20].H_F.mean()),
            "H_F_half2": float(prompts.iloc[20:].H_F.mean()),
        }
        row["gate_headroom"] = bool(
            row["H_F"] >= .10 and row["H_F_lo90"] > 0 and
            row["H_F_half1"] >= .05 and row["H_F_half2"] >= .05)
        for metric in SELECTORS:
            g = prompts[f"{metric}_G"].to_numpy(float)
            gf = prompts[f"{metric}_G_F"].to_numpy(float)
            g_lo, g_hi = _interval(g, draws)
            gf_lo, gf_hi = _interval(gf, draws)
            regret = h - g
            regret_f = hf - gf
            regret_lo, regret_hi = _interval(regret, draws)
            regret_f_lo, regret_f_hi = _interval(regret_f, draws)
            captured_lo, captured_hi = _ratio_interval(g, h, draws)
            captured_f_lo, captured_f_hi = _ratio_interval(gf, hf, draws)
            G, GF = float(g.mean()), float(gf.mean())
            row[f"{metric}_selected_mean"] = float(prompts[f"{metric}_score"].mean())
            row[f"{metric}_G"] = G
            row[f"{metric}_G_lo90"] = g_lo
            row[f"{metric}_G_hi90"] = g_hi
            row[f"{metric}_G_F"] = GF
            row[f"{metric}_G_F_lo90"] = gf_lo
            row[f"{metric}_G_F_hi90"] = gf_hi
            row[f"{metric}_regret"] = float(regret.mean())
            row[f"{metric}_regret_lo90"] = regret_lo
            row[f"{metric}_regret_hi90"] = regret_hi
            row[f"{metric}_regret_F"] = float(regret_f.mean())
            row[f"{metric}_regret_F_lo90"] = regret_f_lo
            row[f"{metric}_regret_F_hi90"] = regret_f_hi
            row[f"{metric}_captured"] = G / row["H"] if row["H"] > 0 else math.nan
            row[f"{metric}_captured_lo90"] = captured_lo
            row[f"{metric}_captured_hi90"] = captured_hi
            row[f"{metric}_captured_F"] = GF / row["H_F"] if row["H_F"] > 0 else math.nan
            row[f"{metric}_captured_F_lo90"] = captured_f_lo
            row[f"{metric}_captured_F_hi90"] = captured_f_hi
            row[f"{metric}_G_F_half1"] = float(prompts.iloc[:20][f"{metric}_G_F"].mean())
            row[f"{metric}_G_F_half2"] = float(prompts.iloc[20:][f"{metric}_G_F"].mean())
            row[f"{metric}_tie_hits"] = int(prompts[f"{metric}_tie_hit"].sum())
            row[f"{metric}_rescues"] = int(prompts[f"{metric}_rescued"].sum())
            row[f"{metric}_harms"] = int(prompts[f"{metric}_harmed"].sum())
            row[f"{metric}_selector_tied_prompts"] = int(
                (prompts[f"{metric}_selector_tie_count"] > 1).sum())
            row[f"{metric}_selector_mean_tie_count"] = float(
                prompts[f"{metric}_selector_tie_count"].mean())
            for candidate in prefix:
                row[f"{metric}_selected_{candidate}"] = int(
                    (prompts[f"{metric}_candidate"] == candidate).sum())
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    prompt_audit = pd.concat(all_prompts, ignore_index=True)

    decision = {
        "operating_point_valid": operating_gate,
        "decision": "difficulty_non_replicating" if not operating_gate else None,
        "prefix_size": None, "prefix": None, "best_fixed_dev": None,
        "selector_metric": None, "selector_direction": None,
    }
    if operating_gate:
        full = summary.loc[summary.prefix_size == len(CANDIDATES)].iloc[0]
        qualifying = summary[summary.gate_headroom]
        # The full-set stop rule has precedence: a stronger fixed policy added
        # later can erase apparent routing value from an earlier prefix.
        if full.H_F <= .05:
            decision["decision"] = "stop_candidate_routing"
        elif qualifying.empty:
            decision["decision"] = "one_extension_needed"
        else:
            target = qualifying.sort_values("prefix_size").iloc[0]
            decision.update(prefix_size=int(target.prefix_size),
                            prefix=CANDIDATES[:int(target.prefix_size)],
                            best_fixed_dev=str(target.best_fixed_dev))
            mean_pass = bool(
                target.mean_kl_G_F >= .05 and target.mean_kl_captured_F >= .5 and
                target.mean_kl_G_F_half1 >= 0 and target.mean_kl_G_F_half2 >= 0)
            if mean_pass:
                decision.update(decision="advance_mean_kl", selector_metric="mean_kl",
                                selector_direction="min")
            else:
                passing = [metric for metric in ALTERNATE_ORDER
                           if target[f"{metric}_G_F"] >= .05 and
                           target[f"{metric}_captured_F"] >= .5]
                if passing:
                    # Largest G_F wins; preregistered order resolves an exact tie.
                    selected = min(passing, key=lambda metric: (
                        -float(target[f"{metric}_G_F"]), ALTERNATE_ORDER.index(metric)))
                    decision.update(decision=f"advance_{selected}",
                                    selector_metric=selected,
                                    selector_direction=SELECTORS[selected][1])
                else:
                    decision["decision"] = "no_proxy_advances"
    return summary, prompt_audit, decision


def build_lock_manifest(accuracy_path: os.PathLike[str] | str,
                        diagnostic_path: os.PathLike[str] | str,
                        n_keys: int, summary: pd.DataFrame,
                        decision: Mapping) -> dict | None:
    """Build the immutable V2-C lock, or return None for a non-advance."""
    if not str(decision.get("decision", "")).startswith("advance_"):
        return None
    size = int(decision["prefix_size"])
    row = summary.loc[summary.prefix_size == size].iloc[0]
    accuracy_path, diagnostic_path = Path(accuracy_path), Path(diagnostic_path)
    accuracy_side, diagnostic_side = (_sidecar_path(accuracy_path),
                                      _sidecar_path(diagnostic_path))
    metric = str(decision["selector_metric"])
    prefix = list(decision["prefix"])
    return {
        "schema_version": "r8_policy_v2_lock_v1",
        "reader": "read_policy_v2.py",
        "task": {
            "model": MODEL, "model_id": MODEL_ID, "ctx": CTX, "task": TASK,
            "n_keys": n_keys, "n_values": N_VALUES, "n_hops": N_HOPS,
            "budget": BUDGET, "question_agnostic": True, "window": WINDOW,
            "maxb": MAXB, "generation_limit_version": GENERATION_LIMIT_VERSION,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "ordered_candidate_prefix": prefix,
        "candidate_specs": {candidate: BASELINE_SPECS.get(candidate, candidate)
                            for candidate in prefix},
        "best_fixed_dev": str(decision["best_fixed_dev"]),
        "selector": {
            "metric": metric, "direction": str(decision["selector_direction"]),
            "stable_tie_order": prefix,
        },
        "trace": {
            "rule_version": TRACE_RULE_VERSION, "steps_requested": TRACE_STEPS,
            "teacher": "fp_greedy", "metric_dtype": "float32",
            "vocabulary": "full",
        },
        "development": {
            "prompt_block": [PROMPTS[0], PROMPTS[-1]], "n_prompts": len(PROMPTS),
            "accuracy": {
                "path": str(accuracy_path.resolve()), "sha256": _sha256(accuracy_path),
                "sidecar_path": str(accuracy_side.resolve()),
                "sidecar_sha256": _sha256(accuracy_side),
            },
            "diagnostic": {
                "path": str(diagnostic_path.resolve()), "sha256": _sha256(diagnostic_path),
                "sidecar_path": str(diagnostic_side.resolve()),
                "sidecar_sha256": _sha256(diagnostic_side),
            },
            "metrics": {
                "H_F": float(row.H_F), "H_F_lo90": float(row.H_F_lo90),
                "H_F_half1": float(row.H_F_half1), "H_F_half2": float(row.H_F_half2),
                "G_F": float(row[f"{metric}_G_F"]),
                "G_F_half1": float(row[f"{metric}_G_F_half1"]),
                "G_F_half2": float(row[f"{metric}_G_F_half2"]),
                "captured_F": float(row[f"{metric}_captured_F"]),
            },
        },
        "confirmation": {
            "prompt_block": [460, 499], "n_prompts": 40,
            "ordered_candidate_prefix": prefix,
            "best_fixed_dev": str(decision["best_fixed_dev"]),
            "selector_metric": metric,
            "selector_direction": str(decision["selector_direction"]),
            "expected_accuracy_rows": 40 * (size + 1),
            "expected_diagnostic_rows": 40 * size,
            "gates": {
                "fp_mean_min": .95, "no_incomplete_capped_fp": True,
                "H_F_min": .10, "G_F_min": .05, "captured_F_min": .5,
                "G_F_lo90_min": 0.0,
            },
        },
    }


def _fmt(value: float, signed: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):+.3f}" if signed else f"{float(value):.3f}"


def print_summary(summary: pd.DataFrame, decision: Mapping) -> None:
    print("# Expanded whole-policy development V2-B")
    print("# paired prompt bootstrap: 90%, 10000 replicates, seed 0")
    first = summary.iloc[0]
    print(f"operating point: FP={_fmt(first.fp_mean)}  uniform={_fmt(first.uniform_mean)}  "
          f"incomplete-capped-FP={int(first.incomplete_capped_fp)}  "
          f"valid={'yes' if first.gate_operating_point else 'no'}")
    for row in summary.itertuples(index=False):
        print(f"\nprefix {int(row.prefix_size)}: {row.prefix}")
        print(f"  fixed/envelope  {row.best_fixed_dev}={_fmt(row.best_fixed_mean)}  "
              f"envelope={_fmt(row.envelope_mean)}")
        print(f"  H_F             {_fmt(row.H_F, True)} "
              f"[{_fmt(row.H_F_lo90, True)}, {_fmt(row.H_F_hi90, True)}]  "
              f"halves={_fmt(row.H_F_half1, True)}/{_fmt(row.H_F_half2, True)}  "
              f"gate={'pass' if row.gate_headroom else 'fail'}")
        for metric in SELECTORS:
            print(f"  {metric:14s} G_F={_fmt(getattr(row, metric + '_G_F'), True)} "
                  f"[{_fmt(getattr(row, metric + '_G_F_lo90'), True)}, "
                  f"{_fmt(getattr(row, metric + '_G_F_hi90'), True)}]  "
                  f"regret_F={_fmt(getattr(row, metric + '_regret_F'), True)}  "
                  f"captured={_fmt(getattr(row, metric + '_captured_F'))}  "
                  f"halves={_fmt(getattr(row, metric + '_G_F_half1'), True)}/"
                  f"{_fmt(getattr(row, metric + '_G_F_half2'), True)}")
            print(f"    behavior: rescues={int(getattr(row, metric + '_rescues'))}  "
                  f"harms={int(getattr(row, metric + '_harms'))}  "
                  f"oracle-hit={int(getattr(row, metric + '_tie_hits'))}/40  "
                  f"selector-tied-prompts="
                  f"{int(getattr(row, metric + '_selector_tied_prompts'))}")
    print(f"\ndecision: {decision['decision']}")
    if decision.get("prefix"):
        print("frozen prefix: " + ",".join(decision["prefix"]))
        print(f"best_fixed_dev: {decision['best_fixed_dev']}")
    if decision.get("selector_metric"):
        print(f"selector: {decision['selector_metric']} ({decision['selector_direction']})")


def _write_lock(path: Path, manifest: dict | None) -> bool:
    if manifest is None:
        if path.exists():
            _fail(f"refusing to leave stale lock manifest for a non-advance: {path}")
        return False
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != payload:
            _fail(f"refusing to overwrite a different lock manifest: {path}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accuracy_parquet")
    parser.add_argument("diagnostic_parquet")
    parser.add_argument("--n-keys", type=int, required=True,
                        help="V2-A-selected multikey difficulty; authenticated exactly")
    parser.add_argument("--csv", help="optional nested-prefix summary CSV")
    parser.add_argument("--prompt-csv", help="optional per-prefix/prompt audit CSV")
    parser.add_argument("--lock-out", help="write an immutable V2-C lock on advance")
    parser.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        accuracy, diagnostic, _, _ = load_pair(
            args.accuracy_parquet, args.diagnostic_parquet, args.n_keys)
        if args.validate_only:
            if args.csv or args.prompt_csv or args.lock_out:
                _fail("output options are unavailable with --validate-only")
            print("validated V2-B expanded-policy artifact pair")
            return 0
        summary, prompts, decision = analyze_frames(
            accuracy, diagnostic, args.n_keys, n_boot=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED)
        print_summary(summary, decision)
        if args.csv:
            path = Path(args.csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(path, index=False)
            print(f"wrote {path}")
        if args.prompt_csv:
            path = Path(args.prompt_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            prompts.to_csv(path, index=False)
            print(f"wrote {path}")
        if args.lock_out:
            manifest = build_lock_manifest(args.accuracy_parquet,
                                           args.diagnostic_parquet,
                                           args.n_keys, summary, decision)
            if _write_lock(Path(args.lock_out), manifest):
                print(f"wrote immutable confirmation lock {args.lock_out}")
            else:
                print("no confirmation lock written: selector did not advance")
        return 0
    except (PolicyV2ReaderError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
