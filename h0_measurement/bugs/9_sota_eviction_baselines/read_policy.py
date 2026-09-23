#!/usr/bin/env python3
"""Audit and summarize R8 whole-policy diagnostics.

The diagnostic parquet contains one row for every
``(prompt, task, budget, candidate)``.  The accuracy parquet is deliberately
separate: its scores come from each policy's independent greedy decode, while
the diagnostic metrics use a shared FP teacher-forced prefix.  This reader
performs an exact one-to-one join before computing either oracle.

Usage::

    .venv/bin/python h0_measurement/bugs/9_sota_eviction_baselines/read_policy.py ACCURACY.parquet DIAGNOSTIC.parquet --csv summary.csv


The primary candidate set is fixed to uniform, evict, and interior.  The
end-task oracle retains all policies tied at the best independently decoded
score.  The policy-logit selector minimizes mean KL and resolves exact numeric
ties with the declared ``candidate_order``.
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


PRIMARY_CANDIDATES = ("uniform", "evict", "interior")
TASK_FIELDS = ("n_keys", "n_values", "n_hops")
BASE_KEYS = ("model", "ctx", "task", "prompt_idx", *TASK_FIELDS)
JOIN_KEYS = (*BASE_KEYS, "B", "candidate")
CELL_KEYS = ("model", "ctx", "task", *TASK_FIELDS, "B")

DIAGNOSTIC_COLUMNS = (
    "model", "model_id", "ctx", "native_ctx", "task", "prompt_idx", "B",
    *TASK_FIELDS, "candidate", "candidate_order", "trace_rule_version",
    "trace_steps_requested", "trace_len", "trace_token_hash", "vocab_size",
    "mean_kl", "max_kl", "fp_token_ce", "top1_agreement",
    "question_agnostic", "window", "maxb", "corpus_sha", "synthetic",
    "rot_seed", "norm_correct", "t_policy_trace",
)
ACCURACY_COLUMNS = (
    "model", "model_id", "ctx", "native_ctx", "task", "prompt_idx", "arm",
    "B", "score", *TASK_FIELDS, "question_agnostic", "window", "maxb",
    "corpus_sha", "synthetic", "rot_seed", "norm_correct",
)
SHARED_PROVENANCE = (
    "model_id", "native_ctx", "question_agnostic", "window", "maxb",
    "corpus_sha", "synthetic", "rot_seed", "norm_correct",
)
OPTIONAL_SHARED_PROVENANCE = ("allocator_budget_rule",)
TRACE_PROVENANCE = (
    "trace_rule_version", "trace_steps_requested", "vocab_size",
)
PROMPT_TRACE_PROVENANCE = ("trace_len", "trace_token_hash")
RAW_LOGIT_COLUMN = re.compile(r"logit", re.IGNORECASE)


class PolicyReaderError(ValueError):
    """The two artifacts do not satisfy the policy-diagnostic contract."""


def _fail(message: str) -> None:
    raise PolicyReaderError(message)


def _require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        _fail(f"{label} is missing required columns: {', '.join(missing)}")


def _require_nonnull(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    bad = [column for column in columns if df[column].isna().any()]
    if bad:
        _fail(f"{label} has null values in: {', '.join(bad)}")


def _as_integer(df: pd.DataFrame, column: str, label: str, minimum: int = 0) -> None:
    try:
        value = pd.to_numeric(df[column], errors="raise")
    except (TypeError, ValueError) as exc:
        _fail(f"{label}.{column} is not numeric: {exc}")
    if not np.isfinite(value.to_numpy(dtype=float)).all():
        _fail(f"{label}.{column} contains a non-finite value")
    if (value % 1 != 0).any() or (value < minimum).any():
        _fail(f"{label}.{column} must contain integers >= {minimum}")
    df[column] = value.astype(np.int64)


def _as_finite_float(df: pd.DataFrame, column: str, label: str) -> None:
    try:
        value = pd.to_numeric(df[column], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        _fail(f"{label}.{column} is not numeric: {exc}")
    if not np.isfinite(value.to_numpy()).all():
        _fail(f"{label}.{column} contains a non-finite value")
    df[column] = value


def _one_value(df: pd.DataFrame, columns: Sequence[str], by: Sequence[str], label: str) -> None:
    """Require each provenance column to have one value in every group."""
    for column in columns:
        counts = df.groupby(list(by), dropna=False, sort=False)[column].nunique(dropna=False)
        if len(counts) and int(counts.max()) != 1:
            bad = counts[counts != 1].index[0]
            _fail(f"{label} mixes {column} within {dict(zip(by, bad if isinstance(bad, tuple) else (bad,)))}")


def _same_values(left: pd.Series, right: pd.Series) -> np.ndarray:
    """Elementwise equality that handles numeric dtype differences safely."""
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        return np.isclose(left.to_numpy(float), right.to_numpy(float), rtol=0.0, atol=0.0,
                          equal_nan=False)
    return left.astype(str).to_numpy() == right.astype(str).to_numpy()


def _candidate_mapping(diagnostic: pd.DataFrame,
                       candidates: Sequence[str]) -> dict[str, int]:
    pairs = diagnostic[["candidate", "candidate_order"]].drop_duplicates()
    if pairs.candidate.duplicated().any():
        _fail("a candidate has more than one candidate_order")
    if pairs.candidate_order.duplicated().any():
        _fail("candidate_order is not one-to-one")
    mapping = dict(zip(pairs.candidate.astype(str), pairs.candidate_order.astype(int)))
    if set(mapping) != set(candidates):
        _fail(f"diagnostic candidate set is {sorted(mapping)}, expected {list(candidates)}")
    return mapping


def validate_frames(accuracy: pd.DataFrame, diagnostic: pd.DataFrame,
                    candidates: Sequence[str] = PRIMARY_CANDIDATES) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate both frames and return the exact candidate-score join.

    This function is intentionally side-effect free: callers may use it in CPU
    tests without writing parquet files.  File/sidecar identity is checked by
    :func:`load_pair`.
    """
    candidates = tuple(candidates)
    if len(candidates) != len(set(candidates)) or not candidates:
        _fail("candidate list must be nonempty and unique")
    accuracy = accuracy.copy()
    diagnostic = diagnostic.copy()
    _require_columns(accuracy, ACCURACY_COLUMNS, "accuracy parquet")
    _require_columns(diagnostic, DIAGNOSTIC_COLUMNS, "diagnostic parquet")
    raw = sorted(column for column in diagnostic.columns if RAW_LOGIT_COLUMN.search(column))
    if raw:
        _fail("diagnostic parquet contains forbidden raw-logit column(s): " + ", ".join(raw))
    reserved = sorted({"score", "arm", "_merge"}.intersection(diagnostic.columns))
    if reserved:
        _fail("diagnostic parquet must not duplicate independently decoded accuracy fields: " +
              ", ".join(reserved))
    _require_nonnull(accuracy, ACCURACY_COLUMNS, "accuracy parquet")
    _require_nonnull(diagnostic, DIAGNOSTIC_COLUMNS, "diagnostic parquet")
    accuracy["arm"] = accuracy.arm.astype(str)
    diagnostic["candidate"] = diagnostic.candidate.astype(str)
    optional_shared = []
    for column in OPTIONAL_SHARED_PROVENANCE:
        if (column in accuracy.columns) != (column in diagnostic.columns):
            _fail(f"accuracy/diagnostic provenance presence disagrees for {column}")
        if column in diagnostic.columns:
            _require_nonnull(accuracy, (column,), "accuracy parquet")
            _require_nonnull(diagnostic, (column,), "diagnostic parquet")
            optional_shared.append(column)
    shared_provenance = (*SHARED_PROVENANCE, *optional_shared)

    for frame, label in ((accuracy, "accuracy parquet"), (diagnostic, "diagnostic parquet")):
        for column in ("ctx", "prompt_idx", *TASK_FIELDS, "window", "maxb", "rot_seed"):
            _as_integer(frame, column, label, minimum=0 if column in ("prompt_idx", "rot_seed") else 1)
        _as_finite_float(frame, "B", label)
    for column in ("candidate_order", "trace_len"):
        _as_integer(diagnostic, column, "diagnostic parquet", minimum=0)
    for column in ("trace_steps_requested", "vocab_size"):
        _as_integer(diagnostic, column, "diagnostic parquet", minimum=1)
    for column in ("mean_kl", "max_kl", "fp_token_ce", "top1_agreement", "t_policy_trace"):
        _as_finite_float(diagnostic, column, "diagnostic parquet")
    _as_finite_float(accuracy, "score", "accuracy parquet")

    if (diagnostic.B <= 0).any():
        _fail("diagnostic candidate budgets must be positive")
    if ((accuracy.score < 0) | (accuracy.score > 1)).any():
        _fail("accuracy score must be in [0, 1]")
    if (diagnostic[["mean_kl", "max_kl", "fp_token_ce", "t_policy_trace"]] < -1e-6).any().any():
        _fail("KL, cross entropy, and timing metrics must be nonnegative")
    if (diagnostic.max_kl + 1e-6 < diagnostic.mean_kl).any():
        _fail("max_kl is smaller than mean_kl")
    if ((diagnostic.top1_agreement < 0) | (diagnostic.top1_agreement > 1)).any():
        _fail("top1_agreement must be in [0, 1]")
    if (diagnostic.trace_len.lt(1).any() or
            diagnostic.trace_len.gt(diagnostic.trace_steps_requested).any()):
        _fail("trace_len must be between 1 and trace_steps_requested")
    if not diagnostic.trace_token_hash.astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        _fail("trace_token_hash must be a lowercase SHA-256 digest")
    if "fp_argmax_verified" in diagnostic:
        if diagnostic.fp_argmax_verified.isna().any() or not diagnostic.fp_argmax_verified.eq(True).all():
            _fail("diagnostic parquet contains an unverified FP replay")

    mapping = _candidate_mapping(diagnostic, candidates)
    prompt_keys = [*BASE_KEYS, "B"]
    expected = set(candidates)
    for key, group in diagnostic.groupby(prompt_keys, dropna=False, sort=False):
        got = set(group.candidate.astype(str))
        if len(group) != len(candidates) or got != expected:
            _fail(f"diagnostic prompt cell {key} has candidates {sorted(got)}; expected {list(candidates)}")
        if "selected_policy" in diagnostic:
            if group.selected_policy.isna().any() or group.selected_policy.nunique() != 1:
                _fail(f"diagnostic prompt cell {key} mixes selected_policy")
            expected_selection = min(candidates,
                                     key=lambda name: (float(group.loc[group.candidate == name,
                                                                       "mean_kl"].iloc[0]),
                                                       mapping[name]))
            if str(group.selected_policy.iloc[0]) != expected_selection:
                _fail(f"diagnostic prompt cell {key} records selected_policy="
                      f"{group.selected_policy.iloc[0]!r}, expected {expected_selection!r}")
    if diagnostic.duplicated(list(JOIN_KEYS)).any():
        _fail("diagnostic parquet has duplicate prompt/candidate join keys")

    # The candidates must replay one shared FP prefix.  Hash and length may vary
    # by prompt, but never by candidate for that prompt.
    _one_value(diagnostic, (*shared_provenance, *TRACE_PROVENANCE), CELL_KEYS,
               "diagnostic parquet")
    _one_value(diagnostic, (*shared_provenance, *TRACE_PROVENANCE,
                            *PROMPT_TRACE_PROVENANCE), prompt_keys,
               "diagnostic parquet")

    arms = set(accuracy.arm.astype(str))
    unexpected = arms - expected - {"fp"}
    missing_arms = expected.union({"fp"}) - arms
    if unexpected or missing_arms:
        _fail(f"accuracy arms mismatch: missing={sorted(missing_arms)}, unexpected={sorted(unexpected)}")
    candidate_accuracy = accuracy[accuracy.arm.astype(str).isin(expected)].copy()
    candidate_accuracy["candidate"] = candidate_accuracy.arm.astype(str)
    if candidate_accuracy.duplicated(list(JOIN_KEYS)).any():
        _fail("accuracy parquet has duplicate prompt/candidate join keys")
    for key, group in candidate_accuracy.groupby(prompt_keys, dropna=False, sort=False):
        got = set(group.candidate)
        if len(group) != len(candidates) or got != expected:
            _fail(f"accuracy prompt cell {key} has candidates {sorted(got)}; expected {list(candidates)}")

    # FP is a single B=0 ceiling row per prompt.  Candidate budgets may be
    # repeated, but the independent FP decode must not be duplicated.
    fp = accuracy[accuracy.arm.astype(str) == "fp"]
    if fp.duplicated(list(BASE_KEYS)).any():
        _fail("accuracy parquet has duplicate FP rows")
    candidate_prompts = diagnostic[list(BASE_KEYS)].drop_duplicates()
    fp_prompts = fp[list(BASE_KEYS)].drop_duplicates()
    coverage = candidate_prompts.merge(fp_prompts, on=list(BASE_KEYS), how="outer", indicator=True)
    if not (coverage._merge == "both").all() or len(fp) != len(fp_prompts):
        _fail("accuracy FP rows do not exactly cover diagnostic prompts")
    if not np.allclose(fp.B.to_numpy(float), 0.0, rtol=0.0, atol=0.0):
        _fail("accuracy FP rows must use B=0")

    acc_for_join = candidate_accuracy[[*JOIN_KEYS, "score", *shared_provenance]].copy()
    joined = diagnostic.merge(acc_for_join, on=list(JOIN_KEYS), how="outer",
                              validate="one_to_one", indicator=True,
                              suffixes=("", "_accuracy"))
    if not (joined._merge == "both").all():
        samples = joined.loc[joined._merge != "both", [*JOIN_KEYS, "_merge"]].head(3)
        _fail("diagnostic and accuracy candidate rows are not an exact one-to-one match: " +
              samples.to_dict("records").__repr__())
    joined = joined.drop(columns="_merge")
    for column in shared_provenance:
        other = f"{column}_accuracy"
        if not _same_values(joined[column], joined[other]).all():
            _fail(f"accuracy and diagnostic provenance disagree for {column}")
        joined = joined.drop(columns=other)
    return joined, mapping


def _bootstrap(h: np.ndarray, g: np.ndarray, n_boot: int, rng: np.random.Generator) -> Mapping[str, tuple[float, float]]:
    if n_boot < 1:
        _fail("bootstrap replicate count must be positive")
    n = len(h)
    draw = rng.integers(0, n, size=(n_boot, n))
    hb = h[draw].mean(axis=1)
    gb = g[draw].mean(axis=1)
    rb = (h - g)[draw].mean(axis=1)

    def interval(values: np.ndarray) -> tuple[float, float]:
        return float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))

    ratio = np.divide(gb, hb, out=np.full_like(gb, np.nan), where=hb > 0)
    finite_ratio = ratio[np.isfinite(ratio)]
    return {
        "H": interval(hb),
        "G": interval(gb),
        "regret": interval(rb),
        "captured": interval(finite_ratio) if len(finite_ratio) else (float("nan"), float("nan")),
    }


def analyze_frames(accuracy: pd.DataFrame, diagnostic: pd.DataFrame,
                   candidates: Sequence[str] = PRIMARY_CANDIDATES,
                   n_boot: int = 10_000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(cell_summary, prompt_audit)`` after strict validation."""
    joined, mapping = validate_frames(accuracy, diagnostic, candidates)
    order = tuple(sorted(candidates, key=lambda candidate: mapping[candidate]))
    rng = np.random.default_rng(seed)
    prompt_rows: list[dict] = []

    prompt_keys = [*CELL_KEYS, "prompt_idx"]
    for key, group in joined.groupby(prompt_keys, dropna=False, sort=True):
        group = group.set_index("candidate", drop=False)
        scores = {candidate: float(group.loc[candidate, "score"]) for candidate in order}
        kls = {candidate: float(group.loc[candidate, "mean_kl"]) for candidate in order}
        best_score = max(scores.values())
        ties = tuple(candidate for candidate in order if scores[candidate] == best_score)
        selected = min(order, key=lambda candidate: (kls[candidate], mapping[candidate]))
        uniform = scores["uniform"]
        selected_score = scores[selected]
        row = dict(zip(prompt_keys, key if isinstance(key, tuple) else (key,)))
        row.update(
            uniform_score=uniform,
            envelope_score=best_score,
            endtask_ties="|".join(ties),
            endtask_tie_count=len(ties),
            kl_candidate=selected,
            kl_candidate_order=mapping[selected],
            kl_mean_kl=kls[selected],
            kl_score=selected_score,
            H=best_score - uniform,
            G=selected_score - uniform,
            regret=best_score - selected_score,
            rescued=bool(uniform < 1.0 and selected_score > uniform),
            harmed=bool(uniform >= 1.0 and selected_score < uniform),
            tie_hit=bool(selected in ties),
        )
        for candidate in order:
            row[f"score_{candidate}"] = scores[candidate]
            row[f"mean_kl_{candidate}"] = kls[candidate]
            row[f"optimal_{candidate}"] = candidate in ties
        prompt_rows.append(row)

    prompts = pd.DataFrame(prompt_rows)
    summaries: list[dict] = []
    for key, group in prompts.groupby(list(CELL_KEYS), dropna=False, sort=True):
        h = group.H.to_numpy(float)
        g = group.G.to_numpy(float)
        intervals = _bootstrap(h, g, n_boot, rng)
        H = float(h.mean())
        G = float(g.mean())
        regret = float((h - g).mean())
        captured = G / H if H > 0 else float("nan")
        summary = dict(zip(CELL_KEYS, key if isinstance(key, tuple) else (key,)))
        summary.update(
            n_prompts=len(group),
            uniform_mean=float(group.uniform_score.mean()),
            envelope_mean=float(group.envelope_score.mean()),
            kl_selected_mean=float(group.kl_score.mean()),
            H=H, H_lo=intervals["H"][0], H_hi=intervals["H"][1],
            G=G, G_lo=intervals["G"][0], G_hi=intervals["G"][1],
            regret=regret, regret_lo=intervals["regret"][0],
            regret_hi=intervals["regret"][1],
            captured=captured, captured_lo=intervals["captured"][0],
            captured_hi=intervals["captured"][1],
            rescues=int(group.rescued.sum()), harms=int(group.harmed.sum()),
            tie_hits=int(group.tie_hit.sum()), tie_hit_rate=float(group.tie_hit.mean()),
        )
        for candidate in order:
            summary[f"selected_{candidate}"] = int((group.kl_candidate == candidate).sum())
            summary[f"optimal_ties_{candidate}"] = int(group[f"optimal_{candidate}"].sum())
        if H >= 0.10 and G >= 0.05 and captured >= 0.5:
            decision = "advance_mean_kl"
        elif H <= 0.05:
            decision = "revise_candidates"
        elif H >= 0.10 and (G <= 0 or captured < 0.25):
            decision = "reject_mean_kl"
        else:
            decision = "more_development_prompts"
        summary["decision"] = decision
        summaries.append(summary)
    return pd.DataFrame(summaries), prompts


def _sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _sidecar_equal(sidecar: Mapping, key: str, actual, label: str) -> None:
    if key not in sidecar:
        _fail(f"{label} sidecar is missing {key}")
    expected = sidecar[key]
    if isinstance(actual, (np.integer, np.floating)):
        actual = actual.item()
    if expected != actual:
        _fail(f"{label} sidecar {key}={expected!r}, artifact has {actual!r}")


def validate_sidecars(accuracy_path: os.PathLike[str] | str,
                      diagnostic_path: os.PathLike[str] | str,
                      accuracy: pd.DataFrame, diagnostic: pd.DataFrame,
                      mapping: Mapping[str, int],
                      candidates: Sequence[str] = PRIMARY_CANDIDATES) -> tuple[dict, dict]:
    """Validate file identity and row-level provenance against both sidecars."""
    accuracy_path, diagnostic_path = Path(accuracy_path), Path(diagnostic_path)
    accuracy_side = _read_sidecar(_sidecar_path(accuracy_path), "accuracy")
    diag_side = _read_sidecar(_sidecar_path(diagnostic_path), "diagnostic")
    required = (
        "parquet", "model", "model_id", "ctx", "native_ctx", "tasks",
        "task_config", "budgets", "n_prompts", "prompt_offset", "window", "maxb",
        "question_agnostic", "candidates", "trace_rule_version",
        "trace_steps_requested", "teacher", "metric_dtype", "vocabulary",
        "allocator_budget_rule", "rot_seed", "norm_correct", "rows",
        "expected_rows", "corpus_sha", "accuracy_parquet", "accuracy_sidecar",
        "accuracy_sha256", "accuracy_rows", "no_raw_logits",
    )
    missing = sorted(set(required) - set(diag_side))
    if missing:
        _fail("diagnostic sidecar is missing: " + ", ".join(missing))
    if diag_side["no_raw_logits"] is not True:
        _fail("diagnostic sidecar does not assert no_raw_logits=true")
    expected_method = {"teacher": "fp_greedy", "metric_dtype": "float32",
                       "vocabulary": "full", "allocator_budget_rule": "feasible"}
    for field, expected_value in expected_method.items():
        if diag_side[field] != expected_value:
            _fail(f"diagnostic sidecar {field}={diag_side[field]!r}, expected {expected_value!r}")
    if diag_side["parquet"] != diagnostic_path.name:
        _fail("diagnostic sidecar parquet does not name the supplied diagnostic file")
    if diag_side["accuracy_parquet"] != accuracy_path.name:
        _fail("diagnostic sidecar accuracy_parquet does not name the supplied accuracy file")
    if diag_side["accuracy_sidecar"] != _sidecar_path(accuracy_path).name:
        _fail("diagnostic sidecar accuracy_sidecar does not name the supplied accuracy sidecar")
    if diag_side["accuracy_sha256"] != _sha256(accuracy_path):
        _fail("diagnostic sidecar accuracy_sha256 does not match the supplied accuracy file")
    if int(diag_side["accuracy_rows"]) != len(accuracy):
        _fail("diagnostic sidecar accuracy_rows does not match the supplied accuracy file")
    if int(diag_side["rows"]) != len(diagnostic):
        _fail("diagnostic sidecar rows does not match the diagnostic parquet")
    if int(diag_side["expected_rows"]) != len(diagnostic):
        _fail("diagnostic sidecar expected_rows does not match the diagnostic parquet")

    ordered = list(sorted(candidates, key=lambda candidate: mapping[candidate]))
    if diag_side["candidates"] != ordered:
        _fail(f"diagnostic sidecar candidates={diag_side['candidates']!r}, expected {ordered!r}")
    singleton_fields = ("model", "model_id", "ctx", "native_ctx", "window", "maxb",
                        "question_agnostic", "trace_rule_version",
                        "trace_steps_requested", "corpus_sha", "rot_seed",
                        "norm_correct", "allocator_budget_rule")
    for field in singleton_fields:
        values = diagnostic[field].drop_duplicates().tolist()
        if len(values) != 1:
            _fail(f"diagnostic parquet mixes sidecar field {field}: {values!r}")
        _sidecar_equal(diag_side, field, values[0], "diagnostic")
    _sidecar_equal(diag_side, "tasks", diagnostic.task.drop_duplicates().tolist(), "diagnostic")
    _sidecar_equal(diag_side, "budgets", diagnostic.B.drop_duplicates().tolist(), "diagnostic")
    _sidecar_equal(diag_side, "n_prompts", int(diagnostic.prompt_idx.nunique()), "diagnostic")
    _sidecar_equal(diag_side, "prompt_offset", int(diagnostic.prompt_idx.min()), "diagnostic")
    task_config = {field: int(diagnostic[field].iloc[0]) for field in TASK_FIELDS}
    for field in TASK_FIELDS:
        if diagnostic[field].nunique() != 1:
            _fail(f"diagnostic parquet mixes task configuration field {field}")
    _sidecar_equal(diag_side, "task_config", task_config, "diagnostic")

    # The accuracy sidecar is checked for the fields that define the artifact;
    # the diagnostic sidecar's SHA pins every remaining byte.
    for field in ("parquet", "model", "model_id", "ctx", "native_ctx", "tasks",
                  "arms", "task_config", "budgets", "n_prompts", "prompt_offset",
                  "window", "maxb", "question_agnostic", "corpus_sha", "rows"):
        if field not in accuracy_side:
            _fail(f"accuracy sidecar is missing {field}")
    if accuracy_side["parquet"] != accuracy_path.name or int(accuracy_side["rows"]) != len(accuracy):
        _fail("accuracy sidecar file name or row count does not match the accuracy parquet")
    for field in ("model", "model_id", "ctx", "native_ctx", "window", "maxb",
                  "question_agnostic", "corpus_sha"):
        values = accuracy[field].drop_duplicates().tolist()
        if len(values) != 1 or accuracy_side[field] != values[0]:
            _fail(f"accuracy sidecar provenance does not match {field}")
    if accuracy_side["tasks"] != accuracy.task.drop_duplicates().tolist():
        _fail("accuracy sidecar tasks do not match the accuracy parquet")
    if accuracy_side["arms"] != accuracy.arm.drop_duplicates().tolist():
        _fail("accuracy sidecar arms do not match the accuracy parquet")
    candidate_budgets = accuracy.loc[accuracy.arm != "fp", "B"].drop_duplicates().tolist()
    if accuracy_side["budgets"] != candidate_budgets:
        _fail("accuracy sidecar budgets do not match the candidate rows")
    actual_task_config = {field: int(accuracy[field].iloc[0]) for field in TASK_FIELDS}
    if (any(accuracy[field].nunique() != 1 for field in TASK_FIELDS) or
            accuracy_side["task_config"] != actual_task_config):
        _fail("accuracy sidecar task_config does not match the accuracy parquet")
    if int(accuracy_side["n_prompts"]) != int(accuracy.prompt_idx.nunique()):
        _fail("accuracy sidecar n_prompts does not match the accuracy parquet")
    if int(accuracy_side["prompt_offset"]) != int(accuracy.prompt_idx.min()):
        _fail("accuracy sidecar prompt_offset does not match the accuracy parquet")
    return accuracy_side, diag_side


def load_pair(accuracy_path: os.PathLike[str] | str,
              diagnostic_path: os.PathLike[str] | str,
              candidates: Sequence[str] = PRIMARY_CANDIDATES,
              n_boot: int = 10_000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, authenticate, validate, and analyze one artifact pair."""
    accuracy_path, diagnostic_path = Path(accuracy_path), Path(diagnostic_path)
    if not accuracy_path.is_file():
        _fail(f"missing accuracy parquet: {accuracy_path}")
    if not diagnostic_path.is_file():
        _fail(f"missing diagnostic parquet: {diagnostic_path}")
    accuracy = pd.read_parquet(accuracy_path)
    diagnostic = pd.read_parquet(diagnostic_path)
    # Validate frames first so candidate ordering is safe to use for sidecars.
    _, mapping = validate_frames(accuracy, diagnostic, candidates)
    validate_sidecars(accuracy_path, diagnostic_path, accuracy, diagnostic, mapping, candidates)
    return analyze_frames(accuracy, diagnostic, candidates, n_boot=n_boot, seed=seed)


def _fmt(value: float, signed: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):+.3f}" if signed else f"{float(value):.3f}"


def print_summary(summary: pd.DataFrame, candidates: Sequence[str] = PRIMARY_CANDIDATES) -> None:
    print("# Whole-policy diagnostic (paired prompt bootstrap, 90% intervals)")
    for row in summary.itertuples(index=False):
        print(f"\n{row.model} @ {int(row.ctx):,}  {row.task}  "
              f"[k{int(row.n_keys)}/v{int(row.n_values)}/h{int(row.n_hops)}; "
              f"B={row.B:g}; n={int(row.n_prompts)}]")
        print(f"  accuracy       uniform={_fmt(row.uniform_mean)}  "
              f"KL-selected={_fmt(row.kl_selected_mean)}  envelope={_fmt(row.envelope_mean)}")
        print(f"  H              {_fmt(row.H, True)} [{_fmt(row.H_lo, True)}, {_fmt(row.H_hi, True)}]")
        print(f"  G              {_fmt(row.G, True)} [{_fmt(row.G_lo, True)}, {_fmt(row.G_hi, True)}]")
        print(f"  H-G (regret)   {_fmt(row.regret, True)} "
              f"[{_fmt(row.regret_lo, True)}, {_fmt(row.regret_hi, True)}]")
        print(f"  G/H captured   {_fmt(row.captured)} "
              f"[{_fmt(row.captured_lo)}, {_fmt(row.captured_hi)}]")
        selections = ", ".join(f"{candidate}={int(getattr(row, 'selected_' + candidate))}"
                               for candidate in candidates)
        optimal = ", ".join(f"{candidate}={int(getattr(row, 'optimal_ties_' + candidate))}"
                            for candidate in candidates)
        print(f"  behavior       rescues={int(row.rescues)}  harms={int(row.harms)}  "
              f"tie-hit={int(row.tie_hits)}/{int(row.n_prompts)} ({row.tie_hit_rate:.1%})")
        print(f"  KL selections  {selections}")
        print(f"  oracle ties    {optimal}  (membership counts; ties retained)")
        print(f"  decision       {row.decision}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("accuracy_parquet", help="R8 independently greedy accuracy parquet")
    parser.add_argument("diagnostic_parquet", help="r8policy diagnostic parquet")
    parser.add_argument("--csv", help="write one summary row per model/context/task/config/budget")
    parser.add_argument("--prompt-csv", help="write the joined per-prompt selector audit")
    parser.add_argument("--bootstrap", type=int, default=10_000,
                        help="paired prompt-bootstrap replicates (default: 10000)")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed (default: 0)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary, prompts = load_pair(args.accuracy_parquet, args.diagnostic_parquet,
                                     n_boot=args.bootstrap, seed=args.seed)
    except (PolicyReaderError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print_summary(summary)
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv} ({len(summary)} summary rows)")
    if args.prompt_csv:
        Path(args.prompt_csv).parent.mkdir(parents=True, exist_ok=True)
        prompts.to_csv(args.prompt_csv, index=False)
        print(f"wrote {args.prompt_csv} ({len(prompts)} prompt rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
