#!/usr/bin/env python3
"""Authenticate and summarize the preregistered R8 k40 follow-up cell.

This reader accepts exactly one ordinary R8 accuracy parquet plus its JSON
sidecar.  Its contract is deliberately narrow: Llama-3.1-8B at 32K,
question-agnostic multikey NIAH, B=2, k40/v4/h4, prompts 540--579, and exactly
``fp`` plus ``uniform``.  It reports the standard task score as the primary
outcome and ``first_ok`` only as a secondary outcome.

Usage::

    .venv/bin/python \
      h0_measurement/bugs/9_sota_eviction_baselines/read_op2_k40.py \
      RESULT_K40.parquet [--csv SUMMARY.csv]

A valid artifact remains a successful read (exit status 0) when k40 is
ineligible. Contract or provenance failures use exit status 2.
"""
from __future__ import annotations

import argparse
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
PROMPTS = tuple(range(540, 580))
HALF_1 = tuple(range(540, 560))
HALF_2 = tuple(range(560, 580))
N_VALUES = 4
N_HOPS = 4
KEY_COUNT = 40
BUDGET = 2.0
WINDOW = 32
MAXB = 8
GENERATION_LIMIT_VERSION = "difficulty_v1"
MAX_NEW_TOKENS = 24
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 0

REQUIRED_COLUMNS = (
    "model", "model_id", "ctx", "native_ctx", "task", "prompt_idx",
    "arm", "B", "score", "hits", "n_expected", "n_keys", "n_values", "n_hops",
    "task_n_needles", "max_new_tokens", "reached_max_new", "gen_len",
    "bits_per_token", "n_prompt_tokens", "ctx_len", "n_question_tokens",
    "question_agnostic", "window", "observed_queries", "allocator_budget_rule", "maxb",
    "corpus_sha", "corpus_doc", "corpus_offset", "corpus_spliced", "synthetic",
    "rot_seed",
    "norm_correct", "first_ok", "needle_depths", "target_needle_rank",
    "target_needle_depth",
)
SINGLETON_COLUMNS = (
    "model", "model_id", "ctx", "native_ctx", "task", "n_keys",
    "n_values", "n_hops", "task_n_needles", "max_new_tokens", "n_expected",
    "question_agnostic", "window", "observed_queries", "allocator_budget_rule", "maxb", "corpus_sha",
    "corpus_spliced", "synthetic", "rot_seed",
    "norm_correct",
)
SIDECAR_FIELDS = (
    "parquet", "model", "model_id", "ctx", "native_ctx", "tasks", "arms",
    "task_config", "generation_limit_version", "generation_limits", "budgets",
    "n_prompts", "prompt_offset", "window", "allocator_budget_rule", "maxb",
    "rows", "rot_seed", "norm_correct", "attn_impl", "compress_from",
    "question_agnostic", "observation_queries", "corpus_sha", "baselines", "p2",
    "task_generation_version",
    "target_needle_provenance_version",
)
REAL_CORPUS_RE = re.compile(r"[0-9a-f]{16,64}")
EXPECTED_CORPUS_SHA = "0a26bc1e05a1eea8"
TASK_GENERATION_VERSION = "ruler_pg19_v1"
TARGET_PROVENANCE_VERSION = "queried_needle_v1"


class Op2ReaderError(ValueError):
    """An input does not satisfy the frozen operating-point contract."""


def _fail(message: str) -> None:
    raise Op2ReaderError(message)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        _fail("parquet is missing required columns: " + ", ".join(missing))


def _numeric(frame: pd.DataFrame, column: str, *, integer: bool = False) -> pd.Series:
    try:
        values = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as exc:
        _fail(f"{column} is not numeric: {exc}")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        _fail(f"{column} contains a non-finite value")
    if integer and not np.equal(array, np.floor(array)).all():
        _fail(f"{column} must contain integers")
    return values.astype(np.int64) if integer else values.astype(float)


def _only(frame: pd.DataFrame, column: str):
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        _fail(f"parquet mixes {column}: {values!r}")
    value = values[0]
    return value.item() if isinstance(value, (np.integer, np.floating)) else value


def _sidecar_path(parquet: os.PathLike[str] | str) -> Path:
    return Path(parquet).with_suffix(".json")


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
        _fail(f"sidecar {path} is missing: {', '.join(missing)}")
    return value


def _same(expected, actual, description: str) -> None:
    if isinstance(actual, (np.integer, np.floating)):
        actual = actual.item()
    if expected != actual:
        _fail(f"{description}: expected {expected!r}, got {actual!r}")


def validate_frame(frame: pd.DataFrame, *, label: str = "parquet") -> int:
    """Validate one in-memory artifact and return its key count.

    File identity and sidecar fields are handled by :func:`load_artifact`.
    Keeping row validation separate makes the numerical contract easy to test
    without touching the filesystem.
    """
    frame = frame.copy()
    _require_columns(frame, REQUIRED_COLUMNS)
    if len(frame) != 80:
        _fail(f"{label} has {len(frame)} rows; expected exactly 80")
    null_columns = [column for column in REQUIRED_COLUMNS if frame[column].isna().any()]
    if null_columns:
        _fail(f"{label} has null values in: {', '.join(null_columns)}")

    for column in ("ctx", "native_ctx", "prompt_idx", "n_keys", "n_values",
                   "n_hops", "task_n_needles", "max_new_tokens", "gen_len",
                   "n_prompt_tokens", "ctx_len", "n_question_tokens", "window", "maxb",
                   "rot_seed", "corpus_offset", "target_needle_rank", "hits",
                   "n_expected", "observed_queries"):
        frame[column] = _numeric(frame, column, integer=True)
    for column in ("B", "score", "bits_per_token", "first_ok",
                   "target_needle_depth"):
        frame[column] = _numeric(frame, column)

    expected_singletons = {
        "model": MODEL, "model_id": MODEL_ID, "ctx": CTX,
        "native_ctx": NATIVE_CTX, "task": TASK, "n_values": N_VALUES,
        "n_hops": N_HOPS, "task_n_needles": KEY_COUNT,
        "max_new_tokens": MAX_NEW_TOKENS, "n_expected": 1,
        "question_agnostic": True, "window": WINDOW, "observed_queries": WINDOW,
        "allocator_budget_rule": "feasible", "maxb": MAXB,
        "corpus_spliced": False, "synthetic": False, "rot_seed": 0,
        "norm_correct": True,
    }
    for column in SINGLETON_COLUMNS:
        value = _only(frame, column)
        if column in expected_singletons:
            _same(expected_singletons[column], value, f"{label}.{column}")
    for column in ("reached_max_new", "question_agnostic", "corpus_spliced",
                   "synthetic", "norm_correct"):
        if not pd.api.types.is_bool_dtype(frame[column].dtype):
            _fail(f"{label}.{column} must have boolean dtype")

    n_keys = int(_only(frame, "n_keys"))
    if n_keys != KEY_COUNT:
        _fail(f"{label}.n_keys={n_keys}; expected exactly {KEY_COUNT}")

    corpus_sha = str(_only(frame, "corpus_sha"))
    if not REAL_CORPUS_RE.fullmatch(corpus_sha) or corpus_sha == "0" * len(corpus_sha):
        _fail(f"{label}.corpus_sha is not a real corpus identity: {corpus_sha!r}")
    if corpus_sha != EXPECTED_CORPUS_SHA:
        _fail(f"{label}.corpus_sha: expected the frozen corpus "
              f"{EXPECTED_CORPUS_SHA!r}, got {corpus_sha!r}")

    if set(frame.prompt_idx.tolist()) != set(PROMPTS):
        _fail(f"{label} must contain exactly prompts 540..579")
    if set(frame.arm.astype(str)) != {"fp", "uniform"}:
        _fail(f"{label} arms must be exactly fp and uniform")
    keys = frame.groupby(["prompt_idx", "arm"], dropna=False).size()
    expected_keys = {(prompt, arm) for prompt in PROMPTS for arm in ("fp", "uniform")}
    if set(keys.index.tolist()) != expected_keys or not keys.eq(1).all():
        _fail(f"{label} must have one unique fp and uniform row for every prompt")

    # The 40-ID block was chosen to traverse the complete 40-book corpus once.
    # Authenticate both the cycle and identical source windows across arms.
    if (frame.corpus_doc.astype(str).str.len() == 0).any():
        _fail(f"{label}.corpus_doc contains an empty document identity")
    if (frame.corpus_offset < 0).any():
        _fail(f"{label}.corpus_offset must be nonnegative")
    if ((frame.n_prompt_tokens <= 0) | (frame.n_prompt_tokens > CTX)).any():
        _fail(f"{label}.n_prompt_tokens must be in (0, ctx]")
    if ((frame.ctx_len <= 0) | (frame.n_question_tokens <= 0)).any():
        _fail(f"{label}.ctx_len and n_question_tokens must be positive")
    source = frame.groupby("prompt_idx")[[
        "corpus_doc", "corpus_offset", "n_prompt_tokens", "ctx_len",
        "n_question_tokens",
    ]].nunique(dropna=False)
    if source.ne(1).any().any():
        _fail(f"{label} source/task provenance differs between arms")
    fp_sources = frame[frame.arm.astype(str) == "fp"]
    if fp_sources.corpus_doc.nunique() != len(PROMPTS):
        _fail(f"{label} does not cover 40 distinct corpus documents")

    fp = frame[frame.arm.astype(str) == "fp"]
    uniform = frame[frame.arm.astype(str) == "uniform"]
    if not np.equal(fp.B.to_numpy(float), 0.0).all():
        _fail(f"{label} FP rows must use B=0")
    if not np.equal(uniform.B.to_numpy(float), BUDGET).all():
        _fail(f"{label} uniform rows must use B=2")
    if ((frame.score < 0.0) | (frame.score > 1.0)).any():
        _fail(f"{label} score must be in [0, 1]")
    if not frame.first_ok.isin([0.0, 1.0]).all():
        _fail(f"{label} multikey first_ok must be binary")
    if not frame.hits.isin([0, 1]).all() or not frame.n_expected.eq(1).all():
        _fail(f"{label} multikey hits must be binary with n_expected=1")
    if not np.equal(frame.score.to_numpy(float), frame.hits.to_numpy(float)).all():
        _fail(f"{label} score must equal hits for one-answer multikey")
    if (frame.first_ok > frame.score).any():
        _fail(f"{label} first_ok cannot exceed substring score")
    if not np.isclose(fp.bits_per_token.to_numpy(float), 16.0,
                      rtol=0.0, atol=1e-7).all():
        _fail(f"{label} FP rows must record 16 bits/token")
    uniform_bits = uniform.bits_per_token.to_numpy(float)
    if not np.isclose(uniform_bits, BUDGET, rtol=0.0, atol=1e-7).all():
        _fail(f"{label} uniform rows must realize exactly the B=2 width")

    # The runner's invariant is equality, including correct answers that happen
    # to consume the complete allowance.  The eligibility gate below is narrower:
    # only an incomplete FP answer at the cap invalidates the operating point.
    expected_cap = frame.gen_len.to_numpy(int) >= frame.max_new_tokens.to_numpy(int)
    actual_cap = frame.reached_max_new.astype(bool).to_numpy()
    if not np.array_equal(actual_cap, expected_cap):
        _fail(f"{label} violates reached_max_new == (gen_len >= max_new_tokens)")
    if ((frame.gen_len < 0) | (frame.gen_len > frame.max_new_tokens)).any():
        _fail(f"{label}.gen_len must be between zero and max_new_tokens")

    # V2 artifacts must expose the queried needle independently of the full
    # depth list; this reader deliberately does not accept the legacy schema.
    ranks = frame.target_needle_rank
    depths = frame.target_needle_depth
    if ((ranks < 0) | (ranks >= n_keys)).any():
        _fail(f"{label}.target_needle_rank must be in [0, n_keys)")
    if ((depths < .05) | (depths > .95)).any():
        _fail(f"{label}.target_needle_depth must be in [.05, .95]")
    target = pd.DataFrame({"prompt_idx": frame.prompt_idx,
                           "rank": ranks, "depth": depths})
    if (target.groupby("prompt_idx")[["rank", "depth"]]
            .nunique(dropna=False).ne(1).any().any()):
        _fail(f"{label} queried-needle provenance differs between arms")

    parsed_vectors = []
    for row, rank, depth in zip(frame.itertuples(index=False), ranks, depths):
        try:
            raw_values = json.loads(row.needle_depths)
        except (TypeError, json.JSONDecodeError) as exc:
            _fail(f"{label}.needle_depths is not valid JSON: {exc}")
        if not isinstance(raw_values, list) or len(raw_values) != n_keys:
            _fail(f"{label}.needle_depths must contain exactly n_keys entries")
        try:
            values = [float(value) for value in raw_values]
        except (TypeError, ValueError) as exc:
            _fail(f"{label}.needle_depths contains a nonnumeric entry: {exc}")
        if not all(math.isfinite(value) and .05 <= value <= .95 for value in values):
            _fail(f"{label}.needle_depths entries must be finite and in [.05, .95]")
        if any(left > right for left, right in zip(values, values[1:])):
            _fail(f"{label}.needle_depths must be nondecreasing")
        parsed_vectors.append(tuple(values))
        selected_depth = values[int(rank)]
        if not math.isclose(selected_depth, float(depth), rel_tol=0.0, abs_tol=5e-5):
            _fail(f"{label}.target_needle_depth does not match needle_depths[rank]")
    vector_frame = pd.DataFrame({"prompt_idx": frame.prompt_idx,
                                 "depths": parsed_vectors})
    if vector_frame.groupby("prompt_idx").depths.nunique(dropna=False).ne(1).any():
        _fail(f"{label}.needle_depths differs between arms")
    return n_keys


def validate_sidecar(parquet: Path, frame: pd.DataFrame, sidecar: Mapping) -> None:
    """Authenticate the sidecar against both the supplied path and row data."""
    n_keys = int(_only(frame, "n_keys"))
    expected = {
        "parquet": parquet.name,
        "model": MODEL,
        "model_id": MODEL_ID,
        "ctx": CTX,
        "native_ctx": NATIVE_CTX,
        "tasks": [TASK],
        "arms": ["fp", "uniform"],
        "task_config": {"n_keys": n_keys, "n_values": N_VALUES, "n_hops": N_HOPS},
        "generation_limit_version": GENERATION_LIMIT_VERSION,
        "generation_limits": {TASK: MAX_NEW_TOKENS},
        "budgets": [2],
        "n_prompts": len(PROMPTS),
        "prompt_offset": PROMPTS[0],
        "window": WINDOW,
        "allocator_budget_rule": "feasible",
        "maxb": MAXB,
        "rows": 80,
        "rot_seed": 0,
        "norm_correct": True,
        "attn_impl": "sieve_compress",
        "compress_from": "first_answer_token",
        "question_agnostic": True,
        "observation_queries": [WINDOW],
        "corpus_sha": str(_only(frame, "corpus_sha")),
    }
    for field, value in expected.items():
        _same(value, sidecar[field], f"sidecar.{field}")

    # Cross-check every scalar provenance field represented in both places.
    row_fields = ("model", "model_id", "ctx", "native_ctx", "window", "maxb",
                  "question_agnostic", "allocator_budget_rule", "rot_seed",
                  "norm_correct", "corpus_sha")
    for field in row_fields:
        _same(sidecar[field], _only(frame, field), f"sidecar/row {field}")

    _same(TASK_GENERATION_VERSION, sidecar["task_generation_version"],
          "sidecar.task_generation_version")
    _same(TARGET_PROVENANCE_VERSION, sidecar["target_needle_provenance_version"],
          "sidecar.target_needle_provenance_version")
    _same(None, sidecar["baselines"], "sidecar.baselines")
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
    """Load and authenticate one parquet/sidecar pair."""
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
    sidecar = _read_sidecar(_sidecar_path(parquet))
    validate_sidecar(parquet, frame, sidecar)
    return frame


def _paired_interval(differences: np.ndarray) -> tuple[float, float]:
    if differences.shape != (len(PROMPTS),):
        _fail(f"paired bootstrap expected {len(PROMPTS)} prompt differences")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.integers(0, len(differences),
                        size=(BOOTSTRAP_REPLICATES, len(differences)))
    means = differences[draw].mean(axis=1)
    return float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def analyze_frame(frame: pd.DataFrame) -> dict:
    """Return the preregistered summary and eligibility gates for one k."""
    n_keys = validate_frame(frame)
    pivot = frame.pivot(index="prompt_idx", columns="arm", values="score").loc[list(PROMPTS)]
    fp = pivot.fp.to_numpy(float)
    uniform = pivot.uniform.to_numpy(float)
    first_ok = frame.pivot(index="prompt_idx", columns="arm", values="first_ok").loc[
        list(PROMPTS)]
    differences = fp - uniform
    delta_lo, delta_hi = _paired_interval(differences)

    first = pivot.loc[list(HALF_1)]
    second = pivot.loc[list(HALF_2)]
    fp_rows = frame[frame.arm.astype(str) == "fp"]
    incomplete_capped_fp = int(((fp_rows.score < 1.0) &
                                fp_rows.reached_max_new.astype(bool)).sum())
    fp_mean = float(fp.mean())
    uniform_mean = float(uniform.mean())
    fp_gate = fp_mean >= 0.95
    cap_gate = incomplete_capped_fp == 0
    uniform_gate = 0.50 <= uniform_mean <= 0.75
    half_gate = (0.35 <= float(first.uniform.mean()) <= 0.85 and
                 0.35 <= float(second.uniform.mean()) <= 0.85)
    delta_gate = delta_lo > 0.05
    return {
        "n_keys": n_keys,
        "n_prompts": len(PROMPTS),
        "fp_mean": fp_mean,
        "uniform_mean": uniform_mean,
        "fp_first_ok": float(first_ok.fp.mean()),
        "uniform_first_ok": float(first_ok.uniform.mean()),
        "fp_540_559": float(first.fp.mean()),
        "uniform_540_559": float(first.uniform.mean()),
        "uniform_first_ok_540_559": float(first_ok.loc[list(HALF_1)].uniform.mean()),
        "fp_560_579": float(second.fp.mean()),
        "uniform_560_579": float(second.uniform.mean()),
        "uniform_first_ok_560_579": float(first_ok.loc[list(HALF_2)].uniform.mean()),
        "delta_fp_minus_uniform": float(differences.mean()),
        "delta_lo90": delta_lo,
        "delta_hi90": delta_hi,
        "incomplete_capped_fp": incomplete_capped_fp,
        "gate_fp_ge_095": fp_gate,
        "gate_no_incomplete_capped_fp": cap_gate,
        "gate_uniform_050_075": uniform_gate,
        "gate_each_half_035_085": half_gate,
        "gate_delta_lo_gt_005": delta_gate,
        "eligible": bool(fp_gate and cap_gate and uniform_gate and half_gate and delta_gate),
    }


def analyze_artifact(path: os.PathLike[str] | str) -> tuple[pd.DataFrame, bool]:
    """Authenticate the single frozen k40 input and summarize its gates."""
    frame = load_artifact(path)
    summary = pd.DataFrame([analyze_frame(frame)])
    eligible = bool(summary.loc[0, "eligible"])
    summary["selected"] = eligible
    return summary, eligible


def _fmt(value: float) -> str:
    return "NA" if value is None or not math.isfinite(float(value)) else f"{float(value):.3f}"


def print_summary(summary: pd.DataFrame, eligible: bool) -> None:
    print("# R8 operating-point k40 follow-up")
    print("# primary: standard RULER string_match_all; secondary: first_ok")
    print("# paired prompt bootstrap: 90%, 10000 replicates, seed 0")
    row = summary.iloc[0]
    print(
        f"k={int(row.n_keys)}  FP={_fmt(row.fp_mean)}  "
        f"uniform={_fmt(row.uniform_mean)}  "
        f"540-559(FP/uniform)={_fmt(row.fp_540_559)}/{_fmt(row.uniform_540_559)}  "
        f"560-579(FP/uniform)={_fmt(row.fp_560_579)}/{_fmt(row.uniform_560_579)}"
    )
    print(
        f"  first_ok FP/uniform={_fmt(row.fp_first_ok)}/{_fmt(row.uniform_first_ok)}  "
        f"uniform halves={_fmt(row.uniform_first_ok_540_559)}/"
        f"{_fmt(row.uniform_first_ok_560_579)}"
    )
    print(
        f"  FP-uniform={_fmt(row.delta_fp_minus_uniform)} "
        f"[{_fmt(row.delta_lo90)}, {_fmt(row.delta_hi90)}]  "
        f"incomplete-capped-FP={int(row.incomplete_capped_fp)}  "
        f"eligible={'yes' if eligible else 'no'}"
    )
    print(
        "  gates: "
        f"FP>=.95={'pass' if row.gate_fp_ge_095 else 'fail'}, "
        f"FP-cap={'pass' if row.gate_no_incomplete_capped_fp else 'fail'}, "
        f"uniform[.50,.75]={'pass' if row.gate_uniform_050_075 else 'fail'}, "
        f"halves[.35,.85]={'pass' if row.gate_each_half_035_085 else 'fail'}, "
        f"delta-lo>.05={'pass' if row.gate_delta_lo_gt_005 else 'fail'}"
    )
    print(f"selection: {'k=40' if eligible else 'no selection'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", help="the frozen k40 fp+uniform parquet")
    parser.add_argument("--csv", help="optional path for the one-row summary CSV")
    parser.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            if args.csv:
                _fail("--csv is unavailable with --validate-only")
            load_artifact(args.parquet)
            print("validated 1 k40 operating-point artifact")
            return 0
        summary, eligible = analyze_artifact(args.parquet)
        print_summary(summary, eligible)
        if args.csv:
            output = Path(args.csv)
            output.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(output, index=False)
            print(f"wrote {output}")
        return 0
    except (Op2ReaderError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
