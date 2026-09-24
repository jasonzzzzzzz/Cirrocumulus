#!/usr/bin/env python3
"""Authenticate and analyze the frozen R10 tier-set experiment.

The reader has four deliberately separate entry points:

* ``--preflight`` authenticates the source ledger before a GPU is used.
* ``--validate-task-dir`` authenticates one task artifact.  The Slurm worker
  uses ``--allow-incomplete`` after the parquet is written and before it writes
  the atomic COMPLETE sentinel.
* ``--pilot`` applies the excluded implementation-control gates and writes the
  only lock which authorizes the main array.
* ``--main`` applies the frozen scientific gates.  A valid design failure is a
  successful analysis; an artifact/provenance failure exits nonzero and is
  labelled ``invalid_r10``.

All statistical definitions and thresholds come from the protocol frozen in
``plan.md`` before R10 model output existed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import stat
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
DEFAULT_RESULTS = PROJECT / "h0_measurement" / "results"
DEFAULT_LEDGER = HERE / "source_ledger.json"
CORPUS_MANIFEST = PROJECT / ".h0_corpus" / "pg19" / "MANIFEST.json"
PROTOCOL = "r10_tier_set_v1"
COMPLETE_TEXT = PROTOCOL + "\n"
LEDGER_VERSION = "r10_tier_set_source_ledger_v1"

TIERS: dict[str, tuple[int, ...]] = {
    "full": (0, 1, 2, 3, 4, 5, 6, 8),
    "no1": (0, 2, 3, 4, 5, 6, 8),
    "base3_dense": (0, 3, 4, 5, 6, 8),
    "nested3": (0, 3, 4, 6, 8),
    "nested4": (0, 4, 6, 8),
}
LABELS = tuple(TIERS)
BUDGETS = (2, 3)
BIT_LIST = (1, 2, 3, 4, 5, 6, 8)
STEP = 4
BOOTSTRAP_DRAWS = 10_000

REQUIRED_LEDGER_SOURCES = {
    "sievelib/alloc.py",
    "sievelib/quant.py",
    "sievelib/evict.py",
    "sievelib/prompts.py",
    "sievelib/probe.py",
    "h0_measurement/run_h0.py",
    "h0_measurement/models.yaml",
    "h0_measurement/prefetch_corpus.py",
    "h0_measurement/submit_r10_tier_set.slurm",
    ".h0_corpus/pg19/MANIFEST.json",
    "h0_measurement/bugs/10_tier_set_rederivation/plan.md",
    "h0_measurement/bugs/10_tier_set_rederivation/read_tier_set.py",
    "h0_measurement/bugs/10_tier_set_rederivation/script.sh",
}


@dataclass(frozen=True)
class TaskSpec:
    model: str
    ctx: int
    n_prompts: int
    prompt_offset: int
    families: tuple[str, ...]
    mha_control: bool = False


PILOT_TASKS = {
    0: TaskSpec("qwen3-1.7b", 2048, 1, 0, ("cont",)),
    1: TaskSpec("qwen15-moe-a2.7b", 8192, 1, 0, ("cont",), True),
}
MAIN_TASKS = {
    0: TaskSpec("llama31-8b", 32768, 6, 18, ("niah", "qa", "cont")),
    1: TaskSpec("llama31-8b", 131072, 6, 18, ("niah", "qa", "cont")),
    2: TaskSpec("qwen3-8b", 8192, 6, 18, ("niah", "qa", "cont")),
    3: TaskSpec("qwen3-30b-a3b-2507", 8192, 4, 12,
                ("niah", "qa", "cont")),
}

MODEL_PROVENANCE = {
    "qwen3-1.7b": {
        "model_id": "Qwen/Qwen3-1.7B",
        "model_cache": "models--Qwen--Qwen3-1.7B",
        "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "proxy_id": "Qwen/Qwen3-0.6B",
        "proxy_cache": "models--Qwen--Qwen3-0.6B",
        "proxy_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    },
    "qwen15-moe-a2.7b": {
        "model_id": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "model_cache": "models--Qwen--Qwen1.5-MoE-A2.7B-Chat",
        "model_revision": "ec052fda178e241c7c443468d2fa1db6618996be",
        "proxy_id": "Qwen/Qwen1.5-0.5B-Chat",
        "proxy_cache": "models--Qwen--Qwen1.5-0.5B-Chat",
        "proxy_revision": "4d14e384a4b037942bb3f3016665157c8bcb70ea",
    },
    "llama31-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "model_cache": "models--meta-llama--Llama-3.1-8B-Instruct",
        "model_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "proxy_id": "meta-llama/Llama-3.2-1B-Instruct",
        "proxy_cache": "models--meta-llama--Llama-3.2-1B-Instruct",
        "proxy_revision": "9213176726f574b556790deb65791e0c5aa438b6",
    },
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "model_cache": "models--Qwen--Qwen3-8B",
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "proxy_id": "Qwen/Qwen3-0.6B",
        "proxy_cache": "models--Qwen--Qwen3-0.6B",
        "proxy_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    },
    "qwen3-30b-a3b-2507": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "model_cache": "models--Qwen--Qwen3-30B-A3B-Instruct-2507",
        "model_revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "proxy_id": "Qwen/Qwen3-0.6B",
        "proxy_cache": "models--Qwen--Qwen3-0.6B",
        "proxy_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    },
}


class InvalidR10(RuntimeError):
    """An authentication or completeness failure, never a design verdict."""


def _fail(message: str) -> None:
    raise InvalidR10(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _regular(path: Path, what: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        _fail(f"missing {what}: {path}")
    if not stat.S_ISREG(mode):
        _fail(f"{what} must be a regular file (no symlink): {path}")
    return path


def _read_json(path: Path, what: str) -> dict[str, Any]:
    _regular(path, what)
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {what} {path}: {exc}")
    if not isinstance(obj, dict):
        _fail(f"{what} must contain a JSON object: {path}")
    return obj


def ledger_content_sha256(version: str, sources: Mapping[str, str]) -> str:
    """Established source-ledger hash: version plus the ordered source map."""
    payload = {"ledger_version": version, "source_sha256": dict(sources)}
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(wire).hexdigest()


def verify_source_ledger(
    path: Path,
    *,
    project_root: Path = PROJECT,
    required_sources: set[str] = REQUIRED_LEDGER_SOURCES,
) -> dict[str, Any]:
    ledger = _read_json(path, "source ledger")
    if set(ledger) != {"ledger_version", "content_sha256", "source_sha256"}:
        _fail("source ledger needs exactly ledger_version, content_sha256, and "
              "source_sha256")
    if ledger["ledger_version"] != LEDGER_VERSION:
        _fail(f"wrong source-ledger version {ledger['ledger_version']!r}; "
              f"expected {LEDGER_VERSION!r}")
    sources = ledger["source_sha256"]
    if not isinstance(sources, dict) or not sources:
        _fail("source_sha256 must be a non-empty path->SHA256 object")
    missing = sorted(required_sources - set(sources))
    if missing:
        _fail(f"source ledger is missing required sources: {missing}")
    want_content = ledger_content_sha256(ledger["ledger_version"], sources)
    if ledger["content_sha256"] != want_content:
        _fail("source-ledger content_sha256 does not authenticate its version "
              "and source_sha256 map")
    for rel, expected in sorted(sources.items()):
        if (not isinstance(rel, str) or not rel or Path(rel).is_absolute()
                or ".." in Path(rel).parts):
            _fail(f"unsafe source-ledger path: {rel!r}")
        if not isinstance(expected, str) or len(expected) != 64:
            _fail(f"invalid SHA256 for {rel!r}: {expected!r}")
        src = _regular(project_root / rel, f"ledger source {rel}")
        observed = sha256_file(src)
        if observed != expected:
            _fail(f"source drift for {rel}: ledger={expected}, current={observed}")
    return ledger


def expected_corpus_sha() -> str:
    manifest = _read_json(CORPUS_MANIFEST, "frozen PG-19 manifest")
    value = manifest.get("corpus_sha")
    if not isinstance(value, str) or not value:
        _fail("frozen PG-19 manifest has no corpus_sha")
    return value


def _same(observed: Any, expected: Any, what: str) -> None:
    if observed != expected:
        _fail(f"{what}: expected {expected!r}, observed {observed!r}")


def _as_int(value: Any, what: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        _fail(f"{what} must be an integer, got {value!r}")
    return out


def _finite(frame: pd.DataFrame, columns: Iterable[str], what: str) -> None:
    cols = list(columns)
    missing = [c for c in cols if c not in frame]
    if missing:
        _fail(f"missing {what} columns: {missing}")
    bad = ~np.isfinite(frame[cols].to_numpy(dtype=float))
    if bool(bad.any()):
        where = np.argwhere(bad)[0]
        _fail(f"non-finite {what}: row {frame.index[where[0]]}, "
              f"column {cols[where[1]]}")


def metric_prefix(label: str, budget: int) -> str:
    return f"grp_tier_{label}_csv_b4_accum_{budget}"


def err_col(label: str, budget: int) -> str:
    return "err_wf_" + metric_prefix(label, budget)


def mean_col(label: str, budget: int) -> str:
    return "mean_bits_" + metric_prefix(label, budget)


def evict_col(label: str, budget: int) -> str:
    return "evict_frac_" + metric_prefix(label, budget)


def tier_col(label: str, budget: int, bit: int) -> str:
    return f"tier_frac_{metric_prefix(label, budget)}_b{bit}"


def _normalise_panel(value: Any) -> dict[str, tuple[int, ...]] | None:
    if value == "r10":
        return TIERS
    if isinstance(value, Mapping):
        try:
            return {str(k): tuple(int(x) for x in v) for k, v in value.items()}
        except (TypeError, ValueError):
            return None
    return None


def validate_sidecar(side: Mapping[str, Any], spec: TaskSpec) -> None:
    provenance = MODEL_PROVENANCE[spec.model]
    _same(side.get("model"), spec.model, "sidecar model")
    _same(side.get("model_id"), provenance["model_id"], "sidecar model ID")
    _same(_as_int(side.get("ctx"), "sidecar ctx"), spec.ctx, "sidecar ctx")
    _same(side.get("kind"), "h0", "sidecar kind")
    _same(tuple(int(x) for x in side.get("budgets", ())), BUDGETS,
          "sidecar budgets")
    _same(tuple(int(x) for x in side.get("bit_list", ())), BIT_LIST,
          "sidecar bit_list")
    _same(_as_int(side.get("maxb"), "sidecar maxb"), 8, "sidecar maxb")
    _same(_as_int(side.get("quant_every"), "sidecar quant_every"), 4,
          "sidecar quant_every")
    _same(_as_int(side.get("n_prompts"), "sidecar n_prompts"), spec.n_prompts,
          "sidecar n_prompts")
    _same(_as_int(side.get("n_decode"), "sidecar n_decode"), 8,
          "sidecar n_decode")
    _same(_as_int(side.get("prompt_offset"), "sidecar prompt_offset"),
          spec.prompt_offset, "sidecar prompt_offset")
    _same(side.get("prompt_block"),
          [spec.prompt_offset, spec.prompt_offset + spec.n_prompts - 1],
          "sidecar prompt block")
    _same(_as_int(side.get("rot_seed"), "sidecar rot_seed"), 2,
          "sidecar rot_seed")
    _same(tuple(int(x) for x in side.get("measure_steps", ())), tuple(range(8)),
          "sidecar dense probe-row steps")
    _same(side.get("schedule"), "dense", "sidecar decode schedule")
    _same(float(side.get("decode_temperature", math.nan)), 0.0,
          "sidecar decode temperature")
    _same(float(side.get("decode_top_p", math.nan)), 1.0,
          "sidecar decode top-p")
    _same(side.get("interior_unseen_policy"), "floor_maxb",
          "sidecar unseen-position policy")
    _same(side.get("group_alloc"), True, "sidecar physical group allocation")
    _same(tuple(int(x) for x in side.get("coarse_bits", ())), (4,),
          "sidecar cascade observation width")
    _same(tuple(int(x) for x in side.get("extra_budgets", ())), BUDGETS,
          "sidecar group/cascade budgets")
    _same(bool(side.get("synthetic", True)), False, "sidecar real-corpus flag")
    _same(side.get("corpus_sha"), expected_corpus_sha(),
          "sidecar frozen PG-19 corpus hash")
    _same(bool(side.get("norm_correct", False)), True,
          "sidecar TurboQuant norm correction")

    cfg = side.get("config")
    if not isinstance(cfg, Mapping):
        _fail("sidecar config must be an object")
    _same(tuple(cfg.get("families", ())), spec.families, "configured families")
    _same(bool(cfg.get("group_alloc", False)), True,
          "configured physical group allocation")
    _same(tuple(int(x) for x in cfg.get("coarse_bits", ())), (4,),
          "configured cascade observation width")
    _same(tuple(int(x) for x in cfg.get("extra_budgets", ())), BUDGETS,
          "configured extra budgets")
    panel = _normalise_panel(side.get("tier_panel", cfg.get("tier_panel")))
    if panel != TIERS:
        _fail(f"sidecar tier panel is not the frozen R10 panel: {panel!r}")
    _same(side.get("tier_panel_score"), "csv_b4_accum",
          "tier-panel score")
    _same(cfg.get("validate_with"), provenance["proxy_id"],
          "sidecar validation-proxy ID")

    corner = side.get("corner")
    if not isinstance(corner, Mapping):
        _fail("sidecar has no eviction-corner record")
    _same(tuple(corner.get("evictor_labels", ())), ("oracle", "accum"),
          "evictor labels")
    _same(tuple(corner.get("practical", ())), ("accum",),
          "practical corner")
    _same(tuple(corner.get("policies", ())), ("frac",), "corner policy")


def validate_run_info(info: Mapping[str, Any], phase: str, task: int,
                      job: str | None, spec: TaskSpec,
                      ledger_file_sha: str) -> None:
    _same(info.get("protocol"), PROTOCOL, "RUN_INFO protocol")
    _same(info.get("phase"), phase, "RUN_INFO phase")
    _same(_as_int(info.get("task"), "RUN_INFO task"), task, "RUN_INFO task")
    if job is not None:
        _same(str(info.get("job")), str(job), "RUN_INFO job")
    _same(info.get("model"), spec.model, "RUN_INFO model")
    _same(_as_int(info.get("ctx"), "RUN_INFO ctx"), spec.ctx, "RUN_INFO ctx")
    _same(_as_int(info.get("n_prompts"), "RUN_INFO n_prompts"), spec.n_prompts,
          "RUN_INFO n_prompts")
    _same(_as_int(info.get("prompt_offset"), "RUN_INFO prompt_offset"),
          spec.prompt_offset, "RUN_INFO prompt offset")
    _same(tuple(info.get("families", ())), spec.families, "RUN_INFO families")
    provenance = MODEL_PROVENANCE[spec.model]
    for key in ("model_cache", "model_revision", "proxy_cache", "proxy_revision"):
        _same(info.get(key), provenance[key], f"RUN_INFO {key}")
    expected_overrides = [
        f"ctx={spec.ctx}", f"n_prompts={spec.n_prompts}",
        f"prompt_offset={spec.prompt_offset}",
        f"families={','.join(spec.families)}", "n_decode=8", "quant_every=4",
        "rot_seed=2", "bit_list=1,2,3,4,5,6,8", "budgets=2,3",
        "evictors=oracle,accum", "corner_policies=frac",
        "interior_scores=accum", "group_alloc=true", "coarse_bits=4",
        "extra_budgets=2,3", "tier_panel=r10",
    ]
    _same(info.get("overrides"), expected_overrides, "RUN_INFO exact overrides")
    _same(info.get("source_ledger_sha256"), ledger_file_sha,
          "RUN_INFO source-ledger file hash")


def _check_frame_grid(df: pd.DataFrame, spec: TaskSpec) -> None:
    identity = ["model", "ctx", "prompt", "prompt_offset", "family", "step",
                "layer", "head", "L", "synthetic", "quantized", "norm_correct",
                "rot_seed", "corpus_sha"]
    missing = [c for c in identity if c not in df]
    if missing:
        _fail(f"parquet is missing identity columns: {missing}")
    if len(df) == 0:
        _fail("parquet contains no rows")
    if set(df["model"].astype(str)) != {spec.model}:
        _fail(f"row model identity differs from {spec.model}")
    if set(df["ctx"].astype(int)) != {spec.ctx}:
        _fail(f"row context identity differs from {spec.ctx}")
    expected_prompts = set(range(spec.prompt_offset,
                                 spec.prompt_offset + spec.n_prompts))
    if set(df["prompt"].astype(int)) != expected_prompts:
        _fail("row prompt block is incomplete or contains an unplanned prompt")
    if set(df["prompt_offset"].astype(int)) != {spec.prompt_offset}:
        _fail("row prompt_offset differs from the frozen task")
    if set(df["family"].astype(str)) != set(spec.families):
        _fail("row family set differs from the frozen task")
    if set(df["step"].astype(int)) != set(range(8)):
        _fail("dense row step set must be exactly 0..7")
    if set(df["rot_seed"].astype(int)) != {2}:
        _fail("row quantizer rotation seed is not 2")
    if bool(df["synthetic"].astype(bool).any()):
        _fail("synthetic input is invalid for R10")
    quantized = df["quantized"].astype(bool)
    expected_quantized = df["step"].astype(int).isin((0, STEP))
    if not bool((quantized == expected_quantized).all()):
        _fail("quantized rows must be exactly dense steps 0 and 4")
    if not bool(df["norm_correct"].astype(bool).all()):
        _fail("R10 artifact did not use TurboQuant norm correction")
    if df["corpus_sha"].isna().any() or (df["corpus_sha"].astype(str) == "").any():
        _fail("one or more rows lack corpus provenance")

    key = ["prompt", "family", "step", "layer", "head"]
    if bool(df.duplicated(key).any()):
        _fail("duplicate prompt/family/step/layer/head rows")
    topology: set[tuple[int, int]] | None = None
    for _, block in df.groupby(["prompt", "family", "step"], sort=False):
        got = set(zip(block["layer"].astype(int), block["head"].astype(int)))
        if topology is None:
            topology = got
        elif got != topology:
            _fail("layer/head topology is incomplete in at least one prompt cell")
    expected_blocks = spec.n_prompts * len(spec.families) * 8
    if df.groupby(["prompt", "family", "step"]).ngroups != expected_blocks:
        _fail("prompt/family/step grid is incomplete")



def validate_panel_rows(df4: pd.DataFrame, *, mha_control: bool) -> None:
    """V1, V2, V5 and primary-value checks on step-4 rows."""
    if df4.empty:
        _fail("no step-4 rows")
    _finite(df4, ["n_rep", "kv_head", "grp_size"],
            "step-4 physical-group identity")
    nrep = df4["n_rep"].astype(int)
    kv = df4["kv_head"].astype(int)
    if (nrep < 1).any() or not bool((df4["head"].astype(int) // nrep == kv).all()):
        _fail("step-4 head->physical-KV-head mapping is inconsistent")
    if (df4["grp_size"].astype(int) < 1).any():
        _fail("invalid step-4 group size")
    for budget in BUDGETS:
        base_err = f"err_wf_grp_csv_b4_accum_{budget}"
        base_evict = f"evict_frac_grp_csv_b4_accum_{budget}"
        endpoint = f"err_e{budget}_grp_pp_accum_frac"
        _finite(df4, [f"err_uniform{budget}", base_err, base_evict, endpoint],
                f"budget-{budget} matched endpoint")

        fm = f"alloc_mismatch_full_vs_existing_csv_b4_accum_{budget}"
        fd = f"max_bit_diff_full_vs_existing_csv_b4_accum_{budget}"
        _finite(df4, [fm, fd], "full duplicate diagnostics")
        if not bool((df4[[fm, fd]].to_numpy(dtype=float) == 0).all()):
            _fail(f"V1 full allocation differs from existing path at B={budget}")

        all_summary_cols: list[str] = []
        for label, tiers in TIERS.items():
            ec, mc, vc = (err_col(label, budget), mean_col(label, budget),
                          evict_col(label, budget))
            fcols = [tier_col(label, budget, b) for b in tiers]
            _finite(df4, [ec, mc, vc, *fcols],
                    f"{label}/B={budget} primary")
            fr = df4[fcols].to_numpy(dtype=float)
            if bool(((fr < -1e-12) | (fr > 1 + 1e-12)).any()):
                _fail(f"tier fraction outside [0,1] for {label}/B={budget}")
            if not np.allclose(fr.sum(axis=1), 1.0, rtol=0, atol=2e-9):
                _fail(f"tier fractions do not sum to one for {label}/B={budget}")
            weighted = fr @ np.asarray(tiers, dtype=float)
            if not np.allclose(weighted, df4[mc].to_numpy(float),
                               rtol=0, atol=2e-9):
                _fail(f"tier fractions do not reproduce mean bits for "
                      f"{label}/B={budget}")
            if not np.allclose(fr[:, tiers.index(0)], df4[vc].to_numpy(float),
                               rtol=0, atol=2e-9):
                _fail(f"tier-0 fraction does not reproduce eviction fraction for "
                      f"{label}/B={budget}")

            # A forbidden-tier column is itself a schema failure: it makes the
            # action set ambiguous even when all observed values happen to be 0.
            stem = f"tier_frac_{metric_prefix(label, budget)}_b"
            observed_bits = {
                int(c[len(stem):]) for c in df4 if c.startswith(stem)
                and c[len(stem):].isdigit()
            }
            if observed_bits != set(tiers):
                _fail(f"tier columns for {label}/B={budget} are {observed_bits}; "
                      f"expected {set(tiers)}")
            all_summary_cols.extend([mc, vc, *fcols])

            if mha_control:
                mm = ("alloc_mismatch_grp_vs_head_tier_"
                      f"{label}_csv_b4_accum_{budget}")
                md = ("max_bit_diff_grp_vs_head_tier_"
                      f"{label}_csv_b4_accum_{budget}")
                _finite(df4, [mm, md], "MHA identity diagnostics")
                if not bool((df4[[mm, md]].to_numpy(float) == 0).all()):
                    _fail(f"V5 n_rep=1 identity fails for {label}/B={budget}")

        full_err = err_col("full", budget)
        full_evict = evict_col("full", budget)
        if not np.allclose(df4[full_err], df4[base_err], rtol=1e-12,
                           atol=1e-14):
            _fail(f"V1 full exact error differs from existing path at B={budget}")
        if not np.allclose(df4[full_evict], df4[base_evict], rtol=0,
                           atol=1e-14):
            _fail(f"V1 full eviction fraction differs at B={budget}")

        # Allocation summaries are repeated on every query head.  Check the
        # copies before selecting one physical (layer, KV-head) row.
        gkey = ["prompt", "family", "step", "layer", "kv_head"]
        for _, group in df4.groupby(gkey, sort=False):
            vals = group[all_summary_cols].to_numpy(dtype=float)
            if not np.allclose(vals, vals[:1], rtol=0, atol=2e-12):
                _fail("query heads sharing a physical KV head disagree on the "
                      "tier allocation summary")
            declared = set(group["grp_size"].astype(int))
            if len(declared) != 1 or len(group) != next(iter(declared)):
                _fail("physical KV group row count differs from grp_size")

    if mha_control:
        if set(df4["n_rep"].astype(int)) != {1} or set(df4["grp_size"].astype(int)) != {1}:
            _fail("the declared MHA control is not n_rep=grp_size=1")

    # V2 uses one de-duplicated allocation per physical group.  The waterfill
    # can leave less than one 8-bit token of indivisible slack.
    physical = physical_group_table(df4)
    for budget in BUDGETS:
        for label in LABELS:
            mean = physical[mean_col(label, budget)].to_numpy(float)
            length = physical["L"].to_numpy(float)
            if bool((mean > budget + 1e-9).any()):
                _fail(f"V2 overspend for {label}/B={budget}")
            if bool(((budget - mean) > (8.0 / length + 1e-9)).any()):
                _fail(f"V2 excessive unused budget for {label}/B={budget}")


def physical_group_table(df4: pd.DataFrame) -> pd.DataFrame:
    key = (["cell"] if "cell" in df4.columns else []) + [
        "prompt", "family", "step", "layer", "kv_head"]
    if not set(key + ["L"]).issubset(df4.columns):
        _fail("cannot de-duplicate physical groups: missing identity column")
    return df4.sort_values(key + ["head"]).drop_duplicates(key, keep="first")


def _verify_main_pilot_lock(task_dir: Path, info: Mapping[str, Any],
                            ledger_sha: str) -> None:
    pilot_job = str(info.get("pilot_job", ""))
    if not pilot_job.isdigit():
        _fail("main RUN_INFO has no numeric pilot_job")
    local = _regular(task_dir / "PILOT_LOCK.json", "task-local pilot lock")
    local_sha = sha256_file(local)
    _same(info.get("pilot_lock_sha256"), local_sha, "RUN_INFO pilot-lock hash")
    lock = _read_json(local, "task-local pilot lock")
    _same(lock.get("protocol"), PROTOCOL, "pilot-lock protocol")
    _same(lock.get("decision"), "advance_main", "pilot-lock decision")
    _same(str(lock.get("pilot_job")), pilot_job, "pilot-lock job")
    _same(lock.get("source_ledger_sha256"), ledger_sha,
          "pilot-lock source-ledger hash")
    arts = lock.get("task_artifacts")
    if not isinstance(arts, Mapping) or set(arts) != {"0", "1"}:
        _fail("pilot lock does not authenticate both pilot tasks")
    canonical = HERE / f"pilot_advance_lock_{pilot_job}.json"
    _regular(canonical, "canonical pilot lock")
    if canonical.read_bytes() != local.read_bytes():
        _fail("task-local pilot lock is not byte-identical to the canonical lock")


def validate_task_dir(task_dir: Path, phase: str, task: int, *,
                      source_ledger: Path, job: str | None = None,
                      allow_incomplete: bool = False) -> dict[str, Any]:
    specs = PILOT_TASKS if phase == "pilot" else MAIN_TASKS
    if task not in specs:
        _fail(f"no frozen {phase} task {task}")
    spec = specs[task]
    if not task_dir.is_dir():
        _fail(f"missing task directory: {task_dir}")
    if job is None:
        prefix, suffix = f"r10_tiers_{phase}_", f"_{task}"
        if not task_dir.name.startswith(prefix) or not task_dir.name.endswith(suffix):
            _fail(f"task directory does not match {prefix}<JOB>{suffix}: {task_dir.name}")
        job = task_dir.name[len(prefix):-len(suffix)]
        if not job.isdigit():
            _fail(f"task directory has a non-numeric job ID: {task_dir.name}")
    expected_name = f"r10_tiers_{phase}_{job}_{task}"
    if task_dir.name != expected_name:
        _fail(f"task directory must be named {expected_name}, got {task_dir.name}")

    verify_source_ledger(source_ledger)
    canonical_ledger_sha = sha256_file(source_ledger)
    local_ledger = _regular(task_dir / "SOURCE_LEDGER.json",
                            "task-local source ledger")
    if local_ledger.read_bytes() != source_ledger.read_bytes():
        _fail("task-local source ledger is not byte-identical to canonical ledger")
    local_ledger_sha = sha256_file(local_ledger)
    info = _read_json(task_dir / "RUN_INFO.json", "RUN_INFO")
    validate_run_info(info, phase, task, job, spec, local_ledger_sha)
    if phase == "main":
        _verify_main_pilot_lock(task_dir, info, canonical_ledger_sha)

    complete = task_dir / "COMPLETE"
    if complete.exists():
        _regular(complete, "COMPLETE sentinel")
        if complete.read_text() != COMPLETE_TEXT:
            _fail(f"wrong COMPLETE sentinel in {task_dir}")
    elif not allow_incomplete:
        _fail(f"missing COMPLETE sentinel in {task_dir}")

    parquets = list(task_dir.glob("h0_*.parquet"))
    if len(parquets) != 1:
        _fail(f"expected exactly one h0 parquet in {task_dir}, found {len(parquets)}")
    parquet = _regular(parquets[0], "H0 parquet")
    side_path = parquet.with_suffix(".json")
    side = _read_json(side_path, "H0 sidecar")
    _same(side.get("parquet"), parquet.name, "sidecar parquet basename")
    validate_sidecar(side, spec)
    try:
        df = pd.read_parquet(parquet)
    except Exception as exc:
        _fail(f"cannot read parquet footer/data {parquet}: {exc}")
    _same(_as_int(side.get("rows"), "sidecar rows"), len(df),
          "sidecar/parquet row count")
    _check_frame_grid(df, spec)
    if set(df["corpus_sha"].astype(str)) != {str(side["corpus_sha"])}:
        _fail("row corpus hashes differ from the sidecar corpus hash")
    df4 = df.loc[df["step"].astype(int) == STEP].copy()
    validate_panel_rows(df4, mha_control=spec.mha_control)
    if phase == "main":
        niah = df4.loc[df4["family"].astype(str) == "niah"]
        if niah.empty or "needle_hit" not in niah:
            _fail("V4 main task has no NIAH validity rows")
        prompt_hits = niah.groupby("prompt")["needle_hit"].agg(
            lambda x: bool(pd.Series(x).astype(bool).all()))
        if set(prompt_hits.index.astype(int)) != set(range(
                spec.prompt_offset, spec.prompt_offset + spec.n_prompts)):
            _fail("V4 NIAH prompt set is incomplete")
        if not bool(prompt_hits.all()):
            missed = [int(p) for p, ok in prompt_hits.items() if not ok]
            _fail(f"V4 needle retrieval failed for prompts {missed}")

    return {
        "phase": phase, "task": task, "spec": spec, "dir": task_dir,
        "parquet": parquet, "sidecar": side_path, "sidecar_data": side,
        "frame": df4, "parquet_sha256": sha256_file(parquet),
        "sidecar_sha256": sha256_file(side_path),
        "source_ledger_sha256": canonical_ledger_sha,
        "run_info": info,
    }


def _gmean(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=float)
    if len(a) == 0 or bool((a <= 0).any()) or not bool(np.isfinite(a).all()):
        _fail("geometric mean received an empty, non-positive, or non-finite value")
    return float(np.exp(np.log(a).mean()))


def _rms(values: pd.Series) -> float:
    a = values.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(a * a)))


def _q90(draws: np.ndarray) -> tuple[float, float]:
    q = np.quantile(draws, (0.05, 0.95))
    return float(q[0]), float(q[1])


def _bootstrap_equal_cells(
    prompt_values: Mapping[str, np.ndarray],
    *,
    geometric_within: bool,
    geometric_across: bool,
    rng: np.random.Generator,
) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    per_cell_draws: dict[str, np.ndarray] = {}
    for cell in sorted(prompt_values):
        a = np.asarray(prompt_values[cell], dtype=float)
        if not len(a) or not np.isfinite(a).all():
            _fail(f"bootstrap input for {cell} is empty or non-finite")
        idx = rng.integers(0, len(a), size=(BOOTSTRAP_DRAWS, len(a)))
        sampled = a[idx]
        if geometric_within:
            if bool((sampled <= 0).any()):
                _fail("geometric bootstrap received a non-positive value")
            draw = np.exp(np.log(sampled).mean(axis=1))
        else:
            draw = sampled.mean(axis=1)
        per_cell_draws[cell] = draw
    stack = np.stack([per_cell_draws[c] for c in sorted(per_cell_draws)], axis=1)
    macro = (np.exp(np.log(stack).mean(axis=1)) if geometric_across
             else stack.mean(axis=1))
    return ({c: _q90(v) for c, v in per_cell_draws.items()}, _q90(macro))


def _prompt_statistics(frame: pd.DataFrame, budget: int,
                       label: str) -> pd.DataFrame:
    endpoint = np.minimum(frame[f"err_uniform{budget}"].to_numpy(float),
                          frame[f"err_e{budget}_grp_pp_accum_frac"].to_numpy(float))
    err = frame[err_col(label, budget)].to_numpy(float)
    if bool((err <= 0).any()) or not bool(np.isfinite(endpoint).all()):
        _fail(f"invalid exact error or endpoint for {label}/B={budget}")
    work = frame[["cell", "prompt"]].copy()
    work["sqerr"] = err * err
    gain = endpoint / err
    work["log_routed_gain"] = np.log(np.maximum(gain, 1.0))
    work["band"] = gain >= 2.0
    return work.groupby(["cell", "prompt"], as_index=False).agg(
        E=("sqerr", lambda x: float(np.sqrt(np.mean(x)))),
        routed_gain=("log_routed_gain", lambda x: float(np.exp(np.mean(x)))),
        band=("band", "mean"),
    )


def _allocation_cell_summary(frame: pd.DataFrame, budget: int,
                             label: str) -> dict[str, dict[str, float]]:
    phys = physical_group_table(frame)
    out: dict[str, dict[str, float]] = {}
    for cell, block in phys.groupby("cell", sort=True):
        weight = block["L"].to_numpy(float)
        den = float(weight.sum())
        row = {
            "mean_bits": float(np.dot(block[mean_col(label, budget)], weight) / den),
            "evict_frac": float(np.dot(block[evict_col(label, budget)], weight) / den),
        }
        for bit in TIERS[label]:
            row[f"tier{bit}_frac"] = float(
                np.dot(block[tier_col(label, budget, bit)], weight) / den)
        out[str(cell)] = row
    return out


def analyze_main(tasks: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    corpus_hashes = {str(x["sidecar_data"]["corpus_sha"])
                     for x in tasks if "sidecar_data" in x}
    if corpus_hashes and corpus_hashes != {expected_corpus_sha()}:
        _fail(f"main tasks do not share the frozen corpus hash: {corpus_hashes}")
    frames = []
    for item in tasks:
        f = item["frame"].copy()
        spec: TaskSpec = item["spec"]
        f["cell"] = f"{spec.model}@{spec.ctx}"
        frames.append(f)
    frame = pd.concat(frames, ignore_index=True)
    cells = sorted(frame["cell"].unique())
    if len(cells) != 4:
        _fail(f"main analysis needs four cells, observed {cells}")

    rng = np.random.Generator(np.random.PCG64(0))
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {"protocol": PROTOCOL, "bootstrap_draws": BOOTSTRAP_DRAWS,
                              "bootstrap_rng": "PCG64(0)", "budgets": {}}

    for budget in BUDGETS:
        prompt = {lab: _prompt_statistics(frame, budget, lab) for lab in LABELS}
        full = prompt["full"].set_index(["cell", "prompt"])
        bdetail: dict[str, Any] = {"labels": {}, "transitions": {}}

        for label in LABELS:
            cur = prompt[label].set_index(["cell", "prompt"])
            if not cur.index.equals(full.index):
                _fail(f"prompt alignment differs for {label}/B={budget}")
            joined = pd.DataFrame({
                "error_ratio": cur["E"] / full["E"],
                "routed_retention": cur["routed_gain"] / full["routed_gain"],
                "band_delta": cur["band"] - full["band"],
            }).reset_index()
            alloc_summary = _allocation_cell_summary(frame, budget, label)
            error_arrays = {c: g["error_ratio"].to_numpy(float)
                            for c, g in joined.groupby("cell")}
            gain_arrays = {c: g["routed_retention"].to_numpy(float)
                           for c, g in joined.groupby("cell")}
            band_arrays = {c: g["band_delta"].to_numpy(float)
                           for c, g in joined.groupby("cell")}
            err_ci, macro_err_ci = _bootstrap_equal_cells(
                error_arrays, geometric_within=True, geometric_across=True, rng=rng)
            gain_ci, macro_gain_ci = _bootstrap_equal_cells(
                gain_arrays, geometric_within=True, geometric_across=True, rng=rng)
            band_ci, macro_band_ci = _bootstrap_equal_cells(
                band_arrays, geometric_within=False, geometric_across=False, rng=rng)

            cell_values = {}
            for cell in cells:
                g = joined[joined.cell == cell]
                ps = prompt[label][prompt[label].cell == cell]
                cv = {
                    "error_ratio": _gmean(g.error_ratio),
                    "error_ratio_ci90": err_ci[cell],
                    "routed_gain": _gmean(ps.routed_gain),
                    "routed_gain_retention": _gmean(g.routed_retention),
                    "routed_gain_retention_ci90": gain_ci[cell],
                    "band": float(ps.band.mean()),
                    "band_delta": float(g.band_delta.mean()),
                    "band_delta_ci90": band_ci[cell],
                    **alloc_summary[cell],
                }
                cell_values[cell] = cv
                rows.append({
                    "scope": "cell", "budget": budget, "label": label,
                    "reference": "full", "cell": cell, **cv,
                })
            ratios = [cell_values[c]["error_ratio"] for c in cells]
            retentions = [cell_values[c]["routed_gain_retention"] for c in cells]
            bands = [cell_values[c]["band_delta"] for c in cells]
            macro = {
                "error_ratio": _gmean(ratios),
                "error_ratio_ci90": macro_err_ci,
                "median_cell_error_ratio": float(np.median(ratios)),
                "worst_cell_error_ratio": float(np.max(ratios)),
                "routed_gain_retention": _gmean(retentions),
                "routed_gain_retention_ci90": macro_gain_ci,
                "band_delta": float(np.mean(bands)),
                "band_delta_ci90": macro_band_ci,
                "worst_cell_band_delta": float(np.min(bands)),
            }
            rows.append({"scope": "macro", "budget": budget, "label": label,
                         "reference": "full", "cell": "macro", **macro})
            bdetail["labels"][label] = {"cells": cell_values, "macro": macro}

        # Fixed sequential attribution compares adjacent action spaces rather
        # than reusing each arm's ratio against full.
        chain = (("full", "no1"), ("no1", "base3_dense"),
                 ("base3_dense", "nested3"))
        first_failure = None
        for source, target in chain:
            a = prompt[source].set_index(["cell", "prompt"])
            b = prompt[target].set_index(["cell", "prompt"])
            ratio = (b["E"] / a["E"]).rename("ratio").reset_index()
            cell_ratio = {c: _gmean(g.ratio) for c, g in ratio.groupby("cell")}
            median = float(np.median(list(cell_ratio.values())))
            worst = float(np.max(list(cell_ratio.values())))
            passed = median <= 1.05 and worst <= 1.10
            name = f"{source}->{target}"
            bdetail["transitions"][name] = {
                "cell_error_ratio": cell_ratio, "median": median,
                "worst": worst, "pass": passed,
            }
            rows.append({"scope": "transition", "budget": budget,
                         "label": target, "reference": source, "cell": "macro",
                         "median_cell_error_ratio": median,
                         "worst_cell_error_ratio": worst, "pass": passed})
            if not passed and first_failure is None:
                first_failure = name
        bdetail["first_attribution_failure"] = first_failure

        nested = bdetail["labels"]["nested3"]
        nr = [nested["cells"][c]["error_ratio"] for c in cells]
        nret = [nested["cells"][c]["routed_gain_retention"] for c in cells]
        nband = [nested["cells"][c]["band_delta"] for c in cells]
        gates = {
            "G1_median_error_le_1.05": float(np.median(nr)) <= 1.05,
            "G2_worst_error_le_1.10": float(np.max(nr)) <= 1.10,
            "G3_macro_gain_retention_ge_0.95": _gmean(nret) >= 0.95,
            "G4_each_gain_retention_ge_0.90": min(nret) >= 0.90,
            "G5_each_band_loss_le_0.05": min(nband) >= -0.05,
        }
        bdetail["nested3_gates"] = gates
        bdetail["nested3_pass"] = bool(all(gates.values()))

        # Directly re-budgetable nested4 is compared with nested3, including
        # the de-duplicated physical-token occupancy of tier 3.
        p3 = prompt["nested3"].set_index(["cell", "prompt"])
        p4 = prompt["nested4"].set_index(["cell", "prompt"])
        er43 = (p4["E"] / p3["E"]).rename("ratio").reset_index()
        gr43 = (p4["routed_gain"] / p3["routed_gain"]).rename("ratio").reset_index()
        er43_cell = {c: _gmean(g.ratio) for c, g in er43.groupby("cell")}
        gr43_cell = {c: _gmean(g.ratio) for c, g in gr43.groupby("cell")}
        tier3 = {c: nested["cells"][c].get("tier3_frac", math.nan) for c in cells}
        nested4_gates = {
            "median_error_le_1.02": float(np.median(list(er43_cell.values()))) <= 1.02,
            "each_error_le_1.05": max(er43_cell.values()) <= 1.05,
            "macro_gain_retention_ge_0.98": _gmean(gr43_cell.values()) >= 0.98,
            "nested3_tier3_lt_0.05_each_cell": all(v < 0.05 for v in tier3.values()),
        }
        bdetail["nested4_vs_nested3"] = {
            "cell_error_ratio": er43_cell,
            "cell_routed_gain_retention": gr43_cell,
            "macro_routed_gain_retention": _gmean(gr43_cell.values()),
            "nested3_tier3_fraction": tier3,
            "gates": nested4_gates,
            "select_nested4": bool(all(nested4_gates.values())),
        }
        detail["budgets"][str(budget)] = bdetail

    b3 = detail["budgets"]["3"]
    b2 = detail["budgets"]["2"]
    if not b3["nested3_pass"]:
        decision = "stop_before_r11_nested3_failed_b3"
    elif b3["nested4_vs_nested3"]["select_nested4"]:
        decision = ("select_nested4_b2_b3" if b2["nested3_pass"]
                    else "select_nested4_b3_only")
    else:
        decision = ("retain_nested3_price_4bit_observation_b2_b3"
                    if b2["nested3_pass"] else
                    "retain_nested3_price_4bit_observation_b3_only")
    detail["decision"] = decision
    detail["validity"] = "valid_r10"
    return pd.DataFrame(rows), detail


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return f"[{_fmt(x[0], digits)}, {_fmt(x[1], digits)}]"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


def render_main(detail: Mapping[str, Any]) -> str:
    lines = [
        f"R10 tier-set analysis ({PROTOCOL})",
        "Validity: valid_r10 (V1-V5 all passed)",
        f"Bootstrap: {BOOTSTRAP_DRAWS:,} prompt-resampling draws, PCG64(0)",
        "",
    ]
    for budget in (3, 2):
        bd = detail["budgets"][str(budget)]
        lines += [f"B={budget}",
                  "cell | ladder | error/full | routed retention | band delta | mean bits | evict"]
        for label in LABELS:
            for cell, v in bd["labels"][label]["cells"].items():
                lines.append(" | ".join([
                    cell, label, _fmt(v["error_ratio"]),
                    _fmt(v["routed_gain_retention"]), _fmt(v["band_delta"]),
                    _fmt(v["mean_bits"]), _fmt(v["evict_frac"]),
                ]))
        lines += ["", "nested3 frozen gates:"]
        lines += [f"  {k}: {'PASS' if v else 'FAIL'}"
                  for k, v in bd["nested3_gates"].items()]
        lines.append(f"  overall: {'PASS' if bd['nested3_pass'] else 'FAIL'}")
        lines.append("sequential attribution:")
        for name, v in bd["transitions"].items():
            lines.append(f"  {name}: median={v['median']:.4f}, worst={v['worst']:.4f} "
                         f"=> {'PASS' if v['pass'] else 'FAIL'}")
        nv = bd["nested4_vs_nested3"]
        lines.append("nested4 versus nested3:")
        for k, v in nv["gates"].items():
            lines.append(f"  {k}: {'PASS' if v else 'FAIL'}")
        lines.append(f"  select_nested4: {nv['select_nested4']}")
        lines.append("")
    lines.append(f"Frozen decision: {detail['decision']}")
    return "\n".join(lines) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def _task_dir(results_root: Path, phase: str, job: str, task: int) -> Path:
    return results_root / f"r10_tiers_{phase}_{job}_{task}"


def run_pilot(job: str, results_root: Path, source_ledger: Path,
              lock_path: Path, txt_path: Path | None,
              csv_path: Path | None) -> list[dict[str, Any]]:
    tasks = [validate_task_dir(_task_dir(results_root, "pilot", job, task),
                               "pilot", task, source_ledger=source_ledger,
                               job=job)
             for task in sorted(PILOT_TASKS)]
    corpus_hashes = {str(x["sidecar_data"]["corpus_sha"]) for x in tasks}
    if corpus_hashes != {expected_corpus_sha()}:
        _fail(f"pilot tasks do not share the frozen corpus hash: {corpus_hashes}")
    ledger_sha = sha256_file(source_ledger)
    lock = {
        "protocol": PROTOCOL,
        "decision": "advance_main",
        "pilot_job": str(job),
        "source_ledger_sha256": ledger_sha,
        "task_artifacts": {
            str(x["task"]): {
                "parquet": x["parquet"].name,
                "parquet_sha256": x["parquet_sha256"],
                "sidecar_sha256": x["sidecar_sha256"],
            } for x in tasks
        },
        "checks": {
            "V1_full_identity": True,
            "V2_tier_and_budget": True,
            "V3_provenance": True,
            "V4_completeness": True,
            "V5_mha_identity": True,
        },
    }
    wire = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    _write_atomic(lock_path, wire)
    text = (f"R10 excluded pilot job {job}: PASS\n"
            f"V1 full identity: PASS\nV2 tier/budget: PASS\n"
            f"V3 provenance: PASS\nV4 completeness: PASS\n"
            f"V5 n_rep=1 identity: PASS\nadvance lock: {lock_path}\n")
    if txt_path:
        _write_atomic(txt_path, text)
    if csv_path:
        pd.DataFrame([{"job": str(job), "task": x["task"], "model": x["spec"].model,
                       "ctx": x["spec"].ctx, "status": "PASS",
                       "parquet_sha256": x["parquet_sha256"]} for x in tasks]
                     ).to_csv(csv_path, index=False)
    print(text, end="")
    return tasks


def run_main(job: str, results_root: Path, source_ledger: Path,
             txt_path: Path, csv_path: Path) -> dict[str, Any]:
    tasks = [validate_task_dir(_task_dir(results_root, "main", job, task),
                               "main", task, source_ledger=source_ledger,
                               job=job)
             for task in sorted(MAIN_TASKS)]
    table, detail = analyze_main(tasks)
    text = render_main(detail)
    _write_atomic(txt_path, text)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    json_path = txt_path.with_suffix(".json")
    _write_atomic(json_path, json.dumps(detail, indent=2, sort_keys=True) + "\n")
    print(text, end="")
    print(f"saved: {txt_path}\nsaved: {csv_path}\nsaved: {json_path}")
    return detail


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--validate-task-dir", type=Path)
    action.add_argument("--pilot", metavar="JOB")
    action.add_argument("--main", metavar="JOB")
    ap.add_argument("--phase", choices=("pilot", "main"))
    ap.add_argument("--task", type=int)
    ap.add_argument("--job")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--source-ledger", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--lock", type=Path)
    ap.add_argument("--txt", type=Path)
    ap.add_argument("--csv", type=Path)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.preflight:
            verify_source_ledger(args.source_ledger)
            print(f"PASS source ledger: {args.source_ledger} "
                  f"sha256={sha256_file(args.source_ledger)}")
            return 0
        if args.validate_task_dir is not None:
            if args.phase is None or args.task is None:
                _fail("--validate-task-dir requires --phase and --task")
            item = validate_task_dir(
                args.validate_task_dir, args.phase, args.task,
                source_ledger=args.source_ledger, job=args.job,
                allow_incomplete=args.allow_incomplete)
            print(f"PASS {args.phase} task {args.task}: {item['parquet']}")
            return 0
        if args.pilot is not None:
            lock = args.lock or HERE / f"pilot_advance_lock_{args.pilot}.json"
            txt = args.txt or HERE / f"pilot_{args.pilot}.txt"
            csv = args.csv or HERE / f"pilot_{args.pilot}_summary.csv"
            run_pilot(str(args.pilot), args.results_root, args.source_ledger,
                      lock, txt, csv)
            return 0
        if args.main is not None:
            txt = args.txt or HERE / f"main_{args.main}.txt"
            csv = args.csv or HERE / f"main_{args.main}_summary.csv"
            run_main(str(args.main), args.results_root, args.source_ledger,
                     txt, csv)
            return 0
        _fail("no action selected")
    except InvalidR10 as exc:
        print(f"INVALID_R10: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
