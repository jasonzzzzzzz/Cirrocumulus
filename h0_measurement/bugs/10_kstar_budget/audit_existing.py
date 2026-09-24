#!/usr/bin/env python3
"""Fail-closed CPU audit of the 24 canonical legacy K* artifacts.

This script deliberately reads existing scalar Parquet outputs only.  It does
not reconstruct attention logits or claim to measure a shared GQA ranking.

Typical Compute Canada invocation::

    module --force purge
    module load StdEnv/2026 python/3.14 arrow/25.0.1
    python h0_measurement/bugs/10_kstar_budget/audit_existing.py

Use ``--self-check`` for the dependency-free arithmetic checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[2]
DEFAULT_MANIFEST = HERE / "input_manifest.json"
SEALED_MANIFEST_CONTENT_SHA256 = (
    "f082668eb5f170884897dc83dd0660334945b7726c1ca66506e95db2bf4000bd"
)
ELIGIBLE_N_PRACTICAL = 3.0
SAT_EPS = 1e-12


class AuditError(RuntimeError):
    """An authenticated input or a measurement invariant did not match."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def _manifest_content_sha(manifest: dict[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_content_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _schema_record(schema: Any) -> list[dict[str, Any]]:
    return [{"name": field.name, "type": str(field.type),
             "nullable": bool(field.nullable)} for field in schema]


def _schema_sha(schema: Any) -> str:
    return hashlib.sha256(_canonical_json(_schema_record(schema))).hexdigest()


def _kstar_grid(m: int, points: int = 12) -> list[int]:
    """Copy of the legacy geometric candidate construction in alloc.py."""
    _require(m >= 1, f"invalid full keep count {m}")
    if m <= 1:
        return [1]
    n = max(2, int(points))
    ks = {1, m}
    for i in range(1, n):
        ks.add(max(1, int(round(m ** (i / (n - 1))))))
    return sorted(ks)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    _require(bool(values), "mean requested for an empty population")
    return math.fsum(values) / len(values)


def _metrics(values: list[float]) -> tuple[float, float, float]:
    _require(bool(values), "metrics requested for an empty population")
    mean_ratio = _mean(values)
    saturation = _mean(abs(value - 1.0) <= SAT_EPS for value in values)
    return mean_ratio, 1.0 - mean_ratio, saturation


def _self_check() -> None:
    expected = [1, 2, 6, 13, 31, 72, 170, 400, 942, 2218, 5220, 12288]
    _require(_kstar_grid(12288, 12) == expected,
             "12-point geometric-grid regression")
    mean_ratio, slack, saturation = _metrics([1.0, 0.5, 1.0, 0.25])
    _require(abs(mean_ratio - 0.6875) < 1e-15, "mean regression")
    _require(abs(slack - 0.3125) < 1e-15, "slack regression")
    _require(abs(saturation - 0.5) < 1e-15, "saturation regression")
    # Two synthetic n_rep=2 groups: max([.5,1]) and max([.25,.75]).
    groups = {(0, 0): 0.0, (0, 1): 0.0}
    for head, value in enumerate([0.5, 1.0, 0.25, 0.75]):
        key = (0, head // 2)
        groups[key] = max(groups[key], value)
    _require(list(groups.values()) == [1.0, 0.75],
             "physical-group max regression")
    print("PASS self-check: grid, mean/slack/saturation, and group max")


def _load_manifest(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing input manifest: {path}")
    manifest = json.loads(path.read_text())
    _require(manifest.get("schema_version") == 1,
             "input manifest schema_version must be 1")
    expected = manifest.get("manifest_content_sha256")
    actual = _manifest_content_sha(manifest)
    _require(expected == SEALED_MANIFEST_CONTENT_SHA256,
             f"manifest seal mismatch: expected "
             f"{SEALED_MANIFEST_CONTENT_SHA256}, got {expected}")
    _require(isinstance(expected, str) and expected == actual,
             f"manifest content SHA mismatch: expected {expected}, got {actual}")
    _require(manifest.get("expected_cells") == 24,
             "manifest must enumerate exactly 24 cells")
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, list) and len(artifacts) == 24,
             "manifest artifacts must contain exactly 24 entries")
    cells = [(item.get("model"), item.get("ctx")) for item in artifacts]
    _require(len(set(cells)) == 24, "manifest contains duplicate canonical cells")
    return manifest


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise AuditError(f"manifest path escapes project root: {relative}") from exc
    return path


def _check_sidecar(side: dict[str, Any], item: dict[str, Any],
                   manifest: dict[str, Any]) -> None:
    expected_common = manifest["sidecar_expect"]
    for key, expected in expected_common.items():
        _require(side.get(key) == expected,
                 f"{item['model']}@{item['ctx']}: sidecar {key!r}: "
                 f"expected {expected!r}, got {side.get(key)!r}")
    for key in ("model", "model_id", "ctx", "rows", "n_prompts"):
        expected = item[key]
        _require(side.get(key) == expected,
                 f"{item['model']}@{item['ctx']}: sidecar {key!r}: "
                 f"expected {expected!r}, got {side.get(key)!r}")
    _require(side.get("parquet") == Path(item["parquet"]).name,
             f"{item['model']}@{item['ctx']}: sidecar parquet basename mismatch")
    _require(side.get("corner") == manifest["corner_expect"],
             f"{item['model']}@{item['ctx']}: exact corner config mismatch")

    config = side.get("config")
    _require(isinstance(config, dict),
             f"{item['model']}@{item['ctx']}: missing effective config")
    config_expect = dict(manifest["config_expect"])
    config_expect.update({"tag": item["model"], "id": item["model_id"],
                          "ctx": item["ctx"], "n_prompts": item["n_prompts"]})
    for key, expected in config_expect.items():
        _require(config.get(key) == expected,
                 f"{item['model']}@{item['ctx']}: config {key!r}: "
                 f"expected {expected!r}, got {config.get(key)!r}")


def _check_constant(column: list[Any], expected: Any, label: str) -> None:
    observed = set(column)
    _require(observed == {expected},
             f"{label}: expected only {expected!r}, got {observed!r}")


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def _audit(args: argparse.Namespace) -> None:
    root = args.project_root.resolve()
    manifest = _load_manifest(args.manifest.resolve())

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise AuditError(
            "pyarrow is required. On Compute Canada: module --force purge && "
            "module load StdEnv/2026 python/3.14 arrow/25.0.1"
        ) from exc
    report_spec = manifest["canonical_report"]
    report_path = _safe_path(root, report_spec["path"])
    _require(_sha256(report_path) == report_spec["sha256"],
             "canonical bugs/2 report SHA mismatch")

    required_types = manifest["required_columns"]
    budget = int(manifest["selection"]["budget"])
    maxb = int(manifest["selection"]["maxb"])
    points = int(manifest["corner_expect"]["kstar_points"])
    tol = float(manifest["corner_expect"]["kstar_tol"])
    _require((budget, maxb, points, tol) == (3, 8, 12, 0.1),
             "this audit is sealed to B=3, maxb=8, 12 points, tolerance=0.1")

    cell_results: list[dict[str, Any]] = []
    grid_by_ctx: dict[int, list[tuple[float, float, bool]]] = defaultdict(list)
    total_eligible = 0
    total_geometric = 0
    total_abs_only = 0
    total_both = 0

    for item in manifest["artifacts"]:
        tag = f"{item['model']}@{item['ctx']} ({item['job']})"
        parquet_path = _safe_path(root, item["parquet"])
        sidecar_path = _safe_path(root, item["sidecar"])
        for path, size, expected_sha in (
            (parquet_path, item["parquet_bytes"], item["parquet_sha256"]),
            (sidecar_path, item["sidecar_bytes"], item["sidecar_sha256"]),
        ):
            _require(path.is_file(), f"{tag}: missing {path}")
            _require(path.stat().st_size == size,
                     f"{tag}: size mismatch for {path.name}")
            actual_sha = _sha256(path)
            _require(actual_sha == expected_sha,
                     f"{tag}: SHA mismatch for {path.name}: "
                     f"expected {expected_sha}, got {actual_sha}")

        side = json.loads(sidecar_path.read_text())
        _check_sidecar(side, item, manifest)
        parquet = pq.ParquetFile(parquet_path)
        schema = parquet.schema_arrow
        _require(_schema_sha(schema) == item["schema_sha256"],
                 f"{tag}: exact Arrow schema fingerprint mismatch")
        schema_types = {field.name: str(field.type) for field in schema}
        for column, expected_type in required_types.items():
            _require(schema_types.get(column) == expected_type,
                     f"{tag}: column {column!r} expected {expected_type!r}, "
                     f"got {schema_types.get(column)!r}")

        table = pq.read_table(parquet_path, columns=list(required_types))
        _require(table.num_rows == item["rows"],
                 f"{tag}: table rows do not match authenticated sidecar")
        columns = {name: table[name].to_pylist() for name in required_types}
        _check_constant(columns["model"], item["model"], f"{tag} model")
        _check_constant(columns["ctx"], item["ctx"], f"{tag} ctx")
        _check_constant(columns["synthetic"], False, f"{tag} synthetic")
        _check_constant(columns["evictors"], "oracle,accum,window,recency",
                        f"{tag} evictors")
        _check_constant(columns["corner_policies"], "frac,abs",
                        f"{tag} corner_policies")
        _check_constant(columns["corner_kappa"], 4.0, f"{tag} corner_kappa")
        _check_constant(columns["corner_floor"], 256.0, f"{tag} corner_floor")
        _check_constant(columns["kstar_tol"], 0.1, f"{tag} kstar_tol")

        ratios: list[float] = []
        per_head: dict[tuple[int, int], list[float]] = defaultdict(list)
        group_values: dict[tuple[Any, ...], list[tuple[int, float]]] = defaultdict(list)
        n_rep = int(item["n_rep"])
        eligible = 0
        for index in range(table.num_rows):
            if columns["quantized"][index] is not True:
                continue
            if columns["n_practical"][index] != ELIGIBLE_N_PRACTICAL:
                continue
            kstar_value = columns["kstar3"][index]
            if kstar_value is None:
                continue
            eligible += 1
            values = {name: columns[name][index] for name in required_types}
            for key in ("family", "prompt", "step", "layer", "head", "L",
                        "n95", "kstar3", "kstar_frac3",
                        "corner_tokens3_frac", "corner_tokens3_abs",
                        "err_e3_oracle_frac"):
                _require(values[key] is not None,
                         f"{tag}: null {key} in eligible row {index}")

            length = int(values["L"])
            full = int(values["corner_tokens3_frac"])
            absolute = int(values["corner_tokens3_abs"])
            n95 = int(values["n95"])
            kstar = int(values["kstar3"])
            _require(float(full) == values["corner_tokens3_frac"] and
                     float(absolute) == values["corner_tokens3_abs"] and
                     float(kstar) == values["kstar3"],
                     f"{tag}: non-integral keep count in row {index}")
            expected_full = max(1, int(round(budget * length / maxb)))
            _require(full == expected_full,
                     f"{tag}: fractional keep-count mismatch in row {index}")
            expected_abs = max(1, min(full, int(max(4.0 * n95, 256))))
            _require(absolute == expected_abs,
                     f"{tag}: absolute keep-count mismatch in row {index}")
            grid = _kstar_grid(full, points)
            candidates = set(grid) | {absolute}
            _require(kstar in candidates and 1 <= kstar <= full,
                     f"{tag}: K*={kstar} outside legacy candidates in row {index}")
            fraction = kstar / full
            _require(math.isclose(values["kstar_frac3"], fraction,
                                  rel_tol=0.0, abs_tol=1e-15),
                     f"{tag}: stored K*/budget mismatch in row {index}")
            error = float(values["err_e3_oracle_frac"])
            _require(math.isfinite(error) and error >= 0.0,
                     f"{tag}: invalid full-budget oracle error in row {index}")

            in_grid = kstar in grid
            is_abs = kstar == absolute
            if in_grid and is_abs:
                total_both += 1
            elif in_grid:
                total_geometric += 1
            else:
                total_abs_only += 1
            below = [candidate for candidate in candidates if candidate < full]
            effective_penultimate = max(below) if below else full
            grid_by_ctx[item["ctx"]].append(
                (grid[-2] / full, effective_penultimate / full,
                 abs(fraction - 1.0) <= SAT_EPS)
            )
            ratios.append(fraction)
            per_head[(int(values["layer"]), int(values["head"]))].append(fraction)
            group_key = (values["family"], int(values["prompt"]),
                         int(values["step"]), int(values["layer"]),
                         int(values["head"]) // n_rep)
            group_values[group_key].append((int(values["head"]), fraction))

        _require(eligible == item["eligible_rows"],
                 f"{tag}: expected {item['eligible_rows']} eligible rows, got {eligible}")
        _require(ratios, f"{tag}: no eligible rows")
        total_eligible += eligible
        legacy = statistics.median(
            statistics.median(head_values) for head_values in per_head.values()
        )

        result: dict[str, Any] = {
            "model": item["model"], "ctx": item["ctx"], "n_rep": n_rep,
            "eligible": eligible, "legacy_median": legacy,
            "query_metrics": _metrics(ratios),
        }
        if item["ctx"] == 32768:
            maxima: list[float] = []
            for group_key, head_values in group_values.items():
                heads = [head for head, _ in head_values]
                _require(len(head_values) == n_rep and len(set(heads)) == n_rep,
                         f"{tag}: incomplete/duplicate physical group {group_key}")
                base = group_key[-1] * n_rep
                _require(sorted(heads) == list(range(base, base + n_rep)),
                         f"{tag}: non-contiguous physical group {group_key}: {heads}")
                maxima.append(max(value for _, value in head_values))
            result["group_count"] = len(maxima)
            result["group_metrics"] = _metrics(maxima)
        cell_results.append(result)

    _require(sum(abs(row["legacy_median"] - 1.0) <= SAT_EPS
                 for row in cell_results) == 24,
             "legacy per-head-median reproduction is not 24/24 at 1.0")

    print(f"PASS manifest content SHA: {manifest['manifest_content_sha256']}")
    print(f"PASS canonical report SHA: {report_spec['sha256']}")
    print("PASS 24/24 exact Parquet+sidecar hashes, schemas, provenance, and corner configs")
    print(f"PASS {total_eligible:,} eligible B=3 rows: keep arithmetic, candidate membership, "
          "stored fractions, and finite full-budget errors")
    print("PASS legacy aggregation: per-head medians aggregate to K*/budget=100% in 24/24 cells")
    print(f"Candidate source counts: geometric-only={total_geometric:,}, "
          f"abs-only={total_abs_only:,}, both={total_both:,}")

    print("\n32K QUERY-HEAD AND PHYSICAL-GROUP MAX PROXY")
    print("model                       n_rep  queries  q_mean   q_slack  q_sat     "
          "groups  g_mean   g_slack  g_sat")
    order = {item["model"]: i for i, item in enumerate(manifest["model_order"])}
    rows_32k = sorted((row for row in cell_results if row["ctx"] == 32768),
                      key=lambda row: order[row["model"]])
    for row in rows_32k:
        q_mean, q_slack, q_sat = row["query_metrics"]
        g_mean, g_slack, g_sat = row["group_metrics"]
        print(f"{row['model']:<28} {row['n_rep']:>5} {row['eligible']:>8} "
              f"{q_mean:>8.6f} {q_slack:>8.6f} {_format_pct(q_sat):>9} "
              f"{row['group_count']:>7} {g_mean:>8.6f} {g_slack:>8.6f} "
              f"{_format_pct(g_sat):>9}")

    print("\nB=3 LEGACY GRID RESOLUTION OVER AUTHENTICATED ROWS")
    print("ctx      rows       geom_penult/full (min/mean/max)       "
          "saturated  effective_penult/full_on_saturated (median/mean/max)")
    for ctx in sorted(grid_by_ctx):
        values = grid_by_ctx[ctx]
        geometric = [value[0] for value in values]
        saturated = [value[1] for value in values if value[2]]
        print(f"{ctx:>6} {len(values):>9,}   "
              f"{min(geometric):.6f}/{_mean(geometric):.6f}/{max(geometric):.6f}   "
              f"{len(saturated):>9,}   "
              f"{statistics.median(saturated):.6f}/{_mean(saturated):.6f}/"
              f"{max(saturated):.6f}")

    print("\nINTERPRETATION")
    print("- The physical-group result is max(query-head K*/budget), not a directly "
          "measured shared-ranking curve.")
    print("- Max aggregation nearly saturates every n_rep=4/8 GQA group, so a naive "
          "max-over-query-head budget policy has no useful slack.")
    print("- Existing Parquets contain K*, the full-budget error, and no intermediate "
          "curve errors/rankings; direct group K* cannot be reconstructed.")
    print("- The 24/24 100% statement is a median over a coarse candidate set. It is "
          "reproduced, but it does not establish integer-resolution saturation.")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT,
                        help="project root (defaults to the script's repository)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="authenticated input manifest")
    parser.add_argument("--self-check", action="store_true",
                        help="run dependency-free arithmetic checks and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.self_check:
            _self_check()
        else:
            _audit(args)
    except (AuditError, OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
