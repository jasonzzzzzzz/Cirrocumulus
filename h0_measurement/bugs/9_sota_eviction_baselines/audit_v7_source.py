#!/usr/bin/env python3
"""Label-blind source audit for a possible Qwen LongBench-v2 V7.

The audit counts every complete official prompt at several fixed context caps,
then removes the 117 rows already assigned to V4--V6 and every connected
context/question component that touches one of those rows.  It never reads an
answer to select, filter, group, or count a row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h0_measurement import audit_longbench_v2 as SOURCE_AUDIT  # noqa: E402
from h0_measurement import audit_longbench_v2_qwen as QWEN_AUDIT  # noqa: E402
from sievelib import forced_choice as FC  # noqa: E402
from sievelib import tasks_longbench_v2 as LB  # noqa: E402


THRESHOLDS = (40_960, 49_152, 65_536, 81_920, 98_304, 131_072)
SCAFFOLD_TOKENS = len(QWEN_AUDIT.SCAFFOLD_TOKEN_IDS)
MAX_ELIGIBLE_INPUT = max(THRESHOLDS) - SCAFFOLD_TOKENS
# One token beyond the largest eligible prompt.  A tokenizer result equal to
# this sentinel has true length >= the sentinel and is therefore ineligible at
# every threshold; a shorter result is the exact, untruncated prompt length.
TRUNCATION_SENTINEL = MAX_ELIGIBLE_INPUT + 1
CACHE_VERSION = "longbench_v2_v7_qwen_source_lengths_v1"


def _same(expected: Any, actual: Any, label: str) -> None:
    if expected != actual:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _render_length(tokenizer: Any, item: Mapping[str, Any]) -> tuple[int, bool]:
    prompt = LB.render_prompt(dict(item))
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        truncation=True,
        max_length=TRUNCATION_SENTINEL,
    )
    token_ids = FC.token_list(rendered)
    length = len(token_ids)
    if length > TRUNCATION_SENTINEL:
        raise ValueError("tokenizer ignored the frozen truncation sentinel")
    return length, length == TRUNCATION_SENTINEL


def build_lengths(
    data: Sequence[Mapping[str, Any]], tokenizer: Any
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in data:
        length, at_sentinel = _render_length(tokenizer, item)
        rows.append(
            {
                "id": str(item["_id"]),
                "input_tokens": length,
                "at_truncation_sentinel": at_sentinel,
                "context_hash": LB.context_hash(dict(item)),
                "question_hash": QWEN_AUDIT.sha256_text(
                    str(item["question"]).strip()
                ),
            }
        )
    rows.sort(key=lambda row: str(row["id"]))
    return rows


def validate_lengths(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 503:
        raise ValueError(f"length cache has {len(rows)} rows, expected 503")
    ids = [str(row.get("id")) for row in rows]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("length-cache IDs must be unique and sorted")
    required = {
        "id", "input_tokens", "at_truncation_sentinel",
        "context_hash", "question_hash",
    }
    for row in rows:
        if set(row) != required:
            raise ValueError("length-cache row schema drifted")
        length = int(row["input_tokens"])
        sentinel = bool(row["at_truncation_sentinel"])
        if length < 1 or length > TRUNCATION_SENTINEL:
            raise ValueError(f"invalid token length for {row['id']}")
        if sentinel != (length == TRUNCATION_SENTINEL):
            raise ValueError(f"invalid truncation sentinel for {row['id']}")
        for field in ("context_hash", "question_hash"):
            value = str(row[field])
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"invalid {field} for {row['id']}")


def load_legacy_cache(path: str | Path) -> list[dict[str, Any]]:
    """Read the temporary tuple cache produced by the exact audit command.

    This option exists only to reproduce the written table without retokenizing.
    The authoritative command omits it and computes lengths from the pinned
    source and tokenizer.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("legacy cache must contain one JSON list")
    rows = [
        {
            "id": str(item[0]),
            "input_tokens": int(item[1]),
            "at_truncation_sentinel": bool(item[2]),
            "context_hash": str(item[3]),
            "question_hash": str(item[4]),
        }
        for item in raw
    ]
    rows.sort(key=lambda row: str(row["id"]))
    validate_lengths(rows)
    return rows


def summarize(
    rows: Sequence[Mapping[str, Any]], old_ids: set[str]
) -> list[dict[str, int]]:
    output: list[dict[str, int]] = []
    for threshold in THRESHOLDS:
        eligible = [
            row
            for row in rows
            if not bool(row["at_truncation_sentinel"])
            and int(row["input_tokens"]) + SCAFFOLD_TOKENS <= threshold
        ]
        missing_old = old_ids - {str(row["id"]) for row in eligible}
        if missing_old:
            raise ValueError(
                f"threshold {threshold} excludes frozen old IDs: {sorted(missing_old)}"
            )
        by_component, all_components = SOURCE_AUDIT.connected_component_ids(eligible)
        contaminated = {by_component[item_id] for item_id in old_ids}
        unused = [row for row in eligible if str(row["id"]) not in old_ids]
        _, unused_components = SOURCE_AUDIT.connected_component_ids(unused)
        clean = [
            row
            for row in unused
            if by_component[str(row["id"])] not in contaminated
        ]
        _, clean_components = SOURCE_AUDIT.connected_component_ids(clean)
        output.append(
            {
                "threshold": threshold,
                "all_rows": len(eligible),
                "all_components": len(all_components),
                "unused_rows": len(unused),
                "unused_components": len(unused_components),
                "old_linked_rows_removed": len(unused) - len(clean),
                "clean_rows": len(clean),
                "clean_components": len(clean_components),
            }
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    default_snapshot = (
        ROOT / ".hf_cache/hub/"
        "models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/"
        f"{QWEN_AUDIT.MODEL_REVISION}"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default=str(ROOT / ".h0_corpus/longbench_v2/data-2b48e494.json")
    )
    parser.add_argument(
        "--source-manifest", default=str(HERE / "longbench_v2_manifest.json")
    )
    parser.add_argument(
        "--qwen-manifest", default=str(HERE / "longbench_v2_qwen30_manifest.json")
    )
    parser.add_argument("--model-snapshot", default=str(default_snapshot))
    parser.add_argument(
        "--legacy-length-cache",
        help="reproduce the table from the exact temporary tuple cache",
    )
    return parser


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = build_parser().parse_args()
    tokenizer = QWEN_AUDIT.load_tokenizer(args.model_snapshot)
    manifest, _ = QWEN_AUDIT.authenticate_manifest(
        args.qwen_manifest,
        dataset_path=args.dataset,
        source_manifest_path=args.source_manifest,
        tokenizer=tokenizer,
    )
    old_ids = {str(row["id"]) for row in manifest["examples"]}
    _same(117, len(old_ids), "old V4--V6 ID count")
    if args.legacy_length_cache:
        rows = load_legacy_cache(args.legacy_length_cache)
    else:
        data = LB.load_dataset(args.dataset, authenticate=True)
        rows = build_lengths(data, tokenizer)
        validate_lengths(rows)

    table = summarize(rows, old_ids)
    print(
        "| context cap | all complete rows/components | unused rows/components | "
        "old-linked rows removed | leakage-clean rows/components |"
    )
    print("|---:|---:|---:|---:|---:|")
    for row in table:
        print(
            f"| {row['threshold']:,} | {row['all_rows']}/{row['all_components']} | "
            f"{row['unused_rows']}/{row['unused_components']} | "
            f"{row['old_linked_rows_removed']} | "
            f"{row['clean_rows']}/{row['clean_components']} |"
        )
    print(f"truncation_sentinel={TRUNCATION_SENTINEL}")
    print(f"rows_at_sentinel={sum(bool(row['at_truncation_sentinel']) for row in rows)}")
    print(f"length_table_sha256={_canonical_hash(rows)}")


if __name__ == "__main__":
    main()
