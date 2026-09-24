#!/usr/bin/env python3
"""Focused CPU contracts for the fresh V7 Qwen LongBench-v2 manifest."""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h0_measurement import audit_longbench_v2_qwen_v7 as V7  # noqa: E402
from sievelib import tasks_longbench_v2 as LB  # noqa: E402

DATA = ROOT / ".h0_corpus/longbench_v2/data-2b48e494.json"
LEGACY = ROOT / "h0_measurement/bugs/9_sota_eviction_baselines/longbench_v2_qwen30_manifest.json"
CANONICAL = ROOT / "h0_measurement/bugs/9_sota_eviction_baselines/longbench_v2_qwen30_v7_manifest.json"
SNAPSHOT = (
    ROOT
    / ".hf_cache/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots"
    / V7.MODEL_REVISION
)


def expect_error(action, contains: str) -> None:
    try:
        action()
        assert False, "invalid V7 contract was accepted"
    except V7.V7ManifestError as caught:
        assert contains in str(caught), str(caught)


def test_frozen_public_constants_and_endpoint_sampling_formula():
    assert V7.COMPONENT_ID_NAMESPACE == b"longbench_v2_sieve_v7_qwen131072_component_v1\0"
    assert V7.SELECTION_NAMESPACE == b"longbench_v2_sieve_v7_qwen131072_selection_v1\0"
    assert V7.CONTEXT_WINDOW == 131_072
    assert V7.SCAFFOLD_TOKEN_IDS == (785, 4396, 4226, 374, 320)
    assert V7.CONTENT_GROUPS == ("question", "A", "B", "C", "D")
    assert V7.SPLIT_ROWS == {"qualification": 20, "development": 52, "confirmation": 45}
    assert V7.evenly_spaced_positions([11]) == [11]
    assert V7.evenly_spaced_positions(list(range(10))) == [0, 1, 2, 3, 5, 6, 7, 9]
    assert V7.evenly_spaced_positions(list(range(8))) == list(range(8))
    expect_error(lambda: V7.evenly_spaced_positions([]), "nonempty")
    expect_error(lambda: V7.evenly_spaced_positions([2, 2]), "strictly increasing")


def test_transitive_component_partition_and_single_salted_order():
    def item(item_id: str, context: str, question: str):
        return {
            "_id": item_id,
            "context": context,
            "question": question,
        }

    rows = [
        item("a", "ctx-1", "q-1"),
        item("b", "ctx-1", "q-2"),
        item("c", "ctx-3", "q-2"),
        item("d", "ctx-4", "q-4"),
    ]
    by_id, members = V7.connected_components(rows)
    assert by_id["a"] == by_id["b"] == by_id["c"]
    assert len(members[by_id["a"]]) == 3
    assert members[by_id["d"]] == ["d"]

    candidates = [{"id": name} for name in "abcdef"]
    observed = V7.salted_order(candidates, V7.SELECTION_NAMESPACE)
    expected = sorted(
        "abcdef",
        key=lambda value: (
            hashlib.sha256(V7.SELECTION_NAMESPACE + value.encode("ascii")).digest(),
            value,
        ),
    )
    assert observed == expected


class CharacterOffsetTokenizer:
    """Tiny exact-offset chat tokenizer for span mapping unit tests."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert add_generation_prompt is True
        assert enable_thinking is False
        text = "<user>" + messages[0]["content"] + "</user><assistant>"
        if tokenize:
            return [ord(character) for character in text]
        return text

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping, return_attention_mask):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        assert return_attention_mask is False
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _synthetic_item() -> dict[str, str]:
    return {
        "_id": "synthetic",
        "domain": "Code Repository Understanding",
        "sub_domain": "repo",
        "difficulty": "hard",
        "length": "medium",
        "question": "ABCDEFGHIJK",
        "choice_A": "choice alpha text",
        "choice_B": "choice beta text",
        "choice_C": "choice gamma text",
        "choice_D": "choice delta text",
        "answer": "A",
        "context": "context",
    }


def test_chat_render_offsets_produce_five_disjoint_absolute_groups():
    tokenizer = CharacterOffsetTokenizer()
    item = _synthetic_item()
    ids, groups = V7.prompt_ids_and_content_positions(tokenizer, item)
    assert set(groups) == set(V7.CONTENT_GROUPS)
    assert all(1 <= len(groups[name]) <= 8 for name in V7.CONTENT_GROUPS)
    flattened = [position for name in V7.CONTENT_GROUPS for position in groups[name]]
    assert len(flattened) == len(set(flattened))
    assert max(flattened) < len(ids) - 1
    # Eleven question characters select the endpoint-inclusive floor positions.
    prompt, spans = V7.prompt_with_content_spans(item)
    chat_start = len("<user>")
    question_positions = list(range(
        chat_start + spans["question"][0],
        chat_start + spans["question"][1],
    ))
    assert groups["question"] == V7.evenly_spaced_positions(question_positions)
    assert prompt == LB.render_prompt(item)


@functools.lru_cache(maxsize=1)
def _actual_manifest() -> dict:
    tokenizer = V7.load_tokenizer(str(SNAPSHOT))
    return V7.build_manifest(
        dataset_path=DATA,
        legacy_manifest_path=LEGACY,
        tokenizer=tokenizer,
    )


def test_actual_cpu_build_has_expected_fresh_partition_and_no_outcomes():
    manifest = _actual_manifest()
    V7.validate_manifest_schema(manifest, enforce_frozen=False)
    assert manifest["counts"] == {
        "dataset_rows": 503,
        "all_components": 447,
        "legacy_rows": 117,
        "tainted_components": 108,
        "tainted_rows": 121,
        "untainted_singletons": 324,
        "singleton_rows_over_cap": 172,
        "eligible_singletons": 152,
        "selected_rows": 117,
    }
    examples = manifest["examples"]
    assert len(examples) == len({row["id"] for row in examples}) == 117
    assert len(examples) == len({row["group_id"] for row in examples})
    assert {split: sum(row["split"] == split for row in examples) for split in V7.SPLITS} == V7.SPLIT_ROWS
    legacy_ids = {
        row["id"] for row in json.loads(LEGACY.read_text(encoding="utf-8"))["examples"]
    }
    assert not (legacy_ids & {row["id"] for row in examples})
    for row in examples:
        assert not (V7.FORBIDDEN_EXAMPLE_KEYS & set(row))
        assert row["input_tokens"] + len(V7.SCAFFOLD_TOKEN_IDS) <= V7.CONTEXT_WINDOW
        flattened = [
            position
            for group in V7.CONTENT_GROUPS
            for position in row["content_token_positions"][group]
        ]
        assert len(flattened) == len(set(flattened))
        assert max(flattened) < row["input_tokens"] - 1
    encoded = json.dumps(manifest, sort_keys=True).lower()
    assert '"answer"' not in encoded
    assert '"gold"' not in encoded
    assert '"label"' not in encoded


def test_manifest_schema_rejects_rehashed_overlap_and_final_token():
    original = _actual_manifest()
    bad = copy.deepcopy(original)
    row = bad["examples"][0]
    row["content_token_positions"]["A"] = [row["content_token_positions"]["question"][0]]
    bad["split_counts"] = V7._split_summaries(bad["examples"])
    bad["content_sha256"] = V7.manifest_content_sha256(bad)
    expect_error(lambda: V7.validate_manifest_schema(bad, enforce_frozen=False), "overlap")

    bad = copy.deepcopy(original)
    row = bad["examples"][0]
    row["content_token_positions"]["D"] = [row["input_tokens"] - 1]
    bad["split_counts"] = V7._split_summaries(bad["examples"])
    bad["content_sha256"] = V7.manifest_content_sha256(bad)
    expect_error(
        lambda: V7.validate_manifest_schema(bad, enforce_frozen=False),
        "final uncached prompt token",
    )


def test_canonical_file_matches_builder_when_present():
    if not CANONICAL.exists():
        return
    expected = _actual_manifest()
    observed = json.loads(CANONICAL.read_text(encoding="utf-8"))
    assert observed == expected
    assert V7.sha256_file(CANONICAL) == V7.FROZEN_MANIFEST_FILE_SHA256
    assert observed["content_sha256"] == V7.FROZEN_MANIFEST_CONTENT_SHA256


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS: {len(tests)} V7 Qwen manifest tests")
