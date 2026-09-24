#!/usr/bin/env python3
"""Focused CPU contract tests for the pinned LongBench-v2 audit manifest."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections import UserDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h0_measurement import audit_longbench_v2 as AUDIT  # noqa: E402
from sievelib import tasks_longbench_v2 as LB  # noqa: E402


class FakeBatchEncoding(UserDict):
    pass


class FakeTokenizer:
    chat_template = "fake-test-template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        prompt = messages[0]["content"]
        marker = "TOKEN_COUNT="
        start = prompt.index(marker) + len(marker)
        count = int(prompt[start:].split()[0])
        return FakeBatchEncoding(
            {"input_ids": list(range(count)), "attention_mask": [1] * count}
        )


def record(
    item_id: str,
    *,
    context: str,
    question: str,
    length: str = "short",
) -> dict[str, str]:
    return {
        "_id": item_id,
        "domain": "Single-Document QA",
        "sub_domain": "Academic",
        "difficulty": "hard",
        "length": length,
        "question": question,
        "choice_A": "alpha",
        "choice_B": "bravo",
        "choice_C": "charlie",
        "choice_D": "delta",
        "answer": "A",
        "context": context,
    }


def test_official_prompt_parser_and_stripped_hash_are_pinned():
    assert hashlib.sha256(LB.OFFICIAL_PROMPT.encode()).hexdigest() == (
        LB.OFFICIAL_PROMPT_SHA256
    )
    assert not LB.OFFICIAL_PROMPT.endswith("\n")
    item = record(
        "one", context="  same rendered context\n", question="TOKEN_COUNT=7 query"
    )
    prompt = LB.render_prompt(item)
    assert "<text>\nsame rendered context\n</text>" in prompt
    assert LB.context_hash(item) == hashlib.sha256(
        b"same rendered context"
    ).hexdigest()
    assert LB.parse_answer("**The correct answer is (C)**") == "C"
    assert LB.parse_answer("The correct answer is D") == "D"
    assert LB.parse_answer("(A)") is None
    assert LB.score_response("The correct answer is (B)", "B")["score"] == 1.0


def test_batchencoding_is_unwrapped_and_token_cap_is_input_only():
    tokenizer = FakeTokenizer()
    item = record("one", context="doc", question="TOKEN_COUNT=7 query")
    assert LB.model_input_ids(tokenizer, item) == list(range(7))

    data = [
        item,
        record("two", context="other", question="TOKEN_COUNT=8 query"),
        record(
            "medium", context="skip", question="TOKEN_COUNT=1 query", length="medium"
        ),
        record("over", context="skip", question="TOKEN_COUNT=10 query"),
    ]
    examples, counts = AUDIT.collect_eligible(data, tokenizer, max_input_tokens=8)
    assert [row["id"] for row in examples] == ["one", "two"]
    assert counts == {
        "dataset_rows": 4,
        "official_short_rows": 3,
        "short_rows_over_token_cap": 1,
        "eligible_rows": 2,
        "eligible_context_groups": 2,
    }


def test_context_or_question_components_never_cross_splits():
    tokenizer = FakeTokenizer()
    data = [
        record("a", context=" shared context ", question="TOKEN_COUNT=5 qa"),
        record("b", context="shared context", question="TOKEN_COUNT=6 qb"),
        record("c", context="context c", question="TOKEN_COUNT=7 duplicate"),
        record("d", context="context d", question="TOKEN_COUNT=7 duplicate"),
        record("e", context="context e", question="TOKEN_COUNT=8 unique"),
    ]
    examples, _ = AUDIT.collect_eligible(data, tokenizer, max_input_tokens=20)
    component_by_id, members = AUDIT.connected_component_ids(examples)
    assert len(members) == 3
    assert component_by_id["a"] == component_by_id["b"]
    assert component_by_id["c"] == component_by_id["d"]
    assert component_by_id["a"] != component_by_id["c"]

    emitted, order = AUDIT.assign_splits(
        examples,
        component_by_id=component_by_id,
        members_by_component=members,
        qualification_stop=1,
        development_stop=2,
    )
    assert len(order) == 3
    by_id = {row["id"]: row for row in emitted}
    assert by_id["a"]["split"] == by_id["b"]["split"]
    assert by_id["c"]["split"] == by_id["d"]["split"]
    for left, right in (("qualification", "development"),
                        ("qualification", "confirmation"),
                        ("development", "confirmation")):
        left_rows = [row for row in emitted if row["split"] == left]
        right_rows = [row for row in emitted if row["split"] == right]
        assert not ({row["group_id"] for row in left_rows} &
                    {row["group_id"] for row in right_rows})
        assert not ({row["context_hash"] for row in left_rows} &
                    {row["context_hash"] for row in right_rows})


def test_manifest_is_gold_free_self_authenticating_and_deterministic():
    tokenizer = FakeTokenizer()
    data = [
        record("a", context="same", question="TOKEN_COUNT=5 qa"),
        record("b", context="same", question="TOKEN_COUNT=6 qb"),
        record("c", context="c", question="TOKEN_COUNT=7 duplicate"),
        record("d", context="d", question="TOKEN_COUNT=7 duplicate"),
        record("e", context="e", question="TOKEN_COUNT=8 qe"),
    ]
    kwargs = dict(
        transformers_version=LB.TRANSFORMERS_VERSION,
        tokenizers_version=LB.TOKENIZERS_VERSION,
        strict_reference=False,
        qualification_stop=1,
        development_stop=2,
    )
    first = AUDIT.build_manifest(data, tokenizer, **kwargs)
    second = AUDIT.build_manifest(list(reversed(data)), tokenizer, **kwargs)
    assert first == second
    AUDIT.verify_manifest_content_hash(first)
    assert first["counts"]["eligible_rows"] == 5
    assert first["counts"]["eligible_split_components"] == 3
    assert first["eligible_list_sha256"] == hashlib.sha256(
        b"a\t5\nb\t6\nc\t7\nd\t7\ne\t8\n"
    ).hexdigest()
    assert set(first["examples"][0]) == {
        "id", "group_id", "context_hash", "input_tokens", "metadata", "split"
    }
    serialized = json.dumps(first)
    assert '"answer"' not in serialized
    assert '"question_hash"' not in serialized

    corrupted = json.loads(json.dumps(first))
    corrupted["examples"][0]["input_tokens"] += 1
    try:
        AUDIT.verify_manifest_content_hash(corrupted)
        assert False, "content drift must fail authentication"
    except ValueError:
        pass


def test_safe_manifest_write_is_atomic_idempotent_and_refuses_drift():
    manifest = AUDIT.seal_manifest(
        {"manifest_version": "test", "protocol": {}, "examples": []}
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.json"
        assert AUDIT.write_manifest(path, manifest) == "written"
        first = path.read_bytes()
        assert AUDIT.write_manifest(path, manifest) == "unchanged"
        assert path.read_bytes() == first

        changed = AUDIT.seal_manifest(
            {"manifest_version": "changed", "protocol": {}, "examples": []}
        )
        try:
            AUDIT.write_manifest(path, changed)
            assert False, "differing output must require --force"
        except FileExistsError:
            pass
        assert AUDIT.write_manifest(path, changed, force=True) == "written"
        assert json.loads(path.read_text())["manifest_version"] == "changed"

        link = Path(tmp) / "link.json"
        link.symlink_to(path)
        try:
            AUDIT.write_manifest(link, changed, force=True)
            assert False, "symlink output must be refused"
        except FileExistsError:
            pass


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS: {len(tests)} LongBench-v2 audit tests")
