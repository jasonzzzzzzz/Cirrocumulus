#!/usr/bin/env python3
"""Focused CPU contract tests for the V3 contrastive multikey panel."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sievelib import tasks_ruler as TR  # noqa: E402


class PanelTokenizer:
    """A tiny tokenizer with an explicit common-prefix/one-suffix contract."""

    _cluster_names = {"alpha": 0, "bravo": 1, "charlie": 2, "delta": 3}

    def __init__(self):
        self._suffix_ids = {}

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text.startswith("panel cluster "):
            pieces = text.split(" ")
            cluster = self._cluster_names[pieces[2]]
            suffix = pieces[-1]
            if suffix not in self._suffix_ids:
                self._suffix_ids[suffix] = 1000 + len(self._suffix_ids)
            return SimpleNamespace(input_ids=[10, 100 + cluster, 20,
                                              self._suffix_ids[suffix]])
        return SimpleNamespace(input_ids=list(range(1, max(2, len(text) // 4))))

    @staticmethod
    def decode(ids):
        return "".join(f"<H{int(token)}>" for token in ids)


@contextmanager
def panel_env():
    calls = []

    def haystack(tok, ctx, corpus_dir, hay_key, require_real):
        calls.append((ctx, corpus_dir, hay_key, require_real))
        return list(range(1, 4001)), {
            "synthetic": False,
            "source": "corpus",
            "doc": f"book-{hay_key}.txt",
            "offset": hay_key * 10,
            "spliced": False,
            "corpus_sha": "panel-corpus",
            "n_haystack_tokens": 4000,
        }

    old_resolve = TR.prompts.resolve_corpus_dir
    old_haystack = TR.prompts._build_haystack
    TR.prompts.resolve_corpus_dir = lambda path: path
    TR.prompts._build_haystack = haystack
    try:
        yield PanelTokenizer(), calls
    finally:
        TR.prompts.resolve_corpus_dir = old_resolve
        TR.prompts._build_haystack = old_haystack


def _plain_queries(queries):
    return tuple(dict(query) for query in queries)


def test_panel_is_deterministic_exact_and_immutable():
    with panel_env() as (tok, calls):
        first = TR.build_multikey_panel(tok, 32768, prompt_idx=700,
                                        corpus_dir="/corpus", require_real=True)
        second = TR.build_multikey_panel(tok, 32768, prompt_idx=700,
                                         corpus_dir="/corpus", require_real=True)
        context, meta, queries = first

        assert first[0] == second[0]
        assert first[1] == second[1]
        assert _plain_queries(first[2]) == _plain_queries(second[2])
        assert calls[0] == calls[1]
        assert calls[0][2] == 700 + TR.prompts._seed(
            TR.MULTIKEY_PANEL_RNG_NAMESPACES["haystack"])

        assert meta["task"] == "niah_multikey"
        assert meta["task_variant"] == "multikey_panel_v1"
        assert meta["panel_version"] == "contrastive_multikey_panel_v1"
        assert meta["n_clusters"] == 4
        assert meta["cluster_size"] == 12
        assert meta["n_needles"] == 48
        assert meta["n_queries"] == 4
        assert len(meta["needle_keys"]) == len(meta["needle_values"]) == 48
        assert len(set(meta["needle_keys"])) == len(set(meta["needle_values"])) == 48
        assert all(len(value) == 7 and value.isdigit()
                   for value in meta["needle_values"])
        assert context.count("One of the special magic numbers for ") == 48
        assert len(queries) == 4

        for query in queries:
            assert len(query["distractors"]) == 47
            assert query["distractors"] == query["all_distractor_values"]
            assert len(query["distractor_keys"]) == 11
            assert len(query["distractor_values"]) == 11
            assert query["target_value"] not in query["distractors"]
            try:
                query["query_idx"] = 9
                assert False, "query mappings must be immutable"
            except TypeError:
                pass


def test_suffix_contrast_is_tokenizer_validated_per_cluster():
    with panel_env() as (tok, _):
        _, meta, _ = TR.build_multikey_panel(tok, 32768, prompt_idx=701)
        assert meta["key_suffix_contract"] == (
            "token_ids(key)==common_prefix_token_ids+[unique_suffix_token_id]"
        )
        for cluster in range(4):
            indices = [i for i, c in enumerate(meta["needle_clusters"])
                       if c == cluster]
            assert len(indices) == 12
            expected_prefix = tuple(meta["cluster_common_prefix_token_ids"][cluster])
            suffix_ids = []
            for index in indices:
                ids = TR._plain_token_ids(tok, meta["needle_keys"][index])
                assert ids[:-1] == expected_prefix
                assert ids[-1] == meta["needle_suffix_token_ids"][index]
                suffix_ids.append(ids[-1])
            assert len(set(suffix_ids)) == 12


def test_depth_slots_ranks_and_cluster_rotation_are_fixed():
    with panel_env() as (tok, _):
        ranks = (5, 17, 30, 42)
        depths = (0.15, 0.38, 0.62, 0.85)
        for prompt_idx in range(700, 704):
            context, meta, queries = TR.build_multikey_panel(
                tok, 32768, prompt_idx=prompt_idx)
            rotation = prompt_idx % 4
            assert meta["target_needle_ranks"] == ranks
            assert meta["target_needle_depths"] == depths
            assert meta["cluster_rotation"] == rotation
            assert tuple(meta["needle_depths"][rank] for rank in ranks) == depths
            assert all(a < b for a, b in zip(meta["needle_depths"],
                                              meta["needle_depths"][1:]))
            assert meta["cluster_to_query_idx"] == tuple(
                (cluster + rotation) % 4 for cluster in range(4))
            assert meta["query_idx_to_cluster"] == tuple(
                (slot - rotation) % 4 for slot in range(4))

            for slot, query in enumerate(queries):
                assert query["query_idx"] == slot
                assert query["target_cluster"] == (slot - rotation) % 4
                assert query["target_needle_rank"] == ranks[slot]
                assert query["target_needle_depth"] == depths[slot]
                assert meta["needle_values"][ranks[slot]] == query["target_value"]
                values_in_text_order = sorted(meta["needle_values"],
                                              key=context.index)
                assert values_in_text_order[ranks[slot]] == query["target_value"]


def test_distractor_extension_cannot_change_targets():
    with panel_env() as (tok, _):
        short = TR._panel_cluster_identities(tok, 702, n_distractors=11)
        extended = TR._panel_cluster_identities(tok, 702, n_distractors=25)
        target_fields = ("cluster", "key", "value", "suffix_literal",
                         "suffix_token_id", "common_prefix_token_ids")
        for a, b in zip(short, extended):
            assert {key: a["target"][key] for key in target_fields} == {
                key: b["target"][key] for key in target_fields}
            assert len(a["distractors"]) == 11
            assert len(b["distractors"]) == 25

        _, _, queries = TR.build_multikey_panel(tok, 32768, prompt_idx=702)
        rotation = 702 % 4
        for query in queries:
            cluster = query["target_cluster"]
            assert query["target_key"] == short[cluster]["target"]["key"]
            assert query["target_value"] == short[cluster]["target"]["value"]
            assert query["query_idx"] == (cluster + rotation) % 4


def test_legacy_builder_bytes_and_constants_remain_pinned():
    class GoldenTokenizer:
        @staticmethod
        def decode(ids):
            return "".join(f"<{int(i):03d}>" for i in ids)

    old_resolve = TR.prompts.resolve_corpus_dir
    old_haystack = TR.prompts._build_haystack
    TR.prompts.resolve_corpus_dir = lambda _: "/fake"
    TR.prompts._build_haystack = lambda tok, ctx, corpus, p, require: (
        list(range(1, 201)),
        {"synthetic": True, "doc": "golden", "offset": p},
    )
    try:
        text, _ = TR.build(GoldenTokenizer(), "niah_multikey", 2048,
                           prompt_idx=3)
    finally:
        TR.prompts.resolve_corpus_dir = old_resolve
        TR.prompts._build_haystack = old_haystack
    assert hashlib.sha256(text.encode()).hexdigest() == (
        "055feec60ee218e36d438a8f546ccd9c4efb4dc5ffa35dc5ba72267895fde88d"
    )
    assert TR.TASKS == ("niah_single", "niah_multikey",
                        "niah_multivalue", "vt")
    assert TR.DEFAULT_TASK_CONFIG == {"n_keys": 4, "n_values": 4, "n_hops": 4}
    assert TR.TASK_GENERATION_VERSION == "ruler_pg19_v1"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS: {len(tests)} multikey-panel tests")
