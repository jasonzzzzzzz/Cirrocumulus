#!/usr/bin/env python3
"""CPU-only contract tests for R8's opt-in multikey panel runner."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "h0_measurement"))
from sievelib import compress as C, tasks_ruler as TR  # noqa: E402
import run_r8 as RR  # noqa: E402


def panel_fixture(prompt_idx=700):
    meta = {
        "task": "niah_multikey",
        "task_variant": RR.PANEL_VARIANT,
        "task_generation_version": TR.MULTIKEY_PANEL_VERSION,
        "panel_version": TR.MULTIKEY_PANEL_VERSION,
        "panel_rng_version": TR.MULTIKEY_PANEL_RNG_VERSION,
        "panel_rng_namespaces": dict(TR.MULTIKEY_PANEL_RNG_NAMESPACES),
        "key_suffix_contract": TR.MULTIKEY_PANEL_KEY_SUFFIX_CONTRACT,
        "prompt_idx": prompt_idx,
        "n_clusters": 4,
        "cluster_size": 12,
        "n_queries": 4,
        "n_needles": 48,
        "target_needle_ranks": tuple(TR.MULTIKEY_PANEL_TARGET_RANKS),
        "target_needle_depths": tuple(TR.MULTIKEY_PANEL_TARGET_DEPTHS),
        "needle_depths": tuple(round(.03 + i * .019, 4) for i in range(48)),
        "synthetic": False,
        "doc": f"doc-{prompt_idx}",
        "offset": prompt_idx,
        "spliced": False,
        "corpus_sha": "a" * 64,
    }
    queries = []
    for qi in range(4):
        target = f"{7_100_000 + qi}"
        dvals = tuple(str(7_200_000 + qi * 100 + j) for j in range(11))
        queries.append({
            "query_idx": qi,
            "cluster": qi,
            "target_cluster": qi,
            "question": f"\n\nquestion-{qi} answer is",
            "expected": (target,),
            "target_key": f"cluster-{qi}-target",
            "target_value": target,
            "target_needle_rank": TR.MULTIKEY_PANEL_TARGET_RANKS[qi],
            "target_needle_depth": TR.MULTIKEY_PANEL_TARGET_DEPTHS[qi],
            "distractors": tuple(dvals) + tuple(
                str(8_000_000 + qi * 100 + j) for j in range(36)),
            "distractor_keys": tuple(f"cluster-{qi}-d{j}" for j in range(11)),
            "distractor_values": dvals,
        })
    return "shared panel context", meta, tuple(queries)


def test_mode_and_names():
    cfg = TR.task_config(48, 4, 4)
    assert RR.result_stem("m", 32768, cfg) == "r8_m_32768_k48_v4_h4"
    assert RR.result_stem("m", 32768, cfg, RR.PANEL_VARIANT) == (
        "r8_m_32768_k48_v4_h4_multikey_panel_v1")
    assert RR.validate_panel_mode(
        RR.PANEL_VARIANT, ["niah_multikey"], cfg, ctx=32768,
        question_agnostic=True, head_error=False, routes="", write_routes="",
        arms=["fp", "uniform"])
    bad = [
        dict(ctx=16384), dict(question_agnostic=False),
        dict(head_error=True), dict(routes="routes.json"),
        dict(arms=["fp", "router_oracle"]),
    ]
    base = dict(ctx=32768, question_agnostic=True, head_error=False,
                routes="", write_routes="", arms=["fp", "uniform"])
    for change in bad:
        args = {**base, **change}
        try:
            RR.validate_panel_mode(RR.PANEL_VARIANT, ["niah_multikey"], cfg, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"panel mode accepted invalid contract {change}")


def test_allocation_hashes_and_provenance():
    fp = RR.allocation_id(None)
    assert fp == hashlib.sha256(b"fp16").hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", fp)
    bits = {1: torch.tensor([[0, 2, 2]], dtype=torch.uint8),
            0: torch.tensor([[2, 2, 2]], dtype=torch.long)}
    same = {0: bits[0].clone(), 1: bits[1].clone()}
    changed = {0: bits[0].clone(), 1: bits[1].clone()}
    changed[1][0, 0] = 1
    assert RR.allocation_id(bits) == RR.allocation_id(same)
    assert RR.allocation_id(bits) != RR.allocation_id(changed)

    context, meta, queries = panel_fixture()
    chash = RR.validate_panel_payload(context, meta, queries, 700)
    assert chash == hashlib.sha256(context.encode("utf-8")).hexdigest()
    rows = [RR.panel_row_provenance(meta, q, chash, RR.allocation_id(bits))
            for q in queries]
    assert {r["allocation_id"] for r in rows} == {RR.allocation_id(bits)}
    assert [r["query_idx"] for r in rows] == [0, 1, 2, 3]
    for row in rows:
        assert len(json.loads(row["panel_distractor_keys"])) == 11
        assert len(json.loads(row["panel_distractor_values"])) == 11
        assert " " not in row["panel_distractor_keys"]

    sidecar = RR.panel_sidecar_record(
        meta, expected_accuracy_rows=320, expected_policy_rows=0)
    assert sidecar["query_count"] == 4
    assert sidecar["expected_accuracy_rows"] == 320
    assert sidecar["fp_allocation_sentinel"] == fp
    assert sidecar["allocation_id_algorithm"] == RR.ALLOCATION_ID_ALGORITHM


def test_one_prefill_one_precompute_and_query_isolation():
    """Exercise the production schedule with CPU fakes around model kernels."""
    context, meta, queries = panel_fixture()

    class Enc:
        def __init__(self, ids):
            self.input_ids = torch.tensor([ids], dtype=torch.long)

    class Tok:
        def __init__(self):
            self.answers = {900 + qi: q["target_value"] for qi, q in enumerate(queries)}
        def __call__(self, text, **kwargs):
            if text == context:
                return Enc(list(range(1, 101)))
            qi = int(text.split("question-")[1].split()[0])
            return Enc([200 + qi, 300 + qi])
        def decode(self, ids):
            vals = list(ids)
            return self.answers.get(vals[0], "") if vals else ""

    counts = {"prefill": 0, "precompute": 0, "crop": [], "question": []}

    class Past(dict):
        def get_seq_length(self):
            return self["len"]
    originals = {
        "builder": TR.build_multikey_panel,
        "prefill": RR.prefill,
        "precompute": RR.precompute,
        "crop": C.crop_to,
        "apply": C.apply_bits,
        "question": RR._question,
        "decode": RR._decode,
    }

    def fake_builder(*args, **kwargs):
        return context, meta, queries

    def fake_prefill(model, ids, window, chunk, h2o=False):
        counts["prefill"] += 1
        C.STATE.ctx_len = ids.shape[1] - 1 - window
        return Past(len=ids.shape[1] - 1), ids.shape[1]

    def fake_precompute(*args, **kwargs):
        counts["precompute"] += 1
        return {("uniform", 2): {0: torch.full((1, C.STATE.ctx_len), 2,
                                                       dtype=torch.uint8)}}, {}, {}, {}

    def fake_crop(past, length):
        counts["crop"].append((past["len"], length))
        past["len"] = length

    def fake_apply(past, by_layer, R, norm_correct):
        C.STATE.reset_arm()
        C.STATE.bits = {li: b.clone() for li, b in by_layer.items()}
        C.STATE.enabled = True

    def fake_question(model, past, q_ids):
        assert past["len"] == 100, "question did not start at shared context boundary"
        qi = int(q_ids[0, 0]) - 200
        counts["question"].append(qi)
        past["query_idx"] = qi
        past["len"] += q_ids.shape[1] - 1
        return past

    def fake_decode(model, past, first, max_new, eos, tok):
        qi = past["query_idx"]
        past["len"] += 3
        return [900 + qi], past

    try:
        TR.build_multikey_panel = fake_builder
        RR.prefill = fake_prefill
        RR.precompute = fake_precompute
        C.crop_to = fake_crop
        C.apply_bits = fake_apply
        RR._question = fake_question
        RR._decode = fake_decode
        rows, policy, returned_meta = RR.run_panel_prompt(
            object(), Tok(), prompt_idx=700, ctx=32768, corpus="/corpus",
            require_real=True, task_cfg=TR.task_config(48, 4, 4),
            plan=[("fp", 0), ("uniform", 2)], arms=["fp", "uniform"],
            budgets=[2], want={"uniform"}, bls={}, wo=None, R=torch.eye(1),
            norm_correct=True, maxb=8, bit_list=[1, 2, 3, 4, 5, 6, 8],
            n_layers=1, cascade_bits=4, eos=set(), max_new=24,
            window=32, chunk=128, dev=torch.device("cpu"), model_tag="m",
            model_id="model-id", native_ctx=32768, rot_seed=0, theta=1.0,
            policy_requested=False, policy_candidates=[], policy_trace_steps=8)
    finally:
        TR.build_multikey_panel = originals["builder"]
        RR.prefill = originals["prefill"]
        RR.precompute = originals["precompute"]
        C.crop_to = originals["crop"]
        C.apply_bits = originals["apply"]
        RR._question = originals["question"]
        RR._decode = originals["decode"]
        C.STATE.reset_prompt()

    assert counts["prefill"] == 1
    assert counts["precompute"] == 1
    assert counts["question"] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert len(counts["crop"]) == 8
    assert all(destination == 100 for _, destination in counts["crop"])
    assert any(source > 100 for source, _ in counts["crop"][1:])
    assert len(rows) == 8 and not policy and returned_meta is meta
    assert all(row["first_ok"] == 1.0 for row in rows)
    for (_, group) in __import__("itertools").groupby(
            sorted(rows, key=lambda r: (r["arm"], r["B"], r["query_idx"])),
            key=lambda r: (r["arm"], r["B"])):
        assert len({row["allocation_id"] for row in group}) == 1


def main():
    tests = [test_mode_and_names, test_allocation_hashes_and_provenance,
             test_one_prefill_one_precompute_and_query_isolation]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} panel runner tests")


if __name__ == "__main__":
    main()
