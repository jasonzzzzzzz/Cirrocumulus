#!/usr/bin/env python3
"""CPU-only contract tests for the whole-policy reader."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
READER = ROOT / "h0_measurement/bugs/9_sota_eviction_baselines/read_policy.py"
spec = importlib.util.spec_from_file_location("read_policy", READER)
RP = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(RP)

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0


def check(name, condition, detail=""):
    global fails
    print(f"  {OK if condition else BAD}  {name} {detail}")
    if not condition:
        fails += 1


def frames():
    shared = dict(model="m", model_id="org/m", ctx=32768, native_ctx=131072,
                  task="niah_multikey", n_keys=16, n_values=4, n_hops=4,
                  question_agnostic=True, window=32, maxb=8, corpus_sha="abc123",
                  synthetic=False, rot_seed=0, norm_correct=True,
                  allocator_budget_rule="feasible")
    candidates = ("uniform", "evict", "interior")
    scores = (
        dict(uniform=0.0, evict=1.0, interior=0.0),
        dict(uniform=1.0, evict=0.0, interior=1.0),
        dict(uniform=0.0, evict=0.0, interior=1.0),
        dict(uniform=1.0, evict=1.0, interior=0.0),
    )
    # Selected policies: evict, evict, uniform (exact KL tie with interior),
    # uniform (exact KL tie with evict).
    kls = (
        dict(uniform=.3, evict=.1, interior=.2),
        dict(uniform=.2, evict=.1, interior=.3),
        dict(uniform=.1, evict=.3, interior=.1),
        dict(uniform=.1, evict=.1, interior=.3),
    )
    selected = ("evict", "evict", "uniform", "uniform")
    accuracy, diagnostic = [], []
    for prompt_idx in range(4):
        accuracy.append(dict(**shared, prompt_idx=prompt_idx, arm="fp", B=0, score=1.0))
        for order, candidate in enumerate(candidates):
            accuracy.append(dict(**shared, prompt_idx=prompt_idx, arm=candidate, B=2,
                                 score=scores[prompt_idx][candidate]))
            mean_kl = kls[prompt_idx][candidate]
            diagnostic.append(dict(
                **shared, prompt_idx=prompt_idx, B=2, candidate=candidate,
                candidate_order=order, trace_rule_version="fp_teacher_v1",
                trace_steps_requested=8, trace_len=8,
                trace_token_hash=hashlib.sha256(f"trace-{prompt_idx}".encode()).hexdigest(),
                vocab_size=128, mean_kl=mean_kl, max_kl=mean_kl + .1,
                fp_token_ce=.5, top1_agreement=.75, fp_argmax_verified=True,
                selected_policy=selected[prompt_idx], t_policy_trace=.01,
            ))
    return pd.DataFrame(accuracy), pd.DataFrame(diagnostic)


def expect_reject(name, accuracy, diagnostic, contains):
    try:
        RP.analyze_frames(accuracy, diagnostic, n_boot=100)
        check(name, False, "(accepted invalid input)")
    except RP.PolicyReaderError as exc:
        check(name, contains in str(exc), f"({exc})")


def test_metrics_and_integrity():
    print("\n[policy reader] exact join and oracle metrics")
    accuracy, diagnostic = frames()
    summary, prompts = RP.analyze_frames(accuracy, diagnostic, n_boot=1000, seed=7)
    row = summary.iloc[0]
    check("one summary cell and four audited prompts", len(summary) == 1 and len(prompts) == 4)
    check("end-task headroom H is exact", row.H == .5)
    check("KL selector gain G is exact", row.G == 0.0)
    check("regret H-G is exact", row.regret == .5 and row.captured == 0.0)
    check("rescues and harms use paired prompt outcomes", row.rescues == 1 and row.harms == 1)
    check("selector membership in end-task ties is retained", row.tie_hits == 2)
    check("exact KL ties use declared candidate order",
          prompts.sort_values("prompt_idx").kl_candidate.tolist() ==
          ["evict", "evict", "uniform", "uniform"])
    check("every end-task tie is retained",
          prompts.sort_values("prompt_idx").endtask_ties.tolist() ==
          ["evict", "uniform|interior", "interior", "uniform|evict"])

    duplicate = pd.concat([diagnostic, diagnostic.iloc[[0]]], ignore_index=True)
    expect_reject("duplicate diagnostic key is rejected", accuracy, duplicate, "candidate")
    missing = diagnostic.drop(diagnostic.index[0]).reset_index(drop=True)
    expect_reject("missing candidate row is rejected", accuracy, missing, "expected")
    mixed = diagnostic.copy()
    mixed.loc[(mixed.prompt_idx == 0) & (mixed.candidate == "interior"), "trace_token_hash"] = hashlib.sha256(b"other").hexdigest()
    expect_reject("candidate-specific FP trace is rejected", accuracy, mixed, "trace_token_hash")
    mismatched = diagnostic.copy()
    mismatched.loc[mismatched.candidate == "evict", "corpus_sha"] = "wrong"
    expect_reject("mixed corpus provenance is rejected", accuracy, mismatched, "corpus_sha")
    raw = diagnostic.copy()
    raw["candidate_logits"] = [[0.0, 1.0]] * len(raw)
    expect_reject("raw logits are rejected by column contract", accuracy, raw, "raw-logit")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_file_identity():
    print("\n[policy reader] parquet and sidecar identity")
    accuracy, diagnostic = frames()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ap = tmp / "r8_m_32768_k16_v4_h4.parquet"
        dp = tmp / "r8policy_m_32768_k16_v4_h4.parquet"
        accuracy.to_parquet(ap, index=False)
        diagnostic.to_parquet(dp, index=False)
        accuracy_side = dict(
            parquet=ap.name, model="m", model_id="org/m", ctx=32768,
            native_ctx=131072, tasks=["niah_multikey"],
            arms=["fp", "uniform", "evict", "interior"],
            task_config=dict(n_keys=16, n_values=4, n_hops=4), budgets=[2.0],
            n_prompts=4, prompt_offset=0, window=32, maxb=8,
            question_agnostic=True, corpus_sha="abc123", rows=len(accuracy),
        )
        diagnostic_side = dict(
            parquet=dp.name, model="m", model_id="org/m", ctx=32768,
            native_ctx=131072, tasks=["niah_multikey"],
            task_config=dict(n_keys=16, n_values=4, n_hops=4), budgets=[2.0],
            n_prompts=4, prompt_offset=0, window=32, maxb=8,
            question_agnostic=True, candidates=["uniform", "evict", "interior"],
            trace_rule_version="fp_teacher_v1", trace_steps_requested=8,
            teacher="fp_greedy", metric_dtype="float32", vocabulary="full",
            allocator_budget_rule="feasible", rot_seed=0, norm_correct=True,
            rows=len(diagnostic), expected_rows=len(diagnostic), corpus_sha="abc123",
            accuracy_parquet=ap.name, accuracy_sidecar=ap.with_suffix(".json").name,
            accuracy_sha256=sha256(ap), accuracy_rows=len(accuracy), no_raw_logits=True,
        )
        ap.with_suffix(".json").write_text(json.dumps(accuracy_side))
        dp.with_suffix(".json").write_text(json.dumps(diagnostic_side))
        summary, prompts = RP.load_pair(ap, dp, n_boot=100)
        check("authenticated file pair reads", len(summary) == 1 and len(prompts) == 4)

        diagnostic_side["accuracy_sha256"] = "0" * 64
        dp.with_suffix(".json").write_text(json.dumps(diagnostic_side))
        try:
            RP.load_pair(ap, dp, n_boot=10)
            check("accuracy SHA mismatch is rejected", False)
        except RP.PolicyReaderError as exc:
            check("accuracy SHA mismatch is rejected", "sha256" in str(exc))


if __name__ == "__main__":
    test_metrics_and_integrity()
    test_file_identity()
    print(f"\n{fails} failure(s)")
    raise SystemExit(1 if fails else 0)
