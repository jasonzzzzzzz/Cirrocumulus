#!/usr/bin/env python3
"""CPU-only contracts for the expanded whole-policy V2-B reader."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
READER = ROOT / "h0_measurement/bugs/9_sota_eviction_baselines/read_policy_v2.py"
spec = importlib.util.spec_from_file_location("read_policy_v2", READER)
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames(mode="mean", *, laprox_perfect=False, uniform_easy=False, n_keys=24):
    depths = np.linspace(.05, .95, n_keys).round(4).tolist()
    corpus_sha = "a" * 64
    accuracy, diagnostic = [], []
    for offset, prompt in enumerate(RP.PROMPTS):
        # 12 successes in each half gives uniform=.60 and balanced headroom.
        uniform_score = int((offset % 20) < (18 if uniform_easy else 12))
        evict_score = 1 - uniform_score
        scores = {candidate: 0 for candidate in RP.CANDIDATES}
        scores.update(uniform=uniform_score, evict=evict_score)
        if laprox_perfect:
            scores["laprox"] = 1
        rank = offset % n_keys
        shared = dict(
            model=RP.MODEL, model_id=RP.MODEL_ID, ctx=RP.CTX,
            native_ctx=RP.NATIVE_CTX, task=RP.TASK, prompt_idx=prompt,
            n_keys=n_keys, n_values=RP.N_VALUES, n_hops=RP.N_HOPS,
            question_agnostic=True, window=RP.WINDOW, maxb=RP.MAXB,
            allocator_budget_rule="feasible", target_needle_rank=rank,
            target_needle_depth=depths[rank], corpus_doc=f"book-{offset:02d}",
            corpus_offset=1000 + offset, corpus_spliced=False,
            corpus_sha=corpus_sha, synthetic=False, rot_seed=0,
            norm_correct=True,
        )
        accuracy.append(dict(
            **shared, arm="fp", B=0, score=1, hits=1, n_expected=1,
            first_ok=1, task_n_needles=n_keys, max_new_tokens=24,
            reached_max_new=False, gen_len=1, bits_per_token=16,
            n_prompt_tokens=30_000, ctx_len=29_968, observed_queries=32,
            needle_depths=json.dumps(depths),
        ))
        metric_rows = []
        for order, candidate in enumerate(RP.CANDIDATES):
            score = scores[candidate]
            accuracy.append(dict(
                **shared, arm=candidate, B=2, score=score, hits=score,
                n_expected=1, first_ok=score, task_n_needles=n_keys,
                max_new_tokens=24, reached_max_new=False, gen_len=1,
                bits_per_token=2 if candidate == "uniform" else 1.999,
                n_prompt_tokens=30_000, ctx_len=29_968, observed_queries=32,
                needle_depths=json.dumps(depths),
            ))
            oracle = "uniform" if uniform_score else "evict"
            mean_kl = (.01 if candidate == oracle else .2) if mode == "mean" else (
                .01 if candidate == "uniform" else .2)
            fp_ce = (.01 if candidate == oracle else .4) if mode == "alternate" else (
                .01 if candidate == "uniform" else .4)
            max_kl = mean_kl + (.01 if candidate == "uniform" else .1)
            top1 = .9 if candidate == "uniform" else .1
            metric_rows.append(dict(
                **shared, B=2, candidate=candidate, candidate_order=order,
                trace_rule_version=RP.TRACE_RULE_VERSION,
                trace_steps_requested=8, trace_len=1,
                trace_token_hash=hashlib.sha256(f"trace-{prompt}".encode()).hexdigest(),
                vocab_size=RP.VOCAB_SIZE, mean_kl=mean_kl, max_kl=max_kl,
                fp_token_ce=fp_ce, top1_agreement=top1,
                fp_argmax_verified=True, selected_policy="", t_policy_trace=.01,
            ))
        selected = min(RP.CANDIDATES,
                       key=lambda candidate: (next(row["mean_kl"] for row in metric_rows
                                                   if row["candidate"] == candidate),
                                              RP.CANDIDATES.index(candidate)))
        for row in metric_rows:
            row["selected_policy"] = selected
        diagnostic.extend(metric_rows)
    return pd.DataFrame(accuracy), pd.DataFrame(diagnostic)


def expect_reject(name, accuracy, diagnostic, contains, n_keys=24):
    try:
        RP.validate_frames(accuracy, diagnostic, n_keys)
        check(name, False, "(accepted invalid input)")
    except RP.PolicyV2ReaderError as exc:
        check(name, contains in str(exc), f"({exc})")


def sidecars(ap: Path, dp: Path, accuracy, diagnostic, n_keys=24):
    accuracy_side = dict(
        parquet=ap.name, model=RP.MODEL, model_id=RP.MODEL_ID, ctx=RP.CTX,
        native_ctx=RP.NATIVE_CTX, tasks=[RP.TASK], arms=list(RP.ARMS),
        task_config=dict(n_keys=n_keys, n_values=4, n_hops=4),
        task_generation_version=RP.TASK_GENERATION_VERSION,
        target_needle_provenance_version=RP.TARGET_PROVENANCE_VERSION,
        generation_limit_version=RP.GENERATION_LIMIT_VERSION,
        generation_limits={RP.TASK: 24}, budgets=[2], n_prompts=40,
        prompt_offset=540, window=32, observation_queries=[32],
        allocator_budget_rule="feasible", maxb=8, rows=360, rot_seed=0,
        norm_correct=True, attn_impl="sieve_compress",
        compress_from="first_answer_token", question_agnostic=True,
        baselines=json.loads(json.dumps(RP.EXPECTED_BASELINES)), corpus_sha="a" * 64,
        p2=dict(enabled=True,
                want=["evict", "interior", "interior_cascade", "interior_pool",
                      "uniform"],
                routers=[], head_error=False, cascade_bits=4,
                routes=None, routes_meta=None),
    )
    diagnostic_side = dict(
        parquet=dp.name, model=RP.MODEL, model_id=RP.MODEL_ID, ctx=RP.CTX,
        native_ctx=RP.NATIVE_CTX, tasks=[RP.TASK],
        task_config=dict(n_keys=n_keys, n_values=4, n_hops=4),
        task_generation_version=RP.TASK_GENERATION_VERSION,
        target_needle_provenance_version=RP.TARGET_PROVENANCE_VERSION,
        generation_limit_version=RP.GENERATION_LIMIT_VERSION,
        generation_limits={RP.TASK: 24}, budgets=[2], n_prompts=40,
        prompt_offset=540, window=32, maxb=8, question_agnostic=True,
        allocator_budget_rule="feasible", candidates=list(RP.CANDIDATES),
        trace_rule_version=RP.TRACE_RULE_VERSION, trace_steps_requested=8,
        teacher="fp_greedy", metric_dtype="float32", vocabulary="full",
        rows=320, expected_rows=320, corpus_sha="a" * 64, rot_seed=0,
        norm_correct=True, accuracy_parquet=ap.name,
        accuracy_sidecar=ap.with_suffix(".json").name,
        accuracy_sha256=_sha(ap), accuracy_rows=360, no_raw_logits=True,
    )
    ap.with_suffix(".json").write_text(json.dumps(accuracy_side))
    dp.with_suffix(".json").write_text(json.dumps(diagnostic_side))
    return accuracy_side, diagnostic_side


def test_analysis_and_prefix_rules():
    print("\n[policy v2 reader] prefixes, comparators, selectors")
    accuracy, diagnostic = frames("mean")
    summary, prompts, decision = RP.analyze_frames(
        accuracy, diagnostic, 24, n_boot=1000, seed=0)
    first = summary.iloc[0]
    check("all six nested prefixes and per-prompt audits are emitted",
          summary.prefix_size.tolist() == [3, 4, 5, 6, 7, 8] and len(prompts) == 240)
    check("strongest fixed comparator is selected on mean development accuracy",
          first.best_fixed_dev == "uniform" and first.best_fixed_mean == .6)
    check("routing headroom is relative to strongest fixed policy",
          first.H_F == .4 and first.H_F_half1 == .4 and first.H_F_half2 == .4)
    check("mean-KL selector advances at the smallest eligible prefix",
          decision["decision"] == "advance_mean_kl" and
          decision["prefix"] == RP.CANDIDATES[:3])
    check("alternates are not evaluated when mean KL advances",
          "fp_token_ce_G_F" not in summary.columns)
    check("prefix selector is recomputed and captures all headroom",
          first.mean_kl_G_F == .4 and first.mean_kl_captured_F == 1.0)

    accuracy, diagnostic = frames("alternate")
    summary, _, decision = RP.analyze_frames(accuracy, diagnostic, 24, n_boot=100)
    check("prespecified CE alternate advances after mean KL fails",
          decision["decision"] == "advance_fp_token_ce" and
          decision["selector_metric"] == "fp_token_ce")
    check("alternates are evaluated on only the frozen prefix",
          summary.fp_token_ce_G_F.notna().sum() == 1 and
          summary.loc[summary.fp_token_ce_G_F.notna(), "prefix_size"].iloc[0] == 3)

    accuracy, diagnostic = frames("mean", laprox_perfect=True)
    summary, _, decision = RP.analyze_frames(accuracy, diagnostic, 24, n_boot=100)
    check("full-prefix H_F<=.05 overrides an earlier prefix advance",
          summary.iloc[0].gate_headroom and summary.iloc[-1].H_F == 0 and
          decision["decision"] == "stop_candidate_routing")

    accuracy, diagnostic = frames("mean", uniform_easy=True)
    summary, prompts, decision = RP.analyze_frames(
        accuracy, diagnostic, 24, n_boot=100)
    check("independent block outside the uniform gate is not interpreted",
          decision["decision"] == "difficulty_non_replicating" and
          prompts.empty and len(summary) == 1 and "H_F" not in summary.columns)


def test_integrity_rejections():
    print("\n[policy v2 reader] strict row provenance")
    accuracy, diagnostic = frames()
    raw = diagnostic.copy()
    raw["raw_logits"] = [[0.0]] * len(raw)
    expect_reject("raw logits are rejected", accuracy, raw, "raw-logit")

    bad = diagnostic.copy()
    mask = (bad.prompt_idx == 540) & (bad.candidate == "evict")
    bad.loc[mask, "corpus_offset"] += 1
    expect_reject("candidate-specific corpus provenance is rejected",
                  accuracy, bad, "corpus_offset")

    bad = accuracy.copy()
    mask = (bad.prompt_idx == 540) & (bad.arm == "evict")
    bad.loc[mask, "target_needle_depth"] += .01
    expect_reject("arm-specific queried-needle provenance is rejected",
                  bad, diagnostic, "target_needle_depth")

    bad = diagnostic.copy()
    bad.loc[bad.candidate == "obcache_k", "candidate_order"] = 7
    expect_reject("candidate labels have one frozen order", accuracy, bad,
                  "candidate order")

    bad = diagnostic.copy()
    bad["trace_len"] = 8
    expect_reject("trace length is pinned to the FP generation", accuracy, bad,
                  "min(8, FP gen_len)")

    bad = diagnostic.copy()
    bad["vocab_size"] = 128
    expect_reject("Llama vocabulary size is exact", accuracy, bad, "vocab_size")

    bad_accuracy, bad_diagnostic = frames(n_keys=20)
    expect_reject("only a V2-A selectable key count is accepted",
                  bad_accuracy, bad_diagnostic, "24 or 32", n_keys=20)


def test_file_authentication_and_lock():
    print("\n[policy v2 reader] sidecars, SHA link, immutable lock")
    accuracy, diagnostic = frames()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ap = tmp / "r8_llama31-8b_32768_k24_v4_h4.parquet"
        dp = tmp / "r8policy_llama31-8b_32768_k24_v4_h4.parquet"
        accuracy.to_parquet(ap, index=False)
        diagnostic.to_parquet(dp, index=False)
        accuracy_side, diagnostic_side = sidecars(ap, dp, accuracy, diagnostic)
        loaded_a, loaded_d, _, _ = RP.load_pair(ap, dp, 24)
        check("exact artifact pair and both sidecars authenticate",
              len(loaded_a) == 360 and len(loaded_d) == 320)

        summary, _, decision = RP.analyze_frames(loaded_a, loaded_d, 24, n_boot=100)
        manifest = RP.build_lock_manifest(ap, dp, 24, summary, decision)
        lock = tmp / "lock.json"
        check("advance produces a confirmation-ready lock",
              manifest is not None and RP._write_lock(lock, manifest))
        written = json.loads(lock.read_text())
        check("lock pins both artifacts, sidecars, comparator, and row counts",
              written["development"]["accuracy"]["sha256"] == _sha(ap) and
              written["development"]["diagnostic"]["sha256"] == _sha(dp) and
              written["best_fixed_dev"] == "uniform" and
              written["confirmation"]["expected_accuracy_rows"] == 160 and
              written["confirmation"]["expected_diagnostic_rows"] == 120)
        check("identical lock creation is idempotent", RP._write_lock(lock, manifest))

        accuracy_side["baselines"]["obck_ada"]["alpha"] = .3
        ap.with_suffix(".json").write_text(json.dumps(accuracy_side))
        try:
            RP.load_pair(ap, dp, 24)
            check("baseline config mutation is rejected", False)
        except RP.PolicyV2ReaderError as exc:
            check("baseline config mutation is rejected", "baselines" in str(exc),
                  f"({exc})")

        accuracy_side["baselines"] = json.loads(json.dumps(RP.EXPECTED_BASELINES))
        accuracy_side["p2"]["cascade_bits"] = 3
        ap.with_suffix(".json").write_text(json.dumps(accuracy_side))
        try:
            RP.load_pair(ap, dp, 24)
            check("cascade base tier drift is rejected", False)
        except RP.PolicyV2ReaderError as exc:
            check("cascade base tier drift is rejected", "cascade_bits" in str(exc),
                  f"({exc})")

        # Restore the accuracy sidecar, then break the parquet SHA link.
        accuracy_side["baselines"] = json.loads(json.dumps(RP.EXPECTED_BASELINES))
        accuracy_side["p2"]["cascade_bits"] = 4
        ap.with_suffix(".json").write_text(json.dumps(accuracy_side))
        diagnostic_side["accuracy_sha256"] = "0" * 64
        dp.with_suffix(".json").write_text(json.dumps(diagnostic_side))
        try:
            RP.load_pair(ap, dp, 24)
            check("accuracy parquet SHA mismatch is rejected", False)
        except RP.PolicyV2ReaderError as exc:
            check("accuracy parquet SHA mismatch is rejected", "accuracy_sha256" in str(exc),
                  f"({exc})")

        no_advance = tmp / "stale.json"
        no_advance.write_text("{}\n")
        try:
            RP._write_lock(no_advance, None)
            check("a stale lock cannot survive a non-advance", False)
        except RP.PolicyV2ReaderError as exc:
            check("a stale lock cannot survive a non-advance", "stale" in str(exc))


if __name__ == "__main__":
    test_analysis_and_prefix_rules()
    test_integrity_rejections()
    test_file_authentication_and_lock()
    print(f"\n{fails} failure(s)")
    raise SystemExit(1 if fails else 0)
