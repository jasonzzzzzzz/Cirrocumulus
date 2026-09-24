#!/usr/bin/env python3
"""CPU-only contract tests for the V3 panel qualification reader."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
READER = ROOT / "h0_measurement/bugs/9_sota_eviction_baselines/read_panel_qual.py"
spec = importlib.util.spec_from_file_location("read_panel_qual", READER)
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


def depths():
    anchors = ((-1, .03),) + tuple(zip(RP.TARGET_RANKS, RP.TARGET_DEPTHS)) + ((48, .97),)
    out = [None] * 48
    for (left_rank, left_depth), (right_rank, right_depth) in zip(anchors[:-1], anchors[1:]):
        for rank in range(left_rank + 1, right_rank):
            frac = (rank - left_rank) / (right_rank - left_rank)
            out[rank] = round(left_depth + frac * (right_depth - left_depth), 4)
    for rank, depth in zip(RP.TARGET_RANKS, RP.TARGET_DEPTHS):
        out[rank] = depth
    return out


def fixture():
    rows = []
    depth_vector = depths()
    for prompt in RP.PROMPTS:
        local = prompt - RP.PROMPTS[0]
        context_hash = hashlib.sha256(f"context-{prompt}".encode()).hexdigest()
        uniform_allocation = hashlib.sha256(f"uniform-{prompt}".encode()).hexdigest()
        for query in RP.QUERY_SLOTS:
            cluster = (query - prompt % 4) % 4
            stem = RP.CLUSTER_STEMS[cluster]
            target_key = f"{stem} target{prompt}"
            target_value = f"{1000000 + local * 100 + cluster:07d}"
            distractor_keys = [f"{stem} distractor{i}_{prompt}" for i in range(11)]
            distractor_values = [f"{2000000 + local * 1000 + cluster * 20 + i:07d}" for i in range(11)]
            target = dict(
                task_variant=RP.TASK_VARIANT, panel_version=RP.PANEL_VERSION,
                panel_rng_version=RP.PANEL_RNG_VERSION,
                panel_rng_namespaces=json.dumps(
                    RP.PANEL_RNG_NAMESPACES, sort_keys=True, separators=(",", ":")
                ),
                query_idx=query, target_cluster=cluster, target_key=target_key,
                target_value=target_value, target_needle_rank=RP.TARGET_RANKS[query],
                target_needle_depth=RP.TARGET_DEPTHS[query],
                panel_cluster_size=12, panel_query_count=4,
                panel_context_hash=context_hash,
                panel_distractor_keys=json.dumps(distractor_keys, separators=(",", ":")),
                panel_distractor_values=json.dumps(distractor_values, separators=(",", ":")),
                needle_depths=json.dumps(depth_vector, separators=(",", ":")),
                corpus_doc=f"book_{local:02d}.txt", corpus_offset=prompt * 100,
                n_prompt_tokens=30732 + query, ctx_len=30600,
                n_question_tokens=100 + query,
            )
            common = dict(
                model=RP.MODEL, model_id=RP.MODEL_ID, ctx=RP.CTX,
                native_ctx=RP.NATIVE_CTX, task=RP.TASK, prompt_idx=prompt,
                n_keys=48, n_values=4, n_hops=4, task_n_needles=48,
                max_new_tokens=24, question_agnostic=True, window=32,
                observed_queries=32, allocator_budget_rule="feasible", maxb=8,
                corpus_sha=RP.EXPECTED_CORPUS_SHA, corpus_spliced=False,
                synthetic=False, rot_seed=0, norm_correct=True, **target,
            )
            rows.append(dict(
                **common, arm="fp", B=0.0, score=1.0, hits=1,
                n_expected=1, first_ok=1.0, gen_len=6,
                reached_max_new=False, bits_per_token=16.0,
                allocation_id=RP.FP_ALLOCATION_ID,
            ))
            success = float(local % 20 < 12)
            rows.append(dict(
                **common, arm="uniform", B=2.0, score=success,
                hits=int(success), n_expected=1, first_ok=success, gen_len=6,
                reached_max_new=False, bits_per_token=2.0,
                allocation_id=uniform_allocation,
            ))
    return pd.DataFrame(rows)


def sidecar(path):
    return dict(
        parquet=path.name, model=RP.MODEL, model_id=RP.MODEL_ID,
        ctx=RP.CTX, native_ctx=RP.NATIVE_CTX, tasks=[RP.TASK],
        arms=["fp", "uniform"],
        task_config=dict(n_keys=48, n_values=4, n_hops=4),
        task_variant=RP.TASK_VARIANT,
        panel=dict(
            version=RP.PANEL_VERSION, clusters=4, cluster_size=12,
            query_count=4, target_ranks=list(RP.TARGET_RANKS),
            target_depths=list(RP.TARGET_DEPTHS),
            rng_version=RP.PANEL_RNG_VERSION,
            rng_namespaces=RP.PANEL_RNG_NAMESPACES,
            key_suffix_contract=RP.KEY_SUFFIX_CONTRACT,
            context_hash_algorithm=RP.CONTEXT_HASH_ALGORITHM,
            allocation_id_algorithm=RP.ALLOCATION_ID_ALGORITHM,
            expected_accuracy_rows=320, expected_policy_rows=0,
            fp_allocation_sentinel=RP.FP_ALLOCATION_ID,
        ),
        generation_limit_version=RP.GENERATION_LIMIT_VERSION,
        generation_limits={RP.TASK: 24}, budgets=[2], n_prompts=40,
        prompt_offset=700, window=32, observation_queries=[32],
        allocator_budget_rule="feasible", maxb=8, rows=320,
        rot_seed=0, norm_correct=True, attn_impl="sieve_compress",
        compress_from="first_answer_token",
        compress_at=("context_end: context prefilled and scored alone "
                     "(window = its last W tokens), question prefilled "
                     "through the compressed cache"),
        question_agnostic=True,
        corpus_sha=RP.EXPECTED_CORPUS_SHA, baselines=None,
        p2=dict(enabled=False, want=["uniform"], routers=[], head_error=False,
                routes=None, routes_meta=None, theta=.05, cascade_bits="2,4,8"),
        task_generation_version=RP.TASK_GENERATION_VERSION,
        target_needle_provenance_version=RP.TARGET_PROVENANCE_VERSION,
    )


def write_artifact(directory, frame, name="r8_llama31-8b_32768_k48_v4_h4_multikey_panel_v1.parquet"):
    path = directory / name
    frame.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(sidecar(path)))
    return path


def rejects(name, action, contains):
    try:
        action()
        check(name, False, "(accepted invalid input)")
    except RP.PanelQualificationError as exc:
        check(name, contains in str(exc), f"({exc})")


def test_metrics():
    print("\n[panel reader] context aggregation and frozen gates")
    frame = fixture()
    row = RP.analyze_frame(frame)
    check("eligible fixture passes", row["eligible"])
    check("four queries aggregate within 40 contexts",
          row["n_contexts"] == 40 and row["n_queries"] == 160 and
          row["uniform_prompt_mean_first_ok"] == .6)
    check("balanced halves pass",
          row["uniform_700_719"] == .6 and row["uniform_720_739"] == .6)
    check("all query slots reported", all(row[f"uniform_slot_{q}"] == .6 for q in range(4)))
    check("paired context delta computed", row["delta_fp_minus_uniform"] == .4 and
          row["delta_lo90"] > .05)
    again = RP.analyze_frame(frame)
    check("10k seed-0 bootstrap deterministic",
          (row["delta_lo90"], row["delta_hi90"]) ==
          (again["delta_lo90"], again["delta_hi90"]))
    rejects("bootstrap refuses 160 independent query rows",
            lambda: RP._paired_interval(np.ones(160)), "40 context")

    # A later correct substring may coexist with a wrong first number. The
    # qualification must use first_ok rather than standard score.
    primary = frame.copy()
    mask = primary.arm.eq("uniform")
    primary.loc[mask, ["score", "hits"]] = [1.0, 1]
    primary_row = RP.analyze_frame(primary)
    check("first_ok is primary and score remains diagnostic",
          primary_row["uniform_prompt_mean_first_ok"] == .6 and
          primary_row["uniform_standard_score_micro"] == 1.0 and
          primary_row["eligible"])

    slot_bad = frame.copy()
    uniform = slot_bad.arm.eq("uniform")
    local = slot_bad.prompt_idx - 700
    successes = {
        0: local < 40,       # 1.0
        1: local % 20 < 4,  # .2
        2: local % 20 < 12, # .6
        3: local % 20 < 12, # .6
    }
    for query, values in successes.items():
        mask = uniform & slot_bad.query_idx.eq(query)
        score = values[mask].astype(float)
        slot_bad.loc[mask, "first_ok"] = score
        slot_bad.loc[mask, "score"] = score
        slot_bad.loc[mask, "hits"] = score.astype(int)
    slot_row = RP.analyze_frame(slot_bad)
    check("slot spread gate catches hidden imbalance",
          slot_row["uniform_prompt_mean_first_ok"] == .6 and
          not slot_row["gate_uniform_slot_spread_le_025"] and
          not slot_row["eligible"])

    fp_slot = frame.copy()
    mask = fp_slot.arm.eq("fp") & fp_slot.query_idx.eq(0) & fp_slot.prompt_idx.lt(705)
    fp_slot.loc[mask, ["first_ok", "score", "hits"]] = [0.0, 0.0, 0]
    fp_row = RP.analyze_frame(fp_slot)
    check("each FP slot must reach .90", fp_row["fp_micro_first_ok"] >= .95 and
          fp_row["fp_slot_0"] < .90 and not fp_row["eligible"])

    capped = frame.copy()
    idx = capped.index[(capped.arm == "fp") & (capped.prompt_idx == 700) &
                       (capped.query_idx == 0)][0]
    capped.loc[idx, ["first_ok", "score", "hits", "gen_len",
                     "reached_max_new"]] = [0.0, 0.0, 0, 24, True]
    capped["reached_max_new"] = capped.reached_max_new.astype(bool)
    capped_row = RP.analyze_frame(capped)
    check("incomplete capped FP query invalidates qualification",
          capped_row["incomplete_capped_fp"] == 1 and not capped_row["eligible"])


def test_row_rejections():
    print("\n[panel reader] row, context, allocation, and task provenance")
    frame = fixture()
    duplicate = pd.concat([frame.iloc[:-1], frame.iloc[[-2]]], ignore_index=True)
    rejects("duplicate or missing query row rejected",
            lambda: RP.validate_frame(duplicate), "one row per prompt")

    bad = frame.copy()
    idx = bad.index[(bad.prompt_idx == 700) & (bad.arm == "uniform") &
                    (bad.query_idx == 3)][0]
    bad.loc[idx, "allocation_id"] = hashlib.sha256(b"other").hexdigest()
    rejects("allocation drift across questions rejected",
            lambda: RP.validate_frame(bad), "allocation_id changes")

    bad = frame.copy()
    idx = bad.index[(bad.prompt_idx == 700) & (bad.arm == "uniform")][0]
    bad.loc[idx, "panel_context_hash"] = hashlib.sha256(b"other").hexdigest()
    rejects("context drift across arms rejected",
            lambda: RP.validate_frame(bad), "shared-context provenance")

    bad = frame.copy()
    bad["task_variant"] = "legacy"
    rejects("task variant exact", lambda: RP.validate_frame(bad), "task_variant")

    bad = frame.copy()
    bad["panel_rng_version"] = "drift"
    rejects("row RNG version exact", lambda: RP.validate_frame(bad),
            "panel_rng_version")

    bad = frame.copy()
    bad["panel_rng_namespaces"] = json.dumps({"haystack": "drift"})
    rejects("row RNG namespaces exact", lambda: RP.validate_frame(bad),
            "panel_rng_namespaces")

    bad = frame.copy()
    mask = bad.prompt_idx.eq(700) & bad.query_idx.eq(0)
    bad.loc[mask, "target_cluster"] = 1
    rejects("cluster rotation exact", lambda: RP.validate_frame(bad),
            "target cluster")

    bad = frame.copy()
    idx = bad.index[0]
    keys = json.loads(bad.loc[idx, "panel_distractor_keys"])
    keys[-1] = keys[0]
    bad.loc[idx, "panel_distractor_keys"] = json.dumps(keys)
    # Keep both arms equal so this reaches the semantic identity check.
    mate = bad.index[(bad.prompt_idx == bad.loc[idx, "prompt_idx"]) &
                     (bad.query_idx == bad.loc[idx, "query_idx"]) &
                     (bad.arm != bad.loc[idx, "arm"])][0]
    bad.loc[mate, "panel_distractor_keys"] = bad.loc[idx, "panel_distractor_keys"]
    rejects("duplicate distractor identity rejected",
            lambda: RP.validate_frame(bad), "distractor keys")

    bad = frame.copy()
    bad.loc[bad.prompt_idx == 739, "corpus_doc"] = "book_00.txt"
    rejects("40 real documents required", lambda: RP.validate_frame(bad),
            "40 distinct")

    bad = frame.copy()
    bad.loc[bad.arm == "uniform", "bits_per_token"] = 1.99
    rejects("uniform bit accounting exact", lambda: RP.validate_frame(bad),
            "exactly B=2")

    bad = frame.copy()
    bad.loc[bad.index[0], "n_prompt_tokens"] += 1
    rejects("prompt token accounting exact", lambda: RP.validate_frame(bad),
            "ctx_len + observed_queries + n_question_tokens")


def test_files_and_exit_codes():
    print("\n[panel reader] sidecar, CSV, and valid-ineligible semantics")
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        path = write_artifact(tmp, fixture())
        summary, eligible = RP.analyze_artifact(path)
        check("authenticated artifact eligible", eligible and len(summary) == 1)
        csv = tmp / "summary.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            status = RP.main([str(path), "--csv", str(csv)])
        saved = pd.read_csv(csv)
        check("CLI writes one-row CSV and exits zero",
              status == 0 and len(saved) == 1 and bool(saved.eligible.iloc[0]))
        with contextlib.redirect_stdout(io.StringIO()):
            status = RP.main([str(path), "--validate-only"])
        check("validation-only exits zero", status == 0)

        invalid = fixture()
        mask = invalid.arm.eq("uniform")
        invalid.loc[mask, ["first_ok", "score", "hits"]] = [1.0, 1.0, 1]
        other = tmp / "ineligible"
        other.mkdir()
        other_path = write_artifact(other, invalid)
        with contextlib.redirect_stdout(io.StringIO()):
            status = RP.main([str(other_path)])
        _, eligible = RP.analyze_artifact(other_path)
        check("valid ineligible qualification exits zero", status == 0 and not eligible)

        meta = sidecar(path)
        meta["panel"]["rng_version"] = "drift"
        path.with_suffix(".json").write_text(json.dumps(meta))
        rejects("panel RNG provenance exact", lambda: RP.load_artifact(path),
                "rng_version")
        with contextlib.redirect_stderr(io.StringIO()):
            check("contract errors exit two", RP.main([str(path)]) == 2)

        path.with_suffix(".json").write_text(json.dumps(sidecar(path)))
        meta = sidecar(path)
        meta["panel"]["allocation_id_algorithm"] = "label_only"
        path.with_suffix(".json").write_text(json.dumps(meta))
        rejects("allocation hash algorithm exact", lambda: RP.load_artifact(path),
                "allocation_id_algorithm")


if __name__ == "__main__":
    test_metrics()
    test_row_rejections()
    test_files_and_exit_codes()
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
    raise SystemExit(1 if fails else 0)
