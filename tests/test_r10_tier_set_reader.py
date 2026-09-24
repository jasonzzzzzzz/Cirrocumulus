from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
from contextlib import contextmanager

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
READER_PATH = (PROJECT / "h0_measurement" / "bugs" /
               "10_tier_set_rederivation" / "read_tier_set.py")
SPEC = importlib.util.spec_from_file_location("r10_tier_reader", READER_PATH)
R = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = R
SPEC.loader.exec_module(R)

@contextmanager
def expect_invalid(pattern: str):
    try:
        yield
    except R.InvalidR10 as exc:
        assert re.search(pattern, str(exc)), (pattern, str(exc))
    else:
        raise AssertionError(f"expected InvalidR10 matching {pattern!r}")


def fractions(label: str, budget: int) -> dict[int, float]:
    tiers = R.TIERS[label]
    if budget in tiers:
        return {b: float(b == budget) for b in tiers}
    hi = min(b for b in tiers if b > budget)
    lo = max(b for b in tiers if b < budget)
    ph = (budget - lo) / (hi - lo)
    return {b: (1 - ph if b == lo else ph if b == hi else 0.0) for b in tiers}


def panel_frame(*, n_rep=2, mha=False) -> pd.DataFrame:
    rows = []
    for head in range(n_rep):
        row = {
            "prompt": 0, "family": "cont", "step": 4, "layer": 0,
            "head": head, "kv_head": 0, "n_rep": n_rep, "grp_size": n_rep,
            "L": 800, "err_uniform2": 2.5, "err_uniform3": 2.5,
            "err_e2_grp_pp_accum_frac": 2.0,
            "err_e3_grp_pp_accum_frac": 2.0,
        }
        for budget in R.BUDGETS:
            row[f"err_wf_grp_csv_b4_accum_{budget}"] = 1.0
            row[f"evict_frac_grp_csv_b4_accum_{budget}"] = fractions("full", budget)[0]
            row[f"alloc_mismatch_full_vs_existing_csv_b4_accum_{budget}"] = 0
            row[f"max_bit_diff_full_vs_existing_csv_b4_accum_{budget}"] = 0
            for label, tiers in R.TIERS.items():
                fr = fractions(label, budget)
                row[R.err_col(label, budget)] = 1.0 + 0.01 * list(R.TIERS).index(label)
                row[R.mean_col(label, budget)] = sum(b * fr[b] for b in tiers)
                row[R.evict_col(label, budget)] = fr[0]
                for bit in tiers:
                    row[R.tier_col(label, budget, bit)] = fr[bit]
                if mha:
                    row[("alloc_mismatch_grp_vs_head_tier_"
                         f"{label}_csv_b4_accum_{budget}")] = 0
                    row[("max_bit_diff_grp_vs_head_tier_"
                         f"{label}_csv_b4_accum_{budget}")] = 0
        # The explicit full arm must duplicate the existing path exactly.
        for budget in R.BUDGETS:
            row[R.err_col("full", budget)] = row[f"err_wf_grp_csv_b4_accum_{budget}"]
            row[R.evict_col("full", budget)] = row[f"evict_frac_grp_csv_b4_accum_{budget}"]
        rows.append(row)
    return pd.DataFrame(rows)


def analysis_frame(error_scale: dict[str, float], *, prompt0: int) -> pd.DataFrame:
    rows = []
    for prompt in (prompt0, prompt0 + 1):
        for head in (0, 1):
            row = {
                "prompt": prompt, "family": "cont", "step": 4,
                "layer": 0, "head": head, "kv_head": 0, "n_rep": 2,
                "grp_size": 2, "L": 1000,
            }
            for budget in R.BUDGETS:
                row[f"err_uniform{budget}"] = 2.5
                row[f"err_e{budget}_grp_pp_accum_frac"] = 3.0
                for label, tiers in R.TIERS.items():
                    row[R.err_col(label, budget)] = error_scale[label]
                    fr = fractions(label, budget)
                    if label == "nested3":
                        # Exercise the physical-token occupancy selector exactly
                        # at a value below its strict 5% threshold.
                        fr = {b: 0.0 for b in tiers}
                        fr[3] = 0.04
                        fr[4] = 0.96
                    row[R.mean_col(label, budget)] = sum(b * fr[b] for b in tiers)
                    row[R.evict_col(label, budget)] = fr[0]
                    for bit in tiers:
                        row[R.tier_col(label, budget, bit)] = fr[bit]
            rows.append(row)
    return pd.DataFrame(rows)


def test_source_ledger_authenticates_version_map_and_file_bytes(tmp_path=None):
    if tmp_path is None:
        with tempfile.TemporaryDirectory() as td:
            return test_source_ledger_authenticates_version_map_and_file_bytes(Path(td))
    source = tmp_path / "a.py"
    source.write_text("x = 1\n")
    sources = {"a.py": R.sha256_file(source)}
    ledger = {
        "ledger_version": R.LEDGER_VERSION,
        "content_sha256": R.ledger_content_sha256(R.LEDGER_VERSION, sources),
        "source_sha256": sources,
    }
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger))
    assert R.verify_source_ledger(path, project_root=tmp_path,
                                  required_sources={"a.py"}) == ledger
    source.write_text("x = 2\n")
    with expect_invalid("source drift"):
        R.verify_source_ledger(path, project_root=tmp_path,
                               required_sources={"a.py"})


def test_dense_grid_feeds_accum_but_quantizes_only_steps_zero_and_four():
    rows = []
    spec = R.TaskSpec("qwen3-1.7b", 2048, 1, 0, ("cont",))
    for step in range(8):
        for head in (0, 1):
            rows.append({
                "model": spec.model, "ctx": spec.ctx, "prompt": 0,
                "prompt_offset": 0, "family": "cont", "step": step,
                "layer": 0, "head": head, "L": 2048 + step,
                "synthetic": False, "quantized": step in (0, 4),
                "norm_correct": True, "rot_seed": 2,
                "corpus_sha": "frozen",
            })
    frame = pd.DataFrame(rows)
    R._check_frame_grid(frame, spec)
    frame.loc[(frame["step"] == 1) & (frame["head"] == 0), "quantized"] = True
    with expect_invalid("quantized rows"):
        R._check_frame_grid(frame, spec)


def test_panel_checks_full_identity_tiers_budget_and_physical_deduplication():
    frame = panel_frame()
    R.validate_panel_rows(frame, mha_control=False)
    assert len(R.physical_group_table(frame)) == 1

    forbidden = frame.copy()
    forbidden["tier_frac_grp_tier_nested4_csv_b4_accum_3_b3"] = 0.0
    with expect_invalid("tier columns"):
        R.validate_panel_rows(forbidden, mha_control=False)

    mismatch = frame.copy()
    mismatch.loc[0, "alloc_mismatch_full_vs_existing_csv_b4_accum_3"] = 1
    with expect_invalid("V1 full allocation"):
        R.validate_panel_rows(mismatch, mha_control=False)


def test_mha_control_requires_explicit_per_head_allocation_identity():
    frame = panel_frame(n_rep=1, mha=True)
    R.validate_panel_rows(frame, mha_control=True)
    col = "max_bit_diff_grp_vs_head_tier_nested3_csv_b4_accum_3"
    frame.loc[0, col] = 1
    with expect_invalid("V5 n_rep=1 identity"):
        R.validate_panel_rows(frame, mha_control=True)


def test_main_statistics_use_prompt_rmse_equal_cells_and_frozen_gates():
    scales = {
        "full": 1.0, "no1": 1.01, "base3_dense": 1.02,
        "nested3": 1.04, "nested4": 1.041,
    }
    specs = [
        R.TaskSpec("llama31-8b", 32768, 2, 0, ("cont",)),
        R.TaskSpec("llama31-8b", 131072, 2, 0, ("cont",)),
        R.TaskSpec("qwen3-8b", 8192, 2, 0, ("cont",)),
        R.TaskSpec("qwen3-30b-a3b-2507", 8192, 2, 0, ("cont",)),
    ]
    tasks = [{"spec": spec, "frame": analysis_frame(scales, prompt0=10 * i)}
             for i, spec in enumerate(specs)]
    table, detail = R.analyze_main(tasks)
    assert detail["validity"] == "valid_r10"
    assert detail["decision"] == "select_nested4_b2_b3"
    b3 = detail["budgets"]["3"]
    assert b3["nested3_pass"]
    assert b3["nested4_vs_nested3"]["select_nested4"]
    assert np.isclose(b3["labels"]["nested3"]["macro"]["error_ratio"], 1.04)
    assert all(np.isclose(x, 0.04) for x in
               b3["nested4_vs_nested3"]["nested3_tier3_fraction"].values())
    assert set(table["scope"]) == {"cell", "macro", "transition"}

    # PCG64(0) and stable iteration order make the prespecified bootstrap exact.
    _, repeated = R.analyze_main(tasks)
    assert repeated == detail


if __name__ == "__main__":
    tests = [
        test_source_ledger_authenticates_version_map_and_file_bytes,
        test_dense_grid_feeds_accum_but_quantizes_only_steps_zero_and_four,
        test_panel_checks_full_identity_tiers_budget_and_physical_deduplication,
        test_mha_control_requires_explicit_per_head_allocation_identity,
        test_main_statistics_use_prompt_rmse_equal_cells_and_frozen_gates,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
