"""Protect the camera-ready runner's exact-threshold contract."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paper_threshold_transport",
    ROOT / "paper/analysis_code/temporal/run_2024_threshold_transport.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def canonical_rows():
    return pd.DataFrame([
        {"target": target, "protocol": "fixed_calibration_threshold",
         "nominal_fraction": 0.10, "availability_scenario": "all_record_upper_bound",
         "threshold": 0.123456789}
        for target in MODULE.TARGETS
    ])


def test_full_precision_cutoffs(tmp_path):
    path = tmp_path / "thresholds.csv"
    canonical_rows().to_csv(path, index=False)
    cutoffs = MODULE.load_cutoffs(path)
    assert set(cutoffs) == set(MODULE.TARGETS)
    assert all(value == 0.123456789 for value in cutoffs.values())
    assert all(value != round(value, 6) for value in cutoffs.values())


@pytest.mark.parametrize("case", ["missing", "duplicate", "nan", "infinite", "negative"])
def test_invalid_cutoffs_rejected(tmp_path, case):
    rows = canonical_rows()
    if case == "missing":
        rows = rows.iloc[:-1]
    elif case == "duplicate":
        rows = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
    else:
        rows.loc[0, "threshold"] = {
            "nan": float("nan"), "infinite": float("inf"), "negative": -0.1,
        }[case]
    path = tmp_path / "thresholds.csv"
    rows.to_csv(path, index=False)
    with pytest.raises(ValueError):
        MODULE.load_cutoffs(path)


def test_archived_cutoffs_match_corrected_results():
    cutoffs = MODULE.load_cutoffs(ROOT / "paper/source_data/canonical_protocol_results.csv")
    results = pd.read_csv(ROOT / "paper/source_data/replication_2024_results.csv")
    for row in results.itertuples(index=False):
        assert cutoffs[row.target] == row.threshold_used
