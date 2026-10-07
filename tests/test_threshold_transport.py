"""Protect the full-precision threshold transport contract."""

from pathlib import Path

import pandas as pd
import pytest
import thresholds as MODULE


ROOT = Path(__file__).resolve().parents[1]


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
    cutoffs = MODULE.load_cutoffs(ROOT / "results/reference/2023/canonical_protocol_results.csv")
    results = pd.read_csv(ROOT / "results/reference/2024/replication_2024_results.csv")
    for row in results.itertuples(index=False):
        assert cutoffs[row.target] == row.threshold_used
