"""Threshold inputs must be complete before any clinical data or models are read."""

import sys

import numpy as np
import pandas as pd
import pytest

import run_temporal_four_cell as runner


@pytest.fixture
def thresholds():
    return pd.DataFrame([
        {"target": target, "nominal_fraction": q, "threshold": 0.2,
         "source_year": 2022, "comparator": ">="}
        for target in runner.TARGETS for q in [0.05, 0.1]
    ])


def test_complete_thresholds_load(tmp_path, thresholds):
    path = tmp_path / "thresholds.csv"
    thresholds.to_csv(path, index=False)
    pd.testing.assert_frame_equal(runner.load_thresholds(path), thresholds)


@pytest.mark.parametrize("case", ["empty", "missing", "duplicate", "extra_target", "wrong_fraction"])
def test_threshold_grid_must_be_complete(tmp_path, thresholds, case):
    if case == "empty":
        thresholds = thresholds.iloc[:0]
    elif case == "missing":
        thresholds = thresholds.iloc[:-1]
    elif case == "duplicate":
        thresholds.iloc[-1] = thresholds.iloc[0]
    elif case == "extra_target":
        thresholds.loc[0, "target"] = "unknown_target"
    else:
        thresholds.loc[0, "nominal_fraction"] = 0.2
    path = tmp_path / "thresholds.csv"
    thresholds.to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly one row"):
        runner.load_thresholds(path)


@pytest.mark.parametrize("cutoff", [np.nan, np.inf, -0.1, 1.1])
def test_threshold_values_are_finite_probabilities(tmp_path, thresholds, cutoff):
    thresholds.loc[0, "threshold"] = cutoff
    path = tmp_path / "thresholds.csv"
    thresholds.to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite probabilities"):
        runner.load_thresholds(path)


@pytest.mark.parametrize("column,value", [("source_year", 2023), ("threshold_source_year", 2024),
                                          ("comparator", ">")])
def test_threshold_provenance_fields_are_validated(tmp_path, thresholds, column, value):
    thresholds[column] = value
    path = tmp_path / "thresholds.csv"
    thresholds.to_csv(path, index=False)
    with pytest.raises(ValueError, match="2022 calibration|comparator"):
        runner.load_thresholds(path)


def test_threshold_columns_are_required(tmp_path, thresholds):
    path = tmp_path / "thresholds.csv"
    thresholds.drop(columns="threshold").to_csv(path, index=False)
    with pytest.raises(ValueError, match="requires columns"):
        runner.load_thresholds(path)


def test_main_rejects_empty_thresholds_before_clinical_io(tmp_path, thresholds, monkeypatch):
    path = tmp_path / "thresholds.csv"
    thresholds.iloc[:0].to_csv(path, index=False)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "audit", "--csv", "not-read.csv", "--models", "not-read-models",
        "--thresholds", str(path), "--year", "2023", "--output-dir", str(output),
        "--expected-sha256", "not-read",
    ])
    monkeypatch.setattr(runner, "sha256", lambda path: pytest.fail("Clinical input was accessed"))
    with pytest.raises(ValueError, match="exactly one row"):
        runner.main()
    assert not output.exists()
