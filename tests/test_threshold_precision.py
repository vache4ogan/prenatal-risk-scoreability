"""Transported score thresholds retain their exact binary64 boundary."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_temporal_four_cell as runner
from thresholds import load_cutoffs


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "results/reference/2023/thresholds_2022.csv"


def test_temporal_csv_preserves_frozen_boundaries():
    loaded = runner.load_thresholds(RELEASE)
    for row in loaded.itertuples(index=False):
        metadata = json.loads(
            (ROOT / "pretrained" / row.target / "preprocessing.json").read_text()
        )
        frozen = float(metadata["thresholds"][str(row.nominal_fraction)])
        scores = np.array([np.nextafter(frozen, -np.inf), frozen,
                           np.nextafter(frozen, np.inf)])
        assert (scores >= row.threshold).tolist() == [False, True, True]
        assert row.threshold == frozen


def test_protocol_csv_preserves_frozen_boundaries(tmp_path):
    with RELEASE.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if float(r["nominal_fraction"]) == 0.1]
    for row in rows:
        row.update(protocol="fixed_calibration_threshold",
                   availability_scenario="all_record_upper_bound")
    path = tmp_path / "protocols.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cutoffs = load_cutoffs(path)
    for row in rows:
        frozen = float(row["threshold"])
        scores = np.array([np.nextafter(frozen, -np.inf), frozen,
                           np.nextafter(frozen, np.inf)])
        assert (scores >= cutoffs[row["target"]]).tolist() == [False, True, True]
        assert cutoffs[row["target"]] == frozen


def test_bundle_rejects_a_one_ulp_threshold_shift():
    frozen = 0.12452148240158581
    bundle = {"thresholds": {0.05: frozen, 0.1: frozen}}
    rows = pd.DataFrame([
        {"nominal_fraction": q, "threshold": np.nextafter(frozen, -np.inf)}
        for q in [0.05, 0.1]
    ])
    with pytest.raises(ValueError, match="full-precision"):
        runner.validate_bundle_thresholds(bundle, rows)
