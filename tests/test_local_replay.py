"""Local replay contracts on synthetic records and mocked model bundles."""

import sys
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import canonical
import four_cell
import run_temporal_four_cell as runner


@pytest.mark.parametrize("script", [
    "preprocessing/extract_canonical_cdc.py",
    "preprocessing/harmonize_cdc_2024.py",
    "experiments/fit_canonical_local.py",
    "experiments/run_temporal_four_cell.py",
    "experiments/run_tae_synthetic_shapley81.py",
])
def test_advertised_command_help(script):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-B", str(root / "scripts" / script), "--help"],
        capture_output=True, env={**os.environ, "PYTHONIOENCODING": "ascii"},
    )
    assert result.returncode == 0, result.stderr.decode("ascii", errors="replace")
    assert b"usage:" in result.stdout


def test_native_fallback_only_when_joblib_is_absent(tmp_path, monkeypatch):
    expected = {"synthetic": True}
    calls = []
    monkeypatch.setattr(four_cell, "load_native_target",
                        lambda root, target: calls.append((root, target)) or expected)
    assert four_cell.load_frozen_model_bundle(tmp_path, "target_preterm") is expected
    assert calls == [(tmp_path, "target_preterm")]
    name = "target_preterm__landmark_strict__lightgbm.joblib"
    for subdir in ["a", "b"]:
        folder = tmp_path / subdir
        folder.mkdir()
        (folder / name).write_bytes(b"not a model")
    with pytest.raises(FileNotFoundError):
        four_cell.load_frozen_model_bundle(tmp_path, "target_preterm")
    assert len(calls) == 1


def test_invalid_joblib_does_not_fall_back(tmp_path, monkeypatch):
    (tmp_path / "target_preterm__landmark_strict__lightgbm.joblib").write_bytes(b"not a model")
    def fail(*args):
        pytest.fail("Native fallback must not hide an invalid joblib")
    monkeypatch.setattr(four_cell, "load_native_target", fail)
    monkeypatch.setattr(four_cell.joblib, "load",
                        lambda path: (_ for _ in ()).throw(ValueError("invalid bundle")))
    with pytest.raises(ValueError, match="invalid bundle"):
        four_cell.load_frozen_model_bundle(tmp_path, "target_preterm")


def test_native_thresholds_keep_full_precision():
    values = {0.05: 0.123456789012345, 0.1: 0.098765432109876}
    rows = pd.DataFrame([{"nominal_fraction": q, "threshold": t} for q, t in values.items()])
    runner.validate_bundle_thresholds({"thresholds": values}, rows)
    rows["threshold"] = rows.threshold.round(6)
    with pytest.raises(ValueError, match="full-precision"):
        runner.validate_bundle_thresholds({"thresholds": values}, rows)


@pytest.mark.parametrize("filename,name", [
    ("experiment_config_tae_canonical.yaml", "landmark_strict"),
    ("experiment_config_noprior.yaml", "no_priorterm"),
    ("experiment_config_no_race.yaml", "no_race"),
])
def test_named_feature_sets_preserve_order(filename, name):
    root = Path(__file__).resolve().parents[1]
    config = canonical.load_local_config(root / "configs/canonical" / filename)
    canonical.validate_config(config)
    assert config["feature_set"]["features"] == canonical.FEATURE_SETS[name]
    assert config["model"]["parameters"] == canonical.DEFAULT_CONFIG["model"]["parameters"]


def test_replay_emits_four_cells_and_three_care_scenarios(tmp_path, monkeypatch):
    n = 2000
    rng = np.random.default_rng(20261007)
    frame = pd.DataFrame({f: rng.random(n) for f in canonical.LANDMARK_STRICT_FEATURES})
    frame["mother_race"] = np.resize(np.arange(1, 32), n)
    frame["prenatal_care_month"] = np.resize([1, 2, 3, 4, 0, np.nan], n)
    frame["is_us_resident"] = 1
    frame["is_singleton"] = 1
    for index, target in enumerate(runner.TARGETS):
        frame[target] = (np.arange(n) % (index + 4) == 0).astype(int)
    data = tmp_path / "synthetic.csv"
    frame.to_csv(data, index=False)
    cutoffs = tmp_path / "thresholds.csv"
    pd.DataFrame([{"target": t, "nominal_fraction": q, "threshold": cutoff}
                  for t in runner.TARGETS for q, cutoff in [(0.05, 0.95), (0.1, 0.9)]]
                 ).to_csv(cutoffs, index=False)

    class Preprocessor:
        def transform(self, values):
            assert values.mother_race.dtype.kind in "if"
            return values.to_numpy(dtype=np.float64)

    class Model:
        def predict_proba(self, matrix, num_threads):
            assert matrix.dtype == np.float32
            scores = matrix[:, 1].astype(np.float64)
            return np.column_stack([1 - scores, scores])

    def bundle(root, target, feature_set):
        return {"target": target, "feature_set": feature_set,
                "features": canonical.LANDMARK_STRICT_FEATURES,
                "preprocessor": Preprocessor(), "model": Model(),
                "thresholds": {0.05: 0.95, 0.1: 0.9},
                "_model_sha256": "synthetic-model", "_preprocessing_sha256": "synthetic-preprocessing"}
    monkeypatch.setattr(runner.audit, "load_frozen_model_bundle", bundle)
    out = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", [
        "replay", "--csv", str(data), "--thresholds", str(cutoffs), "--year", "2023",
        "--expected-sha256", runner.sha256(data), "--output-dir", str(out),
        "--bootstrap-replicates", "3", "--threads", "1",
    ])
    runner.main()
    assert len(pd.read_csv(out / "four_cells.csv")) == 24
    assert len(pd.read_csv(out / "bootstrap_replicates.csv")) == 18
    protocols = pd.read_csv(out / "canonical_protocol_results.csv")
    assert len(protocols) == 54
    assert set(protocols.availability_scenario) == set(canonical.SCENARIOS)
    assert set(protocols.protocol) == set(canonical.PROTOCOLS)
    transported = pd.read_csv(out / "threshold_transport.csv")
    assert set(transported.threshold_precision) == {"exact", "rounded_6dp"}
    summary = pd.read_csv(out / "summary.csv")
    zero_contrast = summary.total_topq_contrast.eq(0)
    assert zero_contrast.any()
    assert summary.loc[zero_contrast, "capacity_share"].isna().all()
