"""Invariant: CDC 2023 is temporal test only and never fits models or calibrators."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_tae_canonical_clearml import build_preprocessor
from run_tae_canonical_clearml import DEFAULT_CONFIG, LANDMARK_STRICT_FEATURES
from fit_canonical_local import fit_target


def test_held_out_values_do_not_change_fitted_preprocessing():
    pre = build_preprocessor(["numeric", "mother_race"], ["mother_race"],
                             {"numeric_imputation": "median", "categorical_imputation": "most_frequent", "standardize_numeric": True})
    pre.fit(pd.DataFrame({"numeric": [1.0, 2.0, np.nan, 3.0], "mother_race": [1, 2, 1, 2]}))
    imputer = pre.named_transformers_["numeric"].named_steps["imputer"]
    before = imputer.statistics_.copy()
    pre.transform(pd.DataFrame({"numeric": [1e8, np.nan], "mother_race": [1, 2]}))
    assert np.array_equal(before, imputer.statistics_)
    assert before[0] == 2.0


def test_recorded_lineage_and_prediction_count():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "results/tae_2026_canonical/run_manifest.json").read_text())
    assert manifest["design"]["model_fit_count"] == 3
    assert manifest["design"]["temporal_prediction_pass_count"] == 3
    assert manifest["clearml_task_id"] == "4f9d98dd39f3438181f48b256b23bd94"


def test_local_training_uses_only_training_rows_for_imputation():
    n = 1000
    frame = pd.DataFrame({feature: np.ones(n) for feature in LANDMARK_STRICT_FEATURES})
    frame["is_us_resident"] = 1
    frame["is_singleton"] = 1
    frame["target_preterm"] = np.arange(n) % 2
    train = np.arange(n) < 800
    frame.loc[~train, "mother_bmi"] = 10000
    bundle, thresholds = fit_target(frame, train, ~train, "target_preterm", DEFAULT_CONFIG, threads=1)
    numeric = [f for f in LANDMARK_STRICT_FEATURES if f != "mother_race"]
    stats = bundle["preprocessor"].named_transformers_["numeric"].named_steps["imputer"].statistics_
    assert stats[numeric.index("mother_bmi")] == 1
    assert len(thresholds) == 2
    assert all(row["calibration_n"] == 200 for row in thresholds)
