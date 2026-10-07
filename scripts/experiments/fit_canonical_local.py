"""Fit the frozen canonical specification using local CDC 2022 data, without ClearML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

import run_tae_canonical_clearml as canonical


def fit_target(frame, train_rows, calibration_rows, target, config, threads=4):
    population = canonical.resident_singleton_mask(frame)
    known = frame[target].notna().to_numpy()
    train = population & known & train_rows
    calibration = population & known & calibration_rows
    if np.any(train & calibration):
        raise ValueError("Training/calibration overlap")
    features = list(canonical.LANDMARK_STRICT_FEATURES)
    pre = canonical.build_preprocessor(features, ["mother_race"], config["preprocessing"])
    x = canonical.consistent_matrix(pre.fit_transform(frame.loc[train, features]))
    params = canonical.lightgbm_parameters(config)
    params["n_jobs"] = threads
    model = LGBMClassifier(**params)
    model.fit(x, frame.loc[train, target].to_numpy(np.int8))
    calibration_x = canonical.consistent_matrix(pre.transform(frame.loc[calibration, features]))
    scores = model.predict_proba(calibration_x)[:, 1]
    thresholds = [{"target": target, "nominal_fraction": q,
                   "threshold": canonical.threshold_for_capacity(scores, q),
                   "source_year": 2022, "calibration_n": len(scores)}
                  for q in config["evaluation"]["fractions"]]
    return {"target": target, "feature_set": "landmark_strict", "features": features,
            "preprocessor": pre, "model": model}, thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/canonical/experiment_config_tae_canonical.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    config = canonical.load_local_config(args.config)
    canonical.validate_config(config)
    observed_hash = canonical.sha256_file(args.train_csv)
    if observed_hash != config["data"]["expected_sha256"]["2022"]:
        raise ValueError("CDC 2022 input hash differs from the frozen data release")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame = canonical.read_cdc_frame(args.train_csv, canonical.required_columns(),
                                     config["data"]["expected_rows"]["2022"], "CDC 2022")
    for column in ["is_us_resident", "is_singleton", *canonical.TARGETS]:
        canonical.validate_binary_column(frame, column, "2022", allow_missing=column in canonical.TARGETS)
    rng = np.random.default_rng(config["split"]["seed"])
    calibration = rng.random(len(frame)) < config["split"]["calibration_fraction"]
    thresholds = []
    for target in canonical.TARGETS:
        print(f"Fitting {target} on CDC 2022 only", flush=True)
        bundle, rows = fit_target(frame, ~calibration, calibration, target, config, args.threads)
        joblib.dump(bundle, args.output_dir / f"{target}__landmark_strict__lightgbm.joblib")
        thresholds.extend(rows)
    pd.DataFrame(thresholds).to_csv(args.output_dir / "thresholds_2022.csv", index=False)
    (args.output_dir / "local_training_manifest.json").write_text(json.dumps({
        "input_sha256": observed_hash, "training_year": 2022, "seed": config["split"]["seed"],
        "model_fit_count": 3, "configuration": config,
        "threads": args.threads, "script_sha256": canonical.sha256_file(Path(__file__)),
    }, indent=2), encoding="utf-8")
    print("PASS: three local models and CDC 2022 cutoffs saved", flush=True)


if __name__ == "__main__":
    main()
