#!/usr/bin/env python3
"""Apply stored CDC 2022 q=10% score cutoffs to harmonized CDC 2024 data."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


TARGETS = ("target_preterm", "target_nicu", "target_lbw")


def build_targets(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    gestation = pd.to_numeric(frame["gestation_oe_weeks"], errors="coerce")
    birth_weight = pd.to_numeric(frame["birth_weight_g"], errors="coerce")

    frame["target_preterm"] = np.where(
        gestation.between(17, 47), (gestation < 37).astype(float), np.nan
    )

    nicu = frame["admit_nicu"].replace({"Y": 1, "N": 0, "U": np.nan})
    frame["target_nicu"] = pd.to_numeric(nicu, errors="coerce")
    frame["target_lbw"] = np.where(
        birth_weight.between(227, 8165), (birth_weight < 2500).astype(float), np.nan
    )
    return frame


def load_cutoffs(path: Path) -> dict[str, tuple[float, float]]:
    rows = pd.read_csv(path)
    rows = rows[
        (rows["protocol"] == "fixed_calibration_threshold")
        & np.isclose(rows["nominal_fraction"], 0.10)
        & (rows["availability_scenario"] == "all_record_upper_bound")
    ]
    if set(rows["target"]) != set(TARGETS):
        raise ValueError("Canonical q=10% threshold rows are incomplete")
    return {
        row.target: (float(row.threshold), round(float(row.threshold), 6))
        for row in rows.itertuples(index=False)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument(
        "--canonical-results",
        type=Path,
        default=Path("source_data/canonical_protocol_results.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = build_targets(pd.read_csv(args.data, dtype=str))
    frame["is_us_resident"] = pd.to_numeric(frame["is_us_resident"], errors="coerce")
    frame["is_singleton"] = pd.to_numeric(frame["is_singleton"], errors="coerce")
    cutoffs = load_cutoffs(args.canonical_results)
    results: list[dict[str, object]] = []

    for target in TARGETS:
        bundle = joblib.load(
            args.models_dir / f"{target}__landmark_strict__lightgbm.joblib"
        )
        features = list(bundle["features"])
        evaluation = frame[
            (frame["is_us_resident"] == 1)
            & (frame["is_singleton"] == 1)
            & frame[target].notna()
        ].copy()
        for feature in features:
            if feature != "mother_race":
                evaluation[feature] = pd.to_numeric(
                    evaluation[feature], errors="coerce"
                )

        transformed = bundle["preprocessor"].transform(evaluation[features])
        scores = bundle["model"].predict_proba(transformed)[:, 1]
        canonical_threshold, threshold_used = cutoffs[target]
        selected = scores >= threshold_used
        outcomes = evaluation[target].astype(int).to_numpy()

        n_all = len(outcomes)
        e_all = int(outcomes.sum())
        selected_n = int(selected.sum())
        true_positive_n = int(outcomes[selected].sum())
        nominal_selected_n = round(0.10 * n_all)
        results.append(
            {
                "target": target,
                "threshold_source_year": 2022,
                "nominal_fraction": 0.10,
                "canonical_threshold_exact": canonical_threshold,
                "threshold_used_6dp": threshold_used,
                "N_all": n_all,
                "E_all": e_all,
                "prevalence": e_all / n_all,
                "nominal_selected_n": nominal_selected_n,
                "selected_n": selected_n,
                "workload_difference_n": selected_n - nominal_selected_n,
                "selection_rate": selected_n / n_all,
                "TP": true_positive_n,
                "precision": true_positive_n / selected_n,
                "population_event_capture": true_positive_n / e_all,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
