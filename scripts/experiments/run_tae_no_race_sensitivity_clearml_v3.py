#!/usr/bin/env python3
"""
TAE 2026 no-race sensitivity.

Standalone ClearML sensitivity requested after the feature-timing audit.

Scientific question
-------------------
Do the main decision-aware evaluation conclusions materially change when
`mother_race` is removed from the conservative landmark feature set?

Design
------
- Same finalized ClearML Dataset as the canonical analysis.
- Same CDC 2022 -> CDC 2023 temporal design.
- Same target-specific primary populations.
- Same 2022 split assignment:
      seed = 2026
      calibration_fraction = 0.20
      split assigned BEFORE target-specific filtering.
- Same canonical LightGBM hyperparameters.
- Same canonical preprocessing settings.
- Remove ONLY `mother_race`.
- Fit exactly one new LightGBM per target (3 fits total).
- Evaluate CDC 2023 exactly once per target.
- q = 10% only, per mentor request.
- Evaluate:
      1) cohort-specific top-q
      2) fixed absolute budget
      3) transported 2022 score threshold
- Compute the four-cell early/all x early/all-budget design and symmetric
  Shapley point decomposition.
- Compare no-race point results directly with the frozen strict9 canonical
  protocol results and the completed strict9 Shapley Task.

No bootstrap is performed in this sensitivity script. The mentor requested
LightGBM, 3 targets, q=10%; this run is a point-estimate robustness analysis.
This intentionally avoids introducing an additional bootstrap convention while
the primary Shapley uncertainty analysis remains a separate artifact.

ClearML
-------
No GitHub checkout is required.
Task.force_store_standalone_script(True) stores this exact file in ClearML.

Inputs are obtained only from:
- finalized ClearML Dataset;
- frozen canonical ClearML Task;
- completed strict9 Shapley ClearML Task.

No local CDC path or repository file is required.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_VERSION = "2026-08-24-v3-no-race-auto-requirements"

DEFAULT_PROJECT_NAME = "pershin-medailab/Vache_Oganisyan/CDC Natality Audit"
DEFAULT_DATASET_ID = "062ba26c0ca24cef99549c2a2ab34e65"
DEFAULT_QUEUE = "6be0e69f7fab49d48bc305ef1fb03a6a"
DEFAULT_CANONICAL_TASK_ID = "4f9d98dd39f3438181f48b256b23bd94"
DEFAULT_SHAPLEY_TASK_ID = "72eae4b717794763a927e4c9a93220c2"
DEFAULT_TASK_NAME = "TAE 2026 sensitivity: LightGBM without maternal race"

TARGETS = ("target_preterm", "target_nicu", "target_lbw")
SCENARIOS = (
    "early_entry",
    "any_prenatal_care",
    "all_record_upper_bound",
)
Q = 0.10

STRICT9_FEATURES = (
    "mother_race",
    "mother_height_inches",
    "mother_bmi",
    "mother_weight_pre",
    "prior_live_births",
    "prior_dead_births",
    "prior_terminations",
    "diab_pre",
    "hyper_pre",
)

NO_RACE_FEATURES = tuple(
    feature for feature in STRICT9_FEATURES
    if feature != "mother_race"
)

EXPECTED_ROWS = {
    2022: 3_676_029,
    2023: 3_605_081,
}

EXPECTED_SHA256 = {
    2022: "c5f1ab62b0e9ac63795e75f6075d2523a3f4c5dc3e95c7aa1f7df6763205a59c",
    2023: "82b38b24d2f795a4a5233ce3a2fcb43b70789339306ba9044bde56f50bb270a8",
}

CANONICAL_ARTIFACTS = {
    "config": "experiment_config.locked.yaml",
    "manifest": "run_manifest.json",
    "protocol_results": "canonical_protocol_results.csv",
    "model_diagnostics": "model_diagnostics.csv",
}

SHAPLEY_ARTIFACTS = {
    "manifest": "run_manifest.json",
    "main_results": "shapley_main_results.csv",
}

TAGS = [
    "tae-2026",
    "cdc-natality",
    "sensitivity",
    "no-race",
    "lightgbm",
    "q-10pct",
    "care-entry-eligibility",
    "four-cell",
    "shapley",
    "transported-2022-threshold",
    "standalone-script",
]




# ---------------------------------------------------------------------------
# I/O and metadata
# ---------------------------------------------------------------------------

def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> str:
    lines = [
        f"python=={platform.python_version()}",
        f"platform={platform.platform()}",
    ]
    for package in (
        "clearml",
        "joblib",
        "lightgbm",
        "numpy",
        "pandas",
        "PyYAML",
        "scikit-learn",
    ):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT_INSTALLED"
        lines.append(f"{package}=={version}")
    return "\n".join(lines) + "\n"


def git_information() -> dict[str, Any]:
    def run_git(*args: str) -> Optional[str]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=Path.cwd(),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except Exception:
            return None

    status = run_git("status", "--porcelain")
    return {
        "commit_hash": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
        "required_for_remote_execution": False,
    }


# ---------------------------------------------------------------------------
# ClearML artifact and Dataset helpers
# ---------------------------------------------------------------------------

def artifact_local_path(task: Any, name: str) -> Path:
    if name not in task.artifacts:
        raise KeyError(
            f"Task {task.id} is missing artifact {name!r}. "
            f"Available: {sorted(task.artifacts.keys())}"
        )

    local = task.artifacts[name].get_local_copy()
    if not local:
        raise RuntimeError(
            f"Task {task.id} artifact {name!r} returned no local copy."
        )

    path = Path(local).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def bind_dataset(dataset_id: str, workers: int) -> Path:
    from clearml import Dataset

    dataset = Dataset.get(dataset_id=dataset_id)
    local = dataset.get_local_copy(
        max_workers=workers,
        raise_on_error=True,
    )
    root = Path(local).resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"ClearML Dataset local copy is not a directory: {root}"
        )
    return root


def resolve_dataset_file(dataset_root: Path, configured: str) -> Path:
    configured_path = Path(configured)

    candidates = (
        dataset_root / configured_path,
        dataset_root / configured_path.name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = [
        path for path in dataset_root.rglob(configured_path.name)
        if path.is_file()
    ]
    if len(matches) == 1:
        return matches[0].resolve()

    if len(matches) > 1:
        suffix = str(configured_path).replace("\\", "/")
        exact = [
            path for path in matches
            if str(path).replace("\\", "/").endswith(suffix)
        ]
        if len(exact) == 1:
            return exact[0].resolve()

    raise FileNotFoundError(
        f"Could not uniquely resolve {configured!r} under {dataset_root}; "
        f"matches={len(matches)}"
    )


# ---------------------------------------------------------------------------
# Canonical configuration validation
# ---------------------------------------------------------------------------

def load_source_lineage(
    canonical_task_id: str,
    shapley_task_id: str,
) -> dict[str, Any]:
    from clearml import Task

    canonical_task = Task.get_task(task_id=canonical_task_id)
    shapley_task = Task.get_task(task_id=shapley_task_id)

    canonical_config_path = artifact_local_path(
        canonical_task, CANONICAL_ARTIFACTS["config"]
    )
    canonical_manifest_path = artifact_local_path(
        canonical_task, CANONICAL_ARTIFACTS["manifest"]
    )
    canonical_protocol_path = artifact_local_path(
        canonical_task, CANONICAL_ARTIFACTS["protocol_results"]
    )

    # model_diagnostics is useful but not essential for the sensitivity.
    diagnostics_path: Optional[Path]
    if CANONICAL_ARTIFACTS["model_diagnostics"] in canonical_task.artifacts:
        diagnostics_path = artifact_local_path(
            canonical_task, CANONICAL_ARTIFACTS["model_diagnostics"]
        )
    else:
        diagnostics_path = None

    shapley_manifest_path = artifact_local_path(
        shapley_task, SHAPLEY_ARTIFACTS["manifest"]
    )
    shapley_main_path = artifact_local_path(
        shapley_task, SHAPLEY_ARTIFACTS["main_results"]
    )

    canonical_config = yaml.safe_load(
        canonical_config_path.read_text(encoding="utf-8")
    )
    canonical_manifest = json.loads(
        canonical_manifest_path.read_text(encoding="utf-8")
    )
    shapley_manifest = json.loads(
        shapley_manifest_path.read_text(encoding="utf-8")
    )

    if canonical_manifest.get("status") != "PASS":
        raise AssertionError("Canonical source manifest is not PASS.")
    if shapley_manifest.get("status") != "PASS":
        raise AssertionError("Strict9 Shapley source manifest is not PASS.")

    validate_canonical_config(canonical_config)

    if str(canonical_manifest.get("clearml_task_id")) != str(canonical_task_id):
        raise AssertionError("Canonical manifest Task ID mismatch.")
    if str(shapley_manifest.get("clearml_task_id")) != str(shapley_task_id):
        raise AssertionError("Shapley manifest Task ID mismatch.")

    canonical_dataset = str(
        canonical_manifest.get("clearml_dataset_id", "")
    )
    if canonical_dataset and canonical_dataset != DEFAULT_DATASET_ID:
        raise AssertionError("Canonical Dataset lineage mismatch.")

    shapley_dataset = str(
        shapley_manifest.get("canonical_source", {}).get("dataset_id", "")
    )
    if shapley_dataset and shapley_dataset != DEFAULT_DATASET_ID:
        raise AssertionError("Shapley Dataset lineage mismatch.")

    return {
        "canonical_task": canonical_task,
        "shapley_task": shapley_task,
        "canonical_config_path": canonical_config_path,
        "canonical_manifest_path": canonical_manifest_path,
        "canonical_protocol_path": canonical_protocol_path,
        "canonical_diagnostics_path": diagnostics_path,
        "shapley_manifest_path": shapley_manifest_path,
        "shapley_main_path": shapley_main_path,
        "canonical_config": canonical_config,
        "canonical_manifest": canonical_manifest,
        "shapley_manifest": shapley_manifest,
        "canonical_protocol": pd.read_csv(canonical_protocol_path),
        "canonical_diagnostics": (
            pd.read_csv(diagnostics_path)
            if diagnostics_path is not None
            else None
        ),
        "shapley_main": pd.read_csv(shapley_main_path),
    }


def validate_canonical_config(config: dict[str, Any]) -> None:
    if tuple(str(v) for v in config["targets"]) != TARGETS:
        raise AssertionError("Canonical targets changed.")

    feature_set = config["feature_set"]
    if str(feature_set["name"]) != "landmark_strict":
        raise AssertionError("Canonical feature-set name changed.")

    if tuple(str(v) for v in feature_set["features"]) != STRICT9_FEATURES:
        raise AssertionError("Canonical strict9 feature list changed.")

    if int(feature_set["n_features"]) != 9:
        raise AssertionError("Canonical raw feature count is not 9.")

    split = config["split"]
    if int(split["seed"]) != 2026:
        raise AssertionError("Canonical split seed changed.")
    if not math.isclose(
        float(split["calibration_fraction"]),
        0.20,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise AssertionError("Canonical calibration fraction changed.")

    if str(config["model"]["name"]) != "lightgbm":
        raise AssertionError("Canonical model family changed.")

    params = config["model"]["parameters"]
    expected_params = {
        "objective": "binary",
        "n_estimators": 350,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 100,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "class_weight": None,
        "scale_pos_weight": 1.0,
        "random_state": 2026,
        "n_jobs": -1,
        "verbosity": -1,
        "device_type": "cpu",
    }
    for key, expected in expected_params.items():
        if params.get(key) != expected:
            raise AssertionError(
                f"Canonical LightGBM parameter {key}={params.get(key)!r} "
                f"!= expected {expected!r}"
            )

    preprocessing = config["preprocessing"]
    if str(preprocessing["numeric_imputation"]) != "median":
        raise AssertionError("Canonical numeric imputation changed.")
    if not bool(preprocessing["standardize_numeric"]):
        raise AssertionError("Canonical numeric standardization changed.")

    hashes = config["data"]["expected_sha256"]
    rows = config["data"]["expected_rows"]
    for year in ("2022", "2023"):
        if str(hashes[year]) != EXPECTED_SHA256[int(year)]:
            raise AssertionError(f"Canonical CDC {year} hash changed.")
        if int(rows[year]) != EXPECTED_ROWS[int(year)]:
            raise AssertionError(f"Canonical CDC {year} row count changed.")


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

def required_columns() -> list[str]:
    return sorted(
        {
            "is_us_resident",
            "is_singleton",
            "prenatal_care_month",
            *TARGETS,
            *NO_RACE_FEATURES,
        }
    )


def validate_care_month(values: np.ndarray, label: str) -> None:
    care = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(care)
    observed = care[finite]

    if observed.size == 0:
        raise ValueError(f"{label}: no finite prenatal_care_month.")

    if np.any(observed < 0) or np.any(observed > 10):
        raise ValueError(
            f"{label}: prenatal_care_month outside harmonized 0..10."
        )

    if not np.allclose(
        observed,
        np.round(observed),
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError(
            f"{label}: non-integer prenatal_care_month values."
        )


def read_year(path: Path, year: int) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=required_columns(),
        low_memory=False,
    )

    if len(frame) != EXPECTED_ROWS[year]:
        raise AssertionError(
            f"CDC {year} rows={len(frame):,}, expected={EXPECTED_ROWS[year]:,}"
        )

    for column in ("is_us_resident", "is_singleton"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"CDC {year}/{column}: missing values.")
        if not set(values.unique()).issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"CDC {year}/{column}: not binary.")
        frame[column] = values

    for target in TARGETS:
        values = pd.to_numeric(frame[target], errors="coerce")
        if not set(values.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"CDC {year}/{target}: not binary.")
        frame[target] = values

    frame["prenatal_care_month"] = pd.to_numeric(
        frame["prenatal_care_month"], errors="coerce"
    )
    validate_care_month(
        frame["prenatal_care_month"].to_numpy(dtype=np.float64),
        f"CDC {year}",
    )

    for feature in NO_RACE_FEATURES:
        frame[feature] = pd.to_numeric(
            frame[feature], errors="coerce"
        )

    print(
        f"CDC {year}: {len(frame):,} rows, "
        f"{len(frame.columns)} required columns loaded"
    )
    return frame


def resident_singleton_mask(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["is_us_resident"].eq(1)
        & frame["is_singleton"].eq(1)
    ).to_numpy(dtype=bool)


def availability_masks(care_month: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(care_month, dtype=np.float64)
    validate_care_month(values, "availability_masks")

    finite = np.isfinite(values)
    early = finite & (values >= 1) & (values <= 3)
    any_care = finite & (values >= 1) & (values <= 10)
    all_record = np.ones(len(values), dtype=bool)

    if not np.all(~early | any_care):
        raise AssertionError("early is not subset of any-care.")
    if not np.all(~any_care | all_record):
        raise AssertionError("any-care is not subset of all-record.")

    return {
        "early_entry": early,
        "any_prenatal_care": any_care,
        "all_record_upper_bound": all_record,
    }


# ---------------------------------------------------------------------------
# Preprocessing/model
# ---------------------------------------------------------------------------

def build_preprocessor(
    features: list[str],
    categorical_features: list[str],
    config: dict[str, Any],
) -> ColumnTransformer:
    # Exact canonical implementation.
    categorical = [
        feature for feature in features
        if feature in categorical_features
    ]
    numeric = [
        feature for feature in features
        if feature not in categorical
    ]

    numeric_steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(
                strategy=str(config["numeric_imputation"])
            ),
        ),
    ]
    if bool(config.get("standardize_numeric", True)):
        numeric_steps.append(
            ("scaler", StandardScaler())
        )
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy=str(
                        config["categorical_imputation"]
                    )
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    transformers: list[
        tuple[str, Any, list[str]]
    ] = []
    if numeric:
        transformers.append(
            ("numeric", numeric_pipeline, numeric)
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical,
            )
        )
    if not transformers:
        raise ValueError("No features supplied.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )


def consistent_matrix(matrix: Any) -> Any:
    if isinstance(matrix, pd.DataFrame):
        return matrix.to_numpy(
            dtype=np.float32, copy=False
        )
    if hasattr(matrix, "astype"):
        return matrix.astype(
            np.float32, copy=False
        )
    return np.asarray(matrix, dtype=np.float32)


# ---------------------------------------------------------------------------
# Selection and metrics
# ---------------------------------------------------------------------------

def rank_eligible_indices(
    probabilities: np.ndarray,
    eligible_mask: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(
        probabilities, dtype=np.float64
    )
    eligible_indices = np.flatnonzero(
        np.asarray(eligible_mask, dtype=bool)
    )
    eligible_probabilities = probabilities[
        eligible_indices
    ]
    order = np.lexsort(
        (eligible_indices, -eligible_probabilities)
    )
    return eligible_indices[order]


def threshold_for_capacity(
    probabilities: np.ndarray,
    capacity: float,
) -> float:
    values = np.asarray(
        probabilities, dtype=np.float64
    )
    if not 0.0 < capacity < 1.0:
        raise ValueError(f"Invalid capacity {capacity}")
    selected_n = max(
        1,
        min(
            len(values),
            int(round(capacity * len(values))),
        ),
    )
    partition_index = len(values) - selected_n
    return float(
        np.partition(
            values, partition_index
        )[partition_index]
    )


def access_metrics(
    *,
    y: np.ndarray,
    eligible_mask: np.ndarray,
    selected_indices: np.ndarray,
) -> dict[str, Any]:
    y_values = np.asarray(y, dtype=np.int8)
    eligible = np.asarray(
        eligible_mask, dtype=bool
    )
    selected = np.asarray(
        selected_indices, dtype=np.int64
    )
    eligible_indices = np.flatnonzero(eligible)

    full_n = len(y_values)
    full_events = int(
        np.sum(y_values == 1)
    )
    eligible_n = len(eligible_indices)
    eligible_events = int(
        np.sum(
            y_values[eligible_indices] == 1
        )
    )
    selected_n = len(selected)
    tp = (
        int(np.sum(y_values[selected] == 1))
        if selected_n
        else 0
    )

    return {
        "full_population_n": int(full_n),
        "full_population_event_n": full_events,
        "eligible_n": int(eligible_n),
        "eligible_event_n": eligible_events,
        "selected_n": int(selected_n),
        "true_positive_n": tp,
        "selection_rate_among_eligible": (
            float(selected_n / eligible_n)
            if eligible_n
            else math.nan
        ),
        "precision": (
            float(tp / selected_n)
            if selected_n
            else math.nan
        ),
        "recall_among_eligible": (
            float(tp / eligible_events)
            if eligible_events
            else math.nan
        ),
        "population_event_capture": (
            float(tp / full_events)
            if full_events
            else math.nan
        ),
        "event_availability_ceiling": (
            float(eligible_events / full_events)
            if full_events
            else math.nan
        ),
    }


def evaluate_protocols(
    *,
    target: str,
    y: np.ndarray,
    probabilities: np.ndarray,
    care_month: np.ndarray,
    threshold_2022: float,
) -> list[dict[str, Any]]:
    masks = availability_masks(care_month)
    rankings = {
        scenario: rank_eligible_indices(
            probabilities, masks[scenario]
        )
        for scenario in SCENARIOS
    }

    rows: list[dict[str, Any]] = []
    n_all = len(y)

    # Protocol A: cohort-specific top 10%.
    for scenario in SCENARIOS:
        eligible = masks[scenario]
        eligible_n = int(np.sum(eligible))
        budget = max(
            1, int(round(Q * eligible_n))
        )
        selected = rankings[scenario][:budget]
        if len(selected) != budget:
            raise AssertionError(
                "Top-q exact budget failed."
            )
        rows.append(
            {
                "target": target,
                "feature_set": "landmark_no_race_8",
                "model": "lightgbm",
                "evaluation": "temporal_evaluation_cohort_2023",
                "protocol": "cohort_specific_top_fraction",
                "availability_scenario": scenario,
                "nominal_fraction": Q,
                "budget_n": budget,
                "threshold": "",
                "threshold_source_year": "",
                **access_metrics(
                    y=y,
                    eligible_mask=eligible,
                    selected_indices=selected,
                ),
            }
        )

    # Protocol B: fixed all-record absolute budget.
    fixed_budget = max(
        1, int(round(Q * n_all))
    )
    for scenario in SCENARIOS:
        eligible = masks[scenario]
        if fixed_budget > int(np.sum(eligible)):
            raise ValueError(
                f"{target}/{scenario}: fixed budget exceeds candidate set."
            )
        selected = rankings[scenario][
            :fixed_budget
        ]
        rows.append(
            {
                "target": target,
                "feature_set": "landmark_no_race_8",
                "model": "lightgbm",
                "evaluation": "temporal_evaluation_cohort_2023",
                "protocol": "fixed_absolute_budget",
                "availability_scenario": scenario,
                "nominal_fraction": Q,
                "budget_n": fixed_budget,
                "threshold": "",
                "threshold_source_year": "",
                **access_metrics(
                    y=y,
                    eligible_mask=eligible,
                    selected_indices=selected,
                ),
            }
        )

    # Protocol C: threshold transported from 2022 no-race calibration scores.
    for scenario in SCENARIOS:
        eligible = masks[scenario]
        selected = np.flatnonzero(
            eligible
            & (
                np.asarray(
                    probabilities,
                    dtype=np.float64,
                )
                >= float(threshold_2022)
            )
        )
        rows.append(
            {
                "target": target,
                "feature_set": "landmark_no_race_8",
                "model": "lightgbm",
                "evaluation": "temporal_evaluation_cohort_2023",
                "protocol": "transported_2022_score_threshold",
                "availability_scenario": scenario,
                "nominal_fraction": Q,
                "budget_n": "",
                "threshold": float(
                    threshold_2022
                ),
                "threshold_source_year": 2022,
                **access_metrics(
                    y=y,
                    eligible_mask=eligible,
                    selected_indices=selected,
                ),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Four-cell/Shapley point decomposition
# ---------------------------------------------------------------------------

def component_values(
    c00: float,
    c10: float,
    c01: float,
    c11: float,
) -> dict[str, float]:
    availability_main = c10 - c00
    capacity_main = c01 - c00
    interaction = (
        c11 - c10 - c01 + c00
    )

    availability_at_all_budget = c11 - c01
    capacity_in_all = c11 - c10

    shapley_availability = 0.5 * (
        availability_main
        + availability_at_all_budget
    )
    shapley_capacity = 0.5 * (
        capacity_main
        + capacity_in_all
    )
    total = c11 - c00

    return {
        "availability_main_at_early_budget": availability_main,
        "capacity_main_in_early_cohort": capacity_main,
        "interaction": interaction,
        "ordered_availability_at_all_budget": availability_at_all_budget,
        "ordered_capacity_within_early_cohort": capacity_main,
        "availability_at_early_budget": availability_main,
        "capacity_in_all_cohort": capacity_in_all,
        "shapley_availability": shapley_availability,
        "shapley_capacity": shapley_capacity,
        "total_topq_contrast": total,
        "identity_factorial_error": total - (
            availability_main
            + capacity_main
            + interaction
        ),
        "identity_shapley_error": total - (
            shapley_availability
            + shapley_capacity
        ),
    }


def four_cell_shapley(
    *,
    target: str,
    y: np.ndarray,
    probabilities: np.ndarray,
    care_month: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    masks = availability_masks(care_month)
    early = masks["early_entry"]
    all_record = masks[
        "all_record_upper_bound"
    ]

    rank_early = rank_eligible_indices(
        probabilities, early
    )
    rank_all = rank_eligible_indices(
        probabilities, all_record
    )

    n_early = int(np.sum(early))
    n_all = len(y)
    e_early = int(np.sum(y[early] == 1))
    e_all = int(np.sum(y == 1))

    b_early = max(
        1, int(round(Q * n_early))
    )
    b_all = max(
        1, int(round(Q * n_all))
    )
    if b_all > n_early:
        raise ValueError(
            f"{target}: B_all exceeds early candidate set."
        )

    specs = (
        (
            "C00",
            "early_entry",
            "early",
            rank_early,
            b_early,
            n_early,
            e_early,
        ),
        (
            "C10",
            "all_record_upper_bound",
            "early",
            rank_all,
            b_early,
            n_all,
            e_all,
        ),
        (
            "C01",
            "early_entry",
            "all",
            rank_early,
            b_all,
            n_early,
            e_early,
        ),
        (
            "C11",
            "all_record_upper_bound",
            "all",
            rank_all,
            b_all,
            n_all,
            e_all,
        ),
    )

    rows: list[dict[str, Any]] = []
    captures: dict[str, float] = {}
    tps: dict[str, int] = {}

    for (
        cell,
        candidate_set,
        budget_source,
        rank,
        budget,
        candidate_n,
        candidate_events,
    ) in specs:
        selected = rank[:budget]
        tp = int(
            np.sum(y[selected] == 1)
        )
        capture = float(tp / e_all)

        rows.append(
            {
                "target": target,
                "feature_set": "landmark_no_race_8",
                "model": "lightgbm",
                "evaluation": "temporal_evaluation_cohort_2023",
                "nominal_fraction": Q,
                "cell": cell,
                "candidate_set": candidate_set,
                "budget_source": budget_source,
                "candidate_n": candidate_n,
                "candidate_event_n": candidate_events,
                "full_population_n": n_all,
                "full_population_event_n": e_all,
                "budget_n": budget,
                "selected_n": budget,
                "true_positive_n": tp,
                "precision": float(tp / budget),
                "population_event_capture": capture,
            }
        )
        captures[cell] = capture
        tps[cell] = tp

    components = component_values(
        captures["C00"],
        captures["C10"],
        captures["C01"],
        captures["C11"],
    )

    if abs(
        components[
            "identity_factorial_error"
        ]
    ) > 1e-12:
        raise AssertionError(
            "No-race factorial identity failed."
        )
    if abs(
        components[
            "identity_shapley_error"
        ]
    ) > 1e-12:
        raise AssertionError(
            "No-race Shapley identity failed."
        )

    decomposition = {
        "target": target,
        "feature_set": "landmark_no_race_8",
        "model": "lightgbm",
        "evaluation": "temporal_evaluation_cohort_2023",
        "nominal_fraction": Q,
        "n_early": n_early,
        "n_all": n_all,
        "event_n_early": e_early,
        "event_n_all": e_all,
        "B_early": b_early,
        "B_all": b_all,
        "C00_early_Bearly": captures["C00"],
        "C10_all_Bearly": captures["C10"],
        "C01_early_Ball": captures["C01"],
        "C11_all_Ball": captures["C11"],
        "C00_tp": tps["C00"],
        "C10_tp": tps["C10"],
        "C01_tp": tps["C01"],
        "C11_tp": tps["C11"],
        **components,
    }
    return rows, decomposition


def validate_split_counts_against_canonical(
    *,
    canonical_diagnostics: Optional[pd.DataFrame],
    target: str,
    train_n: int,
    calibration_n: int,
    evaluation_n: int,
) -> dict[str, Any]:
    """
    Feature removal must not change sample membership.

    The canonical model_diagnostics artifact records the target-specific
    train/calibration/evaluation counts. Compare those counts exactly when the
    artifact is available.
    """
    row = {
        "target": target,
        "no_race_train_n": int(train_n),
        "no_race_calibration_n": int(calibration_n),
        "no_race_evaluation_n": int(evaluation_n),
        "status": "PASS",
    }

    if canonical_diagnostics is None:
        row.update(
            {
                "canonical_train_n": "",
                "canonical_calibration_n": "",
                "canonical_evaluation_n": "",
                "status": "NOT_AVAILABLE",
            }
        )
        return row

    frame = canonical_diagnostics
    if "target" not in frame.columns:
        raise ValueError(
            "Canonical model_diagnostics.csv lacks target column."
        )

    rows = frame.loc[
        frame["target"].astype(str).eq(target)
    ]
    if len(rows) != 1:
        raise AssertionError(
            f"Expected one canonical diagnostics row for {target}; got {len(rows)}"
        )
    canonical = rows.iloc[0]

    required = (
        "train_n",
        "calibration_2022_n",
        "temporal_test_2023_n",
    )
    missing = [
        column for column in required
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"Canonical model_diagnostics.csv lacks columns: {missing}"
        )

    expected = {
        "train_n": int(canonical["train_n"]),
        "calibration_n": int(
            canonical["calibration_2022_n"]
        ),
        "evaluation_n": int(
            canonical["temporal_test_2023_n"]
        ),
    }
    observed = {
        "train_n": int(train_n),
        "calibration_n": int(calibration_n),
        "evaluation_n": int(evaluation_n),
    }
    if observed != expected:
        raise AssertionError(
            f"{target}: no-race split counts {observed} "
            f"!= canonical split counts {expected}"
        )

    row.update(
        {
            "canonical_train_n": expected["train_n"],
            "canonical_calibration_n": expected[
                "calibration_n"
            ],
            "canonical_evaluation_n": expected[
                "evaluation_n"
            ],
        }
    )
    return row


# ---------------------------------------------------------------------------
# Canonical cross-checks and strict9 comparison
# ---------------------------------------------------------------------------

def canonical_row(
    frame: pd.DataFrame,
    *,
    target: str,
    protocol: str,
    scenario: str,
) -> pd.Series:
    actual_protocol = protocol
    if protocol == "transported_2022_score_threshold":
        # Internal canonical artifact uses the old technical label.
        actual_protocol = "fixed_calibration_threshold"

    mask = (
        frame["target"].astype(str).eq(
            target
        )
        & frame["protocol"].astype(str).eq(
            actual_protocol
        )
        & frame[
            "availability_scenario"
        ].astype(str).eq(scenario)
        & np.isclose(
            frame[
                "nominal_fraction"
            ].astype(float),
            Q,
            atol=0.0,
            rtol=0.0,
        )
    )
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise AssertionError(
            f"Expected one canonical row for "
            f"{target}/{protocol}/{scenario}; got {len(rows)}"
        )
    return rows.iloc[0]


def validate_population_against_canonical(
    no_race_rows: list[dict[str, Any]],
    canonical_protocol: pd.DataFrame,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    # Candidate/event counts must be feature-invariant.
    for target in TARGETS:
        for scenario in SCENARIOS:
            nr = [
                row
                for row in no_race_rows
                if row["target"] == target
                and row["availability_scenario"]
                == scenario
                and row["protocol"]
                == "fixed_absolute_budget"
            ]
            if len(nr) != 1:
                raise AssertionError(
                    "No-race population cross-check row missing."
                )
            nr_row = nr[0]

            canonical = canonical_row(
                canonical_protocol,
                target=target,
                protocol="fixed_absolute_budget",
                scenario=scenario,
            )

            fields = (
                "full_population_n",
                "full_population_event_n",
                "eligible_n",
                "eligible_event_n",
            )
            for field in fields:
                got = int(nr_row[field])
                want = int(canonical[field])
                if got != want:
                    raise AssertionError(
                        f"{target}/{scenario}/{field}: "
                        f"no-race={got} canonical={want}"
                    )

            checks.append(
                {
                    "target": target,
                    "scenario": scenario,
                    "status": "PASS",
                    "full_population_n": int(
                        nr_row["full_population_n"]
                    ),
                    "full_population_event_n": int(
                        nr_row[
                            "full_population_event_n"
                        ]
                    ),
                    "eligible_n": int(
                        nr_row["eligible_n"]
                    ),
                    "eligible_event_n": int(
                        nr_row["eligible_event_n"]
                    ),
                }
            )

    return checks


def strict9_shapley_row(
    frame: pd.DataFrame,
    target: str,
) -> pd.Series:
    mask = (
        frame["target"].astype(str).eq(target)
        & np.isclose(
            frame[
                "nominal_fraction"
            ].astype(float),
            Q,
            atol=0.0,
            rtol=0.0,
        )
    )
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise AssertionError(
            f"Expected one strict9 Shapley q=10 row for {target}; "
            f"got {len(rows)}"
        )
    return rows.iloc[0]


def early_all_contrast(
    rows: list[dict[str, Any]],
    *,
    target: str,
    protocol: str,
) -> tuple[float, int, int]:
    subset = [
        row
        for row in rows
        if row["target"] == target
        and row["protocol"] == protocol
    ]

    by_scenario = {
        row["availability_scenario"]: row
        for row in subset
    }
    if (
        "early_entry" not in by_scenario
        or "all_record_upper_bound"
        not in by_scenario
    ):
        raise AssertionError(
            f"No-race contrast missing {target}/{protocol}"
        )

    early = by_scenario["early_entry"]
    all_row = by_scenario[
        "all_record_upper_bound"
    ]

    delta_capture = float(
        all_row["population_event_capture"]
        - early["population_event_capture"]
    )
    delta_selected = int(
        all_row["selected_n"]
        - early["selected_n"]
    )
    delta_tp = int(
        all_row["true_positive_n"]
        - early["true_positive_n"]
    )
    return (
        delta_capture,
        delta_selected,
        delta_tp,
    )


def canonical_early_all_contrast(
    canonical: pd.DataFrame,
    *,
    target: str,
    protocol: str,
) -> tuple[float, int, int]:
    early = canonical_row(
        canonical,
        target=target,
        protocol=protocol,
        scenario="early_entry",
    )
    all_row = canonical_row(
        canonical,
        target=target,
        protocol=protocol,
        scenario="all_record_upper_bound",
    )

    return (
        float(
            all_row[
                "population_event_capture"
            ]
            - early[
                "population_event_capture"
            ]
        ),
        int(
            all_row["selected_n"]
            - early["selected_n"]
        ),
        int(
            all_row["true_positive_n"]
            - early["true_positive_n"]
        ),
    )


def make_comparison_to_strict9(
    *,
    no_race_protocol_rows: list[dict[str, Any]],
    no_race_shapley_rows: list[dict[str, Any]],
    canonical_protocol: pd.DataFrame,
    strict9_shapley: pd.DataFrame,
) -> list[dict[str, Any]]:
    nr_shapley_by_target = {
        row["target"]: row
        for row in no_race_shapley_rows
    }

    rows: list[dict[str, Any]] = []

    for target in TARGETS:
        nr_topq = early_all_contrast(
            no_race_protocol_rows,
            target=target,
            protocol="cohort_specific_top_fraction",
        )
        s9_topq = canonical_early_all_contrast(
            canonical_protocol,
            target=target,
            protocol="cohort_specific_top_fraction",
        )

        nr_fixed = early_all_contrast(
            no_race_protocol_rows,
            target=target,
            protocol="fixed_absolute_budget",
        )
        s9_fixed = canonical_early_all_contrast(
            canonical_protocol,
            target=target,
            protocol="fixed_absolute_budget",
        )

        nr_threshold = early_all_contrast(
            no_race_protocol_rows,
            target=target,
            protocol="transported_2022_score_threshold",
        )
        s9_threshold = canonical_early_all_contrast(
            canonical_protocol,
            target=target,
            protocol="transported_2022_score_threshold",
        )

        nr_s = nr_shapley_by_target[target]
        s9_s = strict9_shapley_row(
            strict9_shapley, target
        )

        row = {
            "target": target,
            "nominal_fraction": Q,

            "strict9_topq_total_pp": 100.0 * s9_topq[0],
            "no_race_topq_total_pp": 100.0 * nr_topq[0],
            "delta_no_race_minus_strict9_topq_total_pp": (
                100.0 * (nr_topq[0] - s9_topq[0])
            ),

            "strict9_fixed_budget_early_all_pp": 100.0 * s9_fixed[0],
            "no_race_fixed_budget_early_all_pp": 100.0 * nr_fixed[0],
            "delta_no_race_minus_strict9_fixed_budget_pp": (
                100.0 * (nr_fixed[0] - s9_fixed[0])
            ),

            "strict9_threshold_early_all_pp": 100.0 * s9_threshold[0],
            "no_race_threshold_early_all_pp": 100.0 * nr_threshold[0],
            "delta_no_race_minus_strict9_threshold_pp": (
                100.0 * (nr_threshold[0] - s9_threshold[0])
            ),

            "strict9_shapley_availability_pp": float(
                s9_s["shapley_availability_point_pp"]
            ),
            "no_race_shapley_availability_pp": 100.0 * float(
                nr_s["shapley_availability"]
            ),
            "delta_no_race_minus_strict9_shapley_availability_pp": (
                100.0 * float(
                    nr_s["shapley_availability"]
                )
                - float(
                    s9_s[
                        "shapley_availability_point_pp"
                    ]
                )
            ),

            "strict9_shapley_capacity_pp": float(
                s9_s["shapley_capacity_point_pp"]
            ),
            "no_race_shapley_capacity_pp": 100.0 * float(
                nr_s["shapley_capacity"]
            ),
            "delta_no_race_minus_strict9_shapley_capacity_pp": (
                100.0 * float(
                    nr_s["shapley_capacity"]
                )
                - float(
                    s9_s[
                        "shapley_capacity_point_pp"
                    ]
                )
            ),

            "strict9_interaction_pp": float(
                s9_s["interaction_point_pp"]
            ),
            "no_race_interaction_pp": 100.0 * float(
                nr_s["interaction"]
            ),
            "delta_no_race_minus_strict9_interaction_pp": (
                100.0 * float(
                    nr_s["interaction"]
                )
                - float(
                    s9_s["interaction_point_pp"]
                )
            ),
        }

        # Useful relative stability quantities, but do not define an arbitrary
        # pass/fail threshold in code.
        if abs(row["strict9_topq_total_pp"]) > 0:
            row["no_race_over_strict9_topq_ratio"] = (
                row["no_race_topq_total_pp"]
                / row["strict9_topq_total_pp"]
            )
        else:
            row["no_race_over_strict9_topq_ratio"] = math.nan

        if abs(
            row["strict9_shapley_availability_pp"]
        ) > 0:
            row[
                "no_race_over_strict9_shapley_availability_ratio"
            ] = (
                row["no_race_shapley_availability_pp"]
                / row[
                    "strict9_shapley_availability_pp"
                ]
            )
        else:
            row[
                "no_race_over_strict9_shapley_availability_ratio"
            ] = math.nan

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Diagnostics and summary
# ---------------------------------------------------------------------------

def diagnostics_row(
    *,
    target: str,
    split: str,
    y: np.ndarray,
    probabilities: np.ndarray,
    transformed_feature_n: int,
) -> dict[str, Any]:
    return {
        "target": target,
        "feature_set": "landmark_no_race_8",
        "model": "lightgbm",
        "split": split,
        "n": int(len(y)),
        "event_n": int(np.sum(y == 1)),
        "prevalence": float(np.mean(y)),
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),
        "average_precision": float(
            average_precision_score(
                y, probabilities
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y, probabilities
            )
        ),
        "raw_feature_n": len(
            NO_RACE_FEATURES
        ),
        "transformed_feature_n": transformed_feature_n,
    }


def summary_markdown(
    comparisons: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> str:
    labels = {
        "target_preterm": "Preterm delivery",
        "target_nicu": "NICU admission",
        "target_lbw": "Low birth weight",
    }

    lines = [
        "# No-race sensitivity",
        "",
        "This sensitivity removes `mother_race` and refits one LightGBM per target using the same CDC 2022 development/calibration split, the same canonical hyperparameters, and the remaining eight conservative landmark features.",
        "",
        "CDC 2023 remains the temporal evaluation cohort. q=10% is used throughout.",
        "",
        "## Decision-aware comparison to strict9",
        "",
        "| Outcome | Top-q total strict9 | Top-q total no-race | Shapley availability strict9 | Shapley availability no-race | Shapley capacity strict9 | Shapley capacity no-race | Interaction strict9 | Interaction no-race |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in comparisons:
        lines.append(
            f"| {labels.get(row['target'], row['target'])} "
            f"| {row['strict9_topq_total_pp']:+.3f} "
            f"| {row['no_race_topq_total_pp']:+.3f} "
            f"| {row['strict9_shapley_availability_pp']:+.3f} "
            f"| {row['no_race_shapley_availability_pp']:+.3f} "
            f"| {row['strict9_shapley_capacity_pp']:+.3f} "
            f"| {row['no_race_shapley_capacity_pp']:+.3f} "
            f"| {row['strict9_interaction_pp']:+.3f} "
            f"| {row['no_race_interaction_pp']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "All contrast values are percentage-point differences in population event capture.",
            "",
            "## Fixed-budget and transported-threshold contrasts",
            "",
            "| Outcome | Fixed B strict9 | Fixed B no-race | Transported threshold strict9 | Transported threshold no-race |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {labels.get(row['target'], row['target'])} "
            f"| {row['strict9_fixed_budget_early_all_pp']:+.3f} "
            f"| {row['no_race_fixed_budget_early_all_pp']:+.3f} "
            f"| {row['strict9_threshold_early_all_pp']:+.3f} "
            f"| {row['no_race_threshold_early_all_pp']:+.3f} |"
        )

    d = pd.DataFrame(diagnostics)
    eval_rows = d.loc[
        d["split"].eq(
            "temporal_evaluation_cohort_2023"
        )
    ]

    lines.extend(
        [
            "",
            "## No-race model diagnostics on CDC 2023",
            "",
            "| Outcome | ROC AUC | Average precision | Brier |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in eval_rows.iterrows():
        lines.append(
            f"| {labels.get(row['target'], row['target'])} "
            f"| {float(row['roc_auc']):.4f} "
            f"| {float(row['average_precision']):.4f} "
            f"| {float(row['brier_score']):.4f} |"
        )

    lines.extend(
        [
            "",
            "This is a point-estimate sensitivity analysis; no new bootstrap uncertainty is introduced here.",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ClearML upload
# ---------------------------------------------------------------------------

def upload_outputs(
    task: Any,
    output: Path,
) -> None:
    for filename in (
        "no_race_protocol_results_10pct.csv",
        "no_race_four_cell_10pct.csv",
        "no_race_shapley_10pct.csv",
        "no_race_thresholds_2022_10pct.csv",
        "no_race_model_diagnostics.csv",
        "no_race_population_crosscheck.csv",
        "no_race_split_crosscheck.csv",
        "no_race_comparison_to_strict9.csv",
        "no_race_summary.md",
        "run_manifest.json",
        "package_versions.txt",
        "commands.txt",
        "git_info.json",
    ):
        path = output / filename
        if path.is_file():
            task.upload_artifact(
                name=filename,
                artifact_object=str(path),
                wait_on_upload=True,
            )

    models = output / "models"
    if models.is_dir():
        task.upload_artifact(
            name="tae_no_race_models",
            artifact_object=str(models),
            wait_on_upload=True,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test() -> int:
    rng = np.random.default_rng(20260824)
    n = 20_000

    z = rng.normal(size=n)
    y = rng.binomial(
        1,
        1.0 / (
            1.0
            + np.exp(-(-2.4 + 0.9 * z))
        ),
    ).astype(np.int8)
    scores = z + rng.normal(
        scale=0.8, size=n
    )

    care = rng.choice(
        [
            1.0, 2.0, 3.0,
            4.0, 6.0, 9.0,
            0.0, np.nan,
        ],
        size=n,
        p=[
            0.22, 0.20, 0.18,
            0.10, 0.08, 0.06,
            0.08, 0.08,
        ],
    )

    masks = availability_masks(care)
    if not np.all(
        ~masks["early_entry"]
        | masks["any_prenatal_care"]
    ):
        raise AssertionError(
            "Smoke scenario nesting failed."
        )

    cells, shapley = four_cell_shapley(
        target="smoke_target",
        y=y,
        probabilities=scores,
        care_month=care,
    )
    if len(cells) != 4:
        raise AssertionError(
            "Smoke four-cell row count failed."
        )
    if abs(
        shapley["identity_shapley_error"]
    ) > 1e-12:
        raise AssertionError(
            "Smoke Shapley identity failed."
        )

    threshold = threshold_for_capacity(
        scores[:10_000], Q
    )
    protocol_rows = evaluate_protocols(
        target="smoke_target",
        y=y,
        probabilities=scores,
        care_month=care,
        threshold_2022=threshold,
    )
    if len(protocol_rows) != 9:
        raise AssertionError(
            "Smoke protocol row count failed."
        )

    print("NO-RACE SENSITIVITY SMOKE TEST PASS")
    print("scenario_nesting=PASS")
    print("three_protocols=PASS")
    print("four_cell=PASS")
    print("shapley_identity=PASS")
    return 0


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TAE 2026 standalone ClearML "
            "no-race LightGBM sensitivity"
        )
    )
    parser.add_argument(
        "--dataset-id",
        default=(
            os.getenv("CLEARML_DATASET_ID")
            or DEFAULT_DATASET_ID
        ),
    )
    parser.add_argument(
        "--canonical-task-id",
        default=(
            os.getenv(
                "TAE_CANONICAL_TASK_ID"
            )
            or DEFAULT_CANONICAL_TASK_ID
        ),
    )
    parser.add_argument(
        "--shapley-task-id",
        default=(
            os.getenv(
                "TAE_SHAPLEY_TASK_ID"
            )
            or DEFAULT_SHAPLEY_TASK_ID
        ),
    )
    parser.add_argument(
        "--queue",
        default=(
            os.getenv("CLEARML_QUEUE_NAME")
            or os.getenv("CLEARML_QUEUE_ID")
            or DEFAULT_QUEUE
        ),
    )
    parser.add_argument(
        "--project-name",
        default=(
            os.getenv(
                "CLEARML_PROJECT_NAME"
            )
            or DEFAULT_PROJECT_NAME
        ),
    )
    parser.add_argument(
        "--task-name",
        default=(
            os.getenv(
                "CLEARML_TASK_NAME_TAE_NO_RACE"
            )
            or DEFAULT_TASK_NAME
        ),
    )
    parser.add_argument(
        "--dataset-download-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results/"
            "tae_2026_no_race_sensitivity"
        ),
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Run on current machine while still "
            "retrieving ClearML Dataset/artifacts. "
            "Normal publication run should omit --local."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.smoke_test:
        return smoke_test()

    if str(args.dataset_id) != DEFAULT_DATASET_ID:
        raise ValueError(
            "Dataset differs from locked canonical Dataset."
        )
    if str(args.canonical_task_id) != DEFAULT_CANONICAL_TASK_ID:
        raise ValueError(
            "Canonical Task differs from locked source."
        )
    if str(args.shapley_task_id) != DEFAULT_SHAPLEY_TASK_ID:
        raise ValueError(
            "Strict9 Shapley Task differs from audited source."
        )
    if str(args.queue) != DEFAULT_QUEUE:
        raise ValueError(
            "Queue differs from locked canonical queue."
        )

    from clearml import Task

    Task.force_store_standalone_script(True)
    # No Task.add_requirements call: ClearML must capture exact installed
    # package versions from top-level imports only.
    task = Task.init(
        project_name=args.project_name,
        task_name=args.task_name,
        task_type=Task.TaskTypes.training,
        tags=TAGS,
        reuse_last_task_id=False,
        output_uri=True,
    )

    runtime = task.connect(
        {
            "dataset_id": str(
                args.dataset_id
            ),
            "canonical_task_id": str(
                args.canonical_task_id
            ),
            "strict9_shapley_task_id": str(
                args.shapley_task_id
            ),
            "queue": str(args.queue),
            "dataset_download_workers": int(
                args.dataset_download_workers
            ),
            "output_dir": str(
                args.output_dir
            ),
            "q": Q,
            "removed_feature": "mother_race",
        },
        name="runtime",
    )

    args.dataset_id = str(
        runtime["dataset_id"]
    )
    args.canonical_task_id = str(
        runtime["canonical_task_id"]
    )
    args.shapley_task_id = str(
        runtime[
            "strict9_shapley_task_id"
        ]
    )
    args.queue = str(runtime["queue"])
    args.dataset_download_workers = int(
        runtime[
            "dataset_download_workers"
        ]
    )
    args.output_dir = str(
        runtime["output_dir"]
    )

    # Prevent remote parameter edits from silently changing scientific lineage.
    if args.dataset_id != DEFAULT_DATASET_ID:
        raise ValueError("Runtime changed Dataset ID.")
    if args.canonical_task_id != DEFAULT_CANONICAL_TASK_ID:
        raise ValueError("Runtime changed canonical Task ID.")
    if args.shapley_task_id != DEFAULT_SHAPLEY_TASK_ID:
        raise ValueError("Runtime changed Shapley Task ID.")
    if args.queue != DEFAULT_QUEUE:
        raise ValueError("Runtime changed queue.")

    if not args.local:
        task.execute_remotely(
            queue_name=args.queue,
            clone=False,
            exit_process=True,
        )

    started = time.time()

    print(
        f"No-race ClearML Task ID: {task.id}"
    )
    print(
        f"Canonical source Task: "
        f"{args.canonical_task_id}"
    )
    print(
        f"Strict9 Shapley source Task: "
        f"{args.shapley_task_id}"
    )
    print(
        f"Dataset: {args.dataset_id}"
    )

    lineage = load_source_lineage(
        args.canonical_task_id,
        args.shapley_task_id,
    )
    config = lineage[
        "canonical_config"
    ]

    dataset_root = bind_dataset(
        args.dataset_id,
        args.dataset_download_workers,
    )

    paths = {
        2022: resolve_dataset_file(
            dataset_root,
            str(config["data"]["cdc2022"]),
        ),
        2023: resolve_dataset_file(
            dataset_root,
            str(config["data"]["cdc2023"]),
        ),
    }

    observed_hashes: dict[int, str] = {}
    for year in (2022, 2023):
        print(f"Hashing CDC {year}...")
        digest = sha256_file(
            paths[year]
        )
        if digest != EXPECTED_SHA256[year]:
            raise AssertionError(
                f"CDC {year} hash mismatch."
            )
        observed_hashes[year] = digest

    frame_2022 = read_year(
        paths[2022], 2022
    )
    frame_2023 = read_year(
        paths[2023], 2023
    )

    primary_2022 = resident_singleton_mask(
        frame_2022
    )
    primary_2023 = resident_singleton_mask(
        frame_2023
    )

    # Exact canonical split assignment: before target filtering.
    split_rng = np.random.default_rng(
        int(config["split"]["seed"])
    )
    calibration_mask_all_2022 = (
        split_rng.random(
            len(frame_2022)
        )
        < float(
            config["split"][
                "calibration_fraction"
            ]
        )
    )
    train_mask_all_2022 = (
        ~calibration_mask_all_2022
    )

    if np.any(
        train_mask_all_2022
        & calibration_mask_all_2022
    ):
        raise AssertionError(
            "Train/calibration overlap."
        )
    if not np.all(
        train_mask_all_2022
        | calibration_mask_all_2022
    ):
        raise AssertionError(
            "Train/calibration split does not cover CDC 2022."
        )

    output = Path(
        args.output_dir
    ).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output

    if output.exists() and any(
        output.iterdir()
    ):
        if not args.overwrite:
            raise FileExistsError(
                f"{output} is not empty. Use --overwrite."
            )
        shutil.rmtree(output)
    output.mkdir(
        parents=True, exist_ok=True
    )
    models_dir = output / "models"
    models_dir.mkdir(
        parents=True, exist_ok=True
    )

    atomic_text(
        output / "commands.txt",
        " ".join(
            shlex.quote(str(v))
            for v in [
                sys.executable,
                *sys.argv,
            ]
        )
        + "\n",
    )
    atomic_text(
        output / "package_versions.txt",
        package_versions(),
    )
    atomic_json(
        output / "git_info.json",
        git_information(),
    )


    features = list(
        NO_RACE_FEATURES
    )
    if "mother_race" in features:
        raise AssertionError(
            "mother_race leaked into no-race feature list."
        )
    if len(features) != 8:
        raise AssertionError(
            "No-race feature count is not 8."
        )

    # No categorical feature remains after removing mother_race.
    categorical_features: list[str] = []
    parameters = dict(
        config["model"]["parameters"]
    )

    protocol_rows: list[
        dict[str, Any]
    ] = []
    threshold_rows: list[
        dict[str, Any]
    ] = []
    four_cell_rows: list[
        dict[str, Any]
    ] = []
    shapley_rows: list[
        dict[str, Any]
    ] = []
    diagnostics_rows: list[
        dict[str, Any]
    ] = []
    model_lineage: list[
        dict[str, Any]
    ] = []
    split_crosscheck_rows: list[
        dict[str, Any]
    ] = []

    model_fit_count = 0
    temporal_prediction_count = 0

    for target in TARGETS:
        print(
            f"\n===== NO-RACE: {target} ====="
        )

        known_2022 = frame_2022[
            target
        ].notna().to_numpy(dtype=bool)
        known_2023 = frame_2023[
            target
        ].notna().to_numpy(dtype=bool)

        train_mask = (
            primary_2022
            & known_2022
            & train_mask_all_2022
        )
        calibration_mask = (
            primary_2022
            & known_2022
            & calibration_mask_all_2022
        )
        test_mask = (
            primary_2023
            & known_2023
        )

        y_train = frame_2022.loc[
            train_mask, target
        ].to_numpy(dtype=np.int8)
        y_cal = frame_2022.loc[
            calibration_mask, target
        ].to_numpy(dtype=np.int8)
        y_test = frame_2023.loc[
            test_mask, target
        ].to_numpy(dtype=np.int8)

        for split_name, y_values in (
            ("train_2022", y_train),
            ("calibration_2022", y_cal),
            (
                "temporal_evaluation_cohort_2023",
                y_test,
            ),
        ):
            if len(
                np.unique(y_values)
            ) != 2:
                raise ValueError(
                    f"{target}/{split_name} lacks both classes."
                )

        split_crosscheck_rows.append(
            validate_split_counts_against_canonical(
                canonical_diagnostics=lineage[
                    "canonical_diagnostics"
                ],
                target=target,
                train_n=len(y_train),
                calibration_n=len(y_cal),
                evaluation_n=len(y_test),
            )
        )

        preprocessor = build_preprocessor(
            features,
            categorical_features,
            config["preprocessing"],
        )

        x_train = consistent_matrix(
            preprocessor.fit_transform(
                frame_2022.loc[
                    train_mask,
                    features,
                ]
            )
        )
        transformed_feature_n = int(
            x_train.shape[1]
        )
        if transformed_feature_n != 8:
            raise AssertionError(
                f"{target}: no-race transformed feature count "
                f"{transformed_feature_n} != 8"
            )

        model = lightgbm.LGBMClassifier(
            **parameters
        )
        fit_started = time.time()
        model.fit(
            x_train, y_train
        )
        fit_seconds = (
            time.time() - fit_started
        )
        train_prob = np.asarray(
            model.predict_proba(
                x_train
            )[:, 1],
            dtype=np.float64,
        )
        del x_train
        gc.collect()

        model_fit_count += 1

        x_cal = consistent_matrix(
            preprocessor.transform(
                frame_2022.loc[
                    calibration_mask,
                    features,
                ]
            )
        )
        cal_prob = np.asarray(
            model.predict_proba(
                x_cal
            )[:, 1],
            dtype=np.float64,
        )
        del x_cal
        gc.collect()

        threshold = threshold_for_capacity(
            cal_prob, Q
        )
        calibration_nominal_selected = max(
            1,
            int(round(Q * len(y_cal))),
        )
        calibration_realized_selected = int(
            np.sum(
                cal_prob >= threshold
            )
        )

        threshold_rows.append(
            {
                "target": target,
                "feature_set": "landmark_no_race_8",
                "model": "lightgbm",
                "source_year": 2022,
                "source_population": (
                    "calibration target-specific all-record"
                ),
                "nominal_fraction": Q,
                "calibration_n": int(
                    len(y_cal)
                ),
                "nominal_selected_n": calibration_nominal_selected,
                "realized_selected_n_with_ties": calibration_realized_selected,
                "threshold": float(
                    threshold
                ),
                "comparator": ">=",
            }
        )

        x_test = consistent_matrix(
            preprocessor.transform(
                frame_2023.loc[
                    test_mask,
                    features,
                ]
            )
        )
        prediction_started = time.time()
        test_prob = np.asarray(
            model.predict_proba(
                x_test
            )[:, 1],
            dtype=np.float64,
        )
        prediction_seconds = (
            time.time()
            - prediction_started
        )
        del x_test
        gc.collect()

        temporal_prediction_count += 1

        care_test = frame_2023.loc[
            test_mask,
            "prenatal_care_month",
        ].to_numpy(dtype=np.float64)

        diagnostics_rows.extend(
            [
                diagnostics_row(
                    target=target,
                    split="train_2022",
                    y=y_train,
                    probabilities=train_prob,
                    transformed_feature_n=transformed_feature_n,
                ),
                diagnostics_row(
                    target=target,
                    split="calibration_2022",
                    y=y_cal,
                    probabilities=cal_prob,
                    transformed_feature_n=transformed_feature_n,
                ),
                diagnostics_row(
                    target=target,
                    split="temporal_evaluation_cohort_2023",
                    y=y_test,
                    probabilities=test_prob,
                    transformed_feature_n=transformed_feature_n,
                ),
            ]
        )

        target_protocol = evaluate_protocols(
            target=target,
            y=y_test,
            probabilities=test_prob,
            care_month=care_test,
            threshold_2022=threshold,
        )
        protocol_rows.extend(
            target_protocol
        )

        target_cells, target_shapley = (
            four_cell_shapley(
                target=target,
                y=y_test,
                probabilities=test_prob,
                care_month=care_test,
            )
        )
        four_cell_rows.extend(
            target_cells
        )
        shapley_rows.append(
            target_shapley
        )

        model_path = (
            models_dir
            / f"{target}__landmark_no_race_8__lightgbm.joblib"
        )
        joblib.dump(
            {
                "target": target,
                "feature_set": "landmark_no_race_8",
                "paper_feature_set": (
                    "conservative landmark feature set without maternal race"
                ),
                "features": features,
                "removed_feature": "mother_race",
                "preprocessor": preprocessor,
                "model": model,
                "parameters": parameters,
                "split_seed": int(
                    config["split"][
                        "seed"
                    ]
                ),
                "calibration_fraction": float(
                    config["split"][
                        "calibration_fraction"
                    ]
                ),
            },
            model_path,
            compress=3,
        )

        model_lineage.append(
            {
                "target": target,
                "path": str(
                    model_path
                ),
                "sha256": sha256_file(
                    model_path
                ),
                "raw_feature_n": 8,
                "transformed_feature_n": transformed_feature_n,
                "fit_seconds": fit_seconds,
                "prediction_seconds_2023": prediction_seconds,
                "train_n": int(
                    len(y_train)
                ),
                "calibration_n": int(
                    len(y_cal)
                ),
                "evaluation_n": int(
                    len(y_test)
                ),
            }
        )

        print(
            f"{target}: fit={fit_seconds:.1f}s, "
            f"2023 prediction={prediction_seconds:.1f}s, "
            f"threshold={threshold:.6f}"
        )

        del (
            preprocessor,
            model,
            train_prob,
            cal_prob,
            test_prob,
            y_train,
            y_cal,
            y_test,
            care_test,
        )
        gc.collect()

    if model_fit_count != 3:
        raise AssertionError(
            f"Expected 3 no-race model fits; got {model_fit_count}"
        )
    if temporal_prediction_count != 3:
        raise AssertionError(
            "Expected one temporal prediction pass per target."
        )

    # Candidate-set and event-count invariance relative to canonical strict9.
    population_crosscheck = (
        validate_population_against_canonical(
            protocol_rows,
            lineage["canonical_protocol"],
        )
    )

    comparisons = (
        make_comparison_to_strict9(
            no_race_protocol_rows=protocol_rows,
            no_race_shapley_rows=shapley_rows,
            canonical_protocol=lineage[
                "canonical_protocol"
            ],
            strict9_shapley=lineage[
                "shapley_main"
            ],
        )
    )

    # Final row-count / identity checks.
    invariants = {
        "dataset_2022_hash_match": (
            observed_hashes[2022]
            == EXPECTED_SHA256[2022]
        ),
        "dataset_2023_hash_match": (
            observed_hashes[2023]
            == EXPECTED_SHA256[2023]
        ),
        "canonical_config_validated": True,
        "exactly_mother_race_removed": (
            set(STRICT9_FEATURES)
            - set(NO_RACE_FEATURES)
            == {"mother_race"}
        ),
        "eight_raw_features": (
            len(NO_RACE_FEATURES) == 8
        ),
        "mother_race_absent": (
            "mother_race"
            not in NO_RACE_FEATURES
        ),
        "same_split_assignment_rule": True,
        "split_assigned_before_target_filtering": True,
        "canonical_split_counts_reproduced": (
            len(split_crosscheck_rows) == 3
            and all(
                row["status"] in {"PASS", "NOT_AVAILABLE"}
                for row in split_crosscheck_rows
            )
            and (
                lineage["canonical_diagnostics"] is None
                or all(
                    row["status"] == "PASS"
                    for row in split_crosscheck_rows
                )
            )
        ),
        "same_lightgbm_parameters": True,
        "same_target_populations_as_canonical": (
            len(population_crosscheck)
            == 9
            and all(
                row["status"] == "PASS"
                for row in population_crosscheck
            )
        ),
        "three_model_fits": (
            model_fit_count == 3
        ),
        "three_temporal_prediction_vectors": (
            temporal_prediction_count == 3
        ),
        "q_exactly_10pct": (
            Q == 0.10
        ),
        "protocol_row_count": (
            len(protocol_rows)
            == 3 * 3 * 3
        ),
        "four_cell_row_count": (
            len(four_cell_rows)
            == 3 * 4
        ),
        "shapley_row_count": (
            len(shapley_rows) == 3
        ),
        "shapley_identities": all(
            abs(
                float(
                    row[
                        "identity_shapley_error"
                    ]
                )
            )
            <= 1e-12
            and abs(
                float(
                    row[
                        "identity_factorial_error"
                    ]
                )
            )
            <= 1e-12
            for row in shapley_rows
        ),
        "no_bootstrap_in_sensitivity": True,
    }

    if not all(
        invariants.values()
    ):
        failed = [
            key
            for key, value
            in invariants.items()
            if not value
        ]
        raise AssertionError(
            f"No-race invariants failed: {failed}"
        )

    atomic_csv(
        output
        / "no_race_protocol_results_10pct.csv",
        protocol_rows,
    )
    atomic_csv(
        output
        / "no_race_four_cell_10pct.csv",
        four_cell_rows,
    )
    atomic_csv(
        output
        / "no_race_shapley_10pct.csv",
        shapley_rows,
    )
    atomic_csv(
        output
        / "no_race_thresholds_2022_10pct.csv",
        threshold_rows,
    )
    atomic_csv(
        output
        / "no_race_model_diagnostics.csv",
        diagnostics_rows,
    )
    atomic_csv(
        output
        / "no_race_population_crosscheck.csv",
        population_crosscheck,
    )
    atomic_csv(
        output
        / "no_race_split_crosscheck.csv",
        split_crosscheck_rows,
    )
    atomic_csv(
        output
        / "no_race_comparison_to_strict9.csv",
        comparisons,
    )
    atomic_text(
        output / "no_race_summary.md",
        summary_markdown(
            comparisons,
            diagnostics_rows,
        ),
    )

    manifest = {
        "status": "PASS",
        "script_version": SCRIPT_VERSION,
        "clearml_task_id": task.id,
        "clearml_dataset_id": args.dataset_id,
        "clearml_queue": args.queue,
        "source_lineage": {
            "canonical_task_id": args.canonical_task_id,
            "strict9_shapley_task_id": args.shapley_task_id,
            "canonical_manifest_status": lineage[
                "canonical_manifest"
            ].get("status"),
            "strict9_shapley_manifest_status": lineage[
                "shapley_manifest"
            ].get("status"),
        },
        "design": {
            "sensitivity": "remove_mother_race",
            "q": Q,
            "targets": list(
                TARGETS
            ),
            "strict9_features": list(
                STRICT9_FEATURES
            ),
            "no_race_features": list(
                NO_RACE_FEATURES
            ),
            "removed_feature": "mother_race",
            "model": "lightgbm",
            "model_fit_count": model_fit_count,
            "temporal_prediction_vector_count": temporal_prediction_count,
            "cdc2022_role": (
                "development and calibration"
            ),
            "cdc2023_role": (
                "temporal evaluation cohort"
            ),
            "split_seed": int(
                config["split"]["seed"]
            ),
            "calibration_fraction": float(
                config["split"][
                    "calibration_fraction"
                ]
            ),
            "split_assigned_before_target_filtering": True,
            "protocols": [
                "cohort_specific_top_fraction",
                "fixed_absolute_budget",
                "transported_2022_score_threshold",
            ],
            "four_cell_shapley": True,
            "bootstrap": False,
        },
        "input_sha256": {
            "cdc2022": observed_hashes[2022],
            "cdc2023": observed_hashes[2023],
            "canonical_locked_config": sha256_file(
                Path(
                    lineage[
                        "canonical_config_path"
                    ]
                )
            ),
            "canonical_protocol_results": sha256_file(
                Path(
                    lineage[
                        "canonical_protocol_path"
                    ]
                )
            ),
            "strict9_shapley_main_results": sha256_file(
                Path(
                    lineage[
                        "shapley_main_path"
                    ]
                )
            ),
            "script": sha256_file(
                SCRIPT_PATH
            ),
        },
        "model_lineage": model_lineage,
        "invariants": invariants,
        "outputs": {
            "protocol_rows": len(
                protocol_rows
            ),
            "four_cell_rows": len(
                four_cell_rows
            ),
            "shapley_rows": len(
                shapley_rows
            ),
            "threshold_rows": len(
                threshold_rows
            ),
            "diagnostics_rows": len(
                diagnostics_rows
            ),
            "population_crosscheck_rows": len(
                population_crosscheck
            ),
            "split_crosscheck_rows": len(
                split_crosscheck_rows
            ),
            "comparison_rows": len(
                comparisons
            ),
        },
        "elapsed_seconds": float(
            time.time() - started
        ),
        "git": git_information(),
    }

    atomic_json(
        output / "run_manifest.json",
        manifest,
    )

    output_hashes: dict[
        str, str
    ] = {}
    for path in sorted(
        output.rglob("*")
    ):
        if (
            path.is_file()
            and path.name
            != "run_manifest.json"
        ):
            output_hashes[
                str(
                    path.relative_to(
                        output
                    )
                )
            ] = sha256_file(path)
    manifest[
        "output_sha256"
    ] = output_hashes
    atomic_json(
        output / "run_manifest.json",
        manifest,
    )

    upload_outputs(
        task, output
    )
    task.close()

    print("TAE NO-RACE SENSITIVITY PASS")
    print(f"Output: {output}")
    print(
        "3 no-race LightGBM fits: PASS"
    )
    print(
        "Canonical population cross-check: PASS"
    )
    print(
        "Four-cell/Shapley identities: PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
