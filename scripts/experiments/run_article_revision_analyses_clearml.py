#!/usr/bin/env python3
"""Standalone ClearML analysis bundle for the ML4H article revision.

The script performs four locked analyses with LightGBM and booking_strict:

1. Paired full-size bootstrap for the corrected fixed-model equal-budget result.
2. Mechanism table by prenatal-care entry group at a 10% absolute budget.
3. Probability calibration of the primary singleton LightGBM models.
4. Corrected all-birth sensitivity using the same fixed-model equal-budget design.

Scientific design
-----------------
Primary population:
    U.S.-resident singleton births.

Sensitivity population:
    U.S.-resident births of all pluralities.

Targets:
    target_preterm, target_nicu, target_lbw.

Features:
    booking_strict only.

CDC 2022:
    80% model training;
    10% Platt calibrator fitting;
    10% independent held-out evaluation.

CDC 2023:
    temporal test only.

Early eligibility:
    prenatal_care_month in months 1 through 3.

Fixed-budget comparison:
    the same trained model and the same absolute referral count are used for
    early-entry and full-cohort policies. No cohort-specific refitting occurs.

Paired bootstrap:
    exact record-level, full-size nonparametric bootstrap. The same bootstrap
    multiplicities are used for early and full policies. In each replicate,
    top-B is reconstructed separately for both policies.

This file is intentionally self-contained:
- no Git clone is required by the remote ClearML Agent;
- no neighbouring project modules are imported;
- the locked configuration and booking_strict features are embedded;
- CDC 2022 and CDC 2023 are read from one ClearML Dataset object;
- Task.force_store_standalone_script(True) is called before Task.init().
"""
from __future__ import annotations

import argparse
import copy
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
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

DEFAULT_PROJECT_NAME = "pershin-medailab/Vache_Oganisyan/CDC Natality Audit"
DEFAULT_DATASET_ID = "062ba26c0ca24cef99549c2a2ab34e65"
DEFAULT_QUEUE = "78daf150d5ba4a8c95c24350ed757a98"
DEFAULT_TASK_NAME = (
    "CDC article revision: bootstrap mechanism calibration all-birth sensitivity"
)

TARGETS = ("target_preterm", "target_nicu", "target_lbw")
FEATURE_SET_NAMES = ("booking_strict",)

EMBEDDED_FEATURE_SETS: dict[str, list[str]] = {
    "booking_strict": [
        "diab_pre",
        "hyper_pre",
        "marital_status",
        "mother_age",
        "mother_bmi",
        "mother_educ",
        "mother_height_inches",
        "mother_race",
        "mother_weight_pre",
        "prior_dead_births",
        "prior_live_births",
        "prior_terminations",
    ],
}

DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "cdc2022": (
            "data/cdc_temporal_harmonized/"
            "cdc_natality_2022_harmonized.csv"
        ),
        "cdc2023": (
            "data/cdc_temporal_harmonized/"
            "cdc_natality_2023_harmonized.csv"
        ),
        "feature_sets": "data/cdc_factorial_design/feature_sets.json",
        "expected_rows": {"2022": 3676029, "2023": 3605081},
        "verify_input_hashes": True,
    },
    "output": {
        "directory": "results/article_revision_fixed_model",
        "overwrite": False,
        "save_models": True,
        "upload_models_to_clearml": True,
        "save_predictions": False,
        "upload_predictions_to_clearml": False,
    },
    "split": {
        "seed": 2026,
        "calibration_fraction": 0.20,
    },
    "cohorts": {
        "primary_population": "is_us_resident == 1 and is_singleton == 1",
        "all_births_sensitivity_population": "is_us_resident == 1",
        "early_entry_min_month": 1,
        "early_entry_max_month": 3,
    },
    "targets": list(TARGETS),
    "feature_sets": list(FEATURE_SET_NAMES),
    "model": {
        "name": "lightgbm",
        "parameters": {
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
        },
    },
    "preprocessing": {
        "categorical_features": [
            "mother_race",
            "marital_status",
            "mother_educ",
        ],
        "numeric_imputation": "median",
        "categorical_imputation": "most_frequent",
        "standardize_numeric": True,
    },
    "fixed_absolute_budget": {
        "fractions_of_full_population": [0.05, 0.10],
        "scenarios": ["early_entry", "full_cohort"],
    },
    "bootstrap": {
        "enabled": True,
        "replicates": 500,
        "seed": 2027,
        "confidence_level": 0.95,
        "evaluations": [
            "calibration_evaluation_2022",
            "temporal_test_2023",
        ],
        "full_size": True,
        "paired": True,
        "progress_every": 25,
    },
    "mechanism": {
        "evaluation": "temporal_test_2023",
        "budget_fraction_of_full_population": 0.10,
        "groups": [
            "months_1_3",
            "after_month_3",
            "no_prenatal_care",
            "unknown",
        ],
    },
    "probability_calibration": {
        "method": "platt_sigmoid",
        "fit_fraction_within_calibration_pool": 0.50,
        "partition_seed": 2028,
        "clip_eps": 1.0e-6,
        "curve_bins": 20,
        "optimizer_max_iter": 300,
    },
    "execution": {
        "expected_singleton_model_count": 3,
        "expected_all_birth_model_count": 3,
        "expected_total_model_count": 6,
    },
}

TAGS = [
    "cdc-natality",
    "temporal-test",
    "fixed-model",
    "equal-absolute-budget",
    "paired-bootstrap",
    "full-size-bootstrap",
    "mechanism-table",
    "probability-calibration",
    "all-birth-sensitivity",
    "lightgbm",
    "booking-strict",
    "standalone-script",
]

REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("clearml", ">=1.16,<3"),
    ("joblib", ">=1.3,<2"),
    ("lightgbm", ">=4.3,<5"),
    ("numpy", ">=1.26,<3"),
    ("pandas", ">=2.2,<4"),
    ("PyYAML", ">=6,<7"),
    ("scikit-learn", ">=1.4,<2"),
    ("scipy", ">=1.11,<2"),
)

# Imported top-level packages are captured by ClearML. LightGBM is imported
# after remote hand-off, therefore it is the only explicit extra requirement.
CLEARML_EXTRA_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("lightgbm", ">=4.3,<5"),
)


# ---------------------------------------------------------------------------
# Generic, ClearML Dataset, preprocessing, and model helpers
# ---------------------------------------------------------------------------

def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
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
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> str:
    lines = [
        f"python=={platform.python_version()}",
        f"platform={platform.platform()}",
    ]
    for package, _ in REQUIREMENTS:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT_INSTALLED"
        lines.append(f"{package}=={version}")
    return "\n".join(lines) + "\n"


def git_information() -> dict[str, Any]:
    """Best-effort only. Standalone remote execution does not require Git."""

    def run_git(*arguments: str) -> Optional[str]:
        try:
            return subprocess.run(
                ["git", *arguments],
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


def load_local_config(path: Optional[Path]) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config

    candidate = path.expanduser()
    if not candidate.is_file():
        # On a remote standalone task the local config path normally does not
        # exist. The connected ClearML configuration will override defaults.
        print(f"Config file not present in this process, using connected/default config: {candidate}")
        return config

    payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("YAML config root must be a mapping.")
    return deep_update(config, payload)


def to_plain_python(value: Any) -> Any:
    """Convert ClearML proxy/config objects to YAML/JSON-safe built-ins."""
    if isinstance(value, Mapping):
        return {
            str(key): to_plain_python(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_plain_python(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_plain_python(to_dict())

    return value


def yaml_dump_safe(value: Any) -> str:
    """Serialize plain Python containers, never ClearML proxy mappings."""
    return yaml.safe_dump(
        to_plain_python(value),
        sort_keys=False,
        allow_unicode=True,
    )


def prepare_output(config: dict[str, Any], overwrite_cli: bool) -> Path:
    output = Path(str(config["output"]["directory"])).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output

    overwrite = overwrite_cli or bool(config["output"].get("overwrite", False))
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output} is not empty. Use --overwrite.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


# ---------------------------------------------------------------------------
# ClearML Dataset and CDC files
# ---------------------------------------------------------------------------

def bind_clearml_dataset(dataset_id: str, max_workers: int) -> Path:
    from clearml import Dataset

    dataset = Dataset.get(dataset_id=dataset_id)
    local_copy = dataset.get_local_copy(
        max_workers=max_workers,
        raise_on_error=True,
    )
    root = Path(local_copy).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ClearML Dataset local copy is not a directory: {root}")
    return root


def resolve_dataset_file(dataset_root: Path, configured: str) -> Path:
    configured_path = Path(configured)
    candidates = [
        dataset_root / configured_path,
        dataset_root / configured_path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = list(dataset_root.rglob(configured_path.name))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        exact_suffix = [
            path for path in matches
            if str(path).replace("\\", "/").endswith(str(configured_path).replace("\\", "/"))
        ]
        if len(exact_suffix) == 1:
            return exact_suffix[0].resolve()
        raise RuntimeError(
            f"Ambiguous dataset file {configured!r}; matches: {[str(path) for path in matches[:10]]}"
        )
    raise FileNotFoundError(
        f"Dataset file {configured!r} was not found under {dataset_root}"
    )


def load_feature_sets(dataset_root: Path, config: dict[str, Any]) -> dict[str, list[str]]:
    configured = str(config["data"].get("feature_sets", ""))
    if configured:
        try:
            path = resolve_dataset_file(dataset_root, configured)
            payload = json.loads(path.read_text(encoding="utf-8"))
            result: dict[str, list[str]] = {}
            for name in FEATURE_SET_NAMES:
                value = payload[name]
                if isinstance(value, dict):
                    features = value.get("features")
                else:
                    features = value
                if not isinstance(features, list) or not features:
                    raise ValueError(f"Invalid feature set {name!r} in {path}")
                result[name] = [str(feature) for feature in features]
            print(f"Loaded feature sets from Dataset: {path}")
            return result
        except FileNotFoundError:
            pass

    print("Feature-set JSON was not found; using embedded locked feature sets.")
    return copy.deepcopy(EMBEDDED_FEATURE_SETS)


def read_cdc_frame(path: Path, columns: list[str], expected_rows: Optional[int], label: str) -> pd.DataFrame:
    print(f"Reading {label}: {path}")
    frame = pd.read_csv(
        path,
        usecols=columns,
        low_memory=False,
    )
    if expected_rows is not None and len(frame) != int(expected_rows):
        raise ValueError(f"{label} row count {len(frame)} != expected {expected_rows}")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    print(f"{label}: {len(frame):,} rows, {len(frame.columns)} columns")
    return frame


def validate_binary_column(frame: pd.DataFrame, column: str, label: str, allow_missing: bool) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    if not allow_missing and values.isna().any():
        raise ValueError(f"{label}.{column} contains missing/non-numeric values.")
    unique = set(values.dropna().unique().tolist())
    if not unique.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"{label}.{column} is not binary: {sorted(unique)[:20]}")
    frame[column] = values



# ---------------------------------------------------------------------------
# Preprocessing and model
# ---------------------------------------------------------------------------

def build_preprocessor(
    features: list[str],
    categorical_features: list[str],
    config: dict[str, Any],
) -> ColumnTransformer:
    categorical = [feature for feature in features if feature in categorical_features]
    numeric = [feature for feature in features if feature not in categorical]

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=str(config["numeric_imputation"]))),
    ]
    if bool(config.get("standardize_numeric", True)):
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy=str(config["categorical_imputation"])),
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

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        transformers.append(("categorical", categorical_pipeline, categorical))

    if not transformers:
        raise ValueError("No features were supplied to the preprocessor.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )


def consistent_matrix(matrix: Any) -> Any:
    if isinstance(matrix, pd.DataFrame):
        return matrix.to_numpy(dtype=np.float32, copy=False)
    if hasattr(matrix, "astype"):
        return matrix.astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def lightgbm_parameters(config: dict[str, Any]) -> dict[str, Any]:
    parameters = dict(config["model"]["parameters"])
    for key in ("device", "gpu_platform_id", "gpu_device_id", "num_gpu"):
        parameters.pop(key, None)
    parameters["device_type"] = "cpu"
    parameters["class_weight"] = None
    parameters["scale_pos_weight"] = 1.0
    return parameters

# ---------------------------------------------------------------------------
# Locked configuration checks
# ---------------------------------------------------------------------------

def validate_config(config: dict[str, Any]) -> None:
    if tuple(map(str, config.get("targets", []))) != TARGETS:
        raise ValueError(f"targets must be exactly {list(TARGETS)}")
    if tuple(map(str, config.get("feature_sets", []))) != FEATURE_SET_NAMES:
        raise ValueError(
            f"feature_sets must be exactly {list(FEATURE_SET_NAMES)}"
        )

    primary = str(
        config["cohorts"]["primary_population"]
    ).replace(" ", "")
    if primary != "is_us_resident==1andis_singleton==1":
        raise ValueError(
            "Primary population must be U.S.-resident singleton births."
        )

    sensitivity = str(
        config["cohorts"]["all_births_sensitivity_population"]
    ).replace(" ", "")
    if sensitivity != "is_us_resident==1":
        raise ValueError(
            "All-birth sensitivity must include U.S.-resident births."
        )

    if (
        int(config["cohorts"]["early_entry_min_month"]) != 1
        or int(config["cohorts"]["early_entry_max_month"]) != 3
    ):
        raise ValueError("Early entry must be prenatal-care months 1 through 3.")

    calibration_fraction = float(config["split"]["calibration_fraction"])
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("split.calibration_fraction must be in (0, 1).")

    budgets = [
        float(value)
        for value in config["fixed_absolute_budget"][
            "fractions_of_full_population"
        ]
    ]
    if budgets != [0.05, 0.10]:
        raise ValueError("Fixed absolute budgets must be [0.05, 0.10].")

    bootstrap = config["bootstrap"]
    if not bool(bootstrap.get("paired", False)):
        raise ValueError("Bootstrap must be paired.")
    if not bool(bootstrap.get("full_size", False)):
        raise ValueError("Bootstrap must use the full evaluation size.")
    if int(bootstrap["replicates"]) < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")
    if not 0.0 < float(bootstrap["confidence_level"]) < 1.0:
        raise ValueError("Invalid bootstrap confidence level.")
    allowed_evaluations = {
        "calibration_evaluation_2022",
        "temporal_test_2023",
    }
    evaluations = set(map(str, bootstrap["evaluations"]))
    if not evaluations or not evaluations.issubset(allowed_evaluations):
        raise ValueError(
            "bootstrap.evaluations contains an unsupported evaluation."
        )

    mechanism = config["mechanism"]
    if float(
        mechanism["budget_fraction_of_full_population"]
    ) != 0.10:
        raise ValueError("Mechanism table is locked to the 10% budget.")

    calibration = config["probability_calibration"]
    if str(calibration["method"]).lower() not in {
        "platt",
        "sigmoid",
        "platt_sigmoid",
    }:
        raise ValueError("Only Platt/sigmoid calibration is supported.")
    fit_fraction = float(
        calibration["fit_fraction_within_calibration_pool"]
    )
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError(
            "fit_fraction_within_calibration_pool must be in (0, 1)."
        )
    if int(calibration["curve_bins"]) < 5:
        raise ValueError("curve_bins must be at least 5.")
    if not 0.0 < float(calibration["clip_eps"]) < 0.01:
        raise ValueError("clip_eps must be in (0, 0.01).")

    model = config["model"]
    if str(model.get("name", "")) != "lightgbm":
        raise ValueError("This experiment allows LightGBM only.")
    parameters = model["parameters"]
    if parameters.get("class_weight") is not None:
        raise ValueError("LightGBM class_weight must remain null.")
    if float(parameters.get("scale_pos_weight", 1.0)) != 1.0:
        raise ValueError("LightGBM scale_pos_weight must remain 1.0.")

    execution = config["execution"]
    if int(execution["expected_singleton_model_count"]) != 3:
        raise ValueError("expected_singleton_model_count must be 3.")
    if int(execution["expected_all_birth_model_count"]) != 3:
        raise ValueError("expected_all_birth_model_count must be 3.")
    if int(execution["expected_total_model_count"]) != 6:
        raise ValueError("expected_total_model_count must be 6.")


# ---------------------------------------------------------------------------
# Dataset columns and population definitions
# ---------------------------------------------------------------------------

def required_columns(feature_sets: dict[str, list[str]]) -> list[str]:
    columns = {
        "is_us_resident",
        "is_singleton",
        "prenatal_care_month",
        *TARGETS,
    }
    for features in feature_sets.values():
        columns.update(features)
    return sorted(columns)


def resident_singleton_mask(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["is_us_resident"].eq(1)
        & frame["is_singleton"].eq(1)
    ).to_numpy(dtype=bool)


def resident_all_births_mask(frame: pd.DataFrame) -> np.ndarray:
    return frame["is_us_resident"].eq(1).to_numpy(dtype=bool)


def early_entry_mask(
    care_month: np.ndarray,
    minimum_month: int = 1,
    maximum_month: int = 3,
) -> np.ndarray:
    values = np.asarray(care_month, dtype=np.float64)
    return (
        np.isfinite(values)
        & (values >= minimum_month)
        & (values <= maximum_month)
    )


def care_entry_group_masks(
    care_month: np.ndarray,
) -> dict[str, np.ndarray]:
    values = np.asarray(care_month, dtype=np.float64)
    groups = {
        "months_1_3": (
            np.isfinite(values)
            & (values >= 1)
            & (values <= 3)
        ),
        "after_month_3": (
            np.isfinite(values)
            & (values > 3)
        ),
        "no_prenatal_care": (
            np.isfinite(values)
            & (values == 0)
        ),
        "unknown": ~np.isfinite(values),
    }
    coverage = np.zeros(len(values), dtype=np.int8)
    for mask in groups.values():
        coverage += mask.astype(np.int8)
    if not np.all(coverage == 1):
        bad = int(np.sum(coverage != 1))
        raise ValueError(
            f"Care-entry groups do not partition the population; bad={bad}"
        )
    return groups


# ---------------------------------------------------------------------------
# Fixed-model equal-absolute-budget analysis
# ---------------------------------------------------------------------------

def rank_eligible_indices(
    probabilities: np.ndarray,
    eligible_mask: np.ndarray,
) -> np.ndarray:
    eligible_indices = np.flatnonzero(eligible_mask)
    eligible_probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )[eligible_indices]
    order = np.lexsort(
        (
            eligible_indices,
            -eligible_probabilities,
        )
    )
    return eligible_indices[order]


def access_metrics(
    *,
    y: np.ndarray,
    eligible_mask: np.ndarray,
    selected_indices: np.ndarray,
) -> dict[str, int | float]:
    y_values = np.asarray(y, dtype=np.int8)
    eligible_indices = np.flatnonzero(eligible_mask)
    full_population_n = int(len(y_values))
    full_population_event_n = int(np.sum(y_values == 1))
    eligible_n = int(len(eligible_indices))
    eligible_event_n = int(np.sum(y_values[eligible_indices] == 1))
    selected_n = int(len(selected_indices))
    true_positive_n = int(np.sum(y_values[selected_indices] == 1))

    return {
        "full_population_n": full_population_n,
        "full_population_event_n": full_population_event_n,
        "eligible_n": eligible_n,
        "eligible_event_n": eligible_event_n,
        "selected_n": selected_n,
        "true_positive_n": true_positive_n,
        "selection_rate_among_eligible": (
            float(selected_n / eligible_n)
            if eligible_n
            else math.nan
        ),
        "precision": (
            float(true_positive_n / selected_n)
            if selected_n
            else math.nan
        ),
        "recall_among_eligible": (
            float(true_positive_n / eligible_event_n)
            if eligible_event_n
            else math.nan
        ),
        "population_event_capture": (
            float(true_positive_n / full_population_event_n)
            if full_population_event_n
            else math.nan
        ),
        "maximum_event_ceiling": (
            float(eligible_event_n / full_population_event_n)
            if full_population_event_n
            else math.nan
        ),
    }


def fixed_budget_rows(
    *,
    population: str,
    evaluation: str,
    target: str,
    y: np.ndarray,
    probabilities: np.ndarray,
    early_mask: np.ndarray,
    budget_fractions: list[float],
) -> tuple[list[dict[str, Any]], dict[tuple[float, str], np.ndarray]]:
    y_values = np.asarray(y, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    full_mask = np.ones(len(y_values), dtype=bool)
    rankings = {
        "early_entry": rank_eligible_indices(
            probabilities,
            early_mask,
        ),
        "full_cohort": rank_eligible_indices(
            probabilities,
            full_mask,
        ),
    }

    rows: list[dict[str, Any]] = []
    selections: dict[tuple[float, str], np.ndarray] = {}
    for budget_fraction in budget_fractions:
        budget_n = max(
            1,
            int(round(float(budget_fraction) * len(y_values))),
        )
        for scenario, eligible_mask in (
            ("early_entry", early_mask),
            ("full_cohort", full_mask),
        ):
            eligible_n = int(np.sum(eligible_mask))
            if budget_n > eligible_n:
                raise ValueError(
                    f"{population}/{target}/{evaluation}/{scenario}: "
                    f"budget={budget_n:,} exceeds eligible={eligible_n:,}"
                )
            selected = rankings[scenario][:budget_n]
            selections[(float(budget_fraction), scenario)] = selected
            rows.append(
                {
                    "population": population,
                    "target": target,
                    "feature_set": "booking_strict",
                    "model": "lightgbm",
                    "evaluation": evaluation,
                    "policy": "fixed_absolute_budget",
                    "scenario": scenario,
                    "budget_fraction_of_full_population": float(
                        budget_fraction
                    ),
                    "budget_n": budget_n,
                    **access_metrics(
                        y=y_values,
                        eligible_mask=eligible_mask,
                        selected_indices=selected,
                    ),
                }
            )
    return rows, selections


def compare_early_full(
    result_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, float],
        dict[str, dict[str, Any]],
    ] = {}
    for row in result_rows:
        key = (
            str(row["population"]),
            str(row["evaluation"]),
            str(row["target"]),
            float(row["budget_fraction_of_full_population"]),
        )
        grouped.setdefault(key, {})[str(row["scenario"])] = row

    comparisons: list[dict[str, Any]] = []
    for key, scenarios in sorted(grouped.items()):
        if set(scenarios) != {"early_entry", "full_cohort"}:
            raise ValueError(f"Missing scenario for comparison: {key}")
        early = scenarios["early_entry"]
        full = scenarios["full_cohort"]
        if int(early["budget_n"]) != int(full["budget_n"]):
            raise ValueError("Equal-budget comparison has unequal budgets.")

        comparisons.append(
            {
                "population": key[0],
                "evaluation": key[1],
                "target": key[2],
                "feature_set": "booking_strict",
                "model": "lightgbm",
                "budget_fraction_of_full_population": key[3],
                "budget_n": int(full["budget_n"]),
                "early_selected_n": int(early["selected_n"]),
                "full_selected_n": int(full["selected_n"]),
                "early_true_positive_n": int(early["true_positive_n"]),
                "full_true_positive_n": int(full["true_positive_n"]),
                "additional_true_positives_full_minus_early": int(
                    full["true_positive_n"] - early["true_positive_n"]
                ),
                "early_precision": float(early["precision"]),
                "full_precision": float(full["precision"]),
                "precision_difference_full_minus_early": float(
                    full["precision"] - early["precision"]
                ),
                "early_recall_among_eligible": float(
                    early["recall_among_eligible"]
                ),
                "full_recall_among_eligible": float(
                    full["recall_among_eligible"]
                ),
                "early_population_event_capture": float(
                    early["population_event_capture"]
                ),
                "full_population_event_capture": float(
                    full["population_event_capture"]
                ),
                "population_event_capture_difference_full_minus_early": float(
                    full["population_event_capture"]
                    - early["population_event_capture"]
                ),
                "early_maximum_event_ceiling": float(
                    early["maximum_event_ceiling"]
                ),
                "full_maximum_event_ceiling": float(
                    full["maximum_event_ceiling"]
                ),
            }
        )
    return comparisons


# ---------------------------------------------------------------------------
# Exact paired full-size bootstrap
# ---------------------------------------------------------------------------

def bootstrap_selected_true_positives(
    *,
    ordered_counts: np.ndarray,
    ordered_y: np.ndarray,
    budgets: list[int],
) -> dict[int, int]:
    cumulative_count = np.cumsum(
        ordered_counts,
        dtype=np.int64,
    )
    event_counts = (
        ordered_counts.astype(np.int64, copy=False)
        * ordered_y.astype(np.int64, copy=False)
    )
    cumulative_events = np.cumsum(
        event_counts,
        dtype=np.int64,
    )

    selected: dict[int, int] = {}
    total_available = int(cumulative_count[-1])
    for budget in budgets:
        if budget > total_available:
            raise ValueError(
                f"Bootstrap budget={budget:,} exceeds available={total_available:,}"
            )
        cutoff = int(
            np.searchsorted(
                cumulative_count,
                budget,
                side="left",
            )
        )
        previous_count = (
            int(cumulative_count[cutoff - 1])
            if cutoff > 0
            else 0
        )
        previous_events = (
            int(cumulative_events[cutoff - 1])
            if cutoff > 0
            else 0
        )
        remaining = int(budget - previous_count)
        selected[budget] = int(
            previous_events
            + remaining * int(ordered_y[cutoff])
        )
    return selected


def paired_full_size_bootstrap(
    *,
    target: str,
    evaluation: str,
    y: np.ndarray,
    probabilities: np.ndarray,
    early_mask: np.ndarray,
    budget_fractions: list[float],
    replicates: int,
    seed: int,
    confidence_level: float,
    progress_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y_values = np.asarray(y, dtype=np.int8)
    probability_values = np.asarray(probabilities, dtype=np.float64)
    early_values = np.asarray(early_mask, dtype=bool)
    n = int(len(y_values))
    if n == 0:
        raise ValueError("Bootstrap population is empty.")

    original_indices = np.arange(n, dtype=np.int64)
    full_order = np.lexsort(
        (
            original_indices,
            -probability_values,
        )
    )
    early_indices = np.flatnonzero(early_values)
    early_order_local = np.lexsort(
        (
            early_indices,
            -probability_values[early_indices],
        )
    )
    early_order = early_indices[early_order_local]

    full_y_order = y_values[full_order]
    early_y_order = y_values[early_order]
    budgets = [
        max(1, int(round(float(fraction) * n)))
        for fraction in budget_fractions
    ]
    if max(budgets) > len(early_order):
        raise ValueError("Original early cohort is smaller than the budget.")

    # Point estimates from the original CDC evaluation population.
    point_rows, _ = fixed_budget_rows(
        population="singleton_primary",
        evaluation=evaluation,
        target=target,
        y=y_values,
        probabilities=probability_values,
        early_mask=early_values,
        budget_fractions=budget_fractions,
    )
    point_comparison = {
        float(row["budget_fraction_of_full_population"]): row
        for row in compare_early_full(point_rows)
    }

    rng = np.random.default_rng(seed)
    replicate_rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        sampled = rng.integers(
            0,
            n,
            size=n,
            dtype=np.int32,
        )
        counts = np.bincount(
            sampled,
            minlength=n,
        ).astype(np.int32, copy=False)
        del sampled

        event_n = int(
            np.sum(
                counts[y_values == 1],
                dtype=np.int64,
            )
        )
        if event_n <= 0:
            raise RuntimeError(
                f"{target}/{evaluation}/replicate={replicate}: no events."
            )

        full_selected = bootstrap_selected_true_positives(
            ordered_counts=counts[full_order],
            ordered_y=full_y_order,
            budgets=budgets,
        )
        early_selected = bootstrap_selected_true_positives(
            ordered_counts=counts[early_order],
            ordered_y=early_y_order,
            budgets=budgets,
        )
        early_sample_n = int(np.sum(counts[early_indices], dtype=np.int64))

        for fraction, budget in zip(budget_fractions, budgets):
            full_tp = int(full_selected[budget])
            early_tp = int(early_selected[budget])
            replicate_rows.append(
                {
                    "target": target,
                    "feature_set": "booking_strict",
                    "model": "lightgbm",
                    "evaluation": evaluation,
                    "replicate": replicate,
                    "bootstrap_sample_n": n,
                    "bootstrap_event_n": event_n,
                    "bootstrap_early_eligible_n": early_sample_n,
                    "budget_fraction_of_full_population": float(fraction),
                    "budget_n": budget,
                    "early_true_positive_n": early_tp,
                    "full_true_positive_n": full_tp,
                    "additional_true_positives_full_minus_early": (
                        full_tp - early_tp
                    ),
                    "early_population_event_capture": float(
                        early_tp / event_n
                    ),
                    "full_population_event_capture": float(
                        full_tp / event_n
                    ),
                    "population_event_capture_difference_full_minus_early": float(
                        (full_tp - early_tp) / event_n
                    ),
                }
            )

        del counts
        if (
            progress_every > 0
            and (
                (replicate + 1) % progress_every == 0
                or replicate + 1 == replicates
            )
        ):
            print(
                f"Bootstrap {target}/{evaluation}: "
                f"{replicate + 1}/{replicates}"
            )

    alpha = 1.0 - float(confidence_level)
    summary_rows: list[dict[str, Any]] = []
    replicate_frame = pd.DataFrame(replicate_rows)
    for fraction in budget_fractions:
        subset = replicate_frame.loc[
            replicate_frame[
                "budget_fraction_of_full_population"
            ].eq(float(fraction))
        ]
        values = subset[
            "population_event_capture_difference_full_minus_early"
        ].to_numpy(dtype=np.float64)
        tp_values = subset[
            "additional_true_positives_full_minus_early"
        ].to_numpy(dtype=np.float64)
        point = point_comparison[float(fraction)]
        summary_rows.append(
            {
                "target": target,
                "feature_set": "booking_strict",
                "model": "lightgbm",
                "evaluation": evaluation,
                "bootstrap_method": (
                    "paired full-size record-level nonparametric bootstrap"
                ),
                "replicates": int(replicates),
                "bootstrap_sample_n": n,
                "confidence_level": float(confidence_level),
                "budget_fraction_of_full_population": float(fraction),
                "budget_n": int(point["budget_n"]),
                "point_early_population_event_capture": float(
                    point["early_population_event_capture"]
                ),
                "point_full_population_event_capture": float(
                    point["full_population_event_capture"]
                ),
                "point_difference_full_minus_early": float(
                    point[
                        "population_event_capture_difference_full_minus_early"
                    ]
                ),
                "bootstrap_mean_difference": float(np.mean(values)),
                "bootstrap_standard_error": float(
                    np.std(values, ddof=1)
                ),
                "ci_lower": float(
                    np.quantile(values, alpha / 2.0)
                ),
                "ci_upper": float(
                    np.quantile(values, 1.0 - alpha / 2.0)
                ),
                "point_additional_true_positives_full_minus_early": int(
                    point[
                        "additional_true_positives_full_minus_early"
                    ]
                ),
                "bootstrap_mean_additional_true_positives": float(
                    np.mean(tp_values)
                ),
                "additional_true_positives_ci_lower": float(
                    np.quantile(tp_values, alpha / 2.0)
                ),
                "additional_true_positives_ci_upper": float(
                    np.quantile(tp_values, 1.0 - alpha / 2.0)
                ),
            }
        )
    return replicate_rows, summary_rows


# ---------------------------------------------------------------------------
# Mechanism table
# ---------------------------------------------------------------------------

def mechanism_rows(
    *,
    target: str,
    evaluation: str,
    y: np.ndarray,
    probabilities: np.ndarray,
    care_month: np.ndarray,
    budget_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y_values = np.asarray(y, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    early = early_entry_mask(care_month)
    point_rows, selections = fixed_budget_rows(
        population="singleton_primary",
        evaluation=evaluation,
        target=target,
        y=y_values,
        probabilities=probabilities,
        early_mask=early,
        budget_fractions=[budget_fraction],
    )
    del point_rows

    groups = care_entry_group_masks(care_month)
    rows: list[dict[str, Any]] = []
    by_group_scenario: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for scenario in ("early_entry", "full_cohort"):
        selected_indices = selections[(float(budget_fraction), scenario)]
        selected_mask = np.zeros(len(y_values), dtype=bool)
        selected_mask[selected_indices] = True
        for group_name, group_mask in groups.items():
            selected_group = selected_mask & group_mask
            row = {
                "target": target,
                "feature_set": "booking_strict",
                "model": "lightgbm",
                "evaluation": evaluation,
                "budget_fraction_of_full_population": float(
                    budget_fraction
                ),
                "budget_n": int(len(selected_indices)),
                "scenario": scenario,
                "care_entry_group": group_name,
                "group_population_n": int(np.sum(group_mask)),
                "event_n": int(np.sum(y_values[group_mask] == 1)),
                "selected_n": int(np.sum(selected_group)),
                "true_positive_n": int(
                    np.sum(y_values[selected_group] == 1)
                ),
            }
            rows.append(row)
            by_group_scenario[(group_name, scenario)] = row

    comparison_rows: list[dict[str, Any]] = []
    for group_name in groups:
        early_row = by_group_scenario[(group_name, "early_entry")]
        full_row = by_group_scenario[(group_name, "full_cohort")]
        comparison_rows.append(
            {
                "target": target,
                "feature_set": "booking_strict",
                "model": "lightgbm",
                "evaluation": evaluation,
                "budget_fraction_of_full_population": float(
                    budget_fraction
                ),
                "care_entry_group": group_name,
                "group_population_n": int(full_row["group_population_n"]),
                "event_n": int(full_row["event_n"]),
                "early_selected_n": int(early_row["selected_n"]),
                "full_selected_n": int(full_row["selected_n"]),
                "additional_selected_n_full_minus_early": int(
                    full_row["selected_n"] - early_row["selected_n"]
                ),
                "early_true_positive_n": int(
                    early_row["true_positive_n"]
                ),
                "full_true_positive_n": int(
                    full_row["true_positive_n"]
                ),
                "additional_true_positives_full_minus_early": int(
                    full_row["true_positive_n"]
                    - early_row["true_positive_n"]
                ),
            }
        )
    return rows, comparison_rows


# ---------------------------------------------------------------------------
# Calibration metrics by full and early populations
# ---------------------------------------------------------------------------

def calibration_metrics_by_scenario(
    *,
    target: str,
    evaluation: str,
    y: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    early_mask: np.ndarray,
    curve_bins: int,
    eps: float,
    optimizer_max_iter: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    metrics_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    scenarios = {
        "full_cohort": np.ones(len(y), dtype=bool),
        "early_entry": np.asarray(early_mask, dtype=bool),
    }
    for scenario, mask in scenarios.items():
        y_subset = np.asarray(y, dtype=np.int8)[mask]
        raw_subset = np.asarray(
            raw_probabilities,
            dtype=np.float64,
        )[mask]
        calibrated_subset = np.asarray(
            calibrated_probabilities,
            dtype=np.float64,
        )[mask]

        raw_metrics, raw_curve = probability_metrics(
            target,
            "booking_strict",
            evaluation,
            "raw",
            y_subset,
            raw_subset,
            curve_bins,
            eps,
            optimizer_max_iter,
            False,
        )
        calibrated_metrics, calibrated_curve = probability_metrics(
            target,
            "booking_strict",
            evaluation,
            "calibrated",
            y_subset,
            calibrated_subset,
            curve_bins,
            eps,
            optimizer_max_iter,
            False,
        )

        for metric in (raw_metrics, calibrated_metrics):
            prevalence = float(metric["prevalence"])
            null_brier = prevalence * (1.0 - prevalence)
            metric["population"] = "singleton_primary"
            metric["scenario"] = scenario
            metric["null_brier"] = float(null_brier)
            metric["brier_skill_score"] = (
                float(1.0 - float(metric["brier"]) / null_brier)
                if null_brier > 0.0
                else math.nan
            )

        for row in raw_curve + calibrated_curve:
            row["population"] = "singleton_primary"
            row["scenario"] = scenario

        comparison = compare_raw_calibrated(
            raw_metrics,
            calibrated_metrics,
        )
        comparison["population"] = "singleton_primary"
        comparison["scenario"] = scenario
        comparison["raw_null_brier"] = float(
            raw_metrics["null_brier"]
        )
        comparison["calibrated_null_brier"] = float(
            calibrated_metrics["null_brier"]
        )
        comparison["raw_brier_skill_score"] = float(
            raw_metrics["brier_skill_score"]
        )
        comparison["calibrated_brier_skill_score"] = float(
            calibrated_metrics["brier_skill_score"]
        )

        metrics_rows.extend([raw_metrics, calibrated_metrics])
        comparison_rows.append(comparison)
        curve_rows.extend(raw_curve)
        curve_rows.extend(calibrated_curve)

    return metrics_rows, comparison_rows, curve_rows


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def article_revision_summary(
    singleton_comparisons: list[dict[str, Any]],
    bootstrap_summary: list[dict[str, Any]],
    mechanism_comparisons: list[dict[str, Any]],
    calibration_comparisons: list[dict[str, Any]],
    all_birth_comparisons: list[dict[str, Any]],
) -> str:
    lines = [
        "# Article-revision analysis summary",
        "",
        "All analyses use LightGBM and `booking_strict` features. "
        "CDC 2023 is described as a temporal test.",
        "",
        "## Primary singleton fixed-model equal-budget result",
        "",
        "| Evaluation | Target | Budget | Early capture | Full capture | Full − early | Additional TP |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        singleton_comparisons,
        key=lambda item: (
            str(item["evaluation"]),
            str(item["target"]),
            float(item["budget_fraction_of_full_population"]),
        ),
    ):
        lines.append(
            f"| {row['evaluation']} | {row['target']} | "
            f"{100 * row['budget_fraction_of_full_population']:.0f}% | "
            f"{row['early_population_event_capture']:.4f} | "
            f"{row['full_population_event_capture']:.4f} | "
            f"{row['population_event_capture_difference_full_minus_early']:+.4f} | "
            f"{row['additional_true_positives_full_minus_early']:,} |"
        )

    lines.extend(
        [
            "",
            "## Paired full-size bootstrap",
            "",
            "| Evaluation | Target | Budget | Point difference | 95% CI | Additional TP (point) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        bootstrap_summary,
        key=lambda item: (
            str(item["evaluation"]),
            str(item["target"]),
            float(item["budget_fraction_of_full_population"]),
        ),
    ):
        lines.append(
            f"| {row['evaluation']} | {row['target']} | "
            f"{100 * row['budget_fraction_of_full_population']:.0f}% | "
            f"{row['point_difference_full_minus_early']:+.4f} | "
            f"[{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}] | "
            f"{row['point_additional_true_positives_full_minus_early']:,} |"
        )

    lines.extend(
        [
            "",
            "## Mechanism at the 10% budget",
            "",
            "| Target | Care-entry group | Group events | Additional selected (full − early) | Additional TP (full − early) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(
        mechanism_comparisons,
        key=lambda item: (
            str(item["target"]),
            str(item["care_entry_group"]),
        ),
    ):
        lines.append(
            f"| {row['target']} | {row['care_entry_group']} | "
            f"{row['event_n']:,} | "
            f"{row['additional_selected_n_full_minus_early']:,} | "
            f"{row['additional_true_positives_full_minus_early']:,} |"
        )

    lines.extend(
        [
            "",
            "## Primary LightGBM calibration",
            "",
            "| Evaluation | Scenario | Target | Raw Brier | Calibrated Brier | Raw BSS | Calibrated BSS | Raw intercept/slope | Calibrated intercept/slope |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        calibration_comparisons,
        key=lambda item: (
            str(item["evaluation"]),
            str(item["scenario"]),
            str(item["target"]),
        ),
    ):
        lines.append(
            f"| {row['evaluation']} | {row['scenario']} | {row['target']} | "
            f"{row['raw_brier']:.6f} | {row['calibrated_brier']:.6f} | "
            f"{row['raw_brier_skill_score']:.4f} | "
            f"{row['calibrated_brier_skill_score']:.4f} | "
            f"{row['raw_calibration_intercept']:+.3f}/{row['raw_calibration_slope']:.3f} | "
            f"{row['calibrated_calibration_intercept']:+.3f}/{row['calibrated_calibration_slope']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## All-birth sensitivity",
            "",
            "| Target | Budget | Early capture | Full capture | Full − early | Additional TP |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        all_birth_comparisons,
        key=lambda item: (
            str(item["target"]),
            float(item["budget_fraction_of_full_population"]),
        ),
    ):
        lines.append(
            f"| {row['target']} | "
            f"{100 * row['budget_fraction_of_full_population']:.0f}% | "
            f"{row['early_population_event_capture']:.4f} | "
            f"{row['full_population_event_capture']:.4f} | "
            f"{row['population_event_capture_difference_full_minus_early']:+.4f} | "
            f"{row['additional_true_positives_full_minus_early']:,} |"
        )

    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `full precision` and `early precision` are compared at the same absolute budget.",
            "- `recall_among_eligible` is conditional on scoreability and is not population event capture.",
            "- The all-birth analysis is sensitivity only because multiple infant records may be dependent and no pregnancy-level identifier is available.",
            "- Platt scaling is monotone when its fitted slope is positive; the script verifies this and checks that top-B ranking is unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ClearML reporting
# ---------------------------------------------------------------------------

def report_to_clearml(
    task: Any,
    singleton_results: list[dict[str, Any]],
    singleton_comparisons: list[dict[str, Any]],
    bootstrap_summary: list[dict[str, Any]],
    mechanism_comparisons: list[dict[str, Any]],
    calibration_metrics_rows: list[dict[str, Any]],
    calibration_comparisons: list[dict[str, Any]],
    calibration_curve_rows: list[dict[str, Any]],
    all_birth_results: list[dict[str, Any]],
    all_birth_comparisons: list[dict[str, Any]],
) -> None:
    logger = task.get_logger()
    tables = [
        (
            "01 Singleton fixed-budget results",
            singleton_results,
        ),
        (
            "02 Singleton early versus full",
            singleton_comparisons,
        ),
        (
            "03 Paired full-size bootstrap",
            bootstrap_summary,
        ),
        (
            "04 Mechanism table",
            mechanism_comparisons,
        ),
        (
            "05 LightGBM calibration metrics",
            calibration_metrics_rows,
        ),
        (
            "06 Raw versus calibrated",
            calibration_comparisons,
        ),
        (
            "07 All-birth fixed-budget sensitivity",
            all_birth_results,
        ),
        (
            "08 All-birth early versus full",
            all_birth_comparisons,
        ),
    ]
    for title, rows in tables:
        logger.report_table(
            title=title,
            series="article revision",
            iteration=0,
            table_plot=pd.DataFrame(rows).round(8),
        )

    curves = pd.DataFrame(calibration_curve_rows)
    for target in TARGETS:
        for scenario in ("early_entry", "full_cohort"):
            subset = curves.loc[
                (curves["target"] == target)
                & (curves["evaluation"] == "temporal_test_2023")
                & (curves["scenario"] == scenario)
            ]
            if subset.empty:
                continue
            logger.report_plotly(
                title=f"Reliability curve · {target} · {scenario}",
                series="CDC 2023 temporal test",
                iteration=0,
                figure=calibration_plotly_figure(
                    subset,
                    f"{target} / {scenario} / CDC 2023 temporal test",
                ),
            )

    for row in bootstrap_summary:
        if (
            row["evaluation"] == "temporal_test_2023"
            and float(row["budget_fraction_of_full_population"]) == 0.10
        ):
            logger.report_single_value(
                name=(
                    f"capture_difference_10pct_"
                    f"{row['target']}"
                ),
                value=float(row["point_difference_full_minus_early"]),
            )
            logger.report_single_value(
                name=(
                    f"capture_difference_10pct_ci_lower_"
                    f"{row['target']}"
                ),
                value=float(row["ci_lower"]),
            )
            logger.report_single_value(
                name=(
                    f"capture_difference_10pct_ci_upper_"
                    f"{row['target']}"
                ),
                value=float(row["ci_upper"]),
            )
    logger.flush(wait=True)

# ---------------------------------------------------------------------------
# Probability calibration and diagnostics
# ---------------------------------------------------------------------------

def clip_probabilities(probabilities: np.ndarray, eps: float) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Probabilities must be one-dimensional.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Probabilities contain NaN or infinity.")
    return np.clip(values, eps, 1.0 - eps)


def probability_logit(probabilities: np.ndarray, eps: float) -> np.ndarray:
    clipped = clip_probabilities(probabilities, eps)
    return np.log(clipped) - np.log1p(-clipped)


def fit_logit_mapping(
    y: np.ndarray,
    probabilities: np.ndarray,
    eps: float,
    max_iter: int,
    purpose: str,
    require_success: bool,
) -> dict[str, Any]:
    y_values = np.asarray(y, dtype=np.float64)
    if y_values.ndim != 1 or len(y_values) != len(probabilities) or len(y_values) == 0:
        raise ValueError(f"{purpose}: invalid input dimensions.")
    unique = np.unique(y_values)
    if not np.array_equal(unique, np.array([0.0, 1.0])):
        raise ValueError(f"{purpose}: both binary classes are required; got {unique}.")

    x = probability_logit(probabilities, eps)
    event_rate = float(np.mean(y_values))
    mean_probability = float(np.mean(clip_probabilities(probabilities, eps)))

    initial = np.array(
        [
            math.log(event_rate / (1.0 - event_rate))
            - math.log(mean_probability / (1.0 - mean_probability)),
            1.0,
        ],
        dtype=np.float64,
    )

    def objective(theta: np.ndarray) -> float:
        linear = theta[0] + theta[1] * x
        return float(np.mean(np.logaddexp(0.0, linear) - y_values * linear))

    def gradient(theta: np.ndarray) -> np.ndarray:
        linear = theta[0] + theta[1] * x
        residual = expit(linear) - y_values
        return np.array(
            [float(np.mean(residual)), float(np.mean(residual * x))],
            dtype=np.float64,
        )

    attempts: list[tuple[str, Any]] = []
    result = minimize(
        objective,
        x0=initial,
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(-30.0, 30.0), (-20.0, 20.0)],
        options={
            "maxiter": int(max_iter),
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 50,
        },
    )
    attempts.append(("L-BFGS-B", result))
    best = result
    best_method = "L-BFGS-B"

    if not result.success:
        fallback = minimize(
            objective,
            x0=np.asarray(result.x, dtype=np.float64),
            method="Powell",
            bounds=[(-30.0, 30.0), (-20.0, 20.0)],
            options={
                "maxiter": int(max_iter) * 3,
                "xtol": 1e-8,
                "ftol": 1e-10,
            },
        )
        attempts.append(("Powell", fallback))
        if np.isfinite(fallback.fun) and (
            not np.isfinite(result.fun) or float(fallback.fun) <= float(result.fun)
        ):
            best = fallback
            best_method = "Powell"

    finite = np.all(np.isfinite(best.x))
    success = bool(best.success and finite)
    if require_success and not success:
        detail = "; ".join(
            f"{name}: success={attempt.success}, message={attempt.message}"
            for name, attempt in attempts
        )
        raise RuntimeError(f"{purpose} optimizer failed. {detail}")

    return {
        "intercept": float(best.x[0]) if finite else math.nan,
        "slope": float(best.x[1]) if finite else math.nan,
        "optimizer_success": success,
        "optimizer_method": best_method,
        "optimizer_message": str(best.message),
        "optimizer_iterations": int(getattr(best, "nit", -1)),
        "negative_log_likelihood": float(best.fun) if np.isfinite(best.fun) else math.nan,
    }


def apply_platt(
    probabilities: np.ndarray,
    intercept: float,
    slope: float,
    eps: float,
) -> np.ndarray:
    calibrated = expit(intercept + slope * probability_logit(probabilities, eps))
    return clip_probabilities(calibrated, eps)


def quantile_calibration_curve(
    y: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int,
) -> tuple[list[dict[str, Any]], float, float]:
    y_values = np.asarray(y, dtype=np.int8)
    probability_values = np.asarray(probabilities, dtype=np.float64)
    if len(y_values) != len(probability_values) or len(y_values) == 0:
        raise ValueError("Calibration curve received invalid arrays.")

    actual_bins = min(int(n_bins), len(y_values))
    order = np.argsort(probability_values, kind="mergesort")
    boundaries = np.linspace(0, len(order), actual_bins + 1, dtype=np.int64)

    rows: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    for bin_index in range(actual_bins):
        indices = order[boundaries[bin_index] : boundaries[bin_index + 1]]
        if len(indices) == 0:
            continue
        bin_y = y_values[indices]
        bin_p = probability_values[indices]
        observed = float(np.mean(bin_y))
        predicted = float(np.mean(bin_p))
        gap = abs(observed - predicted)
        ece += (len(indices) / len(y_values)) * gap
        mce = max(mce, gap)
        rows.append(
            {
                "bin": int(bin_index + 1),
                "bin_n": int(len(indices)),
                "event_n": int(np.sum(bin_y == 1)),
                "mean_predicted_probability": predicted,
                "observed_event_rate": observed,
                "absolute_gap": float(gap),
                "minimum_probability": float(np.min(bin_p)),
                "maximum_probability": float(np.max(bin_p)),
            }
        )
    return rows, float(ece), float(mce)


def probability_metrics(
    target: str,
    feature_set: str,
    evaluation: str,
    probability_type: str,
    y: np.ndarray,
    probabilities: np.ndarray,
    curve_bins: int,
    eps: float,
    optimizer_max_iter: int,
    fit_split: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_values = np.asarray(y, dtype=np.int8)
    probability_values = clip_probabilities(probabilities, eps)
    if len(np.unique(y_values)) != 2:
        raise ValueError(f"{target}/{feature_set}/{evaluation}: both classes are required.")

    curve, ece, mce = quantile_calibration_curve(
        y_values,
        probability_values,
        curve_bins,
    )
    diagnostic = fit_logit_mapping(
        y_values,
        probability_values,
        eps,
        optimizer_max_iter,
        f"diagnostic {target}/{feature_set}/{evaluation}/{probability_type}",
        False,
    )

    metrics = {
        "target": target,
        "feature_set": feature_set,
        "model": "lightgbm",
        "evaluation": evaluation,
        "probability_type": probability_type,
        "is_calibrator_fit_split": bool(fit_split),
        "n": int(len(y_values)),
        "event_n": int(np.sum(y_values == 1)),
        "prevalence": float(np.mean(y_values)),
        "mean_predicted_probability": float(np.mean(probability_values)),
        "brier": float(brier_score_loss(y_values, probability_values)),
        "log_loss": float(log_loss(y_values, probability_values, labels=[0, 1])),
        "pr_auc": float(average_precision_score(y_values, probability_values)),
        "roc_auc": float(roc_auc_score(y_values, probability_values)),
        "calibration_intercept": float(diagnostic["intercept"]),
        "calibration_slope": float(diagnostic["slope"]),
        "calibration_fit_optimizer_success": bool(diagnostic["optimizer_success"]),
        "calibration_fit_optimizer_method": str(diagnostic["optimizer_method"]),
        "calibration_fit_optimizer_message": str(diagnostic["optimizer_message"]),
        "ece": float(ece),
        "mce": float(mce),
        "curve_bins_requested": int(curve_bins),
        "curve_bins_observed": int(len(curve)),
    }

    enriched = [
        {
            "target": target,
            "feature_set": feature_set,
            "model": "lightgbm",
            "evaluation": evaluation,
            "probability_type": probability_type,
            **row,
        }
        for row in curve
    ]
    return metrics, enriched


def compare_raw_calibrated(raw: dict[str, Any], calibrated: dict[str, Any]) -> dict[str, Any]:
    for key in ("target", "feature_set", "model", "evaluation", "n", "event_n"):
        if raw[key] != calibrated[key]:
            raise ValueError(f"Raw/calibrated mismatch for {key}: {raw[key]} != {calibrated[key]}")

    return {
        "target": raw["target"],
        "feature_set": raw["feature_set"],
        "model": raw["model"],
        "evaluation": raw["evaluation"],
        "n": int(raw["n"]),
        "event_n": int(raw["event_n"]),
        "raw_brier": float(raw["brier"]),
        "calibrated_brier": float(calibrated["brier"]),
        "brier_delta_calibrated_minus_raw": float(calibrated["brier"] - raw["brier"]),
        "brier_improvement_raw_minus_calibrated": float(raw["brier"] - calibrated["brier"]),
        "raw_log_loss": float(raw["log_loss"]),
        "calibrated_log_loss": float(calibrated["log_loss"]),
        "log_loss_delta_calibrated_minus_raw": float(calibrated["log_loss"] - raw["log_loss"]),
        "raw_ece": float(raw["ece"]),
        "calibrated_ece": float(calibrated["ece"]),
        "ece_delta_calibrated_minus_raw": float(calibrated["ece"] - raw["ece"]),
        "raw_calibration_intercept": float(raw["calibration_intercept"]),
        "calibrated_calibration_intercept": float(calibrated["calibration_intercept"]),
        "raw_calibration_slope": float(raw["calibration_slope"]),
        "calibrated_calibration_slope": float(calibrated["calibration_slope"]),
        "raw_pr_auc": float(raw["pr_auc"]),
        "calibrated_pr_auc": float(calibrated["pr_auc"]),
        "pr_auc_delta_calibrated_minus_raw": float(calibrated["pr_auc"] - raw["pr_auc"]),
        "raw_roc_auc": float(raw["roc_auc"]),
        "calibrated_roc_auc": float(calibrated["roc_auc"]),
        "roc_auc_delta_calibrated_minus_raw": float(calibrated["roc_auc"] - raw["roc_auc"]),
    }

# ---------------------------------------------------------------------------
# Reliability-curve Plotly figure
# ---------------------------------------------------------------------------

def calibration_plotly_figure(rows: pd.DataFrame, title: str) -> dict[str, Any]:
    traces: list[dict[str, Any]] = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Ideal",
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "line": {"dash": "dash"},
        }
    ]
    for probability_type, label in (("raw", "Raw"), ("calibrated", "Calibrated")):
        subset = rows.loc[rows["probability_type"] == probability_type].sort_values(
            "mean_predicted_probability"
        )
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": label,
                "x": subset["mean_predicted_probability"].astype(float).tolist(),
                "y": subset["observed_event_rate"].astype(float).tolist(),
                "text": [f"n={int(value):,}" for value in subset["bin_n"]],
                "hovertemplate": (
                    "Predicted=%{x:.4f}<br>Observed=%{y:.4f}<br>%{text}<extra></extra>"
                ),
            }
        )
    return {
        "data": traces,
        "layout": {
            "title": {"text": title, "x": 0.02},
            "height": 560,
            "xaxis": {"title": "Mean predicted probability", "range": [0.0, 0.5]},
            "yaxis": {"title": "Observed event rate", "range": [0.0, 0.5]},
            "legend": {"orientation": "h", "y": 1.08},
        },
    }

def output_hashes(output: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "run_manifest.json":
            continue
        if "models" in path.parts or "predictions" in path.parts:
            continue
        result[str(path.relative_to(output))] = sha256_file(path)
    return result

# ---------------------------------------------------------------------------
# Reproducibility bundle and smoke test
# ---------------------------------------------------------------------------

def save_reproducibility_bundle(
    output: Path,
    config: dict[str, Any],
) -> None:
    code_directory = output / "reproducibility" / "code"
    config_directory = output / "reproducibility" / "configs"
    code_directory.mkdir(parents=True, exist_ok=True)
    config_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT_PATH, code_directory / SCRIPT_PATH.name)
    atomic_text(
        config_directory / "experiment_config_article_revision.locked.yaml",
        yaml_dump_safe(config),
    )


def smoke_test() -> int:
    rng = np.random.default_rng(2026)
    n = 20_000
    latent = rng.normal(size=n)
    true_probability = expit(-2.0 + 0.8 * latent)
    y = rng.binomial(1, true_probability).astype(np.int8)
    raw = expit(-2.7 + 1.2 * latent)
    care = rng.choice(
        [1.0, 2.0, 3.0, 4.0, 0.0, np.nan],
        size=n,
        p=[0.25, 0.20, 0.15, 0.20, 0.05, 0.15],
    )
    early = early_entry_mask(care)

    fitted = fit_logit_mapping(
        y[:10_000],
        raw[:10_000],
        1e-6,
        300,
        "synthetic Platt smoke test",
        True,
    )
    calibrated = apply_platt(
        raw[10_000:],
        float(fitted["intercept"]),
        float(fitted["slope"]),
        1e-6,
    )
    raw_brier = brier_score_loss(
        y[10_000:],
        raw[10_000:],
    )
    calibrated_brier = brier_score_loss(
        y[10_000:],
        calibrated,
    )
    if calibrated_brier >= raw_brier:
        raise AssertionError("Synthetic Platt calibration did not improve.")

    point_rows, _ = fixed_budget_rows(
        population="smoke",
        evaluation="smoke",
        target="target_smoke",
        y=y,
        probabilities=raw,
        early_mask=early,
        budget_fractions=[0.05, 0.10],
    )
    comparisons = compare_early_full(point_rows)
    if len(comparisons) != 2:
        raise AssertionError("Fixed-budget smoke comparison failed.")

    replicate_rows, summary_rows = paired_full_size_bootstrap(
        target="target_smoke",
        evaluation="smoke",
        y=y,
        probabilities=raw,
        early_mask=early,
        budget_fractions=[0.05, 0.10],
        replicates=20,
        seed=2027,
        confidence_level=0.95,
        progress_every=0,
    )
    if len(replicate_rows) != 40 or len(summary_rows) != 2:
        raise AssertionError("Paired-bootstrap smoke test failed.")

    mechanism, mechanism_comparison = mechanism_rows(
        target="target_smoke",
        evaluation="smoke",
        y=y,
        probabilities=raw,
        care_month=care,
        budget_fraction=0.10,
    )
    if len(mechanism) != 8 or len(mechanism_comparison) != 4:
        raise AssertionError("Mechanism-table smoke test failed.")

    print("SMOKE TEST PASS")
    print(f"raw_brier={raw_brier:.8f}")
    print(f"calibrated_brier={calibrated_brier:.8f}")
    print("fixed_budget=PASS")
    print("paired_full_size_bootstrap=PASS")
    print("mechanism_table=PASS")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone ClearML article-revision analyses: paired bootstrap, "
            "mechanism table, LightGBM calibration, and all-birth sensitivity"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML override.",
    )
    parser.add_argument(
        "--dataset-id",
        default=os.getenv("CLEARML_DATASET_ID") or DEFAULT_DATASET_ID,
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
        default=os.getenv("CLEARML_PROJECT_NAME") or DEFAULT_PROJECT_NAME,
    )
    parser.add_argument(
        "--task-name",
        default=(
            os.getenv("CLEARML_TASK_NAME_ARTICLE_REVISION")
            or DEFAULT_TASK_NAME
        ),
    )
    parser.add_argument("--dataset-download-workers", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--skip-clearml", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if args.smoke_test:
        return smoke_test()

    config = load_local_config(args.config)
    if args.bootstrap_replicates is not None:
        config["bootstrap"]["replicates"] = int(
            args.bootstrap_replicates
        )
    validate_config(config)

    task: Optional[Any] = None
    dataset_root: Optional[Path] = None

    if args.skip_clearml:
        if not args.dataset_id:
            raise ValueError("--dataset-id is required.")
        dataset_root = bind_clearml_dataset(
            args.dataset_id,
            args.dataset_download_workers,
        )
    else:
        from clearml import Task

        Task.force_store_standalone_script(True)
        for package_name, package_version in CLEARML_EXTRA_REQUIREMENTS:
            Task.add_requirements(package_name, package_version)

        task = Task.init(
            project_name=args.project_name,
            task_name=args.task_name,
            task_type=Task.TaskTypes.training,
            tags=TAGS,
            reuse_last_task_id=False,
            output_uri=True,
        )

        config = task.connect_configuration(
            configuration=config,
            name="experiment_config",
            description=(
                "Locked article-revision analysis configuration"
            ),
        )
        config = to_plain_python(config)
        runtime = task.connect(
            {
                "dataset_id": str(args.dataset_id),
                "queue": str(args.queue),
                "dataset_download_workers": int(
                    args.dataset_download_workers
                ),
                "bootstrap_replicates": int(
                    config["bootstrap"]["replicates"]
                ),
            },
            name="runtime",
        )
        runtime = to_plain_python(runtime)
        args.dataset_id = str(runtime["dataset_id"])
        args.queue = str(runtime["queue"])
        args.dataset_download_workers = int(
            runtime["dataset_download_workers"]
        )
        config["bootstrap"]["replicates"] = int(
            runtime["bootstrap_replicates"]
        )
        validate_config(config)

        if not args.local:
            if not args.queue:
                raise ValueError("--queue is required.")
            if not args.dataset_id:
                raise ValueError("--dataset-id is required.")
            task.execute_remotely(
                queue_name=args.queue,
                clone=False,
                exit_process=True,
            )

        dataset_root = bind_clearml_dataset(
            args.dataset_id,
            args.dataset_download_workers,
        )

    if dataset_root is None:
        raise RuntimeError("Dataset root was not resolved.")

    started = time.time()
    output = prepare_output(config, args.overwrite)
    models_directory = output / "models"
    predictions_directory = output / "predictions"
    if bool(config["output"].get("save_models", True)):
        models_directory.mkdir(parents=True, exist_ok=True)
    if bool(config["output"].get("save_predictions", False)):
        predictions_directory.mkdir(parents=True, exist_ok=True)

    atomic_text(
        output / "commands.txt",
        " ".join(
            shlex.quote(str(value))
            for value in [sys.executable, *sys.argv]
        )
        + "\n",
    )
    atomic_text(output / "package_versions.txt", package_versions())
    atomic_text(
        output / "experiment_config.locked.yaml",
        yaml_dump_safe(config),
    )
    atomic_json(output / "git_info.json", git_information())
    save_reproducibility_bundle(output, config)

    cdc2022_path = resolve_dataset_file(
        dataset_root,
        str(config["data"]["cdc2022"]),
    )
    cdc2023_path = resolve_dataset_file(
        dataset_root,
        str(config["data"]["cdc2023"]),
    )
    feature_sets = load_feature_sets(dataset_root, config)
    features = feature_sets["booking_strict"]
    columns = required_columns(feature_sets)

    expected_rows = config["data"].get("expected_rows", {})
    frame_2022 = read_cdc_frame(
        cdc2022_path,
        columns,
        int(expected_rows["2022"])
        if "2022" in expected_rows
        else None,
        "CDC 2022",
    )
    frame_2023 = read_cdc_frame(
        cdc2023_path,
        columns,
        int(expected_rows["2023"])
        if "2023" in expected_rows
        else None,
        "CDC 2023",
    )

    for label, frame in (
        ("CDC 2022", frame_2022),
        ("CDC 2023", frame_2023),
    ):
        validate_binary_column(
            frame,
            "is_us_resident",
            label,
            allow_missing=False,
        )
        validate_binary_column(
            frame,
            "is_singleton",
            label,
            allow_missing=False,
        )
        frame["prenatal_care_month"] = pd.to_numeric(
            frame["prenatal_care_month"],
            errors="coerce",
        )
        for target in TARGETS:
            validate_binary_column(
                frame,
                target,
                label,
                allow_missing=True,
            )

    observed_input_hashes: dict[str, Optional[str]] = {
        "2022": None,
        "2023": None,
    }
    if bool(config["data"].get("verify_input_hashes", True)):
        print("Hashing CDC 2022 input...")
        observed_input_hashes["2022"] = sha256_file(cdc2022_path)
        print("Hashing CDC 2023 input...")
        observed_input_hashes["2023"] = sha256_file(cdc2023_path)

    split_seed = int(config["split"]["seed"])
    calibration_fraction = float(
        config["split"]["calibration_fraction"]
    )
    calibration_config = config["probability_calibration"]
    partition_seed = int(calibration_config["partition_seed"])
    fit_fraction = float(
        calibration_config[
            "fit_fraction_within_calibration_pool"
        ]
    )
    eps = float(calibration_config["clip_eps"])
    curve_bins = int(calibration_config["curve_bins"])
    optimizer_max_iter = int(
        calibration_config["optimizer_max_iter"]
    )
    budget_fractions = [
        float(value)
        for value in config["fixed_absolute_budget"][
            "fractions_of_full_population"
        ]
    ]

    split_rng = np.random.default_rng(split_seed)
    calibration_pool = (
        split_rng.random(len(frame_2022))
        < calibration_fraction
    )
    train_split = ~calibration_pool
    partition_rng = np.random.default_rng(partition_seed)
    calibrator_draw = (
        partition_rng.random(len(frame_2022))
        < fit_fraction
    )
    calibrator_split = calibration_pool & calibrator_draw
    calibration_evaluation_split = (
        calibration_pool & ~calibrator_draw
    )

    if np.any(
        train_split
        & (
            calibrator_split
            | calibration_evaluation_split
        )
    ):
        raise RuntimeError("Train/calibration overlap.")
    if np.any(
        calibrator_split
        & calibration_evaluation_split
    ):
        raise RuntimeError("Calibration sub-splits overlap.")
    if not np.array_equal(
        calibrator_split | calibration_evaluation_split,
        calibration_pool,
    ):
        raise RuntimeError("Calibration sub-splits do not cover pool.")

    singleton_2022 = resident_singleton_mask(frame_2022)
    singleton_2023 = resident_singleton_mask(frame_2023)
    all_births_2022 = resident_all_births_mask(frame_2022)
    all_births_2023 = resident_all_births_mask(frame_2023)

    categorical_features = [
        str(value)
        for value in config["preprocessing"]["categorical_features"]
    ]
    parameters = lightgbm_parameters(config)

    from lightgbm import LGBMClassifier

    singleton_results: list[dict[str, Any]] = []
    singleton_comparisons: list[dict[str, Any]] = []
    bootstrap_replicates_rows: list[dict[str, Any]] = []
    bootstrap_summary_rows: list[dict[str, Any]] = []
    mechanism_table_rows: list[dict[str, Any]] = []
    mechanism_comparison_rows: list[dict[str, Any]] = []
    calibration_metrics_rows: list[dict[str, Any]] = []
    calibration_comparison_rows: list[dict[str, Any]] = []
    calibration_curve_rows: list[dict[str, Any]] = []
    calibrator_rows: list[dict[str, Any]] = []
    all_birth_results: list[dict[str, Any]] = []
    all_birth_comparisons: list[dict[str, Any]] = []
    saved_models: list[str] = []
    saved_predictions: list[str] = []
    singleton_model_count = 0
    all_birth_model_count = 0
    temporal_prediction_pass_count = 0

    # ---------------------------------------------------------------
    # Primary singleton models: fixed budget, bootstrap, mechanism,
    # and calibration.
    # ---------------------------------------------------------------
    for target_index, target in enumerate(TARGETS):
        print(f"\n===== SINGLETON PRIMARY: {target} =====")
        known_2022 = (
            frame_2022[target].notna().to_numpy(dtype=bool)
        )
        known_2023 = (
            frame_2023[target].notna().to_numpy(dtype=bool)
        )
        train_mask = (
            singleton_2022
            & known_2022
            & train_split
        )
        calibrator_mask = (
            singleton_2022
            & known_2022
            & calibrator_split
        )
        evaluation_mask = (
            singleton_2022
            & known_2022
            & calibration_evaluation_split
        )
        test_mask = singleton_2023 & known_2023

        y_train = (
            frame_2022.loc[train_mask, target]
            .to_numpy(dtype=np.int8)
        )
        y_calibrator = (
            frame_2022.loc[calibrator_mask, target]
            .to_numpy(dtype=np.int8)
        )
        y_evaluation = (
            frame_2022.loc[evaluation_mask, target]
            .to_numpy(dtype=np.int8)
        )
        y_test = (
            frame_2023.loc[test_mask, target]
            .to_numpy(dtype=np.int8)
        )
        for name, values in (
            ("train", y_train),
            ("calibrator_fit", y_calibrator),
            ("heldout_2022", y_evaluation),
            ("temporal_test_2023", y_test),
        ):
            if len(np.unique(values)) != 2:
                raise ValueError(
                    f"{target}/{name} does not contain both classes."
                )

        fit_started = time.time()
        preprocessor = build_preprocessor(
            features,
            categorical_features,
            config["preprocessing"],
        )
        x_train = consistent_matrix(
            preprocessor.fit_transform(
                frame_2022.loc[train_mask, features]
            )
        )
        model = LGBMClassifier(**parameters)
        model.fit(x_train, y_train)
        transformed_feature_n = int(x_train.shape[1])
        del x_train
        gc.collect()
        singleton_model_count += 1

        x_calibrator = consistent_matrix(
            preprocessor.transform(
                frame_2022.loc[calibrator_mask, features]
            )
        )
        raw_calibrator = np.asarray(
            model.predict_proba(x_calibrator)[:, 1],
            dtype=np.float64,
        )
        del x_calibrator
        gc.collect()

        x_evaluation = consistent_matrix(
            preprocessor.transform(
                frame_2022.loc[evaluation_mask, features]
            )
        )
        raw_evaluation = np.asarray(
            model.predict_proba(x_evaluation)[:, 1],
            dtype=np.float64,
        )
        del x_evaluation
        gc.collect()

        x_test = consistent_matrix(
            preprocessor.transform(
                frame_2023.loc[test_mask, features]
            )
        )
        raw_test = np.asarray(
            model.predict_proba(x_test)[:, 1],
            dtype=np.float64,
        )
        del x_test
        gc.collect()
        temporal_prediction_pass_count += 1

        platt = fit_logit_mapping(
            y_calibrator,
            raw_calibrator,
            eps,
            optimizer_max_iter,
            f"Platt singleton/{target}/booking_strict",
            True,
        )
        if float(platt["slope"]) <= 0.0:
            raise RuntimeError(
                f"{target}: non-positive Platt slope."
            )
        calibrated_evaluation = apply_platt(
            raw_evaluation,
            float(platt["intercept"]),
            float(platt["slope"]),
            eps,
        )
        calibrated_test = apply_platt(
            raw_test,
            float(platt["intercept"]),
            float(platt["slope"]),
            eps,
        )

        calibrator_rows.append(
            {
                "target": target,
                "feature_set": "booking_strict",
                "model": "lightgbm",
                "population": "singleton_primary",
                "method": "platt_sigmoid",
                "fit_n": int(len(y_calibrator)),
                "fit_event_n": int(np.sum(y_calibrator == 1)),
                "fit_prevalence": float(np.mean(y_calibrator)),
                "intercept": float(platt["intercept"]),
                "slope": float(platt["slope"]),
                "optimizer_success": bool(
                    platt["optimizer_success"]
                ),
                "optimizer_method": str(
                    platt["optimizer_method"]
                ),
                "optimizer_message": str(
                    platt["optimizer_message"]
                ),
            }
        )

        care_evaluation = (
            frame_2022.loc[
                evaluation_mask,
                "prenatal_care_month",
            ].to_numpy(dtype=np.float64)
        )
        care_test = (
            frame_2023.loc[
                test_mask,
                "prenatal_care_month",
            ].to_numpy(dtype=np.float64)
        )
        early_evaluation = early_entry_mask(
            care_evaluation,
            int(config["cohorts"]["early_entry_min_month"]),
            int(config["cohorts"]["early_entry_max_month"]),
        )
        early_test = early_entry_mask(
            care_test,
            int(config["cohorts"]["early_entry_min_month"]),
            int(config["cohorts"]["early_entry_max_month"]),
        )

        evaluation_payloads = [
            (
                "calibration_evaluation_2022",
                y_evaluation,
                raw_evaluation,
                calibrated_evaluation,
                early_evaluation,
                care_evaluation,
            ),
            (
                "temporal_test_2023",
                y_test,
                raw_test,
                calibrated_test,
                early_test,
                care_test,
            ),
        ]

        for (
            evaluation_name,
            y_values,
            raw_values,
            calibrated_values,
            early_values,
            care_values,
        ) in evaluation_payloads:
            point_rows, _ = fixed_budget_rows(
                population="singleton_primary",
                evaluation=evaluation_name,
                target=target,
                y=y_values,
                probabilities=raw_values,
                early_mask=early_values,
                budget_fractions=budget_fractions,
            )
            singleton_results.extend(point_rows)
            singleton_comparisons.extend(
                compare_early_full(point_rows)
            )

            (
                metric_rows,
                comparison_rows,
                curve_rows,
            ) = calibration_metrics_by_scenario(
                target=target,
                evaluation=evaluation_name,
                y=y_values,
                raw_probabilities=raw_values,
                calibrated_probabilities=calibrated_values,
                early_mask=early_values,
                curve_bins=curve_bins,
                eps=eps,
                optimizer_max_iter=optimizer_max_iter,
            )
            calibration_metrics_rows.extend(metric_rows)
            calibration_comparison_rows.extend(comparison_rows)
            calibration_curve_rows.extend(curve_rows)

            # Positive-slope Platt scaling must preserve ranking and top-B.
            calibrated_point_rows, _ = fixed_budget_rows(
                population="singleton_primary",
                evaluation=evaluation_name,
                target=target,
                y=y_values,
                probabilities=calibrated_values,
                early_mask=early_values,
                budget_fractions=budget_fractions,
            )
            raw_comparison = compare_early_full(point_rows)
            calibrated_comparison = compare_early_full(
                calibrated_point_rows
            )
            for raw_row, calibrated_row in zip(
                raw_comparison,
                calibrated_comparison,
            ):
                if (
                    raw_row["early_true_positive_n"]
                    != calibrated_row["early_true_positive_n"]
                    or raw_row["full_true_positive_n"]
                    != calibrated_row["full_true_positive_n"]
                ):
                    raise AssertionError(
                        f"{target}/{evaluation_name}: "
                        "Platt scaling changed top-B ranking."
                    )

            if (
                bool(config["bootstrap"]["enabled"])
                and evaluation_name
                in set(config["bootstrap"]["evaluations"])
            ):
                replicate_rows, summary_rows = (
                    paired_full_size_bootstrap(
                        target=target,
                        evaluation=evaluation_name,
                        y=y_values,
                        probabilities=raw_values,
                        early_mask=early_values,
                        budget_fractions=budget_fractions,
                        replicates=int(
                            config["bootstrap"]["replicates"]
                        ),
                        seed=(
                            int(config["bootstrap"]["seed"])
                            + target_index * 10_000
                            + (
                                1_000
                                if evaluation_name
                                == "temporal_test_2023"
                                else 0
                            )
                        ),
                        confidence_level=float(
                            config["bootstrap"]["confidence_level"]
                        ),
                        progress_every=int(
                            config["bootstrap"]["progress_every"]
                        ),
                    )
                )
                bootstrap_replicates_rows.extend(replicate_rows)
                bootstrap_summary_rows.extend(summary_rows)

            if (
                evaluation_name
                == str(config["mechanism"]["evaluation"])
            ):
                (
                    target_mechanism_rows,
                    target_mechanism_comparisons,
                ) = mechanism_rows(
                    target=target,
                    evaluation=evaluation_name,
                    y=y_values,
                    probabilities=raw_values,
                    care_month=care_values,
                    budget_fraction=float(
                        config["mechanism"][
                            "budget_fraction_of_full_population"
                        ]
                    ),
                )
                mechanism_table_rows.extend(
                    target_mechanism_rows
                )
                mechanism_comparison_rows.extend(
                    target_mechanism_comparisons
                )

        if bool(config["output"].get("save_models", True)):
            model_path = (
                models_directory
                / f"singleton__{target}__booking_strict__lightgbm_platt.joblib"
            )
            joblib.dump(
                {
                    "population": "U.S.-resident singleton births",
                    "target": target,
                    "feature_set": "booking_strict",
                    "features": features,
                    "preprocessor": preprocessor,
                    "model": model,
                    "calibrator": {
                        "method": "platt_sigmoid",
                        "intercept": float(platt["intercept"]),
                        "slope": float(platt["slope"]),
                        "clip_eps": eps,
                    },
                },
                model_path,
                compress=3,
            )
            saved_models.append(str(model_path.relative_to(output)))

        if bool(config["output"].get("save_predictions", False)):
            prediction_path = (
                predictions_directory
                / f"singleton__{target}__cdc2023.npz"
            )
            np.savez_compressed(
                prediction_path,
                y=y_test.astype(np.uint8),
                care_month=care_test.astype(np.float32),
                raw_probability=raw_test.astype(np.float32),
                calibrated_probability=calibrated_test.astype(
                    np.float32
                ),
            )
            saved_predictions.append(
                str(prediction_path.relative_to(output))
            )

        print(
            f"Singleton {target}: "
            f"elapsed={time.time() - fit_started:.1f}s, "
            f"features={transformed_feature_n}, "
            f"Platt={platt['intercept']:+.4f}"
            f"+{platt['slope']:.4f}*logit(p)"
        )

        del (
            preprocessor,
            model,
            raw_calibrator,
            raw_evaluation,
            calibrated_evaluation,
            raw_test,
            calibrated_test,
        )
        gc.collect()

    # ---------------------------------------------------------------
    # Corrected all-birth sensitivity: fixed model, equal absolute
    # budget, LightGBM, booking_strict, 5% and 10%.
    # ---------------------------------------------------------------
    for target in TARGETS:
        print(f"\n===== ALL-BIRTH SENSITIVITY: {target} =====")
        known_2022 = (
            frame_2022[target].notna().to_numpy(dtype=bool)
        )
        known_2023 = (
            frame_2023[target].notna().to_numpy(dtype=bool)
        )
        train_mask = (
            all_births_2022
            & known_2022
            & train_split
        )
        test_mask = all_births_2023 & known_2023
        y_train = (
            frame_2022.loc[train_mask, target]
            .to_numpy(dtype=np.int8)
        )
        y_test = (
            frame_2023.loc[test_mask, target]
            .to_numpy(dtype=np.int8)
        )
        if (
            len(np.unique(y_train)) != 2
            or len(np.unique(y_test)) != 2
        ):
            raise ValueError(
                f"{target}: all-birth split is missing a class."
            )

        fit_started = time.time()
        preprocessor = build_preprocessor(
            features,
            categorical_features,
            config["preprocessing"],
        )
        x_train = consistent_matrix(
            preprocessor.fit_transform(
                frame_2022.loc[train_mask, features]
            )
        )
        model = LGBMClassifier(**parameters)
        model.fit(x_train, y_train)
        transformed_feature_n = int(x_train.shape[1])
        del x_train
        gc.collect()
        all_birth_model_count += 1

        x_test = consistent_matrix(
            preprocessor.transform(
                frame_2023.loc[test_mask, features]
            )
        )
        probabilities = np.asarray(
            model.predict_proba(x_test)[:, 1],
            dtype=np.float64,
        )
        del x_test
        gc.collect()
        temporal_prediction_pass_count += 1

        care_test = (
            frame_2023.loc[
                test_mask,
                "prenatal_care_month",
            ].to_numpy(dtype=np.float64)
        )
        early_test = early_entry_mask(
            care_test,
            int(config["cohorts"]["early_entry_min_month"]),
            int(config["cohorts"]["early_entry_max_month"]),
        )
        point_rows, _ = fixed_budget_rows(
            population="all_births_sensitivity",
            evaluation="temporal_test_2023",
            target=target,
            y=y_test,
            probabilities=probabilities,
            early_mask=early_test,
            budget_fractions=budget_fractions,
        )
        all_birth_results.extend(point_rows)
        all_birth_comparisons.extend(
            compare_early_full(point_rows)
        )

        if bool(config["output"].get("save_models", True)):
            model_path = (
                models_directory
                / f"all_births__{target}__booking_strict__lightgbm.joblib"
            )
            joblib.dump(
                {
                    "population": (
                        "U.S.-resident births, all pluralities"
                    ),
                    "target": target,
                    "feature_set": "booking_strict",
                    "features": features,
                    "preprocessor": preprocessor,
                    "model": model,
                    "calibrator": None,
                },
                model_path,
                compress=3,
            )
            saved_models.append(str(model_path.relative_to(output)))

        print(
            f"All births {target}: "
            f"elapsed={time.time() - fit_started:.1f}s, "
            f"features={transformed_feature_n}"
        )
        del preprocessor, model, probabilities
        gc.collect()

    # ---------------------------------------------------------------
    # Assertions and output
    # ---------------------------------------------------------------
    expected_singleton = int(
        config["execution"]["expected_singleton_model_count"]
    )
    expected_all_birth = int(
        config["execution"]["expected_all_birth_model_count"]
    )
    expected_total = int(
        config["execution"]["expected_total_model_count"]
    )
    if singleton_model_count != expected_singleton:
        raise AssertionError(
            f"Singleton model count {singleton_model_count} "
            f"!= {expected_singleton}"
        )
    if all_birth_model_count != expected_all_birth:
        raise AssertionError(
            f"All-birth model count {all_birth_model_count} "
            f"!= {expected_all_birth}"
        )
    if singleton_model_count + all_birth_model_count != expected_total:
        raise AssertionError("Unexpected total model count.")
    if temporal_prediction_pass_count != expected_total:
        raise AssertionError(
            "Expected one CDC 2023 prediction pass per fitted model."
        )
    if len(calibrator_rows) != len(TARGETS):
        raise AssertionError("Expected three singleton calibrators.")

    expected_singleton_result_rows = (
        len(TARGETS) * 2 * len(budget_fractions) * 2
    )
    if len(singleton_results) != expected_singleton_result_rows:
        raise AssertionError(
            f"Singleton result rows {len(singleton_results)} "
            f"!= {expected_singleton_result_rows}"
        )
    expected_all_birth_rows = (
        len(TARGETS) * len(budget_fractions) * 2
    )
    if len(all_birth_results) != expected_all_birth_rows:
        raise AssertionError(
            f"All-birth result rows {len(all_birth_results)} "
            f"!= {expected_all_birth_rows}"
        )
    if len(mechanism_table_rows) != len(TARGETS) * 8:
        raise AssertionError("Unexpected mechanism-table row count.")
    if len(mechanism_comparison_rows) != len(TARGETS) * 4:
        raise AssertionError(
            "Unexpected mechanism-comparison row count."
        )

    bootstrap_evaluation_count = len(
        config["bootstrap"]["evaluations"]
    )
    expected_bootstrap_replicate_rows = (
        len(TARGETS)
        * bootstrap_evaluation_count
        * int(config["bootstrap"]["replicates"])
        * len(budget_fractions)
    )
    expected_bootstrap_summary_rows = (
        len(TARGETS)
        * bootstrap_evaluation_count
        * len(budget_fractions)
    )
    if len(bootstrap_replicates_rows) != expected_bootstrap_replicate_rows:
        raise AssertionError(
            f"Bootstrap replicate rows {len(bootstrap_replicates_rows)} "
            f"!= {expected_bootstrap_replicate_rows}"
        )
    if len(bootstrap_summary_rows) != expected_bootstrap_summary_rows:
        raise AssertionError(
            f"Bootstrap summary rows {len(bootstrap_summary_rows)} "
            f"!= {expected_bootstrap_summary_rows}"
        )

    expected_calibration_metric_rows = (
        len(TARGETS) * 2 * 2 * 2
    )
    expected_calibration_comparison_rows = (
        len(TARGETS) * 2 * 2
    )
    expected_reliability_curve_rows = (
        len(TARGETS) * 2 * 2 * 2 * curve_bins
    )
    if len(calibration_metrics_rows) != expected_calibration_metric_rows:
        raise AssertionError(
            f"Calibration metric rows {len(calibration_metrics_rows)} "
            f"!= {expected_calibration_metric_rows}"
        )
    if (
        len(calibration_comparison_rows)
        != expected_calibration_comparison_rows
    ):
        raise AssertionError(
            "Unexpected calibration comparison row count."
        )
    if len(calibration_curve_rows) != expected_reliability_curve_rows:
        raise AssertionError(
            f"Reliability curve rows {len(calibration_curve_rows)} "
            f"!= {expected_reliability_curve_rows}"
        )

    atomic_csv(
        output / "singleton_fixed_budget_results.csv",
        singleton_results,
    )
    atomic_csv(
        output / "singleton_fixed_budget_comparisons.csv",
        singleton_comparisons,
    )
    atomic_csv(
        output / "paired_bootstrap_replicates.csv",
        bootstrap_replicates_rows,
    )
    atomic_csv(
        output / "paired_bootstrap_summary.csv",
        bootstrap_summary_rows,
    )
    atomic_csv(
        output / "mechanism_table_10pct.csv",
        mechanism_table_rows,
    )
    atomic_csv(
        output / "mechanism_table_10pct_comparisons.csv",
        mechanism_comparison_rows,
    )
    atomic_csv(
        output / "lightgbm_calibration_metrics.csv",
        calibration_metrics_rows,
    )
    atomic_csv(
        output / "lightgbm_raw_vs_calibrated.csv",
        calibration_comparison_rows,
    )
    atomic_csv(
        output / "lightgbm_reliability_curve_points.csv",
        calibration_curve_rows,
    )
    atomic_csv(
        output / "lightgbm_platt_calibrators.csv",
        calibrator_rows,
    )
    atomic_csv(
        output / "all_births_fixed_budget_results.csv",
        all_birth_results,
    )
    atomic_csv(
        output / "all_births_fixed_budget_comparisons.csv",
        all_birth_comparisons,
    )
    atomic_text(
        output / "article_revision_summary.md",
        article_revision_summary(
            singleton_comparisons,
            bootstrap_summary_rows,
            mechanism_comparison_rows,
            calibration_comparison_rows,
            all_birth_comparisons,
        ),
    )

    manifest: dict[str, Any] = {
        "status": "PASS",
        "elapsed_seconds": float(time.time() - started),
        "clearml_task_id": task.id if task is not None else None,
        "clearml_dataset_id": args.dataset_id,
        "clearml_dataset_root": str(dataset_root),
        "standalone_remote_execution": True,
        "git_clone_required": False,
        "git": git_information(),
        "inputs": {
            "2022": {
                "path": str(cdc2022_path),
                "rows": int(len(frame_2022)),
                "sha256": observed_input_hashes["2022"],
            },
            "2023": {
                "path": str(cdc2023_path),
                "rows": int(len(frame_2023)),
                "sha256": observed_input_hashes["2023"],
            },
        },
        "split": {
            "seed": split_seed,
            "calibration_fraction": calibration_fraction,
            "partition_seed": partition_seed,
            "calibrator_fit_fraction_within_calibration_pool": fit_fraction,
            "assigned_before_target_filters": True,
            "train_rows_all_2022": int(np.sum(train_split)),
            "calibrator_fit_rows_all_2022": int(
                np.sum(calibrator_split)
            ),
            "calibration_evaluation_rows_all_2022": int(
                np.sum(calibration_evaluation_split)
            ),
            "subsplits_disjoint": True,
        },
        "design": {
            "targets": list(TARGETS),
            "feature_sets": ["booking_strict"],
            "model": "lightgbm",
            "model_device_type": "cpu",
            "primary_population": (
                "U.S.-resident singleton births"
            ),
            "sensitivity_population": (
                "U.S.-resident births, all pluralities"
            ),
            "early_entry_definition": (
                "prenatal_care_month in months 1 through 3"
            ),
            "fixed_absolute_budget_fractions": budget_fractions,
            "single_fixed_model_per_target_population": True,
            "cohort_specific_refitting": False,
            "bootstrap": {
                "method": (
                    "paired full-size record-level nonparametric bootstrap"
                ),
                "replicates": int(
                    config["bootstrap"]["replicates"]
                ),
                "evaluations": list(
                    config["bootstrap"]["evaluations"]
                ),
                "same_resampled_records_for_early_and_full": True,
                "top_b_recomputed_in_each_replicate": True,
            },
            "mechanism_budget_fraction": float(
                config["mechanism"][
                    "budget_fraction_of_full_population"
                ]
            ),
            "calibration_method": "platt_sigmoid",
            "calibrator_fit_on_cdc2022_only": True,
            "cdc2023_role": "temporal_test",
            "cdc2023_used_for_model_training": False,
            "cdc2023_used_for_calibrator_fit": False,
            "cdc2023_used_for_threshold_selection": False,
            "class_weighting": False,
            "oversampling": False,
            "hyperparameter_optimization": False,
            "all_birth_dependency_caveat": (
                "Multiple infant records may be dependent; "
                "pregnancy-level ID is unavailable."
            ),
        },
        "counts": {
            "singleton_model_count": singleton_model_count,
            "all_birth_model_count": all_birth_model_count,
            "total_model_count": (
                singleton_model_count + all_birth_model_count
            ),
            "calibrator_count": len(calibrator_rows),
            "temporal_prediction_pass_count": (
                temporal_prediction_pass_count
            ),
        },
        "outputs": {
            "singleton_fixed_budget_rows": len(singleton_results),
            "singleton_comparison_rows": len(singleton_comparisons),
            "bootstrap_replicate_rows": len(
                bootstrap_replicates_rows
            ),
            "bootstrap_summary_rows": len(bootstrap_summary_rows),
            "mechanism_rows": len(mechanism_table_rows),
            "mechanism_comparison_rows": len(
                mechanism_comparison_rows
            ),
            "calibration_metric_rows": len(
                calibration_metrics_rows
            ),
            "calibration_comparison_rows": len(
                calibration_comparison_rows
            ),
            "reliability_curve_rows": len(
                calibration_curve_rows
            ),
            "all_birth_result_rows": len(all_birth_results),
            "all_birth_comparison_rows": len(
                all_birth_comparisons
            ),
            "saved_models": saved_models,
            "saved_predictions": saved_predictions,
        },
    }
    manifest["output_sha256"] = output_hashes(output)
    atomic_json(output / "run_manifest.json", manifest)

    if task is not None:
        report_to_clearml(
            task,
            singleton_results,
            singleton_comparisons,
            bootstrap_summary_rows,
            mechanism_comparison_rows,
            calibration_metrics_rows,
            calibration_comparison_rows,
            calibration_curve_rows,
            all_birth_results,
            all_birth_comparisons,
        )

        artifact_filenames = (
            "singleton_fixed_budget_results.csv",
            "singleton_fixed_budget_comparisons.csv",
            "paired_bootstrap_replicates.csv",
            "paired_bootstrap_summary.csv",
            "mechanism_table_10pct.csv",
            "mechanism_table_10pct_comparisons.csv",
            "lightgbm_calibration_metrics.csv",
            "lightgbm_raw_vs_calibrated.csv",
            "lightgbm_reliability_curve_points.csv",
            "lightgbm_platt_calibrators.csv",
            "all_births_fixed_budget_results.csv",
            "all_births_fixed_budget_comparisons.csv",
            "article_revision_summary.md",
            "run_manifest.json",
            "git_info.json",
            "package_versions.txt",
            "commands.txt",
            "experiment_config.locked.yaml",
        )
        for filename in artifact_filenames:
            task.upload_artifact(
                name=filename,
                artifact_object=str(output / filename),
                wait_on_upload=True,
            )

        task.upload_artifact(
            name="reproducibility_bundle",
            artifact_object=str(output / "reproducibility"),
            wait_on_upload=True,
        )

        if (
            bool(config["output"].get("save_models", True))
            and bool(
                config["output"].get(
                    "upload_models_to_clearml",
                    True,
                )
            )
        ):
            task.upload_artifact(
                name="article_revision_models",
                artifact_object=str(models_directory),
                wait_on_upload=True,
            )

        if (
            bool(config["output"].get("save_predictions", False))
            and bool(
                config["output"].get(
                    "upload_predictions_to_clearml",
                    False,
                )
            )
            and predictions_directory.exists()
        ):
            task.upload_artifact(
                name="article_revision_predictions",
                artifact_object=str(predictions_directory),
                wait_on_upload=True,
            )

        task.get_logger().report_single_value(
            "total_model_count",
            float(singleton_model_count + all_birth_model_count),
        )
        task.get_logger().report_single_value(
            "bootstrap_replicates_per_target_evaluation",
            float(config["bootstrap"]["replicates"]),
        )
        task.get_logger().report_single_value(
            "runtime_hours",
            float((time.time() - started) / 3600.0),
        )
        task.close()

    print("\n===== COMPLETE =====")
    print(f"Results: {output}")
    print(f"Singleton models: {singleton_model_count}")
    print(f"All-birth models: {all_birth_model_count}")
    print(f"Platt calibrators: {len(calibrator_rows)}")
    print(
        "Bootstrap replicate rows: "
        f"{len(bootstrap_replicates_rows)}"
    )
    print(
        "Bootstrap summary rows: "
        f"{len(bootstrap_summary_rows)}"
    )
    print(f"Mechanism rows: {len(mechanism_table_rows)}")
    print(
        "Calibration metric rows: "
        f"{len(calibration_metrics_rows)}"
    )
    print(
        "All-birth fixed-budget rows: "
        f"{len(all_birth_results)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
