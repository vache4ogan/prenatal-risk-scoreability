#!/usr/bin/env python3
"""Canonical TAE 2026 evaluation for CDC Natality 2022 -> 2023.

Scientific design
-----------------
For each of three targets, this script:

1. Uses U.S.-resident singleton births with the target known.
2. Fits exactly ONE LightGBM on CDC 2022 using the frozen 9-feature
   ``landmark_strict`` set.
3. Selects common score thresholds ONLY on the CDC 2022 calibration pool.
4. Generates CDC 2023 probabilities exactly ONCE per target.
5. Applies three nested availability scenarios to that one score vector:
      early_entry             : prenatal_care_month 1..3
      any_prenatal_care       : prenatal_care_month 1..10
      all_record_upper_bound  : every target-specific primary-population row
6. Evaluates three protocols:
      cohort_specific_top_fraction : same fraction q of each eligible cohort
      fixed_absolute_budget        : same absolute B = round(q * N_all)
      fixed_calibration_threshold  : same CDC-2022-derived threshold
7. Keeps the full target-specific event denominator unchanged across scenarios.
8. Runs paired, full-size record-level bootstrap on CDC 2023.
9. Produces an exact same-model decomposition of the naive top-q contrast into
   an availability component and an additional-capacity component.

No cohort-specific model refitting occurs. CDC 2023 is temporal test only and
is never used for training, feature selection, HPO, or threshold selection.

ClearML execution
-----------------
The default remote queue is locked to the user-specified GPU-agent queue ID
``6be0e69f7fab49d48bc305ef1fb03a6a``. LightGBM itself remains CPU-only by
intent for reproducibility with the validated historical pipeline; routing to
a GPU-equipped ClearML agent does not imply GPU LightGBM execution.

The file is standalone for remote ClearML execution: no project-local Python
modules are imported.
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Locked design
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()

DEFAULT_PROJECT_NAME = "pershin-medailab/Vache_Oganisyan/CDC Natality Audit"
DEFAULT_DATASET_ID = "062ba26c0ca24cef99549c2a2ab34e65"
DEFAULT_QUEUE = "6be0e69f7fab49d48bc305ef1fb03a6a"
DEFAULT_TASK_NAME = "TAE 2026 Canonical: NO-PRIORTERM Ablation"

TARGETS = ("target_preterm", "target_nicu", "target_lbw")
FEATURE_SET_NAME = "landmark_strict"
SCENARIOS = (
    "early_entry",
    "any_prenatal_care",
    "all_record_upper_bound",
)
PROTOCOLS = (
    "cohort_specific_top_fraction",
    "fixed_absolute_budget",
    "fixed_calibration_threshold",
)
PAIRWISE_COMPARISONS = (
    ("early_entry", "any_prenatal_care"),
    ("any_prenatal_care", "all_record_upper_bound"),
    ("early_entry", "all_record_upper_bound"),
)
DECOMPOSITION_COMPARISONS = (
    ("early_entry", "all_record_upper_bound"),
)

LANDMARK_STRICT_FEATURES = [
    "mother_race",
    "mother_height_inches",
    "mother_bmi",
    "mother_weight_pre",
    "prior_live_births",
    "prior_dead_births",
    #"prior_terminations",
    "diab_pre",
    "hyper_pre",
]

LEGACY_BOOKING_STRICT_12 = [
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
]

EXCLUDED_FROM_LANDMARK_STRICT = [
    "mother_age",
    "marital_status",
    "mother_educ",
]

FORBIDDEN_MODEL_FEATURES = {
    "prenatal_care_month",
    "prenatal_visits",
    "weight_gain",
    "diab_gest",
    "hyper_gest",
    "eclampsia",
    "delivery_method",
    "delivery_method_binary",
    "gestation_combined_weeks",
    "gestation_oe_weeks",
    "gestation_oe_recode3",
    "birth_weight_g",
    "admit_nicu",
    "an_vent",
    "an_seiz",
    "child_sex",
    "target_preterm",
    "target_nicu",
    "target_lbw",
    "target_gdm",
    "birth_year",
    "birth_month",
    "residence_status",
    "is_us_resident",
    "plurality",
    "is_singleton",
}

TAGS = [
    "tae-2026",
    "cdc-natality",
    "temporal-test",
    "fixed-model",
    "selective-availability",
    "landmark-strict",
    "cohort-specific-top-q",
    "fixed-absolute-budget",
    "fixed-calibration-threshold",
    "paired-bootstrap",
    "full-size-bootstrap",
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
)

# LightGBM is imported after remote hand-off, so add it explicitly.
CLEARML_EXTRA_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("lightgbm", ">=4.3,<5"),
)

DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "cdc2022": "data/cdc_temporal_harmonized/cdc_natality_2022_harmonized.csv",
        "cdc2023": "data/cdc_temporal_harmonized/cdc_natality_2023_harmonized.csv",
        "expected_rows": {"2022": 3_676_029, "2023": 3_605_081},
        "verify_input_hashes": True,
        "expected_sha256": {
            "2022": "c5f1ab62b0e9ac63795e75f6075d2523a3f4c5dc3e95c7aa1f7df6763205a59c",
            "2023": "82b38b24d2f795a4a5233ce3a2fcb43b70789339306ba9044bde56f50bb270a8",
        },
    },
    "output": {
        "directory": "results/tae_2026_canonical",
        "overwrite": False,
        "save_models": True,
        "upload_models_to_clearml": True,
        "save_predictions": False,
        "upload_predictions_to_clearml": False,
        "save_bootstrap_replicates": True,
    },
    "split": {
        "seed": 2026,
        "calibration_fraction": 0.20,
    },
    "cohorts": {
        "primary_population": "is_us_resident == 1 and is_singleton == 1",
        "early_entry_min_month": 1,
        "early_entry_max_month": 3,
        "any_care_min_month": 1,
        "any_care_max_month": 10,
        "no_care_value": 0,
        "unknown_care": "missing",
    },
    "targets": list(TARGETS),
    "feature_set": {
        "name": FEATURE_SET_NAME,
        "features": list(LANDMARK_STRICT_FEATURES),
        "n_features": 8,
    },
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
        "categorical_features": ["mother_race"],
        "numeric_imputation": "median",
        "categorical_imputation": "most_frequent",
        "standardize_numeric": True,
    },
    "evaluation": {
        "fractions": [0.05, 0.10],
        "scenarios": list(SCENARIOS),
        "protocols": list(PROTOCOLS),
        "threshold_source": "cdc2022_calibration_all_record",
        "threshold_comparator": ">=",
        "tie_breaker_top_b": "original_row_position_ascending",
    },
    "bootstrap": {
        "enabled": True,
        "replicates": 500,
        "seed": 2027,
        "confidence_level": 0.95,
        "full_size": True,
        "paired": True,
        "progress_every": 25,
    },
    "execution": {
        "expected_primary_model_count": 3,
        "expected_temporal_prediction_pass_count": 3,
        "clearml_default_queue": DEFAULT_QUEUE,
        "lightgbm_device_type": "cpu",
    },
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def to_plain_python(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_plain_python(item) for key, item in value.items()}
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


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


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
        print(f"Config file not present; using embedded defaults: {candidate}")
        return config
    payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("YAML config root must be a mapping.")
    return deep_update(config, payload)


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
# ClearML Dataset and data loading
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
    candidates = [dataset_root / configured_path, dataset_root / configured_path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = list(dataset_root.rglob(configured_path.name))
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
        raise RuntimeError(
            f"Ambiguous dataset file {configured!r}; matches: "
            f"{[str(path) for path in matches[:10]]}"
        )
    raise FileNotFoundError(
        f"Dataset file {configured!r} was not found under {dataset_root}"
    )


def required_columns() -> list[str]:
    return sorted(
        {
            "is_us_resident",
            "is_singleton",
            "prenatal_care_month",
            *TARGETS,
            *LANDMARK_STRICT_FEATURES,
        }
    )


def read_cdc_frame(
    path: Path,
    columns: list[str],
    expected_rows: Optional[int],
    label: str,
) -> pd.DataFrame:
    print(f"Reading {label}: {path}")
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    if expected_rows is not None and len(frame) != int(expected_rows):
        raise ValueError(f"{label} row count {len(frame)} != expected {expected_rows}")
    print(f"{label}: {len(frame):,} rows, {len(frame.columns)} columns loaded")
    return frame


def validate_binary_column(
    frame: pd.DataFrame,
    column: str,
    label: str,
    *,
    allow_missing: bool,
) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    if not allow_missing and values.isna().any():
        raise ValueError(f"{label}.{column} contains missing/non-numeric values.")
    unique = set(values.dropna().unique().tolist())
    if not unique.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"{label}.{column} is not binary: {sorted(unique)[:20]}")
    frame[column] = values


def validate_care_month(values: np.ndarray, label: str) -> None:
    care = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(care)
    finite_values = care[finite]
    if finite_values.size == 0:
        raise ValueError(f"{label}: prenatal_care_month has no finite values.")
    if np.any(finite_values < 0) or np.any(finite_values > 10):
        bad = finite_values[(finite_values < 0) | (finite_values > 10)][:20]
        raise ValueError(f"{label}: care month outside 0..10: {bad.tolist()}")
    if not np.allclose(finite_values, np.round(finite_values), atol=0.0, rtol=0.0):
        bad = finite_values[finite_values != np.round(finite_values)][:20]
        raise ValueError(f"{label}: non-integer care month values: {bad.tolist()}")


# ---------------------------------------------------------------------------
# Configuration and feature audit invariants
# ---------------------------------------------------------------------------


def lightgbm_parameters(config: dict[str, Any]) -> dict[str, Any]:
    parameters = dict(config["model"]["parameters"])
    # Keep the validated CPU LightGBM behavior even on a GPU-equipped agent.
    for key in ("device", "gpu_platform_id", "gpu_device_id", "num_gpu"):
        parameters.pop(key, None)
    parameters["device_type"] = "cpu"
    parameters["class_weight"] = None
    parameters["scale_pos_weight"] = 1.0
    return parameters


def validate_config(config: dict[str, Any]) -> None:
    if tuple(map(str, config.get("targets", []))) != TARGETS:
        raise ValueError(f"targets must be exactly {list(TARGETS)}")

    feature_cfg = config.get("feature_set", {})
    if str(feature_cfg.get("name")) != FEATURE_SET_NAME:
        raise ValueError(f"feature_set.name must be {FEATURE_SET_NAME!r}")
    features = [str(value) for value in feature_cfg.get("features", [])]
    if features != LANDMARK_STRICT_FEATURES:
        raise ValueError(
            "feature_set.features must match the frozen landmark_strict list exactly."
        )
    if int(feature_cfg.get("n_features", -1)) != len(LANDMARK_STRICT_FEATURES):
        raise ValueError("feature_set.n_features must equal 9.")
    overlap = sorted(set(features) & FORBIDDEN_MODEL_FEATURES)
    if overlap:
        raise ValueError(f"Forbidden features entered landmark_strict: {overlap}")
    if set(EXCLUDED_FROM_LANDMARK_STRICT) & set(features):
        raise ValueError("mother_age/marital_status/mother_educ must remain excluded.")

    primary = str(config["cohorts"]["primary_population"]).replace(" ", "")
    if primary != "is_us_resident==1andis_singleton==1":
        raise ValueError("Primary population must be U.S.-resident singleton births.")
    expected_cohort_values = {
        "early_entry_min_month": 1,
        "early_entry_max_month": 3,
        "any_care_min_month": 1,
        "any_care_max_month": 10,
        "no_care_value": 0,
    }
    for key, expected in expected_cohort_values.items():
        if int(config["cohorts"].get(key, -999)) != expected:
            raise ValueError(f"cohorts.{key} must equal {expected}")
    if str(config["cohorts"].get("unknown_care")) != "missing":
        raise ValueError("cohorts.unknown_care must be 'missing'.")

    fractions = [float(v) for v in config["evaluation"]["fractions"]]
    if fractions != [0.05, 0.10]:
        raise ValueError("evaluation.fractions must be exactly [0.05, 0.10].")
    if tuple(config["evaluation"]["scenarios"]) != SCENARIOS:
        raise ValueError(f"evaluation.scenarios must be exactly {list(SCENARIOS)}")
    if tuple(config["evaluation"]["protocols"]) != PROTOCOLS:
        raise ValueError(f"evaluation.protocols must be exactly {list(PROTOCOLS)}")
    if str(config["evaluation"].get("threshold_source")) != "cdc2022_calibration_all_record":
        raise ValueError("Threshold source must be CDC 2022 calibration all-record population.")
    if str(config["evaluation"].get("threshold_comparator")) != ">=":
        raise ValueError("Threshold comparator must be >=.")

    calibration_fraction = float(config["split"]["calibration_fraction"])
    if calibration_fraction != 0.20:
        raise ValueError("split.calibration_fraction is locked to 0.20.")
    if int(config["split"]["seed"]) != 2026:
        raise ValueError("split.seed is locked to 2026.")

    categorical = [str(v) for v in config["preprocessing"]["categorical_features"]]
    if categorical != ["mother_race"]:
        raise ValueError("Only mother_race may be categorical in landmark_strict.")

    if str(config["model"].get("name")) != "lightgbm":
        raise ValueError("Only LightGBM is allowed in the canonical main run.")
    parameters = config["model"]["parameters"]
    if parameters.get("class_weight") is not None:
        raise ValueError("class_weight must remain null.")
    if float(parameters.get("scale_pos_weight", 1.0)) != 1.0:
        raise ValueError("scale_pos_weight must remain 1.0.")
    if str(parameters.get("device_type", "cpu")) != "cpu":
        raise ValueError(
            "Canonical LightGBM is CPU-only; the ClearML task may still run on the GPU agent."
        )

    bootstrap = config["bootstrap"]
    if not bool(bootstrap.get("enabled", False)):
        raise ValueError("Canonical run requires bootstrap.enabled=true.")
    if not bool(bootstrap.get("paired", False)):
        raise ValueError("Bootstrap must be paired.")
    if not bool(bootstrap.get("full_size", False)):
        raise ValueError("Bootstrap must be full-size.")
    if int(bootstrap.get("replicates", 0)) < 100:
        raise ValueError("Bootstrap requires at least 100 replicates.")
    if int(bootstrap.get("seed", -1)) != 2027:
        raise ValueError("bootstrap.seed is locked to 2027.")

    execution = config["execution"]
    if int(execution.get("expected_primary_model_count", -1)) != 3:
        raise ValueError("expected_primary_model_count must equal 3.")
    if int(execution.get("expected_temporal_prediction_pass_count", -1)) != 3:
        raise ValueError("expected_temporal_prediction_pass_count must equal 3.")
    if str(execution.get("clearml_default_queue")) != DEFAULT_QUEUE:
        raise ValueError(
            f"execution.clearml_default_queue must be the locked queue {DEFAULT_QUEUE}."
        )

    expected_hashes = config["data"].get("expected_sha256", {})
    for year in ("2022", "2023"):
        digest = str(expected_hashes.get(year, ""))
        if len(digest) != 64:
            raise ValueError(f"Missing/invalid expected SHA-256 for CDC {year}.")


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
        raise ValueError("No features supplied to preprocessor.")

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


# ---------------------------------------------------------------------------
# Cohorts / availability scenarios
# ---------------------------------------------------------------------------


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
    later = finite & (values >= 4) & (values <= 10)
    no_care = finite & (values == 0)
    unknown = ~finite
    any_care = finite & (values >= 1) & (values <= 10)
    all_record = np.ones(len(values), dtype=bool)

    # Partition and nesting invariants.
    partition_count = (
        early.astype(np.int8)
        + later.astype(np.int8)
        + no_care.astype(np.int8)
        + unknown.astype(np.int8)
    )
    if not np.all(partition_count == 1):
        raise AssertionError("Care-entry groups do not partition the population exactly.")
    if not np.array_equal(any_care, early | later):
        raise AssertionError("any_prenatal_care != early_entry UNION later_entry")
    if not np.all(~early | any_care):
        raise AssertionError("early_entry is not a subset of any_prenatal_care")
    if not np.all(~any_care | all_record):
        raise AssertionError("any_prenatal_care is not a subset of all_record_upper_bound")
    if np.any(any_care & no_care):
        raise AssertionError("no_care entered any_prenatal_care")
    if np.any(any_care & unknown):
        raise AssertionError("unknown entered any_prenatal_care")

    return {
        "early_entry": early,
        "any_prenatal_care": any_care,
        "all_record_upper_bound": all_record,
        "later_entry": later,
        "no_care": no_care,
        "unknown_care": unknown,
    }


def cohort_validation_rows(
    *,
    year: int,
    target: str,
    y: np.ndarray,
    care_month: np.ndarray,
) -> list[dict[str, Any]]:
    masks = availability_masks(care_month)
    order = [
        "early_entry",
        "later_entry",
        "any_prenatal_care",
        "no_care",
        "unknown_care",
        "all_record_upper_bound",
    ]
    rows: list[dict[str, Any]] = []
    for scenario in order:
        mask = masks[scenario]
        rows.append(
            {
                "year": int(year),
                "target": target,
                "scenario": scenario,
                "n_records": int(np.sum(mask)),
                "event_n": int(np.sum(np.asarray(y, dtype=np.int8)[mask] == 1)),
            }
        )

    by_name = {row["scenario"]: row for row in rows}
    if (
        by_name["early_entry"]["n_records"]
        + by_name["later_entry"]["n_records"]
        != by_name["any_prenatal_care"]["n_records"]
    ):
        raise AssertionError("Cohort count identity failed for any-care records.")
    if (
        by_name["early_entry"]["event_n"]
        + by_name["later_entry"]["event_n"]
        != by_name["any_prenatal_care"]["event_n"]
    ):
        raise AssertionError("Cohort count identity failed for any-care events.")
    if (
        by_name["any_prenatal_care"]["n_records"]
        + by_name["no_care"]["n_records"]
        + by_name["unknown_care"]["n_records"]
        != by_name["all_record_upper_bound"]["n_records"]
    ):
        raise AssertionError("All-record record-count partition failed.")
    if (
        by_name["any_prenatal_care"]["event_n"]
        + by_name["no_care"]["event_n"]
        + by_name["unknown_care"]["event_n"]
        != by_name["all_record_upper_bound"]["event_n"]
    ):
        raise AssertionError("All-record event-count partition failed.")
    return rows


# ---------------------------------------------------------------------------
# Ranking, thresholds, metrics, and protocols
# ---------------------------------------------------------------------------


def rank_eligible_indices(
    probabilities: np.ndarray,
    eligible_mask: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    eligible_indices = np.flatnonzero(np.asarray(eligible_mask, dtype=bool))
    eligible_probabilities = probabilities[eligible_indices]
    order = np.lexsort((eligible_indices, -eligible_probabilities))
    return eligible_indices[order]


def threshold_for_capacity(probabilities: np.ndarray, capacity: float) -> float:
    """Return kth-largest score on CDC 2022 all-record calibration population."""
    values = np.asarray(probabilities, dtype=np.float64)
    if not 0.0 < capacity < 1.0:
        raise ValueError(f"Invalid capacity: {capacity}")
    if len(values) == 0:
        raise ValueError("Cannot choose threshold from empty probability vector.")
    selected_n = max(1, min(len(values), int(round(capacity * len(values)))))
    partition_index = len(values) - selected_n
    return float(np.partition(values, partition_index)[partition_index])


def access_metrics(
    *,
    y: np.ndarray,
    eligible_mask: np.ndarray,
    selected_indices: np.ndarray,
) -> dict[str, int | float]:
    y_values = np.asarray(y, dtype=np.int8)
    eligible = np.asarray(eligible_mask, dtype=bool)
    selected = np.asarray(selected_indices, dtype=np.int64)
    eligible_indices = np.flatnonzero(eligible)

    full_population_n = int(len(y_values))
    full_population_event_n = int(np.sum(y_values == 1))
    eligible_n = int(len(eligible_indices))
    eligible_event_n = int(np.sum(y_values[eligible_indices] == 1))
    selected_n = int(len(selected))
    true_positive_n = int(np.sum(y_values[selected] == 1)) if selected_n else 0

    return {
        "full_population_n": full_population_n,
        "full_population_event_n": full_population_event_n,
        "eligible_n": eligible_n,
        "eligible_event_n": eligible_event_n,
        "selected_n": selected_n,
        "true_positive_n": true_positive_n,
        "selection_rate_among_eligible": (
            float(selected_n / eligible_n) if eligible_n else math.nan
        ),
        "precision": float(true_positive_n / selected_n) if selected_n else math.nan,
        "recall_among_eligible": (
            float(true_positive_n / eligible_event_n) if eligible_event_n else math.nan
        ),
        "population_event_capture": (
            float(true_positive_n / full_population_event_n)
            if full_population_event_n
            else math.nan
        ),
        "event_availability_ceiling": (
            float(eligible_event_n / full_population_event_n)
            if full_population_event_n
            else math.nan
        ),
    }


def evaluate_protocols(
    *,
    target: str,
    y: np.ndarray,
    probabilities: np.ndarray,
    care_month: np.ndarray,
    fractions: list[float],
    thresholds: dict[float, float],
) -> tuple[list[dict[str, Any]], dict[tuple[str, float, str], np.ndarray]]:
    y_values = np.asarray(y, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    masks = availability_masks(care_month)
    scenario_masks = {name: masks[name] for name in SCENARIOS}
    rankings = {
        name: rank_eligible_indices(scores, mask)
        for name, mask in scenario_masks.items()
    }

    rows: list[dict[str, Any]] = []
    selections: dict[tuple[str, float, str], np.ndarray] = {}
    n_all = int(len(y_values))

    for q in fractions:
        q = float(q)

        # Protocol A: same fraction of each eligible cohort.
        for scenario in SCENARIOS:
            eligible_mask = scenario_masks[scenario]
            eligible_n = int(np.sum(eligible_mask))
            budget_n = max(1, int(round(q * eligible_n)))
            selected = rankings[scenario][:budget_n]
            if len(selected) != budget_n:
                raise AssertionError("Protocol A failed to select exact cohort-specific budget.")
            selections[("cohort_specific_top_fraction", q, scenario)] = selected
            rows.append(
                {
                    "target": target,
                    "feature_set": FEATURE_SET_NAME,
                    "model": "lightgbm",
                    "evaluation": "temporal_test_2023",
                    "availability_scenario": scenario,
                    "protocol": "cohort_specific_top_fraction",
                    "nominal_fraction": q,
                    "budget_n": budget_n,
                    "threshold": "",
                    "threshold_source_year": "",
                    **access_metrics(
                        y=y_values,
                        eligible_mask=eligible_mask,
                        selected_indices=selected,
                    ),
                }
            )

        # Protocol B: same absolute B based on all-record target population.
        fixed_budget_n = max(1, int(round(q * n_all)))
        for scenario in SCENARIOS:
            eligible_mask = scenario_masks[scenario]
            eligible_n = int(np.sum(eligible_mask))
            if fixed_budget_n > eligible_n:
                raise ValueError(
                    f"{target}/{scenario}: fixed budget {fixed_budget_n:,} exceeds "
                    f"eligible population {eligible_n:,}."
                )
            selected = rankings[scenario][:fixed_budget_n]
            if len(selected) != fixed_budget_n:
                raise AssertionError("Protocol B failed to select exact common budget.")
            selections[("fixed_absolute_budget", q, scenario)] = selected
            rows.append(
                {
                    "target": target,
                    "feature_set": FEATURE_SET_NAME,
                    "model": "lightgbm",
                    "evaluation": "temporal_test_2023",
                    "availability_scenario": scenario,
                    "protocol": "fixed_absolute_budget",
                    "nominal_fraction": q,
                    "budget_n": fixed_budget_n,
                    "threshold": "",
                    "threshold_source_year": "",
                    **access_metrics(
                        y=y_values,
                        eligible_mask=eligible_mask,
                        selected_indices=selected,
                    ),
                }
            )

        # Protocol C: one threshold chosen on CDC 2022 calibration all-record pool.
        threshold = float(thresholds[q])
        for scenario in SCENARIOS:
            eligible_mask = scenario_masks[scenario]
            selected = np.flatnonzero(eligible_mask & (scores >= threshold))
            selections[("fixed_calibration_threshold", q, scenario)] = selected
            rows.append(
                {
                    "target": target,
                    "feature_set": FEATURE_SET_NAME,
                    "model": "lightgbm",
                    "evaluation": "temporal_test_2023",
                    "availability_scenario": scenario,
                    "protocol": "fixed_calibration_threshold",
                    "nominal_fraction": q,
                    "budget_n": "",
                    "threshold": threshold,
                    "threshold_source_year": 2022,
                    **access_metrics(
                        y=y_values,
                        eligible_mask=eligible_mask,
                        selected_indices=selected,
                    ),
                }
            )

    validate_protocol_rows(rows)
    return rows, selections


def validate_protocol_rows(rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    expected = len(SCENARIOS) * len(PROTOCOLS) * 2
    if len(frame) != expected:
        raise AssertionError(f"Expected {expected} protocol rows per target, got {len(frame)}")

    # Full-population denominator must be invariant within target.
    if frame["full_population_n"].nunique() != 1:
        raise AssertionError("full_population_n varies across scenarios/protocols.")
    if frame["full_population_event_n"].nunique() != 1:
        raise AssertionError("full_population_event_n varies across scenarios/protocols.")

    for q in sorted(frame["nominal_fraction"].unique()):
        # Protocol A exact fraction-derived selected count.
        a = frame.loc[
            (frame["protocol"] == "cohort_specific_top_fraction")
            & frame["nominal_fraction"].eq(q)
        ]
        for _, row in a.iterrows():
            expected_n = max(1, int(round(float(q) * int(row["eligible_n"]))))
            if int(row["selected_n"]) != expected_n:
                raise AssertionError("Protocol A selected_n invariant failed.")

        # Protocol B same absolute B across all scenarios.
        b = frame.loc[
            (frame["protocol"] == "fixed_absolute_budget")
            & frame["nominal_fraction"].eq(q)
        ]
        if b["selected_n"].nunique() != 1 or b["budget_n"].nunique() != 1:
            raise AssertionError("Protocol B common-budget invariant failed.")
        if not np.all(b["selected_n"].astype(int) == b["budget_n"].astype(int)):
            raise AssertionError("Protocol B selected_n != budget_n.")

        # Protocol C one identical threshold across scenarios.
        c = frame.loc[
            (frame["protocol"] == "fixed_calibration_threshold")
            & frame["nominal_fraction"].eq(q)
        ]
        if c["threshold"].astype(float).nunique() != 1:
            raise AssertionError("Protocol C common-threshold invariant failed.")

    # Availability ceilings must be nested non-decreasing.
    ceilings = (
        frame.loc[
            (frame["protocol"] == "fixed_absolute_budget")
            & frame["nominal_fraction"].eq(frame["nominal_fraction"].min())
        ]
        .set_index("availability_scenario")["event_availability_ceiling"]
        .to_dict()
    )
    if not (
        float(ceilings["early_entry"])
        <= float(ceilings["any_prenatal_care"])
        <= float(ceilings["all_record_upper_bound"])
    ):
        raise AssertionError("Event availability ceilings are not nested.")
    if not math.isclose(
        float(ceilings["all_record_upper_bound"]), 1.0, rel_tol=0.0, abs_tol=1e-15
    ):
        raise AssertionError("all_record_upper_bound event ceiling must equal 1.0.")


def protocol_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    comparisons: list[dict[str, Any]] = []
    for (target, protocol, q), group in frame.groupby(
        ["target", "protocol", "nominal_fraction"], sort=True
    ):
        by_scenario = {
            str(row["availability_scenario"]): row
            for _, row in group.iterrows()
        }
        if set(by_scenario) != set(SCENARIOS):
            raise AssertionError(f"Missing scenario in {target}/{protocol}/{q}")
        for from_scenario, to_scenario in PAIRWISE_COMPARISONS:
            left = by_scenario[from_scenario]
            right = by_scenario[to_scenario]
            comparisons.append(
                {
                    "target": target,
                    "feature_set": FEATURE_SET_NAME,
                    "model": "lightgbm",
                    "evaluation": "temporal_test_2023",
                    "protocol": protocol,
                    "nominal_fraction": float(q),
                    "from_scenario": from_scenario,
                    "to_scenario": to_scenario,
                    "from_eligible_n": int(left["eligible_n"]),
                    "to_eligible_n": int(right["eligible_n"]),
                    "from_selected_n": int(left["selected_n"]),
                    "to_selected_n": int(right["selected_n"]),
                    "delta_selected_n": int(right["selected_n"] - left["selected_n"]),
                    "from_true_positive_n": int(left["true_positive_n"]),
                    "to_true_positive_n": int(right["true_positive_n"]),
                    "delta_true_positive_n": int(
                        right["true_positive_n"] - left["true_positive_n"]
                    ),
                    "from_precision": float(left["precision"]),
                    "to_precision": float(right["precision"]),
                    "delta_precision": float(right["precision"] - left["precision"]),
                    "from_recall_among_eligible": float(left["recall_among_eligible"]),
                    "to_recall_among_eligible": float(right["recall_among_eligible"]),
                    "delta_recall_among_eligible": float(
                        right["recall_among_eligible"] - left["recall_among_eligible"]
                    ),
                    "from_population_event_capture": float(left["population_event_capture"]),
                    "to_population_event_capture": float(right["population_event_capture"]),
                    "delta_population_event_capture": float(
                        right["population_event_capture"]
                        - left["population_event_capture"]
                    ),
                    "from_event_availability_ceiling": float(
                        left["event_availability_ceiling"]
                    ),
                    "to_event_availability_ceiling": float(
                        right["event_availability_ceiling"]
                    ),
                }
            )
    return comparisons


def effect_decomposition_rows(
    *,
    target: str,
    y: np.ndarray,
    probabilities: np.ndarray,
    care_month: np.ndarray,
    fractions: list[float],
) -> list[dict[str, Any]]:
    y_values = np.asarray(y, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    masks = availability_masks(care_month)
    rankings = {
        scenario: rank_eligible_indices(scores, masks[scenario])
        for scenario in SCENARIOS
    }
    full_events = int(np.sum(y_values == 1))
    if full_events <= 0:
        raise ValueError(f"{target}: no events for decomposition.")

    rows: list[dict[str, Any]] = []
    for q in fractions:
        for from_scenario, to_scenario in DECOMPOSITION_COMPARISONS:
            n_from = int(np.sum(masks[from_scenario]))
            n_to = int(np.sum(masks[to_scenario]))
            b_from = max(1, int(round(float(q) * n_from)))
            b_to = max(1, int(round(float(q) * n_to)))
            if b_to > n_from:
                raise ValueError(
                    f"Decomposition requires B_to <= N_from; got {b_to} > {n_from} "
                    f"for {target}/{from_scenario}->{to_scenario}/q={q}."
                )

            def capture(scenario: str, budget: int) -> tuple[int, float]:
                selected = rankings[scenario][:budget]
                tp = int(np.sum(y_values[selected] == 1))
                return tp, float(tp / full_events)

            tp_from_bfrom, cap_from_bfrom = capture(from_scenario, b_from)
            tp_from_bto, cap_from_bto = capture(from_scenario, b_to)
            tp_to_bto, cap_to_bto = capture(to_scenario, b_to)

            naive_effect = cap_to_bto - cap_from_bfrom
            availability_effect = cap_to_bto - cap_from_bto
            capacity_effect = cap_from_bto - cap_from_bfrom
            identity_error = naive_effect - (availability_effect + capacity_effect)
            if not math.isclose(identity_error, 0.0, rel_tol=0.0, abs_tol=1e-15):
                raise AssertionError(
                    f"Decomposition identity failed: error={identity_error}"
                )

            rows.append(
                {
                    "target": target,
                    "feature_set": FEATURE_SET_NAME,
                    "model": "lightgbm",
                    "evaluation": "temporal_test_2023",
                    "nominal_fraction": float(q),
                    "from_scenario": from_scenario,
                    "to_scenario": to_scenario,
                    "from_cohort_budget_n": b_from,
                    "to_cohort_budget_n": b_to,
                    "from_tp_at_from_budget": tp_from_bfrom,
                    "from_tp_at_to_budget": tp_from_bto,
                    "to_tp_at_to_budget": tp_to_bto,
                    "naive_top_fraction_effect": naive_effect,
                    "availability_effect_at_to_budget": availability_effect,
                    "capacity_effect_within_from_scenario": capacity_effect,
                    "identity_error": identity_error,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def selected_tp_from_cumulative(
    *,
    cumulative_count: np.ndarray,
    cumulative_events: np.ndarray,
    ordered_y: np.ndarray,
    budget: int,
) -> int:
    """Exact top-B TP from one precomputed ranked cumulative scan."""
    if budget <= 0:
        raise ValueError("Budget must be positive.")
    if len(cumulative_count) == 0 or int(cumulative_count[-1]) < budget:
        raise ValueError(
            f"Bootstrap budget={budget:,} exceeds resampled eligible count="
            f"{int(cumulative_count[-1]) if len(cumulative_count) else 0:,}."
        )
    cutoff = int(np.searchsorted(cumulative_count, budget, side="left"))
    previous_count = int(cumulative_count[cutoff - 1]) if cutoff > 0 else 0
    previous_events = int(cumulative_events[cutoff - 1]) if cutoff > 0 else 0
    remaining = int(budget - previous_count)
    return int(previous_events + remaining * int(ordered_y[cutoff]))


def bootstrap_protocol_comparisons(
    *,
    target: str,
    y: np.ndarray,
    probabilities: np.ndarray,
    care_month: np.ndarray,
    fractions: list[float],
    thresholds: dict[float, float],
    point_comparisons: list[dict[str, Any]],
    replicates: int,
    seed: int,
    confidence_level: float,
    progress_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y_values = np.asarray(y, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    n = int(len(y_values))
    if n == 0:
        raise ValueError("Bootstrap population is empty.")

    masks_all = availability_masks(care_month)
    masks = {scenario: masks_all[scenario] for scenario in SCENARIOS}
    scenario_indices = {scenario: np.flatnonzero(mask) for scenario, mask in masks.items()}
    scenario_orders: dict[str, np.ndarray] = {}
    scenario_y_order: dict[str, np.ndarray] = {}
    for scenario in SCENARIOS:
        order = rank_eligible_indices(scores, masks[scenario])
        scenario_orders[scenario] = order
        scenario_y_order[scenario] = y_values[order]

    # Because every scenario ranking is sorted by the same score descending,
    # the fixed-threshold selected set is a prefix of that scenario ranking.
    # Store the original prefix length once; bootstrap selected_n/TP can then be
    # read from the same cumulative arrays used by Protocols A/B.
    threshold_prefix_n = {
        (float(q), scenario): int(
            np.sum(masks[scenario] & (scores >= float(thresholds[float(q)])))
        )
        for q in fractions
        for scenario in SCENARIOS
    }

    point_lookup = {
        (
            str(row["protocol"]),
            float(row["nominal_fraction"]),
            str(row["from_scenario"]),
            str(row["to_scenario"]),
        ): row
        for row in point_comparisons
        if str(row["target"]) == target
    }

    rng = np.random.default_rng(seed)
    replicate_rows: list[dict[str, Any]] = []

    for replicate in range(replicates):
        sampled = rng.integers(0, n, size=n, dtype=np.int32)
        counts = np.bincount(sampled, minlength=n).astype(np.int32, copy=False)
        del sampled

        full_event_n = int(np.sum(counts[y_values == 1], dtype=np.int64))
        if full_event_n <= 0:
            raise RuntimeError(f"{target}/bootstrap replicate {replicate}: no events.")

        # metrics[(protocol, q, scenario)] = (selected_n, tp, capture)
        metrics: dict[tuple[str, float, str], tuple[int, int, float]] = {}

        for scenario in SCENARIOS:
            eligible_indices = scenario_indices[scenario]
            eligible_sample_n = int(np.sum(counts[eligible_indices], dtype=np.int64))
            if eligible_sample_n <= 0:
                raise RuntimeError(
                    f"{target}/bootstrap replicate {replicate}/{scenario}: empty eligible sample."
                )

            order = scenario_orders[scenario]
            ordered_counts = counts[order].astype(np.int64, copy=False)
            ordered_y = scenario_y_order[scenario].astype(np.int64, copy=False)
            cumulative_count = np.cumsum(ordered_counts, dtype=np.int64)
            cumulative_events = np.cumsum(
                ordered_counts * ordered_y,
                dtype=np.int64,
            )

            for q in fractions:
                q = float(q)
                # Protocol A: fraction of resampled eligible count.
                budget_a = max(1, int(round(q * eligible_sample_n)))
                tp_a = selected_tp_from_cumulative(
                    cumulative_count=cumulative_count,
                    cumulative_events=cumulative_events,
                    ordered_y=ordered_y,
                    budget=budget_a,
                )
                metrics[("cohort_specific_top_fraction", q, scenario)] = (
                    budget_a,
                    tp_a,
                    float(tp_a / full_event_n),
                )

                # Protocol B: full-size bootstrap keeps n fixed, so common B is fixed.
                budget_b = max(1, int(round(q * n)))
                tp_b = selected_tp_from_cumulative(
                    cumulative_count=cumulative_count,
                    cumulative_events=cumulative_events,
                    ordered_y=ordered_y,
                    budget=budget_b,
                )
                metrics[("fixed_absolute_budget", q, scenario)] = (
                    budget_b,
                    tp_b,
                    float(tp_b / full_event_n),
                )

                # Protocol C: same fixed threshold. In the score-sorted order,
                # all threshold-selected original rows occupy one prefix.
                prefix_n = threshold_prefix_n[(q, scenario)]
                if prefix_n <= 0:
                    selected_n_c = 0
                    tp_c = 0
                else:
                    selected_n_c = int(cumulative_count[prefix_n - 1])
                    tp_c = int(cumulative_events[prefix_n - 1])
                metrics[("fixed_calibration_threshold", q, scenario)] = (
                    selected_n_c,
                    tp_c,
                    float(tp_c / full_event_n),
                )

            del ordered_counts, ordered_y, cumulative_count, cumulative_events

        for protocol in PROTOCOLS:
            for q in fractions:
                q = float(q)
                for from_scenario, to_scenario in PAIRWISE_COMPARISONS:
                    from_selected, from_tp, from_capture = metrics[
                        (protocol, q, from_scenario)
                    ]
                    to_selected, to_tp, to_capture = metrics[
                        (protocol, q, to_scenario)
                    ]
                    replicate_rows.append(
                        {
                            "target": target,
                            "feature_set": FEATURE_SET_NAME,
                            "model": "lightgbm",
                            "evaluation": "temporal_test_2023",
                            "replicate": int(replicate),
                            "bootstrap_sample_n": n,
                            "bootstrap_event_n": full_event_n,
                            "protocol": protocol,
                            "nominal_fraction": q,
                            "from_scenario": from_scenario,
                            "to_scenario": to_scenario,
                            "from_selected_n": from_selected,
                            "to_selected_n": to_selected,
                            "delta_selected_n": to_selected - from_selected,
                            "from_true_positive_n": from_tp,
                            "to_true_positive_n": to_tp,
                            "delta_true_positive_n": to_tp - from_tp,
                            "from_population_event_capture": from_capture,
                            "to_population_event_capture": to_capture,
                            "delta_population_event_capture": to_capture - from_capture,
                        }
                    )

        del counts
        if progress_every > 0 and (
            (replicate + 1) % progress_every == 0 or replicate + 1 == replicates
        ):
            print(f"Bootstrap {target}: {replicate + 1}/{replicates}")

    frame = pd.DataFrame(replicate_rows)
    alpha = 1.0 - float(confidence_level)
    summary_rows: list[dict[str, Any]] = []
    group_columns = [
        "target",
        "protocol",
        "nominal_fraction",
        "from_scenario",
        "to_scenario",
    ]
    for key, group in frame.groupby(group_columns, sort=True):
        target_value, protocol, q, from_scenario, to_scenario = key
        captures = group["delta_population_event_capture"].to_numpy(dtype=np.float64)
        tps = group["delta_true_positive_n"].to_numpy(dtype=np.float64)
        selected = group["delta_selected_n"].to_numpy(dtype=np.float64)
        point = point_lookup[(protocol, float(q), from_scenario, to_scenario)]
        summary_rows.append(
            {
                "target": target_value,
                "feature_set": FEATURE_SET_NAME,
                "model": "lightgbm",
                "evaluation": "temporal_test_2023",
                "bootstrap_method": "paired full-size record-level nonparametric bootstrap",
                "replicates": int(replicates),
                "confidence_level": float(confidence_level),
                "protocol": protocol,
                "nominal_fraction": float(q),
                "from_scenario": from_scenario,
                "to_scenario": to_scenario,
                "point_delta_population_event_capture": float(
                    point["delta_population_event_capture"]
                ),
                "bootstrap_mean_delta_population_event_capture": float(np.mean(captures)),
                "bootstrap_se_delta_population_event_capture": float(
                    np.std(captures, ddof=1)
                ),
                "ci_lower_delta_population_event_capture": float(
                    np.quantile(captures, alpha / 2.0)
                ),
                "ci_upper_delta_population_event_capture": float(
                    np.quantile(captures, 1.0 - alpha / 2.0)
                ),
                "point_delta_true_positive_n": int(point["delta_true_positive_n"]),
                "bootstrap_mean_delta_true_positive_n": float(np.mean(tps)),
                "ci_lower_delta_true_positive_n": float(
                    np.quantile(tps, alpha / 2.0)
                ),
                "ci_upper_delta_true_positive_n": float(
                    np.quantile(tps, 1.0 - alpha / 2.0)
                ),
                "point_delta_selected_n": int(point["delta_selected_n"]),
                "bootstrap_mean_delta_selected_n": float(np.mean(selected)),
            }
        )

    expected_replicate_rows = (
        replicates * len(PROTOCOLS) * len(fractions) * len(PAIRWISE_COMPARISONS)
    )
    if len(replicate_rows) != expected_replicate_rows:
        raise AssertionError(
            f"Unexpected bootstrap row count for {target}: "
            f"{len(replicate_rows)} != {expected_replicate_rows}"
        )
    return replicate_rows, summary_rows


# ---------------------------------------------------------------------------
# Diagnostics, summaries, reproducibility
# ---------------------------------------------------------------------------


def model_diagnostics(
    *,
    target: str,
    y_train: np.ndarray,
    y_calibration: np.ndarray,
    calibration_probabilities: np.ndarray,
    y_test: np.ndarray,
    test_probabilities: np.ndarray,
    transformed_feature_n: int,
    fit_seconds: float,
    prediction_seconds: float,
) -> dict[str, Any]:
    def block(y: np.ndarray, probabilities: np.ndarray, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_n": int(len(y)),
            f"{prefix}_event_n": int(np.sum(y == 1)),
            f"{prefix}_prevalence": float(np.mean(y)),
            f"{prefix}_pr_auc": float(average_precision_score(y, probabilities)),
            f"{prefix}_roc_auc": float(roc_auc_score(y, probabilities)),
            f"{prefix}_brier": float(brier_score_loss(y, probabilities)),
        }

    return {
        "target": target,
        "feature_set": FEATURE_SET_NAME,
        "model": "lightgbm",
        "training_population": "target-specific U.S.-resident singleton births",
        "n_raw_features": len(LANDMARK_STRICT_FEATURES),
        "n_transformed_features": int(transformed_feature_n),
        "fit_seconds": float(fit_seconds),
        "temporal_prediction_seconds": float(prediction_seconds),
        **block(y_calibration, calibration_probabilities, "calibration_2022"),
        **block(y_test, test_probabilities, "temporal_test_2023"),
        "train_n": int(len(y_train)),
        "train_event_n": int(np.sum(y_train == 1)),
        "train_prevalence": float(np.mean(y_train)),
    }


def summary_markdown(
    protocol_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    decomposition_rows: list[dict[str, Any]],
    bootstrap_summary: list[dict[str, Any]],
) -> str:
    results = pd.DataFrame(protocol_rows)
    comparisons = pd.DataFrame(comparison_rows)
    decomposition = pd.DataFrame(decomposition_rows)
    bootstrap = pd.DataFrame(bootstrap_summary)

    lines = [
        "# TAE 2026 canonical analysis summary",
        "",
        "- Feature set: `landmark_strict` (9 features).",
        "- One LightGBM per target; no cohort-specific refitting.",
        "- CDC 2023 is temporal test only.",
        "- Availability scenarios: early-entry, any prenatal care, retrospective all-record upper bound.",
        "- Protocols: cohort-specific top-q, fixed absolute budget, common CDC-2022 threshold.",
        "",
        "## Main 10% results",
        "",
        "| Target | Protocol | Scenario | Eligible N | Selected N | TP | Precision | Recall among eligible | Population capture |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    subset = results.loc[results["nominal_fraction"].eq(0.10)]
    for _, row in subset.sort_values(
        ["target", "protocol", "availability_scenario"]
    ).iterrows():
        lines.append(
            f"| {row['target']} | {row['protocol']} | {row['availability_scenario']} | "
            f"{int(row['eligible_n']):,} | {int(row['selected_n']):,} | "
            f"{int(row['true_positive_n']):,} | {100*float(row['precision']):.3f}% | "
            f"{100*float(row['recall_among_eligible']):.3f}% | "
            f"{100*float(row['population_event_capture']):.3f}% |"
        )

    lines += [
        "",
        "## Early-entry → all-record comparison at 10%",
        "",
        "| Target | Protocol | Δ selected | Δ TP | Δ population capture |",
        "|---|---|---:|---:|---:|",
    ]
    subset = comparisons.loc[
        comparisons["nominal_fraction"].eq(0.10)
        & comparisons["from_scenario"].eq("early_entry")
        & comparisons["to_scenario"].eq("all_record_upper_bound")
    ]
    for _, row in subset.sort_values(["target", "protocol"]).iterrows():
        lines.append(
            f"| {row['target']} | {row['protocol']} | {int(row['delta_selected_n']):+,} | "
            f"{int(row['delta_true_positive_n']):+,} | "
            f"{100*float(row['delta_population_event_capture']):+.3f} pp |"
        )

    lines += [
        "",
        "## Same-model decomposition at 10%",
        "",
        "| Target | Contrast | Naive top-q | Availability | Extra capacity | Identity error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in decomposition.loc[
        decomposition["nominal_fraction"].eq(0.10)
    ].sort_values(["target", "from_scenario", "to_scenario"]).iterrows():
        lines.append(
            f"| {row['target']} | {row['from_scenario']} → {row['to_scenario']} | "
            f"{100*float(row['naive_top_fraction_effect']):+.3f} pp | "
            f"{100*float(row['availability_effect_at_to_budget']):+.3f} pp | "
            f"{100*float(row['capacity_effect_within_from_scenario']):+.3f} pp | "
            f"{float(row['identity_error']):.3e} |"
        )

    lines += [
        "",
        "## Bootstrap: fixed-budget early-entry → all-record at 10%",
        "",
        "| Target | Point Δ capture | 95% CI | Point Δ TP |",
        "|---|---:|---:|---:|",
    ]
    subset = bootstrap.loc[
        bootstrap["protocol"].eq("fixed_absolute_budget")
        & bootstrap["nominal_fraction"].eq(0.10)
        & bootstrap["from_scenario"].eq("early_entry")
        & bootstrap["to_scenario"].eq("all_record_upper_bound")
    ]
    for _, row in subset.sort_values("target").iterrows():
        lines.append(
            f"| {row['target']} | "
            f"{100*float(row['point_delta_population_event_capture']):+.3f} pp | "
            f"[{100*float(row['ci_lower_delta_population_event_capture']):+.3f}, "
            f"{100*float(row['ci_upper_delta_population_event_capture']):+.3f}] pp | "
            f"{int(row['point_delta_true_positive_n']):+,} |"
        )

    lines += [
        "",
        "Interpretation guardrail: the fitted model is identical across availability scenarios; differences are not differences between separate 'early' and 'late' predictors.",
        "",
    ]
    return "\n".join(lines)


def feature_set_payload() -> dict[str, Any]:
    return {
        "landmark_strict": {
            "description": (
                "Maximally conservative early-landmark feature set after official "
                "CDC/NCHS timing audit."
            ),
            "features": list(LANDMARK_STRICT_FEATURES),
            "n_features": len(LANDMARK_STRICT_FEATURES),
        },
        "booking_strict_legacy12": {
            "description": "Legacy 12-feature set retained for reproducibility only.",
            "features": list(LEGACY_BOOKING_STRICT_12),
            "n_features": len(LEGACY_BOOKING_STRICT_12),
        },
        "excluded_from_landmark_strict": {
            "features": list(EXCLUDED_FROM_LANDMARK_STRICT),
            "reason": (
                "Official public-use definitions are not maximally anchored to the "
                "early prediction landmark."
            ),
        },
        "target_specific_sets": {
            target: {"landmark_strict": list(LANDMARK_STRICT_FEATURES)}
            for target in TARGETS
        },
    }


def save_reproducibility_bundle(output: Path, config: dict[str, Any]) -> None:
    bundle = output / "reproducibility"
    code_dir = bundle / "code"
    config_dir = bundle / "configs"
    metadata_dir = bundle / "metadata"
    code_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT_PATH, code_dir / SCRIPT_PATH.name)
    atomic_text(
        config_dir / "experiment_config_tae_canonical.locked.yaml",
        yaml.safe_dump(to_plain_python(config), sort_keys=False, allow_unicode=True),
    )
    atomic_json(metadata_dir / "feature_sets_tae.locked.json", feature_set_payload())


def report_to_clearml(
    task: Any,
    protocol_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    decomposition_rows: list[dict[str, Any]],
    bootstrap_summary: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> None:
    logger = task.get_logger()
    tables = [
        ("01 Canonical protocol results", protocol_rows),
        ("02 Protocol comparisons", comparison_rows),
        ("03 Effect decomposition", decomposition_rows),
        ("04 Paired bootstrap summary", bootstrap_summary),
        ("05 Model diagnostics", diagnostics),
    ]
    for title, rows in tables:
        logger.report_table(
            title=title,
            series="TAE 2026 canonical",
            iteration=0,
            table_plot=pd.DataFrame(rows).round(8),
        )
    logger.flush(wait=True)


# ---------------------------------------------------------------------------
# Smoke / invariant test
# ---------------------------------------------------------------------------


def smoke_test() -> int:
    rng = np.random.default_rng(12345)
    n = 3000
    latent = rng.normal(size=n)
    y = (latent + rng.normal(scale=1.2, size=n) > 1.0).astype(np.int8)
    probabilities = 1.0 / (1.0 + np.exp(-(latent + rng.normal(scale=0.5, size=n))))

    # Ensure every care group exists.
    care = rng.choice(
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan]),
        size=n,
        p=np.array([0.02, 0.15, 0.25, 0.25, 0.08, 0.06, 0.05, 0.04, 0.03, 0.025, 0.015, 0.030]),
    )
    masks = availability_masks(care)
    if not (
        masks["early_entry"].sum()
        < masks["any_prenatal_care"].sum()
        < masks["all_record_upper_bound"].sum()
    ):
        raise AssertionError("Smoke-test cohort nesting failed.")

    calibration_probabilities = np.clip(probabilities[:1000], 1e-6, 1 - 1e-6)
    thresholds = {q: threshold_for_capacity(calibration_probabilities, q) for q in (0.05, 0.10)}
    rows, _ = evaluate_protocols(
        target="target_smoke",
        y=y,
        probabilities=probabilities,
        care_month=care,
        fractions=[0.05, 0.10],
        thresholds=thresholds,
    )
    comparisons = protocol_comparison_rows(rows)
    decomposition = effect_decomposition_rows(
        target="target_smoke",
        y=y,
        probabilities=probabilities,
        care_month=care,
        fractions=[0.05, 0.10],
    )
    if len(rows) != 18:
        raise AssertionError(f"Smoke test expected 18 protocol rows, got {len(rows)}")
    if len(comparisons) != 18:
        raise AssertionError(f"Smoke test expected 18 comparisons, got {len(comparisons)}")
    if len(decomposition) != 2:
        raise AssertionError(f"Smoke test expected 2 decomposition rows, got {len(decomposition)}")

    bootstrap_reps, bootstrap_summary = bootstrap_protocol_comparisons(
        target="target_smoke",
        y=y,
        probabilities=probabilities,
        care_month=care,
        fractions=[0.05, 0.10],
        thresholds=thresholds,
        point_comparisons=comparisons,
        replicates=20,
        seed=2027,
        confidence_level=0.95,
        progress_every=0,
    )
    if len(bootstrap_reps) != 20 * 18:
        raise AssertionError("Smoke bootstrap replicate row count failed.")
    if len(bootstrap_summary) != 18:
        raise AssertionError("Smoke bootstrap summary row count failed.")

    print("SMOKE TEST PASS")
    print("cohort_nesting=PASS")
    print("protocol_A_exact_fraction=PASS")
    print("protocol_B_equal_absolute_budget=PASS")
    print("protocol_C_common_threshold=PASS")
    print("population_denominator_invariant=PASS")
    print("decomposition_identity=PASS")
    print("paired_bootstrap=PASS")
    return 0


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TAE 2026 canonical selective-availability evaluation"
    )
    parser.add_argument("--config", type=Path, default=None)
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
        default=os.getenv("CLEARML_TASK_NAME_TAE_CANONICAL") or DEFAULT_TASK_NAME,
    )
    parser.add_argument("--dataset-download-workers", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--skip-clearml", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate config, dataset hashes, columns, and cohorts; do not fit models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        return smoke_test()

    config = load_local_config(args.config)
    if args.bootstrap_replicates is not None:
        config["bootstrap"]["replicates"] = int(args.bootstrap_replicates)
    validate_config(config)

    task: Optional[Any] = None
    dataset_root: Optional[Path] = None

    if args.skip_clearml:
        if not args.dataset_id:
            raise ValueError("--dataset-id is required.")
        dataset_root = bind_clearml_dataset(args.dataset_id, args.dataset_download_workers)
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
            description="Locked TAE 2026 canonical configuration",
        )
        config = to_plain_python(config)
        runtime = task.connect(
            {
                "dataset_id": str(args.dataset_id),
                "queue": str(args.queue),
                "dataset_download_workers": int(args.dataset_download_workers),
                "bootstrap_replicates": int(config["bootstrap"]["replicates"]),
                "preflight_only": bool(args.preflight_only),
            },
            name="runtime",
        )
        runtime = to_plain_python(runtime)
        args.dataset_id = str(runtime["dataset_id"])
        args.queue = str(runtime["queue"])
        args.dataset_download_workers = int(runtime["dataset_download_workers"])
        config["bootstrap"]["replicates"] = int(runtime["bootstrap_replicates"])
        args.preflight_only = bool(runtime["preflight_only"])
        validate_config(config)

        if not args.local:
            if not args.queue:
                raise ValueError("--queue is required.")
            if args.queue != DEFAULT_QUEUE:
                raise ValueError(
                    f"Canonical remote run is locked to queue {DEFAULT_QUEUE}; got {args.queue}."
                )
            if not args.dataset_id:
                raise ValueError("--dataset-id is required.")
            task.execute_remotely(
                queue_name=args.queue,
                clone=False,
                exit_process=True,
            )

        dataset_root = bind_clearml_dataset(args.dataset_id, args.dataset_download_workers)

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
        " ".join(shlex.quote(str(value)) for value in [sys.executable, *sys.argv]) + "\n",
    )
    atomic_text(output / "package_versions.txt", package_versions())
    atomic_text(
        output / "experiment_config.locked.yaml",
        yaml.safe_dump(to_plain_python(config), sort_keys=False, allow_unicode=True),
    )
    atomic_json(output / "feature_sets_tae.locked.json", feature_set_payload())
    atomic_json(output / "git_info.json", git_information())
    save_reproducibility_bundle(output, config)

    cdc2022_path = resolve_dataset_file(dataset_root, str(config["data"]["cdc2022"]))
    cdc2023_path = resolve_dataset_file(dataset_root, str(config["data"]["cdc2023"]))

    observed_hashes: dict[str, str] = {}
    if bool(config["data"].get("verify_input_hashes", True)):
        for year, path in (("2022", cdc2022_path), ("2023", cdc2023_path)):
            print(f"Hashing CDC {year} input...")
            observed = sha256_file(path)
            expected = str(config["data"]["expected_sha256"][year])
            if observed != expected:
                raise ValueError(
                    f"CDC {year} SHA-256 mismatch: observed {observed} != expected {expected}"
                )
            observed_hashes[year] = observed

    columns = required_columns()
    expected_rows = config["data"].get("expected_rows", {})
    frame_2022 = read_cdc_frame(
        cdc2022_path,
        columns,
        int(expected_rows["2022"]),
        "CDC 2022",
    )
    frame_2023 = read_cdc_frame(
        cdc2023_path,
        columns,
        int(expected_rows["2023"]),
        "CDC 2023",
    )

    for label, frame in (("CDC 2022", frame_2022), ("CDC 2023", frame_2023)):
        validate_binary_column(frame, "is_us_resident", label, allow_missing=False)
        validate_binary_column(frame, "is_singleton", label, allow_missing=False)
        frame["prenatal_care_month"] = pd.to_numeric(
            frame["prenatal_care_month"], errors="coerce"
        )
        validate_care_month(frame["prenatal_care_month"].to_numpy(), label)
        for target in TARGETS:
            validate_binary_column(frame, target, label, allow_missing=True)

    primary_2022 = resident_singleton_mask(frame_2022)
    primary_2023 = resident_singleton_mask(frame_2023)

    # Validate cohort definitions on full target-specific populations before fit.
    cohort_rows: list[dict[str, Any]] = []
    for year, frame, primary in (
        (2022, frame_2022, primary_2022),
        (2023, frame_2023, primary_2023),
    ):
        for target in TARGETS:
            known = frame[target].notna().to_numpy(dtype=bool)
            population_mask = primary & known
            y_population = frame.loc[population_mask, target].to_numpy(dtype=np.int8)
            care_population = frame.loc[
                population_mask, "prenatal_care_month"
            ].to_numpy(dtype=np.float64)
            cohort_rows.extend(
                cohort_validation_rows(
                    year=year,
                    target=target,
                    y=y_population,
                    care_month=care_population,
                )
            )
    atomic_csv(output / "cohort_counts_validation.csv", cohort_rows)

    # Hard checks against the frozen cohort counts from 01_COHORT_DEFINITIONS.md.
    frozen_counts = {
        (2022, "target_preterm"): {
            "early_entry": (2671654, 221566), "later_entry": (727408, 57664),
            "any_prenatal_care": (3399062, 279230), "no_care": (73709, 16945),
            "unknown_care": (74970, 11387), "all_record_upper_bound": (3547741, 307562),
        },
        (2022, "target_nicu"): {
            "early_entry": (2667311, 211274), "later_entry": (726286, 62734),
            "any_prenatal_care": (3393597, 274008), "no_care": (74710, 14581),
            "unknown_care": (75236, 9914), "all_record_upper_bound": (3543543, 298503),
        },
        (2022, "target_lbw"): {
            "early_entry": (2670591, 173294), "later_entry": (726981, 53740),
            "any_prenatal_care": (3397572, 227034), "no_care": (74670, 12701),
            "unknown_care": (75426, 9010), "all_record_upper_bound": (3547668, 248745),
        },
        (2023, "target_preterm"): {
            "early_entry": (2601060, 217632), "later_entry": (739298, 58282),
            "any_prenatal_care": (3340358, 275914), "no_care": (76520, 17387),
            "unknown_care": (63504, 9700), "all_record_upper_bound": (3480382, 303001),
        },
        (2023, "target_nicu"): {
            "early_entry": (2597274, 214174), "later_entry": (738055, 65142),
            "any_prenatal_care": (3335329, 279316), "no_care": (77422, 15194),
            "unknown_care": (63833, 8775), "all_record_upper_bound": (3476584, 303285),
        },
        (2023, "target_lbw"): {
            "early_entry": (2600093, 169152), "later_entry": (738893, 54066),
            "any_prenatal_care": (3338986, 223218), "no_care": (77325, 12870),
            "unknown_care": (64005, 7688), "all_record_upper_bound": (3480316, 243776),
        },
    }
    cohort_lookup = {
        (int(row["year"]), str(row["target"]), str(row["scenario"])): (
            int(row["n_records"]), int(row["event_n"])
        )
        for row in cohort_rows
    }
    for (year, target), expected_by_scenario in frozen_counts.items():
        for scenario, expected_pair in expected_by_scenario.items():
            observed_pair = cohort_lookup[(year, target, scenario)]
            if observed_pair != expected_pair:
                raise AssertionError(
                    f"Frozen cohort mismatch {year}/{target}/{scenario}: "
                    f"observed={observed_pair}, expected={expected_pair}"
                )

    if args.preflight_only:
        manifest = {
            "status": "PREFLIGHT_PASS",
            "clearml_task_id": task.id if task is not None else None,
            "clearml_dataset_id": args.dataset_id,
            "clearml_queue": args.queue,
            "input_sha256": observed_hashes,
            "feature_set": feature_set_payload()["landmark_strict"],
            "cohort_validation": "PASS",
            "elapsed_seconds": float(time.time() - started),
        }
        atomic_json(output / "run_manifest.json", manifest)
        if task is not None:
            task.upload_artifact(
                name="preflight_manifest",
                artifact_object=str(output / "run_manifest.json"),
                wait_on_upload=True,
            )
            task.upload_artifact(
                name="reproducibility_bundle",
                artifact_object=str(output / "reproducibility"),
                wait_on_upload=True,
            )
            task.close()
        print("PREFLIGHT PASS — no models were fitted.")
        return 0

    # CDC 2022 split is assigned before target filters.
    split_rng = np.random.default_rng(int(config["split"]["seed"]))
    calibration_mask_all_2022 = (
        split_rng.random(len(frame_2022)) < float(config["split"]["calibration_fraction"])
    )
    train_mask_all_2022 = ~calibration_mask_all_2022
    if np.any(train_mask_all_2022 & calibration_mask_all_2022):
        raise AssertionError("Train/calibration overlap.")
    if not np.all(train_mask_all_2022 | calibration_mask_all_2022):
        raise AssertionError("Train/calibration split does not cover CDC 2022.")

    features = list(LANDMARK_STRICT_FEATURES)
    categorical_features = ["mother_race"]
    parameters = lightgbm_parameters(config)
    fractions = [float(v) for v in config["evaluation"]["fractions"]]

    from lightgbm import LGBMClassifier

    protocol_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    bootstrap_replicate_rows: list[dict[str, Any]] = []
    bootstrap_summary_rows: list[dict[str, Any]] = []
    saved_models: list[str] = []
    saved_predictions: list[str] = []

    model_fit_count = 0
    temporal_prediction_pass_count = 0

    for target_index, target in enumerate(TARGETS):
        print(f"\n===== TAE CANONICAL: {target} =====")
        known_2022 = frame_2022[target].notna().to_numpy(dtype=bool)
        known_2023 = frame_2023[target].notna().to_numpy(dtype=bool)
        train_mask = primary_2022 & known_2022 & train_mask_all_2022
        calibration_mask = primary_2022 & known_2022 & calibration_mask_all_2022
        test_mask = primary_2023 & known_2023

        y_train = frame_2022.loc[train_mask, target].to_numpy(dtype=np.int8)
        y_calibration = frame_2022.loc[calibration_mask, target].to_numpy(dtype=np.int8)
        y_test = frame_2023.loc[test_mask, target].to_numpy(dtype=np.int8)
        for name, values in (
            ("train", y_train),
            ("calibration_2022", y_calibration),
            ("temporal_test_2023", y_test),
        ):
            if len(np.unique(values)) != 2:
                raise ValueError(f"{target}/{name} does not contain both classes.")

        preprocessor = build_preprocessor(
            features,
            categorical_features,
            config["preprocessing"],
        )
        fit_started = time.time()
        x_train = consistent_matrix(
            preprocessor.fit_transform(frame_2022.loc[train_mask, features])
        )
        model = LGBMClassifier(**parameters)
        model.fit(x_train, y_train)
        fit_seconds = time.time() - fit_started
        transformed_feature_n = int(x_train.shape[1])
        del x_train
        gc.collect()
        model_fit_count += 1

        x_calibration = consistent_matrix(
            preprocessor.transform(frame_2022.loc[calibration_mask, features])
        )
        calibration_probabilities = np.asarray(
            model.predict_proba(x_calibration)[:, 1], dtype=np.float64
        )
        del x_calibration
        gc.collect()

        thresholds = {
            float(q): threshold_for_capacity(calibration_probabilities, float(q))
            for q in fractions
        }
        for q in fractions:
            threshold = thresholds[float(q)]
            realized_selected_n = int(np.sum(calibration_probabilities >= threshold))
            nominal_selected_n = max(1, int(round(float(q) * len(y_calibration))))
            threshold_rows.append(
                {
                    "target": target,
                    "feature_set": FEATURE_SET_NAME,
                    "model": "lightgbm",
                    "source_year": 2022,
                    "source_population": "calibration target-specific all-record",
                    "nominal_fraction": float(q),
                    "calibration_n": int(len(y_calibration)),
                    "nominal_selected_n": nominal_selected_n,
                    "realized_selected_n_with_ties": realized_selected_n,
                    "threshold": float(threshold),
                    "comparator": ">=",
                }
            )

        prediction_started = time.time()
        x_test = consistent_matrix(
            preprocessor.transform(frame_2023.loc[test_mask, features])
        )
        test_probabilities = np.asarray(
            model.predict_proba(x_test)[:, 1], dtype=np.float64
        )
        prediction_seconds = time.time() - prediction_started
        del x_test
        gc.collect()
        temporal_prediction_pass_count += 1

        care_test = frame_2023.loc[
            test_mask, "prenatal_care_month"
        ].to_numpy(dtype=np.float64)

        target_protocol_rows, _ = evaluate_protocols(
            target=target,
            y=y_test,
            probabilities=test_probabilities,
            care_month=care_test,
            fractions=fractions,
            thresholds=thresholds,
        )
        protocol_rows.extend(target_protocol_rows)
        target_comparisons = protocol_comparison_rows(target_protocol_rows)
        comparison_rows.extend(target_comparisons)
        decomposition_rows.extend(
            effect_decomposition_rows(
                target=target,
                y=y_test,
                probabilities=test_probabilities,
                care_month=care_test,
                fractions=fractions,
            )
        )

        diagnostics_rows.append(
            model_diagnostics(
                target=target,
                y_train=y_train,
                y_calibration=y_calibration,
                calibration_probabilities=calibration_probabilities,
                y_test=y_test,
                test_probabilities=test_probabilities,
                transformed_feature_n=transformed_feature_n,
                fit_seconds=fit_seconds,
                prediction_seconds=prediction_seconds,
            )
        )

        bootstrap_seed = int(config["bootstrap"]["seed"]) + target_index * 100_000
        reps, summary = bootstrap_protocol_comparisons(
            target=target,
            y=y_test,
            probabilities=test_probabilities,
            care_month=care_test,
            fractions=fractions,
            thresholds=thresholds,
            point_comparisons=target_comparisons,
            replicates=int(config["bootstrap"]["replicates"]),
            seed=bootstrap_seed,
            confidence_level=float(config["bootstrap"]["confidence_level"]),
            progress_every=int(config["bootstrap"]["progress_every"]),
        )
        bootstrap_replicate_rows.extend(reps)
        bootstrap_summary_rows.extend(summary)

        if bool(config["output"].get("save_models", True)):
            model_path = models_directory / f"{target}__landmark_strict__lightgbm.joblib"
            joblib.dump(
                {
                    "preprocessor": preprocessor,
                    "model": model,
                    "features": features,
                    "target": target,
                    "feature_set": FEATURE_SET_NAME,
                    "training_population": "target-specific U.S.-resident singleton births",
                    "thresholds": thresholds,
                },
                model_path,
            )
            saved_models.append(str(model_path.relative_to(output)))

        if bool(config["output"].get("save_predictions", False)):
            prediction_path = predictions_directory / f"{target}__cdc2023.npz"
            np.savez_compressed(
                prediction_path,
                y=y_test.astype(np.uint8),
                prenatal_care_month=care_test.astype(np.float32),
                probability=test_probabilities.astype(np.float32),
            )
            saved_predictions.append(str(prediction_path.relative_to(output)))

        print(
            f"fit={fit_seconds:.1f}s, predict={prediction_seconds:.1f}s, "
            f"test_n={len(y_test):,}, transformed_features={transformed_feature_n}"
        )

        del (
            preprocessor,
            model,
            calibration_probabilities,
            test_probabilities,
            y_train,
            y_calibration,
            y_test,
            care_test,
        )
        gc.collect()

    if model_fit_count != int(config["execution"]["expected_primary_model_count"]):
        raise AssertionError(
            f"Expected 3 model fits, received {model_fit_count}."
        )
    if temporal_prediction_pass_count != int(
        config["execution"]["expected_temporal_prediction_pass_count"]
    ):
        raise AssertionError(
            f"Expected 3 CDC 2023 prediction passes, received {temporal_prediction_pass_count}."
        )

    expected_protocol_rows = len(TARGETS) * len(SCENARIOS) * len(PROTOCOLS) * len(fractions)
    expected_comparison_rows = (
        len(TARGETS) * len(PROTOCOLS) * len(fractions) * len(PAIRWISE_COMPARISONS)
    )
    expected_decomposition_rows = len(TARGETS) * len(fractions) * len(DECOMPOSITION_COMPARISONS)
    if len(protocol_rows) != expected_protocol_rows:
        raise AssertionError(
            f"Protocol row count {len(protocol_rows)} != {expected_protocol_rows}"
        )
    if len(comparison_rows) != expected_comparison_rows:
        raise AssertionError(
            f"Comparison row count {len(comparison_rows)} != {expected_comparison_rows}"
        )
    if len(decomposition_rows) != expected_decomposition_rows:
        raise AssertionError(
            f"Decomposition row count {len(decomposition_rows)} != {expected_decomposition_rows}"
        )

    atomic_csv(output / "canonical_protocol_results.csv", protocol_rows)
    atomic_csv(output / "canonical_protocol_comparisons.csv", comparison_rows)
    atomic_csv(output / "effect_decomposition.csv", decomposition_rows)
    atomic_csv(output / "model_diagnostics.csv", diagnostics_rows)
    atomic_csv(output / "thresholds_2022.csv", threshold_rows)
    atomic_csv(output / "bootstrap_summary.csv", bootstrap_summary_rows)
    if bool(config["output"].get("save_bootstrap_replicates", True)):
        atomic_csv(output / "bootstrap_replicates.csv", bootstrap_replicate_rows)

    atomic_text(
        output / "tae_canonical_summary.md",
        summary_markdown(
            protocol_rows,
            comparison_rows,
            decomposition_rows,
            bootstrap_summary_rows,
        ),
    )

    manifest = {
        "status": "PASS",
        "elapsed_seconds": float(time.time() - started),
        "clearml_task_id": task.id if task is not None else None,
        "clearml_dataset_id": args.dataset_id,
        "clearml_queue": args.queue,
        "input_sha256": observed_hashes,
        "design": {
            "primary_population": "U.S.-resident singleton births with target known",
            "feature_set": FEATURE_SET_NAME,
            "features": list(LANDMARK_STRICT_FEATURES),
            "n_features": len(LANDMARK_STRICT_FEATURES),
            "model": "lightgbm",
            "model_fit_count": model_fit_count,
            "temporal_prediction_pass_count": temporal_prediction_pass_count,
            "scenarios": list(SCENARIOS),
            "protocols": list(PROTOCOLS),
            "fractions": fractions,
            "threshold_source": "CDC 2022 target-specific all-record calibration pool",
            "cdc2023_role": "temporal test only",
            "lightgbm_device_type": "cpu",
            "clearml_agent_queue": args.queue,
        },
        "bootstrap": {
            "method": "paired full-size record-level nonparametric bootstrap",
            "replicates": int(config["bootstrap"]["replicates"]),
            "base_seed": int(config["bootstrap"]["seed"]),
            "confidence_level": float(config["bootstrap"]["confidence_level"]),
        },
        "outputs": {
            "protocol_rows": len(protocol_rows),
            "comparison_rows": len(comparison_rows),
            "decomposition_rows": len(decomposition_rows),
            "bootstrap_summary_rows": len(bootstrap_summary_rows),
            "bootstrap_replicate_rows": len(bootstrap_replicate_rows),
            "saved_models": saved_models,
            "saved_predictions": saved_predictions,
        },
        "invariants": {
            "one_model_per_target": True,
            "one_cdc2023_prediction_vector_per_target": True,
            "early_subset_any_subset_all": True,
            "same_full_population_denominator_across_scenarios": True,
            "protocol_A_exact_cohort_fraction_budget": True,
            "protocol_B_exact_equal_absolute_budget": True,
            "protocol_C_common_2022_threshold": True,
            "decomposition_identity_exact": True,
            "cdc2023_not_used_for_fit_or_threshold": True,
        },
    }
    atomic_json(output / "run_manifest.json", manifest)

    # Hash all small/medium outputs except models/predictions and manifest itself.
    output_hashes: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "run_manifest.json":
            continue
        if "models" in path.parts or "predictions" in path.parts:
            continue
        output_hashes[str(path.relative_to(output))] = sha256_file(path)
    manifest["output_sha256"] = output_hashes
    atomic_json(output / "run_manifest.json", manifest)

    if task is not None:
        report_to_clearml(
            task,
            protocol_rows,
            comparison_rows,
            decomposition_rows,
            bootstrap_summary_rows,
            diagnostics_rows,
        )
        artifact_files = (
            "canonical_protocol_results.csv",
            "canonical_protocol_comparisons.csv",
            "effect_decomposition.csv",
            "model_diagnostics.csv",
            "thresholds_2022.csv",
            "cohort_counts_validation.csv",
            "bootstrap_summary.csv",
            "tae_canonical_summary.md",
            "run_manifest.json",
            "experiment_config.locked.yaml",
            "feature_sets_tae.locked.json",
            "package_versions.txt",
            "commands.txt",
            "git_info.json",
        )
        if bool(config["output"].get("save_bootstrap_replicates", True)):
            artifact_files = (*artifact_files, "bootstrap_replicates.csv")
        for filename in artifact_files:
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
        if bool(config["output"].get("upload_models_to_clearml", True)):
            task.upload_artifact(
                name="tae_canonical_models",
                artifact_object=str(models_directory),
                wait_on_upload=True,
            )
        task.close()

    print("TAE CANONICAL RUN PASS")
    print(f"Output: {output}")
    print(f"Models fitted: {model_fit_count}")
    print(f"CDC 2023 prediction passes: {temporal_prediction_pass_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
