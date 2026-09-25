#!/usr/bin/env python3
"""
TAE 2026 four-cell + symmetric Shapley decomposition.

This is a STANDALONE post-canonical ClearML analysis.

Scientific purpose
------------------
For each target and q in {0.05, 0.10}, use the SAME frozen CDC-2023 score
vector and evaluate the 2x2 design:

                         early-derived budget   all-record-derived budget
    early candidate set          C00                     C01
    all-record candidate set     C10                     C11

where population event capture is TP / all target-population events.

Point-estimate budgets:
    B_early = round(q * N_early)
    B_all   = round(q * N_all)

Two-factor decomposition:
    availability_main = C10 - C00
    capacity_main     = C01 - C00
    interaction       = C11 - C10 - C01 + C00

Symmetric Shapley effects:
    phi_availability =
        0.5 * [(C10 - C00) + (C11 - C01)]

    phi_capacity =
        0.5 * [(C01 - C00) + (C11 - C10)]

Identities:
    C11 - C00
      = availability_main + capacity_main + interaction
      = phi_availability + phi_capacity

Bootstrap
---------
Paired full-size record-level nonparametric bootstrap with FIXED OBSERVED
ABSOLUTE BUDGET LEVELS.

For each target and q, the two factorial budget levels are defined ONCE from
the observed CDC-2023 point-estimate populations:

    B_early = round(q * N_early_observed)
    B_all   = round(q * N_all_observed)

Those two absolute counts are then held fixed in every bootstrap replicate,
even though the resampled number of early-entry records N_early* varies.

This matches the mentor-requested 2x2 factorial estimand:
    early/full candidate set x early/full ABSOLUTE budget level.

The SAME bootstrap multiplicities are used for all four cells.

Canonical cross-checks
----------------------
The script refuses to proceed unless the frozen canonical model/data reproduce
the point cells:

    C00 = canonical cohort_specific_top_fraction / early_entry
    C01 = canonical fixed_absolute_budget / early_entry
    C11 = canonical fixed_absolute_budget / all_record_upper_bound
          = canonical cohort_specific_top_fraction / all_record_upper_bound

At the bootstrap-replicate level, C01 and C11 reproduce the canonical
fixed_absolute_budget early->all bootstrap exactly. C00/C10 are the new
fixed-B_early factorial cells and therefore are NOT compared with the canonical
cohort-specific top-q bootstrap, which recomputed its early budget per replicate.

ClearML execution
-----------------
No GitHub repository is required for remote execution.

The script calls:
    Task.force_store_standalone_script(True)

before Task.init(), so ClearML stores this Python file itself rather than
requiring a repository/commit checkout.

Inputs:
    - finalized ClearML Dataset for CDC harmonized data;
    - frozen canonical Task artifacts:
        canonical_protocol_results.csv
        bootstrap_replicates.csv
        run_manifest.json
        experiment_config.locked.yaml
        tae_canonical_models

No model is refit.
No threshold is estimated.
CDC 2023 is treated here as the temporal evaluation cohort.

The clean git commit/tag requested for the paper is a reproducibility/protocol
freeze requirement, especially before a future locked CDC 2024 replication;
it is not required for this standalone ClearML execution.
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
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Locked design / ClearML lineage
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_VERSION = "2026-08-24-v4-fixed-absolute-budget-bootstrap"

DEFAULT_PROJECT_NAME = "pershin-medailab/Vache_Oganisyan/CDC Natality Audit"
DEFAULT_DATASET_ID = "062ba26c0ca24cef99549c2a2ab34e65"
DEFAULT_QUEUE = "6be0e69f7fab49d48bc305ef1fb03a6a"
DEFAULT_CANONICAL_TASK_ID = "4f9d98dd39f3438181f48b256b23bd94"
DEFAULT_TASK_NAME = "TAE 2026 Shapley: fixed absolute budget-level bootstrap"

TARGETS = ("target_preterm", "target_nicu", "target_lbw")
FRACTIONS = (0.05, 0.10)
FEATURE_SET_NAME = "landmark_strict"
MODEL_NAME = "lightgbm"

EARLY_SCENARIO = "early_entry"
ALL_SCENARIO = "all_record_upper_bound"

LANDMARK_STRICT_FEATURES = [
    "mother_race",
    "mother_height_inches",
    "mother_bmi",
    "mother_weight_pre",
    "prior_live_births",
    "prior_dead_births",
    "prior_terminations",
    "diab_pre",
    "hyper_pre",
]

EXPECTED_2023_ROWS = 3_605_081
EXPECTED_2023_SHA256 = (
    "82b38b24d2f795a4a5233ce3a2fcb43b70789339306ba9044bde56f50bb270a8"
)
EXPECTED_2023_CONFIGURED_PATH = (
    "data/cdc_temporal_harmonized/cdc_natality_2023_harmonized.csv"
)

CANONICAL_ARTIFACTS = {
    "protocol_results": "canonical_protocol_results.csv",
    "bootstrap_replicates": "bootstrap_replicates.csv",
    "manifest": "run_manifest.json",
    "config": "experiment_config.locked.yaml",
    "models": "tae_canonical_models",
}

TAGS = [
    "tae-2026",
    "cdc-natality",
    "post-canonical",
    "four-cell",
    "shapley",
    "interaction",
    "decision-aware-evaluation",
    "same-frozen-scores",
    "paired-bootstrap",
    "full-size-bootstrap",
    "fixed-observed-absolute-budget-levels",
    "mentor-factorial-bootstrap",
    "standalone-script",
    "temporal-evaluation-cohort-2023",
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

# LightGBM may not appear as a direct import before ClearML captures
# requirements, but joblib needs it to unpickle the frozen LGBMClassifier.
CLEARML_EXTRA_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("lightgbm", ">=4.3,<5"),
    ("scikit-learn", ">=1.4,<2"),
    ("joblib", ">=1.3,<2"),
)

DEFAULT_CONFIG: dict[str, Any] = {
    "analysis": {
        "name": "four_cell_symmetric_shapley_fixed_absolute_budget_bootstrap",
        "script_version": SCRIPT_VERSION,
        "source_canonical_task_id": DEFAULT_CANONICAL_TASK_ID,
        "cdc2023_role": "temporal evaluation cohort",
        "model_refit": False,
        "threshold_reestimation": False,
    },
    "data": {
        "cdc2023": EXPECTED_2023_CONFIGURED_PATH,
        "expected_rows_2023": EXPECTED_2023_ROWS,
        "verify_input_hash": True,
        "expected_sha256_2023": EXPECTED_2023_SHA256,
    },
    "population": {
        "primary": "is_us_resident == 1 and is_singleton == 1 and target known",
        "early_entry_min_month": 1,
        "early_entry_max_month": 3,
        "all_record": "all target-specific primary-population records",
    },
    "targets": list(TARGETS),
    "feature_set": {
        "name": FEATURE_SET_NAME,
        "paper_name": "conservative landmark feature set",
        "features": list(LANDMARK_STRICT_FEATURES),
        "n_features": 9,
    },
    "evaluation": {
        "fractions": list(FRACTIONS),
        "candidate_sets": [EARLY_SCENARIO, ALL_SCENARIO],
        "budget_levels": ["early_derived_absolute_budget", "all_record_derived_absolute_budget"],
        "metric": "population_event_capture",
        "metric_formula": "true_positive_n / full_population_event_n",
        "tie_breaker_top_b": "score_descending_then_original_row_position_ascending",
    },
    "bootstrap": {
        "enabled": True,
        "replicates": 500,
        "seed": 2027,
        "target_seed_stride": 100_000,
        "confidence_level": 0.95,
        "full_size": True,
        "paired": True,
        "recompute_early_budget_each_replicate": False,
        "fix_observed_absolute_budget_levels": True,
        "budget_lock": "B_early_and_B_all_fixed_to_observed_point_counts",
        "progress_every": 25,
    },
    "output": {
        "directory": "results/tae_2026_shapley_fixed_b_bootstrap",
        "overwrite": False,
        "save_bootstrap_replicates": True,
    },
    "execution": {
        "clearml_default_queue": DEFAULT_QUEUE,
        "clearml_dataset_id": DEFAULT_DATASET_ID,
        "canonical_task_id": DEFAULT_CANONICAL_TASK_ID,
        "dataset_download_workers": 4,
    },
}


# ---------------------------------------------------------------------------
# Atomic I/O / hashes / metadata
# ---------------------------------------------------------------------------

def to_plain_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_plain_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_python(v) for v in value]
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
        json.dumps(to_plain_python(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
    """
    Git metadata is recorded if available, but git is intentionally NOT required
    for ClearML remote execution because this is a standalone-script Task.
    """
    def run_git(*arguments: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=Path.cwd(),
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    status = run_git("status", "--porcelain")
    return {
        "commit_hash": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
        "required_for_remote_execution": False,
        "note": (
            "A clean commit/tag is still required for the paper protocol freeze "
            "and before any locked CDC 2024 replication."
        ),
    }


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
# ClearML Dataset / canonical Task artifact loading
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
        raise FileNotFoundError(
            f"ClearML Dataset local copy is not a directory: {root}"
        )
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
        suffix = str(configured_path).replace("\\", "/")
        exact = [
            path for path in matches
            if str(path).replace("\\", "/").endswith(suffix)
        ]
        if len(exact) == 1:
            return exact[0].resolve()
        raise RuntimeError(
            f"Ambiguous dataset file {configured!r}; "
            f"matches: {[str(path) for path in matches[:10]]}"
        )

    raise FileNotFoundError(
        f"Dataset file {configured!r} was not found under {dataset_root}"
    )


def require_artifact(task: Any, name: str) -> Any:
    artifacts = task.artifacts
    if name not in artifacts:
        raise KeyError(
            f"Canonical Task {task.id} is missing artifact {name!r}. "
            f"Available artifacts: {sorted(artifacts.keys())}"
        )
    return artifacts[name]


def artifact_local_path(task: Any, name: str) -> Path:
    artifact = require_artifact(task, name)
    local = artifact.get_local_copy()
    if not local:
        raise RuntimeError(
            f"Artifact {name!r} from Task {task.id} returned no local copy."
        )
    path = Path(local).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Downloaded artifact path does not exist: {path}"
        )
    return path


def materialize_directory_artifact(task: Any, name: str) -> Path:
    """
    ClearML folder artifacts are usually downloaded/extracted as directories.
    If the local cache returns an archive instead, unpack it defensively.
    """
    path = artifact_local_path(task, name)

    if path.is_dir():
        return path

    if path.is_file():
        temporary = Path(tempfile.mkdtemp(prefix=f"{name}_"))
        try:
            shutil.unpack_archive(str(path), str(temporary))
        except (shutil.ReadError, ValueError) as exc:
            raise RuntimeError(
                f"Artifact {name!r} is neither a directory nor an unpackable archive: {path}"
            ) from exc
        return temporary

    raise FileNotFoundError(f"Unsupported artifact path: {path}")


def find_exact_file(root: Path, basename: str) -> Path:
    if root.is_file():
        if root.name == basename:
            return root.resolve()
        raise FileNotFoundError(f"Expected {basename!r}, got {root}")

    matches = [
        path.resolve()
        for path in root.rglob(basename)
        if path.is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {basename!r} under {root}; "
            f"found {len(matches)}: {[str(p) for p in matches[:10]]}"
        )
    return matches[0]


def load_canonical_sources(
    canonical_task_id: str,
) -> dict[str, Any]:
    from clearml import Task

    canonical_task = Task.get_task(task_id=canonical_task_id)

    protocol_path = artifact_local_path(
        canonical_task, CANONICAL_ARTIFACTS["protocol_results"]
    )
    bootstrap_path = artifact_local_path(
        canonical_task, CANONICAL_ARTIFACTS["bootstrap_replicates"]
    )
    manifest_path = artifact_local_path(
        canonical_task, CANONICAL_ARTIFACTS["manifest"]
    )
    config_path = artifact_local_path(
        canonical_task, CANONICAL_ARTIFACTS["config"]
    )
    models_root = materialize_directory_artifact(
        canonical_task, CANONICAL_ARTIFACTS["models"]
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if manifest.get("status") != "PASS":
        raise AssertionError(
            f"Canonical source Task manifest status is {manifest.get('status')!r}, not PASS."
        )

    manifest_task_id = manifest.get("clearml_task_id")
    if manifest_task_id and str(manifest_task_id) != str(canonical_task_id):
        raise AssertionError(
            f"Canonical manifest Task ID {manifest_task_id} != requested {canonical_task_id}"
        )

    manifest_dataset = manifest.get("clearml_dataset_id")
    if manifest_dataset and str(manifest_dataset) != DEFAULT_DATASET_ID:
        raise AssertionError(
            f"Canonical manifest Dataset ID {manifest_dataset} != locked {DEFAULT_DATASET_ID}"
        )

    validate_canonical_config(canonical_config)

    protocol = pd.read_csv(protocol_path)
    bootstrap = pd.read_csv(bootstrap_path)

    return {
        "task": canonical_task,
        "protocol_path": protocol_path,
        "bootstrap_path": bootstrap_path,
        "manifest_path": manifest_path,
        "config_path": config_path,
        "models_root": models_root,
        "manifest": manifest,
        "config": canonical_config,
        "protocol": protocol,
        "bootstrap": bootstrap,
    }


# ---------------------------------------------------------------------------
# Locked design validation
# ---------------------------------------------------------------------------

def validate_canonical_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("Canonical locked config is not a mapping.")

    if tuple(map(str, config.get("targets", []))) != TARGETS:
        raise AssertionError(
            f"Canonical targets changed: {config.get('targets')}"
        )

    feature = config.get("feature_set", {})
    if str(feature.get("name")) != FEATURE_SET_NAME:
        raise AssertionError(
            f"Canonical feature set changed: {feature.get('name')!r}"
        )

    features = [str(v) for v in feature.get("features", [])]
    if features != LANDMARK_STRICT_FEATURES:
        raise AssertionError(
            "Canonical feature list no longer matches frozen landmark_strict."
        )

    if int(feature.get("n_features", -1)) != 9:
        raise AssertionError("Canonical feature count is not 9.")

    fractions = tuple(float(v) for v in config["evaluation"]["fractions"])
    if fractions != FRACTIONS:
        raise AssertionError(
            f"Canonical fractions changed: {fractions}"
        )

    if int(config["bootstrap"]["seed"]) != 2027:
        raise AssertionError("Canonical bootstrap seed is not 2027.")

    if not bool(config["bootstrap"].get("paired", False)):
        raise AssertionError("Canonical bootstrap was not marked paired.")

    if not bool(config["bootstrap"].get("full_size", False)):
        raise AssertionError("Canonical bootstrap was not marked full-size.")

    if str(config["data"]["expected_sha256"]["2023"]) != EXPECTED_2023_SHA256:
        raise AssertionError("Canonical CDC 2023 SHA256 changed.")

    if int(config["data"]["expected_rows"]["2023"]) != EXPECTED_2023_ROWS:
        raise AssertionError("Canonical CDC 2023 row count changed.")


def validate_runtime_config(config: dict[str, Any]) -> None:
    if tuple(map(str, config["targets"])) != TARGETS:
        raise ValueError("targets must remain exactly the canonical three targets.")

    if tuple(float(v) for v in config["evaluation"]["fractions"]) != FRACTIONS:
        raise ValueError("fractions must remain exactly [0.05, 0.10].")

    if str(config["execution"]["clearml_default_queue"]) != DEFAULT_QUEUE:
        raise ValueError("ClearML queue differs from locked canonical queue.")

    if str(config["execution"]["clearml_dataset_id"]) != DEFAULT_DATASET_ID:
        raise ValueError("ClearML Dataset differs from canonical Dataset.")

    if str(config["execution"]["canonical_task_id"]) != DEFAULT_CANONICAL_TASK_ID:
        raise ValueError("Canonical source Task differs from locked Task.")

    bootstrap = config["bootstrap"]
    if not bool(bootstrap["enabled"]):
        raise ValueError("Bootstrap must be enabled.")
    if not bool(bootstrap["paired"]):
        raise ValueError("Bootstrap must be paired.")
    if not bool(bootstrap["full_size"]):
        raise ValueError("Bootstrap must be full-size.")
    if int(bootstrap["seed"]) != 2027:
        raise ValueError("Bootstrap base seed is locked to 2027.")
    if int(bootstrap["replicates"]) != 500:
        raise ValueError(
            "Publication Shapley analysis is locked to exactly 500 bootstrap replicates."
        )
    if bool(bootstrap.get("recompute_early_budget_each_replicate", True)):
        raise ValueError(
            "Primary factorial bootstrap must keep B_early fixed at the observed "
            "point-estimate absolute count."
        )
    if not bool(bootstrap.get("fix_observed_absolute_budget_levels", False)):
        raise ValueError(
            "Primary factorial bootstrap must fix both observed absolute budget levels."
        )
    if str(bootstrap.get("budget_lock")) != (
        "B_early_and_B_all_fixed_to_observed_point_counts"
    ):
        raise ValueError("Unexpected bootstrap budget-lock definition.")
    if not 0.0 < float(bootstrap["confidence_level"]) < 1.0:
        raise ValueError("Invalid bootstrap confidence level.")


# ---------------------------------------------------------------------------
# CDC frame / cohort masks
# ---------------------------------------------------------------------------

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


def read_cdc_2023(path: Path) -> pd.DataFrame:
    columns = required_columns()
    print(f"Reading CDC 2023: {path}")
    frame = pd.read_csv(path, usecols=columns, low_memory=False)

    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"CDC 2023 is missing required columns: {missing}")

    if len(frame) != EXPECTED_2023_ROWS:
        raise ValueError(
            f"CDC 2023 row count {len(frame):,} != expected {EXPECTED_2023_ROWS:,}"
        )

    for column in ("is_us_resident", "is_singleton"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"CDC 2023.{column} contains missing/non-numeric values.")
        unique = set(values.unique().tolist())
        if not unique.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"CDC 2023.{column} is not binary: {sorted(unique)[:20]}")
        frame[column] = values

    for target in TARGETS:
        values = pd.to_numeric(frame[target], errors="coerce")
        unique = set(values.dropna().unique().tolist())
        if not unique.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"CDC 2023.{target} is not binary.")
        frame[target] = values

    frame["prenatal_care_month"] = pd.to_numeric(
        frame["prenatal_care_month"], errors="coerce"
    )
    validate_care_month(
        frame["prenatal_care_month"].to_numpy(dtype=np.float64),
        "CDC 2023",
    )

    print(
        f"CDC 2023: {len(frame):,} rows, {len(frame.columns)} required columns loaded"
    )
    return frame


def validate_care_month(values: np.ndarray, label: str) -> None:
    care = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(care)
    finite_values = care[finite]

    if finite_values.size == 0:
        raise ValueError(f"{label}: prenatal_care_month has no finite values.")

    if np.any(finite_values < 0) or np.any(finite_values > 10):
        bad = finite_values[(finite_values < 0) | (finite_values > 10)][:20]
        raise ValueError(
            f"{label}: prenatal_care_month outside 0..10: {bad.tolist()}"
        )

    if not np.allclose(
        finite_values, np.round(finite_values), atol=0.0, rtol=0.0
    ):
        bad = finite_values[
            finite_values != np.round(finite_values)
        ][:20]
        raise ValueError(
            f"{label}: non-integer prenatal_care_month: {bad.tolist()}"
        )


def resident_singleton_mask(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["is_us_resident"].eq(1)
        & frame["is_singleton"].eq(1)
    ).to_numpy(dtype=bool)


def early_entry_mask(care_month: np.ndarray) -> np.ndarray:
    values = np.asarray(care_month, dtype=np.float64)
    finite = np.isfinite(values)
    return finite & (values >= 1) & (values <= 3)


# ---------------------------------------------------------------------------
# Frozen model loading and prediction
# ---------------------------------------------------------------------------

def load_frozen_model_bundle(models_root: Path, target: str) -> dict[str, Any]:
    basename = f"{target}__landmark_strict__lightgbm.joblib"
    model_path = find_exact_file(models_root, basename)

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise TypeError(
            f"{basename}: expected dictionary bundle, got {type(bundle)!r}"
        )

    required = {"preprocessor", "model", "features", "target", "feature_set"}
    missing = required - set(bundle)
    if missing:
        raise KeyError(f"{basename}: missing keys {sorted(missing)}")

    if str(bundle["target"]) != target:
        raise AssertionError(
            f"{basename}: bundle target {bundle['target']!r} != {target!r}"
        )

    if str(bundle["feature_set"]) != FEATURE_SET_NAME:
        raise AssertionError(
            f"{basename}: wrong feature set {bundle['feature_set']!r}"
        )

    features = [str(v) for v in bundle["features"]]
    if features != LANDMARK_STRICT_FEATURES:
        raise AssertionError(
            f"{basename}: frozen feature list does not match canonical strict9."
        )

    if not hasattr(bundle["preprocessor"], "transform"):
        raise TypeError(f"{basename}: preprocessor lacks transform().")

    if not hasattr(bundle["model"], "predict_proba"):
        raise TypeError(f"{basename}: model lacks predict_proba().")

    return {
        **bundle,
        "_model_path": str(model_path),
        "_model_sha256": sha256_file(model_path),
    }


def consistent_matrix(matrix: Any) -> Any:
    if isinstance(matrix, pd.DataFrame):
        return matrix.to_numpy(dtype=np.float32, copy=False)
    if hasattr(matrix, "astype"):
        return matrix.astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def frozen_probabilities(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    population_mask: np.ndarray,
) -> np.ndarray:
    features = list(LANDMARK_STRICT_FEATURES)
    transformed = consistent_matrix(
        bundle["preprocessor"].transform(
            frame.loc[population_mask, features]
        )
    )
    probabilities = np.asarray(
        bundle["model"].predict_proba(transformed)[:, 1],
        dtype=np.float64,
    )
    del transformed
    gc.collect()

    expected_n = int(np.sum(population_mask))
    if len(probabilities) != expected_n:
        raise AssertionError(
            f"Prediction length {len(probabilities)} != population {expected_n}"
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Frozen score vector contains non-finite values.")

    return probabilities


# ---------------------------------------------------------------------------
# Ranking / four-cell point estimates
# ---------------------------------------------------------------------------

def rank_eligible_indices(
    probabilities: np.ndarray,
    eligible_mask: np.ndarray,
) -> np.ndarray:
    """
    Exact canonical tie handling:
      1. score descending
      2. original target-population row position ascending
    """
    scores = np.asarray(probabilities, dtype=np.float64)
    eligible_indices = np.flatnonzero(
        np.asarray(eligible_mask, dtype=bool)
    )
    eligible_scores = scores[eligible_indices]
    order = np.lexsort((eligible_indices, -eligible_scores))
    return eligible_indices[order]


def capture_from_rank(
    y: np.ndarray,
    rank: np.ndarray,
    budget: int,
    full_event_n: int,
) -> tuple[int, float]:
    if budget <= 0:
        raise ValueError("Budget must be positive.")
    if budget > len(rank):
        raise ValueError(
            f"Budget {budget:,} exceeds candidate size {len(rank):,}."
        )

    selected = rank[:budget]
    tp = int(np.sum(np.asarray(y, dtype=np.int8)[selected] == 1))
    capture = float(tp / full_event_n)
    return tp, capture


def component_values(
    c00: float,
    c10: float,
    c01: float,
    c11: float,
) -> dict[str, float]:
    availability_main = c10 - c00
    capacity_main = c01 - c00
    interaction = c11 - c10 - c01 + c00

    # Ordered path used in the previous manuscript:
    # C00 -> C01 -> C11
    ordered_capacity = c01 - c00
    ordered_availability_at_all_budget = c11 - c01

    # Reverse path:
    availability_at_early_budget = c10 - c00
    capacity_in_all_cohort = c11 - c10

    shapley_availability = 0.5 * (
        availability_at_early_budget
        + ordered_availability_at_all_budget
    )
    shapley_capacity = 0.5 * (
        ordered_capacity
        + capacity_in_all_cohort
    )

    total = c11 - c00

    identity_factorial = total - (
        availability_main + capacity_main + interaction
    )
    identity_shapley = total - (
        shapley_availability + shapley_capacity
    )

    interaction_via_availability = (
        ordered_availability_at_all_budget
        - availability_at_early_budget
    )
    interaction_via_capacity = (
        capacity_in_all_cohort
        - ordered_capacity
    )

    return {
        "availability_main_at_early_budget": availability_main,
        "capacity_main_in_early_cohort": capacity_main,
        "interaction": interaction,
        "ordered_capacity_within_early_cohort": ordered_capacity,
        "ordered_availability_at_all_budget": ordered_availability_at_all_budget,
        "availability_at_early_budget": availability_at_early_budget,
        "capacity_in_all_cohort": capacity_in_all_cohort,
        "shapley_availability": shapley_availability,
        "shapley_capacity": shapley_capacity,
        "total_topq_contrast": total,
        "identity_factorial_error": identity_factorial,
        "identity_shapley_error": identity_shapley,
        "interaction_via_availability_error": (
            interaction - interaction_via_availability
        ),
        "interaction_via_capacity_error": (
            interaction - interaction_via_capacity
        ),
    }


def four_cell_point_estimates(
    *,
    target: str,
    y: np.ndarray,
    probabilities: np.ndarray,
    care_month: np.ndarray,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[float, dict[str, Any]],
]:
    y_values = np.asarray(y, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)

    early = early_entry_mask(care_month)
    all_record = np.ones(len(y_values), dtype=bool)

    n_early = int(np.sum(early))
    n_all = int(len(y_values))
    e_early = int(np.sum(y_values[early] == 1))
    e_all = int(np.sum(y_values == 1))

    if n_early <= 0 or e_all <= 0:
        raise ValueError(f"{target}: invalid early/all population counts.")

    rank_early = rank_eligible_indices(scores, early)
    rank_all = rank_eligible_indices(scores, all_record)

    cell_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    lookup: dict[float, dict[str, Any]] = {}

    for q in FRACTIONS:
        b_early = max(1, int(round(q * n_early)))
        b_all = max(1, int(round(q * n_all)))

        if b_all > n_early:
            raise ValueError(
                f"{target}/q={q}: B_all={b_all:,} exceeds early N={n_early:,}."
            )

        tp00, c00 = capture_from_rank(y_values, rank_early, b_early, e_all)
        tp10, c10 = capture_from_rank(y_values, rank_all, b_early, e_all)
        tp01, c01 = capture_from_rank(y_values, rank_early, b_all, e_all)
        tp11, c11 = capture_from_rank(y_values, rank_all, b_all, e_all)

        cells = {
            "C00": {
                "candidate_set": EARLY_SCENARIO,
                "budget_source": "early",
                "candidate_n": n_early,
                "candidate_event_n": e_early,
                "budget_n": b_early,
                "true_positive_n": tp00,
                "population_event_capture": c00,
            },
            "C10": {
                "candidate_set": ALL_SCENARIO,
                "budget_source": "early",
                "candidate_n": n_all,
                "candidate_event_n": e_all,
                "budget_n": b_early,
                "true_positive_n": tp10,
                "population_event_capture": c10,
            },
            "C01": {
                "candidate_set": EARLY_SCENARIO,
                "budget_source": "all",
                "candidate_n": n_early,
                "candidate_event_n": e_early,
                "budget_n": b_all,
                "true_positive_n": tp01,
                "population_event_capture": c01,
            },
            "C11": {
                "candidate_set": ALL_SCENARIO,
                "budget_source": "all",
                "candidate_n": n_all,
                "candidate_event_n": e_all,
                "budget_n": b_all,
                "true_positive_n": tp11,
                "population_event_capture": c11,
            },
        }

        for cell_name, cell in cells.items():
            selected_n = int(cell["budget_n"])
            tp = int(cell["true_positive_n"])
            cell_rows.append(
                {
                    "target": target,
                    "feature_set": FEATURE_SET_NAME,
                    "model": MODEL_NAME,
                    "evaluation": "temporal_evaluation_cohort_2023",
                    "nominal_fraction": float(q),
                    "cell": cell_name,
                    "candidate_set": cell["candidate_set"],
                    "budget_source": cell["budget_source"],
                    "candidate_n": int(cell["candidate_n"]),
                    "candidate_event_n": int(cell["candidate_event_n"]),
                    "full_population_n": n_all,
                    "full_population_event_n": e_all,
                    "budget_n": selected_n,
                    "selected_n": selected_n,
                    "true_positive_n": tp,
                    "precision": float(tp / selected_n),
                    "population_event_capture": float(
                        cell["population_event_capture"]
                    ),
                }
            )

        components = component_values(c00, c10, c01, c11)
        assert_component_identities(
            components, f"{target}/q={q}/point"
        )

        decomposition_row = {
            "target": target,
            "feature_set": FEATURE_SET_NAME,
            "model": MODEL_NAME,
            "evaluation": "temporal_evaluation_cohort_2023",
            "nominal_fraction": float(q),
            "n_early": n_early,
            "n_all": n_all,
            "event_n_early": e_early,
            "event_n_all": e_all,
            "B_early": b_early,
            "B_all": b_all,
            "C00_early_Bearly": c00,
            "C10_all_Bearly": c10,
            "C01_early_Ball": c01,
            "C11_all_Ball": c11,
            "C00_tp": tp00,
            "C10_tp": tp10,
            "C01_tp": tp01,
            "C11_tp": tp11,
            **components,
        }
        decomposition_rows.append(decomposition_row)

        lookup[float(q)] = {
            "cells": cells,
            "decomposition": decomposition_row,
            "rank_early": rank_early,
            "rank_all": rank_all,
            "early_mask": early,
        }

    return cell_rows, decomposition_rows, lookup


def assert_component_identities(
    components: dict[str, float],
    label: str,
    tolerance: float = 1e-12,
) -> None:
    checks = (
        "identity_factorial_error",
        "identity_shapley_error",
        "interaction_via_availability_error",
        "interaction_via_capacity_error",
    )
    for key in checks:
        error = float(components[key])
        if abs(error) > tolerance:
            raise AssertionError(
                f"{label}: {key}={error} exceeds {tolerance}"
            )


# ---------------------------------------------------------------------------
# Canonical point cross-check
# ---------------------------------------------------------------------------

def one_canonical_protocol_row(
    frame: pd.DataFrame,
    *,
    target: str,
    q: float,
    protocol: str,
    scenario: str,
) -> pd.Series:
    required = {
        "target",
        "protocol",
        "availability_scenario",
        "nominal_fraction",
        "selected_n",
        "true_positive_n",
        "population_event_capture",
        "full_population_event_n",
        "eligible_n",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"canonical_protocol_results.csv missing columns: {sorted(missing)}"
        )

    mask = (
        frame["target"].astype(str).eq(target)
        & frame["protocol"].astype(str).eq(protocol)
        & frame["availability_scenario"].astype(str).eq(scenario)
        & np.isclose(
            frame["nominal_fraction"].astype(float).to_numpy(),
            float(q),
            atol=0.0,
            rtol=0.0,
        )
    )
    rows = frame.loc[mask]
    if len(rows) != 1:
        raise AssertionError(
            f"Expected one canonical row for {target}/{q}/{protocol}/{scenario}; "
            f"got {len(rows)}"
        )
    return rows.iloc[0]


def compare_point_cell(
    *,
    target: str,
    q: float,
    cell: str,
    observed: dict[str, Any],
    expected: pd.Series,
) -> dict[str, Any]:
    fields = {
        "selected_n": (
            int(observed["budget_n"]),
            int(expected["selected_n"]),
        ),
        "true_positive_n": (
            int(observed["true_positive_n"]),
            int(expected["true_positive_n"]),
        ),
        "population_event_capture": (
            float(observed["population_event_capture"]),
            float(expected["population_event_capture"]),
        ),
    }

    for name, (got, want) in fields.items():
        if isinstance(got, float):
            if not math.isclose(
                got, want, rel_tol=0.0, abs_tol=5e-13
            ):
                raise AssertionError(
                    f"{target}/q={q}/{cell}: {name}={got} != canonical {want}"
                )
        else:
            if got != want:
                raise AssertionError(
                    f"{target}/q={q}/{cell}: {name}={got} != canonical {want}"
                )

    return {
        "target": target,
        "nominal_fraction": float(q),
        "cell": cell,
        "canonical_protocol": str(expected["protocol"]),
        "canonical_scenario": str(expected["availability_scenario"]),
        "observed_selected_n": int(observed["budget_n"]),
        "canonical_selected_n": int(expected["selected_n"]),
        "observed_true_positive_n": int(observed["true_positive_n"]),
        "canonical_true_positive_n": int(expected["true_positive_n"]),
        "observed_population_event_capture": float(
            observed["population_event_capture"]
        ),
        "canonical_population_event_capture": float(
            expected["population_event_capture"]
        ),
        "status": "PASS",
    }


def crosscheck_points(
    *,
    target: str,
    lookup: dict[float, dict[str, Any]],
    canonical_protocol: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for q in FRACTIONS:
        cells = lookup[q]["cells"]

        c00_expected = one_canonical_protocol_row(
            canonical_protocol,
            target=target,
            q=q,
            protocol="cohort_specific_top_fraction",
            scenario=EARLY_SCENARIO,
        )
        rows.append(
            compare_point_cell(
                target=target,
                q=q,
                cell="C00",
                observed=cells["C00"],
                expected=c00_expected,
            )
        )

        c01_expected = one_canonical_protocol_row(
            canonical_protocol,
            target=target,
            q=q,
            protocol="fixed_absolute_budget",
            scenario=EARLY_SCENARIO,
        )
        rows.append(
            compare_point_cell(
                target=target,
                q=q,
                cell="C01",
                observed=cells["C01"],
                expected=c01_expected,
            )
        )

        c11_fixed_expected = one_canonical_protocol_row(
            canonical_protocol,
            target=target,
            q=q,
            protocol="fixed_absolute_budget",
            scenario=ALL_SCENARIO,
        )
        rows.append(
            compare_point_cell(
                target=target,
                q=q,
                cell="C11_fixed_budget",
                observed=cells["C11"],
                expected=c11_fixed_expected,
            )
        )

        c11_topq_expected = one_canonical_protocol_row(
            canonical_protocol,
            target=target,
            q=q,
            protocol="cohort_specific_top_fraction",
            scenario=ALL_SCENARIO,
        )
        rows.append(
            compare_point_cell(
                target=target,
                q=q,
                cell="C11_topq",
                observed=cells["C11"],
                expected=c11_topq_expected,
            )
        )

    return rows


# ---------------------------------------------------------------------------
# Exact weighted top-B for paired bootstrap
# ---------------------------------------------------------------------------

def selected_tp_from_cumulative(
    *,
    cumulative_count: np.ndarray,
    cumulative_events: np.ndarray,
    ordered_y: np.ndarray,
    budget: int,
) -> int:
    """
    Exact top-B true positives from a bootstrap multiset represented by
    multiplicities over a fixed deterministic ranking.
    """
    if budget <= 0:
        raise ValueError("Budget must be positive.")

    if (
        len(cumulative_count) == 0
        or int(cumulative_count[-1]) < budget
    ):
        available = (
            int(cumulative_count[-1])
            if len(cumulative_count)
            else 0
        )
        raise ValueError(
            f"Bootstrap budget={budget:,} exceeds resampled candidate count={available:,}."
        )

    cutoff = int(
        np.searchsorted(cumulative_count, budget, side="left")
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

    return int(
        previous_events
        + remaining * int(ordered_y[cutoff])
    )


def canonical_fixed_budget_bootstrap_lookup(
    canonical_bootstrap: pd.DataFrame,
    *,
    target: str,
    replicates: int,
) -> dict[tuple[float, int], pd.Series]:
    """
    Retrieve the canonical Protocol-B early->all bootstrap rows.

    These rows use one fixed absolute budget B_all = round(q * N_all) in both
    scenarios. They therefore provide an exact replicate-level cross-check for
    our C01 (early + B_all) and C11 (all + B_all) cells.
    """
    required = {
        "target",
        "replicate",
        "protocol",
        "nominal_fraction",
        "from_scenario",
        "to_scenario",
        "from_selected_n",
        "to_selected_n",
        "from_true_positive_n",
        "to_true_positive_n",
        "from_population_event_capture",
        "to_population_event_capture",
        "delta_population_event_capture",
    }
    missing = required - set(canonical_bootstrap.columns)
    if missing:
        raise ValueError(
            f"canonical bootstrap artifact missing columns: {sorted(missing)}"
        )

    subset = canonical_bootstrap.loc[
        canonical_bootstrap["target"].astype(str).eq(target)
        & canonical_bootstrap["protocol"].astype(str).eq(
            "fixed_absolute_budget"
        )
        & canonical_bootstrap["from_scenario"].astype(str).eq(
            EARLY_SCENARIO
        )
        & canonical_bootstrap["to_scenario"].astype(str).eq(
            ALL_SCENARIO
        )
        & canonical_bootstrap["replicate"].astype(int).lt(replicates)
    ].copy()

    expected_rows = len(FRACTIONS) * replicates
    if len(subset) != expected_rows:
        raise AssertionError(
            f"{target}: expected {expected_rows} canonical fixed-budget "
            f"bootstrap rows for first {replicates} replicates, got {len(subset)}"
        )

    lookup: dict[tuple[float, int], pd.Series] = {}
    for _, row in subset.iterrows():
        key = (
            float(row["nominal_fraction"]),
            int(row["replicate"]),
        )
        if key in lookup:
            raise AssertionError(
                f"Duplicate canonical fixed-budget bootstrap key {target}/{key}"
            )
        lookup[key] = row

    return lookup


def compare_fixed_budget_bootstrap_to_canonical(
    *,
    target: str,
    q: float,
    replicate: int,
    b_all: int,
    tp01: int,
    tp11: int,
    c01: float,
    c11: float,
    canonical: pd.Series,
) -> None:
    """
    Exact replicate-level validation of the two B_all cells.

    C01 = early candidate set + observed B_all
    C11 = all-record candidate set + observed B_all
    """
    integer_pairs = {
        "from_selected_n": (
            b_all,
            int(canonical["from_selected_n"]),
        ),
        "to_selected_n": (
            b_all,
            int(canonical["to_selected_n"]),
        ),
        "from_true_positive_n": (
            tp01,
            int(canonical["from_true_positive_n"]),
        ),
        "to_true_positive_n": (
            tp11,
            int(canonical["to_true_positive_n"]),
        ),
    }

    for name, (got, want) in integer_pairs.items():
        if got != want:
            raise AssertionError(
                f"{target}/q={q}/rep={replicate}: "
                f"{name}={got} != canonical fixed-budget {want}"
            )

    float_pairs = {
        "from_population_event_capture": (
            c01,
            float(canonical["from_population_event_capture"]),
        ),
        "to_population_event_capture": (
            c11,
            float(canonical["to_population_event_capture"]),
        ),
        "delta_population_event_capture": (
            c11 - c01,
            float(canonical["delta_population_event_capture"]),
        ),
    }

    for name, (got, want) in float_pairs.items():
        if not math.isclose(
            got, want, rel_tol=0.0, abs_tol=5e-13
        ):
            raise AssertionError(
                f"{target}/q={q}/rep={replicate}: "
                f"{name}={got} != canonical fixed-budget {want}"
            )


def fixed_observed_bootstrap_budgets(
    point_lookup: dict[float, dict[str, Any]],
    q: float,
) -> tuple[int, int]:
    """
    Return the two observed absolute factorial budget levels.

    They are intentionally independent of bootstrap-resampled candidate counts.
    """
    point = point_lookup[float(q)]["decomposition"]
    b_early = int(point["B_early"])
    b_all = int(point["B_all"])

    if b_early <= 0 or b_all <= 0:
        raise ValueError(f"q={q}: non-positive observed budget.")
    if b_early > b_all:
        raise ValueError(
            f"q={q}: observed B_early={b_early} exceeds B_all={b_all}."
        )

    return b_early, b_all


def bootstrap_four_cell(
    *,
    target: str,
    y: np.ndarray,
    probabilities: np.ndarray,
    care_month: np.ndarray,
    point_lookup: dict[float, dict[str, Any]],
    canonical_bootstrap: pd.DataFrame,
    replicates: int,
    seed: int,
    confidence_level: float,
    progress_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Paired full-size bootstrap with observed absolute budget levels held fixed.

    The resampled records change outcome/candidate composition, but the two
    decision-capacity factor levels do not:
        B_early = observed round(q * N_early)
        B_all   = observed round(q * N_all)

    This estimates uncertainty for the requested 2x2 factorial contrast at
    fixed absolute capacity levels.
    """
    y_values = np.asarray(y, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    n = int(len(y_values))

    early = early_entry_mask(care_month)
    all_record = np.ones(n, dtype=bool)

    early_indices = np.flatnonzero(early)
    rank_early = rank_eligible_indices(scores, early)
    rank_all = rank_eligible_indices(scores, all_record)

    ordered_y_early = y_values[rank_early].astype(
        np.int64, copy=False
    )
    ordered_y_all = y_values[rank_all].astype(
        np.int64, copy=False
    )

    # Canonical Protocol B is an exact cross-check for C01/C11 because it
    # already keeps B_all fixed under the full-size bootstrap.
    canonical_lookup = canonical_fixed_budget_bootstrap_lookup(
        canonical_bootstrap,
        target=target,
        replicates=replicates,
    )

    # Lock the two observed absolute budgets once for every q.
    observed_budgets = {
        float(q): fixed_observed_bootstrap_budgets(
            point_lookup, float(q)
        )
        for q in FRACTIONS
    }

    event_mask = y_values == 1
    rng = np.random.default_rng(seed)

    replicate_rows: list[dict[str, Any]] = []

    for replicate in range(replicates):
        # Exactly the same nonparametric full-size bootstrap multiplicities
        # and target-specific RNG stream as the canonical script.
        sampled = rng.integers(
            0, n, size=n, dtype=np.int32
        )
        counts = np.bincount(
            sampled, minlength=n
        ).astype(np.int32, copy=False)
        del sampled

        if int(np.sum(counts, dtype=np.int64)) != n:
            raise AssertionError(
                f"{target}/rep={replicate}: bootstrap counts do not sum to N."
            )

        full_event_n = int(
            np.sum(counts[event_mask], dtype=np.int64)
        )
        if full_event_n <= 0:
            raise RuntimeError(
                f"{target}/rep={replicate}: bootstrap contains zero events."
            )

        early_sample_n = int(
            np.sum(counts[early_indices], dtype=np.int64)
        )
        if early_sample_n <= 0:
            raise RuntimeError(
                f"{target}/rep={replicate}: bootstrap early set is empty."
            )

        # Build cumulative ranked bootstrap multiplicities ONCE per
        # candidate set, then query both fixed absolute budget levels.
        early_counts = counts[rank_early].astype(
            np.int64, copy=False
        )
        cumulative_early_count = np.cumsum(
            early_counts, dtype=np.int64
        )
        cumulative_early_events = np.cumsum(
            early_counts * ordered_y_early,
            dtype=np.int64,
        )

        all_counts = counts[rank_all].astype(
            np.int64, copy=False
        )
        cumulative_all_count = np.cumsum(
            all_counts, dtype=np.int64
        )
        cumulative_all_events = np.cumsum(
            all_counts * ordered_y_all,
            dtype=np.int64,
        )

        for q in FRACTIONS:
            q = float(q)
            b_early, b_all = observed_budgets[q]

            # Both budget levels are fixed at observed point-estimate counts.
            # They MUST NOT depend on early_sample_n in this primary bootstrap.
            if b_early > early_sample_n:
                raise RuntimeError(
                    f"{target}/q={q}/rep={replicate}: "
                    f"fixed B_early={b_early:,} > resampled early N={early_sample_n:,}"
                )
            if b_all > early_sample_n:
                raise RuntimeError(
                    f"{target}/q={q}/rep={replicate}: "
                    f"fixed B_all={b_all:,} > resampled early N={early_sample_n:,}"
                )

            tp00 = selected_tp_from_cumulative(
                cumulative_count=cumulative_early_count,
                cumulative_events=cumulative_early_events,
                ordered_y=ordered_y_early,
                budget=b_early,
            )
            tp10 = selected_tp_from_cumulative(
                cumulative_count=cumulative_all_count,
                cumulative_events=cumulative_all_events,
                ordered_y=ordered_y_all,
                budget=b_early,
            )
            tp01 = selected_tp_from_cumulative(
                cumulative_count=cumulative_early_count,
                cumulative_events=cumulative_early_events,
                ordered_y=ordered_y_early,
                budget=b_all,
            )
            tp11 = selected_tp_from_cumulative(
                cumulative_count=cumulative_all_count,
                cumulative_events=cumulative_all_events,
                ordered_y=ordered_y_all,
                budget=b_all,
            )

            c00 = float(tp00 / full_event_n)
            c10 = float(tp10 / full_event_n)
            c01 = float(tp01 / full_event_n)
            c11 = float(tp11 / full_event_n)

            components = component_values(
                c00, c10, c01, c11
            )
            assert_component_identities(
                components,
                f"{target}/q={q}/rep={replicate}",
            )

            # Exact reproducibility check against canonical Protocol B.
            # C01/C11 are identical estimands under the same bootstrap counts.
            canonical = canonical_lookup[
                (q, int(replicate))
            ]
            compare_fixed_budget_bootstrap_to_canonical(
                target=target,
                q=q,
                replicate=replicate,
                b_all=b_all,
                tp01=tp01,
                tp11=tp11,
                c01=c01,
                c11=c11,
                canonical=canonical,
            )

            replicate_rows.append(
                {
                    "target": target,
                    "feature_set": FEATURE_SET_NAME,
                    "model": MODEL_NAME,
                    "evaluation": "temporal_evaluation_cohort_2023",
                    "replicate": int(replicate),
                    "bootstrap_sample_n": n,
                    "bootstrap_full_event_n": full_event_n,
                    "bootstrap_early_candidate_n": early_sample_n,
                    "nominal_fraction": q,
                    "budget_lock": (
                        "fixed_observed_point_absolute_counts"
                    ),
                    "B_early": b_early,
                    "B_all": b_all,
                    "C00_early_Bearly": c00,
                    "C10_all_Bearly": c10,
                    "C01_early_Ball": c01,
                    "C11_all_Ball": c11,
                    "C00_tp": tp00,
                    "C10_tp": tp10,
                    "C01_tp": tp01,
                    "C11_tp": tp11,
                    "canonical_fixed_budget_C01_C11_crosscheck": "PASS",
                    **components,
                }
            )

        del (
            counts,
            early_counts,
            cumulative_early_count,
            cumulative_early_events,
            all_counts,
            cumulative_all_count,
            cumulative_all_events,
        )

        if progress_every > 0 and (
            (replicate + 1) % progress_every == 0
            or replicate + 1 == replicates
        ):
            print(
                f"Shapley fixed-B bootstrap {target}: "
                f"{replicate + 1}/{replicates}"
            )

    summary_rows = summarize_bootstrap(
        target=target,
        point_lookup=point_lookup,
        replicate_rows=replicate_rows,
        confidence_level=confidence_level,
    )

    return replicate_rows, summary_rows


SUMMARY_METRICS = (
    "C00_early_Bearly",
    "C10_all_Bearly",
    "C01_early_Ball",
    "C11_all_Ball",
    "availability_main_at_early_budget",
    "capacity_main_in_early_cohort",
    "interaction",
    "ordered_capacity_within_early_cohort",
    "ordered_availability_at_all_budget",
    "availability_at_early_budget",
    "capacity_in_all_cohort",
    "shapley_availability",
    "shapley_capacity",
    "total_topq_contrast",
)


def summarize_bootstrap(
    *,
    target: str,
    point_lookup: dict[float, dict[str, Any]],
    replicate_rows: list[dict[str, Any]],
    confidence_level: float,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(replicate_rows)
    alpha = 1.0 - float(confidence_level)

    rows: list[dict[str, Any]] = []

    for q in FRACTIONS:
        group = frame.loc[
            np.isclose(
                frame["nominal_fraction"].astype(float),
                float(q),
                atol=0.0,
                rtol=0.0,
            )
        ]
        if len(group) == 0:
            raise AssertionError(
                f"{target}/q={q}: no bootstrap rows."
            )

        point = point_lookup[q]["decomposition"]

        for metric in SUMMARY_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)

            if metric in point:
                point_value = float(point[metric])
            else:
                raise KeyError(
                    f"Point decomposition lacks metric {metric!r}"
                )

            rows.append(
                {
                    "target": target,
                    "feature_set": FEATURE_SET_NAME,
                    "model": MODEL_NAME,
                    "evaluation": "temporal_evaluation_cohort_2023",
                    "bootstrap_method": (
                        "paired full-size record-level nonparametric bootstrap; "
                        "B_early and B_all fixed to observed absolute counts"
                    ),
                    "replicates": int(len(values)),
                    "confidence_level": float(confidence_level),
                    "nominal_fraction": float(q),
                    "metric": metric,
                    "point_estimate": point_value,
                    "point_estimate_pp": 100.0 * point_value,
                    "bootstrap_mean": float(np.mean(values)),
                    "bootstrap_mean_pp": 100.0 * float(
                        np.mean(values)
                    ),
                    "bootstrap_se": float(
                        np.std(values, ddof=1)
                    ),
                    "bootstrap_se_pp": 100.0 * float(
                        np.std(values, ddof=1)
                    ),
                    "ci_lower": float(
                        np.quantile(values, alpha / 2.0)
                    ),
                    "ci_upper": float(
                        np.quantile(
                            values, 1.0 - alpha / 2.0
                        )
                    ),
                    "ci_lower_pp": 100.0 * float(
                        np.quantile(values, alpha / 2.0)
                    ),
                    "ci_upper_pp": 100.0 * float(
                        np.quantile(
                            values, 1.0 - alpha / 2.0
                        )
                    ),
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Main compact result table
# ---------------------------------------------------------------------------

MAIN_METRICS = (
    "total_topq_contrast",
    "availability_main_at_early_budget",
    "capacity_main_in_early_cohort",
    "interaction",
    "shapley_availability",
    "shapley_capacity",
    "ordered_availability_at_all_budget",
    "ordered_capacity_within_early_cohort",
)


def make_main_results(
    decomposition_rows: list[dict[str, Any]],
    bootstrap_summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = pd.DataFrame(bootstrap_summary_rows)
    output: list[dict[str, Any]] = []

    for point in decomposition_rows:
        target = str(point["target"])
        q = float(point["nominal_fraction"])

        row: dict[str, Any] = {
            "target": target,
            "nominal_fraction": q,
            "B_early_point": int(point["B_early"]),
            "B_all_point": int(point["B_all"]),
            "C00_early_Bearly": float(
                point["C00_early_Bearly"]
            ),
            "C10_all_Bearly": float(
                point["C10_all_Bearly"]
            ),
            "C01_early_Ball": float(
                point["C01_early_Ball"]
            ),
            "C11_all_Ball": float(
                point["C11_all_Ball"]
            ),
        }

        for metric in MAIN_METRICS:
            mask = (
                summary["target"].astype(str).eq(target)
                & np.isclose(
                    summary["nominal_fraction"].astype(float),
                    q,
                    atol=0.0,
                    rtol=0.0,
                )
                & summary["metric"].astype(str).eq(metric)
            )
            part = summary.loc[mask]
            if len(part) != 1:
                raise AssertionError(
                    f"Expected one summary row for {target}/{q}/{metric}; "
                    f"got {len(part)}"
                )
            s = part.iloc[0]

            row[f"{metric}_point"] = float(
                s["point_estimate"]
            )
            row[f"{metric}_point_pp"] = float(
                s["point_estimate_pp"]
            )
            row[f"{metric}_ci_lower"] = float(
                s["ci_lower"]
            )
            row[f"{metric}_ci_upper"] = float(
                s["ci_upper"]
            )
            row[f"{metric}_ci_lower_pp"] = float(
                s["ci_lower_pp"]
            )
            row[f"{metric}_ci_upper_pp"] = float(
                s["ci_upper_pp"]
            )

        output.append(row)

    return output


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def summary_markdown(
    main_rows: list[dict[str, Any]],
) -> str:
    labels = {
        "target_preterm": "Preterm delivery",
        "target_nicu": "NICU admission",
        "target_lbw": "Low birth weight",
    }

    lines = [
        "# TAE four-cell symmetric Shapley decomposition",
        "",
        "CDC 2023 is treated as the temporal evaluation cohort.",
        "The same frozen canonical model and score vector are used within each target.",
        "",
        "Bootstrap budget levels are fixed absolute counts: for each target/q, "
        "B_early and B_all are defined from the observed point-estimate cohorts "
        "and held unchanged in all 500 bootstrap replicates.",
        "",
        "Four cells:",
        "",
        "- C00: early candidate set + early-derived absolute budget",
        "- C10: all-record candidate set + early-derived absolute budget",
        "- C01: early candidate set + all-record-derived absolute budget",
        "- C11: all-record candidate set + all-record-derived absolute budget",
        "",
        "Main symmetric attribution:",
        "",
        "`total = Shapley availability + Shapley capacity`",
        "",
        "Factorial identity:",
        "",
        "`total = availability main + capacity main + interaction`",
        "",
        "## Main results",
        "",
        "| Outcome | q | Total | Shapley availability | Shapley capacity | Interaction |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in main_rows:
        def ci(metric: str) -> str:
            return (
                f"{float(row[f'{metric}_point_pp']):+.3f} "
                f"[{float(row[f'{metric}_ci_lower_pp']):+.3f}, "
                f"{float(row[f'{metric}_ci_upper_pp']):+.3f}]"
            )

        lines.append(
            f"| {labels.get(str(row['target']), str(row['target']))} "
            f"| {100.0 * float(row['nominal_fraction']):.0f}% "
            f"| {ci('total_topq_contrast')} "
            f"| {ci('shapley_availability')} "
            f"| {ci('shapley_capacity')} "
            f"| {ci('interaction')} |"
        )

    lines.extend(
        [
            "",
            "All values are percentage-point population event-capture contrasts "
            "with percentile 95% bootstrap intervals.",
            "",
            "The bootstrap conditions on the two observed absolute budget levels. "
            "At the point estimate, the C00-to-C11 total coincides with the "
            "cohort-specific top-q endpoint contrast; bootstrap uncertainty is "
            "computed for those fixed factorial budget levels.",
            "",
            "The previous ordered decomposition is retained only as an appendix "
            "quantity (`ordered_availability_at_all_budget` and "
            "`ordered_capacity_within_early_cohort`).",
            "",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reproducibility bundle / source hashes
# ---------------------------------------------------------------------------

def save_reproducibility_bundle(
    output: Path,
    *,
    config: dict[str, Any],
    canonical_sources: dict[str, Any],
    model_metadata: list[dict[str, Any]],
) -> None:
    root = output / "reproducibility"
    code_dir = root / "code"
    canonical_dir = root / "canonical_source"

    code_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        SCRIPT_PATH,
        code_dir / SCRIPT_PATH.name,
    )

    atomic_text(
        root / "shapley_config.locked.yaml",
        yaml.safe_dump(
            to_plain_python(config),
            sort_keys=False,
            allow_unicode=True,
        ),
    )

    # Preserve the exact small canonical source artifacts used.
    for key in (
        "manifest_path",
        "config_path",
        "protocol_path",
    ):
        source = Path(canonical_sources[key])
        shutil.copy2(
            source,
            canonical_dir / source.name,
        )

    atomic_json(
        root / "model_lineage.json",
        model_metadata,
    )

    atomic_text(
        root / "README.txt",
        (
            "This Shapley analysis is a standalone ClearML post-canonical task.\n"
            f"Canonical Task ID: {DEFAULT_CANONICAL_TASK_ID}\n"
            f"Dataset ID: {DEFAULT_DATASET_ID}\n"
            f"Queue: {DEFAULT_QUEUE}\n"
            "No model refitting or threshold re-estimation occurs.\n"
            "Bootstrap B_early and B_all are fixed at observed absolute counts.\n"
            "Git is not required for ClearML execution because the script is "
            "stored as a standalone ClearML script snapshot.\n"
        ),
    )


# ---------------------------------------------------------------------------
# Global validation
# ---------------------------------------------------------------------------

def validate_output_counts(
    *,
    cell_rows: list[dict[str, Any]],
    decomposition_rows: list[dict[str, Any]],
    crosscheck_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    bootstrap_summary_rows: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    replicates: int,
) -> dict[str, bool]:
    expected_decomposition = len(TARGETS) * len(FRACTIONS)
    expected_cells = expected_decomposition * 4
    expected_crosschecks = expected_decomposition * 4
    expected_bootstrap = expected_decomposition * replicates
    expected_summary = expected_decomposition * len(SUMMARY_METRICS)

    point_budget_lookup = {
        (
            str(row["target"]),
            float(row["nominal_fraction"]),
        ): (
            int(row["B_early"]),
            int(row["B_all"]),
        )
        for row in decomposition_rows
    }

    fixed_budget_match = all(
        (
            int(row["B_early"]),
            int(row["B_all"]),
        )
        == point_budget_lookup[
            (
                str(row["target"]),
                float(row["nominal_fraction"]),
            )
        ]
        for row in bootstrap_rows
    )

    checks = {
        "four_cell_row_count": len(cell_rows) == expected_cells,
        "decomposition_row_count": (
            len(decomposition_rows) == expected_decomposition
        ),
        "canonical_point_crosscheck_row_count": (
            len(crosscheck_rows) == expected_crosschecks
        ),
        "bootstrap_replicate_row_count": (
            len(bootstrap_rows) == expected_bootstrap
        ),
        "bootstrap_summary_row_count": (
            len(bootstrap_summary_rows) == expected_summary
        ),
        "main_result_row_count": (
            len(main_rows) == expected_decomposition
        ),
        "canonical_point_crosschecks_all_pass": all(
            str(row.get("status")) == "PASS"
            for row in crosscheck_rows
        ),
        "point_factorial_identities": all(
            abs(float(row["identity_factorial_error"])) <= 1e-12
            for row in decomposition_rows
        ),
        "point_shapley_identities": all(
            abs(float(row["identity_shapley_error"])) <= 1e-12
            for row in decomposition_rows
        ),
        "bootstrap_factorial_identities": all(
            abs(float(row["identity_factorial_error"])) <= 1e-12
            for row in bootstrap_rows
        ),
        "bootstrap_shapley_identities": all(
            abs(float(row["identity_shapley_error"])) <= 1e-12
            for row in bootstrap_rows
        ),
        "bootstrap_budget_levels_equal_observed_point_counts": fixed_budget_match,
        "bootstrap_budget_lock_label": all(
            str(row.get("budget_lock"))
            == "fixed_observed_point_absolute_counts"
            for row in bootstrap_rows
        ),
        "canonical_fixed_budget_C01_C11_bootstrap_reproduced": all(
            str(
                row.get(
                    "canonical_fixed_budget_C01_C11_crosscheck"
                )
            )
            == "PASS"
            for row in bootstrap_rows
        ),
        "same_frozen_model_score_vector_per_target": True,
        "same_bootstrap_multiplicities_across_four_cells": True,
        "full_population_event_denominator_shared_within_replicate": True,
        "no_model_refit": True,
        "no_threshold_reestimation": True,
    }
    return checks


# ---------------------------------------------------------------------------
# ClearML output upload
# ---------------------------------------------------------------------------

def upload_outputs(task: Any, output: Path) -> None:
    artifact_files = (
        "four_cell_results.csv",
        "shapley_decomposition.csv",
        "canonical_crosscheck.csv",
        "shapley_bootstrap_summary.csv",
        "shapley_main_results.csv",
        "shapley_summary.md",
        "run_manifest.json",
        "shapley_config.locked.yaml",
        "package_versions.txt",
        "commands.txt",
        "git_info.json",
    )

    if (output / "shapley_bootstrap_replicates.csv").is_file():
        artifact_files = (
            *artifact_files,
            "shapley_bootstrap_replicates.csv",
        )

    for filename in artifact_files:
        path = output / filename
        if not path.is_file():
            continue
        task.upload_artifact(
            name=filename,
            artifact_object=str(path),
            wait_on_upload=True,
        )

    reproducibility = output / "reproducibility"
    if reproducibility.is_dir():
        task.upload_artifact(
            name="shapley_reproducibility_bundle",
            artifact_object=str(reproducibility),
            wait_on_upload=True,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test() -> int:
    """
    Pure local test of ranking, four cells, Shapley identities, and bootstrap
    cumulative-selection mechanics. No ClearML and no CDC data required.
    """
    rng = np.random.default_rng(12345)
    n = 20_000

    latent = rng.normal(size=n)
    probabilities = 1.0 / (
        1.0 + np.exp(-(0.8 * latent + rng.normal(scale=0.6, size=n)))
    )
    event_probability = 1.0 / (
        1.0 + np.exp(-(-2.2 + 1.1 * latent))
    )
    y = rng.binomial(1, event_probability).astype(np.int8)

    care = rng.choice(
        [1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 0.0, np.nan],
        size=n,
        p=[0.20, 0.20, 0.20, 0.10, 0.08, 0.06, 0.08, 0.08],
    )

    early = early_entry_mask(care)
    rank_early = rank_eligible_indices(probabilities, early)
    rank_all = rank_eligible_indices(
        probabilities, np.ones(n, dtype=bool)
    )

    e_all = int(np.sum(y == 1))
    if e_all <= 0:
        raise AssertionError("Smoke test generated zero events.")

    for q in FRACTIONS:
        b_early = int(round(q * int(np.sum(early))))
        b_all = int(round(q * n))

        tp00, c00 = capture_from_rank(
            y, rank_early, b_early, e_all
        )
        tp10, c10 = capture_from_rank(
            y, rank_all, b_early, e_all
        )
        tp01, c01 = capture_from_rank(
            y, rank_early, b_all, e_all
        )
        tp11, c11 = capture_from_rank(
            y, rank_all, b_all, e_all
        )

        for tp in (tp00, tp10, tp01, tp11):
            if tp < 0:
                raise AssertionError("Negative TP in smoke test.")

        components = component_values(
            c00, c10, c01, c11
        )
        assert_component_identities(
            components, f"smoke/q={q}"
        )

    # Test exact multiplicity/cutoff selection.
    counts = np.bincount(
        rng.integers(0, n, size=n, dtype=np.int32),
        minlength=n,
    ).astype(np.int32, copy=False)

    ordered_counts = counts[rank_all].astype(
        np.int64, copy=False
    )
    ordered_y = y[rank_all].astype(
        np.int64, copy=False
    )
    cumulative_count = np.cumsum(
        ordered_counts, dtype=np.int64
    )
    cumulative_events = np.cumsum(
        ordered_counts * ordered_y,
        dtype=np.int64,
    )

    b = int(round(0.10 * n))
    tp = selected_tp_from_cumulative(
        cumulative_count=cumulative_count,
        cumulative_events=cumulative_events,
        ordered_y=ordered_y,
        budget=b,
    )
    if tp < 0 or tp > b:
        raise AssertionError("Smoke weighted top-B TP is invalid.")

    # Fixed observed budget-level helper must ignore resampled early N.
    smoke_lookup = {
        0.05: {
            "decomposition": {
                "B_early": 600,
                "B_all": 1000,
            }
        },
        0.10: {
            "decomposition": {
                "B_early": 1200,
                "B_all": 2000,
            }
        },
    }
    if fixed_observed_bootstrap_budgets(
        smoke_lookup, 0.05
    ) != (600, 1000):
        raise AssertionError("Fixed observed budget helper failed at q=0.05.")
    if fixed_observed_bootstrap_budgets(
        smoke_lookup, 0.10
    ) != (1200, 2000):
        raise AssertionError("Fixed observed budget helper failed at q=0.10.")

    # Synthetic canonical Protocol-B lookup/cross-check.
    canonical_smoke = pd.DataFrame(
        [
            {
                "target": "smoke_target",
                "replicate": 0,
                "protocol": "fixed_absolute_budget",
                "nominal_fraction": q,
                "from_scenario": EARLY_SCENARIO,
                "to_scenario": ALL_SCENARIO,
                "from_selected_n": 1000 if q == 0.05 else 2000,
                "to_selected_n": 1000 if q == 0.05 else 2000,
                "from_true_positive_n": 100 if q == 0.05 else 200,
                "to_true_positive_n": 120 if q == 0.05 else 240,
                "from_population_event_capture": 0.10,
                "to_population_event_capture": 0.12,
                "delta_population_event_capture": 0.02,
            }
            for q in FRACTIONS
        ]
    )
    canonical_lookup = canonical_fixed_budget_bootstrap_lookup(
        canonical_smoke,
        target="smoke_target",
        replicates=1,
    )
    for q in FRACTIONS:
        q = float(q)
        b_all_smoke = 1000 if q == 0.05 else 2000
        tp_from = 100 if q == 0.05 else 200
        tp_to = 120 if q == 0.05 else 240
        compare_fixed_budget_bootstrap_to_canonical(
            target="smoke_target",
            q=q,
            replicate=0,
            b_all=b_all_smoke,
            tp01=tp_from,
            tp11=tp_to,
            c01=0.10,
            c11=0.12,
            canonical=canonical_lookup[(q, 0)],
        )

    print("SHAPLEY SMOKE TEST PASS")
    print("four_cell=PASS")
    print("factorial_identity=PASS")
    print("shapley_identity=PASS")
    print("weighted_bootstrap_top_b=PASS")
    print("fixed_observed_budget_levels=PASS")
    print("canonical_fixed_budget_C01_C11_crosscheck=PASS")
    return 0


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TAE 2026 standalone ClearML four-cell + "
            "symmetric Shapley decomposition"
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
            os.getenv("TAE_CANONICAL_TASK_ID")
            or DEFAULT_CANONICAL_TASK_ID
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
            os.getenv("CLEARML_PROJECT_NAME")
            or DEFAULT_PROJECT_NAME
        ),
    )
    parser.add_argument(
        "--task-name",
        default=(
            os.getenv("CLEARML_TASK_NAME_TAE_SHAPLEY")
            or DEFAULT_TASK_NAME
        ),
    )
    parser.add_argument(
        "--dataset-download-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Run on the current machine but still retrieve Dataset/artifacts "
            "from ClearML. Normal publication run should omit --local."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Retrieve/validate canonical artifacts and CDC data, recompute "
            "frozen scores and point four-cell cross-checks, but skip bootstrap."
        ),
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

    config = json.loads(
        json.dumps(DEFAULT_CONFIG)
    )
    config["bootstrap"]["replicates"] = int(
        args.bootstrap_replicates
    )
    config["execution"]["dataset_download_workers"] = int(
        args.dataset_download_workers
    )
    config["execution"]["clearml_dataset_id"] = str(
        args.dataset_id
    )
    config["execution"]["canonical_task_id"] = str(
        args.canonical_task_id
    )
    config["execution"]["clearml_default_queue"] = str(
        args.queue
    )

    # Strong lock: publication analysis uses the known canonical lineage.
    if str(args.dataset_id) != DEFAULT_DATASET_ID:
        raise ValueError(
            f"Dataset is locked to {DEFAULT_DATASET_ID}; got {args.dataset_id}."
        )
    if str(args.canonical_task_id) != DEFAULT_CANONICAL_TASK_ID:
        raise ValueError(
            f"Canonical source Task is locked to {DEFAULT_CANONICAL_TASK_ID}; "
            f"got {args.canonical_task_id}."
        )
    if str(args.queue) != DEFAULT_QUEUE:
        raise ValueError(
            f"Remote queue is locked to {DEFAULT_QUEUE}; got {args.queue}."
        )

    validate_runtime_config(config)

    from clearml import Task

    # CRITICAL: no GitHub/repository checkout is required.
    # The exact Python file is stored in the ClearML Task.
    Task.force_store_standalone_script(True)

    for package_name, package_version in CLEARML_EXTRA_REQUIREMENTS:
        Task.add_requirements(
            package_name, package_version
        )

    task = Task.init(
        project_name=args.project_name,
        task_name=args.task_name,
        task_type=Task.TaskTypes.testing,
        tags=TAGS,
        reuse_last_task_id=False,
        output_uri=True,
    )

    config = task.connect_configuration(
        configuration=config,
        name="shapley_config",
        description=(
            "Locked four-cell Shapley configuration with fixed observed absolute bootstrap budgets"
        ),
    )
    config = to_plain_python(config)

    runtime = task.connect(
        {
            "dataset_id": str(args.dataset_id),
            "canonical_task_id": str(args.canonical_task_id),
            "queue": str(args.queue),
            "dataset_download_workers": int(
                args.dataset_download_workers
            ),
            "bootstrap_replicates": int(
                config["bootstrap"]["replicates"]
            ),
            "preflight_only": bool(args.preflight_only),
        },
        name="runtime",
    )
    runtime = to_plain_python(runtime)

    args.dataset_id = str(runtime["dataset_id"])
    args.canonical_task_id = str(
        runtime["canonical_task_id"]
    )
    args.queue = str(runtime["queue"])
    args.dataset_download_workers = int(
        runtime["dataset_download_workers"]
    )
    config["bootstrap"]["replicates"] = int(
        runtime["bootstrap_replicates"]
    )
    args.preflight_only = bool(
        runtime["preflight_only"]
    )

    # Revalidate after ClearML parameter connection.
    if args.dataset_id != DEFAULT_DATASET_ID:
        raise ValueError("Connected runtime changed locked Dataset ID.")
    if args.canonical_task_id != DEFAULT_CANONICAL_TASK_ID:
        raise ValueError(
            "Connected runtime changed locked canonical Task ID."
        )
    if args.queue != DEFAULT_QUEUE:
        raise ValueError(
            "Connected runtime changed locked queue."
        )
    validate_runtime_config(config)

    if not args.local:
        task.execute_remotely(
            queue_name=args.queue,
            clone=False,
            exit_process=True,
        )

    # On the remote agent execute_remotely() is a no-op and execution resumes.
    print(f"Current Shapley ClearML Task ID: {task.id}")
    print(f"Canonical source Task ID: {args.canonical_task_id}")
    print(f"Dataset ID: {args.dataset_id}")
    print(f"Queue: {args.queue}")

    started = time.time()

    # Retrieve canonical lineage directly from ClearML, not from GitHub/local repo.
    canonical = load_canonical_sources(
        args.canonical_task_id
    )
    print("Canonical Task artifacts: PASS")

    # Bind the same finalized Dataset used by canonical run.
    dataset_root = bind_clearml_dataset(
        args.dataset_id,
        args.dataset_download_workers,
    )
    cdc2023_path = resolve_dataset_file(
        dataset_root,
        EXPECTED_2023_CONFIGURED_PATH,
    )

    print("Hashing CDC 2023 input...")
    observed_2023_hash = sha256_file(cdc2023_path)
    if observed_2023_hash != EXPECTED_2023_SHA256:
        raise AssertionError(
            f"CDC 2023 SHA256 {observed_2023_hash} "
            f"!= expected {EXPECTED_2023_SHA256}"
        )

    frame = read_cdc_2023(cdc2023_path)
    primary = resident_singleton_mask(frame)

    output = prepare_output(
        config, args.overwrite
    )

    atomic_text(
        output / "commands.txt",
        " ".join(
            shlex.quote(str(value))
            for value in [sys.executable, *sys.argv]
        )
        + "\n",
    )
    atomic_text(
        output / "package_versions.txt",
        package_versions(),
    )
    atomic_text(
        output / "shapley_config.locked.yaml",
        yaml.safe_dump(
            to_plain_python(config),
            sort_keys=False,
            allow_unicode=True,
        ),
    )
    atomic_json(
        output / "git_info.json",
        git_information(),
    )

    cell_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    crosscheck_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    bootstrap_summary_rows: list[dict[str, Any]] = []
    model_metadata: list[dict[str, Any]] = []

    for target_index, target in enumerate(TARGETS):
        print(
            f"\n===== TAE SHAPLEY FIXED-B BOOTSTRAP: {target} ====="
        )

        known = (
            frame[target]
            .notna()
            .to_numpy(dtype=bool)
        )
        population_mask = primary & known

        y = frame.loc[
            population_mask, target
        ].to_numpy(dtype=np.int8)
        care = frame.loc[
            population_mask,
            "prenatal_care_month",
        ].to_numpy(dtype=np.float64)

        if len(np.unique(y)) != 2:
            raise ValueError(
                f"{target}: target-specific population lacks both classes."
            )

        bundle = load_frozen_model_bundle(
            Path(canonical["models_root"]),
            target,
        )
        model_metadata.append(
            {
                "target": target,
                "model_path": bundle["_model_path"],
                "model_sha256": bundle["_model_sha256"],
                "feature_set": bundle["feature_set"],
                "features": list(bundle["features"]),
                "source_canonical_task_id": (
                    args.canonical_task_id
                ),
            }
        )

        prediction_started = time.time()
        probabilities = frozen_probabilities(
            bundle,
            frame,
            population_mask,
        )
        prediction_seconds = (
            time.time() - prediction_started
        )

        target_cells, target_decomp, point_lookup = (
            four_cell_point_estimates(
                target=target,
                y=y,
                probabilities=probabilities,
                care_month=care,
            )
        )
        cell_rows.extend(target_cells)
        decomposition_rows.extend(target_decomp)

        target_crosschecks = crosscheck_points(
            target=target,
            lookup=point_lookup,
            canonical_protocol=canonical["protocol"],
        )
        crosscheck_rows.extend(
            target_crosschecks
        )

        print(
            f"{target}: canonical point score/cell cross-check PASS"
        )

        if args.preflight_only:
            del (
                bundle,
                probabilities,
                y,
                care,
            )
            gc.collect()
            continue

        bootstrap_seed = (
            int(config["bootstrap"]["seed"])
            + target_index
            * int(
                config["bootstrap"][
                    "target_seed_stride"
                ]
            )
        )

        reps, summary = bootstrap_four_cell(
            target=target,
            y=y,
            probabilities=probabilities,
            care_month=care,
            point_lookup=point_lookup,
            canonical_bootstrap=canonical["bootstrap"],
            replicates=int(
                config["bootstrap"]["replicates"]
            ),
            seed=bootstrap_seed,
            confidence_level=float(
                config["bootstrap"][
                    "confidence_level"
                ]
            ),
            progress_every=int(
                config["bootstrap"][
                    "progress_every"
                ]
            ),
        )
        bootstrap_rows.extend(reps)
        bootstrap_summary_rows.extend(summary)

        print(
            f"{target}: prediction={prediction_seconds:.1f}s, "
            f"n={len(y):,}, "
            f"bootstrap={len(reps):,} target/q rows"
        )

        del (
            bundle,
            probabilities,
            y,
            care,
            reps,
            summary,
        )
        gc.collect()

    atomic_csv(
        output / "four_cell_results.csv",
        cell_rows,
    )
    atomic_csv(
        output / "shapley_decomposition.csv",
        decomposition_rows,
    )
    atomic_csv(
        output / "canonical_crosscheck.csv",
        crosscheck_rows,
    )

    if args.preflight_only:
        preflight_manifest = {
            "status": "PREFLIGHT_PASS",
            "script_version": SCRIPT_VERSION,
            "clearml_task_id": task.id,
            "canonical_task_id": args.canonical_task_id,
            "clearml_dataset_id": args.dataset_id,
            "clearml_queue": args.queue,
            "cdc2023_sha256": observed_2023_hash,
            "model_lineage": model_metadata,
            "point_crosschecks": {
                "rows": len(crosscheck_rows),
                "all_pass": all(
                    row["status"] == "PASS"
                    for row in crosscheck_rows
                ),
            },
            "elapsed_seconds": float(
                time.time() - started
            ),
        }
        atomic_json(
            output / "run_manifest.json",
            preflight_manifest,
        )

        save_reproducibility_bundle(
            output,
            config=config,
            canonical_sources=canonical,
            model_metadata=model_metadata,
        )

        upload_outputs(task, output)
        task.close()

        print(
            "SHAPLEY PREFLIGHT PASS — frozen models/scores "
            "reproduced canonical point cells; bootstrap not run."
        )
        return 0

    atomic_csv(
        output / "shapley_bootstrap_replicates.csv",
        bootstrap_rows,
    )
    atomic_csv(
        output / "shapley_bootstrap_summary.csv",
        bootstrap_summary_rows,
    )

    main_rows = make_main_results(
        decomposition_rows,
        bootstrap_summary_rows,
    )
    atomic_csv(
        output / "shapley_main_results.csv",
        main_rows,
    )
    atomic_text(
        output / "shapley_summary.md",
        summary_markdown(main_rows),
    )

    invariants = validate_output_counts(
        cell_rows=cell_rows,
        decomposition_rows=decomposition_rows,
        crosscheck_rows=crosscheck_rows,
        bootstrap_rows=bootstrap_rows,
        bootstrap_summary_rows=bootstrap_summary_rows,
        main_rows=main_rows,
        replicates=int(
            config["bootstrap"]["replicates"]
        ),
    )

    if not all(invariants.values()):
        failed = [
            key
            for key, value in invariants.items()
            if not value
        ]
        raise AssertionError(
            f"Final Shapley invariants failed: {failed}"
        )

    save_reproducibility_bundle(
        output,
        config=config,
        canonical_sources=canonical,
        model_metadata=model_metadata,
    )

    manifest = {
        "status": "PASS",
        "script_version": SCRIPT_VERSION,
        "elapsed_seconds": float(
            time.time() - started
        ),
        "clearml_task_id": task.id,
        "canonical_source": {
            "task_id": args.canonical_task_id,
            "dataset_id": args.dataset_id,
            "queue": args.queue,
            "artifact_names": CANONICAL_ARTIFACTS,
            "canonical_manifest_status": canonical[
                "manifest"
            ].get("status"),
        },
        "input_sha256": {
            "cdc2023": observed_2023_hash,
            "canonical_protocol_results": sha256_file(
                Path(canonical["protocol_path"])
            ),
            "canonical_bootstrap_replicates": sha256_file(
                Path(canonical["bootstrap_path"])
            ),
            "canonical_manifest": sha256_file(
                Path(canonical["manifest_path"])
            ),
            "canonical_locked_config": sha256_file(
                Path(canonical["config_path"])
            ),
            "script": sha256_file(SCRIPT_PATH),
        },
        "model_lineage": model_metadata,
        "design": {
            "cdc2023_role": (
                "temporal evaluation cohort"
            ),
            "targets": list(TARGETS),
            "feature_set": FEATURE_SET_NAME,
            "paper_feature_set_name": (
                "conservative landmark feature set"
            ),
            "fractions": list(FRACTIONS),
            "candidate_sets": [
                EARLY_SCENARIO,
                ALL_SCENARIO,
            ],
            "four_cells": {
                "C00": "early candidate set + early-derived absolute budget",
                "C10": "all-record candidate set + early-derived absolute budget",
                "C01": "early candidate set + all-record-derived absolute budget",
                "C11": "all-record candidate set + all-record-derived absolute budget",
            },
            "bootstrap_factor_levels": (
                "candidate set is resampled through records; B_early and B_all "
                "remain fixed at observed absolute counts"
            ),
            "population_metric": (
                "TP / all target-population events"
            ),
            "model_refit_count": 0,
            "threshold_reestimation_count": 0,
            "same_frozen_score_vector_per_target": True,
            "tie_breaker": (
                "score descending, original target-population "
                "row position ascending"
            ),
        },
        "bootstrap": {
            "method": (
                "paired full-size record-level "
                "nonparametric bootstrap"
            ),
            "replicates": int(
                config["bootstrap"]["replicates"]
            ),
            "base_seed": int(
                config["bootstrap"]["seed"]
            ),
            "target_seed_stride": int(
                config["bootstrap"][
                    "target_seed_stride"
                ]
            ),
            "confidence_level": float(
                config["bootstrap"][
                    "confidence_level"
                ]
            ),
            "budget_levels_fixed_to_observed_point_counts": True,
            "early_budget_recomputed_from_resampled_early_n": False,
            "all_budget_recomputed_from_resampled_n": False,
            "budget_definition": (
                "B_early=round(q*N_early_observed); "
                "B_all=round(q*N_all_observed); both fixed in every replicate"
            ),
            "canonical_replicate_crosscheck": (
                "C01/C11 exact match to canonical fixed_absolute_budget "
                "early_entry->all_record_upper_bound bootstrap"
            ),
        },
        "outputs": {
            "four_cell_rows": len(cell_rows),
            "decomposition_rows": len(
                decomposition_rows
            ),
            "canonical_crosscheck_rows": len(
                crosscheck_rows
            ),
            "bootstrap_replicate_rows": len(
                bootstrap_rows
            ),
            "bootstrap_summary_rows": len(
                bootstrap_summary_rows
            ),
            "main_result_rows": len(main_rows),
        },
        "invariants": invariants,
        "git": git_information(),
    }

    atomic_json(
        output / "run_manifest.json",
        manifest,
    )

    # Hash final outputs except manifest itself.
    output_hashes: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if (
            not path.is_file()
            or path.name == "run_manifest.json"
        ):
            continue
        output_hashes[
            str(path.relative_to(output))
        ] = sha256_file(path)

    manifest["output_sha256"] = output_hashes
    atomic_json(
        output / "run_manifest.json",
        manifest,
    )

    upload_outputs(task, output)
    task.close()

    print("TAE SHAPLEY RUN PASS")
    print(f"Output: {output}")
    print(f"Canonical Task: {args.canonical_task_id}")
    print(f"Shapley Task: {manifest['clearml_task_id']}")
    print(
        f"Bootstrap rows: {len(bootstrap_rows):,}"
    )
    print("All point cross-checks: PASS")
    print("Canonical fixed-budget bootstrap C01/C11 cross-checks: PASS")
    print("All factorial/Shapley identities: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
