"""Load frozen research scores from native LightGBM text and summary-only JSON.

No pickle loading, record access, model fitting, or threshold recalibration.
Inputs must use the canonical numeric CDC representation, including mother_race.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGETS = ("target_preterm", "target_nicu", "target_lbw")
FEATURES = (
    "mother_race", "mother_height_inches", "mother_bmi", "mother_weight_pre",
    "prior_live_births", "prior_dead_births", "prior_terminations", "diab_pre",
    "hyper_pre",
)
DEFAULT_PRETRAINED_DIR = Path(__file__).resolve().parents[2] / "pretrained"


def _finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"Invalid {label}: expected {size} finite values")
    return vector


class NativePreprocessor:
    """The frozen strict9 transform, with float32 conversion after assembly."""

    def __init__(self, config: dict[str, Any]) -> None:
        numeric = config["numeric"]
        categorical = config["categorical"]
        self.numeric_features = list(FEATURES[1:])
        if numeric["features"] != self.numeric_features:
            raise ValueError("Unexpected numeric feature order")
        if categorical["feature"] != "mother_race":
            raise ValueError("Expected numeric mother_race categories")
        if config["matrix_dtype"] != "float32":
            raise ValueError("Expected float32 model matrix")
        medians = _finite_vector(numeric["medians"], 8, "medians")
        means = _finite_vector(numeric["means"], 8, "means")
        scales = _finite_vector(numeric["scales"], 8, "scales")
        if np.any(scales <= 0):
            raise ValueError("Numeric scales must be positive")
        categories = _finite_vector(categorical["categories"], 31, "categories")
        if not np.array_equal(categories, np.arange(1, 32)):
            raise ValueError("Expected the numeric MRACE31 codebook 1..31")
        mode = float(categorical["mode"])
        if mode not in categories:
            raise ValueError("Categorical mode must belong to the codebook")

        # Initialize sklearn validation state from summaries, never records.
        # A single median row and a single mode row reproduce the imputers.
        self._numeric_imputer = SimpleImputer(strategy="median").fit(
            pd.DataFrame([medians], columns=self.numeric_features)
        )
        self._categorical_imputer = SimpleImputer(strategy="most_frequent").fit(
            pd.DataFrame({"mother_race": [int(mode)]}, dtype=np.int64)
        )
        self._scaler = StandardScaler()
        self._scaler.n_features_in_ = len(self.numeric_features)
        self._scaler.mean_ = means
        self._scaler.scale_ = scales
        self._onehot = OneHotEncoder(
            categories=[categories.astype(np.int64)], handle_unknown="ignore",
            sparse_output=True, dtype=np.float32,
        ).fit(categories.astype(np.int64).reshape(-1, 1))

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        """Impute, standardize, one-hot encode, then cast the full matrix.

        NaN is missing; unseen numeric race codes yield an all-zero one-hot
        block. String race codes are rejected. Extra columns are
        ignored, as in the original ColumnTransformer(remainder='drop').
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame with canonical features")
        if not frame.columns.is_unique:
            raise ValueError("Duplicate input columns are not supported")
        missing = set(FEATURES) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")
        race_kind = pd.api.types.infer_dtype(frame["mother_race"].dropna())
        if race_kind not in {"integer", "floating", "mixed-integer-float", "empty"}:
            raise TypeError("mother_race must contain numeric CDC codes, not strings")
        numeric = self._numeric_imputer.transform(frame[self.numeric_features])
        numeric = self._scaler.transform(numeric)
        categorical = self._categorical_imputer.transform(frame[["mother_race"]])
        categorical = self._onehot.transform(categorical)
        return sparse.hstack([numeric, categorical], format="csr").astype(
            np.float32, copy=False
        )


class NativeBinaryModel:
    """Small predict_proba adapter around the official LightGBM reader."""

    def __init__(self, model_path: Path) -> None:
        # Reading text also works when the repository path contains Unicode.
        self.booster_ = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
        if self.booster_.num_feature() != 39:
            raise ValueError("Expected 39 transformed strict9 features")
        if self.booster_.params.get("objective") != "binary":
            raise ValueError("Expected a binary LightGBM model")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = self.booster_.num_feature()

    def predict_proba(
        self, matrix: Any, *, num_threads: int = 1,
        start_iteration: int = 0, num_iteration: int | None = None,
    ) -> np.ndarray:
        """Return columns [P(class=0), P(class=1)], without recalibration."""
        positive = np.asarray(self.booster_.predict(
            matrix, num_threads=num_threads, start_iteration=start_iteration,
            num_iteration=num_iteration,
        ), dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


def load_native_bundle(path: str | Path) -> dict[str, Any]:
    """Load a target directory containing model.txt and preprocessing.json.

    The JSON path itself is also accepted. Published checksums are in the
    pretrained root's SHA256SUMS; the runtime does not deserialize Python code.
    """
    path = Path(path)
    directory = path.parent if path.name == "preprocessing.json" else path
    metadata_path = directory / "preprocessing.json"
    model_path = directory / "model.txt"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["schema_version"] != 1:
        raise ValueError("Unsupported native bundle schema")
    if metadata["target"] not in TARGETS:
        raise ValueError("Unsupported native target")
    if metadata["features"] != list(FEATURES):
        raise ValueError("Expected canonical strict9 feature order")
    if metadata["feature_set"] != "landmark_strict":
        raise ValueError("Expected landmark_strict feature set")
    thresholds = {float(k): float(v) for k, v in metadata["thresholds"].items()}
    if set(thresholds) != {0.05, 0.1} or not all(
        np.isfinite(v) and 0 <= v <= 1 for v in thresholds.values()
    ):
        raise ValueError("Expected frozen probability thresholds for 0.05 and 0.1")
    return {
        "features": list(FEATURES),
        "target": metadata["target"],
        "feature_set": metadata["feature_set"],
        "training_population": metadata["training_population"],
        "thresholds": thresholds,
        "preprocessor": NativePreprocessor(metadata["preprocessing"]),
        "model": NativeBinaryModel(model_path),
        "_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "_preprocessing_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    }


def find_native_bundle(models_root: str | Path, target: str) -> Path:
    """Find exactly one target directory, accepting that directory as root."""
    if target not in TARGETS:
        raise ValueError(f"Unsupported native target: {target}")
    root = Path(models_root)
    candidates = {p.parent for p in root.rglob("preprocessing.json")
                  if p.parent.name == target and (p.parent / "model.txt").is_file()}
    if not candidates:
        raise FileNotFoundError(f"No native bundle for {target}")
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous native bundles for {target}")
    return candidates.pop()


def load_native_target(
    models_root: str | Path, target: str,
) -> dict[str, Any]:
    """Drop-in target lookup: load_native_target(pretrained_root, target)."""
    bundle = load_native_bundle(find_native_bundle(models_root, target))
    if bundle["target"] != target:
        raise ValueError("Native directory and metadata target disagree")
    return bundle
