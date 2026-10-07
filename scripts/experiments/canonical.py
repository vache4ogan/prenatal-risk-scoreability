"""Canonical training, cohort definitions, and three evaluation protocols."""
from __future__ import annotations
import copy
import hashlib
import math
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_CONFIG = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/canonical/experiment_config_tae_canonical.yaml").read_text(encoding="utf-8"))

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
    "prior_terminations",
    "diab_pre",
    "hyper_pre",
]

FEATURE_SETS = {
    "landmark_strict": LANDMARK_STRICT_FEATURES,
    "no_priorterm": [f for f in LANDMARK_STRICT_FEATURES if f != "prior_terminations"],
    "no_race": [f for f in LANDMARK_STRICT_FEATURES if f != "mother_race"],
}


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


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_local_config(path: Optional[Path]) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config
    candidate = path.expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("YAML config root must be a mapping.")
    return deep_update(config, payload)


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


def lightgbm_parameters(config: dict[str, Any]) -> dict[str, Any]:
    parameters = dict(config["model"]["parameters"])
    # Preserve the canonical CPU model specification.
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
    name = str(feature_cfg.get("name"))
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {name}")
    expected_features = FEATURE_SETS[name]
    features = [str(value) for value in feature_cfg.get("features", [])]
    if features != expected_features:
        raise ValueError(
            "feature_set.features must match the named frozen feature list exactly."
        )
    if int(feature_cfg.get("n_features", -1)) != len(expected_features):
        raise ValueError("feature_set.n_features does not match the frozen feature list.")
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
    if categorical != (["mother_race"] if "mother_race" in features else []):
        raise ValueError("Only mother_race, when retained, may be categorical.")

    if str(config["model"].get("name")) != "lightgbm":
        raise ValueError("Only LightGBM is allowed in the canonical main run.")
    parameters = config["model"]["parameters"]
    if parameters.get("class_weight") is not None:
        raise ValueError("class_weight must remain null.")
    if float(parameters.get("scale_pos_weight", 1.0)) != 1.0:
        raise ValueError("scale_pos_weight must remain 1.0.")
    if str(parameters.get("device_type", "cpu")) != "cpu":
        raise ValueError(
            "Canonical LightGBM is CPU-only."
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

    expected_hashes = config["data"].get("expected_sha256", {})
    for year in ("2022", "2023"):
        digest = str(expected_hashes.get(year, ""))
        if len(digest) != 64:
            raise ValueError(f"Missing/invalid expected SHA-256 for CDC {year}.")


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
