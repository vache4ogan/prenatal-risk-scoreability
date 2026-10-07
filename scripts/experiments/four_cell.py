"""Fixed-model four-cell audit with observed absolute budgets and symmetric Shapley effects."""
from __future__ import annotations
import gc
import hashlib
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from canonical import FEATURE_SETS
from native_model import load_native_target

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


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


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


def load_frozen_model_bundle(models_root: Path, target: str,
                             feature_set: str = FEATURE_SET_NAME) -> dict[str, Any]:
    basename = f"{target}__{feature_set}__lightgbm.joblib"
    try:
        model_path = find_exact_file(models_root, basename)
    except FileNotFoundError:
        if (feature_set == FEATURE_SET_NAME and models_root.is_dir()
                and not any(models_root.rglob(basename))):
            return load_native_target(models_root, target)
        raise

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

    if str(bundle["feature_set"]) != feature_set:
        raise AssertionError(
            f"{basename}: wrong feature set {bundle['feature_set']!r}"
        )

    features = [str(v) for v in bundle["features"]]
    if features != FEATURE_SETS[feature_set]:
        raise AssertionError(
            f"{basename}: frozen feature list does not match {feature_set}."
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
