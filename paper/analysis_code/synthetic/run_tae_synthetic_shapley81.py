#!/usr/bin/env python3
"""TAE 2026 synthetic evaluation experiment.

This simulation isolates how candidate-set availability and absolute selection
capacity interact under three evaluation protocols:

1. cohort_specific_top_fraction
2. fixed_absolute_budget
3. fixed_calibration_threshold

The data-generating process is intentionally minimal. A latent risk Z drives the
binary outcome Y; a noisy score R = Z + epsilon represents model quality; and
availability A depends on Z through a logistic model. The all-population and
available-subset scenarios share the same score vector within each replicate.

The primary synthetic quantities are the naive cohort-specific top-q
contrast, the fixed-budget availability effect, the symmetric Shapley
availability/capacity effects, and their interaction.

The full factorial grid is locked to 81 conditions:

    3 availability fractions x 3 availability-outcome relationships
    x 3 model qualities x 3 selection fractions = 81 conditions.

For each condition, the same latent population, outcome, availability mask,
and score vector are used across the three evaluation protocols. The script
reports all 81 conditions, 95% replicate-level Monte Carlo intervals, sign
reversals, intervals below zero, Shapley components, and the maximum interaction.
"""

from __future__ import annotations

SYNTHETIC_SCRIPT_VERSION = "2026-08-25-v1-shapley-81-grid"

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from numpy.polynomial.hermite import hermgauss


PROTOCOL_A = "cohort_specific_top_fraction"
PROTOCOL_B = "fixed_absolute_budget"
PROTOCOL_C = "fixed_calibration_threshold"
SCENARIO_AVAILABLE = "available_subset"
SCENARIO_ALL = "all_population"
EXPECTED_PROTOCOLS = [PROTOCOL_A, PROTOCOL_B, PROTOCOL_C]
EXPECTED_SCENARIOS = [SCENARIO_AVAILABLE, SCENARIO_ALL]


@dataclass(frozen=True)
class AvailabilityAssociation:
    name: str
    slope: float


@dataclass(frozen=True)
class ModelQuality:
    name: str
    noise_sd: float


@dataclass
class MetricResult:
    full_population_n: int
    full_population_event_n: int
    eligible_n: int
    eligible_event_n: int
    selected_n: int
    true_positive_n: int
    precision: float
    recall_among_eligible: float
    population_event_capture: float
    event_availability_ceiling: float


def stable_sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic function."""
    arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(arr)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    neg = ~pos
    exp_x = np.exp(arr[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    if np.isscalar(x):
        return float(out)
    return out


def standard_normal_quadrature(n_nodes: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    """Nodes and normalized weights for expectations under N(0, 1)."""
    nodes, weights = hermgauss(n_nodes)
    z = np.sqrt(2.0) * nodes
    w = weights / np.sqrt(np.pi)
    return z.astype(np.float64), w.astype(np.float64)


def solve_logistic_intercept(
    target_mean: float,
    slope: float,
    quadrature_nodes: np.ndarray,
    quadrature_weights: np.ndarray,
    tol: float = 1e-13,
    max_iter: int = 200,
) -> float:
    """Solve E[sigmoid(intercept + slope*Z)] = target_mean for Z~N(0,1)."""
    if not (0.0 < target_mean < 1.0):
        raise ValueError(f"target_mean must be in (0,1), got {target_mean}")

    lo, hi = -40.0, 40.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        mean = float(
            np.sum(
                quadrature_weights
                * stable_sigmoid(mid + slope * quadrature_nodes)
            )
        )
        if abs(mean - target_mean) <= tol:
            return mid
        if mean < target_mean:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def round_budget(q: float, n: int) -> int:
    """Match the canonical definition B(q)=round(q*N) using Python round()."""
    b = int(round(float(q) * int(n)))
    return max(0, min(b, int(n)))


def stable_descending_order(scores: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
    """Sort by score descending, breaking exact ties by original index ascending."""
    candidate_scores = scores[candidate_indices]
    local_order = np.lexsort((candidate_indices, -candidate_scores))
    return candidate_indices[local_order]


def top_b_metrics_from_order(
    y: np.ndarray,
    ordered_indices: np.ndarray,
    budget: int,
    full_event_n: int,
) -> MetricResult:
    """Compute top-B metrics from an already score-sorted candidate index array.

    This is algebraically identical to ``top_b_metrics`` but avoids repeatedly
    sorting the same score vector. The supplied order must be descending score
    with original-row-index ascending tie breaking.
    """
    eligible_n = int(ordered_indices.size)
    if budget < 0 or budget > eligible_n:
        raise ValueError(
            f"budget={budget} is outside [0, eligible_n={eligible_n}]"
        )
    if budget == 0:
        tp = 0
    else:
        tp = int(y[ordered_indices[:budget]].sum(dtype=np.int64))
    eligible_event_n = int(y[ordered_indices].sum(dtype=np.int64))
    return build_metric_result(
        full_population_n=int(y.size),
        full_event_n=full_event_n,
        eligible_n=eligible_n,
        eligible_event_n=eligible_event_n,
        selected_n=int(budget),
        tp=tp,
    )


def top_b_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    eligible_mask: np.ndarray,
    budget: int,
    full_event_n: int,
) -> MetricResult:
    eligible_idx = np.flatnonzero(eligible_mask)
    eligible_n = int(eligible_idx.size)
    eligible_event_n = int(y[eligible_idx].sum(dtype=np.int64))

    if budget < 0:
        raise ValueError(f"budget must be >=0, got {budget}")
    if budget > eligible_n:
        raise ValueError(
            f"Absolute budget {budget} exceeds eligible_n={eligible_n}. "
            "Choose q below the minimum availability fraction or adjust the design."
        )

    if budget == 0:
        selected_n = 0
        tp = 0
    else:
        ranked = stable_descending_order(scores, eligible_idx)
        selected = ranked[:budget]
        selected_n = int(selected.size)
        tp = int(y[selected].sum(dtype=np.int64))

    return build_metric_result(
        full_population_n=int(y.size),
        full_event_n=full_event_n,
        eligible_n=eligible_n,
        eligible_event_n=eligible_event_n,
        selected_n=selected_n,
        tp=tp,
    )


def threshold_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    eligible_mask: np.ndarray,
    threshold: float,
    full_event_n: int,
) -> MetricResult:
    eligible_n = int(eligible_mask.sum())
    eligible_event_n = int(y[eligible_mask].sum(dtype=np.int64))
    selected_mask = eligible_mask & (scores >= threshold)
    selected_n = int(selected_mask.sum())
    tp = int(y[selected_mask].sum(dtype=np.int64))

    return build_metric_result(
        full_population_n=int(y.size),
        full_event_n=full_event_n,
        eligible_n=eligible_n,
        eligible_event_n=eligible_event_n,
        selected_n=selected_n,
        tp=tp,
    )


def build_metric_result(
    full_population_n: int,
    full_event_n: int,
    eligible_n: int,
    eligible_event_n: int,
    selected_n: int,
    tp: int,
) -> MetricResult:
    precision = float(tp / selected_n) if selected_n > 0 else float("nan")
    recall_eligible = (
        float(tp / eligible_event_n) if eligible_event_n > 0 else float("nan")
    )
    population_capture = (
        float(tp / full_event_n) if full_event_n > 0 else float("nan")
    )
    event_ceiling = (
        float(eligible_event_n / full_event_n) if full_event_n > 0 else float("nan")
    )
    return MetricResult(
        full_population_n=full_population_n,
        full_population_event_n=full_event_n,
        eligible_n=eligible_n,
        eligible_event_n=eligible_event_n,
        selected_n=selected_n,
        true_positive_n=tp,
        precision=precision,
        recall_among_eligible=recall_eligible,
        population_event_capture=population_capture,
        event_availability_ceiling=event_ceiling,
    )


def calibration_threshold(scores: np.ndarray, q: float) -> Tuple[float, int]:
    """Return kth-highest score threshold from an independent all-population sample."""
    n = int(scores.size)
    k = round_budget(q, n)
    if k <= 0:
        return float("inf"), 0
    if k >= n:
        return float("-inf"), n
    ranked_idx = stable_descending_order(scores, np.arange(n, dtype=np.int64))
    threshold = float(scores[ranked_idx[k - 1]])
    return threshold, k


def metric_to_row(
    metric: MetricResult,
    *,
    replicate: int,
    availability_target_fraction: float,
    availability_relation: str,
    availability_slope: float,
    availability_intercept: float,
    model_quality: str,
    score_noise_sd: float,
    expected_score_latent_correlation: float,
    nominal_fraction: float,
    protocol: str,
    scenario: str,
    threshold: float | None,
    calibration_target_selected_n: int | None,
) -> Dict[str, Any]:
    return {
        "replicate": replicate,
        "availability_target_fraction": availability_target_fraction,
        "availability_relation": availability_relation,
        "availability_slope": availability_slope,
        "availability_intercept": availability_intercept,
        "model_quality": model_quality,
        "score_noise_sd": score_noise_sd,
        "expected_score_latent_correlation": expected_score_latent_correlation,
        "nominal_fraction": nominal_fraction,
        "protocol": protocol,
        "scenario": scenario,
        "threshold": threshold,
        "calibration_target_selected_n": calibration_target_selected_n,
        "full_population_n": metric.full_population_n,
        "full_population_event_n": metric.full_population_event_n,
        "eligible_n": metric.eligible_n,
        "eligible_event_n": metric.eligible_event_n,
        "eligible_fraction": metric.eligible_n / metric.full_population_n,
        "selected_n": metric.selected_n,
        "true_positive_n": metric.true_positive_n,
        "precision": metric.precision,
        "recall_among_eligible": metric.recall_among_eligible,
        "population_event_capture": metric.population_event_capture,
        "event_availability_ceiling": metric.event_availability_ceiling,
    }


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("YAML root must be a mapping")
    validate_config(config)
    return config


def parse_availability_associations(config: Mapping[str, Any]) -> List[AvailabilityAssociation]:
    items = config["simulation"]["availability_associations"]
    return [
        AvailabilityAssociation(name=str(x["name"]), slope=float(x["slope"]))
        for x in items
    ]


def parse_model_qualities(config: Mapping[str, Any]) -> List[ModelQuality]:
    items = config["simulation"]["model_qualities"]
    return [
        ModelQuality(name=str(x["name"]), noise_sd=float(x["noise_sd"]))
        for x in items
    ]


def validate_config(config: Mapping[str, Any]) -> None:
    for section in ["simulation", "evaluation", "output"]:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    sim = config["simulation"]
    ev = config["evaluation"]

    population_n = int(sim["population_n"])
    calibration_n = int(sim["calibration_n"])
    replicates = int(sim["replicates"])
    prevalence = float(sim["outcome_prevalence"])
    slope = float(sim["outcome_risk_slope"])
    fractions = [float(x) for x in sim["availability_fractions"]]
    q_values = [float(x) for x in ev["fractions"]]

    if population_n <= 0 or calibration_n <= 0 or replicates <= 0:
        raise ValueError("population_n, calibration_n, and replicates must be positive")
    if not (0.0 < prevalence < 1.0):
        raise ValueError("outcome_prevalence must be in (0,1)")
    if not math.isfinite(slope) or slope == 0.0:
        raise ValueError("outcome_risk_slope must be finite and non-zero")
    if not fractions or any(not (0.0 < x <= 1.0) for x in fractions):
        raise ValueError("availability_fractions must all be in (0,1]")
    if not q_values or any(not (0.0 < q < 1.0) for q in q_values):
        raise ValueError("evaluation fractions must all be in (0,1)")

    min_avail = min(fractions)
    max_q = max(q_values)
    if max_q >= min_avail:
        raise ValueError(
            "For fixed_absolute_budget, max evaluation fraction must be strictly "
            f"below minimum availability fraction. Got max_q={max_q}, "
            f"min_availability={min_avail}."
        )

    protocols = list(ev["protocols"])
    scenarios = list(ev["scenarios"])
    if protocols != EXPECTED_PROTOCOLS:
        raise ValueError(
            f"protocols must be exactly {EXPECTED_PROTOCOLS} in this canonical synthetic design"
        )
    if scenarios != EXPECTED_SCENARIOS:
        raise ValueError(
            f"scenarios must be exactly {EXPECTED_SCENARIOS} in this canonical synthetic design"
        )

    assoc = parse_availability_associations(config)
    qualities = parse_model_qualities(config)
    if len({x.name for x in assoc}) != len(assoc):
        raise ValueError("availability association names must be unique")
    if len({x.name for x in qualities}) != len(qualities):
        raise ValueError("model quality names must be unique")
    if any(not math.isfinite(x.slope) for x in assoc):
        raise ValueError("availability slopes must be finite")
    if any((not math.isfinite(x.noise_sd)) or x.noise_sd < 0 for x in qualities):
        raise ValueError("model noise_sd values must be finite and >=0")

    n_conditions = (
        len(fractions)
        * len(assoc)
        * len(qualities)
        * len(q_values)
    )
    if n_conditions != 81:
        raise ValueError(
            "The publication synthetic experiment must contain exactly 81 "
            f"conditions (got {n_conditions})."
        )

    expected_fractions = [0.05, 0.10, 0.20]
    if q_values != expected_fractions:
        raise ValueError(
            f"evaluation fractions must be exactly {expected_fractions}; got {q_values}"
        )
    expected_availability = [0.50, 0.70, 0.90]
    if fractions != expected_availability:
        raise ValueError(
            f"availability fractions must be exactly {expected_availability}; got {fractions}"
        )
    expected_assoc = ["higher_risk_less_available", "risk_independent", "higher_risk_more_available"]
    if [x.name for x in assoc] != expected_assoc:
        raise ValueError(
            f"availability associations must be exactly {expected_assoc}; got {[x.name for x in assoc]}"
        )
    expected_qualities = ["good", "medium", "weak"]
    if [x.name for x in qualities] != expected_qualities:
        raise ValueError(
            f"model qualities must be exactly {expected_qualities}; got {[x.name for x in qualities]}"
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_git_info(cwd: Path) -> Dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=cwd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    commit = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    status = git("status", "--porcelain")
    return {
        "commit_hash": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
    }


def package_versions() -> Dict[str, str]:
    names = ["numpy", "pandas", "PyYAML"]
    out: Dict[str, str] = {"python": platform.python_version()}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def dataframe_quantile(series: pd.Series, q: float) -> float:
    return float(series.quantile(q))


def summarize_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "availability_target_fraction",
        "availability_relation",
        "availability_slope",
        "model_quality",
        "score_noise_sd",
        "expected_score_latent_correlation",
        "nominal_fraction",
        "protocol",
    ]
    rows: List[Dict[str, Any]] = []
    for keys, g in df.groupby(group_cols, sort=True, dropna=False):
        row = dict(zip(group_cols, keys))
        for col in [
            "delta_population_event_capture",
            "delta_selected_n",
            "delta_true_positive_n",
        ]:
            s = g[col].astype(float)
            row[f"mean_{col}"] = float(s.mean())
            row[f"sd_{col}"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
            row[f"q025_{col}"] = dataframe_quantile(s, 0.025)
            row[f"q975_{col}"] = dataframe_quantile(s, 0.975)
        row["replicates"] = int(len(g))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize all 81 synthetic conditions and expose Shapley diagnostics."""
    group_cols = [
        "availability_target_fraction",
        "availability_relation",
        "availability_slope",
        "model_quality",
        "score_noise_sd",
        "expected_score_latent_correlation",
        "nominal_fraction",
    ]
    metrics = [
        "realized_available_fraction",
        "event_availability_ceiling",
        "naive_top_fraction_effect",
        "availability_effect_at_all_budget",
        "capacity_effect_within_available_subset",
        "availability_main_effect",
        "capacity_main_effect",
        "distortion_top_fraction_minus_fixed_budget",
        "shapley_availability_effect",
        "shapley_capacity_effect",
        "interaction_effect",
        "total_four_cell_effect",
        "naive_delta_true_positive_n",
        "availability_delta_true_positive_n",
        "capacity_delta_true_positive_n",
    ]
    rows: List[Dict[str, Any]] = []
    for keys, g in df.groupby(group_cols, sort=True, dropna=False):
        row = dict(zip(group_cols, keys))
        for col in metrics:
            s = g[col].astype(float)
            row[f"mean_{col}"] = float(s.mean())
            row[f"sd_{col}"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
            row[f"q025_{col}"] = dataframe_quantile(s, 0.025)
            row[f"q975_{col}"] = dataframe_quantile(s, 0.975)

        naive = g["naive_top_fraction_effect"].to_numpy(dtype=float)
        fixed = g["availability_effect_at_all_budget"].to_numpy(dtype=float)
        shapley_a = g["shapley_availability_effect"].to_numpy(dtype=float)
        shapley_c = g["shapley_capacity_effect"].to_numpy(dtype=float)
        interaction = g["interaction_effect"].to_numpy(dtype=float)

        row["replicates"] = int(len(g))
        row["naive_fixed_budget_sign_reversal"] = bool(
            np.sign(row["mean_naive_top_fraction_effect"])
            != np.sign(row["mean_availability_effect_at_all_budget"])
            and row["mean_naive_top_fraction_effect"] != 0.0
            and row["mean_availability_effect_at_all_budget"] != 0.0
        )
        row["sign_reversal_replicate_fraction"] = float(
            np.mean((np.sign(naive) != np.sign(fixed)) & (naive != 0.0) & (fixed != 0.0))
        )
        row["fixed_budget_interval_below_zero"] = bool(
            row["q975_availability_effect_at_all_budget"] < 0.0
        )
        row["fixed_budget_interval_above_zero"] = bool(
            row["q025_availability_effect_at_all_budget"] > 0.0
        )
        row["shapley_availability_interval_below_zero"] = bool(
            row["q975_shapley_availability_effect"] < 0.0
        )
        row["shapley_capacity_interval_below_zero"] = bool(
            row["q975_shapley_capacity_effect"] < 0.0
        )
        row["interaction_interval_below_zero"] = bool(
            row["q975_interaction_effect"] < 0.0
        )
        row["interaction_interval_above_zero"] = bool(
            row["q025_interaction_effect"] > 0.0
        )
        row["max_abs_interaction_replicate"] = float(np.max(np.abs(interaction)))
        row["mean_abs_interaction"] = float(np.mean(np.abs(interaction)))
        row["max_abs_identity_error"] = float(np.max(np.abs(g["identity_error"])))
        row["max_abs_tp_identity_error"] = int(np.max(np.abs(g["tp_identity_error"])))
        row["max_abs_factorial_identity_error"] = float(
            np.max(np.abs(g["factorial_identity_error"]))
        )
        row["max_abs_shapley_identity_error"] = float(
            np.max(np.abs(g["shapley_identity_error"]))
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) != 81:
        raise AssertionError(f"Expected exactly 81 condition summaries, got {len(out)}")
    return out


def build_figure3_data(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical 81-row data table consumed by the Figure 3 plot."""
    columns = [
        "availability_target_fraction",
        "availability_relation",
        "model_quality",
        "score_noise_sd",
        "expected_score_latent_correlation",
        "nominal_fraction",
        "mean_naive_top_fraction_effect",
        "q025_naive_top_fraction_effect",
        "q975_naive_top_fraction_effect",
        "mean_availability_effect_at_all_budget",
        "q025_availability_effect_at_all_budget",
        "q975_availability_effect_at_all_budget",
        "mean_availability_main_effect",
        "q025_availability_main_effect",
        "q975_availability_main_effect",
        "mean_capacity_main_effect",
        "q025_capacity_main_effect",
        "q975_capacity_main_effect",
        "mean_shapley_availability_effect",
        "q025_shapley_availability_effect",
        "q975_shapley_availability_effect",
        "mean_shapley_capacity_effect",
        "q025_shapley_capacity_effect",
        "q975_shapley_capacity_effect",
        "mean_interaction_effect",
        "q025_interaction_effect",
        "q975_interaction_effect",
        "mean_distortion_top_fraction_minus_fixed_budget",
        "q025_distortion_top_fraction_minus_fixed_budget",
        "q975_distortion_top_fraction_minus_fixed_budget",
        "naive_fixed_budget_sign_reversal",
        "sign_reversal_replicate_fraction",
        "fixed_budget_interval_below_zero",
        "fixed_budget_interval_above_zero",
        "shapley_availability_interval_below_zero",
        "shapley_capacity_interval_below_zero",
        "interaction_interval_below_zero",
        "interaction_interval_above_zero",
        "max_abs_interaction_replicate",
    ]
    out = summary[columns].copy().sort_values(
        ["availability_target_fraction", "availability_relation", "model_quality", "nominal_fraction"]
    ).reset_index(drop=True)
    out.insert(0, "condition_id", np.arange(1, len(out) + 1, dtype=int))
    return out


def dataframe_to_markdown_no_deps(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table without optional packages.

    Pandas markdown rendering normally depends on the optional ``tabulate`` package.
    The synthetic experiment should not require that package merely to write its
    human-readable summary, so we render the table directly.
    """

    def format_cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value)
        # Keep generated Markdown structurally valid if a value ever contains
        # a pipe or a newline.
        return text.replace("|", "\\|").replace("\n", "<br>")

    headers = [format_cell(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_cell(v) for v in row) + " |")
    return "\n".join(lines)


def write_markdown_summary(
    path: Path,
    config: Mapping[str, Any],
    decomp_summary: pd.DataFrame,
    comparison_summary: pd.DataFrame,
) -> None:
    """Write a dependency-free report emphasizing the full 81-condition grid."""
    sim = config["simulation"]
    ev = config["evaluation"]
    sign_reversals = decomp_summary[decomp_summary["naive_fixed_budget_sign_reversal"]]
    ci_below_zero = decomp_summary[
        decomp_summary["fixed_budget_interval_below_zero"]
        | decomp_summary["shapley_availability_interval_below_zero"]
        | decomp_summary["shapley_capacity_interval_below_zero"]
        | decomp_summary["interaction_interval_below_zero"]
    ]
    max_row = decomp_summary.iloc[
        int(decomp_summary["max_abs_interaction_replicate"].abs().idxmax())
    ]

    lines: List[str] = [
        "# TAE synthetic 81-condition Shapley experiment",
        "",
        "This file is generated by `run_tae_synthetic_shapley81.py`.",
        "",
        "## Locked design",
        "",
        f"- Replicates per condition: {int(sim['replicates'])}",
        f"- Evaluation population per replicate: {int(sim['population_n']):,}",
        f"- Independent calibration population: {int(sim['calibration_n']):,}",
        f"- Availability fractions: {sim['availability_fractions']}",
        f"- Availability relationships: {[x['name'] for x in sim['availability_associations']]}",
        f"- Model qualities: {[x['name'] for x in sim['model_qualities']]}",
        f"- Selection fractions: {ev['fractions']}",
        "- Total factorial conditions: 81",
        "",
        "## Estimands",
        "",
        "- Naive contrast: C11 − C00 under cohort-specific top-q.",
        "- Fixed-budget availability effect (ordered reporting contrast): C11 − C01.",
        "- Extra-capacity effect in the available subset: C01 − C00.",
        "- Factorial availability main effect: C10 − C00.",
        "- Factorial capacity main effect: C01 − C00.",
        "- Shapley availability: 0.5[(C10−C00)+(C11−C01)].",
        "- Shapley capacity: 0.5[(C01−C00)+(C11−C10)].",
        "- Interaction: C11 − C10 − C01 + C00.",
        "- Primary outcome scale: population event capture; all events remain in the denominator.",
        "",
        "## Full-grid diagnostics",
        "",
        f"- Conditions with mean sign reversal (naive vs fixed-budget): {len(sign_reversals)}",
        f"- Conditions with at least one requested interval entirely below zero: {len(ci_below_zero)}",
        f"- Maximum absolute replicate-level interaction: {100.0 * float(decomp_summary['max_abs_interaction_replicate'].max()):.3f} pp",
        "",
        "### Conditions with sign reversal",
        "",
    ]

    if sign_reversals.empty:
        lines.append("None.")
    else:
        shown = sign_reversals[[
            "availability_target_fraction", "availability_relation", "model_quality",
            "nominal_fraction", "mean_naive_top_fraction_effect",
            "mean_availability_effect_at_all_budget", "mean_shapley_availability_effect",
            "mean_shapley_capacity_effect", "mean_interaction_effect",
            "sign_reversal_replicate_fraction",
        ]].copy()
        for col in shown.columns[4:9]:
            shown[col] = shown[col].map(lambda x: f"{100.0 * x:+.3f} pp")
        shown["sign_reversal_replicate_fraction"] = shown["sign_reversal_replicate_fraction"].map(
            lambda x: f"{100.0 * x:.1f}%"
        )
        lines.append(dataframe_to_markdown_no_deps(shown))

    lines.extend([
        "",
        "### Condition with maximum absolute interaction",
        "",
        f"- Availability fraction: {float(max_row['availability_target_fraction']):.2f}",
        f"- Availability relationship: {max_row['availability_relation']}",
        f"- Model quality: {max_row['model_quality']}",
        f"- q: {100.0 * float(max_row['nominal_fraction']):.0f}%",
        f"- Maximum absolute interaction across replicates: {100.0 * float(max_row['max_abs_interaction_replicate']):.3f} pp",
        "",
        "The machine-readable full 81-condition table is `synthetic_figure3_data.csv`.",
        "It is the canonical input for the Figure 3 plotting script.",
        "",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        existing = list(output_dir.iterdir())
        if existing and not overwrite:
            raise FileExistsError(
                f"Output directory {output_dir} is not empty and overwrite=false"
            )
        if overwrite:
            for p in existing:
                if p.is_dir():
                    import shutil
                    shutil.rmtree(p)
                else:
                    p.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def run_experiment(config: Mapping[str, Any], output_dir: Path) -> Dict[str, Any]:
    validate_config(config)
    start = time.time()

    sim = config["simulation"]
    ev = config["evaluation"]
    output_cfg = config["output"]

    base_seed = int(sim["seed"])
    replicates = int(sim["replicates"])
    n_eval = int(sim["population_n"])
    n_cal = int(sim["calibration_n"])
    target_prevalence = float(sim["outcome_prevalence"])
    outcome_slope = float(sim["outcome_risk_slope"])
    availability_fractions = [float(x) for x in sim["availability_fractions"]]
    availability_associations = parse_availability_associations(config)
    model_qualities = parse_model_qualities(config)
    q_values = [float(x) for x in ev["fractions"]]

    quadrature_nodes, quadrature_weights = standard_normal_quadrature(
        int(sim.get("quadrature_nodes", 80))
    )
    outcome_intercept = solve_logistic_intercept(
        target_prevalence,
        outcome_slope,
        quadrature_nodes,
        quadrature_weights,
    )

    availability_intercepts: Dict[Tuple[float, float], float] = {}
    for target_fraction in availability_fractions:
        for assoc in availability_associations:
            availability_intercepts[(target_fraction, assoc.slope)] = solve_logistic_intercept(
                target_fraction,
                assoc.slope,
                quadrature_nodes,
                quadrature_weights,
            )

    protocol_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    decomposition_rows: List[Dict[str, Any]] = []
    condition_rows: List[Dict[str, Any]] = []

    seed_sequence = np.random.SeedSequence(base_seed)
    child_sequences = seed_sequence.spawn(replicates)

    progress_every = int(sim.get("progress_every", 10))

    for replicate_idx, child_seed in enumerate(child_sequences, start=1):
        rng = np.random.default_rng(child_seed)

        # Common random numbers across all conditions in a replicate.
        z_eval = rng.standard_normal(n_eval)
        u_outcome = rng.random(n_eval)
        u_availability = rng.random(n_eval)
        epsilon_eval_base = rng.standard_normal(n_eval)

        z_cal = rng.standard_normal(n_cal)
        epsilon_cal_base = rng.standard_normal(n_cal)

        p_y = stable_sigmoid(outcome_intercept + outcome_slope * z_eval)
        y = (u_outcome < p_y).astype(np.uint8)
        full_event_n = int(y.sum(dtype=np.int64))
        if full_event_n <= 0:
            raise RuntimeError(
                f"Replicate {replicate_idx} generated zero events; increase population_n "
                "or outcome_prevalence."
            )

        all_mask = np.ones(n_eval, dtype=bool)

        # Availability masks do not depend on model-quality noise.
        availability_masks: Dict[Tuple[float, str], np.ndarray] = {}
        for target_fraction in availability_fractions:
            for assoc in availability_associations:
                intercept = availability_intercepts[(target_fraction, assoc.slope)]
                p_a = stable_sigmoid(intercept + assoc.slope * z_eval)
                mask = u_availability < p_a
                available_n = int(mask.sum())
                if available_n <= 0:
                    raise RuntimeError("Generated an empty available subset")
                available_event_n = int(y[mask].sum(dtype=np.int64))
                availability_masks[(target_fraction, assoc.name)] = mask
                condition_rows.append(
                    {
                        "replicate": replicate_idx,
                        "availability_target_fraction": target_fraction,
                        "availability_relation": assoc.name,
                        "availability_slope": assoc.slope,
                        "availability_intercept": intercept,
                        "realized_available_n": available_n,
                        "realized_available_fraction": available_n / n_eval,
                        "full_population_event_n": full_event_n,
                        "available_event_n": available_event_n,
                        "event_availability_ceiling": available_event_n / full_event_n,
                    }
                )

        for quality in model_qualities:
            score = z_eval + quality.noise_sd * epsilon_eval_base
            score_cal = z_cal + quality.noise_sd * epsilon_cal_base
            expected_corr = 1.0 / math.sqrt(1.0 + quality.noise_sd ** 2)

            # Sort each score vector once. Filtering the full order by an
            # availability mask preserves the exact canonical tie-breaking rule.
            full_order = stable_descending_order(
                score, np.arange(n_eval, dtype=np.int64)
            )
            cal_order = stable_descending_order(
                score_cal, np.arange(n_cal, dtype=np.int64)
            )

            thresholds: Dict[float, Tuple[float, int]] = {}
            for q in q_values:
                k_cal = round_budget(q, n_cal)
                if k_cal <= 0:
                    threshold = float("inf")
                elif k_cal >= n_cal:
                    threshold = float("-inf")
                else:
                    threshold = float(score_cal[cal_order[k_cal - 1]])
                thresholds[q] = (threshold, k_cal)

            full_event_cumsum = np.cumsum(y[full_order], dtype=np.int64)

            for target_fraction in availability_fractions:
                for assoc in availability_associations:
                    available_mask = availability_masks[(target_fraction, assoc.name)]
                    available_n = int(available_mask.sum())
                    available_event_n = int(y[available_mask].sum(dtype=np.int64))
                    intercept = availability_intercepts[(target_fraction, assoc.slope)]

                    # Filtering a globally sorted order is equivalent to sorting
                    # only the eligible records, but much faster for 100k x 100 runs.
                    available_order = full_order[available_mask[full_order]]
                    available_event_cumsum = np.cumsum(
                        y[available_order], dtype=np.int64
                    )

                    for q in q_values:
                        b_available = round_budget(q, available_n)
                        b_all = round_budget(q, n_eval)
                        if b_all > available_n:
                            raise RuntimeError(
                                f"Replicate {replicate_idx}: fixed budget B_all={b_all} "
                                f"exceeds available_n={available_n}."
                            )

                        threshold, k_cal = thresholds[q]

                        # C00 / C01 use the filtered available order.
                        available_a = top_b_metrics_from_order(
                            y, available_order, b_available, full_event_n
                        )
                        available_b = top_b_metrics_from_order(
                            y, available_order, b_all, full_event_n
                        )

                        # Protocol C: the common calibration threshold is applied
                        # to each scenario without changing the threshold.
                        available_c = threshold_metrics(
                            y, score, available_mask, threshold, full_event_n
                        )

                        # Full-population top-B metrics from the same global order.
                        all_a = top_b_metrics_from_order(
                            y, full_order, b_all, full_event_n
                        )
                        all_b = all_a
                        all_c = threshold_metrics(
                            y, score, all_mask, threshold, full_event_n
                        )

                        # C10: all-population candidate set at the early-derived budget.
                        c10 = top_b_metrics_from_order(
                            y, full_order, b_available, full_event_n
                        )

                        available_by_protocol = {
                            PROTOCOL_A: available_a,
                            PROTOCOL_B: available_b,
                            PROTOCOL_C: available_c,
                        }
                        full_by_protocol = {
                            PROTOCOL_A: all_a,
                            PROTOCOL_B: all_b,
                            PROTOCOL_C: all_c,
                        }

                        condition_key = {
                            "replicate": replicate_idx,
                            "availability_target_fraction": target_fraction,
                            "availability_relation": assoc.name,
                            "availability_slope": assoc.slope,
                            "availability_intercept": intercept,
                            "model_quality": quality.name,
                            "score_noise_sd": quality.noise_sd,
                            "expected_score_latent_correlation": expected_corr,
                            "nominal_fraction": q,
                        }

                        per_protocol_metrics: Dict[str, Tuple[MetricResult, MetricResult]] = {}

                        for protocol in EXPECTED_PROTOCOLS:
                            m_available = available_by_protocol[protocol]
                            m_all = full_by_protocol[protocol]
                            per_protocol_metrics[protocol] = (m_available, m_all)

                            protocol_rows.append(
                                metric_to_row(
                                    m_available,
                                    **condition_key,
                                    protocol=protocol,
                                    scenario=SCENARIO_AVAILABLE,
                                    threshold=threshold if protocol == PROTOCOL_C else None,
                                    calibration_target_selected_n=(
                                        k_cal if protocol == PROTOCOL_C else None
                                    ),
                                )
                            )
                            protocol_rows.append(
                                metric_to_row(
                                    m_all,
                                    **condition_key,
                                    protocol=protocol,
                                    scenario=SCENARIO_ALL,
                                    threshold=threshold if protocol == PROTOCOL_C else None,
                                    calibration_target_selected_n=(
                                        k_cal if protocol == PROTOCOL_C else None
                                    ),
                                )
                            )

                            comparison_rows.append(
                                {
                                    **condition_key,
                                    "protocol": protocol,
                                    "from_scenario": SCENARIO_AVAILABLE,
                                    "to_scenario": SCENARIO_ALL,
                                    "from_eligible_n": m_available.eligible_n,
                                    "to_eligible_n": m_all.eligible_n,
                                    "from_selected_n": m_available.selected_n,
                                    "to_selected_n": m_all.selected_n,
                                    "delta_selected_n": m_all.selected_n - m_available.selected_n,
                                    "from_true_positive_n": m_available.true_positive_n,
                                    "to_true_positive_n": m_all.true_positive_n,
                                    "delta_true_positive_n": (
                                        m_all.true_positive_n - m_available.true_positive_n
                                    ),
                                    "from_population_event_capture": m_available.population_event_capture,
                                    "to_population_event_capture": m_all.population_event_capture,
                                    "delta_population_event_capture": (
                                        m_all.population_event_capture
                                        - m_available.population_event_capture
                                    ),
                                    "threshold": threshold if protocol == PROTOCOL_C else None,
                                }
                            )

                        a_available, a_all = per_protocol_metrics[PROTOCOL_A]
                        b_available_metric, b_all_metric = per_protocol_metrics[PROTOCOL_B]

                        # Four-cell design for symmetric Shapley attribution.
                        c00 = a_available
                        c01 = b_available_metric
                        c11 = b_all_metric

                        if a_all.selected_n != b_all_metric.selected_n:
                            raise AssertionError("Protocol A/B all-population budgets differ")
                        if a_all.true_positive_n != b_all_metric.true_positive_n:
                            raise AssertionError("Protocol A/B all-population TP differ")
                        if c10.selected_n != b_available:
                            raise AssertionError("C10 did not use the early-derived absolute budget")

                        naive = c11.population_event_capture - c00.population_event_capture
                        fixed_budget_availability_effect = c11.population_event_capture - c01.population_event_capture
                        ordered_capacity_effect = c01.population_event_capture - c00.population_event_capture
                        distortion = naive - fixed_budget_availability_effect

                        availability_main = c10.population_event_capture - c00.population_event_capture
                        capacity_main = c01.population_event_capture - c00.population_event_capture

                        shapley_availability = 0.5 * (
                            (c10.population_event_capture - c00.population_event_capture)
                            + (c11.population_event_capture - c01.population_event_capture)
                        )
                        shapley_capacity = 0.5 * (
                            (c01.population_event_capture - c00.population_event_capture)
                            + (c11.population_event_capture - c10.population_event_capture)
                        )
                        interaction = (
                            c11.population_event_capture
                            - c10.population_event_capture
                            - c01.population_event_capture
                            + c00.population_event_capture
                        )
                        total_four_cell_effect = c11.population_event_capture - c00.population_event_capture
                        factorial_identity_error = (
                            total_four_cell_effect
                            - availability_main
                            - capacity_main
                            - interaction
                        )
                        shapley_identity_error = total_four_cell_effect - shapley_availability - shapley_capacity
                        ordered_identity_error = naive - fixed_budget_availability_effect - ordered_capacity_effect

                        naive_tp = c11.true_positive_n - c00.true_positive_n
                        fixed_budget_availability_tp = c11.true_positive_n - c01.true_positive_n
                        ordered_capacity_tp = c01.true_positive_n - c00.true_positive_n
                        availability_main_tp = c10.true_positive_n - c00.true_positive_n
                        capacity_main_tp = c01.true_positive_n - c00.true_positive_n
                        tp_interaction = (
                            c11.true_positive_n
                            - c10.true_positive_n
                            - c01.true_positive_n
                            + c00.true_positive_n
                        )
                        tp_factorial_identity_error = (
                            naive_tp - availability_main_tp - capacity_main_tp - tp_interaction
                        )
                        tp_ordered_identity_error = naive_tp - fixed_budget_availability_tp - ordered_capacity_tp

                        if tp_ordered_identity_error != 0:
                            raise AssertionError(
                                f"TP ordered decomposition failed: {tp_ordered_identity_error}"
                            )
                        if tp_factorial_identity_error != 0:
                            raise AssertionError(
                                f"TP factorial decomposition failed: {tp_factorial_identity_error}"
                            )
                        if abs(ordered_identity_error) > 1e-12:
                            raise AssertionError(
                                f"Ordered capture decomposition failed: identity_error={ordered_identity_error}"
                            )
                        if abs(distortion - ordered_capacity_effect) > 1e-12:
                            raise AssertionError("Distortion != ordered extra-capacity component")
                        if abs(factorial_identity_error) > 1e-12:
                            raise AssertionError("Factorial capture identity failed")
                        if abs(shapley_identity_error) > 1e-12:
                            raise AssertionError("Shapley identity failed")

                        decomposition_rows.append(
                            {
                                **condition_key,
                                "realized_available_n": available_n,
                                "realized_available_fraction": available_n / n_eval,
                                "available_event_n": available_event_n,
                                "full_population_event_n": full_event_n,
                                "event_availability_ceiling": available_event_n / full_event_n,
                                "budget_available_top_fraction": b_available,
                                "budget_all_top_fraction": b_all,
                                "c00_capture": c00.population_event_capture,
                                "c10_capture": c10.population_event_capture,
                                "c01_capture": c01.population_event_capture,
                                "c11_capture": c11.population_event_capture,
                                "naive_top_fraction_effect": naive,
                                "availability_effect_at_all_budget": fixed_budget_availability_effect,
                                "capacity_effect_within_available_subset": ordered_capacity_effect,
                                "availability_main_effect": availability_main,
                                "capacity_main_effect": capacity_main,
                                "distortion_top_fraction_minus_fixed_budget": distortion,
                                "shapley_availability_effect": shapley_availability,
                                "shapley_capacity_effect": shapley_capacity,
                                "interaction_effect": interaction,
                                "total_four_cell_effect": total_four_cell_effect,
                                "identity_error": ordered_identity_error,
                                "factorial_identity_error": factorial_identity_error,
                                "shapley_identity_error": shapley_identity_error,
                                "naive_delta_true_positive_n": naive_tp,
                                "availability_delta_true_positive_n": fixed_budget_availability_tp,
                                "capacity_delta_true_positive_n": ordered_capacity_tp,
                                "availability_main_delta_true_positive_n": availability_main_tp,
                                "capacity_main_delta_true_positive_n": capacity_main_tp,
                                "tp_interaction_n": tp_interaction,
                                "tp_factorial_identity_error": tp_factorial_identity_error,
                                "tp_identity_error": tp_ordered_identity_error,
                            }
                        )

        if progress_every > 0 and (
            replicate_idx % progress_every == 0 or replicate_idx == replicates
        ):
            print(f"Completed replicate {replicate_idx}/{replicates}", flush=True)

    protocol_df = pd.DataFrame(protocol_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    decomposition_df = pd.DataFrame(decomposition_rows)
    condition_df = pd.DataFrame(condition_rows)

    # Global invariants.
    assert_protocol_invariants(protocol_df, comparison_df, decomposition_df, config)

    comparison_summary_df = summarize_comparisons(comparison_df)
    decomposition_summary_df = summarize_decomposition(decomposition_df)
    figure3_df = build_figure3_data(decomposition_summary_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, pd.DataFrame] = {
        "synthetic_protocol_results.csv": protocol_df,
        "synthetic_protocol_comparisons.csv": comparison_df,
        "synthetic_decomposition.csv": decomposition_df,
        "synthetic_condition_replicates.csv": condition_df,
        "synthetic_comparison_summary.csv": comparison_summary_df,
        "synthetic_decomposition_summary.csv": decomposition_summary_df,
        "synthetic_shapley_replicates.csv": decomposition_df.copy(),
        "synthetic_shapley_summary.csv": decomposition_summary_df,
        "synthetic_figure3_data.csv": figure3_df,
        "synthetic_sign_reversals.csv": figure3_df[figure3_df["naive_fixed_budget_sign_reversal"]].copy(),
        "synthetic_ci_below_zero.csv": figure3_df[
            figure3_df["fixed_budget_interval_below_zero"]
            | figure3_df["shapley_availability_interval_below_zero"]
            | figure3_df["shapley_capacity_interval_below_zero"]
            | figure3_df["interaction_interval_below_zero"]
        ].copy(),
    }
    for filename, df in outputs.items():
        df.to_csv(output_dir / filename, index=False)

    summary_path = output_dir / "synthetic_summary.md"
    write_markdown_summary(
        summary_path, config, decomposition_summary_df, comparison_summary_df
    )

    elapsed = time.time() - start
    return {
        "elapsed_seconds": elapsed,
        "outcome_intercept": outcome_intercept,
        "availability_intercepts": {
            f"fraction={fraction}|slope={slope}": intercept
            for (fraction, slope), intercept in availability_intercepts.items()
        },
        "protocol_rows": len(protocol_df),
        "comparison_rows": len(comparison_df),
        "decomposition_rows": len(decomposition_df),
        "condition_rows": len(condition_df),
        "comparison_summary_rows": len(comparison_summary_df),
        "decomposition_summary_rows": len(decomposition_summary_df),
        "shapley_replicate_rows": len(decomposition_df),
        "figure3_rows": len(figure3_df),
        "sign_reversal_rows": int(len(figure3_df[figure3_df["naive_fixed_budget_sign_reversal"]])),
        "ci_below_zero_rows": int(len(figure3_df[
            figure3_df["fixed_budget_interval_below_zero"]
            | figure3_df["shapley_availability_interval_below_zero"]
            | figure3_df["shapley_capacity_interval_below_zero"]
            | figure3_df["interaction_interval_below_zero"]
        ])),
        "max_abs_interaction": float(decomposition_summary_df["max_abs_interaction_replicate"].max()),
    }


def assert_protocol_invariants(
    protocol_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    decomposition_df: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    sim = config["simulation"]
    ev = config["evaluation"]
    n_eval = int(sim["population_n"])

    key_cols = [
        "replicate",
        "availability_target_fraction",
        "availability_relation",
        "model_quality",
        "nominal_fraction",
    ]

    # Exactly 3 protocols x 2 scenarios per condition/q.
    counts = protocol_df.groupby(key_cols).size()
    if not (counts == 6).all():
        raise AssertionError("Expected exactly 6 protocol rows per condition/q")

    # Same population denominator across all protocols/scenarios for each condition/q.
    denom_nunique = protocol_df.groupby(key_cols)["full_population_event_n"].nunique()
    if not (denom_nunique == 1).all():
        raise AssertionError("Population event denominator is not invariant")

    # Available subset eligible_n is scenario-specific but must not depend on protocol.
    available = protocol_df[protocol_df["scenario"] == SCENARIO_AVAILABLE]
    if not (
        available.groupby(key_cols)["eligible_n"].nunique() == 1
    ).all():
        raise AssertionError("available_subset eligible_n varies across protocols")

    # All-population eligible_n is always N.
    all_rows = protocol_df[protocol_df["scenario"] == SCENARIO_ALL]
    if not (all_rows["eligible_n"] == n_eval).all():
        raise AssertionError("all_population eligible_n is not population_n")

    # Protocol A exact cohort-specific q budget.
    a = protocol_df[protocol_df["protocol"] == PROTOCOL_A]
    expected_a = [
        round_budget(q, n)
        for q, n in zip(a["nominal_fraction"], a["eligible_n"])
    ]
    if not np.array_equal(a["selected_n"].to_numpy(), np.asarray(expected_a)):
        raise AssertionError("Protocol A does not use exact cohort-specific top-q budget")

    # Protocol B same B_all in both scenarios and equals round(q*N_all).
    b = protocol_df[protocol_df["protocol"] == PROTOCOL_B]
    expected_b = [round_budget(q, n_eval) for q in b["nominal_fraction"]]
    if not np.array_equal(b["selected_n"].to_numpy(), np.asarray(expected_b)):
        raise AssertionError("Protocol B does not use exact fixed absolute budget")
    if not (b.groupby(key_cols)["selected_n"].nunique() == 1).all():
        raise AssertionError("Protocol B selected_n differs across scenarios")

    # Protocol C threshold identical across scenarios for each condition/q.
    c = protocol_df[protocol_df["protocol"] == PROTOCOL_C]
    if c["threshold"].isna().any():
        raise AssertionError("Protocol C has missing thresholds")
    if not (c.groupby(key_cols)["threshold"].nunique() == 1).all():
        raise AssertionError("Protocol C threshold differs across scenarios")

    # Comparisons: exactly 3 protocols per condition/q.
    comp_counts = comparison_df.groupby(key_cols).size()
    if not (comp_counts == 3).all():
        raise AssertionError("Expected exactly 3 comparison rows per condition/q")

    # Decomposition exact.
    if (decomposition_df["tp_identity_error"] != 0).any():
        raise AssertionError("Non-zero TP three-term decomposition identity error")
    if (decomposition_df["tp_factorial_identity_error"] != 0).any():
        raise AssertionError("Non-zero TP factorial decomposition identity error")
    if (decomposition_df["identity_error"].abs() > 1e-12).any():
        raise AssertionError("Non-zero capture decomposition identity error")
    if (decomposition_df["factorial_identity_error"].abs() > 1e-12).any():
        raise AssertionError("Non-zero capture factorial identity error")
    if (decomposition_df["shapley_identity_error"].abs() > 1e-12).any():
        raise AssertionError("Non-zero Shapley identity error")
    if not np.allclose(
        decomposition_df["distortion_top_fraction_minus_fixed_budget"],
        decomposition_df["capacity_effect_within_available_subset"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError("Distortion != capacity component")
    if decomposition_df.groupby([
        "availability_target_fraction", "availability_relation",
        "model_quality", "nominal_fraction"
    ]).ngroups != 81:
        raise AssertionError("Expected exactly 81 synthetic conditions")


def build_manifest(
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    run_info: Mapping[str, Any],
) -> Dict[str, Any]:
    output_files = sorted(
        p for p in output_dir.iterdir() if p.is_file() and p.name != "synthetic_manifest.json"
    )
    output_hashes = {p.name: sha256_file(p) for p in output_files}

    sim = config["simulation"]
    ev = config["evaluation"]
    return {
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "git": capture_git_info(Path.cwd()),
        "packages": package_versions(),
        "design": {
            "seed": int(sim["seed"]),
            "replicates": int(sim["replicates"]),
            "population_n": int(sim["population_n"]),
            "calibration_n": int(sim["calibration_n"]),
            "outcome_prevalence_target": float(sim["outcome_prevalence"]),
            "outcome_risk_slope": float(sim["outcome_risk_slope"]),
            "availability_fractions": [float(x) for x in sim["availability_fractions"]],
            "availability_associations": sim["availability_associations"],
            "model_qualities": sim["model_qualities"],
            "evaluation_fractions": [float(x) for x in ev["fractions"]],
            "protocols": list(ev["protocols"]),
            "scenarios": list(ev["scenarios"]),
            "threshold_source": ev["threshold_source"],
            "tie_breaker_top_b": ev["tie_breaker_top_b"],
        },
        "solved_intercepts": {
            "outcome": run_info["outcome_intercept"],
            "availability": run_info["availability_intercepts"],
        },
        "invariants": {
            "same_score_vector_across_availability_scenarios": True,
            "same_population_event_denominator_across_scenarios": True,
            "protocol_A_exact_cohort_fraction_budget": True,
            "protocol_B_exact_equal_absolute_budget": True,
            "protocol_C_common_independent_calibration_threshold": True,
            "decomposition_identity_within_tolerance": True,
            "tp_decomposition_identity_exact": True,
            "distortion_equals_capacity_component": True,
            "four_cell_factorial_identity_exact": True,
            "symmetric_shapley_identity_exact": True,
            "exactly_81_condition_grid": True,
            "figure3_data_contains_all_81_conditions": True,
        },
        "outputs": {
            "protocol_rows": run_info["protocol_rows"],
            "comparison_rows": run_info["comparison_rows"],
            "decomposition_rows": run_info["decomposition_rows"],
            "condition_rows": run_info["condition_rows"],
            "comparison_summary_rows": run_info["comparison_summary_rows"],
            "decomposition_summary_rows": run_info["decomposition_summary_rows"],
            "shapley_replicate_rows": run_info["shapley_replicate_rows"],
            "figure3_rows": run_info["figure3_rows"],
            "sign_reversal_rows": run_info["sign_reversal_rows"],
            "ci_below_zero_rows": run_info["ci_below_zero_rows"],
            "max_abs_interaction": run_info["max_abs_interaction"],
        },
        "output_sha256": output_hashes,
        "elapsed_seconds": run_info["elapsed_seconds"],
    }



def make_smoke_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Create a deterministic small configuration for a local smoke test."""
    smoke = copy.deepcopy(dict(config))
    smoke["simulation"]["replicates"] = 2
    smoke["simulation"]["population_n"] = 5000
    smoke["simulation"]["calibration_n"] = 2500
    smoke["simulation"]["progress_every"] = 1
    smoke["output"]["overwrite"] = True
    return smoke

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the locked synthetic experiment YAML config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory override",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing non-empty output directory",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny 81-condition smoke test instead of the publication run",
    )
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    base_config = load_config(config_path)
    config = make_smoke_config(base_config) if args.smoke_test else base_config

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    elif args.smoke_test:
        output_dir = Path("/tmp/tae_synth_shapley81_smoke").resolve()
    else:
        output_dir = Path(config["output"]["directory"]).resolve()

    overwrite = bool(args.overwrite or args.smoke_test or config["output"].get("overwrite", False))
    prepare_output_dir(output_dir, overwrite)

    locked_config_path = output_dir / "synthetic_experiment.locked.yaml"
    with locked_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, sort_keys=False, allow_unicode=True)

    mode = "SMOKE TEST" if args.smoke_test else "FULL 81-CONDITION RUN"
    print(f"Running TAE synthetic Shapley experiment [{mode}] -> {output_dir}", flush=True)
    run_info = run_experiment(config, output_dir)

    manifest = build_manifest(config, config_path, output_dir, run_info)
    manifest_path = output_dir / "synthetic_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.smoke_test:
        if run_info["figure3_rows"] != 81:
            raise AssertionError("Smoke test did not produce all 81 condition summaries")
        print("TAE SYNTHETIC SHAPLEY 81-GRID SMOKE TEST PASS", flush=True)
        print("four_cell_identity=PASS", flush=True)
        print("shapley_identity=PASS", flush=True)
        print("all_81_conditions=PASS", flush=True)
        print("figure3_data=PASS", flush=True)
    else:
        print(
            "TAE SYNTHETIC SHAPLEY 81-GRID RUN PASS | "
            f"conditions={run_info['figure3_rows']} | "
            f"replicate_rows={run_info['shapley_replicate_rows']} | "
            f"sign_reversals={run_info['sign_reversal_rows']} | "
            f"ci_below_zero={run_info['ci_below_zero_rows']} | "
            f"max_abs_interaction={100.0 * run_info['max_abs_interaction']:.3f} pp | "
            f"elapsed={run_info['elapsed_seconds']:.2f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())