#!/usr/bin/env python3
"""
Generate TAE 2026 Figure 3 from the locked 81-condition synthetic experiment.

The script is intentionally data-driven:
- reads canonical synthetic CSV outputs;
- validates the expected 81-condition grid;
- validates four-cell and Shapley identities when replicate-level decomposition
  is available;
- computes condition-level means and empirical 95% simulation intervals;
- marks sign reversals and fixed-budget intervals entirely below zero;
- highlights the condition with the largest absolute interaction;
- writes publication-ready PNG/PDF/SVG plus machine-readable summary CSVs.

No values are hard-coded from a previous run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_AVAILABILITY = [0.50, 0.70, 0.90]
EXPECTED_RELATIONS = [
    "higher_risk_less_available",
    "risk_independent",
    "higher_risk_more_available",
]
EXPECTED_QUALITY = ["good", "medium", "weak"]
EXPECTED_Q = [0.05, 0.10, 0.20]
EXPECTED_PROTOCOLS = [
    "cohort_specific_top_fraction",
    "fixed_absolute_budget",
    "fixed_calibration_threshold",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--comparison",
        required=True,
        type=Path,
        help="synthetic_comparison_summary.csv",
    )
    p.add_argument(
        "--decomposition",
        required=True,
        type=Path,
        help="synthetic_decomposition.csv",
    )
    p.add_argument(
        "--ci-below-zero",
        default=None,
        type=Path,
        help="Optional synthetic_ci_below_zero.csv from the same run.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    p.add_argument(
        "--figure-stem",
        default="figure3_synthetic_81_conditions",
    )
    p.add_argument(
        "--dpi",
        default=300,
        type=int,
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Validate and render a small smoke figure, then exit.",
    )
    return p.parse_args()


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        fail(f"{name}: missing required columns: {missing}")


def pct(x):
    """Convert fractions to percentage points; works for scalars and pandas objects."""
    return 100.0 * x


def condition_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["availability_target_fraction"].round(6).astype(str)
        + "|"
        + df["availability_relation"].astype(str)
        + "|"
        + df["model_quality"].astype(str)
        + "|"
        + df["nominal_fraction"].round(6).astype(str)
    )


def validate_grid(decomp: pd.DataFrame) -> None:
    required = [
        "replicate",
        "availability_target_fraction",
        "availability_relation",
        "model_quality",
        "nominal_fraction",
        "c00_capture",
        "c10_capture",
        "c01_capture",
        "c11_capture",
        "naive_top_fraction_effect",
        "shapley_availability_effect",
        "shapley_capacity_effect",
        "interaction_effect",
        "total_four_cell_effect",
        "factorial_identity_error",
        "shapley_identity_error",
    ]
    require_columns(decomp, required, "decomposition")

    combos = (
        decomp[
            [
                "availability_target_fraction",
                "availability_relation",
                "model_quality",
                "nominal_fraction",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "availability_target_fraction",
                "availability_relation",
                "model_quality",
                "nominal_fraction",
            ]
        )
    )

    if len(combos) != 81:
        fail(f"Expected 81 unique synthetic conditions, found {len(combos)}.")

    if sorted(combos["availability_target_fraction"].unique().tolist()) != EXPECTED_AVAILABILITY:
        fail("Unexpected availability grid.")

    if sorted(combos["availability_relation"].unique().tolist()) != sorted(EXPECTED_RELATIONS):
        fail("Unexpected availability-relation grid.")

    if sorted(combos["model_quality"].unique().tolist()) != sorted(EXPECTED_QUALITY):
        fail("Unexpected model-quality grid.")

    if sorted(combos["nominal_fraction"].unique().tolist()) != EXPECTED_Q:
        fail("Unexpected budget grid.")

    counts = decomp.groupby(
        [
            "availability_target_fraction",
            "availability_relation",
            "model_quality",
            "nominal_fraction",
        ]
    )["replicate"].nunique()

    if counts.min() != counts.max() or counts.min() < 2:
        fail(f"Each condition must have the same >=2 replicates; got {counts.describe().to_dict()}.")


def validate_identities(decomp: pd.DataFrame) -> dict:
    max_factorial = float(decomp["factorial_identity_error"].abs().max())
    max_shapley = float(decomp["shapley_identity_error"].abs().max())
    max_total = float(
        (
            decomp["naive_top_fraction_effect"]
            - decomp["total_four_cell_effect"]
        ).abs().max()
    )

    if max_factorial > 1e-10:
        fail(f"Factorial identity failed: max error={max_factorial:.3e}")
    if max_shapley > 1e-10:
        fail(f"Shapley identity failed: max error={max_shapley:.3e}")
    if max_total > 1e-10:
        fail(f"Naive/total identity failed: max error={max_total:.3e}")

    return {
        "max_factorial_identity_error": max_factorial,
        "max_shapley_identity_error": max_shapley,
        "max_naive_total_identity_error": max_total,
    }


def aggregate_decomposition(decomp: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "availability_target_fraction",
        "availability_relation",
        "model_quality",
        "nominal_fraction",
    ]

    rows = []
    for key, g in decomp.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, key))
        row["replicates"] = int(g["replicate"].nunique())

        for col in [
            "c00_capture",
            "c10_capture",
            "c01_capture",
            "c11_capture",
            "naive_top_fraction_effect",
            "availability_effect_at_all_budget",
            "shapley_availability_effect",
            "shapley_capacity_effect",
            "interaction_effect",
            "total_four_cell_effect",
        ]:
            vals = g[col].to_numpy(dtype=float)
            row[f"mean_{col}"] = float(np.mean(vals))
            row[f"q025_{col}"] = float(np.quantile(vals, 0.025))
            row[f"q975_{col}"] = float(np.quantile(vals, 0.975))

        naive = g["naive_top_fraction_effect"].to_numpy(dtype=float)
        fixed = g["total_four_cell_effect"].to_numpy(dtype=float)
        row["mean_distortion_top_fraction_minus_fixed_budget"] = float(
            np.mean(naive - fixed)
        )
        row["q025_distortion_top_fraction_minus_fixed_budget"] = float(
            np.quantile(naive - fixed, 0.025)
        )
        row["q975_distortion_top_fraction_minus_fixed_budget"] = float(
            np.quantile(naive - fixed, 0.975)
        )

        # Fixed-budget early -> full cohort effect is C11 - C01,
        # i.e. availability_effect_at_all_budget.  total_four_cell_effect
        # is instead C11 - C00 (the naive top-q contrast).
        fixed_q025 = row["q025_availability_effect_at_all_budget"]
        fixed_q975 = row["q975_availability_effect_at_all_budget"]
        naive_mean = row["mean_naive_top_fraction_effect"]
        fixed_mean = row["mean_availability_effect_at_all_budget"]

        row["sign_reversal"] = bool(naive_mean > 0 and fixed_mean < 0)
        row["fixed_budget_interval_below_zero"] = bool(fixed_q975 < 0)
        row["fixed_budget_interval_above_zero"] = bool(fixed_q025 > 0)
        row["interaction_abs_mean"] = abs(row["mean_interaction_effect"])
        row["interaction_abs_q975"] = max(
            abs(row["q025_interaction_effect"]),
            abs(row["q975_interaction_effect"]),
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) != 81:
        fail(f"Aggregated decomposition has {len(out)} conditions, expected 81.")

    return out


def add_condition_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    relation_short = {
        "higher_risk_less_available": "risk ↓ availability",
        "risk_independent": "independent",
        "higher_risk_more_available": "risk ↑ availability",
    }
    out["relation_short"] = out["availability_relation"].map(relation_short)
    out["row_label"] = (
        (out["availability_target_fraction"] * 100).round().astype(int).astype(str)
        + "% / "
        + out["relation_short"]
    )
    out["quality_q"] = (
        out["model_quality"].str.capitalize()
        + " / "
        + (out["nominal_fraction"] * 100).round().astype(int).astype(str)
        + "%"
    )
    return out


def make_grid(df: pd.DataFrame, value_col: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    # 9 rows: availability target × availability relation.
    # 9 columns: model quality × q.
    row_order = [
        f"{int(a*100)}% / {rel}"
        for a in EXPECTED_AVAILABILITY
        for rel in [
            "risk ↓ availability",
            "independent",
            "risk ↑ availability",
        ]
    ]
    col_order = [
        f"{q} / {int(frac*100)}%"
        for q in ["Good", "Medium", "Weak"]
        for frac in EXPECTED_Q
    ]

    tmp = df.copy()
    pivot = tmp.pivot(index="row_label", columns="quality_q", values=value_col)
    pivot = pivot.reindex(index=row_order, columns=col_order)
    return pivot, row_order, col_order


def save_grid_csv(df: pd.DataFrame, value_col: str, path: Path) -> None:
    pivot, _, _ = make_grid(df, value_col)
    pivot.to_csv(path)


def draw_heatmap(
    ax,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    annotate: bool = False,
    mark_sign_reversal: bool = False,
    mark_ci_below_zero: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
):
    pivot, rows, cols = make_grid(df, value_col)
    arr = pivot.to_numpy(dtype=float) * 100.0
    finite = arr[np.isfinite(arr)]

    if vmin is None:
        lim = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))))
        vmin, vmax = -lim, lim

    im = ax.imshow(arr, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    ax.set_xlabel("Model quality / referral fraction")
    ax.set_ylabel("Availability / relation")

    for i, row_label in enumerate(rows):
        for j, col_label in enumerate(cols):
            value = arr[i, j]
            if np.isfinite(value) and annotate:
                ax.text(
                    j, i, f"{value:.1f}",
                    ha="center", va="center", fontsize=6
                )

            mask = (
                (df["row_label"] == row_label)
                & (df["quality_q"] == col_label)
            )
            if not mask.any():
                continue
            r = df.loc[mask].iloc[0]

            if mark_sign_reversal and bool(r["sign_reversal"]):
                ax.scatter(
                    j, i, s=55, marker="o", zorder=5
                )

            if mark_ci_below_zero and bool(r["fixed_budget_interval_below_zero"]):
                ax.scatter(
                    j, i, s=18, marker="x", zorder=6
                )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("percentage points", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return im


def build_figure(df: pd.DataFrame, output_dir: Path, stem: str, dpi: int) -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "figure.titlesize": 13,
    })

    fig, axes = plt.subplots(
        2, 2,
        figsize=(15, 11),
        constrained_layout=True,
    )

    # ------------------------------------------------------------------
    # Panel A: naive vs fixed-budget effect across all 81 conditions.
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    x = pct(df["mean_naive_top_fraction_effect"])
    # Y is the fixed-budget early -> full cohort contrast C11 - C01.
    y = pct(df["mean_availability_effect_at_all_budget"])
    yerr_low = y - pct(df["q025_availability_effect_at_all_budget"])
    yerr_high = pct(df["q975_availability_effect_at_all_budget"]) - y

    normal = ~df["sign_reversal"].to_numpy(dtype=bool)
    reversal = df["sign_reversal"].to_numpy(dtype=bool)
    ci_below = df["fixed_budget_interval_below_zero"].to_numpy(dtype=bool)

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    lo = min(float(x.min()), float(y.min()), -1)
    hi = max(float(x.max()), float(y.max()), 1)
    ax.plot([lo, hi], [lo, hi], linewidth=0.8)

    ax.errorbar(
        x[normal],
        y[normal],
        yerr=np.vstack([yerr_low[normal], yerr_high[normal]]),
        fmt="o",
        markersize=3.5,
        alpha=0.45,
        linewidth=0.6,
        capsize=1.5,
        label="No sign reversal",
    )
    ax.errorbar(
        x[reversal],
        y[reversal],
        yerr=np.vstack([yerr_low[reversal], yerr_high[reversal]]),
        fmt="^",
        markersize=5.5,
        alpha=0.95,
        linewidth=0.8,
        capsize=2,
        label="Sign reversal",
    )

    # Extra black X for conditions where the entire fixed-budget interval < 0.
    for i in np.where(ci_below)[0]:
        ax.scatter(
            x.iloc[i], y.iloc[i],
            marker="x", s=32, linewidths=1.0, zorder=7
        )

    max_int_idx = df["interaction_abs_mean"].idxmax()
    max_row = df.loc[max_int_idx]
    mx = pct(max_row["mean_naive_top_fraction_effect"])
    my = pct(max_row["mean_total_four_cell_effect"])
    ax.annotate(
        "max |interaction|",
        xy=(mx, my),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", linewidth=0.7),
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Naive cohort-specific top-q effect (pp)")
    ax.set_ylabel("Fixed-budget effect (pp)")
    ax.set_title("A. Naive evaluation can change the qualitative conclusion")
    ax.legend(fontsize=7, loc="upper left", frameon=False)

    # ------------------------------------------------------------------
    # Panel B: fixed-budget effect across all 81 conditions.
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    fixed_lim = max(
        abs(pct(df["q025_availability_effect_at_all_budget"]).min()),
        abs(pct(df["q975_availability_effect_at_all_budget"]).max()),
        abs(pct(df["mean_availability_effect_at_all_budget"]).min()),
        abs(pct(df["mean_availability_effect_at_all_budget"]).max()),
    )
    draw_heatmap(
        ax,
        df,
        "mean_availability_effect_at_all_budget",
        "B. Fixed-budget population-capture effect across all 81 conditions",
        annotate=False,
        mark_sign_reversal=True,
        mark_ci_below_zero=True,
        vmin=-fixed_lim,
        vmax=fixed_lim,
    )

    # ------------------------------------------------------------------
    # Panel C: Shapley availability vs capacity, all 81 conditions.
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    xa = pct(df["mean_shapley_availability_effect"])
    yc = pct(df["mean_shapley_capacity_effect"])
    size = 25 + 160 * (
        np.abs(df["mean_interaction_effect"].to_numpy()) /
        max(np.abs(df["mean_interaction_effect"]).max(), 1e-12)
    )

    relation_markers = {
        "higher_risk_less_available": "o",
        "risk_independent": "s",
        "higher_risk_more_available": "^",
    }

    for rel, marker in relation_markers.items():
        mask = df["availability_relation"].eq(rel).to_numpy()
        ax.scatter(
            xa[mask], yc[mask],
            s=size[mask],
            marker=marker,
            alpha=0.55,
            label=rel.replace("_", " "),
        )

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Shapley availability effect (pp)")
    ax.set_ylabel("Shapley capacity effect (pp)")
    ax.set_title("C. Shapley components across all 81 conditions")
    ax.legend(fontsize=7, frameon=False, loc="best")

    # Highlight the maximum interaction.
    ax.scatter(
        xa[max_int_idx], yc[max_int_idx],
        s=120, marker="o", zorder=8
    )

    # ------------------------------------------------------------------
    # Panel D: interaction heatmap.
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    interaction_lim = max(
        abs(pct(df["mean_interaction_effect"]).min()),
        abs(pct(df["mean_interaction_effect"]).max()),
    )
    if interaction_lim == 0:
        interaction_lim = 1.0

    draw_heatmap(
        ax,
        df,
        "mean_interaction_effect",
        "D. Availability × capacity interaction",
        annotate=True,
        mark_sign_reversal=False,
        mark_ci_below_zero=False,
        vmin=-interaction_lim,
        vmax=interaction_lim,
    )

    max_pos = df.reset_index(drop=True)["interaction_abs_mean"].idxmax()
    # Map max row to its heatmap position.
    max_row2 = df.reset_index(drop=True).iloc[max_pos]
    row_idx = make_grid(df, "mean_interaction_effect")[1].index(max_row2["row_label"])
    col_idx = make_grid(df, "mean_interaction_effect")[2].index(max_row2["quality_q"])
    ax.scatter(
        col_idx, row_idx,
        s=100, marker="o", zorder=8
    )

    fig.suptitle(
        "Figure 3. Synthetic stress test of cohort-specific top-q evaluation",
        fontsize=14,
    )

    fig.text(
        0.5, 0.005,
        "Each point/cell is one of 81 conditions "
        "(3 availability levels × 3 availability–outcome relations × "
        "3 model-quality levels × 3 referral fractions). "
        "Error bars and intervals are empirical 95% simulation intervals across replicates. "
        "Fixed-budget effect is C11 − C01; naive top-q effect is C11 − C00. "
        "Open circles mark naive > 0 but fixed-budget < 0; crosses mark fixed-budget intervals entirely below zero. "
        "Point size in panel C scales with |interaction|.",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    out_png = output_dir / f"{stem}.png"
    out_pdf = output_dir / f"{stem}.pdf"
    out_svg = output_dir / f"{stem}.svg"

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    df: pd.DataFrame,
    identity_info: dict,
    output_dir: Path,
    decomp: pd.DataFrame,
) -> None:
    reversal = df[df["sign_reversal"]].copy()
    below = df[df["fixed_budget_interval_below_zero"]].copy()
    max_idx = df["interaction_abs_mean"].idxmax()
    max_row = df.loc[max_idx]
    max_rep_idx = decomp["interaction_effect"].abs().idxmax()
    max_rep_row = decomp.loc[max_rep_idx]

    summary = {
        "n_conditions": int(len(df)),
        "n_sign_reversals": int(len(reversal)),
        "n_fixed_budget_intervals_below_zero": int(len(below)),
        "max_abs_mean_interaction_pp": float(
            pct(max_row["interaction_abs_mean"])
        ),
        "max_interaction_condition": {
            "availability_target_fraction": float(max_row["availability_target_fraction"]),
            "availability_relation": str(max_row["availability_relation"]),
            "model_quality": str(max_row["model_quality"]),
            "nominal_fraction": float(max_row["nominal_fraction"]),
        },
        "max_interaction_components_pp": {
            "availability": float(pct(max_row["mean_shapley_availability_effect"])),
            "capacity": float(pct(max_row["mean_shapley_capacity_effect"])),
            "interaction": float(pct(max_row["mean_interaction_effect"])),
            "naive": float(pct(max_row["mean_naive_top_fraction_effect"])),
            "fixed_budget": float(pct(max_row["mean_availability_effect_at_all_budget"])),
        },
        "max_abs_replicate_interaction_pp": float(pct(abs(max_rep_row["interaction_effect"]))),
        "max_abs_replicate_interaction_condition": {
            "availability_target_fraction": float(max_rep_row["availability_target_fraction"]),
            "availability_relation": str(max_rep_row["availability_relation"]),
            "model_quality": str(max_rep_row["model_quality"]),
            "nominal_fraction": float(max_rep_row["nominal_fraction"]),
            "replicate": int(max_rep_row["replicate"]),
        },
        **identity_info,
    }

    (output_dir / "figure3_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    reversal_cols = [
        "availability_target_fraction",
        "availability_relation",
        "model_quality",
        "nominal_fraction",
        "mean_naive_top_fraction_effect",
        "q025_naive_top_fraction_effect",
        "q975_naive_top_fraction_effect",
        "mean_availability_effect_at_all_budget",
        "q025_availability_effect_at_all_budget",
        "q975_availability_effect_at_all_budget",
        "mean_total_four_cell_effect",
        "q025_total_four_cell_effect",
        "q975_total_four_cell_effect",
        "mean_shapley_availability_effect",
        "mean_shapley_capacity_effect",
        "mean_interaction_effect",
        "sign_reversal",
        "fixed_budget_interval_below_zero",
    ]
    reversal[reversal_cols].to_csv(
        output_dir / "figure3_sign_reversals.csv",
        index=False,
    )

    below[reversal_cols].to_csv(
        output_dir / "figure3_fixed_budget_ci_below_zero.csv",
        index=False,
    )

    df.to_csv(output_dir / "figure3_condition_level_data.csv", index=False)

    # Grid sources for downstream figure editing.
    save_grid_csv(df, "mean_naive_top_fraction_effect",
                  output_dir / "grid_naive_effect.csv")
    save_grid_csv(df, "mean_total_four_cell_effect",
                  output_dir / "grid_fixed_budget_effect.csv")
    save_grid_csv(df, "mean_shapley_availability_effect",
                  output_dir / "grid_shapley_availability.csv")
    save_grid_csv(df, "mean_shapley_capacity_effect",
                  output_dir / "grid_shapley_capacity.csv")
    save_grid_csv(df, "mean_interaction_effect",
                  output_dir / "grid_interaction.csv")


def write_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    df: pd.DataFrame,
) -> None:
    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    manifest = {
        "script": Path(__file__).name,
        "script_sha256": sha256(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "comparison_input": str(args.comparison.resolve()),
        "comparison_sha256": sha256(args.comparison),
        "decomposition_input": str(args.decomposition.resolve()),
        "decomposition_sha256": sha256(args.decomposition),
        "ci_below_zero_input": (
            str(args.ci_below_zero.resolve())
            if args.ci_below_zero else None
        ),
        "n_conditions": int(len(df)),
        "expected_grid": "3 availability × 3 relation × 3 model quality × 3 q = 81",
        "interval_type": "empirical 95% simulation interval across replicates",
        "hardcoded_effect_values": False,
    }
    if args.ci_below_zero:
        manifest["ci_below_zero_sha256"] = sha256(args.ci_below_zero)

    (output_dir / "figure3_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(args.comparison)
    decomposition = pd.read_csv(args.decomposition)

    require_columns(
        comparison,
        [
            "availability_target_fraction",
            "availability_relation",
            "model_quality",
            "nominal_fraction",
            "protocol",
            "mean_delta_population_event_capture",
        ],
        "comparison",
    )

    unexpected_protocols = sorted(set(comparison["protocol"]) - set(EXPECTED_PROTOCOLS))
    if unexpected_protocols:
        fail(f"Unexpected protocol names: {unexpected_protocols}")

    validate_grid(decomposition)
    identity_info = validate_identities(decomposition)
    df = add_condition_labels(aggregate_decomposition(decomposition))

    # Cross-check condition count and protocol-level source coverage.
    if comparison.groupby(
        [
            "availability_target_fraction",
            "availability_relation",
            "model_quality",
            "nominal_fraction",
            "protocol",
        ]
    ).size().shape[0] != 81 * 3:
        fail("Comparison summary does not contain exactly 81 × 3 protocol rows.")

    if args.ci_below_zero:
        ci = pd.read_csv(args.ci_below_zero)
        require_columns(
            ci,
            [
                "condition_id",
                "naive_fixed_budget_sign_reversal",
                "fixed_budget_interval_below_zero",
            ],
            "ci-below-zero",
        )

    if args.smoke_test:
        # Render only the first 9 conditions to a tiny smoke image after all
        # validation. The actual figure is generated only in the full run.
        smoke = df.head(9).copy()
        smoke_dir = args.output_dir / "smoke"
        smoke_dir.mkdir(exist_ok=True)
        plt.figure(figsize=(4, 3))
        plt.scatter(
            pct(smoke["mean_naive_top_fraction_effect"]),
            pct(smoke["mean_total_four_cell_effect"]),
            s=20,
        )
        plt.xlabel("Naive effect (pp)")
        plt.ylabel("Fixed-budget effect (pp)")
        plt.tight_layout()
        plt.savefig(smoke_dir / "smoke.png", dpi=100)
        plt.close()
        print("TAE FIGURE 3 SMOKE TEST PASS")
        print(f"conditions_validated={len(df)}")
        print("four_cell_identity=PASS")
        print("shapley_identity=PASS")
        print("grid=81=PASS")
        print("smoke_render=PASS")
        return 0

    write_summary(df, identity_info, args.output_dir, decomposition)
    build_figure(df, args.output_dir, args.figure_stem, args.dpi)
    write_manifest(args, args.output_dir, df)

    reversal_count = int(df["sign_reversal"].sum())
    below_count = int(df["fixed_budget_interval_below_zero"].sum())
    max_row = df.loc[df["interaction_abs_mean"].idxmax()]
    max_rep_row = decomposition.loc[decomposition["interaction_effect"].abs().idxmax()]

    print("TAE FIGURE 3 GENERATION PASS")
    print(f"conditions={len(df)}")
    print(f"sign_reversals={reversal_count}")
    print(f"fixed_budget_intervals_below_zero={below_count}")
    print(
        "max_abs_mean_interaction_pp="
        f"{pct(max_row['interaction_abs_mean']):.3f}"
    )
    print(
        "max_abs_replicate_interaction_pp="
        f"{pct(abs(max_rep_row['interaction_effect'])):.3f}"
    )
    print(f"output_dir={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
