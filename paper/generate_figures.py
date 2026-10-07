from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle


DATA = ROOT / "source_data"
OUT = ROOT / "figures"

TARGET_ORDER = ["target_preterm", "target_nicu", "target_lbw"]
TARGET_LABEL = {
    "target_preterm": "Preterm birth",
    "target_nicu": "NICU admission",
    "target_lbw": "Low birth weight",
}
TARGET_SHORT = {
    "target_preterm": "Preterm",
    "target_nicu": "NICU",
    "target_lbw": "LBW",
}
TARGET_COLOR = {
    "target_preterm": "#0072B2",
    "target_nicu": "#D55E00",
    "target_lbw": "#009E73",
}

NAVY = "#17324D"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#777777"
LIGHT_GRAY = "#E6E8EB"
GRID = "#D9DDE2"
INK = "#20252B"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 10.7,
            "xtick.labelsize": 10.7,
            "ytick.labelsize": 10.7,
            "legend.fontsize": 10.7,
            "axes.edgecolor": "#8C939C",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [OUT / f"{stem}.pdf", OUT / f"{stem}.png"]
    # Preserve the designed width and the font scale at a 6.5-inch insertion.
    fig.savefig(paths[0], facecolor="white")
    fig.savefig(paths[1], dpi=320, facecolor="white")
    plt.close(fig)
    return paths


SYNTHETIC_LEVELS = {
    "availability_relation": [
        "higher_risk_less_available", "risk_independent", "higher_risk_more_available"
    ],
    "model_quality": ["good", "medium", "weak"],
    "availability_target_fraction": [0.5, 0.7, 0.9],
    "nominal_fraction": [0.05, 0.10, 0.20],
}
ZOOM_X = (-3.5, 0.5)
ZOOM_Y = (0.0, 18.0)


def validate_synthetic(synthetic: pd.DataFrame) -> None:
    keys = list(SYNTHETIC_LEVELS)
    expected = pd.MultiIndex.from_product(SYNTHETIC_LEVELS.values(), names=keys)
    observed = pd.MultiIndex.from_frame(synthetic[keys])
    if (
        len(synthetic) != len(expected)
        or not observed.is_unique
        or not synthetic["condition_id"].is_unique
        or synthetic["condition_id"].isna().any()
        or len(expected.difference(observed))
        or len(observed.difference(expected))
    ):
        raise ValueError("Synthetic data must contain all 81 unique condition keys and IDs")

    numeric = [
        "mean_naive_top_fraction_effect",
        "mean_availability_effect_at_all_budget",
        "q025_availability_effect_at_all_budget",
        "q975_availability_effect_at_all_budget",
        "sign_reversal_replicate_fraction",
    ]
    if not np.isfinite(synthetic[numeric].to_numpy()).all():
        raise ValueError("Synthetic plot inputs must be finite")
    lower = synthetic["q025_availability_effect_at_all_budget"]
    upper = synthetic["q975_availability_effect_at_all_budget"]
    if (lower > upper).any():
        raise ValueError("Synthetic fixed-budget interval bounds are reversed")
    if not synthetic["sign_reversal_replicate_fraction"].between(0, 1).all():
        raise ValueError("Synthetic reversal frequencies must lie in [0, 1]")

    expected_flags = {
        "naive_fixed_budget_sign_reversal": (
            (synthetic["mean_naive_top_fraction_effect"] > 0)
            & (synthetic["mean_availability_effect_at_all_budget"] < 0)
        ),
        "fixed_budget_interval_below_zero": upper < 0,
        "fixed_budget_interval_above_zero": lower > 0,
    }
    for name, expected_flag in expected_flags.items():
        if (
            not pd.api.types.is_bool_dtype(synthetic[name])
            or not synthetic[name].eq(expected_flag).all()
        ):
            raise ValueError(f"Synthetic flag {name} is inconsistent with its effects")


def load_inputs() -> dict[str, pd.DataFrame]:
    paths = {
        "cohort": DATA / "cohort_counts_validation.csv",
        "shapley": DATA / "shapley_main_results.csv",
        "bootstrap": DATA / "shapley_bootstrap_replicates.csv",
        "synthetic": DATA / "synthetic_figure3_data.csv",
    }
    frames = {name: pd.read_csv(path) for name, path in paths.items()}

    shapley = frames["shapley"]
    if len(shapley) != 6:
        raise ValueError(f"Expected 6 Shapley rows, found {len(shapley)}")
    if set(shapley["target"]) != set(TARGET_ORDER):
        raise ValueError("Unexpected Shapley targets")
    if set(np.round(shapley["nominal_fraction"], 2)) != {0.05, 0.10}:
        raise ValueError("Expected q=5% and q=10%")
    identity_error = np.max(
        np.abs(
            shapley["total_topq_contrast_point"]
            - shapley["shapley_availability_point"]
            - shapley["shapley_capacity_point"]
        )
    )
    if identity_error > 1e-12:
        raise ValueError(f"Shapley identity failed: {identity_error}")

    bootstrap = frames["bootstrap"]
    expected_bootstrap = 3 * 2 * 500
    if len(bootstrap) != expected_bootstrap:
        raise ValueError(
            f"Expected {expected_bootstrap} bootstrap rows, found {len(bootstrap)}"
        )
    group_sizes = bootstrap.groupby(["target", "nominal_fraction"]).size()
    if not (group_sizes == 500).all():
        raise ValueError("Every target/q condition must have 500 bootstrap replicates")
    if bootstrap[["identity_factorial_error", "identity_shapley_error"]].abs().max().max() > 1e-12:
        raise ValueError("Bootstrap decomposition identity failed")

    synthetic = frames["synthetic"]
    validate_synthetic(synthetic)
    if int(synthetic["naive_fixed_budget_sign_reversal"].sum()) != 18:
        raise ValueError("Expected 18 mean sign reversals")
    if int(synthetic["fixed_budget_interval_below_zero"].sum()) != 9:
        raise ValueError("Expected 9 robust negative fixed-budget effects")

    cohort = frames["cohort"]
    required_groups = {"early_entry", "later_entry", "no_care", "unknown_care"}
    observed = set(cohort.loc[cohort["year"] == 2023, "scenario"])
    if not required_groups.issubset(observed):
        raise ValueError("CDC 2023 care-entry groups are incomplete")

    return frames


def figure1(shapley: pd.DataFrame) -> tuple[list[Path], pd.DataFrame]:
    q10 = (
        shapley[np.isclose(shapley["nominal_fraction"], 0.10)]
        .set_index("target")
        .loc[TARGET_ORDER]
        .reset_index()
    )
    q10["workload_increase_pct"] = 100 * (
        q10["B_all_point"] / q10["B_early_point"] - 1
    )
    q10["capture_relative_increase_pct"] = 100 * (
        q10["C11_all_Ball"] / q10["C00_early_Bearly"] - 1
    )
    q10["capacity_share_pct"] = 100 * (
        q10["shapley_capacity_point"] / q10["total_topq_contrast_point"]
    )
    q10.to_csv(DATA / "figure1_headline_data.csv", index=False)

    fig, (ax_workload, ax_gain) = plt.subplots(
        1,
        2,
        figsize=(8.6, 3.6),
        gridspec_kw={"width_ratios": [1.06, 1.14]},
    )

    mean_early_budget = q10["B_early_point"].mean()
    mean_all_budget = q10["B_all_point"].mean()
    mean_early_n = mean_early_budget / 0.10
    mean_all_n = mean_all_budget / 0.10
    y = np.array([1, 0])
    candidate_n = np.array([mean_early_n, mean_all_n]) / 1_000_000
    selected_n = np.array([mean_early_budget, mean_all_budget]) / 1_000_000

    ax_workload.barh(y, candidate_n, height=0.54, color="#E2E7EC", zorder=1)
    ax_workload.barh(y, selected_n, height=0.54, color=NAVY, zorder=2)
    for ypos, pool, chosen in zip(y, candidate_n, selected_n):
        ax_workload.text(
            chosen + 0.05,
            ypos + 0.13,
            f"{chosen * 1000:.0f}k selected",
            ha="left",
            va="center",
            color=NAVY,
            fontsize=10.7,
            weight="bold",
        )
        ax_workload.text(
            pool - 0.08,
            ypos - 0.13,
            f"{pool:.2f}M candidates",
            ha="right",
            va="center",
            color=INK,
            fontsize=10.7,
        )
    increase = 100 * (mean_all_budget / mean_early_budget - 1)
    ax_workload.text(
        1.72,
        0.50,
        f"+{increase:.1f}% selected workload",
        ha="center",
        va="center",
        color=ORANGE,
        fontsize=10.7,
        weight="bold",
    )
    ax_workload.set_yticks(y, ["Early entry", "All-record\nupper bound"])
    ax_workload.set_xlim(0, 3.72)
    ax_workload.set_xlabel("Target-specific cohort size (millions)")
    ax_workload.set_title("(a) Same fraction, different workload", loc="left", weight="bold", fontsize=11.5)
    ax_workload.spines[["top", "right", "left"]].set_visible(False)
    ax_workload.tick_params(axis="y", length=0)
    ax_workload.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)

    ypos = np.arange(len(TARGET_ORDER))[::-1]
    eligibility = 100 * q10["shapley_availability_point"].to_numpy()
    capacity = 100 * q10["shapley_capacity_point"].to_numpy()
    total = eligibility + capacity
    ax_gain.barh(ypos, eligibility, color=BLUE, height=0.55, label="Eligibility", zorder=2)
    ax_gain.barh(
        ypos,
        capacity,
        left=eligibility,
        color=ORANGE,
        height=0.55,
        label="Additional capacity",
        zorder=2,
    )
    for index, ypos_value in enumerate(ypos):
        ax_gain.text(
            eligibility[index] / 2,
            ypos_value,
            f"{eligibility[index]:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=10.7,
            weight="bold",
        )
        ax_gain.text(
            eligibility[index] + capacity[index] / 2,
            ypos_value,
            f"{capacity[index]:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=10.7,
            weight="bold",
        )
        ax_gain.text(
            total[index] + 0.18,
            ypos_value,
            f"{total[index]:.2f} pp\n{q10.loc[index, 'capacity_share_pct']:.1f}% capacity",
            ha="left",
            va="center",
            color=INK,
            fontsize=10.7,
        )
    ax_gain.set_yticks(ypos, [TARGET_SHORT[target] for target in TARGET_ORDER])
    ax_gain.set_xlim(0, total.max() + 3.5)
    ax_gain.set_xlabel("Increase in population capture (pp)")
    ax_gain.set_title("(b) Allocation of the joint gain", loc="left", weight="bold", fontsize=11.5)
    ax_gain.spines[["top", "right", "left"]].set_visible(False)
    ax_gain.tick_params(axis="y", length=0)
    ax_gain.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax_gain.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, -0.24),
        ncol=2,
    )

    fig.subplots_adjust(left=0.12, right=0.985, top=0.87, bottom=0.28, wspace=0.35)
    return save(fig, "figure1_headline"), q10


def figure_method() -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.6, 2.95))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, face: str, edge: str = "#A7AFB8") -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.0,
            )
        )

    box(0.015, 0.17, 0.205, 0.69, "#F5F6F7")
    ax.text(0.03, 0.79, "1. Declare the policy", weight="bold", fontsize=10.2, color=INK)
    ax.text(
        0.03,
        0.67,
        "Target population  $P$\nEligibility rule  $S$\nAbsolute budget  $b$\nFrozen scores  $r$",
        fontsize=8.9,
        va="top",
        linespacing=1.52,
        color=INK,
    )

    ax.annotate(
        "",
        xy=(0.268, 0.515),
        xytext=(0.225, 0.52),
        arrowprops={"arrowstyle": "-|>", "color": "#7B838C", "lw": 1.4},
    )

    ax.text(0.285, 0.91, "2. Evaluate the complete 2 x 2 design", weight="bold", fontsize=10.2, color=INK)
    ax.text(0.485, 0.79, "Early budget  $b_e$", ha="center", fontsize=8.8, weight="bold", color=NAVY)
    ax.text(0.655, 0.79, "Expanded budget  $b_a$", ha="center", fontsize=8.8, weight="bold", color=ORANGE)
    ax.text(0.285, 0.605, "Early eligible\n$S_e$", ha="left", va="center", fontsize=8.8, weight="bold", color=INK)
    ax.text(0.285, 0.335, "All records\n$S_a$", ha="left", va="center", fontsize=8.8, weight="bold", color=INK)

    cells = [
        (0.405, 0.50, "#E8F2F8", NAVY, "$C_{00}$\n$S_e, b_e$"),
        (0.575, 0.50, "#FBEDE4", ORANGE, "$C_{01}$\n$S_e, b_a$"),
        (0.405, 0.23, "#E8F2F8", NAVY, "$C_{10}$\n$S_a, b_e$"),
        (0.575, 0.23, "#FBEDE4", ORANGE, "$C_{11}$\n$S_a, b_a$"),
    ]
    for x, y, face, edge, label in cells:
        box(x, y, 0.14, 0.18, face, edge)
        ax.text(x + 0.07, y + 0.09, label, ha="center", va="center", fontsize=9.7, color=INK)

    ax.text(
        0.56,
        0.075,
        "Same scores and full-population event denominator in all four cells",
        ha="center",
        fontsize=8.1,
        color="#525A63",
    )

    ax.annotate(
        "",
        xy=(0.765, 0.515),
        xytext=(0.72, 0.515),
        arrowprops={"arrowstyle": "-|>", "color": "#7B838C", "lw": 1.4},
    )
    box(0.775, 0.17, 0.21, 0.69, "#F5F6F7")
    ax.text(0.792, 0.79, "3. Report the decision", weight="bold", fontsize=10.2, color=INK)
    ax.text(
        0.792,
        0.67,
        "Eligible $n$ and event ceiling\nSelected $n$ and precision\nEligible recall\nPopulation event capture\nShapley components\nInteraction",
        fontsize=8.25,
        va="top",
        linespacing=1.34,
        color=INK,
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    return save(fig, "figure2_four_cell_audit")


def figure2(
    shapley: pd.DataFrame, bootstrap: pd.DataFrame
) -> tuple[list[Path], pd.DataFrame]:
    point_rows = []
    for _, row in shapley.iterrows():
        point_rows.append(
            {
                "target": row["target"],
                "nominal_fraction": row["nominal_fraction"],
                "capacity_share_pct": 100
                * row["shapley_capacity_point"]
                / row["total_topq_contrast_point"],
            }
        )
    point = pd.DataFrame(point_rows)

    boot = bootstrap.copy()
    boot["capacity_share_pct"] = 100 * boot["shapley_capacity"] / boot["total_topq_contrast"]
    share_ci = (
        boot.groupby(["target", "nominal_fraction"])["capacity_share_pct"]
        .agg(
            ci_low=lambda values: np.quantile(values, 0.025),
            ci_high=lambda values: np.quantile(values, 0.975),
        )
        .reset_index()
        .merge(point, on=["target", "nominal_fraction"], how="left")
    )
    share_ci.to_csv(DATA / "figure2_capacity_share.csv", index=False)

    row_order = [
        (target, fraction)
        for target in TARGET_ORDER
        for fraction in (0.05, 0.10)
    ]
    lookup = shapley.set_index(["target", "nominal_fraction"])
    y = np.arange(len(row_order))[::-1]

    fig, (ax_effect, ax_share) = plt.subplots(
        1,
        2,
        figsize=(8.6, 3.55),
        gridspec_kw={"width_ratios": [1.34, 0.92]},
    )

    components = [
        (
            "Total cohort-relative contrast",
            "total_topq_contrast_point_pp",
            "total_topq_contrast_ci_lower_pp",
            "total_topq_contrast_ci_upper_pp",
            NAVY,
            "D",
            0.22,
        ),
        (
            "Shapley eligibility",
            "shapley_availability_point_pp",
            "shapley_availability_ci_lower_pp",
            "shapley_availability_ci_upper_pp",
            BLUE,
            "o",
            0.0,
        ),
        (
            "Shapley capacity",
            "shapley_capacity_point_pp",
            "shapley_capacity_ci_lower_pp",
            "shapley_capacity_ci_upper_pp",
            ORANGE,
            "s",
            -0.22,
        ),
    ]
    for label, value_col, low_col, high_col, color, marker, offset in components:
        values = np.array([lookup.loc[key, value_col] for key in row_order])
        lows = np.array([lookup.loc[key, low_col] for key in row_order])
        highs = np.array([lookup.loc[key, high_col] for key in row_order])
        ax_effect.errorbar(
            values,
            y + offset,
            xerr=np.vstack([values - lows, highs - values]),
            fmt=marker,
            color=color,
            markerfacecolor=color,
            markersize=5.5,
            capsize=2.5,
            linewidth=1.4,
            label=label,
            zorder=3,
        )
    labels = [f"{TARGET_SHORT[target]}  q={int(100 * fraction)}%" for target, fraction in row_order]
    ax_effect.set_yticks(y, labels)
    ax_effect.set_xlim(0, 7.35)
    ax_effect.set_xlabel("Change in population event capture (percentage points)")
    ax_effect.set_title("(a) Symmetric allocation of the joint contrast", loc="left", weight="bold")
    ax_effect.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax_effect.spines["top"].set_visible(False)
    ax_effect.spines["right"].set_visible(False)
    ax_effect.legend(frameon=False, loc="lower right")

    x = np.arange(2)
    offsets = [-0.18, 0.0, 0.18]
    markers = ["o", "s", "^"]
    for target, offset, marker in zip(TARGET_ORDER, offsets, markers):
        rows = share_ci[share_ci["target"] == target].set_index("nominal_fraction").loc[[0.05, 0.10]]
        values = rows["capacity_share_pct"].to_numpy()
        lows = rows["ci_low"].to_numpy()
        highs = rows["ci_high"].to_numpy()
        ax_share.errorbar(
            x + offset,
            values,
            yerr=np.vstack([values - lows, highs - values]),
            fmt=marker,
            color=TARGET_COLOR[target],
            markerfacecolor=TARGET_COLOR[target],
            markersize=6,
            capsize=3,
            linewidth=1.6,
            label=TARGET_SHORT[target],
            zorder=3,
        )
        for x_value, value in zip(x + offset, values):
            ax_share.text(
                x_value,
                value + 0.62,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.4,
                color=TARGET_COLOR[target],
                weight="bold",
            )
    ax_share.axhline(50, color="#8C939C", linestyle="--", linewidth=1.0)
    ax_share.set_xticks(x, ["q=5%", "q=10%"])
    ax_share.set_ylim(49, 69)
    ax_share.set_ylabel("Symmetric capacity share of joint contrast (%)")
    ax_share.set_title("(b) Capacity receives 62--66% of the allocation", loc="left", weight="bold")
    ax_share.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax_share.spines["top"].set_visible(False)
    ax_share.spines["right"].set_visible(False)
    ax_share.legend(frameon=False, ncol=3, loc="lower center")

    fig.subplots_adjust(left=0.105, right=0.995, top=0.88, bottom=0.17, wspace=0.34)
    return save(fig, "figure2_shapley_capacity"), share_ci


def figure3(synthetic: pd.DataFrame) -> list[Path]:
    validate_synthetic(synthetic)
    relation_color = {
        "higher_risk_less_available": BLUE,
        "risk_independent": GRAY,
        "higher_risk_more_available": ORANGE,
    }
    relation_label = {
        "higher_risk_less_available": "Less available",
        "risk_independent": "Risk-independent",
        "higher_risk_more_available": "More available",
    }
    quality_marker = {"good": "o", "medium": "s", "weak": "^"}

    x_all = 100 * synthetic["mean_availability_effect_at_all_budget"]
    y_all = 100 * synthetic["mean_naive_top_fraction_effect"]
    zoom_mask = x_all.between(*ZOOM_X) & y_all.between(*ZOOM_Y)
    reversal_count = int(synthetic["naive_fixed_budget_sign_reversal"].sum())

    fig = plt.figure(figsize=(8.6, 5.9))
    scatter_grid = fig.add_gridspec(
        1, 2, left=0.085, right=0.98, bottom=0.54, top=0.92, wspace=0.34
    )
    ax_scatter = fig.add_subplot(scatter_grid[0, 0])
    ax_zoom = fig.add_subplot(scatter_grid[0, 1])
    ax_heat = fig.add_axes([0.085, 0.105, 0.795, 0.135])
    ax_colorbar = fig.add_axes([0.905, 0.105, 0.016, 0.135])

    def scatter_panel(ax: plt.Axes, rows: pd.DataFrame, xlim: tuple, ylim: tuple) -> None:
        ax.add_patch(
            Rectangle(
                (xlim[0], 0), -xlim[0], ylim[1],
                facecolor="#FBE8DE", edgecolor="none", alpha=0.65, zorder=0,
            )
        )
        for relation, color in relation_color.items():
            for quality, marker in quality_marker.items():
                subset = rows[
                    (rows["availability_relation"] == relation)
                    & (rows["model_quality"] == quality)
                ]
                ax.scatter(
                    100 * subset["mean_availability_effect_at_all_budget"],
                    100 * subset["mean_naive_top_fraction_effect"],
                    color=color, marker=marker, s=48, alpha=0.9,
                    edgecolors="white", linewidths=0.5, zorder=3,
                )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.axvline(0, color="#555B62", linewidth=0.9)
        ax.axhline(0, color="#555B62", linewidth=0.9)
        ax.set_axisbelow(True)
        ax.grid(color=GRID, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel(r"Eligibility contrast at fixed $b_a$ (pp)")
        ax.set_ylabel("Joint top-q contrast (pp)")

    scatter_panel(
        ax_scatter, synthetic,
        (min(ZOOM_X[0], x_all.min() - 0.5), x_all.max() + 1.0),
        (min(-0.6, y_all.min() - 0.5), y_all.max() + 1.5),
    )
    # The detail is a geometric subset, not a filter for sign reversals or color.
    scatter_panel(ax_zoom, synthetic.loc[zoom_mask], ZOOM_X, ZOOM_Y)
    ax_scatter.set_title(
        f"(a) Full grid ({len(synthetic)} conditions)", loc="left", weight="bold"
    )
    ax_zoom.set_title(
        f"(b) Near-zero detail ({int(zoom_mask.sum())}/{len(synthetic)})",
        loc="left", weight="bold",
    )
    ax_scatter.set_xticks([-3, 0, 5, 10, 15, 20])
    ax_scatter.set_yticks(np.arange(0, y_all.max() + 1, 5))
    ax_zoom.set_xticks([-3, -2, -1, 0, 0.5])
    ax_zoom.set_yticks(np.arange(ZOOM_Y[0], ZOOM_Y[1] + 1, 3))
    ax_scatter.text(
        0.025, 0.96, f"{reversal_count}/{len(synthetic)} mean sign reversals",
        transform=ax_scatter.transAxes, color=ORANGE,
        fontsize=10.7, weight="bold", va="top",
    )

    relation_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="white",
            label=relation_label[relation],
            markersize=7,
        )
        for relation, color in relation_color.items()
    ]
    quality_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            markerfacecolor="#555B62",
            markeredgecolor="white",
            label=quality.title(),
            markersize=7,
        )
        for quality, marker in quality_marker.items()
    ]
    for ypos, heading, handles in [
        (0.428, "Higher risk:", relation_handles),
        (0.384, "Score quality:", quality_handles),
    ]:
        fig.text(0.085, ypos, heading, fontsize=10.7, weight="bold", va="center")
        fig.legend(
            handles=handles, frameon=False, loc="center left",
            bbox_to_anchor=(0.26, ypos), ncol=3, borderaxespad=0,
            columnspacing=1.6, handletextpad=0.4, handlelength=1.0,
        )
    fig.text(
        0.98, 0.384, "Equal size across q", ha="right", va="center",
        fontsize=10.7, color=INK,
    )
    fig.text(
        0.085, 0.339,
        "Shading: positive joint / negative fixed-budget contrast.",
        fontsize=10.7, color=INK, va="center",
    )

    high_access = synthetic[
        synthetic["availability_relation"] == "higher_risk_more_available"
    ].copy()
    qualities = SYNTHETIC_LEVELS["model_quality"]
    availability = SYNTHETIC_LEVELS["availability_target_fraction"]
    fractions = SYNTHETIC_LEVELS["nominal_fraction"]
    columns = pd.MultiIndex.from_product(
        [qualities, availability], names=["model_quality", "availability_target_fraction"]
    )
    heat_data = high_access.pivot(
        index="nominal_fraction",
        columns=["model_quality", "availability_target_fraction"],
        values="sign_reversal_replicate_fraction",
    ).reindex(index=fractions, columns=columns)
    matrix = 100 * heat_data.to_numpy()
    if matrix.shape != (3, 9) or not np.isfinite(matrix).all():
        raise ValueError("Synthetic heatmap must contain all 27 high-risk-available conditions")

    cmap = LinearSegmentedColormap.from_list(
        "reversal",
        ["#F6F7F8", "#F8D6C4", "#E88D5A", ORANGE],
    )
    image = ax_heat.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax_heat.set_xticks(range(len(columns)), [f"{100 * a:.0f}" for _, a in columns])
    ax_heat.set_yticks(range(len(fractions)), [f"{100 * q:.0f}%" for q in fractions])
    ax_heat.set_ylabel("Top fraction q")
    ax_heat.set_xlabel("Availability within score-quality group (%)", labelpad=5)
    ax_heat.tick_params(length=0, pad=4)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax_heat.text(
                column,
                row,
                f"{value:.0f}%",
                ha="center",
                va="center",
                fontsize=10.7,
                weight="bold",
                color="white" if value >= 65 else INK,
            )
    for separator in [2.5, 5.5]:
        ax_heat.axvline(separator, color="white", linewidth=3)
    for separator in [0.5, 1.5]:
        ax_heat.axhline(separator, color="white", linewidth=0.7)
    for index, quality in enumerate(qualities):
        ax_heat.text(
            (index + 0.5) / len(qualities), 1.12, quality.title(),
            transform=ax_heat.transAxes, ha="center", va="bottom",
            fontsize=10.7, weight="bold",
        )
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    fig.text(
        0.085, 0.285, "(c) Reversal frequency: higher risk more available",
        fontsize=12, weight="bold", va="bottom",
    )
    cbar = fig.colorbar(image, cax=ax_colorbar, ticks=[0, 50, 100])
    cbar.set_label("Frequency (%)", fontsize=10.7)
    cbar.ax.tick_params(labelsize=10.7, length=2, pad=2)
    cbar.outline.set_visible(False)
    return save(fig, "figure3_synthetic_stress_test")


def write_manifest(inputs: list[Path], outputs: list[Path], frames: dict[str, pd.DataFrame]) -> None:
    shapley = frames["shapley"]
    synthetic = frames["synthetic"]
    summary = {
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in outputs},
        "invariants": {
            "shapley_rows": int(len(shapley)),
            "bootstrap_rows": int(len(frames["bootstrap"])),
            "synthetic_conditions": int(len(synthetic)),
            "synthetic_unique_keys": int(
                len(synthetic.drop_duplicates(list(SYNTHETIC_LEVELS)))
            ),
            "mean_sign_reversals": int(synthetic["naive_fixed_budget_sign_reversal"].sum()),
            "robust_negative_fixed_budget_conditions": int(
                synthetic["fixed_budget_interval_below_zero"].sum()
            ),
            "max_shapley_identity_error": float(
                np.max(
                    np.abs(
                        shapley["total_topq_contrast_point"]
                        - shapley["shapley_availability_point"]
                        - shapley["shapley_capacity_point"]
                    )
                )
            ),
        },
        "figure3": {
            "zoom_x_pp": list(ZOOM_X),
            "zoom_y_pp": list(ZOOM_Y),
            "zoom_conditions": int((
                (100 * synthetic["mean_availability_effect_at_all_budget"]).between(*ZOOM_X)
                & (100 * synthetic["mean_naive_top_fraction_effect"]).between(*ZOOM_Y)
            ).sum()),
            "heatmap_shape": [3, 9],
            "scatter_color": "availability_relation",
            "scatter_shape": "model_quality",
            "scatter_size": "constant; nominal fraction not encoded",
        },
    }
    (DATA / "figure_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    style()
    frames = load_inputs()
    outputs: list[Path] = []
    figure1_outputs, _ = figure1(frames["shapley"])
    outputs.extend(figure1_outputs)
    outputs.append(DATA / "figure1_headline_data.csv")
    outputs.extend(figure3(frames["synthetic"]))

    inputs = [
        DATA / "cohort_counts_validation.csv",
        DATA / "shapley_main_results.csv",
        DATA / "shapley_bootstrap_replicates.csv",
        DATA / "synthetic_figure3_data.csv",
    ]
    write_manifest(inputs, outputs, frames)
    print("PASS: generated 2 data-driven figures from canonical CSV inputs")


if __name__ == "__main__":
    main()
