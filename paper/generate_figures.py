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
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9.2,
            "axes.edgecolor": "#8C939C",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [OUT / f"{stem}.pdf", OUT / f"{stem}.png"]
    fig.savefig(paths[0], bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


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
    if len(synthetic) != 81:
        raise ValueError(f"Expected 81 synthetic conditions, found {len(synthetic)}")
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
        figsize=(8.6, 3.25),
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
            fontsize=7.8,
            weight="bold",
        )
        ax_workload.text(
            pool - 0.08,
            ypos - 0.13,
            f"{pool:.2f}M candidates",
            ha="right",
            va="center",
            color=INK,
            fontsize=7.8,
        )
    increase = 100 * (mean_all_budget / mean_early_budget - 1)
    ax_workload.text(
        1.72,
        0.50,
        f"+{increase:.1f}% selected workload",
        ha="center",
        va="center",
        color=ORANGE,
        fontsize=9.0,
        weight="bold",
    )
    ax_workload.set_yticks(y, ["Early entry", "All-record\nupper bound"])
    ax_workload.set_xlim(0, 3.72)
    ax_workload.set_xlabel("Target-specific cohort size (millions)")
    ax_workload.set_title("(a) Same fraction, different workload", loc="left", weight="bold")
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
            fontsize=8.5,
            weight="bold",
        )
        ax_gain.text(
            eligibility[index] + capacity[index] / 2,
            ypos_value,
            f"{capacity[index]:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=8.5,
            weight="bold",
        )
        ax_gain.text(
            total[index] + 0.18,
            ypos_value,
            f"{total[index]:.2f} pp  |  {q10.loc[index, 'capacity_share_pct']:.1f}% capacity",
            ha="left",
            va="center",
            color=INK,
            fontsize=7.6,
        )
    ax_gain.set_yticks(ypos, [TARGET_SHORT[target] for target in TARGET_ORDER])
    ax_gain.set_xlim(0, 9.05)
    ax_gain.set_xlabel("Increase in population event capture (percentage points)")
    ax_gain.set_title("(b) What the apparent gain contains", loc="left", weight="bold")
    ax_gain.spines[["top", "right", "left"]].set_visible(False)
    ax_gain.tick_params(axis="y", length=0)
    ax_gain.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax_gain.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, -0.25),
        ncol=2,
    )

    fig.subplots_adjust(left=0.075, right=0.995, top=0.87, bottom=0.24, wspace=0.33)
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
    relation_color = {
        "higher_risk_less_available": BLUE,
        "risk_independent": GRAY,
        "higher_risk_more_available": ORANGE,
    }
    relation_label = {
        "higher_risk_less_available": "Higher risk less available",
        "risk_independent": "Risk-independent availability",
        "higher_risk_more_available": "Higher risk more available",
    }
    quality_marker = {"good": "o", "medium": "s", "weak": "^"}

    fig, (ax_scatter, ax_heat) = plt.subplots(
        1,
        2,
        figsize=(8.6, 3.75),
        gridspec_kw={"width_ratios": [1.08, 1.0]},
    )

    x_all = 100 * synthetic["mean_availability_effect_at_all_budget"]
    y_all = 100 * synthetic["mean_naive_top_fraction_effect"]
    x_min = min(-3.55, x_all.min() - 0.4)
    x_max = x_all.max() + 0.6
    y_max = y_all.max() + 0.8
    ax_scatter.add_patch(
        Rectangle(
            (x_min, 0),
            -x_min,
            y_max,
            facecolor="#FBE8DE",
            edgecolor="none",
            alpha=0.75,
            zorder=0,
        )
    )
    for relation, color in relation_color.items():
        for quality, marker in quality_marker.items():
            rows = synthetic[
                (synthetic["availability_relation"] == relation)
                & (synthetic["model_quality"] == quality)
            ]
            ax_scatter.scatter(
                100 * rows["mean_availability_effect_at_all_budget"],
                100 * rows["mean_naive_top_fraction_effect"],
                color=color,
                marker=marker,
                s=34 + 115 * rows["nominal_fraction"],
                alpha=0.82,
                edgecolors="white",
                linewidths=0.45,
                zorder=3,
            )
    ax_scatter.axvline(0, color="#555B62", linewidth=1.0)
    ax_scatter.axhline(0, color="#555B62", linewidth=1.0)
    diagonal = np.linspace(max(0, x_min), min(x_max, y_max), 100)
    ax_scatter.plot(diagonal, diagonal, color="#9BA1A8", linestyle="--", linewidth=1.0)
    ax_scatter.text(
        x_min + 0.22,
        y_max - 1.0,
        "18/81 mean\nsign reversals",
        color=ORANGE,
        weight="bold",
        fontsize=10,
        va="top",
    )
    ax_scatter.set_xlim(x_min, x_max)
    ax_scatter.set_ylim(-0.6, y_max)
    ax_scatter.set_xlabel(r"Eligibility effect at fixed budget $b_a$ (pp)")
    ax_scatter.set_ylabel("Cohort-relative effect (pp)")
    ax_scatter.set_title("(a) Joint and fixed-budget effects", loc="left", weight="bold")
    ax_scatter.grid(color=GRID, linewidth=0.7, zorder=0)
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)

    relation_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="white",
            label=relation_label[relation],
            markersize=7.5,
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
            markersize=6.5,
        )
        for quality, marker in quality_marker.items()
    ]
    ax_scatter.legend(
        handles=quality_handles,
        frameon=False,
        title="Score quality",
        loc="upper right",
        ncol=3,
        fontsize=7.2,
        title_fontsize=7.4,
        columnspacing=0.7,
        handletextpad=0.3,
    )

    high_access = synthetic[
        synthetic["availability_relation"] == "higher_risk_more_available"
    ].copy()
    qualities = ["good", "medium", "weak"]
    availability = [0.5, 0.7, 0.9]
    fractions = [0.05, 0.10, 0.20]
    matrix = np.zeros((9, 3))
    row_labels = []
    for row_index, (quality, fraction_available) in enumerate(
        [(quality, fraction_available) for quality in qualities for fraction_available in availability]
    ):
        row_labels.append(f"{quality.title()} {int(100 * fraction_available)}%")
        for column_index, q in enumerate(fractions):
            match = high_access[
                (high_access["model_quality"] == quality)
                & np.isclose(high_access["availability_target_fraction"], fraction_available)
                & np.isclose(high_access["nominal_fraction"], q)
            ]
            if len(match) != 1:
                raise ValueError("Synthetic heatmap key is not unique")
            matrix[row_index, column_index] = 100 * match.iloc[0]["sign_reversal_replicate_fraction"]

    cmap = LinearSegmentedColormap.from_list(
        "reversal",
        ["#F6F7F8", "#F8D6C4", "#E88D5A", ORANGE],
    )
    image = ax_heat.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax_heat.set_xticks(range(3), ["q=5%", "q=10%", "q=20%"])
    ax_heat.set_yticks(range(9), row_labels)
    ax_heat.tick_params(length=0, pad=5)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax_heat.text(
                column,
                row,
                f"{value:.0f}%",
                ha="center",
                va="center",
                fontsize=8.7,
                weight="bold",
                color="white" if value >= 65 else INK,
            )
    for separator in [2.5, 5.5]:
        ax_heat.axhline(separator, color="white", linewidth=3)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.set_title(
        "(b) Reversals when higher risk is more available",
        loc="left",
        weight="bold",
    )
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.045, pad=0.025)
    cbar.set_label("Reversal frequency (%)")
    cbar.outline.set_visible(False)

    fig.legend(
        handles=relation_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.012),
        ncol=3,
        fontsize=8.7,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.22, wspace=0.34)
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
