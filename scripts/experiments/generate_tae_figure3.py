import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()

def pct(x):
    return 100.0 * x

def validate_data(input_dir):
    decomp_path = input_dir / "synthetic_decomposition.csv"
    comp_path = input_dir / "synthetic_comparison_summary.csv"
    fig3_path = input_dir / "synthetic_figure3_data.csv"
    
    decomp = pd.read_csv(decomp_path)
    comp = pd.read_csv(comp_path)
    fig3 = pd.read_csv(fig3_path)
    
    # 1. 81 conditions
    keys = ["availability_target_fraction", "availability_relation", "model_quality", "nominal_fraction"]
    decomp_conds = decomp[keys].drop_duplicates()
    if len(decomp_conds) != 81:
        raise ValueError(f"Expected 81 conditions in decomposition, got {len(decomp_conds)}")
        
    comp_conds = comp[keys].drop_duplicates()
    if len(comp_conds) != 81:
        raise ValueError(f"Expected 81 conditions in comparison, got {len(comp_conds)}")
        
    fig3_conds = fig3[keys].drop_duplicates()
    if len(fig3_conds) != 81:
        raise ValueError(f"Expected 81 conditions in figure3_data, got {len(fig3_conds)}")
        
    # 2. Key matching
    decomp_keys_set = set(tuple(x) for x in decomp_conds.values)
    comp_keys_set = set(tuple(x) for x in comp_conds.values)
    if decomp_keys_set != comp_keys_set:
        raise ValueError("Keys mismatch between decomposition and comparison")
        
    # 3. No duplicates in fig3
    if len(fig3) != 81:
        raise ValueError(f"Duplicates found in fig3_data! Length is {len(fig3)}")
        
    # 4. Check formulas on decomposition (C11 - C00 = naive, C11 - C01 = fixed_budget)
    # naive_top_fraction_effect
    computed_naive = decomp["c11_capture"] - decomp["c00_capture"]
    if not np.allclose(computed_naive, decomp["naive_top_fraction_effect"]):
        raise ValueError("Formula validation failed: C11 - C00 != naive_top_fraction_effect")
        
    computed_fixed_budget = decomp["c11_capture"] - decomp["c01_capture"]
    if not np.allclose(computed_fixed_budget, decomp["availability_effect_at_all_budget"]):
        raise ValueError("Formula validation failed: C11 - C01 != availability_effect_at_all_budget")
        
    # Shapley validations
    computed_shapley_avail = 0.5 * ((decomp["c10_capture"] - decomp["c00_capture"]) + (decomp["c11_capture"] - decomp["c01_capture"]))
    if not np.allclose(computed_shapley_avail, decomp["shapley_availability_effect"]):
        raise ValueError("Formula validation failed: Shapley availability")

    computed_shapley_cap = 0.5 * ((decomp["c01_capture"] - decomp["c00_capture"]) + (decomp["c11_capture"] - decomp["c10_capture"]))
    if not np.allclose(computed_shapley_cap, decomp["shapley_capacity_effect"]):
        raise ValueError("Formula validation failed: Shapley capacity")

    print("Validation passed: 81 conditions, keys match, formulas are correct, no duplicates.")
    return fig3

def plot_figure(df, output_dir, smoke_test=False):
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "figure.titlesize": 13,
    })
    
    if smoke_test:
        df = df.head(9).copy()
        
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    
    # --- PANEL A: Naive vs Fixed Budget ---
    ax = axes[0]
    
    x = pct(df["mean_naive_top_fraction_effect"])
    y = pct(df["mean_availability_effect_at_all_budget"])
    
    # 95% CIs
    x_err_low = x - pct(df["q025_naive_top_fraction_effect"])
    x_err_high = pct(df["q975_naive_top_fraction_effect"]) - x
    y_err_low = y - pct(df["q025_availability_effect_at_all_budget"])
    y_err_high = pct(df["q975_availability_effect_at_all_budget"]) - y
    
    # Sign reversals: naive > 0, fixed-budget < 0
    # Actually wait, let's use the explicit column if it exists or compute it.
    is_reversal = (x > 0) & (y < 0)
    
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    
    lo = min(x.min(), y.min(), -1)
    hi = max(x.max(), y.max(), 1)
    ax.plot([lo, hi], [lo, hi], color="gray", linewidth=0.8, linestyle=":", zorder=1)
    
    ax.errorbar(
        x[~is_reversal], y[~is_reversal],
        xerr=np.vstack([x_err_low[~is_reversal], x_err_high[~is_reversal]]),
        yerr=np.vstack([y_err_low[~is_reversal], y_err_high[~is_reversal]]),
        fmt="o", color="#1f77b4", alpha=0.6, markersize=5, label="Consistent Sign"
    )
    
    ax.errorbar(
        x[is_reversal], y[is_reversal],
        xerr=np.vstack([x_err_low[is_reversal], x_err_high[is_reversal]]),
        yerr=np.vstack([y_err_low[is_reversal], y_err_high[is_reversal]]),
        fmt="^", color="#d62728", alpha=0.9, markersize=7, label="Sign Reversal"
    )
    
    ax.set_xlabel("Naive Effect (pp)")
    ax.set_ylabel("Fixed-Budget Effect (pp)")
    ax.set_title(f"A. Effect Comparison ({is_reversal.sum()} Reversals)")
    ax.legend(frameon=False)
    
    # --- PANEL B: Shapley Mechanism ---
    ax = axes[1]
    sa = pct(df["mean_shapley_availability_effect"])
    sc = pct(df["mean_shapley_capacity_effect"])
    interaction = pct(df["mean_interaction_effect"])
    
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    
    # Point sizes based on interaction magnitude
    max_interaction_abs = interaction.abs().max()
    sizes = 20 + 150 * (interaction.abs() / max_interaction_abs)
    
    scatter = ax.scatter(sa, sc, s=sizes, c=interaction, cmap="coolwarm", alpha=0.8, edgecolors='k', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Interaction Effect (pp)")
    
    ax.set_xlabel("Shapley Availability Effect (pp)")
    ax.set_ylabel("Shapley Capacity Effect (pp)")
    ax.set_title(f"B. Shapley Components (Max |Interaction|: {max_interaction_abs:.2f} pp)")
    
    # Mark max absolute interaction
    max_idx = interaction.abs().idxmax()
    ax.annotate("Max Interaction", 
                xy=(sa[max_idx], sc[max_idx]),
                xytext=(15, 15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
                
    # --- PANEL C: Grid ---
    ax = axes[2]
    # Prepare the grid data
    df_grid = df.copy()
    
    EXPECTED_AVAILABILITY = [0.50, 0.70, 0.90]
    EXPECTED_RELATIONS = [
        "higher_risk_less_available",
        "risk_independent",
        "higher_risk_more_available",
    ]
    EXPECTED_QUALITY = ["good", "medium", "weak"]
    EXPECTED_Q = [0.05, 0.10, 0.20]
    
    row_order = [f"{int(a*100)}%\n{rel.replace('_', ' ')}" for a in EXPECTED_AVAILABILITY for rel in EXPECTED_RELATIONS]
    col_order = [f"{q.capitalize()}\nq={int(frac*100)}%" for q in EXPECTED_QUALITY for frac in EXPECTED_Q]
    
    df_grid["row_label"] = (df_grid["availability_target_fraction"] * 100).round().astype(int).astype(str) + "%\n" + df_grid["availability_relation"].str.replace("_", " ")
    df_grid["col_label"] = df_grid["model_quality"].str.capitalize() + "\nq=" + (df_grid["nominal_fraction"] * 100).round().astype(int).astype(str) + "%"
    
    pivot = df_grid.pivot(index="row_label", columns="col_label", values="mean_availability_effect_at_all_budget")
    pivot = pivot.reindex(index=row_order, columns=col_order)
    arr = pivot.to_numpy(dtype=float) * 100.0
    
    lim = max(abs(np.nanmin(arr)), abs(np.nanmax(arr)))
    im = ax.imshow(arr, aspect="auto", cmap="RdBu", vmin=-lim, vmax=lim)
    
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=8)
    ax.set_xlabel("Model Quality / Budget")
    ax.set_ylabel("Availability / Target")
    ax.set_title("C. Fixed-Budget Effect Grid")
    
    cbar2 = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label("Fixed-Budget Effect (pp)")
    
    # Mark sign reversals and ci_below_zero
    for i, r in enumerate(row_order):
        for j, c in enumerate(col_order):
            mask = (df_grid["row_label"] == r) & (df_grid["col_label"] == c)
            if not mask.any():
                continue
            row_data = df_grid.loc[mask].iloc[0]
            
            val_x = row_data["mean_naive_top_fraction_effect"]
            val_y = row_data["mean_availability_effect_at_all_budget"]
            rev = (val_x > 0) and (val_y < 0)
            ci_below = row_data["fixed_budget_interval_below_zero"]
            
            if rev:
                ax.scatter(j, i, marker="o", facecolors='none', edgecolors='k', s=80, linewidths=1.5, zorder=5)
            if ci_below:
                ax.scatter(j, i, marker="x", color="k", s=40, zorder=6)
                
    # Legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Sign Reversal', markerfacecolor='none', markeredgecolor='k', markersize=8),
        Line2D([0], [0], marker='x', color='w', label='CI Below 0', markeredgecolor='k', markersize=8)
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.8)

    # Save
    fig.savefig(output_dir / "figure3.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "figure3.png", bbox_inches="tight", dpi=300)
    fig.savefig(output_dir / "figure3.svg", bbox_inches="tight")
    plt.close(fig)
    
    # Validation report
    n_reversals = int(is_reversal.sum())
    n_ci_below = int(df["fixed_budget_interval_below_zero"].sum())
    
    print("--- Validation Report ---")
    print(f"Total conditions: {len(df)}")
    print(f"Sign reversals (Naive > 0, Fixed < 0): {n_reversals}")
    print(f"Conditions with 95% CI completely below zero: {n_ci_below}")
    print(f"Max absolute interaction (mean): {max_interaction_abs:.3f} pp")
    if "max_abs_interaction_replicate" in df.columns:
        max_rep_int = pct(df["max_abs_interaction_replicate"].max())
        print(f"Max absolute interaction (replicate-level): {max_rep_int:.3f} pp")

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    df = validate_data(args.input_dir)
    plot_figure(df, args.output_dir, smoke_test=args.smoke_test)

if __name__ == "__main__":
    main()
