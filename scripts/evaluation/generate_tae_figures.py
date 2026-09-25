import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def setup_style():
    # Clean publication-ready styling
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True})
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11

def generate_figure_1(cohort_csv):
    df = pd.read_csv(cohort_csv)
    # Extract CDC 2023 preterm target for representative numbers
    rep_df = df[(df['year'] == 2023) & (df['target'] == 'target_preterm')]
    
    counts = dict(zip(rep_df['scenario'], rep_df['n_records']))
    
    early = counts['early_entry']
    later = counts['later_entry']
    no_care = counts['no_care']
    unknown = counts['unknown_care']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Scenarios to plot
    scenarios = ['Early Entry\nCohort', 'Any-Care\nCohort', 'All-Record\nUpper Bound']
    
    # Compositions
    # Early Entry: early
    # Any Care: early + later
    # All Record: early + later + no_care + unknown
    
    bar_width = 0.5
    y_pos = np.arange(len(scenarios))
    
    colors = sns.color_palette("colorblind")
    
    # Base early
    ax.barh(y_pos, [early, early, early], bar_width, label='Early Entry (months 1-3)', color=colors[0])
    
    # Later
    ax.barh(y_pos[1:], [later, later], bar_width, left=[early, early], label='Later Entry (months 4-10)', color=colors[1])
    
    # No care & unknown (only on all-record)
    ax.barh(y_pos[2], [no_care], bar_width, left=[early + later], label='No Prenatal Care', color=colors[2])
    ax.barh(y_pos[2], [unknown], bar_width, left=[early + later + no_care], label='Unknown Care', color=colors[3])
    
    # Budgets (10% of each cohort)
    budget_early = 0.1 * early
    budget_any = 0.1 * (early + later)
    budget_all = 0.1 * (early + later + no_care + unknown)
    
    # Add vertical markers for 10% budget
    ax.scatter([budget_early], [y_pos[0]], color='red', marker='|', s=200, zorder=3, label='10% Cohort Budget')
    ax.scatter([budget_any], [y_pos[1]], color='red', marker='|', s=200, zorder=3)
    ax.scatter([budget_all], [y_pos[2]], color='red', marker='|', s=200, zorder=3)
    
    # Connect budgets with a dashed line to show capacity shift
    ax.plot([budget_early, budget_any, budget_all], y_pos, color='red', linestyle='--', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scenarios)
    ax.invert_yaxis()  # top-to-bottom
    ax.set_xlabel('Number of Patients (N)')
    ax.set_title('Figure 1: Selective Availability Scenarios and Composition')
    
    # Use scientific/comma notation for x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "figure_1_methodology.pdf"
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Figure 1 to {out_file}")

def generate_figure_2(shapley_csv):
    df = pd.read_csv(shapley_csv)
    # Filter for q=10% (0.1)
    df_10 = df[df['nominal_fraction'] == 0.1].copy()
    
    targets = ['target_preterm', 'target_nicu', 'target_lbw']
    labels = ['Preterm', 'NICU', 'LBW']
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    x = np.arange(len(targets))
    width = 0.35
    
    availability = []
    availability_err_lower = []
    availability_err_upper = []
    
    capacity = []
    capacity_err_lower = []
    capacity_err_upper = []
    
    naive = []
    
    for t in targets:
        row = df_10[df_10['target'] == t].iloc[0]
        
        # Availability
        a_pt = row['shapley_availability_point_pp']
        availability.append(a_pt)
        availability_err_lower.append(a_pt - row['shapley_availability_ci_lower_pp'])
        availability_err_upper.append(row['shapley_availability_ci_upper_pp'] - a_pt)
        
        # Capacity
        c_pt = row['shapley_capacity_point_pp']
        capacity.append(c_pt)
        capacity_err_lower.append(c_pt - row['shapley_capacity_ci_lower_pp'])
        capacity_err_upper.append(row['shapley_capacity_ci_upper_pp'] - c_pt)
        
        # Naive Contrast
        n_pt = row['total_topq_contrast_point_pp']
        naive.append(n_pt)
        
    colors = sns.color_palette("colorblind")
    
    # Plot Availability (bottom part of the stack)
    ax.bar(x, availability, width, label='Availability Effect (Shapley)', color=colors[0])
    
    # Plot Capacity (top part of the stack)
    ax.bar(x, capacity, width, bottom=availability, label='Capacity Effect (Shapley)', color=colors[1])
    
    # Add error bars for the total naive sum, or components
    # The sum of availability and capacity equals the naive contrast
    # We will plot the error bars for the components individually to be statistically rigorous
    
    # Error bars for Availability (centered at availability height)
    ax.errorbar(x, availability, 
                yerr=[availability_err_lower, availability_err_upper], 
                fmt='none', ecolor='black', capsize=5, capthick=1.5, zorder=3)
    
    # Error bars for Capacity (centered at top of stack: availability + capacity)
    ax.errorbar(x, [a + c for a, c in zip(availability, capacity)], 
                yerr=[capacity_err_lower, capacity_err_upper], 
                fmt='none', ecolor='black', capsize=5, capthick=1.5, zorder=3)
    
    # Mark the naive total clearly as a point or text
    for i, pt in enumerate(naive):
        ax.text(i, pt + max(capacity_err_upper[i], 0.2), f'Δ = {pt:.2f}pp', 
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Capture Rate Contrast (Percentage Points)')
    ax.set_title('Figure 2: Shapley Decomposition of Top-10% Cohort Bias')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "figure_2_shapley_results.pdf"
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Figure 2 to {out_file}")

if __name__ == "__main__":
    setup_style()
    
    # Ensure correct file paths
    # The user mentioned 'results/fixed_B_shapley/shapley_main_results.csv', but `ls` showed it has `(1)` suffix
    # Let's dynamically find it in the dir
    cohort_csv = list(Path('results/tae_2026_canonical').glob('cohort_counts_validation*.csv'))[0]
    shapley_csv = list(Path('results/fixed_B_shapley').glob('shapley_main_results*.csv'))[0]
    
    generate_figure_1(cohort_csv)
    generate_figure_2(shapley_csv)
