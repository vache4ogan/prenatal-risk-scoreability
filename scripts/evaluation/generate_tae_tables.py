import pandas as pd
from pathlib import Path

def format_ci(point, lower, upper):
    return f"{point:+.2f} [{lower:+.2f}, {upper:+.2f}]"

def target_name(t):
    mapping = {
        'target_preterm': 'Preterm',
        'target_nicu': 'NICU',
        'target_lbw': 'LBW'
    }
    return mapping.get(t, t)

def scenario_name(s):
    mapping = {
        'early_entry': 'Early Entry',
        'any_prenatal_care': 'Any Care',
        'all_record_upper_bound': 'All-Record'
    }
    return mapping.get(s, s)

def protocol_name(p):
    mapping = {
        'cohort_specific_top_fraction': 'Cohort-Specific Top-q',
        'fixed_absolute_budget': 'Fixed Absolute Budget',
        'fixed_calibration_threshold': 'Fixed Calibration Threshold'
    }
    return mapping.get(p, p)

def generate_main_table():
    # 1. Read Shapley results (q=10%)
    shapley_paths = list(Path('results/fixed_B_shapley').glob('shapley_main_results*.csv'))
    if not shapley_paths:
        raise FileNotFoundError("Shapley main results CSV not found.")
    df_shapley = pd.read_csv(shapley_paths[0])
    df_shapley_10 = df_shapley[df_shapley['nominal_fraction'] == 0.1]

    # 2. Read 2024 Replication results
    rep_path = Path('results/tae_2026_replication/replication_2024_results.csv')
    df_rep = pd.read_csv(rep_path)

    # Begin LaTeX table
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Main Results: Shapley Decomposition of Naive Bias (q=10\%) and 2024 Locked Temporal Replication}",
        r"\label{tab:main_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l r r r r r}",
        r"\toprule",
        r"\textbf{Target} & \textbf{Naive $\Delta$ (pp)} & \textbf{Shapley $\phi_A$ (pp)} & \textbf{Shapley $\phi_K$ (pp)} & \textbf{Interaction $I$ (pp)} & \textbf{2024 Replication} \\",
        r"\midrule"
    ]

    for target in ['target_preterm', 'target_nicu', 'target_lbw']:
        # Shapley data
        row_s = df_shapley_10[df_shapley_10['target'] == target].iloc[0]
        naive = row_s['total_topq_contrast_point_pp']
        phi_a = format_ci(row_s['shapley_availability_point_pp'], row_s['shapley_availability_ci_lower_pp'], row_s['shapley_availability_ci_upper_pp'])
        phi_k = format_ci(row_s['shapley_capacity_point_pp'], row_s['shapley_capacity_ci_lower_pp'], row_s['shapley_capacity_ci_upper_pp'])
        inter = format_ci(row_s['interaction_point_pp'], row_s['interaction_ci_lower_pp'], row_s['interaction_ci_upper_pp'])
        
        # Replication data
        row_r = df_rep[df_rep['target'] == target].iloc[0]
        capture = row_r['Population Event Capture'] * 100
        sel_pct = (row_r['selected_n'] / row_r['N_all']) * 100
        rep_str = f"\\textbf{{{capture:.2f}\\%}} ({sel_pct:.1f}\\% sel)"

        lines.append(f"{target_name(target)} & {naive:+.2f} & {phi_a} & {phi_k} & {inter} & {rep_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}"
    ])

    out_dir = Path("paper/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_main_results.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated {out_path}:\n")
    print("\n".join(lines))
    print("\n")

def generate_appendix_table():
    canonical_path = Path('results/tae_2026_canonical/canonical_protocol_results.csv')
    df = pd.read_csv(canonical_path)
    
    # Filter for nominal_fraction == 0.1
    df = df[df['nominal_fraction'] == 0.1].copy()

    # Sort logic to maintain 27 rows (3 targets * 3 protocols * 3 scenarios)
    targets = ['target_preterm', 'target_nicu', 'target_lbw']
    protocols = ['cohort_specific_top_fraction', 'fixed_absolute_budget', 'fixed_calibration_threshold']
    scenarios = ['early_entry', 'any_prenatal_care', 'all_record_upper_bound']
    
    df['target_cat'] = pd.Categorical(df['target'], categories=targets, ordered=True)
    df['protocol_cat'] = pd.Categorical(df['protocol'], categories=protocols, ordered=True)
    df['scenario_cat'] = pd.Categorical(df['availability_scenario'], categories=scenarios, ordered=True)
    
    df = df.sort_values(['target_cat', 'protocol_cat', 'scenario_cat'])

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Full Evaluation Protocol Results (q=10\%) across Target, Protocol, and Availability Scenario}",
        r"\label{tab:full_protocols}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lll rrr rrr}",
        r"\toprule",
        r"\textbf{Target} & \textbf{Protocol} & \textbf{Scenario} & \textbf{Eligible $N$} & \textbf{Budget $N$} & \textbf{Selected $N$} & \textbf{Precision} & \textbf{Capture} \\",
        r"\midrule"
    ]

    current_target = None
    current_protocol = None

    for _, row in df.iterrows():
        t = target_name(row['target'])
        p = protocol_name(row['protocol'])
        s = scenario_name(row['availability_scenario'])
        
        elig_n = f"{int(row['eligible_n']):,}"
        budget_n = f"{int(row['budget_n']):,}" if pd.notna(row['budget_n']) else "---"
        sel_n = f"{int(row['selected_n']):,}"
        prec = f"{row['precision']*100:.2f}\\%"
        cap = f"{row['population_event_capture']*100:.2f}\\%"
        
        # Omit repeating names for clean formatting
        t_str = t if t != current_target else ""
        p_str = p if p != current_protocol else ""
        
        if t != current_target and current_target is not None:
            lines.append(r"\midrule")
            
        lines.append(f"{t_str} & {p_str} & {s} & {elig_n} & {budget_n} & {sel_n} & {prec} & \\textbf{{{cap}}} \\\\")
        
        current_target = t
        current_protocol = p

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}"
    ])

    out_dir = Path("paper/appendix")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_full_protocols_appendix.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated {out_path}")

if __name__ == "__main__":
    generate_main_table()
    generate_appendix_table()
