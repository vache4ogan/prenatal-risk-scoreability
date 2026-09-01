from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
OUT = ROOT / "tables"

TARGET_ORDER = ["target_preterm", "target_nicu", "target_lbw"]
TARGET_LABEL = {
    "target_preterm": "Preterm birth",
    "target_nicu": "NICU admission",
    "target_lbw": "Low birth weight",
}


def interval(point: float, low: float, high: float, digits: int = 2) -> str:
    return f"{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def load_shapley() -> tuple[pd.DataFrame, pd.DataFrame]:
    shapley = pd.read_csv(DATA / "shapley_main_results.csv")
    bootstrap = pd.read_csv(DATA / "shapley_bootstrap_replicates.csv")
    if len(shapley) != 6 or len(bootstrap) != 3000:
        raise ValueError("Unexpected Shapley artifact dimensions")
    if not np.allclose(
        shapley["total_topq_contrast_point"],
        shapley["shapley_availability_point"] + shapley["shapley_capacity_point"],
        atol=1e-12,
    ):
        raise ValueError("Shapley identity failed")
    bootstrap["capacity_share_pct"] = (
        100 * bootstrap["shapley_capacity"] / bootstrap["total_topq_contrast"]
    )
    shares = (
        bootstrap.groupby(["target", "nominal_fraction"])["capacity_share_pct"]
        .agg(
            share_low=lambda values: np.quantile(values, 0.025),
            share_high=lambda values: np.quantile(values, 0.975),
        )
        .reset_index()
    )
    point_share = shapley[
        ["target", "nominal_fraction", "shapley_capacity_point", "total_topq_contrast_point"]
    ].copy()
    point_share["capacity_share_pct"] = (
        100 * point_share["shapley_capacity_point"] / point_share["total_topq_contrast_point"]
    )
    shares = shares.merge(
        point_share[["target", "nominal_fraction", "capacity_share_pct"]],
        on=["target", "nominal_fraction"],
        how="left",
    )
    return shapley, shares


def table_main_shapley(shapley: pd.DataFrame, shares: pd.DataFrame) -> None:
    data = shapley[np.isclose(shapley["nominal_fraction"], 0.10)].set_index("target")
    share = shares[np.isclose(shares["nominal_fraction"], 0.10)].set_index("target")
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Same nominal top 10\%, different operational result. Population capture uses the full target-population event denominator. $\phi_E$ and $\phi_B$ are the symmetric eligibility and capacity allocations; conditional empirical-resampling ranges appear in Appendix~\ref{tab:full_shapley}.}",
        r"\label{tab:main_results}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}",
        r"\toprule",
        r"Outcome & Selected $n$ & Workload & Capture (\%) & Rel. gain & $\phi_E/\phi_B$ (pp) & $s_B$ \\",
        r"\midrule",
    ]
    for target in TARGET_ORDER:
        row = data.loc[target]
        share_row = share.loc[target]
        lines.append(
            " & ".join(
                [
                    TARGET_LABEL[target],
                    f"{int(row['B_early_point']):,}$\\rightarrow${int(row['B_all_point']):,}",
                    f"+{100 * (row['B_all_point'] / row['B_early_point'] - 1):.1f}\\%",
                    f"{100 * row['C00_early_Bearly']:.2f}$\\rightarrow${100 * row['C11_all_Ball']:.2f}\\%",
                    f"+{100 * (row['C11_all_Ball'] / row['C00_early_Bearly'] - 1):.1f}\\%",
                    f"{row['shapley_availability_point_pp']:.2f}/\\textbf{{{row['shapley_capacity_point_pp']:.2f}}}",
                    r"\textbf{" + f"{share_row['capacity_share_pct']:.1f}\\%" + "}",
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular*}",
            r"\end{table}",
        ]
    )
    write(OUT / "table1_shapley_q10.tex", lines)


def table_candidate_gate() -> None:
    data = pd.read_csv(DATA / "cohort_counts_validation.csv")
    data = data[data["year"] == 2023].copy()
    lookup = data.set_index(["target", "scenario"])
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{The candidate gate fixes reach before ranking. Early entry denotes prenatal-care initiation in months 1--3; documented start includes months 1--10. Risk ratios compare outcome prevalence in excluded groups with the early-entry group.}",
        r"\label{tab:candidate_gate}",
        r"\small",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}",
        r"\toprule",
        r"Outcome & Early eligible $n$ & Early share & Early ceiling & Documented ceiling & No-care / unknown risk \\",
        r"\midrule",
    ]
    for target in TARGET_ORDER:
        early = lookup.loc[(target, "early_entry")]
        documented = lookup.loc[(target, "any_prenatal_care")]
        no_care = lookup.loc[(target, "no_care")]
        unknown = lookup.loc[(target, "unknown_care")]
        all_record = lookup.loc[(target, "all_record_upper_bound")]
        early_prevalence = early["event_n"] / early["n_records"]
        no_care_ratio = (no_care["event_n"] / no_care["n_records"]) / early_prevalence
        unknown_ratio = (unknown["event_n"] / unknown["n_records"]) / early_prevalence
        lines.append(
            f"{TARGET_LABEL[target]} & {int(early['n_records']):,} & "
            f"{100 * early['n_records'] / all_record['n_records']:.1f}\\% & "
            f"{100 * early['event_n'] / all_record['event_n']:.1f}\\% & "
            f"{100 * documented['event_n'] / all_record['event_n']:.1f}\\% & "
            f"{no_care_ratio:.1f}$\\times$ / {unknown_ratio:.1f}$\\times$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
    write(OUT / "table_candidate_gate.tex", lines)


def table_policy_contrasts() -> None:
    data = pd.read_csv(DATA / "canonical_protocol_results.csv")
    data = data[
        np.isclose(data["nominal_fraction"], 0.10)
        & data["availability_scenario"].isin(
            ["early_entry", "all_record_upper_bound"]
        )
    ].copy()
    lookup = data.set_index(["target", "protocol", "availability_scenario"])
    policies = [
        ("cohort_specific_top_fraction", "Cohort-relative top 10\\%"),
        ("fixed_absolute_budget", "Fixed absolute $b_a$"),
        ("fixed_calibration_threshold", "Transported 2022 threshold"),
    ]
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{One score vector, three resource policies at the nominal 10\% operating point. Counts and population capture compare early-entry with all-record eligibility. The score threshold is chosen on CDC 2022 and transported unchanged to CDC 2023.}",
        r"\label{tab:policy_contrasts}",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrr@{}}",
        r"\toprule",
        r"Outcome & Resource policy & Selected $n$: early $\rightarrow$ all & Capture: early $\rightarrow$ all & $\Delta$ (pp) \\",
        r"\midrule",
    ]
    for target in TARGET_ORDER:
        for policy_index, (protocol, policy_label) in enumerate(policies):
            early = lookup.loc[(target, protocol, "early_entry")]
            all_record = lookup.loc[(target, protocol, "all_record_upper_bound")]
            outcome = TARGET_LABEL[target] if policy_index == 0 else ""
            early_capture = 100 * early["population_event_capture"]
            all_capture = 100 * all_record["population_event_capture"]
            lines.append(
                f"{outcome} & {policy_label} & "
                f"{int(early['selected_n']):,}$\\rightarrow${int(all_record['selected_n']):,} & "
                f"{early_capture:.2f}$\\rightarrow${all_capture:.2f}\\% & "
                f"{all_capture - early_capture:+.2f} \\\\"
            )
        if target != TARGET_ORDER[-1]:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
    write(OUT / "table_policy_contrasts_q10.tex", lines)


def table_score_transport() -> None:
    data = pd.read_csv(DATA / "model_diagnostics.csv").set_index("target")
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Frozen-score temporal check. Brier baseline is the constant CDC 2023 prevalence prediction; skill is $1-\mathrm{Brier}_{model}/\mathrm{Brier}_{baseline}$.}",
        r"\label{tab:score_transport}",
        r"\small",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}",
        r"\toprule",
        r"Outcome & ROC-AUC: 2022 $\rightarrow$ 2023 & AP: 2022 $\rightarrow$ 2023 & Brier: model / baseline & Brier skill \\",
        r"\midrule",
    ]
    for target in TARGET_ORDER:
        row = data.loc[target]
        prevalence = row["temporal_test_2023_prevalence"]
        baseline = prevalence * (1 - prevalence)
        skill = 1 - row["temporal_test_2023_brier"] / baseline
        lines.append(
            f"{TARGET_LABEL[target]} & "
            f"{row['calibration_2022_roc_auc']:.3f}$\\rightarrow${row['temporal_test_2023_roc_auc']:.3f} & "
            f"{row['calibration_2022_pr_auc']:.3f}$\\rightarrow${row['temporal_test_2023_pr_auc']:.3f} & "
            f"{row['temporal_test_2023_brier']:.4f} / {baseline:.4f} & "
            f"{100 * skill:.1f}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
    write(OUT / "table_score_transport.tex", lines)


def table_full_shapley(shapley: pd.DataFrame, shares: pd.DataFrame) -> None:
    data = shapley.merge(
        shares,
        on=["target", "nominal_fraction"],
        how="left",
        validate="one_to_one",
    )
    order = {target: index for index, target in enumerate(TARGET_ORDER)}
    data["target_order"] = data["target"].map(order)
    data = data.sort_values(["target_order", "nominal_fraction"])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Complete symmetric-allocation results. All effects are percentage-point changes in population event capture; brackets give central 95\% paired empirical-resampling ranges conditional on frozen scores and observed budgets.}",
        r"\label{tab:full_shapley}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Outcome & $q$ & Joint $\Delta$ & $\phi_E$ & $\phi_B$ & Symmetric share $s_B$ & Interaction \\",
        r"\midrule",
    ]
    for _, row in data.iterrows():
        lines.append(
            " & ".join(
                [
                    TARGET_LABEL[row["target"]],
                    f"{int(100 * row['nominal_fraction'])}\\%",
                    interval(
                        row["total_topq_contrast_point_pp"],
                        row["total_topq_contrast_ci_lower_pp"],
                        row["total_topq_contrast_ci_upper_pp"],
                    ),
                    interval(
                        row["shapley_availability_point_pp"],
                        row["shapley_availability_ci_lower_pp"],
                        row["shapley_availability_ci_upper_pp"],
                    ),
                    interval(
                        row["shapley_capacity_point_pp"],
                        row["shapley_capacity_ci_lower_pp"],
                        row["shapley_capacity_ci_upper_pp"],
                    ),
                    interval(
                        row["capacity_share_pct"],
                        row["share_low"],
                        row["share_high"],
                        digits=1,
                    )
                    + r"\%",
                    interval(
                        row["interaction_point_pp"],
                        row["interaction_ci_lower_pp"],
                        row["interaction_ci_upper_pp"],
                    ),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    write(OUT / "table_full_shapley.tex", lines)


def table_four_cells(shapley: pd.DataFrame) -> None:
    cohorts = pd.read_csv(DATA / "cohort_counts_validation.csv")
    event_denominator = (
        cohorts[
            (cohorts["year"] == 2023)
            & (cohorts["scenario"] == "all_record_upper_bound")
        ]
        .set_index("target")["event_n"]
        .to_dict()
    )
    order = {target: index for index, target in enumerate(TARGET_ORDER)}
    data = shapley.copy()
    data["target_order"] = data["target"].map(order)
    data = data.sort_values(["target_order", "nominal_fraction"])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Auditable four-cell point estimates. Each cell reports population event capture in percent followed by selected true events in parentheses. The two budgets are exact selected counts.}",
        r"\label{tab:four_cells}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccrrrr}",
        r"\toprule",
        r"Outcome & $q$ & Budgets $b_e/b_a$ & $C_{00}$: $S_e,b_e$ & $C_{01}$: $S_e,b_a$ & $C_{10}$: $S_a,b_e$ & $C_{11}$: $S_a,b_a$ \\",
        r"\midrule",
    ]
    for _, row in data.iterrows():
        event_n = int(event_denominator[row["target"]])
        cells = []
        for column in [
            "C00_early_Bearly",
            "C01_early_Ball",
            "C10_all_Bearly",
            "C11_all_Ball",
        ]:
            capture = float(row[column])
            selected_events = int(round(capture * event_n))
            cells.append(f"{100 * capture:.2f} ({selected_events:,})")
        lines.append(
            f"{TARGET_LABEL[row['target']]} & {int(100 * row['nominal_fraction'])}\\% & "
            f"{int(row['B_early_point']):,}/{int(row['B_all_point']):,} & "
            + " & ".join(cells)
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    write(OUT / "table_four_cells.tex", lines)


def table_model_sanity() -> None:
    data = pd.read_csv(DATA / "model_diagnostics.csv").set_index("target")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{CDC 2023 model-signal check for the frozen LightGBM score. These metrics characterize the ranking used by the protocol audit.}",
        r"\label{tab:model_sanity}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Outcome & Prevalence & ROC-AUC & PR-AUC \\",
        r"\midrule",
    ]
    for target in TARGET_ORDER:
        row = data.loc[target]
        lines.append(
            f"{TARGET_LABEL[target]} & {row['temporal_test_2023_prevalence']:.3f} & "
            f"{row['temporal_test_2023_roc_auc']:.3f} & "
            f"{row['temporal_test_2023_pr_auc']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write(OUT / "table_model_sanity.tex", lines)


def table_three_protocols() -> None:
    data = pd.read_csv(DATA / "canonical_protocol_results.csv")
    data = data[np.isclose(data["nominal_fraction"], 0.10)].copy()
    if len(data) != 27:
        raise ValueError(f"Expected 27 q=10% protocol rows, found {len(data)}")
    scenario_order = {
        "early_entry": 0,
        "any_prenatal_care": 1,
        "all_record_upper_bound": 2,
    }
    protocol_order = {
        "cohort_specific_top_fraction": 0,
        "fixed_absolute_budget": 1,
        "fixed_calibration_threshold": 2,
    }
    scenario_label = {
        "early_entry": "Early entry",
        "any_prenatal_care": "Documented start",
        "all_record_upper_bound": "All-record candidate bound",
    }
    protocol_label = {
        "cohort_specific_top_fraction": "Cohort-relative",
        "fixed_absolute_budget": "Fixed absolute",
        "fixed_calibration_threshold": "2022 threshold",
    }
    target_order = {target: index for index, target in enumerate(TARGET_ORDER)}
    data["target_order"] = data["target"].map(target_order)
    data["scenario_order"] = data["availability_scenario"].map(scenario_order)
    data["protocol_order"] = data["protocol"].map(protocol_order)
    data = data.sort_values(["target_order", "scenario_order", "protocol_order"])
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\caption{Complete CDC 2023 results at nominal $q=10\%$. Population capture and the event ceiling use the full target-population event denominator. The fixed-absolute protocol uses the all-record-derived absolute budget in this three-policy comparison.}",
        r"\label{tab:three_protocols}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.7pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Outcome & Candidate set & Policy & Eligible $n$ & Selected $n$ & Precision & Eligible recall & Population capture & Ceiling \\",
        r"\midrule",
    ]
    previous_target = None
    for _, row in data.iterrows():
        if previous_target is not None and row["target"] != previous_target:
            lines.append(r"\addlinespace")
        lines.append(
            f"{TARGET_LABEL[row['target']]} & {scenario_label[row['availability_scenario']]} & "
            f"{protocol_label[row['protocol']]} & {int(row['eligible_n']):,} & "
            f"{int(row['selected_n']):,} & {row['precision']:.3f} & "
            f"{row['recall_among_eligible']:.3f} & {row['population_event_capture']:.3f} & "
            f"{row['event_availability_ceiling']:.3f} \\\\"
        )
        previous_target = row["target"]
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    write(OUT / "table_three_protocols_q10.tex", lines)


def main() -> None:
    shapley, shares = load_shapley()
    table_main_shapley(shapley, shares)
    table_candidate_gate()
    table_policy_contrasts()
    table_score_transport()
    table_full_shapley(shapley, shares)
    table_four_cells(shapley)
    table_model_sanity()
    table_three_protocols()
    print("PASS: generated 8 LaTeX tables from source-data CSV files")


if __name__ == "__main__":
    main()