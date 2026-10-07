"""Numerical invariants for aggregate reference results; no individual records."""
from __future__ import annotations
import csv
import hashlib
import math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/reference"


def fail(message: str) -> None:
    raise AssertionError(message)


def close(a: float, b: float, tol: float = 1e-10) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def read_csv(name: str) -> list[dict[str, str]]:
    group = ("sensitivity" if name.startswith("sensitivity_") else
             "synthetic" if name.startswith("synthetic_") else
             "2024" if name == "replication_2024_results.csv" else "2023")
    with (DATA / group / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def verify_shapley() -> None:
    rows = read_csv("shapley_main_results.csv")
    if len(rows) != 6:
        fail(f"Expected 6 primary Shapley rows, found {len(rows)}")

    for row in rows:
        c00 = float(row["C00_early_Bearly"])
        c10 = float(row["C10_all_Bearly"])
        c01 = float(row["C01_early_Ball"])
        c11 = float(row["C11_all_Ball"])
        total = float(row["total_topq_contrast_point"])
        phi_e = float(row["shapley_availability_point"])
        phi_b = float(row["shapley_capacity_point"])
        interaction = float(row["interaction_point"])

        if not close(total, c11 - c00):
            fail(f"Joint contrast identity failed for {row['target']} q={row['nominal_fraction']}")
        if not close(total, phi_e + phi_b):
            fail(f"Shapley efficiency failed for {row['target']} q={row['nominal_fraction']}")
        if not close(interaction, c11 - c10 - c01 + c00):
            fail(f"Interaction identity failed for {row['target']} q={row['nominal_fraction']}")
        if c01 + 1e-12 < c00 or c11 + 1e-12 < c10 or phi_b < -1e-12:
            fail(f"Capacity monotonicity failed for {row['target']} q={row['nominal_fraction']}")

    fixed_budget_eligibility_pp = []
    for row in rows:
        c00 = float(row["C00_early_Bearly"])
        c01 = float(row["C01_early_Ball"])
        c10 = float(row["C10_all_Bearly"])
        c11 = float(row["C11_all_Ball"])
        fixed_budget_eligibility_pp.extend([100 * (c10 - c00), 100 * (c11 - c01)])
    if not math.isclose(min(fixed_budget_eligibility_pp), 0.989436, abs_tol=1e-5):
        fail(f"Unexpected minimum fixed-budget eligibility contrast: {min(fixed_budget_eligibility_pp)}")
    if not math.isclose(max(fixed_budget_eligibility_pp), 2.930969, abs_tol=1e-5):
        fail(f"Unexpected maximum fixed-budget eligibility contrast: {max(fixed_budget_eligibility_pp)}")

    bootstrap = read_csv("shapley_bootstrap_replicates.csv")
    if len(bootstrap) != 3000:
        fail(f"Expected 3000 paired bootstrap rows, found {len(bootstrap)}")

    cohorts = read_csv("cohort_counts_validation.csv")
    cohort_lookup = {
        (row["target"], row["scenario"]): row
        for row in cohorts
        if row["year"] == "2023"
    }
    expected_random_shares = {
        "target_preterm": 87.928162,
        "target_nicu": 83.725241,
        "target_lbw": 79.678537,
    }
    q10 = [row for row in rows if close(float(row["nominal_fraction"]), 0.10)]
    for row in q10:
        target = row["target"]
        early = cohort_lookup[(target, "early_entry")]
        all_records = cohort_lookup[(target, "all_record_upper_bound")]
        n_e = int(early["n_records"])
        e_e = int(early["event_n"])
        n_a = int(all_records["n_records"])
        e_a = int(all_records["event_n"])
        b_e = int(float(row["B_early_point"]))
        b_a = int(float(row["B_all_point"]))
        c00 = (b_e / n_e) * (e_e / e_a)
        c01 = (b_a / n_e) * (e_e / e_a)
        c10 = b_e / n_a
        c11 = b_a / n_a
        phi_b = 0.5 * ((c01 - c00) + (c11 - c10))
        share = 100 * phi_b / (c11 - c00)
        if not math.isclose(share, expected_random_shares[target], abs_tol=1e-5):
            fail(f"Random-ranking benchmark drifted for {target}: {share}")


def verify_synthetic() -> None:
    rows = read_csv("synthetic_condition_summary.csv")
    if len(rows) != 81:
        fail(f"Expected 81 synthetic conditions, found {len(rows)}")
    if sum(as_bool(row["naive_fixed_budget_sign_reversal"]) for row in rows) != 18:
        fail("Expected 18 mean sign reversals")
    if sum(as_bool(row["fixed_budget_interval_below_zero"]) for row in rows) != 9:
        fail("Expected 9 robust negative fixed-budget conditions")

    max_mean_interaction = max(abs(float(row["mean_interaction_effect"])) for row in rows)
    max_replicate_interaction = max(float(row["max_abs_interaction_replicate"]) for row in rows)
    if not (0.0948 <= max_mean_interaction <= 0.0950):
        fail(f"Unexpected maximum mean interaction: {max_mean_interaction}")
    if not (0.1071 <= max_replicate_interaction <= 0.1073):
        fail(f"Unexpected maximum replicate interaction: {max_replicate_interaction}")

    capacity_main = [100 * float(row["mean_capacity_main_effect"]) for row in rows]
    if not math.isclose(min(capacity_main), 0.740265, abs_tol=1e-5):
        fail(f"Unexpected minimum B0 capacity increment: {min(capacity_main)}")
    if not math.isclose(max(capacity_main), 15.947923, abs_tol=1e-5):
        fail(f"Unexpected maximum B0 capacity increment: {max(capacity_main)}")

    example = [
        row
        for row in rows
        if row["availability_relation"] == "higher_risk_more_available"
        and row["model_quality"] == "medium"
        and close(float(row["availability_target_fraction"]), 0.7)
        and close(float(row["nominal_fraction"]), 0.1)
    ]
    if len(example) != 1:
        fail("Synthetic narrative example is not unique")
    row = example[0]
    if not (0.0550 <= float(row["mean_naive_top_fraction_effect"]) <= 0.0554):
        fail("Synthetic example cohort-relative effect drifted")
    if not (-0.0030 <= float(row["mean_availability_effect_at_all_budget"]) <= -0.0026):
        fail("Synthetic example fixed-budget effect drifted")
    if not (0.92 <= float(row["sign_reversal_replicate_fraction"]) <= 0.94):
        fail("Synthetic example reversal frequency drifted")



def verify_synthetic_specification() -> None:
    import numpy as np
    import yaml

    config = yaml.safe_load(
        (ROOT / "configs/canonical/synthetic_experiment_shapley81.yaml")
        .read_text(encoding="utf-8")
    )
    simulation = config["simulation"]
    expected = {
        "seed": 2028, "replicates": 100, "population_n": 100000,
        "calibration_n": 50000, "outcome_prevalence": 0.1,
        "outcome_risk_slope": 1.0, "quadrature_nodes": 80,
        "availability_fractions": [0.5, 0.7, 0.9],
    }
    for key, value in expected.items():
        if simulation[key] != value:
            fail(f"Synthetic manuscript/config mismatch: {key}")
    if [x["slope"] for x in simulation["availability_associations"]] != [-1, 0, 1]:
        fail("Synthetic eligibility slopes drifted")
    if [x["noise_sd"] for x in simulation["model_qualities"]] != [0.5, 1, 2]:
        fail("Synthetic score noise levels drifted")
    if config["evaluation"]["fractions"] != [0.05, 0.1, 0.2]:
        fail("Synthetic selection fractions drifted")

    # Independently check the rounded intercepts printed in Appendix C.
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    z, w = np.sqrt(2) * nodes, weights / np.sqrt(np.pi)
    intercepts = [
        (0.1, 1, -2.5642215001), (0.5, 0, 0), (0.5, 1, 0),
        (0.7, 0, 0.8472978604), (0.7, 1, 1.0184006520),
        (0.9, 0, 2.1972245773), (0.9, 1, 2.5642215001),
    ]
    for target, slope, intercept in intercepts:
        marginal = float(np.sum(w / (1 + np.exp(-intercept - slope * z))))
        if abs(marginal - target) > 1e-10:
            fail(f"Incorrect synthetic intercept: {target}, {slope}, {intercept}")


def verify_protocol_and_score_claims() -> None:
    protocol_rows = read_csv("canonical_protocol_results.csv")
    q10 = [row for row in protocol_rows if close(float(row["nominal_fraction"]), 0.10)]
    lookup = {
        (row["target"], row["protocol"], row["availability_scenario"]): row
        for row in q10
    }
    targets = ["target_preterm", "target_nicu", "target_lbw"]

    added_slots: list[int] = []
    added_events: list[int] = []
    marginal_yields: list[float] = []
    threshold_early_rates: list[float] = []
    threshold_all_rates: list[float] = []
    threshold_workload_ratios: list[float] = []
    threshold_capture_gains: list[float] = []
    for target in targets:
        early = lookup[(target, "cohort_specific_top_fraction", "early_entry")]
        all_records = lookup[
            (target, "cohort_specific_top_fraction", "all_record_upper_bound")
        ]
        slots = int(all_records["selected_n"]) - int(early["selected_n"])
        events = int(all_records["true_positive_n"]) - int(early["true_positive_n"])
        added_slots.append(slots)
        added_events.append(events)
        marginal_yields.append(100 * events / slots)

        threshold_early = lookup[
            (target, "fixed_calibration_threshold", "early_entry")
        ]
        threshold_all = lookup[
            (target, "fixed_calibration_threshold", "all_record_upper_bound")
        ]
        threshold_early_rates.append(
            100 * int(threshold_early["selected_n"]) / int(threshold_early["eligible_n"])
        )
        threshold_all_rates.append(
            100 * int(threshold_all["selected_n"]) / int(threshold_all["eligible_n"])
        )
        threshold_workload_ratios.append(
            int(threshold_all["selected_n"]) / int(threshold_early["selected_n"])
        )
        threshold_capture_gains.append(
            100
            * (
                float(threshold_all["population_event_capture"])
                - float(threshold_early["population_event_capture"])
            )
        )

    if (min(added_slots), max(added_slots)) != (87931, 88023):
        fail(f"Unexpected added-slot range: {min(added_slots)}--{max(added_slots)}")
    if (min(added_events), max(added_events)) != (16505, 17177):
        fail(f"Unexpected added-event range: {min(added_events)}--{max(added_events)}")
    if not (18.74 <= min(marginal_yields) <= 18.76 and 19.53 <= max(marginal_yields) <= 19.54):
        fail(f"Unexpected marginal-yield range: {min(marginal_yields)}--{max(marginal_yields)}")
    if not (9.14 <= min(threshold_early_rates) <= 9.16 and 9.81 <= max(threshold_early_rates) <= 9.83):
        fail("Transported-threshold early selection rates drifted")
    if not (10.19 <= min(threshold_all_rates) <= 10.21 and 10.39 <= max(threshold_all_rates) <= 10.41):
        fail("Transported-threshold all-record selection rates drifted")
    if not (1.41 <= min(threshold_workload_ratios) <= 1.43 and 1.49 <= max(threshold_workload_ratios) <= 1.50):
        fail("Transported-threshold workload ratios drifted")
    if not (6.35 <= min(threshold_capture_gains) <= 6.37 and 8.18 <= max(threshold_capture_gains) <= 8.20):
        fail("Transported-threshold capture gains drifted")

    diagnostics = read_csv("model_diagnostics.csv")
    if len(diagnostics) != 3:
        fail(f"Expected three model-diagnostic rows, found {len(diagnostics)}")
    roc_drifts = []
    ap_drifts = []
    brier_skills = []
    for row in diagnostics:
        roc_drifts.append(
            abs(float(row["temporal_test_2023_roc_auc"]) - float(row["calibration_2022_roc_auc"]))
        )
        ap_drifts.append(
            abs(float(row["temporal_test_2023_pr_auc"]) - float(row["calibration_2022_pr_auc"]))
        )
        prevalence = float(row["temporal_test_2023_prevalence"])
        baseline = prevalence * (1 - prevalence)
        brier_skills.append(1 - float(row["temporal_test_2023_brier"]) / baseline)
    if max(roc_drifts) >= 0.0035 or max(ap_drifts) >= 0.0030:
        fail("Frozen-score temporal metric drift exceeded the manuscript bound")
    if not (0.016 <= min(brier_skills) <= 0.018 and 0.023 <= max(brier_skills) <= 0.025):
        fail("Brier-skill range drifted")


def verify_temporal_2024() -> None:
    rows = read_csv("replication_2024_results.csv")
    if len(rows) != 3:
        fail(f"Expected three CDC 2024 workload rows, found {len(rows)}")

    expected_differences = {
        "target_preterm": 21112,
        "target_nicu": 22400,
        "target_lbw": 6952,
    }
    expected_rates = {
        "target_preterm": 0.10601005471449963,
        "target_nicu": 0.10638111795976517,
        "target_lbw": 0.10197807776072423,
    }
    for row in rows:
        target = row["target"]
        n_all = int(row["N_all"])
        e_all = int(row["E_all"])
        selected_n = int(row["selected_n"])
        true_positive_n = int(row["TP"])
        nominal_selected_n = int(row["nominal_selected_n"])
        exact_threshold = float(row["canonical_threshold_exact"])
        used_threshold = float(row["threshold_used"])

        if not close(used_threshold, exact_threshold, tol=1e-15):
            fail(f"CDC 2024 threshold precision mismatch for {target}")
        if selected_n - nominal_selected_n != expected_differences[target]:
            fail(f"CDC 2024 workload difference drifted for {target}")
        if not close(float(row["prevalence"]), e_all / n_all):
            fail(f"CDC 2024 prevalence arithmetic failed for {target}")
        if not close(float(row["selection_rate"]), selected_n / n_all):
            fail(f"CDC 2024 selection-rate arithmetic failed for {target}")
        if not close(float(row["precision"]), true_positive_n / selected_n):
            fail(f"CDC 2024 precision arithmetic failed for {target}")
        if not close(float(row["population_event_capture"]), true_positive_n / e_all):
            fail(f"CDC 2024 capture arithmetic failed for {target}")
        if not close(float(row["selection_rate"]), expected_rates[target]):
            fail(f"CDC 2024 selection rate drifted for {target}")

    import numpy as np
    import pandas as pd

    directory = DATA / "2024"
    cells = pd.read_csv(directory / "four_cells.csv")
    summary = pd.read_csv(directory / "summary.csv")
    replicates = pd.read_csv(directory / "bootstrap_replicates.csv")
    if (len(cells), len(summary), len(replicates)) != (24, 6, 3000):
        fail("Incomplete CDC 2024 four-cell replication")
    for row in summary.to_dict("records"):
        c00, c10, c01, c11 = [row[k] for k in
                            ["C00_early_Bearly", "C10_all_Bearly", "C01_early_Ball", "C11_all_Ball"]]
        phi_e = ((c10-c00)+(c11-c01))/2
        phi_b = ((c01-c00)+(c11-c10))/2
        for actual, expected in [(row["shapley_capacity"], phi_b),
                                 (row["shapley_availability"], phi_e),
                                 (row["total_topq_contrast"], c11-c00),
                                 (row["capacity_share"], phi_b/(c11-c00))]:
            if not close(actual, expected):
                fail("CDC 2024 point-estimate identity failed")
        rep = replicates.loc[(replicates.target == row["target"]) &
                             np.isclose(replicates.nominal_fraction, row["nominal_fraction"])].copy()
        if len(rep) != 500 or rep.replicate.nunique() != 500:
            fail("CDC 2024 bootstrap group incomplete")
        rep["capacity_share"] = rep.shapley_capacity / rep.total_topq_contrast
        if not np.allclose(rep.shapley_capacity + rep.shapley_availability,
                           rep.total_topq_contrast, atol=1e-12):
            fail("CDC 2024 replicate accounting failed")
        if not (rep.B_early.eq(row["B_early"]).all() and rep.B_all.eq(row["B_all"]).all()):
            fail("CDC 2024 resample budgets changed")
        for key in ["shapley_capacity", "shapley_availability", "interaction", "total_topq_contrast",
                    "availability_at_early_budget", "ordered_availability_at_all_budget", "capacity_share"]:
            low, high = np.quantile(rep[key], [0.025, 0.975])
            if not close(row[key + "_lower"], low) or not close(row[key + "_upper"], high):
                fail(f"CDC 2024 percentile mismatch: {key}")
    if not np.allclose(cells.true_positive_n / cells.full_population_event_n,
                       cells.population_event_capture, atol=1e-12):
        fail("CDC 2024 cell event-denominator mismatch")


def verify_feature_sensitivities() -> None:
    no_prior = read_csv("sensitivity_no_priorterm.csv")
    if len(no_prior) != 6:
        fail(f"Expected six no-PRIORTERM rows, found {len(no_prior)}")
    no_prior_shares = []
    for row in no_prior:
        total = float(row["total_topq_contrast_point"])
        capacity = float(row["shapley_capacity_point"])
        no_prior_shares.append(100 * capacity / total)
        if float(row["interaction_ci_lower"]) <= 0:
            fail("No-PRIORTERM interaction interval crossed zero")
    if not (61.5 <= min(no_prior_shares) <= 61.6 and 64.2 <= max(no_prior_shares) <= 64.4):
        fail("No-PRIORTERM capacity-share range drifted")

    no_race = read_csv("sensitivity_no_race.csv")
    if len(no_race) != 3:
        fail(f"Expected three no-race rows, found {len(no_race)}")
    no_race_shares = [
        100
        * float(row["no_race_shapley_capacity_pp"])
        / float(row["no_race_topq_total_pp"])
        for row in no_race
    ]
    if not (60.9 <= min(no_race_shares) <= 61.0 and 62.3 <= max(no_race_shares) <= 62.5):
        fail("No-race capacity-share range drifted")
