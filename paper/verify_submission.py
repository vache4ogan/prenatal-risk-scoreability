#!/usr/bin/env python3
"""Verify numerical and manuscript invariants before packaging the paper."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"


def fail(message: str) -> None:
    raise AssertionError(message)


def close(a: float, b: float, tol: float = 1e-10) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def verify_files() -> None:
    required = [
        "main.tex",
        "main.pdf",
        "references.bib",
        "neurips_2026.sty",
        "checklist.tex",
        "generate_figures.py",
        "generate_tables.py",
        "analysis_code/synthetic/run_tae_synthetic_shapley81.py",
        "analysis_code/temporal/run_2024_threshold_transport.py",
        "figures/figure1_headline.pdf",
        "figures/figure3_synthetic_stress_test.pdf",
        "source_data/replication_2024_results.csv",
        "source_data/sensitivity_no_priorterm.csv",
        "source_data/sensitivity_no_race.csv",
        "tables/table1_shapley_q10.tex",
        "tables/table_candidate_gate.tex",
        "tables/table_four_cells.tex",
        "tables/table_full_shapley.tex",
        "tables/table_model_sanity.tex",
        "tables/table_policy_contrasts_q10.tex",
        "tables/table_score_transport.tex",
        "tables/table_three_protocols_q10.tex",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        fail(f"Missing required files: {missing}")


def verify_hash_manifest() -> None:
    manifest = ROOT / "PACKAGE_SHA256.txt"
    if not manifest.is_file():
        return
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        entries[relative] = expected

    if len(entries) < 20:
        fail(f"Hash manifest is unexpectedly small: {len(entries)} files")

    for relative, expected in entries.items():
        path = ROOT / relative
        if not path.is_file():
            fail(f"Hash manifest references a missing file: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            fail(f"Hash mismatch: {relative}")


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
    rows = read_csv("synthetic_figure3_data.csv")
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

    manifest = json.loads((DATA / "synthetic_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        fail("Synthetic manifest is not PASS")
    if not all(manifest.get("invariants", {}).values()):
        fail("At least one synthetic manifest invariant failed")


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
        "target_preterm": 22668,
        "target_nicu": -17621,
        "target_lbw": -59716,
    }
    expected_rates = {
        "target_preterm": 0.10645300872812986,
        "target_nicu": 0.09498023126528163,
        "target_lbw": 0.08300824431442781,
    }
    for row in rows:
        target = row["target"]
        n_all = int(row["N_all"])
        e_all = int(row["E_all"])
        selected_n = int(row["selected_n"])
        true_positive_n = int(row["TP"])
        nominal_selected_n = int(row["nominal_selected_n"])
        exact_threshold = float(row["canonical_threshold_exact"])
        used_threshold = float(row["threshold_used_6dp"])

        if not close(used_threshold, round(exact_threshold, 6), tol=1e-12):
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


def verify_latex() -> None:
    text = (ROOT / "main.tex").read_text(encoding="utf-8")
    required = [
        r"\usepackage[dblblindworkshop]{neurips_2026}",
        r"\workshoptitle{TAE (Trust-AI-Eval): Can We Trust AI Evaluation?}",
        "Same Top Fraction, Different Workload",
        "Eligibility--capacity audit",
        "Fixed-budget and transported-threshold policies",
        "Applicability across deterministic score vectors",
        "Implementation in existing evaluation pipelines",
        r"\input{tables/table_candidate_gate.tex}",
        r"\input{tables/table_policy_contrasts_q10.tex}",
        r"\input{tables/table_score_transport.tex}",
        r"\input{tables/table_four_cells.tex}",
        r"\input{checklist}",
    ]
    for phrase in required:
        if phrase not in text:
            fail(f"Required manuscript phrase missing: {phrase}")

    banned = [
        "TODO",
        "jmlr",
        "pre-registered",
        "preregistered",
        "untouched test",
        "calibrated probability threshold",
        "brier_skill_vs_prevalence",
        "capacity accounts for about two thirds",
        "retrospective scoring upper bound",
        "does not guarantee the same policy",
        "does not replace the top-$q$ metric",
        "neither convention",
        "the unresolved case",
        "exploratory analysis",
        "no claim",
    ]
    lower = text.lower()
    for phrase in banned:
        if phrase.lower() in lower:
            fail(f"Banned or stale manuscript phrase found: {phrase}")

    included_tex = text + "\n" + "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((ROOT / "tables").glob("*.tex"))
    )
    labels = set(re.findall(r"\\label\{([^}]+)\}", included_tex))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text))
    if refs - labels:
        fail(f"References without labels: {sorted(refs - labels)}")

    bib = (ROOT / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@[A-Za-z]+\{\s*([^,]+),", bib))
    citation_groups = re.findall(r"\\cite[pt]?\{([^}]+)\}", text)
    cite_keys = {key.strip() for group in citation_groups for key in group.split(",")}
    if cite_keys - bib_keys:
        fail(f"Citations without BibTeX entries: {sorted(cite_keys - bib_keys)}")

    log = ROOT / "build" / "main.log"
    if log.is_file():
        log_text = log.read_text(encoding="utf-8", errors="replace")
        diagnostics = ["undefined references", "citation `", "overfull \\hbox", "overfull \\vbox"]
        found = [item for item in diagnostics if item in log_text.lower()]
        if found:
            fail(f"LaTeX diagnostics found in build/main.log: {found}")

    try:
        from pypdf import PdfReader
    except ImportError:
        return

    pdf = PdfReader(str(ROOT / "main.pdf"))
    if not (17 <= len(pdf.pages) <= 20):
        fail(f"Unexpected packaged review PDF length: {len(pdf.pages)} pages")
    page_text = [page.extract_text() or "" for page in pdf.pages]
    reference_pages = [index for index, text_page in enumerate(page_text) if "References" in text_page]
    conclusion_pages = [index for index, text_page in enumerate(page_text) if "Conclusion" in text_page]
    if not reference_pages or min(reference_pages) != 8:
        fail("References must begin on page 9 after eight main-text pages")
    if not conclusion_pages or min(conclusion_pages) != 7:
        fail("Conclusion must appear on main-text page 8")
    if any("??" in text_page for text_page in page_text):
        fail("Unresolved markers found in packaged review PDF")


def main() -> int:
    checks = [
        verify_files,
        verify_hash_manifest,
        verify_shapley,
        verify_synthetic,
        verify_protocol_and_score_claims,
        verify_temporal_2024,
        verify_feature_sensitivities,
        verify_latex,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print("PASS submission package")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1)
