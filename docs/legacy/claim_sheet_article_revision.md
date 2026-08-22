Claim Sheet

Paper working title

Who Can Be Scored? Cohort Availability Limits Population-Level Capture in Prenatal Risk Triage

Research question

Does restricting prenatal risk scoring to records with prenatal-care entry during months 1–3 reduce population-level event capture when early-entry and full-cohort policies use the same fitted model and the same absolute referral budget?

Primary estimand

For each outcome and budget:

Full-cohort population event capture − Early-entry population event capture

Both policies use:

the same fitted LightGBM model;

the same CDC 2023 prediction vector;

the same absolute number of selected records;

separate top-B selection within the records eligible for each policy.

Primary population and model

Population: U.S.-resident singleton births.

Development data: CDC Natality 2022.

Temporal test: CDC Natality 2023.

Model: LightGBM.

Feature set: booking_strict.

Outcomes:

preterm delivery;

NICU admission;

low birth weight.

Budgets: 5% and 10% of the full target-specific population.

Early-entry eligibility: prenatal-care entry during months 1–3.

Main claim

Restricting scoring to early-entry records reduces population event capture even after model identity and absolute referral capacity are held fixed.

Canonical CDC 2023 result at the 10% budget

Outcome

Early capture

Full capture

Full − early

95% paired-bootstrap CI

Additional captured events

Preterm delivery

18.83%

21.41%

+2.58 pp

2.47–2.70 pp

7,821

NICU admission

17.77%

20.50%

+2.73 pp

2.62–2.84 pp

8,271

Low birth weight

20.56%

23.98%

+3.43 pp

3.28–3.57 pp

8,356

Supporting claims and evidence

Claim

Evidence

Article placement

Early scoreability imposes a structural ceiling before classifier errors.

About 74.7% of records were early-entry eligible, but only 69.4–71.8% of events occurred inside that cohort.

Main text

Full-cohort eligibility improves event capture at the same absolute budget.

Positive full-minus-early differences for all three outcomes at 5% and 10%.

Main text

The primary effect is statistically stable.

Paired full-size bootstrap, 500 replicates; all CDC 2023 95% intervals above zero.

Main text

The effect reproduces temporally.

Same direction and similar magnitude on held-out CDC 2022 and CDC 2023 temporal test.

Main text or appendix

The gain comes from events outside the early-entry cohort.

Mechanism table: gains in after-month-3, no-care, and unknown groups exceed losses within months 1–3.

Main text

The effect is not caused by a larger referral budget.

Early and full select exactly the same absolute number of records in the primary analysis.

Main text

The effect is not explained by substantial LightGBM miscalibration.

Raw calibration intercepts near 0, slopes near 1, and no Brier improvement after Platt scaling.

Appendix / brief main-text statement

The direction is robust to inclusion of multiple births.

Corrected all-birth fixed-model equal-budget sensitivity remains positive for all outcomes.

Appendix

Pregnancy-period variables add retrospective predictive information.

pregnancy_oracle improves PR-AUC, ROC-AUC, Brier, and event capture.

Appendix only

Mechanism numbers at the 10% budget

Outcome

Additional TP outside early cohort

TP reduction within months 1–3

Net additional TP

Preterm delivery

+22,271

−14,450

+7,821

NICU admission

+20,557

−12,286

+8,271

Low birth weight

+21,857

−13,501

+8,356

Sensitivity result

All-birth analysis at the 10% budget

Outcome

Full − early

Additional captured events

Preterm delivery

+2.18 pp

8,159

NICU admission

+2.51 pp

8,821

Low birth weight

+2.85 pp

8,797

This is sensitivity evidence only because multiple infant records from one pregnancy may be dependent and a reliable pregnancy-level identifier is unavailable.

Secondary oracle result

pregnancy_oracle is a retrospective upper-bound analysis, not a deployable early clinical model.

At the 10% fixed budget, full-minus-early differences were:

preterm: +2.698 pp;

NICU: +2.838 pp;

low birth weight: +3.842 pp.

Interpretation guardrails

Use these formulations:

“CDC 2023 temporal test,” not “external validation.”

“Full precision was higher at the same absolute budget.”

“Recall among eligible records is conditional and is not population event capture.”

“The primary effect is based on booking_strict LightGBM.”

“The shared-threshold analysis had unequal realized referral counts and is supplementary.”

“The all-birth analysis is sensitivity only.”

“The oracle feature set provides retrospective predictive information but is not an early deployable model.”

Claims the paper does not make

The analysis does not establish that:

LightGBM is the optimal model architecture;

5% or 10% is the clinically optimal referral budget;

deployment would improve maternal or neonatal outcomes;

prenatal-care entry causally changes outcomes;

currently unscoreable records can be scored without additional data collection;

pregnancy_oracle is valid at the booking visit;

results automatically generalize beyond CDC Natality or the United States;

class balancing improves training;

all-birth uncertainty is valid under independent-record assumptions.

Canonical evidence files

singleton_fixed_budget_comparisons.csv

paired_bootstrap_summary.csv

mechanism_table_10pct_comparisons.csv

lightgbm_calibration_metrics.csv

all_births_fixed_budget_comparisons.csv

article_revision_summary.md

sanitized run_manifest.json

Decision

Primary hypothesis: supported.

The corrected fixed-model equal-budget analysis shows that cohort availability creates a separate population-level performance limitation beyond classifier error.