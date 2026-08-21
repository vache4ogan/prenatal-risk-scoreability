# TAE Canonical Main Results

## Status and source of truth

This document is the authoritative main-results summary for the **TAE 2026 canonical analysis**.

Canonical analysis characteristics:

- feature set: `landmark_strict` (9 raw features);
- one LightGBM model per target;
- no cohort-specific refitting;
- CDC Natality 2022 for development/calibration;
- CDC Natality 2023 for temporal testing only;
- primary population: target-specific U.S.-resident singleton births;
- availability scenarios: `early_entry`, `any_prenatal_care`, and `all_record_upper_bound`;
- evaluation protocols: `cohort_specific_top_fraction`, `fixed_absolute_budget`, and `fixed_calibration_threshold`;
- nominal fractions: 5% and 10%;
- paired full-size record-level nonparametric bootstrap with 500 replicates.

**Important evidence-status note.** The existing legacy `claim_sheet.md` contains results from the previous `booking_strict` / article-revision generation and must not be used as the numerical source of truth for the TAE canonical paper. The current canonical numbers are those in `canonical_protocol_results.csv`, `canonical_protocol_comparisons.csv`, `effect_decomposition.csv`, and `bootstrap_summary.csv`.

---

## 1. Primary evaluation question

The primary question is:

> Holding the fitted model, CDC 2023 score vector, and absolute selection capacity fixed, how much does population-level event capture change when the candidate set expands beyond early prenatal-care entry?

For target $t$ and absolute budget $B$,

**population event capture**

= true positives selected / all events in the target-specific CDC 2023 population.

The primary contrast is:

**all-record upper-bound population event capture − early-entry population event capture**

under `fixed_absolute_budget`.

This contrast changes the eligible candidate set while holding the number selected fixed.

The `all_record_upper_bound` scenario is a **retrospective evaluation counterfactual**, not a deployable early clinical policy.

---

## 2. Structural availability before model selection

Availability itself limits the maximum population reach before classifier errors are considered.

| Target | Availability scenario | Eligible N | Records available | Eligible events | Event availability ceiling |
|---|---|---:|---:|---:|---:|
| Preterm | Early entry | 2,601,060 | 74.735% | 217,632 | 71.826% |
| Preterm | Any prenatal care | 3,340,358 | 95.977% | 275,914 | 91.060% |
| NICU | Early entry | 2,597,274 | 74.708% | 214,174 | 70.618% |
| NICU | Any prenatal care | 3,335,329 | 95.937% | 279,316 | 92.097% |
| LBW | Early entry | 2,600,093 | 74.709% | 169,152 | 69.388% |
| LBW | Any prenatal care | 3,338,986 | 95.939% | 223,218 | 91.567% |

Early-entry eligibility covers about 74.7% of target-specific records, but only 69.4–71.8% of events. Thus, the early-entry candidate set is disproportionately missing outcome events relative to its share of records.

Extending eligibility to any documented prenatal-care entry raises record availability to about 95.9–96.0%, but the corresponding event ceiling remains about 91.1–92.1%. Events among `no_care` and `unknown_care` therefore remain outside the any-prenatal-care candidate set.

These ceilings are structural properties of the candidate set; they arise before ranking or top-B selection.

---

## 3. Primary result: 10% fixed absolute budget

For the nominal 10% analysis, the budget is defined as approximately 10% of the target-specific **all-record** CDC 2023 population. That same absolute number of records is then selected from each availability scenario.

Consequently, “10% fixed budget” does **not** mean that 10% of every eligible cohort is selected. For example, in preterm the same 348,038 selections correspond to 13.381% of `early_entry`, 10.419% of `any_prenatal_care`, and 10.000% of `all_record_upper_bound`.

| Target | Scenario | Eligible N | Selected N | TP | Precision | Recall among eligible | Population capture | Event ceiling |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Preterm | Early entry | 2,601,060 | 348,038 | 54,216 | 15.578% | 24.912% | 17.893% | 71.826% |
| Preterm | Any prenatal care | 3,340,358 | 348,038 | 57,413 | 16.496% | 20.808% | 18.948% | 91.060% |
| Preterm | All-record upper bound | 3,480,382 | 348,038 | 61,503 | 17.671% | 20.298% | 20.298% | 100.000% |
| NICU | Early entry | 2,597,274 | 347,658 | 52,160 | 15.003% | 24.354% | 17.198% | 70.618% |
| NICU | Any prenatal care | 3,335,329 | 347,658 | 56,161 | 16.154% | 20.107% | 18.518% | 92.097% |
| NICU | All-record upper bound | 3,476,584 | 347,658 | 59,201 | 17.029% | 19.520% | 19.520% | 100.000% |
| LBW | Early entry | 2,600,093 | 348,032 | 48,505 | 13.937% | 28.675% | 19.897% | 69.388% |
| LBW | Any prenatal care | 3,338,986 | 348,032 | 52,674 | 15.135% | 23.598% | 21.608% | 91.567% |
| LBW | All-record upper bound | 3,480,316 | 348,032 | 55,650 | 15.990% | 22.828% | 22.828% | 100.000% |

### Primary early-entry → all-record contrasts

| Target | Contrast | Δ selected | Δ TP | Δ population capture | Paired-bootstrap 95% CI |
|---|---|---:|---:|---:|---:|
| Preterm | Early entry → All-record upper bound | 0 | 7,287 | +2.405 pp | [2.300, 2.511] pp |
| NICU | Early entry → All-record upper bound | 0 | 7,041 | +2.322 pp | [2.222, 2.431] pp |
| LBW | Early entry → All-record upper bound | 0 | 7,145 | +2.931 pp | [2.799, 3.078] pp |

At the same absolute selection budget, expanding eligibility from early entry to the retrospective all-record upper bound increased CDC 2023 population event capture for all three outcomes:

- **Preterm:** +2.405 percentage points, corresponding to +7,287 captured events; 95% CI [+2.300, +2.511] pp.
- **NICU:** +2.322 percentage points, corresponding to +7,041 captured events; 95% CI [+2.222, +2.431] pp.
- **LBW:** +2.931 percentage points, corresponding to +7,145 captured events; 95% CI [+2.799, +3.078] pp.

All three paired-bootstrap intervals are above zero.

---

## 4. Where the gain appears as availability expands

The intermediate `any_prenatal_care` scenario shows that broader care-access availability recovers part, but not all, of the early-to-all-record gap.

| Target | Contrast | Δ selected | Δ TP | Δ population capture | Paired-bootstrap 95% CI |
|---|---|---:|---:|---:|---:|
| Preterm | Early entry → Any prenatal care | 0 | 3,197 | +1.055 pp | [0.956, 1.144] pp |
| Preterm | Any prenatal care → All-record upper bound | 0 | 4,090 | +1.350 pp | [1.293, 1.412] pp |
| Preterm | Early entry → All-record upper bound | 0 | 7,287 | +2.405 pp | [2.300, 2.511] pp |
| NICU | Early entry → Any prenatal care | 0 | 4,001 | +1.319 pp | [1.229, 1.418] pp |
| NICU | Any prenatal care → All-record upper bound | 0 | 3,040 | +1.002 pp | [0.954, 1.063] pp |
| NICU | Early entry → All-record upper bound | 0 | 7,041 | +2.322 pp | [2.222, 2.431] pp |
| LBW | Early entry → Any prenatal care | 0 | 4,169 | +1.710 pp | [1.596, 1.829] pp |
| LBW | Any prenatal care → All-record upper bound | 0 | 2,976 | +1.221 pp | [1.156, 1.288] pp |
| LBW | Early entry → All-record upper bound | 0 | 7,145 | +2.931 pp | [2.799, 3.078] pp |

Under fixed absolute capacity, `early_entry → any_prenatal_care` increases capture by:

- Preterm: +1.055 pp (+3,197 TP);
- NICU: +1.319 pp (+4,001 TP);
- LBW: +1.710 pp (+4,169 TP).

The remaining `any_prenatal_care → all_record_upper_bound` increment is:

- Preterm: +1.350 pp (+4,090 TP);
- NICU: +1.002 pp (+3,040 TP);
- LBW: +1.221 pp (+2,976 TP).

This pattern supports the interpretation that expanding candidate-set availability through later prenatal-care entry recovers some events, while events among no-care and unknown-care records remain outside the any-care candidate pool.

---

## 5. Why the evaluation protocol changes the apparent effect

The three canonical protocols answer different operational questions.

### Protocol A — `cohort_specific_top_fraction`

Each availability cohort receives its own top 10%. Because the cohorts have different sizes, the absolute number selected changes.

### Protocol B — `fixed_absolute_budget`

Every availability scenario receives the same target-specific absolute number of selections. This is the primary protocol for isolating candidate-set expansion at fixed resource capacity.

### Protocol C — `fixed_calibration_threshold`

One threshold selected on the CDC 2022 target-specific all-record calibration pool is applied unchanged to all CDC 2023 availability scenarios. This fixes the score decision rule, but realized selection counts differ.

### Early-entry → all-record comparison at nominal 10%

| Target | Protocol | Δ selected | Δ TP | Δ population capture |
|---|---|---:|---:|---:|
| Preterm | Cohort-specific top-q | 87,932 | 17,177 | +5.669 pp |
| Preterm | Fixed absolute budget | 0 | 7,287 | +2.405 pp |
| Preterm | Fixed calibration threshold | 116,222 | 20,530 | +6.776 pp |
| NICU | Cohort-specific top-q | 87,931 | 16,949 | +5.588 pp |
| NICU | Fixed absolute budget | 0 | 7,041 | +2.322 pp |
| NICU | Fixed calibration threshold | 106,548 | 19,277 | +6.356 pp |
| LBW | Cohort-specific top-q | 88,023 | 16,505 | +6.771 pp |
| LBW | Fixed absolute budget | 0 | 7,145 | +2.931 pp |
| LBW | Fixed calibration threshold | 117,061 | 19,959 | +8.187 pp |

The cohort-specific top-10% protocol yields an apparent early-to-all-record gain of 5.6–6.8 pp, but that comparison simultaneously expands the candidate set and increases absolute selection capacity by about 88,000 records.

The fixed-threshold protocol likewise produces unequal realized selection counts, so its larger capture differences should not be interpreted as a pure fixed-resource availability effect.

The fixed-absolute-budget result is therefore the primary same-model, fixed-capacity contrast.

---

## 6. Exact same-model decomposition

For the early-entry → all-record comparison, the cohort-specific top-q difference can be decomposed exactly into:

**naive top-q effect = availability effect at the larger budget + extra-capacity effect within the early-entry cohort**

At 10%:

| Target | Naive top-q effect | Availability component | Extra-capacity component | Identity error |
|---|---:|---:|---:|---:|
| Preterm | +5.669 pp | +2.405 pp | +3.264 pp | 0.0e+00 |
| NICU | +5.588 pp | +2.322 pp | +3.267 pp | 0.0e+00 |
| LBW | +6.771 pp | +2.931 pp | +3.840 pp | 0.0e+00 |

The decomposition identity is exact for every target (`identity_error = 0`).

Thus, more than half of the apparent gain in the cohort-specific top-10% comparison is attributable to the additional number of available selection slots rather than candidate-set availability alone.

This is a central evaluation result: comparing the same nominal top-q across differently sized eligible cohorts changes both **who can be selected** and **how many can be selected**.

---

## 7. Robustness at the 5% fixed absolute budget

The direction of the primary availability effect is unchanged at the smaller nominal budget.

| Target | Additional TP, early → all | Δ population capture | Paired-bootstrap 95% CI |
|---|---:|---:|---:|
| Preterm | 4,013 | +1.324 pp | [1.239, 1.404] pp |
| NICU | 4,059 | +1.338 pp | [1.262, 1.412] pp |
| LBW | 3,974 | +1.630 pp | [1.533, 1.738] pp |

The 5% results are suitable for the appendix or a concise robustness statement in the main text.

---

## 8. Model-performance context

The paper is not a SOTA-model paper. The same `landmark_strict` LightGBM ranking is intentionally held fixed across availability scenarios.

| Target | CDC 2023 N | Event prevalence | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|
| Preterm | 3,480,382 | 8.706% | 0.146 | 0.612 | 0.0780 |
| NICU | 3,476,584 | 8.724% | 0.141 | 0.608 | 0.0783 |
| LBW | 3,480,316 | 7.004% | 0.132 | 0.652 | 0.0636 |

There are exactly three fitted models, one per target. Each uses 9 raw `landmark_strict` features and 39 transformed features after preprocessing.

These moderate discrimination values do not invalidate the availability analysis: the main estimand concerns how candidate-set restriction changes population capture for the **same ranking** under controlled evaluation protocols.

---

## 9. Interpretation guardrails

### Supported formulations

Use:

- `early-entry eligibility`;
- `any prenatal-care eligibility`;
- `retrospective all-record upper bound`;
- `candidate set`;
- `eligible population`;
- `same fitted model`;
- `same CDC 2023 prediction vector`;
- `same absolute selection/referral budget`;
- `population event capture`;
- `recall among eligible`;
- `event availability ceiling`;
- `CDC 2023 temporal test`.

### Do not imply

Do not write that:

- the all-record upper bound is a deployable early policy;
- an “early model” and a “full model” were compared;
- CDC 2023 is external validation;
- `recall_among_eligible` is population recall;
- the common threshold is a clinically validated risk cutoff;
- prenatal-care timing causally changes the outcome;
- the 5% or 10% budget is clinically optimal;
- LightGBM is the optimal model architecture;
- the observed capture gain establishes downstream clinical benefit.

The model is the same across availability scenarios. Differences arise from eligibility, selection protocol, and—in protocols with unequal selected counts—selection capacity.

---

## 10. Main-text-ready result summary

A concise canonical result statement is:

> In the CDC 2023 temporal test, early-entry eligibility included approximately 74.7% of target-specific records but only 69.4–71.8% of outcome events. Holding the fitted LightGBM ranking and absolute selection budget fixed, expanding the candidate set from early entry to the retrospective all-record upper bound increased population event capture by 2.405 percentage points for preterm delivery (95% CI 2.300–2.511; +7,287 events), 2.322 points for NICU admission (2.222–2.431; +7,041), and 2.931 points for low birth weight (2.799–3.078; +7,145). In contrast, cohort-specific top-10% comparisons yielded larger apparent gains of 5.588–6.771 points because the larger candidate set also received approximately 88,000 additional selections. Exact same-model decomposition attributed the remainder to this extra capacity.

A shorter methodological takeaway is:

> When eligibility is selective, applying the same nominal top-q to differently sized cohorts changes both the candidate set and the absolute number selected. Fixed-budget evaluation separates these effects.

---

## 11. Evidence hierarchy for the paper

### Primary

- `landmark_strict`;
- CDC 2023 temporal test;
- singleton primary population;
- 10% `fixed_absolute_budget`;
- population event capture;
- early-entry → all-record contrast;
- paired bootstrap uncertainty.

### Supporting

- `any_prenatal_care` intermediate availability scenario;
- structural event-availability ceilings;
- exact top-q availability/capacity decomposition;
- protocol comparison.

### Robustness / appendix

- 5% fixed-budget results;
- full bootstrap tables;
- calibration-threshold details;
- model diagnostics;
- legacy `booking_strict` results;
- all-birth sensitivity;
- retrospective oracle analyses.

---

## 12. Required downstream cleanup

Before manuscript numbers are copied from legacy documents:

1. update `claim_sheet.md` from `booking_strict` to `landmark_strict`;
2. replace the old 10% values (+2.58, +2.73, +3.43 pp) with the TAE canonical values (+2.405, +2.322, +2.931 pp);
3. update canonical evidence-file references to the TAE outputs;
4. keep the old values only in `legacy_results.md` or clearly labeled article-revision history.

No model retraining is required for this documentation update.
