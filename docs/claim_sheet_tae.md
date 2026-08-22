# Claim Sheet — TAE 2026 Canonical

## Paper working title

**Who Can Be Scored? Cohort Availability Limits Population-Level Capture in Prenatal Risk Triage**

## Evidence status

This claim sheet supersedes the earlier article-revision / `booking_strict` claim sheet for the TAE canonical paper.

The numerical source of truth is the current `landmark_strict` canonical run:

- `canonical_protocol_results.csv`
- `canonical_protocol_comparisons.csv`
- `bootstrap_summary.csv`
- `effect_decomposition.csv`
- `cohort_counts_validation.csv`
- `model_diagnostics.csv`
- `thresholds_2022.csv`
- `tae_canonical_summary.md`

Legacy article-revision values must not be used as current canonical results.

---

## Research question

When clinical risk scoring is selectively available, how much population-level event capture is lost because the candidate set is restricted, and how much of the apparent difference under conventional cohort-specific top-q evaluation is instead caused by a change in absolute selection capacity?

The empirical setting is prenatal risk triage in CDC Natality 2022→2023.

---

## Primary estimand

For each outcome and nominal budget:

**all-record upper-bound population event capture − early-entry population event capture**

under `fixed_absolute_budget`.

Both scenarios use:

- the same fitted LightGBM model;
- the same `landmark_strict` feature set;
- the same CDC 2023 prediction vector;
- the same absolute number of selected records;
- separate deterministic top-B selection within each eligible candidate set.

Population event capture is:

**TP / all target events in the target-specific CDC 2023 population.**

This is distinct from `recall_among_eligible`, whose denominator is only events inside the eligible cohort.

---

## Primary population and model

- Population: target-specific U.S.-resident singleton births with the outcome known.
- Development data: CDC Natality 2022.
- Temporal test: CDC Natality 2023.
- Model: one LightGBM per target; no cohort-specific refitting.
- Feature set: `landmark_strict`.
- Number of raw features: 9.
- Outcomes:
  - preterm delivery;
  - NICU admission;
  - low birth weight.
- Nominal fractions: 5% and 10%.
- Main-text budget: 10%.
- Bootstrap: paired full-size record-level nonparametric bootstrap, 500 replicates, 95% intervals.

Availability scenarios:

- `early_entry`: prenatal-care entry in months 1–3;
- `any_prenatal_care`: prenatal-care entry in months 1–10;
- `all_record_upper_bound`: all records in the target-specific primary population.

`all_record_upper_bound` is a **retrospective evaluation counterfactual**, not a deployable early clinical policy.

---

## Primary claim

**Restricting scoring to early-entry records reduces population event capture even when the fitted model, prediction vector, and absolute selection capacity are held fixed.**

### Canonical CDC 2023 result at the 10% fixed absolute budget

| Outcome | Early capture | All-record capture | All − early | 95% paired-bootstrap CI | Additional captured events | Selected in both scenarios |
|---|---:|---:|---:|---:|---:|---:|
| Preterm delivery | 17.893% | 20.298% | +2.405 pp | [2.300, 2.511] pp | 7,287 | 348,038 |
| NICU admission | 17.198% | 19.520% | +2.322 pp | [2.222, 2.431] pp | 7,041 | 347,658 |
| Low birth weight | 19.897% | 22.828% | +2.931 pp | [2.799, 3.078] pp | 7,145 | 348,032 |

All three paired-bootstrap intervals are above zero.

---

## Structural availability claim

**Selective eligibility imposes a population-level ceiling before classifier ranking errors are considered.**

| Outcome | Early records / all | Early events / all | Any-care records / all | Any-care events / all |
|---|---:|---:|---:|---:|
| Preterm delivery | 74.735% | 71.826% | 95.977% | 91.060% |
| NICU admission | 74.708% | 70.618% | 95.937% | 92.097% |
| Low birth weight | 74.709% | 69.388% | 95.939% | 91.567% |

Early-entry eligibility includes about 74.7% of records but only 69.4–71.8% of outcome events.

Any-prenatal-care eligibility includes about 95.9–96.0% of records but only 91.1–92.1% of events.

Therefore, some target events remain structurally outside the candidate set even before the model ranks eligible records.

---

## Intermediate availability claim

**Expanding availability from early entry to any prenatal care recovers part, but not all, of the fixed-budget early-to-all-record gap.**

At the 10% fixed absolute budget:

| Outcome | Early → any prenatal care | Any prenatal care → all-record |
|---|---|---|
| Preterm delivery | +1.055 pp (3,197 TP), 95% CI [0.956, 1.144] | +1.350 pp (4,090 TP), 95% CI [1.293, 1.412] |
| NICU admission | +1.319 pp (4,001 TP), 95% CI [1.229, 1.418] | +1.002 pp (3,040 TP), 95% CI [0.954, 1.063] |
| Low birth weight | +1.710 pp (4,169 TP), 95% CI [1.596, 1.829] | +1.221 pp (2,976 TP), 95% CI [1.156, 1.288] |

This supports the narrower claim that later prenatal-care availability recovers some population capture, while the `no_care + unknown_care` portion remains outside the `any_prenatal_care` candidate set.

It does **not** establish a causal effect of prenatal-care timing on outcomes.

---

## Evaluation-protocol claim

**Using the same nominal top-q across differently sized eligible cohorts changes both the candidate set and the absolute number selected.**

At nominal 10%, early-entry → all-record:

| Outcome | Cohort-specific top-q | Fixed absolute budget | Fixed calibration threshold |
|---|---|---|---|
| Preterm delivery | +5.669 pp; Δ selected 87,932 | +2.405 pp; Δ selected 0 | +6.776 pp; Δ selected 116,222 |
| NICU admission | +5.588 pp; Δ selected 87,931 | +2.322 pp; Δ selected 0 | +6.356 pp; Δ selected 106,548 |
| Low birth weight | +6.771 pp; Δ selected 88,023 | +2.931 pp; Δ selected 0 | +8.187 pp; Δ selected 117,061 |

Interpretation:

- `cohort_specific_top_fraction` fixes the fraction selected within each cohort, not absolute capacity;
- `fixed_absolute_budget` fixes absolute capacity and is the primary resource-controlled comparison;
- `fixed_calibration_threshold` fixes the CDC-2022 score decision rule, not realized selection count.

Therefore, the three protocols answer different operational questions and should not be interpreted as interchangeable estimates of one effect.

---

## Exact same-model decomposition claim

For the early-entry → all-record comparison,

**naive cohort-specific top-q difference = availability component + extra-capacity component.**

At 10%:

| Outcome | Naive top-q | Availability component | Extra-capacity component | Identity error |
|---|---:|---:|---:|---:|
| Preterm delivery | +5.669 pp | +2.405 pp | +3.264 pp | 0.0e+00 |
| NICU admission | +5.588 pp | +2.322 pp | +3.267 pp | 0.0e+00 |
| Low birth weight | +6.771 pp | +2.931 pp | +3.840 pp | 0.0e+00 |

The identity error is exactly zero for all three outcomes.

This is the main methodological claim:

> Cohort-specific top-q evaluation can overstate the apparent benefit of broader availability because the larger eligible cohort also receives more selection slots.

The fixed-budget component isolates candidate-set expansion at the larger common absolute budget; the remaining difference is attributable to additional selection capacity.

---

## 5% robustness claim

**The fixed-budget early-entry → all-record effect remains positive at the smaller 5% budget.**

| Outcome | Δ population capture | 95% paired-bootstrap CI | Additional captured events | Selected in both scenarios |
|---|---:|---:|---:|---:|
| Preterm delivery | +1.324 pp | [1.239, 1.404] pp | 4,013 | 174,019 |
| NICU admission | +1.338 pp | [1.262, 1.412] pp | 4,059 | 173,829 |
| Low birth weight | +1.630 pp | [1.533, 1.738] pp | 3,974 | 174,016 |

These results are robustness / appendix evidence, not the main numerical headline.

---

## Supported claims and article placement

| Claim | Canonical evidence | Placement |
|---|---|---|
| Early scoreability creates a structural ceiling before classifier errors. | Early event availability ceiling is 69.4–71.8% despite ~74.7% record availability. | Main text |
| Broader eligibility improves population event capture at the same absolute budget. | Positive fixed-budget early→all differences for all three outcomes at 5% and 10%. | Main text |
| The 10% primary effect is statistically stable under paired resampling. | 500 paired full-size bootstrap replicates; all three 95% CIs above zero. | Main text |
| Any-prenatal-care availability recovers part, but not all, of the gap. | Positive early→any and any→all fixed-budget contrasts for all three outcomes. | Main text / supporting |
| Conventional cohort-specific top-q mixes availability and capacity. | Exact decomposition of naive top-q into availability + extra capacity, identity error 0. | Main text; central methodological result |
| A common threshold answers a different question from a fixed resource budget. | Same CDC-2022 threshold across scenarios but unequal CDC-2023 selected counts. | Main text / appendix |
| Conditional recall is not population capture. | `recall_among_eligible` and `population_event_capture` use different denominators. | Methods + interpretation |

---

## Claims NOT currently supported by the TAE canonical evidence

The current canonical evidence does **not** support using the following as TAE-primary claims without an additional strict9 analysis:

1. **“The effect reproduces on held-out CDC 2022.”**  
   The canonical uploaded results establish CDC 2023 temporal-test effects. A separate strict9 held-out-2022 result is not part of the current canonical evidence bundle.

2. **Care-group mechanism counts from the old article-revision run.**  
   The old `mechanism_table_10pct*` values belong to the previous feature/model generation and must not be presented as strict9 canonical mechanism results.

3. **Calibration conclusions from the old article-revision run.**  
   Legacy `lightgbm_calibration_*` files were produced for the previous generation. Do not claim that strict9 canonical effects are “not explained by miscalibration” unless calibration is re-run or separately validated for the current canonical models.

4. **All-birth sensitivity values from the old article-revision run.**  
   They remain legacy sensitivity evidence unless the all-birth analysis is re-run under the current strict9 canonical design.

5. **Pregnancy-oracle numerical results from the old article-revision run.**  
   They are legacy / retrospective sensitivity evidence and are not part of the current canonical result.

These analyses may remain documented in `legacy_results.md`, but their numerical values must not be mixed into current TAE canonical tables or claims.

---

## Interpretation guardrails

Use:

- “CDC 2023 temporal test,” not “external validation”;
- “early-entry eligibility,” not “early predictor”;
- “any prenatal-care eligibility” or “broader care-access availability scenario,” not “late model”;
- “retrospective all-record upper bound,” not “full policy”;
- “same fitted model” and “same prediction vector”;
- “same absolute selection/referral budget” for Protocol B;
- “population event capture” for TP divided by all target events;
- “recall among eligible” for TP divided by eligible events;
- “candidate set” or “eligible population” when describing availability.

Do not write:

- “the full model is more accurate”;
- “the late model outperforms the early model”;
- “population recall” when referring to `recall_among_eligible`;
- “the common threshold is clinically validated”;
- “10% is the clinically optimal referral budget”;
- “prenatal-care timing causes or prevents the outcome”;
- “the all-record upper bound is currently deployable”;
- “LightGBM is the optimal architecture”;
- “deployment would improve maternal or neonatal outcomes”;
- “results automatically generalize beyond CDC Natality / the United States.”

---

## Claims the paper does not make

The analysis does not establish that:

- all required early-landmark inputs are observable for every person in the all-record scenario;
- currently unscoreable individuals can be scored without additional data collection;
- prenatal-care entry causally changes outcomes;
- fixed-budget evaluation is the only valid evaluation protocol;
- cohort-specific top-q evaluation is intrinsically invalid;
- a fixed calibration threshold is a clinical risk threshold;
- LightGBM is the best possible classifier;
- hyperparameter optimization would not change discrimination;
- the observed event-capture changes imply clinical benefit;
- CDC 2023 constitutes external validation;
- findings automatically generalize outside the U.S. natality setting.

---

## Canonical evidence files

Primary evidence:

- `canonical_protocol_results.csv`
- `canonical_protocol_comparisons.csv`
- `bootstrap_summary.csv`
- `bootstrap_replicates.csv`
- `effect_decomposition.csv`
- `cohort_counts_validation.csv`
- `model_diagnostics.csv`
- `thresholds_2022.csv`
- `tae_canonical_summary.md`

Design / reproducibility evidence:

- `experiment_config.locked.yaml`
- `feature_sets_tae.locked.json`
- `run_manifest.json`
- `package_versions.txt`
- `commands.txt`
- `git_info.json`

Legacy article-revision files are not canonical evidence for current strict9 numerical claims.

---

## Decision

**Primary hypothesis: supported under the current TAE canonical estimand.**

The CDC 2023 temporal test shows that selective candidate-set availability limits population-level event capture beyond classifier ranking error. Under a fixed absolute budget, expanding eligibility from early entry to the retrospective all-record upper bound increases capture by approximately 2.3–2.9 percentage points across all three outcomes.

The canonical protocol comparison further shows that conventional cohort-specific top-q evaluation yields substantially larger apparent differences because the larger candidate set also receives additional selection capacity. The exact same-model decomposition separates these two components with zero identity error.
