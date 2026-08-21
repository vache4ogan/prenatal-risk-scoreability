# TAE 2026 canonical analysis summary

- Feature set: `landmark_strict` (9 features).
- One LightGBM per target; no cohort-specific refitting.
- CDC 2023 is temporal test only.
- Availability scenarios: early-entry, any prenatal care, retrospective all-record upper bound.
- Protocols: cohort-specific top-q, fixed absolute budget, common CDC-2022 threshold.

## Main 10% results

| Target | Protocol | Scenario | Eligible N | Selected N | TP | Precision | Recall among eligible | Population capture |
|---|---|---|---:|---:|---:|---:|---:|---:|
| target_lbw | cohort_specific_top_fraction | all_record_upper_bound | 3,480,316 | 348,032 | 55,650 | 15.990% | 22.828% | 22.828% |
| target_lbw | cohort_specific_top_fraction | any_prenatal_care | 3,338,986 | 333,899 | 51,025 | 15.282% | 22.859% | 20.931% |
| target_lbw | cohort_specific_top_fraction | early_entry | 2,600,093 | 260,009 | 39,145 | 15.055% | 23.142% | 16.058% |
| target_lbw | fixed_absolute_budget | all_record_upper_bound | 3,480,316 | 348,032 | 55,650 | 15.990% | 22.828% | 22.828% |
| target_lbw | fixed_absolute_budget | any_prenatal_care | 3,338,986 | 348,032 | 52,674 | 15.135% | 23.598% | 21.608% |
| target_lbw | fixed_absolute_budget | early_entry | 2,600,093 | 348,032 | 48,505 | 13.937% | 28.675% | 19.897% |
| target_lbw | fixed_calibration_threshold | all_record_upper_bound | 3,480,316 | 355,034 | 56,481 | 15.909% | 23.169% | 23.169% |
| target_lbw | fixed_calibration_threshold | any_prenatal_care | 3,338,986 | 332,032 | 50,823 | 15.307% | 22.768% | 20.848% |
| target_lbw | fixed_calibration_threshold | early_entry | 2,600,093 | 237,973 | 36,522 | 15.347% | 21.591% | 14.982% |
| target_nicu | cohort_specific_top_fraction | all_record_upper_bound | 3,476,584 | 347,658 | 59,201 | 17.029% | 19.520% | 19.520% |
| target_nicu | cohort_specific_top_fraction | any_prenatal_care | 3,335,329 | 333,533 | 54,548 | 16.355% | 19.529% | 17.986% |
| target_nicu | cohort_specific_top_fraction | early_entry | 2,597,274 | 259,727 | 42,252 | 16.268% | 19.728% | 13.931% |
| target_nicu | fixed_absolute_budget | all_record_upper_bound | 3,476,584 | 347,658 | 59,201 | 17.029% | 19.520% | 19.520% |
| target_nicu | fixed_absolute_budget | any_prenatal_care | 3,335,329 | 347,658 | 56,161 | 16.154% | 20.107% | 18.518% |
| target_nicu | fixed_absolute_budget | early_entry | 2,597,274 | 347,658 | 52,160 | 15.003% | 24.354% | 17.198% |
| target_nicu | fixed_calibration_threshold | all_record_upper_bound | 3,476,584 | 361,714 | 61,011 | 16.867% | 20.117% | 20.117% |
| target_nicu | fixed_calibration_threshold | any_prenatal_care | 3,335,329 | 337,404 | 55,021 | 16.307% | 19.698% | 18.142% |
| target_nicu | fixed_calibration_threshold | early_entry | 2,597,274 | 255,166 | 41,734 | 16.356% | 19.486% | 13.761% |
| target_preterm | cohort_specific_top_fraction | all_record_upper_bound | 3,480,382 | 348,038 | 61,503 | 17.671% | 20.298% | 20.298% |
| target_preterm | cohort_specific_top_fraction | any_prenatal_care | 3,340,358 | 334,036 | 55,801 | 16.705% | 20.224% | 18.416% |
| target_preterm | cohort_specific_top_fraction | early_entry | 2,601,060 | 260,106 | 44,326 | 17.042% | 20.367% | 14.629% |
| target_preterm | fixed_absolute_budget | all_record_upper_bound | 3,480,382 | 348,038 | 61,503 | 17.671% | 20.298% | 20.298% |
| target_preterm | fixed_absolute_budget | any_prenatal_care | 3,340,358 | 348,038 | 57,413 | 16.496% | 20.808% | 18.948% |
| target_preterm | fixed_absolute_budget | early_entry | 2,601,060 | 348,038 | 54,216 | 15.578% | 24.912% | 17.893% |
| target_preterm | fixed_calibration_threshold | all_record_upper_bound | 3,480,382 | 361,709 | 63,145 | 17.457% | 20.840% | 20.840% |
| target_preterm | fixed_calibration_threshold | any_prenatal_care | 3,340,358 | 334,755 | 55,894 | 16.697% | 20.258% | 18.447% |
| target_preterm | fixed_calibration_threshold | early_entry | 2,601,060 | 245,487 | 42,615 | 17.359% | 19.581% | 14.064% |

## Early-entry → all-record comparison at 10%

| Target | Protocol | Δ selected | Δ TP | Δ population capture |
|---|---|---:|---:|---:|
| target_lbw | cohort_specific_top_fraction | +88,023 | +16,505 | +6.771 pp |
| target_lbw | fixed_absolute_budget | +0 | +7,145 | +2.931 pp |
| target_lbw | fixed_calibration_threshold | +117,061 | +19,959 | +8.187 pp |
| target_nicu | cohort_specific_top_fraction | +87,931 | +16,949 | +5.588 pp |
| target_nicu | fixed_absolute_budget | +0 | +7,041 | +2.322 pp |
| target_nicu | fixed_calibration_threshold | +106,548 | +19,277 | +6.356 pp |
| target_preterm | cohort_specific_top_fraction | +87,932 | +17,177 | +5.669 pp |
| target_preterm | fixed_absolute_budget | +0 | +7,287 | +2.405 pp |
| target_preterm | fixed_calibration_threshold | +116,222 | +20,530 | +6.776 pp |

## Same-model decomposition at 10%

| Target | Contrast | Naive top-q | Availability | Extra capacity | Identity error |
|---|---|---:|---:|---:|---:|
| target_lbw | early_entry → all_record_upper_bound | +6.771 pp | +2.931 pp | +3.840 pp | 0.000e+00 |
| target_nicu | early_entry → all_record_upper_bound | +5.588 pp | +2.322 pp | +3.267 pp | 0.000e+00 |
| target_preterm | early_entry → all_record_upper_bound | +5.669 pp | +2.405 pp | +3.264 pp | 0.000e+00 |

## Bootstrap: fixed-budget early-entry → all-record at 10%

| Target | Point Δ capture | 95% CI | Point Δ TP |
|---|---:|---:|---:|
| target_lbw | +2.931 pp | [+2.799, +3.078] pp | +7,145 |
| target_nicu | +2.322 pp | [+2.222, +2.431] pp | +7,041 |
| target_preterm | +2.405 pp | [+2.300, +2.511] pp | +7,287 |

Interpretation guardrail: the fitted model is identical across availability scenarios; differences are not differences between separate 'early' and 'late' predictors.
