# Hypothesis Verdict

## Canonical article-version verdict

**Study question.** Does restricting prenatal risk scoring to records with prenatal-care entry during months 1–3 reduce population-level event capture after removing differences caused by cohort-specific model training, cohort-specific thresholds, and unequal absolute referral counts?

**Primary verdict: supported.** In the corrected fixed-model, equal-absolute-budget analysis, the full-cohort policy captured more population events than the early-entry policy for preterm delivery, NICU admission, and low birth weight on the CDC 2023 temporal test. At a referral budget equal to 10% of the full target-specific population, the gains were **+2.58 percentage points for preterm, +2.73 points for NICU, and +3.43 points for low birth weight**. These differences corresponded to **7,821, 8,271, and 8,356 additional captured events**, respectively.

Paired full-size bootstrap with 500 replicates produced 95% confidence intervals entirely above zero for all three outcomes. The same direction was reproduced on held-out CDC 2022. Mechanism analysis showed that the gain arose from access to events among records with prenatal-care entry after month 3, no prenatal care, or unknown entry timing. The effect was not explained by a larger referral budget, cohort-specific refitting, or substantial miscalibration of the primary LightGBM models.

The main contribution is not a new model architecture. It is the distinction between:

1. **scoreability ceiling** — the maximum fraction of all events located inside the scoreable cohort;
2. **conditional model recall** — recall among records eligible for scoring;
3. **population event capture** — the fraction of all population events captured under a fixed absolute resource budget.

---

## 1. Analysis scope

### Data

The analysis used harmonized U.S. CDC Natality data:

- CDC 2022: 3,676,029 records;
- CDC 2023: 3,605,081 records.

CDC 2022 was used for model development and internal held-out analyses. CDC 2023 was used only as a **temporal test**. It was not used for model training, Platt-calibrator fitting, threshold selection, or hyperparameter optimization.

### Primary population

The primary analysis included U.S.-resident singleton births with a known target value.

| Outcome | Full CDC 2023 population | Early-entry eligible | Eligible fraction |
|---|---:|---:|---:|
| Preterm delivery | 3,480,382 | 2,601,060 | 74.735% |
| NICU admission | 3,476,584 | 2,597,274 | 74.708% |
| Low birth weight | 3,480,316 | 2,600,093 | 74.709% |

### Outcomes

- `target_preterm`
- `target_nicu`
- `target_lbw`

### Primary model

The primary article analysis used one LightGBM model per outcome with `booking_strict` features. For each outcome:

- one model was trained on CDC 2022;
- the same fitted model generated predictions for early-entry and full-cohort policies;
- no early-specific or full-specific model was fitted;
- no class weighting or oversampling was used;
- no hyperparameter optimization was performed.

The LightGBM configuration used 350 trees, `learning_rate=0.05`, `num_leaves=31`, and `min_child_samples=100`.

### Eligibility policies

- **Early-entry:** prenatal care began during months 1–3.
- **Full-cohort:** all records in the target-specific primary population were eligible.

Records with entry after month 3, no prenatal care, or unknown entry timing remained in the full-population denominator but were unavailable to the early-entry policy.

---

## 2. Why the corrected design was required

The original factorial comparison mixed the access effect with:

1. separately trained early-entry and full-cohort models;
2. separately selected thresholds;
3. top-5% and top-10% computed from cohorts of different sizes.

The corrected design used one fixed model, one common prediction vector, and the same absolute number of selected records in both policies. Top-B was reconstructed separately inside each eligible set.

Therefore, the corrected comparison isolates the operational effect of **who can be considered for selection** under the same referral capacity.

---

## 3. Access ceiling before classifier performance

The early-entry event ceiling is the fraction of all target events occurring inside the early-entry cohort.

| Outcome | CDC 2022 event ceiling | CDC 2023 event ceiling |
|---|---:|---:|
| Preterm delivery | 72.039% | 71.826% |
| NICU admission | 70.778% | 70.618% |
| Low birth weight | 69.667% | 69.388% |

Although about 74.7% of CDC 2023 records were early-entry eligible, only 69.4–71.8% of events occurred inside that scoreable cohort. The unavailable quarter of the population contained a disproportionate share of adverse outcomes.

Even a hypothetical perfect classifier operating only inside the early-entry cohort could not capture events outside it.

---

## 4. Primary fixed-model equal-budget result

The absolute referral budget was defined as 5% or 10% of the full target-specific evaluation population. Early-entry and full-cohort selected exactly the same number of records.

### CDC 2023 temporal test

| Outcome | Budget | Early capture | Full capture | Full − early | Additional events |
|---|---:|---:|---:|---:|---:|
| Preterm delivery | 5% | 11.49% | 12.86% | +1.38 pp | 4,167 |
| Preterm delivery | 10% | 18.83% | 21.41% | **+2.58 pp** | **7,821** |
| NICU admission | 5% | 10.79% | 12.29% | +1.51 pp | 4,570 |
| NICU admission | 10% | 17.77% | 20.50% | **+2.73 pp** | **8,271** |
| Low birth weight | 5% | 12.34% | 14.22% | +1.88 pp | 4,585 |
| Low birth weight | 10% | 20.56% | 23.98% | **+3.43 pp** | **8,356** |

The direction was identical for all three outcomes and both budgets.

### Precision and conditional recall

At the 10% budget:

| Outcome | Early precision | Full precision |
|---|---:|---:|
| Preterm delivery | 16.390% | 18.637% |
| NICU admission | 15.503% | 17.882% |
| Low birth weight | 14.399% | 16.800% |

Full-cohort precision was higher in every primary comparison. Early-entry may have higher recall among eligible records because its denominator excludes events outside the scoreable cohort. Conditional recall and population event capture must therefore be reported separately.

---

## 5. Paired bootstrap uncertainty

The uncertainty analysis used a paired, full-size, record-level nonparametric bootstrap:

- 500 replicates;
- bootstrap size equal to the full evaluation size;
- identical bootstrap multiplicities for early and full;
- top-B recomputed separately in every replicate;
- difference defined as full capture minus early capture.

### CDC 2023, 10% budget

| Outcome | Point difference | 95% paired-bootstrap CI | Additional events |
|---|---:|---:|---:|
| Preterm delivery | **+2.581 pp** | **[+2.47, +2.70] pp** | 7,821 |
| NICU admission | **+2.727 pp** | **[+2.62, +2.84] pp** | 8,271 |
| Low birth weight | **+3.428 pp** | **[+3.28, +3.57] pp** | 8,356 |

All intervals were entirely above zero.

### Held-out CDC 2022, 10% budget

| Outcome | Point difference | 95% paired-bootstrap CI |
|---|---:|---:|
| Preterm delivery | +2.60 pp | [+2.26, +2.98] pp |
| NICU admission | +2.59 pp | [+2.29, +2.95] pp |
| Low birth weight | +3.12 pp | [+2.65, +3.62] pp |

The effect reproduced in direction and approximate magnitude across held-out CDC 2022 and temporal-test CDC 2023.

The new intervals supersede older intervals calculated on arbitrary 50,000-record subsamples or from shared-threshold analyses with unequal realized referral counts.

---

## 6. Mechanism of the effect

At the 10% budget, CDC 2023 records were divided into:

1. months 1–3;
2. after month 3;
3. no prenatal care;
4. unknown entry timing.

The full-cohort policy did not receive a larger budget. It reallocated part of the same budget toward groups unavailable to early-entry.

| Outcome | Additional TP outside early cohort | TP reduction in months 1–3 | Net additional TP |
|---|---:|---:|---:|
| Preterm delivery | +22,271 | −14,450 | **+7,821** |
| NICU admission | +20,557 | −12,286 | **+8,271** |
| Low birth weight | +21,857 | −13,501 | **+8,356** |

Detailed contributions:

| Outcome | After month 3 | No prenatal care | Unknown timing | Months 1–3 change | Net |
|---|---:|---:|---:|---:|---:|
| Preterm delivery | +14,343 | +5,330 | +2,598 | −14,450 | **+7,821** |
| NICU admission | +13,876 | +4,393 | +2,288 | −12,286 | **+8,271** |
| Low birth weight | +15,254 | +4,390 | +2,213 | −13,501 | **+8,356** |

The true positives gained in later-entry, no-care, and unknown groups exceeded those lost inside the early-entry group.

---

## 7. Calibration check

A separate Platt calibrator was fitted for each `booking_strict` LightGBM model using CDC 2022 only. Calibration was evaluated on held-out CDC 2022 and CDC 2023, separately for early and full populations.

### CDC 2023 full cohort

| Outcome | Raw Brier | Calibrated Brier | Raw intercept | Raw slope |
|---|---:|---:|---:|---:|
| Preterm delivery | 0.077603 | 0.077604 | −0.011 | 0.998 |
| NICU admission | 0.077968 | 0.077969 | +0.015 | 0.995 |
| Low birth weight | 0.063265 | 0.063265 | −0.019 | 0.997 |

The raw models were already close to ideal calibration. Platt scaling did not improve Brier score and sometimes produced a microscopic worsening. The primary analysis should therefore use raw unweighted LightGBM probabilities.

The availability effect is not explained by substantial model miscalibration.

---

## 8. Corrected all-birth sensitivity

The singleton restriction was removed and the same fixed-model equal-budget design was repeated for all U.S.-resident births.

### CDC 2023, 10% budget

| Outcome | Early capture | Full capture | Full − early | Additional events |
|---|---:|---:|---:|---:|
| Preterm delivery | 18.18% | 20.36% | **+2.18 pp** | 8,159 |
| NICU admission | 17.29% | 19.80% | **+2.51 pp** | 8,821 |
| Low birth weight | 19.01% | 21.86% | **+2.85 pp** | 8,797 |

The direction was preserved for all outcomes.

Because multiple infant records from one pregnancy may be dependent and no reliable pregnancy-level identifier is available, singleton results remain primary and all-birth results remain sensitivity evidence.

---

## 9. Secondary pregnancy-oracle analysis

`pregnancy_oracle` added gestational diabetes, gestational hypertension, and eclampsia.

### LightGBM discrimination on CDC 2023 full cohort

| Outcome | Booking PR-AUC | Oracle PR-AUC | Difference |
|---|---:|---:|---:|
| Preterm delivery | 0.1537 | 0.1782 | +0.0246 |
| NICU admission | 0.1483 | 0.1641 | +0.0158 |
| Low birth weight | 0.1399 | 0.1599 | +0.0201 |

ROC-AUC increased from 0.6299 to 0.6608 for preterm, 0.6231 to 0.6451 for NICU, and 0.6683 to 0.6950 for low birth weight. Brier score decreased by approximately 0.00068–0.00109.

At the 10% fixed budget:

| Outcome | Oracle early capture | Oracle full capture | Full − early |
|---|---:|---:|---:|
| Preterm delivery | 22.229% | 24.927% | +2.698 pp |
| NICU admission | 20.273% | 23.111% | +2.838 pp |
| Low birth weight | 23.471% | 27.313% | +3.842 pp |

These results indicate additional retrospective predictive information. They do not establish valid real-time availability. `pregnancy_oracle` is not a deployable early clinical model and belongs in a secondary analysis or appendix.

---

## 10. Shared-threshold analysis

A supplementary analysis selected one threshold on CDC 2022 and applied it unchanged to early and full CDC 2023 populations.

This answers a different question: what happens when the same decision rule is deployed across populations with different scoreability?

It does not hold the realized referral count constant. The full population selects more records, so its capture differences are larger. Older verdict tables reporting approximately +5.5 to +8.5 percentage points at “10% calibrated capacity” correspond to this unequal-realized-budget design and must not be used as the primary hypothesis test.

---

## 11. Reproducibility

The final ClearML run completed with status `PASS`.

- Dataset ID: `062ba26c0ca24cef99549c2a2ab34e65`
- Task ID: `29717840f05144579af240f6ded4852a`

Completed computation:

- 3 singleton LightGBM models;
- 3 all-birth LightGBM models;
- 3 Platt calibrators;
- 6 CDC 2023 prediction passes;
- 6,000 bootstrap replicate rows;
- 12 bootstrap summary rows;
- 480 reliability-curve rows.

The run stored the standalone script, locked configuration, command, package versions, input hashes, output hashes, result CSVs, serialized models, and run manifest.


---

## 12. Supported claims

1. **Early scoreability imposes a structural ceiling.** Supported by eligibility near 74.7% but event ceilings of only 69.4–71.8%.
2. **At the same absolute budget, full eligibility increases population capture.** Supported by 3/3 positive corrected comparisons, positive results at 5% and 10%, paired bootstrap, and held-out CDC 2022 replication.
3. **The gain comes from events outside the early cohort.** Supported by the mechanism table.
4. **The effect is not a calibration artifact.** Supported by near-ideal raw calibration and no Platt benefit.
5. **The direction persists when multiple births are included.** Supported by all-birth sensitivity.
6. **Pregnancy-period variables add retrospective predictive information.** Supported by higher PR-AUC/ROC-AUC and lower Brier, with timing caveats.

---

## 13. What the analysis does not establish

The analysis does not demonstrate that:

- LightGBM is the optimal architecture;
- 5% or 10% is the clinically optimal referral budget;
- deployment would improve clinical outcomes;
- prenatal-care entry causally changes outcomes;
- currently unscoreable records can be scored without new data collection;
- `pregnancy_oracle` is a valid early clinical model;
- findings automatically generalize beyond CDC Natality or the United States;
- all-birth uncertainty would be unchanged under pregnancy-clustered bootstrap;
- class balancing improves training, because the primary analysis did not compare balanced and unbalanced training.

---

## 14. Final verdict

The hypothesis is supported.

Restricting prenatal risk scoring to records with prenatal-care entry during months 1–3 creates a population-level performance loss that remains after controlling for model identity, prediction set, and absolute referral capacity.

At a 10% absolute budget on CDC 2023, full-cohort eligibility increased population event capture by **2.58 percentage points for preterm delivery, 2.73 points for NICU admission, and 3.43 points for low birth weight**, corresponding to **7,821, 8,271, and 8,356 additional captured events**.

Paired full-size bootstrap confirmed that all three differences were above zero. Mechanism analysis showed that the gains arose from events among records unavailable to early scoring. The result reproduced on held-out CDC 2022, persisted in all-birth sensitivity, and was not attributable to substantial model miscalibration.

Clinical triage studies should therefore report scoreability ceiling, conditional performance among eligible records, and population event capture under an explicitly defined absolute resource budget.
