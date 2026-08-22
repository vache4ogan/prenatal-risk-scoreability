# TAE Canonical Effect Decomposition

## Status

This document is the authoritative decomposition report for the **TAE 2026 canonical `landmark_strict` analysis**.

It uses only the current canonical outputs:

- `effect_decomposition.csv`
- `canonical_protocol_results.csv`
- `canonical_protocol_comparisons.csv`
- `bootstrap_summary.csv`

The analysis uses one fitted LightGBM model per target and one CDC 2023 prediction vector per target. Availability scenarios therefore do **not** represent separately fitted “early” and “full” models.

---

## 1. Why a decomposition is necessary

Suppose two eligible cohorts have different sizes:

- `early_entry`;
- `all_record_upper_bound`.

A conventional cohort-specific top-q evaluation selects the top q fraction **inside each cohort**.

For target-specific candidate sets $S_{early}$ and $S_{all}$,

$B_{early} = round(q N_{early})$

and

$B_{all} = round(q N_{all})$.

Because $N_{all} > N_{early}$, the all-record scenario receives more absolute selection slots whenever the same q is applied separately.

Therefore, the observed difference in population event capture changes two things at once:

1. **availability / candidate-set composition** — who can be selected;
2. **capacity** — how many records can be selected.

This means a cohort-specific top-q contrast is not a pure fixed-resource availability effect.

---

## 2. Exact decomposition

Let $C(S,B)$ denote population event capture when the candidate set is $S$ and the absolute selection budget is $B$.

For the early-entry → all-record comparison:

### Naive cohort-specific top-q effect

$C(S_{all}, B_{all}) - C(S_{early}, B_{early})$

This is what is observed when each cohort receives its own top-q.

### Availability component

$C(S_{all}, B_{all}) - C(S_{early}, B_{all})$

The absolute budget is held at the larger common budget $B_{all}$. Only the candidate set changes.

This is exactly the canonical `fixed_absolute_budget` early-entry → all-record contrast.

### Extra-capacity component

$C(S_{early}, B_{all}) - C(S_{early}, B_{early})$

The candidate set is held fixed at `early_entry`, while the number selected increases from $B_{early}$ to $B_{all}$.

Therefore:

**naive top-q effect = availability component + extra-capacity component**

The canonical implementation verifies this identity exactly.

---

## 3. Main decomposition at 10%

| Outcome | Naive top-q effect | Availability component | Extra-capacity component | Availability share of naive | Capacity share of naive | Identity error |
|---|---:|---:|---:|---:|---:|---:|
| Preterm delivery | +5.669 pp | +2.405 pp | +3.264 pp | 42.4% | 57.6% | 0.0e+00 |
| NICU admission | +5.588 pp | +2.322 pp | +3.267 pp | 41.5% | 58.5% | 0.0e+00 |
| Low birth weight | +6.771 pp | +2.931 pp | +3.840 pp | 43.3% | 56.7% | 0.0e+00 |

At the nominal 10% level:

- **Preterm:** +5.669 pp naive = +2.405 pp availability + +3.264 pp extra capacity.
- **NICU:** +5.588 pp naive = +2.322 pp availability + +3.267 pp extra capacity.
- **LBW:** +6.771 pp naive = +2.931 pp availability + +3.840 pp extra capacity.

The capacity component accounts for approximately **57–58%** of the naive cohort-specific top-10% contrast across the three outcomes.

Thus, in this setting, more than half of the apparent early-to-all-record improvement under conventional top-10% evaluation comes from giving the larger candidate set additional selection slots.

---

## 4. Where the additional capacity enters

At 10%, cohort-specific top-q produces different absolute numbers selected:

| Outcome | Early top-10% selected | All-record top-10% selected | Additional selections under all-record | Early fixed-budget selected | All-record fixed-budget selected |
|---|---:|---:|---:|---:|---:|
| Preterm delivery | 260,106 | 348,038 | 87,932 | 348,038 | 348,038 |
| NICU admission | 259,727 | 347,658 | 87,931 | 347,658 | 347,658 |
| Low birth weight | 260,009 | 348,032 | 88,023 | 348,032 | 348,032 |

Under `cohort_specific_top_fraction`, the all-record scenario receives roughly **88,000 additional selections**.

Under `fixed_absolute_budget`, the selected count is identical across early-entry and all-record scenarios.

Therefore, the two protocols do not ask the same question:

- cohort-specific top-q: *What happens if each eligible cohort may act on the same fraction of its own members?*
- fixed absolute budget: *What happens if the system has the same number of action slots but can choose from a broader candidate set?*

---

## 5. Consequence for the apparent early → all-record effect

| Outcome | Cohort-specific top-10% capture | Fixed-absolute-budget capture |
|---|---|---|
| Preterm delivery | 14.629% → 20.298% (+5.669 pp) | 17.893% → 20.298% (+2.405 pp) |
| NICU admission | 13.931% → 19.520% (+5.588 pp) | 17.198% → 19.520% (+2.322 pp) |
| Low birth weight | 16.058% → 22.828% (+6.771 pp) | 19.897% → 22.828% (+2.931 pp) |

The cohort-specific top-10% contrast is substantially larger because it combines broader availability with additional absolute capacity.

The fixed-budget result should therefore be used when the scientific question is specifically the population-level consequence of candidate-set restriction **at fixed resource capacity**.

This does not make cohort-specific top-q “wrong.” It answers a different operational question.

---

## 6. Uncertainty for the primary availability component

The availability component is the primary fixed-budget estimand and has paired-bootstrap uncertainty estimates.

| Outcome | Availability component | 95% paired-bootstrap CI | Additional captured events |
|---|---:|---:|---:|
| Preterm delivery | +2.405 pp | [2.300, 2.511] pp | 7,287 |
| NICU admission | +2.322 pp | [2.222, 2.431] pp | 7,041 |
| Low birth weight | +2.931 pp | [2.799, 3.078] pp | 7,145 |

All three confidence intervals are above zero.

The bootstrap pertains to the fixed-budget early-entry → all-record contrast. It should not be presented as uncertainty for the decomposition identity itself: the algebraic identity is exact by construction in the canonical output.

---

## 7. Robustness at 5%

The same qualitative pattern appears at the smaller nominal budget.

| Outcome | Naive top-q effect | Availability component | Extra-capacity component | Availability share of naive | Capacity share of naive | Identity error |
|---|---:|---:|---:|---:|---:|---:|
| Preterm delivery | +3.365 pp | +1.324 pp | +2.041 pp | 39.4% | 60.6% | 0.0e+00 |
| NICU admission | +3.315 pp | +1.338 pp | +1.976 pp | 40.4% | 59.6% | 0.0e+00 |
| Low birth weight | +3.964 pp | +1.630 pp | +2.334 pp | 41.1% | 58.9% | 0.0e+00 |

At 5%:

- **Preterm:** +3.365 pp = +1.324 pp availability + +2.041 pp capacity.
- **NICU:** +3.315 pp = +1.338 pp availability + +1.976 pp capacity.
- **LBW:** +3.964 pp = +1.630 pp availability + +2.334 pp capacity.

The decomposition identity remains exact.

These 5% values are best treated as robustness / appendix evidence, while the 10% decomposition is the main-text result.

---

## 8. Interpretation

The decomposition supports a methodological rather than causal interpretation.

### Supported interpretation

> Applying the same nominal top-q to differently sized eligible cohorts changes both the candidate set and the absolute number selected. In the CDC Natality canonical analysis, the larger all-record candidate set receives substantially more selection slots under cohort-specific top-q, making the apparent early-to-all-record gain larger than the corresponding fixed-budget availability effect.

### Strong concise version

> At 10%, the apparent early-to-all-record gain was 5.6–6.8 percentage points under cohort-specific top-q, but only 2.3–2.9 points when absolute selection capacity was held constant. The remainder was exactly attributable to additional capacity.

### What the decomposition does not establish

It does not establish that:

- prenatal-care timing causally changes outcome risk;
- currently unavailable individuals can be scored in a live early workflow;
- the all-record scenario is deployable;
- a fixed budget is the uniquely correct evaluation protocol;
- cohort-specific top-q is invalid;
- one selection slot necessarily corresponds to one real clinical referral;
- expanding availability would improve clinical outcomes.

“Referral budget” is an operational shorthand for the number of records that can be selected for downstream action. The canonical outputs themselves measure `selected_n`; they do not observe actual clinical referrals.

---

## 9. Terminology guardrails

Prefer:

- `selection budget` or `absolute selection capacity` in methodological text;
- `referral budget` only when explicitly framed as an operational interpretation;
- `candidate set`;
- `early-entry eligibility`;
- `retrospective all-record upper bound`;
- `same fitted model`;
- `same prediction vector`;
- `population event capture`;
- `availability component`;
- `extra-capacity component`.

Avoid:

- `early model` versus `full model`;
- `late predictor`;
- `full policy`;
- `model improvement` when describing the early→all difference;
- `causal effect of prenatal care`;
- claiming that the decomposition proves deployment benefit.

---

## 10. Main-text-ready paragraph

> Conventional top-q evaluation can confound candidate-set availability with resource capacity when eligible cohorts differ in size. In the CDC 2023 temporal test, applying a separate top-10% rule to the early-entry and all-record cohorts increased the absolute number selected by approximately 88,000 records in the all-record scenario. The resulting apparent gain in population event capture was 5.669 percentage points for preterm delivery, 5.588 points for NICU admission, and 6.771 points for low birth weight. Holding the absolute selection budget fixed reduced these contrasts to 2.405, 2.322, and 2.931 points, respectively. Exact same-model decomposition attributed the remaining 3.264, 3.267, and 3.840 points to additional selection capacity, with zero identity error for all outcomes.

---

## 11. Figure implication

The main real-data figure should make the capacity distinction visually explicit.

A suitable design is:

- x-axis: outcome;
- y-axis: early-entry → all-record difference in population event capture;
- one estimate for `cohort_specific_top_fraction`;
- one estimate for `fixed_absolute_budget`;
- optionally one estimate for `fixed_calibration_threshold`;
- annotate or otherwise show the change in `selected_n`.

A complementary decomposition visualization can show, for each outcome:

**naive top-q = availability + capacity**

as two stacked components.

The figure caption must state that all availability scenarios use the **same fitted model and score vector**.

---

## 12. Evidence status

### PRIMARY

- 10% fixed-budget availability component;
- 10% cohort-specific top-q comparison;
- exact 10% availability/capacity decomposition;
- paired-bootstrap CI for the fixed-budget component.

### ROBUSTNESS

- 5% decomposition.

### NOT PART OF THIS DECOMPOSITION

- legacy `booking_strict` mechanism tables;
- legacy calibration analyses;
- legacy all-birth analysis;
- oracle feature analyses.

Those analyses belong to prior generations unless separately reproduced under the current `landmark_strict` canonical design.
