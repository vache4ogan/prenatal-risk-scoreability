# Article-revision analysis summary

All analyses use LightGBM and `booking_strict` features. CDC 2023 is described as a temporal test.

## Primary singleton fixed-model equal-budget result

| Evaluation | Target | Budget | Early capture | Full capture | Full − early | Additional TP |
|---|---|---:|---:|---:|---:|---:|
| calibration_evaluation_2022 | target_lbw | 5% | 0.1211 | 0.1397 | +0.0186 | 464 |
| calibration_evaluation_2022 | target_lbw | 10% | 0.2050 | 0.2362 | +0.0312 | 777 |
| calibration_evaluation_2022 | target_nicu | 5% | 0.1074 | 0.1222 | +0.0148 | 445 |
| calibration_evaluation_2022 | target_nicu | 10% | 0.1790 | 0.2049 | +0.0259 | 779 |
| calibration_evaluation_2022 | target_preterm | 5% | 0.1149 | 0.1282 | +0.0133 | 408 |
| calibration_evaluation_2022 | target_preterm | 10% | 0.1876 | 0.2135 | +0.0260 | 795 |
| temporal_test_2023 | target_lbw | 5% | 0.1234 | 0.1422 | +0.0188 | 4,585 |
| temporal_test_2023 | target_lbw | 10% | 0.2056 | 0.2398 | +0.0343 | 8,356 |
| temporal_test_2023 | target_nicu | 5% | 0.1079 | 0.1229 | +0.0151 | 4,570 |
| temporal_test_2023 | target_nicu | 10% | 0.1777 | 0.2050 | +0.0273 | 8,271 |
| temporal_test_2023 | target_preterm | 5% | 0.1149 | 0.1286 | +0.0138 | 4,167 |
| temporal_test_2023 | target_preterm | 10% | 0.1883 | 0.2141 | +0.0258 | 7,821 |

## Paired full-size bootstrap

| Evaluation | Target | Budget | Point difference | 95% CI | Additional TP (point) |
|---|---|---:|---:|---:|---:|
| calibration_evaluation_2022 | target_lbw | 5% | +0.0186 | [+0.0153, +0.0219] | 464 |
| calibration_evaluation_2022 | target_lbw | 10% | +0.0312 | [+0.0265, +0.0362] | 777 |
| calibration_evaluation_2022 | target_nicu | 5% | +0.0148 | [+0.0124, +0.0178] | 445 |
| calibration_evaluation_2022 | target_nicu | 10% | +0.0259 | [+0.0229, +0.0295] | 779 |
| calibration_evaluation_2022 | target_preterm | 5% | +0.0133 | [+0.0105, +0.0160] | 408 |
| calibration_evaluation_2022 | target_preterm | 10% | +0.0260 | [+0.0226, +0.0298] | 795 |
| temporal_test_2023 | target_lbw | 5% | +0.0188 | [+0.0177, +0.0199] | 4,585 |
| temporal_test_2023 | target_lbw | 10% | +0.0343 | [+0.0328, +0.0357] | 8,356 |
| temporal_test_2023 | target_nicu | 5% | +0.0151 | [+0.0143, +0.0159] | 4,570 |
| temporal_test_2023 | target_nicu | 10% | +0.0273 | [+0.0262, +0.0284] | 8,271 |
| temporal_test_2023 | target_preterm | 5% | +0.0138 | [+0.0129, +0.0146] | 4,167 |
| temporal_test_2023 | target_preterm | 10% | +0.0258 | [+0.0247, +0.0270] | 7,821 |

## Mechanism at the 10% budget

| Target | Care-entry group | Group events | Additional selected (full − early) | Additional TP (full − early) |
|---|---|---:|---:|---:|
| target_lbw | after_month_3 | 54,066 | 96,119 | 15,254 |
| target_lbw | months_1_3 | 169,152 | -122,118 | -13,501 |
| target_lbw | no_prenatal_care | 12,870 | 16,118 | 4,390 |
| target_lbw | unknown | 7,688 | 9,881 | 2,213 |
| target_nicu | after_month_3 | 65,142 | 81,040 | 13,876 |
| target_nicu | months_1_3 | 214,174 | -105,197 | -12,286 |
| target_nicu | no_prenatal_care | 15,194 | 14,109 | 4,393 |
| target_nicu | unknown | 8,775 | 10,048 | 2,288 |
| target_preterm | after_month_3 | 58,282 | 90,332 | 14,343 |
| target_preterm | months_1_3 | 217,632 | -116,965 | -14,450 |
| target_preterm | no_prenatal_care | 17,387 | 16,243 | 5,330 |
| target_preterm | unknown | 9,700 | 10,390 | 2,598 |

## Primary LightGBM calibration

| Evaluation | Scenario | Target | Raw Brier | Calibrated Brier | Raw BSS | Calibrated BSS | Raw intercept/slope | Calibrated intercept/slope |
|---|---|---|---:|---:|---:|---:|---:|---:|
| calibration_evaluation_2022 | early_entry | target_lbw | 0.058965 | 0.058965 | 0.0262 | 0.0262 | -0.070/0.988 | -0.083/0.982 |
| calibration_evaluation_2022 | early_entry | target_nicu | 0.071865 | 0.071872 | 0.0190 | 0.0189 | -0.068/0.990 | -0.144/0.956 |
| calibration_evaluation_2022 | early_entry | target_preterm | 0.074069 | 0.074070 | 0.0222 | 0.0222 | -0.029/0.997 | -0.064/0.981 |
| calibration_evaluation_2022 | full_cohort | target_lbw | 0.063302 | 0.063303 | 0.0285 | 0.0285 | -0.014/0.995 | -0.026/0.989 |
| calibration_evaluation_2022 | full_cohort | target_nicu | 0.075877 | 0.075877 | 0.0207 | 0.0207 | +0.032/1.012 | -0.045/0.978 |
| calibration_evaluation_2022 | full_cohort | target_preterm | 0.076918 | 0.076919 | 0.0236 | 0.0236 | +0.017/1.011 | -0.018/0.995 |
| temporal_test_2023 | early_entry | target_lbw | 0.059214 | 0.059215 | 0.0265 | 0.0265 | -0.073/0.989 | -0.085/0.983 |
| temporal_test_2023 | early_entry | target_nicu | 0.074188 | 0.074194 | 0.0195 | 0.0194 | -0.070/0.977 | -0.145/0.944 |
| temporal_test_2023 | early_entry | target_preterm | 0.074948 | 0.074950 | 0.0225 | 0.0224 | -0.052/0.985 | -0.086/0.969 |
| temporal_test_2023 | full_cohort | target_lbw | 0.063265 | 0.063265 | 0.0288 | 0.0288 | -0.019/0.997 | -0.032/0.992 |
| temporal_test_2023 | full_cohort | target_nicu | 0.077968 | 0.077969 | 0.0208 | 0.0208 | +0.015/0.995 | -0.061/0.962 |
| temporal_test_2023 | full_cohort | target_preterm | 0.077603 | 0.077604 | 0.0236 | 0.0236 | -0.011/0.998 | -0.046/0.982 |

## All-birth sensitivity

| Target | Budget | Early capture | Full capture | Full − early | Additional TP |
|---|---:|---:|---:|---:|---:|
| target_lbw | 5% | 0.1125 | 0.1280 | +0.0155 | 4,779 |
| target_lbw | 10% | 0.1901 | 0.2186 | +0.0285 | 8,797 |
| target_nicu | 5% | 0.1041 | 0.1180 | +0.0139 | 4,874 |
| target_nicu | 10% | 0.1729 | 0.1980 | +0.0251 | 8,821 |
| target_preterm | 5% | 0.1093 | 0.1208 | +0.0115 | 4,293 |
| target_preterm | 10% | 0.1818 | 0.2036 | +0.0218 | 8,159 |

Notes:

- `full precision` and `early precision` are compared at the same absolute budget.
- `recall_among_eligible` is conditional on scoreability and is not population event capture.
- The all-birth analysis is sensitivity only because multiple infant records may be dependent and no pregnancy-level identifier is available.
- Platt scaling is monotone when its fitted slope is positive; the script verifies this and checks that top-B ranking is unchanged.
