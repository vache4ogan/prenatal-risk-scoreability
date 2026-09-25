# All-birth Sensitivity Analysis Design

## Purpose

Test whether the primary singleton-birth conclusions remain stable after including twins and higher-order multiple births.

## Prespecified change from the primary analysis

- Primary analysis: U.S.-resident singleton births.
- Sensitivity analysis: all U.S.-resident births.
- No change to targets, feature sets, early-entry definition, split seed, calibration fraction, capacities, thresholds, metrics, bootstrap logic, or CDC 2022-to-2023 temporal design.
- Logistic Regression max_iter: 2000.

## Interpretation caveat

Multiple infants from one pregnancy may create dependent records. The available harmonized birth-record data do not contain a reliable pregnancy-level cluster identifier, so this analysis is descriptive sensitivity evidence rather than a replacement for the singleton primary analysis.
