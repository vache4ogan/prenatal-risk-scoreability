# Data and cohorts

## Temporal design

CDC Natality 2022 is the model-development dataset. CDC Natality 2023 is the temporal test; it is not external validation. The primary population is U.S.-resident singleton births. The all-birth population is a sensitivity analysis only because multiple infant records can be dependent and no pregnancy-level identifier is available.

Raw source artifacts found in the parent tree were `data/cdc2022/nat2022us.zip`, `data/cdc2022_checked/cdc_natality_2022_official.csv`, `data/cdc2022_checked/cdc_natality_2022_official_with_plurality.csv`, `data/cdc2023/Nat2023us.zip`, and `data/cdc2023/cdc_natality_2023_audit_ready.csv`. None are committed here.

## Preprocessing order

1. Extract fixed-width source records with `scripts/preprocessing/extractor.py`.
2. Add or audit 2022 plurality with `add_plurality_cdc2022.py` and `audit_cdc_2022_raw_coordinates.py`.
3. Harmonize 2022 and 2023 with `preprocess_harmonize_cdc_2022_2023.py`.
4. Build timing metadata and cohorts with `build_feature_timing_and_cohorts.py`.

Expected inputs are `data/cdc_temporal_harmonized/cdc_natality_2022_harmonized.csv` (3,676,029 rows; SHA-256 `c5f1ab62b0e9ac63795e75f6075d2523a3f4c5dc3e95c7aa1f7df6763205a59c`) and `data/cdc_temporal_harmonized/cdc_natality_2023_harmonized.csv` (3,605,081 rows; SHA-256 `82b38b24d2f795a4a5233ce3a2fcb43b70789339306ba9044bde56f50bb270a8`). These full datasets are not committed.

Early entry means prenatal-care entry in months 1–3. Target-specific missing outcomes are filtered independently, so target-specific denominators may differ. The same fixed model and absolute target-specific budget are used for early-entry and full-cohort policies.

