# Target Definitions and Feature Audit (CDC Natality 2022-2023)

## 1. Official CDC Definition of PRIORTERM
**Field Name:** `PRIORTERM` (Prior Other Outcomes / Spontaneous or Induced Losses)
**CDC Official Definition:** The number of prior other pregnancy outcomes, which includes both spontaneous losses (miscarriages, stillbirths) and induced terminations. 
**Source:** CDC Natality Public Use File User Guide (2022/2023), Section: Maternal Characteristics / Prior Pregnancy History.
**Valid Values:** `0` to `30` (Number of terminations), `99` (Unknown/Not Stated).
**Handling:** Values of `99` or blank were mapped to missing (NaN) and imputed via median imputation during preprocessing, or treated natively by LightGBM's missing value handler.

## 2. Target Formulation and Data Dictionaries

### A. Preterm Delivery (Преждевременные роды)
* **Target Logic:** Binary target `1` if Gestational Age < 37 weeks, `0` otherwise.
* **CDC Field Used:** `OEGEST_COMB` (Obstetric Estimate of Gestation) or harmonized `ESTGEST`.
* **COMBGEST vs Obstetric Estimate Comparison:** 
  The CDC provides two primary gestational age measures: `COMBGEST` (based primarily on Last Menstrual Period - LMP) and the Obstetric Estimate (OE). Following CDC standard recommendations since 2014, we strictly use the Obstetric Estimate (`OEGEST_COMB`) for our target because LMP is subject to recall bias and delayed ovulation errors.
* **Missingness:** Records with `99` (Unknown) in the OE field were explicitly dropped from the primary target-specific population.

### B. NICU Admission (Реанимация новорожденных)
* **Target Logic:** Binary target `1` if admitted to NICU, `0` otherwise.
* **CDC Field Used:** `UCA_NICU` (Admission to NICU).
* **Valid Values:** `Y` (Yes), `N` (No), `U` (Unknown), Blank.
* **Handling:** Mapped `Y` -> 1, `N` -> 0. `U` and blanks were explicitly excluded from the target-specific denominator to prevent false negatives.

### C. Low Birth Weight (Низкий вес при рождении)
* **Target Logic:** Binary target `1` if birth weight < 2500 grams, `0` otherwise.
* **CDC Field Used:** `DBWT` (Birth Weight in Grams).
* **Valid Values:** `0227`–`8165` grams, `9999` (Unknown/Not Stated).
* **Handling:** Records with `9999` were excluded from the target-specific denominator.

## 3. Environment & Reproducibility
* **Code Version:** Locked at Git Tag `tae-2026-locked-replication-v1`
* **Python Environment:** Python 3.9+, LightGBM 4.x
* **Random Seeds:** 2026 (Model Initialization/Splitting), 2027 (Bootstrap Resampling)
* **Bootstrap Specs:** 500 paired, full-size record-level replicates