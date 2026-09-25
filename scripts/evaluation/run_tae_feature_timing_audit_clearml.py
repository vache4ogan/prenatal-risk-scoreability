#!/usr/bin/env python3
"""
TAE 2026 feature-timing audit v2 for the conservative landmark feature set.

Purpose
-------
This is a standalone ClearML audit, not a model-training experiment.

It validates and documents the nine frozen canonical model features:

    mother_race
    mother_height_inches
    mother_bmi
    mother_weight_pre
    prior_live_births
    prior_dead_births
    prior_terminations
    diab_pre
    hyper_pre

For CDC 2022 and CDC 2023, it computes feature missingness within the same
target-specific primary populations used by the canonical analysis and within:

    early_entry             prenatal_care_month 1..3
    any_prenatal_care       prenatal_care_month 1..10
    all_record_upper_bound  all target-specific primary-population records

It outputs:
    feature_missingness_long.csv
    feature_missingness_summary_2023.csv
    feature_timing_audit_v2.csv
    feature_timing_audit_v2.md
    legacy_excluded_feature_timing.csv
    feature_audit_manifest.json

Scientific interpretation
-------------------------
The audit separates three concepts:

1. Semantic timing:
   what time period the CDC variable refers to (e.g. pre-pregnancy).

2. Source/collection route:
   Mother's Worksheet, prenatal-care record via Facility Worksheet,
   medical record via Facility Worksheet, or NCHS-derived value.

3. Public-use fixation:
   the value appears in the natality birth record/public-use file, which is
   assembled for birth registration. This DOES NOT independently establish
   that every value was present in an EHR by months 1-3.

Therefore the paper should call the empirical scenario "care-entry eligibility",
not verified EHR feature availability.

The script also records revised timing rationales for the three legacy excluded
features:
    mother_age
    marital_status
    mother_educ

No model is fitted.
No prediction is generated.
No Git repository is required for ClearML execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Locked canonical lineage
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_VERSION = "2026-08-24-v1-feature-timing-audit"

DEFAULT_PROJECT_NAME = "pershin-medailab/Vache_Oganisyan/CDC Natality Audit"
DEFAULT_DATASET_ID = "062ba26c0ca24cef99549c2a2ab34e65"
DEFAULT_QUEUE = "6be0e69f7fab49d48bc305ef1fb03a6a"
DEFAULT_TASK_NAME = "TAE 2026 feature timing audit v2"

TARGETS = ("target_preterm", "target_nicu", "target_lbw")
SCENARIOS = (
    "early_entry",
    "any_prenatal_care",
    "all_record_upper_bound",
)

FEATURES = (
    "mother_race",
    "mother_height_inches",
    "mother_bmi",
    "mother_weight_pre",
    "prior_live_births",
    "prior_dead_births",
    "prior_terminations",
    "diab_pre",
    "hyper_pre",
)

EXPECTED_ROWS = {
    2022: 3_676_029,
    2023: 3_605_081,
}

EXPECTED_SHA256 = {
    2022: "c5f1ab62b0e9ac63795e75f6075d2523a3f4c5dc3e95c7aa1f7df6763205a59c",
    2023: "82b38b24d2f795a4a5233ce3a2fcb43b70789339306ba9044bde56f50bb270a8",
}

CONFIGURED_DATA_PATHS = {
    2022: "data/cdc_temporal_harmonized/cdc_natality_2022_harmonized.csv",
    2023: "data/cdc_temporal_harmonized/cdc_natality_2023_harmonized.csv",
}

TAGS = [
    "tae-2026",
    "cdc-natality",
    "feature-audit",
    "timing-audit",
    "conservative-landmark-feature-set",
    "care-entry-eligibility",
    "missingness",
    "standalone-script",
]

REQUIREMENTS = (
    ("clearml", ">=1.16,<3"),
    ("numpy", ">=1.26,<3"),
    ("pandas", ">=2.2,<4"),
)


# ---------------------------------------------------------------------------
# Official CDC/NCHS variable audit
# ---------------------------------------------------------------------------

# Sources were checked against:
# - NCHS User Guide to the 2022 Natality Public Use File.
# - NCHS User Guide to the 2023 Natality Public Use File.
#
# Exact public-use positions are stable for these variables in 2022 and 2023.
#
# Important:
# "public_use_fixation" is deliberately conservative. A semantic pre-pregnancy
# variable can still be recorded into the natality file at birth registration;
# we do not infer verified month-1-to-3 EHR availability from the natality file.

FEATURE_AUDIT = {
    "mother_race": {
        "source_field": "MRACE31",
        "public_use_position": "105-106",
        "cdc_definition": "Mother's Race Recode 31; one or more self-identified race categories recoded to 31 combinations.",
        "semantic_timing": "Pre-existing maternal characteristic; not created by the current pregnancy.",
        "recommended_source": "Reported directly by the mother via the Mother's Worksheet.",
        "public_use_fixation": "Stored in the birth certificate/public-use natality record. The public-use value may be NCHS-imputed when race is not reported.",
        "inclusion_reason": "Retained as a baseline demographic characteristic in the frozen conservative landmark feature set; it is not an outcome or pregnancy-course variable.",
        "timing_decision": "RETAIN_WITH_CAVEAT",
        "important_caveat": "The natality file does not prove that race was present in an early EHR, and MRACE31 can reflect NCHS imputation. A no-race sensitivity is required.",
        "raw_missing_code_harmonized": "99 -> missing",
    },
    "mother_height_inches": {
        "source_field": "M_Ht_In",
        "public_use_position": "280-281",
        "cdc_definition": "Mother's height in total inches; public-use range 30-78, 99 unknown/not stated.",
        "semantic_timing": "Stable maternal physical characteristic, conceptually available before pregnancy and throughout pregnancy.",
        "recommended_source": "Mother's Worksheet.",
        "public_use_fixation": "Stored in the birth certificate/public-use natality record; no public-use timestamp establishes availability specifically by months 1-3.",
        "inclusion_reason": "Retained because height is temporally stable and precedes outcome development.",
        "timing_decision": "RETAIN",
        "important_caveat": "Birth-record collection timing is distinct from verified EHR availability.",
        "raw_missing_code_harmonized": "99 -> missing",
    },
    "mother_bmi": {
        "source_field": "BMI",
        "public_use_position": "283-286",
        "cdc_definition": "Pre-pregnancy body mass index, calculated from pre-pregnancy weight and maternal height.",
        "semantic_timing": "Explicitly pre-pregnancy.",
        "recommended_source": "NCHS-derived from maternal height and pre-pregnancy weight in the natality record.",
        "public_use_fixation": "Derived for the birth/public-use record; it is not an independently timestamped early-pregnancy EHR measurement.",
        "inclusion_reason": "Retained because the construct refers to pre-pregnancy status and uses only baseline maternal measurements.",
        "timing_decision": "RETAIN",
        "important_caveat": "Derived value; its availability depends on the availability of height and pre-pregnancy weight.",
        "raw_missing_code_harmonized": "99.9 -> missing",
    },
    "mother_weight_pre": {
        "source_field": "PWgt_R",
        "public_use_position": "292-294",
        "cdc_definition": "Pre-pregnancy weight recode in pounds; public-use range 075-375, 999 unknown/not stated.",
        "semantic_timing": "Explicitly pre-pregnancy.",
        "recommended_source": "Reported directly by the mother via the Mother's Worksheet.",
        "public_use_fixation": "Stored in the birth certificate/public-use natality record; not independently timestamped to care entry.",
        "inclusion_reason": "Retained because the value refers to the pre-pregnancy state rather than pregnancy-course information.",
        "timing_decision": "RETAIN",
        "important_caveat": "Self-reported birth-record item rather than verified early-EHR availability.",
        "raw_missing_code_harmonized": "999 -> missing",
    },
    "prior_live_births": {
        "source_field": "PRIORLIVE",
        "public_use_position": "171-172",
        "cdc_definition": "Number of children still living from previous live births.",
        "semantic_timing": "Prior reproductive history, but the 'now living' split reflects status at record completion rather than an immutable count at early entry.",
        "recommended_source": "Recommended to be collected from the prenatal care record using the Facility Worksheet.",
        "public_use_fixation": "Stored in the birth certificate/public-use natality record. The exact living/dead split is not timestamped to months 1-3.",
        "inclusion_reason": "Retained in the frozen feature set as prior obstetric history; it does not depend on the current pregnancy outcome.",
        "timing_decision": "RETAIN_WITH_CAVEAT",
        "important_caveat": "A previous child's living status can theoretically change during the current pregnancy; the split PRIORLIVE/PRIORDEAD is therefore less perfectly landmark-anchored than total prior live births.",
        "raw_missing_code_harmonized": "99 -> missing",
    },
    "prior_dead_births": {
        "source_field": "PRIORDEAD",
        "public_use_position": "173-174",
        "cdc_definition": "Number of children dead from previous live births.",
        "semantic_timing": "Prior reproductive history, but the 'now dead' split reflects status at record completion rather than an immutable count at early entry.",
        "recommended_source": "Recommended to be collected from the prenatal care record using the Facility Worksheet.",
        "public_use_fixation": "Stored in the birth certificate/public-use natality record. The exact living/dead split is not timestamped to months 1-3.",
        "inclusion_reason": "Retained in the frozen feature set as prior obstetric history; it does not encode the current infant outcome.",
        "timing_decision": "RETAIN_WITH_CAVEAT",
        "important_caveat": "A previous child's living status can theoretically change during the current pregnancy; interpret as birth-record prior-history information, not verified early-EHR state.",
        "raw_missing_code_harmonized": "99 -> missing",
    },
    "prior_terminations": {
        "source_field": "PRIORTERM",
        "public_use_position": "175-176",
        "cdc_definition": "Number of previous other pregnancy outcomes/terminations; the certificate item covers other outcomes such as spontaneous or induced losses or ectopic pregnancies.",
        "semantic_timing": "Prior pregnancy history preceding the current birth.",
        "recommended_source": "Recommended to be collected from the prenatal care record using the Facility Worksheet as part of total-birth-order/prior-pregnancy information.",
        "public_use_fixation": "Stored in the birth certificate/public-use natality record; the file does not timestamp when this history was first available in care.",
        "inclusion_reason": "Retained because it refers to previous pregnancy outcomes and precedes the current target outcome.",
        "timing_decision": "RETAIN",
        "important_caveat": "The public-use field is a birth-record history item, not a direct audit of an early EHR.",
        "raw_missing_code_harmonized": "99 -> missing",
    },
    "diab_pre": {
        "source_field": "RF_PDIAB",
        "public_use_position": "313",
        "cdc_definition": "Pre-pregnancy diabetes; the certificate specifies diagnosis prior to this pregnancy.",
        "semantic_timing": "Explicitly pre-pregnancy / diagnosed before the current pregnancy.",
        "recommended_source": "Medical record via the Facility Worksheet.",
        "public_use_fixation": "Recorded as a risk-factor checkbox in the birth certificate/public-use natality record; source documentation can come from the medical record.",
        "inclusion_reason": "Retained because the diagnosis explicitly predates the current pregnancy.",
        "timing_decision": "RETAIN",
        "important_caveat": "The natality record does not independently establish the exact date the diagnosis became electronically available to a model.",
        "raw_missing_code_harmonized": "U -> missing; Y -> 1; N -> 0",
    },
    "hyper_pre": {
        "source_field": "RF_PHYPE",
        "public_use_position": "315",
        "cdc_definition": "Pre-pregnancy hypertension; the certificate identifies the prepregnancy category as chronic hypertension.",
        "semantic_timing": "Explicitly pre-pregnancy / chronic.",
        "recommended_source": "Medical record via the Facility Worksheet.",
        "public_use_fixation": "Recorded as a risk-factor checkbox in the birth certificate/public-use natality record; source documentation can come from the medical record.",
        "inclusion_reason": "Retained because the condition explicitly predates the current pregnancy.",
        "timing_decision": "RETAIN",
        "important_caveat": "The natality record does not independently establish the exact date the diagnosis became electronically available to a model.",
        "raw_missing_code_harmonized": "U -> missing; Y -> 1; N -> 0",
    },
}

LEGACY_EXCLUDED = {
    "mother_age": {
        "source_field": "MAGER",
        "public_use_position": "75-76",
        "cdc_definition": "Mother's single years of age, derived from reported month/year of birth; NCHS statistics treat maternal age as age at birth.",
        "semantic_timing": "Maternal date of birth is baseline information, but the public-use age is tied to the birth record rather than recomputed at the early-entry landmark.",
        "reason_excluded": "Excluded from the conservative landmark set to avoid using an age-at-birth public-use value as if it were a timestamped age-at-care-entry variable. This is a landmark-definition choice, not a claim that maternal age is inherently unavailable early.",
        "decision": "EXCLUDE_FROM_CONSERVATIVE_SET",
    },
    "marital_status": {
        "source_field": "DMAR",
        "public_use_position": "120",
        "cdc_definition": "Mother married? The standard certificate asks whether married at birth, conception, or any time between.",
        "semantic_timing": "Can incorporate status occurring after early care entry during the current pregnancy.",
        "reason_excluded": "Excluded because the public-use field spans conception through birth and is not fixed to the early-entry landmark.",
        "decision": "EXCLUDE_FROM_CONSERVATIVE_SET",
    },
    "mother_educ": {
        "source_field": "MEDUC",
        "public_use_position": "124",
        "cdc_definition": "Highest degree or level of school completed at the time of delivery.",
        "semantic_timing": "Explicitly defined at delivery; educational attainment can change during pregnancy.",
        "reason_excluded": "Excluded because the public-use definition is anchored to delivery rather than early care entry.",
        "decision": "EXCLUDE_FROM_CONSERVATIVE_SET",
    },
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> str:
    lines = [
        f"python=={platform.python_version()}",
        f"platform={platform.platform()}",
    ]
    for package, _ in REQUIREMENTS:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT_INSTALLED"
        lines.append(f"{package}=={version}")
    return "\n".join(lines) + "\n"


def git_information() -> dict[str, Any]:
    def run_git(*args: str) -> Optional[str]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=Path.cwd(),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except Exception:
            return None

    status = run_git("status", "--porcelain")
    return {
        "commit_hash": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
        "required_for_remote_execution": False,
    }


def bind_clearml_dataset(dataset_id: str, workers: int) -> Path:
    from clearml import Dataset

    dataset = Dataset.get(dataset_id=dataset_id)
    local_copy = dataset.get_local_copy(
        max_workers=workers,
        raise_on_error=True,
    )
    root = Path(local_copy).resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"ClearML Dataset local copy is not a directory: {root}"
        )
    return root


def resolve_dataset_file(dataset_root: Path, configured: str) -> Path:
    configured_path = Path(configured)

    candidates = [
        dataset_root / configured_path,
        dataset_root / configured_path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = list(dataset_root.rglob(configured_path.name))
    if len(matches) == 1:
        return matches[0].resolve()

    if len(matches) > 1:
        suffix = str(configured_path).replace("\\", "/")
        exact = [
            p for p in matches
            if str(p).replace("\\", "/").endswith(suffix)
        ]
        if len(exact) == 1:
            return exact[0].resolve()
        raise RuntimeError(
            f"Ambiguous Dataset file {configured!r}: "
            f"{[str(p) for p in matches[:10]]}"
        )

    raise FileNotFoundError(
        f"Dataset file {configured!r} was not found under {dataset_root}"
    )


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

def required_columns() -> list[str]:
    return sorted(
        {
            "is_us_resident",
            "is_singleton",
            "prenatal_care_month",
            *TARGETS,
            *FEATURES,
        }
    )


def validate_care_month(values: np.ndarray, label: str) -> None:
    care = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(care)
    observed = care[finite]

    if observed.size == 0:
        raise ValueError(f"{label}: no finite prenatal_care_month values.")

    if np.any(observed < 0) or np.any(observed > 10):
        bad = observed[(observed < 0) | (observed > 10)][:20]
        raise ValueError(
            f"{label}: prenatal_care_month outside harmonized 0..10: {bad.tolist()}"
        )

    if not np.allclose(observed, np.round(observed), atol=0.0, rtol=0.0):
        bad = observed[observed != np.round(observed)][:20]
        raise ValueError(
            f"{label}: prenatal_care_month contains non-integers: {bad.tolist()}"
        )


def validate_harmonized_feature_values(
    frame: pd.DataFrame,
    *,
    year: int,
) -> None:
    """
    Detect accidental use of raw sentinel-coded data instead of harmonized data.
    """
    checks = {
        "mother_race": (1.0, 31.0),
        "mother_height_inches": (30.0, 78.0),
        "mother_bmi": (13.0, 69.9),
        "mother_weight_pre": (75.0, 375.0),
        "prior_live_births": (0.0, 30.0),
        "prior_dead_births": (0.0, 30.0),
        "prior_terminations": (0.0, 30.0),
        "diab_pre": (0.0, 1.0),
        "hyper_pre": (0.0, 1.0),
    }

    for feature, (lo, hi) in checks.items():
        numeric = pd.to_numeric(frame[feature], errors="coerce")
        finite = numeric.dropna().to_numpy(dtype=np.float64)
        if finite.size == 0:
            raise ValueError(
                f"CDC {year}/{feature}: all values are missing after numeric conversion."
            )

        bad = finite[(finite < lo) | (finite > hi)]
        if bad.size:
            raise ValueError(
                f"CDC {year}/{feature}: values outside harmonized range "
                f"[{lo}, {hi}], examples={bad[:20].tolist()}. "
                "This may indicate raw sentinel codes were not harmonized."
            )

        if feature in {"prior_live_births", "prior_dead_births", "prior_terminations"}:
            if not np.allclose(finite, np.round(finite), atol=0.0, rtol=0.0):
                raise ValueError(
                    f"CDC {year}/{feature}: non-integer prior-history values found."
                )

        if feature in {"diab_pre", "hyper_pre"}:
            unique = set(np.unique(finite).tolist())
            if not unique.issubset({0.0, 1.0}):
                raise ValueError(
                    f"CDC {year}/{feature}: expected binary 0/1, got {sorted(unique)}"
                )


def read_year(path: Path, year: int) -> pd.DataFrame:
    columns = required_columns()
    print(f"Reading CDC {year}: {path}")

    frame = pd.read_csv(
        path,
        usecols=columns,
        low_memory=False,
    )

    missing_columns = sorted(set(columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"CDC {year} missing required columns: {missing_columns}"
        )

    if len(frame) != EXPECTED_ROWS[year]:
        raise ValueError(
            f"CDC {year} row count {len(frame):,} != expected {EXPECTED_ROWS[year]:,}"
        )

    for column in ("is_us_resident", "is_singleton"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"CDC {year}/{column}: missing or nonnumeric values."
            )
        unique = set(values.unique().tolist())
        if not unique.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"CDC {year}/{column}: expected 0/1, got {sorted(unique)[:20]}"
            )
        frame[column] = values

    for target in TARGETS:
        values = pd.to_numeric(frame[target], errors="coerce")
        unique = set(values.dropna().unique().tolist())
        if not unique.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"CDC {year}/{target}: expected binary with missing allowed."
            )
        frame[target] = values

    frame["prenatal_care_month"] = pd.to_numeric(
        frame["prenatal_care_month"], errors="coerce"
    )
    validate_care_month(
        frame["prenatal_care_month"].to_numpy(dtype=np.float64),
        f"CDC {year}",
    )

    # Convert features to numeric exactly as the canonical preprocessing expects
    # for harmonized values (mother_race is numeric categorical in the DataFrame).
    for feature in FEATURES:
        frame[feature] = pd.to_numeric(
            frame[feature], errors="coerce"
        )

    validate_harmonized_feature_values(frame, year=year)

    print(
        f"CDC {year}: {len(frame):,} rows, "
        f"{len(frame.columns)} required columns loaded"
    )
    return frame


# ---------------------------------------------------------------------------
# Cohorts and missingness
# ---------------------------------------------------------------------------

def resident_singleton_mask(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["is_us_resident"].eq(1)
        & frame["is_singleton"].eq(1)
    ).to_numpy(dtype=bool)


def scenario_masks(care_month: np.ndarray) -> dict[str, np.ndarray]:
    care = np.asarray(care_month, dtype=np.float64)
    validate_care_month(care, "scenario_masks")

    finite = np.isfinite(care)
    early = finite & (care >= 1) & (care <= 3)
    any_care = finite & (care >= 1) & (care <= 10)
    all_record = np.ones(len(care), dtype=bool)

    if not np.all(~early | any_care):
        raise AssertionError("early_entry is not a subset of any_prenatal_care.")
    if not np.all(~any_care | all_record):
        raise AssertionError("any_prenatal_care is not a subset of all-record.")

    return {
        "early_entry": early,
        "any_prenatal_care": any_care,
        "all_record_upper_bound": all_record,
    }


def missingness_rows_for_year(
    frame: pd.DataFrame,
    *,
    year: int,
) -> list[dict[str, Any]]:
    primary = resident_singleton_mask(frame)
    rows: list[dict[str, Any]] = []

    for target in TARGETS:
        known = frame[target].notna().to_numpy(dtype=bool)
        population_mask = primary & known

        target_frame = frame.loc[
            population_mask,
            [*FEATURES, "prenatal_care_month"],
        ].reset_index(drop=True)

        care = target_frame["prenatal_care_month"].to_numpy(dtype=np.float64)
        masks = scenario_masks(care)

        n_all = len(target_frame)
        if n_all <= 0:
            raise ValueError(f"CDC {year}/{target}: empty primary population.")

        previous_n = -1
        for scenario in SCENARIOS:
            mask = masks[scenario]
            scenario_n = int(mask.sum())

            if scenario_n <= 0:
                raise ValueError(
                    f"CDC {year}/{target}/{scenario}: empty candidate set."
                )

            if previous_n > scenario_n:
                raise AssertionError(
                    f"CDC {year}/{target}: scenario sizes are not nested."
                )
            previous_n = scenario_n

            for feature in FEATURES:
                values = target_frame.loc[mask, feature]
                missing_n = int(values.isna().sum())
                observed_n = int(scenario_n - missing_n)
                missing_fraction = float(missing_n / scenario_n)

                rows.append(
                    {
                        "year": year,
                        "target": target,
                        "scenario": scenario,
                        "feature": feature,
                        "scenario_n": scenario_n,
                        "observed_n": observed_n,
                        "missing_n": missing_n,
                        "missing_fraction": missing_fraction,
                        "missing_percent": 100.0 * missing_fraction,
                    }
                )

    expected = len(TARGETS) * len(SCENARIOS) * len(FEATURES)
    if len(rows) != expected:
        raise AssertionError(
            f"CDC {year}: missingness rows {len(rows)} != expected {expected}"
        )

    return rows


def summarize_2023(
    missingness_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(missingness_rows)
    frame = frame.loc[frame["year"].eq(2023)].copy()

    rows: list[dict[str, Any]] = []

    for feature in FEATURES:
        for scenario in SCENARIOS:
            part = frame.loc[
                frame["feature"].eq(feature)
                & frame["scenario"].eq(scenario)
            ].copy()

            if len(part) != len(TARGETS):
                raise AssertionError(
                    f"2023/{feature}/{scenario}: expected {len(TARGETS)} target rows, "
                    f"got {len(part)}"
                )

            by_target = {
                str(row["target"]): row
                for _, row in part.iterrows()
            }

            rows.append(
                {
                    "feature": feature,
                    "scenario": scenario,
                    "preterm_n": int(by_target["target_preterm"]["scenario_n"]),
                    "preterm_missing_n": int(
                        by_target["target_preterm"]["missing_n"]
                    ),
                    "preterm_missing_percent": float(
                        by_target["target_preterm"]["missing_percent"]
                    ),
                    "nicu_n": int(by_target["target_nicu"]["scenario_n"]),
                    "nicu_missing_n": int(
                        by_target["target_nicu"]["missing_n"]
                    ),
                    "nicu_missing_percent": float(
                        by_target["target_nicu"]["missing_percent"]
                    ),
                    "lbw_n": int(by_target["target_lbw"]["scenario_n"]),
                    "lbw_missing_n": int(
                        by_target["target_lbw"]["missing_n"]
                    ),
                    "lbw_missing_percent": float(
                        by_target["target_lbw"]["missing_percent"]
                    ),
                    "min_missing_percent_across_targets": float(
                        part["missing_percent"].min()
                    ),
                    "max_missing_percent_across_targets": float(
                        part["missing_percent"].max()
                    ),
                    "mean_missing_percent_across_targets": float(
                        part["missing_percent"].mean()
                    ),
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Static + dynamic audit tables
# ---------------------------------------------------------------------------

def static_feature_audit_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        info = FEATURE_AUDIT[feature]
        rows.append(
            {
                "feature": feature,
                **info,
            }
        )
    return rows


def excluded_feature_rows() -> list[dict[str, Any]]:
    return [
        {"feature": feature, **info}
        for feature, info in LEGACY_EXCLUDED.items()
    ]


def feature_audit_with_missingness(
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = pd.DataFrame(summary_rows)
    rows: list[dict[str, Any]] = []

    for feature in FEATURES:
        info = FEATURE_AUDIT[feature]

        row = {
            "feature": feature,
            **info,
        }

        for scenario in SCENARIOS:
            part = summary.loc[
                summary["feature"].eq(feature)
                & summary["scenario"].eq(scenario)
            ]
            if len(part) != 1:
                raise AssertionError(
                    f"Expected one summary row for {feature}/{scenario}."
                )
            s = part.iloc[0]

            prefix = {
                "early_entry": "early",
                "any_prenatal_care": "any_care",
                "all_record_upper_bound": "all_record",
            }[scenario]

            row[f"{prefix}_missing_percent_preterm"] = float(
                s["preterm_missing_percent"]
            )
            row[f"{prefix}_missing_percent_nicu"] = float(
                s["nicu_missing_percent"]
            )
            row[f"{prefix}_missing_percent_lbw"] = float(
                s["lbw_missing_percent"]
            )
            row[f"{prefix}_missing_percent_min"] = float(
                s["min_missing_percent_across_targets"]
            )
            row[f"{prefix}_missing_percent_max"] = float(
                s["max_missing_percent_across_targets"]
            )

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Markdown generator
# ---------------------------------------------------------------------------

def fmt_range(row: pd.Series, prefix: str) -> str:
    lo = float(row[f"{prefix}_missing_percent_min"])
    hi = float(row[f"{prefix}_missing_percent_max"])
    if math.isclose(lo, hi, rel_tol=0.0, abs_tol=0.0005):
        return f"{lo:.3f}%"
    return f"{lo:.3f}–{hi:.3f}%"


def generate_markdown(
    audit_rows: list[dict[str, Any]],
    missingness_long: list[dict[str, Any]],
) -> str:
    audit = pd.DataFrame(audit_rows)
    missing = pd.DataFrame(missingness_long)

    lines = [
        "# Feature Timing Audit v2",
        "",
        "## Purpose",
        "",
        "This audit documents the **conservative landmark feature set** used by the canonical LightGBM models and separates three questions that must not be conflated:",
        "",
        "1. **Semantic timing** — what time period a variable refers to.",
        "2. **Collection/source route** — Mother's Worksheet, prenatal-care record, medical record, or an NCHS-derived value.",
        "3. **Public-use fixation** — when the value exists in the natality birth record/public-use file.",
        "",
        "The empirical availability scenarios in the paper are defined by **care-entry eligibility** from `prenatal_care_month`. They are not a direct audit of whether every model input was present in an EHR by months 1–3.",
        "",
        "The public-use natality record is a birth-registration record. A variable can be semantically pre-pregnancy and still lack a public-use timestamp proving electronic availability at the early-entry landmark.",
        "",
        "## Sources",
        "",
        "- CDC/NCHS, *User Guide to the 2022 Natality Public Use File*.",
        "- CDC/NCHS, *User Guide to the 2023 Natality Public Use File*.",
        "- Relevant sections: file layout for maternal race, prior pregnancy history, maternal height/BMI/pre-pregnancy weight, and risk factors; detailed technical notes on race, live-birth order/parity, maternal behavior and health characteristics, and risk factors; U.S. Standard Certificate of Live Birth worksheets reproduced in the User Guide.",
        "",
        "## Conservative landmark feature set",
        "",
        "| Feature | CDC field | Semantic timing | Source / collection route | Decision | Key timing caveat | 2023 early missingness* | 2023 any-care missingness* | 2023 all-record missingness* |",
        "|---|---|---|---|---|---|---:|---:|---:|",
    ]

    for _, row in audit.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['feature']}`",
                    f"`{row['source_field']}`",
                    str(row["semantic_timing"]).replace("|", "/"),
                    str(row["recommended_source"]).replace("|", "/"),
                    str(row["timing_decision"]),
                    str(row["important_caveat"]).replace("|", "/"),
                    fmt_range(row, "early"),
                    fmt_range(row, "any_care"),
                    fmt_range(row, "all_record"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "\\* Missingness is computed in each **target-specific U.S.-resident singleton population with that target observed**. Because the three target populations differ slightly, the compact table reports the range across preterm, NICU, and LBW. Exact target-specific counts and percentages are in `feature_missingness_long.csv`.",
            "",
            "## Variable-by-variable audit",
            "",
        ]
    )

    for _, row in audit.iterrows():
        lines.extend(
            [
                f"### `{row['feature']}`",
                "",
                f"- **CDC field / position:** `{row['source_field']}`, {row['public_use_position']}.",
                f"- **CDC definition:** {row['cdc_definition']}",
                f"- **Semantic timing:** {row['semantic_timing']}",
                f"- **Recommended source:** {row['recommended_source']}",
                f"- **Public-use fixation:** {row['public_use_fixation']}",
                f"- **Inclusion rationale:** {row['inclusion_reason']}",
                f"- **Decision:** `{row['timing_decision']}`.",
                f"- **Caveat:** {row['important_caveat']}",
                f"- **Harmonized missing code:** {row['raw_missing_code_harmonized']}.",
                "",
            ]
        )

    lines.extend(
        [
            "## Revised audit of the three legacy excluded variables",
            "",
            "These fields are **not** model predictors in the conservative landmark feature set. The rationale below supersedes the older shorthand that treated all three as simply 'late' variables.",
            "",
            "| Legacy field | CDC field | Revised timing interpretation | Reason for exclusion |",
            "|---|---|---|---|",
        ]
    )

    for feature, info in LEGACY_EXCLUDED.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{feature}`",
                    f"`{info['source_field']}`",
                    str(info["semantic_timing"]).replace("|", "/"),
                    str(info["reason_excluded"]).replace("|", "/"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation for the manuscript",
            "",
            "The nine-feature set is best described as a **conservative landmark feature set**, not as a set of inputs whose month-1-to-3 EHR availability has been directly observed.",
            "",
            "The strongest timing anchors are the explicitly pre-pregnancy variables (`mother_bmi`, `mother_weight_pre`, `diab_pre`, `hyper_pre`) and prior pregnancy history. Maternal height is temporally stable. Maternal race is a baseline characteristic but the public-use value may be NCHS-imputed. `prior_live_births` and `prior_dead_births` are prior-history variables with a narrower caveat: the living/dead split is defined at record completion and could theoretically change during the current pregnancy.",
            "",
            "Accordingly, the empirical scenarios should be described as **care-entry eligibility**. The separate no-race sensitivity addresses the strongest source/imputation concern for `mother_race`.",
            "",
            "## Missingness details",
            "",
            "Exact 2023 target-specific missingness:",
            "",
        ]
    )

    # Compact exact table, one row per feature/scenario.
    m23 = missing.loc[missing["year"].eq(2023)].copy()
    for scenario in SCENARIOS:
        lines.extend(
            [
                f"### `{scenario}`",
                "",
                "| Feature | Preterm | NICU | LBW |",
                "|---|---:|---:|---:|",
            ]
        )

        for feature in FEATURES:
            part = m23.loc[
                m23["scenario"].eq(scenario)
                & m23["feature"].eq(feature)
            ]
            by_target = {
                str(r["target"]): r
                for _, r in part.iterrows()
            }

            def val(target: str) -> str:
                r = by_target[target]
                return (
                    f"{int(r['missing_n']):,}/"
                    f"{int(r['scenario_n']):,} "
                    f"({float(r['missing_percent']):.3f}%)"
                )

            lines.append(
                f"| `{feature}` | "
                f"{val('target_preterm')} | "
                f"{val('target_nicu')} | "
                f"{val('target_lbw')} |"
            )

        lines.append("")

    lines.extend(
        [
            "## Preprocessing relationship",
            "",
            "The audit reports missingness **before model imputation**. In the canonical pipeline, numeric features are median-imputed and standardized; `mother_race` is most-frequent-imputed and one-hot encoded with unknown-category handling.",
            "",
            "## Paper-ready wording",
            "",
            "> We used a conservative nine-feature landmark set comprising maternal race, height, pre-pregnancy BMI and weight, prior pregnancy history, and pre-pregnancy diabetes and hypertension. CDC definitions anchor BMI, pre-pregnancy weight, diabetes, and hypertension to the pre-pregnancy period; prior-history variables refer to pregnancies preceding the index birth, and height is temporally stable. The natality file is nevertheless a birth-registration record and does not timestamp EHR availability at care entry. We therefore define empirical availability through prenatal-care entry and report feature missingness separately across early-entry, any-care, and all-record candidate sets.",
            "",
            "For `mother_race`, the manuscript should additionally state that the public-use value can be NCHS-imputed and point to the prespecified no-race sensitivity.",
            "",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ClearML upload
# ---------------------------------------------------------------------------

def upload_outputs(task: Any, output: Path) -> None:
    for filename in (
        "feature_missingness_long.csv",
        "feature_missingness_summary_2023.csv",
        "feature_timing_audit_v2.csv",
        "feature_timing_audit_v2.md",
        "legacy_excluded_feature_timing.csv",
        "feature_audit_manifest.json",
        "package_versions.txt",
        "commands.txt",
        "git_info.json",
    ):
        path = output / filename
        if path.is_file():
            task.upload_artifact(
                name=filename,
                artifact_object=str(path),
                wait_on_upload=True,
            )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test() -> int:
    rng = np.random.default_rng(20260824)
    n = 10_000

    frame = pd.DataFrame(
        {
            "is_us_resident": np.ones(n, dtype=np.int8),
            "is_singleton": np.ones(n, dtype=np.int8),
            "prenatal_care_month": rng.choice(
                [1.0, 2.0, 3.0, 4.0, 7.0, 10.0, 0.0, np.nan],
                size=n,
                p=[0.22, 0.20, 0.18, 0.10, 0.08, 0.06, 0.08, 0.08],
            ),
            "target_preterm": rng.binomial(1, 0.09, n).astype(float),
            "target_nicu": rng.binomial(1, 0.08, n).astype(float),
            "target_lbw": rng.binomial(1, 0.07, n).astype(float),
            "mother_race": rng.integers(1, 32, n).astype(float),
            "mother_height_inches": rng.normal(64, 3, n),
            "mother_bmi": np.clip(rng.normal(27, 6, n), 13, 60),
            "mother_weight_pre": np.clip(rng.normal(155, 35, n), 75, 300),
            "prior_live_births": rng.integers(0, 5, n).astype(float),
            "prior_dead_births": rng.integers(0, 2, n).astype(float),
            "prior_terminations": rng.integers(0, 4, n).astype(float),
            "diab_pre": rng.binomial(1, 0.03, n).astype(float),
            "hyper_pre": rng.binomial(1, 0.04, n).astype(float),
        }
    )

    # Inject feature missingness.
    for i, feature in enumerate(FEATURES):
        missing = rng.random(n) < (0.002 + i * 0.0005)
        frame.loc[missing, feature] = np.nan

    rows = missingness_rows_for_year(frame, year=2023)
    if len(rows) != len(TARGETS) * len(SCENARIOS) * len(FEATURES):
        raise AssertionError("Smoke missingness row count failed.")

    summary = summarize_2023(rows)
    if len(summary) != len(FEATURES) * len(SCENARIOS):
        raise AssertionError("Smoke summary row count failed.")

    audit = feature_audit_with_missingness(summary)
    if len(audit) != len(FEATURES):
        raise AssertionError("Smoke audit row count failed.")

    md = generate_markdown(audit, rows)
    if "conservative landmark feature set" not in md:
        raise AssertionError("Smoke markdown generation failed.")

    print("FEATURE TIMING AUDIT SMOKE TEST PASS")
    print("cohort_missingness=PASS")
    print("scenario_nesting=PASS")
    print("audit_schema=PASS")
    print("markdown_generation=PASS")
    return 0


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TAE 2026 standalone ClearML feature timing audit v2"
    )
    parser.add_argument(
        "--dataset-id",
        default=os.getenv("CLEARML_DATASET_ID") or DEFAULT_DATASET_ID,
    )
    parser.add_argument(
        "--queue",
        default=(
            os.getenv("CLEARML_QUEUE_NAME")
            or os.getenv("CLEARML_QUEUE_ID")
            or DEFAULT_QUEUE
        ),
    )
    parser.add_argument(
        "--project-name",
        default=os.getenv("CLEARML_PROJECT_NAME") or DEFAULT_PROJECT_NAME,
    )
    parser.add_argument(
        "--task-name",
        default=(
            os.getenv("CLEARML_TASK_NAME_TAE_FEATURE_AUDIT")
            or DEFAULT_TASK_NAME
        ),
    )
    parser.add_argument("--dataset-download-workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="results/tae_2026_feature_timing_audit",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Run on current machine while still obtaining the Dataset from ClearML. "
            "Normal run should omit --local."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.smoke_test:
        return smoke_test()

    if str(args.dataset_id) != DEFAULT_DATASET_ID:
        raise ValueError(
            f"Feature audit is locked to Dataset {DEFAULT_DATASET_ID}; "
            f"got {args.dataset_id}."
        )
    if str(args.queue) != DEFAULT_QUEUE:
        raise ValueError(
            f"Feature audit remote run is locked to queue {DEFAULT_QUEUE}; "
            f"got {args.queue}."
        )

    from clearml import Task

    Task.force_store_standalone_script(True)

    task = Task.init(
        project_name=args.project_name,
        task_name=args.task_name,
        task_type=Task.TaskTypes.testing,
        tags=TAGS,
        reuse_last_task_id=False,
        output_uri=True,
    )

    runtime = task.connect(
        {
            "dataset_id": str(args.dataset_id),
            "queue": str(args.queue),
            "dataset_download_workers": int(args.dataset_download_workers),
            "output_dir": str(args.output_dir),
        },
        name="runtime",
    )

    # ClearML can return mapping-like values; normalize explicitly.
    args.dataset_id = str(runtime["dataset_id"])
    args.queue = str(runtime["queue"])
    args.dataset_download_workers = int(runtime["dataset_download_workers"])
    args.output_dir = str(runtime["output_dir"])

    if args.dataset_id != DEFAULT_DATASET_ID:
        raise ValueError("Connected runtime changed locked Dataset ID.")
    if args.queue != DEFAULT_QUEUE:
        raise ValueError("Connected runtime changed locked queue.")

    if not args.local:
        task.execute_remotely(
            queue_name=args.queue,
            clone=False,
            exit_process=True,
        )

    started = time.time()
    print(f"Feature audit ClearML Task ID: {task.id}")
    print(f"Dataset ID: {args.dataset_id}")
    print(f"Queue: {args.queue}")

    dataset_root = bind_clearml_dataset(
        args.dataset_id,
        args.dataset_download_workers,
    )

    paths = {
        year: resolve_dataset_file(
            dataset_root,
            CONFIGURED_DATA_PATHS[year],
        )
        for year in (2022, 2023)
    }

    observed_hashes: dict[int, str] = {}
    for year, path in paths.items():
        print(f"Hashing CDC {year}...")
        digest = sha256_file(path)
        expected = EXPECTED_SHA256[year]
        if digest != expected:
            raise AssertionError(
                f"CDC {year} SHA256 {digest} != expected {expected}"
            )
        observed_hashes[year] = digest

    output = Path(args.output_dir).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output

    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{output} is not empty. Use --overwrite."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    atomic_text(
        output / "commands.txt",
        " ".join(
            shlex.quote(str(v))
            for v in [sys.executable, *sys.argv]
        )
        + "\n",
    )
    atomic_text(
        output / "package_versions.txt",
        package_versions(),
    )
    atomic_json(
        output / "git_info.json",
        git_information(),
    )

    all_missingness: list[dict[str, Any]] = []

    for year in (2022, 2023):
        frame = read_year(paths[year], year)
        rows = missingness_rows_for_year(frame, year=year)
        all_missingness.extend(rows)
        del frame

    expected_long = (
        2 * len(TARGETS) * len(SCENARIOS) * len(FEATURES)
    )
    if len(all_missingness) != expected_long:
        raise AssertionError(
            f"Long missingness row count {len(all_missingness)} "
            f"!= expected {expected_long}"
        )

    summary_2023 = summarize_2023(all_missingness)
    audit_rows = feature_audit_with_missingness(summary_2023)
    excluded_rows = excluded_feature_rows()

    atomic_csv(
        output / "feature_missingness_long.csv",
        all_missingness,
    )
    atomic_csv(
        output / "feature_missingness_summary_2023.csv",
        summary_2023,
    )
    atomic_csv(
        output / "feature_timing_audit_v2.csv",
        audit_rows,
    )
    atomic_csv(
        output / "legacy_excluded_feature_timing.csv",
        excluded_rows,
    )
    atomic_text(
        output / "feature_timing_audit_v2.md",
        generate_markdown(audit_rows, all_missingness),
    )

    invariants = {
        "dataset_2022_hash_match": observed_hashes[2022] == EXPECTED_SHA256[2022],
        "dataset_2023_hash_match": observed_hashes[2023] == EXPECTED_SHA256[2023],
        "exact_nine_features": tuple(FEATURE_AUDIT.keys()) == FEATURES,
        "target_specific_primary_population": True,
        "early_subset_any_subset_all": True,
        "missingness_before_model_imputation": True,
        "all_three_scenarios_reported": True,
        "all_three_targets_reported": True,
        "both_2022_and_2023_reported": True,
        "legacy_excluded_fields_not_model_features": not (
            set(LEGACY_EXCLUDED) & set(FEATURES)
        ),
        "prenatal_care_month_not_model_feature": (
            "prenatal_care_month" not in FEATURES
        ),
    }

    if not all(invariants.values()):
        failed = [k for k, v in invariants.items() if not v]
        raise AssertionError(
            f"Feature audit invariants failed: {failed}"
        )

    manifest = {
        "status": "PASS",
        "script_version": SCRIPT_VERSION,
        "elapsed_seconds": float(time.time() - started),
        "clearml_task_id": task.id,
        "clearml_dataset_id": args.dataset_id,
        "clearml_queue": args.queue,
        "input_sha256": {
            "2022": observed_hashes[2022],
            "2023": observed_hashes[2023],
        },
        "design": {
            "paper_feature_set_name": "conservative landmark feature set",
            "internal_feature_set_name": "landmark_strict",
            "features": list(FEATURES),
            "targets": list(TARGETS),
            "scenarios": list(SCENARIOS),
            "missingness_population": (
                "target-specific U.S.-resident singleton births "
                "with corresponding target observed"
            ),
            "missingness_stage": "harmonized values before model imputation",
            "empirical_availability_construct": "care-entry eligibility",
            "verified_early_ehr_availability_claim": False,
        },
        "outputs": {
            "feature_missingness_long_rows": len(all_missingness),
            "feature_missingness_summary_2023_rows": len(summary_2023),
            "feature_timing_audit_rows": len(audit_rows),
            "legacy_excluded_rows": len(excluded_rows),
        },
        "invariants": invariants,
        "git": git_information(),
        "script_sha256": sha256_file(SCRIPT_PATH),
    }

    atomic_json(
        output / "feature_audit_manifest.json",
        manifest,
    )

    output_hashes: dict[str, str] = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "feature_audit_manifest.json":
            output_hashes[path.name] = sha256_file(path)

    manifest["output_sha256"] = output_hashes
    atomic_json(
        output / "feature_audit_manifest.json",
        manifest,
    )

    upload_outputs(task, output)
    task.close()

    print("TAE FEATURE TIMING AUDIT PASS")
    print(f"Output: {output}")
    print(f"Rows: {len(all_missingness)} missingness rows")
    print("Markdown: feature_timing_audit_v2.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
