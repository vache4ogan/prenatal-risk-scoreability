#!/usr/bin/env python3
"""
Create the prospective-style feature timing audit and cohort flow tables for
the CDC Natality 2022 -> 2023 experiment.

Inputs:
    data/cdc_temporal_harmonized/cdc_natality_2022_harmonized.csv
    data/cdc_temporal_harmonized/cdc_natality_2023_harmonized.csv

Primary targets:
    target_preterm
    target_nicu
    target_lbw

Primary population:
    U.S.-resident singleton births.

Cohorts:
    full cohort:
        primary population with known target, regardless of care entry;
    early-entry cohort:
        full cohort with prenatal_care_month in 1..early_entry_max_month;
    later-entry cohort:
        full cohort with prenatal_care_month after early_entry_max_month;
    no-care:
        full cohort with prenatal_care_month == 0;
    unknown-care:
        full cohort with prenatal_care_month missing.

Feature sets:
    booking_strict:
        features defensibly available before pregnancy or by the first
        prenatal visit;
    pregnancy_oracle:
        booking_strict plus selected pregnancy-period variables without a
        reliable timestamp. This is a retrospective upper bound, not a late
        clinical prediction model.

Outputs:
    feature_timing.csv
    feature_sets.json
    cohort_flow.csv
    cohort_definitions.json
    feature_cohort_report.md
    feature_cohort_manifest.json

The script streams the full CSVs and never modifies them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


PRIMARY_TARGETS = (
    "target_preterm",
    "target_nicu",
    "target_lbw",
)

TARGET_SOURCE_COLUMNS = {
    "target_preterm": "gestation_oe_weeks",
    "target_nicu": "admit_nicu",
    "target_lbw": "birth_weight_g",
}

TARGET_LABELS = {
    "target_preterm": "Preterm birth (<37 completed OE weeks)",
    "target_nicu": "NICU admission",
    "target_lbw": "Low birth weight (<2500 g)",
}

EXPECTED_ROWS = {
    2022: 3_676_029,
    2023: 3_605_081,
}

EXPECTED_COLUMNS = (
    "birth_year",
    "birth_month",
    "residence_status",
    "mother_age",
    "mother_race",
    "marital_status",
    "mother_educ",
    "wic_benefits",
    "mother_height_inches",
    "mother_bmi",
    "mother_weight_pre",
    "weight_gain",
    "prior_live_births",
    "prior_dead_births",
    "prior_terminations",
    "prenatal_care_month",
    "prenatal_visits",
    "cigarettes_1_tri",
    "diab_pre",
    "diab_gest",
    "hyper_pre",
    "hyper_gest",
    "eclampsia",
    "delivery_method",
    "delivery_method_binary",
    "plurality",
    "child_sex",
    "gestation_combined_weeks",
    "gestation_oe_weeks",
    "gestation_oe_recode3",
    "birth_weight_g",
    "an_vent",
    "admit_nicu",
    "an_seiz",
    "is_us_resident",
    "is_singleton",
    "target_preterm",
    "target_nicu",
    "target_lbw",
    "target_gdm",
)

FEATURE_CANDIDATES = EXPECTED_COLUMNS


@dataclass(frozen=True)
class FeatureDecision:
    feature: str
    timing_class: str
    booking_strict: bool
    pregnancy_oracle: bool
    oracle_incremental: bool
    reliable_timestamp: str
    availability_basis: str
    general_exclusion_reason: str = ""


# Conservative locked timing decisions.
#
# pregnancy_oracle is a SUPERSET of booking_strict. Only selected pregnancy
# variables are added. Final pregnancy totals, care-use variables, delivery
# variables and newborn outcomes are excluded even from the oracle set.
FEATURE_DECISIONS: dict[str, FeatureDecision] = {}


def register(
    feature: str,
    timing_class: str,
    booking_strict: bool,
    pregnancy_oracle: bool,
    oracle_incremental: bool,
    reliable_timestamp: str,
    availability_basis: str,
    general_exclusion_reason: str = "",
) -> None:
    FEATURE_DECISIONS[feature] = FeatureDecision(
        feature=feature,
        timing_class=timing_class,
        booking_strict=booking_strict,
        pregnancy_oracle=pregnancy_oracle,
        oracle_incremental=oracle_incremental,
        reliable_timestamp=reliable_timestamp,
        availability_basis=availability_basis,
        general_exclusion_reason=general_exclusion_reason,
    )


# Time/index variables: not patient predictors.
register(
    "birth_year", "study_design", False, False, False, "not_applicable",
    "Defines temporal train/test year.",
    "Excluded to avoid encoding the 2022/2023 split as a predictor.",
)
register(
    "birth_month", "post_booking_timing", False, False, False, "no",
    "Actual month of birth is known only at delivery.",
    "Actual delivery timing is unavailable at booking.",
)

# Population/cohort variables.
register(
    "residence_status", "cohort_definition", False, False, False, "yes",
    "Used to define U.S.-resident population.",
    "Population filter, not a model feature.",
)
register(
    "plurality", "cohort_definition", False, False, False, "no",
    "Used to restrict the primary analysis to singleton births.",
    "Primary cohort filter; plurality is not guaranteed at first booking.",
)
register(
    "is_us_resident", "cohort_definition", False, False, False, "yes",
    "Derived from residence_status.",
    "Derived population filter, not a model feature.",
)
register(
    "is_singleton", "cohort_definition", False, False, False, "no",
    "Derived from plurality.",
    "Derived primary-cohort filter, not a model feature.",
)
register(
    "prenatal_care_month", "care_entry_definition", False, False, False, "retrospective",
    "Defines when the system could first observe the pregnancy.",
    "Defines early/full/no-care/unknown-care cohorts and must not be used as a predictor.",
)

# Booking-strict features.
for feature, basis in {
    "mother_age": "Maternal age is known at booking.",
    "mother_race": "Maternal race is a pre-existing characteristic known at booking.",
    "marital_status": "Current marital status can be recorded at booking.",
    "mother_educ": "Maternal education can be recorded at booking.",
    "mother_height_inches": "Maternal height can be measured at the first visit.",
    "mother_bmi": "Pre-pregnancy BMI is available from history or first-visit assessment.",
    "mother_weight_pre": "Pre-pregnancy weight is available from history at booking.",
    "prior_live_births": "Prior obstetric history is available at booking.",
    "prior_dead_births": "Prior obstetric history is available at booking.",
    "prior_terminations": "Prior obstetric history is available at booking.",
    "diab_pre": "Pre-pregnancy diabetes is part of baseline medical history.",
    "hyper_pre": "Pre-pregnancy hypertension is part of baseline medical history.",
}.items():
    register(
        feature, "booking_strict", True, True, False, "yes", basis
    )

# Pregnancy-oracle incremental features.
for feature, basis in {
    "diab_gest": (
        "Gestational diabetes is a pregnancy complication with no reliable "
        "diagnosis timestamp in the birth record."
    ),
    "hyper_gest": (
        "Gestational hypertension is a pregnancy complication with no reliable "
        "diagnosis timestamp in the birth record."
    ),
    "eclampsia": (
        "Eclampsia is a pregnancy complication with no reliable onset timestamp "
        "in the birth record."
    ),
}.items():
    register(
        feature, "pregnancy_oracle_increment", False, True, True, "no", basis
    )

# Excluded pregnancy/care-use variables.
register(
    "cigarettes_1_tri", "pregnancy_exposure_unselected", False, False, False, "no",
    "First-trimester smoking is reported retrospectively without timing relative to booking.",
    "Not guaranteed at the first visit and not a pregnancy complication; excluded from the strict complication-only oracle.",
)
register(
    "wic_benefits", "informative_observation", False, False, False, "no",
    "WIC receipt is reported for pregnancy without a reliable booking timestamp.",
    "Potential healthcare-access/informative-observation variable; excluded from both sets.",
)
register(
    "weight_gain", "cumulative_pregnancy_measure", False, False, False, "no",
    "Final pregnancy weight gain is only known retrospectively.",
    "Cumulative post-booking measure; excluded even from the complication oracle.",
)
register(
    "prenatal_visits", "informative_observation", False, False, False, "no",
    "Total number of prenatal visits is only known after pregnancy.",
    "Cumulative care-utilization variable and direct informative-observation pathway.",
)

# Infant characteristic without a defensible booking timestamp.
register(
    "child_sex", "pregnancy_information_unselected", False, False, False, "no",
    "May become known during pregnancy, but not necessarily by booking.",
    "Not a booking feature and not part of the complication-only oracle increment.",
)

# Delivery and newborn outcomes.
for feature, reason in {
    "delivery_method": "Delivery outcome observed at birth.",
    "delivery_method_binary": "Delivery outcome observed at birth.",
    "gestation_combined_weeks": "Final gestational duration observed at birth.",
    "gestation_oe_weeks": "Final OE gestational duration and source of the preterm label.",
    "gestation_oe_recode3": "Direct recode of the final preterm label.",
    "birth_weight_g": "Birth weight is observed only after delivery and is the LBW label source.",
    "an_vent": "Newborn intervention/outcome observed after birth.",
    "admit_nicu": "Newborn outcome and direct source of the NICU label.",
    "an_seiz": "Newborn morbidity observed after birth.",
}.items():
    register(
        feature, "post_outcome", False, False, False, "no",
        reason, "Post-delivery or direct outcome information."
    )

# Targets: never predictors.
for target in ("target_preterm", "target_nicu", "target_lbw", "target_gdm"):
    register(
        target, "target", False, False, False, "not_applicable",
        "Derived outcome label.", "Target labels are never predictors."
    )

if set(FEATURE_DECISIONS) != set(FEATURE_CANDIDATES):
    missing = sorted(set(FEATURE_CANDIDATES) - set(FEATURE_DECISIONS))
    extra = sorted(set(FEATURE_DECISIONS) - set(FEATURE_CANDIDATES))
    raise RuntimeError(
        f"Feature decision registry mismatch. Missing={missing}; extra={extra}"
    )


@dataclass
class CohortAccumulator:
    n_records: int = 0
    target_known_n: int = 0
    target_missing_n: int = 0
    event_n: int = 0

    def add(self, target_value: str) -> None:
        self.n_records += 1
        if target_value == "":
            self.target_missing_n += 1
        elif target_value == "1":
            self.target_known_n += 1
            self.event_n += 1
        elif target_value == "0":
            self.target_known_n += 1
        else:
            raise ValueError(f"Invalid target value: {target_value!r}")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_binary(value: str, *, column: str, row: int, year: int) -> str:
    value = value.strip()
    if value not in {"", "0", "1"}:
        raise ValueError(
            f"CDC {year}, row {row:,}: {column}={value!r}; expected blank/0/1."
        )
    return value


def parse_care_month(value: str, *, row: int, year: int) -> Optional[int]:
    value = value.strip()
    if value == "":
        return None
    try:
        month = int(value)
    except ValueError as exc:
        raise ValueError(
            f"CDC {year}, row {row:,}: prenatal_care_month={value!r}"
        ) from exc
    if not 0 <= month <= 10:
        raise ValueError(
            f"CDC {year}, row {row:,}: prenatal_care_month={month}; expected 0..10."
        )
    return month


def recompute_target(
    target: str,
    row: dict[str, str],
    *,
    row_number: int,
    year: int,
) -> str:
    if target == "target_preterm":
        value = row["gestation_oe_weeks"].strip()
        if value == "":
            return ""
        weeks = int(value)
        return "1" if weeks < 37 else "0"

    if target == "target_nicu":
        return parse_binary(
            row["admit_nicu"],
            column="admit_nicu",
            row=row_number,
            year=year,
        )

    if target == "target_lbw":
        value = row["birth_weight_g"].strip()
        if value == "":
            return ""
        grams = int(value)
        return "1" if grams < 2500 else "0"

    raise ValueError(f"Unsupported target: {target}")


def care_partition(care_month: Optional[int], early_entry_max_month: int) -> str:
    if care_month is None:
        return "unknown_care"
    if care_month == 0:
        return "no_care"
    if 1 <= care_month <= early_entry_max_month:
        return "early_entry"
    return "later_entry"


def read_manifest_hashes(manifest_path: Path) -> dict[int, str]:
    if not manifest_path.exists():
        return {}

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes: dict[int, str] = {}
    for year in (2022, 2023):
        entry = payload.get("outputs", {}).get(str(year), {})
        digest = entry.get("sha256")
        if digest:
            hashes[year] = str(digest)
    return hashes


def target_specific_reason(feature: str, target: str) -> str:
    decision = FEATURE_DECISIONS[feature]

    if decision.booking_strict or decision.pregnancy_oracle:
        return ""

    direct_sources = {
        "target_preterm": {
            "gestation_oe_weeks",
            "gestation_oe_recode3",
            "target_preterm",
        },
        "target_nicu": {
            "admit_nicu",
            "target_nicu",
        },
        "target_lbw": {
            "birth_weight_g",
            "target_lbw",
        },
    }

    if feature in direct_sources[target]:
        return f"Direct source or deterministic recode of {target}."

    if feature.startswith("target_"):
        return "Outcome label; never used as a feature."

    if feature in {
        "gestation_combined_weeks",
        "gestation_oe_weeks",
        "gestation_oe_recode3",
        "birth_weight_g",
        "delivery_method",
        "delivery_method_binary",
        "an_vent",
        "admit_nicu",
        "an_seiz",
    }:
        return "Observed at delivery or after birth; unavailable at the prediction landmark."

    return decision.general_exclusion_reason


def build_feature_timing_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for feature in FEATURE_CANDIDATES:
        decision = FEATURE_DECISIONS[feature]
        row: dict[str, object] = {
            "feature": feature,
            "cdc_2022_field": feature,
            "cdc_2023_field": feature,
            "timing_class": decision.timing_class,
            "booking_strict": int(decision.booking_strict),
            "pregnancy_oracle": int(decision.pregnancy_oracle),
            "oracle_incremental": int(decision.oracle_incremental),
            "reliable_timestamp": decision.reliable_timestamp,
            "availability_basis": decision.availability_basis,
            "general_exclusion_reason": decision.general_exclusion_reason,
        }

        for target in PRIMARY_TARGETS:
            row[f"in_booking_{target.removeprefix('target_')}"] = int(
                decision.booking_strict
            )
            row[f"in_oracle_{target.removeprefix('target_')}"] = int(
                decision.pregnancy_oracle
            )
            excluded = not decision.pregnancy_oracle
            row[f"excluded_{target.removeprefix('target_')}"] = int(excluded)
            row[f"exclusion_reason_{target.removeprefix('target_')}"] = (
                target_specific_reason(feature, target) if excluded else ""
            )

        rows.append(row)

    return rows


def scan_year(
    *,
    year: int,
    csv_path: Path,
    early_entry_max_month: int,
    progress_every: int,
) -> tuple[
    dict[str, dict[str, CohortAccumulator]],
    int,
    Counter,
]:
    """
    Returns:
        target -> step -> accumulator
        row count
        formula mismatch counts
    """
    counts: dict[str, dict[str, CohortAccumulator]] = {
        target: defaultdict(CohortAccumulator)
        for target in PRIMARY_TARGETS
    }
    formula_mismatches = Counter()
    started = time.time()

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
        buffering=1024 * 1024,
    ) as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CDC {year}: CSV has no header.")

        missing_columns = sorted(
            set(EXPECTED_COLUMNS) - set(reader.fieldnames)
        )
        if missing_columns:
            raise ValueError(
                f"CDC {year}: required harmonized columns are missing: "
                f"{missing_columns}"
            )

        rows = 0
        for rows, row in enumerate(reader, start=1):
            is_us = parse_binary(
                row["is_us_resident"],
                column="is_us_resident",
                row=rows,
                year=year,
            )
            is_singleton = parse_binary(
                row["is_singleton"],
                column="is_singleton",
                row=rows,
                year=year,
            )
            care_month = parse_care_month(
                row["prenatal_care_month"],
                row=rows,
                year=year,
            )

            expected_is_us = (
                "1" if row["residence_status"].strip() in {"1", "2", "3"}
                else "0" if row["residence_status"].strip() == "4"
                else ""
            )
            expected_singleton = (
                "1" if row["plurality"].strip() == "1"
                else "0" if row["plurality"].strip() in {"2", "3", "4"}
                else ""
            )
            if is_us != expected_is_us:
                formula_mismatches["is_us_resident"] += 1
            if is_singleton != expected_singleton:
                formula_mismatches["is_singleton"] += 1

            for target in PRIMARY_TARGETS:
                stored_target = parse_binary(
                    row[target],
                    column=target,
                    row=rows,
                    year=year,
                )
                recomputed = recompute_target(
                    target,
                    row,
                    row_number=rows,
                    year=year,
                )
                if stored_target != recomputed:
                    formula_mismatches[target] += 1

                counts[target]["all_records"].add(recomputed)

                if is_us == "1":
                    counts[target]["us_residents"].add(recomputed)

                if is_us != "1" or is_singleton != "1":
                    continue

                counts[target]["us_resident_singletons"].add(recomputed)

                if recomputed == "":
                    counts[target]["target_missing_excluded"].add(recomputed)
                    continue

                counts[target]["full_cohort"].add(recomputed)
                partition = care_partition(
                    care_month,
                    early_entry_max_month=early_entry_max_month,
                )
                counts[target][partition].add(recomputed)

            if progress_every and rows % progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"CDC {year}: scanned {rows:,} rows "
                    f"in {elapsed:,.1f}s"
                )

    if rows != EXPECTED_ROWS[year]:
        raise ValueError(
            f"CDC {year}: scanned {rows:,} rows; "
            f"expected {EXPECTED_ROWS[year]:,}."
        )

    mismatch_total = sum(formula_mismatches.values())
    if mismatch_total:
        raise ValueError(
            f"CDC {year}: {mismatch_total:,} derived-column/formula "
            f"mismatches: {dict(formula_mismatches)}"
        )

    # Partition integrity checks.
    for target in PRIMARY_TARGETS:
        full = counts[target]["full_cohort"]
        partitions = [
            counts[target]["early_entry"],
            counts[target]["later_entry"],
            counts[target]["no_care"],
            counts[target]["unknown_care"],
        ]
        partition_n = sum(item.n_records for item in partitions)
        partition_events = sum(item.event_n for item in partitions)

        if partition_n != full.n_records:
            raise ValueError(
                f"CDC {year}, {target}: care partitions sum to "
                f"{partition_n:,}, full cohort has {full.n_records:,}."
            )
        if partition_events != full.event_n:
            raise ValueError(
                f"CDC {year}, {target}: care-partition events sum to "
                f"{partition_events:,}, full cohort has {full.event_n:,}."
            )

    return counts, rows, formula_mismatches


STEP_ORDER = {
    "all_records": 10,
    "us_residents": 20,
    "us_resident_singletons": 30,
    "target_missing_excluded": 40,
    "full_cohort": 50,
    "early_entry": 60,
    "later_entry": 70,
    "no_care": 80,
    "unknown_care": 90,
}

STEP_DEFINITIONS = {
    "all_records": "All occurrence-file records.",
    "us_residents": "residence_status in {1,2,3}.",
    "us_resident_singletons": (
        "U.S. residents with plurality=1; primary population before "
        "target-availability filtering."
    ),
    "target_missing_excluded": (
        "Primary population records with unknown target; excluded from "
        "target-specific modelling."
    ),
    "full_cohort": (
        "U.S.-resident singleton births with known target, regardless of "
        "prenatal-care entry."
    ),
    "early_entry": (
        "Full cohort with prenatal care beginning between month 1 and the "
        "configured early-entry maximum."
    ),
    "later_entry": (
        "Full cohort with prenatal care beginning after the configured "
        "early-entry maximum through month 10."
    ),
    "no_care": "Full cohort with prenatal_care_month=0.",
    "unknown_care": "Full cohort with unknown prenatal_care_month.",
}


def build_cohort_rows(
    all_counts: dict[int, dict[str, dict[str, CohortAccumulator]]],
    early_entry_max_month: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for year in (2022, 2023):
        for target in PRIMARY_TARGETS:
            target_counts = all_counts[year][target]
            full = target_counts["full_cohort"]

            for step in sorted(STEP_ORDER, key=STEP_ORDER.get):
                acc = target_counts[step]

                if step in {
                    "full_cohort",
                    "early_entry",
                    "later_entry",
                    "no_care",
                    "unknown_care",
                }:
                    population_coverage = (
                        100.0 * acc.n_records / full.n_records
                        if full.n_records else math.nan
                    )
                    event_capture = (
                        100.0 * acc.event_n / full.event_n
                        if full.event_n else math.nan
                    )
                else:
                    population_coverage = ""
                    event_capture = ""

                rows.append({
                    "year": year,
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "flow_order": STEP_ORDER[step],
                    "step": step,
                    "is_primary_analysis_cohort": int(
                        step in {"full_cohort", "early_entry"}
                    ),
                    "n_records": acc.n_records,
                    "target_known_n": acc.target_known_n,
                    "target_missing_n": acc.target_missing_n,
                    "event_n": acc.event_n,
                    "event_rate_pct_among_known": (
                        100.0 * acc.event_n / acc.target_known_n
                        if acc.target_known_n else ""
                    ),
                    "population_coverage_pct_of_full_cohort": population_coverage,
                    "event_capture_pct_of_full_cohort_events": event_capture,
                    "early_entry_max_month": early_entry_max_month,
                    "definition": STEP_DEFINITIONS[step],
                })

    return rows


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, object]],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {path}. Use --overwrite to replace it."
        )

    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temp_path.exists():
        temp_path.unlink()

    try:
        with temp_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())

        if path.exists():
            path.unlink()
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def write_json_atomic(
    path: Path,
    payload: dict[str, object],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {path}. Use --overwrite to replace it."
        )

    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if path.exists():
            path.unlink()
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def build_feature_sets_payload() -> dict[str, object]:
    booking = sorted(
        feature
        for feature, decision in FEATURE_DECISIONS.items()
        if decision.booking_strict
    )
    oracle = sorted(
        feature
        for feature, decision in FEATURE_DECISIONS.items()
        if decision.pregnancy_oracle
    )
    incremental = sorted(
        feature
        for feature, decision in FEATURE_DECISIONS.items()
        if decision.oracle_incremental
    )

    return {
        "booking_strict": {
            "description": (
                "Features defensibly available before pregnancy or by the "
                "first prenatal visit."
            ),
            "features": booking,
            "n_features": len(booking),
        },
        "pregnancy_oracle": {
            "description": (
                "Retrospective upper bound: booking_strict plus selected "
                "pregnancy-period variables without reliable timestamps. "
                "Do not describe as late clinical prediction."
            ),
            "features": oracle,
            "n_features": len(oracle),
        },
        "oracle_incremental_over_booking": {
            "features": incremental,
            "n_features": len(incremental),
        },
        "target_specific_sets": {
            target: {
                "booking_strict": booking,
                "pregnancy_oracle": oracle,
            }
            for target in PRIMARY_TARGETS
        },
    }


def build_report(
    *,
    feature_rows: list[dict[str, object]],
    cohort_rows: list[dict[str, object]],
    early_entry_max_month: int,
    hashes: dict[int, str],
) -> str:
    booking = [
        row["feature"] for row in feature_rows
        if row["booking_strict"] == 1
    ]
    incremental = [
        row["feature"] for row in feature_rows
        if row["oracle_incremental"] == 1
    ]

    lines = [
        "# CDC 2022→2023 Feature Timing and Cohort Audit",
        "",
        "## Locked definitions",
        "",
        f"- Early entry: prenatal care began in months 1–{early_entry_max_month}.",
        "- Full cohort: U.S.-resident singleton births with known target.",
        "- No-care: prenatal_care_month=0; unavailable to the prenatal system.",
        "- Unknown-care: missing prenatal_care_month; kept as a separate scenario.",
        "- pregnancy_oracle is a retrospective upper bound, not a late clinical model.",
        "",
        "## Feature sets",
        "",
        f"- booking_strict ({len(booking)}): {', '.join(booking)}",
        f"- oracle increment ({len(incremental)}): {', '.join(incremental)}",
        "",
        "## Input hashes",
        "",
        f"- CDC 2022 harmonized SHA-256: `{hashes[2022]}`",
        f"- CDC 2023 harmonized SHA-256: `{hashes[2023]}`",
        "",
        "## Cohort summary",
        "",
        "| Year | Target | Full N | Full events | Early N | Early events | "
        "Early population coverage | Early event capture |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    indexed = {
        (row["year"], row["target"], row["step"]): row
        for row in cohort_rows
    }
    for year in (2022, 2023):
        for target in PRIMARY_TARGETS:
            full = indexed[(year, target, "full_cohort")]
            early = indexed[(year, target, "early_entry")]
            lines.append(
                f"| {year} | {target} | {full['n_records']:,} | "
                f"{full['event_n']:,} | {early['n_records']:,} | "
                f"{early['event_n']:,} | "
                f"{early['population_coverage_pct_of_full_cohort']:.3f}% | "
                f"{early['event_capture_pct_of_full_cohort_events']:.3f}% |"
            )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "The cohort table separates model-eligible recall from population-level "
        "event capture. Events in later-entry, no-care and unknown-care groups "
        "remain in the full-population denominator even when an early-entry "
        "system could not score them.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate feature_timing.csv and cohort_flow.csv for the "
            "CDC 2022→2023 prospective-style audit."
        )
    )
    parser.add_argument(
        "--cdc2022",
        type=Path,
        default=Path(
            "metadata/harmonization/"
            "cdc_natality_2022_harmonized.csv"
        ),
    )
    parser.add_argument(
        "--cdc2023",
        type=Path,
        default=Path(
            "data/cdc_temporal_harmonized/"
            "cdc_natality_2023_harmonized.csv"
        ),
    )
    parser.add_argument(
        "--preprocessing-manifest",
        type=Path,
        default=Path(
            "data/cdc_temporal_harmonized/"
            "preprocessing_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metadata/design"),
    )
    parser.add_argument(
        "--early-entry-max-month",
        type=int,
        default=3,
        help=(
            "Latest prenatal-care start month included in early entry. "
            "Default 3 means first trimester."
        ),
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip comparison with preprocessing_manifest.json hashes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500_000,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()

    if not 1 <= args.early_entry_max_month <= 10:
        print(
            "ERROR: --early-entry-max-month must be between 1 and 10.",
            file=sys.stderr,
        )
        return 2

    input_paths = {
        2022: args.cdc2022,
        2023: args.cdc2023,
    }
    for year, path in input_paths.items():
        if not path.exists():
            print(f"ERROR: CDC {year} file not found: {path}", file=sys.stderr)
            return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    expected_hashes = read_manifest_hashes(args.preprocessing_manifest)
    observed_hashes: dict[int, str] = {}

    try:
        for year, path in input_paths.items():
            print(f"Calculating CDC {year} SHA-256...")
            digest = sha256_file(path)
            observed_hashes[year] = digest

            if not args.skip_hash_check and expected_hashes:
                expected = expected_hashes.get(year)
                if expected and digest != expected:
                    raise ValueError(
                        f"CDC {year} harmonized hash mismatch.\n"
                        f"Observed: {digest}\nExpected: {expected}"
                    )

        all_counts: dict[
            int,
            dict[str, dict[str, CohortAccumulator]],
        ] = {}

        for year in (2022, 2023):
            print(f"\n=== SCANNING CDC {year} ===")
            counts, rows, _ = scan_year(
                year=year,
                csv_path=input_paths[year],
                early_entry_max_month=args.early_entry_max_month,
                progress_every=args.progress_every,
            )
            all_counts[year] = counts
            print(f"CDC {year}: PASS, {rows:,} rows.")

        feature_rows = build_feature_timing_rows()
        cohort_rows = build_cohort_rows(
            all_counts,
            early_entry_max_month=args.early_entry_max_month,
        )
        feature_sets = build_feature_sets_payload()

        cohort_definitions = {
            "primary_population": (
                "is_us_resident=1 AND is_singleton=1"
            ),
            "target_known_requirement": (
                "Each target-specific modelling cohort excludes only rows "
                "with that target missing."
            ),
            "full_cohort": (
                "Primary population with known target, irrespective of "
                "prenatal-care entry."
            ),
            "early_entry": (
                f"Full cohort with prenatal_care_month in "
                f"1..{args.early_entry_max_month}."
            ),
            "later_entry": (
                f"Full cohort with prenatal_care_month in "
                f"{args.early_entry_max_month + 1}..10."
            ),
            "no_care": (
                "Full cohort with prenatal_care_month=0; considered "
                "unavailable to a prenatal-care-based system."
            ),
            "unknown_care": (
                "Full cohort with missing prenatal_care_month; analyzed "
                "separately rather than merged with no-care."
            ),
            "four_factorial_cells": [
                "booking_strict × early_entry",
                "booking_strict × full_cohort",
                "pregnancy_oracle × early_entry",
                "pregnancy_oracle × full_cohort",
            ],
        }

        feature_path = args.output_dir / "feature_timing.csv"
        cohort_path = args.output_dir / "cohort_flow.csv"
        feature_sets_path = args.output_dir / "feature_sets.json"
        definitions_path = args.output_dir / "cohort_definitions.json"
        report_path = args.output_dir / "feature_cohort_report.md"
        manifest_path = args.output_dir / "feature_cohort_manifest.json"

        write_csv_atomic(
            feature_path,
            feature_rows,
            overwrite=args.overwrite,
        )
        write_csv_atomic(
            cohort_path,
            cohort_rows,
            overwrite=args.overwrite,
        )
        write_json_atomic(
            feature_sets_path,
            feature_sets,
            overwrite=args.overwrite,
        )
        write_json_atomic(
            definitions_path,
            cohort_definitions,
            overwrite=args.overwrite,
        )

        report = build_report(
            feature_rows=feature_rows,
            cohort_rows=cohort_rows,
            early_entry_max_month=args.early_entry_max_month,
            hashes=observed_hashes,
        )
        if report_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output exists: {report_path}. Use --overwrite."
            )
        report_path.write_text(report, encoding="utf-8")

        manifest = {
            "status": "PASS",
            "elapsed_seconds": round(time.time() - started, 3),
            "inputs": {
                str(year): {
                    "path": str(input_paths[year].resolve()),
                    "sha256": observed_hashes[year],
                    "rows": EXPECTED_ROWS[year],
                }
                for year in (2022, 2023)
            },
            "configuration": {
                "primary_targets": list(PRIMARY_TARGETS),
                "primary_population": (
                    "U.S.-resident singleton births"
                ),
                "early_entry_max_month": (
                    args.early_entry_max_month
                ),
                "no_care_policy": (
                    "Unavailable to the system; retained in full population."
                ),
                "unknown_care_policy": (
                    "Separate scenario; not merged with no-care."
                ),
                "pregnancy_oracle_label": (
                    "Retrospective upper bound"
                ),
            },
            "outputs": {
                "feature_timing": str(feature_path.resolve()),
                "feature_sets": str(feature_sets_path.resolve()),
                "cohort_flow": str(cohort_path.resolve()),
                "cohort_definitions": str(definitions_path.resolve()),
                "report": str(report_path.resolve()),
            },
            "feature_counts": {
                "booking_strict": feature_sets[
                    "booking_strict"
                ]["n_features"],
                "pregnancy_oracle": feature_sets[
                    "pregnancy_oracle"
                ]["n_features"],
                "oracle_incremental": feature_sets[
                    "oracle_incremental_over_booking"
                ]["n_features"],
            },
            "formula_checks": {
                "status": "PASS",
                "mismatches": 0,
            },
        }
        write_json_atomic(
            manifest_path,
            manifest,
            overwrite=args.overwrite,
        )

        print("\n=== FEATURE/COHORT AUDIT COMPLETE ===")
        print("Status: PASS")
        print(
            "booking_strict features: "
            f"{feature_sets['booking_strict']['n_features']}"
        )
        print(
            "pregnancy_oracle features: "
            f"{feature_sets['pregnancy_oracle']['n_features']}"
        )
        print(f"Output directory: {args.output_dir}")
        print(f"- {feature_path}")
        print(f"- {cohort_path}")
        print(f"- {report_path}")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
