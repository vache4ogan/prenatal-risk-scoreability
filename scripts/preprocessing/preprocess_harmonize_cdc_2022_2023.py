#!/usr/bin/env python3
"""
Common preprocessing and schema harmonization for CDC Natality 2022 and 2023.

Inputs are the already audited parsed CSV files. The script:

1. Never modifies source files.
2. Verifies the locked SHA-256 hashes by default.
3. Harmonizes the 2022/2023 column names into one canonical schema.
4. Converts official CDC unknown codes to blank/NA consistently.
5. Converts Y/N fields to nullable 1/0 values.
6. Creates:
       is_us_resident
       is_singleton
       target_preterm   (OEGest_Comb < 37)
       target_nicu
       target_lbw       (birth weight < 2500 g)
       target_gdm
7. Scans every row and validates value domains.
8. Generates:
       cdc_natality_2022_harmonized.csv
       cdc_natality_2023_harmonized.csv
       schema_diff_2022_2023.csv
       harmonized_schema.json
       column_quality.csv
       target_prevalence.csv
       existing_derived_column_check.csv
       preprocessing_manifest.json

Important:
- target_preterm is calculated ONLY from the obstetric estimate
  OEGest_Comb, canonical name gestation_oe_weeks.
- gestation_combined_weeks (COMBGEST) is retained but never used to define
  target_preterm.
- no-care prenatal_care_month=0 is retained as a real category.
- unknown target labels remain blank; they are never silently changed to 0.
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


# Locked source files that have already passed independent raw-coordinate audits.
EXPECTED_INPUT_SHA256 = {
    2022: "22ecb8084eed4bf2eaa4bd770eac458ca5c945a04edd0b0c18dcb0d6108354c1",
    2023: "1726c8feaf472628f5972122b0f389831c3e919676909c714334b48c60cd0af8",
}

EXPECTED_ROWS = {
    2022: 3_676_029,
    2023: 3_605_081,
}

TARGETS = (
    "target_preterm",
    "target_nicu",
    "target_lbw",
    "target_gdm",
)

DERIVED_COLUMNS = (
    "is_us_resident",
    "is_singleton",
    *TARGETS,
)

SCOPES = (
    "all_records",
    "us_residents",
    "us_resident_singletons",
)


@dataclass(frozen=True)
class FieldSpec:
    canonical_name: str
    source_2022: str
    source_2023: str
    dtype: str
    role: str
    missing_codes: frozenset[str] = frozenset()
    allowed_codes: frozenset[str] = frozenset()
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    decimals: Optional[int] = None
    notes: str = ""

    def source_name(self, year: int) -> str:
        if year == 2022:
            return self.source_2022
        if year == 2023:
            return self.source_2023
        raise ValueError(f"Unsupported year: {year}")


# Canonical common schema: only fields available in both years.
FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("birth_year", "birth_year", "birth_year", "Int64", "time",
              min_value=2022, max_value=2023),
    FieldSpec("birth_month", "birth_month", "birth_month", "Int64", "time",
              min_value=1, max_value=12),
    FieldSpec("residence_status", "residence_status", "residence_status",
              "category", "population", allowed_codes=frozenset({"1", "2", "3", "4"}),
              notes="1-3 U.S. resident; 4 foreign resident"),

    FieldSpec("mother_age", "mother_age", "mother_age", "Int64", "demographic",
              min_value=12, max_value=50),
    FieldSpec("mother_race", "mother_race", "mother_race", "category", "demographic",
              missing_codes=frozenset({"99"}),
              allowed_codes=frozenset(f"{value:02d}" for value in range(1, 32))),
    FieldSpec("marital_status", "marital_status", "marital_status",
              "category", "demographic", missing_codes=frozenset({"9"}),
              allowed_codes=frozenset({"1", "2", "3"})),
    FieldSpec("mother_educ", "mother_educ", "mother_educ",
              "category", "demographic", missing_codes=frozenset({"9"}),
              allowed_codes=frozenset(str(value) for value in range(1, 9))),
    FieldSpec("wic_benefits", "wic_benefits", "wic_benefits",
              "boolean", "social", missing_codes=frozenset({"U"}),
              allowed_codes=frozenset({"Y", "N"})),

    FieldSpec("mother_height_inches", "mother_height_inches", "mother_height_inches",
              "Int64", "physical", missing_codes=frozenset({"99"}),
              min_value=30, max_value=78),
    FieldSpec("mother_bmi", "mother_bmi", "mother_bmi",
              "Float64", "physical", missing_codes=frozenset({"99.9"}),
              min_value=13.0, max_value=69.9, decimals=1),
    FieldSpec("mother_weight_pre", "mother_weight_pre", "mother_weight_pre",
              "Int64", "physical", missing_codes=frozenset({"999"}),
              min_value=75, max_value=375),
    FieldSpec("weight_gain", "weight_gain", "weight_gain",
              "Int64", "pregnancy_information", missing_codes=frozenset({"99"}),
              min_value=0, max_value=98,
              notes="98 means 98 pounds or more"),

    FieldSpec("prior_live_births", "prior_live_births", "prior_live_births",
              "Int64", "obstetric_history", missing_codes=frozenset({"99"}),
              min_value=0, max_value=30),
    FieldSpec("prior_dead_births", "prior_dead_births", "prior_dead_births",
              "Int64", "obstetric_history", missing_codes=frozenset({"99"}),
              min_value=0, max_value=30),
    FieldSpec("prior_terminations", "prior_terminations", "prior_terminations",
              "Int64", "obstetric_history", missing_codes=frozenset({"99"}),
              min_value=0, max_value=30),

    FieldSpec("prenatal_care_month", "prenatal_care_month", "prenatal_care_month",
              "Int64", "cohort_variable", missing_codes=frozenset({"99"}),
              min_value=0, max_value=10,
              notes="0 means no prenatal care; blank means unknown"),
    FieldSpec("prenatal_visits", "prenatal_visits", "prenatal_visits",
              "Int64", "pregnancy_information", missing_codes=frozenset({"99"}),
              min_value=0, max_value=98),
    FieldSpec("cigarettes_1_tri", "cigarettes_1_tri", "cigarettes_1_tri",
              "Int64", "pregnancy_information", missing_codes=frozenset({"99"}),
              min_value=0, max_value=98,
              notes="98 means 98 cigarettes or more"),

    FieldSpec("diab_pre", "diab_pre", "diab_pre",
              "boolean", "maternal_risk", missing_codes=frozenset({"U"}),
              allowed_codes=frozenset({"Y", "N"})),
    FieldSpec("diab_gest", "diab_gest", "diab_gest",
              "boolean", "pregnancy_oracle", missing_codes=frozenset({"U"}),
              allowed_codes=frozenset({"Y", "N"})),
    FieldSpec("hyper_pre", "hyper_pre", "hyper_pre",
              "boolean", "maternal_risk", missing_codes=frozenset({"U"}),
              allowed_codes=frozenset({"Y", "N"})),
    FieldSpec("hyper_gest", "hyper_gest", "hyper_gest",
              "boolean", "pregnancy_oracle", missing_codes=frozenset({"U"}),
              allowed_codes=frozenset({"Y", "N"})),
    FieldSpec("eclampsia", "eclampsia", "eclampsia",
              "boolean", "pregnancy_oracle", missing_codes=frozenset({"U"}),
              allowed_codes=frozenset({"Y", "N"})),

    FieldSpec("delivery_method", "delivery_method", "delivery_method",
              "category", "delivery_outcome", missing_codes=frozenset({"9"}),
              allowed_codes=frozenset({"1", "2", "3", "4", "5", "6"})),
    FieldSpec("delivery_method_binary", "delivery_method_binary", "delivery_method_binary",
              "category", "delivery_outcome", missing_codes=frozenset({"9"}),
              allowed_codes=frozenset({"1", "2"})),

    FieldSpec("plurality", "plurality", "plurality",
              "category", "population", allowed_codes=frozenset({"1", "2", "3", "4"})),
    FieldSpec("child_sex", "child_sex", "child_sex",
              "category", "infant", allowed_codes=frozenset({"M", "F"})),

    # 2022 names are aliases from the already audited raw CSV.
    FieldSpec("gestation_combined_weeks",
              "gestation_weeks_combined", "gestation_combined_weeks",
              "Int64", "outcome_source", missing_codes=frozenset({"99"}),
              min_value=17, max_value=47,
              notes="COMBGEST; retained for audit, not used for target_preterm"),
    FieldSpec("gestation_oe_weeks",
              "gestation_weeks", "gestation_oe_weeks",
              "Int64", "outcome_source", missing_codes=frozenset({"99"}),
              min_value=17, max_value=47,
              notes="OEGest_Comb; official source for target_preterm"),
    FieldSpec("gestation_oe_recode3",
              "preterm_recode", "gestation_oe_recode3",
              "category", "outcome_source", missing_codes=frozenset({"3"}),
              allowed_codes=frozenset({"1", "2"}),
              notes="OEGest_R3: 1 under 37 weeks, 2 at least 37 weeks"),

    FieldSpec("birth_weight_g", "birth_weight_g", "birth_weight_g",
              "Int64", "outcome_source", missing_codes=frozenset({"9999"}),
              min_value=227, max_value=8165),
    FieldSpec("an_vent", "an_vent", "an_vent",
              "boolean", "newborn_outcome", missing_codes=frozenset({"U"}),
              allowed_codes=frozenset({"Y", "N"})),
    FieldSpec("admit_nicu", "admit_nicu", "admit_nicu",
              "boolean", "outcome_source", missing_codes=frozenset({"U"}),
              allowed_codes=frozenset({"Y", "N"})),
    FieldSpec("an_seiz", "an_seiz", "an_seiz",
              "boolean", "newborn_outcome", missing_codes=frozenset({"U"}),
              allowed_codes=frozenset({"Y", "N"})),
)

CANONICAL_COLUMNS = tuple(spec.canonical_name for spec in FIELDS)
OUTPUT_COLUMNS = (*CANONICAL_COLUMNS, *DERIVED_COLUMNS)
FIELD_BY_CANONICAL = {spec.canonical_name: spec for spec in FIELDS}


@dataclass
class ColumnStats:
    total: int = 0
    missing: int = 0
    invalid: int = 0
    distinct: set[str] = field(default_factory=set)
    numeric_min: Optional[float] = None
    numeric_max: Optional[float] = None
    invalid_examples: list[dict[str, object]] = field(default_factory=list)

    def observe(self, value: str, *, row_number: int, raw_value: str, max_examples: int) -> None:
        self.total += 1
        if value == "":
            self.missing += 1
            return

        self.distinct.add(value)
        try:
            numeric = float(value)
        except ValueError:
            return

        if self.numeric_min is None or numeric < self.numeric_min:
            self.numeric_min = numeric
        if self.numeric_max is None or numeric > self.numeric_max:
            self.numeric_max = numeric


@dataclass
class YearResult:
    year: int
    input_path: Path
    temp_output_path: Path
    final_output_path: Path
    input_sha256: str
    rows: int
    source_header: list[str]
    source_mapping: dict[str, str]
    stats: dict[str, ColumnStats]
    target_counts: dict[str, dict[str, Counter]]
    existing_derived_mismatches: Counter
    existing_derived_compared: Counter
    unexpected_columns: list[str]
    invalid_findings: list[dict[str, object]]
    elapsed_seconds: float


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def normalized_raw(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def canonicalize(spec: FieldSpec, raw_value: object) -> tuple[str, Optional[str]]:
    """
    Return (canonical_value, error_message).

    Blank and official missing codes become blank.
    Boolean Y/N fields become 1/0.
    """
    value = normalized_raw(raw_value)

    if value == "" or value in spec.missing_codes:
        return "", None

    if spec.dtype == "boolean":
        if value == "Y":
            return "1", None
        if value == "N":
            return "0", None
        # 2023 harmonized input may already contain 1/0 in a future rerun.
        if value in {"0", "1"}:
            return value, None
        return "", f"invalid boolean code {value!r}"

    if spec.dtype == "category":
        if spec.allowed_codes and value not in spec.allowed_codes:
            return "", f"invalid category code {value!r}"
        return value, None

    if spec.dtype == "Int64":
        try:
            number_float = float(value)
        except ValueError:
            return "", f"not an integer: {value!r}"

        if not number_float.is_integer():
            return "", f"not an integer: {value!r}"

        number = int(number_float)
        if spec.min_value is not None and number < spec.min_value:
            return "", f"value {number} below minimum {spec.min_value:g}"
        if spec.max_value is not None and number > spec.max_value:
            return "", f"value {number} above maximum {spec.max_value:g}"
        return str(number), None

    if spec.dtype == "Float64":
        try:
            number = float(value)
        except ValueError:
            return "", f"not a float: {value!r}"

        if not math.isfinite(number):
            return "", f"non-finite float: {value!r}"
        if spec.min_value is not None and number < spec.min_value:
            return "", f"value {number} below minimum {spec.min_value:g}"
        if spec.max_value is not None and number > spec.max_value:
            return "", f"value {number} above maximum {spec.max_value:g}"

        decimals = spec.decimals if spec.decimals is not None else 10
        rendered = f"{number:.{decimals}f}".rstrip("0").rstrip(".")
        return rendered, None

    return "", f"unsupported dtype {spec.dtype!r}"


def binary_from_category(value: str, true_codes: set[str], false_codes: set[str]) -> str:
    if value in true_codes:
        return "1"
    if value in false_codes:
        return "0"
    return ""


def derive_columns(row: dict[str, str]) -> dict[str, str]:
    residence = row["residence_status"]
    plurality = row["plurality"]
    gestation = row["gestation_oe_weeks"]
    birth_weight = row["birth_weight_g"]

    is_us_resident = binary_from_category(
        residence,
        true_codes={"1", "2", "3"},
        false_codes={"4"},
    )
    is_singleton = binary_from_category(
        plurality,
        true_codes={"1"},
        false_codes={"2", "3", "4"},
    )

    if gestation == "":
        target_preterm = ""
    else:
        target_preterm = "1" if int(gestation) < 37 else "0"

    if birth_weight == "":
        target_lbw = ""
    else:
        target_lbw = "1" if int(birth_weight) < 2500 else "0"

    # admit_nicu and diab_gest have already been converted from Y/N to 1/0.
    target_nicu = row["admit_nicu"]
    target_gdm = row["diab_gest"]

    return {
        "is_us_resident": is_us_resident,
        "is_singleton": is_singleton,
        "target_preterm": target_preterm,
        "target_nicu": target_nicu,
        "target_lbw": target_lbw,
        "target_gdm": target_gdm,
    }


def update_target_counts(
    target_counts: dict[str, dict[str, Counter]],
    derived: dict[str, str],
) -> None:
    scopes = ["all_records"]
    if derived["is_us_resident"] == "1":
        scopes.append("us_residents")
        if derived["is_singleton"] == "1":
            scopes.append("us_resident_singletons")

    for scope in scopes:
        target_counts[scope]["population"]["n"] += 1
        for target in TARGETS:
            value = derived[target]
            if value == "":
                target_counts[scope][target]["missing"] += 1
            elif value == "1":
                target_counts[scope][target]["known"] += 1
                target_counts[scope][target]["positive"] += 1
            elif value == "0":
                target_counts[scope][target]["known"] += 1
                target_counts[scope][target]["negative"] += 1
            else:
                raise ValueError(f"Unexpected derived target value: {target}={value!r}")


def process_year(
    *,
    year: int,
    input_path: Path,
    temp_output_path: Path,
    final_output_path: Path,
    input_sha256: str,
    progress_every: int,
    max_invalid_examples: int,
) -> YearResult:
    started = time.time()
    stats = {name: ColumnStats() for name in CANONICAL_COLUMNS}
    target_counts = {
        scope: defaultdict(Counter)
        for scope in SCOPES
    }
    existing_derived_mismatches = Counter()
    existing_derived_compared = Counter()
    invalid_findings: list[dict[str, object]] = []

    with input_path.open("r", encoding="utf-8-sig", newline="", buffering=1024 * 1024) as input_file, \
         temp_output_path.open("w", encoding="utf-8", newline="", buffering=1024 * 1024) as output_file:

        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"{year}: input CSV has no header")

        source_header = [name.strip() for name in reader.fieldnames]
        if len(source_header) != len(set(source_header)):
            duplicates = sorted(
                name for name in set(source_header)
                if source_header.count(name) > 1
            )
            raise ValueError(f"{year}: duplicate source columns: {duplicates}")

        source_mapping = {
            spec.canonical_name: spec.source_name(year)
            for spec in FIELDS
        }
        missing_source_columns = [
            source_name
            for source_name in source_mapping.values()
            if source_name not in source_header
        ]
        if missing_source_columns:
            raise ValueError(
                f"{year}: required source columns are missing: "
                f"{sorted(missing_source_columns)}"
            )

        used_source_columns = set(source_mapping.values())
        unexpected_columns = sorted(
            name for name in source_header
            if name not in used_source_columns
        )

        writer = csv.DictWriter(
            output_file,
            fieldnames=list(OUTPUT_COLUMNS),
            extrasaction="ignore",
        )
        writer.writeheader()

        rows = 0
        for rows, source_row in enumerate(reader, start=1):
            canonical_row: dict[str, str] = {}

            for spec in FIELDS:
                source_name = source_mapping[spec.canonical_name]
                raw_value = source_row.get(source_name, "")
                value, error = canonicalize(spec, raw_value)

                if error is not None:
                    stats[spec.canonical_name].invalid += 1
                    if len(invalid_findings) < max_invalid_examples:
                        invalid_findings.append({
                            "year": year,
                            "row": rows,
                            "canonical_column": spec.canonical_name,
                            "source_column": source_name,
                            "raw_value": normalized_raw(raw_value),
                            "error": error,
                        })

                stats[spec.canonical_name].observe(
                    value,
                    row_number=rows,
                    raw_value=normalized_raw(raw_value),
                    max_examples=max_invalid_examples,
                )
                canonical_row[spec.canonical_name] = value

            # Strong year check after normalization.
            if canonical_row["birth_year"] != str(year):
                raise ValueError(
                    f"{year}: row {rows:,} has birth_year="
                    f"{canonical_row['birth_year']!r}"
                )

            derived = derive_columns(canonical_row)
            output_row = {**canonical_row, **derived}
            writer.writerow(output_row)
            update_target_counts(target_counts, derived)

            # Existing target columns are diagnostic only. Recomputed values win.
            for column in DERIVED_COLUMNS:
                if column not in source_header:
                    continue
                existing = normalized_raw(source_row.get(column, ""))
                if existing not in {"", "0", "1"}:
                    continue
                existing_derived_compared[column] += 1
                if existing != derived[column]:
                    existing_derived_mismatches[column] += 1

            if progress_every and rows % progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"{year}: processed {rows:,} rows "
                    f"in {elapsed:,.1f}s"
                )

        output_file.flush()
        os.fsync(output_file.fileno())

    if rows != EXPECTED_ROWS[year]:
        raise ValueError(
            f"{year}: observed {rows:,} rows, "
            f"expected {EXPECTED_ROWS[year]:,}"
        )

    invalid_total = sum(stat.invalid for stat in stats.values())
    if invalid_total:
        failure_path = temp_output_path.with_suffix(
            temp_output_path.suffix + ".invalid_values.json"
        )
        failure_path.write_text(
            json.dumps({
                "year": year,
                "invalid_total": invalid_total,
                "examples": invalid_findings,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise ValueError(
            f"{year}: found {invalid_total:,} values outside the locked "
            f"schema. Examples: {failure_path}"
        )

    return YearResult(
        year=year,
        input_path=input_path,
        temp_output_path=temp_output_path,
        final_output_path=final_output_path,
        input_sha256=input_sha256,
        rows=rows,
        source_header=source_header,
        source_mapping=source_mapping,
        stats=stats,
        target_counts=target_counts,
        existing_derived_mismatches=existing_derived_mismatches,
        existing_derived_compared=existing_derived_compared,
        unexpected_columns=unexpected_columns,
        invalid_findings=invalid_findings,
        elapsed_seconds=time.time() - started,
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_schema_diff(results: dict[int, YearResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in FIELDS:
        stats_2022 = results[2022].stats[spec.canonical_name]
        stats_2023 = results[2023].stats[spec.canonical_name]

        missing_pct_2022 = 100.0 * stats_2022.missing / stats_2022.total
        missing_pct_2023 = 100.0 * stats_2023.missing / stats_2023.total

        rows.append({
            "canonical_field": spec.canonical_name,
            "cdc_2022_column": spec.source_2022,
            "cdc_2023_column": spec.source_2023,
            "dtype": spec.dtype,
            "role": spec.role,
            "normalization": (
                "Y/N -> 1/0; unknown -> blank"
                if spec.dtype == "boolean"
                else "official unknown code -> blank"
            ),
            "official_missing_codes": "|".join(sorted(spec.missing_codes)),
            "missing_n_2022": stats_2022.missing,
            "missing_pct_2022": missing_pct_2022,
            "missing_n_2023": stats_2023.missing,
            "missing_pct_2023": missing_pct_2023,
            "missing_pct_difference_2023_minus_2022": (
                missing_pct_2023 - missing_pct_2022
            ),
            "invalid_n_2022": stats_2022.invalid,
            "invalid_n_2023": stats_2023.invalid,
            "status": (
                "MATCH"
                if stats_2022.invalid == 0 and stats_2023.invalid == 0
                else "FAIL"
            ),
            "notes": spec.notes,
        })
    return rows


def build_quality_rows(results: dict[int, YearResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in (2022, 2023):
        result = results[year]
        for spec in FIELDS:
            stats = result.stats[spec.canonical_name]
            rows.append({
                "year": year,
                "canonical_field": spec.canonical_name,
                "source_field": result.source_mapping[spec.canonical_name],
                "dtype": spec.dtype,
                "role": spec.role,
                "n_rows": stats.total,
                "missing_n": stats.missing,
                "missing_pct": 100.0 * stats.missing / stats.total,
                "non_missing_n": stats.total - stats.missing,
                "distinct_non_missing": len(stats.distinct),
                "numeric_min": stats.numeric_min,
                "numeric_max": stats.numeric_max,
                "invalid_n": stats.invalid,
                "status": "PASS" if stats.invalid == 0 else "FAIL",
            })
    return rows


def build_target_prevalence_rows(
    results: dict[int, YearResult],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in (2022, 2023):
        for scope in SCOPES:
            population_n = results[year].target_counts[scope]["population"]["n"]
            for target in TARGETS:
                counts = results[year].target_counts[scope][target]
                known = counts["known"]
                positive = counts["positive"]
                negative = counts["negative"]
                missing = counts["missing"]
                rows.append({
                    "year": year,
                    "scope": scope,
                    "target": target,
                    "population_n": population_n,
                    "known_target_n": known,
                    "positive_n": positive,
                    "negative_n": negative,
                    "missing_target_n": missing,
                    "prevalence_pct_among_known": (
                        100.0 * positive / known if known else ""
                    ),
                })
    return rows


def build_existing_check_rows(
    results: dict[int, YearResult],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in (2022, 2023):
        result = results[year]
        for column in DERIVED_COLUMNS:
            compared = result.existing_derived_compared[column]
            mismatches = result.existing_derived_mismatches[column]
            rows.append({
                "year": year,
                "derived_column": column,
                "present_in_source": column in result.source_header,
                "compared_n": compared,
                "mismatch_n": mismatches,
                "status": (
                    "NOT_PRESENT"
                    if column not in result.source_header
                    else "PASS" if mismatches == 0
                    else "MISMATCH"
                ),
                "note": "Source value is diagnostic only; output is always recomputed.",
            })
    return rows


def write_schema_json(path: Path) -> None:
    payload = {
        "description": "Locked common schema for CDC Natality 2022-2023 temporal audit.",
        "missing_representation_in_csv": "",
        "preterm_definition": "target_preterm = 1[gestation_oe_weeks < 37]",
        "lbw_definition": "target_lbw = 1[birth_weight_g < 2500]",
        "nicu_definition": "target_nicu = admit_nicu after Y/N -> 1/0",
        "gdm_definition": "target_gdm = diab_gest after Y/N -> 1/0",
        "fields": [
            {
                **asdict(spec),
                "missing_codes": sorted(spec.missing_codes),
                "allowed_codes": sorted(spec.allowed_codes),
            }
            for spec in FIELDS
        ],
        "derived_columns": {
            "is_us_resident": "1 for residence_status in {1,2,3}; 0 for 4",
            "is_singleton": "1 for plurality=1; 0 for plurality in {2,3,4}",
            "target_preterm": "1 if gestation_oe_weeks < 37; blank if unknown",
            "target_nicu": "1/0 from admit_nicu; blank if unknown",
            "target_lbw": "1 if birth_weight_g < 2500; blank if unknown",
            "target_gdm": "1/0 from diab_gest; blank if unknown",
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Harmonize and preprocess audited CDC Natality 2022 and 2023 CSVs."
        )
    )
    parser.add_argument(
        "--cdc2022",
        type=Path,
        default=Path(
            "data/cdc2022_checked/"
            "cdc_natality_2022_official_with_plurality.csv"
        ),
    )
    parser.add_argument(
        "--cdc2023",
        type=Path,
        default=Path(
            "data/cdc2023/cdc_natality_2023_audit_ready.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cdc_temporal_harmonized"),
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help=(
            "Skip locked SHA-256 verification. Use only when the same audited "
            "data were intentionally re-encoded without changing rows."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated outputs.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500_000,
    )
    parser.add_argument(
        "--max-invalid-examples",
        type=int,
        default=50,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()

    inputs = {
        2022: args.cdc2022,
        2023: args.cdc2023,
    }

    for year, path in inputs.items():
        if not path.exists():
            print(f"ERROR: {year} input not found: {path}", file=sys.stderr)
            return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    final_outputs = {
        2022: args.output_dir / "cdc_natality_2022_harmonized.csv",
        2023: args.output_dir / "cdc_natality_2023_harmonized.csv",
    }
    temp_outputs = {
        year: path.with_name(f".{path.name}.tmp.{os.getpid()}")
        for year, path in final_outputs.items()
    }

    report_paths = [
        args.output_dir / "schema_diff_2022_2023.csv",
        args.output_dir / "harmonized_schema.json",
        args.output_dir / "column_quality.csv",
        args.output_dir / "target_prevalence.csv",
        args.output_dir / "existing_derived_column_check.csv",
        args.output_dir / "preprocessing_manifest.json",
    ]

    existing_outputs = [
        path for path in [*final_outputs.values(), *report_paths]
        if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        print(
            "ERROR: generated outputs already exist. Use --overwrite or "
            "choose another --output-dir:\n  "
            + "\n  ".join(str(path) for path in existing_outputs),
            file=sys.stderr,
        )
        return 2

    for temp_path in temp_outputs.values():
        if temp_path.exists():
            temp_path.unlink()

    try:
        input_hashes = {}
        for year, path in inputs.items():
            print(f"Calculating SHA-256 for CDC {year}: {path}")
            digest = sha256_file(path)
            input_hashes[year] = digest

            if not args.skip_hash_check:
                expected = EXPECTED_INPUT_SHA256[year]
                if digest != expected:
                    raise ValueError(
                        f"CDC {year} SHA-256 mismatch.\n"
                        f"Observed: {digest}\n"
                        f"Expected: {expected}\n"
                        "The script refuses to process an unrecognized file. "
                        "Use --skip-hash-check only after verifying why the "
                        "bytes changed."
                    )

        results: dict[int, YearResult] = {}
        for year in (2022, 2023):
            print(f"\n=== PREPROCESSING CDC {year} ===")
            results[year] = process_year(
                year=year,
                input_path=inputs[year],
                temp_output_path=temp_outputs[year],
                final_output_path=final_outputs[year],
                input_sha256=input_hashes[year],
                progress_every=args.progress_every,
                max_invalid_examples=args.max_invalid_examples,
            )

        # Reports are generated before outputs are committed.
        schema_rows = build_schema_diff(results)
        quality_rows = build_quality_rows(results)
        prevalence_rows = build_target_prevalence_rows(results)
        existing_check_rows = build_existing_check_rows(results)

        schema_path = args.output_dir / "schema_diff_2022_2023.csv"
        quality_path = args.output_dir / "column_quality.csv"
        prevalence_path = args.output_dir / "target_prevalence.csv"
        existing_check_path = args.output_dir / "existing_derived_column_check.csv"
        schema_json_path = args.output_dir / "harmonized_schema.json"

        write_csv(schema_path, schema_rows, list(schema_rows[0].keys()))
        write_csv(quality_path, quality_rows, list(quality_rows[0].keys()))
        write_csv(
            prevalence_path,
            prevalence_rows,
            list(prevalence_rows[0].keys()),
        )
        write_csv(
            existing_check_path,
            existing_check_rows,
            list(existing_check_rows[0].keys()),
        )
        write_schema_json(schema_json_path)

        # Any disagreement with already present 2023 derived columns is a hard
        # warning in the manifest, but the clean outputs remain based on the
        # independently recomputed formulas.
        derived_mismatch_total = sum(
            sum(result.existing_derived_mismatches.values())
            for result in results.values()
        )

        # Atomically commit both huge CSV files only after all checks pass.
        for year in (2022, 2023):
            if final_outputs[year].exists():
                final_outputs[year].unlink()
            os.replace(temp_outputs[year], final_outputs[year])

        print("\nCalculating output SHA-256 hashes...")
        output_hashes = {
            year: sha256_file(final_outputs[year])
            for year in (2022, 2023)
        }

        manifest = {
            "status": "PASS",
            "created_at_unix": time.time(),
            "elapsed_seconds": round(time.time() - started, 3),
            "inputs": {
                str(year): {
                    "path": str(inputs[year].resolve()),
                    "sha256": input_hashes[year],
                    "expected_rows": EXPECTED_ROWS[year],
                    "rows_processed": results[year].rows,
                }
                for year in (2022, 2023)
            },
            "outputs": {
                str(year): {
                    "path": str(final_outputs[year].resolve()),
                    "sha256": output_hashes[year],
                    "rows": results[year].rows,
                }
                for year in (2022, 2023)
            },
            "canonical_columns": list(CANONICAL_COLUMNS),
            "derived_columns": list(DERIVED_COLUMNS),
            "target_definitions": {
                "target_preterm": "gestation_oe_weeks < 37",
                "target_nicu": "admit_nicu == 1",
                "target_lbw": "birth_weight_g < 2500",
                "target_gdm": "diab_gest == 1",
            },
            "source_aliases": {
                str(year): results[year].source_mapping
                for year in (2022, 2023)
            },
            "ignored_source_columns": {
                str(year): results[year].unexpected_columns
                for year in (2022, 2023)
            },
            "existing_derived_mismatch_total": derived_mismatch_total,
            "notes": [
                "Source files were never modified.",
                "All official raw unknown codes were converted to blank.",
                "No rows were dropped.",
                "target_preterm uses OEGest_Comb only.",
                "prenatal_care_month=0 remains no-care, not missing.",
            ],
        }
        manifest_path = args.output_dir / "preprocessing_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n=== HARMONIZATION COMPLETE ===")
        print("Status: PASS")
        print(f"CDC 2022 rows: {results[2022].rows:,}")
        print(f"CDC 2023 rows: {results[2023].rows:,}")
        print(f"Existing derived-column mismatches: {derived_mismatch_total:,}")
        print(f"Output directory: {args.output_dir}")
        print(f"2022 output SHA-256: {output_hashes[2022]}")
        print(f"2023 output SHA-256: {output_hashes[2023]}")
        return 0

    except Exception as exc:
        for temp_path in temp_outputs.values():
            if temp_path.exists():
                temp_path.unlink()
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
