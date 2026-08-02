#!/usr/bin/env python3
"""
Strict RAW-coordinate audit for the user's CDC Natality 2022 CSV.

This script is intentionally different from a preprocessing audit:
- CDC raw unknown codes (99, 999, 9999, 99.9, U) are VALID here.
- Derived targets and reporting flags are NOT required.
- Every CSV value is compared row-by-row with the exact official fixed-width
  slice from Nat2022PublicUS.c20230504.r20230822.txt.
- The three gestation columns receive an additional semantic mapping check.

PASS means: for every listed CSV column and every row, the value equals the
value extracted from the documented CDC 2022 coordinate assigned to that
column, with no row-order or row-count discrepancy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

EXPECTED_RECORD_LENGTH = 1330
EXPECTED_ROWS = 3_676_029
EXPECTED_US_RESIDENTS = 3_667_758
EXPECTED_FOREIGN_RESIDENTS = 8_271


@dataclass(frozen=True)
class FieldSpec:
    csv_name: str
    cdc_name: str
    start: int          # official 1-based inclusive position
    end: int            # official 1-based inclusive position
    description: str

    @property
    def width(self) -> int:
        return self.end - self.start + 1

    def extract(self, payload: str) -> str:
        return payload[self.start - 1:self.end].strip()


# Locked mapping for exactly the columns listed by the user.
# Coordinates are from the official 2022 Natality User Guide.
FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("birth_year", "DOB_YY", 9, 12, "Birth year"),
    FieldSpec("birth_month", "DOB_MM", 13, 14, "Birth month"),
    FieldSpec("mother_age", "MAGER", 75, 76, "Mother's single years of age"),
    FieldSpec("mother_race", "MRACE31", 105, 106, "Mother's race recode 31"),
    FieldSpec("marital_status", "DMAR", 120, 120, "Marital status"),
    FieldSpec("mother_educ", "MEDUC", 124, 124, "Mother's education"),
    FieldSpec("wic_benefits", "WIC", 251, 251, "WIC during pregnancy"),
    FieldSpec("mother_height_inches", "M_Ht_In", 280, 281, "Mother's height in inches"),
    FieldSpec("mother_bmi", "BMI", 283, 286, "Body mass index"),
    FieldSpec("mother_weight_pre", "PWgt_R", 292, 294, "Pre-pregnancy weight"),
    FieldSpec("weight_gain", "WTGAIN", 304, 305, "Weight gain"),
    FieldSpec("prior_live_births", "PRIORLIVE", 171, 172, "Prior births now living"),
    FieldSpec("prior_dead_births", "PRIORDEAD", 173, 174, "Prior births now dead"),
    FieldSpec("prior_terminations", "PRIORTERM", 175, 176, "Prior other terminations"),
    FieldSpec("prenatal_care_month", "PRECARE", 224, 225, "Month prenatal care began"),
    FieldSpec("prenatal_visits", "PREVIS", 238, 239, "Number of prenatal visits"),
    FieldSpec("cigarettes_1_tri", "CIG_1", 255, 256, "Cigarettes first trimester"),
    FieldSpec("diab_pre", "RF_PDIAB", 313, 313, "Pre-pregnancy diabetes"),
    FieldSpec("diab_gest", "RF_GDIAB", 314, 314, "Gestational diabetes"),
    FieldSpec("hyper_pre", "RF_PHYPE", 315, 315, "Pre-pregnancy hypertension"),
    FieldSpec("hyper_gest", "RF_GHYPE", 316, 316, "Gestational hypertension"),
    FieldSpec("eclampsia", "RF_EHYPE", 317, 317, "Eclampsia"),
    FieldSpec("delivery_method", "RDMETH_REC", 407, 407, "Detailed delivery method recode"),
    FieldSpec("delivery_method_binary", "DMETH_REC", 408, 408, "Vaginal/C-section recode"),
    FieldSpec("child_sex", "SEX", 475, 475, "Sex of infant"),
    FieldSpec("gestation_weeks_combined", "COMBGEST", 490, 491, "Combined gestation in weeks"),
    FieldSpec("gestation_weeks", "OEGest_Comb", 499, 500, "Edited obstetric estimate in weeks"),
    FieldSpec("preterm_recode", "OEGest_R3", 503, 503, "OE gestation recode: under 37 / 37+ / unknown"),
    FieldSpec("birth_weight_g", "DBWT", 504, 507, "Birth weight in grams"),
    FieldSpec("an_vent", "AB_AVEN1", 517, 517, "Immediate assisted ventilation"),
    FieldSpec("admit_nicu", "AB_NICU", 519, 519, "Admission to NICU"),
    FieldSpec("an_seiz", "AB_SEIZ", 522, 522, "Seizures"),
    FieldSpec("residence_status", "RESTATUS", 104, 104, "Residence status"),
)

FIELD_BY_NAME = {f.csv_name: f for f in FIELDS}
EXPECTED_HEADER = [f.csv_name for f in FIELDS]

# Alternative official fields used only to diagnose ambiguous/incorrect naming.
GESTATION_CANDIDATES: dict[str, tuple[FieldSpec, ...]] = {
    "gestation_weeks_combined": (
        FieldSpec("gestation_weeks_combined", "COMBGEST", 490, 491, "Combined gestation"),
        FieldSpec("gestation_weeks_combined", "OEGest_Comb", 499, 500, "Obstetric estimate"),
    ),
    "gestation_weeks": (
        FieldSpec("gestation_weeks", "OEGest_Comb", 499, 500, "Obstetric estimate"),
        FieldSpec("gestation_weeks", "COMBGEST", 490, 491, "Combined gestation"),
    ),
    "preterm_recode": (
        FieldSpec("preterm_recode", "OEGest_R3", 503, 503, "OE preterm recode"),
        FieldSpec("preterm_recode", "GESTREC3", 494, 494, "Combined-gestation preterm recode"),
    ),
}


@dataclass
class Finding:
    severity: str
    check: str
    column: str
    message: str
    observed: object
    expected: object


@dataclass
class Benchmark:
    name: str
    observed: Optional[float]
    official: Optional[float]
    status: str
    note: str = ""


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def normalized_csv_value(value: str) -> str:
    """Only remove surrounding whitespace. Do NOT normalize CDC codes."""
    return value.strip()


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else math.nan


def per_1000(numerator: int, denominator: int) -> float:
    return 1000.0 * numerator / denominator if denominator else math.nan


def rounded_equal(observed: float, official: float, decimals: int) -> bool:
    return round(observed + 1e-12, decimals) == round(official, decimals)


def validate_raw_code(field: str, value: str) -> bool:
    """Validate RAW codes, including CDC unknown codes as legitimate values."""
    if value == "":
        # Blank is allowed for non-reporting areas.
        return True

    categorical = {
        "birth_year": {"2022"},
        "birth_month": {f"{m:02d}" for m in range(1, 13)},
        "mother_race": {f"{n:02d}" for n in range(1, 32)},
        "marital_status": {"1", "2", "3", "9"},
        "mother_educ": {str(n) for n in range(1, 10)},
        "wic_benefits": {"Y", "N", "U"},
        "diab_pre": {"Y", "N", "U"},
        "diab_gest": {"Y", "N", "U"},
        "hyper_pre": {"Y", "N", "U"},
        "hyper_gest": {"Y", "N", "U"},
        "eclampsia": {"Y", "N", "U"},
        "delivery_method": {"1", "2", "3", "4", "5", "6", "9"},
        "delivery_method_binary": {"1", "2", "9"},
        "child_sex": {"M", "F"},
        "preterm_recode": {"1", "2", "3"},
        "an_vent": {"Y", "N", "U"},
        "admit_nicu": {"Y", "N", "U"},
        "an_seiz": {"Y", "N", "U"},
        "residence_status": {"1", "2", "3", "4"},
    }
    if field in categorical:
        return value in categorical[field]

    try:
        number = float(value)
    except ValueError:
        return False

    if field == "mother_age":
        return 12 <= number <= 50
    if field == "mother_height_inches":
        return 30 <= number <= 78 or number == 99
    if field == "mother_bmi":
        return 13.0 <= number <= 69.9 or number == 99.9
    if field == "mother_weight_pre":
        return 75 <= number <= 375 or number == 999
    if field == "weight_gain":
        return 0 <= number <= 99       # 98=98+, 99=unknown
    if field in {"prior_live_births", "prior_dead_births", "prior_terminations"}:
        return 0 <= number <= 30 or number == 99
    if field == "prenatal_care_month":
        return 0 <= number <= 10 or number == 99
    if field == "prenatal_visits":
        return 0 <= number <= 99       # 99=unknown
    if field == "cigarettes_1_tri":
        return 0 <= number <= 99       # 98=98+, 99=unknown
    if field in {"gestation_weeks_combined", "gestation_weeks"}:
        return 17 <= number <= 47 or number == 99
    if field == "birth_weight_g":
        return 227 <= number <= 8165 or number == 9999
    return True


def run_audit(
    csv_path: Path,
    raw_path: Path,
    output_dir: Path,
    max_examples: int,
    progress_every: int,
) -> int:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    findings: list[Finding] = []
    mismatch_counts = Counter()
    invalid_code_counts = Counter()
    missing_counts = Counter()
    value_counts: dict[str, Counter] = {f.csv_name: Counter() for f in FIELDS}
    examples: list[dict[str, object]] = []

    candidate_mismatches: dict[str, Counter] = {
        column: Counter() for column in GESTATION_CANDIDATES
    }

    length_counts = Counter()
    short_records = 0
    nonblank_trailing_records = 0
    padded_records = 0

    rows_compared = 0
    extra_csv_rows = 0
    extra_raw_rows = 0

    # Independent official-sanity accumulators, calculated from RAW fixed-width data.
    official = Counter()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file, \
         raw_path.open("r", encoding="utf-8", newline="") as raw_file:

        reader = csv.reader(csv_file)
        try:
            header = next(reader)
        except StopIteration:
            raise RuntimeError("CSV is empty")

        header = [h.strip() for h in header]
        duplicates = sorted({h for h in header if header.count(h) > 1})
        if duplicates:
            findings.append(Finding("CRITICAL", "schema", "header", "Duplicate CSV columns.", duplicates, "no duplicates"))

        missing_header = [name for name in EXPECTED_HEADER if name not in header]
        unexpected_header = [name for name in header if name not in EXPECTED_HEADER]
        for name in missing_header:
            findings.append(Finding("CRITICAL", "schema", name, "Required raw parsed column is missing.", "missing", "present"))
        for name in unexpected_header:
            findings.append(Finding("WARNING", "schema", name, "Unexpected column in raw-coordinate CSV.", "present", "not expected"))

        if missing_header:
            # We can still audit available columns, but outcome will be FAIL.
            pass

        indexes = {name: header.index(name) for name in EXPECTED_HEADER if name in header}
        max_index = max(indexes.values(), default=-1)

        for row_number, pair in enumerate(itertools.zip_longest(reader, raw_file), start=1):
            csv_row, raw_line = pair

            if csv_row is None:
                extra_raw_rows += 1
                continue
            if raw_line is None:
                extra_csv_rows += 1
                continue

            if len(csv_row) <= max_index:
                findings.append(Finding(
                    "CRITICAL", "row_width", "csv",
                    f"CSV row {row_number} has fewer fields than the header.",
                    len(csv_row), len(header),
                ))
                continue

            line = raw_line.rstrip("\r\n")
            actual_length = len(line)
            length_counts[actual_length] += 1
            if actual_length < EXPECTED_RECORD_LENGTH:
                short_records += 1
                payload = line
            else:
                payload = line[:EXPECTED_RECORD_LENGTH]
                trailing = line[EXPECTED_RECORD_LENGTH:]
                if trailing:
                    if trailing.strip(" \t\x00") == "":
                        padded_records += 1
                    else:
                        nonblank_trailing_records += 1

            # Exact coordinate checks for all available columns.
            for spec in FIELDS:
                if spec.csv_name not in indexes:
                    continue
                csv_value = normalized_csv_value(csv_row[indexes[spec.csv_name]])
                expected = spec.extract(payload)

                value_counts[spec.csv_name][csv_value] += 1
                if csv_value == "":
                    missing_counts[spec.csv_name] += 1
                if not validate_raw_code(spec.csv_name, csv_value):
                    invalid_code_counts[spec.csv_name] += 1

                if csv_value != expected:
                    mismatch_counts[spec.csv_name] += 1
                    if sum(1 for ex in examples if ex["column"] == spec.csv_name) < max_examples:
                        examples.append({
                            "row": row_number,
                            "column": spec.csv_name,
                            "cdc_field": spec.cdc_name,
                            "official_position": f"{spec.start}-{spec.end}",
                            "csv_value": csv_value,
                            "raw_expected": expected,
                            "raw_fragment_repr": repr(payload[spec.start - 1:spec.end]),
                        })

            # Semantic candidate mapping for gestation fields.
            for column, candidates in GESTATION_CANDIDATES.items():
                if column not in indexes:
                    continue
                csv_value = normalized_csv_value(csv_row[indexes[column]])
                for candidate in candidates:
                    if csv_value != candidate.extract(payload):
                        candidate_mismatches[column][candidate.cdc_name] += 1

            # Independent official sanity counts from raw positions.
            official["raw_records"] += 1
            residence = payload[103:104].strip()  # RESTATUS 104
            us_resident = residence in {"1", "2", "3"}
            if us_resident:
                official["us_resident_births"] += 1
            elif residence == "4":
                official["foreign_resident_births"] += 1

            if us_resident:
                oe = payload[498:500].strip()      # OEGest_Comb 499-500
                if oe == "99" or oe == "":
                    official["preterm_missing"] += 1
                elif oe.isdigit() and 17 <= int(oe) <= 47:
                    official["preterm_stated"] += 1
                    if int(oe) < 37:
                        official["preterm_positive"] += 1

                bw = payload[503:507].strip()      # DBWT 504-507
                if bw == "9999" or bw == "":
                    official["lbw_missing"] += 1
                elif bw.isdigit() and 227 <= int(bw) <= 8165:
                    official["lbw_stated"] += 1
                    if int(bw) < 2500:
                        official["lbw_positive"] += 1

                dmeth = payload[407:408].strip()   # DMETH_REC 408
                if dmeth == "9" or dmeth == "":
                    official["delivery_missing"] += 1
                elif dmeth in {"1", "2"}:
                    official["delivery_stated"] += 1
                    if dmeth == "2":
                        official["cesarean_positive"] += 1

                care = payload[223:225].strip()    # PRECARE 224-225
                if care == "99" or care == "":
                    official["care_missing"] += 1
                else:
                    official["care_stated"] += 1
                    if care in {"01", "02", "03"}:
                        official["first_trimester_care"] += 1
                    if care in {"00", "07", "08", "09", "10"}:
                        official["late_or_no_care"] += 1

                nicu = payload[518:519].strip()    # AB_NICU 519
                if nicu in {"Y", "N"}:
                    official["nicu_stated"] += 1
                    if nicu == "Y":
                        official["nicu_positive"] += 1

                gdm = payload[313:314].strip()     # RF_GDIAB 314
                if gdm in {"Y", "N"}:
                    official["gdm_stated"] += 1
                    if gdm == "Y":
                        official["gdm_positive"] += 1

            rows_compared += 1
            if progress_every and rows_compared % progress_every == 0:
                elapsed = time.time() - started
                total_mismatches = sum(mismatch_counts.values())
                print(
                    f"Проверено {rows_compared:,} строк; "
                    f"несовпадений полей {total_mismatches:,}; "
                    f"время {elapsed:,.1f} сек."
                )

    # Structural findings.
    if rows_compared != EXPECTED_ROWS:
        findings.append(Finding("CRITICAL", "row_count", "csv/raw", "Compared row count differs from the official occurrence-file count.", rows_compared, EXPECTED_ROWS))
    if extra_csv_rows:
        findings.append(Finding("CRITICAL", "row_count", "csv", "CSV contains rows after raw TXT ended.", extra_csv_rows, 0))
    if extra_raw_rows:
        findings.append(Finding("CRITICAL", "row_count", "raw_txt", "Raw TXT contains rows after CSV ended.", extra_raw_rows, 0))
    if short_records:
        findings.append(Finding("CRITICAL", "record_length", "raw_txt", "Raw records shorter than 1330 characters.", short_records, 0))
    if nonblank_trailing_records:
        findings.append(Finding("CRITICAL", "record_length", "raw_txt", "Nonblank content exists after documented position 1330.", nonblank_trailing_records, 0))

    # Coordinate and code findings.
    for spec in FIELDS:
        mismatches = mismatch_counts[spec.csv_name]
        if mismatches:
            findings.append(Finding(
                "CRITICAL", "coordinate_match", spec.csv_name,
                f"CSV does not exactly match official field {spec.cdc_name} at positions {spec.start}-{spec.end}.",
                mismatches, 0,
            ))
        invalid = invalid_code_counts[spec.csv_name]
        if invalid:
            findings.append(Finding(
                "CRITICAL", "raw_code_domain", spec.csv_name,
                "Values outside the official RAW code/range domain.", invalid, 0,
            ))

    # Gestation semantic map: expected mapping must be the unique best mapping.
    mapping_rows = []
    for column, candidates in GESTATION_CANDIDATES.items():
        ranked = sorted(
            ((candidate, candidate_mismatches[column][candidate.cdc_name]) for candidate in candidates),
            key=lambda item: item[1],
        )
        best_candidate, best_mismatches = ranked[0]
        expected_candidate = candidates[0]
        status = "PASS" if best_candidate.cdc_name == expected_candidate.cdc_name and best_mismatches == 0 else "FAIL"
        for candidate, mismatches in ranked:
            mapping_rows.append({
                "csv_column": column,
                "candidate_cdc_field": candidate.cdc_name,
                "candidate_position": f"{candidate.start}-{candidate.end}",
                "mismatch_count": mismatches,
                "is_expected_semantic_mapping": candidate.cdc_name == expected_candidate.cdc_name,
                "is_best_match": candidate.cdc_name == best_candidate.cdc_name,
                "column_status": status,
            })
        if status == "FAIL":
            findings.append(Finding(
                "CRITICAL", "gestation_semantics", column,
                "Gestation column does not uniquely match its expected official semantic field.",
                f"best={best_candidate.cdc_name}, mismatches={best_mismatches}",
                f"{expected_candidate.cdc_name} with 0 mismatches",
            ))

    # Official sanity benchmarks. These support the coordinate audit but do not
    # replace exact raw-vs-CSV comparison.
    observed_benchmarks = {
        "Occurrence rows": float(official["raw_records"]),
        "U.S.-resident births": float(official["us_resident_births"]),
        "Foreign-resident births": float(official["foreign_resident_births"]),
        "Preterm births (<37 weeks, OE)": float(official["preterm_positive"]),
        "Gestational age not stated": float(official["preterm_missing"]),
        "Preterm rate (%)": pct(official["preterm_positive"], official["preterm_stated"]),
        "Low-birthweight births": float(official["lbw_positive"]),
        "Birthweight not stated": float(official["lbw_missing"]),
        "Low-birthweight rate (%)": pct(official["lbw_positive"], official["lbw_stated"]),
        "Cesarean deliveries": float(official["cesarean_positive"]),
        "Delivery method not stated": float(official["delivery_missing"]),
        "Cesarean rate (%)": pct(official["cesarean_positive"], official["delivery_stated"]),
        "Prenatal-care month not stated": float(official["care_missing"]),
        "First-trimester prenatal care (%)": pct(official["first_trimester_care"], official["care_stated"]),
        "Third-trimester or no prenatal care (%)": pct(official["late_or_no_care"], official["care_stated"]),
        "NICU admission (%)": pct(official["nicu_positive"], official["nicu_stated"]),
        "Gestational diabetes (per 1,000)": per_1000(official["gdm_positive"], official["gdm_stated"]),
    }
    official_values = {
        "Occurrence rows": (3_676_029.0, 0),
        "U.S.-resident births": (3_667_758.0, 0),
        "Foreign-resident births": (8_271.0, 0),
        "Preterm births (<37 weeks, OE)": (380_548.0, 0),
        "Gestational age not stated": (2_671.0, 0),
        "Preterm rate (%)": (10.38, 2),
        "Low-birthweight births": (315_288.0, 0),
        "Birthweight not stated": (2_990.0, 0),
        "Low-birthweight rate (%)": (8.60, 2),
        "Cesarean deliveries": (1_178_066.0, 0),
        "Delivery method not stated": (2_729.0, 0),
        "Cesarean rate (%)": (32.1, 1),
        "Prenatal-care month not stated": (79_212.0, 0),
        "First-trimester prenatal care (%)": (77.0, 1),
        "Third-trimester or no prenatal care (%)": (6.8, 1),
        "NICU admission (%)": (9.5, 1),
        "Gestational diabetes (per 1,000)": (81.0, 1),
    }

    benchmark_rows = []
    for name, observed in observed_benchmarks.items():
        expected, decimals = official_values[name]
        status = "PASS" if rounded_equal(observed, expected, decimals) else "FAIL"
        benchmark_rows.append({
            "benchmark": name,
            "observed": observed,
            "official": expected,
            "rounding_decimals": decimals,
            "status": status,
        })
        if status == "FAIL":
            findings.append(Finding(
                "WARNING", "official_sanity", name,
                "Independent raw-file estimate does not match the published 2022 value after official rounding.",
                observed, expected,
            ))

    # Write outputs.
    coordinate_rows = []
    for spec in FIELDS:
        coordinate_rows.append({
            "csv_column": spec.csv_name,
            "cdc_field": spec.cdc_name,
            "official_position": f"{spec.start}-{spec.end}",
            "width": spec.width,
            "rows_compared": rows_compared,
            "mismatch_count": mismatch_counts[spec.csv_name],
            "invalid_raw_code_count": invalid_code_counts[spec.csv_name],
            "blank_count": missing_counts[spec.csv_name],
            "status": "PASS" if mismatch_counts[spec.csv_name] == 0 and invalid_code_counts[spec.csv_name] == 0 else "FAIL",
            "description": spec.description,
        })

    with (output_dir / "coordinate_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(coordinate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(coordinate_rows)

    with (output_dir / "gestation_mapping_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mapping_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mapping_rows)

    with (output_dir / "mismatch_examples.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["row", "column", "cdc_field", "official_position", "csv_value", "raw_expected", "raw_fragment_repr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(examples)

    with (output_dir / "official_sanity_checks.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(benchmark_rows[0].keys()))
        writer.writeheader()
        writer.writerows(benchmark_rows)

    profile_rows = []
    for spec in FIELDS:
        counter = value_counts[spec.csv_name]
        for value, count in counter.most_common():
            profile_rows.append({
                "column": spec.csv_name,
                "value": value,
                "count": count,
                "percent": 100.0 * count / rows_compared if rows_compared else math.nan,
                "is_blank": value == "",
                "raw_code_valid": validate_raw_code(spec.csv_name, value),
            })
    with (output_dir / "raw_value_profile.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["column", "value", "count", "percent", "is_blank", "raw_code_valid"])
        writer.writeheader()
        writer.writerows(profile_rows)

    length_rows = []
    for length, count in sorted(length_counts.items()):
        length_rows.append({
            "length_without_newline": length,
            "record_count": count,
            "classification": (
                "EXACT_1330" if length == EXPECTED_RECORD_LENGTH
                else "SHORT_FAIL" if length < EXPECTED_RECORD_LENGTH
                else "LONG_PADDING_OR_TRAILING"
            ),
        })
    with (output_dir / "record_length_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["length_without_newline", "record_count", "classification"])
        writer.writeheader()
        writer.writerows(length_rows)

    finding_rows = [asdict(f) for f in findings]
    with (output_dir / "audit_findings.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["severity", "check", "column", "message", "observed", "expected"])
        writer.writeheader()
        writer.writerows(finding_rows)

    critical = sum(1 for f in findings if f.severity == "CRITICAL")
    warnings = sum(1 for f in findings if f.severity == "WARNING")
    total_field_mismatches = sum(mismatch_counts.values())
    overall_status = "PASS" if critical == 0 else "FAIL"

    manifest = {
        "audit_type": "raw_coordinate_audit",
        "overall_status": overall_status,
        "csv_path": str(csv_path.resolve()),
        "raw_txt_path": str(raw_path.resolve()),
        "csv_sha256": sha256_file(csv_path),
        "raw_txt_sha256": sha256_file(raw_path),
        "rows_compared": rows_compared,
        "total_field_mismatches": total_field_mismatches,
        "critical_findings": critical,
        "warnings": warnings,
        "short_records": short_records,
        "blank_or_nul_padded_records": padded_records,
        "nonblank_trailing_records": nonblank_trailing_records,
        "elapsed_seconds": time.time() - started,
        "locked_columns": EXPECTED_HEADER,
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# CDC Natality 2022 — Raw Coordinate Audit",
        "",
        f"**Overall status:** {overall_status}",
        "",
        f"- Rows compared: {rows_compared:,}",
        f"- Exact raw/CSV field mismatches: {total_field_mismatches:,}",
        f"- Critical findings: {critical}",
        f"- Warnings: {warnings}",
        f"- CSV SHA-256: `{manifest['csv_sha256']}`",
        f"- Raw TXT SHA-256: `{manifest['raw_txt_sha256']}`",
        "",
        "## Interpretation",
        "",
        "This audit checks raw parsing coordinates only. CDC encoded unknown values such as 99, 999, 9999, 99.9 and U are intentionally retained and are not errors.",
        "Derived targets, reporting flags, plurality and preprocessing outputs are not required because they are not columns of this raw parsed CSV.",
        "",
        "## Record structure",
        "",
        f"- Records shorter than 1330: {short_records:,}",
        f"- Records with harmless blank/NUL padding after 1330: {padded_records:,}",
        f"- Records with nonblank trailing content: {nonblank_trailing_records:,}",
        "",
        "## Coordinate results",
        "",
        "| CSV column | CDC field | Position | Mismatches | Invalid raw codes | Status |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for row in coordinate_rows:
        report.append(
            f"| {row['csv_column']} | {row['cdc_field']} | {row['official_position']} | "
            f"{row['mismatch_count']} | {row['invalid_raw_code_count']} | {row['status']} |"
        )

    report.extend([
        "",
        "## Gestation semantic mapping",
        "",
        "The intended mapping is COMBGEST → gestation_weeks_combined, OEGest_Comb → gestation_weeks, and OEGest_R3 → preterm_recode.",
        "See `gestation_mapping_audit.csv` for comparison against alternative official gestation fields.",
        "",
        "## Findings",
        "",
    ])
    if not findings:
        report.append("No critical errors or warnings were detected.")
    else:
        report.extend([
            "| Severity | Check | Column | Message | Observed | Expected |",
            "|---|---|---|---|---:|---:|",
        ])
        for finding in findings:
            report.append(
                f"| {finding.severity} | {finding.check} | {finding.column} | "
                f"{finding.message} | {finding.observed} | {finding.expected} |"
            )

    (output_dir / "audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n=== CDC 2022 RAW COORDINATE AUDIT ===")
    print(f"Status: {overall_status}")
    print(f"Rows compared: {rows_compared:,}")
    print(f"Exact field mismatches: {total_field_mismatches:,}")
    print(f"Critical findings: {critical}")
    print(f"Warnings: {warnings}")
    print(f"Output: {output_dir}")

    return 0 if overall_status == "PASS" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict raw-coordinate audit for CDC Natality 2022 CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/cdc2022_checked/cdc_natality_2022_official.csv"),
    )
    parser.add_argument(
        "--raw-txt",
        type=Path,
        default=Path("data/cdc2022_checked/Nat2022PublicUS.c20230504.r20230822.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cdc2022_checked/raw_coordinate_audit"),
    )
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=500_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")
    if not args.raw_txt.exists():
        raise SystemExit(f"Raw TXT not found: {args.raw_txt}")
    raise SystemExit(run_audit(
        csv_path=args.csv,
        raw_path=args.raw_txt,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        progress_every=args.progress_every,
    ))
