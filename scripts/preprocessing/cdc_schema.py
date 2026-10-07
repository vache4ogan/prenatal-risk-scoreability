"""CDC 2022/2023 common schema, missing codes, and outcome definitions."""
from __future__ import annotations
import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
EXPECTED_RECORD_LENGTH = 1330

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

# One-based, inclusive coordinates in the official US public-use layouts.
COORDINATES = {'birth_year': (9, 12), 'birth_month': (13, 14), 'residence_status': (104, 104), 'mother_age': (75, 76), 'mother_race': (105, 106), 'marital_status': (120, 120), 'mother_educ': (124, 124), 'wic_benefits': (251, 251), 'mother_height_inches': (280, 281), 'mother_bmi': (283, 286), 'mother_weight_pre': (292, 294), 'weight_gain': (304, 305), 'prior_live_births': (171, 172), 'prior_dead_births': (173, 174), 'prior_terminations': (175, 176), 'prenatal_care_month': (224, 225), 'prenatal_visits': (238, 239), 'cigarettes_1_tri': (255, 256), 'diab_pre': (313, 313), 'diab_gest': (314, 314), 'hyper_pre': (315, 315), 'hyper_gest': (316, 316), 'eclampsia': (317, 317), 'delivery_method': (407, 407), 'delivery_method_binary': (408, 408), 'plurality': (454, 454), 'child_sex': (475, 475), 'gestation_combined_weeks': (490, 491), 'gestation_oe_weeks': (499, 500), 'gestation_oe_recode3': (503, 503), 'birth_weight_g': (504, 507), 'an_vent': (517, 517), 'admit_nicu': (519, 519), 'an_seiz': (522, 522)}
