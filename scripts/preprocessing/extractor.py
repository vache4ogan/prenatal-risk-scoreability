#!/usr/bin/env python3
"""
Парсер CDC Natality 2023 Public Use fixed-width file.

Что исправлено относительно исходного скрипта:
- позиции берутся из официального User Guide 2023 как 1-based inclusive;
- преобразование в Python-срез выполняется централизованно, поэтому меньше риска off-by-one;
- исправлены WTGAIN, WIC, PWgt_R, CIG_1, RDMETH_REC, AB_AVEN1, AB_NICU, AB_SEIZ;
- специальные коды Unknown/Not stated превращаются в пустые значения CSV;
- 00 для prenatal_care_month сохраняется: это "No prenatal care", а не пропуск;
- добавлены plurality и три основных таргета: preterm, NICU, LBW;
- добавлены reporting flags для важных полей;
- исправлены синтаксические ошибки processed_count += 1 и закрывающая скобка.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class FieldSpec:
    # Позиции из CDC User Guide: 1-based, обе границы включены.
    start: int
    end: int
    missing_codes: frozenset[str] = frozenset()

    def extract(self, line: str) -> str:
        # Например, позиции CDC 519-519 -> Python line[518:519].
        value = line[self.start - 1 : self.end].strip()
        if not value or value in self.missing_codes:
            return ""
        return value


# Официальная разметка CDC Natality 2023.
# Название справа в комментарии — официальное имя поля CDC.
CDC_2023_LAYOUT: dict[str, FieldSpec] = {
    # Временные метки
    "birth_year": FieldSpec(9, 12),                         # DOB_YY
    "birth_month": FieldSpec(13, 14),                       # DOB_MM

    # Демография
    "mother_age": FieldSpec(75, 76),                        # MAGER
    "mother_race": FieldSpec(105, 106, frozenset({"99"})), # MRACE31
    "marital_status": FieldSpec(120, 120, frozenset({"9"})), # DMAR
    "mother_educ": FieldSpec(124, 124, frozenset({"9"})),   # MEDUC
    "wic_benefits": FieldSpec(251, 251, frozenset({"U"})),  # WIC
    "f_wic": FieldSpec(252, 252),                           # F_WIC

    # Физические параметры
    "mother_height_inches": FieldSpec(
        280, 281, frozenset({"99"})
    ),                                                       # M_Ht_In
    "mother_bmi": FieldSpec(
        283, 286, frozenset({"99.9"})
    ),                                                       # BMI
    "mother_weight_pre": FieldSpec(
        292, 294, frozenset({"999"})
    ),                                                       # PWgt_R
    "weight_gain": FieldSpec(
        304, 305, frozenset({"99"})
    ),                                                       # WTGAIN; 98 = 98+ lb, это валидно
    "f_weight_gain": FieldSpec(307, 307),                    # F_WTGAIN

    # Акушерский анамнез
    "prior_live_births": FieldSpec(
        171, 172, frozenset({"99"})
    ),                                                       # PRIORLIVE
    "prior_dead_births": FieldSpec(
        173, 174, frozenset({"99"})
    ),                                                       # PRIORDEAD
    "prior_terminations": FieldSpec(
        175, 176, frozenset({"99"})
    ),                                                       # PRIORTERM

    # Дородовое наблюдение
    # ВАЖНО: 00 = no prenatal care, поэтому 00 не удаляем.
    "prenatal_care_month": FieldSpec(
        224, 225, frozenset({"99"})
    ),                                                       # PRECARE
    "f_prenatal_care_month": FieldSpec(226, 226),            # F_MPCB
    "prenatal_visits": FieldSpec(
        238, 239, frozenset({"99"})
    ),                                                       # PREVIS
    "f_prenatal_visits": FieldSpec(244, 244),                # F_TPCV

    # Курение
    # 98 = 98+ сигарет в день, это валидное значение; 99 = unknown.
    "cigarettes_1_tri": FieldSpec(
        255, 256, frozenset({"99"})
    ),                                                       # CIG_1

    # Болезни матери: U = unknown/not stated -> пустое значение
    "diab_pre": FieldSpec(313, 313, frozenset({"U"})),       # RF_PDIAB
    "diab_gest": FieldSpec(314, 314, frozenset({"U"})),      # RF_GDIAB
    "hyper_pre": FieldSpec(315, 315, frozenset({"U"})),      # RF_PHYPE
    "hyper_gest": FieldSpec(316, 316, frozenset({"U"})),     # RF_GHYPE
    "eclampsia": FieldSpec(317, 317, frozenset({"U"})),      # RF_EHYPE
    "f_diab_pre": FieldSpec(319, 319),                       # F_RF_PDIAB
    "f_diab_gest": FieldSpec(320, 320),                      # F_RF_GDIAB
    "f_hyper_pre": FieldSpec(321, 321),                      # F_RF_PHYPER
    "f_hyper_gest": FieldSpec(322, 322),                     # F_RF_GHYPER
    "f_eclampsia": FieldSpec(323, 323),                      # F_RF_ECLAMP

    # Метод родоразрешения
    "delivery_method": FieldSpec(
        407, 407, frozenset({"9"})
    ),                                                       # RDMETH_REC
    "f_delivery_method": FieldSpec(409, 409),                # F_DMETH_REC

    # Ребенок и исходы
    "plurality": FieldSpec(454, 454),                        # DPLURAL: 1 = singleton
    "child_sex": FieldSpec(475, 475),                        # SEX
    "gestation_weeks": FieldSpec(
        490, 491, frozenset({"99"})
    ),                                                       # COMBGEST
    "birth_weight_g": FieldSpec(
        504, 507, frozenset({"9999"})
    ),                                                       # DBWT

    # Abnormal Conditions of the Newborn
    "an_vent": FieldSpec(517, 517, frozenset({"U"})),        # AB_AVEN1
    "admit_nicu": FieldSpec(519, 519, frozenset({"U"})),     # AB_NICU
    "an_seiz": FieldSpec(522, 522, frozenset({"U"})),        # AB_SEIZ
    "f_an_vent": FieldSpec(524, 524),                        # F_AB_VENT
    "f_admit_nicu": FieldSpec(526, 526),                     # F_AB_NIUC
    "f_an_seiz": FieldSpec(529, 529),                        # F_AB_SEIZ
}

MAX_REQUIRED_POSITION = max(spec.end for spec in CDC_2023_LAYOUT.values())
TARGET_COLUMNS = ["is_singleton", "target_preterm", "target_nicu", "target_lbw"]


def safe_int(value: str) -> Optional[int]:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def binary_target(condition: Optional[bool]) -> str:
    if condition is None:
        return ""
    return "1" if condition else "0"


def build_targets(row: dict[str, str]) -> dict[str, str]:
    plurality = safe_int(row["plurality"])
    gestation = safe_int(row["gestation_weeks"])
    birth_weight = safe_int(row["birth_weight_g"])
    nicu = row["admit_nicu"]

    # Не превращаем неизвестный таргет в отрицательный класс.
    is_singleton = (
        ""
        if plurality is None or plurality not in {1, 2, 3, 4}
        else binary_target(plurality == 1)
    )

    target_preterm = (
        ""
        if gestation is None or not 17 <= gestation <= 47
        else binary_target(gestation < 37)
    )

    target_lbw = (
        ""
        if birth_weight is None or not 227 <= birth_weight <= 8165
        else binary_target(birth_weight < 2500)
    )

    if nicu == "Y":
        target_nicu = "1"
    elif nicu == "N":
        target_nicu = "0"
    else:
        target_nicu = ""

    return {
        "is_singleton": is_singleton,
        "target_preterm": target_preterm,
        "target_nicu": target_nicu,
        "target_lbw": target_lbw,
    }


def parse_txt_to_csv(
    txt_input: Path,
    csv_output: Path,
    limit_rows: Optional[int] = None,
) -> None:
    started_at = time.time()
    print(f"Читаю: {txt_input}")
    print(f"Пишу:  {csv_output}")

    if not txt_input.exists():
        raise FileNotFoundError(f"Файл не найден: {txt_input}")

    csv_output.parent.mkdir(parents=True, exist_ok=True)

    headers = list(CDC_2023_LAYOUT.keys()) + TARGET_COLUMNS
    non_missing = Counter()
    target_counts = {
        target: Counter()
        for target in ["target_preterm", "target_nicu", "target_lbw"]
    }

    processed_count = 0
    skipped_short_lines = 0

    with txt_input.open("r", encoding="utf-8", newline="") as infile, \
         csv_output.open("w", encoding="utf-8", newline="") as outfile:

        writer = csv.DictWriter(outfile, fieldnames=headers)
        writer.writeheader()

        for line_number, raw_line in enumerate(infile, start=1):
            # Удаляем только перевод строки. Пробелы внутри fixed-width записи нужны.
            line = raw_line.rstrip("\r\n")

            if not line:
                continue

            if len(line) < MAX_REQUIRED_POSITION:
                skipped_short_lines += 1
                print(
                    f"WARNING: строка {line_number} имеет длину {len(line)}, "
                    f"нужно минимум {MAX_REQUIRED_POSITION}; строка пропущена."
                )
                continue

            row = {
                field_name: spec.extract(line)
                for field_name, spec in CDC_2023_LAYOUT.items()
            }
            row.update(build_targets(row))
            writer.writerow(row)

            processed_count += 1

            for field_name, value in row.items():
                if value != "":
                    non_missing[field_name] += 1

            for target_name in target_counts:
                target_value = row[target_name]
                if target_value != "":
                    target_counts[target_name][target_value] += 1

            if processed_count % 500_000 == 0:
                elapsed = time.time() - started_at
                print(
                    f"Обработано: {processed_count:,} строк "
                    f"за {elapsed:,.1f} сек."
                )

            if limit_rows is not None and processed_count >= limit_rows:
                print(f"Достигнут лимит: {limit_rows:,} строк.")
                break

    elapsed = time.time() - started_at

    print("\n=== ПАРСИНГ ЗАВЕРШЕН ===")
    print(f"Сохранено записей: {processed_count:,}")
    print(f"Пропущено коротких строк: {skipped_short_lines:,}")
    print(f"Время: {elapsed:,.2f} сек.")

    print("\nПроверка критичных колонок — число НЕпропущенных значений:")
    for field_name in [
        "weight_gain",
        "admit_nicu",
        "an_seiz",
        "an_vent",
        "gestation_weeks",
        "birth_weight_g",
        "plurality",
    ]:
        count = non_missing[field_name]
        pct = 100 * count / processed_count if processed_count else 0
        print(f"  {field_name:20s}: {count:>10,} ({pct:6.2f}%)")
        if count == 0:
            print(f"  !!! ERROR: {field_name} полностью пустая")

    print("\nРаспределение таргетов среди размеченных строк:")
    for target_name, counts in target_counts.items():
        n0 = counts["0"]
        n1 = counts["1"]
        labeled = n0 + n1
        prevalence = n1 / labeled if labeled else float("nan")
        print(
            f"  {target_name:16s}: "
            f"0={n0:,}, 1={n1:,}, labeled={labeled:,}, "
            f"prevalence={prevalence:.4%}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse CDC Natality 2023 fixed-width public-use file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/cdc2023/"
            "Nat2023PublicUS.c20240509.r20240724.txt"
        ),
        help="Путь к исходному fixed-width TXT.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/cdc2023/"
            "cdc_natality_2023_full.csv"
        ),
        help="Путь к выходному CSV.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Для теста можно указать, например, 100000.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    parse_txt_to_csv(
        txt_input=args.input,
        csv_output=args.output,
        limit_rows=args.limit_rows,
    )