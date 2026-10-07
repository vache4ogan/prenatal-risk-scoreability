"""Synthetic fixed-width records only; no CDC records or fitted models are used."""

import csv
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import extract_canonical_cdc as extractor
import preprocess_harmonize_cdc_2022_2023 as harmonizer
import run_tae_canonical_clearml as canonical
from run_temporal_four_cell import normalize_frame


def fixed_width(year, replacements=None):
    # Independent fixture coordinates from the official layout, not COORDINATES.
    fields = {
        (9, 12): str(year), (13, 14): "01", (75, 76): "30", (104, 104): "1",
        (105, 106): "01", (120, 120): "1", (124, 124): "4", (171, 172): "01",
        (173, 174): "00", (175, 176): "00", (224, 225): "03", (238, 239): "12",
        (251, 251): "Y", (255, 256): "00", (280, 281): "64", (283, 286): "25.0",
        (292, 294): "150", (304, 305): "30", (313, 313): "N", (314, 314): "Y",
        (315, 315): "N", (316, 316): "N", (317, 317): "N", (407, 407): "1",
        (408, 408): "1", (454, 454): "1", (475, 475): "F", (490, 491): "36",
        (499, 500): "39", (503, 503): "2", (504, 507): "3500",
        (517, 517): "N", (519, 519): "Y", (522, 522): "N",
    }
    fields.update(replacements or {})
    record = [" "] * 1330
    for (start, end), value in fields.items():
        assert len(value) == end - start + 1
        record[start - 1:end] = value
    return "".join(record)


def records(year):
    return [
        fixed_width(year),
        fixed_width(year, {(490, 491): "39", (499, 500): "36", (503, 503): "1",
                           (504, 507): "2499", (519, 519): "N", (105, 106): "31",
                           (224, 225): "00", (104, 104): "4", (454, 454): "2"}),
        fixed_width(year, {(490, 491): "35", (499, 500): "99", (503, 503): "3",
                           (504, 507): "9999", (519, 519): "U", (314, 314): "U",
                           (105, 106): "99", (224, 225): "99", (280, 281): "99",
                           (283, 286): "99.9", (292, 294): "999", (171, 172): "99",
                           (173, 174): "99", (175, 176): "99", (407, 407): "9",
                           (408, 408): "9"}),
        fixed_width(year, {(499, 500): "37", (504, 507): "2500", (519, 519): "N",
                           (105, 106): "02", (224, 225): "10", (104, 104): "3",
                           (454, 454): "4"}),
        fixed_width(year, {(490, 491): "99", (499, 500): "17", (503, 503): "1",
                           (504, 507): "0227", (105, 106): "06", (224, 225): "01",
                           (104, 104): "2"}),
        fixed_width(year, {(490, 491): "47", (499, 500): "47", (504, 507): "8165",
                           (519, 519): "N", (105, 106): "30", (224, 225): "04",
                           (283, 286): "13.0"}),
    ]


def reference_csv(tmp_path, year, payloads, monkeypatch):
    source = tmp_path / "source.csv"
    rows = [extractor.extract_source_row(payload, year) for payload in payloads]
    assert set(rows[0]) == {spec.source_name(year) for spec in harmonizer.FIELDS}
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "reference.csv"
    with monkeypatch.context() as patch:
        patch.setitem(harmonizer.EXPECTED_ROWS, year, len(payloads))
        result = harmonizer.process_year(
            year=year, input_path=source, temp_output_path=output, final_output_path=output,
            input_sha256=harmonizer.sha256_file(source), progress_every=0, max_invalid_examples=0,
        )
    assert result.rows == len(payloads)
    assert not result.invalid_findings
    return output


@pytest.mark.parametrize("year", [2022, 2023])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_raw_to_canonical_roundtrip(tmp_path, monkeypatch, year, newline):
    payloads = records(year)
    raw = tmp_path / "synthetic.txt"
    raw_bytes = (newline.join(payloads) + newline).encode("ascii")
    raw.write_bytes(raw_bytes)
    reference = reference_csv(tmp_path, year, payloads, monkeypatch)
    lock = {"expected_rows": len(payloads), "expected_sha256": harmonizer.sha256_file(reference),
            "config_sha256": "synthetic-test-lock"}
    # Production CLI has no count/hash override; only this fixture replaces its lock.
    monkeypatch.setattr(extractor, "canonical_lock", lambda requested_year: lock)
    output = tmp_path / "new_release"
    assert extractor.main(["--year", str(year), "--input", str(raw), "--output-dir", str(output)]) == 0
    csv_path = output / f"cdc_natality_{year}_harmonized.csv"
    assert csv_path.read_bytes() == reference.read_bytes()
    assert raw.read_bytes() == raw_bytes
    manifest = json.loads((output / "extraction_manifest.json").read_text())
    assert manifest["input_sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    assert manifest["output_sha256"] == lock["expected_sha256"]
    assert manifest["rows"] == 6 and manifest["model_fitting"] is False
    assert (output / "harmonized_schema.json").is_file()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == list(harmonizer.OUTPUT_COLUMNS)
    assert [row["target_preterm"] for row in rows] == ["0", "1", "", "0", "1", "0"]
    assert [row["target_lbw"] for row in rows] == ["0", "1", "", "0", "1", "0"]
    assert [row["target_nicu"] for row in rows] == ["1", "0", "", "0", "1", "0"]
    assert [row["mother_race"] for row in rows] == ["01", "31", "", "02", "06", "30"]
    assert [row["prenatal_care_month"] for row in rows] == ["3", "0", "", "10", "1", "4"]
    assert [row["is_us_resident"] for row in rows] == ["1", "0", "1", "1", "1", "1"]
    assert [row["is_singleton"] for row in rows] == ["1", "0", "1", "0", "1", "1"]
    assert rows[0]["gestation_combined_weeks"] == "36"
    assert rows[0]["gestation_oe_weeks"] == "39"
    assert rows[0]["gestation_oe_recode3"] == "2"
    assert rows[0]["delivery_method_binary"] == "1"
    assert rows[0]["target_gdm"] == "1" and rows[2]["target_gdm"] == ""
    assert rows[2]["mother_bmi"] == "" and rows[5]["mother_bmi"] == "13"

    frame = normalize_frame(pd.read_csv(csv_path))
    assert frame.mother_race.dropna().tolist() == [1, 31, 2, 6, 30]
    pre = canonical.build_preprocessor(canonical.LANDMARK_STRICT_FEATURES, ["mother_race"],
                                       canonical.DEFAULT_CONFIG["preprocessing"])
    pre.fit(frame[canonical.LANDMARK_STRICT_FEATURES])
    encoded = pre.named_transformers_["categorical"].transform(frame[["mother_race"]])
    assert np.asarray(encoded.sum(axis=1)).ravel().tolist() == [1] * len(rows)


@pytest.mark.parametrize("year", [2022, 2023])
def test_frozen_production_lock_is_not_weakened(year):
    lock = extractor.canonical_lock(year)
    assert lock["expected_rows"] == {2022: 3_676_029, 2023: 3_605_081}[year]
    assert lock["expected_sha256"] == {
        2022: "c5f1ab62b0e9ac63795e75f6075d2523a3f4c5dc3e95c7aa1f7df6763205a59c",
        2023: "82b38b24d2f795a4a5233ce3a2fcb43b70789339306ba9044bde56f50bb270a8",
    }[year]


@pytest.mark.parametrize("case", ["short", "long", "blank", "wrong_year", "non_ascii", "invalid_code"])
def test_invalid_raw_never_leaves_a_release(tmp_path, case):
    record = fixed_width(2023)
    if case == "short":
        record = record[:-1]
    elif case == "long":
        record += " "
    elif case == "blank":
        record = ""
    elif case == "wrong_year":
        record = fixed_width(2022)
    elif case == "non_ascii":
        record = "\u00e9" + record[1:]
    else:
        record = fixed_width(2023, {(105, 106): "32"})
    raw = tmp_path / "synthetic.txt"
    raw.write_bytes((record + "\n").encode("utf-8"))
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="Record 1"):
        extractor.reconstruct_year(raw, output, 2023, expected_rows=1,
                                   expected_sha256="unused", config_sha256="synthetic")
    assert not output.exists()
    assert not list(tmp_path.glob(".canonical_*"))


@pytest.mark.parametrize("wrong_count", [True, False])
def test_count_and_hash_fail_closed(tmp_path, wrong_count):
    raw = tmp_path / "synthetic.txt"
    raw.write_text(fixed_width(2023) + "\n", encoding="ascii")
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="rows|SHA-256"):
        extractor.reconstruct_year(raw, output, 2023, expected_rows=2 if wrong_count else 1,
                                   expected_sha256="0" * 64, config_sha256="synthetic")
    assert not output.exists()
    assert not list(tmp_path.glob(".canonical_*"))


def test_existing_output_is_never_overwritten(tmp_path):
    raw = tmp_path / "synthetic.txt"
    raw.write_text(fixed_width(2023) + "\n", encoding="ascii")
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "untouched.txt"
    sentinel.write_text("unchanged", encoding="ascii")
    with pytest.raises(FileExistsError):
        extractor.reconstruct_year(raw, output, 2023, expected_rows=1,
                                   expected_sha256="unused", config_sha256="synthetic")
    assert sentinel.read_text() == "unchanged"


def test_cli_rejects_toy_input_under_real_lock(tmp_path, capsys):
    raw = tmp_path / "synthetic.txt"
    raw.write_text(fixed_width(2023) + "\n", encoding="ascii")
    output = tmp_path / "release"
    assert extractor.main(["--year", "2023", "--input", str(raw), "--output-dir", str(output)]) == 2
    assert "expected 3,605,081" in capsys.readouterr().err
    assert not output.exists()
