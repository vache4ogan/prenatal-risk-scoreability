"""Reconstruct canonical CDC 2022/2023 CSVs from official fixed-width US files.

The legacy extractor is deliberately unchanged. Coordinates come from the
independent 2022 raw audit and plurality parser; the shared coordinates were
checked against the 2023 User Guide. All value/target rules are delegated to
the existing canonical harmonizer, not reimplemented here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import yaml

import add_plurality_cdc2022 as plurality
import audit_cdc_2022_raw_coordinates as raw_audit
import preprocess_harmonize_cdc_2022_2023 as harmonizer


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/canonical/experiment_config_tae_canonical.yaml"
GUIDE_URL = (
    "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/"
    "Dataset_Documentation/DVS/natality/UserGuide{year}.pdf"
)
COORDINATES = {
    spec.canonical_name: (
        (plurality.DPLURAL_START, plurality.DPLURAL_END)
        if spec.canonical_name == "plurality"
        else (raw_audit.FIELD_BY_NAME[spec.source_2022].start,
              raw_audit.FIELD_BY_NAME[spec.source_2022].end)
    )
    for spec in harmonizer.FIELDS
}


def extract_source_row(payload: str, year: int) -> dict[str, str]:
    """Keep raw codes and the harmonizer's year-specific source aliases."""
    if year not in harmonizer.EXPECTED_ROWS:
        raise ValueError("Only the audited 2022 and 2023 layouts are supported")
    if len(payload) != raw_audit.EXPECTED_RECORD_LENGTH or not payload.isascii():
        raise ValueError("Expected exactly 1330 ASCII characters per record")
    row = {}
    for spec in harmonizer.FIELDS:
        start, end = COORDINATES[spec.canonical_name]
        row[spec.source_name(year)] = payload[start - 1:end].strip()
    if row["birth_year"] != str(year):
        raise ValueError("Record birth year does not match --year")
    return row


def harmonize_row(source: dict[str, str], year: int) -> dict[str, str]:
    row = {}
    for spec in harmonizer.FIELDS:
        value, error = harmonizer.canonicalize(spec, source[spec.source_name(year)])
        if error is not None:
            # Do not include record values in diagnostic logs.
            raise ValueError(f"Invalid value in canonical field {spec.canonical_name}")
        row[spec.canonical_name] = value
    return {**row, **harmonizer.derive_columns(row)}


def canonical_lock(year: int) -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rows = int(config["data"]["expected_rows"][str(year)])
    if rows != harmonizer.EXPECTED_ROWS[year]:
        raise ValueError("Canonical configuration and harmonizer row counts disagree")
    return {
        "expected_rows": rows,
        "expected_sha256": config["data"]["expected_sha256"][str(year)],
        "config_sha256": harmonizer.sha256_file(CONFIG),
    }


def reconstruct_year(raw_path: Path, output_dir: Path, year: int, *,
                     expected_rows: int, expected_sha256: str,
                     config_sha256: str, progress_every: int = 500_000) -> dict:
    """Write a new directory only after row, domain, and frozen-hash checks pass."""
    if year not in harmonizer.EXPECTED_ROWS:
        raise ValueError("Only the audited 2022 and 2023 layouts are supported")
    if expected_rows <= 0:
        raise ValueError("Expected row count must be positive")
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw TXT not found: {raw_path}")
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    filename = f"cdc_natality_{year}_harmonized.csv"
    with tempfile.TemporaryDirectory(prefix=".canonical_", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "release"
        staged.mkdir()
        csv_path = staged / filename
        raw_digest = hashlib.sha256()
        rows = 0
        with raw_path.open("rb") as raw, csv_path.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=harmonizer.OUTPUT_COLUMNS, lineterminator="\r\n")
            writer.writeheader()
            for rows, line in enumerate(raw, start=1):
                raw_digest.update(line)
                try:
                    payload = line.rstrip(b"\r\n").decode("ascii")
                    writer.writerow(harmonize_row(extract_source_row(payload, year), year))
                except (UnicodeError, ValueError) as exc:
                    if isinstance(exc, UnicodeError):
                        raise ValueError(f"Record {rows}: expected ASCII bytes") from None
                    raise ValueError(f"Record {rows}: {exc}") from None
                if progress_every and rows % progress_every == 0:
                    print(f"CDC {year}: processed {rows:,} records", flush=True)
        if rows != expected_rows:
            raise ValueError(f"CDC {year}: observed {rows:,} rows; expected {expected_rows:,}")
        output_hash = harmonizer.sha256_file(csv_path)
        if output_hash != expected_sha256:
            raise ValueError(
                f"CDC {year}: reconstructed CSV SHA-256 {output_hash} does not match "
                f"the frozen release {expected_sha256}; no output directory was published"
            )
        manifest = {
            "status": "PASS", "year": year, "rows": rows,
            "input_name": raw_path.name, "input_sha256": raw_digest.hexdigest(),
            "output_name": filename, "output_sha256": output_hash,
            "expected_output_sha256": expected_sha256,
            "canonical_config_sha256": config_sha256,
            "official_layout": GUIDE_URL.format(year=year),
            "record_length": raw_audit.EXPECTED_RECORD_LENGTH,
            "canonical_columns": list(harmonizer.OUTPUT_COLUMNS),
            "source_aliases": {spec.canonical_name: spec.source_name(year) for spec in harmonizer.FIELDS},
            "coordinates_1based_inclusive": COORDINATES,
            "code_sha256": {
                path.name: harmonizer.sha256_file(path)
                for path in [Path(__file__), Path(raw_audit.__file__),
                             Path(plurality.__file__), Path(harmonizer.__file__)]
            },
            "model_fitting": False,
        }
        harmonizer.write_schema_json(staged / "harmonized_schema.json")
        (staged / "extraction_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            raise FileExistsError(f"Output directory appeared during extraction: {output_dir}")
        staged.rename(output_dir)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=[2022, 2023], required=True)
    parser.add_argument("--input", type=Path, required=True, help="Uncompressed US public-use TXT")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwritten")
    args = parser.parse_args(argv)
    try:
        manifest = reconstruct_year(args.input, args.output_dir, args.year, **canonical_lock(args.year))
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"PASS: CDC {args.year}, {manifest['rows']:,} rows; frozen output SHA-256 matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
