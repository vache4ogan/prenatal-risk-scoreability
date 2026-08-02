#!/usr/bin/env python3
"""
Safely add the CDC 2022 DPLURAL field to an existing parsed CSV.

Official CDC field:
    DPLURAL — Plurality Recode
    fixed-width position 454 (1-based, inclusive)
    1 = Single
    2 = Twin
    3 = Triplet
    4 = Quadruplet or higher

Safety properties:
- streams both files; does not load 3.7M rows into RAM;
- never modifies the source CSV by default;
- compares CSV and TXT row counts exactly;
- validates every extracted plurality code;
- writes to a temporary file first and atomically renames it;
- can optionally replace the original CSV with an automatic backup;
- writes a JSON manifest with counts and SHA-256 checksums.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TextIO


EXPECTED_ROWS = 3_676_029
EXPECTED_RECORD_LENGTH = 1330

# CDC positions are 1-based inclusive.
DPLURAL_START = 454
DPLURAL_END = 454
VALID_DPLURAL_CODES = {"1", "2", "3", "4"}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def extract_plurality(raw_line: str) -> str:
    """
    Extract DPLURAL from official CDC position 454.

    Python slice:
        position 454 -> raw_line[453:454]
    """
    line = raw_line.rstrip("\r\n")

    if len(line) < DPLURAL_END:
        raise ValueError(
            f"Raw record is too short: {len(line)} characters; "
            f"position {DPLURAL_END} is required."
        )

    return line[DPLURAL_START - 1:DPLURAL_END].strip()


def make_default_output(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_with_plurality{csv_path.suffix}")


def unique_backup_path(path: Path) -> Path:
    candidate = path.with_suffix(path.suffix + ".bak")
    counter = 1
    while candidate.exists():
        candidate = path.with_suffix(path.suffix + f".bak{counter}")
        counter += 1
    return candidate


def open_text(path: Path, mode: str) -> TextIO:
    return path.open(mode, encoding="utf-8", newline="")


def add_plurality(
    csv_path: Path,
    raw_txt_path: Path,
    output_path: Path,
    *,
    in_place: bool,
    replace_existing: bool,
    expected_rows: int | None,
    progress_every: int,
) -> dict[str, object]:
    started_at = time.time()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if not raw_txt_path.exists():
        raise FileNotFoundError(f"Raw TXT not found: {raw_txt_path}")

    if csv_path.resolve() == raw_txt_path.resolve():
        raise ValueError("CSV and raw TXT paths point to the same file.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Temporary file must be in the same directory for atomic os.replace().
    temp_path = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}"
    )
    manifest_path = output_path.with_suffix(output_path.suffix + ".plurality_manifest.json")

    if temp_path.exists():
        temp_path.unlink()

    plurality_counts: Counter[str] = Counter()
    padded_raw_records = 0
    exact_length_raw_records = 0
    short_raw_records = 0
    csv_rows = 0
    raw_rows = 0
    backup_path: Path | None = None

    source_csv_sha256 = sha256_file(csv_path)
    raw_txt_sha256 = sha256_file(raw_txt_path)

    try:
        with open_text(csv_path, "r") as csv_file, \
             open_text(raw_txt_path, "r") as raw_file, \
             open_text(temp_path, "w") as output_file:

            reader = csv.reader(csv_file)
            writer = csv.writer(output_file)

            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError("Source CSV is empty.") from exc

            header = [column.strip() for column in header]

            if len(header) != len(set(header)):
                duplicates = sorted(
                    column for column in set(header)
                    if header.count(column) > 1
                )
                raise ValueError(f"Duplicate CSV columns: {duplicates}")

            plurality_exists = "plurality" in header

            if plurality_exists and not replace_existing:
                raise ValueError(
                    "Column 'plurality' already exists. "
                    "Use --replace-existing only if you intentionally want "
                    "to overwrite it from the official raw TXT."
                )

            if plurality_exists:
                plurality_index = header.index("plurality")
                output_header = header
            else:
                plurality_index = len(header)
                output_header = header + ["plurality"]

            writer.writerow(output_header)

            while True:
                try:
                    csv_row = next(reader)
                    csv_finished = False
                except StopIteration:
                    csv_row = None
                    csv_finished = True

                raw_line = raw_file.readline()
                raw_finished = raw_line == ""

                if csv_finished and raw_finished:
                    break

                if csv_finished and not raw_finished:
                    # Count remaining raw lines for a precise error message.
                    extra_raw = 1 + sum(1 for _ in raw_file)
                    raise ValueError(
                        f"Raw TXT has {extra_raw:,} more data rows than CSV."
                    )

                if raw_finished and not csv_finished:
                    # Count current plus remaining CSV rows.
                    extra_csv = 1 + sum(1 for _ in reader)
                    raise ValueError(
                        f"CSV has {extra_csv:,} more data rows than raw TXT."
                    )

                assert csv_row is not None

                csv_rows += 1
                raw_rows += 1

                raw_without_newline = raw_line.rstrip("\r\n")
                raw_length = len(raw_without_newline)

                if raw_length < EXPECTED_RECORD_LENGTH:
                    short_raw_records += 1
                elif raw_length == EXPECTED_RECORD_LENGTH:
                    exact_length_raw_records += 1
                else:
                    trailing = raw_without_newline[EXPECTED_RECORD_LENGTH:]
                    if trailing.strip(" \t\x00") == "":
                        padded_raw_records += 1
                    else:
                        raise ValueError(
                            f"Row {raw_rows:,}: nonblank content exists after "
                            f"official position {EXPECTED_RECORD_LENGTH}: "
                            f"{trailing[:80]!r}"
                        )

                plurality = extract_plurality(raw_line)

                if plurality not in VALID_DPLURAL_CODES:
                    raise ValueError(
                        f"Row {raw_rows:,}: invalid DPLURAL value "
                        f"{plurality!r} at official position 454."
                    )

                plurality_counts[plurality] += 1

                if plurality_exists:
                    if len(csv_row) != len(header):
                        raise ValueError(
                            f"CSV row {csv_rows:,} has {len(csv_row)} fields; "
                            f"header has {len(header)}."
                        )
                    csv_row[plurality_index] = plurality
                    output_row = csv_row
                else:
                    if len(csv_row) != len(header):
                        raise ValueError(
                            f"CSV row {csv_rows:,} has {len(csv_row)} fields; "
                            f"header has {len(header)}."
                        )
                    output_row = csv_row + [plurality]

                writer.writerow(output_row)

                if progress_every and csv_rows % progress_every == 0:
                    elapsed = time.time() - started_at
                    print(
                        f"Processed {csv_rows:,} rows | "
                        f"singletons={plurality_counts['1']:,} | "
                        f"elapsed={elapsed:,.1f}s"
                    )

            output_file.flush()
            os.fsync(output_file.fileno())

        if csv_rows != raw_rows:
            raise ValueError(
                f"Internal row-count mismatch: CSV={csv_rows:,}, "
                f"TXT={raw_rows:,}."
            )

        if expected_rows is not None and csv_rows != expected_rows:
            raise ValueError(
                f"Observed {csv_rows:,} rows, expected {expected_rows:,}."
            )

        if sum(plurality_counts.values()) != csv_rows:
            raise ValueError(
                "Plurality distribution does not sum to the total row count."
            )

        # Commit only after every row has passed.
        if in_place:
            backup_path = unique_backup_path(csv_path)
            shutil.copy2(csv_path, backup_path)
            os.replace(temp_path, csv_path)
            final_path = csv_path
        else:
            if output_path.exists():
                raise FileExistsError(
                    f"Output already exists: {output_path}. "
                    "Choose another --output path or remove it explicitly."
                )
            os.replace(temp_path, output_path)
            final_path = output_path

        final_sha256 = sha256_file(final_path)

        manifest: dict[str, object] = {
            "status": "PASS",
            "source_csv": str(csv_path.resolve()),
            "raw_txt": str(raw_txt_path.resolve()),
            "output_csv": str(final_path.resolve()),
            "backup_csv": str(backup_path.resolve()) if backup_path else None,
            "source_csv_sha256": source_csv_sha256,
            "raw_txt_sha256": raw_txt_sha256,
            "output_csv_sha256": final_sha256,
            "rows_written": csv_rows,
            "plurality_field": {
                "cdc_name": "DPLURAL",
                "official_position_1_based_inclusive": "454-454",
                "python_slice": "[453:454]",
                "codes": {
                    "1": "Single",
                    "2": "Twin",
                    "3": "Triplet",
                    "4": "Quadruplet or higher",
                },
            },
            "plurality_counts": {
                "1_singleton": plurality_counts["1"],
                "2_twin": plurality_counts["2"],
                "3_triplet": plurality_counts["3"],
                "4_quadruplet_or_higher": plurality_counts["4"],
            },
            "record_lengths": {
                "exact_1330": exact_length_raw_records,
                "shorter_than_1330": short_raw_records,
                "harmless_blank_or_nul_padding_after_1330": padded_raw_records,
            },
            "elapsed_seconds": round(time.time() - started_at, 3),
        }

        manifest_path = final_path.with_suffix(
            final_path.suffix + ".plurality_manifest.json"
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n=== PLURALITY COLUMN ADDED SUCCESSFULLY ===")
        print(f"Rows written: {csv_rows:,}")
        print(f"Singleton (1): {plurality_counts['1']:,}")
        print(f"Twin (2): {plurality_counts['2']:,}")
        print(f"Triplet (3): {plurality_counts['3']:,}")
        print(
            "Quadruplet or higher (4): "
            f"{plurality_counts['4']:,}"
        )
        print(f"Output: {final_path}")
        if backup_path:
            print(f"Backup: {backup_path}")
        print(f"Manifest: {manifest_path}")
        print(f"Output SHA-256: {final_sha256}")

        return manifest

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely add CDC 2022 DPLURAL (position 454) "
            "to an existing parsed CSV."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "data/cdc2022_checked/cdc_natality_2022_official.csv"
        ),
        help="Existing parsed CDC 2022 CSV.",
    )
    parser.add_argument(
        "--raw-txt",
        type=Path,
        default=Path(
            "data/cdc2022_checked/"
            "Nat2022PublicUS.c20230504.r20230822.txt"
        ),
        help="Official raw fixed-width CDC 2022 TXT.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "New output CSV. By default, creates "
            "<source>_with_plurality.csv."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Atomically replace the source CSV after creating an automatic "
            ".bak backup. Without this flag, the source is never modified."
        ),
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help=(
            "Allow replacing an existing 'plurality' column from raw TXT."
        ),
    )
    parser.add_argument(
        "--no-expected-row-check",
        action="store_true",
        help=(
            "Do not require the official 3,676,029 occurrence-file rows. "
            "Useful only for deliberately truncated test files."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500_000,
        help="Progress logging interval; use 0 to disable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.in_place and args.output is not None:
        print(
            "ERROR: use either --in-place or --output, not both.",
            file=sys.stderr,
        )
        return 2

    output_path = (
        args.csv
        if args.in_place
        else args.output or make_default_output(args.csv)
    )

    try:
        add_plurality(
            csv_path=args.csv,
            raw_txt_path=args.raw_txt,
            output_path=output_path,
            in_place=args.in_place,
            replace_existing=args.replace_existing,
            expected_rows=(
                None
                if args.no_expected_row_check
                else EXPECTED_ROWS
            ),
            progress_every=args.progress_every,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
