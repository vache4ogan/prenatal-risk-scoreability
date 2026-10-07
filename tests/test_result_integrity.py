"""Invariant: reference aggregates retain hashes, row counts, and locked effects."""

import hashlib
from pathlib import Path
import reference_checks as verify_submission


def test_reference_hashes():
    root = Path(__file__).resolve().parents[1] / "results/reference"
    entries = (root / "SHA256SUMS").read_text().splitlines()
    assert len(entries) >= 29
    for entry in entries:
        digest, name = entry.split("  ", 1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name


def test_canonical_result_integrity():
    verify_submission.verify_shapley()
    verify_submission.verify_protocol_and_score_claims()
    verify_submission.verify_feature_sensitivities()
    verify_submission.verify_synthetic()
    verify_submission.verify_synthetic_specification()
    verify_submission.verify_temporal_2024()
