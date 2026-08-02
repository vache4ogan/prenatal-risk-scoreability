"""Invariant: canonical tables match manifest row counts, hashes, and locked effects."""

import pytest


@pytest.mark.skip(reason="Add a fixture resolving canonical tables and sanitized manifest paths.")
def test_canonical_result_integrity():
    """Verify expected rows, output hashes, and the three 10% temporal-test effects."""
