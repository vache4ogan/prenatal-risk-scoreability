"""Prenatal-care grouping invariant: 0 is none; 1, 2, 3 are months 1–3; 4 and valid later months are after month 3; NaN/blank is unknown. Source code 99 must be normalized to missing before grouping."""

import pytest


@pytest.mark.skip(reason="Extract care-entry normalization/grouping and add raw-code fixtures.")
def test_canonical_care_entry_groups():
    """Verify 0, months 1–3, valid later months, blank/NaN, and source code 99."""
