"""Invariant: monotone Platt calibration preserves LightGBM score ranking and top-B sets."""

import pytest


@pytest.mark.skip(reason="Expose calibration/ranking helpers and add tied-score fixtures.")
def test_platt_calibration_preserves_ranking():
    """Compare raw and calibrated orderings and selected record identities."""
