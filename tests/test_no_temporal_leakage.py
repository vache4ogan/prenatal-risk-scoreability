"""Invariant: CDC 2023 is temporal test only and never fits models or calibrators."""

import pytest


@pytest.mark.skip(reason="Expose split provenance and provide miniature 2022/2023 fixtures.")
def test_cdc2023_not_used_for_fitting():
    """Assert training, calibration fitting, and threshold decisions use 2022 only."""
