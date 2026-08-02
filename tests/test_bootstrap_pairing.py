"""Invariant: every full-size bootstrap replicate pairs policies on identical records."""

import pytest


@pytest.mark.skip(reason="Expose bootstrap index generation and add a seeded miniature fixture.")
def test_paired_full_size_resamples():
    """Check shared multiplicities, full sample size, and top-B recomputation."""
