"""Invariant: early and full policies use one model and the same absolute top-B budget."""

import pytest


@pytest.mark.skip(reason="Refactor top-B selection into an importable helper and add tie fixtures.")
def test_equal_absolute_budget_selection():
    """Compare selected counts and deterministic tie handling for both policies."""
