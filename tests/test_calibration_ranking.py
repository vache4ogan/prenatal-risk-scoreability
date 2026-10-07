"""Invariant: monotone Platt calibration preserves LightGBM score ranking and top-B sets."""

import numpy as np
from run_article_revision_analyses_clearml import apply_platt, rank_eligible_indices


def test_platt_calibration_preserves_ranking():
    raw = np.array([0.3, 0.1, 0.3, 0.9, 0.7, 0.7])
    calibrated = apply_platt(raw, intercept=-0.7, slope=1.4, eps=1e-12)
    mask = np.ones(len(raw), dtype=bool)
    assert np.array_equal(rank_eligible_indices(raw, mask), rank_eligible_indices(calibrated, mask))
