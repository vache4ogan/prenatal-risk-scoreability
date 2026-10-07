"""Prenatal-care grouping invariant: 0 is none; 1, 2, 3 are months 1–3; 4 and valid later months are after month 3; NaN/blank is unknown. Source code 99 must be normalized to missing before grouping."""

import pytest
import numpy as np
from run_tae_canonical_clearml import availability_masks


def test_canonical_care_entry_groups():
    masks = availability_masks(np.array([0, 1, 2, 3, 4, 10, np.nan]))
    assert np.flatnonzero(masks["early_entry"]).tolist() == [1, 2, 3]
    assert np.flatnonzero(masks["any_prenatal_care"]).tolist() == [1, 2, 3, 4, 5]
    assert np.flatnonzero(masks["no_care"]).tolist() == [0]
    assert np.flatnonzero(masks["unknown_care"]).tolist() == [6]
    assert masks["all_record_upper_bound"].all()
    with pytest.raises(ValueError):
        availability_masks(np.array([1, 99]))
