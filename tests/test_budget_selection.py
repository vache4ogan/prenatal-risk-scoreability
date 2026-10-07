"""Invariant: early and full policies use one model and the same absolute top-B budget."""

import numpy as np
import run_tae_canonical_clearml as canonical
import run_tae_shapley_fixed_b_bootstrap_clearml as audit


def test_equal_absolute_budget_selection():
    y = (np.arange(1000) % 7 == 0).astype(np.int8)
    scores = (np.arange(1000) % 5) / 5
    care = np.where(np.arange(1000) < 700, 2, 5)
    rows, _, _ = audit.four_cell_point_estimates(target="test", y=y, probabilities=scores, care_month=care)
    for q in [0.05, 0.1]:
        cells = {r["cell"]: r for r in rows if r["nominal_fraction"] == q}
        assert cells["C00"]["selected_n"] == cells["C10"]["selected_n"] == round(q * 700)
        assert cells["C01"]["selected_n"] == cells["C11"]["selected_n"] == round(q * 1000)
    order = canonical.rank_eligible_indices(np.array([0.5, 0.9, 0.9, 0.9]), np.array([1, 1, 0, 1], bool))
    assert order.tolist() == [1, 3, 0]
