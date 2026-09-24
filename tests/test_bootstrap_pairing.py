"""Invariant: every full-size bootstrap replicate pairs policies on identical records."""

import numpy as np
from run_temporal_four_cell import bootstrap
import run_tae_shapley_fixed_b_bootstrap_clearml as audit


def test_paired_full_size_resamples():
    n, seed = 1000, 27
    y = (np.arange(n) % 4 == 0).astype(np.int8)
    scores = (np.arange(n) % 13) / 13
    care = np.where(np.arange(n) < 700, 2, 5)
    _, _, lookup = audit.four_cell_point_estimates(target="test", y=y, probabilities=scores, care_month=care)
    actual = bootstrap(y, scores, care, lookup, repeats=3, seed=seed, target="test", year=2024)
    rng = np.random.default_rng(seed)
    for rep in range(3):
        sampled = rng.integers(0, n, size=n, dtype=np.int32)
        expanded = np.repeat(np.arange(n), np.bincount(sampled, minlength=n))
        assert len(expanded) == n
        available = expanded[care[expanded] <= 3]
        early_rank = available[np.lexsort((available, -scores[available]))]
        all_rank = expanded[np.lexsort((expanded, -scores[expanded]))]
        for row in [r for r in actual if r["replicate"] == rep]:
            assert row["full_event_n"] == y[expanded].sum()
            for name, rank, b in [("C00", early_rank, row["B_early"]),
                                  ("C01", early_rank, row["B_all"]),
                                  ("C10", all_rank, row["B_early"]),
                                  ("C11", all_rank, row["B_all"])]:
                assert row[f"{name}_tp"] == y[rank[:b]].sum()
