"""Full-precision CDC 2022 q=10% cutoff validation."""
from pathlib import Path
import numpy as np
import pandas as pd

TARGETS = ("target_preterm", "target_nicu", "target_lbw")


def load_cutoffs(path: Path) -> dict[str, float]:
    rows = pd.read_csv(path)
    rows = rows[
        (rows["protocol"] == "fixed_calibration_threshold")
        & np.isclose(rows["nominal_fraction"], 0.10)
        & (rows["availability_scenario"] == "all_record_upper_bound")
    ]
    if len(rows) != len(TARGETS) or set(rows["target"]) != set(TARGETS):
        raise ValueError("Canonical q=10% threshold rows must contain each target exactly once")
    thresholds = pd.to_numeric(rows["threshold"], errors="raise")
    if not np.isfinite(thresholds).all() or not thresholds.between(0, 1).all():
        raise ValueError("Canonical thresholds must be finite probabilities")
    return {
        row.target: float(row.threshold)
        for row in rows.itertuples(index=False)
    }
