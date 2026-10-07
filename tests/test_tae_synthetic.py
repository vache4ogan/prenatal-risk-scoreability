"""Small 81-condition end-to-end synthetic experiment and accounting identities."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_synthetic_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "simulation"
    subprocess.run([
        sys.executable, "-B", str(root / "scripts/experiments/run_tae_synthetic_shapley81.py"),
        "--config", str(root / "configs/canonical/synthetic_experiment_shapley81.yaml"),
        "--output-dir", str(output), "--smoke-test",
    ], cwd=root, check=True)
    protocol = pd.read_csv(output / "synthetic_protocol_results.csv")
    decomposition = pd.read_csv(output / "synthetic_shapley_replicates.csv")
    summary = pd.read_csv(output / "synthetic_condition_summary.csv")
    assert len(summary) == 81
    assert len(protocol) == 2 * 81 * 6
    assert len(decomposition) == 2 * 81
    keys = ["replicate", "availability_target_fraction", "availability_relation",
            "model_quality", "nominal_fraction"]
    fixed = protocol[protocol.protocol == "fixed_absolute_budget"]
    assert (fixed.groupby(keys).selected_n.nunique() == 1).all()
    common = protocol[protocol.protocol == "fixed_calibration_threshold"]
    assert (common.groupby(keys).threshold.nunique() == 1).all()
    assert (decomposition.tp_identity_error == 0).all()
    assert (decomposition.tp_factorial_identity_error == 0).all()
    for column in ["identity_error", "factorial_identity_error", "shapley_identity_error"]:
        assert (decomposition[column].abs() <= 1e-12).all()
    np.testing.assert_allclose(
        decomposition.shapley_capacity_effect + decomposition.shapley_availability_effect,
        decomposition.total_four_cell_effect, rtol=0, atol=1e-12,
    )
    assert not list(output.glob("*.md"))
