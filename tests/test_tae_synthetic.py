#!/usr/bin/env python3
"""Small end-to-end smoke/invariant test for run_tae_synthetic.py.

Run from repository root after placing files in the suggested paths:
    python tests/test_tae_synthetic.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "experiments" / "run_tae_synthetic.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")

    source = script.read_text(encoding="utf-8")
    assert 'SYNTHETIC_SCRIPT_VERSION = "2026-08-22-v3-no-tabulate"' in source, (
        "Wrong run_tae_synthetic.py revision: expected v3-no-tabulate"
    )
    assert ".to_markdown(" not in source, (
        "Old pandas .to_markdown() call is still present; replace the script with v3"
    )

    with tempfile.TemporaryDirectory(prefix="tae_synth_test_") as td:
        td_path = Path(td)
        out_dir = td_path / "out"
        cfg_path = td_path / "smoke.yaml"

        cfg = {
            "simulation": {
                "seed": 92028,
                "replicates": 3,
                "population_n": 5000,
                "calibration_n": 3000,
                "progress_every": 0,
                "outcome_prevalence": 0.10,
                "outcome_risk_slope": 1.0,
                "quadrature_nodes": 60,
                "availability_fractions": [0.50, 0.80],
                "availability_associations": [
                    {"name": "higher_risk_less_available", "slope": -1.0},
                    {"name": "risk_independent", "slope": 0.0},
                    {"name": "higher_risk_more_available", "slope": 1.0},
                ],
                "model_qualities": [
                    {"name": "good", "noise_sd": 0.5},
                    {"name": "weak", "noise_sd": 2.0},
                ],
            },
            "evaluation": {
                "fractions": [0.10, 0.20],
                "scenarios": ["available_subset", "all_population"],
                "protocols": [
                    "cohort_specific_top_fraction",
                    "fixed_absolute_budget",
                    "fixed_calibration_threshold",
                ],
                "threshold_source": "independent_calibration_all_population",
                "threshold_comparator": ">=",
                "tie_breaker_top_b": "original_row_position_ascending",
                "primary_metric": "population_event_capture",
                "primary_contrast": "all_population_minus_available_subset",
                "primary_distortion": "cohort_specific_top_fraction_minus_fixed_absolute_budget",
            },
            "output": {
                "directory": str(out_dir),
                "overwrite": False,
                "save_replicate_level_results": True,
                "save_summaries": True,
                "save_manifest": True,
            },
        }
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        subprocess.run(
            [sys.executable, str(script), "--config", str(cfg_path)],
            cwd=repo,
            check=True,
        )

        required = [
            "synthetic_protocol_results.csv",
            "synthetic_protocol_comparisons.csv",
            "synthetic_decomposition.csv",
            "synthetic_condition_replicates.csv",
            "synthetic_comparison_summary.csv",
            "synthetic_decomposition_summary.csv",
            "synthetic_summary.md",
            "synthetic_experiment.locked.yaml",
            "synthetic_manifest.json",
        ]
        for name in required:
            p = out_dir / name
            assert p.exists() and p.stat().st_size > 0, f"Missing/empty output: {name}"

        protocol = pd.read_csv(out_dir / "synthetic_protocol_results.csv")
        decomp = pd.read_csv(out_dir / "synthetic_decomposition.csv")

        # 3 reps * 2 availability fractions * 3 associations * 2 qualities
        # * 2 q * 3 protocols * 2 scenarios
        assert len(protocol) == 3 * 2 * 3 * 2 * 2 * 3 * 2
        assert len(decomp) == 3 * 2 * 3 * 2 * 2

        fixed = protocol[protocol.protocol == "fixed_absolute_budget"]
        key = [
            "replicate",
            "availability_target_fraction",
            "availability_relation",
            "model_quality",
            "nominal_fraction",
        ]
        assert (fixed.groupby(key).selected_n.nunique() == 1).all()

        assert (decomp.tp_identity_error == 0).all()
        assert (decomp.identity_error.abs() <= 1e-12).all()
        assert (
            (
                decomp.distortion_top_fraction_minus_fixed_budget
                - decomp.capacity_effect_within_available_subset
            ).abs()
            <= 1e-12
        ).all()

    print("PASS: synthetic smoke test and invariants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
