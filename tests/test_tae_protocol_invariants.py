#!/usr/bin/env python3
"""Fast invariant tests for run_tae_canonical_clearml.py.

These tests do not require CDC data or ClearML. They test the scientific
selection logic on synthetic data before any expensive remote run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
CANDIDATES = [
    HERE.with_name("run_tae_canonical_clearml.py"),
    HERE.parents[1] / "scripts" / "experiments" / "run_tae_canonical_clearml.py",
]
SCRIPT = next((candidate for candidate in CANDIDATES if candidate.is_file()), CANDIDATES[-1])
spec = importlib.util.spec_from_file_location("tae", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {SCRIPT}")
tae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tae)


def make_data(n: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    y = (latent + rng.normal(scale=1.1, size=n) > 1.0).astype(np.int8)
    score = 1.0 / (1.0 + np.exp(-(latent + rng.normal(scale=0.4, size=n))))
    care = rng.choice(
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan]),
        size=n,
        p=np.array([0.02, 0.17, 0.27, 0.25, 0.08, 0.055, 0.045, 0.035, 0.025, 0.02, 0.01, 0.020]),
    )
    return y, score, care


def main() -> int:
    tae.validate_config(tae.DEFAULT_CONFIG)
    y, score, care = make_data()
    masks = tae.availability_masks(care)

    assert np.all(~masks["early_entry"] | masks["any_prenatal_care"])
    assert np.all(~masks["any_prenatal_care"] | masks["all_record_upper_bound"])
    assert not np.any(masks["any_prenatal_care"] & masks["no_care"])
    assert not np.any(masks["any_prenatal_care"] & masks["unknown_care"])

    calibration = score[:1500]
    thresholds = {q: tae.threshold_for_capacity(calibration, q) for q in (0.05, 0.10)}
    rows, _ = tae.evaluate_protocols(
        target="target_test",
        y=y,
        probabilities=score,
        care_month=care,
        fractions=[0.05, 0.10],
        thresholds=thresholds,
    )
    assert len(rows) == 18

    comparisons = tae.protocol_comparison_rows(rows)
    assert len(comparisons) == 18

    decomposition = tae.effect_decomposition_rows(
        target="target_test",
        y=y,
        probabilities=score,
        care_month=care,
        fractions=[0.05, 0.10],
    )
    assert len(decomposition) == 2
    assert max(abs(float(r["identity_error"])) for r in decomposition) <= 1e-15

    replicate_rows, summary = tae.bootstrap_protocol_comparisons(
        target="target_test",
        y=y,
        probabilities=score,
        care_month=care,
        fractions=[0.05, 0.10],
        thresholds=thresholds,
        point_comparisons=comparisons,
        replicates=25,
        seed=2027,
        confidence_level=0.95,
        progress_every=0,
    )
    assert len(replicate_rows) == 25 * 18
    assert len(summary) == 18

    # Fixed-budget delta_selected must be zero in every point comparison.
    fixed = [r for r in comparisons if r["protocol"] == "fixed_absolute_budget"]
    assert fixed and all(int(r["delta_selected_n"]) == 0 for r in fixed)

    # Common threshold must be identical across scenarios for the same q.
    for q in (0.05, 0.10):
        vals = {
            float(r["threshold"])
            for r in rows
            if r["protocol"] == "fixed_calibration_threshold"
            and float(r["nominal_fraction"]) == q
        }
        assert len(vals) == 1

    print("ALL TAE PROTOCOL INVARIANT TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
