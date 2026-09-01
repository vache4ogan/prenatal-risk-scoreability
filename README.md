# Same Top Fraction, Different Workload

Code and reproducibility artifacts for the TAE 2026 submission **"Same Top
Fraction, Different Workload: Auditing Eligibility and Capacity in Clinical
Risk Evaluation."** The exact submitted repository snapshot is preserved by
the Git tag `tae-2026-submitted`.

## Repository layout

- `paper/`: final anonymous manuscript package, compiled PDF, aggregate source
  data, figure and table generators, and the package verifier.
- `configs/canonical/`: locked configurations for the final experiments.
- `scripts/preprocessing/`: CDC Natality extraction, harmonization, timing
  audit, and 2024 validation code.
- `scripts/experiments/`: clinical evaluation, Shapley decomposition,
  sensitivities, temporal replication, and the 81-condition synthetic study.
- `metadata/`: cohort, feature-timing, schema, and target-definition metadata.
- `results/`: retained final run artifacts and sensitivity outputs.
- `tests/`: data-independent checks of selection, cohort, bootstrap,
  calibration, and leakage invariants.

Raw individual-level Natality records, fitted models, predictions, credentials,
and private service addresses are not stored in this repository.

## Verify the submitted package

Use Python 3.10-3.12 for the repository code. The paper verifier also works in
Python 3.9.

```bash
python -m pip install -e ".[dev]"
python paper/verify_submission.py
python -m pytest -q
```

To regenerate manuscript artifacts:

```bash
cd paper
python -m pip install -r requirements_anonymous.txt
python generate_figures.py
python generate_tables.py
tectonic main.tex --outdir build --keep-logs
python verify_submission.py
```

## Re-run the experiments

The clinical experiments require the NCHS Natality public-use files and the
locked ClearML dataset lineage. Configure access through an untracked `.env`
based on `.env.example`; never commit credentials.

```bash
python scripts/experiments/run_tae_canonical_clearml.py \
  --config configs/canonical/experiment_config_tae_canonical.yaml \
  --preflight-only

python scripts/experiments/run_tae_shapley_fixed_b_bootstrap_clearml.py \
  --smoke-test

python scripts/experiments/run_tae_synthetic_shapley81.py \
  --config configs/canonical/synthetic_experiment_shapley81.yaml \
  --overwrite
```

Exact package versions, locked configurations, manifests, aggregate outputs,
and the final PDF are retained in `results/` and `paper/`.
