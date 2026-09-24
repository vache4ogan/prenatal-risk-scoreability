# Same Top Fraction, Different Workload

Code and reproducibility artifacts for the accepted TAE 2026 workshop paper **"Same Top
Fraction, Different Workload: Auditing Eligibility and Capacity in Clinical
Risk Evaluation."**

**Vache Oganisyan, Dmitry Lvov, Ilya Pershin.** Poster at
TAE: Can We Trust AI Evaluation?, a non-archival NeurIPS 2026 workshop.

[Paper and decision](https://openreview.net/forum?id=puYbvC47rw) |
[Camera-ready PDF](paper/main.pdf) |
[Source package instructions](paper/SUBMISSION_README.md)

The audit crosses two candidate populations with two absolute selection
budgets, holding the score vector and full-population event denominator fixed.
It distinguishes broader eligibility from additional service capacity.
The reviewed snapshot is preserved by the Git tag `tae-2026-submitted`.

## Repository layout

- `paper/`: camera-ready manuscript, compiled PDF, aggregate source
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

Individual-level Natality records, fitted models, predictions, and credentials
are excluded from this repository. Original run manifests retain provenance;
private ClearML identifiers are references, not access credentials.

## Quick verification

Use Python 3.12 in a virtual environment for model replay and repository tests.
The locked core versions below match the archived canonical training run.

```bash
python -m pip install -r requirements-replication.txt
python paper/verify_submission.py
python -m pytest -q
```

These tests execute selection, shared-resample, preprocessing, categorical-code,
calibration-ordering, and artifact checks. They require neither raw CDC records
nor access to ClearML. The former disabled placeholders have been replaced by
executable tests.

## Reproduce the simulation

```bash
python paper/analysis_code/synthetic/run_tae_synthetic_shapley81.py \
  --config paper/analysis_code/configs/synthetic_experiment_shapley81.yaml \
  --output-dir results/local_synthetic
python paper/verify_synthetic_rerun.py results/local_synthetic
```

The full experiment has 81 conditions and 100 paired repetitions. Comparison
against the archived summaries and 8,100 replicate rows is automatic. Use a
new output directory for each run.

## Rebuild the paper

For byte-compatible legacy figure generation, use a separate Python 3.9-3.11
environment with `paper/requirements_anonymous.txt`. This rendering environment
is separate from the canonical model-replay environment above.

```bash
cd paper
python -m pip install -r requirements_anonymous.txt
python generate_figures.py
python generate_tables.py
tectonic main.tex --outdir build --keep-logs
```

Copy `build/main.pdf` to `main.pdf`, then run
`python verify_submission.py --skip-hashes`. The original delivery hashes apply
before regeneration; generated PDF metadata may differ. The official NeurIPS
2026 style is included without modification.

## Clinical experiments without ClearML

Obtain the [NCHS Natality public-use files](https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm)
under the NCHS Data User Agreement. Extraction and harmonization code is in
`scripts/preprocessing/`. The frozen harmonized file hashes and schema are
documented in `configs/canonical/` and `metadata/harmonization/`.

The local entrypoints use local files and never create or update ClearML tasks:

```bash
python scripts/experiments/fit_canonical_local.py \
  --train-csv data/cdc_temporal_harmonized/cdc_natality_2022_harmonized.csv \
  --output-dir models/local_canonical
python scripts/experiments/run_temporal_four_cell.py \
  --csv data/cdc_2024/harmonized_cdc_2024.csv \
  --models models/local_canonical \
  --thresholds models/local_canonical/thresholds_2022.csv \
  --year 2024 --bootstrap-replicates 500 \
  --expected-sha256 65d04e37c416e09a3004a4dae159e600de0a20d2fb873661ba63f12a2163cd92 \
  --output-dir results/local_2024
```

The same replay command supports `--year 2023` with the corresponding canonical
CSV and hash. Existing frozen model bundles can be used instead of refitting.
Individual scores are saved only with explicit `--save-scores` and remain in an
ignored `predictions/` directory. Archived remote-training entrypoints remain
available for authors with ClearML access; `.env.example` contains no secrets.

## Reproduction scope and correction

The camera-ready preparation independently replayed the frozen models on CDC
2023: all 24 four-cell point estimates matched the reviewed results exactly.
The 81-condition synthetic grid and every manuscript figure and table were
also regenerated. These checks distinguish frozen-model replay from a full
raw-file-to-training rebuild; the latter has its own data acquisition and
harmonization requirements.

The old CDC 2024 threshold runner read numeric MRACE31 codes as strings,
silently disabling their one-hot categories. Its historical outputs in
`results/tae_2026_replication/` are retained for provenance and are superseded.
The corrected runner uses the canonical numeric representation; a regression
test covers the failure. Main CDC 2023 estimates and the synthetic experiment
are unaffected. See `results/README.md` for the current artifact locations.

This is evaluation research code, not a deployed prenatal-care decision tool.
No private credentials, clinical records, or fitted clinical model are required
for the aggregate and synthetic verification routes.
