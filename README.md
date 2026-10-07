# Eligibility and Capacity in Clinical Risk Evaluation

Experiments for **Same Top Fraction, Different Workload: Auditing Eligibility and
Capacity in Clinical Risk Evaluation** ([TAE 2026 paper](https://openreview.net/forum?id=puYbvC47rw)).
The four-cell audit separates candidate-set availability from absolute selection
capacity using one frozen score vector, symmetric Shapley effects, and paired
full-size bootstrap resamples. This is research code, not a clinical decision tool.

## Install

Use Python 3.12 and run commands from the repository root. No external service account is needed.

```bash
python -m venv .venv
```

Activate the environment before installing: `source .venv/bin/activate` on Linux/macOS,
or `.\.venv\Scripts\Activate.ps1` in PowerShell. Then install the pinned CPU dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data

Download and unzip the **US** CDC/NCHS natality public-use files, not the territory
files, following the [official natality documentation](https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm).
Individual records are not distributed here. Substitute your downloaded TXT filenames below.

```bash
python scripts/preprocessing/extract_canonical_cdc.py --year 2022 --input data/raw/Nat2022PublicUS.txt --output-dir data/reconstructed_2022
python scripts/preprocessing/extract_canonical_cdc.py --year 2023 --input data/raw/Nat2023PublicUS.txt --output-dir data/reconstructed_2023
python scripts/preprocessing/harmonize_cdc_2024.py --input data/raw/Nat2024PublicUS.txt --output data/reconstructed_2024/harmonized_cdc_2024.csv
```

The 2022/2023 extractor checks 1,330-character records, domains, row counts, and
the exact harmonized CSV hashes in `configs/canonical/experiment_config_tae_canonical.yaml`.
It publishes a new directory only after those checks pass; do not harmonize its output again.
The 2024 route preserves the separate evaluation schema; evaluation verifies its hash.

The population is US-resident singleton births with each target observed.
Preterm uses obstetric-estimate gestation (`OEGest`), never `COMBGEST`; other
targets are NICU admission and birth weight below 2,500 g. Unknown targets remain
missing. Early entry is prenatal care in months 1-3; 0 means no care, not early entry.
The model uses nine baseline-oriented birth-certificate variables. PRIORTERM may
include the current pregnancy; prior live/dead births are assessed at delivery.
A sensitivity analysis excluding PRIORTERM is available.

## Train

Training and preprocessing fit only CDC 2022 training rows. A shared seed-2026
80/20 split reserves calibration rows for the 5% and 10% cutoffs.

```bash
python scripts/experiments/fit_canonical_local.py --train-csv data/reconstructed_2022/cdc_natality_2022_harmonized.csv --output-dir models/local_canonical
```

This optional refit produces three model bundles and `thresholds_2022.csv`;
checkpoint replay below uses the released models without retraining.
For sensitivities, pass `--config configs/canonical/experiment_config_noprior.yaml`
or `--config configs/canonical/experiment_config_no_race.yaml` and use a separate output directory.

## Evaluate

Use the released native models with their original calibration cutoffs:

```bash
python scripts/experiments/run_temporal_four_cell.py --csv data/reconstructed_2023/cdc_natality_2023_harmonized.csv --models pretrained --thresholds results/reference/2023/thresholds_2022.csv --year 2023 --expected-sha256 82b38b24d2f795a4a5233ce3a2fcb43b70789339306ba9044bde56f50bb270a8 --output-dir results/local_2023
python scripts/experiments/run_temporal_four_cell.py --csv data/reconstructed_2024/harmonized_cdc_2024.csv --models pretrained --thresholds results/reference/2023/thresholds_2022.csv --year 2024 --expected-sha256 65d04e37c416e09a3004a4dae159e600de0a20d2fb873661ba63f12a2163cd92 --output-dir results/local_2024
```

For locally trained bundles, replace both `--models` and `--thresholds` with
matching outputs from that fit. Load joblib files only from a trusted source.
For sensitivity models also pass `--feature-set no_priorterm` or `--feature-set no_race`.

Each run emits 24 four-cell estimates, six decompositions, three-protocol results for all care scenarios, threshold transport,
and 500 shared, full-size bootstrap replicates per target. Observed absolute
budgets remain fixed across resamples; ties use original row order; model matrices
are float32 and race codes retain their numeric training representation.
`--bootstrap-replicates 0` computes point estimates only. The runner never refits.
Exact and six-decimal cutoffs are labelled separately; use exact-cutoff results.

Reference aggregates are under `results/reference/{2023,2024,synthetic,sensitivity}`.
`results/reference/SHA256SUMS` authenticates their bytes. No patient-level predictions
are included. Scientific definitions and seeds are in `configs/canonical/`.

## Synthetic

The full design has 81 conditions and 100 independent simulation replicates.

```bash
python scripts/experiments/run_tae_synthetic_shapley81.py --config configs/canonical/synthetic_experiment_shapley81.yaml --output-dir results/local_synthetic
```

Add `--smoke-test` for two small replicates over the same 81 conditions.
Synthetic intervals describe independent simulation replicates, not clinical bootstrap resampling.

## Tests

```bash
python -m pytest -q
```

Tests use fictional fixed-width records and synthetic scores, and verify reference
aggregates, categories, float32 transforms, tie order, paired bootstrap, and threshold precision.

## Cite

Use [CITATION.cff](CITATION.cff) for authors and the associated publication.

## License

Code and the three released frozen models are [MIT licensed](LICENSE).
CDC data and the associated article retain their separate terms; see
[CDC/NCHS data-use restrictions](https://www.cdc.gov/nchs/data_access/restrictions.htm).
Dependency licenses remain separate.
