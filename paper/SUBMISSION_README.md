# TAE 2026 camera-ready source and reproducibility package

This package accompanies the accepted poster paper "Same Top Fraction, Different Workload: Auditing Eligibility and Capacity in Clinical Risk Evaluation" by Vache Oganisyan, Dmitry Lvov, and Ilya Pershin. TAE is a non-archival NeurIPS 2026 workshop.

## Contents

- `main.tex`, `main.pdf`, `references.bib`, and `neurips_2026.sty`: manuscript source and compiled paper.
- `source_data/`: aggregate, non-record-level inputs behind every manuscript figure and table.
- `generate_figures.py` and `generate_tables.py`: deterministic manuscript generators.
- `analysis_code/synthetic/`: source for the 81-condition controlled experiment.
- `analysis_code/temporal/`: sanitized CDC 2024 score-cutoff transport script.
- `source_data/temporal_2024/`: corrected full four-cell replication, 3,000 paired-resample rows, threshold transport at both precisions, and a hash manifest.
- `analysis_code/synthetic_outputs/`: archived replicate-level outputs from that experiment.
- `verify_submission.py`: numerical, citation, layout, and manuscript-claim checks.
- `verify_synthetic_rerun.py`: comparison of a fresh simulation against all 81 archived condition summaries and 8,100 replicate rows.
- `PACKAGE_SHA256.txt`: byte-level checksums of the delivery package.

The package contains no individual-level Natality records, fitted clinical model, or internal service address. CDC Natality public-use files are available from the NCHS Vital Statistics Online portal and remain subject to the NCHS Data User Agreement.

## Verify the delivered package

Create an environment with Python 3.9 or newer, install `requirements_anonymous.txt` (the dependency filename is retained from the reviewed package), and run from this directory:

```bash
python verify_submission.py
```

This checks every archived checksum, all reported numerical invariants, the synthetic parameter specification, author information, and the eight-page main-text limit. The official NeurIPS 2026 style is unmodified; its SHA-256 is `c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a`.

## Regenerate the manuscript artifacts

Run in a working copy after verifying the delivered files:

```bash
python generate_figures.py
python generate_tables.py
tectonic main.tex --outdir build --keep-logs
```

Copy `build/main.pdf` to `main.pdf`, then run `python verify_submission.py --skip-hashes`. This retains all numerical and manuscript checks while allowing regenerated PDF timestamps and local environment metadata to differ from the delivery hashes. A standard `pdflatex`/`bibtex` sequence may replace Tectonic. Import the ZIP into Overleaf and select `main.tex` as the main document to edit the manuscript.

## Re-run the controlled experiment

```bash
python analysis_code/synthetic/run_tae_synthetic_shapley81.py \
  --config analysis_code/configs/synthetic_experiment_shapley81.yaml \
  --output-dir build/synthetic_rerun
python verify_synthetic_rerun.py build/synthetic_rerun
```

The output directory must be new; choose another directory or explicitly use `--overwrite` for a repeated local run. The archived manifest records the environment and hashes from the reported synthetic run. The runner's console summary counts 12 conditions with a negative range for either fixed-budget contrast; the paper reports the 9 conditions negative specifically at the all-record-derived budget. These are different, compatible counts.

The aggregate CDC audit, feature sensitivities, corrected CDC 2024 four-cell replication, and every manuscript figure and table can be checked from this package. The camera-ready preparation independently replayed the frozen models on CDC 2023, reproducing all 24 four-cell point estimates exactly, and completed the same audit on CDC 2024 with 500 paired resamples per outcome. The reviewed 2024 threshold-only calculation used strings for numeric race codes; its results are superseded by the corrected replication. `replication_2024_results.csv` now uses full-precision 2022 thresholds. Both threshold precisions remain in `temporal_2024/threshold_transport.csv`.

For raw-data reconstruction and local fitting without ClearML, use the repository at https://github.com/vache4ogan/prenatal-risk-scoreability and its `requirements-replication.txt`, `scripts/preprocessing/`, `scripts/experiments/fit_canonical_local.py`, and `scripts/experiments/run_temporal_four_cell.py`. A complete raw-file-to-training rerun was not performed during camera-ready preparation; frozen-model replay and aggregate/synthetic regeneration were performed. Repository visibility is controlled by its owner and public release is pending.
