# Anonymous supplementary package

This package accompanies the double-blind TAE 2026 submission "Same Top Fraction, Different Workload: Auditing Eligibility and Capacity in Clinical Risk Evaluation."

## Contents

- `main.tex`, `main.pdf`, `references.bib`, and `neurips_2026.sty`: manuscript source and compiled paper.
- `source_data/`: aggregate, non-record-level inputs behind every manuscript figure and table.
- `generate_figures.py` and `generate_tables.py`: deterministic manuscript generators.
- `analysis_code/synthetic/`: source for the 81-condition controlled experiment.
- `analysis_code/temporal/`: sanitized CDC 2024 score-cutoff transport script.
- `analysis_code/synthetic_outputs/`: archived replicate-level outputs from that experiment.
- `verify_submission.py`: numerical, citation, layout, and manuscript-claim checks.

The package contains no individual-level Natality records, fitted clinical model, internal service address, or author-identifying path. CDC Natality public-use files are available from the NCHS Vital Statistics Online portal and remain subject to the NCHS Data User Agreement.

## Regenerate the manuscript artifacts

Create an environment with Python 3.9 or newer and install `requirements_anonymous.txt`, then run:

```bash
python generate_figures.py
python generate_tables.py
tectonic main.tex --outdir build --keep-logs
python verify_submission.py
```

The official NeurIPS LaTeX style is included. A standard `pdflatex`/`bibtex` sequence may replace Tectonic.

## Re-run the controlled experiment

```bash
python analysis_code/synthetic/run_tae_synthetic_shapley81.py \
  --config analysis_code/configs/synthetic_experiment_shapley81.yaml \
  --overwrite
```

The archived manifest records the environment and hashes from the reported synthetic run. The aggregate CDC audit, feature sensitivities, CDC 2024 workload arithmetic, and all manuscript rendering can be checked from this package. End-to-end regeneration of the frozen CDC scores requires the public-use source files, the raw-to-harmonized extraction lineage, and fitted score artifacts planned for the archival release.
