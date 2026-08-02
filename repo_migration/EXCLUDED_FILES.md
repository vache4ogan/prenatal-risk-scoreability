# Excluded files and artifacts

Counts below describe explicitly observed or named artifacts, not exhaustive contents of excluded directories.

## Secrets

- Parent `.env` (one excluded secret-bearing file category).

## Data

- Two raw ZIP archives, three observed source/derived full CSVs outside harmonization, two harmonized full datasets, and the historical raw dataset copy under `archive/legacy/` (eight observed files). Only four small harmonization metadata CSVs were copied.

## Model/prediction artifacts

- Six saved model paths named in the manifest, the excluded replicate-level bootstrap CSV, and all prediction/model directories (seven explicitly enumerated files plus directory categories).

## Legacy source

- Everything under `archive/legacy/`, factorial scripts, LR-convergence scripts, old sensitivity/calibration/predecessor scripts, `project_env.py`, `requirements_factorial.txt`, and the parent `README.md`.

## Legacy results

- `sens_run/`, root-level old result CSVs, and the named legacy result directories: `calibration_res`, `fixed_model_access_audit`, `lr_res_wighouth_warning`, `new_lr_wihout_warning`, `sensitivity`, and `xb_cat_calibartor`.

## Caches/environments

- `venv/`, `.venv/`, all `__pycache__/`, `.pyc`, and tool caches.

## Audit-only files

- `repo_audit_output/` and raw-coordinate/audit-output tables outside the four allowlisted harmonization metadata files.

For migration accounting, the excluded artifact count is **17 explicitly enumerated files** (8 data, 6 manifest-listed models, 1 bootstrap replicate table, 1 parent `.env`, and 1 parent README), in addition to excluded directory and source categories.
