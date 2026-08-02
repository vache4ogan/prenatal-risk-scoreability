# Reproduction

Use Python 3.10–3.12. Install runtime dependencies from `pyproject.toml`; exact historical versions still need to be restored from the canonical ClearML `package_versions.txt` artifact.

The canonical entry point is `scripts/experiments/run_article_revision_analyses_clearml.py` with `configs/canonical/article_revision.yaml`. Environment variable names used for ClearML setup are `CLEARML_API_ACCESS_KEY`, `CLEARML_API_SECRET_KEY`, `CLEARML_API_HOST`, `CLEARML_WEB_HOST`, `CLEARML_FILES_HOST`, `CLEARML_PROJECT_NAME`, `CLEARML_PROJECT_ID`, `CLEARML_DATASET_ID`, `CLEARML_DATASET_NAME`, `CLEARML_DATASET_PROJECT`, `CLEARML_QUEUE_NAME`, and `CLEARML_QUEUE_ID`. Copy `.env.example` to an untracked `.env` and supply values locally; never commit secrets.

Expected inputs and hashes:

- `data/cdc_temporal_harmonized/cdc_natality_2022_harmonized.csv`: `c5f1ab62b0e9ac63795e75f6075d2523a3f4c5dc3e95c7aa1f7df6763205a59c` (3,676,029 rows).
- `data/cdc_temporal_harmonized/cdc_natality_2023_harmonized.csv`: `82b38b24d2f795a4a5233ce3a2fcb43b70789339306ba9044bde56f50bb270a8` (3,605,081 rows).

Local command template:

```bash
python scripts/experiments/run_article_revision_analyses_clearml.py --config configs/canonical/article_revision.yaml --local
```

ClearML remote-run template (confirm project, dataset, and queue variables first):

```bash
python scripts/experiments/run_article_revision_analyses_clearml.py --config configs/canonical/article_revision.yaml --enqueue
```

Expected article-facing outputs are the 12 files documented in `results/README.md`. Expected row counts are 24 singleton results, 12 singleton comparisons, 12 bootstrap summaries, 24 mechanism rows, 12 mechanism comparisons, 24 calibration metrics, 3 calibrators, 12 raw-versus-calibrated comparisons, 480 reliability-curve points, 12 all-birth results, and 6 all-birth comparisons. Verify output hashes against `metadata/manifests/article_revision_run.json`; the Markdown summary has no row count.

The historical run has no Git branch, commit hash, or dirty-state identity. A future clean rerun should begin from a reviewed clean commit, restore exact package versions, verify both input hashes, run all validation tests, archive the full ClearML artifacts, compare output hashes and row counts, and record the resulting commit. After review, create an annotated article tag naming the canonical ClearML task and dataset IDs; push neither commit nor tag until a human approves them.
