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
The pinned core versions below match
`results/tae_2026_canonical/package_versions.txt`, not the older
`metadata/manifests/package_versions.txt` from the legacy article-revision run.
Activate the environment before running these commands; no ClearML account,
credentials, or network access to ClearML is needed.

```bash
python -m pip install -r requirements-replication.txt
python paper/verify_submission.py
python -m pytest -q
```

These tests execute selection, shared-resample, preprocessing, categorical-code,
calibration-ordering, and artifact checks. They require neither raw CDC records
nor access to ClearML. The former disabled placeholders have been replaced by
executable tests.

`pyproject.toml` contains unpinned package dependencies; installing the project
alone does not select the canonical replay versions. Use the requirements file
above for this route. The figure-rendering environment below is separate.

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
`scripts/preprocessing/`. The frozen harmonized file hashes and row counts are
in `configs/canonical/experiment_config_tae_canonical.yaml`; aggregate schema
audits are in `metadata/harmonization/`.

### Reconstruct the 2022/2023 canonical inputs

Unpack the **US** public-use TXT files, not the territories files. Adjust
`--input` to their extracted locations. Run from the repository root in the
Python 3.12 replication environment; each output directory must be new:

```bash
python scripts/preprocessing/extract_canonical_cdc.py \
  --year 2022 --input data/raw/Nat2022PublicUS.c20230504.r20230822.txt \
  --output-dir data/reconstructed_2022
python scripts/preprocessing/extract_canonical_cdc.py \
  --year 2023 --input data/raw/Nat2023PublicUS.c20240509.r20240724.txt \
  --output-dir data/reconstructed_2023
```

This entrypoint includes extraction **and** the existing canonical harmonizer's
value/target rules. It reads all required fields, including RESTATUS, DMETH_REC,
COMBGEST, OEGest_Comb, OEGest_R3, and DPLURAL. Coordinates use the audited 2022
parser and the official [2022](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/natality/UserGuide2022.pdf)
and [2023](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/natality/UserGuide2023.pdf)
layouts. Preterm is derived only from OEGest_Comb; COMBGEST is retained for audit.
Unknown labels stay missing, no-care month 0 is retained, and MRACE31 codes keep
the canonical CSV representation and numeric interpretation on model input.

Each successful command writes `cdc_natality_YEAR_harmonized.csv`,
`harmonized_schema.json`, and `extraction_manifest.json`. The manifest records
input/output/code hashes and field coordinates. Publication into the new local
directory requires exactly 3,676,029 / 3,605,081 rows and the frozen output
SHA-256 from the canonical config. Invalid records, unexpected row counts, or
hash mismatches fail without leaving a completed output directory. Do not
change the locked hashes to force a mismatching reconstruction through.

Do **not** run `preprocess_harmonize_cdc_2022_2023.py` again on these already
harmonized outputs. That script remains the unchanged route for historical
audited intermediate CSVs with their own frozen input hashes. `extractor.py`
is retained as legacy: its schema and COMBGEST-based preterm label are not the
canonical reconstruction route.

The new raw route has passed synthetic fixed-width integration tests for both
years, including byte-for-byte comparison with the existing harmonizer. The
full raw US files have **not** been reconstructed in this audit, and full model
training has **not** been rerun. Successful full-file reconstruction remains a
separate check enforced by the row-count and output-hash gates above.

### Fit and replay locally

After both extraction commands pass, the following uses only local files and
never creates or updates ClearML tasks:

```bash
python scripts/experiments/fit_canonical_local.py \
  --train-csv data/reconstructed_2022/cdc_natality_2022_harmonized.csv \
  --output-dir models/local_canonical
python scripts/experiments/run_temporal_four_cell.py \
  --csv data/reconstructed_2023/cdc_natality_2023_harmonized.csv \
  --models models/local_canonical \
  --thresholds models/local_canonical/thresholds_2022.csv \
  --year 2023 --bootstrap-replicates 0 \
  --expected-sha256 82b38b24d2f795a4a5233ce3a2fcb43b70789339306ba9044bde56f50bb270a8 \
  --output-dir results/local_2023
```

The 2023 command computes point estimates without a new bootstrap; its `PASS`
does not assert equality with archived results. For 2024, place the uncompressed
US TXT in `data/cdc_2024/` and run the year-specific parser below from the
repository root. It writes `data/cdc_2024/harmonized_cdc_2024.csv`; run it only
when that generated destination does not already exist. The temporal runner
then verifies the frozen CSV hash before scoring. The new extractor above
supports 2022/2023 only.

```bash
python scripts/preprocessing/harmonize_cdc_2024.py data/cdc_2024/Nat2024PublicUS.txt
python scripts/experiments/run_temporal_four_cell.py \
  --csv data/cdc_2024/harmonized_cdc_2024.csv \
  --models models/local_canonical \
  --thresholds models/local_canonical/thresholds_2022.csv \
  --year 2024 --bootstrap-replicates 500 \
  --expected-sha256 65d04e37c416e09a3004a4dae159e600de0a20d2fb873661ba63f12a2163cd92 \
  --output-dir results/local_2024
```

Existing frozen model bundles can be used instead of refitting, but are not
distributed here. A new local fit has not been verified to reproduce the
archived models or results exactly. Never mix a refitted model with another
model's calibration thresholds. The threshold CSV must contain exactly one
row for each of the three targets at both fractions (0.05 and 0.10).

Use `run_temporal_four_cell.py` for the corrected full-precision temporal route.
`run_2024_locked_replication.py` requires author-specific ClearML resources and
uses hardcoded rounded cutoffs; it is not the local reproduction entrypoint.
`scripts/preprocessing/check_cdc_2024_drift.py` still retains the old
string-category path and must not be used as corrected evidence. The corrected
threshold-only script under `paper/analysis_code/temporal/` now validates three
unique full-precision q=10% cutoffs before reading data and writes
`threshold_used`; it is not the full four-cell/bootstrap runner.

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

The 2026-10-07 audit rechecked the saved rerun artifacts, without fitting models
or reading individual records or weights: the local 2023 replay's 24 cells
match all 16 shared non-label columns exactly (no new 2023 bootstrap); the
packaged 2024 run contains 500 paired replicates per outcome/fraction. A fresh
full synthetic run on 7 October repeated all 81 conditions and 100 repetitions
in Python 3.12, took 81 seconds, and matched all 81 summaries and 8,100 replicate
rows within `1e-12`. Repository tests and the package verifier were run again.
The clinical checks reverify stored reruns; they are not new clinical training.
The public package contains the 2024 aggregates and archived synthetic
reference outputs; the independent 2023 replay and fresh synthetic comparison
outputs are local audit evidence, not additional public data releases.

The old CDC 2024 threshold runner read numeric MRACE31 codes as strings,
silently disabling their one-hot categories. Its historical outputs in
`results/tae_2026_replication/` are retained for provenance and are superseded.
The corrected runner uses the canonical numeric representation; a regression
test covers the failure. Main CDC 2023 estimates and the synthetic experiment
are unaffected. See `results/README.md` for the current artifact locations.

This is evaluation research code, not a deployed prenatal-care decision tool.
No private credentials, clinical records, or fitted clinical model are required
for the aggregate and synthetic verification routes.

## Licenses

The authors' source code and accompanying software documentation are available
under the [MIT License](LICENSE), copyright 2026 Vache Oganisyan, Dmitry Lvov,
and Ilya Pershin. This grant does not relicense CDC records, research data, the
manuscript, figures, tables, or third-party materials.

CDC Natality data remain subject to the NCHS Data User Agreement. The published
[article on OpenReview](https://openreview.net/forum?id=puYbvC47rw) is licensed
separately under CC BY 4.0. Bundled third-party materials retain their own
license terms. `CITATION.cff` describes the software license, not the CDC data
or article license.
