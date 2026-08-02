# Validation report

- **PASS — Source isolation.** Parent `git status --short --untracked-files=all` before and after migration differed only by the new `clear_research/` directory. No pre-existing path outside it was modified.
- **PASS — Python syntax.** All six copied Python scripts compiled with `PYTHONPYCACHEPREFIX` directed to a temporary directory; no bytecode was written into the destination.
- **NOT RUN — YAML parser.** `configs/canonical/article_revision.yaml` is byte-identical to its source, but no PyYAML, Ruby, or `yq` parser is installed. Network/package installation was prohibited.
- **PASS — JSON parsing.** All JSON files, including `metadata/manifests/article_revision_run.json`, parsed with Python's standard library.
- **PASS — TOML parsing.** `pyproject.toml` parsed with Python `tomllib`.
- **PASS — Forbidden artifacts.** No `.env`, ZIP, joblib, pickle, pyc, `__pycache__`, model, or prediction artifact exists under `clear_research/`.
- **PASS — Credential scan.** No likely literal credential was found. Evidence: `.env.example` credential fields are empty; the scan reported zero path/key findings and did not print values.
- **PASS — Interpretation wording.** Contextual review found no prohibited affirmative claim. Instances of “external validation,” “class balancing improves training,” and “available at booking” occur only in explicit negations/guardrails in `README.md`, `docs/data_and_cohorts.md`, and the copied `docs/claim_sheet.md`. Legacy +5.5 to +8.5 pp effects are labeled non-primary.
- **PASS — README requirements.** `README.md` contains +2.58, +2.73, +3.43, temporal test, equal absolute budget, full precision higher, paired full-size bootstrap, and 500 replicates.
- **FAIL — Hypothesis-verdict requirements.** `docs/hypothesis_verdict.md` is the required placeholder because the canonical source was absent; it cannot truthfully satisfy the eight content checks.
- **PASS — Copy integrity.** All 28 untransformed copies are byte-identical to their sources; evidence is `repo_migration/COPIED_FILES.tsv`.
- **PASS — Source/config integrity.** Copied experiment script hash `c9dd2e3c…` and config hash `afc1d8d1…` match their parent sources.
- **PASS — Result row counts.** The eleven CSV row counts match the sanitized manifest: 24/12/12/24/12/24/3/12/480/12/6. The Markdown summary has no row-count expectation.
- **PASS — Result output hashes.** All 12 copied canonical result files match the manifest output SHA-256 values.
- **PASS — Canonical values.** CDC 2023 10% values match +2.58/+2.73/+3.43 pp, 7,821/8,271/8,356 additional events, and paired-bootstrap CIs 2.47–2.70/2.62–2.84/3.28–3.57 pp.
- **PASS — Manifest selection and sanitization.** Task `29717840f05144579af240f6ded4852a`, PASS status, 500 replicates, 6,000 replicate rows, canonical values, removed runtime paths, and valid JSON were confirmed.
- **PARTIAL — Test execution.** The six required tests are intentionally marked skipped and were not run as substantive tests because the documented refactors/fixtures do not yet exist.
- **PARTIAL — Clean-clone reproduction.** Exact package versions, data access, two narrative documents, and historical Git identity remain unresolved; see `docs/reproduction.md`.
- **PARTIAL — Prenatal-care unknown normalization.** Source code 99 and blank normalization still require verification before grouping; see `docs/data_and_cohorts.md`.
- **PASS — Publication hashes.** `repo_migration/SHA256SUMS.txt` hashes every destination file except itself.

Summary: **15 PASS, 3 PARTIAL, 1 FAIL, 1 NOT RUN**.
