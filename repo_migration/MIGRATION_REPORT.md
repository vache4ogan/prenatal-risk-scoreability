# Migration report

- Source root: `/home/vache/Projects/mustafa-c`
- Destination: `/home/vache/Projects/mustafa-c/clear_research`
- Execution time: 2026-08-02 (Europe/Moscow)
- Copied files: 29 total inputs, of which 28 are byte-identical
- Transformed files: 1 sanitized run manifest
- Generated files: 22
- Excluded artifacts: 17 explicitly enumerated files plus excluded directory/source categories

## Copied and transformed files

See `COPIED_FILES.tsv` for every source/destination pair and both hashes. The only transformed copy is `metadata/manifests/article_revision_run.json`; absolute runtime/cache paths were replaced with `<runtime-path-removed>` and a publication note was added.

## Generated files

See `GENERATED_FILES.tsv`. Documentation, project metadata, skipped invariant tests, and migration records were generated without fabricating scientific results or version/Git identity.

## Excluded and missing files

See `EXCLUDED_FILES.md` for secrets, data, model/prediction artifacts, legacy sources/results, caches/environments, and audit-only categories. See `MISSING_FILES.md` for the absent canonical verdict and Russian report, unavailable package-version artifact, and absent Git identity.

## Validation results

See `VALIDATION_REPORT.md`: 15 PASS, 3 PARTIAL, 1 FAIL, and 1 NOT RUN. The failure is the required-content check for the placeholder hypothesis verdict. YAML parsing was not run because no parser is installed and network/package installation was prohibited.

## Unresolved scientific issues

- Verify that prenatal-care source code 99 and blank values are normalized to missing before canonical grouping.
- Replace the hypothesis-verdict placeholder only with a reviewed canonical corrected verdict.
- Replace the Russian-report placeholder only with a reviewed report satisfying all six acceptance statements.

## Unresolved reproducibility issues

- Restore exact dependency versions from the canonical ClearML `package_versions.txt` artifact.
- Obtain and verify the two harmonized datasets by the manifest hashes.
- Establish a clean Git commit identity for a future rerun; the historical manifest has null Git fields.
- Refactor/import fixtures required to activate the six scientific invariant tests.

## Exact next manual steps

1. Review all destination documentation and the sanitized manifest.
2. Supply the two missing canonical narrative documents and validate their required statements.
3. Audit prenatal-care unknown-code normalization, then document the evidence.
4. Retrieve `package_versions.txt` from canonical ClearML task `29717840f05144579af240f6ded4852a` and lock exact versions without changing historical claims.
5. Install dependencies in an isolated environment, parse the YAML, and activate/run the six tests after the required refactors and fixtures.
6. Obtain the datasets, verify both SHA-256 values, and conduct a future clean rerun from a reviewed commit.
7. Compare all row counts and hashes, then create an annotated article tag after human approval.

## Git options (not executed)

**Option A:** Use `clear_research/` as a normal directory tracked by the parent repository.

**Option B:** After human review, move `clear_research/` outside the parent repository and initialize it as an independent Git repository.

No nested repository was initialized, no files were staged, and no commit or Git configuration change was made. `clear_research/` will appear as an untracked directory in the parent repository. No file outside `clear_research/` was modified by this migration.
