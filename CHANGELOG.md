# Changes

## Public-code audit, 2026-10-07

- Check the current TAE camera-ready requirements: ten content pages, official
  unmodified NeurIPS 2026 final workshop style, and named author affiliations.
- Promote the complete CDC 2024 replication to the main results. Explain the
  historical categorical-code correction and distinguish it from cutoff rounding.
- Clarify signed, noncausal Shapley allocations, budget-aware capture bounds,
  and the policy-choice criterion when access costs differ. Correct rounding
  and distinguish sensitivity point-estimate ranges from uncertainty intervals.
- Redesign the headline and synthetic figures for readable full-width layout,
  including a near-zero zoom and a 27-condition reversal-frequency heatmap.
  Improve table typesetting without changing the underlying estimates.
- Repeat the full synthetic grid in Python 3.12: all 81 summaries and 8,100
  replicate rows agree with the archive within 1e-12. Reverify package checksums,
  numerical identities, uncertainty summaries, author details, and PDF layout.
- Add the standard MIT license for the authors' code and software documentation,
  with 2026 copyright for Vache Oganisyan, Dmitry Lvov, and Ilya Pershin. CDC data,
  research data, the article and its figures/tables, and third-party materials
  are outside that grant; the published OpenReview article is separately CC BY 4.0.
- Distinguish the current canonical package versions from legacy article-revision
  metadata. Document the local 2023 replay command, required audited inputs,
  and which stored reruns were rechecked.
- Add `extract_canonical_cdc.py`, a separate 2022/2023 raw-TXT-to-canonical-CSV
  entrypoint. Reuse the audited coordinates and unchanged harmonization/target
  functions, including OEGest rather than COMBGEST for preterm. Require frozen
  row counts and output hashes before creating a completed output directory;
  retain legacy extraction and historical files unchanged. Synthetic fixed-width
  integration tests cover both years, missing codes, categorical encoding,
  failure cleanup, and byte equality with the existing harmonizer. No full raw
  dataset reconstruction or full training was performed in this audit.
- Update the README for the corrected full-precision paper threshold runner
  maintained separately from the full four-cell local entrypoint.
- Reject empty, incomplete, duplicate, or invalid calibration-threshold inputs
  before reading clinical data or creating replay outputs. Validate source year
  and comparator when supplied, record the threshold-file hash, and add
  data-independent regression tests. Scoring, cohort selection, and bootstrap
  calculations are unchanged.
- Preserve every archived result and manifest. The CDC 2024 manifest's
  `script_sha256` (`7f2a789347e71416359b7c2a1e40b47863a3696e2c329d5811b26064ac384bba`)
  identifies the historical runner before the threshold-input validation above;
  it is not an assertion that the current runner has identical bytes. That
  source is retained at commit `262b0b94cb97809d75dd9b352254dfa554022a92`.
  The shared `run_tae_shapley_fixed_b_bootstrap_clearml.py` audit module is
  unchanged, including its historical SHA-256
  `fd5ee0acb4ce87e578cf8ce2719bd91042291da70f62744db277ea9d5b3940bd`.

## Camera-ready preparation, 2026-09-24

- Use the official unmodified NeurIPS 2026 final workshop style, named authors,
  dual affiliations, funding acknowledgment, and repository release link.
- Fully specify the synthetic data-generating process, parameters, shared draws,
  and interpretation of replicate ranges. Re-run all 81 conditions and 8,100
  replicate rows against the archived results.
- Clarify the four-cell design relative to established Shapley decomposition
  and resource-constrained evaluation. Correct the combined-policy yield label
  and provide an explicit value/cost criterion for policy comparison.
- Independently replay the frozen CDC 2022 models on CDC 2023: all 24 original
  four-cell point estimates match exactly.
- Correct CDC 2024 race encoding. The reviewed threshold-only runner loaded
  numeric MRACE31 codes as strings, silently producing unknown one-hot categories.
  The corrected runner uses the same numeric representation as training, with
  a regression test. This correction does not change CDC 2023 results.
- Complete a CDC 2024 four-cell replication with the same frozen models and 500
  paired full-size resamples per outcome. At top 10%, capacity shares are
  61.86% (preterm), 62.78% (NICU), and 60.85% (low birth weight).
- Replace old CDC 2024 threshold figures: full-precision 2022 cutoffs select
  372,390 / 373,434 / 358,394 records, respectively. Full-precision and rounded
  cutoff results are both retained in the new aggregate artifacts.
- Add local-only fitting/replay entrypoints, locked replay dependencies,
  executable tests in place of skipped placeholders, and artifact-verification CI.
  No individual records, fitted models, or private credentials are published.

The reviewed snapshot remains available at `tae-2026-submitted`; historical
2024 outputs are retained for provenance and are explicitly superseded.
