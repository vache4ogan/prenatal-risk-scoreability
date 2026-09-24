# Changes

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
