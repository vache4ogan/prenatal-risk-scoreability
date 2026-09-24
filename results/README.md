# Final result artifacts

This directory contains retained outputs from the final evaluation protocol.
Human-readable weekly reports and manuscript drafts are intentionally excluded.

- `tae_2026_canonical/`: canonical CDC 2022/2023 evaluation outputs and locked
  run metadata.
- `fixed_B_shapley/`: four-cell and Shapley decomposition outputs.
- `tae_2026_synthetic_shapley81/`: final 81-condition synthetic stress test.
- `tae_2026_replication/`: historical 2024 threshold-transport output from the
  reviewed version. **Superseded:** the old runner read MRACE31 as strings,
  disabling numeric one-hot categories. Do not use these rows as corrected
  camera-ready evidence.
- `sensitivities/`: no-race and no-prior-term sensitivity runs.

The compact aggregate inputs used directly by the manuscript are stored in
`paper/source_data/` and checked by `paper/verify_submission.py`.
# Current camera-ready replication

The corrected CDC 2024 four-cell audit is in
`paper/source_data/temporal_2024/`: 24 policy cells, six contrasts, 3,000 paired
resamples, exact and rounded threshold transport, and input/model/output hashes.
The historical `tae_2026_replication/` threshold-only results are superseded:
that run treated numeric race categories as strings. See `CHANGELOG.md`.
