"""Invariant: canonical tables match manifest row counts, hashes, and locked effects."""

import verify_submission


def test_canonical_result_integrity():
    verify_submission.verify_shapley()
    verify_submission.verify_protocol_and_score_claims()
    verify_submission.verify_feature_sensitivities()
    verify_submission.verify_synthetic()
    verify_submission.verify_synthetic_specification()
    verify_submission.verify_temporal_2024()
