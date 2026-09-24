import numpy as np
import pandas as pd
from run_temporal_four_cell import normalize_frame
from run_tae_canonical_clearml import build_preprocessor, LANDMARK_STRICT_FEATURES


def test_raw_numeric_race_strings_match_training_encoding():
    values = {f: [1.0, 2.0] for f in LANDMARK_STRICT_FEATURES}
    values.update(mother_race=["01", "02"], is_us_resident=[1, 1],
                  is_singleton=[1, 1], prenatal_care_month=[2, 5],
                  target_preterm=[0, 1], target_nicu=[0, 1], target_lbw=[0, 1])
    raw = pd.DataFrame(values)
    normalized = normalize_frame(raw)
    train = normalized.copy()
    pre = build_preprocessor(LANDMARK_STRICT_FEATURES, ["mother_race"],
                             {"numeric_imputation": "median", "categorical_imputation": "most_frequent", "standardize_numeric": True})
    pre.fit(train)
    assert np.array_equal(pre.transform(train), pre.transform(normalized))
    encoder = pre.named_transformers_["categorical"]
    encoded = encoder.transform(normalized[["mother_race"]]).toarray()
    assert (encoded.sum(axis=1) == 1).all()
    assert not (encoder.transform(raw[["mother_race"]]).toarray().sum(axis=1) == 1).any()
