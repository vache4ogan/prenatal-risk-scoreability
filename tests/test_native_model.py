"""Native model verification on artificial inputs only; never reads records."""

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from native_model import (
    DEFAULT_PRETRAINED_DIR, FEATURES, TARGETS, NativePreprocessor,
    find_native_bundle, load_native_bundle, load_native_target,
)


def synthetic_frames():
    """Fixed-seed artificial probes, unrelated to any individual CDC record."""
    rng = np.random.default_rng(20261007)
    frame = pd.DataFrame(rng.uniform(-100, 500, (512, 9)), columns=FEATURES)
    frame["mother_race"] = np.resize(
        np.r_[np.arange(1, 32), np.nan, -1, 0, 32, 99, 1.5], len(frame)
    )
    frame.loc[:, ["diab_pre", "hyper_pre"]] = rng.integers(0, 2, (512, 2))
    frame = frame.mask(rng.random(frame.shape) < 0.17)
    frame.iloc[0] = np.nan
    frame.iloc[1, 1:] = 0.0
    frame.iloc[2, 1:] = 1e9
    frame.iloc[3, 1:] = -1e9
    frames = {"float64_missing": frame, "float32_missing": frame.astype(np.float32)}
    frames["mixed_dtypes"] = frame.astype({FEATURES[i]: np.float32 for i in (1, 3, 7)})
    frames["nullable_float64"] = frame.astype("Float64")
    frames["nullable_float32"] = frame.astype("Float32")
    frames["integer"] = frame.fillna(0).astype(np.int64)
    frames["nullable_integer"] = frame.round().astype("Int64")
    frames["object_nan"] = frame.astype(object)
    frames["object_none"] = frame.astype(object).where(frame.notna(), None)
    frames["numeric_strings"] = frame.astype(str)
    strings = frame.copy()
    strings["mother_race"] = pd.Series(
        np.resize(np.array(["01", "1", "31", "unseen", np.nan], dtype=object), len(frame)),
        dtype=object,
    )
    frames["race_strings"] = strings
    frames["all_missing"] = frame.iloc[[0]].copy()
    frames["single_row"] = frame.iloc[[4]].copy()
    frames["reordered_extra_columns"] = frame.iloc[:, ::-1].assign(unused="synthetic")
    codebook = pd.DataFrame(0.0, index=range(36), columns=FEATURES)
    codebook["mother_race"] = np.r_[np.arange(1, 32), np.nan, 0, 32, -1, 1.5]
    frames["complete_codebook"] = codebook
    return frames


@pytest.fixture(scope="module", params=TARGETS)
def bundle(request):
    return load_native_target(DEFAULT_PRETRAINED_DIR, request.param)


@pytest.fixture(scope="module", params=TARGETS)
def trusted_pair(request):
    root = os.environ.get("TRUSTED_FROZEN_MODELS")
    if not root:
        pytest.skip("Set TRUSTED_FROZEN_MODELS to explicitly trust local frozen joblibs")
    target = request.param
    files = list(Path(root).rglob(f"{target}__landmark_strict__lightgbm.joblib"))
    assert len(files) == 1, f"Expected exactly one trusted bundle for {target}"
    manifest = json.loads((DEFAULT_PRETRAINED_DIR / "manifest.json").read_text())
    assert hashlib.sha256(files[0].read_bytes()).hexdigest() == (
        manifest["targets"][target]["source_bundle_sha256"]
    ), "Trusted source does not match the frozen export"
    import joblib

    return joblib.load(files[0]), load_native_target(DEFAULT_PRETRAINED_DIR, target)


@pytest.mark.parametrize("case", [k for k in synthetic_frames()
                                 if k not in {"numeric_strings", "race_strings"}])
def test_trusted_joblib_exact_parity(trusted_pair, case):
    trusted, native = trusted_pair
    frame = synthetic_frames()[case]
    before = frame.copy(deep=True)
    expected = trusted["preprocessor"].transform(frame).astype(np.float32)
    actual = native["preprocessor"].transform(frame)
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())
    probabilities = trusted["model"].predict_proba(expected, num_threads=1)
    result = native["model"].predict_proba(actual, num_threads=1)
    np.testing.assert_array_equal(result, probabilities)
    for threshold in trusted["thresholds"].values():
        np.testing.assert_array_equal(result[:, 1] >= threshold, probabilities[:, 1] >= threshold)
    for key in ("features", "target", "feature_set", "training_population", "thresholds"):
        assert native[key] == trusted[key]
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("case", list(synthetic_frames()))
def test_native_smoke_without_pickles(bundle, case):
    frame = synthetic_frames()[case]
    if case in {"numeric_strings", "race_strings"}:
        with pytest.raises(TypeError, match="numeric CDC codes"):
            bundle["preprocessor"].transform(frame)
        return
    matrix = bundle["preprocessor"].transform(frame)
    assert sparse.isspmatrix_csr(matrix)
    assert matrix.dtype == np.float32
    assert matrix.shape == (len(frame), 39)
    prediction = bundle["model"].predict_proba(matrix)
    assert prediction.shape == (len(frame), 2)
    assert np.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 1)).all()
    np.testing.assert_allclose(prediction.sum(axis=1), 1, rtol=0, atol=1e-15)


def test_missing_unknown_and_numeric_race_semantics(bundle):
    frame = synthetic_frames()["complete_codebook"]
    result = bundle["preprocessor"].transform(frame).toarray()
    np.testing.assert_array_equal(result[:31, 8:], np.eye(31, dtype=np.float32))
    metadata = json.loads((DEFAULT_PRETRAINED_DIR / bundle["target"] / "preprocessing.json").read_text())
    preprocessing = metadata["preprocessing"]
    mode = preprocessing["categorical"]["mode"]
    np.testing.assert_array_equal(result[31, 8:], np.eye(31)[int(mode) - 1])
    assert not result[32:, 8:].any()
    missing = bundle["preprocessor"].transform(synthetic_frames()["all_missing"]).toarray()
    numeric = preprocessing["numeric"]
    expected = ((np.array(numeric["medians"]) - numeric["means"]) / numeric["scales"]).astype(np.float32)
    np.testing.assert_array_equal(missing[0, :8], expected)


def test_numeric_race_strings_are_not_coerced(bundle):
    frame = synthetic_frames()["complete_codebook"].iloc[:31].copy()
    frame["mother_race"] = [f"{i:02d}" for i in range(1, 32)]
    with pytest.raises(TypeError, match="numeric CDC codes"):
        bundle["preprocessor"].transform(frame)


def test_column_selection_and_no_input_mutation(bundle):
    frame = synthetic_frames()["float64_missing"]
    before = frame.copy(deep=True)
    expected = bundle["preprocessor"].transform(frame)
    actual = bundle["preprocessor"].transform(frame.iloc[:, ::-1].assign(unused="synthetic"))
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())
    pd.testing.assert_frame_equal(frame, before)


def test_rejects_invalid_inputs(bundle):
    preprocessor = bundle["preprocessor"]
    frame = synthetic_frames()["single_row"]
    with pytest.raises(TypeError, match="DataFrame"):
        preprocessor.transform(frame.to_numpy())
    with pytest.raises(ValueError, match="Missing features"):
        preprocessor.transform(frame.drop(columns="mother_race"))
    with pytest.raises(ValueError, match="Duplicate"):
        preprocessor.transform(pd.concat([frame, frame[["mother_race"]]], axis=1))
    with pytest.raises(ValueError):
        preprocessor.transform(frame.iloc[:0])
    for column in ("mother_bmi", "mother_race"):
        for value in (np.inf, -np.inf):
            with pytest.raises(ValueError):
                preprocessor.transform(frame.assign(**{column: value}))


@pytest.mark.parametrize("target", TARGETS)
def test_lookup_and_json_path(target):
    directory = DEFAULT_PRETRAINED_DIR / target
    assert find_native_bundle(DEFAULT_PRETRAINED_DIR, target) == directory
    assert find_native_bundle(directory, target) == directory
    assert load_native_bundle(directory / "preprocessing.json")["target"] == target


def test_lookup_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        find_native_bundle(tmp_path, "../target_preterm")
    with pytest.raises(FileNotFoundError):
        find_native_bundle(tmp_path, TARGETS[0])
    for name in ("a", "b"):
        directory = tmp_path / name / TARGETS[0]
        directory.mkdir(parents=True)
        (directory / "model.txt").touch()
        (directory / "preprocessing.json").touch()
    with pytest.raises(ValueError, match="Ambiguous"):
        find_native_bundle(tmp_path, TARGETS[0])


def test_rejects_invalid_metadata(tmp_path):
    source = DEFAULT_PRETRAINED_DIR / TARGETS[0]
    directory = tmp_path / TARGETS[0]
    shutil.copytree(source, directory)
    path = directory / "preprocessing.json"
    metadata = json.loads(path.read_text())
    for key, value in (("schema_version", 99), ("features", []),
                       ("feature_set", "other"), ("thresholds", {"0.05": 2})):
        modified = {**metadata, key: value}
        path.write_text(json.dumps(modified), encoding="utf-8")
        with pytest.raises(ValueError):
            load_native_bundle(directory)
    path.write_text(json.dumps({**metadata, "target": TARGETS[1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        load_native_target(tmp_path, TARGETS[0])
    for field, value in (("scales", [0] * 8), ("means", [np.nan] * 8),
                         ("medians", [1] * 7), ("features", [])):
        config = copy.deepcopy(metadata["preprocessing"])
        config["numeric"][field] = value
        with pytest.raises(ValueError):
            NativePreprocessor(config)


def test_published_checksums_and_summary_only_metadata():
    root = DEFAULT_PRETRAINED_DIR
    checksums = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = root / name
        assert path.resolve().is_relative_to(root.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        checksums[name] = digest
    assert set(checksums) == {
        path.relative_to(root).as_posix() for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    for target in TARGETS:
        metadata = json.loads((root / target / "preprocessing.json").read_text())
        assert set(metadata) == {
            "schema_version", "target", "feature_set", "training_population",
            "features", "thresholds", "preprocessing",
        }
        assert set(metadata["preprocessing"]) == {"numeric", "categorical", "matrix_dtype"}
        assert set(metadata["preprocessing"]["numeric"]) == {"features", "medians", "means", "scales"}
        assert set(metadata["preprocessing"]["categorical"]) == {"feature", "mode", "categories"}
        for name in ("model.txt", "preprocessing.json"):
            content = (root / target / name).read_text(encoding="utf-8")
            assert not re.search(r"[A-Za-z]:[\\/]|/(?:home|Users|mnt|tmp|workspace)/", content)
        assert {p.name for p in (root / target).iterdir()} == {"model.txt", "preprocessing.json"}
