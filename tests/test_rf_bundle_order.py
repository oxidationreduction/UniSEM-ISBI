"""A source RF's stored full55 order must be applied before prediction."""
from copy import deepcopy

import joblib
import numpy as np
import pytest

from unisem.rag import RAG_FEATURE_NAMES, load_random_forest


class FeatureProbeForest:
    n_features_in_ = 55
    classes_ = np.asarray([0, 1])

    def predict_proba(self, features):
        positive = features[:, 0] / 100
        return np.column_stack((1 - positive, positive))


def bundle(names=RAG_FEATURE_NAMES):
    return {"model": FeatureProbeForest(), "feature_columns": names,
            "p_same": 0.65, "min_instance_area": 16}


def test_source_order_is_applied_exactly(tmp_path):
    names = tuple(reversed(RAG_FEATURE_NAMES))
    path = tmp_path / "rf.joblib"
    payload = bundle(names)
    joblib.dump(payload, path)
    features = np.arange(110, dtype=np.float32).reshape(2, 55) / 2
    actual = load_random_forest(path).predict_proba(features)
    expected = payload["model"].predict_proba(features[:, ::-1])
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual, payload["model"].predict_proba(features))


def test_public_order_remains_unchanged(tmp_path):
    path = tmp_path / "rf.joblib"
    payload = bundle()
    joblib.dump(payload, path)
    features = np.ones((2, 55), dtype=np.float32)
    np.testing.assert_array_equal(load_random_forest(path).predict_proba(features),
                                  payload["model"].predict_proba(features))


@pytest.mark.parametrize("change", ["duplicate", "unknown", "threshold", "conflicting"])
def test_invalid_schema_and_threshold_still_rejected(tmp_path, change):
    payload = deepcopy(bundle())
    if change == "duplicate":
        payload["feature_columns"] = (RAG_FEATURE_NAMES[0],) * 55
    elif change == "unknown":
        payload["feature_columns"] = (*RAG_FEATURE_NAMES[:-1], "unknown")
    elif change == "threshold":
        payload["p_same"] = 0.5
    else:
        payload["feature_names"] = tuple(reversed(RAG_FEATURE_NAMES))
    path = tmp_path / "invalid.joblib"
    joblib.dump(payload, path)
    with pytest.raises(ValueError):
        load_random_forest(path)
