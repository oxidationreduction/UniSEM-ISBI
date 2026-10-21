from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_instances() -> np.ndarray:
    yy, xx = np.mgrid[:96, :112]
    labels = np.zeros((96, 112), dtype=np.uint32)
    labels[((yy - 34) / 17) ** 2 + ((xx - 32) / 20) ** 2 <= 1] = 1
    labels[((yy - 58) / 20) ** 2 + ((xx - 78) / 17) ** 2 <= 1] = 2
    return labels


@pytest.fixture
def synthetic_raw(synthetic_instances: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    raw = np.full(synthetic_instances.shape, 0.75, dtype=np.float32)
    raw[synthetic_instances > 0] = 0.4
    raw += rng.normal(0.0, 0.02, raw.shape).astype(np.float32)
    return np.clip(raw, 0.0, 1.0)
