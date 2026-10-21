from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np

from unisem.inference import (
    CHANNEL_NAMES,
    hann_window,
    overlap_fusion,
    seeded_watershed,
    tile_starts,
    write_h5,
)
from unisem.rag import (
    RAG_FEATURE_NAMES,
    MERGE_THRESHOLD,
    SCALE_NAMES,
    extract_edge_features,
    fit_random_forest,
    merge_fragments,
)


EXPECTED_FEATURE_NAMES = (
    "area_a",
    "area_b",
    "area_union",
    "contact_length",
    "fg_disagreement_512_256",
    "fg_disagreement_512_384",
    "is_large_pair",
    "is_medium_pair",
    "is_small_pair",
    "mean_boundary_distances_tile256_overlap_edge",
    "mean_boundary_distances_tile256_overlap_fragments",
    "mean_boundary_distances_tile384_overlap_edge",
    "mean_boundary_distances_tile384_overlap_fragments",
    "mean_boundary_distances_tile512_overlap_edge",
    "mean_boundary_distances_tile512_overlap_fragments",
    "mean_center_distances_tile256_overlap_edge",
    "mean_center_distances_tile256_overlap_fragments",
    "mean_center_distances_tile384_overlap_edge",
    "mean_center_distances_tile384_overlap_fragments",
    "mean_center_distances_tile512_overlap_edge",
    "mean_center_distances_tile512_overlap_fragments",
    "mean_foreground_tile256_overlap_edge",
    "mean_foreground_tile256_overlap_fragments",
    "mean_foreground_tile384_overlap_edge",
    "mean_foreground_tile384_overlap_fragments",
    "mean_foreground_tile512_overlap_edge",
    "mean_foreground_tile512_overlap_fragments",
    "purity_vote_a_tile256_overlap",
    "purity_vote_a_tile384_overlap",
    "purity_vote_a_tile512_overlap",
    "purity_vote_b_tile256_overlap",
    "purity_vote_b_tile384_overlap",
    "purity_vote_b_tile512_overlap",
    "raw_edge_contrast",
    "same_label_vote_tile256_overlap",
    "same_label_vote_tile384_overlap",
    "same_label_vote_tile512_overlap",
    "std_boundary_distances_tile256_overlap_edge",
    "std_boundary_distances_tile256_overlap_fragments",
    "std_boundary_distances_tile384_overlap_edge",
    "std_boundary_distances_tile384_overlap_fragments",
    "std_boundary_distances_tile512_overlap_edge",
    "std_boundary_distances_tile512_overlap_fragments",
    "std_center_distances_tile256_overlap_edge",
    "std_center_distances_tile256_overlap_fragments",
    "std_center_distances_tile384_overlap_edge",
    "std_center_distances_tile384_overlap_fragments",
    "std_center_distances_tile512_overlap_edge",
    "std_center_distances_tile512_overlap_fragments",
    "std_foreground_tile256_overlap_edge",
    "std_foreground_tile256_overlap_fragments",
    "std_foreground_tile384_overlap_edge",
    "std_foreground_tile384_overlap_fragments",
    "std_foreground_tile512_overlap_edge",
    "std_foreground_tile512_overlap_fragments",
)


class FixedProbabilityModel:
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def __init__(self, positive_probabilities: list[float]) -> None:
        self.positive_probabilities = np.asarray(positive_probabilities, dtype=np.float64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        assert len(features) == len(self.positive_probabilities)
        return np.column_stack((1.0 - self.positive_probabilities, self.positive_probabilities))


def _rag_inputs() -> tuple[
    np.ndarray,
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    height, width = 24, 48
    fragments = np.zeros((height, width), dtype=np.uint32)
    fragments[:, :16] = 1
    fragments[:, 16:32] = 2
    fragments[:, 32:] = 3

    yy, xx = np.mgrid[:height, :width]
    raw = (0.2 * yy / (height - 1) + 0.8 * xx / (width - 1)).astype(np.float32)
    states_by_scale: dict[str, dict[str, np.ndarray]] = {}
    segmentations_by_scale: dict[str, np.ndarray] = {}
    ground_truth = np.where(fragments < 3, 1, 2).astype(np.uint32)
    for scale_index, scale in enumerate(SCALE_NAMES):
        scale_offset = np.float32(scale_index * 0.05)
        states_by_scale[scale] = {
            "foreground": np.clip(0.8 - 0.1 * raw + scale_offset, 0.0, 1.0),
            "center_distances": np.clip(0.2 + 0.3 * raw + scale_offset, 0.0, 1.0),
            "boundary_distances": np.clip(0.1 + 0.5 * raw + scale_offset, 0.0, 1.0),
        }
        segmentations_by_scale[scale] = ground_truth.copy()
    return fragments, states_by_scale, segmentations_by_scale, raw, ground_truth


def test_hann_overlap_fusion_is_numerically_consistent_and_fully_covered() -> None:
    height, width = 73, 91
    image = np.arange(height * width, dtype=np.float32).reshape(height, width) / (height * width)
    tile_shape = (32, 28)
    stride = (19, 17)
    offsets = ((0, 0), (7, 5))

    def predict_tile(tile: np.ndarray) -> np.ndarray:
        return np.stack((tile, 2.0 * tile, np.full_like(tile, 0.25)), axis=0)

    fused = overlap_fusion(
        image,
        predict_tile,
        tile_shape=tile_shape,
        stride=stride,
        offsets=offsets,
    )
    expected = np.stack((image, 2.0 * image, np.full_like(image, 0.25)), axis=0)
    np.testing.assert_allclose(fused, expected, rtol=1.0e-6, atol=1.0e-6)

    coverage = np.zeros_like(image)
    for offset_y, offset_x in offsets:
        for y0 in tile_starts(height, tile_shape[0], stride[0], offset_y):
            for x0 in tile_starts(width, tile_shape[1], stride[1], offset_x):
                y1 = min(height, y0 + tile_shape[0])
                x1 = min(width, x0 + tile_shape[1])
                coverage[y0:y1, x0:x1] += hann_window(y1 - y0, x1 - x0)
    assert np.all(np.isfinite(coverage))
    assert float(coverage.min()) > 0.0


def test_seeded_watershed_recovers_two_instances() -> None:
    yy, xx = np.mgrid[:96, :112]
    object_a = (yy - 34) ** 2 + (xx - 30) ** 2 <= 18**2
    object_b = (yy - 62) ** 2 + (xx - 82) ** 2 <= 19**2
    foreground = (object_a | object_b).astype(np.float32)

    center = np.ones_like(foreground)
    boundary = np.ones_like(foreground)
    center[(yy - 34) ** 2 + (xx - 30) ** 2 <= 7**2] = 0.0
    center[(yy - 62) ** 2 + (xx - 82) ** 2 <= 7**2] = 0.0
    boundary[(yy - 34) ** 2 + (xx - 30) ** 2 <= 7**2] = 0.0
    boundary[(yy - 62) ** 2 + (xx - 82) ** 2 <= 7**2] = 0.0

    segmentation = seeded_watershed(
        {
            "foreground": foreground,
            "center_distances": center,
            "boundary_distances": boundary,
        }
    )
    assert segmentation.dtype == np.uint32
    assert set(np.unique(segmentation).tolist()) == {0, 1, 2}
    assert segmentation[34, 30] != 0
    assert segmentation[62, 82] != 0
    assert segmentation[34, 30] != segmentation[62, 82]


def test_rag_feature_schema_is_fixed_and_supervision_contains_both_classes() -> None:
    fragments, states, segmentations, raw, ground_truth = _rag_inputs()
    batch = extract_edge_features(
        fragments,
        states,
        segmentations,
        raw,
        ground_truth=ground_truth,
    )

    assert RAG_FEATURE_NAMES == EXPECTED_FEATURE_NAMES
    assert len(RAG_FEATURE_NAMES) == 55
    np.testing.assert_array_equal(batch.edges, np.asarray([[1, 2], [2, 3]]))
    assert batch.features.shape == (2, 55)
    assert np.all(np.isfinite(batch.features))
    np.testing.assert_array_equal(batch.labels, np.asarray([1, 0], dtype=np.uint8))
    np.testing.assert_allclose(batch.features[:, RAG_FEATURE_NAMES.index("area_union")], 768.0)
    np.testing.assert_allclose(batch.features[:, RAG_FEATURE_NAMES.index("contact_length")], 24.0)


def test_random_forest_learns_positive_and_negative_edges() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(0.0, 0.05, size=(48, len(RAG_FEATURE_NAMES))).astype(np.float32)
    labels = np.repeat(np.asarray([0, 1], dtype=np.uint8), 24)
    features[:24] -= 1.0
    features[24:] += 1.0

    model = fit_random_forest(features, labels, n_jobs=1)
    probabilities = model.predict_proba(features)
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    assert model.n_features_in_ == 55
    assert float(probabilities[labels == 1, positive_column].mean()) > 0.9
    assert float(probabilities[labels == 0, positive_column].mean()) < 0.1


def test_merge_threshold_and_union_find_transitivity() -> None:
    fragments = np.zeros((20, 30), dtype=np.uint32)
    fragments[:, :10] = 1
    fragments[:, 10:20] = 2
    fragments[:, 20:] = 3
    edges = np.asarray([[1, 2], [2, 3]], dtype=np.int64)
    features = np.zeros((2, len(RAG_FEATURE_NAMES)), dtype=np.float32)

    thresholded = merge_fragments(
        fragments,
        edges,
        features,
        FixedProbabilityModel([MERGE_THRESHOLD, MERGE_THRESHOLD - 1.0e-6]),
    )
    assert thresholded[0, 5] == thresholded[0, 15]
    assert thresholded[0, 15] != thresholded[0, 25]
    assert len(np.unique(thresholded)) == 2

    transitive = merge_fragments(
        fragments,
        edges,
        features,
        FixedProbabilityModel([0.9, 0.9]),
    )
    assert np.all(transitive == 1)


def test_h5_writer_round_trip() -> None:
    arrays = {
        CHANNEL_NAMES[0]: np.arange(30, dtype=np.float32).reshape(5, 6),
        "merged_segmentation": np.arange(60, dtype=np.uint32).reshape(2, 5, 6),
    }
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "nested" / "prediction.h5"
        write_h5(output_path, arrays)
        with h5py.File(output_path, "r") as handle:
            assert set(handle.keys()) == set(arrays)
            for name, expected in arrays.items():
                np.testing.assert_array_equal(handle[name][...], expected)
                assert handle[name].compression == "gzip"
    assert not output_path.exists()
