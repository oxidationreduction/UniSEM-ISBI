"""Portable candidate-fragment graph features and random-forest merging."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCALE_NAMES = ("tile512_overlap", "tile384_overlap", "tile256_overlap")
CHANNEL_NAMES = ("foreground", "center_distances", "boundary_distances")
PURITY_THRESHOLD = 0.5
MERGE_THRESHOLD = 0.65
MIN_INSTANCE_AREA = 16
RF_PARAMETERS = {
    "n_estimators": 200,
    "max_depth": 12,
    "class_weight": "balanced",
    "random_state": 0,
}

RAG_FEATURE_NAMES = (
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

if len(RAG_FEATURE_NAMES) != 55:
    raise RuntimeError("RAG_FEATURE_NAMES must contain exactly 55 entries")


@dataclass(frozen=True)
class EdgeFeatureBatch:
    """Strict edge order, feature matrix, and optional supervision metadata."""

    edges: np.ndarray
    features: np.ndarray
    labels: np.ndarray | None = None
    major_ground_truth: np.ndarray | None = None
    purities: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError("edges must have shape (N, 2)")
        if self.features.shape != (len(self.edges), len(RAG_FEATURE_NAMES)):
            raise ValueError("features must have shape (N, 55)")
        if self.labels is not None and self.labels.shape != (len(self.edges),):
            raise ValueError("labels must have shape (N,)")
        if self.major_ground_truth is not None and self.major_ground_truth.shape != (len(self.edges), 2):
            raise ValueError("major_ground_truth must have shape (N, 2)")
        if self.purities is not None and self.purities.shape != (len(self.edges), 2):
            raise ValueError("purities must have shape (N, 2)")


def four_neighbor_adjacency(fragments: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Return positive-label 4-neighbor edges and their boundary contacts."""
    fragments = _validate_label_image(fragments, "fragments")
    contacts: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)

    left = fragments[:, :-1]
    right = fragments[:, 1:]
    rows, columns = np.nonzero((left != right) & (left > 0) & (right > 0))
    for row, column in zip(rows.tolist(), columns.tolist()):
        label_a = int(left[row, column])
        label_b = int(right[row, column])
        edge = (label_a, label_b) if label_a < label_b else (label_b, label_a)
        contacts[edge].append((row, column, row, column + 1))

    top = fragments[:-1, :]
    bottom = fragments[1:, :]
    rows, columns = np.nonzero((top != bottom) & (top > 0) & (bottom > 0))
    for row, column in zip(rows.tolist(), columns.tolist()):
        label_a = int(top[row, column])
        label_b = int(bottom[row, column])
        edge = (label_a, label_b) if label_a < label_b else (label_b, label_a)
        contacts[edge].append((row, column, row + 1, column))

    return {
        edge: np.asarray(edge_contacts, dtype=np.int64).reshape(-1, 4)
        for edge, edge_contacts in sorted(contacts.items())
    }


def _validate_label_image(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError(f"{name} must contain integer labels")
    array = array.astype(np.int64, copy=False)
    if np.any(array < 0):
        raise ValueError(f"{name} must contain nonnegative labels")
    return array


def _validate_dense_array(values: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _fragment_index(fragments: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels, inverse, counts = np.unique(fragments, return_inverse=True, return_counts=True)
    positive = labels > 0
    positive_labels = labels[positive].astype(np.int64, copy=False)
    dense_index = np.full(labels.shape, -1, dtype=np.int64)
    dense_index[positive] = np.arange(int(positive.sum()), dtype=np.int64)
    pixel_index = dense_index[inverse].reshape(fragments.shape)
    return positive_labels, counts[positive].astype(np.int64, copy=False), pixel_index


def _array_statistics(
    pixel_index: np.ndarray,
    values: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = pixel_index >= 0
    indices = pixel_index[valid]
    flattened = values[valid].astype(np.float64, copy=False)
    sums = np.bincount(indices, weights=flattened, minlength=len(counts))
    sums_squared = np.bincount(indices, weights=flattened * flattened, minlength=len(counts))
    means = sums / counts
    variances = np.maximum((sums_squared / counts) - means * means, 0.0)
    return means, np.sqrt(variances)


def _combined_mean_std(
    index_a: int,
    index_b: int,
    means: np.ndarray,
    standard_deviations: np.ndarray,
    counts: np.ndarray,
) -> tuple[float, float]:
    count_a = int(counts[index_a])
    count_b = int(counts[index_b])
    total = count_a + count_b
    mean = (float(means[index_a]) * count_a + float(means[index_b]) * count_b) / total
    second_moment = (
        (float(standard_deviations[index_a]) ** 2 + float(means[index_a]) ** 2) * count_a
        + (float(standard_deviations[index_b]) ** 2 + float(means[index_b]) ** 2) * count_b
    ) / total
    return mean, float(np.sqrt(max(second_moment - mean * mean, 0.0)))


def _major_labels(
    pixel_index: np.ndarray,
    reference: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    major = np.zeros(len(counts), dtype=np.int64)
    purities = np.zeros(len(counts), dtype=np.float64)
    valid = (pixel_index >= 0) & (reference > 0)
    if not np.any(valid):
        return major, purities

    pairs = np.stack((pixel_index[valid], reference[valid]), axis=1)
    unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)
    best_counts = np.zeros(len(counts), dtype=np.int64)
    for (fragment_index, reference_label), count in zip(unique_pairs.tolist(), pair_counts.tolist()):
        if int(count) > int(best_counts[int(fragment_index)]):
            major[int(fragment_index)] = int(reference_label)
            best_counts[int(fragment_index)] = int(count)
    purities = best_counts / counts
    return major, purities


def purity_edge_labels(
    fragments: np.ndarray,
    ground_truth: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label an edge positive when both fragments are pure for one GT object."""
    fragments = _validate_label_image(fragments, "fragments")
    ground_truth = _validate_label_image(ground_truth, "ground_truth")
    if ground_truth.shape != fragments.shape:
        raise ValueError("ground_truth and fragments must have the same shape")
    edges = np.asarray(edges, dtype=np.int64)
    if edges.shape != (len(edges), 2):
        raise ValueError("edges must have shape (N, 2)")

    fragment_labels, counts, pixel_index = _fragment_index(fragments)
    label_to_index = {int(label): index for index, label in enumerate(fragment_labels.tolist())}
    major, purity = _major_labels(pixel_index, ground_truth, counts)
    edge_major = np.zeros((len(edges), 2), dtype=np.int64)
    edge_purity = np.zeros((len(edges), 2), dtype=np.float32)
    labels = np.zeros(len(edges), dtype=np.uint8)
    for edge_index, (label_a, label_b) in enumerate(edges.tolist()):
        if int(label_a) not in label_to_index or int(label_b) not in label_to_index:
            raise ValueError(f"edge references an unknown fragment: {(label_a, label_b)}")
        index_a = label_to_index[int(label_a)]
        index_b = label_to_index[int(label_b)]
        major_a, major_b = int(major[index_a]), int(major[index_b])
        purity_a, purity_b = float(purity[index_a]), float(purity[index_b])
        edge_major[edge_index] = (major_a, major_b)
        edge_purity[edge_index] = (purity_a, purity_b)
        labels[edge_index] = np.uint8(
            purity_a >= PURITY_THRESHOLD
            and purity_b >= PURITY_THRESHOLD
            and major_a != 0
            and major_a == major_b
        )
    return labels, edge_major, edge_purity


def extract_edge_features(
    fragments: np.ndarray,
    states_by_scale: Mapping[str, Mapping[str, np.ndarray]],
    segmentations_by_scale: Mapping[str, np.ndarray],
    raw: np.ndarray,
    *,
    ground_truth: np.ndarray | None = None,
) -> EdgeFeatureBatch:
    """Extract the fixed 55-dimensional feature vector for every RAG edge."""
    fragments = _validate_label_image(fragments, "fragments")
    shape = fragments.shape
    raw = _validate_dense_array(raw, shape, "raw")
    if set(states_by_scale) != set(SCALE_NAMES):
        raise ValueError(f"states_by_scale must contain exactly {SCALE_NAMES}")
    if set(segmentations_by_scale) != set(SCALE_NAMES):
        raise ValueError(f"segmentations_by_scale must contain exactly {SCALE_NAMES}")

    states: dict[str, dict[str, np.ndarray]] = {}
    segmentations: dict[str, np.ndarray] = {}
    for scale in SCALE_NAMES:
        if set(states_by_scale[scale]) != set(CHANNEL_NAMES):
            raise ValueError(f"state {scale!r} must contain exactly {CHANNEL_NAMES}")
        states[scale] = {
            channel: _validate_dense_array(states_by_scale[scale][channel], shape, f"{scale}/{channel}")
            for channel in CHANNEL_NAMES
        }
        segmentation = _validate_label_image(segmentations_by_scale[scale], scale)
        if segmentation.shape != shape:
            raise ValueError(f"segmentation {scale!r} must have shape {shape}")
        segmentations[scale] = segmentation

    adjacency = four_neighbor_adjacency(fragments)
    edges = np.asarray(list(adjacency), dtype=np.int64).reshape(-1, 2)
    fragment_labels, counts, pixel_index = _fragment_index(fragments)
    label_to_index = {int(label): index for index, label in enumerate(fragment_labels.tolist())}

    statistics = {
        scale: {
            channel: _array_statistics(pixel_index, states[scale][channel], counts)
            for channel in CHANNEL_NAMES
        }
        for scale in SCALE_NAMES
    }
    votes = {
        scale: _major_labels(pixel_index, segmentations[scale], counts)
        for scale in SCALE_NAMES
    }
    disagreements = {
        "fg_disagreement_512_384": _array_statistics(
            pixel_index,
            np.abs(states["tile512_overlap"]["foreground"] - states["tile384_overlap"]["foreground"]),
            counts,
        ),
        "fg_disagreement_512_256": _array_statistics(
            pixel_index,
            np.abs(states["tile512_overlap"]["foreground"] - states["tile256_overlap"]["foreground"]),
            counts,
        ),
    }

    feature_rows: list[list[float]] = []
    for label_a, label_b in edges.tolist():
        index_a = label_to_index[int(label_a)]
        index_b = label_to_index[int(label_b)]
        area_a = int(counts[index_a])
        area_b = int(counts[index_b])
        contacts = adjacency[(int(label_a), int(label_b))]
        endpoint_rows = np.concatenate((contacts[:, 0], contacts[:, 2]))
        endpoint_columns = np.concatenate((contacts[:, 1], contacts[:, 3]))
        row: dict[str, float] = {
            "area_a": float(area_a),
            "area_b": float(area_b),
            "area_union": float(area_a + area_b),
            "contact_length": float(len(contacts)),
            "is_small_pair": float(area_a < 128 and area_b < 128),
            "is_medium_pair": float((128 <= area_a <= 1024) or (128 <= area_b <= 1024)),
            "is_large_pair": float(area_a > 1024 or area_b > 1024),
        }
        for scale in SCALE_NAMES:
            for channel in CHANNEL_NAMES:
                values = states[scale][channel]
                edge_values = values[endpoint_rows, endpoint_columns]
                means, standard_deviations = statistics[scale][channel]
                union_mean, union_std = _combined_mean_std(
                    index_a,
                    index_b,
                    means,
                    standard_deviations,
                    counts,
                )
                row[f"mean_{channel}_{scale}_edge"] = float(np.mean(edge_values))
                row[f"std_{channel}_{scale}_edge"] = float(np.std(edge_values))
                row[f"mean_{channel}_{scale}_fragments"] = union_mean
                row[f"std_{channel}_{scale}_fragments"] = union_std
        for scale in SCALE_NAMES:
            major, purities = votes[scale]
            major_a = int(major[index_a])
            major_b = int(major[index_b])
            row[f"same_label_vote_{scale}"] = float(major_a != 0 and major_a == major_b)
            row[f"purity_vote_a_{scale}"] = float(purities[index_a])
            row[f"purity_vote_b_{scale}"] = float(purities[index_b])
        for name, (means, standard_deviations) in disagreements.items():
            row[name] = _combined_mean_std(
                index_a,
                index_b,
                means,
                standard_deviations,
                counts,
            )[0]
        row["raw_edge_contrast"] = float(
            np.mean(np.abs(raw[contacts[:, 0], contacts[:, 1]] - raw[contacts[:, 2], contacts[:, 3]]))
        )
        if set(row) != set(RAG_FEATURE_NAMES):
            missing = sorted(set(RAG_FEATURE_NAMES) - set(row))
            extra = sorted(set(row) - set(RAG_FEATURE_NAMES))
            raise RuntimeError(f"feature schema mismatch: missing={missing}, extra={extra}")
        feature_rows.append([row[name] for name in RAG_FEATURE_NAMES])

    features = np.asarray(feature_rows, dtype=np.float32).reshape(-1, len(RAG_FEATURE_NAMES))
    labels = None
    edge_major = None
    edge_purity = None
    if ground_truth is not None:
        labels, edge_major, edge_purity = purity_edge_labels(fragments, ground_truth, edges)
    return EdgeFeatureBatch(
        edges=edges,
        features=features,
        labels=labels,
        major_ground_truth=edge_major,
        purities=edge_purity,
    )


def _validate_feature_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(RAG_FEATURE_NAMES):
        raise ValueError("features must have shape (N, 55)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features contain non-finite values")
    return matrix


def fit_random_forest(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_jobs: int = 1,
    output_path: str | Path | None = None,
    training_metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Fit the fixed random forest and optionally save a validated bundle."""
    from sklearn.ensemble import RandomForestClassifier

    matrix = _validate_feature_matrix(features)
    if not 1 <= int(n_jobs) <= 8:
        raise ValueError("n_jobs must be between 1 and 8")
    target = np.asarray(labels)
    if target.shape != (len(matrix),):
        raise ValueError("labels must have shape (N,)")
    if not np.all(np.isin(target, (0, 1))) or set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError("random-forest training requires both edge classes")
    target = target.astype(np.uint8, copy=False)
    parameters = {**RF_PARAMETERS, "n_jobs": int(n_jobs)}
    model = RandomForestClassifier(**parameters)
    model.fit(matrix, target)
    if output_path is not None:
        _save_random_forest(output_path, model, parameters, training_metadata)
    return model


def _save_random_forest(path: str | Path, model: Any, parameters: Mapping[str, Any],
                        training_metadata: Mapping[str, Any] | None = None) -> None:
    from joblib import dump

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": model,
            "feature_names": RAG_FEATURE_NAMES,
            "rf_parameters": dict(parameters),
            "merge_threshold": MERGE_THRESHOLD,
            "min_instance_area": MIN_INSTANCE_AREA,
            "training_metadata": dict(training_metadata or {}),
        },
        output_path,
    )


def load_random_forest(path: str | Path) -> Any:
    """Load a random-forest bundle and reject incompatible feature schemas."""
    from joblib import load

    payload = load(Path(path))
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("invalid random-forest bundle")
    feature_names = payload.get("feature_names", payload.get("feature_columns", ()))
    if tuple(feature_names) != RAG_FEATURE_NAMES:
        raise ValueError("random-forest bundle uses an incompatible feature schema")
    if "merge_threshold" in payload and float(payload["merge_threshold"]) != MERGE_THRESHOLD:
        raise ValueError("random-forest bundle uses an incompatible merge threshold")
    if "min_instance_area" in payload and int(payload["min_instance_area"]) != MIN_INSTANCE_AREA:
        raise ValueError("random-forest bundle uses an incompatible minimum area")
    model = payload["model"]
    if int(getattr(model, "n_features_in_", -1)) != len(RAG_FEATURE_NAMES):
        raise ValueError("random-forest model does not accept 55 features")
    if set(np.asarray(getattr(model, "classes_", ())).tolist()) != {0, 1}:
        raise ValueError("random-forest model must contain edge classes 0 and 1")
    return model


def _positive_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.classes_)
    positive_columns = np.flatnonzero(classes == 1)
    if probabilities.shape != (len(features), len(classes)) or len(positive_columns) != 1:
        raise ValueError("model must expose binary predict_proba with positive class 1")
    return probabilities[:, int(positive_columns[0])]


def _filter_and_relabel(segmentation: np.ndarray) -> np.ndarray:
    output = np.zeros(segmentation.shape, dtype=np.uint32)
    labels, counts = np.unique(segmentation, return_counts=True)
    next_label = 1
    for label, count in zip(labels.tolist(), counts.tolist()):
        if int(label) == 0 or int(count) < MIN_INSTANCE_AREA:
            continue
        output[segmentation == label] = np.uint32(next_label)
        next_label += 1
    return output


def merge_fragments(
    fragments: np.ndarray,
    edges: np.ndarray,
    features: np.ndarray,
    model: Any,
) -> np.ndarray:
    """Merge edges with ``p(same) >= 0.65`` and remove instances below 16 px."""
    fragments = _validate_label_image(fragments, "fragments")
    edges = np.asarray(edges, dtype=np.int64)
    if edges.shape != (len(edges), 2):
        raise ValueError("edges must have shape (N, 2)")
    matrix = _validate_feature_matrix(features)
    if len(matrix) != len(edges):
        raise ValueError("edges and features must have the same number of rows")

    labels = np.unique(fragments)
    labels = labels[labels > 0].astype(np.int64, copy=False)
    parent = {int(label): int(label) for label in labels.tolist()}

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(label_a: int, label_b: int) -> None:
        if label_a not in parent or label_b not in parent:
            raise ValueError(f"edge references an unknown fragment: {(label_a, label_b)}")
        root_a = find(label_a)
        root_b = find(label_b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    if len(edges):
        probabilities = _positive_probabilities(model, matrix)
        for (label_a, label_b), probability in zip(edges.tolist(), probabilities.tolist()):
            if float(probability) >= MERGE_THRESHOLD:
                union(int(label_a), int(label_b))

    merged = np.zeros(fragments.shape, dtype=np.int64)
    for label in labels.tolist():
        merged[fragments == label] = find(int(label))
    return _filter_and_relabel(merged)
