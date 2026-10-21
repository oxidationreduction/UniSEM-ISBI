from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def cremi_score(
    voi_split: float,
    voi_merge: float,
    adapted_rand_error: float,
) -> float:
    """Compute CREMI as sqrt((VOI split + VOI merge) * ARAND error)."""

    voi_sum = float(voi_split) + float(voi_merge)
    arand = float(adapted_rand_error)
    if not math.isfinite(voi_sum) or not math.isfinite(arand):
        raise ValueError("CREMI components must be finite")
    if voi_sum < 0.0 or arand < 0.0:
        raise ValueError("CREMI components must be non-negative")
    return float(math.sqrt(voi_sum * arand))


def evaluate_segmentation(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, Any]:
    """Evaluate an instance segmentation with the official ELF metrics."""

    try:
        from elf.evaluation import rand_index, variation_of_information
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "UniSEM metrics require python-elf with elf.evaluation"
        ) from error

    prediction = np.asarray(prediction)
    ground_truth = np.asarray(ground_truth)
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Prediction and ground-truth shapes differ: "
            f"{prediction.shape} vs {ground_truth.shape}"
        )
    if prediction.size == 0:
        raise ValueError("Cannot evaluate empty arrays")
    if int(prediction.min()) < 0 or int(ground_truth.min()) < 0:
        raise ValueError("Instance labels must be non-negative")
    from .data import component_pseudo_relabel_2d

    if prediction.ndim != 2:
        raise ValueError("Paper evaluation is 2D; use evaluate_per_slice for stacks")
    prediction = component_pseudo_relabel_2d(prediction, min_component_area=1)
    ground_truth = component_pseudo_relabel_2d(ground_truth, min_component_area=1)

    adapted_rand_error, rand_score = rand_index(prediction, ground_truth)
    voi_split, voi_merge = variation_of_information(prediction, ground_truth)
    result = {
        "voi_split": float(voi_split),
        "voi_merge": float(voi_merge),
        "voi_sum": float(voi_split + voi_merge),
        "adapted_rand_error": float(adapted_rand_error),
        "arand": float(adapted_rand_error),
        "rand_index": float(rand_score),
        "pred_num_instances": int(
            np.count_nonzero(np.unique(prediction) != 0)
        ),
        "gt_num_instances": int(
            np.count_nonzero(np.unique(ground_truth) != 0)
        ),
    }
    result["cremi"] = cremi_score(
        result["voi_split"],
        result["voi_merge"],
        result["adapted_rand_error"],
    )
    return result


def aggregate_slice_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-slice metrics without conflating two CREMI summaries.

    ``cremi_from_aggregate_components`` applies the CREMI formula after
    averaging VOI and ARAND components. ``mean_per_slice_cremi`` is the
    arithmetic mean of CREMI computed independently for each slice.
    """

    if not rows:
        raise ValueError("Cannot aggregate an empty metric sequence")

    def mean(key: str) -> float:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values found for {key}")
        return float(values.mean())

    mean_voi_split = mean("voi_split")
    mean_voi_merge = mean("voi_merge")
    mean_arand = mean("adapted_rand_error")
    result = {
        "num_slices": int(len(rows)),
        "mean_voi_split": mean_voi_split,
        "mean_voi_merge": mean_voi_merge,
        "mean_voi_sum": float(mean_voi_split + mean_voi_merge),
        "mean_adapted_rand_error": mean_arand,
        "mean_arand": mean_arand,
        "mean_rand_index": mean("rand_index"),
        "cremi_from_aggregate_components": cremi_score(
            mean_voi_split,
            mean_voi_merge,
            mean_arand,
        ),
        "mean_per_slice_cremi": mean("cremi"),
    }
    return result


def evaluate_per_slice(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, Any]:
    """Evaluate a 2D map or a ZYX stack and return rows plus aggregation."""

    prediction = np.asarray(prediction)
    ground_truth = np.asarray(ground_truth)
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Prediction and ground-truth shapes differ: "
            f"{prediction.shape} vs {ground_truth.shape}"
        )
    if prediction.ndim == 2:
        prediction = prediction[None]
        ground_truth = ground_truth[None]
    elif prediction.ndim != 3:
        raise ValueError(f"Expected a 2D or ZYX label array, got {prediction.shape}")

    rows: list[dict[str, Any]] = []
    for slice_index in range(int(prediction.shape[0])):
        row = evaluate_segmentation(
            prediction[slice_index],
            ground_truth[slice_index],
        )
        rows.append({"slice_index": slice_index, **row})
    return {
        "slices": rows,
        "aggregate": aggregate_slice_metrics(rows),
    }


def aggregate_volumes(
    volumes: Mapping[str, Mapping[str, Any]],
    *,
    expected_volumes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Average volume means equally after evaluating slices within each volume."""
    if not volumes or (expected_volumes is not None and set(volumes) != set(expected_volumes)):
        raise ValueError("Missing, duplicate or unexpected evaluation volumes")
    summaries = [value.get("aggregate", value) for value in volumes.values()]
    means = {}
    keys = ("mean_voi_split", "mean_voi_merge", "mean_voi_sum", "mean_arand", "mean_per_slice_cremi")
    for key in keys:
        values = np.asarray([row[key] for row in summaries], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite volume values for {key}")
        means[key] = float(values.mean())
    means["cremi_from_aggregate_components"] = cremi_score(
        means["mean_voi_split"], means["mean_voi_merge"], means["mean_arand"],
    )
    return {"aggregation": "equal_volume_mean_of_per_slice_means", "num_volumes": len(summaries),
            "num_slices": sum(int(row["num_slices"]) for row in summaries), **means}


__all__ = [
    "aggregate_slice_metrics",
    "cremi_score",
    "evaluate_per_slice",
    "evaluate_segmentation",
    "aggregate_volumes",
]
