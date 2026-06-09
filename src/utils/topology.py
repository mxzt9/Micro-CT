from __future__ import annotations

from math import ceil

import numpy as np

from .dependencies import require_gudhi


TOPOLOGY_FEATURE_DIM = 6


def _limit_volume_size(volume: np.ndarray, max_size: int | None) -> np.ndarray:
    if max_size is None:
        return volume
    max_size = int(max_size)
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    largest = max(volume.shape)
    if largest <= max_size:
        return volume
    step = int(ceil(largest / max_size))
    return volume[::step, ::step, ::step]


def _prepare_filtration(volume: np.ndarray) -> np.ndarray:
    if volume.dtype == np.bool_:
        return np.where(volume, 0.0, 1.0).astype(np.float64, copy=False)

    filtration = np.asarray(volume, dtype=np.float64)
    if not np.isfinite(filtration).all():
        filtration = np.nan_to_num(filtration, copy=False)

    lo = float(filtration.min())
    hi = float(filtration.max())
    if hi > lo:
        filtration = (filtration - lo) / (hi - lo)
    else:
        filtration = np.zeros_like(filtration, dtype=np.float64)
    return filtration


def cubical_persistence_summary(
    volume: np.ndarray,
    *,
    max_size: int | None = 32,
) -> np.ndarray:
    """Return [H0_count, H0_life_sum, H0_life_max, H1_count, H1_life_sum, H1_life_max].

    Numeric grayscale volumes are normalized and used directly as a lower-star
    filtration. Boolean masks treat True voxels as early cells and False voxels
    as late cells, which keeps target topology out of the model input path.
    """

    gudhi = require_gudhi()
    arr = np.asarray(volume)
    if arr.ndim != 3:
        raise ValueError("cubical_persistence_summary expects a 3D volume")
    if arr.size == 0:
        return np.zeros(TOPOLOGY_FEATURE_DIM, dtype=np.float32)

    arr = _limit_volume_size(arr, max_size=max_size)
    filtration = _prepare_filtration(arr)
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=filtration)
    complex_.persistence(homology_coeff_field=2, min_persistence=0.0)

    features: list[float] = []
    for dim in (0, 1):
        intervals = np.asarray(complex_.persistence_intervals_in_dimension(dim), dtype=np.float64)
        if intervals.size == 0:
            features.extend([0.0, 0.0, 0.0])
            continue
        intervals = intervals.reshape(-1, 2)
        finite = np.isfinite(intervals[:, 1])
        lifetimes = np.maximum(intervals[finite, 1] - intervals[finite, 0], 0.0)
        if lifetimes.size == 0:
            features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([float(lifetimes.size), float(lifetimes.sum()), float(lifetimes.max())])

    return np.asarray(features, dtype=np.float32)
