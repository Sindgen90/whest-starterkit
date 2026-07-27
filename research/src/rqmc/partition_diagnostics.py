"""Utilities for reproducible disjoint partitions of a fixed point set."""
from __future__ import annotations

import numpy as np


def random_disjoint_partitions(
    n_points: int,
    n_groups: int,
    n_partitions: int,
    seed: int,
) -> np.ndarray:
    """Return independent random partitions with equal-sized disjoint groups."""
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    if n_partitions <= 0:
        raise ValueError("n_partitions must be positive")
    if n_points % n_groups:
        raise ValueError("n_points must be divisible by n_groups")
    rng = np.random.default_rng(seed)
    group_size = n_points // n_groups
    partitions = np.empty(
        (n_partitions, n_groups, group_size),
        dtype=np.int32,
    )
    for partition in range(n_partitions):
        partitions[partition] = rng.permutation(n_points).reshape(
            n_groups,
            group_size,
        )
    return partitions


def partition_means(
    values: np.ndarray,
    partitions: np.ndarray,
) -> np.ndarray:
    """Average values for every group in every partition."""
    array = np.asarray(values)
    indices = np.asarray(partitions)
    if array.ndim != 2:
        raise ValueError("values must have shape (points, outputs)")
    if indices.ndim != 3:
        raise ValueError(
            "partitions must have shape (partitions, groups, group_size)"
        )
    n_partitions, n_groups, group_size = indices.shape
    if n_groups * group_size != len(array):
        raise ValueError("each partition must cover every point exactly once")
    if indices.min() < 0 or indices.max() >= len(array):
        raise ValueError("partition index is out of range")

    means = np.empty(
        (n_partitions, n_groups, array.shape[1]),
        dtype=np.float64,
    )
    for partition in range(n_partitions):
        ordered = array[indices[partition].reshape(-1)]
        means[partition] = ordered.reshape(
            n_groups,
            group_size,
            array.shape[1],
        ).mean(axis=1, dtype=np.float64)
    return means


def partition_mse(
    values: np.ndarray,
    target: np.ndarray,
    partitions: np.ndarray,
) -> np.ndarray:
    """Return final-output MSE for every group in every partition."""
    means = partition_means(values, partitions)
    target_array = np.asarray(target, dtype=np.float64)
    if target_array.shape != (values.shape[1],):
        raise ValueError("target must have shape (outputs,)")
    return np.mean((means - target_array[None, None, :]) ** 2, axis=2)
