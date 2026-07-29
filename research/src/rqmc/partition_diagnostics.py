"""Utilities for reproducible disjoint partitions of a fixed point set."""
from __future__ import annotations

import math

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


def _bit_reversal_permutation(n_points: int) -> np.ndarray:
    """Order integers by their fixed-width base-2 digit reversal."""
    bits = max(1, (n_points - 1).bit_length())
    values = np.arange(n_points, dtype=np.uint32)
    work = values.copy()
    reversed_values = np.zeros(n_points, dtype=np.uint32)
    for _ in range(bits):
        reversed_values = (reversed_values << 1) | (work & 1)
        work >>= 1
    return np.argsort(reversed_values, kind="stable").astype(np.int32)


def structured_six_way_partitions(
    n_points: int,
    n_groups: int,
    affine_multiplier: int,
    affine_offset: int,
) -> dict[str, np.ndarray]:
    """Construct fixed, non-learned index partitions with distinct structure."""
    if n_points <= 0 or n_groups <= 0:
        raise ValueError("n_points and n_groups must be positive")
    if n_points % n_groups:
        raise ValueError("n_points must be divisible by n_groups")
    if n_points % 2:
        raise ValueError("antithetic partitions require an even point count")
    group_size = n_points // n_groups
    if group_size % 2:
        raise ValueError("antithetic groups require an even group size")
    if math.gcd(affine_multiplier, n_points) != 1:
        raise ValueError("affine_multiplier must be coprime to n_points")

    indices = np.arange(n_points, dtype=np.int64)
    residue = np.stack(
        [indices[group::n_groups] for group in range(n_groups)],
        axis=0,
    ).astype(np.int32)
    contiguous = indices.reshape(n_groups, group_size).astype(np.int32)
    half_groups = indices[: n_points // 2].reshape(
        n_groups,
        group_size // 2,
    )
    antithetic = np.concatenate(
        [half_groups, half_groups + n_points // 2],
        axis=1,
    ).astype(np.int32)
    affine_order = (
        affine_multiplier * indices + affine_offset
    ) % n_points
    affine = affine_order.reshape(n_groups, group_size).astype(np.int32)
    bit_reversed = _bit_reversal_permutation(n_points).reshape(
        n_groups,
        group_size,
    )
    return {
        "true_residue_cosets": residue,
        "contiguous_blocks": contiguous,
        "antithetic_pair_blocks": antithetic,
        "affine_index_blocks": affine,
        "bit_reversed_blocks": bit_reversed,
    }


def validate_disjoint_partition(partition: np.ndarray, n_points: int) -> None:
    """Raise when a proposed partition does not cover each point exactly once."""
    indices = np.asarray(partition)
    if indices.ndim != 2:
        raise ValueError("partition must have shape (groups, group_size)")
    if indices.size != n_points:
        raise ValueError("partition does not contain n_points entries")
    if not np.array_equal(np.sort(indices.reshape(-1)), np.arange(n_points)):
        raise ValueError("partition must contain each point exactly once")
