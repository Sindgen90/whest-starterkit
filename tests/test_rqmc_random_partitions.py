"""Tests for the repeated random-partition diagnostic."""
from __future__ import annotations

import numpy as np
import pytest

from research.src.rqmc.partition_diagnostics import (
    partition_means,
    partition_mse,
    random_disjoint_partitions,
)


def test_random_partitions_are_reproducible_disjoint_covers() -> None:
    first = random_disjoint_partitions(24, 6, 5, seed=17)
    second = random_disjoint_partitions(24, 6, 5, seed=17)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (5, 6, 4)
    for partition in first:
        np.testing.assert_array_equal(
            np.sort(partition.reshape(-1)),
            np.arange(24),
        )
    assert not np.array_equal(first[0], first[1])


def test_partition_group_average_reconstructs_full_average() -> None:
    rng = np.random.default_rng(9)
    values = rng.normal(size=(60, 7))
    partitions = random_disjoint_partitions(60, 6, 8, seed=11)
    means = partition_means(values, partitions)
    expected = np.repeat(values.mean(axis=0)[None, :], 8, axis=0)
    np.testing.assert_allclose(means.mean(axis=1), expected, atol=1e-15)


def test_partition_mse_matches_manual_group() -> None:
    values = np.arange(48, dtype=np.float64).reshape(24, 2)
    target = np.asarray([20.0, 21.0])
    partitions = random_disjoint_partitions(24, 6, 3, seed=3)
    actual = partition_mse(values, target, partitions)
    group = values[partitions[1, 4]].mean(axis=0)
    expected = np.mean((group - target) ** 2)
    assert actual[1, 4] == pytest.approx(expected)


def test_partition_validation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        random_disjoint_partitions(25, 6, 2, seed=0)
    with pytest.raises(ValueError, match="shape"):
        partition_means(np.ones(24), np.arange(24).reshape(1, 6, 4))
