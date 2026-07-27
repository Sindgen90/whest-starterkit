"""Correctness tests for the embedded-lattice coset diagnostic."""
from __future__ import annotations

import math

import numpy as np
import pytest

from research.scripts.run_rqmc_coset_diagnostic import (
    combinations_by_size,
    coset_means,
    equivalent_flops,
)


def test_coset_means_reconstruct_full_average() -> None:
    rng = np.random.default_rng(7)
    activations = rng.normal(size=(60, 8))
    means = coset_means(activations, n_cosets=6)
    np.testing.assert_allclose(means.mean(axis=0), activations.mean(axis=0), rtol=0, atol=1e-15)


def test_coset_means_use_residue_classes() -> None:
    activations = np.arange(24, dtype=np.float64)[:, None]
    actual = coset_means(activations, n_cosets=6)[:, 0]
    expected = np.asarray(
        [np.arange(residue, 24, 6).mean() for residue in range(6)],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(actual, expected)


def test_coset_shape_validation() -> None:
    with pytest.raises(ValueError, match="shape"):
        coset_means(np.ones(24), n_cosets=6)
    with pytest.raises(ValueError, match="divisible"):
        coset_means(np.ones((25, 2)), n_cosets=6)


def test_combination_counts() -> None:
    combinations = combinations_by_size()
    assert len(combinations[1]) == 6
    assert len(combinations[2]) == math.comb(6, 2)
    assert len(combinations[3]) == math.comb(6, 3)
    assert combinations[6] == [(0, 1, 2, 3, 4, 5)]


def test_equivalent_flop_model_matches_frozen_measurements() -> None:
    assert equivalent_flops(4_096) == 17_322_488_832
    assert equivalent_flops(24_576) == 103_934_927_872
