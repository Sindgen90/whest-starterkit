"""Tests for positive empirical-cubature utilities."""
from __future__ import annotations

import numpy as np

from research.scripts.run_learned_cubature_4k import selected_lattice_points
from research.src.cubature.empirical import (
    mixed_leverage_subset,
    positive_moment_weights,
    project_simplex,
    standardize_rows,
    top_right_singular_modes,
)


def test_simplex_projection() -> None:
    projected = project_simplex(np.asarray([-1.0, 0.2, 2.0]))
    assert np.all(projected >= 0.0)
    assert np.isclose(projected.sum(), 1.0)


def test_singular_modes_and_leverage_are_reproducible() -> None:
    rng = np.random.default_rng(3)
    snapshots = rng.normal(size=(8, 40))
    standardized, _, _ = standardize_rows(snapshots)
    modes, singular = top_right_singular_modes(standardized, rank=4)
    assert modes.shape == (4, 40)
    assert np.all(singular > 0.0)
    gaussian = rng.normal(size=(6, 40))
    first = mixed_leverage_subset(modes, gaussian, support=12, seed=19)
    repeated = mixed_leverage_subset(modes, gaussian, support=12, seed=19)
    np.testing.assert_array_equal(first, repeated)


def test_positive_weights_match_simple_moments() -> None:
    nodes = np.linspace(-1.0, 1.0, 41)
    features = np.stack((nodes, nodes * nodes - 1.0 / 3.0))
    target = np.zeros(2)
    selected = np.arange(len(nodes))
    active, weights, diagnostics = positive_moment_weights(features, target, selected)
    assert diagnostics["max_constraint_residual"] < 1e-7
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0)
    np.testing.assert_allclose(features[:, active] @ weights, target, atol=1e-7)


def test_selected_lattice_generation_matches_full_pool() -> None:
    from submissions.rqmc_lattice_002 import estimator as source

    source.RQMC_SEED_OFFSET = 8_675_309
    source.RQMC_MULTIPLIER = 17_797
    selected = np.asarray([0, 17, 4095, 24_575], dtype=np.int64)
    full = np.asarray(source.normal_lattice_samples(24_576, 8, 0))
    subset = np.asarray(selected_lattice_points(selected, 8))
    np.testing.assert_array_equal(subset, full[selected])
