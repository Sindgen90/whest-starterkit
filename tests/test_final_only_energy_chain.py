"""Correctness tests for final-only uncentered second-moment propagation."""
from __future__ import annotations

import math
from types import SimpleNamespace

import flopscope.numpy as fnp
import numpy as np

from submissions.final_only_energy_chain.estimator import Estimator


def make_mlp(weights: np.ndarray, seed: int = 0) -> SimpleNamespace:
    depth, width, _ = weights.shape
    stacked = fnp.array(weights.astype(np.float32))
    return SimpleNamespace(
        width=width,
        depth=depth,
        weights=[stacked[layer] for layer in range(depth)],
        seed=seed,
        name=f"energy-chain-{seed}",
    )


def reference_prediction(weights: np.ndarray) -> np.ndarray:
    depth, width, _ = weights.shape
    q = np.ones(width, dtype=np.float64)
    for w in weights:
        final_variance = (w * w).T @ q
        q = 0.5 * final_variance
    output = np.zeros((depth, width), dtype=np.float64)
    output[-1] = np.sqrt(np.maximum(final_variance, 0.0) / (2.0 * math.pi))
    return output


def test_matches_numpy_energy_recurrence() -> None:
    rng = np.random.default_rng(7)
    weights = rng.normal(scale=0.3, size=(4, 8, 8))
    actual = np.asarray(Estimator().predict(make_mlp(weights), 1), dtype=np.float64)
    np.testing.assert_allclose(actual, reference_prediction(weights), rtol=2e-6, atol=1e-8)


def test_single_layer_mean_is_exact_for_gaussian_input() -> None:
    weights = np.array(
        [[[1.0, 0.0], [0.0, 2.0]]],
        dtype=np.float64,
    )
    actual = np.asarray(Estimator().predict(make_mlp(weights), 1), dtype=np.float64)
    expected = np.array([[1.0, 2.0]]) / math.sqrt(2.0 * math.pi)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-8)


def test_only_final_layer_is_populated() -> None:
    weights = np.ones((3, 4, 4), dtype=np.float64) * 0.25
    actual = np.asarray(Estimator().predict(make_mlp(weights), 1), dtype=np.float64)
    np.testing.assert_array_equal(actual[:-1], 0.0)
    assert np.all(actual[-1] > 0.0)
