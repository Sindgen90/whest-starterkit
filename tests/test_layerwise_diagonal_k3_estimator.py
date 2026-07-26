"""Tests for layerwise diagonal K3 cohort propagation."""
from __future__ import annotations

from types import SimpleNamespace

import flopscope.numpy as fnp
import numpy as np

from experiments.layerwise_diagonal_k3_estimator import Estimator


def make_mlp(seed: int = 0, width: int = 8, depth: int = 5) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    weights = fnp.array(rng.normal(scale=0.2, size=(depth, width, width)).astype(np.float32))
    return SimpleNamespace(
        width=width,
        depth=depth,
        weights=[weights[layer] for layer in range(depth)],
        seed=seed,
        name=f"test-{seed}",
    )


def test_zero_cutoff_ignores_third_scale() -> None:
    mlp = make_mlp()
    low = np.asarray(Estimator(third_scale=-0.001, max_k3_layer=0).predict(mlp, 1), dtype=np.float64)
    high = np.asarray(Estimator(third_scale=-10.0, max_k3_layer=0).predict(mlp, 1), dtype=np.float64)
    np.testing.assert_allclose(low, high, rtol=0.0, atol=0.0)


def test_first_layer_has_no_k3_correction_from_gaussian_input() -> None:
    mlp = make_mlp()
    no_k3 = np.asarray(Estimator(third_scale=-0.001, max_k3_layer=0).predict(mlp, 1), dtype=np.float64)
    with_k3 = np.asarray(Estimator(third_scale=-10.0, max_k3_layer=32).predict(mlp, 1), dtype=np.float64)
    np.testing.assert_allclose(no_k3[0], with_k3[0], rtol=0.0, atol=0.0)


def test_layerwise_cohort_count_grows_one_per_active_layer() -> None:
    mlp = make_mlp(depth=5)
    estimator = Estimator(third_scale=-0.3, max_k3_layer=3, generic_horizon=None)
    estimator.predict(mlp, 1)

    assert [row["cohort_count"] for row in estimator.cohort_history] == [1, 2, 3, 0, 0]
    assert estimator.cohort_history[2]["creation_layers"] == [0, 1, 2]


def test_horizon_limits_layerwise_cohort_count() -> None:
    mlp = make_mlp(depth=5)
    estimator = Estimator(third_scale=-0.3, max_k3_layer=5, generic_horizon=1)
    estimator.predict(mlp, 1)

    assert [row["cohort_count"] for row in estimator.cohort_history] == [1, 2, 2, 2, 2]
    assert estimator.cohort_history[-1]["creation_layers"] == [3, 4]
