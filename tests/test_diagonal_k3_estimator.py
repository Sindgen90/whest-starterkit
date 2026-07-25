"""Tests for the cheap diagonal-only K3 experiment."""
from __future__ import annotations

from types import SimpleNamespace

import flopscope.numpy as fnp
import numpy as np

from experiments.diagonal_k3_estimator import Estimator


def make_mlp(seed: int = 0, width: int = 8, depth: int = 4) -> SimpleNamespace:
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
