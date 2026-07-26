"""Tests for the final-layer conditional-Gaussian RQMC estimator."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import flopscope.numpy as fnp
import numpy as np

from submissions.rqmc_conditional_gaussian_24576 import estimator as conditional


def load_baseline_module():
    path = Path("submissions/rqmc_lattice_002/estimator.py")
    spec = importlib.util.spec_from_file_location("rqmc_baseline_for_conditional_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_mlp(seed: int = 0, width: int = 8, depth: int = 4) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    weights = fnp.array(
        rng.normal(scale=math.sqrt(2.0 / width), size=(depth, width, width)).astype(np.float32)
    )
    return SimpleNamespace(
        width=width,
        depth=depth,
        weights=[weights[layer] for layer in range(depth)],
        seed=seed,
        name=f"conditional-{seed}",
    )


def test_gaussian_relu_mean_matches_closed_form() -> None:
    mean = fnp.array([-1.0, 0.0, 2.0], dtype=fnp.float64)
    variance = fnp.array([0.25, 1.0, 4.0], dtype=fnp.float64)
    actual = np.asarray(conditional.gaussian_relu_mean(mean, variance), dtype=np.float64)
    sigma = np.sqrt(np.asarray(variance))
    alpha = np.asarray(mean) / sigma
    expected = np.asarray(mean) * (
        0.5 * (1.0 + np.vectorize(math.erf)(alpha / math.sqrt(2.0)))
    ) + sigma * np.exp(-0.5 * alpha * alpha) / math.sqrt(2.0 * math.pi)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)


def test_zero_variance_uses_deterministic_relu() -> None:
    mean = fnp.array([-2.0, 0.0, 3.0], dtype=fnp.float64)
    variance = fnp.zeros(3, dtype=fnp.float64)
    actual = np.asarray(conditional.gaussian_relu_mean(mean, variance), dtype=np.float64)
    np.testing.assert_array_equal(actual, [0.0, 0.0, 3.0])


def test_layers_before_final_match_ordinary_rqmc() -> None:
    baseline = load_baseline_module()
    baseline.N_SAMPLES = 257
    baseline.RQMC_SEED_OFFSET = conditional.RQMC_SEED_OFFSET
    baseline.RQMC_MULTIPLIER = conditional.RQMC_MULTIPLIER
    old_count = conditional.N_SAMPLES
    conditional.N_SAMPLES = 257
    try:
        mlp = make_mlp()
        ordinary = np.asarray(baseline.Estimator().predict(mlp, 1), dtype=np.float64)
        corrected = np.asarray(conditional.Estimator().predict(mlp, 1), dtype=np.float64)
    finally:
        conditional.N_SAMPLES = old_count
    np.testing.assert_allclose(corrected[:-1], ordinary[:-1], rtol=0.0, atol=0.0)
    assert corrected.shape == ordinary.shape
    assert np.all(np.isfinite(corrected))
