"""Correctness tests for high-sample radial Rao-Blackwellized RQMC."""
from __future__ import annotations

import math
from types import SimpleNamespace

import flopscope.numpy as fnp
import numpy as np
import pytest

from submissions.rqmc_radial_24576 import estimator as radial


def make_mlp(
    weights: list[np.ndarray],
    seed: int = 0,
    biases: list[np.ndarray] | None = None,
) -> SimpleNamespace:
    width = weights[0].shape[0]
    return SimpleNamespace(
        width=width,
        depth=len(weights),
        weights=[fnp.array(weight, dtype=fnp.float32) for weight in weights],
        biases=None
        if biases is None
        else [fnp.array(bias, dtype=fnp.float32) for bias in biases],
        seed=seed,
        name=f"radial-{seed}",
    )


def forward(samples: np.ndarray, weights: list[np.ndarray]) -> np.ndarray:
    activations = samples
    for weight in weights:
        activations = np.maximum(activations @ weight, 0.0)
    return activations


def test_expected_gaussian_radius_known_dimensions() -> None:
    assert math.isclose(
        radial.expected_gaussian_radius(1),
        math.sqrt(2.0 / math.pi),
        rel_tol=1e-12,
    )
    assert math.isclose(
        radial.expected_gaussian_radius(2),
        math.sqrt(math.pi / 2.0),
        rel_tol=1e-12,
    )


def test_sphere_normalization_is_finite_and_unit_norm() -> None:
    samples = radial.normal_lattice_samples(257, width=8, seed=19)
    sphere = np.asarray(radial.sphere_samples_from_gaussians(samples), dtype=np.float64)
    assert np.all(np.isfinite(sphere))
    np.testing.assert_allclose(np.linalg.norm(sphere, axis=1), 1.0, rtol=2e-6, atol=2e-6)


def test_relu_mlp_is_positive_homogeneous() -> None:
    rng = np.random.default_rng(13)
    weights = [
        rng.normal(size=(8, 8)).astype(np.float64) / math.sqrt(8.0)
        for _ in range(4)
    ]
    samples = rng.normal(size=(31, 8))
    scale = 3.7
    np.testing.assert_allclose(
        forward(scale * samples, weights),
        scale * forward(samples, weights),
        rtol=1e-12,
        atol=1e-12,
    )


def test_nonzero_bias_is_rejected() -> None:
    weight = np.eye(4, dtype=np.float32)
    mlp = make_mlp([weight], biases=[np.ones(4, dtype=np.float32)])
    with pytest.raises(ValueError, match="zero-bias"):
        radial.assert_positive_homogeneous_mlp(mlp)


def test_one_dimensional_relu_matches_analytic_mean() -> None:
    weight = np.ones((1, 1), dtype=np.float32)
    mlp = make_mlp([weight], seed=29)
    old_count = radial.N_SAMPLES
    radial.N_SAMPLES = 256
    try:
        estimate = float(np.asarray(radial.Estimator().predict(mlp, 1))[0, 0])
    finally:
        radial.N_SAMPLES = old_count
    expected = 1.0 / math.sqrt(2.0 * math.pi)
    assert math.isclose(estimate, expected, rel_tol=2e-6, abs_tol=2e-7)
