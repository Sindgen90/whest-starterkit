"""Correctness tests for centrally symmetrized Gaussian RQMC."""
from __future__ import annotations

import math
from types import SimpleNamespace

import flopscope.numpy as fnp
import numpy as np

from submissions.rqmc_antithetic_24576 import estimator as antithetic


def make_mlp(weights: list[np.ndarray], seed: int = 0) -> SimpleNamespace:
    width = weights[0].shape[0]
    return SimpleNamespace(
        width=width,
        depth=len(weights),
        weights=[fnp.array(weight, dtype=fnp.float32) for weight in weights],
        seed=seed,
        name=f"antithetic-{seed}",
    )


def test_antithetic_samples_are_exact_central_pairs() -> None:
    samples = np.asarray(
        antithetic.antithetic_normal_lattice_samples(31, width=8, seed=17),
        dtype=np.float64,
    )
    assert samples.shape == (62, 8)
    np.testing.assert_array_equal(samples[31:], -samples[:31])
    np.testing.assert_array_equal(samples.mean(axis=0), np.zeros(8))


def test_antithetic_samples_are_reproducible_and_randomized() -> None:
    first = np.asarray(antithetic.antithetic_normal_lattice_samples(31, 8, 17))
    repeated = np.asarray(antithetic.antithetic_normal_lattice_samples(31, 8, 17))
    independent = np.asarray(antithetic.antithetic_normal_lattice_samples(31, 8, 18))
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, independent)


def test_one_layer_relu_pair_identity() -> None:
    rng = np.random.default_rng(11)
    weight = rng.normal(size=(8, 8)).astype(np.float32)
    mlp = make_mlp([weight], seed=23)
    old_count = antithetic.N_BASE_SAMPLES
    antithetic.N_BASE_SAMPLES = 257
    try:
        prediction = np.asarray(antithetic.Estimator().predict(mlp, 1))[0]
        base = np.asarray(antithetic.normal_lattice_samples(257, 8, mlp.seed))
    finally:
        antithetic.N_BASE_SAMPLES = old_count
    projected = base @ weight
    expected = 0.5 * np.mean(np.abs(projected), axis=0)
    np.testing.assert_allclose(prediction, expected, rtol=2e-6, atol=2e-7)


def test_one_dimensional_relu_is_unbiased_across_random_shifts() -> None:
    weight = np.ones((1, 1), dtype=np.float32)
    old_count = antithetic.N_BASE_SAMPLES
    antithetic.N_BASE_SAMPLES = 127
    try:
        estimates = [
            float(np.asarray(antithetic.Estimator().predict(make_mlp([weight], seed), 1))[0, 0])
            for seed in range(32)
        ]
    finally:
        antithetic.N_BASE_SAMPLES = old_count
    expected = 1.0 / math.sqrt(2.0 * math.pi)
    assert abs(float(np.mean(estimates)) - expected) < 0.01
