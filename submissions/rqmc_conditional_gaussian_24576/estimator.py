"""RQMC with a conditional-Gaussian correction at the final layer.

Samples are propagated normally through layers 1..L-1. At the final layer,
the estimator computes the empirical mean and variance of each preactivation
and replaces ``mean(ReLU(preactivation))`` with the corresponding Gaussian
ReLU expectation.
"""
from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

N_SAMPLES = 24_576
RQMC_SEED_OFFSET = 8_675_309
RQMC_MULTIPLIER = 17_797
VARIANCE_FLOOR = 1e-12


def normal_lattice_samples(n_samples: int, width: int, seed: int) -> fnp.ndarray:
    """Generate one randomly shifted rank-1 lattice in Gaussian coordinates."""
    rng = fnp.random.default_rng(seed + RQMC_SEED_OFFSET)
    powers = []
    value = 1
    modulus = max(2, n_samples)
    for _ in range(width):
        powers.append(float(value))
        value = (value * RQMC_MULTIPLIER) % modulus
        if value == 0:
            value = 1
    generator = fnp.array(powers, dtype=fnp.float64)
    indices = fnp.arange(n_samples, dtype=fnp.float64) + 0.5
    shift = rng.uniform(0.0, 1.0, size=(width,))
    uniforms = fnp.mod(fnp.outer(indices, generator) / n_samples + shift, 1.0)
    uniforms = fnp.clip(uniforms, 1e-7, 1.0 - 1e-7)
    return flops.stats.norm.ppf(uniforms).astype(fnp.float32)


def gaussian_relu_mean(mean: fnp.ndarray, variance: fnp.ndarray) -> fnp.ndarray:
    """Return E[ReLU(Z)] for coordinatewise Gaussian Z."""
    safe_variance = fnp.maximum(variance, VARIANCE_FLOOR)
    sigma = fnp.sqrt(safe_variance)
    alpha = mean / sigma
    estimate = mean * flops.stats.norm.cdf(alpha) + sigma * flops.stats.norm.pdf(alpha)
    return fnp.where(variance > VARIANCE_FLOOR, estimate, fnp.maximum(mean, 0.0))


class Estimator(BaseEstimator):
    """Ordinary RQMC through L-1, Gaussian moment closure at layer L."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        x = normal_lattice_samples(N_SAMPLES, mlp.width, mlp.seed)
        rows = []

        for w in mlp.weights[:-1]:
            x = fnp.maximum(fnp.matmul(x, w), 0.0)
            rows.append(fnp.mean(x, axis=0))

        final_preactivation = fnp.matmul(x, mlp.weights[-1])
        preactivation_mean = fnp.mean(final_preactivation, axis=0)
        centered = final_preactivation - preactivation_mean
        preactivation_variance = fnp.mean(centered * centered, axis=0)
        rows.append(gaussian_relu_mean(preactivation_mean, preactivation_variance))
        return fnp.stack(rows, axis=0)
