"""Submission-safe centrally symmetrized RQMC with 24,576 evaluations."""
from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

N_BASE_SAMPLES = 12_288
RQMC_SEED_OFFSET = 8_675_309
RQMC_MULTIPLIER = 17_797


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


def antithetic_normal_lattice_samples(
    n_base_samples: int,
    width: int,
    seed: int,
) -> fnp.ndarray:
    """Return a Gaussian lattice together with its exact central reflection."""
    base = normal_lattice_samples(n_base_samples, width, seed)
    return fnp.concatenate((base, -base), axis=0)


class Estimator(BaseEstimator):
    """Propagate a centrally symmetric Gaussian RQMC design through the MLP."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        x = antithetic_normal_lattice_samples(N_BASE_SAMPLES, mlp.width, mlp.seed)
        rows = []
        for w in mlp.weights:
            x = fnp.maximum(fnp.matmul(x, w), 0.0)
            rows.append(fnp.mean(x, axis=0))
        return fnp.stack(rows, axis=0)
