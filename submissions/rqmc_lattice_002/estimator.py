"""Submission-safe RQMC lattice estimator at the Phase 1 two-percent budget point."""
from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

N_SAMPLES = 1293
RQMC_SEED_OFFSET = 1_000_003


def normal_lattice_samples(n_samples: int, width: int, seed: int) -> fnp.ndarray:
    """Randomly shifted rank-1 lattice mapped through the normal inverse CDF."""
    rng = fnp.random.default_rng(seed + RQMC_SEED_OFFSET)
    powers = np.empty(width, dtype=np.float64)
    value = 1
    modulus = max(2, n_samples)
    for index in range(width):
        powers[index] = value
        value = (value * 17797) % modulus
        if value == 0:
            value = 1
    generator = fnp.array(powers)
    indices = fnp.arange(n_samples, dtype=fnp.float64) + 0.5
    shift = rng.uniform(0.0, 1.0, size=(width,))
    u = fnp.mod(fnp.outer(indices, generator) / n_samples + shift, 1.0)
    u = fnp.clip(u, 1e-7, 1.0 - 1e-7)
    return flops.stats.norm.ppf(u).astype(fnp.float32)


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        x = normal_lattice_samples(N_SAMPLES, mlp.width, mlp.seed)
        rows = []
        for w in mlp.weights:
            x = fnp.maximum(fnp.matmul(x, w), 0.0)
            rows.append(fnp.mean(x, axis=0))
        return fnp.stack(rows, axis=0)
