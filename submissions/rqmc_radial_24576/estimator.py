"""Radial Rao-Blackwellized RQMC with 24,576 samples."""
from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

N_SAMPLES = 24_576
RQMC_SEED_OFFSET = 8_675_309
RQMC_MULTIPLIER = 17_797


def assert_positive_homogeneous_mlp(mlp: MLP) -> None:
    """Reject architectures for which analytic radial integration is invalid."""
    if not hasattr(mlp, "weights"):
        raise ValueError("unsupported MLP: missing weights")
    if hasattr(mlp, "biases") and getattr(mlp, "biases") is not None:
        for bias in getattr(mlp, "biases"):
            if float(fnp.max(fnp.abs(bias))) != 0.0:
                raise ValueError("radial RQMC requires zero-bias positive-homogeneous networks")
    if len(mlp.weights) != mlp.depth:
        raise ValueError("unsupported MLP: len(weights) != depth")
    for weight in mlp.weights:
        if len(weight.shape) != 2:
            raise ValueError("unsupported MLP: expected matrix weights")


def expected_gaussian_radius(dim: int) -> float:
    """Return E[||X||] for X distributed as a d-dimensional standard Gaussian."""
    return math.exp(
        0.5 * math.log(2.0)
        + math.lgamma((dim + 1.0) / 2.0)
        - math.lgamma(dim / 2.0)
    )


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


def sphere_samples_from_gaussians(samples: fnp.ndarray) -> fnp.ndarray:
    """Normalize nonzero Gaussian samples onto the unit sphere."""
    norms = fnp.sqrt(fnp.sum(samples * samples, axis=1))
    safe_norms = fnp.maximum(norms, 1e-30)
    return samples / safe_norms[:, None]


class Estimator(BaseEstimator):
    """Integrate the independent Gaussian radius exactly and sample directions."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        assert_positive_homogeneous_mlp(mlp)
        gaussian = normal_lattice_samples(N_SAMPLES, mlp.width, mlp.seed)
        x = sphere_samples_from_gaussians(gaussian)
        expected_radius = expected_gaussian_radius(mlp.width)
        rows = []
        for weight in mlp.weights:
            x = fnp.maximum(fnp.matmul(x, weight), 0.0)
            rows.append(fnp.mean(x, axis=0) * expected_radius)
        return fnp.stack(rows, axis=0)
