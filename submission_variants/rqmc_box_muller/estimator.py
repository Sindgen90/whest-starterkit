"""RQMC estimator using a paired Box--Muller Gaussian transformation."""
from __future__ import annotations

import math

import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

N_SAMPLES = 4_096
RQMC_SEED_OFFSET = 8_675_309
RQMC_MULTIPLIER = 17_797
UNIFORM_EPSILON = 1e-7


def lattice_uniforms(n_samples: int, width: int, seed: int) -> fnp.ndarray:
    """Generate the same randomly shifted rank-1 lattice as the RQMC baseline."""
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
    uniforms = fnp.mod(
        fnp.outer(indices, generator) / n_samples + shift,
        1.0,
    )
    return fnp.clip(uniforms, UNIFORM_EPSILON, 1.0 - UNIFORM_EPSILON)


def box_muller_transform(uniforms: fnp.ndarray) -> fnp.ndarray:
    """Map paired uniform coordinates to independent standard Gaussians."""
    if uniforms.ndim != 2:
        raise ValueError("uniforms must have shape (samples, width)")
    n_samples, width = uniforms.shape
    if width % 2:
        raise ValueError("Box--Muller requires an even input width")
    radial_uniforms = fnp.clip(
        uniforms[:, 0::2],
        UNIFORM_EPSILON,
        1.0 - UNIFORM_EPSILON,
    )
    angular_uniforms = uniforms[:, 1::2]
    radii = fnp.sqrt(-2.0 * fnp.log(radial_uniforms))
    angles = (2.0 * math.pi) * angular_uniforms
    paired = fnp.stack(
        (
            radii * fnp.cos(angles),
            radii * fnp.sin(angles),
        ),
        axis=2,
    )
    return paired.reshape((n_samples, width))


def normal_lattice_samples(n_samples: int, width: int, seed: int) -> fnp.ndarray:
    """Generate Box--Muller Gaussian samples from the shared lattice."""
    uniforms = lattice_uniforms(n_samples, width, seed)
    return box_muller_transform(uniforms).astype(fnp.float32)


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        x = normal_lattice_samples(N_SAMPLES, mlp.width, mlp.seed)
        rows = []
        for weight in mlp.weights:
            x = fnp.maximum(fnp.matmul(x, weight), 0.0)
            rows.append(fnp.mean(x, axis=0))
        return fnp.stack(rows, axis=0)
