from __future__ import annotations

import math
from dataclasses import dataclass

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


@dataclass(frozen=True)
class LatticeConfig:
    n_samples: int = 1293
    seed_offset: int = 1_000_003
    multiplier: int = 17_797


def lattice_uniforms(config: LatticeConfig, width: int, seed: int) -> fnp.ndarray:
    rng = fnp.random.default_rng(seed + config.seed_offset)
    powers = []
    value = 1
    modulus = max(2, config.n_samples)
    for _ in range(width):
        powers.append(float(value))
        value = (value * config.multiplier) % modulus
        if value == 0:
            value = 1
    generator = fnp.array(powers, dtype=fnp.float64)
    indices = fnp.arange(config.n_samples, dtype=fnp.float64) + 0.5
    shift = rng.uniform(0.0, 1.0, size=(width,))
    u = fnp.mod(fnp.outer(indices, generator) / config.n_samples + shift, 1.0)
    return fnp.clip(u, 1e-7, 1.0 - 1e-7)


def lattice_gaussians(config: LatticeConfig, width: int, seed: int) -> fnp.ndarray:
    return flops.stats.norm.ppf(lattice_uniforms(config, width, seed)).astype(fnp.float32)


def independent_lattice_gaussian_folds(
    n_per_fold: int,
    width: int,
    seed: int,
    seed_offset: int,
    multiplier: int,
) -> tuple[fnp.ndarray, fnp.ndarray]:
    fold_a = lattice_gaussians(LatticeConfig(n_per_fold, seed_offset, multiplier), width, seed)
    fold_b = lattice_gaussians(LatticeConfig(n_per_fold, seed_offset + 10_000_019, multiplier), width, seed)
    return fold_a, fold_b


def radial_expected_norm(dim: int) -> float:
    return math.exp(0.5 * math.log(2.0) + math.lgamma((dim + 1.0) / 2.0) - math.lgamma(dim / 2.0))


def normalize_to_sphere(samples: fnp.ndarray, eps: float = 1e-30) -> fnp.ndarray:
    norm = fnp.sqrt(fnp.sum(samples * samples, axis=1))
    safe_norm = fnp.maximum(norm, eps)
    return samples / safe_norm[:, None]


def numpy_lattice_gaussians(config: LatticeConfig, width: int, seed: int) -> np.ndarray:
    """NumPy reference implementation for statistical tests."""
    rng = np.random.default_rng(seed + config.seed_offset)
    powers = np.empty(width, dtype=np.float64)
    value = 1
    modulus = max(2, config.n_samples)
    for index in range(width):
        powers[index] = float(value)
        value = (value * config.multiplier) % modulus
        if value == 0:
            value = 1
    indices = np.arange(config.n_samples, dtype=np.float64) + 0.5
    shift = rng.uniform(0.0, 1.0, size=width)
    u = np.mod(np.outer(indices, powers) / config.n_samples + shift, 1.0)
    u = np.clip(u, 1e-7, 1.0 - 1e-7)
    return np.asarray(flops.stats.norm.ppf(fnp.array(u, dtype=fnp.float64)), dtype=np.float64)
