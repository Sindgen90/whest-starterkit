"""Cross-fitted degree-2 Hermite/Stein control variates in a fixed basis."""
from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

FOLD_SAMPLES = 646
RANK = 4
SHRINKAGE = 0.5
RQMC_SEED_OFFSET_A = 1_000_003
RQMC_SEED_OFFSET_B = 11_000_022
RQMC_MULTIPLIER = 17_797
RANDOM_HADAMARD_COLUMNS = (11, 23, 37, 58, 91, 117, 149, 203, 221, 7, 171, 45, 199, 83, 139, 245)


def normal_lattice_samples(n_samples: int, width: int, seed: int, seed_offset: int) -> fnp.ndarray:
    rng = fnp.random.default_rng(seed + seed_offset)
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
    u = fnp.mod(fnp.outer(indices, generator) / n_samples + shift, 1.0)
    u = fnp.clip(u, 1e-7, 1.0 - 1e-7)
    return flops.stats.norm.ppf(u).astype(fnp.float32)


def fixed_hadamard_basis(width: int, rank: int) -> fnp.ndarray:
    if width <= 0 or width & (width - 1):
        raise ValueError("fixed Hermite basis requires power-of-two input width")
    if rank > len(RANDOM_HADAMARD_COLUMNS):
        raise ValueError("rank exceeds packaged Hadamard columns")
    scale = 1.0 / math.sqrt(width)
    rows = []
    for row in range(width):
        values = []
        for col in RANDOM_HADAMARD_COLUMNS[:rank]:
            parity = (row & col).bit_count() & 1
            values.append(-scale if parity else scale)
        rows.append(values)
    return fnp.array(rows, dtype=fnp.float32)


def hermite_degree2_features(x: fnp.ndarray, q: fnp.ndarray) -> fnp.ndarray:
    z = fnp.matmul(x, q)
    rank = int(z.shape[1])
    features = []
    for i in range(rank):
        features.append(z[:, i])
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for i in range(rank):
        features.append((z[:, i] * z[:, i] - 1.0) * inv_sqrt2)
    for i in range(rank):
        for j in range(i + 1, rank):
            features.append(z[:, i] * z[:, j])
    return fnp.stack(features, axis=1)


def projection_coefficients(phi: fnp.ndarray, y: fnp.ndarray) -> fnp.ndarray:
    centered = y - fnp.mean(y, axis=0)
    return (SHRINKAGE / float(phi.shape[0])) * fnp.matmul(fnp.swapaxes(phi, 0, 1), centered)


def crossfit_projection_mean(
    y_a: fnp.ndarray,
    phi_a: fnp.ndarray,
    y_b: fnp.ndarray,
    phi_b: fnp.ndarray,
) -> fnp.ndarray:
    beta_a = projection_coefficients(phi_a, y_a)
    beta_b = projection_coefficients(phi_b, y_b)
    mu_b_given_a = fnp.mean(y_b, axis=0) - fnp.matmul(fnp.mean(phi_b, axis=0), beta_a)
    mu_a_given_b = fnp.mean(y_a, axis=0) - fnp.matmul(fnp.mean(phi_a, axis=0), beta_b)
    n_a = float(y_a.shape[0])
    n_b = float(y_b.shape[0])
    return (n_b * mu_b_given_a + n_a * mu_a_given_b) / (n_a + n_b)


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        q = fixed_hadamard_basis(mlp.width, RANK)
        x_a = normal_lattice_samples(FOLD_SAMPLES, mlp.width, mlp.seed, RQMC_SEED_OFFSET_A)
        x_b = normal_lattice_samples(FOLD_SAMPLES, mlp.width, mlp.seed, RQMC_SEED_OFFSET_B)
        phi_a = hermite_degree2_features(x_a, q)
        phi_b = hermite_degree2_features(x_b, q)
        rows = []
        y_a = x_a
        y_b = x_b
        for w in mlp.weights:
            y_a = fnp.maximum(fnp.matmul(y_a, w), 0.0)
            y_b = fnp.maximum(fnp.matmul(y_b, w), 0.0)
            rows.append(crossfit_projection_mean(y_a, phi_a, y_b, phi_b))
        return fnp.stack(rows, axis=0)
