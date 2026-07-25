"""Submission-safe cheap diagonal-K3 residual stack.

Prediction form frozen from development calibration:

    cubature + a * (corrected_k2 - cubature) + b * (diagonal_k3 - corrected_k2)

The diagonal-K3 component keeps only per-neuron marginal third cumulants.  It
does not carry generic K3 tensor factors.

Validation manifest result from research script:

    mean final MSE     3.34838e-06
    adjusted score     3.58333e-07
    mean FLOPs         2.91086e10
"""
from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

K2_COEFFICIENT = 0.9918858550287327
STACK_A = 0.10653087569429524
STACK_B = -2.6372834899095565
DIAG_K3_THIRD_SCALE = -0.3
DIAG_K3_MAX_LAYER = 8
ROTATIONS = 12
ROTATION_SEED_OFFSET = 23_000_123
_COV_RESCALE_THRESHOLD = 1e100


def relu_gaussian_moments(mu: fnp.ndarray, var: fnp.ndarray):
    sigma = fnp.sqrt(fnp.maximum(var, 1e-12))
    alpha = mu / sigma
    phi = flops.stats.norm.pdf(alpha)
    Phi = flops.stats.norm.cdf(alpha)
    mean = mu * Phi + sigma * phi
    second = (mu * mu + var) * Phi + mu * sigma * phi
    third = (mu * mu * mu + 3.0 * mu * var) * Phi + (mu * mu * sigma + 2.0 * sigma * var) * phi
    variance = fnp.maximum(second - mean * mean, 0.0)
    third_cumulant = third - 3.0 * mean * second + 2.0 * mean * mean * mean
    return mean, second, variance, third_cumulant, phi, Phi


def corrected_k2_prediction(mlp: MLP) -> fnp.ndarray:
    width = mlp.width
    mu = fnp.zeros(width)
    cov = fnp.eye(width)
    log_scale = 0.0
    rows = []

    for w in mlp.weights:
        cov_diag = fnp.diag(cov)
        max_var_np = float(fnp.max(cov_diag))
        if max_var_np > _COV_RESCALE_THRESHOLD:
            scale = float(fnp.sqrt(max_var_np))
            mu = mu / scale
            cov = cov / (scale * scale)
            log_scale += float(fnp.log(scale))

        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
        mean, _second, var_post, _diag_k3, _phi, Phi = relu_gaussian_moments(mu_pre, var_pre)
        mu = mean

        gain = fnp.array(fnp.asarray(Phi, dtype=fnp.float64).astype(fnp.float32))
        cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
        fnp.fill_diagonal(cov, var_post)

        scale_factor = float(fnp.exp(log_scale))
        rows.append(mu * scale_factor * K2_COEFFICIENT)

    return fnp.stack(rows, axis=0)


def diagonal_k3_prediction(mlp: MLP) -> fnp.ndarray:
    width = mlp.width
    mu = fnp.zeros(width)
    cov = fnp.eye(width)
    diag_k3 = fnp.zeros(width)
    log_scale = 0.0
    rows = []

    for layer_index, w in enumerate(mlp.weights):
        use_k3 = layer_index < DIAG_K3_MAX_LAYER
        cov_diag = fnp.diag(cov)
        max_var_np = float(fnp.max(cov_diag))
        if max_var_np > _COV_RESCALE_THRESHOLD:
            scale = float(fnp.sqrt(max_var_np))
            mu = mu / scale
            cov = cov / (scale * scale)
            diag_k3 = diag_k3 / (scale ** 3)
            log_scale += float(fnp.log(scale))

        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        pre_diag_k3 = (w * w * w).T @ diag_k3 if use_k3 else fnp.zeros(width)

        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
        mean_gauss, _second, var_post, exact_diag_k3, phi, Phi = relu_gaussian_moments(mu_pre, var_pre)
        alpha = mu_pre / fnp.sqrt(var_pre)
        f3 = -alpha * phi / var_pre
        mean_correction = DIAG_K3_THIRD_SCALE * pre_diag_k3 * f3 / 6.0 if use_k3 else 0.0
        mu = mean_gauss + mean_correction

        gain = fnp.array(fnp.asarray(Phi, dtype=fnp.float64).astype(fnp.float32))
        cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
        fnp.fill_diagonal(cov, var_post)
        diag_k3 = exact_diag_k3 if use_k3 else fnp.zeros(width)

        scale_factor = float(fnp.exp(log_scale))
        rows.append(mu * scale_factor * K2_COEFFICIENT)

    return fnp.stack(rows, axis=0)


def hadamard_matrix(width: int) -> fnp.ndarray:
    if width <= 0 or width & (width - 1):
        raise ValueError("signed Hadamard cubature requires power-of-two width")
    scale = 1.0 / math.sqrt(width)
    rows = []
    for row in range(width):
        values = []
        for col in range(width):
            parity = (row & col).bit_count() & 1
            values.append(-scale if parity else scale)
        rows.append(values)
    return fnp.array(rows, dtype=fnp.float32)


def signed_hadamard_cubature_points(width: int, seed: int) -> tuple[fnp.ndarray, fnp.ndarray]:
    radius = math.sqrt(width)
    base = hadamard_matrix(width)
    rng = fnp.random.default_rng(seed + ROTATION_SEED_OFFSET + 10_007 * ROTATIONS)
    blocks = []
    for _ in range(ROTATIONS):
        permutation = rng.permutation(width)
        signs = rng.choice(fnp.array([-1.0, 1.0], dtype=fnp.float32), size=width).astype(fnp.float32)
        basis = signs[:, None] * base[permutation, :]
        blocks.append(radius * basis.T)
        blocks.append(-radius * basis.T)
    points = fnp.concatenate(blocks, axis=0).astype(fnp.float32)
    weights = fnp.full((2 * width * ROTATIONS,), 1.0 / (2 * width * ROTATIONS), dtype=fnp.float32)
    return points, weights


def cubature_prediction(mlp: MLP) -> fnp.ndarray:
    x, weights = signed_hadamard_cubature_points(mlp.width, mlp.seed)
    rows = []
    for w in mlp.weights:
        x = fnp.maximum(fnp.matmul(x, w), 0.0)
        rows.append(fnp.sum(x * weights[:, None], axis=0))
    return fnp.stack(rows, axis=0)


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        k2 = corrected_k2_prediction(mlp)
        diag = diagonal_k3_prediction(mlp)
        cubature = cubature_prediction(mlp)
        return fnp.add(
            cubature,
            fnp.add(
                fnp.multiply(fnp.subtract(k2, cubature), STACK_A),
                fnp.multiply(fnp.subtract(diag, k2), STACK_B),
            ),
        )
