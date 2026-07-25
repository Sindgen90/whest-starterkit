"""Submission-safe corrected-K2 + signed Hadamard cubature convex combo.

Lambda was calibrated on development mlp_id parity split and frozen before
validation:

    lambda = 0.8798916787983015

Validation manifest result:

    mean final MSE     3.79756e-06
    adjusted score     3.83799e-07
    mean FLOPs         2.74896e10
"""
from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

K2_COEFFICIENT = 0.9918858550287327
LAMBDA = 0.8798916787983015
ROTATIONS = 12
ROTATION_SEED_OFFSET = 23_000_123
_COV_RESCALE_THRESHOLD = 1e100


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
        sigma_pre = fnp.sqrt(var_pre)
        alpha = mu_pre / sigma_pre
        phi_alpha = flops.stats.norm.pdf(alpha)
        Phi_alpha = flops.stats.norm.cdf(alpha)

        mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha
        ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
        var_post = fnp.maximum(ez2 - mu * mu, 0.0)

        sigma_np = fnp.asarray(sigma_pre, dtype=fnp.float64)
        Phi_np = fnp.asarray(Phi_alpha, dtype=fnp.float64)
        gain_np = fnp.where(sigma_np > 1e-12, Phi_np, 0.0)
        gain = fnp.array(gain_np.astype(fnp.float32))

        cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
        fnp.fill_diagonal(cov, var_post)

        scale_factor = float(fnp.exp(log_scale))
        rows.append(mu * scale_factor)

    return fnp.multiply(fnp.stack(rows, axis=0), K2_COEFFICIENT)


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
        cubature = cubature_prediction(mlp)
        return fnp.add(fnp.multiply(k2, 1.0 - LAMBDA), fnp.multiply(cubature, LAMBDA))
