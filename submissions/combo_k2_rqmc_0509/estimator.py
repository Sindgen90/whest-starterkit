"""Submission-safe convex combination of corrected K2 and RQMC lattice."""
from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

K2_COEFFICIENT = 0.9918858550287327
LAMBDA = 0.5093843892112996
N_SAMPLES = 1293
RQMC_SEED_OFFSET = 1_000_003
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


def normal_lattice_samples(n_samples: int, width: int, seed: int) -> fnp.ndarray:
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


def rqmc_prediction(mlp: MLP) -> fnp.ndarray:
    x = normal_lattice_samples(N_SAMPLES, mlp.width, mlp.seed)
    rows = []
    for w in mlp.weights:
        x = fnp.maximum(fnp.matmul(x, w), 0.0)
        rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0)


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        k2 = corrected_k2_prediction(mlp)
        rqmc = rqmc_prediction(mlp)
        return fnp.add(fnp.multiply(k2, 1.0 - LAMBDA), fnp.multiply(rqmc, LAMBDA))
