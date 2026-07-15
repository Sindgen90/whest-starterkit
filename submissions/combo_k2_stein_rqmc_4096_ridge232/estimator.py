"""Experimental corrected-K2/RQMC combo with a final-layer Stein CV.

Constants frozen from results/mini_stein_ridge232_ridge1_final_rqmc4096_offset8675309_mult17797.json:
even mlp_id calibration, odd mlp_id holdout, Mini split.
"""
from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

K2_COEFFICIENT = 0.9918858550287327
LAMBDA = 0.8065181275829477
N_SAMPLES = 4096
RQMC_SEED_OFFSET = 8_675_309
RQMC_MULTIPLIER = 17_797
STEIN_FEATURE_COUNT = 32
STEIN_RIDGE = 1.0
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


def ridge2_stein_features(samples: fnp.ndarray, mlp: MLP) -> fnp.ndarray:
    count = min(STEIN_FEATURE_COUNT, mlp.width)
    basis = mlp.weights[0][:, :count].astype(fnp.float64)
    z = fnp.matmul(samples.astype(fnp.float64), basis)
    norm2 = fnp.sum(basis * basis, axis=0)
    return fnp.concatenate([z, z * z - norm2], axis=1)


def fit_beta(features: fnp.ndarray, values: fnp.ndarray) -> fnp.ndarray:
    features = features.astype(fnp.float64)
    values = values.astype(fnp.float64)
    features = features - fnp.mean(features, axis=0)
    values = values - fnp.mean(values, axis=0)
    gram = fnp.matmul(features.T, features)
    scale = float(fnp.trace(gram)) / max(1, features.shape[1])
    penalty = max(1e-12, STEIN_RIDGE * scale)
    gram = gram + fnp.eye(features.shape[1]) * penalty
    rhs = fnp.matmul(features.T, values)
    return fnp.linalg.solve(gram, rhs)


def crossfit_cv_mean(features: fnp.ndarray, values: fnp.ndarray) -> fnp.ndarray:
    n_samples, n_features = features.shape
    split = n_samples // 2
    if split <= n_features:
        return fnp.mean(values, axis=0)

    g0 = features[:split].astype(fnp.float64)
    g1 = features[split:].astype(fnp.float64)
    y0 = values[:split].astype(fnp.float64)
    y1 = values[split:].astype(fnp.float64)

    beta0 = fit_beta(g0, y0)
    beta1 = fit_beta(g1, y1)

    estimate1 = fnp.mean(y1, axis=0) - fnp.matmul(fnp.mean(g1, axis=0), beta0)
    estimate0 = fnp.mean(y0, axis=0) - fnp.matmul(fnp.mean(g0, axis=0), beta1)
    return ((estimate0 + estimate1) * 0.5).astype(fnp.float32)


def stein_rqmc_prediction(mlp: MLP) -> fnp.ndarray:
    x = normal_lattice_samples(N_SAMPLES, mlp.width, mlp.seed)
    features = ridge2_stein_features(x, mlp)
    rows = []
    for layer_index, w in enumerate(mlp.weights):
        x = fnp.maximum(fnp.matmul(x, w), 0.0)
        if layer_index == mlp.depth - 1:
            rows.append(crossfit_cv_mean(features, x))
        else:
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
        stein = stein_rqmc_prediction(mlp)
        return fnp.add(fnp.multiply(k2, 1.0 - LAMBDA), fnp.multiply(stein, LAMBDA))
