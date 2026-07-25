"""Cheap diagonal-only K3 propagation experiment.

This estimator is deliberately *not* the full factorized K3 method.  It keeps
only the marginal third cumulant of each neuron, propagates those diagonal
cumulants through affine layers with ``W**3``, and uses the resulting
preactivation diagonal K3 as a first-order Edgeworth correction to the ReLU
mean.

The purpose is to test whether the useful early-K3 residual can be captured at
roughly K2 cost, without generic factor growth.
"""
from __future__ import annotations

from typing import Any

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

K2_COEFFICIENT = 0.9918858550287327
_COV_RESCALE_THRESHOLD = 1e100


def relu_gaussian_moments(mu: Any, var: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
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


class Estimator(BaseEstimator):
    def __init__(
        self,
        third_scale: float = -0.001,
        max_k3_layer: int | None = 8,
        output_coefficient: float = K2_COEFFICIENT,
    ) -> None:
        if max_k3_layer is not None and max_k3_layer < 0:
            raise ValueError("max_k3_layer must be non-negative or None")
        self.third_scale = third_scale
        self.max_k3_layer = max_k3_layer
        self.output_coefficient = output_coefficient
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        width = mlp.width
        mu = fnp.zeros(width)
        cov = fnp.eye(width)
        diag_k3 = fnp.zeros(width)
        log_scale = 0.0
        rows = []

        for layer_index, w in enumerate(mlp.weights):
            use_k3 = self.max_k3_layer is None or layer_index < self.max_k3_layer
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
            if use_k3:
                pre_diag_k3 = (w * w * w).T @ diag_k3
            else:
                pre_diag_k3 = fnp.zeros(width)

            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            mean_gauss, _second, var_post, exact_diag_k3, phi, Phi = relu_gaussian_moments(mu_pre, var_pre)
            alpha = mu_pre / fnp.sqrt(var_pre)
            f3 = -alpha * phi / var_pre
            mean_correction = self.third_scale * pre_diag_k3 * f3 / 6.0 if use_k3 else 0.0
            mu = mean_gauss + mean_correction

            gain = fnp.array(fnp.asarray(Phi, dtype=fnp.float64).astype(fnp.float32))
            cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
            fnp.fill_diagonal(cov, var_post)

            diag_k3 = exact_diag_k3 if use_k3 else fnp.zeros(width)

            scale_factor = float(fnp.exp(log_scale))
            rows.append(mu * scale_factor * self.output_coefficient)

        return fnp.stack(rows, axis=0)
