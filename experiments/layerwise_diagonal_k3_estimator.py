"""Layerwise diagonal K3 cohort propagation experiment.

This is a cheap middle ground between single-vector diagonal K3 and full
factorized K3.  Each source layer owns one diagonal third-cumulant "cube"
represented by a length-width vector.  Cohorts are propagated independently
through affine layers using ``W**3`` and through ReLU by the cubic local gain.

No generic off-diagonal tensor factors are stored.
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


def empty_cohorts(width: int) -> Any:
    return fnp.zeros((0, width))


class Estimator(BaseEstimator):
    def __init__(
        self,
        third_scale: float = -0.3,
        max_k3_layer: int | None = 8,
        generic_horizon: int | None = None,
        output_coefficient: float = K2_COEFFICIENT,
    ) -> None:
        if max_k3_layer is not None and max_k3_layer < 0:
            raise ValueError("max_k3_layer must be non-negative or None")
        if generic_horizon is not None and generic_horizon < 0:
            raise ValueError("generic_horizon must be non-negative or None")
        self.third_scale = third_scale
        self.max_k3_layer = max_k3_layer
        self.generic_horizon = generic_horizon
        self.output_coefficient = output_coefficient
        self._setup_rng = None
        self.cohort_history: list[dict[str, Any]] = []

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        width = mlp.width
        mu = fnp.zeros(width)
        cov = fnp.eye(width)
        cohorts = empty_cohorts(width)
        creation_layers: list[int] = []
        log_scale = 0.0
        rows = []
        self.cohort_history = []

        for layer_index, w in enumerate(mlp.weights):
            use_k3 = self.max_k3_layer is None or layer_index < self.max_k3_layer
            cov_diag = fnp.diag(cov)
            max_var_np = float(fnp.max(cov_diag))
            if max_var_np > _COV_RESCALE_THRESHOLD:
                scale = float(fnp.sqrt(max_var_np))
                mu = mu / scale
                cov = cov / (scale * scale)
                if creation_layers:
                    cohorts = cohorts / (scale ** 3)
                log_scale += float(fnp.log(scale))

            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)

            if use_k3 and creation_layers:
                kept_indices = [
                    idx
                    for idx, created in enumerate(creation_layers)
                    if self.generic_horizon is None or layer_index - created <= self.generic_horizon
                ]
                if len(kept_indices) != len(creation_layers):
                    cohorts = cohorts[kept_indices, :] if kept_indices else empty_cohorts(width)
                    creation_layers = [creation_layers[idx] for idx in kept_indices]
                pre_cohorts = cohorts @ (w * w * w)
                pre_diag_k3 = fnp.sum(pre_cohorts, axis=0)
            else:
                pre_cohorts = empty_cohorts(width)
                pre_diag_k3 = fnp.zeros(width)
                if not use_k3:
                    creation_layers = []

            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            mean_gauss, _second, var_post, exact_diag_k3, phi, Phi = relu_gaussian_moments(mu_pre, var_pre)
            alpha = mu_pre / fnp.sqrt(var_pre)
            f3 = -alpha * phi / var_pre
            mean_correction = self.third_scale * pre_diag_k3 * f3 / 6.0 if use_k3 else 0.0
            mu = mean_gauss + mean_correction

            gain = fnp.array(fnp.asarray(Phi, dtype=fnp.float64).astype(fnp.float32))
            cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
            fnp.fill_diagonal(cov, var_post)

            if use_k3:
                if creation_layers:
                    post_cohorts = pre_cohorts * (gain[None, :] * gain[None, :] * gain[None, :])
                    propagated_diag = fnp.sum(post_cohorts, axis=0)
                else:
                    post_cohorts = empty_cohorts(width)
                    propagated_diag = fnp.zeros(width)
                new_cohort = exact_diag_k3 - propagated_diag
                cohorts = fnp.concatenate([post_cohorts, new_cohort[None, :]], axis=0)
                creation_layers = [*creation_layers, layer_index]
            else:
                cohorts = empty_cohorts(width)

            scale_factor = float(fnp.exp(log_scale))
            rows.append(mu * scale_factor * self.output_coefficient)
            self.cohort_history.append(
                {
                    "layer": layer_index,
                    "cohort_count": len(creation_layers),
                    "creation_layers": list(creation_layers),
                    "used_k3": use_k3,
                }
            )

        return fnp.stack(rows, axis=0)
