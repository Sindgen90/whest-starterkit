"""Factorized K3 prototype with source metadata.

This is an experiment module, not a submission candidate.  It keeps the K2
mean/covariance propagation path and adds a factorized third cumulant tensor
``T = F(A, B, C)`` used for a first-order Edgeworth correction to the ReLU mean.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

K2_COEFFICIENT = 0.9918858550287327
_COV_RESCALE_THRESHOLD = 1e100


@dataclass(frozen=True)
class FactorMeta:
    creation_layer: int
    source_type: str
    protected: bool


@dataclass
class Factors:
    a: Any
    b: Any
    c: Any
    meta: list[FactorMeta]

    @property
    def count(self) -> int:
        return len(self.meta)


def empty_factors(width: int) -> Factors:
    empty = fnp.zeros((width, 0))
    return Factors(empty, empty, empty, [])


def append_factors(base: Factors, a: Any, b: Any, c: Any, meta: list[FactorMeta]) -> Factors:
    if not meta:
        return base
    if base.count == 0:
        return Factors(a, b, c, meta)
    return Factors(
        fnp.concatenate([base.a, a], axis=1),
        fnp.concatenate([base.b, b], axis=1),
        fnp.concatenate([base.c, c], axis=1),
        [*base.meta, *meta],
    )


def factor_diag(factors: Factors) -> Any:
    if factors.count == 0:
        return fnp.zeros(factors.a.shape[0])
    return fnp.sum(factors.a * factors.b * factors.c, axis=1)


def factor_repeated(factors: Factors) -> Any:
    width = factors.a.shape[0]
    if factors.count == 0:
        return fnp.zeros((width, width))
    repeated = (
        (factors.b * factors.c) @ factors.a.T
        + (factors.a * factors.c) @ factors.b.T
        + (factors.a * factors.b) @ factors.c.T
    ) / 3.0
    fnp.fill_diagonal(repeated, 0.0)
    return repeated


def dense_from_factors_np(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Reconstruct the symmetric dense tensor for small-n tests only."""
    tensor = np.einsum("ij,kj,lj->ikl", a, b, c)
    tensor += np.einsum("ij,kj,lj->ilk", a, b, c)
    tensor += np.einsum("ij,kj,lj->kil", a, b, c)
    tensor += np.einsum("ij,kj,lj->kli", a, b, c)
    tensor += np.einsum("ij,kj,lj->lik", a, b, c)
    tensor += np.einsum("ij,kj,lj->lki", a, b, c)
    return tensor / 6.0


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


def relu_square_hermite(mu: Any, var: Any, mean: Any, Phi: Any) -> tuple[Any, Any]:
    _ = mu, var
    h1 = 2.0 * mean
    h2 = 2.0 * Phi
    return h1, h2


class Estimator(BaseEstimator):
    def __init__(self, third_scale: float = 1.0) -> None:
        self.third_scale = third_scale
        self._setup_rng = None
        self.factor_history: list[dict[str, Any]] = []

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        _rng = fnp.random.default_rng(mlp.seed)
        _ = _rng
        width = mlp.width
        mu = fnp.zeros(width)
        cov = fnp.eye(width)
        factors = empty_factors(width)
        log_scale = 0.0
        rows = []
        self.factor_history = []

        for layer_index, w in enumerate(mlp.weights):
            cov_diag = fnp.diag(cov)
            max_var_np = float(fnp.max(cov_diag))
            if max_var_np > _COV_RESCALE_THRESHOLD:
                scale = float(fnp.sqrt(max_var_np))
                mu = mu / scale
                cov = cov / (scale * scale)
                factors = Factors(factors.a / (scale ** 3), factors.b, factors.c, factors.meta)
                log_scale += float(fnp.log(scale))

            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
            if factors.count:
                pre_factors = Factors(w.T @ factors.a, w.T @ factors.b, w.T @ factors.c, factors.meta)
            else:
                pre_factors = factors

            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            mean_gauss, second, var_post, exact_diag_k3, phi, Phi = relu_gaussian_moments(
                mu_pre, var_pre
            )
            alpha = mu_pre / fnp.sqrt(var_pre)
            pre_diag_k3 = factor_diag(pre_factors)
            f3 = -alpha * phi / var_pre
            mean_correction = self.third_scale * pre_diag_k3 * f3 / 6.0
            mu = mean_gauss + mean_correction

            gain = fnp.array(fnp.asarray(Phi, dtype=fnp.float64).astype(fnp.float32))
            if pre_factors.count:
                factors = Factors(
                    gain[:, None] * pre_factors.a,
                    gain[:, None] * pre_factors.b,
                    gain[:, None] * pre_factors.c,
                    pre_factors.meta,
                )
            else:
                factors = pre_factors

            cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
            fnp.fill_diagonal(cov, var_post)

            factors = self._append_repeated_correction(
                layer_index, factors, cov_pre, mu_pre, var_pre, mean_gauss, second, Phi
            )
            factors = self._append_diagonal_correction(layer_index, factors, exact_diag_k3)

            scale_factor = float(fnp.exp(log_scale))
            rows.append(mu * scale_factor * K2_COEFFICIENT)
            self.factor_history.append(self._factor_summary(layer_index, factors))

        return fnp.stack(rows, axis=0)

    def _append_diagonal_correction(self, layer_index: int, factors: Factors, target_diag: Any) -> Factors:
        current_diag = factor_diag(factors)
        adjustment = target_diag - current_diag
        width = factors.a.shape[0]
        a = fnp.outer(adjustment, fnp.ones(width))
        eye = fnp.eye(width)
        meta = [FactorMeta(layer_index, "diagonal_correction", True) for _ in range(width)]
        return append_factors(factors, a, eye, eye, meta)

    def _append_repeated_correction(
        self,
        layer_index: int,
        factors: Factors,
        cov_pre: Any,
        mu_pre: Any,
        var_pre: Any,
        mean: Any,
        second: Any,
        Phi: Any,
    ) -> Factors:
        target = self._repeated_target(cov_pre, mu_pre, var_pre, mean, second, Phi)
        current = factor_repeated(factors)
        adjustment = target - current
        fnp.fill_diagonal(adjustment, 0.0)
        width = factors.a.shape[0]
        eye = fnp.eye(width)
        meta = [FactorMeta(layer_index, "repeated_correction", True) for _ in range(width)]
        return append_factors(factors, 3.0 * adjustment.T, eye, eye, meta)

    def _repeated_target(
        self,
        cov_pre: Any,
        mu_pre: Any,
        var_pre: Any,
        mean: Any,
        second: Any,
        Phi: Any,
    ) -> Any:
        sigma = fnp.sqrt(fnp.maximum(var_pre, 1e-12))
        f1 = Phi
        f2 = flops.stats.norm.pdf(mu_pre / sigma) / sigma
        square_h1, square_h2 = relu_square_hermite(mu_pre, var_pre, mean, Phi)
        e_a2_b = second[:, None] * mean[None, :]
        e_a2_b = e_a2_b + square_h1[:, None] * f1[None, :] * cov_pre
        e_a2_b = e_a2_b + 0.5 * square_h2[:, None] * f2[None, :] * cov_pre * cov_pre
        e_ab = mean[:, None] * mean[None, :] + f1[:, None] * f1[None, :] * cov_pre
        e_ab = e_ab + 0.5 * f2[:, None] * f2[None, :] * cov_pre * cov_pre
        target = e_a2_b - second[:, None] * mean[None, :]
        target = target - 2.0 * e_ab * mean[:, None]
        target = target + 2.0 * mean[:, None] * mean[:, None] * mean[None, :]
        fnp.fill_diagonal(target, 0.0)
        return target

    def _factor_summary(self, layer_index: int, factors: Factors) -> dict[str, Any]:
        source_counts: dict[str, int] = {}
        protected = 0
        for meta in factors.meta:
            source_counts[meta.source_type] = source_counts.get(meta.source_type, 0) + 1
            protected += int(meta.protected)
        return {
            "layer": layer_index,
            "factor_count": factors.count,
            "protected_count": protected,
            "generic_count": factors.count - protected,
            "source_counts": source_counts,
        }
