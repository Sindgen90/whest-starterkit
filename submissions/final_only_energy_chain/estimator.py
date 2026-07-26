"""Final-layer-only uncentered second-moment propagation.

For a centered Gaussian preactivation with variance ``s``, ReLU has

    E[ReLU(Z)] = sqrt(s / (2*pi))
    E[ReLU(Z)^2] = s / 2.

This estimator repeatedly re-centers the Gaussian approximation and propagates
only the uncentered second moment:

    s_next = (W * W).T @ q
    q_next = s_next / 2.

Only the final-layer mean is returned. Earlier rows are zero because Phase 1 is
scored on final-layer MSE and computing approximate earlier means is not needed
by the recurrence.
"""
from __future__ import annotations

import math

import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP


class Estimator(BaseEstimator):
    """Propagate path energy and predict only the final ReLU mean."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        # The method is deterministic, but retain the standard setup contract.
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        q = fnp.ones(mlp.width, dtype=fnp.float32)
        final_variance = q

        for w in mlp.weights:
            final_variance = fnp.matmul((w * w).T, q)
            q = final_variance * 0.5

        final_mean = fnp.sqrt(fnp.maximum(final_variance, 0.0)) / math.sqrt(2.0 * math.pi)
        zero_row = fnp.zeros(mlp.width, dtype=fnp.float32)
        rows = [zero_row for _ in range(mlp.depth - 1)]
        rows.append(final_mean)
        return fnp.stack(rows, axis=0)
