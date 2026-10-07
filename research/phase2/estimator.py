"""Phase-two public-method replication, written from published equations.

Default: baghyeonsig's 0.75 covariance / 0.25 whitened antithetic MC blend
(AIcrowd topic 18200, submission 328274), with the disclosed 4038 samples.
Also exposes an experimental inter-layer discrepancy transport inspired by
Team Puffi (topic 18175). This is NOT their A716/Kerdock/ridge estimator.
All evaluation-time numerical work uses flopscope. No fitted phase-one constants.
See README.md for provenance, differences, and benchmark limitations.
"""
from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


def covariance_path(mlp):
    """Gain closure with exact Gaussian marginal moments and a symmetric state."""
    mu = fnp.zeros(mlp.width, dtype=fnp.float32)
    cov = flops.as_symmetric(fnp.eye(mlp.width, dtype=fnp.float32), symmetry=(0, 1))
    means, gains = [], []
    for weight in mlp.weights:
        w = fnp.asarray(weight, dtype=fnp.float32)
        pre = mu @ w
        pre_cov = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var = fnp.maximum(fnp.diag(pre_cov), 1e-12)
        sigma = fnp.sqrt(var)
        alpha = pre / sigma
        gain = flops.stats.norm.cdf(alpha).astype(fnp.float32)
        density = flops.stats.norm.pdf(alpha).astype(fnp.float32)
        mu = pre * gain + sigma * density
        second = (pre * pre + var) * gain + pre * sigma * density
        cov = fnp.outer(gain, gain) * pre_cov
        fnp.fill_diagonal(cov, fnp.maximum(second - mu * mu, 0.0))
        cov = flops.as_symmetric(cov, symmetry=(0, 1))
        means.append(mu)
        gains.append(gain)
    return fnp.stack(means), gains


def whitened_path(mlp, samples, seed):
    """Antithetic input moment matching; symmetric inverse root fused into W0.

    Whitening changes finite-sample marginals; this is not an unbiased Gaussian
    sampler. In particular it must be evaluated for bias on unseen networks.
    """
    rng = fnp.random.default_rng(seed)
    half = samples // 2
    xh = rng.standard_normal((half, mlp.width)).astype(fnp.float32)
    x = fnp.concatenate((xh, -xh), axis=0)
    gram = (x.T @ x) / samples
    eigenvalues, eigenvectors = fnp.linalg.eigh(gram)
    inv_root = (eigenvectors * fnp.power(fnp.maximum(eigenvalues, 1e-6), -0.5)) @ eigenvectors.T
    fused = inv_root @ fnp.asarray(mlp.weights[0], dtype=fnp.float32)
    y = fnp.maximum(x @ fused, 0.0)
    means = [fnp.mean(y, axis=0)]
    for weight in mlp.weights[1:]:
        y = fnp.maximum(y @ fnp.asarray(weight, dtype=fnp.float32), 0.0)
        means.append(fnp.mean(y, axis=0))
    return fnp.stack(means)


class Estimator(BaseEstimator):
    # Frozen before opening target values. Overrides are for research ablations.
    mode = "blend"
    samples = 4038
    covariance_weight = 0.75
    anchor_layer = 4  # one-based post-ReLU layer
    control_strength = 0.75
    fixed_seed = None  # None uses the grader-supplied estimator RNG seed.

    def predict(self, mlp, budget):
        if self.mode == "sampling":
            # Do not charge the sampler for an unused analytical branch.
            fixed = 14 * mlp.width ** 3
            per_sample = 2 * (mlp.depth + 1) * mlp.width ** 2 + 40 * mlp.width
            affordable = max(0, int((0.85 * budget - fixed) / per_sample))
            count = min(self.samples, affordable) // 2 * 2
            if count // 2 <= mlp.width:
                return covariance_path(mlp)[0]
            return whitened_path(mlp, count, mlp.seed if self.fixed_seed is None else self.fixed_seed)
        analytic, gains = covariance_path(mlp)
        if self.mode == "covariance":
            return analytic
        # Geometry/budget bookkeeping only. Preserve full-rank whitening.
        fixed_upper = (4 * mlp.depth + 14) * mlp.width ** 3
        per_sample_upper = 2 * (mlp.depth + 1) * mlp.width ** 2 + 40 * mlp.width
        affordable = max(0, int((0.85 * budget - fixed_upper) / per_sample_upper))
        count = min(self.samples, affordable) // 2 * 2
        if count // 2 <= mlp.width:
            return analytic
        sampled = whitened_path(mlp, count, mlp.seed if self.fixed_seed is None else self.fixed_seed)
        if self.mode == "blend":
            return self.covariance_weight * analytic + (1.0 - self.covariance_weight) * sampled
        if self.mode != "transport":
            raise ValueError("Unknown estimator mode")
        # Linear response approximation to the propagation of early mean error.
        # The anchor is approximate except at layer 1; no unbiasedness claim.
        anchor = min(self.anchor_layer, mlp.depth) - 1
        delta = sampled[anchor] - analytic[anchor]
        rows = [sampled[layer] for layer in range(anchor)]
        rows.append(sampled[anchor] - self.control_strength * delta)
        for layer in range(anchor + 1, mlp.depth):
            delta = (delta @ mlp.weights[layer]) * gains[layer]
            rows.append(sampled[layer] - self.control_strength * delta)
        return fnp.stack(rows)
