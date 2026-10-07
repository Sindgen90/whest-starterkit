"""Runtime-optimized derivative of venture_agent_ai submission 329951.

Frozen original calibration and eight quadrature nodes. Factor rho^2 out of
the quadrature sum and omit the unused covariance update after the last output.
See PROVENANCE.md for upstream attribution and validation.

Covariance propagation with an EXACT bivariate-Gaussian off-diagonal ReLU
covariance (replacing the starter kit's "gain-product" approximation), plus
a per-layer empirical calibration refit on top of it.

See ../RESULTS.md for the full derivation, numerical verification, FLOP
model, and before/after measurements. Summary of what changed relative to
examples/03_covariance_propagation.py:

Starter kit's Step 7 (gain-product approximation):
    cov_post[i,j] ~= gain[i] * gain[j] * cov_pre[i,j]
    where gain[i] = Phi(alpha[i]), alpha[i] = mu_pre[i]/sigma_pre[i].

This file's Step 7 (EXACT, for a bivariate Gaussian (u,v) with means
mu_pre[i], mu_pre[j], std sigma_pre[i], sigma_pre[j], correlation
rho[i,j] = cov_pre[i,j] / (sigma_pre[i]*sigma_pre[j])):

    E[relu(u) relu(v)] = mu_relu[i]*mu_relu[j]
        + sigma_pre[i]*sigma_pre[j] * ( rho*Phi(a)*Phi(b)
              + integral_0^rho (rho - t) * phi2(a, b; t) dt )

    where a=alpha[i], b=alpha[j], phi2(a,b;t) is the standard bivariate
    normal density at (a,b) with correlation t. Subtracting mu_relu[i]*mu_relu[j]
    (mu_relu is the exact per-neuron ReLU mean already computed at Step 5)
    gives cov_post[i,j] directly, and the first term inside the brackets,
    rho*Phi(a)*Phi(b), multiplied by sigma_pre[i]*sigma_pre[j], is *exactly*
    cov_pre[i,j]*Phi(a)*Phi(b) -- i.e. the starter kit's own gain-product
    term. So:

        cov_post[i,j] = gain_product_term[i,j] + correction[i,j]

    correction[i,j] = sigma_pre[i]*sigma_pre[j]
                       * integral_0^{rho[i,j]} (rho[i,j]-t) * phi2(a,b;t) dt

This is an additive, exact correction on top of the existing approximation,
not a replacement algorithm -- see RESULTS.md Section 1 for the derivation
(Price's theorem + Plackett's identity) and Section 2 for verification
against 20M-sample Monte Carlo and scipy's independent bivariate-normal CDF
implementation (max abs error ~1.7e-16 on the CDF check; every one of 720
(mean, std, rho) combinations landed within 4 Monte Carlo standard errors).

The remaining 1-D integral over t in [0, rho[i,j]] is evaluated by an
N_QUAD-node Gauss-Legendre quadrature (nodes/weights precomputed offline on
[0,1] and hardcoded as constants below -- no scipy/numpy.polynomial at
inference time, only flopscope.numpy + stdlib, per docs/concepts/allowed-code.md).
phi2's exponential form needs only exp/sqrt/arithmetic (flopscope.stats has
no bivariate normal CDF, so this quadrature-of-the-density approach avoids
needing one). RESULTS.md Section 3 shows the quadrature has converged (no
visible accuracy change) by N_QUAD=6-8; N_QUAD=8 is shipped for margin. Total
added cost (RESULTS.md Section 4): well under 1% of the 2**41 budget on top
of covariance_propagation's existing 2.35%.

Everything else (Steps 1-6, 8, the calibration hook, the overflow guard) is
unchanged from ../lens_higher_order's calibrated covariance_propagation.
"""

from __future__ import annotations

from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_COV_RESCALE_THRESHOLD = 1e30
CALIBRATION_FILE = "calibration_weights.npz"
CALIBRATION_SCALE = 1.0

# 8-node Gauss-Legendre nodes/weights on [0, 1], precomputed offline via
# numpy.polynomial.legendre.leggauss(8) and hardcoded here (no scipy/numpy at
# inference time). Rescaled per-pair at runtime to [0, rho[i,j]] via
# node * rho, weight * rho (see the loop in predict()).
_QUAD_NODES01 = (
    0.019855071751231912,
    0.10166676129318664,
    0.2372337950418355,
    0.4082826787521751,
    0.591717321247825,
    0.7627662049581645,
    0.8983332387068134,
    0.9801449282487681,
)
_QUAD_WEIGHTS01 = (
    0.050614268145188185,
    0.11119051722668709,
    0.15685332293894372,
    0.18134189168918102,
    0.18134189168918102,
    0.15685332293894372,
    0.11119051722668709,
    0.050614268145188185,
)


# Per-layer calibration (16 layers x 5 coefficients), embedded so the estimator
# ships as a single file; identical to calibration_weights.npz.
_CALIB = (
    (-5.8901288866763934e-05, 0.00014663842739537358, 0.0, 8.958482067100704e-05, -0.0002657242876011878),
    (-0.0007936688489280641, 0.00014058445231057703, 0.00043832763913087547, 0.00012623323709703982, -7.115971675375476e-05),
    (-0.0005232683615759015, -0.00028669173480011523, 5.421238165581599e-05, -3.299415539004258e-06, 7.834398275008425e-05),
    (-0.0005290278932079673, -4.6808803745079786e-05, -7.390147948171943e-05, -8.756271563470364e-05, -0.00030575718847103417),
    (-0.000711230852175504, -0.0001018924594973214, -7.03133555362001e-05, 0.00010085706162499264, -0.0002604360633995384),
    (-0.0009258362697437406, 0.00025708350585773587, -4.915930185234174e-05, 7.617778464918956e-05, -0.00039322354132309556),
    (-0.0009707512217573822, -0.00016728529590182006, -5.382432937039994e-05, 0.0004773754917550832, -0.00020201831648591906),
    (-0.001045495504513383, 0.00013469025725498796, -5.690031321137212e-05, 0.00022824171173851937, -0.0003133636782877147),
    (-0.0011691047111526132, -0.0001343699696008116, -4.768569124280475e-05, 0.00021642399951815605, -0.00020413448510225862),
    (-0.00114838732406497, -0.0005145969917066395, -6.767985905753449e-05, 0.00017472950275987387, -0.00014417343481909484),
    (-0.0012639987980946898, 0.0003387455944903195, -4.9771952035371214e-05, 0.0004985918058082461, -0.00032444196403957903),
    (-0.0013581309467554092, 0.0005392395542003214, -4.031981006846763e-05, 0.0006424990715458989, -0.0003384149167686701),
    (-0.0014375358587130904, 0.0007309210486710072, -3.3749187423381954e-05, 0.0004579271480906755, -0.00032388593535870314),
    (-0.0014232001267373562, 0.0007004060898907483, -4.170860847807489e-05, 0.0006092237890698016, -0.0003455811529420316),
    (-0.0014793600421398878, 0.001080659800209105, -3.5051198210567236e-05, 0.0004620171384885907, -0.0003657388442661613),
    (-0.0015159399481490254, 0.0012455069227144122, -3.2144707802217454e-05, 0.0005487297894433141, -0.00037651564343832433),
)

class CalibrationWeights(flops.Module):
    def __init__(self) -> None:
        self.calibration = fnp.zeros((16, 5), dtype=fnp.float32)


class Estimator(BaseEstimator):
    """Covariance propagation with exact off-diagonal ReLU covariance, plus
    a small fitted per-layer calibration on top."""

    def __init__(self) -> None:
        self._setup_rng = None
        self._calibration = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)
        self._calibration = None
        if ctx.submission_dir is not None:
            path = Path(ctx.submission_dir) / CALIBRATION_FILE
            if path.exists():
                self._calibration = CalibrationWeights.from_file(str(path)).calibration
        if self._calibration is None:
            # Single-file submission: build the calibration from the embedded literal.
            # Plain Python floats: no array construction from Python objects on the
            # grader's flopscope server; scalar-times-array ops are standard.
            self._calibration = _CALIB

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        width = mlp.width

        mu = fnp.zeros(width, dtype=fnp.float32)
        cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
        log_scale = 0.0

        rows = []
        for li, w in enumerate(mlp.weights):
            cov_diag = fnp.diag(cov)
            max_var_np = float(fnp.max(cov_diag))
            if max_var_np > _COV_RESCALE_THRESHOLD:
                s = float(fnp.sqrt(max_var_np))
                mu = mu / s
                cov = cov / (s * s)
                log_scale += float(fnp.log(s))

            # --- Step 3: linear layer (exact) ---
            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)

            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            sigma_pre = fnp.sqrt(var_pre)

            # --- Step 4: alpha, Phi, phi (exact) ---
            alpha = mu_pre / sigma_pre
            phi_alpha = flops.stats.norm.pdf(alpha).astype(fnp.float32)
            Phi_alpha = flops.stats.norm.cdf(alpha).astype(fnp.float32)

            # --- Step 5-6: exact per-neuron ReLU mean/variance (unchanged) ---
            mu_new = mu_pre * Phi_alpha + sigma_pre * phi_alpha
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var_post = fnp.maximum(ez2 - mu_new * mu_new, 0.0)

            _zero32 = fnp.zeros((), dtype=fnp.float32)
            gain = fnp.where(sigma_pre > 1e-12, Phi_alpha, _zero32)

            row_cov_mass = (fnp.sum(fnp.abs(cov_pre), axis=1) - fnp.abs(var_pre)) / width

            # Frozen calibration changes only the returned row, as upstream.
            mu_out = mu_new
            if (self._calibration is not None and width == 1024
                    and len(mlp.weights) == len(self._calibration)):
                c = self._calibration[li]
                resid_hat = (
                    mu_new * c[0] + var_pre * c[1] + alpha * c[2] + row_cov_mass * c[3] + c[4]
                )
                mu_out = mu_new + resid_hat * CALIBRATION_SCALE
            mu = mu_new
            rows.append(mu_out * float(fnp.exp(log_scale)))
            if li == len(mlp.weights) - 1:
                break  # No subsequent layer consumes this covariance.

            # --- Step 7 (EXACT): gain-product term + closed-form correction ---
            gain_product_term = fnp.multiply(fnp.outer(gain, gain), cov_pre)

            sigma_outer = fnp.outer(sigma_pre, sigma_pre)
            rho = cov_pre / fnp.maximum(sigma_outer, 1e-30)
            # clamp to (-0.999999, 0.999999) with documented ops only (no fnp.clip)
            rho = fnp.maximum(rho, -0.999999)
            rho = -fnp.maximum(-rho, -0.999999)
            alpha2 = alpha * alpha
            # a^2 + b^2 for every (i,j) pair, and the a*b outer product --
            # both O(width^2), computed once per layer (not once per
            # quadrature node). fnp.outer(alpha2, ones) + fnp.outer(ones,
            # alpha2) keeps this on the documented outer/broadcast path
            # rather than relying on numpy-style None-indexing in fnp.
            _ones_w = fnp.ones(width, dtype=fnp.float32)
            a2_plus_b2 = fnp.outer(alpha2, _ones_w) + fnp.outer(_ones_w, alpha2)
            ab_outer = fnp.outer(alpha, alpha)

            correction = fnp.zeros(rho.shape, dtype=fnp.float32)
            for node, weight in zip(_QUAD_NODES01, _QUAD_WEIGHTS01):
                s_k = rho * node
                one_minus_s2 = fnp.maximum((s_k * s_k) * -1.0 + 1.0, 1e-12)
                q = a2_plus_b2 - (s_k * ab_outer) * 2.0
                denom = one_minus_s2 * 2.0
                quad = q / denom
                dens = fnp.exp(-quad) / (fnp.sqrt(one_minus_s2) * 6.283185307179586)
                correction = correction + (weight * (1.0 - node)) * dens

            correction = (sigma_outer * (rho * rho)) * correction
            cov = gain_product_term + correction
            fnp.fill_diagonal(cov, var_post)
            cov = flops.as_symmetric(cov, symmetry=(0, 1))

        return fnp.stack(rows, axis=0)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=1024, depth=16, seed=0)
    compare_against_monte_carlo(Estimator(), mlp)
