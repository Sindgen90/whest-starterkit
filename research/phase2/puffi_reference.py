"""Puffi-inspired finite cumulant reference, NOT the original A716 artifact.

Published equations: Puffi appendix A.7, A19-A28. Missing transport definitions
are completed by an explicit symmetric tensor projection described in
PUFFI_REFERENCE.md. Original beta and g4 values were not published. Unavailable
radial channel is disabled. Default beta=0; fitted subclass supplies our own.
Only numerical flopscope operations are used at inference.
"""
from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


def responses(mean, variance):
    """Gaussian positive-part moments M0..M4 and derivatives F[k][q], q=1..4."""
    sigma = fnp.sqrt(fnp.maximum(variance, 1e-12))
    r = mean / sigma
    phi = flops.stats.norm.pdf(r).astype(fnp.float32)
    cdf = flops.stats.norm.cdf(r).astype(fnp.float32)
    moments = [cdf, mean * cdf + sigma * phi]
    for k in range(2, 5):
        moments.append(mean * moments[k-1] + (k-1) * variance * moments[k-2])
    f1 = [None, cdf, phi / sigma, -r * phi / variance,
          (r*r - 1) * phi / (variance*sigma)]
    f2 = [None, 2*moments[1], 2*cdf, 2*f1[2], 2*f1[3]]
    f3 = [None, 3*moments[2], 6*moments[1], 6*cdf, 6*f1[2]]
    f4 = [None, 4*moments[3], 12*moments[2], 24*moments[1], 24*cdf]
    return moments, [None, f1, f2, f3, f4], sigma, r, phi


def third_slices(a, b):
    """T=sum sym(a_r,b_r,b_r): return T[i,i,i] and P[i,j]=T[i,i,j]."""
    ab = a*b
    b2 = b*b
    diagonal = fnp.sum(ab*b, axis=1)
    repeated = (b2 @ a.T + 2 * (ab @ b.T)) / 3
    return diagonal, repeated


def new_third_source(diagonal, repeated):
    # sym(A[:,i],e_i,e_i) has off-diagonal T[i,i,j]=A[j,i]/3.
    a = 3 * repeated.T
    fnp.fill_diagonal(a, diagonal)
    return a, fnp.eye(diagonal.shape[0], dtype=fnp.float32)


def fourth_transport(paired, weight):
    """Explicit closure: retain only source U[i,i,j,j] and permutations.

    The diagonal contraction is exact for that projected tensor. The off-
    diagonal plane uses a symmetric row-mean approximation, not exact transport.
    """
    square = weight*weight
    gram = square.T @ paired @ square
    diagonal = 3*fnp.diag(gram) - 2*((square*square).T @ fnp.diag(paired))
    row_mean = fnp.mean(gram, axis=1)
    surrogate = 0.5*(row_mean[:, None] + row_mean[None, :])
    fnp.fill_diagonal(surrogate, diagonal)
    return diagonal, surrogate, row_mean


def response_kernels(cov, repeated, fourth, diagonal3, propagation_scale):
    pl = repeated*cov
    dl = (propagation_scale/6) * diagonal3[:, None] * cov
    return (cov, 0.5*cov*cov + 0.25*fourth, 0.5*repeated,
            0.5*repeated.T, 0.5*pl, 0.5*pl.T,
            0.25*repeated*repeated.T, dl, dl.T)


def cross_covariance(left, right, kernels):
    # Published A25/A26. Diagonal matrices are applied as elementwise scalings.
    orders = ((1,1), (2,2), (2,1), (1,2), (3,2), (2,3), (3,3), (4,1), (1,4))
    terms = [left[a][:, None] * q * right[b][None, :]
             for q, (a,b) in zip(kernels, orders)]
    result = terms[0]
    for term in terms[1:]:
        result = result + term
    return result


class Estimator(BaseEstimator):
    reference_layers = 9
    history = 8
    propagation_scale = 0.8
    beta = None
    collect_diagnostics = False

    def predict(self, mlp, budget):
        n = mlp.width
        mu = fnp.zeros(n, dtype=fnp.float32)
        cov = flops.as_symmetric(fnp.eye(n, dtype=fnp.float32), symmetry=(0,1))
        fourth = fnp.zeros((n,n), dtype=fnp.float32)
        sources = []  # (creation layer, A, B), newest finite-window injections
        correction = fnp.zeros(n, dtype=fnp.float32)
        rows = []
        self.trace = []
        beta = (fnp.zeros((self.reference_layers,10), dtype=fnp.float32)
                if self.beta is None else fnp.asarray(self.beta, dtype=fnp.float32))
        for layer, weight in enumerate(mlp.weights):
            w = fnp.asarray(weight, dtype=fnp.float32)
            pre_mean = mu @ w
            pre_cov = fnp.einsum('ij,ia,jb->ab', cov, w, w)
            variance = fnp.maximum(fnp.diag(pre_cov), 1e-12)
            moments, f, sigma, r, phi = responses(pre_mean, variance)
            rich = layer < self.reference_layers
            if rich:
                sources = [(born, w.T @ a, w.T @ b) for born,a,b in sources
                           if layer-born <= self.history]
                if sources:
                    all_a = fnp.concatenate([a for _,a,_ in sources], axis=1)
                    all_b = fnp.concatenate([b for _,_,b in sources], axis=1)
                    d3, repeated = third_slices(all_a, all_b)
                else:
                    d3 = fnp.zeros(n, dtype=fnp.float32)
                    repeated = fnp.zeros((n,n), dtype=fnp.float32)
                d4, paired_pre, tau = fourth_transport(fourth, w)
                corrected = [moments[0]]
                corrections = []
                for k in range(1,5):
                    delta = f[k][3]*d3/6 + f[k][4]*d4/24
                    bound = 0.25*fnp.power(sigma,k)
                    delta = fnp.clip(delta, -bound, bound)
                    corrections.append(delta)
                    corrected.append(moments[k]+delta)
                mu = corrected[1]
                marginal_var = fnp.maximum(corrected[2]-mu*mu, 1e-12)
                k3 = corrected[3]-3*mu*corrected[2]+2*mu*mu*mu
                k4 = (corrected[4]-4*mu*corrected[3]-3*corrected[2]*corrected[2]
                      +12*mu*mu*corrected[2]-6*fnp.power(mu,4))
                kernels = response_kernels(pre_cov, repeated, paired_pre, d3, self.propagation_scale)
                cov = cross_covariance(f[1], f[1], kernels)
                cov = 0.5*(cov+cov.T)
                fnp.fill_diagonal(cov, marginal_var)

                # Reconstruction choices for the missing source-update equations:
                # mixed moments are obtained by applying the same response kernel
                # to h^2/h and h^2/h^2, then converting to centered cumulants.
                cov21 = cross_covariance(f[2], f[1], kernels)
                target21 = cov21 - 2*mu[:,None]*cov
                fnp.fill_diagonal(target21, k3)
                cov22 = cross_covariance(f[2], f[2], kernels)
                fourth = (cov22 - 2*mu[None,:]*cov21 - 2*mu[:,None]*cov21.T
                          +4*fnp.outer(mu,mu)*cov - 2*cov*cov)
                fourth = 0.5*(fourth+fourth.T)
                fnp.fill_diagonal(fourth, k4)

                # Carry old symmetric rank terms through local gains. Inject the
                # difference needed to match the current repeated/diagonal slices.
                gain = f[1][1]
                sources = [(born, self.propagation_scale*gain[:,None]*a, gain[:,None]*b)
                           for born,a,b in sources]
                # Drop the oldest BEFORE injection; the new term repairs slices
                # after truncation, avoiding a one-generation stale correction.
                sources = [(born,a,b) for born,a,b in sources if layer-born < self.history]
                if sources:
                    old3, old21 = third_slices(
                        fnp.concatenate([a for _,a,_ in sources],axis=1),
                        fnp.concatenate([b for _,_,b in sources],axis=1))
                    k3 = k3-old3
                    target21 = target21-old21
                a,b = new_third_source(k3,target21)
                sources.append((layer,a,b))

                safe_var = fnp.maximum(variance,1e-6)
                gamma3 = d3/(safe_var*fnp.sqrt(safe_var))
                gamma4 = d4/(safe_var*safe_var)
                chi = sigma*phi
                h2 = r*r-1
                h4 = fnp.power(r,4)-6*r*r+3
                zero = fnp.zeros(n,dtype=fnp.float32)
                # g4,L is unpublished, so h3 is explicitly disabled; h5 is zero
                # in the published estimator too. These beta columns stay zero.
                features = fnp.stack((corrections[0],f[1][3]*d3/6,zero,
                    f[1][4]*d4/24,zero,chi,chi*tau/(safe_var*safe_var),
                    chi*h2*gamma3*gamma3,chi*(3*r-r*r*r)*gamma3*gamma4,
                    chi*h4*gamma3*gamma3),axis=1)
                correction = (correction @ w)*gain + features @ beta[layer]
            else:
                # Continuation for the required 16-row API: ordinary gain closure.
                # This tail is our extension, not Puffi's sampled phase-one tail.
                sources = []
                mu = moments[1]
                gain = f[1][1]
                cov = fnp.outer(gain,gain)*pre_cov
                fnp.fill_diagonal(cov,fnp.maximum(moments[2]-mu*mu,1e-12))
                features = fnp.zeros((n,10),dtype=fnp.float32)
                correction = (correction @ w)*gain
            cov = flops.as_symmetric(cov,symmetry=(0,1))
            rows.append(fnp.maximum(mu-correction,0.0))
            if self.collect_diagnostics:
                self.trace.append((mu,features,gain))
        return fnp.stack(rows)
