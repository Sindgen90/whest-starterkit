from __future__ import annotations

import math
import importlib.util
import sys
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

from research.src.active_subspaces.gradients import input_jacobian_transpose_probe, relu_mlp_forward
from research.src.control_variates.hermite import (
    crossfit_projection_estimate_numpy,
    hadamard_random_basis_numpy,
    hermite_degree2_features_numpy,
)
from research.src.rqmc.points import LatticeConfig, numpy_lattice_gaussians, radial_expected_norm


def load_submission_module(relative_path: str):
    path = Path(relative_path)
    spec = importlib.util.spec_from_file_location(path.parent.name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rqmc_randomizations_are_reproducible_and_independent() -> None:
    config_a = LatticeConfig(n_samples=31, seed_offset=101, multiplier=17)
    config_b = LatticeConfig(n_samples=31, seed_offset=211, multiplier=17)
    x_a1 = numpy_lattice_gaussians(config_a, width=4, seed=123)
    x_a2 = numpy_lattice_gaussians(config_a, width=4, seed=123)
    x_b = numpy_lattice_gaussians(config_b, width=4, seed=123)
    np.testing.assert_allclose(x_a1, x_a2)
    assert not np.allclose(x_a1, x_b)


def test_fixed_hadamard_basis_is_orthonormal() -> None:
    q = hadamard_random_basis_numpy(width=256, rank=8)
    np.testing.assert_allclose(q.T @ q, np.eye(8), atol=1e-12)


def test_submission_hadamard_bases_handle_validator_and_challenge_widths() -> None:
    for relative_path in [
        "submission_variants/rqmc_hermite_random/estimator.py",
        "submission_variants/rqmc_hermite_active/estimator.py",
    ]:
        module = load_submission_module(relative_path)
        for width in [4, 256]:
            q = np.asarray(module.fixed_hadamard_basis(width, rank=4), dtype=np.float64)
            np.testing.assert_allclose(q.T @ q, np.eye(4), atol=1e-12)


def test_active_power_basis_is_deterministic_and_orthonormal() -> None:
    module = load_submission_module("submission_variants/rqmc_hermite_active/estimator.py")
    rng = np.random.default_rng(12345)
    gradients = fnp.array(rng.normal(size=(16, 8)), dtype=fnp.float32)
    with flops.BudgetContext(flop_budget=10_000_000, quiet=True):
        q1 = module.active_basis_from_gradients(gradients, width=16, rank=4)
    with flops.BudgetContext(flop_budget=10_000_000, quiet=True):
        q2 = module.active_basis_from_gradients(gradients, width=16, rank=4)
    q1_np = np.asarray(q1, dtype=np.float64)
    q2_np = np.asarray(q2, dtype=np.float64)
    np.testing.assert_allclose(q1_np, q2_np, atol=1e-12)
    np.testing.assert_allclose(q1_np.T @ q1_np, np.eye(4), atol=2e-6)


def test_downstream_sensitive_basis_is_deterministic_and_orthonormal() -> None:
    module = load_submission_module("research/scripts/run_hybrid_checkpoint_correction.py")
    rng = np.random.default_rng(424242)
    weights = [fnp.array(rng.normal(scale=0.2, size=(8, 8)), dtype=fnp.float32) for _ in range(4)]
    with flops.BudgetContext(flop_budget=10_000_000, quiet=True):
        q1 = module.downstream_sensitive_basis(weights, checkpoint_layer=1, width=8, rank=4, block_size=2, iterations=2)
    with flops.BudgetContext(flop_budget=10_000_000, quiet=True):
        q2 = module.downstream_sensitive_basis(weights, checkpoint_layer=1, width=8, rank=4, block_size=2, iterations=2)
    q1_np = np.asarray(q1, dtype=np.float64)
    q2_np = np.asarray(q2, dtype=np.float64)
    np.testing.assert_allclose(q1_np, q2_np, atol=1e-12)
    np.testing.assert_allclose(q1_np.T @ q1_np, np.eye(4), atol=2e-6)


def test_hermite_degree2_features_have_zero_mean_and_identity_covariance() -> None:
    rng = np.random.default_rng(20260723)
    x = rng.normal(size=(80_000, 16))
    q = hadamard_random_basis_numpy(width=16, rank=4, columns=(1, 3, 5, 9))
    phi = hermite_degree2_features_numpy(x, q)
    assert phi.shape == (80_000, 14)
    assert np.max(np.abs(phi.mean(axis=0))) < 0.02
    np.testing.assert_allclose(np.cov(phi, rowvar=False), np.eye(14), atol=0.045)


def test_crossfit_cv_is_unbiased_and_reduces_span_function_error() -> None:
    rng = np.random.default_rng(260505179)
    q = np.eye(2)
    x_a = rng.normal(size=(10_000, 2))
    x_b = rng.normal(size=(10_000, 2))
    phi_a = hermite_degree2_features_numpy(x_a, q)
    phi_b = hermite_degree2_features_numpy(x_b, q)
    y_a = (x_a[:, 0] + x_a[:, 1] * x_a[:, 1] - 1.0)[:, None]
    y_b = (x_b[:, 0] + x_b[:, 1] * x_b[:, 1] - 1.0)[:, None]
    ordinary = float(0.5 * (y_a.mean() + y_b.mean()))
    cv = float(crossfit_projection_estimate_numpy(y_a, phi_a, y_b, phi_b, shrinkage=1.0)[0])
    assert abs(cv) < 0.004
    assert abs(cv) < abs(ordinary)


def test_radial_expected_norm_and_relu_analytic_mean() -> None:
    assert math.isclose(radial_expected_norm(1), math.sqrt(2.0 / math.pi), rel_tol=1e-12)
    assert math.isclose(radial_expected_norm(2), math.sqrt(math.pi / 2.0), rel_tol=1e-12)
    rng = np.random.default_rng(8675309)
    dim = 256
    a = rng.normal(size=dim)
    z = rng.normal(size=(50_000, dim))
    u = z / np.linalg.norm(z, axis=1, keepdims=True)
    radial_estimate = radial_expected_norm(dim) * np.maximum(u @ a, 0.0).mean()
    exact = np.linalg.norm(a) / math.sqrt(2.0 * math.pi)
    assert abs(radial_estimate - exact) / exact < 0.015


def test_relu_mlp_is_positively_homogeneous_without_biases() -> None:
    rng = np.random.default_rng(314159)
    weights = [rng.normal(scale=0.1, size=(5, 5)) for _ in range(3)]
    x = rng.normal(size=5)
    scale = 2.7
    np.testing.assert_allclose(
        relu_mlp_forward(weights, scale * x),
        scale * relu_mlp_forward(weights, x),
        atol=1e-12,
    )


def test_manual_input_jacobian_transpose_probe_matches_finite_difference() -> None:
    rng = np.random.default_rng(271828)
    weights = [rng.normal(scale=0.2, size=(4, 4)) for _ in range(3)]
    x = rng.normal(size=4)
    probe = rng.normal(size=4)
    manual = input_jacobian_transpose_probe(weights, x, probe)
    eps = 1e-5
    finite_difference = np.empty_like(x)
    for index in range(x.size):
        perturb = np.zeros_like(x)
        perturb[index] = eps
        plus = probe @ relu_mlp_forward(weights, x + perturb)
        minus = probe @ relu_mlp_forward(weights, x - perturb)
        finite_difference[index] = (plus - minus) / (2.0 * eps)
    np.testing.assert_allclose(manual, finite_difference, rtol=2e-5, atol=2e-6)
