"""Small-n correctness tests for the factorized K3 experiment module."""
from __future__ import annotations

import numpy as np

from experiments.factorized_k3 import dense_from_factors_np


def repeated_from_factors_np(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    repeated = ((b * c) @ a.T + (a * c) @ b.T + (a * b) @ c.T) / 3.0
    np.fill_diagonal(repeated, 0.0)
    return repeated


def test_factor_diag_matches_dense_tensor() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 3))
    b = rng.normal(size=(4, 3))
    c = rng.normal(size=(4, 3))
    dense = dense_from_factors_np(a, b, c)
    expected = np.einsum("ij,ij,ij->i", a, b, c)
    actual = np.array([dense[i, i, i] for i in range(4)])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_factor_repeated_matches_dense_tensor() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=(5, 4))
    b = rng.normal(size=(5, 4))
    c = rng.normal(size=(5, 4))
    dense = dense_from_factors_np(a, b, c)
    expected = repeated_from_factors_np(a, b, c)
    actual = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if i != j:
                actual[i, j] = dense[i, i, j]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_diagonal_correction_sets_requested_diagonal() -> None:
    rng = np.random.default_rng(2)
    n, j = 4, 3
    a = rng.normal(size=(n, j))
    b = rng.normal(size=(n, j))
    c = rng.normal(size=(n, j))
    target = rng.normal(size=n)
    current = np.einsum("ij,ij,ij->i", a, b, c)
    adjustment = target - current
    a2 = np.concatenate([a, np.outer(adjustment, np.ones(n))], axis=1)
    b2 = np.concatenate([b, np.eye(n)], axis=1)
    c2 = np.concatenate([c, np.eye(n)], axis=1)
    dense = dense_from_factors_np(a2, b2, c2)
    actual = np.array([dense[i, i, i] for i in range(n)])
    np.testing.assert_allclose(actual, target, rtol=1e-12, atol=1e-12)


def test_repeated_correction_sets_requested_repeated_slice() -> None:
    rng = np.random.default_rng(3)
    n, j = 4, 3
    a = rng.normal(size=(n, j))
    b = rng.normal(size=(n, j))
    c = rng.normal(size=(n, j))
    target = rng.normal(size=(n, n))
    np.fill_diagonal(target, 0.0)
    current = repeated_from_factors_np(a, b, c)
    adjustment = target - current
    a2 = np.concatenate([a, 3.0 * adjustment.T], axis=1)
    b2 = np.concatenate([b, np.eye(n)], axis=1)
    c2 = np.concatenate([c, np.eye(n)], axis=1)
    dense = dense_from_factors_np(a2, b2, c2)
    actual = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            if i != k:
                actual[i, k] = dense[i, i, k]
    np.testing.assert_allclose(actual, target, rtol=1e-12, atol=1e-12)


def test_linear_factor_propagation_matches_dense_tensor() -> None:
    rng = np.random.default_rng(4)
    n, j = 4, 5
    a = rng.normal(size=(n, j))
    b = rng.normal(size=(n, j))
    c = rng.normal(size=(n, j))
    w = rng.normal(size=(n, n))
    dense = dense_from_factors_np(a, b, c)
    propagated_dense = np.einsum("ijk,ia,jb,kc->abc", dense, w, w, w)
    propagated_factors = dense_from_factors_np(w.T @ a, w.T @ b, w.T @ c)
    np.testing.assert_allclose(propagated_factors, propagated_dense, rtol=1e-12, atol=1e-12)
