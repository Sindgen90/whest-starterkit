"""NumPy research utilities for sparse positive empirical cubature."""
from __future__ import annotations

from typing import Any

import numpy as np


def standardize_rows(values: np.ndarray, floor: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center and scale each row across candidate nodes."""
    mean = values.mean(axis=1)
    scale = values.std(axis=1)
    scale = np.maximum(scale, floor)
    standardized = (values - mean[:, None]) / scale[:, None]
    return standardized, mean, scale


def top_right_singular_modes(
    snapshots: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return leading right singular modes without forming a huge full SVD."""
    if snapshots.ndim != 2:
        raise ValueError("snapshots must be a matrix")
    if rank <= 0 or rank > min(snapshots.shape):
        raise ValueError("invalid rank")
    gram = np.asarray(snapshots @ snapshots.T, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:rank]
    eigenvalues = np.maximum(eigenvalues[order], 1e-20)
    left = eigenvectors[:, order]
    singular_values = np.sqrt(eigenvalues)
    right = (left.T @ snapshots) / singular_values[:, None]
    return np.asarray(right, dtype=np.float64), singular_values


def mixed_leverage_subset(
    learned_modes: np.ndarray,
    gaussian_features: np.ndarray,
    support: int,
    seed: int,
    learned_fraction: float = 0.65,
    gaussian_fraction: float = 0.20,
) -> np.ndarray:
    """Sample a reproducible candidate subset from mixed leverage scores."""
    n_candidates = learned_modes.shape[1]
    if gaussian_features.shape[1] != n_candidates:
        raise ValueError("feature matrices must share the candidate axis")
    if support <= 0 or support > n_candidates:
        raise ValueError("invalid support")
    uniform_fraction = 1.0 - learned_fraction - gaussian_fraction
    if min(learned_fraction, gaussian_fraction, uniform_fraction) < 0.0:
        raise ValueError("invalid leverage mixture")
    learned = np.sum(learned_modes * learned_modes, axis=0)
    learned /= learned.sum()
    gaussian = np.sum(gaussian_features * gaussian_features, axis=0)
    gaussian /= gaussian.sum()
    probability = (
        learned_fraction * learned
        + gaussian_fraction * gaussian
        + uniform_fraction / n_candidates
    )
    probability /= probability.sum()
    rng = np.random.default_rng(seed)
    selected = rng.choice(n_candidates, size=support, replace=False, p=probability)
    return np.sort(selected.astype(np.int64))


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto nonnegative weights summing to one."""
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, len(values) + 1)
    positive = ordered - (cumulative - 1.0) / indices > 0.0
    rho = int(np.flatnonzero(positive)[-1])
    threshold = (cumulative[rho] - 1.0) / (rho + 1)
    return np.maximum(values - threshold, 0.0)


def _projected_moment_fit(
    features: np.ndarray,
    target: np.ndarray,
    iterations: int,
) -> np.ndarray:
    count = features.shape[1]
    weights = np.full(count, 1.0 / count, dtype=np.float64)
    vector = np.full(count, 1.0 / np.sqrt(count), dtype=np.float64)
    for _ in range(20):
        image = features.T @ (features @ vector)
        norm = np.linalg.norm(image)
        if norm == 0.0:
            break
        vector = image / norm
    lipschitz = float(vector @ (features.T @ (features @ vector)))
    step = 1.0 / max(lipschitz, 1e-12)
    for _ in range(iterations):
        gradient = features.T @ (features @ weights - target)
        weights = project_simplex(weights - step * gradient)
    return weights


def positive_moment_weights(
    features: np.ndarray,
    target: np.ndarray,
    selected: np.ndarray,
    max_active_iterations: int = 12,
    projected_iterations: int = 800,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit positive weights, first attempting exact affine moment matching."""
    if features.ndim != 2 or target.shape != (features.shape[0],):
        raise ValueError("incompatible feature target")
    original = np.asarray(selected, dtype=np.int64)
    active = original.copy()
    exact_weights: np.ndarray | None = None
    exact_residual = float("inf")
    for iteration in range(max_active_iterations):
        matrix = np.vstack(
            (np.ones((1, len(active)), dtype=np.float64), features[:, active])
        )
        rhs = np.concatenate(([1.0], target))
        base = np.full(len(active), 1.0 / len(active), dtype=np.float64)
        correction_rhs = rhs - matrix @ base
        gram = matrix @ matrix.T
        dual = np.linalg.lstsq(gram, correction_rhs, rcond=1e-11)[0]
        weights = base + matrix.T @ dual
        residual = float(np.max(np.abs(matrix @ weights - rhs)))
        minimum = float(weights.min())
        if minimum >= -1e-10 and residual < 1e-7:
            weights = np.maximum(weights, 0.0)
            weights /= weights.sum()
            exact_weights = weights
            exact_residual = residual
            return active, exact_weights, {
                "mode": "exact_active_set",
                "active_iterations": iteration + 1,
                "max_constraint_residual": exact_residual,
                "minimum_raw_weight": minimum,
            }
        negative = np.flatnonzero(weights < -1e-10)
        if not len(negative):
            break
        keep = np.ones(len(active), dtype=bool)
        keep[negative] = False
        if int(keep.sum()) <= features.shape[0] + 1:
            break
        active = active[keep]

    subset_features = features[:, original]
    fitted = _projected_moment_fit(
        subset_features,
        target,
        iterations=projected_iterations,
    )
    residual = subset_features @ fitted - target
    return original, fitted, {
        "mode": "projected_simplex",
        "active_iterations": max_active_iterations,
        "max_constraint_residual": float(np.max(np.abs(residual))),
        "rms_constraint_residual": float(np.sqrt(np.mean(residual * residual))),
        "minimum_raw_weight": float(fitted.min()),
    }
