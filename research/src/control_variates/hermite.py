from __future__ import annotations

import math
from collections.abc import Sequence

import flopscope.numpy as fnp
import numpy as np


DEFAULT_RANDOM_HADAMARD_COLUMNS = (11, 23, 37, 58, 91, 117, 149, 203, 221, 7, 171, 45, 199, 83, 139, 245)


def feature_count(rank: int) -> int:
    return rank + rank * (rank + 1) // 2


def _check_hadamard_width(width: int) -> None:
    if width <= 0 or width & (width - 1):
        raise ValueError(f"Hadamard basis requires a power-of-two width, got {width}")


def hadamard_random_basis_numpy(
    width: int,
    rank: int,
    columns: Sequence[int] = DEFAULT_RANDOM_HADAMARD_COLUMNS,
) -> np.ndarray:
    """Fixed dense orthonormal basis independent of both network and RQMC points.

    The columns are a deterministic, preselected subset of a Walsh-Hadamard matrix.
    Challenge width is 256, so this is equivalent to storing a fixed random
    orthogonal matrix while keeping the submission file compact.
    """
    _check_hadamard_width(width)
    if rank > len(columns):
        raise ValueError(f"rank {rank} exceeds available fixed columns {len(columns)}")
    selected = tuple(int(c) for c in columns[:rank])
    if len(set(selected)) != rank or min(selected) < 0 or max(selected) >= width:
        raise ValueError(f"invalid Hadamard column selection for width {width}: {selected}")
    scale = 1.0 / math.sqrt(width)
    basis = np.empty((width, rank), dtype=np.float64)
    for row in range(width):
        for col_index, col in enumerate(selected):
            parity = (row & col).bit_count() & 1
            basis[row, col_index] = -scale if parity else scale
    return basis


def hadamard_random_basis_fnp(
    width: int,
    rank: int,
    columns: Sequence[int] = DEFAULT_RANDOM_HADAMARD_COLUMNS,
) -> fnp.ndarray:
    return fnp.array(hadamard_random_basis_numpy(width, rank, columns), dtype=fnp.float32)


def hermite_degree2_features_numpy(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) @ np.asarray(q, dtype=np.float64)
    rank = int(z.shape[1])
    features: list[np.ndarray] = []
    for i in range(rank):
        features.append(z[:, i])
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for i in range(rank):
        features.append((z[:, i] * z[:, i] - 1.0) * inv_sqrt2)
    for i in range(rank):
        for j in range(i + 1, rank):
            features.append(z[:, i] * z[:, j])
    return np.stack(features, axis=1)


def hermite_degree2_features_fnp(x: fnp.ndarray, q: fnp.ndarray) -> fnp.ndarray:
    z = fnp.matmul(x, q)
    rank = int(z.shape[1])
    features = []
    for i in range(rank):
        features.append(z[:, i])
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for i in range(rank):
        features.append((z[:, i] * z[:, i] - 1.0) * inv_sqrt2)
    for i in range(rank):
        for j in range(i + 1, rank):
            features.append(z[:, i] * z[:, j])
    return fnp.stack(features, axis=1)


def projection_coefficients_numpy(phi: np.ndarray, y: np.ndarray, shrinkage: float = 1.0) -> np.ndarray:
    centered = np.asarray(y, dtype=np.float64) - np.mean(y, axis=0, keepdims=True)
    return float(shrinkage) * (np.asarray(phi, dtype=np.float64).T @ centered) / float(phi.shape[0])


def crossfit_projection_estimate_numpy(
    y_a: np.ndarray,
    phi_a: np.ndarray,
    y_b: np.ndarray,
    phi_b: np.ndarray,
    shrinkage: float = 1.0,
) -> np.ndarray:
    beta_a = projection_coefficients_numpy(phi_a, y_a, shrinkage)
    beta_b = projection_coefficients_numpy(phi_b, y_b, shrinkage)
    mu_b_given_a = np.mean(y_b, axis=0) - np.mean(phi_b, axis=0) @ beta_a
    mu_a_given_b = np.mean(y_a, axis=0) - np.mean(phi_a, axis=0) @ beta_b
    n_a = float(y_a.shape[0])
    n_b = float(y_b.shape[0])
    return (n_b * mu_b_given_a + n_a * mu_a_given_b) / (n_a + n_b)

