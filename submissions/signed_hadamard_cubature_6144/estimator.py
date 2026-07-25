"""Submission-safe signed/permuted Hadamard spherical-cubature estimator.

For the Phase 1 width-256 benchmark this uses 12 independent signed/permuted
Hadamard bases.  Each basis contributes the 2d spherical-radial cubature points
``+/- sqrt(d) q_i``.  At d=256 this is 6144 propagated points, just below the
10% score-floor compute region observed locally.
"""
from __future__ import annotations

import math

import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

ROTATIONS = 12
ROTATION_SEED_OFFSET = 23_000_123


def hadamard_matrix(width: int) -> fnp.ndarray:
    if width <= 0 or width & (width - 1):
        raise ValueError("signed Hadamard cubature requires power-of-two width")
    scale = 1.0 / math.sqrt(width)
    rows = []
    for row in range(width):
        values = []
        for col in range(width):
            parity = (row & col).bit_count() & 1
            values.append(-scale if parity else scale)
        rows.append(values)
    return fnp.array(rows, dtype=fnp.float32)


def signed_hadamard_cubature_points(width: int, seed: int) -> tuple[fnp.ndarray, fnp.ndarray]:
    radius = math.sqrt(width)
    base = hadamard_matrix(width)
    rng = fnp.random.default_rng(seed + ROTATION_SEED_OFFSET + 10_007 * ROTATIONS)
    blocks = []
    for _ in range(ROTATIONS):
        permutation = rng.permutation(width)
        signs = rng.choice(fnp.array([-1.0, 1.0], dtype=fnp.float32), size=width).astype(fnp.float32)
        basis = signs[:, None] * base[permutation, :]
        blocks.append(radius * basis.T)
        blocks.append(-radius * basis.T)
    points = fnp.concatenate(blocks, axis=0).astype(fnp.float32)
    weights = fnp.full((2 * width * ROTATIONS,), 1.0 / (2 * width * ROTATIONS), dtype=fnp.float32)
    return points, weights


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        x, weights = signed_hadamard_cubature_points(mlp.width, mlp.seed)
        rows = []
        for w in mlp.weights:
            x = fnp.maximum(fnp.matmul(x, w), 0.0)
            rows.append(fnp.sum(x * weights[:, None], axis=0))
        return fnp.stack(rows, axis=0)
