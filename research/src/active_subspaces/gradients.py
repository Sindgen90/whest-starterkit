from __future__ import annotations

import numpy as np


def relu_forward_with_cache(weights: list[np.ndarray], x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Forward pass for row-vector ReLU MLPs, returning final activation and ReLU gates."""
    activation = np.asarray(x, dtype=np.float64)
    gates: list[np.ndarray] = []
    for weight in weights:
        preactivation = activation @ np.asarray(weight, dtype=np.float64)
        gate = preactivation > 0.0
        gates.append(gate)
        activation = np.maximum(preactivation, 0.0)
    return activation, gates


def relu_mlp_forward(weights: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    return relu_forward_with_cache(weights, x)[0]


def input_jacobian_transpose_probe(
    weights: list[np.ndarray],
    x: np.ndarray,
    output_probe: np.ndarray,
) -> np.ndarray:
    """Manual J_f(x)^T r for the challenge's bias-free row-vector ReLU MLP convention."""
    _, gates = relu_forward_with_cache(weights, x)
    v = np.asarray(output_probe, dtype=np.float64)
    for weight, gate in zip(reversed(weights), reversed(gates), strict=True):
        v_pre = v * gate
        v = np.asarray(weight, dtype=np.float64) @ v_pre
    return v

