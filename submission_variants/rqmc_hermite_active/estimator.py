"""Cross-fitted Hermite/Stein RQMC in a network-specific active basis."""
from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

FOLD_SAMPLES = 646
ACTIVE_RANK = 4
RANK = ACTIVE_RANK
PILOT_SAMPLES = 8
OUTPUT_PROBES_PER_SAMPLE = 2
POWER_ITERATIONS = 3
SHRINKAGE = 0.5
PILOT_SEED_OFFSET = 21_000_041
RQMC_SEED_OFFSET_A = 1_000_003
RQMC_SEED_OFFSET_B = 11_000_022
PROBE_SEED_OFFSET = 27_182_818
RQMC_MULTIPLIER = 17_797
RANDOM_HADAMARD_COLUMNS = (11, 23, 37, 58, 91, 117, 149, 203, 221, 7, 171, 45, 199, 83, 139, 245)


def normal_lattice_samples(n_samples: int, width: int, seed: int, seed_offset: int) -> fnp.ndarray:
    rng = fnp.random.default_rng(seed + seed_offset)
    powers = []
    value = 1
    modulus = max(2, n_samples)
    for _ in range(width):
        powers.append(float(value))
        value = (value * RQMC_MULTIPLIER) % modulus
        if value == 0:
            value = 1
    generator = fnp.array(powers, dtype=fnp.float64)
    indices = fnp.arange(n_samples, dtype=fnp.float64) + 0.5
    shift = rng.uniform(0.0, 1.0, size=(width,))
    u = fnp.mod(fnp.outer(indices, generator) / n_samples + shift, 1.0)
    u = fnp.clip(u, 1e-7, 1.0 - 1e-7)
    return flops.stats.norm.ppf(u).astype(fnp.float32)


def fixed_hadamard_basis(width: int, rank: int) -> fnp.ndarray:
    if width <= 0 or width & (width - 1):
        raise ValueError("active Hermite basis requires power-of-two input width")
    if rank > width:
        raise ValueError("active rank exceeds input width")
    columns = []
    for col in RANDOM_HADAMARD_COLUMNS:
        candidate = col % width
        if candidate not in columns:
            columns.append(candidate)
        if len(columns) == rank:
            break
    fill = 1
    while len(columns) < rank:
        if fill not in columns:
            columns.append(fill)
        fill += 1
    scale = 1.0 / math.sqrt(width)
    rows = []
    for row in range(width):
        values = []
        for col in columns:
            parity = (row & col).bit_count() & 1
            values.append(-scale if parity else scale)
        rows.append(values)
    return fnp.array(rows, dtype=fnp.float32)


def rademacher_probes(width: int, count: int, seed: int) -> fnp.ndarray:
    rng = fnp.random.default_rng(seed + PROBE_SEED_OFFSET)
    uniforms = rng.uniform(0.0, 1.0, size=(count, width))
    scale = 1.0 / math.sqrt(width)
    return fnp.where(uniforms < 0.5, -scale, scale).astype(fnp.float32)


def forward_single_with_gates(x: fnp.ndarray, weights: list[fnp.ndarray]) -> tuple[list[fnp.ndarray], fnp.ndarray]:
    activation = x
    gates = []
    for w in weights:
        preactivation = fnp.matmul(activation, w)
        gate = preactivation > 0.0
        gates.append(gate)
        activation = fnp.maximum(preactivation, 0.0)
    return gates, activation


def reverse_probe(gates: list[fnp.ndarray], weights: list[fnp.ndarray], probe: fnp.ndarray) -> fnp.ndarray:
    v = probe
    for layer in range(len(weights) - 1, -1, -1):
        v_pre = v * gates[layer]
        v = fnp.matmul(weights[layer], v_pre)
    return v


def pilot_forward_gates(mlp: MLP, pilot_x: fnp.ndarray) -> list[list[fnp.ndarray]]:
    all_gates = []
    for sample_index in range(int(pilot_x.shape[0])):
        gates, _ = forward_single_with_gates(pilot_x[sample_index], mlp.weights)
        all_gates.append(gates)
    return all_gates


def pilot_reverse_gradients(
    mlp: MLP,
    all_gates: list[list[fnp.ndarray]],
    probes: fnp.ndarray,
) -> fnp.ndarray:
    gradients = []
    probe_index = 0
    for gates in all_gates:
        for _ in range(OUTPUT_PROBES_PER_SAMPLE):
            gradients.append(reverse_probe(gates, mlp.weights, probes[probe_index]))
            probe_index += 1
    return fnp.stack(gradients, axis=1)


def pilot_gradients(mlp: MLP, pilot_x: fnp.ndarray, probes: fnp.ndarray) -> fnp.ndarray:
    gates = pilot_forward_gates(mlp, pilot_x)
    return pilot_reverse_gradients(mlp, gates, probes)


def modified_gram_schmidt(q: fnp.ndarray) -> fnp.ndarray:
    columns = []
    rank = int(q.shape[1])
    for j in range(rank):
        v = q[:, j]
        for previous in columns:
            coefficient = fnp.sum(previous * v)
            v = v - coefficient * previous
        norm = fnp.sqrt(fnp.sum(v * v))
        v = v / fnp.maximum(norm, 1e-30)
        columns.append(v)
    return fnp.stack(columns, axis=1)


def active_basis_from_gradients(gradients: fnp.ndarray, width: int, rank: int) -> fnp.ndarray:
    q = fixed_hadamard_basis(width, rank)
    for _ in range(POWER_ITERATIONS):
        q = fnp.matmul(gradients, fnp.matmul(fnp.swapaxes(gradients, 0, 1), q))
        q = modified_gram_schmidt(q)
    return q


def hermite_degree2_features(x: fnp.ndarray, q: fnp.ndarray) -> fnp.ndarray:
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


def projection_coefficients(phi: fnp.ndarray, y: fnp.ndarray) -> fnp.ndarray:
    centered = y - fnp.mean(y, axis=0)
    return (SHRINKAGE / float(phi.shape[0])) * fnp.matmul(fnp.swapaxes(phi, 0, 1), centered)


def crossfit_projection_mean(
    y_a: fnp.ndarray,
    phi_a: fnp.ndarray,
    y_b: fnp.ndarray,
    phi_b: fnp.ndarray,
) -> fnp.ndarray:
    beta_a = projection_coefficients(phi_a, y_a)
    beta_b = projection_coefficients(phi_b, y_b)
    mu_b_given_a = fnp.mean(y_b, axis=0) - fnp.matmul(fnp.mean(phi_b, axis=0), beta_a)
    mu_a_given_b = fnp.mean(y_a, axis=0) - fnp.matmul(fnp.mean(phi_a, axis=0), beta_b)
    n_a = float(y_a.shape[0])
    n_b = float(y_b.shape[0])
    return (n_b * mu_b_given_a + n_a * mu_a_given_b) / (n_a + n_b)


def fold_forward_layers(
    x_a: fnp.ndarray,
    x_b: fnp.ndarray,
    weights: list[fnp.ndarray],
) -> tuple[list[fnp.ndarray], list[fnp.ndarray]]:
    layers_a = []
    layers_b = []
    y_a = x_a
    y_b = x_b
    for w in weights:
        y_a = fnp.maximum(fnp.matmul(y_a, w), 0.0)
        y_b = fnp.maximum(fnp.matmul(y_b, w), 0.0)
        layers_a.append(y_a)
        layers_b.append(y_b)
    return layers_a, layers_b


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def _pilot_inputs_and_probes(self, mlp: MLP) -> tuple[fnp.ndarray, fnp.ndarray]:
        pilot_x = normal_lattice_samples(PILOT_SAMPLES, mlp.width, mlp.seed, PILOT_SEED_OFFSET)
        probes = rademacher_probes(
            mlp.width,
            PILOT_SAMPLES * OUTPUT_PROBES_PER_SAMPLE,
            mlp.seed,
        )
        return pilot_x, probes

    def _active_basis(self, mlp: MLP, pilot_x: fnp.ndarray, probes: fnp.ndarray) -> fnp.ndarray:
        gradients = pilot_gradients(mlp, pilot_x, probes)
        return active_basis_from_gradients(gradients, mlp.width, ACTIVE_RANK)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        pilot_x, probes = self._pilot_inputs_and_probes(mlp)
        q = self._active_basis(mlp, pilot_x, probes)
        x_a = normal_lattice_samples(FOLD_SAMPLES, mlp.width, mlp.seed, RQMC_SEED_OFFSET_A)
        x_b = normal_lattice_samples(FOLD_SAMPLES, mlp.width, mlp.seed, RQMC_SEED_OFFSET_B)
        phi_a = hermite_degree2_features(x_a, q)
        phi_b = hermite_degree2_features(x_b, q)
        rows = []
        y_a = x_a
        y_b = x_b
        for w in mlp.weights:
            y_a = fnp.maximum(fnp.matmul(y_a, w), 0.0)
            y_b = fnp.maximum(fnp.matmul(y_b, w), 0.0)
            rows.append(crossfit_projection_mean(y_a, phi_a, y_b, phi_b))
        return fnp.stack(rows, axis=0)

    def profile_components(self, mlp: MLP, budget: int) -> dict[str, int]:
        """Research-only component FLOP profile; not called by the official grader."""
        component_budget = max(int(budget), 1)
        profile = {}
        with flops.BudgetContext(flop_budget=component_budget, quiet=True, namespace="active.pilot_sampling") as context:
            pilot_x, probes = self._pilot_inputs_and_probes(mlp)
        profile["pilot_sampling_flops"] = int(context.flops_used)
        with flops.BudgetContext(flop_budget=component_budget, quiet=True, namespace="active.pilot_forward") as context:
            gates = pilot_forward_gates(mlp, pilot_x)
        profile["pilot_forward_flops"] = int(context.flops_used)
        with flops.BudgetContext(flop_budget=component_budget, quiet=True, namespace="active.pilot_reverse") as context:
            gradients = pilot_reverse_gradients(mlp, gates, probes)
        profile["pilot_reverse_flops"] = int(context.flops_used)
        with flops.BudgetContext(flop_budget=component_budget, quiet=True, namespace="active.basis") as context:
            q = active_basis_from_gradients(gradients, mlp.width, ACTIVE_RANK)
        profile["basis_estimation_flops"] = int(context.flops_used)
        with flops.BudgetContext(flop_budget=component_budget, quiet=True, namespace="active.fold_sampling") as context:
            x_a = normal_lattice_samples(FOLD_SAMPLES, mlp.width, mlp.seed, RQMC_SEED_OFFSET_A)
            x_b = normal_lattice_samples(FOLD_SAMPLES, mlp.width, mlp.seed, RQMC_SEED_OFFSET_B)
        profile["fold_sampling_flops"] = int(context.flops_used)
        with flops.BudgetContext(flop_budget=component_budget, quiet=True, namespace="active.features") as context:
            phi_a = hermite_degree2_features(x_a, q)
            phi_b = hermite_degree2_features(x_b, q)
        profile["feature_flops"] = int(context.flops_used)
        with flops.BudgetContext(flop_budget=component_budget, quiet=True, namespace="active.forward_eval") as context:
            layers_a, layers_b = fold_forward_layers(x_a, x_b, mlp.weights)
        profile["evaluation_forward_flops"] = int(context.flops_used)
        with flops.BudgetContext(flop_budget=component_budget, quiet=True, namespace="active.regression") as context:
            rows = []
            for y_a, y_b in zip(layers_a, layers_b, strict=True):
                rows.append(crossfit_projection_mean(y_a, phi_a, y_b, phi_b))
            _ = fnp.stack(rows, axis=0)
        profile["regression_flops"] = int(context.flops_used)
        profile["profiled_total_flops"] = int(sum(profile.values()))
        return profile

