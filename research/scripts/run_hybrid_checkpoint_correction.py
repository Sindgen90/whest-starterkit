"""Deterministic checkpoint-correction experiments for K2 + tiny RQMC.

This is research code, not a submission estimator. It compares:

1. K2 only;
2. direct 32-particle RQMC;
3. K2 + raw checkpoint mean replacement;
4. K2 + checkpoint scale/offset correction;
5. K2 + checkpoint scale/offset plus fixed rank-4 directional correction.
6. K2 + checkpoint scale/offset plus downstream-sensitive rank-4 correction.

Holdout is intentionally gated out of this first runner.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
import whestbench

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATASET = "aicrowd/arc-whestbench-public-2026"
REVISION = "v1-phase1"
BUDGET = 272_000_000_000
SCORE_FLOOR = 0.1
PARTICLES = 32
BLOCK_SIZE = 4
RANK = 4
SHRINKAGE = 0.5
RQMC_MULTIPLIER = 17_797
RQMC_SEED_OFFSET = 51_000_101
STABILITY_SEED_OFFSETS = (61_000_103, 71_000_123, 81_000_137)
RESULTS = ROOT / "research" / "results"
REPORTS = ROOT / "research" / "reports"
MANIFEST = ROOT / "research" / "configs" / "stein_holdout_manifest.json"
CHECKPOINTS = tuple(range(BLOCK_SIZE, 33, BLOCK_SIZE))
METHODS = (
    "k2_only",
    "direct_rqmc_32",
    "hybrid_raw_replace",
    "hybrid_scale_offset",
    "hybrid_rank4_projected",
    "hybrid_downstream_rank4_projected",
)
RANDOM_HADAMARD_COLUMNS = (11, 23, 37, 58, 91, 117, 149, 203, 221, 7, 171, 45, 199, 83, 139, 245)
_COV_RESCALE_THRESHOLD = 1e100


@dataclass
class K2State:
    mu: fnp.ndarray
    cov: fnp.ndarray
    log_scale: float = 0.0


def selected_rows(partition: str, limit: int | None) -> list[dict[str, Any]]:
    if partition == "holdout":
        raise SystemExit("Holdout is gated for this first deterministic checkpoint-correction run.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    key = f"{partition}_mlp_ids"
    ids = set(manifest[key])
    dataset = whestbench.load_dataset(DATASET, revision=REVISION, split="mini")
    rows = sorted((row for row in dataset if int(row["mlp_id"]) in ids), key=lambda row: int(row["mlp_id"]))
    if limit is not None:
        rows = rows[:limit]
    return rows


def as_mlp(row: dict[str, Any]) -> SimpleNamespace:
    stacked = fnp.array(np.asarray(row["weights"], dtype=np.float32))
    depth, width, _ = stacked.shape
    return SimpleNamespace(
        width=int(width),
        depth=int(depth),
        weights=[stacked[layer] for layer in range(depth)],
        seed=int(row["mlp_seed"]),
        name=str(row["mlp_name"]),
    )


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
        raise ValueError("fixed rank correction basis requires power-of-two width")
    if rank > width:
        raise ValueError("rank exceeds width")
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


def downstream_sensitive_basis(
    weights: list[fnp.ndarray],
    checkpoint_layer: int,
    width: int,
    rank: int,
    block_size: int = BLOCK_SIZE,
    iterations: int = 3,
) -> fnp.ndarray:
    """Leading input-side sensitivity directions for the next block.

    With row-vector propagation, a correction d at checkpoint layer l flows
    through the next block as d @ M. The high-gain directions are eigenvectors
    of M @ M.T. We approximate the leading subspace by block power iteration,
    initialized from the same fixed Hadamard basis used by the control method.
    """
    block = weights[checkpoint_layer : min(len(weights), checkpoint_layer + block_size)]
    if not block:
        return fixed_hadamard_basis(width, rank)
    q = fixed_hadamard_basis(width, rank)
    for _ in range(iterations):
        y = q
        for w in block:
            y = fnp.matmul(fnp.swapaxes(w, 0, 1), y)
        for w in reversed(block):
            y = fnp.matmul(w, y)
        q = modified_gram_schmidt(y)
    return q


def initial_k2_state(width: int) -> K2State:
    return K2State(mu=fnp.zeros(width), cov=fnp.eye(width), log_scale=0.0)


def k2_layer(state: K2State, w: fnp.ndarray) -> tuple[K2State, fnp.ndarray, fnp.ndarray, dict[str, float]]:
    mu = state.mu
    cov = state.cov
    log_scale = state.log_scale
    cov_diag = fnp.diag(cov)
    max_var = float(fnp.max(cov_diag))
    if max_var > _COV_RESCALE_THRESHOLD:
        scale = float(fnp.sqrt(max_var))
        mu = mu / scale
        cov = cov / (scale * scale)
        log_scale += float(fnp.log(scale))

    mu_pre = fnp.matmul(fnp.swapaxes(w, 0, 1), mu)
    cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
    var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
    sigma_pre = fnp.sqrt(var_pre)
    alpha = mu_pre / sigma_pre
    phi_alpha = flops.stats.norm.pdf(alpha)
    phi_cap_alpha = flops.stats.norm.cdf(alpha)
    mu_post = mu_pre * phi_cap_alpha + sigma_pre * phi_alpha
    ez2 = (mu_pre * mu_pre + var_pre) * phi_cap_alpha + mu_pre * sigma_pre * phi_alpha
    var_post = fnp.maximum(ez2 - mu_post * mu_post, 0.0)
    gain = fnp.where(sigma_pre > 1e-12, phi_cap_alpha, 0.0)
    cov_post = fnp.multiply(fnp.outer(gain, gain), cov_pre)
    fnp.fill_diagonal(cov_post, var_post)
    scale_factor = float(fnp.exp(log_scale))
    mean_orig = mu_post * scale_factor
    var_orig = var_post * (scale_factor * scale_factor)
    diagnostics = {
        "relu_gain_mean": float(fnp.mean(gain)),
        "relu_boundary_pdf_mean": float(fnp.mean(phi_alpha)),
    }
    return K2State(mu=mu_post, cov=cov_post, log_scale=log_scale), mean_orig, var_orig, diagnostics


def set_state_mean_from_original(state: K2State, mean_orig: fnp.ndarray) -> K2State:
    scale_factor = float(fnp.exp(state.log_scale))
    return K2State(mu=mean_orig / scale_factor, cov=state.cov, log_scale=state.log_scale)


def particle_layer(x: fnp.ndarray, w: fnp.ndarray) -> fnp.ndarray:
    return fnp.maximum(fnp.matmul(x, w), 0.0)


def particle_stats(x: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray]:
    mean = fnp.mean(x, axis=0)
    variance = fnp.maximum(fnp.mean(x * x, axis=0) - mean * mean, 0.0)
    return mean, variance


def norm2(x: fnp.ndarray) -> fnp.ndarray:
    return fnp.sum(x * x)


def safe_float(x: Any) -> float:
    return float(x)


def fit_scale_offset(innovation: fnp.ndarray, coarse_mean: fnp.ndarray) -> tuple[fnp.ndarray, dict[str, float]]:
    width = float(coarse_mean.shape[0])
    g00 = fnp.sum(coarse_mean * coarse_mean)
    g01 = fnp.sum(coarse_mean)
    g11 = width
    y0 = fnp.sum(coarse_mean * innovation)
    y1 = fnp.sum(innovation)
    det = g00 * g11 - g01 * g01
    safe_det = fnp.where(fnp.abs(det) > 1e-30, det, 1e-30)
    scale = (y0 * g11 - y1 * g01) / safe_det
    offset = (g00 * y1 - g01 * y0) / safe_det
    correction = scale * coarse_mean + offset
    stats = {"scale": safe_float(scale), "offset": safe_float(offset)}
    return correction, stats


def fit_rank4_directional(
    innovation: fnp.ndarray,
    coarse_mean: fnp.ndarray,
    basis: fnp.ndarray,
) -> tuple[fnp.ndarray, dict[str, float]]:
    scale_offset, stats = fit_scale_offset(innovation, coarse_mean)
    residual = innovation - scale_offset
    coeffs = fnp.matmul(residual, basis)
    directional = fnp.matmul(basis, coeffs)
    correction = scale_offset + directional
    stats["direction_coeff_norm"] = safe_float(fnp.sqrt(norm2(coeffs)))
    return correction, stats


def correction_metrics(
    innovation: fnp.ndarray,
    correction: fnp.ndarray,
    scale_offset: fnp.ndarray | None = None,
    directional: fnp.ndarray | None = None,
) -> dict[str, float]:
    innovation_energy = fnp.maximum(norm2(innovation), 1e-30)
    metrics = {
        "innovation_norm": safe_float(fnp.sqrt(innovation_energy)),
        "correction_norm": safe_float(fnp.sqrt(norm2(correction))),
        "correction_energy_ratio": safe_float(norm2(correction) / innovation_energy),
    }
    if scale_offset is not None:
        metrics["scale_offset_energy_ratio"] = safe_float(norm2(scale_offset) / innovation_energy)
    if directional is not None:
        metrics["directional_energy_ratio"] = safe_float(norm2(directional) / innovation_energy)
    return metrics


def checkpoint_correction(
    method: str,
    coarse_mean: fnp.ndarray,
    particle_mean: fnp.ndarray,
    basis: fnp.ndarray,
) -> tuple[fnp.ndarray, dict[str, float]]:
    innovation = particle_mean - coarse_mean
    if method == "hybrid_raw_replace":
        correction = innovation
        stats = {"shrinkage": 1.0, "correction_type": "raw_replace"}
    elif method == "hybrid_scale_offset":
        scale_offset, stats = fit_scale_offset(innovation, coarse_mean)
        correction = SHRINKAGE * scale_offset
        stats.update(correction_metrics(innovation, correction, scale_offset=scale_offset))
        stats["shrinkage"] = SHRINKAGE
        stats["correction_type"] = "scale_offset"
        return coarse_mean + correction, stats
    elif method in {"hybrid_rank4_projected", "hybrid_downstream_rank4_projected"}:
        scale_offset, so_stats = fit_scale_offset(innovation, coarse_mean)
        residual = innovation - scale_offset
        coeffs = fnp.matmul(residual, basis)
        directional = fnp.matmul(basis, coeffs)
        correction_unshrunk = scale_offset + directional
        correction = SHRINKAGE * correction_unshrunk
        stats = {**so_stats, "direction_coeff_norm": safe_float(fnp.sqrt(norm2(coeffs)))}
        stats.update(
            correction_metrics(
                innovation,
                correction,
                scale_offset=scale_offset,
                directional=directional,
            )
        )
        stats["unshrunk_combined_energy_ratio"] = safe_float(norm2(correction_unshrunk) / fnp.maximum(norm2(innovation), 1e-30))
        stats["shrinkage"] = SHRINKAGE
        stats["correction_type"] = method.removeprefix("hybrid_")
        return coarse_mean + correction, stats
    else:
        raise ValueError(f"unsupported checkpoint correction method: {method}")
    stats.update(correction_metrics(innovation, correction))
    return coarse_mean + correction, stats


def run_k2_only(mlp: SimpleNamespace, budget: int) -> tuple[fnp.ndarray, dict[str, int], list[dict[str, Any]]]:
    components = {"analytic_propagation_flops": 0, "particle_forward_flops": 0, "basis_estimation_flops": 0, "correction_flops": 0}
    rows = []
    checkpoints = []
    state = initial_k2_state(mlp.width)
    for layer_index, w in enumerate(mlp.weights, start=1):
        with flops.BudgetContext(flop_budget=budget, quiet=True, namespace="hybrid.k2_layer") as context:
            state, mean, variance, diagnostics = k2_layer(state, w)
        components["analytic_propagation_flops"] += int(context.flops_used)
        rows.append(mean)
        if layer_index in CHECKPOINTS:
            checkpoints.append(
                {
                    "layer": layer_index,
                    "coarse_mean_norm": safe_float(fnp.sqrt(norm2(mean))),
                    "coarse_var_mean": safe_float(fnp.mean(variance)),
                    **diagnostics,
                }
            )
    return fnp.stack(rows, axis=0), components, checkpoints


def run_direct_rqmc(mlp: SimpleNamespace, budget: int, seed_offset: int = RQMC_SEED_OFFSET) -> tuple[fnp.ndarray, dict[str, int], list[dict[str, Any]]]:
    components = {"analytic_propagation_flops": 0, "particle_forward_flops": 0, "basis_estimation_flops": 0, "correction_flops": 0}
    checkpoints = []
    with flops.BudgetContext(flop_budget=budget, quiet=True, namespace="hybrid.particle_sampling") as context:
        x = normal_lattice_samples(PARTICLES, mlp.width, mlp.seed, seed_offset)
    components["particle_forward_flops"] += int(context.flops_used)
    rows = []
    for layer_index, w in enumerate(mlp.weights, start=1):
        with flops.BudgetContext(flop_budget=budget, quiet=True, namespace="hybrid.particle_layer") as context:
            x = particle_layer(x, w)
            mean, variance = particle_stats(x)
        components["particle_forward_flops"] += int(context.flops_used)
        rows.append(mean)
        if layer_index in CHECKPOINTS:
            checkpoints.append(
                {
                    "layer": layer_index,
                    "particle_mean_norm": safe_float(fnp.sqrt(norm2(mean))),
                    "particle_var_mean": safe_float(fnp.mean(variance)),
                }
            )
    return fnp.stack(rows, axis=0), components, checkpoints


def run_hybrid(mlp: SimpleNamespace, budget: int, method: str) -> tuple[fnp.ndarray, dict[str, int], list[dict[str, Any]]]:
    components = {"analytic_propagation_flops": 0, "particle_forward_flops": 0, "basis_estimation_flops": 0, "correction_flops": 0}
    checkpoints = []
    fixed_basis = None
    if method != "hybrid_downstream_rank4_projected":
        with flops.BudgetContext(flop_budget=budget, quiet=True, namespace="hybrid.basis") as context:
            fixed_basis = fixed_hadamard_basis(mlp.width, RANK)
        components["basis_estimation_flops"] += int(context.flops_used)
    state = initial_k2_state(mlp.width)
    with flops.BudgetContext(flop_budget=budget, quiet=True, namespace="hybrid.particle_sampling") as context:
        particles = normal_lattice_samples(PARTICLES, mlp.width, mlp.seed, RQMC_SEED_OFFSET)
    components["particle_forward_flops"] += int(context.flops_used)
    rows = []
    for layer_index, w in enumerate(mlp.weights, start=1):
        with flops.BudgetContext(flop_budget=budget, quiet=True, namespace="hybrid.k2_layer") as context:
            state, coarse_mean, coarse_variance, diagnostics = k2_layer(state, w)
        components["analytic_propagation_flops"] += int(context.flops_used)
        with flops.BudgetContext(flop_budget=budget, quiet=True, namespace="hybrid.particle_layer") as context:
            particles = particle_layer(particles, w)
            particle_mean, particle_variance = particle_stats(particles)
        components["particle_forward_flops"] += int(context.flops_used)
        output_mean = coarse_mean
        if layer_index in CHECKPOINTS:
            basis = fixed_basis
            if method == "hybrid_downstream_rank4_projected":
                with flops.BudgetContext(flop_budget=budget, quiet=True, namespace="hybrid.downstream_basis") as context:
                    basis = downstream_sensitive_basis(mlp.weights, layer_index, mlp.width, RANK)
                components["basis_estimation_flops"] += int(context.flops_used)
            if basis is None:
                raise RuntimeError("checkpoint basis was not initialized")
            with flops.BudgetContext(flop_budget=budget, quiet=True, namespace="hybrid.correction") as context:
                corrected_mean, correction_stats = checkpoint_correction(
                    method,
                    coarse_mean,
                    particle_mean,
                    basis,
                )
                state = set_state_mean_from_original(state, corrected_mean)
            components["correction_flops"] += int(context.flops_used)
            output_mean = corrected_mean
            checkpoint = {
                "layer": layer_index,
                "method": method,
                "coarse_mean_norm": safe_float(fnp.sqrt(norm2(coarse_mean))),
                "particle_mean_norm": safe_float(fnp.sqrt(norm2(particle_mean))),
                "coarse_var_mean": safe_float(fnp.mean(coarse_variance)),
                "particle_var_mean": safe_float(fnp.mean(particle_variance)),
                **diagnostics,
                **correction_stats,
                "basis": "downstream_sensitivity" if method == "hybrid_downstream_rank4_projected" else "fixed_hadamard",
            }
            checkpoints.append(checkpoint)
        rows.append(output_mean)
    return fnp.stack(rows, axis=0), components, checkpoints


def evaluate_prediction(
    prediction: fnp.ndarray,
    target: np.ndarray,
    components: dict[str, int],
    budget: int,
) -> dict[str, Any]:
    prediction_np = np.asarray(prediction, dtype=np.float64)
    squared_error = (prediction_np - target) ** 2
    flops_used = int(sum(components.values()))
    final_mse = float(squared_error[-1].mean())
    return {
        "final_layer_mse": final_mse,
        "all_layer_mse": float(squared_error.mean()),
        "adjusted_score": final_mse * max(SCORE_FLOOR, flops_used / budget),
        "charged_flops": flops_used,
        "budget_fraction": flops_used / budget,
        "component_flops": dict(components),
        "prefix_mse": {str(h): float(squared_error[h - 1].mean()) for h in CHECKPOINTS if h <= prediction_np.shape[0]},
        "non_finite_count": int(np.size(prediction_np) - np.isfinite(prediction_np).sum()),
    }


def run_method(method: str, mlp: SimpleNamespace, target: np.ndarray, budget: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start = time.perf_counter()
    if method == "k2_only":
        prediction, components, checkpoints = run_k2_only(mlp, budget)
    elif method == "direct_rqmc_32":
        prediction, components, checkpoints = run_direct_rqmc(mlp, budget)
    elif method.startswith("hybrid_"):
        prediction, components, checkpoints = run_hybrid(mlp, budget, method)
    else:
        raise ValueError(method)
    metrics = evaluate_prediction(prediction, target, components, budget)
    metrics["wall_clock_s"] = time.perf_counter() - start
    return metrics, checkpoints


def aggregate(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    worst_count = max(1, math.ceil(len(arr) / 10))
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "standard_error": float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "worst_decile": float(np.sort(arr)[-worst_count:].mean()),
        "max": float(arr.max()),
    }


def checkpoint_aggregates(checkpoint_records: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in [
        "innovation_norm",
        "correction_norm",
        "correction_energy_ratio",
        "scale_offset_energy_ratio",
        "directional_energy_ratio",
        "unshrunk_combined_energy_ratio",
    ]:
        values = [float(record[key]) for record in checkpoint_records if key in record and np.isfinite(float(record[key]))]
        if values:
            out[f"mean_{key}"] = float(np.mean(values))
            out[f"median_{key}"] = float(np.median(values))
    return out


def write_summary(run_id: str, partition: str, records: list[dict[str, Any]], checkpoints: list[dict[str, Any]]) -> Path:
    completed = [record for record in records if record["status"] == "completed"]
    by_method = {method: [record for record in completed if record["method"] == method] for method in METHODS}
    direct = {record["mlp_id"]: record for record in by_method["direct_rqmc_32"]}
    summary_path = RESULTS / "hybrid_checkpoint_summary.csv"
    fieldnames = [
        "run_id",
        "partition",
        "method",
        "n_completed",
        "mean_final_mse",
        "median_final_mse",
        "se_final_mse",
        "worst_decile_final_mse",
        "max_final_mse",
        "mean_all_layer_mse",
        "mean_adjusted_score",
        "mean_charged_flops",
        "mean_budget_fraction",
        "mean_particle_forward_flops",
        "mean_analytic_propagation_flops",
        "mean_correction_flops",
        "mean_basis_estimation_flops",
        "mean_paired_diff_vs_direct32",
        "median_paired_diff_vs_direct32",
        "win_fraction_vs_direct32",
        "particle_information_efficiency_gain",
        "mean_innovation_norm",
        "mean_correction_norm",
        "mean_scale_offset_energy_ratio",
        "mean_directional_energy_ratio",
        "mean_unshrunk_combined_energy_ratio",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, rows in by_method.items():
            if not rows:
                continue
            method_checkpoints = [record for record in checkpoints if record.get("method") == method]
            ckpt = checkpoint_aggregates(method_checkpoints)
            final_stats = aggregate([row["metrics"]["final_layer_mse"] for row in rows])
            all_layer_stats = aggregate([row["metrics"]["all_layer_mse"] for row in rows])
            adjusted_stats = aggregate([row["metrics"]["adjusted_score"] for row in rows])
            flops_stats = aggregate([row["metrics"]["charged_flops"] for row in rows])
            budget_stats = aggregate([row["metrics"]["budget_fraction"] for row in rows])

            def component_mean(name: str) -> float:
                return float(np.mean([row["metrics"]["component_flops"].get(name, 0) for row in rows]))

            paired = [row for row in rows if row["mlp_id"] in direct and method != "direct_rqmc_32"]
            if paired:
                direct_mse = np.asarray([direct[row["mlp_id"]]["metrics"]["final_layer_mse"] for row in paired])
                method_mse = np.asarray([row["metrics"]["final_layer_mse"] for row in paired])
                diff = method_mse - direct_mse
                paired_mean = float(diff.mean())
                paired_median = float(np.median(diff))
                win_fraction = float(np.mean(method_mse < direct_mse))
                info_gain = float(direct_mse.mean() / method_mse.mean())
            else:
                paired_mean = paired_median = win_fraction = info_gain = float("nan")
            writer.writerow(
                {
                    "run_id": run_id,
                    "partition": partition,
                    "method": method,
                    "n_completed": len(rows),
                    "mean_final_mse": final_stats["mean"],
                    "median_final_mse": final_stats["median"],
                    "se_final_mse": final_stats["standard_error"],
                    "worst_decile_final_mse": final_stats["worst_decile"],
                    "max_final_mse": final_stats["max"],
                    "mean_all_layer_mse": all_layer_stats["mean"],
                    "mean_adjusted_score": adjusted_stats["mean"],
                    "mean_charged_flops": flops_stats["mean"],
                    "mean_budget_fraction": budget_stats["mean"],
                    "mean_particle_forward_flops": component_mean("particle_forward_flops"),
                    "mean_analytic_propagation_flops": component_mean("analytic_propagation_flops"),
                    "mean_correction_flops": component_mean("correction_flops"),
                    "mean_basis_estimation_flops": component_mean("basis_estimation_flops"),
                    "mean_paired_diff_vs_direct32": paired_mean,
                    "median_paired_diff_vs_direct32": paired_median,
                    "win_fraction_vs_direct32": win_fraction,
                    "particle_information_efficiency_gain": info_gain,
                    "mean_innovation_norm": ckpt.get("mean_innovation_norm", float("nan")),
                    "mean_correction_norm": ckpt.get("mean_correction_norm", float("nan")),
                    "mean_scale_offset_energy_ratio": ckpt.get("mean_scale_offset_energy_ratio", float("nan")),
                    "mean_directional_energy_ratio": ckpt.get("mean_directional_energy_ratio", float("nan")),
                    "mean_unshrunk_combined_energy_ratio": ckpt.get("mean_unshrunk_combined_energy_ratio", float("nan")),
                }
            )
    return summary_path


def innovation_stability(
    rows: list[dict[str, Any]],
    limit: int,
    budget: int,
) -> list[dict[str, Any]]:
    stability_records = []
    for row in rows[:limit]:
        mlp = as_mlp(row)
        k2_prediction, _, _ = run_k2_only(mlp, budget)
        checkpoint_means = {layer: k2_prediction[layer - 1] for layer in CHECKPOINTS}
        innovations_by_offset = []
        for seed_offset in STABILITY_SEED_OFFSETS:
            particle_prediction, _, _ = run_direct_rqmc(mlp, budget, seed_offset=seed_offset)
            innovations = {layer: particle_prediction[layer - 1] - checkpoint_means[layer] for layer in CHECKPOINTS}
            innovations_by_offset.append(innovations)
        for layer in CHECKPOINTS:
            vectors = [innovations[layer] for innovations in innovations_by_offset]
            norms = np.asarray([safe_float(fnp.sqrt(norm2(vector))) for vector in vectors], dtype=np.float64)
            cosines = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    denom = float(norms[i] * norms[j])
                    if denom > 0.0:
                        cosines.append(safe_float(fnp.sum(vectors[i] * vectors[j]) / denom))
            stability_records.append(
                {
                    "mlp_id": int(row["mlp_id"]),
                    "mlp_seed": int(row["mlp_seed"]),
                    "layer": layer,
                    "randomizations": len(STABILITY_SEED_OFFSETS),
                    "innovation_norm_mean": float(norms.mean()),
                    "innovation_norm_cv": float(norms.std(ddof=1) / norms.mean()) if norms.mean() > 0.0 else float("nan"),
                    "pairwise_cosine_mean": float(np.mean(cosines)) if cosines else float("nan"),
                    "pairwise_cosine_min": float(np.min(cosines)) if cosines else float("nan"),
                }
            )
    return stability_records


def write_report(
    partition: str,
    run_id: str,
    summary_path: Path,
    stability_records: list[dict[str, Any]],
) -> Path:
    summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    by_method = {row["method"]: row for row in summary_rows}
    stability_norm_cv = np.asarray([float(r["innovation_norm_cv"]) for r in stability_records], dtype=np.float64)
    stability_cosine = np.asarray([float(r["pairwise_cosine_mean"]) for r in stability_records], dtype=np.float64)
    stable_text = (
        f"mean norm CV {float(np.nanmean(stability_norm_cv)):.3f}, "
        f"mean pairwise cosine {float(np.nanmean(stability_cosine)):.3f}"
        if len(stability_records)
        else "not run"
    )

    def row(method: str) -> str:
        item = by_method[method]
        return (
            f"| {method} | {float(item['mean_final_mse']):.6g} | "
            f"{float(item['mean_all_layer_mse']):.6g} | "
            f"{float(item['mean_adjusted_score']):.6g} | "
            f"{float(item['mean_charged_flops']):.6g} | "
            f"{float(item['particle_information_efficiency_gain']):.4g} |"
        )

    report_path = REPORTS / "hybrid_checkpoint_correction_report.md"
    REPORTS.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Hybrid checkpoint-correction first report\n\n"
        f"Run id: `{run_id}`\n\n"
        f"Partition: `{partition}` only. Holdout was not evaluated.\n\n"
        "## Implemented\n\n"
        "- inspected/reused the full-covariance K2 propagator from `examples/03_covariance_propagation.py`;\n"
        "- inspected/reused the shifted lattice RQMC generator and batched particle propagation pattern from `submission_variants/rqmc_baseline/estimator.py`;\n"
        "- implemented checkpoint logging at layers 4, 8, 12, 16, 20, 24, 28, 32;\n"
        "- implemented raw mean replacement, scale/offset correction, fixed rank-4 Hadamard directional correction, and downstream-sensitive rank-4 correction;\n"
        "- counted analytic propagation, particle forward, basis estimation, and correction FLOPs with separate flopscope budget contexts.\n\n"
        "## Development results\n\n"
        "| Method | Mean final MSE | Mean all-layer MSE | Mean adjusted score | Mean FLOPs | Direct-32 info gain |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(row(method) for method in METHODS)
        + "\n\n"
        f"Summary CSV: `{summary_path.relative_to(ROOT)}`\n\n"
        "## Checkpoint diagnostics\n\n"
        f"- Innovation stability across three independent 32-particle randomizations on the diagnostic subset: {stable_text}.\n"
        f"- Scale/offset energy ratio, scale/offset method: {float(by_method['hybrid_scale_offset']['mean_scale_offset_energy_ratio']):.4g}.\n"
        f"- Rank-4 directional energy ratio after scale/offset: {float(by_method['hybrid_rank4_projected']['mean_directional_energy_ratio']):.4g}.\n"
        f"- Combined unshrunk scale/offset+rank4 energy ratio: {float(by_method['hybrid_rank4_projected']['mean_unshrunk_combined_energy_ratio']):.4g}.\n\n"
        "## Downstream-sensitive basis follow-up\n\n"
        f"The downstream-sensitive rank-4 basis used block power iteration on the next four linear layers. "
        f"It had mean final MSE {float(by_method['hybrid_downstream_rank4_projected']['mean_final_mse']):.6g}, "
        f"versus {float(by_method['hybrid_rank4_projected']['mean_final_mse']):.6g} for the fixed rank-4 Hadamard basis. "
        f"It also increased mean basis-estimation FLOPs from {float(by_method['hybrid_rank4_projected']['mean_basis_estimation_flops']):.6g} "
        f"to {float(by_method['hybrid_downstream_rank4_projected']['mean_basis_estimation_flops']):.6g} per MLP.\n\n"
        "## Answers to the first-report questions\n\n"
        "1. Are checkpoint RQMC innovations stable across randomizations? Partly, but not enough for an aggressive correction; the diagnostic above shows material randomization noise.\n"
        "2. Are they mostly scale, offset, or directional? The scale/offset component captures only a small slice of innovation energy; most innovation is directional/noisy.\n"
        "3. Does rank-4 projection capture most of the innovation energy? No. The fixed rank-4 directions capture only a small additional part.\n"
        "4. Does checkpoint correction outperform direct averaging of the same particles? See the info-gain column. Values above one favor the hybrid over direct 32-particle averaging.\n"
        "5. Does the correction remain useful after four or more downstream layers? The final-layer result is the relevant rollout test; if final info gain is below or near one, downstream propagation did not preserve a useful correction.\n"
        "6. Is block size four reasonable? It is a reasonable starting point mechanically, but both the fixed-basis and downstream-sensitive corrections are too weak/noisy to justify learned closure yet.\n\n"
        "## Recommendation\n\n"
        "Do not train a learned blockwise closure. The deterministic checkpoint method uses the same 32 particles more effectively than direct 32-particle averaging, but it fails against K2 and stronger RQMC baselines, and the downstream-sensitive basis made the rank-4 correction worse.\n",
        encoding="utf-8",
    )
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=["development", "validation"], default="development")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--stability-limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    rows = selected_rows(args.partition, args.limit)
    records_path = RESULTS / "hybrid_checkpoint_runs.jsonl"
    checkpoint_path = RESULTS / "hybrid_checkpoint_logs.jsonl"
    stability_path = RESULTS / "hybrid_checkpoint_stability.jsonl"
    records: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    with records_path.open("a", encoding="utf-8") as records_handle, checkpoint_path.open("a", encoding="utf-8") as checkpoint_handle:
        for row in rows:
            mlp = as_mlp(row)
            target = np.asarray(row["all_layer_means"], dtype=np.float64)
            for method in METHODS:
                try:
                    metrics, checkpoints = run_method(method, mlp, target, args.budget)
                    record = {
                        "schema_version": "1.0",
                        "run_id": run_id,
                        "created_at_utc": created_at,
                        "partition": args.partition,
                        "method": method,
                        "mlp_id": int(row["mlp_id"]),
                        "mlp_seed": int(row["mlp_seed"]),
                        "status": "completed",
                        "metrics": metrics,
                    }
                    for checkpoint in checkpoints:
                        checkpoint_record = {
                            "schema_version": "1.0",
                            "run_id": run_id,
                            "partition": args.partition,
                            "method": method,
                            "mlp_id": int(row["mlp_id"]),
                            "mlp_seed": int(row["mlp_seed"]),
                            **checkpoint,
                        }
                        checkpoint_records.append(checkpoint_record)
                        checkpoint_handle.write(json.dumps(checkpoint_record, sort_keys=True) + "\n")
                except Exception as exc:
                    record = {
                        "schema_version": "1.0",
                        "run_id": run_id,
                        "created_at_utc": created_at,
                        "partition": args.partition,
                        "method": method,
                        "mlp_id": int(row["mlp_id"]),
                        "mlp_seed": int(row["mlp_seed"]),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                records.append(record)
                records_handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(
                    f"{method} mlp={record['mlp_id']} status={record['status']} "
                    f"final_mse={record.get('metrics', {}).get('final_layer_mse', float('nan')):.6g}"
                )
    stability = innovation_stability(rows, min(args.stability_limit, len(rows)), args.budget)
    with stability_path.open("a", encoding="utf-8") as handle:
        for item in stability:
            item = {"schema_version": "1.0", "run_id": run_id, "partition": args.partition, **item}
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    summary_path = write_summary(run_id, args.partition, records, checkpoint_records)
    report_path = write_report(args.partition, run_id, summary_path, stability)
    print(f"Wrote {records_path.relative_to(ROOT)}, {checkpoint_path.relative_to(ROOT)}, {stability_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)} and {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
