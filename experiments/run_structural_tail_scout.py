"""Scout dead/on/kink classification and selective two-layer tail evaluation.

This is offline experiment code. It deliberately uses NumPy for diagnostics;
submission-facing estimators must keep using only the grader-supported imports.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
import whestbench
from run_baselines import BUDGET, DATASET, RESULTS, REVISION, SCORE_FLOOR, as_mlp
from run_grid_combo import (
    DEFAULT_SOBOL_DIRECTION_FILE,
    SOBOL_BITS,
    SOBOL_SCALE,
    load_sobol_direction_rows,
    sobol_direction_vectors,
)
from whestbench.budget import LAMBDA_FLOPS_PER_SECOND

ROOT = Path(__file__).resolve().parents[1]
K2_COEFFICIENT = 0.9918858550287327
_COV_RESCALE_THRESHOLD = 1e100


def aggregate(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def sobol_base_integers(
    n_samples: int,
    width: int,
    direction_file: Path,
    backend: str,
) -> fnp.ndarray | np.ndarray:
    if n_samples < 2 or n_samples % 2:
        raise ValueError("antithetic Sobol sample count must be positive and even")
    if n_samples // 2 > (1 << SOBOL_BITS):
        raise ValueError(f"Sobol half-sample count exceeds {SOBOL_BITS}-bit implementation")
    rows = load_sobol_direction_rows(direction_file, width)
    direction_values = sobol_direction_vectors(width, SOBOL_BITS, rows)
    half = n_samples // 2
    if backend == "numpy":
        directions = np.asarray(direction_values, dtype=np.uint32)
        current = np.zeros(width, dtype=np.uint32)
        integers = np.empty((half, width), dtype=np.uint32)
        for index in range(half):
            if index:
                bit = (index & -index).bit_length()
                current ^= directions[:, bit - 1]
            integers[index] = current
        return integers
    elif backend == "fnp":
        directions = fnp.array(direction_values, dtype=fnp.uint32)
        indices = fnp.arange(half, dtype=fnp.uint32)
        gray = fnp.bitwise_xor(indices, fnp.right_shift(indices, 1))
        integers = fnp.zeros((half, width), dtype=fnp.uint32)
        for bit in range(max(1, (half - 1).bit_length())):
            mask = fnp.bitwise_and(fnp.right_shift(gray, bit), 1)
            contribution = mask[:, None] * directions[:, bit][None, :]
            integers = fnp.bitwise_xor(integers, contribution)
        return integers
    raise ValueError(f"unknown Sobol backend: {backend}")


def antithetic_sobol_samples(
    base_integers: fnp.ndarray | np.ndarray,
    seed: int,
    seed_offset: int,
    backend: str,
) -> fnp.ndarray:
    width = base_integers.shape[1]
    scramble_values = [
        random.Random(seed + seed_offset + 104729 * dim).getrandbits(SOBOL_BITS)
        for dim in range(width)
    ]
    if backend == "numpy":
        scramble = np.asarray(scramble_values, dtype=np.uint32)
        shifted = np.bitwise_xor(base_integers, scramble[None, :])
        uniforms = np.clip(
            (shifted.astype(np.float64) + 0.5) / SOBOL_SCALE, 1e-7, 1.0 - 1e-7
        )
        uniforms_fnp = fnp.array(uniforms, dtype=fnp.float64)
    elif backend == "fnp":
        scramble = fnp.array(scramble_values, dtype=fnp.uint32)
        shifted = fnp.bitwise_xor(base_integers, scramble[None, :])
        uniforms_fnp = fnp.clip(
            (shifted.astype(fnp.float64) + 0.5) / SOBOL_SCALE, 1e-7, 1.0 - 1e-7
        )
    else:
        raise ValueError(f"unknown Sobol backend: {backend}")
    base = flops.stats.norm.ppf(uniforms_fnp).astype(fnp.float32)
    return fnp.concatenate((base, -base), axis=0)


def gaussian_classifier(mlp: Any) -> tuple[fnp.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Return corrected-K2 means and Gaussian preactivation z-scores for the tail."""
    width = mlp.width
    mu = fnp.zeros(width)
    cov = fnp.eye(width)
    log_scale = 0.0
    rows = []
    tail_alphas: list[np.ndarray] = []

    for layer_index, w in enumerate(mlp.weights):
        cov_diag = fnp.diag(cov)
        max_var = float(fnp.max(cov_diag))
        if max_var > _COV_RESCALE_THRESHOLD:
            scale = float(fnp.sqrt(max_var))
            mu = mu / scale
            cov = cov / (scale * scale)
            log_scale += float(fnp.log(scale))

        mu_pre = w.T @ mu
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
        var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
        sigma_pre = fnp.sqrt(var_pre)
        alpha = mu_pre / sigma_pre
        if layer_index >= mlp.depth - 2:
            tail_alphas.append(np.asarray(alpha, dtype=np.float64))

        phi_alpha = flops.stats.norm.pdf(alpha)
        Phi_alpha = flops.stats.norm.cdf(alpha)
        mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha
        ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
        var_post = fnp.maximum(ez2 - mu * mu, 0.0)

        sigma64 = fnp.asarray(sigma_pre, dtype=fnp.float64)
        gain64 = fnp.where(sigma64 > 1e-12, fnp.asarray(Phi_alpha, dtype=fnp.float64), 0.0)
        gain = fnp.array(gain64.astype(fnp.float32))
        cov = fnp.multiply(fnp.outer(gain, gain), cov_pre)
        fnp.fill_diagonal(cov, var_post)
        rows.append(mu * float(fnp.exp(log_scale)))

    if len(tail_alphas) != 2:
        raise ValueError("structural tail scout requires an MLP with at least two layers")
    return fnp.multiply(fnp.stack(rows, axis=0), K2_COEFFICIENT), (
        tail_alphas[0],
        tail_alphas[1],
    )


def classes(alpha: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = np.flatnonzero(alpha >= threshold)
    dead = np.flatnonzero(alpha <= -threshold)
    uncertain = np.flatnonzero(np.abs(alpha) < threshold)
    return active, dead, uncertain


def estimated_matmul_flops(m: int, k: int, n: int) -> int:
    if m <= 0 or k <= 0 or n <= 0:
        return 0
    return m * n * (2 * k - 1)


def estimated_candidate_flops(
    n_samples: int,
    width: int,
    depth: int,
    k2_flops: int,
    active1: int,
    uncertain1: int,
    uncertain2: int,
) -> int:
    # Prefix estimate excludes point generation and elementwise operations. It
    # is only used by --classification-only to select thresholds cheaply.
    prefix = (depth - 2) * estimated_matmul_flops(n_samples, width, width)
    tail1 = estimated_matmul_flops(n_samples, width, uncertain1)
    collapse = estimated_matmul_flops(width, active1, uncertain2)
    active_path = estimated_matmul_flops(n_samples, width, uncertain2)
    uncertain_path = estimated_matmul_flops(n_samples, uncertain1, uncertain2)
    return k2_flops + prefix + tail1 + collapse + active_path + uncertain_path


def sampled_prefix(
    mlp: Any,
    sobol_base: fnp.ndarray | np.ndarray,
    fixed_samples: fnp.ndarray | None,
    seed_offset: int,
    sobol_backend: str,
) -> tuple[fnp.ndarray, list[fnp.ndarray]]:
    x = (
        fixed_samples
        if fixed_samples is not None
        else antithetic_sobol_samples(
            sobol_base,
            mlp.seed,
            seed_offset,
            sobol_backend,
        )
    )
    rows = []
    for w in mlp.weights[:-2]:
        x = fnp.maximum(x @ w, 0.0)
        rows.append(fnp.mean(x, axis=0))
    return x, rows


def dense_tail(
    x: fnp.ndarray,
    prefix_rows: list[fnp.ndarray],
    w1: fnp.ndarray,
    w2: fnp.ndarray,
) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray]:
    z1 = x @ w1
    h1 = fnp.maximum(z1, 0.0)
    z2 = h1 @ w2
    h2 = fnp.maximum(z2, 0.0)
    prediction = fnp.stack(prefix_rows + [fnp.mean(h1, axis=0), fnp.mean(h2, axis=0)], axis=0)
    return prediction, z1, z2


def structural_tail(
    x: fnp.ndarray,
    prefix_rows: list[fnp.ndarray],
    w1: fnp.ndarray,
    w2: fnp.ndarray,
    alpha1: np.ndarray,
    alpha2: np.ndarray,
    threshold: float,
) -> tuple[fnp.ndarray, dict[str, int]]:
    active1, dead1, uncertain1 = classes(alpha1, threshold)
    active2, dead2, uncertain2 = classes(alpha2, threshold)

    mean_x = fnp.mean(x, axis=0)
    mean1 = np.zeros(w1.shape[1], dtype=np.float64)
    h1_uncertain = None
    if active1.size:
        mean1[active1] = np.asarray(mean_x @ w1[:, active1], dtype=np.float64)
    if uncertain1.size:
        h1_uncertain = fnp.maximum(x @ w1[:, uncertain1], 0.0)
        mean1[uncertain1] = np.asarray(fnp.mean(h1_uncertain, axis=0), dtype=np.float64)
    mean1_fnp = fnp.array(mean1, dtype=fnp.float32)

    mean2 = np.zeros(w2.shape[1], dtype=np.float64)
    if active2.size:
        mean2[active2] = np.asarray(mean1_fnp @ w2[:, active2], dtype=np.float64)
    if uncertain2.size:
        z2_uncertain = None
        if active1.size:
            collapsed = w1[:, active1] @ w2[np.ix_(active1, uncertain2)]
            z2_uncertain = x @ collapsed
        if uncertain1.size:
            uncertain_contribution = h1_uncertain @ w2[np.ix_(uncertain1, uncertain2)]
            z2_uncertain = (
                uncertain_contribution
                if z2_uncertain is None
                else z2_uncertain + uncertain_contribution
            )
        if z2_uncertain is None:
            mean2[uncertain2] = 0.0
        else:
            mean2[uncertain2] = np.asarray(
                fnp.mean(fnp.maximum(z2_uncertain, 0.0), axis=0), dtype=np.float64
            )

    prediction = fnp.stack(
        prefix_rows + [mean1_fnp, fnp.array(mean2, dtype=fnp.float32)], axis=0
    )
    return prediction, {
        "layer_1_active": int(active1.size),
        "layer_1_dead": int(dead1.size),
        "layer_1_uncertain": int(uncertain1.size),
        "layer_2_active": int(active2.size),
        "layer_2_dead": int(dead2.size),
        "layer_2_uncertain": int(uncertain2.size),
    }


def sign_diagnostics(
    z: fnp.ndarray,
    alpha: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    active, dead, _ = classes(alpha, threshold)
    result: dict[str, float | None] = {
        "active_wrong_sign_fraction": None,
        "active_worst_neuron_wrong_sign_fraction": None,
        "dead_wrong_sign_fraction": None,
        "dead_worst_neuron_wrong_sign_fraction": None,
    }
    if active.size:
        rates = np.asarray(fnp.mean(z[:, active] <= 0.0, axis=0), dtype=np.float64)
        result["active_wrong_sign_fraction"] = float(rates.mean())
        result["active_worst_neuron_wrong_sign_fraction"] = float(rates.max())
    if dead.size:
        rates = np.asarray(fnp.mean(z[:, dead] > 0.0, axis=0), dtype=np.float64)
        result["dead_wrong_sign_fraction"] = float(rates.mean())
        result["dead_worst_neuron_wrong_sign_fraction"] = float(rates.max())
    return result


def mse(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    error = (prediction - target) ** 2
    return float(error[-1].mean()), float(error.mean())


def fit_lambda(records: list[dict[str, Any]], threshold_key: str, parity: int) -> float:
    numerator = 0.0
    denominator = 0.0
    for record in records:
        if record["mlp_id"] % 2 != parity:
            continue
        k2 = record["_k2"][-1]
        structural = record["_structural"][threshold_key][-1]
        target = record["_target"][-1]
        delta = structural - k2
        numerator += float(np.vdot(delta, k2 - target))
        denominator += float(np.vdot(delta, delta))
    raw = -numerator / denominator if denominator else 0.0
    return min(1.0, max(0.0, raw))


def summarize_full(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    evaluation = [record for record in records if record["mlp_id"] % 2 != args.calibration_parity]
    dense_final = []
    dense_all = []
    dense_scores = []
    for record in evaluation:
        final, all_layer = mse(record["_dense"], record["_target"])
        dense_final.append(final)
        dense_all.append(all_layer)
        dense_scores.append(
            final * max(SCORE_FLOOR, record["dense_effective_compute"] / args.budget)
        )

    threshold_summaries = {}
    for threshold in args.thresholds:
        key = str(threshold)
        lambda_ = fit_lambda(records, key, args.calibration_parity)
        direct_final = []
        direct_all = []
        direct_scores = []
        combo_final = []
        combo_all = []
        combo_scores = []
        structural_vs_dense_final = []
        structural_vs_dense_all = []
        flops_used = []
        effective_compute = []
        residual_wall_time = []
        budget_exceeded = []
        counts = []
        sign_rows = []
        for record in evaluation:
            structural = record["_structural"][key]
            k2 = record["_k2"]
            target = record["_target"]
            candidate_flops = record["candidate_flops"][key]
            candidate_effective = record["candidate_effective_compute"][key]
            direct_mse, direct_all_mse = mse(structural, target)
            difference_final, difference_all = mse(structural, record["_dense"])
            combo = k2 + lambda_ * (structural - k2)
            combo_mse, combo_all_mse = mse(combo, target)
            multiplier = max(SCORE_FLOOR, candidate_effective / args.budget)
            direct_final.append(direct_mse)
            direct_all.append(direct_all_mse)
            direct_scores.append(direct_mse * multiplier)
            combo_final.append(combo_mse)
            combo_all.append(combo_all_mse)
            combo_scores.append(combo_mse * multiplier)
            structural_vs_dense_final.append(difference_final)
            structural_vs_dense_all.append(difference_all)
            flops_used.append(candidate_flops)
            effective_compute.append(candidate_effective)
            residual_wall_time.append(record["candidate_residual_wall_time_s"][key])
            budget_exceeded.append(candidate_effective > args.budget)
            counts.append(record["class_counts"][key])
            sign_rows.append(record["sign_diagnostics"][key])

        threshold_summaries[key] = {
            "lambda": lambda_,
            "direct_final_mse": aggregate(direct_final),
            "direct_all_layer_mse": aggregate(direct_all),
            "direct_adjusted_score": aggregate(direct_scores),
            "combo_final_mse": aggregate(combo_final),
            "combo_all_layer_mse": aggregate(combo_all),
            "combo_adjusted_score": aggregate(combo_scores),
            "structural_vs_dense_final_mse": aggregate(structural_vs_dense_final),
            "structural_vs_dense_all_layer_mse": aggregate(structural_vs_dense_all),
            "candidate_flops": aggregate(flops_used),
            "candidate_effective_compute": aggregate(effective_compute),
            "candidate_residual_wall_time_s": aggregate(residual_wall_time),
            "budget_exceeded_fraction": sum(budget_exceeded) / len(budget_exceeded),
            "class_counts": {
                field: aggregate(row[field] for row in counts) for field in counts[0]
            },
            "sign_diagnostics": {
                f"layer_{layer}_{field}": aggregate(
                    row[f"layer_{layer}"][field]
                    for row in sign_rows
                    if row[f"layer_{layer}"][field] is not None
                )
                for layer in (1, 2)
                for field in sign_rows[0][f"layer_{layer}"]
            },
        }

    return {
        "evaluation_mlps": len(evaluation),
        "dense_final_mse": aggregate(dense_final),
        "dense_all_layer_mse": aggregate(dense_all),
        "dense_adjusted_score": aggregate(dense_scores),
        "dense_flops": aggregate(record["dense_flops"] for record in evaluation),
        "dense_effective_compute": aggregate(
            record["dense_effective_compute"] for record in evaluation
        ),
        "dense_residual_wall_time_s": aggregate(
            record["dense_residual_wall_time_s"] for record in evaluation
        ),
        "thresholds": threshold_summaries,
    }


def classification_only(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    records = []
    for index, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        with flops.BudgetContext(flop_budget=10**15, quiet=True) as context:
            _, alphas = gaussian_classifier(mlp)
        threshold_rows = {}
        for threshold in args.thresholds:
            active1, dead1, uncertain1 = classes(alphas[0], threshold)
            active2, dead2, uncertain2 = classes(alphas[1], threshold)
            estimate = estimated_candidate_flops(
                args.n_samples,
                mlp.width,
                mlp.depth,
                int(context.flops_used),
                int(active1.size),
                int(uncertain1.size),
                int(uncertain2.size),
            )
            threshold_rows[str(threshold)] = {
                "layer_1_active": int(active1.size),
                "layer_1_dead": int(dead1.size),
                "layer_1_uncertain": int(uncertain1.size),
                "layer_2_active": int(active2.size),
                "layer_2_dead": int(dead2.size),
                "layer_2_uncertain": int(uncertain2.size),
                "estimated_candidate_flops": estimate,
                "estimated_budget_fraction": estimate / args.budget,
            }
        records.append({"mlp_id": int(row["mlp_id"]), "thresholds": threshold_rows})
        print(f"{index}/{len(rows)} {row['mlp_name']} classified", flush=True)

    return {
        "records": records,
        "summary": {
            str(threshold): {
                field: aggregate(record["thresholds"][str(threshold)][field] for record in records)
                for field in next(iter(records))["thresholds"][str(threshold)]
            }
            for threshold in args.thresholds
        },
    }


def run_full(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    records = []
    first_mlp = as_mlp(rows[0])
    setup_started = time.perf_counter()
    sobol_base = sobol_base_integers(
        args.n_samples,
        first_mlp.width,
        args.sobol_direction_file,
        args.sobol_backend,
    )
    fixed_samples = (
        antithetic_sobol_samples(
            sobol_base,
            args.fixed_pointset_seed,
            args.seed_offset,
            args.sobol_backend,
        )
        if args.shift_mode == "fixed_setup"
        else None
    )
    setup_runtime_s = time.perf_counter() - setup_started
    for index, row in enumerate(rows, start=1):
        started = time.perf_counter()
        mlp = as_mlp(row)
        target = np.asarray(row["all_layer_means"], dtype=np.float64)

        with flops.BudgetContext(flop_budget=10**15, quiet=True) as k2_context:
            k2_prediction, alphas = gaussian_classifier(mlp)
        with flops.BudgetContext(flop_budget=10**15, quiet=True) as prefix_context:
            x, prefix_rows = sampled_prefix(
                mlp,
                sobol_base,
                fixed_samples,
                args.seed_offset,
                args.sobol_backend,
            )
        with flops.BudgetContext(flop_budget=10**15, quiet=True) as dense_context:
            dense_prediction, z1, z2 = dense_tail(
                x, prefix_rows, mlp.weights[-2], mlp.weights[-1]
            )

        structural_predictions = {}
        candidate_flops = {}
        candidate_effective_compute = {}
        candidate_residual_wall_time_s = {}
        class_counts = {}
        sign_rows = {}
        for threshold in args.thresholds:
            key = str(threshold)
            with flops.BudgetContext(flop_budget=10**15, quiet=True) as tail_context:
                structural, counts = structural_tail(
                    x,
                    prefix_rows,
                    mlp.weights[-2],
                    mlp.weights[-1],
                    alphas[0],
                    alphas[1],
                    threshold,
                )
            structural_predictions[key] = np.asarray(structural, dtype=np.float64)
            candidate_flops[key] = (
                int(k2_context.flops_used)
                + int(prefix_context.flops_used)
                + int(tail_context.flops_used)
            )
            residual_s = sum(
                float(context.residual_wall_time_s or 0.0)
                for context in (k2_context, prefix_context, tail_context)
            )
            candidate_residual_wall_time_s[key] = residual_s
            candidate_effective_compute[key] = (
                candidate_flops[key] + LAMBDA_FLOPS_PER_SECOND * residual_s
            )
            class_counts[key] = counts
            sign_rows[key] = {
                "layer_1": sign_diagnostics(z1, alphas[0], threshold),
                "layer_2": sign_diagnostics(z2, alphas[1], threshold),
            }

        dense_residual_s = sum(
            float(context.residual_wall_time_s or 0.0)
            for context in (prefix_context, dense_context)
        )
        dense_flops = int(prefix_context.flops_used) + int(dense_context.flops_used)
        records.append({
            "mlp_id": int(row["mlp_id"]),
            "mlp_name": str(row["mlp_name"]),
            "dense_flops": dense_flops,
            "dense_residual_wall_time_s": dense_residual_s,
            "dense_effective_compute": dense_flops
            + LAMBDA_FLOPS_PER_SECOND * dense_residual_s,
            "candidate_flops": candidate_flops,
            "candidate_residual_wall_time_s": candidate_residual_wall_time_s,
            "candidate_effective_compute": candidate_effective_compute,
            "component_residual_wall_time_s": {
                "k2": float(k2_context.residual_wall_time_s or 0.0),
                "prefix": float(prefix_context.residual_wall_time_s or 0.0),
                "dense_tail": float(dense_context.residual_wall_time_s or 0.0),
                "structural_tail": {
                    key: candidate_residual_wall_time_s[key]
                    - float(k2_context.residual_wall_time_s or 0.0)
                    - float(prefix_context.residual_wall_time_s or 0.0)
                    for key in candidate_residual_wall_time_s
                },
            },
            "class_counts": class_counts,
            "sign_diagnostics": sign_rows,
            "runtime_s": time.perf_counter() - started,
            "_target": target,
            "_k2": np.asarray(k2_prediction, dtype=np.float64),
            "_dense": np.asarray(dense_prediction, dtype=np.float64),
            "_structural": structural_predictions,
        })
        print(f"{index}/{len(rows)} {row['mlp_name']} completed", flush=True)

    public_records = [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in records
    ]
    return {
        "setup": {
            "sobol_base_shape": list(sobol_base.shape),
            "sobol_base_bytes": int(sobol_base.size * sobol_base.itemsize),
            "fixed_samples_bytes": (
                int(fixed_samples.size * fixed_samples.itemsize)
                if fixed_samples is not None
                else 0
            ),
            "runtime_s": setup_runtime_s,
        },
        "records": public_records,
        "summary": summarize_full(records, args),
    }


def run(args: argparse.Namespace) -> Path:
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="mini"))
    if args.limit_mlps is not None:
        rows = rows[: args.limit_mlps]
    payload = (
        classification_only(rows, args)
        if args.classification_only
        else run_full(rows, args)
    )
    result = {
        "schema_version": "1.0",
        "experiment": "dead_on_kink_structural_tail_scout",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset": DATASET,
            "revision": REVISION,
            "split": "mini",
            "limit_mlps": args.limit_mlps,
            "n_samples": args.n_samples,
            "thresholds": args.thresholds,
            "seed_offset": args.seed_offset,
            "classification_only": args.classification_only,
            "calibration_parity": args.calibration_parity,
            "budget": args.budget,
            "score_floor": SCORE_FLOOR,
            "sobol_direction_file": str(args.sobol_direction_file),
            "sobol_backend": args.sobol_backend,
            "shift_mode": args.shift_mode,
            "fixed_pointset_seed": args.fixed_pointset_seed,
            "classification": "corrected-K2 Gaussian preactivation z-score",
            "tail": "sample through depth-2; omit dead; collapse active; sample uncertain",
        },
        **payload,
    }
    RESULTS.mkdir(exist_ok=True)
    mode = "classification" if args.classification_only else "full"
    limit = "all" if args.limit_mlps is None else str(args.limit_mlps)
    output = RESULTS / f"structural_tail_{mode}_limit{limit}_n{args.n_samples}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-mlps", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=65536)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[3.0, 4.0, 5.0])
    parser.add_argument("--seed-offset", type=int, default=8_675_309)
    parser.add_argument("--sobol-direction-file", type=Path, default=DEFAULT_SOBOL_DIRECTION_FILE)
    parser.add_argument("--sobol-backend", choices=("fnp", "numpy"), default="fnp")
    parser.add_argument("--shift-mode", choices=("fixed_setup", "per_mlp"), default="fixed_setup")
    parser.add_argument("--fixed-pointset-seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--calibration-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--classification-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.thresholds = sorted(set(args.thresholds))
    if args.limit_mlps is not None and args.limit_mlps <= 1:
        raise SystemExit("--limit-mlps must be greater than one")
    if args.n_samples < 2 or args.n_samples % 2:
        raise SystemExit("--n-samples must be positive and even")
    if any(not math.isfinite(value) or value <= 0.0 for value in args.thresholds):
        raise SystemExit("all thresholds must be finite and positive")
    if not args.sobol_direction_file.exists():
        raise SystemExit(f"Sobol direction file not found: {args.sobol_direction_file}")
    output = run(args)
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
