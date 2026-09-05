"""Residual-aware Fourier scout for 4096-node lattice rules.

This is offline research code, not a submission estimator.  The raw
Fourier/Walsh spectra are invariant to subtracting a constant estimator such
as corrected K2, except at the zero bin.  This script therefore studies the
useful residual question:

* does the 4096 lattice sample mean move corrected K2 toward the final target?
* do nonzero Fourier features of the sampled final outputs predict that error?
* do alternate lattice multipliers improve the corrected-K2 convex combo on a
  held-out half of the 50-MLP Mini scout slice?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
import whestbench

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from run_baselines import (  # noqa: E402
    BUDGET,
    DATASET,
    PREFIXES,
    RESULTS,
    REVISION,
    SCORE_FLOOR,
    aggregate,
    as_mlp,
    git_value,
    load_estimator,
)
from run_grid_combo import (  # noqa: E402
    DEFAULT_RQMC_MULTIPLIER,
    DEFAULT_RQMC_SEED_OFFSET,
    lattice_samples,
    metrics,
)

CORRECTED_K2 = ROOT / "submissions" / "corrected_k2" / "estimator.py"
DEFAULT_CANDIDATE_MULTIPLIERS = [
    DEFAULT_RQMC_MULTIPLIER,
    50_933,
    117_917,
    1_048_573,
    2_891_336_453,
]
EPS = 1e-12


def _rademacher_projection_matrix(width: int, n_random: int, seed: int) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    columns = [np.ones(width, dtype=np.float64) / math.sqrt(width)]
    labels = ["ones"]
    selected_neurons = [0, width // 4, width // 2, (3 * width) // 4, width - 1]
    for neuron in selected_neurons:
        vector = np.zeros(width, dtype=np.float64)
        vector[neuron] = 1.0
        columns.append(vector)
        labels.append(f"neuron_{neuron}")
    for index in range(n_random):
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=width)
        columns.append(signs / math.sqrt(width))
        labels.append(f"rademacher_{index}")
    return np.stack(columns, axis=1), labels


def _band_edges(n: int) -> list[tuple[int, int, str]]:
    edges = []
    start = 1
    while start < n:
        end = min(n, start * 2)
        edges.append((start, end, f"{start}-{end - 1}"))
        start = end
    return edges


def _fft_power_by_projection(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    normalized = centered / np.maximum(scale, EPS)
    coeffs = np.fft.fft(normalized, axis=0) / normalized.shape[0]
    power = np.abs(coeffs) ** 2
    power[0, :] = 0.0
    total = power.sum(axis=0, keepdims=True)
    return power / np.maximum(total, EPS)


def _effective_fraction(power: np.ndarray) -> np.ndarray:
    nonzero = power[1:, :]
    entropy_terms = np.where(nonzero > 0.0, nonzero * np.log(np.maximum(nonzero, EPS)), 0.0)
    entropy = -entropy_terms.sum(axis=0)
    return np.exp(entropy) / max(1, len(power) - 1)


def _spectral_summary(profile: np.ndarray, top_k: int) -> dict[str, Any]:
    nonzero = profile[1:] if len(profile) > 1 else profile
    positive = nonzero[nonzero > 0.0]
    if len(positive) == 0:
        effective_bins = 0.0
    else:
        entropy = float(-(positive * np.log(positive)).sum())
        effective_bins = float(math.exp(entropy))
    order = np.argsort(profile)[::-1]
    top = [
        {"index": int(index), "energy": float(profile[index])}
        for index in order[:top_k]
        if index != 0 and profile[index] > 0.0
    ]
    return {
        "top_bins": top,
        "top_8_energy": float(np.sort(profile)[-8:].sum()) if len(profile) >= 8 else float(profile.sum()),
        "top_32_energy": float(np.sort(profile)[-32:].sum()) if len(profile) >= 32 else float(profile.sum()),
        "effective_bins": effective_bins,
        "effective_fraction": effective_bins / max(1, len(nonzero)),
        "max_bin_energy": float(profile[order[0]]) if len(order) else 0.0,
        "dyadic_bands": [
            {"band": label, "energy": float(profile[start:end].sum())}
            for start, end, label in _band_edges(len(profile))
        ],
    }


def _pearson(x_values: Iterable[float], y_values: Iterable[float]) -> float | None:
    x = np.asarray(list(x_values), dtype=np.float64)
    y = np.asarray(list(y_values), dtype=np.float64)
    if len(x) < 3 or len(y) != len(x):
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= EPS:
        return None
    return float(np.dot(x, y) / denom)


def _prediction_with_flops(estimator: Any, row: dict[str, Any], budget: int) -> tuple[np.ndarray, int, float]:
    mlp = as_mlp(row)
    started = time.perf_counter()
    with flops.BudgetContext(flop_budget=budget, quiet=True) as context:
        prediction = estimator.predict(mlp, budget)
    return np.asarray(prediction, dtype=np.float64), int(context.flops_used), time.perf_counter() - started


def _lattice_prediction_and_final_outputs(
    row: dict[str, Any],
    n_samples: int,
    seed_offset: int,
    multiplier: int,
    budget: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    mlp = as_mlp(row)
    started = time.perf_counter()
    with flops.BudgetContext(flop_budget=budget, quiet=True) as context:
        x = lattice_samples(n_samples, mlp.width, mlp.seed, seed_offset, multiplier)
        rows = []
        for w in mlp.weights:
            x = fnp.maximum(fnp.matmul(x, w), 0.0)
            rows.append(fnp.mean(x, axis=0))
        prediction = fnp.stack(rows, axis=0)
    return (
        np.asarray(prediction, dtype=np.float64),
        np.asarray(x, dtype=np.float64),
        int(context.flops_used),
        time.perf_counter() - started,
    )


def _prepare_multipliers(
    multipliers: list[int],
    n_samples: int,
    random_count: int,
    random_seed: int,
) -> list[int]:
    if n_samples <= 1:
        raise ValueError("sample count must be greater than one")
    selected: list[int] = []
    seen_moduli: set[int] = set()

    def add(value: int) -> None:
        residue = value % n_samples
        if residue == 0 or math.gcd(residue, n_samples) != 1:
            return
        if residue not in seen_moduli:
            seen_moduli.add(residue)
            selected.append(int(value))

    for multiplier in multipliers:
        add(multiplier)

    rng = np.random.default_rng(random_seed)
    attempts = 0
    while len(selected) < len(multipliers) + random_count and attempts < 1000 * max(1, random_count):
        attempts += 1
        add(int(rng.integers(1, n_samples)) | 1)
    return selected


def _common_row_data(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    estimator = load_estimator(CORRECTED_K2)()
    common: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        try:
            prediction, flops_used, runtime_s = _prediction_with_flops(estimator, row, args.budget)
            target = np.asarray(row["all_layer_means"], dtype=np.float64)
            common[int(row["mlp_id"])] = {
                "target": target,
                "k2_prediction": prediction,
                "k2_flops": flops_used,
                "k2_runtime_s": runtime_s,
                "k2_metrics": metrics(prediction, target, args.prefixes),
            }
            print(f"k2 {index}/{len(rows)} {row['mlp_name']} done", flush=True)
        except Exception as exc:
            failures.append({
                "mlp_id": int(row["mlp_id"]),
                "mlp_name": str(row["mlp_name"]),
                "stage": "corrected_k2",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"k2 {index}/{len(rows)} {row['mlp_name']} failed", flush=True)
    return common, failures


def _spectral_observations(
    final_outputs: np.ndarray,
    projections: np.ndarray,
    projection_labels: list[str],
    target_final: np.ndarray,
    k2_final: np.ndarray,
    grid_final: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    projected = final_outputs @ projections
    power = _fft_power_by_projection(projected)
    band_slices = _band_edges(power.shape[0])
    band_values = {
        label: power[start:end, :].sum(axis=0)
        for start, end, label in band_slices
    }
    top8 = np.sort(power, axis=0)[-8:, :].sum(axis=0) if power.shape[0] >= 8 else power.sum(axis=0)
    top32 = np.sort(power, axis=0)[-32:, :].sum(axis=0) if power.shape[0] >= 32 else power.sum(axis=0)
    max_bin = power.max(axis=0)
    effective_fraction = _effective_fraction(power)

    target_proj = target_final @ projections
    k2_proj = k2_final @ projections
    grid_proj = grid_final @ projections
    k2_error = k2_proj - target_proj
    delta = grid_proj - k2_proj

    observations = []
    for index, label in enumerate(projection_labels):
        features = {
            "top_8_energy": float(top8[index]),
            "top_32_energy": float(top32[index]),
            "max_bin_energy": float(max_bin[index]),
            "effective_fraction": float(effective_fraction[index]),
        }
        for band_label, values in band_values.items():
            features[f"band_{band_label}"] = float(values[index])
        observations.append({
            "projection": label,
            "features": features,
            "responses": {
                "k2_error": float(k2_error[index]),
                "grid_error": float(k2_error[index] + delta[index]),
                "grid_delta": float(delta[index]),
                "target_minus_k2": float(-k2_error[index]),
                "alignment": float(delta[index] * (-k2_error[index])),
            },
        })
    return observations, power.mean(axis=1)


def _response_value(observation: dict[str, Any], response_name: str, lambda_: float) -> float:
    responses = observation["responses"]
    k2_error = float(responses["k2_error"])
    delta = float(responses["grid_delta"])
    if response_name == "abs_k2_error":
        return abs(k2_error)
    if response_name == "abs_grid_error":
        return abs(float(responses["grid_error"]))
    if response_name == "abs_delta":
        return abs(delta)
    if response_name == "signed_alignment":
        return float(responses["alignment"])
    if response_name == "combo_improvement":
        combo_error = k2_error + lambda_ * delta
        return k2_error * k2_error - combo_error * combo_error
    if response_name == "abs_combo_error":
        return abs(k2_error + lambda_ * delta)
    raise ValueError(f"unknown response: {response_name}")


def _feature_correlations(
    observations: list[dict[str, Any]],
    lambda_: float,
    top_k: int,
) -> dict[str, Any]:
    if not observations:
        return {"n_observations": 0, "top_abs_correlations": []}
    feature_names = sorted(observations[0]["features"])
    response_names = [
        "abs_k2_error",
        "abs_grid_error",
        "abs_delta",
        "signed_alignment",
        "combo_improvement",
        "abs_combo_error",
    ]
    correlations = []
    for feature_name in feature_names:
        x = [observation["features"][feature_name] for observation in observations]
        for response_name in response_names:
            y = [_response_value(observation, response_name, lambda_) for observation in observations]
            corr = _pearson(x, y)
            if corr is not None:
                correlations.append({
                    "feature": feature_name,
                    "response": response_name,
                    "pearson": corr,
                    "abs_pearson": abs(corr),
                })
    correlations.sort(key=lambda record: record["abs_pearson"], reverse=True)
    return {
        "n_observations": len(observations),
        "top_abs_correlations": correlations[:top_k],
    }


def _fit_lambda(row_records: list[dict[str, Any]], calibration_parity: int) -> dict[str, Any]:
    calibration = [record for record in row_records if record["mlp_id"] % 2 == calibration_parity]
    numerator = sum(record["residual_fit"]["delta_dot_target_minus_k2"] for record in calibration)
    denominator = sum(record["residual_fit"]["delta_norm2"] for record in calibration)
    raw_lambda = numerator / denominator if denominator > 0.0 else 0.0
    lambda_ = min(1.0, max(0.0, raw_lambda))
    return {
        "objective": "final-layer least squares",
        "formula": "argmin_lambda ||k2 + lambda * (grid - k2) - target||^2",
        "raw_lambda": float(raw_lambda),
        "lambda": float(lambda_),
        "n_mlps": len(calibration),
        "selector": f"mlp_id % 2 == {calibration_parity}",
    }


def _summarize_records(records: list[dict[str, Any]], prefixes: Iterable[int]) -> dict[str, Any]:
    done = [record for record in records if record["status"] == "completed"]
    combo_wins = [record["paired_vs_corrected_k2"]["combo_final_mse_delta"] < 0.0 for record in done]
    grid_wins = [record["paired_vs_corrected_k2"]["grid_final_mse_delta"] < 0.0 for record in done]
    summary = {
        "n_requested": len(records),
        "n_completed": len(done),
        "n_failed": len(records) - len(done),
        "corrected_k2_final_mse": aggregate(r["metrics"]["corrected_k2"]["final_mse"] for r in done),
        "grid_final_mse": aggregate(r["metrics"]["grid"]["final_mse"] for r in done),
        "combo_final_mse": aggregate(r["metrics"]["combo"]["final_mse"] for r in done),
        "corrected_k2_all_layer_mse": aggregate(r["metrics"]["corrected_k2"]["all_layer_mse"] for r in done),
        "grid_all_layer_mse": aggregate(r["metrics"]["grid"]["all_layer_mse"] for r in done),
        "combo_all_layer_mse": aggregate(r["metrics"]["combo"]["all_layer_mse"] for r in done),
        "combo_adjusted_final_score": aggregate(r["compute"]["combo_adjusted_final_score"] for r in done),
        "grid_runtime_s": aggregate(r["compute"]["grid_runtime_s"] for r in done),
        "paired_vs_corrected_k2": {
            "combo_win_fraction": sum(combo_wins) / len(combo_wins) if combo_wins else None,
            "grid_win_fraction": sum(grid_wins) / len(grid_wins) if grid_wins else None,
            "combo_final_mse_delta": aggregate(r["paired_vs_corrected_k2"]["combo_final_mse_delta"] for r in done),
            "combo_final_mse_ratio": aggregate(r["paired_vs_corrected_k2"]["combo_final_mse_ratio"] for r in done),
            "grid_final_mse_delta": aggregate(r["paired_vs_corrected_k2"]["grid_final_mse_delta"] for r in done),
            "grid_final_mse_ratio": aggregate(r["paired_vs_corrected_k2"]["grid_final_mse_ratio"] for r in done),
        },
        "residual_fit": {
            "alignment_cosine": aggregate(r["residual_fit"]["alignment_cosine"] for r in done),
            "delta_norm_over_target_minus_k2_norm": aggregate(
                r["residual_fit"]["delta_norm_over_target_minus_k2_norm"] for r in done
            ),
        },
        "prefix_mse": {},
    }
    for prefix in prefixes:
        key = str(prefix)
        summary["prefix_mse"][key] = {
            "corrected_k2": aggregate(
                r["metrics"]["corrected_k2"]["prefix_mse"][key]
                for r in done
                if key in r["metrics"]["corrected_k2"]["prefix_mse"]
            ),
            "grid": aggregate(
                r["metrics"]["grid"]["prefix_mse"][key]
                for r in done
                if key in r["metrics"]["grid"]["prefix_mse"]
            ),
            "combo": aggregate(
                r["metrics"]["combo"]["prefix_mse"][key]
                for r in done
                if key in r["metrics"]["combo"]["prefix_mse"]
            ),
        }
    return summary


def _evaluate_candidate(
    rows: list[dict[str, Any]],
    common: dict[int, dict[str, Any]],
    projections: np.ndarray,
    projection_labels: list[str],
    multiplier: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate_started = time.perf_counter()
    row_payloads: list[dict[str, Any]] = []
    observations_by_subset = {"all": [], "calibration": [], "evaluation": []}
    mean_profile_accum: np.ndarray | None = None
    mean_profile_count = 0
    failures: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        mlp_id = int(row["mlp_id"])
        if mlp_id not in common:
            continue
        try:
            grid_prediction, final_outputs, grid_flops, grid_runtime_s = _lattice_prediction_and_final_outputs(
                row=row,
                n_samples=args.n_samples,
                seed_offset=args.seed_offset,
                multiplier=multiplier,
                budget=args.budget,
            )
            target = common[mlp_id]["target"]
            k2_prediction = common[mlp_id]["k2_prediction"]
            delta = grid_prediction[-1] - k2_prediction[-1]
            target_minus_k2 = target[-1] - k2_prediction[-1]
            delta_norm = float(np.linalg.norm(delta))
            residual_norm = float(np.linalg.norm(target_minus_k2))
            denominator = delta_norm * residual_norm
            alignment_cosine = (
                float(np.dot(delta, target_minus_k2) / denominator)
                if denominator > EPS
                else 0.0
            )
            observations, mean_profile = _spectral_observations(
                final_outputs=final_outputs,
                projections=projections,
                projection_labels=projection_labels,
                target_final=target[-1],
                k2_final=k2_prediction[-1],
                grid_final=grid_prediction[-1],
            )
            if mean_profile_accum is None:
                mean_profile_accum = mean_profile.copy()
            else:
                mean_profile_accum += mean_profile
            mean_profile_count += 1
            subset = "calibration" if mlp_id % 2 == args.calibration_parity else "evaluation"
            observations_by_subset["all"].extend(observations)
            observations_by_subset[subset].extend(observations)
            row_payloads.append({
                "mlp_id": mlp_id,
                "mlp_name": str(row["mlp_name"]),
                "mlp_seed": int(row["mlp_seed"]),
                "status": "completed",
                "subset": subset,
                "grid_prediction": grid_prediction,
                "metrics": {
                    "corrected_k2": common[mlp_id]["k2_metrics"],
                    "grid": metrics(grid_prediction, target, args.prefixes),
                },
                "residual_fit": {
                    "delta_dot_target_minus_k2": float(np.dot(delta, target_minus_k2)),
                    "delta_norm2": float(np.dot(delta, delta)),
                    "target_minus_k2_norm2": float(np.dot(target_minus_k2, target_minus_k2)),
                    "alignment_cosine": alignment_cosine,
                    "delta_norm_over_target_minus_k2_norm": delta_norm / max(residual_norm, EPS),
                },
                "compute": {
                    "corrected_k2_flops": common[mlp_id]["k2_flops"],
                    "grid_flops": grid_flops,
                    "corrected_k2_runtime_s": common[mlp_id]["k2_runtime_s"],
                    "grid_runtime_s": grid_runtime_s,
                },
            })
            print(
                f"lattice mult={multiplier} {index}/{len(rows)} {row['mlp_name']} done",
                flush=True,
            )
        except Exception as exc:
            failures.append({
                "mlp_id": mlp_id,
                "mlp_name": str(row["mlp_name"]),
                "stage": f"lattice_{multiplier}",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(
                f"lattice mult={multiplier} {index}/{len(rows)} {row['mlp_name']} failed",
                flush=True,
            )

    fit = _fit_lambda(row_payloads, args.calibration_parity)
    lambda_ = fit["lambda"]
    completed_records = []
    for record in row_payloads:
        mlp_id = record["mlp_id"]
        target = common[mlp_id]["target"]
        k2_prediction = common[mlp_id]["k2_prediction"]
        grid_prediction = record.pop("grid_prediction")
        combo_prediction = k2_prediction + lambda_ * (grid_prediction - k2_prediction)
        combo_metrics = metrics(combo_prediction, target, args.prefixes)
        record["metrics"]["combo"] = combo_metrics
        record["paired_vs_corrected_k2"] = {
            "combo_final_mse_delta": combo_metrics["final_mse"] - record["metrics"]["corrected_k2"]["final_mse"],
            "combo_final_mse_ratio": combo_metrics["final_mse"] / record["metrics"]["corrected_k2"]["final_mse"],
            "grid_final_mse_delta": record["metrics"]["grid"]["final_mse"]
            - record["metrics"]["corrected_k2"]["final_mse"],
            "grid_final_mse_ratio": record["metrics"]["grid"]["final_mse"]
            / record["metrics"]["corrected_k2"]["final_mse"],
        }
        combo_flops = record["compute"]["corrected_k2_flops"] + record["compute"]["grid_flops"]
        record["compute"]["combo_flops_estimate"] = combo_flops
        record["compute"]["combo_flop_fraction_estimate"] = combo_flops / args.budget
        record["compute"]["combo_adjusted_final_score"] = (
            combo_metrics["final_mse"] * max(SCORE_FLOOR, combo_flops / args.budget)
        )
        completed_records.append(record)

    calibration_records = [record for record in completed_records if record["subset"] == "calibration"]
    evaluation_records = [record for record in completed_records if record["subset"] == "evaluation"]
    mean_profile = (
        mean_profile_accum / mean_profile_count
        if mean_profile_accum is not None and mean_profile_count
        else np.zeros(args.n_samples, dtype=np.float64)
    )
    return {
        "multiplier": int(multiplier),
        "multiplier_mod_n_samples": int(multiplier % args.n_samples),
        "status": "completed" if not failures else "completed_with_failures",
        "runtime_s": time.perf_counter() - candidate_started,
        "fit": fit,
        "spectral_summary": _spectral_summary(mean_profile, args.top_bins),
        "correlations": {
            subset: _feature_correlations(observations, lambda_, args.top_correlations)
            for subset, observations in observations_by_subset.items()
        },
        "summary": {
            "all": _summarize_records(completed_records, args.prefixes),
            "calibration": _summarize_records(calibration_records, args.prefixes),
            "evaluation": _summarize_records(evaluation_records, args.prefixes),
        },
        "per_mlp": completed_records,
        "failures": failures,
    }


def _leaderboard(candidates: list[dict[str, Any]], subset: str, metric: str) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        metric_summary = candidate["summary"][subset][metric]
        value = metric_summary["mean"]
        rows.append({
            "multiplier": candidate["multiplier"],
            "multiplier_mod_n_samples": candidate["multiplier_mod_n_samples"],
            metric: value,
            "lambda": candidate["fit"]["lambda"],
            "grid_final_mse": candidate["summary"][subset]["grid_final_mse"]["mean"],
            "combo_win_fraction": candidate["summary"][subset]["paired_vs_corrected_k2"]["combo_win_fraction"],
        })
    return sorted(rows, key=lambda row: float("inf") if row[metric] is None else row[metric])


def run(args: argparse.Namespace) -> Path:
    args.prefixes = sorted(set(args.prefixes))
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="mini"))
    if args.limit_mlps is not None:
        rows = rows[: args.limit_mlps]
    if not rows:
        raise SystemExit("no rows selected")
    width = int(np.asarray(rows[0]["weights"]).shape[1])
    projections, projection_labels = _rademacher_projection_matrix(
        width, args.random_projections, args.projection_seed
    )
    multipliers = _prepare_multipliers(
        args.candidate_multipliers,
        args.n_samples,
        args.random_multipliers,
        args.random_seed,
    )
    print(f"evaluating multipliers: {multipliers}", flush=True)
    common, common_failures = _common_row_data(rows, args)
    candidate_results = [
        _evaluate_candidate(rows, common, projections, projection_labels, multiplier, args)
        for multiplier in multipliers
    ]
    result = {
        "schema_version": "1.0",
        "experiment": "residual_spectral_lattice_scout",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "configuration": {
            "phase": "phase1",
            "dataset": DATASET,
            "dataset_revision": REVISION,
            "dataset_split": "mini",
            "limit_mlps": args.limit_mlps,
            "flop_budget": args.budget,
            "score_floor": SCORE_FLOOR,
            "depth_prefixes": args.prefixes,
            "calibration_parity": args.calibration_parity,
            "n_samples": args.n_samples,
            "seed_offset": args.seed_offset,
            "candidate_multipliers_requested": args.candidate_multipliers,
            "candidate_multipliers_evaluated": multipliers,
            "random_multipliers": args.random_multipliers,
            "random_seed": args.random_seed,
            "projection_labels": projection_labels,
            "residual_definition": "target_final - corrected_k2_final",
        },
        "source": {
            "corrected_k2_path": str(CORRECTED_K2.relative_to(ROOT)),
            "corrected_k2_sha256": hashlib.sha256(CORRECTED_K2.read_bytes()).hexdigest(),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--porcelain")),
            "python": sys.version,
            "whestbench": getattr(whestbench, "__version__", None),
        },
        "leaderboard": {
            "calibration_combo_final_mse": _leaderboard(candidate_results, "calibration", "combo_final_mse"),
            "evaluation_combo_final_mse": _leaderboard(candidate_results, "evaluation", "combo_final_mse"),
            "evaluation_grid_final_mse": _leaderboard(candidate_results, "evaluation", "grid_final_mse"),
        },
        "candidates": candidate_results,
        "common_failures": common_failures,
        "decision_hints": [
            "Use calibration_combo_final_mse only for candidate selection.",
            "Use evaluation_combo_final_mse as the first 50-MLP held-out check.",
            "Strong residual spectra should show stable feature/error correlations, not just concentrated raw energy.",
        ],
    }
    RESULTS.mkdir(exist_ok=True)
    suffix = f"limit{args.limit_mlps or 'all'}_n{args.n_samples}_c{len(multipliers)}"
    output = RESULTS / f"residual_spectral_scout_{suffix}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-mlps", type=int, default=50)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--prefixes", type=int, nargs="+", default=list(PREFIXES))
    parser.add_argument("--calibration-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--seed-offset", type=int, default=DEFAULT_RQMC_SEED_OFFSET)
    parser.add_argument("--candidate-multipliers", type=int, nargs="*", default=DEFAULT_CANDIDATE_MULTIPLIERS)
    parser.add_argument("--random-multipliers", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20260802)
    parser.add_argument("--random-projections", type=int, default=8)
    parser.add_argument("--projection-seed", type=int, default=12345)
    parser.add_argument("--top-bins", type=int, default=16)
    parser.add_argument("--top-correlations", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_mlps is not None and args.limit_mlps <= 0:
        raise SystemExit("--limit-mlps must be positive")
    if args.n_samples <= 1:
        raise SystemExit("--n-samples must be greater than one")
    output = run(args)
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
