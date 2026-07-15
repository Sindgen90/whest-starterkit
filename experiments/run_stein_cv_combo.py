"""Evaluate cross-fitted Stein control variates on top of RQMC samples.

This is deliberately narrow: it keeps the corrected-K2 component fixed and
changes only the sample estimator used in the convex combination.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
import whestbench
from run_baselines import (
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

ROOT = Path(__file__).resolve().parents[1]
CORRECTED_K2 = ROOT / "submissions" / "corrected_k2" / "estimator.py"

DEFAULT_N_SAMPLES = 4096
DEFAULT_RQMC_SEED_OFFSET = 8_675_309
DEFAULT_RQMC_MULTIPLIER = 17_797


def normal_lattice_samples(n_samples: int, width: int, seed: int, seed_offset: int, multiplier: int) -> fnp.ndarray:
    rng = fnp.random.default_rng(seed + seed_offset)
    powers = []
    value = 1
    modulus = max(2, n_samples)
    for _ in range(width):
        powers.append(float(value))
        value = (value * multiplier) % modulus
        if value == 0:
            value = 1
    generator = fnp.array(powers, dtype=fnp.float64)
    indices = fnp.arange(n_samples, dtype=fnp.float64) + 0.5
    shift = rng.uniform(0.0, 1.0, size=(width,))
    u = fnp.mod(fnp.outer(indices, generator) / n_samples + shift, 1.0)
    u = fnp.clip(u, 1e-7, 1.0 - 1e-7)
    return flops.stats.norm.ppf(u).astype(fnp.float32)


def stein_features(samples: fnp.ndarray, mlp: Any, feature_family: str, feature_count: int) -> fnp.ndarray:
    width = mlp.width
    count = min(feature_count, width)
    if feature_family == "coords2":
        z = samples[:, :count].astype(fnp.float64)
        return fnp.concatenate([z, z * z - 1.0], axis=1)
    if feature_family == "ridge2":
        basis = mlp.weights[0][:, :count].astype(fnp.float64)
        z = fnp.matmul(samples.astype(fnp.float64), basis)
        norm2 = fnp.sum(basis * basis, axis=0)
        return fnp.concatenate([z, z * z - norm2], axis=1)
    raise ValueError(f"unknown feature family: {feature_family}")


def crossfit_cv_mean(features: fnp.ndarray, values: fnp.ndarray, ridge: float) -> fnp.ndarray:
    n_samples, n_features = features.shape
    split = n_samples // 2
    if split <= n_features:
        raise ValueError(f"Need more split samples than features: split={split}, features={n_features}")

    g0 = features[:split].astype(fnp.float64)
    g1 = features[split:].astype(fnp.float64)
    y0 = values[:split].astype(fnp.float64)
    y1 = values[split:].astype(fnp.float64)

    beta0 = fit_beta(g0, y0, ridge)
    beta1 = fit_beta(g1, y1, ridge)

    estimate1 = fnp.mean(y1, axis=0) - fnp.matmul(fnp.mean(g1, axis=0), beta0)
    estimate0 = fnp.mean(y0, axis=0) - fnp.matmul(fnp.mean(g0, axis=0), beta1)
    return (estimate0 + estimate1) * 0.5


def fit_beta(features: fnp.ndarray, values: fnp.ndarray, ridge: float) -> fnp.ndarray:
    features = features - fnp.mean(features, axis=0)
    values = values - fnp.mean(values, axis=0)
    gram = fnp.matmul(features.T, features)
    scale = float(fnp.trace(gram)) / max(1, features.shape[1])
    penalty = max(1e-12, ridge * scale)
    gram = gram + fnp.eye(features.shape[1]) * penalty
    rhs = fnp.matmul(features.T, values)
    return fnp.linalg.solve(gram, rhs)


class SteinCVRQMC:
    def __init__(
        self,
        *,
        n_samples: int,
        seed_offset: int,
        multiplier: int,
        feature_family: str,
        feature_count: int,
        ridge: float,
        apply_layers: str,
    ) -> None:
        self.n_samples = n_samples
        self.seed_offset = seed_offset
        self.multiplier = multiplier
        self.feature_family = feature_family
        self.feature_count = feature_count
        self.ridge = ridge
        self.apply_layers = apply_layers

    def predict(self, mlp: Any, budget: int) -> fnp.ndarray:
        _ = budget
        x = normal_lattice_samples(self.n_samples, mlp.width, mlp.seed, self.seed_offset, self.multiplier)
        features = stein_features(x, mlp, self.feature_family, self.feature_count)
        rows = []
        for layer_index, w in enumerate(mlp.weights):
            x = fnp.maximum(fnp.matmul(x, w), 0.0)
            if self.apply_layers == "all" or layer_index == mlp.depth - 1:
                rows.append(crossfit_cv_mean(features, x, self.ridge).astype(fnp.float32))
            else:
                rows.append(fnp.mean(x, axis=0))
        return fnp.stack(rows, axis=0)


def predict(estimator: Any, row: dict[str, Any], budget: int) -> tuple[np.ndarray, int, float]:
    mlp = as_mlp(row)
    started = time.perf_counter()
    with flops.BudgetContext(flop_budget=budget, quiet=True) as context:
        prediction = estimator.predict(mlp, budget)
    runtime_s = time.perf_counter() - started
    return np.asarray(prediction, dtype=np.float64), int(context.flops_used), runtime_s


def metrics(prediction: np.ndarray, target: np.ndarray, prefixes: Iterable[int]) -> dict[str, Any]:
    error = (prediction - target) ** 2
    return {
        "final_mse": float(error[-1].mean()),
        "all_layer_mse": float(error.mean()),
        "prefix_mse": {
            str(prefix): float(error[prefix - 1].mean())
            for prefix in prefixes
            if prefix <= prediction.shape[0]
        },
    }


def fit_lambda(rows: list[dict[str, Any]], k2: Any, stein: Any, args: argparse.Namespace) -> dict[str, Any]:
    numerator = 0.0
    denominator = 0.0
    records = []
    for row in rows:
        k2_prediction, _, _ = predict(k2, row, args.budget)
        stein_prediction, _, _ = predict(stein, row, args.budget)
        target = np.asarray(row["all_layer_means"], dtype=np.float64)
        delta = stein_prediction[-1] - k2_prediction[-1]
        residual = k2_prediction[-1] - target[-1]
        numerator += float(np.vdot(delta, residual))
        denominator += float(np.vdot(delta, delta))
        records.append({
            "mlp_id": int(row["mlp_id"]),
            "mlp_name": str(row["mlp_name"]),
            "corrected_k2_final_mse": metrics(k2_prediction, target, args.prefixes)["final_mse"],
            "stein_cv_final_mse": metrics(stein_prediction, target, args.prefixes)["final_mse"],
        })
    raw_lambda = -numerator / denominator if denominator else 0.0
    return {
        "objective": "final-layer least squares",
        "formula": "argmin_lambda ||k2 + lambda * (stein_cv - k2) - target||^2",
        "raw_lambda": raw_lambda,
        "lambda": min(1.0, max(0.0, raw_lambda)),
        "n_mlps": len(rows),
        "selector": f"mlp_id % 2 == {args.calibration_parity}",
        "records": records,
    }


def evaluate_row(row: dict[str, Any], k2: Any, stein: Any, lambda_: float, args: argparse.Namespace) -> dict[str, Any]:
    target = np.asarray(row["all_layer_means"], dtype=np.float64)
    k2_prediction, k2_flops, k2_runtime = predict(k2, row, args.budget)
    stein_prediction, stein_flops, stein_runtime = predict(stein, row, args.budget)

    combo_started = time.perf_counter()
    mlp = as_mlp(row)
    with flops.BudgetContext(flop_budget=args.budget, quiet=True) as context:
        combo_k2 = k2.predict(mlp, args.budget)
        combo_stein = stein.predict(mlp, args.budget)
        combo_prediction = fnp.add(fnp.multiply(combo_k2, 1.0 - lambda_), fnp.multiply(combo_stein, lambda_))
    combo_runtime = time.perf_counter() - combo_started
    combo_prediction_np = np.asarray(combo_prediction, dtype=np.float64)

    k2_metrics = metrics(k2_prediction, target, args.prefixes)
    stein_metrics = metrics(stein_prediction, target, args.prefixes)
    combo_metrics = metrics(combo_prediction_np, target, args.prefixes)
    combo_flops = int(context.flops_used)
    return {
        "mlp_id": int(row["mlp_id"]),
        "mlp_name": str(row["mlp_name"]),
        "mlp_seed": int(row["mlp_seed"]),
        "status": "completed",
        "metrics": {
            "corrected_k2": k2_metrics,
            "stein_cv_rqmc": stein_metrics,
            "combo": combo_metrics,
        },
        "paired_vs_corrected_k2": {
            "combo_final_mse_delta": combo_metrics["final_mse"] - k2_metrics["final_mse"],
            "combo_final_mse_ratio": combo_metrics["final_mse"] / k2_metrics["final_mse"],
            "stein_cv_final_mse_delta": stein_metrics["final_mse"] - k2_metrics["final_mse"],
            "stein_cv_final_mse_ratio": stein_metrics["final_mse"] / k2_metrics["final_mse"],
        },
        "compute": {
            "corrected_k2_flops": k2_flops,
            "stein_cv_flops": stein_flops,
            "combo_flops": combo_flops,
            "flop_budget": args.budget,
            "corrected_k2_runtime_s": k2_runtime,
            "stein_cv_runtime_s": stein_runtime,
            "combo_runtime_s": combo_runtime,
            "combo_flop_fraction": combo_flops / args.budget,
            "combo_adjusted_final_score": combo_metrics["final_mse"] * max(SCORE_FLOOR, combo_flops / args.budget),
        },
    }


def summarize(records: list[dict[str, Any]], prefixes: Iterable[int]) -> dict[str, Any]:
    done = [record for record in records if record["status"] == "completed"]
    combo_wins = [record["paired_vs_corrected_k2"]["combo_final_mse_delta"] < 0.0 for record in done]
    stein_wins = [record["paired_vs_corrected_k2"]["stein_cv_final_mse_delta"] < 0.0 for record in done]
    summary = {
        "n_requested": len(records),
        "n_completed": len(done),
        "n_failed": len(records) - len(done),
        "corrected_k2_final_mse": aggregate(r["metrics"]["corrected_k2"]["final_mse"] for r in done),
        "stein_cv_final_mse": aggregate(r["metrics"]["stein_cv_rqmc"]["final_mse"] for r in done),
        "combo_final_mse": aggregate(r["metrics"]["combo"]["final_mse"] for r in done),
        "corrected_k2_all_layer_mse": aggregate(r["metrics"]["corrected_k2"]["all_layer_mse"] for r in done),
        "stein_cv_all_layer_mse": aggregate(r["metrics"]["stein_cv_rqmc"]["all_layer_mse"] for r in done),
        "combo_all_layer_mse": aggregate(r["metrics"]["combo"]["all_layer_mse"] for r in done),
        "combo_adjusted_final_score": aggregate(r["compute"]["combo_adjusted_final_score"] for r in done),
        "combo_flops": aggregate(r["compute"]["combo_flops"] for r in done),
        "combo_runtime_s": aggregate(r["compute"]["combo_runtime_s"] for r in done),
        "paired_vs_corrected_k2": {
            "combo_win_fraction": sum(combo_wins) / len(combo_wins) if combo_wins else None,
            "stein_cv_win_fraction": sum(stein_wins) / len(stein_wins) if stein_wins else None,
            "combo_final_mse_delta": aggregate(r["paired_vs_corrected_k2"]["combo_final_mse_delta"] for r in done),
            "combo_final_mse_ratio": aggregate(r["paired_vs_corrected_k2"]["combo_final_mse_ratio"] for r in done),
            "stein_cv_final_mse_delta": aggregate(r["paired_vs_corrected_k2"]["stein_cv_final_mse_delta"] for r in done),
            "stein_cv_final_mse_ratio": aggregate(r["paired_vs_corrected_k2"]["stein_cv_final_mse_ratio"] for r in done),
        },
        "prefix_mse": {},
    }
    for prefix in prefixes:
        summary["prefix_mse"][str(prefix)] = {
            "corrected_k2": aggregate(r["metrics"]["corrected_k2"]["prefix_mse"][str(prefix)] for r in done),
            "stein_cv_rqmc": aggregate(r["metrics"]["stein_cv_rqmc"]["prefix_mse"][str(prefix)] for r in done),
            "combo": aggregate(r["metrics"]["combo"]["prefix_mse"][str(prefix)] for r in done),
        }
    return summary


def run(args: argparse.Namespace) -> Path:
    args.prefixes = sorted(set(args.prefixes))
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="mini"))
    if args.limit_mlps is not None:
        rows = rows[: args.limit_mlps]
    calibration_rows = [row for row in rows if int(row["mlp_id"]) % 2 == args.calibration_parity]
    evaluation_rows = [row for row in rows if int(row["mlp_id"]) % 2 != args.calibration_parity]
    k2 = load_estimator(CORRECTED_K2)()
    stein = SteinCVRQMC(
        n_samples=args.n_samples,
        seed_offset=args.rqmc_seed_offset,
        multiplier=args.rqmc_multiplier,
        feature_family=args.feature_family,
        feature_count=args.feature_count,
        ridge=args.ridge,
        apply_layers=args.apply_layers,
    )
    calibration = fit_lambda(calibration_rows, k2, stein, args)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(evaluation_rows, start=1):
        try:
            records.append(evaluate_row(row, k2, stein, calibration["lambda"], args))
            print(f"{index}/{len(evaluation_rows)} {row['mlp_name']} completed", flush=True)
        except Exception as exc:
            failures.append({
                "mlp_id": int(row["mlp_id"]),
                "mlp_name": str(row["mlp_name"]),
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            })
            records.append({
                "mlp_id": int(row["mlp_id"]),
                "mlp_name": str(row["mlp_name"]),
                "mlp_seed": int(row["mlp_seed"]),
                "status": "failed",
            })
            print(f"{index}/{len(evaluation_rows)} {row['mlp_name']} failed", flush=True)

    result = {
        "schema_version": "1.0",
        "experiment": "stein_cv_rqmc_corrected_k2_combo",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if not failures else "completed_with_failures",
        "configuration": {
            "phase": "phase1",
            "dataset": DATASET,
            "dataset_revision": REVISION,
            "dataset_split": "mini",
            "flop_budget": args.budget,
            "score_floor": SCORE_FLOOR,
            "depth_prefixes": args.prefixes,
            "calibration_parity": args.calibration_parity,
            "limit_mlps": args.limit_mlps,
            "n_samples": args.n_samples,
            "rqmc_seed_offset": args.rqmc_seed_offset,
            "rqmc_multiplier": args.rqmc_multiplier,
            "feature_family": args.feature_family,
            "feature_count": args.feature_count,
            "ridge": args.ridge,
            "apply_layers": args.apply_layers,
        },
        "calibration": calibration,
        "source": {
            "corrected_k2_path": str(CORRECTED_K2.relative_to(ROOT)),
            "corrected_k2_sha256": hashlib.sha256(CORRECTED_K2.read_bytes()).hexdigest(),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--porcelain")),
        },
        "summary": summarize(records, args.prefixes),
        "per_mlp": records,
        "failures": failures,
    }
    RESULTS.mkdir(exist_ok=True)
    suffix = "mini" if args.limit_mlps is None else f"mini_limit{args.limit_mlps}"
    suffix = (
        f"{suffix}_stein_{args.feature_family}{args.feature_count}"
        f"_ridge{args.ridge:g}_{args.apply_layers}"
        f"_rqmc{args.n_samples}_offset{args.rqmc_seed_offset}_mult{args.rqmc_multiplier}"
    )
    output = RESULTS / f"{suffix}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--prefixes", type=int, nargs="+", default=list(PREFIXES))
    parser.add_argument("--calibration-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--limit-mlps", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--rqmc-seed-offset", type=int, default=DEFAULT_RQMC_SEED_OFFSET)
    parser.add_argument("--rqmc-multiplier", type=int, default=DEFAULT_RQMC_MULTIPLIER)
    parser.add_argument("--feature-family", choices=("coords2", "ridge2"), default="ridge2")
    parser.add_argument("--feature-count", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--apply-layers", choices=("final", "all"), default="final")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_mlps is not None and args.limit_mlps <= 1:
        raise SystemExit("--limit-mlps must be greater than 1")
    if args.n_samples <= 0:
        raise SystemExit("--n-samples must be positive")
    if args.rqmc_multiplier <= 0 or args.rqmc_multiplier % 2 == 0:
        raise SystemExit("--rqmc-multiplier must be positive and odd")
    if args.feature_count <= 0:
        raise SystemExit("--feature-count must be positive")
    if args.ridge < 0:
        raise SystemExit("--ridge must be non-negative")
    output = run(args)
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
