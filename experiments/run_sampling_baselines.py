"""Run paired sampling baselines against frozen corrected K2 on the Mini split."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
DEFAULT_FRACTIONS = (0.001, 0.0025, 0.005, 0.01, 0.02)
CALIBRATION_SAMPLES = 64
RQMC_SEED_OFFSET = 1_000_003


def normal_lattice_samples(n_samples: int, width: int, seed: int) -> fnp.ndarray:
    """Randomly shifted rank-1 lattice mapped through the normal inverse CDF."""
    rng = fnp.random.default_rng(seed + RQMC_SEED_OFFSET)
    powers = np.empty(width, dtype=np.float64)
    value = 1
    modulus = max(2, n_samples)
    for index in range(width):
        powers[index] = value
        value = (value * 17797) % modulus
        if value == 0:
            value = 1
    generator = fnp.array(powers)
    indices = fnp.arange(n_samples, dtype=fnp.float64) + 0.5
    shift = rng.uniform(0.0, 1.0, size=(width,))
    u = fnp.mod(fnp.outer(indices, generator) / n_samples + shift, 1.0)
    u = fnp.clip(u, 1e-7, 1.0 - 1e-7)
    return flops.stats.norm.ppf(u).astype(fnp.float32)


def initial_samples(mlp: Any, n_samples: int, seed: int, method: str) -> fnp.ndarray:
    if method == "ordinary_mc":
        rng = fnp.random.default_rng(seed)
        return fnp.array(rng.standard_normal((n_samples, mlp.width)).astype(fnp.float32))
    elif method == "rqmc_lattice":
        return normal_lattice_samples(n_samples, mlp.width, seed)
    raise ValueError(f"unknown sampling method: {method}")


def sampling_prediction(mlp: Any, n_samples: int, seed: int, method: str) -> fnp.ndarray:
    x = initial_samples(mlp, n_samples, seed, method)
    rows = []
    for w in mlp.weights:
        x = fnp.maximum(fnp.matmul(x, w), 0.0)
        rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0)


def sampling_predictions_by_count(
    mlp: Any,
    sample_counts: Iterable[int],
    seed: int,
    method: str,
) -> dict[int, fnp.ndarray]:
    counts = sorted(set(sample_counts))
    max_count = counts[-1]
    x = initial_samples(mlp, max_count, seed, method)
    rows_by_count = {count: [] for count in counts}
    for w in mlp.weights:
        x = fnp.maximum(fnp.matmul(x, w), 0.0)
        for count in counts:
            rows_by_count[count].append(fnp.mean(x[:count], axis=0))
    return {count: fnp.stack(rows, axis=0) for count, rows in rows_by_count.items()}


def squared_metrics(prediction: np.ndarray, target: np.ndarray, prefixes: Iterable[int]) -> dict[str, Any]:
    squared_error = (prediction - target) ** 2
    return {
        "final_mse": float(squared_error[-1].mean()),
        "all_layer_mse": float(squared_error.mean()),
        "prefix_mse": {
            str(prefix): float(squared_error[prefix - 1].mean())
            for prefix in prefixes
            if prefix <= prediction.shape[0]
        },
    }


def evaluate_corrected_k2(estimator: Any, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mlp = as_mlp(row)
    target = np.asarray(row["all_layer_means"], dtype=np.float64)
    started = time.perf_counter()
    with flops.BudgetContext(flop_budget=args.budget, quiet=True) as context:
        prediction = estimator.predict(mlp, args.budget)
    runtime_s = time.perf_counter() - started
    prediction_np = np.asarray(prediction, dtype=np.float64)
    metrics = squared_metrics(prediction_np, target, args.prefixes)
    flops_used = int(context.flops_used)
    return {
        "mlp_id": int(row["mlp_id"]),
        "mlp_name": str(row["mlp_name"]),
        "mlp_seed": int(row["mlp_seed"]),
        "status": "completed",
        "metrics": metrics,
        "compute": {
            "flops_used": flops_used,
            "flop_budget": args.budget,
            "runtime_s": runtime_s,
            "flop_fraction": flops_used / args.budget,
            "adjusted_final_score": metrics["final_mse"] * max(SCORE_FLOOR, flops_used / args.budget),
        },
    }


def evaluate_sampling(
    row: dict[str, Any],
    args: argparse.Namespace,
    fraction: float,
    n_samples: int,
    prediction: fnp.ndarray,
    flops_used: int,
    runtime_s: float,
    joint_flops_used: int,
    joint_runtime_s: float,
    corrected: dict[str, Any],
) -> dict[str, Any]:
    target = np.asarray(row["all_layer_means"], dtype=np.float64)
    prediction_np = np.asarray(prediction, dtype=np.float64)
    metrics = squared_metrics(prediction_np, target, args.prefixes)
    corrected_metrics = corrected["metrics"]
    return {
        "mlp_id": int(row["mlp_id"]),
        "mlp_name": str(row["mlp_name"]),
        "mlp_seed": int(row["mlp_seed"]),
        "method": args.method,
        "target_budget_fraction": fraction,
        "n_samples": n_samples,
        "status": "completed",
        "metrics": metrics,
        "paired_vs_corrected_k2": {
            "final_mse_delta": metrics["final_mse"] - corrected_metrics["final_mse"],
            "final_mse_ratio": metrics["final_mse"] / corrected_metrics["final_mse"],
            "prefix_mse_delta": {
                str(prefix): metrics["prefix_mse"][str(prefix)] - corrected_metrics["prefix_mse"][str(prefix)]
                for prefix in args.prefixes
                if str(prefix) in metrics["prefix_mse"] and str(prefix) in corrected_metrics["prefix_mse"]
            },
        },
        "compute": {
            "flops_used": flops_used,
            "joint_flops_used": joint_flops_used,
            "flop_budget": args.budget,
            "runtime_s": runtime_s,
            "joint_runtime_s": joint_runtime_s,
            "flop_fraction": flops_used / args.budget,
            "adjusted_final_score": metrics["final_mse"] * max(SCORE_FLOOR, flops_used / args.budget),
        },
    }


def calibrate_sample_counts(row: dict[str, Any], args: argparse.Namespace) -> dict[str, int]:
    mlp = as_mlp(row)
    with flops.BudgetContext(flop_budget=args.budget, quiet=True) as context:
        _ = sampling_prediction(mlp, CALIBRATION_SAMPLES, int(row["mlp_seed"]), "ordinary_mc")
    flops_per_sample = context.flops_used / CALIBRATION_SAMPLES
    return {
        f"{fraction:g}": max(1, math.floor(args.budget * fraction / flops_per_sample))
        for fraction in args.fractions
    }


def calibrate_method_flops(row: dict[str, Any], args: argparse.Namespace, sample_counts: Iterable[int]) -> dict[int, int]:
    mlp = as_mlp(row)
    seed = int(row["mlp_seed"])
    flops_by_count = {}
    for count in sorted(set(sample_counts)):
        with flops.BudgetContext(flop_budget=args.budget, quiet=True) as context:
            _ = sampling_prediction(mlp, count, seed, args.method)
        flops_by_count[count] = int(context.flops_used)
    return flops_by_count


def summarize_records(records: list[dict[str, Any]], prefixes: Iterable[int]) -> dict[str, Any]:
    done = [record for record in records if record["status"] == "completed"]
    wins = [record["paired_vs_corrected_k2"]["final_mse_delta"] < 0.0 for record in done]
    return {
        "n_requested": len(records),
        "n_completed": len(done),
        "n_failed": len(records) - len(done),
        "n_samples": int(done[0]["n_samples"]) if done else None,
        "target_budget_fraction": float(done[0]["target_budget_fraction"]) if done else None,
        "actual_flop_fraction": aggregate(r["compute"]["flop_fraction"] for r in done),
        "final_mse": aggregate(r["metrics"]["final_mse"] for r in done),
        "all_layer_mse": aggregate(r["metrics"]["all_layer_mse"] for r in done),
        "adjusted_final_score": aggregate(r["compute"]["adjusted_final_score"] for r in done),
        "flops": aggregate(r["compute"]["flops_used"] for r in done),
        "runtime_s": aggregate(r["compute"]["runtime_s"] for r in done),
        "paired_vs_corrected_k2": {
            "win_fraction": sum(wins) / len(wins) if wins else None,
            "final_mse_delta": aggregate(r["paired_vs_corrected_k2"]["final_mse_delta"] for r in done),
            "final_mse_ratio": aggregate(r["paired_vs_corrected_k2"]["final_mse_ratio"] for r in done),
        },
        "prefix_mse": {
            str(prefix): aggregate(r["metrics"]["prefix_mse"][str(prefix)] for r in done)
            for prefix in prefixes
        },
    }


def summarize_corrected(records: list[dict[str, Any]], prefixes: Iterable[int]) -> dict[str, Any]:
    done = [record for record in records if record["status"] == "completed"]
    return {
        "n_requested": len(records),
        "n_completed": len(done),
        "n_failed": len(records) - len(done),
        "final_mse": aggregate(r["metrics"]["final_mse"] for r in done),
        "all_layer_mse": aggregate(r["metrics"]["all_layer_mse"] for r in done),
        "adjusted_final_score": aggregate(r["compute"]["adjusted_final_score"] for r in done),
        "flops": aggregate(r["compute"]["flops_used"] for r in done),
        "runtime_s": aggregate(r["compute"]["runtime_s"] for r in done),
        "prefix_mse": {
            str(prefix): aggregate(r["metrics"]["prefix_mse"][str(prefix)] for r in done)
            for prefix in prefixes
        },
    }


def grouped_summary(records: list[dict[str, Any]], prefixes: Iterable[int]) -> dict[str, Any]:
    summaries = {}
    for key in sorted({record["target_budget_fraction"] for record in records}):
        group = [record for record in records if record["target_budget_fraction"] == key]
        summaries[f"{key:g}"] = summarize_records(group, prefixes)
    return summaries


def run(args: argparse.Namespace) -> Path:
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="mini"))
    if args.limit_mlps is not None:
        rows = rows[: args.limit_mlps]
    args.prefixes = sorted(set(args.prefixes))
    args.fractions = tuple(sorted(set(args.fractions)))
    sample_counts = calibrate_sample_counts(rows[0], args)
    method_flops = calibrate_method_flops(rows[0], args, sample_counts.values())
    corrected_estimator = load_estimator(CORRECTED_K2)()
    corrected_records: list[dict[str, Any]] = []
    sampling_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        try:
            corrected = evaluate_corrected_k2(corrected_estimator, row, args)
            corrected_records.append(corrected)
            counts = [sample_counts[f"{fraction:g}"] for fraction in args.fractions]
            mlp = as_mlp(row)
            started = time.perf_counter()
            with flops.BudgetContext(flop_budget=args.budget, quiet=True) as context:
                predictions = sampling_predictions_by_count(mlp, counts, int(row["mlp_seed"]), args.method)
            joint_runtime_s = time.perf_counter() - started
            joint_flops_used = int(context.flops_used)
            for fraction in args.fractions:
                n_samples = sample_counts[f"{fraction:g}"]
                sampling_records.append(evaluate_sampling(
                    row, args, fraction, n_samples, predictions[n_samples], method_flops[n_samples],
                    joint_runtime_s / len(counts), joint_flops_used, joint_runtime_s, corrected,
                ))
            print(f"{index}/{len(rows)} {row['mlp_name']} completed", flush=True)
        except Exception as exc:
            failures.append({
                "mlp_id": int(row["mlp_id"]),
                "mlp_name": str(row["mlp_name"]),
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"{index}/{len(rows)} {row['mlp_name']} failed", flush=True)

    estimator_hash = hashlib.sha256(CORRECTED_K2.read_bytes()).hexdigest()
    result = {
        "schema_version": "1.0",
        "experiment": f"sampling_baseline_{args.method}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if not failures else "completed_with_failures",
        "configuration": {
            "phase": "phase1",
            "dataset": DATASET,
            "dataset_revision": REVISION,
            "dataset_split": "mini",
            "method": args.method,
            "flop_budget": args.budget,
            "score_floor": SCORE_FLOOR,
            "target_budget_fractions": list(args.fractions),
            "sample_counts": sample_counts,
            "standalone_flops_by_sample_count": {str(k): v for k, v in method_flops.items()},
            "execution_mode": "prefixes_from_one_max_sample_run_per_mlp",
            "depth_prefixes": args.prefixes,
            "limit_mlps": args.limit_mlps,
            "rqmc_note": "randomly shifted rank-1 lattice" if args.method == "rqmc_lattice" else None,
        },
        "source": {
            "corrected_k2_path": str(CORRECTED_K2.relative_to(ROOT)),
            "corrected_k2_sha256": estimator_hash,
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--porcelain")),
        },
        "corrected_k2_summary": summarize_corrected(corrected_records, args.prefixes),
        "summary_by_fraction": grouped_summary(sampling_records, args.prefixes),
        "per_mlp_corrected_k2": corrected_records,
        "per_mlp_sampling": sampling_records,
        "failures": failures,
    }
    RESULTS.mkdir(exist_ok=True)
    suffix = "mini" if args.limit_mlps is None else f"mini_limit{args.limit_mlps}"
    output = RESULTS / f"sampling_{args.method}_{suffix}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("ordinary_mc", "rqmc_lattice"), default="ordinary_mc")
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--prefixes", type=int, nargs="+", default=list(PREFIXES))
    parser.add_argument("--limit-mlps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_mlps is not None and args.limit_mlps <= 0:
        raise SystemExit("--limit-mlps must be positive")
    if any(fraction <= 0.0 or fraction >= 1.0 for fraction in args.fractions):
        raise SystemExit("--fractions must be between 0 and 1")
    output = run(args)
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
