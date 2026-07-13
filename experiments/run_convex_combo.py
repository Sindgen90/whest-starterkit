"""Calibrate a scalar convex combination of corrected K2 and RQMC on Mini."""
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
RQMC = ROOT / "submissions" / "rqmc_lattice_002" / "estimator.py"


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


def fit_lambda(rows: list[dict[str, Any]], k2: Any, rqmc: Any, args: argparse.Namespace) -> dict[str, Any]:
    numerator = 0.0
    denominator = 0.0
    records = []
    for row in rows:
        k2_prediction, _, _ = predict(k2, row, args.budget)
        rqmc_prediction, _, _ = predict(rqmc, row, args.budget)
        target = np.asarray(row["all_layer_means"], dtype=np.float64)
        delta = rqmc_prediction[-1] - k2_prediction[-1]
        residual = k2_prediction[-1] - target[-1]
        numerator += float(np.vdot(delta, residual))
        denominator += float(np.vdot(delta, delta))
        records.append({
            "mlp_id": int(row["mlp_id"]),
            "mlp_name": str(row["mlp_name"]),
            "k2_final_mse": metrics(k2_prediction, target, args.prefixes)["final_mse"],
            "rqmc_final_mse": metrics(rqmc_prediction, target, args.prefixes)["final_mse"],
        })
    raw_lambda = -numerator / denominator if denominator else 0.0
    return {
        "objective": "final-layer least squares",
        "formula": "argmin_lambda ||k2 + lambda * (rqmc - k2) - target||^2",
        "raw_lambda": raw_lambda,
        "lambda": min(1.0, max(0.0, raw_lambda)),
        "n_mlps": len(rows),
        "selector": f"mlp_id % 2 == {args.calibration_parity}",
        "records": records,
    }


def evaluate_row(row: dict[str, Any], k2: Any, rqmc: Any, lambda_: float, args: argparse.Namespace) -> dict[str, Any]:
    target = np.asarray(row["all_layer_means"], dtype=np.float64)
    k2_prediction, k2_flops, k2_runtime = predict(k2, row, args.budget)
    rqmc_prediction, rqmc_flops, rqmc_runtime = predict(rqmc, row, args.budget)

    combo_started = time.perf_counter()
    mlp = as_mlp(row)
    with flops.BudgetContext(flop_budget=args.budget, quiet=True) as context:
        combo_k2 = k2.predict(mlp, args.budget)
        combo_rqmc = rqmc.predict(mlp, args.budget)
        combo_prediction = fnp.add(fnp.multiply(combo_k2, 1.0 - lambda_), fnp.multiply(combo_rqmc, lambda_))
    combo_runtime = time.perf_counter() - combo_started
    combo_prediction_np = np.asarray(combo_prediction, dtype=np.float64)

    k2_metrics = metrics(k2_prediction, target, args.prefixes)
    rqmc_metrics = metrics(rqmc_prediction, target, args.prefixes)
    combo_metrics = metrics(combo_prediction_np, target, args.prefixes)
    combo_flops = int(context.flops_used)
    return {
        "mlp_id": int(row["mlp_id"]),
        "mlp_name": str(row["mlp_name"]),
        "mlp_seed": int(row["mlp_seed"]),
        "status": "completed",
        "metrics": {
            "corrected_k2": k2_metrics,
            "rqmc": rqmc_metrics,
            "combo": combo_metrics,
        },
        "paired_vs_corrected_k2": {
            "combo_final_mse_delta": combo_metrics["final_mse"] - k2_metrics["final_mse"],
            "combo_final_mse_ratio": combo_metrics["final_mse"] / k2_metrics["final_mse"],
            "rqmc_final_mse_delta": rqmc_metrics["final_mse"] - k2_metrics["final_mse"],
            "rqmc_final_mse_ratio": rqmc_metrics["final_mse"] / k2_metrics["final_mse"],
        },
        "compute": {
            "corrected_k2_flops": k2_flops,
            "rqmc_flops": rqmc_flops,
            "combo_flops": combo_flops,
            "flop_budget": args.budget,
            "corrected_k2_runtime_s": k2_runtime,
            "rqmc_runtime_s": rqmc_runtime,
            "combo_runtime_s": combo_runtime,
            "combo_flop_fraction": combo_flops / args.budget,
            "combo_adjusted_final_score": combo_metrics["final_mse"] * max(SCORE_FLOOR, combo_flops / args.budget),
        },
    }


def summarize(records: list[dict[str, Any]], prefixes: Iterable[int]) -> dict[str, Any]:
    done = [record for record in records if record["status"] == "completed"]
    combo_wins = [record["paired_vs_corrected_k2"]["combo_final_mse_delta"] < 0.0 for record in done]
    rqmc_wins = [record["paired_vs_corrected_k2"]["rqmc_final_mse_delta"] < 0.0 for record in done]
    summary = {
        "n_requested": len(records),
        "n_completed": len(done),
        "n_failed": len(records) - len(done),
        "corrected_k2_final_mse": aggregate(r["metrics"]["corrected_k2"]["final_mse"] for r in done),
        "rqmc_final_mse": aggregate(r["metrics"]["rqmc"]["final_mse"] for r in done),
        "combo_final_mse": aggregate(r["metrics"]["combo"]["final_mse"] for r in done),
        "corrected_k2_all_layer_mse": aggregate(r["metrics"]["corrected_k2"]["all_layer_mse"] for r in done),
        "rqmc_all_layer_mse": aggregate(r["metrics"]["rqmc"]["all_layer_mse"] for r in done),
        "combo_all_layer_mse": aggregate(r["metrics"]["combo"]["all_layer_mse"] for r in done),
        "combo_adjusted_final_score": aggregate(r["compute"]["combo_adjusted_final_score"] for r in done),
        "combo_flops": aggregate(r["compute"]["combo_flops"] for r in done),
        "combo_runtime_s": aggregate(r["compute"]["combo_runtime_s"] for r in done),
        "paired_vs_corrected_k2": {
            "combo_win_fraction": sum(combo_wins) / len(combo_wins) if combo_wins else None,
            "rqmc_win_fraction": sum(rqmc_wins) / len(rqmc_wins) if rqmc_wins else None,
            "combo_final_mse_delta": aggregate(r["paired_vs_corrected_k2"]["combo_final_mse_delta"] for r in done),
            "combo_final_mse_ratio": aggregate(r["paired_vs_corrected_k2"]["combo_final_mse_ratio"] for r in done),
            "rqmc_final_mse_delta": aggregate(r["paired_vs_corrected_k2"]["rqmc_final_mse_delta"] for r in done),
            "rqmc_final_mse_ratio": aggregate(r["paired_vs_corrected_k2"]["rqmc_final_mse_ratio"] for r in done),
        },
        "prefix_mse": {},
    }
    for prefix in prefixes:
        summary["prefix_mse"][str(prefix)] = {
            "corrected_k2": aggregate(r["metrics"]["corrected_k2"]["prefix_mse"][str(prefix)] for r in done),
            "rqmc": aggregate(r["metrics"]["rqmc"]["prefix_mse"][str(prefix)] for r in done),
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
    rqmc = load_estimator(RQMC)()
    calibration = fit_lambda(calibration_rows, k2, rqmc, args)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(evaluation_rows, start=1):
        try:
            records.append(evaluate_row(row, k2, rqmc, calibration["lambda"], args))
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
        "experiment": "corrected_k2_rqmc_convex_combo",
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
        },
        "calibration": calibration,
        "source": {
            "corrected_k2_path": str(CORRECTED_K2.relative_to(ROOT)),
            "corrected_k2_sha256": hashlib.sha256(CORRECTED_K2.read_bytes()).hexdigest(),
            "rqmc_path": str(RQMC.relative_to(ROOT)),
            "rqmc_sha256": hashlib.sha256(RQMC.read_bytes()).hexdigest(),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--porcelain")),
        },
        "summary": summarize(records, args.prefixes),
        "per_mlp": records,
        "failures": failures,
    }
    RESULTS.mkdir(exist_ok=True)
    suffix = "mini" if args.limit_mlps is None else f"mini_limit{args.limit_mlps}"
    output = RESULTS / f"convex_combo_{suffix}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--prefixes", type=int, nargs="+", default=list(PREFIXES))
    parser.add_argument("--calibration-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--limit-mlps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_mlps is not None and args.limit_mlps <= 1:
        raise SystemExit("--limit-mlps must be greater than 1")
    output = run(args)
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
