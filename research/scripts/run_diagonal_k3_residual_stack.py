"""Fit a two-scalar cheap K3 residual stack.

Prediction form:

    cubature + a * (corrected_k2 - cubature) + b * (diagonal_k3 - corrected_k2)

This separates the usual K2/cubature correction from the cheap diagonal-K3
residual, instead of forcing both to share one convex weight.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.diagonal_k3_estimator import Estimator as DiagonalK3
from run_k2_cubature_combo import (
    BUDGET,
    CORRECTED_K2,
    CUBATURE,
    RESULTS,
    SCORE_FLOOR,
    aggregate,
    load_estimator,
    metrics,
    predict,
    select_rows,
)

REPORTS = ROOT / "research" / "reports"


def fit_stack(
    rows: list[dict[str, Any]],
    k2_pred: dict[int, np.ndarray],
    diag_pred: dict[int, np.ndarray],
    cub_pred: dict[int, np.ndarray],
    ridge: float,
) -> dict[str, Any]:
    gram = np.zeros((2, 2), dtype=np.float64)
    rhs = np.zeros(2, dtype=np.float64)
    for row in rows:
        mlp_id = int(row["mlp_id"])
        target = np.asarray(row["all_layer_means"], dtype=np.float64)[-1]
        d1 = k2_pred[mlp_id][-1] - cub_pred[mlp_id][-1]
        d2 = diag_pred[mlp_id][-1] - k2_pred[mlp_id][-1]
        residual = target - cub_pred[mlp_id][-1]
        design = np.stack([d1, d2], axis=1)
        gram += design.T @ design
        rhs += design.T @ residual
    coef = np.linalg.solve(gram + ridge * np.eye(2), rhs)
    return {
        "a_k2_minus_cubature": float(coef[0]),
        "b_diag_minus_k2": float(coef[1]),
        "ridge": ridge,
        "n_calibration_mlps": len(rows),
        "equivalent_lambda_if_b_zero": float(1.0 - coef[0]),
    }


def stack_prediction(k2: np.ndarray, diag: np.ndarray, cub: np.ndarray, a: float, b: float) -> tuple[np.ndarray, int]:
    with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as context:
        prediction = fnp.add(
            fnp.array(cub),
            fnp.add(
                fnp.multiply(fnp.subtract(fnp.array(k2), fnp.array(cub)), a),
                fnp.multiply(fnp.subtract(fnp.array(diag), fnp.array(k2)), b),
            ),
        )
    return np.asarray(prediction, dtype=np.float64), int(context.flops_used)


def summarize(per_mlp: list[dict[str, Any]], split: str, method: str) -> dict[str, Any]:
    subset = [row for row in per_mlp if row["split"] == split and row["method"] == method]
    return {
        "split": split,
        "method": method,
        "n_completed": len(subset),
        "mean_final_mse": aggregate([row["final_mse"] for row in subset])["mean"],
        "median_final_mse": aggregate([row["final_mse"] for row in subset])["median"],
        "se_final_mse": aggregate([row["final_mse"] for row in subset])["standard_error"],
        "worst_decile_final_mse": aggregate([row["final_mse"] for row in subset])["worst_decile"],
        "max_final_mse": aggregate([row["final_mse"] for row in subset])["max"],
        "mean_adjusted_score": aggregate([row["adjusted_score"] for row in subset])["mean"],
        "mean_flops": aggregate([row["flops"] for row in subset])["mean"],
        "win_fraction_vs_current_k2_combo": float(
            np.mean([row["win_vs_current_k2_combo"] for row in subset if row["win_vs_current_k2_combo"] is not None])
        )
        if method == "stack"
        else float("nan"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, args: argparse.Namespace, result: dict[str, Any]) -> None:
    c = result["calibration"]
    lines = [
        "# Cheap diagonal-K3 residual stack",
        "",
        "This tests a two-scalar residual form: `cubature + a*(K2-cubature) + b*(diagK3-K2)`.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{' '.join(args.command)}\n```",
        "",
        "## Calibration",
        "",
        f"- `a`: `{c['a_k2_minus_cubature']:.8g}`",
        f"- `b`: `{c['b_diag_minus_k2']:.8g}`",
        f"- equivalent K2/cubature lambda if `b=0`: `{c['equivalent_lambda_if_b_zero']:.8g}`",
        "",
        "## Result matrix",
        "",
        "| split | method | final MSE | adjusted score | FLOPs | win vs current K2 combo |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summary"]:
        lines.append(
            "| {split} | {method} | {mse:.6g} | {score:.6g} | {flops:.6g} | {win:.3g} |".format(
                split=row["split"],
                method=row["method"],
                mse=row["mean_final_mse"],
                score=row["mean_adjusted_score"],
                flops=row["mean_flops"],
                win=row["win_fraction_vs_current_k2_combo"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dev_rows = select_rows("development", args.dev_limit)
    validation_rows = select_rows("validation", args.validation_limit)
    dev_calibration = [row for row in dev_rows if int(row["mlp_id"]) % 2 == args.calibration_parity]
    dev_selection = [row for row in dev_rows if int(row["mlp_id"]) % 2 != args.calibration_parity]
    all_rows = [*dev_rows, *validation_rows]

    k2_estimator = load_estimator(CORRECTED_K2)()
    cub_estimator = load_estimator(CUBATURE)()
    diag_estimator = DiagonalK3(third_scale=args.third_scale, max_k3_layer=args.max_k3_layer)

    k2_pred: dict[int, np.ndarray] = {}
    diag_pred: dict[int, np.ndarray] = {}
    cub_pred: dict[int, np.ndarray] = {}
    k2_flops: dict[int, int] = {}
    diag_flops: dict[int, int] = {}
    cub_flops: dict[int, int] = {}
    for index, row in enumerate(all_rows, start=1):
        mlp_id = int(row["mlp_id"])
        k2_prediction, k2_used, _ = predict(k2_estimator, row, BUDGET)
        diag_prediction, diag_used, _ = predict(diag_estimator, row, BUDGET)
        cub_prediction, cub_used, _ = predict(cub_estimator, row, BUDGET)
        k2_pred[mlp_id] = k2_prediction
        diag_pred[mlp_id] = diag_prediction
        cub_pred[mlp_id] = cub_prediction
        k2_flops[mlp_id] = k2_used
        diag_flops[mlp_id] = diag_used
        cub_flops[mlp_id] = cub_used
        print(f"{index}/{len(all_rows)} mlp_id={mlp_id} predicted", flush=True)

    calibration = fit_stack(dev_calibration, k2_pred, diag_pred, cub_pred, args.ridge)
    a = float(calibration["a_k2_minus_cubature"])
    b = float(calibration["b_diag_minus_k2"])
    per_mlp = []
    for split_name, rows in (("development_selection", dev_selection), ("validation_frozen", validation_rows)):
        for row in rows:
            mlp_id = int(row["mlp_id"])
            target = np.asarray(row["all_layer_means"], dtype=np.float64)
            current_pred, current_blend_flops = stack_prediction(
                k2_pred[mlp_id],
                k2_pred[mlp_id],
                cub_pred[mlp_id],
                1.0 - args.current_lambda,
                0.0,
            )
            current_flops = k2_flops[mlp_id] + cub_flops[mlp_id] + current_blend_flops
            stacked_pred, stack_flops = stack_prediction(
                k2_pred[mlp_id],
                diag_pred[mlp_id],
                cub_pred[mlp_id],
                a,
                b,
            )
            stacked_flops = k2_flops[mlp_id] + diag_flops[mlp_id] + cub_flops[mlp_id] + stack_flops
            method_metrics = {
                "cubature": metrics(cub_pred[mlp_id], target, cub_flops[mlp_id]),
                "current_k2_combo": metrics(current_pred, target, current_flops),
                "stack": metrics(stacked_pred, target, stacked_flops),
            }
            for method, values in method_metrics.items():
                per_mlp.append(
                    {
                        "split": split_name,
                        "method": method,
                        "mlp_id": mlp_id,
                        "mlp_name": str(row["mlp_name"]),
                        "mlp_seed": int(row["mlp_seed"]),
                        "final_mse": values["final_mse"],
                        "all_layer_mse": values["all_layer_mse"],
                        "adjusted_score": values["adjusted_score"],
                        "flops": values["flops"],
                        "budget_fraction": values["budget_fraction"],
                        "win_vs_current_k2_combo": values["final_mse"] < method_metrics["current_k2_combo"]["final_mse"]
                        if method == "stack"
                        else None,
                    }
                )

    summary_rows = [
        summarize(per_mlp, split, method)
        for split in ("development_selection", "validation_frozen")
        for method in ("cubature", "current_k2_combo", "stack")
    ]
    result = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "created_at_utc": args.created_at,
        "configuration": {
            "third_scale": args.third_scale,
            "max_k3_layer": args.max_k3_layer,
            "ridge": args.ridge,
            "current_lambda": args.current_lambda,
            "calibration_parity": args.calibration_parity,
            "dev_limit": args.dev_limit,
            "validation_limit": args.validation_limit,
        },
        "calibration": calibration,
        "summary": summary_rows,
        "per_mlp": per_mlp,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = args.result_tag
    json_path = RESULTS / f"diagonal_k3_residual_stack_{tag}.json"
    csv_path = RESULTS / f"diagonal_k3_residual_stack_{tag}.csv"
    report_path = REPORTS / f"diagonal_k3_residual_stack_{tag}.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, summary_rows)
    write_report(report_path, args, result)
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path), "summary": summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--calibration-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--third-scale", type=float, default=-0.3)
    parser.add_argument("--max-k3-layer", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-12)
    parser.add_argument("--current-lambda", type=float, default=0.8798916787983015)
    parser.add_argument("--result-tag", default="full")
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.created_at = datetime.now(timezone.utc).isoformat()
    args.command = [str(item) for item in sys.argv]
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
