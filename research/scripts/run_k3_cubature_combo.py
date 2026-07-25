"""Calibrate finite-memory K3 + signed-Hadamard-cubature convex blend.

This is a K3-only follow-up to ``run_k2_cubature_combo.py``.  It keeps the
sampling estimator unchanged and swaps the analytic component from corrected K2
to the finite-memory factorized K3 prototype.
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

from run_k2_cubature_combo import (
    BUDGET,
    CUBATURE,
    RESULTS,
    SCORE_FLOOR,
    aggregate,
    fit_lambda,
    load_estimator,
    metrics,
    predict,
    select_rows,
)

from experiments.factorized_k3 import Estimator as FiniteMemoryK3

REPORTS = ROOT / "research" / "reports"


def blend(analytic: np.ndarray, cubature: np.ndarray, lambda_: float) -> tuple[np.ndarray, int]:
    with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as context:
        prediction = fnp.add(
            fnp.multiply(fnp.array(analytic), 1.0 - lambda_),
            fnp.multiply(fnp.array(cubature), lambda_),
        )
    return np.asarray(prediction, dtype=np.float64), int(context.flops_used)


def evaluate_split(
    name: str,
    rows: list[dict[str, Any]],
    lambda_: float,
    analytic_pred: dict[int, np.ndarray],
    analytic_flops: dict[int, int],
    cub_pred: dict[int, np.ndarray],
    cub_flops: dict[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_mlp = []
    summary_rows = []
    for row in rows:
        mlp_id = int(row["mlp_id"])
        target = np.asarray(row["all_layer_means"], dtype=np.float64)
        combo_pred, blend_flops = blend(analytic_pred[mlp_id], cub_pred[mlp_id], lambda_)
        combo_flops = analytic_flops[mlp_id] + cub_flops[mlp_id] + blend_flops
        row_metrics = {
            "finite_memory_k3": metrics(analytic_pred[mlp_id], target, analytic_flops[mlp_id]),
            "cubature": metrics(cub_pred[mlp_id], target, cub_flops[mlp_id]),
            "combo": metrics(combo_pred, target, combo_flops),
        }
        per_mlp.append(
            {
                "split": name,
                "mlp_id": mlp_id,
                "mlp_name": str(row["mlp_name"]),
                "mlp_seed": int(row["mlp_seed"]),
                "lambda": lambda_,
                "metrics": row_metrics,
                "combo_win_vs_cubature": row_metrics["combo"]["final_mse"] < row_metrics["cubature"]["final_mse"],
                "combo_win_vs_k3": row_metrics["combo"]["final_mse"] < row_metrics["finite_memory_k3"]["final_mse"],
            }
        )
    for method in ("finite_memory_k3", "cubature", "combo"):
        values = [record["metrics"][method]["final_mse"] for record in per_mlp]
        adjusted = [record["metrics"][method]["adjusted_score"] for record in per_mlp]
        flops_values = [record["metrics"][method]["flops"] for record in per_mlp]
        summary_rows.append(
            {
                "split": name,
                "method": method,
                "n_completed": len(per_mlp),
                "lambda": lambda_ if method == "combo" else float("nan"),
                "mean_final_mse": aggregate(values)["mean"],
                "median_final_mse": aggregate(values)["median"],
                "se_final_mse": aggregate(values)["standard_error"],
                "worst_decile_final_mse": aggregate(values)["worst_decile"],
                "max_final_mse": aggregate(values)["max"],
                "mean_adjusted_score": aggregate(adjusted)["mean"],
                "mean_flops": aggregate(flops_values)["mean"],
                "win_fraction_vs_cubature": float(
                    np.mean([record["combo_win_vs_cubature"] for record in per_mlp])
                )
                if method == "combo"
                else float("nan"),
            }
        )
    return per_mlp, summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    args: argparse.Namespace,
    calibration: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Finite-memory K3 + signed Hadamard cubature combo",
        "",
        "This tests whether the K3-only finite-memory analytic estimator improves the current cubature blend enough to justify its extra charged FLOPs.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{' '.join(args.command)}\n```",
        "",
        "## K3 configuration",
        "",
        f"- `third_scale`: `{args.third_scale}`",
        f"- `generic_horizon`: `{args.generic_horizon}`",
        f"- `max_k3_layer`: `{args.max_k3_layer}`",
        "",
        f"Calibrated lambda: `{calibration['lambda']:.6g}` (`raw={calibration['raw_lambda']:.6g}`)",
        "",
        "## Result matrix",
        "",
        "| split | method | final MSE | adjusted score | FLOPs | win vs cubature |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {split} | {method} | {mse:.6g} | {score:.6g} | {flops:.6g} | {win:.3g} |".format(
                split=row["split"],
                method=row["method"],
                mse=row["mean_final_mse"],
                score=row["mean_adjusted_score"],
                flops=row["mean_flops"],
                win=row["win_fraction_vs_cubature"],
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
    analytic_estimator = FiniteMemoryK3(
        third_scale=args.third_scale,
        generic_horizon=args.generic_horizon,
        max_k3_layer=args.max_k3_layer,
    )
    cub_estimator = load_estimator(CUBATURE)()
    analytic_pred: dict[int, np.ndarray] = {}
    cub_pred: dict[int, np.ndarray] = {}
    analytic_flops: dict[int, int] = {}
    cub_flops: dict[int, int] = {}
    for index, row in enumerate(all_rows, start=1):
        mlp_id = int(row["mlp_id"])
        analytic_prediction, analytic_used, _ = predict(analytic_estimator, row, BUDGET)
        cub_prediction, cub_used, _ = predict(cub_estimator, row, BUDGET)
        analytic_pred[mlp_id] = analytic_prediction
        cub_pred[mlp_id] = cub_prediction
        analytic_flops[mlp_id] = analytic_used
        cub_flops[mlp_id] = cub_used
        print(f"{index}/{len(all_rows)} mlp_id={mlp_id} predicted", flush=True)
    calibration = fit_lambda(dev_calibration, analytic_pred, cub_pred)
    dev_per_mlp, dev_summary = evaluate_split(
        "development_selection",
        dev_selection,
        float(calibration["lambda"]),
        analytic_pred,
        analytic_flops,
        cub_pred,
        cub_flops,
    )
    val_per_mlp, val_summary = evaluate_split(
        "validation_frozen_lambda",
        validation_rows,
        float(calibration["lambda"]),
        analytic_pred,
        analytic_flops,
        cub_pred,
        cub_flops,
    )
    summary_rows = [*dev_summary, *val_summary]
    result = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "created_at_utc": args.created_at,
        "configuration": {
            "third_scale": args.third_scale,
            "generic_horizon": args.generic_horizon,
            "max_k3_layer": args.max_k3_layer,
            "calibration_parity": args.calibration_parity,
            "dev_limit": args.dev_limit,
            "validation_limit": args.validation_limit,
        },
        "calibration": calibration,
        "summary": summary_rows,
        "per_mlp": [*dev_per_mlp, *val_per_mlp],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"scale{args.third_scale:g}".replace(".", "p")
    json_path = RESULTS / f"k3_cubature_combo_{tag}_H{args.generic_horizon}_L{args.max_k3_layer}.json"
    csv_path = RESULTS / f"k3_cubature_combo_{tag}_H{args.generic_horizon}_L{args.max_k3_layer}.csv"
    report_path = REPORTS / f"k3_cubature_combo_{tag}_H{args.generic_horizon}_L{args.max_k3_layer}.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, summary_rows)
    write_report(report_path, args, calibration, summary_rows)
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path), "summary": summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--calibration-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--third-scale", type=float, default=-0.001)
    parser.add_argument("--generic-horizon", type=int, default=2)
    parser.add_argument("--max-k3-layer", type=int, default=8)
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.created_at = datetime.now(timezone.utc).isoformat()
    args.command = [str(item) for item in __import__("sys").argv]
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
