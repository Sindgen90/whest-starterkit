"""Benchmark cheap diagonal-K3 residual corrections with signed cubature.

The diagonal-K3 estimator is a cheap proxy for the finite-memory factorized K3
teacher.  It does not carry generic K3 factors; it keeps only marginal third
cumulants and therefore avoids factor growth.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
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

REPORTS = ROOT / "research" / "reports"


def blend(analytic: np.ndarray, cubature: np.ndarray, lambda_: float) -> tuple[np.ndarray, int]:
    with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as context:
        prediction = fnp.add(
            fnp.multiply(fnp.array(analytic), 1.0 - lambda_),
            fnp.multiply(fnp.array(cubature), lambda_),
        )
    return np.asarray(prediction, dtype=np.float64), int(context.flops_used)


def summarize_records(records: list[dict[str, Any]], split: str, method: str) -> dict[str, Any]:
    subset = [record for record in records if record["split"] == split and record["method"] == method]
    values = [record["final_mse"] for record in subset]
    adjusted = [record["adjusted_score"] for record in subset]
    flops_values = [record["flops"] for record in subset]
    win_values = [record["win_vs_cubature"] for record in subset if record["win_vs_cubature"] is not None]
    return {
        "split": split,
        "method": method,
        "n_completed": len(subset),
        "mean_final_mse": aggregate(values)["mean"],
        "median_final_mse": aggregate(values)["median"],
        "se_final_mse": aggregate(values)["standard_error"],
        "worst_decile_final_mse": aggregate(values)["worst_decile"],
        "max_final_mse": aggregate(values)["max"],
        "mean_adjusted_score": aggregate(adjusted)["mean"],
        "mean_flops": aggregate(flops_values)["mean"],
        "win_fraction_vs_cubature": float(np.mean(win_values)) if win_values else float("nan"),
    }


def evaluate_config(
    config_name: str,
    rows: list[dict[str, Any]],
    split_name: str,
    lambda_: float,
    analytic_pred: dict[int, np.ndarray],
    analytic_flops: dict[int, int],
    cub_pred: dict[int, np.ndarray],
    cub_flops: dict[int, int],
) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        mlp_id = int(row["mlp_id"])
        target = np.asarray(row["all_layer_means"], dtype=np.float64)
        combo_pred, blend_flops = blend(analytic_pred[mlp_id], cub_pred[mlp_id], lambda_)
        combo_flops = analytic_flops[mlp_id] + cub_flops[mlp_id] + blend_flops
        analytic_metrics = metrics(analytic_pred[mlp_id], target, analytic_flops[mlp_id])
        cub_metrics = metrics(cub_pred[mlp_id], target, cub_flops[mlp_id])
        combo_metrics = metrics(combo_pred, target, combo_flops)
        for method, method_metrics, win in (
            ("diagonal_k3", analytic_metrics, None),
            ("cubature", cub_metrics, None),
            ("combo", combo_metrics, combo_metrics["final_mse"] < cub_metrics["final_mse"]),
        ):
            records.append(
                {
                    "config": config_name,
                    "split": split_name,
                    "method": method,
                    "mlp_id": mlp_id,
                    "mlp_name": str(row["mlp_name"]),
                    "mlp_seed": int(row["mlp_seed"]),
                    "lambda": lambda_ if method == "combo" else float("nan"),
                    "final_mse": method_metrics["final_mse"],
                    "all_layer_mse": method_metrics["all_layer_mse"],
                    "adjusted_score": method_metrics["adjusted_score"],
                    "flops": method_metrics["flops"],
                    "budget_fraction": method_metrics["budget_fraction"],
                    "win_vs_cubature": win,
                }
            )
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, args: argparse.Namespace, result: dict[str, Any]) -> None:
    lines = [
        "# Cheap diagonal-K3 residual combo",
        "",
        "This tests a cheap K3 residual proxy: retain only per-neuron marginal third cumulants, propagate them with `W^3`, and use them as a first-order Edgeworth mean correction. No generic K3 factors are used.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{' '.join(args.command)}\n```",
        "",
        "## Result matrix",
        "",
        "| config | split | method | final MSE | adjusted score | FLOPs | win vs cubature | lambda |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summary"]:
        if row["method"] != "combo":
            continue
        calibration = result["calibration"][row["config"]]
        lines.append(
            "| {config} | {split} | {method} | {mse:.6g} | {score:.6g} | {flops:.6g} | {win:.3g} | {lam:.6g} |".format(
                config=row["config"],
                split=row["split"],
                method=row["method"],
                mse=row["mean_final_mse"],
                score=row["mean_adjusted_score"],
                flops=row["mean_flops"],
                win=row["win_fraction_vs_cubature"],
                lam=calibration["lambda"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision note",
            "",
            "Promote only if validation adjusted score beats the current K2+cubature submission candidate (`3.838e-7`) or if raw MSE improves enough to justify a separate experimental submission.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dev_rows = select_rows("development", args.dev_limit)
    validation_rows = select_rows("validation", args.validation_limit)
    dev_calibration = [row for row in dev_rows if int(row["mlp_id"]) % 2 == args.calibration_parity]
    dev_selection = [row for row in dev_rows if int(row["mlp_id"]) % 2 != args.calibration_parity]
    all_rows = [*dev_rows, *validation_rows]

    cub_estimator = load_estimator(CUBATURE)()
    cub_pred: dict[int, np.ndarray] = {}
    cub_flops: dict[int, int] = {}
    for index, row in enumerate(all_rows, start=1):
        mlp_id = int(row["mlp_id"])
        cub_prediction, cub_used, _ = predict(cub_estimator, row, BUDGET)
        cub_pred[mlp_id] = cub_prediction
        cub_flops[mlp_id] = cub_used
        print(f"cubature {index}/{len(all_rows)} mlp_id={mlp_id}", flush=True)

    per_mlp_records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    calibration_by_config: dict[str, Any] = {}
    for third_scale in args.third_scales:
        for max_k3_layer in args.max_k3_layers:
            config_name = f"scale={third_scale:g},L={max_k3_layer}"
            estimator = DiagonalK3(
                third_scale=third_scale,
                max_k3_layer=max_k3_layer,
                output_coefficient=args.output_coefficient,
            )
            analytic_pred: dict[int, np.ndarray] = {}
            analytic_flops: dict[int, int] = {}
            start = time.perf_counter()
            for row in all_rows:
                mlp_id = int(row["mlp_id"])
                prediction, used, _ = predict(estimator, row, BUDGET)
                analytic_pred[mlp_id] = prediction
                analytic_flops[mlp_id] = used
            calibration = fit_lambda(dev_calibration, analytic_pred, cub_pred)
            calibration["runtime_s"] = time.perf_counter() - start
            calibration_by_config[config_name] = calibration
            config_records = [
                *evaluate_config(
                    config_name,
                    dev_selection,
                    "development_selection",
                    float(calibration["lambda"]),
                    analytic_pred,
                    analytic_flops,
                    cub_pred,
                    cub_flops,
                ),
                *evaluate_config(
                    config_name,
                    validation_rows,
                    "validation_frozen_lambda",
                    float(calibration["lambda"]),
                    analytic_pred,
                    analytic_flops,
                    cub_pred,
                    cub_flops,
                ),
            ]
            per_mlp_records.extend(config_records)
            for split in ("development_selection", "validation_frozen_lambda"):
                for method in ("diagonal_k3", "cubature", "combo"):
                    row = summarize_records(config_records, split, method)
                    row["config"] = config_name
                    row["third_scale"] = third_scale
                    row["max_k3_layer"] = max_k3_layer
                    row["lambda"] = calibration["lambda"] if method == "combo" else float("nan")
                    summary_rows.append(row)
            print(
                f"{config_name} lambda={calibration['lambda']:.6g} "
                f"done in {calibration['runtime_s']:.1f}s",
                flush=True,
            )

    result = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "created_at_utc": args.created_at,
        "configuration": {
            "third_scales": args.third_scales,
            "max_k3_layers": args.max_k3_layers,
            "output_coefficient": args.output_coefficient,
            "dev_limit": args.dev_limit,
            "validation_limit": args.validation_limit,
            "calibration_parity": args.calibration_parity,
        },
        "calibration": calibration_by_config,
        "summary": summary_rows,
        "per_mlp": per_mlp_records,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    result_tag = args.result_tag
    json_path = RESULTS / f"diagonal_k3_residual_combo_{result_tag}.json"
    csv_path = RESULTS / f"diagonal_k3_residual_combo_{result_tag}.csv"
    report_path = REPORTS / f"diagonal_k3_residual_combo_{result_tag}.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, summary_rows)
    write_report(report_path, args, result)
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path), "summary": summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--calibration-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--third-scales", type=float, nargs="+", default=[-0.001, -0.003, -0.01])
    parser.add_argument("--max-k3-layers", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--output-coefficient", type=float, default=0.9918858550287327)
    parser.add_argument("--result-tag", default="grid")
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
