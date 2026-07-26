"""Fit cheap layerwise-diagonal K3 residual stacks.

Prediction form:

    cubature + a * (corrected_k2 - cubature) + b * (layerwise_diag_k3 - corrected_k2)

Layerwise diagonal K3 keeps one diagonal third-cumulant cohort per source layer.
"""
from __future__ import annotations

import argparse
import csv
import json
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

from experiments.layerwise_diagonal_k3_estimator import Estimator as LayerwiseDiagonalK3
from run_diagonal_k3_residual_stack import fit_stack, stack_prediction
from run_k2_cubature_combo import (
    BUDGET,
    CORRECTED_K2,
    CUBATURE,
    RESULTS,
    aggregate,
    load_estimator,
    metrics,
    predict,
    select_rows,
)

REPORTS = ROOT / "research" / "reports"


def horizon_label(value: int | None) -> str:
    return "full" if value is None else f"H{value}"


def summarize(per_mlp: list[dict[str, Any]], config: str, split: str, method: str) -> dict[str, Any]:
    subset = [
        row for row in per_mlp if row["config"] == config and row["split"] == split and row["method"] == method
    ]
    wins = [row["win_vs_current_k2_combo"] for row in subset if row["win_vs_current_k2_combo"] is not None]
    return {
        "config": config,
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
        "win_fraction_vs_current_k2_combo": float(np.mean(wins)) if wins else float("nan"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, args: argparse.Namespace, result: dict[str, Any]) -> None:
    lines = [
        "# Layerwise diagonal-K3 residual stack",
        "",
        "This tests source-layer-aware diagonal K3: one diagonal third-cumulant cube per source layer, no generic off-diagonal tensor factors.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{' '.join(args.command)}\n```",
        "",
        "## Result matrix",
        "",
        "| config | split | method | final MSE | adjusted score | FLOPs | win vs current K2 combo | a | b |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summary"]:
        if row["method"] != "stack":
            continue
        calibration = result["calibration"][row["config"]]
        lines.append(
            "| {config} | {split} | {method} | {mse:.6g} | {score:.6g} | {flops:.6g} | {win:.3g} | {a:.6g} | {b:.6g} |".format(
                config=row["config"],
                split=row["split"],
                method=row["method"],
                mse=row["mean_final_mse"],
                score=row["mean_adjusted_score"],
                flops=row["mean_flops"],
                win=row["win_fraction_vs_current_k2_combo"],
                a=calibration["a_k2_minus_cubature"],
                b=calibration["b_diag_minus_k2"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_horizons(values: list[str]) -> list[int | None]:
    horizons: list[int | None] = []
    for value in values:
        if value.lower() in {"none", "full", "null"}:
            horizons.append(None)
        else:
            horizons.append(int(value))
    return horizons


def run(args: argparse.Namespace) -> dict[str, Any]:
    horizons = parse_horizons(args.generic_horizons)
    dev_rows = select_rows("development", args.dev_limit)
    validation_rows = select_rows("validation", args.validation_limit)
    dev_calibration = [row for row in dev_rows if int(row["mlp_id"]) % 2 == args.calibration_parity]
    dev_selection = [row for row in dev_rows if int(row["mlp_id"]) % 2 != args.calibration_parity]
    all_rows = [*dev_rows, *validation_rows]

    k2_estimator = load_estimator(CORRECTED_K2)()
    cub_estimator = load_estimator(CUBATURE)()
    k2_pred: dict[int, np.ndarray] = {}
    cub_pred: dict[int, np.ndarray] = {}
    k2_flops: dict[int, int] = {}
    cub_flops: dict[int, int] = {}
    for index, row in enumerate(all_rows, start=1):
        mlp_id = int(row["mlp_id"])
        k2_prediction, k2_used, _ = predict(k2_estimator, row, BUDGET)
        cub_prediction, cub_used, _ = predict(cub_estimator, row, BUDGET)
        k2_pred[mlp_id] = k2_prediction
        cub_pred[mlp_id] = cub_prediction
        k2_flops[mlp_id] = k2_used
        cub_flops[mlp_id] = cub_used
        print(f"base {index}/{len(all_rows)} mlp_id={mlp_id}", flush=True)

    per_mlp: list[dict[str, Any]] = []
    calibration_by_config: dict[str, Any] = {}
    for third_scale in args.third_scales:
        for max_k3_layer in args.max_k3_layers:
            for generic_horizon in horizons:
                config = f"scale={third_scale:g},L={max_k3_layer},{horizon_label(generic_horizon)}"
                estimator = LayerwiseDiagonalK3(
                    third_scale=third_scale,
                    max_k3_layer=max_k3_layer,
                    generic_horizon=generic_horizon,
                )
                layerwise_pred: dict[int, np.ndarray] = {}
                layerwise_flops: dict[int, int] = {}
                for row in all_rows:
                    mlp_id = int(row["mlp_id"])
                    prediction, used, _ = predict(estimator, row, BUDGET)
                    layerwise_pred[mlp_id] = prediction
                    layerwise_flops[mlp_id] = used
                calibration = fit_stack(dev_calibration, k2_pred, layerwise_pred, cub_pred, args.ridge)
                calibration_by_config[config] = calibration
                a = float(calibration["a_k2_minus_cubature"])
                b = float(calibration["b_diag_minus_k2"])
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
                            layerwise_pred[mlp_id],
                            cub_pred[mlp_id],
                            a,
                            b,
                        )
                        stacked_flops = k2_flops[mlp_id] + layerwise_flops[mlp_id] + cub_flops[mlp_id] + stack_flops
                        method_metrics = {
                            "cubature": metrics(cub_pred[mlp_id], target, cub_flops[mlp_id]),
                            "current_k2_combo": metrics(current_pred, target, current_flops),
                            "stack": metrics(stacked_pred, target, stacked_flops),
                        }
                        for method, values in method_metrics.items():
                            per_mlp.append(
                                {
                                    "config": config,
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
                                    "win_vs_current_k2_combo": values["final_mse"]
                                    < method_metrics["current_k2_combo"]["final_mse"]
                                    if method == "stack"
                                    else None,
                                }
                            )
                print(
                    f"{config} a={calibration['a_k2_minus_cubature']:.6g} "
                    f"b={calibration['b_diag_minus_k2']:.6g}",
                    flush=True,
                )

    configs = list(calibration_by_config)
    summary_rows = [
        summarize(per_mlp, config, split, method)
        for config in configs
        for split in ("development_selection", "validation_frozen")
        for method in ("cubature", "current_k2_combo", "stack")
    ]
    result = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "created_at_utc": args.created_at,
        "configuration": {
            "third_scales": args.third_scales,
            "max_k3_layers": args.max_k3_layers,
            "generic_horizons": args.generic_horizons,
            "ridge": args.ridge,
            "current_lambda": args.current_lambda,
            "dev_limit": args.dev_limit,
            "validation_limit": args.validation_limit,
            "calibration_parity": args.calibration_parity,
        },
        "calibration": calibration_by_config,
        "summary": summary_rows,
        "per_mlp": per_mlp,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / f"layerwise_diagonal_k3_residual_stack_{args.result_tag}.json"
    csv_path = RESULTS / f"layerwise_diagonal_k3_residual_stack_{args.result_tag}.csv"
    report_path = REPORTS / f"layerwise_diagonal_k3_residual_stack_{args.result_tag}.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, summary_rows)
    write_report(report_path, args, result)
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path), "summary": summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--calibration-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--third-scales", type=float, nargs="+", default=[-0.3])
    parser.add_argument("--max-k3-layers", type=int, nargs="+", default=[8])
    parser.add_argument("--generic-horizons", nargs="+", default=["full"])
    parser.add_argument("--ridge", type=float, default=1e-12)
    parser.add_argument("--current-lambda", type=float, default=0.8798916787983015)
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
