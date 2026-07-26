"""Fit layerwise diagonal-K3 residual stack on randomized RQMC.

Prediction form:

    rqmc + a * (corrected_k2 - rqmc) + b * (layerwise_diag_k3 - corrected_k2)

This tests whether the cheap layerwise diagonal-K3 residual helps the existing
randomly shifted rank-1 lattice RQMC path.
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
    RESULTS,
    aggregate,
    load_estimator,
    metrics,
    predict,
    select_rows,
)

REPORTS = ROOT / "research" / "reports"
SCORE_FLOOR = 0.1


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


class RQMCEstimator:
    def __init__(self, n_samples: int, seed_offset: int, multiplier: int) -> None:
        self.n_samples = n_samples
        self.seed_offset = seed_offset
        self.multiplier = multiplier

    def predict(self, mlp: Any, budget: int) -> fnp.ndarray:
        _ = budget
        x = normal_lattice_samples(self.n_samples, mlp.width, mlp.seed, self.seed_offset, self.multiplier)
        rows = []
        for w in mlp.weights:
            x = fnp.maximum(fnp.matmul(x, w), 0.0)
            rows.append(fnp.mean(x, axis=0))
        return fnp.stack(rows, axis=0)


def summarize(per_mlp: list[dict[str, Any]], split: str, method: str) -> dict[str, Any]:
    subset = [row for row in per_mlp if row["split"] == split and row["method"] == method]
    wins = [row["win_vs_current_rqmc_combo"] for row in subset if row["win_vs_current_rqmc_combo"] is not None]
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
        "win_fraction_vs_current_rqmc_combo": float(np.mean(wins)) if wins else float("nan"),
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
        "# Layerwise diagonal-K3 + randomized RQMC residual stack",
        "",
        "Prediction form: `rqmc + a*(K2-rqmc) + b*(layerwiseDiagK3-K2)`.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{' '.join(args.command)}\n```",
        "",
        "## Calibration",
        "",
        f"- `a`: `{c['a_k2_minus_cubature']:.8g}`",
        f"- `b`: `{c['b_diag_minus_k2']:.8g}`",
        f"- equivalent RQMC/K2 lambda if `b=0`: `{c['equivalent_lambda_if_b_zero']:.8g}`",
        "",
        "## Result matrix",
        "",
        "| split | method | final MSE | adjusted score | FLOPs | win vs current RQMC combo |",
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
                win=row["win_fraction_vs_current_rqmc_combo"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision note",
            "",
            "Promote only if validation adjusted score beats the current cubature-based K3 residual candidate.",
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

    k2_estimator = load_estimator(CORRECTED_K2)()
    rqmc_estimator = RQMCEstimator(args.n_samples, args.rqmc_seed_offset, args.rqmc_multiplier)
    layerwise_estimator = LayerwiseDiagonalK3(
        third_scale=args.third_scale,
        max_k3_layer=args.max_k3_layer,
        generic_horizon=args.generic_horizon,
    )

    k2_pred: dict[int, np.ndarray] = {}
    rqmc_pred: dict[int, np.ndarray] = {}
    layerwise_pred: dict[int, np.ndarray] = {}
    k2_flops: dict[int, int] = {}
    rqmc_flops: dict[int, int] = {}
    layerwise_flops: dict[int, int] = {}
    for index, row in enumerate(all_rows, start=1):
        mlp_id = int(row["mlp_id"])
        k2_prediction, k2_used, _ = predict(k2_estimator, row, BUDGET)
        rqmc_prediction, rqmc_used, _ = predict(rqmc_estimator, row, BUDGET)
        layerwise_prediction, layerwise_used, _ = predict(layerwise_estimator, row, BUDGET)
        k2_pred[mlp_id] = k2_prediction
        rqmc_pred[mlp_id] = rqmc_prediction
        layerwise_pred[mlp_id] = layerwise_prediction
        k2_flops[mlp_id] = k2_used
        rqmc_flops[mlp_id] = rqmc_used
        layerwise_flops[mlp_id] = layerwise_used
        print(f"{index}/{len(all_rows)} mlp_id={mlp_id} predicted", flush=True)

    calibration = fit_stack(dev_calibration, k2_pred, layerwise_pred, rqmc_pred, args.ridge)
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
                rqmc_pred[mlp_id],
                1.0 - args.current_rqmc_lambda,
                0.0,
            )
            current_flops = k2_flops[mlp_id] + rqmc_flops[mlp_id] + current_blend_flops
            stacked_pred, stack_flops = stack_prediction(
                k2_pred[mlp_id],
                layerwise_pred[mlp_id],
                rqmc_pred[mlp_id],
                a,
                b,
            )
            stacked_flops = k2_flops[mlp_id] + layerwise_flops[mlp_id] + rqmc_flops[mlp_id] + stack_flops
            method_metrics = {
                "rqmc": metrics(rqmc_pred[mlp_id], target, rqmc_flops[mlp_id]),
                "current_k2_rqmc_combo": metrics(current_pred, target, current_flops),
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
                        "win_vs_current_rqmc_combo": values["final_mse"]
                        < method_metrics["current_k2_rqmc_combo"]["final_mse"]
                        if method == "stack"
                        else None,
                    }
                )

    summary_rows = [
        summarize(per_mlp, split, method)
        for split in ("development_selection", "validation_frozen")
        for method in ("rqmc", "current_k2_rqmc_combo", "stack")
    ]
    result = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "created_at_utc": args.created_at,
        "configuration": {
            "n_samples": args.n_samples,
            "rqmc_seed_offset": args.rqmc_seed_offset,
            "rqmc_multiplier": args.rqmc_multiplier,
            "third_scale": args.third_scale,
            "max_k3_layer": args.max_k3_layer,
            "generic_horizon": args.generic_horizon,
            "ridge": args.ridge,
            "current_rqmc_lambda": args.current_rqmc_lambda,
            "dev_limit": args.dev_limit,
            "validation_limit": args.validation_limit,
            "calibration_parity": args.calibration_parity,
        },
        "calibration": calibration,
        "summary": summary_rows,
        "per_mlp": per_mlp,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / f"layerwise_k3_rqmc_residual_stack_{args.result_tag}.json"
    csv_path = RESULTS / f"layerwise_k3_rqmc_residual_stack_{args.result_tag}.csv"
    report_path = REPORTS / f"layerwise_k3_rqmc_residual_stack_{args.result_tag}.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, summary_rows)
    write_report(report_path, args, result)
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path), "summary": summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--calibration-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--rqmc-seed-offset", type=int, default=8_675_309)
    parser.add_argument("--rqmc-multiplier", type=int, default=17_797)
    parser.add_argument("--current-rqmc-lambda", type=float, default=0.8037657183352081)
    parser.add_argument("--third-scale", type=float, default=-1.0)
    parser.add_argument("--max-k3-layer", type=int, default=8)
    parser.add_argument("--generic-horizon", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1e-12)
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
