"""Calibrate corrected-K2 + signed-Hadamard-cubature convex blend."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
import whestbench

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "research" / "configs" / "stein_holdout_manifest.json"
RESULTS = ROOT / "research" / "results"
REPORTS = ROOT / "research" / "reports"
CORRECTED_K2 = ROOT / "submissions" / "corrected_k2" / "estimator.py"
CUBATURE = ROOT / "submissions" / "signed_hadamard_cubature_6144" / "estimator.py"
BUDGET = 272_000_000_000
SCORE_FLOOR = 0.1


def load_estimator(path: Path) -> type[Any]:
    module_name = f"combo_source_{path.parent.name}_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import estimator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Estimator


def as_mlp(row: dict[str, Any]) -> SimpleNamespace:
    stacked = fnp.array(np.asarray(row["weights"], dtype=np.float32))
    depth, width, _ = stacked.shape
    return SimpleNamespace(
        width=int(width),
        depth=int(depth),
        weights=[stacked[layer] for layer in range(depth)],
        seed=int(row["mlp_seed"]),
        name=str(row["mlp_name"]),
    )


def select_rows(partition: str, limit: int | None) -> list[dict[str, Any]]:
    if partition == "holdout":
        raise SystemExit("Holdout is gated for this combo calibration.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = list(
        whestbench.load_dataset(
            manifest["dataset"],
            revision=manifest["dataset_revision"],
            split=manifest["official_split"],
        )
    )
    ids = set(int(x) for x in manifest[f"{partition}_mlp_ids"])
    selected = sorted([row for row in rows if int(row["mlp_id"]) in ids], key=lambda row: int(row["mlp_id"]))
    if limit is not None:
        selected = selected[:limit]
    return selected


def predict(estimator: Any, row: dict[str, Any], budget: int) -> tuple[np.ndarray, int, float]:
    mlp = as_mlp(row)
    start = time.perf_counter()
    with flops.BudgetContext(flop_budget=budget, quiet=True) as context:
        prediction = estimator.predict(mlp, budget)
    runtime = time.perf_counter() - start
    prediction_np = np.asarray(prediction, dtype=np.float64)
    if prediction_np.shape != (mlp.depth, mlp.width):
        raise ValueError(f"bad prediction shape {prediction_np.shape}; expected {(mlp.depth, mlp.width)}")
    if not np.all(np.isfinite(prediction_np)):
        raise ValueError("prediction contains non-finite values")
    return prediction_np, int(context.flops_used), runtime


def fit_lambda(
    rows: list[dict[str, Any]],
    k2: dict[int, np.ndarray],
    cubature: dict[int, np.ndarray],
) -> dict[str, float | int | str]:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        mlp_id = int(row["mlp_id"])
        target = np.asarray(row["all_layer_means"], dtype=np.float64)
        delta = cubature[mlp_id][-1] - k2[mlp_id][-1]
        residual = k2[mlp_id][-1] - target[-1]
        numerator += float(np.vdot(delta, residual))
        denominator += float(np.vdot(delta, delta))
    raw_lambda = -numerator / denominator if denominator else 0.0
    return {
        "objective": "final-layer least squares on development calibration rows",
        "raw_lambda": raw_lambda,
        "lambda": min(1.0, max(0.0, raw_lambda)),
        "n_calibration_mlps": len(rows),
    }


def metrics(prediction: np.ndarray, target: np.ndarray, flops_used: int) -> dict[str, float]:
    err = (prediction - target) ** 2
    final = float(err[-1].mean())
    return {
        "final_mse": final,
        "all_layer_mse": float(err.mean()),
        "adjusted_score": final * max(SCORE_FLOOR, flops_used / BUDGET),
        "flops": float(flops_used),
        "budget_fraction": flops_used / BUDGET,
    }


def blend(k2: np.ndarray, cubature: np.ndarray, lambda_: float) -> tuple[np.ndarray, int]:
    with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as context:
        prediction = fnp.add(
            fnp.multiply(fnp.array(k2), 1.0 - lambda_),
            fnp.multiply(fnp.array(cubature), lambda_),
        )
    return np.asarray(prediction, dtype=np.float64), int(context.flops_used)


def aggregate(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    worst_count = max(1, math.ceil(len(arr) / 10))
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "standard_error": float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "worst_decile": float(np.sort(arr)[-worst_count:].mean()),
        "max": float(arr.max()),
    }


def evaluate_split(
    name: str,
    rows: list[dict[str, Any]],
    lambda_: float,
    k2_pred: dict[int, np.ndarray],
    k2_flops: dict[int, int],
    cub_pred: dict[int, np.ndarray],
    cub_flops: dict[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_mlp = []
    summary_rows = []
    for row in rows:
        mlp_id = int(row["mlp_id"])
        target = np.asarray(row["all_layer_means"], dtype=np.float64)
        combo_pred, blend_flops = blend(k2_pred[mlp_id], cub_pred[mlp_id], lambda_)
        combo_flops = k2_flops[mlp_id] + cub_flops[mlp_id] + blend_flops
        row_metrics = {
            "k2": metrics(k2_pred[mlp_id], target, k2_flops[mlp_id]),
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
                "combo_win_vs_k2": row_metrics["combo"]["final_mse"] < row_metrics["k2"]["final_mse"],
            }
        )
    for method in ("k2", "cubature", "combo"):
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


def write_report(path: Path, args: argparse.Namespace, calibration: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Corrected K2 + signed Hadamard cubature combo",
        "",
        "This tests whether corrected K2 adds enough bias correction to justify its extra FLOPs on top of the 6144-point cubature estimator.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{' '.join(args.command)}\n```",
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
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "Promote the combo only if validation adjusted score improves over raw signed Hadamard cubature.  Because K2 pushes compute above the 10% floor, a tiny raw-MSE improvement is not enough.",
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
        print(f"{index}/{len(all_rows)} mlp_id={mlp_id} predicted", flush=True)
    calibration = fit_lambda(dev_calibration, k2_pred, cub_pred)
    dev_per_mlp, dev_summary = evaluate_split(
        "development_selection", dev_selection, float(calibration["lambda"]), k2_pred, k2_flops, cub_pred, cub_flops
    )
    val_per_mlp, val_summary = evaluate_split(
        "validation_frozen_lambda", validation_rows, float(calibration["lambda"]), k2_pred, k2_flops, cub_pred, cub_flops
    )
    summary_rows = [*dev_summary, *val_summary]
    result = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "created_at_utc": args.created_at,
        "calibration": calibration,
        "summary": summary_rows,
        "per_mlp": [*dev_per_mlp, *val_per_mlp],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "k2_cubature_combo_result.json"
    csv_path = RESULTS / "k2_cubature_combo_summary.csv"
    report_path = REPORTS / "k2_cubature_combo_report.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, summary_rows)
    write_report(report_path, args, calibration, summary_rows)
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path), "summary": summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--calibration-parity", type=int, choices=(0, 1), default=0)
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
