"""Compare same-envelope recalibrations of diagonal-K3 residual stacks.

This is a narrow follow-up to the collapsed-vs-layerwise diagonal-K3 tie.
It keeps the mathematical ingredients fixed:

    cubature + a * (corrected_k2 - cubature) + b * (diag_family - corrected_k2)

and only changes the calibration rows/objective for ``a`` and ``b``.  Each
candidate uses exactly one K3 residual family at evaluation time, so the
submission FLOP envelope stays comparable to the current diagonal-K3 stack.
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

from experiments.diagonal_k3_estimator import Estimator as CollapsedDiagonalK3
from experiments.layerwise_diagonal_k3_estimator import Estimator as LayerwiseDiagonalK3
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

REFERENCE_ROWS = [
    {
        "candidate": "current collapsed package",
        "validation_final_mse": 3.3483804790133336e-06,
        "validation_adjusted_score": 3.5833449514268906e-07,
        "flops": 29108693976.0,
    },
    {
        "candidate": "raw-MSE layerwise package",
        "validation_final_mse": 3.348361953063994e-06,
        "validation_adjusted_score": 3.5833909899594703e-07,
        "flops": 29109229018.0,
    },
]


def fit_stack_on_layers(
    rows: list[dict[str, Any]],
    k2_pred: dict[int, np.ndarray],
    diag_pred: dict[int, np.ndarray],
    cub_pred: dict[int, np.ndarray],
    layer_mode: str,
    ridge: float,
) -> dict[str, Any]:
    gram = np.zeros((2, 2), dtype=np.float64)
    rhs = np.zeros(2, dtype=np.float64)
    for row in rows:
        mlp_id = int(row["mlp_id"])
        target = np.asarray(row["all_layer_means"], dtype=np.float64)
        if layer_mode == "final":
            layer_indices = [target.shape[0] - 1]
        elif layer_mode == "all":
            layer_indices = list(range(target.shape[0]))
        elif layer_mode == "late_16_24_32":
            layer_indices = [idx for idx in (15, 23, 31) if idx < target.shape[0]]
        else:
            raise ValueError(f"unknown layer_mode {layer_mode!r}")
        for layer_index in layer_indices:
            d1 = k2_pred[mlp_id][layer_index] - cub_pred[mlp_id][layer_index]
            d2 = diag_pred[mlp_id][layer_index] - k2_pred[mlp_id][layer_index]
            residual = target[layer_index] - cub_pred[mlp_id][layer_index]
            design = np.stack([d1, d2], axis=1)
            gram += design.T @ design
            rhs += design.T @ residual
    coef = np.linalg.solve(gram + ridge * np.eye(2), rhs)
    return {
        "a_k2_minus_cubature": float(coef[0]),
        "b_diag_minus_k2": float(coef[1]),
        "ridge": ridge,
        "layer_mode": layer_mode,
        "n_calibration_mlps": len(rows),
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


def calibration_specs(dev_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    even = [row for row in dev_rows if int(row["mlp_id"]) % 2 == 0]
    odd = [row for row in dev_rows if int(row["mlp_id"]) % 2 == 1]
    return [
        {"name": "even_final", "rows": even, "layer_mode": "final", "selection": odd},
        {"name": "odd_final", "rows": odd, "layer_mode": "final", "selection": even},
        {"name": "all_dev_final", "rows": dev_rows, "layer_mode": "final", "selection": dev_rows},
        {"name": "all_dev_all_layers", "rows": dev_rows, "layer_mode": "all", "selection": dev_rows},
        {"name": "all_dev_late_16_24_32", "rows": dev_rows, "layer_mode": "late_16_24_32", "selection": dev_rows},
    ]


def summarize(per_mlp: list[dict[str, Any]], family: str, calibration: str, split: str) -> dict[str, Any]:
    subset = [
        row
        for row in per_mlp
        if row["family"] == family and row["calibration"] == calibration and row["split"] == split
    ]
    wins = [row["win_vs_current_collapsed"] for row in subset]
    return {
        "family": family,
        "calibration": calibration,
        "split": split,
        "n_completed": len(subset),
        "mean_final_mse": aggregate([row["final_mse"] for row in subset])["mean"],
        "median_final_mse": aggregate([row["final_mse"] for row in subset])["median"],
        "se_final_mse": aggregate([row["final_mse"] for row in subset])["standard_error"],
        "worst_decile_final_mse": aggregate([row["final_mse"] for row in subset])["worst_decile"],
        "max_final_mse": aggregate([row["final_mse"] for row in subset])["max"],
        "mean_all_layer_mse": aggregate([row["all_layer_mse"] for row in subset])["mean"],
        "mean_adjusted_score": aggregate([row["adjusted_score"] for row in subset])["mean"],
        "mean_flops": aggregate([row["flops"] for row in subset])["mean"],
        "win_fraction_vs_current_collapsed": float(np.mean(wins)) if wins else float("nan"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, args: argparse.Namespace, result: dict[str, Any]) -> None:
    validation_rows = [row for row in result["summary"] if row["split"] == "validation_frozen"]
    best = min(validation_rows, key=lambda row: row["mean_adjusted_score"])
    best_is_existing = best["family"] == "collapsed_s-0.3_L8" and best["calibration"] == "even_final"
    lines = [
        "# Diagonal-K3 residual recalibration comparison",
        "",
        "This keeps the same estimator envelope and only changes residual-family/calibration constants.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{' '.join(args.command)}\n```",
        "",
        "## Result matrix",
        "",
        "| family | calibration | split | final MSE | adjusted score | all-layer MSE | FLOPs | a | b |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summary"]:
        if row["split"] != "validation_frozen":
            continue
        coef = result["calibration"][row["family"]][row["calibration"]]
        lines.append(
            "| {family} | {calib} | {split} | {mse:.8g} | {score:.8g} | {all_mse:.8g} | {flops:.8g} | {a:.8g} | {b:.8g} |".format(
                family=row["family"],
                calib=row["calibration"],
                split=row["split"],
                mse=row["mean_final_mse"],
                score=row["mean_adjusted_score"],
                all_mse=row["mean_all_layer_mse"],
                flops=row["mean_flops"],
                a=coef["a_k2_minus_cubature"],
                b=coef["b_diag_minus_k2"],
            )
        )
    lines.extend(
        [
            "",
            "## Current full-validation references",
            "",
            "| candidate | final MSE | adjusted score | FLOPs |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in REFERENCE_ROWS:
        lines.append(
            "| {candidate} | {mse:.8g} | {score:.8g} | {flops:.8g} |".format(
                candidate=row["candidate"],
                mse=row["validation_final_mse"],
                score=row["validation_adjusted_score"],
                flops=row["flops"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Best recalibration by validation adjusted score: "
            f"`{best['family']} / {best['calibration']}` with final MSE "
            f"`{best['mean_final_mse']:.8g}`, adjusted score `{best['mean_adjusted_score']:.8g}`, "
            f"and FLOPs `{best['mean_flops']:.8g}`.",
        ]
    )
    if best["mean_adjusted_score"] < REFERENCE_ROWS[0]["validation_adjusted_score"]:
        lines.append("")
        if best_is_existing:
            lines.append(
                "This recovers the existing collapsed package calibration.  The tiny adjusted-score difference "
                "is a script/package FLOP-counting detail, not a new estimator.  Do not package a replacement."
            )
        else:
            lines.append("This beats the current collapsed package on adjusted score and should be packaged as a new candidate.")
    else:
        lines.append("")
        lines.append("This does not beat the current collapsed package on adjusted score. Do not package a replacement.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dev_rows = select_rows("development", args.dev_limit)
    validation_rows = select_rows("validation", args.validation_limit)
    all_rows = [*dev_rows, *validation_rows]

    families = {
        "collapsed_s-0.3_L8": CollapsedDiagonalK3(third_scale=-0.3, max_k3_layer=8),
        "layerwise_s-1_L8_H2": LayerwiseDiagonalK3(third_scale=-1.0, max_k3_layer=8, generic_horizon=2),
    }
    k2_estimator = load_estimator(CORRECTED_K2)()
    cub_estimator = load_estimator(CUBATURE)()

    k2_pred: dict[int, np.ndarray] = {}
    cub_pred: dict[int, np.ndarray] = {}
    family_pred: dict[str, dict[int, np.ndarray]] = {name: {} for name in families}
    k2_flops: dict[int, int] = {}
    cub_flops: dict[int, int] = {}
    family_flops: dict[str, dict[int, int]] = {name: {} for name in families}

    for index, row in enumerate(all_rows, start=1):
        mlp_id = int(row["mlp_id"])
        k2_prediction, k2_used, _ = predict(k2_estimator, row, BUDGET)
        cub_prediction, cub_used, _ = predict(cub_estimator, row, BUDGET)
        k2_pred[mlp_id] = k2_prediction
        cub_pred[mlp_id] = cub_prediction
        k2_flops[mlp_id] = k2_used
        cub_flops[mlp_id] = cub_used
        for family_name, estimator in families.items():
            prediction, used, _ = predict(estimator, row, BUDGET)
            family_pred[family_name][mlp_id] = prediction
            family_flops[family_name][mlp_id] = used
        print(f"{index}/{len(all_rows)} mlp_id={mlp_id} predicted", flush=True)

    current_reference: dict[int, np.ndarray] = {}
    current_reference_flops: dict[int, int] = {}
    current_a = args.current_collapsed_a
    current_b = args.current_collapsed_b
    for row in all_rows:
        mlp_id = int(row["mlp_id"])
        pred, blend_flops = stack_prediction(
            k2_pred[mlp_id],
            family_pred["collapsed_s-0.3_L8"][mlp_id],
            cub_pred[mlp_id],
            current_a,
            current_b,
        )
        current_reference[mlp_id] = pred
        current_reference_flops[mlp_id] = (
            k2_flops[mlp_id] + cub_flops[mlp_id] + family_flops["collapsed_s-0.3_L8"][mlp_id] + blend_flops
        )

    specs = calibration_specs(dev_rows)
    calibration: dict[str, dict[str, Any]] = {family: {} for family in families}
    per_mlp: list[dict[str, Any]] = []
    for family_name in families:
        for spec in specs:
            coef = fit_stack_on_layers(
                spec["rows"],
                k2_pred,
                family_pred[family_name],
                cub_pred,
                spec["layer_mode"],
                args.ridge,
            )
            calibration[family_name][spec["name"]] = coef
            a = float(coef["a_k2_minus_cubature"])
            b = float(coef["b_diag_minus_k2"])
            split_rows = {
                "development_selection": spec["selection"],
                "validation_frozen": validation_rows,
            }
            for split_name, rows in split_rows.items():
                for row in rows:
                    mlp_id = int(row["mlp_id"])
                    target = np.asarray(row["all_layer_means"], dtype=np.float64)
                    prediction, blend_flops = stack_prediction(
                        k2_pred[mlp_id],
                        family_pred[family_name][mlp_id],
                        cub_pred[mlp_id],
                        a,
                        b,
                    )
                    total_flops = (
                        k2_flops[mlp_id] + cub_flops[mlp_id] + family_flops[family_name][mlp_id] + blend_flops
                    )
                    values = metrics(prediction, target, total_flops)
                    reference_values = metrics(
                        current_reference[mlp_id],
                        target,
                        current_reference_flops[mlp_id],
                    )
                    per_mlp.append(
                        {
                            "family": family_name,
                            "calibration": spec["name"],
                            "split": split_name,
                            "mlp_id": mlp_id,
                            "mlp_name": str(row["mlp_name"]),
                            "mlp_seed": int(row["mlp_seed"]),
                            "final_mse": values["final_mse"],
                            "all_layer_mse": values["all_layer_mse"],
                            "adjusted_score": values["adjusted_score"],
                            "flops": values["flops"],
                            "budget_fraction": values["budget_fraction"],
                            "win_vs_current_collapsed": values["final_mse"] < reference_values["final_mse"],
                        }
                    )
            print(f"{family_name} {spec['name']} a={a:.8g} b={b:.8g}", flush=True)

    summary = [
        summarize(per_mlp, family_name, spec["name"], split)
        for family_name in families
        for spec in specs
        for split in ("development_selection", "validation_frozen")
    ]
    result = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "created_at_utc": args.created_at,
        "configuration": {
            "dev_limit": args.dev_limit,
            "validation_limit": args.validation_limit,
            "ridge": args.ridge,
            "current_collapsed_a": args.current_collapsed_a,
            "current_collapsed_b": args.current_collapsed_b,
            "families": {
                "collapsed_s-0.3_L8": {"third_scale": -0.3, "max_k3_layer": 8},
                "layerwise_s-1_L8_H2": {"third_scale": -1.0, "max_k3_layer": 8, "generic_horizon": 2},
            },
        },
        "calibration": calibration,
        "summary": summary,
        "per_mlp": per_mlp,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = args.result_tag
    json_path = RESULTS / f"diagonal_k3_recalibration_compare_{tag}.json"
    csv_path = RESULTS / f"diagonal_k3_recalibration_compare_{tag}.csv"
    report_path = REPORTS / f"diagonal_k3_recalibration_compare_{tag}.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, summary)
    write_report(report_path, args, result)
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path), "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=1e-12)
    parser.add_argument("--current-collapsed-a", type=float, default=0.10653087569429524)
    parser.add_argument("--current-collapsed-b", type=float, default=-2.6372834899095565)
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
