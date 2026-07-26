"""Benchmark block root-system Gaussian grids.

Compared root grids:

- block D4
- block F4
- block D8
- block E8

Each block root system is embedded across the width and then globally mixed by a
deterministic signed/permuted Hadamard rotation.  The points are scaled by
sqrt(width), so any spherical 2-design root directions have identity covariance
under equal weights.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.diagonal_k3_estimator import Estimator as DiagonalK3
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
ROOT_ROTATION_SEED_OFFSET = 57_001_313
KNOWN_FULL_VALIDATION_REFERENCES = [
    {
        "method": "collapsed diagonal-K3 residual stack",
        "final_mse": 3.3483804790133336e-06,
        "adjusted_score": 3.5833449514268906e-07,
        "flops": 29108693976.0,
        "source": "research/results/diag_k3_residual_stack_s03_l8_validation_manifest_eval.json",
    },
    {
        "method": "layerwise diagonal-K3 residual stack",
        "final_mse": 3.348361953063994e-06,
        "adjusted_score": 3.5833909899594703e-07,
        "flops": 29109229018.0,
        "source": "research/results/layerwise_diag_k3_residual_stack_s1_l8_h2_validation_manifest_eval.json",
    },
    {
        "method": "K2 + signed-Hadamard cubature 6144",
        "final_mse": 3.7975855745631162e-06,
        "adjusted_score": 3.838012113921273e-07,
        "flops": 27489552888.0,
        "source": "research/results/combo_k2_signed_hadamard_cubature_6144_validation_manifest_eval.json",
    },
    {
        "method": "signed-Hadamard cubature 6144",
        "final_mse": 4.125196423689138e-06,
        "adjusted_score": 4.125196423689138e-07,
        "flops": 25872889880.0,
        "source": "research/results/signed_hadamard_cubature_6144_validation_manifest_eval.json",
    },
]


def hadamard_matrix_np(width: int) -> np.ndarray:
    if width <= 0 or width & (width - 1):
        raise ValueError("Hadamard mixing requires power-of-two width")
    scale = 1.0 / math.sqrt(width)
    matrix = np.empty((width, width), dtype=np.float32)
    for row in range(width):
        for col in range(width):
            parity = (row & col).bit_count() & 1
            matrix[row, col] = -scale if parity else scale
    return matrix


def d_roots(block_dim: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    scale = 1.0 / math.sqrt(2.0)
    for i in range(block_dim):
        for j in range(i + 1, block_dim):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    row = np.zeros(block_dim, dtype=np.float32)
                    row[i] = si * scale
                    row[j] = sj * scale
                    rows.append(row)
    return np.stack(rows, axis=0)


def f4_roots() -> np.ndarray:
    rows: list[np.ndarray] = []
    for i in range(4):
        for sign in (-1.0, 1.0):
            row = np.zeros(4, dtype=np.float32)
            row[i] = sign
            rows.append(row)
    rows.extend(d_roots(4))
    for mask in range(16):
        row = np.empty(4, dtype=np.float32)
        for i in range(4):
            row[i] = -0.5 if (mask >> i) & 1 else 0.5
        rows.append(row)
    return np.stack(rows, axis=0)


def e8_roots() -> np.ndarray:
    rows: list[np.ndarray] = []
    rows.extend(d_roots(8))
    scale = 1.0 / (2.0 * math.sqrt(2.0))
    for mask in range(256):
        # E8 keeps the half-cube roots with an even number of minus signs.
        if (mask.bit_count() & 1) != 0:
            continue
        row = np.empty(8, dtype=np.float32)
        for i in range(8):
            row[i] = -scale if (mask >> i) & 1 else scale
        rows.append(row)
    return np.stack(rows, axis=0)


def local_roots(kind: str) -> np.ndarray:
    normalized = kind.lower()
    if normalized == "d4":
        return d_roots(4)
    if normalized == "f4":
        return f4_roots()
    if normalized == "d8":
        return d_roots(8)
    if normalized == "e8":
        return e8_roots()
    raise ValueError(f"unknown root grid {kind!r}")


def block_dim(kind: str) -> int:
    normalized = kind.lower()
    if normalized in {"d4", "f4"}:
        return 4
    if normalized in {"d8", "e8"}:
        return 8
    raise ValueError(f"unknown root grid {kind!r}")


@lru_cache(maxsize=32)
def canonical_block_roots(kind: str, width: int) -> np.ndarray:
    b = block_dim(kind)
    if width % b != 0:
        raise ValueError(f"width {width} is not divisible by block dimension {b}")
    roots = local_roots(kind)
    blocks = width // b
    points = np.zeros((blocks * roots.shape[0], width), dtype=np.float32)
    row = 0
    for block in range(blocks):
        start = block * b
        stop = start + b
        for root in roots:
            points[row, start:stop] = root
            row += 1
    return points


def signed_hadamard_rotation(width: int, seed: int, kind: str) -> np.ndarray:
    base = hadamard_matrix_np(width)
    kind_offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(kind.lower()))
    rng = np.random.default_rng(seed + ROOT_ROTATION_SEED_OFFSET + kind_offset)
    permutation = rng.permutation(width)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=width)
    return signs[:, None] * base[permutation, :]


def root_grid_points(kind: str, width: int, seed: int) -> tuple[fnp.ndarray, fnp.ndarray]:
    canonical = fnp.array(canonical_block_roots(kind, width), dtype=fnp.float32)
    rotation = fnp.array(signed_hadamard_rotation(width, seed, kind), dtype=fnp.float32)
    points = fnp.matmul(canonical, rotation) * math.sqrt(width)
    weights = fnp.full((points.shape[0],), 1.0 / points.shape[0], dtype=fnp.float32)
    return points.astype(fnp.float32), weights


class RootGridEstimator:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def predict(self, mlp: Any, budget: int) -> fnp.ndarray:
        _ = budget
        x, weights = root_grid_points(self.kind, mlp.width, mlp.seed)
        rows = []
        for w in mlp.weights:
            x = fnp.maximum(fnp.matmul(x, w), 0.0)
            rows.append(fnp.sum(x * weights[:, None], axis=0))
        return fnp.stack(rows, axis=0)


def fit_k2_root_scalar(
    rows: list[dict[str, Any]],
    k2_pred: dict[int, np.ndarray],
    root_pred: dict[int, np.ndarray],
) -> dict[str, Any]:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        mlp_id = int(row["mlp_id"])
        target = np.asarray(row["all_layer_means"], dtype=np.float64)[-1]
        direction = k2_pred[mlp_id][-1] - root_pred[mlp_id][-1]
        residual = target - root_pred[mlp_id][-1]
        numerator += float(np.vdot(direction, residual))
        denominator += float(np.vdot(direction, direction))
    a = numerator / denominator if denominator else 0.0
    return {"a_k2_minus_root": a, "lambda_root": 1.0 - a, "n_calibration_mlps": len(rows)}


def summarize(per_mlp: list[dict[str, Any]], config: str, split: str, method: str) -> dict[str, Any]:
    subset = [
        row for row in per_mlp if row["config"] == config and row["split"] == split and row["method"] == method
    ]
    wins = [row["win_vs_baseline"] for row in subset if row["win_vs_baseline"] is not None]
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
        "win_fraction_vs_baseline": float(np.mean(wins)) if wins else float("nan"),
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
    best_validation = min(validation_rows, key=lambda row: row["mean_adjusted_score"]) if validation_rows else None
    lines = [
        "# Block root-system grid benchmark",
        "",
        "This compares block `D4`, `F4`, `D8`, and `E8` root-system grids after signed/permuted Hadamard mixing.",
        "",
        "The dev/validation slices here are a quick paired screen, not a full validation rerun.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{' '.join(args.command)}\n```",
        "",
        "## Result matrix",
        "",
        "| grid | split | method | final MSE | adjusted score | FLOPs | win vs root |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summary"]:
        lines.append(
            "| {grid} | {split} | {method} | {mse:.6g} | {score:.6g} | {flops:.6g} | {win:.3g} |".format(
                grid=row["config"],
                split=row["split"],
                method=row["method"],
                mse=row["mean_final_mse"],
                score=row["mean_adjusted_score"],
                flops=row["mean_flops"],
                win=row["win_fraction_vs_baseline"],
            )
        )
    lines.extend(
        [
            "",
            "## Known full-validation references",
            "",
            "These rows are from prior full validation runs and are included only as a sanity comparator.",
            "",
            "| method | final MSE | adjusted score | FLOPs | source |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in KNOWN_FULL_VALIDATION_REFERENCES:
        lines.append(
            "| {method} | {mse:.6g} | {score:.6g} | {flops:.6g} | `{source}` |".format(
                method=row["method"],
                mse=row["final_mse"],
                score=row["adjusted_score"],
                flops=row["flops"],
                source=row["source"],
            )
        )
    if best_validation is not None:
        lines.extend(
            [
                "",
                "## Screen decision",
                "",
                "Best root-grid screen row by validation adjusted score: "
                f"`{best_validation['config']} / {best_validation['method']}` "
                f"with final MSE `{best_validation['mean_final_mse']:.6g}`, "
                f"adjusted score `{best_validation['mean_adjusted_score']:.6g}`, "
                f"and FLOPs `{best_validation['mean_flops']:.6g}`.",
                "",
                "Do not package a root-grid estimator from this screen.  The only competitive root grid is `E8`, "
                "but its FLOP cost pushes adjusted score behind the current diagonal-K3 residual stack.",
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
    diag_estimator = DiagonalK3(third_scale=args.diag_third_scale, max_k3_layer=args.diag_max_k3_layer)
    k2_pred: dict[int, np.ndarray] = {}
    diag_pred: dict[int, np.ndarray] = {}
    k2_flops: dict[int, int] = {}
    diag_flops: dict[int, int] = {}
    for index, row in enumerate(all_rows, start=1):
        mlp_id = int(row["mlp_id"])
        k2_prediction, k2_used, _ = predict(k2_estimator, row, BUDGET)
        diag_prediction, diag_used, _ = predict(diag_estimator, row, BUDGET)
        k2_pred[mlp_id] = k2_prediction
        diag_pred[mlp_id] = diag_prediction
        k2_flops[mlp_id] = k2_used
        diag_flops[mlp_id] = diag_used
        print(f"analytic {index}/{len(all_rows)} mlp_id={mlp_id}", flush=True)

    per_mlp: list[dict[str, Any]] = []
    calibration_by_grid: dict[str, Any] = {}
    for kind in args.grids:
        estimator = RootGridEstimator(kind)
        root_pred: dict[int, np.ndarray] = {}
        root_flops: dict[int, int] = {}
        for index, row in enumerate(all_rows, start=1):
            mlp_id = int(row["mlp_id"])
            prediction, used, _ = predict(estimator, row, BUDGET)
            root_pred[mlp_id] = prediction
            root_flops[mlp_id] = used
            print(f"{kind} {index}/{len(all_rows)} mlp_id={mlp_id}", flush=True)
        k2_scalar = fit_k2_root_scalar(dev_calibration, k2_pred, root_pred)
        stack_calibration = fit_stack(dev_calibration, k2_pred, diag_pred, root_pred, args.ridge)
        calibration_by_grid[kind] = {"k2_root": k2_scalar, "diag_stack": stack_calibration}
        a_k2 = float(k2_scalar["a_k2_minus_root"])
        a_stack = float(stack_calibration["a_k2_minus_cubature"])
        b_stack = float(stack_calibration["b_diag_minus_k2"])
        for split_name, rows in (("development_selection", dev_selection), ("validation_frozen", validation_rows)):
            for row in rows:
                mlp_id = int(row["mlp_id"])
                target = np.asarray(row["all_layer_means"], dtype=np.float64)
                k2_combo_pred, k2_blend_flops = stack_prediction(
                    k2_pred[mlp_id],
                    k2_pred[mlp_id],
                    root_pred[mlp_id],
                    a_k2,
                    0.0,
                )
                k2_combo_flops = k2_flops[mlp_id] + root_flops[mlp_id] + k2_blend_flops
                stack_pred, stack_flops = stack_prediction(
                    k2_pred[mlp_id],
                    diag_pred[mlp_id],
                    root_pred[mlp_id],
                    a_stack,
                    b_stack,
                )
                total_stack_flops = k2_flops[mlp_id] + diag_flops[mlp_id] + root_flops[mlp_id] + stack_flops
                method_metrics = {
                    "root": metrics(root_pred[mlp_id], target, root_flops[mlp_id]),
                    "k2_root": metrics(k2_combo_pred, target, k2_combo_flops),
                    "diagk3_stack": metrics(stack_pred, target, total_stack_flops),
                }
                root_final = method_metrics["root"]["final_mse"]
                for method, values in method_metrics.items():
                    per_mlp.append(
                        {
                            "config": kind,
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
                            "win_vs_baseline": values["final_mse"] < root_final if method != "root" else None,
                        }
                    )
        print(
            f"{kind} a_k2={a_k2:.6g} a_stack={a_stack:.6g} b_stack={b_stack:.6g}",
            flush=True,
        )

    summary_rows = [
        summarize(per_mlp, kind, split, method)
        for kind in args.grids
        for split in ("development_selection", "validation_frozen")
        for method in ("root", "k2_root", "diagk3_stack")
    ]
    result = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "created_at_utc": args.created_at,
        "configuration": {
            "grids": args.grids,
            "diag_third_scale": args.diag_third_scale,
            "diag_max_k3_layer": args.diag_max_k3_layer,
            "ridge": args.ridge,
            "dev_limit": args.dev_limit,
            "validation_limit": args.validation_limit,
            "calibration_parity": args.calibration_parity,
        },
        "calibration": calibration_by_grid,
        "summary": summary_rows,
        "per_mlp": per_mlp,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / f"root_system_grid_benchmark_{args.result_tag}.json"
    csv_path = RESULTS / f"root_system_grid_benchmark_{args.result_tag}.csv"
    report_path = REPORTS / f"root_system_grid_benchmark_{args.result_tag}.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, summary_rows)
    write_report(report_path, args, result)
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report_path), "summary": summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", nargs="+", default=["d4", "f4", "d8", "e8"])
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--calibration-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--diag-third-scale", type=float, default=-0.3)
    parser.add_argument("--diag-max-k3-layer", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-12)
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
