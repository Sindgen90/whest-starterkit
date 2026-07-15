"""Compare Gaussian sample grids inside a corrected-K2 convex combination."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
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

DEFAULT_N_SAMPLES = 4096
DEFAULT_RQMC_SEED_OFFSET = 8_675_309
DEFAULT_RQMC_MULTIPLIER = 17_797
DEFAULT_SOBOL_DIRECTION_FILE = Path("C:/tmp/new-joe-kuo-6.21201")
SOBOL_BITS = 32
SOBOL_SCALE = float(1 << SOBOL_BITS)


def lattice_samples(n_samples: int, width: int, seed: int, seed_offset: int, multiplier: int) -> fnp.ndarray:
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


def load_sobol_direction_rows(path: Path, width: int) -> dict[int, tuple[int, int, list[int]]]:
    rows: dict[int, tuple[int, int, list[int]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            dimension = int(parts[0])
            if dimension > width:
                break
            degree = int(parts[1])
            coefficient = int(parts[2])
            initial = [int(value) for value in parts[3:]]
            rows[dimension] = (degree, coefficient, initial)
    missing = [dimension for dimension in range(2, width + 1) if dimension not in rows]
    if missing:
        raise ValueError(f"Sobol direction file missing dimensions: {missing[:5]}")
    return rows


def sobol_direction_vectors(width: int, bits: int, rows: dict[int, tuple[int, int, list[int]]]) -> list[list[int]]:
    directions: list[list[int]] = []
    directions.append([1 << (bits - bit) for bit in range(1, bits + 1)])
    for dimension in range(2, width + 1):
        degree, coefficient, initial = rows[dimension]
        v = [0] * bits
        for bit in range(1, degree + 1):
            v[bit - 1] = initial[bit - 1] << (bits - bit)
        for bit in range(degree + 1, bits + 1):
            value = v[bit - degree - 1] ^ (v[bit - degree - 1] >> degree)
            for k in range(1, degree):
                if (coefficient >> (degree - 1 - k)) & 1:
                    value ^= v[bit - k - 1]
            v[bit - 1] = value
        directions.append(v)
    return directions


def sobol_samples(
    n_samples: int,
    width: int,
    seed: int,
    seed_offset: int,
    direction_file: Path,
) -> fnp.ndarray:
    if n_samples <= 0 or n_samples & (n_samples - 1):
        raise ValueError("Sobol sample count must be a positive power of two")
    if n_samples > (1 << SOBOL_BITS):
        raise ValueError(f"Sobol sample count exceeds {SOBOL_BITS}-bit implementation")
    rows = load_sobol_direction_rows(direction_file, width)
    directions = sobol_direction_vectors(width, SOBOL_BITS, rows)
    scramble = [random.Random(seed + seed_offset + 104729 * dim).getrandbits(SOBOL_BITS) for dim in range(width)]
    current = [0] * width
    samples: list[list[float]] = []
    for index in range(n_samples):
        if index:
            bit = (index & -index).bit_length()
            for dim in range(width):
                current[dim] ^= directions[dim][bit - 1]
        samples.append([
            min(1.0 - 1e-7, max(1e-7, ((current[dim] ^ scramble[dim]) + 0.5) / SOBOL_SCALE))
            for dim in range(width)
        ])
    u = fnp.array(samples, dtype=fnp.float64)
    return flops.stats.norm.ppf(u).astype(fnp.float32)


def orthogonal_gaussian_samples(n_samples: int, width: int, seed: int, seed_offset: int) -> fnp.ndarray:
    if n_samples % width:
        raise ValueError("--n-samples must be a multiple of width for orthogonal Gaussian sampling")
    rng = fnp.random.default_rng(seed + seed_offset)
    blocks = []
    for _ in range(n_samples // width):
        gaussian = rng.standard_normal((width, width)).astype(fnp.float64)
        q, r = fnp.linalg.qr(gaussian)
        signs = fnp.where(fnp.diag(r) >= 0.0, 1.0, -1.0)
        directions = (q * signs).T
        radius_noise = rng.standard_normal((width, width)).astype(fnp.float64)
        radii = fnp.sqrt(fnp.sum(radius_noise * radius_noise, axis=1))
        blocks.append((directions * radii[:, None]).astype(fnp.float32))
    return fnp.concatenate(blocks, axis=0)


class GridEstimator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def predict(self, mlp: Any, budget: int) -> fnp.ndarray:
        _ = budget
        if self.args.grid == "lattice":
            x = lattice_samples(
                self.args.n_samples, mlp.width, mlp.seed, self.args.seed_offset, self.args.multiplier
            )
        elif self.args.grid == "sobol":
            x = sobol_samples(
                self.args.n_samples, mlp.width, mlp.seed, self.args.seed_offset, self.args.sobol_direction_file
            )
        elif self.args.grid == "orthogonal":
            x = orthogonal_gaussian_samples(self.args.n_samples, mlp.width, mlp.seed, self.args.seed_offset)
        else:
            raise ValueError(f"unknown grid: {self.args.grid}")
        rows = []
        for w in mlp.weights:
            x = fnp.maximum(fnp.matmul(x, w), 0.0)
            rows.append(fnp.mean(x, axis=0))
        return fnp.stack(rows, axis=0)


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


def fit_lambda(rows: list[dict[str, Any]], k2: Any, grid: Any, args: argparse.Namespace) -> dict[str, Any]:
    numerator = 0.0
    denominator = 0.0
    records = []
    for row in rows:
        k2_prediction, _, _ = predict(k2, row, args.budget)
        grid_prediction, _, _ = predict(grid, row, args.budget)
        target = np.asarray(row["all_layer_means"], dtype=np.float64)
        delta = grid_prediction[-1] - k2_prediction[-1]
        residual = k2_prediction[-1] - target[-1]
        numerator += float(np.vdot(delta, residual))
        denominator += float(np.vdot(delta, delta))
        records.append({
            "mlp_id": int(row["mlp_id"]),
            "mlp_name": str(row["mlp_name"]),
            "corrected_k2_final_mse": metrics(k2_prediction, target, args.prefixes)["final_mse"],
            "grid_final_mse": metrics(grid_prediction, target, args.prefixes)["final_mse"],
        })
    raw_lambda = -numerator / denominator if denominator else 0.0
    return {
        "objective": "final-layer least squares",
        "formula": "argmin_lambda ||k2 + lambda * (grid - k2) - target||^2",
        "raw_lambda": raw_lambda,
        "lambda": min(1.0, max(0.0, raw_lambda)),
        "n_mlps": len(rows),
        "selector": f"mlp_id % 2 == {args.calibration_parity}",
        "records": records,
    }


def evaluate_row(row: dict[str, Any], k2: Any, grid: Any, lambda_: float, args: argparse.Namespace) -> dict[str, Any]:
    target = np.asarray(row["all_layer_means"], dtype=np.float64)
    k2_prediction, k2_flops, k2_runtime = predict(k2, row, args.budget)
    grid_prediction, grid_flops, grid_runtime = predict(grid, row, args.budget)

    combo_started = time.perf_counter()
    mlp = as_mlp(row)
    with flops.BudgetContext(flop_budget=args.budget, quiet=True) as context:
        combo_k2 = k2.predict(mlp, args.budget)
        combo_grid = grid.predict(mlp, args.budget)
        combo_prediction = fnp.add(fnp.multiply(combo_k2, 1.0 - lambda_), fnp.multiply(combo_grid, lambda_))
    combo_runtime = time.perf_counter() - combo_started
    combo_prediction_np = np.asarray(combo_prediction, dtype=np.float64)

    k2_metrics = metrics(k2_prediction, target, args.prefixes)
    grid_metrics = metrics(grid_prediction, target, args.prefixes)
    combo_metrics = metrics(combo_prediction_np, target, args.prefixes)
    combo_flops = int(context.flops_used)
    return {
        "mlp_id": int(row["mlp_id"]),
        "mlp_name": str(row["mlp_name"]),
        "mlp_seed": int(row["mlp_seed"]),
        "status": "completed",
        "metrics": {"corrected_k2": k2_metrics, "grid": grid_metrics, "combo": combo_metrics},
        "paired_vs_corrected_k2": {
            "combo_final_mse_delta": combo_metrics["final_mse"] - k2_metrics["final_mse"],
            "combo_final_mse_ratio": combo_metrics["final_mse"] / k2_metrics["final_mse"],
            "grid_final_mse_delta": grid_metrics["final_mse"] - k2_metrics["final_mse"],
            "grid_final_mse_ratio": grid_metrics["final_mse"] / k2_metrics["final_mse"],
        },
        "compute": {
            "corrected_k2_flops": k2_flops,
            "grid_flops": grid_flops,
            "combo_flops": combo_flops,
            "flop_budget": args.budget,
            "corrected_k2_runtime_s": k2_runtime,
            "grid_runtime_s": grid_runtime,
            "combo_runtime_s": combo_runtime,
            "combo_flop_fraction": combo_flops / args.budget,
            "combo_adjusted_final_score": combo_metrics["final_mse"] * max(SCORE_FLOOR, combo_flops / args.budget),
        },
    }


def summarize(records: list[dict[str, Any]], prefixes: Iterable[int]) -> dict[str, Any]:
    done = [record for record in records if record["status"] == "completed"]
    combo_wins = [record["paired_vs_corrected_k2"]["combo_final_mse_delta"] < 0.0 for record in done]
    grid_wins = [record["paired_vs_corrected_k2"]["grid_final_mse_delta"] < 0.0 for record in done]
    summary = {
        "n_requested": len(records),
        "n_completed": len(done),
        "n_failed": len(records) - len(done),
        "corrected_k2_final_mse": aggregate(r["metrics"]["corrected_k2"]["final_mse"] for r in done),
        "grid_final_mse": aggregate(r["metrics"]["grid"]["final_mse"] for r in done),
        "combo_final_mse": aggregate(r["metrics"]["combo"]["final_mse"] for r in done),
        "corrected_k2_all_layer_mse": aggregate(r["metrics"]["corrected_k2"]["all_layer_mse"] for r in done),
        "grid_all_layer_mse": aggregate(r["metrics"]["grid"]["all_layer_mse"] for r in done),
        "combo_all_layer_mse": aggregate(r["metrics"]["combo"]["all_layer_mse"] for r in done),
        "combo_adjusted_final_score": aggregate(r["compute"]["combo_adjusted_final_score"] for r in done),
        "combo_flops": aggregate(r["compute"]["combo_flops"] for r in done),
        "combo_runtime_s": aggregate(r["compute"]["combo_runtime_s"] for r in done),
        "paired_vs_corrected_k2": {
            "combo_win_fraction": sum(combo_wins) / len(combo_wins) if combo_wins else None,
            "grid_win_fraction": sum(grid_wins) / len(grid_wins) if grid_wins else None,
            "combo_final_mse_delta": aggregate(r["paired_vs_corrected_k2"]["combo_final_mse_delta"] for r in done),
            "combo_final_mse_ratio": aggregate(r["paired_vs_corrected_k2"]["combo_final_mse_ratio"] for r in done),
            "grid_final_mse_delta": aggregate(r["paired_vs_corrected_k2"]["grid_final_mse_delta"] for r in done),
            "grid_final_mse_ratio": aggregate(r["paired_vs_corrected_k2"]["grid_final_mse_ratio"] for r in done),
        },
        "prefix_mse": {},
    }
    for prefix in prefixes:
        summary["prefix_mse"][str(prefix)] = {
            "corrected_k2": aggregate(r["metrics"]["corrected_k2"]["prefix_mse"][str(prefix)] for r in done),
            "grid": aggregate(r["metrics"]["grid"]["prefix_mse"][str(prefix)] for r in done),
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
    grid = GridEstimator(args)
    calibration = fit_lambda(calibration_rows, k2, grid, args)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(evaluation_rows, start=1):
        try:
            records.append(evaluate_row(row, k2, grid, calibration["lambda"], args))
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
        "experiment": "grid_corrected_k2_convex_combo",
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
            "grid": args.grid,
            "n_samples": args.n_samples,
            "seed_offset": args.seed_offset,
            "multiplier": args.multiplier,
            "sobol_direction_file": str(args.sobol_direction_file) if args.grid == "sobol" else None,
            "sobol_direction_sha256": (
                hashlib.sha256(args.sobol_direction_file.read_bytes()).hexdigest()
                if args.grid == "sobol"
                else None
            ),
        },
        "calibration": calibration,
        "source": {
            "corrected_k2_path": str(CORRECTED_K2.relative_to(ROOT)),
            "corrected_k2_sha256": hashlib.sha256(CORRECTED_K2.read_bytes()).hexdigest(),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--porcelain")),
        },
        "summary": summarize(records, args.prefixes),
        "per_mlp": records,
        "failures": failures,
    }
    RESULTS.mkdir(exist_ok=True)
    suffix = "mini" if args.limit_mlps is None else f"mini_limit{args.limit_mlps}"
    suffix = f"{suffix}_grid_{args.grid}_n{args.n_samples}_seed{args.seed_offset}"
    if args.grid == "lattice":
        suffix = f"{suffix}_mult{args.multiplier}"
    output = RESULTS / f"{suffix}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=("lattice", "sobol", "orthogonal"), required=True)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--prefixes", type=int, nargs="+", default=list(PREFIXES))
    parser.add_argument("--calibration-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--limit-mlps", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--seed-offset", type=int, default=DEFAULT_RQMC_SEED_OFFSET)
    parser.add_argument("--multiplier", type=int, default=DEFAULT_RQMC_MULTIPLIER)
    parser.add_argument("--sobol-direction-file", type=Path, default=DEFAULT_SOBOL_DIRECTION_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_mlps is not None and args.limit_mlps <= 1:
        raise SystemExit("--limit-mlps must be greater than 1")
    if args.n_samples <= 0:
        raise SystemExit("--n-samples must be positive")
    if args.grid == "sobol" and not args.sobol_direction_file.exists():
        raise SystemExit(f"Sobol direction file not found: {args.sobol_direction_file}")
    if args.grid == "lattice" and (args.multiplier <= 0 or args.multiplier % 2 == 0):
        raise SystemExit("--multiplier must be positive and odd")
    output = run(args)
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
