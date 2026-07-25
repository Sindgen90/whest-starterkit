"""Benchmark fixed sigma/cubature integration rules against equal-count RQMC.

This is a falsification test for the higher-order unscented-transform idea:
if fixed 512/513-point deterministic rules lose badly to 513-point RQMC when
propagated through the full MLP, blockwise moment rematching is unlikely to be
worth implementing in Phase 1.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
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
BUDGET = 272_000_000_000
SCORE_FLOOR = 0.1
PREFIXES = (2, 4, 8, 16, 24, 32)
RQMC_MULTIPLIER = 17_797
RQMC_SEED_OFFSET = 1_000_003
MC_SEED_OFFSET = 11_000_019
METHODS = (
    "mc_513",
    "rqmc_513",
    "axis_cubature_512",
    "hadamard_cubature_512",
    "signed_hadamard_cubature_1024",
    "signed_hadamard_cubature_2048",
    "signed_hadamard_cubature_4096",
    "signed_hadamard_cubature_6144",
    "axis_genut_513",
    "hadamard_genut_513",
)
SIGNED_HADAMARD_ROTATIONS = {
    "signed_hadamard_cubature_1024": 2,
    "signed_hadamard_cubature_2048": 4,
    "signed_hadamard_cubature_4096": 8,
    "signed_hadamard_cubature_6144": 12,
}
ROTATION_SEED_OFFSET = 23_000_123


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
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if partition == "holdout":
        raise SystemExit("Holdout is gated for sigma-point diagnostics.")
    ids = set(int(x) for x in manifest[f"{partition}_mlp_ids"])
    rows = list(
        whestbench.load_dataset(
            manifest["dataset"],
            revision=manifest["dataset_revision"],
            split=manifest["official_split"],
        )
    )
    selected = sorted(
        [row for row in rows if int(row["mlp_id"]) in ids],
        key=lambda row: int(row["mlp_id"]),
    )
    if limit is not None:
        selected = selected[:limit]
    return selected


def normal_lattice_samples(n_samples: int, width: int, seed: int) -> tuple[fnp.ndarray, fnp.ndarray]:
    rng = fnp.random.default_rng(seed + RQMC_SEED_OFFSET)
    powers = []
    value = 1
    modulus = max(2, n_samples)
    for _ in range(width):
        powers.append(float(value))
        value = (value * RQMC_MULTIPLIER) % modulus
        if value == 0:
            value = 1
    generator = fnp.array(powers, dtype=fnp.float64)
    indices = fnp.arange(n_samples, dtype=fnp.float64) + 0.5
    shift = rng.uniform(0.0, 1.0, size=(width,))
    u = fnp.mod(fnp.outer(indices, generator) / n_samples + shift, 1.0)
    u = fnp.clip(u, 1e-7, 1.0 - 1e-7)
    x = flops.stats.norm.ppf(u).astype(fnp.float32)
    weights = fnp.full((n_samples,), 1.0 / n_samples, dtype=fnp.float32)
    return x, weights


def mc_samples(n_samples: int, width: int, seed: int) -> tuple[fnp.ndarray, fnp.ndarray]:
    rng = fnp.random.default_rng(seed + MC_SEED_OFFSET)
    x = rng.normal(size=(n_samples, width)).astype(fnp.float32)
    weights = fnp.full((n_samples,), 1.0 / n_samples, dtype=fnp.float32)
    return x, weights


def hadamard_matrix(width: int) -> np.ndarray:
    if width <= 0 or width & (width - 1):
        raise ValueError("Hadamard sigma rules require power-of-two width")
    indices = np.arange(width, dtype=np.uint32)
    h = np.empty((width, width), dtype=np.float32)
    scale = 1.0 / math.sqrt(width)
    for col in range(width):
        parity = np.bitwise_count(np.bitwise_and(indices, np.uint32(col))) & 1
        h[:, col] = np.where(parity, -scale, scale)
    return h


def cubature_samples(width: int, rotated: bool) -> tuple[fnp.ndarray, fnp.ndarray]:
    radius = math.sqrt(width)
    basis = hadamard_matrix(width) if rotated else np.eye(width, dtype=np.float32)
    points = np.concatenate([radius * basis.T, -radius * basis.T], axis=0).astype(np.float32)
    weights = np.full((2 * width,), 1.0 / (2 * width), dtype=np.float32)
    return fnp.array(points), fnp.array(weights)


def signed_hadamard_cubature_samples(width: int, rotations: int, seed: int) -> tuple[fnp.ndarray, fnp.ndarray]:
    radius = math.sqrt(width)
    base = hadamard_matrix(width)
    rng = np.random.default_rng(seed + ROTATION_SEED_OFFSET + 10_007 * rotations)
    blocks = []
    for _ in range(rotations):
        permutation = rng.permutation(width)
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=width).astype(np.float32)
        basis = signs[:, None] * base[permutation, :]
        blocks.append(radius * basis.T)
        blocks.append(-radius * basis.T)
    points = np.concatenate(blocks, axis=0).astype(np.float32)
    weights = np.full((2 * width * rotations,), 1.0 / (2 * width * rotations), dtype=np.float32)
    return fnp.array(points), fnp.array(weights)


def genut_samples(width: int, rotated: bool) -> tuple[fnp.ndarray, fnp.ndarray]:
    # For a standard Gaussian marginal, radius sqrt(3) and side weight 1/6
    # match variance and fourth central moment in each represented direction.
    # In high dimension this necessarily gives a large negative center weight.
    radius = math.sqrt(3.0)
    basis = hadamard_matrix(width) if rotated else np.eye(width, dtype=np.float32)
    zeros = np.zeros((1, width), dtype=np.float32)
    side_points = np.concatenate([radius * basis.T, -radius * basis.T], axis=0).astype(np.float32)
    points = np.concatenate([zeros, side_points], axis=0)
    weights = np.concatenate(
        [
            np.asarray([1.0 - width / 3.0], dtype=np.float32),
            np.full((2 * width,), 1.0 / 6.0, dtype=np.float32),
        ]
    )
    return fnp.array(points), fnp.array(weights)


def samples_for_method(method: str, width: int, seed: int) -> tuple[fnp.ndarray, fnp.ndarray]:
    if method == "mc_513":
        return mc_samples(513, width, seed)
    if method == "rqmc_513":
        return normal_lattice_samples(513, width, seed)
    if method == "axis_cubature_512":
        return cubature_samples(width, rotated=False)
    if method == "hadamard_cubature_512":
        return cubature_samples(width, rotated=True)
    if method in SIGNED_HADAMARD_ROTATIONS:
        return signed_hadamard_cubature_samples(width, SIGNED_HADAMARD_ROTATIONS[method], seed)
    if method == "axis_genut_513":
        return genut_samples(width, rotated=False)
    if method == "hadamard_genut_513":
        return genut_samples(width, rotated=True)
    raise ValueError(f"Unknown method: {method}")


def weighted_forward(mlp: SimpleNamespace, method: str) -> fnp.ndarray:
    x, weights = samples_for_method(method, mlp.width, mlp.seed)
    rows = []
    for w in mlp.weights:
        x = fnp.maximum(fnp.matmul(x, w), 0.0)
        rows.append(fnp.sum(x * weights[:, None], axis=0))
    return fnp.stack(rows, axis=0)


def evaluate_one(row: dict[str, Any], method: str, args: argparse.Namespace) -> dict[str, Any]:
    mlp = as_mlp(row)
    target = np.asarray(row["all_layer_means"], dtype=np.float64)
    start = time.perf_counter()
    with flops.BudgetContext(flop_budget=args.budget, quiet=True) as context:
        prediction = weighted_forward(mlp, method)
    runtime_s = time.perf_counter() - start
    prediction_np = np.asarray(prediction, dtype=np.float64)
    if prediction_np.shape != (mlp.depth, mlp.width):
        raise ValueError(f"bad shape {prediction_np.shape}; expected {(mlp.depth, mlp.width)}")
    non_finite = int(np.size(prediction_np) - np.isfinite(prediction_np).sum())
    if non_finite:
        raise ValueError(f"prediction contains {non_finite} non-finite entries")
    squared_error = (prediction_np - target) ** 2
    flops_used = int(context.flops_used)
    final_mse = float(squared_error[-1].mean())
    return {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "created_at_utc": args.created_at,
        "partition": args.partition,
        "method": method,
        "mlp_id": int(row["mlp_id"]),
        "mlp_name": str(row["mlp_name"]),
        "mlp_seed": int(row["mlp_seed"]),
        "status": "completed",
        "final_layer_mse": final_mse,
        "all_layer_mse": float(squared_error.mean()),
        "adjusted_challenge_score": final_mse * max(args.score_floor, flops_used / args.budget),
        "charged_flops": flops_used,
        "budget_fraction": flops_used / args.budget,
        "wall_clock_s": runtime_s,
        "prefix_mse": {str(h): float(squared_error[h - 1].mean()) for h in PREFIXES if h <= mlp.depth},
    }


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


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    completed = [record for record in records if record["status"] == "completed"]
    baseline = {
        record["mlp_id"]: record
        for record in completed
        if record["method"] == "rqmc_513"
    }
    rows = []
    for method in args.methods:
        method_rows = [record for record in completed if record["method"] == method]
        if not method_rows:
            continue
        final_values = [record["final_layer_mse"] for record in method_rows]
        adjusted_values = [record["adjusted_challenge_score"] for record in method_rows]
        flops_values = [record["charged_flops"] for record in method_rows]
        runtime_values = [record["wall_clock_s"] for record in method_rows]
        paired = [record for record in method_rows if record["mlp_id"] in baseline and method != "rqmc_513"]
        if paired:
            base = np.asarray([baseline[record["mlp_id"]]["final_layer_mse"] for record in paired], dtype=np.float64)
            cand = np.asarray([record["final_layer_mse"] for record in paired], dtype=np.float64)
            diff = cand - base
            win_fraction = float(np.mean(cand < base))
            gain = float(base.mean() / cand.mean()) if cand.mean() else float("nan")
            mean_diff = float(diff.mean())
            median_diff = float(np.median(diff))
        else:
            win_fraction = gain = mean_diff = median_diff = float("nan")
        final = aggregate(final_values)
        adjusted = aggregate(adjusted_values)
        flops_stats = aggregate(flops_values)
        runtime = aggregate(runtime_values)
        rows.append(
            {
                "partition": args.partition,
                "method": method,
                "n_completed": len(method_rows),
                "mean_final_mse": final["mean"],
                "median_final_mse": final["median"],
                "se_final_mse": final["standard_error"],
                "worst_decile_final_mse": final["worst_decile"],
                "max_final_mse": final["max"],
                "mean_adjusted_score": adjusted["mean"],
                "mean_flops": flops_stats["mean"],
                "mean_budget_fraction": float(np.mean([record["budget_fraction"] for record in method_rows])),
                "mean_runtime_s": runtime["mean"],
                "mean_paired_mse_diff_vs_rqmc513": mean_diff,
                "median_paired_mse_diff_vs_rqmc513": median_diff,
                "win_fraction_vs_rqmc513": win_fraction,
                "raw_mse_gain_vs_rqmc513": gain,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, args: argparse.Namespace, summary_rows: list[dict[str, Any]]) -> None:
    ordered = sorted(summary_rows, key=lambda row: row["mean_final_mse"])
    lines = [
        "# Sigma/cubature benchmark",
        "",
        "Fixed full-network propagation benchmark for the higher-order unscented/sigma-point idea.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{' '.join(args.command)}\n```",
        "",
        "## Result matrix",
        "",
        "| method | mean final MSE | adjusted score | win vs RQMC-513 | gain vs RQMC-513 | FLOPs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        lines.append(
            "| {method} | {mse:.6g} | {score:.6g} | {win:.3g} | {gain:.3g} | {flops:.6g} |".format(
                method=row["method"],
                mse=row["mean_final_mse"],
                score=row["mean_adjusted_score"],
                win=row["win_fraction_vs_rqmc513"]
                if not math.isnan(row["win_fraction_vs_rqmc513"])
                else float("nan"),
                gain=row["raw_mse_gain_vs_rqmc513"]
                if not math.isnan(row["raw_mse_gain_vs_rqmc513"])
                else float("nan"),
                flops=row["mean_flops"],
            )
        )
    best = ordered[0] if ordered else None
    lines.extend(["", "## Decision", ""])
    if best and best["method"] == "rqmc_513":
        lines.append("RQMC-513 is the best 513-point method here; do not build blockwise GenUT yet.")
    elif best:
        lines.append(
            f"`{best['method']}` beats RQMC-513 on this partition and deserves a validation check before any larger implementation."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = select_rows(args.partition, args.limit)
    records = []
    jsonl = RESULTS / "sigma_point_benchmark_runs.jsonl"
    if not args.append_jsonl:
        jsonl.write_text("", encoding="utf-8")
    with jsonl.open("a", encoding="utf-8") as handle:
        for row in rows:
            for method in args.methods:
                try:
                    record = evaluate_one(row, method, args)
                except Exception as exc:
                    record = {
                        "schema_version": "1.0",
                        "run_id": args.run_id,
                        "created_at_utc": args.created_at,
                        "partition": args.partition,
                        "method": method,
                        "mlp_id": int(row["mlp_id"]),
                        "mlp_name": str(row["mlp_name"]),
                        "mlp_seed": int(row["mlp_seed"]),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(
                    f"{method} mlp={record['mlp_id']} status={record['status']} "
                    f"mse={record.get('final_layer_mse', float('nan')):.6g}",
                    flush=True,
                )
    summary_rows = summarize(records, args)
    suffix = args.partition if args.limit is None else f"{args.partition}_limit{args.limit}"
    csv_path = RESULTS / f"sigma_point_benchmark_{suffix}_summary.csv"
    report_path = REPORTS / f"sigma_point_benchmark_{suffix}.md"
    write_csv(csv_path, summary_rows)
    write_report(report_path, args, summary_rows)
    return {
        "run_id": args.run_id,
        "csv": str(csv_path),
        "jsonl": str(jsonl),
        "report": str(report_path),
        "summary": sorted(summary_rows, key=lambda row: row["mean_final_mse"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=["development", "validation", "holdout"], default="development")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--score-floor", type=float, default=SCORE_FLOOR)
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    parser.add_argument("--append-jsonl", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.budget <= 0:
        raise SystemExit("--budget must be positive")
    args.created_at = datetime.now(timezone.utc).isoformat()
    args.command = [str(item) for item in __import__("sys").argv]
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
