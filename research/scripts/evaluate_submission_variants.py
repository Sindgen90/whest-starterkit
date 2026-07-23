"""Evaluate RQMC submission variants on frozen whole-MLP manifest partitions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
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
RESULTS = ROOT / "research" / "results"
MANIFEST = ROOT / "research" / "configs" / "stein_holdout_manifest.json"
DATASET = "aicrowd/arc-whestbench-public-2026"
REVISION = "v1-phase1"
BUDGET = 272_000_000_000
SCORE_FLOOR = 0.1
PREFIXES = (2, 4, 8, 16, 24, 32)
METHODS = {
    "rqmc_baseline": ROOT / "submission_variants" / "rqmc_baseline" / "estimator.py",
    "rqmc_radial": ROOT / "submission_variants" / "rqmc_radial" / "estimator.py",
    "rqmc_hermite_random": ROOT / "submission_variants" / "rqmc_hermite_random" / "estimator.py",
}


def load_estimator(path: Path, args: argparse.Namespace | None = None, method: str | None = None) -> type[Any]:
    module_name = f"stein_variant_{path.parent.name}_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import estimator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args is not None and method == "rqmc_hermite_random":
        if args.hermite_rank is not None:
            module.RANK = int(args.hermite_rank)
        if args.hermite_shrinkage is not None:
            module.SHRINKAGE = float(args.hermite_shrinkage)
    return module.Estimator


def estimator_config(path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(f"config_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    keys = [
        "N_SAMPLES",
        "FOLD_SAMPLES",
        "RANK",
        "SHRINKAGE",
        "RQMC_SEED_OFFSET",
        "RQMC_SEED_OFFSET_A",
        "RQMC_SEED_OFFSET_B",
        "RQMC_MULTIPLIER",
    ]
    return {key.lower(): getattr(module, key) for key in keys if hasattr(module, key)}


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


def select_rows(rows: list[dict[str, Any]], partition: str, limit: int | None) -> list[dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if "splits" in manifest:
        ids = set(manifest["splits"][partition]["mlp_ids"])
    else:
        ids = set(manifest[f"{partition}_mlp_ids"])
    selected = [row for row in rows if int(row["mlp_id"]) in ids]
    selected = sorted(selected, key=lambda row: int(row["mlp_id"]))
    if limit is not None:
        selected = selected[:limit]
    return selected


def evaluate_one(method: str, estimator: Any, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mlp = as_mlp(row)
    target = np.asarray(row["all_layer_means"], dtype=np.float64)
    start = time.perf_counter()
    with flops.BudgetContext(flop_budget=args.budget, quiet=True) as context:
        prediction = estimator.predict(mlp, args.budget)
    runtime_s = time.perf_counter() - start
    prediction_np = np.asarray(prediction, dtype=np.float64)
    if prediction_np.shape != (mlp.depth, mlp.width):
        raise ValueError(f"bad prediction shape {prediction_np.shape}; expected {(mlp.depth, mlp.width)}")
    non_finite = int(np.size(prediction_np) - np.isfinite(prediction_np).sum())
    if non_finite:
        raise ValueError(f"prediction contains {non_finite} non-finite values")
    squared_error = (prediction_np - target) ** 2
    flops_used = int(context.flops_used)
    final_mse = float(squared_error[-1].mean())
    record = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "created_at_utc": args.created_at,
        "partition": args.partition,
        "dataset": DATASET,
        "dataset_revision": REVISION,
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
        "non_finite_count": 0,
        "prefix_mse": {str(h): float(squared_error[h - 1].mean()) for h in PREFIXES if h <= mlp.depth},
        "method_config": estimator_config(METHODS[method]),
    }
    if method == "rqmc_hermite_random":
        if args.hermite_rank is not None:
            record["method_config"]["rank"] = int(args.hermite_rank)
        if args.hermite_shrinkage is not None:
            record["method_config"]["shrinkage"] = float(args.hermite_shrinkage)
        record["method_config"]["feature_count"] = int(
            record["method_config"].get("rank", 8)
            + record["method_config"].get("rank", 8) * (record["method_config"].get("rank", 8) + 1) // 2
        )
    return record


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


def bootstrap_ci(diff: np.ndarray, seed: int = 20260723, reps: int = 2000) -> tuple[float, float]:
    if len(diff) <= 1:
        value = float(diff[0]) if len(diff) else float("nan")
        return value, value
    rng = np.random.default_rng(seed)
    draws = rng.choice(diff, size=(reps, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def write_summary(records: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    completed = [record for record in records if record["status"] == "completed"]
    by_method = {method: [r for r in completed if r["method"] == method] for method in args.methods}
    baseline = {r["mlp_id"]: r for r in by_method.get("rqmc_baseline", [])}
    summary_path = RESULTS / "stein_rqmc_holdout_summary.csv"
    fieldnames = [
        "run_id",
        "partition",
        "method",
        "n_completed",
        "mean_final_mse",
        "median_final_mse",
        "se_final_mse",
        "worst_decile_final_mse",
        "max_final_mse",
        "mean_adjusted_score",
        "mean_flops",
        "mean_budget_fraction",
        "mean_runtime_s",
        "mean_paired_mse_diff_vs_rqmc",
        "median_paired_mse_diff_vs_rqmc",
        "win_fraction_vs_rqmc",
        "paired_diff_ci95_low",
        "paired_diff_ci95_high",
        "raw_mse_gain_vs_rqmc",
        "adjusted_score_gain_vs_rqmc",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, rows in by_method.items():
            if not rows:
                continue
            final = aggregate([r["final_layer_mse"] for r in rows])
            adjusted = aggregate([r["adjusted_challenge_score"] for r in rows])
            flops_stats = aggregate([r["charged_flops"] for r in rows])
            budget_fraction = aggregate([r["budget_fraction"] for r in rows])
            runtime = aggregate([r["wall_clock_s"] for r in rows])
            paired_rows = [r for r in rows if r["mlp_id"] in baseline and method != "rqmc_baseline"]
            if paired_rows:
                base_mse = np.asarray([baseline[r["mlp_id"]]["final_layer_mse"] for r in paired_rows])
                cand_mse = np.asarray([r["final_layer_mse"] for r in paired_rows])
                base_score = np.asarray([baseline[r["mlp_id"]]["adjusted_challenge_score"] for r in paired_rows])
                cand_score = np.asarray([r["adjusted_challenge_score"] for r in paired_rows])
                diff = cand_mse - base_mse
                ci_low, ci_high = bootstrap_ci(diff)
                mean_diff = float(diff.mean())
                median_diff = float(np.median(diff))
                win_fraction = float(np.mean(cand_mse < base_mse))
                raw_gain = float(base_mse.mean() / cand_mse.mean())
                adjusted_gain = float(base_score.mean() / cand_score.mean())
            else:
                ci_low = ci_high = mean_diff = median_diff = win_fraction = raw_gain = adjusted_gain = float("nan")
            writer.writerow(
                {
                    "run_id": args.run_id,
                    "partition": args.partition,
                    "method": method,
                    "n_completed": len(rows),
                    "mean_final_mse": final["mean"],
                    "median_final_mse": final["median"],
                    "se_final_mse": final["standard_error"],
                    "worst_decile_final_mse": final["worst_decile"],
                    "max_final_mse": final["max"],
                    "mean_adjusted_score": adjusted["mean"],
                    "mean_flops": flops_stats["mean"],
                    "mean_budget_fraction": budget_fraction["mean"],
                    "mean_runtime_s": runtime["mean"],
                    "mean_paired_mse_diff_vs_rqmc": mean_diff,
                    "median_paired_mse_diff_vs_rqmc": median_diff,
                    "win_fraction_vs_rqmc": win_fraction,
                    "paired_diff_ci95_low": ci_low,
                    "paired_diff_ci95_high": ci_high,
                    "raw_mse_gain_vs_rqmc": raw_gain,
                    "adjusted_score_gain_vs_rqmc": adjusted_gain,
                }
            )
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=["development", "validation", "holdout"], default="development")
    parser.add_argument("--allow-holdout", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=list(METHODS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--score-floor", type=float, default=SCORE_FLOOR)
    parser.add_argument("--hermite-rank", type=int, choices=[4, 8, 16], default=None)
    parser.add_argument("--hermite-shrinkage", type=float, choices=[0.25, 0.5, 0.75, 1.0], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.partition == "holdout" and not args.allow_holdout:
        raise SystemExit("Holdout is gated: use --allow-holdout only after configurations are frozen.")
    args.run_id = str(uuid.uuid4())
    args.created_at = datetime.now(timezone.utc).isoformat()
    RESULTS.mkdir(parents=True, exist_ok=True)
    dataset = whestbench.load_dataset(DATASET, revision=REVISION, split="mini")
    rows = select_rows(list(dataset), args.partition, args.limit)
    if not rows:
        raise SystemExit(f"No rows selected for partition {args.partition}")
    estimators = {method: load_estimator(METHODS[method], args, method)() for method in args.methods}
    records: list[dict[str, Any]] = []
    output_jsonl = RESULTS / "stein_rqmc_runs.jsonl"
    with output_jsonl.open("a", encoding="utf-8") as handle:
        for row in rows:
            for method, estimator in estimators.items():
                try:
                    record = evaluate_one(method, estimator, row, args)
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
                    f"final_mse={record.get('final_layer_mse', float('nan')):.6g}"
                )
    summary_path = write_summary(records, args)
    print(f"Wrote {output_jsonl.relative_to(ROOT)} and {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
