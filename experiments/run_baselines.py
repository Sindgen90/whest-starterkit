"""Run immutable Phase 1 baselines on complete MLP splits."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
import whestbench

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATASET = "aicrowd/arc-whestbench-public-2026"
REVISION = "v1-phase1"
BUDGET = 272_000_000_000
SCORE_FLOOR = 0.1
PREFIXES = (2, 4, 8, 16, 24, 32)
BASELINES = {
    "starter": ROOT / "estimator.py",
    "mean": ROOT / "examples" / "02_mean_propagation.py",
    "covariance": ROOT / "examples" / "03_covariance_propagation.py",
}
SPLITS = {"development": "mini", "validation": "full"}


def load_estimator(path: Path) -> type[Any]:
    spec = importlib.util.spec_from_file_location(f"baseline_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import estimator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Estimator


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def aggregate(values: Iterable[float]) -> dict[str, float | None]:
    data = np.asarray(list(values), dtype=np.float64)
    if not len(data):
        return {"mean": None, "median": None, "standard_error": None, "worst_decile_mean": None}
    worst_count = max(1, math.ceil(len(data) / 10))
    return {
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "standard_error": float(data.std(ddof=1) / math.sqrt(len(data))) if len(data) > 1 else 0.0,
        "worst_decile_mean": float(np.sort(data)[-worst_count:].mean()),
    }


def as_mlp(row: dict[str, Any]) -> SimpleNamespace:
    """Minimal interface used by the three unmodified bundled baselines."""
    stacked = fnp.array(np.asarray(row["weights"], dtype=np.float32))
    depth, width, _ = stacked.shape
    return SimpleNamespace(
        width=int(width), depth=int(depth), weights=[stacked[i] for i in range(depth)],
        seed=int(row["mlp_seed"]), name=str(row["mlp_name"]),
    )


def evaluate(estimator: Any, row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mlp = as_mlp(row)
    target = np.asarray(row["all_layer_means"], dtype=np.float64)
    started = time.perf_counter()
    with flops.BudgetContext(flop_budget=args.budget, quiet=True) as context:
        prediction = estimator.predict(mlp, args.budget)
    runtime_s = time.perf_counter() - started
    prediction = np.asarray(prediction, dtype=np.float64)
    expected_shape = (mlp.depth, mlp.width)
    if prediction.shape != expected_shape:
        raise ValueError(f"prediction shape {prediction.shape}; expected {expected_shape}")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("prediction contains non-finite values")
    squared_error = (prediction - target) ** 2
    flops_used = int(context.flops_used)
    final_mse = float(squared_error[-1].mean())
    prefix_mse = {str(h): float(squared_error[h - 1].mean()) for h in args.prefixes if h <= mlp.depth}
    return {
        "mlp_id": int(row["mlp_id"]), "mlp_name": str(row["mlp_name"]),
        "mlp_seed": int(row["mlp_seed"]), "status": "completed",
        "metrics": {"final_mse": final_mse, "all_layer_mse": float(squared_error.mean()), "prefix_mse": prefix_mse},
        "compute": {
            "flops_used": flops_used, "flop_budget": args.budget, "runtime_s": runtime_s,
            "flop_fraction": flops_used / args.budget,
            "adjusted_final_score": final_mse * max(args.score_floor, flops_used / args.budget),
        },
    }


def summarize(records: list[dict[str, Any]], prefixes: Iterable[int]) -> dict[str, Any]:
    done = [record for record in records if record["status"] == "completed"]
    return {
        "n_requested": len(records), "n_completed": len(done), "n_failed": len(records) - len(done),
        "final_mse": aggregate(r["metrics"]["final_mse"] for r in done),
        "all_layer_mse": aggregate(r["metrics"]["all_layer_mse"] for r in done),
        "adjusted_final_score": aggregate(r["compute"]["adjusted_final_score"] for r in done),
        "flops": aggregate(r["compute"]["flops_used"] for r in done),
        "runtime_s": aggregate(r["compute"]["runtime_s"] for r in done),
        "prefix_mse": {str(h): aggregate(r["metrics"]["prefix_mse"][str(h)] for r in done if str(h) in r["metrics"]["prefix_mse"]) for h in prefixes},
    }


def run_baseline(name: str, args: argparse.Namespace) -> Path:
    estimator_path = BASELINES[name]
    estimator = load_estimator(estimator_path)()
    dataset = whestbench.load_dataset(DATASET, revision=REVISION, split=SPLITS[args.split])
    rows = list(dataset)
    if args.limit_mlps is not None:
        rows = rows[:args.limit_mlps]
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        try:
            records.append(evaluate(estimator, row, args))
        except Exception as exc:
            records.append({"mlp_id": int(row["mlp_id"]), "mlp_name": str(row["mlp_name"]), "mlp_seed": int(row["mlp_seed"]), "status": "failed"})
            failures.append({"mlp_id": int(row["mlp_id"]), "mlp_name": str(row["mlp_name"]), "error_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    source_hash = hashlib.sha256(estimator_path.read_bytes()).hexdigest()
    fingerprint = json.dumps(
        {"name": name, "split": SPLITS[args.split], "budget": args.budget,
         "prefixes": args.prefixes, "limit": args.limit_mlps, "source": source_hash},
        sort_keys=True,
    )
    result = {
        "schema_version": "1.0", "run_id": str(uuid.uuid5(uuid.NAMESPACE_URL, fingerprint)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if not failures else "completed_with_failures",
        "configuration": {
            "phase": "phase1", "dataset": DATASET, "dataset_revision": REVISION,
            "dataset_split": SPLITS[args.split], "baseline": name, "flop_budget": args.budget,
            "score_floor": args.score_floor, "depth_prefixes": args.prefixes,
            "limit_mlps": args.limit_mlps,
        },
        "source": {
            "estimator_path": str(estimator_path.relative_to(ROOT)), "estimator_sha256": source_hash,
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--porcelain")), "python": sys.version,
            "whestbench": getattr(whestbench, "__version__", None),
            "flopscope": getattr(flops, "__version__", None),
        },
        "summary": summarize(records, args.prefixes), "per_mlp": records, "failures": failures,
    }
    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / f"{name}_{SPLITS[args.split]}_{result['run_id']}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def table(paths: list[Path]) -> str:
    rows = []
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        summary, config = result["summary"], result["configuration"]
        rows.append(
            f"| {config['baseline']} | {config['dataset_split']} | "
            f"{summary['n_completed']}/{summary['n_requested']} | "
            f"{summary['final_mse']['mean']:.6g} | {summary['all_layer_mse']['mean']:.6g} | "
            f"{summary['adjusted_final_score']['mean']:.6g} | {summary['flops']['mean']:.3g} | {summary['runtime_s']['mean']:.3f} |"
        )
    header = "| Estimator | Split | Completed | Final MSE | All-layer MSE | Adjusted score | FLOPs / MLP | Runtime s / MLP |"
    separator = "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    return "\n".join(["# Phase 1 baseline table", "", header, separator, *rows]) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=[*BASELINES, "all"], default="all")
    parser.add_argument("--split", choices=SPLITS, default="development")
    parser.add_argument("--allow-validation", action="store_true")
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--score-floor", type=float, default=SCORE_FLOOR)
    parser.add_argument("--limit-mlps", type=int, default=None)
    parser.add_argument("--prefixes", type=int, nargs="+", default=list(PREFIXES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.budget <= 0:
        raise SystemExit("--budget must be positive")
    if not 0 <= args.score_floor <= 1:
        raise SystemExit("--score-floor must be between zero and one")
    if args.limit_mlps is not None and args.limit_mlps <= 0:
        raise SystemExit("--limit-mlps must be positive")
    if args.split == "validation" and not args.allow_validation:
        raise SystemExit("Validation is gated: use --allow-validation only for frozen candidates.")
    args.prefixes = sorted(set(args.prefixes))
    names = list(BASELINES) if args.baseline == "all" else [args.baseline]
    paths = [run_baseline(name, args) for name in names]
    table_path = RESULTS / f"baseline_table_{SPLITS[args.split]}.md"
    table_path.write_text(table(paths), encoding="utf-8")
    print(f"Wrote {len(paths)} result documents and {table_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
