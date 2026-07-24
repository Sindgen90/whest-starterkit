"""Evaluate one estimator.py against a frozen manifest partition."""
from __future__ import annotations

import argparse
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
BUDGET = 272_000_000_000
SCORE_FLOOR = 0.1
PREFIXES = (2, 4, 8, 16, 24, 32)


def load_estimator(path: Path) -> type[Any]:
    module_name = f"single_eval_{path.parent.name}_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"
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
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = list(
        whestbench.load_dataset(
            manifest["dataset"],
            revision=manifest["dataset_revision"],
            split=manifest["official_split"],
        )
    )
    ids = set(int(x) for x in manifest[f"{partition}_mlp_ids"])
    selected = sorted(
        [row for row in rows if int(row["mlp_id"]) in ids],
        key=lambda row: int(row["mlp_id"]),
    )
    if limit is not None:
        selected = selected[:limit]
    return selected


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


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    estimator_path = Path(args.estimator_path)
    if not estimator_path.is_absolute():
        estimator_path = ROOT / estimator_path
    estimator = load_estimator(estimator_path)()
    rows = select_rows(args.partition, args.limit)
    records = []
    for row in rows:
        mlp = as_mlp(row)
        target = np.asarray(row["all_layer_means"], dtype=np.float64)
        start = time.perf_counter()
        with flops.BudgetContext(flop_budget=args.budget, quiet=True) as context:
            prediction = estimator.predict(mlp, args.budget)
        runtime_s = time.perf_counter() - start
        prediction_np = np.asarray(prediction, dtype=np.float64)
        if prediction_np.shape != (mlp.depth, mlp.width):
            raise ValueError(f"bad prediction shape {prediction_np.shape}; expected {(mlp.depth, mlp.width)}")
        if not np.all(np.isfinite(prediction_np)):
            raise ValueError("prediction contains non-finite values")
        squared_error = (prediction_np - target) ** 2
        flops_used = int(context.flops_used)
        final_mse = float(squared_error[-1].mean())
        record = {
            "mlp_id": int(row["mlp_id"]),
            "mlp_name": str(row["mlp_name"]),
            "mlp_seed": int(row["mlp_seed"]),
            "final_mse": final_mse,
            "all_layer_mse": float(squared_error.mean()),
            "adjusted_score": final_mse * max(args.score_floor, flops_used / args.budget),
            "flops": flops_used,
            "runtime_s": runtime_s,
            "prefix_mse": {str(h): float(squared_error[h - 1].mean()) for h in PREFIXES if h <= mlp.depth},
        }
        records.append(record)
        print(
            f"{estimator_path.parent.name} mlp={record['mlp_id']} "
            f"mse={record['final_mse']:.6g} adjusted={record['adjusted_score']:.6g}",
            flush=True,
        )
    summary = {
        "final_mse": aggregate([record["final_mse"] for record in records]),
        "all_layer_mse": aggregate([record["all_layer_mse"] for record in records]),
        "adjusted_score": aggregate([record["adjusted_score"] for record in records]),
        "flops": aggregate([record["flops"] for record in records]),
        "runtime_s": aggregate([record["runtime_s"] for record in records]),
    }
    result = {
        "schema_version": "1.0",
        "run_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition": args.partition,
        "limit": args.limit,
        "estimator_path": str(estimator_path.relative_to(ROOT)),
        "estimator_sha256": hashlib.sha256(estimator_path.read_bytes()).hexdigest(),
        "budget": args.budget,
        "score_floor": args.score_floor,
        "summary": summary,
        "per_mlp": records,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator-path", required=True)
    parser.add_argument("--partition", choices=["development", "validation", "holdout"], default="validation")
    parser.add_argument("--allow-holdout", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--score-floor", type=float, default=SCORE_FLOOR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.partition == "holdout" and not args.allow_holdout:
        raise SystemExit("Holdout is gated: use --allow-holdout only after configurations are frozen.")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    result = evaluate(args)
    output = args.output
    if output is None:
        stem = Path(args.estimator_path).parent.name
        output = RESULTS / f"{stem}_{args.partition}_manifest_eval.json"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"SUMMARY {json.dumps(result['summary'], sort_keys=True)}")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
