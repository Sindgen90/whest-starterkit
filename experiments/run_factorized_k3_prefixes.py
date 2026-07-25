"""Compare the factorized K3 prototype against corrected K2 on Mini prefixes."""
from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import flopscope as flops
import numpy as np
import whestbench
from factorized_k3 import Estimator as K3Estimator
from run_baselines import (
    BUDGET,
    DATASET,
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


def truncate_mlp(mlp: Any, depth: int) -> Any:
    return SimpleNamespace(
        width=mlp.width,
        depth=depth,
        weights=mlp.weights[:depth],
        seed=mlp.seed,
        name=f"{mlp.name}-prefix{depth}",
    )


def evaluate(estimator: Any, mlp: Any, target: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    with flops.BudgetContext(flop_budget=args.budget, quiet=True) as context:
        prediction = estimator.predict(mlp, args.budget)
    runtime_s = time.perf_counter() - started
    prediction_np = np.asarray(prediction, dtype=np.float64)
    error = (prediction_np - target) ** 2
    flops_used = int(context.flops_used)
    return {
        "final_mse": float(error[-1].mean()),
        "all_layer_mse": float(error.mean()),
        "flops_used": flops_used,
        "runtime_s": runtime_s,
        "adjusted_final_score": float(error[-1].mean()) * max(SCORE_FLOOR, flops_used / args.budget),
    }


def summarize(records: list[dict[str, Any]], prefix: int) -> dict[str, Any]:
    done = [r for r in records if r["status"] == "completed" and r["prefix_depth"] == prefix]
    wins = [r["k3"]["final_mse"] < r["corrected_k2"]["final_mse"] for r in done]
    return {
        "n_completed": len(done),
        "k2_final_mse": aggregate(r["corrected_k2"]["final_mse"] for r in done),
        "k3_final_mse": aggregate(r["k3"]["final_mse"] for r in done),
        "k2_all_layer_mse": aggregate(r["corrected_k2"]["all_layer_mse"] for r in done),
        "k3_all_layer_mse": aggregate(r["k3"]["all_layer_mse"] for r in done),
        "k2_adjusted_final_score": aggregate(r["corrected_k2"]["adjusted_final_score"] for r in done),
        "k3_adjusted_final_score": aggregate(r["k3"]["adjusted_final_score"] for r in done),
        "k3_flops": aggregate(r["k3"]["flops_used"] for r in done),
        "k3_runtime_s": aggregate(r["k3"]["runtime_s"] for r in done),
        "k3_win_fraction": sum(wins) / len(wins) if wins else None,
        "k3_final_mse_ratio": aggregate(
            r["k3"]["final_mse"] / r["corrected_k2"]["final_mse"] for r in done
        ),
        "factor_count_last_layer": aggregate(
            r["k3_factor_history"][-1]["factor_count"] for r in done if r["k3_factor_history"]
        ),
    }


def run(args: argparse.Namespace) -> Path:
    rows = list(whestbench.load_dataset(DATASET, revision=REVISION, split="mini"))
    if args.limit_mlps is not None:
        rows = rows[: args.limit_mlps]
    prefixes = sorted(set(args.prefixes))
    k2 = load_estimator(CORRECTED_K2)()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows, start=1):
        base_mlp = as_mlp(row)
        full_target = np.asarray(row["all_layer_means"], dtype=np.float64)
        for prefix in prefixes:
            if prefix > base_mlp.depth:
                continue
            prefix_mlp = truncate_mlp(base_mlp, prefix)
            target = full_target[:prefix]
            k3 = K3Estimator(
                third_scale=args.third_scale,
                generic_horizon=args.generic_horizon,
                max_k3_layer=args.max_k3_layer,
            )
            try:
                k2_result = evaluate(k2, prefix_mlp, target, args)
                k3_result = evaluate(k3, prefix_mlp, target, args)
                records.append({
                    "mlp_id": int(row["mlp_id"]),
                    "mlp_name": str(row["mlp_name"]),
                    "prefix_depth": prefix,
                    "status": "completed",
                    "corrected_k2": k2_result,
                    "k3": k3_result,
                    "k3_factor_history": k3.factor_history,
                })
                print(f"{row_index}/{len(rows)} {row['mlp_name']} prefix={prefix} completed", flush=True)
            except Exception as exc:
                failures.append({
                    "mlp_id": int(row["mlp_id"]),
                    "mlp_name": str(row["mlp_name"]),
                    "prefix_depth": prefix,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                })
                records.append({
                    "mlp_id": int(row["mlp_id"]),
                    "mlp_name": str(row["mlp_name"]),
                    "prefix_depth": prefix,
                    "status": "failed",
                })
                print(f"{row_index}/{len(rows)} {row['mlp_name']} prefix={prefix} failed", flush=True)

    result = {
        "schema_version": "1.0",
        "experiment": "factorized_k3_prefixes",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if not failures else "completed_with_failures",
        "configuration": {
            "dataset": DATASET,
            "dataset_revision": REVISION,
            "dataset_split": "mini",
            "flop_budget": args.budget,
            "score_floor": SCORE_FLOOR,
            "prefixes": prefixes,
            "limit_mlps": args.limit_mlps,
            "third_scale": args.third_scale,
            "generic_horizon": args.generic_horizon,
            "max_k3_layer": args.max_k3_layer,
        },
        "source": {
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--porcelain")),
        },
        "summary_by_prefix": {str(prefix): summarize(records, prefix) for prefix in prefixes},
        "per_mlp": records,
        "failures": failures,
    }
    RESULTS.mkdir(exist_ok=True)
    scale_tag = f"scale{args.third_scale:g}".replace(".", "p")
    horizon_tag = "full" if args.generic_horizon is None else f"H{args.generic_horizon}"
    cutoff_tag = "all" if args.max_k3_layer is None else f"L{args.max_k3_layer}"
    suffix = "mini" if args.limit_mlps is None else f"mini_limit{args.limit_mlps}"
    suffix = f"{suffix}_{scale_tag}_{horizon_tag}_{cutoff_tag}"
    output = RESULTS / f"factorized_k3_prefixes_{suffix}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--prefixes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--limit-mlps", type=int, default=5)
    parser.add_argument("--third-scale", type=float, default=1.0)
    parser.add_argument("--generic-horizon", type=int, default=None)
    parser.add_argument("--max-k3-layer", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_mlps is not None and args.limit_mlps <= 0:
        raise SystemExit("--limit-mlps must be positive")
    output = run(args)
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
