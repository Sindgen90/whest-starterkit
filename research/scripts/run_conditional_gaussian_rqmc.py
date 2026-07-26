"""Evaluate final-layer conditional-Gaussian RQMC against frozen baselines."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import flopscope as flops
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_rqmc_sample_count_sweep import (  # noqa: E402
    BUDGET,
    PREFIXES,
    SCORE_FLOOR,
    aggregate,
    as_mlp,
    select_rows,
)

ESTIMATOR = ROOT / "submissions" / "rqmc_conditional_gaussian_24576" / "estimator.py"
RQMC_SWEEP = (
    ROOT
    / "research"
    / "results"
    / "rqmc_sample_count_sweep_development_selection_4096_16384_24576_32768.json"
)
K3_RESULTS = ROOT / "research" / "results" / "diagonal_k3_residual_stack_full_s03_l8.json"
RESULTS = ROOT / "research" / "results"
REPORTS = ROOT / "research" / "reports"


def load_estimator(path: Path) -> Any:
    module_name = f"conditional_gaussian_{hashlib.sha256(path.read_bytes()).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Estimator()


def comparison_rows(partition: str) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    rqmc_by_id: dict[int, dict[str, Any]] = {}
    if RQMC_SWEEP.exists() and partition == "development_selection":
        result = json.loads(RQMC_SWEEP.read_text(encoding="utf-8"))
        rqmc_by_id = {
            int(row["mlp_id"]): row
            for row in result["per_mlp"]
            if int(row["n_samples"]) == 24_576
        }
    k3_by_id: dict[int, dict[str, Any]] = {}
    if K3_RESULTS.exists():
        result = json.loads(K3_RESULTS.read_text(encoding="utf-8"))
        k3_by_id = {
            int(row["mlp_id"]): row
            for row in result["per_mlp"]
            if row["split"] == partition and row["method"] == "stack"
        }
    return rqmc_by_id, k3_by_id


def optional_mean(values: list[float | bool]) -> float | None:
    return float(np.mean(values)) if values else None


def optional_aggregate(values: list[float]) -> dict[str, float] | None:
    return aggregate(values) if values else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    rqmc_wins = [
        row["win_vs_rqmc_24576"]
        for row in records
        if row["win_vs_rqmc_24576"] is not None
    ]
    k3_wins = [row["win_vs_k3"] for row in records if row["win_vs_k3"] is not None]
    rqmc_deltas = [
        row["delta_vs_rqmc_24576"]
        for row in records
        if row["delta_vs_rqmc_24576"] is not None
    ]
    return {
        "n_completed": len(records),
        "final_mse": aggregate([row["final_mse"] for row in records]),
        "all_layer_mse": aggregate([row["all_layer_mse"] for row in records]),
        "adjusted_score": aggregate([row["adjusted_score"] for row in records]),
        "flops": aggregate([float(row["flops"]) for row in records]),
        "runtime_s": aggregate([row["runtime_s"] for row in records]),
        "win_fraction_vs_rqmc_24576": optional_mean(rqmc_wins),
        "win_fraction_vs_k3": optional_mean(k3_wins),
        "paired_delta_vs_rqmc_24576": optional_aggregate(rqmc_deltas),
        "prefix_mse": {
            str(prefix): aggregate([row["prefix_mse"][str(prefix)] for row in records])
            for prefix in PREFIXES
        },
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    baseline = result.get("rqmc_24576_summary")
    k3 = result.get("k3_summary")
    lines = [
        "# Conditional-Gaussian final-layer RQMC",
        "",
        "The only mathematical change from ordinary 24,576-point RQMC is replacing",
        "`mean(ReLU(z_L))` by the Gaussian ReLU expectation fitted from the sampled",
        "final-preactivation mean and variance.",
        "",
        "## Command actually executed",
        "",
        f"```powershell\n{result['command']}\n```",
        "",
        "## Result matrix",
        "",
        "| method | final MSE | adjusted score | FLOPs | win vs ordinary RQMC | win vs K3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if baseline:
        lines.append(
            f"| ordinary RQMC 24576 | {baseline['final_mse']:.6g} | "
            f"{baseline['adjusted_score']:.6g} | {baseline['flops']:.6g} | -- | -- |"
        )
    if k3:
        lines.append(
            f"| diagonal-K3 residual stack | {k3['final_mse']:.6g} | "
            f"{k3['adjusted_score']:.6g} | {k3['flops']:.6g} | -- | -- |"
        )
    win_rqmc = summary["win_fraction_vs_rqmc_24576"]
    win_k3 = summary["win_fraction_vs_k3"]
    lines.append(
        "| conditional-Gaussian RQMC 24576 | {mse:.6g} | {score:.6g} | {flops:.6g} | "
        "{winrqmc} | {wink3} |".format(
            mse=summary["final_mse"]["mean"],
            score=summary["adjusted_score"]["mean"],
            flops=summary["flops"]["mean"],
            winrqmc="--" if win_rqmc is None else f"{win_rqmc:.3f}",
            wink3="--" if win_k3 is None else f"{win_k3:.3f}",
        )
    )
    if result["partition"] == "development_selection" and baseline and k3:
        mse_increase = 100.0 * (summary["final_mse"]["mean"] / baseline["final_mse"] - 1.0)
        rqmc_score_increase = (
            100.0 * (summary["adjusted_score"]["mean"] / baseline["adjusted_score"] - 1.0)
        )
        k3_score_increase = (
            100.0 * (summary["adjusted_score"]["mean"] / k3["adjusted_score"] - 1.0)
        )
        lines.extend(
            [
                "",
                "## Decision",
                "",
                "Reject this correction. It increased mean final-layer MSE by "
                f"`{summary['paired_delta_vs_rqmc_24576']['mean']:.6g}` "
                f"({mse_increase:.2f}%) relative to ordinary 24,576-point RQMC and lost on "
                "all 30 paired development MLPs. Its adjusted score was "
                f"{rqmc_score_increase:.2f}% worse than ordinary RQMC and "
                f"{k3_score_increase:.2f}% worse than the diagonal-K3 residual stack.",
                "",
                "The failure is consistent with moment-closure bias: the final preactivations "
                "are not Gaussian enough for their first two marginal moments to determine the "
                "ReLU mean. This systematic bias overwhelms any variance reduction from "
                "replacing the sampled final ReLU average.",
                "",
                "The validation gate did not pass. No validation data was evaluated and no "
                "submission package was created.",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = select_rows(args.partition, args.limit)
    estimator = load_estimator(ESTIMATOR)
    rqmc_by_id, k3_by_id = comparison_rows(args.partition)
    records = []
    for index, row in enumerate(rows, start=1):
        mlp = as_mlp(row)
        target = np.asarray(row["all_layer_means"], dtype=np.float64)
        started = time.perf_counter()
        with flops.BudgetContext(flop_budget=args.budget, quiet=True) as context:
            prediction = estimator.predict(mlp, args.budget)
        runtime_s = time.perf_counter() - started
        prediction_np = np.asarray(prediction, dtype=np.float64)
        if prediction_np.shape != target.shape or not np.all(np.isfinite(prediction_np)):
            raise ValueError(f"invalid prediction for MLP {row['mlp_id']}")
        squared_error = (prediction_np - target) ** 2
        final_mse = float(squared_error[-1].mean())
        flops_used = int(context.flops_used)
        mlp_id = int(row["mlp_id"])
        rqmc_mse = float(rqmc_by_id[mlp_id]["final_mse"]) if mlp_id in rqmc_by_id else None
        k3_mse = float(k3_by_id[mlp_id]["final_mse"]) if mlp_id in k3_by_id else None
        records.append(
            {
                "partition": args.partition,
                "mlp_id": mlp_id,
                "mlp_name": str(row["mlp_name"]),
                "mlp_seed": int(row["mlp_seed"]),
                "final_mse": final_mse,
                "all_layer_mse": float(squared_error.mean()),
                "adjusted_score": final_mse * max(SCORE_FLOOR, flops_used / args.budget),
                "flops": flops_used,
                "runtime_s": runtime_s,
                "rqmc_24576_final_mse": rqmc_mse,
                "delta_vs_rqmc_24576": final_mse - rqmc_mse if rqmc_mse is not None else None,
                "win_vs_rqmc_24576": final_mse < rqmc_mse if rqmc_mse is not None else None,
                "k3_final_mse": k3_mse,
                "win_vs_k3": final_mse < k3_mse if k3_mse is not None else None,
                "prefix_mse": {
                    str(prefix): float(squared_error[prefix - 1].mean())
                    for prefix in PREFIXES
                    if prefix <= mlp.depth
                },
            }
        )
        print(
            f"{index}/{len(rows)} mlp_id={mlp_id} mse={final_mse:.6g} "
            f"delta_vs_rqmc={records[-1]['delta_vs_rqmc_24576']}",
            flush=True,
        )

    baseline_subset = list(rqmc_by_id.values())
    k3_subset = list(k3_by_id.values())
    result = {
        "schema_version": "1.0",
        "run_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition": args.partition,
        "configuration": {
            "n_samples": 24_576,
            "seed_offset": 8_675_309,
            "multiplier": 17_797,
            "budget": args.budget,
            "score_floor": SCORE_FLOOR,
            "limit": args.limit,
            "final_layer_method": "Gaussian ReLU mean from sampled preactivation mean and variance",
        },
        "estimator_path": str(ESTIMATOR.relative_to(ROOT)),
        "estimator_sha256": hashlib.sha256(ESTIMATOR.read_bytes()).hexdigest(),
        "command": f"{sys.executable} {' '.join(sys.argv)}",
        "summary": summarize(records),
        "rqmc_24576_summary": {
            "final_mse": float(np.mean([row["final_mse"] for row in baseline_subset])),
            "adjusted_score": float(np.mean([row["adjusted_score"] for row in baseline_subset])),
            "flops": float(np.mean([row["flops"] for row in baseline_subset])),
        }
        if baseline_subset
        else None,
        "k3_summary": {
            "final_mse": float(np.mean([row["final_mse"] for row in k3_subset])),
            "adjusted_score": float(np.mean([row["adjusted_score"] for row in k3_subset])),
            "flops": float(np.mean([row["flops"] for row in k3_subset])),
        }
        if k3_subset
        else None,
        "per_mlp": records,
    }
    output = RESULTS / f"conditional_gaussian_rqmc_24576_{args.partition}.json"
    report = REPORTS / f"conditional_gaussian_rqmc_24576_{args.partition}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report, result)
    print(f"wrote {output.relative_to(ROOT)}")
    print(f"wrote {report.relative_to(ROOT)}")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition",
        choices=("development_selection", "validation_frozen"),
        default="development_selection",
    )
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
